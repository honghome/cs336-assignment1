import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from transformer import (
    transformer_lm,
    adamw_cls,
    cross_entropy,
    lr_cosine_schedule,
    run_gradient_clipping,
    get_batch,
)


class MetricsLogger:
    def __init__(self, backend: str = "none") -> None:
        self.backend = backend
        self.mlflow = None
        self.azureml_run = None
        self.enabled = False
        self._warned = False

        if backend == "auto":
            backend = "azureml" if os.environ.get("AZUREML_RUN_ID") else "none"

        if backend == "azureml":
            try:
                from azureml.core import Run

                self.azureml_run = Run.get_context()
                self.enabled = True
                print("Metrics backend: azureml")
            except Exception as exc:
                print(f"Metrics backend disabled: could not initialize AzureML run context ({exc})")
        elif backend == "mlflow":
            try:
                import mlflow

                self.mlflow = mlflow
                self.enabled = True
                print("Metrics backend: mlflow")
            except Exception as exc:
                print(f"Metrics backend disabled: could not import mlflow ({exc})")
        else:
            print("Metrics backend: none")

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        if not self.enabled:
            return
        try:
            if self.azureml_run is not None:
                self.azureml_run.log(name, float(value), step=step)
            elif self.mlflow is not None:
                self.mlflow.log_metric(name, float(value), step=step)
        except Exception as exc:
            self._warn_once(f"Metrics logging warning: could not log {name} ({exc})")

    def log_params(self, params: dict) -> None:
        if not self.enabled:
            return
        try:
            if self.azureml_run is not None:
                for name, value in params.items():
                    self.azureml_run.log(f"param/{name}", value)
            elif self.mlflow is not None:
                self.mlflow.log_params(params)
        except Exception as exc:
            self._warn_once(f"Metrics logging warning: could not log params ({exc})")

    def _warn_once(self, message: str) -> None:
        if not self._warned:
            print(message)
            self._warned = True

# Step 0: Script skeleton and config
# Goal: parse arguments so script can be controlled from command line.
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--train_data", type=str, required=True)
    p.add_argument("--valid_data", type=str, required=True)

    # Model
    p.add_argument("--vocab_size", type=int, required=True)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=8)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=1365)
    p.add_argument("--theta", type=float, default=10000.0)

    # Train
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_iters", type=int, default=20000)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--warmup_iters", type=int, default=1000)
    p.add_argument("--cosine_cycle_iters", type=int, default=20000)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Logging and checkpointing
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--ckpt_every", type=int, default=1000)
    p.add_argument("--ckpt_path", type=str, default="checkpoint.pt")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--metrics_backend",
        choices=["none", "auto", "azureml", "mlflow"],
        default="none",
        help="Optional metric backend. Use 'azureml' in Azure ML to show metrics in the portal.",
    )

    # Device
    p.add_argument("--device", type=str, default="cuda:0")

    return p

# Step 1: Memory-mapped data loading
# Goal: large datasets without loading full arrays into RAM.
def load_memmap(path: str) -> np.ndarray:
    return np.memmap(path, dtype=np.int32, mode="r")

# Step 2: build_model_and_optimizer
def build_model_and_optimizer(args, device):
    model = transformer_lm(
        vocab_size=args.vocab_size,
        max_seq_len=args.context_length,
        d_model = args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
        device=device
    )
    optimizer = adamw_cls(
        params=model.parameters(),
        lr=args.max_lr,
        weight_decay=args.weight_decay
    )
    return model, optimizer

# Step 3: train_step — forward, loss, backward, clip
def train_step(model, optimizerm, x, y, args):
    model.train()
    logits = model(x)
    B, T, V = logits.shape
    loss = cross_entropy(logits.view(B * T, V), y.view(B * T))
    loss.backward()
    run_gradient_clipping(model.parameters(), args.grad_clip)
    return loss.item()

# Step 4: evaluate — validation loss + perplexity
@torch.no_grad()
def evaluate(model, dataset, args, device):
    model.eval()
    total_loss = 0.0
    for _ in range(args.eval_batches):
        x, y = get_batch(dataset, args.batch_size, args.context_length, device)
        logits = model(x)
        B, T, V = logits.shape
        total_loss += cross_entropy(logits.view(B * T, V), y.view(B * T)).item()
    avg_loss = total_loss / args.eval_batches
    return avg_loss, math.exp(avg_loss)

# Step 5: save and load checkpoint
def save_checkpoint(model, optimizer, iteration, path):
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration},
        path,
    )

def load_checkpoint(path, model, optimizer):
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["iteration"]

# main function
def main():
    parser = build_parser()
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    train_data = load_memmap(args.train_data)
    valid_data = load_memmap(args.valid_data)
    print(f"Train tokens: {len(train_data):,} | Valid tokens: {len(valid_data):,}")

    metrics = MetricsLogger(args.metrics_backend)
    metrics.log_params(
        {
            "vocab_size": args.vocab_size,
            "context_length": args.context_length,
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "d_ff": args.d_ff,
            "batch_size": args.batch_size,
            "max_iters": args.max_iters,
            "max_lr": args.max_lr,
            "min_lr": args.min_lr,
            "warmup_iters": args.warmup_iters,
            "cosine_cycle_iters": args.cosine_cycle_iters,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "train_tokens": len(train_data),
            "valid_tokens": len(valid_data),
        }
    )

    model, optimizer = build_model_and_optimizer(args, device)
    model_parameters = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {model_parameters:,}")
    metrics.log_params({"model_parameters": model_parameters})

    start_iter = 0
    ckpt_path = Path(args.ckpt_path)
    if args.resume and ckpt_path.exists():
        start_iter = load_checkpoint(str(ckpt_path), model, optimizer)
        print(f"Resumed from iter {start_iter}")

    t0 = time.time()
    for it in range(start_iter, args.max_iters):
        # Set learning rate
        lr = lr_cosine_schedule(it, args.max_lr, args.min_lr, args.warmup_iters, args.cosine_cycle_iters)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Train
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        optimizer.zero_grad()
        loss = train_step(model, optimizer, x, y, args)
        optimizer.step()

        if (it + 1) % args.log_every == 0:
            elapsed_seconds = time.time() - t0
            print(f"iter {it+1:6d} | loss {loss:.4f} | lr {lr:.2e} | {elapsed_seconds:.1f}s")
            metrics.log_metric("train_loss", loss, step=it + 1)
            metrics.log_metric("learning_rate", lr, step=it + 1)
            metrics.log_metric("elapsed_seconds", elapsed_seconds, step=it + 1)

        if (it + 1) % args.eval_every == 0:
            val_loss, val_ppl = evaluate(model, valid_data, args, device)
            print(f"  [eval] val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}")
            metrics.log_metric("val_loss", val_loss, step=it + 1)
            metrics.log_metric("val_ppl", val_ppl, step=it + 1)

        if (it + 1) % args.ckpt_every == 0:
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, optimizer, it + 1, str(ckpt_path))
            print(f"  [ckpt] saved → {ckpt_path}")
            metrics.log_metric("checkpoint_iteration", it + 1, step=it + 1)

    val_loss, val_ppl = evaluate(model, valid_data, args, device)
    print(f"Done. Final val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}")
    metrics.log_metric("final_val_loss", val_loss, step=args.max_iters)
    metrics.log_metric("final_val_ppl", val_ppl, step=args.max_iters)
    metrics.log_metric("final_iteration", args.max_iters, step=args.max_iters)
    save_checkpoint(model, optimizer, args.max_iters, str(ckpt_path))


if __name__ == "__main__":
    main()