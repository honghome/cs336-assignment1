from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cs336_basics.generate import generate_text, load_model_from_checkpoint
from cs336_basics.tokenizer import Tokenizer


DEFAULT_SETTINGS = [
    ("greedy", 0.0, None, 11),
    ("t0.8_p0.9", 0.8, 0.9, 12),
    ("t1.0_p0.9", 1.0, 0.9, 13),
    ("t1.1_p0.95", 1.1, 0.95, 14),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate TinyStories samples from a trained checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to transformer_train.py checkpoint.pt")
    parser.add_argument("--vocab", default="training/TinyStoriesV2-GPT4-train_bpe_vocab.pkl")
    parser.add_argument("--merges", default="training/TinyStoriesV2-GPT4-train_bpe_merges.pkl")
    parser.add_argument("--output", default="docs/generation_samples.md")
    parser.add_argument("--prompt", default="Once upon a time, there was a little girl named Lily")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--theta", type=float, default=10000.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)

    tokenizer = Tokenizer.from_file(args.vocab, args.merges, special_tokens=["<|endoftext|>"])
    model = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
        device=device,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Generation Samples",
        "",
        f"Checkpoint: `{args.checkpoint}`",
        "",
        f"Prompt: `{args.prompt}`",
        "",
        f"Max new tokens: `{args.max_new_tokens}`",
        "",
    ]

    for label, temperature, top_p, seed in DEFAULT_SETTINGS:
        text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            context_length=args.context_length,
            temperature=temperature,
            top_p=top_p,
            device=device,
            seed=seed,
        )
        top_p_label = "none" if top_p is None else str(top_p)
        lines.extend(
            [
                f"## {label}",
                "",
                f"Temperature: `{temperature}`; top-p: `{top_p_label}`; seed: `{seed}`",
                "",
                "```text",
                text.strip(),
                "```",
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()