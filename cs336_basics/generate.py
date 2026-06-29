from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .tokenizer import Tokenizer
from .transformer import transformer_lm


def top_p_filtering(probs: torch.Tensor, top_p: float | None) -> torch.Tensor:
	"""Keep the smallest high-probability token set with cumulative mass >= top_p.

	Args:
		probs: A 1D probability distribution over vocabulary items.
		top_p: Nucleus sampling threshold. If None or >= 1, no filtering is applied.

	Returns:
		A renormalized 1D probability distribution after top-p filtering.
	"""
	if probs.ndim != 1:
		raise ValueError(f"probs must be 1D, got shape {tuple(probs.shape)}")
	if top_p is None or top_p >= 1.0:
		return probs
	if top_p <= 0.0:
		raise ValueError(f"top_p must be in (0, 1] or None, got {top_p}")

	sorted_probs, sorted_indices = torch.sort(probs, descending=True)
	cumulative_probs = torch.cumsum(sorted_probs, dim=0)

	keep_sorted = cumulative_probs <= top_p
	keep_sorted[0] = True
	first_above = torch.nonzero(cumulative_probs > top_p, as_tuple=False)
	if first_above.numel() > 0:
		keep_sorted[first_above[0].item()] = True

	filtered = torch.zeros_like(probs)
	kept_indices = sorted_indices[keep_sorted]
	filtered[kept_indices] = probs[kept_indices]

	total = filtered.sum()
	if total <= 0:
		raise RuntimeError("top-p filtering removed all probability mass")
	return filtered / total


def sample_next_token(
	logits: torch.Tensor,
	temperature: float = 1.0,
	top_p: float | None = None,
	generator: torch.Generator | None = None,
) -> torch.Tensor:
	"""Sample one token ID from next-token logits.

	Args:
		logits: A 1D tensor of unnormalized next-token logits.
		temperature: Sampling temperature. Use 0 for greedy argmax decoding.
		top_p: Optional nucleus sampling threshold.
		generator: Optional torch random generator for reproducible sampling.

	Returns:
		A scalar tensor containing the sampled token ID.
	"""
	if logits.ndim != 1:
		raise ValueError(f"logits must be 1D, got shape {tuple(logits.shape)}")
	if temperature < 0.0:
		raise ValueError(f"temperature must be non-negative, got {temperature}")

	if temperature == 0.0:
		return torch.argmax(logits)

	probs = torch.softmax(logits / temperature, dim=-1)
	probs = top_p_filtering(probs, top_p)
	return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(0)


@torch.no_grad()
def generate_ids(
	model: torch.nn.Module,
	prompt_ids: list[int] | torch.Tensor,
	max_new_tokens: int,
	context_length: int,
	eos_token_id: int | None = None,
	temperature: float = 1.0,
	top_p: float | None = None,
	device: str | torch.device | None = None,
	generator: torch.Generator | None = None,
) -> torch.Tensor:
	"""Generate token IDs autoregressively from a prompt.

	Args:
		model: Language model returning logits of shape (batch, sequence, vocab).
		prompt_ids: Prompt token IDs as a list, a 1D tensor, or a batch-size-1 2D tensor.
		max_new_tokens: Maximum number of new tokens to append.
		context_length: Maximum context length to feed into the model.
		eos_token_id: Optional end-of-text token ID. Generation stops when sampled.
		temperature: Sampling temperature. Use 0 for greedy argmax decoding.
		top_p: Optional nucleus sampling threshold.
		device: Device for generation. If None, uses the model's parameter device.
		generator: Optional torch random generator for reproducible sampling.

	Returns:
		A tensor of shape (1, prompt_length + generated_length).
	"""
	if max_new_tokens < 0:
		raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
	if context_length <= 0:
		raise ValueError(f"context_length must be positive, got {context_length}")

	if device is None:
		try:
			device = next(model.parameters()).device
		except StopIteration:
			device = torch.device("cpu")
	device = torch.device(device)

	if isinstance(prompt_ids, torch.Tensor):
		generated = prompt_ids.to(device=device, dtype=torch.long)
	else:
		generated = torch.tensor(prompt_ids, dtype=torch.long, device=device)

	if generated.ndim == 1:
		generated = generated.unsqueeze(0)
	if generated.ndim != 2 or generated.shape[0] != 1:
		raise ValueError(f"prompt_ids must be 1D or batch-size-1 2D, got shape {tuple(generated.shape)}")
	if generated.shape[1] == 0:
		raise ValueError("prompt_ids must contain at least one token")

	was_training = model.training
	model.eval()
	try:
		for _ in range(max_new_tokens):
			model_input = generated[:, -context_length:]
			logits = model(model_input)
			next_logits = logits[0, -1, :]
			next_id = sample_next_token(
				next_logits,
				temperature=temperature,
				top_p=top_p,
				generator=generator,
			).to(device=device, dtype=torch.long)

			generated = torch.cat([generated, next_id.view(1, 1)], dim=1)
			if eos_token_id is not None and next_id.item() == eos_token_id:
				break
	finally:
		model.train(was_training)

	return generated


def _get_eos_token_id(tokenizer: Tokenizer, eos_token: str | None) -> int | None:
	if eos_token is None:
		return None
	token_bytes = eos_token.encode("utf-8")
	if token_bytes in tokenizer.vocab_token_id:
		return tokenizer.vocab_token_id[token_bytes]
	if eos_token in tokenizer.vocab_token_id:
		return tokenizer.vocab_token_id[eos_token]
	return None


def generate_text(
	model: torch.nn.Module,
	tokenizer: Tokenizer,
	prompt: str,
	max_new_tokens: int = 256,
	context_length: int = 256,
	temperature: float = 1.0,
	top_p: float | None = 0.9,
	eos_token: str | None = "<|endoftext|>",
	device: str | torch.device | None = None,
	seed: int | None = None,
) -> str:
	"""Generate text from a prompt using a tokenizer and language model."""
	prompt_ids = tokenizer.encode(prompt)
	if not prompt_ids:
		raise ValueError("prompt must encode to at least one token")

	if device is None:
		try:
			device = next(model.parameters()).device
		except StopIteration:
			device = torch.device("cpu")
	device = torch.device(device)

	generator = None
	if seed is not None:
		generator = torch.Generator(device=device)
		generator.manual_seed(seed)

	output_ids = generate_ids(
		model=model,
		prompt_ids=prompt_ids,
		max_new_tokens=max_new_tokens,
		context_length=context_length,
		eos_token_id=_get_eos_token_id(tokenizer, eos_token),
		temperature=temperature,
		top_p=top_p,
		device=device,
		generator=generator,
	)
	return tokenizer.decode(output_ids[0].tolist())


def load_model_from_checkpoint(
	checkpoint_path: str | Path,
	vocab_size: int,
	context_length: int,
	d_model: int,
	num_layers: int,
	num_heads: int,
	d_ff: int,
	theta: float,
	device: str | torch.device = "cpu",
) -> transformer_lm:
	"""Build a transformer_lm and load weights saved by transformer_train.py."""
	device = torch.device(device)
	model = transformer_lm(
		vocab_size=vocab_size,
		max_seq_len=context_length,
		d_model=d_model,
		num_layers=num_layers,
		num_heads=num_heads,
		d_ff=d_ff,
		theta=theta,
		device=device,
	)
	checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location=device, weights_only=False)
	model.load_state_dict(checkpoint["model"])
	model.eval()
	return model
