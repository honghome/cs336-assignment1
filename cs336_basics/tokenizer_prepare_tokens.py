#!/usr/bin/env python
"""
Prepare token datasets for training.

Tokenizes large text files and saves token IDs to disk using lazy loading.
Supports training and validation splits with progress tracking.

Usage:
    python tokenizer_prepare_tokens.py \
        --train data/TinyStoriesV2-GPT4-train.txt \
        --valid data/TinyStoriesV2-GPT4-valid.txt \
        --vocab training/TinyStoriesV2-GPT4-train_bpe_vocab.pkl \
        --merges training/TinyStoriesV2-GPT4-train_bpe_merges.pkl \
        --output-train data/train_tokens.bin \
        --output-valid data/valid_tokens.bin
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Generator
import numpy as np

from cs336_basics.tokenizer import Tokenizer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def iter_lines(path: str) -> Generator[str, None, None]:
    """
    Lazily yield lines from a file.
    Handles all platform newlines automatically (\\r\\n, \\n, \\r).
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            yield line


def count_file_lines(path: str) -> int:
    """Count total lines in file for progress tracking."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)


def tokenize_and_save(
    tokenizer: Tokenizer,
    input_path: str,
    output_path: str,
    dtype=np.int32,
    buffer_size: int = 1_000_000,
    show_progress: bool = True,
) -> int:
    """
    Tokenize a large text file and save token IDs to disk.

    Args:
        tokenizer: Tokenizer instance (loaded from vocab/merges)
        input_path: Path to input text file
        output_path: Path to output binary file
        dtype: NumPy dtype for token IDs (default: int32)
        buffer_size: How many tokens to buffer before writing (default: 1M)
        show_progress: Whether to show progress bar (default: True)

    Returns:
        Total number of tokens written
    """
    input_path = str(input_path)
    output_path = str(output_path)

    if not Path(input_path).exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Count total lines for progress
    if show_progress:
        total_lines = count_file_lines(input_path)
        logger.info(f"Input file has {total_lines:,} lines")

    # Get lazy token stream
    token_stream = tokenizer.encode_iterable(iter_lines(input_path))

    # Buffer and write to disk
    buffer = []
    total_tokens = 0
    line_count = 0

    logger.info(f"Tokenizing {input_path} → {output_path}")

    with open(output_path, "wb") as out_f:
        for token_id in token_stream:
            buffer.append(token_id)
            total_tokens += 1

            # Flush buffer when it gets large
            if len(buffer) >= buffer_size:
                arr = np.array(buffer, dtype=dtype)
                arr.tofile(out_f)
                buffer.clear()

                # Progress update
                if show_progress and total_tokens % (buffer_size * 10) == 0:
                    logger.info(f"  Processed {total_tokens:,} tokens...")

        # Flush remaining tokens
        if buffer:
            arr = np.array(buffer, dtype=dtype)
            arr.tofile(out_f)

    logger.info(
        f"✓ Saved {total_tokens:,} tokens to {output_path} "
        f"({total_tokens * dtype(0).itemsize / 1e9:.2f} GB)"
    )

    return total_tokens


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Prepare token datasets for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                # Tokenize with custom vocab/merges
                python tokenizer_prepare_tokens.py \\
                    --train data/train.txt --valid data/valid.txt \\
                    --vocab path/to/vocab.json --merges path/to/merges.txt \\
                    --output-train data/train_tokens.bin --output-valid data/valid_tokens.bin

                # Tokenize with defaults (TinyStoriesV2 train/valid and BPE from training/)
                python tokenizer_prepare_tokens.py

                # Override only validation split
                python tokenizer_prepare_tokens.py --valid data/TinyStoriesV2-GPT4-valid.txt
            """,
    )

    # Required arguments
    parser.add_argument(
        "--train",
        type=str,
        required=False,
        default="data/TinyStoriesV2-GPT4-train.txt",
        help="Path to training text file (default: data/TinyStoriesV2-GPT4-train.txt)",
    )
    parser.add_argument(
        "--valid",
        type=str,
        required=False,
        default="data/TinyStoriesV2-GPT4-valid.txt",
        help="Path to validation text file (default: data/TinyStoriesV2-GPT4-valid.txt)",
    )

    # Optional arguments
    parser.add_argument(
        "--vocab",
        type=str,
        default="training/TinyStoriesV2-GPT4-train_bpe_vocab.pkl",
        help="Path to BPE vocabulary file (default: training/TinyStoriesV2-GPT4-train_bpe_vocab.pkl)",
    )
    parser.add_argument(
        "--merges",
        type=str,
        default="training/TinyStoriesV2-GPT4-train_bpe_merges.pkl",
        help="Path to BPE merges file (default: training/TinyStoriesV2-GPT4-train_bpe_merges.pkl)",
    )
    parser.add_argument(
        "--output-train",
        type=str,
        default="training/TinyStoriesV2-GPT4-train_tokens.bin",
        help="Output path for training tokens (default: training/TinyStoriesV2-GPT4-train_tokens.bin)",
    )
    parser.add_argument(
        "--output-valid",
        type=str,
        default="training/TinyStoriesV2-GPT4-valid_tokens.bin",
        help="Output path for validation tokens (default: training/TinyStoriesV2-GPT4-valid_tokens.bin)",
    )
    parser.add_argument(
        "--special-tokens",
        type=str,
        nargs="+",
        default=["<|endoftext|>"],
        help="Special tokens to preserve (default: <|endoftext|>)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=1_000_000,
        help="Token buffer size before write (default: 1,000,000)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["int32", "int64", "uint32"],
        default="int32",
        help="NumPy dtype for token IDs (default: int32)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress messages",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging (debug level)",
    )

    args = parser.parse_args()

    # Defaults include both splits; this check remains for explicit empty overrides
    if not args.train and not args.valid:
        parser.error("At least one of --train or --valid must be provided")

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("=" * 70)
    logger.info("Tokenizer Prepare Tokens")
    logger.info("=" * 70)

    # Validate vocab/merges files exist
    for path, name in [
        (args.vocab, "Vocabulary"),
        (args.merges, "Merges"),
    ]:
        if not Path(path).exists():
            logger.error(f"{name} file not found: {path}")
            sys.exit(1)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.vocab} and {args.merges}")
    try:
        tokenizer = Tokenizer.from_file(
            vocab_filepath=args.vocab,
            merges_filepath=args.merges,
            special_tokens=args.special_tokens,
        )
        logger.info(f"✓ Tokenizer loaded (special tokens: {args.special_tokens})")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        sys.exit(1)

    # Determine dtype
    dtype_map = {
        "int32": np.int32,
        "int64": np.int64,
        "uint32": np.uint32,
    }
    dtype = dtype_map[args.dtype]

    # Create output directory if needed
    output_dir = Path(args.output_train).parent if args.train else Path(args.output_valid).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tokenize training split
    total_tokens_train = 0
    if args.train:
        try:
            total_tokens_train = tokenize_and_save(
                tokenizer,
                args.train,
                args.output_train,
                dtype=dtype,
                buffer_size=args.buffer_size,
                show_progress=not args.no_progress,
            )
        except Exception as e:
            logger.error(f"Failed to tokenize training split: {e}")
            sys.exit(1)

    # Tokenize validation split
    total_tokens_valid = 0
    if args.valid:
        try:
            total_tokens_valid = tokenize_and_save(
                tokenizer,
                args.valid,
                args.output_valid,
                dtype=dtype,
                buffer_size=args.buffer_size,
                show_progress=not args.no_progress,
            )
        except Exception as e:
            logger.error(f"Failed to tokenize validation split: {e}")
            sys.exit(1)

    # Summary
    logger.info("=" * 70)
    if args.train:
        logger.info(f"Training tokens:   {total_tokens_train:,}")
    if args.valid:
        logger.info(f"Validation tokens: {total_tokens_valid:,}")
    if args.train and args.valid:
        logger.info(f"Total tokens:      {total_tokens_train + total_tokens_valid:,}")
    logger.info("=" * 70)
    logger.info("✓ Tokenization complete!")


if __name__ == "__main__":
    main()
