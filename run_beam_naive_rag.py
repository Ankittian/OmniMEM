#!/usr/bin/env python
"""run_beam_naive_rag.py — CLI entry point for the Naive RAG BEAM baseline.

Usage:
    python run_beam_naive_rag.py --size 100K
    python run_beam_naive_rag.py --size 100K --start 0 --end 5 --verbose
"""
import argparse
from src.benchmarks.beam_naive_rag import run, SIZES

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BEAM benchmark with Naive RAG baseline.")
    parser.add_argument("--size", required=True, choices=SIZES,
                        help="Chat corpus size to evaluate (100K / 500K / 1M / 10M).")
    parser.add_argument("--start", type=int, default=0,
                        help="Index of first chat to process (default: 0).")
    parser.add_argument("--end", type=int, default=None,
                        help="Index of last chat to process, exclusive (default: all).")
    parser.add_argument("--verbose", action="store_true",
                        help="Print a panel for each question/answer.")
    args = parser.parse_args()

    run(size=args.size, start=args.start, end=args.end, verbose=args.verbose)

    print(f"\nDone. Evaluate with:\n"
          f"  python evaluate_beam_gemini.py --size {args.size} --results-dir results/beam-naive/{args.size}\n")
