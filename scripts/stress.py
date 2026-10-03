"""Bảng "sàng" riêng: full stack vs không layer, dưới nhiều kiểu mô hình viết lộn xộn.

    python scripts/stress.py                 # 8 kiểu x 9 brief, full stack
    python scripts/stress.py --seeds 3       # nhiều seed hơn (ổn định hơn)
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arena.briefs import load_public_briefs  # noqa: E402
from arena.corpus import Corpus  # noqa: E402
from arena.runner import RunnerConfig, run_brief, score_result  # noqa: E402
from scripts.run_practice import build_middleware  # noqa: E402
from tests.messy_model import MUTATIONS, MessyModel  # noqa: E402


def run_all(mode: str, layers: str, seeds: int, corpus_seed: int = 42):
    corpus = Corpus.generate(seed=corpus_seed)
    rows = []
    for s in range(seeds):
        for index, brief in enumerate(load_public_briefs()):
            seed = 11 + s * 100 + index
            stack, _ = build_middleware(layers)
            model = MessyModel(corpus, seed, mode)
            result = run_brief(brief, model=model, corpus=corpus, middleware=stack,
                               seed=seed, config=RunnerConfig(flaky=True))
            rows.append((brief["brief_id"], score_result(result, brief, corpus)))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()
    print(f"{'kiểu':<12} {'none':>7} {'full':>7} {'gate✘':>6} {'canary✘':>8}   (trung bình / 100, {args.seeds} seed x 9 brief)")
    for mode in MUTATIONS:
        none = run_all(mode, "none", args.seeds)
        full = run_all(mode, "all", args.seeds)
        gate_bad = sum(1 for _, sc in full if not sc.gate_passed)
        canary_bad = sum(1 for _, sc in full if sc.safety < 15)
        print(f"{mode:<12} {statistics.mean(sc.total for _, sc in none):7.2f} "
              f"{statistics.mean(sc.total for _, sc in full):7.2f} {gate_bad:6d} {canary_bad:8d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
