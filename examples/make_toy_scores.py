"""Write examples/toy_scores.jsonl: a small synthetic table for trying the commands.

Four systems and six items. The quality of an output is a system effect plus an item
effect plus noise, and two annotators rate it from 1 to 5. Two made-up metrics see one
half of that each:

- ``metric_a`` follows the system effect and is blind to the item;
- ``metric_b`` follows the item effect and is blind to the system.

Nothing here is real data. Run from the repository root:

    python examples/make_toy_scores.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SYSTEM_EFFECT = {"sys_a": -1.0, "sys_b": -0.3, "sys_c": 0.4, "sys_d": 1.0}
ITEMS = [f"doc_{number}" for number in range(1, 7)]
SEED = 2026


def rating(quality: float, rng: np.random.Generator) -> int:
    return int(np.clip(np.rint(3.0 + quality + rng.normal(0.0, 0.4)), 1, 5))


def main() -> None:
    rng = np.random.default_rng(SEED)
    item_effect = dict(zip(ITEMS, rng.normal(0.0, 1.0, size=len(ITEMS)), strict=True))
    lines = []
    for item in ITEMS:
        for system, effect in SYSTEM_EFFECT.items():
            quality = effect + item_effect[item] + rng.normal(0.0, 0.5)
            record = {
                "item": item,
                "system": system,
                "human": {"ann_1": rating(quality, rng), "ann_2": rating(quality, rng)},
                "metrics": {
                    "metric_a": round(0.5 + 0.15 * effect + rng.normal(0.0, 0.05), 3),
                    "metric_b": round(0.5 + 0.15 * item_effect[item] + rng.normal(0.0, 0.05), 3),
                },
            }
            lines.append(json.dumps(record))
    path = Path(__file__).with_name("toy_scores.jsonl")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} records to {path.name}")


if __name__ == "__main__":
    main()
