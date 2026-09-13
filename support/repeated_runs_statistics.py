"""Summarize repeated Double DQN training runs.

Only the ``Mean Loss`` and ``Mean Reward`` fields are read from the raw
training logs. Standard deviations are calculated across independent runs,
never from the legacy ``Std Loss`` or ``Std Reward`` fields.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev


FILE_PATTERN = re.compile(
    r"train(?P<run>\d+)_nn_(?P<radar>radar_)?double-"
    r"(?P<role>predator|prey)\.txt$"
)
LINE_PATTERN = re.compile(
    r"Episode: (?P<episode>\d+).*?"
    r"Mean Loss: (?P<mean_loss>[-+\d.eE]+).*?"
    r"Mean Reward: (?P<mean_reward>[-+\d.eE]+)"
)


def parse_arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Calculate mean and sample SD across repeated Double DQN runs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=repository / "train" / "double_repeated_runs",
        help="Directory containing the repeated-run training logs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository / "support" / "double_repeated_runs_summary.csv",
        help="Destination CSV file.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=50,
        help="Number of final episodes averaged within each run (default: 50).",
    )
    return parser.parse_args()


def read_run(path: Path, window: int) -> tuple[float, float, int]:
    records: list[tuple[int, float, float]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = LINE_PATTERN.search(line)
        if not match:
            raise ValueError(f"Malformed training record: {path.name}:{line_number}")
        records.append(
            (
                int(match["episode"]),
                float(match["mean_loss"]),
                float(match["mean_reward"]),
            )
        )

    if len(records) < window:
        raise ValueError(f"{path.name} has only {len(records)} episodes; window={window}")

    final_records = records[-window:]
    return (
        fmean(record[1] for record in final_records),
        fmean(record[2] for record in final_records),
        len(final_records),
    )


def main() -> None:
    args = parse_arguments()
    if args.window <= 0:
        raise ValueError("--window must be greater than zero")

    groups: dict[tuple[str, str], list[tuple[int, float, float, int]]] = defaultdict(list)
    unexpected_files: list[str] = []

    for path in sorted(args.input.glob("*.txt")):
        match = FILE_PATTERN.fullmatch(path.name)
        if not match:
            unexpected_files.append(path.name)
            continue
        architecture = "RADAR" if match["radar"] else "CNN-MLP"
        role = match["role"]
        mean_loss, mean_reward, episodes = read_run(path, args.window)
        groups[(architecture, role)].append(
            (int(match["run"]), mean_loss, mean_reward, episodes)
        )

    if unexpected_files:
        raise ValueError(f"Unexpected training filenames: {unexpected_files}")
    if not groups:
        raise ValueError(f"No repeated-run training logs found in {args.input}")

    rows = []
    for (architecture, role), runs in sorted(groups.items()):
        runs.sort()
        run_ids = [run[0] for run in runs]
        if run_ids != list(range(1, 11)):
            raise ValueError(
                f"{architecture}/{role}: expected runs 1..10, found {run_ids}"
            )
        losses = [run[1] for run in runs]
        rewards = [run[2] for run in runs]
        rows.append(
            {
                "architecture": architecture,
                "role": role,
                "runs": len(runs),
                "final_episode_window": args.window,
                "mean_loss_across_runs": fmean(losses),
                "sd_loss_across_runs": stdev(losses),
                "mean_reward_across_runs": fmean(rewards),
                "sd_reward_across_runs": stdev(rewards),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.6f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )

    print(f"Validated {sum(len(runs) for runs in groups.values())} training logs.")
    print(f"Summary written to: {args.output}")
    for row in rows:
        print(
            f"{row['architecture']:7} {row['role']:8} | "
            f"loss={row['mean_loss_across_runs']:.4f} ± {row['sd_loss_across_runs']:.4f} | "
            f"reward={row['mean_reward_across_runs']:.4f} ± {row['sd_reward_across_runs']:.4f}"
        )


if __name__ == "__main__":
    main()
