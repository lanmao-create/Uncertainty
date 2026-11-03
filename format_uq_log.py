"""Turn an existing UQ JSONL history into readable reports without training."""
import argparse
import json
from pathlib import Path

import uq_config as cfg
from uq_reporting import configure_console, confusion_metrics, epoch_report, setup_report


def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=cfg.MODEL, choices=cfg.BASELINE_MODELS)
    parser.add_argument("--run-name", default=cfg.RUN_NAME)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    directory = args.run_dir or cfg.OUTPUT / args.model / "uq" / args.run_name
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    content = (directory / "history.jsonl").read_text(encoding="utf-8")
    rows = []
    for line in content.splitlines(keepends=True):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if line.endswith("\n"):
                raise
            # A live training process may still be writing the final line.
    if not rows:
        parser.error("No completed epochs in history.jsonl")
    blocks = [setup_report(config)]
    best = -1.0
    for record in rows:
        score = (confusion_metrics(record["confusion"])["miou_5"]
                 if "confusion" in record else record.get("miou_5", record.get("miou", -1.0)))
        best = max(best, score if score is not None else -1.0)
        blocks.append(epoch_report(record, config, best))
    output = directory / "readable_history.txt"
    output.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(blocks[0])
    print(blocks[-1])
    print(f"\nConverted {len(rows)} epochs. Readable log: {output}")


if __name__ == "__main__":
    main()
