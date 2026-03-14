from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_DATA_PATH = Path("exports") / "reversed_messages.jsonl"
DEFAULT_OUTPUT_DIR = Path("outputs") / "qwen35-9b-lora-user-sim"
DEFAULT_MODEL_PATH = "Qwen/Qwen3.5-9B"
DEFAULT_NUM_EPOCHS = 5
DEFAULT_LEARNING_RATE = 1.5e-4
DEFAULT_MICRO_BATCH_SIZE = 1
DEFAULT_MAX_SEQ_LEN = 32768
DEFAULT_VALIDATION_RATIO = 0.1
DEFAULT_SEED = 3407
DEFAULT_EVAL_STRATEGY = "epoch"
DEFAULT_LORA_R = 64
DEFAULT_LOAD_IN_4BIT = False
DEFAULT_SAMPLE_PACKING = False


@dataclass(frozen=True)
class DatasetSplit:
    train_path: Path
    validation_path: Path
    train_rows: list[Any]
    validation_rows: list[Any]

    @property
    def train_count(self) -> int:
        return len(self.train_rows)

    @property
    def validation_count(self) -> int:
        return len(self.validation_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a LoRA user-simulator model with training_hub."
    )
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--ckpt-output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--micro-batch-size", type=int, default=DEFAULT_MICRO_BATCH_SIZE
    )
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument(
        "--validation-ratio", type=float, default=DEFAULT_VALIDATION_RATIO
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_LOAD_IN_4BIT,
    )
    parser.add_argument(
        "--sample-packing",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SAMPLE_PACKING,
    )
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument(
        "--eval-strategy",
        choices=("epoch", "steps"),
        default=DEFAULT_EVAL_STRATEGY,
    )
    parser.add_argument("--eval-steps", type=int)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_train_kwargs(
    args: argparse.Namespace, train_data_path: Path
) -> dict[str, Any]:
    return {
        "model_path": args.model_path,
        "data_path": str(train_data_path),
        "ckpt_output_dir": args.ckpt_output_dir,
        "dataset_type": "chat_template",
        "field_messages": "messages",
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "micro_batch_size": args.micro_batch_size,
        "max_seq_len": args.max_seq_len,
        "seed": args.seed,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "load_in_4bit": args.load_in_4bit,
        "bf16": True,
        "sample_packing": args.sample_packing,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_run_name": args.wandb_run_name,
    }


def validate_args(args: argparse.Namespace) -> None:
    if not Path(args.data_path).exists():
        raise SystemExit(f"Training data not found: {args.data_path}")
    if not 0.0 < args.validation_ratio < 1.0:
        raise SystemExit("--validation-ratio must be between 0 and 1")
    if args.eval_strategy == "steps" and args.eval_steps is None:
        raise SystemExit("--eval-steps is required when --eval-strategy=steps")
    if args.eval_steps is not None and args.eval_steps < 1:
        raise SystemExit("--eval-steps must be at least 1")


def build_split_path(data_path: Path, split_name: str) -> Path:
    suffix = "".join(data_path.suffixes)
    if suffix:
        base_name = data_path.name[: -len(suffix)]
    else:
        base_name = data_path.name
    return data_path.with_name(f"{base_name}.{split_name}{suffix}")


def load_dataset_rows(data_path: Path) -> list[Any]:
    if data_path.suffix == ".jsonl":
        rows = []
        with data_path.open() as file_handle:
            for raw_line in file_handle:
                line = raw_line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    if data_path.suffix == ".json":
        with data_path.open() as file_handle:
            rows = json.load(file_handle)
        if not isinstance(rows, list):
            raise SystemExit("JSON training data must be a top-level list")
        return rows

    raise SystemExit("Training data must be .jsonl or .json")


def build_dataset_split(
    rows: list[Any],
    data_path: Path,
    validation_ratio: float,
    seed: int,
) -> DatasetSplit:
    if len(rows) < 2:
        raise SystemExit(
            "Need at least 2 dataset rows to create a train/validation split"
        )

    shuffled_rows = list(rows)
    random.Random(seed).shuffle(shuffled_rows)

    validation_count = max(1, int(len(shuffled_rows) * validation_ratio))
    validation_count = min(validation_count, len(shuffled_rows) - 1)

    validation_rows = shuffled_rows[:validation_count]
    train_rows = shuffled_rows[validation_count:]

    return DatasetSplit(
        train_path=build_split_path(data_path, "train"),
        validation_path=build_split_path(data_path, "validation"),
        train_rows=train_rows,
        validation_rows=validation_rows,
    )


def write_dataset_rows(data_path: Path, rows: list[Any]) -> None:
    data_path.parent.mkdir(parents=True, exist_ok=True)

    if data_path.suffix == ".jsonl":
        with data_path.open("w") as file_handle:
            for row in rows:
                file_handle.write(json.dumps(row))
                file_handle.write("\n")
        return

    if data_path.suffix == ".json":
        with data_path.open("w") as file_handle:
            json.dump(rows, file_handle)
        return

    raise SystemExit("Split data must be written as .jsonl or .json")


def write_dataset_split(dataset_split: DatasetSplit) -> None:
    write_dataset_rows(dataset_split.train_path, dataset_split.train_rows)
    write_dataset_rows(dataset_split.validation_path, dataset_split.validation_rows)


def build_dry_run_output(
    train_kwargs: dict[str, Any],
    dataset_split: DatasetSplit,
    args: argparse.Namespace,
) -> dict[str, Any]:
    dry_run_output = dict(train_kwargs)
    dry_run_output["validation_data_path"] = str(dataset_split.validation_path)
    dry_run_output["train_examples"] = dataset_split.train_count
    dry_run_output["validation_examples"] = dataset_split.validation_count
    dry_run_output["validation_ratio"] = args.validation_ratio
    dry_run_output["eval_strategy"] = args.eval_strategy
    return dry_run_output


def run_training(
    args: argparse.Namespace,
    train_kwargs: dict[str, Any],
    dataset_split: DatasetSplit,
) -> None:
    try:
        import unsloth  # noqa: F401  # pyright: ignore[reportMissingImports]
        from training_hub.algorithms.lora import (
            JSONLLoggingCallback,
            UnslothLoRABackend,
        )
        from trl import SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "training_hub LoRA deps are not installed. Install them first, then rerun this script."
        ) from exc

    backend = UnslothLoRABackend()
    model, tokenizer = backend._load_unsloth_model(train_kwargs)
    model = backend._apply_lora_config(model, train_kwargs)
    train_dataset = backend._prepare_dataset(train_kwargs, tokenizer)

    validation_kwargs = dict(train_kwargs)
    validation_kwargs["data_path"] = str(dataset_split.validation_path)
    validation_dataset = backend._prepare_dataset(validation_kwargs, tokenizer)

    training_args = backend._build_training_args(train_kwargs)
    training_args.max_length = args.max_seq_len
    training_args.packing = args.sample_packing
    training_args.eval_packing = args.sample_packing
    training_args.eval_strategy = args.eval_strategy
    training_args.do_eval = True
    training_args.per_device_eval_batch_size = args.micro_batch_size
    if args.eval_strategy == "steps":
        training_args.eval_steps = args.eval_steps

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        callbacks=[JSONLLoggingCallback(train_kwargs["ckpt_output_dir"])],
    )
    trainer.train()
    trainer.save_model(train_kwargs["ckpt_output_dir"])
    tokenizer.save_pretrained(train_kwargs["ckpt_output_dir"])


def main() -> None:
    args = parse_args()
    validate_args(args)

    data_path = Path(args.data_path)
    rows = load_dataset_rows(data_path)
    dataset_split = build_dataset_split(
        rows=rows,
        data_path=data_path,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )

    train_kwargs = build_train_kwargs(args, dataset_split.train_path)

    if args.dry_run:
        for key, value in build_dry_run_output(
            train_kwargs, dataset_split, args
        ).items():
            print(f"{key}={value}")
        return

    write_dataset_split(dataset_split)
    run_training(args, train_kwargs, dataset_split)


if __name__ == "__main__":
    main()
