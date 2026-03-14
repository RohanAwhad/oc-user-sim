from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

DEFAULT_DATA_PATH = Path("exports") / "reversed_messages.jsonl"
DEFAULT_OUTPUT_DIR = Path("outputs") / "lora-user-sim"
DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_MICRO_BATCH_SIZE = 2
DEFAULT_MAX_SEQ_LEN = 32768


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a LoRA user-simulator model with training_hub."
    )
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--ckpt-output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--micro-batch-size", type=int, default=DEFAULT_MICRO_BATCH_SIZE
    )
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--sample-packing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_train_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "ckpt_output_dir": args.ckpt_output_dir,
        "dataset_type": "chat_template",
        "field_messages": "messages",
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "micro_batch_size": args.micro_batch_size,
        "max_seq_len": args.max_seq_len,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "load_in_4bit": args.load_in_4bit,
        "bf16": True,
        "sample_packing": args.sample_packing,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_run_name": args.wandb_run_name,
    }


def validate_args(args: argparse.Namespace) -> None:
    if not Path(args.data_path).exists():
        raise SystemExit(f"Training data not found: {args.data_path}")


def main() -> None:
    args = parse_args()
    validate_args(args)

    train_kwargs = build_train_kwargs(args)

    if args.dry_run:
        for key, value in train_kwargs.items():
            print(f"{key}={value}")
        return

    try:
        import unsloth  # noqa: F401
        from training_hub import lora_sft
    except ImportError as exc:
        raise SystemExit(
            "training_hub is not installed. Install it first, then rerun this script."
        ) from exc

    lora_sft(**train_kwargs)


if __name__ == "__main__":
    main()
