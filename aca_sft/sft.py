from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from blob_utils import BlobClientHelper
    from data_utils import (
        create_eval_loader,
        create_train_loader,
        disable_hf_transfer_runtime,
        ensure_tokenized_parquet,
        load_parquet_datasets,
    )
    from modeling import GPTConfig, GPTModel, count_parameters
else:
    from aca_sft.blob_utils import BlobClientHelper
    from aca_sft.data_utils import (
        create_eval_loader,
        create_train_loader,
        disable_hf_transfer_runtime,
        ensure_tokenized_parquet,
        load_parquet_datasets,
    )
    from aca_sft.modeling import GPTConfig, GPTModel, count_parameters


# CONFIG
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"
MODEL_INPUTS_DIR = PROJECT_ROOT / "model_inputs"
TRAIN_PARQUET_PATH = DATA_DIR / "fact_qa_train.parquet"
TEST_PARQUET_PATH = DATA_DIR / "fact_qa_test.parquet"
LOCAL_PRETRAINED_CHECKPOINT_PATH = Path(
    os.getenv("LOCAL_PRETRAINED_CHECKPOINT_PATH", str(MODEL_INPUTS_DIR / "pretrain_final.pth"))
)

TOKENIZER_NAME = "gpt2"
HF_DATASET_NAME = "rubenroy/GammaCorpus-Fact-QA-450k"
DISABLE_HF_TRANSFER = True

RUN_MODE = "full"  # "smoke" or "full"

SMOKE_TEST_CONFIG = {
    "TRAIN_ROWS": 2_000,
    "TEST_ROWS": 100,
    "BATCH_SIZE": 16,
    "NUM_STEPS": 100,
    "EVAL_INTERVAL_STEPS": 10,
    "CHECKPOINT_INTERVAL_STEPS": 50,
}

FULL_TRAINING_CONFIG = {
    "TRAIN_ROWS": 440_000,
    "TEST_ROWS": 500,
    "BATCH_SIZE": 64,
    "NUM_STEPS": 50_000,
    "EVAL_INTERVAL_STEPS": 100,
    "CHECKPOINT_INTERVAL_STEPS": 25_000,
}

ACTIVE_CONFIG = SMOKE_TEST_CONFIG if RUN_MODE == "smoke" else FULL_TRAINING_CONFIG

TRAIN_ROWS = ACTIVE_CONFIG["TRAIN_ROWS"]
TEST_ROWS = ACTIVE_CONFIG["TEST_ROWS"]
SEQUENCE_LEN = 128
BATCH_SIZE = ACTIVE_CONFIG["BATCH_SIZE"]
NUM_STEPS = ACTIVE_CONFIG["NUM_STEPS"]
EVAL_INTERVAL_STEPS = ACTIVE_CONFIG["EVAL_INTERVAL_STEPS"]
CHECKPOINT_INTERVAL_STEPS = ACTIVE_CONFIG["CHECKPOINT_INTERVAL_STEPS"]

LEARNING_RATE = 5e-4
MAX_GRAD_NORM = 1.0
LR_SCHEDULER_FACTOR = 0.2
LR_SCHEDULER_PATIENCE = 10
LR_SCHEDULER_MIN_LR = 5e-6
LR_SCHEDULER_THRESHOLD = 1e-4

RANDOM_SEED = 42
UPLOAD_TO_BLOB = True
REQUIRE_GPU = True
RESUME_FROM_LATEST_CHECKPOINT = True

PRETRAINED_MODEL_BLOB_PATH = os.getenv("PRETRAINED_MODEL_BLOB_PATH", "").strip()
REQUIRE_PRETRAINED_MODEL = True


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def configure_logging(log_file_path: Path) -> logging.Logger:
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("aca_sft")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def detect_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_runtime_device(logger: logging.Logger) -> None:
    logger.info("Torch version: %s", torch.__version__)
    logger.info("Torch CUDA runtime version: %s", torch.version.cuda)
    logger.info("CUDA available: %s", torch.cuda.is_available())

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info("Detected GPU: %s", gpu_name)
        logger.info("Detected GPU count: %s", gpu_count)
        logger.info("Detected GPU total memory (GiB): %.2f", total_memory_gb)
        return

    if REQUIRE_GPU:
        raise RuntimeError(
            "GPU is required for this job, but torch.cuda.is_available() is False. "
            "The container will not continue."
        )


def create_run_paths(run_timestamp: str) -> dict[str, Path]:
    run_root = OUTPUTS_ROOT / "sft_runs" / run_timestamp
    paths = {
        "run_root": run_root,
        "logs_dir": run_root / "logs",
        "checkpoints_dir": run_root / "checkpoints",
        "model_dir": run_root / "model",
        "metrics_dir": run_root / "metrics",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_metrics(metrics_path: Path, payload: dict) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def evaluate_model(model: GPTModel, eval_loader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    batch_count = 0
    with torch.no_grad():
        for batch in eval_loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            total_loss += loss.item()
            batch_count += 1
    return total_loss / max(batch_count, 1)


def save_checkpoint(
    checkpoint_path: Path,
    *,
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: GPTConfig,
    step: int,
    latest_train_loss: float | None,
    latest_eval_loss: float | None,
    pretrained_checkpoint_source: str,
) -> None:
    torch.save(
        {
            "step": step,
            "model_config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "latest_train_loss": latest_train_loss,
            "latest_eval_loss": latest_eval_loss,
            "pretrained_checkpoint_source": pretrained_checkpoint_source,
        },
        checkpoint_path,
    )


def save_final_model(
    final_model_path: Path,
    *,
    model: GPTModel,
    config: GPTConfig,
    step: int,
    pretrained_checkpoint_source: str,
) -> None:
    torch.save(
        {
            "step": step,
            "model_config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "pretrained_checkpoint_source": pretrained_checkpoint_source,
        },
        final_model_path,
    )


def find_latest_checkpoint(checkpoints_dir: Path) -> Path | None:
    checkpoint_paths = sorted(
        checkpoints_dir.glob("sft_model_checkpoint_*.pt"),
        key=lambda path: path.stat().st_mtime,
    )
    return checkpoint_paths[-1] if checkpoint_paths else None


def maybe_resume_from_checkpoint(
    *,
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    checkpoints_dir: Path,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[int, float | None, float | None]:
    if not RESUME_FROM_LATEST_CHECKPOINT:
        logger.info("Checkpoint resume disabled in config.")
        return 0, None, None

    checkpoint_path = find_latest_checkpoint(checkpoints_dir)
    if checkpoint_path is None:
        logger.info("No SFT checkpoint found to resume from in %s", checkpoints_dir)
        return 0, None, None

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    resumed_step = int(checkpoint.get("step", 0))
    latest_train_loss = checkpoint.get("latest_train_loss")
    latest_eval_loss = checkpoint.get("latest_eval_loss")
    logger.info("Resumed from SFT checkpoint: %s", checkpoint_path)
    logger.info("Resume step: %s", resumed_step)
    return resumed_step, latest_train_loss, latest_eval_loss


def maybe_upload_run_directory(logger: logging.Logger, run_root: Path, run_timestamp: str) -> None:
    if not UPLOAD_TO_BLOB:
        logger.info("Blob upload disabled in config. Skipping upload.")
        return

    blob_helper = BlobClientHelper.from_env(logger)
    if blob_helper is None:
        return

    blob_helper.upload_directory(run_root, blob_prefix=f"runs/sft/{run_timestamp}")


def resolve_pretrained_checkpoint(logger: logging.Logger) -> Path:
    if LOCAL_PRETRAINED_CHECKPOINT_PATH.exists():
        logger.info("Using local pretrained checkpoint: %s", LOCAL_PRETRAINED_CHECKPOINT_PATH)
        return LOCAL_PRETRAINED_CHECKPOINT_PATH

    blob_path = PRETRAINED_MODEL_BLOB_PATH.strip()
    if blob_path:
        blob_helper = BlobClientHelper.from_env(logger)
        if blob_helper is not None and blob_helper.download_file(blob_path, LOCAL_PRETRAINED_CHECKPOINT_PATH):
            logger.info("Using downloaded pretrained checkpoint: %s", LOCAL_PRETRAINED_CHECKPOINT_PATH)
            return LOCAL_PRETRAINED_CHECKPOINT_PATH

    if REQUIRE_PRETRAINED_MODEL:
        raise FileNotFoundError(
            "No pretrained checkpoint was found locally, and no downloadable blob checkpoint was resolved. "
            "Set LOCAL_PRETRAINED_CHECKPOINT_PATH or PRETRAINED_MODEL_BLOB_PATH."
        )

    raise FileNotFoundError("Pretrained checkpoint not found.")


def load_pretrained_model(pretrained_checkpoint_path: Path, device: torch.device, logger: logging.Logger) -> GPTModel:
    checkpoint = torch.load(pretrained_checkpoint_path, map_location=device)
    config_payload = checkpoint["model_config"]
    model_config = GPTConfig(
        vocab_size=config_payload["vocab_size"],
        n_layer=config_payload["n_layer"],
        n_head=config_payload["n_head"],
        n_embd=config_payload["n_embd"],
        seq_len=config_payload["seq_len"],
    )
    model = GPTModel(model_config, device=device).to(device)

    state_dict = checkpoint["model_state_dict"].copy()
    state_dict.pop("position_encoding", None)
    load_result = model.load_state_dict(state_dict, strict=False)

    logger.info("Loaded pretrained checkpoint from: %s", pretrained_checkpoint_path)
    logger.info("Pretrained checkpoint step: %s", checkpoint.get("step"))
    if load_result.missing_keys:
        logger.info("Missing keys while loading pretrained model: %s", load_result.missing_keys)
    if load_result.unexpected_keys:
        logger.info("Unexpected keys while loading pretrained model: %s", load_result.unexpected_keys)
    return model


def log_startup_config(logger: logging.Logger, device: torch.device, run_paths: dict[str, Path]) -> None:
    payload = {
        "device": str(device),
        "run_mode": RUN_MODE,
        "train_rows": TRAIN_ROWS,
        "test_rows": TEST_ROWS,
        "sequence_len": SEQUENCE_LEN,
        "batch_size": BATCH_SIZE,
        "num_steps": NUM_STEPS,
        "eval_interval_steps": EVAL_INTERVAL_STEPS,
        "checkpoint_interval_steps": CHECKPOINT_INTERVAL_STEPS,
        "learning_rate": LEARNING_RATE,
        "train_parquet_path": str(TRAIN_PARQUET_PATH),
        "test_parquet_path": str(TEST_PARQUET_PATH),
        "local_pretrained_checkpoint_path": str(LOCAL_PRETRAINED_CHECKPOINT_PATH),
        "pretrained_model_blob_path": PRETRAINED_MODEL_BLOB_PATH,
        "run_root": str(run_paths["run_root"]),
    }
    logger.info("Startup config: %s", json.dumps(payload, sort_keys=True))


def main() -> int:
    load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)

    if DISABLE_HF_TRANSFER:
        disable_hf_transfer_runtime()

    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.cuda.empty_cache()

    run_timestamp = utc_timestamp()
    run_paths = create_run_paths(run_timestamp)
    log_file_path = run_paths["logs_dir"] / "train.log"
    metrics_path = run_paths["metrics_dir"] / "metrics.json"
    final_model_path = run_paths["model_dir"] / "sft_final.pth"

    logger = configure_logging(log_file_path)
    device = detect_device()

    model: GPTModel | None = None
    optimizer = None
    scheduler = None
    last_completed_step = 0
    latest_train_loss: float | None = None
    latest_eval_loss: float | None = None
    pretrained_checkpoint_path: Path | None = None
    metrics_payload = {
        "run_timestamp": run_timestamp,
        "status": "running",
        "device": str(device),
        "run_mode": RUN_MODE,
        "train_losses": [],
        "eval_losses": [],
        "checkpoint_paths": [],
        "final_model_path": str(final_model_path),
    }

    try:
        logger.info("Starting SFT job.")
        logger.info("Device detected: %s", device)
        validate_runtime_device(logger)
        log_startup_config(logger, device, run_paths)

        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
        logger.info("Loaded tokenizer: %s (vocab_size=%s)", TOKENIZER_NAME, tokenizer.vocab_size)

        ensure_tokenized_parquet(
            train_parquet_path=TRAIN_PARQUET_PATH,
            test_parquet_path=TEST_PARQUET_PATH,
            dataset_name=HF_DATASET_NAME,
            tokenizer_name=TOKENIZER_NAME,
            chunk_size=SEQUENCE_LEN,
            train_rows=TRAIN_ROWS,
            test_rows=TEST_ROWS,
            logger=logger,
        )

        train_ds, test_ds = load_parquet_datasets(TRAIN_PARQUET_PATH, TEST_PARQUET_PATH)
        logger.info("Dataset sizes: train=%s test=%s", len(train_ds), len(test_ds))

        train_loader = create_train_loader(train_ds, batch_size=BATCH_SIZE)
        eval_loader = create_eval_loader(test_ds, batch_size=BATCH_SIZE)

        pretrained_checkpoint_path = resolve_pretrained_checkpoint(logger)
        model = load_pretrained_model(pretrained_checkpoint_path, device, logger)
        parameter_count = count_parameters(model)
        logger.info("Model parameter count: %s", parameter_count)
        metrics_payload["parameter_count"] = parameter_count
        metrics_payload["model_config"] = model.config.to_dict()
        metrics_payload["pretrained_checkpoint_path"] = str(pretrained_checkpoint_path)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=LR_SCHEDULER_FACTOR,
            patience=LR_SCHEDULER_PATIENCE,
            min_lr=LR_SCHEDULER_MIN_LR,
            threshold=LR_SCHEDULER_THRESHOLD,
        )

        resumed_step, resumed_train_loss, resumed_eval_loss = maybe_resume_from_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoints_dir=run_paths["checkpoints_dir"],
            device=device,
            logger=logger,
        )
        last_completed_step = resumed_step
        latest_train_loss = resumed_train_loss
        latest_eval_loss = resumed_eval_loss
        metrics_payload["resumed_from_step"] = resumed_step

        running_train_loss = 0.0
        start_time = time.time()

        for step in range(resumed_step + 1, NUM_STEPS + 1):
            model.train()
            batch = next(train_loader)
            train_input = batch["input"].to(device)
            train_target = batch["target"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(train_input)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), train_target.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
            optimizer.step()

            last_completed_step = step
            running_train_loss += loss.item()

            if step % EVAL_INTERVAL_STEPS == 0:
                latest_train_loss = running_train_loss / EVAL_INTERVAL_STEPS
                running_train_loss = 0.0

                latest_eval_loss = evaluate_model(model, eval_loader, device)
                scheduler.step(latest_eval_loss)

                elapsed_time = time.time() - start_time
                lr = optimizer.param_groups[0]["lr"]
                metrics_payload["train_losses"].append({"step": step, "loss": latest_train_loss})
                metrics_payload["eval_losses"].append({"step": step, "loss": latest_eval_loss})
                write_metrics(metrics_path, metrics_payload)

                logger.info(
                    "Evaluation report | step=%s/%s | train_loss=%.6f | eval_loss=%.6f | lr=%.8f | elapsed_seconds=%.2f",
                    step,
                    NUM_STEPS,
                    latest_train_loss,
                    latest_eval_loss,
                    lr,
                    elapsed_time,
                )

                eval_loader = create_eval_loader(test_ds, batch_size=BATCH_SIZE)

            if step % CHECKPOINT_INTERVAL_STEPS == 0:
                checkpoint_path = run_paths["checkpoints_dir"] / f"sft_model_checkpoint_{step}.pt"
                save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=model.config,
                    step=step,
                    latest_train_loss=latest_train_loss,
                    latest_eval_loss=latest_eval_loss,
                    pretrained_checkpoint_source=str(pretrained_checkpoint_path),
                )
                metrics_payload["checkpoint_paths"].append(str(checkpoint_path))
                write_metrics(metrics_path, metrics_payload)
                logger.info("Saved checkpoint: %s", checkpoint_path)

        save_final_model(
            final_model_path,
            model=model,
            config=model.config,
            step=last_completed_step,
            pretrained_checkpoint_source=str(pretrained_checkpoint_path),
        )
        logger.info("Saved final model: %s", final_model_path)
        metrics_payload["status"] = "completed"
        metrics_payload["completed_step"] = last_completed_step
        write_metrics(metrics_path, metrics_payload)

        try:
            maybe_upload_run_directory(logger, run_paths["run_root"], run_timestamp)
        except Exception:
            logger.exception("Blob upload failed after successful SFT training. Local artifacts were preserved.")
        logger.info("SFT job completed successfully.")
        return 0

    except Exception:
        logger.exception("SFT job failed with an exception.")
        metrics_payload["status"] = "failed"
        metrics_payload["completed_step"] = last_completed_step
        write_metrics(metrics_path, metrics_payload)

        if model is not None and optimizer is not None and scheduler is not None and last_completed_step > 0:
            crash_checkpoint_path = run_paths["checkpoints_dir"] / f"sft_model_checkpoint_{last_completed_step}_crash.pt"
            try:
                save_checkpoint(
                    crash_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=model.config,
                    step=last_completed_step,
                    latest_train_loss=latest_train_loss,
                    latest_eval_loss=latest_eval_loss,
                    pretrained_checkpoint_source=str(pretrained_checkpoint_path) if pretrained_checkpoint_path else "",
                )
                logger.info("Saved crash checkpoint: %s", crash_checkpoint_path)
            except Exception:
                logger.exception("Failed to save crash checkpoint.")

        try:
            maybe_upload_run_directory(logger, run_paths["run_root"], run_timestamp)
        except Exception:
            logger.exception("Failed to upload partial SFT artifacts after crash.")

        return 1

    finally:
        for handler in logging.getLogger("aca_sft").handlers:
            handler.flush()
        logging.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
