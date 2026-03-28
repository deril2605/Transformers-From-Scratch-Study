from __future__ import annotations

import os
from itertools import cycle
from pathlib import Path
from typing import Iterator

from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def disable_hf_transfer_runtime() -> None:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    try:
        import huggingface_hub.constants as hf_constants

        hf_constants.HF_HUB_ENABLE_HF_TRANSFER = False
    except Exception:
        pass


def local_parquet_files_exist(train_path: Path, test_path: Path) -> bool:
    return train_path.exists() and test_path.exists()


def tokenize_and_chunk(dataset, tokenizer: AutoTokenizer, chunk_size: int, total_rows: int):
    row_count = 0

    for example in dataset:
        if row_count >= total_rows:
            return

        question_plus_answer = (
            "<Question>"
            + example["question"]
            + "</Question>"
            + "<Answer>"
            + example["answer"]
            + "</Answer>"
        )
        input_tokens = tokenizer(question_plus_answer, truncation=False, padding=False)["input_ids"]

        if len(input_tokens) >= chunk_size:
            continue

        input_tokens = input_tokens + [tokenizer.eos_token_id] * (chunk_size - len(input_tokens))
        target_tokens = input_tokens[1:] + [tokenizer.eos_token_id]

        yield {
            "input": input_tokens,
            "target": target_tokens,
        }

        row_count += 1


def ensure_tokenized_parquet(
    *,
    train_parquet_path: Path,
    test_parquet_path: Path,
    dataset_name: str,
    tokenizer_name: str,
    chunk_size: int,
    train_rows: int,
    test_rows: int,
    logger,
) -> None:
    if local_parquet_files_exist(train_parquet_path, test_parquet_path):
        logger.info("Tokenized SFT parquet files already exist: %s and %s", train_parquet_path, test_parquet_path)
        return

    disable_hf_transfer_runtime()
    logger.info("Tokenized SFT parquet files not found. Generating them from %s.", dataset_name)
    source_dataset = load_dataset(dataset_name, split="train", streaming=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    tokenized_dataset = Dataset.from_generator(
        lambda: tokenize_and_chunk(
            source_dataset,
            tokenizer,
            chunk_size=chunk_size,
            total_rows=train_rows + test_rows,
        )
    )

    dataset_splits = tokenized_dataset.train_test_split(
        train_size=train_rows,
        test_size=test_rows,
        seed=42,
    )

    train_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    test_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_splits["train"].to_parquet(str(train_parquet_path))
    dataset_splits["test"].to_parquet(str(test_parquet_path))
    logger.info("Saved tokenized SFT parquet datasets: train=%s test=%s", train_parquet_path, test_parquet_path)


def load_parquet_datasets(train_parquet_path: Path, test_parquet_path: Path):
    train_ds = load_dataset("parquet", data_files=str(train_parquet_path), split="train")
    test_ds = load_dataset("parquet", data_files=str(test_parquet_path), split="train")
    train_ds.set_format("torch", columns=["input", "target"])
    test_ds.set_format("torch", columns=["input", "target"])
    return train_ds, test_ds


def create_train_loader(train_ds, batch_size: int) -> Iterator[dict]:
    return cycle(DataLoader(train_ds, batch_size=batch_size, shuffle=False))


def create_eval_loader(test_ds, batch_size: int) -> DataLoader:
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False)
