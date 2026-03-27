from __future__ import annotations

from itertools import cycle
from pathlib import Path
from typing import Iterator

from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def local_parquet_files_exist(train_path: Path, test_path: Path) -> bool:
    return train_path.exists() and test_path.exists()


def tokenize_and_chunk(
    dataset,
    tokenizer: AutoTokenizer,
    chunk_size: int,
    train_rows: int,
    test_rows: int,
):
    buffer: list[int] = []
    row_count = 0
    total_rows = train_rows + test_rows

    for example in dataset:
        tokens = tokenizer(example["article"], truncation=False, padding=False)["input_ids"]
        buffer.extend(tokens)

        while len(buffer) >= chunk_size + 1:
            if row_count >= total_rows:
                return

            input_chunk = buffer[:chunk_size]
            target_chunk = buffer[1 : chunk_size + 1]

            yield {
                "input": input_chunk,
                "target": target_chunk,
            }

            buffer = buffer[chunk_size:]
            row_count += 1


def ensure_tokenized_parquet(
    *,
    train_parquet_path: Path,
    test_parquet_path: Path,
    dataset_name: str,
    dataset_config: str,
    tokenizer_name: str,
    chunk_size: int,
    train_rows: int,
    test_rows: int,
    logger,
) -> None:
    if local_parquet_files_exist(train_parquet_path, test_parquet_path):
        logger.info("Tokenized parquet files already exist: %s and %s", train_parquet_path, test_parquet_path)
        return

    logger.info("Tokenized parquet files not found. Generating them from cnn_dailymail.")
    source_dataset = load_dataset(dataset_name, dataset_config, split="train")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    tokenized_dataset = Dataset.from_generator(
        lambda: tokenize_and_chunk(
            source_dataset,
            tokenizer,
            chunk_size=chunk_size,
            train_rows=train_rows,
            test_rows=test_rows,
        )
    )

    dataset_splits = tokenized_dataset.train_test_split(
        test_size=test_rows / (train_rows + test_rows),
        seed=42,
    )

    train_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    test_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_splits["train"].to_parquet(str(train_parquet_path))
    dataset_splits["test"].to_parquet(str(test_parquet_path))
    logger.info("Saved tokenized parquet datasets: train=%s test=%s", train_parquet_path, test_parquet_path)


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
