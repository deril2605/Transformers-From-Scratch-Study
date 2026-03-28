from __future__ import annotations

import logging
import os
from pathlib import Path

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv


class BlobClientHelper:
    def __init__(self, connection_string: str, container_name: str, logger: logging.Logger) -> None:
        self.container_name = container_name
        self.logger = logger
        self.service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.service_client.get_container_client(container_name)
        try:
            self.container_client.create_container()
        except ResourceExistsError:
            self.logger.info("Blob container already exists: %s", container_name)

    @classmethod
    def from_env(cls, logger: logging.Logger) -> "BlobClientHelper | None":
        load_dotenv()

        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "").strip()
        container_name = os.getenv("AZURE_BLOB_CONTAINER", "").strip()

        if not connection_string or not container_name:
            logger.warning(
                "Blob operations disabled because AZURE_STORAGE_CONNECTION_STRING or AZURE_BLOB_CONTAINER is missing."
            )
            return None

        return cls(connection_string=connection_string, container_name=container_name, logger=logger)

    def upload_file(self, local_path: Path, blob_path: str) -> None:
        with local_path.open("rb") as handle:
            self.container_client.upload_blob(
                name=blob_path.replace("\\", "/"),
                data=handle,
                overwrite=True,
            )
        self.logger.info("Uploaded blob: %s -> %s", local_path, blob_path)

    def upload_directory(self, local_dir: Path, blob_prefix: str) -> None:
        for file_path in sorted(path for path in local_dir.rglob("*") if path.is_file()):
            relative_path = file_path.relative_to(local_dir).as_posix()
            blob_path = f"{blob_prefix.rstrip('/')}/{relative_path}"
            self.upload_file(file_path, blob_path)

    def download_file(self, blob_path: str, local_path: Path) -> bool:
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with local_path.open("wb") as handle:
                handle.write(self.container_client.download_blob(blob_path).readall())
            self.logger.info("Downloaded blob: %s -> %s", blob_path, local_path)
            return True
        except ResourceNotFoundError:
            self.logger.warning("Blob not found: %s", blob_path)
            return False
