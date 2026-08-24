from pathlib import Path

import app.workers.feature_worker as feature_worker
from app.core import config as config_module


def test_resolve_local_dataset_path_from_backend_storage(tmp_path):
    dataset_file = tmp_path / "storage" / "datasets" / "raw_20260814T121921Z_cb29a9ae" / "data.csv"
    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    dataset_file.write_text("amount,label\n10,1\n", encoding="utf-8")

    resolved = feature_worker._resolve_local_storage_path(
        r"storage\datasets\raw_20260814T121921Z_cb29a9ae\data.csv",
        base_dir=tmp_path,
    )

    assert resolved is not None
    assert resolved == dataset_file
    assert resolved.exists()


def test_local_dataset_bytes_are_loaded_without_azure(tmp_path, monkeypatch):
    dataset_file = tmp_path / "storage" / "datasets" / "raw_20260814T121921Z_cb29a9ae" / "data.csv"
    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    dataset_file.write_bytes(b"amount,label\n10,1\n")

    monkeypatch.setattr(config_module.settings, "AZURE_STORAGE_CONNECTION_STRING", "", raising=False)

    dataset_bytes = feature_worker._read_dataset_bytes(
        r"storage\datasets\raw_20260814T121921Z_cb29a9ae\data.csv",
        file_format="csv",
        base_dir=tmp_path,
    )

    assert dataset_bytes == b"amount,label\n10,1\n"


def test_local_dataset_path_is_used_before_any_azure_call(tmp_path, monkeypatch):
    dataset_file = tmp_path / "storage" / "datasets" / "raw_20260814T121921Z_cb29a9ae" / "data.csv"
    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    dataset_file.write_text("amount,label\n10,1\n", encoding="utf-8")

    monkeypatch.setattr(config_module.settings, "AZURE_STORAGE_CONNECTION_STRING", "", raising=False)
    azure_was_called = False

    def fail_if_used():
        nonlocal azure_was_called
        azure_was_called = True
        raise AssertionError("BlobServiceClient.from_connection_string should never be called for local datasets")

    monkeypatch.setattr(feature_worker, "_get_azure_connection_string", fail_if_used)

    result = feature_worker._read_dataset_bytes(
        r"storage\datasets\raw_20260814T121921Z_cb29a9ae\data.csv",
        base_dir=tmp_path,
    )

    assert result == dataset_file.read_bytes()
    assert azure_was_called is False
