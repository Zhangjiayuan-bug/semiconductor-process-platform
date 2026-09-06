from io import BytesIO

import pytest

import scripts_download_secom


def test_download_script_skips_existing_dataset(tmp_path, monkeypatch):
    archive = tmp_path / "data" / "raw" / "secom.zip"
    destination = tmp_path / "data" / "raw" / "uci_secom"
    destination.mkdir(parents=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(b"already downloaded")
    (destination / "secom.data").write_text("1 2 3\n", encoding="utf-8")
    (destination / "secom_labels.data").write_text('-1 "19/07/2008 11:55:00"\n', encoding="utf-8")

    def fail_urlopen(*_args, **_kwargs):
        raise AssertionError("urlopen should not be called when SECOM is already present")

    monkeypatch.setattr(scripts_download_secom, "ARCHIVE", archive)
    monkeypatch.setattr(scripts_download_secom, "DESTINATION", destination)
    monkeypatch.setattr(scripts_download_secom, "urlopen", fail_urlopen)

    scripts_download_secom.main()


def test_download_script_rejects_untrusted_archive(tmp_path, monkeypatch):
    archive = tmp_path / "data" / "raw" / "secom.zip"
    destination = tmp_path / "data" / "raw" / "uci_secom"
    monkeypatch.setattr(scripts_download_secom, "ARCHIVE", archive)
    monkeypatch.setattr(scripts_download_secom, "DESTINATION", destination)
    monkeypatch.setattr(scripts_download_secom, "urlopen", lambda *_args, **_kwargs: BytesIO(b"not the official archive"))

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        scripts_download_secom.main()

    assert not archive.exists()
