import hashlib
import shutil
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"
EXPECTED_SHA256 = "eea568baf3c2229096d7d294cf0b096b5502bd96d92c0b80a65b84714059be8e"
REQUIRED_FILES = {"secom.data", "secom.names", "secom_labels.data"}
ROOT = Path(__file__).parent
ARCHIVE = ROOT / "data" / "raw" / "secom.zip"
DESTINATION = ROOT / "data" / "raw" / "uci_secom"


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists() and (DESTINATION / "secom.data").exists() and (DESTINATION / "secom_labels.data").exists():
        print(f"SECOM already present at {DESTINATION}")
        return
    with urlopen(URL, timeout=60) as response:
        content = response.read()
    digest = hashlib.sha256(content).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"SECOM archive checksum mismatch: expected {EXPECTED_SHA256}, got {digest}")
    ARCHIVE.write_bytes(content)
    with ZipFile(ARCHIVE) as archive:
        members = set(archive.namelist())
        if not REQUIRED_FILES.issubset(members):
            missing = ", ".join(sorted(REQUIRED_FILES - members))
            raise RuntimeError(f"SECOM archive is missing required files: {missing}")
        for name in REQUIRED_FILES:
            with archive.open(name) as source, (DESTINATION / name).open("wb") as target:
                shutil.copyfileobj(source, target)
    print(f"Downloaded and extracted UCI SECOM to {DESTINATION}")


if __name__ == "__main__":
    main()
