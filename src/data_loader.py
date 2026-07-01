import os
import zipfile
import urllib.request

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
RAW_DATA_DIR = "data/raw"
EXTRACTED_DIR = os.path.join(RAW_DATA_DIR, "ml-latest-small")


def download_movielens():
    zip_path = os.path.join(RAW_DATA_DIR, "ml-latest-small.zip")

    if os.path.exists(EXTRACTED_DIR):
        print(f"Dataset already exists at {EXTRACTED_DIR}. Skipping download.")
        return EXTRACTED_DIR

    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    print(f"Downloading MovieLens dataset...")

    urllib.request.urlretrieve(MOVIELENS_URL, zip_path)
    print("Download complete. Extracting...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(RAW_DATA_DIR)

    os.remove(zip_path)
    print(f"Ready. Files at {EXTRACTED_DIR}")

    return EXTRACTED_DIR


def get_file_paths(data_dir):
    return {
        "ratings": os.path.join(data_dir, "ratings.csv"),
        "movies":  os.path.join(data_dir, "movies.csv"),
        "tags":    os.path.join(data_dir, "tags.csv"),
        "links":   os.path.join(data_dir, "links.csv"),
    }


if __name__ == "__main__":
    data_dir = download_movielens()
    paths = get_file_paths(data_dir)

    print("\nDataset files:")
    for name, path in paths.items():
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  {name:10s} → {path}  ({size_mb:.2f} MB)")