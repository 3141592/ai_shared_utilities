from __future__ import annotations
import shutil
import tarfile
import zipfile
import subprocess
from pathlib import Path
from urllib.request import urlretrieve
from ai_shared_utilities.assets import get_asset_home


TMP_DIR = Path.home() / "tmp"


def ensure_tmp_dir() -> Path:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    return TMP_DIR


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(url: str, dest: Path) -> Path:
    ensure_dir(dest.parent)
    print(f"Downloading {url} -> {dest}")
    subprocess.run(["wget", "-O", str(dest), url], check=True)
    return dest

def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    ensure_dir(dest_dir)
    print(f"Extracting {zip_path} -> {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def extract_tar(tar_path: Path, dest_dir: Path) -> None:
    ensure_dir(dest_dir)
    print(f"Extracting {tar_path} -> {dest_dir}")
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(dest_dir)


def remove_file(path: Path) -> None:
    if path.exists():
        path.unlink()


def remove_tree(path: Path) -> None:
    if path.exists() and path.is_dir():
        shutil.rmtree(path)

# Deep Learning with Python Ch 8 - Dogs vs Cats image classification dataset from Kaggle
def build_dogs_vs_cats() -> None:
    root = get_asset_home("datasets") / "dogs-vs-cats"
    tmp = ensure_tmp_dir()
    zip_path = tmp / "dogs-vs-cats.zip"

    ensure_dir(root)

    try:
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", "dogs-vs-cats", "-p", str(tmp)],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to download dogs_vs_cats from Kaggle.\n"
            "Check Kaggle authentication and confirm you accepted the competition rules."
        ) from exc

    extract_zip(zip_path, root)

    train_zip = root / "train.zip"
    if train_zip.exists():
        extract_zip(train_zip, root)

    remove_file(zip_path)

# Deep Learning with Python Ch 9 
def build_oxford_pets() -> None:
    from ai_shared_utilities.assets import get_data_home
    from pathlib import Path

    data_root = get_data_home() / "datasets"
    tmp = ensure_tmp_dir()

    images_tar = tmp / "images.tar.gz"
    annotations_tar = tmp / "annotations.tar.gz"

    images_url = "http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    annotations_url = "http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

    # download
    download_file(images_url, images_tar)
    download_file(annotations_url, annotations_tar)

    # extract
    extract_tar(images_tar, data_root)
    extract_tar(annotations_tar, data_root)

    # cleanup
    remove_file(images_tar)
    remove_file(annotations_tar)

# Deep Learning with Python Ch 10.2 A temperature-forecasting example   
def build_jena_climate() -> None:
    root = get_asset_home("datasets")
    tmp = ensure_tmp_dir()
    zip_path = tmp / "jena_climate_2009_2016.csv.zip"

    download_file(
        "https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip",
        zip_path,
    )
    extract_zip(zip_path, root)
    remove_file(zip_path)

# Deep Learning with Python Ch 11 - IMDB movie reviews dataset
def build_acl_imdb() -> None:
    root = get_asset_home("datasets")
    tmp = ensure_tmp_dir()
    tar_path = tmp / "aclImdb_v1.tar.gz"

    download_file(
        "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        tar_path,
    )
    extract_tar(tar_path, root)

    unsup_dir = root / "aclImdb" / "train" / "unsup"
    remove_tree(unsup_dir)

    remove_file(tar_path)

# Deep Learning with Python Ch 11 - GloVe word embeddings
def build_glove_6B() -> None:
    root = get_asset_home("embeddings") / "glove.6B"
    tmp = ensure_tmp_dir()
    zip_path = tmp / "glove.6B.zip"

    download_file("http://nlp.stanford.edu/data/glove.6B.zip", zip_path)
    extract_zip(zip_path, root)
    remove_file(zip_path)

# Deep Learning with Python Ch 11 - Spanish-English translation dataset
def build_spa_eng() -> None:
    root = get_asset_home("datasets")
    tmp = ensure_tmp_dir()
    zip_path = tmp / "spa-eng.zip"

    download_file(
        "http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip",
        zip_path,
    )
    extract_zip(zip_path, root)
    remove_file(zip_path)

# Deep Learning with Python Ch 11 - FastText word embeddings
def build_fasttext_wiki_news() -> None:
    root = get_asset_home("embeddings") / "fasttext"
    tmp = ensure_tmp_dir()
    zip_path = tmp / "wiki-news-300d-1M.vec.zip"

    download_file(
        "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip",
        zip_path,
    )
    extract_zip(zip_path, root)
    remove_file(zip_path)

# Deep Learning with Python Ch 12 - Celeb-Gan
def build_celeba_gan() -> None:
    root = get_asset_home("datasets") / "celeba_gan"
    tmp = ensure_tmp_dir()
    zip_path = tmp / "img_align_celeba.zip"

    root.mkdir(parents=True, exist_ok=True)

    # Download using gdown
    subprocess.run([
        "gdown",
        "--id",
        "0B7EVK8r0v71pZjFTYXZWM3FlRnM",
        "-O",
        str(zip_path)
    ], check=True)

    # Extract images
    extract_zip(zip_path, root)

    remove_file(zip_path)

# Build a Large Language Model from Scratch - The Verdict
def build_the_verdict() -> None:
    root = get_asset_home("datasets") / "interpretability"
    tmp = ensure_tmp_dir()
    tmp_path = tmp / "the_verdict.txt"
    out_path = root / "the_verdict.txt"

    ensure_dir(root)

    download_file("https://en.wikisource.org/wiki/The_Verdict", tmp_path)
    shutil.copy2(tmp_path, out_path)
    remove_file(tmp_path)

# Build a Large Language Model from Scratch - ASV Bible text
def build_asv_raw() -> None:
    root = get_asset_home("datasets") / "interpretability"
    tmp = ensure_tmp_dir()
    tmp_path = tmp / "asv.txt"
    out_path = root / "asv.txt"

    ensure_dir(root)

    download_file("https://openbible.com/textfiles/asv.txt", tmp_path)
    shutil.copy2(tmp_path, out_path)
    remove_file(tmp_path)

# Build a Large Language Model from Scratch - Cleaned New Testament text for language modeling
def build_asv_clean_nt() -> None:
    import re

    from ai_shared_utilities.fetch import ensure_asset
    from ai_shared_utilities.assets import ensure_asset_dir

    asv_raw_path = ensure_asset("asv_raw")

    out_dir = ensure_asset_dir("datasets") / "interpretability"
    out_path = out_dir / "asv_clean_nt.txt"

    nt_books = {
        "Matthew", "Mark", "Luke", "John", "Acts", "Romans",
        "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians",
        "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians",
        "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews",
        "James", "1 Peter", "2 Peter", "1 John", "2 John", "3 John",
        "Jude", "Revelation",
    }

    # Matches lines like:
    # Matthew 1:1<TAB>text
    # 1 Corinthians 13:4<TAB>text
    verse_re = re.compile(r"^(.+?)\s+(\d+:\d+)\t(.*)$")

    lines_out = []
    current_book = None

    with asv_raw_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            match = verse_re.match(line)
            if not match:
                continue

            book = match.group(1).strip()
            verse_text = match.group(3).strip()

            if book not in nt_books:
                continue

            if not verse_text:
                continue

            if book != current_book:
                if current_book is not None:
                    lines_out.append("")
                current_book = book

            lines_out.append(verse_text)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")

# Build a Large Language Model from Scratch - Cleaned John's Gospel text for language modeling
def build_john() -> None:
    import re

    from ai_shared_utilities.fetch import ensure_asset
    from ai_shared_utilities.assets import ensure_asset_dir

    asv_raw_path = ensure_asset("asv_raw")

    out_dir = ensure_asset_dir("datasets") / "interpretability"
    out_path = out_dir / "john.txt"

    john = {"John"}

    # Matches lines like:
    # John 1:1<TAB>text
    verse_re = re.compile(r"^(.+?)\s+(\d+:\d+)\t(.*)$")

    lines_out = []
    current_book = None

    with asv_raw_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            match = verse_re.match(line)
            if not match:
                continue

            book = match.group(1).strip()
            verse_text = match.group(3).strip()

            if book not in john:
                continue

            if not verse_text:
                continue

            if book != current_book:
                if current_book is not None:
                    lines_out.append("")
                current_book = book

            lines_out.append(verse_text)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
