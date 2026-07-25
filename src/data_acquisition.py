"""Restructures the raw APTOS gaussian-filtered dataset into the project's
3-class scheme and a stratified train/test/retrain_pool split.

Source: retina data/train.csv + retina data/gaussian_filtered_images/gaussian_filtered_images/<5-class folders>
Output: data/{train,test,retrain_pool}/<3-class folders>/*.png, data/manifest.csv
"""
import os
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_CSV_PATH = "retina_data/train.csv"
RAW_IMAGE_DIR = "retina_data/gaussian_filtered_images/gaussian_filtered_images"
DATA_DIR = "data"
RANDOM_STATE = 42

TEST_FRACTION = 0.15
RETRAIN_POOL_FRACTION = 0.15

# Raw diagnosis (0-4) -> 5-class source folder name
FIVE_CLASS_FOLDER = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferate_DR",
}

# Raw diagnosis (0-4) -> locked 3-class label + folder name (see dataset_decisions memory)
THREE_CLASS_MAP = {
    0: (0, "No_DR"),
    1: (1, "Mild_Moderate"),
    2: (1, "Mild_Moderate"),
    3: (2, "Severe_Proliferate_DR"),
    4: (2, "Severe_Proliferate_DR"),
}


def build_manifest() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV_PATH)
    df["diagnosis_3class"] = df["diagnosis"].map(lambda d: THREE_CLASS_MAP[d][0])
    df["class_name"] = df["diagnosis"].map(lambda d: THREE_CLASS_MAP[d][1])
    df["filename"] = df["id_code"] + ".png"
    df["src_path"] = df["diagnosis"].map(FIVE_CLASS_FOLDER).combine(
        df["filename"], lambda folder, fname: os.path.join(RAW_IMAGE_DIR, folder, fname)
    )
    return df


def stratified_split(df: pd.DataFrame) -> pd.DataFrame:
    train_df, holdout_df = train_test_split(
        df,
        test_size=TEST_FRACTION + RETRAIN_POOL_FRACTION,
        stratify=df["diagnosis_3class"],
        random_state=RANDOM_STATE,
    )
    retrain_share_of_holdout = RETRAIN_POOL_FRACTION / (TEST_FRACTION + RETRAIN_POOL_FRACTION)
    test_df, retrain_df = train_test_split(
        holdout_df,
        test_size=retrain_share_of_holdout,
        stratify=holdout_df["diagnosis_3class"],
        random_state=RANDOM_STATE,
    )

    train_df = train_df.assign(split="train")
    test_df = test_df.assign(split="test")
    retrain_df = retrain_df.assign(split="retrain_pool")
    return pd.concat([train_df, test_df, retrain_df], ignore_index=True)


def move_images(manifest: pd.DataFrame) -> None:
    for row in manifest.itertuples(index=False):
        dest_dir = os.path.join(DATA_DIR, row.split, row.class_name)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, row.filename)
        if os.path.exists(dest_path):
            continue
        if os.path.exists(row.src_path):
            shutil.move(row.src_path, dest_path)


def print_summary(manifest: pd.DataFrame) -> None:
    table = manifest.groupby(["split", "class_name"]).size().unstack(fill_value=0)
    print(table)
    print("\nTotal per split:")
    print(manifest["split"].value_counts())


def main() -> None:
    df = build_manifest()
    manifest = stratified_split(df)
    move_images(manifest)

    manifest_out = manifest[
        ["id_code", "filename", "diagnosis", "diagnosis_3class", "class_name", "split"]
    ]
    manifest_path = os.path.join(DATA_DIR, "manifest.csv")
    manifest_out.to_csv(manifest_path, index=False)

    print_summary(manifest)
    print(f"\nManifest written to {manifest_path}")


if __name__ == "__main__":
    main()
