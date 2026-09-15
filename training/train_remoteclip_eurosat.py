    # training/train_remoteclip_eurosat.py

from pathlib import Path
import json
import random
import pickle

import numpy as np
import torch
import open_clip
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import EuroSAT
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "eurosat"
CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "models--chendelong--RemoteCLIP"
    / "snapshots"
    / "bf1d8a3ccf2ddbf7c875705e46373bfe542bce38"
    / "RemoteCLIP-ViT-B-32.pt"
)

OUTPUT_DIR = ROOT / "checkpoints" / "adaptation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSIFIER_PATH = OUTPUT_DIR / "remoteclip_eurosat_linear_probe.pkl"
METADATA_PATH = OUTPUT_DIR / "remoteclip_eurosat_metadata.json"

DEVICE = "cpu"

SEED = 42

# CPU-friendly SIH training size.
# Increase later if required.
MAX_TRAIN_SAMPLES = 3000
MAX_TEST_SAMPLES = 1000

BATCH_SIZE = 16


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_remoteclip():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"RemoteCLIP checkpoint not found:\n{CHECKPOINT}"
        )

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained=None,
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            checkpoint = checkpoint["model"]

    cleaned_state_dict = {}

    for key, value in checkpoint.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        if new_key.startswith("model."):
            new_key = new_key[len("model."):]

        cleaned_state_dict[new_key] = value

    missing, unexpected = model.load_state_dict(
        cleaned_state_dict,
        strict=False,
    )

    print("REMOTECLIP CHECKPOINT:", CHECKPOINT)
    print("MISSING KEYS:", len(missing))
    print("UNEXPECTED KEYS:", len(unexpected))

    model = model.to(DEVICE)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model, preprocess


def build_dataset(preprocess):
    dataset = EuroSAT(
        root=str(DATA_DIR),
        download=True,
        transform=preprocess,
    )

    print("DATASET SIZE:", len(dataset))
    print("CLASSES:", dataset.classes)

    indices = np.arange(len(dataset))

    rng = np.random.default_rng(SEED)
    rng.shuffle(indices)

    split_index = int(len(indices) * 0.80)

    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    if MAX_TRAIN_SAMPLES is not None:
        train_indices = train_indices[:MAX_TRAIN_SAMPLES]

    if MAX_TEST_SAMPLES is not None:
        test_indices = test_indices[:MAX_TEST_SAMPLES]

    train_dataset = Subset(
        dataset,
        train_indices.tolist(),
    )

    test_dataset = Subset(
        dataset,
        test_indices.tolist(),
    )

    print("TRAIN SAMPLES:", len(train_dataset))
    print("TEST SAMPLES:", len(test_dataset))

    return dataset, train_dataset, test_dataset


@torch.inference_mode()
def extract_features(model, dataset):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    all_features = []
    all_labels = []

    total_batches = len(loader)

    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(DEVICE)

        features = model.encode_image(images)

        features = features / features.norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-12)

        all_features.append(
            features.cpu().numpy().astype(np.float32)
        )

        all_labels.append(
            labels.cpu().numpy().astype(np.int64)
        )

        print(
            f"\rFEATURE EXTRACTION: "
            f"{batch_index}/{total_batches}",
            end="",
            flush=True,
        )

    print()

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return features, labels


def main():
    set_seed(SEED)

    print("=" * 70)
    print("SATQUERY AI - REMOTE SENSING MODEL ADAPTATION")
    print("RemoteCLIP + EuroSAT Linear Probe")
    print("=" * 70)

    model, preprocess = load_remoteclip()

    full_dataset, train_dataset, test_dataset = build_dataset(
        preprocess
    )

    print("\nEXTRACTING TRAIN FEATURES...")
    x_train, y_train = extract_features(
        model,
        train_dataset,
    )

    print("\nEXTRACTING TEST FEATURES...")
    x_test, y_test = extract_features(
        model,
        test_dataset,
    )

    print("\nTRAIN FEATURE SHAPE:", x_train.shape)
    print("TEST FEATURE SHAPE:", x_test.shape)

    print("\nTRAINING LINEAR PROBE...")

    classifier = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=SEED,
    )

    classifier.fit(
        x_train,
        y_train,
    )

    predictions = classifier.predict(x_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    report = classification_report(
        y_test,
        predictions,
        target_names=full_dataset.classes,
        output_dict=True,
        zero_division=0,
    )

    print("\n" + "=" * 70)
    print("REMOTE SENSING ADAPTATION RESULT")
    print("=" * 70)
    print(f"TEST ACCURACY: {accuracy:.4f}")
    print(f"TEST ACCURACY %: {accuracy * 100:.2f}%")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=full_dataset.classes,
            zero_division=0,
        )
    )

    artifact = {
        "classifier": classifier,
        "classes": full_dataset.classes,
        "model_name": "RemoteCLIP-ViT-B-32",
        "adaptation_method": "frozen_encoder_linear_probe",
        "training_dataset": "EuroSAT",
        "feature_normalization": "L2",
    }

    with open(CLASSIFIER_PATH, "wb") as file:
        pickle.dump(
            artifact,
            file,
        )

    metadata = {
        "project": "SatQuery AI",
        "base_model": "RemoteCLIP-ViT-B-32",
        "remote_sensing_adapted": True,
        "adaptation_method": "frozen RemoteCLIP encoder + supervised linear probe",
        "adaptation_dataset": "EuroSAT",
        "dataset_type": "open-source remote-sensing benchmark",
        "num_classes": len(full_dataset.classes),
        "classes": full_dataset.classes,
        "train_samples": int(len(train_dataset)),
        "test_samples": int(len(test_dataset)),
        "feature_dimension": int(x_train.shape[1]),
        "test_accuracy": float(accuracy),
        "classification_report": report,
        "seed": SEED,
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print("\nSAVED CLASSIFIER:")
    print(CLASSIFIER_PATH)

    print("\nSAVED METADATA:")
    print(METADATA_PATH)

    print("\nPRIORITY 9 ADAPTATION: SUCCESS")


if __name__ == "__main__":
    main()