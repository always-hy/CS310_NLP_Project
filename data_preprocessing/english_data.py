import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split


def load_single_file(file_path, target_size=1024, min_size=100):
    """Load a single embedding file, fixing to target size."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            floats = [float(x) for x in f.read().strip().split()]
        if not floats:
            print(f"Warning: {file_path} is empty")
            return None
        size = len(floats)
        if size < min_size:
            print(
                f"Warning: {file_path} has {size} floats, below min_size={min_size}, skipping"
            )
            return None
        if size > target_size:
            floats = floats[:target_size]  # Truncate
        elif size < target_size:
            floats.extend([0.0] * (target_size - size))  # Pad
        return floats
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def load_data(
    main_folders, base_path="dataset/ghostbuster", target_size=1024, min_size=100
):
    """Load individual embeddings from gpt and human subfolders in wp and essay."""
    data = []
    labels = []
    vector_sizes = []

    for main_folder in main_folders:
        main_folder_path = os.path.join(base_path, main_folder)
        if not os.path.exists(main_folder_path):
            print(f"Folder {main_folder_path} does not exist")
            continue

        ai_subfolder = "gpt"
        human_subfolder = "human"

        # Load AI-generated embeddings
        ai_subfolder_path = os.path.join(main_folder_path, ai_subfolder)
        if os.path.exists(ai_subfolder_path):
            for file_name in os.listdir(ai_subfolder_path):
                if file_name.endswith(
                    ("-ada.txt", "-davinci.txt")
                ) and not file_name.startswith("combined"):
                    file_path = os.path.join(ai_subfolder_path, file_name)
                    embedding = load_single_file(file_path, target_size, min_size)
                    if embedding:
                        data.append(embedding)
                        labels.append(1)
                        vector_sizes.append(len(embedding))

        # Load human embeddings
        human_subfolder_path = os.path.join(main_folder_path, human_subfolder)
        if os.path.exists(human_subfolder_path):
            for file_name in os.listdir(human_subfolder_path):
                if file_name.endswith(
                    ("-ada.txt", "-davinci.txt")
                ) and not file_name.startswith("combined"):
                    file_path = os.path.join(human_subfolder_path, file_name)
                    embedding = load_single_file(file_path, target_size, min_size)
                    if embedding:
                        data.append(embedding)
                        labels.append(0)
                        vector_sizes.append(len(embedding))

    if vector_sizes:
        print(
            f"Loaded {len(data)} embeddings: {sum(1 for l in labels if l == 0)} human, {sum(1 for l in labels if l == 1)} AI"
        )
        print(f"All embeddings fixed to target_size={target_size}")
    else:
        print("No embeddings loaded")

    return data, labels


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = [torch.tensor(emb, dtype=torch.float32) for emb in embeddings]
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {
            "embeddings": self.embeddings[idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def get_dataloaders(
    base_path="dataset/ghostbuster", batch_size=16, target_size=1024, min_size=100
):
    """Create train, validation, and test DataLoaders."""
    main_folders = ["wp", "essay"]
    embeddings, labels = load_data(main_folders, base_path, target_size, min_size)

    if not embeddings:
        raise ValueError("No embeddings loaded. Check folder structure and file paths.")

    # Create dataset
    dataset = EmbeddingDataset(embeddings, labels)

    # Split indices (80/10/10)
    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        range(len(dataset)),
        dataset.labels,
        test_size=0.2,
        stratify=dataset.labels,
        random_state=42,
    )
    val_idx, test_idx, val_labels, test_labels = train_test_split(
        temp_idx,
        temp_labels,
        test_size=0.5,
        stratify=temp_labels,
        random_state=42,
    )

    # Create subsets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    try:
        train_loader, val_loader, test_loader = get_dataloaders()
        print(f"Train loader size: {len(train_loader)} batches")
        print(f"Validation loader size: {len(val_loader)} batches")
        print(f"Test loader size: {len(test_loader)} batches")
        print(f"First batch from train loader: {next(iter(train_loader))}")
    except Exception as e:
        print(f"Error in preprocessing: {e}")
