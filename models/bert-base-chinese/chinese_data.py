import json
import os
import pandas as pd
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset


def load_json_data(folder_path, label):
    data = []
    for file_name in ["wiki.json", "news.json", "webnovel.json"]:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            if folder_path.endswith("human"):
                for entry in json_data:
                    text = entry["input"] + entry["output"]
                    data.append({"text": text, "label": label})
            else:
                for key in json_data["input"]:
                    text = json_data["input"][key] + json_data["output"][key]
                    data.append({"text": text, "label": label})
    return data


def clean_text(text):
    text = " ".join(text.split())
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return text


class TextDataset(Dataset):
    def __init__(self, texts, labels, max_length=256):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(".")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def get_dataloaders(
    human_path="../../dataset/face2/human",
    generated_path="../../dataset/face2/generated",
    batch_size=16,
):
    # Load data
    human_data = load_json_data(human_path, 0)
    generated_data = load_json_data(generated_path, 1)
    print(
        f"Loaded {len(human_data)} human samples and {len(generated_data)} generated samples."
    )
    # print(f"First human sample: {human_data[0]}")
    # print(f"First generated sample: {generated_data[0]}")

    # Combine into DataFrame
    df = pd.DataFrame(human_data + generated_data)
    df["text"] = df["text"].apply(clean_text)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print("First 5 cleaned samples:")
    print(df["text"].head())

    # Create dataset
    dataset = TextDataset(df["text"].tolist(), df["label"].tolist())
    print(f"First sample: {dataset[0]['input_ids']}")
    print(f"Second sample: {dataset[1]['input_ids']}")

    # Split indices
    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        range(len(dataset)),
        dataset.labels,
        test_size=0.2,
        stratify=dataset.labels,
        random_state=42,
    )
    val_idx, test_idx, val_labels, test_labels = train_test_split(
        temp_idx, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
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
    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"Train loader size: {len(train_loader)}")
    print(f"Validation loader size: {len(val_loader)}")
    print(f"Test loader size: {len(test_loader)}")
    print(f"First batch from train loader: {next(iter(train_loader))}")
