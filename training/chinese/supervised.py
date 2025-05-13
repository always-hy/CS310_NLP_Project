# Import libraries
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load data
dir_name = '../../dataset/face2/'
human_df = pd.read_csv(dir_name + 'generated/generated_webnovel_data_200.csv')
generated_df = pd.read_csv(dir_name + 'human/human_webnovel_data_200.csv')

test_size = 0.2
bert_base_dir = 'bert-base-chinese'
num_epochs = 3
learning_rate = 2e-5
max_length = 512
batch_size = 8

MODEL_SAVE_PATH = '../../results/chinese_bert_model_1.pt'
save_logits_path = '../../results/chinese_bert_logits_1.pt'

# Combine and shuffle
webnovel_df = pd.concat([human_df, generated_df], axis=0)
webnovel_df = webnovel_df.reset_index(drop=True)
webnovel_df = webnovel_df.sample(frac=1).reset_index(drop=True)

# Get X and y
X = webnovel_df['text']
y = webnovel_df['target'].values

# Define custom Dataset class
class ChineseDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Load BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained(bert_base_dir, num_labels=2)
tokenizer = BertTokenizer.from_pretrained(bert_base_dir)

# Create datasets
train_dataset = ChineseDataset(X_train.values, y_train, tokenizer)
test_dataset = ChineseDataset(X_test.values, y_test, tokenizer)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training function
def train_model(model, train_loader, epochs, learning_rate, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / len(train_loader)
        acc, prec, rec, f1 = compute_metrics(all_labels, all_preds)
        print(f'\nEpoch {epoch + 1} Training Loss: {avg_loss:.4f}')
        print(f'Training Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}')

    # Save trained model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f'\n✅ Model saved to {MODEL_SAVE_PATH}')



def evaluate_model(model, data_loader, device, save_logits_path=None):
    model.eval()
    all_logits = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1)
            all_logits.append(logits.cpu())
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc, prec, rec, f1 = compute_metrics(all_labels, all_preds)
    print(f'\n✅ Evaluation Results:')
    print(f'Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}')

    if save_logits_path:
        logits_tensor = torch.cat(all_logits, dim=0)
        torch.save(logits_tensor, save_logits_path)
        print(f'Logits saved to {save_logits_path}')


def compute_metrics(true_labels, predicted_labels):
    acc = accuracy_score(true_labels, predicted_labels)
    prec, rec, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted', zero_division=0)
    return acc, prec, rec, f1



# Train the model
train_model(model, train_loader, num_epochs, learning_rate, device)
evaluate_model(model, test_loader, device, save_logits_path=save_logits_path)
# Evaluate the model
