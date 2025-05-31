import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import json
import os

# Function to analyze data
def analyse_data(df):
    print(f"Number of rows: {len(df)}")
    print("Maximum length of text:", df['text'].str.len().max())
    print("Minimum length of text:", df['text'].str.len().min())
    print("Average length of text:", df['text'].str.len().mean())

# Verify local model directory
model_dir = '../models/bert-base-uncased'
if not os.path.exists(model_dir):
    raise FileNotFoundError(f"Local model directory {model_dir} does not exist. Please ensure the tokenizer and model files are downloaded.")
if not os.path.exists(os.path.join(model_dir, 'vocab.txt')):
    raise FileNotFoundError(f"Vocabulary file (vocab.txt) not found in {model_dir}. Please ensure the tokenizer files are present.")

# Load data
dir_name = '../dataset/english_dataset/'
with open(dir_name + 'eng_training.json', 'r') as f:
    train_data = json.load(f)
with open(dir_name + 'eng_testing.json', 'r') as f:
    test_data = json.load(f)

# Convert to DataFrame
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

# Combine train and test for analysis
text_df = pd.concat([train_df, test_df], axis=0)
text_df = text_df.reset_index(drop=True)
text_df = text_df.sample(frac=1).reset_index(drop=True)

# Analyze data
analyse_data(text_df)

# Get X and y for train and test
X_train = train_df['text']
y_train = train_df['label'].values
X_test = test_df['text']
y_test = test_df['label'].values

# Define custom Dataset class
class TextDataset(Dataset):
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

# Initialize tokenizer and model from local directory
try:
    tokenizer = BertTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=2,
        local_files_only=True
    )
except Exception as e:
    raise Exception(f"Failed to load tokenizer or model from {model_dir}. Ensure all necessary files are present. Error: {str(e)}")

# Create datasets
train_dataset = TextDataset(X_train.values, y_train, tokenizer)
test_dataset = TextDataset(X_test.values, y_test, tokenizer)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training function with enhanced metrics, logits saving, and model saving
def train_model(model, train_loader, test_loader, epochs=10, save_model_path='../results/bert_model_english.pt', save_logits_path='../results/bert_logits_english.pt'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
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

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}')

        # Evaluation with additional metrics and logits saving
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []  # For AUROC
        all_logits = []  # For saving logits

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)
                probs = torch.softmax(logits, dim=1)[:, 1]  # Probability for positive class

                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_logits.append(logits.cpu())  # Collect logits

        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
        f1 = f1_score(all_labels, all_preds, average='binary')
        auroc = roc_auc_score(all_labels, all_probs)

        # Print metrics
        print(f'Epoch {epoch + 1}, Test Metrics:')
        print(f'  Accuracy: {accuracy:.4f}')
        print(f'  Precision: {precision:.4f}')
        print(f'  Recall: {recall:.4f}')
        print(f'  F1-Score: {f1:.4f}')
        print(f'  AUROC: {auroc:.4f}')

        # Save logits
        logits_tensor = torch.cat(all_logits, dim=0)
        torch.save(logits_tensor, save_logits_path)
        print(f'✅ Logits saved to {save_logits_path}')

    # Save the model after training
    torch.save(model.state_dict(), save_model_path)
    print(f'✅ Model saved to {save_model_path}')

# Train the model
train_model(model, train_loader, test_loader)