# Import libraries
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

# Function to analyze data
def analyse_data(df):
    print(f"Number of rows: {len(df)}")
    print("Maximum length of text:", df['text'].str.len().max())
    print("Minimum length of text:", df['text'].str.len().min())
    print("Average length of text:", df['text'].str.len().mean())

# Load data
dir_name = '../dataset/face2/'
human_df = pd.read_csv(dir_name + 'generated/generated_data.csv')
generated_df = pd.read_csv(dir_name + 'human/human_data.csv')

# Combine and shuffle
text_df = pd.concat([human_df, generated_df], axis=0)
text_df = text_df.reset_index(drop=True)
text_df = text_df.sample(frac=1).reset_index(drop=True)

# Analyze data
analyse_data(text_df)

# Get X and y
X = text_df['text']
y = text_df['target'].values

# Define custom Dataset class
class textDataset(Dataset):
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize tokenizer from the same checkpoint directory
tokenizer = BertTokenizer.from_pretrained('../models/bert-base-chinese')

# Create datasets
train_dataset = textDataset(X_train.values, y_train, tokenizer)
test_dataset = textDataset(X_test.values, y_test, tokenizer)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# Load BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained(
    '../models/bert-base-chinese',
    num_labels=2
)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training function with enhanced metrics, logits saving, and model saving
def train_model(model, train_loader, test_loader, epochs=10, save_model_path='../results/bert_model.pt', save_logits_path='../results/bert_logits.pt'):
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