# Import libraries
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Function to analyze data
def analyse_data(df):
    print(f"Number of rows: {len(df)}")
    print("Maximum length of text:", df['text'].str.len().max())
    print("Minimum length of text:", df['text'].str.len().min())
    print("Average length of text:", df['text'].str.len().mean())

# Load data
dir_name = '../../dataset/face2/'
human_df = pd.read_csv(dir_name + 'generated/generated_webnovel_data_200.csv')
generated_df = pd.read_csv(dir_name + 'human/human_webnovel_data_200.csv')

# Combine and shuffle
webnovel_df = pd.concat([human_df, generated_df], axis=0)
webnovel_df = webnovel_df.reset_index(drop=True)
webnovel_df = webnovel_df.sample(frac=1).reset_index(drop=True)

# Analyze data
analyse_data(webnovel_df)

# Get X and y
X = webnovel_df['text']
y = webnovel_df['target'].values

# Define custom Dataset class
class WebnovelDataset(Dataset):
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

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# Create datasets
train_dataset = WebnovelDataset(X_train.values, y_train, tokenizer)
test_dataset = WebnovelDataset(X_test.values, y_test, tokenizer)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# Load BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training function
def train_model(model, train_loader, test_loader, epochs=3):
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

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f'Epoch {epoch + 1}, Test Accuracy: {accuracy:.4f}')

# Train the model
train_model(model, train_loader, test_loader)