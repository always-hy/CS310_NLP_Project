import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
from datasets import load_dataset

# Define TextDataset class (unchanged)
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

# Analyze data
def analyse_data(df):
    print(f"Number of rows: {len(df)}")
    print("Maximum length of text:", df['text'].str.len().max())
    print("Minimum length of text:", df['text'].str.len().min())
    print("Average length of text:", df['text'].str.len().mean())
    print("Class distribution:", df['label'].value_counts(normalize=True))

# Load dataset
try:
    # dataset = load_dataset('json', data_files="../dataset/chinese_ood/finance/processed_dataset.json")
    dataset = load_dataset('json', data_files="../dataset/chinese_ood/medicine/processed_dataset.json")

    df = dataset['train'].to_pandas()
except Exception as e:
    print(f"Error loading dataset: {e}")
    df = pd.read_json('../dataset/chinese-ood/finance/processed_dataset.json')  # Fallback for single JSON array
print("Dataset columns:", df.columns)

# Verify and analyze dataset
analyse_data(df)

# Get X and y
X = df['text']
y = df['label'].values

# Initialize tokenizer and model from local paths
local_model_path = '../models/bert-base-chinese'  # Adjust if path differs
tokenizer = BertTokenizer.from_pretrained(local_model_path)
model = BertForSequenceClassification.from_pretrained(local_model_path, num_labels=2)
try:
    model.load_state_dict(torch.load('../results/bert_model_chinese.pt'), strict=True)
except Exception as e:
    print(f"Error loading weights: {e}. Trying with strict=False.")
    model.load_state_dict(torch.load('../results/bert_model_chinese.pt'), strict=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Create dataset and DataLoader
test_dataset = TextDataset(X.values, y, tokenizer, max_length=512)
test_loader = DataLoader(test_dataset, batch_size=8)

# Evaluate model
def evaluate_model(model, test_loader, save_logits_path='../results/bert_chinese_finance_logits_ood.pt'):
    all_preds = []
    all_labels = []
    all_probs = []
    all_logits = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating on out-of-domain finance data'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_logits.append(logits.cpu())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    auroc = roc_auc_score(all_labels, all_probs)

    # Print metrics
    print("Out-of-Domain Finance Test Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  AUROC: {auroc:.4f}")

    # Save logits
    logits_tensor = torch.cat(all_logits, dim=0)
    torch.save(logits_tensor, save_logits_path)
    print(f'✅ Logits saved to {save_logits_path}')

    return accuracy, precision, recall, f1, auroc, all_preds, all_labels

# Run evaluation
accuracy, precision, recall, f1, auroc, all_preds, all_labels = evaluate_model(model, test_loader)

# Visualize metrics
{
    "type": "bar",
    "data": {
        "labels": ["Accuracy", "Precision", "Recall", "F1-Score", "AUROC"],
        "datasets": [{
            "label": "OOD Finance Metrics",
            "data": [accuracy, precision, recall, f1, auroc],
            "backgroundColor": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
            "borderColor": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
            "borderWidth": 1
        }]
    },
    "options": {
        "scales": {
            "y": {
                "beginAtZero": True,
                "max": 1
            }
        },
        "plugins": {
            "legend": {
                "display": True
            },
            "title": {
                "display": True,
                "text": "Out-of-Domain Finance Test Metrics"
            }
        }
    }
}