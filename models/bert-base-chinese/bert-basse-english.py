import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# Assuming english_data.py is in the same directory
from english_data import get_dataloaders

# Load data
train_loader, val_loader, test_loader = get_dataloaders()

# Load pre-trained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        embeddings = batch['embeddings'].to(device)
        labels = batch['labels'].to(device)
        # Reshape embeddings to fit BERT's expected input (batch_size, sequence_length, hidden_size)
        outputs = model(embeddings.unsqueeze(1), labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')

# Validation
model.eval()
val_preds = []
val_labels = []
with torch.no_grad():
    for batch in val_loader:
        embeddings = batch['embeddings'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(embeddings.unsqueeze(1))
        preds = torch.argmax(outputs.logits, dim=1)
        val_preds.extend(preds.cpu().tolist())
        val_labels.extend(labels.cpu().tolist())
print(f'Validation Accuracy: {accuracy_score(val_labels, val_preds)}')

# Save model
model.save_pretrained('fine_tuned_bert_english')