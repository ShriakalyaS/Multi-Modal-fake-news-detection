import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
from model import DETECTORModel
from utils import FusionDataset

# Assuming these dicts and label lists are prepared from preprocessing
from features_dicts import text_features_dict, graph_features_dict, stance_features_dict, kg_features_dict
from features_dicts import train_ids, train_labels, val_ids, val_labels, test_ids, test_labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DETECTORModel().to(device)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.05)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

train_dataset = FusionDataset(train_ids, train_labels)
val_dataset = FusionDataset(val_ids, val_labels)
test_dataset = FusionDataset(test_ids, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

best_val_acc = 0
patience = 5
epochs_no_improve = 0
best_model_state = None

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

num_epochs = 15

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch['text'].to(device),
                    batch['graph'].to(device),
                    batch['stance'].to(device),
                    batch['kg'].to(device))
        labels = batch['label'].to(device)
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(out, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for batch in val_loader:
            out = model(batch['text'].to(device),
                        batch['graph'].to(device),
                        batch['stance'].to(device),
                        batch['kg'].to(device))
            labels = batch['label'].to(device)
            loss = criterion(out, labels)
            val_loss += loss.item()
            preds = torch.argmax(out, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1:02}: Train Loss = {train_loss:.4f}, Acc = {train_acc:.4f} | Val Loss = {val_loss:.4f}, Acc = {val_acc:.4f}")

    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict()
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Load best model
if best_model_state:
    model.load_state_dict(best_model_state)

# Final Test Evaluation
model.eval()
test_preds, test_labels_all = [], []
with torch.no_grad():
    for batch in test_loader:
        out = model(batch['text'].to(device),
                    batch['graph'].to(device),
                    batch['stance'].to(device),
                    batch['kg'].to(device))
        labels = batch['label'].to(device)
        preds = torch.argmax(out, dim=1)
        test_preds.extend(preds.cpu().numpy())
        test_labels_all.extend(labels.cpu().numpy())

test_acc = accuracy_score(test_labels_all, test_preds)
print(f"\nFinal Test Accuracy: {test_acc:.4f}\n")
print("Classification Report:")
print(classification_report(test_labels_all, test_preds))

# Loss and Accuracy Plots
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Acc")
plt.plot(val_accuracies, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
