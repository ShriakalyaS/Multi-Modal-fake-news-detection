import torch
from torch.utils.data import DataLoader
from fusion_dataset import FusionDataset
from model import DETECTORModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# ----------- Load feature dicts (replace with your actual paths) -------------
# Assuming these dicts are loaded before calling inference
# text_features_dict = ...
# graph_features_dict = ...
# stance_features_dict = ...
# kg_features_dict = ...

# ----------- Load your test ids and labels -------------
# test_ids = ...
# test_labels = ...

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = DETECTORModel().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))  # Path to saved model
model.eval()

# Create test dataset and loader
test_dataset = FusionDataset(test_ids, test_labels, text_features_dict, graph_features_dict, stance_features_dict, kg_features_dict)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ----------- Inference -------------
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        outputs = model(
            batch['text'].to(device),
            batch['graph'].to(device),
            batch['stance'].to(device),
            batch['kg'].to(device)
        )
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch['label'].cpu().numpy())

# ----------- Evaluation -------------
acc = accuracy_score(all_labels, all_preds)
print(f"\nTest Accuracy: {acc:.4f}\n")
print("Classification Report:\n")
print(classification_report(all_labels, all_preds))

# ----------- Confusion Matrix -------------
cm = confusion_matrix(all_labels, all_preds)
class_names = [f"Class {i}" for i in range(4)]

plt.figure(figsize=(4, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
