import torch
from torch.utils.data import Dataset

class FusionDataset(Dataset):
    def __init__(self, ids, labels, text_dict, graph_dict, stance_dict, kg_dict):
        self.ids = ids
        self.labels = labels
        self.text_dict = text_dict
        self.graph_dict = graph_dict
        self.stance_dict = stance_dict
        self.kg_dict = kg_dict

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        tid = self.ids[idx]
        return {
            'text': self.text_dict[tid].float(),
            'graph': self.graph_dict[tid].float(),
            'stance': self.stance_dict[tid].float(),
            'kg': self.kg_dict[tid].float(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }
