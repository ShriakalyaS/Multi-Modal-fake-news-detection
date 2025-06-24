import torch
import torch.nn as nn

class DETECTORModel(nn.Module):
    def __init__(self, text_dim=384, graph_dim=64, stance_dim=3, kg_dim=64, hidden_dim=256, num_classes=4):
        super(DETECTORModel, self).__init__()

        self.text_branch = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.graph_branch = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.stance_branch = nn.Sequential(
            nn.Linear(stance_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.kg_branch = nn.Sequential(
            nn.Linear(kg_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.gate = nn.Linear(hidden_dim * 4, hidden_dim * 4)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, text, graph, stance, kg):
        t = self.text_branch(text)
        g = self.graph_branch(graph)
        s = self.stance_branch(stance)
        k = self.kg_branch(kg)

        fused = torch.cat([t, g, s, k], dim=1)
        gated = torch.sigmoid(self.gate(fused))
        fused = fused * gated
        out = self.fusion(fused)
        return self.classifier(out)
