import os
import ast
import pandas as pd
import torch
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer

node_encoder = SentenceTransformer('all-MiniLM-L6-v2')

def parse_tree_file(file_path):
    edges = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '->' in line:
                src, tgt = line.strip().split('->')
                src_info = ast.literal_eval(src.strip())
                tgt_info = ast.literal_eval(tgt.strip())
                src_node = f"{src_info[0]}_{src_info[1]}"
                tgt_node = f"{tgt_info[0]}_{tgt_info[1]}"
                edges.append((src_node, tgt_node))
    return edges

def convert_to_pyg(edges, label):
    nodes = list(set([n for edge in edges for n in edge]))
    node2idx = {node: idx for idx, node in enumerate(nodes)}
    edge_index = torch.tensor([[node2idx[src], node2idx[tgt]] for src, tgt in edges], dtype=torch.long).t().contiguous()
    x = torch.randn(len(nodes), 384)  # Random for now, replaceable with actual encodings
    edge_type = torch.zeros(edge_index.size(1), dtype=torch.long)
    y = torch.tensor([label], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y)

def build_graph_features(tree_folder, label_csv_path):
    df = pd.read_csv(label_csv_path)
    label_dict = dict(zip(df['id'].astype(str), df['label_num']))
    graph_features_dict = {}

    for fname in os.listdir(tree_folder):
        if fname.endswith(".txt"):
            tweet_id = fname.replace(".txt", "")
            if tweet_id in label_dict:
                edges = parse_tree_file(os.path.join(tree_folder, fname))
                data = convert_to_pyg(edges, label_dict[tweet_id])
                graph_features_dict[tweet_id] = data.x.mean(dim=0)  # Use mean node feature as representation

    return graph_features_dict
