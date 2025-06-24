import pandas as pd
import torch
from torch_geometric.data import Data
import spacy
from collections import defaultdict
from sentence_transformers import SentenceTransformer

def build_kg_features(tweets_csv, conceptnet_csv):
    nlp = spacy.load("en_core_web_sm")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    tweets_df = pd.read_csv(tweets_csv)
    conceptnet_df = pd.read_csv(conceptnet_csv)
    allowed_relations = {'RelatedTo', 'IsA', 'Causes', 'HasProperty', 'PartOf', 'CapableOf', 'UsedFor'}
    conceptnet_df = conceptnet_df[conceptnet_df['relation'].isin(allowed_relations)]

    conceptnet_df['subject'] = conceptnet_df['subject'].apply(lambda x: str(x).split('/')[-1])
    conceptnet_df['object'] = conceptnet_df['object'].apply(lambda x: str(x).split('/')[-1])

    keyword_index = defaultdict(list)
    for _, row in conceptnet_df.iterrows():
        keyword_index[row['subject']].append((row['subject'], row['relation'], row['object']))
        keyword_index[row['object']].append((row['subject'], row['relation'], row['object']))

    relation2id = {rel: idx for idx, rel in enumerate(sorted(allowed_relations))}
    
    kg_features_dict = {}
    node_cache = {}

    for _, row in tweets_df.iterrows():
        tweet_id = str(row['id'])
        text = row['text'].lower()
        doc = nlp(text)
        keywords = {token.lemma_ for token in doc if token.pos_ in {'NOUN', 'VERB', 'ADJ'} and not token.is_stop}

        triples = set()
        for word in keywords:
            triples.update(keyword_index.get(word, []))

        nodes = set()
        for h, _, t in triples:
            nodes.add(h)
            nodes.add(t)

        node_list = sorted(nodes)
        node_embs = sentence_model.encode(node_list, convert_to_tensor=True)
        node2vec = {node: emb for node, emb in zip(node_list, node_embs)}

        if triples:
            edge_index = torch.tensor([[node_list.index(h), node_list.index(t)] for h, _, t in triples], dtype=torch.long).T
            edge_type = torch.tensor([relation2id[r] for _, r, _ in triples], dtype=torch.long)
            node_features = torch.stack([node2vec[n] for n in node_list])
            kg_features_dict[tweet_id] = node_features.mean(dim=0)

    return kg_features_dict
