import pandas as pd
import torch

def build_stance_features(stance_csv_path):
    df = pd.read_csv(stance_csv_path)
    stance_dict = {}

    for _, row in df.iterrows():
        tweet_id = str(row['id'])
        agree, disagree, neutral = row['agree'], row['disagree'], row['neutral']
        total = agree + disagree + neutral
        if total == 0:
            stance_vec = torch.tensor([0.0, 0.0, 0.0])
        else:
            stance_vec = torch.tensor([agree/total, disagree/total, neutral/total])
        stance_dict[tweet_id] = stance_vec

    return stance_dict
