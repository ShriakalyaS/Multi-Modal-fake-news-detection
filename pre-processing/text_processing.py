import pandas as pd
from sentence_transformers import SentenceTransformer

def build_text_features(text_csv_path):
    df = pd.read_csv(text_csv_path)
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

    tweet_ids = df['id'].astype(str).tolist()
    tweet_texts = df['text'].tolist()
    text_embeddings = sentence_model.encode(tweet_texts, convert_to_tensor=True)

    text_features_dict = {tweet_id: emb for tweet_id, emb in zip(tweet_ids, text_embeddings)}
    return text_features_dict
