import re
import pandas as pd

# Aligné sur le nettoyage du notebook
MIN_CLEAN_LENGTH = 5

def clean_text(text):
    """
    Text cleaning inspired by the training notebook.
    - lowercase
    - remove URLs/HTML
    - expand contractions
    - normalize spaces and reduce repetitions
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    if not text:
        return ""

    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'m", " am", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?\']+", '', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def validate_csv_columns(df, required_columns=['text', 'avis', 'review', 'comment']):
    df_columns_lower = [col.lower() for col in df.columns]
    
    for req_col in required_columns:
        if req_col.lower() in df_columns_lower:
            idx = df_columns_lower.index(req_col.lower())
            return df.columns[idx]
    
    return None

def prepare_texts_for_prediction(texts, return_indices=False):
    if hasattr(texts, 'tolist'):
        texts = texts.tolist()
    
    prepared = []
    valid_indices = []

    for idx, text in enumerate(texts):
        if pd.isna(text):
            continue
        cleaned = clean_text(text)
        if cleaned and len(cleaned) > MIN_CLEAN_LENGTH:
            prepared.append(cleaned)
            valid_indices.append(idx)

    if return_indices:
        return prepared, valid_indices
    return prepared
