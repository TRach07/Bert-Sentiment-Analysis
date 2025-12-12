import os
import pickle
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models_sentiment_yelp')


def _resolve_path(env_value, candidates):
    """
    Returns the first existing path among candidates,
    or the first candidate if nothing is found.
    """
    if env_value:
        return env_value
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


BERT_MODEL_PATH = _resolve_path(
    os.getenv("BERT_MODEL_PATH"),
    [
        os.path.join(MODELS_DIR, 'distilbert_finetuned'),
    ]
)

TFIDF_VECTORIZER_PATH = _resolve_path(
    os.getenv("TFIDF_VECTORIZER_PATH"),
    [
        os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib'),
    ]
)

LR_MODEL_PATH = _resolve_path(
    os.getenv("LR_MODEL_PATH"),
    [
        os.path.join(MODELS_DIR, 'baseline_logreg.joblib'),
    ]
)

_bert_model = None
_bert_tokenizer = None
_tfidf_vectorizer = None
_lr_model = None
_device = None

def get_device():
    global _device
    if _device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return _device

def load_bert_model():
    global _bert_model, _bert_tokenizer
    
    if _bert_model is None or _bert_tokenizer is None:
        if not os.path.exists(BERT_MODEL_PATH):
            raise FileNotFoundError(f"BERT model not found in {BERT_MODEL_PATH}")
        
        print(f"Loading BERT model from {BERT_MODEL_PATH}...")
        _bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
        _bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
        _bert_model.to(get_device())
        _bert_model.eval()
        print(" BERT model loaded successfully")
    
    return _bert_model, _bert_tokenizer

def load_tfidf_model():
    global _tfidf_vectorizer, _lr_model
    
    if _tfidf_vectorizer is None or _lr_model is None:
        if not os.path.exists(TFIDF_VECTORIZER_PATH):
            raise FileNotFoundError(f"TF-IDF vectorizer not found in {TFIDF_VECTORIZER_PATH}")
        if not os.path.exists(LR_MODEL_PATH):
            raise FileNotFoundError(f"Logistic Regression model not found in {LR_MODEL_PATH}")
        
        print(f"Loading TF-IDF models from {os.path.dirname(TFIDF_VECTORIZER_PATH)}...")
        try:
            if TFIDF_VECTORIZER_PATH.endswith('.joblib'):
                _tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
            else:
                with open(TFIDF_VECTORIZER_PATH, 'rb') as f:
                    _tfidf_vectorizer = pickle.load(f)

            if LR_MODEL_PATH.endswith('.joblib'):
                _lr_model = joblib.load(LR_MODEL_PATH)
            else:
                with open(LR_MODEL_PATH, 'rb') as f:
                    _lr_model = pickle.load(f)
            
            if not hasattr(_tfidf_vectorizer, 'vocabulary_') or _tfidf_vectorizer.vocabulary_ is None:
                raise ValueError("TF-IDF vectorizer is not properly trained.")
            
            print(" TF-IDF models loaded successfully")
        except Exception as e:
            _tfidf_vectorizer = None
            _lr_model = None
            raise Exception(f"Error loading TF-IDF models: {str(e)}")
    
    return _tfidf_vectorizer, _lr_model

def get_bert_model():
    if _bert_model is None:
        load_bert_model()
    return _bert_model, _bert_tokenizer

def get_tfidf_model():
    if _tfidf_vectorizer is None:
        load_tfidf_model()
    return _tfidf_vectorizer, _lr_model

def check_models_available():
    bert_available = os.path.exists(BERT_MODEL_PATH)
    tfidf_available = os.path.exists(TFIDF_VECTORIZER_PATH) and os.path.exists(LR_MODEL_PATH)
    
    return {
        'bert': bert_available,
        'tfidf': tfidf_available,
        'bert_path': BERT_MODEL_PATH if bert_available else None,
        'tfidf_path': os.path.dirname(TFIDF_VECTORIZER_PATH) if tfidf_available else None
    }
