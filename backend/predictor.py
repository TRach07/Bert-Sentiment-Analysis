import torch
import torch.nn.functional as F
from model_loader import get_bert_model, get_tfidf_model, get_device
from utils import clean_text, MIN_CLEAN_LENGTH

LABEL_MAPPING = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
MAX_SEQ_LENGTH = 512

def predict_bert(text, model=None, tokenizer=None):
    if model is None or tokenizer is None:
        model, tokenizer = get_bert_model()
    
    device = get_device()
    text_cleaned = clean_text(text)

    if not text_cleaned or len(text_cleaned) <= MIN_CLEAN_LENGTH:
        raise ValueError("Le texte est vide ou trop court après nettoyage.")
    
    encoding = tokenizer(
        text_cleaned,
        truncation=True,
        padding='max_length',
        max_length=MAX_SEQ_LENGTH,
        return_tensors='pt'
    )
    
    encoding = {k: v.to(device) for k, v in encoding.items()}
    
    model.eval()
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return {
        'sentiment': LABEL_MAPPING[predicted_class],
        'sentiment_id': predicted_class,
        'confidence': float(confidence),
        'probabilities': {
            LABEL_MAPPING[i]: float(probabilities[0][i].item())
            for i in range(len(LABEL_MAPPING))
        }
    }

def predict_bert_batch(texts, model=None, tokenizer=None, batch_size=16):
    if model is None or tokenizer is None:
        model, tokenizer = get_bert_model()
    
    device = get_device()
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_texts_cleaned = []
        for text in batch_texts:
            cleaned = clean_text(str(text))
            if cleaned and len(cleaned) > MIN_CLEAN_LENGTH:
                batch_texts_cleaned.append(cleaned)
        
        if not batch_texts_cleaned:
            continue
        
        encoding = tokenizer(
            batch_texts_cleaned,
            truncation=True,
            padding='max_length',
            max_length=MAX_SEQ_LENGTH,
            return_tensors='pt'
        )
        
        encoding = {k: v.to(device) for k, v in encoding.items()}
        
        model.eval()
        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            predicted_classes = torch.argmax(probabilities, dim=-1).cpu().numpy()
            confidences = probabilities.max(dim=-1)[0].cpu().numpy()
            all_probs = probabilities.cpu().numpy()
        
        for j, (pred_class, conf) in enumerate(zip(predicted_classes, confidences)):
            results.append({
                'sentiment': LABEL_MAPPING[pred_class],
                'sentiment_id': int(pred_class),
                'confidence': float(conf),
                'probabilities': {
                    LABEL_MAPPING[k]: float(all_probs[j][k])
                    for k in range(len(LABEL_MAPPING))
                }
            })
    
    return results

def predict_tfidf(text, vectorizer=None, model=None):
    if vectorizer is None or model is None:
        vectorizer, model = get_tfidf_model()
    
    if not hasattr(vectorizer, 'vocabulary_') or vectorizer.vocabulary_ is None:
        raise ValueError("TF-IDF vectorizer is not properly trained.")
    
    text_cleaned = clean_text(text)
    
    if not text_cleaned or len(text_cleaned) <= MIN_CLEAN_LENGTH:
        raise ValueError("Text is empty or too short after cleaning.")
    
    try:
        text_vectorized = vectorizer.transform([text_cleaned])
    except AttributeError as e:
        if "idf" in str(e).lower() or "not fitted" in str(e).lower():
            raise ValueError(
                f"TF-IDF vectorizer is not properly trained. "
                f"Error: {str(e)}. "
                f"Make sure the vectorizer was saved after calling fit()."
            )
        raise ValueError(f"Error during TF-IDF vectorization: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error during TF-IDF vectorization: {str(e)}")
    
    predicted_class = model.predict(text_vectorized)[0]
    probabilities = model.predict_proba(text_vectorized)[0]
    confidence = probabilities[predicted_class]
    
    if predicted_class not in LABEL_MAPPING:
        if predicted_class == 0:
            sentiment = 'Negative'
        elif predicted_class == 2:
            sentiment = 'Positive'
        else:
            sentiment = LABEL_MAPPING.get(predicted_class, f'Label {predicted_class}')
    else:
        sentiment = LABEL_MAPPING[predicted_class]
    
    num_classes = len(probabilities)
    prob_dict = {}
    model_classes = model.classes_ if hasattr(model, 'classes_') else list(range(num_classes))
    
    for i, class_id in enumerate(model_classes):
        if class_id in LABEL_MAPPING:
            prob_dict[LABEL_MAPPING[class_id]] = float(probabilities[i])
        else:
            prob_dict[f'Label {class_id}'] = float(probabilities[i])
    
    if 'Neutral' not in prob_dict:
        prob_dict['Neutral'] = 0.0
    
    return {
        'sentiment': sentiment,
        'sentiment_id': int(predicted_class),
        'confidence': float(confidence),
        'probabilities': prob_dict
    }

def predict_tfidf_batch(texts, vectorizer=None, model=None):
    if vectorizer is None or model is None:
        vectorizer, model = get_tfidf_model()
    
    texts_cleaned = [clean_text(str(text)) for text in texts]
    texts_vectorized = vectorizer.transform(texts_cleaned)
    
    predicted_classes = model.predict(texts_vectorized)
    probabilities = model.predict_proba(texts_vectorized)
    confidences = probabilities.max(axis=1)
    
    results = []
    for pred_class, conf, probs in zip(predicted_classes, confidences, probabilities):
        results.append({
            'sentiment': LABEL_MAPPING[pred_class],
            'sentiment_id': int(pred_class),
            'confidence': float(conf),
            'probabilities': {
                LABEL_MAPPING[i]: float(probs[i])
                for i in range(len(LABEL_MAPPING))
            }
        })
    
    return results
