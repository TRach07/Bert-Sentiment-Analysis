from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import os
import uuid
from datetime import datetime
import traceback

from model_loader import check_models_available, load_bert_model, load_tfidf_model
from predictor import predict_bert, predict_bert_batch, predict_tfidf, predict_tfidf_batch
from utils import validate_csv_columns, prepare_texts_for_prediction

app = FastAPI(title="Sentiment Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    print(" Starting API...")
    models_status = check_models_available()
    print(f" Models status: {models_status}")
    
    if models_status['bert']:
        try:
            load_bert_model()
        except Exception as e:
            print(f" Error loading BERT: {e}")
    
    if models_status['tfidf']:
        try:
            load_tfidf_model()
        except Exception as e:
            print(f" Error loading TF-IDF: {e}")
    
    print(" API ready!")

class TextInput(BaseModel):
    text: str
    model: Optional[str] = "bert"

class BatchTextInput(BaseModel):
    texts: List[str]
    model: Optional[str] = "bert"

@app.get("/")
async def root():
    return {
        "message": "Sentiment Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "models": "/api/models/status",
            "predict_single": "/api/predict/single",
            "predict_batch": "/api/predict/batch",
            "upload": "/api/upload"
        }
    }

@app.get("/health")
async def health_check():
    models_status = check_models_available()
    return {
        "status": "healthy",
        "models": models_status,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/models/status")
async def get_models_status():
    return check_models_available()

@app.post("/api/predict/single")
async def predict_single(input_data: TextInput):
    model_type = input_data.model.lower()
    
    try:
        if model_type == "bert":
            if not check_models_available()['bert']:
                raise HTTPException(status_code=404, detail="BERT model not available")
            result = predict_bert(input_data.text)
        elif model_type == "tfidf":
            if not check_models_available()['tfidf']:
                raise HTTPException(status_code=404, detail="TF-IDF model not available")
            result = predict_tfidf(input_data.text)
        else:
            raise HTTPException(status_code=400, detail="Model must be 'bert' or 'tfidf'")
        
        return {
            "text": input_data.text,
            "model": model_type,
            "prediction": result
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during prediction with {model_type}: {str(e)}")

@app.post("/api/predict/batch")
async def predict_batch(input_data: BatchTextInput):
    model_type = input_data.model.lower()

    texts = prepare_texts_for_prediction(input_data.texts)
    if not texts:
        raise HTTPException(status_code=400, detail="No valid text after cleaning.")
    
    try:
        if model_type == "bert":
            if not check_models_available()['bert']:
                raise HTTPException(status_code=404, detail="BERT model not available")
            results = predict_bert_batch(texts)
        elif model_type == "tfidf":
            if not check_models_available()['tfidf']:
                raise HTTPException(status_code=404, detail="TF-IDF model not available")
            results = predict_tfidf_batch(texts)
        else:
            raise HTTPException(status_code=400, detail="Model must be 'bert' or 'tfidf'")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return {
        "model": model_type,
        "count": len(results),
        "predictions": results
    }

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    model: str = "bert",
    text_column: Optional[str] = None
):
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or Excel.")
    
    file_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[1]
    file_path = os.path.join(UPLOADS_DIR, f"{file_id}{file_ext}")
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    try:
        if file_ext == '.csv':
            try:
                df = pd.read_csv(file_path, encoding='utf-8', quotechar='"')
            except Exception as csv_error:
                try:
                    df = pd.read_csv(file_path, encoding='utf-8', sep=',', quotechar='"', engine='python')
                except Exception:
                    raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(csv_error)}")
        else:
            df = pd.read_excel(file_path)
        
        if text_column:
            if text_column not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{text_column}' not found")
            text_col = text_column
        else:
            text_col = validate_csv_columns(df)
            if text_col is None:
                raise HTTPException(
                    status_code=400,
                    detail="No text column found. Available columns: " + ", ".join(df.columns)
                )
        
        texts, valid_indices = prepare_texts_for_prediction(df[text_col], return_indices=True)
        
        if len(texts) == 0:
            raise HTTPException(status_code=400, detail="No valid text found after cleaning")
        
        df_valid = df.iloc[valid_indices].reset_index(drop=True)
        
        model_type = model.lower()
        if model_type == "bert":
            if not check_models_available()['bert']:
                raise HTTPException(status_code=404, detail="BERT model not available")
            predictions = predict_bert_batch(texts)
        elif model_type == "tfidf":
            if not check_models_available()['tfidf']:
                raise HTTPException(status_code=404, detail="TF-IDF model not available")
            predictions = predict_tfidf_batch(texts)
        else:
            raise HTTPException(status_code=400, detail="Model must be 'bert' or 'tfidf'")
        
        df_valid['sentiment'] = [p['sentiment'] for p in predictions]
        df_valid['sentiment_id'] = [p['sentiment_id'] for p in predictions]
        df_valid['confidence'] = [p['confidence'] for p in predictions]
        
        result_file = os.path.join(RESULTS_DIR, f"{file_id}_results.csv")
        df_valid.to_csv(result_file, index=False, encoding='utf-8')
        
        stats = {
            'total': len(predictions),
            'negative': sum(1 for p in predictions if p['sentiment'] == 'Negative'),
            'neutral': sum(1 for p in predictions if p['sentiment'] == 'Neutral'),
            'positive': sum(1 for p in predictions if p['sentiment'] == 'Positive'),
            'avg_confidence': sum(p['confidence'] for p in predictions) / len(predictions)
        }
        
        return {
            "file_id": file_id,
            "model": model_type,
            "stats": stats,
            "results_file": f"/api/results/{file_id}/download",
            "message": f"{len(predictions)} reviews analyzed successfully"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during processing: {str(e)}")
    
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/api/results/{file_id}/download")
async def download_results(file_id: str):
    result_file = os.path.join(RESULTS_DIR, f"{file_id}_results.csv")
    
    if not os.path.exists(result_file):
        raise HTTPException(status_code=404, detail="Results file not found")
    
    return FileResponse(
        result_file,
        media_type='text/csv',
        filename=f"sentiment_analysis_results_{file_id}.csv"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
