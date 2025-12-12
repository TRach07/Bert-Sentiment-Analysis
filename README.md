# Sentiment Analysis with BERT

Complete sentiment analysis system capable of classifying customer reviews into 3 categories: **positive**, **neutral**, **negative**. The project offers two approaches: a fine-tuned BERT/DistilBERT model for maximum accuracy, and a TF-IDF + Logistic Regression model for optimal speed.

## Overview

This project is a full-stack sentiment analysis application that combines:
- **Backend**: REST API with FastAPI for data processing
- **Frontend**: Interactive web interface with real-time visualizations
- **ML Models**: Two machine learning models for different use cases

### Use Cases

- Customer review analysis (e-commerce, restaurants, services)
- Social media monitoring
- Product feedback analysis
- Market research and studies
- Automated customer support

## Features

### Machine Learning Models

1. **BERT/DistilBERT Fine-tuned**
   - Pre-trained transformer model fine-tuned on a sentiment dataset
   - High accuracy thanks to contextual understanding
   - Automatic GPU/CPU support
   - Batch processing for bulk processing

2. **TF-IDF + Logistic Regression**
   - Fast and lightweight classic model
   - Ideal for real-time processing
   - Lower resource requirements
   - Good speed/accuracy trade-off

### REST API (FastAPI)

- **Single prediction**: Real-time text analysis
- **Batch prediction**: Process multiple texts simultaneously
- **File upload**: CSV and Excel support with automatic column detection
- **Results download**: Export analyses as CSV
- **Health check**: API and models status verification
- **Interactive documentation**: Integrated Swagger UI and ReDoc

### Web Interface

- **File upload**: Drag-and-drop interface for CSV/Excel
- **Visualizations**: Pie and bar charts for sentiment distribution
- **Interactive table**: Detailed display with search and filtering
- **Real-time analysis**: Instant testing of individual texts
- **Statistics**: Counters and confidence metrics
- **Export**: Download analyzed results

### Data Processing

- **Automatic cleaning**: Remove URLs, HTML, text normalization
- **Encoding management**: UTF-8 and other encodings support
- **Validation**: Automatic text column detection
- **Error handling**: Clear and informative error messages

## Project Structure

```
bert-sentiment-analysis/
├── backend/                          # FastAPI API
│   ├── app.py                       # Main FastAPI application
│   ├── model_loader.py              # Model loading and management
│   ├── predictor.py                 # Prediction logic (BERT & TF-IDF)
│   ├── utils.py                     # Utility functions (cleaning, validation)
│   └── requirements.txt             # Python backend dependencies
│
├── frontend/                        # Web interface
│   ├── index.html                   # Main page
│   └── static/
│       ├── css/
│       │   └── style.css           # Custom styles
│       └── js/
│           └── app.js               # Frontend JavaScript logic
│
├── models_sentiment_yelp/           # Trained models (download from Google Drive after running notebook)
│   ├── distilbert_finetuned/        # Fine-tuned DistilBERT model
│   │   ├── config.json
│   │   ├── model.safetensors (or pytorch_model.bin)
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── special_tokens_map.json
│   │   └── vocab.txt
│   ├── tfidf_vectorizer.joblib      # TF-IDF vectorizer
│   └── baseline_logreg.joblib       # Logistic Regression model
│
├── uploads/                         # Uploaded files (temporary)
├── results/                         # Generated results
│
├── Sentiment_Analysis_BERT_vs_TFIDF.ipynb  # Training notebook
│
├── start_backend.bat               # Backend startup script (Windows)
├── start_backend.sh                # Backend startup script (Linux/Mac)
├── start_all.bat                   # Complete startup script (Windows)
├── requirements.txt                # Global dependencies (notebook)
└── README.md                       # This file
```

## Technologies Used

### Backend
- **FastAPI**: Modern and fast web framework
- **PyTorch**: Deep learning library
- **Transformers (Hugging Face)**: Pre-trained BERT models
- **scikit-learn**: Classical machine learning (TF-IDF, Logistic Regression)
- **Pandas**: Data manipulation
- **Uvicorn**: ASGI server

### Frontend
- **HTML5/CSS3**: Structure and style
- **JavaScript (Vanilla)**: Interactive logic
- **Bootstrap 5**: Responsive CSS framework
- **Chart.js**: Chart visualizations

### Machine Learning
- **BERT/DistilBERT**: Transformer architecture for NLP
- **TF-IDF**: Text vectorization
- **Logistic Regression**: Linear classifier

## Installation

### Prerequisites

- **Python** 3.8 or higher
- **pip** (Python package manager)
- **Git** (optional, for cloning the project)
- **NVIDIA GPU** (optional, for BERT acceleration)

### 1. Clone the project (if applicable)

```bash
git clone <repository-url>
cd bert-sentiment-analysis
```

### 2. Install backend dependencies

```bash
cd backend
pip install -r requirements.txt
```

**Main dependencies:**
- `fastapi==0.104.1`
- `uvicorn[standard]==0.24.0`
- `torch>=2.1.0`
- `transformers>=4.35.0`
- `scikit-learn>=1.5.1`
- `pandas>=2.2.0`
- `numpy>=1.24.3`
- `openpyxl>=3.1.2` (for Excel)
- `python-multipart==0.0.6` (for file uploads)

### 3. Download the Models

**Important**: After running the complete notebook `Sentiment_Analysis_BERT_vs_TFIDF.ipynb` in Google Colab, the models will be saved to a folder (typically named `models_sentiment_yelp` or similar, depending on your notebook configuration). You need to download this folder from Google Drive and place it at the root of your project.

#### Step 1: Run the Notebook in Google Colab

Execute all cells in `Sentiment_Analysis_BERT_vs_TFIDF.ipynb` to train and save the models. The notebook will save the models to a folder (check the notebook's save directory configuration).

#### Step 2: Download Models from Google Drive

**Option A: Via Google Drive (Recommended)**

In your Colab notebook, copy the models folder to Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy models to Drive (replace 'models_sentiment_yelp' with your actual folder name)
!cp -r models_sentiment_yelp /content/drive/MyDrive/bert-sentiment-analysis/
```

Then download the folder from Google Drive to your local project root. **Rename the downloaded folder to `models_sentiment_yelp`** if it has a different name.

**Option B: Direct Download**

```python
from google.colab import files
import shutil

# Create an archive (replace 'models_sentiment_yelp' with your actual folder name)
shutil.make_archive('models_sentiment_yelp', 'zip', 'models_sentiment_yelp')
files.download('models_sentiment_yelp.zip')
```

Extract the archive in your project root and **rename the folder to `models_sentiment_yelp`** if needed.

#### Step 3: Verify Model Structure

Ensure you have the following structure at the project root:

```
models_sentiment_yelp/
├── distilbert_finetuned/
│   ├── config.json
│   ├── model.safetensors (or pytorch_model.bin)
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── vocab.txt
├── tfidf_vectorizer.joblib
└── baseline_logreg.joblib
```

**Note**: The folder name `models_sentiment_yelp` is what the application expects. If your notebook saves models with a different name, rename the downloaded folder to `models_sentiment_yelp`.

#### Alternative: Custom Model Paths

If your models are in a different location, set environment variables:
```bash
export BERT_MODEL_PATH=/path/to/distilbert_finetuned
export TFIDF_VECTORIZER_PATH=/path/to/tfidf_vectorizer.joblib
export LR_MODEL_PATH=/path/to/baseline_logreg.joblib
```

## Usage

### Quick Start

#### Windows

**Option 1: Automatic Script**
```bash
start_all.bat
```

**Option 2: Manual Start**

Backend:
```bash
start_backend.bat
```

Frontend: Open `frontend/index.html` in your browser.

#### Linux/Mac

**Backend:**
```bash
cd backend
python -m uvicorn app:app --reload --port 8000
```

Or with the script:
```bash
chmod +x start_backend.sh
./start_backend.sh
```

**Frontend:**
```bash
cd frontend
python -m http.server 8080
```

Then open `http://localhost:8080` in your browser.

### Using the Web Interface

1. **Open the interface**: `http://localhost:8080` (or open `frontend/index.html` directly)

2. **Upload a file**:
   - Click on "Select a CSV or Excel file"
   - Choose a CSV or Excel file
   - Select the model (BERT or TF-IDF)
   - Optional: Specify the text column name
   - Click on "Analyze Reviews"

3. **View results**:
   - Global statistics (total, positive, neutral, negative)
   - Interactive charts (pie and bar charts)
   - Detailed table with search
   - Download CSV with results

4. **Real-time analysis**:
   - Enter text in the "Real-time Analysis" section
   - Select the model
   - Click on "Analyze"
   - View sentiment, confidence, and probabilities

### Using the API Directly

#### 1. Check Status

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "models": {
    "bert": true,
    "tfidf": true
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

#### 2. Single Text Prediction

```bash
curl -X POST "http://localhost:8000/api/predict/single" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This product is absolutely amazing!",
    "model": "bert"
  }'
```

Response:
```json
{
  "text": "This product is absolutely amazing!",
  "model": "bert",
  "prediction": {
    "sentiment": "Positive",
    "sentiment_id": 2,
    "confidence": 0.95,
    "probabilities": {
      "Negative": 0.02,
      "Neutral": 0.03,
      "Positive": 0.95
    }
  }
}
```

#### 3. Batch Prediction

```bash
curl -X POST "http://localhost:8000/api/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "I love this product!",
      "It's okay, nothing special.",
      "I don't recommend it at all."
    ],
    "model": "bert"
  }'
```

#### 4. Upload CSV File

```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@reviews.csv" \
  -F "model=bert" \
  -F "text_column=text"
```

Response:
```json
{
  "file_id": "file-uuid",
  "model": "bert",
  "stats": {
    "total": 100,
    "negative": 20,
    "neutral": 30,
    "positive": 50,
    "avg_confidence": 0.87
  },
  "results_file": "/api/results/file-uuid/download",
  "message": "100 reviews analyzed successfully"
}
```

#### 5. Download Results

```bash
curl -O "http://localhost:8000/api/results/{file_id}/download"
```

### Interactive API Documentation

Once the API is started, access:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## CSV/Excel File Format

### Accepted Columns

The system automatically detects text columns with the following names (case-insensitive):
- `text`
- `avis`
- `review`
- `comment`

### CSV Example

```csv
text
This product is excellent! I highly recommend it.
I don't recommend it, poor quality.
It's okay, nothing special.
```

### Example with Multiple Columns

```csv
id,date,text,rating
1,2024-01-01,"Excellent service!",5
2,2024-01-02,"Not satisfied at all.",1
3,2024-01-03,"Okay, nothing more.",3
```

In this case, specify `text_column=text` during upload.

### Excel Example

Same format as CSV, with support for multiple sheets (first sheet used by default).

## Configuration

### Environment Variables

Create a `.env` file or set these variables:

```bash
# Model paths (optional, automatic detection by default)
BERT_MODEL_PATH=/path/to/bert_model
TFIDF_VECTORIZER_PATH=/path/to/tfidf.pkl
LR_MODEL_PATH=/path/to/lr_model.pkl

# API configuration (optional)
API_HOST=0.0.0.0
API_PORT=8000
```

### Modify API URL (Frontend)

In `frontend/static/js/app.js`, line 1:
```javascript
const API_BASE_URL = 'http://localhost:8000';  // Modify according to your configuration
```

### Prediction Parameters

In `backend/predictor.py`, you can adjust:

```python
MAX_SEQ_LENGTH = 512        # Maximum sequence length for BERT
batch_size = 16             # Batch size for batch processing
```

### CORS (Cross-Origin Resource Sharing)

If you serve the frontend from a different port/domain, modify `backend/app.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "https://your-domain.com"],
    # ...
)
```

## Performance

### BERT/DistilBERT
- **Accuracy**: ~85-95% (depending on training dataset)
- **Speed**: ~50-200 ms per text (CPU), ~10-50 ms (GPU)
- **Resources**: ~500 MB RAM, GPU recommended for production

### TF-IDF + Logistic Regression
- **Accuracy**: ~75-85% (depending on training dataset)
- **Speed**: ~1-5 ms per text
- **Resources**: ~50 MB RAM, CPU sufficient

### Optimizations

- **Batch processing**: Batch processing to improve throughput
- **Lazy loading**: Models loaded only at startup
- **Caching**: Models in memory to avoid reloads

## Troubleshooting

### Models Not Found

**Error**: `FileNotFoundError: BERT model not found`

**Solutions**:
1. Verify that models are in `models_sentiment_yelp/`
2. Check read permissions
3. Use environment variables to specify paths
4. Check startup logs to see searched paths

### CORS Error

**Error**: `Access to fetch at '...' from origin '...' has been blocked by CORS policy`

**Solution**: Modify `backend/app.py` to allow your origin:
```python
allow_origins=["http://localhost:8080"]  # Your frontend port
```

### GPU Not Available

**Message**: `Using CPU device`

**Note**: This is normal if you don't have an NVIDIA GPU. Models work on CPU, but slower. To use GPU:
1. Install PyTorch with CUDA support
2. Verify with `python -c "import torch; print(torch.cuda.is_available())"`

### Memory Error

**Error**: `CUDA out of memory` or `MemoryError`

**Solutions**:
1. Reduce `batch_size` in `predictor.py`
2. Use the TF-IDF model (lighter)
3. Process files in small batches
4. Increase available RAM/VRAM

### CSV Encoding Error

**Error**: `UnicodeDecodeError`

**Solution**: The system automatically tries several encodings. If the problem persists:
1. Verify that your CSV is in UTF-8
2. Open the CSV in an editor and save as UTF-8
3. Specify the encoding in `app.py` if necessary

### Text Column Not Detected

**Error**: `No text column found`

**Solution**:
1. Verify that your file contains a column named: `text`, `avis`, `review`, or `comment`
2. Use the `text_column` parameter to manually specify the column
3. Verify that the column contains text (not just numbers)

### API Won't Start

**Checks**:
1. Port 8000 already in use? Change the port in `app.py` or `start_backend.bat`
2. Dependencies installed? `pip install -r backend/requirements.txt`
3. Python 3.8+? `python --version`
4. Check error logs in the terminal

## Detailed API Endpoints

### GET `/`
General information about the API.

### GET `/health`
API health check and models status.

**Response**:
```json
{
  "status": "healthy",
  "models": {
    "bert": true,
    "tfidf": true,
    "bert_path": "/path/to/model",
    "tfidf_path": "/path/to/model"
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

### GET `/api/models/status`
Detailed status of available models.

### POST `/api/predict/single`
Prediction for a single text.

**Body**:
```json
{
  "text": "Your text here",
  "model": "bert"  // or "tfidf"
}
```

### POST `/api/predict/batch`
Prediction for multiple texts.

**Body**:
```json
{
  "texts": ["Text 1", "Text 2", "Text 3"],
  "model": "bert"
}
```

### POST `/api/upload`
Upload and analyze a CSV/Excel file.

**Form Data**:
- `file`: CSV/Excel file
- `model`: "bert" or "tfidf"
- `text_column`: (optional) Text column name

### GET `/api/results/{file_id}/download`
Download results file.

## Technical Architecture

### Processing Flow

1. **Upload** → File received and temporarily saved
2. **Validation** → Text column detection
3. **Cleaning** → Text preprocessing (URLs, HTML, normalization)
4. **Prediction** → Application of selected model
5. **Post-processing** → Results formatting
6. **Export** → CSV results generation
7. **Cleanup** → Temporary file deletion

### Model Management

- **Lazy loading**: Loaded on first use
- **Singleton pattern**: Single instance in memory
- **Device detection**: Automatic GPU/CPU
- **Error handling**: Graceful error handling for loading

### Text Processing

Cleaning includes:
- Lowercase conversion
- URL and HTML removal
- Contraction expansion ("n't" → "not")
- Space normalization
- Character repetition reduction
- Filtering texts that are too short (< 5 characters)

## Model Training

To train your own models, use the notebook `Sentiment_Analysis_BERT_vs_TFIDF.ipynb`:

1. **Data preparation**: CSV format with `text` and `label` columns
2. **BERT fine-tuning**: Using Hugging Face Transformers
3. **TF-IDF training**: Vectorization + Logistic Regression
4. **Evaluation**: Performance metrics
5. **Export**: Model saving

## Known Limitations

- **Language**: Models optimized for the language used in training (may require retraining for other languages)
- **Length**: Texts truncated to 512 tokens for BERT
- **Context**: Models trained on a specific dataset, performance may vary by domain
- **GPU**: Required for large-scale production with BERT

## Future Improvements

- [ ] Multi-language support
- [ ] Administration interface
- [ ] Authentication and authorization
- [ ] Database for history
- [ ] Online retraining API
- [ ] Additional format support (JSON, XML)
- [ ] Streaming for large files
- [ ] Advanced analytics dashboard
- [ ] Excel export with formatting
- [ ] Cloud services integration (AWS, GCP)

## License

This project is provided for educational and demonstration purposes.

## Contributing

Contributions are welcome! Feel free to:
- Open issues to report bugs
- Propose improvements
- Submit pull requests


---

**Developed with ❤️ for sentiment analysis**
