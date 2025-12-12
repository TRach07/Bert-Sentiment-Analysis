#!/bin/bash
echo "Starting FastAPI Backend..."
cd backend
python -m uvicorn app:app --reload --port 8000

