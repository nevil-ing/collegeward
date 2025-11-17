import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Collegeward API",
    description="AI-powered interactive study companion for medical students",
    version="0.1.0",
    docs_url="/",
    redoc_url="/redoc",
)

