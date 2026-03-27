"""
Lancement :
    uvicorn main:app --reload

Endpoints :
    GET  /              -> Interface web (index.html)
    GET  /models        -> Liste des modeles disponibles
    POST /transcribe    -> Transcription + calcul WER
"""

import os
import tempfile
import shutil

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from transcription import transcribe_whisper, transcribe_voxtral
from evaluation import compute_wer

app = FastAPI(
    title="API pour l'évaluation d'ASR",
    description="Interface de transcription et d'evaluation WER pour Whisper et Voxtral.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# pour la page HTML
@app.get("/", response_class=HTMLResponse, summary="Interface web")
async def root():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()

# liste des modèles ASR
@app.get("/models", summary="Modeles disponibles")
async def list_models():
    return {
        "models": [
            {
                "id": "whisper",
                "label": "Whisper",
                "requires_api_key": False,
                "description": "Faster-Whisper, modele 'small', execution CPU locale.",
            },
            {
                "id": "voxtral",
                "label": "Voxtral",
                "requires_api_key": True,
                "description": "Voxtral-mini-2602 via l'API Mistral AI. Cle API requise.",
            },
        ]
    }

# transcription + WER
@app.post("/transcribe", summary="Transcription d'un audio et calcul du WER")
async def transcribe(
    audio: UploadFile = File(..., description="Fichier audio WAV ou MP3"),
    reference_file: UploadFile = File(..., description="Fichier .txt contenant la transcription de reference"),
    model: str = Form(..., description="Modele a utiliser : 'whisper' ou 'voxtral'"),
    api_key: str = Form("", description="Cle API Mistral (requise pour Voxtral uniquement)"),
):
    # Lecture du fichier de reference (.txt)
    ref = await reference_file.read()
    try:
        reference = ref.decode("utf-8").strip()
    except UnicodeDecodeError:
        reference = ref.decode("latin-1").strip()

    # Validation du modele
    if model not in ("whisper", "voxtral"):
        raise HTTPException(status_code=400, detail="Modele invalide. Choisir 'whisper' ou 'voxtral'.")

    if model == "voxtral" and not api_key.strip():
        raise HTTPException(status_code=400, detail="Une cle API Mistral est requise pour Voxtral.")

    # Sauvegarde du fichier audio dans un repertoire temporaire
    suffix = os.path.splitext(audio.filename)[-1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        shutil.copyfileobj(audio.file, tmp)

    try:
        if model == "whisper":
            hypothesis = transcribe_whisper(tmp_path)
        else:
            hypothesis = transcribe_voxtral(tmp_path, api_key=api_key.strip())
 
        wer_result = compute_wer(reference, hypothesis)
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)
 
    return JSONResponse({
        "model": model,
        "filename": audio.filename,
        "hypothesis": hypothesis,
        "reference": reference,
        "wer": wer_result["wer"],
        "wer_percent": wer_result["wer_percent"],
        "reference_norm": wer_result["reference_norm"],
        "hypothesis_norm": wer_result["hypothesis_norm"],
    })
