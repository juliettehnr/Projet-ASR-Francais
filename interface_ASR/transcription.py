import os
import tempfile

# Whisper

def transcribe_whisper(audio_path: str, model_size: str = "small") -> str:
    from faster_whisper import WhisperModel
    
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, language="fr")
    transcription = " ".join(segment.text.strip() for segment in segments)
    return transcription

# Voxtral
def transcribe_voxtral(audio_path: str, api_key: str) -> str:
    if not api_key:
        raise ValueError("Une clé API Mistral est requise pour utiliser Voxtral.")

    from mistralai import Mistral
    from pydub import AudioSegment

    client = Mistral(api_key=api_key)

    # Conversion WAV vers MP3 dans un fichier temporaire
    audio = AudioSegment.from_file(audio_path)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name
        audio.export(tmp_path, format="mp3")

    try:
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        with open(tmp_path, "rb") as f:
            response = client.audio.transcriptions.complete(
                model="voxtral-mini-2602",
                file={
                    "content": f,
                    "file_name": f"{base_name}.mp3",
                },
            )
        return response.text
    finally:
        os.remove(tmp_path)
