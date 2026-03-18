import os
import argparse
from glob import glob
from faster_whisper import WhisperModel

# Chemin des dossiers
parser = argparse.ArgumentParser(description="Transcrire des fichiers audio avec Faster Whisper")
parser.add_argument("--input_dir", "-i", type=str, required=True, help="Chemin du dossier contenant les fichiers audio (WAV/MP3)")
parser.add_argument("--output_dir", "-o", type=str, required=True, help="Chemin du dossier où enregistrer les transcriptions")
args = parser.parse_args()

audio_dir = args.input_dir
output_dir = args.output_dir

# Trouver tous les fichiers audio WAV et MP3
liste_audios = glob(os.path.join(audio_dir, "*.wav")) + glob(os.path.join(audio_dir, "*.mp3"))
liste_audios.sort() # Tri pour obtenir un ordre stable

# Chargement du modèle Whisper
model_size = "small"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# Boucle sur chaque audio
for i, audio_path in enumerate(liste_audios, 1):
    print(f"Processing {i}/{len(liste_audios)} : {os.path.basename(audio_path)}")

    # Transcription
    segments, info = model.transcribe(audio_path, language="fr")  # change la langue si nécessaire
    transcription = " ".join([segment.text for segment in segments])

    # Nom du fichier de sortie
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_file = os.path.join(output_dir, f"{base_name}_model.txt")

    # Écriture du fichier
    with open(output_file, "w") as f:
        f.write(transcription)

print("Transcriptions terminées.")
