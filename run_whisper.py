import os
from glob import glob
from faster_whisper import WhisperModel

# Chemins vers les dossiers
audio_dir = "./audios"
output_dir = "./transcriptions_model"

liste_audios = glob(os.path.join(audio_dir, "*.wav"))

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

