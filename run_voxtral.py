import os
from mistralai.client import Mistral
from pydub import AudioSegment
import tempfile

audio_dir = "./audios"
output_dir = "./transcriptions_voxtral"

api_key = ""  # mettre une API key
client = Mistral(api_key=api_key)

os.makedirs(output_dir, exist_ok=True)

liste_audios = []
for root, dirs, files in os.walk(audio_dir):
    for file in files:
        if file.lower().endswith(".wav"):
            liste_audios.append(os.path.join(root, file))

print(f"{len(liste_audios)} fichier(s) WAV trouvé(s).")

for i, audio_path in enumerate(liste_audios, 1):
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_file = os.path.join(output_dir, f"{base_name}_model.txt")

    audio = AudioSegment.from_wav(audio_path)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name
        audio.export(tmp_path, format="mp3")

    try:
        with open(tmp_path, "rb") as f:
            response = client.audio.transcriptions.complete(
                model="voxtral-mini-2602",
                file={
                    "content": f,
                    "file_name": f"{base_name}.mp3",
                },
                # language="fr"
            )

        with open(output_file, "w", encoding="utf-8") as out_f:
            out_f.write(response.text)

        print(f"[{i}/{len(liste_audios)}] Transcrit : {base_name}")

    finally:
        os.remove(tmp_path)

print("Transcriptions terminées.")
