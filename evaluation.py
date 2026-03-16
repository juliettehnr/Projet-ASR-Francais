import jiwer
from pathlib import Path

# Chemins des dossiers
model_dir = Path("./transcriptions_model")
verified_dir = Path("./transcriptions_verified")

# Fichier de sortie
output_file = Path("./evaluation_resultats.txt")

# Transformation pour normalisation avec jiwer
transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.RemoveMultipleSpaces()
])

def read_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read().strip()

def compute_wer(ref_text, hyp_text):
    ref = transformation(ref_text)
    hyp = transformation(hyp_text)
    return jiwer.wer(ref, hyp)


# Stockage des résultats
results = []


# Évaluation des résultats
for model_file in sorted(model_dir.glob("*_model.txt")):
    base_name = model_file.stem.replace("_model", "")
    verified_file = verified_dir / f"{base_name}_verified.txt"

    if not verified_file.exists():
        print(f"Reference file not found for {model_file.name}, skipping.")
        continue

    reference = read_file(verified_file)
    hypothesis = read_file(model_file)

    wer = compute_wer(reference, hypothesis)
    results.append((model_file.name, wer))


# Calcul de la WER moyenne
if results:
    avg_wer = sum(wer for _, wer in results) / len(results)
else:
    avg_wer = 0.0


# Écriture dans le fichier de sortie
with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"{'Fichier':<30} | {'WER (%)':>7}\n")
    f.write("-"*40 + "\n")
    for filename, wer in results:
        f.write(f"{filename:<30} | {wer*100:7.2f}\n")
    f.write("-"*40 + "\n")
    f.write(f"{'Average WER':<30} | {avg_wer*100:7.2f}\n")

print(f"Évaluation terminée. Résultats sauvegardés dans {output_file}")
