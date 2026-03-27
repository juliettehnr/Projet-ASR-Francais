import jiwer

# Normalisation avec jiwer
transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.RemoveMultipleSpaces(),
])


def compute_wer(reference, hypothesis):
    ref_norm = transformation(reference)
    hyp_norm = transformation(hypothesis)

    wer_score = jiwer.wer(ref_norm, hyp_norm)

    return {
        "wer": round(wer_score, 4),
        "wer_percent": f"{wer_score * 100:.2f} %",
        "reference_norm": ref_norm,
        "hypothesis_norm": hyp_norm,
    }