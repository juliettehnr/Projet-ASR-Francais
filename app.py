import os
import json
import tempfile
import zipfile
import io
from pathlib import Path

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Transcription Audio")

c1, c2 = st.columns([0.1, 2])
with c1:
    st.markdown("")
with c2:
    st.title("Transcription Audio")
    st.caption("Whisper · Voxtral")


# ── Évaluation — logique exacte de evaluation.py ─────────────────────────────
def compute_wer(ref_text: str, hyp_text: str) -> float:
    """Même transformation et calcul que evaluation.py."""
    import jiwer
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.RemoveMultipleSpaces(),
    ])
    ref = transformation(ref_text)
    hyp = transformation(hyp_text)
    return jiwer.wer(ref, hyp)


def format_results_txt(results: list[tuple[str, float]]) -> str:
    """Même format de sortie que evaluation.py."""
    avg_wer = sum(w for _, w in results) / len(results) if results else 0.0
    lines = []
    lines.append(f"{'Fichier':<30} | {'WER (%)':>7}")
    lines.append("-" * 40)
    for filename, wer in results:
        lines.append(f"{filename:<30} | {wer * 100:7.2f}")
    lines.append("-" * 40)
    lines.append(f"{'Average WER':<30} | {avg_wer * 100:7.2f}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
MainTab, InfoTab = st.tabs(["🎙️ Transcription", "ℹ️ Info"])

with MainTab:

    engine = st.radio(
        "Moteur de transcription",
        ["Whisper (local)", "Voxtral (API Mistral)"],
        horizontal=True,
    )
    st.markdown("---")

    # ── Uploads communs aux deux moteurs ──────────────────────────────────────
    accepted_audio = ["wav", "mp3"] if engine == "Whisper (local)" else ["wav"]
    label_audio    = "Fichiers audio (WAV / MP3)" if engine == "Whisper (local)" else "Fichiers audio (WAV)"

    col_audio, col_ref = st.columns(2)
    with col_audio:
        audio_files = st.file_uploader(label_audio, type=accepted_audio, accept_multiple_files=True)
    with col_ref:
        ref_files = st.file_uploader(
            "Transcriptions vérifiées (TXT) — optionnel",
            type=["txt"],
            accept_multiple_files=True,
            help="Nommées `{nom}_verified.txt` ou simplement `{nom}.txt`. Même nom de base que l'audio.",
        )

    # ══════════════════════════════════════════════════════════════════════════
    # WHISPER
    # ══════════════════════════════════════════════════════════════════════════
    if engine == "Whisper (local)":

        language = st.selectbox("Langue", ["fr", "en", "es", "de", "it", "pt", "auto"], index=0)

        run_btn = st.button("▶ Transcrire", type="primary", disabled=not audio_files, use_container_width=True)

        if run_btn and audio_files:
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                st.error("❌ `faster-whisper` non installé. Lancez : `pip install faster-whisper`")
                st.stop()

            # Index références : accepte "nom_verified.txt" ou "nom.txt"
            ref_index = {}
            for rf in (ref_files or []):
                stem = Path(rf.name).stem.replace("_verified", "")
                ref_index[stem] = rf.read().decode("utf-8")

            results     = {}
            evaluations = []

            with st.spinner("Chargement du modèle Whisper `small`…"):
                model = WhisperModel("small", device="cpu", compute_type="int8")

            progress = st.progress(0, text="Transcription en cours…")

            for idx, uploaded in enumerate(audio_files, 1):
                base_name = Path(uploaded.name).stem
                suffix    = Path(uploaded.name).suffix

                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name

                lang_param    = None if language == "auto" else language
                segments, _   = model.transcribe(tmp_path, language=lang_param)
                transcription = " ".join([seg.text for seg in segments])
                os.unlink(tmp_path)

                results[base_name] = transcription

                if base_name in ref_index:
                    wer = compute_wer(ref_index[base_name], transcription)
                    # Nom du fichier comme dans evaluation.py : base_name_model.txt
                    evaluations.append((f"{base_name}_model.txt", wer))

                progress.progress(idx / len(audio_files), text=f"{idx}/{len(audio_files)} — {uploaded.name}")

            st.session_state["results"]     = results
            st.session_state["evaluations"] = evaluations
            st.success(f"✅ {len(results)} fichier(s) transcrits.")

    # ══════════════════════════════════════════════════════════════════════════
    # VOXTRAL
    # ══════════════════════════════════════════════════════════════════════════
    else:

        api_key = st.text_input(
            "Clé API Mistral",
            value=os.environ.get("MISTRAL_API_KEY", ""),
            type="password",
        )

        run_btn = st.button(
            "▶ Transcrire",
            type="primary",
            disabled=not (audio_files and api_key),
            use_container_width=True,
        )
        if not api_key:
            st.warning("⚠️ Renseignez votre clé API Mistral.")

        if run_btn and audio_files and api_key:
            try:
                from mistralai.client import Mistral
                from pydub import AudioSegment
            except ImportError as e:
                st.error(f"❌ Dépendance manquante : {e}. Lancez : `pip install mistralai pydub`")
                st.stop()

            ref_index = {}
            for rf in (ref_files or []):
                stem = Path(rf.name).stem.replace("_verified", "")
                ref_index[stem] = rf.read().decode("utf-8")

            client      = Mistral(api_key=api_key)
            results     = {}
            evaluations = []
            progress    = st.progress(0, text="Transcription en cours…")

            for idx, uploaded in enumerate(audio_files, 1):
                base_name = Path(uploaded.name).stem

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_tmp:
                    wav_tmp.write(uploaded.read())
                    wav_path = wav_tmp.name

                audio = AudioSegment.from_wav(wav_path)
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_tmp:
                    mp3_path = mp3_tmp.name
                    audio.export(mp3_path, format="mp3")
                os.unlink(wav_path)

                try:
                    with open(mp3_path, "rb") as f:
                        response = client.audio.transcriptions.complete(
                            model="voxtral-mini-2602",
                            file={"content": f, "file_name": f"{base_name}.mp3"},
                        )
                    results[base_name] = response.text

                    if base_name in ref_index:
                        wer = compute_wer(ref_index[base_name], response.text)
                        evaluations.append((f"{base_name}_model.txt", wer))

                except Exception as e:
                    results[base_name] = f"[ERREUR] {e}"
                finally:
                    os.unlink(mp3_path)

                progress.progress(idx / len(audio_files), text=f"{idx}/{len(audio_files)} — {uploaded.name}")

            st.session_state["results"]     = results
            st.session_state["evaluations"] = evaluations
            st.success(f"✅ {len(results)} fichier(s) transcrits.")

    # ══════════════════════════════════════════════════════════════════════════
    # RÉSULTATS
    # ══════════════════════════════════════════════════════════════════════════
    results     = st.session_state.get("results", {})
    evaluations = st.session_state.get("evaluations", [])   # list[tuple[str, float]]

    if results:
        st.markdown("---")
        st.markdown("### Résultats")

        for base_name, text in results.items():
            with st.expander(f"📄 {base_name}", expanded=len(results) == 1):
                edited = st.text_area(
                    "Transcription",
                    value=text,
                    height=180,
                    key=f"edit_{base_name}",
                )
                results[base_name] = edited

        # ── Évaluation ────────────────────────────────────────────────────────
        if evaluations:
            st.markdown("---")
            st.markdown("### Évaluation WER")

            import pandas as pd

            avg_wer = sum(w for _, w in evaluations) / len(evaluations)

            df = pd.DataFrame(
                [(f, round(w * 100, 2)) for f, w in evaluations],
                columns=["Fichier", "WER (%)"],
            )
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.metric("Average WER (%)", round(avg_wer * 100, 2))

        st.markdown("---")
        st.markdown("### Export")

        ecol1, ecol2, ecol3 = st.columns(3)

        # TXT ZIP
        with ecol1:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for base_name, text in results.items():
                    zf.writestr(f"{base_name}_model.txt", text)
            st.download_button(
                "⬇ Transcriptions (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="transcriptions.zip",
                mime="application/zip",
                use_container_width=True,
            )

        # Résultats évaluation TXT — même format que evaluation.py
        with ecol2:
            if evaluations:
                eval_txt = format_results_txt(evaluations)
                st.download_button(
                    "⬇ Évaluation (TXT)",
                    data=eval_txt.encode("utf-8"),
                    file_name="evaluation_resultats.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            else:
                st.button("⬇ Évaluation (TXT)", disabled=True, use_container_width=True,
                          help="Importez des fichiers de référence pour activer l'évaluation.")

        # JSON global
        with ecol3:
            export_data = {"transcriptions": results}
            if evaluations:
                export_data["evaluations"] = {f: round(w * 100, 2) for f, w in evaluations}
            st.download_button(
                "⬇ JSON",
                data=json.dumps(export_data, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="transcriptions.json",
                mime="application/json",
                use_container_width=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
with InfoTab:
    st.markdown("""
## Scripts utilisés

| Rôle | Fichier | Détail |
|------|---------|--------|
| Transcription | `run_whisper.py` | `faster-whisper small`, CPU, int8 |
| Transcription | `run_voxtral.py` | `voxtral-mini-2602` via API Mistral |
| Évaluation | `evaluation.py` | `jiwer` — ToLowerCase, RemovePunctuation, Strip, RemoveMultipleSpaces |

## Convention de nommage
- Audio : `nom.wav` → transcription : `nom_model.txt`
- Référence : `nom_verified.txt` ou `nom.txt`

## Installation

```bash
pip install faster-whisper mistralai pydub jiwer streamlit pandas
streamlit run app.py
```
    """)
