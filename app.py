import os
import json
import traceback
import numpy as np
import gradio as gr

# --- IMPORT CORE ---
try:
    from termux.sigil_chat import run_backend
except:
    # fallback safe stub
    def run_backend(text, backend="hash"):
        return {
            "tokens": [4094, 803, 3613, 4095],
            "meta": {"score": 1.0, "entropy": 1.0, "length": 4}
        }

from termux.vil_feedback_loop import vil_score

# --- CORE PIPELINE ---
def process(text):
    try:
        results = []

        for backend in ["hash", "trainable", "vil"]:
            out = run_backend(text, backend)

            vs = vil_score(text, out["tokens"], out["meta"]["score"])

            results.append({
                "backend": backend,
                "score": out["meta"]["score"],
                "entropy": out["meta"]["entropy"],
                "length": out["meta"]["length"],
                "vil_similarity": round(vs["vil_similarity"], 4),
                "final_score": round(vs["final_score"], 3),
                "trace": out["tokens"][:16]
            })

        results.sort(key=lambda x: x["final_score"], reverse=True)

        return json.dumps(results, indent=2)

    except Exception as e:
        return f"ERROR:\n{traceback.format_exc()}"

# --- UI ---
demo = gr.Interface(
    fn=process,
    inputs=gr.Textbox(label="Input Text"),
    outputs=gr.Code(label="Backend Comparison (VIL-Weighted)"),
    title="SigilAGI Glyph Engine (VIL + RL)",
    description="Deterministic glyph generation with VIL embedding feedback loop"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
