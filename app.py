import json
import numpy as np
import gradio as gr
from functools import lru_cache

# ---- IMPORT YOUR SYSTEM ----
# Adjust paths if needed
import sys
sys.path.insert(0, "termux")

from sigillm_numpy import NGramSigilLM
from sigil_chat import readable_reply
from vil_feedback_loop import vil_score

# ---- INIT ----
model = NGramSigilLM()

# ---- CACHE LAYER (CRITICAL) ----
@lru_cache(maxsize=256)
def cached_vil(text, tokens_tuple, base_score):
    return vil_score(text, list(tokens_tuple), base_score)

# ---- GLYPH LATTICE VISUAL ----
def render_lattice(tokens):
    grid = []
    for t in tokens[:64]:
        g = (int(t) >> 8) & 0xF
        b = int(t) & 0xFF
        grid.append(f"G{g}:{b:02X}")
    return " ".join(grid)

# ---- CORE INFERENCE ----
def run_sigil(input_text):
    # seed (simple hash fallback; your backend routing can replace this)
    seed = [4094] + [(ord(c) % 4096) for c in input_text[:8]]

    tokens = model.generate(seed)
    base_score = 5.0  # or your score() output

    vs = cached_vil(input_text, tuple(tokens), base_score)

    reply = readable_reply(input_text, tokens, {
        "entropy": 0,
        "score": vs["final_score"]
    })

    lattice = render_lattice(tokens)

    inspect = {
        "vil_similarity": vs["vil_similarity"],
        "vil_score": vs["vil_score"],
        "final_score": vs["final_score"],
        "length": len(tokens)
    }

    return reply, lattice, json.dumps(inspect, indent=2)

# ---- UI ----
with gr.Blocks() as demo:
    gr.Markdown("# SigilAGI Space (Real Backend)")

    inp = gr.Textbox(label="Input")
    btn = gr.Button("Run")

    out_text = gr.Textbox(label="Response")
    out_lattice = gr.Textbox(label="Glyph Lattice")
    out_inspect = gr.Code(label="/inspect")

    btn.click(fn=run_sigil, inputs=inp, outputs=[out_text, out_lattice, out_inspect])

demo.launch()
