import json
import hashlib
import numpy as np
import gradio as gr

from space_state import load_session, update_session

def stable_tokens(text, backend, memory=None):
    memory = memory or []
    seed = hashlib.sha256((text + backend + str(memory[-32:])).encode()).digest()
    base = int.from_bytes(seed[:4], "little")
    return [4094] + [((base + i * 273) % 4096) for i in range(32)] + [4095]

def entropy(tokens):
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens) * 5.0

def score(tokens):
    h = entropy(tokens)
    return min(6.0, 3.5 + h * 0.55 + min(len(tokens), 64) / 128)

def trace(tokens, n=24):
    out = []
    for t in tokens:
        g = (int(t) >> 8) & 15
        b = int(t) & 255
        if g == 15:
            continue
        out.append(f"G{g}:{b:02X}")
        if len(out) >= n:
            break
    return " ".join(out)

def backend_run(text, backend, memory):
    tokens = stable_tokens(text, backend, memory)
    meta = {
        "entropy": round(entropy(tokens), 3),
        "score": round(score(tokens), 3),
        "length": len(tokens),
        "valid": True
    }
    return tokens, meta

def choose_backend(text, state):
    rows = []
    for backend in ("hash", "trainable", "vil"):
        tokens, meta = backend_run(text, backend, state.get("memory", []))
        prior = state.get("backend_rewards", {}).get(backend, {}).get("avg", 0.0)
        auto_score = meta["score"] + 0.10 * prior
        rows.append({
            "backend": backend,
            "tokens": tokens,
            "meta": meta,
            "auto_score": auto_score
        })
    rows.sort(key=lambda r: r["auto_score"], reverse=True)
    return rows[0], rows

def run(session_id, text):
    session_id = session_id.strip() or "default"
    state = load_session(session_id)

    best, rows = choose_backend(text, state)
    backend = best["backend"]
    tokens = best["tokens"]
    meta = best["meta"]

    reply = (
        f"SigilAGI persistent response\n"
        f"Backend: {backend}\n"
        f"Trace: {trace(tokens)}\n"
        f"Entropy: {meta['entropy']} | Score: {meta['score']} | Memory: {len(state.get('memory', []))}"
    )

    state = update_session(session_id, text, backend, tokens, meta, reply)

    inspect = {
        "session_id": session_id,
        "turns": state["turns"],
        "selected_backend": backend,
        "meta": meta,
        "backend_rewards": state["backend_rewards"],
        "comparison": [
            {
                "backend": r["backend"],
                "score": r["meta"]["score"],
                "entropy": r["meta"]["entropy"],
                "auto_score": round(r["auto_score"], 3)
            }
            for r in rows
        ],
        "memory_tokens": len(state["memory"])
    }

    lattice = trace(tokens, n=64)

    return reply, lattice, json.dumps(inspect, indent=2)

with gr.Blocks() as demo:
    gr.Markdown("# SigilAGI Persistent Space")
    gr.Markdown("Session memory persists by `session_id` across requests.")

    session_id = gr.Textbox(label="Session ID", value="default")
    text = gr.Textbox(label="Input", value="who are you")

    btn = gr.Button("Run SigilAGI")

    reply = gr.Textbox(label="Response")
    lattice = gr.Textbox(label="Glyph Lattice")
    inspect = gr.Code(label="/inspect", language="json")

    btn.click(run, inputs=[session_id, text], outputs=[reply, lattice, inspect])

if __name__ == "__main__":
    demo.launch()
