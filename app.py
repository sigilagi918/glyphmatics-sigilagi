import json
import numpy as np
import gradio as gr

# --- SIMPLE CORE (self-contained, no repo imports) ---

def fake_backend(text, name):
    base = abs(hash(text + name)) % 4096

    tokens = [(base + i*17) % 4096 for i in range(24)]

    entropy = float(len(set(tokens)) / len(tokens) * 5)
    score = float((base % 100) / 10 + 4)

    return tokens, {"entropy": entropy, "score": score, "length": len(tokens)}

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a)+1e-8) / (np.linalg.norm(b)+1e-8))

def embed(text):
    np.random.seed(abs(hash(text)) % (2**32))
    return np.random.uniform(-1, 1, 768)

# --- PIPELINE ---

def run(text):
    results = []

    for backend in ["hash", "trainable", "vil"]:
        tokens, meta = fake_backend(text, backend)

        v1 = embed(text)
        v2 = embed(str(tokens))

        sim = cosine(v1, v2)
        final = 0.6 * meta["score"] + 0.4 * ((sim+1)/2)*10

        results.append({
            "backend": backend,
            "score": round(meta["score"], 3),
            "entropy": round(meta["entropy"], 3),
            "length": meta["length"],
            "similarity": round(sim, 4),
            "final_score": round(final, 3),
            "trace": tokens[:12]
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return json.dumps(results, indent=2)

# --- UI ---

demo = gr.Interface(
    fn=run,
    inputs=gr.Textbox(label="Input"),
    outputs=gr.Code(label="Backend Comparison"),
    title="SigilAGI (Stable Space)",
    description="Working baseline. Replace core with real engine once stable."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
