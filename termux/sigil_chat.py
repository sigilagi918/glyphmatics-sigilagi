#!/usr/bin/env python3
from collections import deque
from pathlib import Path
import json

from sigillm_numpy import (
    GlyphTokenizer,
    NGramSigilLM,
    load_dataset,
    vector_to_seed,
    score,
    target_profile,
    render_tokens_png,
    START,
)

from glyph_trainable_embedding import embedding_model
from sigil_neural_model import NeuralGlyphModel

# =========================
# INIT
# =========================
tok = GlyphTokenizer()
model = NGramSigilLM()
model.use_transformer = False
model.transformer = None
ds = load_dataset(tok)
model.fit(ds, epochs=10)

neural_model = NeuralGlyphModel()
neural_model.load()

USE_NEURAL = False
CURRENT_TARGET = None
MEMORY = deque(maxlen=128)

# =========================
# COMMANDS
# =========================
def parse_command(text):
    global CURRENT_TARGET, USE_NEURAL

    if text.startswith("/mode"):
        mode = text.split(" ", 1)[-1].strip()
        CURRENT_TARGET = target_profile(mode)
        print("[MODE]", mode)
        return True

    if text == "/neural on":
        USE_NEURAL = True
        print("[NEURAL ON]")
        return True

    if text == "/neural off":
        USE_NEURAL = False
        print("[NEURAL OFF]")
        return True

    if text == "/clear":
        MEMORY.clear()
        print("[MEMORY CLEARED]")
        return True

    return False

# =========================
# MEMORY
# =========================
def inject_memory(seed):
    if not MEMORY:
        return seed
    return seed + list(MEMORY)[-8:]

# =========================
# TRACE
# =========================
def trace(tokens, n=24):
    out=[]
    for t in tokens:
        g=(t>>8)&15
        b=t&255
        if g==15: continue
        out.append(f"G{g}:{b:02X}")
        if len(out)>=n: break
    return " ".join(out)

# =========================
# GENERATION
# =========================
def generate(seed):
    if USE_NEURAL:
        tokens = list(seed)
        for _ in range(32):
            nxt = neural_model.next_token(tokens)
            tokens.append(nxt)
        return tokens
    else:
        return model.generate(seed)

# =========================
# CHAT
# =========================
def chat():
    print("[SIGIL CHAT READY]")
    print("Commands: /mode, /neural on/off, /clear, exit")

    while True:
        user = input("\nYou> ").strip()

        if user == "exit":
            embedding_model.save()
            break

        if parse_command(user):
            continue

        seed = vector_to_seed(user, START, 8)
        seed = inject_memory(seed)

        tokens = generate(seed)

        if CURRENT_TARGET:
            meta = score(tokens, CURRENT_TARGET)
        else:
            meta = score(tokens)

        MEMORY.extend(tokens[-12:])

        # 🔥 ONLINE LEARNING
        if meta["score"] > 4.5:
            embedding_model.update(user, reward=1.0)

        Path("chat_exports").mkdir(exist_ok=True)
        render_tokens_png(tokens, "chat_exports/latest_reply.png")

        print(
            "SigilAGI>",
            trace(tokens),
            "| H=", round(meta["entropy"],3),
            "| score=", round(meta["score"],3),
            "| neural=", USE_NEURAL
        )
        print("[PNG] chat_exports/latest_reply.png")

if __name__ == "__main__":
    chat()
