#!/usr/bin/env python3
from collections import deque
from pathlib import Path
import json

from sigillm_numpy import (
    GlyphTokenizer,
    NGramSigilLM,
    load_dataset,
    score,
    target_profile,
    render_tokens_png,
    START,
)

from vil_chat_bridge import make_seed, update_backend
from glyph_trainable_embedding import embedding_model
from sigil_neural_model import NeuralGlyphModel

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
SEED_BACKEND = "trainable"
MEMORY = deque(maxlen=128)

def parse_command(text):
    global CURRENT_TARGET, USE_NEURAL, SEED_BACKEND

    if text.startswith("/mode"):
        mode = text.split(" ", 1)[-1].strip()
        CURRENT_TARGET = target_profile(mode)
        print("[MODE]", mode)
        return True

    if text.startswith("/seed"):
        backend = text.split(" ", 1)[-1].strip().lower()
        if backend not in ("hash", "trainable", "vil"):
            print("[SEED ERROR] use: /seed hash | /seed trainable | /seed vil")
            return True
        SEED_BACKEND = backend
        print("[SEED]", SEED_BACKEND)
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

    if text == "/status":
        print({
            "memory_size": len(MEMORY),
            "mode_active": CURRENT_TARGET is not None,
            "seed_backend": SEED_BACKEND,
            "neural": USE_NEURAL
        })
        return True

    return False

def inject_memory(seed):
    if not MEMORY:
        return seed
    return seed + list(MEMORY)[-8:]

def trace(tokens, n=24):
    out=[]
    for t in tokens:
        g=(int(t)>>8)&15
        b=int(t)&255
        if g==15:
            continue
        out.append(f"G{g}:{b:02X}")
        if len(out)>=n:
            break
    return " ".join(out)

def generate(seed):
    if USE_NEURAL:
        tokens = list(seed)
        for _ in range(32):
            nxt = neural_model.next_token(tokens)
            tokens.append(nxt)
        return tokens
    return model.generate(seed)

def readable_reply(user, tokens, meta):
    q = user.lower()
    if "who are you" in q:
        lead = "I am SigilAGI: a glyph-native symbolic/neural chat engine using selectable semantic seed backends."
    elif any(x in q for x in ("hello", "hi", "hey")):
        lead = "Signal received."
    elif any(x in q for x in ("build", "make", "create", "write")):
        lead = "Execution path formed."
    elif any(x in q for x in ("why", "how", "what")):
        lead = "Decoded answer path formed."
    else:
        lead = "Glyph response generated."

    return (
        f"{lead}\n"
        f"Trace: {trace(tokens)}\n"
        f"Metrics: entropy={meta['entropy']:.3f}, score={meta['score']:.3f}, "
        f"length={meta['length']}, seed={SEED_BACKEND}, neural={USE_NEURAL}"
    )

def save_turn(user, reply, tokens, meta):
    Path("chat_exports").mkdir(exist_ok=True)
    render_tokens_png(tokens, "chat_exports/latest_reply.png")
    Path("chat_exports/latest_reply.json").write_text(json.dumps({
        "user": user,
        "reply": reply,
        "tokens": tokens,
        "glyphs": tok.decode(tokens),
        "meta": meta,
        "seed_backend": SEED_BACKEND,
        "neural": USE_NEURAL
    }, indent=2))

def chat():
    print("[SIGIL CHAT READY]")
    print("Commands: /mode <balanced|water|flow|transform>, /seed <hash|trainable|vil>, /neural on/off, /clear, /status, exit")

    while True:
        user = input("\nYou> ").strip()

        if user.lower() in ("exit", "quit"):
            embedding_model.save()
            break

        if parse_command(user):
            continue

        seed = make_seed(user, backend=SEED_BACKEND, start=START, n=8)
        seed = inject_memory(seed)

        tokens = generate(seed)

        if CURRENT_TARGET:
            meta = score(tokens, CURRENT_TARGET)
        else:
            meta = score(tokens)

        MEMORY.extend(tokens[-12:])

        if meta["score"] > 4.5:
            update_backend(user, SEED_BACKEND, reward=1.0)

        reply = readable_reply(user, tokens, meta)
        save_turn(user, reply, tokens, meta)

        print("SigilAGI>", reply)
        print("[PNG] chat_exports/latest_reply.png")

if __name__ == "__main__":
    chat()
