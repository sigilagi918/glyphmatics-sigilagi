#!/usr/bin/env python3
import json
from collections import deque
from pathlib import Path

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

tok = GlyphTokenizer()
model = NGramSigilLM()
ds = load_dataset(tok)
model.fit(ds, epochs=10)

CURRENT_TARGET = None
MEMORY = deque(maxlen=128)

def parse_command(text):
    global CURRENT_TARGET

    if text.startswith("/mode"):
        mode = text.split(" ", 1)[-1].strip()
        CURRENT_TARGET = target_profile(mode)
        print(f"[MODE] {mode}")
        return True

    if text == "/clear":
        MEMORY.clear()
        print("[MEMORY CLEARED]")
        return True

    if text == "/status":
        print({
            "memory_size": len(MEMORY),
            "mode_active": CURRENT_TARGET is not None
        })
        return True

    return False

def inject_memory(seed):
    if not MEMORY:
        return seed
    recent = list(MEMORY)[-16:]
    return seed + recent[:8]

def compact_trace(tokens, limit=24):
    parts = []
    for t in tokens:
        g = (int(t) >> 8) & 15
        b = int(t) & 255
        if g == 15 and b in (254, 255):
            continue
        parts.append(f"G{g}:{b:02X}")
        if len(parts) >= limit:
            break
    return " ".join(parts)

def readable_reply(user, tokens, meta):
    q = user.lower()

    if any(x in q for x in ["who are you", "what are you"]):
        lead = "I am SigilAGI, a glyph-native symbolic chat engine. I map text into G0-G15 glyph roles, generate a structured token path, and return a compressed semantic trace."
    elif any(x in q for x in ["hello", "hi", "hey"]):
        lead = "Signal received. I am ready."
    elif any(x in q for x in ["build", "make", "create", "write"]):
        lead = "Execution path formed. The glyph trace below is the generated build vector."
    elif any(x in q for x in ["why", "how", "what"]):
        lead = "Decoded answer path formed. The glyph trace below shows the symbolic route."
    else:
        lead = "Glyph response generated."

    trace = compact_trace(tokens)
    return f"{lead}\nTrace: {trace}\nMetrics: entropy={meta['entropy']:.3f}, length={meta['length']}, score={meta['score']:.3f}"

def save_chat_turn(user, reply, tokens, meta):
    Path("chat_exports").mkdir(exist_ok=True)
    render_tokens_png(tokens, "chat_exports/latest_reply.png")
    Path("chat_exports/latest_reply.json").write_text(json.dumps({
        "user": user,
        "reply": reply,
        "tokens": tokens,
        "glyphs": tok.decode(tokens),
        "meta": meta
    }, indent=2))

def chat():
    print("[SIGIL CHAT READY]")
    print("Commands: /mode <balanced|water|flow|transform>, /clear, /status, exit")

    while True:
        user = input("\nYou> ").strip()

        if user.lower() in ("exit", "quit"):
            break

        if parse_command(user):
            continue

        seed = vector_to_seed(user, START, 8)
        seed = inject_memory(seed)

        if CURRENT_TARGET:
            tokens = model.generate(seed, target_roles=CURRENT_TARGET["roles"])
            meta = score(tokens, CURRENT_TARGET)
        else:
            tokens = model.generate(seed)
            meta = score(tokens)

        MEMORY.extend(tokens[-12:])
        reply = readable_reply(user, tokens, meta)
        save_chat_turn(user, reply, tokens, meta)

        print("SigilAGI>", reply)
        print("[PNG] chat_exports/latest_reply.png")

if __name__ == "__main__":
    chat()
