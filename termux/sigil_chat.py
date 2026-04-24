import json, time
from pathlib import Path
from sigillm_numpy import (
    GlyphTokenizer, NGramSigilLM, load_dataset, vector_to_seed,
    score, render_tokens_png, tokens_to_bytes
)

MEM = Path("chat_memory.jsonl")

def save_turn(user, reply, meta):
    with MEM.open("a") as f:
        f.write(json.dumps({
            "time": time.time(),
            "user": user,
            "reply": reply,
            "meta": meta
        }) + "\n")

def glyph_summary(tokens):
    glyphs = []
    for t in tokens:
        g = (t >> 8) & 15
        b = t & 255
        if g == 15 and b in (254, 255):
            continue
        glyphs.append(f"G{g}:{b:02X}")
    return " ".join(glyphs[:24])

def build_reply(user_text, tokens, meta):
    b = tokens_to_bytes(tokens)
    sig = glyph_summary(tokens)

    if "hello" in user_text.lower() or "hi" in user_text.lower():
        prefix = "Signal received."
    elif "what" in user_text.lower():
        prefix = "Decoded structure:"
    elif "build" in user_text.lower() or "make" in user_text.lower():
        prefix = "Execution path:"
    else:
        prefix = "Glyph response:"

    return f"{prefix} {sig} | entropy={meta['entropy']:.3f} bytes={len(b)}"

def main():
    tok = GlyphTokenizer()
    ds = load_dataset(tok)

    model = NGramSigilLM()
    model.fit(ds, epochs=5)

    print("[SIGIL CHAT READY]")
    print("Type 'exit' to quit.")

    while True:
        user = input("\nYou> ").strip()
        if user.lower() in ("exit", "quit"):
            break

        seed = vector_to_seed(user, 4094, 8)
        tokens = model.generate(seed)
        meta = score(tokens)

        Path("chat_exports").mkdir(exist_ok=True)
        render_tokens_png(tokens, "chat_exports/latest_reply.png")

        reply = build_reply(user, tokens, meta)
        save_turn(user, reply, meta)

        print("SigilAGI>", reply)
        print("[PNG] chat_exports/latest_reply.png")

if __name__ == "__main__":
    main()
