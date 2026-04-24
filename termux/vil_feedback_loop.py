#!/usr/bin/env python3
import math
import hashlib
import numpy as np

from vil_glyph_compiler import compile_base111

EMBED_DIM = 768
ALPHA = 0.60
BETA = 0.40

def _unit(v):
    n = np.linalg.norm(v)
    if n <= 1e-9:
        return v
    return v / n

def cosine_sim(a, b):
    return float(np.dot(_unit(a), _unit(b)))

def _basis_vec(index, pos, channel="vil"):
    seed = f"{channel}:{index}:{pos}".encode("utf-8")
    raw = bytearray()

    h = hashlib.sha256(seed).digest()
    while len(raw) < EMBED_DIM:
        raw.extend(h)
        h = hashlib.sha256(h).digest()

    v = np.frombuffer(bytes(raw[:EMBED_DIM]), dtype=np.uint8).astype(np.float32)
    v = (v / 127.5) - 1.0
    return v

def vil_encode_text(text):
    """
    Real deterministic VIL-style text encoding:
    text -> canonical base-111 IR -> temporal 768-d embedding.
    """
    ir = compile_base111(text, max_len=256)
    vec = np.zeros(EMBED_DIM, dtype=np.float32)

    for pos, idx in enumerate(ir):
        # temporal weight: later glyphs matter slightly more
        temporal = 1.0 + (pos / max(1, len(ir))) * 0.5

        # role-like modulation from base-111 index
        phase = math.sin((idx + 1) * (pos + 1) * 0.017)
        weight = temporal * (1.0 + 0.15 * phase)

        vec += _basis_vec(idx, pos, "input") * weight

    return _unit(vec).astype(np.float32)

def vil_encode_tokens(tokens):
    """
    Real deterministic token-stream encoding:
    4096-token glyph stream -> 768-d temporal embedding.
    """
    vec = np.zeros(EMBED_DIM, dtype=np.float32)

    for pos, tok in enumerate(tokens[:256]):
        tok = int(tok)
        g = (tok >> 8) & 15
        b = tok & 255

        # encode α-role + β-byte + position
        idx = (g * 256 + b) % 4096
        temporal = 1.0 + (pos / max(1, len(tokens))) * 0.5
        role_weight = 1.0 + (g / 15.0) * 0.25

        vec += _basis_vec(idx, pos, "output") * temporal * role_weight

    return _unit(vec).astype(np.float32)

def vil_score(input_text, tokens, base_score):
    input_vec = vil_encode_text(input_text)
    output_vec = vil_encode_tokens(tokens)

    sim = cosine_sim(input_vec, output_vec)
    sim_score = (sim + 1.0) / 2.0

    final = ALPHA * float(base_score) + BETA * sim_score * 10.0

    return {
        "base_score": float(base_score),
        "vil_similarity": float(sim),
        "vil_score": float(sim_score),
        "final_score": float(final)
    }
