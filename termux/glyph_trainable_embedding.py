import json
import hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np

DIM = 64
EMB_PATH = "trainable_embedding.json"

def _hash_vec(text):
    h = hashlib.sha256(text.encode("utf-8")).digest()
    raw = list(h)
    while len(raw) < DIM:
        h = hashlib.sha256(h).digest()
        raw.extend(list(h))
    return np.array(raw[:DIM], dtype=np.float32) / 255.0

class TrainableGlyphEmbedding:
    def __init__(self, dim=DIM):
        self.dim = dim
        self.word_vecs = {}
        self.counts = defaultdict(int)

    def encode_text(self, text):
        words = [w.lower() for w in text.split() if w.strip()]
        if not words:
            return np.zeros(self.dim, dtype=np.float32)

        vec = np.zeros(self.dim, dtype=np.float32)
        for w in words:
            if w not in self.word_vecs:
                self.word_vecs[w] = _hash_vec(w)
            vec += np.array(self.word_vecs[w], dtype=np.float32)

        return vec / max(1, len(words))

    def update(self, text, reward=1.0, lr=0.05):
        words = [w.lower() for w in text.split() if w.strip()]
        if not words:
            return

        direction = _hash_vec(text)
        direction = direction / max(1e-9, np.linalg.norm(direction))

        for w in words:
            if w not in self.word_vecs:
                self.word_vecs[w] = _hash_vec(w)

            v = np.array(self.word_vecs[w], dtype=np.float32)
            v = v + lr * reward * direction
            v = np.clip(v, 0.0, 1.0)
            self.word_vecs[w] = v.tolist()
            self.counts[w] += 1

    def vector_to_seed(self, text, start=4094, n=8):
        vec = self.encode_text(text)
        out = [start]
        for i in range(n):
            v = float(vec[i % self.dim]) * 255.0
            g = int(abs(v + i * 7)) % 15
            b = int((v * 31 + i * 17)) % 256
            out.append((g << 8) | b)
        return out

    def save(self, path=EMB_PATH):
        data = {
            "dim": self.dim,
            "word_vecs": self.word_vecs,
            "counts": dict(self.counts)
        }
        Path(path).write_text(json.dumps(data))

    def load(self, path=EMB_PATH):
        p = Path(path)
        if not p.exists():
            return False
        data = json.loads(p.read_text())
        self.dim = data.get("dim", DIM)
        self.word_vecs = data.get("word_vecs", {})
        self.counts = defaultdict(int, {k:int(v) for k,v in data.get("counts", {}).items()})
        return True
