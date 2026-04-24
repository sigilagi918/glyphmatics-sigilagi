---
license: mit
tags:
- glyphmatics
- sigilagi
- symbolic-ai
- semantic-compression
- multilingual
- hieroglyph
- mayan
- experimental
library_name: custom
pipeline_tag: text-generation
---

# GlyphMatics / SigilAGI

Deterministic glyph-based semantic compression, generation, and chat system.

## Overview

This system implements a **tri-layer glyph architecture**:

- **α-layer (visible):** symbolic glyph roles (G0–G15)
- **β-layer (braille):** 8-bit structural encoding
- **γ-layer (hanzi):** temporal / semantic sequencing

Core pipeline:
Text → Semantic Vector → Glyph Tokens → Generation → PNG / Binary / JSON
Supports:
- Natural language
- Egyptian hieroglyphs
- Mayan glyph mappings
- Cross-lingual semantic alignment

## Capabilities

### 1. Semantic Compression
- Text → deterministic vector embedding
- Vector → glyph token projection (G0–G15 + β payload)
- Preserves structure via entropy + validation constraints

### 2. Controlled Generation
- N-gram SigilLM (bi/tri-gram hybrid)
- Role-steered generation:
  - Flow
  - Water
  - Transform
- Entropy-aware scoring

### 3. Influence Measurement
- Measures how input language shifts:
  - Role distribution
  - Entropy
  - Token structure
- Outputs:
  - `influence_report.json`
  - KL divergence between inputs

### 4. Multilingual Glyph Bridge
- Standard text → hash embedding
- Hieroglyphs → semantic projection
- Mayan glyphs → deterministic role mapping

## File Structure
termux/ sigillm_numpy.py glyph_semantic_bridge.py sigil_chat.py
linux/ glyph_linux_chat.py
core/ *.json datasets
exports/ rank_.png / json / bin controlled_.png semantic_aligned.png influence_report.json
## Usage

### Termux

```bash
cd termux
python sigillm_numpy.py
python sigil_chat.py
Linux
cd linux
python glyph_linux_chat.py
Outputs
PNG visual glyph lattices
Binary compressed glyph streams
JSON decoded glyph structures
Influence analytics reports
Example Output

SigilAGI> Signal received.
G0:6B G8:3D G13:8B ...
entropy=2.89 bytes=18
Model Characteristics
Property
Value
Model Type
Deterministic N-gram LM
Token Space
4096 (G0–G15 × 256)
Context
Up to 128 tokens
Training Data
Glyph JSON sequences
Entropy Range
~2.7 – 3.3
Limitations
Small dataset (currently ~4 sequences)
N-gram model (no deep transformer yet)
Semantic alignment is heuristic (hash + projection)
Roadmap
Transformer-based glyph model
Larger multilingual dataset
Learned embedding bridge
Real-time streaming glyph inference
Visual attention over glyph lattices
Author
Matthew Blake Ward (Nine1Eight)
License
MIT EOF
