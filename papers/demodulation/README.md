# Demodulate, Then Attend

**Exposing Amplitude Dynamics for Compact EEG Models**

- [Overleaf project](https://www.overleaf.com/project/6aeccfc695be3d39c99405dd)
- [PDF preview](https://github.com/Joni-z/CFM4EEG/blob/main/papers/demodulation/main.pdf)
- [Editable LaTeX](main.tex)
- [References](references.bib)

This draft contains the abstract, introduction, and related work. It asks how
explicit bandwise amplitude dynamics and complementary waveform content can
help compact EEG encoders. The three related-work sections cover EEG
pretraining and tokenization, frequency-resolved representations, and
demodulation for compact models.

Edit `main.tex` directly. This is a standalone two-column preview, not the
complete conference manuscript. Methods, results, and appendices have not been
added here. Experimental records are maintained in the
[project archive](../../docs/EXPERIMENTS.md).

## Build

With Tectonic installed, run from this directory:

```bash
tectonic main.tex
```

The output is `main.pdf`. References are compiled automatically.

Drafting support: the scientific-writing skill; see the
[Scientific Agent Skills reference](https://doi.org/10.48550/arXiv.2609.00065).
