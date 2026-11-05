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

Edit `main.tex` directly. This draft uses the official CVPR 2027 anonymous review format. It is not
the complete conference manuscript. Methods, results, and appendices have not been
added here. Experimental records are maintained in the
[project archive](../../docs/EXPERIMENTS.md).

## Build

With Tectonic installed, run from this directory:

```bash
tectonic main.tex
```

The output is `main.pdf`. References are compiled automatically. On Overleaf,
select `main.tex` as the main document and use pdfLaTeX.

Drafting support: the scientific-writing skill; see the
[Scientific Agent Skills reference](https://doi.org/10.48550/arXiv.2609.00065).

## Official template

Downloaded on 2026-11-06 from the submission-template link on the
[CVPR 2027 Author Guidelines](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines):
[official author kit ZIP](https://media.eventhosts.cc/Conferences/CVPR2027/author-kit-CVPR2027-v1-latex.zip).
The bundled `cvpr.sty` and `ieeenat_fullname.bst` are unmodified. The main
file uses review mode, the official hyperlink settings and bibliography
formatting, with no custom margins, fonts, or spacing.

The required LLM Usage Statement describes assistance with this opening
draft. Its scope should be updated when the full manuscript is completed.

SHA-256:

```text
author-kit.zip       da1ec5db9b73531ea82a87723e4aedb631083331fe655d1972279b48a7bdb2fe
cvpr.sty             1cd7f434b13d41c475df4a1901a5d82d9ca8b80b8cf6205dc4602a5c164286b4
ieeenat_fullname.bst e38e6166bd7b1e6d23a1b79dcdb55c656e4fcdbe91bdf6b50d827e6b5d1aacfc
```
