# Demodulate, Then Attend

**Exposing Amplitude Dynamics for Compact EEG Models**

- [Compiled paper](main.pdf)
- [Editable LaTeX](main.tex)
- [References](references.bib)
- [Figure sources](make_figures.py)
- [Measurements behind the figures and tables](evidence/README.md)

The paper studies a simple EEG representation: pair a local band-limited waveform
with its amplitude trajectory before attention. A compact three-axis Transformer
reads the electrode–band–time grid. The manuscript covers the signal-processing
motivation, token construction, frozen-token recovery, TUEV/CHB-MIT evaluation,
and content controls with a fixed reader. The appendix documents the actual data
protocol, training recipe, reconstruction check, and per-class TUEV results.

Edit `main.tex` directly. Unfinished comparisons have empty cells. The reference and matched content controls use seed 2 and validation-selected
checkpoints. Historical baseline cells retain their recorded seeds and report
mean and sample SD where multiple runs exist. Original results, recipe identities,
and dataset bindings are retained in `evidence/results.json` and
`evidence/baseline-results.json`.

## Build

From this directory:

```bash
tectonic main.tex
```

The output is `main.pdf`. References are compiled automatically. To rebuild the
vector figures from the included measurements, use Python with NumPy and
Matplotlib:

```bash
python make_figures.py
tectonic main.tex
```

On Overleaf, select `main.tex` and use pdfLaTeX. The linked
[Overleaf project](https://www.overleaf.com/project/6aeccfc695be3d39c99405dd)
is a separate destination; a GitHub update does not upload to it automatically.

## Official template

This paper uses the official CVPR 2027 anonymous review template, downloaded on
2026-11-06 from the
[Author Guidelines](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines):
[official author kit](https://media.eventhosts.cc/Conferences/CVPR2027/author-kit-CVPR2027-v1-latex.zip).
The bundled `cvpr.sty` and `ieeenat_fullname.bst` are unmodified. There are no custom
margins, body fonts, or line-spacing reductions. The LLM Usage Statement covers
drafting, code/evidence inspection, figure preparation and compilation.

SHA-256:

```text
cvpr.sty             1cd7f434b13d41c475df4a1901a5d82d9ca8b80b8cf6205dc4602a5c164286b4
ieeenat_fullname.bst e38e6166bd7b1e6d23a1b79dcdb55c656e4fcdbe91bdf6b50d827e6b5d1aacfc
```

Drafting used the scientific-writing skill, documented in
[Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065).
