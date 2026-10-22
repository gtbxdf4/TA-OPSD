# Manuscript and LaTeX source

The anonymous manuscript **Learning When to Continue and When to Stop: Termination-Aware OPSD** is available as [PDF](manuscript.pdf). The [LaTeX project](source/main.tex) is provided with its figures, bibliography and ICLR template dependencies.

To compile with Tectonic:

```sh
cd paper/source
sh build.sh
```

Set `TECTONIC=/path/to/tectonic` if it is not on PATH. Standard TeX packages and fonts are supplied by the TeX installation; Tectonic may download them on its first run. The project was compiled with Tectonic 0.17.0 and visually checked as a 15-page document. It retains the anonymous review format. This repository does not imply an arXiv or conference submission update.

The five figure PDFs preserve the exact vector Form objects embedded in the manuscript PDF; they were not reconstructed from approximate plots. The original plotting outputs and the embedded paper figures should not be assumed byte-identical. The source preserves the manuscript text and numerical tables, with equivalent TeX punctuation and explicit pdfLaTeX-compatible font encoding for the Tectonic build. The supplied PDF is the original manuscript, so rendering with a different TeX engine may differ.

Template files come from the official [ICLR Master Template](https://github.com/ICLR/Master-Template/tree/46ed6f4c6cef5b175dde23639e77d44c3463b230/iclr2027); their existing headers are retained. The repository code license does not replace authorship or third-party notices for the manuscript and template materials.

Additional experiments and research proposals prepared after this manuscript are reported separately and have not been inserted into this public snapshot.
