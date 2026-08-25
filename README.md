# Reproducibility code — An Iterative, Non-Compensatory, Rank-Based Multicriteria Selection Method

This repository is the Code Availability companion to the paper (submitted to *4OR – A Quarterly Journal of Operations Research*):

> [Author name], "An Iterative, Non-Compensatory, Rank-Based Multicriteria Selection Method," 4OR (under review), 2026.

It reproduces **every table and every figure** in the paper's Results and Discussion section (Section 5), plus the two illustrative figures used earlier in the paper (Sections 3.4 and 3.7), from a single, self-contained Jupyter notebook.

## Contents

```
.
├── reproducibility.ipynb   # the notebook — run top to bottom, in order
├── data/                   # place the two downloaded CSVs here (see below)
├── figures/                # populated by the notebook when it runs
├── requirements.txt
└── README.md
```

There is no other code in this repository: the method, the metrics, the baseline
implementations, the dataset loading, and every table/figure are all defined and
computed inside `reproducibility.ipynb`, in the order they appear in the paper.

## How to run

1. Install the dependencies (the notebook's own first cell also does this via
   `%pip install`, so a plain `jupyter notebook` / `jupyter lab` launch followed
   by "Run All" is sufficient on its own):

   ```
   pip install -r requirements.txt
   ```

2. Download the two public datasets used in Sections 5.3 and 5.4 and place them,
   unmodified, in `data/`:

   | File | Source |
   |---|---|
   | `data/qs_university_rankings_2026.csv` | [QS World University Rankings 2026 (Kaggle)](https://www.kaggle.com/datasets/dhrubangtalukdar/qs-world-university-rankings-2026-top-1500) |
   | `data/nba_players_1996_2022.csv` | [NBA Players dataset, 1996–2022 (Kaggle)](https://www.kaggle.com/datasets/justinas/nba-players-data) |

   The two small benchmark matrices used in Sections 5.1 and 5.2 (Bhangale et
   al., 2004; Karsak et al., 2012) need no download — they are transcribed
   directly from the original publications inside the notebook.

3. Open `reproducibility.ipynb` and run every cell from top to bottom, in order.
   Later cells depend on variables defined in earlier ones; there are no
   out-of-order dependencies. A full run takes well under a minute on an
   ordinary laptop (the slowest single step is the runtime-complexity benchmark
   in Section 13, which is deliberately CPU-bound).

Every cell is preceded by a markdown cell explaining what it computes and which
table or figure in the paper its output corresponds to, so the notebook can be
read section by section alongside the paper.

## Table / figure map

| Notebook section | Produces |
|---|---|
| 1–5. Setup, imports, core method, metrics, baselines | definitions only |
| 6. Datasets | Table `tab:bhangale`, Table `tab:karsak`; loads the QS/NBA CSVs |
| 7. Figure — method schematic | **Figure 1** (Section 3.4) |
| 8. Figure — WS-coefficient saturation | **Figure 2** (Section 3.7) |
| 9. Figure — retention-ratio sensitivity | **Figure 3** (opening of Section 5) |
| 10. Section 5.1 — Bhangale (2004) | Tables `tab:literature-validation-{composition,p50,comparison}` |
| 11. Section 5.2 — Karsak (2012) | Tables `tab:illustrative-{composition,comparison,ranks}` |
| 12. Section 5.3 — QS World University Rankings | Tables `tab:qs-{composition,mechanism,comparison}` |
| 13. Section 5.4 — NBA player performance | Tables `tab:nba-{composition,comparison,complexity}`, **Figures 4 and 5** |
| 14. Section 5.5 — Cross-dataset synthesis | Tables `tab:cross-dataset-{ranks,wsm}`, **Figure 6** |

Figure numbers follow the compiled paper's reading order, not the order the
notebook computes them in (Figure 2, the WS-saturation plot, is produced in
notebook Section 8 because it illustrates Section 3.7, even though Sections 10–14
come later in the notebook). Every file written to `figures/` is named after its
paper figure number, e.g. `figure2_ws_saturation.pdf`.

Section 14 (cross-dataset synthesis) does not hardcode any of the pooled
agreement values: it reuses the `*_agreement` tables built live in Sections
10–13, so Figure 6 and Tables `tab:cross-dataset-ranks` / `tab:cross-dataset-wsm`
are a genuine end-to-end recomputation, not a redrawing of numbers copied from
the paper.

## Reproducibility notes

Building and executing this notebook against the actual cited datasets surfaced
a few small discrepancies with the paper text as currently written. Each is
flagged in place inside the notebook, immediately above the cell where it
becomes visible:

- **Section 8 (Figure 2, WS-saturation).** The paper's caption states Scenario
  A's Spearman ρ ≈ 0.07; the notebook computes ρ ≈ 0.09 with the same fixed
  seed and procedure used to generate the figure.
- **Section 11 (Table `tab:illustrative-ranks`).** Borda's total rank-sum on the
  Karsak matrix has two exact ties (R2/R12 and R3/R11). The notebook reports
  these as fractional (average) ranks rather than silently reproducing a
  particular whole-number tie-break.
- **Section 12 (Table `tab:qs-mechanism`).** The paper names "MIT, Stanford, and
  Harvard" as the three highest-scoring universities on QS's own Overall SCORE.
  The cited dataset's Overall SCORE column actually ranks Imperial College
  London 2nd — ahead of Stanford, Oxford, and Harvard — which is also the
  proposed method's own top pick at several values of *p*, so it cannot serve as
  an example of a top-scoring university *excluded* from C\*. The same table's
  University of Cambridge row has its worst-criterion label misattributed
  (`Citations per Faculty SCORE` instead of `International Student SCORE`); the
  worst-rank value itself (118) is correct.

These do not affect the method or its implementation — only a handful of
numbers and names in the manuscript prose, which should be revised to match
before submission.

## Citation

If you use this code, please cite the paper (full citation to be added once
the DOI/venue is finalized) and, where applicable, the third-party datasets:

- Bhangale, P. P., Agrawal, V. P., & Saha, S. K. (2004). Attribute based
  specification, comparison and selection of a robot. *Mechanism and Machine
  Theory*, 39(12), 1345–1366.
- Karsak, E. E., Sener, Z., & Dursun, M. (2012). Robot selection using a fuzzy
  regression-based decision making approach. *International Journal of
  Production Research*, 50(23), 6826–6834.
- QS Quacquarelli Symonds. *QS World University Rankings 2026* [Data set].
  Kaggle.
- Kaggle (user: justinas). *NBA Players dataset, 1996–2022 seasons* [Data set].

## License

Add a license here (e.g. MIT) before publishing this repository.

## Generative AI use

Parts of this notebook (code structure, docstrings, and the transcription of
the reference implementation into notebook form) were produced with the
assistance of a large language model (Claude). All method design, results, and
conclusions are the responsibility of the paper's author.
