# Reproducibility code — IQIS: Iterative Quantile-Intersection Selection

This repository is the Code Availability companion to the paper (submitted to *4OR – A Quarterly Journal of Operations Research*):

> [Author name], "IQIS: An Iterative, Non-Compensatory, Rank-Based Multicriteria Selection Method," 4OR (under review), 2026.

It reproduces **every table and every figure** in the paper's Results and Discussion section (Section 5) — including the ELECTRE III / ELECTRE TRI-B comparison, the rank-reversal test, the cross-dataset Nemenyi post-hoc analysis, and the synthetic (Monte Carlo) robustness study added in this revision — plus the two illustrative figures used earlier in the paper (Sections 3.4 and 3.7), from a single, self-contained Jupyter notebook.

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

2. Download the two public datasets used in Sections 5.5 and 5.6 and place them,
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
   out-of-order dependencies. A full run takes about a minute and a half on an
   ordinary laptop. Most of that time is ELECTRE III (Section 12 onward), whose
   $O(N^2K)$ cost is paid once per dataset via a small caching helper
   (`score_cache`) rather than once per value of $p$; on the QS dataset
   ($N=690$) that single computation is itself close to a minute. The
   1,440-run synthetic sweep in Section 17 and the runtime-complexity benchmark
   in Section 15 are both fast in comparison (a few seconds each).

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
| 12. Section 5.3 — ELECTRE TRI-B sorting comparison | Table `tab:electre-tri` |
| 13. Section 5.4 — Rank-reversal test | Table `tab:rank-reversal` |
| 14. Section 5.5 — QS World University Rankings | Tables `tab:qs-{composition,mechanism,comparison}` |
| 15. Section 5.6 — NBA player performance | Tables `tab:nba-{composition,comparison,complexity}`, **Figures 4 and 5** |
| 16. Section 5.7 — Cross-dataset synthesis | Tables `tab:cross-dataset-{ranks,wsm}`, **Figure 6**, Nemenyi post-hoc test, **Figure 7** (critical-difference diagrams) |
| 17. Section 5.8 — Synthetic robustness study (Monte Carlo) | **Figure 8** (`figure7_monte_carlo` on disk — see the note on figure numbering below) |
| 18. Section 5.8 — Validating the degeneracy-avoiding retention ratio | Table `tab:degenerate-fix` (p=0.50 vs p=0.75 comparison, L-sensitivity check) |

Figure numbers follow the compiled paper's reading order, not the order the
notebook computes them in (Figure 2, the WS-saturation plot, is produced in
notebook Section 8 because it illustrates Section 3.7, even though Sections
10–17 come later in the notebook). Every file written to `figures/` is named
after the number the figure has *in the paper*, e.g. `figure2_ws_saturation.pdf`;
the one exception is the synthetic-sweep figure, saved as
`figure7_monte_carlo.{pdf,png}` for historical reasons even though it appears
as **Figure 8** in the current manuscript — the filename and the in-paper
number are simply out of step by one and this is not a bug.

Section 16 (cross-dataset synthesis) does not hardcode any of the pooled
agreement values: it reuses the `*_agreement` tables built live in Sections
10, 11, 14, and 15, so Figure 6, the Nemenyi post-hoc test, Figure 7, and
Tables `tab:cross-dataset-ranks` / `tab:cross-dataset-wsm` are a genuine
end-to-end recomputation, not a redrawing of numbers copied from the paper.
Section 17's 1,440-run synthetic sweep is likewise generated live (seeded, so
it is exactly reproducible run to run) rather than replayed from a cached CSV.

## Reproducibility notes

Building and executing this notebook against the actual cited datasets surfaced
a few small discrepancies with earlier drafts of the paper text. The QS
Overall-SCORE discrepancy below has since been corrected in the manuscript
(the paper now names Imperial College London and Oxford, matching the data);
it is kept here, flagged in place inside the notebook, as a record of why the
text reads the way it does and as a check for anyone re-verifying the dataset:

- **Section 8 (Figure 2, WS-saturation).** The paper's caption states Scenario
  A's Spearman ρ ≈ 0.07; the notebook computes ρ ≈ 0.09 with the same fixed
  seed and procedure used to generate the figure.
- **Section 11 (Table `tab:illustrative-ranks`).** Borda's total rank-sum on the
  Karsak matrix has two exact ties (R2/R12 and R3/R11). The notebook reports
  these as fractional (average) ranks rather than silently reproducing a
  particular whole-number tie-break.
- **Section 14 (Table `tab:qs-mechanism`), resolved.** An earlier draft named
  "MIT, Stanford, and Harvard" as the three highest-scoring universities on
  QS's own Overall SCORE. The cited dataset's Overall SCORE column actually
  ranks Imperial College London 2nd — ahead of Stanford, Oxford, and Harvard —
  which is also the proposed method's own top pick at several values of *p*,
  so it could not serve as an example of a top-scoring university *excluded*
  from C\*; the manuscript now names Imperial College London and Oxford
  instead. The same table's University of Cambridge row previously had its
  worst-criterion label misattributed (`Citations per Faculty SCORE` instead
  of `International Student SCORE`); this has also been corrected in the text
  (the worst-rank value itself, 118, was always correct).

None of these affected the method or its implementation — only a handful of
numbers and names in the manuscript prose.

## New in this revision

Relative to the version of this notebook accompanying the first submission,
this release adds:

- **ELECTRE III** as a sixth ranking baseline throughout every agreement
  table (`BASELINES_6`), and **ELECTRE TRI-B** as a sorting-based comparison
  (Section 12, Table `tab:electre-tri`) — both via
  [`pyDecision`](https://pypi.org/project/pyDecision/), with parameters
  (equal weights; indifference/preference/veto thresholds at 10%/30%/70% of
  each criterion's range) documented in the docstrings of
  `electre_iii_score` and `electre_tri_b_good_set`.
- A **rank-reversal test** (Section 13, Table `tab:rank-reversal`): removing
  robots R2 and R12 from the Karsak matrix, replacing them with their
  elementwise-average hybrid R\_hybrid, and rerunning the method and all six
  baselines on the perturbed, 11-candidate matrix.
- A **Nemenyi post-hoc test** and critical-difference diagrams (Section 16,
  Figure 7) following the omnibus Friedman test already present in the first
  submission, via
  [`scikit-posthocs`](https://pypi.org/project/scikit-posthocs/).
- A **1,440-run synthetic (Monte Carlo) robustness study** (Section 17,
  Figure 8 in the paper / `figure7_monte_carlo` on disk) sweeping
  $N\in\{50,\dots,2000\}$, $K\in\{3,5,8,12\}$, and inter-criterion correlation
  $\rho\in\{0,0.5,0.9\}$ — including the threshold-violation "degenerate
  mode" finding reported in the paper's third cautionary note.

`requirements.txt` has been updated to add `pyDecision` and `scikit-posthocs`.

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
