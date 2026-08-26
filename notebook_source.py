# %% [markdown]
# # Reproducibility Notebook — An Iterative Rank-Based Multicriteria Selection Method
#
# This notebook reproduces **every table and every figure** reported in the paper's
# Results and Discussion section (Section 5), plus the two illustrative figures used
# earlier in the paper (Sections 3.4 and 3.7), from the method's reference
# implementation and the raw/public datasets described in the paper.
#
# **How to use this notebook.** Run all cells from top to bottom, in order — later
# cells depend on variables defined in earlier ones (in particular, the dataset
# loading cells in Section 6 must run before any per-dataset section 10-14). There
# are no hidden dependencies on external `.py` files: every function used below is
# defined in this notebook, in the "Core method", "Metrics", and "Baseline methods"
# sections.
#
# **Before running:** download the two public datasets cited in the paper and place
# them, unmodified, in a `data/` folder next to this notebook:
#
# | File expected at | Source |
# |---|---|
# | `data/qs_university_rankings_2026.csv` | [QS World University Rankings 2026 (Kaggle)](https://www.kaggle.com/datasets/dhrubangtalukdar/qs-world-university-rankings-2026-top-1500) |
# | `data/nba_players_1996_2022.csv` | [NBA Players dataset, 1996–2022 (Kaggle)](https://www.kaggle.com/datasets/justinas/nba-players-data) |
#
# The two small benchmark matrices (Bhangale et al., 2004; Karsak et al., 2012) need
# no download — they are transcribed directly from the original publications in
# Section 6 below, byte-for-byte identical to the tables printed in the paper.
#
# **Notebook map** (which section produces which paper table/figure):
#
# | Notebook section | Produces |
# |---|---|
# | 1. Setup | — |
# | 2. Imports | — |
# | 3. Core method (`iqis_select`) | — (definitions only) |
# | 4. Metrics | — (definitions only) |
# | 5. Baseline methods | — (definitions only) |
# | 6. Datasets | Table `tab:bhangale`, Table `tab:karsak`, loads QS/NBA CSVs |
# | 7. Figure — method schematic | **Figure 1** in the paper (Section 3.4) |
# | 8. Figure — WS-coefficient saturation | **Figure 2** in the paper (Section 3.7) |
# | 9. Figure — retention-ratio sensitivity | **Figure 3** in the paper (opening of Section 5) |
# | 10. Section 5.1 — Bhangale (2004) | Tables `tab:literature-validation-{composition,p50,comparison}` |
# | 11. Section 5.2 — Karsak (2012) | Tables `tab:illustrative-{composition,comparison,ranks}` |
# | 12. Section 5.3 — ELECTRE TRI-B sorting comparison | Table `tab:electre-tri` |
# | 13. Section 5.4 — Rank-reversal test | Table `tab:rank-reversal` |
# | 14. Section 5.5 — QS World University Rankings | Tables `tab:qs-{composition,mechanism,comparison}` |
# | 15. Section 5.6 — NBA player performance | Tables `tab:nba-{composition,comparison,complexity}`, **Figures 4 and 5** |
# | 16. Section 5.7 — Cross-dataset synthesis | Tables `tab:cross-dataset-{ranks,wsm}`, **Figure 6**, **Figure 7** (Nemenyi CD) |
# | 17. Section 5.8 — Synthetic robustness study (Monte Carlo) | **Figure 8** (`figure7_monte_carlo` in the paper's figure files) |
#
# Figure numbers above are the numbers the figures receive *in the compiled
# paper*, which follow reading order rather than the order this notebook computes
# them in (the WS-saturation figure, for instance, is Figure 2 in the paper because
# it sits early, in Section 3.7, even though it is generated in notebook Section 8).
# The filenames written to `figures/` spell out which paper figure each one is.
#
# **Reproducibility notes.** Building this notebook surfaced a few small
# discrepancies between the already-written article text and what the reference
# implementation actually computes on the exact cited data. Each is flagged
# in-place, in a markdown cell immediately above where it becomes visible, rather
# than silently "corrected" here:
#
# - Section 8 (Figure 2 / WS-saturation): the article's caption states
#   Scenario A's Spearman $\rho\approx0.07$; this notebook computes
#   $\rho\approx0.09$ with the same seed and procedure.
# - Section 11 (Table `tab:illustrative-ranks`): Borda has two exact rank-sum
#   ties on the Karsak matrix (R2/R12 and R3/R11); this notebook reports them as
#   fractional (average) ranks rather than reproducing the specific whole-number
#   tie-break printed in the article.
# - Section 14 (Table `tab:qs-mechanism`): the article names "MIT, Stanford, and
#   Harvard" as the Overall-SCORE top 3; the cited dataset's own Overall SCORE
#   column ranks Imperial College London 2nd (ahead of Stanford, Oxford, and
#   Harvard), which also happens to be the method's own top pick -- see the note
#   in Section 14 for why this affects the argument, not just the wording. The
#   same table's University of Cambridge row also has its worst criterion
#   mislabeled in the article (`Citations per Faculty SCORE` instead of the
#   correct `International Student SCORE`; the worst-rank *value*, 118, is right).

# %% [markdown]
# ## 1. Setup
#
# Run this cell once. It installs every third-party library used anywhere in this
# notebook. Nothing below this cell requires any other installation step.

# %%
# %pip install -q numpy pandas matplotlib scipy pymcdm pyDecision scikit-posthocs

# %% [markdown]
# ## 2. Imports

# %%
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; figures are saved to disk and also
                        # displayed inline via plt.show() at the end of each figure cell
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.stats import spearmanr, kendalltau, friedmanchisquare
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

from pymcdm.methods import TOPSIS as _TOPSIS
from pymcdm.methods import VIKOR as _VIKOR
from pymcdm.methods import WSM as _WSM
from pymcdm.methods import PROMETHEE_II as _PROMETHEE_II

DATA_DIR = Path("data")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# Shared plotting style (serif to match the LaTeX body text of the paper; a
# colorblind-checked categorical palette; every series also carries a distinct
# marker/linestyle so figures stay legible under grayscale printing).
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#52514e",
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.dpi": 110,
    "savefig.dpi": 300,
})
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#256abf", "#0d366b"])
INK, MUTED = "#0b0b0b", "#898781"


def save_fig(fig, name):
    """Save a figure as both PDF (vector, for \\includegraphics in the paper's
    LaTeX source) and PNG (for quick preview), then display it inline."""
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    print(f"Saved figures/{name}.pdf and figures/{name}.png")
    plt.show()

# %% [markdown]
# ## 3. Core method — `iqis_select`
#
# Reference implementation of the iterative, per-criterion top-$p$ retention and
# intersection procedure described in Section 3 ("Proposed Method") and
# Algorithm 1 (`alg:proposed`) of the paper.
#
# Two design choices are worth flagging because they are what make the complexity
# and correctness results in the paper hold, not just implementation detail:
#
# - **Ranking happens once, up front.** Every criterion is sorted a single time,
#   before the main loop (`orders` below); each iteration only *filters* that fixed
#   order against the current candidate set. This is what gives the
#   $O(NK\log N)$ time complexity of Proposition `prop:complexity`, instead of
#   $O(NKT\log N)$ from re-sorting at every iteration.
# - **The loop stops on either of two conditions** — a threshold violation
#   ($|C^{(t+1)}|<L$) or stabilization ($C^{(t+1)}=C^{(t)}$) — matching
#   Proposition `prop:termination`. The stabilization check is required for
#   correctness, not an optimization: without it, nothing guarantees the loop
#   terminates.
#
# The output is not just the final set $C^*$, but a `classes` array giving every
# original candidate's elimination round (Remark `rem:induced-ranking`): class 1
# is eliminated first, class $T+1$ is $C^*$ itself. This `classes` array is what
# every rank-correlation comparison against a baseline uses below.

# %%
Direction = str  # "max" or "min"


@dataclass
class IQISResult:
    classes: np.ndarray             # class(c) for every candidate, higher = better
    final_class_indices: np.ndarray  # indices of C*
    n_iterations: int                # T
    stopped_by: str                  # "threshold" or "stabilization"
    history: List[dict]
    p: np.ndarray
    L: int

    @property
    def selected(self) -> np.ndarray:
        return self.final_class_indices

    @property
    def n_selected(self) -> int:
        return int(self.final_class_indices.shape[0])


def _validate_inputs(X, directions, p, L):
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array of shape (N, K), got shape {X.shape}")
    N, K = X.shape
    if len(directions) != K:
        raise ValueError(f"directions must have length K={K}, got {len(directions)}")
    for d in directions:
        if d not in ("max", "min"):
            raise ValueError(f'each direction must be "max" or "min", got {d!r}')
    p_vec = np.full(K, float(p)) if np.isscalar(p) else np.asarray(p, dtype=float)
    if p_vec.shape != (K,):
        raise ValueError(f"p must be a scalar or a sequence of length K={K}")
    if np.any((p_vec <= 0) | (p_vec >= 1)):
        raise ValueError("all retention ratios p_k must lie strictly in (0, 1)")
    if not (1 <= L <= N):
        raise ValueError(f"L must satisfy 1 <= L <= N (N={N}), got L={L}")
    return N, K, p_vec


def _top_p_indices(order, alive, p):
    """Filter the (fixed, precomputed) best->worst order against the current
    `alive` mask, keeping relative order, then truncate to the top ceil(p * n_alive)
    among currently-alive candidates. O(len(order)) per call."""
    alive_in_order = order[alive[order]]
    n_alive = alive_in_order.shape[0]
    k = int(np.ceil(p * n_alive))
    return alive_in_order[:k]


def iqis_select(X, directions, p, L, max_iterations=None) -> IQISResult:
    """Run the iterative multicriteria intersection selection method
    (Algorithm 1 / `alg:proposed`) on candidate-by-criterion matrix X.

    Parameters
    ----------
    X : array-like, shape (N, K)
    directions : sequence of K strings, each "max" or "min"
    p : float or sequence of K floats in (0, 1) -- retention ratio(s); a scalar is
        broadcast to the equal-priority case p_1 = ... = p_K = p.
    L : int -- minimum acceptable cardinality, 1 <= L <= N.
    """
    X = np.asarray(X, dtype=float)
    N, K, p_vec = _validate_inputs(X, directions, p, L)
    signs = np.array([1.0 if d == "max" else -1.0 for d in directions])

    orders = [np.argsort(-signs[k] * X[:, k], kind="stable") for k in range(K)]

    alive = np.ones(N, dtype=bool)
    classes = np.zeros(N, dtype=int)
    history: List[dict] = []

    t = 0
    stopped_by = "threshold"
    cap = max_iterations if max_iterations is not None else N + 1

    while True:
        if t >= cap:
            stopped_by = "max_iterations_cap"
            break

        S = [_top_p_indices(orders[k], alive, p_vec[k]) for k in range(K)]
        new_alive = np.zeros(N, dtype=bool)
        new_alive[S[0]] = True
        for k in range(1, K):
            mask_k = np.zeros(N, dtype=bool)
            mask_k[S[k]] = True
            new_alive &= mask_k

        n_new = int(new_alive.sum())
        history.append({"iteration": t, "n_current": int(alive.sum()), "n_new": n_new,
                         "subset_sizes": [int(len(s)) for s in S]})

        if n_new < L:
            stopped_by = "threshold"
            break
        if np.array_equal(new_alive, alive):
            stopped_by = "stabilization"
            break

        eliminated = alive & ~new_alive
        classes[eliminated] = t + 1
        alive = new_alive
        t += 1

    final_label = t + 1
    classes[alive] = final_label

    return IQISResult(classes=classes, final_class_indices=np.flatnonzero(alive),
                       n_iterations=t, stopped_by=stopped_by, history=history,
                       p=p_vec, L=L)

# %% [markdown]
# ## 4. Metrics — rank correlation and the WS coefficient
#
# `spearman_rho` and `kendall_tau` are thin wrappers fixing the convention used
# throughout (higher score = better, matching `IQISResult.classes`). `ws_coefficient`
# implements the WS rank-similarity coefficient of Sałabun & Urbaniak (2020), whose
# top-weighted construction is the subject of Figure 2 in the paper and the caveat
# in Remark `rem:induced-ranking`.

# %%
def spearman_rho(scores_a, scores_b) -> float:
    rho, _ = spearmanr(np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float))
    return float(rho)


def kendall_tau(scores_a, scores_b) -> float:
    tau, _ = kendalltau(np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float))
    return float(tau)


def ws_coefficient(reference_scores, compared_scores) -> float:
    """WS(x, y) = 1 - sum_i 2^-Rx(i) * |Rx(i)-Ry(i)| / max(|Rx(i)-1|, |Rx(i)-N|),
    where Rx, Ry are 1-indexed rank positions (1 = best). Asymmetric: `reference_scores`
    plays the role of x, so disagreements near the top of the REFERENCE ranking are
    penalized far more than disagreements near the bottom -- the mechanism behind
    Figure 2 in the paper."""
    x = np.asarray(reference_scores, dtype=float)
    y = np.asarray(compared_scores, dtype=float)
    N = x.shape[0]
    if N < 2:
        return 1.0
    Rx = stats.rankdata(-x, method="average")
    Ry = stats.rankdata(-y, method="average")
    denom = np.maximum(np.abs(Rx - 1), np.abs(Rx - N))
    denom = np.where(denom == 0, 1.0, denom)
    terms = (2.0 ** (-Rx)) * np.abs(Rx - Ry) / denom
    return float(1.0 - terms.sum())


def jaccard_overlap(set_a, set_b) -> float:
    """|A n B| / |A u B| -- used to compare C* against set-based baselines
    (skyline, naive screening) at their native output size, no truncation."""
    A, B = set(set_a), set(set_b)
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B)


def rank_reversal_test(X, directions, p, L, remove_index, max_iterations=None) -> dict:
    """Triantaphyllou (2000)-style rank-reversal test: remove a single candidate
    NOT in the original C*, rerun, and check whether the relative order of the
    original C* (read off the induced ranking) is preserved. Not used to produce
    any table or figure in the current version of the paper -- included here as
    part of the method's reference implementation, for the systematic Monte Carlo
    robustness study identified as future work in the Conclusion."""
    X = np.asarray(X, dtype=float)
    base = iqis_select(X, directions, p, L, max_iterations=max_iterations)
    if remove_index in base.final_class_indices:
        raise ValueError("remove_index must NOT be a member of the original C*.")
    N = X.shape[0]
    keep_mask = np.ones(N, dtype=bool)
    keep_mask[remove_index] = False
    X_reduced = X[keep_mask]
    orig_to_reduced = -np.ones(N, dtype=int)
    orig_to_reduced[keep_mask] = np.arange(int(keep_mask.sum()))
    perturbed = iqis_select(X_reduced, directions, p, L, max_iterations=max_iterations)
    original_winners = base.final_class_indices
    reduced_positions = orig_to_reduced[original_winners]
    original_order = base.classes[original_winners]
    perturbed_order = perturbed.classes[reduced_positions]
    if len(set(original_order.tolist())) < 2:
        perturbed_winner_set = set(reduced_positions[perturbed_order == perturbed_order.max()].tolist())
        all_still_top = perturbed_winner_set == set(reduced_positions.tolist())
        return {"tau_within_original_winners": None, "reversal_detected": not all_still_top,
                "note": "original C* was a single tied class; compared via set membership"}
    tau, _ = kendalltau(original_order, perturbed_order)
    return {"tau_within_original_winners": float(tau), "reversal_detected": bool(tau < 1.0)}

# %% [markdown]
# ## 5. Baseline methods
#
# Four baselines with no third-party dependency (naive single-pass screening, the
# Pareto skyline, Borda count, and single-pass top-$k$ intersection), plus four
# baselines taken from `pymcdm` — a maintained, citable, third-party
# implementation, used precisely to avoid a home-grown bug undermining the
# comparison (TOPSIS, VIKOR, WSM, PROMETHEE II). Every function here returns a
# higher-is-better score array of shape `(N,)`, regardless of the underlying
# method's native convention (VIKOR is natively lower-is-better and is negated).

# %%
def _signs(directions):
    return np.array([1.0 if d == "max" else -1.0 for d in directions])


def naive_conjunctive_screening(X, directions, p):
    """A single, non-iterated pass of the intersection rule -- exactly C^(1) in
    the proposed method's notation, computed once and returned as-is regardless
    of size (Section 3.8, 'Versus naive single-pass conjunctive screening')."""
    X = np.asarray(X, dtype=float)
    N, K = X.shape
    p_vec = np.full(K, float(p)) if np.isscalar(p) else np.asarray(p, dtype=float)
    signs = _signs(directions)
    selected = None
    for k in range(K):
        order = np.argsort(-signs[k] * X[:, k], kind="stable")
        kk = int(np.ceil(p_vec[k] * N))
        top_k_set = set(order[:kk].tolist())
        selected = top_k_set if selected is None else (selected & top_k_set)
    return np.array(sorted(selected), dtype=int)


def pareto_front(X, directions):
    """Classical Pareto skyline (Borzsonyi, Kossmann & Stocker, 2001),
    block-nested-loop, O(N^2 K)."""
    X = np.asarray(X, dtype=float)
    N = X.shape[0]
    signs = _signs(directions)
    Y = X * signs
    is_dominated = np.zeros(N, dtype=bool)
    for i in range(N):
        if is_dominated[i]:
            continue
        ge = np.all(Y >= Y[i], axis=1)
        gt = np.any(Y > Y[i], axis=1)
        dominates_i = ge & gt
        dominates_i[i] = False
        if np.any(dominates_i):
            is_dominated[i] = True
    return np.flatnonzero(~is_dominated)


def borda_count(X, directions):
    """Classical Borda count: sum per-criterion rank positions (1=best), negate
    so higher = better. Compensatory -- included precisely to make that contrast
    with the proposed method measurable (Section 3.8, 'Versus rank aggregation')."""
    X = np.asarray(X, dtype=float)
    N, K = X.shape
    signs = _signs(directions)
    rank_sum = np.zeros(N, dtype=float)
    for k in range(K):
        order = np.argsort(-signs[k] * X[:, k], kind="stable")
        ranks = np.empty(N, dtype=float)
        ranks[order] = np.arange(1, N + 1)
        rank_sum += ranks
    return -rank_sum


def topk_rank_intersection(X, directions, k):
    """Kumar & Punera (2011)-style single-pass top-k aggregation: the
    intersection of per-criterion top-k sets, k fixed exogenously (no iteration,
    no adaptive stopping rule)."""
    X = np.asarray(X, dtype=float)
    N, K = X.shape
    signs = _signs(directions)
    selected = None
    for kk in range(K):
        order = np.argsort(-signs[kk] * X[:, kk], kind="stable")
        top_k_set = set(order[:k].tolist())
        selected = top_k_set if selected is None else (selected & top_k_set)
    return np.array(sorted(selected), dtype=int)


def _types_from_directions(directions):
    return np.array([1 if d == "max" else -1 for d in directions])


def _equal_weights(K):
    return np.ones(K) / K


def topsis_score(X, directions, weights=None):
    """TOPSIS (Hwang & Yoon, 1981), via pymcdm. Higher = better (native)."""
    X = np.asarray(X, dtype=float)
    types = _types_from_directions(directions)
    w = _equal_weights(X.shape[1]) if weights is None else np.asarray(weights, dtype=float)
    return _TOPSIS()(X, w, types)


def vikor_score(X, directions, weights=None, v=0.5):
    """VIKOR (Opricovic & Tzeng), via pymcdm. Native convention is lower=better
    (a regret/distance measure); negated here so higher = better, like every
    other function in this section."""
    X = np.asarray(X, dtype=float)
    types = _types_from_directions(directions)
    w = _equal_weights(X.shape[1]) if weights is None else np.asarray(weights, dtype=float)
    raw = _VIKOR(v=v)(X, w, types)
    return -raw


def wsm_score(X, directions, weights=None):
    """Weighted Sum Model (equal weights), via pymcdm. Higher = better. NOTE:
    pymcdm's sum-normalization requires strictly positive input on every
    criterion -- raises ValueError if any column has negative values (this is
    exactly why WSM is omitted from the NBA comparison in Section 5.6, since
    net rating is frequently negative)."""
    X = np.asarray(X, dtype=float)
    types = _types_from_directions(directions)
    w = _equal_weights(X.shape[1]) if weights is None else np.asarray(weights, dtype=float)
    return _WSM()(X, w, types)


def promethee_ii_score(X, directions, weights=None, preference_function="vshape", p=None, q=None):
    """PROMETHEE II (Brans & Vincke, 1985), via pymcdm, under explicitly reported,
    analyst-fixed parameters (v-shape preference function, per-criterion range as
    the default preference threshold p)."""
    X = np.asarray(X, dtype=float)
    types = _types_from_directions(directions)
    w = _equal_weights(X.shape[1]) if weights is None else np.asarray(weights, dtype=float)
    if p is None:
        p = np.ptp(X, axis=0)
    return _PROMETHEE_II(preference_function, p=p, q=q)(X, w, types)


try:
    from pyDecision.algorithm import electre_iii, electre_tri_b
except ImportError:
    raise ImportError(
        "pyDecision is required for ELECTRE III / ELECTRE TRI-B (Section 5.3 "
        "of the paper). Install it with `pip install pyDecision`."
    )


def _electre_qpv(Y, q_frac=0.10, p_frac=0.30, v_frac=0.70):
    """Indifference/preference/veto thresholds as fixed fractions of each
    criterion's observed range -- the same 'per-criterion range' convention
    already used for PROMETHEE II's threshold in `promethee_ii_score`,
    extended to three thresholds instead of one (Section 4.3 of the paper)."""
    rng = np.ptp(Y, axis=0)
    rng = np.where(rng == 0, 1.0, rng)
    return q_frac * rng, p_frac * rng, v_frac * rng


def electre_iii_score(X, directions, q_frac=0.10, p_frac=0.30, v_frac=0.70):
    """ELECTRE III (Roy, 1991), via pyDecision. Returns a higher-is-better score
    array of shape (N,), for direct use alongside every other baseline in this
    notebook. ELECTRE III natively returns a (possibly incomparable) pre-order
    rather than a single score, via two independent distillations (descending
    and ascending). Following standard ELECTRE III practice for producing a
    single comparable ranking, this wrapper takes the median of each
    alternative's rank in the two distillations -- a comparability device, not
    a claim that ELECTRE III itself resolves genuine incomparabilities --
    exactly as reported in Section 5.3 of the paper."""
    X = np.asarray(X, dtype=float)
    N, K = X.shape
    signs = np.array([1.0 if d == "max" else -1.0 for d in directions])
    Y = X * signs
    Q, P, V = _electre_qpv(Y, q_frac, p_frac, v_frac)
    W = np.full(K, 1.0 / K)
    _, _, rank_D, rank_A, _, _ = electre_iii(Y, P=P, Q=Q, V=V, W=W, graph=False)

    def group_rank(groups):
        # each element of `groups` is a string like "a1; a5; a6" when
        # alternatives 1, 5, 6 are tied at that position of the distillation.
        r = np.zeros(N)
        for pos, group in enumerate(groups, start=1):
            for name in group.split(";"):
                idx = int(name.strip()[1:]) - 1
                r[idx] = pos
        return r

    rD, rA = group_rank(rank_D), group_rank(rank_A)
    return -(rD + rA) / 2.0  # higher = better


def electre_tri_b_good_set(X, directions, p, q_frac=0.10, p_frac=0.30, v_frac=0.70):
    """ELECTRE TRI-B (pessimistic 'pc' rule via pyDecision), with a single
    boundary profile placed at the (1-p) quantile of each criterion (in
    benefit orientation) -- the same retention ratio p already used for IQIS
    on this dataset, so the two methods' 'good' sets are compared at matched
    nominal selectivity (Section 5.3 of the paper). Returns a boolean mask,
    True = assigned to the top category."""
    X = np.asarray(X, dtype=float)
    N, K = X.shape
    signs = np.array([1.0 if d == "max" else -1.0 for d in directions])
    Y = X * signs
    Q, P, V = _electre_qpv(Y, q_frac, p_frac, v_frac)
    W = np.full(K, 1.0 / K)
    B = np.percentile(Y, 100.0 * (1.0 - p), axis=0)
    classification = electre_tri_b(Y, W=list(W), Q=list(Q), P=list(P), V=list(V),
                                    B=list(B), cut_level=1.0, rule='pc', verbose=False)
    # classification[i] is a plain int category index. With a single boundary
    # profile, pyDecision's pessimistic ('pc') rule assigns the TOP category
    # index 0 to alternatives that outrank the profile, and index 1 to those
    # that do not -- verified empirically on a synthetic 3-point matrix with a
    # known best/middle/worst alternative before trusting it on real data.
    return np.array([c == 0 for c in classification], dtype=bool)


BASELINES_6 = {
    "TOPSIS": topsis_score,
    "VIKOR": vikor_score,
    "WSM": wsm_score,
    "PROMETHEE II": promethee_ii_score,
    "Borda": lambda X, directions: borda_count(X, directions),
    "ELECTRE III": electre_iii_score,
}
# Retained under its original name for any external code that imports it;
# from Section 5.3 onward this notebook uses BASELINES_6 (six baselines,
# ELECTRE III included) as the default for every agreement table.
BASELINES_5 = BASELINES_6


def agreement_table(result: IQISResult, X, directions, baselines=BASELINES_6) -> pd.DataFrame:
    """Compare the induced ranking (`result.classes`) against every baseline in
    `baselines` using Spearman's rho, Kendall's tau, and the WS coefficient.
    Skips a baseline (with a printed note) if it raises -- this is how WSM is
    excluded from the NBA comparison in Section 5.6. `baselines` values may be
    either a callable `fn(X, directions)` or an already-computed score array
    (see `score_cache` below) -- the latter avoids recomputing the same
    baseline, in particular the O(N^2 K) ELECTRE III, once per value of p."""
    rows = []
    for name, val in baselines.items():
        try:
            score = val(X, directions) if callable(val) else val
        except Exception as e:
            print(f"  [skipped] {name}: {type(e).__name__}: {e}")
            continue
        rows.append({
            "Method": name,
            "Spearman rho": round(spearman_rho(result.classes, score), 3),
            "Kendall tau": round(kendall_tau(result.classes, score), 3),
            "WS": round(ws_coefficient(result.classes, score), 3),
        })
    return pd.DataFrame(rows).set_index("Method")


def score_cache(X, directions, baselines=BASELINES_6):
    """Compute every baseline's score once for a given dataset, so that
    multiple `agreement_table` calls at different values of p (which do not
    change any baseline's own score) can reuse it instead of recomputing --
    most relevant for ELECTRE III, whose O(N^2 K) cost is otherwise paid once
    per p value for no benefit. A baseline that raises here is stored as None
    and re-raises (and is skipped, exactly as before) inside `agreement_table`."""
    out = {}
    for name, fn in baselines.items():
        try:
            out[name] = fn(X, directions)
        except Exception as e:
            out[name] = _RaisingPlaceholder(e)
    return out


class _RaisingPlaceholder:
    """Stand-in for a baseline score that failed in `score_cache`, so the
    original exception is re-raised (and caught) inside `agreement_table`
    exactly as if the baseline function had been called directly there."""
    def __init__(self, exc):
        self._exc = exc

    def __call__(self, *args, **kwargs):
        raise self._exc

# %% [markdown]
# ## 6. Datasets
#
# ### 6.1 Small benchmark matrices (Bhangale, 2004; Karsak, 2012)
#
# Hardcoded, byte-for-byte transcriptions of Table `tab:bhangale` and Table
# `tab:karsak` in the paper. No download needed.

# %%
BHANGALE_LABELS = [
    "ASEA-IRB 60/2", "Cincinnati Milacrone T3-726", "Cybotech V15 Electric Robot",
    "Hitachi America Process Robot", "Unimation PUMA 500/600",
    "US Robots Maker 110", "Yaskawa Electric Motoman L3C",
]
BHANGALE_COLUMNS = ["LC (kg)", "RE (mm)", "MTS (mm/s)", "MC (pts)", "MR (mm)"]
BHANGALE_DIRECTIONS = ["max", "min", "max", "max", "max"]  # RE (repeatability) is minimized
BHANGALE_X = np.array([
    [60.0, 0.4, 2540.0, 500.0, 990.0],
    [6.35, 0.15, 1016.0, 3000.0, 1041.0],
    [6.8, 0.1, 1727.2, 1500.0, 1676.0],
    [10.0, 0.2, 1000.0, 2000.0, 965.0],
    [2.5, 0.1, 560.0, 500.0, 915.0],
    [4.5, 0.08, 1016.0, 350.0, 508.0],
    [3.0, 0.1, 1778.0, 1000.0, 920.0],
])
print("Table tab:bhangale -- Bhangale (2004), 7 robots x 5 criteria")
display(pd.DataFrame(BHANGALE_X, index=BHANGALE_LABELS, columns=BHANGALE_COLUMNS))

# %%
KARSAK_LABELS = [f"R{i}" for i in range(1, 13)]
KARSAK_COLUMNS = ["Cost", "HC", "LC (kg)", "Repeatability", "Velocity"]
KARSAK_DIRECTIONS = ["min", "max", "max", "max", "max"]  # Cost is minimized
KARSAK_X = np.array([
    [100000, 0.995, 85, 1.7, 3.0],
    [75000, 0.933, 45, 2.5, 3.6],
    [56250, 0.875, 18, 5.0, 2.2],
    [28125, 0.409, 16, 1.7, 1.5],
    [46875, 0.818, 20, 5.0, 1.1],
    [78125, 0.664, 60, 2.5, 1.35],
    [87500, 0.88, 90, 2.0, 1.4],
    [56250, 0.633, 10, 8.0, 2.5],
    [56250, 0.653, 25, 4.0, 2.5],
    [87500, 0.747, 100, 2.0, 2.5],
    [68750, 0.88, 100, 4.0, 1.5],
    [43750, 0.633, 70, 5.0, 3.0],
], dtype=float)
print("Table tab:karsak -- Karsak et al. (2012), 12 robots x 5 criteria")
display(pd.DataFrame(KARSAK_X, index=KARSAK_LABELS, columns=KARSAK_COLUMNS))

# %% [markdown]
# ### 6.2 QS World University Rankings ($N=690$ after filtering)
#
# Restricted to universities with complete values on all six official QS
# sub-indicator scores and a numeric Overall SCORE (Section 5.5). All six
# criteria are maximized.

# %%
QS_CRITERIA_COLUMNS = [
    "Academic Reputation SCORE", "Employer Reputation SCORE",
    "Faculty Student Ratio SCORE", "Citations per Faculty SCORE",
    "International Faculty  SCORE", "International Student SCORE",
]
QS_DIRECTIONS = ["max"] * len(QS_CRITERIA_COLUMNS)

qs_raw = pd.read_csv(DATA_DIR / "qs_university_rankings_2026.csv")
qs_df = qs_raw.dropna(subset=QS_CRITERIA_COLUMNS).copy()
qs_df["Overall SCORE"] = pd.to_numeric(qs_df["Overall SCORE"], errors="coerce")
qs_df = qs_df.dropna(subset=["Overall SCORE"]).reset_index(drop=True)

QS_LABELS = qs_df["Name"].astype(str).tolist()
QS_X = qs_df[QS_CRITERIA_COLUMNS].to_numpy(dtype=float)
QS_OVERALL_SCORE = qs_df["Overall SCORE"].to_numpy(dtype=float)

print(f"QS World University Rankings: N = {len(QS_LABELS)} universities "
      f"(expected 690)")
assert len(QS_LABELS) == 690, "Filtering does not match the paper -- check the source CSV."

# %% [markdown]
# ### 6.3 NBA player performance
#
# Two views of the same source file are used in the paper: the 2021--22 season
# restricted to players with at least 40 games played ($N=358$, Section 5.6's
# main comparison), and the full multi-season panel with complete cases on the
# five criteria ($N=12{,}844$, Section 5.6's runtime-complexity table).

# %%
NBA_CRITERIA_COLUMNS = ["pts", "reb", "ast", "net_rating", "ts_pct"]
NBA_DIRECTIONS = ["max"] * len(NBA_CRITERIA_COLUMNS)

nba_raw = pd.read_csv(DATA_DIR / "nba_players_1996_2022.csv")

nba_season_df = nba_raw[(nba_raw["season"] == "2021-22") & (nba_raw["gp"] >= 40)].copy()
nba_season_df = nba_season_df.dropna(subset=NBA_CRITERIA_COLUMNS).reset_index(drop=True)
NBA_LABELS = nba_season_df["player_name"].astype(str).tolist()
NBA_X = nba_season_df[NBA_CRITERIA_COLUMNS].to_numpy(dtype=float)

print(f"NBA 2021--22 season, gp >= 40: N = {len(NBA_LABELS)} players (expected 358)")
assert len(NBA_LABELS) == 358, "Filtering does not match the paper -- check the source CSV."

nba_full_df = nba_raw.dropna(subset=NBA_CRITERIA_COLUMNS).reset_index(drop=True)
NBA_FULL_X = nba_full_df[NBA_CRITERIA_COLUMNS].to_numpy(dtype=float)
print(f"NBA full panel (all seasons, complete cases): N = {len(NBA_FULL_X)} "
      f"player-seasons (expected 12,844)")
assert len(NBA_FULL_X) == 12844, "Filtering does not match the paper -- check the source CSV."

# %% [markdown]
# ## 7. Figure — method schematic
#
# **Produces Figure 1 in the paper** (`fig:method-schematic`, Section 3.4, placed
# just before Algorithm 1). Pure illustration: no data is used, it only draws the
# mechanism described in Section 3.

# %%
def make_method_schematic():
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.0, 5)
    ax.axis("off")

    def box(x, y, w, h, text, fc="white", ec=INK, fs=8, lw=1.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                     boxstyle="round,pad=0.06,rounding_size=0.08",
                                     linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3)

    def arrow(x0, y0, x1, y1, color=INK, lw=1.1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                      mutation_scale=9, linewidth=lw, color=color, zorder=1))

    box(0.1, 2.0, 1.3, 1.0, r"$C^{(t)}$" + "\n" + r"$N_t$ candidates", fc="#f0efec")
    filt_y = [3.6, 2.25, 0.9]
    labels = [r"$S_1^{(t)}=\mathrm{Top}_{p_1}(C^{(t)},f_1)$",
              r"$S_2^{(t)}=\mathrm{Top}_{p_2}(C^{(t)},f_2)$",
              r"$S_K^{(t)}=\mathrm{Top}_{p_K}(C^{(t)},f_K)$"]
    for y, lab, c in zip(filt_y, labels, [BLUE, ORANGE, AQUA]):
        arrow(1.4, 2.5, 2.7, y + 0.4)
        box(2.7, y, 3.0, 0.8, lab, fc="white", ec=c, fs=7.5)
        arrow(5.7, y + 0.4, 6.9, 2.5)
    ax.text(3.4, 1.75, r"$\vdots$", fontsize=11, ha="center")
    box(6.9, 2.05, 1.0, 0.9, r"$\bigcap$", fc="#f0efec", fs=14)
    arrow(7.9, 2.5, 9.0, 2.5)
    box(9.0, 2.0, 0.9, 1.0, r"$C^{(t+1)}$", fc="white", ec=INK)
    ax.annotate("", xy=(0.75, 0.75), xytext=(9.45, 0.75),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0,
                                 connectionstyle="arc3,rad=-0.22"))
    ax.text(5.1, -0.75, r"repeat with $C^{(t+1)}$ in place of $C^{(t)}$, until stabilization or"
                        r" $|C^{(t+1)}|<L$   $\Rightarrow$   $C^*$",
            ha="center", fontsize=7.5, color="#3a3a38")
    ax.set_title("Ranking $\\to$ per-criterion retention $\\to$ intersection $\\to$ iterate",
                  fontsize=9, pad=6)
    return fig

fig = make_method_schematic()
save_fig(fig, "figure1_method_schematic")

# %% [markdown]
# ## 8. Figure — WS-coefficient saturation
#
# **Produces Figure 2 in the paper** (`fig:ws-saturation`, Section 3.7, placed
# right after Remark `rem:induced-ranking`). Panel (a) is a pure property of the
# WS formula's exponential weights (no dataset needed, $N=690$ is used only as a
# representative size matching the QS dataset). Panel (b) uses two synthetic
# rankings, constructed here, to make the practical consequence concrete.

# %%
def make_ws_saturation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.7))

    N = 690
    i = np.arange(1, N + 1)
    w = 2.0 ** (-i.astype(float))
    cum = np.cumsum(w) / w.sum()
    ax1.plot(i, cum * 100, color=BLUE, lw=1.3)
    ax1.set_xscale("log")
    ax1.axvline(20, color=MUTED, lw=0.8, ls="--")
    ax1.annotate("top 20 $\\to$ >99.999%", xy=(20, cum[19] * 100),
                 xytext=(28, 80), fontsize=7.5,
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax1.set_xlabel("reference rank position $i$ (log scale)", fontsize=8)
    ax1.set_ylabel("cumulative share of\ntotal $2^{-i}$ weight (\\%)", fontsize=8)
    ax1.set_ylim(0, 105)
    ax1.set_title("(a) WS weight concentration", fontsize=8.5)

    print(f"Top-20 cumulative weight share at N={N}: {cum[19]*100:.6f}%  "
          f"(paper reports >99.999%)")

    rng = np.random.default_rng(0)
    Nb = 690
    Rx = np.arange(1, Nb + 1, dtype=float)

    def ws(Rx, Ry):
        denom = np.maximum(np.abs(Rx - 1), np.abs(Rx - Nb))
        denom = np.where(denom == 0, 1.0, denom)
        return 1 - ((2.0 ** (-Rx)) * np.abs(Rx - Ry) / denom).sum()

    Ry_a = Rx.copy()
    tail = Ry_a[20:].copy()
    rng.shuffle(tail)
    Ry_a[20:] = tail
    rho_a, tau_a, ws_a = spearman_rho(-Rx, -Ry_a), kendall_tau(-Rx, -Ry_a), ws(Rx, Ry_a)

    Ry_b = Rx.copy()
    idx1, idx2 = np.arange(10), np.arange(300, 310)
    Ry_b[idx1], Ry_b[idx2] = Rx[idx2].copy(), Rx[idx1].copy()
    rho_b, tau_b, ws_b = spearman_rho(-Rx, -Ry_b), kendall_tau(-Rx, -Ry_b), ws(Rx, Ry_b)

    print(f"Scenario A (top 20 exact, rest scrambled):    rho={rho_a:.3f}  "
          f"tau={tau_a:.3f}  WS={ws_a:.6f}")
    print(f"Scenario B (top 10 swapped with ranks ~300-310): rho={rho_b:.3f}  "
          f"tau={tau_b:.3f}  WS={ws_b:.3f}")

    metrics = [r"Spearman $\rho$", r"Kendall $\tau$", "WS"]
    x = np.arange(3)
    width = 0.32
    ax2.bar(x - width / 2, [rho_a, tau_a, ws_a], width, color=BLUE, label="Scenario A")
    ax2.bar(x + width / 2, [rho_b, tau_b, ws_b], width, color=ORANGE, hatch="///",
            edgecolor="white", label="Scenario B")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=8)
    ax2.set_ylim(-0.05, 1.05)
    ax2.axhline(0, color=INK, lw=0.6)
    ax2.set_title("(b) Two disagreement patterns ($N{=}690$)", fontsize=8.5)
    ax2.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    return fig

fig = make_ws_saturation()
save_fig(fig, "figure2_ws_saturation")

# %% [markdown]
# ## 9. Figure — retention-ratio sensitivity across all four datasets
#
# **Produces Figure 3 in the paper** (`fig:p-sensitivity`, opening of Section 5,
# before Section 5.1). Requires all four datasets loaded above (Section 6).

# %%
def p_sweep(X, directions, N, p_grid=None, L=1):
    if p_grid is None:
        p_grid = [round(0.10 + 0.05 * i, 2) for i in range(18)]  # 0.10 .. 0.95 step 0.05
    rows = []
    for p in p_grid:
        r = iqis_select(X, directions, p=p, L=L)
        rows.append((p, r.n_selected, r.stopped_by))
    return rows


def make_p_sensitivity():
    datasets = [
        ("Bhangale (2004), N=7", BHANGALE_X, BHANGALE_DIRECTIONS, 7),
        ("Karsak (2012), N=12", KARSAK_X, KARSAK_DIRECTIONS, 12),
        ("QS Rankings, N=690", QS_X, QS_DIRECTIONS, 690),
        ("NBA 2021--22, N=358", NBA_X, NBA_DIRECTIONS, 358),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.2))
    for ax, (title, X, dirs, N) in zip(axes.flat, datasets):
        rows = p_sweep(X, dirs, N)
        ps = [r[0] for r in rows]
        cs = [r[1] for r in rows]
        stab = [r[2] == "stabilization" for r in rows]
        ax.plot(ps, cs, color=BLUE, lw=1.1, zorder=2)
        stab_p = [p for p, s in zip(ps, stab) if s]
        stab_c = [c for c, s in zip(cs, stab) if s]
        thr_p = [p for p, s in zip(ps, stab) if not s]
        thr_c = [c for c, s in zip(cs, stab) if not s]
        ax.scatter(stab_p, stab_c, marker="o", s=22, color=BLUE, edgecolor=BLUE, zorder=3)
        ax.scatter(thr_p, thr_c, marker="o", s=22, facecolor="white", edgecolor=BLUE,
                   linewidth=1.1, zorder=3)
        ax.axhline(N, color=MUTED, lw=0.7, ls=":")
        ax.text(0.11, N, "N", fontsize=6.5, color=MUTED, va="bottom")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("$p$", fontsize=8)
        ax.set_ylabel("$|C^*|$ (log scale)", fontsize=8)
        ax.set_xlim(0.05, 1.0)
    handles = [Line2D([0], [0], marker="o", color=BLUE, markerfacecolor=BLUE,
                       linestyle="none", markersize=5, label="stabilization"),
               Line2D([0], [0], marker="o", color=BLUE, markerfacecolor="white",
                       linestyle="none", markersize=5, markeredgewidth=1.1,
                       label="threshold violation")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(r"$|C^*|$ as a function of $p$ ($L=1$, equal priority)", fontsize=9.5, y=1.01)
    fig.tight_layout()
    return fig

fig = make_p_sensitivity()
save_fig(fig, "figure3_p_sensitivity")

# %% [markdown]
# ## 10. Section 5.1 — Bhangale (2004): validation against literature consensus
#
# **Produces:**
# - Table `tab:literature-validation-composition` — composition of $C^*$ as a function of $p$
# - Table `tab:literature-validation-p50` — top-ranked candidate per baseline at $p=0.50$
# - Table `tab:literature-validation-comparison` — full agreement table across all four $p$
#
# **Used in the article at:** Section 5.1 ("Validation Against Literature Consensus:
# Industrial Robot Selection"). Bhangale et al. (2004), Rao (2007), and Chatterjee et al.
# (2010) unanimously identify the Cybotech V15 Electric Robot as the best alternative
# using three different published methods; the cell below checks whether the proposed
# method recovers this consensus, and how robust the result is to $p$.

# %%
BHANGALE_P_GRID = [0.50, 0.60, 0.75, 0.80]
bhangale_results = {p: iqis_select(BHANGALE_X, BHANGALE_DIRECTIONS, p=p, L=1) for p in BHANGALE_P_GRID}

rows = []
for p, r in bhangale_results.items():
    T = r.n_iterations
    cstar = [BHANGALE_LABELS[i] for i in r.final_class_indices]
    runner_up = [BHANGALE_LABELS[i] for i in np.flatnonzero(r.classes == T)] if T >= 2 else []
    rows.append({"p": p, "T": T, "|C*|": len(cstar),
                 "C* (class T+1)": ", ".join(cstar),
                 "class T (runner-up)": ", ".join(runner_up) if runner_up else "--",
                 "stopped_by": r.stopped_by})
bhangale_composition = pd.DataFrame(rows).set_index("p")
print("Table tab:literature-validation-composition")
display(bhangale_composition)

consensus = "Cybotech V15 Electric Robot"
print(f"\nLiterature consensus candidate '{consensus}' recovered at every tested p:",
      all(consensus in r.split(", ") for r in bhangale_composition["C* (class T+1)"]))

# %% [markdown]
# Table `tab:literature-validation-p50`: at $p=0.50$, compare each baseline's own
# top-ranked candidate and its agreement with the induced ranking.

# %%
r50 = bhangale_results[0.50]
rows = []
for name, fn in BASELINES_5.items():
    score = fn(BHANGALE_X, BHANGALE_DIRECTIONS)
    top_idx = int(np.argmax(score))
    rows.append({"Method": name, "Top-ranked candidate": BHANGALE_LABELS[top_idx],
                 "Spearman rho": round(spearman_rho(r50.classes, score), 3),
                 "Kendall tau": round(kendall_tau(r50.classes, score), 3),
                 "WS": round(ws_coefficient(r50.classes, score), 3)})
bhangale_p50 = pd.DataFrame(rows).set_index("Method")
print("Table tab:literature-validation-p50")
display(bhangale_p50)

# %% [markdown]
# Table `tab:literature-validation-comparison`: full agreement table (all four $p$,
# all five baselines).

# %%
_bhangale_scores = score_cache(BHANGALE_X, BHANGALE_DIRECTIONS)
bhangale_agreement = {p: agreement_table(r, BHANGALE_X, BHANGALE_DIRECTIONS, baselines=_bhangale_scores) for p, r in bhangale_results.items()}
bhangale_comparison = pd.concat(bhangale_agreement, names=["p", "Method"])
print("Table tab:literature-validation-comparison")
display(bhangale_comparison)

# %% [markdown]
# ## 11. Section 5.2 — Karsak (2012): illustrative example and parameter sensitivity
#
# **Produces:**
# - Table `tab:illustrative-composition` — composition of $C^*$ as a function of $p$
# - Table `tab:illustrative-comparison` — full agreement table across all four $p$
# - Table `tab:illustrative-ranks` — rank position of the $C^*$ candidates (plus R12)
#   within each baseline's own full ranking
#
# **Used in the article at:** Section 5.2 ("Illustrative Numerical Example").
# This section works through the method's behavior on a single dataset as $p$
# varies, and is where the paper makes its main non-monotonicity-across-runs
# argument (Remark `rem:monotonicity` concerns a single run, not $C^*$ across
# independent runs at different $p$).

# %%
KARSAK_P_GRID = [0.60, 0.70, 0.80, 0.85]
karsak_results = {p: iqis_select(KARSAK_X, KARSAK_DIRECTIONS, p=p, L=1) for p in KARSAK_P_GRID}

rows = []
for p, r in karsak_results.items():
    T = r.n_iterations
    cstar = [KARSAK_LABELS[i] for i in r.final_class_indices]
    runner_up = [KARSAK_LABELS[i] for i in np.flatnonzero(r.classes == T)] if T >= 2 else []
    rows.append({"p": p, "T": T, "|C*|": len(cstar),
                 "C* (class T+1)": ", ".join(cstar),
                 "class T (runner-up)": ", ".join(runner_up) if runner_up else "--",
                 "stopped_by": r.stopped_by})
karsak_composition = pd.DataFrame(rows).set_index("p")
print("Table tab:illustrative-composition")
display(karsak_composition)

sizes = karsak_composition["|C*|"].tolist()
print(f"\n|C*| sequence across p={KARSAK_P_GRID}: {sizes}  "
      f"(paper reports 1, 3, 2, 4 -- not monotone in p, discussed in the text)")

# %% [markdown]
# Table `tab:illustrative-comparison`: full agreement table (all four $p$, all five
# baselines).

# %%
_karsak_scores = score_cache(KARSAK_X, KARSAK_DIRECTIONS)
karsak_agreement = {p: agreement_table(r, KARSAK_X, KARSAK_DIRECTIONS, baselines=_karsak_scores) for p, r in karsak_results.items()}
karsak_comparison = pd.concat(karsak_agreement, names=["p", "Method"])
print("Table tab:illustrative-comparison")
display(karsak_comparison)

# %% [markdown]
# Table `tab:illustrative-ranks`: rank position (1 = best of 12) of the candidates
# appearing in $C^*$ across all four $p$ (plus R12, the unanimous top pick of four
# of the five baselines that is nonetheless excluded from $C^*$ at every $p$
# tested), within each baseline's own parameter-independent full ranking.
#
# **Reproducibility note.** Borda's total rank-sum on this matrix has two *exact*
# ties -- R2 and R12 both sum to 25, R3 and R11 both sum to 28 -- which is a
# genuine feature of the data (confirmed below), not a bug: no single-criterion
# tie forces it, the two totals simply coincide. `scipy.stats.rankdata` with
# `method="average"` (used everywhere else in this notebook, including inside
# `spearman_rho`/`kendall_tau`) reports these as fractional ranks (e.g.\ 1.5)
# rather than silently picking a tie-break, which is why the Borda column below
# can show `1.5`/`3.5` where the printed article's table shows whole numbers
# (R12: article `2`, computed `1.5`; R11: article `3`, computed `3.5`). This does
# not affect any interpretive claim in the surrounding text -- VIKOR ranking R9
# 4th and WSM ranking R2 6th, the two facts the prose actually relies on, both
# reproduce exactly (see the printed ties check below).

# %%
ranks_candidates = ["R2", "R7", "R9", "R11", "R12"]
karsak_borda_score = BASELINES_5["Borda"](KARSAK_X, KARSAK_DIRECTIONS)
tied_groups = (pd.Series(-karsak_borda_score, index=KARSAK_LABELS)
               .round(6).reset_index().groupby(0)["index"].apply(list))
print("Exact ties in the Borda rank-sum (rank-sum -> tied candidates):")
for rank_sum, members in tied_groups.items():
    if len(members) > 1:
        print(f"  {rank_sum}: {members}")

rows = []
for cand in ranks_candidates:
    idx = KARSAK_LABELS.index(cand)
    row = {"Candidate": cand}
    for name, fn in BASELINES_5.items():
        score = fn(KARSAK_X, KARSAK_DIRECTIONS)
        rank_of_idx = float(stats.rankdata(-score, method="average")[idx])
        row[name] = rank_of_idx
    rows.append(row)
karsak_ranks = pd.DataFrame(rows).set_index("Candidate")
print("\nTable tab:illustrative-ranks (average-rank tie convention; see note above)")
display(karsak_ranks)

# %% [markdown]
# ## 12. Section 5.3 — ELECTRE TRI-B: a sorting-based comparison
#
# **Produces:**
# - Table `tab:electre-tri` — ELECTRE TRI-B's top category vs. the proposed
#   method's $C^*$, at the same $p$ used for IQIS on each dataset, on both small
#   benchmark matrices (Bhangale, Karsak)
#
# **Used in the article at:** Section 5.3 ("ELECTRE TRI-B: A Sorting-Based
# Comparison"). ELECTRE TRI-B sorts each candidate into one of two categories
# against a single boundary profile $B$, placed criterion-by-criterion at the
# $(1-p)$-quantile of the benefit-oriented data -- the same retention ratio $p$
# already tested for the proposed method on each dataset, so the two methods'
# "good" sets are compared at matched nominal selectivity. This offers a more
# direct structural comparison to the proposed method's own binary good/not-good
# partition than treating ELECTRE III as a sixth ranking baseline does.

# %%
def electre_tri_comparison(results_by_p, X, directions, labels):
    """For each p already used for IQIS on this dataset, compare C* against
    ELECTRE TRI-B's top category at the matching (1-p)-quantile boundary
    profile, via Jaccard overlap -- exactly Table tab:electre-tri."""
    rows = []
    for p, r in results_by_p.items():
        cstar = set(r.final_class_indices)
        tri_good = set(np.flatnonzero(electre_tri_b_good_set(X, directions, p)))
        jaccard = len(cstar & tri_good) / len(cstar | tri_good) if (cstar | tri_good) else float("nan")
        rows.append({"p": p, "|C*|": len(cstar), "|TRI-B top category|": len(tri_good),
                     "Jaccard overlap": round(jaccard, 2)})
    return pd.DataFrame(rows).set_index("p")

electre_tri_bhangale = electre_tri_comparison(bhangale_results, BHANGALE_X, BHANGALE_DIRECTIONS, BHANGALE_LABELS)
electre_tri_karsak = electre_tri_comparison(karsak_results, KARSAK_X, KARSAK_DIRECTIONS, KARSAK_LABELS)

electre_tri_comparison_table = pd.concat({"Bhangale": electre_tri_bhangale, "Karsak": electre_tri_karsak},
                                          names=["Dataset", "p"])
print("Table tab:electre-tri")
display(electre_tri_comparison_table)
print("\nPaper values -- Bhangale: |C*|=[1,2,2,4], |TRI-B|=[2,3,4,4], Jaccard=[0.50,0.67,0.50,1.00]")
print("Paper values -- Karsak:   |C*|=[1,3,2,4], |TRI-B|=[2,6,9,9], Jaccard=[0.50,0.50,0.22,0.44]")

# %% [markdown]
# ## 13. Section 5.4 — Rank-reversal test: perturbing the Karsak matrix
#
# **Produces:**
# - Table `tab:rank-reversal` — induced ranking before and after removing R2 and
#   R12 and adding their elementwise-average hybrid, R\_hybrid
# - The baseline-consensus check: does R\_hybrid become the new top pick of
#   every compensatory/outranking baseline on the perturbed matrix?
#
# **Used in the article at:** Section 5.4 ("Rank-Reversal Test: Perturbing the
# Karsak Matrix"). Because the per-iteration admission threshold is recomputed
# relative to the *current* population (Remark `rem:population-change`), adding
# or removing a candidate can in principle alter $C^*$; this section tests that
# directly with a Triantaphyllou (2000)-style rank-reversal test adapted to
# remove **two** candidates -- R2 and R12, whose divergent treatment by the
# proposed method and the compensatory/outranking baselines structures the
# preceding discussion -- and replace them with a constructed hybrid, R\_hybrid,
# defined as their elementwise average.

# %%
r2_idx = KARSAK_LABELS.index("R2")
r12_idx = KARSAK_LABELS.index("R12")
keep_idx = [i for i in range(len(KARSAK_LABELS)) if i not in (r2_idx, r12_idx)]

rr_hybrid = (KARSAK_X[r2_idx] + KARSAK_X[r12_idx]) / 2.0
RR_X = np.vstack([KARSAK_X[keep_idx], rr_hybrid[None, :]])
RR_LABELS = [KARSAK_LABELS[i] for i in keep_idx] + ["R_hybrid"]

print("Removed R2:", dict(zip(KARSAK_COLUMNS, KARSAK_X[r2_idx])))
print("Removed R12:", dict(zip(KARSAK_COLUMNS, KARSAK_X[r12_idx])))
print("R_hybrid (elementwise average):", dict(zip(KARSAK_COLUMNS, rr_hybrid.round(3))))
print(f"Perturbed matrix: N={RR_X.shape[0]} (was 12)")

rr_results = {p: iqis_select(RR_X, KARSAK_DIRECTIONS, p=p, L=1) for p in KARSAK_P_GRID}

# %% [markdown]
# Table `tab:rank-reversal`: $C^*$ before and after perturbation, and Kendall's
# $\tau$ on the 10 robots common to both matrices (comparing each one's
# elimination class before vs. after -- undefined at $p=0.60$, where the
# perturbed run collapses to the single-member class {R\_hybrid}, leaving no
# variance to compute $\tau$ against).

# %%
rows = []
for p in KARSAK_P_GRID:
    orig, new = karsak_results[p], rr_results[p]
    cstar_orig = [KARSAK_LABELS[i] for i in orig.final_class_indices]
    cstar_new = [RR_LABELS[i] for i in new.final_class_indices]
    common = [l for l in RR_LABELS if l != "R_hybrid"]
    orig_class = {KARSAK_LABELS[i]: orig.classes[i] for i in range(12)}
    new_class = {RR_LABELS[i]: new.classes[i] for i in range(len(RR_LABELS))}
    orig_order = [orig_class[l] for l in common]
    new_order = [new_class[l] for l in common]
    if len(set(new_order)) < 2 or len(set(orig_order)) < 2:
        tau_str = "n/a (degenerate)"
    else:
        tau, _ = stats.kendalltau(orig_order, new_order)
        tau_str = f"{tau:.3f}"
    rows.append({"p": p, "C* original (12 robots)": ", ".join(cstar_orig),
                 "C* perturbed (11 robots)": ", ".join(cstar_new),
                 "Kendall tau, 10 common robots": tau_str})
rank_reversal_table = pd.DataFrame(rows).set_index("p")
print("Table tab:rank-reversal")
display(rank_reversal_table)
print("\nPaper values: tau = n/a, 0.667, 1.000, 0.923 at p = 0.60, 0.70, 0.80, 0.85")

# %% [markdown]
# Baseline-side check: does R\_hybrid become the new unanimous top pick of the
# compensatory/outranking baselines on the perturbed matrix? (Paper: yes for
# TOPSIS, VIKOR, WSM, PROMETHEE~II, and ELECTRE~III; only Borda still prefers a
# different candidate, R3.)

# %%
_rr_scores = score_cache(RR_X, KARSAK_DIRECTIONS)
rows = []
for name, score in _rr_scores.items():
    top = RR_LABELS[int(np.argmax(score))]
    hyb_rank = int(stats.rankdata(-score, method="average")[RR_LABELS.index("R_hybrid")])
    rows.append({"Method": name, "Top-ranked candidate": top, "R_hybrid rank": hyb_rank})
rank_reversal_baselines = pd.DataFrame(rows).set_index("Method")
print(f"(of {len(RR_LABELS)} candidates)")
display(rank_reversal_baselines)

# %% [markdown]
# ## 14. Section 5.5 — QS World University Rankings: large-scale validation
#
# **Produces:**
# - Table `tab:qs-composition` — composition of $C^*$ as a function of $p$
# - Table `tab:qs-mechanism` — worst per-criterion rank (out of 690) for the
#   Overall-SCORE top-3 versus two members of $C^*$
# - Table `tab:qs-comparison` — full agreement table across all four $p$
# - The "Overall SCORE direct comparison" figures quoted in the text
#   ($\rho=0.273\to0.693$) and the skyline size (16 of 690)
#
# **Used in the article at:** Section 5.5 ("Large-Scale Validation: QS World
# University Rankings"). This is the first of the two large-$N$ real-world
# datasets and the section that isolates the non-compensatory mechanism most
# clearly (a single weak sub-indicator disqualifying an otherwise top-ranked
# university).

# %%
QS_P_GRID = [0.30, 0.50, 0.70, 0.90]
qs_results = {p: iqis_select(QS_X, QS_DIRECTIONS, p=p, L=1) for p in QS_P_GRID}

rows = []
for p, r in qs_results.items():
    cstar = [QS_LABELS[i] for i in r.final_class_indices]
    preview = ", ".join(cstar[:6]) + (", ..." if len(cstar) > 6 else "")
    rows.append({"p": p, "T": r.n_iterations, "|C*|": len(cstar),
                 "C* (class T+1), first 6": preview, "stopped_by": r.stopped_by})
qs_composition = pd.DataFrame(rows).set_index("p")
print("Table tab:qs-composition")
display(qs_composition)

imperial_name = None
for p in (0.70,):  # C* is a singleton at p=0.70 in the paper
    cstar = [QS_LABELS[i] for i in qs_results[p].final_class_indices]
    if len(cstar) == 1:
        imperial_name = cstar[0]
print(f"\nSingleton C* at p=0.70: {imperial_name!r} (paper: 'Imperial College London')")
always_present = all(imperial_name in [QS_LABELS[i] for i in qs_results[p].final_class_indices]
                      for p in QS_P_GRID)
print(f"Present in C* at every tested p: {always_present}")

# %% [markdown]
# Table `tab:qs-mechanism`. The three "Overall-SCORE top-3" universities are read
# off directly from the dataset's own published `Overall SCORE` column, not
# hardcoded, so this cell degrades gracefully if a different release of the CSV
# changes the top 3. University of Cambridge is read off the already-computed
# $C^*$ set at $p=0.90$ rather than hardcoded.
#
# **Reproducibility note.** The article's Section 5.5 prose names "MIT, Stanford,
# and Harvard" as the three highest-scoring universities on QS's own Overall
# SCORE. Reading that column directly off `data/qs_university_rankings_2026.csv`
# (printed below) instead gives **MIT, Imperial College London, Stanford** --
# Imperial ranks *2nd* on Overall SCORE (99.4), ahead of Stanford (98.9), Oxford
# (97.9), and Harvard, which is actually 5th (97.7). This matters for the
# article's argument, not just its wording: Imperial is precisely the method's
# own singleton pick at $p=0.70$, so it cannot also serve as an example of a
# top-Overall-SCORE university *excluded* from $C^*$. This looks like a genuine
# mismatch between the prose and the cited dataset rather than a notebook bug --
# the ranking above matches the real QS World University Rankings 2026 release
# (Imperial's rise to 2nd place was widely reported). The cell below reports the
# universities exactly as the data gives them; the corresponding LaTeX prose and
# Table `tab:qs-mechanism` should be revised to match (e.g.\ using Oxford in
# place of Harvard, and dropping Imperial from the "excluded" list since it is
# not excluded).

# %%
def per_criterion_ranks(X, directions):
    """Rank of every candidate on every criterion, 1 = best, ties broken by
    stable sort order (matches the ranking convention used by iqis_select)."""
    signs = _signs(directions)
    N, K = X.shape
    ranks = np.empty((N, K), dtype=int)
    for k in range(K):
        order = np.argsort(-signs[k] * X[:, k], kind="stable")
        r = np.empty(N, dtype=int)
        r[order] = np.arange(1, N + 1)
        ranks[:, k] = r
    return ranks

qs_ranks = per_criterion_ranks(QS_X, QS_DIRECTIONS)

overall_top5_idx = np.argsort(-QS_OVERALL_SCORE)[:5]
print("QS Overall SCORE top 5 (article claims top 3 = MIT, Stanford, Harvard):")
for rank, i in enumerate(overall_top5_idx, start=1):
    print(f"  {rank}. {QS_LABELS[i]}  (Overall SCORE = {QS_OVERALL_SCORE[i]:.1f})")
overall_top3_names = [QS_LABELS[i] for i in overall_top5_idx[:3]]

cambridge_name = next(n for n in [QS_LABELS[i] for i in qs_results[0.90].final_class_indices]
                       if "Cambridge" in n)

mechanism_names = list(dict.fromkeys(overall_top3_names + [cambridge_name]))  # de-duplicated, order-preserving
rows = []
for name in mechanism_names:
    idx = QS_LABELS.index(name)
    worst_k = int(np.argmax(qs_ranks[idx]))
    rows.append({"University": name, "Worst rank": int(qs_ranks[idx, worst_k]),
                 "Criterion attaining it": QS_CRITERIA_COLUMNS[worst_k]})
qs_mechanism = pd.DataFrame(rows).set_index("University")
print("Table tab:qs-mechanism")
display(qs_mechanism)

# %% [markdown]
# Table `tab:qs-comparison`: full agreement table (all four $p$, all five
# baselines). Unlike Karsak (2012), all six QS criteria are non-negative SCORE
# values, so WSM's sum-normalization succeeds here (no skipped baseline).

# %%
_qs_scores = score_cache(QS_X, QS_DIRECTIONS)  # ELECTRE III alone takes ~1 minute on N=690; compute once
qs_agreement = {p: agreement_table(r, QS_X, QS_DIRECTIONS, baselines=_qs_scores) for p, r in qs_results.items()}
qs_comparison = pd.concat(qs_agreement, names=["p", "Method"])
print("Table tab:qs-comparison")
display(qs_comparison)

# %% [markdown]
# Direct comparison against QS's own published `Overall SCORE` (rather than
# against a baseline recomputed from the six raw criteria), and the classical
# Pareto skyline size quoted in the text.

# %%
for p in (0.30, 0.90):
    r = qs_results[p]
    rho_o = spearman_rho(r.classes, QS_OVERALL_SCORE)
    tau_o = kendall_tau(r.classes, QS_OVERALL_SCORE)
    ws_o = ws_coefficient(r.classes, QS_OVERALL_SCORE)
    print(f"p={p}: rho={rho_o:.3f}  tau={tau_o:.3f}  WS={ws_o:.3f}  "
          f"(paper: rho=0.273 at p=0.30, rho=0.693 at p=0.90)")

qs_skyline = pareto_front(QS_X, QS_DIRECTIONS)
print(f"\nQS classical Pareto skyline: {len(qs_skyline)} of {len(QS_LABELS)} "
      f"universities (paper: 16 of 690)")

# %% [markdown]
# ## 15. Section 5.6 — NBA player performance: large-scale validation, monotonicity
# caveat, and runtime complexity
#
# **Produces:**
# - Table `tab:nba-composition` — composition of $C^*$ as a function of $p$
# - The fine-grid non-monotone $|C^*|$ sequence quoted in the text
# - Table `tab:nba-comparison` — agreement table across all four $p$ (WSM skipped:
#   net rating takes negative values, so pymcdm's WSM implementation raises)
# - Skyline size (26 of 358) and the non-containment specifics ($C^*\not\subseteq$
#   skyline at $p=0.30$ and $p=0.90$)
# - **Figure 4** (`fig:nba-class-scatter`) — elimination class vs. skyline scatter
# - Table `tab:nba-complexity` and **Figure 5** (`fig:nba-complexity`) — empirical
#   runtime on the full multi-season panel
#
# **Used in the article at:** Section 5.6 ("Large-Scale Validation: NBA Player
# Performance and a Cautionary Note on Monotonicity"). This is the only section
# where the cautionary methodological point about $|C^*|$ non-comparability
# *across* independent runs at different $p$ is demonstrated on a fine grid, and
# the only dataset where $C^*$ is not always a subset of the classical skyline.

# %%
NBA_P_GRID = [0.30, 0.50, 0.70, 0.90]
nba_results = {p: iqis_select(NBA_X, NBA_DIRECTIONS, p=p, L=1) for p in NBA_P_GRID}

rows = []
for p, r in nba_results.items():
    cstar = [NBA_LABELS[i] for i in r.final_class_indices]
    rows.append({"p": p, "T": r.n_iterations, "|C*|": len(cstar),
                 "C* (class T+1)": ", ".join(cstar), "stopped_by": r.stopped_by})
nba_composition = pd.DataFrame(rows).set_index("p")
print("Table tab:nba-composition")
display(nba_composition)

# %% [markdown]
# The fine-grid non-monotonicity claim: over $p\in\{0.35,0.40,0.55,0.60,0.70,0.75,
# 0.80,0.85,0.90,0.95\}$ (the step-$0.05$ values at which this dataset happens to
# terminate by stabilization rather than threshold violation), $|C^*|$ dips from
# $3$ back to $2$ between $p=0.70$ and $p=0.75$ before resuming its increase --
# restricting to stabilization-terminated runs removes one confound (the
# threshold-violation snapshot artifact) but does not by itself restore
# monotonicity, because the per-iteration threshold is recomputed against the
# population still alive at each step.

# %%
fine_grid = [round(0.35 + 0.05 * i, 2) for i in range(13)]  # 0.35 .. 0.95
fine_rows = []
for p in fine_grid:
    r = iqis_select(NBA_X, NBA_DIRECTIONS, p=p, L=1)
    fine_rows.append({"p": p, "|C*|": r.n_selected, "stopped_by": r.stopped_by})
fine_df = pd.DataFrame(fine_rows)
stabilized = fine_df[fine_df["stopped_by"] == "stabilization"]
print("Stabilization-terminated runs only:")
display(stabilized.set_index("p"))
print(f"\n|C*| sequence (stabilization-terminated only): {stabilized['|C*|'].tolist()}")
print("(paper reports 1, 1, 2, 2, 3, 2, 3, 4, 7, 18 over p in "
      "{0.35,0.40,0.55,0.60,0.70,0.75,0.80,0.85,0.90,0.95} -- exact grid depends on "
      "which values terminate by stabilization, printed above)")

# %% [markdown]
# Table `tab:nba-comparison`: WSM is expected to be skipped here (net rating is
# negative for a majority of players; pymcdm's WSM implementation requires a
# strictly positive input matrix). `agreement_table` prints a `[skipped]` line
# rather than silently omitting it.

# %%
_nba_scores = score_cache(NBA_X, NBA_DIRECTIONS)
nba_agreement = {p: agreement_table(r, NBA_X, NBA_DIRECTIONS, baselines=_nba_scores) for p, r in nba_results.items()}
nba_comparison = pd.concat(nba_agreement, names=["p", "Method"])
print("Table tab:nba-comparison")
display(nba_comparison)

# %% [markdown]
# Skyline size and the non-containment specifics: unlike the two small benchmark
# matrices and the QS dataset, $C^*$ is *not* always a subset of the skyline here.

# %%
nba_skyline_idx = pareto_front(NBA_X, NBA_DIRECTIONS)
nba_skyline_set = set(nba_skyline_idx.tolist())
print(f"NBA classical Pareto skyline: {len(nba_skyline_idx)} of {len(NBA_LABELS)} "
      f"players (paper: 26 of 358)")

for p, r in nba_results.items():
    not_on_sky = [NBA_LABELS[i] for i in r.final_class_indices if i not in nba_skyline_set]
    print(f"p={p}: members of C* NOT on the skyline: {not_on_sky if not_on_sky else '(none)'}")

# %% [markdown]
# **Figure 4** (`fig:nba-class-scatter`, produced here even though it is *Figure 4*
# in the compiled paper): elimination class vs. skyline membership, plotted
# against two of the five criteria (points and net rating per game); stars mark
# $C^*$ at $p=0.70$.

# %%
def make_nba_class_scatter():
    p = 0.70
    r = nba_results[p]
    classes = r.classes
    sky = nba_skyline_set
    cstar_idx = set(r.final_class_indices.tolist())

    x = NBA_X[:, NBA_CRITERIA_COLUMNS.index("pts")]
    y = NBA_X[:, NBA_CRITERIA_COLUMNS.index("net_rating")]
    labels = NBA_LABELS

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    is_sky = np.array([i in sky for i in range(len(labels))])

    ax.scatter(x[~is_sky], y[~is_sky], c=classes[~is_sky], cmap=SEQ_BLUE,
               vmin=1, vmax=classes.max(), s=20, edgecolor="none", zorder=2,
               label="not on skyline")
    sc = ax.scatter(x[is_sky], y[is_sky], c=classes[is_sky], cmap=SEQ_BLUE,
                     vmin=1, vmax=classes.max(), s=32, edgecolor=INK, linewidth=0.9,
                     zorder=3, label="on skyline")

    label_offset = {
        "Nikola Jokic": (-100, 2, "left"),
        "Giannis Antetokounmpo": (-70, 11, "left"),
        "Kevin Durant": (-58, -15, "left"),
    }
    for i in cstar_idx:
        dx, dy, ha = label_offset.get(labels[i], (4, 4, "left"))
        ax.annotate(labels[i], (x[i], y[i]), fontsize=6.5, xytext=(dx, dy),
                    textcoords="offset points", zorder=4, ha=ha)
        ax.scatter([x[i]], [y[i]], marker="*", s=140, facecolor="none",
                   edgecolor=ORANGE, linewidth=1.3, zorder=5)

    ax.set_xlim(x.min() - 1.5, x.max() + 2.5)
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(r"$\operatorname{class}(c)$  (higher = survives longer)", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5)
    ax.set_xlabel("points per game")
    ax.set_ylabel("net rating")
    ax.set_title(rf"NBA 2021--22 ($N{{=}}358$): elimination class vs. skyline"
                 f"\n(stars: $C^*$ at $p={p}$, black-edged: skyline)", fontsize=8.5)
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    fig.tight_layout()
    return fig

fig = make_nba_class_scatter()
save_fig(fig, "figure4_nba_class_scatter")

# %% [markdown]
# Table `tab:nba-complexity` and **Figure 5** (`fig:nba-complexity`, *Figure 5* in
# the compiled paper): empirical running time on subsamples of the full
# multi-season NBA panel ($N_{\max}=12{,}844$, loaded as `NBA_FULL_X` in Section 6),
# $p=0.7$, $L=1$, median of 3--5 repetitions, against a reference $N\log_2 N$
# curve. Absolute timings depend on the machine executing this notebook; the
# claim the paper draws from this table is the qualitative one -- the normalized
# quantity time$/(N\log_2 N)$ stays within a small factor across three orders of
# magnitude of $N$, not the literal millisecond values.

# %%
import time as _time

def nba_complexity_benchmark():
    rng = np.random.default_rng(42)
    Nfull = NBA_FULL_X.shape[0]
    sizes = sorted(set(min(s, Nfull) for s in [200, 500, 1000, 2000, 4000, 8000, Nfull]))
    Ns, times_ms = [], []
    for n in sizes:
        idx = rng.choice(Nfull, size=n, replace=False)
        Xs = NBA_FULL_X[idx]
        reps = 5 if n <= 2000 else 3
        ts = []
        for _ in range(reps):
            t0 = _time.perf_counter()
            iqis_select(Xs, NBA_DIRECTIONS, p=0.7, L=1)
            ts.append(_time.perf_counter() - t0)
        Ns.append(n)
        times_ms.append(float(np.median(ts)) * 1000)
    return np.array(Ns, dtype=float), np.array(times_ms)

nba_complexity_N, nba_complexity_time_ms = nba_complexity_benchmark()
# NOTE: the paper's Table tab:nba-complexity displays "Time (ms)" but the
# adjacent "Time / (N log2 N)" column is computed with time in SECONDS, not
# milliseconds (confirmed by back-solving the paper's own published values --
# e.g. N=200: 0.23 ms = 0.00023 s, and 0.00023 / (200*log2(200)) = 1.50e-7,
# matching the paper's printed 1.49e-7 to within rounding). Using milliseconds
# here instead, as an earlier version of this cell did, inflates every entry in
# this column by a factor of 1000 relative to the paper -- reproduced with
# seconds below to match the paper's own table exactly.
nba_complexity_time_s = nba_complexity_time_ms / 1000.0
nba_complexity_table = pd.DataFrame({
    "N": nba_complexity_N.astype(int),
    "Time (ms)": np.round(nba_complexity_time_ms, 2),
    "Time (s) / (N log2 N)": nba_complexity_time_s / (nba_complexity_N * np.log2(nba_complexity_N)),
}).set_index("N")
print("Table tab:nba-complexity (this machine's timings -- see markdown note above)")
display(nba_complexity_table)

ratio = nba_complexity_table["Time (s) / (N log2 N)"]
print(f"\nmax/min ratio of the normalized quantity: {ratio.max() / ratio.min():.2f}x "
      f"(paper reports 4.3x on its own hardware; this ratio is unaffected by the "
      f"ms-vs-s choice above -- only the column's absolute scale was. The "
      f"qualitative claim is 'stays within a small factor', not the exact ratio)")

# %%
def make_nba_complexity_figure(Ns, times_ms):
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.loglog(Ns, times_ms, "o-", color=BLUE, lw=1.2, markersize=5,
              label="measured (median of 3--5)")
    ref = Ns * np.log2(Ns)
    ref = ref / ref[-1] * times_ms[-1]
    ax.loglog(Ns, ref, "--", color=MUTED, lw=1.1, label=r"$\propto N\log_2 N$ (reference)")
    ax.set_xlabel("$N$ (log scale)")
    ax.set_ylabel("time (ms, log scale)")
    ax.set_title("Empirical running time, full NBA panel\n(1996--2022, $p{=}0.7$, $L{=}1$)", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    return fig

fig = make_nba_complexity_figure(nba_complexity_N, nba_complexity_time_ms)
save_fig(fig, "figure5_nba_complexity")

# %% [markdown]
# ## 16. Section 5.7 — Cross-dataset statistical synthesis
#
# **Produces:**
# - **Figure 6** (`fig:cross-dataset-heatmap`) — $\rho$ and WS for all sixteen
#   (dataset, $p$) combinations and five baselines
# - Table `tab:cross-dataset-ranks` — average rank across all 16 combinations,
#   restricted to the four baselines available on every dataset, with a Friedman
#   test
# - Table `tab:cross-dataset-wsm` — average rank across the 12 combinations where
#   WSM is defined, all five baselines, with a Friedman test
#
# **Used in the article at:** Section 5.7 ("Cross-Dataset Statistical Synthesis").
# Unlike `generate_figures.py`'s original authoring script, every number in this
# section is computed live from the four `*_agreement` dictionaries built in
# Sections 10, 11, 14, and 15 above (`bhangale_agreement`, `karsak_agreement`,
# `qs_agreement`, `nba_agreement`) -- nothing here is hardcoded, so this section
# is a genuine, self-contained reproduction of Table `tab:cross-dataset-ranks`
# and Table `tab:cross-dataset-wsm`, not merely a redrawing of numbers copied
# from the paper.

# %%
AGREEMENT_BY_DATASET = {
    "Bhangale": bhangale_agreement,
    "Karsak": karsak_agreement,
    "QS": qs_agreement,
    "NBA": nba_agreement,
}

def build_cross_dataset_long(agreement_by_dataset):
    """Pool every (dataset, p) agreement table built above into one long
    DataFrame indexed by dataset, p, Method, with columns Spearman rho / Kendall
    tau / WS. This is the pooled table underlying both Figure 6 and Tables
    tab:cross-dataset-ranks / tab:cross-dataset-wsm."""
    frames = []
    for ds, by_p in agreement_by_dataset.items():
        for p, df in by_p.items():
            f = df.reset_index().copy()
            f.insert(0, "p", p)
            f.insert(0, "dataset", ds)
            frames.append(f)
    return pd.concat(frames, ignore_index=True)

cross_dataset_long = build_cross_dataset_long(AGREEMENT_BY_DATASET)
print(f"Pooled {cross_dataset_long[['dataset', 'p']].drop_duplicates().shape[0]} "
      f"(dataset, p) combinations (paper: 16).")
display(cross_dataset_long.head())

# %% [markdown]
# **Figure 6**: heatmap of $\rho$ (top) and WS (bottom) across all sixteen
# combinations, columns ordered dataset-by-dataset exactly as in the paper.

# %%
def make_cross_dataset_heatmap(long_df):
    baselines = ["TOPSIS", "VIKOR", "WSM", "PROMETHEE II", "Borda", "ELECTRE III"]
    dataset_p_order = [
        ("Bhangale", 0.50), ("Bhangale", 0.60), ("Bhangale", 0.75), ("Bhangale", 0.80),
        ("Karsak", 0.60), ("Karsak", 0.70), ("Karsak", 0.80), ("Karsak", 0.85),
        ("QS", 0.30), ("QS", 0.50), ("QS", 0.70), ("QS", 0.90),
        ("NBA", 0.30), ("NBA", 0.50), ("NBA", 0.70), ("NBA", 0.90),
    ]
    col_labels = [f"{ds}\np={p:.2f}" for ds, p in dataset_p_order]

    idx = long_df.set_index(["dataset", "p", "Method"])
    rho_mat, ws_mat = [], []
    for b in baselines:
        rho_row, ws_row = [], []
        for ds, p in dataset_p_order:
            try:
                cell = idx.loc[(ds, p, b)]
                rho_row.append(cell["Spearman rho"])
                ws_row.append(cell["WS"])
            except KeyError:
                rho_row.append(np.nan)
                ws_row.append(np.nan)
        rho_mat.append(rho_row)
        ws_mat.append(ws_row)
    rho_mat = np.array(rho_mat, dtype=float)
    ws_mat = np.array(ws_mat, dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 6.3), sharex=True)
    for ax, mat, title, vmin, vmax in [
        (ax1, rho_mat, r"Spearman $\rho$ vs induced ranking", 0.0, 0.9),
        (ax2, ws_mat, "WS coefficient vs induced ranking", 0.6, 1.0),
    ]:
        masked = np.ma.masked_invalid(mat)
        im = ax.imshow(masked, aspect="auto", cmap=SEQ_BLUE, vmin=vmin, vmax=vmax)
        ax.set_yticks(range(len(baselines)))
        ax.set_yticklabels(baselines, fontsize=7.5)
        for r_ in range(mat.shape[0]):
            for c_ in range(mat.shape[1]):
                if np.isnan(mat[r_, c_]):
                    ax.text(c_, r_, "n/a", ha="center", va="center", fontsize=5.5, color=MUTED)
                else:
                    ax.text(c_, r_, f"{mat[r_, c_]:.2f}", ha="center", va="center", fontsize=5.5,
                             color="white" if (mat[r_, c_] - vmin) / (vmax - vmin) > 0.55 else INK)
        cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        cb.ax.tick_params(labelsize=6)
        ax.set_title(title, fontsize=8, loc="left")
        for x in [3.5, 7.5, 11.5]:
            ax.axvline(x, color="white", lw=1.4)

    ax2.set_xticks(range(len(col_labels)))
    ax2.set_xticklabels(col_labels, fontsize=5.3, rotation=90)
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.subplots_adjust(top=0.91, hspace=0.32)
    fig.text(0.5, 0.97,
              "Agreement between the induced ranking and six baselines, all 16 (dataset, $p$) combinations",
              ha="center", fontsize=9)
    return fig

fig = make_cross_dataset_heatmap(cross_dataset_long)
save_fig(fig, "figure6_cross_dataset_heatmap")

# %% [markdown]
# Tables `tab:cross-dataset-ranks` and `tab:cross-dataset-wsm`: within each
# (dataset, $p$) block, baselines are ranked $1$ (closest to the induced ranking,
# i.e.\ highest correlation/WS) to worst, separately per metric, then averaged
# across blocks. `scipy.stats.friedmanchisquare` is run on the raw metric values
# per baseline across blocks (it ranks internally; the chi-square statistic is
# invariant to the ascending/descending convention used for the *reported*
# average ranks above).

# %%
def average_ranks_and_friedman(long_df, metric_col, baselines):
    """Restrict to blocks where every baseline in `baselines` has a value for
    `metric_col` (this is what naturally drops NBA when WSM is included), rank
    baselines 1=best (highest value) within each block, and run a Friedman test
    on the raw per-baseline values across blocks."""
    wide = long_df.pivot_table(index=["dataset", "p"], columns="Method", values=metric_col)
    wide = wide[baselines].dropna(how="any")
    ranks = wide.rank(axis=1, ascending=False, method="average")
    avg_rank = ranks.mean(axis=0)
    chi2, pval = friedmanchisquare(*[wide[b].to_numpy() for b in baselines])
    return avg_rank, chi2, pval, wide

BASELINES_5E = ["TOPSIS", "VIKOR", "PROMETHEE II", "Borda", "ELECTRE III"]
metrics = [("Spearman rho", "Avg. rank (rho)"), ("Kendall tau", "Avg. rank (tau)"), ("WS", "Avg. rank (WS)")]

rows = {b: {} for b in BASELINES_5E}
friedman_row = {}
wide_by_metric = {}
for metric_col, out_col in metrics:
    avg_rank, chi2, pval, wide = average_ranks_and_friedman(cross_dataset_long, metric_col, BASELINES_5E)
    wide_by_metric[metric_col] = wide
    for b in BASELINES_5E:
        rows[b][out_col] = round(float(avg_rank[b]), 3)
    friedman_row[out_col] = (round(chi2, 3), pval)

cross_dataset_ranks = pd.DataFrame(rows).T
print("Table tab:cross-dataset-ranks (5 baselines incl. ELECTRE III, all 16 combinations)")
display(cross_dataset_ranks)
for out_col, (chi2, pval) in friedman_row.items():
    print(f"Friedman chi2 (df=4) on {out_col}: {chi2}  p-value: {pval:.3e}")

# %% [markdown]
# **Nemenyi post-hoc test** (Section 5.7 of the paper): the omnibus Friedman
# test above does not by itself license pairwise claims between baselines;
# `scikit_posthocs.posthoc_nemenyi_friedman` runs the standard post-hoc test
# recommended by Demšar (2006) directly on the same block-design `wide` table
# already produced by `average_ranks_and_friedman`.

# %%
import scikit_posthocs as sp

nemenyi_5e = {metric_col: sp.posthoc_nemenyi_friedman(wide_by_metric[metric_col].to_numpy())
              for metric_col, _ in metrics}
for metric_col, _ in metrics:
    nemenyi_5e[metric_col].index = nemenyi_5e[metric_col].columns = BASELINES_5E
print("Nemenyi pairwise adjusted p-values, Spearman rho (5 baselines, 16 combinations):")
display(nemenyi_5e["Spearman rho"].round(4))

# %%
BASELINES_6E = ["TOPSIS", "VIKOR", "WSM", "PROMETHEE II", "Borda", "ELECTRE III"]
rows = {b: {} for b in BASELINES_6E}
friedman_row_wsm = {}
n_blocks_wsm = None
wide_by_metric_6e = {}
for metric_col, out_col in metrics:
    avg_rank, chi2, pval, wide = average_ranks_and_friedman(cross_dataset_long, metric_col, BASELINES_6E)
    wide_by_metric_6e[metric_col] = wide
    n_blocks_wsm = wide.shape[0]
    for b in BASELINES_6E:
        rows[b][out_col] = round(float(avg_rank[b]), 3)
    friedman_row_wsm[out_col] = (round(chi2, 3), pval)

cross_dataset_wsm = pd.DataFrame(rows).T
print(f"Table tab:cross-dataset-wsm (6 baselines incl. WSM and ELECTRE III, {n_blocks_wsm} combinations where WSM is defined; paper: 12)")
display(cross_dataset_wsm)
for out_col, (chi2, pval) in friedman_row_wsm.items():
    print(f"Friedman chi2 (df=5) on {out_col}: {chi2}  p-value: {pval:.3e}")

# %%
nemenyi_6e = {metric_col: sp.posthoc_nemenyi_friedman(wide_by_metric_6e[metric_col].to_numpy())
              for metric_col, _ in metrics}
for metric_col, _ in metrics:
    nemenyi_6e[metric_col].index = nemenyi_6e[metric_col].columns = BASELINES_6E
print("Nemenyi pairwise adjusted p-values, Spearman rho (6 baselines, 12 combinations):")
display(nemenyi_6e["Spearman rho"].round(4))
print("\nWSM vs VIKOR:", nemenyi_6e["Spearman rho"].loc["WSM", "VIKOR"],
      " WSM vs Borda:", nemenyi_6e["Spearman rho"].loc["WSM", "Borda"],
      " (paper's headline pairwise-significant results, Table tab:cross-dataset-wsm discussion)")

# %% [markdown]
# **Figure 7** (`fig8_nemenyi_cd.pdf` in the paper -- see Section 5.7): critical-
# difference diagrams following Demšar (2006), one panel per baseline set above.
# A thick bar connects baselines whose average ranks differ by less than the
# Nemenyi critical difference $CD=q_\alpha\sqrt{k(k+1)/(6N)}$; baselines not
# joined by a common bar differ significantly at $\alpha=0.05$.

# %%
def _nemenyi_cd(k, N, q_alpha):
    return q_alpha * np.sqrt(k * (k + 1) / (6.0 * N))

def _maximal_cliques(ranks, cd):
    n = len(ranks)
    raw = []
    for i in range(n):
        j = i
        while j + 1 < n and (ranks[j + 1] - ranks[i]) <= cd:
            j += 1
        if j > i:
            raw.append((i, j))
    return sorted(set(a for a in raw if not any(a != b and b[0] <= a[0] and a[1] <= b[1] for b in raw)))

def make_nemenyi_cd_figure(rank_dict_5e, rank_dict_6e, N5e=16, N6e=12):
    # Standard Nemenyi q_alpha (alpha=0.05) studentized-range critical values.
    Q_ALPHA = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.6))
    for ax, rank_dict, N, title in [
        (axes[0], rank_dict_5e, N5e, "5 baselines incl. ELECTRE III, 16 (dataset, $p$) combinations"),
        (axes[1], rank_dict_6e, N6e, "6 baselines incl. WSM + ELECTRE III, 12 combinations"),
    ]:
        k = len(rank_dict)
        cd = _nemenyi_cd(k, N, Q_ALPHA[k])
        names = sorted(rank_dict, key=lambda n: rank_dict[n])
        ranks = [rank_dict[n] for n in names]
        groups = _maximal_cliques(ranks, cd)
        ax.plot([1, k], [0, 0], color="black", lw=1.2)
        for r in range(1, k + 1):
            ax.plot([r, r], [-0.03, 0.03], color="black", lw=1.0)
            ax.text(r, -0.10, str(r), ha="center", va="top", fontsize=10)
        for i, (name, r) in enumerate(zip(names, ranks)):
            y = 0.22 + i * 0.16
            ax.plot([r, r], [0, y], color=BLUE, lw=1.1)
            ax.plot(r, 0, marker="o", color=BLUE, ms=4)
            ax.text(r + 0.05, y, f"{name}  ({r:.2f})", ha="left", va="center", fontsize=9)
        for gi, (ia, ib) in enumerate(groups):
            y = -0.16 - gi * 0.11
            ax.plot([ranks[ia], ranks[ib]], [y, y], color="black", lw=3.2, solid_capstyle="butt")
        ax.text((1 + k) / 2, -0.16 - max(len(groups), 1) * 0.11 - 0.10,
                 f"CD = {cd:.3f}  (Nemenyi, $\\alpha$=0.05, $k$={k}, $N$={N})",
                 ha="center", va="top", fontsize=9.5)
        ax.set_xlim(0.5, k + 0.5)
        ax.set_ylim(-0.30 - 0.11 * max(len(groups), 1), 0.30 + 0.16 * k)
        ax.set_yticks([]); ax.set_xticks([])
        ax.set_title(title, fontsize=10.5, pad=10)
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(False)
    fig.supxlabel("Average rank (Spearman $\\rho$, 1 = closest to the induced ranking)", fontsize=10)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    return fig

rank_dict_5e = cross_dataset_ranks["Avg. rank (rho)"].to_dict()
rank_dict_6e = cross_dataset_wsm["Avg. rank (rho)"].to_dict()
fig = make_nemenyi_cd_figure(rank_dict_5e, rank_dict_6e)
save_fig(fig, "figure8_nemenyi_cd")

# %% [markdown]
# ## 17. Section 5.8 — Synthetic robustness study: scaling across $N$, $K$, and criterion correlation
#
# **Produces:**
# - `monte_carlo_df` — all 1,440 synthetic-run records ($N$, $K$, $\rho$, trial,
#   $T$, $|C^*|$, running time, stop reason)
# - **Figure 8** (`figure7_monte_carlo` here / `fig7_monte_carlo.pdf` in the
#   paper, `fig:monte-carlo`) — three panels: (a) iterations $T$ vs $N$, one
#   line per $K$; (b) normalized running time vs $N$ at $\rho=0$; (c) final
#   selectivity $|C^*|/N$ vs inter-criterion correlation $\rho$, one line per $K$
# - The threshold-violation "degenerate mode" finding at $\rho=0$, $K=12$
#
# **Used in the article at:** Section 5.8 ("Synthetic Robustness Study: Scaling
# Across $N$, $K$, and Criterion Correlation"). The four real datasets above fix
# $N$ and $K$ at whatever values the underlying data happen to provide, and
# their criteria's correlation structure is whatever it is; this section
# complements them with a synthetic sweep that varies $N$, $K$, and
# inter-criterion correlation independently and systematically, both to test
# Proposition~2's complexity argument well beyond what any single real dataset
# can cover, and to provide the empirical bound on $T$ that the complexity
# claim of the Abstract and Conclusion relies on. Criteria are drawn from a
# multivariate normal distribution with a compound-symmetric covariance matrix
# (unit variance, equal pairwise correlation $\rho$ between every pair of
# criteria), for $N\in\{50,100,200,500,1000,2000\}$, $K\in\{3,5,8,12\}$, and
# $\rho\in\{0.0,0.5,0.9\}$, with 20 independent trials per $(N,K,\rho)$ cell
# (1,440 runs total). Throughout, $p=0.5$, $L=1$, priority is equal across
# criteria, and all criteria are treated as maximized (an arbitrary but
# inconsequential choice for synthetic, symmetric criteria).

# %%
_mc_rng_master = np.random.default_rng(12345)

MC_N_GRID = [50, 100, 200, 500, 1000, 2000]
MC_K_GRID = [3, 5, 8, 12]
MC_RHO_GRID = [0.0, 0.5, 0.9]
MC_TRIALS = 20
MC_P, MC_L = 0.5, 1

mc_records = []
_mc_t0 = _time.time()
for K in MC_K_GRID:
    for rho in MC_RHO_GRID:
        Sigma = np.full((K, K), rho)
        np.fill_diagonal(Sigma, 1.0)
        for N in MC_N_GRID:
            for trial in range(MC_TRIALS):
                seed = int(_mc_rng_master.integers(0, 2**31 - 1))
                local_rng = np.random.default_rng(seed)
                X = local_rng.multivariate_normal(mean=np.zeros(K), cov=Sigma, size=N)
                directions = ["max"] * K
                _t0 = _time.perf_counter()
                res = iqis_select(X, directions, p=MC_P, L=MC_L)
                dt = _time.perf_counter() - _t0
                mc_records.append({"N": N, "K": K, "rho": rho, "trial": trial,
                                    "T": res.n_iterations + 1, "n_cstar": res.n_selected,
                                    "time_s": dt, "stopped_by": res.stopped_by})

monte_carlo_df = pd.DataFrame(mc_records)
print(f"Total synthetic runs: {len(monte_carlo_df)} (paper: 1,440)  wall time: {_time.time()-_mc_t0:.1f}s")
print(f"T across all runs: max={monte_carlo_df['T'].max()}, mean={monte_carlo_df['T'].mean():.2f}, "
      f"median={monte_carlo_df['T'].median():.0f}  (paper: max=8, mean=3.44, median=3)")
print("T by K (mean):")
print(monte_carlo_df.groupby("K")["T"].mean().round(2))
print("(paper: mean T = 4.69 at K=3, down to 2.58 at K=12)")

# %% [markdown]
# Final selectivity $|C^*|/N$ as a function of inter-criterion correlation, and
# the threshold-violation "degenerate mode" at $\rho=0$, $K=12$: with many
# mutually weakly-related criteria, the very first intersection frequently
# already falls below $L$, so the run stops by threshold violation
# (Proposition~2) and returns the entire, untouched initial population.

# %%
print("Mean |C*| by rho (paper: 137.2 at rho=0.0, 5.5 at rho=0.5, 2.45 at rho=0.9):")
print(monte_carlo_df.groupby("rho")["n_cstar"].mean().round(2))

_k12_rho0 = monte_carlo_df[(monte_carlo_df["K"] == 12) & (monte_carlo_df["rho"] == 0.0)]
print(f"\nAt K=12, rho=0.0: mean |C*|/N = {(_k12_rho0['n_cstar']/_k12_rho0['N']).mean():.2%} "
      f"(paper: 70-95% of N across every N tested)")

_tv_rho0 = ((monte_carlo_df["rho"] == 0.0) & (monte_carlo_df["stopped_by"] == "threshold")).sum()
_n_rho0 = (monte_carlo_df["rho"] == 0.0).sum()
_tv_rho9 = ((monte_carlo_df["rho"] == 0.9) & (monte_carlo_df["stopped_by"] == "threshold")).sum()
_n_rho9 = (monte_carlo_df["rho"] == 0.9).sum()
print(f"Threshold-violation stops at rho=0.0: {_tv_rho0} of {_n_rho0} ({_tv_rho0/_n_rho0:.1%})  "
      f"(paper: 345 of 480, 71.9%)")
print(f"Threshold-violation stops at rho=0.9: {_tv_rho9} of {_n_rho9} ({_tv_rho9/_n_rho9:.1%})  "
      f"(paper: 221 of 480, 46.0%)")

# %% [markdown]
# **Figure 8** (three panels, `figure7_monte_carlo` / `fig7_monte_carlo.pdf` in
# the paper, `fig:monte-carlo`).

# %%
def make_monte_carlo_figure(df):
    colors = {3: BLUE, 5: ORANGE, 8: AQUA, 12: MAGENTA}
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.1))

    # Panel (a): T vs N, one line per K (median, shaded IQR), rho pooled
    ax = axes[0]
    for K in sorted(df["K"].unique()):
        sub = df[df["K"] == K]
        g = sub.groupby("N")["T"]
        med, q25, q75 = g.median(), g.quantile(0.25), g.quantile(0.75)
        ax.plot(med.index, med.values, marker="o", ms=3.5, lw=1.3, color=colors[K], label=f"K={K}")
        ax.fill_between(med.index, q25.values, q75.values, color=colors[K], alpha=0.15, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlabel(r"$N$ (log scale)")
    ax.set_ylabel(r"$T$ (number of iterations)")
    ax.set_title("(a) Iterations $T$ across 1,440 trials", fontsize=8, loc="left")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax.set_ylim(0, 9)

    # Panel (b): time / (N log2 N) vs N, one line per K, rho=0 only
    ax = axes[1]
    for K in sorted(df["K"].unique()):
        sub = df[(df["K"] == K) & (df["rho"] == 0.0)]
        med = sub.groupby("N")["time_s"].median()
        Ns = med.index.to_numpy(dtype=float)
        norm = med.to_numpy() / (Ns * np.log2(Ns))
        ax.plot(Ns, norm, marker="s", ms=3.5, lw=1.3, color=colors[K], label=f"K={K}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$N$ (log scale)")
    ax.set_ylabel(r"Time$/(N\log_2 N)$ (s, log scale)")
    ax.set_title(r"(b) Normalized running time ($\rho=0$)", fontsize=8, loc="left")
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")

    # Panel (c): |C*|/N (median, log scale) vs rho, one line per K, shaded IQR
    ax = axes[2]
    for K in sorted(df["K"].unique()):
        sub = df[df["K"] == K].copy()
        sub["sel"] = sub["n_cstar"] / sub["N"]
        g = sub.groupby("rho")["sel"]
        med, q25, q75 = g.median(), g.quantile(0.25), g.quantile(0.75)
        ax.plot(med.index, med.values, marker="^", ms=4.5, lw=1.3, color=colors[K], label=f"K={K}")
        ax.fill_between(med.index, q25.values, q75.values, color=colors[K], alpha=0.15, linewidth=0)
    ax.set_yscale("log")
    ax.set_xlabel(r"Inter-criterion correlation $\rho$")
    ax.set_ylabel(r"$|C^*|/N$ (median, log scale)")
    ax.set_title("(c) Selectivity vs. criterion correlation", fontsize=8, loc="left")
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")

    fig.tight_layout()
    return fig

fig = make_monte_carlo_figure(monte_carlo_df)
save_fig(fig, "figure7_monte_carlo")
print("Saved figure7_monte_carlo.pdf/.png (paper figure fig7_monte_carlo.pdf)")

# %% [markdown]
# ## 18. Section 5.8 — Validating the degeneracy-avoiding retention ratio
#
# **Produces:** Table `tab:degenerate-fix` (p=0.50 vs p=0.75 comparison) and the
# L-sensitivity check cited alongside it.
#
# **Used in the article at:** Section 5.8, paragraph "Why the first round
# collapses, and how to choose p to avoid it" and Table `tab:degenerate-fix`.
# Re-runs the identical 1,440-configuration sweep above (same master seed
# 12345, same generation order as Section 17) at p=0.75 instead of p=0.5,
# holding L=1, so every run is evaluated on the exact same synthetic dataset
# under both retention ratios.

# %%
_fix_rng_master = np.random.default_rng(12345)
_fix_datasets = {}
for K in MC_K_GRID:
    for rho in MC_RHO_GRID:
        Sigma = np.full((K, K), rho)
        np.fill_diagonal(Sigma, 1.0)
        for N in MC_N_GRID:
            for trial in range(MC_TRIALS):
                seed = int(_fix_rng_master.integers(0, 2**31 - 1))
                local_rng = np.random.default_rng(seed)
                X = local_rng.multivariate_normal(mean=np.zeros(K), cov=Sigma, size=N)
                _fix_datasets[(K, rho, N, trial)] = X

def _run_fix_sweep(p, L):
    records = []
    for (K, rho, N, trial), X in _fix_datasets.items():
        directions = ["max"] * K
        res = iqis_select(X, directions, p=p, L=L)
        degenerate = (res.stopped_by == "threshold") and (res.n_iterations == 0)
        records.append({"N": N, "K": K, "rho": rho, "trial": trial,
                         "T": res.n_iterations + 1, "n_cstar": res.n_selected,
                         "degenerate": degenerate,
                         "any_violation": res.stopped_by == "threshold"})
    return pd.DataFrame(records)

df_p050 = _run_fix_sweep(p=0.50, L=1)
df_p075 = _run_fix_sweep(p=0.75, L=1)

print("Table tab:degenerate-fix -- p=0.50 vs p=0.75 (L=1), same 1,440 synthetic datasets")
for name, df in [("p=0.50", df_p050), ("p=0.75", df_p075)]:
    print(f"  {name}: true-degenerate={df['degenerate'].mean():.1%}  "
          f"any-violation={df['any_violation'].mean():.1%}  "
          f"mean|C*|/N={(df['n_cstar']/df['N']).mean():.3f}  "
          f"T mean/max={df['T'].mean():.2f}/{df['T'].max()}")

_k12r0_050 = df_p050[(df_p050["K"] == 12) & (df_p050["rho"] == 0.0)]
_k12r0_075 = df_p075[(df_p075["K"] == 12) & (df_p075["rho"] == 0.0)]
print(f"\nK=12, rho=0.0 worst-case cell: mean |C*|/N = "
      f"{(_k12r0_050['n_cstar']/_k12r0_050['N']).mean():.1%} at p=0.50 -> "
      f"{(_k12r0_075['n_cstar']/_k12r0_075['N']).mean():.1%} at p=0.75")

print("\nL-sensitivity at p=0.75 (paper cites L=1 -> 0.1% vs L=2 -> 0.8%):")
for L in [1, 2, 3]:
    df = _run_fix_sweep(p=0.75, L=L)
    print(f"  L={L}: true-degenerate={df['degenerate'].mean():.1%}")
