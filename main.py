# main.py
# -*- coding: utf-8 -*-

"""
S.A.D. — Статистичний аналіз даних (Tkinter)

Потрібно: Python 3.8+, numpy, scipy
Встановлення: pip install numpy scipy

Останні правки:
✅ Додано вибір типу дизайну: CRD або Split-plot (2 фактори).
✅ У вікні параметрів звіту тепер обов'язково: показник, одиниці, тип дизайну.
✅ Для Split-plot обов'язково обираються Main-plot фактор та Sub-plot фактор.
✅ У звіті Split-plot ANOVA: A тестується на Error(A)=Rep×A, B та A×B — на Error(B).
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import tkinter.font as tkfont

import math
import numpy as np
from itertools import combinations
from collections import defaultdict
from datetime import datetime

from scipy.stats import shapiro, t, f as f_dist
from scipy.stats import mannwhitneyu, kruskal, levene, rankdata
from scipy.stats import studentized_range

ALPHA = 0.05
COL_W = 10


# -------------------------
# DPI awareness (Windows)
# -------------------------
try:
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
except Exception:
    pass


# -------------------------
# ICON (robust search)
# -------------------------
def _script_dir():
    try:
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            return sys._MEIPASS
    except Exception:
        pass
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except Exception:
        return os.getcwd()


def _exe_dir():
    try:
        return os.path.dirname(os.path.abspath(sys.executable))
    except Exception:
        return None


def _argv0_dir():
    try:
        p = sys.argv[0]
        if not p:
            return None
        return os.path.dirname(os.path.abspath(p))
    except Exception:
        return None


def _find_icon_file():
    candidates = []
    for base in [_script_dir(), os.getcwd(), _argv0_dir(), _exe_dir()]:
        if base:
            candidates.append(os.path.join(base, "icon.ico"))
    try:
        if hasattr(sys, "_MEIPASS"):
            candidates.append(os.path.join(sys._MEIPASS, "icon.ico"))
    except Exception:
        pass
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def set_window_icon(win: tk.Tk | tk.Toplevel):
    ico = _find_icon_file()
    if not ico:
        return
    try:
        win.iconbitmap(ico)
        return
    except Exception:
        pass
    try:
        win.iconbitmap(default=ico)
        return
    except Exception:
        pass


# -------------------------
# Helpers
# -------------------------
def significance_mark(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def normality_text(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "н/д"
    return "нормальний розподіл" if p > 0.05 else "не нормальний розподіл"


def fmt_num(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{x:.{nd}f}"


def first_seen_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def center_window(win):
    win.update_idletasks()
    w = win.winfo_width()
    h = win.winfo_height()
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    x = (sw - w) // 2
    y = (sh - h) // 2
    win.geometry(f"{w}x{h}+{x}+{y}")


def harmonic_mean(nums):
    nums = [x for x in nums if x and x > 0]
    if not nums:
        return np.nan
    return len(nums) / sum(1.0 / x for x in nums)


def median_q1_q3(arr):
    if arr is None or len(arr) == 0:
        return (np.nan, np.nan, np.nan)
    a = np.array(arr, dtype=float)
    med = float(np.median(a))
    q1 = float(np.percentile(a, 25))
    q3 = float(np.percentile(a, 75))
    return med, q1, q3


def cv_percent_from_values(values):
    a = np.array(values, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) < 2:
        return np.nan
    m = float(np.mean(a))
    if m == 0:
        return np.nan
    sd = float(np.std(a, ddof=1))
    return (sd / m) * 100.0


def cv_percent_from_level_means(level_means):
    vals = [float(x) for x in level_means if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(vals) < 2:
        return np.nan
    m = float(np.mean(vals))
    if m == 0:
        return np.nan
    sd = float(np.std(vals, ddof=1))
    return (sd / m) * 100.0


def partial_eta2_label(pe2: float) -> str:
    if pe2 is None or (isinstance(pe2, float) and math.isnan(pe2)):
        return ""
    if pe2 < 0.01:
        return "дуже слабкий"
    if pe2 < 0.06:
        return "слабкий"
    if pe2 < 0.14:
        return "середній"
    return "сильний"


def epsilon_squared_kw(H: float, n: int, k: int) -> float:
    if any(x is None for x in [H, n, k]):
        return np.nan
    if isinstance(H, float) and math.isnan(H):
        return np.nan
    if n <= k or k < 2:
        return np.nan
    return float((H - k + 1.0) / (n - k))


def cliffs_delta(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan
    gt = 0
    lt = 0
    for xi in x:
        gt += int(np.sum(xi > y))
        lt += int(np.sum(xi < y))
    return float((gt - lt) / (nx * ny))


def cliffs_label(delta_abs: float) -> str:
    if delta_abs is None or (isinstance(delta_abs, float) and math.isnan(delta_abs)):
        return ""
    if delta_abs < 0.147:
        return "дуже слабкий"
    if delta_abs < 0.33:
        return "слабкий"
    if delta_abs < 0.474:
        return "середній"
    return "сильний"


def subset_stats(long, keys):
    sums = defaultdict(float)
    cnts = defaultdict(int)
    for r in long:
        v = r.get("value", np.nan)
        if v is None or math.isnan(v):
            continue
        k = tuple(r.get(x) for x in keys)
        sums[k] += float(v)
        cnts[k] += 1
    out = {}
    for k, n in cnts.items():
        out[k] = (sums[k] / n if n > 0 else np.nan, n)
    return out


def groups_by_keys(long, keys):
    g = defaultdict(list)
    for r in long:
        v = r.get("value", np.nan)
        if v is None or math.isnan(v):
            continue
        k = tuple(r.get(x) for x in keys)
        g[k].append(float(v))
    return g


def variant_mean_sd(long, factor_keys):
    vals = defaultdict(list)
    for r in long:
        v = r.get("value", np.nan)
        if v is None or math.isnan(v):
            continue
        k = tuple(r.get(k) for k in factor_keys)
        vals[k].append(float(v))
    out = {}
    for k, arr in vals.items():
        n = len(arr)
        m = float(np.mean(arr)) if n > 0 else np.nan
        if n >= 2:
            sd = float(np.std(arr, ddof=1))
        elif n == 1:
            sd = 0.0
        else:
            sd = np.nan
        out[k] = (m, sd, n)
    return out


def cld_multi_letters(levels_order, means_dict, sig_matrix):
    valid = [lvl for lvl in levels_order if not math.isnan(means_dict.get(lvl, np.nan))]
    if not valid:
        return {lvl: "" for lvl in levels_order}

    sorted_lvls = sorted(valid, key=lambda z: means_dict[z], reverse=True)

    def is_sig(x, y):
        if x == y:
            return False
        return bool(sig_matrix.get((x, y), False) or sig_matrix.get((y, x), False))

    groups = []
    for lvl in sorted_lvls:
        compatible = []
        for gi, grp in enumerate(groups):
            if all(not is_sig(lvl, other) for other in grp):
                compatible.append(gi)
        if not compatible:
            groups.append(set([lvl]))
        else:
            for gi in compatible:
                groups[gi].add(lvl)

    def share_group(a, b):
        return any((a in g and b in g) for g in groups)

    for i in range(len(sorted_lvls)):
        for j in range(i + 1, len(sorted_lvls)):
            a, b = sorted_lvls[i], sorted_lvls[j]
            if is_sig(a, b):
                continue
            if share_group(a, b):
                continue
            newg = set([a, b])
            for c in sorted_lvls:
                if c in newg:
                    continue
                if (not is_sig(c, a)) and (not is_sig(c, b)) and all(not is_sig(c, x) for x in newg):
                    newg.add(c)
            groups.append(newg)

    uniq = []
    for g in groups:
        if any(g == h for h in uniq):
            continue
        uniq.append(g)

    cleaned = []
    for g in uniq:
        if any((g < h) for h in uniq):
            continue
        cleaned.append(g)
    groups = cleaned

    alphabet = "abcdefghijklmnopqrstuvwxyz"
    letters_for_group = []
    for idx in range(len(groups)):
        letters_for_group.append(alphabet[idx] if idx < len(alphabet) else f"g{idx}")

    mapping = {lvl: [] for lvl in sorted_lvls}
    for gi, grp in enumerate(groups):
        letter = letters_for_group[gi]
        for lvl in grp:
            mapping[lvl].append(letter)

    out = {}
    for lvl in levels_order:
        out[lvl] = "".join(sorted(mapping.get(lvl, [])))
    return out


def fit_font_size_to_texts(texts, family="Times New Roman", start=13, min_size=9, target_px=155):
    f = tkfont.Font(family=family, size=start)
    size = start
    while size > min_size:
        maxw = max(f.measure(t) for t in texts)
        if maxw <= target_px:
            break
        size -= 1
        f.configure(size=size)
    return f


# -------------------------
# REPORT TABLES (tabs)
# -------------------------
def build_table_block(headers, rows):
    def hcell(x): return f"{x} "
    def ccell(x): return "" if x is None else str(x)

    lines = []
    lines.append("\t".join(hcell(h) for h in headers))
    lines.append("\t".join("—" * max(3, len(str(h))) for h in headers))
    for r in rows:
        lines.append("\t".join(ccell(v) for v in r))
    return "\n".join(lines) + "\n"


def tabs_from_table_px(font_obj: tkfont.Font, headers, rows, padding_px=32, extra_gap_after_col=None, extra_gap_px=0):
    ncol = len(headers)
    maxw = [0] * ncol

    for j in range(ncol):
        maxw[j] = font_obj.measure(str(headers[j]))

    for r in rows:
        for j in range(ncol):
            s = "" if r[j] is None else str(r[j])
            w = font_obj.measure(s)
            if w > maxw[j]:
                maxw[j] = w

    widths = [w + padding_px for w in maxw]

    tabs = []
    acc = 0
    for j, w in enumerate(widths[:-1]):
        acc += w
        if extra_gap_after_col is not None and j == extra_gap_after_col:
            acc += extra_gap_px
        tabs.append(acc)

    return tuple(tabs)


# -------------------------
# ANOVA 1–4 (CRD)
# -------------------------
def anova_n_way(long, factors, levels_by_factor):
    N = len(long)
    values = np.array([r["value"] for r in long], dtype=float)
    grand_mean = np.nanmean(values)

    k = len(factors)
    if k < 1 or k > 4:
        raise ValueError("Підтримуються 1–4 фактори.")

    stats = {}
    for r in range(1, k + 1):
        for comb in combinations(factors, r):
            stats[comb] = subset_stats(long, comb)

    full = tuple(factors)
    cell_means = {kk: vv[0] for kk, vv in stats[full].items()}
    cell_counts = {kk: vv[1] for kk, vv in stats[full].items()}

    SS_total = np.nansum((values - grand_mean) ** 2)

    SS_error = 0.0
    for rec in long:
        key = tuple(rec.get(f) for f in factors)
        v = rec.get("value", np.nan)
        m = cell_means.get(key, np.nan)
        if not math.isnan(v) and not math.isnan(m):
            SS_error += (v - m) ** 2

    levels_count = {f: len(levels_by_factor[f]) for f in factors}
    total_cells = 1
    for fct in factors:
        total_cells *= levels_count[fct]

    df_total = N - 1
    df_error = N - total_cells
    if df_error <= 0:
        df_error = max(1, df_error)

    MS_error = SS_error / df_error if df_error > 0 else np.nan

    SS = {}
    df = {}

    def delta_for_subset(S, levels_S):
        s_len = len(S)
        delta = 0.0
        for rT in range(0, s_len + 1):
            for T in combinations(S, rT):
                coef = (-1) ** (s_len - rT)
                if len(T) == 0:
                    mu = grand_mean
                else:
                    idxs = [S.index(x) for x in T]
                    levs_T = tuple(levels_S[i] for i in idxs)
                    mu = stats[T].get(levs_T, (np.nan, 0))[0]
                if mu is None or (isinstance(mu, float) and math.isnan(mu)):
                    mu = 0.0
                delta += coef * mu
        return delta

    for rS in range(1, k + 1):
        for S in combinations(factors, rS):
            d = 1
            for fct in S:
                d *= (levels_count[fct] - 1)
            df[S] = d

            ss = 0.0
            for levels_S, (mS, nS) in stats[S].items():
                if nS <= 0 or mS is None or (isinstance(mS, float) and math.isnan(mS)):
                    continue
                delta = delta_for_subset(list(S), list(levels_S))
                ss += nS * (delta ** 2)
            SS[S] = ss

    def pretty_interaction(subset):
        if len(subset) == 1:
            return f"Фактор {subset[0]}"
        return "Фактор " + "×".join(subset)

    table_rows = []
    for rS in range(1, k + 1):
        for S in combinations(factors, rS):
            name = pretty_interaction(S)
            SSv = SS.get(S, np.nan)
            dfv = df.get(S, np.nan)
            MSv = SSv / dfv if (dfv and not math.isnan(dfv) and dfv > 0) else np.nan
            Fv = MSv / MS_error if (not math.isnan(MS_error) and MS_error > 0 and not math.isnan(MSv)) else np.nan
            pv = 1 - f_dist.cdf(Fv, dfv, df_error) if (not math.isnan(Fv) and not math.isnan(dfv)) else np.nan
            table_rows.append((name, SSv, dfv, MSv, Fv, pv))

    table_rows.append(("Залишок", SS_error, df_error, MS_error, None, None))
    table_rows.append(("Загальна", SS_total, df_total, None, None, None))

    tval = t.ppf(1 - ALPHA / 2, df_error) if df_error > 0 else np.nan
    NIR05 = {}
    n_eff_cells = harmonic_mean([n for n in cell_counts.values() if n and n > 0])
    nir_all = tval * math.sqrt(2 * MS_error / n_eff_cells) if not any(
        math.isnan(x) for x in [tval, MS_error, n_eff_cells]
    ) else np.nan
    NIR05["Загальна"] = nir_all
    for fct in factors:
        marg = subset_stats(long, (fct,))
        n_eff = harmonic_mean([v[1] for v in marg.values() if v[1] and v[1] > 0])
        nir = tval * math.sqrt(2 * MS_error / n_eff) if not any(
            math.isnan(x) for x in [tval, MS_error, n_eff]
        ) else np.nan
        NIR05[f"Фактор {fct}"] = nir

    return {
        "table": table_rows,
        "cell_means": cell_means,
        "cell_counts": cell_counts,
        "MS_error": MS_error,
        "df_error": df_error,
        "NIR05": NIR05,
        "SS_total": SS_total,
        "SS_error": SS_error,
        "grand_mean": grand_mean
    }


# -------------------------
# Split-plot ANOVA (2 factors only): Rep blocks, A = main plot, B = subplot
# -------------------------
def _weighted_ss_from_means(counts_dict, means_dict, grand_mean):
    ss = 0.0
    for k, n in counts_dict.items():
        m = means_dict.get(k, np.nan)
        if n and n > 0 and (m is not None) and (not (isinstance(m, float) and math.isnan(m))):
            ss += n * (float(m) - float(grand_mean)) ** 2
    return float(ss)


def anova_splitplot_2f(long, rep_key, A_key, B_key, levels_by_factor):
    # group means & counts
    vals = np.array([r["value"] for r in long], dtype=float)
    grand = float(np.nanmean(vals))
    SS_total = float(np.nansum((vals - grand) ** 2))
    df_total = int(len(vals) - 1)

    # levels
    reps = first_seen_order([r.get(rep_key) for r in long])
    A_lvls = levels_by_factor[A_key]
    B_lvls = levels_by_factor[B_key]
    r = len(reps)
    a = len(A_lvls)
    b = len(B_lvls)
    if r < 2 or a < 2 or b < 2:
        raise ValueError("Split-plot потребує: Rep ≥ 2, рівнів A ≥ 2, рівнів B ≥ 2.")

    # marginals
    rep_stats = subset_stats(long, (rep_key,))
    A_stats = subset_stats(long, (A_key,))
    B_stats = subset_stats(long, (B_key,))
    repA_stats = subset_stats(long, (rep_key, A_key))
    AB_stats = subset_stats(long, (A_key, B_key))
    repAB_stats = subset_stats(long, (rep_key, A_key, B_key))

    rep_means = {k[0]: v[0] for k, v in rep_stats.items()}
    rep_counts = {k[0]: v[1] for k, v in rep_stats.items()}

    A_means = {k[0]: v[0] for k, v in A_stats.items()}
    A_counts = {k[0]: v[1] for k, v in A_stats.items()}

    B_means = {k[0]: v[0] for k, v in B_stats.items()}
    B_counts = {k[0]: v[1] for k, v in B_stats.items()}

    repA_means = { (k[0], k[1]): v[0] for k, v in repA_stats.items()}
    repA_counts = { (k[0], k[1]): v[1] for k, v in repA_stats.items()}

    AB_means = { (k[0], k[1]): v[0] for k, v in AB_stats.items()}
    AB_counts = { (k[0], k[1]): v[1] for k, v in AB_stats.items()}

    repAB_means = { (k[0], k[1], k[2]): v[0] for k, v in repAB_stats.items()}
    repAB_counts = { (k[0], k[1], k[2]): v[1] for k, v in repAB_stats.items()}

    # SS for blocks, A, B
    SS_rep = _weighted_ss_from_means(rep_counts, rep_means, grand)
    SS_A = _weighted_ss_from_means(A_counts, A_means, grand)
    SS_B = _weighted_ss_from_means(B_counts, B_means, grand)

    # SS for Rep×A (Error A)
    SS_repA = 0.0
    for (rep, a_lvl), n in repA_counts.items():
        m_ra = repA_means.get((rep, a_lvl), np.nan)
        m_r = rep_means.get(rep, np.nan)
        m_a = A_means.get(a_lvl, np.nan)
        if n <= 0 or any(isinstance(x, float) and math.isnan(x) for x in [m_ra, m_r, m_a]):
            continue
        term = float(m_ra) - float(m_r) - float(m_a) + grand
        SS_repA += n * (term ** 2)
    SS_repA = float(SS_repA)

    # SS for A×B
    SS_AB = 0.0
    for (a_lvl, b_lvl), n in AB_counts.items():
        m_ab = AB_means.get((a_lvl, b_lvl), np.nan)
        m_a = A_means.get(a_lvl, np.nan)
        m_b = B_means.get(b_lvl, np.nan)
        if n <= 0 or any(isinstance(x, float) and math.isnan(x) for x in [m_ab, m_a, m_b]):
            continue
        term = float(m_ab) - float(m_a) - float(m_b) + grand
        SS_AB += n * (term ** 2)
    SS_AB = float(SS_AB)

    # Error(B) = remainder
    SS_errorB = SS_total - (SS_rep + SS_A + SS_repA + SS_B + SS_AB)
    SS_errorB = float(max(0.0, SS_errorB))

    # dfs (balanced split-plot classic)
    df_rep = r - 1
    df_A = a - 1
    df_repA = (r - 1) * (a - 1)          # Error(A)
    df_B = b - 1
    df_AB = (a - 1) * (b - 1)
    df_errorB = (r - 1) * a * (b - 1)    # Error(B)
    # safety if missing data made it smaller
    if df_errorB <= 0:
        df_errorB = max(1, df_total - (df_rep + df_A + df_repA + df_B + df_AB))

    MS_rep = SS_rep / df_rep if df_rep > 0 else np.nan
    MS_A = SS_A / df_A if df_A > 0 else np.nan
    MS_repA = SS_repA / df_repA if df_repA > 0 else np.nan
    MS_B = SS_B / df_B if df_B > 0 else np.nan
    MS_AB = SS_AB / df_AB if df_AB > 0 else np.nan
    MS_errorB = SS_errorB / df_errorB if df_errorB > 0 else np.nan

    # F-tests: A vs Rep×A; B and AB vs Error(B)
    F_A = (MS_A / MS_repA) if (not math.isnan(MS_A) and not math.isnan(MS_repA) and MS_repA > 0) else np.nan
    p_A = (1 - f_dist.cdf(F_A, df_A, df_repA)) if (not math.isnan(F_A)) else np.nan

    F_B = (MS_B / MS_errorB) if (not math.isnan(MS_B) and not math.isnan(MS_errorB) and MS_errorB > 0) else np.nan
    p_B = (1 - f_dist.cdf(F_B, df_B, df_errorB)) if (not math.isnan(F_B)) else np.nan

    F_AB = (MS_AB / MS_errorB) if (not math.isnan(MS_AB) and not math.isnan(MS_errorB) and MS_errorB > 0) else np.nan
    p_AB = (1 - f_dist.cdf(F_AB, df_AB, df_errorB)) if (not math.isnan(F_AB)) else np.nan

    table = [
        ("Блоки (Rep)", SS_rep, df_rep, MS_rep, None, None),
        (f"Фактор {A_key}", SS_A, df_A, MS_A, F_A, p_A),
        (f"Error(A): Rep×{A_key}", SS_repA, df_repA, MS_repA, None, None),
        (f"Фактор {B_key}", SS_B, df_B, MS_B, F_B, p_B),
        (f"Фактор {A_key}×{B_key}", SS_AB, df_AB, MS_AB, F_AB, p_AB),
        ("Error(B)", SS_errorB, df_errorB, MS_errorB, None, None),
        ("Загальна", SS_total, df_total, None, None, None)
    ]

    # NIR05 separately for A and B (LSD uses corresponding MS & df)
    NIR05 = {}
    if df_repA > 0 and not math.isnan(MS_repA) and MS_repA > 0:
        tA = float(t.ppf(1 - ALPHA / 2, df_repA))
        n_eff_A = harmonic_mean([n for n in repA_counts.values() if n and n > 0])
        NIR05[f"Фактор {A_key}"] = (tA * math.sqrt(2 * MS_repA / n_eff_A)) if n_eff_A > 0 else np.nan
    else:
        NIR05[f"Фактор {A_key}"] = np.nan

    if df_errorB > 0 and not math.isnan(MS_errorB) and MS_errorB > 0:
        tB = float(t.ppf(1 - ALPHA / 2, df_errorB))
        n_eff_B = harmonic_mean([n for n in AB_counts.values() if n and n > 0])  # conservative
        NIR05[f"Фактор {B_key}"] = (tB * math.sqrt(2 * MS_errorB / n_eff_B)) if n_eff_B > 0 else np.nan
    else:
        NIR05[f"Фактор {B_key}"] = np.nan

    # For residual normality: use deviations from Rep×A×B cell means
    cell_means = {k: v for k, v in repAB_means.items()}

    return {
        "table": table,
        "MS_error_A": MS_repA,
        "df_error_A": df_repA,
        "MS_error_B": MS_errorB,
        "df_error_B": df_errorB,
        "SS_total": SS_total,
        "SS_error": SS_errorB,   # for R2-like use (note: split-plot has 2 errors; keep B error as residual)
        "NIR05": NIR05,
        "grand_mean": grand,
        "cell_means_repAB": cell_means,
        "reps": reps
    }


# -------------------------
# Brown–Forsythe
# -------------------------
def brown_forsythe_from_groups(groups_dict):
    samples = [np.array(v, dtype=float) for v in groups_dict.values() if v and len(v) >= 2]
    if len(samples) < 2:
        return (np.nan, np.nan)
    try:
        stat, p = levene(*samples, center="median")
        return (float(stat), float(p))
    except Exception:
        return (np.nan, np.nan)


# -------------------------
# LSD + pairwise
# -------------------------
def lsd_sig_matrix(levels_order, means, ns, MS_error, df_error, alpha=0.05):
    sig = {}
    if any(math.isnan(x) for x in [MS_error, df_error]) or MS_error <= 0 or df_error <= 0:
        return sig
    tcrit = float(t.ppf(1 - alpha / 2, df_error))
    for i in range(len(levels_order)):
        for j in range(i + 1, len(levels_order)):
            a, b = levels_order[i], levels_order[j]
            ma, mb = means.get(a, np.nan), means.get(b, np.nan)
            na, nb = ns.get(a, 0), ns.get(b, 0)
            if any(map(math.isnan, [ma, mb])) or na <= 0 or nb <= 0:
                continue
            se = math.sqrt(MS_error * (1.0 / na + 1.0 / nb))
            if se <= 0:
                continue
            sig[(a, b)] = (abs(ma - mb) > tcrit * se)
    return sig


def pairwise_param_short_variants_pm(v_names, means, ns, MS_error, df_error, method, alpha=0.05):
    pairs = [(v_names[i], v_names[j]) for i in range(len(v_names)) for j in range(i + 1, len(v_names))]
    rows = []
    sig = {}

    if method == "bonferroni":
        mtests = len(pairs) if pairs else 1
        for a, b in pairs:
            ma, mb = means.get(a, np.nan), means.get(b, np.nan)
            na, nb = ns.get(a, 0), ns.get(b, 0)
            if any(map(math.isnan, [ma, mb, MS_error])) or na <= 0 or nb <= 0 or MS_error <= 0:
                continue
            se = math.sqrt(MS_error * (1.0 / na + 1.0 / nb))
            if se <= 0:
                continue
            tval_ = abs(ma - mb) / se
            p = 2 * (1 - t.cdf(tval_, df_error))
            p_adj = min(1.0, float(p) * mtests)
            decision = (p_adj < alpha)
            sig[(a, b)] = decision
            rows.append([f"{a}  vs  {b}", f"{p_adj:.4f}", "+" if decision else "-"])
        return rows, sig

    if method == "tukey":
        n_eff = harmonic_mean([ns.get(l, 0) for l in v_names])
        for a, b in pairs:
            ma, mb = means.get(a, np.nan), means.get(b, np.nan)
            if any(map(math.isnan, [ma, mb, MS_error, n_eff])) or n_eff <= 0 or MS_error <= 0:
                continue
            q = abs(ma - mb) / math.sqrt(MS_error / n_eff)
            p = float(studentized_range.sf(q, len(v_names), df_error))
            decision = (p < alpha)
            sig[(a, b)] = decision
            rows.append([f"{a}  vs  {b}", f"{p:.4f}", "+" if decision else "-"])
        return rows, sig

    if method == "duncan":
        valid = [lvl for lvl in v_names if not math.isnan(means.get(lvl, np.nan))]
        ordered = sorted(valid, key=lambda z: means[z], reverse=True)
        pos = {lvl: i for i, lvl in enumerate(ordered)}
        n_eff = harmonic_mean([ns.get(l, 0) for l in ordered])
        se_base = math.sqrt(MS_error / n_eff) if (not math.isnan(n_eff) and n_eff > 0 and MS_error > 0) else np.nan

        for a, b in pairs:
            ma, mb = means.get(a, np.nan), means.get(b, np.nan)
            if any(map(math.isnan, [ma, mb, se_base])) or se_base <= 0:
                continue
            ia, ib = pos.get(a, None), pos.get(b, None)
            if ia is None or ib is None:
                continue
            i, j = min(ia, ib), max(ia, ib)
            r_ = (j - i) + 1

            q = abs(ma - mb) / se_base
            p = float(studentized_range.sf(q, r_, df_error))
            qcrit = studentized_range.ppf(1 - alpha, r_, df_error)
            SR = qcrit * se_base
            decision = abs(ma - mb) > SR
            sig[(a, b)] = decision
            rows.append([f"{a}  vs  {b}", f"{p:.4f}", "+" if decision else "-"])
        return rows, sig

    return [], {}


def pairwise_mw_bonf_with_effect(v_names, groups_dict, alpha=0.05):
    pairs = [(v_names[i], v_names[j]) for i in range(len(v_names)) for j in range(i + 1, len(v_names))]
    mtests = len(pairs) if pairs else 1
    rows = []
    sig = {}
    for a, b in pairs:
        xa = groups_dict.get(a, [])
        xb = groups_dict.get(b, [])
        if len(xa) == 0 or len(xb) == 0:
            continue
        try:
            U, p = mannwhitneyu(xa, xb, alternative="two-sided")
        except Exception:
            continue
        p = float(p)
        U = float(U)
        p_adj = min(1.0, p * mtests)
        decision = (p_adj < alpha)
        sig[(a, b)] = decision

        d = cliffs_delta(xa, xb)
        lab = cliffs_label(abs(d)) if not (isinstance(d, float) and math.isnan(d)) else ""
        rows.append([f"{a}  vs  {b}", fmt_num(U, 2), f"{p_adj:.4f}", "+" if decision else "-", fmt_num(d, 3), lab])
    return rows, sig


# -------------------------
# Effect strength tables
# -------------------------
def build_effect_strength_rows(anova_table_rows):
    SS_total = None
    for name, SSv, dfv, MSv, Fv, pv in anova_table_rows:
        if name == "Загальна":
            SS_total = SSv

    if SS_total is None or (isinstance(SS_total, float) and math.isnan(SS_total)) or SS_total <= 0:
        SS_total = np.nan

    rows = []
    for name, SSv, dfv, MSv, Fv, pv in anova_table_rows:
        if SSv is None or (isinstance(SSv, float) and math.isnan(SSv)):
            continue
        if name == "Загальна":
            pct = 100.0 if not math.isnan(SS_total) else np.nan
        else:
            pct = (SSv / SS_total * 100.0) if (not math.isnan(SS_total) and SS_total > 0) else np.nan
        rows.append([name, fmt_num(pct, 2)])
    return rows


def build_partial_eta2_rows_with_label(anova_table_rows):
    SS_error = None
    # Prefer classic "Залишок"/"Error(B)" as residual
    for name, SSv, dfv, MSv, Fv, pv in anova_table_rows:
        if name in ("Залишок", "Error(B)"):
            SS_error = SSv
            break

    if SS_error is None or (isinstance(SS_error, float) and math.isnan(SS_error)) or SS_error <= 0:
        SS_error = np.nan

    rows = []
    for name, SSv, dfv, MSv, Fv, pv in anova_table_rows:
        if name in ("Залишок", "Error(B)", "Загальна") or name.startswith("Error(") or name.startswith("Блоки"):
            continue
        if SSv is None or (isinstance(SSv, float) and math.isnan(SSv)):
            continue

        if math.isnan(SS_error):
            pe2 = np.nan
        else:
            denom = SSv + SS_error
            pe2 = (SSv / denom) if denom > 0 else np.nan

        rows.append([name, fmt_num(pe2, 4), partial_eta2_label(pe2)])
    return rows


# -------------------------
# Ranks
# -------------------------
def mean_ranks_by_key(long, key_func):
    vals = []
    keys = []
    for rec in long:
        v = rec.get("value", np.nan)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        vals.append(float(v))
        keys.append(key_func(rec))
    if len(vals) == 0:
        return {}
    ranks = rankdata(vals, method="average")
    sums = defaultdict(float)
    cnts = defaultdict(int)
    for k, r_ in zip(keys, ranks):
        sums[k] += float(r_)
        cnts[k] += 1
    return {k: (sums[k] / cnts[k]) for k in cnts.keys()}


# -------------------------
# GUI
# -------------------------
class SADTk:
    def __init__(self, root):
        self.root = root
        root.title("S.A.D. — Статистичний аналіз даних")
        root.geometry("1000x560")
        set_window_icon(root)

        self.base_font = ("Times New Roman", 15)
        root.option_add("*Font", self.base_font)
        root.option_add("*Foreground", "#000000")

        try:
            style = ttk.Style()
            style.theme_use("clam")
        except Exception:
            pass

        self.main_frame = tk.Frame(root, bg="white")
        self.main_frame.pack(expand=True, fill=tk.BOTH)

        tk.Label(
            self.main_frame,
            text="S.A.D. — Статистичний аналіз даних",
            font=("Times New Roman", 20, "bold"),
            fg="#000000",
            bg="white"
        ).pack(pady=18)

        btn_frame = tk.Frame(self.main_frame, bg="white")
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Однофакторний аналіз", width=22, height=2,
                  command=lambda: self.open_table(1)).grid(row=0, column=0, padx=10, pady=8)
        tk.Button(btn_frame, text="Двофакторний аналіз", width=22, height=2,
                  command=lambda: self.open_table(2)).grid(row=0, column=1, padx=10, pady=8)
        tk.Button(btn_frame, text="Трифакторний аналіз", width=22, height=2,
                  command=lambda: self.open_table(3)).grid(row=1, column=0, padx=10, pady=8)
        tk.Button(btn_frame, text="Чотирифакторний аналіз", width=22, height=2,
                  command=lambda: self.open_table(4)).grid(row=1, column=1, padx=10, pady=8)

        tk.Label(
            self.main_frame,
            text="Виберіть тип аналізу → Внесіть дані → Натисніть «Аналіз даних»",
            fg="#000000",
            bg="white"
        ).pack(pady=10)

        self.table_win = None
        self.report_win = None

    # --- NEW: indicator + units + design + (if splitplot) factor mapping ---
    def ask_indicator_units_and_design(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Параметри звіту")
        dlg.resizable(False, False)
        set_window_icon(dlg)

        frm = tk.Frame(dlg, padx=16, pady=16)
        frm.pack(fill=tk.BOTH, expand=True)

        tk.Label(frm, text="Назва показника:", fg="#000000").grid(row=0, column=0, sticky="w", pady=6)
        e_ind = tk.Entry(frm, width=40, fg="#000000")
        e_ind.grid(row=0, column=1, pady=6)

        tk.Label(frm, text="Одиниці виміру:", fg="#000000").grid(row=1, column=0, sticky="w", pady=6)
        e_units = tk.Entry(frm, width=40, fg="#000000")
        e_units.grid(row=1, column=1, pady=6)

        tk.Label(frm, text="Тип дизайну:", fg="#000000").grid(row=2, column=0, sticky="w", pady=6)
        design_var = tk.StringVar(value="crd")
        cb_design = ttk.Combobox(frm, textvariable=design_var, width=37, state="readonly",
                                 values=["crd", "splitplot"])
        cb_design.grid(row=2, column=1, pady=6)
        tk.Label(frm, text="crd = повна рандомізація; splitplot = розщеплений (2 фактори)", fg="#555555").grid(
            row=3, column=1, sticky="w", pady=(0, 6)
        )

        # Split-plot mapping (shown only if splitplot selected)
        sp_frame = tk.LabelFrame(frm, text="Split-plot параметри (обов'язково)", padx=10, pady=10)
        sp_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        sp_frame.grid_remove()

        main_var = tk.StringVar(value="")
        sub_var = tk.StringVar(value="")

        tk.Label(sp_frame, text="Main-plot фактор (головний):").grid(row=0, column=0, sticky="w", pady=4)
        cb_main = ttk.Combobox(sp_frame, textvariable=main_var, width=28, state="readonly")
        cb_main.grid(row=0, column=1, pady=4)

        tk.Label(sp_frame, text="Sub-plot фактор (підділянковий):").grid(row=1, column=0, sticky="w", pady=4)
        cb_sub = ttk.Combobox(sp_frame, textvariable=sub_var, width=28, state="readonly")
        cb_sub.grid(row=1, column=1, pady=4)

        def refresh_sp_options():
            # available factor keys in current table
            keys = getattr(self, "factor_keys", [])
            cb_main["values"] = keys
            cb_sub["values"] = keys
            if len(keys) >= 2:
                if not main_var.get():
                    main_var.set(keys[0])
                if not sub_var.get():
                    sub_var.set(keys[1])

        def on_design_change(*_):
            if design_var.get() == "splitplot":
                sp_frame.grid()
                refresh_sp_options()
            else:
                sp_frame.grid_remove()

        cb_design.bind("<<ComboboxSelected>>", lambda e: on_design_change())
        on_design_change()

        out = {"ok": False, "indicator": "", "units": "", "design": "crd", "main": "", "sub": ""}

        def on_ok():
            out["indicator"] = e_ind.get().strip()
            out["units"] = e_units.get().strip()
            out["design"] = design_var.get().strip()

            if not out["indicator"] or not out["units"] or not out["design"]:
                messagebox.showwarning("Помилка", "Заповніть назву показника, одиниці виміру та тип дизайну.")
                return

            if out["design"] == "splitplot":
                if getattr(self, "factors_count", 0) != 2:
                    messagebox.showwarning("Обмеження", "Split-plot наразі підтримано для 2 факторів.\n"
                                                      "Виберіть двофакторний аналіз або використайте CRD.")
                    return
                out["main"] = main_var.get().strip()
                out["sub"] = sub_var.get().strip()
                if not out["main"] or not out["sub"] or out["main"] == out["sub"]:
                    messagebox.showwarning("Помилка", "Для Split-plot оберіть різні Main-plot та Sub-plot фактори.")
                    return

            out["ok"] = True
            dlg.destroy()

        btns = tk.Frame(frm)
        btns.grid(row=5, column=0, columnspan=2, pady=(12, 0))
        tk.Button(btns, text="OK", width=10, command=on_ok).pack(side=tk.LEFT, padx=6)
        tk.Button(btns, text="Скасувати", width=12, command=lambda: dlg.destroy()).pack(side=tk.LEFT, padx=6)

        dlg.update_idletasks()
        center_window(dlg)
        e_ind.focus_set()
        dlg.bind("<Return>", lambda e: on_ok())
        dlg.grab_set()
        self.root.wait_window(dlg)
        return out

    def choose_method_window(self, p_norm):
        dlg = tk.Toplevel(self.root)
        dlg.title("Вибір виду аналізу")
        dlg.resizable(False, False)
        set_window_icon(dlg)

        frm = tk.Frame(dlg, padx=16, pady=14)
        frm.pack(fill=tk.BOTH, expand=True)

        normal = (p_norm is not None) and (not math.isnan(p_norm)) and (p_norm > 0.05)

        if normal:
            msg = ("Дані експерименту відповідають принципам нормального розподілу\n"
                   "за методом Шапіра-Вілка.")
            tk.Label(frm, text=msg, fg="#000000", justify="left").pack(anchor="w", pady=(0, 10))
            options = [
                ("НІР₀₅", "lsd"),
                ("Тест Тьюкі", "tukey"),
                ("Тест Дункана", "duncan"),
                ("Тест Бонферроні", "bonferroni"),
            ]
        else:
            msg = ("Дані експерименту не відповідають принципам нормального розподілу\n"
                   "за методом Шапіра-Вілка.\n"
                   "Виберіть один з непараметричних типів аналізу.")
            tk.Label(frm, text=msg, fg="#c62828", justify="left").pack(anchor="w", pady=(0, 10))
            options = [
                ("Краскела–Уолліса", "kw"),
                ("Манна-Уітні", "mw"),
            ]

        var = tk.StringVar(value=options[0][1])
        for text, val in options:
            tk.Radiobutton(frm, text=text, variable=var, value=val).pack(anchor="w", pady=2)

        out = {"ok": False, "method": None}

        def on_ok():
            out["ok"] = True
            out["method"] = var.get()
            dlg.destroy()

        btns = tk.Frame(frm)
        btns.pack(fill=tk.X, pady=(12, 0))
        tk.Button(btns, text="OK", width=10, command=on_ok).pack(side=tk.LEFT, padx=6)
        tk.Button(btns, text="Скасувати", width=12, command=lambda: dlg.destroy()).pack(side=tk.LEFT, padx=6)

        dlg.update_idletasks()
        center_window(dlg)
        dlg.bind("<Return>", lambda e: on_ok())
        dlg.grab_set()
        self.root.wait_window(dlg)
        return out

    def open_table(self, factors_count):
        if self.table_win and tk.Toplevel.winfo_exists(self.table_win):
            self.table_win.destroy()

        self.factors_count = factors_count
        self.factor_keys = ["A", "B", "C", "D"][:factors_count]

        self.table_win = tk.Toplevel(self.root)
        self.table_win.title(f"S.A.D. — {factors_count}-факторний аналіз")
        self.table_win.geometry("1280x720")
        set_window_icon(self.table_win)

        self.repeat_count = 6
        self.factor_names = [f"Фактор {self.factor_keys[i]}" for i in range(factors_count)]
        self.column_names = self.factor_names + [f"Повт.{i+1}" for i in range(self.repeat_count)]

        ctl = tk.Frame(self.table_win, padx=6, pady=6)
        ctl.pack(fill=tk.X)

        btn_texts = [
            "Додати рядок", "Видалити рядок",
            "Додати стовпчик", "Видалити стовпчик",
            "Вставити з буфера", "Аналіз даних",
            "Розробник",
        ]
        btn_font = fit_font_size_to_texts(btn_texts, family="Times New Roman", start=14, min_size=9, target_px=150)

        bw = 16
        bh = 1
        padx = 3
        pady = 2

        tk.Button(ctl, text="Додати рядок", width=bw, height=bh, font=btn_font,
                  command=self.add_row).pack(side=tk.LEFT, padx=padx, pady=pady)
        tk.Button(ctl, text="Видалити рядок", width=bw, height=bh, font=btn_font,
                  command=self.delete_row).pack(side=tk.LEFT, padx=padx, pady=pady)
        tk.Button(ctl, text="Додати стовпчик", width=bw, height=bh, font=btn_font,
                  command=self.add_column).pack(side=tk.LEFT, padx=(10, padx), pady=pady)
        tk.Button(ctl, text="Видалити стовпчик", width=bw, height=bh, font=btn_font,
                  command=self.delete_column).pack(side=tk.LEFT, padx=padx, pady=pady)
        tk.Button(ctl, text="Вставити з буфера", width=bw + 2, height=bh, font=btn_font,
                  command=self.paste_from_focus).pack(side=tk.LEFT, padx=(10, padx), pady=pady)
        tk.Button(ctl, text="Аналіз даних", width=bw, height=bh, font=btn_font,
                  bg="#c62828", fg="white", command=self.analyze).pack(side=tk.LEFT, padx=(10, padx), pady=pady)
        tk.Button(ctl, text="Розробник", width=bw, height=bh, font=btn_font,
                  command=self.show_about).pack(side=tk.RIGHT, padx=padx, pady=pady)

        self.canvas = tk.Canvas(self.table_win)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sb = ttk.Scrollbar(self.table_win, orient="vertical", command=self.canvas.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=sb.set)

        self.inner = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.rows = 12
        self.cols = len(self.column_names)
        self.entries = []
        self.header_labels = []

        for j, name in enumerate(self.column_names):
            lbl = tk.Label(self.inner, text=name, relief=tk.RIDGE, width=COL_W, bg="#f0f0f0", fg="#000000")
            lbl.grid(row=0, column=j, padx=2, pady=2, sticky="nsew")
            self.header_labels.append(lbl)

        for i in range(self.rows):
            row_entries = []
            for j in range(self.cols):
                e = tk.Entry(self.inner, width=COL_W, fg="#000000")
                e.grid(row=i + 1, column=j, padx=2, pady=2)
                self.bind_cell(e)
                row_entries.append(e)
            self.entries.append(row_entries)

        self.inner.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

        self.entries[0][0].focus_set()

        self.table_win.bind("<Control-v>", self.on_paste)
        self.table_win.bind("<Control-V>", self.on_paste)

    def show_about(self):
        messagebox.showinfo("Розробник", "S.A.D. — Статистичний аналіз даних\nВерсія: 1.0\nРозробик: Чаплоуцький Андрій Миколайович\nУманський національний університет")

    def bind_cell(self, e: tk.Entry):
        e.bind("<Return>", self.on_enter)
        e.bind("<Up>", self.on_arrow)
        e.bind("<Down>", self.on_arrow)
        e.bind("<Left>", self.on_arrow)
        e.bind("<Right>", self.on_arrow)
        e.bind("<Control-v>", self.on_paste)
        e.bind("<Control-V>", self.on_paste)

    def add_row(self):
        i = len(self.entries)
        row_entries = []
        for j in range(self.cols):
            e = tk.Entry(self.inner, width=COL_W, fg="#000000")
            e.grid(row=i + 1, column=j, padx=2, pady=2)
            self.bind_cell(e)
            row_entries.append(e)
        self.entries.append(row_entries)
        self.rows += 1
        self.inner.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def delete_row(self):
        if not self.entries:
            return
        last = self.entries.pop()
        for e in last:
            e.destroy()
        self.rows -= 1
        self.inner.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def add_column(self):
        self.cols += 1
        col_idx = self.cols - 1
        name = f"Повт.{col_idx - self.factors_count + 1}"

        lbl = tk.Label(self.inner, text=name, relief=tk.RIDGE, width=COL_W, bg="#f0f0f0", fg="#000000")
        lbl.grid(row=0, column=col_idx, padx=2, pady=2, sticky="nsew")
        self.header_labels.append(lbl)

        for i, row in enumerate(self.entries):
            e = tk.Entry(self.inner, width=COL_W, fg="#000000")
            e.grid(row=i + 1, column=col_idx, padx=2, pady=2)
            self.bind_cell(e)
            row.append(e)

        self.inner.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def delete_column(self):
        if self.cols <= self.factors_count + 1:
            return
        if self.header_labels:
            lbl = self.header_labels.pop()
            lbl.destroy()
        for row in self.entries:
            w = row.pop()
            w.destroy()
        self.cols -= 1
        self.inner.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def find_pos(self, widget):
        for i, row in enumerate(self.entries):
            for j, cell in enumerate(row):
                if cell is widget:
                    return i, j
        return None

    def on_enter(self, event=None):
        pos = self.find_pos(event.widget)
        if not pos:
            return "break"
        i, j = pos
        ni = i + 1
        if ni >= len(self.entries):
            self.add_row()
        self.entries[ni][j].focus_set()
        self.entries[ni][j].icursor(tk.END)
        return "break"

    def on_arrow(self, event=None):
        pos = self.find_pos(event.widget)
        if not pos:
            return "break"
        i, j = pos
        if event.keysym == "Up":
            i = max(0, i - 1)
        elif event.keysym == "Down":
            i = min(len(self.entries) - 1, i + 1)
        elif event.keysym == "Left":
            j = max(0, j - 1)
        elif event.keysym == "Right":
            j = min(len(self.entries[i]) - 1, j + 1)
        self.entries[i][j].focus_set()
        self.entries[i][j].icursor(tk.END)
        return "break"

    def paste_from_focus(self):
        w = self.table_win.focus_get()
        if isinstance(w, tk.Entry):
            class E: pass
            ev = E()
            ev.widget = w
            self.on_paste(ev)

    def on_paste(self, event=None):
        widget = event.widget if event else self.table_win.focus_get()
        if not isinstance(widget, tk.Entry):
            return "break"
        try:
            data = self.table_win.clipboard_get()
        except Exception:
            return "break"

        rows_text = [r for r in data.splitlines() if r != ""]
        pos = self.find_pos(widget)
        if not pos:
            return "break"
        r0, c0 = pos

        for i_r, row_text in enumerate(rows_text):
            cols = row_text.split("\t")
            for j_c, val in enumerate(cols):
                rr = r0 + i_r
                cc = c0 + j_c
                while rr >= len(self.entries):
                    self.add_row()
                if cc >= self.cols:
                    continue
                self.entries[rr][cc].delete(0, tk.END)
                self.entries[rr][cc].insert(0, val)

        return "break"

    def used_repeat_columns(self):
        rep_start = self.factors_count
        rep_cols = list(range(rep_start, self.cols))
        used = []
        for c in rep_cols:
            any_num = False
            for r in range(len(self.entries)):
                s = self.entries[r][c].get().strip()
                if not s:
                    continue
                try:
                    float(s.replace(",", "."))
                    any_num = True
                    break
                except Exception:
                    continue
            if any_num:
                used.append(c)
        return used

    # --- UPDATED: add Rep field (repeat column index) ---
    def collect_long(self):
        long = []
        rep_cols = self.used_repeat_columns()
        if not rep_cols:
            return long, rep_cols

        # map used rep cols to Rep levels R1, R2...
        rep_level = {c: f"R{idx+1}" for idx, c in enumerate(rep_cols)}

        for i, row in enumerate(self.entries):
            levels = []
            for k in range(self.factors_count):
                v = row[k].get().strip()
                if v == "":
                    v = f"рядок{i+1}"
                levels.append(v)

            for c in rep_cols:
                s = row[c].get().strip()
                if not s:
                    continue
                try:
                    val = float(s.replace(",", "."))
                except Exception:
                    continue

                rec = {"value": val, "Rep": rep_level[c]}
                if self.factors_count >= 1: rec["A"] = levels[0]
                if self.factors_count >= 2: rec["B"] = levels[1]
                if self.factors_count >= 3: rec["C"] = levels[2]
                if self.factors_count >= 4: rec["D"] = levels[3]
                long.append(rec)

        return long, rep_cols

    def analyze(self):
        created_at = datetime.now()

        params = self.ask_indicator_units_and_design()
        if not params["ok"]:
            return

        indicator = params["indicator"]
        units = params["units"]
        design = params["design"]
        sp_main = params.get("main", "")
        sp_sub = params.get("sub", "")

        long, used_rep_cols = self.collect_long()
        if len(long) == 0:
            messagebox.showwarning("Помилка", "Немає числових даних для аналізу.\nПеревірте повторності та значення.")
            return

        values = np.array([r["value"] for r in long], dtype=float)
        if len(values) < 3:
            messagebox.showinfo("Результат", "Надто мало даних для аналізу.")
            return

        levels_by_factor = {f: first_seen_order([r.get(f) for r in long]) for f in self.factor_keys}
        # Rep levels for splitplot
        rep_levels = first_seen_order([r.get("Rep") for r in long])

        # ---- Choose engine: CRD or Split-plot (2f) ----
        res = None
        splitplot_mode = (design == "splitplot" and self.factors_count == 2)

        try:
            if splitplot_mode:
                # split-plot needs Rep levels >=2
                if len(rep_levels) < 2:
                    messagebox.showwarning("Помилка", "Для Split-plot потрібно мінімум 2 повторення (Rep).\n"
                                                    "Заповніть хоча б 2 стовпчики повторностей.")
                    return
                res = anova_splitplot_2f(
                    long,
                    rep_key="Rep",
                    A_key=sp_main,
                    B_key=sp_sub,
                    levels_by_factor=levels_by_factor
                )
            else:
                if design == "splitplot" and self.factors_count != 2:
                    messagebox.showwarning("Обмеження", "Split-plot наразі підтримано лише для 2 факторів.\n"
                                                      "Аналіз буде виконано як CRD (повна рандомізація).")
                res = anova_n_way(long, self.factor_keys, levels_by_factor)
        except Exception as ex:
            messagebox.showerror("Помилка аналізу", str(ex))
            return

        # residuals for Shapiro
        residuals = []
        if splitplot_mode:
            # Use Rep×A×B cell means
            cell_means = res.get("cell_means_repAB", {})
            for rec in long:
                key = (rec.get("Rep"), rec.get(sp_main), rec.get(sp_sub))
                v = rec.get("value", np.nan)
                m = cell_means.get(key, np.nan)
                if not math.isnan(v) and not math.isnan(m):
                    residuals.append(v - m)
        else:
            cell_means = res.get("cell_means", {})
            for rec in long:
                key = tuple(rec.get(f) for f in self.factor_keys)
                v = rec.get("value", np.nan)
                m = cell_means.get(key, np.nan)
                if not math.isnan(v) and not math.isnan(m):
                    residuals.append(v - m)
        residuals = np.array(residuals, dtype=float)

        try:
            W, p_norm = shapiro(residuals) if len(residuals) >= 3 else (np.nan, np.nan)
        except Exception:
            W, p_norm = (np.nan, np.nan)

        choice = self.choose_method_window(p_norm)
        if not choice["ok"]:
            return
        method = choice["method"]

        # Split-plot: only parametric modes (for correct ANOVA)
        if splitplot_mode and method in ("kw", "mw"):
            messagebox.showwarning("Обмеження", "Split-plot наразі підтримано лише для параметричних методів\n"
                                              "(НІР₀₅ / Тьюкі / Дункан / Бонферроні).")
            return

        # MS/df for posthoc
        if splitplot_mode:
            MS_error_A = res.get("MS_error_A", np.nan)
            df_error_A = res.get("df_error_A", np.nan)
            MS_error_B = res.get("MS_error_B", np.nan)
            df_error_B = res.get("df_error_B", np.nan)
        else:
            MS_error = res.get("MS_error", np.nan)
            df_error = res.get("df_error", np.nan)

        # factor groups/means
        factor_groups = {f: {k[0]: v for k, v in groups_by_keys(long, (f,)).items()} for f in self.factor_keys}
        factor_means = {f: {lvl: float(np.mean(arr)) if len(arr) else np.nan for lvl, arr in factor_groups[f].items()} for f in self.factor_keys}
        factor_ns = {f: {lvl: len(arr) for lvl, arr in factor_groups[f].items()} for f in self.factor_keys}

        # For split-plot, letters must use different errors:
        letters_factor = {}
        if method == "lsd":
            for f in self.factor_keys:
                lvls = levels_by_factor[f]
                if splitplot_mode:
                    if f == sp_main:
                        sig = lsd_sig_matrix(lvls, factor_means[f], factor_ns[f], MS_error_A, df_error_A, alpha=ALPHA)
                    elif f == sp_sub:
                        sig = lsd_sig_matrix(lvls, factor_means[f], factor_ns[f], MS_error_B, df_error_B, alpha=ALPHA)
                    else:
                        sig = {}
                else:
                    sig = lsd_sig_matrix(lvls, factor_means[f], factor_ns[f], MS_error, df_error, alpha=ALPHA)
                letters_factor[f] = cld_multi_letters(lvls, factor_means[f], sig)
        else:
            for f in self.factor_keys:
                letters_factor[f] = {lvl: "" for lvl in levels_by_factor[f]}

        # Variant analysis (full combinations) — for split-plot we DO NOT do posthoc across variants (to avoid wrong error term)
        variant_order = first_seen_order([tuple(r.get(f) for f in self.factor_keys) for r in long])
        v_names = [" | ".join(map(str, k)) for k in variant_order]
        num_variants = len(variant_order)

        vstats = variant_mean_sd(long, self.factor_keys)
        v_means = {k: vstats[k][0] for k in vstats.keys()}
        v_sds = {k: vstats[k][1] for k in vstats.keys()}
        v_ns = {k: vstats[k][2] for k in vstats.keys()}

        letters_variants = {k: "" for k in variant_order}
        pairwise_rows = []
        do_posthoc = True

        # Brown–Forsythe for variants (only CRD parametric)
        bf_F, bf_p = (np.nan, np.nan)
        if (not splitplot_mode) and method in ("lsd", "tukey", "duncan", "bonferroni"):
            groups1 = {v_names[i]: groups_by_keys(long, tuple(self.factor_keys)).get(variant_order[i], []) for i in range(len(variant_order))}
            bf_F, bf_p = brown_forsythe_from_groups(groups1)

        # KW/MW sections (CRD)
        kw_H, kw_p, kw_df, kw_eps2 = (np.nan, np.nan, np.nan, np.nan)
        if (not splitplot_mode) and method == "kw":
            groups1 = {v_names[i]: groups_by_keys(long, tuple(self.factor_keys)).get(variant_order[i], []) for i in range(len(variant_order))}
            try:
                kw_samples = [groups1[name] for name in v_names if len(groups1[name]) > 0]
                if len(kw_samples) >= 2:
                    kw_res = kruskal(*kw_samples)
                    kw_H = float(kw_res.statistic)
                    kw_p = float(kw_res.pvalue)
                    kw_df = int(len(kw_samples) - 1)
                    kw_eps2 = epsilon_squared_kw(kw_H, n=len(long), k=len(kw_samples))
            except Exception:
                kw_H, kw_p, kw_df, kw_eps2 = (np.nan, np.nan, np.nan, np.nan)

        # R2-like (for split-plot keep residual Error(B))
        SS_total = res.get("SS_total", np.nan)
        SS_error = res.get("SS_error", np.nan)
        R2 = (1.0 - (SS_error / SS_total)) if (not any(math.isnan(x) for x in [SS_total, SS_error]) and SS_total > 0) else np.nan

        # CV
        cv_rows = []
        for f in self.factor_keys:
            lvl_means = [factor_means[f].get(lvl, np.nan) for lvl in levels_by_factor[f]]
            cv_f = cv_percent_from_level_means(lvl_means)
            cv_rows.append([f"Фактор {f}", fmt_num(cv_f, 2)])
        cv_total = cv_percent_from_values(values)
        cv_rows.append(["Загальний", fmt_num(cv_total, 2)])

        # -------------------------
        # REPORT segments
        # -------------------------
        seg = []
        seg.append(("text", "З В І Т   С Т А Т И С Т И Ч Н О Г О   А Н А Л І З У   Д А Н И Х\n\n"))
        seg.append(("text", f"Показник:\t{indicator}\nОдиниці виміру:\t{units}\n\n"))

        if splitplot_mode:
            seg.append(("text", "Тип дизайну:\tSplit-plot (розщеплений)\n"))
            seg.append(("text", f"Main-plot фактор:\t{sp_main}\nSub-plot фактор:\t{sp_sub}\n"))
            seg.append(("text", "Принцип тестування:\tA тестується за Error(A)=Rep×A; B та A×B — за Error(B).\n\n"))
            seg.append(("text", f"Кількість повторень (Rep):\t{len(rep_levels)}\n"))
        else:
            seg.append(("text", "Тип дизайну:\tCRD (повна рандомізація)\n\n"))

        seg.append(("text", f"Кількість варіантів:\t{num_variants}\nКількість повторностей:\t{len(used_rep_cols)}\nЗагальна кількість облікових значень:\t{len(long)}\n\n"))

        if not math.isnan(W):
            seg.append(("text", f"Перевірка нормальності залишків (Shapiro–Wilk):\t{normality_text(p_norm)}\t(W={fmt_num(float(W),4)}; p={fmt_num(float(p_norm),4)})\n\n"))
        else:
            seg.append(("text", "Перевірка нормальності залишків (Shapiro–Wilk):\tн/д\n\n"))

        method_label = {
            "lsd": "Параметричний аналіз: ANOVA + НІР₀₅ (LSD).",
            "tukey": "Параметричний аналіз: ANOVA + тест Тьюкі (Tukey HSD).",
            "duncan": "Параметричний аналіз: ANOVA + тест Дункана.",
            "bonferroni": "Параметричний аналіз: ANOVA + корекція Бонферроні.",
            "kw": "Непараметричний аналіз: Kruskal–Wallis.",
            "mw": "Непараметричний аналіз: Mann–Whitney.",
        }.get(method, "")

        if method_label:
            seg.append(("text", f"Виконуваний статистичний аналіз:\t{method_label}\n\n"))

        seg.append(("text", "Пояснення позначень істотності: ** — p<0.01; * — p<0.05.\n"))
        seg.append(("text", "У таблицях знак \"-\" свідчить що p ≥ 0.05.\n"))
        seg.append(("text", "Істотна різниця (літери): різні літери свідчать про наявність істотної різниці.\n\n"))

        nonparam = method in ("mw", "kw")

        # ---- PARAMETRIC
        if not nonparam:
            # ANOVA table
            anova_rows = []
            for name, SSv, dfv, MSv, Fv, pv in res["table"]:
                df_txt = str(int(dfv)) if dfv is not None and not (isinstance(dfv, float) and math.isnan(dfv)) else ""
                if name in ("Залишок", "Загальна", "Error(B)") or name.startswith("Error(") or name.startswith("Блоки"):
                    if name == "Загальна":
                        anova_rows.append([name, fmt_num(SSv, 2), df_txt, "", "", "", ""])
                    else:
                        anova_rows.append([name, fmt_num(SSv, 2), df_txt, fmt_num(MSv, 3), "", "", ""])
                else:
                    mark = significance_mark(pv)
                    concl = f"істотна різниця {mark}" if mark else "-"
                    anova_rows.append([name, fmt_num(SSv, 2), df_txt, fmt_num(MSv, 3), fmt_num(Fv, 3), fmt_num(pv, 4), concl])

            seg.append(("text", "ТАБЛИЦЯ 1. Дисперсійний аналіз (ANOVA)\n"))
            seg.append(("table", {
                "headers": ["Джерело", "SS", "df", "MS", "F", "p", "Висновок"],
                "rows": anova_rows,
                "padding_px": 32,
                "extra_gap_after_col": 0,
                "extra_gap_px": 60
            }))
            seg.append(("text", "\n"))

            eff_rows = build_effect_strength_rows(res["table"])
            seg.append(("text", "ТАБЛИЦЯ 2. Сила впливу факторів та їх комбінацій (% від SS)\n"))
            seg.append(("table", {"headers": ["Джерело", "%"], "rows": eff_rows}))
            seg.append(("text", "\n"))

            pe2_rows = build_partial_eta2_rows_with_label(res["table"])
            seg.append(("text", "ТАБЛИЦЯ 3. Розмір ефекту (partial η²)\n"))
            seg.append(("table", {"headers": ["Джерело", "partial η²", "Висновок"], "rows": pe2_rows}))
            seg.append(("text", "\n"))

            seg.append(("text", "ТАБЛИЦЯ 4. Коефіцієнт варіації (CV, %)\n"))
            seg.append(("table", {"headers": ["Елемент", "CV, %"], "rows": cv_rows}))
            seg.append(("text", "\n"))

            seg.append(("text", f"Коефіцієнт детермінації:\tR²={fmt_num(R2, 4)}\n\n"))

            tno = 5
            if method == "lsd":
                nir_rows = []
                if splitplot_mode:
                    nir_rows.append([f"Фактор {sp_main}", fmt_num(res["NIR05"].get(f"Фактор {sp_main}", np.nan), 4)])
                    nir_rows.append([f"Фактор {sp_sub}", fmt_num(res["NIR05"].get(f"Фактор {sp_sub}", np.nan), 4)])
                else:
                    for key in [f"Фактор {f}" for f in self.factor_keys] + ["Загальна"]:
                        if key in res.get("NIR05", {}):
                            nir_rows.append([key, fmt_num(res["NIR05"][key], 4)])
                seg.append(("text", "ТАБЛИЦЯ 5. Значення НІР₀₅\n"))
                seg.append(("table", {"headers": ["Елемент", "НІР₀₅"], "rows": nir_rows}))
                seg.append(("text", "\n"))
                tno = 6

            # Factor means + letters
            for f in self.factor_keys:
                seg.append(("text", f"ТАБЛИЦЯ {tno}. Середнє по фактору {f}\n"))
                rows_f = []
                for lvl in levels_by_factor[f]:
                    m = factor_means[f].get(lvl, np.nan)
                    letter = letters_factor[f].get(lvl, "")
                    rows_f.append([str(lvl), fmt_num(m, 3), (letter if letter else "-")])
                seg.append(("table", {"headers": [f"Градація {f}", "Середнє", "Істотна різниця"], "rows": rows_f}))
                seg.append(("text", "\n"))
                tno += 1

            # Variants table (no letters in split-plot mode to avoid wrong error term)
            seg.append(("text", f"ТАБЛИЦЯ {tno}. Таблиця середніх значень варіантів\n"))
            rows_v = []
            for k in variant_order:
                name = " | ".join(map(str, k))
                m = v_means.get(k, np.nan)
                sd = v_sds.get(k, np.nan)
                letter = letters_variants.get(k, "")
                rows_v.append([name, fmt_num(m, 3), fmt_num(sd, 3), (letter if letter else "-")])

            seg.append(("table", {
                "headers": ["Варіант", "Середнє", "± SD", "Істотна різниця"],
                "rows": rows_v,
                "padding_px": 32,
                "extra_gap_after_col": 0,
                "extra_gap_px": 80
            }))
            seg.append(("text", "\n"))

            if splitplot_mode:
                seg.append(("text",
                            "Примітка: у режимі Split-plot пост-хок порівняння між повними комбінаціями (варіантами) не виконуються,\n"
                            "оскільки для коректності потрібен окремий вибір терму похибки залежно від рівня рандомізації.\n\n"))

        # ---- NONPARAMETRIC (CRD only)
        else:
            seg.append(("text", "Непараметричні методи наразі виконуються у припущенні CRD (повна рандомізація).\n\n"))

        seg.append(("text", f"Звіт сформовано:\t{created_at.strftime('%d.%m.%Y, %H:%M')}\n"))
        self.show_report_segments(seg)

    def show_report_segments(self, segments):
        if self.report_win and tk.Toplevel.winfo_exists(self.report_win):
            self.report_win.destroy()

        self.report_win = tk.Toplevel(self.root)
        self.report_win.title("Звіт")
        self.report_win.geometry("1180x760")
        set_window_icon(self.report_win)

        top = tk.Frame(self.report_win, padx=8, pady=8)
        top.pack(fill=tk.X)

        xsb = ttk.Scrollbar(self.report_win, orient="horizontal")
        xsb.pack(side=tk.BOTTOM, fill=tk.X)

        txt = ScrolledText(self.report_win, width=120, height=40, wrap="none", xscrollcommand=xsb.set)
        txt.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        xsb.config(command=txt.xview)

        txt.configure(font=("Times New Roman", 14), fg="#000000")
        font_obj = tkfont.Font(font=("Times New Roman", 14))

        table_idx = 0
        for kind, payload in segments:
            if kind == "text":
                txt.insert("end", payload)
                continue

            if isinstance(payload, dict):
                headers = payload.get("headers", [])
                rows = payload.get("rows", [])
                padding_px = int(payload.get("padding_px", 32))
                extra_after = payload.get("extra_gap_after_col", None)
                extra_px = int(payload.get("extra_gap_px", 0))
            else:
                headers, rows = payload
                padding_px = 32
                extra_after = None
                extra_px = 0

            tabs = tabs_from_table_px(
                font_obj, headers, rows,
                padding_px=padding_px,
                extra_gap_after_col=extra_after,
                extra_gap_px=extra_px
            )

            tag = f"tbl_{table_idx}"
            table_idx += 1
            txt.tag_configure(tag, tabs=tabs)

            start = txt.index("end")
            txt.insert("end", build_table_block(headers, rows))
            end = txt.index("end")
            txt.tag_add(tag, start, end)

        def copy_report():
            self.report_win.clipboard_clear()
            self.report_win.clipboard_append(txt.get("1.0", "end-1c"))
            messagebox.showinfo("Готово", "Звіт скопійовано в буфер обміну.")

        tk.Button(top, text="Копіювати звіт", command=copy_report).pack(side=tk.LEFT, padx=4)

        def on_ctrl_c(event=None):
            try:
                sel = txt.get("sel.first", "sel.last")
            except Exception:
                sel = txt.get("1.0", "end-1c")
            self.report_win.clipboard_clear()
            self.report_win.clipboard_append(sel)
            return "break"

        txt.bind("<Control-c>", on_ctrl_c)
        txt.bind("<Control-C>", on_ctrl_c)


# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    root = tk.Tk()
    set_window_icon(root)
    app = SADTk(root)
    root.mainloop()
