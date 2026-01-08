# main.py
# -*- coding: utf-8 -*-

"""
S.A.D. — Статистичний аналіз даних (Tkinter)

Потрібно: Python 3.8+, numpy, scipy, matplotlib, pillow
Встановлення:
  pip install numpy scipy matplotlib pillow

Що додано/виправлено:
✅ Дизайн експерименту: CRD / RCBD / Split-plot (коректно)
✅ У вікні параметрів звіту додано вибір дизайну + (для Split-plot) вибір головного фактора
✅ Вікно параметрів автоматично змінює розмір, щоб усе влізло
✅ Split-plot звіт: %SS, partial η², CV (main-plot і sub-plot)
✅ Графічний звіт: один графік блоками факторів + літери над верхнім вусом + правильна легенда
✅ Копіювання графіка в буфер у форматі PNG (через CF_DIB) без помилки GlobalLock

(ДОДАНО В ЦІЙ ВЕРСІЇ:)
✅ Літери по факторах для ВСІХ параметричних методів (LSD / Tukey / Duncan / Bonferroni) — у Частині 2
✅ Порядок інформації у текстовому звіті як у вашому прикладі — у Частині 2
"""

import os
import sys
import math
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
import tkinter.font as tkfont
from collections import defaultdict
from itertools import combinations
from datetime import datetime

import numpy as np
from scipy.stats import shapiro, t, f as f_dist
from scipy.stats import mannwhitneyu, kruskal, levene, rankdata
from scipy.stats import studentized_range

# plotting
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from PIL import Image

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


def cv_from_ms(ms_error, grand_mean):
    if ms_error is None or grand_mean is None:
        return np.nan
    if any(isinstance(x, float) and math.isnan(x) for x in [ms_error, grand_mean]):
        return np.nan
    if grand_mean == 0:
        return np.nan
    if ms_error < 0:
        return np.nan
    return float((math.sqrt(ms_error) / grand_mean) * 100.0)


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
            r = (j - i) + 1

            q = abs(ma - mb) / se_base
            p = float(studentized_range.sf(q, r, df_error))
            qcrit = studentized_range.ppf(1 - alpha, r, df_error)
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
    for name, SSv, dfv, MSv, Fv, pv, denom in anova_table_rows:
        if name == "Загальна":
            SS_total = SSv

    if SS_total is None or (isinstance(SS_total, float) and math.isnan(SS_total)) or SS_total <= 0:
        SS_total = np.nan

    rows = []
    for name, SSv, dfv, MSv, Fv, pv, denom in anova_table_rows:
        if SSv is None or (isinstance(SSv, float) and math.isnan(SSv)):
            continue
        if name == "Загальна":
            pct = 100.0 if not math.isnan(SS_total) else np.nan
        else:
            pct = (SSv / SS_total * 100.0) if (not math.isnan(SS_total) and SS_total > 0) else np.nan
        rows.append([name, fmt_num(pct, 2)])
    return rows


def build_partial_eta2_rows_with_label_splitaware(anova_table_rows, ss_error_main, ss_error_sub):
    """
    anova_table_rows: [(name, SS, df, MS, F, p, denom)]
    denom: 'main' or 'sub' or None (для підсумків/блоків)
    partial η² для split-plot: SS_effect/(SS_effect+SS_error_stratum)
    """
    rows = []
    for name, SSv, dfv, MSv, Fv, pv, denom in anova_table_rows:
        if name in ("Загальна", "Залишок (sub-plot)", "Залишок (main-plot)", "Блоки"):
            continue
        if SSv is None or (isinstance(SSv, float) and math.isnan(SSv)):
            continue

        if denom == "main":
            SSerr = ss_error_main
        elif denom == "sub":
            SSerr = ss_error_sub
        else:
            SSerr = np.nan

        if SSerr is None or (isinstance(SSerr, float) and math.isnan(SSerr)) or SSerr <= 0:
            pe2 = np.nan
        else:
            pe2 = SSv / (SSv + SSerr) if (SSv + SSerr) > 0 else np.nan

        rows.append([name, fmt_num(pe2, 4), partial_eta2_label(pe2)])
    return rows


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
    for k, r in zip(keys, ranks):
        sums[k] += float(r)
        cnts[k] += 1
    return {k: (sums[k] / cnts[k]) for k in cnts.keys()}


# -------------------------
# Core SS for factorial effects (orthogonal, balanced-ish)
# -------------------------
def anova_effect_ss_fixed(long, factors, levels_by_factor):
    """
    Повертає SS для всіх ефектів факторів/взаємодій як у звичайній факторній ANOVA.
    (Блок/повторність тут НЕ враховується: це використовуємо як ортогональну частину treatment SS)
    """
    N = len(long)
    values = np.array([r["value"] for r in long], dtype=float)
    grand_mean = np.nanmean(values)

    k = len(factors)
    stats = {}
    for r in range(1, k + 1):
        for comb in combinations(factors, r):
            stats[comb] = subset_stats(long, comb)

    full = tuple(factors)
    cell_means = {kk: vv[0] for kk, vv in stats[full].items()}
    cell_counts = {kk: vv[1] for kk, vv in stats[full].items()}

    SS_total = np.nansum((values - grand_mean) ** 2)

    # CRD-like residual (не використовуємо для RCBD/Split як error, лише для residuals і нормальності)
    SS_error_crd = 0.0
    for rec in long:
        key = tuple(rec.get(f) for f in factors)
        v = rec.get("value", np.nan)
        m = cell_means.get(key, np.nan)
        if not math.isnan(v) and not math.isnan(m):
            SS_error_crd += (v - m) ** 2

    levels_count = {f: len(levels_by_factor[f]) for f in factors}

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

    return {
        "SS_total": float(SS_total),
        "SS_error_crd": float(SS_error_crd),
        "SS_effects": SS,
        "df_effects": df,
        "cell_means": cell_means,
        "cell_counts": cell_counts,
        "grand_mean": float(grand_mean),
    }


# -------------------------
# RCBD model (blocks = repeat columns)
# -------------------------
def rcbd_anova(long, factors, levels_by_factor):
    """
    RCBD: one observation per treatment combination per block (repeat column).
    Treatment SS orthogonal to block SS.
    """
    blocks = first_seen_order([r.get("BLOCK") for r in long])
    blocks = [b for b in blocks if b is not None]
    r = len(blocks)
    if r < 2:
        raise ValueError("RCBD потребує щонайменше 2 блоки (повторності).")

    treat_keys = tuple(factors)
    treat_combos = first_seen_order([tuple(rec.get(f) for f in treat_keys) for rec in long])
    tcount = len(treat_combos)
    if tcount < 2:
        raise ValueError("Недостатньо варіантів для RCBD.")

    values = np.array([rec["value"] for rec in long], dtype=float)
    grand = float(np.nanmean(values))
    SS_total = float(np.nansum((values - grand) ** 2))

    block_vals = defaultdict(list)
    for rec in long:
        b = rec.get("BLOCK")
        if b is None:
            continue
        v = rec.get("value", np.nan)
        if v is None or math.isnan(v):
            continue
        block_vals[b].append(float(v))
    block_mean = {b: float(np.mean(block_vals[b])) for b in block_vals if len(block_vals[b]) > 0}

    treat_vals = defaultdict(list)
    for rec in long:
        key = tuple(rec.get(f) for f in treat_keys)
        v = rec.get("value", np.nan)
        if v is None or math.isnan(v):
            continue
        treat_vals[key].append(float(v))
    treat_mean = {k: float(np.mean(treat_vals[k])) for k in treat_vals if len(treat_vals[k]) > 0}

    SS_block = 0.0
    for b in blocks:
        if b in block_mean:
            SS_block += (block_mean[b] - grand) ** 2
    SS_block *= tcount

    SS_treat = 0.0
    for k in treat_combos:
        if k in treat_mean:
            SS_treat += (treat_mean[k] - grand) ** 2
    SS_treat *= r

    SS_error = SS_total - SS_block - SS_treat
    if SS_error < -1e-9:
        SS_error = max(0.0, SS_error)

    df_block = r - 1
    df_treat = tcount - 1
    df_error = df_block * df_treat
    if df_error <= 0:
        df_error = 1

    MS_block = SS_block / df_block if df_block > 0 else np.nan
    MS_error = SS_error / df_error if df_error > 0 else np.nan

    base = anova_effect_ss_fixed(long, factors, levels_by_factor)
    SS_eff = base["SS_effects"]
    df_eff = base["df_effects"]

    def pretty_interaction(subset):
        if len(subset) == 1:
            return f"Фактор {subset[0]}"
        return "Фактор " + "×".join(subset)

    table = []
    table.append(("Блоки", float(SS_block), float(df_block),
                  float(MS_block) if df_block > 0 else np.nan, np.nan, np.nan, None))

    for rS in range(1, len(factors) + 1):
        for S in combinations(factors, rS):
            name = pretty_interaction(S)
            SSv = float(SS_eff.get(S, np.nan))
            dfv = float(df_eff.get(S, np.nan))
            MSv = SSv / dfv if (dfv and not math.isnan(dfv) and dfv > 0) else np.nan
            Fv = MSv / MS_error if (not math.isnan(MS_error) and MS_error > 0 and not math.isnan(MSv)) else np.nan
            pv = 1 - f_dist.cdf(Fv, dfv, df_error) if (not math.isnan(Fv) and not math.isnan(dfv)) else np.nan
            table.append((name, SSv, dfv, MSv, Fv, pv, "sub"))

    table.append(("Залишок", float(SS_error), float(df_error),
                  float(MS_error), None, None, None))
    table.append(("Загальна", float(SS_total), float((r * tcount) - 1), None, None, None, None))

    return {
        "table": table,
        "MS_error": float(MS_error),
        "df_error": int(df_error),
        "SS_total": float(SS_total),
        "SS_error": float(SS_error),
        "grand_mean": grand,
        "cell_means": base["cell_means"],
    }


# -------------------------
# Split-plot model
# -------------------------
def splitplot_anova(long, factors, levels_by_factor, main_factor):
    """
    Split-plot with blocks = repeat columns (BLOCK), main_factor = one of factors.
    Error terms:
      - main_factor tested against MS(Block×main_factor)
      - all other effects tested against MS(subplot error)
    """
    if "BLOCK" not in long[0]:
        raise ValueError("Split-plot потребує поля BLOCK (повторності).")

    if main_factor not in factors:
        raise ValueError("Невірний головний фактор для split-plot.")

    blocks = first_seen_order([r.get("BLOCK") for r in long])
    blocks = [b for b in blocks if b is not None]
    r = len(blocks)
    if r < 2:
        raise ValueError("Split-plot потребує щонайменше 2 блоки (повторності).")

    a_levels = levels_by_factor[main_factor]
    a = len(a_levels)
    if a < 2:
        raise ValueError("Головний фактор має мати щонайменше 2 градації.")

    sub_factors = [f for f in factors if f != main_factor]
    t_sub = 1
    for f in sub_factors:
        t_sub *= max(1, len(levels_by_factor[f]))

    values = np.array([rec["value"] for rec in long], dtype=float)
    grand = float(np.nanmean(values))
    SS_total = float(np.nansum((values - grand) ** 2))

    block_vals = defaultdict(list)
    for rec in long:
        b = rec.get("BLOCK")
        v = rec.get("value", np.nan)
        if b is None or v is None or math.isnan(v):
            continue
        block_vals[b].append(float(v))
    block_mean = {b: float(np.mean(block_vals[b])) for b in block_vals if len(block_vals[b]) > 0}

    A_vals = defaultdict(list)
    for rec in long:
        av = rec.get(main_factor)
        v = rec.get("value", np.nan)
        if av is None or v is None or math.isnan(v):
            continue
        A_vals[av].append(float(v))
    A_mean = {lv: float(np.mean(A_vals[lv])) for lv in A_vals if len(A_vals[lv]) > 0}

    BA_vals = defaultdict(list)
    for rec in long:
        b = rec.get("BLOCK")
        av = rec.get(main_factor)
        v = rec.get("value", np.nan)
        if b is None or av is None or v is None or math.isnan(v):
            continue
        BA_vals[(b, av)].append(float(v))
    BA_mean = {k: float(np.mean(BA_vals[k])) for k in BA_vals if len(BA_vals[k]) > 0}

    SS_block = 0.0
    for b0 in blocks:
        if b0 in block_mean:
            SS_block += (block_mean[b0] - grand) ** 2
    SS_block *= (a * t_sub)

    SS_A = 0.0
    for lv in a_levels:
        if lv in A_mean:
            SS_A += (A_mean[lv] - grand) ** 2
    SS_A *= (r * t_sub)

    SS_BA = 0.0
    for b0 in blocks:
        for lv in a_levels:
            key = (b0, lv)
            if key not in BA_mean:
                continue
            bm = block_mean.get(b0, np.nan)
            am = A_mean.get(lv, np.nan)
            mij = BA_mean[key]
            if any(isinstance(x, float) and math.isnan(x) for x in [bm, am, mij]):
                continue
            SS_BA += (mij - bm - am + grand) ** 2
    SS_BA *= t_sub

    df_block = r - 1
    df_A = a - 1
    df_BA = df_block * df_A
    if df_BA <= 0:
        df_BA = 1

    MS_BA = SS_BA / df_BA if df_BA > 0 else np.nan

    base = anova_effect_ss_fixed(long, factors, levels_by_factor)
    SS_eff = base["SS_effects"]
    df_eff = base["df_effects"]

    sum_SS_treat_effects = 0.0
    for S, ss in SS_eff.items():
        if ss is None or (isinstance(ss, float) and math.isnan(ss)):
            continue
        sum_SS_treat_effects += float(ss)

    SS_other = sum_SS_treat_effects - float(SS_eff.get((main_factor,), 0.0))
    SS_Esub = SS_total - SS_block - SS_A - SS_BA - SS_other
    if SS_Esub < -1e-9:
        SS_Esub = max(0.0, SS_Esub)

    df_other = 0
    for S, d in df_eff.items():
        if S == (main_factor,):
            continue
        df_other += int(d)

    N = len([1 for rec in long if rec.get("value") is not None and not math.isnan(rec.get("value", np.nan))])
    df_total = N - 1
    df_Esub = df_total - df_block - df_A - df_BA - df_other
    if df_Esub <= 0:
        df_Esub = 1

    MS_Esub = SS_Esub / df_Esub if df_Esub > 0 else np.nan

    def pretty_interaction(subset):
        if len(subset) == 1:
            return f"Фактор {subset[0]}"
        return "Фактор " + "×".join(subset)

    table = []
    table.append(("Блоки", float(SS_block), float(df_block),
                  float(SS_block / df_block) if df_block > 0 else np.nan, np.nan, np.nan, None))

    MS_A = SS_A / df_A if df_A > 0 else np.nan
    F_A = MS_A / MS_BA if (not math.isnan(MS_A) and not math.isnan(MS_BA) and MS_BA > 0) else np.nan
    p_A = 1 - f_dist.cdf(F_A, df_A, df_BA) if (not math.isnan(F_A)) else np.nan
    table.append((pretty_interaction((main_factor,)), float(SS_A), float(df_A), float(MS_A), float(F_A), float(p_A), "main"))

    table.append(("Залишок (main-plot)", float(SS_BA), float(df_BA), float(MS_BA), None, None, None))

    for rS in range(1, len(factors) + 1):
        for S in combinations(factors, rS):
            if S == (main_factor,):
                continue
            name = pretty_interaction(S)
            SSv = float(SS_eff.get(S, np.nan))
            dfv = float(df_eff.get(S, np.nan))
            MSv = SSv / dfv if (dfv and not math.isnan(dfv) and dfv > 0) else np.nan
            Fv = MSv / MS_Esub if (not math.isnan(MS_Esub) and MS_Esub > 0 and not math.isnan(MSv)) else np.nan
            pv = 1 - f_dist.cdf(Fv, dfv, df_Esub) if (not math.isnan(Fv) and not math.isnan(dfv)) else np.nan
            table.append((name, SSv, dfv, MSv, Fv, pv, "sub"))

    table.append(("Залишок (sub-plot)", float(SS_Esub), float(df_Esub), float(MS_Esub), None, None, None))
    table.append(("Загальна", float(SS_total), float(df_total), None, None, None, None))

    return {
        "table": table,
        "MS_error_main": float(MS_BA),
        "df_error_main": int(df_BA),
        "SS_error_main": float(SS_BA),
        "MS_error_sub": float(MS_Esub),
        "df_error_sub": int(df_Esub),
        "SS_error_sub": float(SS_Esub),
        "SS_total": float(SS_total),
        "grand_mean": grand,
        "cell_means": base["cell_means"],
        "main_factor": main_factor,
    }


# -------------------------
# Clipboard: copy PNG image in Windows via CF_DIB
# -------------------------
def copy_png_to_clipboard_windows(png_path: str):
    """
    Копіює PNG у буфер Windows так, щоб Word/PowerPoint вставляли як зображення.
    Робимо CF_DIB (BMP без file header).
    """
    if os.name != "nt":
        raise RuntimeError("Копіювання графіка у буфер підтримується лише у Windows.")

    import io
    import ctypes
    from ctypes import wintypes

    CF_DIB = 8
    GMEM_MOVEABLE = 0x0002

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    img = Image.open(png_path).convert("RGB")
    bio = io.BytesIO()
    img.save(bio, "BMP")
    bmp = bio.getvalue()
    dib = bmp[14:]  # strip BITMAPFILEHEADER

    if not user32.OpenClipboard(None):
        raise RuntimeError("Не вдалося відкрити буфер обміну.")

    try:
        user32.EmptyClipboard()

        hglob = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(dib))
        if not hglob:
            raise RuntimeError("GlobalAlloc не спрацював.")

        ptr = kernel32.GlobalLock(hglob)
        if not ptr:
            kernel32.GlobalFree(hglob)
            raise RuntimeError("GlobalLock не спрацював.")

        try:
            ctypes.memmove(ptr, dib, len(dib))
        finally:
            kernel32.GlobalUnlock(hglob)

        if not user32.SetClipboardData(CF_DIB, hglob):
            kernel32.GlobalFree(hglob)
            raise RuntimeError("SetClipboardData не спрацював.")
    finally:
        user32.CloseClipboard()


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
        self.plot_win = None

    # (далі без змін у цій частині: ask_indicator_units_design, choose_method_window,
    #  open_table, show_about, bind_cell, add_row/delete_row, add_column/delete_column,
    #  paste/collect_long, show_plot_window, build_factor_block_boxplot_png, show_report_segments)
    # -------------------------
    # Table window / paste / collect_long
    # -------------------------
    def ask_indicator_units_design(self):
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

        tk.Label(frm, text="Дизайн експерименту:", fg="#000000").grid(row=2, column=0, sticky="w", pady=6)

        design_var = tk.StringVar(value="CRD")
        cb = ttk.Combobox(frm, textvariable=design_var, state="readonly", width=18,
                          values=["CRD", "RCBD", "Split-plot"])
        cb.grid(row=2, column=1, sticky="w", pady=6)

        # Split-plot extra: main factor
        split_frame = tk.Frame(frm)
        split_frame.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        split_frame.grid_remove()

        tk.Label(split_frame, text="Головний фактор (main-plot):", fg="#000000").grid(row=0, column=0, sticky="w", padx=(0, 10))
        main_var = tk.StringVar(value="")
        cb_main = ttk.Combobox(split_frame, textvariable=main_var, state="readonly", width=10,
                               values=self.factor_keys if hasattr(self, "factor_keys") else ["A"])
        cb_main.grid(row=0, column=1, sticky="w")

        def refresh_split_ui(*_):
            if design_var.get() == "Split-plot":
                vals = self.factor_keys if hasattr(self, "factor_keys") else ["A"]
                cb_main.configure(values=vals)
                if main_var.get() not in vals:
                    main_var.set(vals[0] if vals else "")
                split_frame.grid()
            else:
                split_frame.grid_remove()

            dlg.update_idletasks()
            dlg.geometry("")
            center_window(dlg)

        cb.bind("<<ComboboxSelected>>", refresh_split_ui)

        out = {"ok": False, "indicator": "", "units": "", "design": "CRD", "main_factor": ""}

        def on_ok():
            out["indicator"] = e_ind.get().strip()
            out["units"] = e_units.get().strip()
            out["design"] = design_var.get().strip()

            if not out["indicator"] or not out["units"]:
                messagebox.showwarning("Помилка", "Заповніть назву показника та одиниці виміру.")
                return

            if out["design"] == "Split-plot":
                mf = main_var.get().strip()
                if not mf:
                    messagebox.showwarning("Помилка", "Для Split-plot потрібно обрати головний фактор.")
                    return
                out["main_factor"] = mf

            out["ok"] = True
            dlg.destroy()

        btns = tk.Frame(frm)
        btns.grid(row=4, column=0, columnspan=2, pady=(14, 0))
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
        messagebox.showinfo(
            "Розробник",
            "S.A.D. — Статистичний аналіз даних\n"
            "Версія: 1.1\n"
            "Розробик: Чаплоуцький Андрій Миколайович\n"
            "Уманський національний університет"
        )

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

    def collect_long(self):
        """
        long records:
          value, A,B,C,D, BLOCK (для RCBD/Split-plot)
        BLOCK = назва повторності (Повт.1, Повт.2, ...) відповідно до стовпчика
        """
        long = []
        rep_cols = self.used_repeat_columns()
        if not rep_cols:
            return long, rep_cols

        rep_start = self.factors_count
        rep_labels = {}
        for idx, c in enumerate(range(rep_start, self.cols)):
            rep_labels[c] = f"Повт.{idx+1}"

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

                rec = {"value": val, "BLOCK": rep_labels.get(c, f"Повт.?")}
                if self.factors_count >= 1: rec["A"] = levels[0]
                if self.factors_count >= 2: rec["B"] = levels[1]
                if self.factors_count >= 3: rec["C"] = levels[2]
                if self.factors_count >= 4: rec["D"] = levels[3]
                long.append(rec)

        return long, rep_cols

    # -------------------------
    # Graph window
    # -------------------------
    def show_plot_window(self, plot_png_path: str, factor_plot_meta: dict):
        if self.plot_win and tk.Toplevel.winfo_exists(self.plot_win):
            self.plot_win.destroy()

        self.plot_win = tk.Toplevel(self.root)
        self.plot_win.title("Графічний звіт")
        self.plot_win.geometry("1180x520")
        set_window_icon(self.plot_win)

        top = tk.Frame(self.plot_win, padx=8, pady=8)
        top.pack(fill=tk.X)

        fig = Figure(figsize=(10.8, 3.4), dpi=100)
        ax = fig.add_subplot(111)
        ax.axis("off")
        img = Image.open(plot_png_path)
        ax.imshow(img)

        canvas = FigureCanvasTkAgg(fig, master=self.plot_win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        def save_png():
            fn = filedialog.asksaveasfilename(
                parent=self.plot_win,
                defaultextension=".png",
                filetypes=[("PNG", "*.png")],
                title="Зберегти графік"
            )
            if not fn:
                return
            try:
                img2 = Image.open(plot_png_path)
                img2.save(fn, "PNG")
                messagebox.showinfo("Готово", "Графік збережено.")
            except Exception as ex:
                messagebox.showerror("Помилка", str(ex))

        def copy_png():
            try:
                copy_png_to_clipboard_windows(plot_png_path)
                messagebox.showinfo("Готово", "Графік скопійовано у буфер обміну.\nТепер вставте його у Word/PowerPoint (Ctrl+V).")
            except Exception as ex:
                messagebox.showerror("Помилка", str(ex))

        tk.Button(top, text="Зберегти графік (PNG)", command=save_png).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Копіювати графік", command=copy_png).pack(side=tk.LEFT, padx=4)

    def build_factor_block_boxplot_png(self, long, factor_keys, levels_by_factor, letters_factor,
                                       indicator, units, design, split_meta=None):
        """
        ONE-axis boxplot with blocks per factor.
        Letters placed above upper whisker.
        Legend: ▲ mean, ● outliers
        """
        x_positions = []
        data_list = []
        labels = []
        factor_block_ranges = []

        pos = 1
        gap = 1.2
        for f in factor_keys:
            lvls = levels_by_factor[f]
            start = pos
            for lv in lvls:
                arr = [r["value"] for r in long if r.get(f) == lv and r.get("value") is not None and not math.isnan(r.get("value", np.nan))]
                if len(arr) == 0:
                    arr = [np.nan]
                x_positions.append(pos)
                data_list.append(arr)
                labels.append(str(lv))
                pos += 1
            end = pos - 1
            factor_block_ranges.append((start, end, f))
            pos += gap

        fig = Figure(figsize=(12.5, 3.6), dpi=130)
        ax = fig.add_subplot(111)

        bp = ax.boxplot(
            data_list,
            positions=x_positions,
            widths=0.6,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker="^", markersize=6),
            flierprops=dict(marker="o", markersize=3),
        )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=0, fontsize=9)
        ax.set_ylabel(f"{indicator}, {units}")
        ax.set_title("")

        y_min, y_max = ax.get_ylim()
        y_text = y_min - (y_max - y_min) * 0.12
        ax.set_ylim(y_text, y_max)

        for (start, end, f) in factor_block_ranges:
            mid = (start + end) / 2.0
            ax.text(mid, y_min - (y_max - y_min) * 0.08, f"Фактор {f}", ha="center", va="top", fontsize=10)
            ax.axvline(end + 0.5, linewidth=0.6)

        whiskers = bp["whiskers"]

        fl_list = []
        for f in factor_keys:
            for lv in levels_by_factor[f]:
                fl_list.append((f, lv))

        for i, x in enumerate(x_positions):
            if i >= len(fl_list):
                continue
            f, lv = fl_list[i]
            letter = letters_factor.get(f, {}).get(lv, "")
            if not letter:
                continue
            try:
                wy = max(whiskers[2*i+1].get_ydata())
            except Exception:
                wy = np.nanmax(data_list[i])
            if wy is None or (isinstance(wy, float) and math.isnan(wy)):
                continue
            ax.text(x, wy + (y_max - y_min) * 0.03, letter, ha="center", va="bottom", fontsize=10)

        from matplotlib.lines import Line2D
        legend_items = [
            Line2D([0], [0], marker="^", linestyle="None", markersize=7, label="Середнє"),
            Line2D([0], [0], marker="o", linestyle="None", markersize=5, label="Викиди"),
        ]
        ax.legend(handles=legend_items, loc="upper right", frameon=True)

        out_dir = _script_dir()
        out_path = os.path.join(out_dir, "SAD_boxplot_factors.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        return out_path

    # -------------------------
    # Report window
    # -------------------------
    def show_report_segments(self, segments, plot_png_path=None, plot_meta=None):
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

        if plot_png_path:
            self.show_plot_window(plot_png_path, plot_meta or {})

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

    # =========================================================
    # ГОЛОВНІ ПРАВКИ — analyze():
    # 1) Літери по факторах для ВСІХ параметричних методів:
    #    LSD / Tukey / Duncan / Bonferroni (із CLD)
    # 2) Порядок текстових блоків у звіті як у вашому прикладі
    # =========================================================
    def analyze(self):
        created_at = datetime.now()

        params = self.ask_indicator_units_design()
        if not params["ok"]:
            return

        indicator = params["indicator"]
        units = params["units"]
        design = params["design"]
        main_factor = params.get("main_factor", "")

        long, used_rep_cols = self.collect_long()
        if len(long) == 0:
            messagebox.showwarning("Помилка", "Немає числових даних для аналізу.\nПеревірте повторності та значення.")
            return

        values = np.array([r["value"] for r in long], dtype=float)
        if len(values) < 3:
            messagebox.showinfo("Результат", "Надто мало даних для аналізу.")
            return

        levels_by_factor = {f: first_seen_order([r.get(f) for r in long]) for f in self.factor_keys}

        # Residuals for Shapiro: use cell means (no-block means)
        base = anova_effect_ss_fixed(long, self.factor_keys, levels_by_factor)
        cell_means = base.get("cell_means", {})
        residuals = []
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

        # Variant order for summary table
        variant_order = first_seen_order([tuple(r.get(f) for f in self.factor_keys) for r in long])
        v_names = [" | ".join(map(str, k)) for k in variant_order]
        num_variants = len(variant_order)

        vstats = variant_mean_sd(long, self.factor_keys)
        v_means = {k: vstats[k][0] for k in vstats.keys()}
        v_sds = {k: vstats[k][1] for k in vstats.keys()}
        v_ns = {k: vstats[k][2] for k in vstats.keys()}

        means1 = {v_names[i]: v_means.get(variant_order[i], np.nan) for i in range(len(variant_order))}
        ns1 = {v_names[i]: v_ns.get(variant_order[i], 0) for i in range(len(variant_order))}
        groups1 = {v_names[i]: groups_by_keys(long, tuple(self.factor_keys)).get(variant_order[i], []) for i in range(len(variant_order))}

        # Factor means + ns (for letters)
        factor_groups = {f: {k[0]: v for k, v in groups_by_keys(long, (f,)).items()} for f in self.factor_keys}
        factor_means = {f: {lvl: float(np.mean(arr)) if len(arr) else np.nan for lvl, arr in factor_groups[f].items()} for f in self.factor_keys}
        factor_ns = {f: {lvl: len(arr) for lvl, arr in factor_groups[f].items()} for f in self.factor_keys}

        # Choose model by design
        split_meta = None
        if design == "CRD":
            SS_eff = base["SS_effects"]
            df_eff = base["df_effects"]
            SS_total = base["SS_total"]

            total_cells = 1
            for f in self.factor_keys:
                total_cells *= len(levels_by_factor[f])
            N = len(values)
            df_total = N - 1
            df_error = max(1, N - total_cells)

            SS_error = base["SS_error_crd"]
            MS_error = SS_error / df_error if df_error > 0 else np.nan

            table = []
            def pretty_interaction(subset):
                if len(subset) == 1:
                    return f"Фактор {subset[0]}"
                return "Фактор " + "×".join(subset)

            for rS in range(1, len(self.factor_keys) + 1):
                for S in combinations(self.factor_keys, rS):
                    name = pretty_interaction(S)
                    SSv = float(SS_eff.get(S, np.nan))
                    dfv = float(df_eff.get(S, np.nan))
                    MSv = SSv / dfv if (dfv and dfv > 0) else np.nan
                    Fv = MSv / MS_error if (MS_error and MS_error > 0 and not math.isnan(MSv)) else np.nan
                    pv = 1 - f_dist.cdf(Fv, dfv, df_error) if (not math.isnan(Fv)) else np.nan
                    table.append((name, SSv, dfv, MSv, Fv, pv, "sub"))

            table.append(("Залишок", float(SS_error), float(df_error), float(MS_error), None, None, None))
            table.append(("Загальна", float(SS_total), float(df_total), None, None, None, None))

            model = {
                "table": table,
                "MS_error": float(MS_error),
                "df_error": int(df_error),
                "SS_total": float(SS_total),
                "SS_error": float(SS_error),
                "grand_mean": float(base["grand_mean"]),
            }

            ms_for_factor_letters = {f: (model["MS_error"], model["df_error"]) for f in self.factor_keys}

        elif design == "RCBD":
            model = rcbd_anova(long, self.factor_keys, levels_by_factor)
            ms_for_factor_letters = {f: (model["MS_error"], model["df_error"]) for f in self.factor_keys}

        else:
            model = splitplot_anova(long, self.factor_keys, levels_by_factor, main_factor=main_factor)
            split_meta = {
                "main_factor": model["main_factor"],
                "df_error_main": model["df_error_main"],
                "df_error_sub": model["df_error_sub"],
            }
            ms_for_factor_letters = {}
            for f in self.factor_keys:
                if f == model["main_factor"]:
                    ms_for_factor_letters[f] = (model["MS_error_main"], model["df_error_main"])
                else:
                    ms_for_factor_letters[f] = (model["MS_error_sub"], model["df_error_sub"])

        nonparam = method in ("mw", "kw")

        # Homogeneity test (Brown–Forsythe) for parametric methods
        bf_F, bf_p = (np.nan, np.nan)
        if not nonparam and method in ("lsd", "tukey", "duncan", "bonferroni"):
            bf_F, bf_p = brown_forsythe_from_groups(groups1)

        # =========================
        # ЛІТЕРИ ПО ФАКТОРАХ (NEW):
        # =========================
        letters_factor = {f: {lvl: "" for lvl in levels_by_factor[f]} for f in self.factor_keys}

        if (not nonparam) and (method in ("lsd", "tukey", "duncan", "bonferroni")):
            for f in self.factor_keys:
                lvls = levels_by_factor[f]
                MS_e, df_e = ms_for_factor_letters.get(f, (np.nan, np.nan))

                # prepare per-factor means/ns dict keyed by level names (strings)
                means_f = {str(lvl): factor_means[f].get(lvl, np.nan) for lvl in lvls}
                ns_f = {str(lvl): factor_ns[f].get(lvl, 0) for lvl in lvls}
                lvls_str = [str(lvl) for lvl in lvls]

                sig = {}
                if method == "lsd":
                    sig = lsd_sig_matrix(lvls_str, means_f, ns_f, MS_e, df_e, alpha=ALPHA)

                elif method in ("tukey", "duncan"):
                    # reuse studentized_range-based engine:
                    _rows_tmp, sig = pairwise_param_short_variants_pm(
                        lvls_str, means_f, ns_f, MS_e, df_e, method, alpha=ALPHA
                    )

                elif method == "bonferroni":
                    # strict Bonferroni t-tests using same engine:
                    _rows_tmp, sig = pairwise_param_short_variants_pm(
                        lvls_str, means_f, ns_f, MS_e, df_e, "bonferroni", alpha=ALPHA
                    )

                letters_map = cld_multi_letters(lvls_str, means_f, sig)
                # write back using original level objects:
                for lvl in lvls:
                    letters_factor[f][lvl] = letters_map.get(str(lvl), "")

        # =========================
        # Letters for variants (existing + unchanged logic)
        # =========================
        letters_named = {name: "" for name in v_names}
        pairwise_rows = []
        do_posthoc = True

        kw_H, kw_p, kw_df, kw_eps2 = (np.nan, np.nan, np.nan, np.nan)
        if method == "kw":
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

        if not nonparam:
            if design == "Split-plot":
                MS_e_var = model["MS_error_sub"]
                df_e_var = model["df_error_sub"]
            else:
                MS_e_var = model["MS_error"]
                df_e_var = model["df_error"]

            if method == "lsd":
                sigv = lsd_sig_matrix(v_names, means1, ns1, MS_e_var, df_e_var, alpha=ALPHA)
                letters_named = cld_multi_letters(v_names, means1, sigv)
            elif method in ("tukey", "duncan", "bonferroni"):
                pairwise_rows, sig = pairwise_param_short_variants_pm(v_names, means1, ns1, MS_e_var, df_e_var, method, alpha=ALPHA)
                letters_named = cld_multi_letters(v_names, means1, sig)
        else:
            if method == "kw":
                if not (isinstance(kw_p, float) and math.isnan(kw_p)) and kw_p >= ALPHA:
                    do_posthoc = False
                if do_posthoc:
                    pairwise_rows, sig = pairwise_mw_bonf_with_effect(v_names, groups1, alpha=ALPHA)
                    med_tmp = {name: float(np.median(groups1[name])) if len(groups1[name]) else np.nan for name in v_names}
                    letters_named = cld_multi_letters(v_names, med_tmp, sig)
                else:
                    letters_named = {name: "" for name in v_names}
            elif method == "mw":
                pairwise_rows, sig = pairwise_mw_bonf_with_effect(v_names, groups1, alpha=ALPHA)
                med_tmp = {name: float(np.median(groups1[name])) if len(groups1[name]) else np.nan for name in v_names}
                letters_named = cld_multi_letters(v_names, med_tmp, sig)

        letters_variants = {variant_order[i]: letters_named.get(v_names[i], "") for i in range(len(variant_order))}

        # =========================
        # REPORT segments — порядок як у вашому прикладі
        # =========================
        seg = []
        seg.append(("text", "З В І Т   С Т А Т И С Т И Ч Н О Г О   А Н А Л І З У   Д А Н И Х\n\n"))
        seg.append(("text", f"Показник:\t{indicator}\nОдиниці виміру:\t{units}\n"))
        seg.append(("text", f"Дизайн експерименту:\t{design}\n"))
        if design == "Split-plot":
            seg.append(("text", f"Split-plot: головний фактор (main-plot):\t{main_factor}\n"))
        seg.append(("text", "\n"))

        # 1) Кількості
        seg.append(("text",
                    f"Кількість варіантів:\t{num_variants}\n"
                    f"Кількість повторностей:\t{len(used_rep_cols)}\n"
                    f"Загальна кількість облікових значень:\t{len(long)}\n"))

        # 2) Виконуваний аналіз
        method_label = {
            "lsd": "Параметричний аналіз: Brown–Forsythe + ANOVA + НІР₀₅ (LSD).",
            "tukey": "Параметричний аналіз: Brown–Forsythe + ANOVA + тест Тьюкі (Tukey HSD).",
            "duncan": "Параметричний аналіз: Brown–Forsythe + ANOVA + тест Дункана.",
            "bonferroni": "Параметричний аналіз: Brown–Forsythe + ANOVA + корекція Бонферроні.",
            "kw": "Непараметричний аналіз: Kruskal–Wallis.",
            "mw": "Непараметричний аналіз: Mann–Whitney.",
        }.get(method, "")
        if method_label:
            seg.append(("text", f"Виконуваний статистичний аналіз:\t{method_label}\n\n"))

        # 3) Пояснення
        seg.append(("text", "Пояснення позначень істотності: ** — p<0.01; * — p<0.05.\n"))
        seg.append(("text", "У таблицях знак \"-\" свідчить що p ≥ 0.05.\n"))
        seg.append(("text", "Істотна різниця (літери): різні літери свідчать про наявність істотної різниці.\n\n"))

        # 4) Нормальність
        if not math.isnan(W):
            seg.append(("text",
                        f"Перевірка нормальності залишків (Shapiro–Wilk):\t"
                        f"{normality_text(p_norm)}\t(W={fmt_num(float(W),4)}; p={fmt_num(float(p_norm),4)})\n\n"))
        else:
            seg.append(("text", "Перевірка нормальності залишків (Shapiro–Wilk):\tн/д\n\n"))

        # 5) Однорідність (як у прикладі: з крапкою в кінці фрази)
        if not nonparam:
            if not any(math.isnan(x) for x in [bf_F, bf_p]):
                bf_concl = "умова виконується" if bf_p >= ALPHA else f"умова порушена {significance_mark(bf_p)}"
                seg.append(("text",
                            f"Перевірка однорідності дисперсій (Brown–Forsythe):\t"
                            f"F={fmt_num(bf_F,4)}; p={fmt_num(bf_p,4)}\t{bf_concl}.\n\n"))
            else:
                seg.append(("text", "Перевірка однорідності дисперсій (Brown–Forsythe):\tн/д\n\n"))

        # Nonparam global test line
        if method == "kw":
            if not (isinstance(kw_p, float) and math.isnan(kw_p)):
                concl = "істотна різниця " + significance_mark(kw_p) if kw_p < ALPHA else "-"
                seg.append(("text",
                            f"Глобальний тест між варіантами (Kruskal–Wallis):\t"
                            f"H={fmt_num(kw_H,4)}; df={int(kw_df)}; p={fmt_num(kw_p,4)}\t{concl}\n"))
                seg.append(("text", f"Розмір ефекту (ε²):\t{fmt_num(kw_eps2,4)}\n\n"))

        # =========================
        # PARAMETRIC report
        # =========================
        if not nonparam:
            anova_rows = []
            for name, SSv, dfv, MSv, Fv, pv, denom in model["table"]:
                df_txt = str(int(dfv)) if dfv is not None and not math.isnan(dfv) else ""
                if name.startswith("Залишок") or name in ("Загальна", "Блоки"):
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

            eff_rows = build_effect_strength_rows(model["table"])
            seg.append(("text", "ТАБЛИЦЯ 2. Сила впливу (% від SS)\n"))
            seg.append(("table", {"headers": ["Джерело", "%"], "rows": eff_rows}))
            seg.append(("text", "\n"))

            if design == "Split-plot":
                pe2_rows = build_partial_eta2_rows_with_label_splitaware(
                    model["table"], model["SS_error_main"], model["SS_error_sub"]
                )
            else:
                SS_error = model["SS_error"]
                pe2_rows = []
                for name, SSv, dfv, MSv, Fv, pv, denom in model["table"]:
                    if name in ("Залишок", "Загальна", "Блоки"):
                        continue
                    if SSv is None or (isinstance(SSv, float) and math.isnan(SSv)):
                        continue
                    denom2 = SSv + SS_error
                    pe2 = (SSv / denom2) if denom2 > 0 else np.nan
                    pe2_rows.append([name, fmt_num(pe2, 4), partial_eta2_label(pe2)])

            seg.append(("text", "ТАБЛИЦЯ 3. Розмір ефекту (partial η²)\n"))
            seg.append(("table", {"headers": ["Джерело", "partial η²", "Висновок"], "rows": pe2_rows}))
            seg.append(("text", "\n"))

            gmean = model.get("grand_mean", float(np.mean(values)))
            cv_rows = []
            if design == "Split-plot":
                cv_main = cv_from_ms(model["MS_error_main"], gmean)
                cv_sub = cv_from_ms(model["MS_error_sub"], gmean)
                cv_total = cv_percent_from_values(values)
                cv_rows.append(["Main-plot (за MS Block×A)", fmt_num(cv_main, 2)])
                cv_rows.append(["Sub-plot (за MS залишку)", fmt_num(cv_sub, 2)])
                cv_rows.append(["Загальний (за даними)", fmt_num(cv_total, 2)])
            else:
                cv_model = cv_from_ms(model["MS_error"], gmean)
                cv_total = cv_percent_from_values(values)
                cv_rows.append(["Модельний (за MS_error)", fmt_num(cv_model, 2)])
                cv_rows.append(["Загальний (за даними)", fmt_num(cv_total, 2)])

            seg.append(("text", "ТАБЛИЦЯ 4. Коефіцієнт варіації (CV, %)\n"))
            seg.append(("table", {"headers": ["Елемент", "CV, %"], "rows": cv_rows}))
            seg.append(("text", "\n"))

            # Means per factor with letters (NOW works for ALL parametric)
            tno = 5
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
            tno += 1

            if method in ("tukey", "duncan", "bonferroni") and pairwise_rows:
                seg.append(("text", f"ТАБЛИЦЯ {tno}. Парні порівняння варіантів\n"))
                seg.append(("table", {"headers": ["Комбінація варіантів", "p", "Істотна різниця"], "rows": pairwise_rows}))
                seg.append(("text", "\n"))

        # =========================
        # NONPARAMETRIC report
        # =========================
        else:
            seg.append(("text", "Непараметричний звіт подається у спрощеному вигляді.\n"))
            if method == "kw" and not do_posthoc:
                seg.append(("text", "Пост-хок порівняння не виконувалися, оскільки Kruskal–Wallis не виявив істотної різниці (p ≥ 0.05).\n\n"))
            else:
                if pairwise_rows:
                    seg.append(("text", "ТАБЛИЦЯ. Парні порівняння + ефект (Cliff’s δ)\n"))
                    seg.append(("table", {"headers": ["Комбінація", "U", "p (Bonf.)", "Істотна", "δ", "Висновок"], "rows": pairwise_rows}))
                    seg.append(("text", "\n"))

        seg.append(("text", f"Звіт сформовано:\t{created_at.strftime('%d.%m.%Y, %H:%M')}\n"))

        plot_png = self.build_factor_block_boxplot_png(
            long=long,
            factor_keys=self.factor_keys,
            levels_by_factor=levels_by_factor,
            letters_factor=letters_factor,
            indicator=indicator,
            units=units,
            design=design,
            split_meta=split_meta
        )

        self.show_report_segments(seg, plot_png_path=plot_png, plot_meta={"design": design, "split": split_meta})


# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    root = tk.Tk()
    set_window_icon(root)
    app = SADTk(root)
    root.mainloop()
