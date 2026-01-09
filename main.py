# main.py
# -*- coding: utf-8 -*-

"""
S.A.D. — Статистичний аналіз даних (Tkinter)

Потрібно: Python 3.8+, numpy, scipy
Встановлення: pip install numpy scipy

ЗМІНИ (лише за твоїм завданням):
✅ Додано вибір дизайну експерименту у вікні «Параметри звіту»: CRD / RCBD.
✅ Для RCBD:
   - Нормальність (Shapiro–Wilk) рахується по залишках МОДЕЛІ RCBD (із вилученням ефекту блоку).
   - Параметричні аналізи (LSD/Tukey/Duncan/Bonferroni) враховують блоки:
       SS_error = SS_total - SS_treat - SS_block; df_error = df_total - df_treat - df_block.
   - Непараметричні: замість KW/MW використовуються Friedman (>=3 варіанти) або Wilcoxon (2 варіанти),
     з відповідними звітами (mean ranks по блоках, Kendall’s W, парні Wilcoxon+Bonferroni при потребі).
✅ KW/MW залишені лише для CRD.
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

# ✅ нове (для RCBD непараметричних)
from scipy.stats import friedmanchisquare, wilcoxon

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
    # 1) PyInstaller temp dir
    try:
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            return sys._MEIPASS
    except Exception:
        pass
    # 2) normal script dir
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
    """
    Шукаємо icon.ico у:
    - папці скрипта (корінь програми)
    - поточній папці запуску (cwd)
    - папці sys.argv[0]
    - папці sys.executable (для зібраних exe)
    - _MEIPASS (PyInstaller)
    """
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
# ANOVA 1–4 (CRD base)
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
# LSD + pairwise (parametric)
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
# RCBD nonparam helpers
# -------------------------
def build_rcbd_block_map(long, factor_keys, block_key="_block"):
    """
    Повертає:
      - variant_order: список кортежів рівнів факторів (варіанти)
      - v_names: строкові назви варіантів
      - blocks_order: порядок блоків
      - block_values: dict[block][v_name] = value (якщо дублікати — середнє)
      - complete_blocks: список блоків, де є ВСІ варіанти
    """
    variant_order = first_seen_order([tuple(r.get(f) for f in factor_keys) for r in long])
    v_names = [" | ".join(map(str, k)) for k in variant_order]

    blocks_order = first_seen_order([r.get(block_key) for r in long if r.get(block_key) is not None])

    # збір значень у блоках
    tmp = defaultdict(lambda: defaultdict(list))
    for r in long:
        b = r.get(block_key, None)
        if b is None:
            continue
        vk = tuple(r.get(f) for f in factor_keys)
        vname = " | ".join(map(str, vk))
        val = r.get("value", np.nan)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        tmp[b][vname].append(float(val))

    block_values = {}
    for b in blocks_order:
        block_values[b] = {}
        for vn, arr in tmp[b].items():
            if len(arr) == 1:
                block_values[b][vn] = arr[0]
            else:
                block_values[b][vn] = float(np.mean(arr))

    complete_blocks = []
    for b in blocks_order:
        ok = True
        for vn in v_names:
            if vn not in block_values[b]:
                ok = False
                break
        if ok:
            complete_blocks.append(b)

    return variant_order, v_names, blocks_order, block_values, complete_blocks


def friedman_mean_ranks(v_names, complete_blocks, block_values):
    """
    Ранги рахуються ВСЕРЕДИНІ кожного блоку, потім усереднюються.
    """
    if not complete_blocks:
        return {}
    sums = {vn: 0.0 for vn in v_names}
    nb = 0
    for b in complete_blocks:
        vals = [block_values[b][vn] for vn in v_names]
        ranks = rankdata(vals, method="average")  # 1..k (менше значення -> менший ранг)
        for vn, rk in zip(v_names, ranks):
            sums[vn] += float(rk)
        nb += 1
    return {vn: (sums[vn] / nb if nb else np.nan) for vn in v_names}


def wilcoxon_rank_biserial(x, y):
    """
    Rank-biserial correlation для Wilcoxon signed-rank.
    Обчислюємо W+ і W- по рангах |d|, d=x-y.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = (~np.isnan(x)) & (~np.isnan(y))
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return np.nan

    d = x - y
    # відкидаємо нульові різниці (як робить wilcoxon за замовчуванням)
    nz = d != 0
    d = d[nz]
    if len(d) == 0:
        return np.nan

    absd = np.abs(d)
    r = rankdata(absd, method="average")
    w_pos = float(np.sum(r[d > 0]))
    w_neg = float(np.sum(r[d < 0]))
    denom = w_pos + w_neg
    if denom <= 0:
        return np.nan
    return float((w_pos - w_neg) / denom)


def rbc_label(abs_rbc: float) -> str:
    if abs_rbc is None or (isinstance(abs_rbc, float) and math.isnan(abs_rbc)):
        return ""
    if abs_rbc < 0.10:
        return "дуже слабкий"
    if abs_rbc < 0.30:
        return "слабкий"
    if abs_rbc < 0.50:
        return "середній"
    return "сильний"


def pairwise_wilcoxon_bonf(v_names, complete_blocks, block_values, alpha=0.05):
    pairs = [(v_names[i], v_names[j]) for i in range(len(v_names)) for j in range(i + 1, len(v_names))]
    mtests = len(pairs) if pairs else 1
    rows = []
    sig = {}
    for a, b in pairs:
        xa = [block_values[bl][a] for bl in complete_blocks]
        xb = [block_values[bl][b] for bl in complete_blocks]
        if len(xa) < 2:
            continue
        try:
            stat, p = wilcoxon(xa, xb, alternative="two-sided")
        except Exception:
            continue
        p = float(p)
        stat = float(stat)
        p_adj = min(1.0, p * mtests)
        decision = (p_adj < alpha)
        sig[(a, b)] = decision

        rbc = wilcoxon_rank_biserial(np.array(xa), np.array(xb))
        lab = rbc_label(abs(rbc)) if not (isinstance(rbc, float) and math.isnan(rbc)) else ""
        rows.append([f"{a}  vs  {b}", fmt_num(stat, 2), f"{p_adj:.4f}", "+" if decision else "-", fmt_num(rbc, 3), lab])
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
    for name, SSv, dfv, MSv, Fv, pv in anova_table_rows:
        if name == "Залишок":
            SS_error = SSv
            break

    if SS_error is None or (isinstance(SS_error, float) and math.isnan(SS_error)) or SS_error <= 0:
        SS_error = np.nan

    rows = []
    for name, SSv, dfv, MSv, Fv, pv in anova_table_rows:
        if name in ("Залишок", "Загальна"):
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
# Ranks (CRD / general)
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
    for k, r in zip(keys, ranks):
        sums[k] += float(r)
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

    # ✅ додано вибір дизайну
    def ask_indicator_units(self):
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

        # ✅ дизайн експерименту
        tk.Label(frm, text="Дизайн експерименту:", fg="#000000").grid(row=2, column=0, sticky="w", pady=(10, 6))
        design_var = tk.StringVar(value="crd")
        box = tk.Frame(frm)
        box.grid(row=2, column=1, sticky="w", pady=(10, 6))
        tk.Radiobutton(box, text="Повна рандомізація (CRD)", variable=design_var, value="crd").pack(anchor="w")
        tk.Radiobutton(box, text="Блочна рандомізація (RCBD)", variable=design_var, value="rcbd").pack(anchor="w")

        out = {"ok": False, "indicator": "", "units": "", "design": "crd"}

        def on_ok():
            out["indicator"] = e_ind.get().strip()
            out["units"] = e_units.get().strip()
            out["design"] = design_var.get().strip() or "crd"
            if not out["indicator"] or not out["units"]:
                messagebox.showwarning("Помилка", "Заповніть назву показника та одиниці виміру.")
                return
            out["ok"] = True
            dlg.destroy()

        btns = tk.Frame(frm)
        btns.grid(row=3, column=0, columnspan=2, pady=(12, 0))
        tk.Button(btns, text="OK", width=10, command=on_ok).pack(side=tk.LEFT, padx=6)
        tk.Button(btns, text="Скасувати", width=12, command=lambda: dlg.destroy()).pack(side=tk.LEFT, padx=6)

        dlg.update_idletasks()
        center_window(dlg)
        e_ind.focus_set()
        dlg.bind("<Return>", lambda e: on_ok())
        dlg.grab_set()
        self.root.wait_window(dlg)
        return out

    # ✅ змінено: залежить від дизайну + к-ті варіантів
    def choose_method_window(self, p_norm, design, num_variants):
        dlg = tk.Toplevel(self.root)
        dlg.title("Вибір виду аналізу")
        dlg.resizable(False, False)
        set_window_icon(dlg)

        frm = tk.Frame(dlg, padx=16, pady=14)
        frm.pack(fill=tk.BOTH, expand=True)

        normal = (p_norm is not None) and (not math.isnan(p_norm)) and (p_norm > 0.05)

        # PARAMETRIC (однаково для CRD/RCBD – відрізняються обчисленням MS_error/df_error)
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
            # NONPARAMETRIC: CRD -> KW/MW, RCBD -> Friedman/Wilcoxon
            if design == "rcbd":
                msg = ("Дані експерименту не відповідають принципам нормального розподілу\n"
                       "за методом Шапіра-Вілка.\n"
                       "Для блочного дизайну застосовуються спеціальні непараметричні тести.")
                tk.Label(frm, text=msg, fg="#c62828", justify="left").pack(anchor="w", pady=(0, 10))
                if num_variants <= 1:
                    options = []
                elif num_variants == 2:
                    options = [("Wilcoxon (парний)", "wilcoxon")]
                else:
                    options = [("Friedman", "friedman")]
            else:
                msg = ("Дані експерименту не відповідають принципам нормального розподілу\n"
                       "за методом Шапіра-Вілка.\n"
                       "Виберіть один з непараметричних типів аналізу.")
                tk.Label(frm, text=msg, fg="#c62828", justify="left").pack(anchor="w", pady=(0, 10))
                options = [
                    ("Краскела–Уолліса", "kw"),
                    ("Манна-Уітні", "mw"),
                ]

        if not options:
            messagebox.showwarning("Помилка", "Недостатньо даних/варіантів для вибору методу.")
            dlg.destroy()
            return {"ok": False, "method": None}

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

    # ✅ додано design: для RCBD у long додається "_block"
    def collect_long(self, design="crd"):
        long = []
        rep_cols = self.used_repeat_columns()
        if not rep_cols:
            return long, rep_cols

        # порядок блоків — за порядком використаних повторностей
        block_names = {}
        for idx, c in enumerate(rep_cols, start=1):
            block_names[c] = f"Блок {idx}"

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

                rec = {"value": val}
                if self.factors_count >= 1: rec["A"] = levels[0]
                if self.factors_count >= 2: rec["B"] = levels[1]
                if self.factors_count >= 3: rec["C"] = levels[2]
                if self.factors_count >= 4: rec["D"] = levels[3]

                if design == "rcbd":
                    rec["_block"] = block_names.get(c, "Блок")

                long.append(rec)

        return long, rep_cols

    def analyze(self):
        created_at = datetime.now()

        params = self.ask_indicator_units()
        if not params["ok"]:
            return

        indicator = params["indicator"]
        units = params["units"]
        design = params.get("design", "crd")  # "crd" або "rcbd"

        long, used_rep_cols = self.collect_long(design=design)
        if len(long) == 0:
            messagebox.showwarning("Помилка", "Немає числових даних для аналізу.\nПеревірте повторності та значення.")
            return

        values = np.array([r["value"] for r in long], dtype=float)
        if len(values) < 3:
            messagebox.showinfo("Результат", "Надто мало даних для аналізу.")
            return

        levels_by_factor = {f: first_seen_order([r.get(f) for r in long]) for f in self.factor_keys}

        # базова ANOVA по факторах (для SS факторів/комбінацій)
        try:
            res = anova_n_way(long, self.factor_keys, levels_by_factor)
        except Exception as ex:
            messagebox.showerror("Помилка аналізу", str(ex))
            return

        # Варіанти (комбінації факторів)
        variant_order = first_seen_order([tuple(r.get(f) for f in self.factor_keys) for r in long])
        v_names = [" | ".join(map(str, k)) for k in variant_order]
        num_variants = len(variant_order)

        # Групи по варіантах (для BF/параметричних постхок)
        groups1 = {v_names[i]: groups_by_keys(long, tuple(self.factor_keys)).get(variant_order[i], [])
                   for i in range(len(variant_order))}

        # -------------------------
        # Нормальність (Shapiro–Wilk)
        # CRD: залишки від cell_means (як було)
        # RCBD: залишки від моделі: y_hat = block_mean + treat_mean - grand
        # -------------------------
        try:
            if design == "rcbd":
                # treatment means by full combination
                treat_means = subset_stats(long, tuple(self.factor_keys))
                treat_mean = {k: v[0] for k, v in treat_means.items()}

                # blocks
                blocks = first_seen_order([r.get("_block") for r in long if r.get("_block") is not None])
                block_groups = groups_by_keys(long, ("_block",))
                block_mean = {b: float(np.mean(block_groups.get((b,), []))) for b in blocks}
                grand_mean = float(np.mean(values))

                residuals = []
                for rec in long:
                    b = rec.get("_block", None)
                    key = tuple(rec.get(f) for f in self.factor_keys)
                    y = rec.get("value", np.nan)
                    if b is None or key not in treat_mean or b not in block_mean:
                        continue
                    yhat = block_mean[b] + treat_mean[key] - grand_mean
                    residuals.append(float(y - yhat))
                residuals = np.array(residuals, dtype=float)
            else:
                cell_means = res.get("cell_means", {})
                residuals = []
                for rec in long:
                    key = tuple(rec.get(f) for f in self.factor_keys)
                    v = rec.get("value", np.nan)
                    m = cell_means.get(key, np.nan)
                    if not math.isnan(v) and not math.isnan(m):
                        residuals.append(v - m)
                residuals = np.array(residuals, dtype=float)

            W, p_norm = shapiro(residuals) if len(residuals) >= 3 else (np.nan, np.nan)
        except Exception:
            W, p_norm = (np.nan, np.nan)

        # вибір методу з урахуванням дизайну
        choice = self.choose_method_window(p_norm, design=design, num_variants=num_variants)
        if not choice["ok"]:
            return
        method = choice["method"]

        # -------------------------
        # Підготовка описових по факторах (як було)
        # -------------------------
        factor_groups = {f: {k[0]: v for k, v in groups_by_keys(long, (f,)).items()} for f in self.factor_keys}
        factor_means = {f: {lvl: float(np.mean(arr)) if len(arr) else np.nan for lvl, arr in factor_groups[f].items()}
                        for f in self.factor_keys}
        factor_ns = {f: {lvl: len(arr) for lvl, arr in factor_groups[f].items()} for f in self.factor_keys}

        factor_medians = {}
        factor_q = {}
        for f in self.factor_keys:
            factor_medians[f] = {}
            factor_q[f] = {}
            for lvl, arr in factor_groups[f].items():
                med, q1, q3 = median_q1_q3(arr)
                factor_medians[f][lvl] = med
                factor_q[f][lvl] = (q1, q3)

        vstats = variant_mean_sd(long, self.factor_keys)
        v_means = {k: vstats[k][0] for k in vstats.keys()}
        v_sds = {k: vstats[k][1] for k in vstats.keys()}
        v_ns = {k: vstats[k][2] for k in vstats.keys()}

        means1 = {v_names[i]: v_means.get(variant_order[i], np.nan) for i in range(len(variant_order))}
        ns1 = {v_names[i]: v_ns.get(variant_order[i], 0) for i in range(len(variant_order))}

        ranks_by_variant = mean_ranks_by_key(
            long, key_func=lambda rec: " | ".join(str(rec.get(f)) for f in self.factor_keys)
        )
        ranks_by_factor = {f: mean_ranks_by_key(long, key_func=lambda rec, ff=f: rec.get(ff)) for f in self.factor_keys}

        v_medians = {}
        v_q = {}
        for i, k in enumerate(variant_order):
            name = v_names[i]
            arr = groups1.get(name, [])
            med, q1, q3 = median_q1_q3(arr)
            v_medians[k] = med
            v_q[k] = (q1, q3)

        # -------------------------
        # PARAMETRIC: MS_error/df_error
        # CRD: як є
        # RCBD: SS_error = SS_total - SS_treat - SS_block; df_error = df_total - df_treat - df_block
        # -------------------------
        MS_error = res.get("MS_error", np.nan)
        df_error = res.get("df_error", np.nan)
        SS_total = res.get("SS_total", np.nan)
        SS_error = res.get("SS_error", np.nan)

        SS_block = np.nan
        df_block = np.nan

        if design == "rcbd":
            # розрахунок SS_block
            blocks = first_seen_order([r.get("_block") for r in long if r.get("_block") is not None])
            b = len(blocks)
            if b >= 2:
                grand = float(np.mean(values))
                block_groups = groups_by_keys(long, ("_block",))
                SS_block_val = 0.0
                n_total = 0
                for bl in blocks:
                    arr = block_groups.get((bl,), [])
                    if not arr:
                        continue
                    mb = float(np.mean(arr))
                    nb = len(arr)
                    n_total += nb
                    SS_block_val += nb * (mb - grand) ** 2
                SS_block = float(SS_block_val)
                df_block = int(b - 1)
            else:
                SS_block = np.nan
                df_block = np.nan

            # сума SS/df для всіх джерел treatment (усі фактори та їх взаємодії)
            SS_treat = 0.0
            df_treat = 0.0
            for name, SSv, dfv, MSv, Fv, pv in res["table"]:
                if name in ("Залишок", "Загальна"):
                    continue
                if SSv is None or (isinstance(SSv, float) and math.isnan(SSv)):
                    continue
                if dfv is None or (isinstance(dfv, float) and math.isnan(dfv)):
                    continue
                SS_treat += float(SSv)
                df_treat += float(dfv)

            df_total = int(len(values) - 1)
            if not any(math.isnan(x) for x in [SS_total, SS_block]) and not any(math.isnan(x) for x in [df_block]):
                SS_error_rcbd = float(SS_total - SS_treat - SS_block)
                df_error_rcbd = int(df_total - int(df_treat) - int(df_block))
                if df_error_rcbd <= 0:
                    df_error_rcbd = max(1, df_error_rcbd)
                if SS_error_rcbd < 0:
                    # на випадок сильної незбалансованості / пропусків
                    SS_error_rcbd = max(0.0, SS_error_rcbd)

                SS_error = SS_error_rcbd
                df_error = df_error_rcbd
                MS_error = (SS_error / df_error) if df_error > 0 else np.nan

        # -------------------------
        # Brown–Forsythe (тільки для параметричних)
        # -------------------------
        bf_F, bf_p = (np.nan, np.nan)
        if method in ("lsd", "tukey", "duncan", "bonferroni"):
            bf_F, bf_p = brown_forsythe_from_groups(groups1)

        # -------------------------
        # Nonparametric globals:
        # CRD: KW/MW (як є)
        # RCBD: Friedman/Wilcoxon
        # -------------------------
        kw_H, kw_p, kw_df, kw_eps2 = (np.nan, np.nan, np.nan, np.nan)

        # RCBD nonparam results
        fr_stat, fr_p, fr_df, fr_W = (np.nan, np.nan, np.nan, np.nan)
        w_stat, w_p, w_n = (np.nan, np.nan, np.nan)

        # для RCBD непараметричних (матриця блоків)
        rcbd_variant_order = None
        rcbd_v_names = None
        rcbd_blocks_order = None
        rcbd_block_values = None
        rcbd_complete_blocks = None
        fr_mean_ranks = {}
        rcbd_pairwise_rows = []
        rcbd_sig = {}
        do_posthoc_rcbd = True

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

        if method in ("friedman", "wilcoxon"):
            rcbd_variant_order, rcbd_v_names, rcbd_blocks_order, rcbd_block_values, rcbd_complete_blocks = \
                build_rcbd_block_map(long, self.factor_keys, block_key="_block")

            if not rcbd_complete_blocks or len(rcbd_complete_blocks) < 2:
                messagebox.showwarning(
                    "Помилка",
                    "Для RCBD непараметричних тестів потрібні повні блоки.\n"
                    "Переконайтеся, що у кожному блоці заповнені ВСІ варіанти."
                )
                return

            if method == "friedman":
                k = len(rcbd_v_names)
                n_blocks = len(rcbd_complete_blocks)
                arrays = []
                for vn in rcbd_v_names:
                    arrays.append([rcbd_block_values[bl][vn] for bl in rcbd_complete_blocks])

                try:
                    fr = friedmanchisquare(*arrays)
                    fr_stat = float(fr.statistic)
                    fr_p = float(fr.pvalue)
                    fr_df = int(k - 1)
                    fr_W = float(fr_stat / (n_blocks * (k - 1))) if (n_blocks > 0 and k > 1) else np.nan
                except Exception:
                    fr_stat, fr_p, fr_df, fr_W = (np.nan, np.nan, np.nan, np.nan)

                fr_mean_ranks = friedman_mean_ranks(rcbd_v_names, rcbd_complete_blocks, rcbd_block_values)

                # posthoc: тільки якщо p<alpha
                if not (isinstance(fr_p, float) and math.isnan(fr_p)) and fr_p >= ALPHA:
                    do_posthoc_rcbd = False
                if do_posthoc_rcbd:
                    rcbd_pairwise_rows, rcbd_sig = pairwise_wilcoxon_bonf(
                        rcbd_v_names, rcbd_complete_blocks, rcbd_block_values, alpha=ALPHA
                    )

            elif method == "wilcoxon":
                # лише 2 варіанти
                if len(rcbd_v_names) != 2:
                    messagebox.showwarning("Помилка", "Wilcoxon (парний) застосовується лише для 2 варіантів у RCBD.")
                    return
                a, b = rcbd_v_names[0], rcbd_v_names[1]
                xa = [rcbd_block_values[bl][a] for bl in rcbd_complete_blocks]
                xb = [rcbd_block_values[bl][b] for bl in rcbd_complete_blocks]
                w_n = int(len(xa))
                try:
                    stat, p = wilcoxon(xa, xb, alternative="two-sided")
                    w_stat = float(stat)
                    w_p = float(p)
                except Exception:
                    w_stat, w_p = (np.nan, np.nan)
                # для узгодженості звіту — зробимо "парне порівняння" як одну строку (без Bonf.)
                rbc = wilcoxon_rank_biserial(np.array(xa), np.array(xb))
                rcbd_pairwise_rows = [[f"{a}  vs  {b}", fmt_num(w_stat, 2), f"{w_p:.4f}",
                                       "+" if (not (isinstance(w_p, float) and math.isnan(w_p)) and w_p < ALPHA) else "-",
                                       fmt_num(rbc, 3), rbc_label(abs(rbc))]]
                rcbd_sig = {(a, b): (not (isinstance(w_p, float) and math.isnan(w_p)) and w_p < ALPHA)}
                fr_mean_ranks = friedman_mean_ranks(rcbd_v_names, rcbd_complete_blocks, rcbd_block_values)

        # -------------------------
        # Letters
        # -------------------------
        letters_factor = {}
        letters_named = {name: "" for name in v_names}
        pairwise_rows = []
        do_posthoc = True

        if method == "lsd":
            for f in self.factor_keys:
                lvls = levels_by_factor[f]
                sig = lsd_sig_matrix(lvls, factor_means[f], factor_ns[f], MS_error, df_error, alpha=ALPHA)
                letters_factor[f] = cld_multi_letters(lvls, factor_means[f], sig)

            sigv = lsd_sig_matrix(v_names, means1, ns1, MS_error, df_error, alpha=ALPHA)
            letters_named = cld_multi_letters(v_names, means1, sigv)

        elif method in ("tukey", "duncan", "bonferroni"):
            for f in self.factor_keys:
                letters_factor[f] = {lvl: "" for lvl in levels_by_factor[f]}
            pairwise_rows, sig = pairwise_param_short_variants_pm(v_names, means1, ns1, MS_error, df_error, method, alpha=ALPHA)
            letters_named = cld_multi_letters(v_names, means1, sig)

        elif method == "kw":
            for f in self.factor_keys:
                letters_factor[f] = {lvl: "" for lvl in levels_by_factor[f]}
            if not (isinstance(kw_p, float) and math.isnan(kw_p)) and kw_p >= ALPHA:
                do_posthoc = False
            if do_posthoc:
                pairwise_rows, sig = pairwise_mw_bonf_with_effect(v_names, groups1, alpha=ALPHA)
                med_tmp = {name: float(np.median(groups1[name])) if len(groups1[name]) else np.nan for name in v_names}
                letters_named = cld_multi_letters(v_names, med_tmp, sig)
            else:
                letters_named = {name: "" for name in v_names}

        elif method == "mw":
            for f in self.factor_keys:
                letters_factor[f] = {lvl: "" for lvl in levels_by_factor[f]}
            pairwise_rows, sig = pairwise_mw_bonf_with_effect(v_names, groups1, alpha=ALPHA)
            med_tmp = {name: float(np.median(groups1[name])) if len(groups1[name]) else np.nan for name in v_names}
            letters_named = cld_multi_letters(v_names, med_tmp, sig)

        elif method in ("friedman", "wilcoxon"):
            # факторні літери не формуємо (не змінюємо структуру вводу/факторів)
            for f in self.factor_keys:
                letters_factor[f] = {lvl: "" for lvl in levels_by_factor[f]}

            # літери по варіантах — на основі RCBD pairwise sig (або порожньо, якщо постхок не робився)
            if method == "friedman":
                if not do_posthoc_rcbd:
                    letters_named = {vn: "" for vn in rcbd_v_names}
                else:
                    # для CLD потрібні "means"; для Friedman беремо mean ranks (менший ранг = менше значення)
                    # але CLD сортує за спаданням; тому інвертуємо (чим МЕНШИЙ ранг — тим "краще", ставимо більший score)
                    score = {vn: (-fr_mean_ranks.get(vn, np.nan)) for vn in rcbd_v_names}
                    letters_named = cld_multi_letters(rcbd_v_names, score, rcbd_sig)
            else:  # wilcoxon (2 варіанти)
                score = {vn: (-fr_mean_ranks.get(vn, np.nan)) for vn in rcbd_v_names}
                letters_named = cld_multi_letters(rcbd_v_names, score, rcbd_sig)

        # мапа літер на variant_order
        letters_variants = {variant_order[i]: letters_named.get(v_names[i], "") for i in range(len(variant_order))}

        # -------------------------
        # R2, CV (тільки для параметричних)
        # -------------------------
        R2 = (1.0 - (SS_error / SS_total)) if (not any(math.isnan(x) for x in [SS_total, SS_error]) and SS_total > 0) else np.nan

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
        seg.append(("text",
                    f"Дизайн експерименту:\t{'Повна рандомізація (CRD)' if design=='crd' else 'Блочна рандомізація (RCBD)'}\n\n"))
        seg.append(("text", f"Кількість варіантів:\t{num_variants}\nКількість повторностей:\t{len(used_rep_cols)}\nЗагальна кількість облікових значень:\t{len(long)}\n\n"))

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

seg.append(("text", "Пояснення позначень істотності: ** — p<0.01; * — p<0.05.\n"))
seg.append(("text", "У таблицях знак \"-\" свідчить що p ≥ 0.05.\n"))
seg.append(("text", "Істотна різниця (літери): різні літери свідчать про наявність істотної різниці.\n\n"))

if not math.isnan(W):
    seg.append(("text", f"Перевірка нормальності залишків (Shapiro–Wilk):\t{normality_text(p_norm)}\t(W={fmt_num(float(W),4)}; p={fmt_num(float(p_norm),4)})\n\n"))
else:
    seg.append(("text", "Перевірка нормальності залишків (Shapiro–Wilk):\tн/д\n\n"))


        nonparam = method in ("mw", "kw", "friedman", "wilcoxon")

        if not math.isnan(W):
            seg.append(("text", f"Перевірка нормальності залишків (Shapiro–Wilk):\t{normality_text(p_norm)}\t(W={fmt_num(float(W),4)}; p={fmt_num(float(p_norm),4)})\n\n"))
        else:
            seg.append(("text", "Перевірка нормальності залишків (Shapiro–Wilk):\tн/д\n\n"))
           
        # ---- CRD nonparam global test
        if method == "kw":
            if not (isinstance(kw_p, float) and math.isnan(kw_p)):
                concl = "істотна різниця " + significance_mark(kw_p) if kw_p < ALPHA else "-"
                seg.append(("text",
                            f"Глобальний тест між варіантами (Kruskal–Wallis):\t"
                            f"H={fmt_num(kw_H,4)}; df={int(kw_df)}; p={fmt_num(kw_p,4)}\t{concl}\n"))
                seg.append(("text", f"Розмір ефекту (ε²):\t{fmt_num(kw_eps2,4)}\n\n"))
            else:
                seg.append(("text", "Глобальний тест між варіантами (Kruskal–Wallis):\tн/д\n\n"))

        # ---- RCBD nonparam globals
        if method == "friedman":
            if not (isinstance(fr_p, float) and math.isnan(fr_p)):
                concl = "істотна різниця " + significance_mark(fr_p) if fr_p < ALPHA else "-"
                seg.append(("text",
                            f"Глобальний тест між варіантами (Friedman):\t"
                            f"χ²={fmt_num(fr_stat,4)}; df={int(fr_df)}; p={fmt_num(fr_p,4)}\t{concl}\n"))
                seg.append(("text", f"Коефіцієнт узгодженості Kendall’s W:\t{fmt_num(fr_W,4)}\n\n"))
            else:
                seg.append(("text", "Глобальний тест між варіантами (Friedman):\tн/д\n\n"))

        if method == "wilcoxon":
            if not (isinstance(w_p, float) and math.isnan(w_p)):
                concl = "істотна різниця " + significance_mark(w_p) if w_p < ALPHA else "-"
                seg.append(("text",
                            f"Тест Wilcoxon (парний):\t"
                            f"W={fmt_num(w_stat,4)}; n={int(w_n)}; p={fmt_num(w_p,4)}\t{concl}\n\n"))
            else:
                seg.append(("text", "Тест Wilcoxon (парний):\tн/д\n\n"))

        # ---- PARAMETRIC
        if not nonparam:
            if not any(math.isnan(x) for x in [bf_F, bf_p]):
                bf_concl = "умова виконується" if bf_p >= ALPHA else f"умова порушена {significance_mark(bf_p)}"
                seg.append(("text",
                            f"Перевірка однорідності дисперсій (Brown–Forsythe):\t"
                            f"F={fmt_num(bf_F,4)}; p={fmt_num(bf_p,4)}\t{bf_concl}\n\n"))
            else:
                seg.append(("text", "Перевірка однорідності дисперсій (Brown–Forsythe):\tн/д\n\n"))

            # ANOVA table: для RCBD додаємо рядок "Блок", і перераховуємо F,p для всіх факторів на MS_error_rcbd/df_error_rcbd
            anova_rows = []

            # ✅ RCBD: блок
            if design == "rcbd" and not any(math.isnan(x) for x in [SS_block, df_block, MS_error, df_error]) and df_block is not None:
                MS_block = (SS_block / df_block) if df_block > 0 else np.nan
                F_block = (MS_block / MS_error) if (MS_error > 0 and not math.isnan(MS_block)) else np.nan
                p_block = 1 - f_dist.cdf(F_block, df_block, df_error) if (not math.isnan(F_block)) else np.nan
                mark = significance_mark(p_block)
                concl = f"істотна різниця {mark}" if mark else "-"
                anova_rows.append(["Блок", fmt_num(SS_block, 2), str(int(df_block)), fmt_num(MS_block, 3),
                                   fmt_num(F_block, 3), fmt_num(p_block, 4), concl])

            # фактори/комбінації (оновлені F,p)
            for name, SSv, dfv, MSv, Fv, pv in res["table"]:
                df_txt = str(int(dfv)) if dfv is not None and not math.isnan(dfv) else ""
                if name in ("Залишок", "Загальна"):
                    if name == "Залишок":
                        anova_rows.append([name, fmt_num(SS_error, 2), str(int(df_error)), fmt_num(MS_error, 3), "", "", ""])
                    else:
                        anova_rows.append([name, fmt_num(SS_total, 2), str(int(len(values) - 1)), "", "", "", ""])
                    continue

                # recompute MS, F, p with corrected MS_error/df_error
                SSv_ = float(SSv) if SSv is not None and not (isinstance(SSv, float) and math.isnan(SSv)) else np.nan
                dfv_ = float(dfv) if dfv is not None and not (isinstance(dfv, float) and math.isnan(dfv)) else np.nan
                MSv_ = (SSv_ / dfv_) if (not math.isnan(SSv_) and not math.isnan(dfv_) and dfv_ > 0) else np.nan
                Fv_ = (MSv_ / MS_error) if (not math.isnan(MSv_) and not math.isnan(MS_error) and MS_error > 0) else np.nan
                pv_ = (1 - f_dist.cdf(Fv_, dfv_, df_error)) if (not math.isnan(Fv_) and not math.isnan(dfv_)) else np.nan

                mark = significance_mark(pv_)
                concl = f"істотна різниця {mark}" if mark else "-"
                anova_rows.append([name, fmt_num(SSv_, 2), df_txt, fmt_num(MSv_, 3), fmt_num(Fv_, 3), fmt_num(pv_, 4), concl])

            seg.append(("text", "ТАБЛИЦЯ 1. Дисперсійний аналіз (ANOVA)\n"))
            seg.append(("table", {
                "headers": ["Джерело", "SS", "df", "MS", "F", "p", "Висновок"],
                "rows": anova_rows,
                "padding_px": 32,
                "extra_gap_after_col": 0,
                "extra_gap_px": 60
            }))
            seg.append(("text", "\n"))

            # ефект (%SS) та partial eta^2: у RCBD блок включаємо як частину SS_total (але не в partial eta^2 для факторів)
            eff_rows = build_effect_strength_rows(
                [("Блок", SS_block, df_block, (SS_block/df_block if (df_block and df_block > 0 and not math.isnan(SS_block)) else np.nan), None, None)]
                + [r for r in res["table"] if r[0] != "Залишок"]
                + [("Залишок", SS_error, df_error, MS_error, None, None)]
            ) if (design == "rcbd" and not (isinstance(SS_block, float) and math.isnan(SS_block))) else build_effect_strength_rows(res["table"])

            seg.append(("text", "ТАБЛИЦЯ 2. Сила впливу факторів та їх комбінацій (% від SS)\n"))
            seg.append(("table", {"headers": ["Джерело", "%"], "rows": eff_rows}))
            seg.append(("text", "\n"))

            # partial eta^2: використовуємо SS_error (скоригований)
            pe2_rows = build_partial_eta2_rows_with_label(
                [(name, SSv, dfv, None, None, None) for (name, SSv, dfv, MSv, Fv, pv) in
                 ([("Блок", SS_block, df_block, None, None, None)] + [r for r in res["table"] if r[0] != "Загальна"] +
                  [("Загальна", SS_total, len(values)-1, None, None, None)])
                 ]
            )
            # Примітка: рядок "Блок" теж матиме partial η² відносно скоригованої похибки — це методично допустимо.
            seg.append(("text", "ТАБЛИЦЯ 3. Розмір ефекту (partial η²)\n"))
            seg.append(("table", {"headers": ["Джерело", "partial η²", "Висновок"], "rows": pe2_rows}))
            seg.append(("text", "\n"))

            seg.append(("text", "ТАБЛИЦЯ 4. Коефіцієнт варіації (CV, %)\n"))
            seg.append(("table", {"headers": ["Елемент", "CV, %"], "rows": cv_rows}))
            seg.append(("text", "\n"))

            seg.append(("text", f"Коефіцієнт детермінації:\tR²={fmt_num(R2, 4)}\n\n"))

            tno = 5
            if method == "lsd":
                # перерахунок НІР₀₅ на скоригованих MS_error/df_error
                tval = t.ppf(1 - ALPHA / 2, df_error) if df_error > 0 else np.nan
                nir_rows = []

                # Загальна (по варіантах)
                n_eff_cells = harmonic_mean([ns1.get(name, 0) for name in v_names])
                nir_all = tval * math.sqrt(2 * MS_error / n_eff_cells) if not any(
                    math.isnan(x) for x in [tval, MS_error, n_eff_cells]
                ) else np.nan

                # Факторні
                NIR05 = {}
                NIR05["Загальна"] = nir_all
                for fct in self.factor_keys:
                    marg = subset_stats(long, (fct,))
                    n_eff = harmonic_mean([v[1] for v in marg.values() if v[1] and v[1] > 0])
                    nir = tval * math.sqrt(2 * MS_error / n_eff) if not any(
                        math.isnan(x) for x in [tval, MS_error, n_eff]
                    ) else np.nan
                    NIR05[f"Фактор {fct}"] = nir

                for key in [f"Фактор {f}" for f in self.factor_keys] + ["Загальна"]:
                    if key in NIR05:
                        nir_rows.append([key, fmt_num(NIR05[key], 4)])

                seg.append(("text", "ТАБЛИЦЯ 5. Значення НІР₀₅\n"))
                seg.append(("table", {"headers": ["Елемент", "НІР₀₅"], "rows": nir_rows}))
                seg.append(("text", "\n"))
                tno = 6

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

        # ---- NONPARAMETRIC
        else:
            # CRD nonparam (як було)
            if method in ("kw", "mw"):
                tno = 1
                for f in self.factor_keys:
                    seg.append(("text", f"ТАБЛИЦЯ {tno}. Описова статистика по фактору {f} (непараметрична)\n"))
                    rows = []
                    for lvl in levels_by_factor[f]:
                        med = factor_medians[f].get(lvl, np.nan)
                        q1, q3 = factor_q[f].get(lvl, (np.nan, np.nan))
                        rank_m = ranks_by_factor[f].get(lvl, np.nan)
                        letter = letters_factor[f].get(lvl, "")
                        rows.append([
                            str(lvl),
                            str(int(factor_ns[f].get(lvl, 0))),
                            fmt_num(med, 3),
                            f"{fmt_num(q1,3)}–{fmt_num(q3,3)}" if not any(math.isnan(x) for x in [q1, q3]) else "",
                            fmt_num(rank_m, 2),
                            (letter if letter else "-")
                        ])
                    seg.append(("table", {"headers": [f"Градація {f}", "n", "Медіана", "Q1–Q3", "Середній ранг", "Істотна різниця"], "rows": rows}))
                    seg.append(("text", "\n"))
                    tno += 1

                seg.append(("text", f"ТАБЛИЦЯ {tno}. Описова статистика варіантів (непараметрична)\n"))
                rows = []
                for k in variant_order:
                    name = " | ".join(map(str, k))
                    med = v_medians.get(k, np.nan)
                    q1, q3 = v_q.get(k, (np.nan, np.nan))
                    rank_m = ranks_by_variant.get(name, np.nan)
                    letter = letters_variants.get(k, "")
                    rows.append([
                        name,
                        str(int(v_ns.get(k, 0))),
                        fmt_num(med, 3),
                        f"{fmt_num(q1,3)}–{fmt_num(q3,3)}" if not any(math.isnan(x) for x in [q1, q3]) else "",
                        fmt_num(rank_m, 2),
                        (letter if letter else "-")
                    ])
                seg.append(("table", {
                    "headers": ["Варіант", "n", "Медіана", "Q1–Q3", "Середній ранг", "Істотна різниця"],
                    "rows": rows,
                    "padding_px": 32,
                    "extra_gap_after_col": 0,
                    "extra_gap_px": 80
                }))
                seg.append(("text", "\n"))
                tno += 1

                if method == "kw" and not do_posthoc:
                    seg.append(("text", "Пост-хок порівняння не виконувалися, оскільки глобальний тест Kruskal–Wallis не виявив істотної різниці (p ≥ 0.05).\n\n"))
                else:
                    if pairwise_rows:
                        seg.append(("text", f"ТАБЛИЦЯ {tno}. Парні порівняння + ефект (Cliff’s δ)\n"))
                        seg.append(("table", {"headers": ["Комбінація варіантів", "U", "p (Bonf.)", "Істотна різниця", "δ", "Висновок"], "rows": pairwise_rows}))
                        seg.append(("text", "\n"))

            # RCBD nonparam (нові звіти)
            else:
                tno = 1
                # mean ranks (по блоках)
                seg.append(("text", f"ТАБЛИЦЯ {tno}. Середні ранги варіантів (в межах блоків)\n"))
                rows = []
                for vn in rcbd_v_names:
                    rows.append([vn, fmt_num(fr_mean_ranks.get(vn, np.nan), 3)])
                seg.append(("table", {"headers": ["Варіант", "Середній ранг"], "rows": rows}))
                seg.append(("text", "\n"))
                tno += 1

                if method == "friedman":
                    # глобальна таблиця
                    seg.append(("text", f"ТАБЛИЦЯ {tno}. Friedman test\n"))
                    concl = "+" if (not (isinstance(fr_p, float) and math.isnan(fr_p)) and fr_p < ALPHA) else "-"
                    seg.append(("table", {"headers": ["χ²", "df", "p", "Істотна різниця", "Kendall’s W"],
                                          "rows": [[fmt_num(fr_stat, 4), str(int(fr_df)), fmt_num(fr_p, 4), concl, fmt_num(fr_W, 4)]]}))
                    seg.append(("text", "\n"))
                    tno += 1

                    if not do_posthoc_rcbd:
                        seg.append(("text", "Пост-хок порівняння не виконувалися, оскільки глобальний тест Friedman не виявив істотної різниці (p ≥ 0.05).\n\n"))
                    else:
                        if rcbd_pairwise_rows:
                            seg.append(("text", f"ТАБЛИЦЯ {tno}. Парні порівняння (Wilcoxon) + Bonferroni + ефект (rank-biserial)\n"))
                            seg.append(("table", {"headers": ["Комбінація варіантів", "W", "p (Bonf.)", "Істотна різниця", "r (RBC)", "Висновок"],
                                                  "rows": rcbd_pairwise_rows}))
                            seg.append(("text", "\n"))

                else:  # wilcoxon (2 варіанти)
                    seg.append(("text", f"ТАБЛИЦЯ {tno}. Wilcoxon signed-rank (парний)\n"))
                    seg.append(("table", {"headers": ["Комбінація варіантів", "W", "p", "Істотна різниця", "r (RBC)", "Висновок"],
                                          "rows": rcbd_pairwise_rows}))
                    seg.append(("text", "\n"))

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
