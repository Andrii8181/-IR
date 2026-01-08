# main.py
# -*- coding: utf-8 -*-

"""
S.A.D. — Статистичний аналіз даних (Tkinter)

Потрібно: Python 3.8+, numpy, scipy, matplotlib
Встановлення: pip install numpy scipy matplotlib

Останні правки:
✅ Заголовок звіту: «ЗВІТ СТАТИСТИЧНОГО АНАЛІЗУ ДАНИХ».
✅ Пояснення позначень (**/*/літери/"-") перенесено одразу після рядка «Виконуваний статистичний аналіз».
✅ Іконка icon.ico: пошук у корені програми (папка скрипта), cwd, папка запуску (argv0), папка exe (sys.executable),
   підтримка PyInstaller (_MEIPASS). Ставимо iconbitmap для всіх вікон.
✅ ОБОВ’ЯЗКОВИЙ додатковий матеріал: boxplot генерується як PNG для всіх типів звітів
   і зберігається кнопкою «Зберегти boxplot (PNG)…» у вікні «Звіт».
✅ ДЛЯ ДВОФАКТОРНОГО ДОСЛІДУ: при натисканні «Аналіз даних» запитуємо:
   назву показника, одиниці виміру, тип дизайну (CRD / RCBD / Split-plot),
   а для Split-plot — головний фактор (A або B).
✅ Для RCBD і Split-plot реалізовано методично коректні терми похибки:
   RCBD: Error = Block×A×B
   Split-plot: Error(Main) = Block×Main; Error(Sub, Interaction) = Block×Main×Sub
✅ У режимі Split-plot пост-хок між повними комбінаціями (варіантами) НЕ виконується (методично правильно),
   натомість пост-хок виконується по факторах з відповідним термом похибки.
"""

import os
import sys
import math
import tempfile
from itertools import combinations
from collections import defaultdict
from datetime import datetime
from typing import Union

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
import tkinter.font as tkfont

import numpy as np

from scipy.stats import shapiro, t, f as f_dist
from scipy.stats import mannwhitneyu, kruskal, levene, rankdata
from scipy.stats import studentized_range

# matplotlib — для PNG графіка
import matplotlib
matplotlib.use("Agg")  # стабільно працює і у зібраному .exe
import matplotlib.pyplot as plt


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


def set_window_icon(win: Union[tk.Tk, tk.Toplevel]):
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


def autosize_toplevel_to_content(win: tk.Toplevel, min_w=520, min_h=220):
    """
    Робимо вікно достатнім за розміром під контент (щоб нічого не обрізалося),
    і ставимо мінімальні розміри.
    """
    win.update_idletasks()
    req_w = max(int(win.winfo_reqwidth()), min_w)
    req_h = max(int(win.winfo_reqheight()), min_h)
    win.geometry(f"{req_w}x{req_h}")
    win.minsize(req_w, req_h)
    win.update_idletasks()


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


def make_boxplot_png(groups_dict, title, ylabel, out_path, max_groups=40):
    items = [(k, v) for k, v in groups_dict.items() if v and len(v) > 0]
    if not items:
        return False, "Немає даних для boxplot."

    if len(items) > max_groups:
        items = items[:max_groups]
        title = f"{title} (перші {max_groups})"

    labels = [k for k, _ in items]
    data = [np.array(v, dtype=float) for _, v in items]

    cleaned = []
    cleaned_labels = []
    for lab, arr in zip(labels, data):
        arr = arr[~np.isnan(arr)]
        if len(arr) > 0:
            cleaned.append(arr)
            cleaned_labels.append(lab)

    if not cleaned:
        return False, "Усі значення порожні/NaN для boxplot."

    fig = plt.figure(figsize=(max(10, len(cleaned_labels) * 0.45), 6))
    ax = fig.add_subplot(111)
    ax.boxplot(cleaned, labels=cleaned_labels, showmeans=True)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True, ""


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
# ANOVA (generic 1–4) + expose SS/df dict
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

    # "внутрішньоклітинна" похибка (для CRD з повторностями)
    SS_error_within = 0.0
    for rec in long:
        key = tuple(rec.get(f) for f in factors)
        v = rec.get("value", np.nan)
        m = cell_means.get(key, np.nan)
        if not math.isnan(v) and not math.isnan(m):
            SS_error_within += (v - m) ** 2

    levels_count = {f: len(levels_by_factor[f]) for f in factors}
    total_cells = 1
    for fct in factors:
        total_cells *= levels_count[fct]

    df_total = N - 1
    df_error_within = N - total_cells
    if df_error_within <= 0:
        df_error_within = max(1, df_error_within)

    MS_error_within = SS_error_within / df_error_within if df_error_within > 0 else np.nan

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
            Fv = MSv / MS_error_within if (not math.isnan(MS_error_within) and MS_error_within > 0 and not math.isnan(MSv)) else np.nan
            pv = 1 - f_dist.cdf(Fv, dfv, df_error_within) if (not math.isnan(Fv) and not math.isnan(dfv)) else np.nan
            table_rows.append((name, SSv, dfv, MSv, Fv, pv))

    table_rows.append(("Залишок", SS_error_within, df_error_within, MS_error_within, None, None))
    table_rows.append(("Загальна", SS_total, df_total, None, None, None))

    # NIR05 (для CRD) — за внутрішньоклітинною похибкою
    tval = t.ppf(1 - ALPHA / 2, df_error_within) if df_error_within > 0 else np.nan
    NIR05 = {}
    n_eff_cells = harmonic_mean([n for n in cell_counts.values() if n and n > 0])
    nir_all = tval * math.sqrt(2 * MS_error_within / n_eff_cells) if not any(
        math.isnan(x) for x in [tval, MS_error_within, n_eff_cells]
    ) else np.nan
    NIR05["Загальна"] = nir_all
    for fct in factors:
        marg = subset_stats(long, (fct,))
        n_eff = harmonic_mean([v[1] for v in marg.values() if v[1] and v[1] > 0])
        nir = tval * math.sqrt(2 * MS_error_within / n_eff) if not any(
            math.isnan(x) for x in [tval, MS_error_within, n_eff]
        ) else np.nan
        NIR05[f"Фактор {fct}"] = nir

    return {
        "table": table_rows,
        "cell_means": cell_means,
        "cell_counts": cell_counts,
        "MS_error": MS_error_within,
        "df_error": df_error_within,
        "NIR05": NIR05,
        "SS_total": SS_total,
        "SS_error": SS_error_within,
        "SS_dict": SS,
        "df_dict": df,
        "grand_mean": float(grand_mean) if not math.isnan(grand_mean) else np.nan,
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
def build_effect_strength_rows_from_named_ss(named_ss: dict, ss_total: float):
    rows = []
    for name, SSv in named_ss.items():
        if SSv is None or (isinstance(SSv, float) and math.isnan(SSv)):
            continue
        pct = (SSv / ss_total * 100.0) if (ss_total and not math.isnan(ss_total) and ss_total > 0) else np.nan
        rows.append([name, fmt_num(pct, 2)])
    return rows


def build_partial_eta2_rows_with_label_from_named_ss(named_ss: dict, ss_error: float):
    rows = []
    for name, SSv in named_ss.items():
        if name.lower().startswith("error") or name.lower() in ("залишок", "загальна"):
            continue
        if SSv is None or (isinstance(SSv, float) and math.isnan(SSv)):
            continue
        if ss_error is None or (isinstance(ss_error, float) and math.isnan(ss_error)) or ss_error <= 0:
            pe2 = np.nan
        else:
            denom = SSv + ss_error
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
    for k, r in zip(keys, ranks):
        sums[k] += float(r)
        cnts[k] += 1
    return {k: (sums[k] / cnts[k]) for k in cnts.keys()}


# -------------------------
# 2-factor designs: CRD / RCBD / Split-plot (balanced assumption)
# -------------------------
def twofactor_design_anova(long, factorA="A", factorB="B", block_key="Block", design="crd", main_factor=None):
    """
    Повертає структуру:
    {
      "design": ...,
      "anova_rows": [(name, SS, df, MS, F, p, concl), ...],
      "ss_total": SS_total,
      "named_ss": {name: SS, ...},         # для "сили впливу"
      "error_terms": {effect_name: ("Error ...", MS_err, df_err)},  # який терм похибки використано
      "ms_for_posthoc": { "A": (MS, df), "B": (MS, df) },           # для LSD по факторах
      "note": str or ""
    }
    """
    # визначаємо рівні
    levelsA = first_seen_order([r.get(factorA) for r in long])
    levelsB = first_seen_order([r.get(factorB) for r in long])
    levelsBlock = first_seen_order([r.get(block_key) for r in long])

    # CRD: використовуємо загальну anova_n_way
    if design == "crd":
        levels_by_factor = {factorA: levelsA, factorB: levelsB}
        res = anova_n_way(long, [factorA, factorB], levels_by_factor)
        # Перетворимо в "уніфікований" формат
        anova_rows = []
        named_ss = {}
        ss_total = res.get("SS_total", np.nan)
        ss_error = res.get("SS_error", np.nan)

        for name, SSv, dfv, MSv, Fv, pv in res["table"]:
            if name == "Загальна":
                continue
            if name == "Залишок":
                named_ss["Error"] = SSv
                df_txt = str(int(dfv)) if dfv is not None and not math.isnan(dfv) else ""
                anova_rows.append(("Залишок", SSv, dfv, MSv, None, None, ""))
                continue
            named_ss[name] = SSv
            mark = significance_mark(pv)
            concl = f"істотна різниця {mark}" if mark else "-"
            anova_rows.append((name, SSv, dfv, MSv, Fv, pv, concl))

        # додаємо total окремо якщо треба
        return {
            "design": "crd",
            "anova_rows": anova_rows,
            "ss_total": ss_total,
            "named_ss": named_ss,
            "ss_error_primary": ss_error,
            "error_terms": {
                f"Фактор {factorA}": ("Залишок", res["MS_error"], res["df_error"]),
                f"Фактор {factorB}": ("Залишок", res["MS_error"], res["df_error"]),
                f"Фактор {factorA}×{factorB}": ("Залишок", res["MS_error"], res["df_error"]),
            },
            "ms_for_posthoc": {
                factorA: (res["MS_error"], res["df_error"]),
                factorB: (res["MS_error"], res["df_error"]),
            },
            "note": ""
        }

    # RCBD / Split-plot: запускаємо 3-факторну ANOVA на [Block, A, B]
    # і беремо потрібні SS/df для термів
    factors = [block_key, factorA, factorB]
    levels_by_factor = {block_key: levelsBlock, factorA: levelsA, factorB: levelsB}
    res3 = anova_n_way(long, factors, levels_by_factor)
    SSd = res3.get("SS_dict", {})
    dfd = res3.get("df_dict", {})
    ss_total = res3.get("SS_total", np.nan)

    def ss(term): return SSd.get(tuple(term), np.nan)
    def df(term): return dfd.get(tuple(term), np.nan)

    # назви для звіту
    name_block = "Блоки"
    name_A = f"Фактор {factorA}"
    name_B = f"Фактор {factorB}"
    name_AB = f"Фактор {factorA}×{factorB}"

    # терми взаємодій
    name_BlockA = f"{name_block}×{name_A.replace('Фактор ','')}"
    name_BlockB = f"{name_block}×{name_B.replace('Фактор ','')}"
    name_BlockAB = f"{name_block}×{name_A.replace('Фактор ','')}×{name_B.replace('Фактор ','')}"

    # ВАЖЛИВО: у нашій таблиці даних "повторності" — це блоки.
    # За 1 спостереження на клітинку, внутрішньоклітинна похибка ~0,
    # а "похибка" береться як найвища взаємодія (Block×A×B).
    SS_Block = ss([block_key])
    df_Block = df([block_key])

    SS_A = ss([factorA]); df_A = df([factorA])
    SS_B = ss([factorB]); df_B = df([factorB])
    SS_AB = ss([factorA, factorB]); df_AB = df([factorA, factorB])

    SS_BlockA = ss([block_key, factorA]); df_BlockA = df([block_key, factorA])
    SS_BlockB = ss([block_key, factorB]); df_BlockB = df([block_key, factorB])
    SS_BlockAB = ss([block_key, factorA, factorB]); df_BlockAB = df([block_key, factorA, factorB])

    # якщо дизайн неповний — може бути NaN/0
    def MS(SSv, dfv):
        if SSv is None or dfv is None or (isinstance(dfv, float) and math.isnan(dfv)) or dfv <= 0:
            return np.nan
        if isinstance(SSv, float) and math.isnan(SSv):
            return np.nan
        return float(SSv) / float(dfv)

    MS_Block = MS(SS_Block, df_Block)
    MS_A = MS(SS_A, df_A)
    MS_B = MS(SS_B, df_B)
    MS_AB = MS(SS_AB, df_AB)
    MS_BlockA = MS(SS_BlockA, df_BlockA)
    MS_BlockB = MS(SS_BlockB, df_BlockB)
    MS_BlockAB = MS(SS_BlockAB, df_BlockAB)

    def F_p(MS_eff, df_eff, MS_err, df_err):
        if any(x is None for x in [MS_eff, df_eff, MS_err, df_err]):
            return (np.nan, np.nan)
        if any(isinstance(x, float) and math.isnan(x) for x in [MS_eff, df_eff, MS_err, df_err]):
            return (np.nan, np.nan)
        if df_eff <= 0 or df_err <= 0 or MS_err <= 0:
            return (np.nan, np.nan)
        Fv = float(MS_eff) / float(MS_err)
        pv = float(1 - f_dist.cdf(Fv, df_eff, df_err))
        return (Fv, pv)

    anova_rows = []
    named_ss = {}

    # Загальні рядки SS для "сили впливу"
    # (показуємо всі ключові компоненти моделі, включно з error-термами)
    named_ss[name_block] = SS_Block
    named_ss[name_A] = SS_A
    named_ss[name_B] = SS_B
    named_ss[name_AB] = SS_AB

    error_terms = {}
    ms_for_posthoc = {}
    note = ""

    if design == "rcbd":
        # Error = Block×A×B
        MS_err = MS_BlockAB
        df_err = df_BlockAB
        err_name = f"Error (Блок×{factorA}×{factorB})"

        # A,B,AB тестуємо на цьому error
        F_A, p_A = F_p(MS_A, df_A, MS_err, df_err)
        F_B, p_B = F_p(MS_B, df_B, MS_err, df_err)
        F_AB, p_AB = F_p(MS_AB, df_AB, MS_err, df_err)

        def concl(pv):
            mk = significance_mark(pv)
            return f"істотна різниця {mk}" if mk else "-"

        anova_rows.append((name_block, SS_Block, df_Block, MS_Block, None, None, ""))
        anova_rows.append((name_A, SS_A, df_A, MS_A, F_A, p_A, concl(p_A)))
        anova_rows.append((name_B, SS_B, df_B, MS_B, F_B, p_B, concl(p_B)))
        anova_rows.append((name_AB, SS_AB, df_AB, MS_AB, F_AB, p_AB, concl(p_AB)))

        named_ss[err_name] = SS_BlockAB
        anova_rows.append((err_name, SS_BlockAB, df_BlockAB, MS_BlockAB, None, None, ""))

        error_terms[name_A] = (err_name, MS_err, df_err)
        error_terms[name_B] = (err_name, MS_err, df_err)
        error_terms[name_AB] = (err_name, MS_err, df_err)
        ms_for_posthoc[factorA] = (MS_err, df_err)
        ms_for_posthoc[factorB] = (MS_err, df_err)

        return {
            "design": "rcbd",
            "anova_rows": anova_rows,
            "ss_total": ss_total,
            "named_ss": named_ss,
            "ss_error_primary": SS_BlockAB,
            "error_terms": error_terms,
            "ms_for_posthoc": ms_for_posthoc,
            "note": ""
        }

    if design == "splitplot":
        # Main factor вибирає користувач
        if main_factor not in (factorA, factorB):
            main_factor = factorA
        sub_factor = factorB if main_factor == factorA else factorA

        name_M = f"Фактор {main_factor}"
        name_S = f"Фактор {sub_factor}"
        name_MS = f"Фактор {main_factor}×{sub_factor}"

        # Error(Main) = Block×Main
        # Error(Sub, Main×Sub) = Block×Main×Sub
        # У термінах нашої 3-way: Block×Main це або Block×A, або Block×B.
        if main_factor == factorA:
            SS_err_main, df_err_main, MS_err_main = SS_BlockA, df_BlockA, MS_BlockA
        else:
            SS_err_main, df_err_main, MS_err_main = SS_BlockB, df_BlockB, MS_BlockB

        SS_err_sub, df_err_sub, MS_err_sub = SS_BlockAB, df_BlockAB, MS_BlockAB

        # ефекти:
        # M: SS(M), df(M), MS(M) тестуємо на MS_err_main
        # S: SS(S), df(S), MS(S) тестуємо на MS_err_sub
        # M×S: SS(AB), df(AB), MS(AB) тестуємо на MS_err_sub
        if main_factor == factorA:
            SS_M, df_M, MS_M = SS_A, df_A, MS_A
            SS_S, df_S, MS_S = SS_B, df_B, MS_B
        else:
            SS_M, df_M, MS_M = SS_B, df_B, MS_B
            SS_S, df_S, MS_S = SS_A, df_A, MS_A

        SS_MS, df_MS, MS_MS = SS_AB, df_AB, MS_AB

        F_M, p_M = F_p(MS_M, df_M, MS_err_main, df_err_main)
        F_S, p_S = F_p(MS_S, df_S, MS_err_sub, df_err_sub)
        F_MS, p_MS = F_p(MS_MS, df_MS, MS_err_sub, df_err_sub)

        def concl(pv):
            mk = significance_mark(pv)
            return f"істотна різниця {mk}" if mk else "-"

        anova_rows.append((name_block, SS_Block, df_Block, MS_Block, None, None, ""))

        anova_rows.append((name_M, SS_M, df_M, MS_M, F_M, p_M, concl(p_M)))
        err_main_name = f"Error(Main): Блок×{main_factor}"
        named_ss[err_main_name] = SS_err_main
        anova_rows.append((err_main_name, SS_err_main, df_err_main, MS_err_main, None, None, ""))

        anova_rows.append((name_S, SS_S, df_S, MS_S, F_S, p_S, concl(p_S)))
        anova_rows.append((name_MS, SS_MS, df_MS, MS_MS, F_MS, p_MS, concl(p_MS)))

        err_sub_name = f"Error(Sub): Блок×{main_factor}×{sub_factor}"
        named_ss[err_sub_name] = SS_err_sub
        anova_rows.append((err_sub_name, SS_err_sub, df_err_sub, MS_err_sub, None, None, ""))

        error_terms[name_M] = (err_main_name, MS_err_main, df_err_main)
        error_terms[name_S] = (err_sub_name, MS_err_sub, df_err_sub)
        error_terms[name_MS] = (err_sub_name, MS_err_sub, df_err_sub)

        ms_for_posthoc[main_factor] = (MS_err_main, df_err_main)
        ms_for_posthoc[sub_factor] = (MS_err_sub, df_err_sub)

        note = ("Примітка: у режимі Split-plot пост-хок порівняння між повними комбінаціями (варіантами) "
                "не виконуються, оскільки для коректності потрібен окремий вибір терму похибки залежно від рівня рандомізації. "
                "Натомість пост-хок виконується по факторах із відповідними термами похибки.")

        # Оновимо named_ss для факторів у термінах M,S (а не просто A/B) — щоб у "силі впливу" було зрозуміло
        # (але залишимо і A/B не треба — інакше дублювання)
        named_ss.pop(name_A, None)
        named_ss.pop(name_B, None)
        named_ss.pop(name_AB, None)

        named_ss[name_block] = SS_Block
        named_ss[name_M] = SS_M
        named_ss[name_S] = SS_S
        named_ss[name_MS] = SS_MS

        # primary error для partial eta2: логічно брати Sub-error (найнижчий рівень)
        return {
            "design": "splitplot",
            "main_factor": main_factor,
            "sub_factor": sub_factor,
            "anova_rows": anova_rows,
            "ss_total": ss_total,
            "named_ss": named_ss,
            "ss_error_primary": SS_err_sub,
            "error_terms": error_terms,
            "ms_for_posthoc": ms_for_posthoc,
            "note": note
        }

    # fallback
    raise ValueError("Невідомий дизайн.")


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

        self._last_boxplot_path = None
        self._last_boxplot_error = ""

        self._last_design_code = None  # "crd"|"rcbd"|"splitplot"
        self._last_main_factor = None  # "A"|"B" для splitplot

    def ask_indicator_units(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Параметри звіту")
        dlg.resizable(True, True)
        set_window_icon(dlg)

        frm = tk.Frame(dlg, padx=16, pady=16)
        frm.pack(fill=tk.BOTH, expand=True)

        tk.Label(frm, text="Назва показника:", fg="#000000").grid(row=0, column=0, sticky="w", pady=6)
        e_ind = tk.Entry(frm, width=44, fg="#000000")
        e_ind.grid(row=0, column=1, pady=6, sticky="we")

        tk.Label(frm, text="Одиниці виміру:", fg="#000000").grid(row=1, column=0, sticky="w", pady=6)
        e_units = tk.Entry(frm, width=44, fg="#000000")
        e_units.grid(row=1, column=1, pady=6, sticky="we")

        frm.grid_columnconfigure(1, weight=1)

        out = {"ok": False, "indicator": "", "units": ""}

        def on_ok():
            out["indicator"] = e_ind.get().strip()
            out["units"] = e_units.get().strip()
            if not out["indicator"] or not out["units"]:
                messagebox.showwarning("Помилка", "Заповніть назву показника та одиниці виміру.")
                return
            out["ok"] = True
            dlg.destroy()

        btns = tk.Frame(frm)
        btns.grid(row=2, column=0, columnspan=2, pady=(12, 0))
        tk.Button(btns, text="OK", width=10, command=on_ok).pack(side=tk.LEFT, padx=6)
        tk.Button(btns, text="Скасувати", width=12, command=lambda: dlg.destroy()).pack(side=tk.LEFT, padx=6)

        autosize_toplevel_to_content(dlg, min_w=560, min_h=220)
        center_window(dlg)
        e_ind.focus_set()
        dlg.bind("<Return>", lambda e: on_ok())
        dlg.grab_set()
        self.root.wait_window(dlg)
        return out

    def ask_indicator_units_and_design_twofactor(self):
        """
        Для 2-факторного досліду:
        назва + одиниці + дизайн (CRD/RCBD/Split-plot),
        а для Split-plot — головний фактор (A або B).
        """
        dlg = tk.Toplevel(self.root)
        dlg.title("Параметри звіту (2-факторний дослід)")
        dlg.resizable(True, True)
        set_window_icon(dlg)

        frm = tk.Frame(dlg, padx=16, pady=16)
        frm.pack(fill=tk.BOTH, expand=True)

        tk.Label(frm, text="Назва показника:", fg="#000000").grid(row=0, column=0, sticky="w", pady=6)
        e_ind = tk.Entry(frm, width=44, fg="#000000")
        e_ind.grid(row=0, column=1, pady=6, sticky="we")

        tk.Label(frm, text="Одиниці виміру:", fg="#000000").grid(row=1, column=0, sticky="w", pady=6)
        e_units = tk.Entry(frm, width=44, fg="#000000")
        e_units.grid(row=1, column=1, pady=6, sticky="we")

        tk.Label(frm, text="Тип дизайну експерименту:", fg="#000000").grid(row=2, column=0, sticky="w", pady=(10, 6))

        design_var = tk.StringVar(value="crd")
        rb_frame = tk.Frame(frm)
        rb_frame.grid(row=2, column=1, sticky="w", pady=(10, 6))

        tk.Radiobutton(rb_frame, text="CRD — повна рандомізація", variable=design_var, value="crd").pack(anchor="w")
        tk.Radiobutton(rb_frame, text="RCBD — блочна рандомізація", variable=design_var, value="rcbd").pack(anchor="w")
        tk.Radiobutton(rb_frame, text="Split-plot", variable=design_var, value="splitplot").pack(anchor="w")

        # Головний фактор (показуємо тільки для splitplot)
        main_row = 3
        tk.Label(frm, text="Головний фактор (лише для Split-plot):", fg="#000000").grid(
            row=main_row, column=0, sticky="w", pady=(10, 6)
        )

        main_var = tk.StringVar(value="A")
        main_combo = ttk.Combobox(frm, textvariable=main_var, values=["A", "B"], width=10, state="readonly")
        main_combo.grid(row=main_row, column=1, sticky="w", pady=(10, 6))

        hint = tk.Label(frm, text="", fg="#555555", justify="left", wraplength=520)
        hint.grid(row=main_row + 1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        def refresh_visibility(*_):
            is_sp = (design_var.get() == "splitplot")
            # в splitplot — показуємо; інакше — приховуємо, але залишаємо значення
            if is_sp:
                main_combo.configure(state="readonly")
                hint.configure(text="У Split-plot головний фактор визначає рівень рандомізації та терм похибки "
                                    "для перевірки значущості. Пост-хок між повними комбінаціями не виконується.")
            else:
                main_combo.configure(state="disabled")
                hint.configure(text="")
            dlg.update_idletasks()
            autosize_toplevel_to_content(dlg, min_w=660, min_h=320)
            center_window(dlg)

        design_var.trace_add("write", refresh_visibility)
        refresh_visibility()

        frm.grid_columnconfigure(1, weight=1)

        out = {"ok": False, "indicator": "", "units": "", "design": "crd", "main_factor": "A"}

        def on_ok():
            out["indicator"] = e_ind.get().strip()
            out["units"] = e_units.get().strip()
            out["design"] = design_var.get().strip() or "crd"
            out["main_factor"] = main_var.get().strip() or "A"

            if not out["indicator"] or not out["units"]:
                messagebox.showwarning("Помилка", "Заповніть назву показника та одиниці виміру.")
                return
            if out["design"] not in ("crd", "rcbd", "splitplot"):
                out["design"] = "crd"
            if out["design"] == "splitplot" and out["main_factor"] not in ("A", "B"):
                out["main_factor"] = "A"

            out["ok"] = True
            dlg.destroy()

        btns = tk.Frame(frm)
        btns.grid(row=main_row + 2, column=0, columnspan=2, pady=(14, 0))
        tk.Button(btns, text="OK", width=10, command=on_ok).pack(side=tk.LEFT, padx=6)
        tk.Button(btns, text="Скасувати", width=12, command=lambda: dlg.destroy()).pack(side=tk.LEFT, padx=6)

        autosize_toplevel_to_content(dlg, min_w=660, min_h=320)
        center_window(dlg)
        e_ind.focus_set()
        dlg.bind("<Return>", lambda e: on_ok())
        dlg.grab_set()
        self.root.wait_window(dlg)
        return out

    def choose_method_window(self, p_norm):
        dlg = tk.Toplevel(self.root)
        dlg.title("Вибір виду аналізу")
        dlg.resizable(True, True)
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

        autosize_toplevel_to_content(dlg, min_w=560, min_h=260)
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
            "Версія: 1.0\n"
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

    def collect_long(self, design_code=None):
        """
        Для CRD: просто збираємо значення.
        Для RCBD/Split-plot (2-факторний): повторності трактуємо як блоки (Block = 1..r).
        """
        long = []
        rep_cols = self.used_repeat_columns()
        if not rep_cols:
            return long, rep_cols

        for i, row in enumerate(self.entries):
            levels = []
            for k in range(self.factors_count):
                v = row[k].get().strip()
                if v == "":
                    v = f"рядок{i+1}"
                levels.append(v)

            for idx_c, c in enumerate(rep_cols):
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

                # Для 2-факторного RCBD/Split-plot: додаємо Block
                if self.factors_count == 2 and design_code in ("rcbd", "splitplot"):
                    rec["Block"] = f"Блок {idx_c+1}"

                long.append(rec)

        return long, rep_cols

    def analyze(self):
        created_at = datetime.now()

        # ---- параметри звіту
        if self.factors_count == 2:
            params = self.ask_indicator_units_and_design_twofactor()
            if not params["ok"]:
                return
            indicator = params["indicator"]
            units = params["units"]
            design_code = params.get("design", "crd")
            main_factor = params.get("main_factor", "A")
        else:
            params = self.ask_indicator_units()
            if not params["ok"]:
                return
            indicator = params["indicator"]
            units = params["units"]
            design_code = None
            main_factor = None

        self._last_design_code = design_code
        self._last_main_factor = main_factor

        long, used_rep_cols = self.collect_long(design_code=design_code if self.factors_count == 2 else None)
        if len(long) == 0:
            messagebox.showwarning("Помилка", "Немає числових даних для аналізу.\nПеревірте повторності та значення.")
            return

        values = np.array([r["value"] for r in long], dtype=float)
        if len(values) < 3:
            messagebox.showinfo("Результат", "Надто мало даних для аналізу.")
            return

        levels_by_factor = {f: first_seen_order([r.get(f) for r in long]) for f in self.factor_keys}

        # ---- ANOVA
        try:
            if self.factors_count == 2 and design_code in ("crd", "rcbd", "splitplot"):
                design_res = twofactor_design_anova(
                    long,
                    factorA="A", factorB="B",
                    block_key="Block",
                    design=design_code,
                    main_factor=main_factor if design_code == "splitplot" else None
                )
                # для залишків та Shapiro використовуємо "повну" модель клітинок (A×B [+ Block якщо є])
                # як і раніше: залишки = y - mean(cell)
                if design_code in ("rcbd", "splitplot"):
                    cell_keys = ["Block", "A", "B"]
                else:
                    cell_keys = ["A", "B"]
                cell_means = subset_stats(long, tuple(cell_keys))
                cell_mean_map = {k: v[0] for k, v in cell_means.items()}

                residuals = []
                for rec in long:
                    key = tuple(rec.get(k) for k in cell_keys)
                    v = rec.get("value", np.nan)
                    m = cell_mean_map.get(key, np.nan)
                    if not math.isnan(v) and not math.isnan(m):
                        residuals.append(v - m)
                residuals = np.array(residuals, dtype=float)

                # Для подальших частин коду (пост-хок, таблиці середніх) — робимо "імітацію" res як у старому форматі
                res = None  # не використовуємо anova_n_way напряму тут
            else:
                res = anova_n_way(long, self.factor_keys, levels_by_factor)
                cell_means = res.get("cell_means", {})
                residuals = []
                for rec in long:
                    key = tuple(rec.get(f) for f in self.factor_keys)
                    v = rec.get("value", np.nan)
                    m = cell_means.get(key, np.nan)
                    if not math.isnan(v) and not math.isnan(m):
                        residuals.append(v - m)
                residuals = np.array(residuals, dtype=float)
                design_res = None
        except Exception as ex:
            messagebox.showerror("Помилка аналізу", str(ex))
            return

        # ---- Shapiro
        try:
            W, p_norm = shapiro(residuals) if len(residuals) >= 3 else (np.nan, np.nan)
        except Exception:
            W, p_norm = (np.nan, np.nan)

        choice = self.choose_method_window(p_norm)
        if not choice["ok"]:
            return
        method = choice["method"]

        # ---- групи/середні по факторах (завжди)
        factor_groups = {f: {k[0]: v for k, v in groups_by_keys(long, (f,)).items()} for f in self.factor_keys}
        factor_means = {f: {lvl: float(np.mean(arr)) if len(arr) else np.nan for lvl, arr in factor_groups[f].items()} for f in self.factor_keys}
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

        # ---- варіанти (повні комбінації A×B×...)
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

        # -------------------------
        # ОБОВ’ЯЗКОВИЙ boxplot PNG (для всіх звітів)
        # -------------------------
        tmp_dir = tempfile.gettempdir()
        safe_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tmp_plot_path = os.path.join(tmp_dir, f"SAD_boxplot_{safe_ts}.png")

        ok_plot, plot_err = make_boxplot_png(
            groups_dict=groups1,
            title=f"Boxplot: {indicator}",
            ylabel=f"{indicator} ({units})",
            out_path=tmp_plot_path,
            max_groups=40
        )
        self._last_boxplot_path = tmp_plot_path if ok_plot else None
        self._last_boxplot_error = plot_err if not ok_plot else ""

        ranks_by_variant = mean_ranks_by_key(long, key_func=lambda rec: " | ".join(str(rec.get(f)) for f in self.factor_keys))
        ranks_by_factor = {f: mean_ranks_by_key(long, key_func=lambda rec, ff=f: rec.get(ff)) for f in self.factor_keys}

        # ---- Brown–Forsythe (параметричні методи)
        bf_F, bf_p = (np.nan, np.nan)
        if method in ("lsd", "tukey", "duncan", "bonferroni"):
            bf_F, bf_p = brown_forsythe_from_groups(groups1)

        # ---- KW глобальний (як раніше)
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

        # ---- letters for factors (LSD) — з урахуванням дизайну для 2-факторного
        letters_factor = {}
        if method == "lsd":
            if self.factors_count == 2 and design_res is not None:
                # беремо MS/df відповідно до дизайну:
                msA, dfA = design_res["ms_for_posthoc"].get("A", (np.nan, np.nan))
                msB, dfB = design_res["ms_for_posthoc"].get("B", (np.nan, np.nan))

                lvlsA = levels_by_factor["A"]
                lvlsB = levels_by_factor["B"]

                sigA = lsd_sig_matrix(lvlsA, factor_means["A"], factor_ns["A"], msA, dfA, alpha=ALPHA)
                sigB = lsd_sig_matrix(lvlsB, factor_means["B"], factor_ns["B"], msB, dfB, alpha=ALPHA)

                letters_factor["A"] = cld_multi_letters(lvlsA, factor_means["A"], sigA)
                letters_factor["B"] = cld_multi_letters(lvlsB, factor_means["B"], sigB)
            else:
                for f in self.factor_keys:
                    lvls = levels_by_factor[f]
                    # беремо MS_error з res
                    MS_error = res.get("MS_error", np.nan) if res else np.nan
                    df_error = res.get("df_error", np.nan) if res else np.nan
                    sig = lsd_sig_matrix(lvls, factor_means[f], factor_ns[f], MS_error, df_error, alpha=ALPHA)
                    letters_factor[f] = cld_multi_letters(lvls, factor_means[f], sig)
        else:
            for f in self.factor_keys:
                letters_factor[f] = {lvl: "" for lvl in levels_by_factor[f]}

        # ---- letters for variants + pairwise (з урахуванням split-plot)
        letters_named = {name: "" for name in v_names}
        pairwise_rows = []
        do_posthoc = True
        splitplot_block = (self.factors_count == 2 and design_res is not None and design_res.get("design") == "splitplot")

        if method == "lsd":
            if splitplot_block:
                # у split-plot не робимо пост-хок для повних варіантів
                letters_named = {name: "" for name in v_names}
                pairwise_rows = []
            else:
                MS_error = (res.get("MS_error", np.nan) if res else
                            (design_res["ms_for_posthoc"].get("A", (np.nan, np.nan))[0] if design_res else np.nan))
                df_error = (res.get("df_error", np.nan) if res else
                            (design_res["ms_for_posthoc"].get("A", (np.nan, np.nan))[1] if design_res else np.nan))
                sigv = lsd_sig_matrix(v_names, means1, ns1, MS_error, df_error, alpha=ALPHA)
                letters_named = cld_multi_letters(v_names, means1, sigv)

        elif method in ("tukey", "duncan", "bonferroni"):
            if splitplot_block:
                letters_named = {name: "" for name in v_names}
                pairwise_rows = []
            else:
                MS_error = res.get("MS_error", np.nan) if res else np.nan
                df_error = res.get("df_error", np.nan) if res else np.nan
                pairwise_rows, sig = pairwise_param_short_variants_pm(v_names, means1, ns1, MS_error, df_error, method, alpha=ALPHA)
                letters_named = cld_multi_letters(v_names, means1, sig)

        elif method == "kw":
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

        # ---- CV
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
        seg.append(("text", f"Показник:\t{indicator}\nОдиниці виміру:\t{units}\n"))

        if self.factors_count == 2:
            if design_code == "crd":
                design_label = "CRD — повна рандомізація"
            elif design_code == "rcbd":
                design_label = "RCBD — блочна рандомізація"
            else:
                design_label = "Split-plot"
            seg.append(("text", f"Тип дизайну експерименту:\t{design_label}\n"))
            if design_code == "splitplot":
                seg.append(("text", f"Головний фактор:\t{main_factor}\n"))
        seg.append(("text", "\n"))

        seg.append(("text", f"Кількість варіантів:\t{num_variants}\nКількість повторностей:\t{len(used_rep_cols)}\nЗагальна кількість облікових значень:\t{len(long)}\n\n"))

        if not math.isnan(W):
            seg.append(("text", f"Перевірка нормальності залишків (Shapiro–Wilk):\t{normality_text(p_norm)}\t(W={fmt_num(float(W),4)}; p={fmt_num(float(p_norm),4)})\n\n"))
        else:
            seg.append(("text", "Перевірка нормальності залишків (Shapiro–Wilk):\tн/д\n\n"))

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

        nonparam = method in ("mw", "kw")

        if method == "kw":
            if not (isinstance(kw_p, float) and math.isnan(kw_p)):
                concl = "істотна різниця " + significance_mark(kw_p) if kw_p < ALPHA else "-"
                seg.append(("text",
                            f"Глобальний тест між варіантами (Kruskal–Wallis):\t"
                            f"H={fmt_num(kw_H,4)}; df={int(kw_df)}; p={fmt_num(kw_p,4)}\t{concl}\n"))
                seg.append(("text", f"Розмір ефекту (ε²):\t{fmt_num(kw_eps2,4)}\n\n"))
            else:
                seg.append(("text", "Глобальний тест між варіантами (Kruskal–Wallis):\tн/д\n\n"))

        if not nonparam:
            # Brown–Forsythe
            if not any(math.isnan(x) for x in [bf_F, bf_p]):
                bf_concl = "умова виконується" if bf_p >= ALPHA else f"умова порушена {significance_mark(bf_p)}"
                seg.append(("text",
                            f"Перевірка однорідності дисперсій (Brown–Forsythe):\t"
                            f"F={fmt_num(bf_F,4)}; p={fmt_num(bf_p,4)}\t{bf_concl}\n\n"))
            else:
                seg.append(("text", "Перевірка однорідності дисперсій (Brown–Forsythe):\tн/д\n\n"))

            # --- ANOVA table
            seg.append(("text", "ТАБЛИЦЯ 1. Дисперсійний аналіз (ANOVA)\n"))

            if self.factors_count == 2 and design_res is not None:
                anova_rows_print = []
                for name, SSv, dfv, MSv, Fv, pv, concl in design_res["anova_rows"]:
                    df_txt = str(int(dfv)) if dfv is not None and not (isinstance(dfv, float) and math.isnan(dfv)) else ""
                    if Fv is None or (isinstance(Fv, float) and math.isnan(Fv)):
                        anova_rows_print.append([name, fmt_num(SSv, 2), df_txt, fmt_num(MSv, 3), "", "", ""])
                    else:
                        anova_rows_print.append([name, fmt_num(SSv, 2), df_txt, fmt_num(MSv, 3), fmt_num(Fv, 3), fmt_num(pv, 4), concl])

                seg.append(("table", {
                    "headers": ["Джерело", "SS", "df", "MS", "F", "p", "Висновок"],
                    "rows": anova_rows_print,
                    "padding_px": 32,
                    "extra_gap_after_col": 0,
                    "extra_gap_px": 60
                }))
                seg.append(("text", "\n"))

                # --- Effect strength / partial eta2 with primary error
                named_ss = design_res.get("named_ss", {})
                ss_total = design_res.get("ss_total", np.nan)
                ss_err_primary = design_res.get("ss_error_primary", np.nan)

                eff_rows = build_effect_strength_rows_from_named_ss(named_ss, ss_total)
                seg.append(("text", "ТАБЛИЦЯ 2. Сила впливу факторів та компонентів моделі (% від SS)\n"))
                seg.append(("table", {"headers": ["Джерело", "%"], "rows": eff_rows}))
                seg.append(("text", "\n"))

                pe2_rows = build_partial_eta2_rows_with_label_from_named_ss(named_ss, ss_err_primary)
                seg.append(("text", "ТАБЛИЦЯ 3. Розмір ефекту (partial η²)\n"))
                seg.append(("table", {"headers": ["Джерело", "partial η²", "Висновок"], "rows": pe2_rows}))
                seg.append(("text", "\n"))

                seg.append(("text", "ТАБЛИЦЯ 4. Коефіцієнт варіації (CV, %)\n"))
                seg.append(("table", {"headers": ["Елемент", "CV, %"], "rows": cv_rows}))
                seg.append(("text", "\n"))

                # примітка splitplot
                if design_res.get("note"):
                    seg.append(("text", f"{design_res['note']}\n\n"))

                tno = 5

                # NIR05 (тільки якщо LSD) — для 2-факторного, залежно від дизайну
                if method == "lsd":
                    nir_rows = []
                    if design_code == "crd":
                        # NIR у CRD розраховувався в anova_n_way — тут не доступний, тому пропускаємо, щоб не вигадувати
                        nir_rows = []
                    else:
                        # Для RCBD/Split-plot: показуємо "робочі" MS/df для LSD по факторах (це корисніше)
                        msA, dfA = design_res["ms_for_posthoc"].get("A", (np.nan, np.nan))
                        msB, dfB = design_res["ms_for_posthoc"].get("B", (np.nan, np.nan))
                        nir_rows.append(["Фактор A (терм похибки)", f"MS={fmt_num(msA,4)}; df={int(dfA) if dfA and not math.isnan(dfA) else ''}"])
                        nir_rows.append(["Фактор B (терм похибки)", f"MS={fmt_num(msB,4)}; df={int(dfB) if dfB and not math.isnan(dfB) else ''}"])

                    if nir_rows:
                        seg.append(("text", "ТАБЛИЦЯ 5. Параметри похибки для LSD по факторах\n"))
                        seg.append(("table", {"headers": ["Елемент", "Параметри"], "rows": nir_rows}))
                        seg.append(("text", "\n"))
                        tno = 6

                # факторні середні
                for f in self.factor_keys:
                    seg.append(("text", f"ТАБЛИЦЯ {tno}. Середнє по фактору {f}\n"))
                    rows_f = []
                    for lvl in levels_by_factor[f]:
                        m = factor_means[f].get(lvl, np.nan)
                        letter = letters_factor.get(f, {}).get(lvl, "")
                        rows_f.append([str(lvl), fmt_num(m, 3), (letter if letter else "-")])
                    seg.append(("table", {"headers": [f"Градація {f}", "Середнє", "Істотна різниця"], "rows": rows_f}))
                    seg.append(("text", "\n"))
                    tno += 1

                # таблиця варіантів (у splitplot — без літер)
                seg.append(("text", f"ТАБЛИЦЯ {tno}. Таблиця середніх значень варіантів\n"))
                rows_v = []
                for k in variant_order:
                    name = " | ".join(map(str, k))
                    m = v_means.get(k, np.nan)
                    sd = v_sds.get(k, np.nan)
                    if splitplot_block:
                        letter = ""
                        rows_v.append([name, fmt_num(m, 3), fmt_num(sd, 3), "-"])
                    else:
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

                # парні порівняння варіантів (у splitplot — не показуємо)
                if (method in ("tukey", "duncan", "bonferroni")) and pairwise_rows and (not splitplot_block):
                    seg.append(("text", f"ТАБЛИЦЯ {tno}. Парні порівняння варіантів\n"))
                    seg.append(("table", {"headers": ["Комбінація варіантів", "p", "Істотна різниця"], "rows": pairwise_rows}))
                    seg.append(("text", "\n"))

            else:
                # старий шлях для 1/3/4 факторів
                MS_error = res.get("MS_error", np.nan)
                df_error = res.get("df_error", np.nan)

                anova_rows = []
                for name, SSv, dfv, MSv, Fv, pv in res["table"]:
                    df_txt = str(int(dfv)) if dfv is not None and not math.isnan(dfv) else ""
                    if name in ("Залишок", "Загальна"):
                        anova_rows.append([name, fmt_num(SSv, 2), df_txt, fmt_num(MSv, 3), "", "", ""])
                    else:
                        mark = significance_mark(pv)
                        concl = f"істотна різниця {mark}" if mark else "-"
                        anova_rows.append([name, fmt_num(SSv, 2), df_txt, fmt_num(MSv, 3), fmt_num(Fv, 3), fmt_num(pv, 4), concl])

                seg.append(("table", {
                    "headers": ["Джерело", "SS", "df", "MS", "F", "p", "Висновок"],
                    "rows": anova_rows,
                    "padding_px": 32,
                    "extra_gap_after_col": 0,
                    "extra_gap_px": 60
                }))
                seg.append(("text", "\n"))

                # сила впливу та partial eta2 у старому форматі
                # (залишаємо як було)
                def build_effect_strength_rows(anova_table_rows):
                    SS_total_local = None
                    for name2, SSv2, dfv2, MSv2, Fv2, pv2 in anova_table_rows:
                        if name2 == "Загальна":
                            SS_total_local = SSv2
                    if SS_total_local is None or (isinstance(SS_total_local, float) and math.isnan(SS_total_local)) or SS_total_local <= 0:
                        SS_total_local = np.nan
                    rows2 = []
                    for name2, SSv2, dfv2, MSv2, Fv2, pv2 in anova_table_rows:
                        if SSv2 is None or (isinstance(SSv2, float) and math.isnan(SSv2)):
                            continue
                        if name2 == "Загальна":
                            pct = 100.0 if not math.isnan(SS_total_local) else np.nan
                        else:
                            pct = (SSv2 / SS_total_local * 100.0) if (not math.isnan(SS_total_local) and SS_total_local > 0) else np.nan
                        rows2.append([name2, fmt_num(pct, 2)])
                    return rows2

                def build_partial_eta2_rows_with_label(anova_table_rows):
                    SS_error_local = None
                    for name2, SSv2, dfv2, MSv2, Fv2, pv2 in anova_table_rows:
                        if name2 == "Залишок":
                            SS_error_local = SSv2
                            break
                    if SS_error_local is None or (isinstance(SS_error_local, float) and math.isnan(SS_error_local)) or SS_error_local <= 0:
                        SS_error_local = np.nan
                    rows2 = []
                    for name2, SSv2, dfv2, MSv2, Fv2, pv2 in anova_table_rows:
                        if name2 in ("Залишок", "Загальна"):
                            continue
                        if SSv2 is None or (isinstance(SSv2, float) and math.isnan(SSv2)):
                            continue
                        if math.isnan(SS_error_local):
                            pe2 = np.nan
                        else:
                            denom = SSv2 + SS_error_local
                            pe2 = (SSv2 / denom) if denom > 0 else np.nan
                        rows2.append([name2, fmt_num(pe2, 4), partial_eta2_label(pe2)])
                    return rows2

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

            seg.append(("text", f"Звіт сформовано:\t{created_at.strftime('%d.%m.%Y, %H:%M')}\n"))

        else:
            # непараметричні — як раніше (без ANOVA)
            tno = 1

            for f in self.factor_keys:
                seg.append(("text", f"ТАБЛИЦЯ {tno}. Описова статистика по фактору {f} (непараметрична)\n"))
                rows = []
                for lvl in levels_by_factor[f]:
                    med = factor_medians[f].get(lvl, np.nan)
                    q1, q3 = factor_q[f].get(lvl, (np.nan, np.nan))
                    rank_m = ranks_by_factor[f].get(lvl, np.nan)
                    letter = letters_factor.get(f, {}).get(lvl, "")
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
                arr = groups1.get(name, [])
                med, q1, q3 = median_q1_q3(arr)
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

        def save_boxplot_png():
            if (not self._last_boxplot_path) or (not os.path.exists(self._last_boxplot_path)):
                msg = self._last_boxplot_error or "Boxplot не створено."
                messagebox.showwarning("Boxplot", msg)
                return

            out = filedialog.asksaveasfilename(
                title="Зберегти boxplot (PNG)",
                defaultextension=".png",
                filetypes=[("PNG image", "*.png")]
            )
            if not out:
                return
            try:
                with open(self._last_boxplot_path, "rb") as fsrc:
                    data = fsrc.read()
                with open(out, "wb") as fdst:
                    fdst.write(data)
                messagebox.showinfo("Готово", "Boxplot збережено у PNG.")
            except Exception as ex:
                messagebox.showerror("Помилка", str(ex))

        tk.Button(top, text="Зберегти boxplot (PNG)…", command=save_boxplot_png).pack(side=tk.LEFT, padx=4)

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
