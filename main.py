# main.py
# -*- coding: utf-8 -*-
"""
S.A.D. — Статистичний аналіз даних (Tkinter)

Потрібно: Python 3.8+, numpy, scipy
Встановлення: pip install numpy scipy
"""

import os
import sys
import math
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import tkinter.font as tkfont
from itertools import combinations
from collections import defaultdict
from datetime import datetime

from scipy.stats import shapiro, kruskal, mannwhitneyu, friedmanchisquare, wilcoxon
from scipy.stats import f as f_dist, t as t_dist, norm
from scipy.stats import studentized_range



import io
import tempfile

# matplotlib (graphical report)
try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
except Exception:
    Figure = None
    FigureCanvasTkAgg = None
    NavigationToolbar2Tk = None

# Pillow for clipboard (optional)
try:
    from PIL import Image
except Exception:
    Image = None

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
# Clipboard helper (PNG -> Windows clipboard as DIB)
# -------------------------
def _copy_pil_image_to_clipboard_windows(pil_img):
    """
    Windows-only: копіює зображення в буфер обміну так, щоб його можна було вставити у Word (Ctrl+V),
    використовуючи формат CF_DIB (Device Independent Bitmap).

    Важливо для 64-bit Windows:
    у ctypes потрібно явно задати restype/argtypes для WinAPI, інакше GlobalLock може "падати"/повертати 0.
    """
    try:
        import ctypes
        from ctypes import wintypes
        import io
    except Exception:
        return False, "ctypes/IO недоступні."

    if sys.platform != "win32":
        return False, "Копіювання у буфер реалізовано лише для Windows."

    if pil_img is None:
        return False, "Немає зображення для копіювання."

    # 1) Готуємо DIB байти: зберігаємо як BMP і відкидаємо 14-байтний BMP header,
    # лишаючи BITMAPINFOHEADER + bitmap bits (це і є CF_DIB).
    try:
        with io.BytesIO() as output:
            pil_img.convert("RGB").save(output, "BMP")
            bmp = output.getvalue()
        if len(bmp) <= 14:
            return False, "Невірний BMP буфер."
        data = bmp[14:]
    except Exception as ex:
        return False, f"Не вдалося підготувати зображення: {ex}"

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    # 2) Прототипи (критично для 64-bit)
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype  = wintypes.BOOL
    user32.CloseClipboard.argtypes = []
    user32.CloseClipboard.restype  = wintypes.BOOL
    user32.EmptyClipboard.argtypes = []
    user32.EmptyClipboard.restype  = wintypes.BOOL
    user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
    user32.SetClipboardData.restype  = wintypes.HANDLE

    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype  = wintypes.HGLOBAL
    kernel32.GlobalLock.argtypes  = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype   = wintypes.LPVOID
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalUnlock.restype  = wintypes.BOOL
    kernel32.GlobalFree.argtypes  = [wintypes.HGLOBAL]
    kernel32.GlobalFree.restype   = wintypes.HGLOBAL

    CF_DIB = 8
    GMEM_MOVEABLE = 0x0002
    GMEM_ZEROINIT = 0x0040

    # 3) Запис у буфер обміну
    if not user32.OpenClipboard(None):
        err = ctypes.get_last_error()
        return False, f"Не вдалося відкрити буфер обміну (код {err})."
    try:
        user32.EmptyClipboard()

        h_global = kernel32.GlobalAlloc(GMEM_MOVEABLE | GMEM_ZEROINIT, len(data))
        if not h_global:
            err = ctypes.get_last_error()
            return False, f"GlobalAlloc не спрацював (код {err})."

        p_global = kernel32.GlobalLock(h_global)
        if not p_global:
            err = ctypes.get_last_error()
            kernel32.GlobalFree(h_global)
            return False, f"GlobalLock не спрацював (код {err})."

        try:
            ctypes.memmove(p_global, data, len(data))
        finally:
            kernel32.GlobalUnlock(h_global)

        # Важливо: після SetClipboardData власність переходить буферу обміну — НЕ звільняємо h_global
        if not user32.SetClipboardData(CF_DIB, h_global):
            err = ctypes.get_last_error()
            kernel32.GlobalFree(h_global)
            return False, f"SetClipboardData не спрацював (код {err})."

        return True, ""
    finally:
        user32.CloseClipboard()

def copy_figure_png_to_clipboard(fig):
    """Save matplotlib Figure to PNG bytes and copy to clipboard (Windows)."""
    if Figure is None or Image is None:
        return False, "Для копіювання потрібні matplotlib та Pillow."
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        pil = Image.open(buf)
        ok, msg = _copy_pil_image_to_clipboard_windows(pil)
        buf.close()
        return ok, msg
    except Exception as ex:
        return False, str(ex)

def save_figure_png(fig, filepath):
    if Figure is None:
        raise RuntimeError("matplotlib недоступний.")
    fig.savefig(filepath, format="png", dpi=300, bbox_inches="tight")

# -------------------------
# Small helpers
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
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return ""

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

def median_q1_q3(arr):
    if arr is None or len(arr) == 0:
        return (np.nan, np.nan, np.nan)
    a = np.array(arr, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return (np.nan, np.nan, np.nan)
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

def kendalls_w_from_friedman(chisq: float, n_blocks: int, k_treat: int) -> float:
    if any(x is None for x in [chisq, n_blocks, k_treat]):
        return np.nan
    if isinstance(chisq, float) and math.isnan(chisq):
        return np.nan
    if n_blocks <= 0 or k_treat <= 1:
        return np.nan
    return float(chisq / (n_blocks * (k_treat - 1)))

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


# -------------------------
# Text table helpers (report)
# -------------------------

def build_table_block(headers, rows):
    """
    Таблиця для текстового звіту (tab-separated).
    Важливо: без зайвих пробілів у заголовках, щоб таб-стопи не "зʼїжджали".
    """
    # convert all to strings, preserve order
    headers_s = ["" if h is None else str(h) for h in headers]
    rows_s = []
    for r in rows:
        rows_s.append(["" if v is None else str(v) for v in r])

    ncol = len(headers_s)
    # estimate column widths in characters (for underline only)
    col_w = [len(headers_s[j]) for j in range(ncol)]
    for r in rows_s:
        for j in range(ncol):
            col_w[j] = max(col_w[j], len(r[j]))

    lines = []
    lines.append("\t".join(headers_s))
    lines.append("\t".join("—" * max(3, col_w[j]) for j in range(ncol)))
    for r in rows_s:
        lines.append("\t".join(r))
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
# Data grouping helpers
# -------------------------
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

def mean_ranks_by_key(long, key_func):
    # rank over all observations, then average ranks within key
    vals = []
    keys = []
    for r in long:
        v = r.get("value", np.nan)
        if v is None or math.isnan(v):
            continue
        vals.append(float(v))
        keys.append(key_func(r))
    if len(vals) == 0:
        return {}
    # rankdata: small value -> small rank
    order = np.argsort(vals)
    ranks = np.empty(len(vals), dtype=float)
    # average ranks for ties
    sorted_vals = np.array(vals)[order]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j < len(sorted_vals) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    agg = defaultdict(list)
    for k, rnk in zip(keys, ranks):
        agg[k].append(float(rnk))
    return {k: float(np.mean(v)) for k, v in agg.items()}


# -------------------------
# CLD letters (compact letter display)
# -------------------------
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


# -------------------------
# Brown–Forsythe (robust homogeneity) from groups
# -------------------------
def brown_forsythe_from_groups(groups_dict):
    # groups_dict: name -> list
    arrs = [np.array(v, dtype=float) for v in groups_dict.values() if len(v) > 0]
    if len(arrs) < 2:
        return (np.nan, np.nan)
    z_groups = []
    for a in arrs:
        med = np.median(a)
        z_groups.append(np.abs(a - med))
    # one-way ANOVA on z
    allz = np.concatenate(z_groups)
    grand = np.mean(allz)
    ss_between = 0.0
    ss_within = 0.0
    n_total = len(allz)
    k = len(z_groups)
    for zg in z_groups:
        ni = len(zg)
        mi = np.mean(zg)
        ss_between += ni * (mi - grand) ** 2
        ss_within += np.sum((zg - mi) ** 2)
    dfb = k - 1
    dfw = n_total - k
    if dfb <= 0 or dfw <= 0 or ss_within <= 0:
        return (np.nan, np.nan)
    msb = ss_between / dfb
    msw = ss_within / dfw
    F = msb / msw
    p = float(1.0 - f_dist.cdf(F, dfb, dfw))
    return (float(F), p)


# -------------------------
# Pairwise helpers (parametric)
# -------------------------
def lsd_sig_matrix(levels, means, ns, MS, df, alpha=ALPHA):
    # Fisher LSD (uses t critical)
    sig = {}
    if MS is None or df is None or any(math.isnan(x) for x in [MS, df]):
        return sig
    df = int(df)
    if df <= 0:
        return sig
    tcrit = float(t_dist.ppf(1 - alpha / 2.0, df))
    for a, b in combinations(levels, 2):
        ma, mb = means.get(a, np.nan), means.get(b, np.nan)
        na, nb = ns.get(a, 0), ns.get(b, 0)
        if any(math.isnan(x) for x in [ma, mb]) or na <= 0 or nb <= 0:
            continue
        se = math.sqrt(MS * (1.0 / na + 1.0 / nb))
        lsd = tcrit * se
        sig[(a, b)] = (abs(ma - mb) > lsd)
    return sig

def pairwise_param_short_variants_pm(levels, means, ns, MS, df, method, alpha=ALPHA):
    # Returns rows: [A vs B, p, conclusion] + sig-matrix for CLD
    rows = []
    sig = {}
    if MS is None or df is None or any(math.isnan(x) for x in [MS, df]):
        return rows, sig
    df = int(df)
    if df <= 0:
        return rows, sig

    lvls = [x for x in levels if not math.isnan(means.get(x, np.nan)) and ns.get(x, 0) > 0]
    if len(lvls) < 2:
        return rows, sig

    m = len(lvls)
    # Tukey/Duncan use studentized range; Bonferroni uses t
    for a, b in combinations(lvls, 2):
        ma, mb = means[a], means[b]
        na, nb = ns[a], ns[b]
        se = math.sqrt(MS * (1.0 / na + 1.0 / nb))
        if se <= 0:
            continue
        tval = abs(ma - mb) / se
        # two-sided p for t
        p_raw = 2 * (1 - float(t_dist.cdf(tval, df)))

        if method == "bonferroni":
            p_adj = min(1.0, p_raw * (m * (m - 1) / 2.0))
        elif method == "tukey":
            # Tukey p-value via studentized range approx
            # q = |ma-mb| / sqrt(MS/2 * (1/na+1/nb)) ??? (unequal n) - we use harmonic-like: approximate with se_tukey
            # A pragmatic approximation: q = |ma-mb| / sqrt(MS/2 * (1/na+1/nb)) = sqrt(2)*tval
            q = math.sqrt(2.0) * tval
            p_adj = float(1 - studentized_range.cdf(q, m, df))
        elif method == "duncan":
            # Duncan is stepwise; implement conservative approx using Tukey p (close behavior, avoids wrong claims)
            q = math.sqrt(2.0) * tval
            p_adj = float(1 - studentized_range.cdf(q, m, df))
        else:
            p_adj = p_raw

        is_sig = (p_adj < alpha)
        sig[(a, b)] = is_sig
        rows.append([f"{a} vs {b}", fmt_num(p_adj, 4), ("істотна різниця " + significance_mark(p_adj)) if is_sig else "-"])
    return rows, sig


# -------------------------
# Pairwise helpers (nonparametric)
# -------------------------
def pairwise_mw_bonf_with_effect(levels, groups, alpha=ALPHA):
    rows = []
    sig = {}
    lvls = [x for x in levels if len(groups.get(x, [])) > 0]
    m = len(lvls)
    if m < 2:
        return rows, sig
    mtests = m * (m - 1) / 2.0
    for a, b in combinations(lvls, 2):
        x = np.array(groups[a], dtype=float)
        y = np.array(groups[b], dtype=float)
        try:
            U, p = mannwhitneyu(x, y, alternative="two-sided")
            p_adj = min(1.0, float(p) * mtests)
            delta = cliffs_delta(x, y)
            concl = "істотна різниця " + significance_mark(p_adj) if p_adj < alpha else "-"
            sig[(a, b)] = (p_adj < alpha)
            rows.append([f"{a} vs {b}", fmt_num(float(U), 3), fmt_num(p_adj, 4), concl, fmt_num(delta, 4), cliffs_label(abs(delta))])
        except Exception:
            continue
    return rows, sig

def pairwise_wilcoxon_bonf(levels, mat_rows, alpha=ALPHA):
    # mat_rows: list of blocks, each row has len(levels) values
    rows = []
    sig = {}
    k = len(levels)
    if k < 2:
        return rows, sig
    mtests = k * (k - 1) / 2.0
    arr = np.array(mat_rows, dtype=float)
    for i in range(k):
        for j in range(i + 1, k):
            x = arr[:, i]
            y = arr[:, j]
            try:
                stat, p = wilcoxon(x, y, zero_method="wilcox", correction=False, alternative="two-sided", mode="auto")
                p_adj = min(1.0, float(p) * mtests)
                # effect r ~ z/sqrt(n)
                # approximate z from p
                if p_adj > 0 and p_adj < 1:
                    z = abs(norm.ppf(p_adj / 2.0))
                else:
                    z = 0.0
                r_eff = z / math.sqrt(len(x)) if len(x) > 0 else np.nan
                concl = "істотна різниця " + significance_mark(p_adj) if p_adj < alpha else "-"
                sig[(levels[i], levels[j])] = (p_adj < alpha)
                rows.append([f"{levels[i]} vs {levels[j]}", fmt_num(float(stat), 3), fmt_num(p_adj, 4), concl, fmt_num(r_eff, 4)])
            except Exception:
                continue
    return rows, sig


# -------------------------
# RCBD matrix builder for Friedman/Wilcoxon
# -------------------------
def rcbd_matrix_from_long(long, variant_names, block_names, variant_key="VARIANT", block_key="BLOCK"):
    # keep only full blocks without missing variants
    by_block = defaultdict(dict)
    for r in long:
        v = r.get("value", np.nan)
        if v is None or math.isnan(v):
            continue
        b = r.get(block_key)
        vname = r.get(variant_key)
        if b is None or vname is None:
            continue
        by_block[b][vname] = float(v)

    mat = []
    kept = []
    for b in block_names:
        d = by_block.get(b, {})
        ok = all(vn in d for vn in variant_names)
        if ok:
            mat.append([d[vn] for vn in variant_names])
            kept.append(b)
    return mat, kept


# -------------------------
# GLM utilities (OLS)
# -------------------------
def _encode_factor(col_values, levels):
    # returns design columns for this factor: k-1 dummies (reference = first)
    ref = levels[0]
    cols = []
    names = []
    for lvl in levels[1:]:
        x = np.array([1.0 if v == lvl else 0.0 for v in col_values], dtype=float)
        cols.append(x)
        names.append(str(lvl))
    return cols, names, ref

def _interaction_cols(base_cols, base_names):
    # base_cols: list of arrays, base_names: list of names
    cols = []
    names = []
    for i in range(len(base_cols)):
        for j in range(i + 1, len(base_cols)):
            cols.append(base_cols[i] * base_cols[j])
            names.append(f"{base_names[i]}×{base_names[j]}")
    return cols, names

def _build_full_factorial_design(long, factor_keys, levels_by_factor, extra_terms=None):
    """
    Build full factorial model matrix:
    Intercept + main effects + all interactions among factor_keys.
    extra_terms: list of (term_name, array) to append (e.g., BLOCK dummies)
    Returns: X, term_slices (name -> columns indices list), colnames
    """
    n = len(long)
    y = np.array([float(r["value"]) for r in long], dtype=float)

    X_cols = [np.ones(n, dtype=float)]
    colnames = ["Intercept"]
    term_slices = {"Intercept": [0]}

    # main effects: each factor -> (k-1) dummies
    factor_dummy_cols = {}
    factor_dummy_names = {}
    for f in factor_keys:
        vals = [r.get(f) for r in long]
        lvls = levels_by_factor[f]
        cols, names, ref = _encode_factor(vals, lvls)
        factor_dummy_cols[f] = cols
        factor_dummy_names[f] = names
        if cols:
            idxs = []
            for c, nm in zip(cols, names):
                X_cols.append(c)
                colnames.append(f"{f}:{nm}")
                idxs.append(len(X_cols) - 1)
            term_slices[f"Фактор {f}"] = idxs
        else:
            term_slices[f"Фактор {f}"] = []

    # interactions up to full order
    # build interaction dummies from existing dummy columns (not including intercept)
    # For each interaction, take products of involved factors' dummy cols.
    def _all_interactions(keys):
        out = []
        for r in range(2, len(keys) + 1):
            for comb in combinations(keys, r):
                out.append(comb)
        return out

    for comb in _all_interactions(factor_keys):
        # collect cols lists per factor
        lists = [factor_dummy_cols[f] for f in comb]
        names_lists = [factor_dummy_names[f] for f in comb]
        if any(len(L) == 0 for L in lists):
            term_slices["Фактор " + "×".join(comb)] = []
            continue
        # cartesian product
        idxs = []
        def rec_build(i, cur_col, cur_name):
            nonlocal idxs
            if i == len(lists):
                X_cols.append(cur_col)
                colnames.append("×".join([f"{comb[j]}:{cur_name[j]}" for j in range(len(comb))]))
                idxs.append(len(X_cols) - 1)
                return
            for ci, nm in zip(lists[i], names_lists[i]):
                if cur_col is None:
                    rec_build(i + 1, ci.copy(), cur_name + [nm])
                else:
                    rec_build(i + 1, cur_col * ci, cur_name + [nm])
        rec_build(0, None, [])
        term_slices["Фактор " + "×".join(comb)] = idxs

    # extra terms (e.g. BLOCK dummies)
    if extra_terms:
        for name, cols, coln in extra_terms:
            idxs = []
            for c, nm in zip(cols, coln):
                X_cols.append(c)
                colnames.append(f"{name}:{nm}")
                idxs.append(len(X_cols) - 1)
            term_slices[name] = idxs

    X = np.column_stack(X_cols)
    return y, X, term_slices, colnames

def _ols_fit(y, X):
    # returns beta, yhat, residuals, SSE, df_error, MSE
    # robust least squares via lstsq
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    sse = float(np.sum(resid ** 2))
    n, p = X.shape
    df_e = n - p
    mse = sse / df_e if df_e > 0 else np.nan
    return beta, yhat, resid, sse, df_e, mse

def _ss_terms(y, X_full, term_slices):
    # Type III-ish via drop-term (partial) SS:
    # SS_term = SSE_reduced - SSE_full, where reduced drops term columns
    beta, yhat, resid, sse_full, df_e_full, mse_full = _ols_fit(y, X_full)
    results = {}
    for term, idxs in term_slices.items():
        if term == "Intercept":
            continue
        if not idxs:
            results[term] = (np.nan, 0, np.nan, np.nan, np.nan)
            continue
        keep = [i for i in range(X_full.shape[1]) if i not in idxs]
        X_red = X_full[:, keep]
        _, _, _, sse_red, df_e_red, _ = _ols_fit(y, X_red)
        ss = float(sse_red - sse_full)
        df = len(idxs)
        ms = ss / df if df > 0 else np.nan
        F = (ms / mse_full) if (df > 0 and not math.isnan(mse_full) and mse_full > 0) else np.nan
        p = float(1.0 - f_dist.cdf(F, df, df_e_full)) if (not math.isnan(F) and df_e_full > 0) else np.nan
        results[term] = (ss, df, ms, F, p)
    return results, sse_full, df_e_full, mse_full, resid

def build_effect_strength_rows(anova_table):
    # anova_table: list rows [name, SS, df, MS, F, p]
    ss_total = 0.0
    for row in anova_table:
        name, SSv, *_ = row
        if name == "Загальна":
            ss_total = float(SSv) if not (SSv is None or (isinstance(SSv, float) and math.isnan(SSv))) else 0.0
            break
    if ss_total <= 0:
        # fallback: sum SS except residual
        ss_total = 0.0
        for row in anova_table:
            if row[0].startswith("Залишок"):
                continue
            if row[1] is None or (isinstance(row[1], float) and math.isnan(row[1])):
                continue
            ss_total += float(row[1])

    out = []
    for row in anova_table:
        name, SSv, *_ = row
        if name.startswith("Залишок") or name == "Загальна":
            continue
        if SSv is None or (isinstance(SSv, float) and math.isnan(SSv)):
            continue
        pct = (float(SSv) / ss_total * 100.0) if ss_total > 0 else np.nan
        out.append([name, fmt_num(pct, 2)])
    return out

def build_partial_eta2_rows_with_label(anova_table):
    # partial eta2 = SS_effect / (SS_effect + SS_error)
    ss_err = None
    for row in anova_table:
        if row[0].startswith("Залишок"):
            ss_err = row[1]
            break
    if ss_err is None or (isinstance(ss_err, float) and math.isnan(ss_err)):
        ss_err = np.nan
    out = []
    for row in anova_table:
        name, SSv, *_ = row
        if name.startswith("Залишок") or name == "Загальна":
            continue
        if SSv is None or (isinstance(SSv, float) and math.isnan(SSv)) or (isinstance(ss_err, float) and math.isnan(ss_err)):
            continue
        pe2 = float(SSv) / (float(SSv) + float(ss_err)) if (float(SSv) + float(ss_err)) > 0 else np.nan
        out.append([name, fmt_num(pe2, 4), partial_eta2_label(pe2)])
    return out


# -------------------------
# CRD / RCBD / Split-plot analyses
# -------------------------
def anova_n_way(long, factor_keys, levels_by_factor):
    # CRD: full factorial GLM
    y, X, term_slices, colnames = _build_full_factorial_design(long, factor_keys, levels_by_factor)
    terms, sse, df_e, mse, resid = _ss_terms(y, X, term_slices)

    ss_total = float(np.sum((y - np.mean(y)) ** 2))
    table = []
    # print in logical order: main effects, interactions, residual, total
    ordered = []
    for f in factor_keys:
        ordered.append(f"Фактор {f}")
    for r in range(2, len(factor_keys) + 1):
        for comb in combinations(factor_keys, r):
            ordered.append("Фактор " + "×".join(comb))

    for name in ordered:
        ss, df, ms, F, p = terms.get(name, (np.nan, 0, np.nan, np.nan, np.nan))
        table.append([name, ss, df, ms, F, p])

    table.append(["Залишок", sse, df_e, mse, np.nan, np.nan])
    table.append(["Загальна", ss_total, len(y) - 1, np.nan, np.nan, np.nan])

    # cell means for residuals quick use
    cell = defaultdict(list)
    for r in long:
        key = tuple(r.get(f) for f in factor_keys)
        cell[key].append(float(r["value"]))
    cell_means = {k: float(np.mean(v)) for k, v in cell.items()}

    # NIR05 for factors + overall variants (used in report)
    NIR05 = {}
    if not (math.isnan(mse) or df_e <= 0):
        tcrit = float(t_dist.ppf(1 - ALPHA / 2.0, int(df_e)))
        # for each factor: approximate LSD using n per level (harmonic mean not needed here)
        for f in factor_keys:
            # n for each level
            n_level = defaultdict(int)
            for r in long:
                if r.get(f) is None:
                    continue
                if r.get("value") is None or math.isnan(r.get("value", np.nan)):
                    continue
                n_level[r.get(f)] += 1
            # use harmonic mean of ns
            ns = [n for n in n_level.values() if n > 0]
            if ns:
                nh = len(ns) / sum(1.0 / n for n in ns)
                NIR05[f"Фактор {f}"] = tcrit * math.sqrt(2.0 * mse / nh)
        # overall variants (cell means): use harmonic n per cell
        n_cell = defaultdict(int)
        for r in long:
            key = tuple(r.get(f) for f in factor_keys)
            if r.get("value") is None or math.isnan(r.get("value", np.nan)):
                continue
            n_cell[key] += 1
        ns = [n for n in n_cell.values() if n > 0]
        if ns:
            nh = len(ns) / sum(1.0 / n for n in ns)
            NIR05["Загальна"] = tcrit * math.sqrt(2.0 * mse / nh)

    return {
        "table": table,
        "SS_error": sse,
        "df_error": df_e,
        "MS_error": mse,
        "SS_total": ss_total,
        "cell_means": cell_means,
        "residuals": resid.tolist(),
        "NIR05": NIR05,
    }

def _block_dummies(long, block_key="BLOCK"):
    blocks = first_seen_order([r.get(block_key) for r in long if r.get(block_key) is not None])
    if not blocks:
        return [], [], blocks
    ref = blocks[0]
    cols = []
    names = []
    vals = [r.get(block_key) for r in long]
    for b in blocks[1:]:
        cols.append(np.array([1.0 if v == b else 0.0 for v in vals], dtype=float))
        names.append(str(b))
    return cols, names, blocks

def anova_rcbd_ols(long, factor_keys, levels_by_factor, block_key="BLOCK"):
    # RCBD: full factorial + block main effect
    bcols, bnames, blocks = _block_dummies(long, block_key=block_key)
    extra = [("Блоки", bcols, bnames)] if bcols else []
    y, X, term_slices, colnames = _build_full_factorial_design(long, factor_keys, levels_by_factor, extra_terms=extra)
    terms, sse, df_e, mse, resid = _ss_terms(y, X, term_slices)

    ss_total = float(np.sum((y - np.mean(y)) ** 2))
    table = []

    if bcols:
        ss, df, ms, F, p = terms.get("Блоки", (np.nan, 0, np.nan, np.nan, np.nan))
        table.append(["Блоки", ss, df, ms, F, p])

    ordered = []
    for f in factor_keys:
        ordered.append(f"Фактор {f}")
    for r in range(2, len(factor_keys) + 1):
        for comb in combinations(factor_keys, r):
            ordered.append("Фактор " + "×".join(comb))
    for name in ordered:
        ss, df, ms, F, p = terms.get(name, (np.nan, 0, np.nan, np.nan, np.nan))
        table.append([name, ss, df, ms, F, p])

    table.append(["Залишок", sse, df_e, mse, np.nan, np.nan])
    table.append(["Загальна", ss_total, len(y) - 1, np.nan, np.nan, np.nan])

    NIR05 = {}
    if not (math.isnan(mse) or df_e <= 0):
        tcrit = float(t_dist.ppf(1 - ALPHA / 2.0, int(df_e)))
        # for each factor
        for f in factor_keys:
            n_level = defaultdict(int)
            for r in long:
                if r.get(f) is None:
                    continue
                if r.get("value") is None or math.isnan(r.get("value", np.nan)):
                    continue
                n_level[r.get(f)] += 1
            ns = [n for n in n_level.values() if n > 0]
            if ns:
                nh = len(ns) / sum(1.0 / n for n in ns)
                NIR05[f"Фактор {f}"] = tcrit * math.sqrt(2.0 * mse / nh)

        # overall variants (cell means): harmonic
        n_cell = defaultdict(int)
        for r in long:
            key = tuple(r.get(f) for f in factor_keys)
            if r.get("value") is None or math.isnan(r.get("value", np.nan)):
                continue
            n_cell[key] += 1
        ns = [n for n in n_cell.values() if n > 0]
        if ns:
            nh = len(ns) / sum(1.0 / n for n in ns)
            NIR05["Загальна"] = tcrit * math.sqrt(2.0 * mse / nh)

    return {
        "table": table,
        "SS_error": sse,
        "df_error": df_e,
        "MS_error": mse,
        "SS_total": ss_total,
        "residuals": resid.tolist(),
        "NIR05": NIR05,
    }

def anova_splitplot_ols(long, factor_keys, main_factor="A", block_key="BLOCK"):
    """
    Split-plot:
      Whole-plot error term: BLOCK × main_factor interaction (and its MS/df)
      Sub-plot error term: residual
    We fit full model including:
      - blocks
      - main_factor
      - block×main_factor (whole-plot error)
      - other factors and interactions
    For reporting:
      - Effects of main_factor tested against MS_whole / df_whole
      - All other effects tested against MS_error / df_error
    """
    if main_factor not in factor_keys:
        main_factor = factor_keys[0]

    # Build blocks dummies (k-1)
    bcols, bnames, blocks = _block_dummies(long, block_key=block_key)

    # Build dummies for main factor and for blocks×main dummies
    # We'll create explicit columns for block×main interaction using products of their dummies.
    main_levels = first_seen_order([r.get(main_factor) for r in long if r.get(main_factor) is not None])
    if len(main_levels) < 2:
        raise ValueError("Split-plot: головний фактор має мати щонайменше 2 рівні.")

    main_vals = [r.get(main_factor) for r in long]
    main_cols, main_names, main_ref = _encode_factor(main_vals, main_levels)

    # interaction block×main:
    wp_cols = []
    wp_names = []
    # use non-ref block dummies and non-ref main dummies
    for bi, bcol in enumerate(bcols):
        for mi, mcol in enumerate(main_cols):
            wp_cols.append(bcol * mcol)
            wp_names.append(f"{bnames[bi]}×{main_names[mi]}")
    # extra terms: blocks main effect + wp error
    extra = []
    if bcols:
        extra.append(("Блоки", bcols, bnames))
    # add main factor explicitly as "Фактор A" columns will be built by generic builder; we keep it there.
    if wp_cols:
        extra.append((f"Whole-plot error (Блоки×{main_factor})", wp_cols, wp_names))

    # Build full factorial with extra terms
    levels_by_factor = {f: first_seen_order([r.get(f) for r in long if r.get(f) is not None]) for f in factor_keys}
    y, X, term_slices, colnames = _build_full_factorial_design(long, factor_keys, levels_by_factor, extra_terms=extra)

    # Fit full model
    beta, yhat, resid, sse_full, df_e, mse = _ols_fit(y, X)

    # Determine SS and MS for whole-plot error term:
    wp_term = f"Whole-plot error (Блоки×{main_factor})"
    wp_idxs = term_slices.get(wp_term, [])
    if not wp_idxs:
        # If blocks or main factor missing, cannot estimate properly
        raise ValueError("Split-plot: неможливо сформувати whole-plot error (перевірте блоки/головний фактор).")

    # SS for wp term (partial): drop wp cols
    keep = [i for i in range(X.shape[1]) if i not in wp_idxs]
    X_red = X[:, keep]
    _, _, _, sse_red, df_e_red, _ = _ols_fit(y, X_red)
    ss_wp = float(sse_red - sse_full)
    df_wp = len(wp_idxs)
    ms_wp = ss_wp / df_wp if df_wp > 0 else np.nan

    ss_total = float(np.sum((y - np.mean(y)) ** 2))

    # Build ANOVA-like table with correct F for main factor
    # We still compute partial SS for all terms (excluding wp, blocks) to show in table,
    # but F/p for main factor uses ms_wp instead of mse.
    terms, _, _, _, _ = _ss_terms(y, X, term_slices)

    table = []
    # blocks (optional)
    if bcols:
        ss, df, ms, F, p = terms.get("Блоки", (np.nan, 0, np.nan, np.nan, np.nan))
        table.append(["Блоки", ss, df, ms, np.nan, np.nan])

    # whole-plot error row
    table.append([wp_term, ss_wp, df_wp, ms_wp, np.nan, np.nan])

    # main & others
    ordered = []
    for f in factor_keys:
        ordered.append(f"Фактор {f}")
    for rnk in range(2, len(factor_keys) + 1):
        for comb in combinations(factor_keys, rnk):
            ordered.append("Фактор " + "×".join(comb))

    for name in ordered:
        ss, df, ms, F, p = terms.get(name, (np.nan, 0, np.nan, np.nan, np.nan))
        if name == f"Фактор {main_factor}":
            # test against ms_wp, df_wp
            if df > 0 and not any(math.isnan(x) for x in [ms, ms_wp]) and ms_wp > 0 and df_wp > 0:
                Fm = ms / ms_wp
                pm = float(1.0 - f_dist.cdf(Fm, df, df_wp))
            else:
                Fm, pm = (np.nan, np.nan)
            table.append([name, ss, df, ms, Fm, pm])
        else:
            # test against residual
            if df > 0 and not any(math.isnan(x) for x in [ms, mse]) and mse > 0 and df_e > 0:
                Fm = ms / mse
                pm = float(1.0 - f_dist.cdf(Fm, df, df_e))
            else:
                Fm, pm = (np.nan, np.nan)
            table.append([name, ss, df, ms, Fm, pm])

    table.append(["Залишок", sse_full, df_e, mse, np.nan, np.nan])
    table.append(["Загальна", ss_total, len(y) - 1, np.nan, np.nan, np.nan])

    # NIR05: provide per factor, but for main factor use ms_wp/df_wp, others use mse/df_e
    NIR05 = {}
    # counts per level
    def _harm_n_for_factor(f):
        n_level = defaultdict(int)
        for r in long:
            if r.get(f) is None:
                continue
            if r.get("value") is None or math.isnan(r.get("value", np.nan)):
                continue
            n_level[r.get(f)] += 1
        ns = [n for n in n_level.values() if n > 0]
        if not ns:
            return np.nan
        nh = len(ns) / sum(1.0 / n for n in ns)
        return float(nh)

    if not (math.isnan(mse) or df_e <= 0 or math.isnan(ms_wp) or df_wp <= 0):
        tcrit_sub = float(t_dist.ppf(1 - ALPHA / 2.0, int(df_e))) if df_e > 0 else np.nan
        tcrit_whole = float(t_dist.ppf(1 - ALPHA / 2.0, int(df_wp))) if df_wp > 0 else np.nan
        for f in factor_keys:
            nh = _harm_n_for_factor(f)
            if math.isnan(nh) or nh <= 0:
                continue
            if f == main_factor:
                NIR05[f"Фактор {f} (whole-plot)"] = tcrit_whole * math.sqrt(2.0 * ms_wp / nh)
            else:
                NIR05[f"Фактор {f}"] = tcrit_sub * math.sqrt(2.0 * mse / nh)

    return {
        "table": table,
        "SS_error": sse_full,
        "df_error": df_e,
        "MS_error": mse,
        "SS_total": ss_total,
        "residuals": resid.tolist(),
        "MS_whole": ms_wp,
        "df_whole": df_wp,
        "main_factor": main_factor,
        "NIR05": NIR05,
    }




# -------------------------
# Repeated Measures (balanced) ANOVA via OLS (mixed: between factors + within TIME)
# -------------------------
def anova_repeated_measures_ols(long, factor_keys, time_key="TIME", subject_key="SUBJECT"):
    """
    Balanced repeated-measures ANOVA (classic mixed ANOVA) using OLS with explicit:
      - between-subject effects: factor_keys and their interactions
      - within-subject effect: TIME
      - subject effects (subjects nested in between-cells)
      - subject×TIME (within-subject error term)

    Assumptions:
      • Data are balanced: each SUBJECT has measurements for all TIME levels.
      • One observation per SUBJECT×TIME.
      • factor_keys are between-subject factors (do not vary within SUBJECT).

    Testing:
      • Between-subject effects → MS_effect / MS_subject
      • TIME and any interactions with TIME → MS_effect / MS_subject×TIME
    """
    if not long:
        return None

    # levels
    time_levels = first_seen_order([r.get(time_key) for r in long if r.get(time_key) is not None])
    subj_levels = first_seen_order([r.get(subject_key) for r in long if r.get(subject_key) is not None])
    if len(time_levels) < 2:
        raise ValueError("Повторні вимірювання: потрібно щонайменше 2 рівні часу (рік/фаза).")
    if len(subj_levels) < 2:
        raise ValueError("Повторні вимірювання: потрібно щонайменше 2 суб'єкти (повторності).")

    # quick balance check: each subject must have all time levels
    by_subj = defaultdict(set)
    for r in long:
        by_subj[r.get(subject_key)].add(r.get(time_key))
    bad = [s for s, st in by_subj.items() if len(st) != len(time_levels)]
    if bad:
        raise ValueError(
            "Повторні вимірювання: дані небалансовані (не всі суб'єкти мають значення для кожного року/фази). "
            "Для цієї версії потрібна повна матриця вимірювань."
        )

    # build base factorial for between factors + TIME as a factor
    all_keys = list(factor_keys) + [time_key]
    levels_by_factor = {k: first_seen_order([r.get(k) for r in long if r.get(k) is not None]) for k in all_keys}

    # subject dummies (k-1) and subject×time dummies (subject_dummies × time_dummies)
    subj_vals = [r.get(subject_key) for r in long]
    subj_cols, subj_names, subj_ref = _encode_factor(subj_vals, subj_levels)

    time_vals = [r.get(time_key) for r in long]
    time_cols, time_names, time_ref = _encode_factor(time_vals, time_levels)

    st_cols = []
    st_names = []
    for si, scol in enumerate(subj_cols):
        for ti, tcol in enumerate(time_cols):
            st_cols.append(scol * tcol)
            st_names.append(f"{subj_names[si]}×{time_names[ti]}")

    extra = []
    if subj_cols:
        extra.append(("Суб'єкти", subj_cols, subj_names))
    if st_cols:
        extra.append((f"Суб'єкти×{time_key}", st_cols, st_names))

    y, X, term_slices, colnames = _build_full_factorial_design(long, all_keys, levels_by_factor, extra_terms=extra)

    # full model fit
    beta, yhat, resid_full, sse_full, df_e_full, mse_full = _ols_fit(y, X)
    # partial SS for terms
    terms, _, _, _, _ = _ss_terms(y, X, term_slices)

    # pull subject and subject×time as error terms
    ss_subj, df_subj, ms_subj, _, _ = terms.get("Суб'єкти", (np.nan, 0, np.nan, np.nan, np.nan))
    ss_st, df_st, ms_st, _, _ = terms.get(f"Суб'єкти×{time_key}", (np.nan, 0, np.nan, np.nan, np.nan))

    if df_subj <= 0 or df_st <= 0 or any(math.isnan(x) for x in [ms_subj, ms_st]) or ms_subj <= 0 or ms_st <= 0:
        raise ValueError("Повторні вимірювання: неможливо оцінити помилки для тестів (перевірте структуру даних).")

    # build ANOVA-like table in logical order:
    table = []

    # Between-subject effects: main + interactions among factor_keys
    ordered_between = []
    for f in factor_keys:
        ordered_between.append(f"Фактор {f}")
    for rnk in range(2, len(factor_keys) + 1):
        for comb in combinations(factor_keys, rnk):
            ordered_between.append("Фактор " + "×".join(comb))

    # TIME and TIME interactions
    ordered_time = [f"Фактор {time_key}"]
    for f in factor_keys:
        ordered_time.append("Фактор " + "×".join([f, time_key]))
    for rnk in range(2, len(factor_keys) + 1):
        for comb in combinations(factor_keys, rnk):
            ordered_time.append("Фактор " + "×".join(list(comb) + [time_key]))

    # Between rows with F vs MS_subject
    for name in ordered_between:
        ss, df, ms, _, _ = terms.get(name, (np.nan, 0, np.nan, np.nan, np.nan))
        if df > 0 and not any(math.isnan(x) for x in [ms, ms_subj]) and ms_subj > 0:
            Fv = ms / ms_subj
            pv = float(1.0 - f_dist.cdf(Fv, df, df_subj))
        else:
            Fv, pv = (np.nan, np.nan)
        table.append([name, ss, df, ms, Fv, pv])

    # subject error term for between
    table.append(["Помилка між суб'єктами (Суб'єкти)", ss_subj, df_subj, ms_subj, np.nan, np.nan])

    # TIME rows with F vs MS_subject×TIME
    for name in ordered_time:
        ss, df, ms, _, _ = terms.get(name, (np.nan, 0, np.nan, np.nan, np.nan))
        if df > 0 and not any(math.isnan(x) for x in [ms, ms_st]) and ms_st > 0:
            Fv = ms / ms_st
            pv = float(1.0 - f_dist.cdf(Fv, df, df_st))
        else:
            Fv, pv = (np.nan, np.nan)
        table.append([name, ss, df, ms, Fv, pv])

    # within-subject error
    table.append([f"Помилка в межах суб'єктів (Суб'єкти×{time_key})", ss_st, df_st, ms_st, np.nan, np.nan])

    ss_total = float(np.sum((y - np.mean(y)) ** 2))
    table.append(["Загальна", ss_total, len(y) - 1, np.nan, np.nan, np.nan])

    # NIR05: between factors use ms_subj/df_subj; TIME uses ms_st/df_st
    NIR05 = {}
    tcrit_between = float(t_dist.ppf(1 - ALPHA / 2.0, int(df_subj)))
    tcrit_within = float(t_dist.ppf(1 - ALPHA / 2.0, int(df_st)))

    # harmonic n per level for between factors (count of SUBJECTs in each level)
    # estimate subjects per level from long (unique subjects)
    subj_to_row = {}
    for r in long:
        subj_to_row.setdefault(r.get(subject_key), r)

    subj_records = list(subj_to_row.values())

    def _harm_n_subjects_for_factor(f):
        n_level = defaultdict(int)
        for r in subj_records:
            if r.get(f) is None:
                continue
            n_level[r.get(f)] += 1
        ns = [n for n in n_level.values() if n > 0]
        if not ns:
            return np.nan
        return float(len(ns) / sum(1.0 / n for n in ns))

    for f in factor_keys:
        nh = _harm_n_subjects_for_factor(f)
        if math.isnan(nh) or nh <= 0:
            continue
        NIR05[f"Фактор {f}"] = tcrit_between * math.sqrt(2.0 * ms_subj / nh)

    # TIME LSD (subjects count is denominator)
    n_subj = len(subj_levels)
    if n_subj > 0:
        NIR05[f"Фактор {time_key}"] = tcrit_within * math.sqrt(2.0 * ms_st / n_subj)

    # residuals for normality: fit model WITHOUT subject×time (so residual captures within-subject variation)
    extra2 = []
    if subj_cols:
        extra2.append(("Суб'єкти", subj_cols, subj_names))
    y2, X2, ts2, _ = _build_full_factorial_design(long, all_keys, levels_by_factor, extra_terms=extra2)
    _, _, resid2, _, _, _ = _ols_fit(y2, X2)

    return {
        "table": table,
        "df_between_err": df_subj,
        "MS_between_err": ms_subj,
        "df_within_err": df_st,
        "MS_within_err": ms_st,
        "SS_total": ss_total,
        "residuals": resid2.tolist(),
        "NIR05": NIR05,
        "time_key": time_key,
    }
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
                  command=lambda: self.start_analysis(1)).grid(row=0, column=0, padx=10, pady=8)
        tk.Button(btn_frame, text="Двофакторний аналіз", width=22, height=2,
                  command=lambda: self.start_analysis(2)).grid(row=0, column=1, padx=10, pady=8)
        tk.Button(btn_frame, text="Трифакторний аналіз", width=22, height=2,
                  command=lambda: self.start_analysis(3)).grid(row=1, column=0, padx=10, pady=8)
        tk.Button(btn_frame, text="Чотирифакторний аналіз", width=22, height=2,
                  command=lambda: self.start_analysis(4)).grid(row=1, column=1, padx=10, pady=8)

        tk.Label(
            self.main_frame,
            text="Виберіть тип аналізу → Внесіть дані → Натисніть «Аналіз даних»",
            fg="#000000",
            bg="white"
        ).pack(pady=10)

        self.table_win = None
        self.report_win = None
        self.graph_win = None
        self._graph_fig = None
        self._graph_canvas = None
        self.params_current = None

        # active cell
        self._active_cell = None
        self._active_prev_cfg = None

        # fill-handle drag (one cell down like Excel)
        self._fill_ready = False
        self._fill_dragging = False
        self._fill_src_pos = None
        self._fill_src_text = ""
        self._fill_last_row = None

        self.factor_title_map = {}

    # ---------- Factor titles ----------
    def factor_title(self, fkey: str) -> str:
        return self.factor_title_map.get(fkey, f"Фактор {fkey}")

    def _set_factor_title(self, fkey: str, title: str):
        self.factor_title_map[fkey] = title.strip() if title else f"Фактор {fkey}"

    # ---------- Design help ----------
    def show_design_help(self):
        w = tk.Toplevel(self.root)
        w.title("Пояснення дизайнів")
        w.resizable(False, False)
        set_window_icon(w)

        frm = tk.Frame(w, padx=16, pady=14)
        frm.pack(fill=tk.BOTH, expand=True)

        txt = (
            "CRD (повна рандомізація)\n"
            "• Усі варіанти розміщені випадково без блоків.\n"
            "• Підходить, коли ділянка однорідна.\n\n"
            "RCBD (блочна рандомізація)\n"
            "• Є блоки (повторності), всередині блоку — всі варіанти.\n"
            "• Підходить, коли є градієнт поля (рельєф, ґрунт тощо).\n\n"
            "Split-plot (спліт-плот)\n"
            "• Є головний фактор (whole-plot) і підплощі (sub-plot) всередині.\n"
            "• Для головного фактора використовується інша (більша) помилка, ніж для підфакторів.\n"
        )

        t = tk.Text(frm, width=62, height=16, wrap="word")
        t.insert("1.0", txt)
        t.configure(state="disabled")
        t.pack(fill=tk.BOTH, expand=True)

        btns = tk.Frame(frm)
        btns.pack(fill=tk.X, pady=(10, 0))
        tk.Button(btns, text="OK", width=10, command=w.destroy).pack(side=tk.LEFT, padx=6)

        w.update_idletasks()
        center_window(w)
        w.grab_set()

    # ---------- ask indicator / units / design ----------
    
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

        row_design = tk.Frame(frm)
        row_design.grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 4))
        tk.Label(row_design, text="Дизайн експерименту:", fg="#000000").pack(side=tk.LEFT)
        tk.Button(row_design, text=" ? ", width=3, command=self.show_design_help).pack(side=tk.LEFT, padx=(8, 0))

        design_var = tk.StringVar(value="crd")
        rfrm = tk.Frame(frm)
        rfrm.grid(row=2, column=1, sticky="w", pady=(10, 4), padx=(220, 0))
        rb_font = ("Times New Roman", 16)

        tk.Radiobutton(rfrm, text="Повна рандомізація (CRD)", variable=design_var, value="crd",
                       font=rb_font).pack(anchor="w", pady=2)
        tk.Radiobutton(rfrm, text="Блочна рандомізація (RCBD)", variable=design_var, value="rcbd",
                       font=rb_font).pack(anchor="w", pady=2)
        tk.Radiobutton(rfrm, text="Спліт-плот (Split-plot) — лише параметричний", variable=design_var, value="split",
                       font=rb_font).pack(anchor="w", pady=2)

        # repeated measures
        rm_var = tk.BooleanVar(value=False)
        tk.Checkbutton(frm, text="Повторні вимірювання (одні й ті самі об'єкти у часі)",
                       variable=rm_var, onvalue=True, offvalue=False,
                       font=("Times New Roman", 14), fg="#000000").grid(row=3, column=0, columnspan=2,
                                                                        sticky="w", pady=(10, 2))

        tk.Label(frm, text="Назва повторюваного чинника (напр.: Рік або Фаза):", fg="#000000").grid(
            row=4, column=0, sticky="w", pady=(6, 2)
        )
        e_time = tk.Entry(frm, width=40, fg="#000000")
        e_time.grid(row=4, column=1, pady=(6, 2))
        e_time.insert(0, "Рік")

        def _toggle_time():
            st = "normal" if rm_var.get() else "disabled"
            e_time.configure(state=st)

        rm_var.trace_add("write", lambda *_: _toggle_time())
        _toggle_time()

        main_factor_var = tk.StringVar(value="A")
        main_factor_frame = tk.Frame(frm)
        main_factor_frame.grid(row=5, column=0, columnspan=2, sticky="w", pady=(12, 4))
        lbl_main = tk.Label(main_factor_frame, text="Головний фактор (для спліт-плот):", fg="#000000")
        lbl_main.pack(side=tk.LEFT)
        cb_main = ttk.Combobox(main_factor_frame, textvariable=main_factor_var,
                               values=["A", "B", "C", "D"], state="readonly", width=6)
        cb_main.pack(side=tk.LEFT, padx=(10, 0))

        def update_main_visibility(*_):
            if design_var.get() == "split":
                lbl_main.configure(state="normal")
                cb_main.configure(state="readonly")
            else:
                lbl_main.configure(state="disabled")
                cb_main.configure(state="disabled")

        design_var.trace_add("write", update_main_visibility)
        update_main_visibility()

        out = {"ok": False, "indicator": "", "units": "", "design": "crd", "split_main": "A",
               "rm": False, "rm_label": "Рік"}

        def ok():
            ind = e_ind.get().strip() or "Показник"
            units = e_units.get().strip()
            design = design_var.get()
            rm = bool(rm_var.get())
            rm_label = e_time.get().strip() or "Рік"

            if rm and design != "crd":
                messagebox.showwarning(
                    "Повторні вимірювання",
                    "У цій версії повторні вимірювання підтримуються лише для повної рандомізації (CRD).\n"
                    "Для RCBD/Split-plot потрібна окрема (складніша) модель."
                )
                return
            if rm and not rm_label:
                messagebox.showwarning("Повторні вимірювання", "Вкажіть назву повторюваного чинника (Рік/Фаза).")
                return

            out.update({
                "ok": True,
                "indicator": ind,
                "units": units,
                "design": design,
                "split_main": main_factor_var.get(),
                "rm": rm,
                "rm_label": rm_label
            })
            dlg.destroy()

        def cancel():
            dlg.destroy()

        btns = tk.Frame(frm)
        btns.grid(row=6, column=0, columnspan=2, pady=(14, 0))
        tk.Button(btns, text="OK", width=10, command=ok).pack(side=tk.LEFT, padx=8)
        tk.Button(btns, text="Скасувати", width=10, command=cancel).pack(side=tk.LEFT, padx=8)

        dlg.grab_set()
        dlg.transient(self.root)
        self.root.wait_window(dlg)
        return out

    def choose_method_window(self, p_norm, design, num_variants):
        dlg = tk.Toplevel(self.root)
        dlg.title("Вибір виду аналізу")
        dlg.resizable(False, False)
        set_window_icon(dlg)

        frm = tk.Frame(dlg, padx=16, pady=14)
        frm.pack(fill=tk.BOTH, expand=True)

        normal = (p_norm is not None) and (not math.isnan(p_norm)) and (p_norm > 0.05)
        rb_font = ("Times New Roman", 16)

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
            if design == "split":
                tk.Label(
                    frm,
                    text=("Для спліт-плоту в цій програмі реалізовано лише параметричний аналіз.\n"
                          "Оскільки залишки моделі не відповідають нормальному розподілу (p ≤ 0.05),\n"
                          "параметричний split-plot аналіз є методично некоректним.\n\n"
                          "Що можна зробити:\n"
                          "• спробувати трансформацію даних (log/√/Box-Cox) і повторити;\n"
                          "• або вибрати CRD/RCBD і виконати непараметричний аналіз."),
                    fg="#c62828",
                    justify="left"
                ).pack(anchor="w", pady=(0, 10))
                options = []
            else:
                msg = ("Дані експерименту не відповідають принципам нормального розподілу\n"
                       "за методом Шапіра-Вілка.\n"
                       "Виберіть один з непараметричних типів аналізу.")
                tk.Label(frm, text=msg, fg="#c62828", justify="left").pack(anchor="w", pady=(0, 10))

                if design == "crd":
                    options = [
                        ("Краскела–Уолліса", "kw"),
                        ("Манна-Уітні", "mw"),
                    ]
                else:
                    if num_variants == 2:
                        options = [("Wilcoxon (парний)", "wilcoxon")]
                    else:
                        options = [("Friedman", "friedman")]

        out = {"ok": False, "method": None}

        if not options:
            def on_ok_close():
                dlg.destroy()
            btns = tk.Frame(frm)
            btns.pack(fill=tk.X, pady=(12, 0))
            tk.Button(btns, text="OK", width=10, command=on_ok_close).pack(side=tk.LEFT, padx=6)
            dlg.update_idletasks()
            center_window(dlg)
            dlg.bind("<Return>", lambda e: on_ok_close())
            dlg.grab_set()
            self.root.wait_window(dlg)
            return out

        var = tk.StringVar(value=options[0][1])
        for text, val in options:
            tk.Radiobutton(frm, text=text, variable=var, value=val, font=rb_font).pack(anchor="w", pady=3)

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

    # ---------- About ----------
    def show_about(self):
        messagebox.showinfo(
            "Розробник",
            "S.A.D. — Статистичний аналіз даних\n"
            "Версія: 1.0\n"
            "Розробик: Чаплоуцький Андрій Миколайович\n"
            "Уманський національний університет"
        )

    # ---------- Active cell highlight ----------
    def _set_active_cell(self, widget: tk.Entry):
        if self._active_cell is widget:
            return
        if isinstance(self._active_cell, tk.Entry) and self._active_prev_cfg:
            try:
                self._active_cell.configure(**self._active_prev_cfg)
            except Exception:
                pass
        self._active_cell = widget
        if isinstance(widget, tk.Entry):
            self._active_prev_cfg = {
                "bg": widget.cget("bg"),
                "highlightthickness": int(widget.cget("highlightthickness")),
                "highlightbackground": widget.cget("highlightbackground"),
                "highlightcolor": widget.cget("highlightcolor"),
                "relief": widget.cget("relief"),
                "bd": int(widget.cget("bd")) if str(widget.cget("bd")).isdigit() else 1,
            }
            try:
                widget.configure(
                    bg="#fff3c4",
                    highlightthickness=3,
                    highlightbackground="#c62828",
                    highlightcolor="#c62828",
                    relief=tk.SOLID,
                    bd=1
                )
            except Exception:
                pass

    # ---------- Fill-handle (Excel-like) ----------
    def _near_bottom_right(self, w: tk.Entry, margin=6) -> bool:
        try:
            px = w.winfo_pointerx()
            py = w.winfo_pointery()
            x0 = w.winfo_rootx()
            y0 = w.winfo_rooty()
            ww = w.winfo_width()
            hh = w.winfo_height()
            return (x0 + ww - margin <= px <= x0 + ww) and (y0 + hh - margin <= py <= y0 + hh)
        except Exception:
            return False

    def _fill_update_cursor(self, event):
        w = event.widget
        if not isinstance(w, tk.Entry):
            return
        pos = self.find_pos(w)
        if not pos:
            return
        r, c = pos
        # fill handle дозволяємо ТІЛЬКИ для факторних колонок (щоб не зіпсувати числа повторностей)
        if c >= (self.factors_count + (1 if getattr(self,'rm_enabled',False) else 0)):
            w.configure(cursor="")
            self._fill_ready = False
            return

        if self._near_bottom_right(w):
            w.configure(cursor="crosshair")
            self._fill_ready = True
        else:
            w.configure(cursor="")
            self._fill_ready = False

    def _fill_leave(self, event):
        w = event.widget
        if isinstance(w, tk.Entry):
            w.configure(cursor="")
        self._fill_ready = False

    def _fill_press(self, event):
        w = event.widget
        if not isinstance(w, tk.Entry):
            return
        pos = self.find_pos(w)
        if not pos:
            return
        r, c = pos
        if c >= (self.factors_count + (1 if getattr(self,'rm_enabled',False) else 0)):
            self._fill_dragging = False
            return

        if self._near_bottom_right(w):
            self._fill_dragging = True
            self._fill_src_pos = (r, c)
            self._fill_src_text = w.get()
            self._fill_last_row = r
            return "break"
        self._fill_dragging = False
        return None

    def _fill_drag(self, event):
        if not self._fill_dragging or not self._fill_src_pos:
            return
        w = event.widget
        if not isinstance(w, tk.Entry):
            return

        src_r, src_c = self._fill_src_pos

        try:
            py = w.winfo_pointery()
            target_r = src_r
            for rr in range(src_r, len(self.entries)):
                cell = self.entries[rr][src_c]
                y0 = cell.winfo_rooty()
                y1 = y0 + cell.winfo_height()
                if y0 <= py <= y1:
                    target_r = rr
                    break
        except Exception:
            return

        if target_r <= src_r:
            return

        for rr in range(src_r + 1, target_r + 1):
            while rr >= len(self.entries):
                self.add_row()
            cell = self.entries[rr][src_c]
            cell.delete(0, tk.END)
            cell.insert(0, self._fill_src_text)

        self._fill_last_row = target_r
        return "break"

    def _fill_release(self, event):
        if self._fill_dragging:
            self._fill_dragging = False
            self._fill_src_pos = None
            self._fill_src_text = ""
            self._fill_last_row = None
            return "break"
        return None

    # ---------- Bind cell ----------
    def bind_cell(self, e: tk.Entry):
        e.bind("<Return>", self.on_enter)
        e.bind("<Up>", self.on_arrow)
        e.bind("<Down>", self.on_arrow)
        e.bind("<Left>", self.on_arrow)
        e.bind("<Right>", self.on_arrow)
        e.bind("<Control-v>", self.on_paste)
        e.bind("<Control-V>", self.on_paste)

        e.bind("<FocusIn>", lambda ev: self._set_active_cell(ev.widget))

        # fill handle
        e.bind("<Motion>", self._fill_update_cursor)
        e.bind("<Leave>", self._fill_leave)
        e.bind("<ButtonPress-1>", self._fill_press)
        e.bind("<B1-Motion>", self._fill_drag)
        e.bind("<ButtonRelease-1>", self._fill_release)

    # ---------- Rename factor ----------
    def rename_factor_by_col(self, col_idx: int):
        off = 1 if getattr(self, "rm_enabled", False) else 0
        if col_idx < 0:
            return
        # якщо перша колонка — це Рік/Фаза (повторні вимірювання), перейменовувати її не потрібно
        if off == 1 and col_idx == 0:
            return
        if col_idx < off or col_idx >= off + self.factors_count:
            return
        fkey = self.factor_keys[col_idx - off]
        old = self.factor_title(fkey)

        dlg = tk.Toplevel(self.table_win if self.table_win else self.root)
        dlg.title("Перейменування фактора")
        dlg.resizable(False, False)
        set_window_icon(dlg)

        frm = tk.Frame(dlg, padx=14, pady=12)
        frm.pack(fill=tk.BOTH, expand=True)

        tk.Label(frm, text=f"Нова назва для {fkey}:", fg="#000000").grid(row=0, column=0, sticky="w")
        e = tk.Entry(frm, width=36, fg="#000000")
        e.grid(row=1, column=0, pady=(6, 0))
        e.insert(0, old)
        e.select_range(0, "end")
        e.focus_set()

        def ok():
            new = e.get().strip()
            if not new:
                messagebox.showwarning("Помилка", "Назва не може бути порожньою.")
                return
            self._set_factor_title(fkey, new)

            if self.header_labels and col_idx < len(self.header_labels):
                self.header_labels[col_idx].configure(text=new)

            self.factor_names = [self.factor_title(fk) for fk in self.factor_keys]
            off = 1 if getattr(self,'rm_enabled',False) else 0
            rep_names = [self.header_labels[j].cget("text") for j in range(off + self.factors_count, self.cols)]
            base = ([] if off==0 else [self.header_labels[0].cget("text")]) + self.factor_names
            self.column_names = base + rep_names

            dlg.destroy()

        btns = tk.Frame(frm)
        btns.grid(row=2, column=0, pady=(10, 0), sticky="w")
        tk.Button(btns, text="OK", width=10, command=ok).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(btns, text="Скасувати", width=12, command=dlg.destroy).pack(side=tk.LEFT)

        dlg.update_idletasks()
        center_window(dlg)
        dlg.bind("<Return>", lambda ev: ok())
        dlg.grab_set()

    # ---------- Table window ----------
    

def start_analysis(self, factors_count):
    """
    Новий порядок:
      1) обрати факторність,
      2) задати параметри звіту (показник/одиниці/дизайн/повторні вимірювання),
      3) відкрити таблицю вводу даних.
    """
    params = self.ask_indicator_units()
    if not params.get("ok"):
        return
    self.params_current = params
    self.open_table(factors_count)


    def open_table(self, factors_count):
        if self.table_win and tk.Toplevel.winfo_exists(self.table_win):
            self.table_win.destroy()

        self.factors_count = factors_count
        self.factor_keys = ["A", "B", "C", "D"][:factors_count]

        for fk in self.factor_keys:
            if fk not in self.factor_title_map:
                self._set_factor_title(fk, f"Фактор {fk}")

        self.table_win = tk.Toplevel(self.root)
        self.table_win.title(f"S.A.D. — {factors_count}-факторний аналіз")
        self.table_win.geometry("1280x720")
        set_window_icon(self.table_win)

        self.repeat_count = 6
        self.factor_names = [self.factor_title(fk) for fk in self.factor_keys]

        # repeated measures settings (from params_current)
        self.rm_enabled = bool(getattr(self, "params_current", None) and self.params_current.get("rm"))
        self.rm_label = "Рік"
        if self.rm_enabled:
            self.rm_label = (self.params_current.get("rm_label") or "Рік").strip()

        base_cols = []
        if self.rm_enabled:
            base_cols.append(self.rm_label)
        base_cols += self.factor_names

        self.column_names = base_cols + [f"Повт.{i+1}" for i in range(self.repeat_count)]

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

        # headers
        for j, name in enumerate(self.column_names):
            lbl = tk.Label(self.inner, text=name, relief=tk.RIDGE, width=COL_W, bg="#f0f0f0", fg="#000000")
            lbl.grid(row=0, column=j, padx=2, pady=2, sticky="nsew")
            self.header_labels.append(lbl)

            off = 1 if getattr(self,'rm_enabled',False) else 0
            if off <= j < off + self.factors_count:
                lbl.bind("<Double-Button-1>", lambda e, col=j: self.rename_factor_by_col(col))

        # cells
        for i in range(self.rows):
            row_entries = []
            for j in range(self.cols):
                e = tk.Entry(
                    self.inner,
                    width=COL_W,
                    fg="#000000",
                    highlightthickness=1,
                    highlightbackground="#c0c0c0",
                    highlightcolor="#c0c0c0"
                )
                e.grid(row=i + 1, column=j, padx=2, pady=2)
                self.bind_cell(e)
                row_entries.append(e)
            self.entries.append(row_entries)

        self.inner.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

        self.entries[0][0].focus_set()

        self.table_win.bind("<Control-v>", self.on_paste)
        self.table_win.bind("<Control-V>", self.on_paste)

    # ---------- Table editing ----------
    def add_row(self):
        i = len(self.entries)
        row_entries = []
        for j in range(self.cols):
            e = tk.Entry(
                self.inner,
                width=COL_W,
                fg="#000000",
                highlightthickness=1,
                highlightbackground="#c0c0c0",
                highlightcolor="#c0c0c0"
            )
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
            e = tk.Entry(
                self.inner,
                width=COL_W,
                fg="#000000",
                highlightthickness=1,
                highlightbackground="#c0c0c0",
                highlightcolor="#c0c0c0"
            )
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
            class E:
                pass
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
        rep_start = self.factors_count + (1 if getattr(self, 'rm_enabled', False) else 0)
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

    def collect_long(self, design):
        long = []
        rep_cols = self.used_repeat_columns()
        if not rep_cols:
            return long, rep_cols

        for i, row in enumerate(self.entries):
            levels = []
            col_off = 1 if getattr(self, 'rm_enabled', False) else 0
            # TIME (рік/фаза) зчитуємо окремо, якщо увімкнено повторні вимірювання
            time_val = None
            if col_off == 1:
                tv = row[0].get().strip()
                time_val = tv if tv else f"час{i+1}"

            for k in range(self.factors_count):
                v = row[col_off + k].get().strip()
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


                if getattr(self, "rm_enabled", False):
                    rec["TIME"] = time_val
                    # subjects are repetitions (columns) nested in treatment cell
                    cell_key = "|".join(levels) if levels else f"рядок{i+1}"
                    rec["SUBJECT"] = f"{cell_key}::S{idx_c+1}"

                if design in ("rcbd", "split"):
                    rec["BLOCK"] = f"Блок {idx_c + 1}"

                long.append(rec)

        return long, rep_cols

    # -------------------------
    # ANALYZE
    # -------------------------
    def analyze(self):
        created_at = datetime.now()

        params = getattr(self, 'params_current', None) or self.ask_indicator_units()
        if not params.get("ok"):
            return

        indicator = params.get("indicator", "Показник")
        units = params.get("units", "")
        design = params.get("design", "crd")
        split_main = params.get("split_main", "A")
        rm = bool(params.get("rm"))
        rm_label = (params.get("rm_label") or "Рік").strip()

        long, used_rep_cols = self.collect_long(design)
        if len(long) == 0:
            messagebox.showwarning("Помилка", "Немає числових даних для аналізу.\nПеревірте повторності та значення.")
            return

        values = np.array([r["value"] for r in long], dtype=float)
        if len(values) < 3:
            messagebox.showinfo("Результат", "Надто мало даних для аналізу.")
            return

        # levels
        levels_by_factor = {f: first_seen_order([r.get(f) for r in long]) for f in self.factor_keys}
        variant_order = first_seen_order([tuple(r.get(f) for f in self.factor_keys) for r in long])
        v_names = [" | ".join(map(str, k)) for k in variant_order]
        num_variants = len(variant_order)

        # run design model
        try:
            if rm:
                res = anova_repeated_measures_ols(long, self.factor_keys, time_key="TIME", subject_key="SUBJECT")
                residuals = np.array(res.get("residuals", []), dtype=float)
            elif design == "crd":
                res = anova_n_way(long, self.factor_keys, levels_by_factor)
                residuals = np.array(res.get("residuals", []), dtype=float)
            elif design == "rcbd":
                res = anova_rcbd_ols(long, self.factor_keys, levels_by_factor, block_key="BLOCK")
                residuals = np.array(res.get("residuals", []), dtype=float)
            else:
                if split_main not in self.factor_keys:
                    split_main = self.factor_keys[0]
                res = anova_splitplot_ols(long, self.factor_keys, main_factor=split_main, block_key="BLOCK")
                residuals = np.array(res.get("residuals", []), dtype=float)
        except Exception as ex:
            messagebox.showerror("Помилка аналізу", str(ex))
            return

        # normality on residuals
        try:
            W, p_norm = shapiro(residuals) if len(residuals) >= 3 else (np.nan, np.nan)
        except Exception:
            W, p_norm = (np.nan, np.nan)

        normal = (p_norm is not None) and (not math.isnan(p_norm)) and (p_norm > 0.05)
        if design == "split" and not normal:
            messagebox.showwarning(
                "Split-plot: аналіз неможливий",
                "Обрано дизайн Split-plot, який у цій програмі реалізовано лише для параметричних методів.\n"
                "Оскільки залишки моделі не відповідають нормальному розподілу (p ≤ 0.05),\n"
                "параметричний split-plot аналіз є методично некоректним.\n\n"
                "Рекомендації:\n"
                "• застосувати трансформацію даних (log/√/Box-Cox) і повторити аналіз;\n"
                "• або вибрати CRD/RCBD і виконати непараметричний аналіз."
            )
            return
        # Для повторних вимірювань непараметричний блок (Friedman/Wilcoxon RM) тут не реалізований.
        # Якщо розподіл не нормальний — даємо зрозуміле пояснення.
        if rm and (p_norm is not None) and (not np.isnan(p_norm)) and (p_norm < ALPHA):
            

            messagebox.showwarning(
                "Повторні вимірювання",
                "За результатами тесту Шапіро–Уїлка залишки не відповідають нормальному розподілу (p < 0.05).\n\n"
                "Непараметричний аналіз для повторних вимірювань у цій версії ще не реалізований.\n"
                "Рекомендації:\n"
                "• застосувати трансформацію даних (log/√/Box–Cox) і повторити аналіз; або\n"
                "• виконати аналіз по кожному року/фазі окремо непараметрично (Kruskal–Wallis / Mann–Whitney) та інтерпретувати динаміку."
            )
            return



        choice = self.choose_method_window(p_norm, design, num_variants)
        if not choice["ok"]:
            return
        method = choice["method"]

        if rm:
            MS_error = res.get("MS_between_err", np.nan)
            df_error = res.get("df_between_err", np.nan)
        else:
            MS_error = res.get("MS_error", np.nan)
            df_error = res.get("df_error", np.nan)

        MS_whole = res.get("MS_whole", np.nan)
        df_whole = res.get("df_whole", np.nan)
        split_main_factor = res.get("main_factor", split_main) if design == "split" else None

        # descriptive
        vstats = variant_mean_sd(long, self.factor_keys)
        v_means = {k: vstats[k][0] for k in vstats.keys()}
        v_sds = {k: vstats[k][1] for k in vstats.keys()}
        v_ns = {k: vstats[k][2] for k in vstats.keys()}

        means1 = {v_names[i]: v_means.get(variant_order[i], np.nan) for i in range(len(variant_order))}
        ns1 = {v_names[i]: v_ns.get(variant_order[i], 0) for i in range(len(variant_order))}
        groups1 = {}
        g_variant = groups_by_keys(long, tuple(self.factor_keys))
        for i, k in enumerate(variant_order):
            groups1[v_names[i]] = g_variant.get(k, [])

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

        ranks_by_variant = mean_ranks_by_key(long, key_func=lambda rec: " | ".join(str(rec.get(f)) for f in self.factor_keys))
        ranks_by_factor = {f: mean_ranks_by_key(long, key_func=lambda rec, ff=f: rec.get(ff)) for f in self.factor_keys}

        v_medians = {}
        v_q = {}
        for i, k in enumerate(variant_order):
            name = v_names[i]
            arr = groups1.get(name, [])
            med, q1, q3 = median_q1_q3(arr)
            v_medians[k] = med
            v_q[k] = (q1, q3)

        # homogeneity (param only)
        bf_F, bf_p = (np.nan, np.nan)
        if method in ("lsd", "tukey", "duncan", "bonferroni"):
            bf_F, bf_p = brown_forsythe_from_groups(groups1)

        # nonparam globals
        kw_H, kw_p, kw_df, kw_eps2 = (np.nan, np.nan, np.nan, np.nan)
        do_posthoc = True

        fr_chi2, fr_p, fr_df, fr_W = (np.nan, np.nan, np.nan, np.nan)
        wil_stat, wil_p = (np.nan, np.nan)
        rcbd_pairwise_rows = []
        rcbd_sig = {}

        letters_factor = {f: {lvl: "" for lvl in levels_by_factor[f]} for f in self.factor_keys}
        letters_named = {name: "" for name in v_names}
        pairwise_rows = []
        factor_pairwise_tables = {}

        if method == "lsd":
            for f in self.factor_keys:
                lvls = levels_by_factor[f]
                if design == "split" and f == split_main_factor:
                    sig_f = lsd_sig_matrix(lvls, factor_means[f], factor_ns[f], MS_whole, df_whole, alpha=ALPHA)
                else:
                    sig_f = lsd_sig_matrix(lvls, factor_means[f], factor_ns[f], MS_error, df_error, alpha=ALPHA)
                letters_factor[f] = cld_multi_letters(lvls, factor_means[f], sig_f)

            if design != "split":
                sigv = lsd_sig_matrix(v_names, means1, ns1, MS_error, df_error, alpha=ALPHA)
                letters_named = cld_multi_letters(v_names, means1, sigv)

        elif method in ("tukey", "duncan", "bonferroni"):
            if design != "split":
                pairwise_rows, sig = pairwise_param_short_variants_pm(v_names, means1, ns1, MS_error, df_error, method, alpha=ALPHA)
                letters_named = cld_multi_letters(v_names, means1, sig)

                # letters within each factor (for graphical report)
                for f in self.factor_keys:
                    lvls = levels_by_factor[f]
                    means_f = factor_means[f]
                    ns_f = factor_ns[f]
                    rows_f, sig_f = pairwise_param_short_variants_pm(lvls, means_f, ns_f, MS_error, df_error, method, alpha=ALPHA)
                    factor_pairwise_tables[f] = rows_f
                    letters_factor[f] = cld_multi_letters(lvls, means_f, sig_f)

            if design == "split":
                for f in self.factor_keys:
                    lvls = levels_by_factor[f]
                    means_f = factor_means[f]
                    ns_f = factor_ns[f]

                    if f == split_main_factor:
                        rows_f, sig_f = pairwise_param_short_variants_pm(lvls, means_f, ns_f, MS_whole, df_whole, method, alpha=ALPHA)
                    else:
                        rows_f, sig_f = pairwise_param_short_variants_pm(lvls, means_f, ns_f, MS_error, df_error, method, alpha=ALPHA)

                    factor_pairwise_tables[f] = rows_f
                    letters_factor[f] = cld_multi_letters(lvls, means_f, sig_f)

        elif method == "kw":
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

        elif method == "friedman":
            block_names = first_seen_order([f"Блок {i+1}" for i in range(len(used_rep_cols))])
            long2 = []
            for r in long:
                rr = dict(r)
                rr["VARIANT"] = " | ".join(str(rr.get(f)) for f in self.factor_keys)
                long2.append(rr)

            mat_rows, kept_blocks = rcbd_matrix_from_long(long2, v_names, block_names, variant_key="VARIANT", block_key="BLOCK")
            if len(mat_rows) < 2:
                messagebox.showwarning("Помилка", "Для Friedman потрібні щонайменше 2 повних блоки (без пропусків по варіантах).")
                return

            try:
                cols = list(zip(*mat_rows))
                fr = friedmanchisquare(*[np.array(c, dtype=float) for c in cols])
                fr_chi2 = float(fr.statistic)
                fr_p = float(fr.pvalue)
                fr_df = int(len(v_names) - 1)
                fr_W = kendalls_w_from_friedman(fr_chi2, n_blocks=len(mat_rows), k_treat=len(v_names))
            except Exception:
                fr_chi2, fr_p, fr_df, fr_W = (np.nan, np.nan, np.nan, np.nan)

            if not (isinstance(fr_p, float) and math.isnan(fr_p)) and fr_p < ALPHA:
                rcbd_pairwise_rows, rcbd_sig = pairwise_wilcoxon_bonf(v_names, mat_rows, alpha=ALPHA)
                med_tmp = {name: float(np.median(groups1[name])) if len(groups1[name]) else np.nan for name in v_names}
                letters_named = cld_multi_letters(v_names, med_tmp, rcbd_sig)
            else:
                letters_named = {name: "" for name in v_names}

        elif method == "wilcoxon":
            if len(v_names) != 2:
                messagebox.showwarning("Помилка", "Wilcoxon (парний) застосовується лише для 2 варіантів.")
                return

            block_names = first_seen_order([f"Блок {i+1}" for i in range(len(used_rep_cols))])
            long2 = []
            for r in long:
                rr = dict(r)
                rr["VARIANT"] = " | ".join(str(rr.get(f)) for f in self.factor_keys)
                long2.append(rr)

            mat_rows, kept_blocks = rcbd_matrix_from_long(long2, v_names, block_names, variant_key="VARIANT", block_key="BLOCK")
            if len(mat_rows) < 2:
                messagebox.showwarning("Помилка", "Для Wilcoxon потрібні щонайменше 2 повні блоки (пари значень).")
                return
            arr = np.array(mat_rows, dtype=float)
            x = arr[:, 0]
            y = arr[:, 1]
            try:
                stat, p = wilcoxon(x, y, zero_method="wilcox", correction=False, alternative="two-sided", mode="auto")
                wil_stat = float(stat)
                wil_p = float(p)
            except Exception:
                wil_stat, wil_p = (np.nan, np.nan)

            if not (isinstance(wil_p, float) and math.isnan(wil_p)) and wil_p < ALPHA:
                rcbd_sig = {(v_names[0], v_names[1]): True}
                med_tmp = {name: float(np.median(groups1[name])) if len(groups1[name]) else np.nan for name in v_names}
                letters_named = cld_multi_letters(v_names, med_tmp, rcbd_sig)
            else:
                letters_named = {name: "" for name in v_names}

        letters_variants = {variant_order[i]: letters_named.get(v_names[i], "") for i in range(len(variant_order))}

        SS_total = res.get("SS_total", np.nan)
        SS_error = res.get("SS_error", np.nan)
        R2 = (1.0 - (SS_error / SS_total)) if (not any(math.isnan(x) for x in [SS_total, SS_error]) and SS_total > 0) else np.nan

        cv_rows = []
        for f in self.factor_keys:
            lvl_means = [factor_means[f].get(lvl, np.nan) for lvl in levels_by_factor[f]]
            cv_f = cv_percent_from_level_means(lvl_means)
            cv_rows.append([self.factor_title(f), fmt_num(cv_f, 2)])
        cv_total = cv_percent_from_values(values)
        cv_rows.append(["Загальний", fmt_num(cv_total, 2)])

        seg = []
        seg.append(("text", "З В І Т   С Т А Т И С Т И Ч Н О Г О   А Н А Л І З У   Д А Н И Х\n\n"))
        seg.append(("text", f"Показник:\t{indicator}\nОдиниці виміру:\t{units}\n\n"))

        seg.append(("text",
                    f"Кількість варіантів:\t{num_variants}\n"
                    f"Кількість повторностей:\t{len(used_rep_cols)}\n"
                    f"Загальна кількість облікових значень:\t{len(long)}\n\n"))

        design_label = {"crd": "CRD (повна рандомізація)", "rcbd": "RCBD (блочна рандомізація)", "split": "Split-plot (спліт-плот)"}[design]
        seg.append(("text", f"Дизайн експерименту:\t{design_label}\n"))
        if design == "split":
            seg.append(("text", f"Головний фактор (Whole-plot factor):\t{split_main_factor}\n\n"))
        else:
            seg.append(("text", "\n"))

        method_label = {
            "lsd": "Параметричний аналіз: Brown–Forsythe + ANOVA + НІР₀₅ (LSD).",
            "tukey": "Параметричний аналіз: Brown–Forsythe + ANOVA + тест Тьюкі (Tukey HSD).",
            "duncan": "Параметричний аналіз: Brown–Forsythe + ANOVA + тест Дункана.",
            "bonferroni": "Параметричний аналіз: Brown–Forsythe + ANOVA + корекція Бонферроні.",
            "kw": "Непараметричний аналіз: Kruskal–Wallis.",
            "mw": "Непараметричний аналіз: Mann–Whitney.",
            "friedman": "Непараметричний аналіз (RCBD): Friedman.",
            "wilcoxon": "Непараметричний аналіз (RCBD): Wilcoxon signed-rank (парний).",
        }.get(method, "")

        if method_label:
            seg.append(("text", f"Виконуваний статистичний аналіз:\t{method_label}\n\n"))

        seg.append(("text", "Пояснення позначень істотності: ** — p<0.01; * — p<0.05.\n"))
        seg.append(("text", "У таблицях знак \"-\" свідчить що p ≥ 0.05.\n"))
        seg.append(("text", "Істотна різниця (літери): різні літери свідчать про наявність істотної різниці.\n\n"))

        if not math.isnan(W):
            seg.append(("text",
                        f"Перевірка нормальності залишків (Shapiro–Wilk):\t{normality_text(p_norm)}\t"
                        f"(W={fmt_num(float(W),4)}; p={fmt_num(float(p_norm),4)})\n\n"))
        else:
            seg.append(("text", "Перевірка нормальності залишків (Shapiro–Wilk):\tн/д\n\n"))

        nonparam = method in ("mw", "kw", "friedman", "wilcoxon")

        if method == "kw":
            if not (isinstance(kw_p, float) and math.isnan(kw_p)):
                concl = "істотна різниця " + significance_mark(kw_p) if kw_p < ALPHA else "-"
                seg.append(("text",
                            f"Глобальний тест між варіантами (Kruskal–Wallis):\t"
                            f"H={fmt_num(kw_H,4)}; df={int(kw_df)}; p={fmt_num(kw_p,4)}\t{concl}\n"))
                seg.append(("text", f"Розмір ефекту (ε²):\t{fmt_num(kw_eps2,4)}\n\n"))
            else:
                seg.append(("text", "Глобальний тест між варіантами (Kruskal–Wallis):\tн/д\n\n"))

        if method == "friedman":
            if not (isinstance(fr_p, float) and math.isnan(fr_p)):
                concl = "істотна різниця " + significance_mark(fr_p) if fr_p < ALPHA else "-"
                seg.append(("text",
                            f"Глобальний тест між варіантами (Friedman):\t"
                            f"χ²={fmt_num(fr_chi2,4)}; df={int(fr_df)}; p={fmt_num(fr_p,4)}\t{concl}\n"))
                seg.append(("text", f"Розмір ефекту (Kendall’s W):\t{fmt_num(fr_W,4)}\n\n"))
            else:
                seg.append(("text", "Глобальний тест між варіантами (Friedman):\tн/д\n\n"))

        if method == "wilcoxon":
            if not (isinstance(wil_p, float) and math.isnan(wil_p)):
                concl = "істотна різниця " + significance_mark(wil_p) if wil_p < ALPHA else "-"
                seg.append(("text",
                            f"Парний тест (Wilcoxon signed-rank):\t"
                            f"W={fmt_num(wil_stat,4)}; p={fmt_num(wil_p,4)}\t{concl}\n\n"))
            else:
                seg.append(("text", "Парний тест (Wilcoxon signed-rank):\tн/д\n\n"))

        if not nonparam:
            if not any(math.isnan(x) for x in [bf_F, bf_p]):
                bf_concl = "умова виконується" if bf_p >= ALPHA else f"умова порушена {significance_mark(bf_p)}"
                seg.append(("text",
                            f"Перевірка однорідності дисперсій (Brown–Forsythe):\t"
                            f"F={fmt_num(bf_F,4)}; p={fmt_num(bf_p,4)}\t{bf_concl}\n\n"))
            else:
                seg.append(("text", "Перевірка однорідності дисперсій (Brown–Forsythe):\tн/д\n\n"))

            if design == "split":
                seg.append(("text",
                            "Примітка (Split-plot):\n"
                            f"• {self.factor_title(split_main_factor)} перевірено на MS(Блоки×{split_main_factor}) (whole-plot error).\n"
                            "• Інші ефекти перевірено на MS(Залишок) (sub-plot error).\n\n"))

            # ANOVA table
            anova_rows = []
            for name, SSv, dfv, MSv, Fv, pv in res["table"]:
                df_txt = str(int(dfv)) if dfv is not None and not (isinstance(dfv, float) and math.isnan(dfv)) else ""
                # rename "Фактор A" -> custom
                name2 = name
                if isinstance(name2, str) and name2.startswith("Фактор "):
                    try:
                        rest = name2.replace("Фактор ", "")
                        parts = rest.split("×")
                        parts2 = [self.factor_title(p) if p in self.factor_keys else p for p in parts]
                        name2 = "×".join(parts2)
                    except Exception:
                        pass

                if name2.startswith("Залишок") or name2 == "Загальна" or "Whole-plot error" in name2 or name2 == "Блоки":
                    anova_rows.append([name2, fmt_num(SSv, 2), df_txt, fmt_num(MSv, 3), "", "", ""])
                else:
                    mark = significance_mark(pv)
                    concl = f"істотна різниця {mark}" if mark else "-"
                    anova_rows.append([name2, fmt_num(SSv, 2), df_txt, fmt_num(MSv, 3), fmt_num(Fv, 3), fmt_num(pv, 4), concl])

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
            for r in eff_rows:
                if r and isinstance(r[0], str) and r[0].startswith("Фактор "):
                    rest = r[0].replace("Фактор ", "")
                    parts = rest.split("×")
                    parts2 = [self.factor_title(p) if p in self.factor_keys else p for p in parts]
                    r[0] = "×".join(parts2)

            seg.append(("text", "ТАБЛИЦЯ 2. Сила впливу факторів та їх комбінацій (% від SS)\n"))
            seg.append(("table", {"headers": ["Джерело", "%"], "rows": eff_rows}))
            seg.append(("text", "\n"))

            pe2_rows = build_partial_eta2_rows_with_label(res["table"])
            for r in pe2_rows:
                if r and isinstance(r[0], str) and r[0].startswith("Фактор "):
                    rest = r[0].replace("Фактор ", "")
                    parts = rest.split("×")
                    parts2 = [self.factor_title(p) if p in self.factor_keys else p for p in parts]
                    r[0] = "×".join(parts2)

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
                for key, val in res.get("NIR05", {}).items():
                    nir_rows.append([key.replace("Фактор ", self.factor_title_map.get(key.replace("Фактор ", ""), "Фактор ")), fmt_num(val, 4)])
                if nir_rows:
                    seg.append(("text", "ТАБЛИЦЯ 5. Значення НІР₀₅\n"))
                    seg.append(("table", {"headers": ["Елемент", "НІР₀₅"], "rows": nir_rows}))
                    seg.append(("text", "\n"))
                    tno = 6

            for f in self.factor_keys:
                seg.append(("text", f"ТАБЛИЦЯ {tno}. Середнє по фактору: {self.factor_title(f)}\n"))
                rows_f = []
                for lvl in levels_by_factor[f]:
                    m = factor_means[f].get(lvl, np.nan)
                    letter = letters_factor[f].get(lvl, "")
                    rows_f.append([str(lvl), fmt_num(m, 3), (letter if letter else "-")])
                seg.append(("table", {"headers": [self.factor_title(f), "Середнє", "Істотна різниця"], "rows": rows_f}))
                seg.append(("text", "\n"))
                tno += 1

            seg.append(("text", f"ТАБЛИЦЯ {tno}. Таблиця середніх значень варіантів\n"))
            rows_v = []
            for k in variant_order:
                name = " | ".join(map(str, k))
                m = v_means.get(k, np.nan)
                sd = v_sds.get(k, np.nan)
                if design == "split":
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

            if design != "split":
                if method in ("tukey", "duncan", "bonferroni") and pairwise_rows:
                    seg.append(("text", f"ТАБЛИЦЯ {tno}. Парні порівняння варіантів\n"))
                    seg.append(("table", {"headers": ["Комбінація варіантів", "p", "Істотна різниця"], "rows": pairwise_rows}))
                    seg.append(("text", "\n"))
            else:
                seg.append(("text",
                            "Примітка (Split-plot): парні порівняння для повних варіантів (комбінацій факторів)\n"
                            "не подаються, оскільки для таких порівнянь потрібні спеціальні контрасти та коректний\n"
                            "облік двох різних помилок (whole-plot і sub-plot). Натомість подано парні порівняння\n"
                            "на рівні факторів з правильними error-term.\n\n"))
                if method in ("tukey", "duncan", "bonferroni"):
                    for f in self.factor_keys:
                        rows_pf = factor_pairwise_tables.get(f, [])
                        if rows_pf:
                            seg.append(("text", f"ТАБЛИЦЯ {tno}. Парні порівняння для: {self.factor_title(f)} (Split-plot)\n"))
                            seg.append(("table", {"headers": ["Комбінація", "p", "Істотна різниця"], "rows": rows_pf}))
                            seg.append(("text", "\n"))
                            tno += 1

        else:
            tno = 1
            for f in self.factor_keys:
                seg.append(("text", f"ТАБЛИЦЯ {tno}. Описова статистика (непараметрична): {self.factor_title(f)}\n"))
                rows = []
                for lvl in levels_by_factor[f]:
                    med = factor_medians[f].get(lvl, np.nan)
                    q1, q3 = factor_q[f].get(lvl, (np.nan, np.nan))
                    rank_m = ranks_by_factor[f].get(lvl, np.nan)
                    rows.append([
                        str(lvl),
                        str(int(factor_ns[f].get(lvl, 0))),
                        fmt_num(med, 3),
                        f"{fmt_num(q1,3)}–{fmt_num(q3,3)}" if not any(math.isnan(x) for x in [q1, q3]) else "",
                        fmt_num(rank_m, 2),
                        "-"
                    ])
                seg.append(("table", {"headers": [self.factor_title(f), "n", "Медіана", "Q1–Q3", "Середній ранг", "Істотна різниця"], "rows": rows}))
                seg.append(("text", "\n"))
                tno += 1

            seg.append(("text", f"ТАБЛИЦЯ {tno}. Описова статистика варіантів (непараметрична)\n"))
            rows = []
            for k in variant_order:
                name = " | ".join(map(str, k))
                med = v_medians.get(k, np.nan)
                q1, q3 = v_q.get(k, (np.nan, np.nan))
                rank_m = ranks_by_variant.get(name, np.nan)
                rows.append([
                    name,
                    str(int(v_ns.get(k, 0))),
                    fmt_num(med, 3),
                    f"{fmt_num(q1,3)}–{fmt_num(q3,3)}" if not any(math.isnan(x) for x in [q1, q3]) else "",
                    fmt_num(rank_m, 2),
                    "-"
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

            if method == "kw":
                if not do_posthoc:
                    seg.append(("text", "Пост-хок порівняння не виконувалися, оскільки глобальний тест Kruskal–Wallis не виявив істотної різниці (p ≥ 0.05).\n\n"))
                else:
                    if pairwise_rows:
                        seg.append(("text", f"ТАБЛИЦЯ {tno}. Парні порівняння + ефект (Cliff’s δ)\n"))
                        seg.append(("table", {"headers": ["Комбінація варіантів", "U", "p (Bonf.)", "Істотна різниця", "δ", "Висновок"], "rows": pairwise_rows}))
                        seg.append(("text", "\n"))

            if method == "friedman":
                if not (isinstance(fr_p, float) and math.isnan(fr_p)) and fr_p >= ALPHA:
                    seg.append(("text", "Пост-хок порівняння не виконувалися, оскільки глобальний тест Friedman не виявив істотної різниці (p ≥ 0.05).\n\n"))
                else:
                    if rcbd_pairwise_rows:
                        seg.append(("text", f"ТАБЛИЦЯ {tno}. Парні порівняння (Wilcoxon, Bonferroni) + ефект (r)\n"))
                        seg.append(("table", {"headers": ["Комбінація варіантів", "W", "p (Bonf.)", "Істотна різниця", "r"], "rows": rcbd_pairwise_rows}))
                        seg.append(("text", "\n"))

        seg.append(("text", f"Звіт сформовано:\t{created_at.strftime('%d.%m.%Y, %H:%M')}\n"))
        self.show_report_segments(seg)
        self.show_graphical_report(long, self.factor_keys, levels_by_factor, letters_factor, indicator, units)

    # -------------------------
    # SHOW REPORT
    # -------------------------
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
    # GRAPHICAL REPORT (combined factor boxplots)
    # -------------------------
    def show_graphical_report(self, long, factor_keys, levels_by_factor, letters_factor, indicator, units):
        if Figure is None or FigureCanvasTkAgg is None:
            messagebox.showwarning(
                "Графічний звіт недоступний",
                "Не вдалося завантажити matplotlib. Переконайтеся, що встановлено matplotlib."
            )
            return

        if getattr(self, "graph_win", None) and tk.Toplevel.winfo_exists(self.graph_win):
            self.graph_win.destroy()

        self.graph_win = tk.Toplevel(self.root)
        self.graph_win.title("Графічний звіт (Boxplot за факторами)")
        self.graph_win.geometry("1200x760")
        set_window_icon(self.graph_win)

        top = tk.Frame(self.graph_win, padx=8, pady=8)
        top.pack(fill=tk.X)

        btn_copy = ttk.Button(top, text="Копіювати графік (PNG)", command=lambda: self._copy_graph_png())
        btn_copy.pack(side=tk.LEFT)

        hint = tk.Label(top, text="Порада: після копіювання вставляйте у Word через Ctrl+V.", anchor="w")
        hint.pack(side=tk.LEFT, padx=12)

        # Figure
        fig = Figure(figsize=(11, 6), dpi=100)
        ax = fig.add_subplot(111)

        # Prepare sequential positions with gaps between factors
        positions = []
        data = []
        xticklabels = []
        letters = []
        factor_centers = []  # (center_x, factor_name)
        x = 1.0
        gap = 1.0  # visible space between factors

        for f in factor_keys:
            lvls = levels_by_factor.get(f, [])
            if not lvls:
                continue
            start_x = x
            for lvl in lvls:
                arr = [float(r["value"]) for r in long if r.get(f) == lvl and (r.get("value") is not None)]
                arr = [v for v in arr if not math.isnan(v)]
                data.append(arr)
                positions.append(x)
                xticklabels.append(str(lvl))
                letters.append((f, lvl))
                x += 1.0
            end_x = x - 1.0
            factor_centers.append(((start_x + end_x) / 2.0, self.factor_title(f)))
            x += gap

        if not data:
            messagebox.showinfo("Графічний звіт", "Недостатньо даних для побудови графіка.")
            self.graph_win.destroy()
            return

        # Draw boxplot
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.6,
            showfliers=True
        )

        ax.set_title(f"{indicator}, {units}")
        ax.set_ylabel(units)

        ax.set_xticks(positions)
        ax.set_xticklabels(xticklabels, rotation=90)

        # Grid (light, not dense)
        ax.yaxis.grid(True, linestyle='-', linewidth=0.5, alpha=0.35)

        # Letters above each box (within each factor)
        # compute global offset
        all_vals = [v for arr in data for v in arr]
        if all_vals:
            y_min = float(min(all_vals))
            y_max = float(max(all_vals))
            dy = (y_max - y_min) if y_max > y_min else (abs(y_max) if y_max != 0 else 1.0)
            offset = 0.03 * dy
        else:
            offset = 1.0

        for i, (f, lvl) in enumerate(letters):
            arr = data[i]
            if not arr:
                continue
            letter = ""
            try:
                letter = (letters_factor.get(f, {}) or {}).get(lvl, "")
            except Exception:
                letter = ""
            if not letter:
                continue
            y = max(arr) + offset
            ax.text(positions[i], y, letter, ha="center", va="bottom", fontsize=12)

        # Factor labels below groups
        fig.subplots_adjust(bottom=0.34, top=0.90, left=0.08, right=0.98)
        for cx, fname in factor_centers:
            ax.text(
                cx, -0.22, fname,
                ha="center", va="top",
                transform=ax.get_xaxis_transform(),
                fontsize=12
            )

        self._graph_fig = fig  # keep reference for copy
        self._graph_canvas = FigureCanvasTkAgg(fig, master=self.graph_win)
        self._graph_canvas.draw()
        self._graph_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    def _copy_graph_png(self):
        fig = getattr(self, "_graph_fig", None)
        if fig is None:
            messagebox.showwarning("Копіювання", "Немає графіка для копіювання.")
            return
        ok, msg = copy_figure_png_to_clipboard(fig)
        if ok:
            messagebox.showinfo("Копіювання", "Графік скопійовано у буфер обміну як зображення (PNG).")
        else:
            messagebox.showwarning("Копіювання", f"Не вдалося скопіювати графік. {msg}")



# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    root = tk.Tk()
    set_window_icon(root)
    app = SADTk(root)
    root.mainloop()
