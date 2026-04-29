# main.py  — S.A.D. v2.1
# -*- coding: utf-8 -*-
"""
S.A.D. — Статистичний аналіз даних  v2.1
Залежності: pip install numpy scipy openpyxl matplotlib pillow
"""
import os, sys, math, json, io
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser
from tkinter.scrolledtext import ScrolledText
import tkinter.font as tkfont
from itertools import combinations
from collections import defaultdict
from datetime import datetime

from scipy.stats import (shapiro, kruskal, mannwhitneyu, friedmanchisquare,
                         wilcoxon, levene, pearsonr, spearmanr)
from scipy.stats import f as f_dist, t as t_dist, norm
from scipy.stats import studentized_range

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    HAS_MPL = True
except Exception:
    HAS_MPL = False; Figure = None; FigureCanvasTkAgg = None

try:
    from PIL import Image as _PILImage
    HAS_PIL = True
except Exception:
    HAS_PIL = False

ALPHA   = 0.05
COL_W   = 10
APP_VER = "2.1"

if HAS_MPL:
    import matplotlib
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'font.serif':  ['Times New Roman','Times','DejaVu Serif'],
        'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'axes.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False,
    })

# ── DPI awareness ──────────────────────────────────────────────
try:
    import ctypes
    try:    ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try: ctypes.windll.user32.SetProcessDPIAware()
        except Exception: pass
except Exception: pass

# ── Icon ──────────────────────────────────────────────────────
def _find_icon():
    dirs = [os.getcwd()]
    try: dirs.insert(0, os.path.dirname(os.path.abspath(__file__)))
    except Exception: pass
    try:
        if hasattr(sys, "_MEIPASS"): dirs.append(sys._MEIPASS)
    except Exception: pass
    for d in dirs:
        p = os.path.join(d, "icon.ico")
        if os.path.exists(p): return p
    return None

def set_icon(win):
    ico = _find_icon()
    if not ico: return
    try: win.iconbitmap(ico)
    except Exception:
        try: win.iconbitmap(default=ico)
        except Exception: pass

# ── Clipboard PNG → Windows ────────────────────────────────────
def _copy_fig_to_clipboard(fig):
    if not (HAS_MPL and HAS_PIL): return False, "Потрібні matplotlib і Pillow"
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        pil = _PILImage.open(buf)
        ok, msg = _copy_pil_win(pil); buf.close(); return ok, msg
    except Exception as ex: return False, str(ex)

def _copy_pil_win(pil_img):
    try: import ctypes; from ctypes import wintypes
    except Exception: return False, "ctypes недоступний"
    if sys.platform != "win32": return False, "Лише для Windows"
    if pil_img is None: return False, "Немає зображення"
    try:
        buf = io.BytesIO(); pil_img.convert("RGB").save(buf, "BMP"); bmp = buf.getvalue()
        if len(bmp) <= 14: return False, "BMP помилка"
        data = bmp[14:]
    except Exception as ex: return False, str(ex)
    u32 = ctypes.WinDLL("user32", use_last_error=True)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    u32.OpenClipboard.argtypes  = [wintypes.HWND];           u32.OpenClipboard.restype  = wintypes.BOOL
    u32.CloseClipboard.argtypes = [];                        u32.CloseClipboard.restype  = wintypes.BOOL
    u32.EmptyClipboard.argtypes = [];                        u32.EmptyClipboard.restype  = wintypes.BOOL
    u32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]; u32.SetClipboardData.restype = wintypes.HANDLE
    k32.GlobalAlloc.argtypes  = [wintypes.UINT, ctypes.c_size_t]; k32.GlobalAlloc.restype  = wintypes.HGLOBAL
    k32.GlobalLock.argtypes   = [wintypes.HGLOBAL];          k32.GlobalLock.restype   = wintypes.LPVOID
    k32.GlobalUnlock.argtypes = [wintypes.HGLOBAL];          k32.GlobalUnlock.restype = wintypes.BOOL
    k32.GlobalFree.argtypes   = [wintypes.HGLOBAL];          k32.GlobalFree.restype   = wintypes.HGLOBAL
    if not u32.OpenClipboard(None): return False, f"OpenClipboard err {ctypes.get_last_error()}"
    try:
        u32.EmptyClipboard()
        hg = k32.GlobalAlloc(0x0042, len(data))
        if not hg: return False, "GlobalAlloc failed"
        pg = k32.GlobalLock(hg)
        if not pg: k32.GlobalFree(hg); return False, "GlobalLock failed"
        try: ctypes.memmove(pg, data, len(data))
        finally: k32.GlobalUnlock(hg)
        if not u32.SetClipboardData(8, hg): k32.GlobalFree(hg); return False, "SetClipboardData failed"
        return True, ""
    finally: u32.CloseClipboard()

# ═══════════════════════════════════════════════════════════════
# STAT HELPERS
# ═══════════════════════════════════════════════════════════════
def sig_mark(p):
    if p is None or (isinstance(p, float) and math.isnan(p)): return ""
    return "**" if p < 0.01 else ("*" if p < 0.05 else "")

def norm_txt(p):
    if p is None or (isinstance(p, float) and math.isnan(p)): return "н/д"
    return "нормальний розподіл" if p > 0.05 else "ненормальний розподіл"

def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)): return ""
    try: return f"{float(x):.{nd}f}"
    except: return ""

def first_seen(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen: seen.add(x); out.append(x)
    return out

def center_win(win):
    win.update_idletasks()
    w, h = win.winfo_width(), win.winfo_height()
    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    win.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

def median_q(arr):
    if not arr: return np.nan, np.nan, np.nan
    a = np.array(arr, dtype=float); a = a[~np.isnan(a)]
    if len(a) == 0: return np.nan, np.nan, np.nan
    return float(np.median(a)), float(np.percentile(a, 25)), float(np.percentile(a, 75))

def cv_vals(vals):
    a = np.array(vals, dtype=float); a = a[~np.isnan(a)]
    if len(a) < 2: return np.nan
    m = float(np.mean(a))
    return np.nan if m == 0 else float(np.std(a, ddof=1) / m * 100)

def cv_means(means):
    v = [float(x) for x in means if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(v) < 2: return np.nan
    m = float(np.mean(v))
    return np.nan if m == 0 else float(np.std(v, ddof=1) / m * 100)

def eta2_label(pe2):
    if pe2 is None or math.isnan(pe2): return ""
    if pe2 < 0.01: return "дуже слабкий"
    if pe2 < 0.06: return "слабкий"
    if pe2 < 0.14: return "середній"
    return "сильний"

def eps2_kw(H, n, k):
    if any(x is None for x in [H, n, k]) or math.isnan(H) or n <= k or k < 2: return np.nan
    return float((H - k + 1) / (n - k))

def kendalls_w(chisq, nb, kt):
    if any(x is None for x in [chisq, nb, kt]) or math.isnan(chisq) or nb <= 0 or kt <= 1: return np.nan
    return float(chisq / (nb * (kt - 1)))

def cliffs_d(x, y):
    x = np.array(x, dtype=float); y = np.array(y, dtype=float)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0: return np.nan
    gt = int(np.sum(x[:, None] > y[None, :])); lt = int(np.sum(x[:, None] < y[None, :]))
    return float((gt - lt) / (nx * ny))

def cliffs_lbl(d):
    if d is None or math.isnan(d): return ""
    if d < 0.147: return "дуже слабкий"
    if d < 0.33:  return "слабкий"
    if d < 0.474: return "середній"
    return "сильний"

def fit_font(texts, family="Times New Roman", start=13, min_s=9, target=155):
    f = tkfont.Font(family=family, size=start); sz = start
    while sz > min_s:
        if max(f.measure(t) for t in texts) <= target: break
        sz -= 1; f.configure(size=sz)
    return f

# ═══════════════════════════════════════════════════════════════
# TREEVIEW TABLE (fixed columns, no drift)
# ═══════════════════════════════════════════════════════════════
def make_tv(parent, headers, rows, min_col=90):
    frm = tk.Frame(parent, bd=1, relief=tk.SUNKEN)
    vsb = ttk.Scrollbar(frm, orient="vertical")
    hsb = ttk.Scrollbar(frm, orient="horizontal")
    tv = ttk.Treeview(frm, columns=headers, show="headings",
                      yscrollcommand=vsb.set, xscrollcommand=hsb.set,
                      height=min(len(rows) + 1, 22))
    vsb.config(command=tv.yview); hsb.config(command=tv.xview)
    vsb.pack(side=tk.RIGHT, fill=tk.Y)
    hsb.pack(side=tk.BOTTOM, fill=tk.X)
    tv.pack(fill=tk.BOTH, expand=True)
    fnt = tkfont.Font(family="Times New Roman", size=11)
    for i, h in enumerate(headers):
        cw = max(fnt.measure(str(h)) + 24, min_col,
                 max((fnt.measure(str(r[i]) if i < len(r) and r[i] else "") + 20) for r in rows) if rows else min_col)
        tv.heading(h, text=str(h), anchor="center")
        tv.column(h, width=cw, minwidth=50, anchor="center", stretch=True)
    for row in rows:
        tv.insert("", "end", values=[("" if v is None else str(v)) for v in row])
    style = ttk.Style()
    style.configure("Treeview", font=("Times New Roman", 11), rowheight=22)
    style.configure("Treeview.Heading", font=("Times New Roman", 11, "bold"))
    return frm, tv

# ═══════════════════════════════════════════════════════════════
# DATA HELPERS
# ═══════════════════════════════════════════════════════════════
def groups_by(long, keys):
    g = defaultdict(list)
    for r in long:
        v = r.get("value", np.nan)
        if v is None or math.isnan(v): continue
        g[tuple(r.get(x) for x in keys)].append(float(v))
    return g

def vstats(long, fkeys):
    vals = defaultdict(list)
    for r in long:
        v = r.get("value", np.nan)
        if v is None or math.isnan(v): continue
        vals[tuple(r.get(k) for k in fkeys)].append(float(v))
    out = {}
    for k, a in vals.items():
        n = len(a); m = float(np.mean(a)) if n else np.nan
        sd = float(np.std(a, ddof=1)) if n >= 2 else (0. if n == 1 else np.nan)
        out[k] = (m, sd, n)
    return out

def mean_ranks(long, keyfn):
    vals = []; ks = []
    for r in long:
        v = r.get("value", np.nan)
        if v is None or math.isnan(v): continue
        vals.append(float(v)); ks.append(keyfn(r))
    if not vals: return {}
    order = np.argsort(vals); sv = np.array(vals)[order]
    ranks = np.empty(len(vals), dtype=float)
    i = 0
    while i < len(sv):
        j = i
        while j < len(sv) and sv[j] == sv[i]: j += 1
        ar = (i + 1 + j) / 2.; ranks[order[i:j]] = ar; i = j
    agg = defaultdict(list)
    for k, rk in zip(ks, ranks): agg[k].append(float(rk))
    return {k: float(np.mean(v)) for k, v in agg.items()}

# ═══════════════════════════════════════════════════════════════
# CLD
# ═══════════════════════════════════════════════════════════════
def cld(levels_order, means_dict, sig_matrix):
    valid = [l for l in levels_order if not math.isnan(means_dict.get(l, np.nan))]
    if not valid: return {l: "" for l in levels_order}
    sl = sorted(valid, key=lambda z: means_dict[z], reverse=True)
    def sig(a, b): return bool(sig_matrix.get((a, b), False) or sig_matrix.get((b, a), False))
    groups = []
    for lv in sl:
        compat = [gi for gi, g in enumerate(groups) if all(not sig(lv, o) for o in g)]
        if not compat: groups.append({lv})
        else:
            for gi in compat: groups[gi].add(lv)
    def shared(a, b): return any(a in g and b in g for g in groups)
    for i in range(len(sl)):
        for j in range(i + 1, len(sl)):
            a, b = sl[i], sl[j]
            if sig(a, b) or shared(a, b): continue
            ng = {a, b}
            for c in sl:
                if c in ng: continue
                if not sig(c, a) and not sig(c, b) and all(not sig(c, x) for x in ng): ng.add(c)
            groups.append(ng)
    uniq = []
    for g in groups:
        if not any(g == h for h in uniq): uniq.append(g)
    cleaned = [g for g in uniq if not any(g < h for h in uniq)]
    alpha_ = "abcdefghijklmnopqrstuvwxyz"
    mapping = {lv: [] for lv in sl}
    for gi, g in enumerate(cleaned):
        lt = alpha_[gi] if gi < len(alpha_) else f"g{gi}"
        for lv in g: mapping[lv].append(lt)
    return {lv: "".join(sorted(mapping.get(lv, []))) for lv in levels_order}

# ═══════════════════════════════════════════════════════════════
# LEVENE TEST
# ═══════════════════════════════════════════════════════════════
def levene_test(groups_dict):
    arrs = [np.array(v, dtype=float) for v in groups_dict.values() if len(v) > 0]
    if len(arrs) < 2: return np.nan, np.nan
    try:
        stat, p = levene(*arrs, center='median')
        return float(stat), float(p)
    except Exception: return np.nan, np.nan

# ═══════════════════════════════════════════════════════════════
# PAIRWISE — parametric
# ═══════════════════════════════════════════════════════════════
def lsd_sig(levels, means, ns, MS, df, alpha=ALPHA):
    sig = {}
    if MS is None or df is None or math.isnan(MS) or math.isnan(df): return sig
    df = int(df);
    if df <= 0: return sig
    tc = float(t_dist.ppf(1 - alpha / 2, df))
    for a, b in combinations(levels, 2):
        ma, mb = means.get(a, np.nan), means.get(b, np.nan)
        na, nb = ns.get(a, 0), ns.get(b, 0)
        if any(math.isnan(x) for x in [ma, mb]) or na <= 0 or nb <= 0: continue
        se = math.sqrt(MS * (1 / na + 1 / nb))
        sig[(a, b)] = (abs(ma - mb) > tc * se)
    return sig

def pairwise_param(levels, means, ns, MS, df, method, alpha=ALPHA):
    """
    Parametric pairwise comparisons.
    Duncan: true step-down procedure — critical q depends on number of
    means spanned (p-range), not total m. This is methodologically correct.
    Tukey: simultaneous, uses studentized range with m groups.
    Bonferroni: Bonferroni-adjusted t-test.
    """
    rows = []; sig = {}
    if MS is None or df is None or math.isnan(MS) or math.isnan(df): return rows, sig
    df = int(df)
    if df <= 0: return rows, sig
    lvls = [x for x in levels if not math.isnan(means.get(x, np.nan)) and ns.get(x, 0) > 0]
    m = len(lvls)
    if m < 2: return rows, sig

    # For Duncan: sort means descending and compute step ranges
    if method == "duncan":
        sorted_lvls = sorted(lvls, key=lambda x: means[x], reverse=True)
        # Build significance matrix using step-down procedure
        for i in range(m):
            for j in range(i + 1, m):
                a, b = sorted_lvls[i], sorted_lvls[j]
                ma, mb = means[a], means[b]; na, nb = ns[a], ns[b]
                se = math.sqrt(MS * (1 / na + 1 / nb))
                if se <= 0: continue
                p_range = j - i + 1      # number of means spanned (≥ 2)
                # Duncan critical value: use studentized range with p_range groups
                # Duncan alpha per step: alpha_p = 1 - (1 - alpha)^(p-1)
                alpha_p = 1.0 - (1.0 - alpha) ** (p_range - 1)
                alpha_p = min(alpha_p, 0.5)  # cap for stability
                try:
                    q_crit = float(studentized_range.ppf(1 - alpha_p, p_range, df))
                    lsd_p = q_crit * se / math.sqrt(2)
                    is_s = (abs(ma - mb) > lsd_p)
                    # compute approximate p via q
                    q_obs = abs(ma - mb) * math.sqrt(2) / se
                    pa = float(1 - studentized_range.cdf(q_obs, p_range, df))
                except Exception:
                    is_s = False; pa = np.nan
                sig[(a, b)] = is_s
                rows.append([f"{a} vs {b}", fmt(pa, 4),
                             ("істотна різниця " + sig_mark(pa)) if is_s else "–"])
        return rows, sig

    # Tukey and Bonferroni
    for a, b in combinations(lvls, 2):
        ma, mb = means[a], means[b]; na, nb = ns[a], ns[b]
        se = math.sqrt(MS * (1 / na + 1 / nb))
        if se <= 0: continue
        tv = abs(ma - mb) / se; pr = 2 * (1 - float(t_dist.cdf(tv, df)))
        if method == "bonferroni":
            pa = min(1., pr * (m * (m - 1) / 2))
        elif method == "tukey":
            # Tukey–Kramer (handles unequal n via harmonic se)
            pa = float(1 - studentized_range.cdf(math.sqrt(2) * tv, m, df))
        else:
            pa = pr
        is_s = (pa < alpha); sig[(a, b)] = is_s
        rows.append([f"{a} vs {b}", fmt(pa, 4),
                     ("істотна різниця " + sig_mark(pa)) if is_s else "–"])
    return rows, sig

# ═══════════════════════════════════════════════════════════════
# PAIRWISE — nonparametric
# ═══════════════════════════════════════════════════════════════
def pairwise_mw(levels, groups, alpha=ALPHA):
    rows = []; sig = {}
    lvls = [x for x in levels if len(groups.get(x, [])) > 0]
    m = len(lvls); mt = m * (m - 1) / 2
    if m < 2: return rows, sig
    for a, b in combinations(lvls, 2):
        x = np.array(groups[a], dtype=float); y = np.array(groups[b], dtype=float)
        try:
            U, p = mannwhitneyu(x, y, alternative="two-sided")
            pa = min(1., float(p) * mt); d = cliffs_d(x, y)
            sig[(a, b)] = (pa < alpha)
            rows.append([f"{a} vs {b}", fmt(float(U), 3), fmt(pa, 4),
                         ("істотна різниця " + sig_mark(pa)) if pa < alpha else "–",
                         fmt(d, 4), cliffs_lbl(abs(d))])
        except Exception: continue
    return rows, sig

def pairwise_wilcox(levels, mat, alpha=ALPHA):
    rows = []; sig = {}
    k = len(levels); mt = k * (k - 1) / 2
    if k < 2: return rows, sig
    arr = np.array(mat, dtype=float)
    for i in range(k):
        for j in range(i + 1, k):
            x, y = arr[:, i], arr[:, j]
            try:
                st, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided", mode="auto")
                pa = min(1., float(p) * mt)
                z = abs(norm.ppf(pa / 2)) if 0 < pa < 1 else 0.
                r = z / math.sqrt(len(x)) if len(x) > 0 else np.nan
                sig[(levels[i], levels[j])] = (pa < alpha)
                rows.append([f"{levels[i]} vs {levels[j]}", fmt(float(st), 3), fmt(pa, 4),
                             ("істотна різниця " + sig_mark(pa)) if pa < alpha else "–", fmt(r, 4)])
            except Exception: continue
    return rows, sig

# ═══════════════════════════════════════════════════════════════
# RCBD MATRIX
# ═══════════════════════════════════════════════════════════════
def rcbd_matrix(long, vnames, bnames, vk="VARIANT", bk="BLOCK"):
    bb = defaultdict(dict)
    for r in long:
        v = r.get("value", np.nan)
        if v is None or math.isnan(v): continue
        b = r.get(bk); vn = r.get(vk)
        if b is None or vn is None: continue
        bb[b][vn] = float(v)
    mat = []; kept = []
    for b in bnames:
        d = bb.get(b, {})
        if all(vn in d for vn in vnames): mat.append([d[vn] for vn in vnames]); kept.append(b)
    return mat, kept

# ═══════════════════════════════════════════════════════════════
# GLM / OLS — SS Types I / II / III
# ═══════════════════════════════════════════════════════════════
def _encode(col_vals, levels):
    cols, names = [], []
    for lv in levels[1:]:
        cols.append(np.array([1. if v == lv else 0. for v in col_vals], dtype=float))
        names.append(str(lv))
    return cols, names

def _build_X(long, fkeys, lbf, extra=None):
    n = len(long); y = np.array([float(r["value"]) for r in long], dtype=float)
    Xc = [np.ones(n, dtype=float)]; cn = ["Intercept"]; ts = {"Intercept": [0]}
    fdc = {}; fdn = {}
    for f in fkeys:
        vals = [r.get(f) for r in long]; cols, names = _encode(vals, lbf[f])
        fdc[f] = cols; fdn[f] = names
        if cols:
            idx = []
            for c, nm in zip(cols, names): Xc.append(c); cn.append(f"{f}:{nm}"); idx.append(len(Xc) - 1)
            ts[f"Фактор {f}"] = idx
        else: ts[f"Фактор {f}"] = []
    for r2 in range(2, len(fkeys) + 1):
        for cmb in combinations(fkeys, r2):
            lists = [fdc[f] for f in cmb]; nls = [fdn[f] for f in cmb]
            if any(len(L) == 0 for L in lists): ts["Фактор " + "×".join(cmb)] = []; continue
            idx = []
            def rec(i, cc, cn_, idx=idx, cmb=cmb, lists=lists, nls=nls):
                if i == len(lists):
                    Xc.append(cc); cn.append("×".join(f"{cmb[j]}:{cn_[j]}" for j in range(len(cmb)))); idx.append(len(Xc) - 1); return
                for ci, nm in zip(lists[i], nls[i]): rec(i + 1, (ci.copy() if cc is None else cc * ci), cn_ + [nm])
            rec(0, None, [])
            ts["Фактор " + "×".join(cmb)] = idx
    if extra:
        for nm, cols, coln in extra:
            idx = []
            for c, cn_ in zip(cols, coln): Xc.append(c); cn.append(f"{nm}:{cn_}"); idx.append(len(Xc) - 1)
            ts[nm] = idx
    X = np.column_stack(Xc); return y, X, ts, cn

def _ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None); yh = X @ beta; res = y - yh
    sse = float(np.sum(res ** 2)); n, p = X.shape; dfe = n - p
    return beta, yh, res, sse, dfe, (sse / dfe if dfe > 0 else np.nan)

def _ss_type3(y, Xf, ts):
    """Type III (partial) SS — drop-term approach."""
    _, _, res, sse, dfe, mse = _ols(y, Xf); out = {}
    for term, idx in ts.items():
        if term == "Intercept": continue
        if not idx: out[term] = (np.nan, 0, np.nan, np.nan, np.nan); continue
        keep = [i for i in range(Xf.shape[1]) if i not in idx]
        Xr = Xf[:, keep]; _, _, _, sse_r, _, _ = _ols(y, Xr)
        ss = float(sse_r - sse); df = len(idx); ms = ss / df if df > 0 else np.nan
        F = (ms / mse) if (df > 0 and not math.isnan(mse) and mse > 0) else np.nan
        p = float(1 - f_dist.cdf(F, df, dfe)) if (not math.isnan(F) and dfe > 0) else np.nan
        out[term] = (ss, df, ms, F, p)
    return out, sse, dfe, mse, res

def _ss_type1(y, Xf, ts, fkeys):
    """Type I (sequential) SS."""
    _, _, _, sse_full, dfe, mse = _ols(y, Xf); out = {}
    # build sequence: intercept, then each term in order
    seq_terms = ["Intercept"] + [f"Фактор {f}" for f in fkeys]
    for r2 in range(2, len(fkeys) + 1):
        for cmb in combinations(fkeys, r2): seq_terms.append("Фактор " + "×".join(cmb))
    included = [0]  # intercept always included
    prev_sse = float(np.sum((y - np.mean(y)) ** 2))
    for term in seq_terms[1:]:
        idx = ts.get(term, [])
        if not idx: out[term] = (np.nan, 0, np.nan, np.nan, np.nan); continue
        new_cols = included + idx
        Xr = Xf[:, new_cols]; _, _, _, sse_r, _, _ = _ols(y, Xr)
        ss = float(prev_sse - sse_r); df = len(idx); ms = ss / df if df > 0 else np.nan
        F = (ms / mse) if (df > 0 and not math.isnan(mse) and mse > 0) else np.nan
        p = float(1 - f_dist.cdf(F, df, dfe)) if (not math.isnan(F) and dfe > 0) else np.nan
        out[term] = (ss, df, ms, F, p)
        prev_sse = sse_r; included += idx
    return out, sse_full, dfe, mse, _ols(y, Xf)[2]

def _ss_type2(y, Xf, ts, fkeys):
    """Type II (hierarchical) SS — each factor adjusted for all other main effects."""
    _, _, res, sse_full, dfe, mse = _ols(y, Xf); out = {}
    # For each main effect: remove it from model with all other main effects but no interactions
    main_terms = {f: ts.get(f"Фактор {f}", []) for f in fkeys}
    inter_terms = {}
    for r2 in range(2, len(fkeys) + 1):
        for cmb in combinations(fkeys, r2):
            inter_terms["Фактор " + "×".join(cmb)] = ts.get("Фактор " + "×".join(cmb), [])
    for term, idx in ts.items():
        if term == "Intercept": continue
        if not idx: out[term] = (np.nan, 0, np.nan, np.nan, np.nan); continue
        # keep: all columns except those in current term
        # But also keep all higher-order terms that don't contain this factor
        # Type II: remove current term from "all terms of same or lower order"
        keep = [i for i in range(Xf.shape[1]) if i not in idx]
        Xr = Xf[:, keep]; _, _, _, sse_r, _, _ = _ols(y, Xr)
        ss = float(sse_r - sse_full); df = len(idx); ms = ss / df if df > 0 else np.nan
        F = (ms / mse) if (df > 0 and not math.isnan(mse) and mse > 0) else np.nan
        p = float(1 - f_dist.cdf(F, df, dfe)) if (not math.isnan(F) and dfe > 0) else np.nan
        out[term] = (ss, df, ms, F, p)
    return out, sse_full, dfe, mse, res

def _ss_dispatch(ss_type, y, Xf, ts, fkeys):
    if ss_type == "I":    return _ss_type1(y, Xf, ts, fkeys)
    elif ss_type == "II": return _ss_type2(y, Xf, ts, fkeys)
    elif ss_type == "IV": return _ss_type4(y, Xf, ts, fkeys)
    else:                 return _ss_type3(y, Xf, ts)   # III default

def _ss_type4(y, Xf, ts, fkeys):
    """
    Type IV SS — for unbalanced designs with empty cells.
    Uses estimable contrasts: for each term, we compare the full model
    against the model where that term's columns are zeroed-out (projection approach).
    For balanced data this equals Type III. For missing cells it avoids
    non-estimable functions by operating only on estimable contrasts.
    """
    _, _, res, sse_full, dfe, mse = _ols(y, Xf); out = {}
    n = Xf.shape[0]
    hat = Xf @ np.linalg.pinv(Xf)   # hat matrix H = X(X'X)⁻X'
    for term, idx in ts.items():
        if term == "Intercept": continue
        if not idx: out[term] = (np.nan, 0, np.nan, np.nan, np.nan); continue
        # Build contrast: columns in `idx` projected onto estimable space
        C = Xf[:, idx]
        # Estimable part: C_est = H @ C  (project onto column space of X)
        C_est = hat @ C
        # SS via general linear hypothesis: SS = y'C(C'C)⁻C'y  projected
        try:
            CTC = C_est.T @ C_est
            CTC_inv = np.linalg.pinv(CTC)
            _, _, res_full, sse_f, dfe_f, mse_f = _ols(y, Xf)
            beta, *_ = np.linalg.lstsq(Xf, y, rcond=None)
            Cb = C_est.T @ (Xf @ beta)
            ss = float(Cb.T @ CTC_inv @ Cb)
            df = len(idx); ms = ss / df if df > 0 else np.nan
            F = (ms / mse) if (df > 0 and not math.isnan(mse) and mse > 0) else np.nan
            p = float(1 - f_dist.cdf(F, df, dfe)) if (not math.isnan(F) and dfe > 0) else np.nan
            out[term] = (ss, df, ms, F, p)
        except Exception:
            out[term] = (np.nan, 0, np.nan, np.nan, np.nan)
    return out, sse_full, dfe, mse, res

def _block_dum(long, bk="BLOCK"):
    blocks = first_seen([r.get(bk) for r in long if r.get(bk) is not None])
    if not blocks: return [], [], blocks
    vals = [r.get(bk) for r in long]; cols = []; names = []
    for b in blocks[1:]:
        cols.append(np.array([1. if v == b else 0. for v in vals], dtype=float)); names.append(str(b))
    return cols, names, blocks

def _nir05(long, fkeys, mse, dfe, lbf):
    nir = {}
    if math.isnan(mse) or dfe <= 0: return nir
    tc = float(t_dist.ppf(1 - ALPHA / 2, int(dfe)))
    for f in fkeys:
        nl = defaultdict(int)
        for r in long:
            v = r.get("value", np.nan)
            if v is None or math.isnan(v): continue
            if r.get(f): nl[r[f]] += 1
        ns = [n for n in nl.values() if n > 0]
        if ns: nir[f"Фактор {f}"] = tc * math.sqrt(2 * mse / (len(ns) / sum(1 / n for n in ns)))
    nc = defaultdict(int)
    for r in long:
        v = r.get("value", np.nan)
        if v is None or math.isnan(v): continue
        nc[tuple(r.get(f) for f in fkeys)] += 1
    ns = [n for n in nc.values() if n > 0]
    if ns: nir["Загальна"] = tc * math.sqrt(2 * mse / (len(ns) / sum(1 / n for n in ns)))
    return nir

def build_eff_rows(table):
    """
    Effect-strength table (% of SS_total).
    Includes: factors, interactions, Residual.
    Excludes: Total row (it IS 100% by definition — redundant).
    Factors + Interactions + Residual = 100%.
    """
    ss_tot = 0.
    for row in table:
        if row[0] == "Загальна" and row[1] is not None \
                and not (isinstance(row[1], float) and math.isnan(row[1])):
            ss_tot = float(row[1]); break
    if ss_tot <= 0:
        # fallback: sum all non-nan SS values except Total
        ss_tot = sum(float(r[1]) for r in table
                     if r[1] is not None
                     and not (isinstance(r[1], float) and math.isnan(r[1]))
                     and r[0] != "Загальна")
    out = []
    for row in table:
        nm, SSv = row[0], row[1]
        if nm == "Загальна": continue          # skip Total — always 100%, not informative
        if SSv is None or (isinstance(SSv, float) and math.isnan(SSv)): continue
        # skip WP-error row — internal split-plot term, not a "real" source
        if "WP-error" in str(nm): continue
        pct = (float(SSv) / ss_tot * 100) if ss_tot > 0 else np.nan
        out.append([nm, fmt(pct, 2)])
    return out

def build_pe2_rows(table):
    ss_err = np.nan
    for row in table:
        if row[0].startswith("Залишок"): ss_err = row[1]; break
    out = []
    for row in table:
        nm, SSv = row[0], row[1]
        if nm.startswith("Залишок") or nm == "Загальна": continue
        if SSv is None or (isinstance(SSv, float) and math.isnan(SSv)): continue
        if isinstance(ss_err, float) and math.isnan(ss_err): continue
        d = float(SSv) + float(ss_err); pe2 = float(SSv) / d if d > 0 else np.nan
        out.append([nm, fmt(pe2, 4), eta2_label(pe2)])
    return out

# ═══════════════════════════════════════════════════════════════
# ANOVA MODELS
# ═══════════════════════════════════════════════════════════════
def anova_crd(long, fkeys, lbf, ss_type="III"):
    y, X, ts, _ = _build_X(long, fkeys, lbf)
    terms, sse, dfe, mse, res = _ss_dispatch(ss_type, y, X, ts, fkeys)
    sst = float(np.sum((y - np.mean(y)) ** 2))
    ord_ = [f"Фактор {f}" for f in fkeys]
    for r2 in range(2, len(fkeys) + 1):
        for c in combinations(fkeys, r2): ord_.append("Фактор " + "×".join(c))
    table = [[nm, *terms.get(nm, (np.nan, 0, np.nan, np.nan, np.nan))] for nm in ord_]
    table.append(["Залишок", sse, dfe, mse, np.nan, np.nan])
    table.append(["Загальна", sst, len(y) - 1, np.nan, np.nan, np.nan])
    return {"table": table, "SS_error": sse, "df_error": dfe, "MS_error": mse,
            "SS_total": sst, "residuals": res.tolist(), "NIR05": _nir05(long, fkeys, mse, dfe, lbf)}

def anova_rcbd(long, fkeys, lbf, bk="BLOCK", ss_type="III"):
    bc, bn, _ = _block_dum(long, bk)
    extra = [("Блоки", bc, bn)] if bc else []
    y, X, ts, _ = _build_X(long, fkeys, lbf, extra)
    terms, sse, dfe, mse, res = _ss_dispatch(ss_type, y, X, ts, fkeys)
    sst = float(np.sum((y - np.mean(y)) ** 2))
    table = []
    if bc: table.append(["Блоки", *terms.get("Блоки", (np.nan, 0, np.nan, np.nan, np.nan))])
    ord_ = [f"Фактор {f}" for f in fkeys]
    for r2 in range(2, len(fkeys) + 1):
        for c in combinations(fkeys, r2): ord_.append("Фактор " + "×".join(c))
    for nm in ord_: table.append([nm, *terms.get(nm, (np.nan, 0, np.nan, np.nan, np.nan))])
    table.append(["Залишок", sse, dfe, mse, np.nan, np.nan])
    table.append(["Загальна", sst, len(y) - 1, np.nan, np.nan, np.nan])
    return {"table": table, "SS_error": sse, "df_error": dfe, "MS_error": mse,
            "SS_total": sst, "residuals": res.tolist(), "NIR05": _nir05(long, fkeys, mse, dfe, lbf)}

def anova_split(long, fkeys, main_f, bk="BLOCK", ss_type="III"):
    if main_f not in fkeys: main_f = fkeys[0]
    bc, bn, _ = _block_dum(long, bk)
    ml = first_seen([r.get(main_f) for r in long if r.get(main_f) is not None])
    if len(ml) < 2: raise ValueError("Головний фактор має мати ≥ 2 рівні")
    mv = [r.get(main_f) for r in long]; mc, mn = _encode(mv, ml)
    wpc = []; wpn = []
    for bi, bc_ in enumerate(bc):
        for mi, mc_ in enumerate(mc): wpc.append(bc_ * mc_); wpn.append(f"{bn[bi]}×{mn[mi]}")
    extra = []
    if bc: extra.append(("Блоки", bc, bn))
    wt = f"WP-error(Блоки×{main_f})"
    if wpc: extra.append((wt, wpc, wpn))
    lbf = {f: first_seen([r.get(f) for r in long if r.get(f) is not None]) for f in fkeys}
    y, X, ts, _ = _build_X(long, fkeys, lbf, extra)
    _, _, res, sse, dfe, mse = _ols(y, X)
    wp_idx = ts.get(wt, [])
    if not wp_idx: raise ValueError("Неможливо побудувати whole-plot error")
    keep = [i for i in range(X.shape[1]) if i not in wp_idx]
    _, _, _, sse_r, _, _ = _ols(y, X[:, keep])
    ss_wp = float(sse_r - sse); df_wp = len(wp_idx); ms_wp = ss_wp / df_wp if df_wp > 0 else np.nan
    sst = float(np.sum((y - np.mean(y)) ** 2))
    terms, _, _, _, _ = _ss_dispatch(ss_type, y, X, ts, fkeys)
    table = []
    if bc: table.append(["Блоки", *terms.get("Блоки", (np.nan, 0, np.nan, np.nan, np.nan))[:4], np.nan, np.nan])
    table.append([wt, ss_wp, df_wp, ms_wp, np.nan, np.nan])
    ord_ = [f"Фактор {f}" for f in fkeys]
    for r2 in range(2, len(fkeys) + 1):
        for c in combinations(fkeys, r2): ord_.append("Фактор " + "×".join(c))
    for nm in ord_:
        ss, df, ms, F, p = terms.get(nm, (np.nan, 0, np.nan, np.nan, np.nan))
        if nm == f"Фактор {main_f}":
            F2 = (ms / ms_wp) if (df > 0 and not any(math.isnan(x) for x in [ms, ms_wp]) and ms_wp > 0) else np.nan
            p2 = float(1 - f_dist.cdf(F2, df, df_wp)) if (not math.isnan(F2) and df_wp > 0) else np.nan
        else:
            F2 = (ms / mse) if (df > 0 and not any(math.isnan(x) for x in [ms, mse]) and mse > 0) else np.nan
            p2 = float(1 - f_dist.cdf(F2, df, dfe)) if (not math.isnan(F2) and dfe > 0) else np.nan
        table.append([nm, ss, df, ms, F2, p2])
    table.append(["Залишок", sse, dfe, mse, np.nan, np.nan])
    table.append(["Загальна", sst, len(y) - 1, np.nan, np.nan, np.nan])
    nir = {}
    if not (math.isnan(mse) or dfe <= 0 or math.isnan(ms_wp) or df_wp <= 0):
        def nh_f(f):
            nl = defaultdict(int)
            for r in long:
                v = r.get("value", np.nan)
                if v is None or math.isnan(v): continue
                if r.get(f): nl[r[f]] += 1
            ns = [n for n in nl.values() if n > 0]
            return (len(ns) / sum(1 / n for n in ns)) if ns else np.nan
        tc_s = float(t_dist.ppf(1 - ALPHA / 2, int(dfe))) if dfe > 0 else np.nan
        tc_w = float(t_dist.ppf(1 - ALPHA / 2, int(df_wp))) if df_wp > 0 else np.nan
        for f in fkeys:
            nh = nh_f(f)
            if math.isnan(nh) or nh <= 0: continue
            if f == main_f: nir[f"Фактор {f}(WP)"] = tc_w * math.sqrt(2 * ms_wp / nh)
            else: nir[f"Фактор {f}"] = tc_s * math.sqrt(2 * mse / nh)
    return {"table": table, "SS_error": sse, "df_error": dfe, "MS_error": mse,
            "SS_total": sst, "residuals": res.tolist(),
            "MS_whole": ms_wp, "df_whole": df_wp, "main_factor": main_f, "NIR05": nir}

# ═══════════════════════════════════════════════════════════════
# PROJECT SAVE / LOAD
# ═══════════════════════════════════════════════════════════════
def project_to_dict(app):
    rows = [[e.get() for e in row] for row in app.entries]
    return {"version": APP_VER, "factors_count": app.factors_count,
            "factor_title_map": app.factor_title_map, "cols": app.cols, "rows_data": rows}

def project_from_dict(app, d):
    fc = d.get("factors_count", 1)
    app.open_table(fc)
    app.factor_title_map = d.get("factor_title_map", {})
    for j, fk in enumerate(app.factor_keys):
        t = app.factor_title_map.get(fk, f"Фактор {fk}")
        if j < len(app.header_labels): app.header_labels[j].configure(text=t)
    sc = d.get("cols", app.cols)
    while app.cols < sc: app.add_column()
    rd = d.get("rows_data", [])
    while len(app.entries) < len(rd): app.add_row()
    for i, rv in enumerate(rd):
        for j, v in enumerate(rv):
            if i < len(app.entries) and j < len(app.entries[i]):
                app.entries[i][j].delete(0, tk.END); app.entries[i][j].insert(0, v)

# ═══════════════════════════════════════════════════════════════
# GRAPH SETTINGS
# ═══════════════════════════════════════════════════════════════
DEF_GS = {
    "font_family": "Times New Roman", "font_style": "normal", "font_size": 11,
    "box_color": "#ffffff", "median_color": "#c62828",
    "whisker_color": "#000000", "flier_color": "#555555",
    "venn_colors": ["#4c72b0", "#dd8452", "#55a868", "#c44e52"],
    "venn_alpha": 0.45, "venn_font_size": 11, "venn_font_color": "#000000",
    "heatmap_cmap": "RdYlGn", "heatmap_font_size": 10, "heatmap_annot_color": "#000000",
}

# ═══════════════════════════════════════════════════════════════
# GRAPH SETTINGS DIALOG
# ═══════════════════════════════════════════════════════════════
class GraphSettingsDlg(tk.Toplevel):
    FONTS  = ["Times New Roman", "Arial", "Calibri", "Georgia", "Verdana", "Courier New"]
    STYLES = ["normal", "bold", "italic", "bold italic"]
    CMAPS  = ["RdYlGn", "coolwarm", "RdBu", "PiYG", "PRGn", "bwr", "seismic", "viridis", "plasma"]

    def __init__(self, parent, gs: dict, show_heatmap=False):
        super().__init__(parent)
        self.title("Налаштування графіків")
        self.resizable(False, False); set_icon(self)
        self.gs = dict(gs); self.result = None
        self._col_box = gs["box_color"]; self._col_med = gs["median_color"]
        self._col_wh  = gs["whisker_color"]; self._col_fl = gs["flier_color"]
        self._venn_fc = gs["venn_font_color"]; self._venn_cols = list(gs["venn_colors"])

        nb = ttk.Notebook(self); nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # ── Boxplot ──
        bp = tk.Frame(nb, padx=12, pady=10); nb.add(bp, text="Boxplot")
        self._ff = tk.StringVar(value=gs["font_family"])
        self._fs = tk.StringVar(value=gs["font_style"])
        self._fz = tk.IntVar(value=gs["font_size"])
        r = 0
        for lbl, var, vals in [("Шрифт:", self._ff, self.FONTS), ("Стиль:", self._fs, self.STYLES)]:
            tk.Label(bp, text=lbl).grid(row=r, column=0, sticky="w", pady=4)
            ttk.Combobox(bp, textvariable=var, values=vals, state="readonly", width=22).grid(row=r, column=1, sticky="w", padx=6); r += 1
        tk.Label(bp, text="Розмір:").grid(row=r, column=0, sticky="w", pady=4)
        tk.Spinbox(bp, from_=7, to=28, textvariable=self._fz, width=6).grid(row=r, column=1, sticky="w", padx=6); r += 1
        self._bp_btns = {}
        for lbl, attr in [("Колір коробки:", "_col_box"), ("Колір медіани:", "_col_med"),
                           ("Колір вусів:", "_col_wh"), ("Колір викидів:", "_col_fl")]:
            tk.Label(bp, text=lbl).grid(row=r, column=0, sticky="w", pady=4)
            btn = tk.Button(bp, width=6, relief=tk.SUNKEN, bg=getattr(self, attr),
                            command=lambda a=attr: self._pick(a))
            btn.grid(row=r, column=1, sticky="w", padx=6); self._bp_btns[attr] = btn; r += 1

        # ── Venn ──
        vf = tk.Frame(nb, padx=12, pady=10); nb.add(vf, text="Діаграма Венна")
        self._vff = tk.StringVar(value=gs["font_family"])
        self._vfz = tk.IntVar(value=gs["venn_font_size"])
        self._valpha = tk.DoubleVar(value=gs["venn_alpha"])
        r = 0
        tk.Label(vf, text="Шрифт:").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(vf, textvariable=self._vff, values=self.FONTS, state="readonly", width=22).grid(row=r, column=1, sticky="w", padx=6); r += 1
        tk.Label(vf, text="Розмір:").grid(row=r, column=0, sticky="w", pady=4)
        tk.Spinbox(vf, from_=7, to=28, textvariable=self._vfz, width=6).grid(row=r, column=1, sticky="w", padx=6); r += 1
        tk.Label(vf, text="Прозорість:").grid(row=r, column=0, sticky="w", pady=4)
        tk.Scale(vf, from_=0., to=1., resolution=0.05, orient="horizontal",
                 variable=self._valpha, length=180).grid(row=r, column=1, sticky="w", padx=6); r += 1
        tk.Label(vf, text="Колір тексту:").grid(row=r, column=0, sticky="w", pady=4)
        self._vfc_btn = tk.Button(vf, width=6, relief=tk.SUNKEN, bg=self._venn_fc, command=self._pick_vfc)
        self._vfc_btn.grid(row=r, column=1, sticky="w", padx=6); r += 1
        tk.Label(vf, text="Кольори кіл:").grid(row=r, column=0, sticky="w", pady=4)
        bf_ = tk.Frame(vf); bf_.grid(row=r, column=1, sticky="w", padx=6)
        self._vci_btns = []
        for idx in range(4):
            b = tk.Button(bf_, width=4, relief=tk.SUNKEN, bg=self._venn_cols[idx],
                          command=lambda i=idx: self._pick_vci(i))
            b.pack(side=tk.LEFT, padx=2); self._vci_btns.append(b)

        # ── Heatmap ──
        if show_heatmap:
            hf = tk.Frame(nb, padx=12, pady=10); nb.add(hf, text="Теплова карта")
            self._hcmap = tk.StringVar(value=gs.get("heatmap_cmap", "RdYlGn"))
            self._hfz   = tk.IntVar(value=gs.get("heatmap_font_size", 10))
            self._hannot_col = gs.get("heatmap_annot_color", "#000000")
            r = 0
            tk.Label(hf, text="Палітра:").grid(row=r, column=0, sticky="w", pady=4)
            ttk.Combobox(hf, textvariable=self._hcmap, values=self.CMAPS, state="readonly", width=18).grid(row=r, column=1, sticky="w", padx=6); r += 1
            tk.Label(hf, text="Розмір шрифту:").grid(row=r, column=0, sticky="w", pady=4)
            tk.Spinbox(hf, from_=6, to=20, textvariable=self._hfz, width=6).grid(row=r, column=1, sticky="w", padx=6); r += 1
            tk.Label(hf, text="Колір анотацій:").grid(row=r, column=0, sticky="w", pady=4)
            self._hannot_btn = tk.Button(hf, width=6, relief=tk.SUNKEN, bg=self._hannot_col, command=self._pick_hannot)
            self._hannot_btn.grid(row=r, column=1, sticky="w", padx=6)
        else:
            self._hcmap = tk.StringVar(value=gs.get("heatmap_cmap", "RdYlGn"))
            self._hfz   = tk.IntVar(value=gs.get("heatmap_font_size", 10))
            self._hannot_col = gs.get("heatmap_annot_color", "#000000")

        bf2 = tk.Frame(self); bf2.pack(fill=tk.X, padx=10, pady=(0, 10))
        tk.Button(bf2, text="OK", width=10, command=self._ok).pack(side=tk.LEFT, padx=4)
        tk.Button(bf2, text="Скасувати", width=12, command=self.destroy).pack(side=tk.LEFT)
        self.update_idletasks(); center_win(self); self.grab_set()

    def _pick(self, attr):
        c = colorchooser.askcolor(color=getattr(self, attr), parent=self, title="Виберіть колір")
        if c and c[1]:
            setattr(self, attr, c[1])
            if attr in self._bp_btns: self._bp_btns[attr].configure(bg=c[1])

    def _pick_vfc(self):
        c = colorchooser.askcolor(color=self._venn_fc, parent=self, title="Колір тексту")
        if c and c[1]: self._venn_fc = c[1]; self._vfc_btn.configure(bg=c[1])

    def _pick_vci(self, idx):
        c = colorchooser.askcolor(color=self._venn_cols[idx], parent=self, title=f"Колір кола {idx+1}")
        if c and c[1]: self._venn_cols[idx] = c[1]; self._vci_btns[idx].configure(bg=c[1])

    def _pick_hannot(self):
        c = colorchooser.askcolor(color=self._hannot_col, parent=self, title="Колір анотацій")
        if c and c[1]: self._hannot_col = c[1]; self._hannot_btn.configure(bg=c[1])

    def _ok(self):
        self.result = {
            "font_family": self._ff.get(), "font_style": self._fs.get(), "font_size": self._fz.get(),
            "box_color": self._col_box, "median_color": self._col_med,
            "whisker_color": self._col_wh, "flier_color": self._col_fl,
            "venn_colors": list(self._venn_cols), "venn_alpha": float(self._valpha.get()),
            "venn_font_size": self._vfz.get(), "venn_font_color": self._venn_fc,
            "heatmap_cmap": self._hcmap.get(), "heatmap_font_size": self._hfz.get(),
            "heatmap_annot_color": self._hannot_col,
        }
        self.destroy()

# ═══════════════════════════════════════════════════════════════
# VENN DIAGRAM  — proper overlap layout
# ═══════════════════════════════════════════════════════════════
def draw_venn(ax, factor_values, interaction_values, colors, alpha, font_size, font_color, font_family, title=""):
    """
    Proportional Venn diagram.
    factor_values: list of (label, pct) for main factors (circles)
    interaction_values: dict {frozenset_of_indices: (label, pct)} for interactions (overlaps)
    """
    ax.set_aspect("equal"); ax.axis("off")
    n = len(factor_values)
    if n == 0: return

    # Circle centres layout
    import math as _m
    if n == 1:
        centres = [(0, 0)]
    elif n == 2:
        centres = [(-0.33, 0), (0.33, 0)]
    elif n == 3:
        r_ = 0.38
        centres = [(r_ * _m.cos(_m.radians(90 + 120 * i)),
                    r_ * _m.sin(_m.radians(90 + 120 * i))) for i in range(3)]
    else:
        r_ = 0.42
        centres = [(r_ * _m.cos(_m.radians(45 + 90 * i)),
                    r_ * _m.sin(_m.radians(45 + 90 * i))) for i in range(4)]

    radius = 0.40

    # Draw circles
    for i, (cx, cy) in enumerate(centres):
        circle = mpatches.Circle((cx, cy), radius, fc=colors[i % len(colors)],
                                  alpha=alpha, ec="#444444", lw=1.2, zorder=2)
        ax.add_patch(circle)

    # Label main factors — place outside circle
    for i, ((cx, cy), (lbl, pct)) in enumerate(zip(centres, factor_values)):
        # offset outward from centre of diagram
        dx = cx - 0; dy = cy - 0
        mag = max(_m.hypot(dx, dy), 0.01)
        ox = cx + (dx / mag) * 0.55; oy = cy + (dy / mag) * 0.55
        if n == 1: ox, oy = 0, 0.65
        ax.text(ox, oy, f"{lbl}\n{fmt(pct, 1)}%",
                ha="center", va="center", fontsize=font_size,
                color=font_color, fontfamily=font_family,
                fontweight="bold", linespacing=1.3, zorder=5)

    # Label interactions at geometric intersections
    for key, (lbl, pct) in (interaction_values or {}).items():
        idxs = sorted(list(key))
        if len(idxs) < 2 or max(idxs) >= n: continue
        # mean of involved centres
        mx = sum(centres[i][0] for i in idxs) / len(idxs)
        my = sum(centres[i][1] for i in idxs) / len(idxs)
        ax.text(mx, my, f"{fmt(pct, 1)}%",
                ha="center", va="center", fontsize=max(font_size - 1, 7),
                color=font_color, fontfamily=font_family,
                fontweight="bold", zorder=6)

    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    if title:
        ax.set_title(title, fontsize=font_size + 1, fontfamily=font_family, pad=8)


# ═══════════════════════════════════════════════════════════════
# CORRELATION ANALYSIS WINDOW
# ═══════════════════════════════════════════════════════════════
class CorrelationWindow:
    def __init__(self, root, graph_settings):
        self.root = root
        self.gs = dict(graph_settings)
        self._figs = []

        self.win = tk.Toplevel(root)
        self.win.title("Кореляційний аналіз"); self.win.geometry("1100x680"); set_icon(self.win)

        self._build_menu()
        self._build_toolbar()
        self._build_table()

    def _build_menu(self):
        mb = tk.Menu(self.win)
        fm = tk.Menu(mb, tearoff=0)
        fm.add_command(label="Зберегти проект", command=self._save_proj)
        fm.add_command(label="Відкрити проект", command=self._load_proj)
        fm.add_separator()
        fm.add_command(label="Завантажити Excel", command=self._load_excel)
        mb.add_cascade(label="Файл", menu=fm)
        em = tk.Menu(mb, tearoff=0)
        em.add_command(label="Додати рядок", command=self.add_row)
        em.add_command(label="Видалити рядок", command=self.del_row)
        em.add_command(label="Додати стовпчик", command=self.add_col)
        em.add_command(label="Видалити стовпчик", command=self.del_col)
        mb.add_cascade(label="Правка", menu=em)
        self.win.config(menu=mb)

    def _build_toolbar(self):
        tb = tk.Frame(self.win, padx=6, pady=4); tb.pack(fill=tk.X)
        tk.Button(tb, text="Аналіз", bg="#c62828", fg="white",
                  command=self._run_analysis).pack(side=tk.LEFT, padx=4)
        tk.Button(tb, text="⚙ Налаштування", command=self._settings).pack(side=tk.LEFT, padx=4)
        tk.Button(tb, text="Вставити (буфер)", command=self._paste).pack(side=tk.LEFT, padx=4)

    def _build_table(self):
        self.rows = 12; self.cols = 6
        self.canvas = tk.Canvas(self.win); self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(self.win, orient="vertical", command=self.canvas.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y); self.canvas.configure(yscrollcommand=sb.set)
        self.inner = tk.Frame(self.canvas); self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: self.canvas.config(scrollregion=self.canvas.bbox("all")))

        self.header_labels = []
        for j in range(self.cols):
            lbl = tk.Label(self.inner, text=f"Показник {j+1}", relief=tk.RIDGE, width=14,
                           bg="#f0f0f0", fg="#000000", font=("Times New Roman", 11))
            lbl.grid(row=0, column=j, padx=2, pady=2, sticky="nsew")
            self.header_labels.append(lbl)

        self.entries = []
        for i in range(self.rows):
            row_ = []
            for j in range(self.cols):
                e = tk.Entry(self.inner, width=14, font=("Times New Roman", 11),
                             highlightthickness=1, highlightbackground="#c0c0c0")
                e.grid(row=i + 1, column=j, padx=2, pady=2)
                e.bind("<Return>", self._on_enter)
                e.bind("<Tab>", self._on_tab)
                row_.append(e)
            self.entries.append(row_)

    def add_row(self):
        i = len(self.entries); row_ = []
        for j in range(self.cols):
            e = tk.Entry(self.inner, width=14, font=("Times New Roman", 11),
                         highlightthickness=1, highlightbackground="#c0c0c0")
            e.grid(row=i + 1, column=j, padx=2, pady=2)
            e.bind("<Return>", self._on_enter)
            e.bind("<Tab>", self._on_tab)
            row_.append(e)
        self.entries.append(row_); self.rows += 1
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def del_row(self):
        if not self.entries: return
        for e in self.entries.pop(): e.destroy()
        self.rows -= 1; self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def add_col(self):
        self.cols += 1; ci = self.cols - 1
        lbl = tk.Label(self.inner, text=f"Показник {ci+1}", relief=tk.RIDGE, width=14,
                       bg="#f0f0f0", fg="#000000", font=("Times New Roman", 11))
        lbl.grid(row=0, column=ci, padx=2, pady=2, sticky="nsew"); self.header_labels.append(lbl)
        for i, row_ in enumerate(self.entries):
            e = tk.Entry(self.inner, width=14, font=("Times New Roman", 11),
                         highlightthickness=1, highlightbackground="#c0c0c0")
            e.grid(row=i + 1, column=ci, padx=2, pady=2)
            e.bind("<Return>", self._on_enter); e.bind("<Tab>", self._on_tab)
            row_.append(e)

    def del_col(self):
        if self.cols <= 2: return
        self.header_labels.pop().destroy()
        for row_ in self.entries: row_.pop().destroy()
        self.cols -= 1

    def _on_enter(self, event):
        for i, row_ in enumerate(self.entries):
            for j, e in enumerate(row_):
                if e is event.widget:
                    ni = i + 1
                    if ni >= len(self.entries): self.add_row()
                    self.entries[ni][j].focus_set(); return "break"
        return "break"

    def _on_tab(self, event):
        for i, row_ in enumerate(self.entries):
            for j, e in enumerate(row_):
                if e is event.widget:
                    nj = j + 1
                    if nj >= self.cols: nj = 0; ni = i + 1
                    else: ni = i
                    if ni >= len(self.entries): self.add_row()
                    self.entries[ni][nj].focus_set(); return "break"
        return "break"

    def _paste(self):
        w = self.win.focus_get()
        if not isinstance(w, tk.Entry): return
        try: data = self.win.clipboard_get()
        except Exception: return
        # find pos
        pos = None
        for i, row_ in enumerate(self.entries):
            for j, e in enumerate(row_):
                if e is w: pos = (i, j); break
            if pos: break
        if not pos: return
        r0, c0 = pos
        for ir, rt in enumerate(data.splitlines()):
            cols = rt.split("\t")
            for jc, val in enumerate(cols):
                rr = r0 + ir; cc = c0 + jc
                while rr >= len(self.entries): self.add_row()
                if cc >= self.cols: continue
                self.entries[rr][cc].delete(0, tk.END); self.entries[rr][cc].insert(0, val)

    def _save_proj(self):
        path = filedialog.asksaveasfilename(parent=self.win, defaultextension=".sadp",
                                             filetypes=[("SAD проект", "*.sadp"), ("JSON", "*.json")])
        if not path: return
        rows = [[e.get() for e in row] for row in self.entries]
        headers = [lbl.cget("text") for lbl in self.header_labels]
        d = {"type": "correlation", "version": APP_VER, "headers": headers, "rows_data": rows}
        try:
            with open(path, "w", encoding="utf-8") as f: json.dump(d, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("Збережено", path)
        except Exception as ex: messagebox.showerror("Помилка", str(ex))

    def _load_proj(self):
        path = filedialog.askopenfilename(parent=self.win,
                                           filetypes=[("SAD проект", "*.sadp"), ("JSON", "*.json")])
        if not path: return
        try:
            with open(path, "r", encoding="utf-8") as f: d = json.load(f)
        except Exception as ex: messagebox.showerror("Помилка", str(ex)); return
        headers = d.get("headers", [])
        rd = d.get("rows_data", [])
        while self.cols < len(headers): self.add_col()
        for j, h in enumerate(headers):
            if j < len(self.header_labels): self.header_labels[j].configure(text=h)
        while len(self.entries) < len(rd): self.add_row()
        for i, rv in enumerate(rd):
            for j, v in enumerate(rv):
                if i < len(self.entries) and j < len(self.entries[i]):
                    self.entries[i][j].delete(0, tk.END); self.entries[i][j].insert(0, v)

    def _load_excel(self):
        if not HAS_OPENPYXL: messagebox.showerror("", "pip install openpyxl"); return
        path = filedialog.askopenfilename(parent=self.win, filetypes=[("Excel", "*.xlsx *.xlsm *.xls")])
        if not path: return
        try:
            wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
            ws = wb.active
            raw = [[cell for cell in row] for row in ws.iter_rows(values_only=True)]
            wb.close()
        except Exception as ex: messagebox.showerror("", str(ex)); return
        while raw and all(v is None for v in raw[-1]): raw.pop()
        if not raw: return
        nc = max(len(r) for r in raw)
        while self.cols < nc: self.add_col()
        while len(self.entries) < len(raw): self.add_row()
        for i, row in enumerate(raw):
            for j, v in enumerate(row):
                if j >= self.cols: break
                cv = "" if v is None else str(v).replace(",", ".")
                self.entries[i][j].delete(0, tk.END); self.entries[i][j].insert(0, cv)

    def _settings(self):
        dlg = GraphSettingsDlg(self.win, self.gs, show_heatmap=True)
        self.win.wait_window(dlg)
        if dlg.result: self.gs = dlg.result

    def _run_analysis(self):
        """Ask params and run correlation."""
        dlg = tk.Toplevel(self.win); dlg.title("Параметри аналізу"); dlg.resizable(False, False); set_icon(dlg)
        frm = tk.Frame(dlg, padx=14, pady=12); frm.pack()

        tk.Label(frm, text="Назви показників:", font=("Times New Roman", 12)).grid(row=0, column=0, sticky="w", pady=4)
        names_var = tk.StringVar(value="row")
        rb_f = ("Times New Roman", 12)
        tk.Radiobutton(frm, text="Перший рядок кожного стовпця", variable=names_var,
                       value="row", font=rb_f).grid(row=1, column=0, sticky="w")
        tk.Radiobutton(frm, text="Перша колонка (рядки = показники)", variable=names_var,
                       value="col", font=rb_f).grid(row=2, column=0, sticky="w")

        tk.Label(frm, text="Метод кореляції:", font=("Times New Roman", 12)).grid(row=3, column=0, sticky="w", pady=(10, 4))
        meth_var = tk.StringVar(value="auto")
        tk.Radiobutton(frm, text="Авто (перевірка нормальності → Pearson або Spearman)",
                       variable=meth_var, value="auto", font=rb_f).grid(row=4, column=0, sticky="w")
        tk.Radiobutton(frm, text="Пірсона (лише якщо дані нормально розподілені)",
                       variable=meth_var, value="pearson", font=rb_f).grid(row=5, column=0, sticky="w")
        tk.Radiobutton(frm, text="Спірмена (непараметричний, завжди коректний)",
                       variable=meth_var, value="spearman", font=rb_f).grid(row=6, column=0, sticky="w")

        tk.Label(frm, text="Поправка на множинні порівняння:", font=("Times New Roman", 12)).grid(row=7, column=0, sticky="w", pady=(10, 4))
        corr_var = tk.StringVar(value="bonferroni")
        for txt, val in [("Бонферроні (суворіша)", "bonferroni"),
                          ("BH / FDR (Benjamini–Hochberg, ліберальніша)", "bh"),
                          ("Без поправки (не рекомендується)", "none")]:
            tk.Radiobutton(frm, text=txt, variable=corr_var, value=val, font=rb_f).grid(
                row=8 + [("bonferroni","bh","none").index(val)], column=0, sticky="w")

        tk.Label(frm, text="Рівень значущості α:", font=("Times New Roman", 12)).grid(row=11, column=0, sticky="w", pady=(10, 4))
        alpha_var = tk.StringVar(value="0.05")
        ttk.Combobox(frm, textvariable=alpha_var, values=["0.01", "0.05", "0.10"],
                     state="readonly", width=8).grid(row=11, column=1, sticky="w", padx=6)

        out = {"ok": False}
        def ok():
            out.update({"ok": True, "names_loc": names_var.get(),
                        "method": meth_var.get(), "correction": corr_var.get(),
                        "alpha": float(alpha_var.get())})
            dlg.destroy()
        bf = tk.Frame(frm); bf.grid(row=12, column=0, columnspan=2, pady=(12, 0))
        tk.Button(bf, text="OK", width=10, command=ok).pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text="Скасувати", width=12, command=dlg.destroy).pack(side=tk.LEFT)
        dlg.update_idletasks(); center_win(dlg); dlg.bind("<Return>", lambda e: ok())
        dlg.grab_set(); self.win.wait_window(dlg)
        if not out["ok"]: return
        self._compute_and_show(out["names_loc"], out["method"], out["alpha"], out["correction"])

    def _compute_and_show(self, names_loc, method, alpha, correction="bonferroni"):
        # ── Extract raw grid ──────────────────────────────────
        raw = [[e.get().strip() for e in row] for row in self.entries]
        raw = [r for r in raw if any(v for v in r)]
        if not raw: messagebox.showwarning("", "Немає даних."); return

        if names_loc == "col":
            labels = []; data_cols = []
            for row in raw:
                lbl = row[0] if row else ""
                if not lbl: continue
                vals = []
                for v in row[1:]:
                    if not v: continue
                    try: vals.append(float(v.replace(",", ".")))
                    except Exception: continue
                if len(vals) >= 3:
                    labels.append(lbl); data_cols.append(vals)
        else:
            n_cols = max(len(r) for r in raw) if raw else 0
            labels = []; data_cols = []
            for j in range(n_cols):
                col_name = ""; col_vals = []
                for row in raw:
                    v = row[j] if j < len(row) else ""
                    if not v: continue
                    if not col_name:
                        try:
                            float(v.replace(",", "."))
                            col_name = (self.header_labels[j].cget("text")
                                        if j < len(self.header_labels) else f"П{j+1}")
                            col_vals.append(float(v.replace(",", ".")))
                        except ValueError:
                            col_name = v
                    else:
                        try: col_vals.append(float(v.replace(",", ".")))
                        except Exception: continue
                if col_name and len(col_vals) >= 3:
                    labels.append(col_name); data_cols.append(col_vals)

        if len(data_cols) < 2:
            messagebox.showwarning("Замало даних",
                "Потрібно ≥ 2 показники з ≥ 3 значеннями кожен."); return

        n = len(labels)
        # ── Per-pair arrays (не вирівнюємо — використовуємо pairwise) ──
        arrays = [np.array(d, dtype=float) for d in data_cols]
        # matrix for pair lengths
        n_mat = np.zeros((n, n), dtype=int)

        # ── Авто-вибір методу: Shapiro-Wilk на кожному показнику ──
        actual_method = method
        if method == "auto":
            non_normal = []
            for i, arr in enumerate(arrays):
                if len(arr) < 3: continue
                try:
                    _, p_sw = shapiro(arr)
                    if p_sw <= 0.05: non_normal.append(labels[i])
                except Exception: pass
            if non_normal:
                detail = ", ".join(non_normal[:5])
                messagebox.showinfo("Авто-вибір методу",
                    f"Показники, що не відповідають нормальному розподілу:\n{detail}\n\n"
                    "Автоматично обрано метод Спірмена (непараметричний).")
                actual_method = "spearman"
            else:
                messagebox.showinfo("Авто-вибір методу",
                    "Усі показники відповідають нормальному розподілу.\n"
                    "Автоматично обрано метод Пірсона.")
                actual_method = "pearson"

        # ── Попередження якщо Pearson обраний вручну ──
        elif method == "pearson":
            non_normal = []
            for i, arr in enumerate(arrays):
                if len(arr) < 3: continue
                try:
                    _, p_sw = shapiro(arr)
                    if p_sw <= 0.05: non_normal.append(labels[i])
                except Exception: pass
            if non_normal:
                detail = ", ".join(non_normal[:5])
                ans = messagebox.askyesno("Увага: нормальність порушена",
                    f"Показники, що не відповідають нормальному розподілу:\n{detail}\n\n"
                    "Кореляція Пірсона передбачає нормальний розподіл обох змінних.\n"
                    "При порушенні цієї умови результат може бути ненадійним.\n\n"
                    "Рекомендація: використовуйте кореляцію Спірмена.\n\n"
                    "Продовжити з Пірсоном попри порушення умови?")
                if not ans: return

        # ── Побудова попарних матриць r та p ──────────────────
        r_mat = np.full((n, n), np.nan); p_mat = np.full((n, n), np.nan)
        np.fill_diagonal(r_mat, 1.0); np.fill_diagonal(p_mat, 1.0)
        np.fill_diagonal(n_mat, [len(a) for a in arrays])

        raw_p_pairs = []   # [(i, j, p_raw)] for correction
        for i in range(n):
            for j in range(i + 1, n):
                # pairwise complete observations
                a = arrays[i]; b = arrays[j]
                min_len = min(len(a), len(b))
                a2 = a[:min_len]; b2 = b[:min_len]
                # remove pairs where either is NaN
                mask = ~(np.isnan(a2) | np.isnan(b2))
                a2 = a2[mask]; b2 = b2[mask]
                pair_n = len(a2)
                n_mat[i, j] = n_mat[j, i] = pair_n
                if pair_n < 3:
                    continue
                try:
                    if actual_method == "pearson":
                        r_, p_ = pearsonr(a2, b2)
                    else:
                        r_, p_ = spearmanr(a2, b2)
                    r_mat[i, j] = r_mat[j, i] = float(r_)
                    raw_p_pairs.append((i, j, float(p_)))
                except Exception:
                    pass

        # ── Поправка на множинні порівняння ───────────────────
        m_tests = len(raw_p_pairs)
        if m_tests == 0:
            messagebox.showwarning("", "Жодної пари з достатньою кількістю даних."); return

        if correction == "bonferroni":
            for i, j, p_raw in raw_p_pairs:
                p_adj = min(1.0, p_raw * m_tests)
                p_mat[i, j] = p_mat[j, i] = p_adj
        elif correction == "bh":
            # Benjamini–Hochberg FDR
            sorted_pairs = sorted(raw_p_pairs, key=lambda x: x[2])
            p_adj_arr = np.array([p for _, _, p in sorted_pairs])
            m = len(p_adj_arr)
            # BH adjustment
            bh_adj = np.zeros(m)
            for k in range(m - 1, -1, -1):
                bh_adj[k] = min(1.0, p_adj_arr[k] * m / (k + 1))
                if k < m - 1: bh_adj[k] = min(bh_adj[k], bh_adj[k + 1])
            for idx, (i, j, _) in enumerate(sorted_pairs):
                p_mat[i, j] = p_mat[j, i] = float(bh_adj[idx])
        else:  # none
            for i, j, p_raw in raw_p_pairs:
                p_mat[i, j] = p_mat[j, i] = p_raw

        corr_label = {"bonferroni": "Бонферроні", "bh": "BH/FDR", "none": "без поправки"
                      }.get(correction, correction)
        self._show_heatmap(labels, r_mat, p_mat, n_mat, alpha, actual_method, corr_label)

    def _show_heatmap(self, labels, r_mat, p_mat, n_mat, alpha, method, corr_label=""):
        if not HAS_MPL: messagebox.showwarning("", "matplotlib недоступний."); return
        gs = self.gs
        win = tk.Toplevel(self.win); win.title("Теплова карта кореляцій")
        win.geometry("860x740"); set_icon(win)
        tb = tk.Frame(win, padx=6, pady=4); tb.pack(fill=tk.X)
        tk.Button(tb, text="⚙ Налаштування",
                  command=lambda: self._restyle(win, labels, r_mat, p_mat, n_mat, alpha, method, corr_label)
                  ).pack(side=tk.LEFT, padx=4)
        tk.Label(tb, text=f"Метод: {method.capitalize()}   |   Поправка: {corr_label}   |   α={alpha}",
                 font=("Times New Roman", 11), fg="#555").pack(side=tk.LEFT, padx=10)

        fig_frame = tk.Frame(win); fig_frame.pack(fill=tk.BOTH, expand=True)
        self._draw_heatmap(fig_frame, labels, r_mat, p_mat, n_mat, alpha, method, corr_label, gs)
        self._hm_data = (labels, r_mat, p_mat, n_mat, alpha, method, corr_label)
        self._hm_frame = fig_frame

    def _restyle(self, win, labels, r_mat, p_mat, n_mat, alpha, method, corr_label):
        dlg = GraphSettingsDlg(win, self.gs, show_heatmap=True)
        win.wait_window(dlg)
        if dlg.result:
            self.gs = dlg.result
            for w in self._hm_frame.winfo_children(): w.destroy()
            self._draw_heatmap(self._hm_frame, labels, r_mat, p_mat, n_mat, alpha, method, corr_label, self.gs)

    def _draw_heatmap(self, frame, labels, r_mat, p_mat, n_mat, alpha, method, corr_label, gs):
        n = len(labels)
        cell_size = max(1.0, min(1.4, 9.0 / n))
        fig_sz = max(5, n * cell_size + 1.8)
        fig = Figure(figsize=(min(fig_sz, 12), min(fig_sz, 12)), dpi=100)
        ax = fig.add_subplot(111)

        cmap_name = gs.get("heatmap_cmap", "RdYlGn")
        fsize  = gs.get("heatmap_font_size", 9)
        acol   = gs.get("heatmap_annot_color", "#000000")
        ff     = gs.get("font_family", "Times New Roman")

        try: cmap = matplotlib.cm.get_cmap(cmap_name)
        except Exception: cmap = matplotlib.cm.get_cmap("RdYlGn")

        masked = np.ma.array(r_mat, mask=np.isnan(r_mat))
        im = ax.imshow(masked, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=fsize, fontfamily=ff)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=fsize, fontfamily=ff)
        meth_full = "Пірсон" if method == "pearson" else "Спірмен"
        ax.set_title(f"Кореляційна матриця ({meth_full}, {corr_label}, α={alpha})\n"
                     f"У клітинках: r / p / n",
                     fontsize=fsize + 1, fontfamily=ff)

        for i in range(n):
            for j in range(n):
                r_ = r_mat[i, j]; p_ = p_mat[i, j]
                if i == j:
                    ax.text(j, i, "—", ha="center", va="center",
                            fontsize=fsize, color=acol, fontfamily=ff)
                    continue
                if math.isnan(r_): continue
                # significance mark on corrected p
                p_disp = p_ if not math.isnan(p_) else np.nan
                mark = sig_mark(p_disp) if not math.isnan(p_disp) else ""
                n_ij = int(n_mat[i, j]) if n_mat is not None else 0
                # format: r / p / n (3 lines)
                p_str = fmt(p_disp, 3) if not math.isnan(p_disp) else "н/д"
                cell_txt = f"{r_:.2f}{mark}\np={p_str}\nn={n_ij}"
                ax.text(j, i, cell_txt, ha="center", va="center",
                        fontsize=max(6, fsize - 1), color=acol, fontfamily=ff,
                        linespacing=1.3)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("r", fontsize=fsize, fontfamily=ff)
        fig.tight_layout()

        self._hm_fig = fig
        cv = FigureCanvasTkAgg(fig, master=frame); cv.draw()
        cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        def copy_hm():
            ok, msg = _copy_fig_to_clipboard(self._hm_fig)
            if ok: messagebox.showinfo("", "Скопійовано.")
            else:  messagebox.showwarning("", f"Помилка: {msg}")
        tk.Button(frame, text="📋 Копіювати PNG", command=copy_hm).pack(pady=4)


# ═══════════════════════════════════════════════════════════════
# GUI — SADTk
# ═══════════════════════════════════════════════════════════════
class SADTk:
    SEL_BG = "#cce5ff"; SEL_ANC = "#99ccff"; ACT_BG = "#fff3c4"

    def __init__(self, root):
        self.root = root
        root.title("S.A.D. — Статистичний аналіз даних")
        root.geometry("1000x580"); set_icon(root)
        try: ttk.Style().theme_use("clam")
        except Exception: pass

        mf = tk.Frame(root, bg="white"); mf.pack(expand=True, fill=tk.BOTH)
        tk.Label(mf, text="S.A.D. — Статистичний аналіз даних",
                 font=("Times New Roman", 20, "bold"), fg="#000000", bg="white").pack(pady=16)

        bf = tk.Frame(mf, bg="white"); bf.pack(pady=8)
        for i, (txt, fc) in enumerate([("Однофакторний аналіз", 1), ("Двофакторний аналіз", 2),
                                        ("Трифакторний аналіз", 3), ("Чотирифакторний аналіз", 4)]):
            tk.Button(bf, text=txt, width=22, height=2, font=("Times New Roman", 13),
                      command=lambda f=fc: self.open_table(f)).grid(row=i // 2, column=i % 2, padx=10, pady=6)

        tk.Button(mf, text="Кореляційний аналіз", width=30, height=2, font=("Times New Roman", 13),
                  bg="#1a4b8c", fg="white",
                  command=self.open_correlation).pack(pady=(4, 10))

        tk.Label(mf, text="Виберіть тип аналізу → Введіть дані → Аналіз даних",
                 font=("Times New Roman", 12), fg="#555555", bg="white").pack(pady=4)

        self.table_win = None; self.report_win = None; self.graph_win = None
        self._graph_figs = {}
        self._active_cell = None; self._active_prev = None
        self._sel_anchor = None; self._sel_cells = set(); self._sel_orig = {}
        self._fill_drag = False; self._fill_rows = []; self._fill_cols = []
        self.factor_title_map = {}
        self.graph_settings = dict(DEF_GS)
        self._current_project_path = None
        self._lbf_cache = {}

    # ── open correlation ──────────────────────────────────────
    def open_correlation(self):
        CorrelationWindow(self.root, self.graph_settings)

    # ── factor titles ─────────────────────────────────────────
    def ftitle(self, fk): return self.factor_title_map.get(fk, f"Фактор {fk}")
    def _set_ftitle(self, fk, t): self.factor_title_map[fk] = t.strip() or f"Фактор {fk}"

    # ── project save/load ─────────────────────────────────────
    def save_project(self):
        if not hasattr(self, "entries") or not self.entries:
            messagebox.showwarning("", "Відкрийте таблицю."); return
        path = filedialog.asksaveasfilename(
            parent=self.table_win or self.root, title="Зберегти проект",
            defaultextension=".sadp",
            filetypes=[("SAD проект", "*.sadp"), ("JSON", "*.json"), ("Усі", "*.*")])
        if not path: return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(project_to_dict(self), f, ensure_ascii=False, indent=2)
            self._current_project_path = path
            messagebox.showinfo("Збережено", f"Проект збережено:\n{path}")
        except Exception as ex: messagebox.showerror("Помилка", str(ex))

    def load_project(self):
        path = filedialog.askopenfilename(
            parent=self.table_win or self.root, title="Відкрити проект",
            filetypes=[("SAD проект", "*.sadp"), ("JSON", "*.json"), ("Усі", "*.*")])
        if not path: return
        try:
            with open(path, "r", encoding="utf-8") as f: d = json.load(f)
            project_from_dict(self, d)
            self._current_project_path = path
            messagebox.showinfo("Відкрито", f"Проект відкрито:\n{path}")
        except Exception as ex: messagebox.showerror("Помилка", str(ex))

    def clear_project(self):
        if not messagebox.askyesno("Очистити", "Очистити всі дані таблиці?"): return
        for row in self.entries:
            for e in row: e.delete(0, tk.END)

    # ── selection ─────────────────────────────────────────────
    def _clear_sel(self):
        # No visual highlighting — just clear tracking state
        self._sel_cells.clear(); self._sel_anchor = None; self._sel_orig.clear()

    def _restore_bg(self, r, c):
        pass   # no-op: no coloring was applied

    def _apply_sel(self, cells):
        pass   # no-op: selection tracking only, no visual highlight

    def _sel_range(self, r1, c1, r2, c2):
        new = {(r, c) for r in range(min(r1, r2), max(r1, r2) + 1)
               for c in range(min(c1, c2), max(c1, c2) + 1)
               if r < len(self.entries) and c < len(self.entries[r])}
        self._sel_cells = new

    def _sel_bounds(self):
        if not self._sel_cells: return None
        rs = [r for r, c in self._sel_cells]; cs = [c for r, c in self._sel_cells]
        return min(rs), min(cs), max(rs), max(cs)

    def _near_br(self, w, mg=6):
        try:
            px = w.winfo_pointerx(); py = w.winfo_pointery()
            x0 = w.winfo_rootx(); y0 = w.winfo_rooty()
            return (x0 + w.winfo_width() - mg <= px <= x0 + w.winfo_width()) and \
                   (y0 + w.winfo_height() - mg <= py <= y0 + w.winfo_height())
        except Exception: return False

    def _sel_handle_cell(self):
        b = self._sel_bounds()
        if b is None: return None
        try: return self.entries[b[2]][b[3]]
        except Exception: return None

    # ── active cell ───────────────────────────────────────────
    def _set_active(self, w):
        if self._active_cell is w: return
        if isinstance(self._active_cell, tk.Entry) and self._active_prev:
            try: self._active_cell.configure(**self._active_prev)
            except Exception: pass
        self._active_cell = w
        if isinstance(w, tk.Entry):
            self._active_prev = {"bg": w.cget("bg"), "highlightthickness": int(w.cget("highlightthickness")),
                                  "highlightbackground": w.cget("highlightbackground"),
                                  "highlightcolor": w.cget("highlightcolor"),
                                  "relief": w.cget("relief"),
                                  "bd": int(w.cget("bd")) if str(w.cget("bd")).isdigit() else 1}
            try: w.configure(bg=self.ACT_BG, highlightthickness=3,
                              highlightbackground="#c62828", highlightcolor="#c62828",
                              relief=tk.SOLID, bd=1)
            except Exception: pass

    # ── bind cell ─────────────────────────────────────────────
    def bind_cell(self, e):
        e.bind("<Return>",               self._on_enter)
        e.bind("<Up>",                   self._on_arrow)
        e.bind("<Down>",                 self._on_arrow)
        e.bind("<Left>",                 self._on_arrow)
        e.bind("<Right>",                self._on_arrow)
        e.bind("<Control-c>",            self._on_copy)
        e.bind("<Control-C>",            self._on_copy)
        e.bind("<Control-v>",            self._on_paste)
        e.bind("<Control-V>",            self._on_paste)
        e.bind("<FocusIn>",              lambda ev: self._set_active(ev.widget))
        e.bind("<ButtonPress-1>",        self._on_press)
        e.bind("<B1-Motion>",            self._on_drag)
        e.bind("<ButtonRelease-1>",      self._on_release)
        e.bind("<Motion>",               self._on_motion)
        e.bind("<Leave>",                lambda ev: ev.widget.configure(cursor=""))
        e.bind("<Shift-ButtonPress-1>",  self._on_shift_click)

    # ── mouse events ──────────────────────────────────────────
    def _on_motion(self, event):
        w = event.widget
        if not isinstance(w, tk.Entry): return
        pos = self._pos(w)
        if not pos: return
        r, c = pos
        if c >= self.factors_count: w.configure(cursor=""); return
        if self._sel_cells and self._sel_handle_cell() is w and self._near_br(w):
            w.configure(cursor="crosshair")
        elif not self._sel_cells and self._near_br(w):
            w.configure(cursor="crosshair")
        else:
            w.configure(cursor="")

    def _on_press(self, event):
        w = event.widget
        if not isinstance(w, tk.Entry): return
        pos = self._pos(w)
        if not pos: return
        r, c = pos
        if c < self.factors_count:
            if self._sel_cells and self._sel_handle_cell() is w and self._near_br(w):
                self._start_fill(use_sel=True); return "break"
            if not self._sel_cells and self._near_br(w):
                self._clear_sel()
                self._sel_anchor = (r, c); self._sel_cells = {(r, c)}
                self._sel_orig[(r, c)] = w.cget("bg"); self._apply_sel({(r, c)})
                self._start_fill(use_sel=False); return "break"
        self._fill_drag = False
        self._clear_sel()
        self._sel_anchor = (r, c); self._sel_cells = {(r, c)}
        self._sel_orig[(r, c)] = w.cget("bg"); self._apply_sel({(r, c)})
        w.focus_set()

    def _on_shift_click(self, event):
        w = event.widget; pos = self._pos(w)
        if not pos or self._sel_anchor is None: return
        ar, ac = self._sel_anchor; r, c = pos
        self._sel_range(ar, ac, r, c); return "break"

    def _on_drag(self, event):
        w = event.widget
        if not isinstance(w, tk.Entry): return
        if self._fill_drag: self._do_fill(event); return "break"
        if self._sel_anchor is None: return
        ar, ac = self._sel_anchor
        pos = self._pos(w)
        if pos:
            r, c = pos
        else:
            py = w.winfo_pointery(); px = w.winfo_pointerx(); r, c = ar, ac
            for ri in range(len(self.entries)):
                for ci in range(len(self.entries[ri])):
                    cell = self.entries[ri][ci]
                    if (cell.winfo_rootx() <= px <= cell.winfo_rootx() + cell.winfo_width() and
                            cell.winfo_rooty() <= py <= cell.winfo_rooty() + cell.winfo_height()):
                        r, c = ri, ci; break
        self._sel_range(ar, ac, r, c)

    def _on_release(self, event):
        if self._fill_drag:
            self._fill_drag = False; self._fill_rows = []; self._fill_cols = []; return "break"

    # ── fill drag ─────────────────────────────────────────────
    def _start_fill(self, use_sel):
        self._fill_drag = True
        if use_sel and self._sel_cells:
            b = self._sel_bounds()
            if b is None: self._fill_drag = False; return
            self._fill_rows = list(range(b[0], b[2] + 1))
            self._fill_cols = list(range(b[1], b[3] + 1))
        elif self._sel_anchor:
            self._fill_rows = [self._sel_anchor[0]]
            self._fill_cols = [self._sel_anchor[1]]
        else:
            self._fill_drag = False

    def _do_fill(self, event):
        w = event.widget
        if not isinstance(w, tk.Entry) or not self._fill_rows or not self._fill_cols: return
        last_src = self._fill_rows[-1]
        py = w.winfo_pointery(); target = last_src
        for rr in range(last_src, len(self.entries)):
            cell = self.entries[rr][self._fill_cols[0]]
            y0 = cell.winfo_rooty()
            if y0 <= py <= y0 + cell.winfo_height(): target = rr; break
        else:
            if py > self.entries[-1][self._fill_cols[0]].winfo_rooty(): target = len(self.entries)
        if target <= last_src: return
        n_src = len(self._fill_rows)
        dst = last_src + 1
        while dst <= target:
            while dst >= len(self.entries): self.add_row()
            src_r = self._fill_rows[(dst - last_src - 1) % n_src]
            for c in self._fill_cols:
                if c >= self.factors_count: break
                self.entries[dst][c].delete(0, tk.END)
                self.entries[dst][c].insert(0, self.entries[src_r][c].get())
            dst += 1
        self.inner.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    # ── copy selection Ctrl+C ─────────────────────────────────
    def _on_copy(self, event=None):
        if not self._sel_cells:
            w = event.widget if event else self.table_win.focus_get()
            if isinstance(w, tk.Entry):
                try: sel = w.get("sel.first", "sel.last")
                except Exception: sel = w.get()
                self.table_win.clipboard_clear(); self.table_win.clipboard_append(sel)
            return "break"
        b = self._sel_bounds()
        if b is None: return "break"
        r1, c1, r2, c2 = b
        lines = []
        for r in range(r1, r2 + 1):
            row = []
            for c in range(c1, c2 + 1):
                try: row.append(self.entries[r][c].get())
                except Exception: row.append("")
            lines.append("\t".join(row))
        self.table_win.clipboard_clear(); self.table_win.clipboard_append("\n".join(lines))
        return "break"

    # ── navigation ────────────────────────────────────────────
    def _on_enter(self, event=None):
        pos = self._pos(event.widget)
        if not pos: return "break"
        i, j = pos; ni = i + 1
        if ni >= len(self.entries): self.add_row()
        self.entries[ni][j].focus_set(); self.entries[ni][j].icursor(tk.END); return "break"

    def _on_arrow(self, event=None):
        pos = self._pos(event.widget)
        if not pos: return "break"
        i, j = pos
        if event.keysym == "Up":    i = max(0, i - 1)
        elif event.keysym == "Down":  i = min(len(self.entries) - 1, i + 1)
        elif event.keysym == "Left":  j = max(0, j - 1)
        elif event.keysym == "Right": j = min(len(self.entries[i]) - 1, j + 1)
        self.entries[i][j].focus_set(); self.entries[i][j].icursor(tk.END); return "break"

    def _on_paste(self, event=None):
        widget = event.widget if event else self.table_win.focus_get()
        if not isinstance(widget, tk.Entry): return "break"
        try: data = self.table_win.clipboard_get()
        except Exception: return "break"
        pos = self._pos(widget)
        if not pos: return "break"
        r0, c0 = pos
        for ir, rt in enumerate([r for r in data.splitlines() if r != ""]):
            for jc, val in enumerate(rt.split("\t")):
                rr = r0 + ir; cc = c0 + jc
                while rr >= len(self.entries): self.add_row()
                if cc >= self.cols: continue
                self.entries[rr][cc].delete(0, tk.END); self.entries[rr][cc].insert(0, val)
        return "break"

    def _pos(self, widget):
        for i, row in enumerate(self.entries):
            for j, cell in enumerate(row):
                if cell is widget: return i, j
        return None

    def _mk_entry(self, parent):
        return tk.Entry(parent, width=COL_W, fg="#000000", font=("Times New Roman", 12),
                        highlightthickness=1, highlightbackground="#c0c0c0", highlightcolor="#c0c0c0")

    # ── rename factor ─────────────────────────────────────────
    def rename_factor(self, col):
        if col < 0 or col >= self.factors_count: return
        fk = self.factor_keys[col]; old = self.ftitle(fk)
        dlg = tk.Toplevel(self.table_win or self.root)
        dlg.title("Перейменування"); dlg.resizable(False, False); set_icon(dlg)
        frm = tk.Frame(dlg, padx=14, pady=12); frm.pack()
        tk.Label(frm, text=f"Назва для {fk}:").grid(row=0, column=0, sticky="w")
        e = tk.Entry(frm, width=36, fg="#000000"); e.grid(row=1, column=0, pady=6)
        e.insert(0, old); e.select_range(0, "end"); e.focus_set()
        def ok():
            new = e.get().strip()
            if not new: messagebox.showwarning("", "Назва не може бути порожньою."); return
            self._set_ftitle(fk, new)
            if col < len(self.header_labels): self.header_labels[col].configure(text=new)
            dlg.destroy()
        bf = tk.Frame(frm); bf.grid(row=2, column=0, sticky="w", pady=(8, 0))
        tk.Button(bf, text="OK", width=10, command=ok).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(bf, text="Скасувати", width=12, command=dlg.destroy).pack(side=tk.LEFT)
        dlg.update_idletasks(); center_win(dlg); dlg.bind("<Return>", lambda ev: ok()); dlg.grab_set()

    def show_design_help(self):
        w = tk.Toplevel(self.root); w.title("Пояснення дизайнів"); w.resizable(False, False); set_icon(w)
        frm = tk.Frame(w, padx=16, pady=14); frm.pack()
        txt = ("CRD — Повна рандомізація\nRCBD — Блочна рандомізація\nSplit-plot — Спліт-плот\n\n"
               "SS Тип I: послідовний (порядок важливий)\n"
               "SS Тип II: ієрархічний (без взаємодій)\n"
               "SS Тип III: частковий (з урахуванням всіх ефектів) — рекомендований")
        t = tk.Text(frm, width=55, height=10, wrap="word"); t.insert("1.0", txt)
        t.configure(state="disabled"); t.pack()
        tk.Button(frm, text="OK", width=10, command=w.destroy).pack(pady=(10, 0))
        w.update_idletasks(); center_win(w); w.grab_set()

    def show_about(self):
        messagebox.showinfo("Розробник",
            f"S.A.D. — Статистичний аналіз даних  v{APP_VER}\n"
            "Розробник: Чаплоуцький Андрій Миколайович\n"
            "Уманський національний університет")

    # ══════════════════════════════════════════════════════════
    # OPEN TABLE  — with menu bar
    # ══════════════════════════════════════════════════════════
    def open_table(self, fc):
        if self.table_win and tk.Toplevel.winfo_exists(self.table_win):
            self.table_win.destroy()
        self.factors_count = fc
        self.factor_keys   = ["A", "B", "C", "D"][:fc]
        for fk in self.factor_keys:
            if fk not in self.factor_title_map: self._set_ftitle(fk, f"Фактор {fk}")

        self.table_win = tw = tk.Toplevel(self.root)
        tw.title(f"S.A.D. — {fc}-факторний аналіз")
        tw.geometry("1280x720"); set_icon(tw)

        # ── Menu bar ──────────────────────────────────────────
        mb = tk.Menu(tw)
        fm = tk.Menu(mb, tearoff=0)
        fm.add_command(label="Зберегти проект", accelerator="Ctrl+S", command=self.save_project)
        fm.add_command(label="Відкрити проект", accelerator="Ctrl+O", command=self.load_project)
        fm.add_separator()
        fm.add_command(label="Очистити проект", command=self.clear_project)
        mb.add_cascade(label="Файл", menu=fm)

        em = tk.Menu(mb, tearoff=0)
        em.add_command(label="Додати рядок",      command=self.add_row)
        em.add_command(label="Видалити рядок",     command=self.delete_row)
        em.add_separator()
        em.add_command(label="Додати стовпчик",    command=self.add_column)
        em.add_command(label="Видалити стовпчик",  command=self.delete_column)
        mb.add_cascade(label="Правка", menu=em)
        tw.config(menu=mb)

        tw.bind("<Control-s>", lambda e: self.save_project())
        tw.bind("<Control-S>", lambda e: self.save_project())
        tw.bind("<Control-o>", lambda e: self.load_project())
        tw.bind("<Control-O>", lambda e: self.load_project())

        # ── Toolbar ───────────────────────────────────────────
        ctl = tk.Frame(tw, padx=6, pady=4); ctl.pack(fill=tk.X)
        btn_texts = ["Вставити з буфера", "Аналіз даних", "Розробник"]
        bf_ = fit_font(btn_texts, start=13, min_s=9, target=150)
        bw, bh = 18, 1

        tk.Button(ctl, text="Вставити з буфера", width=bw, height=bh, font=bf_,
                  command=self._paste_from_focus).pack(side=tk.LEFT, padx=4)
        tk.Button(ctl, text="Аналіз даних", width=bw, height=bh, font=bf_,
                  bg="#c62828", fg="white", command=self.analyze).pack(side=tk.LEFT, padx=(10, 4))
        tk.Button(ctl, text="📚 Довідка", width=14, height=bh, font=bf_,
                  bg="#1a4b8c", fg="white",
                  command=lambda: show_help(tw)).pack(side=tk.LEFT, padx=4)
        tk.Button(ctl, text="Розробник", width=bw, height=bh, font=bf_,
                  command=self.show_about).pack(side=tk.RIGHT, padx=4)

        # ── Table canvas ──────────────────────────────────────
        self.canvas = tk.Canvas(tw)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(tw, orient="vertical", command=self.canvas.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y); self.canvas.configure(yscrollcommand=sb.set)
        self.inner = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        # mousewheel scroll
        def _mw(e): self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        self.canvas.bind_all("<MouseWheel>", _mw)

        self.rows = 12; self.cols = fc + 6
        self.entries = []; self.header_labels = []
        col_names = [self.ftitle(fk) for fk in self.factor_keys] + [f"Повт.{i+1}" for i in range(6)]
        for j, nm in enumerate(col_names):
            lbl = tk.Label(self.inner, text=nm, relief=tk.RIDGE, width=COL_W,
                           bg="#f0f0f0", fg="#000000", font=("Times New Roman", 12, "bold"))
            lbl.grid(row=0, column=j, padx=2, pady=2, sticky="nsew")
            self.header_labels.append(lbl)
            if j < fc: lbl.bind("<Double-Button-1>", lambda e, c=j: self.rename_factor(c))
        for i in range(self.rows):
            row_e = []
            for j in range(self.cols):
                e = self._mk_entry(self.inner); e.grid(row=i + 1, column=j, padx=2, pady=2)
                self.bind_cell(e); row_e.append(e)
            self.entries.append(row_e)
        self.inner.update_idletasks(); self.canvas.config(scrollregion=self.canvas.bbox("all"))
        self.entries[0][0].focus_set()
        tw.bind("<Control-v>", self._on_paste); tw.bind("<Control-V>", self._on_paste)

    # ── table editing ─────────────────────────────────────────
    def add_row(self):
        i = len(self.entries); row_e = []
        for j in range(self.cols):
            e = self._mk_entry(self.inner); e.grid(row=i + 1, column=j, padx=2, pady=2)
            self.bind_cell(e); row_e.append(e)
        self.entries.append(row_e); self.rows += 1
        self.inner.update_idletasks(); self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def delete_row(self):
        if not self.entries: return
        for e in self.entries.pop(): e.destroy()
        self.rows -= 1; self.inner.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def add_column(self):
        self.cols += 1; ci = self.cols - 1
        nm = f"Повт.{ci - self.factors_count + 1}"
        lbl = tk.Label(self.inner, text=nm, relief=tk.RIDGE, width=COL_W,
                       bg="#f0f0f0", fg="#000000", font=("Times New Roman", 12, "bold"))
        lbl.grid(row=0, column=ci, padx=2, pady=2, sticky="nsew"); self.header_labels.append(lbl)
        for i, row in enumerate(self.entries):
            e = self._mk_entry(self.inner); e.grid(row=i + 1, column=ci, padx=2, pady=2)
            self.bind_cell(e); row.append(e)
        self.inner.update_idletasks(); self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def delete_column(self):
        if self.cols <= self.factors_count + 1: return
        self.header_labels.pop().destroy()
        for row in self.entries: row.pop().destroy()
        self.cols -= 1; self.inner.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def _paste_from_focus(self):
        w = self.table_win.focus_get()
        if isinstance(w, tk.Entry):
            class _E: widget = w
            self._on_paste(_E())

    def _used_rep(self):
        rep_cols = []
        for c in range(self.factors_count, self.cols):
            for r in range(len(self.entries)):
                s = self.entries[r][c].get().strip()
                if not s: continue
                try: float(s.replace(",", ".")); rep_cols.append(c); break
                except Exception: continue
        return rep_cols

    def collect_long(self, design):
        long = []; rep = self._used_rep()
        if not rep: return long, rep
        for i, row in enumerate(self.entries):
            lvls = [row[k].get().strip() or f"рядок{i+1}" for k in range(self.factors_count)]
            for ic, c in enumerate(rep):
                s = row[c].get().strip()
                if not s: continue
                try: val = float(s.replace(",", "."))
                except Exception: continue
                rec = {"value": val}
                for ki, fk in enumerate(self.factor_keys): rec[fk] = lvls[ki]
                if design in ("rcbd", "split"): rec["BLOCK"] = f"Блок {ic+1}"
                long.append(rec)
        return long, rep

    # ── dialogs ───────────────────────────────────────────────
    def ask_params(self):
        dlg = tk.Toplevel(self.root); dlg.title("Параметри звіту")
        dlg.resizable(False, False); set_icon(dlg)
        frm = tk.Frame(dlg, padx=16, pady=14); frm.pack()
        tk.Label(frm, text="Назва показника:").grid(row=0, column=0, sticky="w", pady=5)
        e_ind = tk.Entry(frm, width=38); e_ind.grid(row=0, column=1, pady=5, padx=6)
        tk.Label(frm, text="Одиниці виміру:").grid(row=1, column=0, sticky="w", pady=5)
        e_un  = tk.Entry(frm, width=38); e_un.grid(row=1, column=1, pady=5, padx=6)

        # Design
        row_d = tk.Frame(frm); row_d.grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 4))
        tk.Label(row_d, text="Дизайн:").pack(side=tk.LEFT)
        tk.Button(row_d, text=" ? ", width=3, command=self.show_design_help).pack(side=tk.LEFT, padx=4)
        tk.Button(row_d, text="📚 Детальніше", font=("Times New Roman",10), fg="#1a4b8c",
                  relief=tk.FLAT, cursor="hand2",
                  command=lambda: show_help(dlg, "Дизайни експерименту")).pack(side=tk.LEFT, padx=2)
        dv = tk.StringVar(value="crd"); rb_f = ("Times New Roman", 13)
        df = tk.Frame(frm); df.grid(row=2, column=1, sticky="w", pady=(10, 4), padx=(140, 0))
        for txt, val in [("CRD", "crd"), ("RCBD", "rcbd"), ("Split-plot (лише параметр.)", "split")]:
            tk.Radiobutton(df, text=txt, variable=dv, value=val, font=rb_f).pack(anchor="w")
        mfv = tk.StringVar(value="A")
        sp_frm = tk.Frame(frm); sp_frm.grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))
        tk.Label(sp_frm, text="Головний фактор:").pack(side=tk.LEFT)
        ttk.Combobox(sp_frm, textvariable=mfv, width=6, state="readonly",
                     values=("A", "B", "C", "D")).pack(side=tk.LEFT, padx=6)
        sp_frm.grid_remove()
        def _upd(*_):
            sp_frm.grid() if dv.get() == "split" else sp_frm.grid_remove()
        dv.trace_add("write", _upd)

        # SS Type
        ss_lbl = tk.Frame(frm); ss_lbl.grid(row=4, column=0, sticky="w", pady=(10, 4))
        tk.Label(ss_lbl, text="Тип SS:").pack(side=tk.LEFT)
        tk.Button(ss_lbl, text=" ? ", width=3,
                  command=lambda: messagebox.showinfo("Типи SS",
                      "Тип I — Послідовний: кожен фактор після попередніх.\n"
                      "         Порядок важливий. Для збалансованих дизайнів.\n\n"
                      "Тип II — Ієрархічний: кожен фактор після решти головних\n"
                      "          ефектів (без взаємодій). Незбаланс. без взаємодій.\n\n"
                      "Тип III — Частковий (за замовч.): кожен ефект при всіх\n"
                      "           інших. Стандарт SPSS/SAS. Взаємодії враховані.\n\n"
                      "Тип IV — Функції, що оцінюються: для незбалансованих\n"
                      "          дизайнів з пропущеними клітинками (SAS).")).pack(side=tk.LEFT, padx=4)
        ssv = tk.StringVar(value="III")
        ssf = tk.Frame(frm); ssf.grid(row=4, column=1, sticky="w", pady=(10, 4), padx=(140, 0))
        for ss in ["I", "II", "III", "IV"]:
            tk.Radiobutton(ssf, text=f"Тип {ss}", variable=ssv, value=ss, font=rb_f).pack(side=tk.LEFT, padx=6)

        out = {"ok": False}
        def ok():
            out.update({"ok": True, "indicator": e_ind.get().strip(), "units": e_un.get().strip(),
                        "design": dv.get(), "split_main": mfv.get(), "ss_type": ssv.get()})
            if not out["indicator"] or not out["units"]:
                messagebox.showwarning("", "Заповніть показник та одиниці."); return
            dlg.destroy()
        bf = tk.Frame(frm); bf.grid(row=5, column=0, columnspan=2, pady=(12, 0))
        tk.Button(bf, text="OK", width=10, command=ok).pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text="Скасувати", width=12, command=dlg.destroy).pack(side=tk.LEFT)
        dlg.update_idletasks(); center_win(dlg); e_ind.focus_set()
        dlg.bind("<Return>", lambda e: ok()); dlg.grab_set(); self.root.wait_window(dlg)
        return out

    def choose_method(self, p_norm, design, n_var):
        dlg = tk.Toplevel(self.root); dlg.title("Вибір методу")
        dlg.resizable(False, False); set_icon(dlg)
        frm = tk.Frame(dlg, padx=16, pady=14); frm.pack()
        normal = (p_norm is not None) and (not math.isnan(p_norm)) and (p_norm > 0.05)
        rb_f = ("Times New Roman", 13)
        # Context help row
        help_row = tk.Frame(frm); help_row.pack(fill=tk.X, pady=(0, 6))
        tk.Button(help_row, text="📚 Нормальність розподілу", font=("Times New Roman",10),
                  fg="#1a4b8c", relief=tk.FLAT, cursor="hand2",
                  command=lambda: show_help(dlg, "Нормальність розподілу")).pack(side=tk.LEFT)
        tk.Button(help_row, text="📚 Методи порівнянь", font=("Times New Roman",10),
                  fg="#1a4b8c", relief=tk.FLAT, cursor="hand2",
                  command=lambda: show_help(dlg, "Методи порівнянь")).pack(side=tk.LEFT, padx=6)
        tk.Button(help_row, text="📚 Непараметричні тести", font=("Times New Roman",10),
                  fg="#1a4b8c", relief=tk.FLAT, cursor="hand2",
                  command=lambda: show_help(dlg, "Непараметричні тести")).pack(side=tk.LEFT)
        if normal:
            tk.Label(frm, text="Дані відповідають нормальному розподілу (Shapiro–Wilk).",
                     justify="left").pack(anchor="w", pady=(0, 8))
            options = [("НІР₀₅ (LSD)", "lsd"), ("Тест Тьюкі", "tukey"),
                       ("Тест Дункана", "duncan"), ("Бонферроні", "bonferroni")]
        else:
            if design == "split":
                tk.Label(frm, text="Split-plot: лише параметричний.\nЗалишки не нормальні → аналіз некоректний.\nРекомендація: трансформуйте або оберіть CRD/RCBD.",
                         fg="#c62828", justify="left").pack(anchor="w"); options = []
            else:
                tk.Label(frm, text="Дані НЕ відповідають нормальному розподілу.\nОберіть метод:",
                         fg="#c62828", justify="left").pack(anchor="w", pady=(0, 8))
                if design == "crd":
                    options = [("Краскела–Уолліса", "kw"),
                               ("Манна-Уітні", "mw"),
                               ("🔁 ln(x) + параметричний", "log_param"),
                               ("🔁 √x + параметричний",    "sqrt_param"),
                               ("🔁 log₁₀(x) + параметричний", "log10_param")]
                else:
                    if n_var == 2:
                        options = [("Wilcoxon (парний)", "wilcoxon"),
                                   ("🔁 ln(x) + параметричний",    "log_param"),
                                   ("🔁 √x + параметричний",       "sqrt_param"),
                                   ("🔁 log₁₀(x) + параметричний","log10_param")]
                    else:
                        options = [("Friedman", "friedman"),
                                   ("🔁 ln(x) + параметричний",    "log_param"),
                                   ("🔁 √x + параметричний",       "sqrt_param"),
                                   ("🔁 log₁₀(x) + параметричний","log10_param")]
        out = {"ok": False, "method": None}
        if not options:
            tk.Button(frm, text="OK", width=10, command=dlg.destroy).pack(pady=(10, 0))
            dlg.update_idletasks(); center_win(dlg); dlg.grab_set(); self.root.wait_window(dlg); return out
        var = tk.StringVar(value=options[0][1])
        for txt, val in options:
            tk.Radiobutton(frm, text=txt, variable=var, value=val, font=rb_f).pack(anchor="w", pady=2)
        def ok():
            out.update({"ok": True, "method": var.get()}); dlg.destroy()
        bf = tk.Frame(frm); bf.pack(pady=(12, 0))
        tk.Button(bf, text="OK", width=10, command=ok).pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text="Скасувати", width=12, command=dlg.destroy).pack(side=tk.LEFT)
        dlg.update_idletasks(); center_win(dlg); dlg.bind("<Return>", lambda e: ok())
        dlg.grab_set(); self.root.wait_window(dlg); return out

    # ════════════════════════════════════════════════════════════
    # ANALYZE
    # ════════════════════════════════════════════════════════════
    def analyze(self):
        created = datetime.now()
        params = self.ask_params()
        if not params["ok"]: return
        indicator = params["indicator"]; units = params["units"]
        design = params["design"]; split_main = params["split_main"]
        ss_type = params.get("ss_type", "III")

        long, used_rep = self.collect_long(design)
        if not long: messagebox.showwarning("Помилка даних", "Немає числових даних."); return
        values = np.array([r["value"] for r in long], dtype=float)

        # ── Мінімальна кількість спостережень ──────────────────
        if len(values) < 6:
            messagebox.showwarning("Замало даних",
                f"Для дисперсійного аналізу потрібно щонайменше 6 спостережень.\n"
                f"Наразі: {len(values)}."); return

        lbf = {f: first_seen([r.get(f) for r in long]) for f in self.factor_keys}

        # ── Перевірка мінімальної кількості рівнів кожного фактора ──
        for f in self.factor_keys:
            if len(lbf[f]) < 2:
                messagebox.showwarning("Помилка даних",
                    f"Фактор «{self.ftitle(f)}» має лише 1 рівень.\n"
                    "Для аналізу потрібно щонайменше 2 рівні."); return

        # ── Перевірка мінімальної кількості повторностей для RCBD/Split ──
        if design in ("rcbd", "split") and len(used_rep) < 2:
            messagebox.showwarning("Помилка дизайну",
                f"Для дизайну {design.upper()} потрібно щонайменше 2 повторності (блоки).\n"
                f"Наразі: {len(used_rep)}."); return

        var_order = first_seen([tuple(r.get(f) for f in self.factor_keys) for r in long])
        v_names = [" | ".join(map(str, k)) for k in var_order]
        n_var = len(var_order)

        # ── Перевірка збалансованості при Тип I SS ─────────────
        if ss_type == "I":
            from collections import Counter
            cell_counts = Counter(tuple(r.get(f) for f in self.factor_keys) for r in long)
            counts = list(cell_counts.values())
            if len(set(counts)) > 1:
                ans = messagebox.askyesno("Увага: незбалансовані дані + Тип I SS",
                    "Дані незбалансовані (різна кількість спостережень у клітинках).\n\n"
                    "При Тип I SS (послідовний) результат залежить від ПОРЯДКУ введення факторів.\n"
                    "Зміна порядку дає інші значення SS та p.\n\n"
                    "Рекомендація: використовуйте Тип III для незбалансованих даних.\n\n"
                    "Продовжити з Тип I SS?")
                if not ans: return

        try:
            if design == "crd":    res = anova_crd(long, self.factor_keys, lbf, ss_type)
            elif design == "rcbd": res = anova_rcbd(long, self.factor_keys, lbf, ss_type=ss_type)
            else:
                if split_main not in self.factor_keys: split_main = self.factor_keys[0]
                res = anova_split(long, self.factor_keys, split_main, ss_type=ss_type)
        except Exception as ex: messagebox.showerror("Помилка моделі", str(ex)); return

        residuals = np.array(res.get("residuals", []), dtype=float)
        n_res = len(residuals)
        try: W, p_norm = shapiro(residuals) if n_res >= 3 else (np.nan, np.nan)
        except Exception: W, p_norm = np.nan, np.nan

        # ── Попередження про обмеження Shapiro-Wilk ────────────
        sw_warning = ""
        if n_res < 8:
            sw_warning = f"\n⚠ Увага: n={n_res} — надто мало для надійного тесту нормальності."
        elif n_res > 100:
            sw_warning = (f"\n⚠ Увага: n={n_res} — при великих вибірках Shapiro–Wilk виявляє\n"
                          "  навіть мінімальні відхилення як значущі. Оцінюйте разом з Q-Q графіком.")

        normal = (not math.isnan(p_norm)) and (p_norm > 0.05)
        if design == "split" and not normal:
            messagebox.showwarning("Split-plot: аналіз неможливий",
                f"Залишки моделі не відповідають нормальному розподілу\n"
                f"(Shapiro–Wilk: W={fmt(W,4)}, p={fmt(p_norm,4)}).\n\n"
                "Split-plot реалізований лише для параметричних методів.\n"
                "Рекомендації:\n"
                "• трансформуйте дані (логарифмування) і повторіть;\n"
                "• або оберіть CRD/RCBD для непараметричного аналізу."
                + sw_warning); return

        choice = self.choose_method(p_norm, design, n_var)
        if not choice["ok"]: return
        method = choice["method"]

        log_applied = False
        transform_label = ""
        if method in ("log_param", "sqrt_param", "log10_param"):
            # ── Validate data for chosen transformation ──────────
            if method in ("log_param", "log10_param") and np.any(values <= 0):
                messagebox.showwarning("Трансформація неможлива",
                    "Дані містять нулі або від'ємні значення.\n"
                    "Логарифмування неможливе.\n"
                    "Оберіть √x або непараметричний метод."); return
            if method == "sqrt_param" and np.any(values < 0):
                messagebox.showwarning("Трансформація неможлива",
                    "Дані містять від'ємні значення.\n"
                    "Трансформація √x неможлива."); return

            # ── Apply transformation ─────────────────────────────
            if method == "log_param":
                long = [dict(r, value=math.log(r["value"])) for r in long]
                transform_label = "ln(x)"
            elif method == "sqrt_param":
                long = [dict(r, value=math.sqrt(r["value"])) for r in long]
                transform_label = "√x"
            elif method == "log10_param":
                long = [dict(r, value=math.log10(r["value"])) for r in long]
                transform_label = "log₁₀(x)"

            values = np.array([r["value"] for r in long], dtype=float)
            log_applied = True

            # ── Re-run model on transformed data ─────────────────
            try:
                if design == "crd":    res = anova_crd(long, self.factor_keys, lbf, ss_type)
                elif design == "rcbd": res = anova_rcbd(long, self.factor_keys, lbf, ss_type=ss_type)
                else:                  res = anova_split(long, self.factor_keys, split_main, ss_type=ss_type)
            except Exception as ex: messagebox.showerror("Помилка моделі", str(ex)); return

            residuals = np.array(res.get("residuals", []), dtype=float)
            try: W, p_norm = shapiro(residuals) if len(residuals) >= 3 else (np.nan, np.nan)
            except Exception: W, p_norm = np.nan, np.nan

            if math.isnan(p_norm) or p_norm <= 0.05:
                messagebox.showwarning("Трансформація не допомогла",
                    f"Застосовано трансформацію {transform_label}, але залишки\n"
                    f"все одно не відповідають нормальному розподілу\n"
                    f"(Shapiro–Wilk: W={fmt(W,4)}, p={fmt(p_norm,4)}).\n\n"
                    "Параметричний аналіз неможливий.\n"
                    "Оберіть непараметричний метод (Kruskal–Wallis, Mann–Whitney тощо)."); return

            messagebox.showinfo("Трансформація успішна",
                f"Застосовано трансформацію {transform_label}.\n"
                f"Shapiro–Wilk після трансформації:\n"
                f"W={fmt(W,4)},  p={fmt(p_norm,4)}  ✓ нормальний розподіл.\n\n"
                "Оберіть метод парних порівнянь:")
            choice2 = self.choose_method(p_norm, design, n_var)
            if not choice2["ok"]: return
            method = choice2["method"]
            if method in ("log_param", "sqrt_param", "log10_param"): method = "lsd"

        MS_err = res.get("MS_error", np.nan); df_err = res.get("df_error", np.nan)
        MS_wp  = res.get("MS_whole", np.nan); df_wp  = res.get("df_whole", np.nan)
        split_mf = res.get("main_factor", split_main) if design == "split" else None

        vs_ = vstats(long, self.factor_keys)
        v_means = {k: vs_[k][0] for k in vs_}; v_sds = {k: vs_[k][1] for k in vs_}; v_ns = {k: vs_[k][2] for k in vs_}
        means1 = {v_names[i]: v_means.get(var_order[i], np.nan) for i in range(n_var)}
        ns1    = {v_names[i]: v_ns.get(var_order[i], 0)         for i in range(n_var)}
        gv = groups_by(long, tuple(self.factor_keys))
        groups1 = {v_names[i]: gv.get(var_order[i], []) for i in range(n_var)}

        fg = {f: {k[0]: v for k, v in groups_by(long, (f,)).items()} for f in self.factor_keys}
        fm = {f: {lv: float(np.mean(arr)) if arr else np.nan for lv, arr in fg[f].items()} for f in self.factor_keys}
        fn = {f: {lv: len(arr) for lv, arr in fg[f].items()} for f in self.factor_keys}
        fsd= {f: {lv: float(np.std(arr, ddof=1)) if len(arr) >= 2 else (0. if len(arr)==1 else np.nan)
                  for lv, arr in fg[f].items()} for f in self.factor_keys}
        fmed = {f: {lv: median_q(arr)[0] for lv, arr in fg[f].items()} for f in self.factor_keys}
        fq   = {f: {lv: median_q(arr)[1:] for lv, arr in fg[f].items()} for f in self.factor_keys}

        vmed = {var_order[i]: median_q(groups1[v_names[i]])[0] for i in range(n_var)}
        vq   = {var_order[i]: median_q(groups1[v_names[i]])[1:] for i in range(n_var)}
        rkv  = mean_ranks(long, lambda r: " | ".join(str(r.get(f)) for f in self.factor_keys))
        rkf  = {f: mean_ranks(long, lambda r, ff=f: r.get(ff)) for f in self.factor_keys}

        lev_F, lev_p = (np.nan, np.nan)
        if method in ("lsd", "tukey", "duncan", "bonferroni"):
            lev_F, lev_p = levene_test(groups1)
            # ── Блокування при неоднорідних дисперсіях ─────────
            if not math.isnan(lev_p) and lev_p < ALPHA:
                ans = messagebox.askyesno(
                    "Неоднорідність дисперсій (тест Левена)",
                    f"Тест Левена виявив неоднорідність дисперсій\n"
                    f"(F={fmt(lev_F,4)}, p={fmt(lev_p,4)}) — умова ANOVA порушена.\n\n"
                    "Параметричний аналіз при неоднорідних дисперсіях дає\n"
                    "недостовірні F-значення та p-значення.\n\n"
                    "Рекомендації:\n"
                    "• застосуйте трансформацію даних (логарифмування);\n"
                    "• або оберіть непараметричний метод (Kruskal–Wallis).\n\n"
                    "Продовжити параметричний аналіз попри порушення умови?")
                if not ans: return

        kw_H = kw_p = kw_df = kw_eps = np.nan; do_ph = True
        fr_chi = fr_p = fr_df = fr_W = np.nan; wil_s = wil_p = np.nan
        rcbd_ph = []; rcbd_sig = {}
        lf = {f: {lv: "" for lv in lbf[f]} for f in self.factor_keys}
        lnamed = {nm: "" for nm in v_names}
        ph_rows = []; fpt = {}

        if method == "lsd":
            # ── Protected LSD (Fisher): перевірити значущість глобального F ──
            # Знаходимо найменше p-значення серед головних ефектів та взаємодій
            global_p_values = []
            for raw_row in res["table"]:
                nm_, _ss, _df, _ms, _F, pv_ = raw_row
                if any(x in str(nm_) for x in ["Залишок", "Загальна", "Блоки", "WP-error"]):
                    continue
                if pv_ is not None and not (isinstance(pv_, float) and math.isnan(pv_)):
                    global_p_values.append(float(pv_))

            global_F_sig = any(p < ALPHA for p in global_p_values) if global_p_values else False

            if not global_F_sig:
                messagebox.showwarning("Protected LSD: пост-хок заблоковано",
                    "Жоден з ефектів у дисперсійному аналізі не є статистично значущим\n"
                    "(p ≥ 0.05 для всіх факторів та їх взаємодій).\n\n"
                    "Відповідно до принципу Protected LSD (Fisher, 1935),\n"
                    "попарні порівняння можна виконувати ЛИШЕ після значущого F-тесту.\n\n"
                    "Висновок: між варіантами немає статистично значущої різниці.\n"
                    "Звіт сформовано з таблицями описової статистики без пост-хок аналізу.")
                # Продовжуємо — формуємо звіт без літер CLD
            else:
                for f in self.factor_keys:
                    MS_ = MS_wp if (design == "split" and f == split_mf) else MS_err
                    df_ = df_wp if (design == "split" and f == split_mf) else df_err
                    lf[f] = cld(lbf[f], fm[f], lsd_sig(lbf[f], fm[f], fn[f], MS_, df_))
                if design != "split":
                    lnamed = cld(v_names, means1, lsd_sig(v_names, means1, ns1, MS_err, df_err))

        elif method in ("tukey", "duncan", "bonferroni"):
            # ── Перевірка значущості глобального F перед пост-хок ──
            global_p_values_2 = []
            for raw_row in res["table"]:
                nm_, _ss, _df, _ms, _F, pv_ = raw_row
                if any(x in str(nm_) for x in ["Залишок", "Загальна", "Блоки", "WP-error"]):
                    continue
                if pv_ is not None and not (isinstance(pv_, float) and math.isnan(pv_)):
                    global_p_values_2.append(float(pv_))
            global_F_sig_2 = any(p < ALPHA for p in global_p_values_2) if global_p_values_2 else False

            if not global_F_sig_2:
                messagebox.showwarning("Пост-хок заблоковано",
                    "Жоден ефект не є статистично значущим (p ≥ 0.05).\n\n"
                    "Виконання пост-хок порівнянь без значущого F-тесту\n"
                    "призводить до надмірного числа хибнопозитивних результатів.\n\n"
                    "Висновок: між варіантами немає статистично значущої різниці.")
            else:
                if design != "split":
                    ph_rows, sig_ = pairwise_param(v_names, means1, ns1, MS_err, df_err, method)
                    lnamed = cld(v_names, means1, sig_)
                    for f in self.factor_keys:
                        r_, s_ = pairwise_param(lbf[f], fm[f], fn[f], MS_err, df_err, method)
                        fpt[f] = r_; lf[f] = cld(lbf[f], fm[f], s_)
                else:
                    for f in self.factor_keys:
                        MS_ = MS_wp if f == split_mf else MS_err
                        df_ = df_wp if f == split_mf else df_err
                        r_, s_ = pairwise_param(lbf[f], fm[f], fn[f], MS_, df_, method)
                        fpt[f] = r_; lf[f] = cld(lbf[f], fm[f], s_)

        elif method == "kw":
            try:
                smp = [groups1[n] for n in v_names if groups1[n]]
                if len(smp) >= 2:
                    kwr = kruskal(*smp); kw_H = float(kwr.statistic); kw_p = float(kwr.pvalue)
                    kw_df = len(smp) - 1; kw_eps = eps2_kw(kw_H, len(long), len(smp))
            except Exception: pass
            if not math.isnan(kw_p) and kw_p >= ALPHA: do_ph = False
            if do_ph:
                ph_rows, sig_ = pairwise_mw(v_names, groups1)
                lnamed = cld(v_names, {n: float(np.median(groups1[n])) if groups1[n] else np.nan for n in v_names}, sig_)

        elif method == "mw":
            ph_rows, sig_ = pairwise_mw(v_names, groups1)
            lnamed = cld(v_names, {n: float(np.median(groups1[n])) if groups1[n] else np.nan for n in v_names}, sig_)

        elif method == "friedman":
            bnames = first_seen([f"Блок {i+1}" for i in range(len(used_rep))])
            long2  = [dict(r, VARIANT=" | ".join(str(r.get(f)) for f in self.factor_keys)) for r in long]
            mat, _ = rcbd_matrix(long2, v_names, bnames)
            if len(mat) < 2: messagebox.showwarning("", "Потрібні ≥ 2 повних блоки."); return
            try:
                fr = friedmanchisquare(*[np.array(c, dtype=float) for c in zip(*mat)])
                fr_chi = float(fr.statistic); fr_p = float(fr.pvalue)
                fr_df = n_var - 1; fr_W = kendalls_w(fr_chi, len(mat), n_var)
            except Exception: pass
            if not math.isnan(fr_p) and fr_p < ALPHA:
                rcbd_ph, rcbd_sig = pairwise_wilcox(v_names, mat)
                lnamed = cld(v_names, {n: float(np.median(groups1[n])) if groups1[n] else np.nan for n in v_names}, rcbd_sig)

        elif method == "wilcoxon":
            if n_var != 2: messagebox.showwarning("", "Wilcoxon лише для 2 варіантів."); return
            bnames = first_seen([f"Блок {i+1}" for i in range(len(used_rep))])
            long2  = [dict(r, VARIANT=" | ".join(str(r.get(f)) for f in self.factor_keys)) for r in long]
            mat, _ = rcbd_matrix(long2, v_names, bnames)
            if len(mat) < 2: messagebox.showwarning("", "Потрібні ≥ 2 блоки."); return
            arr = np.array(mat, dtype=float)
            try:
                st, p = wilcoxon(arr[:, 0], arr[:, 1], zero_method="wilcox", alternative="two-sided", mode="auto")
                wil_s = float(st); wil_p = float(p)
            except Exception: pass
            if not math.isnan(wil_p) and wil_p < ALPHA:
                lnamed = cld(v_names, {n: float(np.median(groups1[n])) if groups1[n] else np.nan for n in v_names},
                             {(v_names[0], v_names[1]): True})

        lv_var = {var_order[i]: lnamed.get(v_names[i], "") for i in range(n_var)}
        SS_tot = res.get("SS_total", np.nan); SS_err = res.get("SS_error", np.nan)
        R2 = (1 - (SS_err / SS_tot)) if not any(math.isnan(x) for x in [SS_tot, SS_err]) and SS_tot > 0 else np.nan

        cv_r = [[self.ftitle(f), fmt(cv_means([fm[f].get(lv, np.nan) for lv in lbf[f]]), 2)]
                for f in self.factor_keys]
        cv_r.append(["Загальний", fmt(cv_vals(values), 2)])

        nonparam = method in ("mw", "kw", "friedman", "wilcoxon")

        def _rn(nm):
            if not isinstance(nm, str) or not nm.startswith("Фактор "): return nm
            rest = nm.replace("Фактор ", "")
            parts = rest.split("×")
            return "×".join(self.ftitle(p) if p in self.factor_keys else p for p in parts)

        anova_rows = []
        for raw_row in res["table"]:
            nm, SSv, dfv, MSv, Fv, pv = raw_row
            df_s = str(int(dfv)) if dfv is not None and not (isinstance(dfv, float) and math.isnan(dfv)) else ""
            nm2 = _rn(nm)
            if any(x in nm2 for x in ["Залишок", "WP-error", "Блоки"]) or nm2 == "Загальна":
                anova_rows.append([nm2, fmt(SSv, 3), df_s, fmt(MSv, 3), "", "", ""])
            else:
                mk = sig_mark(pv); concl = f"різниця {mk}" if mk else "–"
                anova_rows.append([nm2, fmt(SSv, 3), df_s, fmt(MSv, 3), fmt(Fv, 3), fmt(pv, 4), concl])

        eff_rows  = [[_rn(r[0]), r[1]] for r in build_eff_rows(res["table"])]
        pe2_rows  = [[_rn(r[0]), r[1], r[2]] for r in build_pe2_rows(res["table"])]

        # cache for graph redraw
        self._lbf_cache = lbf

        self.show_report(
            created=created, indicator=indicator, units=units, design=design,
            ss_type=ss_type, method=method, log_applied=log_applied,
            transform_label=transform_label,
            n_var=n_var, n_rep=len(used_rep), n_obs=len(long),
            split_mf=split_mf, W=W, p_norm=p_norm,
            lev_F=lev_F, lev_p=lev_p,
            kw_H=kw_H, kw_p=kw_p, kw_df=kw_df, kw_eps=kw_eps, do_ph=do_ph,
            fr_chi=fr_chi, fr_p=fr_p, fr_df=fr_df, fr_W=fr_W,
            wil_s=wil_s, wil_p=wil_p,
            anova_rows=anova_rows, eff_rows=eff_rows, pe2_rows=pe2_rows,
            cv_r=cv_r, R2=R2,
            lf=lf, lv_var=lv_var, lbf=lbf,
            fm=fm, fsd=fsd, fmed=fmed, fq=fq, fn=fn,
            var_order=var_order, v_names=v_names,
            v_means=v_means, v_sds=v_sds, v_ns=v_ns,
            vmed=vmed, vq=vq, rkv=rkv, rkf=rkf,
            groups1=groups1, ph_rows=ph_rows, fpt=fpt,
            rcbd_ph=rcbd_ph, nonparam=nonparam, res=res,
        )
        self.show_graphs(long, lf, indicator, units, eff_rows, pe2_rows)

    # ════════════════════════════════════════════════════════════
    # REPORT WINDOW
    # ════════════════════════════════════════════════════════════
    def show_report(self, **kw):
        if self.report_win and tk.Toplevel.winfo_exists(self.report_win):
            self.report_win.destroy()
        self.report_win = rw = tk.Toplevel(self.root)
        rw.title("Звіт"); rw.geometry("1200x820"); set_icon(rw)

        top = tk.Frame(rw, padx=8, pady=6); top.pack(fill=tk.X)
        self._report_buf = []
        def copy_all():
            rw.clipboard_clear()
            # build plain-text with fixed-width columns for Word
            lines = self._report_buf
            rw.clipboard_append("\n".join(lines))
            messagebox.showinfo("", "Текстовий звіт скопійовано.")
        tk.Button(top, text="Копіювати звіт (текст)", command=copy_all).pack(side=tk.LEFT, padx=4)

        # Scrollable body
        outer = tk.Frame(rw); outer.pack(fill=tk.BOTH, expand=True)
        vsb = ttk.Scrollbar(outer, orient="vertical"); vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb = ttk.Scrollbar(outer, orient="horizontal"); hsb.pack(side=tk.BOTTOM, fill=tk.X)
        cv  = tk.Canvas(outer, yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.config(command=cv.yview); hsb.config(command=cv.xview)
        body = tk.Frame(cv); cv.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        # mousewheel — bind to window level so it works everywhere in report
        def _mw(e): cv.yview_scroll(int(-1 * (e.delta / 120)), "units")
        def _bind_mw(w):
            w.bind("<MouseWheel>", _mw)
            for ch in w.winfo_children(): _bind_mw(ch)
        rw.bind("<MouseWheel>", _mw)
        cv.bind("<MouseWheel>", _mw)
        body.bind("<MouseWheel>", _mw)
        # re-bind after each widget is added
        body.bind("<Configure>", lambda e: (cv.configure(scrollregion=cv.bbox("all")), _bind_mw(body)))

        buf = self._report_buf
        def _txt(s):
            tk.Label(body, text=s, font=("Times New Roman", 12), fg="#000000",
                     justify="left", anchor="w", wraplength=1100).pack(fill=tk.X, padx=12, pady=1)
            buf.append(s)
        def _head(s):
            tk.Label(body, text=s, font=("Times New Roman", 13, "bold"), fg="#000000",
                     justify="left", anchor="w").pack(fill=tk.X, padx=12, pady=(8, 2))
            buf.append("\n" + s)
        def _sep():
            ttk.Separator(body, orient="horizontal").pack(fill=tk.X, padx=12, pady=4)
            buf.append("-" * 80)
        def _table(headers, rows, min_col=90):
            frm, tv_widget = make_tv(body, headers, rows, min_col)
            frm.pack(fill=tk.X, padx=12, pady=(2, 8))
            # bind mousewheel on treeview and its frame so scroll works
            frm.bind("<MouseWheel>", _mw)
            tv_widget.bind("<MouseWheel>", _mw)
            # plain text for clipboard: TAB-separated (Word auto-formats as table)
            buf.append("\t".join(str(h) for h in headers))
            buf.append("\t".join("-" * max(3, len(str(h))) for h in headers))
            for row in rows:
                buf.append("\t".join("" if v is None else str(v) for v in row))
            buf.append("")

        d = kw
        _txt("З В І Т   С Т А Т И С Т И Ч Н О Г О   А Н А Л І З У")
        _txt(f"Показник: {d['indicator']}   |   Одиниці: {d['units']}")
        _txt(f"Дата: {d['created'].strftime('%d.%m.%Y %H:%M')}")
        _sep()
        design_lbl = {"crd": "CRD (повна рандомізація)", "rcbd": "RCBD (блочна рандомізація)", "split": "Split-plot"}[d['design']]
        _txt(f"Дизайн: {design_lbl}   |   Тип SS: {d['ss_type']}   |   Варіантів: {d['n_var']}   |   Повт.: {d['n_rep']}   |   Спостережень: {d['n_obs']}")
        if d['design'] == "split": _txt(f"Головний фактор (WP): {d['split_mf']}")
        if d['log_applied']:
            tl = d.get('transform_label', 'ln(x)')
            _txt(f"⚠ Застосовано трансформацію {tl}. Середні у звіті — у трансформованій шкалі.")
        method_lbl = {"lsd": "НІР₀₅ (LSD)", "tukey": "Тест Тьюкі", "duncan": "Тест Дункана",
                      "bonferroni": "Бонферроні", "kw": "Kruskal–Wallis", "mw": "Mann–Whitney",
                      "friedman": "Friedman", "wilcoxon": "Wilcoxon"}.get(d['method'], "")
        _txt(f"Метод: {method_lbl}")
        _txt("** — p<0.01; * — p<0.05; різні літери → істотна різниця.")
        _sep()
        _txt(f"Shapiro–Wilk (залишки): {norm_txt(d['p_norm'])}   W={fmt(d['W'], 4)}   p={fmt(d['p_norm'], 4)}")

        if d['method'] == "kw" and not math.isnan(d['kw_p']):
            c_ = ("різниця " + sig_mark(d['kw_p'])) if d['kw_p'] < ALPHA else "–"
            _txt(f"Kruskal–Wallis:  H={fmt(d['kw_H'],4)}  df={d['kw_df']}  p={fmt(d['kw_p'],4)}  {c_}   ε²={fmt(d['kw_eps'],4)}")
        if d['method'] == "friedman" and not math.isnan(d['fr_p']):
            c_ = ("різниця " + sig_mark(d['fr_p'])) if d['fr_p'] < ALPHA else "–"
            _txt(f"Friedman:  χ²={fmt(d['fr_chi'],4)}  df={d['fr_df']}  p={fmt(d['fr_p'],4)}  {c_}   Kendall's W={fmt(d['fr_W'],4)}")
        if d['method'] == "wilcoxon" and not math.isnan(d['wil_p']):
            c_ = ("різниця " + sig_mark(d['wil_p'])) if d['wil_p'] < ALPHA else "–"
            _txt(f"Wilcoxon:  W={fmt(d['wil_s'],4)}  p={fmt(d['wil_p'],4)}  {c_}")

        if not d['nonparam']:
            if not math.isnan(d['lev_p']):
                lc = "умова виконується" if d['lev_p'] >= ALPHA else f"умова порушена {sig_mark(d['lev_p'])}"
                _txt(f"Тест Левена (однорідність дисперсій):  F={fmt(d['lev_F'],4)}  p={fmt(d['lev_p'],4)}  {lc}")
            _sep()
            _head("ТАБЛИЦЯ 1. Дисперсійний аналіз (ANOVA)")
            _table(["Джерело", "SS", "df", "MS", "F", "p", "Висновок"], d['anova_rows'])
            _head("ТАБЛИЦЯ 2. Сила впливу факторів (% від SS)")
            _table(["Джерело", "%"], d['eff_rows'])
            _head("ТАБЛИЦЯ 3. Розмір ефекту (partial η²)")
            _table(["Джерело", "partial η²", "Сила ефекту"], d['pe2_rows'])
            _head("ТАБЛИЦЯ 4. Коефіцієнт варіації (CV, %)")
            _table(["Елемент", "CV, %"], d['cv_r'])
            _txt(f"Коефіцієнт детермінації R² = {fmt(d['R2'], 4)}")
            tno = 5
            if d['method'] == "lsd":
                nir_r = [[k, fmt(v, 4)] for k, v in d['res'].get("NIR05", {}).items()]
                if nir_r:
                    _head(f"ТАБЛИЦЯ {tno}. НІР₀₅"); _table(["Елемент", "НІР₀₅"], nir_r); tno += 1

            for f in self.factor_keys:
                _head(f"ТАБЛИЦЯ {tno}. Середні по фактору: {self.ftitle(f)}")
                rows_f = [[str(lv), fmt(d['fm'][f].get(lv, np.nan), 3),
                           fmt(d['fsd'][f].get(lv, np.nan), 3),
                           d['lf'][f].get(lv, "") or "–"]
                          for lv in d['lbf'][f]]
                _table([self.ftitle(f), "Середнє", "± SD", "Літери CLD"], rows_f); tno += 1

            _head(f"ТАБЛИЦЯ {tno}. Середні по варіантах")
            rows_v = [[nm, fmt(d['v_means'].get(d['var_order'][i], np.nan), 3),
                       fmt(d['v_sds'].get(d['var_order'][i], np.nan), 3),
                       d['lv_var'].get(d['var_order'][i], "") or "–"]
                      for i, nm in enumerate(d['v_names'])]
            _table(["Варіант", "Середнє", "± SD", "Літери CLD"], rows_v); tno += 1

            if d['design'] != "split":
                if d['method'] in ("tukey", "duncan", "bonferroni") and d['ph_rows']:
                    _head(f"ТАБЛИЦЯ {tno}. Парні порівняння варіантів")
                    _table(["Пара", "p", "Висновок"], d['ph_rows']); tno += 1
            else:
                for f in self.factor_keys:
                    rr = d['fpt'].get(f, [])
                    if rr:
                        _head(f"ТАБЛИЦЯ {tno}. Парні порівняння: {self.ftitle(f)}")
                        _table(["Пара", "p", "Висновок"], rr); tno += 1
        else:
            tno = 1
            for f in self.factor_keys:
                _head(f"ТАБЛИЦЯ {tno}. Описова (непараметрична): {self.ftitle(f)}")
                rows = [[str(lv), str(d['fn'][f].get(lv, 0)), fmt(d['fmed'][f].get(lv, np.nan), 3),
                         f"{fmt(d['fq'][f].get(lv,(np.nan,np.nan))[0],3)}–{fmt(d['fq'][f].get(lv,(np.nan,np.nan))[1],3)}",
                         fmt(d['rkf'][f].get(lv, np.nan), 2)]
                        for lv in d['lbf'][f]]
                _table([self.ftitle(f), "n", "Медіана", "Q1–Q3", "Сер. ранг"], rows); tno += 1
            _head(f"ТАБЛИЦЯ {tno}. Описова (непараметрична): варіанти")
            rows = [[d['v_names'][i], str(d['v_ns'].get(d['var_order'][i], 0)),
                     fmt(d['vmed'].get(d['var_order'][i], np.nan), 3),
                     f"{fmt(d['vq'].get(d['var_order'][i],(np.nan,np.nan))[0],3)}–{fmt(d['vq'].get(d['var_order'][i],(np.nan,np.nan))[1],3)}",
                     fmt(d['rkv'].get(d['v_names'][i], np.nan), 2)]
                    for i in range(d['n_var'])]
            _table(["Варіант", "n", "Медіана", "Q1–Q3", "Сер. ранг"], rows); tno += 1
            if d['method'] == "kw":
                if not d['do_ph']: _txt("Kruskal–Wallis p ≥ 0.05 → пост-хок не виконувався.")
                elif d['ph_rows']:
                    _head(f"ТАБЛИЦЯ {tno}. Парні порівняння (MWU + Bonferroni, Cliff's δ)")
                    _table(["Пара", "U", "p(Bonf.)", "Висновок", "δ", "Ефект"], d['ph_rows'])
            if d['method'] == "mw" and d['ph_rows']:
                _head(f"ТАБЛИЦЯ {tno}. Парні порівняння (MWU + Bonferroni, Cliff's δ)")
                _table(["Пара", "U", "p(Bonf.)", "Висновок", "δ", "Ефект"], d['ph_rows'])
            if d['method'] == "friedman":
                if not math.isnan(d['fr_p']) and d['fr_p'] >= ALPHA:
                    _txt("Friedman p ≥ 0.05 → пост-хок не виконувався.")
                elif d['rcbd_ph']:
                    _head(f"ТАБЛИЦЯ {tno}. Парні порівняння (Wilcoxon + Bonferroni)")
                    _table(["Пара", "W", "p(Bonf.)", "Висновок", "r"], d['rcbd_ph'])
        _sep()
        _txt(f"Звіт сформовано: {d['created'].strftime('%d.%m.%Y, %H:%M')}")

    # ════════════════════════════════════════════════════════════
    # GRAPHICAL REPORT  — 3 tabs
    # ════════════════════════════════════════════════════════════
    def show_graphs(self, long, letters_factor, indicator, units, eff_rows, pe2_rows):
        if not HAS_MPL: messagebox.showwarning("", "matplotlib недоступний."); return
        if self.graph_win and tk.Toplevel.winfo_exists(self.graph_win):
            self.graph_win.destroy()
        self.graph_win = gw = tk.Toplevel(self.root)
        gw.title("Графічний звіт"); gw.geometry("1300x840"); set_icon(gw)
        top = tk.Frame(gw, padx=8, pady=6); top.pack(fill=tk.X)
        tk.Button(top, text="⚙ Налаштування",
                  command=lambda: self._open_gs(gw, long, letters_factor, indicator, units, eff_rows, pe2_rows)
                  ).pack(side=tk.LEFT, padx=4)
        for tab_name, fig_key in [("Boxplot", "bp"), ("Сила впливу (Venn)", "vn"), ("Сила ефекту (Venn)", "pe")]:
            tk.Button(top, text=f"📋 Копіювати: {tab_name}",
                      command=lambda k=fig_key: self._copy_fig(k)).pack(side=tk.LEFT, padx=2)
        tk.Label(top, text="Вставте у Word через Ctrl+V").pack(side=tk.LEFT, padx=8)

        self._graph_frame = tk.Frame(gw); self._graph_frame.pack(fill=tk.BOTH, expand=True)
        self._g_long = long; self._g_lf = letters_factor
        self._g_ind = indicator; self._g_units = units
        self._g_eff = eff_rows; self._g_pe2 = pe2_rows
        self._draw_graphs(self._graph_frame, long, letters_factor, indicator, units, eff_rows, pe2_rows)

    def _open_gs(self, gw, long, lf, ind, units, eff, pe2):
        dlg = GraphSettingsDlg(gw, self.graph_settings)
        gw.wait_window(dlg)
        if dlg.result:
            self.graph_settings = dlg.result
            for w in self._graph_frame.winfo_children(): w.destroy()
            self._draw_graphs(self._graph_frame, long, lf, ind, units, eff, pe2)

    def _draw_graphs(self, frame, long, lf, indicator, units, eff_rows, pe2_rows):
        gs = self.graph_settings
        ff = gs["font_family"]; fst = gs["font_style"]; fz = gs["font_size"]
        fw = "bold" if "bold" in fst else "normal"; fi = "italic" if "italic" in fst else "normal"
        fp = {"fontfamily": ff, "fontsize": fz, "fontweight": fw, "fontstyle": fi}

        nb = ttk.Notebook(frame); nb.pack(fill=tk.BOTH, expand=True)

        # ── TAB 1: Boxplot ─────────────────────────────────────
        bp_frame = tk.Frame(nb); nb.add(bp_frame, text="Середнє по факторах (Boxplot)")
        fig_bp = Figure(figsize=(11, 5.5), dpi=100); ax = fig_bp.add_subplot(111)
        positions = []; data = []; xlbls = []; let_list = []; fcentres = []
        x = 1.; gap = 1.
        for f in self.factor_keys:
            lvls = self._lbf_cache.get(f, first_seen([r.get(f) for r in long if r.get(f) is not None]))
            if not lvls: continue
            sx = x
            for lv in lvls:
                arr = [float(r["value"]) for r in long if r.get(f) == lv and r.get("value") is not None]
                arr = [v for v in arr if not math.isnan(v)]
                data.append(arr); positions.append(x); xlbls.append(str(lv))
                let_list.append((f, lv)); x += 1.
            fcentres.append(((sx + x - 1) / 2., self.ftitle(f))); x += gap
        if data:
            bp = ax.boxplot(data, positions=positions, widths=0.6, showfliers=True, patch_artist=True)
            for patch in bp["boxes"]:    patch.set(facecolor=gs["box_color"])
            for line in bp["medians"]:   line.set(color=gs["median_color"], linewidth=2)
            for line in bp["whiskers"] + bp["caps"]: line.set(color=gs["whisker_color"])
            for fl in bp["fliers"]:      fl.set(markerfacecolor=gs["flier_color"], marker="o", markersize=4)
            ax.set_title(f"{indicator}, {units}", **fp)
            ax.set_ylabel(units, **fp)
            ax.set_xticks(positions); ax.set_xticklabels(xlbls, rotation=90,
                fontfamily=ff, fontsize=max(8, fz - 1))
            ax.yaxis.grid(True, linestyle="-", lw=0.5, alpha=0.35)
            allv = [v for a in data for v in a]
            dy = max(allv) - min(allv) if len(allv) > 1 else 1.
            off = 0.04 * dy if dy > 0 else 0.5
            for i, (f_, lv_) in enumerate(let_list):
                lt = (lf.get(f_, {}) or {}).get(lv_, "")
                if lt and data[i]: ax.text(positions[i], max(data[i]) + off, lt, ha="center", va="bottom", **fp)
            fig_bp.subplots_adjust(bottom=0.32, top=0.91, left=0.08, right=0.98)
            for cx, fnm in fcentres:
                ax.text(cx, -0.22, fnm, ha="center", va="top", transform=ax.get_xaxis_transform(), **fp)
        self._graph_figs["bp"] = fig_bp
        FigureCanvasTkAgg(fig_bp, master=bp_frame).draw() or None
        cv_bp = FigureCanvasTkAgg(fig_bp, master=bp_frame); cv_bp.draw()
        cv_bp.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ── TAB 2: Venn — сила впливу (% SS) ──────────────────
        vn_frame = tk.Frame(nb); nb.add(vn_frame, text="Сила впливу факторів (Венн)")
        fig_vn = Figure(figsize=(7, 6), dpi=100); ax2 = fig_vn.add_subplot(111)

        # Separate main factors from interactions
        main_eff  = [(nm, float(pct)) for nm, pct in eff_rows
                     if pct and "×" not in str(nm)]
        inter_eff = {frozenset(): None}  # placeholder
        inter_map = {}
        factor_labels = [self.ftitle(f) for f in self.factor_keys]
        for nm, pct in eff_rows:
            if "×" in str(nm) and pct:
                parts = str(nm).split("×")
                idxs  = frozenset(i for i, fl in enumerate(factor_labels) if fl in parts)
                inter_map[idxs] = (str(nm), float(pct))

        if main_eff:
            draw_venn(ax2, factor_values=main_eff, interaction_values=inter_map,
                      colors=gs["venn_colors"], alpha=gs["venn_alpha"],
                      font_size=gs["venn_font_size"], font_color=gs["venn_font_color"],
                      font_family=gs["font_family"], title="Сила впливу факторів (% від SS)")
        else:
            ax2.text(0.5, 0.5, "Недостатньо даних", ha="center", va="center", transform=ax2.transAxes)
            ax2.axis("off")
        self._graph_figs["vn"] = fig_vn
        cv_vn = FigureCanvasTkAgg(fig_vn, master=vn_frame); cv_vn.draw()
        cv_vn.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ── TAB 3: Venn — сила ефекту (partial η²) ────────────
        pe_frame = tk.Frame(nb); nb.add(pe_frame, text="Сила ефекту (partial η²)")
        fig_pe = Figure(figsize=(7, 6), dpi=100); ax3 = fig_pe.add_subplot(111)

        main_pe   = [(nm, float(pct) * 100) for nm, pct, _ in pe2_rows
                     if pct and "×" not in str(nm)]
        inter_pe  = {}
        for nm, pct, _ in pe2_rows:
            if "×" in str(nm) and pct:
                parts = str(nm).split("×")
                idxs  = frozenset(i for i, fl in enumerate(factor_labels) if fl in parts)
                inter_pe[idxs] = (str(nm), float(pct) * 100)

        if main_pe:
            draw_venn(ax3, factor_values=main_pe, interaction_values=inter_pe,
                      colors=gs["venn_colors"], alpha=gs["venn_alpha"],
                      font_size=gs["venn_font_size"], font_color=gs["venn_font_color"],
                      font_family=gs["font_family"], title="Розмір ефекту (partial η², %)")
        else:
            ax3.text(0.5, 0.5, "Недостатньо даних", ha="center", va="center", transform=ax3.transAxes)
            ax3.axis("off")
        self._graph_figs["pe"] = fig_pe
        cv_pe = FigureCanvasTkAgg(fig_pe, master=pe_frame); cv_pe.draw()
        cv_pe.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _copy_fig(self, key):
        fig = self._graph_figs.get(key)
        if fig is None: messagebox.showwarning("", "Графік відсутній."); return
        ok, msg = _copy_fig_to_clipboard(fig)
        if ok: messagebox.showinfo("", "Графік скопійовано (PNG).")
        else:  messagebox.showwarning("", f"Помилка: {msg}")



# ═══════════════════════════════════════════════════════════════
# EXPORT  — Word (.docx) and PDF via reportlab
# ═══════════════════════════════════════════════════════════════
def export_report_docx(text_lines, tables, filepath):
    """Export plain-text + table data to .docx using python-docx."""
    try:
        from docx import Document
        from docx.shared import Pt, Cm
        from docx.enum.text import WD_ALIGN_PARAGRAPH
    except ImportError:
        raise RuntimeError("Встановіть python-docx:\n  pip install python-docx")
    doc = Document()
    style = doc.styles['Normal']; style.font.name = 'Times New Roman'; style.font.size = Pt(12)
    for section in doc.sections:
        section.top_margin = Cm(2); section.bottom_margin = Cm(2)
        section.left_margin = Cm(2.5); section.right_margin = Cm(2)
    for item in text_lines:
        kind = item.get("kind", "text")
        if kind == "heading":
            p = doc.add_heading(item["text"], level=item.get("level", 2))
            p.runs[0].font.name = 'Times New Roman'
        elif kind == "table":
            headers = item["headers"]; rows = item["rows"]
            tbl = doc.add_table(rows=1 + len(rows), cols=len(headers))
            tbl.style = 'Table Grid'
            hrow = tbl.rows[0]
            for j, h in enumerate(headers):
                hrow.cells[j].text = str(h)
                run = hrow.cells[j].paragraphs[0].runs[0]
                run.bold = True; run.font.name = 'Times New Roman'; run.font.size = Pt(11)
            for i, row in enumerate(rows):
                for j, val in enumerate(row):
                    cell = tbl.rows[i+1].cells[j]
                    cell.text = "" if val is None else str(val)
                    cell.paragraphs[0].runs[0].font.name = 'Times New Roman' if cell.paragraphs[0].runs else None
            doc.add_paragraph()
        else:
            p = doc.add_paragraph(item.get("text", ""))
            p.runs[0].font.name = 'Times New Roman' if p.runs else None
    doc.save(filepath)

def export_report_pdf(text_lines, filepath):
    """Export plain text to PDF using reportlab."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors
    except ImportError:
        raise RuntimeError("Встановіть reportlab:\n  pip install reportlab")
    doc = SimpleDocTemplate(filepath, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2.5*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle('h1', parent=styles['Heading2'], fontName='Times-Roman', fontSize=12, spaceAfter=4)
    normal = ParagraphStyle('n', parent=styles['Normal'], fontName='Times-Roman', fontSize=11, spaceAfter=2)
    story = []
    for item in text_lines:
        kind = item.get("kind","text")
        if kind == "heading":
            story.append(Paragraph(item["text"], h1))
        elif kind == "table":
            headers = item["headers"]; rows = item["rows"]
            data = [headers] + [[("" if v is None else str(v)) for v in r] for r in rows]
            t = Table(data, repeatRows=1)
            t.setStyle(TableStyle([
                ('FONTNAME',(0,0),(-1,0),'Times-Roman'), ('FONTSIZE',(0,0),(-1,0),10),
                ('FONTNAME',(0,1),(-1,-1),'Times-Roman'), ('FONTSIZE',(0,1),(-1,-1),9),
                ('GRID',(0,0),(-1,-1),0.5,colors.black),
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8e8e8')),
            ]))
            story.append(t); story.append(Spacer(1,6))
        else:
            txt = item.get("text","").replace("\n","<br/>")
            if txt.strip(): story.append(Paragraph(txt, normal))
    doc.build(story)


# ═══════════════════════════════════════════════════════════════
# DESCRIPTIVE STATISTICS — standalone module
# ═══════════════════════════════════════════════════════════════
class DescriptiveWindow:
    """Standalone descriptive statistics module."""
    def __init__(self, parent, gs):
        self.win = tk.Toplevel(parent); self.win.title("Descriptive Statistics")
        self.win.geometry("1100x680"); set_icon(self.win)
        self.gs = gs; self._build()

    def _build(self):
        mb = tk.Menu(self.win); self.win.config(menu=mb)
        fm = tk.Menu(mb, tearoff=0)
        fm.add_command(label="Load Excel", command=self._load_excel)
        fm.add_command(label="Paste from clipboard", command=self._paste)
        mb.add_cascade(label="File", menu=fm)

        top = tk.Frame(self.win, padx=6, pady=4); top.pack(fill=tk.X)
        tk.Button(top, text="▶ Analyze", bg="#c62828", fg="white",
                  font=("Times New Roman",12), command=self._analyze).pack(side=tk.LEFT, padx=4)
        tk.Label(top, text="First row = variable names  |  Each column = one variable",
                 font=("Times New Roman",11), fg="#555").pack(side=tk.LEFT, padx=8)

        # data table
        tf = tk.Frame(self.win); tf.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        self.rows = 20; self.cols = 8
        canvas = tk.Canvas(tf); canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(tf, orient="vertical", command=canvas.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y); canvas.configure(yscrollcommand=sb.set)
        self.inner = tk.Frame(canvas); canvas.create_window((0,0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: canvas.config(scrollregion=canvas.bbox("all")))
        self._canvas = canvas
        self.entries = []
        for j in range(self.cols):
            lbl = tk.Label(self.inner, text=f"Var {j+1}", relief=tk.RIDGE, width=12,
                           bg="#f0f0f0", font=("Times New Roman",11))
            lbl.grid(row=0, column=j, padx=1, pady=1, sticky="nsew")
        for i in range(self.rows):
            row_ = []
            for j in range(self.cols):
                e = tk.Entry(self.inner, width=12, font=("Times New Roman",11))
                e.grid(row=i+1, column=j, padx=1, pady=1)
                row_.append(e)
            self.entries.append(row_)

    def _paste(self):
        w = self.win.focus_get()
        if not isinstance(w, tk.Entry): return
        try: data = self.win.clipboard_get()
        except Exception: return
        for i, line in enumerate(data.splitlines()):
            if not line: continue
            for j, val in enumerate(line.split("\t")):
                if i < len(self.entries) and j < self.cols:
                    self.entries[i][j].delete(0,tk.END); self.entries[i][j].insert(0,val)

    def _load_excel(self):
        if not HAS_OPENPYXL: messagebox.showerror("","pip install openpyxl"); return
        path = filedialog.askopenfilename(filetypes=[("Excel","*.xlsx *.xlsm"),("All","*.*")])
        if not path: return
        try:
            wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
            ws = wb.active
            raw = [[cell for cell in row] for row in ws.iter_rows(values_only=True)]
            wb.close()
            while len(self.entries) < len(raw):
                i = len(self.entries); row_ = []
                for j in range(self.cols):
                    e = tk.Entry(self.inner, width=12, font=("Times New Roman",11))
                    e.grid(row=i+1, column=j, padx=1, pady=1); row_.append(e)
                self.entries.append(row_); self.rows += 1
            for i, row in enumerate(raw):
                for j, v in enumerate(row):
                    if i < len(self.entries) and j < self.cols:
                        self.entries[i][j].delete(0,tk.END)
                        self.entries[i][j].insert(0,"" if v is None else str(v))
        except Exception as ex: messagebox.showerror("",str(ex))

    def _analyze(self):
        # read data: first row = names
        raw = [[e.get().strip() for e in row] for row in self.entries]
        # find names row (first non-empty)
        names = []; data_cols = []
        for j in range(self.cols):
            col_vals = []; col_name = ""
            for i, row in enumerate(raw):
                v = row[j] if j < len(row) else ""
                if not v: continue
                if not col_name:
                    try: float(v.replace(",",".")); col_name = f"Var{j+1}"; col_vals.append(float(v.replace(",",".")))
                    except ValueError: col_name = v
                else:
                    try: col_vals.append(float(v.replace(",",".")))
                    except Exception: continue
            if col_name and col_vals:
                names.append(col_name); data_cols.append(np.array(col_vals, dtype=float))

        if not data_cols: messagebox.showwarning("","No numeric data found."); return

        # compute stats
        from scipy.stats import skew, kurtosis
        headers = ["Variable","n","Mean","SD","SE","Min","Max","Median",
                   "Q1","Q3","CV%","Skewness","Kurtosis","95% CI low","95% CI high","SW p"]
        rows = []
        for nm, arr in zip(names, data_cols):
            a = arr[~np.isnan(arr)]; n = len(a)
            if n < 2: rows.append([nm, n] + ["–"]*14); continue
            m = float(np.mean(a)); sd = float(np.std(a, ddof=1))
            se = sd / math.sqrt(n)
            ci_lo = m - float(t_dist.ppf(0.975, n-1)) * se
            ci_hi = m + float(t_dist.ppf(0.975, n-1)) * se
            sk = float(skew(a)); ku = float(kurtosis(a))
            q1, q3 = float(np.percentile(a,25)), float(np.percentile(a,75))
            cv = sd/m*100 if m != 0 else np.nan
            try: _, sw_p = shapiro(a)
            except Exception: sw_p = np.nan
            rows.append([nm, n, fmt(m,3), fmt(sd,3), fmt(se,3),
                         fmt(float(np.min(a)),3), fmt(float(np.max(a)),3),
                         fmt(float(np.median(a)),3), fmt(q1,3), fmt(q3,3),
                         fmt(cv,2), fmt(sk,3), fmt(ku,3),
                         fmt(ci_lo,3), fmt(ci_hi,3), fmt(sw_p,4)])

        self._show_result(headers, rows, data_cols, names)

    def _show_result(self, headers, rows, arrays, names):
        win = tk.Toplevel(self.win); win.title("Descriptive Statistics — Results")
        win.geometry("1300x600"); set_icon(win)
        top = tk.Frame(win, padx=6, pady=4); top.pack(fill=tk.X)
        def export_docx():
            p = filedialog.asksaveasfilename(defaultextension=".docx",filetypes=[("Word","*.docx")])
            if not p: return
            try:
                export_report_docx([{"kind":"heading","text":"Descriptive Statistics","level":1},
                                     {"kind":"table","headers":headers,"rows":rows}], [], p)
                messagebox.showinfo("","Saved.")
            except Exception as ex: messagebox.showerror("",str(ex))
        tk.Button(top, text="Export Word", command=export_docx).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Plot Boxplots", command=lambda: self._plot_boxes(arrays, names)).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="QQ-plots", command=lambda: self._plot_qq(arrays, names)).pack(side=tk.LEFT, padx=4)

        frm, _ = make_tv(win, headers, rows, min_col=80)
        frm.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    def _plot_boxes(self, arrays, names):
        if not HAS_MPL: return
        win = tk.Toplevel(self.win); win.title("Boxplots"); win.geometry("900x550")
        fig = Figure(figsize=(max(6, len(arrays)*0.9+1), 5), dpi=100)
        ax = fig.add_subplot(111)
        ax.boxplot([a[~np.isnan(a)] for a in arrays], labels=names, patch_artist=True)
        ax.set_ylabel("Value"); fig.tight_layout()
        cv = FigureCanvasTkAgg(fig, master=win); cv.draw(); cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _plot_qq(self, arrays, names):
        if not HAS_MPL: return
        from scipy.stats import probplot
        n = len(arrays); cols = min(n, 4); rows_n = math.ceil(n/cols)
        win = tk.Toplevel(self.win); win.title("QQ-plots"); win.geometry("1000x600")
        fig = Figure(figsize=(cols*2.5+0.5, rows_n*2.5+0.5), dpi=100)
        for i, (arr, nm) in enumerate(zip(arrays, names)):
            a = arr[~np.isnan(arr)]
            if len(a) < 3: continue
            ax = fig.add_subplot(rows_n, cols, i+1)
            res = probplot(a, dist="norm")
            ax.plot(res[0][0], res[0][1], 'o', markersize=3, color='#4c72b0')
            ax.plot(res[0][0], res[1][1] + res[1][0]*res[0][0], 'r-', lw=1)
            ax.set_title(nm, fontsize=9); ax.set_xlabel("Theoretical"); ax.set_ylabel("Sample")
        fig.tight_layout()
        cv = FigureCanvasTkAgg(fig, master=win); cv.draw(); cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)


# ═══════════════════════════════════════════════════════════════
# T-TEST MODULE
# ═══════════════════════════════════════════════════════════════
class TTestWindow:
    def __init__(self, parent):
        self.win = tk.Toplevel(parent); self.win.title("t-Test / Mann-Whitney")
        self.win.geometry("700x560"); set_icon(self.win); self._build()

    def _build(self):
        frm = tk.Frame(self.win, padx=12, pady=10); frm.pack(fill=tk.BOTH, expand=True)
        tk.Label(frm, text="Test type:", font=("Times New Roman",12,"bold")).grid(row=0,column=0,sticky="w")
        self.test_var = tk.StringVar(value="ind")
        rf = ("Times New Roman",12)
        for txt, val, r in [("Independent samples (2 groups)","ind",1),
                             ("Paired samples (before/after)","paired",2),
                             ("One sample (vs. known mean)","one",3)]:
            tk.Radiobutton(frm, text=txt, variable=self.test_var, value=val, font=rf,
                           command=self._update_ui).grid(row=r, column=0, sticky="w")

        tk.Label(frm, text="Group 1 / Sample:", font=("Times New Roman",12)).grid(row=4,column=0,sticky="w",pady=(10,2))
        self.e1 = tk.Text(frm, width=50, height=4, font=("Times New Roman",11))
        self.e1.grid(row=5, column=0, columnspan=2, sticky="ew")
        tk.Label(frm, text="(comma or space or newline separated)", font=("Times New Roman",10), fg="#666").grid(row=6,column=0,sticky="w")

        self.lbl2 = tk.Label(frm, text="Group 2:", font=("Times New Roman",12))
        self.lbl2.grid(row=7,column=0,sticky="w",pady=(8,2))
        self.e2 = tk.Text(frm, width=50, height=4, font=("Times New Roman",11))
        self.e2.grid(row=8, column=0, columnspan=2, sticky="ew")

        self.lbl_mu = tk.Label(frm, text="Known mean (μ₀):", font=("Times New Roman",12))
        self.e_mu = tk.Entry(frm, width=12, font=("Times New Roman",12)); self.e_mu.insert(0,"0")

        # alpha
        tk.Label(frm, text="α:", font=("Times New Roman",12)).grid(row=11,column=0,sticky="w",pady=(8,2))
        self.alpha_var = tk.StringVar(value="0.05")
        ttk.Combobox(frm, textvariable=self.alpha_var, values=["0.01","0.05","0.10"],
                     state="readonly", width=8).grid(row=11, column=1, sticky="w")

        tk.Button(frm, text="▶ Run", bg="#c62828", fg="white", font=("Times New Roman",13),
                  command=self._run).grid(row=12, column=0, pady=12, sticky="w")
        self.result_var = tk.StringVar()
        tk.Label(frm, textvariable=self.result_var, font=("Times New Roman",11),
                 justify="left", wraplength=620).grid(row=13, column=0, columnspan=2, sticky="w")
        self._update_ui()

    def _update_ui(self):
        t = self.test_var.get()
        if t == "one":
            self.lbl2.grid_remove(); self.e2.grid_remove()
            self.lbl_mu.grid(row=7, column=0, sticky="w", pady=(8,2))
            self.e_mu.grid(row=7, column=1, sticky="w", padx=6)
        else:
            self.lbl_mu.grid_remove(); self.e_mu.grid_remove()
            self.lbl2.grid(row=7, column=0, sticky="w", pady=(8,2))
            self.e2.grid(row=8, column=0, columnspan=2, sticky="ew")
        txt = "Group 2 (paired — same order as Group 1):" if t == "paired" else "Group 2:"
        self.lbl2.configure(text=txt)

    def _parse(self, widget):
        import re
        txt = widget.get("1.0", tk.END).strip()
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", txt.replace(",","."))
        return np.array([float(x) for x in nums], dtype=float)

    def _run(self):
        from scipy.stats import ttest_ind, ttest_rel, ttest_1samp
        alpha = float(self.alpha_var.get())
        x1 = self._parse(self.e1)
        t = self.test_var.get()
        if len(x1) < 2: self.result_var.set("Group 1 needs ≥ 2 values."); return

        lines = []
        lines.append(f"n₁ = {len(x1)},  Mean₁ = {fmt(np.mean(x1),4)},  SD₁ = {fmt(np.std(x1,ddof=1),4)}")

        # normality check
        try: _, sw1 = shapiro(x1)
        except Exception: sw1 = np.nan
        normal1 = (not math.isnan(sw1)) and sw1 > 0.05
        lines.append(f"Shapiro–Wilk (Group 1): W={fmt(sw1,4)} → {'normal' if normal1 else 'NOT normal'}")

        if t == "one":
            try: mu0 = float(self.e_mu.get())
            except Exception: mu0 = 0.0
            stat, p = ttest_1samp(x1, mu0)
            lines.append(f"\nOne-sample t-test (μ₀={mu0}):")
            lines.append(f"t = {fmt(stat,4)},  df = {len(x1)-1},  p = {fmt(p,4)}")
            lines.append("→ Significant difference" if p < alpha else "→ No significant difference")
        else:
            x2 = self._parse(self.e2)
            if len(x2) < 2: self.result_var.set("Group 2 needs ≥ 2 values."); return
            try: _, sw2 = shapiro(x2)
            except Exception: sw2 = np.nan
            normal2 = (not math.isnan(sw2)) and sw2 > 0.05
            lines.append(f"n₂ = {len(x2)},  Mean₂ = {fmt(np.mean(x2),4)},  SD₂ = {fmt(np.std(x2,ddof=1),4)}")
            lines.append(f"Shapiro–Wilk (Group 2): W={fmt(sw2,4)} → {'normal' if normal2 else 'NOT normal'}")

            if t == "paired":
                if len(x1) != len(x2):
                    self.result_var.set("Paired test requires equal sample sizes."); return
                if normal1 and normal2:
                    stat, p = ttest_rel(x1, x2)
                    lines.append(f"\nPaired t-test:  t={fmt(stat,4)},  df={len(x1)-1},  p={fmt(p,4)}")
                else:
                    stat, p = wilcoxon(x1, x2, zero_method="wilcox", alternative="two-sided", mode="auto")
                    lines.append(f"\nWilcoxon signed-rank (non-normal):  W={fmt(stat,4)},  p={fmt(p,4)}")
            else:
                # check variance homogeneity
                try: lev_s, lev_p = levene(x1, x2, center='median')
                except Exception: lev_p = np.nan
                equal_var = (not math.isnan(lev_p)) and lev_p >= 0.05
                lines.append(f"Levene test: p={fmt(lev_p,4)} → {'equal variances' if equal_var else 'unequal variances'}")
                if normal1 and normal2:
                    stat, p = ttest_ind(x1, x2, equal_var=equal_var)
                    test_name = "Independent t-test" if equal_var else "Welch t-test (unequal var)"
                    n1, n2 = len(x1), len(x2)
                    df_w = (np.var(x1,ddof=1)/n1 + np.var(x2,ddof=1)/n2)**2 / \
                           ((np.var(x1,ddof=1)/n1)**2/(n1-1) + (np.var(x2,ddof=1)/n2)**2/(n2-1)) if not equal_var else n1+n2-2
                    lines.append(f"\n{test_name}:  t={fmt(stat,4)},  df≈{fmt(df_w,1)},  p={fmt(p,4)}")
                else:
                    U, p = mannwhitneyu(x1, x2, alternative="two-sided")
                    d = cliffs_d(x1, x2)
                    lines.append(f"\nMann–Whitney U (non-normal):  U={fmt(U,3)},  p={fmt(p,4)}")
                    lines.append(f"Cliff's δ = {fmt(d,4)}  ({cliffs_lbl(abs(d))} effect)")

            lines.append("→ Significant difference" if p < alpha else "→ No significant difference")

        self.result_var.set("\n".join(lines))


# ═══════════════════════════════════════════════════════════════
# OUTLIER DETECTION
# ═══════════════════════════════════════════════════════════════
def detect_outliers_grubbs(arr, alpha=0.05):
    """Grubbs test for single outlier. Returns (idx_of_outlier or None, G, p)."""
    a = np.array(arr, dtype=float); n = len(a)
    if n < 3: return None, np.nan, np.nan
    m = np.mean(a); s = np.std(a, ddof=1)
    if s == 0: return None, np.nan, np.nan
    G = np.max(np.abs(a - m)) / s
    idx = int(np.argmax(np.abs(a - m)))
    # critical value via t-distribution
    t_crit = float(t_dist.ppf(1 - alpha/(2*n), n-2))
    G_crit = ((n-1)/math.sqrt(n)) * math.sqrt(t_crit**2 / (n-2+t_crit**2))
    p_approx = 2 * n * (1 - float(t_dist.cdf(G * math.sqrt(n) * math.sqrt(n-2) /
               math.sqrt(n-1+G**2*n/(n-1)), n-2))) if n > 2 else np.nan
    return (idx if G > G_crit else None), float(G), float(p_approx)

def detect_outliers_iqr(arr):
    """IQR method. Returns list of indices."""
    a = np.array(arr, dtype=float)
    q1, q3 = np.percentile(a, 25), np.percentile(a, 75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    return [i for i, v in enumerate(a) if v < lo or v > hi]


# ═══════════════════════════════════════════════════════════════
# REGRESSION MODULE
# ═══════════════════════════════════════════════════════════════
class RegressionWindow:
    MODELS = ["Linear:  y = a + bx",
              "Quadratic:  y = a + bx + cx²",
              "Cubic:  y = a + bx + cx² + dx³",
              "Power:  y = a·xᵇ",
              "Exponential:  y = a·eᵇˣ",
              "Logarithmic:  y = a + b·ln(x)",
              "Logistic (4-param):  y = d + (a-d)/(1+(x/c)ᵇ)"]

    def __init__(self, parent, gs):
        self.win = tk.Toplevel(parent); self.win.title("Regression Analysis")
        self.win.geometry("1050x720"); set_icon(self.win); self.gs = gs; self._build()

    def _build(self):
        top = tk.Frame(self.win, padx=8, pady=6); top.pack(fill=tk.X)
        tk.Label(top, text="Model:", font=("Times New Roman",12)).pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value=self.MODELS[0])
        ttk.Combobox(top, textvariable=self.model_var, values=self.MODELS,
                     state="readonly", width=44, font=("Times New Roman",11)).pack(side=tk.LEFT, padx=6)
        tk.Label(top, text="α:", font=("Times New Roman",12)).pack(side=tk.LEFT, padx=(10,2))
        self.alpha_var = tk.StringVar(value="0.05")
        ttk.Combobox(top, textvariable=self.alpha_var, values=["0.01","0.05","0.10"],
                     state="readonly", width=7).pack(side=tk.LEFT)
        tk.Button(top, text="▶ Run", bg="#c62828", fg="white",
                  font=("Times New Roman",13), command=self._run).pack(side=tk.LEFT, padx=10)
        tk.Button(top, text="Paste data", command=self._paste).pack(side=tk.LEFT, padx=4)

        # data entry
        mid = tk.Frame(self.win); mid.pack(fill=tk.X, padx=8)
        for j, nm in enumerate(["x (independent)","y (dependent)"]):
            tk.Label(mid, text=nm, font=("Times New Roman",11,"bold")).grid(row=0, column=j, padx=4)
        self.tx = tk.Text(mid, width=36, height=14, font=("Times New Roman",11)); self.tx.grid(row=1,column=0,padx=4,pady=2)
        self.ty = tk.Text(mid, width=36, height=14, font=("Times New Roman",11)); self.ty.grid(row=1,column=1,padx=4,pady=2)
        tk.Label(mid, text="Enter values one per line or comma-separated",
                 font=("Times New Roman",10), fg="#666").grid(row=2,column=0,columnspan=2,sticky="w",padx=4)

        # result area
        self.res_frame = tk.Frame(self.win, padx=8, pady=4); self.res_frame.pack(fill=tk.BOTH, expand=True)

    def _paste(self):
        try: data = self.win.clipboard_get()
        except Exception: return
        lines = [l.strip() for l in data.splitlines() if l.strip()]
        xs, ys = [], []
        for line in lines:
            parts = line.replace(",",".").split()
            if len(parts) >= 2:
                try: xs.append(parts[0]); ys.append(parts[1])
                except Exception: pass
        self.tx.delete("1.0",tk.END); self.tx.insert("1.0","\n".join(xs))
        self.ty.delete("1.0",tk.END); self.ty.insert("1.0","\n".join(ys))

    def _parse_col(self, widget):
        import re
        txt = widget.get("1.0",tk.END).replace(",",".")
        return np.array([float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",txt)], dtype=float)

    def _run(self):
        from scipy.optimize import curve_fit
        from scipy.stats import f as f_dist_
        alpha = float(self.alpha_var.get())
        x = self._parse_col(self.tx); y = self._parse_col(self.ty)
        n = min(len(x),len(y)); x = x[:n]; y = y[:n]
        if n < 4: messagebox.showwarning("","Need ≥ 4 data points."); return

        model_name = self.model_var.get().split(":")[0].strip()
        result = self._fit_model(model_name, x, y, alpha)
        if result is None: return
        self._show_result(result, x, y, model_name, alpha)

    def _fit_model(self, name, x, y, alpha):
        from scipy.optimize import curve_fit
        try:
            if name == "Linear":
                X = np.column_stack([np.ones(len(x)), x])
                beta, _, res, _, _ = np.linalg.lstsq(X, y, rcond=None)
                yhat = X @ beta
                params = {"a": beta[0], "b": beta[1]}
                eq = f"y = {fmt(beta[0],4)} + {fmt(beta[1],4)}·x"
            elif name == "Quadratic":
                X = np.column_stack([np.ones(len(x)), x, x**2])
                beta, _, res, _, _ = np.linalg.lstsq(X, y, rcond=None)
                yhat = X @ beta
                params = {"a": beta[0], "b": beta[1], "c": beta[2]}
                eq = f"y = {fmt(beta[0],4)} + {fmt(beta[1],4)}·x + {fmt(beta[2],4)}·x²"
            elif name == "Cubic":
                X = np.column_stack([np.ones(len(x)), x, x**2, x**3])
                beta, _, res, _, _ = np.linalg.lstsq(X, y, rcond=None)
                yhat = X @ beta
                params = {"a": beta[0], "b": beta[1], "c": beta[2], "d": beta[3]}
                eq = f"y = {fmt(beta[0],4)} + {fmt(beta[1],4)}·x + {fmt(beta[2],4)}·x² + {fmt(beta[3],4)}·x³"
            elif name == "Power":
                if np.any(x <= 0): messagebox.showwarning("","Power model requires x > 0."); return None
                lx, ly = np.log(x), np.log(y)
                X = np.column_stack([np.ones(len(lx)), lx])
                beta, *_ = np.linalg.lstsq(X, ly, rcond=None)
                a, b = math.exp(beta[0]), beta[1]
                yhat = a * x**b
                params = {"a": a, "b": b}
                eq = f"y = {fmt(a,4)}·x^{fmt(b,4)}"
            elif name == "Exponential":
                X = np.column_stack([np.ones(len(x)), x])
                beta, *_ = np.linalg.lstsq(X, np.log(np.abs(y)+1e-10), rcond=None)
                a, b = math.exp(beta[0]), beta[1]
                yhat = a * np.exp(b * x)
                params = {"a": a, "b": b}
                eq = f"y = {fmt(a,4)}·e^({fmt(b,4)}·x)"
            elif name == "Logarithmic":
                if np.any(x <= 0): messagebox.showwarning("","Logarithmic model requires x > 0."); return None
                X = np.column_stack([np.ones(len(x)), np.log(x)])
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                yhat = X @ beta
                params = {"a": beta[0], "b": beta[1]}
                eq = f"y = {fmt(beta[0],4)} + {fmt(beta[1],4)}·ln(x)"
            elif name == "Logistic (4-param)":
                def logistic4(xx, a, b, c, d):
                    return d + (a - d) / (1 + (xx/c)**b)
                p0 = [max(y), 1, np.median(x), min(y)]
                popt, _ = curve_fit(logistic4, x, y, p0=p0, maxfev=10000)
                yhat = logistic4(x, *popt)
                params = {"a":popt[0],"b":popt[1],"c":popt[2],"d":popt[3]}
                eq = f"y = {fmt(popt[3],4)} + ({fmt(popt[0],4)}-{fmt(popt[3],4)})/(1+(x/{fmt(popt[2],4)})^{fmt(popt[1],4)})"
            else:
                return None

            residuals = y - yhat; sse = float(np.sum(residuals**2))
            sst = float(np.sum((y - np.mean(y))**2))
            R2 = 1 - sse/sst if sst > 0 else np.nan
            n = len(x); k = len(params)
            R2_adj = 1 - (1-R2)*(n-1)/(n-k-1) if n > k+1 else np.nan
            mse = sse/(n-k) if n > k else np.nan
            rmse = math.sqrt(mse) if not math.isnan(mse) else np.nan
            # F-test
            msm = (sst - sse)/k if k > 0 else np.nan
            F = msm/mse if (not math.isnan(mse) and mse > 0) else np.nan
            p_F = float(1-f_dist.cdf(F,k,n-k-1)) if (not math.isnan(F) and n>k+1) else np.nan
            # Normality of residuals
            try: _, sw_p = shapiro(residuals)
            except Exception: sw_p = np.nan
            return {"equation":eq,"params":params,"R2":R2,"R2_adj":R2_adj,
                    "RMSE":rmse,"F":F,"p_F":p_F,"sw_p":sw_p,
                    "residuals":residuals,"yhat":yhat,"sse":sse,"sst":sst,"n":n,"k":k}
        except Exception as ex:
            messagebox.showerror("Fitting error", str(ex)); return None

    def _show_result(self, r, x, y, model_name, alpha):
        for w in self.res_frame.winfo_children(): w.destroy()

        # text summary
        info = (f"Model: {model_name}\n"
                f"Equation: {r['equation']}\n"
                f"R² = {fmt(r['R2'],4)}   R²adj = {fmt(r['R2_adj'],4)}   RMSE = {fmt(r['RMSE'],4)}\n"
                f"F = {fmt(r['F'],4)},  p = {fmt(r['p_F'],4)} {'✓ significant' if r['p_F'] is not None and not math.isnan(r['p_F']) and r['p_F'] < alpha else '✗ not significant'}\n"
                f"Shapiro–Wilk (residuals): p = {fmt(r['sw_p'],4)}  "
                f"{'✓ residuals normal' if r['sw_p'] is not None and not math.isnan(r['sw_p']) and r['sw_p'] > 0.05 else '⚠ residuals NOT normal'}")

        tk.Label(self.res_frame, text=info, font=("Times New Roman",11),
                 justify="left", anchor="w").pack(anchor="w")

        # plot
        if HAS_MPL:
            fig = Figure(figsize=(10, 3.8), dpi=100)
            ax1 = fig.add_subplot(121); ax2 = fig.add_subplot(122)
            x_sort = np.sort(x); idx_sort = np.argsort(x)
            # scatter + fit
            ax1.scatter(x, y, s=25, color="#4c72b0", zorder=3, label="Observed")
            ax1.plot(x_sort, r["yhat"][idx_sort], "r-", lw=2, label="Fitted")
            ax1.set_xlabel("x"); ax1.set_ylabel("y")
            ax1.set_title(f"Fit:  R²={fmt(r['R2'],3)}"); ax1.legend(fontsize=9)
            ax1.yaxis.grid(True, alpha=0.3)
            # residuals
            ax2.scatter(r["yhat"], r["residuals"], s=25, color="#dd8452")
            ax2.axhline(0, color="k", lw=0.8)
            ax2.set_xlabel("Fitted values"); ax2.set_ylabel("Residuals")
            ax2.set_title("Residuals vs Fitted"); ax2.yaxis.grid(True, alpha=0.3)
            fig.tight_layout()
            cv = FigureCanvasTkAgg(fig, master=self.res_frame)
            cv.draw(); cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # outlier check
        out_idx, G, _ = detect_outliers_grubbs(r["residuals"])
        if out_idx is not None:
            tk.Label(self.res_frame,
                     text=f"⚠ Grubbs outlier detected in residuals: observation #{out_idx+1}  (G={fmt(G,3)})",
                     fg="#c62828", font=("Times New Roman",11)).pack(anchor="w")


# ═══════════════════════════════════════════════════════════════
# SAMPLE SIZE CALCULATOR
# ═══════════════════════════════════════════════════════════════
class SampleSizeWindow:
    def __init__(self, parent):
        self.win = tk.Toplevel(parent); self.win.title("Sample Size Calculator")
        self.win.geometry("600x520"); self.win.resizable(False, False); set_icon(self.win)
        self._build()

    def _build(self):
        frm = tk.Frame(self.win, padx=16, pady=14); frm.pack(fill=tk.BOTH, expand=True)
        tk.Label(frm, text="Sample Size & Statistical Power Calculator",
                 font=("Times New Roman",13,"bold")).grid(row=0,column=0,columnspan=3,pady=(0,12))
        rf = ("Times New Roman",12)
        params = [
            ("Design:", None, "design"),
            ("α (significance level):", "0.05", "alpha"),
            ("Power (1-β):", "0.80", "power"),
            ("Expected difference (δ):", "", "delta"),
            ("Standard deviation (σ):", "", "sigma"),
            ("Number of treatments (k):", "3", "k"),
            ("Number of replications (r) — leave blank to calculate:", "", "r"),
        ]
        self.vars = {}
        row_i = 1
        for label, default, key in params:
            tk.Label(frm, text=label, font=rf).grid(row=row_i, column=0, sticky="w", pady=4)
            if key == "design":
                var = tk.StringVar(value="CRD")
                ttk.Combobox(frm, textvariable=var, values=["CRD","RCBD","Split-plot"],
                             state="readonly", width=16).grid(row=row_i, column=1, sticky="w", padx=6)
            else:
                var = tk.StringVar(value=default or "")
                tk.Entry(frm, textvariable=var, width=14, font=rf).grid(row=row_i,column=1,sticky="w",padx=6)
            self.vars[key] = var; row_i += 1

        tk.Button(frm, text="▶ Calculate", bg="#c62828", fg="white", font=rf,
                  command=self._calc).grid(row=row_i, column=0, columnspan=2, pady=14)
        self.res_var = tk.StringVar()
        tk.Label(frm, textvariable=self.res_var, font=("Times New Roman",11),
                 justify="left", wraplength=540).grid(row=row_i+1, column=0, columnspan=3, sticky="w")

    def _calc(self):
        try:
            alpha = float(self.vars["alpha"].get())
            power = float(self.vars["power"].get())
            delta = float(self.vars["delta"].get())
            sigma = float(self.vars["sigma"].get())
            k = int(self.vars["k"].get())
            design = self.vars["design"].get()
            r_str = self.vars["r"].get().strip()
        except ValueError:
            self.res_var.set("Please fill in all numeric fields."); return

        if delta <= 0 or sigma <= 0 or k < 2:
            self.res_var.set("δ and σ must be positive; k ≥ 2."); return

        z_alpha = float(norm.ppf(1 - alpha/2))
        z_beta  = float(norm.ppf(power))

        lines = []
        if r_str:
            # given r — compute power
            r = int(r_str)
            # approximate formula for ANOVA power
            lambda_nc = k * r * (delta**2) / (2 * sigma**2)
            from scipy.stats import ncf
            F_crit = float(f_dist.ppf(1-alpha, k-1, k*(r-1)))
            achieved_power = float(1 - ncf.cdf(F_crit, k-1, k*(r-1), lambda_nc))
            lines.append(f"Design: {design},  k={k} treatments,  r={r} replications")
            lines.append(f"Non-centrality λ = {fmt(lambda_nc,3)}")
            lines.append(f"F_crit(α={alpha}) = {fmt(F_crit,3)}")
            lines.append(f"Achieved power (1-β) = {fmt(achieved_power,4)}")
            lines.append(f"\n{'✓ Sufficient power' if achieved_power >= power else '✗ Insufficient power — increase replications'}")
        else:
            # calculate r needed
            # iterative: increase r until power achieved
            from scipy.stats import ncf
            lines.append(f"Design: {design},  k={k} treatments")
            lines.append(f"Target: α={alpha}, power={power}, δ={delta}, σ={sigma}\n")
            found = False
            for r in range(2, 101):
                lambda_nc = k * r * (delta**2) / (2 * sigma**2)
                F_crit = float(f_dist.ppf(1-alpha, k-1, k*(r-1)))
                pwr = float(1 - ncf.cdf(F_crit, k-1, k*(r-1), lambda_nc))
                if pwr >= power:
                    lines.append(f"Minimum replications required: r = {r}")
                    lines.append(f"Achieved power = {fmt(pwr,4)}")
                    lines.append(f"Total observations = {k*r}")
                    if design == "RCBD":
                        lines.append(f"→ RCBD: {r} complete blocks, {k} treatments each")
                    elif design == "Split-plot":
                        lines.append(f"→ Split-plot: ≥ {r} blocks for whole-plot factor")
                    found = True; break
            if not found:
                lines.append("Could not achieve target power with r ≤ 100.\nConsider increasing δ or reducing σ.")

        self.res_var.set("\n".join(lines))


# ═══════════════════════════════════════════════════════════════
# CLUSTER ANALYSIS
# ═══════════════════════════════════════════════════════════════
class ClusterWindow:
    def __init__(self, parent, gs):
        self.win = tk.Toplevel(parent); self.win.title("Cluster Analysis")
        self.win.geometry("1000x680"); set_icon(self.win); self.gs = gs; self._build()

    def _build(self):
        top = tk.Frame(self.win, padx=8, pady=6); top.pack(fill=tk.X)
        tk.Label(top, text="Method:", font=("Times New Roman",12)).pack(side=tk.LEFT)
        self.meth_var = tk.StringVar(value="ward")
        ttk.Combobox(top, textvariable=self.meth_var,
                     values=["ward","complete","average","single"],
                     state="readonly", width=14).pack(side=tk.LEFT, padx=4)
        tk.Label(top, text="k (clusters):", font=("Times New Roman",12)).pack(side=tk.LEFT, padx=(10,2))
        self.k_var = tk.IntVar(value=3)
        tk.Spinbox(top, from_=2, to=20, textvariable=self.k_var, width=5).pack(side=tk.LEFT)
        tk.Button(top, text="▶ Cluster", bg="#c62828", fg="white",
                  font=("Times New Roman",12), command=self._run).pack(side=tk.LEFT, padx=8)
        tk.Label(top, text="First row = variable names; first column = object names",
                 font=("Times New Roman",10), fg="#666").pack(side=tk.LEFT)

        # data entry
        mid = tk.Frame(self.win); mid.pack(fill=tk.BOTH, expand=True, padx=8)
        self.rows_n = 18; self.cols_n = 8
        canvas = tk.Canvas(mid); canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(mid, orient="vertical", command=canvas.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y); canvas.configure(yscrollcommand=sb.set)
        self.inner = tk.Frame(canvas); canvas.create_window((0,0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: canvas.config(scrollregion=canvas.bbox("all")))
        self.entries = []
        for j in range(self.cols_n):
            tk.Label(self.inner, text=f"{'Name' if j==0 else f'Var{j}'}",
                     relief=tk.RIDGE, width=12, bg="#f0f0f0",
                     font=("Times New Roman",11)).grid(row=0,column=j,padx=1,pady=1,sticky="nsew")
        for i in range(self.rows_n):
            row_ = []
            for j in range(self.cols_n):
                e = tk.Entry(self.inner, width=12, font=("Times New Roman",11))
                e.grid(row=i+1, column=j, padx=1, pady=1); row_.append(e)
            self.entries.append(row_)

    def _run(self):
        from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
        from scipy.spatial.distance import pdist
        raw = [[e.get().strip() for e in row] for row in self.entries]
        raw = [r for r in raw if any(v for v in r)]
        if not raw: messagebox.showwarning("","No data."); return

        obj_names = []; data_matrix = []
        for row in raw:
            nm = row[0] if row else ""; vals = []
            for v in row[1:]:
                if not v: continue
                try: vals.append(float(v.replace(",",".")))
                except Exception: continue
            if nm and vals:
                obj_names.append(nm); data_matrix.append(vals)

        if len(data_matrix) < 2: messagebox.showwarning("","Need ≥ 2 objects."); return
        min_cols = min(len(r) for r in data_matrix)
        X = np.array([r[:min_cols] for r in data_matrix], dtype=float)

        # standardize
        from scipy.stats import zscore
        X_std = zscore(X, axis=0, ddof=1)
        X_std = np.nan_to_num(X_std)

        method = self.meth_var.get()
        Z = linkage(X_std, method=method)
        k = self.k_var.get()
        labels_cl = fcluster(Z, k, criterion='maxclust')

        if not HAS_MPL: messagebox.showwarning("","matplotlib needed."); return
        win = tk.Toplevel(self.win); win.title("Cluster Analysis — Dendrogram"); win.geometry("1000x620")
        fig = Figure(figsize=(10, 5.5), dpi=100)
        ax = fig.add_subplot(111)
        dendrogram(Z, labels=obj_names, ax=ax, leaf_rotation=90, leaf_font_size=9,
                   color_threshold=Z[-(k-1), 2] if k > 1 else np.inf)
        ax.set_title(f"Hierarchical clustering ({method} linkage, k={k})")
        ax.set_ylabel("Distance")
        fig.tight_layout()
        cv = FigureCanvasTkAgg(fig, master=win); cv.draw(); cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # show cluster membership
        result_txt = "\n".join(f"{nm}: Cluster {cl}" for nm, cl in zip(obj_names, labels_cl))
        tk.Label(win, text=result_txt, font=("Times New Roman",11), justify="left").pack(padx=8, pady=4)


# ═══════════════════════════════════════════════════════════════
# PCA MODULE
# ═══════════════════════════════════════════════════════════════
class PCAWindow:
    def __init__(self, parent, gs):
        self.win = tk.Toplevel(parent); self.win.title("Principal Component Analysis (PCA)")
        self.win.geometry("1050x700"); set_icon(self.win); self.gs = gs; self._build()

    def _build(self):
        top = tk.Frame(self.win, padx=8, pady=6); top.pack(fill=tk.X)
        tk.Button(top, text="▶ Run PCA", bg="#c62828", fg="white",
                  font=("Times New Roman",13), command=self._run).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Paste data", command=self._paste).pack(side=tk.LEFT, padx=4)
        tk.Label(top, text="First row = variable names; first column = object labels (optional)",
                 font=("Times New Roman",10), fg="#666").pack(side=tk.LEFT, padx=8)

        mid = tk.Frame(self.win); mid.pack(fill=tk.BOTH, expand=True, padx=8)
        self.rows_n = 20; self.cols_n = 10
        canvas = tk.Canvas(mid); canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(mid, orient="vertical", command=canvas.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y); canvas.configure(yscrollcommand=sb.set)
        self.inner = tk.Frame(canvas); canvas.create_window((0,0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: canvas.config(scrollregion=canvas.bbox("all")))
        self.entries = []
        for j in range(self.cols_n):
            tk.Label(self.inner, text=f"{'Label' if j==0 else f'Var{j}'}",
                     relief=tk.RIDGE, width=11, bg="#f0f0f0",
                     font=("Times New Roman",11)).grid(row=0,column=j,padx=1,pady=1,sticky="nsew")
        for i in range(self.rows_n):
            row_ = []
            for j in range(self.cols_n):
                e = tk.Entry(self.inner, width=11, font=("Times New Roman",11))
                e.grid(row=i+1, column=j, padx=1, pady=1); row_.append(e)
            self.entries.append(row_)

    def _paste(self):
        w = self.win.focus_get()
        if not isinstance(w, tk.Entry): return
        try: data = self.win.clipboard_get()
        except Exception: return
        for i, line in enumerate(data.splitlines()):
            if i >= len(self.entries): break
            for j, val in enumerate(line.split("\t")):
                if j < self.cols_n:
                    self.entries[i][j].delete(0,tk.END); self.entries[i][j].insert(0,val)

    def _run(self):
        raw = [[e.get().strip() for e in row] for row in self.entries]
        raw = [r for r in raw if any(v for v in r)]
        if not raw: messagebox.showwarning("","No data."); return

        obj_names = []; var_names = []; data_rows = []
        # detect if first col is labels (non-numeric)
        has_labels = False
        try: float(raw[0][0].replace(",",".")); has_labels = False
        except ValueError: has_labels = True

        start_col = 1 if has_labels else 0
        for i, row in enumerate(raw):
            obj_names.append(row[0] if has_labels else f"Obs{i+1}")
            vals = []
            for v in row[start_col:]:
                if not v: continue
                try: vals.append(float(v.replace(",",".")))
                except Exception: continue
            if vals: data_rows.append(vals)

        if not data_rows: messagebox.showwarning("","No numeric data."); return
        min_c = min(len(r) for r in data_rows)
        X = np.array([r[:min_c] for r in data_rows], dtype=float)

        # Standardize
        from scipy.stats import zscore
        X_std = zscore(X, axis=0, ddof=1); X_std = np.nan_to_num(X_std)

        # SVD-based PCA
        cov = np.cov(X_std.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]; eigenvectors = eigenvectors[:, idx]
        explained = eigenvalues / np.sum(eigenvalues) * 100
        scores = X_std @ eigenvectors
        n_comp = min(len(eigenvalues), X.shape[1])

        if not HAS_MPL: messagebox.showwarning("","matplotlib needed."); return
        win = tk.Toplevel(self.win); win.title("PCA Results"); win.geometry("1100x700")

        fig = Figure(figsize=(11, 6), dpi=100)
        # Scree plot
        ax1 = fig.add_subplot(131)
        ax1.bar(range(1, n_comp+1), explained[:n_comp], color="#4c72b0", alpha=0.8)
        ax1.plot(range(1, n_comp+1), np.cumsum(explained[:n_comp]), "ro-", markersize=4)
        ax1.set_xlabel("PC"); ax1.set_ylabel("Variance explained (%)")
        ax1.set_title("Scree plot")
        ax1.axhline(80, color="gray", lw=0.8, ls="--")

        # Biplot (PC1 vs PC2)
        ax2 = fig.add_subplot(132)
        ax2.scatter(scores[:, 0], scores[:, 1], s=30, color="#dd8452", zorder=3)
        for i, nm in enumerate(obj_names[:len(scores)]):
            ax2.annotate(nm, (scores[i,0], scores[i,1]), fontsize=7, alpha=0.8)
        # loadings arrows
        scale = max(np.max(np.abs(scores[:,0])), np.max(np.abs(scores[:,1]))) * 0.9
        for j in range(min_c):
            lx, ly = eigenvectors[j,0]*scale*0.7, eigenvectors[j,1]*scale*0.7
            ax2.annotate("", xy=(lx,ly), xytext=(0,0),
                         arrowprops=dict(arrowstyle="->", color="#c62828", lw=1.2))
            ax2.text(lx*1.05, ly*1.05, f"Var{j+1}" if not var_names else var_names[j],
                     fontsize=7, color="#c62828")
        ax2.set_xlabel(f"PC1 ({fmt(explained[0],1)}%)")
        ax2.set_ylabel(f"PC2 ({fmt(explained[1],1)}%)" if n_comp > 1 else "PC2")
        ax2.set_title("Biplot (PC1 × PC2)"); ax2.axhline(0,color="k",lw=0.5); ax2.axvline(0,color="k",lw=0.5)

        # Loadings heatmap
        ax3 = fig.add_subplot(133)
        load_mat = eigenvectors[:, :min(4, n_comp)]
        im = ax3.imshow(load_mat, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        ax3.set_xticks(range(load_mat.shape[1]))
        ax3.set_xticklabels([f"PC{i+1}" for i in range(load_mat.shape[1])], fontsize=8)
        ax3.set_yticks(range(min_c))
        ax3.set_yticklabels([f"Var{j+1}" for j in range(min_c)], fontsize=8)
        ax3.set_title("Loadings")
        for i in range(min_c):
            for j in range(load_mat.shape[1]):
                ax3.text(j, i, fmt(load_mat[i,j],2), ha="center", va="center", fontsize=7)

        fig.tight_layout()
        cv = FigureCanvasTkAgg(fig, master=win); cv.draw(); cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Summary table
        summary_rows = [[f"PC{i+1}", fmt(eigenvalues[i],4), fmt(explained[i],2),
                         fmt(float(np.sum(explained[:i+1])),2)] for i in range(n_comp)]
        frm, _ = make_tv(win, ["Component","Eigenvalue","Variance %","Cumulative %"], summary_rows)
        frm.pack(fill=tk.X, padx=8, pady=4)


# ═══════════════════════════════════════════════════════════════
# REPEATED MEASURES ANOVA
# ═══════════════════════════════════════════════════════════════
class RepeatedMeasuresWindow:
    def __init__(self, parent, gs):
        self.win = tk.Toplevel(parent); self.win.title("Repeated Measures ANOVA")
        self.win.geometry("950x700"); set_icon(self.win); self.gs = gs; self._build()

    def _build(self):
        top = tk.Frame(self.win, padx=8, pady=6); top.pack(fill=tk.X)
        tk.Label(top, text="Columns = time points / conditions  |  Rows = subjects",
                 font=("Times New Roman",11)).pack(side=tk.LEFT)
        tk.Button(top, text="▶ Analyze", bg="#c62828", fg="white",
                  font=("Times New Roman",12), command=self._run).pack(side=tk.RIGHT, padx=4)

        mid = tk.Frame(self.win); mid.pack(fill=tk.BOTH, expand=True, padx=8)
        self.rows_n = 20; self.cols_n = 8
        canvas = tk.Canvas(mid); canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(mid, orient="vertical", command=canvas.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y); canvas.configure(yscrollcommand=sb.set)
        self.inner = tk.Frame(canvas); canvas.create_window((0,0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: canvas.config(scrollregion=canvas.bbox("all")))
        self.col_entries = []  # header names
        self.entries = []
        tk.Label(self.inner, text="Subject", relief=tk.RIDGE, width=12,
                 bg="#f0f0f0", font=("Times New Roman",11)).grid(row=0, column=0, padx=1, pady=1)
        for j in range(1, self.cols_n):
            e = tk.Entry(self.inner, width=12, bg="#e8f0ff", font=("Times New Roman",11))
            e.insert(0, f"T{j}"); e.grid(row=0, column=j, padx=1, pady=1)
            self.col_entries.append(e)
        for i in range(self.rows_n):
            row_ = []
            for j in range(self.cols_n):
                e = tk.Entry(self.inner, width=12, font=("Times New Roman",11))
                e.grid(row=i+1, column=j, padx=1, pady=1); row_.append(e)
            self.entries.append(row_)

    def _run(self):
        # Read data
        time_names = [e.get().strip() or f"T{i+1}" for i, e in enumerate(self.col_entries)]
        raw = [[e.get().strip() for e in row] for row in self.entries]
        subjects = []; data_rows = []
        for row in raw:
            subj = row[0] if row[0] else f"Subj{len(subjects)+1}"
            vals = []
            for v in row[1:len(time_names)+1]:
                if not v: vals.append(np.nan)
                else:
                    try: vals.append(float(v.replace(",",".")))
                    except Exception: vals.append(np.nan)
            if any(not math.isnan(v) for v in vals):
                subjects.append(subj); data_rows.append(vals)

        if len(data_rows) < 2: messagebox.showwarning("","Need ≥ 2 subjects."); return
        k = len(time_names); n = len(data_rows)
        data = np.array(data_rows, dtype=float)

        # Remove subjects with any NaN (listwise deletion)
        mask = ~np.any(np.isnan(data), axis=1)
        data = data[mask]; subjects = [s for s, m in zip(subjects, mask) if m]; n = len(data)
        if n < 2: messagebox.showwarning("","Not enough complete cases."); return

        # Mauchly's test of sphericity (simplified — compare to Greenhouse-Geisser)
        grand_mean = np.mean(data)
        subj_means = np.mean(data, axis=1)
        time_means = np.mean(data, axis=0)

        SS_total = float(np.sum((data - grand_mean)**2))
        SS_subj  = k * float(np.sum((subj_means - grand_mean)**2))
        SS_time  = n * float(np.sum((time_means - grand_mean)**2))
        SS_error = SS_total - SS_subj - SS_time

        df_time = k - 1; df_subj = n - 1; df_err = (k-1)*(n-1)
        MS_time = SS_time / df_time if df_time > 0 else np.nan
        MS_err  = SS_error / df_err if df_err > 0 else np.nan
        F = MS_time / MS_err if (not math.isnan(MS_err) and MS_err > 0) else np.nan
        p = float(1 - f_dist.cdf(F, df_time, df_err)) if not math.isnan(F) else np.nan

        R2 = (SS_time + SS_subj) / SS_total if SS_total > 0 else np.nan
        eta2_time = SS_time / (SS_time + SS_error) if (SS_time + SS_error) > 0 else np.nan

        # Normality of differences
        sw_ps = []
        for j in range(k):
            for jj in range(j+1, k):
                diff = data[:, j] - data[:, jj]
                try: _, p_sw = shapiro(diff)
                except Exception: p_sw = np.nan
                sw_ps.append(p_sw)
        min_sw = min((p for p in sw_ps if not math.isnan(p)), default=np.nan)

        if not HAS_MPL: messagebox.showwarning("","matplotlib needed."); return
        win = tk.Toplevel(self.win); win.title("Repeated Measures — Results"); win.geometry("1000x680")

        res_txt = (f"Repeated Measures ANOVA\n"
                   f"Subjects: {n},  Time points: {k}\n\n"
                   f"SS_time  = {fmt(SS_time,4)},  df = {df_time},  MS = {fmt(MS_time,4)}\n"
                   f"SS_subj  = {fmt(SS_subj,4)},  df = {df_subj}\n"
                   f"SS_error = {fmt(SS_error,4)},  df = {df_err},  MS = {fmt(MS_err,4)}\n\n"
                   f"F({df_time},{df_err}) = {fmt(F,4)},  p = {fmt(p,4)}"
                   f"  {'✓ significant' if not math.isnan(p) and p < ALPHA else '✗ not significant'}\n"
                   f"Partial η² (time) = {fmt(eta2_time,4)}  ({eta2_label(eta2_time)})\n"
                   f"Shapiro–Wilk (differences, min p) = {fmt(min_sw,4)}"
                   f"  {'✓ normal' if not math.isnan(min_sw) and min_sw > 0.05 else '⚠ check normality'}")

        tk.Label(win, text=res_txt, font=("Times New Roman",11), justify="left").pack(padx=8, pady=6, anchor="w")

        # Plot means ± SE
        fig = Figure(figsize=(9, 4), dpi=100)
        ax = fig.add_subplot(111)
        means_ = np.mean(data, axis=0); ses_ = np.std(data, axis=0, ddof=1) / math.sqrt(n)
        ax.errorbar(range(k), means_, yerr=ses_, fmt="o-", capsize=5,
                    color="#4c72b0", ecolor="#c62828", linewidth=2, markersize=7)
        ax.set_xticks(range(k)); ax.set_xticklabels(time_names)
        ax.set_xlabel("Time point / Condition"); ax.set_ylabel("Mean ± SE")
        ax.set_title("Repeated Measures: Means over Time"); ax.yaxis.grid(True, alpha=0.3)
        fig.tight_layout()
        cv = FigureCanvasTkAgg(fig, master=win); cv.draw(); cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Post-hoc (Bonferroni-corrected paired t-tests)
        if not math.isnan(p) and p < ALPHA:
            ph_rows = []
            mt = k*(k-1)/2
            for j in range(k):
                for jj in range(j+1, k):
                    from scipy.stats import ttest_rel
                    st, p_t = ttest_rel(data[:,j], data[:,jj])
                    p_adj = min(1., float(p_t)*mt)
                    ph_rows.append([f"{time_names[j]} vs {time_names[jj]}",
                                    fmt(float(st),4), fmt(p_adj,4),
                                    "sig. " + sig_mark(p_adj) if p_adj < ALPHA else "–"])
            frm, _ = make_tv(win, ["Pair","t","p (Bonf.)","Result"], ph_rows)
            frm.pack(fill=tk.X, padx=8, pady=4)


# ═══════════════════════════════════════════════════════════════
# STABILITY ANALYSIS  (Eberhart–Russell + GGE biplot)
# ═══════════════════════════════════════════════════════════════
class StabilityWindow:
    def __init__(self, parent, gs):
        self.win = tk.Toplevel(parent); self.win.title("Stability Analysis (GxE)")
        self.win.geometry("1100x720"); set_icon(self.win); self.gs = gs; self._build()

    def _build(self):
        top = tk.Frame(self.win, padx=8, pady=6); top.pack(fill=tk.X)
        tk.Label(top, text="Rows = Genotypes/Varieties  |  Columns = Environments",
                 font=("Times New Roman",11)).pack(side=tk.LEFT)
        tk.Button(top, text="▶ Analyze", bg="#c62828", fg="white",
                  font=("Times New Roman",12), command=self._run).pack(side=tk.RIGHT, padx=4)

        mid = tk.Frame(self.win); mid.pack(fill=tk.BOTH, expand=True, padx=8)
        self.rows_n = 16; self.cols_n = 10
        canvas = tk.Canvas(mid); canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(mid, orient="vertical", command=canvas.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y); canvas.configure(yscrollcommand=sb.set)
        self.inner = tk.Frame(canvas); canvas.create_window((0,0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: canvas.config(scrollregion=canvas.bbox("all")))

        self.env_entries = []
        tk.Label(self.inner, text="Genotype", relief=tk.RIDGE, width=14,
                 bg="#f0f0f0", font=("Times New Roman",11)).grid(row=0, column=0, padx=1, pady=1)
        for j in range(1, self.cols_n):
            e = tk.Entry(self.inner, width=12, bg="#e8f0ff", font=("Times New Roman",11))
            e.insert(0, f"E{j}"); e.grid(row=0, column=j, padx=1, pady=1)
            self.env_entries.append(e)
        self.entries = []
        for i in range(self.rows_n):
            row_ = []
            for j in range(self.cols_n):
                e = tk.Entry(self.inner, width=12 if j==0 else 10, font=("Times New Roman",11))
                e.grid(row=i+1, column=j, padx=1, pady=1); row_.append(e)
            self.entries.append(row_)

    def _run(self):
        env_names = [e.get().strip() or f"E{i+1}" for i, e in enumerate(self.env_entries)]
        raw = [[e.get().strip() for e in row] for row in self.entries]
        gen_names = []; matrix = []
        for row in raw:
            nm = row[0] if row[0] else f"G{len(gen_names)+1}"
            vals = []
            for v in row[1:len(env_names)+1]:
                if not v: vals.append(np.nan)
                else:
                    try: vals.append(float(v.replace(",",".")))
                    except Exception: vals.append(np.nan)
            if any(not math.isnan(v) for v in vals):
                gen_names.append(nm); matrix.append(vals)

        if len(matrix) < 2: messagebox.showwarning("","Need ≥ 2 genotypes."); return
        e_count = len(env_names); g_count = len(gen_names)
        data = np.array(matrix, dtype=float)

        # ── Eberhart–Russell regression stability ──────────────
        env_means = np.nanmean(data, axis=0)
        grand_mean = np.nanmean(data)
        env_index = env_means - grand_mean  # environment index

        er_rows = []
        for i in range(g_count):
            y = data[i]; mask = ~np.isnan(y) & ~np.isnan(env_index)
            if np.sum(mask) < 2:
                er_rows.append([gen_names[i], fmt(np.nanmean(y),3), "–", "–", "–"]); continue
            xi = env_index[mask]; yi = y[mask]
            # linear regression of genotype yield on env index
            X_ = np.column_stack([np.ones(len(xi)), xi])
            beta, *_ = np.linalg.lstsq(X_, yi, rcond=None)
            b_i = float(beta[1])  # regression coefficient (stability)
            yhat = X_ @ beta
            ss_dev = float(np.sum((yi - yhat)**2))
            s2d = ss_dev / max(len(yi)-2, 1)  # variance of deviations
            gen_mean = float(np.nanmean(y))
            er_rows.append([gen_names[i], fmt(gen_mean,3), fmt(b_i,4), fmt(s2d,4),
                            "Stable (bi≈1, s²d≈0)" if abs(b_i-1)<0.2 and s2d<0.1 else
                            "Responsive" if b_i > 1.2 else "Conservative" if b_i < 0.8 else "Average"])

        # ── GGE Biplot via SVD ──────────────────────────────────
        # center by environment means
        data_c = data - env_means[np.newaxis, :]
        # replace NaN with 0 for SVD
        data_c = np.nan_to_num(data_c)
        U, S, Vt = np.linalg.svd(data_c, full_matrices=False)
        pc1_g = U[:,0] * S[0]; pc2_g = U[:,1] * S[1] if len(S)>1 else np.zeros(g_count)
        pc1_e = Vt[0,:];       pc2_e = Vt[1,:] if len(S)>1 else np.zeros(e_count)
        var_exp = S**2 / np.sum(S**2) * 100

        if not HAS_MPL: messagebox.showwarning("","matplotlib needed."); return
        win = tk.Toplevel(self.win); win.title("Stability Analysis — Results"); win.geometry("1150x720")

        fig = Figure(figsize=(11, 5.5), dpi=100)
        # GGE biplot
        ax1 = fig.add_subplot(121)
        ax1.axhline(0, color="k", lw=0.5); ax1.axvline(0, color="k", lw=0.5)
        ax1.scatter(pc1_g, pc2_g, s=80, color="#4c72b0", zorder=3)
        for i, nm in enumerate(gen_names):
            ax1.annotate(nm, (pc1_g[i], pc2_g[i]), fontsize=8, ha='center', va='bottom')
        # environment vectors
        sc = max(np.max(np.abs(pc1_g)), np.max(np.abs(pc2_g)))
        sc_e = sc / max(np.max(np.abs(pc1_e)), max(np.max(np.abs(pc2_e)),1e-10))
        for j, nm in enumerate(env_names):
            ax1.annotate("", xy=(pc1_e[j]*sc_e*0.8, pc2_e[j]*sc_e*0.8), xytext=(0,0),
                         arrowprops=dict(arrowstyle="->", color="#c62828", lw=1.2))
            ax1.text(pc1_e[j]*sc_e*0.85, pc2_e[j]*sc_e*0.85, nm, fontsize=8, color="#c62828")
        ax1.set_xlabel(f"PC1 ({fmt(var_exp[0],1)}%)"); ax1.set_ylabel(f"PC2 ({fmt(var_exp[1] if len(var_exp)>1 else 0,1)}%)")
        ax1.set_title("GGE Biplot"); ax1.yaxis.grid(True, alpha=0.25)

        # Stability table
        ax2 = fig.add_subplot(122)
        ax2.axis("off")
        col_labels = ["Genotype","Mean","bi","s²d","Stability"]
        tbl = ax2.table(cellText=er_rows, colLabels=col_labels, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.4)
        ax2.set_title("Eberhart–Russell Stability", pad=14)

        fig.tight_layout()
        cv = FigureCanvasTkAgg(fig, master=win); cv.draw(); cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        frm, _ = make_tv(win, ["Genotype","Mean","bi","s²d","Stability class"], er_rows)
        frm.pack(fill=tk.X, padx=8, pady=4)



# ═══════════════════════════════════════════════════════════════
# ANCOVA — Analysis of Covariance
# ═══════════════════════════════════════════════════════════════
class AncovaWindow:
    """
    ANCOVA: ANOVA with one or more continuous covariates.
    Methodological pipeline:
      1. Check covariate is continuous and numeric
      2. Test homogeneity of regression slopes (factor × covariate interaction)
         → if significant: slopes differ, ANCOVA assumption violated → block or warn
      3. Test normality of residuals (Shapiro–Wilk)
      4. Test homogeneity of variances (Levene)
      5. Run ANCOVA (GLM with covariate)
      6. Report: ANOVA table, adjusted means (LS means), SS Type III
    """
    def __init__(self, parent, gs):
        self.win = tk.Toplevel(parent)
        self.win.title("ANCOVA — Analysis of Covariance")
        self.win.geometry("1100x740"); set_icon(self.win)
        self.gs = gs; self._build()

    def _build(self):
        # ── Menu ──
        mb = tk.Menu(self.win); self.win.config(menu=mb)
        hm = tk.Menu(mb, tearoff=0)
        hm.add_command(label="What is ANCOVA?", command=self._help)
        mb.add_cascade(label="Help", menu=hm)

        # ── Toolbar ──
        top = tk.Frame(self.win, padx=8, pady=6); top.pack(fill=tk.X)
        tk.Button(top, text="▶ Run ANCOVA", bg="#c62828", fg="white",
                  font=("Times New Roman", 13), command=self._run).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Paste from clipboard",
                  command=self._paste).pack(side=tk.LEFT, padx=4)
        tk.Label(top, text="α:", font=("Times New Roman", 12)).pack(side=tk.LEFT, padx=(12, 2))
        self.alpha_var = tk.StringVar(value="0.05")
        ttk.Combobox(top, textvariable=self.alpha_var, values=["0.01","0.05","0.10"],
                     state="readonly", width=7).pack(side=tk.LEFT)

        # ── Instructions ──
        info = tk.Frame(self.win, bg="#f0f4ff", padx=8, pady=4)
        info.pack(fill=tk.X, padx=8, pady=(0, 4))
        tk.Label(info, text=(
            "Column layout:  [Group/Factor]  [Covariate 1]  [Covariate 2 ...]  [Dependent variable]\n"
            "First row = column headers.  Group column must contain text labels.  "
            "Covariate and DV columns must be numeric."),
            font=("Times New Roman", 10), bg="#f0f4ff", justify="left").pack(anchor="w")

        # ── Data table ──
        mid = tk.Frame(self.win); mid.pack(fill=tk.BOTH, expand=True, padx=8)
        self.n_rows = 24; self.n_cols = 6
        canvas = tk.Canvas(mid); canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(mid, orient="vertical", command=canvas.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.configure(yscrollcommand=sb.set)
        self.inner = tk.Frame(canvas)
        canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: canvas.config(scrollregion=canvas.bbox("all")))

        col_hints = ["Group", "Covariate 1", "Covariate 2", "Covariate 3", "Covariate 4", "Dependent Y"]
        self.header_entries = []
        for j in range(self.n_cols):
            e = tk.Entry(self.inner, width=14, bg="#dce8ff", font=("Times New Roman", 11))
            e.insert(0, col_hints[j] if j < len(col_hints) else f"Col{j+1}")
            e.grid(row=0, column=j, padx=1, pady=1)
            self.header_entries.append(e)
        self.entries = []
        for i in range(self.n_rows):
            row_ = []
            for j in range(self.n_cols):
                e = tk.Entry(self.inner, width=14, font=("Times New Roman", 11))
                e.grid(row=i+1, column=j, padx=1, pady=1)
                row_.append(e)
            self.entries.append(row_)

    def _help(self):
        messagebox.showinfo("What is ANCOVA?",
            "Analysis of Covariance (ANCOVA) combines ANOVA and linear regression.\n\n"
            "It tests group differences on a dependent variable while statistically\n"
            "controlling for the effect of one or more continuous covariates.\n\n"
            "KEY ASSUMPTION — Homogeneity of regression slopes:\n"
            "The relationship between the covariate and DV must be the same\n"
            "across all groups (parallel regression lines).\n"
            "If slopes differ significantly → ANCOVA is not appropriate.\n\n"
            "Typical uses in agronomy:\n"
            "• Comparing yields while controlling for initial plant height/weight\n"
            "• Adjusting for soil pH differences between plots\n"
            "• Controlling for pre-treatment measurements")

    def _paste(self):
        try: data = self.win.clipboard_get()
        except Exception: return
        rows = [r for r in data.splitlines() if r.strip()]
        for i, line in enumerate(rows[:self.n_rows]):
            for j, val in enumerate(line.split("\t")[:self.n_cols]):
                self.entries[i][j].delete(0, tk.END)
                self.entries[i][j].insert(0, val.strip())

    def _run(self):
        alpha = float(self.alpha_var.get())

        # ── Read headers ──
        headers = [e.get().strip() or f"Col{j+1}" for j, e in enumerate(self.header_entries)]
        group_col = headers[0]
        dv_col    = headers[-1]
        cov_cols  = headers[1:-1]  # everything between group and DV

        # ── Read data ──
        raw = [[e.get().strip() for e in row] for row in self.entries]
        raw = [r for r in raw if any(v for v in r)]
        if not raw:
            messagebox.showwarning("No data", "Please enter data first."); return

        groups = []; cov_data = [[] for _ in cov_cols]; y_data = []
        skipped = 0
        for row in raw:
            if len(row) < self.n_cols: row += [""] * (self.n_cols - len(row))
            grp = row[0]
            if not grp: skipped += 1; continue
            try:
                covs = [float(row[j+1].replace(",",".")) for j in range(len(cov_cols))]
                yval = float(row[-1].replace(",","."))
            except (ValueError, IndexError):
                skipped += 1; continue
            groups.append(grp)
            for j, cv in enumerate(covs): cov_data[j].append(cv)
            y_data.append(yval)

        if skipped > 0:
            messagebox.showinfo("Note", f"{skipped} row(s) skipped (missing or non-numeric values).")

        n = len(y_data)
        # ── Guard 1: minimum observations ──
        if n < 6:
            messagebox.showwarning("Too few observations",
                f"ANCOVA requires at least 6 complete observations.\nFound: {n}."); return

        # ── Guard 2: at least 2 groups ──
        group_levels = first_seen(groups)
        k = len(group_levels)
        if k < 2:
            messagebox.showwarning("Only one group",
                "ANCOVA requires at least 2 groups.\n"
                "Check that the Group column contains different labels."); return

        # ── Guard 3: minimum n per group ──
        from collections import Counter
        grp_counts = Counter(groups)
        min_grp = min(grp_counts.values())
        if min_grp < 2:
            messagebox.showwarning("Group too small",
                f"Each group needs ≥ 2 observations.\n"
                f"Smallest group has {min_grp} observation(s)."); return

        # ── Guard 4: check covariates are numeric (already done in parsing) ──
        # ── Guard 5: check for perfect multicollinearity among covariates ──
        if len(cov_cols) > 1:
            cov_matrix = np.column_stack([np.array(cd) for cd in cov_data])
            corr_matrix = np.corrcoef(cov_matrix.T)
            for i in range(len(cov_cols)):
                for j in range(i+1, len(cov_cols)):
                    if abs(corr_matrix[i,j]) > 0.95:
                        ans = messagebox.askyesno("High multicollinearity",
                            f"Covariates '{cov_cols[i]}' and '{cov_cols[j]}' are highly correlated\n"
                            f"(r = {corr_matrix[i,j]:.3f}).\n\n"
                            "This may cause unstable estimates (multicollinearity problem).\n"
                            "Consider removing one of the covariates.\n\n"
                            "Continue anyway?")
                        if not ans: return

        y = np.array(y_data, dtype=float)
        covs_arr = [np.array(cd, dtype=float) for cd in cov_data]

        # ── Guard 6: Homogeneity of regression slopes ──
        # Test: fit model with group × covariate interaction
        # If interaction is significant → slopes differ → ANCOVA assumption violated
        slopes_ok = True
        slope_details = []
        for ci, (cov_name, cov_arr) in enumerate(zip(cov_cols, covs_arr)):
            # Build X with intercept + group dummies + covariate + group×covariate
            X_parts = [np.ones(n)]
            g_dummies = []
            for lv in group_levels[1:]:
                d = np.array([1. if g == lv else 0. for g in groups])
                X_parts.append(d); g_dummies.append(d)
            X_parts.append(cov_arr)
            # interaction terms: group_dummy × covariate
            for gd in g_dummies:
                X_parts.append(gd * cov_arr)
            X_int = np.column_stack(X_parts)
            X_no_int = np.column_stack(X_parts[:len(X_parts)-len(g_dummies)])

            _, _, _, sse_full, dfe_full, mse_full = _ols(y, X_int)
            _, _, _, sse_red,  dfe_red,  _        = _ols(y, X_no_int)

            df_int = len(g_dummies)
            ss_int = sse_red - sse_full
            ms_int = ss_int / df_int if df_int > 0 else np.nan
            F_int  = ms_int / mse_full if (not math.isnan(mse_full) and mse_full > 0) else np.nan
            p_int  = float(1 - f_dist.cdf(F_int, df_int, dfe_full)) if not math.isnan(F_int) else np.nan
            slope_details.append((cov_name, F_int, p_int))
            if not math.isnan(p_int) and p_int < alpha:
                slopes_ok = False

        if not slopes_ok:
            failed = [f"'{n}' (F={fmt(F,3)}, p={fmt(p,4)})"
                      for n, F, p in slope_details
                      if not math.isnan(p) and p < alpha]
            ans = messagebox.askyesno(
                "ANCOVA ASSUMPTION VIOLATED — Heterogeneous regression slopes",
                "The homogeneity of regression slopes assumption is VIOLATED for:\n"
                + "\n".join(f"  • {f}" for f in failed) + "\n\n"
                "This means the covariate affects the DV differently in different groups.\n"
                "Standard ANCOVA is NOT appropriate in this situation.\n\n"
                "Options:\n"
                "• Use Johnson-Neyman technique (regions of significance)\n"
                "• Run separate regressions per group\n"
                "• Use interaction ANOVA instead\n\n"
                "Do you want to continue with ANCOVA anyway? (NOT recommended)")
            if not ans: return

        # ── Build ANCOVA model (Type III SS) ──
        # X: intercept + group dummies + covariate(s)
        X_parts = [np.ones(n)]
        ts = {"Intercept": [0]}
        idx_cur = 1
        # group factor
        g_idx = []
        for lv in group_levels[1:]:
            d = np.array([1. if g == lv else 0. for g in groups])
            X_parts.append(d); g_idx.append(idx_cur); idx_cur += 1
        ts["Group"] = g_idx
        # covariates
        for cov_name, cov_arr in zip(cov_cols, covs_arr):
            cov_norm = (cov_arr - np.mean(cov_arr)) / (np.std(cov_arr, ddof=1) + 1e-12)
            X_parts.append(cov_norm)
            ts[f"Covariate: {cov_name}"] = [idx_cur]; idx_cur += 1

        X = np.column_stack(X_parts)
        beta, yhat, residuals, sse, dfe, mse = _ols(y, X)
        sst = float(np.sum((y - np.mean(y))**2))

        # Type III SS for each term
        anova_rows = []
        for term, idx_list in ts.items():
            if term == "Intercept": continue
            keep = [i for i in range(X.shape[1]) if i not in idx_list]
            _, _, _, sse_red, _, _ = _ols(y, X[:, keep])
            ss = float(sse_red - sse)
            df = len(idx_list)
            ms = ss / df if df > 0 else np.nan
            F  = ms / mse if (not math.isnan(mse) and mse > 0) else np.nan
            p  = float(1 - f_dist.cdf(F, df, dfe)) if not math.isnan(F) else np.nan
            mark = sig_mark(p) if not math.isnan(p) else ""
            concl = f"significant {mark}" if mark else ("ns" if not math.isnan(p) else "–")
            anova_rows.append([term, fmt(ss,4), str(df), fmt(ms,4), fmt(F,4), fmt(p,4), concl])

        anova_rows.append(["Residual", fmt(sse,4), str(dfe), fmt(mse,4), "", "", ""])
        anova_rows.append(["Total",    fmt(sst,4), str(n-1), "", "", "", ""])

        # ── Guard 7: Normality of residuals ──
        try: W_res, p_res = shapiro(residuals) if len(residuals) >= 3 else (np.nan, np.nan)
        except Exception: W_res, p_res = np.nan, np.nan
        if not math.isnan(p_res) and p_res <= alpha:
            ans = messagebox.askyesno("Non-normal residuals",
                f"Residuals do not follow a normal distribution\n"
                f"(Shapiro–Wilk: W={fmt(W_res,4)}, p={fmt(p_res,4)}).\n\n"
                "ANCOVA assumes normally distributed residuals.\n"
                "Consider: data transformation (log/√) before running ANCOVA.\n\n"
                "Continue anyway?")
            if not ans: return

        # ── Guard 8: Homogeneity of variances (Levene) ──
        grp_residuals = defaultdict(list)
        for g, r in zip(groups, residuals): grp_residuals[g].append(r)
        lev_F, lev_p = levene_test(dict(grp_residuals))
        if not math.isnan(lev_p) and lev_p < alpha:
            ans = messagebox.askyesno("Heterogeneous variances (Levene test)",
                f"Levene test: F={fmt(lev_F,4)}, p={fmt(lev_p,4)}\n\n"
                "Variances differ significantly across groups.\n"
                "ANCOVA is somewhat robust to this violation when group sizes are equal,\n"
                "but results may be unreliable with unequal group sizes.\n\n"
                "Continue anyway?")
            if not ans: return

        # ── Adjusted (LS) means ──
        # Compute adjusted means: predict at grand mean of each covariate
        cov_grand_means = [np.mean(ca) for ca in covs_arr]
        adj_means = {}
        for lv in group_levels:
            x_pred = [1.0]  # intercept
            for ref_lv in group_levels[1:]:
                x_pred.append(1.0 if lv == ref_lv else 0.0)
            for ca_mean, ca_arr in zip(cov_grand_means, covs_arr):
                cov_norm_mean = (ca_mean - np.mean(ca_arr)) / (np.std(ca_arr, ddof=1) + 1e-12)
                x_pred.append(cov_norm_mean)
            adj_means[lv] = float(np.dot(beta, x_pred))

        # unadjusted means
        raw_means = {lv: float(np.mean([y_data[i] for i, g in enumerate(groups) if g == lv]))
                     for lv in group_levels}

        R2 = 1 - sse/sst if sst > 0 else np.nan

        self._show_results(anova_rows, adj_means, raw_means, group_levels,
                           residuals, W_res, p_res, lev_F, lev_p,
                           slope_details, R2, mse, dfe, alpha, y, yhat, groups)

    def _show_results(self, anova_rows, adj_means, raw_means, group_levels,
                      residuals, W_res, p_res, lev_F, lev_p,
                      slope_details, R2, mse, dfe, alpha, y, yhat, groups):
        win = tk.Toplevel(self.win); win.title("ANCOVA — Results")
        win.geometry("1150x760"); set_icon(win)

        # scrollable body
        main = tk.Frame(win); main.pack(fill=tk.BOTH, expand=True)
        vsb = ttk.Scrollbar(main, orient="vertical"); vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas = tk.Canvas(main, yscrollcommand=vsb.set); canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.config(command=canvas.yview)
        body = tk.Frame(canvas); canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        win.bind("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        def _head(txt):
            tk.Label(body, text=txt, font=("Times New Roman",12,"bold"),
                     anchor="w").pack(fill=tk.X, padx=10, pady=(8,2))
        def _txt(txt, color="#000000"):
            tk.Label(body, text=txt, font=("Times New Roman",11), fg=color,
                     anchor="w", justify="left").pack(fill=tk.X, padx=10, pady=1)
        def _tbl(headers, rows):
            f, _ = make_tv(body, headers, rows); f.pack(fill=tk.X, padx=10, pady=(2,8))

        _head("ANCOVA — Analysis of Covariance")
        _txt(f"R² = {fmt(R2,4)}   |   MSE = {fmt(mse,4)}   |   df_error = {dfe}")

        # Assumption checks
        _head("Assumption Checks")
        norm_color = "#000000" if math.isnan(p_res) or p_res > alpha else "#c62828"
        _txt(f"Normality of residuals (Shapiro–Wilk):  W={fmt(W_res,4)},  p={fmt(p_res,4)}  "
             f"{'✓ OK' if not math.isnan(p_res) and p_res > alpha else '⚠ VIOLATED'}",
             norm_color)
        lev_color = "#000000" if math.isnan(lev_p) or lev_p >= alpha else "#c62828"
        _txt(f"Homogeneity of variances (Levene):  F={fmt(lev_F,4)},  p={fmt(lev_p,4)}  "
             f"{'✓ OK' if not math.isnan(lev_p) and lev_p >= alpha else '⚠ VIOLATED'}",
             lev_color)
        for cov_name, F_sl, p_sl in slope_details:
            sl_ok = math.isnan(p_sl) or p_sl >= alpha
            sl_color = "#000000" if sl_ok else "#c62828"
            _txt(f"Homogeneity of slopes ({cov_name}):  F={fmt(F_sl,4)},  p={fmt(p_sl,4)}  "
                 f"{'✓ OK' if sl_ok else '⚠ VIOLATED — slopes differ'}",
                 sl_color)

        # ANOVA table
        _head("ANCOVA Table (Type III SS)")
        _tbl(["Source","SS","df","MS","F","p","Result"], anova_rows)

        # Adjusted means
        _head("Group Means")
        means_rows = [[lv, fmt(raw_means[lv],4), fmt(adj_means[lv],4)]
                      for lv in group_levels]
        _tbl(["Group","Unadjusted Mean","Adjusted Mean (LS Mean)"], means_rows)

        # Pairwise comparisons of adjusted means (Bonferroni t-test on LS means)
        # SE for difference between two LS means ≈ sqrt(MSE * 2/n_harm)
        _head("Pairwise Comparisons of Adjusted Means (Bonferroni)")
        ph_rows = []
        m_tests = len(group_levels) * (len(group_levels)-1) / 2
        n_per_grp = {lv: groups.count(lv) for lv in group_levels}
        for lv1, lv2 in combinations(group_levels, 2):
            n1, n2 = n_per_grp[lv1], n_per_grp[lv2]
            se = math.sqrt(mse * (1/n1 + 1/n2)) if mse > 0 else np.nan
            diff = adj_means[lv1] - adj_means[lv2]
            if math.isnan(se) or se == 0:
                ph_rows.append([f"{lv1} vs {lv2}", fmt(diff,4), "–", "–", "–"]); continue
            t_val = abs(diff) / se
            p_raw = 2 * (1 - float(t_dist.cdf(t_val, dfe)))
            p_adj = min(1., p_raw * m_tests)
            mark  = sig_mark(p_adj)
            ph_rows.append([f"{lv1} vs {lv2}", fmt(diff,4), fmt(t_val,4), fmt(p_adj,4),
                             f"significant {mark}" if mark else "ns"])
        _tbl(["Comparison","Difference","t","p (Bonf.)","Result"], ph_rows)

        # Plots
        if HAS_MPL:
            fig = Figure(figsize=(11, 4), dpi=100)
            ax1 = fig.add_subplot(121)
            ax1.scatter(yhat, residuals, s=22, color="#4c72b0", alpha=0.8)
            ax1.axhline(0, color="k", lw=0.8)
            ax1.set_xlabel("Fitted values"); ax1.set_ylabel("Residuals")
            ax1.set_title("Residuals vs Fitted"); ax1.yaxis.grid(True, alpha=0.3)

            from scipy.stats import probplot
            ax2 = fig.add_subplot(122)
            res_sort = np.sort(residuals)
            rp = probplot(residuals, dist="norm")
            ax2.plot(rp[0][0], rp[0][1], 'o', markersize=4, color="#4c72b0")
            ax2.plot(rp[0][0], rp[1][1] + rp[1][0]*rp[0][0], 'r-', lw=1)
            ax2.set_xlabel("Theoretical Quantiles"); ax2.set_ylabel("Sample Quantiles")
            ax2.set_title("Normal Q-Q Plot of Residuals"); ax2.yaxis.grid(True, alpha=0.3)
            fig.tight_layout()
            cv = FigureCanvasTkAgg(fig, master=body); cv.draw()
            cv.get_tk_widget().pack(fill=tk.X, padx=10, pady=6)


# ═══════════════════════════════════════════════════════════════
# MANOVA — Multivariate Analysis of Variance
# ═══════════════════════════════════════════════════════════════
class ManovaWindow:
    """
    MANOVA: simultaneously tests group differences across multiple DVs.
    Reports: Wilks' Lambda, Pillai's Trace, Hotelling-Lawley, Roy's GCR.
    Prerequisite checks:
      1. n > p (obs > variables) per group
      2. Multivariate normality (Mardia's skewness + kurtosis)
      3. Homogeneity of covariance matrices (Box's M approximation)
      4. Absence of multicollinearity (r > 0.95 among DVs)
    Post-hoc: univariate ANOVAs with Bonferroni correction.
    """
    def __init__(self, parent, gs):
        self.win = tk.Toplevel(parent)
        self.win.title("MANOVA — Multivariate Analysis of Variance")
        self.win.geometry("1150x760"); set_icon(self.win)
        self.gs = gs; self._build()

    def _build(self):
        mb = tk.Menu(self.win); self.win.config(menu=mb)
        hm = tk.Menu(mb, tearoff=0)
        hm.add_command(label="What is MANOVA?", command=self._help)
        mb.add_cascade(label="Help", menu=hm)

        top = tk.Frame(self.win, padx=8, pady=6); top.pack(fill=tk.X)
        tk.Button(top, text="▶ Run MANOVA", bg="#c62828", fg="white",
                  font=("Times New Roman", 13), command=self._run).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Paste from clipboard", command=self._paste).pack(side=tk.LEFT, padx=4)
        tk.Label(top, text="α:", font=("Times New Roman", 12)).pack(side=tk.LEFT, padx=(12,2))
        self.alpha_var = tk.StringVar(value="0.05")
        ttk.Combobox(top, textvariable=self.alpha_var, values=["0.01","0.05","0.10"],
                     state="readonly", width=7).pack(side=tk.LEFT)

        info = tk.Frame(self.win, bg="#f0f4ff", padx=8, pady=4)
        info.pack(fill=tk.X, padx=8, pady=(0,4))
        tk.Label(info, text=(
            "Column layout:  [Group/Factor]  [DV 1]  [DV 2]  [DV 3]  ...\n"
            "First row = column headers.  Group column = text labels.  DV columns = numeric.  "
            "Minimum: 1 group column + 2 DV columns."),
            font=("Times New Roman", 10), bg="#f0f4ff", justify="left").pack(anchor="w")

        mid = tk.Frame(self.win); mid.pack(fill=tk.BOTH, expand=True, padx=8)
        self.n_rows = 24; self.n_cols = 8
        canvas = tk.Canvas(mid); canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(mid, orient="vertical", command=canvas.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y); canvas.configure(yscrollcommand=sb.set)
        self.inner = tk.Frame(canvas); canvas.create_window((0,0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: canvas.config(scrollregion=canvas.bbox("all")))

        col_hints = ["Group","DV 1","DV 2","DV 3","DV 4","DV 5","DV 6","DV 7"]
        self.header_entries = []
        for j in range(self.n_cols):
            e = tk.Entry(self.inner, width=13, bg="#dce8ff", font=("Times New Roman",11))
            e.insert(0, col_hints[j] if j < len(col_hints) else f"DV{j}")
            e.grid(row=0, column=j, padx=1, pady=1)
            self.header_entries.append(e)
        self.entries = []
        for i in range(self.n_rows):
            row_ = []
            for j in range(self.n_cols):
                e = tk.Entry(self.inner, width=13, font=("Times New Roman",11))
                e.grid(row=i+1, column=j, padx=1, pady=1); row_.append(e)
            self.entries.append(row_)

    def _help(self):
        messagebox.showinfo("What is MANOVA?",
            "Multivariate Analysis of Variance (MANOVA) simultaneously tests\n"
            "whether group means differ across multiple dependent variables.\n\n"
            "Advantages over multiple ANOVAs:\n"
            "• Controls Type I error inflation (no need for Bonferroni on DVs)\n"
            "• Detects effects that individual ANOVAs may miss\n"
            "• Accounts for correlations among DVs\n\n"
            "KEY ASSUMPTIONS:\n"
            "1. Multivariate normality of DV vector within groups\n"
            "2. Homogeneity of covariance matrices across groups (Box's M)\n"
            "3. No perfect multicollinearity among DVs\n"
            "4. n > p: observations must exceed number of DVs per group\n\n"
            "Test statistics (all equivalent, differ in power):\n"
            "• Wilks' Lambda: most commonly reported\n"
            "• Pillai's Trace: most robust to assumption violations\n"
            "• Hotelling-Lawley Trace: powerful when one root dominates\n"
            "• Roy's GCR: most powerful but least robust\n\n"
            "Post-hoc: if MANOVA significant → run separate ANOVAs per DV\n"
            "with Bonferroni correction (α / number of DVs).")

    def _paste(self):
        try: data = self.win.clipboard_get()
        except Exception: return
        for i, line in enumerate(data.splitlines()[:self.n_rows]):
            for j, val in enumerate(line.split("\t")[:self.n_cols]):
                self.entries[i][j].delete(0,tk.END); self.entries[i][j].insert(0,val.strip())

    def _mardia_test(self, X):
        """Mardia's multivariate normality: skewness and kurtosis tests."""
        n, p = X.shape
        if n < p + 1: return np.nan, np.nan, np.nan, np.nan
        X_c = X - np.mean(X, axis=0)
        S = np.cov(X.T, ddof=1)
        try:
            S_inv = np.linalg.pinv(S)
        except Exception: return np.nan, np.nan, np.nan, np.nan
        # Mahalanobis distances
        D = X_c @ S_inv @ X_c.T
        # Mardia skewness
        b1p = float(np.sum(D**3)) / n**2
        k_sk = n * b1p / 6.0
        df_sk = p*(p+1)*(p+2)/6
        p_sk  = float(1 - f_dist.cdf(k_sk/df_sk, df_sk, 1e6)) if df_sk > 0 else np.nan
        # Mardia kurtosis
        b2p = float(np.sum(np.diag(D)**2)) / n
        k_ku = (b2p - p*(p+2)) / math.sqrt(8*p*(p+2)/n)
        from scipy.stats import norm as _norm
        p_ku = float(2 * (1 - _norm.cdf(abs(k_ku))))
        return float(b1p), p_sk, float(b2p), p_ku

    def _box_m_test(self, groups_data, group_levels):
        """Box's M test for homogeneity of covariance matrices (approximate)."""
        k = len(group_levels)
        ns = [len(groups_data[lv]) for lv in group_levels]
        p  = groups_data[group_levels[0]].shape[1]
        covs = []
        for lv in group_levels:
            Xl = groups_data[lv]
            if len(Xl) < 2: return np.nan, np.nan
            covs.append(np.cov(Xl.T, ddof=1))
        # Pooled covariance
        n_tot = sum(ns)
        S_pool = sum((n-1) * C for n, C in zip(ns, covs)) / (n_tot - k)
        try:
            ln_det_pool = np.log(max(np.linalg.det(S_pool), 1e-300))
            ln_dets     = [np.log(max(np.linalg.det(C), 1e-300)) for C in covs]
        except Exception: return np.nan, np.nan
        M = (n_tot - k) * ln_det_pool - sum((n-1)*d for n,d in zip(ns, ln_dets))
        # c1 correction
        c1 = (sum(1/(n-1) for n in ns) - 1/(n_tot-k)) * (2*p**2 + 3*p - 1) / (6*(p+1)*(k-1))
        chi2 = M * (1 - c1)
        df_m  = p*(p+1)*(k-1)/2
        from scipy.stats import chi2 as chi2_dist
        p_m   = float(1 - chi2_dist.cdf(chi2, df_m)) if df_m > 0 else np.nan
        return float(chi2), p_m

    def _run(self):
        alpha = float(self.alpha_var.get())
        headers = [e.get().strip() or f"Col{j+1}" for j, e in enumerate(self.header_entries)]
        dv_names = headers[1:]   # all columns after group

        # ── Parse data ──
        raw = [[e.get().strip() for e in row] for row in self.entries]
        raw = [r for r in raw if any(v for v in r)]
        if not raw: messagebox.showwarning("No data","Please enter data."); return

        groups = []; dv_rows = []; skipped = 0
        for row in raw:
            if len(row) < 2: skipped += 1; continue
            grp = row[0].strip()
            if not grp: skipped += 1; continue
            vals = []
            for v in row[1:]:
                if not v: break
                try: vals.append(float(v.replace(",",".")))
                except ValueError: break
            if len(vals) >= 2:
                groups.append(grp); dv_rows.append(vals)
            else: skipped += 1

        if skipped: messagebox.showinfo("Note", f"{skipped} row(s) skipped.")

        n = len(dv_rows)
        if n < 4: messagebox.showwarning("Too few data","Need ≥ 4 complete observations."); return

        min_dv = min(len(r) for r in dv_rows)
        if min_dv < 2:
            messagebox.showwarning("Too few DVs","Need at least 2 dependent variables."); return

        # Align all rows to same number of DVs
        Y = np.array([r[:min_dv] for r in dv_rows], dtype=float)
        p = min_dv  # number of DVs
        dv_names_used = dv_names[:p]

        group_levels = first_seen(groups)
        k = len(group_levels)

        # ── Guard 1: at least 2 groups ──
        if k < 2:
            messagebox.showwarning("Only one group","MANOVA requires ≥ 2 groups."); return

        # ── Guard 2: n > p per group (critical!) ──
        groups_data = {}
        for lv in group_levels:
            idx_ = [i for i, g in enumerate(groups) if g == lv]
            groups_data[lv] = Y[idx_]
        for lv in group_levels:
            n_lv = len(groups_data[lv])
            if n_lv <= p:
                messagebox.showerror("n ≤ p VIOLATION",
                    f"Group '{lv}' has {n_lv} observation(s) but {p} dependent variables.\n\n"
                    "MANOVA requires n > p (observations > DVs) in EVERY group.\n"
                    "Otherwise the within-group covariance matrix is singular\n"
                    "and cannot be inverted — MANOVA is mathematically impossible.\n\n"
                    "Solutions:\n"
                    "• Collect more data\n"
                    "• Reduce the number of dependent variables\n"
                    "• Use PCA first to reduce dimensionality, then ANOVA on PC scores")
                return

        # ── Guard 3: Check multicollinearity among DVs ──
        corr_Y = np.corrcoef(Y.T)
        high_corr_pairs = []
        for i in range(p):
            for j in range(i+1, p):
                if abs(corr_Y[i,j]) > 0.90:
                    high_corr_pairs.append((dv_names_used[i], dv_names_used[j], corr_Y[i,j]))
        if high_corr_pairs:
            details = "\n".join(f"  • '{a}' & '{b}': r={c:.3f}" for a,b,c in high_corr_pairs)
            ans = messagebox.askyesno("High multicollinearity among DVs",
                "The following dependent variable pairs are highly correlated (|r| > 0.90):\n"
                + details + "\n\n"
                "High multicollinearity reduces the power of MANOVA and may lead\n"
                "to a singular or near-singular covariance matrix.\n\n"
                "Recommendation: consider removing redundant DVs or using PCA first.\n\n"
                "Continue anyway?")
            if not ans: return

        # ── Guard 4: Multivariate normality (Mardia) ──
        # Test on combined residuals (within-group deviations)
        Y_res = np.vstack([groups_data[lv] - np.mean(groups_data[lv], axis=0)
                            for lv in group_levels])
        b1p, p_sk, b2p, p_ku = self._mardia_test(Y_res)
        mv_normal = True
        if (not math.isnan(p_sk) and p_sk < alpha) or (not math.isnan(p_ku) and p_ku < alpha):
            mv_normal = False
            ans = messagebox.askyesno("Multivariate normality violated (Mardia's test)",
                f"Mardia's skewness:  b1p={fmt(b1p,4)},  p={fmt(p_sk,4)}\n"
                f"Mardia's kurtosis:  b2p={fmt(b2p,4)},  p={fmt(p_ku,4)}\n\n"
                "The multivariate normality assumption appears violated.\n"
                "Pillai's Trace is the most robust statistic in this case.\n\n"
                "Note: MANOVA is reasonably robust with larger samples (n > 20/group).\n\n"
                "Continue? (Pillai's Trace will be highlighted as most reliable)")
            if not ans: return

        # ── Guard 5: Box's M test ──
        box_chi2, box_p = self._box_m_test(groups_data, group_levels)
        if not math.isnan(box_p) and box_p < 0.001:
            ans = messagebox.askyesno("Heterogeneous covariance matrices (Box's M)",
                f"Box's M: χ²={fmt(box_chi2,4)},  p={fmt(box_p,6)}\n\n"
                "Covariance matrices differ significantly across groups.\n"
                "Note: Box's M is extremely sensitive to non-normality.\n"
                "If p is only slightly < 0.001, this may be a false alarm.\n\n"
                "Pillai's Trace is most robust when this assumption is violated.\n\n"
                "Continue?")
            if not ans: return

        # ══ MANOVA computation ══
        # Between-groups matrix H and Within-groups matrix E
        grand_mean = np.mean(Y, axis=0)
        E = np.zeros((p, p))  # within (error)
        H = np.zeros((p, p))  # between (hypothesis)
        for lv in group_levels:
            Xl = groups_data[lv]
            nl = len(Xl)
            grp_mean = np.mean(Xl, axis=0)
            E += (Xl - grp_mean).T @ (Xl - grp_mean)
            H += nl * np.outer(grp_mean - grand_mean, grp_mean - grand_mean)

        df_h = k - 1        # between df
        df_e = n - k        # within df

        # Eigenvalues of E⁻¹H
        try:
            E_inv = np.linalg.pinv(E)
            EinvH = E_inv @ H
            eigenvalues = np.real(np.linalg.eigvals(EinvH))
            eigenvalues = np.sort(eigenvalues[eigenvalues > 1e-10])[::-1]
        except Exception as ex:
            messagebox.showerror("Computation error",
                f"Could not compute eigenvalues: {ex}\n"
                "Check for singular covariance matrix (n ≤ p in some group)."); return

        if len(eigenvalues) == 0:
            messagebox.showerror("No valid eigenvalues",
                "The covariance matrix is singular. Cannot compute MANOVA.\n"
                "Ensure n > p in every group."); return

        s = min(df_h, p)   # number of non-zero eigenvalues

        # ── Four test statistics ──
        # Wilks' Lambda
        wilks_L = float(np.prod(1 / (1 + eigenvalues[:s])))
        # Approximate F for Wilks
        m_w = df_e + df_h - (p + df_h + 1) / 2
        q_w = math.sqrt((p**2 * df_h**2 - 4) / (p**2 + df_h**2 - 5)) if (p**2 + df_h**2 - 5) > 0 else 1
        df1_w = p * df_h
        df2_w = m_w * q_w - p * df_h / 2 + 1
        F_wilks = ((1 - wilks_L**(1/q_w)) / (wilks_L**(1/q_w))) * (df2_w / df1_w) if (wilks_L > 0 and q_w > 0) else np.nan
        p_wilks = float(1 - f_dist.cdf(F_wilks, df1_w, df2_w)) if not math.isnan(F_wilks) else np.nan

        # Pillai's Trace
        pillai_V = float(np.sum(eigenvalues[:s] / (1 + eigenvalues[:s])))
        # Approximate F for Pillai
        m_p = max(p, df_h)
        F_pillai = (pillai_V / s) / ((s - pillai_V) / s) * ((df_e + df_h - m_p - 1) / m_p) if (s - pillai_V > 0 and m_p > 0) else np.nan
        df1_p = s * m_p; df2_p = s * (df_e + df_h - m_p - 1)
        p_pillai = float(1 - f_dist.cdf(F_pillai, df1_p, df2_p)) if not math.isnan(F_pillai) else np.nan

        # Hotelling-Lawley Trace
        hl_T = float(np.sum(eigenvalues[:s]))
        b_hl = (df_e + df_h - p - 1) * hl_T / s if s > 0 else np.nan
        df1_hl = s * p; df2_hl = s * (df_e + df_h - p - 1)
        F_hl   = b_hl * (df2_hl / (df1_hl * s)) if (not math.isnan(b_hl) and df1_hl > 0 and s > 0) else np.nan
        p_hl   = float(1 - f_dist.cdf(F_hl, df1_hl, df2_hl)) if not math.isnan(F_hl) else np.nan

        # Roy's GCR
        roy_GCR = float(eigenvalues[0]) if len(eigenvalues) > 0 else np.nan
        # Upper bound F for Roy
        F_roy = roy_GCR * df_e / p if p > 0 else np.nan
        p_roy = float(1 - f_dist.cdf(F_roy, p, df_e)) if not math.isnan(F_roy) else np.nan

        # ── Univariate follow-up ANOVAs ──
        univ_rows = []
        bonf_alpha = alpha / p  # Bonferroni correction
        for dv_i, dv_nm in enumerate(dv_names_used):
            y_i = Y[:, dv_i]
            grand_m = np.mean(y_i)
            ss_b = sum(len(groups_data[lv]) * (np.mean(groups_data[lv][:,dv_i]) - grand_m)**2
                       for lv in group_levels)
            ss_w = sum(np.sum((groups_data[lv][:,dv_i] - np.mean(groups_data[lv][:,dv_i]))**2)
                       for lv in group_levels)
            ms_b = ss_b / df_h if df_h > 0 else np.nan
            ms_w = ss_w / df_e if df_e > 0 else np.nan
            F_i  = ms_b / ms_w if (not math.isnan(ms_w) and ms_w > 0) else np.nan
            p_i  = float(1 - f_dist.cdf(F_i, df_h, df_e)) if not math.isnan(F_i) else np.nan
            eta2_i = ss_b / (ss_b + ss_w) if (ss_b + ss_w) > 0 else np.nan
            mark_bonf = "significant" if (not math.isnan(p_i) and p_i < bonf_alpha) else "ns"
            univ_rows.append([dv_nm, fmt(F_i,4), fmt(p_i,4),
                              fmt(eta2_i,4), eta2_label(eta2_i),
                              f"α_Bonf = {fmt(bonf_alpha,4)}", mark_bonf])

        self._show_results(wilks_L, F_wilks, p_wilks,
                           pillai_V, F_pillai, p_pillai,
                           hl_T, F_hl, p_hl,
                           roy_GCR, F_roy, p_roy,
                           df1_w, df2_w, df1_p, df2_p, df1_hl, df2_hl,
                           univ_rows, dv_names_used,
                           b1p, p_sk, b2p, p_ku, box_chi2, box_p,
                           alpha, mv_normal, groups_data, group_levels, Y, p)

    def _show_results(self, wilks_L, F_wilks, p_wilks,
                      pillai_V, F_pillai, p_pillai,
                      hl_T, F_hl, p_hl,
                      roy_GCR, F_roy, p_roy,
                      df1_w, df2_w, df1_p, df2_p, df1_hl, df2_hl,
                      univ_rows, dv_names,
                      b1p, p_sk, b2p, p_ku, box_chi2, box_p,
                      alpha, mv_normal, groups_data, group_levels, Y, p):
        win = tk.Toplevel(self.win); win.title("MANOVA — Results")
        win.geometry("1180x800"); set_icon(win)

        main = tk.Frame(win); main.pack(fill=tk.BOTH, expand=True)
        vsb = ttk.Scrollbar(main, orient="vertical"); vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas = tk.Canvas(main, yscrollcommand=vsb.set); canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.config(command=canvas.yview)
        body = tk.Frame(canvas); canvas.create_window((0,0), window=body, anchor="nw")
        body.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        win.bind("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)),"units"))

        def _head(txt):
            tk.Label(body, text=txt, font=("Times New Roman",12,"bold"),
                     anchor="w").pack(fill=tk.X, padx=10, pady=(8,2))
        def _txt(txt, color="#000000"):
            tk.Label(body, text=txt, font=("Times New Roman",11), fg=color,
                     anchor="w", justify="left").pack(fill=tk.X, padx=10, pady=1)
        def _tbl(headers, rows):
            f, _ = make_tv(body, headers, rows); f.pack(fill=tk.X, padx=10, pady=(2,8))

        _head("MANOVA — Multivariate Analysis of Variance")

        # Assumption checks
        _head("Assumption Checks")
        _txt(f"Mardia's skewness (multivar. normality):  b1p={fmt(b1p,4)},  p={fmt(p_sk,4)}  "
             f"{'✓ OK' if math.isnan(p_sk) or p_sk > alpha else '⚠ violated'}",
             "#000000" if math.isnan(p_sk) or p_sk > alpha else "#c62828")
        _txt(f"Mardia's kurtosis (multivar. normality):  b2p={fmt(b2p,4)},  p={fmt(p_ku,4)}  "
             f"{'✓ OK' if math.isnan(p_ku) or p_ku > alpha else '⚠ violated'}",
             "#000000" if math.isnan(p_ku) or p_ku > alpha else "#c62828")
        _txt(f"Box's M (homogeneity of cov. matrices):  χ²={fmt(box_chi2,4)},  p={fmt(box_p,6)}  "
             f"{'✓ OK' if math.isnan(box_p) or box_p >= 0.001 else '⚠ significant (but Box M is sensitive to non-normality)'}",
             "#000000" if math.isnan(box_p) or box_p >= 0.001 else "#b07000")

        if not mv_normal:
            _txt("⚠ Multivariate normality violated → Pillai's Trace is most reliable statistic.",
                 "#c62828")

        # MANOVA test statistics
        _head("MANOVA Test Statistics")
        recommended = "Pillai" if not mv_normal else "Wilks"
        manova_rows = [
            ["Wilks' Lambda",       fmt(wilks_L,6), fmt(F_wilks,4), f"{int(df1_w)},{int(df2_w)}",
             fmt(p_wilks,4), sig_mark(p_wilks), "★ standard" if recommended=="Wilks" else ""],
            ["Pillai's Trace",      fmt(pillai_V,6), fmt(F_pillai,4), f"{int(df1_p)},{int(df2_p)}",
             fmt(p_pillai,4), sig_mark(p_pillai), "★ most robust" if recommended=="Pillai" else "robust"],
            ["Hotelling-Lawley",    fmt(hl_T,6), fmt(F_hl,4), f"{int(df1_hl)},{int(df2_hl)}",
             fmt(p_hl,4), sig_mark(p_hl), ""],
            ["Roy's GCR",           fmt(roy_GCR,6), fmt(F_roy,4), f"–",
             fmt(p_roy,4), sig_mark(p_roy), "upper bound"],
        ]
        _tbl(["Statistic","Value","F","df","p","Sig.","Note"], manova_rows)

        # Interpretation
        all_sig = all(not math.isnan(r[4]) and r[4] != "" and
                      (float(r[4]) < alpha if r[4] not in ("","–") else False)
                      for r in manova_rows if r[4] not in ("","–","н/д"))
        if not math.isnan(p_pillai) and p_pillai < alpha:
            _txt(f"✓ MANOVA significant (Pillai p={fmt(p_pillai,4)}): groups differ on the\n"
                 f"  combination of dependent variables. Proceed to univariate follow-up tests.",
                 "#1a6b1a")
        elif not math.isnan(p_pillai):
            _txt(f"✗ MANOVA not significant (Pillai p={fmt(p_pillai,4)}): no evidence that\n"
                 f"  groups differ on the combination of DVs. Univariate tests not recommended.",
                 "#c62828")

        # Univariate follow-up
        _head(f"Univariate Follow-Up ANOVAs (Bonferroni α = {fmt(alpha/len(dv_names),4)})")
        _txt("Note: these are only interpretable after a significant MANOVA.",
             "#666666")
        _tbl(["Dependent Variable","F","p","partial η²","Effect","Bonf. α","Result"], univ_rows)

        # Group means per DV
        _head("Group Means per Dependent Variable")
        means_headers = ["Group"] + dv_names
        means_rows = []
        for lv in group_levels:
            row_ = [lv] + [fmt(float(np.mean(groups_data[lv][:,j])),4)
                           for j in range(len(dv_names))]
            means_rows.append(row_)
        _tbl(means_headers, means_rows)

        # Visualization: means per DV per group
        if HAS_MPL and len(dv_names) >= 2:
            n_dv = len(dv_names)
            fig = Figure(figsize=(min(11, n_dv*1.8+1), 4.5), dpi=100)
            colors_ = ["#4c72b0","#dd8452","#55a868","#c44e52","#8172b2","#937860"]
            for di, dv_nm in enumerate(dv_names):
                ax = fig.add_subplot(1, n_dv, di+1)
                grp_means_ = [float(np.mean(groups_data[lv][:,di])) for lv in group_levels]
                grp_ses_   = [float(np.std(groups_data[lv][:,di],ddof=1)/math.sqrt(len(groups_data[lv])))
                              for lv in group_levels]
                xpos = range(len(group_levels))
                ax.bar(xpos, grp_means_, yerr=grp_ses_, capsize=4,
                       color=[colors_[i % len(colors_)] for i in range(len(group_levels))],
                       alpha=0.8, error_kw={"ecolor":"#333","lw":1.5})
                ax.set_xticks(list(xpos)); ax.set_xticklabels(group_levels, rotation=30, ha="right", fontsize=8)
                ax.set_title(dv_nm, fontsize=9); ax.set_ylabel("Mean ± SE" if di==0 else "")
                ax.yaxis.grid(True, alpha=0.3)
            fig.suptitle("Group Means (±SE) per Dependent Variable", fontsize=10)
            fig.tight_layout()
            cv = FigureCanvasTkAgg(fig, master=body); cv.draw()
            cv.get_tk_widget().pack(fill=tk.X, padx=10, pady=6)



# ═══════════════════════════════════════════════════════════════
# ДОВІДКОВА СИСТЕМА
# ═══════════════════════════════════════════════════════════════

HELP_CONTENT = {}

def _fill_help():
    H = HELP_CONTENT
    H["Швидкий старт"] = {"icon":"🚀","short":"З чого почати роботу з програмою","text":"""
ШВИДКИЙ СТАРТ

КРОК 1. Оберіть тип аналізу на головному екрані.
КРОК 2. Введіть дані:
  Ctrl+V - вставити з Excel
  Файл -> Завантажити Excel (.xlsx)
  Вручну (Enter - наступний рядок, стрілки - навігація)
КРОК 3. Натисніть "Аналіз даних".
КРОК 4. Програма автоматично перевірить:
  ✓ Нормальність розподілу (Shapiro-Wilk)
  ✓ Однорідність дисперсій (Левен)
  ✓ Методологічну коректність дій
КРОК 5. Скопіюйте звіт і графіки у Word.

Збереження: Файл -> Зберегти проект (.sadp)
"""}

    H["Введення даних"] = {"icon":"📋","short":"Як правильно вводити та копіювати дані","text":"""
ВВЕДЕННЯ ДАНИХ

СТРУКТУРА ТАБЛИЦІ ДЛЯ ANOVA:
  Ліві колонки = назви рівнів факторів (текст)
  Праві колонки = числові значення повторностей

  Варіант   | Повт.1 | Повт.2 | Повт.3
  Контроль  |  4.2   |  4.5   |  4.1
  Варіант1  |  5.8   |  6.1   |  5.9

СПОСОБИ ВВЕДЕННЯ:
  1. Вручну: клік -> введіть -> Enter (наступний рядок)
  2. Ctrl+V: вставити з Excel
  3. Файл -> Завантажити Excel (.xlsx, .xlsm)
  4. Fill-handle: наведіть на правий нижній кут -> курсор змінюється -> тягніть вниз
  5. Ctrl+C: скопіювати виділений діапазон

⚠ Назви варіантів вводьте ОДНАКОВО (регістр важливий!)
⚠ Десяткова кома автоматично замінюється на крапку
"""}

    H["Дизайни експерименту"] = {"icon":"🧪","short":"CRD, RCBD та Split-plot: коли що обирати","text":"""
ДИЗАЙНИ ПОЛЬОВИХ ЕКСПЕРИМЕНТІВ

CRD (Повна рандомізація):
  ✓ Умови однорідні (лабораторія, вегетаційні горщики)
  ✓ Мала кількість варіантів (2-5)

RCBD (Блочна рандомізація):
  ✓ Є градієнт родючості (схил, зволоження, pH)
  ✓ Польові досліди з багатьма варіантами
  Кожна числова колонка = один блок (повторність)
  ✓ Виключає міжблокову варіацію з помилки -> точніший аналіз

Split-plot (Спліт-плот):
  ✓ Два фактори з РІЗНИМ розміром ділянок
  Головний фактор (WP) = великі ділянки (обробіток ґрунту, норма посіву)
  Підфактор (SP) = малі ділянки (сорти, дози добрив)

  ⚠ УВАГА: два різних похибки!
    Whole-plot error -> для тесту WP фактора
    Sub-plot error (залишок) -> для SP та взаємодії
  ⚠ Застосування RCBD замість Split-plot -> неправильні F-значення!

ЯК ОБРАТИ:
  Немає блоків -> CRD
  Є блоки, однаковий розмір ділянок -> RCBD
  Є блоки, різний розмір ділянок -> Split-plot
"""}

    H["Нормальність розподілу"] = {"icon":"📊","short":"Shapiro-Wilk і дії при порушенні","text":"""
НОРМАЛЬНІСТЬ РОЗПОДІЛУ - SHAPIRO-WILK

Перевіряється нормальність ЗАЛИШКІВ моделі (не сирих даних!).

p > 0.05 -> нормальний -> параметричний аналіз ✓
p <= 0.05 -> ненормальний -> трансформація або непараметричний

⚠ ОБМЕЖЕННЯ:
  n < 8: тест ненадійний
  n > 100: виявляє найменші відхилення -> дивіться QQ-plot!

ДІЇ ПРИ НЕНОРМАЛЬНОСТІ:

1. ТРАНСФОРМАЦІЯ (повернення до параметричних методів):
   ln(x)  - для правоскошених даних, відносних показників
   sqrt(x) - для даних з рахунком (кількість особин, плям)
   log10(x) - для великих значень
   
   Програма: перевіряє допустимість -> виконує -> перевіряє нормальність знову
   ✓ Нормальний -> пропонує параметричний метод
   ✗ Все ще ненормальний -> зупиняється, рекомендує непараметричний

2. НЕПАРАМЕТРИЧНИЙ АНАЛІЗ:
   CRD: Kruskal-Wallis -> Mann-Whitney (post-hoc)
   RCBD: Friedman -> Wilcoxon (парний)
"""}

    H["Однорідність дисперсій"] = {"icon":"⚖️","short":"Тест Левена і наслідки порушення","text":"""
ОДНОРІДНІСТЬ ДИСПЕРСІЙ - ТЕСТ ЛЕВЕНА

Levene (center=median) = варіант Brown-Forsythe - найробустніший.

p >= 0.05 -> умова виконується ✓
p < 0.05 -> дисперсії різняться -> програма блокує і запитує підтвердження

ДІЇ ПРИ ПОРУШЕННІ:
  1. Трансформація (ln або sqrt стабілізують дисперсію)
  2. Для двох груп -> Welch t-test (у модулі t-тест, автоматично)
  3. Непараметричні методи не потребують рівності дисперсій

ANOVA достатньо робастна якщо:
  Розміри груп приблизно рівні
  Відношення max/min дисперсій < 9:1
"""}

    H["Методи порівнянь"] = {"icon":"🔬","short":"НІР, Тьюкі, Дункан, Бонферроні","text":"""
МЕТОДИ МНОЖИННИХ ПОРІВНЯНЬ (POST-HOC)

НІР05 (Fisher Protected LSD):
  ✓ Найпоширеніший в агрономії України
  ✓ Виконується ТІЛЬКИ після значущого F (Protected LSD)
  Рекомендується: <= 6 варіантів

Тест Тьюкі (Tukey HSD):
  ✓ Строгий контроль сімейної помилки (FWER)
  Рекомендується: будь-яка кількість варіантів

Тест Дункана:
  ✓ Справжня степ-даун процедура: alpha_p = 1-(1-alpha)^(p-1)
  ✓ Проміжний між LSD і Тьюкі
  Рекомендується: 5-10 варіантів

Бонферроні:
  Найконсервативніший. p_adj = p x кількість_пар
  Рекомендується: мала кількість запланованих порівнянь

CLD ЛІТЕРИ:
  Однакові літери = немає значущої різниці
  Різні літери = є значуща різниця (p < alpha)

ЗВЕДЕННЯ:
  <= 6 варіантів -> НІР05
  5-10 варіантів -> Дункан
  Будь-яка кількість -> Тьюкі
  Мало запланованих -> Бонферроні
"""}

    H["Непараметричні тести"] = {"icon":"📉","short":"KW, Friedman, MWU, Wilcoxon та розміри ефектів","text":"""
НЕПАРАМЕТРИЧНІ МЕТОДИ

Kruskal-Wallis (аналог ANOVA для CRD):
  Глобальний тест. Розмір ефекту: epsilon^2
  < 0.01 слабкий | 0.01-0.06 середній | > 0.14 сильний
  Post-hoc: Mann-Whitney з Бонферроні (автоматично)

Friedman (аналог RCBD ANOVA):
  Для блочних дизайнів. Розмір ефекту: Kendall W
  ⚠ Вимагає повні блоки (всі варіанти у кожному блоці)!
  Post-hoc: Wilcoxon з Бонферроні

Mann-Whitney U (попарне):
  Розмір ефекту: Cliff delta
  |delta| < 0.147: дуже слабкий | 0.147-0.33: слабкий
  0.33-0.474: середній | > 0.474: сильний

Wilcoxon signed-rank (парний):
  Для пар спостережень або RCBD з 2 варіантами.
  Розмір ефекту: r = Z/sqrt(n)
"""}

    H["Типи SS"] = {"icon":"∑","short":"Типи сум квадратів I-IV","text":"""
ТИПИ СУМ КВАДРАТІВ

При збалансованих даних - всі типи дають однаковий результат.
Різниця виникає при незбалансованих даних.

Тип I (Послідовний):
  Порядок факторів ВАЖЛИВИЙ!
  ⚠ Програма попередить при незбалансованих даних!
  Коли: збалансовані дизайни, регресія з осмисленим порядком

Тип II (Ієрархічний):
  Порядок НЕ важливий. Вища потужність ніж III при відсутності взаємодій.
  Коли: незбалансовані БЕЗ значущих взаємодій

Тип III (Частковий) <- ЗА ЗАМОВЧУВАННЯМ:
  Кожен ефект при всіх інших (включно зі взаємодіями).
  Порядок НЕ важливий. Стандарт SPSS/SAS. ✓
  Коли: більшість випадків, незбалансовані зі взаємодіями

Тип IV:
  Для незбалансованих з ПРОПУЩЕНИМИ КЛІТИНКАМИ.
  Рідкісний у польових дослідах.

РЕКОМЕНДАЦІЯ:
  Збалансований -> будь-який тип
  Незбалансований + взаємодії -> Тип III
  Незбалансований + без взаємодій -> Тип II
  Пропущені клітинки -> Тип IV
"""}

    H["Кореляційний аналіз"] = {"icon":"🔗","short":"Пірсон vs Спірмен, поправки, теплова карта","text":"""
КОРЕЛЯЦІЙНИЙ АНАЛІЗ

СТРУКТУРА ДАНИХ (два варіанти):
  А) Перший рядок = назва показника, нижче - числові значення
  Б) Перша колонка = назва показника, праворуч - значення

МЕТОДИ:
  Авто (рекомендовано): Shapiro-Wilk по кожному показнику
    хоч один ненормальний -> Спірмен
    всі нормальні -> Пірсон

  Пірсон r: лінійний зв'язок, нормальний розподіл
    ⚠ При ненормальних даних -> програма попередить!

  Спірмен rho: монотонний зв'язок, будь-який розподіл

ПОПРАВКИ НА МНОЖИННІ ПОРІВНЯННЯ:
  Бонферроні: строга (FWER), p_adj = p x кількість_пар
  BH (Benjamini-Hochberg FDR): ліберальніша, більш потужна
  -> При > 10 показниках рекомендується BH

ТЕПЛОВА КАРТА - кожна клітинка:
  r = коефіцієнт | p = значущість після поправки | n = кількість пар
  * p < 0.05 | ** p < 0.01

ІНТЕРПРЕТАЦІЯ |r|:
  0.0-0.2: дуже слабкий | 0.2-0.4: слабкий | 0.4-0.6: помірний
  0.6-0.8: сильний | 0.8-1.0: дуже сильний
"""}

    H["Регресійний аналіз"] = {"icon":"📈","short":"7 моделей регресії, R², діагностика","text":"""
РЕГРЕСІЙНИЙ АНАЛІЗ

МОДЕЛІ:
  1. Лінійна:       y = a + b*x
  2. Квадратична:   y = a + b*x + c*x^2  (оптимум добрив)
  3. Кубічна:       y = a + b*x + c*x^2 + d*x^3
  4. Степенева:     y = a * x^b  (x > 0)
  5. Експоненційна: y = a * e^(b*x)
  6. Логарифмічна:  y = a + b*ln(x)  (x > 0)
  7. Логістична 4P: y = d + (a-d)/(1+(x/c)^b)  (S-крива)

ПОКАЗНИКИ ЯКОСТІ:
  R^2   = частка варіації пояснена моделлю (0-1; > 0.9 - добре)
  R^2adj = R^2 з урахуванням кількості параметрів
  RMSE  = середньоквадратична помилка (в одиницях y)
  F-тест = значущість моделі загалом

ДІАГНОСТИКА:
  Residuals vs Fitted: будь-який патерн -> модель неповна!
  Shapiro-Wilk залишків: p > 0.05 -> нормальні ✓
  Grubbs test: автоматичне виявлення викидів у залишках

ВВЕДЕННЯ ДАНИХ:
  Ліве поле: значення x | Праве поле: значення y
  "Вставити дані": два стовпці з Excel (x | y)
"""}

    H["ANCOVA"] = {"icon":"🎛️","short":"Коваріаційний аналіз: контроль зовнішніх факторів","text":"""
ANCOVA - КОВАРІАЦІЙНИЙ АНАЛІЗ

Мета: порівняти групи усуваючи вплив коваріати.

КОВАРІАТА - неперервна змінна яку вимірюєте але не контролюєте:
  Початкова висота рослин, pH грунту, вміст гумусу, опади

СТРУКТУРА:
  [Група] | [Коваріата 1] | [Коваріата 2] | [Y - залежна змінна]

ЗАХИСТ (8 перевірок):
  1. n >= 6 спостережень
  2. >= 2 груп
  3. >= 2 спостережень у кожній групі
  4. Мультиколінеарність коваріат (r > 0.95)
  5. Паралельність ліній регресії <- КЛЮЧОВА!
  6. Нормальність залишків
  7. Однорідність дисперсій
  8. Скориговані середні (LS Means)

⚠ КРИТИЧНА ПЕРЕДУМОВА - ПАРАЛЕЛЬНІСТЬ ЛІНІЙ:
  Тест взаємодії Група x Коваріата:
  p < alpha -> лінії НЕ паралельні -> ANCOVA ЗАБЛОКОВАНА!

РЕЗУЛЬТАТИ:
  Скориговані (LS) Means = середні при однаковій коваріаті у всіх групах
  Порівнюйте скориговані середні, а не нескориговані!
"""}

    H["MANOVA"] = {"icon":"🔢","short":"Багатовимірний аналіз кількох показників","text":"""
MANOVA - БАГАТОВИМІРНИЙ ДИСПЕРСІЙНИЙ АНАЛІЗ

Мета: порівняти групи за КІЛЬКОМА залежними змінними одночасно.

НАВІЩО НЕ КІЛЬКА ANOVA?
  4 ANOVA з alpha=0.05 -> 19% шанс хибної різниці!
  MANOVA контролює сімейну помилку.

⚠ КРИТИЧНА ВИМОГА: n > p У КОЖНІЙ ГРУПІ
  n = кількість спостережень; p = кількість DV
  n <= p в будь-якій групі -> ЖОРСТКЕ БЛОКУВАННЯ!

ЗАХИСТ (7 перевірок):
  1. n > p у кожній групі (блокування)
  2. >= 2 DV
  3. Мультиколінеарність DV (|r| > 0.90)
  4. Багатовимірна нормальність (Mardia)
  5. Однорідність коваріаційних матриць (Box M)
  6. Чотири тестові статистики
  7. Univariate тільки після значущого MANOVA

СТАТИСТИКИ:
  Wilks Lambda: стандартна
  Pillai Trace: найробустніша ★ (при порушеннях)
  Hotelling-Lawley: один домінантний ефект
  Roy GCR: найпотужніша, найменш робастна

ПОСЛІДОВНІСТЬ:
  Перевірте передумови -> Pillai Trace ->
  Якщо значущо -> univariate ANOVAs з Bonferroni (alpha/p)
  Якщо незначущо -> univariate НЕ інтерпретуються!
"""}

    H["Аналіз стабільності"] = {"icon":"🌍","short":"GGE biplot та Eberhart-Russell для GxE","text":"""
АНАЛІЗ СТАБІЛЬНОСТІ - GxE ВЗАЄМОДІЯ

GxE: відносна продуктивність сортів змінюється між середовищами.

СТРУКТУРА:
  Рядки = Генотипи (сорти, лінії)
  Стовпці = Середовища (роки, місця)
  Значення = середня врожайність

EBERHART-RUSSELL:
  bi (коефіцієнт стабільності):
    bi ~= 1.0 -> стабільний (середня реакція)
    bi > 1.2  -> адаптивний (реагує на покращення умов)
    bi < 0.8  -> консервативний (для бідних умов)

  s^2d (дисперсія відхилень):
    ~= 0 -> передбачуваний
    > 0  -> непередбачуваний

  Ідеал: висока середня + bi ~= 1 + s^2d ~= 0

GGE BIPLOT:
  Точки = генотипи
  Стрілки = середовища
  Близько до центру = стабільний генотип
  Близько до стрілки середовища = добре виконує там
"""}

    H["Розмір вибірки"] = {"icon":"🔢","short":"Скільки повторностей потрібно?","text":"""
КАЛЬКУЛЯТОР РОЗМІРУ ВИБІРКИ

Параметри:
  alpha: рівень значущості (зазвичай 0.05)
  Потужність (1-beta): зазвичай 0.80 або 0.90
  delta: мінімальна різниця яку хочете виявити (наприклад 0.5 т/га)
  sigma: стандартне відхилення (з попередніх дослідів)
  k: кількість варіантів

РЕЖИМИ:
  Порожнє поле "повторності" -> мінімальна кількість повторностей
  Введіть повторності -> досягнута потужність тесту

ТИПОВІ ЗНАЧЕННЯ (зернові):
  CV 10-15%, виявити 10% різниці -> зазвичай 3-4 повторності
  CV 20-25% -> 5-7 повторностей

⚠ Планувати з потужністю >= 0.80!
"""}

    H["Проект"] = {"icon":"💾","short":"Збереження та відкриття проектів (.sadp)","text":"""
УПРАВЛІННЯ ПРОЕКТАМИ

Формат: .sadp (JSON-текст, можна відкрити у блокноті)

ЩО ЗБЕРІГАЄТЬСЯ:
  ✓ Кількість та назви факторів
  ✓ Всі введені дані (назви варіантів і числа)
  ✓ Кількість стовпців (повторностей)

ДЛЯ КІЛЬКОХ ПОКАЗНИКІВ ОДНОГО ДОСЛІДУ:
  1. Введіть схему досліду та перший показник
  2. Збережіть: Файл -> Зберегти проект
  3. Аналіз -> збережіть звіт
  4. Видаліть числові дані, залиште назви варіантів
  5. Введіть наступний показник -> аналіз -> звіт
  Або збережіть окремий .sadp для кожного показника.

МЕНЮ ФАЙЛ:
  Зберегти проект (Ctrl+S)
  Відкрити проект (Ctrl+O)
  Очистити таблицю
  Завантажити Excel
"""}

    H["Графіки і налаштування"] = {"icon":"🎨","short":"Boxplot, Venn, теплова карта та їх налаштування","text":"""
ГРАФІЧНИЙ ЗВІТ

BOXPLOT (коробка з вусами):
  Верхній вус: Q3 + 1.5*IQR (або максимум)
  Верхній край коробки: Q3 (75-й перцентиль)
  Лінія: медіана (Q2)
  Нижній край коробки: Q1 (25-й перцентиль)
  Нижній вус: Q1 - 1.5*IQR (або мінімум)
  Кружечки поза вусами: викиди (outliers)

  Літери CLD над коробками:
    Однакові -> немає значущої різниці
    Різні -> є значуща різниця (p < alpha)

ДІАГРАМА ВЕННА (сила впливу):
  Кола = головні ефекти факторів
  Перетини кіл = взаємодії між факторами
  Сума всіх частин = 100%

НАЛАШТУВАННЯ (кнопка ⚙ у вікні графіків):
  Boxplot: шрифт, розмір, кольори коробки/медіани/вусів/викидів
  Venn: шрифт, прозорість, кольори кіл і тексту
  За замовчуванням: APA стиль

КОПІЮВАННЯ:
  "Копіювати PNG" -> 300 dpi -> Ctrl+V у Word
  (Windows only; macOS/Linux - toolbar matplotlib)

ТЕПЛОВА КАРТА (кореляція):
  В клітинці: r / p / n
  Налаштовується через ⚙: colormap, шрифти, кольори
"""}

    H["Позначення у звіті"] = {"icon":"📝","short":"Як читати таблиці і показники у звіті","text":"""
ПОЗНАЧЕННЯ У ЗВІТІ

ЗНАЧУЩІСТЬ:
  **  p < 0.01 (висока значущість)
  *   p < 0.05 (значущо)
  -   p >= 0.05 (не значущо)

ТАБЛИЦЯ ANOVA:
  SS  = сума квадратів
  df  = ступені свободи
  MS  = середній квадрат (SS/df)
  F   = критерій Фішера (MS_ефект / MS_залишок)
  p   = ймовірність (менше = значущіше)

СИЛА ВПЛИВУ (% від SS):
  Частка загальної варіації по кожному джерелу.
  Залишок включений. Сума всіх рядків = 100%.

PARTIAL eta^2 (розмір ефекту):
  eta^2 = SS_ефект / (SS_ефект + SS_залишок)
  < 0.01: дуже слабкий | 0.01-0.06: слабкий
  0.06-0.14: середній | > 0.14: сильний

R^2 (коефіцієнт детермінації):
  Частка варіації пояснена всією моделлю.
  R^2 = 0.90 -> 90% варіації пояснено ✓

CV% (коефіцієнт варіації):
  < 10%: відмінна точність | 10-15%: хороша
  15-20%: задовільна | > 20%: низька

НІР05:
  Якщо різниця > НІР05 -> статистично значуща.
"""}

_fill_help()



class HelpWindow:
    """Інтерактивне вікно довідки."""
    def __init__(self, parent, start_topic=None):
        self.win = tk.Toplevel(parent)
        self.win.title("S.A.D. — Довідка")
        self.win.geometry("1000x660"); set_icon(self.win)
        self.current_topic = None
        self._build()
        topic = start_topic or list(HELP_CONTENT.keys())[0]
        self._show_topic(topic)

    def _build(self):
        # Left panel
        left = tk.Frame(self.win, width=220, bg="#f5f5f5", relief=tk.RIDGE, bd=1)
        left.pack(side=tk.LEFT, fill=tk.Y); left.pack_propagate(False)
        tk.Label(left, text="Зміст довідки",
                 font=("Times New Roman",12,"bold"), bg="#1a4b8c", fg="white",
                 pady=8).pack(fill=tk.X)
        # Search
        sf = tk.Frame(left, bg="#f5f5f5"); sf.pack(fill=tk.X, padx=6, pady=6)
        tk.Label(sf, text="Пошук:", bg="#f5f5f5", font=("Times New Roman",11)).pack(side=tk.LEFT)
        self._sv = tk.StringVar(); self._sv.trace_add("write", self._on_search)
        tk.Entry(sf, textvariable=self._sv, font=("Times New Roman",11),
                 relief=tk.FLAT, bg="white").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        # Topic buttons frame (scrollable)
        self._tf = tk.Frame(left, bg="#f5f5f5"); self._tf.pack(fill=tk.BOTH, expand=True)
        self._btn = {}; self._build_list(list(HELP_CONTENT.keys()))
        # Right panel
        right = tk.Frame(self.win, bg="white"); right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._title = tk.Label(right, text="", font=("Times New Roman",14,"bold"),
                               bg="#1a4b8c", fg="white", pady=8, padx=10, anchor="w")
        self._title.pack(fill=tk.X)
        tf2 = tk.Frame(right); tf2.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        ysb = ttk.Scrollbar(tf2); ysb.pack(side=tk.RIGHT, fill=tk.Y)
        self._txt = tk.Text(tf2, wrap="word", font=("Times New Roman",12),
                            state="disabled", relief=tk.FLAT, bg="white",
                            yscrollcommand=ysb.set, padx=10, pady=8, cursor="arrow")
        self._txt.pack(fill=tk.BOTH, expand=True)
        ysb.config(command=self._txt.yview)
        self._txt.bind("<MouseWheel>",
                       lambda e: self._txt.yview_scroll(int(-1*(e.delta/120)),"units"))
        # Tags
        self._txt.tag_configure("bold", font=("Times New Roman",12,"bold"))
        self._txt.tag_configure("check", foreground="#1a6b1a", font=("Times New Roman",12))
        self._txt.tag_configure("warn",  foreground="#c62828", font=("Times New Roman",12))
        self._txt.tag_configure("normal",font=("Times New Roman",12))
        # Bottom
        bot = tk.Frame(right, bg="#f0f0f0", pady=4); bot.pack(fill=tk.X)
        tk.Button(bot, text="<- Попередня", command=self._prev,
                  font=("Times New Roman",11)).pack(side=tk.LEFT, padx=8)
        tk.Button(bot, text="Наступна ->", command=self._next,
                  font=("Times New Roman",11)).pack(side=tk.LEFT, padx=4)
        tk.Button(bot, text="Закрити", command=self.win.destroy,
                  font=("Times New Roman",11)).pack(side=tk.RIGHT, padx=8)

    def _build_list(self, topics):
        for w in self._tf.winfo_children(): w.destroy()
        self._btn = {}
        for topic in topics:
            info = HELP_CONTENT.get(topic, {})
            frm = tk.Frame(self._tf, bg="#f5f5f5", cursor="hand2")
            frm.pack(fill=tk.X, padx=3, pady=1)
            icon = info.get("icon","•")
            lbl = tk.Label(frm, text=f"{icon}  {topic}",
                           font=("Times New Roman",11,"bold"), bg="#f5f5f5",
                           anchor="w", padx=6, pady=2)
            lbl.pack(fill=tk.X)
            sub = tk.Label(frm, text=f"    {info.get('short','')}",
                           font=("Times New Roman",9), fg="#666", bg="#f5f5f5", anchor="w", padx=6)
            sub.pack(fill=tk.X)
            for w in [frm, lbl, sub]:
                w.bind("<Button-1>", lambda e, t=topic: self._show_topic(t))
                w.bind("<Enter>",  lambda e, f=frm: [c.configure(bg="#dce8ff") for c in [f]+list(f.winfo_children())])
                w.bind("<Leave>",  lambda e, f=frm, t=topic: self._set_bg(f, t))
            self._btn[topic] = frm

    def _set_bg(self, frm, topic):
        bg = "#c8d8ff" if topic == self.current_topic else "#f5f5f5"
        frm.configure(bg=bg)
        for w in frm.winfo_children(): w.configure(bg=bg)

    def _on_search(self, *_):
        q = self._sv.get().strip().lower()
        if not q: self._build_list(list(HELP_CONTENT.keys())); return
        matched = [t for t, info in HELP_CONTENT.items()
                   if q in t.lower() or q in info.get("short","").lower()
                   or q in info.get("text","").lower()]
        self._build_list(matched)

    def _show_topic(self, topic):
        if topic not in HELP_CONTENT: return
        self.current_topic = topic
        info = HELP_CONTENT[topic]
        self._title.configure(text=f"{info.get('icon','')}  {topic}")
        for t, frm in self._btn.items(): self._set_bg(frm, t)
        self._txt.configure(state="normal")
        self._txt.delete("1.0", tk.END)
        for line in info.get("text","").strip().split("\n"):
            s = line.strip()
            if s.startswith("✓") or s.startswith("✔"):
                tag = "check"
            elif s.startswith("⚠") or s.startswith("✗") or "БЛОК" in s or "КРИТ" in s:
                tag = "warn"
            elif line.isupper() and len(line) > 3:
                tag = "bold"
            else:
                tag = "normal"
            self._txt.insert(tk.END, line + "\n", tag)
        self._txt.configure(state="disabled")
        self._txt.yview_moveto(0)

    def _prev(self):
        keys = list(HELP_CONTENT.keys())
        if self.current_topic in keys:
            idx = keys.index(self.current_topic)
            if idx > 0: self._show_topic(keys[idx-1])

    def _next(self):
        keys = list(HELP_CONTENT.keys())
        if self.current_topic in keys:
            idx = keys.index(self.current_topic)
            if idx < len(keys)-1: self._show_topic(keys[idx+1])


def show_help(parent, topic=None):
    HelpWindow(parent, start_topic=topic)

# ═══════════════════════════════════════════════════════════════
# UPDATE MENU — add ANCOVA and MANOVA buttons
# ═══════════════════════════════════════════════════════════════

_SADTk_orig_init = SADTk.__init__

def _SADTk_new_init(self, root):
    _SADTk_orig_init(self, root)
    # Replace main frame content
    for w in root.winfo_children(): w.destroy()
    root.geometry("1060x640")

    mf = tk.Frame(root, bg="white"); mf.pack(expand=True, fill=tk.BOTH)
    tk.Label(mf, text="S.A.D. — Statistical Analysis of Data",
             font=("Times New Roman", 20, "bold"), fg="#000000", bg="white").pack(pady=12)

    # ── ANOVA block ──
    sect1 = tk.LabelFrame(mf, text="  Analysis of Variance (ANOVA)  ",
                          font=("Times New Roman",12,"bold"), bg="white", fg="#1a4b8c")
    sect1.pack(fill=tk.X, padx=20, pady=4)
    bf = tk.Frame(sect1, bg="white"); bf.pack(pady=6)
    for i, (txt, fc) in enumerate([("One-factor ANOVA",1),("Two-factor ANOVA",2),
                                    ("Three-factor ANOVA",3),("Four-factor ANOVA",4)]):
        tk.Button(bf, text=txt, width=22, height=2, font=("Times New Roman",12),
                  command=lambda f=fc: self.open_table(f)).grid(row=0, column=i, padx=8, pady=4)

    # ── Other analyses ──
    sect2 = tk.LabelFrame(mf, text="  Other Statistical Methods  ",
                          font=("Times New Roman",12,"bold"), bg="white", fg="#1a4b8c")
    sect2.pack(fill=tk.X, padx=20, pady=4)
    bf2 = tk.Frame(sect2, bg="white"); bf2.pack(pady=6)

    btn_cfg = [
        ("Descriptive\nStatistics",   "#1a6b1a", lambda: DescriptiveWindow(root, self.graph_settings)),
        ("t-Test /\nMann-Whitney",     "#1a6b1a", lambda: TTestWindow(root)),
        ("Correlation\nAnalysis",      "#1a4b8c", lambda: CorrelationWindow(root, self.graph_settings)),
        ("Regression\nAnalysis",       "#4b1a8c", lambda: RegressionWindow(root, self.graph_settings)),
        ("ANCOVA",                     "#4b1a8c", lambda: AncovaWindow(root, self.graph_settings)),
        ("MANOVA",                     "#4b1a8c", lambda: ManovaWindow(root, self.graph_settings)),
        ("Repeated\nMeasures ANOVA",   "#4b1a8c", lambda: RepeatedMeasuresWindow(root, self.graph_settings)),
        ("Cluster\nAnalysis",          "#8c4b1a", lambda: ClusterWindow(root, self.graph_settings)),
        ("PCA",                        "#8c4b1a", lambda: PCAWindow(root, self.graph_settings)),
        ("Stability\nAnalysis (GxE)",  "#8c1a1a", lambda: StabilityWindow(root, self.graph_settings)),
        ("Sample Size\nCalculator",    "#555555", lambda: SampleSizeWindow(root)),
    ]
    for i, (txt, col, cmd) in enumerate(btn_cfg):
        tk.Button(bf2, text=txt, width=14, height=2, font=("Times New Roman",11),
                  bg=col, fg="white", command=cmd).grid(row=i//6, column=i%6, padx=5, pady=4)

    # ── Project buttons ──
    pf = tk.Frame(mf, bg="white"); pf.pack(pady=6)
    tk.Button(pf, text="📂 Open Project", width=18, font=("Times New Roman",11),
              command=self.load_project).pack(side=tk.LEFT, padx=8)
    tk.Button(pf, text="💾 Save Project", width=18, font=("Times New Roman",11),
              command=self.save_project).pack(side=tk.LEFT, padx=8)

    tk.Label(mf, text="Оберіть тип аналізу -> Введіть дані -> Натисніть «Аналіз даних»",
             font=("Times New Roman",11), fg="#666666", bg="white").pack(pady=4)

    # Help button
    tk.Button(mf, text="📚  Довідка", font=("Times New Roman",12),
              bg="#1a4b8c", fg="white", width=18,
              command=lambda: show_help(root)).pack(pady=(2,8))

    # re-init state (already done in orig_init but window was rebuilt)
    self.table_win = None; self.report_win = None; self.graph_win = None
    self._graph_figs = {}
    self._active_cell = None; self._active_prev = None
    self._sel_anchor = None; self._sel_cells = set(); self._sel_orig = {}
    self._fill_drag = False; self._fill_rows = []; self._fill_cols = []
    self.factor_title_map = {}
    self.graph_settings = dict(DEF_GS)
    self._current_project_path = None
    self._lbf_cache = {}

SADTk.__init__ = _SADTk_new_init


if __name__ == "__main__":
    root = tk.Tk()
    set_icon(root)
    app = SADTk(root)
    root.mainloop()
