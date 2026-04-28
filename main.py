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
    rows = []; sig = {}
    if MS is None or df is None or math.isnan(MS) or math.isnan(df): return rows, sig
    df = int(df)
    if df <= 0: return rows, sig
    lvls = [x for x in levels if not math.isnan(means.get(x, np.nan)) and ns.get(x, 0) > 0]
    m = len(lvls)
    if m < 2: return rows, sig
    for a, b in combinations(lvls, 2):
        ma, mb = means[a], means[b]; na, nb = ns[a], ns[b]
        se = math.sqrt(MS * (1 / na + 1 / nb))
        if se <= 0: continue
        tv = abs(ma - mb) / se; pr = 2 * (1 - float(t_dist.cdf(tv, df)))
        if method == "bonferroni":          pa = min(1., pr * (m * (m - 1) / 2))
        elif method in ("tukey", "duncan"): pa = float(1 - studentized_range.cdf(math.sqrt(2) * tv, m, df))
        else: pa = pr
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
    if ss_type == "I":   return _ss_type1(y, Xf, ts, fkeys)
    elif ss_type == "II": return _ss_type2(y, Xf, ts, fkeys)
    else:                 return _ss_type3(y, Xf, ts)

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
    ss_tot = 0.
    for row in table:
        if row[0] == "Загальна" and row[1] is not None and not (isinstance(row[1], float) and math.isnan(row[1])):
            ss_tot = float(row[1]); break
    if ss_tot <= 0:
        ss_tot = sum(float(r[1]) for r in table if r[1] is not None
                     and not (isinstance(r[1], float) and math.isnan(r[1])) and not r[0].startswith("Залишок"))
    out = []
    for row in table:
        nm, SSv = row[0], row[1]
        if nm.startswith("Залишок") or nm == "Загальна": continue
        if SSv is None or (isinstance(SSv, float) and math.isnan(SSv)): continue
        out.append([nm, fmt((float(SSv) / ss_tot * 100) if ss_tot > 0 else np.nan, 2)])
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
        names_var = tk.StringVar(value="перший рядок кожного стовпця")
        rb_f = ("Times New Roman", 12)
        tk.Radiobutton(frm, text="Перший рядок кожного стовпця", variable=names_var,
                       value="row", font=rb_f).grid(row=1, column=0, sticky="w")
        tk.Radiobutton(frm, text="Перша колонка (рядки = показники)", variable=names_var,
                       value="col", font=rb_f).grid(row=2, column=0, sticky="w")

        tk.Label(frm, text="Метод:", font=("Times New Roman", 12)).grid(row=3, column=0, sticky="w", pady=(10, 4))
        meth_var = tk.StringVar(value="pearson")
        tk.Radiobutton(frm, text="Пірсона (параметричний)", variable=meth_var,
                       value="pearson", font=rb_f).grid(row=4, column=0, sticky="w")
        tk.Radiobutton(frm, text="Спірмена (непараметричний)", variable=meth_var,
                       value="spearman", font=rb_f).grid(row=5, column=0, sticky="w")

        tk.Label(frm, text="Рівень значущості α:", font=("Times New Roman", 12)).grid(row=6, column=0, sticky="w", pady=(10, 4))
        alpha_var = tk.StringVar(value="0.05")
        ttk.Combobox(frm, textvariable=alpha_var, values=["0.01", "0.05", "0.10"], state="readonly", width=8).grid(row=6, column=1, sticky="w", padx=6)

        out = {"ok": False}
        def ok():
            out.update({"ok": True, "names_loc": names_var.get(),
                        "method": meth_var.get(), "alpha": float(alpha_var.get())})
            dlg.destroy()
        bf = tk.Frame(frm); bf.grid(row=7, column=0, columnspan=2, pady=(12, 0))
        tk.Button(bf, text="OK", width=10, command=ok).pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text="Скасувати", width=12, command=dlg.destroy).pack(side=tk.LEFT)
        dlg.update_idletasks(); center_win(dlg); dlg.bind("<Return>", lambda e: ok())
        dlg.grab_set(); self.win.wait_window(dlg)
        if not out["ok"]: return
        self._compute_and_show(out["names_loc"], out["method"], out["alpha"])

    def _compute_and_show(self, names_loc, method, alpha):
        # Extract data
        raw = [[e.get().strip() for e in row] for row in self.entries]
        # Remove completely empty rows
        raw = [r for r in raw if any(v for v in r)]
        if not raw: messagebox.showwarning("", "Немає даних."); return

        if names_loc == "col":
            # First column = label, rest = values per row
            labels = [row[0] for row in raw if row[0]]
            data_rows = []
            for row in raw:
                try: data_rows.append([float(v.replace(",", ".")) for v in row[1:] if v])
                except Exception: continue
            # transpose: each label is a variable
            n_vars = len(labels)
            # each row is one observation across variables — need equal length
            min_len = min(len(r) for r in data_rows) if data_rows else 0
            data_cols = [[data_rows[i][j] for i in range(min_len)] for j in range(n_vars)] if min_len > 0 else []
        else:
            # First row of each column = label, rest = values
            if not raw: return
            labels = [self.header_labels[j].cget("text") if j < len(self.header_labels) else f"P{j+1}"
                      for j in range(self.cols)]
            data_cols = []
            for j in range(self.cols):
                col_vals = []
                for i, row in enumerate(raw):
                    v = row[j] if j < len(row) else ""
                    if not v: continue
                    try: col_vals.append(float(v.replace(",", "."))); 
                    except Exception: continue
                data_cols.append(col_vals)

        # Keep only non-empty columns
        pairs = [(labels[j], data_cols[j]) for j in range(len(data_cols)) if len(data_cols[j]) >= 3]
        if len(pairs) < 2: messagebox.showwarning("", "Потрібно ≥ 2 показники з ≥ 3 значеннями."); return

        labels_clean = [p[0] for p in pairs]
        n = len(labels_clean)

        # Equalise lengths (pairwise → use minimum)
        min_n = min(len(p[1]) for p in pairs)
        arrays = [np.array(p[1][:min_n], dtype=float) for p in pairs]

        # Build correlation matrix
        r_mat = np.ones((n, n)); p_mat = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    if method == "pearson":
                        r_, p_ = pearsonr(arrays[i], arrays[j])
                    else:
                        r_, p_ = spearmanr(arrays[i], arrays[j])
                    r_mat[i, j] = r_mat[j, i] = float(r_)
                    p_mat[i, j] = p_mat[j, i] = float(p_)
                except Exception:
                    r_mat[i, j] = r_mat[j, i] = np.nan
                    p_mat[i, j] = p_mat[j, i] = np.nan

        self._show_heatmap(labels_clean, r_mat, p_mat, alpha, method)

    def _show_heatmap(self, labels, r_mat, p_mat, alpha, method):
        if not HAS_MPL: messagebox.showwarning("", "matplotlib недоступний."); return
        gs = self.gs
        win = tk.Toplevel(self.win); win.title("Теплова карта кореляцій"); win.geometry("820x720"); set_icon(win)
        tb = tk.Frame(win, padx=6, pady=4); tb.pack(fill=tk.X)
        tk.Button(tb, text="⚙ Налаштування", command=lambda: self._restyle(win, labels, r_mat, p_mat, alpha, method)).pack(side=tk.LEFT, padx=4)

        fig_frame = tk.Frame(win); fig_frame.pack(fill=tk.BOTH, expand=True)
        self._draw_heatmap(fig_frame, labels, r_mat, p_mat, alpha, method, gs)
        # also store ref for settings re-draw
        self._hm_data = (labels, r_mat, p_mat, alpha, method)
        self._hm_frame = fig_frame

    def _restyle(self, win, labels, r_mat, p_mat, alpha, method):
        dlg = GraphSettingsDlg(win, self.gs, show_heatmap=True)
        win.wait_window(dlg)
        if dlg.result:
            self.gs = dlg.result
            for w in self._hm_frame.winfo_children(): w.destroy()
            self._draw_heatmap(self._hm_frame, labels, r_mat, p_mat, alpha, method, self.gs)

    def _draw_heatmap(self, frame, labels, r_mat, p_mat, alpha, method, gs):
        n = len(labels)
        fig_h = max(5, n * 0.55 + 1.5); fig_w = max(5, n * 0.55 + 1.5)
        fig = Figure(figsize=(min(fig_w, 10), min(fig_h, 10)), dpi=100)
        ax = fig.add_subplot(111)

        cmap_name = gs.get("heatmap_cmap", "RdYlGn")
        fsize = gs.get("heatmap_font_size", 10)
        acol  = gs.get("heatmap_annot_color", "#000000")
        ff    = gs.get("font_family", "Times New Roman")

        try: cmap = matplotlib.cm.get_cmap(cmap_name)
        except Exception: cmap = matplotlib.cm.get_cmap("RdYlGn")

        # mask NaN
        masked = np.ma.array(r_mat, mask=np.isnan(r_mat))
        im = ax.imshow(masked, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

        ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right",
                                                      fontsize=fsize, fontfamily=ff)
        ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=fsize, fontfamily=ff)
        ax.set_title(f"Кореляційна матриця ({method.capitalize()}), α={alpha}",
                     fontsize=fsize + 1, fontfamily=ff)

        for i in range(n):
            for j in range(n):
                r_ = r_mat[i, j]; p_ = p_mat[i, j]
                if math.isnan(r_): continue
                mark = sig_mark(p_) if not math.isnan(p_) else ""
                txt = f"{r_:.2f}{mark}"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=fsize, color=acol, fontfamily=ff)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()

        self._hm_fig = fig
        cv = FigureCanvasTkAgg(fig, master=frame); cv.draw()
        cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Copy button
        def copy_hm():
            ok, msg = _copy_fig_to_clipboard(self._hm_fig)
            if ok: messagebox.showinfo("", "Скопійовано.")
            else: messagebox.showwarning("", f"Помилка: {msg}")
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
        for (r, c) in list(self._sel_cells): self._restore_bg(r, c)
        self._sel_cells.clear(); self._sel_anchor = None; self._sel_orig.clear()

    def _restore_bg(self, r, c):
        try: self.entries[r][c].configure(bg=self._sel_orig.get((r, c), "white"))
        except Exception: pass

    def _apply_sel(self, cells):
        for (r, c) in cells:
            try:
                e = self.entries[r][c]
                if (r, c) not in self._sel_orig: self._sel_orig[(r, c)] = e.cget("bg")
                if (r, c) == self._sel_anchor:
                    e.configure(bg=self.SEL_ANC)
                elif e.get().strip():          # highlight only cells with data
                    e.configure(bg=self.SEL_BG)
            except Exception: pass

    def _sel_range(self, r1, c1, r2, c2):
        prev = set(self._sel_cells)
        new = {(r, c) for r in range(min(r1, r2), max(r1, r2) + 1)
               for c in range(min(c1, c2), max(c1, c2) + 1)
               if r < len(self.entries) and c < len(self.entries[r])}
        for rc in prev - new: self._restore_bg(*rc)
        self._apply_sel(new - prev)
        if self._sel_anchor in new:
            try: self.entries[self._sel_anchor[0]][self._sel_anchor[1]].configure(bg=self.SEL_ANC)
            except Exception: pass
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
        tk.Button(row_d, text=" ? ", width=3, command=self.show_design_help).pack(side=tk.LEFT, padx=6)
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
        ssv = tk.StringVar(value="III")
        ssf = tk.Frame(frm); ssf.grid(row=4, column=1, sticky="w", pady=(10, 4), padx=(140, 0))
        for ss in ["I", "II", "III"]:
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
                    options = [("Краскела–Уолліса", "kw"), ("Манна-Уітні", "mw"),
                               ("🔁 Логарифмування + параметричний", "log_param")]
                else:
                    options = ([("Wilcoxon (парний)", "wilcoxon"), ("🔁 Логарифмування + параметричний", "log_param")]
                               if n_var == 2 else
                               [("Friedman", "friedman"), ("🔁 Логарифмування + параметричний", "log_param")])
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
        if not long: messagebox.showwarning("", "Немає числових даних."); return
        values = np.array([r["value"] for r in long], dtype=float)
        if len(values) < 3: messagebox.showinfo("", "Надто мало даних."); return

        lbf = {f: first_seen([r.get(f) for r in long]) for f in self.factor_keys}
        var_order = first_seen([tuple(r.get(f) for f in self.factor_keys) for r in long])
        v_names = [" | ".join(map(str, k)) for k in var_order]
        n_var = len(var_order)

        try:
            if design == "crd":    res = anova_crd(long, self.factor_keys, lbf, ss_type)
            elif design == "rcbd": res = anova_rcbd(long, self.factor_keys, lbf, ss_type=ss_type)
            else:
                if split_main not in self.factor_keys: split_main = self.factor_keys[0]
                res = anova_split(long, self.factor_keys, split_main, ss_type=ss_type)
        except Exception as ex: messagebox.showerror("Помилка моделі", str(ex)); return

        residuals = np.array(res.get("residuals", []), dtype=float)
        try: W, p_norm = shapiro(residuals) if len(residuals) >= 3 else (np.nan, np.nan)
        except Exception: W, p_norm = np.nan, np.nan

        normal = (not math.isnan(p_norm)) and (p_norm > 0.05)
        if design == "split" and not normal:
            messagebox.showwarning("Split-plot", "Залишки не нормальні → некоректний аналіз."); return

        choice = self.choose_method(p_norm, design, n_var)
        if not choice["ok"]: return
        method = choice["method"]

        log_applied = False
        if method == "log_param":
            if np.any(values <= 0):
                messagebox.showwarning("", "Є нулі/від'ємні → логарифмування неможливе."); return
            long = [dict(r, value=math.log(r["value"])) for r in long]
            values = np.array([r["value"] for r in long], dtype=float)
            log_applied = True
            try:
                if design == "crd":    res = anova_crd(long, self.factor_keys, lbf, ss_type)
                elif design == "rcbd": res = anova_rcbd(long, self.factor_keys, lbf, ss_type=ss_type)
                else: res = anova_split(long, self.factor_keys, split_main, ss_type=ss_type)
            except Exception as ex: messagebox.showerror("", str(ex)); return
            residuals = np.array(res.get("residuals", []), dtype=float)
            try: W, p_norm = shapiro(residuals) if len(residuals) >= 3 else (np.nan, np.nan)
            except Exception: W, p_norm = np.nan, np.nan
            method = "lsd"
            messagebox.showinfo("Логарифмування",
                f"Дані прологарифмовано.\nShapiro–Wilk після: p={fmt(p_norm,4)}\n"
                + ("✓ Нормальний" if p_norm > 0.05 else "✗ Все ще ненормальний"))

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

        kw_H = kw_p = kw_df = kw_eps = np.nan; do_ph = True
        fr_chi = fr_p = fr_df = fr_W = np.nan; wil_s = wil_p = np.nan
        rcbd_ph = []; rcbd_sig = {}
        lf = {f: {lv: "" for lv in lbf[f]} for f in self.factor_keys}
        lnamed = {nm: "" for nm in v_names}
        ph_rows = []; fpt = {}

        if method == "lsd":
            for f in self.factor_keys:
                MS_ = MS_wp if (design == "split" and f == split_mf) else MS_err
                df_ = df_wp if (design == "split" and f == split_mf) else df_err
                lf[f] = cld(lbf[f], fm[f], lsd_sig(lbf[f], fm[f], fn[f], MS_, df_))
            if design != "split":
                lnamed = cld(v_names, means1, lsd_sig(v_names, means1, ns1, MS_err, df_err))

        elif method in ("tukey", "duncan", "bonferroni"):
            if design != "split":
                ph_rows, sig_ = pairwise_param(v_names, means1, ns1, MS_err, df_err, method)
                lnamed = cld(v_names, means1, sig_)
                for f in self.factor_keys:
                    r_, s_ = pairwise_param(lbf[f], fm[f], fn[f], MS_err, df_err, method)
                    fpt[f] = r_; lf[f] = cld(lbf[f], fm[f], s_)
            else:
                for f in self.factor_keys:
                    MS_ = MS_wp if f == split_mf else MS_err; df_ = df_wp if f == split_mf else df_err
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
        if d['log_applied']: _txt("⚠ Застосовано логарифмування даних (натуральний логарифм).")
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
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    set_icon(root)
    app = SADTk(root)
    root.mainloop()
