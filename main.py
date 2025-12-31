# main.py
# -*- coding: utf-8 -*-

"""
S.A.D. — Статистичний аналіз даних (Tkinter)
Потрібно: Python 3.8+, numpy, scipy
Встановлення: pip install numpy scipy

ОНОВЛЕНО (за твоїми останніми вимогами):
1) Колонки вводу даних зроблено ще вужчими (шрифти НЕ змінював).
2) Таблиця з НІР подається ТІЛЬКИ якщо обрано НІР05.
3) Таблиці у звіті — з “чіткими комірками” (box-drawing символи), без “плюсиків”.
4) Для непараметричних методів у таблиці варіантів додається Медіана та IQR (Q1–Q3).
5) Для непараметричних методів літери формуються також (на основі парних тестів MW+Bonferroni
   як post-hoc; для χ² — через категоризацію з парними χ²+Bonferroni).

Примітка щодо Word і шрифтів:
- Простий текст з буфера обміну НЕ може гарантувати збереження шрифту у Word.
- Але вікно звіту відображає Times New Roman, і якщо у Word увімкнено “Зберігати вихідне форматування”
  — шрифт зазвичай не “стрибає”. (Для 100% гарантії потрібен RTF/Docx-експорт.)
"""

import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import math
import numpy as np
from itertools import combinations
from collections import defaultdict

from scipy.stats import shapiro, t, f as f_dist
from scipy.stats import mannwhitneyu, kruskal, chi2_contingency
from scipy.stats import studentized_range  # для Тьюкі/Данкана


# -------------------------
# DPI awareness (Windows) — робить шрифти чіткішими
# -------------------------
try:
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # SYSTEM_DPI_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
except Exception:
    pass


ALPHA = 0.05

# Вужчі колонки вводу (ще трохи вужче, ніж було)
COL_W = 10  # було 12


# -------------------------
# Допоміжні функції
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


def median_iqr(arr):
    """Повертає (median, q1, q3) або (nan,nan,nan)"""
    if arr is None or len(arr) == 0:
        return (np.nan, np.nan, np.nan)
    a = np.array(arr, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return (np.nan, np.nan, np.nan)
    med = float(np.median(a))
    q1 = float(np.quantile(a, 0.25))
    q3 = float(np.quantile(a, 0.75))
    return (med, q1, q3)


def make_box_table(headers, rows, aligns=None):
    """
    Таблиця з “комірками” на box-drawing символах (без +---+).
    aligns: 'l'/'r'/'c' на кожен стовпчик
    """
    headers = [str(h) for h in headers]
    rows = [[("" if v is None else str(v)) for v in row] for row in rows]
    ncol = len(headers)

    if aligns is None:
        aligns = ["l"] * ncol

    widths = [len(headers[i]) for i in range(ncol)]
    for row in rows:
        for i in range(ncol):
            widths[i] = max(widths[i], len(row[i]))

    def fmt_cell(text, w, a):
        if a == "r":
            return text.rjust(w)
        if a == "c":
            return text.center(w)
        return text.ljust(w)

    # рамки
    TL, TR, BL, BR = "┌", "┐", "└", "┘"
    H, V = "─", "│"
    TM, BM, LM, RM, MM = "┬", "┴", "├", "┤", "┼"

    top = TL + TM.join([H * (w + 2) for w in widths]) + TR
    mid = LM + MM.join([H * (w + 2) for w in widths]) + RM
    bot = BL + BM.join([H * (w + 2) for w in widths]) + BR

    out = [top]
    out.append(V + V.join([" " + fmt_cell(headers[i], widths[i], "c") + " " for i in range(ncol)]) + V)
    out.append(mid)

    for r in rows:
        out.append(V + V.join([" " + fmt_cell(r[i], widths[i], aligns[i]) + " " for i in range(ncol)]) + V)
        out.append(mid)

    out[-1] = bot
    return "\n".join(out)


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


def build_sig_from_threshold(levels, means_dict, thr):
    sig = {}
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            a, b = levels[i], levels[j]
            ma, mb = means_dict.get(a, np.nan), means_dict.get(b, np.nan)
            if math.isnan(ma) or math.isnan(mb):
                continue
            sig[(a, b)] = (abs(ma - mb) > thr)
    return sig


def sig_matrix_to_letters(levels_order, means_dict, sig_matrix):
    valid = [lvl for lvl in levels_order if not math.isnan(means_dict.get(lvl, np.nan))]
    if not valid:
        return {lvl: "" for lvl in levels_order}

    sorted_lvls = sorted(valid, key=lambda z: means_dict[z], reverse=True)
    letters = "abcdefghijklmnopqrstuvwxyz"
    groups = []

    def is_sig(x, y):
        if x == y:
            return False
        return bool(sig_matrix.get((x, y), False) or sig_matrix.get((y, x), False))

    for lvl in sorted_lvls:
        placed = False
        for grp in groups:
            ok = True
            for other in grp:
                if is_sig(lvl, other):
                    ok = False
                    break
            if ok:
                grp.append(lvl)
                placed = True
                break
        if not placed:
            groups.append([lvl])

    mapping = {}
    for i, grp in enumerate(groups):
        letter = letters[i] if i < len(letters) else f"g{i}"
        for lvl in grp:
            mapping[lvl] = letter

    for lvl in levels_order:
        mapping.setdefault(lvl, "")
    return mapping


# -------------------------
# Узагальнений ANOVA для 1–4 факторів
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
    for r in long:
        key = tuple(r.get(f) for f in factors)
        v = r.get("value", np.nan)
        m = cell_means.get(key, np.nan)
        if not math.isnan(v) and not math.isnan(m):
            SS_error += (v - m) ** 2

    levels_count = {f: len(levels_by_factor[f]) for f in factors}
    total_cells = 1
    for f in factors:
        total_cells *= levels_count[f]

    df_total = N - 1
    df_error = N - total_cells
    if df_error <= 0:
        df_error = max(1, df_error)

    MS_error = SS_error / df_error if df_error > 0 else np.nan

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

    SS = {}
    df = {}
    for rS in range(1, k + 1):
        for S in combinations(factors, rS):
            d = 1
            for f in S:
                d *= (levels_count[f] - 1)
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
            return {"A": "Фактор А", "B": "Фактор В", "C": "Фактор С", "D": "Фактор Д"}.get(subset[0], subset[0])
        letters = []
        for x in subset:
            letters.append({"A": "А", "B": "В", "C": "С", "D": "Д"}.get(x, x))
        return "Фактор " + "*".join(letters)

    table_rows = []
    p_values = {}
    effect_names = {}

    for rS in range(1, k + 1):
        for S in combinations(factors, rS):
            name = pretty_interaction(S)
            effect_names[S] = name
            SSv = SS.get(S, np.nan)
            dfv = df.get(S, np.nan)
            MSv = SSv / dfv if (dfv and not math.isnan(dfv) and dfv > 0) else np.nan
            Fv = MSv / MS_error if (not math.isnan(MS_error) and MS_error > 0 and not math.isnan(MSv)) else np.nan
            pv = 1 - f_dist.cdf(Fv, dfv, df_error) if (not math.isnan(Fv) and not math.isnan(dfv)) else np.nan
            p_values[name] = pv
            table_rows.append((name, SSv, dfv, MSv, Fv, pv))

    table_rows.append(("Залишок", SS_error, df_error, MS_error, None, None))
    table_rows.append(("Загальна", SS_total, df_total, None, None, None))

    eta2 = {}
    for rS in range(1, k + 1):
        for S in combinations(factors, rS):
            name = effect_names[S]
            eta2[name] = (SS[S] / SS_total) if SS_total > 0 else np.nan
    eta2["Залишок"] = (SS_error / SS_total) if SS_total > 0 else np.nan

    # НІР05 (довідково, використовуємо тільки якщо вибрано LSD)
    r_list = [n for n in cell_counts.values() if n > 0]
    r_mean = float(np.mean(r_list)) if r_list else np.nan
    tval = t.ppf(1 - ALPHA / 2, df_error) if df_error > 0 else np.nan

    NIR05 = {}
    pretty_map = {"A": "Фактор А", "B": "Фактор В", "C": "Фактор С", "D": "Фактор Д"}

    if k == 1:
        mean_n = np.mean([v[1] for v in stats[(factors[0],)].values()]) if stats.get((factors[0],)) else np.nan
        nir = tval * math.sqrt(2 * MS_error / mean_n) if not any(math.isnan(x) for x in [tval, MS_error, mean_n]) else np.nan
        NIR05[pretty_map["A"]] = nir
        NIR05["Загальна"] = nir
    else:
        for fct in factors:
            other_prod = 1
            for g in factors:
                if g != fct:
                    other_prod *= levels_count[g]
            divisor = other_prod * r_mean
            nir = tval * math.sqrt(2 * MS_error / divisor) if not any(math.isnan(x) for x in [tval, MS_error, r_mean]) else np.nan
            NIR05[pretty_map.get(fct, fct)] = nir
        nir_all = tval * math.sqrt(2 * MS_error / r_mean) if not any(math.isnan(x) for x in [tval, MS_error, r_mean]) else np.nan
        NIR05["Загальна"] = nir_all

    return {
        "table": table_rows,
        "cell_means": cell_means,
        "MS_error": MS_error,
        "df_error": df_error,
        "eta2": eta2,
        "NIR05": NIR05,
        "p_values": p_values
    }


# -------------------------
# Порівняння (параметричні)
# -------------------------
def posthoc_lsd(levels_order, means_dict, MS_error, df_error, n_eff, alpha=0.05):
    tcrit = t.ppf(1 - alpha / 2, df_error) if df_error > 0 else np.nan
    if any(math.isnan(x) for x in [tcrit, MS_error, n_eff]) or n_eff <= 0:
        thr = np.nan
    else:
        thr = tcrit * math.sqrt(2.0 * MS_error / n_eff)

    sig = build_sig_from_threshold(levels_order, means_dict, thr if not math.isnan(thr) else 0.0)
    letters = sig_matrix_to_letters(levels_order, means_dict, sig)
    return {"thr": thr, "sig": sig, "letters": letters}


def posthoc_bonferroni(levels_order, means_dict, ns_dict, MS_error, df_error, alpha=0.05):
    pairs = [(levels_order[i], levels_order[j]) for i in range(len(levels_order)) for j in range(i + 1, len(levels_order))]
    m = len(pairs) if pairs else 1
    alpha2 = alpha / m

    tcrit = t.ppf(1 - alpha2 / 2, df_error) if df_error > 0 else np.nan

    sig = {}
    for a, b in pairs:
        ma, mb = means_dict.get(a, np.nan), means_dict.get(b, np.nan)
        na, nb = ns_dict.get(a, 0), ns_dict.get(b, 0)
        if any([math.isnan(ma), math.isnan(mb)]) or na <= 0 or nb <= 0 or math.isnan(MS_error) or MS_error <= 0:
            continue
        se = math.sqrt(MS_error * (1.0 / na + 1.0 / nb))
        if se <= 0 or math.isnan(tcrit):
            continue
        diff = abs(ma - mb)
        sig[(a, b)] = (diff > tcrit * se)

    letters = sig_matrix_to_letters(levels_order, means_dict, sig)
    return {"sig": sig, "letters": letters, "m": m}


def posthoc_tukey(levels_order, means_dict, ns_dict, MS_error, df_error, alpha=0.05):
    n_eff = harmonic_mean([ns_dict.get(l, 0) for l in levels_order])
    if any(math.isnan(x) for x in [MS_error, df_error, n_eff]) or n_eff <= 0 or MS_error <= 0:
        qcrit = np.nan
        thr = np.nan
    else:
        qcrit = studentized_range.ppf(1 - alpha, len(levels_order), df_error)
        thr = qcrit * math.sqrt(MS_error / n_eff)

    sig = build_sig_from_threshold(levels_order, means_dict, thr if not math.isnan(thr) else 0.0)
    letters = sig_matrix_to_letters(levels_order, means_dict, sig)
    return {"thr": thr, "qcrit": qcrit, "sig": sig, "letters": letters, "n_eff": n_eff}


def posthoc_duncan(levels_order, means_dict, ns_dict, MS_error, df_error, alpha=0.05):
    valid = [lvl for lvl in levels_order if not math.isnan(means_dict.get(lvl, np.nan))]
    if not valid:
        return {"sig": {}, "letters": {lvl: "" for lvl in levels_order}, "n_eff": np.nan}

    ordered = sorted(valid, key=lambda z: means_dict[z], reverse=True)
    n_eff = harmonic_mean([ns_dict.get(l, 0) for l in ordered])
    sig = {}

    if any(math.isnan(x) for x in [MS_error, df_error, n_eff]) or n_eff <= 0 or MS_error <= 0:
        return {"sig": sig, "letters": {lvl: "" for lvl in levels_order}, "n_eff": n_eff}

    se_base = math.sqrt(MS_error / n_eff)

    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            a, b = ordered[i], ordered[j]
            r = (j - i) + 1
            qcrit = studentized_range.ppf(1 - alpha, r, df_error)
            SR = qcrit * se_base
            sig[(a, b)] = (abs(means_dict[a] - means_dict[b]) > SR)

    letters = sig_matrix_to_letters(levels_order, means_dict, sig)
    return {"sig": sig, "letters": letters, "n_eff": n_eff}


# -------------------------
# Непараметричні
# -------------------------
def pairwise_mannwhitney(levels_order, groups_dict, alpha=0.05):
    pairs = [(levels_order[i], levels_order[j]) for i in range(len(levels_order)) for j in range(i + 1, len(levels_order))]
    m = len(pairs) if pairs else 1
    sig = {}

    for a, b in pairs:
        xa = groups_dict.get(a, [])
        xb = groups_dict.get(b, [])
        if len(xa) == 0 or len(xb) == 0:
            continue
        try:
            _, p = mannwhitneyu(xa, xb, alternative="two-sided")
        except Exception:
            continue
        p_adj = min(1.0, float(p) * m)
        sig[(a, b)] = (p_adj < alpha)

    return {"sig": sig, "m": m}


def kruskal_wallis(levels_order, groups_dict):
    arrays = []
    used = []
    for lvl in levels_order:
        arr = groups_dict.get(lvl, [])
        if len(arr) > 0:
            arrays.append(arr)
            used.append(lvl)
    if len(arrays) < 2:
        return {"H": np.nan, "p": np.nan, "used_levels": used}
    try:
        H, p = kruskal(*arrays)
        return {"H": float(H), "p": float(p), "used_levels": used}
    except Exception:
        return {"H": np.nan, "p": np.nan, "used_levels": used}


def chi_square_on_binned(levels_order, groups_dict, bins=5, alpha=0.05):
    all_vals = []
    for lvl in levels_order:
        all_vals += list(groups_dict.get(lvl, []))
    all_vals = [v for v in all_vals if v is not None and not math.isnan(v)]
    if len(all_vals) < 5:
        return {"chi2": np.nan, "df": np.nan, "p": np.nan, "note": "Надто мало даних для χ²."}

    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(all_vals, qs))
    if len(edges) < 3:
        return {"chi2": np.nan, "df": np.nan, "p": np.nan, "note": "Не вдалося сформувати біни (багато однакових значень)."}

    def bin_index(x):
        for i in range(len(edges) - 2):
            if edges[i] <= x < edges[i + 1]:
                return i
        return len(edges) - 2

    used = [lvl for lvl in levels_order if len(groups_dict.get(lvl, [])) > 0]
    K = len(edges) - 1
    table = []
    for lvl in used:
        row = [0] * (K - 1)
        for x in groups_dict[lvl]:
            row[bin_index(x)] += 1
        table.append(row)

    try:
        chi2, p, df, _ = chi2_contingency(table)
        chi2, p, df = float(chi2), float(p), int(df)
    except Exception:
        return {"chi2": np.nan, "df": np.nan, "p": np.nan, "note": "Помилка обчислення χ²."}

    # парні χ² (Bonferroni) → для літер
    sig = {}
    pairs = [(used[i], used[j]) for i in range(len(used)) for j in range(i + 1, len(used))]
    m = len(pairs) if pairs else 1
    for a, b in pairs:
        rowA = [0] * (K - 1)
        rowB = [0] * (K - 1)
        for x in groups_dict[a]:
            rowA[bin_index(x)] += 1
        for x in groups_dict[b]:
            rowB[bin_index(x)] += 1
        try:
            _, pp, _, _ = chi2_contingency([rowA, rowB])
        except Exception:
            continue
        pp2 = min(1.0, float(pp) * m)
        sig[(a, b)] = (pp2 < alpha)

    note = f"χ² застосовано до числових даних через категоризацію (квантільні біни = {bins})."
    return {"chi2": chi2, "df": df, "p": p, "note": note, "sig": sig, "m": m}


# -------------------------
# GUI
# -------------------------
class SADTk:
    def __init__(self, root):
        self.root = root
        root.title("S.A.D. — Статистичний аналіз даних")
        root.geometry("1000x560")

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

        title = tk.Label(
            self.main_frame,
            text="S.A.D. — Статистичний аналіз даних",
            font=("Times New Roman", 20, "bold"),
            fg="#000000",
            bg="white"
        )
        title.pack(pady=18)

        btn_frame = tk.Frame(self.main_frame, bg="white")
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Однофакторний аналіз", width=22, height=2, command=lambda: self.open_table(1)).grid(row=0, column=0, padx=10, pady=8)
        tk.Button(btn_frame, text="Двофакторний аналіз", width=22, height=2, command=lambda: self.open_table(2)).grid(row=0, column=1, padx=10, pady=8)
        tk.Button(btn_frame, text="Трифакторний аналіз", width=22, height=2, command=lambda: self.open_table(3)).grid(row=1, column=0, padx=10, pady=8)
        tk.Button(btn_frame, text="Чотирифакторний аналіз", width=22, height=2, command=lambda: self.open_table(4)).grid(row=1, column=1, padx=10, pady=8)

        info = tk.Label(
            self.main_frame,
            text="Виберіть тип аналізу → внесіть дані (можна вставляти з Excel) → натисніть «Аналіз даних»",
            fg="#000000",
            bg="white"
        )
        info.pack(pady=10)

        self.table_win = None
        self.report_win = None

    # ---------- Діалог показника + одиниць ----------
    def ask_indicator_units(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Параметри звіту")
        dlg.resizable(False, False)

        frm = tk.Frame(dlg, padx=16, pady=16)
        frm.pack(fill=tk.BOTH, expand=True)

        tk.Label(frm, text="Назва показника:", fg="#000000").grid(row=0, column=0, sticky="w", pady=6)
        e_ind = tk.Entry(frm, width=40, fg="#000000")
        e_ind.grid(row=0, column=1, pady=6)

        tk.Label(frm, text="Одиниці виміру:", fg="#000000").grid(row=1, column=0, sticky="w", pady=6)
        e_units = tk.Entry(frm, width=40, fg="#000000")
        e_units.grid(row=1, column=1, pady=6)

        out = {"ok": False, "indicator": "", "units": ""}

        def on_ok():
            out["indicator"] = e_ind.get().strip()
            out["units"] = e_units.get().strip()
            if not out["indicator"] or not out["units"]:
                messagebox.showwarning("Помилка", "Заповніть назву показника та одиниці виміру.")
                return
            out["ok"] = True
            dlg.destroy()

        def on_cancel():
            dlg.destroy()

        btns = tk.Frame(frm)
        btns.grid(row=2, column=0, columnspan=2, pady=(12, 0))
        tk.Button(btns, text="OK", width=10, command=on_ok).pack(side=tk.LEFT, padx=6)
        tk.Button(btns, text="Скасувати", width=12, command=on_cancel).pack(side=tk.LEFT, padx=6)

        dlg.update_idletasks()
        center_window(dlg)
        e_ind.focus_set()
        dlg.bind("<Return>", lambda e: on_ok())
        dlg.grab_set()
        self.root.wait_window(dlg)
        return out

    # ---------- Вікно вибору методу ----------
    def choose_method_window(self, p_norm):
        dlg = tk.Toplevel(self.root)
        dlg.title("Вибір виду аналізу")
        dlg.resizable(False, False)

        frm = tk.Frame(dlg, padx=16, pady=14)
        frm.pack(fill=tk.BOTH, expand=True)

        normal = (p_norm is not None) and (not math.isnan(p_norm)) and (p_norm > 0.05)

        if normal:
            msg = ("Дані експерименту відповідають принципам нормального розподілу\n"
                   "за методом Шапіра-Вілка.")
            tk.Label(frm, text=msg, fg="#000000", justify="left").pack(anchor="w", pady=(0, 10))
            options = [
                ("НІР05", "lsd"),
                ("Тест Тьюкі", "tukey"),
                ("Тест Данкан", "duncan"),
                ("Тест Бонферроні", "bonferroni"),
            ]
        else:
            msg = ("Дані експерименту не відповідають принципам нормального розподілу\n"
                   "за методом Шапіра-Вілка.\n"
                   "Виберіть один з непараметричних типів аналізу.")
            tk.Label(frm, text=msg, fg="#c62828", justify="left").pack(anchor="w", pady=(0, 10))
            options = [
                ("Манна-Уітні", "mw"),
                ("Крускала-Уолліса", "kw"),
                ("Хі-квадрат", "chi2"),
            ]

        var = tk.StringVar(value=options[0][1])

        box = tk.Frame(frm)
        box.pack(anchor="w")

        for text, val in options:
            tk.Radiobutton(box, text=text, variable=var, value=val).pack(anchor="w", pady=2)

        out = {"ok": False, "method": None}

        def on_ok():
            out["ok"] = True
            out["method"] = var.get()
            dlg.destroy()

        def on_cancel():
            dlg.destroy()

        btns = tk.Frame(frm)
        btns.pack(fill=tk.X, pady=(12, 0))
        tk.Button(btns, text="OK", width=10, command=on_ok).pack(side=tk.LEFT, padx=6)
        tk.Button(btns, text="Скасувати", width=12, command=on_cancel).pack(side=tk.LEFT, padx=6)

        dlg.update_idletasks()
        center_window(dlg)
        dlg.bind("<Return>", lambda e: on_ok())
        dlg.grab_set()
        self.root.wait_window(dlg)
        return out

    # ---------- Вікно таблиці ----------
    def open_table(self, factors_count):
        if self.table_win and tk.Toplevel.winfo_exists(self.table_win):
            self.table_win.destroy()

        self.factors_count = factors_count
        self.factor_keys = ["A", "B", "C", "D"][:factors_count]

        self.table_win = tk.Toplevel(self.root)
        self.table_win.title(f"S.A.D. — {factors_count}-факторний аналіз")
        self.table_win.geometry("1280x720")

        self.repeat_count = 6
        factor_letters = ["А", "В", "С", "Д"]
        self.factor_names = [f"Фактор {factor_letters[i]}" for i in range(factors_count)]
        self.column_names = self.factor_names + [f"Повт.{i+1}" for i in range(self.repeat_count)]

        ctl = tk.Frame(self.table_win, padx=10, pady=8)
        ctl.pack(fill=tk.X)

        tk.Button(ctl, text="Додати рядок", command=self.add_row).pack(side=tk.LEFT, padx=4)
        tk.Button(ctl, text="Видалити рядок", command=self.delete_row).pack(side=tk.LEFT, padx=4)

        tk.Button(ctl, text="Додати стовпчик", command=self.add_column).pack(side=tk.LEFT, padx=10)
        tk.Button(ctl, text="Видалити стовпчик", command=self.delete_column).pack(side=tk.LEFT, padx=4)

        tk.Button(ctl, text="Вставити з буфера", command=self.paste_from_focus).pack(side=tk.LEFT, padx=18)
        tk.Button(ctl, text="Аналіз даних", bg="#c62828", fg="white", command=self.analyze).pack(side=tk.LEFT, padx=16)
        tk.Button(ctl, text="Про розробника", command=self.show_about).pack(side=tk.RIGHT, padx=4)

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
            "Про розробника",
            "S.A.D. — Статистичний аналіз даних\nРозробка: (вкажіть автора/організацію)\nВерсія: 1.0"
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
                long.append(rec)

        return long, rep_cols

    # ---------- Аналіз ----------
    def analyze(self):
        params = self.ask_indicator_units()
        if not params["ok"]:
            return

        indicator = params["indicator"]
        units = params["units"]

        long, used_rep_cols = self.collect_long()
        if len(long) == 0:
            messagebox.showwarning("Помилка", "Немає числових даних для аналізу.\nПеревірте повторності та значення.")
            return

        values = np.array([r["value"] for r in long], dtype=float)
        if len(values) < 3:
            messagebox.showinfo("Результат", "Надто мало даних для аналізу.")
            return

        levels_by_factor = {}
        for fct in self.factor_keys:
            levels_by_factor[fct] = first_seen_order([r.get(fct) for r in long])

        try:
            res = anova_n_way(long, self.factor_keys, levels_by_factor)
        except Exception as ex:
            messagebox.showerror("Помилка аналізу", str(ex))
            return

        cell_means = res.get("cell_means", {})
        residuals = []
        for r in long:
            key = tuple(r.get(f) for f in self.factor_keys)
            v = r.get("value", np.nan)
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

        # Групи по факторах
        factor_groups = {}
        factor_means = {}
        factor_ns = {}
        factor_meds = {}
        for fct in self.factor_keys:
            g = groups_by_keys(long, (fct,))
            g2 = {k[0]: v for k, v in g.items()}
            factor_groups[fct] = g2
            factor_means[fct] = {lvl: float(np.mean(arr)) if len(arr) else np.nan for lvl, arr in g2.items()}
            factor_ns[fct] = {lvl: len(arr) for lvl, arr in g2.items()}
            factor_meds[fct] = {lvl: median_iqr(arr) for lvl, arr in g2.items()}

        # Варіанти
        var_groups_raw = groups_by_keys(long, tuple(self.factor_keys))
        variant_order = first_seen_order([tuple(r.get(f) for f in self.factor_keys) for r in long])

        vstats = variant_mean_sd(long, self.factor_keys)
        v_means = {k: vstats[k][0] for k in vstats.keys()}
        v_ns = {k: vstats[k][2] for k in vstats.keys()}
        v_groups = {k: var_groups_raw.get(k, []) for k in variant_order}
        v_meds = {k: median_iqr(v_groups.get(k, [])) for k in variant_order}

        MS_error = res.get("MS_error", np.nan)
        df_error = res.get("df_error", np.nan)

        # Для літер
        letters_factor = {fct: {lvl: "" for lvl in levels_by_factor[fct]} for fct in self.factor_keys}
        letters_variants = {k: "" for k in variant_order}

        method_text = ""
        method_details_lines = []

        # -------------------------
        # ПАРАМЕТРИЧНІ
        # -------------------------
        if method in ("lsd", "tukey", "duncan", "bonferroni"):
            if method == "lsd":
                method_text = "Параметричний метод порівняння середніх: НІР05."
                # фактори
                for fct in self.factor_keys:
                    lvls = levels_by_factor[fct]
                    means = factor_means[fct]
                    ns = factor_ns[fct]
                    n_eff = harmonic_mean([ns.get(l, 0) for l in lvls])
                    out = posthoc_lsd(lvls, means, MS_error, df_error, n_eff, alpha=ALPHA)
                    letters_factor[fct] = out["letters"]
                # варіанти
                n_eff_variants = harmonic_mean([v_ns.get(k, 0) for k in variant_order])
                outv = posthoc_lsd(variant_order, v_means, MS_error, df_error, n_eff_variants, alpha=ALPHA)
                letters_variants = outv["letters"]
                if not math.isnan(outv.get("thr", np.nan)):
                    method_details_lines.append(f"НІР05 (критична різниця для варіантів): {outv['thr']:.4f}")

            elif method == "bonferroni":
                method_text = "Параметричний метод порівняння середніх: Бонферроні (парні t-тести з поправкою)."
                for fct in self.factor_keys:
                    lvls = levels_by_factor[fct]
                    means = factor_means[fct]
                    ns = factor_ns[fct]
                    out = posthoc_bonferroni(lvls, means, ns, MS_error, df_error, alpha=ALPHA)
                    letters_factor[fct] = out["letters"]
                    method_details_lines.append(f"Фактор { {'A':'А','B':'В','C':'С','D':'Д'}[fct] }: m={out['m']}")
                outv = posthoc_bonferroni(variant_order, v_means, v_ns, MS_error, df_error, alpha=ALPHA)
                letters_variants = outv["letters"]
                method_details_lines.append(f"Варіанти: m={outv['m']}")

            elif method == "tukey":
                method_text = "Параметричний метод порівняння середніх: тест Тьюкі."
                for fct in self.factor_keys:
                    lvls = levels_by_factor[fct]
                    means = factor_means[fct]
                    ns = factor_ns[fct]
                    out = posthoc_tukey(lvls, means, ns, MS_error, df_error, alpha=ALPHA)
                    letters_factor[fct] = out["letters"]
                outv = posthoc_tukey(variant_order, v_means, v_ns, MS_error, df_error, alpha=ALPHA)
                letters_variants = outv["letters"]
                if not math.isnan(outv.get("thr", np.nan)):
                    method_details_lines.append(f"Tukey HSD (критична різниця для варіантів): {outv['thr']:.4f}")

            elif method == "duncan":
                method_text = "Параметричний метод порівняння середніх: тест Дункана."
                for fct in self.factor_keys:
                    lvls = levels_by_factor[fct]
                    means = factor_means[fct]
                    ns = factor_ns[fct]
                    out = posthoc_duncan(lvls, means, ns, MS_error, df_error, alpha=ALPHA)
                    letters_factor[fct] = out["letters"]
                outv = posthoc_duncan(variant_order, v_means, v_ns, MS_error, df_error, alpha=ALPHA)
                letters_variants = outv["letters"]
                method_details_lines.append(f"DMRT: n_eff={outv.get('n_eff', np.nan):.3f}")

        # -------------------------
        # НЕПАРАМЕТРИЧНІ
        # -------------------------
        else:
            if method == "mw":
                method_text = "Непараметричний аналіз: Манна-Уітні (парні порівняння) з поправкою Бонферроні."
                # Літери по варіантах
                outv = pairwise_mannwhitney(variant_order, v_groups, alpha=ALPHA)
                letters_variants = sig_matrix_to_letters(variant_order, v_means, outv["sig"])
                method_details_lines.append(f"Варіанти: m={outv['m']} (MW+Bonferroni).")
                # Літери по факторах (також MW+Bonferroni)
                for fct in self.factor_keys:
                    lvls = levels_by_factor[fct]
                    out = pairwise_mannwhitney(lvls, factor_groups[fct], alpha=ALPHA)
                    letters_factor[fct] = sig_matrix_to_letters(lvls, factor_means[fct], out["sig"])
                    method_details_lines.append(f"Фактор { {'A':'А','B':'В','C':'С','D':'Д'}[fct] }: m={out['m']} (MW+Bonferroni).")

            elif method == "kw":
                method_text = "Непараметричний аналіз: Крускала-Уолліса + post-hoc Манна-Уітні (Bonferroni)."
                kw = kruskal_wallis(variant_order, v_groups)
                method_details_lines.append(f"Крускала-Уолліса (варіанти): H={fmt_num(kw['H'],3)}, p={fmt_num(kw['p'],4)}")
                outv = pairwise_mannwhitney(variant_order, v_groups, alpha=ALPHA)
                letters_variants = sig_matrix_to_letters(variant_order, v_means, outv["sig"])
                method_details_lines.append(f"Post-hoc: m={outv['m']} (MW+Bonferroni).")

                # Для факторів: теж KW + post-hoc MW
                for fct in self.factor_keys:
                    lvls = levels_by_factor[fct]
                    kwf = kruskal_wallis(lvls, factor_groups[fct])
                    method_details_lines.append(f"Крускала-Уолліса (фактор { {'A':'А','B':'В','C':'С','D':'Д'}[fct] }): H={fmt_num(kwf['H'],3)}, p={fmt_num(kwf['p'],4)}")
                    out = pairwise_mannwhitney(lvls, factor_groups[fct], alpha=ALPHA)
                    letters_factor[fct] = sig_matrix_to_letters(lvls, factor_means[fct], out["sig"])

            elif method == "chi2":
                method_text = "Непараметричний аналіз: Хі-квадрат (χ²) на категоризованих даних (частоти по бінів)."
                chi = chi_square_on_binned(variant_order, v_groups, bins=5, alpha=ALPHA)
                method_details_lines.append(f"χ² (варіанти): χ²={fmt_num(chi.get('chi2',np.nan),3)}, df={chi.get('df','')}, p={fmt_num(chi.get('p',np.nan),4)}")
                if chi.get("note"):
                    method_details_lines.append(chi["note"])
                if "sig" in chi:
                    letters_variants = sig_matrix_to_letters(variant_order, v_means, chi["sig"])
                    method_details_lines.append(f"Парні χ²: m={chi.get('m',0)} (Bonferroni).")

                # Для факторів: χ² на категоризованих даних + парні χ² для літер
                for fct in self.factor_keys:
                    lvls = levels_by_factor[fct]
                    chif = chi_square_on_binned(lvls, factor_groups[fct], bins=5, alpha=ALPHA)
                    method_details_lines.append(f"χ² (фактор { {'A':'А','B':'В','C':'С','D':'Д'}[fct] }): χ²={fmt_num(chif.get('chi2',np.nan),3)}, df={chif.get('df','')}, p={fmt_num(chif.get('p',np.nan),4)}")
                    if "sig" in chif:
                        letters_factor[fct] = sig_matrix_to_letters(lvls, factor_means[fct], chif["sig"])

        # -------------------------
        # ЗВІТ
        # -------------------------
        title_map = {
            1: "О Д Н О Ф А К Т О Р Н О Г О",
            2: "Д В О Ф А К Т О Р Н О Г О",
            3: "Т Р И Ф А К Т О Р Н О Г О",
            4: "Ч О Т И Р И Ф А К Т О Р Н О Г О",
        }

        report_lines = []
        report_lines.append(f"Р Е З У Л Ь Т А Т И   {title_map[self.factors_count]}   Д И С П Е Р С І Й Н О Г О   А Н А Л І З У")
        report_lines.append("")
        report_lines.append(f"Показник: {indicator}")
        report_lines.append(f"Одиниці виміру: {units}")
        report_lines.append("")
        report_lines.append(f"Кількість активних повторностей: {len(used_rep_cols)}")
        report_lines.append(f"Загальна кількість облікових значень: {len(long)}")
        report_lines.append("")

        if not math.isnan(W):
            report_lines.append(f"Перевірка нормальності залишків (Shapiro–Wilk): {normality_text(p_norm)} (p = {p_norm:.4f})")
        else:
            report_lines.append("Перевірка нормальності залишків (Shapiro–Wilk): н/д")

        report_lines.append("")
        report_lines.append(method_text)
        if method_details_lines:
            report_lines.append("Додатково:")
            for s in method_details_lines:
                report_lines.append(f"  • {s}")
        report_lines.append("")
        report_lines.append("Пояснення: різні літери біля значень означають істотну різницю на рівні p<0.05 (залежно від обраного тесту).")
        report_lines.append("")

        table_no = 1

        # Таблиця ANOVA
        report_lines.append(f"ТАБЛИЦЯ {table_no}. Дисперсійний аналіз")
        headers = ["Джерело варіації", "Сума квадратів", "Ступені свободи", "Середній квадрат", "F", "p", "Висновок"]
        rows = []
        for name, SSv, dfv, MSv, Fv, pv in res["table"]:
            if name in ("Залишок", "Загальна"):
                rows.append([name, fmt_num(SSv, 2), (str(int(dfv)) if dfv is not None and not math.isnan(dfv) else ""), fmt_num(MSv, 3), "", "", ""])
            else:
                mark = significance_mark(pv)
                concl = "істотна різниця" if mark else ""
                rows.append([name, fmt_num(SSv, 2), (str(int(dfv)) if dfv is not None and not math.isnan(dfv) else ""), fmt_num(MSv, 3),
                             fmt_num(Fv, 3), fmt_num(pv, 4), (concl + (" " + mark if mark else ""))])
        report_lines.append(make_box_table(headers, rows, aligns=["l", "r", "r", "r", "r", "r", "l"]))
        report_lines.append("")
        table_no += 1

        # Сила впливу
        report_lines.append("Сила впливу (%, частка від загальної суми квадратів):")
        eta = res.get("eta2", {})
        for name, *_ in res["table"]:
            if name in eta:
                report_lines.append(f"  • {name:<26} — {eta[name]*100:7.2f}%")
        report_lines.append("")

        # Таблиця НІР — ТІЛЬКИ якщо обрано НІР05
        if method == "lsd":
            report_lines.append(f"ТАБЛИЦЯ {table_no}. Значення НІР₀₅")
            headers = ["Елемент", "НІР₀₅"]
            rows = []
            for key in ["Фактор А", "Фактор В", "Фактор С", "Фактор Д", "Загальна"]:
                if key in res.get("NIR05", {}):
                    rows.append([key, fmt_num(res["NIR05"][key], 3)])
            report_lines.append(make_box_table(headers, rows, aligns=["l", "r"]))
            report_lines.append("")
            table_no += 1

        # Таблиці по факторах (у єдиному стилі)
        for fct in self.factor_keys:
            letter = {"A": "А", "B": "В", "C": "С", "D": "Д"}[fct]
            report_lines.append(f"ТАБЛИЦЯ {table_no}. Середні значення (фактор {letter})")
            lvls = levels_by_factor[fct]
            if method in ("mw", "kw", "chi2"):
                headers = [f"Рівень фактору {letter}", "n", "Медіана", "IQR (Q1–Q3)", "Літера"]
                rows = []
                for lvl in lvls:
                    arr = factor_groups[fct].get(lvl, [])
                    n = len(arr)
                    med, q1, q3 = factor_meds[fct].get(lvl, (np.nan, np.nan, np.nan))
                    rows.append([str(lvl), str(n), fmt_num(med, 2), f"{fmt_num(q1,2)}–{fmt_num(q3,2)}", letters_factor[fct].get(lvl, "")])
                report_lines.append(make_box_table(headers, rows, aligns=["l", "r", "r", "r", "c"]))
            else:
                headers = [f"Рівень фактору {letter}", "n", "Середнє", "Літера"]
                rows = []
                for lvl in lvls:
                    m = factor_means[fct].get(lvl, np.nan)
                    n = factor_ns[fct].get(lvl, 0)
                    rows.append([str(lvl), str(n), fmt_num(m, 2), letters_factor[fct].get(lvl, "")])
                report_lines.append(make_box_table(headers, rows, aligns=["l", "r", "r", "c"]))
            report_lines.append("")
            table_no += 1

        # Таблиця варіантів — “таблиця середніх значень” + медіана для непараметрики
        if method in ("mw", "kw", "chi2"):
            report_lines.append(f"ТАБЛИЦЯ {table_no}. Таблиця середніх значень (варіанти), Медіана та IQR, буквені позначення істотної різниці")
            headers = ["Варіант (рівні факторів)", "n", "Середнє", "SD", "Медіана", "IQR (Q1–Q3)", "Літера"]
            rows = []
            for key in variant_order:
                m, sd, n = vstats.get(key, (np.nan, np.nan, 0))
                med, q1, q3 = v_meds.get(key, (np.nan, np.nan, np.nan))
                name = " | ".join([str(x) for x in key])
                rows.append([name, str(n), fmt_num(m, 2), fmt_num(sd, 2), fmt_num(med, 2), f"{fmt_num(q1,2)}–{fmt_num(q3,2)}", letters_variants.get(key, "")])
            report_lines.append(make_box_table(headers, rows, aligns=["l", "r", "r", "r", "r", "r", "c"]))
        else:
            report_lines.append(f"ТАБЛИЦЯ {table_no}. Таблиця середніх значень (варіанти), SD та буквені позначення істотної різниці")
            headers = ["Варіант (рівні факторів)", "n", "Середнє", "SD", "Літера"]
            rows = []
            for key in variant_order:
                m, sd, n = vstats.get(key, (np.nan, np.nan, 0))
                name = " | ".join([str(x) for x in key])
                rows.append([name, str(n), fmt_num(m, 2), fmt_num(sd, 2), letters_variants.get(key, "")])
            report_lines.append(make_box_table(headers, rows, aligns=["l", "r", "r", "r", "c"]))

        self.show_report("\n".join(report_lines))

    # ---------- Вікно звіту ----------
    def show_report(self, text):
        if self.report_win and tk.Toplevel.winfo_exists(self.report_win):
            self.report_win.destroy()

        self.report_win = tk.Toplevel(self.root)
        self.report_win.title("Звіт (можна копіювати)")
        self.report_win.geometry("1180x760")

        top = tk.Frame(self.report_win, padx=8, pady=8)
        top.pack(fill=tk.X)

        def copy_all():
            self.report_win.clipboard_clear()
            self.report_win.clipboard_append(text)
            messagebox.showinfo("Готово", "Звіт скопійовано в буфер обміну.")

        tk.Button(top, text="Копіювати весь звіт", command=copy_all).pack(side=tk.LEFT, padx=4)

        txt = ScrolledText(self.report_win, width=120, height=40)
        txt.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        txt.insert("1.0", text)
        txt.configure(font=("Times New Roman", 14), fg="#000000")

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
# Run app
# -------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = SADTk(root)
    root.mainloop()
