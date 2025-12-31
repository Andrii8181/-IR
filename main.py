# main.py
# -*- coding: utf-8 -*-

"""
S.A.D. — Статистичний аналіз даних (Tkinter)

Потрібно: Python 3.8+, numpy, scipy
Встановлення: pip install numpy scipy

ОСТАННЯ ПРАВКА (ВАША):
1) Заголовки таблиць у звіті СТРОГО над своїми стовпчиками:
   - таб-стопи тепер рахуємо не через "M"*len, а через ФАКТИЧНУ ширину рядків у пікселях (font.measure).
   - кожна колонка має свою реальну ширину = max(ширина заголовка, ширина значень) + padding.
2) Таблиця парних порівнянь:
   - 1-й стовпчик має ширину найдовшого значення (автоматично через пункт 1).
3) Кнопки над таблицею вводу (Додати/Видалити/...) зроблені трохи меншими:
   - width зменшено, + менший padx.
4) "Про розробника" → "Розробник"
5) "Копіювати весь звіт" → "Копіювати звіт"

ШРИФТИ НЕ ЗМІНЮВАВ.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import tkinter.font as tkfont

import math
import numpy as np
from itertools import combinations
from collections import defaultdict

from scipy.stats import shapiro, t, f as f_dist
from scipy.stats import mannwhitneyu, kruskal
from scipy.stats import studentized_range  # Tukey/Duncan

ALPHA = 0.05
COL_W = 10  # ВУЖЧІ колонки вводу


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


def median_iqr(arr):
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

    # dedupe
    uniq = []
    for g in groups:
        if any(g == h for h in uniq):
            continue
        uniq.append(g)

    # remove strict subsets
    cleaned = []
    for g in uniq:
        if any((g < h) for h in uniq):
            continue
        cleaned.append(g)
    groups = cleaned

    alphabet = "abcdefghijklmnopqrstuvwxyz"
    letters_for_group = []
    for idx in range(len(groups)):
        if idx < len(alphabet):
            letters_for_group.append(alphabet[idx])
        else:
            letters_for_group.append(f"g{idx}")

    mapping = {lvl: [] for lvl in sorted_lvls}
    for gi, grp in enumerate(groups):
        letter = letters_for_group[gi]
        for lvl in grp:
            mapping[lvl].append(letter)

    out = {}
    for lvl in levels_order:
        if lvl in mapping:
            ls = mapping[lvl]
            ls_sorted = sorted(ls, key=lambda s: (0, s) if len(s) == 1 else (1, s))
            out[lvl] = "".join(ls_sorted)
        else:
            out[lvl] = ""
    return out


# -------------------------
# Report tables with pixel-accurate tab stops
# -------------------------
def build_table_block(headers, rows):
    lines = []
    dash = ["—" * max(3, len(h)) for h in headers]
    lines.append("\t".join(headers))
    lines.append("\t".join(dash))
    for r in rows:
        lines.append("\t".join("" if v is None else str(v) for v in r))
    return "\n".join(lines) + "\n"


def tabs_from_table_px(font_obj: tkfont.Font, headers, rows, padding_px=26, min_col_px=60):
    """
    TAB-стопи рахуються з РЕАЛЬНОЇ ширини (px) заголовків і значень.
    Це забезпечує, що заголовок буде строго над своїм стовпчиком.
    """
    ncol = len(headers)
    maxw = [0] * ncol

    # headers
    for j in range(ncol):
        maxw[j] = max(maxw[j], font_obj.measure(str(headers[j])))

    # rows
    for r in rows:
        for j in range(ncol):
            s = "" if r[j] is None else str(r[j])
            maxw[j] = max(maxw[j], font_obj.measure(s))

    widths_px = [max(min_col_px, w + padding_px) for w in maxw]

    tabs = []
    acc = 0
    for w in widths_px[:-1]:
        acc += w
        tabs.extend([acc, "center"])
    return tuple(tabs)


# -------------------------
# ANOVA 1–4
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
            table_rows.append((name, SSv, dfv, MSv, Fv, pv))

    table_rows.append(("Залишок", SS_error, df_error, MS_error, None, None))
    table_rows.append(("Загальна", SS_total, df_total, None, None, None))

    eta2 = {}
    for rS in range(1, k + 1):
        for S in combinations(factors, rS):
            name = effect_names[S]
            eta2[name] = (SS[S] / SS_total) if SS_total > 0 else np.nan
    eta2["Залишок"] = (SS_error / SS_total) if SS_total > 0 else np.nan

    # НІР₀₅ (довідково)
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
        "eta2": eta2,
        "NIR05": NIR05,
    }


# -------------------------
# LSD sig matrix
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


# -------------------------
# Pairwise variants (3 cols) + sig for CLD
# -------------------------
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
            tval = abs(ma - mb) / se
            p = 2 * (1 - t.cdf(tval, df_error))
            p_adj = min(1.0, float(p) * mtests)
            decision = (p_adj < alpha)
            sig[(a, b)] = decision
            rows.append([f"{a}  vs  {b}", fmt_num(p_adj, 4), "+" if decision else "-"])
        return rows, sig, "p_adj"

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
            rows.append([f"{a}  vs  {b}", fmt_num(p, 4), "+" if decision else "-"])
        return rows, sig, "p"

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
            rows.append([f"{a}  vs  {b}", fmt_num(p, 4), "+" if decision else "-"])
        return rows, sig, "p (набл.)"

    return [], {}, "p"


def pairwise_mw_bonf_short_variants_pm(v_names, groups_dict, alpha=0.05):
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
            _, p = mannwhitneyu(xa, xb, alternative="two-sided")
        except Exception:
            continue
        p = float(p)
        p_adj = min(1.0, p * mtests)
        decision = (p_adj < alpha)
        sig[(a, b)] = decision
        rows.append([f"{a}  vs  {b}", fmt_num(p_adj, 4), "+" if decision else "-"])
    return rows, sig, "p_adj"


# -------------------------
# GUI
# -------------------------
class SADTk:
    def __init__(self, root):
        self.root = root
        root.title("S.A.D. — Статистичний аналіз даних")
        root.geometry("1000x560")

        # ШРИФТИ НЕ ЧІПАЄМО
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
                ("НІР₀₅", "lsd"),
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

    def open_table(self, factors_count):
        if self.table_win and tk.Toplevel.winfo_exists(self.table_win):
            self.table_win.destroy()

        self.factors_count = factors_count
        self.factor_keys = ["A", "B", "C", "D"][:factors_count]

        self.table_win = tk.Toplevel(self.root)
        self.table_win.title(f"S.A.D. — {factors_count}-факторний аналіз")
        self.table_win.geometry("1280x720")

        self.repeat_count = 6
        self.factor_names = [f"Фактор {self.factor_keys[i]}" for i in range(factors_count)]
        self.column_names = self.factor_names + [f"Повт.{i+1}" for i in range(self.repeat_count)]

        ctl = tk.Frame(self.table_win, padx=8, pady=6)
        ctl.pack(fill=tk.X)

        # Кнопки трохи менші
        b_w = 14
        pad = 3

        tk.Button(ctl, text="Додати рядок", width=b_w, command=self.add_row).pack(side=tk.LEFT, padx=pad)
        tk.Button(ctl, text="Видалити рядок", width=b_w, command=self.delete_row).pack(side=tk.LEFT, padx=pad)

        tk.Button(ctl, text="Додати стовпчик", width=b_w, command=self.add_column).pack(side=tk.LEFT, padx=(10, pad))
        tk.Button(ctl, text="Видалити стовпчик", width=b_w, command=self.delete_column).pack(side=tk.LEFT, padx=pad)

        tk.Button(ctl, text="Вставити з буфера", width=b_w+2, command=self.paste_from_focus).pack(side=tk.LEFT, padx=(12, pad))
        tk.Button(ctl, text="Аналіз даних", width=b_w, bg="#c62828", fg="white", command=self.analyze).pack(side=tk.LEFT, padx=(12, pad))

        # "Розробник"
        tk.Button(ctl, text="Розробник", width=b_w, command=self.show_about).pack(side=tk.RIGHT, padx=pad)

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

        for row_i, row in enumerate(self.entries):
            e = tk.Entry(self.inner, width=COL_W, fg="#000000")
            e.grid(row=row_i + 1, column=col_idx, padx=2, pady=2)
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

        levels_by_factor = {f: first_seen_order([r.get(f) for r in long]) for f in self.factor_keys}

        try:
            res = anova_n_way(long, self.factor_keys, levels_by_factor)
        except Exception as ex:
            messagebox.showerror("Помилка аналізу", str(ex))
            return

        # residuals → Shapiro
        cell_means = res.get("cell_means", {})
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

        MS_error = res.get("MS_error", np.nan)
        df_error = res.get("df_error", np.nan)

        # факторні групи
        factor_groups = {}
        factor_means = {}
        factor_ns = {}
        factor_meds = {}
        for f in self.factor_keys:
            g = groups_by_keys(long, (f,))
            g2 = {k[0]: v for k, v in g.items()}
            factor_groups[f] = g2
            factor_means[f] = {lvl: float(np.mean(arr)) if len(arr) else np.nan for lvl, arr in g2.items()}
            factor_ns[f] = {lvl: len(arr) for lvl, arr in g2.items()}
            factor_meds[f] = {lvl: median_iqr(arr) for lvl, arr in g2.items()}

        letters_factor = {}
        if method == "lsd":
            for f in self.factor_keys:
                lvls = levels_by_factor[f]
                sig = lsd_sig_matrix(lvls, factor_means[f], factor_ns[f], MS_error, df_error, alpha=ALPHA)
                letters_factor[f] = cld_multi_letters(lvls, factor_means[f], sig)
        else:
            for f in self.factor_keys:
                letters_factor[f] = {lvl: "" for lvl in levels_by_factor[f]}

        # варіанти
        var_groups_raw = groups_by_keys(long, tuple(self.factor_keys))
        variant_order = first_seen_order([tuple(r.get(f) for f in self.factor_keys) for r in long])
        num_variants = len(variant_order)

        vstats = variant_mean_sd(long, self.factor_keys)
        v_means = {k: vstats[k][0] for k in vstats.keys()}
        v_sds = {k: vstats[k][1] for k in vstats.keys()}
        v_ns = {k: vstats[k][2] for k in vstats.keys()}
        v_groups = {k: var_groups_raw.get(k, []) for k in variant_order}
        v_meds = {k: median_iqr(v_groups.get(k, [])) for k in variant_order}

        v_names = [" | ".join(map(str, k)) for k in variant_order]
        means1 = {v_names[i]: v_means.get(variant_order[i], np.nan) for i in range(len(variant_order))}
        ns1 = {v_names[i]: v_ns.get(variant_order[i], 0) for i in range(len(variant_order))}
        groups1 = {v_names[i]: v_groups.get(variant_order[i], []) for i in range(len(variant_order))}

        letters_named = {name: "" for name in v_names}
        pairwise_rows = []
        p_col_name = "p"

        if method == "lsd":
            sigv = lsd_sig_matrix(v_names, means1, ns1, MS_error, df_error, alpha=ALPHA)
            letters_named = cld_multi_letters(v_names, means1, sigv)
            pairwise_rows = []

        elif method in ("tukey", "duncan", "bonferroni"):
            pairwise_rows, sig, p_col_name = pairwise_param_short_variants_pm(
                v_names, means1, ns1, MS_error, df_error, method, alpha=ALPHA
            )
            letters_named = cld_multi_letters(v_names, means1, sig)

        else:
            if method == "kw":
                arrays = [groups1[name] for name in v_names if len(groups1[name]) > 0]
                if len(arrays) >= 2:
                    try:
                        H, pkw = kruskal(*arrays)
                        kw_line = f"KW (варіанти): H={fmt_num(float(H),3)}, p={fmt_num(float(pkw),4)}."
                    except Exception:
                        kw_line = ""
                else:
                    kw_line = ""
            else:
                kw_line = ""

            pairwise_rows, sig, p_col_name = pairwise_mw_bonf_short_variants_pm(v_names, groups1, alpha=ALPHA)
            means_tmp = {name: float(np.mean(groups1[name])) if len(groups1[name]) else np.nan for name in v_names}
            letters_named = cld_multi_letters(v_names, means_tmp, sig)

        letters_variants = {variant_order[i]: letters_named.get(v_names[i], "") for i in range(len(variant_order))}

        method_text = {
            "lsd": "Порівняння середніх: НІР₀₅ (LSD).",
            "tukey": "Порівняння середніх: тест Тьюкі (Tukey HSD).",
            "duncan": "Порівняння середніх: тест Дункана (DMRT).",
            "bonferroni": "Порівняння середніх: Bonferroni (парні t-тести з поправкою).",
            "mw": "Непараметричний аналіз: Манна-Уітні (MW) + Bonferroni.",
            "kw": "Непараметричний аналіз: Крускала-Уолліса (KW) + post-hoc MW (Bonferroni).",
        }[method]
        if method == "kw" and 'kw_line' in locals() and kw_line:
            method_text += f"\n{kw_line}"

        # segments
        seg = []
        title_map = {1: "О Д Н О Ф А К Т О Р Н О Г О", 2: "Д В О Ф А К Т О Р Н О Г О", 3: "Т Р И Ф А К Т О Р Н О Г О", 4: "Ч О Т И Р И Ф А К Т О Р Н О Г О"}
        seg.append(("text", f"Р Е З У Л Ь Т А Т И   {title_map[self.factors_count]}   Д И С П Е Р С І Й Н О Г О   А Н А Л І З У\n\n"))
        seg.append(("text", f"Показник:\t{indicator}\nОдиниці виміру:\t{units}\n\n"))
        seg.append(("text", f"Кількість варіантів:\t{num_variants}\nКількість повторностей:\t{len(used_rep_cols)}\nЗагальна кількість облікових значень:\t{len(long)}\n\n"))
        if not math.isnan(W):
            seg.append(("text", "Перевірка нормальності залишків (Shapiro–Wilk):\t"
                           f"{normality_text(p_norm)}\t(W={fmt_num(float(W),4)}; p={fmt_num(float(p_norm),4)})\n\n"))
        else:
            seg.append(("text", "Перевірка нормальності залишків (Shapiro–Wilk):\tн/д\n\n"))

        seg.append(("text", method_text + "\n\n"))
        seg.append(("text", "Пояснення позначень істотності: ** — p<0.01; * — p<0.05.\n"))
        seg.append(("text", "У таблицях знак \"-\" означає p ≥ 0.05 (істотної різниці немає).\n"))
        seg.append(("text", "Істотна різниця (літери): різні літери означають істотну різницю (за обраним методом).\n\n"))

        # ANOVA table
        anova_rows = []
        for name, SSv, dfv, MSv, Fv, pv in res["table"]:
            if name in ("Залишок", "Загальна"):
                anova_rows.append([name, fmt_num(SSv, 2),
                                   str(int(dfv)) if dfv is not None and not math.isnan(dfv) else "",
                                   fmt_num(MSv, 3), "", "", ""])
            else:
                mark = significance_mark(pv)
                concl = f"істотна різниця {mark}" if mark else "-"
                anova_rows.append([name, fmt_num(SSv, 2),
                                   str(int(dfv)) if dfv is not None and not math.isnan(dfv) else "",
                                   fmt_num(MSv, 3), fmt_num(Fv, 3), fmt_num(pv, 4), concl])
        seg.append(("text", "ТАБЛИЦЯ 1. Дисперсійний аналіз (ANOVA)\n"))
        seg.append(("table", (["Джерело", "SS", "df", "MS", "F", "p", "Висновок"], anova_rows)))
        seg.append(("text", "\n"))

        # eta
        seg.append(("text", "Сила впливу (%, частка від загальної суми квадратів):\n"))
        eta = res.get("eta2", {})
        for name, *_ in res["table"]:
            if name in eta:
                seg.append(("text", f"  • {name}:\t{eta[name]*100:.2f}%\n"))
        seg.append(("text", "\n"))

        tno = 2
        if method == "lsd":
            nir_rows = []
            for key in [f"Фактор {f}" for f in self.factor_keys] + ["Загальна"]:
                if key in res.get("NIR05", {}):
                    nir_rows.append([key, fmt_num(res["NIR05"][key], 4)])
            seg.append(("text", "ТАБЛИЦЯ 2. Значення НІР₀₅ (довідково)\n"))
            seg.append(("table", (["Елемент", "НІР₀₅"], nir_rows)))
            seg.append(("text", "\n"))
            tno = 3

        for f in self.factor_keys:
            seg.append(("text", f"ТАБЛИЦЯ {tno}. Середнє по фактору {f}\n"))
            lvls = levels_by_factor[f]
            if method in ("mw", "kw"):
                rows = []
                for lvl in lvls:
                    arr = factor_groups[f].get(lvl, [])
                    n = len(arr)
                    med, q1, q3 = factor_meds[f].get(lvl, (np.nan, np.nan, np.nan))
                    letter = letters_factor[f].get(lvl, "")
                    rows.append([str(lvl), str(n), fmt_num(med, 3), f"{fmt_num(q1,3)}–{fmt_num(q3,3)}", letter if letter else "-"])
                seg.append(("table", ([f"Градація {f}", "n", "Медіана", "IQR(Q1–Q3)", "Істотна різниця"], rows)))
            else:
                rows = []
                for lvl in lvls:
                    m = factor_means[f].get(lvl, np.nan)
                    n = factor_ns[f].get(lvl, 0)
                    letter = letters_factor[f].get(lvl, "")
                    rows.append([str(lvl), str(n), fmt_num(m, 3), letter if letter else "-"])
                seg.append(("table", ([f"Градація {f}", "n", "Середнє", "Істотна різниця"], rows)))
            seg.append(("text", "\n"))
            tno += 1

        seg.append(("text", f"ТАБЛИЦЯ {tno}. Таблиця середніх значень (варіанти)\n"))
        if method in ("mw", "kw"):
            rows = []
            for k in variant_order:
                m = v_means.get(k, np.nan)
                sd = v_sds.get(k, np.nan)
                med, q1, q3 = v_meds.get(k, (np.nan, np.nan, np.nan))
                name = " | ".join(map(str, k))
                letter = letters_variants.get(k, "")
                rows.append([name, fmt_num(m, 3), fmt_num(sd, 3), fmt_num(med, 3), f"{fmt_num(q1,3)}–{fmt_num(q3,3)}", letter if letter else "-"])
            seg.append(("table", (["Варіант", "Середнє", "± SD", "Медіана", "IQR(Q1–Q3)", "Істотна різниця"], rows)))
        else:
            rows = []
            for k in variant_order:
                m = v_means.get(k, np.nan)
                sd = v_sds.get(k, np.nan)
                name = " | ".join(map(str, k))
                letter = letters_variants.get(k, "")
                rows.append([name, fmt_num(m, 3), fmt_num(sd, 3), letter if letter else "-"])
            seg.append(("table", (["Варіант", "Середнє", "± SD", "Істотна різниця"], rows)))
        seg.append(("text", "\n"))
        tno += 1

        if method in ("tukey", "duncan", "bonferroni", "mw", "kw") and pairwise_rows:
            seg.append(("text", f"ТАБЛИЦЯ {tno}. Результати парних порівнянь (варіанти)\n"))
            seg.append(("table", (["Комбінація варіантів", p_col_name, "Істотна різниця"], pairwise_rows)))

        self.show_report_segments(seg)

    # ---------- Report window with per-table pixel tabs ----------
    def show_report_segments(self, segments):
        if self.report_win and tk.Toplevel.winfo_exists(self.report_win):
            self.report_win.destroy()

        self.report_win = tk.Toplevel(self.root)
        self.report_win.title("Звіт (можна копіювати)")
        self.report_win.geometry("1180x760")

        top = tk.Frame(self.report_win, padx=8, pady=8)
        top.pack(fill=tk.X)

        txt = ScrolledText(self.report_win, width=120, height=40)
        txt.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        txt.configure(font=("Times New Roman", 14), fg="#000000")

        font_obj = tkfont.Font(font=("Times New Roman", 14))

        table_idx = 0
        for kind, payload in segments:
            if kind == "text":
                txt.insert("end", payload)
                continue

            headers, rows = payload
            tabs = tabs_from_table_px(font_obj, headers, rows, padding_px=26, min_col_px=60)
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
# Run app
# -------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = SADTk(root)
    root.mainloop()
