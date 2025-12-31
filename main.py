# main.py
# -*- coding: utf-8 -*-

"""
S.A.D. — Статистичний аналіз даних (Tkinter)
Потрібно: Python 3.8+, numpy, scipy
Опційно (для експорту у Word .docx): python-docx
Встановлення:
  pip install numpy scipy
  pip install python-docx
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
import math
import numpy as np
from itertools import combinations
from scipy.stats import shapiro, t, f as f_dist

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


ALPHA = 0.05  # для НІР₀₅

# Ширина колонок у таблиці введення (шрифт НЕ чіпаємо)
COL_W = 12  # було 14; можна 11/12


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


def subset_stats(long, keys):
    """
    keys: tuple факторів, наприклад ('A','B')
    return: dict {levels_tuple: (mean, n)}
    """
    from collections import defaultdict
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


def variant_mean_sd(long, factor_keys):
    """
    Для кожного варіанта (комбінація рівнів факторів):
    key_tuple -> (mean, sd, n)
    sd: вибіркове (n-1), якщо n>=2; якщо n==1 -> 0
    """
    from collections import defaultdict
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


def compact_letters(levels_in_order, means_dict, nir):
    """
    Літерні групи для середніх за НІР₀₅.
    Якщо |m_i - m_j| <= НІР₀₅ → можуть мати однакову літеру.
    """
    if nir is None or (isinstance(nir, float) and math.isnan(nir)) or nir <= 0:
        return {lvl: "a" for lvl in levels_in_order}

    valid_levels = [lvl for lvl in levels_in_order if not math.isnan(means_dict.get(lvl, np.nan))]
    if not valid_levels:
        return {lvl: "" for lvl in levels_in_order}

    sorted_lvls = sorted(valid_levels, key=lambda z: means_dict[z], reverse=True)

    groups = []
    letters = "abcdefghijklmnopqrstuvwxyz"

    for lvl in sorted_lvls:
        placed = False
        for grp in groups:
            ok = True
            for other in grp:
                if abs(means_dict[lvl] - means_dict[other]) > nir:
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

    for lvl in levels_in_order:
        mapping.setdefault(lvl, "")
    return mapping


def make_ascii_table(headers, rows, aligns=None):
    """
    ASCII-таблиця з межами: стовпчики/рядки відмежовані.
    headers: list[str]
    rows: list[list[str]]
    aligns: list[str] 'l'/'r'/'c'
    """
    rows_s = [[("" if v is None else str(v)) for v in row] for row in rows]
    headers_s = [str(h) for h in headers]
    ncol = len(headers_s)

    if aligns is None:
        aligns = ["l"] * ncol

    widths = [len(headers_s[i]) for i in range(ncol)]
    for row in rows_s:
        for i in range(ncol):
            widths[i] = max(widths[i], len(row[i]))

    def fmt_cell(text, w, a):
        if a == "r":
            return text.rjust(w)
        if a == "c":
            return text.center(w)
        return text.ljust(w)

    sep = "+" + "+".join(["-" * (w + 2) for w in widths]) + "+"

    out = []
    out.append(sep)
    out.append("| " + " | ".join(fmt_cell(headers_s[i], widths[i], "c") for i in range(ncol)) + " |")
    out.append(sep)
    for row in rows_s:
        out.append("| " + " | ".join(fmt_cell(row[i], widths[i], aligns[i]) for i in range(ncol)) + " |")
        out.append(sep)
    return "\n".join(out)


# -------------------------
# Узагальнений ANOVA для 1–4 факторів (на основі маргінальних середніх)
# -------------------------
def anova_n_way(long, factors, levels_by_factor):
    """
    long: [{'A':..., 'B':..., 'C':..., 'D':..., 'value':...}, ...]
    factors: ['A'] ... ['A','B','C','D']
    levels_by_factor: dict {'A':[...], 'B':[...], ...} — порядок введення
    """
    N = len(long)
    values = np.array([r["value"] for r in long], dtype=float)
    grand_mean = np.nanmean(values)

    k = len(factors)
    if k < 1 or k > 4:
        raise ValueError("Підтримуються 1–4 фактори.")

    # Статистики для всіх підмножин факторів
    stats = {}  # subset(tuple of factors) -> dict(levels_tuple -> (mean,n))
    for r in range(1, k + 1):
        for comb in combinations(factors, r):
            stats[comb] = subset_stats(long, comb)

    # Повні комірки (для похибки/залишку)
    full = tuple(factors)
    cell_means = {kk: vv[0] for kk, vv in stats[full].items()}
    cell_counts = {kk: vv[1] for kk, vv in stats[full].items()}

    # SS_total
    SS_total = np.nansum((values - grand_mean) ** 2)

    # SS_error (Залишок): всередині комірок
    SS_error = 0.0
    for r in long:
        key = tuple(r.get(f) for f in factors)
        v = r.get("value", np.nan)
        m = cell_means.get(key, np.nan)
        if not math.isnan(v) and not math.isnan(m):
            SS_error += (v - m) ** 2

    # df
    levels_count = {f: len(levels_by_factor[f]) for f in factors}
    total_cells = 1
    for f in factors:
        total_cells *= levels_count[f]

    df_total = N - 1
    df_error = N - total_cells  # коректно при повному факторіальному плані

    if df_error <= 0:
        df_error = max(1, df_error)

    # MS_error
    MS_error = SS_error / df_error if df_error > 0 else np.nan

    # inclusion-exclusion для SS ефектів
    def delta_for_subset(S, levels_S):
        """
        delta_S = sum_{T ⊆ S} (-1)^(|S|-|T|) μ_T
        μ_∅ = grand_mean
        """
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

    # p-values
    table_rows = []
    p_values = {}
    effect_names = {}

    def pretty_interaction(subset):
        if len(subset) == 1:
            return {"A": "Фактор А", "B": "Фактор В", "C": "Фактор С", "D": "Фактор Д"}.get(subset[0], subset[0])
        letters = []
        for x in subset:
            if x == "A":
                letters.append("А")
            elif x == "B":
                letters.append("В")
            elif x == "C":
                letters.append("С")
            elif x == "D":
                letters.append("Д")
            else:
                letters.append(x)
        return "Фактор " + "*".join(letters)

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

    # сила впливу (%)
    eta2 = {}
    for rS in range(1, k + 1):
        for S in combinations(factors, rS):
            name = effect_names[S]
            eta2[name] = (SS[S] / SS_total) if SS_total > 0 else np.nan
    eta2["Залишок"] = (SS_error / SS_total) if SS_total > 0 else np.nan

    # НІР₀₅
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
# GUI
# -------------------------
class SADTk:
    def __init__(self, root):
        self.root = root
        root.title("S.A.D. — Статистичний аналіз даних")
        root.geometry("1000x560")

        # Загальні параметри стилю (НЕ змінюємо)
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
        self.last_report_text = ""  # для експорту DOCX

    # ---------- Діалог показника + одиниць (в одному вікні, по центру) ----------
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

        # Фактори + повторності
        self.repeat_count = 6  # порожні колонки автоматично ігноруються
        factor_letters = ["А", "В", "С", "Д"]
        self.factor_names = [f"Фактор {factor_letters[i]}" for i in range(factors_count)]
        self.column_names = self.factor_names + [f"Повт.{i+1}" for i in range(self.repeat_count)]

        # Верхня панель
        ctl = tk.Frame(self.table_win, padx=10, pady=8)
        ctl.pack(fill=tk.X)

        tk.Button(ctl, text="Додати рядок", command=self.add_row).pack(side=tk.LEFT, padx=4)
        tk.Button(ctl, text="Видалити рядок", command=self.delete_row).pack(side=tk.LEFT, padx=4)

        tk.Button(ctl, text="Додати стовпчик", command=self.add_column).pack(side=tk.LEFT, padx=10)
        tk.Button(ctl, text="Видалити стовпчик", command=self.delete_column).pack(side=tk.LEFT, padx=4)

        tk.Button(ctl, text="Вставити з буфера", command=self.paste_from_focus).pack(side=tk.LEFT, padx=18)

        tk.Button(ctl, text="Аналіз даних", bg="#c62828", fg="white", command=self.analyze).pack(side=tk.LEFT, padx=16)

        tk.Button(ctl, text="Про розробника", command=self.show_about).pack(side=tk.RIGHT, padx=4)

        # Canvas + Scroll
        self.canvas = tk.Canvas(self.table_win)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sb = ttk.Scrollbar(self.table_win, orient="vertical", command=self.canvas.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=sb.set)

        self.inner = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        # Таблиця (header + entries)
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

        # Ctrl+V має працювати гарантовано
        self.table_win.bind("<Control-v>", self.on_paste)
        self.table_win.bind("<Control-V>", self.on_paste)

    def show_about(self):
        messagebox.showinfo(
            "Про розробника",
            "S.A.D. — Статистичний аналіз даних\nРозробка: (вкажіть автора/організацію)\nВерсія: 1.0"
        )

    # ---------- Прив’язки для клітинки ----------
    def bind_cell(self, e: tk.Entry):
        e.bind("<Return>", self.on_enter)
        e.bind("<Up>", self.on_arrow)
        e.bind("<Down>", self.on_arrow)
        e.bind("<Left>", self.on_arrow)
        e.bind("<Right>", self.on_arrow)
        e.bind("<Control-v>", self.on_paste)
        e.bind("<Control-V>", self.on_paste)

    # ---------- Рядки/стовпчики ----------
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
        col_idx = self.cols - 1

        if self.header_labels:
            lbl = self.header_labels.pop()
            lbl.destroy()

        for row in self.entries:
            w = row.pop()
            w.destroy()

        self.cols -= 1
        self.inner.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    # ---------- Навігація ----------
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

    # ---------- Вставка з буфера (Excel) ----------
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

    # ---------- Які повторності реально активні ----------
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

    # ---------- Збір long-формату ----------
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

        # Порядок рівнів факторів — як введено (не сортуємо)
        levels_by_factor = {}
        for fct in self.factor_keys:
            levels_by_factor[fct] = first_seen_order([r.get(fct) for r in long])

        # ANOVA 1–4
        try:
            res = anova_n_way(long, self.factor_keys, levels_by_factor)
        except Exception as ex:
            messagebox.showerror("Помилка аналізу", str(ex))
            return

        # Residuals для Shapiro
        cell_means = res.get("cell_means", {})
        residuals = []
        for r in long:
            key = tuple(r.get(fct) for fct in self.factor_keys)
            v = r.get("value", np.nan)
            m = cell_means.get(key, np.nan)
            if not math.isnan(v) and not math.isnan(m):
                residuals.append(v - m)
        residuals = np.array(residuals, dtype=float)

        try:
            W, p_norm = shapiro(residuals) if len(residuals) >= 3 else (np.nan, np.nan)
        except Exception:
            W, p_norm = (np.nan, np.nan)

        # Маргінальні середні по факторах
        means_factor = {}
        for fct in self.factor_keys:
            stats_f = subset_stats(long, (fct,))
            means_factor[fct] = {k[0]: v[0] for k, v in stats_f.items()}

        NIR = res.get("NIR05", {})
        letters_factor = {}
        for fct in self.factor_keys:
            pretty = {"A": "Фактор А", "B": "Фактор В", "C": "Фактор С", "D": "Фактор Д"}[fct]
            nir_f = NIR.get(pretty, np.nan)
            letters_factor[fct] = compact_letters(levels_by_factor[fct], means_factor[fct], nir_f)

        # Таблиця варіантів
        vstats = variant_mean_sd(long, self.factor_keys)
        variant_order = first_seen_order([tuple(r.get(fct) for fct in self.factor_keys) for r in long])
        v_means = {k: vstats[k][0] for k in vstats.keys()}
        letters_variants = compact_letters(variant_order, v_means, NIR.get("Загальна", np.nan))

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
        report_lines.append("Пояснення позначень: ** — p<0.01; * — p<0.05; без позначки — p≥0.05.")
        report_lines.append("Істотна різниця між середніми: різні літери біля середніх значень.")
        report_lines.append("")

        # Таблиця 1 — ANOVA (ASCII)
        report_lines.append("ТАБЛИЦЯ 1. Дисперсійний аналіз")
        headers = ["Джерело варіації", "Сума квадратів", "Ступені свободи", "Середній квадрат", "F-критерій", "p-значення", "Висновок"]
        rows = []
        for name, SSv, dfv, MSv, Fv, pv in res["table"]:
            if name in ("Залишок", "Загальна"):
                rows.append([
                    name,
                    fmt_num(SSv, 2),
                    (str(int(dfv)) if dfv is not None and not math.isnan(dfv) else ""),
                    fmt_num(MSv, 3),
                    "",
                    "",
                    ""
                ])
            else:
                mark = significance_mark(pv)
                concl = "істотна різниця" if mark else ""
                rows.append([
                    name,
                    fmt_num(SSv, 2),
                    (str(int(dfv)) if dfv is not None and not math.isnan(dfv) else ""),
                    fmt_num(MSv, 3),
                    fmt_num(Fv, 3),
                    fmt_num(pv, 4),
                    (concl + (" " + mark if mark else ""))
                ])
        report_lines.append(make_ascii_table(headers, rows, aligns=["l", "r", "r", "r", "r", "r", "l"]))
        report_lines.append("")

        # Сила впливу
        report_lines.append("Сила впливу (%, частка від загальної суми квадратів):")
        eta = res.get("eta2", {})
        for name, *_ in res["table"]:
            if name in eta:
                report_lines.append(f"  • {name:<26} — {eta[name]*100:7.2f}%")
        report_lines.append("")

        # Таблиця 2 — НІР₀₅ (ASCII)
        report_lines.append("ТАБЛИЦЯ 2. Значення НІР₀₅")
        headers = ["Елемент", "НІР₀₅"]
        rows = []
        for key in ["Фактор А", "Фактор В", "Фактор С", "Фактор Д", "Загальна"]:
            if key in res.get("NIR05", {}):
                rows.append([key, fmt_num(res["NIR05"][key], 2)])
        report_lines.append(make_ascii_table(headers, rows, aligns=["l", "r"]))
        report_lines.append("")

        # Середні по факторах (порядок введення)
        if self.factors_count >= 1:
            report_lines.append("Середні по фактору А:")
            for lvl in levels_by_factor["A"]:
                m = means_factor["A"].get(lvl, np.nan)
                report_lines.append(f"  {lvl:<24}{fmt_num(m,2):>10}  {letters_factor['A'].get(lvl,'')}")
            report_lines.append("")
        if self.factors_count >= 2:
            report_lines.append("Середні по фактору В:")
            for lvl in levels_by_factor["B"]:
                m = means_factor["B"].get(lvl, np.nan)
                report_lines.append(f"  {lvl:<24}{fmt_num(m,2):>10}  {letters_factor['B'].get(lvl,'')}")
            report_lines.append("")
        if self.factors_count >= 3:
            report_lines.append("Середні по фактору С:")
            for lvl in levels_by_factor["C"]:
                m = means_factor["C"].get(lvl, np.nan)
                report_lines.append(f"  {lvl:<24}{fmt_num(m,2):>10}  {letters_factor['C'].get(lvl,'')}")
            report_lines.append("")
        if self.factors_count >= 4:
            report_lines.append("Середні по фактору Д:")
            for lvl in levels_by_factor["D"]:
                m = means_factor["D"].get(lvl, np.nan)
                report_lines.append(f"  {lvl:<24}{fmt_num(m,2):>10}  {letters_factor['D'].get(lvl,'')}")
            report_lines.append("")

        # Таблиця 3 — перейменована і “нормальна” (ASCII)
        report_lines.append("ТАБЛИЦЯ 3. Таблиця середніх значень (варіанти досліду), SD та буквені позначення істотної різниці")
        headers = ["Варіант (рівні факторів)", "n", "Середнє", "SD", "Літера"]
        rows = []
        for key in variant_order:
            m, sd, n = vstats.get(key, (np.nan, np.nan, 0))
            name = " | ".join([str(x) for x in key])
            rows.append([name, str(n), fmt_num(m, 2), fmt_num(sd, 2), letters_variants.get(key, "")])
        report_lines.append(make_ascii_table(headers, rows, aligns=["l", "r", "r", "r", "c"]))

        report_text = "\n".join(report_lines)
        self.last_report_text = report_text
        self.show_report(report_text)

    # ---------- Вікно звіту (копіювання працює) ----------
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

        def export_docx():
            # Гарантоване збереження Times New Roman у Word: через .docx
            try:
                from docx import Document
                from docx.shared import Pt
            except Exception:
                messagebox.showerror(
                    "Немає бібліотеки",
                    "Для експорту у Word встановіть python-docx:\n\npip install python-docx"
                )
                return

            path = filedialog.asksaveasfilename(
                defaultextension=".docx",
                filetypes=[("Word document", "*.docx")],
                title="Зберегти звіт як DOCX"
            )
            if not path:
                return

            try:
                doc = Document()
                style = doc.styles["Normal"]
                style.font.name = "Times New Roman"
                style.font.size = Pt(14)

                for line in text.splitlines():
                    doc.add_paragraph(line)

                doc.save(path)
                messagebox.showinfo("Готово", f"DOCX збережено:\n{path}")
            except Exception as ex:
                messagebox.showerror("Помилка експорту", str(ex))

        tk.Button(top, text="Копіювати весь звіт", command=copy_all).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Експорт у DOCX (Word)", command=export_docx).pack(side=tk.LEFT, padx=6)

        txt = ScrolledText(self.report_win, width=120, height=40)
        txt.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        txt.insert("1.0", text)
        txt.configure(font=("Times New Roman", 14), fg="#000000")

        # Ctrl+C: копіює виділене, а якщо нема — весь текст
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
