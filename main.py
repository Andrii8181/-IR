# main.py
# -*- coding: utf-8 -*-
"""
SAD — Статистичний аналіз даних (Tkinter версія)
Потрібно: Python 3.8+, numpy, scipy, matplotlib
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from tkinter.scrolledtext import ScrolledText
import math
from itertools import combinations
from collections import defaultdict

import numpy as np
from scipy.stats import shapiro, t, f as f_dist

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# -------------------------
# Допоміжні статистичні функції
# -------------------------
def significance_mark(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def pretty_factor_name(key: str) -> str:
    # A,B,C,D -> А,В,С,D? (в укр. традиції: А, В, С, D — але у вас було А, В, С.
    # Для 4-го дамо "D".
    mapping = {"A": "А", "B": "В", "C": "С", "D": "D"}
    return mapping.get(key, key)


def format_effect_label(effect_keys):
    # ('A','B') -> "А×В"
    return "×".join(pretty_factor_name(k) for k in effect_keys)


def cld_letters(means_dict, LSD):
    """
    Просте (жадібне) присвоєння букв істотності за LSD.
    means_dict: {level: mean}
    return: {level: 'a'/'b'/...}
    """
    items = [(k, v) for k, v in means_dict.items() if v is not None and not math.isnan(v)]
    # Від більшого до меншого
    items.sort(key=lambda x: x[1], reverse=True)

    letters = {}
    groups = []  # list of (letter, [levels in group])

    if LSD is None or (isinstance(LSD, float) and math.isnan(LSD)) or LSD <= 0:
        # без LSD — просто a,b,c по порядку
        for i, (lev, _) in enumerate(items):
            letters[lev] = chr(ord("a") + min(i, 25))
        return letters

    for lev, mean_val in items:
        placed = False
        for gi, (letter, levs) in enumerate(groups):
            ok = True
            for lev2 in levs:
                if abs(mean_val - means_dict[lev2]) > LSD:
                    ok = False
                    break
            if ok:
                levs.append(lev)
                letters[lev] = letter
                placed = True
                break
        if not placed:
            letter = chr(ord("a") + min(len(groups), 25))
            groups.append((letter, [lev]))
            letters[lev] = letter

    return letters


# -------------------------
# Узагальнений n-way ANOVA (1..4 фактори) через ефекти (Möbius / inclusion–exclusion)
# long: list of dicts {'A':..,'B':..,'C':..,'D':..,'value': float}
# -------------------------
def anova_n_way(long, factors):
    """
    Підтримує 1-,2-,3-,4-факторний план.
    Обмеження: розрахунки відповідають класичному повному факторному ANOVA
    і найкраще працюють у збалансованих/майже збалансованих даних.
    """
    k = len(factors)
    if k < 1 or k > 4:
        raise ValueError("Supported number of factors: 1..4")

    # базові величини
    values = np.array([rec["value"] for rec in long], dtype=float)
    values = values[~np.isnan(values)]
    N = int(len(values))
    if N < 3:
        raise ValueError("Недостатньо даних")

    grand_mean = float(np.mean(values))

    # рівні факторів
    levels = {}
    for fk in factors:
        levels[fk] = sorted({rec[fk] for rec in long if fk in rec})

    # повні комірки (усі фактори)
    full_keys = tuple(factors)

    def key_of(rec, keys):
        if len(keys) == 0:
            return ()
        return tuple(rec[k] for k in keys)

    # накопичення сум/кількостей для всіх підмножин факторів
    sums = {(): 0.0}
    counts = {(): 0}
    sums[()] = float(np.nansum([rec["value"] for rec in long]))
    counts[()] = int(np.sum([0 if math.isnan(rec["value"]) else 1 for rec in long]))

    # 1) маргінальні суми/кількості для всіх підмножин факторів
    for r in range(1, k + 1):
        for subset in combinations(factors, r):
            sums[subset] = defaultdict(float)
            counts[subset] = defaultdict(int)

    for rec in long:
        v = rec["value"]
        if v is None or math.isnan(v):
            continue
        for r in range(1, k + 1):
            for subset in combinations(factors, r):
                kk = key_of(rec, subset)
                sums[subset][kk] += float(v)
                counts[subset][kk] += 1

    # 2) середні для підмножин
    means = {(): {(): grand_mean}}
    for r in range(1, k + 1):
        for subset in combinations(factors, r):
            means[subset] = {}
            for kk, s in sums[subset].items():
                n = counts[subset][kk]
                means[subset][kk] = (s / n) if n > 0 else np.nan

    # 3) ефект для конкретного subset-ключа через Möbius inversion:
    # effect(S, x_S) = Σ_{T⊆S} (-1)^{|S|-|T|} mean(T, x_T)
    def effect_value(subset, kk):
        subset = tuple(subset)
        if len(subset) == 0:
            return 0.0
        # зіставлення subset -> kk
        # будування проєкцій на підпідмножини:
        total = 0.0
        m = len(subset)
        for r in range(0, m + 1):
            for T in combinations(subset, r):
                if len(T) == 0:
                    mu = grand_mean
                else:
                    # взяти компоненти kk, що відповідають T
                    idx = [subset.index(tk) for tk in T]
                    kT = tuple(kk[i] for i in idx)
                    mu = means[T].get(kT, np.nan)
                if mu is None or math.isnan(mu):
                    # якщо немає комірки — пропустимо (неідеально для пропусків)
                    continue
                sign = (-1) ** (m - r)
                total += sign * float(mu)
        return total

    # 4) SS для кожного ефекту subset
    SS = {}
    for r in range(1, k + 1):
        for subset in combinations(factors, r):
            ss = 0.0
            for kk, n in counts[subset].items():
                if n <= 0:
                    continue
                ev = effect_value(subset, kk)
                ss += n * (ev ** 2)
            SS[subset] = float(ss)

    # SS_total та SS_error
    SS_total = float(np.sum((values - grand_mean) ** 2))

    # SS_error: within FULL cells
    # full cell mean
    cell_means = {}
    cell_counts = defaultdict(int)
    cell_sums = defaultdict(float)
    for rec in long:
        v = rec["value"]
        if v is None or math.isnan(v):
            continue
        kk = key_of(rec, full_keys)
        cell_sums[kk] += float(v)
        cell_counts[kk] += 1
    for kk, n in cell_counts.items():
        cell_means[kk] = cell_sums[kk] / n if n > 0 else np.nan

    SS_error = 0.0
    for rec in long:
        v = rec["value"]
        if v is None or math.isnan(v):
            continue
        kk = key_of(rec, full_keys)
        mu = cell_means.get(kk, np.nan)
        if mu is None or math.isnan(mu):
            continue
        SS_error += (float(v) - float(mu)) ** 2
    SS_error = float(SS_error)

    # df
    L = {fk: len(levels[fk]) for fk in factors}
    df = {}
    for r in range(1, k + 1):
        for subset in combinations(factors, r):
            d = 1
            for fk in subset:
                d *= max(L[fk] - 1, 0)
            df[subset] = int(d)

    # df_error = N - number_of_nonempty_full_cells
    df_error = int(N - len(cell_counts))
    df_total = int(N - 1)

    MS_error = SS_error / df_error if df_error > 0 else np.nan

    # таблиця
    table_rows = []
    # порядок: main -> 2-way -> 3-way -> 4-way
    for r in range(1, k + 1):
        for subset in combinations(factors, r):
            dfe = df[subset]
            sse = SS[subset]
            mse = sse / dfe if dfe > 0 else np.nan
            Fe = mse / MS_error if (MS_error is not None and not math.isnan(MS_error) and MS_error > 0 and not math.isnan(mse)) else np.nan
            pe = 1 - f_dist.cdf(Fe, dfe, df_error) if (Fe is not None and not math.isnan(Fe) and dfe > 0 and df_error > 0) else np.nan
            Fcrit = f_dist.ppf(0.95, dfe, df_error) if (dfe > 0 and df_error > 0) else np.nan
            concl = "істотний" if (pe is not None and not math.isnan(pe) and pe < 0.05) else "неістотний"
            table_rows.append((subset, sse, dfe, mse, Fe, pe, Fcrit, concl))

    table_rows.append((("ERROR",), SS_error, df_error, MS_error, None, None, None, ""))
    table_rows.append((("TOTAL",), SS_total, df_total, None, None, None, None, ""))

    # eta2 (частка SS_total)
    eta2 = {}
    for subset in SS:
        eta2[subset] = SS[subset] / SS_total if SS_total > 0 else np.nan

    eta2_res = SS_error / SS_total if SS_total > 0 else np.nan

    # LSD для головних факторів і для комірок
    LSD = {}

    tval = t.ppf(0.975, df_error) if df_error > 0 else np.nan

    # середній n на рівень фактору (маргінально)
    for fk in factors:
        subset = (fk,)
        n_per_level = []
        for kk, n in counts[subset].items():
            if n > 0:
                n_per_level.append(n)
        n_eff = float(np.mean(n_per_level)) if len(n_per_level) else np.nan
        LSD[fk] = (tval * math.sqrt(2 * MS_error / n_eff)) if (not any(math.isnan(x) for x in [tval, MS_error, n_eff]) and n_eff > 0) else np.nan

    # LSD для повних комірок
    r_list = [n for n in cell_counts.values() if n > 0]
    r_mean = float(np.mean(r_list)) if len(r_list) else np.nan
    LSD["cell"] = (tval * math.sqrt(2 * MS_error / r_mean)) if (not any(math.isnan(x) for x in [tval, MS_error, r_mean]) and r_mean > 0) else np.nan

    # маргінальні середні головних факторів
    means_main = {}
    for fk in factors:
        subset = (fk,)
        means_main[fk] = {}
        for kk, mu in means[subset].items():
            means_main[fk][kk[0]] = float(mu)

    return {
        "k": k,
        "factors": factors,
        "levels": levels,
        "N": N,
        "grand_mean": grand_mean,
        "SS_total": SS_total,
        "SS_error": SS_error,
        "df_error": df_error,
        "df_total": df_total,
        "MS_error": MS_error,
        "table_rows": table_rows,
        "eta2": eta2,
        "eta2_res": eta2_res,
        "LSD": LSD,
        "means_main": means_main,
        "cell_means": cell_means,
        "cell_counts": dict(cell_counts),
        "p_note": "* — p < 0.05; ** — p < 0.01",
    }


# -------------------------
# GUI
# -------------------------
class SADTk:
    def __init__(self, root):
        self.root = root
        root.title("SAD — Статистичний аналіз даних")
        root.geometry("900x450")

        self.main_frame = tk.Frame(root)
        self.main_frame.pack(expand=True, fill=tk.BOTH)

        title = tk.Label(
            self.main_frame,
            text="SAD — Статистичний аналіз даних",
            font=("Arial", 18, "bold"),
        )
        title.pack(pady=12)

        btn_frame = tk.Frame(self.main_frame)
        btn_frame.pack(pady=8)

        tk.Button(
            btn_frame,
            text="Однофакторний аналіз",
            width=20,
            height=2,
            command=lambda: self.open_table(1),
        ).grid(row=0, column=0, padx=6)

        tk.Button(
            btn_frame,
            text="Двофакторний аналіз",
            width=20,
            height=2,
            command=lambda: self.open_table(2),
        ).grid(row=0, column=1, padx=6)

        tk.Button(
            btn_frame,
            text="Трифакторний аналіз",
            width=20,
            height=2,
            command=lambda: self.open_table(3),
        ).grid(row=0, column=2, padx=6)

        tk.Button(
            btn_frame,
            text="4-факторний аналіз",
            width=20,
            height=2,
            command=lambda: self.open_table(4),
        ).grid(row=0, column=3, padx=6)

        info = tk.Label(
            self.main_frame,
            text="Виберіть тип аналізу → внесіть дані в таблицю → натисніть «Аналіз даних»",
            fg="gray",
        )
        info.pack(pady=10)

        self.table_win = None
        self.figure_win = None

        # експериментальні метадані (вводяться перед аналізом)
        self.indicator_name = ""
        self.units = ""

    def open_table(self, factors_count):
        if self.table_win and tk.Toplevel.winfo_exists(self.table_win):
            self.table_win.destroy()
        if self.figure_win and tk.Toplevel.winfo_exists(self.figure_win):
            self.figure_win.destroy()

        self.factors_count = factors_count
        self.table_win = tk.Toplevel(self.root)
        self.table_win.title(f"SAD — {factors_count}-факторний аналіз")
        self.table_win.geometry("1200x650")

        # фактори + 4 повторення за замовчуванням
        self.factor_keys = ["A", "B", "C", "D"][:factors_count]
        self.repeat_count = 4

        # назви колонок
        factor_labels = []
        for i, fk in enumerate(self.factor_keys):
            # A,B,C,D -> "Фактор А/В/С/D"
            factor_labels.append(f"Фактор {pretty_factor_name(fk)}")
        self.column_names = factor_labels + [f"Повт.{i+1}" for i in range(self.repeat_count)]

        # верхня панель керування
        ctl_frame = tk.Frame(self.table_win)
        ctl_frame.pack(fill=tk.X, padx=8, pady=6)

        # ліворуч: рядки + стовпчики
        tk.Button(ctl_frame, text="Додати рядок", command=self.add_row).pack(side=tk.LEFT, padx=4)
        tk.Button(ctl_frame, text="Видалити рядок", command=self.delete_row).pack(side=tk.LEFT, padx=4)

        tk.Button(ctl_frame, text="Додати стовпчик", command=self.add_column).pack(side=tk.LEFT, padx=14)
        tk.Button(ctl_frame, text="Видалити стовпчик", command=self.delete_column).pack(side=tk.LEFT, padx=4)

        tk.Button(
            ctl_frame,
            text="Аналіз даних",
            bg="#d32f2f",
            fg="white",
            command=self.ask_meta_and_analyze,
        ).pack(side=tk.LEFT, padx=20)

        # праворуч: про розробника
        tk.Button(ctl_frame, text="Про розробника", command=self.show_about).pack(side=tk.RIGHT, padx=4)

        # прокручувана область таблиці
        canvas = tk.Canvas(self.table_win)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(self.table_win, orient="vertical", command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.configure(yscrollcommand=scrollbar.set)

        self.inner = tk.Frame(canvas)
        canvas.create_window((0, 0), window=self.inner, anchor="nw")

        # початкові розміри
        self.rows = 10
        self.cols = len(self.column_names)

        # заголовки
        self.header_widgets = []
        for j, name in enumerate(self.column_names):
            lbl = tk.Label(self.inner, text=name, relief=tk.RIDGE, width=14, bg="#f0f0f0")
            lbl.grid(row=0, column=j, padx=1, pady=1, sticky="nsew")
            self.header_widgets.append(lbl)

        # клітинки
        self.entries = []
        for i in range(self.rows):
            row_entries = []
            for j in range(self.cols):
                e = tk.Entry(self.inner, width=14)
                e.grid(row=i + 1, column=j, padx=1, pady=1)
                self._bind_cell(e)
                row_entries.append(e)
            self.entries.append(row_entries)

        self.inner.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

        # стартовий фокус
        self.entries[0][0].focus_set()

    def _bind_cell(self, e: tk.Entry):
        # Enter -> вниз
        e.bind("<Return>", self.on_enter)

        # стрілки -> навігація
        e.bind("<Up>", self.on_arrow)
        e.bind("<Down>", self.on_arrow)
        e.bind("<Left>", self.on_arrow)
        e.bind("<Right>", self.on_arrow)

        # вставка/копіювання (Excel / Word / табличний текст)
        e.bind("<Control-v>", self.on_paste)
        e.bind("<Control-V>", self.on_paste)
        e.bind("<Control-c>", self.on_copy)
        e.bind("<Control-C>", self.on_copy)

    # -------------------
    # Table operations
    # -------------------
    def add_row(self):
        i = len(self.entries)
        row_entries = []
        for j in range(self.cols):
            e = tk.Entry(self.inner, width=14)
            e.grid(row=i + 1, column=j, padx=1, pady=1)
            self._bind_cell(e)
            row_entries.append(e)
        self.entries.append(row_entries)
        self.rows += 1

    def delete_row(self):
        if not self.entries:
            return
        last = self.entries.pop()
        for e in last:
            e.destroy()
        self.rows -= 1

    def add_column(self):
        """
        Додає додатковий стовпчик повторення (Повт.5, Повт.6, ...).
        """
        self.cols += 1
        col_idx = self.cols - 1

        # це повторення
        self.repeat_count = self.cols - self.factors_count
        header_text = f"Повт.{self.repeat_count}"

        lbl = tk.Label(self.inner, text=header_text, relief=tk.RIDGE, width=14, bg="#f0f0f0")
        lbl.grid(row=0, column=col_idx, padx=1, pady=1, sticky="nsew")
        self.header_widgets.append(lbl)

        for i, row in enumerate(self.entries):
            e = tk.Entry(self.inner, width=14)
            e.grid(row=i + 1, column=col_idx, padx=1, pady=1)
            self._bind_cell(e)
            row.append(e)

    def delete_column(self):
        """
        Видаляє останній стовпчик повторення.
        Мінімум: фактори + 1 повторення.
        """
        min_cols = self.factors_count + 1
        if self.cols <= min_cols:
            messagebox.showinfo("Обмеження", "Мінімум має бути: фактори + 1 повторення.")
            return

        col_idx = self.cols - 1

        # прибрати заголовок
        if self.header_widgets:
            hdr = self.header_widgets.pop()
            hdr.destroy()

        # прибрати комірки
        for row in self.entries:
            cell = row.pop()
            cell.destroy()

        self.cols -= 1
        self.repeat_count = self.cols - self.factors_count

    # -------------------
    # Clipboard helpers
    # -------------------
    def on_copy(self, event=None):
        w = event.widget
        try:
            sel = w.selection_get()
        except Exception:
            sel = w.get()
        self.table_win.clipboard_clear()
        self.table_win.clipboard_append(sel)
        return "break"

    def on_paste(self, event=None):
        """
        Вставляє табличний блок з буфера (Excel -> TSV), починаючи з поточної клітинки.
        """
        widget = event.widget
        try:
            data = widget.selection_get(selection="CLIPBOARD")
        except Exception:
            try:
                data = self.table_win.clipboard_get()
            except Exception:
                return "break"

        # excel -> rows by \n, cols by \t
        rows = data.splitlines()

        # позиція поточної клітинки
        pos = self.find_pos(widget)
        if not pos:
            return "break"
        r0, c0 = pos

        for ir, row_text in enumerate(rows):
            cols = row_text.split("\t")
            for jc, val in enumerate(cols):
                rr = r0 + ir
                cc = c0 + jc

                # при потребі додати рядки
                while rr >= len(self.entries):
                    self.add_row()

                # якщо влізли в колонки — вставляємо; інакше ігноруємо
                if cc >= self.cols:
                    continue

                self.entries[rr][cc].delete(0, tk.END)
                self.entries[rr][cc].insert(0, val)

        # залишаємо фокус на стартовій клітинці
        self.entries[r0][c0].focus_set()
        return "break"

    # -------------------
    # Navigation
    # -------------------
    def find_pos(self, widget):
        for i, row in enumerate(self.entries):
            for j, cell in enumerate(row):
                if cell is widget:
                    return (i, j)
        return None

    def on_enter(self, event=None):
        widget = event.widget
        pos = self.find_pos(widget)
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
        widget = event.widget
        pos = self.find_pos(widget)
        if not pos:
            return "break"
        i, j = pos

        if event.keysym == "Up":
            ni, nj = max(i - 1, 0), j
        elif event.keysym == "Down":
            ni, nj = min(i + 1, len(self.entries) - 1), j
        elif event.keysym == "Left":
            ni, nj = i, max(j - 1, 0)
        elif event.keysym == "Right":
            ni, nj = i, min(j + 1, self.cols - 1)
        else:
            return "break"

        self.entries[ni][nj].focus_set()
        self.entries[ni][nj].icursor(tk.END)
        return "break"

    # -------------------
    # Data collection
    # -------------------
    def collect_long(self):
        """
        Повертає long-формат: [{'A':..,'B':..,'C':..,'D':..,'value':..}, ...]
        """
        long = []
        for i, row in enumerate(self.entries):
            # рівні факторів
            factor_vals = []
            for k in range(self.factors_count):
                v = row[k].get().strip()
                if v == "":
                    v = f"lev_row{i}_col{k}"
                factor_vals.append(v)

            # повторення: всі колонки після факторів
            rep_start = self.factors_count
            rep_end = self.cols
            for rep_col in range(rep_start, rep_end):
                txt = row[rep_col].get().strip()
                if txt == "":
                    continue
                try:
                    val = float(txt.replace(",", "."))
                except Exception:
                    continue
                rec = {"value": val}
                if self.factors_count >= 1:
                    rec["A"] = factor_vals[0]
                if self.factors_count >= 2:
                    rec["B"] = factor_vals[1]
                if self.factors_count >= 3:
                    rec["C"] = factor_vals[2]
                if self.factors_count >= 4:
                    rec["D"] = factor_vals[3]
                long.append(rec)
        return long

    # -------------------
    # About
    # -------------------
    def show_about(self):
        messagebox.showinfo(
            "Про розробника",
            "SAD — Статистичний аналіз даних\n"
            "Версія: 1.0\n"
            "Розробка: (вкажіть ПІБ/організацію)\n"
            "УНУ / кафедра (за потреби)",
        )

    # -------------------
    # Analyze
    # -------------------
    def ask_meta_and_analyze(self):
        indicator = simpledialog.askstring("Показник", "Як називається показник?")
        if indicator is None:
            return
        units = simpledialog.askstring("Одиниці", "В яких одиницях виміру визначається?")
        if units is None:
            return
        self.indicator_name = indicator.strip()
        self.units = units.strip()
        self.analyze()

    def analyze(self):
        long = self.collect_long()
        if len(long) < 3:
            messagebox.showwarning("Помилка", "Недостатньо числових даних для аналізу.")
            return

        factors = ["A", "B", "C", "D"][: self.factors_count]

        try:
            res = anova_n_way(long, factors)
        except Exception as e:
            messagebox.showerror("Помилка аналізу", str(e))
            return

        # залишки: value - mean(full cell)
        residuals = []
        for rec in long:
            v = rec["value"]
            key = tuple(rec[fk] for fk in factors)
            mu = res["cell_means"].get(key, np.nan)
            if mu is None or math.isnan(mu):
                continue
            residuals.append(v - mu)

        residuals = np.array(residuals, dtype=float)
        residuals = residuals[~np.isnan(residuals)]

        W = np.nan
        p_norm = np.nan
        if len(residuals) >= 3:
            try:
                W, p_norm = shapiro(residuals)
            except Exception:
                W, p_norm = np.nan, np.nan

        # -------------------------
        # Формування звіту
        # -------------------------
        title_map = {
            1: "Р Е З У Л Ь Т А Т И   О Д Н О Ф А К Т О Р Н О Г О   Д И С П Е Р С І Й Н О Г О   А Н А Л І З У",
            2: "Р Е З У Л Ь Т А Т И   Д В О Ф А К Т О Р Н О Г О   Д И С П Е Р С І Й Н О Г О   А Н А Л І З У",
            3: "Р Е З У Л Ь Т А Т И   Т Р И Ф А К Т О Р Н О Г О   Д И С П Е Р С І Й Н О Г О   А Н А Л І З У",
            4: "Р Е З У Л Ь Т А Т И   Ч О Т И Р Ь О Х Ф А К Т О Р Н О Г О   Д И С П Е Р С І Й Н О Г О   А Н А Л І З У",
        }

        report = []
        report.append(title_map.get(self.factors_count, "Р Е З У Л Ь Т А Т И   Д И С П Е Р С І Й Н О Г О   А Н А Л І З У"))
        report.append("")
        report.append(f"Показник: {self.indicator_name or '—'}")
        report.append(f"Одиниці виміру: {self.units or '—'}")
        report.append("")

        # фактори + кількість рівнів
        for fk in factors:
            levs = res["levels"][fk]
            report.append(f"Фактор {pretty_factor_name(fk)}: {len(levs)} рівні(в): {', '.join(map(str, levs))}")
        report.append(f"Кількість повторень (поточна в таблиці): {self.repeat_count}")
        report.append(f"Загальна кількість облікових значень (N): {res['N']}")
        report.append("")

        if not (math.isnan(W) or math.isnan(p_norm)):
            report.append(f"Перевірка нормальності залишків (Shapiro–Wilk): W = {W:.4f}, p = {p_norm:.4f} → " +
                          ("нормальний" if p_norm > 0.05 else "НЕ нормальний"))
        else:
            report.append("Перевірка нормальності залишків (Shapiro–Wilk): недостатньо даних/помилка розрахунку")
        report.append("")
        report.append("Позначення істотності: " + res["p_note"])
        report.append("")

        # ANOVA-таблиця
        sep = "─" * 110
        report.append(sep)
        report.append(f"{'Джерело варіації':<30}{'SS':>12}{'df':>8}{'MS':>12}{'F':>12}{'Fтабл(0.05)':>14}{'p':>10}{'Висновок':>14}")
        report.append(sep)

        for subset, SSv, dfv, MSv, Fv, pv, Fcrit, concl in res["table_rows"]:
            if subset == ("ERROR",):
                name = "Випадкова помилка"
                mark = ""
            elif subset == ("TOTAL",):
                name = "Загальна"
                mark = ""
            else:
                name = (f"Фактор {pretty_factor_name(subset[0])}" if len(subset) == 1 else f"Взаємодія {format_effect_label(subset)}")
                mark = significance_mark(pv)

            ss_s = f"{SSv:10.2f}" if SSv is not None and not math.isnan(SSv) else ""
            df_s = f"{int(dfv):6d}" if dfv is not None and not math.isnan(dfv) else ""
            ms_s = f"{MSv:10.3f}" if MSv is not None and not math.isnan(MSv) else ""
            f_s = f"{Fv:10.3f}{mark}" if Fv is not None and not math.isnan(Fv) else ""
            fcrit_s = f"{Fcrit:10.3f}" if Fcrit is not None and not math.isnan(Fcrit) else ""
            p_s = f"{pv:8.4f}" if pv is not None and not math.isnan(pv) else ""
            concl_s = concl if concl else ""
            report.append(f"{name:<30}{ss_s:>12}{df_s:>8}{ms_s:>12}{f_s:>12}{fcrit_s:>14}{p_s:>10}{concl_s:>14}")

        report.append(sep)
        report.append("")
        report.append("Вилучення впливу (η², % від SS загальної):")

        # вивести основні ефекти у логічному порядку
        for r in range(1, self.factors_count + 1):
            for subset in combinations(factors, r):
                v = res["eta2"].get(subset, np.nan)
                if v is None or math.isnan(v):
                    continue
                label = (f"{pretty_factor_name(subset[0])}" if len(subset) == 1 else format_effect_label(subset))
                report.append(f"  • {label:<10} — {v * 100:5.1f}%")

        if res["eta2_res"] is not None and not math.isnan(res["eta2_res"]):
            report.append(f"  • {'Помилка':<10} — {res['eta2_res'] * 100:5.1f}%")
        report.append("")

        # LSD
        report.append("НІР₀.₅ (LSD):")
        for fk in factors:
            v = res["LSD"].get(fk, np.nan)
            if v is None or math.isnan(v):
                continue
            report.append(f"  • По фактору {pretty_factor_name(fk):<2} — {v:.3f} {self.units}")
        vcell = res["LSD"].get("cell", np.nan)
        if vcell is not None and not math.isnan(vcell):
            report.append(f"  • По комбінаціях (повні комірки) — {vcell:.3f} {self.units}")
        report.append("")

        # середні + букви
        report.append("Середні значення по факторах (з буквами істотності за LSD):")
        report.append("")
        self.means_for_plot = {}  # для гістограм

        for fk in factors:
            means_fk = res["means_main"][fk]
            LSD_fk = res["LSD"].get(fk, np.nan)
            letters = cld_letters(means_fk, LSD_fk)

            report.append(f"Фактор {pretty_factor_name(fk)}:")
            # сортуємо за середнім
            items = [(lev, means_fk[lev]) for lev in means_fk]
            items = [it for it in items if it[1] is not None and not math.isnan(it[1])]
            items.sort(key=lambda x: x[1], reverse=True)

            for lev, mu in items:
                report.append(f"  {str(lev):<20} {mu:8.3f} {self.units}   {letters.get(lev, '')}")

            report.append("")
            self.means_for_plot[fk] = (items, letters)

        # -------------------------
        # Вікно звіту + графіки
        # -------------------------
        win = tk.Toplevel(self.root)
        win.title("Звіт результатів")
        win.geometry("1200x750")

        # верх: текст
        txt = ScrolledText(win, width=130, height=26)
        txt.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        txt.configure(font=("Times New Roman", 14))
        txt.insert("1.0", "\n".join(report))
        txt.focus_set()

        # низ: гістограми
        plot_frame = tk.Frame(win)
        plot_frame.pack(fill=tk.BOTH, expand=False, padx=8, pady=(0, 8))

        self.build_histograms(plot_frame)

    def build_histograms(self, parent):
        """
        Автоматичні гістограми для середніх по головних факторах
        з підписами чисел та букв.
        """
        if not hasattr(self, "means_for_plot") or not self.means_for_plot:
            return

        factors = ["A", "B", "C", "D"][: self.factors_count]
        n = len(factors)

        fig = plt.Figure(figsize=(10.5, 2.2 * n), dpi=100)
        axes = fig.subplots(n, 1) if n > 1 else [fig.add_subplot(1, 1, 1)]

        for ax, fk in zip(axes, factors):
            items, letters = self.means_for_plot.get(fk, ([], {}))
            if not items:
                ax.set_visible(False)
                continue
            labels = [str(lev) for lev, _ in items]
            vals = [mu for _, mu in items]
            x = np.arange(len(vals))

            ax.bar(x, vals)
            ax.set_title(f"Середні по фактору {pretty_factor_name(fk)}")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15, ha="right")

            # підписи значень + букви
            for i, (lev, mu) in enumerate(items):
                letter = letters.get(lev, "")
                ax.text(i, mu, f"{mu:.2f}\n{letter}", ha="center", va="bottom")

            ax.set_ylabel(self.units if self.units else "")

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = SADTk(root)
    root.mainloop()
