# main.py
# -*- coding: utf-8 -*-
"""
S.A.D. — Статистичний аналіз даних (Tkinter версія)
Потрібно: Python 3.8+, numpy, scipy, matplotlib
"""

import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import math
from itertools import combinations
from collections import defaultdict, OrderedDict

import numpy as np
from scipy.stats import shapiro, t, f as f_dist

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# =========================
# Візуальні налаштування
# =========================
APP_NAME = "S.A.D."
UI_FONT = ("Segoe UI", 11)          # читабельніше на Windows
UI_FONT_BOLD = ("Segoe UI", 11, "bold")
TITLE_FONT = ("Segoe UI", 18, "bold")
REPORT_FONT = ("Times New Roman", 14)


# =========================
# Допоміжні функції
# =========================
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
        return float(str(x).replace(",", "."))
    except Exception:
        return np.nan


def pretty_factor_name(key: str) -> str:
    # A,B,C,D -> А, В, С, D
    mapping = {"A": "А", "B": "В", "C": "С", "D": "D"}
    return mapping.get(key, key)


def format_effect_label(effect_keys):
    return "×".join(pretty_factor_name(k) for k in effect_keys)


def order_unique_preserve(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def nir05_value(MS_error, df_error, n_eff):
    """
    Загальна НІР05: t(0.975;df)*sqrt(2*MS_error/n_eff)
    n_eff — ефективна кількість повторень/обліків на варіант (середнє по комірках).
    """
    if df_error <= 0 or n_eff is None or math.isnan(n_eff) or n_eff <= 0:
        return np.nan
    tval = t.ppf(0.975, df_error)
    if math.isnan(tval) or math.isnan(MS_error) or MS_error <= 0:
        return np.nan
    return float(tval * math.sqrt(2 * MS_error / n_eff))


def assign_letters_by_nir(means_in_order, nir05):
    """
    Буквені позначення істотності біля середніх значень.
    means_in_order: list[(level, mean)] у потрібному порядку (як у таблиці)
    nir05: загальна НІР05
    return: dict[level] -> letter
    """
    items = [(k, v) for k, v in means_in_order if v is not None and not math.isnan(v)]
    if not items:
        return {}

    # Якщо НІР05 відсутня — просто a,b,c...
    if nir05 is None or math.isnan(nir05) or nir05 <= 0:
        letters = {}
        for i, (lev, _) in enumerate(items):
            letters[lev] = chr(ord("a") + min(i, 25))
        return letters

    # Для літер краще працює сортування за середнім (але виводимо потім у табличному порядку)
    by_mean = sorted(items, key=lambda x: x[1], reverse=True)

    groups = []  # list of [levels]
    letters_map = {}
    for lev, mu in by_mean:
        placed = False
        for gi, group in enumerate(groups):
            ok = True
            for lev2 in group:
                mu2 = dict(by_mean)[lev2]
                if abs(mu - mu2) > nir05:
                    ok = False
                    break
            if ok:
                group.append(lev)
                letters_map[lev] = chr(ord("a") + min(gi, 25))
                placed = True
                break
        if not placed:
            groups.append([lev])
            letters_map[lev] = chr(ord("a") + min(len(groups) - 1, 25))

    return letters_map


# =========================
# n-way ANOVA (1..4 фактори) через ефекти
# =========================
def anova_n_way(long, factors):
    k = len(factors)
    if k < 1 or k > 4:
        raise ValueError("Підтримується 1–4 фактори.")

    # N та grand mean
    vals = [rec["value"] for rec in long if rec.get("value") is not None and not math.isnan(rec["value"])]
    values = np.array(vals, dtype=float)
    N = int(len(values))
    if N < 3:
        raise ValueError("Недостатньо даних для аналізу.")

    grand_mean = float(np.mean(values))

    # Рівні факторів у порядку появи
    levels = {}
    for fk in factors:
        seq = [rec[fk] for rec in long if fk in rec and rec.get("value") is not None and not math.isnan(rec["value"])]
        levels[fk] = order_unique_preserve(seq)

    full_keys = tuple(factors)

    def key_of(rec, keys):
        return tuple(rec[k] for k in keys) if keys else ()

    # Маргінальні суми/кількості для всіх підмножин
    sums = {(): 0.0}
    counts = {(): 0}
    sums[()] = float(np.nansum([rec["value"] for rec in long]))
    counts[()] = int(np.sum([0 if math.isnan(rec["value"]) else 1 for rec in long]))

    for r in range(1, k + 1):
        for subset in combinations(factors, r):
            sums[subset] = defaultdict(float)
            counts[subset] = defaultdict(int)

    for rec in long:
        v = rec.get("value", np.nan)
        if v is None or math.isnan(v):
            continue
        for r in range(1, k + 1):
            for subset in combinations(factors, r):
                kk = key_of(rec, subset)
                sums[subset][kk] += float(v)
                counts[subset][kk] += 1

    # Середні
    means = {(): {(): grand_mean}}
    for r in range(1, k + 1):
        for subset in combinations(factors, r):
            means[subset] = {}
            for kk, s in sums[subset].items():
                n = counts[subset][kk]
                means[subset][kk] = (s / n) if n > 0 else np.nan

    # Ефект (Möbius)
    def effect_value(subset, kk):
        subset = tuple(subset)
        if not subset:
            return 0.0
        total = 0.0
        m = len(subset)
        for r in range(0, m + 1):
            for T in combinations(subset, r):
                if len(T) == 0:
                    mu = grand_mean
                else:
                    idx = [subset.index(tk_) for tk_ in T]
                    kT = tuple(kk[i] for i in idx)
                    mu = means[T].get(kT, np.nan)
                if mu is None or math.isnan(mu):
                    continue
                sign = (-1) ** (m - r)
                total += sign * float(mu)
        return total

    # SS для кожного ефекту
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

    SS_total = float(np.sum((values - grand_mean) ** 2))

    # Коміркові середні для SS похибки
    cell_counts = defaultdict(int)
    cell_sums = defaultdict(float)
    for rec in long:
        v = rec.get("value", np.nan)
        if v is None or math.isnan(v):
            continue
        kk = key_of(rec, full_keys)
        cell_sums[kk] += float(v)
        cell_counts[kk] += 1

    cell_means = {}
    for kk, n in cell_counts.items():
        cell_means[kk] = cell_sums[kk] / n if n > 0 else np.nan

    SS_error = 0.0
    for rec in long:
        v = rec.get("value", np.nan)
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

    df_error = int(N - len(cell_counts))
    df_total = int(N - 1)

    MS_error = SS_error / df_error if df_error > 0 else np.nan

    # Таблиця: SS, df, MS, F, p, Fcrit
    table_rows = []
    for r in range(1, k + 1):
        for subset in combinations(factors, r):
            dfe = df[subset]
            sse = SS[subset]
            mse = sse / dfe if dfe > 0 else np.nan
            Fe = mse / MS_error if (not math.isnan(MS_error) and MS_error > 0 and not math.isnan(mse)) else np.nan
            pe = 1 - f_dist.cdf(Fe, dfe, df_error) if (not math.isnan(Fe) and dfe > 0 and df_error > 0) else np.nan
            Fcrit = f_dist.ppf(0.95, dfe, df_error) if (dfe > 0 and df_error > 0) else np.nan
            concl = "істотний" if (not math.isnan(pe) and pe < 0.05) else "неістотний"
            table_rows.append((subset, sse, dfe, mse, Fe, pe, Fcrit, concl))

    # Сила впливу (η²)
    eta2 = {subset: (SS[subset] / SS_total if SS_total > 0 else np.nan) for subset in SS}
    eta2_err = SS_error / SS_total if SS_total > 0 else np.nan

    # Середні по головних факторах (у порядку рівнів)
    means_main = {}
    for fk in factors:
        subset = (fk,)
        means_main[fk] = OrderedDict()
        for lev in levels[fk]:
            mu = means[subset].get((lev,), np.nan)
            means_main[fk][lev] = float(mu) if not math.isnan(mu) else np.nan

    # Ефективна повторність: середня кількість спостережень на повну комірку
    r_list = [n for n in cell_counts.values() if n > 0]
    r_mean = float(np.mean(r_list)) if len(r_list) else np.nan

    nir05 = nir05_value(MS_error, df_error, r_mean)

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
        "eta2_err": eta2_err,
        "nir05": nir05,
        "means_main": means_main,
        "cell_means": cell_means,
        "cell_counts": dict(cell_counts),
        "p_note": "* — p < 0.05; ** — p < 0.01",
    }


# =========================
# Діалог введення показника + одиниць одним вікном
# =========================
class MetaDialog(tk.Toplevel):
    def __init__(self, master, title="Параметри показника"):
        super().__init__(master)
        self.title(title)
        self.resizable(False, False)
        self.result = None

        self.configure(padx=14, pady=12)
        self.grab_set()

        lbl1 = ttk.Label(self, text="Назва показника:", font=UI_FONT_BOLD)
        lbl1.grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.e1 = ttk.Entry(self, width=42, font=UI_FONT)
        self.e1.grid(row=1, column=0, sticky="we", pady=(0, 10))

        lbl2 = ttk.Label(self, text="Одиниці виміру:", font=UI_FONT_BOLD)
        lbl2.grid(row=2, column=0, sticky="w", pady=(0, 6))
        self.e2 = ttk.Entry(self, width=42, font=UI_FONT)
        self.e2.grid(row=3, column=0, sticky="we", pady=(0, 12))

        btns = ttk.Frame(self)
        btns.grid(row=4, column=0, sticky="e")
        ttk.Button(btns, text="Скасувати", command=self._cancel).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(btns, text="OK", command=self._ok).pack(side=tk.RIGHT)

        self.columnconfigure(0, weight=1)

        self.bind("<Return>", lambda e: self._ok())
        self.bind("<Escape>", lambda e: self._cancel())

        self.e1.focus_set()

    def _ok(self):
        name = self.e1.get().strip()
        units = self.e2.get().strip()
        if not name:
            messagebox.showwarning("Помилка", "Вкажіть назву показника.")
            return
        if not units:
            messagebox.showwarning("Помилка", "Вкажіть одиниці виміру.")
            return
        self.result = (name, units)
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()


# =========================
# GUI
# =========================
class SADTk:
    def __init__(self, root):
        self.root = root
        self.root.title(f"{APP_NAME} — Статистичний аналіз даних")
        self.root.geometry("980x520")

        # трохи підвищимо масштаб — текст виглядає “чіткіше”
        try:
            self.root.tk.call("tk", "scaling", 1.10)
        except Exception:
            pass

        self._setup_style()

        self.main_frame = ttk.Frame(root, padding=16)
        self.main_frame.pack(expand=True, fill=tk.BOTH)

        title = ttk.Label(self.main_frame, text=f"{APP_NAME} — Статистичний аналіз даних", font=TITLE_FONT)
        title.pack(pady=(0, 12))

        cards = ttk.Frame(self.main_frame)
        cards.pack(pady=8)

        ttk.Button(cards, text="Однофакторний аналіз", width=22, command=lambda: self.open_table(1)).grid(row=0, column=0, padx=6, pady=6)
        ttk.Button(cards, text="Двофакторний аналіз", width=22, command=lambda: self.open_table(2)).grid(row=0, column=1, padx=6, pady=6)
        ttk.Button(cards, text="Трифакторний аналіз", width=22, command=lambda: self.open_table(3)).grid(row=0, column=2, padx=6, pady=6)
        ttk.Button(cards, text="4-факторний аналіз", width=22, command=lambda: self.open_table(4)).grid(row=0, column=3, padx=6, pady=6)

        hint = ttk.Label(
            self.main_frame,
            text="Виберіть тип аналізу → вставте/введіть дані в таблицю → натисніть «Аналіз даних».",
            font=UI_FONT,
        )
        hint.pack(pady=(14, 0))

        self.table_win = None
        self.figure_win = None

        self.indicator_name = ""
        self.units = ""

    def _setup_style(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        # базові кольори — без “сірого” тексту
        style.configure(".", font=UI_FONT)
        style.configure("TFrame", background="#ffffff")
        style.configure("TLabel", background="#ffffff", foreground="#111827")
        style.configure("TButton", padding=(12, 8))
        style.configure("Accent.TButton", font=UI_FONT_BOLD, foreground="#ffffff", background="#d32f2f")
        style.map("Accent.TButton", background=[("active", "#b71c1c")])

        style.configure("Header.TLabel", font=UI_FONT_BOLD, background="#f3f4f6", foreground="#111827")

    # -------------------
    # Вікно таблиці
    # -------------------
    def open_table(self, factors_count):
        if self.table_win and tk.Toplevel.winfo_exists(self.table_win):
            self.table_win.destroy()

        self.factors_count = factors_count
        self.factor_keys = ["A", "B", "C", "D"][:factors_count]

        self.table_win = tk.Toplevel(self.root)
        self.table_win.title(f"{APP_NAME} — {factors_count}-факторний аналіз")
        self.table_win.geometry("1250x720")
        self.table_win.configure(bg="#ffffff")

        outer = ttk.Frame(self.table_win, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        # панель керування
        ctl = ttk.Frame(outer)
        ctl.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(ctl, text="Додати рядок", command=self.add_row).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctl, text="Видалити рядок", command=self.delete_row).pack(side=tk.LEFT, padx=4)

        ttk.Button(ctl, text="Додати стовпчик", command=self.add_column).pack(side=tk.LEFT, padx=(16, 4))
        ttk.Button(ctl, text="Видалити стовпчик", command=self.delete_column).pack(side=tk.LEFT, padx=4)

        ttk.Button(ctl, text="Аналіз даних", style="Accent.TButton", command=self.ask_meta_and_analyze).pack(side=tk.LEFT, padx=(18, 6))
        ttk.Button(ctl, text="Про розробника", command=self.show_about).pack(side=tk.RIGHT, padx=4)

        # Підказка про вставку з Excel
        ttk.Label(
            outer,
            text="Порада: можна вставляти з Excel (Ctrl+V) — табличний блок (рядки/стовпчики) вставиться з поточної комірки.",
            font=UI_FONT,
        ).pack(anchor="w", pady=(0, 8))

        # область таблиці зі скролом
        table_area = ttk.Frame(outer)
        table_area.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(table_area, bg="#ffffff", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.vscroll = ttk.Scrollbar(table_area, orient="vertical", command=self.canvas.yview)
        self.vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=self.vscroll.set)

        self.inner = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # колонки: фактори + повтори (за замовчуванням 4)
        self.repeat_cols = 4
        self.rows = 12

        factor_labels = [f"Фактор {pretty_factor_name(fk)}" for fk in self.factor_keys]
        self.column_names = factor_labels + [f"Повт.{i+1}" for i in range(self.repeat_cols)]
        self.cols = len(self.column_names)

        # заголовки
        self.header_widgets = []
        for j, name in enumerate(self.column_names):
            lbl = ttk.Label(self.inner, text=name, style="Header.TLabel", anchor="center")
            lbl.grid(row=0, column=j, padx=1, pady=1, sticky="nsew", ipadx=6, ipady=6)
            self.header_widgets.append(lbl)
            self.inner.columnconfigure(j, weight=1)

        # клітинки
        self.entries = []
        for i in range(self.rows):
            row_entries = []
            for j in range(self.cols):
                e = tk.Entry(self.inner, width=16, font=UI_FONT, relief=tk.SOLID, bd=1)
                e.grid(row=i + 1, column=j, padx=1, pady=1, sticky="nsew", ipady=6)
                self._bind_cell(e)
                row_entries.append(e)
            self.entries.append(row_entries)

        self.entries[0][0].focus_set()

    def _bind_cell(self, e: tk.Entry):
        e.bind("<Return>", self.on_enter)

        e.bind("<Up>", self.on_arrow)
        e.bind("<Down>", self.on_arrow)
        e.bind("<Left>", self.on_arrow)
        e.bind("<Right>", self.on_arrow)

        e.bind("<Control-v>", self.on_paste)
        e.bind("<Control-V>", self.on_paste)
        e.bind("<Control-c>", self.on_copy)
        e.bind("<Control-C>", self.on_copy)

    # -------------------
    # Рядки/стовпчики
    # -------------------
    def add_row(self):
        i = len(self.entries)
        row_entries = []
        for j in range(self.cols):
            e = tk.Entry(self.inner, width=16, font=UI_FONT, relief=tk.SOLID, bd=1)
            e.grid(row=i + 1, column=j, padx=1, pady=1, sticky="nsew", ipady=6)
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
        # додаємо ще один стовпчик повторення
        self.repeat_cols += 1
        self.cols += 1
        col_idx = self.cols - 1

        hdr = ttk.Label(self.inner, text=f"Повт.{self.repeat_cols}", style="Header.TLabel", anchor="center")
        hdr.grid(row=0, column=col_idx, padx=1, pady=1, sticky="nsew", ipadx=6, ipady=6)
        self.header_widgets.append(hdr)
        self.inner.columnconfigure(col_idx, weight=1)

        for i, row in enumerate(self.entries):
            e = tk.Entry(self.inner, width=16, font=UI_FONT, relief=tk.SOLID, bd=1)
            e.grid(row=i + 1, column=col_idx, padx=1, pady=1, sticky="nsew", ipady=6)
            self._bind_cell(e)
            row.append(e)

    def delete_column(self):
        # мінімум: фактори + 1 повтор
        min_cols = self.factors_count + 1
        if self.cols <= min_cols:
            messagebox.showinfo("Обмеження", "Мінімум має бути: фактори + 1 повторення.")
            return

        col_idx = self.cols - 1

        # заголовок
        hdr = self.header_widgets.pop()
        hdr.destroy()

        # клітинки
        for row in self.entries:
            cell = row.pop()
            cell.destroy()

        self.cols -= 1
        self.repeat_cols = self.cols - self.factors_count

    # -------------------
    # Буфер обміну (Excel)
    # -------------------
    def on_copy(self, event=None):
        w = event.widget
        try:
            sel = w.selection_get()
        except Exception:
            sel = w.get()
        try:
            self.table_win.clipboard_clear()
            self.table_win.clipboard_append(sel)
        except Exception:
            pass
        return "break"

    def on_paste(self, event=None):
        """
        Вставка табличного блоку з Excel (TSV) з поточної комірки.
        Працює саме для Ctrl+V у комірці.
        """
        widget = event.widget

        # максимально надійне отримання кліпборду
        data = ""
        try:
            data = self.table_win.clipboard_get()
        except Exception:
            try:
                data = self.root.clipboard_get()
            except Exception:
                return "break"

        if not data:
            return "break"

        pos = self.find_pos(widget)
        if not pos:
            return "break"
        r0, c0 = pos

        rows = data.splitlines()
        for ir, row_text in enumerate(rows):
            cols = row_text.split("\t")
            for jc, val in enumerate(cols):
                rr = r0 + ir
                cc = c0 + jc

                while rr >= len(self.entries):
                    self.add_row()

                if cc >= self.cols:
                    # не розширюємо таблицю автоматично по ширині, щоб не “ламати” структуру
                    continue

                self.entries[rr][cc].delete(0, tk.END)
                self.entries[rr][cc].insert(0, val)

        self.entries[r0][c0].focus_set()
        return "break"

    # -------------------
    # Навігація
    # -------------------
    def find_pos(self, widget):
        for i, row in enumerate(self.entries):
            for j, cell in enumerate(row):
                if cell is widget:
                    return (i, j)
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
    # Збір даних
    # -------------------
    def _active_repeat_columns(self):
        """
        Повертає список індексів колонок повторень, які мають хоч 1 числове значення.
        (порожні стовпчики НЕ враховуємо як повторність)
        """
        rep_cols = list(range(self.factors_count, self.cols))
        active = []
        for c in rep_cols:
            has_num = False
            for r in range(len(self.entries)):
                txt = self.entries[r][c].get().strip()
                if txt == "":
                    continue
                v = safe_float(txt)
                if not math.isnan(v):
                    has_num = True
                    break
            if has_num:
                active.append(c)
        return active

    def collect_long(self):
        long = []
        active_rep_cols = self._active_repeat_columns()

        for i, row in enumerate(self.entries):
            # факторні рівні
            factor_vals = []
            for k in range(self.factors_count):
                v = row[k].get().strip()
                if v == "":
                    # щоб порожні фактори не “злипались”
                    v = f"lev_row{i}_col{k}"
                factor_vals.append(v)

            for rep_col in active_rep_cols:
                txt = row[rep_col].get().strip()
                if txt == "":
                    continue
                val = safe_float(txt)
                if math.isnan(val):
                    continue

                rec = {"value": float(val)}
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
            f"Про {APP_NAME}",
            f"{APP_NAME} — Статистичний аналіз даних\n"
            "Версія: v1.0\n"
            "Розробник: Чаплоуцький А.М."
            "Уманський національний університет",
        )

    # -------------------
    # Аналіз
    # -------------------
    def ask_meta_and_analyze(self):
        dlg = MetaDialog(self.table_win, title="Введіть параметри показника")
        self.table_win.wait_window(dlg)

        if not dlg.result:
            return

        self.indicator_name, self.units = dlg.result
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

        # залишки: value - cell_mean
        residuals = []
        for rec in long:
            key = tuple(rec[fk] for fk in factors)
            mu = res["cell_means"].get(key, np.nan)
            if mu is None or math.isnan(mu):
                continue
            residuals.append(rec["value"] - mu)

        residuals = np.array(residuals, dtype=float)
        residuals = residuals[~np.isnan(residuals)]

        W = np.nan
        p_norm = np.nan
        if len(residuals) >= 3:
            try:
                W, p_norm = shapiro(residuals)
            except Exception:
                W, p_norm = np.nan, np.nan

        # активні повтори для звіту
        active_rep_cols = self._active_repeat_columns()
        repeats_used = len(active_rep_cols)

        # -------------------------
        # Формування звіту (вирівняно, легко копіюється)
        # -------------------------
        title_map = {
            1: "РЕЗУЛЬТАТИ ОДНОФАКТОРНОГО ДИСПЕРСІЙНОГО АНАЛІЗУ",
            2: "РЕЗУЛЬТАТИ ДВОФАКТОРНОГО ДИСПЕРСІЙНОГО АНАЛІЗУ",
            3: "РЕЗУЛЬТАТИ ТРИФАКТОРНОГО ДИСПЕРСІЙНОГО АНАЛІЗУ",
            4: "РЕЗУЛЬТАТИ ЧОТИРЬОХФАКТОРНОГО ДИСПЕРСІЙНОГО АНАЛІЗУ",
        }

        lines = []
        lines.append(title_map.get(self.factors_count, "РЕЗУЛЬТАТИ ДИСПЕРСІЙНОГО АНАЛІЗУ"))
        lines.append("")
        lines.append(f"Показник:\t{self.indicator_name}")
        lines.append(f"Одиниці виміру:\t{self.units}")
        lines.append("")

        for fk in factors:
            levs = res["levels"][fk]
            lines.append(f"Фактор {pretty_factor_name(fk)}:\t{', '.join(map(str, levs))}")
        lines.append(f"Кількість повторень (враховано, без порожніх стовпчиків):\t{repeats_used}")
        lines.append(f"Загальна кількість облікових значень (N):\t{res['N']}")
        lines.append("")

        if not (math.isnan(W) or math.isnan(p_norm)):
            lines.append(f"Нормальність залишків (Shapiro–Wilk):\tW = {W:.4f}\tp = {p_norm:.4f}\t→ " +
                         ("нормальний розподіл" if p_norm > 0.05 else "НЕ нормальний розподіл"))
        else:
            lines.append("Нормальність залишків (Shapiro–Wilk):\tнедостатньо даних/помилка")
        lines.append("")
        lines.append("Позначення істотності:\t" + res["p_note"])
        lines.append("")

        # таблиця ANOVA — рівні стовпчики через таби (Word дуже добре сприймає)
        lines.append("Таблиця дисперсійного аналізу (ANOVA):")
        lines.append("Джерело\tSS\tdf\tMS\tF\tFтабл(0.05)\tp\tВисновок")
        lines.append("-" * 110)

        for subset, SSv, dfv, MSv, Fv, pv, Fcrit, concl in res["table_rows"]:
            name = f"Фактор {pretty_factor_name(subset[0])}" if len(subset) == 1 else f"Взаємодія {format_effect_label(subset)}"
            mark = significance_mark(pv)
            ss_s = f"{SSv:.2f}"
            df_s = f"{int(dfv)}"
            ms_s = f"{MSv:.3f}" if not math.isnan(MSv) else ""
            f_s = f"{Fv:.3f}{mark}" if (Fv is not None and not math.isnan(Fv)) else ""
            fcrit_s = f"{Fcrit:.3f}" if (Fcrit is not None and not math.isnan(Fcrit)) else ""
            p_s = f"{pv:.4f}" if (pv is not None and not math.isnan(pv)) else ""
            lines.append(f"{name}\t{ss_s}\t{df_s}\t{ms_s}\t{f_s}\t{fcrit_s}\t{p_s}\t{concl}")

        # похибка + загальна
        lines.append(f"Похибка\t{res['SS_error']:.2f}\t{res['df_error']}\t{res['MS_error']:.3f}\t\t\t\t")
        lines.append(f"Загальна\t{res['SS_total']:.2f}\t{res['df_total']}\t\t\t\t\t")
        lines.append("-" * 110)
        lines.append("")

        # сила впливу
        lines.append("Сила впливу (η², % від SS загальної):")
        for r in range(1, self.factors_count + 1):
            for subset in combinations(factors, r):
                v = res["eta2"].get(subset, np.nan)
                if math.isnan(v):
                    continue
                label = pretty_factor_name(subset[0]) if len(subset) == 1 else format_effect_label(subset)
                lines.append(f"  {label}:\t{v * 100:.1f}%")
        if res["eta2_err"] is not None and not math.isnan(res["eta2_err"]):
            lines.append(f"  Похибка:\t{res['eta2_err'] * 100:.1f}%")
        lines.append("")

        # НІР05 — лише загальна
        nir = res["nir05"]
        if nir is None or math.isnan(nir):
            lines.append("НІР05:\t(не вдалося обчислити за наявними даними)")
        else:
            lines.append(f"НІР05:\t{nir:.3f}\t{self.units}")
        lines.append("")

        # середні по факторах — У ПОРЯДКУ ЯК У ТАБЛИЦІ (за рівнями)
        lines.append("Середні значення по факторах (у порядку рівнів, з буквами істотності):")
        lines.append("")

        self.means_for_plot = {}
        for fk in factors:
            levs = res["levels"][fk]
            means_od = res["means_main"][fk]  # OrderedDict
            means_in_order = [(lev, means_od.get(lev, np.nan)) for lev in levs]
            letters = assign_letters_by_nir(means_in_order, nir)

            lines.append(f"Фактор {pretty_factor_name(fk)}:")
            lines.append("Рівень\tСереднє\tПозначення")
            for lev, mu in means_in_order:
                if mu is None or math.isnan(mu):
                    continue
                lines.append(f"{lev}\t{mu:.3f}\t{letters.get(lev, '')}")
            lines.append("")

            self.means_for_plot[fk] = (means_in_order, letters)

        # -------------------------
        # Вікно звіту (копіюється у Word) + графік без розтягування
        # -------------------------
        win = tk.Toplevel(self.root)
        win.title(f"{APP_NAME} — Звіт")
        win.geometry("1200x780")
        win.configure(bg="#ffffff")

        top = ttk.Frame(win, padding=10)
        top.pack(fill=tk.BOTH, expand=True)

        txt = ScrolledText(top, width=140, height=28, wrap=tk.NONE)
        txt.pack(fill=tk.BOTH, expand=True)
        txt.configure(font=REPORT_FONT)
        txt.insert("1.0", "\n".join(lines))

        # щоб точно копіювалось: не вимикаємо state, додаємо Ctrl+A
        txt.bind("<Control-a>", lambda e: (txt.tag_add("sel", "1.0", "end-1c"), "break"))
        txt.bind("<Control-A>", lambda e: (txt.tag_add("sel", "1.0", "end-1c"), "break"))

        # Блок графіків (не на всю ширину, пропорційно)
        plot_wrap = ttk.Frame(win, padding=(10, 0, 10, 10))
        plot_wrap.pack(fill=tk.X)

        self.build_histograms(plot_wrap, factors)

    def build_histograms(self, parent, factors):
        if not getattr(self, "means_for_plot", None):
            return

        # Розмір фігури залежить від кількості факторів і кількості рівнів
        n = len(factors)
        max_bars = 1
        for fk in factors:
            items, _ = self.means_for_plot.get(fk, ([], {}))
            max_bars = max(max_bars, len([1 for lev, mu in items if mu is not None and not math.isnan(mu)]))

        # “пропорційний” розмір: ширина ~ кількості стовпчиків, висота ~ кількості факторів
        fig_w = min(max(5.5, 0.8 * max_bars + 3.0), 10.0)
        fig_h = min(max(2.2 * n, 2.2), 8.0)

        fig = plt.Figure(figsize=(fig_w, fig_h), dpi=100)
        axes = fig.subplots(n, 1) if n > 1 else [fig.add_subplot(1, 1, 1)]

        for ax, fk in zip(axes, factors):
            items, letters = self.means_for_plot.get(fk, ([], {}))
            items = [(lev, mu) for lev, mu in items if mu is not None and not math.isnan(mu)]
            if not items:
                ax.set_visible(False)
                continue

            labels = [str(lev) for lev, _ in items]
            vals = [mu for _, mu in items]
            x = np.arange(len(vals))

            ax.bar(x, vals)
            ax.set_title(f"Середні по фактору {pretty_factor_name(fk)}", fontsize=11, pad=8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=10)
            ax.set_ylabel(self.units if self.units else "", fontsize=10)

            for i, (lev, mu) in enumerate(items):
                ax.text(i, mu, f"{mu:.2f}\n{letters.get(lev, '')}", ha="center", va="bottom", fontsize=9)

        fig.tight_layout()

        # Вставляємо графік НЕ “розтягуючи” на всю ширину
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        w = canvas.get_tk_widget()
        w.pack(anchor="center")  # ключове: по центру, без fill=both/expand

# =========================
# Run app
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = SADTk(root)
    root.mainloop()
