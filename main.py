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

# ЗБІЛЬШЕНІ шрифти по всій програмі
UI_FONT = ("Segoe UI", 14)
UI_FONT_BOLD = ("Segoe UI", 14, "bold")
TITLE_FONT = ("Segoe UI", 24, "bold")

# Звіт: Times New Roman 14 (як ти просив)
REPORT_FONT = ("Times New Roman", 14)

BLACK = "#000000"
WHITE = "#ffffff"
HEADER_BG = "#f3f4f6"


# =========================
# Windows clipboard: 100% вставка з Excel (без встановлень)
# =========================
def clipboard_text_windows():
    """
    Повертає Unicode-текст з буфера обміну Windows (CF_UNICODETEXT).
    Це найнадійніший спосіб прочитати скопійоване з Excel.
    """
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        CF_UNICODETEXT = 13

        if not user32.OpenClipboard(None):
            return ""

        try:
            h = user32.GetClipboardData(CF_UNICODETEXT)
            if not h:
                return ""
            p = kernel32.GlobalLock(h)
            if not p:
                return ""
            try:
                text = ctypes.wstring_at(p)
            finally:
                kernel32.GlobalUnlock(h)
            return text
        finally:
            user32.CloseClipboard()
    except Exception:
        return ""


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
    if df_error <= 0 or n_eff is None or math.isnan(n_eff) or n_eff <= 0:
        return np.nan
    tval = t.ppf(0.975, df_error)
    if math.isnan(tval) or math.isnan(MS_error) or MS_error <= 0:
        return np.nan
    return float(tval * math.sqrt(2 * MS_error / n_eff))


def nir05_for_main_factor(MS_error, df_error, r_mean, other_levels_product):
    if df_error <= 0 or r_mean is None or math.isnan(r_mean) or r_mean <= 0:
        return np.nan
    if other_levels_product <= 0:
        return np.nan
    tval = t.ppf(0.975, df_error)
    if math.isnan(tval) or math.isnan(MS_error) or MS_error <= 0:
        return np.nan
    denom = r_mean * other_levels_product
    return float(tval * math.sqrt(2 * MS_error / denom))


def assign_letters_by_nir(means_in_order, nir05):
    items = [(k, v) for k, v in means_in_order if v is not None and not math.isnan(v)]
    if not items:
        return {}

    if nir05 is None or math.isnan(nir05) or nir05 <= 0:
        letters = {}
        for i, (lev, _) in enumerate(items):
            letters[lev] = chr(ord("a") + min(i, 25))
        return letters

    by_mean = sorted(items, key=lambda x: x[1], reverse=True)
    mean_dict = dict(by_mean)

    groups = []
    letters_map = {}
    for lev, mu in by_mean:
        placed = False
        for gi, group in enumerate(groups):
            ok = True
            for lev2 in group:
                mu2 = mean_dict[lev2]
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

    vals = [rec["value"] for rec in long if rec.get("value") is not None and not math.isnan(rec["value"])]
    values = np.array(vals, dtype=float)
    N = int(len(values))
    if N < 3:
        raise ValueError("Недостатньо даних для аналізу.")

    grand_mean = float(np.mean(values))

    levels = {}
    for fk in factors:
        seq = [rec[fk] for rec in long if fk in rec and rec.get("value") is not None and not math.isnan(rec["value"])]
        levels[fk] = order_unique_preserve(seq)

    full_keys = tuple(factors)

    def key_of(rec, keys):
        return tuple(rec[k] for k in keys) if keys else ()

    sums = {(): 0.0}
    counts = {(): 0}
    sums[()] = float(np.nansum([rec["value"] for rec in long if rec.get("value") is not None and not math.isnan(rec["value"])]))
    counts[()] = int(np.sum([0 if (rec.get("value") is None or math.isnan(rec["value"])) else 1 for rec in long]))

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

    means = {(): {(): grand_mean}}
    for r in range(1, k + 1):
        for subset in combinations(factors, r):
            means[subset] = {}
            for kk, s in sums[subset].items():
                n = counts[subset][kk]
                means[subset][kk] = (s / n) if n > 0 else np.nan

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

    table_rows = []
    for r in range(1, k + 1):
        for subset in combinations(factors, r):
            dfe = df[subset]
            sse = SS[subset]
            mse = sse / dfe if dfe > 0 else np.nan
            Fe = mse / MS_error if (not math.isnan(MS_error) and MS_error > 0 and not math.isnan(mse)) else np.nan
            pe = 1 - f_dist.cdf(Fe, dfe, df_error) if (not math.isnan(Fe) and dfe > 0 and df_error > 0) else np.nan
            Fcrit = f_dist.ppf(0.95, dfe, df_error) if (dfe > 0 and df_error > 0) else np.nan
            concl = "істотна різниця" if (not math.isnan(pe) and pe < 0.05) else "різниця неістотна"
            table_rows.append((subset, sse, dfe, mse, Fe, pe, Fcrit, concl))

    eta2 = {subset: (SS[subset] / SS_total if SS_total > 0 else np.nan) for subset in SS}
    eta2_res = SS_error / SS_total if SS_total > 0 else np.nan

    means_main = {}
    for fk in factors:
        subset = (fk,)
        means_main[fk] = OrderedDict()
        for lev in levels[fk]:
            mu = means[subset].get((lev,), np.nan)
            means_main[fk][lev] = float(mu) if not math.isnan(mu) else np.nan

    r_list = [n for n in cell_counts.values() if n > 0]
    r_mean = float(np.mean(r_list)) if len(r_list) else np.nan

    nir_total = nir05_value(MS_error, df_error, r_mean)

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
        "nir_total": nir_total,
        "r_mean": r_mean,
        "means_main": means_main,
        "cell_means": cell_means,
        "cell_counts": dict(cell_counts),
        "p_note": "* — p < 0.05; ** — p < 0.01",
    }


# =========================
# Діалог: показник + одиниці в одному вікні
# =========================
class MetaDialog(tk.Toplevel):
    def __init__(self, master, title="Параметри показника"):
        super().__init__(master)
        self.title(title)
        self.resizable(False, False)
        self.result = None

        self.configure(padx=16, pady=14, bg=WHITE)
        self.grab_set()

        lbl1 = tk.Label(self, text="Назва показника:", font=UI_FONT_BOLD, fg=BLACK, bg=WHITE)
        lbl1.grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.e1 = tk.Entry(self, width=44, font=UI_FONT, fg=BLACK, bg=WHITE, relief=tk.SOLID, bd=1)
        self.e1.grid(row=1, column=0, sticky="we", pady=(0, 12))

        lbl2 = tk.Label(self, text="Одиниці виміру:", font=UI_FONT_BOLD, fg=BLACK, bg=WHITE)
        lbl2.grid(row=2, column=0, sticky="w", pady=(0, 6))
        self.e2 = tk.Entry(self, width=44, font=UI_FONT, fg=BLACK, bg=WHITE, relief=tk.SOLID, bd=1)
        self.e2.grid(row=3, column=0, sticky="we", pady=(0, 14))

        btns = tk.Frame(self, bg=WHITE)
        btns.grid(row=4, column=0, sticky="e")
        tk.Button(btns, text="Скасувати", font=UI_FONT, command=self._cancel).pack(side=tk.RIGHT, padx=(10, 0))
        tk.Button(btns, text="OK", font=UI_FONT_BOLD, command=self._ok).pack(side=tk.RIGHT)

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

        # DPI-awareness (Windows) — зменшує “розмиття”
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

        self.root.title(f"{APP_NAME} — Статистичний аналіз даних")
        self.root.geometry("1100x600")
        self.root.configure(bg=WHITE)

        try:
            self.root.tk.call("tk", "scaling", 1.0)
        except Exception:
            pass

        self._setup_style()

        self.main_frame = tk.Frame(root, bg=WHITE)
        self.main_frame.pack(expand=True, fill=tk.BOTH, padx=18, pady=16)

        title = tk.Label(self.main_frame, text=f"{APP_NAME} — Статистичний аналіз даних", font=TITLE_FONT, fg=BLACK, bg=WHITE)
        title.pack(pady=(0, 16))

        btn_frame = tk.Frame(self.main_frame, bg=WHITE)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Однофакторний аналіз", font=UI_FONT_BOLD, fg=BLACK, command=lambda: self.open_table(1), width=20).grid(row=0, column=0, padx=6, pady=6)
        tk.Button(btn_frame, text="Двофакторний аналіз", font=UI_FONT_BOLD, fg=BLACK, command=lambda: self.open_table(2), width=20).grid(row=0, column=1, padx=6, pady=6)
        tk.Button(btn_frame, text="Трифакторний аналіз", font=UI_FONT_BOLD, fg=BLACK, command=lambda: self.open_table(3), width=20).grid(row=0, column=2, padx=6, pady=6)
        tk.Button(btn_frame, text="Чотирифакторний аналіз", font=UI_FONT_BOLD, fg=BLACK, command=lambda: self.open_table(4), width=22).grid(row=0, column=3, padx=6, pady=6)

        hint = tk.Label(
            self.main_frame,
            text="Вставка з Excel: виділіть комірку → Ctrl+V. Вставляється блок даних.",
            font=UI_FONT,
            fg=BLACK,
            bg=WHITE,
        )
        hint.pack(pady=(18, 0))

        self.table_win = None
        self.indicator_name = ""
        self.units = ""

    def _setup_style(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(".", font=UI_FONT)
        style.configure("TFrame", background=WHITE)
        style.configure("TLabel", background=WHITE, foreground=BLACK)
        style.configure("TButton", foreground=BLACK)
        style.configure("Header.TLabel", font=UI_FONT_BOLD, background=HEADER_BG, foreground=BLACK)

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
        self.table_win.geometry("1400x820")
        self.table_win.configure(bg=WHITE)

        outer = tk.Frame(self.table_win, bg=WHITE)
        outer.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        ctl = tk.Frame(outer, bg=WHITE)
        ctl.pack(fill=tk.X, pady=(0, 10))

        tk.Button(ctl, text="Додати рядок", font=UI_FONT_BOLD, fg=BLACK, command=self.add_row).pack(side=tk.LEFT, padx=4)
        tk.Button(ctl, text="Видалити рядок", font=UI_FONT_BOLD, fg=BLACK, command=self.delete_row).pack(side=tk.LEFT, padx=4)

        tk.Button(ctl, text="Додати стовпчик", font=UI_FONT_BOLD, fg=BLACK, command=self.add_column).pack(side=tk.LEFT, padx=(16, 4))
        tk.Button(ctl, text="Видалити стовпчик", font=UI_FONT_BOLD, fg=BLACK, command=self.delete_column).pack(side=tk.LEFT, padx=4)

        tk.Button(ctl, text="Аналіз даних", font=UI_FONT_BOLD, fg=WHITE, bg="#c62828",
                  activebackground="#b71c1c", command=self.ask_meta_and_analyze).pack(side=tk.LEFT, padx=(18, 6))

        tk.Button(ctl, text="Про розробника", font=UI_FONT_BOLD, fg=BLACK, command=self.show_about).pack(side=tk.RIGHT, padx=4)

        # Додаткова кнопка на випадок, якщо Ctrl+V перехоплюється системою/Excel
        tk.Button(ctl, text="Вставити з буфера", font=UI_FONT_BOLD, fg=BLACK, command=self.paste_from_clipboard_button).pack(side=tk.RIGHT, padx=(4, 8))

        tip = tk.Label(
            outer,
            text="Порада: якщо Ctrl+V не спрацьовує — натисніть «Вставити з буфера».",
            font=UI_FONT,
            fg=BLACK,
            bg=WHITE,
        )
        tip.pack(anchor="w", pady=(0, 8))

        table_area = tk.Frame(outer, bg=WHITE)
        table_area.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(table_area, bg=WHITE, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.vscroll = ttk.Scrollbar(table_area, orient="vertical", command=self.canvas.yview)
        self.vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=self.vscroll.set)

        self.inner = tk.Frame(self.canvas, bg=WHITE)
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # Гарячі клавіші на рівні вікна: Ctrl+V
        self.table_win.bind("<Control-v>", self.on_paste_global)
        self.table_win.bind("<Control-V>", self.on_paste_global)

        # колонки: фактори + повтори (початково 4)
        self.repeat_cols = 4
        self.rows = 12

        factor_labels = [f"Фактор {pretty_factor_name(fk)}" for fk in self.factor_keys]
        self.column_names = factor_labels + [f"Повт.{i+1}" for i in range(self.repeat_cols)]
        self.cols = len(self.column_names)

        self.header_widgets = []
        for j, name in enumerate(self.column_names):
            lbl = ttk.Label(self.inner, text=name, style="Header.TLabel", anchor="center")
            lbl.grid(row=0, column=j, padx=1, pady=1, sticky="nsew", ipadx=12, ipady=8)
            self.header_widgets.append(lbl)
            self.inner.columnconfigure(j, weight=1)

        self.entries = []
        for i in range(self.rows):
            row_entries = []
            for j in range(self.cols):
                e = tk.Entry(self.inner, width=16, font=UI_FONT, fg=BLACK, bg=WHITE,
                             relief=tk.SOLID, bd=1, insertbackground=BLACK)
                e.grid(row=i + 1, column=j, padx=1, pady=1, sticky="nsew", ipady=8)
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
            e = tk.Entry(self.inner, width=16, font=UI_FONT, fg=BLACK, bg=WHITE,
                         relief=tk.SOLID, bd=1, insertbackground=BLACK)
            e.grid(row=i + 1, column=j, padx=1, pady=1, sticky="nsew", ipady=8)
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
        self.repeat_cols += 1
        self.cols += 1
        col_idx = self.cols - 1

        hdr = ttk.Label(self.inner, text=f"Повт.{self.repeat_cols}", style="Header.TLabel", anchor="center")
        hdr.grid(row=0, column=col_idx, padx=1, pady=1, sticky="nsew", ipadx=12, ipady=8)
        self.header_widgets.append(hdr)
        self.inner.columnconfigure(col_idx, weight=1)

        for i, row in enumerate(self.entries):
            e = tk.Entry(self.inner, width=16, font=UI_FONT, fg=BLACK, bg=WHITE,
                         relief=tk.SOLID, bd=1, insertbackground=BLACK)
            e.grid(row=i + 1, column=col_idx, padx=1, pady=1, sticky="nsew", ipady=8)
            self._bind_cell(e)
            row.append(e)

    def delete_column(self):
        min_cols = self.factors_count + 1
        if self.cols <= min_cols:
            messagebox.showinfo("Обмеження", "Мінімум має бути: фактори + 1 повторення.")
            return

        hdr = self.header_widgets.pop()
        hdr.destroy()

        for row in self.entries:
            cell = row.pop()
            cell.destroy()

        self.cols -= 1
        self.repeat_cols = self.cols - self.factors_count

    # -------------------
    # Буфер обміну (копіювання/вставка)
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

    def find_pos(self, widget):
        for i, row in enumerate(self.entries):
            for j, cell in enumerate(row):
                if cell is widget:
                    return (i, j)
        return None

    def paste_from_clipboard_button(self):
        w = self.table_win.focus_get()
        if not isinstance(w, tk.Entry):
            self.entries[0][0].focus_set()
            w = self.entries[0][0]
        self._paste_into_widget(w)

    def on_paste_global(self, event=None):
        w = self.table_win.focus_get()
        if isinstance(w, tk.Entry):
            self._paste_into_widget(w)
        return "break"

    def on_paste(self, event=None):
        widget = event.widget if (event and hasattr(event, "widget")) else self.table_win.focus_get()
        if not isinstance(widget, tk.Entry):
            return "break"
        self._paste_into_widget(widget)
        return "break"

    def _paste_into_widget(self, widget: tk.Entry):
        # 1) Пробуємо Windows API (Excel-safe)
        data = clipboard_text_windows()

        # 2) Якщо з якихось причин пусто — fallback на Tk clipboard
        if not data:
            try:
                data = self.table_win.clipboard_get()
            except Exception:
                data = ""

        if not data:
            messagebox.showwarning("Буфер порожній", "У буфері обміну немає текстових даних.")
            return

        pos = self.find_pos(widget)
        if not pos:
            return
        r0, c0 = pos

        rows = data.splitlines()
        if len(rows) == 1 and "\t" not in rows[0]:
            # простий текст: вставляємо в одну комірку
            widget.delete(0, tk.END)
            widget.insert(0, rows[0])
            widget.focus_set()
            return

        for ir, row_text in enumerate(rows):
            cols = row_text.split("\t")
            for jc, val in enumerate(cols):
                rr = r0 + ir
                cc = c0 + jc
                while rr >= len(self.entries):
                    self.add_row()
                if cc >= self.cols:
                    continue
                self.entries[rr][cc].delete(0, tk.END)
                self.entries[rr][cc].insert(0, val)

        self.entries[r0][c0].focus_set()

    # -------------------
    # Навігація
    # -------------------
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
            factor_vals = []
            for k in range(self.factors_count):
                v = row[k].get().strip()
                if v == "":
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
            "Версія: 1.0\n"
            "Розробник: (вкажіть ПІБ/організацію)\n",
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

        active_rep_cols = self._active_repeat_columns()
        repeats_used = len(active_rep_cols)

        # Таблиця НІР₀₅
        L = {fk: len(res["levels"][fk]) for fk in factors}
        r_mean = res.get("r_mean", np.nan)
        df_error = res["df_error"]
        MS_error = res["MS_error"]

        nir_table = OrderedDict()
        for fk in factors:
            other_prod = 1
            for gk in factors:
                if gk != fk:
                    other_prod *= max(L[gk], 1)
            nir_table[fk] = nir05_for_main_factor(MS_error, df_error, r_mean, other_prod)

        nir_total = res["nir_total"]

        # Звіт
        title_map = {
            1: "РЕЗУЛЬТАТИ ОДНОФАКТОРНОГО ДИСПЕРСІЙНОГО АНАЛІЗУ",
            2: "РЕЗУЛЬТАТИ ДВОФАКТОРНОГО ДИСПЕРСІЙНОГО АНАЛІЗУ",
            3: "РЕЗУЛЬТАТИ ТРИФАКТОРНОГО ДИСПЕРСІЙНОГО АНАЛІЗУ",
            4: "РЕЗУЛЬТАТИ ЧОТИРИФАКТОРНОГО ДИСПЕРСІЙНОГО АНАЛІЗУ",
        }

        lines = []
        lines.append(title_map.get(self.factors_count, "РЕЗУЛЬТАТИ ДИСПЕРСІЙНОГО АНАЛІЗУ"))
        lines.append("")
        lines.append(f"Показник: {self.indicator_name}")
        lines.append(f"Одиниці виміру: {self.units}")
        lines.append("")

        for fk in factors:
            levs = res["levels"][fk]
            lines.append(f"Фактор {pretty_factor_name(fk)}: {', '.join(map(str, levs))}")
        lines.append(f"Кількість повторень (враховано без порожніх стовпчиків): {repeats_used}")
        lines.append(f"Загальна кількість облікових значень (N): {res['N']}")
        lines.append("")

        if not (math.isnan(W) or math.isnan(p_norm)):
            lines.append(f"Нормальність залишків (Shapiro–Wilk): W = {W:.4f}, p = {p_norm:.4f} → " +
                         ("нормальний розподіл" if p_norm > 0.05 else "НЕ нормальний розподіл"))
        else:
            lines.append("Нормальність залишків (Shapiro–Wilk): недостатньо даних/помилка")
        lines.append("")

        lines.append("Істотна різниця: " + res["p_note"])
        lines.append("")

        lines.append("Таблиця дисперсійного аналізу (ANOVA):")
        lines.append("-" * 132)

        header = (
            f"{'Джерело варіації':<34} | "
            f"{'Сума квадратів':>14} | "
            f"{'Ст. віл.':>8} | "
            f"{'Сер. квадрат':>14} | "
            f"{'Fф':>10} | "
            f"{'Fтабл(0.05)':>12} | "
            f"{'p':>10} | "
            f"{'Висновок':<18}"
        )
        lines.append(header)
        lines.append("-" * 132)

        for subset, SSv, dfv, MSv, Fv, pv, Fcrit, concl in res["table_rows"]:
            name = f"Фактор {pretty_factor_name(subset[0])}" if len(subset) == 1 else f"Взаємодія {format_effect_label(subset)}"
            mark = significance_mark(pv)

            ss_s = f"{SSv:.2f}"
            df_s = f"{int(dfv)}"
            ms_s = f"{MSv:.3f}" if not math.isnan(MSv) else ""
            f_s = f"{Fv:.3f}{mark}" if (Fv is not None and not math.isnan(Fv)) else ""
            fcrit_s = f"{Fcrit:.3f}" if (Fcrit is not None and not math.isnan(Fcrit)) else ""
            p_s = f"{pv:.4f}" if (pv is not None and not math.isnan(pv)) else ""

            line = (
                f"{name:<34} | "
                f"{ss_s:>14} | "
                f"{df_s:>8} | "
                f"{ms_s:>14} | "
                f"{f_s:>10} | "
                f"{fcrit_s:>12} | "
                f"{p_s:>10} | "
                f"{concl:<18}"
            )
            lines.append(line)

        lines.append("-" * 132)
        lines.append(
            f"{'Залишок':<34} | "
            f"{res['SS_error']:.2f:>14} | "
            f"{res['df_error']:>8} | "
            f"{res['MS_error']:.3f:>14} | "
            f"{'':>10} | "
            f"{'':>12} | "
            f"{'':>10} | "
            f"{'':<18}"
        )
        lines.append(
            f"{'Загальна':<34} | "
            f"{res['SS_total']:.2f:>14} | "
            f"{res['df_total']:>8} | "
            f"{'':>14} | "
            f"{'':>10} | "
            f"{'':>12} | "
            f"{'':>10} | "
            f"{'':<18}"
        )
        lines.append("-" * 132)
        lines.append("")

        lines.append("Таблиця НІР₀₅:")
        lines.append("-" * 60)
        for fk in factors:
            val = nir_table.get(fk, np.nan)
            if val is None or math.isnan(val):
                lines.append(f"НІР₀₅ для фактора {pretty_factor_name(fk)}: (не обчислено)")
            else:
                lines.append(f"НІР₀₅ для фактора {pretty_factor_name(fk)}: {val:.3f} {self.units}")
        if nir_total is None or math.isnan(nir_total):
            lines.append("НІР₀₅ загальна: (не обчислено)")
        else:
            lines.append(f"НІР₀₅ загальна: {nir_total:.3f} {self.units}")
        lines.append("-" * 60)
        lines.append("")

        # Вікно звіту + копіювання
        win = tk.Toplevel(self.root)
        win.title(f"{APP_NAME} — Звіт")
        win.geometry("1250x850")
        win.configure(bg=WHITE)

        topbar = tk.Frame(win, bg=WHITE)
        topbar.pack(fill=tk.X, padx=10, pady=(10, 6))

        def copy_report():
            try:
                data = txt.get("1.0", "end-1c")
                self.root.clipboard_clear()
                self.root.clipboard_append(data)
                messagebox.showinfo("Готово", "Звіт скопійовано в буфер обміну.")
            except Exception as e:
                messagebox.showerror("Помилка", str(e))

        tk.Button(topbar, text="Копіювати звіт", font=UI_FONT_BOLD, fg=BLACK, command=copy_report).pack(side=tk.LEFT)

        body = tk.Frame(win, bg=WHITE)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        txt = ScrolledText(body, width=160, height=30, wrap=tk.NONE)
        txt.pack(fill=tk.BOTH, expand=True)
        txt.configure(font=REPORT_FONT, fg=BLACK, bg=WHITE, insertbackground=BLACK)
        txt.insert("1.0", "\n".join(lines))

        txt.bind("<Control-a>", lambda e: (txt.tag_add("sel", "1.0", "end-1c"), "break"))
        txt.bind("<Control-A>", lambda e: (txt.tag_add("sel", "1.0", "end-1c"), "break"))

        # Графік (пропорційний)
        plot_wrap = tk.Frame(win, bg=WHITE)
        plot_wrap.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.build_simple_plot(plot_wrap, res, factors)

    def build_simple_plot(self, parent, res, factors):
        # показуємо лише по головних факторах
        n = len(factors)
        fig_w = 9.0
        fig_h = min(max(2.6 * n, 2.6), 9.0)

        fig = plt.Figure(figsize=(fig_w, fig_h), dpi=100)
        axes = fig.subplots(n, 1) if n > 1 else [fig.add_subplot(1, 1, 1)]

        for ax, fk in zip(axes, factors):
            means_od = res["means_main"][fk]
            levs = res["levels"][fk]
            items = [(lev, means_od.get(lev, np.nan)) for lev in levs]
            items = [(lev, mu) for lev, mu in items if mu is not None and not math.isnan(mu)]
            if not items:
                ax.set_visible(False)
                continue
            labels = [str(lev) for lev, _ in items]
            vals = [mu for _, mu in items]
            x = np.arange(len(vals))
            ax.bar(x, vals)
            ax.set_title(f"Середні по фактору {pretty_factor_name(fk)}", fontsize=12, pad=8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=10)
            ax.set_ylabel(self.units if self.units else "", fontsize=10)
            for i, mu in enumerate(vals):
                ax.text(i, mu, f"{mu:.2f}", ha="center", va="bottom", fontsize=10)

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(anchor="center")


# =========================
# Run app
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = SADTk(root)
    root.mainloop()
