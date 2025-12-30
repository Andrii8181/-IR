# main.py
# -*- coding: utf-8 -*-
"""
SAD — Статистичний аналіз даних (Tkinter версія)

Потрібно: Python 3.8+, numpy, scipy, matplotlib
Встановлення:
    pip install numpy scipy matplotlib

Функції:
- 1-,2-,3-,4-факторний дисперсійний аналіз (повний факторний план з повтореннями)
- Ввід вручну + вставка з буфера (Excel) (Ctrl+V) блоком таб/новий рядок
- Навігація: Enter (вниз), стрілки (вгору/вниз/вліво/вправо)
- Додати/видалити рядок, додати/видалити стовпчик (повторення)
- Звіт в окремому вікні, можна копіювати (Times New Roman, 14)
- Буквені групи істотності (спрощено на основі LSD)
- Гістограми середніх по факторах із числами та буквами

Примітка:
- Для коректних df_error очікується, що заповнені комбінації факторів утворюють повний факторний план.
"""

import math
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from tkinter.scrolledtext import ScrolledText

import numpy as np
from scipy.stats import shapiro, t as t_dist, f as f_dist

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# -------------------------
# Допоміжні функції
# -------------------------
ALPHA = 0.05

def significance_mark(p: float) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""

def effect_label(subset_keys):
    # subset_keys is tuple like ('A','B') -> "A×B"
    return "×".join(subset_keys)

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def unique_levels(long, key):
    return sorted(list({rec[key] for rec in long if key in rec}))

def powerset(keys):
    # returns tuples of keys for all non-empty subsets, ordered by size then lexicographic
    keys = tuple(keys)
    out = []
    n = len(keys)
    for r in range(1, n+1):
        for mask in range(1<<n):
            if bin(mask).count("1") == r:
                subset = tuple(keys[i] for i in range(n) if (mask>>i)&1)
                out.append(subset)
    out.sort(key=lambda x: (len(x), x))
    return out

def build_key(rec, subset):
    return tuple(rec[k] for k in subset)

def lsd_grouping(means_dict, LSD):
    """
    Спрощене буквене групування:
    - сортуємо рівні за середнім
    - нова літера, якщо різниця з попереднім > LSD
    Це НЕ повна процедура множинних порівнянь, але дає корисне "агрономічне" групування.
    """
    items = [(k, v) for k, v in means_dict.items() if v is not None and not math.isnan(v)]
    if not items:
        return {}
    items.sort(key=lambda x: x[1])
    letters = "abcdefghijklmnopqrstuvwxyz"
    group = {}
    current_letter_idx = 0
    prev_mean = items[0][1]
    group[items[0][0]] = letters[current_letter_idx]
    if LSD is None or (isinstance(LSD, float) and math.isnan(LSD)) or LSD <= 0:
        # якщо LSD недоступний — усім одна буква
        for k, _ in items:
            group[k] = "a"
        return group

    for k, m in items[1:]:
        if abs(m - prev_mean) > LSD:
            current_letter_idx = min(current_letter_idx + 1, len(letters)-1)
        group[k] = letters[current_letter_idx]
        prev_mean = m
    return group


# -------------------------
# N-way ANOVA (1..4 фактори) без statsmodels
# Повний факторний план з повтореннями (очікується заповнення всіх комірок)
# Ефекти та взаємодії — через інклюзійно-ексклюзійну (factorial) декомпозицію середніх
# -------------------------
def anova_n_way(long, factor_keys):
    """
    long: list of dicts: {'A':..., 'B':..., 'C':..., 'D':..., 'value': float}
    factor_keys: list like ['A']..['A','B','C','D']

    returns dict:
      - table_rows: list of dicts for report
      - MS_error, df_error, SS_total, SS_error
      - means_by_factor: { 'A': {lev: mean}, ... }
      - LSD: { 'A': ..., 'B':..., 'C':..., 'D':..., 'cell':... }
      - cell_means, cell_counts
      - eta2 dict for each effect
    """
    factor_keys = list(factor_keys)
    k = len(factor_keys)
    if k < 1 or k > 4:
        raise ValueError("Supported number of factors: 1..4")

    # keep only numeric values
    clean = []
    for rec in long:
        v = rec.get("value", None)
        if v is None or math.isnan(v):
            continue
        ok = True
        for fk in factor_keys:
            if fk not in rec:
                ok = False
                break
        if ok:
            clean.append(rec)
    if not clean:
        raise ValueError("No numeric data")

    values = np.array([r["value"] for r in clean], dtype=float)
    N = len(values)
    grand_mean = float(np.mean(values))
    SS_total = float(np.sum((values - grand_mean) ** 2))

    # levels
    levels = {fk: unique_levels(clean, fk) for fk in factor_keys}
    a = {fk: len(levels[fk]) for fk in factor_keys}
    n_cells_expected = 1
    for fk in factor_keys:
        n_cells_expected *= a[fk]

    # cell means/counts (full key tuple)
    cell_means = {}
    cell_counts = {}
    cell_vals = {}
    for rec in clean:
        key = tuple(rec[fk] for fk in factor_keys)
        cell_vals.setdefault(key, []).append(rec["value"])
    for key, vals in cell_vals.items():
        cell_counts[key] = len(vals)
        cell_means[key] = float(np.mean(vals))

    # SS_error: within-cell
    SS_error = 0.0
    for key, vals in cell_vals.items():
        m = cell_means[key]
        SS_error += float(np.sum((np.array(vals, dtype=float) - m) ** 2))

    # df_error (для повного плану)
    df_total = N - 1
    df_error = N - n_cells_expected
    if df_error <= 0:
        # якщо даних мало або неповний план
        df_error = max(1, df_total)

    MS_error = SS_error / df_error if df_error > 0 else np.nan

    # precompute marginal means/counts for all subsets (including empty)
    # subset -> dict[level_tuple] = mean, and counts
    mean_sub = {(): {(): grand_mean}}
    n_sub = {(): {(): N}}

    subsets = powerset(factor_keys)
    for subset in subsets:
        m_map = {}
        n_map = {}
        # group by subset levels
        tmp_vals = {}
        for rec in clean:
            key = tuple(rec[fk] for fk in subset)
            tmp_vals.setdefault(key, []).append(rec["value"])
        for key, vals in tmp_vals.items():
            n_map[key] = len(vals)
            m_map[key] = float(np.mean(vals))
        mean_sub[subset] = m_map
        n_sub[subset] = n_map

    # compute SS for each effect subset using inclusion-exclusion delta
    SS_effect = {}
    df_effect = {}

    for subset in subsets:
        # delta(levels in subset) = sum_{T⊆subset} (-1)^{|subset|-|T|} mean_T(levels_T)
        # SS_subset = sum n_subset(levels) * delta^2
        ss = 0.0
        for lev_key, mS in mean_sub[subset].items():
            # compute delta
            delta = 0.0
            # iterate over all T ⊆ subset (including empty)
            # build by masks
            s = subset
            n_s = len(s)
            for mask in range(1 << n_s):
                T = tuple(s[i] for i in range(n_s) if (mask >> i) & 1)
                sign = (-1) ** (n_s - len(T))
                if len(T) == 0:
                    mT = grand_mean
                else:
                    # project lev_key to T coordinates
                    proj = tuple(lev_key[i] for i in range(n_s) if (mask >> i) & 1)
                    mT = mean_sub[T].get(proj, grand_mean)
                delta += sign * mT
            nlev = n_sub[subset].get(lev_key, 0)
            ss += nlev * (delta ** 2)
        SS_effect[subset] = float(ss)

        # df: product (a_i - 1)
        df = 1
        for fk in subset:
            df *= max(a[fk] - 1, 1)
        df_effect[subset] = int(df)

    # build table rows: main + interactions + error + total
    table_rows = []
    eta2 = {}

    # For each subset, compute MS, F, p, Fcrit
    SS_model = 0.0
    for subset in subsets:
        name = effect_label(subset)
        SS = SS_effect[subset]
        df = df_effect[subset]
        MS = SS / df if df > 0 else np.nan
        Fv = MS / MS_error if (MS_error is not None and not math.isnan(MS_error) and MS_error > 0) else np.nan
        p = 1 - f_dist.cdf(Fv, df, df_error) if not math.isnan(Fv) else np.nan
        Fcrit = f_dist.ppf(1 - ALPHA, df, df_error) if (df > 0 and df_error > 0) else np.nan
        concl = "істотний" if (p is not None and not math.isnan(p) and p < ALPHA) else "неістотний"
        table_rows.append({
            "name": name,
            "SS": SS,
            "df": df,
            "MS": MS,
            "F": Fv,
            "Fcrit": Fcrit,
            "p": p,
            "mark": significance_mark(p),
            "conclusion": concl
        })
        SS_model += SS

        eta2[subset] = (SS / SS_total) if SS_total > 0 else np.nan

    # add error & total
    table_rows.append({
        "name": "Випадкова помилка",
        "SS": SS_error,
        "df": df_error,
        "MS": MS_error,
        "F": None,
        "Fcrit": None,
        "p": None,
        "mark": "",
        "conclusion": ""
    })
    table_rows.append({
        "name": "Загальна",
        "SS": SS_total,
        "df": df_total,
        "MS": None,
        "F": None,
        "Fcrit": None,
        "p": None,
        "mark": "",
        "conclusion": ""
    })

    # marginal means by factor
    means_by_factor = {}
    for fk in factor_keys:
        subset = (fk,)
        means_by_factor[fk] = {lev[0]: mean_sub[subset][(lev[0],)] for lev in [(x,) for x in levels[fk]] if (lev in mean_sub[subset])}

    # estimate r (repeat per cell average)
    r_list = list(cell_counts.values())
    r_mean = float(np.mean([x for x in r_list if x > 0])) if any(x > 0 for x in r_list) else np.nan

    # LSD for main factors + cell
    tval = t_dist.ppf(1 - ALPHA/2, df_error) if df_error > 0 else np.nan
    LSD = {}
    for fk in factor_keys:
        # per-level effective n = (product other factor levels) * r_mean
        other_prod = 1
        for gk in factor_keys:
            if gk != fk:
                other_prod *= a[gk]
        denom = other_prod * r_mean
        LSD[fk] = (tval * math.sqrt(2 * MS_error / denom)) if not any(math.isnan(x) for x in [tval, MS_error, denom]) and denom > 0 else np.nan

    # combinations LSD (cell mean comparisons)
    LSD["cell"] = (tval * math.sqrt(2 * MS_error / r_mean)) if not any(math.isnan(x) for x in [tval, MS_error, r_mean]) and r_mean > 0 else np.nan

    return {
        "factor_keys": factor_keys,
        "levels": levels,
        "N": N,
        "grand_mean": grand_mean,
        "SS_total": SS_total,
        "SS_error": SS_error,
        "df_total": df_total,
        "df_error": df_error,
        "MS_error": MS_error,
        "table_rows": table_rows,
        "eta2": eta2,                # subset tuple -> eta2
        "means_by_factor": means_by_factor,
        "cell_means": cell_means,
        "cell_counts": cell_counts,
        "r_mean": r_mean,
        "LSD": LSD
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

        title = tk.Label(self.main_frame, text="SAD — Статистичний аналіз даних", font=("Arial", 18, "bold"))
        title.pack(pady=12)

        btn_frame = tk.Frame(self.main_frame)
        btn_frame.pack(pady=8)

        tk.Button(btn_frame, text="Однофакторний аналіз", width=22, height=2, command=lambda: self.open_table(1)).grid(row=0, column=0, padx=6)
        tk.Button(btn_frame, text="Двофакторний аналіз", width=22, height=2, command=lambda: self.open_table(2)).grid(row=0, column=1, padx=6)
        tk.Button(btn_frame, text="Трифакторний аналіз", width=22, height=2, command=lambda: self.open_table(3)).grid(row=0, column=2, padx=6)
        tk.Button(btn_frame, text="4-факторний аналіз", width=22, height=2, command=lambda: self.open_table(4)).grid(row=0, column=3, padx=6)

        info = tk.Label(self.main_frame, text="Виберіть тип аналізу → внесіть дані → натисніть «Аналіз даних»", fg="gray")
        info.pack(pady=10)

        self.table_win = None
        self.entries = []
        self.headers = []
        self.factor_keys = []
        self.repeat_count = 4

        # Report style for easy copy to Word
        self.report_font = ("Times New Roman", 14)

    def show_about(self):
        messagebox.showinfo(
            "Про розробника",
            "SAD — Статистичний аналіз даних\n"
            "Tkinter версія\n\n"
            "Розробник: (вкажи тут ПІБ/організацію)\n"
            "Кафедра плодівництва і виноградарства УНУ"
        )

    def open_table(self, factors_count: int):
        if self.table_win and tk.Toplevel.winfo_exists(self.table_win):
            self.table_win.destroy()

        self.factors_count = factors_count
        self.factor_keys = list("ABCD")[:factors_count]  # internal keys

        self.table_win = tk.Toplevel(self.root)
        self.table_win.title(f"SAD — {factors_count}-факторний аналіз")
        self.table_win.geometry("1200x650")

        # columns: factors + repeats (start with 4)
        self.repeat_count = 4
        factor_names_ui = []
        # UA letters for labels: А, В, С, D? (для 4-го логічно «D», але можна «D/Г»)
        ua_letters = ["А", "В", "С", "D"]
        for i in range(factors_count):
            factor_names_ui.append(f"Фактор {ua_letters[i]}")
        self.column_names = factor_names_ui + [f"Повт.{i+1}" for i in range(self.repeat_count)]

        # Top control bar
        ctl_frame = tk.Frame(self.table_win)
        ctl_frame.pack(fill=tk.X, padx=8, pady=6)

        # left controls
        tk.Button(ctl_frame, text="Додати рядок", command=self.add_row).pack(side=tk.LEFT, padx=4)
        tk.Button(ctl_frame, text="Видалити рядок", command=self.delete_row).pack(side=tk.LEFT, padx=4)
        tk.Button(ctl_frame, text="Додати стовпчик", command=self.add_column).pack(side=tk.LEFT, padx=12)
        tk.Button(ctl_frame, text="Видалити стовпчик", command=self.delete_column).pack(side=tk.LEFT, padx=4)

        tk.Button(ctl_frame, text="Вставити з буфера (Excel)", command=self.paste_from_clipboard).pack(side=tk.LEFT, padx=12)

        tk.Button(ctl_frame, text="Аналіз даних", bg="#d32f2f", fg="white", command=self.analyze).pack(side=tk.LEFT, padx=18)

        # right controls
        tk.Button(ctl_frame, text="Про розробника", command=self.show_about).pack(side=tk.RIGHT, padx=4)

        # Scrolling area
        self.canvas = tk.Canvas(self.table_win)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.vscroll = ttk.Scrollbar(self.table_win, orient="vertical", command=self.canvas.yview)
        self.vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=self.vscroll.set)

        self.inner = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        # Build table
        self.rows = 10
        self.cols = len(self.column_names)
        self.entries = []
        self.headers = []

        self._render_headers()
        for _ in range(self.rows):
            self._append_row_widgets()

        self.inner.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

        self.entries[0][0].focus_set()

    def _bind_cell(self, e: tk.Entry):
        e.bind("<Return>", self.on_enter)
        e.bind("<Up>", self.on_arrow)
        e.bind("<Down>", self.on_arrow)
        e.bind("<Left>", self.on_arrow)
        e.bind("<Right>", self.on_arrow)

        # Ctrl+C / Ctrl+V (single cell fallback + block paste)
        e.bind("<Control-c>", self.on_copy)
        e.bind("<Control-C>", self.on_copy)
        e.bind("<Control-v>", self.on_paste)
        e.bind("<Control-V>", self.on_paste)

    def _render_headers(self):
        # clear old headers if any
        for w in self.headers:
            w.destroy()
        self.headers = []

        for j, name in enumerate(self.column_names):
            lbl = tk.Label(self.inner, text=name, relief=tk.RIDGE, width=14, bg="#f0f0f0")
            lbl.grid(row=0, column=j, padx=1, pady=1, sticky="nsew")
            self.headers.append(lbl)

    def _append_row_widgets(self):
        i = len(self.entries)
        row_entries = []
        for j in range(self.cols):
            e = tk.Entry(self.inner, width=14)
            e.grid(row=i+1, column=j, padx=1, pady=1)
            self._bind_cell(e)
            row_entries.append(e)
        self.entries.append(row_entries)

    # -------------------
    # Table operations
    # -------------------
    def add_row(self):
        self._append_row_widgets()
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
        # Add a new repeat column at the end
        self.repeat_count += 1
        self.cols += 1
        self.column_names.append(f"Повт.{self.repeat_count}")

        # render new header cell
        j = self.cols - 1
        lbl = tk.Label(self.inner, text=self.column_names[-1], relief=tk.RIDGE, width=14, bg="#f0f0f0")
        lbl.grid(row=0, column=j, padx=1, pady=1, sticky="nsew")
        self.headers.append(lbl)

        # add a new Entry per row
        for i in range(len(self.entries)):
            e = tk.Entry(self.inner, width=14)
            e.grid(row=i+1, column=j, padx=1, pady=1)
            self._bind_cell(e)
            self.entries[i].append(e)

        self.inner.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def delete_column(self):
        # keep minimum: factors + 1 repeat
        min_cols = self.factors_count + 1
        if self.cols <= min_cols:
            return

        # remove last column widgets
        j = self.cols - 1
        # header
        h = self.headers.pop()
        h.destroy()

        # entries
        for i in range(len(self.entries)):
            e = self.entries[i].pop()
            e.destroy()

        self.cols -= 1
        self.repeat_count -= 1
        self.column_names.pop()

        self.inner.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    # -------------------
    # Navigation
    # -------------------
    def _find_pos(self, widget):
        for i, row in enumerate(self.entries):
            for j, cell in enumerate(row):
                if cell is widget:
                    return i, j
        return None

    def on_enter(self, event=None):
        widget = event.widget
        pos = self._find_pos(widget)
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
        pos = self._find_pos(widget)
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
    # Clipboard: copy/paste
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

    def paste_from_clipboard(self):
        # paste into currently focused cell
        w = self.table_win.focus_get()
        if isinstance(w, tk.Entry):
            fake_event = type("E", (), {"widget": w})()
            self.on_paste(fake_event)

    def on_paste(self, event=None):
        widget = event.widget
        try:
            data = widget.selection_get(selection="CLIPBOARD")
        except Exception:
            try:
                data = self.table_win.clipboard_get()
            except Exception:
                return "break"

        data = data.replace("\r\n", "\n").replace("\r", "\n")
        rows = [r for r in data.split("\n") if r != ""]
        if not rows:
            return "break"

        pos = self._find_pos(widget)
        if not pos:
            return "break"
        r0, c0 = pos

        for i_r, row_text in enumerate(rows):
            cols = row_text.split("\t")
            for j_c, val in enumerate(cols):
                rr = r0 + i_r
                cc = c0 + j_c
                while rr >= len(self.entries):
                    self.add_row()
                while cc >= self.cols:
                    # якщо вставляють більше повторів — додаємо стовпчики
                    self.add_column()
                self.entries[rr][cc].delete(0, tk.END)
                self.entries[rr][cc].insert(0, val)

        return "break"

    # -------------------
    # Collect data
    # -------------------
    def collect_long(self):
        long = []
        # repeats are columns after factors_count
        rep_cols = list(range(self.factors_count, self.cols))

        for i, row in enumerate(self.entries):
            # factor level strings
            f_levels = []
            for k in range(self.factors_count):
                v = row[k].get().strip()
                if v == "":
                    # якщо не задано — робимо унікальний рівень, щоб не зливалося
                    v = f"lev_row{i}_col{k}"
                f_levels.append(v)

            for rc in rep_cols:
                txt = row[rc].get().strip()
                if txt == "":
                    continue
                v = safe_float(txt)
                if math.isnan(v):
                    continue

                rec = {"value": float(v)}
                # map to A,B,C,D
                for idx, fk in enumerate(self.factor_keys):
                    rec[fk] = f_levels[idx]
                long.append(rec)

        return long

    # -------------------
    # Reporting + plots
    # -------------------
    def _open_plot_window(self, title, categories, means, letters, ylabel):
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("900x500")

        fig = Figure(figsize=(9, 4.5), dpi=100)
        ax = fig.add_subplot(111)

        x = np.arange(len(categories))
        bars = ax.bar(x, means)  # default color

        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # annotate values + letters
        for i, (b, m) in enumerate(zip(bars, means)):
            txt = f"{m:.2f}\n{letters.get(categories[i], '')}"
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), txt, ha="center", va="bottom")

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def analyze(self):
        # ask indicator & units
        indicator = simpledialog.askstring("Назва показника", "Введіть назву показника:", parent=self.table_win)
        if indicator is None:
            return
        units = simpledialog.askstring("Одиниці виміру", "Введіть одиниці виміру:", parent=self.table_win)
        if units is None:
            return

        long = self.collect_long()
        if len(long) < 3:
            messagebox.showwarning("Помилка", "Надто мало числових даних для аналізу.")
            return

        factor_keys = self.factor_keys  # e.g., ['A','B','C','D']
        try:
            res = anova_n_way(long, factor_keys)
        except Exception as e:
            messagebox.showerror("Помилка аналізу", f"Не вдалося виконати аналіз:\n{e}")
            return

        # residuals for Shapiro: value - cell_mean (full combination)
        residuals = []
        for rec in long:
            cell_key = tuple(rec[fk] for fk in factor_keys)
            m = res["cell_means"].get(cell_key, res["grand_mean"])
            residuals.append(rec["value"] - m)
        residuals = np.array(residuals, dtype=float)
        try:
            W, pW = shapiro(residuals) if len(residuals) >= 3 else (np.nan, np.nan)
        except Exception:
            W, pW = (np.nan, np.nan)

        # letters for marginal means by factor
        letters_by_factor = {}
        for fk in factor_keys:
            means = res["means_by_factor"].get(fk, {})
            LSD_fk = res["LSD"].get(fk, np.nan)
            letters_by_factor[fk] = lsd_grouping(means, LSD_fk)

        # Build report text
        k = len(factor_keys)
        title_map = {1: "О Д Н О Ф А К Т О Р Н О Г О", 2: "Д В О Ф А К Т О Р Н О Г О", 3: "Т Р И Ф А К Т О Р Н О Г О", 4: "Ч О Т И Р И Ф А К Т О Р Н О Г О"}
        report = []
        report.append(f"Р Е З У Л Ь Т А Т И   {title_map.get(k,'')}   Д И С П Е Р С І Й Н О Г О   А Н А Л І З У")
        report.append("")
        report.append(f"Показник: {indicator}")
        report.append(f"Одиниці виміру: {units}")
        report.append("")

        # factor description (list levels)
        ua_letters = {"A": "А", "B": "В", "C": "С", "D": "D"}
        for fk in factor_keys:
            levs = unique_levels(long, fk)
            report.append(f"Фактор {ua_letters[fk]}: {len(levs)} рівні → {', '.join(map(str, levs[:30]))}{' ...' if len(levs)>30 else ''}")

        report.append(f"Загальна кількість облікових значень (N): {res['N']}")
        report.append(f"Оціночна кількість повторень (середня на комірку): {res['r_mean']:.2f}")
        report.append("")

        if not math.isnan(W):
            report.append(f"Перевірка нормальності залишків (Shapiro–Wilk): W = {W:.4f}, p = {pW:.4f} → {'нормальний' if pW>ALPHA else 'НЕ нормальний'}")
        else:
            report.append("Перевірка нормальності залишків (Shapiro–Wilk): недостатньо даних/помилка обчислення")
        report.append("")
        report.append("Позначення істотності за p:")
        report.append("  *  — p < 0.05")
        report.append("  ** — p < 0.01")
        report.append("")

        # ANOVA table
        report.append("────────────────────────────────────────────────────────────────────────────────────────────")
        report.append(f"{'Джерело варіації':<24}{'SS':>12}{'df':>8}{'MS':>12}{'F':>10}{'Fтабл(0.05)':>14}{'p':>10}{'Висновок':>12}")
        report.append("────────────────────────────────────────────────────────────────────────────────────────────")

        for row in res["table_rows"]:
            name = row["name"]
            SS = row["SS"]
            df = row["df"]
            MS = row["MS"]
            Fv = row["F"]
            Fcrit = row["Fcrit"]
            pval = row["p"]
            mark = row.get("mark", "")
            concl = row.get("conclusion", "")

            if name == "Випадкова помилка":
                report.append(f"{name:<24}{SS:12.2f}{df:8d}{MS:12.3f}{'':>10}{'':>14}{'':>10}{'':>12}")
            elif name == "Загальна":
                report.append(f"{name:<24}{SS:12.2f}{df:8d}{'':>12}{'':>10}{'':>14}{'':>10}{'':>12}")
            else:
                Fs = "" if Fv is None or (isinstance(Fv,float) and math.isnan(Fv)) else f"{Fv:10.3f}"
                Fcs = "" if Fcrit is None or (isinstance(Fcrit,float) and math.isnan(Fcrit)) else f"{Fcrit:14.2f}"
                ps = "" if pval is None or (isinstance(pval,float) and math.isnan(pval)) else f"{pval:10.4f}{mark}"
                report.append(f"{name:<24}{SS:12.2f}{df:8d}{MS:12.3f}{Fs}{Fcs}{ps:>14}{concl:>12}")

        report.append("────────────────────────────────────────────────────────────────────────────────────────────")
        report.append("")

        # eta2
        report.append("Вилучення впливу (η², % від загальної дисперсії):")
        # print in same order as subsets
        for subset in powerset(factor_keys):
            et = res["eta2"].get(subset, np.nan)
            if et is None or math.isnan(et):
                continue
            report.append(f"  • {effect_label(subset):<12} — {et*100:5.1f}%")
        report.append("")

        # LSD
        report.append("НІР₀.₅ (LSD):")
        for fk in factor_keys:
            v = res["LSD"].get(fk, np.nan)
            if v is None or math.isnan(v):
                continue
            report.append(f"  • По фактору {ua_letters[fk]:<2} — {v:.3f} {units}")
        vcell = res["LSD"].get("cell", np.nan)
        if vcell is not None and not math.isnan(vcell):
            report.append(f"  • По комбінаціях — {vcell:.3f} {units}")
        report.append("")

        # Means + letters per factor
        report.append("Середні по факторах (з буквами істотності; однакові літери → різниця не перевищує LSD):")
        for fk in factor_keys:
            report.append(f"Фактор {ua_letters[fk]}:")
            means = res["means_by_factor"].get(fk, {})
            letters = letters_by_factor.get(fk, {})
            # sort by mean
            items = sorted([(k2, v2) for k2, v2 in means.items() if v2 is not None and not math.isnan(v2)], key=lambda x: x[1])
            for lev, m in items:
                report.append(f"  {str(lev):<18} {m:8.3f} {units}   {letters.get(lev,'')}")
            report.append("")

        # show report window
        report_text = "\n".join(report)
        self._show_report_window(report_text)

        # plots
        ylabel = f"{indicator}, {units}"
        for fk in factor_keys:
            means = res["means_by_factor"].get(fk, {})
            if not means:
                continue
            items = sorted([(k2, v2) for k2, v2 in means.items() if v2 is not None and not math.isnan(v2)], key=lambda x: x[0])
            cats = [str(k2) for k2, _ in items]
            vals = [v2 for _, v2 in items]
            letters = {str(k2): letters_by_factor.get(fk, {}).get(k2, "") for k2, _ in items}
            self._open_plot_window(
                title=f"Середні по фактору {ua_letters[fk]}",
                categories=cats,
                means=vals,
                letters=letters,
                ylabel=ylabel
            )

    def _show_report_window(self, report_text: str):
        win = tk.Toplevel(self.root)
        win.title("Звіт аналізу (можна копіювати)")
        win.geometry("1100x750")

        # Copy hint
        hint = tk.Label(win, text="Порада: Ctrl+A → Ctrl+C, щоб швидко скопіювати звіт у Word.", fg="gray")
        hint.pack(anchor="w", padx=8, pady=(8, 0))

        txt = ScrolledText(win, wrap=tk.WORD, width=130, height=40, font=self.report_font)
        txt.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        txt.insert("1.0", report_text)
        txt.focus_set()

        # add Ctrl+A handler
        def select_all(event=None):
            txt.tag_add("sel", "1.0", "end-1c")
            return "break"
        txt.bind("<Control-a>", select_all)
        txt.bind("<Control-A>", select_all)


# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = SADTk(root)
    root.mainloop()
