# main.py
# -*- coding: utf-8 -*-
"""
S.A.D. — Статистичний аналіз даних (Tkinter)
Потрібно: Python 3.8+, numpy, scipy
Встановлення: pip install numpy scipy
"""

import math
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
from itertools import combinations

import numpy as np
from scipy.stats import shapiro, t as t_dist, f as f_dist


# -------------------------
# Допоміжні функції
# -------------------------
def safe_float(x: str):
    try:
        if x is None:
            return np.nan
        s = str(x).strip().replace(" ", "").replace(",", ".")
        if s == "":
            return np.nan
        return float(s)
    except Exception:
        return np.nan


def p_mark(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def fmt_num(x, nd=3):
    if x is None:
        return ""
    try:
        if isinstance(x, float) and math.isnan(x):
            return ""
    except Exception:
        pass
    return f"{x:.{nd}f}"


def fmt_int(x):
    if x is None:
        return ""
    try:
        if isinstance(x, float) and math.isnan(x):
            return ""
    except Exception:
        pass
    return f"{int(x)}"


def ask_indicator_units(parent, title="Параметри показника"):
    """
    Одночасний запит назви показника та одиниць виміру.
    Повертає (indicator, units) або None.
    """
    dlg = tk.Toplevel(parent)
    dlg.title(title)
    dlg.transient(parent)
    dlg.grab_set()
    dlg.resizable(False, False)

    frm = ttk.Frame(dlg, padding=14)
    frm.pack(fill=tk.BOTH, expand=True)

    ttk.Label(frm, text="Назва показника:", font=("Segoe UI", 12, "bold"), foreground="black").grid(row=0, column=0, sticky="w")
    e1 = ttk.Entry(frm, width=44)
    e1.grid(row=1, column=0, sticky="we", pady=(4, 10))

    ttk.Label(frm, text="Одиниці виміру:", font=("Segoe UI", 12, "bold"), foreground="black").grid(row=2, column=0, sticky="w")
    e2 = ttk.Entry(frm, width=44)
    e2.grid(row=3, column=0, sticky="we", pady=(4, 12))

    btns = ttk.Frame(frm)
    btns.grid(row=4, column=0, sticky="e")

    result = {"ok": False, "indicator": "", "units": ""}

    def on_ok():
        indicator = e1.get().strip()
        units = e2.get().strip()
        if not indicator:
            messagebox.showwarning("Помилка", "Вкажіть назву показника.", parent=dlg)
            return
        if not units:
            messagebox.showwarning("Помилка", "Вкажіть одиниці виміру.", parent=dlg)
            return
        result["ok"] = True
        result["indicator"] = indicator
        result["units"] = units
        dlg.destroy()

    def on_cancel():
        dlg.destroy()

    ttk.Button(btns, text="Скасувати", command=on_cancel).pack(side=tk.RIGHT, padx=(6, 0))
    ttk.Button(btns, text="OK", command=on_ok).pack(side=tk.RIGHT)

    frm.columnconfigure(0, weight=1)
    e1.focus_set()
    parent.wait_window(dlg)

    if result["ok"]:
        return result["indicator"], result["units"]
    return None


# -------------------------
# ANOVA через OLS (послідовні суми квадратів — Type I)
# Підтримка 1–4 факторів (A,B,C,D) + взаємодії до повного порядку.
# -------------------------
def _one_hot(indices, n_levels, drop_first=True):
    N = len(indices)
    k = n_levels - 1 if drop_first else n_levels
    if k <= 0:
        return np.zeros((N, 0), dtype=float)
    out = np.zeros((N, k), dtype=float)
    for i, idx in enumerate(indices):
        if drop_first:
            if idx == 0:
                continue
            out[i, idx - 1] = 1.0
        else:
            out[i, idx] = 1.0
    return out


def _interaction_columns(*matrices):
    if any(m.shape[1] == 0 for m in matrices):
        return np.zeros((matrices[0].shape[0], 0), dtype=float)
    M = matrices[0]
    for nxt in matrices[1:]:
        cols = []
        for i in range(M.shape[1]):
            for j in range(nxt.shape[1]):
                cols.append((M[:, i] * nxt[:, j])[:, None])
        M = np.hstack(cols) if cols else np.zeros((M.shape[0], 0), dtype=float)
    return M


def _ols_sse(X, y):
    if X.shape[1] == 0:
        resid = y
        return float(np.sum(resid ** 2)), 0
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - (X @ beta)
    sse = float(np.sum(resid ** 2))
    rank = int(np.linalg.matrix_rank(X))
    return sse, rank


def anova_ols(long_records, factor_keys):
    y = np.array([r["value"] for r in long_records], dtype=float)
    N = len(y)
    if N < 3:
        raise ValueError("Недостатньо даних (менше 3 значень).")

    grand_mean = float(np.mean(y))
    SS_total = float(np.sum((y - grand_mean) ** 2))
    df_total = N - 1

    # порядок рівнів — як у введенні
    level_orders = {}
    idx_arrays = {}
    for fk in factor_keys:
        order = []
        mapping = {}
        idx = []
        for r in long_records:
            lv = r[fk]
            if lv not in mapping:
                mapping[lv] = len(order)
                order.append(lv)
            idx.append(mapping[lv])
        level_orders[fk] = order
        idx_arrays[fk] = np.array(idx, dtype=int)

    dummies = {}
    for fk in factor_keys:
        n_lv = len(level_orders[fk])
        dummies[fk] = _one_hot(idx_arrays[fk], n_lv, drop_first=True)

    # ефекти: головні + взаємодії
    effects = []
    for fk in factor_keys:
        effects.append((fk,))
    for r in range(2, len(factor_keys) + 1):
        for comb in combinations(factor_keys, r):
            effects.append(tuple(comb))

    X0 = np.ones((N, 1), dtype=float)
    sse_prev, rank_prev = _ols_sse(X0, y)
    X_curr = X0.copy()

    rows = []
    for eff in effects:
        mats = [dummies[k] for k in eff]
        if len(eff) == 1:
            block = mats[0]
            eff_name = f"Фактор {eff[0]}"
        else:
            block = _interaction_columns(*mats)
            eff_name = " × ".join([f"Фактор {k}" for k in eff])

        df_eff = block.shape[1]
        if df_eff <= 0:
            continue

        X_next = np.hstack([X_curr, block])
        sse_next, rank_next = _ols_sse(X_next, y)

        SS_eff = max(0.0, sse_prev - sse_next)
        MS_eff = SS_eff / df_eff if df_eff > 0 else np.nan

        rows.append({
            "name": eff_name,
            "SS": SS_eff,
            "df": df_eff,
            "MS": MS_eff,
        })

        X_curr = X_next
        sse_prev, rank_prev = sse_next, rank_next

    sse_full = sse_prev
    rank_full = rank_prev
    df_error = N - rank_full
    if df_error <= 0:
        raise ValueError("Неможливо оцінити залишок: df(залишок) ≤ 0 (перевірте дані).")
    MS_error = sse_full / df_error

    alpha = 0.05
    for r in rows:
        Fv = (r["MS"] / MS_error) if (MS_error > 0 and not math.isnan(r["MS"])) else np.nan
        pv = 1.0 - f_dist.cdf(Fv, r["df"], df_error) if not math.isnan(Fv) else np.nan
        Fcrit = f_dist.ppf(1.0 - alpha, r["df"], df_error)
        r["F"] = Fv
        r["p"] = pv
        r["Fcrit"] = Fcrit
        r["conclusion"] = "істотна різниця" if (not math.isnan(pv) and pv < 0.05) else "неістотна"

    rows_with_totals = rows.copy()
    rows_with_totals.append({
        "name": "Залишок",
        "SS": sse_full,
        "df": df_error,
        "MS": MS_error,
        "F": np.nan,
        "p": np.nan,
        "Fcrit": np.nan,
        "conclusion": ""
    })
    rows_with_totals.append({
        "name": "Загальна",
        "SS": SS_total,
        "df": df_total,
        "MS": np.nan,
        "F": np.nan,
        "p": np.nan,
        "Fcrit": np.nan,
        "conclusion": ""
    })

    means = {}
    for fk in factor_keys:
        order = level_orders[fk]
        out = []
        for lv in order:
            vals = [r["value"] for r in long_records if r[fk] == lv]
            out.append((lv, float(np.mean(vals)) if vals else np.nan, len(vals)))
        means[fk] = out

    return rows_with_totals, MS_error, df_error, SS_total, df_total, level_orders, means


# -------------------------
# НІР₀₅ (без згадки LSD)
# -------------------------
def nir05(MS_error, df_error, n_eff):
    if df_error <= 0 or MS_error <= 0 or n_eff <= 0:
        return np.nan
    tval = t_dist.ppf(0.975, df_error)
    return float(tval * math.sqrt(2.0 * MS_error / n_eff))


# -------------------------
# GUI
# -------------------------
class SADApp:
    def __init__(self, root):
        self.root = root
        self.root.title("S.A.D. — Статистичний аналіз даних")
        self.root.geometry("980x520")
        self.root.configure(bg="white")

        # Масштаб для чіткості/читабельності (Windows)
        try:
            self.root.tk.call("tk", "scaling", 1.25)
        except Exception:
            pass

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(".", font=("Segoe UI", 12))
        style.configure("TLabel", foreground="black", background="white")
        style.configure("TFrame", background="white")
        style.configure("TButton", padding=8)
        style.configure("Accent.TButton", font=("Segoe UI", 12, "bold"))

        self.main = ttk.Frame(root, padding=18)
        self.main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            self.main,
            text="S.A.D. — Статистичний аналіз даних",
            font=("Segoe UI", 20, "bold"),
            foreground="black"
        ).pack(pady=(6, 12))

        ttk.Label(
            self.main,
            text="Оберіть тип дисперсійного аналізу",
            font=("Segoe UI", 12),
            foreground="black"
        ).pack(pady=(0, 16))

        btns = ttk.Frame(self.main)
        btns.pack(pady=8)

        ttk.Button(btns, text="Однофакторний аналіз", style="Accent.TButton",
                   command=lambda: self.open_table(1)).grid(row=0, column=0, padx=8, pady=6)
        ttk.Button(btns, text="Двофакторний аналіз", style="Accent.TButton",
                   command=lambda: self.open_table(2)).grid(row=0, column=1, padx=8, pady=6)
        ttk.Button(btns, text="Трифакторний аналіз", style="Accent.TButton",
                   command=lambda: self.open_table(3)).grid(row=0, column=2, padx=8, pady=6)
        ttk.Button(btns, text="Чотирифакторний аналіз", style="Accent.TButton",
                   command=lambda: self.open_table(4)).grid(row=0, column=3, padx=8, pady=6)

        ttk.Label(
            self.main,
            text="Вставка з Excel: виділіть першу комірку → Ctrl+V або кнопка «Вставити з буфера».",
            font=("Segoe UI", 11),
            foreground="black"
        ).pack(pady=(22, 0))

        self.table_win = None
        self.entries = []
        self.headers = []
        self.factors_count = 1
        self.repeat_count = 4
        self.canvas = None
        self.inner = None

    # -------------------------
    # Таблиця
    # -------------------------
    def open_table(self, factors_count: int):
        if self.table_win and self.table_win.winfo_exists():
            self.table_win.destroy()

        self.factors_count = factors_count

        self.table_win = tk.Toplevel(self.root)
        self.table_win.title(f"S.A.D. — {factors_count}-факторний аналіз")
        self.table_win.geometry("1200x680")
        self.table_win.configure(bg="white")

        top = ttk.Frame(self.table_win, padding=(12, 10))
        top.pack(fill=tk.X)

        left = ttk.Frame(top)
        left.pack(side=tk.LEFT, anchor="w")

        ttk.Button(left, text="Додати рядок", command=self.add_row).pack(side=tk.LEFT, padx=6)
        ttk.Button(left, text="Видалити рядок", command=self.delete_row).pack(side=tk.LEFT, padx=6)
        ttk.Button(left, text="Додати стовпчик", command=self.add_repeat_column).pack(side=tk.LEFT, padx=(20, 6))
        ttk.Button(left, text="Видалити стовпчик", command=self.delete_repeat_column).pack(side=tk.LEFT, padx=6)
        ttk.Button(left, text="Вставити з буфера", command=self.paste_from_clipboard_button).pack(side=tk.LEFT, padx=(20, 6))
        ttk.Button(left, text="Аналіз даних", style="Accent.TButton", command=self.analyze).pack(side=tk.LEFT, padx=(20, 6))

        right = ttk.Frame(top)
        right.pack(side=tk.RIGHT, anchor="e")
        ttk.Button(right, text="Про розробника", command=self.show_about).pack(side=tk.RIGHT)

        body = ttk.Frame(self.table_win, padding=(12, 0, 12, 12))
        body.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(body, bg="white", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        vsb = ttk.Scrollbar(body, orient="vertical", command=self.canvas.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=vsb.set)

        self.inner = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # заголовки
        factor_labels = ["А", "В", "С", "D"]
        factor_cols = [f"Фактор {factor_labels[i]}" for i in range(factors_count)]
        self.repeat_count = 4
        repeat_cols = [f"Повт.{i+1}" for i in range(self.repeat_count)]
        self.headers = factor_cols + repeat_cols

        self._render_headers()

        self.entries = []
        for _ in range(10):
            self._append_row()

        # Глобальні гарячі клавіші для Windows (Ctrl+V/ Ctrl+C)
        self.table_win.bind_all("<Control-v>", self.on_paste_global, add="+")
        self.table_win.bind_all("<Control-V>", self.on_paste_global, add="+")
        self.table_win.bind_all("<Control-c>", self.on_copy_global, add="+")
        self.table_win.bind_all("<Control-C>", self.on_copy_global, add="+")

        self.entries[0][0].focus_set()

    def _render_headers(self):
        for w in list(self.inner.grid_slaves(row=0)):
            w.destroy()

        for j, name in enumerate(self.headers):
            lbl = tk.Label(
                self.inner,
                text=name,
                fg="black",
                bg="white",
                font=("Segoe UI", 12, "bold"),
                relief=tk.GROOVE,
                bd=1,
                padx=6,
                pady=4
            )
            lbl.grid(row=0, column=j, padx=(2, 14 if j == 0 else 2), pady=(2, 6), sticky="nsew")
            self.inner.grid_columnconfigure(j, minsize=140)

    def _bind_cell(self, e: tk.Entry):
        e.bind("<Return>", self.on_enter)
        e.bind("<Up>", self.on_arrow)
        e.bind("<Down>", self.on_arrow)
        e.bind("<Left>", self.on_arrow)
        e.bind("<Right>", self.on_arrow)
        e.bind("<Control-v>", self.on_paste_cell)
        e.bind("<Control-V>", self.on_paste_cell)
        e.bind("<Control-c>", self.on_copy_cell)
        e.bind("<Control-C>", self.on_copy_cell)

    def _append_row(self):
        r = len(self.entries) + 1
        row = []
        for c in range(len(self.headers)):
            e = tk.Entry(
                self.inner,
                width=16,
                fg="black",
                bg="white",
                font=("Segoe UI", 13),
                relief=tk.SOLID,
                bd=1
            )
            e.grid(row=r, column=c, padx=(2, 14 if c == 0 else 2), pady=2)
            self._bind_cell(e)
            row.append(e)
        self.entries.append(row)

    def add_row(self):
        self._append_row()

    def delete_row(self):
        if not self.entries:
            return
        last = self.entries.pop()
        for e in last:
            e.destroy()

    def add_repeat_column(self):
        self.repeat_count += 1
        self.headers.append(f"Повт.{self.repeat_count}")
        self._render_headers()

        col_idx = len(self.headers) - 1
        for i, row in enumerate(self.entries):
            r = i + 1
            e = tk.Entry(self.inner, width=16, fg="black", bg="white",
                         font=("Segoe UI", 13), relief=tk.SOLID, bd=1)
            e.grid(row=r, column=col_idx, padx=2, pady=2)
            self._bind_cell(e)
            row.append(e)

    def delete_repeat_column(self):
        if self.repeat_count <= 1:
            return
        self.repeat_count -= 1

        col_idx = len(self.headers) - 1
        for w in self.inner.grid_slaves(row=0, column=col_idx):
            w.destroy()

        for row in self.entries:
            cell = row.pop()
            cell.destroy()

        self.headers.pop()
        self._render_headers()

    # -------------------------
    # Навігація
    # -------------------------
    def _find_pos(self, widget):
        for i, row in enumerate(self.entries):
            for j, cell in enumerate(row):
                if cell is widget:
                    return i, j
        return None

    def on_enter(self, event=None):
        pos = self._find_pos(event.widget)
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
        pos = self._find_pos(event.widget)
        if not pos:
            return "break"
        i, j = pos
        if event.keysym == "Up":
            i2, j2 = max(0, i - 1), j
        elif event.keysym == "Down":
            i2, j2 = min(len(self.entries) - 1, i + 1), j
        elif event.keysym == "Left":
            i2, j2 = i, max(0, j - 1)
        elif event.keysym == "Right":
            i2, j2 = i, min(len(self.entries[i]) - 1, j + 1)
        else:
            return "break"
        self.entries[i2][j2].focus_set()
        self.entries[i2][j2].icursor(tk.END)
        return "break"

    # -------------------------
    # Clipboard: вставка/копіювання
    # -------------------------
    def _paste_into_widget(self, widget):
        try:
            data = self.table_win.clipboard_get()
        except Exception:
            return

        rows = [r for r in data.splitlines() if r != ""]
        if not rows:
            return

        pos = self._find_pos(widget)
        if not pos:
            return
        r0, c0 = pos

        for i_r, row_text in enumerate(rows):
            cols = row_text.split("\t")
            for j_c, val in enumerate(cols):
                rr = r0 + i_r
                cc = c0 + j_c
                while rr >= len(self.entries):
                    self.add_row()
                while cc >= len(self.entries[rr]):
                    self.add_repeat_column()
                self.entries[rr][cc].delete(0, tk.END)
                self.entries[rr][cc].insert(0, val)

    def paste_from_clipboard_button(self):
        w = self.table_win.focus_get()
        if isinstance(w, tk.Entry):
            self._paste_into_widget(w)
        else:
            self._paste_into_widget(self.entries[0][0])

    def on_paste_cell(self, event=None):
        self._paste_into_widget(event.widget)
        return "break"

    def on_paste_global(self, event=None):
        w = self.table_win.focus_get()
        if isinstance(w, tk.Entry):
            self._paste_into_widget(w)
        return "break"

    def on_copy_cell(self, event=None):
        w = event.widget
        try:
            txt = w.selection_get()
        except Exception:
            txt = w.get()
        self.table_win.clipboard_clear()
        self.table_win.clipboard_append(txt)
        return "break"

    def on_copy_global(self, event=None):
        w = self.table_win.focus_get()
        if isinstance(w, tk.Entry):
            return self.on_copy_cell(event)
        return "break"

    # -------------------------
    # Дані: активні повтори (ігнор порожніх стовпчиків)
    # -------------------------
    def _active_repeat_columns(self):
        start = self.factors_count
        end = len(self.headers)
        active = []
        for col in range(start, end):
            found = False
            for row in self.entries:
                v = safe_float(row[col].get())
                if not (isinstance(v, float) and math.isnan(v)):
                    found = True
                    break
            if found:
                active.append(col)
        return active

    def collect_long(self, factor_keys):
        active_rep_cols = self._active_repeat_columns()
        if not active_rep_cols:
            return [], active_rep_cols

        long = []
        for i, row in enumerate(self.entries):
            levels = []
            for k in range(self.factors_count):
                s = row[k].get().strip()
                if s == "":
                    s = f"рядок_{i+1}"
                levels.append(s)

            for col in active_rep_cols:
                v = safe_float(row[col].get())
                if isinstance(v, float) and math.isnan(v):
                    continue
                rec = {"value": float(v)}
                for idx, fk in enumerate(factor_keys):
                    rec[fk] = levels[idx]
                long.append(rec)

        return long, active_rep_cols

    # -------------------------
    # Аналіз
    # -------------------------
    def analyze(self):
        params = ask_indicator_units(self.table_win if self.table_win else self.root)
        if not params:
            return
        indicator, units = params

        factor_keys = ["A", "B", "C", "D"][: self.factors_count]
        long, active_rep_cols = self.collect_long(factor_keys)

        if len(long) < 3:
            messagebox.showwarning("Помилка", "Недостатньо числових даних для аналізу.")
            return

        try:
            anova_rows, MS_error, df_error, SS_total, df_total, level_orders, means = anova_ols(long, factor_keys)
        except Exception as e:
            messagebox.showerror("Помилка", str(e))
            return

        # Залишки (Shapiro-Wilk): y - mean по комірці (комбінації всіх факторів)
        cell_vals = {}
        for r in long:
            key = tuple(r[k] for k in factor_keys)
            cell_vals.setdefault(key, []).append(r["value"])
        cell_means = {k: float(np.mean(v)) for k, v in cell_vals.items()}

        residuals = np.array([r["value"] - cell_means[tuple(r[k] for k in factor_keys)] for r in long], dtype=float)
        W, p_norm = (np.nan, np.nan)
        if len(residuals) >= 3:
            try:
                W, p_norm = shapiro(residuals)
            except Exception:
                W, p_norm = (np.nan, np.nan)

        # Параметри досліду
        N_obs = len(long)
        rep_count_active = len(active_rep_cols)

        # Сила впливу (η², %)
        strength = []
        for r in anova_rows:
            if r["name"] in ("Залишок", "Загальна"):
                continue
            eta2 = (r["SS"] / SS_total) if SS_total > 0 else np.nan
            strength.append((r["name"], eta2))

        # НІР₀₅: загальна + по факторах
        nir_rows = []
        cell_counts = [len(v) for v in cell_vals.values()]
        n_cell_mean = float(np.mean(cell_counts)) if cell_counts else np.nan
        nir_rows.append(("Загальна (по досліду)", nir05(MS_error, df_error, n_cell_mean if n_cell_mean > 0 else np.nan)))

        for fk in factor_keys:
            k_levels = len(level_orders[fk])
            n_eff = (N_obs / k_levels) if k_levels > 0 else np.nan
            nir_rows.append((f"Фактор {fk}", nir05(MS_error, df_error, n_eff if n_eff > 0 else np.nan)))

        # Формування звіту (без графіків)
        report_text = self.build_report(
            indicator, units, factor_keys,
            level_orders, rep_count_active, N_obs,
            W, p_norm,
            anova_rows, strength,
            nir_rows,
            means
        )

        self.show_report(report_text)

    def build_report(
        self,
        indicator, units, factor_keys,
        level_orders, rep_count_active, N_obs,
        W, p_norm,
        anova_rows, strength,
        nir_rows,
        means
    ):
        title_map = {1: "ОДНОФАКТОРНОГО", 2: "ДВОФАКТОРНОГО", 3: "ТРИФАКТОРНОГО", 4: "ЧОТИРИФАКТОРНОГО"}
        header = f"Р Е З У Л Ь Т А Т И   {title_map.get(len(factor_keys), '')}   Д И С П Е Р С І Й Н О Г О   А Н А Л І З У"

        lines = []
        lines.append(header)
        lines.append("")
        lines.append(f"Показник: {indicator}")
        lines.append(f"Одиниці виміру: {units}")
        lines.append("")

        uk = {"A": "А", "B": "В", "C": "С", "D": "D"}
        for fk in factor_keys:
            levels = level_orders[fk]
            lines.append(f"Фактор {uk[fk]}: {', '.join(levels)}")
        lines.append("")
        lines.append(f"Кількість активних повторень: {rep_count_active}")
        lines.append(f"Загальна кількість облікових значень: {N_obs}")
        lines.append("")

        if not (isinstance(W, float) and math.isnan(W)):
            norm_text = "нормальний" if (not math.isnan(p_norm) and p_norm > 0.05) else "НЕ нормальний"
            lines.append(f"Перевірка нормальності залишків (Shapiro-Wilk): {norm_text} (W = {W:.4f}, p = {p_norm:.4f})")
        else:
            lines.append("Перевірка нормальності залишків (Shapiro-Wilk): недостатньо даних")
        lines.append("")
        lines.append("Позначення у таблиці: * — p < 0.05; ** — p < 0.01")
        lines.append("")

        # Таблиця ANOVA (вирівняна)
        cols = ["Джерело варіації", "Сума квадратів", "Ступені свободи", "Середній квадрат", "Fрозрах.", "p", "Fтабл.", "Висновок"]
        widths = [32, 16, 16, 18, 12, 10, 10, 18]

        def row_fmt(vals):
            s = ""
            for v, w in zip(vals, widths):
                txt = v if isinstance(v, str) else str(v)
                if len(txt) > w:
                    txt = txt[:w - 1] + "…"
                s += txt.ljust(w) + " "
            return s.rstrip()

        sep = "-" * (sum(widths) + len(widths) - 1)
        lines.append(sep)
        lines.append(row_fmt(cols))
        lines.append(sep)

        for r in anova_rows:
            mark = p_mark(r.get("p"))
            f_with_mark = (fmt_num(r.get("F"), 3) + mark) if not (isinstance(r.get("F"), float) and math.isnan(r.get("F"))) else ""
            vals = [
                r.get("name", ""),
                fmt_num(r.get("SS"), 2),
                fmt_int(r.get("df")),
                fmt_num(r.get("MS"), 4),
                f_with_mark,
                fmt_num(r.get("p"), 4),
                fmt_num(r.get("Fcrit"), 2),
                r.get("conclusion", "")
            ]
            lines.append(row_fmt(vals))

        lines.append(sep)
        lines.append("")

        # Сила впливу
        lines.append("Сила впливу (η², %):")
        for name, eta2 in strength:
            if isinstance(eta2, float) and math.isnan(eta2):
                continue
            lines.append(f"  • {name} — {eta2 * 100:5.1f}%")
        lines.append("")

        # Таблиця НІР₀₅
        lines.append("Таблиця НІР₀₅:")
        lines.append("-" * 46)
        lines.append("Порівняння".ljust(30) + "НІР₀₅".rjust(14))
        lines.append("-" * 46)
        for name, v in nir_rows:
            vv = "" if (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"
            lines.append(name.ljust(30) + vv.rjust(14))
        lines.append("-" * 46)
        lines.append("")

        # Середні (у порядку введення)
        for fk in factor_keys:
            lines.append(f"Середні по фактору {uk[fk]}:")
            for (lv, m, n) in means[fk]:
                mtxt = "" if (isinstance(m, float) and math.isnan(m)) else f"{m:.4f}"
                lines.append(f"  {lv:<24} {mtxt:>12}   (n={n})")
            lines.append("")

        # Рядок про "Залишок"
        for r in anova_rows:
            if r.get("name") == "Залишок":
                lines.append(f"Залишок: сума квадратів = {r['SS']:.4f}; ступені свободи = {int(r['df'])}; середній квадрат = {r['MS']:.6f}")
                break

        lines.append("")
        lines.append("Кінець звіту.")
        return "\n".join(lines)

    # -------------------------
    # Показ звіту (копіювання в Word)
    # -------------------------
    def show_report(self, report_text: str):
        win = tk.Toplevel(self.root)
        win.title("S.A.D. — Звіт")
        win.geometry("1100x720")
        win.configure(bg="white")

        top = ttk.Frame(win, padding=(12, 10))
        top.pack(fill=tk.X)

        ttk.Label(top, text="Звіт можна копіювати та вставляти у Word.",
                  font=("Segoe UI", 12, "bold"), foreground="black").pack(side=tk.LEFT)

        def copy_all():
            data = txt.get("1.0", "end-1c")
            win.clipboard_clear()
            win.clipboard_append(data)

        ttk.Button(top, text="Копіювати звіт", command=copy_all).pack(side=tk.RIGHT)

        txt = ScrolledText(win, wrap="none", font=("Times New Roman", 14), fg="black", bg="white")
        txt.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        txt.insert("1.0", report_text)
        txt.config(state=tk.NORMAL)

        def select_all(event=None):
            txt.tag_add("sel", "1.0", "end")
            return "break"

        def copy_sel(event=None):
            try:
                data = txt.get("sel.first", "sel.last")
            except Exception:
                data = txt.get("1.0", "end-1c")
            win.clipboard_clear()
            win.clipboard_append(data)
            return "break"

        txt.bind("<Control-a>", select_all)
        txt.bind("<Control-A>", select_all)
        txt.bind("<Control-c>", copy_sel)
        txt.bind("<Control-C>", copy_sel)

    # -------------------------
    # Про розробника
    # -------------------------
    def show_about(self):
        messagebox.showinfo(
            "Про розробника",
            "S.A.D. — Статистичний аналіз даних\n"
            "Версія: 1.0\n"
            "Розробник: Чаплоуцький А.М."
        )


# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = SADApp(root)
    root.mainloop()
