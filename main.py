# main.py
# -*- coding: utf-8 -*-
"""
S.A.D. — Статистичний аналіз даних (Tkinter)
Потрібно: Python 3.8+, numpy, scipy, matplotlib
"""

import math
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
from itertools import combinations

import numpy as np
from scipy.stats import shapiro, t as t_dist, f as f_dist

# Matplotlib (для гістограм)
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


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
    # Для таблиці ANOVA (окремо пояснюється в звіті)
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def is_significant(p):
    return (p is not None) and (not (isinstance(p, float) and math.isnan(p))) and (p < 0.05)


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

    ttk.Label(frm, text="Назва показника:", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
    e1 = ttk.Entry(frm, width=42)
    e1.grid(row=1, column=0, sticky="we", pady=(4, 10))

    ttk.Label(frm, text="Одиниці виміру:", font=("Segoe UI", 11, "bold")).grid(row=2, column=0, sticky="w")
    e2 = ttk.Entry(frm, width=42)
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

    dlg.columnconfigure(0, weight=1)
    frm.columnconfigure(0, weight=1)

    e1.focus_set()
    parent.wait_window(dlg)

    if result["ok"]:
        return result["indicator"], result["units"]
    return None


# -------------------------
# Побудова дизайну для ANOVA (OLS, послідовні SS — Type I)
# Підтримка 1–4 факторів (A,B,C,D) з взаємодіями до повного порядку.
# -------------------------
def _one_hot(indices, n_levels, drop_first=True):
    """
    indices: array shape (N,)
    returns: matrix shape (N, n_levels-1) якщо drop_first, інакше (N, n_levels)
    """
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
    """
    Елементні добутки всіх комбінацій стовпчиків заданих матриць.
    matrices: list of (N, p_i)
    returns: (N, prod p_i)
    """
    if any(m.shape[1] == 0 for m in matrices):
        return np.zeros((matrices[0].shape[0], 0), dtype=float)
    M = matrices[0]
    for nxt in matrices[1:]:
        # комбінуємо стовпчики
        cols = []
        for i in range(M.shape[1]):
            for j in range(nxt.shape[1]):
                cols.append((M[:, i] * nxt[:, j])[:, None])
        M = np.hstack(cols) if cols else np.zeros((M.shape[0], 0), dtype=float)
    return M


def _ols_sse(X, y):
    """
    Повертає SSE і rank для моделі y ~ X
    """
    if X.shape[1] == 0:
        # модель без предикторів
        yhat = np.zeros_like(y)
        resid = y - yhat
        return float(np.sum(resid ** 2)), 0
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    sse = float(np.sum(resid ** 2))
    rank = int(np.linalg.matrix_rank(X))
    return sse, rank


def anova_ols(long_records, factor_keys):
    """
    long_records: list[dict], keys: factor_keys + 'value'
    factor_keys: ['A']..['A','B','C','D']
    Повертає:
      - table_rows: list of dict with fields (name, SS, df, MS, F, p, Fcrit, conclusion)
      - MS_error, df_error, SS_total, df_total
      - level_orders: dict factor -> list levels in encounter order
      - means: dict factor -> list of (level, mean, n)
    """
    y = np.array([r["value"] for r in long_records], dtype=float)
    N = len(y)
    if N < 3:
        raise ValueError("Недостатньо даних (менше 3 значень).")

    grand_mean = float(np.mean(y))
    SS_total = float(np.sum((y - grand_mean) ** 2))
    df_total = N - 1

    # Порядок рівнів — як у введенні (не сортуємо)
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

    # Базові dummy-матриці (drop first)
    dummies = {}
    for fk in factor_keys:
        n_lv = len(level_orders[fk])
        dummies[fk] = _one_hot(idx_arrays[fk], n_lv, drop_first=True)

    # Порядок ефектів (ієрархічно, як звично в агрономічних звітах)
    effects = []
    # main
    for fk in factor_keys:
        effects.append((fk,))
    # interactions 2..k
    for r in range(2, len(factor_keys) + 1):
        for comb in combinations(factor_keys, r):
            effects.append(tuple(comb))

    # Збираємо дизайн покроково (Type I)
    X0 = np.ones((N, 1), dtype=float)  # intercept
    sse_prev, rank_prev = _ols_sse(X0, y)
    X_curr = X0.copy()

    rows = []
    effect_cols_count = {}  # для df ефекту

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
            # якщо фактор має 1 рівень, df=0
            continue

        X_next = np.hstack([X_curr, block])
        sse_next, rank_next = _ols_sse(X_next, y)

        SS_eff = max(0.0, sse_prev - sse_next)
        effect_cols_count[eff] = df_eff

        rows.append({
            "name": eff_name,
            "SS": SS_eff,
            "df": df_eff,
        })

        X_curr = X_next
        sse_prev, rank_prev = sse_next, rank_next

    # Похибка/залишок
    sse_full = sse_prev
    rank_full = rank_prev
    df_error = N - rank_full
    if df_error <= 0:
        raise ValueError("Неможливо оцінити похибку: df_error ≤ 0 (перевірте дані).")
    MS_error = sse_full / df_error

    # Обчислюємо F, p, Fcrit і висновок
    alpha = 0.05
    for r in rows:
        MS_eff = r["SS"] / r["df"] if r["df"] > 0 else np.nan
        Fv = MS_eff / MS_error if MS_error > 0 else np.nan
        pv = 1.0 - f_dist.cdf(Fv, r["df"], df_error) if not math.isnan(Fv) else np.nan
        Fcrit = f_dist.ppf(1.0 - alpha, r["df"], df_error)
        r["MS"] = MS_eff
        r["F"] = Fv
        r["p"] = pv
        r["Fcrit"] = Fcrit
        r["conclusion"] = "істотна різниця" if (not math.isnan(pv) and pv < 0.05) else "неістотна"

    # Додаємо залишок і загальну
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

    # Маргінальні середні (в порядку введення)
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
# НІР₀₅ та буквені групи (просте й практичне групування)
# -------------------------
def nir05_for_factor(MS_error, df_error, n_per_mean):
    """
    НІР₀₅ для порівняння двох середніх (LSD 0.05) = t(0.975)*sqrt(2*MSe/n)
    n_per_mean — ефективна кількість спостережень на середнє.
    """
    if df_error <= 0 or MS_error <= 0 or n_per_mean <= 0:
        return np.nan
    tval = t_dist.ppf(0.975, df_error)
    return float(tval * math.sqrt(2.0 * MS_error / n_per_mean))


def letter_groups_in_order(means_in_order, nir):
    """
    means_in_order: list of floats in table order
    nir: threshold for significant difference
    Повертає list літер (a,b,c,...), у порядку введення.
    Простий “агрономічний” підхід: якщо нове середнє істотно відрізняється
    від будь-кого з поточної групи — нова літера.
    """
    letters = []
    groups = []  # list of lists of indices that share a letter
    alpha_letters = "abcdefghijklmnopqrstuvwxyz"

    for i, mi in enumerate(means_in_order):
        if math.isnan(mi) or math.isnan(nir):
            letters.append("")
            continue

        placed = False
        for g_idx, g in enumerate(groups):
            ok = True
            for j in g:
                mj = means_in_order[j]
                if not math.isnan(mj) and abs(mi - mj) > nir:
                    ok = False
                    break
            if ok:
                g.append(i)
                letters.append(alpha_letters[g_idx] if g_idx < len(alpha_letters) else f"g{g_idx}")
                placed = True
                break

        if not placed:
            groups.append([i])
            g_idx = len(groups) - 1
            letters.append(alpha_letters[g_idx] if g_idx < len(alpha_letters) else f"g{g_idx}")

    return letters


# -------------------------
# GUI
# -------------------------
class SADApp:
    def __init__(self, root):
        self.root = root
        self.root.title("S.A.D. — Статистичний аналіз даних")
        self.root.geometry("980x520")
        self.root.configure(bg="white")

        # Чіткість/масштаб (Windows)
        try:
            self.root.tk.call("tk", "scaling", 1.2)
        except Exception:
            pass

        # ttk стиль
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        default_font = ("Segoe UI", 12)
        style.configure(".", font=default_font)
        style.configure("TLabel", foreground="black", background="white")
        style.configure("TFrame", background="white")
        style.configure("TButton", padding=8)
        style.configure("Accent.TButton", font=("Segoe UI", 12, "bold"))

        # Головний екран
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
            text="Порада: можна вставляти з Excel кнопкою або Ctrl+V у активну комірку.",
            font=("Segoe UI", 11),
            foreground="black"
        ).pack(pady=(22, 0))

        # Табличне вікно
        self.table_win = None
        self.entries = []
        self.headers = []
        self.factors_count = 1
        self.factor_cols = []
        self.repeat_cols = []
        self.repeat_count = 4

        # Scroll helpers
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

        # Верхня панель керування
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

        # Таблична область зі скролом
        body = ttk.Frame(self.table_win, padding=(12, 0, 12, 12))
        body.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(body, bg="white", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        vsb = ttk.Scrollbar(body, orient="vertical", command=self.canvas.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=vsb.set)

        self.inner = ttk.Frame(self.canvas)
        self.inner.configure(style="TFrame")
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # Початкові колонки
        factor_labels = ["А", "В", "С", "D"]
        self.factor_cols = [f"Фактор {factor_labels[i]}" for i in range(factors_count)]
        self.repeat_count = 4
        self.repeat_cols = [f"Повт.{i+1}" for i in range(self.repeat_count)]
        self.headers = self.factor_cols + self.repeat_cols

        # Заголовки
        self._render_headers()

        # Початкові рядки
        self.entries = []
        for _ in range(10):
            self._append_row()

        # Глобальний Ctrl+V (надійно для Windows)
        self.table_win.bind_all("<Control-v>", self.on_paste_global, add="+")
        self.table_win.bind_all("<Control-V>", self.on_paste_global, add="+")
        # Ctrl+C — копіювати (виділене в Entry)
        self.table_win.bind_all("<Control-c>", self.on_copy_global, add="+")
        self.table_win.bind_all("<Control-C>", self.on_copy_global, add="+")

        # Фокус у першу комірку
        self.entries[0][0].focus_set()

    def _render_headers(self):
        # Очистити попередні заголовки
        for w in list(self.inner.grid_slaves(row=0)):
            w.destroy()

        for j, name in enumerate(self.headers):
            lbl = ttk.Label(
                self.inner,
                text=name,
                anchor="center",
                font=("Segoe UI", 11, "bold"),
                foreground="black",
                background="white"
            )
            # Додамо трохи “повітря”, щоб не злипалося з першим стовпчиком
            lbl.grid(row=0, column=j, padx=(2, 10 if j == 0 else 2), pady=(2, 6), sticky="nsew")

        for j in range(len(self.headers)):
            self.inner.grid_columnconfigure(j, weight=0, minsize=120)

    def _bind_cell(self, e: tk.Entry):
        # Навігація
        e.bind("<Return>", self.on_enter)
        e.bind("<Up>", self.on_arrow)
        e.bind("<Down>", self.on_arrow)
        e.bind("<Left>", self.on_arrow)
        e.bind("<Right>", self.on_arrow)
        # Локальні Ctrl+V/C теж (на всяк випадок)
        e.bind("<Control-v>", self.on_paste_cell)
        e.bind("<Control-V>", self.on_paste_cell)
        e.bind("<Control-c>", self.on_copy_cell)
        e.bind("<Control-C>", self.on_copy_cell)

    def _append_row(self):
        r = len(self.entries) + 1  # +1 через заголовок
        row = []
        for c in range(len(self.headers)):
            e = tk.Entry(self.inner, width=16, fg="black", bg="white",
                         font=("Segoe UI", 12), relief=tk.SOLID, bd=1)
            # Відступ у першого стовпчика, щоб не “липло” до наступного
            e.grid(row=r, column=c, padx=(2, 10 if c == 0 else 2), pady=2)
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
        # додаємо ще один стовпчик повторності
        self.repeat_count += 1
        new_name = f"Повт.{self.repeat_count}"
        self.headers.append(new_name)

        # заголовок
        self._render_headers()

        # додаємо Entry у всі рядки
        col_idx = len(self.headers) - 1
        for i, row in enumerate(self.entries):
            r = i + 1
            e = tk.Entry(self.inner, width=16, fg="black", bg="white",
                         font=("Segoe UI", 12), relief=tk.SOLID, bd=1)
            e.grid(row=r, column=col_idx, padx=2, pady=2)
            self._bind_cell(e)
            row.append(e)

    def delete_repeat_column(self):
        # не можна видаляти факторні стовпчики, і залишимо мінімум 1 повторність
        if self.repeat_count <= 1:
            return
        self.repeat_count -= 1

        # видаляємо останній заголовок і колонку
        col_idx = len(self.headers) - 1

        # Видалити заголовок
        for w in self.inner.grid_slaves(row=0, column=col_idx):
            w.destroy()

        # Видалити комірки
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
                    # якщо вставка ширша — додаємо повторності
                    self.add_repeat_column()
                self.entries[rr][cc].delete(0, tk.END)
                self.entries[rr][cc].insert(0, val)

    def paste_from_clipboard_button(self):
        w = self.table_win.focus_get()
        if isinstance(w, tk.Entry):
            self._paste_into_widget(w)
        else:
            # якщо фокус не в таблиці — у першу комірку
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
    # Збір даних (ігноруємо порожні повторності-стовпчики)
    # -------------------------
    def _active_repeat_columns(self):
        """
        Повертає індекси стовпчиків повторностей (у grid), які мають хоча б одне числове значення.
        """
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
        """
        Повертає long_records для ANOVA:
          {'A':..., 'B':..., 'C':..., 'D':..., 'value': float}
        Ураховує лише активні стовпчики повторностей.
        """
        active_rep_cols = self._active_repeat_columns()
        if not active_rep_cols:
            return [], []

        long = []
        for i, row in enumerate(self.entries):
            # фактори
            levels = []
            for k in range(self.factors_count):
                s = row[k].get().strip()
                if s == "":
                    # якщо фактор порожній — робимо унікальну мітку (щоб не ламати аналіз)
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
        try:
            params = ask_indicator_units(self.table_win if self.table_win else self.root)
            if not params:
                return
            indicator, units = params

            factor_keys = ["A", "B", "C", "D"][: self.factors_count]

            long, active_rep_cols = self.collect_long(factor_keys)
            if len(long) < 3:
                messagebox.showwarning("Помилка", "Недостатньо числових даних для аналізу.")
                return

            # ANOVA (OLS)
            table_rows, MS_error, df_error, SS_total, df_total, level_orders, means = anova_ols(long, factor_keys)

            # Залишки для Shapiro-Wilk:
            # як наближення — залишки з повною моделлю в anova_ols вже є в SSE,
            # але ми їх не повертаємо. Тому порахуємо залишки як (y - mean по комбінації факторів).
            # Для будь-якої кількості факторів беремо “комірку” за всіма факторами.
            cell_map = {}
            cell_vals = {}
            for r in long:
                key = tuple(r[k] for k in factor_keys)
                cell_vals.setdefault(key, []).append(r["value"])
            for k, vals in cell_vals.items():
                cell_map[k] = float(np.mean(vals))

            residuals = np.array([r["value"] - cell_map[tuple(r[k] for k in factor_keys)] for r in long], dtype=float)
            W, p_norm = (np.nan, np.nan)
            if len(residuals) >= 3:
                try:
                    W, p_norm = shapiro(residuals)
                except Exception:
                    W, p_norm = (np.nan, np.nan)

            # Оціночна кількість повторень: активні повтори в середньому
            rep_count_active = len(active_rep_cols)
            N_obs = len(long)

            # Сила впливу (η², %)
            effect_strength = []
            for r in table_rows:
                if r["name"] in ("Залишок", "Загальна"):
                    continue
                eta2 = (r["SS"] / SS_total) if SS_total > 0 else np.nan
                effect_strength.append((r["name"], eta2))

            # НІР₀₅: по кожному фактору + загальна (для порівняння середніх комбінацій)
            nir_table = []
            # Загальна (для порівняння середніх “варіантів” / комбінацій): беремо n як середня кількість у комірці
            cell_counts = [len(v) for v in cell_vals.values()]
            n_cell_mean = float(np.mean(cell_counts)) if cell_counts else np.nan
            nir_overall = nir05_for_factor(MS_error, df_error, n_cell_mean if n_cell_mean > 0 else np.nan)
            nir_table.append(("Загальна (по досліду)", nir_overall))

            # Для факторів: ефективний n на середнє = N / k_levels для цього фактора
            for fk in factor_keys:
                k_levels = len(level_orders[fk])
                n_eff = N_obs / k_levels if k_levels > 0 else np.nan
                nir_fk = nir05_for_factor(MS_error, df_error, n_eff if n_eff > 0 else np.nan)
                nir_table.append((f"Фактор {fk}", nir_fk))

            # Буквені позначення для середніх по кожному фактору (в порядку введення)
            letters_by_factor = {}
            for fk in factor_keys:
                means_list = means[fk]  # (level, mean, n) у порядку введення
                means_vals = [m for (_, m, _) in means_list]
                # беремо НІР для цього фактора з nir_table
                nir_fk = next((v for (name, v) in nir_table if name == f"Фактор {fk}"), np.nan)
                letters = letter_groups_in_order(means_vals, nir_fk)
                letters_by_factor[fk] = letters

            # Побудова звіту (вирівнювання колонок)
            report = self.build_report(
                indicator=indicator,
                units=units,
                factor_keys=factor_keys,
                level_orders=level_orders,
                N_obs=N_obs,
                rep_count_active=rep_count_active,
                W=W,
                p_norm=p_norm,
                anova_rows=table_rows,
                SS_total=SS_total,
                MS_error=MS_error,
                df_error=df_error,
                effect_strength=effect_strength,
                nir_table=nir_table,
                means=means,
                letters_by_factor=letters_by_factor
            )

            # Показ звіту + гістограми
            self.show_report_and_plots(indicator, units, factor_keys, means, letters_by_factor, report)

        except Exception as e:
            messagebox.showerror("Помилка", f"Сталася помилка під час аналізу:\n{e}")
            raise

    def build_report(
        self,
        indicator, units,
        factor_keys,
        level_orders,
        N_obs, rep_count_active,
        W, p_norm,
        anova_rows,
        SS_total, MS_error, df_error,
        effect_strength,
        nir_table,
        means,
        letters_by_factor
    ):
        # Заголовок
        title_map = {1: "ОДНОФАКТОРНОГО", 2: "ДВОФАКТОРНОГО", 3: "ТРИФАКТОРНОГО", 4: "ЧОТИРИФАКТОРНОГО"}
        header = f"Р Е З У Л Ь Т А Т И   {title_map.get(len(factor_keys), '')}   Д И С П Е Р С І Й Н О Г О   А Н А Л І З У"

        lines = []
        lines.append(header)
        lines.append("")
        lines.append(f"Показник: {indicator}")
        lines.append(f"Одиниці виміру: {units}")
        lines.append("")

        # Опис факторів (порядок введення)
        for fk in factor_keys:
            levels = level_orders[fk]
            # Українські літери для виводу
            uk = {"A": "А", "B": "В", "C": "С", "D": "D"}[fk]
            lines.append(f"Фактор {uk}: {len(levels)} градацій")
        lines.append(f"Кількість активних повторень: {rep_count_active}")
        lines.append(f"Загальна кількість облікових значень: {N_obs}")
        lines.append("")

        # Нормальність
        if not (isinstance(W, float) and math.isnan(W)):
            norm_text = "нормальний" if (not math.isnan(p_norm) and p_norm > 0.05) else "НЕ нормальний"
            lines.append(f"Перевірка нормальності залишків (Shapiro-Wilk): {norm_text} (W = {W:.4f}, p = {p_norm:.4f})")
        else:
            lines.append("Перевірка нормальності залишків (Shapiro-Wilk): недостатньо даних")
        lines.append("")
        lines.append("Позначення у таблиці: * — p < 0.05; ** — p < 0.01")
        lines.append("")

        # Таблиця ANOVA (україномовні назви колонок; з відступом)
        # Колонки: Джерело | Сума квадратів | Ступені свободи | Середній квадрат | Fрозрах. | p | Fтабл. | Висновок
        col_names = ["Джерело варіації", "Сума квадратів", "Ступені свободи", "Середній квадрат", "Fрозрах.", "p", "Fтабл.", "Висновок"]
        widths = [30, 14, 16, 16, 10, 10, 10, 18]

        def row_fmt(vals):
            s = ""
            for v, w in zip(vals, widths):
                txt = v if isinstance(v, str) else str(v)
                if len(txt) > w:
                    txt = txt[:w-1] + "…"
                s += txt.ljust(w) + " "
            return s.rstrip()

        sep = "-" * (sum(widths) + len(widths) - 1)

        lines.append(sep)
        lines.append(row_fmt(col_names))
        lines.append(sep)

        for r in anova_rows:
            mark = p_mark(r.get("p"))
            # “істотна різниця” відображаємо в колонці Висновок, а зірочку — біля Fрозрах.
            f_with_mark = (fmt_num(r.get("F"), 3) + mark) if not (isinstance(r.get("F"), float) and math.isnan(r.get("F"))) else ""
            vals = [
                r.get("name", ""),
                fmt_num(r.get("SS"), 2),
                fmt_int(r.get("df")),
                fmt_num(r.get("MS"), 3),
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
        for name, eta2 in effect_strength:
            if isinstance(eta2, float) and math.isnan(eta2):
                continue
            lines.append(f"  • {name} — {eta2*100:5.1f}%")
        lines.append("")

        # Таблиця НІР₀₅
        lines.append("Таблиця НІР₀₅:")
        lines.append("-" * 42)
        lines.append("Порівняння".ljust(26) + "НІР₀₅".rjust(14))
        lines.append("-" * 42)
        for name, v in nir_table:
            vv = "" if (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"
            lines.append(name.ljust(26) + vv.rjust(14))
        lines.append("-" * 42)
        lines.append("")

        # Середні по факторах (у порядку введення, без слова “рівень”)
        uk_map = {"A": "А", "B": "В", "C": "С", "D": "D"}
        for fk in factor_keys:
            lines.append(f"Середні по фактору {uk_map[fk]}:")
            means_list = means[fk]  # (level, mean, n) в порядку
            letters = letters_by_factor.get(fk, [""] * len(means_list))
            for (i, (lv, m, n)) in enumerate(means_list):
                mtxt = "" if (isinstance(m, float) and math.isnan(m)) else f"{m:.3f}"
                ltr = letters[i] if i < len(letters) else ""
                lines.append(f"  {lv:<20}  {mtxt:>10}  {ltr}")
            lines.append("")

        # Пояснення щодо букв
        lines.append("Примітка: однакові букви біля середніх означають відсутність істотної різниці (НІР₀₅).")
        lines.append("")
        lines.append(f"Оцінка похибки: MS(залишок) = {MS_error:.4f}, df(залишок) = {df_error}")
        return "\n".join(lines)

    # -------------------------
    # Вікно звіту + графіки
    # -------------------------
    def show_report_and_plots(self, indicator, units, factor_keys, means, letters_by_factor, report_text):
        win = tk.Toplevel(self.root)
        win.title("S.A.D. — Звіт")
        win.geometry("1150x720")
        win.configure(bg="white")

        # верх: кнопки копіювання
        top = ttk.Frame(win, padding=(12, 10))
        top.pack(fill=tk.X)

        ttk.Label(top, text="Звіт можна копіювати та вставляти у Word без додаткового редагування.",
                  font=("Segoe UI", 12, "bold"), foreground="black").pack(side=tk.LEFT)

        def copy_all():
            txt.tag_add("sel", "1.0", "end")
            data = txt.get("1.0", "end-1c")
            win.clipboard_clear()
            win.clipboard_append(data)

        ttk.Button(top, text="Копіювати звіт", command=copy_all).pack(side=tk.RIGHT)

        # нижче: ліворуч звіт, праворуч графіки
        main = ttk.Frame(win, padding=(12, 0, 12, 12))
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)

        # Звіт (Times New Roman 14)
        txt = ScrolledText(left, wrap="none", font=("Times New Roman", 14), fg="black", bg="white")
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert("1.0", report_text)
        txt.config(state=tk.NORMAL)  # потрібно для копіювання

        # Ctrl+A / Ctrl+C у звіті
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

        # Графіки (по головних факторах)
        # Робимо компактно, пропорційно (не на всю ширину)
        nb = ttk.Notebook(right)
        nb.pack(fill=tk.BOTH, expand=True)

        uk_map = {"A": "А", "B": "В", "C": "С", "D": "D"}
        for fk in factor_keys:
            tab = ttk.Frame(nb, padding=10)
            nb.add(tab, text=f"Фактор {uk_map[fk]}")

            means_list = means[fk]
            labels = [lv for (lv, _, _) in means_list]
            vals = [m for (_, m, _) in means_list]
            letters = letters_by_factor.get(fk, [""] * len(vals))

            fig = Figure(figsize=(5.2, 3.6), dpi=110)  # пропорційно, не розтягуємо
            ax = fig.add_subplot(111)

            x = np.arange(len(vals))
            ax.bar(x, vals)

            ax.set_title(f"{indicator} ({units}) — Фактор {uk_map[fk]}", fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
            ax.tick_params(axis="y", labelsize=9)

            # підписи значень + букви
            for i, v in enumerate(vals):
                if isinstance(v, float) and math.isnan(v):
                    continue
                ax.text(i, v, f"{v:.2f}\n{letters[i]}", ha="center", va="bottom", fontsize=9)

            fig.tight_layout()

            canvas = FigureCanvasTkAgg(fig, master=tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # -------------------------
    # Про розробника
    # -------------------------
    def show_about(self):
        messagebox.showinfo(
            "Про розробника",
            "S.A.D. — Статистичний аналіз даних\n"
            "Версія: 1.0\n"
            "Розроблено для агрономічних/садівничих дослідів.\n"
            "Контакти/автор: (вкажіть за потреби)"
        )


# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = SADApp(root)
    root.mainloop()
