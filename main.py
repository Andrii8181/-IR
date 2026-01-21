# main.py  (ЧАСТИНА 1/2)
# -*- coding: utf-8 -*-

"""
S.A.D. — Статистичний аналіз даних (Tkinter)

Потрібно: Python 3.8+, numpy, scipy
Встановлення: pip install numpy scipy
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import tkinter.font as tkfont

import math
import numpy as np
from itertools import combinations
from collections import defaultdict
from datetime import datetime

from scipy.stats import shapiro, t, f as f_dist
from scipy.stats import mannwhitneyu, kruskal, levene, rankdata
from scipy.stats import studentized_range
from scipy.stats import friedmanchisquare, wilcoxon, norm

ALPHA = 0.05
COL_W = 10

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
# ICON (robust search)
# -------------------------
def _script_dir():
    try:
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            return sys._MEIPASS
    except Exception:
        pass
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except Exception:
        return os.getcwd()

def _exe_dir():
    try:
        return os.path.dirname(os.path.abspath(sys.executable))
    except Exception:
        return None

def _argv0_dir():
    try:
        p = sys.argv[0]
        if not p:
            return None
        return os.path.dirname(os.path.abspath(p))
    except Exception:
        return None

def _find_icon_file():
    candidates = []
    for base in [_script_dir(), os.getcwd(), _argv0_dir(), _exe_dir()]:
        if base:
            candidates.append(os.path.join(base, "icon.ico"))
    try:
        if hasattr(sys, "_MEIPASS"):
            candidates.append(os.path.join(sys._MEIPASS, "icon.ico"))
    except Exception:
        pass
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None

def set_window_icon(win: tk.Tk | tk.Toplevel):
    ico = _find_icon_file()
    if not ico:
        return
    try:
        win.iconbitmap(ico)
        return
    except Exception:
        pass
    try:
        win.iconbitmap(default=ico)
        return
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

def median_q1_q3(arr):
    if arr is None or len(arr) == 0:
        return (np.nan, np.nan, np.nan)
    a = np.array(arr, dtype=float)
    med = float(np.median(a))
    q1 = float(np.percentile(a, 25))
    q3 = float(np.percentile(a, 75))
    return med, q1, q3

def cv_percent_from_values(values):
    a = np.array(values, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) < 2:
        return np.nan
    m = float(np.mean(a))
    if m == 0:
        return np.nan
    sd = float(np.std(a, ddof=1))
    return (sd / m) * 100.0

def cv_percent_from_level_means(level_means):
    vals = [float(x) for x in level_means if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(vals) < 2:
        return np.nan
    m = float(np.mean(vals))
    if m == 0:
        return np.nan
    sd = float(np.std(vals, ddof=1))
    return (sd / m) * 100.0

def partial_eta2_label(pe2: float) -> str:
    if pe2 is None or (isinstance(pe2, float) and math.isnan(pe2)):
        return ""
    if pe2 < 0.01:
        return "дуже слабкий"
    if pe2 < 0.06:
        return "слабкий"
    if pe2 < 0.14:
        return "середній"
    return "сильний"

def epsilon_squared_kw(H: float, n: int, k: int) -> float:
    if any(x is None for x in [H, n, k]):
        return np.nan
    if isinstance(H, float) and math.isnan(H):
        return np.nan
    if n <= k or k < 2:
        return np.nan
    return float((H - k + 1.0) / (n - k))

def kendalls_w_from_friedman(chisq: float, n_blocks: int, k_treat: int) -> float:
    if any(x is None for x in [chisq, n_blocks, k_treat]):
        return np.nan
    if isinstance(chisq, float) and math.isnan(chisq):
        return np.nan
    if n_blocks <= 0 or k_treat <= 1:
        return np.nan
    return float(chisq / (n_blocks * (k_treat - 1)))

def cliffs_delta(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan
    gt = 0
    lt = 0
    for xi in x:
        gt += int(np.sum(xi > y))
        lt += int(np.sum(xi < y))
    return float((gt - lt) / (nx * ny))

def cliffs_label(delta_abs: float) -> str:
    if delta_abs is None or (isinstance(delta_abs, float) and math.isnan(delta_abs)):
        return ""
    if delta_abs < 0.147:
        return "дуже слабкий"
    if delta_abs < 0.33:
        return "слабкий"
    if delta_abs < 0.474:
        return "середній"
    return "сильний"

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

    uniq = []
    for g in groups:
        if any(g == h for h in uniq):
            continue
        uniq.append(g)

    cleaned = []
    for g in uniq:
        if any((g < h) for h in uniq):
            continue
        cleaned.append(g)
    groups = cleaned

    alphabet = "abcdefghijklmnopqrstuvwxyz"
    letters_for_group = []
    for idx in range(len(groups)):
        letters_for_group.append(alphabet[idx] if idx < len(alphabet) else f"g{idx}")

    mapping = {lvl: [] for lvl in sorted_lvls}
    for gi, grp in enumerate(groups):
        letter = letters_for_group[gi]
        for lvl in grp:
            mapping[lvl].append(letter)

    out = {}
    for lvl in levels_order:
        out[lvl] = "".join(sorted(mapping.get(lvl, [])))
    return out

def fit_font_size_to_texts(texts, family="Times New Roman", start=13, min_size=9, target_px=155):
    f = tkfont.Font(family=family, size=start)
    size = start
    while size > min_size:
        maxw = max(f.measure(t) for t in texts)
        if maxw <= target_px:
            break
        size -= 1
        f.configure(size=size)
    return f

# -------------------------
# REPORT TABLES (tabs)
# -------------------------
def build_table_block(headers, rows):
    def hcell(x): return f"{x} "
    def ccell(x): return "" if x is None else str(x)

    lines = []
    lines.append("\t".join(hcell(h) for h in headers))
    lines.append("\t".join("—" * max(3, len(str(h))) for h in headers))
    for r in rows:
        lines.append("\t".join(ccell(v) for v in r))
    return "\n".join(lines) + "\n"

def tabs_from_table_px(font_obj: tkfont.Font, headers, rows, padding_px=32, extra_gap_after_col=None, extra_gap_px=0):
    ncol = len(headers)
    maxw = [0] * ncol

    for j in range(ncol):
        maxw[j] = font_obj.measure(str(headers[j]))

    for r in rows:
        for j in range(ncol):
            s = "" if r[j] is None else str(r[j])
            w = font_obj.measure(s)
            if w > maxw[j]:
                maxw[j] = w

    widths = [w + padding_px for w in maxw]

    tabs = []
    acc = 0
    for j, w in enumerate(widths[:-1]):
        acc += w
        if extra_gap_after_col is not None and j == extra_gap_after_col:
            acc += extra_gap_px
        tabs.append(acc)

    return tuple(tabs)

# -------------------------
# CRD / RCBD / Split-plot: функції аналізу (без змін)
# -------------------------
# (Тут залишаються всі твої функції: anova_n_way, anova_rcbd_ols, anova_splitplot_ols, pairwise тощо)
# Вони йдуть у частині 2/2, щоб не дублювати і не ламати структуру.

# -------------------------
# GUI
# -------------------------
class SADTk:
    def __init__(self, root):
        self.root = root
        root.title("S.A.D. — Статистичний аналіз даних")
        root.geometry("1000x560")
        set_window_icon(root)

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

        tk.Label(
            self.main_frame,
            text="S.A.D. — Статистичний аналіз даних",
            font=("Times New Roman", 20, "bold"),
            fg="#000000",
            bg="white"
        ).pack(pady=18)

        btn_frame = tk.Frame(self.main_frame, bg="white")
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Однофакторний аналіз", width=22, height=2,
                  command=lambda: self.open_table(1)).grid(row=0, column=0, padx=10, pady=8)
        tk.Button(btn_frame, text="Двофакторний аналіз", width=22, height=2,
                  command=lambda: self.open_table(2)).grid(row=0, column=1, padx=10, pady=8)
        tk.Button(btn_frame, text="Трифакторний аналіз", width=22, height=2,
                  command=lambda: self.open_table(3)).grid(row=1, column=0, padx=10, pady=8)
        tk.Button(btn_frame, text="Чотирифакторний аналіз", width=22, height=2,
                  command=lambda: self.open_table(4)).grid(row=1, column=1, padx=10, pady=8)

        tk.Label(
            self.main_frame,
            text="Виберіть тип аналізу → Внесіть дані → Натисніть «Аналіз даних»",
            fg="#000000",
            bg="white"
        ).pack(pady=10)

        self.table_win = None
        self.report_win = None

        # Active cell highlighting
        self._active_cell = None
        self._active_prev_cfg = None

        # Fill-handle drag
        self._fill_ready = False
        self._fill_dragging = False
        self._fill_src_pos = None   # (r, c)
        self._fill_src_text = ""
        self._fill_last_row = None

    # ---------- Назви факторів (для звіту) ----------
    def factor_title(self, fkey: str) -> str:
        if not hasattr(self, "factor_title_map"):
            return f"Фактор {fkey}"
        return self.factor_title_map.get(fkey, f"Фактор {fkey}")

    def _set_factor_title(self, fkey: str, title: str):
        if not hasattr(self, "factor_title_map"):
            self.factor_title_map = {}
        self.factor_title_map[fkey] = title.strip()

    # ---------- Вікно підказки по дизайнам ----------
    def show_design_help(self):
        w = tk.Toplevel(self.root)
        w.title("Пояснення дизайнів")
        w.resizable(False, False)
        set_window_icon(w)

        frm = tk.Frame(w, padx=16, pady=14)
        frm.pack(fill=tk.BOTH, expand=True)

        txt = (
            "CRD (повна рандомізація)\n"
            "• Усі варіанти розміщені випадково без блоків.\n"
            "• Підходить, коли ділянка однорідна.\n\n"
            "RCBD (блочна рандомізація)\n"
            "• Є блоки (повторності), всередині блоку — всі варіанти.\n"
            "• Підходить, коли є градієнт поля (рельєф, ґрунт тощо).\n\n"
            "Split-plot (спліт-плот)\n"
            "• Є головний фактор (whole-plot) і підплощі (sub-plot) всередині.\n"
            "• Для головного фактора використовується інша (більша) помилка, ніж для підфакторів.\n"
        )

        t = tk.Text(frm, width=62, height=16, wrap="word")
        t.insert("1.0", txt)
        t.configure(state="disabled")
        t.pack(fill=tk.BOTH, expand=True)

        btns = tk.Frame(frm)
        btns.pack(fill=tk.X, pady=(10, 0))
        tk.Button(btns, text="OK", width=10, command=w.destroy).pack(side=tk.LEFT, padx=6)

        w.update_idletasks()
        center_window(w)
        w.grab_set()

    def ask_indicator_units(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Параметри звіту")
        dlg.resizable(False, False)
        set_window_icon(dlg)

        frm = tk.Frame(dlg, padx=16, pady=16)
        frm.pack(fill=tk.BOTH, expand=True)

        tk.Label(frm, text="Назва показника:", fg="#000000").grid(row=0, column=0, sticky="w", pady=6)
        e_ind = tk.Entry(frm, width=40, fg="#000000")
        e_ind.grid(row=0, column=1, pady=6)

        tk.Label(frm, text="Одиниці виміру:", fg="#000000").grid(row=1, column=0, sticky="w", pady=6)
        e_units = tk.Entry(frm, width=40, fg="#000000")
        e_units.grid(row=1, column=1, pady=6)

        # Дизайн + кнопка "?"
        row_design = tk.Frame(frm)
        row_design.grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 4))

        tk.Label(row_design, text="Дизайн експерименту:", fg="#000000").pack(side=tk.LEFT)

        btn_q = tk.Button(
            row_design, text=" ? ", width=3,
            command=self.show_design_help
        )
        btn_q.pack(side=tk.LEFT, padx=(8, 0))

        design_var = tk.StringVar(value="crd")
        rfrm = tk.Frame(frm)
        rfrm.grid(row=2, column=1, sticky="w", pady=(10, 4), padx=(220, 0))

        rb_font = ("Times New Roman", 16)

        tk.Radiobutton(rfrm, text="Повна рандомізація (CRD)", variable=design_var, value="crd",
                       font=rb_font).pack(anchor="w", pady=2)
        tk.Radiobutton(rfrm, text="Блочна рандомізація (RCBD)", variable=design_var, value="rcbd",
                       font=rb_font).pack(anchor="w", pady=2)
        tk.Radiobutton(rfrm, text="Спліт-плот (Split-plot) — лише параметричний", variable=design_var, value="split",
                       font=rb_font).pack(anchor="w", pady=2)

        # Split-plot main factor
        main_factor_var = tk.StringVar(value="A")

        sp_frame = tk.Frame(frm)
        sp_frame.grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))

        lbl_sp = tk.Label(sp_frame, text="Головний фактор (Whole-plot factor):", fg="#000000")
        lbl_sp.pack(side=tk.LEFT)

        cmb = ttk.Combobox(sp_frame, textvariable=main_factor_var, width=8, state="readonly",
                           values=("A", "B", "C", "D"))
        cmb.pack(side=tk.LEFT, padx=8)

        def _fit_dialog_to_content():
            dlg.update_idletasks()
            req_w = frm.winfo_reqwidth()
            req_h = frm.winfo_reqheight()
            dlg.geometry(f"{req_w + 24}x{req_h + 18}")
            center_window(dlg)

        def sp_visible(is_on: bool):
            if is_on:
                sp_frame.grid()
            else:
                sp_frame.grid_remove()
            _fit_dialog_to_content()

        sp_visible(False)

        def on_design_change(*_):
            sp_visible(design_var.get() == "split")

        design_var.trace_add("write", on_design_change)

        out = {"ok": False, "indicator": "", "units": "", "design": "crd", "split_main": "A"}

        def on_ok():
            out["indicator"] = e_ind.get().strip()
            out["units"] = e_units.get().strip()
            out["design"] = design_var.get()
            out["split_main"] = main_factor_var.get()

            if not out["indicator"] or not out["units"]:
                messagebox.showwarning("Помилка", "Заповніть назву показника та одиниці виміру.")
                return

            out["ok"] = True
            dlg.destroy()

        btns = tk.Frame(frm)
        btns.grid(row=4, column=0, columnspan=2, pady=(12, 0))
        tk.Button(btns, text="OK", width=10, command=on_ok).pack(side=tk.LEFT, padx=6)
        tk.Button(btns, text="Скасувати", width=12, command=lambda: dlg.destroy()).pack(side=tk.LEFT, padx=6)

        _fit_dialog_to_content()
        e_ind.focus_set()
        dlg.bind("<Return>", lambda e: on_ok())
        dlg.grab_set()
        self.root.wait_window(dlg)
        return out

    def choose_method_window(self, p_norm, design, num_variants):
        dlg = tk.Toplevel(self.root)
        dlg.title("Вибір виду аналізу")
        dlg.resizable(False, False)
        set_window_icon(dlg)

        frm = tk.Frame(dlg, padx=16, pady=14)
        frm.pack(fill=tk.BOTH, expand=True)

        normal = (p_norm is not None) and (not math.isnan(p_norm)) and (p_norm > 0.05)
        rb_font = ("Times New Roman", 16)

        if normal:
            msg = ("Дані експерименту відповідають принципам нормального розподілу\n"
                   "за методом Шапіра-Вілка.")
            tk.Label(frm, text=msg, fg="#000000", justify="left").pack(anchor="w", pady=(0, 10))
            options = [
                ("НІР₀₅", "lsd"),
                ("Тест Тьюкі", "tukey"),
                ("Тест Дункана", "duncan"),
                ("Тест Бонферроні", "bonferroni"),
            ]
        else:
            if design == "split":
                tk.Label(
                    frm,
                    text=("Для спліт-плоту в цій програмі реалізовано лише параметричний аналіз.\n"
                          "Оскільки залишки моделі не відповідають нормальному розподілу (p ≤ 0.05),\n"
                          "параметричний split-plot аналіз є методично некоректним.\n\n"
                          "Що можна зробити:\n"
                          "• спробувати трансформацію даних (log/√/Box-Cox) і повторити;\n"
                          "• або вибрати CRD/RCBD і виконати непараметричний аналіз."),
                    fg="#c62828",
                    justify="left"
                ).pack(anchor="w", pady=(0, 10))
                options = []
            else:
                msg = ("Дані експерименту не відповідають принципам нормального розподілу\n"
                       "за методом Шапіра-Вілка.\n"
                       "Виберіть один з непараметричних типів аналізу.")
                tk.Label(frm, text=msg, fg="#c62828", justify="left").pack(anchor="w", pady=(0, 10))

                if design == "crd":
                    options = [
                        ("Краскела–Уолліса", "kw"),
                        ("Манна-Уітні", "mw"),
                    ]
                else:
                    if num_variants == 2:
                        options = [("Wilcoxon (парний)", "wilcoxon")]
                    else:
                        options = [("Friedman", "friedman")]

        out = {"ok": False, "method": None}

        if not options:
            def on_ok_close():
                dlg.destroy()
            btns = tk.Frame(frm)
            btns.pack(fill=tk.X, pady=(12, 0))
            tk.Button(btns, text="OK", width=10, command=on_ok_close).pack(side=tk.LEFT, padx=6)
            dlg.update_idletasks()
            center_window(dlg)
            dlg.bind("<Return>", lambda e: on_ok_close())
            dlg.grab_set()
            self.root.wait_window(dlg)
            return out

        var = tk.StringVar(value=options[0][1])
        for text, val in options:
            tk.Radiobutton(frm, text=text, variable=var, value=val, font=rb_font).pack(anchor="w", pady=3)

        def on_ok():
            out["ok"] = True
            out["method"] = var.get()
            dlg.destroy()

        btns = tk.Frame(frm)
        btns.pack(fill=tk.X, pady=(12, 0))
        tk.Button(btns, text="OK", width=10, command=on_ok).pack(side=tk.LEFT, padx=6)
        tk.Button(btns, text="Скасувати", width=12, command=lambda: dlg.destroy()).pack(side=tk.LEFT, padx=6)

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

        # Початкові назви факторів (можна перейменувати)
        for fk in self.factor_keys:
            if not hasattr(self, "factor_title_map") or fk not in self.factor_title_map:
                self._set_factor_title(fk, f"Фактор {fk}")

        self.table_win = tk.Toplevel(self.root)
        self.table_win.title(f"S.A.D. — {factors_count}-факторний аналіз")
        self.table_win.geometry("1280x720")
        set_window_icon(self.table_win)

        self.repeat_count = 6
        self.factor_names = [self.factor_title(fk) for fk in self.factor_keys]
        self.column_names = self.factor_names + [f"Повт.{i+1}" for i in range(self.repeat_count)]

        ctl = tk.Frame(self.table_win, padx=6, pady=6)
        ctl.pack(fill=tk.X)

        btn_texts = [
            "Додати рядок", "Видалити рядок",
            "Додати стовпчик", "Видалити стовпчик",
            "Вставити з буфера", "Аналіз даних",
            "Розробник",
        ]
        btn_font = fit_font_size_to_texts(btn_texts, family="Times New Roman", start=14, min_size=9, target_px=150)

        bw = 16
        bh = 1
        padx = 3
        pady = 2

        tk.Button(ctl, text="Додати рядок", width=bw, height=bh, font=btn_font,
                  command=self.add_row).pack(side=tk.LEFT, padx=padx, pady=pady)
        tk.Button(ctl, text="Видалити рядок", width=bw, height=bh, font=btn_font,
                  command=self.delete_row).pack(side=tk.LEFT, padx=padx, pady=pady)
        tk.Button(ctl, text="Додати стовпчик", width=bw, height=bh, font=btn_font,
                  command=self.add_column).pack(side=tk.LEFT, padx=(10, padx), pady=pady)
        tk.Button(ctl, text="Видалити стовпчик", width=bw, height=bh, font=btn_font,
                  command=self.delete_column).pack(side=tk.LEFT, padx=padx, pady=pady)
        tk.Button(ctl, text="Вставити з буфера", width=bw + 2, height=bh, font=btn_font,
                  command=self.paste_from_focus).pack(side=tk.LEFT, padx=(10, padx), pady=pady)
        tk.Button(ctl, text="Аналіз даних", width=bw, height=bh, font=btn_font,
                  bg="#c62828", fg="white", command=self.analyze).pack(side=tk.LEFT, padx=(10, padx), pady=pady)
        tk.Button(ctl, text="Розробник", width=bw, height=bh, font=btn_font,
                  command=self.show_about).pack(side=tk.RIGHT, padx=padx, pady=pady)

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

        # Заголовки
        for j, name in enumerate(self.column_names):
            lbl = tk.Label(self.inner, text=name, relief=tk.RIDGE, width=COL_W, bg="#f0f0f0", fg="#000000")
            lbl.grid(row=0, column=j, padx=2, pady=2, sticky="nsew")
            self.header_labels.append(lbl)

            # ✅ перейменування факторів подвійним кліком (тільки для колонок факторів)
            if j < self.factors_count:
                lbl.bind("<Double-Button-1>", lambda e, col=j: self.rename_factor_by_col(col))

        # Таблиця
        for i in range(self.rows):
            row_entries = []
            for j in range(self.cols):
                e = tk.Entry(
                    self.inner,
                    width=COL_W,
                    fg="#000000",
                    highlightthickness=1,
                    highlightbackground="#c0c0c0",
                    highlightcolor="#c0c0c0"
                )
                e.grid(row=i + 1, column=j, padx=2, pady=2)
                self.bind_cell(e)
                row_entries.append(e)
            self.entries.append(row_entries)

        self.inner.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

        self.entries[0][0].focus_set()

        self.table_win.bind("<Control-v>", self.on_paste)
        self.table_win.bind("<Control-V>", self.on_paste)

    # ---------- Перейменування фактора ----------
    def rename_factor_by_col(self, col_idx: int):
        if col_idx < 0 or col_idx >= self.factors_count:
            return
        fkey = self.factor_keys[col_idx]
        old = self.factor_title(fkey)

        dlg = tk.Toplevel(self.table_win if self.table_win else self.root)
        dlg.title("Перейменування фактора")
        dlg.resizable(False, False)
        set_window_icon(dlg)

        frm = tk.Frame(dlg, padx=14, pady=12)
        frm.pack(fill=tk.BOTH, expand=True)

        tk.Label(frm, text=f"Нова назва для {fkey}:", fg="#000000").grid(row=0, column=0, sticky="w")
        e = tk.Entry(frm, width=36, fg="#000000")
        e.grid(row=1, column=0, pady=(6, 0))
        e.insert(0, old)
        e.select_range(0, "end")
        e.focus_set()

        def ok():
            new = e.get().strip()
            if not new:
                messagebox.showwarning("Помилка", "Назва не може бути порожньою.")
                return
            self._set_factor_title(fkey, new)

            # оновити заголовок колонки
            if self.header_labels and col_idx < len(self.header_labels):
                self.header_labels[col_idx].configure(text=new)

            # оновити внутрішні списки назв (щоб далі використовувались нові)
            self.factor_names = [self.factor_title(fk) for fk in self.factor_keys]
            # rep-колонки залишаємо як є
            rep_names = [self.header_labels[j].cget("text") for j in range(self.factors_count, self.cols)]
            self.column_names = self.factor_names + rep_names

            dlg.destroy()

        btns = tk.Frame(frm)
        btns.grid(row=2, column=0, pady=(10, 0), sticky="w")
        tk.Button(btns, text="OK", width=10, command=ok).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(btns, text="Скасувати", width=12, command=dlg.destroy).pack(side=tk.LEFT)

        dlg.update_idletasks()
        center_window(dlg)
        dlg.bind("<Return>", lambda ev: ok())
        dlg.grab_set()

    def show_about(self):
        messagebox.showinfo(
            "Розробник",
            "S.A.D. — Статистичний аналіз даних\n"
            "Версія: 1.0\n"
            "Розробик: Чаплоуцький Андрій Миколайович\n"
            "Уманський національний університет"
        )

    # ---------- Активна комірка (помітніше) ----------
    def _set_active_cell(self, widget: tk.Entry):
        if self._active_cell is widget:
            return

        if isinstance(self._active_cell, tk.Entry) and self._active_prev_cfg:
            try:
                self._active_cell.configure(**self._active_prev_cfg)
            except Exception:
                pass

        self._active_cell = widget
        if isinstance(widget, tk.Entry):
            self._active_prev_cfg = {
                "bg": widget.cget("bg"),
                "highlightthickness": int(widget.cget("highlightthickness")),
                "highlightbackground": widget.cget("highlightbackground"),
                "highlightcolor": widget.cget("highlightcolor"),
                "relief": widget.cget("relief"),
                "bd": int(widget.cget("bd")) if str(widget.cget("bd")).isdigit() else 1,
            }
            try:
                widget.configure(
                    bg="#fff3c4",
                    highlightthickness=3,
                    highlightbackground="#c62828",
                    highlightcolor="#c62828",
                    relief=tk.SOLID,
                    bd=1
                )
            except Exception:
                pass

    # ---------- Excel-подібне дублювання (fill handle) ----------
    def _near_bottom_right(self, w: tk.Entry, margin=6) -> bool:
        try:
            px = w.winfo_pointerx()
            py = w.winfo_pointery()
            x0 = w.winfo_rootx()
            y0 = w.winfo_rooty()
            ww = w.winfo_width()
            hh = w.winfo_height()
            return (x0 + ww - margin <= px <= x0 + ww) and (y0 + hh - margin <= py <= y0 + hh)
        except Exception:
            return False

    def _fill_update_cursor(self, event):
        w = event.widget
        if not isinstance(w, tk.Entry):
            return
        # дозволяємо тільки для факторних колонок (A/B/C/D), щоб не дублювати числові повторності випадково
        pos = self.find_pos(w)
        if not pos:
            return
        r, c = pos
        if c >= self.factors_count:
            w.configure(cursor="")
            self._fill_ready = False
            return

        if self._near_bottom_right(w):
            w.configure(cursor="crosshair")
            self._fill_ready = True
        else:
            w.configure(cursor="")
            self._fill_ready = False

    def _fill_leave(self, event):
        w = event.widget
        if isinstance(w, tk.Entry):
            w.configure(cursor="")
        self._fill_ready = False

    def _fill_press(self, event):
        w = event.widget
        if not isinstance(w, tk.Entry):
            return
        pos = self.find_pos(w)
        if not pos:
            return
        r, c = pos

        if c >= self.factors_count:
            self._fill_dragging = False
            return

        if self._near_bottom_right(w):
            self._fill_dragging = True
            self._fill_src_pos = (r, c)
            self._fill_src_text = w.get()
            self._fill_last_row = r
            return "break"

        self._fill_dragging = False
        return None

    def _fill_drag(self, event):
        if not self._fill_dragging or not self._fill_src_pos:
            return
        w = event.widget
        if not isinstance(w, tk.Entry):
            return

        src_r, src_c = self._fill_src_pos

        # визначити, над яким рядком зараз курсор (за Y у вікні)
        try:
            py = w.winfo_pointery()
            # пробігаємо комірки цієї колонки, шукаємо найближчий рядок
            target_r = src_r
            for rr in range(src_r, len(self.entries)):
                cell = self.entries[rr][src_c]
                y0 = cell.winfo_rooty()
                y1 = y0 + cell.winfo_height()
                if y0 <= py <= y1:
                    target_r = rr
                    break
        except Exception:
            return

        if target_r <= src_r:
            return

        # продублювати значення з src_r у рядки src_r+1 .. target_r
        for rr in range(src_r + 1, target_r + 1):
            while rr >= len(self.entries):
                self.add_row()
            cell = self.entries[rr][src_c]
            cell.delete(0, tk.END)
            cell.insert(0, self._fill_src_text)

        self._fill_last_row = target_r
        return "break"

    def _fill_release(self, event):
        if self._fill_dragging:
            self._fill_dragging = False
            self._fill_src_pos = None
            self._fill_src_text = ""
            self._fill_last_row = None
            return "break"
        return None

    def bind_cell(self, e: tk.Entry):
        e.bind("<Return>", self.on_enter)
        e.bind("<Up>", self.on_arrow)
        e.bind("<Down>", self.on_arrow)
        e.bind("<Left>", self.on_arrow)
        e.bind("<Right>", self.on_arrow)
        e.bind("<Control-v>", self.on_paste)
        e.bind("<Control-V>", self.on_paste)

        # ✅ активна комірка
        e.bind("<FocusIn>", lambda ev: self._set_active_cell(ev.widget))

        # ✅ fill-handle
        e.bind("<Motion>", self._fill_update_cursor)
        e.bind("<Leave>", self._fill_leave)
        e.bind("<ButtonPress-1>", self._fill_press)
        e.bind("<B1-Motion>", self._fill_drag)
        e.bind("<ButtonRelease-1>", self._fill_release)

    def add_row(self):
        i = len(self.entries)
        row_entries = []
        for j in range(self.cols):
            e = tk.Entry(
                self.inner,
                width=COL_W,
                fg="#000000",
                highlightthickness=1,
                highlightbackground="#c0c0c0",
                highlightcolor="#c0c0c0"
            )
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
            e = tk.Entry(
                self.inner,
                width=COL_W,
                fg="#000000",
                highlightthickness=1,
                highlightbackground="#c0c0c0",
                highlightcolor="#c0c0c0"
            )
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
            class E:
                pass
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

    def collect_long(self, design):
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

            for idx_c, c in enumerate(rep_cols):
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

                if design in ("rcbd", "split"):
                    rec["BLOCK"] = f"Блок {idx_c + 1}"

                long.append(rec)

        return long, rep_cols

    # -------------------------
    # analyze() + show_report_segments()  -> у ЧАСТИНІ 2/2
    # -------------------------
    def analyze(self):
        raise NotImplementedError("Див. частину 2/2")

    def show_report_segments(self, segments):
        raise NotImplementedError("Див. частину 2/2")


# -------------------------
# Run  -> у ЧАСТИНІ 2/2
# -------------------------
if __name__ == "__main__":
    root = tk.Tk()
    set_window_icon(root)
    app = SADTk(root)
    root.mainloop()
# =========================
# PART 2/2 (продовження main.py)
# (УВАГА: це продовження після PART 1 — вставляй без дублювання вже вставленого коду)
# =========================

class SADTk:
    # --- у PART 1 вже є __init__, ask_indicator_units, choose_method_window ---
    # --- нижче: open_table + робота з таблицею + analyze + report + Run ---

    # ---------- helpers for factor names ----------
    def factor_label(self, key: str) -> str:
        # Повертає користувацьку назву фактора або дефолтну
        if hasattr(self, "factor_custom_names") and key in self.factor_custom_names:
            name = (self.factor_custom_names.get(key) or "").strip()
            if name:
                return name
        return f"Фактор {key}"

    def _update_factor_headers(self):
        # Оновлює шапку таблиці згідно кастомних назв факторів
        self.factor_names = [self.factor_label(k) for k in self.factor_keys]
        self.column_names = self.factor_names + [f"Повт.{i+1}" for i in range(self.repeat_count)]
        for j, lbl in enumerate(self.header_labels):
            if j < len(self.column_names):
                lbl.config(text=self.column_names[j])

    def _rename_factor_dialog(self, fkey: str):
        dlg = tk.Toplevel(self.table_win)
        dlg.title("Перейменування фактора")
        dlg.resizable(False, False)
        set_window_icon(dlg)

        frm = tk.Frame(dlg, padx=14, pady=14)
        frm.pack(fill=tk.BOTH, expand=True)

        tk.Label(frm, text=f"Нова назва для фактора {fkey}:", fg="#000000").grid(row=0, column=0, sticky="w", pady=(0, 6))
        e = tk.Entry(frm, width=36, fg="#000000")
        e.grid(row=1, column=0, sticky="w")

        cur = ""
        if hasattr(self, "factor_custom_names"):
            cur = (self.factor_custom_names.get(fkey) or "")
        e.insert(0, cur)
        e.select_range(0, tk.END)

        out = {"ok": False, "val": ""}

        def on_ok():
            out["val"] = e.get().strip()
            out["ok"] = True
            dlg.destroy()

        btns = tk.Frame(frm)
        btns.grid(row=2, column=0, sticky="w", pady=(10, 0))
        tk.Button(btns, text="OK", width=10, command=on_ok).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(btns, text="Скасувати", width=12, command=dlg.destroy).pack(side=tk.LEFT)

        dlg.update_idletasks()
        center_window(dlg)
        e.focus_set()
        dlg.bind("<Return>", lambda _e: on_ok())
        dlg.grab_set()
        self.root.wait_window(dlg)

        if out["ok"]:
            if not hasattr(self, "factor_custom_names"):
                self.factor_custom_names = {}
            self.factor_custom_names[fkey] = out["val"]
            self._update_factor_headers()

    # ---------- “Excel-like” fill handle (drag from bottom-right) ----------
    def _is_in_fill_handle(self, widget: tk.Entry, event) -> bool:
        try:
            w = widget.winfo_width()
            h = widget.winfo_height()
            # зона ~6px у правому нижньому куті
            return (event.x >= max(0, w - 7)) and (event.y >= max(0, h - 7))
        except Exception:
            return False

    def _fill_start(self, widget: tk.Entry, event):
        # стартуємо лише якщо натиснули в “ручці”
        if not self._is_in_fill_handle(widget, event):
            self._fill_dragging = False
            return

        pos = self.find_pos(widget)
        if not pos:
            self._fill_dragging = False
            return

        self._fill_dragging = True
        self._fill_start_pos = pos
        self._fill_value = widget.get()
        self._fill_last_to = pos[0]

        try:
            widget.config(cursor="tcross")
        except Exception:
            pass

    def _fill_motion(self, widget: tk.Entry, event):
        if not getattr(self, "_fill_dragging", False):
            # курсор “жирний хрестик” коли наведено на ручку
            try:
                if self._is_in_fill_handle(widget, event):
                    widget.config(cursor="tcross")
                else:
                    widget.config(cursor="xterm")
            except Exception:
                pass
            return

        r0, c0 = self._fill_start_pos

        # визначаємо рядок, до якого “дотягнули”
        try:
            # приблизна висота комірки
            cell_h = widget.winfo_height() + 4
            dy = event.y_root - widget.winfo_rooty()
            delta_rows = int(dy // max(1, cell_h))
        except Exception:
            delta_rows = 0

        r_to = r0 + max(0, delta_rows)
        if r_to >= len(self.entries):
            while r_to >= len(self.entries):
                self.add_row()

        # заповнюємо від r0 до r_to в одному стовпчику
        for rr in range(r0, r_to + 1):
            if c0 >= self.cols:
                break
            self.entries[rr][c0].delete(0, tk.END)
            self.entries[rr][c0].insert(0, self._fill_value)

        self._fill_last_to = r_to

    def _fill_end(self, widget: tk.Entry, event):
        if getattr(self, "_fill_dragging", False):
            self._fill_dragging = False
        try:
            widget.config(cursor="xterm")
        except Exception:
            pass

    # ---------- active cell highlight ----------
    def _set_active_cell(self, widget: tk.Entry, active: bool):
        try:
            if active:
                widget.config(highlightthickness=2, highlightbackground="#c62828", highlightcolor="#c62828", bg="#fff8e1")
            else:
                widget.config(highlightthickness=1, highlightbackground="#9e9e9e", highlightcolor="#9e9e9e", bg="white")
        except Exception:
            pass

    # ---------- header double click ----------
    def _on_header_double_click(self, j: int):
        if j < self.factors_count:
            fkey = self.factor_keys[j]
            self._rename_factor_dialog(fkey)

    def open_table(self, factors_count):
        if self.table_win and tk.Toplevel.winfo_exists(self.table_win):
            self.table_win.destroy()

        self.factors_count = factors_count
        self.factor_keys = ["A", "B", "C", "D"][:factors_count]

        # кастомні назви факторів зберігаємо на рівні програми
        if not hasattr(self, "factor_custom_names"):
            self.factor_custom_names = {}

        self.table_win = tk.Toplevel(self.root)
        self.table_win.title(f"S.A.D. — {factors_count}-факторний аналіз")
        self.table_win.geometry("1280x720")
        set_window_icon(self.table_win)

        self.repeat_count = 6
        self.factor_names = [self.factor_label(k) for k in self.factor_keys]
        self.column_names = self.factor_names + [f"Повт.{i+1}" for i in range(self.repeat_count)]

        ctl = tk.Frame(self.table_win, padx=6, pady=6)
        ctl.pack(fill=tk.X)

        btn_texts = [
            "Додати рядок", "Видалити рядок",
            "Додати стовпчик", "Видалити стовпчик",
            "Вставити з буфера", "Аналіз даних",
            "Розробник",
        ]
        btn_font = fit_font_size_to_texts(btn_texts, family="Times New Roman", start=14, min_size=9, target_px=150)

        bw = 16
        bh = 1
        padx = 3
        pady = 2

        tk.Button(ctl, text="Додати рядок", width=bw, height=bh, font=btn_font,
                  command=self.add_row).pack(side=tk.LEFT, padx=padx, pady=pady)
        tk.Button(ctl, text="Видалити рядок", width=bw, height=bh, font=btn_font,
                  command=self.delete_row).pack(side=tk.LEFT, padx=padx, pady=pady)
        tk.Button(ctl, text="Додати стовпчик", width=bw, height=bh, font=btn_font,
                  command=self.add_column).pack(side=tk.LEFT, padx=(10, padx), pady=pady)
        tk.Button(ctl, text="Видалити стовпчик", width=bw, height=bh, font=btn_font,
                  command=self.delete_column).pack(side=tk.LEFT, padx=padx, pady=pady)
        tk.Button(ctl, text="Вставити з буфера", width=bw + 2, height=bh, font=btn_font,
                  command=self.paste_from_focus).pack(side=tk.LEFT, padx=(10, padx), pady=pady)
        tk.Button(ctl, text="Аналіз даних", width=bw, height=bh, font=btn_font,
                  bg="#c62828", fg="white", command=self.analyze).pack(side=tk.LEFT, padx=(10, padx), pady=pady)
        tk.Button(ctl, text="Розробник", width=bw, height=bh, font=btn_font,
                  command=self.show_about).pack(side=tk.RIGHT, padx=padx, pady=pady)

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

        # header
        for j, name in enumerate(self.column_names):
            lbl = tk.Label(self.inner, text=name, relief=tk.RIDGE, width=COL_W, bg="#f0f0f0", fg="#000000")
            lbl.grid(row=0, column=j, padx=2, pady=2, sticky="nsew")
            self.header_labels.append(lbl)

            # ✅ подвійний клік по шапці фактора → перейменування
            if j < self.factors_count:
                lbl.bind("<Double-Button-1>", lambda e, jj=j: self._on_header_double_click(jj))

        # grid cells
        for i in range(self.rows):
            row_entries = []
            for j in range(self.cols):
                e = tk.Entry(self.inner, width=COL_W, fg="#000000",
                             highlightthickness=1, highlightbackground="#9e9e9e", highlightcolor="#9e9e9e", bg="white")
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
            "S.A.D. — Статистичний аналіз даних\n"
            "Версія: 1.0\n"
            "Розробик: Чаплоуцький Андрій Миколайович\n"
            "Уманський національний університет"
        )

    def bind_cell(self, e: tk.Entry):
        e.bind("<Return>", self.on_enter)
        e.bind("<Up>", self.on_arrow)
        e.bind("<Down>", self.on_arrow)
        e.bind("<Left>", self.on_arrow)
        e.bind("<Right>", self.on_arrow)
        e.bind("<Control-v>", self.on_paste)
        e.bind("<Control-V>", self.on_paste)

        # ✅ помітне виділення активної комірки
        e.bind("<FocusIn>", lambda ev, w=e: self._set_active_cell(w, True))
        e.bind("<FocusOut>", lambda ev, w=e: self._set_active_cell(w, False))

        # ✅ “fill handle” (дублювання вниз мишею)
        e.bind("<ButtonPress-1>", lambda ev, w=e: self._fill_start(w, ev), add="+")
        e.bind("<B1-Motion>", lambda ev, w=e: self._fill_motion(w, ev), add="+")
        e.bind("<ButtonRelease-1>", lambda ev, w=e: self._fill_end(w, ev), add="+")
        e.bind("<Motion>", lambda ev, w=e: self._fill_motion(w, ev), add="+")

    def add_row(self):
        i = len(self.entries)
        row_entries = []
        for j in range(self.cols):
            e = tk.Entry(self.inner, width=COL_W, fg="#000000",
                         highlightthickness=1, highlightbackground="#9e9e9e", highlightcolor="#9e9e9e", bg="white")
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
            e = tk.Entry(self.inner, width=COL_W, fg="#000000",
                         highlightthickness=1, highlightbackground="#9e9e9e", highlightcolor="#9e9e9e", bg="white")
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
            class E:
                pass
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

    def collect_long(self, design):
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

            for idx_c, c in enumerate(rep_cols):
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

                if design in ("rcbd", "split"):
                    rec["BLOCK"] = f"Блок {idx_c + 1}"

                long.append(rec)

        return long, rep_cols

    # -------------------------
    # analyze() + show_report_segments()
    # -------------------------
    def analyze(self):
        created_at = datetime.now()

        params = self.ask_indicator_units()
        if not params["ok"]:
            return

        indicator = params["indicator"]
        units = params["units"]
        design = params["design"]
        split_main = params.get("split_main", "A")

        long, used_rep_cols = self.collect_long(design)
        if len(long) == 0:
            messagebox.showwarning("Помилка", "Немає числових даних для аналізу.\nПеревірте повторності та значення.")
            return

        values = np.array([r["value"] for r in long], dtype=float)
        if len(values) < 3:
            messagebox.showinfo("Результат", "Надто мало даних для аналізу.")
            return

        levels_by_factor = {f: first_seen_order([r.get(f) for r in long]) for f in self.factor_keys}

        variant_order = first_seen_order([tuple(r.get(f) for f in self.factor_keys) for r in long])
        v_names = [" | ".join(map(str, k)) for k in variant_order]
        num_variants = len(variant_order)

        try:
            if design == "crd":
                res = anova_n_way(long, self.factor_keys, levels_by_factor)
                cell_means = res.get("cell_means", {})
                residuals = []
                for rec in long:
                    key = tuple(rec.get(f) for f in self.factor_keys)
                    v = rec.get("value", np.nan)
                    m = cell_means.get(key, np.nan)
                    if not math.isnan(v) and not math.isnan(m):
                        residuals.append(v - m)
                residuals = np.array(residuals, dtype=float)

            elif design == "rcbd":
                res = anova_rcbd_ols(long, self.factor_keys, levels_by_factor, block_key="BLOCK")
                residuals = np.array(res.get("residuals", []), dtype=float)

            else:
                if split_main not in self.factor_keys:
                    split_main = self.factor_keys[0]
                res = anova_splitplot_ols(long, self.factor_keys, main_factor=split_main, block_key="BLOCK")
                residuals = np.array(res.get("residuals", []), dtype=float)

        except Exception as ex:
            messagebox.showerror("Помилка аналізу", str(ex))
            return

        try:
            W, p_norm = shapiro(residuals) if len(residuals) >= 3 else (np.nan, np.nan)
        except Exception:
            W, p_norm = (np.nan, np.nan)

        normal = (p_norm is not None) and (not math.isnan(p_norm)) and (p_norm > 0.05)
        if design == "split" and not normal:
            messagebox.showwarning(
                "Split-plot: аналіз неможливий",
                "Обрано дизайн Split-plot, який у цій програмі реалізовано лише для параметричних методів.\n"
                "Оскільки залишки моделі не відповідають нормальному розподілу (p ≤ 0.05),\n"
                "параметричний split-plot аналіз є методично некоректним.\n\n"
                "Рекомендації:\n"
                "• застосувати трансформацію даних (log/√/Box-Cox) і повторити аналіз;\n"
                "• або вибрати CRD/RCBD і виконати непараметричний аналіз."
            )
            return

        choice = self.choose_method_window(p_norm, design, num_variants)
        if not choice["ok"]:
            return
        method = choice["method"]

        MS_error = res.get("MS_error", np.nan)
        df_error = res.get("df_error", np.nan)

        MS_whole = res.get("MS_whole", np.nan)
        df_whole = res.get("df_whole", np.nan)
        split_main_factor = res.get("main_factor", split_main) if design == "split" else None

        vstats = variant_mean_sd(long, self.factor_keys)
        v_means = {k: vstats[k][0] for k in vstats.keys()}
        v_sds = {k: vstats[k][1] for k in vstats.keys()}
        v_ns = {k: vstats[k][2] for k in vstats.keys()}

        means1 = {v_names[i]: v_means.get(variant_order[i], np.nan) for i in range(len(variant_order))}
        ns1 = {v_names[i]: v_ns.get(variant_order[i], 0) for i in range(len(variant_order))}
        groups1 = {v_names[i]: groups_by_keys(long, tuple(self.factor_keys)).get(variant_order[i], []) for i in range(len(variant_order))}

        factor_groups = {f: {k[0]: v for k, v in groups_by_keys(long, (f,)).items()} for f in self.factor_keys}
        factor_means = {f: {lvl: float(np.mean(arr)) if len(arr) else np.nan for lvl, arr in factor_groups[f].items()} for f in self.factor_keys}
        factor_ns = {f: {lvl: len(arr) for lvl, arr in factor_groups[f].items()} for f in self.factor_keys}

        factor_medians = {}
        factor_q = {}
        for f in self.factor_keys:
            factor_medians[f] = {}
            factor_q[f] = {}
            for lvl, arr in factor_groups[f].items():
                med, q1, q3 = median_q1_q3(arr)
                factor_medians[f][lvl] = med
                factor_q[f][lvl] = (q1, q3)

        ranks_by_variant = mean_ranks_by_key(long, key_func=lambda rec: " | ".join(str(rec.get(f)) for f in self.factor_keys))
        ranks_by_factor = {f: mean_ranks_by_key(long, key_func=lambda rec, ff=f: rec.get(ff)) for f in self.factor_keys}

        v_medians = {}
        v_q = {}
        for i, k in enumerate(variant_order):
            name = v_names[i]
            arr = groups1.get(name, [])
            med, q1, q3 = median_q1_q3(arr)
            v_medians[k] = med
            v_q[k] = (q1, q3)

        bf_F, bf_p = (np.nan, np.nan)
        if method in ("lsd", "tukey", "duncan", "bonferroni"):
            bf_F, bf_p = brown_forsythe_from_groups(groups1)

        kw_H, kw_p, kw_df, kw_eps2 = (np.nan, np.nan, np.nan, np.nan)
        do_posthoc = True

        fr_chi2, fr_p, fr_df, fr_W = (np.nan, np.nan, np.nan, np.nan)
        wil_stat, wil_p = (np.nan, np.nan)
        rcbd_pairwise_rows = []
        rcbd_sig = {}

        letters_factor = {f: {lvl: "" for lvl in levels_by_factor[f]} for f in self.factor_keys}
        letters_named = {name: "" for name in v_names}
        pairwise_rows = []
        factor_pairwise_tables = {}

        if method == "lsd":
            for f in self.factor_keys:
                lvls = levels_by_factor[f]
                if design == "split" and f == split_main_factor:
                    sig_f = lsd_sig_matrix(lvls, factor_means[f], factor_ns[f], MS_whole, df_whole, alpha=ALPHA)
                else:
                    sig_f = lsd_sig_matrix(lvls, factor_means[f], factor_ns[f], MS_error, df_error, alpha=ALPHA)
                letters_factor[f] = cld_multi_letters(lvls, factor_means[f], sig_f)

            if design != "split":
                sigv = lsd_sig_matrix(v_names, means1, ns1, MS_error, df_error, alpha=ALPHA)
                letters_named = cld_multi_letters(v_names, means1, sigv)

        elif method in ("tukey", "duncan", "bonferroni"):
            if design != "split":
                pairwise_rows, sig = pairwise_param_short_variants_pm(v_names, means1, ns1, MS_error, df_error, method, alpha=ALPHA)
                letters_named = cld_multi_letters(v_names, means1, sig)

            if design == "split":
                for f in self.factor_keys:
                    lvls = levels_by_factor[f]
                    means_f = factor_means[f]
                    ns_f = factor_ns[f]
                    if f == split_main_factor:
                        rows_f, _ = pairwise_param_short_variants_pm(lvls, means_f, ns_f, MS_whole, df_whole, method, alpha=ALPHA)
                    else:
                        rows_f, _ = pairwise_param_short_variants_pm(lvls, means_f, ns_f, MS_error, df_error, method, alpha=ALPHA)
                    factor_pairwise_tables[f] = rows_f

        elif method == "kw":
            try:
                kw_samples = [groups1[name] for name in v_names if len(groups1[name]) > 0]
                if len(kw_samples) >= 2:
                    kw_res = kruskal(*kw_samples)
                    kw_H = float(kw_res.statistic)
                    kw_p = float(kw_res.pvalue)
                    kw_df = int(len(kw_samples) - 1)
                    kw_eps2 = epsilon_squared_kw(kw_H, n=len(long), k=len(kw_samples))
            except Exception:
                kw_H, kw_p, kw_df, kw_eps2 = (np.nan, np.nan, np.nan, np.nan)

            if not (isinstance(kw_p, float) and math.isnan(kw_p)) and kw_p >= ALPHA:
                do_posthoc = False

            if do_posthoc:
                pairwise_rows, sig = pairwise_mw_bonf_with_effect(v_names, groups1, alpha=ALPHA)
                med_tmp = {name: float(np.median(groups1[name])) if len(groups1[name]) else np.nan for name in v_names}
                letters_named = cld_multi_letters(v_names, med_tmp, sig)
            else:
                letters_named = {name: "" for name in v_names}

        elif method == "mw":
            pairwise_rows, sig = pairwise_mw_bonf_with_effect(v_names, groups1, alpha=ALPHA)
            med_tmp = {name: float(np.median(groups1[name])) if len(groups1[name]) else np.nan for name in v_names}
            letters_named = cld_multi_letters(v_names, med_tmp, sig)

        elif method == "friedman":
            block_names = first_seen_order([f"Блок {i+1}" for i in range(len(used_rep_cols))])
            long2 = []
            for r in long:
                rr = dict(r)
                rr["VARIANT"] = " | ".join(str(rr.get(f)) for f in self.factor_keys)
                long2.append(rr)

            mat_rows, kept_blocks = rcbd_matrix_from_long(long2, v_names, block_names, variant_key="VARIANT", block_key="BLOCK")
            if len(mat_rows) < 2:
                messagebox.showwarning("Помилка", "Для Friedman потрібні щонайменше 2 повних блоки (без пропусків по варіантах).")
                return

            try:
                cols = list(zip(*mat_rows))
                fr = friedmanchisquare(*[np.array(c, dtype=float) for c in cols])
                fr_chi2 = float(fr.statistic)
                fr_p = float(fr.pvalue)
                fr_df = int(len(v_names) - 1)
                fr_W = kendalls_w_from_friedman(fr_chi2, n_blocks=len(mat_rows), k_treat=len(v_names))
            except Exception:
                fr_chi2, fr_p, fr_df, fr_W = (np.nan, np.nan, np.nan, np.nan)

            if not (isinstance(fr_p, float) and math.isnan(fr_p)) and fr_p < ALPHA:
                rcbd_pairwise_rows, rcbd_sig = pairwise_wilcoxon_bonf(v_names, mat_rows, alpha=ALPHA)
                med_tmp = {name: float(np.median(groups1[name])) if len(groups1[name]) else np.nan for name in v_names}
                letters_named = cld_multi_letters(v_names, med_tmp, rcbd_sig)
            else:
                letters_named = {name: "" for name in v_names}

        elif method == "wilcoxon":
            if len(v_names) != 2:
                messagebox.showwarning("Помилка", "Wilcoxon (парний) застосовується лише для 2 варіантів.")
                return

            block_names = first_seen_order([f"Блок {i+1}" for i in range(len(used_rep_cols))])
            long2 = []
            for r in long:
                rr = dict(r)
                rr["VARIANT"] = " | ".join(str(rr.get(f)) for f in self.factor_keys)
                long2.append(rr)

            mat_rows, kept_blocks = rcbd_matrix_from_long(long2, v_names, block_names, variant_key="VARIANT", block_key="BLOCK")
            if len(mat_rows) < 2:
                messagebox.showwarning("Помилка", "Для Wilcoxon потрібні щонайменше 2 повні блоки (пари значень).")
                return
            arr = np.array(mat_rows, dtype=float)
            x = arr[:, 0]
            y = arr[:, 1]
            try:
                stat, p = wilcoxon(x, y, zero_method="wilcox", correction=False, alternative="two-sided", mode="auto")
                wil_stat = float(stat)
                wil_p = float(p)
            except Exception:
                wil_stat, wil_p = (np.nan, np.nan)

            if not (isinstance(wil_p, float) and math.isnan(wil_p)) and wil_p < ALPHA:
                rcbd_sig = {(v_names[0], v_names[1]): True}
                med_tmp = {name: float(np.median(groups1[name])) if len(groups1[name]) else np.nan for name in v_names}
                letters_named = cld_multi_letters(v_names, med_tmp, rcbd_sig)
            else:
                letters_named = {name: "" for name in v_names}

        letters_variants = {variant_order[i]: letters_named.get(v_names[i], "") for i in range(len(variant_order))}

        SS_total = res.get("SS_total", np.nan)
        SS_error = res.get("SS_error", np.nan)
        R2 = (1.0 - (SS_error / SS_total)) if (not any(math.isnan(x) for x in [SS_total, SS_error]) and SS_total > 0) else np.nan

        cv_rows = []
        for f in self.factor_keys:
            lvl_means = [factor_means[f].get(lvl, np.nan) for lvl in levels_by_factor[f]]
            cv_f = cv_percent_from_level_means(lvl_means)
            cv_rows.append([self.factor_label(f), fmt_num(cv_f, 2)])
        cv_total = cv_percent_from_values(values)
        cv_rows.append(["Загальний", fmt_num(cv_total, 2)])

        seg = []
        seg.append(("text", "З В І Т   С Т А Т И С Т И Ч Н О Г О   А Н А Л І З У   Д А Н И Х\n\n"))
        seg.append(("text", f"Показник:\t{indicator}\nОдиниці виміру:\t{units}\n\n"))

        seg.append(("text",
                    f"Кількість варіантів:\t{num_variants}\n"
                    f"Кількість повторностей:\t{len(used_rep_cols)}\n"
                    f"Загальна кількість облікових значень:\t{len(long)}\n\n"))

        design_label = {"crd": "CRD (повна рандомізація)", "rcbd": "RCBD (блочна рандомізація)", "split": "Split-plot (спліт-плот)"}[design]
        seg.append(("text", f"Дизайн експерименту:\t{design_label}\n"))
        if design == "split":
            seg.append(("text", f"Головний фактор (Whole-plot factor):\t{split_main_factor}\n\n"))
        else:
            seg.append(("text", "\n"))

        method_label = {
            "lsd": "Параметричний аналіз: Brown–Forsythe + ANOVA + НІР₀₅ (LSD).",
            "tukey": "Параметричний аналіз: Brown–Forsythe + ANOVA + тест Тьюкі (Tukey HSD).",
            "duncan": "Параметричний аналіз: Brown–Forsythe + ANOVA + тест Дункана.",
            "bonferroni": "Параметричний аналіз: Brown–Forsythe + ANOVA + корекція Бонферроні.",
            "kw": "Непараметричний аналіз: Kruskal–Wallis.",
            "mw": "Непараметричний аналіз: Mann–Whitney.",
            "friedman": "Непараметричний аналіз (RCBD): Friedman.",
            "wilcoxon": "Непараметричний аналіз (RCBD): Wilcoxon signed-rank (парний).",
        }.get(method, "")

        if method_label:
            seg.append(("text", f"Виконуваний статистичний аналіз:\t{method_label}\n\n"))

        seg.append(("text", "Пояснення позначень істотності: ** — p<0.01; * — p<0.05.\n"))
        seg.append(("text", "У таблицях знак \"-\" свідчить що p ≥ 0.05.\n"))
        seg.append(("text", "Істотна різниця (літери): різні літери свідчать про наявність істотної різниці.\n\n"))

        if not math.isnan(W):
            seg.append(("text",
                        f"Перевірка нормальності залишків (Shapiro–Wilk):\t{normality_text(p_norm)}\t"
                        f"(W={fmt_num(float(W),4)}; p={fmt_num(float(p_norm),4)})\n\n"))
        else:
            seg.append(("text", "Перевірка нормальності залишків (Shapiro–Wilk):\tн/д\n\n"))

        nonparam = method in ("mw", "kw", "friedman", "wilcoxon")

        if method == "kw":
            if not (isinstance(kw_p, float) and math.isnan(kw_p)):
                concl = "істотна різниця " + significance_mark(kw_p) if kw_p < ALPHA else "-"
                seg.append(("text",
                            f"Глобальний тест між варіантами (Kruskal–Wallis):\t"
                            f"H={fmt_num(kw_H,4)}; df={int(kw_df)}; p={fmt_num(kw_p,4)}\t{concl}\n"))
                seg.append(("text", f"Розмір ефекту (ε²):\t{fmt_num(kw_eps2,4)}\n\n"))
            else:
                seg.append(("text", "Глобальний тест між варіантами (Kruskal–Wallis):\tн/д\n\n"))

        if method == "friedman":
            if not (isinstance(fr_p, float) and math.isnan(fr_p)):
                concl = "істотна різниця " + significance_mark(fr_p) if fr_p < ALPHA else "-"
                seg.append(("text",
                            f"Глобальний тест між варіантами (Friedman):\t"
                            f"χ²={fmt_num(fr_chi2,4)}; df={int(fr_df)}; p={fmt_num(fr_p,4)}\t{concl}\n"))
                seg.append(("text", f"Розмір ефекту (Kendall’s W):\t{fmt_num(fr_W,4)}\n\n"))
            else:
                seg.append(("text", "Глобальний тест між варіантами (Friedman):\tн/д\n\n"))

        if method == "wilcoxon":
            if not (isinstance(wil_p, float) and math.isnan(wil_p)):
                concl = "істотна різниця " + significance_mark(wil_p) if wil_p < ALPHA else "-"
                seg.append(("text",
                            f"Парний тест (Wilcoxon signed-rank):\t"
                            f"W={fmt_num(wil_stat,4)}; p={fmt_num(wil_p,4)}\t{concl}\n\n"))
            else:
                seg.append(("text", "Парний тест (Wilcoxon signed-rank):\tн/д\n\n"))

        if not nonparam:
            if not any(math.isnan(x) for x in [bf_F, bf_p]):
                bf_concl = "умова виконується" if bf_p >= ALPHA else f"умова порушена {significance_mark(bf_p)}"
                seg.append(("text",
                            f"Перевірка однорідності дисперсій (Brown–Forsythe):\t"
                            f"F={fmt_num(bf_F,4)}; p={fmt_num(bf_p,4)}\t{bf_concl}\n\n"))
            else:
                seg.append(("text", "Перевірка однорідності дисперсій (Brown–Forsythe):\tн/д\n\n"))

            if design == "split":
                seg.append(("text",
                            "Примітка (Split-plot):\n"
                            f"• {self.factor_label(split_main_factor)} перевірено на MS(Блоки×{split_main_factor}) (whole-plot error).\n"
                            "• Інші ефекти перевірено на MS(Залишок) (sub-plot error).\n\n"))

            anova_rows = []
            for name, SSv, dfv, MSv, Fv, pv in res["table"]:
                df_txt = str(int(dfv)) if dfv is not None and not (isinstance(dfv, float) and math.isnan(dfv)) else ""
                if name.startswith("Фактор "):
                    # підміна "Фактор A" → кастомна назва
                    try:
                        # name = "Фактор A×B" або "Фактор A"
                        rest = name.replace("Фактор ", "")
                        parts = rest.split("×")
                        parts2 = [self.factor_label(p) if p in self.factor_keys else p for p in parts]
                        name2 = "×".join(parts2)
                    except Exception:
                        name2 = name
                else:
                    name2 = name

                if name2.startswith("Залишок") or name2 == "Загальна" or "Whole-plot error" in name2:
                    anova_rows.append([name2, fmt_num(SSv, 2), df_txt, fmt_num(MSv, 3), "", "", ""])
                else:
                    mark = significance_mark(pv)
                    concl = f"істотна різниця {mark}" if mark else "-"
                    anova_rows.append([name2, fmt_num(SSv, 2), df_txt, fmt_num(MSv, 3), fmt_num(Fv, 3), fmt_num(pv, 4), concl])

            seg.append(("text", "ТАБЛИЦЯ 1. Дисперсійний аналіз (ANOVA)\n"))
            seg.append(("table", {
                "headers": ["Джерело", "SS", "df", "MS", "F", "p", "Висновок"],
                "rows": anova_rows,
                "padding_px": 32,
                "extra_gap_after_col": 0,
                "extra_gap_px": 60
            }))
            seg.append(("text", "\n"))

            eff_rows = build_effect_strength_rows(res["table"])
            # заміна назв факторів у таблицях сили впливу / eta2
            for r in eff_rows:
                if r and isinstance(r[0], str) and r[0].startswith("Фактор "):
                    rest = r[0].replace("Фактор ", "")
                    parts = rest.split("×")
                    parts2 = [self.factor_label(p) if p in self.factor_keys else p for p in parts]
                    r[0] = "×".join(parts2)

            seg.append(("text", "ТАБЛИЦЯ 2. Сила впливу факторів та їх комбінацій (% від SS)\n"))
            seg.append(("table", {"headers": ["Джерело", "%"], "rows": eff_rows}))
            seg.append(("text", "\n"))

            pe2_rows = build_partial_eta2_rows_with_label(res["table"])
            for r in pe2_rows:
                if r and isinstance(r[0], str) and r[0].startswith("Фактор "):
                    rest = r[0].replace("Фактор ", "")
                    parts = rest.split("×")
                    parts2 = [self.factor_label(p) if p in self.factor_keys else p for p in parts]
                    r[0] = "×".join(parts2)

            seg.append(("text", "ТАБЛИЦЯ 3. Розмір ефекту (partial η²)\n"))
            seg.append(("table", {"headers": ["Джерело", "partial η²", "Висновок"], "rows": pe2_rows}))
            seg.append(("text", "\n"))

            seg.append(("text", "ТАБЛИЦЯ 4. Коефіцієнт варіації (CV, %)\n"))
            seg.append(("table", {"headers": ["Елемент", "CV, %"], "rows": cv_rows}))
            seg.append(("text", "\n"))

            seg.append(("text", f"Коефіцієнт детермінації:\tR²={fmt_num(R2, 4)}\n\n"))

            tno = 5
            if method == "lsd":
                nir_rows = []
                if design != "split":
                    for key in [f"Фактор {f}" for f in self.factor_keys] + ["Загальна"]:
                        if key in res.get("NIR05", {}):
                            # у NIR ключі як "Фактор A" — друкуємо кастомну назву
                            if key.startswith("Фактор "):
                                fkey = key.replace("Фактор ", "").strip()
                                nir_rows.append([self.factor_label(fkey), fmt_num(res["NIR05"][key], 4)])
                            else:
                                nir_rows.append([key, fmt_num(res["NIR05"][key], 4)])
                else:
                    for key, val in res.get("NIR05", {}).items():
                        nir_rows.append([key, fmt_num(val, 4)])

                seg.append(("text", "ТАБЛИЦЯ 5. Значення НІР₀₅\n"))
                seg.append(("table", {"headers": ["Елемент", "НІР₀₅"], "rows": nir_rows}))
                seg.append(("text", "\n"))
                tno = 6

            for f in self.factor_keys:
                seg.append(("text", f"ТАБЛИЦЯ {tno}. Середнє по фактору: {self.factor_label(f)}\n"))
                rows_f = []
                for lvl in levels_by_factor[f]:
                    m = factor_means[f].get(lvl, np.nan)
                    letter = letters_factor[f].get(lvl, "")
                    rows_f.append([str(lvl), fmt_num(m, 3), (letter if letter else "-")])
                seg.append(("table", {"headers": [self.factor_label(f), "Середнє", "Істотна різниця"], "rows": rows_f}))
                seg.append(("text", "\n"))
                tno += 1

            seg.append(("text", f"ТАБЛИЦЯ {tno}. Таблиця середніх значень варіантів\n"))
            rows_v = []
            for k in variant_order:
                name = " | ".join(map(str, k))
                m = v_means.get(k, np.nan)
                sd = v_sds.get(k, np.nan)
                if design == "split":
                    rows_v.append([name, fmt_num(m, 3), fmt_num(sd, 3), "-"])
                else:
                    letter = letters_variants.get(k, "")
                    rows_v.append([name, fmt_num(m, 3), fmt_num(sd, 3), (letter if letter else "-")])

            seg.append(("table", {
                "headers": ["Варіант", "Середнє", "± SD", "Істотна різниця"],
                "rows": rows_v,
                "padding_px": 32,
                "extra_gap_after_col": 0,
                "extra_gap_px": 80
            }))
            seg.append(("text", "\n"))
            tno += 1

            if design != "split":
                if method in ("tukey", "duncan", "bonferroni") and pairwise_rows:
                    seg.append(("text", f"ТАБЛИЦЯ {tno}. Парні порівняння варіантів\n"))
                    seg.append(("table", {"headers": ["Комбінація варіантів", "p", "Істотна різниця"], "rows": pairwise_rows}))
                    seg.append(("text", "\n"))
            else:
                seg.append(("text",
                            "Примітка (Split-plot): парні порівняння для повних варіантів (комбінацій факторів)\n"
                            "не подаються, оскільки для таких порівнянь потрібні спеціальні контрасти та коректний\n"
                            "облік двох різних помилок (whole-plot і sub-plot). Натомість подано парні порівняння\n"
                            "на рівні факторів з правильними error-term.\n\n"))
                if method in ("tukey", "duncan", "bonferroni"):
                    for f in self.factor_keys:
                        rows_pf = factor_pairwise_tables.get(f, [])
                        if rows_pf:
                            seg.append(("text", f"ТАБЛИЦЯ {tno}. Парні порівняння для: {self.factor_label(f)} (Split-plot)\n"))
                            seg.append(("table", {"headers": ["Комбінація", "p", "Істотна різниця"], "rows": rows_pf}))
                            seg.append(("text", "\n"))
                            tno += 1

        else:
            tno = 1
            for f in self.factor_keys:
                seg.append(("text", f"ТАБЛИЦЯ {tno}. Описова статистика (непараметрична): {self.factor_label(f)}\n"))
                rows = []
                for lvl in levels_by_factor[f]:
                    med = factor_medians[f].get(lvl, np.nan)
                    q1, q3 = factor_q[f].get(lvl, (np.nan, np.nan))
                    rank_m = ranks_by_factor[f].get(lvl, np.nan)
                    letter = letters_factor[f].get(lvl, "")
                    rows.append([
                        str(lvl),
                        str(int(factor_ns[f].get(lvl, 0))),
                        fmt_num(med, 3),
                        f"{fmt_num(q1,3)}–{fmt_num(q3,3)}" if not any(math.isnan(x) for x in [q1, q3]) else "",
                        fmt_num(rank_m, 2),
                        (letter if letter else "-")
                    ])
                seg.append(("table", {"headers": [self.factor_label(f), "n", "Медіана", "Q1–Q3", "Середній ранг", "Істотна різниця"], "rows": rows}))
                seg.append(("text", "\n"))
                tno += 1

            seg.append(("text", f"ТАБЛИЦЯ {tno}. Описова статистика варіантів (непараметрична)\n"))
            rows = []
            for k in variant_order:
                name = " | ".join(map(str, k))
                med = v_medians.get(k, np.nan)
                q1, q3 = v_q.get(k, (np.nan, np.nan))
                rank_m = ranks_by_variant.get(name, np.nan)
                letter = letters_variants.get(k, "")
                rows.append([
                    name,
                    str(int(v_ns.get(k, 0))),
                    fmt_num(med, 3),
                    f"{fmt_num(q1,3)}–{fmt_num(q3,3)}" if not any(math.isnan(x) for x in [q1, q3]) else "",
                    fmt_num(rank_m, 2),
                    (letter if letter else "-")
                ])
            seg.append(("table", {
                "headers": ["Варіант", "n", "Медіана", "Q1–Q3", "Середній ранг", "Істотна різниця"],
                "rows": rows,
                "padding_px": 32,
                "extra_gap_after_col": 0,
                "extra_gap_px": 80
            }))
            seg.append(("text", "\n"))
            tno += 1

            if method == "kw":
                if not do_posthoc:
                    seg.append(("text", "Пост-хок порівняння не виконувалися, оскільки глобальний тест Kruskal–Wallis не виявив істотної різниці (p ≥ 0.05).\n\n"))
                else:
                    if pairwise_rows:
                        seg.append(("text", f"ТАБЛИЦЯ {tno}. Парні порівняння + ефект (Cliff’s δ)\n"))
                        seg.append(("table", {"headers": ["Комбінація варіантів", "U", "p (Bonf.)", "Істотна різниця", "δ", "Висновок"], "rows": pairwise_rows}))
                        seg.append(("text", "\n"))

            if method == "friedman":
                if not (isinstance(fr_p, float) and math.isnan(fr_p)) and fr_p >= ALPHA:
                    seg.append(("text", "Пост-хок порівняння не виконувалися, оскільки глобальний тест Friedman не виявив істотної різниці (p ≥ 0.05).\n\n"))
                else:
                    if rcbd_pairwise_rows:
                        seg.append(("text", f"ТАБЛИЦЯ {tno}. Парні порівняння (Wilcoxon, Bonferroni) + ефект (r)\n"))
                        seg.append(("table", {"headers": ["Комбінація варіантів", "W", "p (Bonf.)", "Істотна різниця", "r"], "rows": rcbd_pairwise_rows}))
                        seg.append(("text", "\n"))

        seg.append(("text", f"Звіт сформовано:\t{created_at.strftime('%d.%m.%Y, %H:%M')}\n"))
        self.show_report_segments(seg)

    def show_report_segments(self, segments):
        if self.report_win and tk.Toplevel.winfo_exists(self.report_win):
            self.report_win.destroy()

        self.report_win = tk.Toplevel(self.root)
        self.report_win.title("Звіт")
        self.report_win.geometry("1180x760")
        set_window_icon(self.report_win)

        top = tk.Frame(self.report_win, padx=8, pady=8)
        top.pack(fill=tk.X)

        xsb = ttk.Scrollbar(self.report_win, orient="horizontal")
        xsb.pack(side=tk.BOTTOM, fill=tk.X)

        txt = ScrolledText(self.report_win, width=120, height=40, wrap="none", xscrollcommand=xsb.set)
        txt.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        xsb.config(command=txt.xview)

        txt.configure(font=("Times New Roman", 14), fg="#000000")
        font_obj = tkfont.Font(font=("Times New Roman", 14))

        table_idx = 0
        for kind, payload in segments:
            if kind == "text":
                txt.insert("end", payload)
                continue

            if isinstance(payload, dict):
                headers = payload.get("headers", [])
                rows = payload.get("rows", [])
                padding_px = int(payload.get("padding_px", 32))
                extra_after = payload.get("extra_gap_after_col", None)
                extra_px = int(payload.get("extra_gap_px", 0))
            else:
                headers, rows = payload
                padding_px = 32
                extra_after = None
                extra_px = 0

            tabs = tabs_from_table_px(
                font_obj, headers, rows,
                padding_px=padding_px,
                extra_gap_after_col=extra_after,
                extra_gap_px=extra_px
            )

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
# Run
# -------------------------
if __name__ == "__main__":
    root = tk.Tk()
    set_window_icon(root)
    app = SADTk(root)
    root.mainloop()
