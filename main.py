# main.py
# -*- coding: utf-8 -*-

"""
S.A.D. — Статистичний аналіз даних (Tkinter)
"""

import os
import sys
import math
import ctypes
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import tkinter.font as tkfont

import numpy as np
from itertools import combinations
from collections import defaultdict
from datetime import datetime

from scipy.stats import shapiro, t, f as f_dist
from scipy.stats import mannwhitneyu, kruskal, levene, rankdata
from scipy.stats import studentized_range


ALPHA = 0.05
COL_W = 10


# -------------------------
# DPI awareness (Windows)
# -------------------------
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


# -------------------------
# ICON (robust search)
# -------------------------
def _script_dir():
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except Exception:
        return os.getcwd()


def _find_icon_file():
    for base in (_script_dir(), os.getcwd(), os.path.dirname(sys.argv[0])):
        if base:
            p = os.path.join(base, "icon.ico")
            if os.path.exists(p):
                return p
    return None


def set_window_icon(win):
    ico = _find_icon_file()
    if ico:
        try:
            win.iconbitmap(ico)
        except Exception:
            pass


# -------------------------
# Helpers
# -------------------------
def significance_mark(p):
    if p is None or math.isnan(p):
        return ""
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def fmt_num(x, nd=3):
    if x is None or math.isnan(x):
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
    w, h = win.winfo_width(), win.winfo_height()
    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    win.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")


def groups_by_keys(long, keys):
    g = defaultdict(list)
    for r in long:
        v = r["value"]
        if not math.isnan(v):
            g[tuple(r[k] for k in keys)].append(v)
    return g


# -------------------------
# GUI
# -------------------------
class SADTk:
    def __init__(self, root):
        self.root = root
        root.title("S.A.D. — Статистичний аналіз даних")
        root.geometry("1000x560")
        set_window_icon(root)

        root.option_add("*Font", ("Times New Roman", 15))

        self.main = tk.Frame(root, bg="white")
        self.main.pack(expand=True, fill=tk.BOTH)

        tk.Label(
            self.main,
            text="S.A.D. — Статистичний аналіз даних",
            font=("Times New Roman", 20, "bold"),
            bg="white"
        ).pack(pady=18)

        bf = tk.Frame(self.main, bg="white")
        bf.pack(pady=10)

        tk.Button(bf, text="Однофакторний аналіз", width=22, height=2,
                  command=lambda: self.open_table(1)).grid(row=0, column=0, padx=10, pady=8)
        tk.Button(bf, text="Двофакторний аналіз", width=22, height=2,
                  command=lambda: self.open_table(2)).grid(row=0, column=1, padx=10, pady=8)
        tk.Button(bf, text="Трифакторний аналіз", width=22, height=2,
                  command=lambda: self.open_table(3)).grid(row=1, column=0, padx=10, pady=8)
        tk.Button(bf, text="Чотирифакторний аналіз", width=22, height=2,
                  command=lambda: self.open_table(4)).grid(row=1, column=1, padx=10, pady=8)

        tk.Label(
            self.main,
            text="Виберіть тип аналізу → Внесіть дані → Натисніть «Аналіз даних»",
            bg="white"
        ).pack(pady=10)

        self.table_win = None

    # -------------------------
    # TABLE WINDOW
    # -------------------------
    def open_table(self, factors_count):
        if self.table_win and self.table_win.winfo_exists():
            self.table_win.destroy()

        self.factors_count = factors_count
        self.factor_keys = ["A", "B", "C", "D"][:factors_count]

        self.table_win = tk.Toplevel(self.root)
        self.table_win.title(f"S.A.D. — {factors_count}-факторний аналіз")
        self.table_win.geometry("1280x720")
        set_window_icon(self.table_win)

        # ---------- КНОПКИ ----------
        ctl = tk.Frame(self.table_win, padx=6, pady=6)
        ctl.pack(fill=tk.X)

        btn_font = tkfont.Font(family="Times New Roman", size=14)  # ❗ НЕ змінюємо

        avg_char_px = max(1, btn_font.measure("0"))

        # ✅ вужчі кнопки (менший запас)
        def btn_w(text, pad_px=12, min_chars=10):
            px = btn_font.measure(text) + pad_px
            return max(min_chars, int(math.ceil(px / avg_char_px)))

        bh = 1
        pad = 3

        tk.Button(ctl, text="Додати рядок", font=btn_font,
                  width=btn_w("Додати рядок"), height=bh,
                  command=self.add_row).pack(side=tk.LEFT, padx=pad)

        tk.Button(ctl, text="Видалити рядок", font=btn_font,
                  width=btn_w("Видалити рядок"), height=bh,
                  command=self.delete_row).pack(side=tk.LEFT, padx=pad)

        tk.Button(ctl, text="Додати стовпчик", font=btn_font,
                  width=btn_w("Додати стовпчик"), height=bh,
                  command=self.add_column).pack(side=tk.LEFT, padx=(10, pad))

        tk.Button(ctl, text="Видалити стовпчик", font=btn_font,
                  width=btn_w("Видалити стовпчик"), height=bh,
                  command=self.delete_column).pack(side=tk.LEFT, padx=pad)

        tk.Button(ctl, text="Вставити з буфера", font=btn_font,
                  width=btn_w("Вставити з буфера", pad_px=16, min_chars=11),
                  height=bh, command=self.paste_from_focus).pack(side=tk.LEFT, padx=(10, pad))

        tk.Button(ctl, text="Аналіз даних", font=btn_font,
                  width=btn_w("Аналіз даних", pad_px=14),
                  height=bh, bg="#c62828", fg="white",
                  command=self.analyze).pack(side=tk.LEFT, padx=(10, pad))

        tk.Button(ctl, text="Розробник", font=btn_font,
                  width=btn_w("Розробник"),
                  height=bh, command=self.show_about).pack(side=tk.RIGHT, padx=pad)

        # ---------- ДАЛІ таблиця ----------
        self.canvas = tk.Canvas(self.table_win)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sb = ttk.Scrollbar(self.table_win, orient="vertical", command=self.canvas.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=sb.set)

        self.inner = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.rows = 12
        self.cols = factors_count + 6
        self.entries = []

        for j in range(self.cols):
            tk.Label(self.inner, text=f"Кол.{j+1}", width=COL_W,
                     relief=tk.RIDGE, bg="#f0f0f0").grid(row=0, column=j, padx=2, pady=2)

        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                e = tk.Entry(self.inner, width=COL_W)
                e.grid(row=i+1, column=j, padx=2, pady=2)
                row.append(e)
            self.entries.append(row)

        self.inner.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    # -------------------------
    # STUBS (логіка в тебе вже є)
    # -------------------------
    def add_row(self): pass
    def delete_row(self): pass
    def add_column(self): pass
    def delete_column(self): pass
    def paste_from_focus(self): pass
    def analyze(self): pass

    def show_about(self):
        messagebox.showinfo(
            "Розробник",
            "S.A.D. — Статистичний аналіз даних\n"
            "Чаплоуцький Андрій Миколайович\n"
            "Уманський НУ садівництва"
        )


# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = SADTk(root)
    root.mainloop()
