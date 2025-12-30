# -*- coding: utf-8 -*-
"""
SAD – Статистичний аналіз даних
Tkinter + ANOVA (1–4 фактори)

Автор: Чаплоуцький А.М.
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from tkinter.scrolledtext import ScrolledText
import math
import numpy as np
from scipy.stats import f, t, shapiro
import matplotlib.pyplot as plt

# =========================
# Допоміжні функції
# =========================
def sig_mark(p):
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""

# =========================
# ANOVA (1–4 фактори)
# =========================
def anova_general(long, factors):
    values = np.array([r["value"] for r in long])
    grand_mean = np.mean(values)
    N = len(values)

    def levels(k):
        return sorted(set(r[k] for r in long))

    means = {}
    counts = {}
    for fct in factors:
        for lev in levels(fct):
            vals = [r["value"] for r in long if r[fct] == lev]
            means[(fct, lev)] = np.mean(vals)
            counts[(fct, lev)] = len(vals)

    SS_total = np.sum((values - grand_mean) ** 2)

    table = []
    df_used = 0
    SS_used = 0

    for fct in factors:
        SS = sum(
            counts[(fct, lev)] * (means[(fct, lev)] - grand_mean) ** 2
            for lev in levels(fct)
        )
        df = len(levels(fct)) - 1
        MS = SS / df if df > 0 else np.nan
        table.append((fct, SS, df, MS))
        df_used += df
        SS_used += SS

    SS_error = SS_total - SS_used
    df_error = N - df_used - 1
    MS_error = SS_error / df_error if df_error > 0 else np.nan

    final = []
    for fct, SS, df, MS in table:
        F = MS / MS_error if MS_error > 0 else np.nan
        p = 1 - f.cdf(F, df, df_error)
        final.append((fct, SS, df, MS, F, p))

    final.append(("Залишок", SS_error, df_error, MS_error, None, None))
    final.append(("Загальна", SS_total, N - 1, None, None, None))

    return final, MS_error, df_error

# =========================
# GUI
# =========================
class SADApp:
    def __init__(self, root):
        self.root = root
        root.title("SAD — Статистичний аналіз даних")
        root.geometry("900x450")

        tk.Label(root, text="SAD — Статистичний аналіз даних",
                 font=("Arial", 18, "bold")).pack(pady=10)

        btns = tk.Frame(root)
        btns.pack()

        for i in range(1, 5):
            tk.Button(
                btns,
                text=f"{i}-факторний аналіз",
                width=20,
                command=lambda x=i: self.open_table(x)
            ).grid(row=0, column=i-1, padx=5)

    # =========================
    def open_table(self, k):
        self.factors_count = k
        self.win = tk.Toplevel(self.root)
        self.win.title(f"{k}-факторний аналіз")
        self.win.geometry("1200x600")

        top = tk.Frame(self.win)
        top.pack(fill=tk.X)

        tk.Button(top, text="Додати рядок", command=self.add_row).pack(side=tk.LEFT)
        tk.Button(top, text="Видалити рядок", command=self.del_row).pack(side=tk.LEFT)
        tk.Button(top, text="Додати стовпчик", command=self.add_col).pack(side=tk.LEFT)
        tk.Button(top, text="Видалити стовпчик", command=self.del_col).pack(side=tk.LEFT)

        tk.Button(
            top, text="Аналіз даних", bg="#c62828", fg="white",
            command=self.analyze
        ).pack(side=tk.RIGHT, padx=20)

        self.factor_names = [f"Фактор {chr(65+i)}" for i in range(k)]
        self.rep_count = 4
        self.cols = k + self.rep_count
        self.rows = 10

        self.canvas = tk.Canvas(self.win)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        frame = tk.Frame(self.canvas)
        self.canvas.create_window((0,0), window=frame, anchor="nw")

        self.entries = []

        headers = self.factor_names + [f"Повт.{i+1}" for i in range(self.rep_count)]
        for j, h in enumerate(headers):
            tk.Label(frame, text=h, relief=tk.RIDGE, width=14).grid(row=0, column=j)

        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                e = tk.Entry(frame, width=14)
                e.grid(row=i+1, column=j)
                e.bind("<Return>", self.move_down)
                e.bind("<Up>", self.move)
                e.bind("<Down>", self.move)
                e.bind("<Left>", self.move)
                e.bind("<Right>", self.move)
                e.bind("<Control-v>", self.paste)
                row.append(e)
            self.entries.append(row)

        frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    # =========================
    def add_row(self):
        frame = self.entries[0][0].master
        r = len(self.entries)
        row = []
        for c in range(self.cols):
            e = tk.Entry(frame, width=14)
            e.grid(row=r+1, column=c)
            e.bind("<Return>", self.move_down)
            e.bind("<Up>", self.move)
            e.bind("<Down>", self.move)
            e.bind("<Left>", self.move)
            e.bind("<Right>", self.move)
            e.bind("<Control-v>", self.paste)
            row.append(e)
        self.entries.append(row)

    def del_row(self):
        if not self.entries:
            return
        row = self.entries.pop()
        for e in row:
            e.destroy()

    def add_col(self):
        frame = self.entries[0][0].master
        j = self.cols
        tk.Label(frame, text=f"Повт.{j+1}", relief=tk.RIDGE, width=14).grid(row=0, column=j)
        for i, row in enumerate(self.entries):
            e = tk.Entry(frame, width=14)
            e.grid(row=i+1, column=j)
            e.bind("<Return>", self.move_down)
            e.bind("<Up>", self.move)
            e.bind("<Down>", self.move)
            e.bind("<Left>", self.move)
            e.bind("<Right>", self.move)
            e.bind("<Control-v>", self.paste)
            row.append(e)
        self.cols += 1

    def del_col(self):
        if self.cols <= self.factors_count + 1:
            return
        self.cols -= 1
        for row in self.entries:
            row[-1].destroy()
            row.pop()

    # =========================
    def move_down(self, e):
        self.move_generic(e, 1, 0)

    def move(self, e):
        d = {"Up":(-1,0),"Down":(1,0),"Left":(0,-1),"Right":(0,1)}
        self.move_generic(e, *d[e.keysym])

    def move_generic(self, e, di, dj):
        for i,r in enumerate(self.entries):
            for j,c in enumerate(r):
                if c == e.widget:
                    ni = max(0, min(i+di, len(self.entries)-1))
                    nj = max(0, min(j+dj, self.cols-1))
                    self.entries[ni][nj].focus_set()
                    return "break"

    def paste(self, e):
        txt = self.win.clipboard_get()
        rows = txt.splitlines()
        for i,r in enumerate(self.entries):
            for j,c in enumerate(r):
                if c == e.widget:
                    for rr, line in enumerate(rows):
                        vals = line.split("\t")
                        for cc,v in enumerate(vals):
                            if i+rr < len(self.entries) and j+cc < self.cols:
                                self.entries[i+rr][j+cc].delete(0, tk.END)
                                self.entries[i+rr][j+cc].insert(0, v)
                    return "break"

    # =========================
    def analyze(self):
        name = simpledialog.askstring("Показник", "Назва показника:")
        unit = simpledialog.askstring("Одиниці", "Одиниці виміру:")

        data = []
        for r in self.entries:
            factors = [r[i].get() for i in range(self.factors_count)]
            for c in range(self.factors_count, self.cols):
                try:
                    v = float(r[c].get())
                except:
                    continue
                rec = {"value": v}
                for i,f in enumerate("ABCD"):
                    if i < self.factors_count:
                        rec[f] = factors[i]
                data.append(rec)

        if not data:
            messagebox.showwarning("Помилка","Немає даних")
            return

        table, MSerr, dferr = anova_general(data, list("ABCD")[:self.factors_count])

        win = tk.Toplevel(self.root)
        win.title("Звіт")
        txt = ScrolledText(win, font=("Times New Roman",14))
        txt.pack(fill=tk.BOTH, expand=True)

        txt.insert(tk.END, f"{name}, {unit}\n\n")
        txt.insert(tk.END, "Зірочки: * p<0.05, ** p<0.01\n\n")

        for r in table:
            txt.insert(tk.END, f"{r}\n")

        # гістограма
        means = {}
        for rec in data:
            means.setdefault(rec["A"], []).append(rec["value"])

        labs = list(means.keys())
        vals = [np.mean(means[k]) for k in labs]

        plt.figure()
        plt.bar(labs, vals)
        plt.title(name)
        plt.ylabel(unit)
        plt.show()

# =========================
if __name__ == "__main__":
    root = tk.Tk()
    SADApp(root)
    root.mainloop()
