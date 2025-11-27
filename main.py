# -*- coding: utf-8 -*-
"""
SAD v1.0 GOLD — Фінальна версія
Класичний український калькулятор польового досліду + Shapiro-Wilk + таблиця як в Excel
Автор: Чаплоуцький Андрій Миколайович, Уманський НУС, 2025
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext, simpledialog
import pandas as pd
import numpy as np
from scipy.stats import f, t, shapiro
import os


# ——————————————————————— Excel-подібна таблиця з сіткою і навігацією ———————————————————————
class ExcelLikeTreeview(ttk.Treeview):
    def __init__(self, master, **kw):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Excel.Treeview",
                        background="white", foreground="black", rowheight=28,
                        fieldbackground="white", borderwidth=1, relief="solid",
                        font=("Arial", 11))
        style.configure("Excel.Treeview.Heading",
                        background="#d9d9d9", font=("Arial", 11, "bold"))
        style.map('Excel.Treeview', background=[('selected', '#4472C4')])

        super().__init__(master, style="Excel.Treeview", **kw)

        self._entry = None
        self._current_item = None
        self._current_col = None

        self.bind("<Double-1>", self._start_edit)
        self.bind("<Return>", self._on_enter)
        self.bind("<Down>", self._on_down)
        self.bind("<Up>", self._on_up)
        self.bind("<Left>", self._on_left)
        self.bind("<Right>", self._on_right)

    def _start_edit(self, event=None):
        if event:
            row = self.identify_row(event.y)
            col = self.identify_column(event.x)
        else:
            row = self.focus()
            col = "#1"

        if not row or not col:
            return

        bbox = self.bbox(row, col)
        if not bbox:
            return

        x, y, w, h = bbox
        col_idx = int(col[1:]) - 1
        values = self.item(row, "values")
        text = values[col_idx] if col_idx < len(values) else ""

        entry = tk.Entry(self, font=("Arial", 11), bd=0, highlightthickness=2,
                         highlightcolor="#4472C4", relief="flat")
        entry.insert(0, text)
        entry.select_range(0, tk.END)
        entry.focus()
        entry.place(x=x, y=y, width=w, height=h)

        self._entry = entry
        self._current_item = row
        self._current_col = col_idx

        def save():
            if self._entry:
                new_val = entry.get()
                vals = list(self.item(row, "values"))
                while len(vals) <= col_idx:
                    vals.append("")
                vals[col_idx] = new_val
                self.item(row, values=vals)
                self._entry.destroy()
                self._entry = None

        entry.bind("<Return>", lambda e: (save(), self._move_down_edit()))
        entry.bind("<FocusOut>", lambda e: save())
        entry.bind("<Escape>", lambda e: entry.destroy())

    def _on_enter(self, event):
        if self._entry:
            self._entry.event_generate("<Return>")
        else:
            self._start_edit()

    def _move_down_edit(self):
        cur = self.focus()
        nxt = self.next(cur)
        if nxt:
            self.focus(nxt)
            self.selection_set(nxt)
            self.see(nxt)
            self.after(50, self._start_edit)

    def _on_down(self, event):
        if self._entry:
            self._entry.event_generate("<Return>")
        else:
            self._move_down_edit()
        return "break"

    def _on_up(self, event):
        if self._entry:
            self._entry.event_generate("<Return>")
        cur = self.focus()
        prv = self.prev(cur)
        if prv:
            self.focus(prv)
            self.selection_set(prv)
            self.see(prv)
            self.after(50, self._start_edit)
        return "break"

    def _on_left(self, event):
        if self._entry and not self._entry.selection_get():
            self._entry.event_generate("<Return>")
        return "break"

    def _on_right(self, event):
        if self._entry and not self._entry.selection_get():
            self._entry.event_generate("<Return>")
        return "break"


# ——————————————————————— Основний додаток ———————————————————————
class SADApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SAD v1.0 GOLD")
        self.root.geometry("1500x950")
        self.root.configure(bg="#f5f5f5")

        tk.Label(self.root, text="SAD v1.0 GOLD", font=("Arial", 42, "bold"),
                 fg="#1a3c6e", bg="#f5f5f5").pack(pady=30)
        tk.Label(self.root, text="Класичний аналіз + Shapiro-Wilk + таблиця як Excel",
                 font=("Arial", 16), bg="#f5f5f5").pack(pady=5)

        tk.Button(self.root, text="РОЗПОЧАТИ АНАЛІЗ", font=("Arial", 20, "bold"),
                  bg="#d32f2f", fg="white", width=40, height=2,
                  command=self.start).pack(pady=50)

        tk.Label(self.root, text="© Чаплоуцький А.М. • Уманський НУС • 2025",
                 fg="gray", bg="#f5f5f5").pack(side="bottom", pady=20)

        self.root.mainloop()

    def start(self):
        self.win = tk.Toplevel(self.root)
        self.win.title("SAD v1.0 GOLD — Ввід даних")
        self.win.geometry("1750x1000")

        # Панель інструментів
        tools = tk.Frame(self.win, bg="#e3f2fd")
        tools.pack(fill="x", padx=15, pady=12)
        tk.Button(tools, text="З Excel", command=self.from_excel).pack(side="left", padx=8)
        tk.Button(tools, text="Очистити", bg="#ff5252", fg="white",
                  command=self.clear).pack(side="left", padx=8)
        tk.Button(tools, text="АНАЛІЗ", font=("Arial", 20, "bold"),
                  bg="#d32f2f", fg="white", width=18,
                  command=self.analyze).pack(side="right", padx=30)

        # Таблиця
        frame = tk.Frame(self.win)
        frame.pack(fill="both", expand=True, padx=15, pady=5)

        self.tree = ExcelLikeTreeview(frame, columns=[f"c{i}" for i in range(20)], show="headings")
        for i in range(20):
            self.tree.heading(f"c{i}", text=str(i+1))
            self.tree.column(f"c{i}", width=115, anchor="center")
        for _ in range(40):
            self.tree.insert("", "end", values=[""] * 20)
        self.tree.pack(side="left", fill="both", expand=True)

        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(self.win, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")

        tk.Label(self.win, text="Подвійний клік / Enter — редагувати • Стрілки • Ctrl+V",
                 font=("Arial", 12, "bold"), fg="#d32f2f").pack(pady=8)

        # Результати
        res = tk.LabelFrame(self.win, text=" РЕЗУЛЬТАТИ ", font=("Arial", 14, "bold"))
        res.pack(fill="both", expand=True, padx=15, pady=10)
        self.result = scrolledtext.ScrolledText(res, font=("Consolas", 11))
        self.result.pack(fill="both", expand=True)

        self.win.bind_all("<Control-v>", lambda e: self.paste())

    # ——— Вставка / завантаження ———
    def paste(self):
        try:
            df = pd.read_clipboard(sep=r"\s+", header=None, dtype=str, engine="python")
            if df.empty: return
            self.clear()
            for _, row in df.iterrows():
                vals = row.astype(str).tolist() + [""] * (20 - len(row))
                self.tree.insert("", "end", values=vals[:20])
        except:
            pass

    def from_excel(self):
        path = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx *.xls")])
        if path:
            df = pd.read_excel(path, header=None).astype(str).fillna("")
            self.clear()
            for _, row in df.iterrows():
                vals = row.tolist() + [""] * (20 - len(row))
                self.tree.insert("", "end", values=vals[:20])

    def clear(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        for _ in range(40):
            self.tree.insert("", "end", values=[""] * 20)

    def get_data(self):
        data = []
        for iid in self.tree.get_children():
            row = [str(x).strip() for x in self.tree.item(iid, "values")]
            if any(row):
                data.append(row)
        return pd.DataFrame(data) if data else pd.DataFrame()

    # ——— АНАЛІЗ ———
    def analyze(self):
        try:
            df = self.get_data()
            if df.empty or df.shape[1] < 3:
                messagebox.showerror("Помилка", "Мінімум 3 стовпці (фактори + повторності)")
                return

            n_factors = simpledialog.askinteger("Фактори", "Кількість стовпців-факторів (1-3):", minvalue=1, maxvalue=3)
            if not n_factors:
                return

            factors = df.iloc[:, :n_factors]
            repeats = df.iloc[:, n_factors:].apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
            if repeats.shape[1] < 2:
                messagebox.showerror("Помилка", "Мінімум 2 повторності!")
                return

            n_rep = repeats.shape[1]
            values = repeats.stack().dropna().values
            grand_mean = values.mean()

            # Залишки для Shapiro-Wilk
            row_means = repeats.mean(axis=1)
            residuals = (repeats - row_means.values.reshape(-1, 1)).stack().dropna().values

            normality_text = "недостатньо даних"
            if len(residuals) >= 8:
                stat, p = shapiro(residuals)
                normality_text = f"W = {stat:.4f}, p = {p:.4f} → {'нормальний' if p > 0.05 else 'НЕ нормальний'}"

            # Класичний однофакторний аналіз
            group_means = repeats.mean(axis=1)
            total_ss = ((values - grand_mean) ** 2).sum()
            factor_ss = n_rep * ((group_means - grand_mean) ** 2).sum()
            error_ss = total_ss - factor_ss

            df_factor = len(group_means) - 1
            df_error = len(values) - len(group_means)
            ms_factor = factor_ss / df_factor if df_factor else 0
            ms_error = error_ss / df_error if df_error else 0
            F_calc = ms_factor / ms_error if ms_error else 0
            F_crit = f.ppf(0.95, df_factor, df_error) if df_error else 0
            p_val = 1 - f.cdf(F_calc, df_factor, df_error) if F_calc else 1

            t_crit = t.ppf(0.975, df_error)
            LSD = t_crit * np.sqrt(2 * ms_error / n_rep)

            eta_sq = factor_ss / total_ss * 100 if total_ss else 0

            # Букви істотності
            sorted_means = group_means.sort_values(ascending=False)
            letters = []
            letter = 'a'
            prev = sorted_means.iloc[0] + LSD + 1
            for idx, m in sorted_means.items():
                if m < prev - LSD:
                    letter = chr(ord(letter) + 1)
                variant = " | ".join(str(x) for x in factors.iloc[idx])
                letters.append(f"{variant:35} → {m:.3f} {letter}")
                prev = m

            # Звіт
            report = [
                "═" * 80,
                "           РЕЗУЛЬТАТИ ДИСПЕРСІЙНОГО АНАЛІЗУ (класичний метод)",
                "═" * 80,
                f"Варіантів: {len(group_means)}   Повторностей: {n_rep}   n = {len(values)}",
                f"Середнє по досліду: {grand_mean:.3f}",
                "",
                "Перевірка нормальності залишків (Shapiro-Wilk):",
                f"   {normality_text}",
                "",
                f"Fрозр = {F_calc:.3f}   F05 = {F_crit:.3f}",
                f"Вплив фактора — {'ІСТОТНИЙ' if F_calc > F_crit else 'НЕІСТОТНИЙ'} (p ≈ {p_val:.4f})",
                f"НІР₀.₅ = {LSD:.4f}    Вилучення впливу: {eta_sq:.1f}%",
                "",
                "СЕРЕДНІ ПО ВАРІАНТАХ:",
            ] + letters

            self.result.delete(1.0, tk.END)
            self.result.insert(tk.END, "\n".join(report))
            self.result.clipboard_clear()
            self.result.clipboard_append("\n".join(report))

            messagebox.showinfo("ГОТОВО!", f"Аналіз завершено!\nНормальність: {normality_text.split('→')[-1].strip()}\nНІР₀.₅ = {LSD:.4f}")

        except Exception as e:
            messagebox.showerror("ПОМИЛКА", str(e))


if __name__ == "__main__":
    SADApp()
