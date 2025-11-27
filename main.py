# -*- coding: utf-8 -*-
"""
SAD v1.0 
Автор: Чаплоуцький Андрій Миколайович, Уманський НУС, 2025
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext, simpledialog
import pandas as pd
import numpy as np
from scipy.stats import f, t, shapiro
import os


class ExcelLikeTreeview(ttk.Treeview):
    def __init__(self, master, **kw):
        # Налаштування стилю — чітка сітка як у Excel
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Excel.Treeview", 
                        background="white",
                        foreground="black",
                        rowheight=28,
                        fieldbackground="white",
                        borderwidth=1,
                        relief="solid",
                        font=("Arial", 11))
        style.configure("Excel.Treeview.Heading",
                        background="#e0e0e0",
                        foreground="black",
                        font=("Arial", 11, "bold"),
                        relief="flat")
        style.map('Excel.Treeview', background=[('selected', '#4472C4')])
        style.map('Excel.Treeview.Heading', background=[('active', '#c0c0c0')])

        super().__init__(master, style="Excel.Treeview", **kw)
        self._entry = None
        self._current = None

        self.bind("<Double-1>", self._start_edit)
        self.bind("<Return>", self._on_enter)
        self.bind("<Down>", self._on_down)
        self.bind("<Up>", self._on_up)
        self.bind("<Left>", self._on_left)
        self.bind("<Right>", self._on_right)

    def _start_edit(self, event=None):
        if event:
            item = self.identify_row(event.y)
            col = self.identify_column(event.x)
        else:
            item = self.focus()
            col = self.bbox(item, "#1")
            if not col: return
            col = "#1"

        if not item: return
        bbox = self.bbox(item, col)
        if not bbox: return

        x, y, width, height = bbox
        col_idx = int(col[1:]) - 1
        values = self.item(item, "values")
        text = values[col_idx] if col_idx < len(values) else ""

        entry = tk.Entry(self, font=("Arial", 11), bd=0, relief="flat", highlightthickness=2, highlightcolor="#4472C4")
        entry.insert(0, text)
        entry.selection_range(0, tk.END)
        entry.focus()
        entry.place(x=x, y=y, width=width, height=height)

        self._entry = entry
        self._current = (item, col_idx)

        def save():
            if self._entry:
                new_val = entry.get()
                vals = list(self.item(item, "values"))
                while len(vals) <= col_idx:
                    vals.append("")
                vals[col_idx] = new_val
                self.item(item, values=vals)
                self._entry.destroy()
                self._entry = None
                self._current = None

        entry.bind("<Return>", lambda e: (save(), self._move_down_and_edit()))
        entry.bind("<FocusOut>", lambda e: save())
        entry.bind("<Escape>", lambda e: entry.destroy())

    def _on_enter(self, event):
        if self._entry:
            self._entry.event_generate("<Return>")
        else:
            self._start_edit()

    def _move_down_and_edit(self):
        item = self.focus()
        next_item = self.next(item)
        if next_item:
            self.focus(next_item)
            self.selection_set(next_item)
            self.see(next_item)
            self.after(10, lambda: self._start_edit())

    def _on_down(self, event):
        if self._entry:
            self._entry.event_generate("<Return>")
        else:
            :
            self._move_down_and_edit()
        return "break"

    def _on_up(self, event):
        if self._entry:
            self._entry.event_generate("<Return>")
        item = self.focus()
        prev = self.prev(item)
        if prev:
            self.focus(prev)
            self.selection_set(prev)
            self.see(prev)
            self.after(10, lambda: self._start_edit())
        return "break"

    def _on_left(self, event):
        if self._entry and self._entry.selection_get() == "":
            self._entry.event_generate("<Return>")
            item, col_idx = self._current
            if col_idx > 0:
                new_col = f"#{col_idx}"
                self.after(10, lambda: self._start_edit_at(item, new_col))
        return "break"

    def _on_right(self, event):
        if self._entry and self._entry.selection_get() == "":
            self._entry.event_generate("<Return>")
            item, col_idx = self._current
            if col_idx < len(self["columns"]) - 1:
                new_col = f"#{col_idx + 2}"
                self.after(10, lambda: self._start_edit_at(item, new_col))
        return "break"

    def _start_edit_at(self, item, col):
        bbox = self.bbox(item, col)
        if bbox:
            event = tk.Event()
            event.x = bbox[0] + 5
            event.y = bbox[1] + 5
            self._start_edit(event)


class SADApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SAD v1.0 GOLD — Український калькулятор польового досліду")
        self.root.geometry("1500x950")
        self.root.configure(bg="#f5f5f5")

        tk.Label(self.root, text="SAD v1.0 GOLD", font=("Arial", 42, "bold"), fg="#1a3c6e", bg="#f5f5f5").pack(pady=30)
        tk.Label(self.root, text="Класичний аналіз + нормальність + таблиця як в Excel", font=("Arial", 16), bg="#f5f5f5").pack(pady=5)

        tk.Button(self.root, text="РОЗПОЧАТИ АНАЛІЗ", font=("Arial", 20, "bold"), bg="#d32f2f", fg="white",
                  width=40, height=2, command=self.start).pack(pady=50)

        tk.Label(self.root, text="© Чаплоуцький А.М. • Уманський НУС • 2025", fg="gray", bg="#f5f5f5").pack(side="bottom", pady=20)
        self.root.mainloop()

    def start(self):
        self.win = tk.Toplevel(self.root)
        self.win.title("SAD v1.0 GOLD — Ввід даних")
        self.win.geometry("1750x1000")

        tools = tk.Frame(self.win, bg="#e3f2fd")
        tools.pack(fill="x", padx=15, pady=12)
        tk.Button(tools, text="З Excel", font=("Arial", 12), command=self.from_excel).pack(side="left", padx=8)
        tk.Button(tools, text="Очистити", font=("Arial", 12), bg="#ff5252", fg="white", command=self.clear).pack(side="left", padx=8)
        tk.Button(tools, text="АНАЛІЗ", font=("Arial", 20, "bold"), bg="#d32f2f", fg="white", width=18,
                  command=self.analyze).pack(side="right", padx=30)

        table_frame = tk.Frame(self.win)
        table_frame.pack(fill="both", expand=True, padx=15, pady=5)

        self.tree = ExcelLikeTreeview(table_frame, columns=[f"c{i}" for i in range(20)], show="headings")
        for i in range(20):
            self.tree.heading(f"c{i}", text=str(i+1))
            self.tree.column(f"c{i}", width=115, anchor="center")
        for _ in range(40):
            self.tree.insert("", "end", values=[""]*20)
        self.tree.pack(side="left", fill="both", expand=True)

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(self.win, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")

        tk.Label(self.win, text="Подвійний клік або Enter — редагувати • Стрілки — переміщення • Ctrl+V — вставка з Excel",
                 font=("Arial", 12, "bold"), fg="#d32f2f").pack(pady=8)

        res_frame = tk.LabelFrame(self.win, text=" РЕЗУЛЬТАТИ АНАЛІЗУ ", font=("Arial", 14, "bold"), bg="#fffde7")
        res_frame.pack(fill="both", expand=True, padx=15, pady=10)
        self.result = scrolledtext.ScrolledText(res_frame, font=("Consolas", 11), bg="white")
        self.result.pack(fill="both", expand=True)

        self.win.bind_all("<Control-v>", lambda e: self.paste())

    def paste(self):
        try:
            df = pd.read_clipboard(sep=r"\s+", header=None, dtype=str, engine='python')
            if df.empty: return
            self.clear()
            for _, row in df.iterrows():
                vals = row.astype(str).tolist() + [""] * (20 - len(row))
                self.tree.insert("", "end", values=vals[:20])
            messagebox.showinfo("Успіх", f"Вставлено {len(df)} рядків")
        except Exception as e:
            pass

    def from_excel(self):
        path = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx *.xls")])
        if path:
            df = pd.read_excel(path, header=None).astype(str).replace("nan", "")
            self.clear()
            for _, row in df.iterrows():
                vals = row.tolist() + [""] * (20 - len(row))
                self.tree.insert("", "end", values=vals[:20])

    def clear(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        for _ in range(40):
            self.tree.insert("", "end", values=[""]*20)

    def get_data(self):
        data = []
        for iid in self.tree.get_children():
            row = [str(x).strip() for x in self.tree.item(iid, "values")]
            if any(row): data.append(row)
        return pd.DataFrame(data) if data else pd.DataFrame()

    def analyze(self):
        try:
            df = self.get_data()
            if df.empty or df.shape[1] < 3:
                messagebox.showerror("Помилка", "Потрібно мінімум 3 стовпці: фактори + повторності")
                return

            n_factors = simpledialog.askinteger("Фактори", "Кількість стовпців-факторів (1-3):", minvalue=1, maxvalue=3)
            if not n_factors: return

            factors = df.iloc[:, :n_factors]
            repeats = df.iloc[:, n_factors:].apply(pd.to_numeric, errors='coerce')
            repeats = repeats.dropna(axis=1, how='all')
            if repeats.shape[1] < 2:
                messagebox.showerror("Помилка", "Мінімум 2 повторності!")
                return

            n_rep = repeats.shape[1]
            values = repeats.stack().dropna().values
            grand_mean = values.mean()

            # Залишки для Shapiro
            row_means = repeats.mean(axis=1)
            residuals = (repeats.sub(row_means, axis=0)).stack().dropna().values

            if len(residuals) >= 8:
                shapiro_stat, shapiro_p = shapiro(residuals)
                normality = "нормальний" if shapiro_p > 0.05 else "НЕ нормальний"
            else:
                shapiro_stat = shapiro_p = None
                normality = "недостатньо даних"

            # Класичний аналіз
            group_means = repeats.mean(axis=1)
            total_ss = ((values - grand_mean)**2).sum()
            factor_ss = n_rep * ((group_means - grand_mean)**2).sum()
            error_ss = total_ss - factor_ss

            df_factor = len(group_means) - 1
            df_error = len(values) - len(group_means)
            ms_factor = factor_ss / df_factor if df_factor else 0
            ms_error = error_ss / df_error if df_error else 0
            F_calc = ms_factor / ms_error if ms_error else 0
            F_crit = f.ppf(0.95, df_factor, df_error) if df_error else 0
            p_value = 1 - f.cdf(F_calc, df_factor, df_error) if F_calc else 1

            t_crit = t.ppf(0.975, df_error)
            LSD = t_crit * np.sqrt(2 * ms_error / n_rep)

            eta_sq = factor_ss / total_ss * 100 if total_ss else 0

            # Букви
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

            report = [
                "Р Е З У Л Ь Т А Т И   Д И С П Е Р С І Й Н О Г О   А Н А Л І З У",
                f"Варіантів: {len(group_means)} | Повторностей: {n_rep} | n = {len(values)}",
                f"Середнє по досліду: {grand_mean:.3f}",
                "",
                "Перевірка нормальності залишків (Shapiro-Wilk):",
                f"   {'W = {:.4f}, p = {:.4f} → {}'.format(shapiro_stat, shapiro_p, normality.upper()) if shapiro_p is not None else '   Недостатньо даних'}",
                "",
                f"Fрозр = {F_calc:.3f}   F05 = {F_crit:.3f} → вплив {'ІСТОТНИЙ' if F_calc > F_crit else 'НЕІСТОТНИЙ'}",
                f"НІР₀.₅ = {LSD:.4f}    Вилучення впливу: {eta_sq:.1f}%",
                "",
                "СЕРЕДНІ ПО ВАРІАНТАХ:",
            ] + letters

            self.result.delete(1.0, tk.END)
            self.result.insert(tk.END, "\n".join(report))
            self.result.clipboard_clear()
            self.result.clipboard_append("\n".join(report))
            messagebox.showinfo("УСПІХ!", f"Аналіз завершено!\nНормальність: {normality}\nНІР = {LSD:.4f}")

        except Exception as e:
            messagebox.showerror("ПОМИЛКА", str(e))


if __name__ == "__main__":
    SADApp()
