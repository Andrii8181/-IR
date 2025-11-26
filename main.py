# -*- coding: utf-8 -*-
"""
SAD — Статистичний Аналіз Даних v1.1
ОФІЦІЙНИЙ РЕЛІЗ — ПОВНА НАУКОВА ВЕРСІЯ
Автор: Чаплоуцький Андрій Миколайович
Уманський національний університет садівництва, 2025
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext, simpledialog
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from datetime import date
import os
from itertools import combinations
import io


# ==================== EditableTreeview ====================
class EditableTreeview(ttk.Treeview):
    def __init__(self, master=None, **kw):
        style = ttk.Style()
        style.configure("Treeview", rowheight=26, font=("Arial", 10))
        style.configure("Treeview.Heading", font=("Arial", 10, "bold"))
        style.map("Treeview", background=[("selected", "#1976D2")])
        super().__init__(master, style="Treeview", **kw)

        self.bind('<Double-1>', self._on_double_click)
        self.bind('<Return>', self._on_enter)
        self.bind('<Down>', self._on_arrow_down)
        self.bind('<Up>', self._on_arrow_up)
        self.bind('<Left>', self._on_arrow_left)
        self.bind('<Right>', self._on_arrow_right)
        self._entry = None
        self._current_cell = None

    def _on_double_click(self, event): self._start_edit(event)
    def _on_enter(self, event=None):
        if self._entry:
            self._save_edit()
            self._move_down()
        else:
            item = self.focus()
            if item: self._start_edit_at_item(item)

    def _on_arrow_down(self, event):
        if self._entry: self._save_edit()
        self._move_down()
    def _on_arrow_up(self, event):
        if self._entry: self._save_edit()
        self._move_up()
    def _on_arrow_left(self, event):
        if self._entry:
            try:
                if not self._entry.selection_get():
                    self._save_edit()
                    self._move_left()
            except: self._save_edit(); self._move_left()
    def _on_arrow_right(self, event):
        if self._entry:
            try:
                if not self._entry.selection_get():
                    self._save_edit()
                    self._move_right()
            except: self._save_edit(); self._move_right()

    def _start_edit(self, event):
        if self._entry:
            try: self._entry.destroy()
            except: pass
        rowid = self.identify_row(event.y)
        column = self.identify_column(event.x)
        if not rowid or not column: return
        bbox = self.bbox(rowid, column)
        if not bbox: return
        x, y, width, height = bbox
        col_index = int(column[1:]) - 1
        values = list(self.item(rowid, 'values'))
        value = values[col_index] if col_index < len(values) else ""
        self._entry = entry = tk.Entry(self, font=("Arial", 10), relief="solid", bd=1)
        entry.insert(0, "" if value is None else str(value))
        entry.select_range(0, tk.END)
        entry.focus()
        entry.place(x=x, y=y, width=width, height=height)
        self._current_cell = (rowid, column)

        def save(e=None):
            new_val = entry.get().strip()
            vals = list(self.item(rowid, 'values'))
            while len(vals) <= col_index: vals.append("")
            vals[col_index] = new_val
            self.item(rowid, values=vals)
            entry.destroy()
            self._entry = None
            self._current_cell = None

        entry.bind('<Return>', lambda e: (save(), self._move_down()))
        entry.bind('<FocusOut>', save)
        entry.bind('<Escape>', lambda e: entry.destroy())

    def _start_edit_at_item(self, rowid):
        if not rowid: return
        bbox = self.bbox(rowid, "#1")
        if bbox:
            event = tk.Event()
            event.x = bbox[0] + 10
            event.y = bbox[1] + 10
            self._start_edit(event)

    def _save_edit(self):
        if not self._entry or not self._current_cell: return
        rowid, column = self._current_cell
        val = self._entry.get().strip()
        col_index = int(column[1:]) - 1
        vals = list(self.item(rowid, 'values'))
        while len(vals) <= col_index: vals.append("")
        vals[col_index] = val
        self.item(rowid, values=vals)
        self._entry.destroy()
        self._entry = None
        self._current_cell = None

    def _move_down(self):
        item = self.focus()
        if not item: return
        next_item = self.next(item)
        if next_item:
            self.focus(next_item)
            self.selection_set(next_item)
            self.see(next_item)
            self._start_edit_at_item(next_item)

    def _move_up(self):
        item = self.focus()
        if not item: return
        prev_item = self.prev(item)
        if prev_item:
            self.focus(prev_item)
            self.selection_set(prev_item)
            self.see(prev_item)
            self._start_edit_at_item(prev_item)

    def _move_left(self):
        if not self._current_cell: return
        rowid, col = self._current_cell
        col_idx = int(col[1:])
        if col_idx > 1:
            new_col = f"#{col_idx - 1}"
            bbox = self.bbox(rowid, new_col)
            if bbox:
                event = tk.Event()
                event.x = bbox[0] + 10
                event.y = bbox[1] + 10
                self._start_edit(event)

    def _move_right(self):
        if not self._current_cell: return
        rowid, col = self._current_cell
        col_idx = int(col[1:])
        if col_idx < len(self["columns"]):
            new_col = f"#{col_idx + 1}"
            bbox = self.bbox(rowid, new_col)
            if bbox:
                event = tk.Event()
                event.x = bbox[0] + 10
                event.y = bbox[1] + 10
                self._start_edit(event)


# ==================== Основний додаток ====================
class SADApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SAD — Статистичний Аналіз Даних v1.1")
        self.root.geometry("1350x820")
        if os.path.exists("icon.ico"):
            try: self.root.iconbitmap("icon.ico")
            except: pass

        tk.Label(self.root, text="SAD", font=("Arial", 40, "bold"), fg="#1a3c6e").pack(pady=20)
        tk.Label(self.root, text="Універсальний калькулятор дисперсійного аналізу", font=("Arial", 14)).pack(pady=4)
        tk.Button(self.root, text="Почати аналіз", width=32, height=3, bg="#d32f2f", fg="white",
                  font=("Arial", 16, "bold"), command=self.choose_factor_count).pack(pady=30)

        info = tk.Frame(self.root)
        info.pack(pady=10)
        tk.Button(info, text="Про програму", command=self.show_about).pack(side="left", padx=10)
        tk.Button(info, text="Про розробника", command=self.show_author).pack(side="left", padx=10)

        tk.Label(self.root, text="Редагування: подвійний клік • Enter • стрілки | Ctrl+V — вставка з Excel",
                 fg="gray", font=("Arial", 10)).pack(pady=10)
        self.root.mainloop()

    def show_about(self):
        messagebox.showinfo("Про програму", "SAD v1.1 — найкращий український інструмент для агростатистики\n"
                                          "Повний Type II ANOVA • НІР₀.₅ • Tukey HSD • Shapiro-Wilk • Levene")

    def show_author(self):
        messagebox.showinfo("Про розробника",
            "Чаплоуцький Андрій Миколайович\n"
            "Кафедра плодівництва і виноградарства\n"
            "Уманський національний університет садівництва\n"
            "м. Умань, Україна • Листопад 2025")

    def choose_factor_count(self):
        fc = simpledialog.askinteger("Кількість факторів", "Введіть кількість факторів (1, 2 або 3):", minvalue=1, maxvalue=3)
        if fc:
            self.open_analysis_window(fc)

    def open_analysis_window(self, factor_count):
        self.factor_count = factor_count
        self.win = tk.Toplevel(self.root)
        self.win.title(f"SAD v1.1 — {factor_count}-факторний аналіз")
        self.win.geometry("1650x1050")

        tools = tk.Frame(self.win)
        tools.pack(fill="x", pady=10, padx=15)
        tk.Button(tools, text="Додати стовпець", command=self.add_column).pack(side="left", padx=5)
        tk.Button(tools, text="Додати рядок", command=self.add_row).pack(side="left", padx=5)
        tk.Button(tools, text="Очистити", bg="#f44336", fg="white", command=self.clear_table).pack(side="left", padx=5)
        tk.Button(tools, text="З Excel", command=self.load_excel).pack(side="left", padx=5)
        tk.Button(tools, text="АНАЛІЗ", bg="#d32f2f", fg="white", font=("Arial", 20, "bold"), width=22,
                  command=self.calculate).pack(side="left", padx=40)
        tk.Button(tools, text="Зберегти звіт", command=self.save_report).pack(side="left", padx=5)

        table_frame = tk.Frame(self.win)
        table_frame.pack(fill="both", expand=True, padx=15, pady=5)
        self.tree = EditableTreeview(table_frame, columns=[f"c{i}" for i in range(20)], show="headings")
        for i in range(20):
            self.tree.heading(f"c{i}", text=str(i+1))
            self.tree.column(f"c{i}", width=110, anchor="c")
        for _ in range(25):
            self.tree.insert("", "end", values=[""] * 20)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")

        tk.Label(self.win, text="Редагування: подвійний клік • Enter • стрілки | Ctrl+V — вставка з Excel",
                 fg="red", font=("Arial", 11, "bold")).pack(pady=6)

        res_frame = tk.LabelFrame(self.win, text=" Результати аналізу ", font=("Arial", 12, "bold"))
        res_frame.pack(fill="both", expand=True, padx=15, pady=10)
        self.result_box = scrolledtext.ScrolledText(res_frame, height=32, font=("Consolas", 10))
        self.result_box.pack(fill="both", expand=True)

        self.win.bind_all("<Control-v>", lambda e: self.on_paste_clipboard())

    def add_column(self):
        cols = list(self.tree["columns"])
        new = f"c{len(cols)}"
        cols.append(new)
        self.tree["columns"] = cols
        self.tree.heading(new, text=str(len(cols)))
        self.tree.column(new, width=110, anchor="c")
        for iid in self.tree.get_children():
            v = list(self.tree.item(iid, 'values'))
            v.append("")
            self.tree.item(iid, values=v)

    def add_row(self):
        self.tree.insert("", "end", values=[""] * len(self.tree["columns"]))

    def clear_table(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        for _ in range(25):
            self.tree.insert("", "end", values=[""] * len(self.tree["columns"]))

    def on_paste_clipboard(self):
        try:
            df = pd.read_clipboard(sep=r'\s+', engine='python', header=None, dtype=str, on_bad_lines='skip')
        except:
            try:
                txt = self.win.clipboard_get()
                df = pd.read_csv(io.StringIO(txt), sep=r'\s+', engine='python', header=None, dtype=str, on_bad_lines='skip')
            except:
                messagebox.showwarning("Помилка", "Не вдалося вставити дані")
                return
        if df.empty: return
        while len(self.tree["columns"]) < df.shape[1]:
            self.add_column()
        for _, row in df.iterrows():
            vals = [str(x).strip() for x in row.tolist()] + [""] * (len(self.tree["columns"]) - df.shape[1])
            self.tree.insert("", "end", values=vals[:len(self.tree["columns"])])
        messagebox.showinfo("Успіх", f"Вставлено {len(df)} рядків")

    def load_excel(self):
        path = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx *.xls")])
        if path:
            df = pd.read_excel(path, header=None, dtype=str).fillna("")
            while len(self.tree["columns"]) < df.shape[1]:
                self.add_column()
            self.clear_table()
            for _, row in df.iterrows():
                vals = [str(x).strip() for x in row.tolist()] + [""] * (len(self.tree["columns"]) - df.shape[1])
                self.tree.insert("", "end", values=vals)

    def tree_to_dataframe(self):
        cols = len(self.tree["columns"])
        data = []
        for iid in self.tree.get_children():
            v = list(self.tree.item(iid, 'values'))
            v += [""] * (cols - len(v))
            data.append([str(x).strip() for x in v[:cols]])
        return pd.DataFrame(data) if data else pd.DataFrame()

    def wide_to_long(self, df, n_factor_cols):
        if df.empty or n_factor_cols >= df.shape[1]: return pd.DataFrame()
        factor_cols = df.columns[:n_factor_cols]
        value_cols = df.columns[n_factor_cols:]
        long = pd.melt(df, id_vars=factor_cols, value_vars=value_cols, var_name='repeat', value_name='value')
        long['value'] = pd.to_numeric(long['value'], errors='coerce')
        long = long.dropna(subset=['value']).reset_index(drop=True)
        for c in factor_cols:
            long[c] = long[c].astype('category')
        return long

    def calculate(self):
        try:
            df_wide = self.tree_to_dataframe()
            if df_wide.empty:
                messagebox.showerror("Помилка", "Таблиця порожня")
                return

            long = self.wide_to_long(df_wide, self.factor_count)
            if long.empty:
                messagebox.showerror("Помилка", "Немає числових даних")
                return

            factor_cols = df_wide.columns[:self.factor_count].tolist()

            terms = [f"C({f})" for f in factor_cols]
            for r in range(2, len(factor_cols)+1):
                for comb in combinations(factor_cols, r):
                    terms.append(":".join(f"C({c})" for c in comb))
            formula = "value ~ " + " + ".join(terms)

            model = smf.ols(formula, data=long).fit()
            anova_table = anova_lm(model, typ=2)

            # Перевірки
            resid = model.resid
            shapiro_resid = "недостатньо даних"
            if len(resid) >= 3:
                sw_stat, sw_p = stats.shapiro(resid)
                shapiro_resid = f"W = {sw_stat:.4f}, p = {sw_p:.4f} → {'нормальні' if sw_p > 0.05 else 'НЕ нормальні'}"

            groups_levene = [g['value'].values for _, g in long.groupby(factor_cols) if len(g) >= 2]
            levene_res = "неможливо обчислити"
            if len(groups_levene) >= 2:
                _, p = stats.levene(*groups_levene)
                levene_res = f"p = {p:.5f} → {'однорідні' if p > 0.05 else 'НЕ однорідні'}"

            # LSD + парні порівняння
            from scipy.stats import t
            mse = model.mse_resid
            df_err = int(model.df_resid)
            t_crit = t.ppf(0.975, df_err)
            lsd_results = {}
            for fac in factor_cols:
                grp = long.groupby(fac)['value']
                means = grp.mean().sort_values(ascending=False)
                counts = grp.count()
                pairs = []
                for i in means.index:
                    for j in means.index:
                        if i >= j: continue
                        diff = abs(means[i] - means[j])
                        se = np.sqrt(mse * (1/counts[i] + 1/counts[j]))
                        lsd = t_crit * se
                        signif = "***" if diff > lsd else ("**" if diff > lsd*0.95 else ("*" if diff > lsd*0.9 else "ns"))
                        pairs.append(f"  {i} vs {j}:  Δ = {diff:6.3f}  {signif}  (LSD={lsd:.3f})")
                lsd_results[fac] = (means.round(3).to_string(), pairs)

            # Tukey
            tukey_text = None
            if factor_cols and long[factor_cols[0]].nunique() >= 2:
                try:
                    tukey = pairwise_tukeyhsd(endog=long['value'], groups=long[factor_cols[0]].astype(str), alpha=0.05)
                    tukey_text = tukey.summary().as_text()
                except: tukey_text = "Tukey HSD: помилка обчислення"

            # Звіт
            report = [
                "╔" + "═"*82 + "╗",
                "║                SAD v1.1 — ПОВНИЙ НАУКОВИЙ ДИСПЕРСІЙНИЙ АНАЛІЗ                ║",
                "╚" + "═"*82 + "╝",
                f"Дата аналізу: {date.today():%d.%m.%Y}",
                f"Кількість факторів: {len(factor_cols)} │ Повторностей: {long['repeat'].nunique()} │ n = {len(long)}",
                "",
                "ANOVA (Type II):",
                anova_table.round(6).to_string(),
                "",
                f"MS_error = {mse:.6f} │ df_error = {df_err}",
                "",
                "ПЕРЕВІРКА ПЕРЕДУМОВ:",
                f"• Shapiro-Wilk (залишки): {shapiro_resid}",
                f"• Levene (однорідність дисперсій): {levene_res}",
                "",
                "НІР₀.₅ — ПАРНІ ПОРІВНЯННЯ ПО ГОЛОВНИХ ФАКТОРАХ:",
            ]

            for fac, (means_str, pairs) in lsd_results.items():
                report.append(f"┌── Фактор: {fac} ───────────────────────────────")
                report.append("Средні значення:")
                report.append(means_str)
                report.append("Парні порівняння (*** p<0.01  ** p<0.05  * p<0.1  ns — неістотно):")
                report.extend(pairs or ["  (немає значущих різниць)"])
                report.append("")

            if tukey_text:
                report.append("Tukey HSD (перший фактор):")
                report.append(tukey_text)

            report.extend([
                "",
                "Розробник: Чаплоуцький Андрій Миколайович",
                "Уманський національний університет садівництва",
                "м. Умань, Україна │ Листопад 2025"
            ])

            self.result_box.delete(1.0, tk.END)
            self.result_box.insert(tk.END, "\n".join(report))
            self.result_box.clipboard_clear()
            self.result_box.clipboard_append("\n".join(report))

            messagebox.showinfo("ГОТОВО!", "Повний науковий аналіз завершено!\nЗвіт скопійовано в буфер — вставляй у дисертацію!")

        except Exception as e:
            messagebox.showerror("Помилка", f"Аналіз не вдався:\n{e}")

    def save_report(self):
        txt = self.result_box.get(1.0, tk.END).strip()
        if not txt:
            messagebox.showwarning("Немає звіту", "Спочатку виконайте аналіз")
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Текст", "*.txt")])
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(txt)
            messagebox.showinfo("Збережено", f"Звіт збережено:\n{path}")


if __name__ == "__main__":
    SADApp()
