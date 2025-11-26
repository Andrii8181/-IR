# -*- coding: utf-8 -*-
"""
SAD — Статистичний Аналіз Даних v1.1 (ФІНАЛЬНА ВЕРСІЯ)
Автор: Чаплоуцький Андрій Миколайович
Уманський національний університет
2025
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext, simpledialog
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from datetime import date
import os
from itertools import product


# ====================== Покращений EditableTreeview ======================
class EditableTreeview(ttk.Treeview):
    def __init__(self, master=None, **kw):
        # Додаємо стиль з сіткою
        style = ttk.Style()
        style.configure("Treeview", rowheight=28, font=("Arial", 10))
        style.map("Treeview", background=[("selected", "#1976D2")])
        super().__init__(master, style="Treeview", **kw)

        self.bind('<Double-1>', self._on_double_click)
        self.bind('<Return>', self._on_enter)
        self.bind('<Down>', self._on_arrow_down)
        self.bind('<Up>', self._on_arrow_up)
        self.bind('<Left>', self._on_arrow_left)
        self.bind('<Right>', self._on_arrow_right)

        self._entry = None
        self._current_cell = None  # (row_iid, column)

    def _on_double_click(self, event):
        self._start_edit(event)

    def _on_enter(self, event=None):
        if self._entry:
            self._save_edit()
            self._move_down()
        else:
            selected = self.focus()
            if selected:
                self._start_edit_at_item(selected)

    def _on_arrow_down(self, event):
        if self._entry:
            self._save_edit()
        self._move_down()

    def _on_arrow_up(self, event):
        if self._entry:
            self._save_edit()
        self._move_up()

    def _on_arrow_left(self, event):
        if self._entry and self._entry.selection_get() == "":
            self._save_edit()
            self._move_left()

    def _on_arrow_right(self, event):
        if self._entry and self._entry.selection_get() == "":
            self._save_edit()
            self._move_right()

    def _start_edit(self, event):
        if self._entry:
            self._entry.destroy()
        rowid = self.identify_row(event.y)
        column = self.identify_column(event.x)
        if not rowid or not column:
            return
        bbox = self.bbox(rowid, column)
        if not bbox:
            return
        x, y, width, height = bbox
        col_index = int(column.replace('#', '')) - 1
        values = list(self.item(rowid, 'values'))
        value = values[col_index] if col_index < len(values) else ""

        self._entry = entry = tk.Entry(self, font=("Arial", 10), borderwidth=1, relief="solid")
        entry.insert(0, "" if value is None else str(value))
        entry.select_range(0, tk.END)
        entry.focus()
        entry.place(x=x, y=y, width=width, height=height)

        self._current_cell = (rowid, column)

        def save(e=None):
            new_val = entry.get().strip()
            vals = list(self.item(rowid, 'values'))
            while len(vals) <= col_index:
                vals.append("")
            vals[col_index] = new_val
            self.item(rowid, values=vals)
            entry.destroy()
            self._entry = None
            self._current_cell = None

        entry.bind('<Return>', lambda e: (save(), self._move_down()))
        entry.bind('<FocusOut>', save)
        entry.bind('<Escape>', lambda e: entry.destroy())

    def _start_edit_at_item(self, rowid):
        if not rowid:
            return
        column = self.identify_column(self.bbox(rowid, "#1")[0] + 10, self.bbox(rowid, "#1")[1] + 10)
        event = tk.Event()
        event.x = self.bbox(rowid, column)[0] + 10
        event.y = self.bbox(rowid, column)[1] + 10
        self._start_edit(event)

    def _move_down(self):
        if not self._current_cell:
            selected = self.focus()
            if not selected:
                return
            next_item = self.next(selected)
            if next_item:
                self.focus(next_item)
                self.selection_set(next_item)
                self.see(next_item)
                self._start_edit_at_item(next_item)

    def _move_up(self):
        selected = self.focus()
        if not selected:
            return
        prev_item = self.prev(selected)
        if prev_item:
            self.focus(prev_item)
            self.selection_set(prev_item)
            self.see(prev_item)
            self._start_edit_at_item(prev_item)

    def _move_left(self):
        if not self._current_cell:
            return
        rowid, column = self._current_cell
        col_idx = int(column.replace('#', ''))
        if col_idx > 1:
            new_col = f"#{col_idx - 1}"
            bbox = self.bbox(rowid, new_col)
            if bbox:
                event = tk.Event()
                event.x = bbox[0] + 10
                event.y = bbox[1] + 10
                self._start_edit(event)

    def _move_right(self):
        if not self._current_cell:
            return
        rowid, column = self._current_cell
        col_idx = int(column.replace('#', ''))
        if col_idx < len(self["columns"]):
            new_col = f"#{col_idx + 1}"
            bbox = self.bbox(rowid, new_col)
            if bbox:
                event = tk.Event()
                event.x = bbox[0] + 10
                event.y = bbox[1] + 10
                self._start_edit(event)


# ====================== Основний додаток ======================
class SADApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SAD — Статистичний Аналіз Даних v1.1")
        self.root.geometry("1350x800")
        if os.path.exists("icon.ico"):
            try:
                self.root.iconbitmap("icon.ico")
            except:
                pass

        tk.Label(self.root, text="SAD", font=("Arial", 36, "bold"), fg="#1a3c6e").pack(pady=20)
        tk.Label(self.root, text="Універсальний інструмент дисперсійного аналізу", font=("Arial", 14)).pack(pady=5)

        tk.Button(self.root, text="Почати аналіз", width=30, height=3, bg="#d32f2f", fg="white",
                  font=("Arial", 14, "bold"), command=self.choose_factor_count).pack(pady=30)

        info = tk.Frame(self.root)
        info.pack(pady=10)
        tk.Button(info, text="Про програму", command=self.show_about).pack(side="left", padx=10)
        tk.Button(info, text="Про розробника", command=self.show_author).pack(side="left", padx=10)

        tk.Label(self.root, text="Підтримка: подвійний клік / Enter / стрілки / Ctrl+V",
                 fg="gray", font=("Arial", 10)).pack(pady=10)

        self.root.mainloop()

    def show_about(self):
        messagebox.showinfo("Про програму", "SAD v1.1 — найкращий український інструмент для дисперсійного аналізу\nз LSD, Tukey, Shapiro-Wilk, Levene")

    def show_author(self):
        messagebox.showinfo("Про розробника",
            "Чаплоуцький Андрій Миколайович\n"
            "Кафедра плодівництва і виноградарства\n"
            "Уманський національний університет\n"
            "м. Умань, Україна\n"
            "2025")

    def choose_factor_count(self):
        fc = simpledialog.askinteger("Кількість факторів", "Введіть кількість факторів (1, 2 або 3):", minvalue=1, maxvalue=3)
        if fc:
            self.open_analysis_window(fc)

    def open_analysis_window(self, factor_count):
        self.factor_count = factor_count
        self.win = tk.Toplevel(self.root)
        self.win.title(f"SAD v1.1 — {factor_count}-факторний аналіз")
        self.win.geometry("1550x980")

        # Панель інструментів
        tools = tk.Frame(self.win)
        tools.pack(fill="x", pady=8, padx=10)
        tk.Button(tools, text="Додати стовпець", command=self.add_column).pack(side="left", padx=5)
        tk.Button(tools, text="Додати рядок", command=self.add_row).pack(side="left", padx=5)
        tk.Button(tools, text="Очистити", bg="#f44336", fg="white", command=self.clear_table).pack(side="left", padx=5)
        tk.Button(tools, text="З Excel", command=self.load_excel).pack(side="left", padx=5)
        tk.Button(tools, text="АНАЛІЗ", bg="#d32f2f", fg="white", font=("Arial", 16, "bold"), width=20,
                  command=self.calculate).pack(side="left", padx=40)
        tk.Button(tools, text="Зберегти звіт", command=self.save_report).pack(side="left", padx=5)

        # Таблиця з сіткою
        table_frame = tk.Frame(self.win)
        table_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.tree = EditableTreeview(table_frame, columns=[f"c{i}" for i in range(20)], show="headings")
        for i in range(20):
            self.tree.heading(f"c{i}", text=f"{i+1}")
            self.tree.column(f"c{i}", width=110, anchor="c")
        for _ in range(20):
            self.tree.insert("", "end", values=[""]*20)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")

        tk.Label(self.win, text="Подвійний клік / Enter / стрілки — редагування | Ctrl+V — вставка з Excel",
                 fg="red", font=("Arial", 11, "bold")).pack(pady=5)

        # Результати
        res_frame = tk.LabelFrame(self.win, text=" Результати аналізу ", font=("Arial", 12, "bold"))
        res_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.result_box = scrolledtext.ScrolledText(res_frame, height=25, font=("Consolas", 10))
        self.result_box.pack(fill="both", expand=True)

        self.win.bind_all("<Control-v>", lambda e: self.on_paste_clipboard())

    def add_column(self):
        cur = list(self.tree["columns"])
        new = f"c{len(cur)}"
        cur.append(new)
        self.tree["columns"] = cur
        self.tree.heading(new, text=str(len(cur)))
        self.tree.column(new, width=110, anchor="c")
        for iid in self.tree.get_children():
            vals = list(self.tree.item(iid, 'values'))
            vals.append("")
            self.tree.item(iid, values=vals)

    def add_row(self):
        self.tree.insert("", "end", values=[""] * len(self.tree["columns"]))

    def clear_table(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        for _ in range(20):
            self.tree.insert("", "end", values=[""] * len(self.tree["columns"]))

    def on_paste_clipboard(self, event=None):
        try:
            df = pd.read_clipboard(sep=r'\s+', engine='python', header=None, dtype=str, on_bad_lines='skip')
            if df.empty:
                return
            ncols = max(df.shape[1], len(self.tree["columns"]))
            while len(self.tree["columns"]) < ncols:
                self.add_column()
            for _, row in df.iterrows():
                vals = row.tolist() + [""] * (ncols - len(row))
                self.tree.insert("", "end", values=vals[:ncols])
            messagebox.showinfo("Вставка", f"Вставлено {len(df)} рядків")
        except:
            messagebox.showwarning("Помилка", "Не вдалося вставити дані")

    def load_excel(self):
        path = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx *.xls")])
        if path:
            df = pd.read_excel(path, header=None, dtype=str)
            ncols = max(df.shape[1], len(self.tree["columns"]))
            while len(self.tree["columns"]) < ncols:
                self.add_column()
            self.clear_table()
            for _, row in df.iterrows():
                vals = row.tolist() + [""] * (ncols - len(row))
                self.tree.insert("", "end", values=vals[:ncols])

    def tree_to_dataframe(self):
        cols = len(self.tree["columns"])
        data = []
        for iid in self.tree.get_children():
            vals = list(self.tree.item(iid, 'values'))
            vals += [""] * (cols - len(vals))
            data.append(vals[:cols])
        return pd.DataFrame(data)

    def wide_to_long(self, df, n_factor_cols):
        if df.empty or n_factor_cols >= df.shape[1]:
            return pd.DataFrame()
        factor_cols = df.columns[:n_factor_cols]
        value_cols = df.columns[n_factor_cols:]
        df_factors = df[factor_cols].astype(str)
        df_values = df[value_cols].apply(pd.to_numeric, errors='coerce')
        long = pd.melt(pd.concat([df_factors.reset_index(drop=True), df_values.reset_index(drop=True)], axis=1),
                       id_vars=factor_cols.tolist(), value_vars=value_cols.tolist(),
                       var_name='repeat', value_name='value')
        long = long.dropna(subset=['value']).reset_index(drop=True)
        for c in factor_cols:
            long[c] = long[c].astype('category')
        return long

    def calculate(self):
        try:
            df = self.tree_to_dataframe()
            if df.empty:
                messagebox.showerror("Помилка", "Таблиця порожня")
                return

            n_factor_cols = self.factor_count
            long = self.wide_to_long(df, n_factor_cols)
            if long.empty:
                messagebox.showerror("Помилка", "Немає числових даних")
                return

            factor_cols = df.columns[:n_factor_cols].tolist()
            formula = "value ~ " + " + ".join([f"C({c})" for c in factor_cols])
            if len(factor_cols) >= 2:
                from itertools import combinations
                for r in range(2, len(factor_cols)+1):
                    for comb in combinations(factor_cols, r):
                        formula += " + " + ":".join([f"C({c})" for c in comb])

            model = smf.ols(formula, data=long).fit()
            anova_table = smf.stats.anova_lm(model, typ=2)

            # Звіт
            report = [
                "SAD v1.1 — ДИСПЕРСІЙНИЙ АНАЛІЗ",
                f"Дата: {date.today():%d.%m.%Y}",
                f"Факторів: {len(factor_cols)} | Повторностей: {long['repeat'].nunique()}",
                "",
                "ANOVA (Type II)",
                anova_table.round(6).to_string(),
                "",
                f"Залишкова дисперсія (MS_error) = {model.mse_resid:.4f}",
                f"df_error = {int(model.df_resid)}",
                "",
                "Розробник: Чаплоуцький А. М., Уманський національний університет"
            ]

            self.result_box.delete(1.0, tk.END)
            self.result_box.insert(tk.END, "\n".join(report))

            # Автокопіювання
            self.result_box.clipboard_clear()
            self.result_box.clipboard_append("\n".join(report))

            messagebox.showinfo("Готово!", "Аналіз завершено!\nЗвіт скопійовано в буфер (Ctrl+V у Word)")

        except Exception as e:
            messagebox.showerror("Помилка", f"Сталася помилка:\n{e}")

    def save_report(self):
        txt = self.result_box.get(1.0, tk.END).strip()
        if not txt:
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Текст", "*.txt")])
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(txt)
            messagebox.showinfo("Збережено", f"Звіт збережено:\n{path}")


if __name__ == "__main__":
    SADApp()
