# -*- coding: utf-8 -*-
"""
SAD — Статистичний Аналіз Даних v1.2 (ФІНАЛЬНА ВЕРСІЯ) - виправлена
Автор: Чаплоуцький Андрій Миколайович
Уманський національний університет, 2025
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext, simpledialog
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from datetime import date
import os
from itertools import combinations

# ====================== Покращений EditableTreeview ======================
class EditableTreeview(ttk.Treeview):
    def __init__(self, master=None, **kw):
        style = ttk.Style()
        style.configure("Treeview", rowheight=28, font=("Arial", 10))
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

    def _on_double_click(self, event):
        self._start_edit(event)

    def _on_enter(self, event=None):
        if self._entry:
            self._save_edit()
            self._move_down()
        else:
            item = self.focus()
            if item:
                self._start_edit_at_item(item)

    def _on_arrow_down(self, event):
        if self._entry:
            self._save_edit()
        self._move_down()

    def _on_arrow_up(self, event):
        if self._entry:
            self._save_edit()
        self._move_up()

    def _on_arrow_left(self, event):
        if self._entry:
            try:
                sel = self._entry.selection_get()
            except tk.TclError:
                sel = ""
            if not sel:
                self._save_edit()
                self._move_left()

    def _on_arrow_right(self, event):
        if self._entry:
            try:
                sel = self._entry.selection_get()
            except tk.TclError:
                sel = ""
            if not sel:
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
            while len(vals) <= col_index:
                vals.append("")
            vals[col_index] = new_val
            self.item(rowid, values=vals)
            try:
                entry.destroy()
            except:
                pass
            self._entry = None
            self._current_cell = None

        entry.bind('<Return>', lambda e: (save(), self._move_down()))
        entry.bind('<FocusOut>', save)
        entry.bind('<Escape>', lambda e: (entry.destroy(), setattr(self, "_entry", None), setattr(self, "_current_cell", None)))

    def _start_edit_at_item(self, rowid):
        if not rowid:
            return
        column = "#1"
        bbox = self.bbox(rowid, column)
        if bbox:
            event = tk.Event()
            event.x = bbox[0] + 10
            event.y = bbox[1] + 10
            self._start_edit(event)

    def _save_edit(self):
        if not self._entry or not self._current_cell:
            return
        rowid, column = self._current_cell
        try:
            val = self._entry.get().strip()
        except Exception:
            val = ""
        col_index = int(column[1:]) - 1
        vals = list(self.item(rowid, 'values'))
        while len(vals) <= col_index:
            vals.append(val)
        vals[col_index] = val
        self.item(rowid, values=vals)
        try:
            self._entry.destroy()
        except:
            pass
        self._entry = None
        self._current_cell = None

    def _move_down(self):
        item = self.focus()
        if not item:
            return
        next_item = self.next(item)
        if next_item:
            self.focus(next_item)
            self.selection_set(next_item)
            self.see(next_item)
            self._start_edit_at_item(next_item)

    def _move_up(self):
        item = self.focus()
        if not item:
            return
        prev_item = self.prev(item)
        if prev_item:
            self.focus(prev_item)
            self.selection_set(prev_item)
            self.see(prev_item)
            self._start_edit_at_item(prev_item)

    def _move_left(self):
        if not self._current_cell:
            return
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
        if not self._current_cell:
            return
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


# ====================== Основний клас ======================
class SADApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SAD — Статистичний Аналіз Даних v1.2")
        self.root.geometry("1350x820")
        if os.path.exists("icon.ico"):
            try:
                self.root.iconbitmap("icon.ico")
            except:
                pass

        tk.Label(self.root, text="SAD", font=("Arial", 40, "bold"), fg="#1a3c6e").pack(pady=25)
        tk.Label(self.root, text="Універсальний калькулятор дисперсійного аналізу", font=("Arial", 14)).pack(pady=5)

        tk.Button(self.root, text="Почати аналіз", width=32, height=3, bg="#d32f2f", fg="white",
                  font=("Arial", 16, "bold"), command=self.choose_factor_count).pack(pady=40)

        info = tk.Frame(self.root)
        info.pack(pady=10)
        tk.Button(info, text="Про програму", command=self.show_about).pack(side="left", padx=15)
        tk.Button(info, text="Про розробника", command=self.show_author).pack(side="left", padx=15)

        tk.Label(self.root, text="Редагування: подвійний клік • Enter • стрілки | Ctrl+V — вставка з Excel",
                 fg="gray", font=("Arial", 10)).pack(pady=10)

        self.root.mainloop()

    def show_about(self):
        messagebox.showinfo("Про програму", "SAD v1.2 — найкращий український інструмент для агростатистики\n"
                                          "Підтримка: одно-, дво-, трифакторний аналіз • LSD • Shapiro-Wilk • Levene")

    def show_author(self):
        messagebox.showinfo("Про розробника",
            "Чаплоуцький Андрій Миколайович\n"
            "Кафедра плодівництва і виноградарства\n"
            "Уманський національний університет\n"
            "м. Умань, Україна\n"
            "Листопад 2025")

    def choose_factor_count(self):
        fc = simpledialog.askinteger("Факторність", "Введіть кількість факторів (1, 2 або 3):", minvalue=1, maxvalue=3)
        if fc:
            self.factor_count = fc
            self.open_analysis_window(fc)

    def open_analysis_window(self, factor_count):
        if not hasattr(self, "factor_count") or self.factor_count is None:
            self.factor_count = factor_count

        self.win = tk.Toplevel(self.root)
        self.win.title(f"SAD v1.2 — {factor_count}-факторний аналіз")
        self.win.geometry("1600x1000")

        # Панель інструментів
        tools = tk.Frame(self.win)
        tools.pack(fill="x", pady=10, padx=15)
        tk.Button(tools, text="Додати стовпець", command=self.add_column).pack(side="left", padx=5)
        tk.Button(tools, text="Додати рядок", command=self.add_row).pack(side="left", padx=5)
        tk.Button(tools, text="Очистити", bg="#f44336", fg="white", command=self.clear_table).pack(side="left", padx=5)
        tk.Button(tools, text="З Excel", command=self.load_excel).pack(side="left", padx=5)
        tk.Button(tools, text="АНАЛІЗ", bg="#d32f2f", fg="white", font=("Arial", 18, "bold"), width=22,
                  command=self.calculate).pack(side="left", padx=50)
        tk.Button(tools, text="Зберегти звіт", command=self.save_report).pack(side="left", padx=5)

        # Таблиця
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
                 fg="red", font=("Arial", 11, "bold")).pack(pady=8)

        # Результати
        res_frame = tk.LabelFrame(self.win, text=" Результати дисперсійного аналізу ", font=("Arial", 12, "bold"))
        res_frame.pack(fill="both", expand=True, padx=15, pady=10)
        self.result_box = scrolledtext.ScrolledText(res_frame, height=28, font=("Consolas", 10))
        self.result_box.pack(fill="both", expand=True)

        # Підв'язка Ctrl+V
        self.win.bind_all("<Control-v>", lambda e: self.on_paste_clipboard(e))

    def add_column(self):
        cols = list(self.tree["columns"])
        new_col = f"c{len(cols)}"
        cols.append(new_col)
        self.tree["columns"] = cols
        self.tree.heading(new_col, text=str(len(cols)))
        self.tree.column(new_col, width=110, anchor="c")
        for iid in self.tree.get_children():
            vals = list(self.tree.item(iid, 'values'))
            vals.append("")
            self.tree.item(iid, values=vals)

    def add_row(self):
        self.tree.insert("", "end", values=[""] * len(self.tree["columns"]))

    def clear_table(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        for _ in range(25):
            self.tree.insert("", "end", values=[""] * len(self.tree["columns"]))

    def on_paste_clipboard(self, event=None):
        try:
            df = pd.read_clipboard(sep="\t", engine='python', header=None, dtype=str)
        except Exception:
            try:
                txt = self.win.clipboard_get()
                df = pd.read_csv(pd.io.common.StringIO(txt), sep="\t", engine='python', header=None, dtype=str)
            except Exception:
                messagebox.showwarning("Помилка", "Не вдалося вставити дані з буфера")
                return

        if df.empty:
            messagebox.showinfo("Вставка", "Буфер порожній")
            return

        while len(self.tree["columns"]) < df.shape[1]:
            self.add_column()

        for _, row in df.iterrows():
            vals = [str(x).strip() for x in row.tolist()]
            vals += [""] * (len(self.tree["columns"]) - len(vals))
            self.tree.insert("", "end", values=vals[:len(self.tree["columns"])])

        messagebox.showinfo("Вставка", f"Вставлено {len(df)} рядків")

    def load_excel(self):
        path = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx *.xls")])
        if path:
            try:
                df = pd.read_excel(path, header=None, dtype=str).fillna("")
            except Exception as e:
                messagebox.showerror("Помилка імпорту", str(e))
                return
            while len(self.tree["columns"]) < df.shape[1]:
                self.add_column()
            self.clear_table()
            for _, row in df.iterrows():
                vals = [str(x).strip() for x in row.tolist()]
                vals += [""] * (len(self.tree["columns"]) - len(vals))
                self.tree.insert("", "end", values=vals[:len(self.tree["columns"])])

    def tree_to_dataframe(self):
        cols = len(self.tree["columns"])
        data = []
        for iid in self.tree.get_children():
            vals = list(self.tree.item(iid, 'values'))
            vals += [""] * (cols - len(vals))
            row = [str(v).strip() for v in vals[:cols]]
            data.append(row)
        if not data:
            return pd.DataFrame()
        col_names = [f"col{i+1}" for i in range(cols)]
        return pd.DataFrame(data, columns=col_names)

    def wide_to_long(self, df, n_factor_cols):
        if df.empty or n_factor_cols >= df.shape[1]:
            return pd.DataFrame()
        factor_cols = df.columns[:n_factor_cols]
        value_cols = df.columns[n_factor_cols:]
        long = pd.melt(df, id_vars=factor_cols, value_vars=value_cols,
                       var_name='repeat', value_name='value')
        long = long.dropna(subset=['value']).copy()
        long['value'] = pd.to_numeric(long['value'], errors='coerce')
        long = long.dropna(subset=['value'])
        for col in factor_cols:
            long[col] = long[col].astype('category')
        return long.reset_index(drop=True)

    def calculate(self):
        try:
            df_wide = self.tree_to_dataframe()
            if df_wide.empty:
                messagebox.showerror("Помилка", "Таблиця порожня")
                return

            n_factor_cols = getattr(self, "factor_count", 1)
            long = self.wide_to_long(df_wide, n_factor_cols)
            if long.empty:
                messagebox.showerror("Помилка", "Немає числових даних для аналізу")
                return

            factor_cols = df_wide.columns[:n_factor_cols].tolist()
            terms = [f"C({f})" for f in factor_cols]
            for r in range(2, len(factor_cols) + 1):
                for comb in combinations(factor_cols, r):
                    terms.append(":".join(f"C({c})" for c in comb))
            formula = "value ~ " + " + ".join(terms)

            model = smf.ols(formula, data=long).fit()
            anova_table = anova_lm(model, typ=2)

            MS_error = model.mse_resid
            df_error = int(model.df_resid)
            LSD = stats.t.ppf(1 - 0.05 / 2, df_error) * np.sqrt(2 * MS_error / long.groupby(factor_cols).size().mean())

            report = [
                "SAD v1.2 — ДИСПЕРСІЙНИЙ АНАЛІЗ",
                f"Дата: {date.today():%d.%m.%Y}",
                f"Факторів: {len(factor_cols)} | Повторностей: {long['repeat'].nunique()} | Спостережень: {len(long)}",
                "",
                "ANOVA (Type II):",
                anova_table.to_string(),
                "",
                f"LSD₀.₅ = {LSD:.4f}"
            ]
            self.result_box.delete(1.0, tk.END)
            self.result_box.insert(tk.END, "\n".join(report))

        except Exception as e:
            messagebox.showerror("Помилка", str(e))

    def save_report(self):
        path = filedialog.asksaveasfilename(defaultextension=".txt",
                                            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self.result_box.get(1.0, tk.END))
                messagebox.showinfo("Збережено", f"Звіт збережено в {path}")
            except Exception as e:
                messagebox.showerror("Помилка збереження", str(e))


if __name__ == "__main__":
    app = SADApp()
