# -*- coding: utf-8 -*-
"""
SAD v1.0 GOLD — ОСТАТОЧНА ВЕРСІЯ
Повна видимість сітки + повний звіт + поведінка як у Excel
Автор: Чаплоуцький Андрій Миколайович, Уманський НУС, 2025
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext, simpledialog
import pandas as pd
import numpy as np
from scipy.stats import f, t, shapiro


class ExcelTable(ttk.Treeview):
    def __init__(self, parent):
        # Стиль — чітка сітка як у Excel
        style = ttk.Style()
        style.theme_use('default')
        style.configure("Excel.Treeview",
                        background="white",
                        foreground="black",
                        rowheight=26,
                        fieldbackground="white",
                        borderwidth=1,
                        relief="solid",
                        font=("Calibri", 11))
        style.configure("Excel.Treeview.Heading",
                        background="#f0f0f0",
                        foreground="black",
                        font=("Calibri", 11, "bold"))
        style.map("Excel.Treeview",
                  background=[("selected", "#4472C4")],
                  foreground=[("selected", "white")])

        super().__init__(parent, style="Excel.Treeview", selectmode="browse")

        self.entry = None
        self._edit_item = None
        self._edit_col = None

        # Події
        self.bind("<Double-1>", self._start_edit)
        self.bind("<Return>", self._on_enter)
        self.bind("<Down>", self._on_arrow)
        self.bind("<Up>", self._on_arrow)
        self.bind("<Left>", self._on_arrow)
        self.bind("<Right>", self._on_arrow)

    def _start_edit(self, event=None):
        if self.entry:
            return

        item = self.focus()
        if not item:
            return

        column = self.identify_column(event.x) if event else "#1"
        bbox = self.bbox(item, column)
        if not bbox:
            return

        col_idx = int(column[1:]) - 1
        value = self.item(item, "values")[col_idx]

        # Створюємо Entry з чіткою рамкою
        entry = tk.Entry(self, font=("Calibri", 11), bd=2, relief="solid", bg="#ffffe0")
        entry.insert(0, value)
        entry.select_range(0, tk.END)
        entry.focus()

        x, y, width, height = bbox
        entry.place(x=x, y=y, width=width+2, height=height+2)

        self.entry = entry
        self._edit_item = item
        self._edit_col = col_idx

        def save():
            if self.entry:
                new_val = entry.get()
                values = list(self.item(item, "values"))
                values[col_idx] = new_val
                self.item(item, values=values)
                entry.destroy()
                self.entry = None

        entry.bind("<Return>", lambda e: (save(), self._move_down_edit()))
        entry.bind("<FocusOut>", lambda e: save())
        entry.bind("<Escape>", lambda e: entry.destroy())

    def _on_enter(self, event):
        if self.entry:
            self.entry.event_generate("<Return>")
        else:
            self._start_edit()

    def _move_down_edit(self):
        item = self.focus()
        next_item = self.next(item)
        if next_item:
            self.focus(next_item)
            self.selection_set(next_item)
            self.see(next_item)
            self.after(50, self._start_edit)

    def _on_arrow(self, event):
        if self.entry:
            self.entry.event_generate("<Return>")
        if event.keysym == "Down":
            self._move_down_edit()
        return "break"


class SADApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SAD v1.0 GOLD — Статистичний аналіз даних")
        self.root.geometry("1600x1000")
        self.root.configure(bg="#f5f5f5")

        # Заголовок
        tk.Label(self.root, text="SAD v1.0 GOLD", font=("Arial", 44, "bold"),
                 fg="#1a3c6e", bg="#f5f5f5").pack(pady=30)
        tk.Label(self.root, text="Повний класичний аналіз + таблиця як в Excel",
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
        self.win.geometry("1800x1050")

        # Панель
        top = tk.Frame(self.win, bg="#e8f5e8")
        top.pack(fill="x", padx=15, pady=10)
        tk.Button(top, text="З Excel", font=("Arial", 12), command=self.load_excel).pack(side="left", padx=5)
        tk.Button(top, text="Очистити", bg="#ff5252", fg="white", command=self.clear).pack(side="left", padx=5)
        tk.Button(top, text="АНАЛІЗ", bg="#d32f2f", fg="white", font=("Arial", 20, "bold"),
                  width=20, command=self.analyze).pack(side="right", padx=30)

        # Таблиця з чіткою сіткою
        frame = tk.Frame(self.win)
        frame.pack(fill="both", expand=True, padx=15, pady=10)

        self.table = ExcelTable(frame)
        self.table["columns"] = [str(i) for i in range(1, 21)]
        for col in self.table["columns"]:
            self.table.heading(col, text=col)
            self.table.column(col, width=110, anchor="center")
        for _ in range(50):
            self.table.insert("", "end", values=[""] * 20)

        # Сітка — найголовніше!
        self.table.configure(show="headings")
        self.table.pack(side="left", fill="both", expand=True)

        # Скроллбари
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.table.yview)
        hsb = ttk.Scrollbar(self.win, orient="horizontal", command=self.table.xview)
        self.table.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")

        tk.Label(self.win, text="Подвійний клік / Enter — редагувати • Стрілки — навігація • Ctrl+V — з Excel",
                 font=("Arial", 12, "bold"), fg="#d32f2f").pack(pady=5)

        # Результати
        res = tk.LabelFrame(self.win, text=" РЕЗУЛЬТАТИ АНАЛІЗУ ", font=("Arial", 14, "bold"))
        res.pack(fill="both", expand=True, padx=15, pady=10)
        self.result = scrolledtext.ScrolledText(res, font=("Consolas", 11), bg="#fffdf0")
        self.result.pack(fill="both", expand=True)

        self.win.bind_all("<Control-v>", lambda e: self.paste())

    def paste(self):
        try:
            df = pd.read_clipboard(sep=r"\s+", header=None, dtype=str)
            self.clear()
            for _, row in df.iterrows():
                vals = row.astype(str).tolist() + [""] * (20 - len(row))
                self.table.insert("", "end", values=vals[:20])
        except:
            pass

    def load_excel(self):
        path = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx")])
        if path:
            df = pd.read_excel(path, header=None).astype(str).fillna("")
            self.clear()
            for _, row in df.iterrows():
                vals = row.tolist() + [""] * (20 - len(row))
                self.table.insert("", "end", values=vals[:20])

    def clear(self):
        for i in self.table.get_children():
            self.table.delete(i)
        for _ in range(50):
            self.table.insert("", "end", values=[""] * 20)

    def get_df(self):
        data = []
        for iid in self.table.get_children():
            row = [str(x).strip() for x in self.table.item(iid, "values")]
            if any(row):
                data.append(row)
        return pd.DataFrame(data)

    def analyze(self):
        try:
            df = self.get_df()
            if df.shape[1] < 3:
                messagebox.showerror("Помилка", "Мінімум 3 стовпці")
                return

            n_factors = simpledialog.askinteger("Фактори", "Кількість факторів (1-3):", minvalue=1, maxvalue=3)
            if not n_factors: return

            factors = df.iloc[:, :n_factors]
            reps = df.iloc[:, n_factors:].apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
            if reps.shape[1] < 2:
                messagebox.showerror("Помилка", "Мінімум 2 повторності")
                return

            n_rep = reps.shape[1]
            values = reps.stack().dropna().values
            grand_mean = values.mean()

            # Shapiro-Wilk
            row_means = reps.mean(axis=1)
            residuals = (reps - row_means.values.reshape(-1, 1)).stack().dropna().values
            normality = "недостатньо даних"
            if len(residuals) >= 8:
                stat, p = shapiro(residuals)
                normality = "нормальний" if p > 0.05 else "НЕ нормальний"

            # Дисперсійний аналіз
            group_means = reps.mean(axis=1)
            total_ss = ((values - grand_mean)**2).sum()
            factor_ss = n_rep * ((group_means - grand_mean)**2).sum()
            error_ss = total_ss - factor_ss

            df_factor = len(group_means) - 1
            df_error = len(values) - len(group_means)
            ms_factor = factor_ss / df_factor
            ms_error = error_ss / df_error
            F_calc = ms_factor / ms_error
            F_crit = f.ppf(0.95, df_factor, df_error)
            p_val = 1 - f.cdf(F_calc, df_factor, df_error)

            t_crit = t.ppf(0.975, df_error)
            LSD = t_crit * np.sqrt(2 * ms_error / n_rep)
            eta = factor_ss / total_ss * 100

            # Букви
            sorted_idx = group_means.sort_values(ascending=False).index
            letters = {}
            current_letter = 'a'
            prev_mean = group_means[sorted_idx[0]] + LSD
            for idx in sorted_idx:
                if group_means[idx] < prev_mean - LSD:
                    current_letter = chr(ord(current_letter) + 1)
                letters[idx] = current_letter
                prev_mean = group_means[idx]

            # Звіт
            report = [
                "╔" + "═" * 78 + "╗",
                "║            РЕЗУЛЬТАТИ ДИСПЕРСІЙНОГО АНАЛІЗУ (2025)            ║",
                "╚" + "═" * 78 + "╝",
                f"Варіантів: {len(group_means)}   Повторностей: {n_rep}   n = {len(values)}",
                f"Середнє по досліду: {grand_mean:.3f}",
                "",
                f"Shapiro-Wilk (залишки): {normality}",
                "",
                f"Fрозр = {F_calc:.3f}   F05 = {F_crit:.3f} → {'ІСТОТНИЙ' if F_calc > F_crit else 'НЕІСТОТНИЙ'}",
                f"НІР₀.₅ = {LSD:.4f}",
                f"Вилучення впливу фактора: {eta:.1f}%",
                "",
                "СЕРЕДНІ ПО ВАРІАНТАХ (з буквами істотності):",
            ]

            for idx in sorted_idx:
                variant = " | ".join(str(x) for x in factors.iloc[idx])
                mean = group_means[idx]
                let = letters[idx]
                report.append(f"  {variant:30} → {mean:7.3f} {let}")

            report.append("\nРозробник: Чаплоуцький Андрій Миколайович")
            report.append("Уманський національний університет садівництва")

            text = "\n".join(report)
            self.result.delete(1.0, tk.END)
            self.result.insert(tk.END, text)
            self.result.clipboard_clear()
            self.result.clipboard_append(text)

            messagebox.showinfo("ГОТОВО!", "Аналіз завершено!\nЗвіт скопійовано в буфер!")

        except Exception as e:
            messagebox.showerror("ПОМИЛКА", str(e))


if __name__ == "__main__":
    SADApp()
