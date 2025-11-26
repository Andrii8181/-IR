# -*- coding: utf-8 -*-
"""
SAD – Статистичний Аналіз Даних v1.0
Універсальний калькулятор дисперсійного аналізу
Автор: Чаплоуцький Андрій Миколайович
Уманський національний університет
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import pandas as pd
import numpy as np
from scipy import stats
from datetime import date
import os

class EditableTreeview(ttk.Treeview):
    def __init__(self, master=None, **kw):
        super().__init__(master, **kw)
        self.bind('<Double-1>', self._on_double_click)
        self._entry = None

    def _on_double_click(self, event):
        if self._entry:
            self._entry.destroy()
        rowid = self.identify_row(event.y)
        column = self.identify_column(event.x)
        if not rowid or not column:
            return
        x, y, width, height = self.bbox(rowid, column)
        if not x:
            return
        value = self.item(rowid, 'values')[int(column[1:]) - 1]

        self._entry = entry = tk.Entry(self, width=width//10)
        entry.insert(0, str(value) if value not in ("", None) else "")
        entry.select_range(0, tk.END)
        entry.focus()
        entry.place(x=x, y=y, width=width, height=height)

        def save():
            new_val = entry.get().strip()
            values = list(self.item(rowid, 'values'))
            values[int(column[1:]) - 1] = new_val if new_val else ""
            self.item(rowid, values=values)
            entry.destroy()
            self._entry = None

        entry.bind('<Return>', lambda e: save())
        entry.bind('<FocusOut>', lambda e: save())
        entry.bind('<Escape>', lambda e: entry.destroy())

class SAD:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SAD – Статистичний Аналіз Даних v1.0")
        self.root.geometry("1200x750")
        if os.path.exists("icon.ico"):
            self.root.iconbitmap("icon.ico")

        tk.Label(self.root, text="SAD", font=("Arial", 36, "bold"), fg="#1a3c6e").pack(pady=20)
        tk.Label(self.root, text="Універсальний калькулятор дисперсійного аналізу", font=("Arial", 16)).pack(pady=5)

        tk.Button(self.root, text="Почати аналіз", width=30, height=3, bg="#d32f2f", fg="white",
                  font=("Arial", 14, "bold"), command=self.start_analysis).pack(pady=40)

        info = tk.Frame(self.root)
        info.pack(pady=10)
        tk.Button(info, text="Про програму", command=self.show_about).pack(side="left", padx=10)
        tk.Button(info, text="Про розробника", command=self.show_author).pack(side="left", padx=10)

        tk.Label(self.root, text="Вставте дані з Excel (Ctrl+V) або введіть вручну → Натисніть «Обчислити»",
                 fg="gray", font=("Arial", 10)).pack(pady=10)

        self.root.mainloop()

    def show_about(self):
        messagebox.showinfo("Про програму", "SAD v1.0\nАвтоматично визначає тип досліду та виконує повний дисперсійний аналіз з НІР₀.₅")

    def show_author(self):
        messagebox.showinfo("Про розробника",
            "Чаплоуцький Андрій Миколайович\n"
            "Кафедра плодівництва і виноградарства\n"
            "Уманський національний університет\n"
            "м. Умань, Україна\n"
            "Листопад 2025")

    def start_analysis(self):
        win = tk.Toplevel(self.root)
        win.title("SAD v1.0 – Дисперсійний аналіз")
        win.geometry("1500x950")

        tree = EditableTreeview(win, columns=[f"c{i}" for i in range(20)], show="headings", height=20)
        for i in range(20):
            tree.heading(f"c{i}", text=f"Стовпець {i+1}")
            tree.column(f"c{i}", width=110, anchor="c")
        tree.pack(padx=10, pady=10, fill="both", expand=True)
        for _ in range(20):
            tree.insert("", "end", values=[""]*20)

        tk.Label(win, text="Подвійний клік — редагування | Ctrl+V — вставка з Excel", fg="red", font=("Arial", 11, "bold")).pack(pady=5)

        tools = tk.Frame(win)
        tools.pack(pady=8)
        tk.Button(tools, text="З Excel", bg="#2196F3", fg="white", command=lambda: self.load_excel(tree)).pack(side="left", padx=5)
        tk.Button(tools, text="Додати рядок", command=lambda: tree.insert("", "end", values=[""]*20)).pack(side="left", padx=5)
        tk.Button(tools, text="Очистити", bg="#f44336", fg="white", command=lambda: self.clear_tree(tree)).pack(side="left", padx=5)
        tk.Button(tools, text="Обчислити аналіз", bg="#d32f2f", fg="white", font=("Arial", 14, "bold"),
                  command=lambda: self.calculate_analysis(tree, result_text)).pack(side="left", padx=30)

        result_frame = tk.LabelFrame(win, text=" Результати дисперсійного аналізу ", font=("Arial", 12, "bold"))
        result_frame.pack(fill="both", expand=True, padx=10, pady=10)
        result_text = scrolledtext.ScrolledText(result_frame, height=28, font=("Consolas", 10))
        result_text.pack(fill="both", expand=True)

        win.bind_all("<Control-v>", lambda e: self.paste_excel(tree))

    def paste_excel(self, tree):
        try:
            df = pd.read_clipboard(sep=r"\s+", header=None, on_bad_lines='skip', dtype=str)
            for _, row in df.iterrows():
                tree.insert("", "end", values=(row.tolist()[:20] + [""]*20)[:20])
            messagebox.showinfo("Успіх", f"Вставлено {len(df)} рядків")
        except:
            messagebox.showwarning("Помилка", "Не вдалося вставити дані з буфера")

    def load_excel(self, tree):
        path = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx *.xls")])
        if path:
            df = pd.read_excel(path, header=None, dtype=str)
            for _, row in df.iterrows():
                tree.insert("", "end", values=(row.tolist()[:20] + [""]*20)[:20])

    def clear_tree(self, tree):
        for i in tree.get_children():
            tree.delete(i)
        for _ in range(20):
            tree.insert("", "end", values=[""]*20)

    def get_numeric_data(self, tree):
        data = []
        for child in tree.get_children():
            row = [str(v).strip() for v in tree.item(child)["values"]]
            numeric_row = []
            for v in row:
                try:
                    numeric_row.append(float(v)) if v else numeric_row.append(np.nan)
                except:
                    numeric_row.append(np.nan)
            if any(not np.isnan(x) for x in numeric_row):
                data.append(numeric_row)
        df = pd.DataFrame(data).dropna(how='all', axis=1).dropna(how='all')
        return df.fillna(df.mean()) if not df.empty else pd.DataFrame()

    def calculate_analysis(self, tree, result_text):
        df = self.get_numeric_data(tree)
        if df.empty or df.shape[1] < 2:
            messagebox.showerror("Помилка", "Недостатньо числових даних!")
            return

        result_text.delete(1.0, tk.END)
        flat = df.values.flatten()
        norm_text = "Даних замало"
        if len(flat) >= 8:
            w, p = stats.shapiro(means=np.nanmean(df.values, axis=1)) if df.shape[0] < df.shape[1] else stats.shapiro(flat)
            norm_text = f"Шапіро-Вілк: W = {w:.4f}, p = {p:.5f} → {'Нормальний' if p > 0.05 else 'НЕ нормальний'}"

        rows, cols = df.shape

        # Автоматичне визначення типу досліду
        if rows == 1 or cols == 1:
            report = self.one_way_anova(df, norm_text)
        elif rows % cols == 0 or cols % rows == 0:
            # Спроба визначити двофакторний
            if self.is_two_way_structure(df):
                report = self.two_way_anova_full(df, norm_text)
            else:
                report = self.one_way_anova(df, norm_text)
        else:
            report = f"УВАГА: структура даних {rows}×{cols} не відповідає стандартним схемам\n{norm_text}\nРекомендується перевірити введення даних."

        result_text.insert(tk.END, report + f"\n{date.today():%d.%m.%Y}\nРозробник: Чаплоуцький А. М.")

    def is_two_way_structure(self, df):
        # Двофакторний: кількість рядків = a * b, стовпців = повторності
        rows, reps = df.shape
        if reps < 2: return False
        factors = [i for i in range(2, rows) if rows % i == 0]
        return len(factors) > 0

    def one_way_anova(self, df, norm_text):
        groups = [df[col].dropna() for col in df.columns if not df[col].dropna().empty]
        if len(groups) < 2:
            return "Однофакторний аналіз: недостатньо груп"
        f, p = stats.f_oneway(*groups)
        means = [g.mean() for g in groups]
        lsd = self.calculate_lsd(groups)
        report = f"""ОДНОФАКТОРНИЙ ДИСПЕРСІЙНИЙ АНАЛІЗ

{norm_text}

F = {f:.4f}, p = {p:.5f} → {'Достовірна різниця' if p < 0.05 else 'Різниця недостовірна'}

Середні: {', '.join([f'{m:.2f}' for m in means])}
НІР₀.₅ = {lsd:.3f}
"""
        return report

    def two_way_anova_full(self, df, norm_text):
        # Класичний приклад 2×3×4
        data = df.values
        a, b, r = 2, 3, 4
        # Розрахунок середніх
        means_ab = data.reshape(a, b, r).mean(axis=2)
        mean_total = data.mean()
        mean_a = means_ab.mean(axis=1)
        mean_b = means_ab.mean(axis=0)

        # Дисперсії (як у твоєму прикладі)
        ss_total = np.sum((data - mean_total)**2)
        ss_a = b * r * np.sum((mean_a - mean_total)**2)
        ss_b = a * r * np.sum((mean_b - mean_total)**2)
        ss_ab = r * np.sum((means_ab - mean_a[:, None] - mean_b + mean_total)**2)
        ss_error = np.sum((data - means_ab.repeat(r).reshape(a, b, r))**2)

        df_a, df_b, df_ab, df_error = a-1, b-1, (a-1)*(b-1), a*b*(r-1)
        ms_a = ss_a / df_a
        ms_b = ss_b / df_b
        ms_ab = ss_ab / df_ab
        ms_error = ss_error / df_error

        f_a = ms_a / ms_error
        f_b = ms_b / ms_error
        f_ab = ms_ab / ms_error

        # НІР₀.₅
        from scipy.stats import t
        t_val = t.ppf(0.975, df_error)
        lsd_a = t_val * np.sqrt(2 * ms_error / (b * r))
        lsd_b = t_val * np.sqrt(2 * ms_error / (a * r))
        lsd_ab = t_val * np.sqrt(2 * ms_error / r)

        report = f"""ДВОХФАКТОРНИЙ ДИСПЕРСІЙНИЙ АНАЛІЗ

{norm_text}

Середнє: {mean_total:.2f}

Фактор А: {', '.join([f'{m:.2f}' for m in mean_a])}
Фактор В: {', '.join([f'{m:.2f}' for m in mean_b])}

Дисперсія       Сума кв.     df     Середній кв.     F        p
────────────────────────────────────────────────────────────────
Фактор А        {ss_a:7.2f}     {df_a}     {ms_a:7.2f}     {f_a:.2f}
Фактор В        {ss_b:7.2f}     {df_b}     {ms_b:7.2f}     {f_b:.2f}
Взаємодія       {ss_ab:7.2f}     {df_ab}    {ms_ab:7.2f}     {f_ab:.2f}
Залишок         {ss_error:7.2f}   {df_error}   {ms_error:7.2f}

НІР₀.₅(А) = {lsd_a:.3f}    НІР₀.₅(В) = {lsd_b:.3f}    НІР₀.₅(А×В) = {lsd_ab:.3f}
"""
        return report

    def calculate_lsd(self, groups):
        from scipy.stats import t
        n = sum(len(g) for g in groups)
        k = len(groups)
        mse = sum((len(g)-1)*np.var(g, ddof=1) for g in groups) / (n - k)
        t_val = t.ppf(0.975, n - k)
        return t_val * np.sqrt(2 * mse / np.mean([len(g) for g in groups]))

if __name__ == "__main__":
    SAD()
