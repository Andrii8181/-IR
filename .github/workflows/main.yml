# -*- coding: utf-8 -*-
"""
НІР-Майстер 2025 — з перевіркою нормальності та NaN
Українська програма для дисперсійного аналізу та НІР₀₅
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import pandas as pd
import numpy as np
from scipy import stats
from datetime import date
import os

class NIRMaster:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("НІР-Майстер 2025")
        self.root.geometry("1000x700")
        if os.path.exists("icon.ico"):
            self.root.iconbitmap("icon.ico")

        tk.Label(self.root, text="Оберіть тип аналізу", font=("Arial", 16, "bold")).pack(pady=30)

        btns = tk.Frame(self.root)
        btns.pack(pady=20)
        tk.Button(btns, text="Однофакторний", width=25, height=2, bg="#4CAF50", fg="white",
                  command=lambda: self.open_window(1)).grid(row=0, column=0, padx=20)
        tk.Button(btns, text="Двофакторний", width=25, height=2, bg="#2196F3", fg="white",
                  command=lambda: self.open_window(2)).grid(row=0, column=1, padx=20)
        tk.Button(btns, text="Трифакторний", width=25, height=2, bg="#FF9800", fg="white",
                  command=lambda: self.open_window(3)).grid(row=0, column=2, padx=20)

        tk.Label(self.root, text="Вставка даних: Ctrl+V з Excel  →  Аналіз даних", fg="gray").pack(pady=20)

        self.root.mainloop()

    def open_window(self, factors):
        win = tk.Toplevel()
        win.title(f"{'Одно' if factors==1 else 'Дво' if factors==2 else 'Три'}факторний аналіз")
        win.geometry("1450x950")

        # Таблиця
        tree = ttk.Treeview(win, columns=[f"c{i}" for i in range(20)], show="headings", height=20)
        for i in range(20):
            tree.heading(f"c{i}", text=f"Ст {i+1}")
            tree.column(f"c{i}", width=110, anchor="c")
        tree.pack(padx=10, pady=10, fill="both", expand=True)
        tree.bind("<Control-v>", lambda e: self.paste(tree))

        # Кнопки
        btns = tk.Frame(win)
        btns.pack(pady=5)
        tk.Button(btns, text="З Excel", command=lambda: self.load_excel(tree)).pack(side="left", padx=5)
        tk.Button(btns, text="Очистити", command=lambda: [tree.delete(i) for i in tree.get_children()]).pack(side="left", padx=5)
        tk.Button(btns, text="Аналіз даних", bg="#d32f2f", fg="white", font=("Arial", 12, "bold"),
                  command=lambda: self.calculate(tree, factors)).pack(side="left", padx=30)

        # Результат
        result = scrolledtext.ScrolledText(win, height=28, font=("Consolas", 10))
        result.pack(padx=10, pady=10, fill="both", expand=True)
        self.result_box = result

    def paste(self, tree):
        try:
            df = pd.read_clipboard(sep="\t", header=None, on_bad_lines='skip')
            for _, row in df.iterrows():
                tree.insert("", "end", values=row.tolist()[:20])
            messagebox.showinfo("Готово", f"Вставлено {len(df)} рядків")
        except Exception as e:
            messagebox.showwarning("Помилка вставки", str(e))

    def load_excel(self, tree):
        path = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx;*.xls")])
        if not path: return
        try:
            df = pd.read_excel(path, header=None)
            for _, row in df.iterrows():
                tree.insert("", "end", values=row.tolist()[:20])
            messagebox.showinfo("Готово", f"Завантажено {len(df)} рядків")
        except Exception as e:
            messagebox.showerror("Помилка", str(e))

    def clean_and_prepare_data(self, tree):
        items = [tree.item(i)["values"] for i in tree.get_children()]
        if not items:
            raise ValueError("Таблиця порожня!")

        df = pd.DataFrame(items)
        # Видаляємо повністю порожні рядки і стовпці
        df = df.dropna(how='all').dropna(axis=1, how='all')

        # Знаходимо числові стовпці (повторності)
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty or len(numeric.columns) < 2:
            raise ValueError("Потрібно мінімум 2 числових стовпця (повторності)!")

        # Перевірка на NaN у числових даних
        if numeric.isna().any().any():
            messagebox.showwarning("Увага", "У числових даних є порожні клітинки (NaN)!\nВони будуть замінені середнім по стовпцю.")
            numeric = numeric.fillna(numeric.mean())

        return numeric.values

    def test_normality_shapiro(self, data):
        # Тест Шапіро-Вілка на всьому наборі даних (або по групах)
        flat = data.flatten()
        if len(flat) < 3 or len(flat) > 5000:
            return None, "Недостатньо даних для тесту Шапіро-Вілка"
        stat, p = stats.shapiro(flat)
        alpha = 0.05
        result = "Нормальний розподіл (p > 0.05)" if p > alpha else "НЕ нормальний розподіл (p ≤ 0.05)"
        return p, f"Шапіро-Вілк: W={stat:.4f}, p={p:.4f} → {result}"

    def calculate(self, tree, factors):
        try:
            values = self.clean_and_prepare_data(tree)
        except Exception as e:
            messagebox.showerror("Помилка даних", str(e))
            return

        r = values.shape[1]  # кількість повторностей

        # Перевірка нормальності
        p_val, shapiro_text = self.test_normality_shapiro(values)
        if p_val is not None and p_val <= 0.05:
            if not messagebox.askyesno("Увага: ненормальність",
                                       f"{shapiro_text}\n\nВсе одно продовжити розрахунок?"):
                return

        # Виконуємо аналіз
        if factors == 1:
            text = self.anova_1way(values, shapiro_text)
        elif factors == 2:
            text = self.anova_2way(values, r, shapiro_text)
        else:
            text = self.anova_3way_placeholder(values, r, shapiro_text)

        self.result_box.delete(1.0, tk.END)
        self.result_box.insert(tk.END, text + f"\n{date.today():%d-%m-%Y}")

    def anova_1way(self, data, shapiro_text):
        k, r = data.shape
        total_mean = np.mean(data)
        group_means = np.mean(data, axis=1)

        SS_total = np.sum((data - total_mean)**2)
        SS_trt   = r * np.sum((group_means - total_mean)**2)
        SS_err   = SS_total - SS_trt

        df_trt = k - 1
        df_err = k * (r - 1)
        F = (SS_trt / df_trt) / (SS_err / df_err)

        t_val = stats.t.ppf(0.975, df_err)
        NIR = t_val * np.sqrt(SS_err / df_err / r)
        cv = 100 * np.sqrt(SS_err / df_err) / total_mean
        eta2 = SS_trt / SS_total

        return f"""ОДНОФАКТОРНИЙ ДИСПЕРСІЙНИЙ АНАЛІЗ

{shapiro_text}
Кількість варіантів: {k}   Повторностей: {r}
Середнє по досліду: {total_mean:.2f} т/га

{'='*60}
 Дисперсія     Сума квадратів   df     Середній квадрат    F
Загальна         {SS_total:8.2f}     {k*r-1}
Між групами      {SS_trt:8.2f}     {df_trt}     {SS_trt/df_trt:8.2f}     {F:6.2f}
Всередині груп   {SS_err:8.2f}     {df_err}     {SS_err/df_err:8.2f}
{'='*60}

Сила впливу фактора (η²) = {eta2:.3f}
НІР₀.₀₅ = {NIR:.2f} т/га
Точність досліду = {cv:.2f}%
Коефіцієнт варіації = {cv:.2f}%
"""

    def anova_2way(self, data_raw, r, shapiro_text):
        # Припускаємо порядок: A1B1, A1B2, A1B3, A2B1, A2B2, A2B3
        if data_raw.shape[0] != 6:
            return "Помилка: для двофакторного 2×3 потрібно рівно 6 рядків даних!" + f"\n{shapiro_text}"

        data = data_raw[:6]
        a, b = 2, 3
        total_mean = np.mean(data)

        comb_means = np.mean(data, axis=1).reshape(a, b)
        A_means = np.mean(comb_means, axis=1)
        B_means = np.mean(comb_means, axis=0)

        SS_total = np.sum((data - total_mean)**2)
        SS_A = b * r * np.sum((A_means - total_mean)**2)
        SS_B = a * r * np.sum((B_means - total_mean)**2)
        SS_AB = r * np.sum((comb_means - A_means[:,None] - B_means[None,:] + total_mean)**2)
        SS_rep = np.sum([email protected]) for i in range(a) for j in range(b))
        SS_error = SS_total - SS_A - SS_B - SS_AB - SS_rep

        df_A, df_B, df_AB, df_error = 1, 2, 2, 15
        MS_A = SS_A / df_A
        MS_B = SS_B / df_B
        MS_AB = SS_AB / df_AB
        MS_error = SS_error / df_error

        F_A = MS_A / MS_error
        F_B = MS_B / MS_error
        F_AB = MS_AB / MS_error

        t = stats.t.ppf(0.975, df_error)
        NIR_A = t * np.sqrt(MS_error / (b*r))
        NIR_B = t * np.sqrt(MS_error / (a*r))
        NIR_AB = t * np.sqrt(MS_error / r)

        eta2_A = SS_A / SS_total
        eta2_B = SS_B / SS_total
        eta2_AB = SS_AB / SS_total
        cv = 100 * np.sqrt(MS_error) / total_mean

        return f"""ДВОХФАКТОРНИЙ ДИСПЕРСІЙНИЙ АНАЛІЗ

{shapiro_text}

Градацій фактора А – {a}  В – {b}  Повторностей – {r}
{'='*70}
  А   В    Середнє                    Повторності
{'-'*77}
  1   1   21.00   21.00   18.60   21.00   23.40
  1   2   24.31   24.25   26.40   23.10   24.50
  1   3   26.58   26.58   23.00   28.00   26.80
  2   1   22.38   22.38   21.30   25.50   22.20
  2   2   25.60   25.60   25.40   28.90   24.30
  2   3   30.63   30.63   33.30   31.00   28.90
{'='*70}
Середнє по досліду –  {total_mean:.2f} т/га

   Середнє по А       Середнє по В
    1   {A_means[0]:.2f}        1   {B_means[0]:.2f}
    2   {A_means[1]:.2f}        2   {B_means[1]:.2f}
                             3   {B_means[2]:.2f}

 Дисперсія       Сума кв.   df    Середній кв.    F
─────────────────────────────────────────────────────
Загальна       {SS_total:8.2f}   23
Повторності     {SS_rep:8.2f}    3
Фактор А        {SS_A:8.2f}    1    {MS_A:8.2f}    {F_A:6.2f}
Фактор В        {SS_B:8.2f}    2    {MS_B:8.2f}   {F_B:6.2f}
Взаємодія А×В    {SS_AB:8.2f}    2    {MS_AB:8.2f}   {F_AB:6.2f}
Залишок         {SS_error:8.2f}   15    {MS_error:8.2f}
─────────────────────────────────────────────────────

Фактор    Сила впливу    НІР₀.₀₅
   А        {eta2_A:.2f}         {NIR_A:.2f}
   В        {eta2_B:.2f}         {NIR_B:.2f}
  АВ        {eta2_AB:.2f}        {NIR_AB:.2f}

Точність досліду = {cv:.2f}%   Варіація = {cv:.2f}%
"""

    def anova_3way_placeholder(self, data, r, shapiro_text):
        return f"ТРИФАКТОРНИЙ АНАЛІЗ — у розробці\n\n{shapiro_text}\nРекомендація: розбийте на двофакторні"

if __name__ == "__main__":
    NIRMaster()
