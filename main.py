# -*- coding: utf-8 -*-
"""
SAD — Статистичний Аналіз Даних v1.0 GOLD
Класичний український калькулятор польового досліду + перевірка нормальності
Автор: Чаплоуцький Андрій Миколайович, Уманський НУС, 2025
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext, simpledialog
import pandas as pd
import numpy as np
from scipy.stats import f, t, shapiro
import os

class SADApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SAD v1.0 GOLD — Калькулятор польового досліду")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f5f5f5")

        tk.Label(self.root, text="SAD v1.0 GOLD", font=("Arial", 38, "bold"), fg="#1a3c6e", bg="#f5f5f5").pack(pady=20)
        tk.Label(self.root, text="Класичний дисперсійний аналіз + перевірка нормальності (Shapiro-Wilk)", 
                 font=("Arial", 14), bg="#f5f5f5").pack(pady=5)

        tk.Button(self.root, text="РОЗПОЧАТИ АНАЛІЗ", font=("Arial", 18, "bold"), bg="#d32f2f", fg="white",
                  width=35, height=2, command=self.start).pack(pady=40)

        tk.Label(self.root, text="Автор: Чаплоуцький А.М. • Уманський національний університет садівництва • 2025",
                 fg="gray", bg="#f5f5f5").pack(pady=20)
        self.root.mainloop()

    def start(self):
        self.win = tk.Toplevel(self.root)
        self.win.title("SAD v1.0 GOLD — Ввід даних")
        self.win.geometry("1650x1000")

        tools = tk.Frame(self.win)
        tools.pack(fill="x", padx=15, pady=10)
        tk.Button(tools, text="З Excel", command=self.from_excel).pack(side="left", padx=5)
        tk.Button(tools, text="Очистити", bg="#e57373", fg="white", command=self.clear).pack(side="left", padx=5)
        tk.Button(tools, text="АНАЛІЗ", bg="#d32f2f", fg="white", font=("Arial", 18, "bold"), width=20,
                  command=self.analyze).pack(side="right", padx=20)

        frame = tk.Frame(self.win)
        frame.pack(fill="both", expand=True, padx=15, pady=5)
        self.tree = ttk.Treeview(frame, columns=[f"c{i}" for i in range(20)], show="headings")
        for i in range(20):
            self.tree.heading(f"c{i}", text=str(i+1))
            self.tree.column(f"c{i}", width=100, anchor="center")
        for _ in range(35):
            self.tree.insert("", "end", values=[""]*20)
        self.tree.pack(side="left", fill="both", expand=True)

        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(self.win, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")

        res = tk.LabelFrame(self.win, text=" РЕЗУЛЬТАТИ АНАЛІЗУ ", font=("Arial", 13, "bold"))
        res.pack(fill="both", expand=True, padx=15, pady=10)
        self.result = scrolledtext.ScrolledText(res, font=("Consolas", 11), bg="#fffdf0")
        self.result.pack(fill="both", expand=True)

        self.win.bind_all("<Control-v>", lambda e: self.paste())

    def paste(self):
        try:
            df = pd.read_clipboard(sep=r"\s+", header=None, dtype=str, engine='python')
            if df.empty: return
            self.clear()
            for _, row in df.iterrows():
                vals = row.tolist() + [""]*(20 - len(row))
                self.tree.insert("", "end", values=vals[:20])
            messagebox.showinfo("Готово", f"Вставлено {len(df)} рядків")
        except: pass

    def from_excel(self):
        path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        if path:
            df = pd.read_excel(path, header=None).astype(str)
            self.clear()
            for _, row in df.iterrows():
                vals = row.tolist() + [""]*(20 - len(row))
                self.tree.insert("", "end", values=vals[:20])

    def clear(self):
        for i in self.tree.get_children(): self.tree.delete(i)
        for _ in range(35): self.tree.insert("", "end", values=[""]*20)

    def get_data(self):
        data = []
        for child in self.tree.get_children():
            row = [x.strip() for x in self.tree.item(child)["values"] if x.strip()]
            if len(row) >= 3: data.append(row)
        return pd.DataFrame(data)

    def analyze(self):
        try:
            df = self.get_data()
            if df.empty:
                messagebox.showerror("Помилка", "Немає даних")
                return

            n_factors = simpledialog.askinteger("Кількість факторів", 
                "Скільки стовпців — фактори? (1–3)", minvalue=1, maxvalue=3)
            if not n_factors: return

            factors = df.iloc[:, :n_factors]
            repeats = df.iloc[:, n_factors:].apply(pd.to_numeric, errors='coerce').dropna(how='all', axis=1)
            if repeats.empty or repeats.shape[1] < 2:
                messagebox.showerror("Помилка", "Мінімум 2 повторності!")
                return

            n_rep = repeats.shape[1]
            values = repeats.stack().values
            n_total = len(values)
            grand_mean = values.mean()

            # Залишки для Shapiro-Wilk
            row_means = repeats.mean(axis=1).values
            residuals = []
            for i, row in repeats.iterrows():
                residuals.extend(row.values - row_means[i])
            residuals = np.array(residuals)

            # Shapiro-Wilk
            if len(residuals) >= 8:
                shapiro_stat, shapiro_p = shapiro(residuals)
                normality = "нормальний" if shapiro_p > 0.05 else "НЕ нормальний"
            else:
                shapiro_stat, shapiro_p = None, None
                normality = "недостатньо даних (n<8)"

            # Дисперсійний аналіз
            group_means = repeats.mean(axis=1)
            total_ss = ((values - grand_mean)**2).sum()
            factor_ss = n_rep * ((group_means - grand_mean)**2).sum()
            error_ss = total_ss - factor_ss

            df_factor = len(group_means) - 1
            df_error = n_total - len(group_means)
            ms_factor = factor_ss / df_factor if df_factor > 0 else 0
            ms_error = error_ss / df_error if df_error > 0 else 0
            F_calc = ms_factor / ms_error if ms_error > 0 else 0
            F_crit = f.ppf(0.95, df_factor, df_error) if df_factor > 0 and df_error > 0 else 0
            p_value = 1 - f.cdf(F_calc, df_factor, df_error) if F_calc > 0 else 1

            # НІР₀.₅
            t_crit = t.ppf(0.975, df_error)
            LSD = t_crit * np.sqrt(2 * ms_error / n_rep)

            # Вилучення впливу
            eta_sq = factor_ss / total_ss * 100 if total_ss > 0 else 0

            # Букви істотності
            means_sorted = group_means.sort_values(ascending=False)
            letters = []
            current_letter = 'a'
            prev_mean = means_sorted.iloc[0] + LSD + 1
            for idx, mean in means_sorted.items():
                variant = " | ".join(str(x) for x in factors.iloc[idx])
                if mean < prev_mean - LSD:
                    current_letter = chr(ord(current_letter) + 1)
                letters.append(f"{variant:25} → {mean:.3f} {current_letter}")
                prev_mean = mean

            # Звіт
            report = [
                "═" * 70,
                "           РЕЗУЛЬТАТИ ДИСПЕРСІЙНОГО АНАЛІЗУ (класичний метод)",
                "═" * 70,
                f"Кількість варіантів: {len(group_means)}     Повторностей: {n_rep}",
                f"Загальна кількість спостережень: {n_total}",
                f"Середнє по досліду: {grand_mean:.3f}",
                "",
                "Перевірка нормальності розподілу залишків (Shapiro-Wilk):",
                f"   W = {shapiro_stat:.4f}, p = {shapiro_p:.4f} → {normality.upper()}" if shapiro_p is not None else "   Недостатньо даних для тесту",
                "",
                "Джерело варіації     СК         df     СКср       Fрозр    F05",
                f"Факторний вплив      {factor_ss:8.2f}  {df_factor:3}  {ms_factor:8.4f}  {F_calc:6.3f}  {F_crit:5.3f}",
                f"Випадкова            {error_ss:8.2f}  {df_error:3}  {ms_error:8.4f}",
                f"Загальна             {total_ss:8.2f}  {n_total-1:3}",
                "",
                f"Висновок: вплив фактора — {'ІСТОТНИЙ' if F_calc > F_crit else 'НЕІСТОТНИЙ'} (p ≈ {p_value:.4f})",
                f"Вилучення впливу фактора: {eta_sq:.2f}%",
                f"НІР₀.₅ = {LSD:.4f}",
                "",
                "СЕРЕДНІ ПО ВАРІАНТАХ (з буквами істотності):",
            ] + letters

            self.result.delete(1.0, tk.END)
            self.result.insert(tk.END, "\n".join(report))
            self.result.clipboard_clear()
            self.result.clipboard_append("\n".join(report))

            messagebox.showinfo("ГОТОВО!", 
                f"Аналіз завершено!\nНормальність: {normality.upper()}\nНІР₀.₅ = {LSD:.4f}\nЗвіт скопійовано в буфер!")

        except Exception as e:
            messagebox.showerror("Помилка", f"Сталася помилка:\n{e}")

if __name__ == "__main__":
    SADApp()
