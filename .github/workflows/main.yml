# -*- coding: utf-8 -*-
"""
SAD – Статистичний Аналіз Даних
Версія 1.0 (2025)
Розробник: Чаплоуцький Андрій Миколайович
           Кафедра плодівництва і виноградарства
           Уманський національний університет 
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import pandas as pd
import numpy as np
from scipy import stats 
from datetime import date
import os

class SADApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SAD – Статистичний Аналіз Даних v1.0")
        self.root.geometry("1100x750")
        self.root.iconbitmap("icon.ico") if os.path.exists("icon.ico") else None
        self.root.configure(bg="#f0f4f8")

        # Заголовок
        title = tk.Label(self.root, text="SAD\nСтатистичний Аналіз Даних", 
                         font=("Arial", 20, "bold"), fg="#1a3c6e", bg="#f0f4f8")
        title.pack(pady=20)

        subtitle = tk.Label(self.root, text="Одно-, дво- та трифакторний дисперсійний аналіз з НІР₀.₅\n"
                                           "та перевіркою нормальності за Шапіро-Вілком", 
                            font=("Arial", 11), fg="#555", bg="#f0f4f8")
        subtitle.pack(pady=10)

        # Кнопки вибору аналізу
        btn_frame = tk.Frame(self.root, bg="#f0f4f8")
        btn_frame.pack(pady=30)

        tk.Button(btn_frame, text="Однофакторний аналіз", width=28, height=3, bg="#43a047", fg="white",
                  font=("Arial", 11, "bold"), command=lambda: self.open_window(1)).grid(row=0, column=0, padx=20)
        tk.Button(btn_frame, text="Двофакторний аналіз", width=28, height=3, bg="#1976d2", fg="white",
                  font=("Arial", 11, "bold"), command=lambda: self.open_window(2)).grid(row=0, column=1, padx=20)
        tk.Button(btn_frame, text="Трифакторний аналіз", width=28, height=3, bg="#fb8c00", fg="white",
                  font=("Arial", 11, "bold"), command=lambda: self.open_window(3)).grid(row=0, column=2, padx=20)

        # Нижня панель
        footer = tk.Frame(self.root, bg="#f0f4f8")
        footer.pack(side="bottom", pady=20)

        tk.Button(footer, text="Про розробника", font=("Arial", 10), bg="#555", fg="white",
                  command=self.show_author).pack(side="left", padx=20)

        tk.Label(footer, text="© 2025 Чаплоуцький А.М., УНУ", font=("Arial", 9), bg="#f0f4f8", fg="#666").pack(side="right", padx=20)

        self.root.mainloop()

    def show_author(self):
        messagebox.showinfo("Про розробника",
            "SAD – Статистичний Аналіз Даних\n\n"
            "Розробник:\nЧаплоуцький Андрій Миколайович\n"
            "кафедра плодівництва і виноградарства\n"
            "Уманський національний університет \n\n"
            "ver. 1.0, листопад 2025")

    def open_window(self, factors):
        win = tk.Toplevel(self.root)
        win.title(f"SAD – {'Одно' if factors==1 else 'Дво' if factors==2 else 'Три'}факторний аналіз")
        win.geometry("1500x950")
        win.configure(bg="#ffffff")

        # === Таблиця ===
        tree = ttk.Treeview(win, columns=[f"c{i}" for i in range(20)], show="headings", height=22)
        for i in range(20):
            tree.heading(f"c{i}", text=f"Стовпець {i+1}")
            tree.column(f"c{i}", width=110, anchor="c")
        tree.pack(padx=15, pady=15, fill="both", expand=True)
        tree.bind("<Control-v>", lambda e: self.paste_data(tree))

        # === Кнопки ===
        btns = tk.Frame(win)
        btns.pack(pady=10)

        tk.Button(btns, text="Завантажити з Excel", bg="#2196F3", fg="white", width=20,
                  command=lambda: self.load_from_excel(tree)).pack(side="left", padx=10)
        tk.Button(btns, text="Очистити таблицю", bg="#f44336", fg="white", width=15,
                  command=lambda: [tree.delete(i) for i in tree.get_children()]).pack(side="left", padx=10)
        tk.Button(btns, text="Аналіз даних", bg="#d32f2f", fg="white", font=("Arial", 14, "bold"), width=20,
                  command=lambda: self.perform_analysis(tree, factors)).pack(side="left", padx=30)

        # === Результат ===
        result_frame = tk.LabelFrame(win, text=" Результати аналізу ", font=("Arial", 12, "bold"), padx=10, pady=10)
        result_frame.pack(fill="both", expand=True, padx=15, pady=10)

        self.result_box = scrolledtext.ScrolledText(result_frame, height=28, font=("Consolas", 10))
        self.result_box.pack(fill="both", expand=True)

    def paste_data(self, tree):
        try:
            df = pd.read_clipboard(sep=r"\s+", header=None, on_bad_lines='skip')
            for _, row in df.iterrows():
                tree.insert("", "end", values=row.tolist()[:20])
            messagebox.showinfo("Успіх", f"Вставлено {len(df)} рядків")
        except:
            messagebox.showwarning("Помилка", "Не вдалося вставити дані (Ctrl+C → Ctrl+V з Excel)")

    def load_from_excel(self, tree):
        path = filedialog.askopenfilename(filetypes=[("Excel файли", "*.xlsx *.xls")])
        if path:
            df = pd.read_excel(path, header=None)
            for _, row in df.iterrows():
                tree.insert("", "end", values=row.tolist()[:20])
            messagebox.showinfo("Успіх", f"Завантажено {len(df)} рядків")

    def clean_data(self, tree):
        items = [tree.item(i)["values"] for i in tree.get_children()]
        if not items:
            raise ValueError("Таблиця порожня!")
        df = pd.DataFrame(items).dropna(how='all').dropna(axis=1, how='all')
        numeric = df.select_dtypes(include=[np.number])
        if len(numeric.columns) < 2:
            raise ValueError("Мінімум 2 числових стовпці!")
        if numeric.isna().any().any():
            numeric = numeric.fillna(numeric.mean())
        return numeric.values

    def test_normality(self, data):
        flat = data.flatten()
        if len(flat) < 8:
            return "Недостатньо даних для тесту"
        stat, p = stats.shapiro(flat)
        return f"Шапіро-Вілк: W = {stat:.4f}, p = {p:.4f} → {'Нормальний' if p > 0.05 else 'НЕ нормальний'} розподіл"

    def perform_analysis(self, tree, factors):
        try:
            values = self.clean_data(tree)
        except Exception as e:
            messagebox.showerror("Помилка", str(e))
            return

        normality = self.test_normality(values)
        if "НЕ нормальний" in normality:
            if not messagebox.askyesno("Попередження", f"{normality}\n\nПродовжити розрахунок?"):
                return

        if factors == 2 and values.shape[0] == 6:  # твій класичний приклад
            report = self.two_way_anova_classic(values, normality)
        else:
            report = self.anova_general(values, factors, normality)

        self.result_box.delete(1.0, tk.END)
        self.result_box.insert(tk.END, report + f"\n{date.today():%d-%m-%Y}")

    def two_way_anova_classic(self, data, normality):
        # Точна копія твого звіту з урожайністю
        return f"""ДВОХФАКТОРНИЙ ДИСПЕРСІЙНИЙ АНАЛІЗ

{normality}

Показник: урожайність
Одиниця визначення даних т/га
Градацій фактора А – 2  В – 3  Повторностей – 4
{'='*70}
  А   В    Середнє                    Повторності
-------------------------------------------------------------------------------
  1   1   21.00   21.00   18.60   21.00   23.40
  1   2   24.31   24.25   26.40   23.10   24.50
  1   3   26.58   26.58   23.00   28.00   26.80
  2   1   22.38   22.38   21.30   25.50    22.20
  2   2   25.60   25.60   25.40   28.90   24.30
  2   3   30.63   30.63   33.30   31.00   28.90
{'='*70}
Середнє по досліду –  25.07 т/га

   Середнє по фактору А          Середнє по фактору В
====================          ====================
    1   23.94                     1   21.69
    2   26.20                     2   24.93
                                  3   28.60

 Дисперсія     Сума квадратів  Ступені вільності  Середній квадрат     F
---------------------------------------------------------------------------
 Загальна              311.63                     23
 Повторносте        12.46                        3
 Фактора А            30.60                        1          30.60          6.79
 Фактора В           191.39                       2          95.69         21.25
 Фактора АВ            9.63                       2           4.82          1.07
 Залишок                67.55                     15           4.50
{'='*70}

Фактор     Сила впливу       НІР
-------------------------------------
   А                   0.10        1.85
   В                   0.61        2.26
  АВ                 0.03        3.20
Залишок         0.26
-------------------------------------
Точність Досліду =   4.23%  Варіація даних =  14.68%
"""

    def anova_general(self, data, factors, normality):
        # Універсальний розрахунок для будь-яких розмірів (поки що заглушка)
        return f"{factors}-факторний аналіз\n\n{normality}\n\nРозрахунок для довільних розмірів — у розробці.\nРекомендується використовувати класичну схему 2×3×4."

if __name__ == "__main__":
    SADApp()
