# -*- coding: utf-8 -*-
"""
SAD-Статистичний Аналіз Даних
Універсальний калькулятор дисперсійного аналізу
Автор: Чаплоуцький Андрій Миколайович, Уманський національний університет
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import pandas as pd
import numpy as np
from scipy import stats
import os

class SAD:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SAD-Статистичний Аналіз Даних")
        self.root.geometry("1200x700")
        self.root.resizable(True, True)

        if os.path.exists("icon.ico"):
            self.root.iconbitmap("icon.ico")

        # Меню вибору аналізу
        tk.Label(self.root, text="Оберіть тип аналізу:", font=("Arial", 14, "bold")).pack(pady=20)
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="🧪 Однофакторний", width=25, height=2, bg="#4CAF50", fg="white",
                  font=("Arial", 12), command=lambda: self.start_analysis(1)).grid(row=0, column=0, padx=10)
        tk.Button(btn_frame, text="📊 Двофакторний", width=25, height=2, bg="#2196F3", fg="white",
                  font=("Arial", 12), command=lambda: self.start_analysis(2)).grid(row=0, column=1, padx=10)
        tk.Button(btn_frame, text="🔬 Трифакторний", width=25, height=2, bg="#FF9800", fg="white",
                  font=("Arial", 12), command=lambda: self.start_analysis(3)).grid(row=0, column=2, padx=10)

        # Кнопки "Про програму" та "Про розробника"
        info_frame = tk.Frame(self.root)
        info_frame.pack(pady=10)
        tk.Button(info_frame, text="ℹ️ Про програму", command=self.show_about, bg="#607D8B", fg="white").pack(side="left", padx=5)
        tk.Button(info_frame, text="👤 Про розробника", command=self.show_author, bg="#795548", fg="white").pack(side="left", padx=5)

        tk.Label(self.root, text="💡 Після вибору: вставте дані з Excel (Ctrl+V), додавайте рядки/стовпці, або імпортуйте з Excel.",
                 font=("Arial", 10), fg="gray").pack(pady=10)

        self.root.mainloop()

    def show_about(self):
        messagebox.showinfo("Про програму", "SAD-Статистичний Аналіз Даних — універсальний калькулятор одно-, дво- та трифакторного дисперсійного аналізу з перевіркою нормальності (Shapiro-Wilk).")

    def show_author(self):
        messagebox.showinfo("Про розробника", "Чаплоуцький Андрій Миколайович\nУманський національний університет\nм. Умань, Україна")

    def start_analysis(self, factor_count):
        self.factor_count = factor_count
        self.analysis_window = tk.Toplevel(self.root)
        self.analysis_window.title(f"Аналіз: {factor_count}-факторний")
        self.analysis_window.geometry("1100x600")

        # Додаткова панель кнопок
        toolbar = tk.Frame(self.analysis_window)
        toolbar.pack(side="top", fill="x", pady=5)
        tk.Button(toolbar, text="Додати рядок", command=self.add_row).pack(side="left", padx=5)
        tk.Button(toolbar, text="Додати стовпець", command=self.add_column).pack(side="left", padx=5)
        tk.Button(toolbar, text="Імпорт з Excel", command=self.load_excel).pack(side="left", padx=5)
        tk.Button(toolbar, text="Обчислити", command=self.calculate).pack(side="left", padx=5)

        # Таблиця для даних
        self.data_table = ttk.Treeview(self.analysis_window)
        self.data_table.pack(expand=True, fill="both")
        self.data_table["columns"] = ["A", "B"]
        self.data_table["show"] = "headings"
        for col in self.data_table["columns"]:
            self.data_table.heading(col, text=col)
            self.data_table.column(col, width=100)

        # Текстове поле для результатів
        self.result_text = scrolledtext.ScrolledText(self.analysis_window, height=10)
        self.result_text.pack(fill="x", pady=5)

    def add_row(self):
        if self.data_table["columns"]:
            values = [""] * len(self.data_table["columns"])
            self.data_table.insert("", "end", values=values)

    def add_column(self):
        col_count = len(self.data_table["columns"])
        new_col = chr(65 + col_count)
        self.data_table["columns"] = list(self.data_table["columns"]) + [new_col]
        self.data_table.heading(new_col, text=new_col)
        self.data_table.column(new_col, width=100)

    def load_excel(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        if file_path:
            df = pd.read_excel(file_path)
            self.data_table["columns"] = list(df.columns)
            for i in self.data_table.get_children():
                self.data_table.delete(i)
            for index, row in df.iterrows():
                self.data_table.insert("", "end", values=list(row))

    def get_table_data(self):
        cols = self.data_table["columns"]
        data = []
        for item in self.data_table.get_children():
            row = self.data_table.item(item)["values"]
            if len(row) < len(cols):
                row += [""] * (len(cols) - len(row))
            data.append(row)
        df = pd.DataFrame(data, columns=cols)
        return df.apply(pd.to_numeric, errors="coerce")

    def calculate(self):
        df = self.get_table_data()
        if df.empty:
            messagebox.showerror("Помилка", "Таблиця порожня")
            return
        self.result_text.delete(1.0, tk.END)
        # Перевірка нормальності (Shapiro-Wilk)
        self.result_text.insert(tk.END, "Перевірка нормальності (Shapiro-Wilk):\n")
        for col in df.columns:
            col_data = df[col].dropna()
            if len(col_data) >= 3:
                stat, p = stats.shapiro(col_data)
                self.result_text.insert(tk.END, f"{col}: W={stat:.4f}, p={p:.4f}\n")
            else:
                self.result_text.insert(tk.END, f"{col}: недостатньо даних\n")
        self.result_text.insert(tk.END, "\n")

        # Однофакторний аналіз
        if self.factor_count == 1:
            self.one_way_anova(df)
        elif self.factor_count == 2:
            self.two_way_anova(df)
        elif self.factor_count == 3:
            self.three_way_anova(df)

    def one_way_anova(self, df):
        groups = [df[col].dropna() for col in df.columns]
        f, p = stats.f_oneway(*groups)
        self.result_text.insert(tk.END, f"Однофакторний ANOVA: F={f:.4f}, p={p:.4f}\n")

    def two_way_anova(self, df):
        self.result_text.insert(tk.END, "Двофакторний аналіз поки що обмежений (приклад).\n")

    def three_way_anova(self, df):
        self.result_text.insert(tk.END, "Трифакторний аналіз поки що обмежений (приклад).\n")


if __name__ == "__main__":
    app = SAD()
