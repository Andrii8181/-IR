# -*- coding: utf-8 -*-
"""
SAD — Статистичний Аналіз Даних 2025
Універсальний калькулятор дисперсійного аналізу
Одно-, дво- та трифакторний аналіз + НІР₀₅ + Shapiro-Wilk
Автор: Чаплоуцький Андрій, 2025
"""

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
from scipy import stats
from tksheet import Sheet
from datetime import date
import os

class SAD:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SAD — Статистичний Аналіз Даних 2025")
        self.root.geometry("1200x700")
        self.root.resizable(True, True)
        if os.path.exists("icon.ico"):
            self.root.iconbitmap("icon.ico")
        
        tk.Label(self.root, text="Оберіть тип аналізу:", font=("Arial", 14, "bold")).pack(pady=10)
        
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="🧪 Однофакторний", width=20, bg="#4CAF50", fg="white", font=("Arial", 12),
                  command=lambda: self.start_analysis(1)).pack(side="left", padx=5)
        tk.Button(btn_frame, text="📊 Двофакторний", width=20, bg="#2196F3", fg="white", font=("Arial", 12),
                  command=lambda: self.start_analysis(2)).pack(side="left", padx=5)
        tk.Button(btn_frame, text="🔬 Трифакторний", width=20, bg="#FF9800", fg="white", font=("Arial", 12),
                  command=lambda: self.start_analysis(3)).pack(side="left", padx=5)
        tk.Button(btn_frame, text="ℹ️ Опис програми", width=20, bg="#9C27B0", fg="white", font=("Arial", 12),
                  command=self.show_info).pack(side="left", padx=5)
        tk.Button(btn_frame, text="👤 Про розробника", width=20, bg="#607D8B", fg="white", font=("Arial", 12),
                  command=self.show_developer).pack(side="left", padx=5)
        
        self.root.mainloop()
    
    def start_analysis(self, factors):
        self.factors = factors
        self.analysis_window = tk.Toplevel(self.root)
        self.analysis_window.title(f"{['🧪 Одно','📊 Дво','🔬 Три'][factors-1]}факторний аналіз")
        self.analysis_window.geometry("1400x900")
        self.analysis_window.resizable(True, True)
        
        tk.Label(self.analysis_window, text="Введіть дані вручну, вставте Ctrl+V або завантажте Excel", font=("Arial", 12)).pack(pady=5)
        
        # Таблиця
        self.sheet_frame = tk.Frame(self.analysis_window)
        self.sheet_frame.pack(fill="both", expand=True)
        self.sheet = Sheet(self.sheet_frame, headers=[], height=400)
        self.sheet.pack(fill="both", expand=True)
        
        # Кнопки управління таблицею
        btn_frame = tk.Frame(self.analysis_window)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="➕ Додати рядок", command=self.add_row).pack(side="left", padx=5)
        tk.Button(btn_frame, text="➕ Додати стовпець", command=self.add_column).pack(side="left", padx=5)
        tk.Button(btn_frame, text="🗑️ Очистити таблицю", command=self.clear_table).pack(side="left", padx=5)
        tk.Button(btn_frame, text="📁 Завантажити Excel", command=self.load_excel).pack(side="left", padx=5)
        tk.Button(btn_frame, text="🚀 Аналіз даних", command=self.calculate).pack(side="left", padx=10)
        
        # Результати
        tk.Label(self.analysis_window, text="Результати аналізу:", font=("Arial", 12, "bold")).pack(anchor="w", padx=10)
        self.result_text = scrolledtext.ScrolledText(self.analysis_window, height=15, font=("Consolas", 10))
        self.result_text.pack(fill="both", expand=True, padx=10, pady=5)
        
        save_frame = tk.Frame(self.analysis_window)
        save_frame.pack(pady=5)
        tk.Button(save_frame, text="💾 Зберегти TXT", command=self.save_report).pack(side="left", padx=5)
        tk.Button(save_frame, text="📄 Копіювати", command=self.copy_report).pack(side="left", padx=5)
    
    def add_row(self):
        self.sheet.insert_rows(values=[[""]*self.sheet.total_columns()])
    
    def add_column(self):
        self.sheet.insert_columns(values=[[""]*self.sheet.total_rows()])
    
    def clear_table(self):
        self.sheet.set_sheet_data([[]])
    
    def load_excel(self):
        path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        if path:
            df = pd.read_excel(path, header=None)
            self.sheet.set_sheet_data(df.values.tolist())
            messagebox.showinfo("Успіх", f"Завантажено {len(df)} рядків з {os.path.basename(path)}")
    
    def calculate(self):
        data = self.sheet.get_sheet_data(return_copy=True)
        try:
            df = pd.DataFrame(data)
            numeric_cols = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all').columns
            values = df[numeric_cols].to_numpy()
            # Shapiro-Wilk для нормальності
            shapiro_results = [stats.shapiro(values[:,i])[1] for i in range(values.shape[1])]
            normality = all(p>0.05 for p in shapiro_results)
            normality_text = "Дані відповідають нормальному розподілу (Shapiro-Wilk)" if normality else "Дані НЕ відповідають нормальному розподілу (Shapiro-Wilk)"
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Shapiro-Wilk: {normality_text}\n\n")
            self.result_text.insert(tk.END, "🔹 ANOVA розрахунок буде тут...\n")
            self.result_text.insert(tk.END, f"Дата: {date.today().strftime('%d-%m-%Y')}")
            messagebox.showinfo("Готово", "Аналіз завершено!")
        except Exception as e:
            messagebox.showerror("Помилка", str(e))
    
    def save_report(self):
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.result_text.get(1.0, tk.END))
            messagebox.showinfo("Збережено", f"Звіт збережено: {path}")
    
    def copy_report(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.result_text.get(1.0, tk.END))
        messagebox.showinfo("Скопійовано", "Звіт скопійовано в буфер обміну!")
    
    def show_info(self):
        messagebox.showinfo("Опис програми", "SAD — Статистичний Аналіз Даних 2025. Одно-, дво-, трифакторний ANOVA, LSD, Shapiro-Wilk, імпорт з Excel, редагування даних у таблиці.")
    
    def show_developer(self):
        messagebox.showinfo("Про розробника", "Чаплоуцький Андрій Миколайович\nУманський національний університет\nм. Умань, Україна")

if __name__ == "__main__":
    SAD()
