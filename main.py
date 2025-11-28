import tkinter as tk
from tkinter import simpledialog, messagebox, scrolledtext

import numpy as np
from scipy.stats import shapiro, f

class SADApp:
    def __init__(self, root):
        self.root = root
        root.title("SAD - Статистичний аналіз даних")
        root.geometry("1200x600")

        # Таблиця 10x7
        self.rows = 10
        self.cols = 7
        self.col_names = ["Фактор А", "Фактор В", "Фактор С",
                          "Повт.1", "Повт.2", "Повт.3", "Повт.4"]

        self.table = []
        for r in range(self.rows):
            row_entries = []
            for c in range(self.cols):
                e = tk.Entry(root, width=12)
                e.grid(row=r, column=c, padx=2, pady=2)
                row_entries.append(e)
            self.table.append(row_entries)
        for c, name in enumerate(self.col_names):
            lbl = tk.Label(root, text=name, font=("Arial", 10, "bold"))
            lbl.grid(row=0, column=c)

        # Кнопка запуску аналізу
        self.analyze_btn = tk.Button(root, text="Аналіз", command=self.run_analysis)
        self.analyze_btn.grid(row=self.rows+1, column=0, columnspan=self.cols, pady=10)

    def get_data(self):
        data = []
        for r in range(1, self.rows):
            row = []
            for c in range(self.cols):
                val = self.table[r][c].get()
                try:
                    row.append(float(val))
                except ValueError:
                    row.append(np.nan)
            data.append(row)
        return np.array(data)

    def run_analysis(self):
        data = self.get_data()
        if np.isnan(data).all():
            messagebox.showwarning("Помилка", "Дані відсутні або некоректні")
            return

        # Перевірка нормальності по всіх числових даних
        numeric_data = data[:, 3:].flatten()
        numeric_data = numeric_data[~np.isnan(numeric_data)]
        W, p = shapiro(numeric_data)

        # Створюємо текстовий вікно для виводу результату
        result_window = tk.Toplevel(self.root)
        result_window.title("Результати аналізу")
        text_area = scrolledtext.ScrolledText(result_window, width=120, height=30)
        text_area.pack(padx=10, pady=10)

        # Вивід результату
        text_area.insert(tk.END, "Р Е З У Л Ь Т А Т И   Т Р И Ф А К Т О Р Н О Г О   Д И С П Е Р С І Й Н О Г О   А Н А Л І З У\n\n")
        text_area.insert(tk.END, f"Перевірка нормальності залишків (Shapiro-Wilk): "
                                 f"{'нормальний' if p>0.05 else 'ненормальний'} (W={W:.4f}, p={p:.4f})\n\n")
        text_area.insert(tk.END, "Джерела варіації та аналіз F - ПОКАЗОВИЙ РЕЗУЛЬТАТ (тестові дані)\n")
        text_area.insert(tk.END, "Фактор А, Фактор В, Фактор С, Повт.1,... Повт.4\n")
        text_area.insert(tk.END, "...\n\n")
        text_area.insert(tk.END, "Вилучення впливу та НІР, середні по факторах...\n")
        text_area.insert(tk.END, "(Цей блок можна адаптувати для реальних обрахунків одно-, дво-, трифакторного досліду)\n")

        # Дозволяємо копіювати текст
        text_area.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = SADApp(root)
    root.mainloop()
