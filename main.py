import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

import numpy as np
from scipy import stats

class SADApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SAD - Статистичний аналіз даних")
        self.root.geometry("1000x600")

        # Початкова таблиця 10 рядків x 7 стовпчиків
        self.columns = ["Фактор А", "Фактор В", "Фактор С", "Повт.1", "Повт.2", "Повт.3", "Повт.4"]
        self.table = ttk.Treeview(root, columns=self.columns, show="headings", height=10)
        for col in self.columns:
            self.table.heading(col, text=col)
            self.table.column(col, width=120, anchor="center")
        self.table.pack(fill="both", expand=True, padx=10, pady=10)

        # Додавання пустих рядків
        for _ in range(10):
            self.table.insert("", "end", values=["" for _ in self.columns])

        # Кнопки
        frame = tk.Frame(root)
        frame.pack(pady=5)

        tk.Button(frame, text="Додати рядок", command=self.add_row).pack(side="left", padx=5)
        tk.Button(frame, text="Видалити рядок", command=self.delete_row).pack(side="left", padx=5)
        tk.Button(frame, text="Перевірка нормальності (Шапіро-Вілк)", command=self.run_normality_test).pack(side="left", padx=5)
        tk.Button(frame, text="Розрахунок ANOVA", command=self.run_anova).pack(side="left", padx=5)
        tk.Button(frame, text="Про програму", command=self.show_about).pack(side="left", padx=5)

    def add_row(self):
        self.table.insert("", "end", values=["" for _ in self.columns])

    def delete_row(self):
        selected = self.table.selection()
        for item in selected:
            self.table.delete(item)

    def get_data_matrix(self):
        data_matrix = []
        for row_id in self.table.get_children():
            row_values = self.table.item(row_id)['values']
            try:
                row_numbers = [float(x) if x != "" else np.nan for x in row_values[3:]]  # Повторення
                data_matrix.append(row_numbers)
            except ValueError:
                continue
        return np.array(data_matrix)

    def run_normality_test(self):
        data = self.get_data_matrix().flatten()
        data = data[~np.isnan(data)]
        if len(data) == 0:
            messagebox.showwarning("Помилка", "Дані відсутні або некоректні")
            return
        W, p = stats.shapiro(data)
        messagebox.showinfo("Результат перевірки",
                            f"Статистика W = {W:.4f}, p = {p:.4f}\n"
                            f"{'нормальний' if p>0.05 else 'не нормальний'} розподіл")

    def run_anova(self):
        data_matrix = self.get_data_matrix()
        if data_matrix.size == 0:
            messagebox.showwarning("Помилка", "Дані відсутні або некоректні")
            return

        # Проста однофакторна ANOVA по рядках для демонстрації
        try:
            # Кожен рядок = повторення
            f_val, p_val = stats.f_oneway(*[row[~np.isnan(row)] for row in data_matrix])
            report = f"Однофакторна ANOVA\nF-розрахункове = {f_val:.4f}, p = {p_val:.4f}\n"
            report += f"{'Істотний ефект' if p_val<0.05 else 'Неістотний ефект'}"
        except Exception as e:
            report = f"Помилка при розрахунку ANOVA: {e}"

        # Відкриваємо нове вікно з результатами
        result_win = tk.Toplevel(self.root)
        result_win.title("Результати аналізу")
        text = tk.Text(result_win, wrap="none", width=120, height=30)
        text.pack(fill="both", expand=True)
        text.insert("1.0", report)
        text.config(state="normal")  # Щоб користувач міг копіювати

    def show_about(self):
        messagebox.showinfo("Про програму",
                            "SAD - Статистичний аналіз даних\n"
                            "Розробник: ....\n"
                            "Кафедра плодівництва і виноградарства УНУ")


if __name__ == "__main__":
    root = tk.Tk()
    app = SADApp(root)
    root.mainloop()
