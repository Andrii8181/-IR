import tkinter as tk
from tkinter import messagebox, simpledialog
import numpy as np
from scipy.stats import shapiro

# --- Функції аналізу ---
def check_normality(data):
    W, p = shapiro(data)
    return {"W": W, "p": p}

def run_analysis():
    # Збираємо дані з таблиці
    data = []
    for i in range(rows):
        for j in range(cols):
            val = table_entries[i][j].get()
            if val.strip():
                try:
                    data.append(float(val))
                except ValueError:
                    pass

    if not data:
        messagebox.showwarning("Помилка", "Дані відсутні або некоректні")
        return

    result = check_normality(data)
    # Формуємо текст звіту
    report = f"Перевірка нормальності (Shapiro-Wilk):\nW = {result['W']:.4f}, p = {result['p']:.4f}\n\n"
    report += "Додатковий аналіз можна додати тут...\n\n"
    report += "Скопіюйте цей текст для Word або Блокноту."

    # Відкриваємо вікно з результатом
    report_window = tk.Toplevel(root)
    report_window.title("Результати аналізу")
    text_widget = tk.Text(report_window, wrap=tk.WORD)
    text_widget.insert(tk.END, report)
    text_widget.pack(expand=True, fill=tk.BOTH)
    text_widget.config(state=tk.NORMAL)  # дозволити копіювання
    tk.Button(report_window, text="Закрити", command=report_window.destroy).pack(pady=5)

# --- Ініціалізація вікна ---
root = tk.Tk()
root.title("SAD - Статистичний аналіз даних")
root.geometry("1000x600")

rows, cols = 10, 7
col_names = ["Фактор А", "Фактор В", "Фактор С", "Повт.1", "Повт.2", "Повт.3", "Повт.4"]

# --- Створюємо таблицю ---
table_entries = []
for i in range(rows):
    row_entries = []
    for j in range(cols):
        e = tk.Entry(root, width=12)
        e.grid(row=i+1, column=j, padx=1, pady=1)
        row_entries.append(e)
    table_entries.append(row_entries)

# Заголовки стовпчиків
for j, name in enumerate(col_names):
    label = tk.Label(root, text=name, relief=tk.RIDGE, width=12)
    label.grid(row=0, column=j, padx=1, pady=1)

# --- Кнопки керування ---
button_frame = tk.Frame(root)
button_frame.grid(row=rows+1, column=0, columnspan=cols, pady=10)

tk.Button(button_frame, text="Додати рядок", command=lambda: add_row()).pack(side=tk.LEFT, padx=5)
tk.Button(button_frame, text="Видалити рядок", command=lambda: delete_row()).pack(side=tk.LEFT, padx=5)
tk.Button(button_frame, text="Додати стовпчик", command=lambda: add_col()).pack(side=tk.LEFT, padx=5)
tk.Button(button_frame, text="Видалити стовпчик", command=lambda: delete_col()).pack(side=tk.LEFT, padx=5)
tk.Button(button_frame, text="Аналіз", command=run_analysis).pack(side=tk.LEFT, padx=5)

# --- Функції керування таблицею ---
def add_row():
    global rows
    row_entries = []
    for j in range(cols):
        e = tk.Entry(root, width=12)
        e.grid(row=rows+1, column=j, padx=1, pady=1)
        row_entries.append(e)
    table_entries.append(row_entries)
    rows += 1

def delete_row():
    global rows
    if rows > 0:
        for j in range(cols):
            table_entries[-1][j].destroy()
        table_entries.pop()
        rows -= 1

def add_col():
    global cols
    for i in range(rows):
        e = tk.Entry(root, width=12)
        e.grid(row=i+1, column=cols, padx=1, pady=1)
        table_entries[i].append(e)
    label = tk.Label(root, text=f"Фактор {cols+1}", relief=tk.RIDGE, width=12)
    label.grid(row=0, column=cols, padx=1, pady=1)
    cols += 1

def delete_col():
    global cols
    if cols > 0:
        for i in range(rows):
            table_entries[i][-1].destroy()
            table_entries[i].pop()
        root.grid_slaves(row=0, column=cols-1)[0].destroy()
        cols -= 1

# --- Запуск ---
root.mainloop()
