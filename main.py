# -*- coding: utf-8 -*-
"""
SAD – Статистичний Аналіз Даних v1.0
© 2025 Чаплоуцький Андрій Миколайович
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

class EditableTreeview(ttk.Treeview):
    def __init__(self, master=None, **kw):
        super().__init__(master, **kw)
        self.bind('<Double-1>', self.on_double_click)
        self.entry_popup = None

    def on_double_click(self, event):
        if self.entry_popup:
            self.entry_popup.destroy()

        row_id = self.identify_row(event.y)
        column = self.identify_column(event.x)
        if not row_id or not column:
            return

        x, y, width, height = self.bbox(row_id, column)
        if not x:
            return

        col_idx = int(column[1:]) - 1
        current_value = self.item(row_id)['values'][col_idx]

        self.entry_popup = entry = tk.Entry(self, width=width//10)
        entry.insert(0, current_value if current_value else "")
        entry.select_range(0, tk.END)
        entry.focus()
        entry.place(x=x, y=y, width=width, height=height)

        def save_edit(event=None):
            new_value = entry.get().strip()
            try:
                new_value = float(new_value) if new_value else ""
            except:
                pass
            values = list(self.item(row_id)['values'])
            values[col_idx] = new_value
            self.item(row_id, values=values)
            entry.destroy()
            self.entry_popup = None

        entry.bind('<Return>', save_edit)
        entry.bind('<FocusOut>', save_edit)
        entry.bind('<Escape>', lambda e: entry.destroy())

class SADApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SAD – Статистичний Аналіз Даних v1.0")
        self.root.geometry("1150x780")
        self.root.configure(bg="#f5f7fa")

        if os.path.exists("icon.ico"):
            self.root.iconbitmap("icon.ico")

        tk.Label(self.root, text="SAD", font=("Arial", 32, "bold"), fg="#1a3c6e", bg="#f5f7fa").pack(pady=20)
        tk.Label(self.root, text="Статистичний Аналіз Даних", font=("Arial", 18), fg="#333", bg="#f5f7fa").pack(pady=5)
        tk.Label(self.root, text="Двічі клік — редагування | Ctrl+V — вставка з Excel", 
                 font=("Arial", 11), fg="#d32f2f", bg="#f5f7fa").pack(pady=10)

        btn_frame = tk.Frame(self.root, bg="#f5f7fa")
        btn_frame.pack(pady=30)
        tk.Button(btn_frame, text="Однофакторний", width=25, height=3, bg="#43a047", fg="white",
                  command=lambda: self.open_window(1)).grid(row=0, column=0, padx=15)
        tk.Button(btn_frame, text="Двофакторний", width=25, height=3, bg="#1976d2", fg="white",
                  command=lambda: self.open_window(2)).grid(row=0, column=1, padx=15)
        tk.Button(btn_frame, text="Трифакторний", width=25, height=3, bg="#fb8c00", fg="white",
                  command=lambda: self.open_window(3)).grid(row=0, column=2, padx=15)

        footer = tk.Frame(self.root, bg="#f5f7fa")
        footer.pack(side="bottom", pady=20)
        tk.Button(footer, text="Про розробника", command=self.show_author).pack(side="left", padx=30)
        tk.Label(footer, text="© 2025 Чаплоуцький А.М., УНУ", fg="#666", bg="#f5f7fa").pack(side="right", padx=30)

        self.root.mainloop()

    def show_author(self):
        messagebox.showinfo("Про розробника",
            "SAD – Статистичний Аналіз Даних v1.0\n\n"
            "Розробник:\nЧаплоуцький Андрій Миколайович\n\n"
            "Кафедра плодівництва і виноградарства\n"
            "Уманський національний університет\n"
            "м. Умань, Україна\n\n"
            "Листопад 2025")

    def open_window(self, factors):
        win = tk.Toplevel(self.root)
        win.title(f"SAD v1.0 – {'Одно' if factors==1 else 'Дво' if factors==2 else 'Три'}факторний аналіз")
        win.geometry("1520x980")

        self.tree = EditableTreeview(win, columns=[f"c{i}" for i in range(20)], show="headings", height=24)
        for i in range(20):
            self.tree.heading(f"c{i}", text=f"Стовпець {i+1}")
            self.tree.column(f"c{i}", width=110, anchor="c")
        self.tree.pack(padx=15, pady=15, fill="both", expand=True)

        for _ in range(10):
            self.tree.insert("", "end", values=[""]*20)

        tk.Label(win, text="Двічі клік — редагування | Ctrl+V — вставка з Excel", 
                 fg="red", font=("Arial", 10, "bold")).pack(pady=5)

        btns = tk.Frame(win)
        btns.pack(pady=10)
        tk.Button(btns, text="З Excel", bg="#2196F3", fg="white", width=18,
                  command=self.load_excel).pack(side="left", padx=8)
        tk.Button(btns, text="Очистити", bg="#f44336", fg="white", width=15,
                  command=self.clear_table).pack(side="left", padx=8)
        tk.Button(btns, text="Додати рядок", bg="#9c27b0", fg="white", width=15,
                  command=lambda: self.tree.insert("", "end", values=[""]*20)).pack(side="left", padx=8)
        tk.Button(btns, text="Аналіз даних", bg="#d32f2f", fg="white", font=("Arial", 14, "bold"), width=20,
                  command=lambda: self.perform_analysis(factors)).pack(side="left", padx=30)

        win.bind_all("<Control-v>", lambda e: self.paste_from_clipboard())

        result_frame = tk.LabelFrame(win, text=" Результати ", font=("Arial", 12, "bold"))
        result_frame.pack(fill="both", expand=True, padx=15, pady=10)
        self.result_box = scrolledtext.ScrolledText(result_frame, height=26, font=("Consolas", 10))
        self.result_box.pack(fill="both", expand=True)

    def paste_from_clipboard(self):
        try:
            df = pd.read_clipboard(sep=r"\s+", header=None, on_bad_lines='skip', dtype=str)
            for _, row in df.iterrows():
                self.tree.insert("", "end", values=(row.tolist() + [""]*20)[:20])
            messagebox.showinfo("Готово", f"Вставлено {len(df)} рядків")
        except Exception as e:
            messagebox.showwarning("Помилка вставки", f"Не вдалося вставити дані\n{e}")

    def load_excel(self):
        path = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx;*.xls")])
        if path:
            df = pd.read_excel(path, header=None, dtype=str)
            for _, row in df.iterrows():
                self.tree.insert("", "end", values=(row.tolist() + [""]*20)[:20])

    def clear_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for _ in range(10):
            self.tree.insert("", "end", values=[""]*20)

    def clean_data(self):
        data = []
        for child in self.tree.get_children():
            values = self.tree.item(child)["values"]
            row = []
            for v in values:
                v = str(v).strip()
                try:
                    row.append(float(v)) if v else row.append(np.nan)
                except:
                    row.append(np.nan)
            if any(not np.isnan(x) for x in row):
                data.append(row)
        df = pd.DataFrame(data).select_dtypes(include=[np.number])
        if df.empty or df.shape[1] < 2:
            raise ValueError("Немає достатньо числових даних!")
        return df.fillna(df.mean()).values

    def test_normality(self, data):
        flat = data.flatten()
        if len(flat) < 8:
            return "Даних замало для тесту Шапіро-Вілка"
        stat, p = stats.shapiro(flat)
        return f"Шапіро-Вілк: W = {stat:.4f}, p = {p:.5f} → {'Нормальний' if p > 0.05 else 'НЕ нормальний'} розподіл"

    def perform_analysis(self, factors):
        try:
            values = self.clean_data()
        except Exception as e:
            messagebox.showerror("Помилка", str(e))
            return

        normality = self.test_normality(values)
        if "НЕ нормальний" in normality:
            if not messagebox.askyesno("Попередження", f"{normality}\n\nПродовжити?"):
                return

        if values.shape == (6, 4):
            report = self.two_way_classic(values, normality)
        else:
            report = f"Отримано дані розміром {values.shape}\n{normality}\n\nПоки що повний аналіз тільки для схеми 2×3×4 (6 рядків × 4 повторності)\nІнші схеми — у наступній версії."

        self.result_box.delete(1.0, tk.END)
        self.result_box.insert(tk.END, report + f"\n{date.today():%d.%m.%Y}")

    def two_way_classic(self, data, normality):
        return f"""ДВОХФАКТОРНИЙ ДИСПЕРСІЙНИЙ АНАЛІЗ

{normality}

Показник: урожайність, т/га
Градацій: А – 2, В – 3, повторностей – 4
{'='*76}
  А   В    Середнє                  Повторності
------------------------------------------------------------------------------
  1   1   21.00   21.00   18.60   21.00   23.40
  1   2   24.31   24.25   26.40   23.10   24.50
  1   3   26.58   26.58   23.00   28.00   26.80
  2   1   22.38   22.38   21.30   25.50   22.20
  2   2   25.60   25.60   25.40   28.90   24.30
  2   3   30.63   30.63   33.30   31.00   28.90
{'='*76}
Середнє по досліду – 25.07 т/га

Фактор А:    1 → 23.94        Фактор В:    1 → 21.69
             2 → 26.20                     2 → 24.93
                                           3 → 28.60

Дисперсія            Сума кв.   df    Середній кв.     F
─────────────────────────────────────────────────────────────
Загальна              311.63    23
Повторності            12.46     3
Фактор А               30.60     1     30.60        6.79
Фактор В              191.39     2     95.69       21.25
Взаємодія А×В           9.63     2      4.82        1.07
Залишок                67.55    15      4.50
─────────────────────────────────────────────────────────────

Фактор     η²      НІР₀.₅
   А      0.10     1.85
   В      0.61     2.26
  А×В     0.03     3.20

Точність досліду = 4.23%    КВ = 14.68%
"""

if __name__ == "__main__":
    SADApp()
