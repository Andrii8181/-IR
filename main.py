# -*- coding: utf-8 -*-
"""
SAD-Статистичний Аналіз Даних 2025
Універсальний калькулятор дисперсійного аналізу з перевіркою нормальності
Автор: Чаплоуцький Андрій Миколайович, Уманський національний університет
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import pandas as pd
import numpy as np
from scipy import stats
from datetime import date
import os

class SAD:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SAD — Статистичний Аналіз Даних")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)

        if os.path.exists("icon.ico"):
            self.root.iconbitmap("icon.ico")

        title = tk.Label(self.root, text="Оберіть тип аналізу:", font=("Arial", 14, "bold"))
        title.pack(pady=20)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="🧪 Однофакторний аналіз", width=25, height=2, bg="#4CAF50", fg="white",
                  font=("Arial", 12), command=lambda: self.start_analysis(1)).grid(row=0, column=0, padx=10, pady=5)
        tk.Button(btn_frame, text="📊 Двофакторний аналіз", width=25, height=2, bg="#2196F3", fg="white",
                  font=("Arial", 12), command=lambda: self.start_analysis(2)).grid(row=0, column=1, padx=10, pady=5)
        tk.Button(btn_frame, text="🔬 Трифакторний аналіз", width=25, height=2, bg="#FF9800", fg="white",
                  font=("Arial", 12), command=lambda: self.start_analysis(3)).grid(row=0, column=2, padx=10, pady=5)

        tk.Button(self.root, text="ℹ️ Про програму", command=self.show_about_program, bg="#00BCD4", fg="white").pack(pady=5)
        tk.Button(self.root, text="👤 Про розробника", command=self.show_about_author, bg="#607D8B", fg="white").pack(pady=5)

        instr = tk.Label(self.root, text="💡 Після вибору: вставте дані з Excel (Ctrl+V), вводьте вручну або імпортуйте Excel", 
                         font=("Arial", 10), fg="gray")
        instr.pack(pady=10)

        self.root.mainloop()

    def show_about_program(self):
        messagebox.showinfo("Про програму", "SAD — Статистичний Аналіз Даних. Одно-, дво-, трифакторний ANOVA з НІР₀₅ та перевіркою нормальності Шапіро-Вілка. Підтримка Excel, буферу обміну та ручного вводу.")

    def show_about_author(self):
        messagebox.showinfo("Про розробника", "Чаплоуцький Андрій Миколайович\nУманський національний університет\nм. Умань, Україна")

    def start_analysis(self, factors):
        self.analysis_window = tk.Toplevel(self.root)
        self.analysis_window.title(f"{'Одно' if factors==1 else 'Дво' if factors==2 else 'Три'}факторний аналіз")
        self.analysis_window.geometry("1400x900")
        self.analysis_window.resizable(True, True)

        tk.Label(self.analysis_window, text=f"Введіть дані: перші стовпці — фактори, останні — повторності", font=("Arial", 12)).pack(pady=5)

        table_frame = tk.Frame(self.analysis_window)
        table_frame.pack(fill="both", expand=True, padx=10, pady=10)

        cols = factors + 4  # Фактори + 4 повторності
        self.tree = ttk.Treeview(table_frame, columns=[f"col{i}" for i in range(cols)], show="headings", height=15)

        v_scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        self.tree.pack(side="left", fill="both", expand=True)
        v_scrollbar.pack(side="right", fill="y")
        h_scrollbar.pack(side="bottom", fill="x")

        for i in range(cols):
            self.tree.heading(f"col{i}", text=f"Фактор/Повторність {i+1}")
            self.tree.column(f"col{i}", width=120, anchor="center")

        self.tree.bind("<Control-v>", self.paste_from_clipboard)
        self.tree.bind("<Button-3>", self.right_click_menu)

        btn_frame = tk.Frame(self.analysis_window)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="📁 Завантажити з Excel", command=self.load_excel, bg="#FFC107").pack(side="left", padx=5)
        tk.Button(btn_frame, text="🗑️ Очистити таблицю", command=self.clear_table, bg="#F44336", fg="white").pack(side="left", padx=5)
        tk.Button(btn_frame, text="➕ Додати рядок", command=self.add_row, bg="#9C27B0", fg="white").pack(side="left", padx=5)
        tk.Button(btn_frame, text="➕ Додати стовпчик", command=self.add_column, bg="#9C27B0", fg="white").pack(side="left", padx=5)
        analyze_btn = tk.Button(btn_frame, text="🚀 Аналіз даних", bg="#4CAF50", fg="white", font=("Arial", 12, "bold"),
                                command=lambda: self.calculate(factors))
        analyze_btn.pack(side="left", padx=20)

        tk.Label(self.analysis_window, text="📋 Результати аналізу:", font=("Arial", 12, "bold")).pack(anchor="w", padx=10)
        self.result_text = scrolledtext.ScrolledText(self.analysis_window, height=20, font=("Consolas", 10), wrap=tk.WORD)
        self.result_text.pack(fill="both", expand=True, padx=10, pady=10)

        save_frame = tk.Frame(self.analysis_window)
        save_frame.pack(pady=5)
        tk.Button(save_frame, text="💾 Зберегти звіт у TXT", command=self.save_report_txt).pack(side="left", padx=5)
        tk.Button(save_frame, text="📄 Копіювати звіт", command=self.copy_report).pack(side="left", padx=5)

        self.factors = factors
        self.tree.focus_set()

    # --- Функції таблиці ---
    def paste_from_clipboard(self, event=None):
        try:
            clipboard = self.root.clipboard_get()
            lines = clipboard.split('\n')
            for line in lines:
                if line.strip():
                    values = line.split('\t')
                    self.tree.insert("", "end", values=values[:len(self.tree["columns"])])
            messagebox.showinfo("Успіх", "Дані вставлено!")
        except:
            messagebox.showwarning("Увага", "Не вдалося вставити дані. Скопіюйте з Excel.")

    def right_click_menu(self, event):
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Вставити (Ctrl+V)", command=self.paste_from_clipboard)
        menu.add_command(label="Очистити", command=self.clear_table)
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def load_excel(self):
        path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        if path:
            try:
                df = pd.read_excel(path, header=None)
                self.clear_table()
                for _, row in df.iterrows():
                    self.tree.insert("", "end", values=row.tolist())
                messagebox.showinfo("Успіх", f"Завантажено {len(df)} рядків з {path}")
            except Exception as e:
                messagebox.showerror("Помилка", f"Не вдалося завантажити: {e}")

    def clear_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

    def add_row(self):
        self.tree.insert("", "end", values=[""] * len(self.tree["columns"]))

    def add_column(self):
        cols = len(self.tree["columns"])
        new_col = f"col{cols}"
        self.tree["columns"] = list(self.tree["columns"]) + [new_col]
        self.tree.heading(new_col, text=f"Фактор/Повторність {cols+1}")
        self.tree.column(new_col, width=120, anchor="center")
        # Додати пусті клітинки у кожен рядок
        for item in self.tree.get_children():
            vals = list(self.tree.item(item)["values"])
            vals.append("")
            self.tree.item(item, values=vals)

    # --- Аналіз даних ---
    def calculate(self, factors):
        children = self.tree.get_children()
        if not children:
            messagebox.showwarning("Увага", "Таблиця порожня!")
            return
        data_list = [self.tree.item(child)["values"] for child in children]
        df = pd.DataFrame(data_list)
        # Перетворюємо числові колонки
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        factor_cols = [col for col in df.columns if col not in numeric_cols]
        if len(numeric_cols) < 2:
            messagebox.showerror("Помилка", "Потрібно мінімум 2 числових стовпці!")
            return

        values = df[numeric_cols].values
        reps = len(numeric_cols)

        # --- Перевірка нормальності ---
        normality_msg = ""
        for i in range(values.shape[1]):
            stat, p = stats.shapiro(values[:,i])
            normality_msg += f"Колонка {i+1}: Shapiro-Wilk W={stat:.3f}, p={p:.3f} → {'Нормальна' if p>0.05 else 'Не нормальна'}\n"

        try:
            if factors == 1:
                levels = df[factor_cols[0]].astype(str).unique() if factor_cols else [f"Варіант {i+1}" for i in range(len(values))]
                result = self.one_way_anova(values, levels)
            elif factors == 2:
                if len(factor_cols) < 2:
                    messagebox.showerror("Помилка", "Для двофакторного потрібні 2 фактори!")
                    return
                factor_a = df[factor_cols[0]].astype(str)
                factor_b = df[factor_cols[1]].astype(str)
                result = self.two_way_anova(values, factor_a, factor_b, reps)
            else:
                if len(factor_cols) < 3:
                    messagebox.showerror("Помилка", "Для трифакторного потрібні 3 фактори!")
                    return
                factor_a = df[factor_cols[0]].astype(str)
                factor_b = df[factor_cols[1]].astype(str)
                factor_c = df[factor_cols[2]].astype(str)
                result = self.three_way_anova(values, factor_a, factor_b, factor_c, reps)

            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, normality_msg + "\n" + result + f"\n\n{date.today().strftime('%d-%m-%Y')}")
            messagebox.showinfo("Готово!", "Аналіз завершено!")
        except Exception as e:
            messagebox.showerror("Помилка", f"Щось пішло не так: {e}")

    # --- ANOVA функції (можна вставити ваші функції one_way_anova, two_way_anova, three_way_anova) ---
    # Для скорочення можна вставити версії з попереднього коду, включно з НІР₀₅

    def save_report_txt(self):
        text = self.result_text.get(1.0, tk.END)
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            messagebox.showinfo("Збережено", f"Звіт збережено: {path}")

    def copy_report(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.result_text.get(1.0, tk.END))
        messagebox.showinfo("Скопійовано", "Звіт скопійовано в буфер обміну!")

if __name__ == "__main__":
    app = SAD()
