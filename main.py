# -*- coding: utf-8 -*-
"""
SAD v2.0 — Повний дво- та трифакторний дисперсійний аналіз
Українська заміна Statistica / SPSS
Автор: Чаплоуцький Андрій Миколайович, Уманський НУС, 2025
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext, simpledialog
import pandas as pd
import numpy as np
from scipy.stats import f, t, shapiro
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings("ignore")


class SADv2:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SAD v2.0 — Дво- та трифакторний аналіз")
        self.root.geometry("1700x1000")
        self.root.configure(bg="#f5f5f5")

        tk.Label(self.root, text="SAD v2.0", font=("Arial", 48, "bold"), fg="#d32f2f", bg="#f5f5f5").pack(pady=30)
        tk.Label(self.root, text="Повний дво- та трифакторний дисперсійний аналіз", font=("Arial", 18), bg="#f5f5f5").pack(pady=10)

        tk.Button(self.root, text="РОЗПОЧАТИ АНАЛІЗ", font=("Arial", 20, "bold"), bg="#d32f2f", fg="white",
                  width=40, height=2, command=self.start).pack(pady=50)

        tk.Label(self.root, text="© Чаплоуцький А.М. • Уманський НУС • 2025", fg="gray", bg="#f5f5f5").pack(side="bottom", pady=20)
        self.root.mainloop()

    def start(self):
        self.win = tk.Toplevel(self.root)
        self.win.title("SAD v2.0 — Ввід даних")
        self.win.geometry("1800x1000")

        # Панель
        top = tk.Frame(self.win, bg="#e8f5e8")
        top.pack(fill="x", padx=15, pady=10)
        tk.Button(top, text="З Excel", command=self.load_excel).pack(side="left", padx=5)
        tk.Button(top, text="Очистити", bg="#ff5252", fg="white", command=self.clear).pack(side="left", padx=5)
        tk.Button(top, text="АНАЛІЗ", bg="#d32f2f", fg="white", font=("Arial", 20, "bold"),
                  command=self.analyze).pack(side="right", padx=30)

        # Таблиця
        frame = tk.Frame(self.win)
        frame.pack(fill="both", expand=True, padx=15, pady=10)
        self.tree = ttk.Treeview(frame, columns=[f"c{i}" for i in range(20)], show="headings", style="Treeview")
        s = ttk.Style()
        s.configure("Treeview", rowheight=28, font=("Calibri", 11), borderwidth=1, relief="solid")
        s.configure("Treeview.Heading", font=("Calibri", 11, "bold"), background="#f0f0f0")
        for i in range(20):
            self.tree.heading(f"c{i}", text=str(i+1))
            self.tree.column(f"c{i}", width=120, anchor="center")
        for _ in range(50):
            self.tree.insert("", "end", values=[""]*20)
        self.tree.pack(side="left", fill="both", expand=True)

        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(self.win, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")

        # Підказка
        tk.Label(self.win, text="Ctrl+V з Excel • Подвійний клік — редагувати", font=("Arial", 12, "bold"), fg="#d32f2f").pack(pady=5)

        # Результати
        res = tk.LabelFrame(self.win, text=" РЕЗУЛЬТАТИ ДВОФАКТОРНОГО АНАЛІЗУ ", font=("Arial", 14, "bold"))
        res.pack(fill="both", expand=True, padx=15, pady=10)
        self.result = scrolledtext.ScrolledText(res, font=("Consolas", 11), bg="#fffdf0")
        self.result.pack(fill="both", expand=True)

        self.win.bind_all("<Control-v>", lambda e: self.paste())

    def paste(self):
        try:
            df = pd.read_clipboard(sep=r"\s+", header=None, dtype=str)
            self.clear()
            for _, row in df.iterrows():
                vals = row.tolist() + [""]*(20-len(row))
                self.tree.insert("", "end", values=vals[:20])
        except: pass

    def load_excel(self):
        path = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx")])
        if path:
            df = pd.read_excel(path, header=None).astype(str).fillna("")
            self.clear()
            for _, row in df.iterrows():
                vals = row.tolist() + [""]*(20-len(row))
                self.tree.insert("", "end", values=vals[:20])

    def clear(self):
        for i in self.tree.get_children(): self.tree.delete(i)
        for _ in range(50): self.tree.insert("", "end", values=[""]*20)

    def get_data(self):
        data = []
        for iid in self.tree.get_children():
            row = [str(x).strip() for x in self.tree.item(iid, "values")]
            if any(row): data.append(row)
        return pd.DataFrame(data)

    def analyze(self):
        try:
            df = self.get_data()
            if df.shape[1] < 4:
                messagebox.showerror("Помилка", "Мінімум 4 стовпці: фактори + повторності")
                return

            n_factors = simpledialog.askinteger("Фактори", "Кількість факторів (1–3):", minvalue=1, maxvalue=3)
            if not n_factors: return

            # Підготовка даних
            factors = df.iloc[:, :n_factors].copy()
            repeats = df.iloc[:, n_factors:].apply(pd.to_numeric, errors='coerce')
            long_df = repeats.stack().reset_index()
            long_df.columns = ["row"] + [f"rep{i}" for i in range(repeats.shape[1])] + ["value"]
            for i in range(n_factors):
                long_df[f"factor{i+1}"] = factors.iloc[long_df["row"], i].values
            long_df = long_df.drop(columns=["row"] + [f"rep{i}" for i in range(repeats.shape[1])])

            # Формула
            formula = "value ~ C(factor1)"
            if n_factors >= 2: formula += " + C(factor2)"
            if n_factors >= 3: formula += " + C(factor3)"
            if n_factors >= 2: formula += " + C(factor1):C(factor2)"
            if n_factors == 3: formula += " + C(factor1):C(factor3) + C(factor2):C(factor3) + C(factor1):C(factor2):C(factor3)"

            model = ols(formula, data=long_df).fit()
            anova_table = model.summary2().tables[1]

            # Shapiro-Wilk
            residuals = model.resid
            if len(residuals) >= 8:
                _, p_sw = shapiro(residuals)
                normality = "нормальний" if p_sw > 0.05 else "НЕ нормальний"
            else:
                normality = "недостатньо даних"

            # Tukey HSD для комбінацій
            combo = long_df[[f"factor{i+1}" for i in range(n_factors)]].apply(lambda x: " | ".join(x.astype(str)), axis=1)
            tukey = pairwise_tukeyhsd(long_df['value'], combo)
            tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])

            # Звіт
            report = [
                "═" * 90,
                "           ПОВНИЙ ДВОФАКТОРНИЙ ДИСПЕРСІЙНИЙ АНАЛІЗ (SAD v2.0)",
                "═" * 90,
                f"Кількість факторів: {n_factors}   Повторностей: {repeats.shape[1]}",
                f"Перевірка нормальності залишків: {normality}",
                "",
                "Таблиця дисперсійного аналізу:",
                anova_table.round(4).to_string(),
                "",
                "НІР₀.₅ (Tukey HSD) для комбінацій:",
                "Найменші істотні різниці між варіантами — у таблиці Tukey нижче:",
                tukey_df.round(4).to_string(index=False),
                "",
                "Розроблено: Чаплоуцький А.М., Уманський НУС, 2025",
                "Це — офіційна українська заміна Statistica"
            ]

            self.result.delete(1.0, tk.END)
            self.result.insert(tk.END, "\n".join(report))
            self.result.clipboard_clear()
            self.result.clipboard_append("\n".join(report))

            messagebox.showinfo("ГОТОВО!", "Двофакторний аналіз завершено!\nЗвіт скопійовано в буфер обміну")

        except Exception as e:
            messagebox.showerror("Помилка", str(e))


if __name__ == "__main__":
    SADv2()
