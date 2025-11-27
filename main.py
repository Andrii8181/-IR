# -*- coding: utf-8 -*-
"""
SAD v3.1 — Універсальний одно-, дво- та трифакторний дисперсійний аналіз
Фінальна версія 2025 – точно за затвердженим шаблоном звіту
Автор: Чаплоуцький Андрій Миколайович, Уманський НУС
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext, simpledialog
import pandas as pd
import numpy as np
from scipy.stats import f, t, shapiro
from itertools import product


class SAD:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SAD v3.1 — Універсальний аналіз (1-3 фактори)")
        self.root.geometry("1850x1050")
        self.root.configure(bg="#f5f5f5")

        tk.Label(self.root, text="SAD v3.1", font=("Arial", 48, "bold"), fg="#d32f2f", bg="#f5f5f5").pack(pady=40)
        tk.Label(self.root, text="Одно-, дво- та трифакторний дисперсійний аналіз", font=("Arial", 18), bg="#f5f5f5").pack(pady=10)
        tk.Button(self.root, text="РОЗПОЧАТИ АНАЛІЗ", font=("Arial", 22, "bold"), bg="#d32f2f", fg="white",
                  width=40, height=2, command=self.start).pack(pady=60)
        tk.Label(self.root, text="© Уманський НУС • 2025", fg="gray", bg="#f5f5f5").pack(side="bottom", pady=20)
        self.root.mainloop()

    def start(self):
        self.win = tk.Toplevel(self.root)
        self.win.title("SAD v3.1 — Ввід даних")
        self.win.geometry("1900x1050")

        # Панель інструментів
        top = tk.Frame(self.win, bg="#e3f2fd")
        top.pack(fill="x", padx=15, pady=10)
        tk.Button(top, text="З Excel", command=self.load_excel).pack(side="left", padx=5)
        tk.Button(top, text="Очистити", bg="#ff5252", fg="white", command=self.clear).pack(side="left", padx=5)
        tk.Button(top, text="АНАЛІЗ", bg="#d32f2f", fg="white", font=("Arial", 20, "bold"),
                  command=self.analyze).pack(side="right", padx=30)

        # Таблиця
        frame = tk.Frame(self.win)
        frame.pack(fill="both", expand=True, padx=15, pady=10)
        self.tree = ttk.Treeview(frame, columns=[f"c{i}" for i in range(30)], show="headings")
        style = ttk.Style()
        style.configure("Treeview", rowheight=28, font=("Calibri", 11), background="white")
        style.configure("Treeview.Heading", font=("Calibri", 11, "bold"), background="#f0f0f0")
        for i in range(30):
            self.tree.heading(f"c{i}", text=str(i+1))
            self.tree.column(f"c{i}", width=110, anchor="center")
        for _ in range(80):
            self.tree.insert("", "end", values=[""]*30)
        self.tree.pack(side="left", fill="both", expand=True)

        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(self.win, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")

        tk.Label(self.win, text="Ctrl+V з Excel • Подвійний клік — редагування", font=("Arial", 12, "bold"), fg="#d32f2f").pack(pady=5)

        # Результати
        res = tk.LabelFrame(self.win, text=" РЕЗУЛЬТАТИ АНАЛІЗУ ", font=("Arial", 14, "bold"))
        res.pack(fill="both", expand=True, padx=15, pady=10)
        self.result = scrolledtext.ScrolledText(res, font=("Consolas", 11), bg="#fffdf0")
        self.result.pack(fill="both", expand=True)

        self.win.bind_all("<Control-v>", lambda e: self.paste())

    def paste(self):
        try:
            df = pd.read_clipboard(sep=r"\s+", header=None, dtype=str, engine="python")
            self.clear()
            for _, row in df.iterrows():
                vals = row.tolist() + [""]*(30-len(row))
                self.tree.insert("", "end", values=vals[:30])
        except: pass

    def load_excel(self):
        path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        if path:
            df = pd.read_excel(path, header=None).astype(str).fillna("")
            self.clear()
            for _, row in df.iterrows():
                vals = row.tolist() + [""]*(30-len(row))
                self.tree.insert("", "end", values=vals[:30])

    def clear(self):
        for i in self.tree.get_children(): self.tree.delete(i)
        for _ in range(80): self.tree.insert("", "end", values=[""]*30)

    def get_data(self):
        data = []
        for iid in self.tree.get_children():
            row = [str(x).strip() for x in self.tree.item(iid, "values")]
            if any(x != "" for x in row):
                data.append(row)
        return pd.DataFrame(data)

    def analyze(self):
        try:
            df = self.get_data()
            if df.shape[1] < 4:
                messagebox.showerror("Помилка", "Потрібно мінімум 4 стовпці")
                return

            n_factors = simpledialog.askinteger("Кількість факторів", "Введіть кількість факторів (1–3):", minvalue=1, maxvalue=3)
            if not n_factors: return

            factors_df = df.iloc[:, :n_factors].copy()
            repeats = df.iloc[:, n_factors:].apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
            if repeats.shape[1] < 2:
                messagebox.showerror("Помилка", "Мінімум 2 повторення")
                return

            n_rep = repeats.shape[1]

            # Перетворення у довгий формат
            long_data = []
            for idx, row in factors_df.iterrows():
                for col in repeats.columns:
                    val = repeats.at[idx, col]
                    if pd.notna(val):
                        long_data.append(list(row) + [val])
            cols = [f"F{i+1}" for i in range(n_factors)] + ["value"]
            data_long = pd.DataFrame(long_data, columns=cols)

            report = self.generate_report(data_long, n_factors, n_rep, factors_df)
            self.result.delete(1.0, tk.END)
            self.result.insert(tk.END, report)
            self.result.clipboard_clear()
            self.result.clipboard_append(report)
            messagebox.showinfo("Готово!", f"{n_factors}-факторний аналіз завершено!")

        except Exception as e:
            messagebox.showerror("Помилка", str(e))

    def generate_report(self, data, n_factors, n_rep, factors_df):
        value = data["value"].astype(float)
        grand_mean = value.mean()

        # Залишки для Shapiro-Wilk
        residuals = value - data.groupby([f"F{i+1}" for i in range(n_factors)])["value"].transform('mean')
        if len(residuals) >= 8:
            _, p_sw = shapiro(residuals)
            normality = f"нормальний (p = {p_sw:.3f})" if p_sw > 0.05 else f"НЕ нормальний (p = {p_sw:.3f})"
        else:
            normality = "недостатньо даних"

        # Факторні назви
        factor_names = ["А", "В", "С"][:n_factors]
        factor_levels = [len(data[f"F{i+1}"].unique()) for i in range(n_factors)]

        # Дисперсійний аналіз (ручний розрахунок)
        ss_total = ((value - grand_mean)**2).sum()
        results = []

        # Головні ефекти
        ss_main = {}
        df_main = {}
        ms_main = {}
        for i in range(n_factors):
            col = f"F{i+1}"
            group_means = data.groupby(col)["value"].mean()
            ss = n_rep * len(data) / len(group_means) * ((group_means - grand_mean)**2).sum()
            df = len(group_means) - 1
            ms = ss / df if df > 0 else 0
            ss_main[i] = ss
            df_main[i] = df
            ms_main[i] = ms

        # Взаємодії (тільки до 3 факторів)
        ss_inter = {}
        df_inter = {}
        ms_inter = {}
        if n_factors >= 2:
            for comb in [(0,1), (0,2), (1,2)] if n_factors == 3 else [(0,1)]:
                cols = [f"F{i+1}" for i in comb]
                group_means = data.groupby(cols)["value"].mean()
                ss = n_rep * ((group_means - grand_mean)**2).sum()
                for i in comb:
                    main_means = data.groupby(f"F{i+1}")["value"].mean()
                    ss -= n_rep * len(data) / len(main_means) * ((main_means - grand_mean)**2).sum()
                df = np.prod([factor_levels[j]-1 for j in comb])
                ms = ss / df if df > 0 else 0
                ss_inter[comb] = ss
                df_inter[comb] = df
                ms_inter[comb] = ms

        # Залишкова сума квадратів
        ss_error = ss_total
        for ss in ss_main.values(): ss_error -= ss
        for ss in ss_inter.values(): ss_error -= ss
        df_error = len(value) - np.prod(factor_levels)
        ms_error = ss_error / df_error if df_error > 0 else 0

        # F-критерії та висновки
        def f_test(ms_effect, name):
            if ms_error == 0: return "—", "—", "невідомо"
            F = ms_effect / ms_error
            df1 = locals()[f"df_{name}"] if "df_" + name in locals() else df_main[int(name.split()[-1])-1]
            df2 = df_error
            F_crit = f.ppf(0.95, df1, df2)
            sign = "**" if F >= f.ppf(0.99, df1, df2) else "*" if F >= F_crit else ""
            conclusion = "істотний" if F >= F_crit else "неістотний"
            return f"{F:.2f}{sign}", f"{F_crit:.2f}", conclusion

        # Формування таблиці ANOVA
        lines = []
        lines.append("────────────────────────────────────────────────────────────────────")
        lines.append("Джерело варіації                  Сума квадратів   Ступені свободи   Середній квадрат   Fрозрахункове   Fтабличне   Висновок")
        lines.append("────────────────────────────────────────────────────────────────────")

        # Головні ефекти
        for i in range(n_factors):
            F_val, F_crit, conc = f_test(ms_main[i], f"main{i}")
            lines.append(f"Фактор {factor_names[i]} ({factors_df.iloc[:,i].name if hasattr(factors_df.iloc[:,i], 'name') else 'А'}) "
                        f"{ss_main[i]:>14.2f} {df_main[i]:>15} {ms_main[i]:>16.3f} {F_val:>14} {F_crit:>10}   {conc}")

        # Взаємодії
        for comb, ss in ss_inter.items():
            name = " × ".join(factor_names[j] for j in comb)
            F_val, F_crit, conc = f_test(ms_inter[comb], f"inter{comb}")
            lines.append(f"Взаємодія {name:<20} {ss:>14.2f} {df_inter[comb]:>15} {ms_inter[comb]:>16.3f} {F_val:>14} {F_crit:>10}   {conc}")

        lines.append(f"Випадкова помилка               {ss_error:>14.2f} {df_error:>15} {ms_error:>16.3f}")
        lines.append("────────────────────────────────────────────────────────────────────")
        lines.append(f"Загальна                        {ss_total:>14.2f} {len(value)-1:>15}")

        # Вилучення впливу
        eta_lines = []
        total_explained = sum(ss_main.values()) + sum(ss_inter.values())
        for i in range(n_factors):
            eta = ss_main[i] / ss_total * 100
            eta_lines.append(f"  • Фактор {factor_names[i]} — {eta:.1f}%")
        for comb, ss in ss_inter.items():
            name = " × ".join(factor_names[j] for j in comb)
            eta = ss / ss_total * 100
            eta_lines.append(f"  • Взаємодія {name} — {eta:.1f}%")

        # НІР 0.05
        t_crit = t.ppf(0.975, df_error)
        lsd_main = {i: t_crit * np.sqrt(2 * ms_error / (n_rep * np.prod([factor_levels[j] for j in range(n_factors) if j != i]))) 
                    for i in range(n_factors)}
        lsd_comb = t_crit * np.sqrt(2 * ms_error / n_rep)

        # Середні + букви (LSD)
        def assign_letters(means):
            sorted_means = means.sort_values(ascending=False)
            letters = []
            letter = 'a'
            prev = sorted_means.iloc[0] + 1
            for val in sorted_means:
                if val < prev - lsd_comb:
                    letter = chr(ord(letter) + 1)
                letters.append(letter)
                prev = val
            return dict(zip(sorted_means.index, letters))

        # Заголовок
        title = "О Д Н О" if n_factors == 1 else "Д В О" if n_factors == 2 else "Т Р И"
        report = [f"Р Е З У Л Ь Т А Т И   {title} Ф А К Т О Р Н О Г О   Д И С П Е Р С І Й Н О Г О   А Н А Л І З У\n"]

        report.append(f"Фактор А: {factors_df.columns[0]} ({factor_levels[0]} рівнів)" if n_factors >= 1 else "")
        if n_factors >= 2: report.append(f"Фактор В: {factors_df.columns[1]} ({factor_levels[1]} рівнів)")
        if n_factors == 3: report.append(f"Фактор С: {factors_df.columns[2]} ({factor_levels[2]} рівнів)")
        report.append(f"Кількість повторень: {n_rep}\n")
        report.append(f"Перевірка нормальності залишків (Shapiro-Wilk): {normality}\n")
        report.extend(lines)
        report.append("\nВилучення впливу:")
        report.extend(eta_lines)
        report.append("\nНІР₀.₅:")
        for i in range(n_factors):
            report.append(f"  • По фактору {factor_names[i]} — {lsd_main[i]:.2f} ц/га")
        report.append(f"  • По комбінаціях — {lsd_comb:.2f} ц/га\n")

        # Середні по кожному фактору
        for i in range(n_factors):
            col = f"F{i+1}"
            means = data.groupby(col)["value"].mean().round(1)
            letters = assign_letters(means)
            report.append(f"Середні по фактору {factor_names[i]}:")
            for level, mean in means.items():
                report.append(f"  {level:<25} {mean} {letters[level]}")
            report.append("")

        # Комбінації
        report.append("Комбінації (з буквами істотності):")
        combo_means = data.groupby([f"F{i+1}" for i in range(n_factors)])["value"].mean().round(1)
        letters = assign_letters(combo_means)
        for idx, mean in combo_means.items():
            combo_str = " | ".join(str(x) for x in idx)
            report.append(f"  {combo_str:<40} → {mean} {letters[idx]}")

        return "\n".join(report)


if __name__ == "__main__":
    SAD()
