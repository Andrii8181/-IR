# -*- coding: utf-8 -*-
"""
SAD — Статистичний Аналіз Даних (максимально крута версія)
Автор: Чаплоуцький Андрій Миколайович
Уманський національний університет
Версія: 1.1 (покращена)
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext, simpledialog
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from datetime import date
import os

# ---------------- Editable Treeview (double-click edit) ----------------
class EditableTreeview(ttk.Treeview):
    def __init__(self, master=None, **kw):
        super().__init__(master, **kw)
        self.bind('<Double-1>', self._on_double_click)
        self._entry = None

    def _on_double_click(self, event):
        if self._entry:
            self._entry.destroy()
        rowid = self.identify_row(event.y)
        column = self.identify_column(event.x)
        if not rowid or not column:
            return
        bbox = self.bbox(rowid, column)
        if not bbox:
            return
        x, y, width, height = bbox
        col_index = int(column.replace('#', '')) - 1
        values = list(self.item(rowid, 'values'))
        value = values[col_index] if col_index < len(values) else ""
        self._entry = entry = tk.Entry(self)
        entry.insert(0, "" if value is None else str(value))
        entry.select_range(0, tk.END)
        entry.focus()
        # place relative to tree widget coordinates
        entry.place(x=x, y=y, width=width, height=height)

        def save(e=None):
            new_val = entry.get().strip()
            vals = list(self.item(rowid, 'values'))
            # extend if needed
            while len(vals) <= col_index:
                vals.append("")
            vals[col_index] = new_val
            self.item(rowid, values=vals)
            entry.destroy()
            self._entry = None

        entry.bind('<Return>', save)
        entry.bind('<FocusOut>', save)
        entry.bind('<Escape>', lambda e: (entry.destroy(), setattr(self, "_entry", None)))

# ---------------- Main Application ----------------
class SADApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SAD — Статистичний Аналіз Даних (v1.1)")
        self.root.geometry("1200x760")
        if os.path.exists("icon.ico"):
            try:
                self.root.iconbitmap("icon.ico")
            except:
                pass

        header = tk.Frame(self.root)
        header.pack(pady=12)
        tk.Label(header, text="SAD — Статистичний Аналіз Даних", font=("Arial", 26, "bold")).pack()
        tk.Label(header, text="Універсальний інструмент ANOVA • Shapiro-Wilk • Levene • LSD • Tukey",
                 font=("Arial", 10), fg="gray").pack()

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=12)
        tk.Button(btn_frame, text="Почати (Оберіть факторність)", width=30, height=2,
                  command=self.choose_factor_count, bg="#1976D2", fg="white", font=("Arial", 12, "bold")).pack()

        info_frame = tk.Frame(self.root)
        info_frame.pack(pady=8)
        tk.Button(info_frame, text="Про програму", command=self.show_about).pack(side="left", padx=8)
        tk.Button(info_frame, text="Про розробника", command=self.show_author).pack(side="left", padx=8)

        tk.Label(self.root, text="Після відкриття вікна аналізу: вставте дані (Ctrl+V), імпортуйте Excel або введіть вручну.",
                 fg="gray").pack(pady=8)

        self.root.mainloop()

    def show_about(self):
        messagebox.showinfo("Про програму",
                            "SAD — універсальний ANOVA інструмент (одно-, дво-, трифакторний).\n"
                            "Автоматичні таблиці ANOVA, Shapiro-Wilk, Levene, LDS (НІР₀.₅), Tukey HSD.")

    def show_author(self):
        messagebox.showinfo("Про розробника",
                            "Чаплоуцький Андрій Миколайович\n"
                            "Уманський національний університет, м. Умань, Україна")

    def choose_factor_count(self):
        fc = simpledialog.askinteger("Кількість факторів",
                                     "Введіть кількість факторів (1, 2 або 3):",
                                     minvalue=1, maxvalue=3)
        if fc:
            self.open_analysis_window(fc)

    def open_analysis_window(self, factor_count):
        self.factor_count = factor_count
        self.win = tk.Toplevel(self.root)
        self.win.title(f"SAD — {factor_count}-факторний аналіз")
        self.win.geometry("1400x920")

        top_frame = tk.Frame(self.win)
        top_frame.pack(fill="x", padx=8, pady=6)

        tk.Button(top_frame, text="Додати стовпець", command=self.add_column).pack(side="left", padx=6)
        tk.Button(top_frame, text="Додати рядок", command=self.add_row).pack(side="left", padx=6)
        tk.Button(top_frame, text="Очистити таблицю", command=self.clear_table).pack(side="left", padx=6)
        tk.Button(top_frame, text="Імпорт з Excel", command=self.load_excel).pack(side="left", padx=6)
        tk.Button(top_frame, text="🚀 Обчислити", command=self.calculate).pack(side="left", padx=12)
        tk.Button(top_frame, text="Зберегти звіт", command=self.save_report).pack(side="left", padx=6)

        # Table area
        table_frame = tk.Frame(self.win)
        table_frame.pack(fill="both", expand=False, padx=8, pady=6)
        # Start with 8 columns (some factor cols + repeats), user can add/remove
        self.initial_cols = max(4, factor_count + 3)
        cols = [f"c{i}" for i in range(self.initial_cols)]
        self.tree = EditableTreeview(table_frame, columns=cols, show="headings", height=18)
        for i, c in enumerate(cols):
            self.tree.heading(c, text=f"Col {i+1}")
            self.tree.column(c, width=110, anchor="center")
        # Fill 12 rows
        for _ in range(12):
            self.tree.insert("", "end", values=[""]*len(cols))
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        # Tips & controls
        tk.Label(self.win, text="Подвійний клік — редагувати комірку. Ctrl+V — вставити з Excel.",
                 fg="gray").pack(pady=4)

        # Results
        res_frame = tk.LabelFrame(self.win, text="Результат", font=("Arial", 12, "bold"))
        res_frame.pack(fill="both", expand=True, padx=8, pady=8)
        self.result_box = scrolledtext.ScrolledText(res_frame, height=18, font=("Consolas", 11))
        self.result_box.pack(fill="both", expand=True)

        # Bind paste
        self.win.bind_all("<Control-v>", self.on_paste_clipboard)

    # ---------- Table operations ----------
    def add_column(self):
        cur = list(self.tree["columns"])
        idx = len(cur)
        new = f"c{idx}"
        cur.append(new)
        self.tree["columns"] = cur
        self.tree.heading(new, text=f"Col {idx+1}")
        self.tree.column(new, width=110, anchor="center")
        # extend rows' values
        for iid in self.tree.get_children():
            vals = list(self.tree.item(iid, 'values'))
            vals += [""] * (len(cur)-len(vals))
            self.tree.item(iid, values=vals)

    def add_row(self):
        self.tree.insert("", "end", values=[""]*len(self.tree["columns"]))

    def clear_table(self):
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        # refill a few empty rows
        for _ in range(12):
            self.tree.insert("", "end", values=[""]*len(self.tree["columns"]))

    def load_excel(self):
        path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx;*.xls")])
        if not path:
            return
        try:
            df = pd.read_excel(path, header=None, dtype=str).fillna("")
            # set columns to match
            ncols = max(df.shape[1], len(self.tree["columns"]))
            # ensure tree has ncols
            while len(self.tree["columns"]) < ncols:
                self.add_column()
            # clear and fill
            self.clear_table()
            for _, row in df.iterrows():
                row_vals = list(row.values) + [""]*(ncols - len(row))
                self.tree.insert("", "end", values=row_vals[:ncols])
            messagebox.showinfo("Імпорт", f"Імпортовано {df.shape[0]} рядків, {df.shape[1]} стовпців")
        except Exception as e:
            messagebox.showerror("Помилка імпорту", str(e))

    def on_paste_clipboard(self, event=None):
        try:
            df = pd.read_clipboard(sep=None, header=None, dtype=str)
        except Exception:
            try:
                txt = self.win.clipboard_get()
                # try TSV
                df = pd.read_csv(pd.io.common.StringIO(txt), sep=None, engine='python', header=None, dtype=str)
            except Exception:
                messagebox.showwarning("Paste", "Не вдалося прочитати дані з буфера")
                return
        # ensure enough cols
        ncols = max(df.shape[1], len(self.tree["columns"]))
        while len(self.tree["columns"]) < ncols:
            self.add_column()
        # insert rows
        for _, row in df.iterrows():
            vals = list(row.values) + [""]*(ncols - df.shape[1])
            self.tree.insert("", "end", values=[str(v) for v in vals[:ncols]])
        messagebox.showinfo("Вставка", f"Вставлено {df.shape[0]} рядків")

    # ---------- Data extraction & conversion ----------
    def tree_to_dataframe(self):
        cols = list(self.tree["columns"])
        data = []
        for iid in self.tree.get_children():
            vals = list(self.tree.item(iid, "values"))
            # pad
            if len(vals) < len(cols):
                vals += [""]*(len(cols)-len(vals))
            data.append(vals)
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=[f"col{i+1}" for i in range(len(cols))])
        # Strip whitespace
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        return df

    def wide_to_long(self, df, n_factor_cols):
        """
        Convert wide table to long format suitable for statsmodels.
        Strategy:
        - First n_factor_cols columns are the factor columns (categorical).
        - Remaining columns are repeated measures (numeric).
        - Melt repeats into 'value' column.
        """
        if df.empty:
            return pd.DataFrame()
        ncols = df.shape[1]
        if n_factor_cols >= ncols:
            raise ValueError("Немає числових колонок для повторностей. Перевірте дані.")
        factor_cols = df.columns[:n_factor_cols].tolist()
        repeat_cols = df.columns[n_factor_cols:].tolist()
        # coerce numeric where possible
        df_repeats = df[repeat_cols].apply(pd.to_numeric, errors='coerce')
        # attach factor cols (keep as string)
        df_factors = df[factor_cols].astype(str)
        # create long
        long = pd.melt(pd.concat([df_factors, df_repeats], axis=1),
                       id_vars=factor_cols, value_vars=repeat_cols,
                       var_name='rep', value_name='value')
        # drop NaN values
        long = long.dropna(subset=['value']).reset_index(drop=True)
        # convert factor columns to categorical
        for c in factor_cols:
            long[c] = long[c].astype('category')
        return long

    # ---------- Statistical helpers ----------
    def fit_anova(self, long_df, factor_cols):
        """
        Build formula dynamically and fit OLS + ANOVA table.
        factor_cols: list of factor column names (categorical)
        """
        # formula: value ~ C(A) + C(B) + C(C) + C(A):C(B) + ... depending on count
        terms = []
        if not factor_cols:
            raise ValueError("No factor columns")
        # main effects
        mains = [f"C({c})" for c in factor_cols]
        terms += mains
        # interactions
        if len(factor_cols) >= 2:
            # include all interactions up to full interaction
            for r in range(2, len(factor_cols)+1):
                from itertools import combinations
                for comb in combinations(factor_cols, r):
                    term = ":".join([f"C({c})" for c in comb])
                    terms.append(term)
        formula = "value ~ " + " + ".join(terms)
        model = smf.ols(formula, data=long_df).fit()
        anova_table = smf.stats.anova_lm(model, typ=2)  # Type II ANOVA
        return model, anova_table

    def calc_lsd_matrix(self, long_df, factor_cols, model, alpha=0.05):
        """
        Compute pairwise LSD thresholds and pairwise comparisons for the highest-order factor combination.
        For 1-factor: compare levels of that factor.
        For >1-factor: user may want to test main effects separately; compute for each main factor.
        Return dict with entries per factor: {factor: (MS_error, dict of pairwise differences and LSD thresholds)}
        """
        from scipy.stats import t
        res = {}
        mse = model.mse_resid
        df_resid = int(model.df_resid)
        tval = t.ppf(1 - alpha/2, df_resid)
        # For each main factor compute pairwise
        for fac in factor_cols:
            groups = long_df.groupby(fac)['value']
            levels = groups.size().index.tolist()
            ns = groups.size().values
            means = groups.mean().values
            # pairwise results
            pairs = []
            for i in range(len(levels)):
                for j in range(i+1, len(levels)):
                    ni = ns[i]; nj = ns[j]
                    se = np.sqrt(mse*(1/ni + 1/nj))
                    lsd = tval * se
                    diff = means[i] - means[j]
                    pairs.append({
                        "level_i": levels[i],
                        "level_j": levels[j],
                        "mean_i": means[i],
                        "mean_j": means[j],
                        "diff": diff,
                        "abs_diff": abs(diff),
                        "lsd": lsd,
                        "signif": abs(diff) > lsd
                    })
            res[fac] = {
                "mse": mse,
                "df_resid": df_resid,
                "pairs": pairs
            }
        return res

    # ---------- Main calculate ----------
    def calculate(self):
        try:
            df = self.tree_to_dataframe()
            if df.empty:
                messagebox.showerror("Помилка", "Таблиця порожня")
                return
            # Ask how many factor columns: default = factor_count (chosen earlier)
            n_factor_cols = self.factor_count
            if n_factor_cols >= df.shape[1]:
                messagebox.showerror("Помилка", "Занадто багато факторних стовпців для наявних колонок")
                return
            long = self.wide_to_long(df, n_factor_cols)
            if long.empty:
                messagebox.showerror("Помилка", "Не знайдено числових повторностей")
                return

            # Shapiro-Wilk on residuals later; first fit model
            factor_cols = list(df.columns[:n_factor_cols])
            model, anova_table = self.fit_anova(long, factor_cols)

            # Levene test for homogeneity of variances (by groups of highest factor if single factor else by interaction)
            # We'll use groups by combination of factor levels
            grouped = long.groupby(factor_cols)['value'].apply(list)
            groups_list = [np.array(x) for x in grouped if len(x) >= 2]
            levene_p = None
            if len(groups_list) >= 2:
                try:
                    stat_levene, levene_p = stats.levene(*groups_list)
                except Exception:
                    levene_p = None

            # Shapiro-Wilk for residuals
            resid = model.resid
            sw_resid_w, sw_resid_p = (None, None)
            try:
                # need >= 3 samples
                if len(resid.dropna()) >= 3:
                    sw_resid_w, sw_resid_p = stats.shapiro(resid)
            except Exception:
                sw_resid_w, sw_resid_p = (None, None)

            # Also Shapiro per group (means or values)
            group_sw = {}
            for name, group in long.groupby(factor_cols):
                arr = group['value'].values
                if len(arr) >= 3:
                    try:
                        w, p = stats.shapiro(arr)
                        group_sw[str(name)] = (w, p)
                    except Exception:
                        group_sw[str(name)] = (None, None)
                else:
                    group_sw[str(name)] = (None, None)

            # LSD calculations
            lsd_info = self.calc_lsd_matrix(long, factor_cols, model)

            # Tukey HSD (optional) - compute for first main factor if present
            tukey_results = None
            try:
                # need at least 2 unique groups
                if len(long[factor_cols[0]].unique()) >= 2:
                    tukey = pairwise_tukeyhsd(endog=long['value'], groups=long[factor_cols[0]], alpha=0.05)
                    tukey_results = tukey.summary().as_text()
            except Exception:
                tukey_results = None

            # Compose report
            report = []
            report.append("=== АНАЛІЗ: SAD (v1.1) ===")
            report.append(f"Дата: {date.today():%d.%m.%Y}")
            report.append(f"Розмір даних (рядків у long): {len(long)}; фактори: {len(factor_cols)} ({', '.join(factor_cols)})")
            report.append("\n--- ANOVA (Type II) ---")
            # format anova_table
            anova_str = anova_table.round(4).to_string()
            report.append(anova_str)
            report.append("\n--- Перевірки нормальності та однорідності ---")
            if sw_resid_w is not None:
                report.append(f"Shapiro-Wilk (залишки): W={sw_resid_w:.4f}, p={sw_resid_p:.5f} -> {'нормальні' if sw_resid_p>0.05 else 'НЕ нормальні'}")
            else:
                report.append("Shapiro-Wilk (залишки): недостатньо даних або тест не застосований")
            if levene_p is not None:
                report.append(f"Levene (однорідність дисперсій): p={levene_p:.5f} -> {'однорідні' if levene_p>0.05 else 'НЕ однорідні'}")
            else:
                report.append("Levene: неможливо обчислити (недостатньо або некоректні групи)")

            # group Shapiro
            report.append("\nShapiro-Wilk по групам:")
            for g, (w, p) in group_sw.items():
                if w is None:
                    report.append(f" {g}: недостатньо даних")
                else:
                    report.append(f" {g}: W={w:.4f}, p={p:.5f} -> {'норм.' if p>0.05 else 'НЕ норм.'}")

            report.append("\n--- НІР₀.₅ (LSD) та парні порівняння ---")
            for fac, info in lsd_info.items():
                report.append(f"\nФактор: {fac} (MSE={info['mse']:.4f}, df_resid={info['df_resid']})")
                for pair in info['pairs']:
                    signif = "YES" if pair['signif'] else "no"
                    report.append(f" {pair['level_i']} vs {pair['level_j']}: mean diff={pair['diff']:.3f}, LSD={pair['lsd']:.3f}, signif={signif}")

            if tukey_results:
                report.append("\n--- Tukey HSD (перший фактор) ---")
                report.append(tukey_results)

            # Write to result box
            self.result_box.delete(1.0, tk.END)
            self.result_box.insert(tk.END, "\n".join(report))
            messagebox.showinfo("Готово", "Аналіз завершено. Результат внизу.")
        except Exception as e:
            messagebox.showerror("Помилка", f"Сталася помилка при розрахунку:\n{e}")

    def save_report(self):
        txt = self.result_box.get(1.0, tk.END)
        if not txt.strip():
            messagebox.showwarning("Збереження", "Немає звіту для збереження")
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(txt)
            messagebox.showinfo("Збережено", f"Звіт збережено: {path}")


if __name__ == "__main__":
    SADApp()
