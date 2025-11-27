# -*- coding: utf-8 -*-
"""
SAD — Статистичний Аналіз Даних (одно-, дво-, трифакторний ANOVA)
Виконує: ANOVA (ручні розрахунки), Shapiro-Wilk, LSD, частки впливу, згрупування середніх буквами
Ввід: перші N стовпців — фактори (1..3), решта — повторності (числові)
Вставка даних: Ctrl+V (Excel) або ручний ввід у таблицю
Автор: Чаплоуцький А. М., Уманський національний університет
2025
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext, simpledialog
import pandas as pd
import numpy as np
from scipy import stats
from math import sqrt
from datetime import date
import os

# ----------------- Допоміжні статистичні функції -----------------
def safe_unique_levels(series):
    # повертає список унікальних рівнів у стабільному порядку
    return list(pd.Categorical(series).categories)

def compute_one_way_anova(long_df, factor):
    # groups by factor, assumes long_df has columns: factor(s), 'value'
    groups = [grp['value'].values for _, grp in long_df.groupby(factor)]
    a = len(groups)
    ns = [len(g) for g in groups]
    N = sum(ns)
    grand = np.mean(np.concatenate(groups))
    ss_total = np.sum((np.concatenate(groups) - grand)**2)
    means = [np.mean(g) for g in groups]
    ss_between = sum(n * (m - grand)**2 for n, m in zip(ns, means))
    ss_error = ss_total - ss_between
    df_between = a - 1
    df_error = N - a
    ms_between = ss_between / df_between if df_between>0 else np.nan
    ms_error = ss_error / df_error if df_error>0 else np.nan
    F = ms_between / ms_error if ms_error>0 else np.nan
    p = 1 - stats.f.cdf(F, df_between, df_error) if not np.isnan(F) else np.nan
    return {
        'levels': safe_unique_levels(long_df[factor]),
        'ns': ns,
        'means': means,
        'N': N,
        'grand': grand,
        'ss_total': ss_total,
        'ss_between': ss_between,
        'ss_error': ss_error,
        'df_between': df_between,
        'df_error': df_error,
        'ms_between': ms_between,
        'ms_error': ms_error,
        'F': F,
        'p': p
    }

def group_means_letters(levels, means, LSD):
    # простий алгоритм призначення букв (sorted ascending)
    # levels: list of labels
    # means: list of corresponding means
    # повертає dict level->"букви"
    pairs = sorted(zip(levels, means), key=lambda x: x[1])  # ascending
    letters = []
    groups = {}
    current_letter_ord = ord('a')
    # each group keeps a set of levels that are not significantly different
    group_reps = []
    for lvl, m in pairs:
        placed = False
        for g in group_reps:
            # compare to representative (first element) mean
            rep_mean = g['rep_mean']
            if abs(m - rep_mean) <= LSD + 1e-12:
                g['levels'].append((lvl, m))
                g['rep_mean'] = np.mean([x[1] for x in g['levels']])
                placed = True
                break
        if not placed:
            group_reps.append({'rep_mean': m, 'levels': [(lvl, m)]})
    # assign letters: first group -> 'a', next -> 'b' etc.
    result = {}
    for i, g in enumerate(group_reps):
        letter = chr(current_letter_ord + i)
        for lvl, _ in g['levels']:
            result[lvl] = letter
    return result

def balanced_shape_info(long_df, factor_cols):
    # determine a,b,c and r for balanced design
    # returns dict with level counts and r if balanced else None
    # group by all factor_cols and count
    gb = long_df.groupby(factor_cols).size()
    counts = gb.values
    if len(counts)==0:
        return None
    if np.all(counts == counts[0]):
        r = int(counts[0])
        levels = [safe_unique_levels(long_df[c]) for c in factor_cols]
        sizes = [len(l) for l in levels]
        return {'r': r, 'levels': levels, 'sizes': sizes}
    else:
        return None

# ----------------- ANOVA для 1,2,3 факторів (balanced) -----------------
def anova_1factor(long_df, A):
    res = compute_one_way_anova(long_df, A)
    # residuals
    expanded = long_df.copy()
    # fill group means
    means = {lvl: np.mean(g['value'].values) for lvl, g in long_df.groupby(A)}
    expanded['pred'] = expanded[A].map(means)
    residuals = expanded['value'] - expanded['pred']
    return res, residuals

def anova_2factor(long_df, A, B):
    # balanced required
    info = balanced_shape_info(long_df, [A,B])
    if info is None:
        raise ValueError("Дані не збалансовані для двофакторного аналізу.")
    a = len(info['levels'][0])
    b = len(info['levels'][1])
    r = info['r']
    grand = long_df['value'].mean()
    # cell means
    cell_means = long_df.groupby([A,B])['value'].mean().unstack()
    A_means = long_df.groupby(A)['value'].mean().values
    B_means = long_df.groupby(B)['value'].mean().values
    # SS
    ss_total = np.sum((long_df['value'] - grand)**2)
    ss_A = b * r * sum((m - grand)**2 for m in A_means)
    ss_B = a * r * sum((m - grand)**2 for m in B_means)
    # SS_AB
    # convert to matrix of cell means aligned
    # create arrays
    A_levels = safe_unique_levels(long_df[A])
    B_levels = safe_unique_levels(long_df[B])
    cell_mean_matrix = np.zeros((a,b))
    for i, ai in enumerate(A_levels):
        for j, bj in enumerate(B_levels):
            cell_mean_matrix[i,j] = cell_means.loc[ai, bj]
    SS_AB = r * np.sum((cell_mean_matrix - A_means[:,None] - B_means[None,:] + grand)**2)
    # SS_error
    # sum (y_ijk - cell_mean_ij)^2
    ss_error = 0.0
    for (ai,bj), grp in long_df.groupby([A,B]):
        cm = cell_means.loc[ai,bj]
        ss_error += np.sum((grp['value'].values - cm)**2)
    df_A = a-1
    df_B = b-1
    df_AB = (a-1)*(b-1)
    df_error = a*b*(r-1)
    ms_A = ss_A/df_A
    ms_B = ss_B/df_B
    ms_AB = SS_AB/df_AB if df_AB>0 else np.nan
    ms_error = ss_error/df_error if df_error>0 else np.nan
    F_A = ms_A/ms_error
    F_B = ms_B/ms_error
    F_AB = ms_AB/ms_error if not np.isnan(ms_AB) else np.nan
    p_A = 1 - stats.f.cdf(F_A, df_A, df_error)
    p_B = 1 - stats.f.cdf(F_B, df_B, df_error)
    p_AB = 1 - stats.f.cdf(F_AB, df_AB, df_error) if not np.isnan(F_AB) else np.nan
    # residuals
    preds = []
    for _, row in long_df.iterrows():
        preds.append(cell_means.loc[row[A], row[B]])
    residuals = long_df['value'].values - np.array(preds)
    return {
        'a':a,'b':b,'r':r,'grand':grand,
        'ss_total':ss_total,'ss_A':ss_A,'ss_B':ss_B,'ss_AB':SS_AB,'ss_error':ss_error,
        'df_A':df_A,'df_B':df_B,'df_AB':df_AB,'df_error':df_error,
        'ms_A':ms_A,'ms_B':ms_B,'ms_AB':ms_AB,'ms_error':ms_error,
        'F_A':F_A,'F_B':F_B,'F_AB':F_AB,'p_A':p_A,'p_B':p_B,'p_AB':p_AB,
        'A_levels': safe_unique_levels(long_df[A]), 'B_levels': safe_unique_levels(long_df[B])
    }, residuals

def anova_3factor(long_df, A, B, C):
    info = balanced_shape_info(long_df, [A,B,C])
    if info is None:
        raise ValueError("Дані не збалансовані для трифакторного аналізу.")
    a = len(info['levels'][0])
    b = len(info['levels'][1])
    c = len(info['levels'][2])
    r = info['r']
    grand = long_df['value'].mean()
    A_levels = safe_unique_levels(long_df[A])
    B_levels = safe_unique_levels(long_df[B])
    C_levels = safe_unique_levels(long_df[C])
    # compute marginal means
    mean_A = long_df.groupby(A)['value'].mean().values
    mean_B = long_df.groupby(B)['value'].mean().values
    mean_C = long_df.groupby(C)['value'].mean().values
    # cell means
    cm_abc = long_df.groupby([A,B,C])['value'].mean()
    # build arrays
    # SS total
    ss_total = np.sum((long_df['value'] - grand)**2)
    # SS factors
    ss_A = b*c*r * np.sum((mean_A - grand)**2)
    ss_B = a*c*r * np.sum((mean_B - grand)**2)
    ss_C = a*b*r * np.sum((mean_C - grand)**2)
    # SS two-way interactions
    # AB: average over C
    # build cell means matrix for AB averaging over C
    # compute AB means
    mean_AB = long_df.groupby([A,B])['value'].mean().unstack()
    mean_AC = long_df.groupby([A,C])['value'].mean().unstack()
    mean_BC = long_df.groupby([B,C])['value'].mean().unstack()
    # compute SS_AB
    # create arrays for cell means across abc and compute sums
    # create 3D array for cell means shape (a,b,c)
    cell = np.zeros((a,b,c))
    for i, ai in enumerate(A_levels):
        for j, bj in enumerate(B_levels):
            for k, ck in enumerate(C_levels):
                cell[i,j,k] = cm_abc.loc[ai,bj,ck]
    # compute marginal arrays
    A_means_arr = np.mean(cell, axis=(1,2))
    B_means_arr = np.mean(cell, axis=(0,2))
    C_means_arr = np.mean(cell, axis=(0,1))
    # SS_AB = c * r * sum((mean_AB - mean_A - mean_B + grand)^2)
    # compute mean_AB matrix
    mean_AB_mat = np.mean(cell, axis=2)
    SS_AB = c * r * np.sum((mean_AB_mat - A_means_arr[:,None] - B_means_arr[None,:] + grand)**2)
    mean_AC_mat = np.mean(cell, axis=1)
    SS_AC = b * r * np.sum((mean_AC_mat - A_means_arr[:,None] - C_means_arr[None,:] + grand)**2)
    mean_BC_mat = np.mean(cell, axis=0)
    SS_BC = a * r * np.sum((mean_BC_mat - B_means_arr[:,None] - C_means_arr[None,:] + grand)**2)
    # SS_ABC
    SS_ABC = r * np.sum((cell - A_means_arr[:,None,None] - B_means_arr[None,:,None] - C_means_arr[None,None,:]
                         + mean_AB_mat[:,:,None] + mean_AC_mat[:,None,:] + mean_BC_mat[None,:,:] - grand)**2)
    # SS_error: sum over (y - cell_mean)^2
    ss_error = 0.0
    for (ai,bj,ck), grp in long_df.groupby([A,B,C]):
        cm = cm_abc.loc[ai,bj,ck]
        ss_error += np.sum((grp['value'].values - cm)**2)
    # dfs
    df_A = a-1
    df_B = b-1
    df_C = c-1
    df_AB = (a-1)*(b-1)
    df_AC = (a-1)*(c-1)
    df_BC = (b-1)*(c-1)
    df_ABC = (a-1)*(b-1)*(c-1)
    df_error = a*b*c*(r-1)
    # MS
    ms_A = ss_A/df_A
    ms_B = ss_B/df_B
    ms_C = ss_C/df_C
    ms_AB = SS_AB/df_AB if df_AB>0 else np.nan
    ms_AC = SS_AC/df_AC if df_AC>0 else np.nan
    ms_BC = SS_BC/df_BC if df_BC>0 else np.nan
    ms_ABC = SS_ABC/df_ABC if df_ABC>0 else np.nan
    ms_error = ss_error/df_error if df_error>0 else np.nan
    F_A = ms_A/ms_error
    F_B = ms_B/ms_error
    F_C = ms_C/ms_error
    F_AB = ms_AB/ms_error if not np.isnan(ms_AB) else np.nan
    F_AC = ms_AC/ms_error if not np.isnan(ms_AC) else np.nan
    F_BC = ms_BC/ms_error if not np.isnan(ms_BC) else np.nan
    F_ABC = ms_ABC/ms_error if not np.isnan(ms_ABC) else np.nan
    # p-values
    p_A = 1 - stats.f.cdf(F_A, df_A, df_error)
    p_B = 1 - stats.f.cdf(F_B, df_B, df_error)
    p_C = 1 - stats.f.cdf(F_C, df_C, df_error)
    p_AB = 1 - stats.f.cdf(F_AB, df_AB, df_error) if not np.isnan(F_AB) else np.nan
    p_AC = 1 - stats.f.cdf(F_AC, df_AC, df_error) if not np.isnan(F_AC) else np.nan
    p_BC = 1 - stats.f.cdf(F_BC, df_BC, df_error) if not np.isnan(F_BC) else np.nan
    p_ABC = 1 - stats.f.cdf(F_ABC, df_ABC, df_error) if not np.isnan(F_ABC) else np.nan
    residuals = []
    for _, row in long_df.iterrows():
        residuals.append(cell[ A_levels.index(row[A]), B_levels.index(row[B]), C_levels.index(row[C]) ])
    residuals = long_df['value'].values - np.array(residuals)
    return {
        'a':a,'b':b,'c':c,'r':r,'grand':grand,
        'ss_total':ss_total,'ss_A':ss_A,'ss_B':ss_B,'ss_C':ss_C,
        'ss_AB':SS_AB,'ss_AC':SS_AC,'ss_BC':SS_BC,'ss_ABC':SS_ABC,'ss_error':ss_error,
        'df_A':df_A,'df_B':df_B,'df_C':df_C,'df_AB':df_AB,'df_AC':df_AC,'df_BC':df_BC,'df_ABC':df_ABC,'df_error':df_error,
        'ms_A':ms_A,'ms_B':ms_B,'ms_C':ms_C,'ms_AB':ms_AB,'ms_AC':ms_AC,'ms_BC':ms_BC,'ms_ABC':ms_ABC,'ms_error':ms_error,
        'F_A':F_A,'F_B':F_B,'F_C':F_C,'F_AB':F_AB,'F_AC':F_AC,'F_BC':F_BC,'F_ABC':F_ABC,
        'p_A':p_A,'p_B':p_B,'p_C':p_C,'p_AB':p_AB,'p_AC':p_AC,'p_BC':p_BC,'p_ABC':p_ABC,
        'A_levels':A_levels,'B_levels':B_levels,'C_levels':C_levels
    }, residuals

# ----------------- GUI та інтерфейс -----------------
class EditableTreeviewSimple(ttk.Treeview):
    # lightweight editable treeview: double click -> edit cell
    def __init__(self, master=None, **kw):
        super().__init__(master, **kw)
        self.bind('<Double-1>', self._on_double)
        self._entry = None
        self._cur = None

    def _on_double(self, event):
        region = self.identify('region', event.x, event.y)
        if region != 'cell':
            return
        row = self.identify_row(event.y)
        col = self.identify_column(event.x)
        x,y,w,h = self.bbox(row, col)
        val = self.item(row,'values')[int(col[1:])-1]
        self._entry = tk.Entry(self)
        self._entry.insert(0, str(val))
        self._entry.place(x=x, y=y, width=w, height=h)
        self._cur = (row, col)
        self._entry.focus()
        self._entry.bind('<Return>', self._save)
        self._entry.bind('<FocusOut>', self._save)
    def _save(self, event=None):
        if not self._entry or not self._cur:
            return
        row, col = self._cur
        vals = list(self.item(row,'values'))
        vals[int(col[1:])-1] = self._entry.get()
        self.item(row, values=vals)
        try: self._entry.destroy()
        except: pass
        self._entry = None
        self._cur = None

class SADApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SAD — Статистичний Аналіз Даних")
        self.root.geometry("1200x700")
        if os.path.exists("icon.ico"):
            try: self.root.iconbitmap("icon.ico")
            except: pass
        tk.Label(self.root, text="SAD — Статистичний Аналіз Даних", font=("Arial",16,"bold")).pack(pady=8)
        tk.Button(self.root, text="Почати аналіз", command=self.choose_factor_count, width=20, height=2).pack(pady=6)
        tk.Label(self.root, text="Введіть дані вручну або вставте з Excel (Ctrl+V). Перші N стовпців — фактори, решта — повторності.",
                 wraplength=1000, fg="gray").pack(pady=4)
        self.root.mainloop()

    def choose_factor_count(self):
        fc = simpledialog.askinteger("Кількість факторів", "Введіть кількість факторів (1, 2 або 3):", minvalue=1, maxvalue=3)
        if fc is None:
            return
        self.factor_count = fc
        self.open_table_window()

    def open_table_window(self):
        self.win = tk.Toplevel(self.root)
        self.win.title(f"SAD — Введення даних ({self.factor_count}-факторний)")
        self.win.geometry("1400x800")
        top = tk.Frame(self.win)
        top.pack(fill='x', padx=8, pady=6)
        tk.Button(top, text="З Excel", command=self.load_excel).pack(side='left', padx=4)
        tk.Button(top, text="Очистити", command=self.clear_table).pack(side='left', padx=4)
        tk.Button(top, text="Додати рядок", command=self.add_row).pack(side='left', padx=4)
        tk.Button(top, text="Додати стовпець", command=self.add_col).pack(side='left', padx=4)
        tk.Button(top, text="АНАЛІЗ", command=self.run_analysis, bg="#d32f2f", fg="white").pack(side='left', padx=20)
        mid = tk.Frame(self.win')
        # create tree
        cols = [f"c{i}" for i in range(1,21)]
        self.tree = EditableTreeviewSimple(mid, columns=cols, show='headings', height=18)
        for i,c in enumerate(cols):
            self.tree.heading(c, text=str(i+1))
            self.tree.column(c, width=100, anchor='center')
        self.tree.pack(side='left', fill='both', expand=True)
        vs = ttk.Scrollbar(mid, orient='vertical', command=self.tree.yview); vs.pack(side='right', fill='y')
        hs = ttk.Scrollbar(self.win, orient='horizontal', command=self.tree.xview); hs.pack(side='bottom', fill='x')
        self.tree.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        mid.pack(fill='both', expand=True, padx=8, pady=4)
        # fill with 10 rows
        for _ in range(10):
            self.tree.insert('', 'end', values=['']*20)
        # results
        self.result_box = scrolledtext.ScrolledText(self.win, height=18, font=("Consolas",10))
        self.result_box.pack(fill='both', expand=True, padx=8, pady=8)
        # bind paste
        self.win.bind_all("<Control-v>", lambda e: self.paste_from_clipboard())

    def add_row(self):
        self.tree.insert('', 'end', values=['']*len(self.tree['columns'])

    def add_col(self):
        cols = list(self.tree['columns'])
        new = f"c{len(cols)+1}"
        cols.append(new)
        self.tree['columns'] = cols
        self.tree.heading(new, text=str(len(cols)))
        self.tree.column(new, width=100, anchor='center')
        for iid in self.tree.get_children():
            vals = list(self.tree.item(iid,'values')); vals.append(''); self.tree.item(iid, values=vals)

    def clear_table(self):
        for iid in self.tree.get_children(): self.tree.delete(iid)
        for _ in range(10): self.tree.insert('', 'end', values=['']*len(self.tree['columns']))

    def paste_from_clipboard(self):
        try:
            df = pd.read_clipboard(header=None, dtype=str)
        except Exception:
            messagebox.showwarning("Помилка", "Не вдалося зчитати буфер")
            return
        # ensure enough columns
        ncols = df.shape[1]
        while len(self.tree['columns']) < ncols:
            self.add_col()
        # insert rows
        for _, row in df.iterrows():
            vals = [str(x).strip() for x in row.tolist()]
            vals += ['']*(len(self.tree['columns'])-len(vals))
            self.tree.insert('', 'end', values=vals[:len(self.tree['columns'])])
        messagebox.showinfo("Вставка", f"Вставлено {len(df)} рядків")

    def load_excel(self):
        path = filedialog.askopenfilename(filetypes=[("Excel files","*.xlsx *.xls")])
        if not path: return
        try:
            df = pd.read_excel(path, header=None, dtype=str).fillna('')
        except Exception as e:
            messagebox.showerror("Помилка", str(e)); return
        ncols = df.shape[1]
        while len(self.tree['columns']) < ncols:
            self.add_col()
        self.clear_table()
        for _, row in df.iterrows():
            vals = [str(x).strip() for x in row.tolist()]; vals += ['']*(len(self.tree['columns'])-len(vals))
            self.tree.insert('', 'end', values=vals[:len(self.tree['columns'])])
        messagebox.showinfo("Імпорт", f"Імпортовано {len(df)} рядків")

    def tree_to_df(self):
        cols = len(self.tree['columns'])
        rows = []
        for iid in self.tree.get_children():
            vals = list(self.tree.item(iid,'values'))
            vals += ['']*(cols - len(vals))
            rows.append([v.strip() for v in vals[:cols]])
        if len(rows)==0:
            return pd.DataFrame()
        colnames = [f"col{i+1}" for i in range(cols)]
        return pd.DataFrame(rows, columns=colnames)

    def wide_to_long(self, df_wide, n_factor_cols):
        # first n_factor_cols are factors, rest are repeats
        if df_wide.empty:
            return pd.DataFrame()
        factor_cols = df_wide.columns[:n_factor_cols]
        value_cols = df_wide.columns[n_factor_cols:]
        if len(value_cols)==0:
            # maybe user pasted long format: last column is value
            if n_factor_cols+1 == df_wide.shape[1]:
                # assume last column is value
                long = df_wide.copy()
                long.columns = list(factor_cols) + ['value']
                long['value'] = pd.to_numeric(long['value'], errors='coerce')
                long = long.dropna(subset=['value'])
                return long
            else:
                return pd.DataFrame()
        wide_vals = df_wide[factor_cols.tolist() + value_cols.tolist()].copy()
        # coerce numeric for value cols
        for c in value_cols:
            wide_vals[c] = pd.to_numeric(wide_vals[c], errors='coerce')
        long = pd.melt(wide_vals, id_vars=factor_cols.tolist(), value_vars=value_cols.tolist(),
                       var_name='repeat', value_name='value')
        long = long.dropna(subset=['value']).reset_index(drop=True)
        # cast factor cols to str categories
        for c in factor_cols:
            long[c] = long[c].astype(str)
        return long

    def run_analysis(self):
        try:
            df_wide = self.tree_to_df()
            if df_wide.empty:
                messagebox.showerror("Помилка", "Таблиця порожня"); return
            n = self.factor_count
            long = self.wide_to_long(df_wide, n)
            if long.empty:
                messagebox.showerror("Помилка", "Не вдалося структурувати дані. Переконайся, що перші N стовпців — фактори, решта — повторності або довга таблиця у форматі (фактори..., value)."); return
            # balanced check
            info = balanced_shape_info(long, list(long.columns[:n]))
            if info is None:
                messagebox.showwarning("УВАГА", "Дані не збалансовані (різна кількість повторностей в комбінаціях). Рекомендується використовувати збалансований план.")
            # compute according to n
            report_lines = []
            title = {1:"ОДНОФАКТОРНИЙ ДИСПЕРСІЙНИЙ АНАЛІЗ", 2:"ДВОХФАКТОРНИЙ ДИСПЕРСІЙНИЙ АНАЛІЗ", 3:"ТРИФАКТОРНИЙ ДИСПЕРСІЙНИЙ АНАЛІЗ"}[n]
            report_lines.append(f"Р Е З У Л Ь Т А Т И   {title}")
            report_lines.append("")
            # header with factor descriptions
            for i in range(n):
                lvl_names = safe_unique_levels(long[long.columns[i]])
                report_lines.append(f"Фактор {chr(ord('A')+i)}: {long.columns[i]} ({len(lvl_names)} рівні)")
            r = info['r'] if info is not None else long.groupby(list(long.columns[:n])).size().min()
            report_lines.append(f"Кількість повторень: {r}")
            report_lines.append("")
            # Shapiro on residuals will be computed after model
            if n==1:
                A = long.columns[0]
                res, residuals = anova_1factor(long, A)
                sw_w, sw_p = stats.shapiro(residuals) if len(residuals)>=3 else (np.nan, np.nan)
                report_lines.append(f"Перевірка нормальності залишків (Shapiro-Wilk): {'нормальний' if sw_p>0.05 else 'НЕ нормальний' if not np.isnan(sw_p) else 'даних замало'} (p = {sw_p:.3f})")
                report_lines.append("")
                # table header
                report_lines.append("────────────────────────────────────────────────────────────────────")
                report_lines.append(f"{'Джерело варіації':35s} {'Сума квадратів':>15s} {'df':>6s} {'Середній квадрат':>17s} {'Fрозрахункове':>14s} {'Fтабличне':>11s}   {'Висновок'}")
                report_lines.append("────────────────────────────────────────────────────────────────────")
                ss_total = res['ss_total']; ss_A = res['ss_between']; ss_err = res['ss_error']
                df_A = res['df_between']; df_err = res['df_error']
                ms_A = res['ms_between']; ms_err = res['ms_error']
                F = res['F']; p = res['p']
                # F critical
                F_tab = stats.f.ppf(1-0.05, df_A, df_err) if df_err>0 else np.nan
                concl = "істотний" if p<0.05 else "неістотний"
                stars = "**" if p<0.01 else "*" if p<0.05 else ""
                report_lines.append(f"{'Фактор A':35s} {ss_A:15.2f} {df_A:6d} {ms_A:17.3f} {F:14.2f}{stars:2s} {F_tab:11.2f}   {concl}")
                report_lines.append(f"{'Випадкова помилка':35s} {ss_err:15.2f} {df_err:6d} {ms_err:17.3f}")
                report_lines.append(f"{'Загальна':35s} {ss_total:15.2f} {res['N']-1:6d}")
                report_lines.append("")
                # percent contributions
                eta_A = ss_A/ss_total*100 if ss_total>0 else 0
                eta_err = ss_err/ss_total*100 if ss_total>0 else 0
                report_lines.append("Вилучення впливу:")
                report_lines.append(f"  • Фактор A                 — {eta_A:.1f}%")
                report_lines.append(f"  • Залишок                  — {eta_err:.1f}%")
                report_lines.append("")
                # LSD
                t_val = stats.t.ppf(1-0.975, df_err) if df_err>0 else np.nan
                # per-group n mean size
                n_mean = np.mean(res['ns'])
                LSD = t_val * sqrt(2*ms_err / n_mean) if df_err>0 else np.nan
                report_lines.append("НІР₀.₅:")
                report_lines.append(f"  • По фактору A              — {LSD:.2f}")
                report_lines.append("")
                # means with letters
                levs = res['levels']; means = res['means']
                letters = group_means_letters(levs, means, LSD if not np.isnan(LSD) else 0)
                report_lines.append("Середні по фактору A:")
                for lvl, m in zip(levs, means):
                    report_lines.append(f"  {str(lvl):20s} {m:6.2f} {letters.get(lvl,'')}")
                report_lines.append("")
            elif n==2:
                A = long.columns[0]; B = long.columns[1]
                stats_res, residuals = anova_2factor(long, A, B)
                sw_w, sw_p = stats.shapiro(residuals) if len(residuals)>=3 else (np.nan, np.nan)
                report_lines.append(f"Перевірка нормальності залишків (Shapiro-Wilk): {'нормальний' if sw_p>0.05 else 'НЕ нормальний' if not np.isnan(sw_p) else 'даних замало'} (p = {sw_p:.3f})")
                report_lines.append("")
                report_lines.append("────────────────────────────────────────────────────────────────────")
                report_lines.append(f"{'Джерело варіації':35s} {'Сума квадратів':>15s} {'df':>6s} {'Середній квадрат':>17s} {'Fрозрахункове':>14s} {'Fтабличне':>11s}   {'Висновок'}")
                report_lines.append("────────────────────────────────────────────────────────────────────")
                ss_total = stats_res['ss_total']
                # rows
                rows = [
                    ("Фактор A", stats_res['ss_A'], stats_res['df_A'], stats_res['ms_A'], stats_res['F_A'], stats.f.ppf(1-0.05, stats_res['df_A'], stats_res['df_error']), stats_res['p_A']),
                    ("Фактор B", stats_res['ss_B'], stats_res['df_B'], stats_res['ms_B'], stats_res['F_B'], stats.f.ppf(1-0.05, stats_res['df_B'], stats_res['df_error']), stats_res['p_B']),
                    ("Взаємодія A×B", stats_res['ss_AB'], stats_res['df_AB'], stats_res['ms_AB'], stats_res['F_AB'], stats.f.ppf(1-0.05, stats_res['df_AB'], stats_res['df_error']), stats_res['p_AB']),
                    ("Випадкова помилка", stats_res['ss_error'], stats_res['df_error'], stats_res['ms_error'], None, None, None),
                ]
                for name, ss, dfv, ms, Fv, Ftab, pv in rows:
                    if Fv is None or np.isnan(Fv):
                        report_lines.append(f"{name:35s} {ss:15.2f} {int(dfv):6d} {ms:17.3f}")
                    else:
                        stars = "**" if pv<0.01 else "*" if pv<0.05 else ""
                        concl = "істотний" if pv<0.05 else "неістотний"
                        report_lines.append(f"{name:35s} {ss:15.2f} {int(dfv):6d} {ms:17.3f} {Fv:14.2f}{stars:2s} {Ftab:11.2f}   {concl}")
                report_lines.append(f"{'Загальна':35s} {ss_total:15.2f} {int(stats_res['df_A']+stats_res['df_B']+stats_res['df_AB']+stats_res['df_error']):6d}")
                report_lines.append("")
                # contributions
                comps = {
                    'A': stats_res['ss_A'],
                    'B': stats_res['ss_B'],
                    'A×B': stats_res['ss_AB'],
                    'Error': stats_res['ss_error']
                }
                report_lines.append("Вилучення впливу:")
                for k,v in comps.items():
                    report_lines.append(f"  • {k:20s} — {v/ss_total*100:4.1f}%")
                report_lines.append("")
                # LSD per factor: for factor A, mean per A is b*r obs
                r = stats_res['r']; a = stats_res['a']; b = stats_res['b']
                ms_err = stats_res['ms_error']; df_err = stats_res['df_error']
                t_val = stats.t.ppf(1-0.975, df_err)
                LSD_A = t_val*sqrt(2*ms_err/(b*r))
                LSD_B = t_val*sqrt(2*ms_err/(a*r))
                # combinations LSD for cell comparisons:
                LSD_comb = t_val*sqrt(2*ms_err/r)
                report_lines.append("НІР₀.₅:")
                report_lines.append(f"  • По фактору A              — {LSD_A:.2f}")
                report_lines.append(f"  • По фактору B              — {LSD_B:.2f}")
                report_lines.append(f"  • По комбінаціям           — {LSD_comb:.2f}")
                report_lines.append("")
                # means with letters per factor using LSD
                Alev = stats_res['A_levels']; Blev = stats_res['B_levels']
                meanA = [np.mean(long[long[stats_res['A_levels'][0]]==lvl]['value'].values) for lvl in Alev] if False else None
                # simpler: compute means from long
                meanA = [long[long[ long.columns[0] ]==lvl]['value'].mean() for lvl in Alev]
                meanB = [long[long[ long.columns[1] ]==lvl]['value'].mean() for lvl in Blev]
                lettersA = group_means_letters(Alev, meanA, LSD_A)
                lettersB = group_means_letters(Blev, meanB, LSD_B)
                report_lines.append(f"Середні по фактору A ({long.columns[0]}):")
                for lvl, m in zip(Alev, meanA):
                    report_lines.append(f"  {str(lvl):20s} {m:6.2f} {lettersA.get(lvl,'')}")
                report_lines.append("")
                report_lines.append(f"Середні по фактору B ({long.columns[1]}):")
                for lvl, m in zip(Blev, meanB):
                    report_lines.append(f"  {str(lvl):20s} {m:6.2f} {lettersB.get(lvl,'')}")
                report_lines.append("")
            else:  # n == 3
                A = long.columns[0]; B = long.columns[1]; C = long.columns[2]
                stats_res, residuals = anova_3factor(long, A, B, C)
                sw_w, sw_p = stats.shapiro(residuals) if len(residuals)>=3 else (np.nan, np.nan)
                report_lines.append(f"Перевірка нормальності залишків (Shapiro-Wilk): {'нормальний' if sw_p>0.05 else 'НЕ нормальний' if not np.isnan(sw_p) else 'даних замало'} (p = {sw_p:.3f})")
                report_lines.append("")
                report_lines.append("────────────────────────────────────────────────────────────────────")
                report_lines.append(f"{'Джерело варіації':35s} {'Сума квадратів':>15s} {'df':>6s} {'Середній квадрат':>17s} {'Fрозрахункове':>14s} {'Fтабличне':>11s}   {'Висновок'}")
                report_lines.append("────────────────────────────────────────────────────────────────────")
                # prepare rows consistent with example
                rows = [
                    ("Фактор A (Сорт)", stats_res['ss_A'], stats_res['df_A'], stats_res['ms_A'], stats_res['F_A'], stats.f.ppf(1-0.05, stats_res['df_A'], stats_res['df_error']), stats_res['p_A']),
                    ("Фактор B (Добрива)", stats_res['ss_B'], stats_res['df_B'], stats_res['ms_B'], stats_res['F_B'], stats.f.ppf(1-0.05, stats_res['df_B'], stats_res['df_error']), stats_res['p_B']),
                    ("Фактор C (Зрошення)", stats_res['ss_C'], stats_res['df_C'], stats_res['ms_C'], stats_res['F_C'], stats.f.ppf(1-0.05, stats_res['df_C'], stats_res['df_error']), stats_res['p_C']),
                    ("Взаємодія A × B", stats_res['ss_AB'], stats_res['df_AB'], stats_res['ms_AB'], stats_res['F_AB'], stats.f.ppf(1-0.05, stats_res['df_AB'], stats_res['df_error']), stats_res['p_AB']),
                    ("Взаємодія A × C", stats_res['ss_AC'], stats_res['df_AC'], stats_res['ms_AC'], stats_res['F_AC'], stats.f.ppf(1-0.05, stats_res['df_AC'], stats_res['df_error']), stats_res['p_AC']),
                    ("Взаємодія B × C", stats_res['ss_BC'], stats_res['df_BC'], stats_res['ms_BC'], stats_res['F_BC'], stats.f.ppf(1-0.05, stats_res['df_BC'], stats_res['df_error']), stats_res['p_BC']),
                    ("Взаємодія A × B × C", stats_res['ss_ABC'], stats_res['df_ABC'], stats_res['ms_ABC'], stats_res['F_ABC'], stats.f.ppf(1-0.05, stats_res['df_ABC'], stats_res['df_error']), stats_res['p_ABC']),
                    ("Випадкова помилка", stats_res['ss_error'], stats_res['df_error'], stats_res['ms_error'], None, None, None)
                ]
                for name, ss, dfv, ms, Fv, Ftab, pv in rows:
                    if Fv is None or np.isnan(Fv):
                        report_lines.append(f"{name:35s} {ss:15.2f} {int(dfv):6d} {ms:17.3f}")
                    else:
                        stars = "**" if pv<0.01 else "*" if pv<0.05 else ""
                        concl = "істотний" if pv<0.05 else "неістотний"
                        report_lines.append(f"{name:35s} {ss:15.2f} {int(dfv):6d} {ms:17.3f} {Fv:14.2f}{stars:2s} {Ftab:11.2f}   {concl}")
                report_lines.append(f"{'Загальна':35s} {stats_res['ss_total']:15.2f} {int(stats_res['df_error']+stats_res['df_A']+stats_res['df_B']+stats_res['df_C']+stats_res['df_AB']+stats_res['df_AC']+stats_res['df_BC']+stats_res['df_ABC']):6d}")
                report_lines.append("")
                # percent contributions
                ss_total = stats_res['ss_total']
                parts = {
                    'Фактор A':stats_res['ss_A'],
                    'Фактор B':stats_res['ss_B'],
                    'Фактор C':stats_res['ss_C'],
                    'A×B':stats_res['ss_AB'],
                    'A×C':stats_res['ss_AC'],
                    'B×C':stats_res['ss_BC'],
                    'A×B×C':stats_res['ss_ABC'],
                    'Залишок':stats_res['ss_error']
                }
                report_lines.append("Вилучення впливу:")
                for k,v in parts.items():
                    report_lines.append(f"  • {k:25s} — {v/ss_total*100:4.1f}%")
                report_lines.append("")
                # LSD
                df_err = stats_res['df_error']; ms_err = stats_res['ms_error']
                t_val = stats.t.ppf(1-0.975, df_err) if df_err>0 else np.nan
                a,b,c,r = stats_res['a'], stats_res['b'], stats_res['c'], stats_res['r']
                LSD_A = t_val*sqrt(2*ms_err/(b*c*r)) if df_err>0 else np.nan
                LSD_B = t_val*sqrt(2*ms_err/(a*c*r)) if df_err>0 else np.nan
                LSD_C = t_val*sqrt(2*ms_err/(a*b*r)) if df_err>0 else np.nan
                LSD_comb = t_val*sqrt(2*ms_err/r) if df_err>0 else np.nan
                report_lines.append("НІР₀.₅:")
                report_lines.append(f"  • По фактору A                    — {LSD_A:.2f}")
                report_lines.append(f"  • По фактору B                    — {LSD_B:.2f}")
                report_lines.append(f"  • По фактору C                    — {LSD_C:.2f}")
                report_lines.append(f"  • По комбінаціях                  — {LSD_comb:.2f}")
                report_lines.append("")
                # means per factor and letters
                Alev = stats_res['A_levels']; Blev = stats_res['B_levels']; Clev = stats_res['C_levels']
                meanA = [long[long[A]==lvl]['value'].mean() for lvl in Alev]
                meanB = [long[long[B]==lvl]['value'].mean() for lvl in Blev]
                meanC = [long[long[C]==lvl]['value'].mean() for lvl in Clev]
                lettersA = group_means_letters(Alev, meanA, LSD_A if not np.isnan(LSD_A) else 0)
                lettersB = group_means_letters(Blev, meanB, LSD_B if not np.isnan(LSD_B) else 0)
                lettersC = group_means_letters(Clev, meanC, LSD_C if not np.isnan(LSD_C) else 0)
                report_lines.append("Середні по фактору A (сорт):")
                for lvl, m in zip(Alev, meanA):
                    report_lines.append(f"  {str(lvl):18s} {m:6.1f} {lettersA.get(lvl,'')}")
                report_lines.append("")
                report_lines.append("Середні по фактору B (добрива):")
                for lvl, m in zip(Blev, meanB):
                    report_lines.append(f"  {str(lvl):18s} {m:6.1f} {lettersB.get(lvl,'')}")
                report_lines.append("")
                report_lines.append("Середні по фактору C (зрошення):")
                for lvl, m in zip(Clev, meanC):
                    report_lines.append(f"  {str(lvl):18s} {m:6.1f} {lettersC.get(lvl,'')}")
                report_lines.append("")
                # combinations: show some top combinations (full listing could be long)
                report_lines.append("Комбінації (з буквами істотності):")
                # compute cell means
                cell_means = long.groupby([A,B,C])['value'].mean()
                # sort by mean descending and show all
                sorted_cells = sorted(cell_means.items(), key=lambda x: -x[1])
                for (ai,bj,ck), val in sorted_cells:
                    # compose letter by concatenating factor letters
                    la = lettersA.get(ai,'')
                    lb = lettersB.get(bj,'')
                    lc = lettersC.get(ck,'')
                    report_lines.append(f"  {ai:12s} | {bj:14s} | {ck:14s} -> {val:6.1f} {la}{lb}{lc}")
                report_lines.append("")
            # output
            self.result_box.delete(1.0, tk.END)
            self.result_box.insert(tk.END, "\n".join(report_lines))
        except Exception as e:
            messagebox.showerror("Помилка", str(e))

# ----------------- Запуск -----------------
if __name__ == "__main__":
    SADApp()
