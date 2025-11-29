# main.py
# -*- coding: utf-8 -*-
"""
SAD - Statистичний Аналіз Даних (Tkinter версія)
Автор: адаптація під вимоги користувача
Потрібно: Python 3.8+, numpy, scipy
Встановлення: pip install numpy scipy
"""
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from tkinter.scrolledtext import ScrolledText
import math
import numpy as np
from scipy.stats import shapiro, t, f as f_dist

# -------------------------
# Допоміжні статистичні функції
# -------------------------
def significance_mark(p):
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""

def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan

# -------------------------
# ANOVA по маргінальним середнім (supports 1-,2-,3-way)
# Формат вхідних даних:
# long = list of dicts: {'A': levelA, 'B': levelB, 'C': levelC, 'value': float}
# Для 1-way: only 'A' and 'value' used
# Для 2-way: 'A','B','value'
# Для 3-way: 'A','B','C','value'
# -------------------------
def anova_n_way(long, factors):
    """
    long: list of dicts with factor keys and 'value'
    factors: list like ['A'] or ['A','B'] or ['A','B','C']
    returns: dict with ANOVA table strings and numeric results
    """
    # collect unique levels and counts
    N = len(long)
    values = np.array([rec['value'] for rec in long], dtype=float)
    grand_mean = np.nanmean(values)

    # create mapping for levels
    from collections import defaultdict
    # counts and cell means
    if len(factors) == 1:
        # One-way
        A = [rec[factors[0]] for rec in long]
        levels_A = sorted(list({x for x in A}))
        n_i = {}
        mean_i = {}
        for lev in levels_A:
            vals = [rec['value'] for rec in long if rec[factors[0]] == lev and not math.isnan(rec['value'])]
            n_i[lev] = len(vals)
            mean_i[lev] = np.nanmean(vals) if len(vals)>0 else np.nan

        # SS total
        SS_total = np.nansum((values - grand_mean)**2)
        # SS_between
        SS_A = sum(n_i[lev]*(mean_i[lev]-grand_mean)**2 for lev in levels_A if not math.isnan(mean_i[lev]))
        # SS_error: sum within groups
        SS_error = 0.0
        for lev in levels_A:
            vals = [rec['value'] for rec in long if rec[factors[0]] == lev and not math.isnan(rec['value'])]
            SS_error += sum((v - mean_i[lev])**2 for v in vals) if len(vals)>0 else 0.0

        df_A = len(levels_A)-1
        df_error = N - len(levels_A)
        df_total = N - 1

        MS_A = SS_A/df_A if df_A>0 else np.nan
        MS_error = SS_error/df_error if df_error>0 else np.nan
        F_A = MS_A / MS_error if MS_error and not math.isnan(MS_error) else np.nan
        p_A = 1 - f_dist.cdf(F_A, df_A, df_error) if not np.isnan(F_A) else np.nan

        # eta2
        eta2_A = SS_A / SS_total if SS_total>0 else np.nan

        # LSD (assume balanced per level n = mean n_i)
        mean_n = np.mean(list(n_i.values())) if len(n_i)>0 else np.nan
        tval = t.ppf(0.975, df_error) if df_error>0 else np.nan
        LSD = tval * math.sqrt(2*MS_error/mean_n) if not any(math.isnan(x) for x in [tval, MS_error, mean_n]) else np.nan

        table = []
        table.append(("Фактор "+factors[0], SS_A, df_A, MS_A, F_A, p_A))
        table.append(("Залишок", SS_error, df_error, MS_error, None, None))
        table.append(("Загальна", SS_total, df_total, None, None, None))

        return {
            'type': 'one',
            'table': table,
            'eta2': {'A': eta2_A},
            'LSD': {factors[0]: LSD},
            'means': {factors[0]: mean_i},
            'p_values': {'A': p_A}
        }

    elif len(factors) == 2:
        # Two-way: factors[0]=A, factors[1]=B
        Akey, Bkey = factors
        levels_A = sorted(list({rec[Akey] for rec in long}))
        levels_B = sorted(list({rec[Bkey] for rec in long}))
        # cell means and counts
        mean_cell = {}
        n_cell = {}
        for a in levels_A:
            for b in levels_B:
                vals = [rec['value'] for rec in long if rec[Akey]==a and rec[Bkey]==b and not math.isnan(rec['value'])]
                n_cell[(a,b)] = len(vals)
                mean_cell[(a,b)] = np.nanmean(vals) if len(vals)>0 else np.nan

        # marginal means
        mean_A = {}
        n_A = {}
        for a in levels_A:
            vals = [rec['value'] for rec in long if rec[Akey]==a and not math.isnan(rec['value'])]
            n_A[a] = len(vals)
            mean_A[a] = np.nanmean(vals) if len(vals)>0 else np.nan
        mean_B = {}
        n_B = {}
        for b in levels_B:
            vals = [rec['value'] for rec in long if rec[Bkey]==b and not math.isnan(rec['value'])]
            n_B[b] = len(vals)
            mean_B[b] = np.nanmean(vals) if len(vals)>0 else np.nan

        SS_total = np.nansum((values - grand_mean)**2)
        # SS_A
        SS_A = sum(n_A[a]*(mean_A[a]-grand_mean)**2 for a in levels_A if not math.isnan(mean_A[a]))
        # SS_B
        SS_B = sum(n_B[b]*(mean_B[b]-grand_mean)**2 for b in levels_B if not math.isnan(mean_B[b]))
        # SS_AB
        SS_AB = 0.0
        for a in levels_A:
            for b in levels_B:
                m_ab = mean_cell[(a,b)]
                if math.isnan(m_ab): continue
                m_a = mean_A[a]
                m_b = mean_B[b]
                SS_AB += n_cell[(a,b)] * (m_ab - m_a - m_b + grand_mean)**2

        # SS_error: within-cell variability
        SS_error = 0.0
        for a in levels_A:
            for b in levels_B:
                vals = [rec['value'] for rec in long if rec[Akey]==a and rec[Bkey]==b and not math.isnan(rec['value'])]
                m_ab = mean_cell[(a,b)]
                SS_error += sum((v - m_ab)**2 for v in vals) if len(vals)>0 and not math.isnan(m_ab) else 0.0

        a = len(levels_A); b = len(levels_B)
        df_A = a-1
        df_B = b-1
        df_AB = (a-1)*(b-1)
        df_error = N - a*b
        df_total = N - 1

        MS_A = SS_A/df_A if df_A>0 else np.nan
        MS_B = SS_B/df_B if df_B>0 else np.nan
        MS_AB = SS_AB/df_AB if df_AB>0 else np.nan
        MS_error = SS_error/df_error if df_error>0 else np.nan

        F_A = MS_A/MS_error if not math.isnan(MS_error) and MS_error>0 else np.nan
        F_B = MS_B/MS_error if not math.isnan(MS_error) and MS_error>0 else np.nan
        F_AB = MS_AB/MS_error if not math.isnan(MS_error) and MS_error>0 else np.nan

        pA = 1 - f_dist.cdf(F_A, df_A, df_error) if not np.isnan(F_A) else np.nan
        pB = 1 - f_dist.cdf(F_B, df_B, df_error) if not np.isnan(F_B) else np.nan
        pAB = 1 - f_dist.cdf(F_AB, df_AB, df_error) if not np.isnan(F_AB) else np.nan

        eta2_A = SS_A/SS_total if SS_total>0 else np.nan
        eta2_B = SS_B/SS_total if SS_total>0 else np.nan
        eta2_AB = SS_AB/SS_total if SS_total>0 else np.nan
        eta2_rem = 1 - (eta2_A + eta2_B + eta2_AB) if SS_total>0 else np.nan

        # LSD: assume balanced with r per cell average
        r_list = [n_cell[(a,b)] for a in levels_A for b in levels_B]
        r_mean = np.mean([x for x in r_list if x>0]) if any(x>0 for x in r_list) else np.nan
        tval = t.ppf(0.975, df_error) if df_error>0 else np.nan
        LSD_A = tval * math.sqrt(2*MS_error/(b*r_mean)) if not any(math.isnan(x) for x in [tval, MS_error, r_mean]) else np.nan
        LSD_B = tval * math.sqrt(2*MS_error/(a*r_mean)) if not any(math.isnan(x) for x in [tval, MS_error, r_mean]) else np.nan
        LSD_AB = tval * math.sqrt(2*MS_error/(r_mean)) if not any(math.isnan(x) for x in [tval, MS_error, r_mean]) else np.nan

        table = [
            (f"А ({Akey})", SS_A, df_A, MS_A, F_A, pA),
            (f"В ({Bkey})", SS_B, df_B, MS_B, F_B, pB),
            ("А×В", SS_AB, df_AB, MS_AB, F_AB, pAB),
            ("Залишок", SS_error, df_error, MS_error, None, None),
            ("Загальна", SS_total, df_total, None, None, None)
        ]

        return {
            'type': 'two',
            'table': table,
            'eta2': {'A': eta2_A, 'B': eta2_B, 'AB': eta2_AB, 'res': eta2_rem},
            'LSD': {'A': LSD_A, 'B': LSD_B, 'AB': LSD_AB},
            'means_A': mean_A,
            'means_B': mean_B,
            'cell_means': mean_cell,
            'p_values': {'A': pA, 'B': pB, 'AB': pAB}
        }

    elif len(factors) == 3:
        Akey, Bkey, Ckey = factors
        levels_A = sorted(list({rec[Akey] for rec in long}))
        levels_B = sorted(list({rec[Bkey] for rec in long}))
        levels_C = sorted(list({rec[Ckey] for rec in long}))

        # cell means and counts
        mean_cell = {}
        n_cell = {}
        for a in levels_A:
            for b in levels_B:
                for c in levels_C:
                    vals = [rec['value'] for rec in long if rec[Akey]==a and rec[Bkey]==b and rec[Ckey]==c and not math.isnan(rec['value'])]
                    n_cell[(a,b,c)] = len(vals)
                    mean_cell[(a,b,c)] = np.nanmean(vals) if len(vals)>0 else np.nan

        # marginal means
        def get_vals(cond):
            return [rec['value'] for rec in long if cond(rec) and not math.isnan(rec['value'])]

        mean_A = {}
        n_A = {}
        for a in levels_A:
            vals = get_vals(lambda r: r[Akey]==a)
            n_A[a] = len(vals)
            mean_A[a] = np.nanmean(vals) if len(vals)>0 else np.nan
        mean_B = {}
        n_B = {}
        for b in levels_B:
            vals = get_vals(lambda r: r[Bkey]==b)
            n_B[b] = len(vals)
            mean_B[b] = np.nanmean(vals) if len(vals)>0 else np.nan
        mean_C = {}
        n_C = {}
        for c in levels_C:
            vals = get_vals(lambda r: r[Ckey]==c)
            n_C[c] = len(vals)
            mean_C[c] = np.nanmean(vals) if len(vals)>0 else np.nan

        # two-way margins
        mean_AB = {}
        for a in levels_A:
            for b in levels_B:
                vals = get_vals(lambda r: r[Akey]==a and r[Bkey]==b)
                mean_AB[(a,b)] = np.nanmean(vals) if len(vals)>0 else np.nan
        mean_AC = {}
        for a in levels_A:
            for c in levels_C:
                vals = get_vals(lambda r: r[Akey]==a and r[Ckey]==c)
                mean_AC[(a,c)] = np.nanmean(vals) if len(vals)>0 else np.nan
        mean_BC = {}
        for b in levels_B:
            for c in levels_C:
                vals = get_vals(lambda r: r[Bkey]==b and r[Ckey]==c)
                mean_BC[(b,c)] = np.nanmean(vals) if len(vals)>0 else np.nan

        SS_total = np.nansum((values - grand_mean)**2)

        # SS main effects
        SS_A = sum(n_A[a]*(mean_A[a]-grand_mean)**2 for a in levels_A if not math.isnan(mean_A[a]))
        SS_B = sum(n_B[b]*(mean_B[b]-grand_mean)**2 for b in levels_B if not math.isnan(mean_B[b]))
        SS_C = sum(n_C[c]*(mean_C[c]-grand_mean)**2 for c in levels_C if not math.isnan(mean_C[c]))

        # SS two-way interactions (AB, AC, BC)
        SS_AB = 0.0
        for a in levels_A:
            for b in levels_B:
                # n_ab: count for AB marginal (summed over C)
                vals = get_vals(lambda r: r[Akey]==a and r[Bkey]==b)
                n_ab = len(vals)
                m_ab = mean_AB.get((a,b), np.nan)
                m_a = mean_A[a]
                m_b = mean_B[b]
                if not math.isnan(m_ab):
                    SS_AB += n_ab * (m_ab - m_a - m_b + grand_mean)**2

        SS_AC = 0.0
        for a in levels_A:
            for c in levels_C:
                vals = get_vals(lambda r: r[Akey]==a and r[Ckey]==c)
                n_ac = len(vals)
                m_ac = mean_AC.get((a,c), np.nan)
                m_a = mean_A[a]
                m_c = mean_C[c]
                if not math.isnan(m_ac):
                    SS_AC += n_ac * (m_ac - m_a - m_c + grand_mean)**2

        SS_BC = 0.0
        for b in levels_B:
            for c in levels_C:
                vals = get_vals(lambda r: r[Bkey]==b and r[Ckey]==c)
                n_bc = len(vals)
                m_bc = mean_BC.get((b,c), np.nan)
                m_b = mean_B[b]
                m_c = mean_C[c]
                if not math.isnan(m_bc):
                    SS_BC += n_bc * (m_bc - m_b - m_c + grand_mean)**2

        # SS three-way ABC
        SS_ABC = 0.0
        for a in levels_A:
            for b in levels_B:
                for c in levels_C:
                    n_abc = n_cell.get((a,b,c), 0)
                    m_abc = mean_cell.get((a,b,c), np.nan)
                    if math.isnan(m_abc): continue
                    term = (m_abc
                            - (mean_AB.get((a,b),0) if not math.isnan(mean_AB.get((a,b),np.nan)) else 0)
                            - (mean_AC.get((a,c),0) if not math.isnan(mean_AC.get((a,c),np.nan)) else 0)
                            - (mean_BC.get((b,c),0) if not math.isnan(mean_BC.get((b,c),np.nan)) else 0)
                            + (mean_A.get(a,0) if not math.isnan(mean_A.get(a,np.nan)) else 0)
                            + (mean_B.get(b,0) if not math.isnan(mean_B.get(b,np.nan)) else 0)
                            + (mean_C.get(c,0) if not math.isnan(mean_C.get(c,np.nan)) else 0)
                            - grand_mean)
                    SS_ABC += n_abc * (term**2)

        # SS_error within cells
        SS_error = 0.0
        for a in levels_A:
            for b in levels_B:
                for c in levels_C:
                    vals = get_vals(lambda r: r[Akey]==a and r[Bkey]==b and r[Ckey]==c)
                    m_abc = mean_cell.get((a,b,c), np.nan)
                    if not math.isnan(m_abc) and len(vals)>0:
                        SS_error += sum((v - m_abc)**2 for v in vals)

        a = len(levels_A); b = len(levels_B); c = len(levels_C)
        df_A = a-1; df_B = b-1; df_C = c-1
        df_AB = (a-1)*(b-1); df_AC = (a-1)*(c-1); df_BC = (b-1)*(c-1)
        df_ABC = (a-1)*(b-1)*(c-1)
        df_error = N - a*b*c
        df_total = N - 1

        MS_A = SS_A/df_A if df_A>0 else np.nan
        MS_B = SS_B/df_B if df_B>0 else np.nan
        MS_C = SS_C/df_C if df_C>0 else np.nan
        MS_AB = SS_AB/df_AB if df_AB>0 else np.nan
        MS_AC = SS_AC/df_AC if df_AC>0 else np.nan
        MS_BC = SS_BC/df_BC if df_BC>0 else np.nan
        MS_ABC = SS_ABC/df_ABC if df_ABC>0 else np.nan
        MS_error = SS_error/df_error if df_error>0 else np.nan

        F_A = MS_A/MS_error if not math.isnan(MS_error) and MS_error>0 else np.nan
        F_B = MS_B/MS_error if not math.isnan(MS_error) and MS_error>0 else np.nan
        F_C = MS_C/MS_error if not math.isnan(MS_error) and MS_error>0 else np.nan
        F_AB = MS_AB/MS_error if not math.isnan(MS_error) and MS_error>0 else np.nan
        F_AC = MS_AC/MS_error if not math.isnan(MS_error) and MS_error>0 else np.nan
        F_BC = MS_BC/MS_error if not math.isnan(MS_error) and MS_error>0 else np.nan
        F_ABC = MS_ABC/MS_error if not math.isnan(MS_error) and MS_error>0 else np.nan

        pA = 1 - f_dist.cdf(F_A, df_A, df_error) if not np.isnan(F_A) else np.nan
        pB = 1 - f_dist.cdf(F_B, df_B, df_error) if not np.isnan(F_B) else np.nan
        pC = 1 - f_dist.cdf(F_C, df_C, df_error) if not np.isnan(F_C) else np.nan
        pAB = 1 - f_dist.cdf(F_AB, df_AB, df_error) if not np.isnan(F_AB) else np.nan
        pAC = 1 - f_dist.cdf(F_AC, df_AC, df_error) if not np.isnan(F_AC) else np.nan
        pBC = 1 - f_dist.cdf(F_BC, df_BC, df_error) if not np.isnan(F_BC) else np.nan
        pABC = 1 - f_dist.cdf(F_ABC, df_ABC, df_error) if not np.isnan(F_ABC) else np.nan

        # eta2
        eta2_A = SS_A/SS_total if SS_total>0 else np.nan
        eta2_B = SS_B/SS_total if SS_total>0 else np.nan
        eta2_C = SS_C/SS_total if SS_total>0 else np.nan
        eta2_AB = SS_AB/SS_total if SS_total>0 else np.nan
        eta2_AC = SS_AC/SS_total if SS_total>0 else np.nan
        eta2_BC = SS_BC/SS_total if SS_total>0 else np.nan
        eta2_ABC = SS_ABC/SS_total if SS_total>0 else np.nan
        eta2_rem = 1 - (eta2_A+eta2_B+eta2_C+eta2_AB+eta2_AC+eta2_BC+eta2_ABC) if SS_total>0 else np.nan

        # LSD: assume r per cell mean
        r_list = [n_cell[(a,b,c)] for a in levels_A for b in levels_B for c in levels_C]
        r_mean = np.mean([x for x in r_list if x>0]) if any(x>0 for x in r_list) else np.nan
        tval = t.ppf(0.975, df_error) if df_error>0 else np.nan
        LSD_A = tval * math.sqrt(2*MS_error/(b*c*r_mean)) if not any(math.isnan(x) for x in [tval, MS_error, r_mean]) else np.nan
        LSD_B = tval * math.sqrt(2*MS_error/(a*c*r_mean)) if not any(math.isnan(x) for x in [tval, MS_error, r_mean]) else np.nan
        LSD_C = tval * math.sqrt(2*MS_error/(a*b*r_mean)) if not any(math.isnan(x) for x in [tval, MS_error, r_mean]) else np.nan
        LSD_cell = tval * math.sqrt(2*MS_error/(r_mean)) if not any(math.isnan(x) for x in [tval, MS_error, r_mean]) else np.nan

        table = [
            ("A", SS_A, df_A, MS_A, F_A, pA),
            ("B", SS_B, df_B, MS_B, F_B, pB),
            ("C", SS_C, df_C, MS_C, F_C, pC),
            ("A×B", SS_AB, df_AB, MS_AB, F_AB, pAB),
            ("A×C", SS_AC, df_AC, MS_AC, F_AC, pAC),
            ("B×C", SS_BC, df_BC, MS_BC, F_BC, pBC),
            ("A×B×C", SS_ABC, df_ABC, MS_ABC, F_ABC, pABC),
            ("Залишок", SS_error, df_error, MS_error, None, None),
            ("Загальна", SS_total, df_total, None, None, None)
        ]

        return {
            'type': 'three',
            'table': table,
            'eta2': {
                'A': eta2_A, 'B': eta2_B, 'C': eta2_C,
                'AB': eta2_AB, 'AC': eta2_AC, 'BC': eta2_BC,
                'ABC': eta2_ABC, 'res': eta2_rem
            },
            'LSD': {'A': LSD_A, 'B': LSD_B, 'C': LSD_C, 'cell': LSD_cell},
            'means_A': mean_A, 'means_B': mean_B, 'means_C': mean_C,
            'cell_means': mean_cell,
            'p_values': {'A': pA, 'B': pB, 'C': pC, 'AB': pAB, 'AC': pAC, 'BC': pBC, 'ABC': pABC}
        }

    else:
        raise ValueError("Unsupported number of factors")

# -------------------------
# GUI
# -------------------------
class SADTk:
    def __init__(self, root):
        self.root = root
        root.title("SAD — Статистичний аналіз даних")
        root.geometry("900x450")
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(expand=True, fill=tk.BOTH)

        title = tk.Label(self.main_frame, text="SAD — Статистичний аналіз даних", font=("Arial", 18, "bold"))
        title.pack(pady=12)

        btn_frame = tk.Frame(self.main_frame)
        btn_frame.pack(pady=8)

        one_btn = tk.Button(btn_frame, text="Однофакторний аналіз", width=24, height=2, command=lambda: self.open_table(1))
        two_btn = tk.Button(btn_frame, text="Двофакторний аналіз", width=24, height=2, command=lambda: self.open_table(2))
        three_btn = tk.Button(btn_frame, text="Трифакторний аналіз", width=24, height=2, command=lambda: self.open_table(3))

        one_btn.grid(row=0, column=0, padx=8)
        two_btn.grid(row=0, column=1, padx=8)
        three_btn.grid(row=0, column=2, padx=8)

        info = tk.Label(self.main_frame, text="Виберіть тип аналізу, внесіть дані в таблицю, натисніть 'Аналіз'", fg="gray")
        info.pack(pady=10)

        # placeholder for table window
        self.table_win = None

    def open_table(self, factors_count):
        # close existing table window if any
        if self.table_win and tk.Toplevel.winfo_exists(self.table_win):
            self.table_win.destroy()

        self.factors_count = factors_count
        self.table_win = tk.Toplevel(self.root)
        self.table_win.title(f"SAD — {factors_count}-факторний аналіз")
        self.table_win.geometry("1100x600")

        # columns: factors + 4 repeats
        self.repeat_count = 4
        self.factor_names = []
        for i in range(factors_count):
            self.factor_names.append(f"Фактор {'АБВ'[i]}")  # 'А','Б','В'
        self.column_names = self.factor_names + [f"Повт.{i+1}" for i in range(self.repeat_count)]

        # top controls
        ctl_frame = tk.Frame(self.table_win)
        ctl_frame.pack(fill=tk.X, padx=8, pady=6)
        tk.Button(ctl_frame, text="Додати рядок", command=self.add_row).pack(side=tk.LEFT, padx=4)
        tk.Button(ctl_frame, text="Видалити рядок", command=self.delete_row).pack(side=tk.LEFT, padx=4)
        tk.Button(ctl_frame, text="Аналіз", bg="#d32f2f", fg="white", command=self.analyze).pack(side=tk.LEFT, padx=20)

        # canvas + frame to allow scrolling
        canvas = tk.Canvas(self.table_win)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(self.table_win, orient="vertical", command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.configure(yscrollcommand=scrollbar.set)
        inner = tk.Frame(canvas)
        canvas.create_window((0,0), window=inner, anchor="nw")

        # build table grid of Entry widgets
        self.rows = 10
        self.cols = len(self.column_names)
        self.entries = []  # 2D list [row][col] -> Entry
        # header row
        for j, name in enumerate(self.column_names):
            lbl = tk.Label(inner, text=name, relief=tk.RIDGE, width=12, bg="#f0f0f0")
            lbl.grid(row=0, column=j, padx=1, pady=1, sticky="nsew")
        # data rows
        for i in range(self.rows):
            row_entries = []
            for j in range(self.cols):
                e = tk.Entry(inner, width=12)
                e.grid(row=i+1, column=j, padx=1, pady=1)
                # bind Enter to move down
                e.bind("<Return>", self.on_enter)
                # bind Ctrl+V paste starting here
                e.bind("<Control-v>", self.on_paste)
                e.bind("<Control-V>", self.on_paste)
                e.bind("<Control-c>", self.on_copy)
                e.bind("<Control-C>", self.on_copy)
                row_entries.append(e)
            self.entries.append(row_entries)

        # update scrollregion
        inner.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

        # click to focus first cell
        self.entries[0][0].focus_set()

    # -------------------
    # Table operations
    # -------------------
    def add_row(self):
        inner = self.entries[0][0].master  # frame
        i = len(self.entries)
        row_entries = []
        for j in range(self.cols):
            e = tk.Entry(inner, width=12)
            e.grid(row=i+1, column=j, padx=1, pady=1)
            e.bind("<Return>", self.on_enter)
            e.bind("<Control-v>", self.on_paste)
            e.bind("<Control-V>", self.on_paste)
            e.bind("<Control-c>", self.on_copy)
            row_entries.append(e)
        self.entries.append(row_entries)
        self.rows += 1

    def delete_row(self):
        if len(self.entries) == 0:
            return
        last = self.entries.pop()
        for e in last:
            e.destroy()
        self.rows -= 1

    # -------------------
    # Clipboard helpers: copy/paste block from/to clipboard
    # -------------------
    def on_copy(self, event=None):
        w = event.widget
        try:
            sel = w.selection_get()
        except Exception:
            sel = w.get()
        self.table_win.clipboard_clear()
        self.table_win.clipboard_append(sel)
        return "break"

    def on_paste(self, event=None):
        # paste block starting at this widget
        widget = event.widget
        try:
            data = widget.selection_get(selection='CLIPBOARD')
        except Exception:
            data = self.table_win.clipboard_get()
        rows = [r for r in data.splitlines() if r!='']
        # find position of widget in entries
        pos = None
        for i, row in enumerate(self.entries):
            for j, cell in enumerate(row):
                if cell is widget:
                    pos = (i, j)
                    break
            if pos: break
        if not pos:
            return "break"
        r0, c0 = pos
        for i_r, row_text in enumerate(rows):
            cols = row_text.split("\t")
            for j_c, val in enumerate(cols):
                rr = r0 + i_r
                cc = c0 + j_c
                # if beyond current table, add rows
                while rr >= len(self.entries):
                    self.add_row()
                if cc >= self.cols:
                    # cannot paste beyond existing cols in this design; skip
                    continue
                self.entries[rr][cc].delete(0, tk.END)
                self.entries[rr][cc].insert(0, val)
        return "break"

    # -------------------
    # Enter key navigation
    # -------------------
    def on_enter(self, event=None):
        widget = event.widget
        # find its position
        pos = None
        for i, row in enumerate(self.entries):
            for j, cell in enumerate(row):
                if cell is widget:
                    pos = (i, j)
                    break
            if pos: break
        if not pos:
            return "break"
        i, j = pos
        ni = i + 1
        if ni >= len(self.entries):
            self.add_row()
        # focus next cell same column
        self.entries[ni][j].focus_set()
        self.entries[ni][j].delete(0, tk.END)
        return "break"

    # -------------------
    # Collect data and analyze
    # -------------------
    def collect_long(self):
        """Return list of dicts {'A':..., 'B':..., 'C':..., 'value':...} depending on factors_count"""
        long = []
        for i, row in enumerate(self.entries):
            # read factor levels
            factors = []
            for k in range(self.factors_count):
                val = row[k].get().strip()
                if val == "":
                    val = f"lev_row{i}_col{k}"  # assign unique level if empty
                factors.append(val)
            # read repeats
            for rep_col in range(self.factors_count, self.factors_count + self.repeat_count):
                val = row[rep_col].get().strip()
                if val == "":
                    continue
                try:
                    v = float(val)
                except:
                    continue
                rec = {'value': v}
                if self.factors_count >= 1:
                    rec['A'] = factors[0]
                if self.factors_count >= 2:
                    rec['B'] = factors[1]
                if self.factors_count >= 3:
                    rec['C'] = factors[2]
                long.append(rec)
        return long

    def analyze(self):
        long = self.collect_long()
        if len(long) == 0:
            messagebox.showwarning("Помилка", "Немає числових даних для аналізу")
            return
        # Shapiro on residuals: we will compute residuals as deviations from cell means
        # build vector of values and cell means
        values = np.array([r['value'] for r in long])
        if len(values) < 3:
            messagebox.showinfo("Результат", "Надто мало даних для тесту Шапіро-Вілк")
            return
        # compute ANOVA
        factors = ['A','B','C'][:self.factors_count]
        res = anova_n_way(long, factors)
        # compute residuals (value - cell_mean)
        cell_mean_map = res.get('cell_means', None)
        if cell_mean_map is None:
            # for 1-way or 2-way build mapping
            if res['type']=='one':
                # mean_i in means
                mean_map = res['means']
                residuals = []
                for rec in long:
                    m = mean_map.get(rec['A'], np.nan)
                    residuals.append(rec['value'] - m)
            elif res['type']=='two':
                mean_map = res['cell_means']
                residuals = []
                for rec in long:
                    m = mean_map.get((rec['A'], rec['B']), np.nan)
                    residuals.append(rec['value'] - m)
        else:
            residuals = []
            for rec in long:
                key = (rec.get('A'), rec.get('B'), rec.get('C'))[:self.factors_count]
                if self.factors_count==1:
                    key = key[0]
                    m = res['means_A'].get(key, np.nan)
                elif self.factors_count==2:
                    m = res['cell_means'].get((rec['A'], rec['B']), np.nan)
                else:
                    m = res['cell_means'].get((rec['A'], rec['B'], rec['C']), np.nan)
                residuals.append(rec['value'] - m)
        residuals = np.array([r for r in residuals if not math.isnan(r)])
        try:
            W, p = shapiro(residuals) if len(residuals)>=3 else (np.nan, np.nan)
        except Exception:
            W, p = (np.nan, np.nan)

        # build textual report
        report_lines = []
        report_lines.append("Р Е З У Л Ь Т А Т И   Т Р И Ф А К Т О Р Н О Г О   Д И С П Е Р С І Й Н О Г О   А Н А Л І З У" if self.factors_count==3 else
                            ("Р Е З У Л Ь Т А Т И   Д В О Ф А К Т О Р Н О Г О   Д И С П Е Р С І Й Н О Г О   А Н А Л І З У" if self.factors_count==2 else
                             "Р Е З У Л Ь Т А Т И   О Д Н О Ф А К Т О Р Н О Г О   Д И С П Е Р С І Й Н О Г О   А Н А Л І З У"))
        report_lines.append("")
        # Factors description
        if self.factors_count>=1:
            report_lines.append(f"Фактор А: {', '.join(sorted({rec['A'] for rec in long}))}")
        if self.factors_count>=2:
            report_lines.append(f"Фактор В: {', '.join(sorted({rec['B'] for rec in long}))}")
        if self.factors_count>=3:
            report_lines.append(f"Фактор С: {', '.join(sorted({rec['C'] for rec in long}))}")
        # repeats and replicates
        total_cells = 1
        for k in range(self.factors_count):
            unique = sorted({rec[factors[k]] for rec in long})
            total_cells *= len(unique)
        total_obs = len(long)
        # try r estimate
        r_est = total_obs / total_cells if total_cells>0 else np.nan
        report_lines.append(f"Кількість повторень (оціночна): {r_est:.2f}")
        report_lines.append("")
        if not math.isnan(W):
            report_lines.append(f"Перевірка нормальності залишків (Shapiro-Wilk): W = {W:.4f}, p = {p:.4f} → {'нoрмальний' if p>0.05 else 'НЕ нормальний'}")
        else:
            report_lines.append("Перевірка нормальності: недостатньо даних для тесту")
        report_lines.append("")
        report_lines.append("-"*70)
        report_lines.append(f"{'Джерело варіації':<30}{'Сума квадратів':>12}{'  df':>8}{'  MS':>12}{'  F':>10}{'   p':>9}{'  Висновок':>12}")
        report_lines.append("-"*70)
        for row in res['table']:
            name, SS, dfv, MS, Fv, pv = row
            SSs = f"{SS:10.2f}" if SS is not None else ""
            df_s = f"{int(dfv):6d}" if dfv is not None and not math.isnan(dfv) else ""
            MSs = f"{MS:10.3f}" if MS is not None and not math.isnan(MS) else ""
            Fs = f"{Fv:8.3f}" if Fv is not None and not math.isnan(Fv) else ""
            ps = f"{pv:8.4f}" if pv is not None and not math.isnan(pv) else ""
            mark = significance_mark(pv) if pv is not None and not math.isnan(pv) else ""
            report_lines.append(f"{name:<30}{SSs:>12}{df_s:>8}{MSs:>12}{Fs:>10}{ps:>9}{mark:>12}")
        report_lines.append("-"*70)
        report_lines.append("")
        # eta2 contributions
        report_lines.append("Вилучення впливу (η², %):")
        eta = res.get('eta2', {})
        for k, v in eta.items():
            if v is None:
                continue
            report_lines.append(f"  • {k:<10} — {v*100:5.1f}%")
        report_lines.append("")
        # LSD
        LSDs = res.get('LSD', {})
        if LSDs:
            report_lines.append("НІР₀.₅ (LSD):")
            for k,v in LSDs.items():
                if v is None or math.isnan(v):
                    continue
                report_lines.append(f"  • {k:<12} — {v:.2f}")
            report_lines.append("")

        # Means by factor (examples)
        if res['type']=='one':
            means = res['means']
            report_lines.append("Середні по фактору A:")
            for lev,m in means.items():
                report_lines.append(f"  {lev:<15}{m:6.2f}")
        elif res['type']=='two':
            report_lines.append("Середні по фактору A:")
            for lev,m in res['means_A'].items():
                report_lines.append(f"  {lev:<15}{m:6.2f}")
            report_lines.append("Середні по фактору B:")
            for lev,m in res['means_B'].items():
                report_lines.append(f"  {lev:<15}{m:6.2f}")
        else:
            report_lines.append("Приклади середніх (перші кілька комбінацій):")
            cnt=0
            for key,m in res['cell_means'].items():
                report_lines.append(f"  {key} -> {m:.2f}")
                cnt+=1
                if cnt>=8: break

        # show results window
        report_text = "\n".join(report_lines)
        win = tk.Toplevel(self.root)
        win.title("Результат аналізу")
        txt = ScrolledText(win, width=110, height=40)
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert("1.0", report_text)
        txt.config(state=tk.NORMAL)  # allow copy
        # done

# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = SADTk(root)
    root.mainloop()
    # -------------------
# Table operations and navigation
# -------------------
def add_row(self):
    inner = self.entries[0][0].master  # frame
    i = len(self.entries)
    row_entries = []
    for j in range(self.cols):
        e = tk.Entry(inner, width=12)
        e.grid(row=i+1, column=j, padx=1, pady=1)
        e.bind("<Return>", self.on_enter)
        e.bind("<Control-v>", self.on_paste)
        e.bind("<Control-V>", self.on_paste)
        e.bind("<Control-c>", self.on_copy)
        e.bind("<Up>", self.on_arrow)
        e.bind("<Down>", self.on_arrow)
        e.bind("<Left>", self.on_arrow)
        e.bind("<Right>", self.on_arrow)
        row_entries.append(e)
    self.entries.append(row_entries)
    self.rows += 1

def delete_row(self):
    if len(self.entries) == 0:
        return
    last = self.entries.pop()
    for e in last:
        e.destroy()
    self.rows -= 1

# -------------------
# Add/remove column
# -------------------
def add_column(self):
    inner = self.entries[0][0].master
    self.cols += 1
    col_idx = self.cols - 1
    lbl = tk.Label(inner, text=f"Повт.{col_idx+1}", relief=tk.RIDGE, width=12, bg="#f0f0f0")
    lbl.grid(row=0, column=col_idx, padx=1, pady=1)
    for i, row in enumerate(self.entries):
        e = tk.Entry(inner, width=12)
        e.grid(row=i+1, column=col_idx, padx=1, pady=1)
        e.bind("<Return>", self.on_enter)
        e.bind("<Control-v>", self.on_paste)
        e.bind("<Control-V>", self.on_paste)
        e.bind("<Control-c>", self.on_copy)
        e.bind("<Up>", self.on_arrow)
        e.bind("<Down>", self.on_arrow)
        e.bind("<Left>", self.on_arrow)
        e.bind("<Right>", self.on_arrow)
        row.append(e)

def delete_column(self):
    if self.cols <= self.factors_count + 1:
        return  # мінімум фактори + 1 повторення
    inner = self.entries[0][0].master
    col_idx = self.cols - 1
    # delete header
    for widget in inner.grid_slaves(row=0, column=col_idx):
        widget.destroy()
    # delete entries
    for row in self.entries:
        row[col_idx].destroy()
        row.pop()
    self.cols -= 1

# -------------------
# Clipboard helpers: copy/paste block from/to clipboard
# -------------------
def on_copy(self, event=None):
    w = event.widget
    try:
        sel = w.selection_get()
    except Exception:
        sel = w.get()
    self.table_win.clipboard_clear()
    self.table_win.clipboard_append(sel)
    return "break"

def on_paste(self, event=None):
    widget = event.widget
    try:
        data = widget.selection_get(selection='CLIPBOARD')
    except Exception:
        data = self.table_win.clipboard_get()
    rows = [r for r in data.splitlines() if r!='']
    pos = None
    for i, row in enumerate(self.entries):
        for j, cell in enumerate(row):
            if cell is widget:
                pos = (i, j)
                break
        if pos: break
    if not pos:
        return "break"
    r0, c0 = pos
    for i_r, row_text in enumerate(rows):
        cols = row_text.split("\t")
        for j_c, val in enumerate(cols):
            rr = r0 + i_r
            cc = c0 + j_c
            while rr >= len(self.entries):
                self.add_row()
            if cc >= self.cols:
                continue
            self.entries[rr][cc].delete(0, tk.END)
            self.entries[rr][cc].insert(0, val)
    return "break"

# -------------------
# Enter key navigation
# -------------------
def on_enter(self, event=None):
    widget = event.widget
    pos = None
    for i, row in enumerate(self.entries):
        for j, cell in enumerate(row):
            if cell is widget:
                pos = (i, j)
                break
        if pos: break
    if not pos:
        return "break"
    i, j = pos
    ni = i + 1
    if ni >= len(self.entries):
        self.add_row()
    self.entries[ni][j].focus_set()
    self.entries[ni][j].delete(0, tk.END)
    return "break"

# -------------------
# Arrow key navigation
# -------------------
def on_arrow(self, event=None):
    widget = event.widget
    pos = None
    for i, row in enumerate(self.entries):
        for j, cell in enumerate(row):
            if cell is widget:
                pos = (i, j)
                break
        if pos: break
    if not pos:
        return "break"
    i, j = pos
    if event.keysym == "Up":
        ni = max(i-1, 0)
        self.entries[ni][j].focus_set()
    elif event.keysym == "Down":
        ni = min(i+1, len(self.entries)-1)
        self.entries[ni][j].focus_set()
    elif event.keysym == "Left":
        nj = max(j-1, 0)
        self.entries[i][nj].focus_set()
    elif event.keysym == "Right":
        nj = min(j+1, len(self.entries[i])-1)
        self.entries[i][nj].focus_set()
    return "break"

# -------------------
# Collect data and analyze
# -------------------
def collect_long(self):
    long = []
    for i, row in enumerate(self.entries):
        factors = []
        for k in range(self.factors_count):
            val = row[k].get().strip()
            if val == "":
                val = f"lev_row{i}_col{k}"
            factors.append(val)
        for rep_col in range(self.factors_count, self.factors_count + self.repeat_count):
            val = row[rep_col].get().strip()
            if val == "":
                continue
            try:
                v = float(val)
            except:
                continue
            rec = {'value': v}
            if self.factors_count >= 1:
                rec['A'] = factors[0]
            if self.factors_count >= 2:
                rec['B'] = factors[1]
            if self.factors_count >= 3:
                rec['C'] = factors[2]
            long.append(rec)
    return long

def analyze(self):
    long = self.collect_long()
    if len(long) == 0:
        messagebox.showwarning("Помилка", "Немає числових даних для аналізу")
        return
    values = np.array([r['value'] for r in long])
    if len(values) < 3:
        messagebox.showinfo("Результат", "Надто мало даних для тесту Шапіро-Вілк")
        return
    factors = ['A','B','C'][:self.factors_count]
    res = anova_n_way(long, factors)
    cell_mean_map = res.get('cell_means', None)
    residuals = []
    if cell_mean_map is None:
        if res['type']=='one':
            mean_map = res['means']
            for rec in long:
                m = mean_map.get(rec['A'], np.nan)
                residuals.append(rec['value'] - m)
        elif res['type']=='two':
            mean_map = res['cell_means']
            for rec in long:
                m = mean_map.get((rec['A'], rec['B']), np.nan)
                residuals.append(rec['value'] - m)
    else:
        for rec in long:
            key = (rec.get('A'), rec.get('B'), rec.get('C'))[:self.factors_count]
            if self.factors_count==1:
                key = key[0]
                m = res['means_A'].get(key, np.nan)
            elif self.factors_count==2:
                m = res['cell_means'].get((rec['A'], rec['B']), np.nan)
            else:
                m = res['cell_means'].get((rec['A'], rec['B'], rec['C']), np.nan)
            residuals.append(rec['value'] - m)
    residuals = np.array([r for r in residuals if not math.isnan(r)])
    try:
        W, p = shapiro(residuals) if len(residuals)>=3 else (np.nan, np.nan)
    except Exception:
        W, p = (np.nan, np.nan)

# -------------------
# About developer
# -------------------
def show_about(self):
    messagebox.showinfo("Про розробника", "Цей додаток створено [Твоє ім'я/компанія]\nВерсія 1.0")

