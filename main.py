# main.py
# -*- coding: utf-8 -*-
"""
SAD — Статистичний Аналіз Даних (Tkinter версія)

Потрібно: Python 3.8+, numpy, scipy
Встановлення: pip install numpy scipy
"""

import math
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText

import numpy as np
from scipy.stats import shapiro, t, f as f_dist


# -------------------------
# Допоміжні статистичні функції
# -------------------------
def significance_mark(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# -------------------------
# ANOVA по маргінальним середнім (supports 1-,2-,3-way)
# Формат вхідних даних:
# long = list of dicts: {'A': levelA, 'B': levelB, 'C': levelC, 'value': float}
# -------------------------
def anova_n_way(long, factors):
    """
    long: list of dicts with factor keys and 'value'
    factors: list like ['A'] or ['A','B'] or ['A','B','C']
    returns: dict with ANOVA table and helper results
    """
    N = len(long)
    values = np.array([rec["value"] for rec in long], dtype=float)
    grand_mean = np.nanmean(values)

    if len(factors) == 1:
        Akey = factors[0]
        levels_A = sorted(list({rec[Akey] for rec in long}))
        n_i = {}
        mean_i = {}
        for lev in levels_A:
            vals = [rec["value"] for rec in long if rec[Akey] == lev and not math.isnan(rec["value"])]
            n_i[lev] = len(vals)
            mean_i[lev] = np.nanmean(vals) if len(vals) > 0 else np.nan

        SS_total = np.nansum((values - grand_mean) ** 2)
        SS_A = sum(n_i[lev] * (mean_i[lev] - grand_mean) ** 2 for lev in levels_A if not math.isnan(mean_i[lev]))

        SS_error = 0.0
        for lev in levels_A:
            vals = [rec["value"] for rec in long if rec[Akey] == lev and not math.isnan(rec["value"])]
            if len(vals) > 0 and not math.isnan(mean_i[lev]):
                SS_error += sum((v - mean_i[lev]) ** 2 for v in vals)

        df_A = len(levels_A) - 1
        df_error = N - len(levels_A)
        df_total = N - 1

        MS_A = SS_A / df_A if df_A > 0 else np.nan
        MS_error = SS_error / df_error if df_error > 0 else np.nan
        F_A = MS_A / MS_error if (not math.isnan(MS_error) and MS_error > 0) else np.nan
        p_A = 1 - f_dist.cdf(F_A, df_A, df_error) if not np.isnan(F_A) else np.nan

        eta2_A = SS_A / SS_total if SS_total > 0 else np.nan

        mean_n = np.mean(list(n_i.values())) if len(n_i) > 0 else np.nan
        tval = t.ppf(0.975, df_error) if df_error > 0 else np.nan
        LSD = tval * math.sqrt(2 * MS_error / mean_n) if not any(math.isnan(x) for x in [tval, MS_error, mean_n]) else np.nan

        table = [
            ("Фактор A", SS_A, df_A, MS_A, F_A, p_A),
            ("Залишок", SS_error, df_error, MS_error, None, None),
            ("Загальна", SS_total, df_total, None, None, None),
        ]

        return {
            "type": "one",
            "table": table,
            "eta2": {"A": eta2_A},
            "LSD": {"A": LSD},
            "means_A": mean_i,
            "p_values": {"A": p_A},
        }

    if len(factors) == 2:
        Akey, Bkey = factors
        levels_A = sorted(list({rec[Akey] for rec in long}))
        levels_B = sorted(list({rec[Bkey] for rec in long}))

        mean_cell = {}
        n_cell = {}
        for a in levels_A:
            for b in levels_B:
                vals = [rec["value"] for rec in long if rec[Akey] == a and rec[Bkey] == b and not math.isnan(rec["value"])]
                n_cell[(a, b)] = len(vals)
                mean_cell[(a, b)] = np.nanmean(vals) if len(vals) > 0 else np.nan

        mean_A = {}
        n_A = {}
        for a in levels_A:
            vals = [rec["value"] for rec in long if rec[Akey] == a and not math.isnan(rec["value"])]
            n_A[a] = len(vals)
            mean_A[a] = np.nanmean(vals) if len(vals) > 0 else np.nan

        mean_B = {}
        n_B = {}
        for b in levels_B:
            vals = [rec["value"] for rec in long if rec[Bkey] == b and not math.isnan(rec["value"])]
            n_B[b] = len(vals)
            mean_B[b] = np.nanmean(vals) if len(vals) > 0 else np.nan

        SS_total = np.nansum((values - grand_mean) ** 2)
        SS_A = sum(n_A[a] * (mean_A[a] - grand_mean) ** 2 for a in levels_A if not math.isnan(mean_A[a]))
        SS_B = sum(n_B[b] * (mean_B[b] - grand_mean) ** 2 for b in levels_B if not math.isnan(mean_B[b]))

        SS_AB = 0.0
        for a in levels_A:
            for b in levels_B:
                m_ab = mean_cell[(a, b)]
                if math.isnan(m_ab):
                    continue
                SS_AB += n_cell[(a, b)] * (m_ab - mean_A[a] - mean_B[b] + grand_mean) ** 2

        SS_error = 0.0
        for a in levels_A:
            for b in levels_B:
                vals = [rec["value"] for rec in long if rec[Akey] == a and rec[Bkey] == b and not math.isnan(rec["value"])]
                m_ab = mean_cell[(a, b)]
                if len(vals) > 0 and not math.isnan(m_ab):
                    SS_error += sum((v - m_ab) ** 2 for v in vals)

        aN = len(levels_A)
        bN = len(levels_B)
        df_A = aN - 1
        df_B = bN - 1
        df_AB = (aN - 1) * (bN - 1)
        df_error = N - aN * bN
        df_total = N - 1

        MS_A = SS_A / df_A if df_A > 0 else np.nan
        MS_B = SS_B / df_B if df_B > 0 else np.nan
        MS_AB = SS_AB / df_AB if df_AB > 0 else np.nan
        MS_error = SS_error / df_error if df_error > 0 else np.nan

        F_A = MS_A / MS_error if (not math.isnan(MS_error) and MS_error > 0) else np.nan
        F_B = MS_B / MS_error if (not math.isnan(MS_error) and MS_error > 0) else np.nan
        F_AB = MS_AB / MS_error if (not math.isnan(MS_error) and MS_error > 0) else np.nan

        pA = 1 - f_dist.cdf(F_A, df_A, df_error) if not np.isnan(F_A) else np.nan
        pB = 1 - f_dist.cdf(F_B, df_B, df_error) if not np.isnan(F_B) else np.nan
        pAB = 1 - f_dist.cdf(F_AB, df_AB, df_error) if not np.isnan(F_AB) else np.nan

        eta2_A = SS_A / SS_total if SS_total > 0 else np.nan
        eta2_B = SS_B / SS_total if SS_total > 0 else np.nan
        eta2_AB = SS_AB / SS_total if SS_total > 0 else np.nan
        eta2_rem = 1 - (eta2_A + eta2_B + eta2_AB) if SS_total > 0 else np.nan

        r_list = [n_cell[(aa, bb)] for aa in levels_A for bb in levels_B]
        r_mean = np.mean([x for x in r_list if x > 0]) if any(x > 0 for x in r_list) else np.nan
        tval = t.ppf(0.975, df_error) if df_error > 0 else np.nan

        # LSD для маргінальних середніх та комбінацій (за середнім r)
        LSD_A = tval * math.sqrt(2 * MS_error / (bN * r_mean)) if not any(math.isnan(x) for x in [tval, MS_error, r_mean]) else np.nan
        LSD_B = tval * math.sqrt(2 * MS_error / (aN * r_mean)) if not any(math.isnan(x) for x in [tval, MS_error, r_mean]) else np.nan
        LSD_AB = tval * math.sqrt(2 * MS_error / (r_mean)) if not any(math.isnan(x) for x in [tval, MS_error, r_mean]) else np.nan

        table = [
            ("Фактор A", SS_A, df_A, MS_A, F_A, pA),
            ("Фактор B", SS_B, df_B, MS_B, F_B, pB),
            ("A×B", SS_AB, df_AB, MS_AB, F_AB, pAB),
            ("Залишок", SS_error, df_error, MS_error, None, None),
            ("Загальна", SS_total, df_total, None, None, None),
        ]

        return {
            "type": "two",
            "table": table,
            "eta2": {"A": eta2_A, "B": eta2_B, "AB": eta2_AB, "res": eta2_rem},
            "LSD": {"A": LSD_A, "B": LSD_B, "AB": LSD_AB},
            "means_A": mean_A,
            "means_B": mean_B,
            "cell_means": mean_cell,
            "p_values": {"A": pA, "B": pB, "AB": pAB},
        }

    if len(factors) == 3:
        Akey, Bkey, Ckey = factors
        levels_A = sorted(list({rec[Akey] for rec in long}))
        levels_B = sorted(list({rec[Bkey] for rec in long}))
        levels_C = sorted(list({rec[Ckey] for rec in long}))

        def get_vals(cond):
            return [rec["value"] for rec in long if cond(rec) and not math.isnan(rec["value"])]

        mean_cell = {}
        n_cell = {}
        for a in levels_A:
            for b in levels_B:
                for c in levels_C:
                    vals = get_vals(lambda r, aa=a, bb=b, cc=c: r[Akey] == aa and r[Bkey] == bb and r[Ckey] == cc)
                    n_cell[(a, b, c)] = len(vals)
                    mean_cell[(a, b, c)] = np.nanmean(vals) if len(vals) > 0 else np.nan

        mean_A = {}
        n_A = {}
        for a in levels_A:
            vals = get_vals(lambda r, aa=a: r[Akey] == aa)
            n_A[a] = len(vals)
            mean_A[a] = np.nanmean(vals) if len(vals) > 0 else np.nan

        mean_B = {}
        n_B = {}
        for b in levels_B:
            vals = get_vals(lambda r, bb=b: r[Bkey] == bb)
            n_B[b] = len(vals)
            mean_B[b] = np.nanmean(vals) if len(vals) > 0 else np.nan

        mean_C = {}
        n_C = {}
        for c in levels_C:
            vals = get_vals(lambda r, cc=c: r[Ckey] == cc)
            n_C[c] = len(vals)
            mean_C[c] = np.nanmean(vals) if len(vals) > 0 else np.nan

        mean_AB = {}
        for a in levels_A:
            for b in levels_B:
                vals = get_vals(lambda r, aa=a, bb=b: r[Akey] == aa and r[Bkey] == bb)
                mean_AB[(a, b)] = np.nanmean(vals) if len(vals) > 0 else np.nan

        mean_AC = {}
        for a in levels_A:
            for c in levels_C:
                vals = get_vals(lambda r, aa=a, cc=c: r[Akey] == aa and r[Ckey] == cc)
                mean_AC[(a, c)] = np.nanmean(vals) if len(vals) > 0 else np.nan

        mean_BC = {}
        for b in levels_B:
            for c in levels_C:
                vals = get_vals(lambda r, bb=b, cc=c: r[Bkey] == bb and r[Ckey] == cc)
                mean_BC[(b, c)] = np.nanmean(vals) if len(vals) > 0 else np.nan

        SS_total = np.nansum((values - grand_mean) ** 2)

        SS_A = sum(n_A[a] * (mean_A[a] - grand_mean) ** 2 for a in levels_A if not math.isnan(mean_A[a]))
        SS_B = sum(n_B[b] * (mean_B[b] - grand_mean) ** 2 for b in levels_B if not math.isnan(mean_B[b]))
        SS_C = sum(n_C[c] * (mean_C[c] - grand_mean) ** 2 for c in levels_C if not math.isnan(mean_C[c]))

        SS_AB = 0.0
        for a in levels_A:
            for b in levels_B:
                vals = get_vals(lambda r, aa=a, bb=b: r[Akey] == aa and r[Bkey] == bb)
                n_ab = len(vals)
                m_ab = mean_AB.get((a, b), np.nan)
                if not math.isnan(m_ab):
                    SS_AB += n_ab * (m_ab - mean_A[a] - mean_B[b] + grand_mean) ** 2

        SS_AC = 0.0
        for a in levels_A:
            for c in levels_C:
                vals = get_vals(lambda r, aa=a, cc=c: r[Akey] == aa and r[Ckey] == cc)
                n_ac = len(vals)
                m_ac = mean_AC.get((a, c), np.nan)
                if not math.isnan(m_ac):
                    SS_AC += n_ac * (m_ac - mean_A[a] - mean_C[c] + grand_mean) ** 2

        SS_BC = 0.0
        for b in levels_B:
            for c in levels_C:
                vals = get_vals(lambda r, bb=b, cc=c: r[Bkey] == bb and r[Ckey] == cc)
                n_bc = len(vals)
                m_bc = mean_BC.get((b, c), np.nan)
                if not math.isnan(m_bc):
                    SS_BC += n_bc * (m_bc - mean_B[b] - mean_C[c] + grand_mean) ** 2

        SS_ABC = 0.0
        for a in levels_A:
            for b in levels_B:
                for c in levels_C:
                    n_abc = n_cell.get((a, b, c), 0)
                    m_abc = mean_cell.get((a, b, c), np.nan)
                    if math.isnan(m_abc):
                        continue
                    term = (
                        m_abc
                        - (mean_AB.get((a, b), np.nan) if not math.isnan(mean_AB.get((a, b), np.nan)) else 0)
                        - (mean_AC.get((a, c), np.nan) if not math.isnan(mean_AC.get((a, c), np.nan)) else 0)
                        - (mean_BC.get((b, c), np.nan) if not math.isnan(mean_BC.get((b, c), np.nan)) else 0)
                        + (mean_A.get(a, np.nan) if not math.isnan(mean_A.get(a, np.nan)) else 0)
                        + (mean_B.get(b, np.nan) if not math.isnan(mean_B.get(b, np.nan)) else 0)
                        + (mean_C.get(c, np.nan) if not math.isnan(mean_C.get(c, np.nan)) else 0)
                        - grand_mean
                    )
                    SS_ABC += n_abc * (term ** 2)

        SS_error = 0.0
        for a in levels_A:
            for b in levels_B:
                for c in levels_C:
                    vals = get_vals(lambda r, aa=a, bb=b, cc=c: r[Akey] == aa and r[Bkey] == bb and r[Ckey] == cc)
                    m_abc = mean_cell.get((a, b, c), np.nan)
                    if len(vals) > 0 and not math.isnan(m_abc):
                        SS_error += sum((v - m_abc) ** 2 for v in vals)

        aN = len(levels_A)
        bN = len(levels_B)
        cN = len(levels_C)

        df_A = aN - 1
        df_B = bN - 1
        df_C = cN - 1
        df_AB = (aN - 1) * (bN - 1)
        df_AC = (aN - 1) * (cN - 1)
        df_BC = (bN - 1) * (cN - 1)
        df_ABC = (aN - 1) * (bN - 1) * (cN - 1)
        df_error = N - aN * bN * cN
        df_total = N - 1

        MS_A = SS_A / df_A if df_A > 0 else np.nan
        MS_B = SS_B / df_B if df_B > 0 else np.nan
        MS_C = SS_C / df_C if df_C > 0 else np.nan
        MS_AB = SS_AB / df_AB if df_AB > 0 else np.nan
        MS_AC = SS_AC / df_AC if df_AC > 0 else np.nan
        MS_BC = SS_BC / df_BC if df_BC > 0 else np.nan
        MS_ABC = SS_ABC / df_ABC if df_ABC > 0 else np.nan
        MS_error = SS_error / df_error if df_error > 0 else np.nan

        def f_p(Fv, dfn):
            if np.isnan(Fv):
                return np.nan
            return 1 - f_dist.cdf(Fv, dfn, df_error)

        F_A = MS_A / MS_error if (not math.isnan(MS_error) and MS_error > 0) else np.nan
        F_B = MS_B / MS_error if (not math.isnan(MS_error) and MS_error > 0) else np.nan
        F_C = MS_C / MS_error if (not math.isnan(MS_error) and MS_error > 0) else np.nan
        F_AB = MS_AB / MS_error if (not math.isnan(MS_error) and MS_error > 0) else np.nan
        F_AC = MS_AC / MS_error if (not math.isnan(MS_error) and MS_error > 0) else np.nan
        F_BC = MS_BC / MS_error if (not math.isnan(MS_error) and MS_error > 0) else np.nan
        F_ABC = MS_ABC / MS_error if (not math.isnan(MS_error) and MS_error > 0) else np.nan

        pA = f_p(F_A, df_A)
        pB = f_p(F_B, df_B)
        pC = f_p(F_C, df_C)
        pAB = f_p(F_AB, df_AB)
        pAC = f_p(F_AC, df_AC)
        pBC = f_p(F_BC, df_BC)
        pABC = f_p(F_ABC, df_ABC)

        eta2_A = SS_A / SS_total if SS_total > 0 else np.nan
        eta2_B = SS_B / SS_total if SS_total > 0 else np.nan
        eta2_C = SS_C / SS_total if SS_total > 0 else np.nan
        eta2_AB = SS_AB / SS_total if SS_total > 0 else np.nan
        eta2_AC = SS_AC / SS_total if SS_total > 0 else np.nan
        eta2_BC = SS_BC / SS_total if SS_total > 0 else np.nan
        eta2_ABC = SS_ABC / SS_total if SS_total > 0 else np.nan
        eta2_rem = 1 - (eta2_A + eta2_B + eta2_C + eta2_AB + eta2_AC + eta2_BC + eta2_ABC) if SS_total > 0 else np.nan

        r_list = [n_cell[(aa, bb, cc)] for aa in levels_A for bb in levels_B for cc in levels_C]
        r_mean = np.mean([x for x in r_list if x > 0]) if any(x > 0 for x in r_list) else np.nan
        tval = t.ppf(0.975, df_error) if df_error > 0 else np.nan

        LSD_A = tval * math.sqrt(2 * MS_error / (bN * cN * r_mean)) if not any(math.isnan(x) for x in [tval, MS_error, r_mean]) else np.nan
        LSD_B = tval * math.sqrt(2 * MS_error / (aN * cN * r_mean)) if not any(math.isnan(x) for x in [tval, MS_error, r_mean]) else np.nan
        LSD_C = tval * math.sqrt(2 * MS_error / (aN * bN * r_mean)) if not any(math.isnan(x) for x in [tval, MS_error, r_mean]) else np.nan
        LSD_cell = tval * math.sqrt(2 * MS_error / (r_mean)) if not any(math.isnan(x) for x in [tval, MS_error, r_mean]) else np.nan

        table = [
            ("Фактор A", SS_A, df_A, MS_A, F_A, pA),
            ("Фактор B", SS_B, df_B, MS_B, F_B, pB),
            ("Фактор C", SS_C, df_C, MS_C, F_C, pC),
            ("A×B", SS_AB, df_AB, MS_AB, F_AB, pAB),
            ("A×C", SS_AC, df_AC, MS_AC, F_AC, pAC),
            ("B×C", SS_BC, df_BC, MS_BC, F_BC, pBC),
            ("A×B×C", SS_ABC, df_ABC, MS_ABC, F_ABC, pABC),
            ("Залишок", SS_error, df_error, MS_error, None, None),
            ("Загальна", SS_total, df_total, None, None, None),
        ]

        return {
            "type": "three",
            "table": table,
            "eta2": {"A": eta2_A, "B": eta2_B, "C": eta2_C, "AB": eta2_AB, "AC": eta2_AC, "BC": eta2_BC, "ABC": eta2_ABC, "res": eta2_rem},
            "LSD": {"A": LSD_A, "B": LSD_B, "C": LSD_C, "cell": LSD_cell},
            "means_A": mean_A,
            "means_B": mean_B,
            "means_C": mean_C,
            "cell_means": mean_cell,
            "p_values": {"A": pA, "B": pB, "C": pC, "AB": pAB, "AC": pAC, "BC": pBC, "ABC": pABC},
        }

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

        tk.Button(btn_frame, text="Однофакторний аналіз", width=24, height=2, command=lambda: self.open_table(1)).grid(row=0, column=0, padx=8)
        tk.Button(btn_frame, text="Двофакторний аналіз", width=24, height=2, command=lambda: self.open_table(2)).grid(row=0, column=1, padx=8)
        tk.Button(btn_frame, text="Трифакторний аналіз", width=24, height=2, command=lambda: self.open_table(3)).grid(row=0, column=2, padx=8)

        info = tk.Label(self.main_frame, text="Виберіть тип аналізу, внесіть дані в таблицю, натисніть 'Аналіз'", fg="gray")
        info.pack(pady=10)

        self.table_win = None

        # config for columns (repeat columns by default)
        self.repeat_count_default = 4
        self.min_repeat_cols = 1  # to keep at least one repeat column

        # developer info
        self.dev_text = (
            "SAD — Статистичний аналіз даних\n"
            "Tkinter-версія.\n"
            "Розробник: (вкажіть ПІБ/кафедру/контакти)\n"
            "© 2025"
        )

    def open_table(self, factors_count):
        if self.table_win and tk.Toplevel.winfo_exists(self.table_win):
            self.table_win.destroy()

        self.factors_count = factors_count
        self.repeat_count = self.repeat_count_default

        self.table_win = tk.Toplevel(self.root)
        self.table_win.title(f"SAD — {factors_count}-факторний аналіз")
        self.table_win.geometry("1150x620")

        # names
        self.factor_names = [f"Фактор {'АБВ'[i]}" for i in range(factors_count)]
        self.column_names = self.factor_names + [f"Повт.{i+1}" for i in range(self.repeat_count)]

        # controls row: left buttons + spacer + right button
        ctl_frame = tk.Frame(self.table_win)
        ctl_frame.pack(fill=tk.X, padx=8, pady=6)

        left = tk.Frame(ctl_frame)
        left.pack(side=tk.LEFT)

        tk.Button(left, text="Додати рядок", command=self.add_row).pack(side=tk.LEFT, padx=4)
        tk.Button(left, text="Видалити рядок", command=self.delete_row).pack(side=tk.LEFT, padx=4)

        tk.Button(left, text="Додати стовпчик", command=self.add_column).pack(side=tk.LEFT, padx=10)
        tk.Button(left, text="Видалити стовпчик", command=self.delete_column).pack(side=tk.LEFT, padx=4)

        tk.Button(left, text="Аналіз", bg="#d32f2f", fg="white", command=self.analyze).pack(side=tk.LEFT, padx=16)

        right = tk.Frame(ctl_frame)
        right.pack(side=tk.RIGHT)
        tk.Button(right, text="Про розробника", command=self.show_about).pack(side=tk.RIGHT, padx=4)

        # canvas + scrollbar
        self.canvas = tk.Canvas(self.table_win)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(self.table_win, orient="vertical", command=self.canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.inner = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        # build initial grid
        self.rows = 10
        self.cols = len(self.column_names)
        self.entries = []

        self.build_header()

        for i in range(self.rows):
            self.append_row_widgets(i)

        self.update_scrollregion()
        self.entries[0][0].focus_set()

    def build_header(self):
        # clear existing header labels in row=0 (if any)
        for widget in self.inner.grid_slaves(row=0):
            widget.destroy()

        for j, name in enumerate(self.column_names):
            lbl = tk.Label(self.inner, text=name, relief=tk.RIDGE, width=12, bg="#f0f0f0")
            lbl.grid(row=0, column=j, padx=1, pady=1, sticky="nsew")

    def bind_cell(self, e):
        e.bind("<Return>", self.on_enter)
        e.bind("<Control-v>", self.on_paste)
        e.bind("<Control-V>", self.on_paste)
        e.bind("<Control-c>", self.on_copy)
        e.bind("<Control-C>", self.on_copy)

        # arrows navigation
        e.bind("<Up>", self.on_arrow)
        e.bind("<Down>", self.on_arrow)
        e.bind("<Left>", self.on_arrow)
        e.bind("<Right>", self.on_arrow)

        return e

    def append_row_widgets(self, i):
        row_entries = []
        for j in range(self.cols):
            e = self.bind_cell(tk.Entry(self.inner, width=12))
            e.grid(row=i + 1, column=j, padx=1, pady=1)
            row_entries.append(e)
        self.entries.append(row_entries)

    def update_scrollregion(self):
        self.inner.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    # -------------------
    # Row operations
    # -------------------
    def add_row(self):
        i = len(self.entries)
        self.append_row_widgets(i)
        self.rows += 1
        self.update_scrollregion()

    def delete_row(self):
        if len(self.entries) == 0:
            return
        last = self.entries.pop()
        for e in last:
            e.destroy()
        self.rows -= 1
        self.update_scrollregion()

    # -------------------
    # Column operations (adds/removes REPEAT columns)
    # -------------------
    def add_column(self):
        # add one more repeat column at the end
        self.repeat_count += 1
        self.column_names = self.factor_names + [f"Повт.{i+1}" for i in range(self.repeat_count)]
        self.cols = len(self.column_names)

        # rebuild header
        self.build_header()

        # add entry in each existing row
        col_idx = self.cols - 1
        for i, row in enumerate(self.entries):
            e = self.bind_cell(tk.Entry(self.inner, width=12))
            e.grid(row=i + 1, column=col_idx, padx=1, pady=1)
            row.append(e)

        self.update_scrollregion()

    def delete_column(self):
        # keep at least factors + min_repeat_cols
        if self.repeat_count <= self.min_repeat_cols:
            return

        # remove last repeat column
        col_idx = self.cols - 1

        # destroy header cell
        for widget in self.inner.grid_slaves(row=0, column=col_idx):
            widget.destroy()

        # destroy cells
        for row in self.entries:
            row[col_idx].destroy()
            row.pop()

        self.repeat_count -= 1
        self.column_names = self.factor_names + [f"Повт.{i+1}" for i in range(self.repeat_count)]
        self.cols = len(self.column_names)

        # rebuild header to keep numbering consistent
        self.build_header()

        self.update_scrollregion()

    # -------------------
    # Clipboard helpers
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
            data = widget.selection_get(selection="CLIPBOARD")
        except Exception:
            data = self.table_win.clipboard_get()

        rows = [r for r in data.splitlines() if r != ""]
        pos = self.find_widget_pos(widget)
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

                # if paste exceeds existing columns -> create more repeat columns
                while cc >= self.cols:
                    self.add_column()

                self.entries[rr][cc].delete(0, tk.END)
                self.entries[rr][cc].insert(0, val)

        return "break"

    # -------------------
    # Navigation: Enter / Arrows
    # -------------------
    def find_widget_pos(self, widget):
        for i, row in enumerate(self.entries):
            for j, cell in enumerate(row):
                if cell is widget:
                    return (i, j)
        return None

    def focus_cell(self, i, j):
        i = max(0, min(i, len(self.entries) - 1))
        j = max(0, min(j, self.cols - 1))
        self.entries[i][j].focus_set()

    def on_enter(self, event=None):
        widget = event.widget
        pos = self.find_widget_pos(widget)
        if not pos:
            return "break"
        i, j = pos
        ni = i + 1
        if ni >= len(self.entries):
            self.add_row()
        self.focus_cell(ni, j)
        return "break"

    def on_arrow(self, event=None):
        widget = event.widget
        pos = self.find_widget_pos(widget)
        if not pos:
            return "break"
        i, j = pos

        if event.keysym == "Up":
            self.focus_cell(i - 1, j)
        elif event.keysym == "Down":
            self.focus_cell(i + 1, j)
        elif event.keysym == "Left":
            self.focus_cell(i, j - 1)
        elif event.keysym == "Right":
            self.focus_cell(i, j + 1)

        return "break"

    # -------------------
    # About developer
    # -------------------
    def show_about(self):
        messagebox.showinfo("Про розробника", self.dev_text)

    # -------------------
    # Collect + analyze
    # -------------------
    def collect_long(self):
        """
        Return list of dicts with factor keys and 'value'.
        Factors are taken from first N columns.
        Repeats are all remaining columns (repeat_count can be changed by add/remove column).
        """
        long = []

        for i, row in enumerate(self.entries):
            factor_levels = []
            for k in range(self.factors_count):
                val = row[k].get().strip()
                if val == "":
                    val = f"lev_row{i}_col{k}"
                factor_levels.append(val)

            # all repeat columns: from factors_count .. cols-1
            for rep_col in range(self.factors_count, self.cols):
                txt = row[rep_col].get().strip()
                if txt == "":
                    continue
                try:
                    v = float(txt.replace(",", "."))
                except Exception:
                    continue

                rec = {"value": v}
                if self.factors_count >= 1:
                    rec["A"] = factor_levels[0]
                if self.factors_count >= 2:
                    rec["B"] = factor_levels[1]
                if self.factors_count >= 3:
                    rec["C"] = factor_levels[2]
                long.append(rec)

        return long

    def analyze(self):
        long = self.collect_long()
        if len(long) == 0:
            messagebox.showwarning("Помилка", "Немає числових даних для аналізу")
            return

        values = np.array([r["value"] for r in long], dtype=float)
        if len(values) < 3:
            messagebox.showinfo("Результат", "Надто мало даних для аналізу / тесту Шапіро-Вілк")
            return

        factors = ["A", "B", "C"][: self.factors_count]
        res = anova_n_way(long, factors)

        # residuals for Shapiro: value - cell mean (or group mean)
        residuals = []
        if res["type"] == "one":
            mean_map = res["means_A"]
            for rec in long:
                residuals.append(rec["value"] - mean_map.get(rec["A"], np.nan))

        elif res["type"] == "two":
            mean_map = res["cell_means"]
            for rec in long:
                residuals.append(rec["value"] - mean_map.get((rec["A"], rec["B"]), np.nan))

        else:
            mean_map = res["cell_means"]
            for rec in long:
                residuals.append(rec["value"] - mean_map.get((rec["A"], rec["B"], rec["C"]), np.nan))

        residuals = np.array([r for r in residuals if not math.isnan(r)], dtype=float)

        try:
            W, p = shapiro(residuals) if len(residuals) >= 3 else (np.nan, np.nan)
        except Exception:
            W, p = (np.nan, np.nan)

        # build report
        report_lines = []
        if self.factors_count == 3:
            report_lines.append("Р Е З У Л Ь Т А Т И   Т Р И Ф А К Т О Р Н О Г О   Д И С П Е Р С І Й Н О Г О   А Н А Л І З У")
        elif self.factors_count == 2:
            report_lines.append("Р Е З У Л Ь Т А Т И   Д В О Ф А К Т О Р Н О Г О   Д И С П Е Р С І Й Н О Г О   А Н А Л І З У")
        else:
            report_lines.append("Р Е З У Л Ь Т А Т И   О Д Н О Ф А К Т О Р Н О Г О   Д И С П Е Р С І Й Н О Г О   А Н А Л І З У")
        report_lines.append("")

        if self.factors_count >= 1:
            report_lines.append(f"Фактор А (рівні): {', '.join(sorted({rec['A'] for rec in long}))}")
        if self.factors_count >= 2:
            report_lines.append(f"Фактор В (рівні): {', '.join(sorted({rec['B'] for rec in long}))}")
        if self.factors_count >= 3:
            report_lines.append(f"Фактор С (рівні): {', '.join(sorted({rec['C'] for rec in long}))}")

        # estimate repeats per cell
        total_cells = 1
        for k in range(self.factors_count):
            unique = sorted({rec[factors[k]] for rec in long})
            total_cells *= len(unique)
        r_est = (len(long) / total_cells) if total_cells > 0 else np.nan
        report_lines.append(f"Кількість повторень (оціночна): {r_est:.2f}")
        report_lines.append("")

        if not math.isnan(W):
            report_lines.append(
                f"Перевірка нормальності залишків (Shapiro-Wilk): W = {W:.4f}, p = {p:.4f} → "
                f"{'нормальний' if p > 0.05 else 'НЕ нормальний'}"
            )
        else:
            report_lines.append("Перевірка нормальності: недостатньо даних для тесту")
        report_lines.append("")

        report_lines.append("─" * 86)
        report_lines.append(f"{'Джерело варіації':<28}{'SS':>12}{'df':>7}{'MS':>12}{'F':>10}{'p':>10}{'':>6}")
        report_lines.append("─" * 86)

        for row in res["table"]:
            name, SS, dfv, MS, Fv, pv = row
            SSs = f"{SS:10.2f}" if SS is not None and not math.isnan(SS) else ""
            df_s = f"{int(dfv):5d}" if dfv is not None and not math.isnan(dfv) else ""
            MSs = f"{MS:10.3f}" if MS is not None and not math.isnan(MS) else ""
            Fs = f"{Fv:8.3f}" if (Fv is not None and not (isinstance(Fv, float) and math.isnan(Fv))) else ""
            ps = f"{pv:8.4f}" if (pv is not None and not (isinstance(pv, float) and math.isnan(pv))) else ""
            mark = significance_mark(pv)
            report_lines.append(f"{name:<28}{SSs:>12}{df_s:>7}{MSs:>12}{Fs:>10}{ps:>10}{mark:>6}")

        report_lines.append("─" * 86)
        report_lines.append("")

        report_lines.append("Вилучення впливу (η², %):")
        for k, v in res.get("eta2", {}).items():
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            report_lines.append(f"  • {k:<6} — {v * 100:5.1f}%")
        report_lines.append("")

        LSDs = res.get("LSD", {})
        if LSDs:
            report_lines.append("НІР₀.₅ (LSD):")
            for k, v in LSDs.items():
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    continue
                report_lines.append(f"  • {k:<6} — {v:.2f}")
            report_lines.append("")

        # Means
        if res["type"] == "one":
            report_lines.append("Середні по фактору A:")
            for lev, m in res["means_A"].items():
                if m is None or math.isnan(m):
                    continue
                report_lines.append(f"  {lev:<20}{m:7.2f}")
        elif res["type"] == "two":
            report_lines.append("Середні по фактору A:")
            for lev, m in res["means_A"].items():
                if m is None or math.isnan(m):
                    continue
                report_lines.append(f"  {lev:<20}{m:7.2f}")
            report_lines.append("Середні по фактору B:")
            for lev, m in res["means_B"].items():
                if m is None or math.isnan(m):
                    continue
                report_lines.append(f"  {lev:<20}{m:7.2f}")
        else:
            report_lines.append("Приклади середніх по комбінаціях (перші 10):")
            cnt = 0
            for key, m in res["cell_means"].items():
                if m is None or math.isnan(m):
                    continue
                report_lines.append(f"  {key} -> {m:.2f}")
                cnt += 1
                if cnt >= 10:
                    break

        report_text = "\n".join(report_lines)

        win = tk.Toplevel(self.root)
        win.title("Результат аналізу (можна копіювати)")
        win.geometry("900x650")

        txt = ScrolledText(win, wrap="none")
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert("1.0", report_text)
        txt.focus_set()

        # buttons for convenience
        btns = tk.Frame(win)
        btns.pack(fill=tk.X)

        def copy_all():
            win.clipboard_clear()
            win.clipboard_append(txt.get("1.0", "end-1c"))
            messagebox.showinfo("Готово", "Звіт скопійовано в буфер обміну.")

        tk.Button(btns, text="Скопіювати весь звіт", command=copy_all).pack(side=tk.LEFT, padx=8, pady=6)


# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = SADTk(root)
    root.mainloop()
