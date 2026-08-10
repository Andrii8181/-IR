# sad_manova.py — MANOVA
# -*- coding: utf-8 -*-
from sad_common import *

class ManovaWindow:
    """MANOVA — Багатовимірний дисперсійний аналіз."""

    HELP_TEXT = """
MANOVA — ПОКРОКОВА ІНСТРУКЦІЯ
══════════════════════════════════════════

ЩО ТАКЕ MANOVA?
  MANOVA (Багатовимірний дисперсійний аналіз) перевіряє чи відрізняються
  групи одночасно за КІЛЬКОМА залежними змінними (показниками).

НАВІЩО НЕ КІЛЬКА ANOVA?
  Якщо провести окремі ANOVA для кожного показника:
  При 5 показниках і α=0.05 → ймовірність хоча б одного хибного
  результату = 1-(0.95)⁵ = 23%!
  MANOVA контролює цю сімейну помилку.
  Крім того MANOVA може виявити ефект який окремі ANOVA пропустять
  (коли ефект є у комбінації показників але не в кожному окремо).

КОЛИ ВИКОРИСТОВУВАТИ?
  ✓ Порівняння сортів за комплексом показників якості
    (врожайність + маса + цукристість + кислотність одночасно)
  ✓ Порівняння варіантів обробки за кількома параметрами росту
  ✓ Будь-коли коли у вас 2+ залежних показники і 2+ групи

КРОК 1. СТРУКТУРА ТАБЛИЦІ

  Перший стовпець: Група (текстові мітки: «Сорт А», «Контроль» тощо)
  Решта стовпців: Залежні змінні — по одній на стовпець (числа)

  Приклад (порівняння 3 сортів за 3 показниками):
  | Сорт    | Врожайність | Висота | Маса зерна |
  | Сорт А  |    5.8      |  95.3  |    38.2    |
  | Сорт А  |    6.1      |  98.1  |    40.5    |
  | Сорт Б  |    4.9      |  88.5  |    35.1    |
  | Сорт Б  |    5.2      |  91.2  |    36.8    |
  | Сорт В  |    6.8      | 102.4  |    43.7    |
  | Сорт В  |    7.1      | 105.8  |    45.2    |

  Мінімум: 2 залежних змінних, 2 групи.

КРОК 2. КРИТИЧНА ВИМОГА: n > p У КОЖНІЙ ГРУПІ

  n = кількість спостережень у групі
  p = кількість залежних змінних

  Якщо у групі 3 спостереження і 4 показники → n ≤ p → MANOVA неможлива!
  Програма ЗАБЛОКУЄ аналіз і пояснить що робити.

  Правило: на кожну залежну змінну потрібно щонайменше 10 спостережень.
  Наприклад: 3 ЗЗ → мінімум 10-15 спостережень на групу.

КРОК 3. АВТОМАТИЧНІ ПЕРЕВІРКИ

  Програма перевіряє 5 передумов:

  ① n > p у кожній групі (критична — блокування)
  ② ≥ 2 залежних змінних
  ③ Мультиколінеарність ЗЗ (|r| > 0.90 → попередження)
  ④ Багатовимірна нормальність (тест Мардіа):
     Перевіряє нормальність векторів спостережень одночасно.
     При порушенні → Pillai's Trace є найнадійнішою статистикою.
  ⑤ Однорідність коваріаційних матриць (Box's M тест):
     Аналог тесту Левена але для матриць, а не дисперсій.

КРОК 4. ІНТЕРПРЕТАЦІЯ РЕЗУЛЬТАТІВ

  Чотири тестові статистики:

  Wilks' Lambda (Λ):
    Найпоширеніша. Від 0 до 1. Менше → сильніший ефект.
    Рекомендується при нормальності і рівних коваріаційних матрицях.

  Pillai's Trace (V): ★ НАЙНАДІЙНІША
    Найробустніша до порушень передумов.
    При порушенні нормальності або Box's M → використовуйте її!
    Програма позначає автоматично.

  Hotelling-Lawley Trace (T):
    Потужна коли один ефект домінує над іншими.

  Roy's GCR:
    Найпотужніша але найменш надійна.
    p-значення — верхня межа, не точне.

  Якщо всі 4 статистики дають p < α → впевнений результат ✓
  Якщо результати суперечливі → орієнтуйтесь на Pillai's Trace

КРОК 5. ПРАВИЛЬНА ПОСЛІДОВНІСТЬ ІНТЕРПРЕТАЦІЇ

  1. Перевірте передумови → при порушеннях читайте попередження
  2. Оцініть Pillai's Trace (p < α → групи відрізняються)
  3. ЯКЩО MANOVA ЗНАЧУЩИЙ → переходьте до univariate ANOVA
  4. Univariate ANOVA використовують поправку Бонферроні: α / кількість ЗЗ
     (наприклад при 4 ЗЗ: 0.05/4 = 0.0125)
  5. ЯКЩО MANOVA НЕЗНАЧУЩИЙ → univariate тести НЕ інтерпретуються!

КРОК 6. РОЗМІР ЕФЕКТУ (partial η²)
  Виводиться у univariate результатах:
  < 0.01: дуже слабкий | 0.01-0.06: слабкий
  0.06-0.14: середній  | > 0.14: сильний
"""

    def __init__(self, parent, gs):
        self.win = tk.Toplevel(parent)
        self.win.title("MANOVA — Багатовимірний дисперсійний аналіз")
        self.win.geometry("1020x700"); set_icon(self.win)
        self.gs = gs; self._build()

    def _build(self):
        # ── Панель інструментів (наш стандарт) ──────────────
        top = tk.Frame(self.win, padx=8, pady=6); top.pack(fill=tk.X)
        tk.Button(top, text="▶ Виконати", bg="#c62828", fg="white",
                  font=("Times New Roman", 13),
                  command=self._run).pack(side=tk.LEFT, padx=4)

        # Налаштування — спадне меню
        mb2 = tk.Menubutton(top, text="⚙ Налаштування ▾",
                            font=("Times New Roman", 11),
                            relief=tk.RAISED, bd=2)
        mb2.pack(side=tk.LEFT, padx=4)
        sm = tk.Menu(mb2, tearoff=0)
        sm.add_command(label="Додати рядок",      command=self._add_row)
        sm.add_command(label="Видалити рядок",    command=self._del_row)
        sm.add_separator()
        sm.add_command(label="Додати стовпець",    command=self._add_col)
        sm.add_command(label="Видалити стовпець",  command=self._del_col)
        sm.add_separator()
        sm.add_command(label="🗑 Очистити таблицю", command=self._clear_table)
        sm.add_separator()
        sm.add_command(label="💾 Зберегти проект", command=self._save_proj)
        sm.add_command(label="📂 Відкрити проект", command=self._load_proj)
        mb2["menu"] = sm

        tk.Button(top, text="Вставити з буфера",
                  font=("Times New Roman", 11),
                  command=self._paste).pack(side=tk.LEFT, padx=4)
        tk.Label(top, text="α:", font=("Times New Roman", 12)).pack(side=tk.LEFT, padx=(10, 2))
        self.alpha_var = tk.StringVar(value="0.05")
        ttk.Combobox(top, textvariable=self.alpha_var, values=["0.01","0.05","0.10"],
                     state="readonly", width=7).pack(side=tk.LEFT)
        tk.Button(top, text="📚 Довідка", bg="#1a4b8c", fg="white",
                  font=("Times New Roman", 11),
                  command=self._show_help).pack(side=tk.LEFT, padx=8)

        # ── Інформаційний рядок ──────────────────────────────
        info = tk.Frame(self.win, bg="#f0f4ff", padx=8, pady=4)
        info.pack(fill=tk.X, padx=8, pady=(0, 4))
        tk.Label(info, text=(
            "Порядок стовпців:  [Група/Фактор]  [Залежна змінна 1]  [Залежна змінна 2]  ...\n"
            "Заголовки (блакитні) можна редагувати.  Перший стовпець — текстові мітки груп.  "
            "Мінімум: 1 група + 2 залежних змінних.  Критично: n > p у кожній групі."),
            font=("Times New Roman", 10), bg="#f0f4ff", justify="left").pack(anchor="w")

        # ── Таблиця даних ────────────────────────────────────
        mid = tk.Frame(self.win); mid.pack(fill=tk.BOTH, expand=True, padx=8)
        self.n_rows = 24; self.n_cols = 8
        self._canvas = tk.Canvas(mid)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(mid, orient="vertical", command=self._canvas.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.configure(yscrollcommand=sb.set)
        self.inner = tk.Frame(self._canvas)
        self._canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>",
                        lambda e: self._canvas.config(scrollregion=self._canvas.bbox("all")))
        self.win.bind("<MouseWheel>",
                      lambda e: self._canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        col_hints = ["Група","Показник 1","Показник 2","Показник 3",
                     "Показник 4","Показник 5","Показник 6","Показник 7"]
        self.header_vars = []
        self.header_entries = []   # for compatibility (used in _run via header_entries)
        for j in range(self.n_cols):
            var = tk.StringVar(value=col_hints[j] if j < len(col_hints) else f"Показник {j}")
            self.header_vars.append(var)
            lbl = tk.Label(self.inner, textvariable=var, width=13,
                           bg="#1a4b8c", fg="white", cursor="hand2",
                           font=("Times New Roman",11,"bold"), relief=tk.RIDGE)
            lbl.grid(row=0, column=j, padx=1, pady=1, sticky="nsew")
            lbl.bind("<Double-Button-1>", lambda e, idx=j: self._rename_manova_col(idx))
            self.header_entries.append(lbl)   # dummy for _run compatibility

        self.entries = []
        for i in range(self.n_rows):
            row_ = []
            for j in range(self.n_cols):
                e = tk.Entry(self.inner, width=13, font=("Times New Roman", 11),
                             highlightthickness=1, highlightbackground="#c0c0c0")
                e.grid(row=i+1, column=j, padx=1, pady=1)
                if j == 0:
                    e.bind("<KeyRelease>",
                           lambda ev: _autofit_col(self.entries, 0, self.header_entries))
                row_.append(e)
            self.entries.append(row_)
        _bind_nav(self.entries, self.win)
        _bind_fill_handle(self.entries, self.win)

    # ── Довідка ───────────────────────────────────────────────
    def _show_help(self):
        win = tk.Toplevel(self.win)
        win.title("Довідка — MANOVA")
        win.geometry("720x660"); set_icon(win)
        frm = tk.Frame(win); frm.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        vsb = ttk.Scrollbar(frm, orient="vertical"); vsb.pack(side=tk.RIGHT, fill=tk.Y)
        txt = tk.Text(frm, wrap="word", font=("Times New Roman", 11),
                      yscrollcommand=vsb.set, relief=tk.FLAT,
                      bg="#fafafa", padx=10, pady=8, cursor="arrow")
        txt.pack(fill=tk.BOTH, expand=True)
        vsb.config(command=txt.yview)
        txt.insert("1.0", self.HELP_TEXT.strip())
        txt.configure(state="disabled")
        txt.bind("<MouseWheel>",
                 lambda e: txt.yview_scroll(int(-1*(e.delta/120)), "units"))
        tk.Button(win, text="Закрити", command=win.destroy,
                  font=("Times New Roman", 11)).pack(pady=6)

    def _help(self):
        self._show_help()

    # ── Управління таблицею ───────────────────────────────────
    def _add_row(self):
        i = self.n_rows; row_ = []
        for j in range(self.n_cols):
            e = tk.Entry(self.inner, width=13, font=("Times New Roman", 11),
                         highlightthickness=1, highlightbackground="#c0c0c0")
            e.grid(row=i+1, column=j, padx=1, pady=1)
            if j == 0:
                e.bind("<KeyRelease>",
                       lambda ev: _autofit_col(self.entries, 0, self.header_entries))
            row_.append(e)
        self.entries.append(row_); self.n_rows += 1
        _bind_nav(self.entries, self.win)
        _bind_fill_handle(self.entries, self.win)
        self.inner.update_idletasks()

    def _del_row(self):
        if not self.entries: return
        for e in self.entries.pop(): e.destroy()
        self.n_rows -= 1

    def _add_col(self):
        ci = self.n_cols; self.n_cols += 1
        var = tk.StringVar(value=f"Показник {ci}")
        self.header_vars.append(var)
        lbl = tk.Label(self.inner, textvariable=var, width=13,
                       bg="#1a4b8c", fg="white", cursor="hand2",
                       font=("Times New Roman",11,"bold"), relief=tk.RIDGE)
        lbl.grid(row=0, column=ci, padx=1, pady=1, sticky="nsew")
        lbl.bind("<Double-Button-1>", lambda e, idx=ci: self._rename_manova_col(idx))
        self.header_entries.append(lbl)
        for i, row_ in enumerate(self.entries):
            e = tk.Entry(self.inner, width=13, font=("Times New Roman", 11),
                         highlightthickness=1, highlightbackground="#c0c0c0")
            e.grid(row=i+1, column=ci, padx=1, pady=1)
            row_.append(e)
        _bind_nav(self.entries, self.win)
        _bind_fill_handle(self.entries, self.win)

    def _del_col(self):
        if self.n_cols <= 3: return
        self.header_entries.pop().destroy()
        self.header_vars.pop()
        for row_ in self.entries: row_.pop().destroy()
        self.n_cols -= 1

    def _clear_table(self):
        if not messagebox.askyesno("Очистити таблицю",
                "Видалити всі числові дані?\n(Заголовки залишаться)"):
            return
        for row in self.entries:
            for e in row: e.delete(0, tk.END)

    def _save_proj(self):
        generic_save_project(self.win, "manova", self.header_vars, self.entries)

    def _load_proj(self):
        d = generic_load_project(self.win)
        if d is None: return
        headers = d.get("headers", []); rd = d.get("rows_data", [])
        while self.n_cols < len(headers): self._add_col()
        for j, h in enumerate(headers):
            if j < len(self.header_vars): self.header_vars[j].set(h)
        while len(self.entries) < len(rd): self._add_row()
        for i, rv in enumerate(rd):
            for j, v in enumerate(rv):
                if i < len(self.entries) and j < len(self.entries[i]):
                    self.entries[i][j].delete(0, tk.END); self.entries[i][j].insert(0, v)

    # ── Вставка з буфера ──────────────────────────────────────
    def _paste(self):
        try: data = self.win.clipboard_get()
        except Exception:
            messagebox.showwarning("Буфер порожній",
                "Скопіюйте дані з Excel (Ctrl+C) і спробуйте знову."); return
        if not data.strip(): return
        r0, c0 = 0, 0
        w = self.win.focus_get()
        if isinstance(w, tk.Entry):
            for i, row_ in enumerate(self.entries):
                for j, e in enumerate(row_):
                    if e is w: r0, c0 = i, j; break
        for ir, line in enumerate(data.splitlines()):
            if not line.strip(): continue
            i = r0 + ir
            while i >= len(self.entries): self._add_row()
            for jc, val in enumerate(line.split("\t")):
                j = c0 + jc
                if j >= self.n_cols: continue
                self.entries[i][j].delete(0, tk.END)
                self.entries[i][j].insert(0, val.strip())
        _autofit_col(self.entries, 0, self.header_entries)



    def _mardia_test(self, X):
        """Mardia's multivariate normality: skewness and kurtosis tests."""
        n, p = X.shape
        if n < p + 1: return np.nan, np.nan, np.nan, np.nan
        X_c = X - np.mean(X, axis=0)
        S = np.cov(X.T, ddof=1)
        try:
            S_inv = np.linalg.pinv(S)
        except Exception: return np.nan, np.nan, np.nan, np.nan
        # Mahalanobis distances
        D = X_c @ S_inv @ X_c.T
        # Mardia skewness
        b1p = float(np.sum(D**3)) / n**2
        k_sk = n * b1p / 6.0
        df_sk = p*(p+1)*(p+2)/6
        p_sk  = float(1 - f_dist.cdf(k_sk/df_sk, df_sk, 1e6)) if df_sk > 0 else np.nan
        # Mardia kurtosis
        b2p = float(np.sum(np.diag(D)**2)) / n
        k_ku = (b2p - p*(p+2)) / math.sqrt(8*p*(p+2)/n)
        from scipy.stats import norm as _norm
        p_ku = float(2 * (1 - _norm.cdf(abs(k_ku))))
        return float(b1p), p_sk, float(b2p), p_ku

    def _box_m_test(self, groups_data, group_levels):
        """Box's M test for homogeneity of covariance matrices (approximate)."""
        k = len(group_levels)
        ns = [len(groups_data[lv]) for lv in group_levels]
        p  = groups_data[group_levels[0]].shape[1]
        covs = []
        for lv in group_levels:
            Xl = groups_data[lv]
            if len(Xl) < 2: return np.nan, np.nan
            covs.append(np.cov(Xl.T, ddof=1))
        # Pooled covariance
        n_tot = sum(ns)
        S_pool = sum((n-1) * C for n, C in zip(ns, covs)) / (n_tot - k)
        try:
            ln_det_pool = np.log(max(np.linalg.det(S_pool), 1e-300))
            ln_dets     = [np.log(max(np.linalg.det(C), 1e-300)) for C in covs]
        except Exception: return np.nan, np.nan
        M = (n_tot - k) * ln_det_pool - sum((n-1)*d for n,d in zip(ns, ln_dets))
        # c1 correction
        c1 = (sum(1/(n-1) for n in ns) - 1/(n_tot-k)) * (2*p**2 + 3*p - 1) / (6*(p+1)*(k-1))
        chi2 = M * (1 - c1)
        df_m  = p*(p+1)*(k-1)/2
        from scipy.stats import chi2 as chi2_dist
        p_m   = float(1 - chi2_dist.cdf(chi2, df_m)) if df_m > 0 else np.nan
        return float(chi2), p_m

    def _run(self):
        alpha = float(self.alpha_var.get())
        # Заголовки з header_vars (tk.StringVar) або header_entries (tk.Entry/Label)
        headers = []
        for j in range(self.n_cols):
            if hasattr(self,'header_vars') and j < len(self.header_vars):
                headers.append(self.header_vars[j].get().strip() or f"Показник {j}")
            else:
                e = self.header_entries[j]
                headers.append(e.get().strip() if hasattr(e,'get') else f"Показник {j}")

        # ── Зчитування даних ─────────────────────────────────
        raw = [[e.get().strip() for e in row] for row in self.entries]
        raw = [r for r in raw if any(v for v in r)]
        if not raw:
            messagebox.showwarning("Немає даних",
                "Будь ласка, введіть дані у таблицю."); return

        groups = []; dv_rows = []; skipped = 0
        for row in raw:
            if len(row) < 2: skipped += 1; continue
            grp = row[0].strip()
            if not grp: skipped += 1; continue
            # Перший стовпець не має бути числом
            try:
                float(grp.replace(",",".")); skipped += 1; continue
            except ValueError: pass
            vals = []
            for v in row[1:]:
                if not v: continue   # пропускаємо порожні, НЕ зупиняємось
                try: vals.append(float(v.replace(",",".")))
                except ValueError: continue
            if len(vals) >= 2:
                groups.append(grp); dv_rows.append(vals)
            else: skipped += 1

        if skipped:
            messagebox.showinfo("Пропущені рядки",
                f"Пропущено {skipped} рядків (порожні або нечислові значення).")

        n = len(dv_rows)
        if n < 4:
            messagebox.showwarning("Замало даних",
                "Потрібно щонайменше 4 повних спостереження."); return

        # Вирівнюємо кількість ЗЗ — беремо мінімальну
        min_dv = min(len(r) for r in dv_rows)
        if min_dv < 2:
            messagebox.showwarning("Замало залежних змінних",
                "Потрібно щонайменше 2 залежних змінних (показники).\n"
                "Переконайтесь що числові дані введені у стовпці 2 і далі."); return

        Y = np.array([r[:min_dv] for r in dv_rows], dtype=float)
        p = min_dv
        dv_names_used = [headers[j+1] if j+1 < len(headers) else f"ЗЗ{j+1}"
                         for j in range(p)]

        group_levels = first_seen(groups)
        k = len(group_levels)

        # ── Guard 1: щонайменше 2 групи ──
        if k < 2:
            messagebox.showwarning("Лише одна група",
                "MANOVA потребує щонайменше 2 групи.\n"
                "Перевірте що перший стовпець містить різні текстові мітки."); return

        # ── Guard 2: n > p у кожній групі (критична!) ──
        groups_data = {}
        for lv in group_levels:
            idx_ = [i for i, g in enumerate(groups) if g == lv]
            groups_data[lv] = Y[idx_]
        for lv in group_levels:
            n_lv = len(groups_data[lv])
            if n_lv <= p:
                messagebox.showerror("ПОРУШЕННЯ: n ≤ p",
                    f"Група '{lv}' має {n_lv} спостережень, але {p} залежних змінних.\n\n"
                    f"MANOVA потребує n > p (спостережень > залежних змінних)\n"
                    f"у КОЖНІЙ групі.\n\n"
                    f"Причина: коваріаційна матриця всередині групи є виродженою\n"
                    f"(singular) і не може бути обернена — MANOVA математично неможлива.\n\n"
                    f"Рішення:\n"
                    f"  • Збільшіть кількість спостережень (рядків) у кожній групі\n"
                    f"    (потрібно мінімум {p+1} спостережень на групу при {p} ЗЗ)\n"
                    f"  • Зменшіть кількість залежних змінних\n"
                    f"  • Спочатку виконайте PCA → аналізуйте головні компоненти")
                return

        # ── Guard 3: мультиколінеарність ЗЗ ──
        corr_Y = np.corrcoef(Y.T)
        high_corr_pairs = []
        for i in range(p):
            for j in range(i+1, p):
                if abs(corr_Y[i,j]) > 0.90:
                    high_corr_pairs.append(
                        (dv_names_used[i], dv_names_used[j], corr_Y[i,j]))
        if high_corr_pairs:
            details = "\n".join(
                f"  • '{a}' та '{b}': r = {c:.3f}" for a,b,c in high_corr_pairs)
            ans = messagebox.askyesno("Висока мультиколінеарність між ЗЗ",
                "Наступні пари залежних змінних сильно корелюють (|r| > 0.90):\n"
                + details + "\n\n"
                "Висока мультиколінеарність знижує потужність MANOVA і може\n"
                "призвести до виродженої коваріаційної матриці.\n\n"
                "Рекомендація: видаліть одну зі змінних або спочатку виконайте PCA.\n\n"
                "Продовжити попри це?")
            if not ans: return

        # ── Guard 4: Багатовимірна нормальність (Мардіа) ──
        Y_res = np.vstack([groups_data[lv] - np.mean(groups_data[lv], axis=0)
                            for lv in group_levels])
        b1p, p_sk, b2p, p_ku = self._mardia_test(Y_res)
        mv_normal = True
        if (not math.isnan(p_sk) and p_sk < alpha) or (not math.isnan(p_ku) and p_ku < alpha):
            mv_normal = False
            ans = messagebox.askyesno(
                "Порушення багатовимірної нормальності (тест Мардіа)",
                f"Тест Мардіа — асиметрія: b1p={fmt(b1p,4)},  p={fmt(p_sk,4)}\n"
                f"Тест Мардіа — ексцес:    b2p={fmt(b2p,4)},  p={fmt(p_ku,4)}\n\n"
                "Передумова багатовимірної нормальності порушена.\n"
                "Найнадійніша статистика у цьому випадку — Pillai's Trace.\n\n"
                "Примітка: MANOVA достатньо робастна при великих вибірках (n > 20 на групу).\n\n"
                "Продовжити? (Pillai's Trace буде позначено як найнадійніша)")
            if not ans: return

        # ── Guard 5: Box's M тест ──
        box_chi2, box_p = self._box_m_test(groups_data, group_levels)
        if not math.isnan(box_p) and box_p < 0.001:
            ans = messagebox.askyesno(
                "Неоднорідність коваріаційних матриць (Box's M)",
                f"Box's M: χ²={fmt(box_chi2,4)},  p={fmt(box_p,6)}\n\n"
                "Коваріаційні матриці значущо відрізняються між групами.\n"
                "Примітка: Box's M дуже чутливий до ненормальності.\n"
                "Якщо p лише трохи < 0.001 — це може бути хибний сигнал.\n\n"
                "Pillai's Trace є найробустнішою при порушенні цієї передумови.\n\n"
                "Продовжити?")
            if not ans: return

        # ══ MANOVA computation ══
        # Between-groups matrix H and Within-groups matrix E
        grand_mean = np.mean(Y, axis=0)
        E = np.zeros((p, p))  # within (error)
        H = np.zeros((p, p))  # between (hypothesis)
        for lv in group_levels:
            Xl = groups_data[lv]
            nl = len(Xl)
            grp_mean = np.mean(Xl, axis=0)
            E += (Xl - grp_mean).T @ (Xl - grp_mean)
            H += nl * np.outer(grp_mean - grand_mean, grp_mean - grand_mean)

        df_h = k - 1        # between df
        df_e = n - k        # within df

        # Eigenvalues of E⁻¹H
        try:
            E_inv = np.linalg.pinv(E)
            EinvH = E_inv @ H
            eigenvalues = np.real(np.linalg.eigvals(EinvH))
            eigenvalues = np.sort(eigenvalues[eigenvalues > 1e-10])[::-1]
        except Exception as ex:
            messagebox.showerror("Помилка обчислення",
                f"Could not compute eigenvalues: {ex}\n"
                "Check for singular covariance matrix (n ≤ p in some group)."); return

        if len(eigenvalues) == 0:
            messagebox.showerror("Немає дійсних власних значень",
                "The covariance matrix is singular. Cannot compute MANOVA.\n"
                "Ensure n > p in every group."); return

        s = min(df_h, p)   # number of non-zero eigenvalues

        # ── Four test statistics ──
        # Wilks' Lambda
        wilks_L = float(np.prod(1 / (1 + eigenvalues[:s])))
        # Approximate F for Wilks
        m_w = df_e + df_h - (p + df_h + 1) / 2
        q_w = math.sqrt((p**2 * df_h**2 - 4) / (p**2 + df_h**2 - 5)) if (p**2 + df_h**2 - 5) > 0 else 1
        df1_w = p * df_h
        df2_w = m_w * q_w - p * df_h / 2 + 1
        F_wilks = ((1 - wilks_L**(1/q_w)) / (wilks_L**(1/q_w))) * (df2_w / df1_w) if (wilks_L > 0 and q_w > 0) else np.nan
        p_wilks = float(1 - f_dist.cdf(F_wilks, df1_w, df2_w)) if not math.isnan(F_wilks) else np.nan

        # Pillai's Trace
        pillai_V = float(np.sum(eigenvalues[:s] / (1 + eigenvalues[:s])))
        # Approximate F for Pillai
        m_p = max(p, df_h)
        F_pillai = (pillai_V / s) / ((s - pillai_V) / s) * ((df_e + df_h - m_p - 1) / m_p) if (s - pillai_V > 0 and m_p > 0) else np.nan
        df1_p = s * m_p; df2_p = s * (df_e + df_h - m_p - 1)
        p_pillai = float(1 - f_dist.cdf(F_pillai, df1_p, df2_p)) if not math.isnan(F_pillai) else np.nan

        # Hotelling-Lawley Trace
        hl_T = float(np.sum(eigenvalues[:s]))
        b_hl = (df_e + df_h - p - 1) * hl_T / s if s > 0 else np.nan
        df1_hl = s * p; df2_hl = s * (df_e + df_h - p - 1)
        F_hl   = b_hl * (df2_hl / (df1_hl * s)) if (not math.isnan(b_hl) and df1_hl > 0 and s > 0) else np.nan
        p_hl   = float(1 - f_dist.cdf(F_hl, df1_hl, df2_hl)) if not math.isnan(F_hl) else np.nan

        # Roy's GCR
        roy_GCR = float(eigenvalues[0]) if len(eigenvalues) > 0 else np.nan
        # Upper bound F for Roy
        F_roy = roy_GCR * df_e / p if p > 0 else np.nan
        p_roy = float(1 - f_dist.cdf(F_roy, p, df_e)) if not math.isnan(F_roy) else np.nan

        # ── Univariate follow-up ANOVAs ──
        univ_rows = []
        bonf_alpha = alpha / p  # Bonferroni correction
        for dv_i, dv_nm in enumerate(dv_names_used):
            y_i = Y[:, dv_i]
            grand_m = np.mean(y_i)
            ss_b = sum(len(groups_data[lv]) * (np.mean(groups_data[lv][:,dv_i]) - grand_m)**2
                       for lv in group_levels)
            ss_w = sum(np.sum((groups_data[lv][:,dv_i] - np.mean(groups_data[lv][:,dv_i]))**2)
                       for lv in group_levels)
            ms_b = ss_b / df_h if df_h > 0 else np.nan
            ms_w = ss_w / df_e if df_e > 0 else np.nan
            F_i  = ms_b / ms_w if (not math.isnan(ms_w) and ms_w > 0) else np.nan
            p_i  = float(1 - f_dist.cdf(F_i, df_h, df_e)) if not math.isnan(F_i) else np.nan
            eta2_i = ss_b / (ss_b + ss_w) if (ss_b + ss_w) > 0 else np.nan
            mark_bonf = "значуще" if (not math.isnan(p_i) and p_i < bonf_alpha) else "незнач."
            univ_rows.append([dv_nm, fmt(F_i,4), fmt(p_i,4),
                              fmt(eta2_i,4), eta2_label(eta2_i),
                              f"α_Bonf = {fmt(bonf_alpha,4)}", mark_bonf])

        self._show_results(wilks_L, F_wilks, p_wilks,
                           pillai_V, F_pillai, p_pillai,
                           hl_T, F_hl, p_hl,
                           roy_GCR, F_roy, p_roy,
                           df1_w, df2_w, df1_p, df2_p, df1_hl, df2_hl,
                           univ_rows, dv_names_used,
                           b1p, p_sk, b2p, p_ku, box_chi2, box_p,
                           alpha, mv_normal, groups_data, group_levels, Y, p)

    def _show_results(self, wilks_L, F_wilks, p_wilks,
                      pillai_V, F_pillai, p_pillai,
                      hl_T, F_hl, p_hl,
                      roy_GCR, F_roy, p_roy,
                      df1_w, df2_w, df1_p, df2_p, df1_hl, df2_hl,
                      univ_rows, dv_names,
                      b1p, p_sk, b2p, p_ku, box_chi2, box_p,
                      alpha, mv_normal, groups_data, group_levels, Y, p):
        win = tk.Toplevel(self.win); win.title("MANOVA — Результати")
        n_dv = len(dv_names); n_grp = len(group_levels)
        # Розмір вікна адаптовано під вміст: більше залежних змінних/груп
        # → ширші таблиці й графіки → трохи більше вікно (в розумних межах).
        est_w = min(1500, max(1050, 220 + 140*n_dv))
        est_h = min(900, max(700, 640 + 18*n_grp))
        win.geometry(f"{est_w}x{est_h}"); set_icon(win)

        self._manova_figs = {}
        self._manova_colors = ["#4c72b0","#dd8452","#55a868","#c44e52","#8172b2","#937860"]
        if not hasattr(self, "_manova_gs"):
            self._manova_gs = {
                "colors": ["#4c72b0","#dd8452","#55a868","#c44e52","#8172b2","#937860"],
                "bar_alpha": 0.85, "lw": 2.0, "ms": 7,
                "font_family": "Times New Roman", "font_size": 9,
            }
        # Єдине джерело правди для параметрів цього аналізу — читається
        # діалогом налаштувань незалежно від того, з якої вкладки (графіка)
        # його відкрито. Раніше кожна вкладка передавала свій власний набір
        # параметрів у _restyle_manova, і з вкладки "Профільний графік"
        # передавались НЕПРАВИЛЬНІ значення (univ_rows=None, alpha=0.05
        # замість реального α) — тому застосування налаштувань звідти
        # ламало позначки значущості на вкладці "Групові середні".
        self._manova_dv_names    = dv_names
        self._manova_groups_data = groups_data
        self._manova_group_levels = group_levels
        self._manova_univ_rows   = univ_rows
        self._manova_alpha       = alpha
        self._manova_p_pillai    = p_pillai
        self._manova_built = {"g1": False, "g2": False}

        main = tk.Frame(win); main.pack(fill=tk.BOTH, expand=True)
        sidebar = tk.Frame(main, width=210, bg="#2c3e50")
        sidebar.pack(side=tk.LEFT, fill=tk.Y); sidebar.pack_propagate(False)
        content = tk.Frame(main); content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(sidebar, text="MANOVA", bg="#2c3e50", fg="#ecf0f1",
                 font=("Times New Roman",12,"bold"), pady=12).pack(fill=tk.X)

        active = {"panel": None, "btn": None}
        def _show_panel(frame, btn):
            if active["panel"] is not None: active["panel"].pack_forget()
            if active["btn"] is not None: active["btn"].configure(bg="#2c3e50", fg="#bdc3c7")
            frame.pack(fill=tk.BOTH, expand=True)
            active["panel"] = frame; active["btn"] = btn
            btn.configure(bg="#c62828", fg="white")

        def _sidebar_btn(text, tooltip):
            fr = tk.Frame(sidebar, bg="#2c3e50"); fr.pack(fill=tk.X)
            b = tk.Button(fr, text=f"  {text}", bg="#2c3e50", fg="#bdc3c7",
                          font=("Times New Roman",11), relief=tk.FLAT,
                          anchor="w", padx=12, pady=6,
                          activebackground="#c62828", activeforeground="white")
            b.pack(fill=tk.X)
            tk.Label(fr, text=f"    {tooltip}", bg="#2c3e50", fg="#7f8c8d",
                     font=("Times New Roman",8), anchor="w").pack(fill=tk.X)
            tk.Frame(sidebar, bg="#3d5166", height=1).pack(fill=tk.X)
            return b

        rpt_frame = tk.Frame(content)
        g1_frame  = tk.Frame(content)
        g2_frame  = tk.Frame(content)
        self._manova_g1_frame = g1_frame
        self._manova_g2_frame = g2_frame

        b_rpt = _sidebar_btn("📋 Звіт",               "Статистики, тести, висновок")
        b_g1  = _sidebar_btn("📊 Групові середні",     "Стовпчикова діаграма ±СП")
        b_g2  = _sidebar_btn("📈 Профільний графік",   "Нормовані середні по ЗЗ")

        def _open_rpt(): _show_panel(rpt_frame, b_rpt)
        def _open_g1():
            _show_panel(g1_frame, b_g1)
            if not self._manova_built["g1"]:
                g1_frame.update_idletasks()
                self._build_manova_g1_panel(g1_frame, dv_names, groups_data,
                                            group_levels, univ_rows, alpha)
                self._manova_built["g1"] = True
        def _open_g2():
            _show_panel(g2_frame, b_g2)
            if not self._manova_built["g2"]:
                g2_frame.update_idletasks()
                self._build_manova_g2_panel(g2_frame, dv_names, groups_data, group_levels)
                self._manova_built["g2"] = True

        b_rpt.configure(command=_open_rpt)
        b_g1.configure( command=_open_g1)
        b_g2.configure( command=_open_g2)

        self._build_manova_report_panel(
            rpt_frame, win, wilks_L, F_wilks, p_wilks, pillai_V, F_pillai, p_pillai,
            hl_T, F_hl, p_hl, roy_GCR, F_roy, p_roy,
            df1_w, df2_w, df1_p, df2_p, df1_hl, df2_hl,
            univ_rows, dv_names, b1p, p_sk, b2p, p_ku, box_chi2, box_p,
            alpha, mv_normal, groups_data, group_levels, Y, p)

        _show_panel(rpt_frame, b_rpt)

    def _build_manova_report_panel(self, frame, win, wilks_L, F_wilks, p_wilks,
                                   pillai_V, F_pillai, p_pillai,
                                   hl_T, F_hl, p_hl, roy_GCR, F_roy, p_roy,
                                   df1_w, df2_w, df1_p, df2_p, df1_hl, df2_hl,
                                   univ_rows, dv_names,
                                   b1p, p_sk, b2p, p_ku, box_chi2, box_p,
                                   alpha, mv_normal, groups_data, group_levels, Y, p):
        tb_res = tk.Frame(frame, padx=6, pady=5); tb_res.pack(fill=tk.X)
        tk.Button(tb_res, text="📋 Копіювати звіт (текст)", font=("Times New Roman",11),
                  command=lambda: self._copy_manova_text(win)).pack(side=tk.LEFT, padx=4)

        main = tk.Frame(frame); main.pack(fill=tk.BOTH, expand=True)
        vsb = ttk.Scrollbar(main, orient="vertical"); vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas = tk.Canvas(main, yscrollcommand=vsb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.config(command=canvas.yview)
        self._manova_body = body = tk.Frame(canvas)
        body_win = canvas.create_window((0,0), window=body, anchor="nw")
        body.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(body_win, width=e.width))
        win.bind("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)),"units"))

        def _head(txt):
            tk.Label(body, text=txt, font=("Times New Roman",12,"bold"),
                     bg="#e8eeff", anchor="w", padx=8, pady=3
                     ).pack(fill=tk.X, padx=6, pady=10)
        def _txt(txt, color="#000000"):
            tk.Label(body, text=txt, font=("Times New Roman",11), fg=color,
                     anchor="w", justify="left").pack(fill=tk.X, padx=14, pady=1)
        def _tbl(headers, rows):
            f, _ = make_tv(body, headers, rows); f.pack(fill=tk.BOTH, expand=True, padx=10, pady=2)

        # ── Заголовок ────────────────────────────────────────
        tk.Label(body,
                 text=f"MANOVA — Багатовимірний дисперсійний аналіз    α = {alpha}",
                 font=("Times New Roman",13,"bold"), anchor="w", padx=10, pady=6
                 ).pack(fill=tk.X)

        # ── sig_mark з урахуванням обраного α ─────────────────
        def _sig(p_val):
            """Позначення значущості відповідно до обраного рівня α."""
            if p_val is None or math.isnan(p_val): return ""
            if p_val < alpha * 0.2:   return "**"   # суттєво нижче α (умовно «дуже значущий»)
            if p_val < alpha:         return "*"
            return "–"

        # ── Перевірка передумов ───────────────────────────────
        _head("Перевірка передумов")

        norm_sk_ok = math.isnan(p_sk) or p_sk > alpha
        norm_ku_ok = math.isnan(p_ku) or p_ku > alpha
        box_ok     = math.isnan(box_p) or box_p >= 0.001

        _txt(f"Тест Мардіа — асиметрія (нормальність):   b1p = {fmt(b1p,4)},  p = {fmt(p_sk,4)}  "
             f"{'✓ ОК' if norm_sk_ok else '⚠ порушено'}",
             "#000000" if norm_sk_ok else "#c62828")
        _txt(f"Тест Мардіа — ексцес (нормальність):      b2p = {fmt(b2p,4)},  p = {fmt(p_ku,4)}  "
             f"{'✓ ОК' if norm_ku_ok else '⚠ порушено'}",
             "#000000" if norm_ku_ok else "#c62828")
        _txt(f"Box's M (однорідність коваріаційних матриць):  χ² = {fmt(box_chi2,4)},  p = {fmt(box_p,6)}  "
             f"{'✓ ОК' if box_ok else '⚠ значущо (Box M чутливий до ненормальності)'}",
             "#000000" if box_ok else "#b07000")
        if not mv_normal:
            _txt("⚠ Багатовимірна нормальність порушена → Pillai's Trace є найнадійнішою статистикою.",
                 "#c62828")

        # ── Тестові статистики MANOVA ─────────────────────────
        _head("Тестові статистики MANOVA")

        s_val = min(len(group_levels)-1, p)
        try:
            eta2_wilks = 1 - wilks_L**(1/max(s_val,1)) if not math.isnan(wilks_L) and wilks_L > 0 else np.nan
        except Exception: eta2_wilks = np.nan
        eta2_pillai = pillai_V / max(s_val,1) if not math.isnan(pillai_V) else np.nan
        eta2_hl    = hl_T / (hl_T+1) if not math.isnan(hl_T) and hl_T >= 0 else np.nan
        eta2_roy   = roy_GCR / (roy_GCR+1) if not math.isnan(roy_GCR) and roy_GCR >= 0 else np.nan

        recommended = "Pillai" if not mv_normal else "Wilks"
        manova_rows = [
            ["Wilks' Lambda",
             fmt(wilks_L,6), fmt(F_wilks,4),
             f"{int(df1_w)},{int(df2_w)}" if not math.isnan(df1_w) else "–",
             fmt(p_wilks,4), _sig(p_wilks),
             fmt(eta2_wilks,4), eta2_label(eta2_wilks),
             "★ стандарт" if recommended=="Wilks" else ""],
            ["Pillai's Trace",
             fmt(pillai_V,6), fmt(F_pillai,4),
             f"{int(df1_p)},{int(df2_p)}" if not math.isnan(df1_p) else "–",
             fmt(p_pillai,4), _sig(p_pillai),
             fmt(eta2_pillai,4), eta2_label(eta2_pillai),
             "★ найробустніша" if recommended=="Pillai" else "робастна"],
            ["Hotelling-Lawley",
             fmt(hl_T,6), fmt(F_hl,4),
             f"{int(df1_hl)},{int(df2_hl)}" if not math.isnan(df1_hl) else "–",
             fmt(p_hl,4), _sig(p_hl),
             fmt(eta2_hl,4), eta2_label(eta2_hl), ""],
            ["Roy's GCR",
             fmt(roy_GCR,6), fmt(F_roy,4), "–",
             fmt(p_roy,4), _sig(p_roy),
             fmt(eta2_roy,4), eta2_label(eta2_roy), "верхня межа"],
        ]
        _tbl(["Статистика","Значення","F","df","p",f"Знач.(α={alpha})",
              "partial η²","Сила ефекту","Примітка"], manova_rows)

        _txt("Сила ефекту (partial η²): < 0.01 дуже слабкий | 0.01–0.06 слабкий | "
             "0.06–0.14 середній | > 0.14 сильний", "#555555")

        # ── Висновок по MANOVA ────────────────────────────────
        _head("Висновок")
        if not math.isnan(p_pillai):
            if p_pillai < alpha:
                _txt(f"✓ MANOVA значущий (Pillai p = {fmt(p_pillai,4)} < α = {alpha}):\n"
                     f"  Групи значуще відрізняються за комбінацією залежних змінних.\n"
                     f"  Перейдіть до одновимірних тестів (таблиця нижче).", "#1a6b1a")
            else:
                _txt(f"✗ MANOVA незначущий (Pillai p = {fmt(p_pillai,4)} ≥ α = {alpha}):\n"
                     f"  Немає достатніх підстав вважати що групи відрізняються за\n"
                     f"  комбінацією залежних змінних.\n"
                     f"  Одновимірні тести у цьому випадку НЕ інтерпретуються!", "#c62828")

        # ── Одновимірні тести (follow-up) ─────────────────────
        bonf_alpha = alpha / max(len(dv_names), 1)
        _head(f"Одновимірні тести (Bonferroni α = {fmt(bonf_alpha,4)})")
        _txt("Примітка: ці результати інтерпретуються ЛИШЕ після значущого MANOVA.",
             "#666666")
        _tbl(["Залежна змінна","F","p",f"Знач.(α={fmt(bonf_alpha,4)})",
              "partial η²","Сила ефекту","Висновок"],
             [[r[0], r[1], r[2], _sig(float(r[2]) if r[2] else float("nan")),
               r[3], r[4], r[6]] for r in univ_rows])

        # ── Групові середні ───────────────────────────────────
        _head("Групові середні (Mean ± SD)")
        means_headers = ["Група"] + [f"{nm}\nМ (SD)" for nm in dv_names]
        means_rows = []
        for lv in group_levels:
            arr = groups_data[lv]
            row_ = [lv]
            for j in range(len(dv_names)):
                m  = float(np.mean(arr[:,j]))
                sd = float(np.std(arr[:,j], ddof=1)) if len(arr) > 1 else 0.
                row_.append(f"{fmt(m,3)} ({fmt(sd,3)})")
            means_rows.append(row_)
        _tbl(means_headers, means_rows)

    # ── Панель: графік групових середніх ±SE ──────────────────
    def _build_manova_g1_panel(self, frame, dv_names, groups_data, group_levels,
                               univ_rows, alpha):
        for w in frame.winfo_children(): w.destroy()
        if not HAS_MPL or len(dv_names) < 1: return
        tb = tk.Frame(frame, padx=6, pady=5); tb.pack(fill=tk.X)
        tk.Button(tb, text="💾 Зберегти PNG", font=("Times New Roman",11),
                  command=lambda: self._save_manova_png(1)).pack(side=tk.LEFT, padx=4)
        tk.Button(tb, text="📋 Копіювати", font=("Times New Roman",11),
                  command=lambda: self._copy_manova_fig(1)).pack(side=tk.LEFT, padx=4)
        tk.Button(tb, text="⚙ Налаштування", font=("Times New Roman",11),
                  command=lambda: self._restyle_manova(frame)
                  ).pack(side=tk.LEFT, padx=4)

        plot_f = tk.Frame(frame); plot_f.pack(fill=tk.BOTH, expand=True)
        self._manova_frame1 = plot_f

        n_dv = len(dv_names)
        colors_ = self._manova_gs.get("colors", self._manova_colors)
        bar_alpha = self._manova_gs.get("bar_alpha", 0.85)
        ff_ = self._manova_gs.get("font_family", "Times New Roman")
        fz_ = self._manova_gs.get("font_size", 9)
        bonf_alpha = alpha / max(n_dv, 1)

        fig1 = Figure(figsize=(max(8, n_dv*2.4), 6), dpi=100)
        for di, dv_nm in enumerate(dv_names):
            ax = fig1.add_subplot(1, n_dv, di+1)
            gm = [float(np.mean(groups_data[lv][:,di])) for lv in group_levels]
            gs_ = [float(np.std(groups_data[lv][:,di],ddof=1) / math.sqrt(len(groups_data[lv])))
                   for lv in group_levels]
            xpos = range(len(group_levels))
            ax.bar(xpos, gm, yerr=gs_, capsize=4,
                   color=[colors_[i % len(colors_)] for i in range(len(group_levels))],
                   alpha=bar_alpha, error_kw={"ecolor":"#333","lw":1.5})
            try:
                p_uv = float(univ_rows[di][2]) if univ_rows and di < len(univ_rows) else float("nan")
                mark = "*" if p_uv < bonf_alpha else ""
                ax.set_title(f"{dv_nm}\n(p={fmt(p_uv,3)}{mark})", fontsize=fz_, fontfamily=ff_)
            except Exception:
                ax.set_title(dv_nm, fontsize=fz_, fontfamily=ff_)
            ax.set_xticks(list(xpos))
            ax.set_xticklabels(group_levels, rotation=30, ha="right", fontsize=max(6,fz_-1))
            ax.set_ylabel("Середнє ± СП" if di==0 else "", fontsize=fz_)
            ax.yaxis.grid(True, alpha=0.3)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig1.suptitle("Групові середні (±СП) по залежних змінних", fontsize=fz_+1, fontfamily=ff_)
        fig1.tight_layout()
        self._manova_figs[1] = fig1
        embed_figure(fig1, plot_f)

    # ── Панель: профільний графік ──────────────────────────────
    def _build_manova_g2_panel(self, frame, dv_names, groups_data, group_levels):
        for w in frame.winfo_children(): w.destroy()
        if not HAS_MPL or len(dv_names) < 2: return
        tb = tk.Frame(frame, padx=6, pady=5); tb.pack(fill=tk.X)
        tk.Button(tb, text="💾 Зберегти PNG", font=("Times New Roman",11),
                  command=lambda: self._save_manova_png(2)).pack(side=tk.LEFT, padx=4)
        tk.Button(tb, text="📋 Копіювати", font=("Times New Roman",11),
                  command=lambda: self._copy_manova_fig(2)).pack(side=tk.LEFT, padx=4)
        tk.Button(tb, text="⚙ Налаштування", font=("Times New Roman",11),
                  command=lambda: self._restyle_manova(frame)
                  ).pack(side=tk.LEFT, padx=4)

        plot_f = tk.Frame(frame); plot_f.pack(fill=tk.BOTH, expand=True)
        self._manova_frame2 = plot_f

        n_dv = len(dv_names)
        colors_ = self._manova_gs.get("colors", self._manova_colors)
        lw_ = self._manova_gs.get("lw", 2.0)
        ms_ = self._manova_gs.get("ms", 7)
        ff_ = self._manova_gs.get("font_family", "Times New Roman")
        fz_ = self._manova_gs.get("font_size", 9)

        fig2 = Figure(figsize=(max(7, n_dv*0.9+2), 5.5), dpi=100)
        ax2 = fig2.add_subplot(111)
        all_means = np.array([[float(np.mean(groups_data[lv][:,j])) for j in range(n_dv)]
                              for lv in group_levels])
        mn_col = all_means.min(axis=0); mx_col = all_means.max(axis=0)
        rng = np.where(mx_col > mn_col, mx_col - mn_col, 1.)
        normed = (all_means - mn_col) / rng
        for gi, lv in enumerate(group_levels):
            ax2.plot(list(range(n_dv)), normed[gi], "o-",
                     color=colors_[gi % len(colors_)],
                     label=str(lv), linewidth=lw_, markersize=ms_)
        ax2.set_xticks(list(range(n_dv)))
        ax2.set_xticklabels(dv_names, rotation=20, ha="right", fontsize=fz_, fontfamily=ff_)
        ax2.set_ylabel("Нормоване середнє (0–1)", fontsize=fz_, fontfamily=ff_)
        ax2.set_title("Профільний графік груп (нормовані середні по ЗЗ)",
                      fontsize=fz_+1, fontfamily=ff_)
        ax2.legend(title="Група", fontsize=fz_, title_fontsize=fz_)
        ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
        fig2.tight_layout()
        self._manova_figs[2] = fig2
        embed_figure(fig2, plot_f)

    def _save_manova_png(self, n):
        fig = self._manova_figs.get(n)
        if fig is None: messagebox.showwarning("","Графік відсутній."); return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                    filetypes=[("PNG зображення","*.png")], title="Зберегти графік")
        if not path: return
        try:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            messagebox.showinfo("Збережено", f"Збережено:\n{path}")
        except Exception as ex:
            messagebox.showerror("Помилка", str(ex))

    # ── Допоміжні методи для результатів MANOVA ───────────────

    def _rebuild_manova_graphs(self, dv_names, groups_data, group_levels,
                                univ_rows, alpha, p_pillai):
        """Перебудовує лише ті графіки MANOVA, які вже були відкриті."""
        if self._manova_built.get("g1") and hasattr(self, '_manova_frame1'):
            self._build_manova_g1_panel(self._manova_frame1.master, dv_names, groups_data,
                                        group_levels, univ_rows, alpha)
        if self._manova_built.get("g2") and hasattr(self, '_manova_frame2'):
            self._build_manova_g2_panel(self._manova_frame2.master, dv_names, groups_data,
                                        group_levels)


    def _copy_manova_text(self, win):
        """Збирає весь текст зі звіту і копіює у буфер."""
        lines = []
        def _collect(w):
            if isinstance(w, tk.Label):
                t = w.cget("text")
                if t: lines.append(t)
            for ch in w.winfo_children(): _collect(ch)
        try:
            _collect(self._manova_body)
        except Exception: pass
        text = "\n".join(lines)
        win.clipboard_clear(); win.clipboard_append(text)
        messagebox.showinfo("Скопійовано",
            "Текст звіту скопійовано у буфер обміну.\nВставте у Word через Ctrl+V.")

    def _copy_manova_fig(self, n):
        fig = self._manova_figs.get(n)
        if fig is None:
            messagebox.showwarning("","Графік не знайдено. Спочатку виконайте аналіз."); return
        ok, msg = _copy_fig_to_clipboard(fig)
        if ok: messagebox.showinfo("","Графік скопійовано.\nВставте у Word через Ctrl+V.")
        else:   messagebox.showwarning("",f"Помилка: {msg}")

    def _restyle_manova(self, win):
        """Діалог налаштувань кольорів/розмірів графіків MANOVA.
        Параметри аналізу (dv_names, groups_data, group_levels, univ_rows,
        alpha, p_pillai) беруться з self._manova_* — це єдине джерело
        правди, незалежно від того, з якої вкладки (графіка) відкрито
        діалог, тож застосування налаштувань завжди перебудовує обидва
        графіки з правильними, узгодженими даними."""
        dv_names     = self._manova_dv_names
        groups_data  = self._manova_groups_data
        group_levels = self._manova_group_levels
        univ_rows    = self._manova_univ_rows
        alpha        = self._manova_alpha
        p_pillai     = self._manova_p_pillai
        if not hasattr(self, '_manova_gs'):
            self._manova_gs = {
                "colors": ["#4c72b0","#dd8452","#55a868","#c44e52","#8172b2","#937860"],
                "bar_alpha": 0.85, "lw": 2.0, "ms": 7,
                "font_family": "Times New Roman", "font_size": 9,
            }
        dlg = tk.Toplevel(win); dlg.title("Налаштування графіків MANOVA")
        dlg.resizable(False, False); set_icon(dlg); dlg.grab_set()
        gs = self._manova_gs
        frm = tk.Frame(dlg, padx=16, pady=14); frm.pack()
        ff_v  = tk.StringVar(value=gs.get("font_family", "Times New Roman"))
        fz_v  = tk.IntVar(value=gs.get("font_size", 9))
        al_v  = tk.DoubleVar(value=gs.get("bar_alpha", 0.85))
        lw_v  = tk.DoubleVar(value=gs.get("lw", 2.0))
        ms_v  = tk.IntVar(value=gs.get("ms", 7))
        rb_f  = ("Times New Roman",12)
        rows_cfg = [
            ("Шрифт:",         "combo",  ff_v, ["Times New Roman","Arial","Calibri","Georgia"]),
            ("Розмір шрифту:", "spin",   fz_v, (7, 18)),
            ("Прозорість стовпців:", "scale", al_v, (0.3, 1.0)),
            ("Товщина ліній:", "scale",  lw_v, (0.5, 4.0)),
            ("Розмір точок:",  "spin",   ms_v, (3, 20)),
        ]
        for ri, (lbl, wt, var, opts) in enumerate(rows_cfg):
            tk.Label(frm, text=lbl, font=rb_f).grid(row=ri, column=0, sticky="w", pady=5)
            if wt == "combo":
                ttk.Combobox(frm, textvariable=var, values=opts,
                             state="readonly", width=20).grid(row=ri, column=1, sticky="w", padx=8)
            elif wt == "spin":
                tk.Spinbox(frm, from_=opts[0], to=opts[1], textvariable=var,
                           width=7).grid(row=ri, column=1, sticky="w", padx=8)
            elif wt == "scale":
                tk.Scale(frm, from_=opts[0], to=opts[1], resolution=0.05,
                         orient="horizontal", variable=var,
                         length=160).grid(row=ri, column=1, sticky="w", padx=8)
        # Colour pickers for groups
        col_refs = list(gs["colors"])
        tk.Label(frm, text="Кольори груп:", font=rb_f).grid(
            row=len(rows_cfg), column=0, sticky="w", pady=5)
        col_frm = tk.Frame(frm); col_frm.grid(row=len(rows_cfg), column=1, sticky="w")
        col_btns = []
        for ci, grp in enumerate(group_levels[:6]):
            c = col_refs[ci] if ci < len(col_refs) else "#999999"
            btn = tk.Button(col_frm, width=4, relief=tk.SUNKEN, bg=c,
                            text=str(ci+1), font=("Times New Roman",9))
            btn.pack(side=tk.LEFT, padx=2)
            def _pick(idx=ci, b=btn, refs=col_refs):
                ch = colorchooser.askcolor(color=refs[idx], parent=dlg)
                if ch and ch[1]: refs[idx]=ch[1]; b.configure(bg=ch[1])
            btn.configure(command=_pick)
            col_btns.append(btn)

        def apply():
            self._manova_gs.update({
                "colors": list(col_refs), "bar_alpha": al_v.get(),
                "lw": lw_v.get(), "ms": ms_v.get(),
                "font_family": ff_v.get(), "font_size": fz_v.get(),
            })
            self._manova_colors = list(col_refs)
            dlg.destroy()
            # Перебудовуємо графіки одразу
            self._rebuild_manova_graphs(dv_names, groups_data, group_levels,
                                         univ_rows, alpha, p_pillai)
        bf = tk.Frame(frm); bf.grid(row=len(rows_cfg)+1, column=0, columnspan=2, pady=(14,0))
        tk.Button(bf, text="OK", bg="#c62828", fg="white",
                  font=rb_f, command=apply).pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text="Скасувати", font=rb_f, command=dlg.destroy).pack(side=tk.LEFT)
        center_win(dlg)



# ═══════════════════════════════════════════════════════════════
# ДОВІДКОВА СИСТЕМА
# ═══════════════════════════════════════════════════════════════

HELP_CONTENT = {}

def _fill_help():
    H = HELP_CONTENT
    H["Швидкий старт"] = {"icon":"🚀","short":"З чого почати роботу з програмою","text":"""
ШВИДКИЙ СТАРТ

КРОК 1. Оберіть тип аналізу на головному екрані.
КРОК 2. Введіть дані:
  Ctrl+V - вставити з Excel
  Файл -> Завантажити Excel (.xlsx)
  Вручну (Enter - наступний рядок, стрілки - навігація)
КРОК 3. Натисніть "Аналіз даних".
КРОК 4. Програма автоматично перевірить:
  ✓ Нормальність розподілу (Shapiro-Wilk)
  ✓ Однорідність дисперсій (Левен)
  ✓ Методологічну коректність дій
КРОК 5. Скопіюйте звіт і графіки у Word.

Збереження: Файл -> Зберегти проект (.sadp)
"""}

    H["Введення даних"] = {"icon":"📋","short":"Як правильно вводити та копіювати дані","text":"""
ВВЕДЕННЯ ДАНИХ

СТРУКТУРА ТАБЛИЦІ ДЛЯ ANOVA:
  Ліві колонки = назви рівнів факторів (текст)
  Праві колонки = числові значення повторностей

  Варіант   | Повт.1 | Повт.2 | Повт.3
  Контроль  |  4.2   |  4.5   |  4.1
  Варіант1  |  5.8   |  6.1   |  5.9

СПОСОБИ ВВЕДЕННЯ:
  1. Вручну: клік -> введіть -> Enter (наступний рядок)
  2. Ctrl+V: вставити з Excel
  3. Файл -> Завантажити Excel (.xlsx, .xlsm)
  4. Fill-handle: наведіть на правий нижній кут -> курсор змінюється -> тягніть вниз
  5. Ctrl+C: скопіювати виділений діапазон

⚠ Назви варіантів вводьте ОДНАКОВО (регістр важливий!)
⚠ Десяткова кома автоматично замінюється на крапку
"""}

    H["Дизайни експерименту"] = {"icon":"🧪","short":"CRD, RCBD та Split-plot: коли що обирати","text":"""
ДИЗАЙНИ ПОЛЬОВИХ ЕКСПЕРИМЕНТІВ

CRD (Повна рандомізація):
  ✓ Умови однорідні (лабораторія, вегетаційні горщики)
  ✓ Мала кількість варіантів (2-5)

RCBD (Блочна рандомізація):
  ✓ Є градієнт родючості (схил, зволоження, pH)
  ✓ Польові досліди з багатьма варіантами
  Кожна числова колонка = один блок (повторність)
  ✓ Виключає міжблокову варіацію з помилки -> точніший аналіз

Split-plot (Спліт-плот):
  ✓ Два фактори з РІЗНИМ розміром ділянок
  Головний фактор (WP) = великі ділянки (обробіток ґрунту, норма посіву)
  Підфактор (SP) = малі ділянки (сорти, дози добрив)

  ⚠ УВАГА: два різних похибки!
    Whole-plot error -> для тесту WP фактора
    Sub-plot error (залишок) -> для SP та взаємодії
  ⚠ Застосування RCBD замість Split-plot -> неправильні F-значення!

ЯК ОБРАТИ:
  Немає блоків -> CRD
  Є блоки, однаковий розмір ділянок -> RCBD
  Є блоки, різний розмір ділянок -> Split-plot
"""}

    H["Нормальність розподілу"] = {"icon":"📊","short":"Shapiro-Wilk і дії при порушенні","text":"""
НОРМАЛЬНІСТЬ РОЗПОДІЛУ - SHAPIRO-WILK

Перевіряється нормальність ЗАЛИШКІВ моделі (не сирих даних!).

p > 0.05 -> нормальний -> параметричний аналіз ✓
p <= 0.05 -> ненормальний -> трансформація або непараметричний

⚠ ОБМЕЖЕННЯ:
  n < 8: тест ненадійний
  n > 100: виявляє найменші відхилення -> дивіться QQ-plot!

ДІЇ ПРИ НЕНОРМАЛЬНОСТІ:

1. ТРАНСФОРМАЦІЯ (повернення до параметричних методів):
   ln(x)  - для правоскошених даних, відносних показників
   sqrt(x) - для даних з рахунком (кількість особин, плям)
   log10(x) - для великих значень
   
   Програма: перевіряє допустимість -> виконує -> перевіряє нормальність знову
   ✓ Нормальний -> пропонує параметричний метод
   ✗ Все ще ненормальний -> зупиняється, рекомендує непараметричний

2. НЕПАРАМЕТРИЧНИЙ АНАЛІЗ:
   CRD: Kruskal-Wallis -> Mann-Whitney (post-hoc)
   RCBD: Friedman -> Wilcoxon (парний)
"""}

    H["Однорідність дисперсій"] = {"icon":"⚖️","short":"Тест Левена і наслідки порушення","text":"""
ОДНОРІДНІСТЬ ДИСПЕРСІЙ - ТЕСТ ЛЕВЕНА

Levene (center=median) = варіант Brown-Forsythe - найробустніший.

p >= 0.05 -> умова виконується ✓
p < 0.05 -> дисперсії різняться -> програма блокує і запитує підтвердження

ДІЇ ПРИ ПОРУШЕННІ:
  1. Трансформація (ln або sqrt стабілізують дисперсію)
  2. Для двох груп -> Welch t-test (у модулі t-тест, автоматично)
  3. Непараметричні методи не потребують рівності дисперсій

ANOVA достатньо робастна якщо:
  Розміри груп приблизно рівні
  Відношення max/min дисперсій < 9:1
"""}

    H["Методи порівнянь"] = {"icon":"🔬","short":"НІР, Тьюкі, Дункан, Бонферроні","text":"""
МЕТОДИ МНОЖИННИХ ПОРІВНЯНЬ (POST-HOC)

НІР05 (Fisher Protected LSD):
  ✓ Найпоширеніший в агрономії України
  ✓ Виконується ТІЛЬКИ після значущого F (Protected LSD)
  Рекомендується: <= 6 варіантів

Тест Тьюкі (Tukey HSD):
  ✓ Строгий контроль сімейної помилки (FWER)
  Рекомендується: будь-яка кількість варіантів

Тест Дункана:
  ✓ Справжня степ-даун процедура: alpha_p = 1-(1-alpha)^(p-1)
  ✓ Проміжний між LSD і Тьюкі
  Рекомендується: 5-10 варіантів

Бонферроні:
  Найконсервативніший. p_adj = p x кількість_пар
  Рекомендується: мала кількість запланованих порівнянь

CLD ЛІТЕРИ:
  Однакові літери = немає значущої різниці
  Різні літери = є значуща різниця (p < alpha)

ЗВЕДЕННЯ:
  <= 6 варіантів -> НІР05
  5-10 варіантів -> Дункан
  Будь-яка кількість -> Тьюкі
  Мало запланованих -> Бонферроні
"""}

    H["Непараметричні тести"] = {"icon":"📉","short":"KW, Friedman, MWU, Wilcoxon та розміри ефектів","text":"""
НЕПАРАМЕТРИЧНІ МЕТОДИ

Kruskal-Wallis (аналог ANOVA для CRD):
  Глобальний тест. Розмір ефекту: epsilon^2
  < 0.01 слабкий | 0.01-0.06 середній | > 0.14 сильний
  Post-hoc: Mann-Whitney з Бонферроні (автоматично)

Friedman (аналог RCBD ANOVA):
  Для блочних дизайнів. Розмір ефекту: Kendall W
  ⚠ Вимагає повні блоки (всі варіанти у кожному блоці)!
  Post-hoc: Wilcoxon з Бонферроні

Mann-Whitney U (попарне):
  Розмір ефекту: Cliff delta
  |delta| < 0.147: дуже слабкий | 0.147-0.33: слабкий
  0.33-0.474: середній | > 0.474: сильний

Wilcoxon signed-rank (парний):
  Для пар спостережень або RCBD з 2 варіантами.
  Розмір ефекту: r = Z/sqrt(n)
"""}

    H["Типи SS"] = {"icon":"∑","short":"Типи сум квадратів I-IV","text":"""
ТИПИ СУМ КВАДРАТІВ

При збалансованих даних - всі типи дають однаковий результат.
Різниця виникає при незбалансованих даних.

Тип I (Послідовний):
  Порядок факторів ВАЖЛИВИЙ!
  ⚠ Програма попередить при незбалансованих даних!
  Коли: збалансовані дизайни, регресія з осмисленим порядком

Тип II (Ієрархічний):
  Порядок НЕ важливий. Вища потужність ніж III при відсутності взаємодій.
  Коли: незбалансовані БЕЗ значущих взаємодій

Тип III (Частковий) <- ЗА ЗАМОВЧУВАННЯМ:
  Кожен ефект при всіх інших (включно зі взаємодіями).
  Порядок НЕ важливий. Стандарт SPSS/SAS. ✓
  Коли: більшість випадків, незбалансовані зі взаємодіями

Тип IV:
  Для незбалансованих з ПРОПУЩЕНИМИ КЛІТИНКАМИ.
  Рідкісний у польових дослідах.

РЕКОМЕНДАЦІЯ:
  Збалансований -> будь-який тип
  Незбалансований + взаємодії -> Тип III
  Незбалансований + без взаємодій -> Тип II
  Пропущені клітинки -> Тип IV
"""}

    H["Кореляційний аналіз"] = {"icon":"🔗","short":"Пірсон vs Спірмен, поправки, теплова карта","text":"""
КОРЕЛЯЦІЙНИЙ АНАЛІЗ

СТРУКТУРА ДАНИХ (два варіанти):
  А) Перший рядок = назва показника, нижче - числові значення
  Б) Перша колонка = назва показника, праворуч - значення

МЕТОДИ:
  Авто (рекомендовано): Shapiro-Wilk по кожному показнику
    хоч один ненормальний -> Спірмен
    всі нормальні -> Пірсон

  Пірсон r: лінійний зв'язок, нормальний розподіл
    ⚠ При ненормальних даних -> програма попередить!

  Спірмен rho: монотонний зв'язок, будь-який розподіл

ПОПРАВКИ НА МНОЖИННІ ПОРІВНЯННЯ:
  Бонферроні: строга (FWER), p_adj = p x кількість_пар
  BH (Benjamini-Hochberg FDR): ліберальніша, більш потужна
  -> При > 10 показниках рекомендується BH

ТЕПЛОВА КАРТА - кожна клітинка:
  r = коефіцієнт | p = значущість після поправки | n = кількість пар
  * p < 0.05 | ** p < 0.01

ІНТЕРПРЕТАЦІЯ |r|:
  0.0-0.2: дуже слабкий | 0.2-0.4: слабкий | 0.4-0.6: помірний
  0.6-0.8: сильний | 0.8-1.0: дуже сильний
"""}

    H["Регресійний аналіз"] = {"icon":"📈","short":"7 моделей регресії, R², діагностика","text":"""
РЕГРЕСІЙНИЙ АНАЛІЗ

МОДЕЛІ:
  1. Лінійна:       y = a + b*x
  2. Квадратична:   y = a + b*x + c*x^2  (оптимум добрив)
  3. Кубічна:       y = a + b*x + c*x^2 + d*x^3
  4. Степенева:     y = a * x^b  (x > 0)
  5. Експоненційна: y = a * e^(b*x)
  6. Логарифмічна:  y = a + b*ln(x)  (x > 0)
  7. Логістична 4P: y = d + (a-d)/(1+(x/c)^b)  (S-крива)

ПОКАЗНИКИ ЯКОСТІ:
  R^2   = частка варіації пояснена моделлю (0-1; > 0.9 - добре)
  R^2adj = R^2 з урахуванням кількості параметрів
  RMSE  = середньоквадратична помилка (в одиницях y)
  F-тест = значущість моделі загалом

ДІАГНОСТИКА:
  Residuals vs Fitted: будь-який патерн -> модель неповна!
  Shapiro-Wilk залишків: p > 0.05 -> нормальні ✓
  Grubbs test: автоматичне виявлення викидів у залишках

ВВЕДЕННЯ ДАНИХ:
  Ліве поле: значення x | Праве поле: значення y
  "Вставити дані": два стовпці з Excel (x | y)
"""}

    H["ANCOVA"] = {"icon":"🎛️","short":"Коваріаційний аналіз: контроль зовнішніх факторів","text":"""
ANCOVA - КОВАРІАЦІЙНИЙ АНАЛІЗ

Мета: порівняти групи усуваючи вплив коваріати.

КОВАРІАТА - неперервна змінна яку вимірюєте але не контролюєте:
  Початкова висота рослин, pH грунту, вміст гумусу, опади

СТРУКТУРА:
  [Група] | [Коваріата 1] | [Коваріата 2] | [Y - залежна змінна]

ЗАХИСТ (8 перевірок):
  1. n >= 6 спостережень
  2. >= 2 груп
  3. >= 2 спостережень у кожній групі
  4. Мультиколінеарність коваріат (r > 0.95)
  5. Паралельність ліній регресії <- КЛЮЧОВА!
  6. Нормальність залишків
  7. Однорідність дисперсій
  8. Скориговані середні (LS Means)

⚠ КРИТИЧНА ПЕРЕДУМОВА - ПАРАЛЕЛЬНІСТЬ ЛІНІЙ:
  Тест взаємодії Група x Коваріата:
  p < alpha -> лінії НЕ паралельні -> ANCOVA ЗАБЛОКОВАНА!

РЕЗУЛЬТАТИ:
  Скориговані (LS) Means = середні при однаковій коваріаті у всіх групах
  Порівнюйте скориговані середні, а не нескориговані!
"""}

    H["MANOVA"] = {"icon":"🔢","short":"Багатовимірний аналіз кількох показників","text":"""
MANOVA - БАГАТОВИМІРНИЙ ДИСПЕРСІЙНИЙ АНАЛІЗ

Мета: порівняти групи за КІЛЬКОМА залежними змінними одночасно.

НАВІЩО НЕ КІЛЬКА ANOVA?
  4 ANOVA з alpha=0.05 -> 19% шанс хибної різниці!
  MANOVA контролює сімейну помилку.

⚠ КРИТИЧНА ВИМОГА: n > p У КОЖНІЙ ГРУПІ
  n = кількість спостережень; p = кількість DV
  n <= p в будь-якій групі -> ЖОРСТКЕ БЛОКУВАННЯ!

ЗАХИСТ (7 перевірок):
  1. n > p у кожній групі (блокування)
  2. >= 2 DV
  3. Мультиколінеарність DV (|r| > 0.90)
  4. Багатовимірна нормальність (Mardia)
  5. Однорідність коваріаційних матриць (Box M)
  6. Чотири тестові статистики
  7. Univariate тільки після значущого MANOVA

СТАТИСТИКИ:
  Wilks Lambda: стандартна
  Pillai Trace: найробустніша ★ (при порушеннях)
  Hotelling-Lawley: один домінантний ефект
  Roy GCR: найпотужніша, найменш робастна

ПОСЛІДОВНІСТЬ:
  Перевірте передумови -> Pillai Trace ->
  Якщо значущо -> univariate ANOVAs з Bonferroni (alpha/p)
  Якщо незначущо -> univariate НЕ інтерпретуються!
"""}

    H["Аналіз стабільності"] = {"icon":"🌍","short":"GGE biplot та Eberhart-Russell для GxE","text":"""
АНАЛІЗ СТАБІЛЬНОСТІ - GxE ВЗАЄМОДІЯ

GxE: відносна продуктивність сортів змінюється між середовищами.

СТРУКТУРА:
  Рядки = Генотипи (сорти, лінії)
  Стовпці = Середовища (роки, місця)
  Значення = середня врожайність

EBERHART-RUSSELL:
  bi (коефіцієнт стабільності):
    bi ~= 1.0 -> стабільний (середня реакція)
    bi > 1.2  -> адаптивний (реагує на покращення умов)
    bi < 0.8  -> консервативний (для бідних умов)

  s^2d (дисперсія відхилень):
    ~= 0 -> передбачуваний
    > 0  -> непередбачуваний

  Ідеал: висока середня + bi ~= 1 + s^2d ~= 0

GGE BIPLOT:
  Точки = генотипи
  Стрілки = середовища
  Близько до центру = стабільний генотип
  Близько до стрілки середовища = добре виконує там
"""}

    H["Розмір вибірки"] = {"icon":"🔢","short":"Скільки повторностей потрібно?","text":"""
КАЛЬКУЛЯТОР РОЗМІРУ ВИБІРКИ

Параметри:
  alpha: рівень значущості (зазвичай 0.05)
  Потужність (1-beta): зазвичай 0.80 або 0.90
  delta: мінімальна різниця яку хочете виявити (наприклад 0.5 т/га)
  sigma: стандартне відхилення (з попередніх дослідів)
  k: кількість варіантів

РЕЖИМИ:
  Порожнє поле "повторності" -> мінімальна кількість повторностей
  Введіть повторності -> досягнута потужність тесту

ТИПОВІ ЗНАЧЕННЯ (зернові):
  CV 10-15%, виявити 10% різниці -> зазвичай 3-4 повторності
  CV 20-25% -> 5-7 повторностей

⚠ Планувати з потужністю >= 0.80!
"""}

    H["Проект"] = {"icon":"💾","short":"Збереження та відкриття проектів (.sadp)","text":"""
УПРАВЛІННЯ ПРОЕКТАМИ

Формат: .sadp (JSON-текст, можна відкрити у блокноті)

ЩО ЗБЕРІГАЄТЬСЯ:
  ✓ Кількість та назви факторів
  ✓ Всі введені дані (назви варіантів і числа)
  ✓ Кількість стовпців (повторностей)

ДЛЯ КІЛЬКОХ ПОКАЗНИКІВ ОДНОГО ДОСЛІДУ:
  1. Введіть схему досліду та перший показник
  2. Збережіть: Файл -> Зберегти проект
  3. Аналіз -> збережіть звіт
  4. Видаліть числові дані, залиште назви варіантів
  5. Введіть наступний показник -> аналіз -> звіт
  Або збережіть окремий .sadp для кожного показника.

МЕНЮ ФАЙЛ:
  Зберегти проект (Ctrl+S)
  Відкрити проект (Ctrl+O)
  Очистити таблицю
  Завантажити Excel
"""}

    H["Графіки і налаштування"] = {"icon":"🎨","short":"Boxplot, Venn, теплова карта та їх налаштування","text":"""
ГРАФІЧНИЙ ЗВІТ

BOXPLOT (коробка з вусами):
  Верхній вус: Q3 + 1.5*IQR (або максимум)
  Верхній край коробки: Q3 (75-й перцентиль)
  Лінія: медіана (Q2)
  Нижній край коробки: Q1 (25-й перцентиль)
  Нижній вус: Q1 - 1.5*IQR (або мінімум)
  Кружечки поза вусами: викиди (outliers)

  Літери CLD над коробками:
    Однакові -> немає значущої різниці
    Різні -> є значуща різниця (p < alpha)

ДІАГРАМА ВЕННА (сила впливу):
  Кола = головні ефекти факторів
  Перетини кіл = взаємодії між факторами
  Сума всіх частин = 100%

НАЛАШТУВАННЯ (кнопка ⚙ у вікні графіків):
  Boxplot: шрифт, розмір, кольори коробки/медіани/вусів/викидів
  Venn: шрифт, прозорість, кольори кіл і тексту
  За замовчуванням: APA стиль

КОПІЮВАННЯ:
  "Копіювати PNG" -> 300 dpi -> Ctrl+V у Word
  (Windows only; macOS/Linux - toolbar matplotlib)

ТЕПЛОВА КАРТА (кореляція):
  В клітинці: r / p / n
  Налаштовується через ⚙: colormap, шрифти, кольори
"""}

    H["Позначення у звіті"] = {"icon":"📝","short":"Як читати таблиці і показники у звіті","text":"""
ПОЗНАЧЕННЯ У ЗВІТІ

ЗНАЧУЩІСТЬ:
  **  p < 0.01 (висока значущість)
  *   p < 0.05 (значущо)
  -   p >= 0.05 (не значущо)

ТАБЛИЦЯ ANOVA:
  SS  = сума квадратів
  df  = ступені свободи
  MS  = середній квадрат (SS/df)
  F   = критерій Фішера (MS_ефект / MS_залишок)
  p   = ймовірність (менше = значущіше)

СИЛА ВПЛИВУ (% від SS):
  Частка загальної варіації по кожному джерелу.
  Залишок включений. Сума всіх рядків = 100%.

PARTIAL eta^2 (розмір ефекту):
  eta^2 = SS_ефект / (SS_ефект + SS_залишок)
  < 0.01: дуже слабкий | 0.01-0.06: слабкий
  0.06-0.14: середній | > 0.14: сильний

R^2 (коефіцієнт детермінації):
  Частка варіації пояснена всією моделлю.
  R^2 = 0.90 -> 90% варіації пояснено ✓

CV% (коефіцієнт варіації):
  < 10%: відмінна точність | 10-15%: хороша
  15-20%: задовільна | > 20%: низька

НІР05:
  Якщо різниця > НІР05 -> статистично значуща.
"""}

_fill_help()

# ── Розширення довідки детальними поясненнями ───────────────
def _extend_help():
    H = HELP_CONTENT

    H["Описова статистика"] = {"icon":"📐","short":"Базові показники: середнє, SD, медіана, довірчий інтервал","text":"""
ЩО ТАКЕ ОПИСОВА СТАТИСТИКА?

Перш ніж виконувати будь-який статистичний аналіз, корисно описати ваші дані
за допомогою базових числових характеристик. Це допомагає:
  - Зрозуміти загальний характер даних (велике чи маленьке розкидання?)
  - Виявити можливі помилки у введенні (занадто великі або малі значення)
  - Оцінити чи підходять параметричні методи

ЯК ВВОДИТИ ДАНІ:
  Кожен стовпець = один показник (змінна).
  Перша клітинка стовпця = назва показника (текст).
  Решта клітинок = числові значення.

  Приклад:
  | Врожайність | Висота рослин | Маса 1000 зерен |
  |    4.2      |     95.3      |      38.2       |
  |    5.1      |    102.1      |      41.5       |
  |    4.8      |     98.7      |      39.8       |

ЩО ОЗНАЧАЄ КОЖЕН ПОКАЗНИК:

n (кількість спостережень):
  Кількість числових значень у вибірці. Чим більше n, тим надійніший аналіз.
  Мінімально рекомендовано n >= 5 для більшості тестів.

Середнє (Mean):
  Середньоарифметичне значення. Сума всіх значень поділена на їх кількість.
  Чутливе до викидів: одне дуже велике або мале значення може суттєво змінити середнє.

SD (стандартне відхилення):
  Показує наскільки в середньому значення відхиляються від середнього.
  Велике SD = великий розкид = висока варіабельність.
  Приблизно 68% значень знаходяться в межах Mean ± 1*SD (для нормального розподілу).

СП (стандартна похибка середнього, SE):
  SE = SD / sqrt(n)
  Показує точність оцінки середнього. Чим більше n, тим менше SE.
  Використовується для побудови довірчих інтервалів.

Мін / Макс:
  Найменше та найбільше значення у вибірці.
  Дуже корисні для виявлення помилок введення даних!

Медіана:
  Значення що ділить впорядкований ряд навпіл: 50% значень нижче, 50% вище.
  На відміну від середнього, стійка до викидів.
  Якщо середнє >> медіани -> дані правоскошені.
  Якщо середнє << медіани -> дані лівоскошені.

Q1 (перший квартиль, 25-й перцентиль):
  25% значень нижче цього рівня.

Q3 (третій квартиль, 75-й перцентиль):
  75% значень нижче цього рівня.
  IQR = Q3 - Q1 = міжквартильний розмах (стійка міра варіабельності).

CV% (коефіцієнт варіації):
  CV = SD / Mean x 100%
  Відносна мінливість у відсотках. Дозволяє порівнювати варіабельність
  показників з різними одиницями виміру.
  Для польових дослідів: CV < 10% = відмінна точність.

Асиметрія (Skewness):
  = 0: симетричний розподіл
  > 0: правостороння асиметрія (хвіст праворуч, більшість значень ліворуч)
  < 0: лівостороння асиметрія
  |асиметрія| > 1 = суттєве відхилення від нормального розподілу.

Ексцес (Kurtosis):
  Показує «гостроту» розподілу порівняно з нормальним.
  = 0: нормальний розподіл
  > 0: гостроверхий (більше значень поблизу середнього та більше викидів)
  < 0: пласковерхий

95% Довірчий інтервал (ДІ):
  Діапазон в якому знаходиться «справжнє» середнє генеральної сукупності
  з ймовірністю 95%.
  Якщо 95% ДІ двох груп не перетинаються -> є підстави вважати що середні різні.
  Але для строгого висновку потрібен статистичний тест!

SW p (Shapiro-Wilk p-значення):
  Тест нормальності розподілу.
  p > 0.05: дані відповідають нормальному розподілу (параметричні тести можна)
  p <= 0.05: дані не відповідають (розгляньте непараметричні тести)
"""}

    H["t-тест"] = {"icon":"🔀","short":"Порівняння двох груп: t-тест, Велш, Манн-Уітні, Вілкоксон","text":"""
ЩО ТАКЕ t-ТЕСТ?

t-тест — це статистичний тест для порівняння ДВОХ груп або вибірок.
Відповідає на питання: «Чи є різниця між середніми статистично значущою,
чи це просто випадкові коливання?»

ПРОГРАМА АВТОМАТИЧНО ОБИРАЄ ПРАВИЛЬНИЙ ТЕСТ:
  1. Перевіряє нормальність розподілу (Shapiro-Wilk) для кожної групи
  2. Перевіряє рівність дисперсій (тест Левена) для незалежних вибірок
  3. Обирає відповідний тест:
     Нормальний + рівні дисперсії -> t-тест Стьюдента
     Нормальний + нерівні дисперсії -> t-тест Велша (Welch)
     Ненормальний -> Mann-Whitney U або Wilcoxon

ТРИ РЕЖИМИ:

1. НЕЗАЛЕЖНІ ВИБІРКИ (2 різні групи):
   Порівнюємо дві незалежні групи.
   Приклад: врожайність сорту А проти сорту Б.
   ⚠ Спостереження у групах НЕ пов'язані між собою.

   Введення: Група 1 і Група 2 - числа через Enter або кому.

   Якщо дані нормальні:
     Рівні дисперсії (Левен p >= 0.05) -> t-тест Стьюдента (класичний)
     Нерівні дисперсії (Левен p < 0.05) -> t-тест Велша (Welch) - більш точний!
   Якщо ненормальні -> Mann-Whitney U (непараметричний аналог)

2. ПАРНІ ВИБІРКИ (до/після, або парні вимірювання):
   Ті самі об'єкти вимірюються двічі.
   Приклад: маса рослин до і після обробки; врожайність на тих самих ділянках у 2 роки.
   ⚠ Вимагає ОДНАКОВУ кількість спостережень у обох групах!
   ⚠ Порядок важливий: перше значення Групи 1 пов'язане з першим Групи 2.

   Якщо дані нормальні -> Парний t-тест
   Якщо ненормальні -> Wilcoxon signed-rank

3. ОДНА ВИБІРКА (проти відомого значення):
   Перевіряємо чи середнє вибірки відрізняється від заданого значення.
   Приклад: чи відрізняється врожайність від нормативного показника 5 т/га?

   Введіть відоме середнє μ₀ і значення вибірки.

ЯК ЧИТАТИ РЕЗУЛЬТАТИ:
  t (або U, W) - значення тестової статистики
  p - ймовірність отримати такий або більший результат якщо H₀ вірна
  p < 0.05: різниця значуща ✓
  p >= 0.05: різниця незначуща (але це не означає що груп немає!)

РОЗМІР ЕФЕКТУ для Mann-Whitney (Cliff's delta):
  Показує не лише ЧИ є різниця, але й НАСКІЛЬКИ вона велика.
  |delta| < 0.147: дуже слабкий (практично немає різниці)
  0.147-0.33: слабкий
  0.33-0.474: середній
  > 0.474: сильний (суттєва практична різниця)

ПОРАДА:
  Значущий p ≠ велика різниця! При великих n навіть мізерна різниця буде значущою.
  Завжди оцінюйте розмір ефекту разом з p-значенням.
"""}

    H["Описова статистика — боксплот"] = {"icon":"📦","short":"Як читати боксплот (діаграму коробку з вусами)","text":"""
БОКСПЛОТ (ДІАГРАМА КОРОБКА З ВУСАМИ)

Боксплот — це графічний спосіб відобразити розподіл даних
не залежно від кількості спостережень.

ЯК ЧИТАТИ:

        ╷  <- верхній вус: Q3 + 1.5*IQR
        │     (або максимальне значення якщо воно менше)
    ┌───┐
    │   │  <- верхній край коробки: Q3 (75-й перцентиль)
    │═══│  <- жирна лінія: МЕДІАНА (Q2, 50-й перцентиль)
    │   │  <- нижній край коробки: Q1 (25-й перцентиль)
    └───┘
        │
        ╵  <- нижній вус: Q1 - 1.5*IQR
           (або мінімальне значення якщо воно більше)

    ○   <- окремі точки: ВИКИДИ (outliers)
           значення далі ніж 1.5*IQR від коробки

ЩО ТАКЕ IQR?
  IQR = Q3 - Q1 = міжквартильний розмах.
  Вміщує 50% «середніх» значень.
  Чим більша коробка, тим більший розкид «типових» значень.

ЯК ПОРІВНЮВАТИ БОКСПЛОТИ:
  Коробки не перекриваються -> можлива значуща різниця
  Медіани дуже різні -> явна різниця між групами
  Коробка однієї групи всередині іншої -> групи схожі

ЛІТЕРИ CLD НАД БОКСПЛОТАМИ:
  Це результат пост-хок аналізу після ANOVA.
  Однакові літери -> немає значущої різниці між цими варіантами
  Різні літери -> є значуща різниця (p < alpha)
  Приклад: ab і a не різняться, ab і b не різняться, але a і b можуть різнятися!

ВИКИДИ (outliers):
  Значення далі 1.5*IQR від коробки.
  Можуть бути:
  - Помилками вимірювання (перевірте журнал польового досліду!)
  - Справжніми екстремальними значеннями (посуха, хвороба тощо)
  - Важливою біологічною інформацією
  Не видаляйте викиди без перевірки!
"""}

    H["Кластерний аналіз — детально"] = {"icon":"🌿","short":"Ієрархічна кластеризація та дендрограма","text":"""
КЛАСТЕРНИЙ АНАЛІЗ — ДЕТАЛЬНЕ ПОЯСНЕННЯ

Кластерний аналіз групує об'єкти (сорти, проби, ділянки) так,
щоб схожі між собою опинились в одному кластері.

КОЛИ ЗАСТОСОВУВАТИ:
  ✓ Класифікація сортів за комплексом ознак
  ✓ Групування ґрунтових проб за хімічним складом
  ✓ Виявлення природних груп у даних без попередньої класифікації

ВВЕДЕННЯ ДАНИХ:
  Перший стовпець: назва об'єкта (сорт, зразок тощо)
  Решта стовпців: числові ознаки (показники)
  Перший рядок: назви показників

  Приклад:
  | Сорт    | Висота | Врожайн. | Стійкість |
  | Поліська|  95.3  |   5.8    |    7.2    |
  | Київська| 102.1  |   6.4    |    8.1    |
  | Одеська |  88.5  |   5.1    |    6.8    |

  Програма автоматично СТАНДАРТИЗУЄ дані (z-оцінки) щоб показники
  з різними одиницями виміру мали однаковий вплив.

МЕТОДИ ЗЧЕПЛЕННЯ (linkage):
  ward:      Мінімізує внутрішньокластерну дисперсію.
             Найпопулярніший метод, зазвичай дає найкращі результати. ✓
  complete:  Дистанція між найдальшими об'єктами кластерів.
             Дає компактні кластери однакового розміру.
  average:   Середня дистанція між всіма парами.
             Компроміс між ward і complete.
  single:    Дистанція між найближчими об'єктами.
             Схильний до «ефекту ланцюга» (довгі ланцюжкові кластери).

ЯК ЧИТАТИ ДЕНДРОГРАМУ:
  Вертикальна вісь = відстань (несхожість).
  Гілки зливаються на рівні що відповідає відстані між кластерами.
  Чим вища точка злиття, тим менш схожі ці кластери.

  Щоб отримати k кластерів: проведіть горизонтальну лінію на відповідній висоті
  так щоб перетнути k вертикальних гілок.

ВИБІР КІЛЬКОСТІ КЛАСТЕРІВ k:
  k задається вручну. Як обрати?
  - Дивіться на дендрограму: де є великий «стрибок» у висоті злиття -> там природна межа
  - k = кількість природних груп у вашому досліді (сорти різного типу, ґрунтові зони тощо)
  - Типово для агрономічних досліджень: k = 2-5

КОЛІРНЕ КОДУВАННЯ НА ДЕНДРОГРАМІ:
  Різні кольори = різні кластери.
  Горизонтальна пунктирна лінія показує поріг відсікання для k кластерів.
"""}

    H["PCA — детально"] = {"icon":"🔮","short":"Як читати biplot та scree plot аналізу головних компонент","text":"""
АНАЛІЗ ГОЛОВНИХ КОМПОНЕНТ (PCA) — ДЕТАЛЬНО

Уявіть що у вас є 10 показників для кожного сорту.
Це важко уявити і проаналізувати. PCA стискає цю інформацію
до 2-3 нових «узагальнених» показників (головних компонент)
які описують більшу частину різноманітності у ваших даних.

КОЛИ ЗАСТОСОВУВАТИ:
  ✓ Багато показників (> 5-7) для кожного об'єкта
  ✓ Хочете виявити природне групування об'єктів
  ✓ Хочете зрозуміти які показники «ходять разом»
  ✓ Як крок перед MANOVA якщо n <= p

ВВЕДЕННЯ ДАНИХ:
  Перша колонка: мітка об'єкта (необов'язково, назва сорту тощо)
  Решта колонок: числові показники
  Перший рядок: назви показників

SCREE PLOT (ліворуч):
  Стовпчасти: % дисперсії пояснений кожною компонентою.
  Лінія (червона): кумулятивний % пояснення.
  Горизонтальна пунктирна: 80% поріг.

  Скільки компонент залишити? «Правило ліктя»:
  Знайдіть точку де графік різко змінює нахил (стає плоским) ->
  це і є оптимальна кількість компонент.
  Зазвичай: PC1+PC2 разом пояснюють 70-85% -> достатньо для аналізу.

BIPLOT (посередині):
  Поєднує розташування об'єктів (точки) і змінних (стрілки) на одному графіку.

  Точки (об'єкти/сорти):
    Близькі точки = схожі об'єкти за всім комплексом показників.
    Далекі точки = дуже різні об'єкти.
    Кластери точок = природні групи.

  Стрілки (змінні/показники):
    Довга стрілка = показник добре описується цими компонентами.
    Напрямок стрілки = в якому напрямку зростає цей показник.
    Стрілки в одному напрямку = показники корелюють (ростуть разом).
    Стрілки протилежних напрямків = показники обернено корельовані.
    Стрілки під прямим кутом = показники не пов'язані.

    Об'єкт близький до стрілки = він має відносно більше значення цього показника.

ТЕПЛОВА КАРТА НАВАНТАЖЕНЬ (праворуч):
  Показує як кожен вихідний показник «вкладається» в кожну головну компоненту.
  Великі значення (темно-зелені або темно-червоні) = сильний зв'язок.
  PC1 зазвичай = «загальний розмір» або «загальна продуктивність».
  PC2 = наступна за важливістю вісь незалежна від PC1.
"""}

    H["Повторні виміри ANOVA"] = {"icon":"⏱️","short":"Аналіз динамічних вимірювань одних і тих самих об'єктів","text":"""
ДИСПЕРСІЙНИЙ АНАЛІЗ ПОВТОРНИХ ВИМІРЮВАНЬ

КОЛИ ЗАСТОСОВУВАТИ:
  Одні й ті самі суб'єкти (рослини, тварини, ділянки) вимірюються КІЛЬКА РАЗІВ:
  - У різні моменти часу (висота рослин через кожні 2 тижні)
  - За різних умов (доза добрив A, потім B, потім C)
  - До і після (більш ніж 2 точки -> потрібен повторний ANOVA; якщо 2 точки -> парний t-тест)

  ✓ Динаміка росту рослин
  ✓ Зміна вмісту поживних речовин по фазах вегетації
  ✓ Відповідь на послідовні обробки

  ⚠ ВІДМІНА від звичайного ANOVA: тут спостереження НЕ незалежні
  (одна рослина вимірюється кілька разів -> між вимірами є зв'язок).

ВВЕДЕННЯ ДАНИХ:
  Рядки = суб'єкти (рослини, ділянки тощо)
  Стовпці = часові точки або умови (T1, T2, T3 ...)
  Перший рядок заголовків (синій) = назви часових точок

  Приклад (висота рослин, см):
  | Суб'єкт | Тиждень1 | Тиждень2 | Тиждень3 | Тиждень4 |
  | Ділянка1|   15.2   |   28.4   |   45.1   |   58.3   |
  | Ділянка2|   14.8   |   26.9   |   43.7   |   56.8   |

ЩО ПОКАЗУЮТЬ РЕЗУЛЬТАТИ:

SS (суми квадратів):
  SS_time = варіація пояснена часом (основний ефект)
  SS_subj = варіація між суб'єктами (усувається з помилки -> підвищує точність!)
  SS_error = залишкова варіація

F-тест для «time»:
  p < 0.05: є значуща динаміка (показник змінюється через час)
  p >= 0.05: немає значущої динаміки

Partial η² (розмір ефекту часу):
  Показує яку частку варіації пояснює фактор «час».
  > 0.14: сильний ефект (виразна динаміка).

ГРАФІК MEANS ± SE:
  Показує як середнє значення змінюється у часі.
  SE (смужки похибок) показують точність оцінки середнього.
  Чим менші смужки, тим точніше середнє визначено.

ПОСТ-ХОК (Бонферроні):
  Після значущого F виконуються попарні порівняння часових точок.
  Показує ЯКІ САМЕ пари часових точок відрізняються.
  p_adj = скориговане p (Бонферроні = множення на кількість пар).

НОРМАЛЬНІСТЬ РІЗНИЦЬ:
  Перевіряється нормальність різниць між парами часових точок.
  p > 0.05: нормальний -> результати надійні.
  p <= 0.05: розгляньте непараметричний аналог (тест Фрідмана).
"""}

    H["Аналіз головних компонент"] = H["PCA — детально"]
    H["Кластерний аналіз — пояснення"] = H["Кластерний аналіз — детально"]
    H["t-тест / Манн-Уітні"] = H["t-тест"]

_extend_help()


