# sad_descriptive_ttest.py — Описова статистика, t-тест
# -*- coding: utf-8 -*-
from sad_common import *

class DescriptiveWindow:
    """Описова статистика — окремий модуль."""

    HELP_TEXT = """
ОПИСОВА СТАТИСТИКА — ПОКРОКОВА ІНСТРУКЦІЯ
══════════════════════════════════════════

ЩО ЦЕ І НАВІЩО?
  Описова статистика — перший крок будь-якого аналізу.
  Вона описує ваші дані числовими характеристиками і
  допомагає виявити аномалії до проведення статистичних тестів.

КРОК 1. ПІДГОТОВКА ДАНИХ
  Кожен стовпець = один показник (змінна).
  Кожен рядок = одне спостереження.
  Двічі клікніть на синій заголовок стовпця щоб задати
  назву показника (наприклад: «Врожайність, т/га»).

  Приклад:
  | Врожайність | Висота | Маса 1000 зерен |
  |    4.2      |  98.5  |      38.2       |
  |    5.1      | 103.2  |      41.5       |

КРОК 2. ЗАПУСК
  Натисніть «▶ Аналіз».
  Мінімум: 2 значення у стовпці для розрахунку SD і SE.

КРОК 3. ІНТЕРПРЕТАЦІЯ ТАБЛИЦІ РЕЗУЛЬТАТІВ

  n — кількість числових значень у стовпці.
    Менше 5 — результати ненадійні.

  Середнє (Mean) — середньоарифметичне.
    Чутливе до викидів: одне екстремальне значення
    може суттєво змінити середнє.

  SD (стандартне відхилення) — середнє відхилення
    від середнього. Велике SD = великий розкид.

  СП (SE, стандартна похибка середнього):
    SE = SD / √n
    Показує точність оцінки середнього.
    Чим більше n, тим менше SE.

  Мін / Макс — найменше і найбільше значення.
    Перевіряйте на помилки введення!

  Медіана — середнє значення впорядкованого ряду.
    Стійка до викидів. Якщо Середнє >> Медіани —
    розподіл правоскошений.

  Q1 / Q3 — 25-й і 75-й перцентилі.
    IQR = Q3 - Q1 (міжквартильний розмах).

  CV% (коефіцієнт варіації):
    CV = SD/Середнє × 100%
    Оцінка точності польового досліду:
    < 10%: відмінна | 10-15%: хороша
    15-20%: задовільна | > 20%: низька

  Асиметрія (Skewness):
    = 0: симетричний розподіл
    > 0: правостороня асиметрія (хвіст праворуч)
    < 0: лівостороня асиметрія

  Ексцес (Kurtosis):
    = 0: нормальний розподіл
    > 0: гостроверхий (більше значень поблизу середнього)
    < 0: пласковерхий

  95% ДІ — довірчий інтервал середнього.
    З ймовірністю 95% «справжнє» середнє знаходиться
    в цьому діапазоні.

  SW p (Shapiro-Wilk):
    Тест нормальності розподілу.
    p > 0.05: розподіл нормальний ✓
    p ≤ 0.05: розподіл ненормальний ⚠
    При n > 50 тест може бути надто чутливим —
    оцінюйте QQ-графік візуально.

КРОК 4. ГРАФІКИ

  Боксплот (коробка з вусами):
    Показує розподіл кожного показника.
    Коробка: Q1 - Q3 | Лінія: медіана
    Вуса: Q1-1.5×IQR до Q3+1.5×IQR
    Точки поза вусами: викиди (outliers)

  QQ-графік (квантиль-квантиль):
    Перевірка нормальності візуально.
    Точки лежать на прямій → нормальний розподіл ✓
    Точки відхиляються від прямої → ненормальний ⚠
"""

    def __init__(self, parent, gs):
        self.win = tk.Toplevel(parent)
        self.win.title("Описова статистика")
        self.win.geometry("1000x640"); set_icon(self.win)
        self.gs = dict(gs)
        self._bp_fig  = None   # боксплот
        self._qq_fig  = None   # QQ-графіки
        self._bp_gs   = {      # налаштування боксплоту
            "font_family": gs.get("font_family","Times New Roman"),
            "font_size": 11,
            "box_color":    gs.get("box_color","#ffffff"),
            "median_color": gs.get("median_color","#c62828"),
            "whisker_color":gs.get("whisker_color","#000000"),
            "flier_color":  gs.get("flier_color","#555555"),
        }
        self._qq_gs   = {      # налаштування QQ
            "font_family": gs.get("font_family","Times New Roman"),
            "font_size": 9,
            "pt_color":    "#4c72b0",
            "line_color":  "#c62828",
        }
        self._build()

    # ── Побудова вікна ───────────────────────────────────────
    def _build(self):
        # ── Меню ──
        mb = tk.Menu(self.win); self.win.config(menu=mb)
        fm = tk.Menu(mb, tearoff=0)
        fm.add_command(label="Завантажити Excel", command=self._load_excel)
        mb.add_cascade(label="Файл", menu=fm)

        # ── Панель інструментів ──
        tb = tk.Frame(self.win, padx=6, pady=5); tb.pack(fill=tk.X)

        tk.Button(tb, text="▶ Аналіз", bg="#c62828", fg="white",
                  font=("Times New Roman", 13),
                  command=self._analyze).pack(side=tk.LEFT, padx=4)

        # Налаштування — спадне меню
        mb2 = tk.Menubutton(tb, text="⚙ Налаштування ▾",
                            font=("Times New Roman", 11),
                            relief=tk.RAISED, bd=2)
        mb2.pack(side=tk.LEFT, padx=4)
        sm = tk.Menu(mb2, tearoff=0)
        sm.add_command(label="Додати рядок",      command=self._add_row)
        sm.add_command(label="Видалити рядок",    command=self._del_row)
        sm.add_separator()
        sm.add_command(label="Додати стовпець",   command=self._add_col)
        sm.add_command(label="Видалити стовпець", command=self._del_col)
        sm.add_separator()
        sm.add_command(label="🗑 Очистити таблицю", command=self._clear_table)
        sm.add_separator()
        sm.add_command(label="💾 Зберегти проект", command=self._save_proj)
        sm.add_command(label="📂 Відкрити проект", command=self._load_proj)
        mb2["menu"] = sm

        tk.Button(tb, text="Вставити з буфера",
                  font=("Times New Roman", 11),
                  command=self._paste).pack(side=tk.LEFT, padx=4)
        tk.Button(tb, text="📚 Довідка", bg="#1a4b8c", fg="white",
                  font=("Times New Roman", 11),
                  command=self._show_help).pack(side=tk.LEFT, padx=4)
        tk.Label(tb,
                 text="Двічі клікніть синій заголовок щоб перейменувати показник",
                 font=("Times New Roman", 9), fg="#666").pack(side=tk.LEFT, padx=10)

        # ── Таблиця ──
        tf = tk.Frame(self.win); tf.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        self.rows = 20; self.cols = 8
        canvas = tk.Canvas(tf); canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(tf, orient="vertical", command=canvas.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y); canvas.configure(yscrollcommand=sb.set)
        self.inner = tk.Frame(canvas)
        canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: canvas.config(scrollregion=canvas.bbox("all")))
        self.win.bind("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)),"units"))
        self._canvas = canvas

        # Заголовки (сині, з перейменуванням)
        self.header_vars   = []
        self.header_labels = []
        for j in range(self.cols):
            var = tk.StringVar(value=f"Показник {j+1}")
            self.header_vars.append(var)
            lbl = tk.Label(self.inner, textvariable=var,
                           relief=tk.RIDGE, width=13, cursor="hand2",
                           bg="#1a4b8c", fg="white",
                           font=("Times New Roman", 11, "bold"))
            lbl.grid(row=0, column=j, padx=2, pady=2, sticky="nsew")
            lbl.bind("<Double-Button-1>", lambda e, idx=j: self._rename_col(idx))
            self.header_labels.append(lbl)

        self.entries = []
        for i in range(self.rows):
            row_ = []
            for j in range(self.cols):
                e = tk.Entry(self.inner, width=13, font=("Times New Roman", 11),
                             highlightthickness=1, highlightbackground="#c0c0c0")
                e.grid(row=i+1, column=j, padx=2, pady=2)
                row_.append(e)
            self.entries.append(row_)
        _bind_nav(self.entries, self.win)
        _bind_fill_handle(self.entries, self.win)

    # ── Перейменування стовпця ───────────────────────────────
    def _rename_col(self, idx):
        dlg = tk.Toplevel(self.win); dlg.title("Перейменувати показник")
        dlg.resizable(False, False); dlg.grab_set()
        tk.Label(dlg, text=f"Назва показника {idx+1}:",
                 font=("Times New Roman", 12)).pack(padx=16, pady=14)
        var = tk.StringVar(value=self.header_vars[idx].get())
        e = tk.Entry(dlg, textvariable=var, font=("Times New Roman", 12), width=28)
        e.pack(padx=16, pady=4); e.select_range(0, tk.END); e.focus_set()
        def apply():
            nm = var.get().strip()
            if nm: self.header_vars[idx].set(nm)
            dlg.destroy()
        tk.Button(dlg, text="ОК", bg="#c62828", fg="white",
                  font=("Times New Roman", 12), command=apply).pack(pady=(4, 14))
        dlg.bind("<Return>", lambda ev: apply()); center_win(dlg)

    # ── Управління таблицею ───────────────────────────────────
    def _add_row(self):
        i = len(self.entries); row_ = []
        for j in range(self.cols):
            e = tk.Entry(self.inner, width=13, font=("Times New Roman", 11),
                         highlightthickness=1, highlightbackground="#c0c0c0")
            e.grid(row=i+1, column=j, padx=2, pady=2); row_.append(e)
        self.entries.append(row_); self.rows += 1
        _bind_nav(self.entries, self.win)
        _bind_fill_handle(self.entries, self.win)
        self._canvas.config(scrollregion=self._canvas.bbox("all"))

    def _del_row(self):
        if not self.entries: return
        for e in self.entries.pop(): e.destroy()
        self.rows -= 1
        self._canvas.config(scrollregion=self._canvas.bbox("all"))

    def _add_col(self):
        ci = self.cols; self.cols += 1
        var = tk.StringVar(value=f"Показник {ci+1}")
        self.header_vars.append(var)
        lbl = tk.Label(self.inner, textvariable=var, relief=tk.RIDGE, width=13,
                       cursor="hand2", bg="#1a4b8c", fg="white",
                       font=("Times New Roman", 11, "bold"))
        lbl.grid(row=0, column=ci, padx=2, pady=2, sticky="nsew")
        lbl.bind("<Double-Button-1>", lambda e, idx=ci: self._rename_col(idx))
        self.header_labels.append(lbl)
        for i, row_ in enumerate(self.entries):
            e = tk.Entry(self.inner, width=13, font=("Times New Roman", 11),
                         highlightthickness=1, highlightbackground="#c0c0c0")
            e.grid(row=i+1, column=ci, padx=2, pady=2); row_.append(e)
        _bind_nav(self.entries, self.win)
        _bind_fill_handle(self.entries, self.win)

    def _del_col(self):
        if self.cols <= 1: return
        self.header_labels.pop().destroy(); self.header_vars.pop()
        for row_ in self.entries: row_.pop().destroy()
        self.cols -= 1

    def _clear_table(self):
        if not messagebox.askyesno("Очистити", "Видалити всі числові дані?\n(Назви стовпців залишаться)"): return
        for row in self.entries:
            for e in row: e.delete(0, tk.END)

    def _save_proj(self):
        generic_save_project(self.win, "descriptive", self.header_vars, self.entries)

    def _load_proj(self):
        d = generic_load_project(self.win)
        if d is None: return
        headers = d.get("headers", []); rd = d.get("rows_data", [])
        while self.cols < len(headers): self._add_col()
        for j, h in enumerate(headers):
            if j < len(self.header_vars): self.header_vars[j].set(h)
        while len(self.entries) < len(rd): self._add_row()
        for i, rv in enumerate(rd):
            for j, v in enumerate(rv):
                if i < len(self.entries) and j < len(self.entries[i]):
                    self.entries[i][j].delete(0, tk.END); self.entries[i][j].insert(0, v)

    # ── Вставка і завантаження ───────────────────────────────
    def _paste(self):
        """Вставити з буфера обміну. Починає з активної клітинки або (0,0)."""
        try: data = self.win.clipboard_get()
        except Exception:
            messagebox.showwarning("Буфер порожній",
                "Буфер обміну порожній або не містить тексту.\n"
                "Скопіюйте дані з Excel (Ctrl+C) і спробуйте знову."); return
        if not data.strip(): return
        w = self.win.focus_get()
        # Знаходимо позицію активної клітинки
        pos = (0, 0)
        if isinstance(w, tk.Entry):
            for i, row_ in enumerate(self.entries):
                for j, e in enumerate(row_):
                    if e is w: pos = (i, j); break
        try: data = self.win.clipboard_get()
        except Exception: return
        r0, c0 = pos
        for ir, line in enumerate(data.splitlines()):
            for jc, val in enumerate(line.split("\t")):
                rr = r0+ir; cc = c0+jc
                while rr >= len(self.entries): self._add_row()
                if cc >= self.cols: continue
                self.entries[rr][cc].delete(0, tk.END)
                self.entries[rr][cc].insert(0, val.strip())

    def _load_excel(self):
        if not HAS_OPENPYXL: messagebox.showerror("","pip install openpyxl"); return
        path = filedialog.askopenfilename(filetypes=[("Excel","*.xlsx *.xlsm"),("All","*.*")])
        if not path: return
        try:
            wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
            raw = list(wb.active.iter_rows(values_only=True)); wb.close()
        except Exception as ex: messagebox.showerror("", str(ex)); return
        if not raw: return
        nc = max(len(r) for r in raw)
        while self.cols < nc: self._add_col()
        while len(self.entries) < len(raw): self._add_row()
        for i, row in enumerate(raw):
            for j, v in enumerate(row):
                if j >= self.cols: break
                self.entries[i][j].delete(0, tk.END)
                self.entries[i][j].insert(0, "" if v is None else str(v).replace(",","."))

    # ── Довідка ──────────────────────────────────────────────
    def _show_help(self):
        win = tk.Toplevel(self.win)
        win.title("Довідка — Описова статистика")
        win.geometry("700x640"); set_icon(win)
        frm = tk.Frame(win); frm.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        vsb = ttk.Scrollbar(frm, orient="vertical"); vsb.pack(side=tk.RIGHT, fill=tk.Y)
        txt = tk.Text(frm, wrap="word", font=("Times New Roman", 11),
                      yscrollcommand=vsb.set, relief=tk.FLAT,
                      bg="#fafafa", padx=10, pady=8, cursor="arrow")
        txt.pack(fill=tk.BOTH, expand=True)
        vsb.config(command=txt.yview)
        txt.insert("1.0", self.HELP_TEXT.strip())
        txt.configure(state="disabled")
        txt.bind("<MouseWheel>", lambda e: txt.yview_scroll(int(-1*(e.delta/120)), "units"))
        tk.Button(win, text="Закрити", command=win.destroy,
                  font=("Times New Roman", 11)).pack(pady=6)

    # ── Аналіз ───────────────────────────────────────────────
    def _analyze(self):
        from scipy.stats import skew, kurtosis
        # Назви беремо з заголовків; дані — з клітинок
        names = []; data_cols = []
        for j in range(self.cols):
            col_name = self.header_vars[j].get().strip() or f"Показник {j+1}"
            col_vals = []
            for row in self.entries:
                v = row[j].get().strip() if j < len(row) else ""
                if not v: continue
                try: col_vals.append(float(v.replace(",",".")))
                except Exception: continue
            if len(col_vals) >= 2:
                names.append(col_name)
                data_cols.append(np.array(col_vals, dtype=float))

        if not data_cols:
            messagebox.showwarning("Замало даних",
                "Не знайдено числових даних.\n"
                "Переконайтесь що введено числа і кожен стовпець містить ≥ 2 значення."); return

        headers = ["Показник","n","Середнє","SD","СП","Мін","Макс","Медіана",
                   "Q1","Q3","CV%","Асиметрія","Ексцес","95% ДІ нижній","95% ДІ верхній","SW p"]
        rows = []
        for nm, arr in zip(names, data_cols):
            a = arr[~np.isnan(arr)]; n = len(a)
            if n < 2: rows.append([nm, n] + ["–"]*14); continue
            m  = float(np.mean(a)); sd = float(np.std(a, ddof=1))
            se = sd / math.sqrt(n)
            ci_lo = m - float(t_dist.ppf(0.975, n-1)) * se
            ci_hi = m + float(t_dist.ppf(0.975, n-1)) * se
            sk = float(skew(a)); ku = float(kurtosis(a))
            q1  = float(np.percentile(a, 25)); q3 = float(np.percentile(a, 75))
            cv  = sd/m*100 if m != 0 else np.nan
            try: _, sw_p = shapiro(a)
            except Exception: sw_p = np.nan
            rows.append([nm, n, fmt(m,3), fmt(sd,3), fmt(se,3),
                         fmt(float(np.min(a)),3), fmt(float(np.max(a)),3),
                         fmt(float(np.median(a)),3), fmt(q1,3), fmt(q3,3),
                         fmt(cv,2), fmt(sk,3), fmt(ku,3),
                         fmt(ci_lo,3), fmt(ci_hi,3), fmt(sw_p,4)])

        self._show_result(headers, rows, data_cols, names)

    # ── Результати ───────────────────────────────────────────
    def _show_result(self, headers, rows, arrays, names):
        win = tk.Toplevel(self.win)
        win.title("Описова статистика — Результати")
        win.geometry("1300x760"); set_icon(win)
        if not hasattr(self, "_bp_gs"): self._bp_gs = {}
        if not hasattr(self, "_qq_gs"): self._qq_gs = {}

        main = tk.Frame(win); main.pack(fill=tk.BOTH, expand=True)
        sidebar = tk.Frame(main, width=210, bg="#2c3e50")
        sidebar.pack(side=tk.LEFT, fill=tk.Y); sidebar.pack_propagate(False)
        content = tk.Frame(main); content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(sidebar, text="ОПИСОВА\nСТАТИСТИКА", bg="#2c3e50", fg="#ecf0f1",
                 font=("Times New Roman",12,"bold"), pady=12, justify="center").pack(fill=tk.X)

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

        tbl_frame = tk.Frame(content)
        bp_frame  = tk.Frame(content)
        qq_frame  = tk.Frame(content)

        b_tbl = _sidebar_btn("📋 Таблиця",     "Описові статистики")
        b_bp  = _sidebar_btn("📊 Боксплоти",   "Розподіл показників")
        b_qq  = _sidebar_btn("📈 QQ-графіки",  "Перевірка нормальності")

        # Панелі з графіками будуються ЛІНИВО — лише при першому переході
        # на них, коли їхній фрейм вже показаний (packed) на екрані. Якщо
        # будувати matplotlib-графік у ще прихованому (unpacked) фреймі
        # заздалегідь, на деяких системах він потім не перемальовується
        # коректно після показу — звідси й «графік не відображається».
        built = {"bp": False, "qq": False}

        def _open_tbl():
            _show_panel(tbl_frame, b_tbl)
        def _open_bp():
            _show_panel(bp_frame, b_bp)
            if not built["bp"]:
                bp_frame.update_idletasks()
                self._build_desc_bp_panel(bp_frame, arrays, names)
                built["bp"] = True
        def _open_qq():
            _show_panel(qq_frame, b_qq)
            if not built["qq"]:
                qq_frame.update_idletasks()
                self._build_desc_qq_panel(qq_frame, arrays, names)
                built["qq"] = True

        b_tbl.configure(command=_open_tbl)
        b_bp.configure( command=_open_bp)
        b_qq.configure( command=_open_qq)

        self._build_desc_table_panel(tbl_frame, headers, rows, win)

        _show_panel(tbl_frame, b_tbl)

    def _build_desc_table_panel(self, frame, headers, rows, win):
        tb = tk.Frame(frame, padx=6, pady=5); tb.pack(fill=tk.X)
        tk.Button(tb, text="📋 Копіювати таблицю", font=("Times New Roman",11),
                  command=lambda: self._copy_table(win, headers, rows)
                  ).pack(side=tk.LEFT, padx=4)
        tbl_frm, _ = make_tv(frame, headers, rows, min_col=80)
        tbl_frm.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    def _build_desc_bp_panel(self, frame, arrays, names):
        tb = tk.Frame(frame, padx=6, pady=5); tb.pack(fill=tk.X)
        tk.Button(tb, text="📋 Копіювати PNG", font=("Times New Roman",11),
                  command=lambda: self._copy_fig(self._bp_fig)
                  ).pack(side=tk.LEFT, padx=4)
        tk.Button(tb, text="⚙ Налаштування", font=("Times New Roman",11),
                  command=lambda: self._restyle_bp(frame, arrays, names)
                  ).pack(side=tk.LEFT, padx=4)
        self._bp_frame = tk.Frame(frame); self._bp_frame.pack(fill=tk.BOTH, expand=True)
        self._bp_arrays = arrays; self._bp_names = names
        self._draw_boxes(self._bp_frame, arrays, names)

    def _build_desc_qq_panel(self, frame, arrays, names):
        tb = tk.Frame(frame, padx=6, pady=5); tb.pack(fill=tk.X)
        tk.Button(tb, text="📋 Копіювати PNG", font=("Times New Roman",11),
                  command=lambda: self._copy_fig(self._qq_fig)
                  ).pack(side=tk.LEFT, padx=4)
        tk.Button(tb, text="⚙ Налаштування", font=("Times New Roman",11),
                  command=lambda: self._restyle_qq(frame, arrays, names)
                  ).pack(side=tk.LEFT, padx=4)
        tk.Label(tb, text="Точки на прямій → нормальний розподіл ✓   |   "
                          "Відхилення від прямої → ненормальний ⚠",
                 font=("Times New Roman", 9), fg="#555").pack(side=tk.LEFT, padx=8)
        self._qq_frame = tk.Frame(frame); self._qq_frame.pack(fill=tk.BOTH, expand=True)
        self._qq_arrays = arrays; self._qq_names = names
        self._draw_qq(self._qq_frame, arrays, names)

    def _copy_table(self, win, headers, rows):
        """Копіює таблицю у буфер обміну у форматі TSV (для вставки у Excel/Word)."""
        lines = ["\t".join(str(h) for h in headers)]
        for row in rows:
            lines.append("\t".join("" if v is None else str(v) for v in row))
        text = "\n".join(lines)
        win.clipboard_clear(); win.clipboard_append(text)
        messagebox.showinfo("Скопійовано",
            "Таблицю скопійовано у буфер обміну.\n"
            "Вставте у Word або Excel через Ctrl+V.")

    # ── Боксплот ─────────────────────────────────────────────
    def _draw_boxes(self, frame, arrays, names):
        for w in frame.winfo_children(): w.destroy()
        gs = self._bp_gs
        ff  = gs.get("font_family", "Times New Roman")
        fz  = gs.get("font_size", 11)
        n   = len(arrays)
        fig = Figure(figsize=(10, 6), dpi=100)
        ax  = fig.add_subplot(111)
        clean_data = [a[~np.isnan(a)] for a in arrays]
        try:
            # matplotlib >= 3.9: параметр перейменовано на tick_labels,
            # старий "labels" остаточно прибрано в 3.11
            bp = ax.boxplot(clean_data, tick_labels=names,
                             patch_artist=True, widths=0.55)
        except TypeError:
            # matplotlib < 3.9: tick_labels ще не існує
            bp = ax.boxplot(clean_data, labels=names,
                             patch_artist=True, widths=0.55)
        for patch in bp["boxes"]:    patch.set(facecolor=gs.get("box_color","#ffffff"))
        for line  in bp["medians"]:  line.set(color=gs.get("median_color","#c62828"), linewidth=2)
        for line  in bp["whiskers"]+bp["caps"]:
            line.set(color=gs.get("whisker_color","#000000"))
        for fl    in bp["fliers"]:
            fl.set(markerfacecolor=gs.get("flier_color","#555555"), marker="o", markersize=4)
        ax.set_ylabel("Значення", fontsize=fz, fontfamily=ff)
        ax.set_title("Boxplot показників", fontsize=fz+1, fontfamily=ff)
        ax.tick_params(axis="x", labelsize=max(8, fz-1))
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.tight_layout()
        self._bp_fig = fig
        embed_figure(fig, frame)

    def _restyle_bp(self, win, arrays, names):
        """Dedicated boxplot settings dialog — no KeyError on missing DEF_GS keys."""
        dlg = tk.Toplevel(win); dlg.title("Налаштування боксплоту")
        dlg.resizable(False, False); set_icon(dlg); dlg.grab_set()
        gs = self._bp_gs
        frm = tk.Frame(dlg, padx=16, pady=14); frm.pack()
        rb_f = ("Times New Roman", 12)

        ff_var = tk.StringVar(value=gs.get("font_family","Times New Roman"))
        fz_var = tk.IntVar(value=gs.get("font_size", 11))
        col_box = [gs.get("box_color",    "#ffffff")]
        col_med = [gs.get("median_color", "#c62828")]
        col_wh  = [gs.get("whisker_color","#000000")]
        col_fl  = [gs.get("flier_color",  "#555555")]

        tk.Label(frm, text="Шрифт:", font=rb_f).grid(row=0, column=0, sticky="w", pady=5)
        ttk.Combobox(frm, textvariable=ff_var,
                     values=["Times New Roman","Arial","Calibri","Georgia","Verdana"],
                     state="readonly", width=22).grid(row=0, column=1, sticky="w", padx=8)
        tk.Label(frm, text="Розмір шрифту:", font=rb_f).grid(row=1, column=0, sticky="w", pady=5)
        tk.Spinbox(frm, from_=7, to=24, textvariable=fz_var, width=6).grid(row=1, column=1, sticky="w", padx=8)

        btn_refs = {}
        color_cfg = [
            ("Колір коробки:", col_box, "box"),
            ("Колір медіани:", col_med, "med"),
            ("Колір вусів:",   col_wh,  "wh"),
            ("Колір викидів:", col_fl,  "fl"),
        ]
        for ri, (lbl, col_lst, key) in enumerate(color_cfg):
            tk.Label(frm, text=lbl, font=rb_f).grid(row=2+ri, column=0, sticky="w", pady=5)
            btn = tk.Button(frm, width=6, relief=tk.SUNKEN, bg=col_lst[0])
            btn.grid(row=2+ri, column=1, sticky="w", padx=8)
            btn_refs[key] = (btn, col_lst)
            def _pick(c=col_lst, b=btn):
                ch = colorchooser.askcolor(color=c[0], parent=dlg, title="Виберіть колір")
                if ch and ch[1]: c[0] = ch[1]; b.configure(bg=ch[1])
            btn.configure(command=_pick)

        def apply():
            self._bp_gs.update({
                "font_family":   ff_var.get(),
                "font_size":     fz_var.get(),
                "box_color":     col_box[0],
                "median_color":  col_med[0],
                "whisker_color": col_wh[0],
                "flier_color":   col_fl[0],
            })
            self._draw_boxes(self._bp_frame, arrays, names)
            dlg.destroy()

        bf = tk.Frame(frm); bf.grid(row=6, column=0, columnspan=2, pady=(14,0))
        tk.Button(bf, text="OK", bg="#c62828", fg="white",
                  font=rb_f, command=apply).pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text="Скасувати", font=rb_f, command=dlg.destroy).pack(side=tk.LEFT)
        center_win(dlg)

    # ── QQ-графіки ───────────────────────────────────────────
    def _draw_qq(self, frame, arrays, names):
        from scipy.stats import probplot
        for w in frame.winfo_children(): w.destroy()
        gs    = self._qq_gs
        ff    = gs.get("font_family", "Times New Roman")
        fz    = gs.get("font_size", 9)
        pt_c  = gs.get("pt_color",   "#4c72b0")
        ln_c  = gs.get("line_color", "#c62828")
        n     = len(arrays); cols_ = min(n, 4); rows_n = math.ceil(n / cols_)
        fig   = Figure(figsize=(cols_*2.5+0.5, rows_n*2.5+0.5), dpi=100)
        for i, (arr, nm) in enumerate(zip(arrays, names)):
            a = arr[~np.isnan(arr)]
            if len(a) < 3: continue
            ax  = fig.add_subplot(rows_n, cols_, i+1)
            res = probplot(a, dist="norm")
            ax.plot(res[0][0], res[0][1], "o", markersize=4, color=pt_c, alpha=0.8)
            ax.plot(res[0][0], res[1][1] + res[1][0]*res[0][0], "-", color=ln_c, lw=1.5)
            ax.set_title(nm, fontsize=fz+1, fontfamily=ff)
            ax.set_xlabel("Теоретичні квантилі", fontsize=fz, fontfamily=ff)
            ax.set_ylabel("Вибіркові квантилі",  fontsize=fz, fontfamily=ff)
            ax.yaxis.grid(True, linestyle="--", alpha=0.35)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.tight_layout()
        self._qq_fig = fig
        embed_figure(fig, frame)

    def _restyle_qq(self, win, arrays, names):
        """Простий діалог налаштувань QQ-графіків."""
        dlg = tk.Toplevel(win); dlg.title("Налаштування QQ-графіків")
        dlg.resizable(False, False); set_icon(dlg); dlg.grab_set()
        frm = tk.Frame(dlg, padx=16, pady=14); frm.pack()
        rb_f = ("Times New Roman", 12)

        ff_var  = tk.StringVar(value=self._qq_gs.get("font_family","Times New Roman"))
        fz_var  = tk.IntVar(value=self._qq_gs.get("font_size", 9))
        pt_col  = [self._qq_gs.get("pt_color","#4c72b0")]
        ln_col  = [self._qq_gs.get("line_color","#c62828")]

        tk.Label(frm, text="Шрифт:", font=rb_f).grid(row=0, column=0, sticky="w", pady=4)
        ttk.Combobox(frm, textvariable=ff_var,
                     values=["Times New Roman","Arial","Calibri","Georgia"],
                     state="readonly", width=20).grid(row=0, column=1, sticky="w", padx=8)
        tk.Label(frm, text="Розмір шрифту:", font=rb_f).grid(row=1, column=0, sticky="w", pady=4)
        tk.Spinbox(frm, from_=6, to=18, textvariable=fz_var, width=6).grid(row=1, column=1, sticky="w", padx=8)

        pt_btn = tk.Button(frm, width=6, relief=tk.SUNKEN, bg=pt_col[0])
        ln_btn = tk.Button(frm, width=6, relief=tk.SUNKEN, bg=ln_col[0])

        def pick_pt():
            c = colorchooser.askcolor(color=pt_col[0], parent=dlg, title="Колір точок")
            if c and c[1]: pt_col[0]=c[1]; pt_btn.configure(bg=c[1])
        def pick_ln():
            c = colorchooser.askcolor(color=ln_col[0], parent=dlg, title="Колір лінії")
            if c and c[1]: ln_col[0]=c[1]; ln_btn.configure(bg=c[1])

        tk.Label(frm, text="Колір точок:", font=rb_f).grid(row=2, column=0, sticky="w", pady=4)
        pt_btn.configure(command=pick_pt); pt_btn.grid(row=2, column=1, sticky="w", padx=8)
        tk.Label(frm, text="Колір лінії:", font=rb_f).grid(row=3, column=0, sticky="w", pady=4)
        ln_btn.configure(command=pick_ln); ln_btn.grid(row=3, column=1, sticky="w", padx=8)

        def apply():
            self._qq_gs.update({"font_family": ff_var.get(), "font_size": fz_var.get(),
                                 "pt_color": pt_col[0], "line_color": ln_col[0]})
            self._draw_qq(self._qq_frame, arrays, names); dlg.destroy()
        bf = tk.Frame(frm); bf.grid(row=4, column=0, columnspan=2, pady=(14,0))
        tk.Button(bf, text="OK", bg="#c62828", fg="white",
                  font=rb_f, command=apply).pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text="Скасувати", font=rb_f, command=dlg.destroy).pack(side=tk.LEFT)
        center_win(dlg)

    # ── Копіювання PNG ───────────────────────────────────────
    def _copy_fig(self, fig):
        if fig is None:
            messagebox.showwarning("", "Спочатку побудуйте графік."); return
        ok, msg = _copy_fig_to_clipboard(fig)
        if ok: messagebox.showinfo("", "Графік скопійовано.\nВставте у Word через Ctrl+V.")
        else:   messagebox.showwarning("", f"Помилка: {msg}")




# ═══════════════════════════════════════════════════════════════
# T-TEST MODULE
# ═══════════════════════════════════════════════════════════════
class TTestWindow:
    """t-тест / Критерій Манна-Уітні."""

    HELP_TEXT = """
t-ТЕСТ / МАНН-УІТНІ — ПОКРОКОВА ІНСТРУКЦІЯ
════════════════════════════════════════════

ЩО ЦЕЙ АНАЛІЗ РОБИТЬ?
  Порівнює ДВІ групи і відповідає:
  «Чи є різниця між середніми статистично значущою,
  чи це просто випадкові коливання?»

  Програма автоматично обирає правильний тест:
  ✓ Нормальний розподіл + рівні дисперсії → t-тест Стьюдента
  ✓ Нормальний розподіл + нерівні дисперсії → t-тест Велша
  ✓ Ненормальний розподіл → Манн-Уітні (непараметричний)
  ✓ Парні + нормальний → Парний t-тест
  ✓ Парні + ненормальний → Вілкоксон

РЕЖИМ 1: НЕЗАЛЕЖНІ ВИБІРКИ
  Дві різні групи. Спостереження не пов'язані.
  Приклад: врожайність Сорту А і Сорту Б.
  Введіть значення у поля «Група 1» і «Група 2».

РЕЖИМ 2: ПАРНІ ВИБІРКИ (до/після)
  Ті самі об'єкти вимірюються двічі.
  Приклад: маса рослин до і після обробки.
  ВАЖЛИВО: порядок значень має бути однаковим!
  Перше значення Групи 1 пов'язане з першим Групи 2.
  Кількість значень в обох групах має бути однаковою.

РЕЖИМ 3: ОДНА ВИБІРКА (проти відомого μ)
  Порівняння середнього вибірки з відомим значенням.
  Приклад: чи відрізняється врожайність від нормативу 5 т/га?
  Введіть дані у «Група 1» і вкажіть μ₀.

ЯК ВВОДИТИ ДАНІ:
  Значення через кому, пробіл або кожне з нового рядка.
  Приклад: 4.2, 5.1, 4.8, 5.3, 4.9
  Або:
  4.2
  5.1
  4.8

ІНТЕРПРЕТАЦІЯ РЕЗУЛЬТАТІВ:
  p < α → різниця значуща ✓ (реальна різниця між групами)
  p ≥ α → різниця незначуща ✗ (можлива випадковість)

  РОЗМІР ЕФЕКТУ (Cliff's delta для Манн-Уітні):
  |δ| < 0.15: дуже слабкий
  0.15-0.33:  слабкий
  0.33-0.47:  середній
  > 0.47:     сильний

  Значущий p ≠ велика різниця!
  При великих n навіть мізерна різниця буде значущою.
  Завжди оцінюйте розмір ефекту разом з p.

SHAPIRO-WILK:
  p > 0.05 → нормальний розподіл (параметричний тест)
  p ≤ 0.05 → ненормальний (непараметричний тест)
"""

    def __init__(self, parent):
        self.win = tk.Toplevel(parent)
        self.win.title("t-тест / Критерій Манна-Уітні")
        self.win.geometry("1080x640"); set_icon(self.win)
        self._build()

    def _build(self):
        # ── Toolbar ──────────────────────────────────────────
        top = tk.Frame(self.win, padx=8, pady=6); top.pack(fill=tk.X)
        tk.Button(top, text="▶ Виконати", bg="#c62828", fg="white",
                  font=("Times New Roman",13),
                  command=self._run).pack(side=tk.LEFT, padx=4)
        tk.Label(top, text="α:", font=("Times New Roman",12)).pack(side=tk.LEFT, padx=(10,2))
        self.alpha_var = tk.StringVar(value="0.05")
        ttk.Combobox(top, textvariable=self.alpha_var, values=["0.01","0.05","0.10"],
                     state="readonly", width=7).pack(side=tk.LEFT)
        tk.Button(top, text="Вставити з буфера",
                  font=("Times New Roman",11),
                  command=self._paste).pack(side=tk.LEFT, padx=8)
        tk.Button(top, text="📋 Копіювати результат",
                  font=("Times New Roman",11),
                  command=self._copy_result).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="📚 Довідка", bg="#1a4b8c", fg="white",
                  font=("Times New Roman",11),
                  command=self._show_help).pack(side=tk.LEFT, padx=8)

        # ── Дві колонки: зліва — введення, справа — звіт ─────
        main = tk.Frame(self.win); main.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0,8))

        left = tk.Frame(main, width=380)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))
        left.pack_propagate(False)

        right = tk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ── ЛІВА КОЛОНКА: тип тесту + дані ───────────────────
        rf = ("Times New Roman",12)
        tk.Label(left, text="Тип тесту:",
                 font=("Times New Roman",12,"bold")).pack(anchor="w", pady=(0,4))
        self.test_var = tk.StringVar(value="ind")
        tests = [("Незалежні вибірки (2 різні групи)", "ind"),
                 ("Парні вибірки (до/після, однакові об'єкти)", "paired"),
                 ("Одна вибірка (проти відомого μ₀)", "one")]
        for txt, val in tests:
            tk.Radiobutton(left, text=txt, variable=self.test_var, value=val,
                           font=rf, command=self._update_ui,
                           wraplength=360, justify="left", anchor="w"
                           ).pack(fill=tk.X, anchor="w")

        tk.Label(left, text="Група 1 / Вибірка:",
                 font=("Times New Roman",12)).pack(anchor="w", pady=(14,2))
        self.e1 = tk.Text(left, width=34, height=6, font=("Times New Roman",11))
        self.e1.pack(fill=tk.X)
        tk.Label(left, text="Через кому, пробіл або кожне значення з нового рядка",
                 font=("Times New Roman",9), fg="#666", wraplength=360, justify="left"
                 ).pack(anchor="w")

        self.lbl2 = tk.Label(left, text="Група 2:", font=("Times New Roman",12))
        self.e2 = tk.Text(left, width=34, height=6, font=("Times New Roman",11))

        # «Відоме середнє» — виразний виділений блок, а не дрібне поле збоку
        self.mu_frame = tk.Frame(left, bg="#eef3f8", padx=12, pady=10,
                                 highlightbackground="#1a4b8c", highlightthickness=1)
        self.lbl_mu = tk.Label(self.mu_frame, text="Відоме (гіпотетичне) середнє μ₀:",
                               font=("Times New Roman",12,"bold"),
                               bg="#eef3f8", fg="#1a4b8c", wraplength=330, justify="left")
        self.lbl_mu.pack(anchor="w")
        self.e_mu = tk.Entry(self.mu_frame, width=14, font=("Times New Roman",14))
        self.e_mu.insert(0, "0")
        self.e_mu.pack(anchor="w", pady=(6,0))

        # ── ПРАВА КОЛОНКА: результати (на всю висоту) ────────
        tk.Label(right, text="Результати:",
                 font=("Times New Roman",12,"bold")).pack(anchor="w", pady=(0,4))
        res_frm = tk.Frame(right); res_frm.pack(fill=tk.BOTH, expand=True)
        vsb = ttk.Scrollbar(res_frm, orient="vertical"); vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.res_txt = tk.Text(res_frm, wrap="word",
                               font=("Times New Roman",12),
                               yscrollcommand=vsb.set,
                               relief=tk.FLAT, bg="#f8f8f8",
                               padx=12, pady=10, cursor="arrow",
                               state="disabled")
        self.res_txt.pack(fill=tk.BOTH, expand=True)
        vsb.config(command=self.res_txt.yview)

        self._update_ui()

    def _update_ui(self):
        t = self.test_var.get()
        if t == "one":
            self.lbl2.pack_forget(); self.e2.pack_forget()
            self.mu_frame.pack(fill=tk.X, pady=(12,2))
        else:
            self.mu_frame.pack_forget()
            self.lbl2.pack(anchor="w", pady=(12,2))
            self.e2.pack(fill=tk.X)
            txt = ("Група 2 (парна — той самий порядок що й Група 1):"
                   if t == "paired" else "Група 2:")
            self.lbl2.configure(text=txt)

    def _parse(self, widget):
        import re
        txt = widget.get("1.0", tk.END).strip().replace(",",".")
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", txt)
        return np.array([float(x) for x in nums], dtype=float)

    def _set_result(self, text, color="#000000"):
        self.res_txt.configure(state="normal")
        self.res_txt.delete("1.0", tk.END)
        self.res_txt.insert("1.0", text)
        self.res_txt.configure(state="disabled", fg=color)

    def _show_help(self):
        win = tk.Toplevel(self.win)
        win.title("Довідка — t-тест / Манн-Уітні")
        win.geometry("680x640"); set_icon(win)
        frm = tk.Frame(win); frm.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        vsb = ttk.Scrollbar(frm, orient="vertical"); vsb.pack(side=tk.RIGHT, fill=tk.Y)
        txt = tk.Text(frm, wrap="word", font=("Times New Roman",11),
                      yscrollcommand=vsb.set, relief=tk.FLAT,
                      bg="#fafafa", padx=10, pady=8, cursor="arrow")
        txt.pack(fill=tk.BOTH, expand=True); vsb.config(command=txt.yview)
        txt.insert("1.0", self.HELP_TEXT.strip()); txt.configure(state="disabled")
        txt.bind("<MouseWheel>",
                 lambda e: txt.yview_scroll(int(-1*(e.delta/120)),"units"))
        tk.Button(win, text="Закрити", command=win.destroy,
                  font=("Times New Roman",11)).pack(pady=6)

    def _run(self):
        from scipy.stats import ttest_ind, ttest_rel, ttest_1samp
        alpha = float(self.alpha_var.get())
        x1 = self._parse(self.e1)
        t = self.test_var.get()
        p = None  # ініціалізуємо явно
        if len(x1) < 2:
            self._set_result("Група 1 потребує ≥ 2 значень.", "#c62828"); return

        lines = []
        sep = "─" * 46
        lines.append(f"n₁ = {len(x1)}   Середнє₁ = {fmt(np.mean(x1),4)}   SD₁ = {fmt(np.std(x1,ddof=1),4)}")

        try: _, sw1 = shapiro(x1)
        except Exception: sw1 = np.nan
        normal1 = not math.isnan(sw1) and sw1 > 0.05
        lines.append(f"Shapiro–Wilk (Група 1): W = {fmt(sw1,4)}"
                     f"  →  {'✓ нормальний' if normal1 else '⚠ НЕ нормальний'}")

        if t == "one":
            try: mu0 = float(self.e_mu.get())
            except Exception: mu0 = 0.0
            stat, p = ttest_1samp(x1, mu0)
            lines.append(sep)
            lines.append(f"Одновибірковий t-тест (μ₀ = {mu0})")
            lines.append(f"t = {fmt(stat,4)},   df = {len(x1)-1},   p = {fmt(p,4)}")
        else:
            x2 = self._parse(self.e2)
            if len(x2) < 2:
                self._set_result("Група 2 потребує ≥ 2 значень.", "#c62828"); return
            try: _, sw2 = shapiro(x2)
            except Exception: sw2 = np.nan
            normal2 = not math.isnan(sw2) and sw2 > 0.05
            lines.append(f"n₂ = {len(x2)}   Середнє₂ = {fmt(np.mean(x2),4)}   SD₂ = {fmt(np.std(x2,ddof=1),4)}")
            lines.append(f"Shapiro–Wilk (Група 2): W = {fmt(sw2,4)}"
                         f"  →  {'✓ нормальний' if normal2 else '⚠ НЕ нормальний'}")
            lines.append(sep)

            if t == "paired":
                if len(x1) != len(x2):
                    self._set_result(
                        "Парний тест вимагає однакового розміру вибірок.\n"
                        f"Група 1: {len(x1)} значень, Група 2: {len(x2)} значень.",
                        "#c62828"); return
                if normal1 and normal2:
                    stat, p = ttest_rel(x1, x2)
                    lines.append(f"Парний t-тест")
                    lines.append(f"t = {fmt(stat,4)},   df = {len(x1)-1},   p = {fmt(p,4)}")
                else:
                    stat, p = wilcoxon(x1, x2, zero_method="wilcox",
                                       alternative="two-sided", mode="auto")
                    lines.append("Критерій Вілкоксона (знакових рангів)")
                    lines.append(f"W = {fmt(stat,4)},   p = {fmt(p,4)}")
            else:
                try: lev_s, lev_p = levene(x1, x2, center='median')
                except Exception: lev_p = np.nan
                equal_var = not math.isnan(lev_p) and lev_p >= 0.05
                lines.append(f"Тест Левена: p = {fmt(lev_p,4)}"
                             f"  →  {'✓ рівні дисперсії' if equal_var else '⚠ нерівні дисперсії'}")

                if normal1 and normal2:
                    stat, p = ttest_ind(x1, x2, equal_var=equal_var)
                    n1, n2 = len(x1), len(x2)
                    if not equal_var:
                        df_w = ((np.var(x1,ddof=1)/n1 + np.var(x2,ddof=1)/n2)**2 /
                                ((np.var(x1,ddof=1)/n1)**2/(n1-1) +
                                 (np.var(x2,ddof=1)/n2)**2/(n2-1)))
                        test_name = "t-тест Велша (нерівні дисперсії)"
                    else:
                        df_w = n1+n2-2
                        test_name = "t-тест Стьюдента (незалежні)"
                    lines.append(f"{test_name}")
                    lines.append(f"t = {fmt(stat,4)},   df ≈ {fmt(df_w,1)},   p = {fmt(p,4)}")
                else:
                    U, p = mannwhitneyu(x1, x2, alternative="two-sided")
                    d = cliffs_d(x1, x2)
                    lines.append("Критерій Манна-Уітні (непараметричний)")
                    lines.append(f"U = {fmt(U,3)},   p = {fmt(p,4)}")
                    lines.append(f"Cliff's δ = {fmt(d,4)}   ({cliffs_lbl(abs(d))} ефект)")

        lines.append(sep)
        if p is not None and not math.isnan(p):
            sig = p < alpha
            lines.append(
                f"{'✓ Різниця ЗНАЧУЩА' if sig else '✗ Різниця НЕЗНАЧУЩА'}"
                f"   (p = {fmt(p,4)},  α = {alpha})")
            if sig and t != "one":
                x2_arr = self._parse(self.e2)
                diff = float(np.mean(x1) - np.mean(x2_arr)) if len(x2_arr) > 0 else float("nan")
                lines.append(f"Різниця середніх: {fmt(diff,4)}")
            elif sig and t == "one":
                try: mu0v = float(self.e_mu.get())
                except Exception: mu0v = 0.0
                lines.append(f"Різниця від μ₀: {fmt(float(np.mean(x1))-mu0v,4)}")

        self._set_result("\n".join(lines))

    def _paste(self):
        """Вставити дані з буфера у Групу 1 або Групу 2 залежно від фокусу."""
        try: data = self.win.clipboard_get()
        except Exception:
            messagebox.showwarning("Буфер порожній",
                "Скопіюйте дані з Excel і спробуйте знову."); return
        if not data.strip(): return
        # Нормалізуємо — замінюємо табуляції і пробіли на нові рядки
        data = data.replace("\t","\n").replace(",",".")
        w = self.win.focus_get()
        target = self.e2 if w is self.e2 else self.e1
        target.delete("1.0", tk.END)
        target.insert("1.0", data.strip())

    def _copy_result(self):
        text = self.res_txt.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("","Спочатку виконайте аналіз."); return
        self.win.clipboard_clear()
        self.win.clipboard_append(text)
        messagebox.showinfo("Скопійовано","Результат скопійовано у буфер обміну.")






# ═══════════════════════════════════════════════════════════════
# OUTLIER DETECTION
# ═══════════════════════════════════════════════════════════════
def detect_outliers_grubbs(arr, alpha=0.05):
    """Grubbs test for single outlier. Returns (idx_of_outlier or None, G, p)."""
    a = np.array(arr, dtype=float); n = len(a)
    if n < 3: return None, np.nan, np.nan
    m = np.mean(a); s = np.std(a, ddof=1)
    if s == 0: return None, np.nan, np.nan
    G = np.max(np.abs(a - m)) / s
    idx = int(np.argmax(np.abs(a - m)))
    # critical value via t-distribution
    t_crit = float(t_dist.ppf(1 - alpha/(2*n), n-2))
    G_crit = ((n-1)/math.sqrt(n)) * math.sqrt(t_crit**2 / (n-2+t_crit**2))
    p_approx = 2 * n * (1 - float(t_dist.cdf(G * math.sqrt(n) * math.sqrt(n-2) /
               math.sqrt(n-1+G**2*n/(n-1)), n-2))) if n > 2 else np.nan
    return (idx if G > G_crit else None), float(G), float(p_approx)

def detect_outliers_iqr(arr):
    """IQR method. Returns list of indices."""
    a = np.array(arr, dtype=float)
    q1, q3 = np.percentile(a, 25), np.percentile(a, 75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    return [i for i, v in enumerate(a) if v < lo or v > hi]


# ═══════════════════════════════════════════════════════════════
# REGRESSION MODULE
# ═══════════════════════════════════════════════════════════════
