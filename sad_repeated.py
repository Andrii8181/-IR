# sad_repeated.py — Повторні виміри, змішаний RM
# -*- coding: utf-8 -*-
from sad_common import *

class RepeatedMeasuresWindow:
    """Дисперсійний аналіз повторних вимірювань."""

    HELP_TEXT = """
АНАЛІЗ ПОВТОРНИХ ВИМІРЮВАНЬ — ПОКРОКОВА ІНСТРУКЦІЯ
══════════════════════════════════════════════════

ЩО ТАКЕ ПОВТОРНІ ВИМІРИ?
  Це ситуація коли одні й ті самі об'єкти (рослини, тварини, ділянки,
  дерева) вимірюються КІЛЬКА РАЗІВ:
  • У різні моменти часу (висота пагонів щомісяця)
  • За різних умов (доза А → доза Б → доза В)
  • По фазах вегетації (бутонізація, цвітіння, дозрівання)

  ВІДМІНА від звичайного ANOVA:
  Вимірювання одного об'єкта залежні між собою. Дерево що було
  вищим у травні — залишається відносно вищим у червні.
  Ігнорування цього зв'язку (звичайна ANOVA) завищує помилку
  і знижує шанс виявити реальну різницю.

КОЛИ ВИКОРИСТОВУВАТИ?
  ✓ Динаміка росту рослин або пагонів (вимірюєте ті самі рослини)
  ✓ Зміна вмісту поживних речовин по фазах вегетації
  ✓ Реакція на послідовні обробки (один об'єкт отримує всі обробки)
  ✓ Порівняння до/після (якщо > 2 точок; для 2 точок — парний t-тест)

КРОК 1. СТРУКТУРА ТАБЛИЦІ

  Перший стовпець: Назва суб'єкта (рослина, дерево, ділянка — текст)
  Решта стовпців: Вимірювання у кожній часовій точці (числа)
  Заголовки стовпців (сині): Назви часових точок або умов

  Приклад (ріст пагонів, 5 дерев, 4 вимірювання):
  | Дерево   | Травень | Червень | Липень | Серпень |
  | Дерево 1 |  12.3   |  24.5   |  38.1  |  45.2   |
  | Дерево 2 |  11.8   |  22.9   |  35.7  |  42.8   |
  | Дерево 3 |  13.1   |  26.2   |  40.3  |  48.6   |

  Перейменуйте заголовки часових точок:
  Подвійний клік на синій клітинці → введіть назву.

  Мінімум: 2 суб'єкти, 2 часові точки.

КРОК 2. ВИКОНАННЯ АНАЛІЗУ
  Натисніть «▶ Аналіз».
  Програма автоматично видаляє рядки з пропущеними даними.

КРОК 3. ІНТЕРПРЕТАЦІЯ РЕЗУЛЬТАТІВ

  Таблиця дисперсійного аналізу:

  SS_час — варіація пояснена зміною у часі (те що нас цікавить)
  SS_суб'єкти — варіація між суб'єктами (виноситься окремо!)
  SS_похибка — залишкова варіація

  F(df_час, df_похибка):
    p < α → є значуща динаміка у часі ✓
    p ≥ α → динаміка незначуща

  Partial η² (розмір ефекту):
    < 0.01: дуже слабкий | 0.01–0.06: слабкий
    0.06–0.14: середній  | > 0.14: сильний ← типово для росту

КРОК 4. POST-HOC АНАЛІЗ (після значущого F)

  Виконуються парні t-тести з поправкою Бонферроні.
  Показують ЯКІ САМЕ пари часових точок відрізняються.

  Приклад: «Травень vs Червень: p=0.003 *» → у червні показник
  значущо вищий ніж у травні.

КРОК 5. НОРМАЛЬНІСТЬ РІЗНИЦЬ

  Перевіряється Shapiro-Wilk для різниць між кожною парою точок.
  p > 0.05 → різниці нормальні → результати надійні ✓
  p ≤ 0.05 → розгляньте тест Фрідмана (непараметричний аналог)

КРОК 6. ГРАФІК ДИНАМІКИ (Середні ± СП)

  Показує як середнє значення змінюється у часі.
  Смужки похибок (СП) = стандартна похибка середнього.
  Чим менші смужки → тим точніше визначено середнє.
  S-подібний підйом → типовий ріст рослин.
  Плато → насичення (ріст сповільнився або зупинився).

ПОРАДА:
  Якщо у вас КІЛЬКА ВАРІАНТІВ (сортів, обробок) і ті самі
  об'єкти щороку — це двофакторний Repeated Measures:
  між-суб'єктний фактор (варіант) + всередині-суб'єктний (час).
  Для такого аналізу використовуйте ТРИФАКТОРНУ ANOVA де рік
  є одним з факторів — це загальноприйнята практика.
"""

    def __init__(self, parent, gs):
        self.win = tk.Toplevel(parent)
        self.win.title("Дисперсійний аналіз повторних вимірювань")
        self.win.geometry("940x660"); set_icon(self.win)
        self.gs = gs
        self._rm_fig = None
        self._rm_gs  = {
            "font_family":  "Times New Roman",
            "font_size":    10,
            "line_color":   "#4c72b0",
            "err_color":    "#c62828",
            "marker":       "o",
            "linewidth":    2.0,
            "markersize":   7,
            "show_grid":    True,
        }
        self._build()

    def _build(self):
        try:
            self._build_inner()
        except Exception as e:
            import traceback
            messagebox.showerror("Помилка ініціалізації",
                f"Помилка при побудові вікна:\n{traceback.format_exc()}")

    def _build_inner(self):
        # ── Toolbar ──────────────────────────────────────────
        top = tk.Frame(self.win, padx=8, pady=6); top.pack(fill=tk.X)
        tk.Button(top, text="▶ Аналіз", bg="#c62828", fg="white",
                  font=("Times New Roman", 13),
                  command=self._run).pack(side=tk.LEFT, padx=4)

        tk.Label(top, text="α:", font=("Times New Roman",12)).pack(side=tk.LEFT, padx=(10,2))
        self.alpha_var = tk.StringVar(value="0.05")
        ttk.Combobox(top, textvariable=self.alpha_var, values=["0.01","0.05","0.10"],
                     state="readonly", width=7).pack(side=tk.LEFT)

        # Налаштування — спадне меню
        mb2 = tk.Menubutton(top, text="⚙ Налаштування ▾",
                            font=("Times New Roman",11), relief=tk.RAISED, bd=2)
        mb2.pack(side=tk.LEFT, padx=4)
        sm = tk.Menu(mb2, tearoff=0)
        sm.add_command(label="Додати рядок",       command=self._add_row)
        sm.add_command(label="Видалити рядок",     command=self._del_row)
        sm.add_separator()
        sm.add_command(label="Додати стовпець",    command=self._add_col)
        sm.add_command(label="Видалити стовпець",  command=self._del_col)
        sm.add_separator()
        sm.add_command(label="💾 Зберегти проект", command=self._save_proj)
        sm.add_command(label="📂 Відкрити проект", command=self._load_proj)
        sm.add_separator()
        sm.add_command(label="🗑 Очистити таблицю", command=self._clear_table)
        mb2["menu"] = sm

        tk.Button(top, text="Вставити з буфера",
                  font=("Times New Roman",11),
                  command=self._paste).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="📚 Довідка", bg="#1a4b8c", fg="white",
                  font=("Times New Roman",11),
                  command=self._show_help).pack(side=tk.LEFT, padx=4)

        tk.Label(top,
                 text="Подвійний клік на заголовку часової точки → перейменувати",
                 font=("Times New Roman",9), fg="#666").pack(side=tk.LEFT, padx=8)

        # ── Таблиця ─────────────────────────────────────────
        mid = tk.Frame(self.win); mid.pack(fill=tk.BOTH, expand=True, padx=8)
        self.rows_n = 20; self.cols_n = 8
        self._canvas = tk.Canvas(mid)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(mid, orient="vertical", command=self._canvas.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.configure(yscrollcommand=sb.set)
        self.inner = tk.Frame(self._canvas)
        self._canvas.create_window((0,0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>",
                        lambda e: self._canvas.config(scrollregion=self._canvas.bbox("all")))
        self.win.bind("<MouseWheel>",
                      lambda e: self._canvas.yview_scroll(int(-1*(e.delta/120)),"units"))

        # Перший заголовок — «Суб'єкт» (фіксований)
        tk.Label(self.inner, text="Суб'єкт", relief=tk.RIDGE, width=13,
                 bg="#444444", fg="white",
                 font=("Times New Roman",11,"bold")).grid(row=0, column=0, padx=1, pady=1, sticky="nsew")

        # Заголовки часових точок (перейменовувані)
        self.col_vars = []; self.col_labels = []
        for j in range(1, self.cols_n):
            var = tk.StringVar(value=f"Точка {j}")
            self.col_vars.append(var)
            lbl = tk.Label(self.inner, textvariable=var, width=12, cursor="hand2",
                           bg="#1a4b8c", fg="white", relief=tk.RIDGE,
                           font=("Times New Roman",11,"bold"))
            lbl.grid(row=0, column=j, padx=1, pady=1, sticky="nsew")
            lbl.bind("<Double-Button-1>", lambda e, idx=j-1: self._rename_time_col(idx))
            self.col_labels.append(lbl)

        self.entries = []
        for i in range(self.rows_n):
            row_ = []
            for j in range(self.cols_n):
                e = tk.Entry(self.inner, width=13 if j==0 else 12,
                             font=("Times New Roman",11))
                e.grid(row=i+1, column=j, padx=1, pady=1)
                if j == 0:
                    e.bind("<KeyRelease>",
                           lambda ev: _autofit_col(self.entries, 0))
                row_.append(e)
            self.entries.append(row_)
        _bind_nav(self.entries, self.win)
        _bind_fill_handle(self.entries, self.win)

    # ── Перейменування часової точки ─────────────────────────
    def _rename_time_col(self, idx):
        dlg = tk.Toplevel(self.win); dlg.title("Перейменувати часову точку")
        dlg.resizable(False, False); dlg.grab_set()
        tk.Label(dlg, text=f"Назва часової точки {idx+1}:",
                 font=("Times New Roman",12)).pack(padx=16, pady=14)
        var = tk.StringVar(value=self.col_vars[idx].get())
        e = tk.Entry(dlg, textvariable=var, font=("Times New Roman",12), width=26)
        e.pack(padx=16, pady=4); e.select_range(0, tk.END); e.focus_set()
        def apply():
            nm = var.get().strip()
            if nm: self.col_vars[idx].set(nm)
            dlg.destroy()
        tk.Button(dlg, text="OK", bg="#c62828", fg="white",
                  font=("Times New Roman",12), command=apply).pack(pady=(4,14))
        dlg.bind("<Return>", lambda ev: apply()); center_win(dlg)

    # ── Управління таблицею ───────────────────────────────────
    def _add_row(self):
        i = self.rows_n; row_ = []
        for j in range(self.cols_n):
            e = tk.Entry(self.inner, width=13 if j==0 else 12,
                         font=("Times New Roman",11))
            e.grid(row=i+1, column=j, padx=1, pady=1)
            if j == 0:
                e.bind("<KeyRelease>", lambda ev: _autofit_col(self.entries, 0))
            row_.append(e)
        self.entries.append(row_); self.rows_n += 1
        _bind_nav(self.entries, self.win)
        _bind_fill_handle(self.entries, self.win)

    def _del_row(self):
        if not self.entries: return
        for e in self.entries.pop(): e.destroy()
        self.rows_n -= 1

    def _add_col(self):
        ci = self.cols_n; self.cols_n += 1
        var = tk.StringVar(value=f"Точка {ci}")
        self.col_vars.append(var)
        lbl = tk.Label(self.inner, textvariable=var, width=12, cursor="hand2",
                       bg="#1a4b8c", fg="white", relief=tk.RIDGE,
                       font=("Times New Roman",11,"bold"))
        lbl.grid(row=0, column=ci, padx=1, pady=1, sticky="nsew")
        lbl.bind("<Double-Button-1>", lambda e, idx=ci-1: self._rename_time_col(idx))
        self.col_labels.append(lbl)
        for i, row_ in enumerate(self.entries):
            e = tk.Entry(self.inner, width=12, font=("Times New Roman",11),
                         )
            e.grid(row=i+1, column=ci, padx=1, pady=1)
            row_.append(e)
        _bind_nav(self.entries, self.win)
        _bind_fill_handle(self.entries, self.win)

    def _del_col(self):
        if self.cols_n <= 3: return
        self.col_labels.pop().destroy(); self.col_vars.pop()
        for row_ in self.entries: row_.pop().destroy()
        self.cols_n -= 1

    def _clear_table(self):
        if not messagebox.askyesno("Очистити",
                "Видалити всі дані?\n(Заголовки залишаться)"): return
        for row in self.entries:
            for e in row: e.delete(0, tk.END)

    def _save_proj(self):
        path = filedialog.asksaveasfilename(
            parent=self.win, defaultextension=".sadp",
            filetypes=[("SAD проект","*.sadp"),("JSON","*.json")],
            title="Зберегти проект повторних вимірювань")
        if not path: return
        d = {
            "type": "repeated_measures", "version": APP_VER,
            "col_vars": [v.get() for v in self.col_vars],
            "rows_data": [[e.get() for e in row] for row in self.entries],
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("Збережено",
                f"Проект збережено:\n{path}\n\n"
                "Щоб додати нові дати: ⚙ → «Додати стовпець»")
        except Exception as ex:
            messagebox.showerror("Помилка збереження", str(ex))

    def _load_proj(self):
        path = filedialog.askopenfilename(
            parent=self.win,
            filetypes=[("SAD проект","*.sadp"),("JSON","*.json")],
            title="Відкрити проект повторних вимірювань")
        if not path: return
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception as ex:
            messagebox.showerror("Помилка відкриття", str(ex)); return
        col_vars = d.get("col_vars", [])
        rows_data = d.get("rows_data", [])
        n_cols_needed = 1 + len(col_vars)
        while self.cols_n < n_cols_needed: self._add_col()
        for i, nm in enumerate(col_vars):
            if i < len(self.col_vars): self.col_vars[i].set(nm)
        while len(self.entries) < len(rows_data): self._add_row()
        for i, row_vals in enumerate(rows_data):
            for j, v in enumerate(row_vals):
                if j < self.cols_n:
                    self.entries[i][j].delete(0, tk.END)
                    self.entries[i][j].insert(0, v)
        messagebox.showinfo("Завантажено",
            "Проект завантажено.\n\n"
            "Щоб додати нові дати: ⚙ → «Додати стовпець»\n"
            "і перейменуйте заголовок подвійним кліком.")

    def _paste(self):
        try: data = self.win.clipboard_get()
        except Exception:
            messagebox.showwarning("Буфер порожній",
                "Скопіюйте дані з Excel (Ctrl+C) і спробуйте знову."); return
        if not data.strip(): return
        pos = (0, 0)
        w = self.win.focus_get()
        if isinstance(w, tk.Entry):
            for i, row_ in enumerate(self.entries):
                for j, e in enumerate(row_):
                    if e is w: pos=(i,j); break
        r0, c0 = pos
        for ir, line in enumerate(data.splitlines()):
            if not line.strip(): continue
            while r0+ir >= len(self.entries): self._add_row()
            for jc, val in enumerate(line.split("\t")):
                cc = c0+jc
                if cc >= self.cols_n: continue
                self.entries[r0+ir][cc].delete(0,tk.END)
                self.entries[r0+ir][cc].insert(0, val.strip())
        _autofit_col(self.entries, 0)

    def _show_help(self):
        win = tk.Toplevel(self.win)
        win.title("Довідка — Повторні виміри ANOVA")
        win.geometry("720x680"); set_icon(win)
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

    # ── Налаштування графіка ──────────────────────────────────
    def _restyle_rm(self, win, time_names, data_arr, n, ph_results=None, alpha=0.05):
        dlg = tk.Toplevel(win); dlg.title("Налаштування графіка")
        dlg.resizable(False, False); set_icon(dlg); dlg.grab_set()
        gs = self._rm_gs
        frm = tk.Frame(dlg, padx=16, pady=14); frm.pack()
        rb_f = ("Times New Roman",12)

        ff_v  = tk.StringVar(value=gs["font_family"])
        fz_v  = tk.IntVar(value=gs["font_size"])
        lw_v  = tk.DoubleVar(value=gs["linewidth"])
        ms_v  = tk.IntVar(value=gs["markersize"])
        mk_v  = tk.StringVar(value=gs["marker"])
        gr_v  = tk.BooleanVar(value=gs["show_grid"])
        lc_ref = [gs["line_color"]]; ec_ref = [gs["err_color"]]

        rows_cfg = [
            ("Шрифт:",          "combo",  ff_v, ["Times New Roman","Arial","Calibri","Georgia"]),
            ("Розмір шрифту:",  "spin",   fz_v, (7,18)),
            ("Товщина лінії:",  "scale",  lw_v, (0.5,5.0)),
            ("Розмір маркера:", "spin",   ms_v, (3,20)),
            ("Тип маркера:",    "combo",  mk_v, ["o","s","^","D","v","*","+"]),
            ("Показати сітку:", "check",  gr_v, None),
        ]
        for ri, (lbl, wt, var, opts) in enumerate(rows_cfg):
            tk.Label(frm, text=lbl, font=rb_f).grid(row=ri, column=0, sticky="w", pady=4)
            if wt=="combo":
                ttk.Combobox(frm, textvariable=var, values=opts,
                             state="readonly", width=18).grid(row=ri, column=1, sticky="w", padx=8)
            elif wt=="spin":
                tk.Spinbox(frm, from_=opts[0], to=opts[1], textvariable=var,
                           width=7).grid(row=ri, column=1, sticky="w", padx=8)
            elif wt=="scale":
                tk.Scale(frm, from_=opts[0], to=opts[1], resolution=0.1,
                         orient="horizontal", variable=var,
                         length=160).grid(row=ri, column=1, sticky="w", padx=8)
            elif wt=="check":
                tk.Checkbutton(frm, variable=var).grid(row=ri, column=1, sticky="w", padx=8)

        base_r = len(rows_cfg)
        btn_lc = tk.Button(frm, width=6, relief=tk.SUNKEN, bg=lc_ref[0])
        btn_ec = tk.Button(frm, width=6, relief=tk.SUNKEN, bg=ec_ref[0])
        for ri2, (lbl, ref, btn) in enumerate([
            ("Колір лінії:", lc_ref, btn_lc),
            ("Колір смужок похибок:", ec_ref, btn_ec)
        ]):
            tk.Label(frm, text=lbl, font=rb_f).grid(row=base_r+ri2, column=0, sticky="w", pady=4)
            btn.grid(row=base_r+ri2, column=1, sticky="w", padx=8)
            def _pick(r=ref, b=btn):
                ch = colorchooser.askcolor(color=r[0], parent=dlg)
                if ch and ch[1]: r[0]=ch[1]; b.configure(bg=ch[1])
            btn.configure(command=_pick)

        def apply():
            self._rm_gs.update({
                "font_family": ff_v.get(), "font_size": fz_v.get(),
                "linewidth":   lw_v.get(), "markersize": ms_v.get(),
                "marker":      mk_v.get(), "show_grid": gr_v.get(),
                "line_color":  lc_ref[0],  "err_color": ec_ref[0],
            })
            dlg.destroy()
            self._redraw_rm(win, time_names, data_arr, n, ph_results, alpha)

        bf = tk.Frame(frm); bf.grid(row=base_r+2, column=0, columnspan=2, pady=(14,0))
        tk.Button(bf, text="OK (застосувати)", bg="#c62828", fg="white",
                  font=rb_f, command=apply).pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text="Скасувати", font=rb_f, command=dlg.destroy).pack(side=tk.LEFT)
        center_win(dlg)

    # ── Виконання аналізу ─────────────────────────────────────
    def _run(self):
        # Назви часових точок — лише заповнені стовпці
        all_time = [v.get().strip() or f"Т{i+1}" for i, v in enumerate(self.col_vars)]
        raw = [[e.get().strip() for e in row] for row in self.entries]
        raw = [r for r in raw if any(v for v in r)]
        if not raw:
            messagebox.showwarning("Немає даних","Введіть дані у таблицю."); return

        # Знаходимо реально заповнені числові стовпці (1..cols_n-1)
        filled_t_cols = []
        for j in range(1, self.cols_n):
            for row in raw:
                v = row[j] if j < len(row) else ""
                if v:
                    try: float(v.replace(",",".")); filled_t_cols.append(j); break
                    except ValueError: pass

        if not filled_t_cols:
            messagebox.showwarning("Немає числових даних",
                "Введіть числові значення у стовпці часових точок (2 і далі)."); return

        time_names = [all_time[j-1] for j in filled_t_cols]
        k = len(time_names)

        subjects = []; data_rows = []
        for row in raw:
            subj = row[0].strip() if row and row[0].strip() else f"Суб'єкт {len(subjects)+1}"
            try: float(subj.replace(",",".")); continue
            except ValueError: pass
            vals = []
            for j in filled_t_cols:
                v = row[j].strip() if j < len(row) else ""
                if not v: vals.append(float("nan"))
                else:
                    try: vals.append(float(v.replace(",",".")))
                    except Exception: vals.append(float("nan"))
            if any(not math.isnan(v) for v in vals):
                subjects.append(subj); data_rows.append(vals)

        if len(data_rows) < 2:
            messagebox.showwarning("Замало суб'єктів",
                "Потрібно щонайменше 2 суб'єкти.\n\n"
                "Переконайтесь що:\n"
                "  • Перший стовпець містить назви (текст)\n"
                "  • Числа введені у стовпці 2 і далі"); return

        n_raw = len(data_rows)
        data = np.array(data_rows, dtype=float)
        mask_complete = ~np.any(np.isnan(data), axis=1)
        n_incomplete = int(np.sum(~mask_complete))
        data = data[mask_complete]
        subjects = [s for s, m in zip(subjects, mask_complete) if m]
        n = len(data)
        if n < 2:
            messagebox.showwarning("Замало повних даних",
                f"Після видалення рядків з пропущеними значеннями залишилось {n}.\n"
                "Заповніть пропущені значення або видаліть неповні рядки."); return
        if n_incomplete > 0:
            messagebox.showinfo("Пропущені дані",
                f"Видалено {n_incomplete} рядків з пропущеними значеннями.\n"
                f"Аналіз на {n} суб'єктах.")

        grand_mean = np.mean(data)
        subj_means = np.mean(data, axis=1)
        time_means = np.mean(data, axis=0)
        SS_total = float(np.sum((data - grand_mean)**2))
        SS_subj  = k * float(np.sum((subj_means - grand_mean)**2))
        SS_time  = n * float(np.sum((time_means - grand_mean)**2))
        SS_error = SS_total - SS_subj - SS_time
        df_time = k-1; df_subj = n-1; df_err = (k-1)*(n-1)
        MS_time = SS_time/df_time if df_time > 0 else float("nan")
        MS_err  = SS_error/df_err if df_err > 0 else float("nan")
        F = MS_time/MS_err if (not math.isnan(MS_err) and MS_err > 1e-12) else float("nan")
        p = float(1 - f_dist.cdf(F, df_time, df_err)) if not math.isnan(F) else float("nan")
        eta2_time = SS_time/(SS_time+SS_error) if (SS_time+SS_error) > 0 else float("nan")
        R2 = (SS_time+SS_subj)/SS_total if SS_total > 0 else float("nan")
        alpha = float(self.alpha_var.get())

        sw_ps = []
        for j in range(k):
            for jj in range(j+1, k):
                diff = data[:,j] - data[:,jj]
                try: _, p_sw = shapiro(diff)
                except Exception: p_sw = float("nan")
                sw_ps.append(p_sw)
        min_sw = min((pp for pp in sw_ps if not math.isnan(pp)), default=float("nan"))
        norm_ok = not math.isnan(min_sw) and min_sw > alpha

        use_friedman = False
        chi2_fr = p_fr = df_fr = float("nan")
        if not norm_ok:
            ans = messagebox.askyesno("Ненормальні різниці",
                f"Різниці між часовими точками не відповідають нормальному розподілу\n"
                f"(Shapiro–Wilk: мін. p = {fmt(min_sw,4)} ≤ α = {alpha}).\n\n"
                "Параметричний дисперсійний аналіз повторних вимірів передбачає\n"
                "нормальність цих різниць.\n\n"
                "Виконати тест Фрідмана (непараметричний аналог) замість нього?\n\n"
                "«Так» — тест Фрідмана + попарні Вілкоксона (Бонферроні)\n"
                "«Ні» — продовжити з параметричним аналізом попри це")
            if ans:
                use_friedman = True
                from scipy.stats import friedmanchisquare
                try:
                    chi2_fr, p_fr = friedmanchisquare(*[data[:,j] for j in range(k)])
                    chi2_fr = float(chi2_fr); p_fr = float(p_fr)
                except Exception:
                    chi2_fr, p_fr = float("nan"), float("nan")
                df_fr = k - 1

        mt = k*(k-1)/2 if k > 1 else 1
        ph_results = {}
        if use_friedman:
            from scipy.stats import wilcoxon
            for j in range(k):
                for jj in range(j+1, k):
                    try:
                        st_, p_w_ = wilcoxon(data[:,j], data[:,jj])
                        st_, p_w_ = float(st_), float(p_w_)
                    except Exception:
                        st_, p_w_ = float("nan"), float("nan")
                    p_adj_ = min(1., p_w_*mt) if not math.isnan(p_w_) else float("nan")
                    ph_results[(j,jj)] = (float(np.mean(data[:,j]-data[:,jj])), st_, p_adj_)
        else:
            from scipy.stats import ttest_rel
            for j in range(k):
                for jj in range(j+1, k):
                    st_, p_t_ = ttest_rel(data[:,j], data[:,jj])
                    p_adj_ = min(1., float(p_t_)*mt)
                    ph_results[(j,jj)] = (float(np.mean(data[:,j]-data[:,jj])), float(st_), p_adj_)

        if not HAS_MPL: messagebox.showwarning("","matplotlib недоступний."); return

        win = tk.Toplevel(self.win)
        win.title("Повторні виміри — Результати")
        win.geometry("1020x760"); set_icon(win)

        tb = tk.Frame(win, padx=6, pady=5); tb.pack(fill=tk.X)
        tk.Button(tb, text="📋 Копіювати графік", font=("Times New Roman",11),
                  command=lambda: self._copy_rm()).pack(side=tk.LEFT, padx=4)
        tk.Button(tb, text="⚙ Налаштування графіка", font=("Times New Roman",11),
                  command=lambda: self._restyle_rm(win, time_names, data, n, ph_results, alpha)
                  ).pack(side=tk.LEFT, padx=4)

        scroll_area = tk.Frame(win); scroll_area.pack(fill=tk.BOTH, expand=True)
        vsb = ttk.Scrollbar(scroll_area, orient="vertical"); vsb.pack(side=tk.RIGHT, fill=tk.Y)
        sc = tk.Canvas(scroll_area, yscrollcommand=vsb.set, highlightthickness=0)
        sc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.config(command=sc.yview)
        body = tk.Frame(sc); sc.create_window((0,0), window=body, anchor="nw")
        body.bind("<Configure>", lambda e: sc.configure(scrollregion=sc.bbox("all")))
        win.bind("<MouseWheel>", lambda e: sc.yview_scroll(int(-1*(e.delta/120)),"units"))

        def _head(txt):
            tk.Label(body, text=txt, font=("Times New Roman",12,"bold"),
                     bg="#e8eeff", anchor="w", padx=8, pady=3).pack(fill=tk.X, padx=6, pady=8)
        def _txt(txt, color="#000000"):
            tk.Label(body, text=txt, font=("Times New Roman",11), fg=color,
                     anchor="w", justify="left").pack(fill=tk.X, padx=14, pady=1)

        _head("Дисперсійний аналіз повторних вимірювань")
        _txt(f"Суб'єктів (n): {n}   |   Часових точок (k): {k}   |   α = {alpha}")

        if use_friedman:
            _head("Тест Фрідмана (непараметричний аналог)")
            fr_sig = not math.isnan(p_fr) and p_fr < alpha
            fr_rows = [["Фрідман", fmt(chi2_fr,4), str(df_fr), fmt(p_fr,4),
                       ("✓ значущий" if fr_sig else "✗ незначущий") if not math.isnan(p_fr) else "–"]]
            frm_a, _ = make_tv(body, ["Тест","χ²","df","p","Висновок"], fr_rows)
            frm_a.pack(fill=tk.X, padx=8, pady=(2,6))
            _head("Показники якості")
            sig_c = "#1a6b1a" if fr_sig else "#c62828"
            _txt(f"χ²({df_fr}) = {fmt(chi2_fr,4)},  p = {fmt(p_fr,4)}  "
                 f"{'✓ значуща динаміка' if fr_sig else '✗ динаміка незначуща'}", sig_c)
            _txt(f"Shapiro–Wilk (різниці) мін. p = {fmt(min_sw,4)}  ⚠ ненормальні "
                 f"→ застосовано тест Фрідмана замість параметричного аналізу", "#c62828")
            p_for_posthoc = p_fr
        else:
            _head("Таблиця дисперсійного аналізу")
            anova_rows = [
                ["Час (within)",fmt(SS_time,4),str(df_time),fmt(MS_time,4),fmt(F,4),fmt(p,4),
                 ("✓ значущий" if p < alpha else "✗ незначущий") if not math.isnan(p) else "–"],
                ["Суб'єкти",fmt(SS_subj,4),str(df_subj),"–","–","–","виноситься окремо"],
                ["Похибка",fmt(SS_error,4),str(df_err),fmt(MS_err,4),"–","–",""],
                ["Загальна",fmt(SS_total,4),str(df_time+df_subj+df_err),"–","–","–",""],
            ]
            frm_a, _ = make_tv(body, ["Джерело","SS","df","MS","F","p","Висновок"], anova_rows)
            frm_a.pack(fill=tk.X, padx=8, pady=(2,6))
            _head("Показники якості")
            sig_c = "#1a6b1a" if not math.isnan(p) and p < alpha else "#c62828"
            _txt(f"F({df_time},{df_err}) = {fmt(F,4)},  p = {fmt(p,4)}  "
                 f"{'✓ значуща динаміка' if not math.isnan(p) and p < alpha else '✗ динаміка незначуща'}", sig_c)
            _txt(f"Partial η² (час) = {fmt(eta2_time,4)}  →  {eta2_label(eta2_time)}")
            _txt(f"R² = {fmt(R2,4)}  (час + суб'єкти)")
            _txt(f"Shapiro–Wilk (різниці) мін. p = {fmt(min_sw,4)}  "
                 f"{'✓ нормальні' if norm_ok else '⚠ ненормальні (продовжено з параметричним аналізом)'}",
                 "#000" if norm_ok else "#c62828")
            p_for_posthoc = p

        _head("Середні по часових точках")
        means_tbl = [[time_names[j], fmt(float(np.mean(data[:,j])),4),
                      fmt(float(np.std(data[:,j],ddof=1)),4),
                      fmt(float(np.std(data[:,j],ddof=1)/math.sqrt(n)),4)] for j in range(k)]
        frm_m, _ = make_tv(body, ["Часова точка","Середнє","SD","СП (SE)"], means_tbl)
        frm_m.pack(fill=tk.X, padx=8, pady=(2,6))

        # Графік — власний Frame всередині body
        self._rm_graph_frame = tk.Frame(body)
        self._rm_graph_frame.pack(fill=tk.X, padx=8, pady=6)
        self._redraw_rm(win, time_names, data, n, ph_results, alpha)

        if not math.isnan(p_for_posthoc) and p_for_posthoc < alpha:
            stat_col = "W" if use_friedman else "t"
            method_lbl = "Вілкоксона" if use_friedman else "парного t-тесту"
            _head(f"Пост-хок порівняння ({method_lbl}, Бонферроні)")
            _txt(f"Скоригований α = {fmt(alpha,2)} / {int(mt)} пар = {fmt(alpha/mt,4)}   "
                 f"│   * p < {alpha}   │   ** p < {alpha*0.2:.3f}","#555")
            ph_rows = []
            for j in range(k):
                for jj in range(j+1, k):
                    d_, st_, pa_ = ph_results[(j,jj)]
                    mark = "**" if not math.isnan(pa_) and pa_<alpha*0.2 else \
                           ("*" if not math.isnan(pa_) and pa_<alpha else "–")
                    ph_rows.append([f"{time_names[j]} vs {time_names[jj]}",
                                    fmt(d_,4), fmt(st_,4), fmt(pa_,4), mark])
            frm_ph, _ = make_tv(body, ["Пара","Різниця",stat_col,"p (Bonf.)","Знач."], ph_rows)
            frm_ph.pack(fill=tk.X, padx=8, pady=(2,4))
            _txt(f"* — p < α={alpha} (значуща різниця)   "
                 f"** — p < {alpha*0.2:.3f} (висока значущість)   – — незначуща","#555")
        else:
            _txt(f"Post-hoc аналіз не виконується при незначущому "
                 f"{'тесті Фрідмана' if use_friedman else 'F-тесті'}.","#888")

    def _redraw_rm(self, win, time_names, data_arr, n, ph_results=None, alpha=0.05):
        if not hasattr(self,"_rm_graph_frame"): return
        for w in self._rm_graph_frame.winfo_children(): w.destroy()
        gs = self._rm_gs; k = len(time_names)
        fig = Figure(figsize=(10, 6), dpi=100)
        ax  = fig.add_subplot(111)
        means_ = np.mean(data_arr, axis=0)
        ses_   = np.std(data_arr, axis=0, ddof=1) / math.sqrt(n)
        ax.errorbar(range(k), means_, yerr=ses_,
                    fmt=gs["marker"]+"-", capsize=5,
                    color=gs["line_color"], ecolor=gs["err_color"],
                    linewidth=gs["linewidth"], markersize=gs["markersize"], zorder=3)
        if ph_results:
            y_range = float(np.max(means_)-np.min(means_)) if np.max(means_)>np.min(means_) else 1.
            offset = y_range*0.06
            for jj in range(1, k):
                j = jj-1; pk = (j,jj)
                if pk in ph_results and ph_results[pk][2] < alpha:
                    pa = ph_results[pk][2]
                    mark = "**" if pa<alpha*0.2 else "*"
                    xm = (j+jj)/2
                    yb = max(means_[j]+ses_[j], means_[jj]+ses_[jj])+offset
                    ax.plot([j,jj],[yb,yb],color="#555",lw=0.8)
                    ax.plot([j,j],[yb-offset*0.3,yb],color="#555",lw=0.8)
                    ax.plot([jj,jj],[yb-offset*0.3,yb],color="#555",lw=0.8)
                    ax.text(xm, yb+offset*0.1, mark, ha="center", va="bottom",
                            fontsize=gs["font_size"]+1, color="#c62828",
                            fontfamily=gs["font_family"])
        ax.set_xticks(range(k))
        ax.set_xticklabels(time_names, fontsize=gs["font_size"], fontfamily=gs["font_family"])
        ax.set_xlabel("Часова точка / Умова", fontsize=gs["font_size"], fontfamily=gs["font_family"])
        ax.set_ylabel("Середнє ± СП",         fontsize=gs["font_size"], fontfamily=gs["font_family"])
        ax.set_title("Динаміка середніх (Середнє ± СП)",
                     fontsize=gs["font_size"]+1, fontfamily=gs["font_family"])
        if gs["show_grid"]: ax.yaxis.grid(True, linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        if ph_results and any(v[2]<alpha for v in ph_results.values()):
            ax.annotate(f"* p<{alpha},  ** p<{alpha*0.2:.3f}  (Бонферроні, між сусідніми точками)",
                        xy=(0.01,0.01), xycoords="axes fraction",
                        fontsize=max(7,gs["font_size"]-1), color="#555", fontfamily=gs["font_family"])
        fig.tight_layout()
        self._rm_fig = fig
        embed_figure(fig, self._rm_graph_frame)


    def _copy_rm(self):
        if self._rm_fig is None:
            messagebox.showwarning("","Спочатку виконайте аналіз."); return
        ok, msg = _copy_fig_to_clipboard(self._rm_fig)
        if ok: messagebox.showinfo("","Графік скопійовано.\nВставте у Word через Ctrl+V.")
        else:   messagebox.showwarning("",f"Помилка: {msg}")





# ═══════════════════════════════════════════════════════════════
# MIXED REPEATED MEASURES  (Split-plot у часі)
# Варіант (between) × Дата (within), кілька повторностей
# ═══════════════════════════════════════════════════════════════
class MixedRepeatedWindow:
    """Змішаний дисперсійний аналіз повторних вимірювань.
    Between-subjects фактор: Варіант/Доза/Сорт (різний між повторностями).
    Within-subjects фактор:  Дата/Час (вимірюється для кожної повторності).
    """

    HELP_TEXT = """
ЗМІШАНИЙ АНАЛІЗ ПОВТОРНИХ ВИМІРЮВАНЬ — ІНСТРУКЦІЯ
══════════════════════════════════════════════════

ДЛЯ ЧОГО ЦЕЙ АНАЛІЗ?
  Коли у вас є КІЛЬКА ВАРІАНТІВ (дози добрива, сорти, обробки)
  і кожен варіант вимірюється КІЛЬКА РАЗІВ у часі.

  Відповідає на три питання одночасно:
  1. Чи відрізняються варіанти між собою? (ефект Варіанту)
  2. Чи є динаміка у часі? (ефект Часу)
  3. Чи по-різному змінюються варіанти у часі?
     (взаємодія Варіант × Час) ← НАЙВАЖЛИВІШЕ!

  Приклад: 4 дози добрива, вимірювання пагонів 4 рази.
  Взаємодія значуща → ефект дози залежить від дати вимірювання.

КРОК 1. ПІДГОТОВКА ДАНИХ

  Якщо у вас кілька дерев/рослин у кожній повторності:
  Спочатку порахуйте СЕРЕДНЄ по деревах для кожної повторності.
  Наприклад: 5 дерев у повторності → одне середнє значення.

  Кожне значення у таблиці = середнє по рослинах повторності.

КРОК 2. СТРУКТУРА ТАБЛИЦІ

  Стовпець 1: Варіант (текст: «Доза 1», «Контроль» тощо)
  Стовпець 2: Повторність (текст або число: «Повт.1», «1» тощо)
  Решта стовпців: Значення по датах/часових точках (числа)

  Синій рядок заголовків = назви дат/часових точок.
  Подвійний клік → перейменувати.

  Приклад (4 варіанти, 4 повторності, 4 дати):
  | Варіант | Повт. | 1.06 | 8.06 | 12.06 | 17.06 |
  | Доза 1  |   1   | 10.2 | 13.4 |  15.8 |  17.9 |
  | Доза 1  |   2   |  9.8 | 12.9 |  15.2 |  17.4 |
  | Доза 1  |   3   | 10.6 | 13.8 |  16.1 |  18.2 |
  | Доза 1  |   4   | 10.1 | 13.1 |  15.5 |  17.7 |
  | Доза 2  |   1   | 11.5 | 14.8 |  17.2 |  19.6 |
  ...

КРОК 3. СТАТИСТИЧНА МОДЕЛЬ

  Це Split-plot у часі. Дві різні помилки:

  Whole-plot error = Варіант × Повторність (помилка між групами)
    Використовується для тесту ефекту Варіанту.

  Sub-plot error = залишок (помилка всередині груп)
    Використовується для тесту Часу і Взаємодії.

  ⚠ Якщо не враховувати цю подвійну структуру — F-значення
  для Варіанту будуть хибними (занижена помилка).

КРОК 4. ІНТЕРПРЕТАЦІЯ

  Ефект Варіанту:
    p < α → варіанти значущо відрізняються (загалом по всіх датах)

  Ефект Часу:
    p < α → є значуща динаміка у часі (загалом по всіх варіантах)

  Взаємодія Варіант × Час:
    p < α → РІЗНА динаміка у різних варіантів!
      Лінії на графіку розходяться або перетинаються.
      Ефект варіанту залежить від дати → аналізуйте по датах окремо.
    p ≥ α → лінії паралельні, ефект варіанту стабільний у часі.

КРОК 5. ГРАФІК

  Кожна лінія = один варіант.
  Вертикальні смужки = ±СП (стандартна похибка середнього).
  Паралельні лінії → взаємодія незначуща.
  Розбіжні/лінії що перетинаються → взаємодія значуща.

КРОК 6. POST-HOC

  Після значущого ефекту Варіанту:
  Парні порівняння між варіантами (Бонферроні).

  Після значущої взаємодії:
  Простий ефект — порівняння варіантів на кожну дату окремо.
"""

    def __init__(self, parent, gs):
        self.win = tk.Toplevel(parent)
        self.win.title("Змішаний аналіз повторних вимірювань")
        self.win.geometry("1060x700"); set_icon(self.win)
        self.gs = gs
        self._fig = None
        self._plot_gs = {
            "font_family": "Times New Roman", "font_size": 10,
            "linewidth": 2.0, "markersize": 7, "marker": "o",
            "show_grid": True, "alpha_fill": 0.12,
            "colors": ["#4c72b0","#dd8452","#55a868","#c44e52",
                       "#8172b2","#937860","#da8bc3","#8c8c8c"],
            "err_mode": "all",       # "all" | "none" | "selected"
            "err_selected": [],      # назви варіантів, коли err_mode == "selected"
            "show_fill_band": True,  # окремо від планок похибки (whiskers)
        }
        self._build()

    def _build(self):
        try:
            self._build_inner()
        except Exception as _be:
            import traceback
            messagebox.showerror("Помилка","Помилка побудови вікна:\n"+traceback.format_exc())

    def _build_inner(self):
        # ── Toolbar ──────────────────────────────────────────
        top = tk.Frame(self.win, padx=8, pady=6); top.pack(fill=tk.X)
        tk.Button(top, text="▶ Аналіз", bg="#c62828", fg="white",
                  font=("Times New Roman",13),
                  command=self._run).pack(side=tk.LEFT, padx=4)

        tk.Label(top, text="α:", font=("Times New Roman",12)).pack(side=tk.LEFT, padx=(10,2))
        self.alpha_var = tk.StringVar(value="0.05")
        ttk.Combobox(top, textvariable=self.alpha_var, values=["0.01","0.05","0.10"],
                     state="readonly", width=7).pack(side=tk.LEFT)

        mb2 = tk.Menubutton(top, text="⚙ Налаштування ▾",
                            font=("Times New Roman",11), relief=tk.RAISED, bd=2)
        mb2.pack(side=tk.LEFT, padx=4)
        sm = tk.Menu(mb2, tearoff=0)
        sm.add_command(label="Додати рядок",       command=self._add_row)
        sm.add_command(label="Видалити рядок",     command=self._del_row)
        sm.add_separator()
        sm.add_command(label="Додати стовпець",    command=self._add_col)
        sm.add_command(label="Видалити стовпець",  command=self._del_col)
        sm.add_separator()
        sm.add_command(label="💾 Зберегти проект", command=self._save_proj)
        sm.add_command(label="📂 Відкрити проект", command=self._load_proj)
        sm.add_separator()
        sm.add_command(label="🗑 Очистити таблицю", command=self._clear)
        mb2["menu"] = sm

        tk.Button(top, text="Вставити з буфера",
                  font=("Times New Roman",11),
                  command=self._paste).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="📚 Довідка", bg="#1a4b8c", fg="white",
                  font=("Times New Roman",11),
                  command=self._show_help).pack(side=tk.LEFT, padx=4)
        tk.Label(top,
                 text="Стовп.1=Варіант  Стовп.2=Повторність  Решта=Значення по датах",
                 font=("Times New Roman",9), fg="#666").pack(side=tk.LEFT, padx=8)

        # ── Таблиця ─────────────────────────────────────────
        mid = tk.Frame(self.win); mid.pack(fill=tk.BOTH, expand=True, padx=8)
        self.rows_n = 24; self.cols_n = 8
        self._canvas = tk.Canvas(mid)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(mid, orient="vertical", command=self._canvas.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.configure(yscrollcommand=sb.set)
        self.inner = tk.Frame(self._canvas)
        self._canvas.create_window((0,0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>",
                        lambda e: self._canvas.config(scrollregion=self._canvas.bbox("all")))
        self.win.bind("<MouseWheel>",
                      lambda e: self._canvas.yview_scroll(int(-1*(e.delta/120)),"units"))

        # Фіксовані заголовки перших двох стовпців
        for j, txt in enumerate(["Варіант","Повторність"]):
            tk.Label(self.inner, text=txt, width=13, relief=tk.RIDGE,
                     bg="#444444", fg="white",
                     font=("Times New Roman",11,"bold")
                     ).grid(row=0, column=j, padx=1, pady=1, sticky="nsew")

        # Заголовки часових точок (перейменовувані)
        self.time_vars = []; self.time_labels = []
        for j in range(2, self.cols_n):
            var = tk.StringVar(value=f"Дата {j-1}")
            self.time_vars.append(var)
            lbl = tk.Label(self.inner, textvariable=var, width=12, cursor="hand2",
                           bg="#1a4b8c", fg="white", relief=tk.RIDGE,
                           font=("Times New Roman",11,"bold"))
            lbl.grid(row=0, column=j, padx=1, pady=1, sticky="nsew")
            lbl.bind("<Double-Button-1>",
                     lambda e, idx=j-2: self._rename_col(idx))
            self.time_labels.append(lbl)

        self.entries = []
        for i in range(self.rows_n):
            row_ = []
            for j in range(self.cols_n):
                w = 13 if j < 2 else 12
                e = tk.Entry(self.inner, width=w, font=("Times New Roman",11),
                             )
                e.grid(row=i+1, column=j, padx=1, pady=1)
                if j == 0:
                    e.bind("<KeyRelease>", lambda ev: _autofit_col(self.entries, 0))
                row_.append(e)
            self.entries.append(row_)
        _bind_nav(self.entries, self.win)
        _bind_fill_handle(self.entries, self.win)

    # ── Перейменування дати ───────────────────────────────────
    def _rename_col(self, idx):
        dlg = tk.Toplevel(self.win); dlg.title("Перейменувати дату")
        dlg.resizable(False, False); dlg.grab_set()
        tk.Label(dlg, text=f"Назва дати/точки {idx+1}:",
                 font=("Times New Roman",12)).pack(padx=16, pady=14)
        var = tk.StringVar(value=self.time_vars[idx].get())
        e = tk.Entry(dlg, textvariable=var, font=("Times New Roman",12), width=24)
        e.pack(padx=16, pady=4); e.select_range(0, tk.END); e.focus_set()
        def apply():
            nm = var.get().strip()
            if nm: self.time_vars[idx].set(nm)
            dlg.destroy()
        tk.Button(dlg, text="OK", bg="#c62828", fg="white",
                  font=("Times New Roman",12), command=apply).pack(pady=(4,14))
        dlg.bind("<Return>", lambda ev: apply()); center_win(dlg)

    # ── Управління таблицею ───────────────────────────────────
    def _add_row(self):
        i = self.rows_n; row_ = []
        for j in range(self.cols_n):
            e = tk.Entry(self.inner, width=13 if j<2 else 12,
                         font=("Times New Roman",11))
            e.grid(row=i+1, column=j, padx=1, pady=1)
            if j == 0:
                e.bind("<KeyRelease>", lambda ev: _autofit_col(self.entries, 0))
            row_.append(e)
        self.entries.append(row_); self.rows_n += 1
        _bind_nav(self.entries, self.win)
        _bind_fill_handle(self.entries, self.win)

    def _del_row(self):
        if not self.entries: return
        for e in self.entries.pop(): e.destroy()
        self.rows_n -= 1

    def _add_col(self):
        ci = self.cols_n; self.cols_n += 1
        var = tk.StringVar(value=f"Дата {ci-1}")
        self.time_vars.append(var)
        lbl = tk.Label(self.inner, textvariable=var, width=12, cursor="hand2",
                       bg="#1a4b8c", fg="white", relief=tk.RIDGE,
                       font=("Times New Roman",11,"bold"))
        lbl.grid(row=0, column=ci, padx=1, pady=1, sticky="nsew")
        lbl.bind("<Double-Button-1>",
                 lambda e, idx=ci-2: self._rename_col(idx))
        self.time_labels.append(lbl)
        for i, row_ in enumerate(self.entries):
            e = tk.Entry(self.inner, width=12, font=("Times New Roman",11),
                         )
            e.grid(row=i+1, column=ci, padx=1, pady=1)
            row_.append(e)
        _bind_nav(self.entries, self.win)
        _bind_fill_handle(self.entries, self.win)

    def _del_col(self):
        if self.cols_n <= 4: return  # мінімум Варіант+Повт+2 дати
        self.time_labels.pop().destroy(); self.time_vars.pop()
        for row_ in self.entries: row_.pop().destroy()
        self.cols_n -= 1

    def _clear(self):
        if not messagebox.askyesno("Очистити","Видалити всі дані?"): return
        for row in self.entries:
            for e in row: e.delete(0, tk.END)

    def _paste(self):
        try: data = self.win.clipboard_get()
        except Exception:
            messagebox.showwarning("","Скопіюйте дані з Excel і спробуйте знову."); return
        if not data.strip(): return
        pos = (0, 0)
        w = self.win.focus_get()
        if isinstance(w, tk.Entry):
            for i, row_ in enumerate(self.entries):
                for j, e in enumerate(row_):
                    if e is w: pos = (i, j); break
        r0, c0 = pos
        for ir, line in enumerate(data.splitlines()):
            if not line.strip(): continue
            while r0+ir >= len(self.entries): self._add_row()
            for jc, val in enumerate(line.split("\t")):
                cc = c0+jc
                if cc >= self.cols_n: continue
                self.entries[r0+ir][cc].delete(0, tk.END)
                self.entries[r0+ir][cc].insert(0, val.strip())
        _autofit_col(self.entries, 0)

    def _save_proj(self):
        path = filedialog.asksaveasfilename(
            parent=self.win, defaultextension=".sadp",
            filetypes=[("SAD проект","*.sadp"),("JSON","*.json")],
            title="Зберегти проект Змішаного Repeated Measures")
        if not path: return
        d = {
            "type": "mixed_repeated_measures", "version": APP_VER,
            "time_vars": [v.get() for v in self.time_vars],
            "rows_data": [[e.get() for e in row] for row in self.entries],
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("Збережено",
                f"Проект збережено:\n{path}\n\n"
                "Щоб додати нові дати: ⚙ → «Додати стовпець»")
        except Exception as ex:
            messagebox.showerror("Помилка збереження", str(ex))

    def _load_proj(self):
        path = filedialog.askopenfilename(
            parent=self.win,
            filetypes=[("SAD проект","*.sadp"),("JSON","*.json")],
            title="Відкрити проект Змішаного Repeated Measures")
        if not path: return
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception as ex:
            messagebox.showerror("Помилка відкриття", str(ex)); return
        time_vars = d.get("time_vars", [])
        rows_data = d.get("rows_data", [])
        # Розширюємо до потрібної кількості стовпців (2 фіксовані + дати)
        n_cols_needed = 2 + len(time_vars)
        while self.cols_n < n_cols_needed: self._add_col()
        # Назви дат
        for i, nm in enumerate(time_vars):
            if i < len(self.time_vars): self.time_vars[i].set(nm)
        # Дані
        while len(self.entries) < len(rows_data): self._add_row()
        for i, row_vals in enumerate(rows_data):
            for j, v in enumerate(row_vals):
                if j < self.cols_n:
                    self.entries[i][j].delete(0, tk.END)
                    self.entries[i][j].insert(0, v)
        messagebox.showinfo("Завантажено",
            "Проект завантажено.\n\n"
            "Щоб додати нові дати вимірювань:\n"
            "  ⚙ → «Додати стовпець» → подвійний клік на заголовку → назва дати.")

    def _paste(self):
        try: data = self.win.clipboard_get()
        except Exception:
            messagebox.showwarning("","Скопіюйте дані з Excel і спробуйте знову."); return
        pos = (0,0)
        w = self.win.focus_get()
        if isinstance(w, tk.Entry):
            for i, row_ in enumerate(self.entries):
                for j, e in enumerate(row_):
                    if e is w: pos=(i,j); break
        r0, c0 = pos
        for ir, line in enumerate(data.splitlines()):
            if not line.strip(): continue
            while r0+ir >= len(self.entries): self._add_row()
            for jc, val in enumerate(line.split("\t")):
                cc = c0+jc
                if cc >= self.cols_n: continue
                self.entries[r0+ir][cc].delete(0,tk.END)
                self.entries[r0+ir][cc].insert(0, val.strip())

    def _show_help(self):
        win = tk.Toplevel(self.win)
        win.title("Довідка — Змішаний Repeated Measures")
        win.geometry("720x680"); set_icon(win)
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

    # ── Зчитування та аналіз ─────────────────────────────────
    def _run(self):
        # Назви дат (лише заповнені)
        all_dates = [v.get().strip() or f"Д{i+1}" for i,v in enumerate(self.time_vars)]

        raw = [[e.get().strip() for e in row] for row in self.entries]
        raw = [r for r in raw if any(v for v in r)]
        if not raw:
            messagebox.showwarning("Немає даних","Введіть дані у таблицю."); return

        # Знаходимо заповнені часові стовпці (індекси 2..)
        filled_t = []
        for j in range(2, self.cols_n):
            for row in raw:
                v = row[j] if j<len(row) else ""
                if v:
                    try: float(v.replace(",",".")); filled_t.append(j); break
                    except ValueError: pass
        if not filled_t:
            messagebox.showwarning("Немає числових даних",
                "Введіть числові значення у стовпці 3 і далі (часові точки)."); return

        time_names = [all_dates[j-2] for j in filled_t]
        k = len(time_names)  # кількість часових точок

        # Зчитуємо рядки: варіант, повторність, значення
        rows_data = []
        for row in raw:
            var_nm = row[0].strip() if row[0].strip() else None
            rep_nm = row[1].strip() if len(row)>1 and row[1].strip() else None
            if not var_nm: continue
            try: float(var_nm.replace(",",".")); continue  # числовий → пропустити
            except ValueError: pass
            vals = []
            for j in filled_t:
                v = row[j].strip() if j<len(row) else ""
                try: vals.append(float(v.replace(",",".")))
                except: vals.append(float("nan"))
            if any(not math.isnan(v) for v in vals):
                rows_data.append({"var": var_nm, "rep": rep_nm or "?", "vals": vals})

        if not rows_data:
            messagebox.showwarning("Немає даних","Не вдалося зчитати жодного рядка."); return

        # Групуємо по варіантах
        var_levels = first_seen([r["var"] for r in rows_data])
        n_vars = len(var_levels)
        if n_vars < 2:
            messagebox.showwarning("Замало варіантів",
                "Потрібно щонайменше 2 варіанти (різні значення у стовпці 1)."); return

        # Будуємо матриці: var_data[var] = np.array(n_reps × k)
        var_data = {}
        for lv in var_levels:
            rr = [r["vals"] for r in rows_data if r["var"]==lv]
            # Видаляємо рядки де є NaN
            rr_clean = [v for v in rr if not any(math.isnan(x) for x in v)]
            if not rr_clean:
                messagebox.showwarning("Замало повних даних",
                    f"Варіант '{lv}' не має жодного рядка з повними даними."); return
            var_data[lv] = np.array([v[:k] for v in rr_clean], dtype=float)

        n_reps = {lv: len(var_data[lv]) for lv in var_levels}
        if min(n_reps.values()) < 2:
            messagebox.showwarning("Замало повторностей",
                "Кожен варіант потребує ≥ 2 повторностей."); return

        alpha = float(self.alpha_var.get())

        # ══ Split-plot у часі: правильні дві помилки ══
        # Загальне середнє
        all_data = np.vstack([var_data[lv] for lv in var_levels])
        grand_mean = float(np.mean(all_data))
        N_total = sum(n_reps[lv] for lv in var_levels)  # всього рядків

        # Середні по варіантах (усереднені по часу)
        var_means = {lv: float(np.mean(var_data[lv])) for lv in var_levels}
        # Середні по часу (усереднені по всіх варіантах і повторностях)
        time_means = np.mean(all_data, axis=0)  # shape (k,)

        # ── SS розкладання (Split-plot) ──────────────────────
        # SS_var (between groups): сума (n_rep_i * k) * (mean_i - grand)^2
        SS_var = float(sum(
            n_reps[lv]*k*(var_means[lv]-grand_mean)**2
            for lv in var_levels))

        # SS_whole_error (within groups, between replicates):
        # для кожного варіанту сума по повторностях k*(rep_mean - var_mean)^2
        SS_wp_err = 0.
        for lv in var_levels:
            rep_means_lv = np.mean(var_data[lv], axis=1)  # середнє кожної повт. по часу
            SS_wp_err += k * float(np.sum((rep_means_lv - var_means[lv])**2))

        # SS_time (within subjects, main effect of time)
        SS_time = float(N_total * np.sum((time_means - grand_mean)**2))

        # SS_var_time (interaction)
        SS_inter = 0.
        for lv in var_levels:
            var_time_means = np.mean(var_data[lv], axis=0)  # shape (k,)
            SS_inter += n_reps[lv] * float(
                np.sum((var_time_means - var_means[lv] - time_means + grand_mean)**2))

        # SS_sub_error (within subjects, residual)
        SS_sub_err = 0.
        for lv in var_levels:
            vm_lv = np.mean(var_data[lv], axis=0)  # var×time means
            rep_means_lv = np.mean(var_data[lv], axis=1, keepdims=True)
            vm_lv_rep = var_means[lv]
            for ri in range(n_reps[lv]):
                for ti in range(k):
                    actual = var_data[lv][ri, ti]
                    expected = (var_means[lv] +
                                (np.mean(var_data[lv], axis=0)[ti] - var_means[lv]) +
                                (np.mean(var_data[lv], axis=1)[ri] - var_means[lv]))
                    SS_sub_err += (actual - expected)**2

        # Degrees of freedom
        df_var     = n_vars - 1
        df_wp_err  = sum(n_reps[lv]-1 for lv in var_levels)   # N - n_vars
        df_time    = k - 1
        df_inter   = df_var * df_time
        df_sub_err = df_wp_err * df_time

        # MS
        MS_var     = SS_var    / df_var    if df_var    > 0 else float("nan")
        MS_wp_err  = SS_wp_err / df_wp_err if df_wp_err > 0 else float("nan")
        MS_time    = SS_time   / df_time   if df_time   > 0 else float("nan")
        MS_inter   = SS_inter  / df_inter  if df_inter  > 0 else float("nan")
        MS_sub_err = SS_sub_err/ df_sub_err if df_sub_err > 0 else float("nan")

        # F-тести (два різних знаменники!)
        F_var   = MS_var  / MS_wp_err  if (MS_wp_err  and MS_wp_err  > 1e-12) else float("nan")
        F_time  = MS_time / MS_sub_err if (MS_sub_err and MS_sub_err > 1e-12) else float("nan")
        F_inter = MS_inter/ MS_sub_err if (MS_sub_err and MS_sub_err > 1e-12) else float("nan")

        p_var   = float(1-f_dist.cdf(F_var,   df_var,  df_wp_err))  if not math.isnan(F_var)   else float("nan")
        p_time  = float(1-f_dist.cdf(F_time,  df_time, df_sub_err)) if not math.isnan(F_time)  else float("nan")
        p_inter = float(1-f_dist.cdf(F_inter, df_inter,df_sub_err)) if not math.isnan(F_inter) else float("nan")

        # Partial η²
        def peta2(SS_eff, SS_err):
            return SS_eff/(SS_eff+SS_err) if (SS_eff+SS_err)>0 else float("nan")
        e2_var   = peta2(SS_var,   SS_wp_err)
        e2_time  = peta2(SS_time,  SS_sub_err)
        e2_inter = peta2(SS_inter, SS_sub_err)

        # ── Post-hoc між варіантами (Бонферроні) ─────────────
        # Використовуємо ТУ САМУ пул-похибку (MS_wp_err, df_wp_err), що й
        # основний F-тест ефекту Варіанту — це методично правильний підхід
        # (Доспєхов, принцип НІР): усі попарні порівняння мають спиратись
        # на єдину, спільну оцінку похибки з повної моделі (з її більшим,
        # надійнішим df), а не на окремо перераховану для кожної пари
        # дисперсію, яка ігнорує дані решти варіантів і дає менш надійний,
        # неузгоджений з омнібус-тестом результат.
        var_pairs = [(var_levels[i],var_levels[j])
                     for i in range(n_vars) for j in range(i+1,n_vars)]
        mt_var = len(var_pairs)
        ph_var = []
        for lv1,lv2 in var_pairs:
            m1 = float(np.mean(var_data[lv1])); m2 = float(np.mean(var_data[lv2]))
            n1, n2 = n_reps[lv1], n_reps[lv2]
            if not math.isnan(MS_wp_err) and df_wp_err > 0:
                se_diff = math.sqrt(MS_wp_err * (1/n1 + 1/n2))
                if se_diff > 1e-12:
                    t_ = (m1 - m2) / se_diff
                    p_raw = 2 * (1 - float(t_dist.cdf(abs(t_), df_wp_err)))
                    p_adj = min(1., p_raw * mt_var)
                else:
                    t_, p_adj = float("nan"), float("nan")
            else:
                t_, p_adj = float("nan"), float("nan")
            mark = ("**" if not math.isnan(p_adj) and p_adj<alpha*0.2 else
                    ("*" if not math.isnan(p_adj) and p_adj<alpha else "–"))
            ph_var.append([f"{lv1} vs {lv2}",
                           fmt(m1,4), fmt(m2,4),
                           fmt(m1-m2,4), fmt(float(t_),4),
                           fmt(p_adj,4), mark])

        # ── Простий ефект: порівняння варіантів на кожну дату ─
        # + літери істотної різниці (CLD) між варіантами В МЕЖАХ кожної дати
        simple_rows = []
        letters_by_date = {}   # dn -> {variant: "a"/"ab"/...}
        mt_simple = n_vars*(n_vars-1)/2 if n_vars > 1 else 1
        for ti, dn in enumerate(time_names):
            col_data = {lv: var_data[lv][:,ti] for lv in var_levels}
            means_t = {lv: float(np.mean(col_data[lv])) for lv in var_levels}
            # ANOVA на цій даті
            grand_t = np.mean(np.concatenate(list(col_data.values())))
            ss_b = sum(len(col_data[lv])*(np.mean(col_data[lv])-grand_t)**2 for lv in var_levels)
            ss_w = sum(np.sum((col_data[lv]-np.mean(col_data[lv]))**2) for lv in var_levels)
            dft_b = n_vars-1
            dft_w = sum(len(col_data[lv])-1 for lv in var_levels)
            ms_b = ss_b/dft_b if dft_b>0 else float("nan")
            ms_w = ss_w/dft_w if dft_w>0 else float("nan")
            F_t = ms_b/ms_w if (ms_w and ms_w>1e-12) else float("nan")
            p_t = float(1-f_dist.cdf(F_t,dft_b,dft_w)) if not math.isnan(F_t) else float("nan")
            mark_t = "**" if p_t<alpha*0.2 else ("*" if p_t<alpha else "–")
            simple_rows.append([dn, fmt(F_t,4), f"{dft_b},{dft_w}", fmt(p_t,4), mark_t])

            # Попарні порівняння варіантів САМЕ на цій даті (LSD, пул-похибка
            # цієї дати ms_w), Бонферроні-корекція → компактні літери (CLD)
            sig_matrix = {}
            if not math.isnan(ms_w) and ms_w > 1e-12 and dft_w > 0:
                for i in range(n_vars):
                    for j in range(i+1, n_vars):
                        lv1, lv2 = var_levels[i], var_levels[j]
                        n1, n2 = len(col_data[lv1]), len(col_data[lv2])
                        se_ = math.sqrt(ms_w*(1/n1+1/n2))
                        if se_ > 1e-12:
                            t_ = (means_t[lv1]-means_t[lv2]) / se_
                            p_ = 2*(1-float(t_dist.cdf(abs(t_), dft_w)))
                            p_adj = min(1., p_*mt_simple)
                            sig_matrix[(lv1,lv2)] = p_adj < alpha
            letters_by_date[dn] = cld(var_levels, means_t, sig_matrix)

        if not HAS_MPL:
            messagebox.showwarning("","matplotlib недоступний."); return

        self._show_results(
            var_levels, var_data, time_names, n_reps, alpha,
            SS_var, SS_wp_err, SS_time, SS_inter, SS_sub_err,
            df_var, df_wp_err, df_time, df_inter, df_sub_err,
            MS_var, MS_wp_err, MS_time, MS_inter, MS_sub_err,
            F_var, F_time, F_inter,
            p_var, p_time, p_inter,
            e2_var, e2_time, e2_inter,
            ph_var, simple_rows, letters_by_date)

    def _show_results(self, var_levels, var_data, time_names, n_reps, alpha,
                      SS_var, SS_wp_err, SS_time, SS_inter, SS_sub_err,
                      df_var, df_wp_err, df_time, df_inter, df_sub_err,
                      MS_var, MS_wp_err, MS_time, MS_inter, MS_sub_err,
                      F_var, F_time, F_inter,
                      p_var, p_time, p_inter,
                      e2_var, e2_time, e2_inter,
                      ph_var, simple_rows, letters_by_date):
        self._letters_by_date = letters_by_date

        win = tk.Toplevel(self.win)
        win.title("Змішаний Repeated Measures — Результати")
        win.geometry("1160x820"); set_icon(win)

        # Toolbar
        tb = tk.Frame(win, padx=6, pady=5); tb.pack(fill=tk.X)
        tk.Button(tb, text="📋 Копіювати графік", font=("Times New Roman",11),
                  command=lambda: self._copy_fig()).pack(side=tk.LEFT, padx=4)
        tk.Button(tb, text="⚙ Налаштування графіка", font=("Times New Roman",11),
                  command=lambda: self._restyle(win, var_levels, var_data, time_names, alpha)
                  ).pack(side=tk.LEFT, padx=4)

        # Scrollable body
        sa = tk.Frame(win); sa.pack(fill=tk.BOTH, expand=True)
        vsb = ttk.Scrollbar(sa, orient="vertical"); vsb.pack(side=tk.RIGHT, fill=tk.Y)
        sc = tk.Canvas(sa, yscrollcommand=vsb.set, highlightthickness=0)
        sc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.config(command=sc.yview)
        body = tk.Frame(sc)
        body_win = sc.create_window((0,0), window=body, anchor="nw")
        body.bind("<Configure>", lambda e: sc.configure(scrollregion=sc.bbox("all")))
        # Ширина body синхронізована з видимою шириною canvas — інакше вміст
        # (таблиці, графік) не звужується/розширюється разом з вікном і
        # виглядає так, ніби «не адаптується» до розміру вікна.
        sc.bind("<Configure>", lambda e: sc.itemconfig(body_win, width=e.width))
        win.bind("<MouseWheel>", lambda e: sc.yview_scroll(int(-1*(e.delta/120)),"units"))

        def _head(t):
            tk.Label(body, text=t, font=("Times New Roman",12,"bold"),
                     bg="#e8eeff", anchor="w", padx=8, pady=3
                     ).pack(fill=tk.X, padx=6, pady=8)
        def _txt(t, color="#000"):
            tk.Label(body, text=t, font=("Times New Roman",11), fg=color,
                     anchor="w", justify="left").pack(fill=tk.X, padx=14, pady=1)
        def _tbl(hdrs, rows):
            f, _ = make_tv(body, hdrs, rows); f.pack(fill=tk.X, padx=8, pady=2)

        _head("Змішаний аналіз повторних вимірювань (Split-plot у часі)")
        _txt(f"Варіантів: {len(var_levels)}   |   Повторностей: {list(n_reps.values())}   "
             f"|   Часових точок: {len(time_names)}   |   α = {alpha}")

        # Основна таблиця ANOVA
        _head("Зведена таблиця дисперсійного аналізу")
        _txt("Ефект Варіанту перевіряється відносно «Похибки між повторностями» —\n"
             "вона відображає, наскільки різняться між собою повторності ОДНОГО Й ТОГО Ж\n"
             "варіанту (усереднено по часу). Ефекти Часу і Взаємодії перевіряються\n"
             "відносно окремої, зазвичай значно меншої «Похибки всередині повторностей»\n"
             "(залишкової мінливості вимірів у часі для кожної повторності окремо).\n"
             "Це і є суть split-plot: варіант — «велика ділянка», час — «підділянка»,\n"
             "кожна має власну похибку — використання не тієї похибки дало б хибний F.",
             "#555")
        def _row(name, SS, df, MS, F, p, e2, note=""):
            mark = ("**" if p<alpha*0.2 else ("*" if p<alpha else "–")) if not math.isnan(p) else "–"
            return [name, fmt(SS,4), str(df), fmt(MS,4), fmt(F,4),
                    fmt(p,4), mark, fmt(e2,4), eta2_label(e2), note]
        anova_tbl = [
            _row("Варіант (between)",    SS_var,    df_var,    MS_var,    F_var,   p_var,   e2_var,
                 "÷ Похибка між повторностями"),
            ["  Похибка між повторностями (Whole-plot)",
             fmt(SS_wp_err,4), str(df_wp_err), fmt(MS_wp_err,4),"–","–","–","–","",""],
            _row("Час (within)",          SS_time,   df_time,   MS_time,   F_time,  p_time,  e2_time,
                 "÷ Похибка всередині повторностей"),
            _row("Варіант × Час",         SS_inter,  df_inter,  MS_inter,  F_inter, p_inter, e2_inter,
                 "÷ Похибка всередині повторностей"),
            ["  Похибка всередині повторностей (Sub-plot)",
             fmt(SS_sub_err,4),str(df_sub_err),fmt(MS_sub_err,4),"–","–","–","–","",""],
        ]
        _tbl(["Джерело","SS","df","MS","F","p","Знач.","η²","Ефект","Знаменник F-тесту"],
             anova_tbl)

        # Висновки
        _head("Висновки")
        for label, p_val, e2 in [
            ("Варіант", p_var, e2_var),
            ("Час",     p_time, e2_time),
            ("Варіант × Час (взаємодія)", p_inter, e2_inter)
        ]:
            if not math.isnan(p_val):
                sig = p_val < alpha
                col = "#1a6b1a" if sig else "#c62828"
                txt = (f"✓ {label}: значущий (p={fmt(p_val,4)}, η²={fmt(e2,3)} — {eta2_label(e2)})"
                       if sig else
                       f"✗ {label}: незначущий (p={fmt(p_val,4)})")
                _txt(txt, col)
        if not math.isnan(p_inter) and p_inter < alpha:
            _txt("⚠ Взаємодія значуща → ефект варіанту залежить від дати.\n"
                 "   Аналізуйте прості ефекти (таблиця нижче) і дивіться чи лінії розходяться.",
                 "#c62828")
        elif not math.isnan(p_inter):
            _txt("✓ Взаємодія незначуща → лінії паралельні, ефект варіанту стабільний у часі.",
                 "#1a6b1a")

        # Середні
        _head("Середні значення (Варіант × Дата)")
        means_hdrs = ["Варіант"] + time_names + ["Загальне"]
        means_rows = []
        for lv in var_levels:
            row_ = [lv]
            for ti in range(len(time_names)):
                row_.append(fmt(float(np.mean(var_data[lv][:,ti])),3))
            row_.append(fmt(float(np.mean(var_data[lv])),3))
            means_rows.append(row_)
        _tbl(means_hdrs, means_rows)

        # Графік
        self._graph_frame = tk.Frame(body)
        self._graph_frame.pack(fill=tk.X, padx=8, pady=6)
        self._draw_graph(var_levels, var_data, time_names, alpha, letters_by_date)

        # Post-hoc варіанти
        if not math.isnan(p_var) and p_var < alpha:
            _head(f"Пост-хок: порівняння варіантів (Бонферроні, α_скор={fmt(alpha/len(ph_var),4)})")
            _tbl(["Пара","Сер.1","Сер.2","Різниця","t","p (Bonf.)","Знач."], ph_var)
            _txt(f"* p<{alpha}   ** p<{alpha*0.2:.3f}   – незначуща","#555")

        # Прості ефекти (по датах) — F-тест по датах показується лише коли
        # взаємодія значуща (це і є методичний привід дивитись на дати окремо)
        if not math.isnan(p_inter) and p_inter < alpha:
            _head("Простий ефект: порівняння варіантів на кожну дату окремо")
            _txt("Виконується при значущій взаємодії — показує на яких датах варіанти відрізняються.",
                 "#555")
            _tbl(["Дата","F","df","p","Знач."], simple_rows)
            _txt(f"* p<{alpha}   ** p<{alpha*0.2:.3f}   – незначуща","#555")

        # Літери істотної різниці — показуємо ЗАВЖДИ (узгоджено з графіком,
        # де вони теж завжди присутні), з відповідним застереженням
        _head("Групи істотної різниці (літери) між варіантами на кожну дату")
        _txt("Попарне порівняння варіантів САМЕ на цій даті (LSD, Бонферроні). "
             "Варіанти з ОДНАКОВОЮ літерою в межах однієї дати статистично не "
             "відрізняються один від одного; з РІЗНИМИ літерами — відрізняються.\n"
             "Ці самі літери показані безпосередньо на графіку біля кожної точки.",
             "#555")
        if math.isnan(p_inter) or p_inter >= alpha:
            _txt("⚠ Взаємодія Варіант × Час НЕзначуща — це означає, що загалом "
                 "варіанти поводяться однаково в часі. Літери нижче все ж показують "
                 "поточний статистичний розподіл на кожну дату, але як основний "
                 "висновок про відмінність варіантів спирайтесь на ефект Варіанту "
                 "вище, а не на розбіжності літер тут.", "#b07000")
        letters_hdrs = ["Варіант"] + list(time_names)
        letters_rows = [[lv] + [letters_by_date[dn].get(lv,"") for dn in time_names]
                        for lv in var_levels]
        _tbl(letters_hdrs, letters_rows)

    def _draw_graph(self, var_levels, var_data, time_names, alpha=0.05, letters_by_date=None):
        for w in self._graph_frame.winfo_children(): w.destroy()
        gs = self._plot_gs
        k = len(time_names)
        colors = gs["colors"]
        err_mode = gs.get("err_mode", "all")
        err_sel  = set(gs.get("err_selected", []))
        show_fill = gs.get("show_fill_band", True)
        letters_by_date = letters_by_date or {}
        fig = Figure(figsize=(10, 6), dpi=100)
        ax  = fig.add_subplot(111)

        for ci, lv in enumerate(var_levels):
            col = colors[ci % len(colors)]
            means_ = np.mean(var_data[lv], axis=0)
            ses_   = np.std(var_data[lv], axis=0, ddof=1) / math.sqrt(len(var_data[lv]))
            show_bars = (err_mode == "all") or (err_mode == "selected" and lv in err_sel)
            if show_bars:
                # Планки похибки (whiskers/caps) — завжди разом з errorbar,
                # незалежно від того, чи ввімкнена тіньова смуга нижче
                ax.errorbar(range(k), means_, yerr=ses_,
                            fmt=gs["marker"]+"-", capsize=5,
                            color=col, ecolor=col,
                            linewidth=gs["linewidth"],
                            markersize=gs["markersize"],
                            label=str(lv), alpha=0.9, zorder=3)
                # Тіньова смуга ±СП — окрема, незалежна опція; її вимкнення
                # НЕ прибирає планки похибки вище
                if show_fill:
                    ax.fill_between(range(k),
                                    means_-ses_, means_+ses_,
                                    alpha=gs["alpha_fill"], color=col)
            else:
                ax.plot(range(k), means_, gs["marker"]+"-",
                       color=col, linewidth=gs["linewidth"],
                       markersize=gs["markersize"], label=str(lv),
                       alpha=0.9, zorder=3)

            # Літери істотної різниці (CLD) біля кожної точки — порівняння
            # варіантів В МЕЖАХ цієї ж дати (не між датами!)
            if letters_by_date:
                for ti, dn in enumerate(time_names):
                    lt = letters_by_date.get(dn, {}).get(lv, "")
                    if lt:
                        y_top = means_[ti] + (ses_[ti] if show_bars else 0)
                        ax.annotate(lt, (ti, y_top),
                                   textcoords="offset points", xytext=(0, 6),
                                   ha="center", va="bottom",
                                   fontsize=max(7, gs["font_size"]-1),
                                   color=col, fontweight="bold")

        if letters_by_date:
            fig.text(0.5, 0.005,
                     "Літери — групи істотної різниці МІЖ ВАРІАНТАМИ в межах кожної дати "
                     "окремо (однакова літера = не відрізняються)",
                     ha="center", fontsize=max(7, gs["font_size"]-2), color="#666")

        ax.set_xticks(range(k))
        ax.set_xticklabels(time_names,
                           fontsize=gs["font_size"],
                           fontfamily=gs["font_family"])
        ax.set_xlabel("Дата / Часова точка",
                      fontsize=gs["font_size"], fontfamily=gs["font_family"])
        ax.set_ylabel("Середнє" + (" ± СП" if err_mode != "none" else ""),
                      fontsize=gs["font_size"], fontfamily=gs["font_family"])
        title_suffix = {"all": "Середнє ± СП, усі варіанти",
                        "none": "лише середні, без смуг похибки",
                        "selected": "Середнє ± СП для обраних варіантів"}[err_mode]
        ax.set_title(f"Динаміка по варіантах ({title_suffix})",
                     fontsize=gs["font_size"]+1, fontfamily=gs["font_family"])
        ax.legend(title="Варіант", fontsize=gs["font_size"],
                  title_fontsize=gs["font_size"])
        if gs["show_grid"]:
            ax.yaxis.grid(True, linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        self._fig = fig

        embed_figure(fig, self._graph_frame)

    def _copy_fig(self):
        if self._fig is None:
            messagebox.showwarning("","Спочатку виконайте аналіз."); return
        ok, msg = _copy_fig_to_clipboard(self._fig)
        if ok: messagebox.showinfo("","Графік скопійовано.\nВставте у Word через Ctrl+V.")
        else:   messagebox.showwarning("",f"Помилка: {msg}")

    def _restyle(self, win, var_levels, var_data, time_names, alpha):
        dlg = tk.Toplevel(win); dlg.title("Налаштування графіка")
        dlg.resizable(False, False); set_icon(dlg); dlg.grab_set()
        gs = self._plot_gs
        frm = tk.Frame(dlg, padx=16, pady=14); frm.pack()
        rb_f = ("Times New Roman",12)
        ff_v = tk.StringVar(value=gs["font_family"])
        fz_v = tk.IntVar(value=gs["font_size"])
        lw_v = tk.DoubleVar(value=gs["linewidth"])
        ms_v = tk.IntVar(value=gs["markersize"])
        mk_v = tk.StringVar(value=gs["marker"])
        gr_v = tk.BooleanVar(value=gs["show_grid"])
        al_v = tk.DoubleVar(value=gs["alpha_fill"])
        rows_cfg = [
            ("Шрифт:",          "combo",  ff_v, ["Times New Roman","Arial","Calibri","Georgia"]),
            ("Розмір шрифту:",  "spin",   fz_v, (7,18)),
            ("Товщина лінії:",  "scale",  lw_v, (0.5,5.0)),
            ("Розмір маркера:", "spin",   ms_v, (3,20)),
            ("Тип маркера:",    "combo",  mk_v, ["o","s","^","D","v","*","+"]),
            ("Показати сітку:", "check",  gr_v, None),
            ("Прозорість тіні:","scale",  al_v, (0.0,0.4)),
        ]
        col_refs = list(gs["colors"])
        col_btns = []
        for ri, (lbl, wt, var, opts) in enumerate(rows_cfg):
            tk.Label(frm, text=lbl, font=rb_f).grid(row=ri, column=0, sticky="w", pady=4)
            if wt=="combo":
                ttk.Combobox(frm, textvariable=var, values=opts,
                             state="readonly", width=18).grid(row=ri, column=1, sticky="w", padx=8)
            elif wt=="spin":
                tk.Spinbox(frm, from_=opts[0], to=opts[1], textvariable=var,
                           width=7).grid(row=ri, column=1, sticky="w", padx=8)
            elif wt=="scale":
                tk.Scale(frm, from_=opts[0], to=opts[1], resolution=0.05,
                         orient="horizontal", variable=var,
                         length=160).grid(row=ri, column=1, sticky="w", padx=8)
            elif wt=="check":
                tk.Checkbutton(frm, variable=var).grid(row=ri, column=1, sticky="w", padx=8)

        # Смуги похибки (±СП) — усі / жодної / обрані варіанти
        base_r = len(rows_cfg)
        err_row = base_r
        tk.Label(frm, text="Смуги ±СП (похибка):", font=rb_f).grid(
            row=err_row, column=0, sticky="w", pady=(10,4))
        mode_map = {"Показати для всіх варіантів": "all",
                    "Не показувати (лише лінії середніх)": "none",
                    "Показати лише для обраних нижче": "selected"}
        mode_rev = {v: k for k, v in mode_map.items()}
        mode_disp_v = tk.StringVar(
            value=mode_rev.get(gs.get("err_mode", "all"), "Показати для всіх варіантів"))
        mode_cb = ttk.Combobox(frm, textvariable=mode_disp_v, values=list(mode_map.keys()),
                               state="readonly", width=32)
        mode_cb.grid(row=err_row, column=1, sticky="w", padx=8, pady=(10,4))

        tk.Label(frm, text="Довірчі смуги — це ±СП (стандартна похибка середнього),\n"
                          "не довірчий інтервал. При багатьох варіантах вони можуть\n"
                          "накладатись одна на одну — оберіть режим вище, щоб показати\n"
                          "лише потрібні.",
                 font=("Times New Roman",9), fg="#666", justify="left"
                 ).grid(row=err_row+1, column=0, columnspan=2, sticky="w", padx=(0,0))

        fill_v = tk.BooleanVar(value=gs.get("show_fill_band", True))
        tk.Label(frm, text="Тіньова смуга (заливка) навколо лінії:", font=rb_f).grid(
            row=err_row+2, column=0, sticky="w", pady=(6,4))
        tk.Checkbutton(frm, variable=fill_v).grid(
            row=err_row+2, column=1, sticky="w", padx=8, pady=(6,4))
        tk.Label(frm, text="Не впливає на планки похибки (вусики) у точках — їх вимикає\n"
                          "лише вибір режиму вище («Не показувати»/«Обрані»).",
                 font=("Times New Roman",9), fg="#666", justify="left"
                 ).grid(row=err_row+3, column=0, columnspan=2, sticky="w")

        tk.Label(frm, text="Обрані варіанти:", font=rb_f).grid(
            row=err_row+4, column=0, sticky="nw", pady=(8,4))
        sel_frame = tk.Frame(frm)
        sel_frame.grid(row=err_row+4, column=1, sticky="w", padx=8, pady=(8,4))
        cur_sel = set(gs.get("err_selected", []))
        sel_vars = {}
        for lv in var_levels:
            v = tk.BooleanVar(value=(lv in cur_sel))
            cb = tk.Checkbutton(sel_frame, text=str(lv), variable=v,
                                font=("Times New Roman",10), anchor="w")
            cb.pack(fill=tk.X, anchor="w")
            sel_vars[lv] = v

        def _toggle_sel_state(*_):
            state = tk.NORMAL if mode_map[mode_disp_v.get()] == "selected" else tk.DISABLED
            for child in sel_frame.winfo_children():
                child.configure(state=state)
        mode_cb.bind("<<ComboboxSelected>>", _toggle_sel_state)
        _toggle_sel_state()

        # Кольори варіантів
        base_r = err_row + 5
        tk.Label(frm, text="Кольори варіантів:", font=rb_f).grid(
            row=base_r, column=0, sticky="w", pady=4)
        cf = tk.Frame(frm); cf.grid(row=base_r, column=1, sticky="w")
        for ci, lv in enumerate(var_levels[:8]):
            c = col_refs[ci] if ci < len(col_refs) else "#999"
            btn = tk.Button(cf, width=3, relief=tk.SUNKEN, bg=c,
                            text=str(ci+1), font=("Times New Roman",8))
            btn.pack(side=tk.LEFT, padx=2)
            def _pick(idx=ci, b=btn):
                ch = colorchooser.askcolor(color=col_refs[idx], parent=dlg)
                if ch and ch[1]: col_refs[idx]=ch[1]; b.configure(bg=ch[1])
            btn.configure(command=_pick); col_btns.append(btn)

        def apply():
            selected = [lv for lv, v in sel_vars.items() if v.get()]
            self._plot_gs.update({
                "font_family": ff_v.get(), "font_size": fz_v.get(),
                "linewidth":   lw_v.get(), "markersize": ms_v.get(),
                "marker":      mk_v.get(), "show_grid": gr_v.get(),
                "alpha_fill":  al_v.get(), "colors": col_refs,
                "err_mode":    mode_map[mode_disp_v.get()],
                "err_selected": selected,
                "show_fill_band": fill_v.get(),
            })
            dlg.destroy()
            self._draw_graph(var_levels, var_data, time_names, alpha, self._letters_by_date)

        bf = tk.Frame(frm); bf.grid(row=base_r+1, column=0, columnspan=2, pady=(14,0))
        tk.Button(bf, text="OK (застосувати)", bg="#c62828", fg="white",
                  font=rb_f, command=apply).pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text="Скасувати", font=rb_f, command=dlg.destroy).pack(side=tk.LEFT)
        center_win(dlg)


# ═══════════════════════════════════════════════════════════════
# STABILITY ANALYSIS  (Eberhart–Russell + GGE biplot)
# ═══════════════════════════════════════════════════════════════
