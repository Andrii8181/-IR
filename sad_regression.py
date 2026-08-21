# sad_regression.py — Регресія, розрахунок вибірки
# -*- coding: utf-8 -*-
from sad_common import *
from sad_journal_trial import open_indicators_for_variant_analysis

class RegressionWindow:
    MODELS = ["Лінійна:  y = a + bx",
              "Квадратична:  y = a + bx + cx²",
              "Кубічна:  y = a + bx + cx² + dx³",
              "Степенева:  y = a·xᵇ",
              "Експоненційна:  y = a·eᵇˣ",
              "Логарифмічна:  y = a + b·ln(x)",
              "Логістична (4-пар.):  y = d + (a-d)/(1+(x/c)ᵇ)"]

    HELP_TEXT = """
РЕГРЕСІЙНИЙ АНАЛІЗ — ПОКРОКОВА ІНСТРУКЦІЯ
══════════════════════════════════════════

КРОК 1. ВВЕДЕННЯ ДАНИХ
  • Ліве поле: значення незалежної змінної x
    (фактор, що ви змінюєте — доза добрива, час, температура тощо)
  • Праве поле: значення залежної змінної y
    (показник, що вимірюєте — врожайність, висота, маса тощо)
  • Вводьте по одному значенню на рядок або через кому
  • Або натисніть «Вставити дані» — два стовпці з Excel (x | y)
  • Мінімум: 4 пари значень

КРОК 2. ВИБІР МОДЕЛІ
  Лінійна (y = a + bx):
    Коли залежність пряма — з ростом x, y рівномірно росте або спадає.
    Найпоширеніша. Починайте з неї.

  Квадратична (y = a + bx + cx²):
    Коли є оптимум — крива з одним піком або западиною.
    Типово для доз добрив, щільності посіву — є оптимальна доза.

  Кубічна (y = a + bx + cx² + dx³):
    Складніша крива з S-подібним характером.
    Використовуйте якщо квадратична дає погану підгонку.

  Степенева (y = a·xᵇ):
    Вимагає x > 0. Для алометричних залежностей (маса-розмір).

  Експоненційна (y = a·eᵇˣ):
    Для процесів росту (b > 0) або спаду (b < 0).

  Логарифмічна (y = a + b·ln(x)):
    Вимагає x > 0. Коли ефект насичується з ростом x.

  Логістична 4-параметрична:
    S-подібна крива. Для доза-відповідь, росту популяцій.
    Параметри: a = верхня асимптота, d = нижня,
               c = точка перегину, b = крутизна.

КРОК 3. ВИКОНАННЯ АНАЛІЗУ
  Натисніть «▶ Виконати» і переглядайте результати.

КРОК 4. ІНТЕРПРЕТАЦІЯ РЕЗУЛЬТАТІВ

  Рівняння регресії:
    Математична формула залежності y від x.
    Підставте будь-яке значення x щоб отримати прогноз y.

  R² (коефіцієнт детермінації):
    Від 0 до 1. Показує яку частку варіації y пояснює модель.
    R² = 0.85 → модель пояснює 85% мінливості y.
    R² > 0.90 → відмінна підгонка для агрономічних даних.
    R² < 0.50 → модель слабка, шукайте інші фактори.

  R²adj (скоригований R²):
    Враховує кількість параметрів моделі.
    При порівнянні моделей з різною кількістю параметрів
    орієнтуйтесь саме на R²adj, а не на R².
    Якщо R²adj < R² — модель надто складна для ваших даних.

  RMSE (середньоквадратична похибка):
    В одиницях вимірювання y.
    Середнє відхилення прогнозу від факту.
    Менше RMSE → точніший прогноз.
    Наприклад: RMSE = 0.3 т/га означає що модель
    помиляється в середньому на ±0.3 т/га.

  F-тест (значущість моделі):
    p < 0.05 → модель значуща, залежність існує ✓
    p ≥ 0.05 → модель незначуща (можливо замало даних
               або залежності взагалі немає) ✗

  Shapiro–Wilk залишків:
    Перевіряє нормальність відхилень від моделі.
    p > 0.05 → залишки нормальні → модель коректна ✓
    p ≤ 0.05 → залишки ненормальні → перевірте наявність
               викидів або спробуйте іншу модель.

КРОК 5. ОЦІНКА ГРАФІКІВ

  Графік «Точкові дані + Крива регресії»:
    Точки — ваші спостереження.
    Червона лінія — підібрана модель.
    Рожева смуга — 95% довірчий інтервал прогнозу.
    Чим ближче точки до лінії → краща підгонка.

  Графік «Залишки vs Підібрані значення»:
    Залишки = різниця між фактом і прогнозом.
    Ідеальний випадок: точки хаотично розкидані
    навколо нуля без жодного патерну.
    ⚠ Якщо є патерн (дуга, воронка) → модель некоректна!

КРОК 6. ПОРІВНЯННЯ КІЛЬКОХ МОДЕЛЕЙ
  Запустіть аналіз послідовно для різних моделей.
  Оберіть ту де:
  1. R²adj найвищий
  2. RMSE найменший
  3. Залишки нормальні (SW p > 0.05)
  4. Немає патерну на графіку залишків
"""

    def __init__(self, parent, gs):
        self.win = tk.Toplevel(parent)
        self.win.title("Регресійний аналіз")
        set_icon(self.win)
        self.win.resizable(True, True)
        maximize_win(self.win)
        self.gs = gs
        self._fig = None
        self._graph_title = ""
        self._build()

    def _build(self):
        rf = ("Times New Roman", 11)

        # ── Toolbar ───────────────────────────────────────────
        top = tk.Frame(self.win, padx=8, pady=5, bg="#f5f5f5")
        top.pack(fill=tk.X)
        tk.Frame(top, bg="#e0e0e0", height=1).pack(fill=tk.X, side=tk.BOTTOM)

        tk.Label(top, text="Модель:", font=rf, bg="#f5f5f5").pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value=self.MODELS[0])
        ttk.Combobox(top, textvariable=self.model_var, values=self.MODELS,
                     state="readonly", width=42, font=rf).pack(side=tk.LEFT, padx=6)
        tk.Label(top, text="α:", font=rf, bg="#f5f5f5").pack(side=tk.LEFT, padx=(8,2))
        self.alpha_var = tk.StringVar(value="0.05")
        ttk.Combobox(top, textvariable=self.alpha_var,
                     values=["0.01","0.05","0.10"],
                     state="readonly", width=7).pack(side=tk.LEFT)
        tk.Button(top, text="📊 Попередній аналіз", bg="#1a6b1a", fg="white",
                  font=rf, relief=tk.FLAT, padx=8, pady=3, cursor="hand2",
                  command=self._recommend_model).pack(side=tk.LEFT, padx=(10,4))
        tk.Button(top, text="▶ Виконати", bg="#c62828", fg="white",
                  font=("Times New Roman",13), relief=tk.FLAT, padx=14, pady=3,
                  cursor="hand2", command=self._run).pack(side=tk.LEFT, padx=(4,4))
        tk.Button(top, text="📋 Вставити",
                  font=rf, relief=tk.FLAT, padx=8, pady=3, cursor="hand2",
                  command=self._paste).pack(side=tk.LEFT, padx=2)
        tk.Button(top, text="📂 Відкрити показники", bg="#1a6b8c", fg="white",
                  font=rf, relief=tk.FLAT, padx=8, pady=3, cursor="hand2",
                  command=self._open_indicators_from_journal).pack(side=tk.LEFT, padx=2)
        tk.Button(top, text="📚 Довідка", bg="#1a4b8c", fg="white",
                  font=rf, relief=tk.FLAT, padx=8, pady=3, cursor="hand2",
                  command=self._show_help).pack(side=tk.LEFT, padx=4)

        # ── ОСНОВНА ОБЛАСТЬ ───────────────────────────────────
        main = tk.Frame(self.win); main.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # ── ЛІВО: поля x і y ──────────────────────────────────
        left = tk.Frame(main, width=230); left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)
        hdr_f = tk.Frame(left, bg="#1a4b8c"); hdr_f.pack(fill=tk.X, pady=(0,4))
        tk.Label(hdr_f, text="  Дані", bg="#1a4b8c", fg="white",
                 font=("Times New Roman",11,"bold"), pady=5).pack(side=tk.LEFT)
        cf = tk.Frame(left); cf.pack(fill=tk.BOTH, expand=True)
        for ci, lbl in enumerate(["x  (незалежна)", "y  (залежна)"]):
            tk.Label(cf, text=lbl, font=("Times New Roman",10,"bold"),
                     fg="#1a4b8c").grid(row=0, column=ci, padx=3, pady=2)
        self.tx = tk.Text(cf, width=10, font=("Times New Roman",11),
                          relief=tk.FLAT, highlightthickness=1,
                          highlightbackground="#c0c0c0", highlightcolor="#1a4b8c")
        self.tx.grid(row=1, column=0, padx=3, pady=2, sticky="nsew")
        self.ty = tk.Text(cf, width=10, font=("Times New Roman",11),
                          relief=tk.FLAT, highlightthickness=1,
                          highlightbackground="#c0c0c0", highlightcolor="#1a4b8c")
        self.ty.grid(row=1, column=1, padx=3, pady=2, sticky="nsew")
        cf.rowconfigure(1, weight=1); cf.columnconfigure(0, weight=1); cf.columnconfigure(1, weight=1)
        tk.Label(left, text="Одне значення на рядок\nабо вставте два стовпці з Excel.",
                 font=("Times New Roman",8), fg="#888", justify="left"
                 ).pack(anchor="w", padx=4, pady=2)

        # Роздільник
        tk.Frame(main, bg="#e0e0e0", width=1).pack(side=tk.LEFT, fill=tk.Y, padx=4)

        # ── ПРАВО: результати (прокручувані) ──────────────────
        right_outer = tk.Frame(main); right_outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        _vsb = ttk.Scrollbar(right_outer, orient="vertical")
        _vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._res_cv = tk.Canvas(right_outer, highlightthickness=0, yscrollcommand=_vsb.set)
        self._res_cv.pack(fill=tk.BOTH, expand=True)
        _vsb.config(command=self._res_cv.yview)
        self.res_frame = tk.Frame(self._res_cv)
        _wid = self._res_cv.create_window((0,0), window=self.res_frame, anchor="nw")
        self.res_frame.bind("<Configure>",
            lambda e: self._res_cv.configure(scrollregion=self._res_cv.bbox("all")))
        self._res_cv.bind("<Configure>",
            lambda e: self._res_cv.itemconfig(_wid, width=e.width))
        def _mw(e): self._res_cv.yview_scroll(int(-1*(e.delta/120)),"units")
        self._res_cv.bind("<MouseWheel>", _mw)
        self.res_frame.bind("<MouseWheel>", _mw)

        def _bind_mw_all(w):
            try: w.bind("<MouseWheel>", _mw)
            except Exception: pass
            for ch in w.winfo_children(): _bind_mw_all(ch)

        def _on_res_configure(e):
            self._res_cv.configure(scrollregion=self._res_cv.bbox("all"))
            _bind_mw_all(self.res_frame)
        self.res_frame.bind("<Configure>", _on_res_configure)
        self._res_cv.bind("<Configure>",
            lambda e: self._res_cv.itemconfig(_wid, width=e.width))

        tk.Label(self.res_frame,
                 text="Введіть дані, оберіть модель і натисніть  ▶ Виконати",
                 font=("Times New Roman",12), fg="#aaa").pack(expand=True, pady=40)

    def _graph_settings(self):
        if self._fig is None:
            messagebox.showinfo("","Спочатку виконайте аналіз."); return
        dlg = tk.Toplevel(self.win); dlg.title("Налаштування графіка регресії")
        dlg.resizable(False, False); dlg.grab_set(); set_icon(dlg)
        rf = ("Times New Roman",11)
        frm = tk.Frame(dlg, padx=16, pady=12); frm.pack()

        tk.Label(frm, text="Заголовок графіка:", font=rf
                 ).grid(row=0, column=0, sticky="w", pady=4)
        tv = tk.StringVar(value=self._graph_title)
        tk.Entry(frm, textvariable=tv, width=36, font=rf
                 ).grid(row=0, column=1, sticky="w", padx=8)

        # Підписи осей
        tk.Label(frm, text="Підпис осі X:", font=rf
                 ).grid(row=1, column=0, sticky="w", pady=4)
        xv = tk.StringVar(value=getattr(self, '_xlabel', 'x'))
        tk.Entry(frm, textvariable=xv, width=24, font=rf
                 ).grid(row=1, column=1, sticky="w", padx=8)

        tk.Label(frm, text="Підпис осі Y:", font=rf
                 ).grid(row=2, column=0, sticky="w", pady=4)
        yv = tk.StringVar(value=getattr(self, '_ylabel', 'y'))
        tk.Entry(frm, textvariable=yv, width=24, font=rf
                 ).grid(row=2, column=1, sticky="w", padx=8)

        # Шрифт графіка
        tk.Label(frm, text="Шрифт:", font=rf
                 ).grid(row=3, column=0, sticky="w", pady=4)
        fv = tk.StringVar(value=self.gs.get("font_family", "Times New Roman"))
        ttk.Combobox(frm, textvariable=fv, state="readonly", width=22,
                     values=["Times New Roman","Arial","Calibri","Georgia"]
                     ).grid(row=3, column=1, sticky="w", padx=8)

        tk.Label(frm, text="Розмір шрифту:", font=rf
                 ).grid(row=4, column=0, sticky="w", pady=4)
        fsv = tk.IntVar(value=self.gs.get("font_size", 9))
        tk.Spinbox(frm, from_=6, to=18, textvariable=fsv, width=6, font=rf
                   ).grid(row=4, column=1, sticky="w", padx=8)

        # Кольори — кожен на власному рядку, без колізій з полями вище
        color_vars = {}
        color_defs = [
            ("Колір точок:",       "scatter_color", "#4c72b0"),
            ("Колір кривої:",      "line_color",    "#c62828"),
            ("Колір ДІ (смуга):",  "ci_color",      "#c62828"),
        ]
        base_row = 5
        for i, (lbl, key, default) in enumerate(color_defs):
            ri = base_row + i
            tk.Label(frm, text=lbl, font=rf
                     ).grid(row=ri, column=0, sticky="w", pady=3)
            v = tk.StringVar(value=self.gs.get(key, default))
            color_vars[key] = v
            def _pick(var=v):
                c = colorchooser.askcolor(color=var.get(), parent=dlg)
                if c and c[1]: var.set(c[1])
            tk.Button(frm, text="Обрати колір", command=_pick, font=rf
                      ).grid(row=ri, column=1, sticky="w", padx=8)

        def _apply():
            self._graph_title = tv.get().strip()
            self._xlabel = xv.get().strip() or "x"
            self._ylabel = yv.get().strip() or "y"
            self.gs["font_family"] = fv.get()
            self.gs["font_size"]   = fsv.get()
            for key, v in color_vars.items():
                self.gs[key] = v.get()
            dlg.destroy()
            if hasattr(self, '_last_run_args'):
                self._show_result(*self._last_run_args)
        bf = tk.Frame(frm); bf.grid(row=base_row+len(color_defs)+1, column=0,
                                    columnspan=2, pady=(12,0))
        tk.Button(bf, text="Застосувати", bg="#c62828", fg="white",
                  font=rf, command=_apply).pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text="Скасувати", font=rf,
                  command=dlg.destroy).pack(side=tk.LEFT)
        center_win(dlg)

    def _save_png(self):
        if self._fig is None:
            messagebox.showinfo("","Спочатку виконайте аналіз."); return
        path = filedialog.asksaveasfilename(
            parent=self.win, defaultextension=".png",
            filetypes=[("PNG","*.png"),("SVG","*.svg")],
            title="Зберегти графік регресії")
        if not path: return
        try:
            self._fig.savefig(path, dpi=150, bbox_inches="tight")
            messagebox.showinfo("Збережено", f"Збережено:\n{path}")
        except Exception as ex:
            messagebox.showerror("Помилка", str(ex))

    # ── Утиліти ──────────────────────────────────────────────
    def _open_indicators_from_journal(self):
        """Завантажує ДВА показники з журналу, усереднені до рівня варіанту
        (спершу в межах повторності, потім по повторностях) — по одному
        значенню X і Y на кожен варіант, зіставлені за тим самим варіантом."""
        result = open_indicators_for_variant_analysis(self.win, multi_select=True, n_required=2)
        if result is None: return
        factor_cols, rows, record_names = result
        x_name, y_name = record_names[0], record_names[1]
        rows = [r for r in rows if r[x_name] is not None and r[y_name] is not None]
        if not rows:
            messagebox.showwarning("", "Немає жодного варіанту з обома показниками одночасно."); return

        self.tx.delete("1.0", tk.END); self.ty.delete("1.0", tk.END)
        self.tx.insert("1.0", "\n".join(str(r[x_name]) for r in rows))
        self.ty.insert("1.0", "\n".join(str(r[y_name]) for r in rows))
        messagebox.showinfo("Дані перенесено",
            f"X = «{x_name}», Y = «{y_name}» — по {len(rows)} варіантах "
            f"(середнє в межах повторності, потім по повторностях).")

    def _paste(self):
        try: data = self.win.clipboard_get()
        except Exception:
            messagebox.showwarning("","Буфер порожній."); return
        lines_ = [l.strip() for l in data.splitlines() if l.strip()]
        if not lines_: return
        xs, ys = [], []
        for line in lines_:
            # Розбиваємо по Tab, потім по пробілу/комі
            parts = line.replace(",", ".").split("\t")
            if len(parts) == 1:
                parts = line.replace(",", ".").split()
            if len(parts) >= 2:
                xs.append(parts[0].strip())
                ys.append(parts[1].strip())
            elif len(parts) == 1:
                xs.append(parts[0].strip())
        if not xs:
            messagebox.showwarning("","Не вдалося розпізнати дані.\n"
                "Скопіюйте два стовпці (x, y) з Excel."); return
        self.tx.delete("1.0", tk.END)
        self.tx.insert("1.0", "\n".join(xs))
        if ys:
            self.ty.delete("1.0", tk.END)
            self.ty.insert("1.0", "\n".join(ys))

    def _parse_col(self, widget):
        import re
        txt = widget.get("1.0", tk.END).replace(",",".")
        return np.array([float(v) for v in
                         re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", txt)], dtype=float)

    def _copy_graph(self):
        if self._fig is None:
            messagebox.showwarning("", "Спочатку виконайте аналіз."); return
        ok, msg = _copy_fig_to_clipboard(self._fig)
        if ok: messagebox.showinfo("", "Графік скопійовано (PNG).\nВставте у Word через Ctrl+V.")
        else:   messagebox.showwarning("", f"Помилка копіювання: {msg}")

    def _show_help(self):
        win = tk.Toplevel(self.win)
        win.title("Довідка — Регресійний аналіз")
        win.geometry("680x620"); set_icon(win)
        frm = tk.Frame(win); frm.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        vsb = ttk.Scrollbar(frm, orient="vertical"); vsb.pack(side=tk.RIGHT, fill=tk.Y)
        txt = tk.Text(frm, wrap="word", font=("Times New Roman",11),
                      yscrollcommand=vsb.set, relief=tk.FLAT, bg="#fafafa",
                      padx=10, pady=8, cursor="arrow")
        txt.pack(fill=tk.BOTH, expand=True)
        vsb.config(command=txt.yview)
        txt.insert("1.0", self.HELP_TEXT.strip())
        txt.configure(state="disabled")
        txt.bind("<MouseWheel>", lambda e: txt.yview_scroll(int(-1*(e.delta/120)), "units"))
        tk.Button(win, text="Закрити", command=win.destroy,
                  font=("Times New Roman",11)).pack(pady=6)

    # ── Попередній аналіз: діаграма розсіювання + рекомендація ──
    def _recommend_model(self):
        x = self._parse_col(self.tx); y = self._parse_col(self.ty)
        n = min(len(x), len(y)); x = x[:n]; y = y[:n]
        if n < 4:
            messagebox.showwarning("Замало даних",
                "Потрібно ≥ 4 пари значень (x, y) для попереднього аналізу."); return

        candidates = []
        for model_full in self.MODELS:
            name = model_full.split(":")[0].strip()
            r = self._fit_model(name, x, y, alpha=0.05, silent=True)
            if r is not None:
                candidates.append((name, model_full, r))

        if not candidates:
            messagebox.showwarning("", "Жодну з моделей не вдалося підібрати до цих даних."); return

        def sort_key(item):
            _, _, r = item
            r2a = r["R2_adj"]
            r2a = r2a if not math.isnan(r2a) else r["R2"]
            r2a = r2a if not math.isnan(r2a) else -1.0
            rmse = r["RMSE"] if not math.isnan(r["RMSE"]) else float("inf")
            return (-r2a, rmse)
        candidates.sort(key=sort_key)
        best_name = candidates[0][0]

        self._show_recommendation(x, y, candidates, best_name)

    def _show_recommendation(self, x, y, candidates, best_name):
        win = tk.Toplevel(self.win)
        win.title("Попередній аналіз — рекомендація моделі")
        win.geometry("1100x760"); set_icon(win)
        rf = ("Times New Roman", 11)

        tk.Label(win, text="Діаграма розсіювання (без підгонки) — оцініть форму залежності "
                          "на око, перш ніж обирати модель:",
                 font=("Times New Roman",11,"bold"), anchor="w"
                 ).pack(fill=tk.X, padx=10, pady=(10,2))

        plot_f = tk.Frame(win); plot_f.pack(fill=tk.BOTH, expand=False)
        fig = Figure(figsize=(9, 3.6), dpi=100)
        ax = fig.add_subplot(111)
        ax.scatter(x, y, s=28, color="#4c72b0", edgecolors="white", linewidths=0.5, zorder=3)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.yaxis.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.tight_layout()
        embed_figure(fig, plot_f)

        tk.Label(win, text="Порівняння моделей (відсортовано за R²adj — враховує кількість "
                          "параметрів; проста R² завжди зростає з ускладненням моделі):",
                 font=("Times New Roman",11,"bold"), anchor="w"
                 ).pack(fill=tk.X, padx=10, pady=(10,2))

        rows = []
        for name, model_full, r in candidates:
            mark = "★ " if name == best_name else "  "
            r2a_txt = fmt(r["R2_adj"],4) if not math.isnan(r["R2_adj"]) else "—"
            rows.append([mark+name, fmt(r["R2"],4), r2a_txt,
                        fmt(r["RMSE"],4) if not math.isnan(r["RMSE"]) else "—",
                        fmt(r["sw_p"],4) if not math.isnan(r["sw_p"]) else "—"])
        tbl_frm, _ = make_tv(win, ["Модель","R²","R²adj","RMSE","Shapiro-Wilk p (залишків)"], rows)
        tbl_frm.pack(fill=tk.X, padx=10, pady=(0,8))

        tk.Label(win,
                 text="⚠ Це орієнтир, а не остаточне рішення. Рекомендація ґрунтується лише на "
                      "статистичній підгонці (R²adj/RMSE) — обов'язково враховуйте форму точок "
                      "на діаграмі вище та практичний сенс моделі для вашого явища (наприклад, "
                      "чи логічно очікувати оптимум/насичення/S-подібність саме тут). "
                      "Shapiro-Wilk p ≤ 0.05 у залишків — сигнал, що модель, ймовірно, не підходить, "
                      "навіть якщо R² високий.",
                 font=("Times New Roman",10), fg="#555", justify="left", wraplength=1060, anchor="w"
                 ).pack(fill=tk.X, padx=10, pady=(0,8))

        bf = tk.Frame(win); bf.pack(pady=(0,10))
        def _apply():
            for model_full in self.MODELS:
                if model_full.split(":")[0].strip() == best_name:
                    self.model_var.set(model_full); break
            win.destroy()
        tk.Button(bf, text=f"Обрати «{best_name}» і закрити", bg="#1a6b1a", fg="white",
                  font=rf, command=_apply).pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text="Закрити (оберу модель сам)", font=rf,
                  command=win.destroy).pack(side=tk.LEFT, padx=4)
        center_win(win)

    # ── Виконання аналізу ─────────────────────────────────────
    def _run(self):
        alpha = float(self.alpha_var.get())
        x = self._parse_col(self.tx); y = self._parse_col(self.ty)
        n = min(len(x), len(y)); x = x[:n]; y = y[:n]
        if n < 4:
            messagebox.showwarning("Замало даних",
                "Потрібно ≥ 4 пари значень (x, y)."); return

        model_name = self.model_var.get().split(":")[0].strip()
        result = self._fit_model(model_name, x, y, alpha)
        if result is None: return
        self._show_result(result, x, y, model_name, alpha)

    def _fit_model(self, name, x, y, alpha, silent=False):
        from scipy.optimize import curve_fit
        n_ = name.strip().lower()
        try:
            if "лінійна" in n_ or n_ == "linear":
                # ── Лінійна ─────────────────────────────────
                X = np.column_stack([np.ones(len(x)), x])
                # lstsq повертає (beta, residuals_sum, rank, sv) — 4 значення
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                yhat = X @ beta
                params = {"a": beta[0], "b": beta[1]}
                eq = f"y = {fmt(beta[0],4)} + {fmt(beta[1],4)}·x"
                k = 2

            elif "квадратична" in n_ or n_ == "quadratic":
                # ── Квадратична ──────────────────────────────
                X = np.column_stack([np.ones(len(x)), x, x**2])
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                yhat = X @ beta
                params = {"a": beta[0], "b": beta[1], "c": beta[2]}
                eq = f"y = {fmt(beta[0],4)} + {fmt(beta[1],4)}·x + {fmt(beta[2],4)}·x²"
                k = 3

            elif "кубічна" in n_ or n_ == "cubic":
                # ── Кубічна ──────────────────────────────────
                X = np.column_stack([np.ones(len(x)), x, x**2, x**3])
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                yhat = X @ beta
                params = {"a": beta[0], "b": beta[1], "c": beta[2], "d": beta[3]}
                eq = f"y = {fmt(beta[0],4)} + {fmt(beta[1],4)}·x + {fmt(beta[2],4)}·x² + {fmt(beta[3],4)}·x³"
                k = 4

            elif "степенева" in n_ or n_ == "power":
                # ── Степенева ────────────────────────────────
                if np.any(x <= 0):
                    if not silent:
                        messagebox.showwarning("Обмеження моделі",
                            "Степенева модель вимагає x > 0 для всіх спостережень.")
                    return None
                lx = np.log(x); ly = np.log(np.abs(y) + 1e-12)
                X = np.column_stack([np.ones(len(lx)), lx])
                beta = np.linalg.lstsq(X, ly, rcond=None)[0]
                a, b = math.exp(beta[0]), beta[1]
                yhat = a * x**b
                params = {"a": a, "b": b}
                eq = f"y = {fmt(a,4)}·x^{fmt(b,4)}"
                k = 2

            elif "експоненційна" in n_ or n_ == "exponential":
                # ── Експоненційна ─────────────────────────────
                X = np.column_stack([np.ones(len(x)), x])
                ly = np.log(np.abs(y) + 1e-12)
                beta = np.linalg.lstsq(X, ly, rcond=None)[0]
                a, b = math.exp(beta[0]), beta[1]
                yhat = a * np.exp(b * x)
                params = {"a": a, "b": b}
                eq = f"y = {fmt(a,4)}·e^({fmt(b,4)}·x)"
                k = 2

            elif "логарифмічна" in n_ or n_ == "logarithmic":
                # ── Логарифмічна ──────────────────────────────
                if np.any(x <= 0):
                    if not silent:
                        messagebox.showwarning("Обмеження моделі",
                            "Логарифмічна модель вимагає x > 0 для всіх спостережень.")
                    return None
                X = np.column_stack([np.ones(len(x)), np.log(x)])
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                yhat = X @ beta
                params = {"a": beta[0], "b": beta[1]}
                eq = f"y = {fmt(beta[0],4)} + {fmt(beta[1],4)}·ln(x)"
                k = 2

            elif "логістична" in n_ or "logistic" in n_:
                # ── Логістична 4-параметрична ─────────────────
                def logistic4(xx, a, b, c, d):
                    return d + (a - d) / (1 + (xx / c) ** b)
                p0 = [float(np.max(y)), 1.0, float(np.median(x)), float(np.min(y))]
                popt, _ = curve_fit(logistic4, x, y, p0=p0, maxfev=15000)
                yhat = logistic4(x, *popt)
                params = {"a": popt[0], "b": popt[1], "c": popt[2], "d": popt[3]}
                eq = (f"y = {fmt(popt[3],4)} + ({fmt(popt[0],4)}−{fmt(popt[3],4)})"
                      f"/(1+(x/{fmt(popt[2],4)})^{fmt(popt[1],4)})")
                k = 4

            else:
                if not silent:
                    messagebox.showerror("Невідома модель",
                        f"Модель '{name}' не розпізнана.\nОберіть модель зі списку.")
                return None

            # ── Загальна статистика ───────────────────────────
            residuals = y - yhat
            sse = float(np.sum(residuals**2))
            sst = float(np.sum((y - np.mean(y))**2))
            n_obs = len(x)
            R2     = 1 - sse / sst if sst > 0 else np.nan
            R2_adj = 1 - (1 - R2) * (n_obs - 1) / (n_obs - k - 1) if n_obs > k + 1 else np.nan
            mse    = sse / (n_obs - k) if n_obs > k else np.nan
            rmse   = math.sqrt(mse) if not math.isnan(mse) else np.nan
            # F-тест
            msm = (sst - sse) / k if k > 0 else np.nan
            F   = msm / mse if (not math.isnan(mse) and mse > 1e-12) else np.nan
            p_F = float(1 - f_dist.cdf(F, k, n_obs - k - 1)) \
                  if (not math.isnan(F) and n_obs > k + 1) else np.nan
            # Нормальність залишків
            try:
                _, sw_p = shapiro(residuals) if len(residuals) >= 3 else (np.nan, np.nan)
            except Exception:
                sw_p = np.nan

            return {"equation": eq, "params": params,
                    "R2": R2, "R2_adj": R2_adj,
                    "RMSE": rmse, "F": F, "p_F": p_F, "sw_p": sw_p,
                    "residuals": residuals, "yhat": yhat,
                    "sse": sse, "sst": sst, "n": n_obs, "k": k}

        except Exception as ex:
            if not silent:
                messagebox.showerror("Помилка підгонки", str(ex))
            return None

    # ── Відображення результатів ──────────────────────────────
    def _show_result(self, r, x, y, model_name, alpha):
        self._last_run_args = (r, x, y, model_name, alpha)
        for w in self.res_frame.winfo_children(): w.destroy()
        # Reset canvas scroll
        self._res_cv.yview_moveto(0)

        p_F_ok  = (not math.isnan(r['p_F'])  and r['p_F']  < alpha)
        sw_ok   = (not math.isnan(r['sw_p']) and r['sw_p'] > alpha)
        ci_pct  = int((1 - alpha) * 100)
        n_pts   = r["n"]; k = r.get("k", 2); rmse_ = r.get("RMSE", np.nan)
        t_crit_ = float(t_dist.ppf(1 - alpha/2, max(1, n_pts - k - 1)))

        def _copy_fig(fig):
            ok, msg = _copy_fig_to_clipboard(fig)
            if ok: messagebox.showinfo("", "Скопійовано. Вставте у Word через Ctrl+V.")
            else:   messagebox.showwarning("", f"Помилка: {msg}")

        def _save_fig(fig, name):
            path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG","*.png"),("SVG","*.svg")],
                title=f"Зберегти {name}")
            if not path: return
            try:
                fig.savefig(path, dpi=150, bbox_inches="tight")
                messagebox.showinfo("Збережено", f"Збережено:\n{path}")
            except Exception as ex:
                messagebox.showerror("Помилка", str(ex))

        # ── РЯД 1: Текст ліво + Графік регресії право ─────────
        row1 = tk.Frame(self.res_frame, height=420)
        row1.pack(fill=tk.X)
        row1.pack_propagate(False)

        # Текстовий звіт (ліворуч у row1)
        txt_f = tk.Frame(row1, bg="#f8f8f8", width=310)
        txt_f.pack(side=tk.LEFT, fill=tk.Y)
        txt_f.pack_propagate(False)

        tk.Label(txt_f, text="РЕЗУЛЬТАТИ РЕГРЕСІЇ",
                 font=("Times New Roman",11,"bold"), fg="#1a4b8c",
                 bg="#f8f8f8", pady=6).pack(anchor="w", padx=8)

        fields = [
            ("Модель:",      model_name),
            ("n:",           str(r['n'])),
            ("α:",           str(alpha)),
            ("Рівняння:",    r['equation']),
            ("R²:",          fmt(r['R2'],4)),
            ("R²adj:",       fmt(r['R2_adj'],4)),
            ("RMSE:",        fmt(r['RMSE'],4)),
            ("F:",           fmt(r['F'],4)),
            ("p (F-тест):",  fmt(r['p_F'],4)),
            ("Значущість:",  "✓ Значуща" if p_F_ok else "✗ Незначуща"),
            (f"ДІ:",         f"{ci_pct}%"),
            ("SW (залишки):", fmt(r['sw_p'],4)),
            ("Нормальність:", "✓ Нормальні" if sw_ok else "⚠ Не норм."),
        ]
        for lbl, val in fields:
            row = tk.Frame(txt_f, bg="#f8f8f8"); row.pack(fill=tk.X, padx=8, pady=1)
            tk.Label(row, text=lbl, font=("Times New Roman",10,"bold"),
                     bg="#f8f8f8", fg="#555", width=14, anchor="w").pack(side=tk.LEFT)
            color = ("#27ae60" if "✓" in val else
                     "#c62828" if ("✗" in val or "⚠" in val) else "#000")
            tk.Label(row, text=val, font=("Times New Roman",10),
                     bg="#f8f8f8", fg=color,
                     wraplength=160, justify="left", anchor="w").pack(side=tk.LEFT)

        # Графік регресії (праворуч у row1)
        if not HAS_MPL:
            messagebox.showwarning("","matplotlib недоступний."); return

        g1_outer = tk.Frame(row1)
        g1_outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Toolbar графіка 1
        tb1 = tk.Frame(g1_outer, bg="#f0f0f0", padx=4, pady=2); tb1.pack(fill=tk.X)
        tk.Label(tb1, text="Графік регресії",
                 font=("Times New Roman",10,"bold"), bg="#f0f0f0").pack(side=tk.LEFT, padx=4)
        tk.Button(tb1, text="💾 Зберегти",
                  font=("Times New Roman",9), relief=tk.FLAT, padx=6,
                  command=lambda: _save_fig(fig1,"графік_регресії")).pack(side=tk.RIGHT, padx=2)
        tk.Button(tb1, text="📋 Копіювати",
                  font=("Times New Roman",9), relief=tk.FLAT, padx=6,
                  command=lambda: _copy_fig(fig1)).pack(side=tk.RIGHT, padx=2)
        tk.Button(tb1, text="⚙ Налаштування",
                  font=("Times New Roman",9), relief=tk.FLAT, padx=6,
                  bg="#1a4b8c", fg="white",
                  command=self._graph_settings).pack(side=tk.RIGHT, padx=2)

        # Figure 1
        ff = self.gs.get("font_family", "Times New Roman")
        fz = self.gs.get("font_size", 9)
        fig1 = Figure(figsize=(5.2, 4.2), dpi=100)
        ax1  = fig1.add_subplot(111)
        x_sort   = np.sort(x); idx_sort = np.argsort(x)
        ax1.scatter(x, y, s=30,
                    color=self.gs.get("scatter_color","#4c72b0"),
                    zorder=3, label="Спостереження",
                    edgecolors="white", linewidths=0.5)
        ax1.plot(x_sort, r["yhat"][idx_sort],
                 color=self.gs.get("line_color","#c62828"),
                 lw=2, label="Регресійна крива")
        if n_pts > k + 2 and not math.isnan(rmse_):
            try:
                x_pred  = np.linspace(x.min(), x.max(), 300)
                x_mean_ = float(np.mean(x))
                ss_xx   = float(np.sum((x - x_mean_)**2))
                if ss_xx > 0:
                    se_fit = rmse_ * np.sqrt(1/n_pts + (x_pred-x_mean_)**2/ss_xx)
                    yhat_p = np.interp(x_pred, x_sort, r["yhat"][idx_sort])
                    ax1.fill_between(x_pred,
                                     yhat_p - t_crit_*se_fit,
                                     yhat_p + t_crit_*se_fit,
                                     alpha=0.12,
                                     color=self.gs.get("ci_color","#c62828"),
                                     label=f"{ci_pct}% ДІ")
            except Exception: pass
        custom_title = getattr(self, '_graph_title', '')
        ax1.set_title(custom_title if custom_title else f"{model_name}",
                      fontsize=fz+1, fontfamily=ff)
        ax1.set_xlabel(getattr(self,'_xlabel','x'), fontsize=fz, fontfamily=ff)
        ax1.set_ylabel(getattr(self,'_ylabel','y'), fontsize=fz, fontfamily=ff)
        ax1.tick_params(labelsize=max(6, fz-1))
        ax1.legend(fontsize=max(6, fz-1))
        r2_str = f"R²={fmt(r['R2'],4)}"
        if not math.isnan(r.get("R2_adj",float("nan"))):
            r2_str += f"  R²adj={fmt(r['R2_adj'],4)}"
        eq = r.get("equation","")
        ax1.text(0.03, 0.97, f"{eq}\n{r2_str}",
                 transform=ax1.transAxes, fontsize=max(6, fz-1), va="top",
                 fontfamily=ff,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#eef4ff",
                           edgecolor="#1a4b8c", alpha=0.9, linewidth=1),
                 zorder=5)
        ax1.yaxis.grid(True, linestyle="--", alpha=0.35)
        ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
        fig1.tight_layout()
        self._fig = fig1
        embed_figure(fig1, g1_outer)

        # ── РЯД 2: Аналіз залишків ─────────────────────────────
        tk.Frame(self.res_frame, bg="#e0e0e0", height=1).pack(fill=tk.X, pady=4)

        row2 = tk.Frame(self.res_frame); row2.pack(fill=tk.BOTH, expand=False)
        tb2 = tk.Frame(row2, bg="#f0f0f0", padx=4, pady=2); tb2.pack(fill=tk.X)
        tk.Label(tb2, text="Аналіз залишків",
                 font=("Times New Roman",10,"bold"), bg="#f0f0f0").pack(side=tk.LEFT, padx=4)

        # Settings for residuals graph
        def _res_settings():
            dlg = tk.Toplevel(self.win); dlg.title("Налаштування залишків")
            dlg.resizable(False,False); dlg.grab_set(); set_icon(dlg)
            rf2 = ("Times New Roman",11)
            frm = tk.Frame(dlg, padx=14, pady=12); frm.pack()
            tk.Label(frm, text="Заголовок графіка:", font=rf2
                     ).grid(row=0, column=0, sticky="w", pady=4)
            tv = tk.StringVar(value=getattr(self,'_res_title',''))
            tk.Entry(frm, textvariable=tv, width=30, font=rf2
                     ).grid(row=0, column=1, sticky="w", padx=8)
            def _ok():
                self._res_title = tv.get().strip()
                dlg.destroy()
                self._show_result(*self._last_run_args)
            bf = tk.Frame(frm); bf.grid(row=1, column=0, columnspan=2, pady=(10,0))
            tk.Button(bf, text="Застосувати", bg="#c62828", fg="white",
                      font=rf2, command=_ok).pack(side=tk.LEFT, padx=4)
            tk.Button(bf, text="Скасувати", font=rf2,
                      command=dlg.destroy).pack(side=tk.LEFT)
            center_win(dlg)

        tk.Button(tb2, text="💾 Зберегти",
                  font=("Times New Roman",9), relief=tk.FLAT, padx=6,
                  command=lambda: _save_fig(fig2,"аналіз_залишків")).pack(side=tk.RIGHT, padx=2)
        tk.Button(tb2, text="📋 Копіювати",
                  font=("Times New Roman",9), relief=tk.FLAT, padx=6,
                  command=lambda: _copy_fig(fig2)).pack(side=tk.RIGHT, padx=2)
        tk.Button(tb2, text="⚙ Налаштування",
                  font=("Times New Roman",9), relief=tk.FLAT, padx=6,
                  bg="#1a4b8c", fg="white",
                  command=_res_settings).pack(side=tk.RIGHT, padx=2)

        res_arr = np.array(r["residuals"])
        res_title = getattr(self, '_res_title', '') or "Аналіз залишків"
        fig2 = Figure(figsize=(10, 3.5), dpi=100)
        fig2.suptitle(res_title, fontsize=10, y=1.0)

        ax2 = fig2.add_subplot(131)
        ax2.scatter(r["yhat"], res_arr, s=22, color="#dd8452",
                    edgecolors="white", linewidths=0.4, zorder=3)
        ax2.axhline(0, color="#333", lw=0.9, linestyle="--")
        if not math.isnan(rmse_):
            for sgn in [1,-1]:
                ax2.axhline(sgn*rmse_, color="#aaa", lw=0.6, linestyle=":")
            ax2.text(0.98, 0.98, f"±RMSE={fmt(rmse_,3)}",
                     transform=ax2.transAxes, fontsize=7, ha="right", va="top", color="#888")
        ax2.set_xlabel("ŷ", fontsize=9); ax2.set_ylabel("e = y−ŷ", fontsize=9)
        ax2.set_title("Залишки vs ŷ", fontsize=9, pad=4)
        ax2.yaxis.grid(True, linestyle="--", alpha=0.3)
        ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

        ax3 = fig2.add_subplot(132)
        ax3.hist(res_arr, bins="auto", color="#4c72b0", edgecolor="white", alpha=0.85)
        try:
            from scipy.stats import norm as _nd
            mu_, sig_ = float(np.mean(res_arr)), float(np.std(res_arr, ddof=1))
            xn = np.linspace(res_arr.min(), res_arr.max(), 100)
            bw = (res_arr.max()-res_arr.min()) / max(1, len(np.histogram_bin_edges(res_arr,"auto"))-1)
            ax3.plot(xn, _nd.pdf(xn,mu_,sig_)*len(res_arr)*bw,
                     color="#c62828", lw=1.5)
        except Exception: pass
        ax3.set_xlabel("Залишок", fontsize=9); ax3.set_ylabel("Частота", fontsize=9)
        ax3.set_title(f"Гістограма  (SW p={fmt(r['sw_p'],4)})", fontsize=9, pad=4)
        ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

        ax4 = fig2.add_subplot(133)
        try:
            from scipy.stats import probplot
            (osm,osr),(slope,intercept,rval) = probplot(res_arr, plot=None)
            ax4.plot(osm, osr, "o", color="#4c72b0", markersize=4, alpha=0.8)
            ax4.plot([min(osm),max(osm)],
                     [slope*min(osm)+intercept, slope*max(osm)+intercept],
                     color="#c62828", lw=1.5)
            ax4.set_title(f"Q-Q  (R²={rval**2:.3f})", fontsize=9, pad=4)
        except Exception:
            ax4.set_title("Q-Q", fontsize=9, pad=4)
        ax4.set_xlabel("Теор. квантилі", fontsize=9)
        ax4.set_ylabel("Вибірк. квантилі", fontsize=9)
        ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)

        fig2.tight_layout()
        self._fig2 = fig2
        embed_figure(fig2, row2)

        # ── Виявлення викидів ─────────────────────────────────
        out_idx, G, _ = detect_outliers_grubbs(r["residuals"])
        if out_idx is not None:
            tk.Label(self.res_frame,
                     text=(f"⚠ Тест Граббса: підозрілий викид — спостереження "
                           f"№{out_idx+1}  (G = {fmt(G,3)}). Перевірте дані."),
                     fg="#c62828", font=("Times New Roman",10),
                     justify="left", padx=8).pack(anchor="w", pady=4)

        out_idx, G, _ = detect_outliers_grubbs(r["residuals"])
        if out_idx is not None:
            tk.Label(self.res_frame,
                     text=(f"⚠ Тест Граббса: підозрілий викид у залишках — "
                           f"спостереження №{out_idx+1}  (G = {fmt(G,3)}).\n"
                           f"   Перевірте це значення у вхідних даних."),
                     fg="#c62828", font=("Times New Roman",11),
                     justify="left", padx=6).pack(anchor="w", pady=2)




# ═══════════════════════════════════════════════════════════════
# SAMPLE SIZE CALCULATOR
# ═══════════════════════════════════════════════════════════════
class SampleSizeWindow:
    """Калькулятор розміру вибірки та статистичної потужності."""

    HELP_TEXT = """
КАЛЬКУЛЯТОР РОЗМІРУ ВИБІРКИ — ІНСТРУКЦІЯ
═════════════════════════════════════════

ДЛЯ ЧОГО ЦЕЙ КАЛЬКУЛЯТОР?
  Перед початком досліду відповідає на питання:
  "Скільки повторностей (r) мені потрібно щоб надійно
  виявити реальну різницю між варіантами?"

  Або навпаки:
  "Якщо я маю r повторностей — яка ймовірність
  що я виявлю різницю якщо вона є?"

ПАРАМЕТРИ:

  Дизайн:
    CRD — повністю рандомізований дослід
    RCBD — рандомізований повний блок
    Split-plot — розщеплені ділянки

  α (рівень значущості):
    Ймовірність хибного виявлення різниці (помилка I роду).
    Стандарт: 0.05 (5%).

  Потужність (1-β):
    Ймовірність виявити реальну різницю якщо вона існує.
    Стандарт: 0.80 (80%). Краще: 0.90 (90%).

  δ (очікувана різниця):
    Мінімальна різниця між варіантами яку важливо виявити.
    В одиницях вашого показника (т/га, см, %).
    Наприклад: δ = 0.5 т/га означає що хочемо виявити різницю ≥ 0.5 т/га.

  σ (стандартне відхилення):
    Варіабельність вашого показника.
    Візьміть з попередніх дослідів або пілотного досліду.
    Або: σ ≈ CV% × Середнє / 100.

  k (кількість варіантів):
    Скільки варіантів (обробок, сортів) у досліді.

  r (кількість повторностей):
    Залиште ПОРОЖНІМ → калькулятор знайде мінімальне r.
    Введіть число → калькулятор розрахує досягнуту потужність.

ПРИКЛАД:
  Дослід з 4 дозами добрива, очікуємо різницю ≥ 0.5 т/га,
  SD з попередніх дослідів = 0.8 т/га.
  Введіть: k=4, δ=0.5, σ=0.8, α=0.05, потужність=0.80.
  Залиште r порожнім.
  Результат покаже скільки повторностей потрібно.

ІНТЕРПРЕТАЦІЯ РЕЗУЛЬТАТУ:
  r = 4 → потрібно 4 повторності на кожен варіант.
  Загальна кількість = k × r ділянок у досліді.
  Досягнута потужність = 0.83 → 83% шанс виявити різницю.

ЯКЩО ПОТРІБНО ЗАБАГАТО ПОВТОРНОСТЕЙ:
  Збільшіть δ (нижня межа практично значущої різниці)
  або зменшіть σ (точніші вимірювання, однорідніші умови).
"""

    def __init__(self, parent):
        self.win = tk.Toplevel(parent)
        self.win.title("Калькулятор розміру вибірки та потужності")
        self.win.geometry("680x700")
        self.win.resizable(True, True)
        set_icon(self.win)
        self._build()

    def _build(self):
        top = tk.Frame(self.win, padx=8, pady=6); top.pack(fill=tk.X)
        tk.Button(top, text="▶ Розрахувати", bg="#c62828", fg="white",
                  font=("Times New Roman",13),
                  command=self._calc).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="📚 Довідка", bg="#1a4b8c", fg="white",
                  font=("Times New Roman",11),
                  command=self._show_help).pack(side=tk.LEFT, padx=4)

        pfrm = tk.LabelFrame(self.win, text="Параметри досліду",
                             font=("Times New Roman",11,"bold"),
                             padx=12, pady=8)
        pfrm.pack(fill=tk.X, padx=10, pady=(0,6))
        rf = ("Times New Roman",12)

        params = [
            ("Дизайн досліду:",                     None,   "design"),
            ("α — рівень значущості:",              "0.05", "alpha"),
            ("Потужність (1-β):",                   "0.80", "power"),
            ("δ — мінімальна різниця яку виявити:", "",     "delta"),
            ("σ — стандартне відхилення:",          "",     "sigma"),
            ("k — кількість варіантів:",            "3",    "k"),
            ("r — повторностей (порожньо→знайти):", "",     "r"),
        ]
        hints = {
            "alpha": "0.01 / 0.05 / 0.10",
            "power": "0.80 / 0.90",
            "delta": "в одиницях показника (т/га, см...)",
            "sigma": "SD з попередніх дослідів",
            "k":     "кількість варіантів/сортів",
            "r":     "порожньо = автоматичний розрахунок",
        }
        self.vars = {}
        for ri, (label, default, key) in enumerate(params):
            tk.Label(pfrm, text=label, font=rf, anchor="w"
                     ).grid(row=ri, column=0, sticky="w", pady=3)
            if key == "design":
                var = tk.StringVar(value="CRD")
                ttk.Combobox(pfrm, textvariable=var,
                             values=["CRD","RCBD","Split-plot"],
                             state="readonly", width=16,
                             font=rf).grid(row=ri, column=1, sticky="w", padx=6)
            else:
                var = tk.StringVar(value=default or "")
                tk.Entry(pfrm, textvariable=var, width=14,
                         font=rf).grid(row=ri, column=1, sticky="w", padx=6)
                if key in hints:
                    tk.Label(pfrm, text=hints[key],
                             font=("Times New Roman",9), fg="#888"
                             ).grid(row=ri, column=2, sticky="w", padx=4)
            self.vars[key] = var

        res_frm = tk.LabelFrame(self.win, text="Результат",
                                font=("Times New Roman",11,"bold"),
                                padx=8, pady=6)
        res_frm.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))
        vsb = ttk.Scrollbar(res_frm, orient="vertical"); vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.res_txt = tk.Text(res_frm, wrap="word",
                               font=("Courier New",11),
                               yscrollcommand=vsb.set,
                               relief=tk.FLAT, bg="#f8f8f8",
                               padx=8, pady=6, cursor="arrow",
                               state="disabled")
        self.res_txt.pack(fill=tk.BOTH, expand=True)
        vsb.config(command=self.res_txt.yview)

    def _set_result(self, text, color="#000000"):
        self.res_txt.configure(state="normal")
        self.res_txt.delete("1.0", tk.END)
        self.res_txt.insert("1.0", text)
        self.res_txt.configure(state="disabled", fg=color)

    def _show_help(self):
        win = tk.Toplevel(self.win)
        win.title("Довідка — Калькулятор вибірки")
        win.geometry("660x640"); set_icon(win)
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

    def _calc(self):
        try:
            alpha  = float(self.vars["alpha"].get())
            power  = float(self.vars["power"].get())
            delta  = float(self.vars["delta"].get())
            sigma  = float(self.vars["sigma"].get())
            k      = int(self.vars["k"].get())
            design = self.vars["design"].get()
            r_str  = self.vars["r"].get().strip()
        except ValueError:
            self._set_result(
                "Заповніть всі числові поля!\n\n"
                "delta i sigma — в одиницях вашого показника.\n"
                "k — ціле число >= 2.", "#c62828")
            return

        if delta <= 0 or sigma <= 0 or k < 2:
            self._set_result("delta та sigma мають бути > 0; k >= 2.", "#c62828"); return
        if not 0 < alpha < 1 or not 0 < power < 1:
            self._set_result("alpha і потужність мають бути між 0 і 1.", "#c62828"); return

        from scipy.stats import ncf
        lines = []; sep = "-" * 44

        if r_str:
            try: r = int(r_str)
            except ValueError:
                self._set_result("r має бути цілим числом.", "#c62828"); return
            if r < 2:
                self._set_result("r має бути >= 2.", "#c62828"); return

            lambda_nc = k * r * (delta**2) / (2 * sigma**2)
            F_crit = float(f_dist.ppf(1-alpha, k-1, k*(r-1)))
            ap = float(1 - ncf.cdf(F_crit, k-1, k*(r-1), lambda_nc))

            lines.append("РЕЖИМ: Розрахунок потужності при заданому r")
            lines.append(sep)
            lines.append(f"Дизайн: {design}  |  k={k}  |  r={r}")
            lines.append(f"delta={delta}  |  sigma={sigma}  |  alpha={alpha}")
            lines.append(sep)
            lines.append(f"Нецентральність lambda = {fmt(lambda_nc,3)}")
            lines.append(f"F критичне (alpha={alpha}) = {fmt(F_crit,3)}")
            lines.append(f"Досягнута потужність (1-beta) = {fmt(ap,4)}")
            lines.append("")
            if ap >= power:
                lines.append(f"OK Потужність ДОСТАТНЯ: {fmt(ap*100,1)}% >= {power*100:.0f}%")
                lines.append(f"При r={r} ви маєте {fmt(ap*100,1)}% шанс")
                lines.append(f"виявити різницю delta>={delta} якщо вона є.")
            else:
                lines.append(f"НЕДОСТАТНЯ: {fmt(ap*100,1)}% < {power*100:.0f}%")
                lines.append("Збільшіть r або delta, або зменшіть sigma.")
            if design == "RCBD":
                lines.append(f"\nRCBD: {r} блоків x {k} варіантів = {k*r} ділянок")
            elif design == "Split-plot":
                lines.append(f"\nSplit-plot: >= {r} блоків для WP фактора")
            else:
                lines.append(f"\nCRD: {k}x{r} = {k*r} ділянок")
        else:
            lines.append("РЕЖИМ: Пошук мінімальних повторностей")
            lines.append(sep)
            lines.append(f"Дизайн: {design}  |  k={k} варіантів")
            lines.append(f"Ціль: alpha={alpha}, потужність>={power}")
            lines.append(f"delta={delta}, sigma={sigma}")
            lines.append(sep)

            found = False
            for r in range(2, 101):
                lambda_nc = k * r * (delta**2) / (2 * sigma**2)
                try:
                    F_crit = float(f_dist.ppf(1-alpha, k-1, k*(r-1)))
                    pwr = float(1 - ncf.cdf(F_crit, k-1, k*(r-1), lambda_nc))
                except Exception: continue
                if pwr >= power:
                    lines.append(f"OK Мінімальне r = {r} повторностей")
                    lines.append(f"   Досягнута потужність: {fmt(pwr*100,1)}%")
                    lines.append(f"   Загальна кількість ділянок: {k}x{r} = {k*r}")
                    if design == "RCBD":
                        lines.append(f"   RCBD: {r} блоків, {k} варіантів у кожному")
                    elif design == "Split-plot":
                        lines.append(f"   Split-plot: >= {r} блоків WP")
                    lines.append("")
                    lines.append(f"{'r':>4}  {'Потужність':>12}  Статус")
                    lines.append("-" * 32)
                    for rr in range(max(2,r-2), min(r+5, 101)):
                        lnc2 = k*rr*(delta**2)/(2*sigma**2)
                        try:
                            fc2 = float(f_dist.ppf(1-alpha, k-1, k*(rr-1)))
                            pw2 = float(1 - ncf.cdf(fc2, k-1, k*(rr-1), lnc2))
                        except Exception: continue
                        mark = " <-- мінімум" if rr == r else ""
                        lines.append(f"{rr:>4}  {pw2*100:>10.1f}%{mark}")
                    found = True; break

            if not found:
                lines.append("Не вдалося знайти r <= 100.")
                lines.append("Спробуйте: збільшити delta або зменшити sigma.")

        self._set_result("\n".join(lines))



# ═══════════════════════════════════════════════════════════════
# CLUSTER ANALYSIS
# ═══════════════════════════════════════════════════════════════
