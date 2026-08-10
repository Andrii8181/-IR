# sad_stability_ancova.py — Стабільність, ANCOVA
# -*- coding: utf-8 -*-
from sad_common import *

class StabilityWindow:
    """Аналіз стабільності генотипів (GxE взаємодія)."""

    HELP_TEXT = """
АНАЛІЗ СТАБІЛЬНОСТІ (GxE) — ПОКРОКОВА ІНСТРУКЦІЯ
══════════════════════════════════════════════════

ЩО ТАКЕ АНАЛІЗ СТАБІЛЬНОСТІ?
  Оцінює як стабільно генотипи (сорти) поводяться
  в різних середовищах (роки, локації, умови).

  Два ключових питання:
  1. Який генотип найбільш продуктивний загалом?
  2. Який генотип найбільш СТАБІЛЬНИЙ (мало залежить від умов)?

  Генотип може бути:
  • Стабільним і продуктивним → ідеальний для широкого впровадження
  • Стабільним але низькопродуктивним → стабільний аутсайдер
  • Нестабільним і продуктивним → хороший лише у сприятливих умовах

КРОК 1. СТРУКТУРА ТАБЛИЦІ

  Перший стовпець: Назва генотипу/сорту (текст)
  Решта стовпців: Значення показника у кожному середовищі

  Значення = СЕРЕДНЄ по повторностях для цього генотипу у цьому середовищі.

  Приклад (4 сорти, 4 роки):
  | Генотип  | 2021 | 2022 | 2023 | 2024 |
  | Сорт А   | 5.8  |  6.2 |  5.5 |  6.8 |
  | Сорт Б   | 4.9  |  7.1 |  4.2 |  7.8 |
  | Контроль | 5.2  |  5.4 |  5.1 |  5.6 |

  Перейменуйте заголовки середовищ (подвійний клік).
  Мінімум: 2 генотипи, 2 середовища.

КРОК 2. МЕТОД EBERHART-RUSSELL

  Рівняння: Yij = μi + bi·Ij + δij
  де:
    Yij — врожай сорту i в середовищі j
    μi  — середнє сорту по всіх середовищах
    bi  — коефіцієнт регресії (відгук на умови)
    Ij  — індекс середовища
    δij — відхилення від регресії

  Параметри стабільності:

  bi (коефіцієнт регресії):
    bi = 1.0 → сорт реагує як середній по популяції
    bi > 1.0 → адаптивний/чутливий (краще у сприятливих, гірше у несприятливих)
    bi < 1.0 → консервативний/стабільний (слабко реагує на умови)

  s²d (дисперсія відхилень від регресії):
    s²d ≈ 0 → точна лінійна відповідь, передбачуваний сорт ✓
    s²d > 0 → непередбачувана реакція ✗

  КЛАСИ СТАБІЛЬНОСТІ:
    Стабільний:      bi ≈ 1, s²d ≈ 0  → рекомендований для всіх зон
    Адаптивний:      bi > 1, s²d ≈ 0  → лише для сприятливих умов
    Консервативний:  bi < 1, s²d ≈ 0  → для несприятливих умов
    Нестабільний:    будь-який bi, s²d > 0  → непередбачуваний

КРОК 3. GGE BIPLOT

  GGE = Genotype + Genotype×Environment interaction.
  Двовимірний графік що показує одночасно:
  • Продуктивність генотипів (відстань від центру)
  • Стабільність (чим ближче до кола — тим стабільніший)
  • Адаптацію до конкретних середовищ

  Стрілки = середовища.
  Точки = генотипи.
  Генотип близько до стрілки середовища → добре адаптований до нього.
"""

    def __init__(self, parent, gs):
        self.win = tk.Toplevel(parent)
        self.win.title("Аналіз стабільності (GxE)")
        self.win.geometry("1020x680"); set_icon(self.win)
        self.gs = gs
        self._stab_fig = None
        self._st_gs = {
            "font_family": "Times New Roman", "font_size": 9,
            "point_color": "#4c72b0", "vector_color": "#c62828",
            "point_size": 80,
        }
        self._build()

    def _build(self):
        # ── Toolbar ──────────────────────────────────────────
        top = tk.Frame(self.win, padx=8, pady=6); top.pack(fill=tk.X)
        tk.Button(top, text="▶ Аналіз", bg="#c62828", fg="white",
                  font=("Times New Roman",13),
                  command=self._run).pack(side=tk.LEFT, padx=4)

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
                 text="Подвійний клік на заголовку → перейменувати середовище",
                 font=("Times New Roman",9), fg="#666").pack(side=tk.LEFT, padx=8)

        # ── Таблиця ─────────────────────────────────────────
        mid = tk.Frame(self.win); mid.pack(fill=tk.BOTH, expand=True, padx=8)
        self.rows_n = 16; self.cols_n = 8
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

        # Перший заголовок — «Генотип» (фіксований)
        tk.Label(self.inner, text="Генотип", relief=tk.RIDGE, width=14,
                 bg="#444444", fg="white",
                 font=("Times New Roman",11,"bold")
                 ).grid(row=0, column=0, padx=1, pady=1, sticky="nsew")

        # Заголовки середовищ (перейменовувані)
        self.env_vars = []; self.env_labels = []
        for j in range(1, self.cols_n):
            var = tk.StringVar(value=f"E{j}")
            self.env_vars.append(var)
            lbl = tk.Label(self.inner, textvariable=var, width=12, cursor="hand2",
                           bg="#1a4b8c", fg="white", relief=tk.RIDGE,
                           font=("Times New Roman",11,"bold"))
            lbl.grid(row=0, column=j, padx=1, pady=1, sticky="nsew")
            lbl.bind("<Double-Button-1>", lambda e, idx=j-1: self._rename_env(idx))
            self.env_labels.append(lbl)

        self.entries = []
        for i in range(self.rows_n):
            row_ = []
            for j in range(self.cols_n):
                e = tk.Entry(self.inner, width=14 if j==0 else 11,
                             font=("Times New Roman",11))
                e.grid(row=i+1, column=j, padx=1, pady=1)
                if j == 0:
                    e.bind("<KeyRelease>", lambda ev: _autofit_col(self.entries, 0))
                row_.append(e)
            self.entries.append(row_)
        _bind_nav(self.entries, self.win)
        _bind_fill_handle(self.entries, self.win)

    # ── Перейменування середовища ─────────────────────────────
    def _rename_env(self, idx):
        dlg = tk.Toplevel(self.win); dlg.title("Перейменувати середовище")
        dlg.resizable(False, False); dlg.grab_set()
        tk.Label(dlg, text=f"Назва середовища {idx+1}:",
                 font=("Times New Roman",12)).pack(padx=16, pady=14)
        var = tk.StringVar(value=self.env_vars[idx].get())
        e = tk.Entry(dlg, textvariable=var, font=("Times New Roman",12), width=24)
        e.pack(padx=16, pady=4); e.select_range(0, tk.END); e.focus_set()
        def apply():
            nm = var.get().strip()
            if nm: self.env_vars[idx].set(nm)
            dlg.destroy()
        tk.Button(dlg, text="OK", bg="#c62828", fg="white",
                  font=("Times New Roman",12), command=apply).pack(pady=(4,14))
        dlg.bind("<Return>", lambda ev: apply()); center_win(dlg)

    # ── Управління таблицею ───────────────────────────────────
    def _add_row(self):
        i = self.rows_n; row_ = []
        for j in range(self.cols_n):
            e = tk.Entry(self.inner, width=14 if j==0 else 11,
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
        var = tk.StringVar(value=f"E{ci}")
        self.env_vars.append(var)
        lbl = tk.Label(self.inner, textvariable=var, width=12, cursor="hand2",
                       bg="#1a4b8c", fg="white", relief=tk.RIDGE,
                       font=("Times New Roman",11,"bold"))
        lbl.grid(row=0, column=ci, padx=1, pady=1, sticky="nsew")
        lbl.bind("<Double-Button-1>", lambda e, idx=ci-1: self._rename_env(idx))
        self.env_labels.append(lbl)
        for i, row_ in enumerate(self.entries):
            e = tk.Entry(self.inner, width=11, font=("Times New Roman",11))
            e.grid(row=i+1, column=ci, padx=1, pady=1)
            row_.append(e)
        _bind_nav(self.entries, self.win)
        _bind_fill_handle(self.entries, self.win)

    def _del_col(self):
        if self.cols_n <= 3: return
        self.env_labels.pop().destroy(); self.env_vars.pop()
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
                    if e is w: pos=(i,j); break
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
            filetypes=[("SAD проект","*.sadp"),("JSON","*.json")])
        if not path: return
        d = {"type":"stability","version":APP_VER,
             "env_vars":[v.get() for v in self.env_vars],
             "rows_data":[[e.get() for e in row] for row in self.entries]}
        try:
            with open(path,"w",encoding="utf-8") as f: json.dump(d,f,ensure_ascii=False,indent=2)
            messagebox.showinfo("Збережено", path)
        except Exception as ex: messagebox.showerror("Помилка",str(ex))

    def _load_proj(self):
        path = filedialog.askopenfilename(
            parent=self.win, filetypes=[("SAD проект","*.sadp"),("JSON","*.json")])
        if not path: return
        try:
            with open(path,"r",encoding="utf-8") as f: d=json.load(f)
        except Exception as ex: messagebox.showerror("Помилка",str(ex)); return
        env_vars = d.get("env_vars",[])
        rows_data = d.get("rows_data",[])
        while self.cols_n < 1+len(env_vars): self._add_col()
        for i,nm in enumerate(env_vars):
            if i<len(self.env_vars): self.env_vars[i].set(nm)
        while len(self.entries)<len(rows_data): self._add_row()
        for i,rv in enumerate(rows_data):
            for j,v in enumerate(rv):
                if j<self.cols_n:
                    self.entries[i][j].delete(0,tk.END); self.entries[i][j].insert(0,v)
        messagebox.showinfo("Завантажено","Проект завантажено.")

    def _show_help(self):
        win = tk.Toplevel(self.win); win.title("Довідка — Аналіз стабільності")
        win.geometry("720x680"); set_icon(win)
        frm = tk.Frame(win); frm.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        vsb = ttk.Scrollbar(frm, orient="vertical"); vsb.pack(side=tk.RIGHT, fill=tk.Y)
        txt = tk.Text(frm, wrap="word", font=("Times New Roman",11),
                      yscrollcommand=vsb.set, relief=tk.FLAT,
                      bg="#fafafa", padx=10, pady=8, cursor="arrow")
        txt.pack(fill=tk.BOTH, expand=True); vsb.config(command=txt.yview)
        txt.insert("1.0", self.HELP_TEXT.strip()); txt.configure(state="disabled")
        txt.bind("<MouseWheel>", lambda e: txt.yview_scroll(int(-1*(e.delta/120)),"units"))
        tk.Button(win, text="Закрити", command=win.destroy,
                  font=("Times New Roman",11)).pack(pady=6)



    def _run(self):
        env_names = [v.get().strip() or f"E{i+1}" for i, v in enumerate(self.env_vars)]
        raw = [[e.get().strip() for e in row] for row in self.entries]
        gen_names = []; matrix = []
        for row in raw:
            nm = row[0] if row[0] else f"G{len(gen_names)+1}"
            vals = []
            for v in row[1:len(env_names)+1]:
                if not v: vals.append(np.nan)
                else:
                    try: vals.append(float(v.replace(",",".")))
                    except Exception: vals.append(np.nan)
            if any(not math.isnan(v) for v in vals):
                gen_names.append(nm); matrix.append(vals)

        if len(matrix) < 2: messagebox.showwarning("Замало генотипів","Потрібно ≥ 2 генотипи."); return
        e_count = len(env_names); g_count = len(gen_names)
        data = np.array(matrix, dtype=float)

        # ── Eberhart–Russell regression stability ──────────────
        env_means = np.nanmean(data, axis=0)
        grand_mean = np.nanmean(data)
        env_index = env_means - grand_mean  # environment index

        er_rows = []
        for i in range(g_count):
            y = data[i]; mask = ~np.isnan(y) & ~np.isnan(env_index)
            if np.sum(mask) < 2:
                er_rows.append([gen_names[i], fmt(np.nanmean(y),3), "–", "–", "–"]); continue
            xi = env_index[mask]; yi = y[mask]
            # linear regression of genotype yield on env index
            X_ = np.column_stack([np.ones(len(xi)), xi])
            beta, *_ = np.linalg.lstsq(X_, yi, rcond=None)
            b_i = float(beta[1])  # regression coefficient (stability)
            yhat = X_ @ beta
            ss_dev = float(np.sum((yi - yhat)**2))
            s2d = ss_dev / max(len(yi)-2, 1)  # variance of deviations
            gen_mean = float(np.nanmean(y))
            er_rows.append([gen_names[i], fmt(gen_mean,3), fmt(b_i,4), fmt(s2d,4),
                            "Стабільний (bi≈1, s²d≈0)" if abs(b_i-1)<0.2 and s2d<0.1 else
                            "Адаптивний" if b_i > 1.2 else "Консервативний" if b_i < 0.8 else "Середній"])

        # ── GGE Biplot via SVD ──────────────────────────────────
        # center by environment means
        data_c = data - env_means[np.newaxis, :]
        # replace NaN with 0 for SVD
        data_c = np.nan_to_num(data_c)
        U, S, Vt = np.linalg.svd(data_c, full_matrices=False)
        pc1_g = U[:,0] * S[0]; pc2_g = U[:,1] * S[1] if len(S)>1 else np.zeros(g_count)
        pc1_e = Vt[0,:];       pc2_e = Vt[1,:] if len(S)>1 else np.zeros(e_count)
        var_exp = S**2 / np.sum(S**2) * 100

        if not HAS_MPL:
            messagebox.showwarning("", "Для побудови графіків потрібен matplotlib."); return

        win = tk.Toplevel(self.win)
        win.title("Аналіз стабільності — Результати")
        n_gen = len(gen_names)
        est_h = min(880, max(680, 560 + 14*n_gen))
        win.geometry(f"1150x{est_h}"); set_icon(win)

        self._st_data = dict(gen_names=gen_names, env_names=env_names, er_rows=er_rows,
                             pc1_g=pc1_g, pc2_g=pc2_g, pc1_e=pc1_e, pc2_e=pc2_e,
                             var_exp=var_exp)
        self._st_built = {"biplot": False}

        main = tk.Frame(win); main.pack(fill=tk.BOTH, expand=True)
        sidebar = tk.Frame(main, width=210, bg="#2c3e50")
        sidebar.pack(side=tk.LEFT, fill=tk.Y); sidebar.pack_propagate(False)
        content = tk.Frame(main); content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(sidebar, text="СТАБІЛЬНІСТЬ\n(GxE)", bg="#2c3e50", fg="#ecf0f1",
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

        rpt_frame = tk.Frame(content)
        bp_frame  = tk.Frame(content)

        b_rpt = _sidebar_btn("📋 Звіт (таблиця)",  "Bi, s²d, клас стабільності")
        b_bp  = _sidebar_btn("📊 GGE Biplot",       "Генотип × Середовище")

        def _open_rpt(): _show_panel(rpt_frame, b_rpt)
        def _open_bp():
            _show_panel(bp_frame, b_bp)
            if not self._st_built["biplot"]:
                bp_frame.update_idletasks()
                self._build_stability_biplot_panel(bp_frame)
                self._st_built["biplot"] = True

        b_rpt.configure(command=_open_rpt)
        b_bp.configure( command=_open_bp)

        self._build_stability_report_panel(rpt_frame, win, gen_names, env_names, er_rows)

        _show_panel(rpt_frame, b_rpt)

    def _build_stability_report_panel(self, frame, win, gen_names, env_names, er_rows):
        tb = tk.Frame(frame, padx=6, pady=5); tb.pack(fill=tk.X)
        tk.Button(tb, text="📋 Копіювати таблицю", font=("Times New Roman",11),
                  command=lambda: self._copy_stability_table(win, er_rows)
                  ).pack(side=tk.LEFT, padx=4)

        tk.Label(frame,
                 text=f"Генотипів: {len(gen_names)}   |   Середовищ: {len(env_names)}",
                 font=("Times New Roman",11), fg="#555", anchor="w"
                 ).pack(fill=tk.X, padx=10, pady=(4,2))

        frm, _ = make_tv(frame, ["Генотип","Середнє","bi","s²d","Клас стабільності"], er_rows)
        frm.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        tk.Label(frame,
                 text="bi (регресійний коефіцієнт Ебергарта–Рассела): bi≈1 — середня "
                      "чутливість до умов середовища; bi>1 — придатний для сприятливих "
                      "умов (адаптивний); bi<1 — стабільніший у несприятливих умовах "
                      "(консервативний).\n"
                      "s²d (варіанса відхилень від лінії регресії): чим ближче до 0 — "
                      "тим передбачуваніша поведінка генотипу за цією моделлю.",
                 font=("Times New Roman",10), fg="#555", justify="left",
                 wraplength=1000, anchor="w").pack(fill=tk.X, padx=10, pady=(2,10))

    def _copy_stability_table(self, win, er_rows):
        lines = ["Генотип\tСереднє\tbi\ts²d\tКлас стабільності"]
        for r in er_rows:
            lines.append("\t".join(str(x) for x in r))
        win.clipboard_clear(); win.clipboard_append("\n".join(lines))
        messagebox.showinfo("Скопійовано",
            "Таблицю скопійовано у буфер обміну.\nВставте у Word/Excel через Ctrl+V.")

    def _build_stability_biplot_panel(self, frame):
        for w in frame.winfo_children(): w.destroy()
        d = self._st_data; gs = self._st_gs
        gen_names = d["gen_names"]; env_names = d["env_names"]
        pc1_g = d["pc1_g"]; pc2_g = d["pc2_g"]
        pc1_e = d["pc1_e"]; pc2_e = d["pc2_e"]; var_exp = d["var_exp"]

        tb = tk.Frame(frame, padx=6, pady=5); tb.pack(fill=tk.X)
        tk.Button(tb, text="💾 Зберегти PNG", font=("Times New Roman",11),
                  command=self._save_stability_png).pack(side=tk.LEFT, padx=4)
        tk.Button(tb, text="📋 Копіювати", font=("Times New Roman",11),
                  command=self._copy_stability_fig).pack(side=tk.LEFT, padx=4)
        tk.Button(tb, text="⚙ Налаштування", font=("Times New Roman",11),
                  command=lambda: self._restyle_stability(frame)
                  ).pack(side=tk.LEFT, padx=4)

        plot_f = tk.Frame(frame); plot_f.pack(fill=tk.BOTH, expand=True)
        self._stab_plot_frame = plot_f

        ff = gs["font_family"]; fz = gs["font_size"]
        pc_ = gs["point_color"]; vc = gs["vector_color"]; ps = gs["point_size"]

        fig = Figure(figsize=(9, 7), dpi=100)
        ax1 = fig.add_subplot(111)
        ax1.axhline(0, color="k", lw=0.5); ax1.axvline(0, color="k", lw=0.5)
        ax1.scatter(pc1_g, pc2_g, s=ps, color=pc_, zorder=3)
        for i, nm in enumerate(gen_names):
            ann = ax1.annotate(nm, xy=(pc1_g[i], pc2_g[i]),
                        xytext=(7, 7), textcoords="offset points",
                        fontsize=fz-1, fontfamily=ff,
                        arrowprops=dict(arrowstyle="-", color="#999",
                                       alpha=0.6, lw=0.6, shrinkA=0, shrinkB=3))
            ann.draggable(True)
        sc = max(np.max(np.abs(pc1_g)), np.max(np.abs(pc2_g)), 1e-10)
        sc_e = sc / max(np.max(np.abs(pc1_e)), max(np.max(np.abs(pc2_e)),1e-10))
        for j, nm in enumerate(env_names):
            ax1.annotate("", xy=(pc1_e[j]*sc_e*0.8, pc2_e[j]*sc_e*0.8), xytext=(0,0),
                         arrowprops=dict(arrowstyle="->", color=vc, lw=1.2))
            ann_e = ax1.annotate(nm, xy=(pc1_e[j]*sc_e*0.8, pc2_e[j]*sc_e*0.8),
                        xytext=(9, 9), textcoords="offset points",
                        fontsize=fz-1, color=vc, fontfamily=ff,
                        arrowprops=dict(arrowstyle="-", color=vc,
                                       alpha=0.5, lw=0.6, shrinkA=0, shrinkB=3))
            ann_e.draggable(True)
        ax1.set_xlabel(f"ГК1 ({fmt(var_exp[0],1)}%)", fontsize=fz, fontfamily=ff)
        ax1.set_ylabel(f"ГК2 ({fmt(var_exp[1] if len(var_exp)>1 else 0,1)}%)",
                       fontsize=fz, fontfamily=ff)
        ax1.set_title("GGE Biplot (Генотип × Середовище)", fontsize=fz+1, fontfamily=ff)
        ax1.tick_params(labelsize=fz)
        ax1.yaxis.grid(True, alpha=0.25)
        fig.tight_layout()
        self._stab_fig = fig
        embed_figure(fig, plot_f)

    def _restyle_stability(self, frame):
        gs = self._st_gs
        dlg = tk.Toplevel(self.win); dlg.title("Налаштування графіка")
        dlg.resizable(False, False); dlg.grab_set(); set_icon(dlg)
        rf = ("Times New Roman",11)
        frm = tk.Frame(dlg, padx=16, pady=14); frm.pack()

        tk.Label(frm, text="Шрифт:", font=rf).grid(row=0, column=0, sticky="w", pady=4)
        ff_v = tk.StringVar(value=gs["font_family"])
        ttk.Combobox(frm, textvariable=ff_v, state="readonly", width=20,
                     values=["Times New Roman","Arial","Calibri","Georgia"]
                     ).grid(row=0, column=1, sticky="w", padx=8)

        tk.Label(frm, text="Розмір шрифту:", font=rf).grid(row=1, column=0, sticky="w", pady=4)
        fz_v = tk.IntVar(value=gs["font_size"])
        tk.Spinbox(frm, from_=6, to=18, textvariable=fz_v, width=6, font=rf
                   ).grid(row=1, column=1, sticky="w", padx=8)

        tk.Label(frm, text="Розмір точок:", font=rf).grid(row=2, column=0, sticky="w", pady=4)
        ps_v = tk.IntVar(value=gs["point_size"])
        tk.Spinbox(frm, from_=20, to=200, increment=10, textvariable=ps_v, width=6, font=rf
                   ).grid(row=2, column=1, sticky="w", padx=8)

        pt_col_v = tk.StringVar(value=gs["point_color"])
        vec_col_v = tk.StringVar(value=gs["vector_color"])
        def _pick(var):
            c = colorchooser.askcolor(color=var.get(), parent=dlg)
            if c and c[1]: var.set(c[1])
        tk.Label(frm, text="Колір генотипів (точок):", font=rf).grid(
            row=3, column=0, sticky="w", pady=4)
        tk.Button(frm, text="Обрати колір", command=lambda: _pick(pt_col_v)
                  ).grid(row=3, column=1, sticky="w", padx=8)
        tk.Label(frm, text="Колір середовищ (векторів):", font=rf).grid(
            row=4, column=0, sticky="w", pady=4)
        tk.Button(frm, text="Обрати колір", command=lambda: _pick(vec_col_v)
                  ).grid(row=4, column=1, sticky="w", padx=8)

        def apply():
            self._st_gs.update({
                "font_family": ff_v.get(), "font_size": fz_v.get(),
                "point_size": ps_v.get(),
                "point_color": pt_col_v.get(), "vector_color": vec_col_v.get(),
            })
            dlg.destroy()
            self._build_stability_biplot_panel(frame)

        bf = tk.Frame(frm); bf.grid(row=5, column=0, columnspan=2, pady=(14,0))
        tk.Button(bf, text="OK", bg="#c62828", fg="white", font=rf,
                  command=apply).pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text="Скасувати", font=rf, command=dlg.destroy).pack(side=tk.LEFT)
        center_win(dlg)

    def _save_stability_png(self):
        if self._stab_fig is None:
            messagebox.showwarning("","Спочатку виконайте аналіз."); return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                    filetypes=[("PNG зображення","*.png")], title="Зберегти графік")
        if not path: return
        try:
            self._stab_fig.savefig(path, dpi=150, bbox_inches="tight")
            messagebox.showinfo("Збережено", f"Збережено:\n{path}")
        except Exception as ex:
            messagebox.showerror("Помилка", str(ex))

    def _copy_stability_fig(self):
        if self._stab_fig is None:
            messagebox.showwarning("","Спочатку виконайте аналіз."); return
        ok, msg = _copy_fig_to_clipboard(self._stab_fig)
        if ok: messagebox.showinfo("","Графік скопійовано.\nВставте у Word через Ctrl+V.")
        else:   messagebox.showwarning("",f"Помилка: {msg}")



# ═══════════════════════════════════════════════════════════════
# ANCOVA — Analysis of Covariance
# ═══════════════════════════════════════════════════════════════
class AncovaWindow:
    """ANCOVA — Коваріаційний аналіз."""

    HELP_TEXT = """
ANCOVA — ПОКРОКОВА ІНСТРУКЦІЯ
══════════════════════════════════════════

ЩО ТАКЕ ANCOVA?
  ANCOVA (Коваріаційний аналіз) = ANOVA + контроль неперервної змінної.
  Порівнює групи за залежною змінною (Y), виключаючи вплив
  однієї або кількох коваріат (змінних що ви вимірюєте але не контролюєте).

КОЛИ ВИКОРИСТОВУВАТИ?
  Коли між групами є відмінності у вихідних умовах:
  • Порівняння врожайності сортів, але pH ґрунту різний на ділянках
  • Порівняння приросту, але початкова маса рослин різна
  • Вплив обробки, але температура чи вологість відрізнялась

КРОК 1. СТРУКТУРА ТАБЛИЦІ ДАНИХ

  Стовпці мають бути у такому порядку:
  [Група] | [Коваріата 1] | [Коваріата 2] | ... | [Залежна Y]

  Перейменуйте заголовки (сині/блакитні клітинки зверху):
    Перший стовпець = Назва групи/фактора (текстові мітки!)
    Останній стовпець = Залежна змінна Y (числа)
    Між ними = Коваріати (числа)

  Приклад (порівняння сортів, коваріата = pH):
  | Сорт     | pH ґрунту | Врожайність |
  | Сорт А   |    6.2    |    5.8      |
  | Сорт А   |    5.9    |    5.4      |
  | Сорт Б   |    6.5    |    6.2      |

  Мінімум: 6 спостережень, 2 групи, 2 спостереження в кожній групі.

КРОК 2. РІВЕНЬ ЗНАЧУЩОСТІ α
  Стандарт: 0.05.
  Строже: 0.01 (при множинних порівняннях).

КРОК 3. ВИКОНАННЯ АНАЛІЗУ
  Натисніть «▶ Виконати» та дочекайтесь результатів.

КРОК 4. АВТОМАТИЧНІ ПЕРЕВІРКИ ПЕРЕДУМОВ

  Програма автоматично перевіряє і БЛОКУЄ аналіз при порушеннях:

  ① Паралельність ліній регресії (КЛЮЧОВА ПЕРЕДУМОВА):
    ANCOVA передбачає що вплив коваріати на Y ОДНАКОВИЙ у всіх групах.
    Тест: взаємодія Група×Коваріата.
    p ≥ 0.05 → лінії паралельні → ANCOVA коректна ✓
    p < 0.05 → лінії НЕ паралельні → ANCOVA ЗАБЛОКОВАНА ✗
    (→ використайте звичайну ANOVA з коваріатою як фактором)

  ② Нормальність залишків (Shapiro-Wilk):
    p > 0.05 → залишки нормальні ✓
    p ≤ 0.05 → програма запитає підтвердження

  ③ Однорідність дисперсій (тест Левена):
    p ≥ 0.05 → дисперсії рівні ✓
    p < 0.05 → програма запитає підтвердження

  ④ Мультиколінеарність коваріат (r > 0.95):
    При дуже сильному зв'язку між коваріатами — попередження

КРОК 5. ІНТЕРПРЕТАЦІЯ РЕЗУЛЬТАТІВ

  Таблиця ANCOVA (Тип III SS):
    Джерело «Група» → p < 0.05: групи відрізняються після контролю коваріати
    Джерело «Коваріата» → p < 0.05: коваріата суттєво впливає на Y
    R² → частка варіації Y пояснена всією моделлю

  Скориговані середні (LS Means):
    Це ГОЛОВНИЙ результат ANCOVA!
    Прогнозоване середнє кожної групи за умови що всі групи
    мають ОДНАКОВЕ значення коваріати (= загальне середнє).
    Порівнюйте СКОРИГОВАНІ, а не нескориговані середні!

  Пост-хок (Бонферроні):
    p < 0.05 → пара груп значуще відрізняється за скоригованим середнім

КРОК 6. ГРАФІКИ ЗАЛИШКІВ
  Залишки vs Підігнані: точки мають бути хаотично навколо нуля
  QQ-графік залишків: точки мають лежати на прямій
  ⚠ Патерн або вигин → модель порушена
"""

    def __init__(self, parent, gs):
        self.win = tk.Toplevel(parent)
        self.win.title("ANCOVA — Коваріаційний аналіз")
        self.win.geometry("980x680"); set_icon(self.win)
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
        sm.add_command(label="➕ Додати коваріату",   command=self._add_covariate)
        sm.add_command(label="➖ Видалити коваріату", command=self._del_covariate)
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
            "Порядок стовпців:  [Група/Фактор]  [Коваріата 1]  [Коваріата 2 ...]  [Залежна Y]\n"
            "Заголовки стовпців (блакитні) можна редагувати.  "
            "Перший стовпець — текстові мітки груп.  Решта — числа."),
            font=("Times New Roman", 10), bg="#f0f4ff", justify="left").pack(anchor="w")

        # ── Таблиця даних ────────────────────────────────────
        mid = tk.Frame(self.win); mid.pack(fill=tk.BOTH, expand=True, padx=8)
        self.n_rows = 24; self.n_cols = 6
        canvas = tk.Canvas(mid); canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(mid, orient="vertical", command=canvas.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.configure(yscrollcommand=sb.set)
        self.inner = tk.Frame(canvas)
        canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: canvas.config(scrollregion=canvas.bbox("all")))
        self.win.bind("<MouseWheel>",
                      lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        col_hints = ["Група", "Коваріата 1", "Коваріата 2", "Коваріата 3", "Коваріата 4", "Залежна Y"]
        self.header_entries = []
        for j in range(self.n_cols):
            e = tk.Entry(self.inner, width=14, bg="#1a4b8c", fg="white",
                         font=("Times New Roman", 11, "bold"),
                         insertbackground="white")
            e.insert(0, col_hints[j] if j < len(col_hints) else f"Стовп{j+1}")
            e.grid(row=0, column=j, padx=1, pady=1)
            self.header_entries.append(e)
        self.entries = []
        for i in range(self.n_rows):
            row_ = []
            for j in range(self.n_cols):
                e = tk.Entry(self.inner, width=14, font=("Times New Roman", 11),
                             highlightthickness=1, highlightbackground="#c0c0c0")
                e.grid(row=i+1, column=j, padx=1, pady=1)
                if j == 0:
                    e.bind("<KeyRelease>",
                           lambda ev: _autofit_col(self.entries, 0, self.header_entries))
                row_.append(e)
            self.entries.append(row_)
        _bind_nav(self.entries, self.win)
        _bind_fill_handle(self.entries, self.win)

    def _show_help(self):
        win = tk.Toplevel(self.win)
        win.title("Довідка — ANCOVA")
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
        txt.bind("<MouseWheel>",
                 lambda e: txt.yview_scroll(int(-1*(e.delta/120)), "units"))
        tk.Button(win, text="Закрити", command=win.destroy,
                  font=("Times New Roman", 11)).pack(pady=6)

    def _help(self):
        self._show_help()   # залишаємо для сумісності

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
        rows = [r for r in data.splitlines() if r.strip()]
        for ir, line in enumerate(rows):
            i = r0 + ir
            while i >= len(self.entries): self._add_row()
            for jc, val in enumerate(line.split("\t")):
                j = c0 + jc
                if j >= self.n_cols: continue
                self.entries[i][j].delete(0, tk.END)
                self.entries[i][j].insert(0, val.strip())
        _autofit_col(self.entries, 0, self.header_entries)

    def _add_row(self):
        i = self.n_rows; row_ = []
        for j in range(self.n_cols):
            e = tk.Entry(self.inner, width=14, font=("Times New Roman", 11),
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

    def _regrid_columns(self):
        for j, e in enumerate(self.header_entries):
            e.grid(row=0, column=j, padx=1, pady=1)
        for i, row_ in enumerate(self.entries):
            for j, e in enumerate(row_):
                e.grid(row=i+1, column=j, padx=1, pady=1)

    def _add_covariate(self):
        # Нова коваріата вставляється ПЕРЕД останнім стовпцем (Залежна Y)
        j_new = self.n_cols - 1
        cov_num = self.n_cols - 1  # Група + (n_cols-2) наявних коваріат → номер нової
        e = tk.Entry(self.inner, width=14, bg="#1a4b8c", fg="white",
                     font=("Times New Roman", 11, "bold"), insertbackground="white")
        e.insert(0, f"Коваріата {cov_num}")
        self.header_entries.insert(j_new, e)
        for row_ in self.entries:
            ne = tk.Entry(self.inner, width=14, font=("Times New Roman", 11),
                          highlightthickness=1, highlightbackground="#c0c0c0")
            row_.insert(j_new, ne)
        self.n_cols += 1
        self._regrid_columns()
        _bind_nav(self.entries, self.win)
        _bind_fill_handle(self.entries, self.win)

    def _del_covariate(self):
        n_covariates = self.n_cols - 2  # без стовпця "Група" і стовпця "Залежна Y"
        if n_covariates <= 1:
            messagebox.showwarning("Неможливо видалити",
                "ANCOVA потребує щонайменше одну коваріату."); return
        j_del = self.n_cols - 2  # остання коваріата (одразу перед Y)
        self.header_entries[j_del].destroy(); del self.header_entries[j_del]
        for row_ in self.entries:
            row_[j_del].destroy(); del row_[j_del]
        self.n_cols -= 1
        self._regrid_columns()

    def _clear_table(self):
        if not messagebox.askyesno("Очистити таблицю",
                "Видалити всі числові дані?\n(Заголовки залишаться)"):
            return
        for row in self.entries:
            for e in row: e.delete(0, tk.END)

    def _save_proj(self):
        generic_save_project(self.win, "ancova", self.header_entries, self.entries)

    def _load_proj(self):
        d = generic_load_project(self.win)
        if d is None: return
        headers = d.get("headers", []); rd = d.get("rows_data", [])
        while self.n_cols < len(headers): self._add_covariate()
        for j, h in enumerate(headers):
            if j < len(self.header_entries): _set_header_text(self.header_entries[j], h)
        while len(self.entries) < len(rd): self._add_row()
        for i, rv in enumerate(rd):
            for j, v in enumerate(rv):
                if i < len(self.entries) and j < len(self.entries[i]):
                    self.entries[i][j].delete(0, tk.END); self.entries[i][j].insert(0, v)



    def _run(self, transform=None):
        alpha = float(self.alpha_var.get())

        # ── Зчитування заголовків ─────────────────────────────
        headers = [e.get().strip() or f"Col{j+1}"
                   for j, e in enumerate(self.header_entries)]

        # ── Зчитування даних ─────────────────────────────────
        raw = [[e.get().strip() for e in row] for row in self.entries]
        raw = [r for r in raw if any(v for v in r)]
        if not raw:
            messagebox.showwarning("Немає даних",
                "Будь ласка, введіть дані у таблицю."); return

        # Визначаємо які стовпці реально заповнені числами
        # Перший стовпець = Група (текст)
        # Останній заповнений числовий стовпець = Залежна Y
        # Між ними = Коваріати (лише заповнені стовпці)

        # Знаходимо останній стовпець з числовими даними
        n_data_cols = self.n_cols
        # Знаходимо реально заповнені числові стовпці (1..n_cols-1)
        filled_numeric_cols = []
        for j in range(1, self.n_cols):
            has_num = False
            for row in raw:
                v = row[j] if j < len(row) else ""
                if v:
                    try: float(v.replace(",",".")); has_num = True; break
                    except ValueError: pass
            if has_num:
                filled_numeric_cols.append(j)

        if not filled_numeric_cols:
            messagebox.showwarning("Немає числових даних",
                "Не знайдено числових даних у стовпцях 2 і далі.\n"
                "Переконайтесь що введено значення коваріати та залежної змінної."); return

        # Останній заповнений числовий стовпець = залежна Y
        dv_col_idx  = filled_numeric_cols[-1]
        # Всі інші заповнені числові стовпці = коваріати
        cov_col_idxs = filled_numeric_cols[:-1]

        group_col = headers[0]
        dv_col    = headers[dv_col_idx]
        cov_cols  = [headers[j] for j in cov_col_idxs]

        if not cov_col_idxs:
            messagebox.showwarning("Немає коваріат",
                "ANCOVA потребує хоча б одну коваріату.\n\n"
                "Структура таблиці:\n"
                "  Стовпець 1: Група (текстові мітки)\n"
                "  Стовпець 2: Коваріата (числові значення)\n"
                "  Стовпець 3 або далі: Залежна змінна Y\n\n"
                "Якщо у вас одна коваріата — заповніть стовпці 1, 2 і 3.\n"
                "Стовпці 4, 5, 6 залиште порожніми."); return

        # Зчитуємо рядки використовуючи лише знайдені стовпці
        groups = []; cov_data = [[] for _ in cov_col_idxs]; y_data = []
        skipped = 0
        for row in raw:
            while len(row) < self.n_cols: row.append("")
            grp = row[0].strip()
            if not grp: skipped += 1; continue
            # Перевіряємо чи грp не є числом (захист від плутанини)
            try:
                float(grp.replace(",","."))
                skipped += 1; continue  # перший стовпець має бути текстом
            except ValueError:
                pass
            try:
                covs = [float(row[j].replace(",",".")) for j in cov_col_idxs]
                yval = float(row[dv_col_idx].replace(",","."))
            except (ValueError, IndexError):
                skipped += 1; continue
            groups.append(grp)
            for j_idx, cv in enumerate(covs): cov_data[j_idx].append(cv)
            y_data.append(yval)

        if skipped > 0:
            messagebox.showinfo("Пропущені рядки",
                f"Пропущено {skipped} рядків (порожні або нечислові значення).")

        if transform == "log":
            if any(v <= 0 for v in y_data):
                messagebox.showwarning("Трансформація неможлива",
                    "У залежній змінній Y є значення ≤ 0 — логарифмічна трансформація "
                    "неможлива. Аналіз продовжено без трансформації.")
                transform = None
            else:
                y_data = [math.log(v) for v in y_data]

        n = len(y_data)
        # ── Guard 1: мінімум спостережень ──
        if n < 6:
            messagebox.showwarning("Замало спостережень",
                f"ANCOVA потребує щонайменше 6 повних спостережень.\n"
                f"Знайдено: {n}.\n\n"
                f"Перевірте:\n"
                f"  • Перший стовпець містить текстові назви груп\n"
                f"  • Числові дані введені у правильні стовпці\n"
                f"  • Немає порожніх клітинок у заповнених рядках"); return

        # ── Guard 2: щонайменше 2 групи ──
        group_levels = first_seen(groups)
        k = len(group_levels)
        if k < 2:
            messagebox.showwarning("Лише одна група",
                "ANCOVA потребує щонайменше 2 групи.\n"
                "Перевірте що перший стовпець містить різні текстові мітки."); return


        from collections import Counter
        grp_counts = Counter(groups)
        min_grp = min(grp_counts.values())
        if min_grp < 2:
            messagebox.showwarning("Занадто мала група",
                f"У кожній групі потрібно щонайменше 2 спостереження.\n"
                f"Найменша група має лише {min_grp} спостереження(нь)."); return

        # ── Guard 4: check covariates are numeric (already done in parsing) ──
        # ── Guard 5: перевірка на повну мультиколінеарність коваріат ──
        if len(cov_cols) > 1:
            cov_matrix = np.column_stack([np.array(cd) for cd in cov_data])
            corr_matrix = np.corrcoef(cov_matrix.T)
            for i in range(len(cov_cols)):
                for j in range(i+1, len(cov_cols)):
                    if abs(corr_matrix[i,j]) > 0.95:
                        ans = messagebox.askyesno("Висока мультиколінеарність",
                            f"Коваріати '{cov_cols[i]}' і '{cov_cols[j]}' сильно корелюють\n"
                            f"(r = {corr_matrix[i,j]:.3f}).\n\n"
                            "Це може спричинити нестійкі оцінки (проблема мультиколінеарності).\n"
                            "Розгляньте можливість виключити одну з коваріат.\n\n"
                            "Продовжити попри це?")
                        if not ans: return

        y = np.array(y_data, dtype=float)
        covs_arr = [np.array(cd, dtype=float) for cd in cov_data]

        # ── Guard 6: Homogeneity of regression slopes ──
        # Test: fit model with group × covariate interaction
        # If interaction is significant → slopes differ → ANCOVA assumption violated
        slopes_ok = True
        slope_details = []
        for ci, (cov_name, cov_arr) in enumerate(zip(cov_cols, covs_arr)):
            # Build X with intercept + group dummies + covariate + group×covariate
            X_parts = [np.ones(n)]
            g_dummies = []
            for lv in group_levels[1:]:
                d = np.array([1. if g == lv else 0. for g in groups])
                X_parts.append(d); g_dummies.append(d)
            X_parts.append(cov_arr)
            # interaction terms: group_dummy × covariate
            for gd in g_dummies:
                X_parts.append(gd * cov_arr)
            X_int = np.column_stack(X_parts)
            X_no_int = np.column_stack(X_parts[:len(X_parts)-len(g_dummies)])

            _, _, _, sse_full, dfe_full, mse_full = _ols(y, X_int)
            _, _, _, sse_red,  dfe_red,  _        = _ols(y, X_no_int)

            df_int = len(g_dummies)
            ss_int = sse_red - sse_full
            ms_int = ss_int / df_int if df_int > 0 else np.nan
            F_int  = ms_int / mse_full if (not math.isnan(mse_full) and mse_full > 0) else np.nan
            p_int  = float(1 - f_dist.cdf(F_int, df_int, dfe_full)) if not math.isnan(F_int) else np.nan
            slope_details.append((cov_name, F_int, p_int))
            if not math.isnan(p_int) and p_int < alpha:
                slopes_ok = False

        if not slopes_ok:
            failed = [f"'{n}' (F={fmt(F,3)}, p={fmt(p,4)})"
                      for n, F, p in slope_details
                      if not math.isnan(p) and p < alpha]
            ans = messagebox.askyesno(
                "ПОРУШЕНО ПЕРЕДУМОВУ ANCOVA — неоднорідні нахили регресії",
                "Передумова про однорідність нахилів регресії ПОРУШЕНА для:\n"
                + "\n".join(f"  • {f}" for f in failed) + "\n\n"
                "Це означає, що коваріата по-різному впливає на залежну змінну в різних групах.\n"
                "Стандартна ANCOVA у цій ситуації НЕ підходить методично.\n\n"
                "Варіанти:\n"
                "• Використати техніку Джонсона-Неймана (зони значущості)\n"
                "• Виконати окремі регресії для кожної групи\n"
                "• Використати ANOVA з взаємодією замість ANCOVA\n\n"
                "Продовжити з ANCOVA попри це? (НЕ рекомендується)")
            if not ans: return

        # ── Build ANCOVA model (Type III SS) ──
        # X: intercept + group dummies + covariate(s)
        X_parts = [np.ones(n)]
        ts = {"Intercept": [0]}
        idx_cur = 1
        # group factor
        g_idx = []
        for lv in group_levels[1:]:
            d = np.array([1. if g == lv else 0. for g in groups])
            X_parts.append(d); g_idx.append(idx_cur); idx_cur += 1
        ts[group_col] = g_idx
        # covariates
        for cov_name, cov_arr in zip(cov_cols, covs_arr):
            cov_norm = (cov_arr - np.mean(cov_arr)) / (np.std(cov_arr, ddof=1) + 1e-12)
            X_parts.append(cov_norm)
            ts[f"Коваріата: {cov_name}"] = [idx_cur]; idx_cur += 1

        X = np.column_stack(X_parts)
        beta, yhat, residuals, sse, dfe, mse = _ols(y, X)
        sst = float(np.sum((y - np.mean(y))**2))

        # Type III SS for each term
        anova_rows = []
        for term, idx_list in ts.items():
            if term == "Intercept": continue
            keep = [i for i in range(X.shape[1]) if i not in idx_list]
            _, _, _, sse_red, _, _ = _ols(y, X[:, keep])
            ss = float(sse_red - sse)
            df = len(idx_list)
            ms = ss / df if df > 0 else np.nan
            F  = ms / mse if (not math.isnan(mse) and mse > 0) else np.nan
            p  = float(1 - f_dist.cdf(F, df, dfe)) if not math.isnan(F) else np.nan
            if math.isnan(p): mark = ""
            elif p < alpha/5: mark = "**"
            elif p < alpha: mark = "*"
            else: mark = ""
            concl = f"значуще {mark}" if mark else ("незнач." if not math.isnan(p) else "–")
            anova_rows.append([term, fmt(ss,4), str(df), fmt(ms,4), fmt(F,4), fmt(p,4), concl])

        anova_rows.append(["Залишок", fmt(sse,4), str(dfe), fmt(mse,4), "", "", ""])
        anova_rows.append(["Загальна",    fmt(sst,4), str(n-1), "", "", "", ""])

        # ── Guard 7: Normality of residuals ──
        try: W_res, p_res = shapiro(residuals) if len(residuals) >= 3 else (np.nan, np.nan)
        except Exception: W_res, p_res = np.nan, np.nan
        if not math.isnan(p_res) and p_res <= alpha and transform != "log":
            ans = messagebox.askyesno("Ненормальні залишки",
                f"Залишки не відповідають нормальному розподілу\n"
                f"(Шапіро–Вілк: W={fmt(W_res,4)}, p={fmt(p_res,4)}).\n\n"
                "ANCOVA передбачає нормальний розподіл залишків.\n\n"
                "Застосувати логарифмічну трансформацію (ln) до залежної змінної Y "
                "і повторити аналіз автоматично?\n\n"
                "«Так» — застосувати ln(Y) і перерахувати\n"
                "«Ні» — продовжити без трансформації (з поточними ненормальними залишками)")
            if ans:
                self._run(transform="log")
                return

        # ── Guard 8: Homogeneity of variances (Levene) ──
        grp_residuals = defaultdict(list)
        for g, r in zip(groups, residuals): grp_residuals[g].append(r)
        lev_F, lev_p = levene_test(dict(grp_residuals))
        if not math.isnan(lev_p) and lev_p < alpha:
            ans = messagebox.askyesno("Неоднорідність дисперсій (тест Лівена)",
                f"Тест Лівена: F={fmt(lev_F,4)}, p={fmt(lev_p,4)}\n\n"
                "Дисперсії істотно відрізняються між групами.\n"
                "ANCOVA доволі стійка до цього порушення при рівних розмірах груп,\n"
                "але результати можуть бути ненадійними при нерівних розмірах груп.\n\n"
                "Продовжити попри це?")
            if not ans: return

        # ── Adjusted (LS) means ──
        # Compute adjusted means: predict at grand mean of each covariate
        cov_grand_means = [np.mean(ca) for ca in covs_arr]
        adj_means = {}
        for lv in group_levels:
            x_pred = [1.0]  # intercept
            for ref_lv in group_levels[1:]:
                x_pred.append(1.0 if lv == ref_lv else 0.0)
            for ca_mean, ca_arr in zip(cov_grand_means, covs_arr):
                cov_norm_mean = (ca_mean - np.mean(ca_arr)) / (np.std(ca_arr, ddof=1) + 1e-12)
                x_pred.append(cov_norm_mean)
            adj_means[lv] = float(np.dot(beta, x_pred))

        # unadjusted means
        raw_means = {lv: float(np.mean([y_data[i] for i, g in enumerate(groups) if g == lv]))
                     for lv in group_levels}

        R2 = 1 - sse/sst if sst > 0 else np.nan

        # ── Попарні порівняння скоригованих середніх ─────────────
        # SE рахується через коваріаційну матрицю оцінок MSE·(X'X)⁻¹,
        # а НЕ через спрощену формулу sqrt(MSE·(1/n1+1/n2)) — та формула
        # коректна лише для порівняння НЕскоригованих середніх (звичайний
        # ANOVA) і ігнорує додаткову невизначеність від коваріати. Для
        # скоригованих (LS) середніх різниця = різниця коефіцієнтів при
        # дамі-змінних групи (доданок коваріати скорочується, оскільки
        # обидві групи прогнозуються при однаковому — середньому — рівні
        # коваріати), а її SE — це коректний контраст c'(X'X)⁻¹c·MSE,
        # що враховує кореляцію між оцінками коефіцієнтів.
        try:
            XtX_inv = np.linalg.pinv(X.T @ X)
        except Exception:
            XtX_inv = None
        grp_col_idx = {group_levels[0]: None}
        for lv, gi in zip(group_levels[1:], g_idx):
            grp_col_idx[lv] = gi

        def _contrast_se(i_pos, i_neg):
            if XtX_inv is None: return np.nan
            c = np.zeros(X.shape[1])
            if i_pos is not None: c[i_pos] = 1.0
            if i_neg is not None: c[i_neg] = -1.0
            var = float(mse * (c @ XtX_inv @ c))
            return math.sqrt(var) if var > 0 else np.nan

        m_tests = len(group_levels) * (len(group_levels)-1) / 2
        ph_data = []
        for lv1, lv2 in combinations(group_levels, 2):
            se = _contrast_se(grp_col_idx[lv1], grp_col_idx[lv2])
            diff = adj_means[lv1] - adj_means[lv2]
            if math.isnan(se) or se == 0:
                ph_data.append((lv1, lv2, diff, None, None)); continue
            t_val = abs(diff) / se
            p_raw = 2 * (1 - float(t_dist.cdf(t_val, dfe)))
            p_adj = min(1., p_raw * m_tests)
            ph_data.append((lv1, lv2, diff, t_val, p_adj))

        transform_label = "ln(Y)" if transform == "log" else None
        self._show_results(anova_rows, adj_means, raw_means, group_levels,
                           residuals, W_res, p_res, lev_F, lev_p,
                           slope_details, R2, mse, dfe, alpha, y, yhat, groups,
                           ph_data, dv_col, transform_label)

    def _show_results(self, anova_rows, adj_means, raw_means, group_levels,
                      residuals, W_res, p_res, lev_F, lev_p,
                      slope_details, R2, mse, dfe, alpha, y, yhat, groups,
                      ph_data=None, dv_col="Y", transform_label=None):
        win = tk.Toplevel(self.win); win.title("ANCOVA — Результати")
        win.geometry("1150x760"); set_icon(win)

        # scrollable body — ширина body синхронізована з canvas, тож усе
        # всередині (зокрема таблиці) може розтягуватись на всю ширину вікна
        main = tk.Frame(win); main.pack(fill=tk.BOTH, expand=True)
        vsb = ttk.Scrollbar(main, orient="vertical"); vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas = tk.Canvas(main, yscrollcommand=vsb.set); canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.config(command=canvas.yview)
        body = tk.Frame(canvas)
        body_win = canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(body_win, width=e.width))
        win.bind("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        def _head(txt):
            tk.Label(body, text=txt, font=("Times New Roman",12,"bold"),
                     anchor="w").pack(fill=tk.X, padx=10, pady=8)
        def _txt(txt, color="#000000"):
            tk.Label(body, text=txt, font=("Times New Roman",11), fg=color,
                     anchor="w", justify="left").pack(fill=tk.X, padx=10, pady=1)
        def _tbl(headers, rows):
            f, _ = make_tv(body, headers, rows); f.pack(fill=tk.BOTH, expand=True, padx=10, pady=2)

        _head("ANCOVA — Коваріаційний аналіз")
        subtitle = f"Залежна змінна: {dv_col}"
        if transform_label:
            subtitle += f"  (застосовано трансформацію {transform_label})"
        _txt(subtitle, "#1a4b8c")
        _txt(f"R² = {fmt(R2,4)}   |   MSE = {fmt(mse,4)}   |   df_error = {dfe}   |   α = {alpha}")

        # Assumption checks
        _head("Перевірка передумов")
        norm_color = "#000000" if math.isnan(p_res) or p_res > alpha else "#c62828"
        _txt(f"Нормальність залишків (Shapiro–Wilk):  W={fmt(W_res,4)},  p={fmt(p_res,4)}  "
             f"{'✓ Гаразд' if not math.isnan(p_res) and p_res > alpha else '⚠ ПОРУШЕНО'}",
             norm_color)
        lev_color = "#000000" if math.isnan(lev_p) or lev_p >= alpha else "#c62828"
        _txt(f"Однорідність дисперсій (Левен):  F={fmt(lev_F,4)},  p={fmt(lev_p,4)}  "
             f"{'✓ Гаразд' if not math.isnan(lev_p) and lev_p >= alpha else '⚠ ПОРУШЕНО'}",
             lev_color)
        for cov_name, F_sl, p_sl in slope_details:
            sl_ok = math.isnan(p_sl) or p_sl >= alpha
            sl_color = "#000000" if sl_ok else "#c62828"
            _txt(f"Однорідність нахилів ({cov_name}):  F={fmt(F_sl,4)},  p={fmt(p_sl,4)}  "
                 f"{'✓ Гаразд' if sl_ok else '⚠ ПОРУШЕНО — нахили відрізняються'}",
                 sl_color)

        # ANOVA table
        _head("Таблиця ANCOVA (Тип III SS)")
        _tbl(["Джерело","SS","df","MS","F","p","Висновок"], anova_rows)

        # Adjusted means
        _head("Середні за групами")
        means_rows = [[lv, fmt(raw_means[lv],4), fmt(adj_means[lv],4)]
                      for lv in group_levels]
        _tbl(["Група","Нескориговане середнє","Скориговане середнє (LS Mean)"], means_rows)

        # Pairwise comparisons of adjusted means (Bonferroni t-test on LS means)
        _head("Попарні порівняння скоригованих середніх (Бонферроні)")
        ph_rows = []
        for lv1, lv2, diff, t_val, p_adj in (ph_data or []):
            if t_val is None:
                ph_rows.append([f"{lv1} vs {lv2}", fmt(diff,4), "–", "–", "–"]); continue
            if p_adj < alpha/5: mark = "**"
            elif p_adj < alpha: mark = "*"
            else: mark = ""
            concl = f"значуще {mark}" if mark else "незнач."
            ph_rows.append([f"{lv1} vs {lv2}", fmt(diff,4), fmt(t_val,4), fmt(p_adj,4), concl])
        _tbl(["Порівняння","Різниця","t","p (Bonf.)","Висновок"], ph_rows)
        _txt(f"Позначення: * p<α  ** p<α/5  (α={alpha}, обраний вище). "
             f"«Незнач.» — статистично значущої різниці не виявлено при цьому рівні α.",
             "#555555")

        # Plots
        if HAS_MPL:
            fig = Figure(figsize=(10, 6), dpi=100)
            ax1 = fig.add_subplot(121)
            ax1.scatter(yhat, residuals, s=22, color="#4c72b0", alpha=0.8)
            ax1.axhline(0, color="k", lw=0.8)
            ax1.set_xlabel("Підігнані значення"); ax1.set_ylabel("Залишки")
            ax1.set_title("Залишки vs Підігнані"); ax1.yaxis.grid(True, alpha=0.3)

            from scipy.stats import probplot
            ax2 = fig.add_subplot(122)
            res_sort = np.sort(residuals)
            rp = probplot(residuals, dist="norm")
            ax2.plot(rp[0][0], rp[0][1], 'o', markersize=4, color="#4c72b0")
            ax2.plot(rp[0][0], rp[1][1] + rp[1][0]*rp[0][0], 'r-', lw=1)
            ax2.set_xlabel("Теоретичні квантилі"); ax2.set_ylabel("Вибіркові квантилі")
            ax2.set_title("QQ-графік залишків"); ax2.yaxis.grid(True, alpha=0.3)
            fig.tight_layout()
            embed_figure(fig, body)


# ═══════════════════════════════════════════════════════════════
# MANOVA — Multivariate Analysis of Variance
# ═══════════════════════════════════════════════════════════════
