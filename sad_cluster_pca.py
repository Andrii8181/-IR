# sad_cluster_pca.py — Кластерний аналіз, PCA
# -*- coding: utf-8 -*-
from sad_common import *

class ClusterWindow:
    """Кластерний аналіз — ієрархічна кластеризація."""

    HELP_TEXT = """
КЛАСТЕРНИЙ АНАЛІЗ — ПОКРОКОВА ІНСТРУКЦІЯ
══════════════════════════════════════════

ЩО ТАКЕ КЛАСТЕРНИЙ АНАЛІЗ?
  Кластерний аналіз групує об'єкти (сорти, зразки, ділянки) так,
  щоб схожі між собою потрапили в один кластер, а несхожі — у різні.
  Результат відображається на ДЕНДРОГРАМІ — деревоподібному графіку.

КОЛИ ВИКОРИСТОВУВАТИ?
  ✓ Класифікація сортів за комплексом показників якості
  ✓ Групування ґрунтових проб за хімічним складом
  ✓ Виявлення природних груп без попередніх гіпотез
  ✓ Як доповнення до PCA для інтерпретації груп

КРОК 1. СТРУКТУРА ТАБЛИЦІ

  Перший стовпець: Назва об'єкта (сорт, зразок, ділянка — текст)
  Решта стовпців: Числові показники (змінні)

  Приклад (4 сорти, 3 показники):
  | Сорт    | Висота | Врожайність | Маса зерна |
  | Сорт А  |  95.3  |    5.8      |    38.2    |
  | Сорт Б  |  88.5  |    4.9      |    35.1    |
  | Сорт В  | 102.4  |    6.8      |    43.7    |
  | Сорт Д  |  91.2  |    5.2      |    36.8    |

  Перейменуйте заголовки (подвійний клік на синій клітинці).
  Мінімум: 2 об'єкти, 1 показник.
  Програма автоматично стандартизує дані (z-оцінки).

КРОК 2. ВИБІР МЕТОДУ ЗЧЕПЛЕННЯ

  Метод зчеплення визначає як вимірюється відстань між кластерами.

  ward (рекомендується для більшості випадків):
    Мінімізує внутрішньокластерну дисперсію.
    Дає компактні, приблизно рівні кластери.
    Найпопулярніший метод у біологічних дослідженнях. ✓

  complete (повне зчеплення):
    Відстань між кластерами = відстань між їх найдальшими об'єктами.
    Дає компактні кластери схожого розміру.
    Добре коли кластери чіткі і компактні.

  average (середнє зчеплення, UPGMA):
    Відстань = середня між усіма парами об'єктів двох кластерів.
    Компроміс між ward і complete.
    Широко використовується у філогенетичному аналізі.

  single (одиночне зчеплення):
    Відстань = відстань між найближчими об'єктами кластерів.
    Схильний до «ефекту ланцюга» — довгих витягнутих кластерів.
    Корисний для виявлення викидів і незвичних груп.

  ЯК ОБРАТИ?
    → Не знаєте яким почати → ward
    → Хочете рівні компактні групи → complete
    → Є підозра на ланцюговий зв'язок → average
    → Шукаєте нетипові об'єкти → single

КРОК 3. ВИБІР КІЛЬКОСТІ КЛАСТЕРІВ k

  k — скільки груп ви хочете отримати.

  ЯК ВИЗНАЧИТИ ПРАВИЛЬНЕ k?

  Спосіб 1: Візуальний аналіз дендрограми (найкращий!)
    Дивіться на дендрограму:
    Де є ВЕЛИКИЙ стрибок у висоті з'єднання між гілками?
    Там і є природна межа кластерів.
    Проведіть уявну горизонтальну лінію нижче цього стрибка →
    кількість вертикальних гілок що її перетинають = k.

  Спосіб 2: Правило великого стрибка
    Висота з'єднання на дендрограмі = «несхожість».
    Найбільший стрибок висоти між двома сусідніми з'єднаннями
    вказує на оптимальне k.

  Спосіб 3: Змістовна логіка
    Якщо ви знаєте що в природі є 3 групи (ранньостиглі,
    середньостиглі, пізньостиглі) → k=3.

  ТИПОВІ ЗНАЧЕННЯ ДЛЯ АГРОНОМІЇ:
    Класифікація сортів: k = 2–5
    Групування ґрунтових проб: k = 3–6
    Екологічні зони: k = 3–8

КРОК 4. ЧИТАННЯ ДЕНДРОГРАМИ

  Горизонтальна вісь: об'єкти (сорти, зразки)
  Вертикальна вісь: відстань (несхожість)

  Об'єкти що з'єднуються НИЗЬКО → дуже схожі
  Об'єкти що з'єднуються ВИСОКО → дуже несхожі
  Різні кольори = різні кластери при обраному k
  Горизонтальна пунктирна лінія = поріг відсікання для k кластерів

  Об'єкт що з'єднується останнім (найвище) →
  найбільш відмінний від усіх інших (потенційний викид!)

КРОК 5. ТАБЛИЦЯ ПРИНАЛЕЖНОСТІ

  Після дендрограми виводиться таблиця:
  кожен об'єкт і номер його кластера (1, 2, 3...).
"""

    def __init__(self, parent, gs):
        self.win = tk.Toplevel(parent)
        self.win.title("Кластерний аналіз")
        self.win.geometry("1300x680"); set_icon(self.win)
        self.gs = gs
        self._cl_fig = None
        self._cl_gs  = {
            "font_family":    "Times New Roman",
            "font_size":      9,
            "leaf_font_size": 9,
            "line_color":     "#2176ae",
            "threshold_color":"#c62828",
            "show_threshold": True,
            "figsize_w":      10,
            "figsize_h":      5.5,
        }
        self._build()

    def _build(self):
        # ── Toolbar ──────────────────────────────────────────
        top = tk.Frame(self.win, padx=8, pady=6); top.pack(fill=tk.X)
        tk.Button(top, text="▶ Кластеризувати", bg="#c62828", fg="white",
                  font=("Times New Roman", 13),
                  command=self._run).pack(side=tk.LEFT, padx=4)

        # Параметри
        tk.Label(top, text="Метод:", font=("Times New Roman",12)).pack(side=tk.LEFT, padx=(8,2))
        self.meth_var = tk.StringVar(value="ward")
        ttk.Combobox(top, textvariable=self.meth_var,
                     values=["ward","complete","average","single"],
                     state="readonly", width=12).pack(side=tk.LEFT, padx=2)
        tk.Label(top, text="k:", font=("Times New Roman",12)).pack(side=tk.LEFT, padx=(8,2))
        self.k_var = tk.IntVar(value=3)
        tk.Spinbox(top, from_=2, to=20, textvariable=self.k_var,
                   width=4, font=("Times New Roman",11)).pack(side=tk.LEFT, padx=2)

        tk.Button(top, text="📊 Попередній аналіз", bg="#1a6b1a", fg="white",
                  font=("Times New Roman",11), relief=tk.FLAT, padx=8, pady=3,
                  cursor="hand2", command=self._recommend_method).pack(side=tk.LEFT, padx=(10,4))

        # Налаштування — спадне меню
        mb2 = tk.Menubutton(top, text="⚙ Налаштування ▾",
                            font=("Times New Roman",11), relief=tk.RAISED, bd=2)
        mb2.pack(side=tk.LEFT, padx=6)
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

        tk.Button(top, text="Вставити з буфера",
                  font=("Times New Roman",11),
                  command=self._paste).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="📚 Довідка", bg="#1a4b8c", fg="white",
                  font=("Times New Roman",11),
                  command=self._show_help).pack(side=tk.LEFT, padx=4)

        tk.Label(top,
                 text="Подвійний клік на заголовку → перейменувати",
                 font=("Times New Roman",9), fg="#666").pack(side=tk.LEFT, padx=6)

        # ── Таблиця ─────────────────────────────────────────
        mid = tk.Frame(self.win); mid.pack(fill=tk.BOTH, expand=True, padx=8)
        self.rows_n = 18; self.cols_n = 8
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

        default_hdr = ["Назва об'єкта"] + [f"Показник {j}" for j in range(1, self.cols_n)]
        self.header_vars = []; self.header_labels = []
        for j in range(self.cols_n):
            var = tk.StringVar(value=default_hdr[j] if j < len(default_hdr) else f"П{j}")
            self.header_vars.append(var)
            lbl = tk.Label(self.inner, textvariable=var, width=12, cursor="hand2",
                           bg="#1a4b8c", fg="white", relief=tk.RIDGE,
                           font=("Times New Roman",11,"bold"))
            lbl.grid(row=0, column=j, padx=1, pady=1, sticky="nsew")
            lbl.bind("<Double-Button-1>", lambda e, idx=j: self._rename_col(idx))
            self.header_labels.append(lbl)

        self.entries = []
        for i in range(self.rows_n):
            row_ = []
            for j in range(self.cols_n):
                e = tk.Entry(self.inner, width=12, font=("Times New Roman",11),
                             highlightthickness=1, highlightbackground="#c0c0c0")
                e.grid(row=i+1, column=j, padx=1, pady=1)
                row_.append(e)
            self.entries.append(row_)
        _bind_nav(self.entries, self.win)
        _bind_fill_handle(self.entries, self.win)

    # ── Перейменування ────────────────────────────────────────
    def _rename_col(self, idx):
        dlg = tk.Toplevel(self.win); dlg.title("Перейменувати")
        dlg.resizable(False, False); dlg.grab_set()
        tk.Label(dlg, text=f"Назва стовпця {idx+1}:",
                 font=("Times New Roman",12)).pack(padx=16, pady=14)
        var = tk.StringVar(value=self.header_vars[idx].get())
        e = tk.Entry(dlg, textvariable=var, font=("Times New Roman",12), width=26)
        e.pack(padx=16, pady=4); e.select_range(0, tk.END); e.focus_set()
        def apply():
            nm = var.get().strip()
            if nm: self.header_vars[idx].set(nm)
            dlg.destroy()
        tk.Button(dlg, text="OK", bg="#c62828", fg="white",
                  font=("Times New Roman",12), command=apply).pack(pady=(4,14))
        dlg.bind("<Return>", lambda ev: apply()); center_win(dlg)

    # ── Управління таблицею ───────────────────────────────────
    def _add_row(self):
        i = self.rows_n; row_ = []
        for j in range(self.cols_n):
            e = tk.Entry(self.inner, width=12, font=("Times New Roman",11),
                         highlightthickness=1, highlightbackground="#c0c0c0")
            e.grid(row=i+1, column=j, padx=1, pady=1)
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
        var = tk.StringVar(value=f"Показник {ci}")
        self.header_vars.append(var)
        lbl = tk.Label(self.inner, textvariable=var, width=12, cursor="hand2",
                       bg="#1a4b8c", fg="white", relief=tk.RIDGE,
                       font=("Times New Roman",11,"bold"))
        lbl.grid(row=0, column=ci, padx=1, pady=1, sticky="nsew")
        lbl.bind("<Double-Button-1>", lambda e, idx=ci: self._rename_col(idx))
        self.header_labels.append(lbl)
        for i, row_ in enumerate(self.entries):
            e = tk.Entry(self.inner, width=12, font=("Times New Roman",11),
                         highlightthickness=1, highlightbackground="#c0c0c0")
            e.grid(row=i+1, column=ci, padx=1, pady=1)
            row_.append(e)
        _bind_nav(self.entries, self.win)
        _bind_fill_handle(self.entries, self.win)

    def _del_col(self):
        if self.cols_n <= 2: return
        self.header_labels.pop().destroy(); self.header_vars.pop()
        for row_ in self.entries: row_.pop().destroy()
        self.cols_n -= 1

    def _clear_table(self):
        if not messagebox.askyesno("Очистити",
                "Видалити всі дані? (Заголовки залишаться)"): return
        for row in self.entries:
            for e in row: e.delete(0, tk.END)

    def _save_proj(self):
        generic_save_project(self.win, "cluster", self.header_vars, self.entries)

    def _load_proj(self):
        d = generic_load_project(self.win)
        if d is None: return
        headers = d.get("headers", []); rd = d.get("rows_data", [])
        while self.cols_n < len(headers): self._add_col()
        for j, h in enumerate(headers):
            if j < len(self.header_vars): self.header_vars[j].set(h)
        while len(self.entries) < len(rd): self._add_row()
        for i, rv in enumerate(rd):
            for j, v in enumerate(rv):
                if i < len(self.entries) and j < len(self.entries[i]):
                    self.entries[i][j].delete(0, tk.END); self.entries[i][j].insert(0, v)

    # ── Вставка / Довідка ────────────────────────────────────
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

    def _show_help(self):
        win = tk.Toplevel(self.win); win.title("Довідка — Кластерний аналіз")
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
    def _restyle_cluster(self, win, obj_names, Z, k, method, graph_frame):
        dlg = tk.Toplevel(win); dlg.title("Налаштування дендрограми")
        dlg.resizable(False, False); set_icon(dlg); dlg.grab_set()
        gs = self._cl_gs
        frm = tk.Frame(dlg, padx=16, pady=14); frm.pack()
        rb_f = ("Times New Roman",12)

        ff_v  = tk.StringVar(value=gs["font_family"])
        fz_v  = tk.IntVar(value=gs["font_size"])
        lf_v  = tk.IntVar(value=gs["leaf_font_size"])
        fw_v  = tk.DoubleVar(value=gs["figsize_w"])
        fh_v  = tk.DoubleVar(value=gs["figsize_h"])
        st_v  = tk.BooleanVar(value=gs["show_threshold"])
        lc_ref = [gs["line_color"]]
        tc_ref = [gs["threshold_color"]]

        rows_cfg = [
            ("Шрифт:",                  "combo",  ff_v, ["Times New Roman","Arial","Calibri","Georgia"]),
            ("Розмір підписів осей:",   "spin",   fz_v, (6,18)),
            ("Розмір підписів об'єктів:", "spin", lf_v, (5,16)),
            ("Ширина графіка:",         "scale",  fw_v, (5.,20.)),
            ("Висота графіка:",         "scale",  fh_v, (3.,12.)),
            ("Показувати поріг k:",     "check",  st_v, None),
        ]
        btn_refs = {}
        for ri, (lbl, wt, var, opts) in enumerate(rows_cfg):
            tk.Label(frm, text=lbl, font=rb_f).grid(row=ri, column=0, sticky="w", pady=4)
            if wt=="combo":
                ttk.Combobox(frm, textvariable=var, values=opts,
                             state="readonly", width=20).grid(row=ri, column=1, sticky="w", padx=8)
            elif wt=="spin":
                tk.Spinbox(frm, from_=opts[0], to=opts[1], textvariable=var,
                           width=7).grid(row=ri, column=1, sticky="w", padx=8)
            elif wt=="scale":
                tk.Scale(frm, from_=opts[0], to=opts[1], resolution=0.5,
                         orient="horizontal", variable=var,
                         length=160).grid(row=ri, column=1, sticky="w", padx=8)
            elif wt=="check":
                tk.Checkbutton(frm, variable=var).grid(row=ri, column=1, sticky="w", padx=8)

        base_r = len(rows_cfg)
        for ri, (lbl, ref) in enumerate([("Колір ліній дендрограми:", lc_ref),
                                          ("Колір порогової лінії:",    tc_ref)]):
            tk.Label(frm, text=lbl, font=rb_f).grid(row=base_r+ri, column=0, sticky="w", pady=4)
            btn = tk.Button(frm, width=6, relief=tk.SUNKEN, bg=ref[0])
            btn.grid(row=base_r+ri, column=1, sticky="w", padx=8)
            def _pick(r=ref, b=btn):
                ch = colorchooser.askcolor(color=r[0], parent=dlg)
                if ch and ch[1]: r[0]=ch[1]; b.configure(bg=ch[1])
            btn.configure(command=_pick); btn_refs[ri] = btn

        def apply():
            self._cl_gs.update({
                "font_family":    ff_v.get(), "font_size": fz_v.get(),
                "leaf_font_size": lf_v.get(), "figsize_w": fw_v.get(),
                "figsize_h":      fh_v.get(), "show_threshold": st_v.get(),
                "line_color":     lc_ref[0],  "threshold_color": tc_ref[0],
            })
            dlg.destroy()
            self._draw_dendrogram(graph_frame, obj_names, Z, k, method)

        bf = tk.Frame(frm); bf.grid(row=base_r+2, column=0, columnspan=2, pady=(14,0))
        tk.Button(bf, text="OK (застосувати)", bg="#c62828", fg="white",
                  font=rb_f, command=apply).pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text="Скасувати", font=rb_f, command=dlg.destroy).pack(side=tk.LEFT)
        center_win(dlg)

    # ── Побудова дендрограми ──────────────────────────────────
    def _draw_dendrogram(self, frame, obj_names, Z, k, method):
        from scipy.cluster.hierarchy import dendrogram as _dendro
        for w in frame.winfo_children(): w.destroy()
        gs = self._cl_gs
        fig = Figure(figsize=(10, 6), dpi=100)
        ax  = fig.add_subplot(111)

        # Кольори гілок через color_threshold
        thresh = float(Z[-(k-1), 2]) if k > 1 else float("inf")
        _dendro(Z, labels=obj_names, ax=ax,
                leaf_rotation=90, leaf_font_size=gs["leaf_font_size"],
                color_threshold=thresh if gs["show_threshold"] else 0,
                above_threshold_color=gs["line_color"])

        if gs["show_threshold"] and k > 1:
            ax.axhline(thresh, color=gs["threshold_color"],
                       lw=1.2, linestyle="--",
                       label=f"Поріг k={k}")
            ax.legend(fontsize=gs["font_size"], framealpha=0.7)

        ax.set_title(f"Ієрархічна кластеризація  |  Метод: {method}  |  k = {k}",
                     fontsize=gs["font_size"]+1, fontfamily=gs["font_family"])
        ax.set_ylabel("Відстань (несхожість)",
                      fontsize=gs["font_size"], fontfamily=gs["font_family"])
        ax.tick_params(labelsize=gs["font_size"])
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.tight_layout()
        self._cl_fig = fig

        embed_figure(fig, frame)

    # ── Попередній аналіз: рекомендація методу кластеризації ──
    def _recommend_method(self):
        from scipy.cluster.hierarchy import linkage, cophenet
        from scipy.spatial.distance import pdist
        from scipy.stats import zscore

        raw = [[e.get().strip() for e in row] for row in self.entries]
        raw = [r for r in raw if any(v for v in r)]
        obj_names = []; data_matrix = []
        for row in raw:
            nm = row[0].strip() if row else ""
            if not nm: continue
            vals = []
            for v in row[1:]:
                if not v: continue
                try: vals.append(float(v.replace(",",".")))
                except Exception: continue
            if vals:
                obj_names.append(nm); data_matrix.append(vals)

        if len(data_matrix) < 4:
            messagebox.showwarning("Замало об'єктів",
                "Потрібно щонайменше 4 об'єкти для попереднього аналізу."); return

        min_cols = min(len(r) for r in data_matrix)
        X = np.array([r[:min_cols] for r in data_matrix], dtype=float)
        X_std = zscore(X, axis=0, ddof=1); X_std = np.nan_to_num(X_std)
        orig_dist = pdist(X_std)

        candidates = []
        for method in ["ward","complete","average","single"]:
            try:
                Z = linkage(X_std, method=method)
                c, _ = cophenet(Z, orig_dist)
                candidates.append((method, float(c)))
            except Exception:
                continue

        if not candidates:
            messagebox.showwarning("", "Не вдалося порівняти методи на цих даних."); return

        candidates.sort(key=lambda x: -x[1])
        best = candidates[0][0]
        self._show_cluster_recommendation(candidates, best)

    def _show_cluster_recommendation(self, candidates, best):
        win = tk.Toplevel(self.win)
        win.title("Попередній аналіз — рекомендація методу кластеризації")
        win.geometry("640x580"); set_icon(win)
        rf = ("Times New Roman",11)

        tk.Label(win, text="Порівняння методів за кофенетичною кореляцією:",
                 font=("Times New Roman",12,"bold"), anchor="w"
                 ).pack(fill=tk.X, padx=12, pady=(12,4))
        tk.Label(win,
                 text="Кофенетична кореляція показує, наскільки добре ієрархія "
                      "кластеризації (дендрограма) зберігає РЕАЛЬНІ відстані між "
                      "об'єктами з вихідних даних. Чим ближче до 1 — тим точніше "
                      "дендрограма відображає структуру даних.",
                 font=("Times New Roman",10), fg="#555", justify="left",
                 wraplength=580, anchor="w").pack(fill=tk.X, padx=12, pady=(0,10))

        def interp(c):
            if c >= 0.75: return "хороша відповідність"
            if c >= 0.5:  return "помірна відповідність"
            return "слабка відповідність"

        rows = [[("★ " if m==best else "  ")+m, fmt(c,4), interp(c)] for m,c in candidates]
        frm, _ = make_tv(win, ["Метод","Кофенетична кореляція","Оцінка"], rows)
        frm.pack(fill=tk.X, padx=12, pady=4)

        tk.Label(win,
                 text="⚠ Це орієнтир, а не остаточне рішення. Ward зазвичай дає "
                      "компактні, збалансовані за розміром кластери й є типовим "
                      "вибором за замовчуванням в агробіологічних дослідженнях, "
                      "навіть якщо його кофенетична кореляція не найвища — тому "
                      "що цей критерій оцінює лише точність відтворення відстаней, "
                      "а не практичну інтерпретованість чи збалансованість груп. "
                      "Single linkage часто дає найвищу кофенетичну кореляцію, "
                      "але схильний до «ланцюгового ефекту» (витягнуті, "
                      "невиразні кластери) — тому висока кореляція сама по собі "
                      "не завжди означає кращий практичний результат.",
                 font=("Times New Roman",10), fg="#555", justify="left",
                 wraplength=580, anchor="w").pack(fill=tk.X, padx=12, pady=(6,8))

        bf = tk.Frame(win); bf.pack(pady=(0,12))
        def _apply():
            self.meth_var.set(best); win.destroy()
        tk.Button(bf, text=f"Обрати «{best}» і закрити", bg="#1a6b1a", fg="white",
                  font=rf, command=_apply).pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text="Закрити (оберу сам)", font=rf,
                  command=win.destroy).pack(side=tk.LEFT, padx=4)
        center_win(win)

    # ── Виконання аналізу ─────────────────────────────────────
    def _run(self):
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist

        raw = [[e.get().strip() for e in row] for row in self.entries]
        raw = [r for r in raw if any(v for v in r)]
        if not raw:
            messagebox.showwarning("Немає даних","Введіть дані у таблицю."); return

        obj_names = []; data_matrix = []
        for row in raw:
            nm = row[0].strip() if row else ""
            if not nm: continue
            vals = []
            for v in row[1:]:
                if not v: continue
                try: vals.append(float(v.replace(",",".")))
                except Exception: continue
            if vals:
                obj_names.append(nm); data_matrix.append(vals)

        if len(data_matrix) < 2:
            messagebox.showwarning("Замало об'єктів",
                "Потрібно щонайменше 2 об'єкти з числовими даними.\n"
                "Перший стовпець = назва об'єкта (текст).\n"
                "Решта стовпців = числові показники."); return

        min_cols = min(len(r) for r in data_matrix)
        X = np.array([r[:min_cols] for r in data_matrix], dtype=float)

        from scipy.stats import zscore
        X_std = zscore(X, axis=0, ddof=1); X_std = np.nan_to_num(X_std)

        method = self.meth_var.get()
        k = self.k_var.get()
        k = max(2, min(k, len(obj_names)))

        try:
            Z = linkage(X_std, method=method)
        except Exception as ex:
            messagebox.showerror("Помилка", str(ex)); return

        labels_cl = fcluster(Z, k, criterion='maxclust')

        if not HAS_MPL:
            messagebox.showwarning("","matplotlib недоступний."); return

        # ── Вікно результатів ──────────────────────────────────
        win = tk.Toplevel(self.win)
        win.title("Кластерний аналіз — Результати")
        win.geometry("1150x800"); set_icon(win)

        # Toolbar результатів (фіксовано зверху)
        tb = tk.Frame(win, padx=6, pady=5); tb.pack(fill=tk.X)

        # Прокручувана область: дендрограма + таблиця приналежності —
        # обидва завжди доступні (раніше graph_frame з expand=True займав
        # увесь простір і таблиця нижче фактично ставала невидимою).
        sa = tk.Frame(win); sa.pack(fill=tk.BOTH, expand=True)
        vsb = ttk.Scrollbar(sa, orient="vertical"); vsb.pack(side=tk.RIGHT, fill=tk.Y)
        sc = tk.Canvas(sa, yscrollcommand=vsb.set, highlightthickness=0)
        sc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.config(command=sc.yview)
        body = tk.Frame(sc)
        body_win = sc.create_window((0,0), window=body, anchor="nw")
        body.bind("<Configure>", lambda e: sc.configure(scrollregion=sc.bbox("all")))
        sc.bind("<Configure>", lambda e: sc.itemconfig(body_win, width=e.width))
        win.bind("<MouseWheel>", lambda e: sc.yview_scroll(int(-1*(e.delta/120)),"units"))

        graph_frame = tk.Frame(body)

        tk.Button(tb, text="📋 Копіювати дендрограму", font=("Times New Roman",11),
                  command=lambda: self._copy_dendro()).pack(side=tk.LEFT, padx=4)
        tk.Button(tb, text="⚙ Налаштування графіка", font=("Times New Roman",11),
                  command=lambda: self._restyle_cluster(
                      win, obj_names, Z, k, method, graph_frame)
                  ).pack(side=tk.LEFT, padx=4)
        tk.Label(tb,
                 text=f"Метод: {method}  |  k = {k}  |  Об'єктів: {len(obj_names)}",
                 font=("Times New Roman",11), fg="#555").pack(side=tk.LEFT, padx=10)

        # Дендрограма
        graph_frame.pack(fill=tk.X, padx=4, pady=4)
        self._draw_dendrogram(graph_frame, obj_names, Z, k, method)

        # Таблиця приналежності
        tbl_frame = tk.Frame(body); tbl_frame.pack(fill=tk.X, padx=8, pady=(4,16))
        tk.Label(tbl_frame, text="Приналежність до кластерів:",
                 font=("Times New Roman",11,"bold"), anchor="w").pack(fill=tk.X)
        membership_rows = sorted(
            [[nm, f"Кластер {cl}"] for nm, cl in zip(obj_names, labels_cl)],
            key=lambda r: r[1])
        frm_m, _ = make_tv(tbl_frame, ["Об'єкт","Кластер"], membership_rows)
        frm_m.pack(fill=tk.X)

    def _copy_dendro(self):
        if self._cl_fig is None:
            messagebox.showwarning("","Спочатку виконайте кластеризацію."); return
        ok, msg = _copy_fig_to_clipboard(self._cl_fig)
        if ok: messagebox.showinfo("","Дендрограму скопійовано.\nВставте у Word через Ctrl+V.")
        else:   messagebox.showwarning("",f"Помилка: {msg}")




# ═══════════════════════════════════════════════════════════════
# PCA MODULE
# ═══════════════════════════════════════════════════════════════
class PCAWindow:
    """Аналіз головних компонент (PCA)."""

    HELP_TEXT = """
PCA — ПОКРОКОВА ІНСТРУКЦІЯ
══════════════════════════════════════════

ЩО ТАКЕ PCA?
  PCA (Principal Component Analysis) — Аналіз головних компонент.
  Стискає багато змінних (показників) до кількох «узагальнених»
  показників (головних компонент, ГК), зберігаючи максимум інформації.

КОЛИ ЗАСТОСОВУВАТИ?
  ✓ У вас 5+ показників для кожного об'єкта (сорту, зразка)
  ✓ Хочете виявити природне групування об'єктів
  ✓ Хочете зрозуміти які показники «ходять разом»
  ✓ Як попередній крок перед MANOVA (якщо n ≤ p)
  ✓ Для виявлення викидів (об'єкти далеко від центру biplot)

КРОК 1. СТРУКТУРА ДАНИХ

  Перший стовпець: Мітка об'єкта (назва сорту, зразка — текст)
                   Необов'язково — якщо числове, вважається показником.
  Решта стовпців: Числові показники (змінні).

  Перший рядок: Назви показників (заголовки, синій рядок).

  Приклад (4 сорти, 4 показники):
  | Сорт    | Врожайн. | Висота | Маса зерна | Вміст білку |
  | Сорт А  |   5.8    |  95.3  |    38.2    |    12.5     |
  | Сорт Б  |   4.9    |  88.5  |    35.1    |    14.2     |

  Мінімум: 2 об'єкти, 2 показники.

КРОК 2. ПІДГОТОВКА І ВИКОНАННЯ
  Перейменуйте заголовки (подвійний клік на синій клітинці).
  Введіть дані або вставте з Excel.
  Натисніть «▶ Виконати PCA».

  Програма автоматично СТАНДАРТИЗУЄ дані (z-оцінки)
  щоб показники з різними одиницями мали однаковий вплив.

КРОК 3. SCREE PLOT (Графік відсіювання)

  Стовпчики: % дисперсії пояснений кожною ГК.
  Червона лінія: кумулятивний % (зростаючий).
  Пунктир: 80% поріг.

  Скільки ГК залишити?
  «Правило ліктя»: знайдіть де графік різко стає пологим
  → точка вище = оптимальна кількість ГК.
  Зазвичай ГК1 + ГК2 пояснюють 70-85% → достатньо.
  Власне значення > 1 (Критерій Кайзера) → включайте.

КРОК 4. BIPLOT (ГК1 × ГК2)

  Точки = об'єкти (сорти, зразки):
    Близькі точки = схожі об'єкти за всіма показниками.
    Далекі точки = дуже різні.
    Кластери = природні групи.

  Червоні стрілки = показники (змінні):
    Довга стрілка = показник добре описаний цими ГК.
    Стрілки в одному напрямку = показники корелюють.
    Стрілки протилежних напрямків = обернена кореляція.
    Стрілки під кутом 90° = незалежні показники.
    Об'єкт близько до стрілки = великe значення цього показника.

КРОК 5. ТАБЛИЦЯ НАВАНТАЖЕНЬ (Loadings)

  Теплова карта: як кожен показник пов'язаний з кожною ГК.
  |Навантаження| > 0.5 вважається значущим.
  Темно-зелений: сильна позитивна кореляція з ГК.
  Темно-червоний: сильна негативна кореляція.

КРОК 6. ТАБЛИЦЯ КОМПОНЕНТ

  Власне значення: дисперсія пояснена кожною ГК.
  % дисперсії: відносний внесок.
  Кумулятивний %: наростаючий підсумок.
"""

    def __init__(self, parent, gs):
        self.win = tk.Toplevel(parent)
        self.win.title("Аналіз головних компонент (PCA)")
        self.win.geometry("1000x680"); set_icon(self.win)
        self.gs = gs
        self._pca_fig = None
        self._pca_gs  = {
            "point_color":   "#dd8452",
            "arrow_color":   "#c62828",
            "bar_color":     "#4c72b0",
            "cum_color":     "#c62828",
            "heatmap_cmap":  "RdYlGn",
            "font_family":   "Times New Roman",
            "font_size":     9,
            "point_size":    30,
            "arrow_scale":   0.7,
            "annotate_obj":  True,
            "annotate_var":  True,
        }
        self._build()

    def _build(self):
        # ── Toolbar ──────────────────────────────────────────
        top = tk.Frame(self.win, padx=8, pady=6); top.pack(fill=tk.X)
        tk.Button(top, text="▶ Виконати PCA", bg="#c62828", fg="white",
                  font=("Times New Roman",13),
                  command=self._run).pack(side=tk.LEFT, padx=4)

        # Налаштування — спадне меню
        mb2 = tk.Menubutton(top, text="⚙ Налаштування ▾",
                            font=("Times New Roman",11), relief=tk.RAISED, bd=2)
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

        tk.Button(top, text="Вставити з буфера",
                  font=("Times New Roman",11),
                  command=self._paste).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="📚 Довідка", bg="#1a4b8c", fg="white",
                  font=("Times New Roman",11),
                  command=self._show_help).pack(side=tk.LEFT, padx=4)

        tk.Label(top,
                 text="Подвійний клік на заголовку → перейменувати показник",
                 font=("Times New Roman",9), fg="#666").pack(side=tk.LEFT, padx=10)

        # ── Таблиця ─────────────────────────────────────────
        mid = tk.Frame(self.win); mid.pack(fill=tk.BOTH, expand=True, padx=8)
        self.rows_n = 18; self.cols_n = 8
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

        self.header_labels = []; self.header_vars = []
        default_headers = ["Мітка об'єкта"] + [f"Показник {j}" for j in range(1, self.cols_n)]
        for j in range(self.cols_n):
            var = tk.StringVar(value=default_headers[j] if j < len(default_headers) else f"П{j}")
            self.header_vars.append(var)
            lbl = tk.Label(self.inner, textvariable=var, relief=tk.RIDGE, width=13,
                           bg="#1a4b8c", fg="white", cursor="hand2",
                           font=("Times New Roman",11,"bold"))
            lbl.grid(row=0, column=j, padx=1, pady=1, sticky="nsew")
            lbl.bind("<Double-Button-1>", lambda e, idx=j: self._rename_col(idx))
            self.header_labels.append(lbl)

        self.entries = []
        for i in range(self.rows_n):
            row_ = []
            for j in range(self.cols_n):
                e = tk.Entry(self.inner, width=13, font=("Times New Roman",11),
                             highlightthickness=1, highlightbackground="#c0c0c0")
                e.grid(row=i+1, column=j, padx=1, pady=1)
                row_.append(e)
            self.entries.append(row_)
        _bind_nav(self.entries, self.win)
        _bind_fill_handle(self.entries, self.win)

    # ── Перейменування заголовка ──────────────────────────────
    def _rename_col(self, idx):
        dlg = tk.Toplevel(self.win); dlg.title("Перейменувати")
        dlg.resizable(False, False); dlg.grab_set()
        tk.Label(dlg, text=f"Назва показника {idx+1}:",
                 font=("Times New Roman",12)).pack(padx=16, pady=14)
        var = tk.StringVar(value=self.header_vars[idx].get())
        e = tk.Entry(dlg, textvariable=var, font=("Times New Roman",12), width=26)
        e.pack(padx=16, pady=4); e.select_range(0, tk.END); e.focus_set()
        def apply():
            nm = var.get().strip()
            if nm: self.header_vars[idx].set(nm)
            dlg.destroy()
        tk.Button(dlg, text="OK", bg="#c62828", fg="white",
                  font=("Times New Roman",12), command=apply).pack(pady=(4,14))
        dlg.bind("<Return>", lambda ev: apply()); center_win(dlg)

    # ── Управління таблицею ───────────────────────────────────
    def _add_row(self):
        i = self.rows_n; row_ = []
        for j in range(self.cols_n):
            e = tk.Entry(self.inner, width=13, font=("Times New Roman",11),
                         highlightthickness=1, highlightbackground="#c0c0c0")
            e.grid(row=i+1, column=j, padx=1, pady=1)
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
        var = tk.StringVar(value=f"Показник {ci}")
        self.header_vars.append(var)
        lbl = tk.Label(self.inner, textvariable=var, relief=tk.RIDGE, width=13,
                       bg="#1a4b8c", fg="white", cursor="hand2",
                       font=("Times New Roman",11,"bold"))
        lbl.grid(row=0, column=ci, padx=1, pady=1, sticky="nsew")
        lbl.bind("<Double-Button-1>", lambda e, idx=ci: self._rename_col(idx))
        self.header_labels.append(lbl)
        for i, row_ in enumerate(self.entries):
            e = tk.Entry(self.inner, width=13, font=("Times New Roman",11),
                         highlightthickness=1, highlightbackground="#c0c0c0")
            e.grid(row=i+1, column=ci, padx=1, pady=1)
            row_.append(e)
        _bind_nav(self.entries, self.win)
        _bind_fill_handle(self.entries, self.win)

    def _del_col(self):
        if self.cols_n <= 3: return
        self.header_labels.pop().destroy(); self.header_vars.pop()
        for row_ in self.entries: row_.pop().destroy()
        self.cols_n -= 1

    def _clear_table(self):
        if not messagebox.askyesno("Очистити",
                "Видалити всі дані? (Заголовки залишаться)"): return
        for row in self.entries:
            for e in row: e.delete(0, tk.END)

    def _save_proj(self):
        generic_save_project(self.win, "pca", self.header_vars, self.entries)

    def _load_proj(self):
        d = generic_load_project(self.win)
        if d is None: return
        headers = d.get("headers", []); rd = d.get("rows_data", [])
        while self.cols_n < len(headers): self._add_col()
        for j, h in enumerate(headers):
            if j < len(self.header_vars): self.header_vars[j].set(h)
        while len(self.entries) < len(rd): self._add_row()
        for i, rv in enumerate(rd):
            for j, v in enumerate(rv):
                if i < len(self.entries) and j < len(self.entries[i]):
                    self.entries[i][j].delete(0, tk.END); self.entries[i][j].insert(0, v)

    # ── Вставка / Довідка ────────────────────────────────────
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

    def _show_help(self):
        win = tk.Toplevel(self.win); win.title("Довідка — PCA")
        win.geometry("700x660"); set_icon(win)
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

    # ── Універсальний діалог налаштувань одного графіка PCA ───
    def _pca_settings_dialog(self, fields, callback=None, title="Налаштування графіка"):
        """fields: список (підпис, ключ_у_gs, тип, опції).
        тип: 'combo' | 'spin' | 'scale' | 'check' | 'color'."""
        dlg = tk.Toplevel(self._pca_win); dlg.title(title)
        dlg.resizable(False, False); set_icon(dlg); dlg.grab_set()
        gs = self._pca_gs
        frm = tk.Frame(dlg, padx=16, pady=14); frm.pack()
        rb_f = ("Times New Roman", 12)
        refs = {}

        for ri, (lbl, key, wtype, opts) in enumerate(fields):
            tk.Label(frm, text=lbl, font=rb_f, wraplength=260, justify="left"
                     ).grid(row=ri, column=0, sticky="w", pady=4)
            if wtype == "combo":
                v = tk.StringVar(value=gs[key])
                ttk.Combobox(frm, textvariable=v, values=opts, state="readonly",
                             width=18).grid(row=ri, column=1, sticky="w", padx=8)
                refs[key] = ("var", v)
            elif wtype == "spin":
                v = tk.IntVar(value=gs[key])
                tk.Spinbox(frm, from_=opts[0], to=opts[1], textvariable=v,
                           width=7).grid(row=ri, column=1, sticky="w", padx=8)
                refs[key] = ("var", v)
            elif wtype == "scale":
                v = tk.DoubleVar(value=gs[key])
                tk.Scale(frm, from_=opts[0], to=opts[1], resolution=0.05,
                         orient="horizontal", variable=v,
                         length=160).grid(row=ri, column=1, sticky="w", padx=8)
                refs[key] = ("var", v)
            elif wtype == "check":
                v = tk.BooleanVar(value=gs[key])
                tk.Checkbutton(frm, variable=v).grid(row=ri, column=1, sticky="w", padx=8)
                refs[key] = ("var", v)
            elif wtype == "color":
                btn = tk.Button(frm, width=6, relief=tk.SUNKEN, bg=gs[key])
                btn.grid(row=ri, column=1, sticky="w", padx=8)
                box = {"v": gs[key]}
                def _pick(k=key, b=btn, box=box):
                    ch = colorchooser.askcolor(color=box["v"], parent=dlg)
                    if ch and ch[1]: box["v"] = ch[1]; b.configure(bg=ch[1])
                btn.configure(command=_pick)
                refs[key] = ("box", box)

        def apply():
            upd = {}
            for key, (kind, obj) in refs.items():
                upd[key] = obj.get() if kind == "var" else obj["v"]
            self._pca_gs.update(upd)
            dlg.destroy()
            if callback: callback()

        bf = tk.Frame(frm); bf.grid(row=len(fields), column=0, columnspan=2, pady=(14,0))
        tk.Button(bf, text="OK", bg="#c62828", fg="white",
                  font=rb_f, command=apply).pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text="Скасувати", font=rb_f, command=dlg.destroy).pack(side=tk.LEFT)
        center_win(dlg)

    # ── Toolbar окремого графіка (як у звітах ANOVA) ──────────
    def _pca_tab_toolbar(self, frame, key, rebuild_fn, settings_fn):
        tb = tk.Frame(frame, bg="#f0f0f0", padx=4, pady=4)
        tb.pack(fill=tk.X, side=tk.TOP)
        tk.Button(tb, text="💾 Зберегти PNG", font=("Times New Roman",10),
                  command=lambda: self._pca_save_png(key)).pack(side=tk.LEFT, padx=4)
        tk.Button(tb, text="📋 Копіювати", font=("Times New Roman",10),
                  command=lambda: self._pca_copy_fig(key)).pack(side=tk.LEFT, padx=4)
        tk.Button(tb, text="⚙ Налаштування", font=("Times New Roman",10),
                  bg="#1a4b8c", fg="white", command=settings_fn).pack(side=tk.LEFT, padx=4)
        return tb

    def _pca_save_png(self, key):
        fig = self._pca_figs.get(key)
        if fig is None: messagebox.showwarning("","Графік відсутній."); return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG","*.png"),("SVG","*.svg")],
            title="Зберегти графік")
        if not path: return
        try:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            messagebox.showinfo("Збережено", f"Збережено:\n{path}")
        except Exception as ex:
            messagebox.showerror("Помилка", str(ex))

    def _pca_copy_fig(self, key):
        fig = self._pca_figs.get(key)
        if fig is None: messagebox.showwarning("","Графік відсутній."); return
        ok, msg = _copy_fig_to_clipboard(fig)
        if ok: messagebox.showinfo("","Графік скопійовано (PNG).\nВставте у Word через Ctrl+V.")
        else:   messagebox.showwarning("",f"Помилка: {msg}")

    @staticmethod
    def _pca_bind_wheel(widget, handler):
        widget.bind("<MouseWheel>", handler)
        for ch in widget.winfo_children():
            PCAWindow._pca_bind_wheel(ch, handler)

    # ── Виконання PCA ─────────────────────────────────────────
    def _run(self):
        raw = [[e.get().strip() for e in row] for row in self.entries]
        raw = [r for r in raw if any(v for v in r)]
        if not raw:
            messagebox.showwarning("Немає даних","Введіть дані у таблицю."); return

        # Визначаємо чи перша колонка = мітки
        has_labels = False
        try: float(raw[0][0].replace(",",".")); has_labels = False
        except ValueError: has_labels = True

        start_col = 1 if has_labels else 0
        obj_names = []; data_rows = []
        for i, row in enumerate(raw):
            nm = row[0].strip() if has_labels else f"Об'єкт {i+1}"
            if not nm: nm = f"Об'єкт {i+1}"
            vals = []
            for v in row[start_col:]:
                if not v: continue
                try: vals.append(float(v.replace(",",".")))
                except Exception: continue
            if vals:
                data_rows.append(vals)
                obj_names.append(nm)

        if len(data_rows) < 2:
            messagebox.showwarning("Замало об'єктів",
                "Потрібно щонайменше 2 об'єкти (рядки з числовими даними)."); return
        min_c = min(len(r) for r in data_rows)
        if min_c < 2:
            messagebox.showwarning("Замало показників",
                "Потрібно щонайменше 2 числові показники."); return

        # Назви змінних з заголовків
        var_names = []
        for j in range(start_col, self.cols_n):
            if j < len(self.header_vars):
                var_names.append(self.header_vars[j].get().strip() or f"П{j}")
            else:
                var_names.append(f"П{j}")
        var_names = var_names[:min_c]

        X = np.array([r[:min_c] for r in data_rows], dtype=float)

        # Стандартизація
        from scipy.stats import zscore
        X_std = zscore(X, axis=0, ddof=1); X_std = np.nan_to_num(X_std)

        # PCA через власні вектори
        cov_m = np.cov(X_std.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_m)
        idx_s = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx_s]; eigenvectors = eigenvectors[:, idx_s]
        # Тільки невід'ємні власні значення
        eigenvalues = np.maximum(eigenvalues, 0)
        total_var = np.sum(eigenvalues)
        explained  = eigenvalues / total_var * 100 if total_var > 0 else eigenvalues*0
        scores     = X_std @ eigenvectors
        n_comp     = len(eigenvalues)

        if not HAS_MPL:
            messagebox.showwarning("","matplotlib недоступний."); return

        self._show_pca_results(obj_names, var_names, eigenvalues, eigenvectors,
                               explained, scores, n_comp, min_c)

    # ── Вікно результатів: бокове меню + окремі панелі ────────
    def _show_pca_results(self, obj_names, var_names, eigenvalues,
                          eigenvectors, explained, scores, n_comp, min_c):
        win = tk.Toplevel(self.win)
        win.title("PCA — Результати")
        win.geometry("1200x860"); set_icon(win)
        self._pca_win = win

        self._pca_data = dict(obj_names=obj_names, var_names=var_names,
                               eigenvalues=eigenvalues, eigenvectors=eigenvectors,
                               explained=explained, scores=scores,
                               n_comp=n_comp, min_c=min_c)
        self._pca_figs = {}

        main = tk.Frame(win); main.pack(fill=tk.BOTH, expand=True)
        sidebar = tk.Frame(main, width=210, bg="#2c3e50")
        sidebar.pack(side=tk.LEFT, fill=tk.Y); sidebar.pack_propagate(False)
        content = tk.Frame(main); content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(sidebar, text="PCA", bg="#2c3e50", fg="#ecf0f1",
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

        rpt_frame    = tk.Frame(content)
        scree_frame  = tk.Frame(content)
        biplot_frame = tk.Frame(content)
        heat_frame   = tk.Frame(content)

        b_rpt    = _sidebar_btn("📄 Звіт (таблиці)", "Компоненти й навантаження")
        b_scree  = _sidebar_btn("📉 Scree plot",     "Пояснена дисперсія")
        b_biplot = _sidebar_btn("🎯 Biplot",         "ГК1 × ГК2")
        b_heat   = _sidebar_btn("🔥 Навантаження",    "Теплова карта")

        b_rpt.configure(   command=lambda: _show_panel(rpt_frame, b_rpt))
        b_scree.configure( command=lambda: _show_panel(scree_frame, b_scree))
        b_biplot.configure(command=lambda: _show_panel(biplot_frame, b_biplot))
        b_heat.configure(  command=lambda: _show_panel(heat_frame, b_heat))

        self._pca_build_report(rpt_frame)
        self._pca_build_scree(scree_frame)
        self._pca_build_biplot(biplot_frame)
        self._pca_build_heat(heat_frame)

        _show_panel(rpt_frame, b_rpt)

    # ── Панель 1: Звіт (таблиці компонент і навантажень) ──────
    def _pca_build_report(self, frame):
        for w in frame.winfo_children(): w.destroy()
        d = self._pca_data
        eigenvalues = d["eigenvalues"]; explained = d["explained"]; n_comp = d["n_comp"]
        eigenvectors = d["eigenvectors"]; var_names = d["var_names"]; min_c = d["min_c"]

        tb = tk.Frame(frame, padx=6, pady=5); tb.pack(fill=tk.X)
        buf = []
        def _copy_tables():
            self._pca_win.clipboard_clear()
            self._pca_win.clipboard_append("\n".join(buf))
            messagebox.showinfo("","Таблиці скопійовано у буфер.\nВставте у Word/Excel через Ctrl+V.")
        tk.Button(tb, text="📋 Копіювати таблиці", font=("Times New Roman",11),
                  command=_copy_tables).pack(side=tk.LEFT, padx=4)

        outer = tk.Frame(frame); outer.pack(fill=tk.BOTH, expand=True)
        vsb = ttk.Scrollbar(outer, orient="vertical"); vsb.pack(side=tk.RIGHT, fill=tk.Y)
        cv = tk.Canvas(outer, yscrollcommand=vsb.set, highlightthickness=0)
        cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.config(command=cv.yview)
        body = tk.Frame(cv); cv.create_window((0,0), window=body, anchor="nw")
        def _mw(e): cv.yview_scroll(int(-1*(e.delta/120)), "units")
        body.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))

        def _txt(s):
            tk.Label(body, text=s, font=("Times New Roman",11), fg="#444",
                     justify="left", anchor="w", wraplength=1100
                     ).pack(fill=tk.X, padx=12, pady=(8,2))
        def _table(headers, rows):
            frm_t, tv = make_tv(body, headers, rows)
            frm_t.pack(fill=tk.X, padx=8, pady=(2,10))
            buf.append("\t".join(str(h) for h in headers))
            for row in rows:
                buf.append("\t".join(str(v) for v in row))
            buf.append("")

        _txt("ГК = Головна компонента — «узагальнений показник», що об'єднує кілька "
             "вихідних змінних. ГК1 пояснює найбільше варіації.")
        summary_rows = [[f"ГК{i+1}", fmt(eigenvalues[i],4), fmt(explained[i],2),
                         fmt(float(np.sum(explained[:i+1])),2),
                         "✓ включити" if eigenvalues[i] >= 1.0 else "розглянути"]
                        for i in range(n_comp)]
        _table(["Компонент","Власне значення (λ)","% дисперсії",
                "Кумулятивний %","Критерій Кайзера (λ≥1)"], summary_rows)

        _txt("Навантаження (loadings): кореляція показника з кожною ГК. "
             "|Навантаження| > 0.5 — значуща роль показника у цій компоненті.")
        n_show2 = min(6, n_comp)
        load_headers = ["Показник"] + [f"ГК{i+1}" for i in range(n_show2)]
        load_rows = []
        for j in range(min_c):
            nm_j = var_names[j] if j < len(var_names) else f"П{j+1}"
            load_rows.append([nm_j] + [fmt(eigenvectors[j,k],4) for k in range(n_show2)])
        _table(load_headers, load_rows)

        self._pca_bind_wheel(cv, _mw)
        self._pca_bind_wheel(body, _mw)

    # ── Панель 2: Scree plot ───────────────────────────────────
    def _pca_build_scree(self, frame):
        for w in frame.winfo_children(): w.destroy()
        d = self._pca_data; gs = self._pca_gs
        def _rebuild(): self._pca_build_scree(frame)
        def _settings():
            self._pca_settings_dialog([
                ("Шрифт:", "font_family", "combo",
                 ["Times New Roman","Arial","Calibri","Georgia"]),
                ("Розмір шрифту:", "font_size", "spin", (6,18)),
                ("Колір стовпців:", "bar_color", "color", None),
                ("Колір кривої кумул.%:", "cum_color", "color", None),
            ], _rebuild, title="Налаштування — Scree plot")
        self._pca_tab_toolbar(frame, "scree", _rebuild, _settings)
        plot_f = tk.Frame(frame); plot_f.pack(fill=tk.BOTH, expand=True)

        ff = gs["font_family"]; fz = gs["font_size"]
        bc = gs["bar_color"];   cc = gs["cum_color"]
        eigenvalues = d["eigenvalues"]; explained = d["explained"]; n_comp = d["n_comp"]

        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        ax.bar(range(1, n_comp+1), explained[:n_comp], color=bc, alpha=0.8)
        ax.plot(range(1, n_comp+1), np.cumsum(explained[:n_comp]),
                "o-", color=cc, markersize=5)
        ax.set_xlabel("ГК", fontsize=fz, fontfamily=ff)
        ax.set_ylabel("Пояснена дисперсія (%)", fontsize=fz, fontfamily=ff)
        ax.set_title("Графік відсіювання (Scree)", fontsize=fz+1, fontfamily=ff)
        ax.axhline(80, color="gray", lw=0.8, ls="--")
        ax.yaxis.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.tight_layout()
        self._pca_figs["scree"] = fig
        embed_figure(fig, plot_f)

    # ── Панель 3: Biplot (ГК1 × ГК2) ───────────────────────────
    def _pca_build_biplot(self, frame):
        for w in frame.winfo_children(): w.destroy()
        d = self._pca_data; gs = self._pca_gs
        def _rebuild(): self._pca_build_biplot(frame)
        def _settings():
            self._pca_settings_dialog([
                ("Шрифт:", "font_family", "combo",
                 ["Times New Roman","Arial","Calibri","Georgia"]),
                ("Розмір шрифту:", "font_size", "spin", (6,18)),
                ("Розмір точок:", "point_size", "spin", (5,80)),
                ("Довжина стрілок\n(не впливає на напрямок — осі завжди пропорційні):",
                 "arrow_scale", "scale", (0.2,1.5)),
                ("Колір точок:", "point_color", "color", None),
                ("Колір стрілок:", "arrow_color", "color", None),
                ("Підписи об'єктів:", "annotate_obj", "check", None),
                ("Підписи змінних:", "annotate_var", "check", None),
            ], _rebuild, title="Налаштування — Biplot")
        self._pca_tab_toolbar(frame, "biplot", _rebuild, _settings)
        plot_f = tk.Frame(frame); plot_f.pack(fill=tk.BOTH, expand=True)

        ff = gs["font_family"]; fz = gs["font_size"]
        pc = gs["point_color"]; ac = gs["arrow_color"]
        ps = gs["point_size"];  sc = gs["arrow_scale"]
        obj_names = d["obj_names"]; var_names = d["var_names"]
        eigenvectors = d["eigenvectors"]; explained = d["explained"]
        scores = d["scores"]; n_comp = d["n_comp"]; min_c = d["min_c"]

        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        ax.scatter(scores[:,0], scores[:,1], s=ps, color=pc, zorder=3,
                   edgecolors="white", linewidths=0.5)
        if gs["annotate_obj"]:
            for i, nm in enumerate(obj_names[:len(scores)]):
                ann = ax.annotate(nm, xy=(scores[i,0], scores[i,1]),
                            xytext=(7, 7), textcoords="offset points",
                            fontsize=max(6, fz-1), alpha=0.9, fontfamily=ff,
                            arrowprops=dict(arrowstyle="-", color="#999",
                                           alpha=0.6, lw=0.6, shrinkA=0, shrinkB=3))
                ann.draggable(True)
        max_s = max(np.max(np.abs(scores[:,0])), np.max(np.abs(scores[:,1])), 1e-6)
        for j in range(min_c):
            lx = eigenvectors[j,0]*max_s*sc; ly = eigenvectors[j,1]*max_s*sc
            ax.annotate("", xy=(lx,ly), xytext=(0,0),
                        arrowprops=dict(arrowstyle="->", color=ac, lw=1.3))
            if gs["annotate_var"]:
                nm_j = var_names[j] if j < len(var_names) else f"П{j+1}"
                ann_v = ax.annotate(nm_j, xy=(lx,ly),
                            xytext=(9, 9), textcoords="offset points",
                            fontsize=max(6, fz-1), color=ac, fontfamily=ff,
                            arrowprops=dict(arrowstyle="-", color=ac,
                                           alpha=0.5, lw=0.6, shrinkA=0, shrinkB=3))
                ann_v.draggable(True)
        ax.axhline(0, color="#888", lw=0.5); ax.axvline(0, color="#888", lw=0.5)
        ax.set_xlabel(f"ГК1 ({fmt(explained[0],1)}%)", fontsize=fz, fontfamily=ff)
        ax.set_ylabel(f"ГК2 ({fmt(explained[1],1)}%)" if n_comp>1 else "ГК2",
                      fontsize=fz, fontfamily=ff)
        ax.set_title("Biplot (ГК1 × ГК2)", fontsize=fz+1, fontfamily=ff)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        # Однаковий масштаб по обох осях: без цього довжина стрілок (і взаємне
        # розтягування осей matplotlib) візуально «повертає» вектори, хоча
        # координати навантажень не змінюються — лише довжина.
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()
        self._pca_figs["biplot"] = fig
        embed_figure(fig, plot_f)

    # ── Панель 4: Теплова карта навантажень ────────────────────
    def _pca_build_heat(self, frame):
        for w in frame.winfo_children(): w.destroy()
        d = self._pca_data; gs = self._pca_gs
        def _rebuild(): self._pca_build_heat(frame)
        def _settings():
            self._pca_settings_dialog([
                ("Шрифт:", "font_family", "combo",
                 ["Times New Roman","Arial","Calibri","Georgia"]),
                ("Розмір шрифту:", "font_size", "spin", (6,18)),
                ("Палітра:", "heatmap_cmap", "combo",
                 ["RdYlGn","coolwarm","RdBu","viridis","plasma"]),
            ], _rebuild, title="Налаштування — Навантаження")
        self._pca_tab_toolbar(frame, "heat", _rebuild, _settings)
        plot_f = tk.Frame(frame); plot_f.pack(fill=tk.BOTH, expand=True)

        ff = gs["font_family"]; fz = gs["font_size"]
        eigenvectors = d["eigenvectors"]; var_names = d["var_names"]
        n_comp = d["n_comp"]; min_c = d["min_c"]

        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        n_show = min(4, n_comp)
        load_mat = eigenvectors[:, :n_show]
        cmap_ = get_cmap_safe(gs["heatmap_cmap"])
        im = ax.imshow(load_mat, cmap=cmap_, vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n_show))
        ax.set_xticklabels([f"ГК{i+1}" for i in range(n_show)], fontsize=fz, fontfamily=ff)
        ax.set_yticks(range(min_c))
        ax.set_yticklabels(var_names[:min_c] if var_names else [f"П{j+1}" for j in range(min_c)],
                            fontsize=fz, fontfamily=ff)
        ax.set_title("Навантаження факторів", fontsize=fz+1, fontfamily=ff)
        for i in range(min_c):
            for j in range(n_show):
                ax.text(j, i, fmt(load_mat[i,j],2),
                        ha="center", va="center", fontsize=max(6, fz-1), fontfamily=ff)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        self._pca_figs["heat"] = fig
        embed_figure(fig, plot_f)




# ═══════════════════════════════════════════════════════════════
# REPEATED MEASURES ANOVA
# ═══════════════════════════════════════════════════════════════
