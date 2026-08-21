
# sad_scheme_constructor.py — Конструктор багатофакторної схеми досліду
# -*- coding: utf-8 -*-
from sad_common import *
from sad_homogeneous import (HPPlant, HPPlotBuilder, hp_apply_design,
    HPResultsViewMixin, HP_DESIGN_LABELS,
    HP_ROLE_RECORDED, HP_ROLE_GUARD_EDGE, HP_ROLE_GUARD_REP,
    HP_ROLE_UNASSIGNED, HP_ROLE_DEAD, HP_ROLE_POLLINIZER, HP_ROLE_EXTRA,
    HP_ROLE_EXCLUDED_CV)
import itertools, re

# ═══════════════════════════════════════════════════════════════
# КОНСТРУКТОР СХЕМИ — вільна багатофакторна схема досліду
# ═══════════════════════════════════════════════════════════════
FACTOR_LETTERS = "ABCDEFGHIJ"

class SchemeConstructorWindow(HPResultsViewMixin):
    """
    Окремий інструмент від «Планування за однорідністю» — той алгоритм
    добре розв'язує ОДНОФАКТОРНЕ розміщення за вихідними даними рослин,
    але для 2-3-4-факторних дослідів автоматична генерація фізичного
    розміщення стає нездійсненною (забагато варіантів дизайну, забагато
    фізичних обмежень поля/саду, які алгоритм наперед не знає).

    Тому тут — вільна для редагування таблиця: користувач сам вписує
    комбінації рівнів факторів у клітинки (короткими кодами на кшталт
    "A2B3"), за бажанням користуючись підказкою-рандомізацією (CRD/RCBD),
    результат якої завжди можна вручну підправити — та протягуванням
    комірки (той самий інструмент, що й в інших таблицях програми).

    Результат зберігається в тому самому форматі .sadp, що й схема з
    модуля планування за однорідністю — тому одразу підключається до
    вже готового польового журналу й зведення даних для аналізу.
    """

    HELP_TEXT = """
КОНСТРУКТОР СХЕМИ ДОСЛІДУ — ІНСТРУКЦІЯ
═══════════════════════════════════════

ДЛЯ ЧОГО ЦЕЙ МОДУЛЬ?
  Коли дослід має 2, 3 чи більше факторів одночасно (наприклад,
  обробка ґрунту × сорт × доза добрива), автоматично згенерувати
  оптимальне фізичне розміщення на полі чи в саду практично
  неможливо — забагато варіантів дизайну і фізичних обмежень.

  Тому тут — вільна таблиця. Ви самі вирішуєте, яка комбінація
  факторів де розташована, за потреби користуючись підказкою.

КРОК 1. ФАКТОРИ
  «📝 Задати фактори» — вкажіть, скільки факторів у досліді, і для
  кожного — назву та рівні. Наприклад:
    Фактор A: Обробка ґрунту → Оранка, Без оранки
    Фактор B: Сорт → Айдаред, Голден, Джонаголд

КРОК 2. РОЗМІРИ ТАБЛИЦІ
  Той самий майстер, що й у плануванні за однорідністю — вказуєте
  кількість рядів і довжину КОЖНОГО окремо (ряди рідко бувають
  однакової довжини в реальному полі чи саду).

КРОК 3. ПОВТОРНІСТЬ КОЖНОГО РЯДУ
  Біля кожного ряду — поле «Повторність». За замовчуванням кожен ряд
  = окрема повторність, але можна об'єднати кілька рядів в одну
  повторність (вписавши їм однаковий номер), якщо блок фізично не
  поміщається в один ряд.

КРОК 4. ЗАПОВНЕННЯ КЛІТИНОК
  Код клітинки об'єднує ВСІ фактори одразу: "A2B3" = рівень 2 фактора
  A і рівень 3 фактора B в одній і тій самій рослині/ділянці.
    • Вписати вручну — і протягнути мишею за кут комірки (як в Excel),
      щоб швидко заповнити однакові значення
    • "🎲 Рандомізувати" — підказка: автоматично розкидає всі комбінації
      по клітинках за обраним дизайном (CRD/RCBD), результат повністю
      редагований після цього
    • "К" — захисна рослина (край ряду), "П" — захисна між
      повтореннями. Вписуються вручну — ви знаєте розташування поля
      краще за будь-який алгоритм.

КРОК 5. ЗБЕРЕЖЕННЯ
  «💾 Зберегти схему» — той самий формат .sadp, що й у плануванні за
  однорідністю. Відкривається так само в польовому журналі.
"""

    def __init__(self, parent, gs=None):
        self._parent = parent
        self.win = tk.Toplevel(parent)
        self.win.title("Конструктор схеми досліду")
        self.win.geometry("1450x900"); set_icon(self.win)
        self.gs = dict(gs) if gs else {}
        self.factor_defs = []      # [{"name":..., "levels":[...]}, ...]
        self.plot_size = 1         # к-сть рослин на ОДНУ повторність кожної комбінації
        self.num_reps = 3          # к-сть повторень на кожну комбінацію
        self.fixed_factor_idx = None   # індекс фактора, вже зафіксованого в наявному саду (не рандомізується)
        self.fixed_level_vars = []     # StringVar на кожен ряд — рівень зафіксованого фактора в цьому ряду
        self._cell_replication = {}    # {(row,pos): rep_num} — перевизначає "Повт." рядка для
                                        # клітинок, сформованих через "за однорідністю"
                                        # (одна повторність може НЕ збігатися з фізичним рядом)
        self._result = None            # заповнюється після "За однорідністю" — для HPResultsViewMixin
        self._cfg = {}
        self.row_lengths = []
        self.rows_n = 0
        self.entries = []
        self.row_labels = []
        self.pos_labels = []
        # (номер повторності більше не окремий стовпець - визначається
        # автоматично під час рандомізації/однорідності або виведенням
        # з послідовних повних наборів комбінацій при збереженні)
        self._table_built = False
        self._build()

    # ─────────────────────────────────────────────────────
    def _build(self):
        rf = ("Times New Roman", 11)
        rb = ("Times New Roman", 11, "bold")

        # ── Завжди доступні дії (не частина послідовності кроків) ──
        top = tk.Frame(self.win, padx=8, pady=6); top.pack(fill=tk.X)
        tk.Button(top, text="📂 Відкрити схему", font=rf,
                  command=self._load_scheme).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="📚 Довідка", bg="#1a4b8c", fg="white", font=rf,
                  command=self._show_help).pack(side=tk.LEFT, padx=4)
        tk.Label(top, text="Виконуйте кроки по порядку зверху вниз — кожен наступний "
                          "стає доступним після завершення попереднього.",
                 font=("Times New Roman",9), fg="#666").pack(side=tk.LEFT, padx=12)

        # ── КРОК 1: Фактори ──────────────────────────────────
        self._step1_frame = tk.LabelFrame(self.win, text="Крок 1 — Фактори досліду",
                                          font=rb, padx=10, pady=8, fg="#1a4b8c")
        self._step1_frame.pack(fill=tk.X, padx=8, pady=(4,2))
        tk.Button(self._step1_frame, text="📝 Задати фактори", bg="#1a4b8c", fg="white",
                  font=rf, command=self._edit_factors).pack(side=tk.LEFT, padx=4)
        self._factors_status = tk.Label(self._step1_frame, text="(фактори не задано)",
                                        font=("Times New Roman",9), fg="#888")
        self._factors_status.pack(side=tk.LEFT, padx=(8,0))

        # ── КРОК 2: Розміри таблиці (заблоковано до Кроку 1) ──
        self._step2_frame = tk.LabelFrame(self.win,
            text="Крок 2 — Розміри таблиці (спочатку виконайте Крок 1)",
            font=rb, padx=10, pady=8, fg="#999")
        self._step2_frame.pack(fill=tk.X, padx=8, pady=2)
        setup_top = tk.Frame(self._step2_frame); setup_top.pack(fill=tk.X)
        tk.Label(setup_top, text="Кількість рядів:", font=rf).pack(side=tk.LEFT)
        self._n_rows_setup_var = tk.StringVar(value="")
        self._n_rows_entry = tk.Entry(setup_top, textvariable=self._n_rows_setup_var,
                                      width=6, font=rf)
        self._n_rows_entry.pack(side=tk.LEFT, padx=6)
        self._rowlen_btn = tk.Button(setup_top, text="Задати довжину кожного ряду →",
                                     font=rf, command=self._build_row_length_inputs)
        self._rowlen_btn.pack(side=tk.LEFT, padx=10)
        self._resize_btn = tk.Button(setup_top, text="🔧 Змінити розміри", font=rf,
                                     command=self._reset_table_size)

        self._rowlen_holder = tk.Frame(self._step2_frame)
        self._rowlen_holder.pack(fill=tk.X, pady=(8,0))
        self._set_step_enabled(self._step2_frame, False)

        # ── КРОК 3: Заповнення таблиці (заблоковано до Кроку 2) ──
        self._step3_frame = tk.LabelFrame(self.win,
            text="Крок 3 — Заповнення таблиці (спочатку виконайте Крок 2)",
            font=rb, padx=10, pady=8, fg="#999")
        self._step3_frame.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(self._step3_frame,
                 text="У клітинках таблиці нижче можна вручну позначити «-» (випад), "
                      "«+» (запилювач) чи вписати вихідний вимір (число) — це "
                      "враховується автоматично при заповненні комбінацій.",
                 font=("Times New Roman",9), fg="#666", justify="left", wraplength=1000
                 ).pack(anchor="w")
        fill_row = tk.Frame(self._step3_frame); fill_row.pack(fill=tk.X, pady=(6,0))
        self._fill_btn = tk.Button(fill_row, text="➡ Заповнити комбінації факторів",
                                   bg="#8c5a1a", fg="white", font=rf,
                                   command=self._open_fill_dialog)
        self._fill_btn.pack(side=tk.LEFT, padx=4)
        self._paste_btn = tk.Button(fill_row, text="Вставити з буфера", font=rf,
                                    command=self._paste)
        self._paste_btn.pack(side=tk.LEFT, padx=4)

        self._pending_action = None
        self._exec_row = tk.Frame(self._step3_frame)
        self._exec_status_lbl = tk.Label(self._exec_row, text="", font=("Times New Roman",9),
                                         fg="#8c5a1a", justify="left", wraplength=700)
        self._exec_status_lbl.pack(side=tk.LEFT, padx=(0,10))
        self._exec_btn = tk.Button(self._exec_row, text="▶ Виконати формування",
                                   bg="#1a6b1a", fg="white", font=rf,
                                   command=self._execute_pending_action)
        self._exec_btn.pack(side=tk.LEFT)
        self._set_step_enabled(self._step3_frame, False)

        # ── КРОК 4: Зберегти (заблоковано до Кроку 3) ──────────
        self._step4_frame = tk.LabelFrame(self.win,
            text="Крок 4 — Зберегти схему (спочатку виконайте Крок 3)",
            font=rb, padx=10, pady=8, fg="#999")
        self._step4_frame.pack(fill=tk.X, padx=8, pady=(2,4))
        self._save_btn = tk.Button(self._step4_frame, text="💾 Зберегти схему",
                                   bg="#1a6b1a", fg="white", font=rf,
                                   command=self._save_scheme)
        self._save_btn.pack(side=tk.LEFT, padx=4)
        self._set_step_enabled(self._step4_frame, False)

        # ── Легенда факторів ────────────────────────────────
        self._legend_f = tk.Frame(self.win, bg="#eef3f8", padx=8, pady=6)
        self._legend_f.pack(fill=tk.X)

        # ── Область таблиці ─────────────────────────────────
        tbl_area = tk.Frame(self.win); tbl_area.pack(fill=tk.BOTH, expand=True, padx=8, pady=(2,4))
        tbl_area.grid_rowconfigure(0, weight=1)
        tbl_area.grid_columnconfigure(0, weight=1)
        self._canvas = tk.Canvas(tbl_area)
        self._canvas.grid(row=0, column=0, sticky="nsew")
        sb_v = ttk.Scrollbar(tbl_area, orient="vertical", command=self._canvas.yview)
        sb_v.grid(row=0, column=1, sticky="ns")
        sb_h = ttk.Scrollbar(tbl_area, orient="horizontal", command=self._canvas.xview)
        sb_h.grid(row=1, column=0, sticky="ew")
        self._canvas.configure(yscrollcommand=sb_v.set, xscrollcommand=sb_h.set)
        self.inner = tk.Frame(self._canvas)
        self._canvas.create_window((0,0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>",
                        lambda e: self._canvas.config(scrollregion=self._canvas.bbox("all")))
        self.win.bind("<MouseWheel>",
                      lambda e: self._canvas.yview_scroll(int(-1*(e.delta/120)),"units"))

    def _set_step_enabled(self, frame, enabled):
        """Вмикає/вимикає всі інтерактивні віджети всередині кроку-рамки
        і візуально притемнює її заголовок, коли крок ще недоступний."""
        state = tk.NORMAL if enabled else tk.DISABLED
        def _walk(w):
            for c in w.winfo_children():
                if isinstance(c, (tk.Button, tk.Entry, tk.Spinbox, ttk.Combobox, tk.Radiobutton)):
                    try: c.configure(state=state)
                    except tk.TclError: pass
                _walk(c)
        _walk(frame)
        frame.configure(fg="#1a4b8c" if enabled else "#999")

    def _unlock_step2(self):
        base_txt = "Крок 2 — Розміри таблиці"
        self._step2_frame.configure(text=base_txt)
        self._set_step_enabled(self._step2_frame, True)

    def _unlock_step3(self):
        base_txt = "Крок 3 — Заповнення таблиці"
        self._step3_frame.configure(text=base_txt)
        self._set_step_enabled(self._step3_frame, True)
        self._unlock_step4()

    def _unlock_step4(self):
        base_txt = "Крок 4 — Зберегти схему"
        self._step4_frame.configure(text=base_txt)
        self._set_step_enabled(self._step4_frame, True)

    def _set_pending_action(self, action, status_text):
        """Зберігає налаштовану дію (не виконуючи одразу) і показує кнопку
        «▶ Виконати» — так вікно параметрів можна закрити й спокійно піти
        позначати випади/запилювачі в таблиці, а не робити це наосліп під
        модальним вікном."""
        self._pending_action = action
        self._exec_status_lbl.configure(
            text=f"Готово до виконання: {status_text}\n"
                 f"Позначте «-»/«+» в таблиці нижче (за потреби), тоді натисніть «▶ Виконати».")
        self._exec_row.pack(fill=tk.X, pady=(8,0))

    def _execute_pending_action(self):
        if self._pending_action is None: return
        action = self._pending_action
        kind = action["type"]
        if kind == "auto" and action.get("use_cv"):
            n_values = 0
            for row in self.entries:
                for e in row:
                    txt = e.get().strip()
                    if txt and not self._is_guard_text(txt):
                        try: float(txt.replace(",",".")); n_values += 1
                        except ValueError: pass
            if n_values == 0:
                messagebox.showwarning("Немає вихідних вимірів",
                    "Позначено «врахувати вихідні виміри», але в таблиці немає "
                    "жодного числового виміру. Впишіть виміри в клітинки й "
                    "натисніть «▶ Виконати» знову, або скасуйте.")
                return  # налаштування лишаються збереженими, кнопка не ховається

        self._pending_action = None
        self._exec_row.pack_forget()
        if kind == "auto":
            self._run_homogeneous_selection(action["eg"], action["rg"], self.num_reps,
                                            action["cv_thr"], action["design"])
        elif kind == "split":
            self._randomize_split_plot(action["main_idx"])
        elif kind == "fixed":
            self._randomize_with_fixed_factor()

    def _open_fill_dialog(self):
        """Крок 3, головна дія — один-єдиний вибір способу заповнення
        замість кількох окремих кнопок, які раніше висіли одночасно."""
        dlg = tk.Toplevel(self.win); dlg.title("Заповнити комбінації факторів")
        dlg.resizable(False, False); set_icon(dlg); dlg.grab_set()
        rf = ("Times New Roman",11)
        frm = tk.Frame(dlg, padx=18, pady=16); frm.pack()
        tk.Label(frm, text="Як заповнити таблицю комбінаціями факторів?",
                 font=("Times New Roman",12,"bold")).pack(anchor="w", pady=(0,4))
        if self.fixed_factor_idx is not None:
            fname = self.factor_defs[self.fixed_factor_idx]["name"]
            tk.Label(frm, text=f"На кроці 1 фактор «{fname}» позначено зафіксованим "
                              f"у наявному саду — тому рекомендований варіант нижче "
                              f"позначено окремо.",
                     font=("Times New Roman",9), fg="#8c1a4a", justify="left", wraplength=440
                     ).pack(anchor="w", pady=(0,8))

        def _pick_manual():
            dlg.destroy()
            messagebox.showinfo("Вручну",
                "Впишіть код комбінації (наприклад, «A2B3») у потрібні клітинки "
                "таблиці нижче. Протягніть за кут комірки мишею, щоб швидко "
                "скопіювати значення на сусідні клітинки — так само, як в Excel.")

        def _pick_auto():
            dlg.destroy(); self._open_auto_dialog()

        def _pick_split():
            dlg.destroy(); self._open_split_plot_dialog()

        def _pick_fixed():
            dlg.destroy(); self._run_fixed_factor_randomize()

        opts = []
        if self.fixed_factor_idx is not None:
            fname = self.factor_defs[self.fixed_factor_idx]["name"]
            opts.append(("🔒 Існуючий сад", f"Рекомендовано: «{fname}» вже фізично "
             f"закріплено по рядах (крок 2) і не рандомізується — рандомізується "
             f"лише решта факторів.", _pick_fixed, "#8c1a4a"))
        opts += [
            ("✍ Вручну", "Самостійно вписуєте код кожної комбінації й "
             "протягуєте комірки мишею.", _pick_manual, "#555"),
            ("🎲 Автоматично", "CRD/RCBD/Латинський квадрат — розміри захисних зон "
             "і спосіб рандомізації задаєте одразу; повторності формуються "
             "РАЗОМ із варіантами. За бажанням враховує вихідні виміри для "
             "однорідності (CV%).", _pick_auto, "#8c5a1a"),
        ]
        if len(self.factor_defs) >= 2:
            opts.append(("⊞ Split-plot", "Головний фактор — на великих ділянках, "
             "але його рівень ВСЕ ОДНО розподіляється випадково (на відміну від "
             "«Існуючий сад», де рівень фіксований наперед).",
             _pick_split, "#6b4a8c"))

        for label, desc, cmd, color in opts:
            row = tk.Frame(frm); row.pack(fill=tk.X, pady=5)
            tk.Button(row, text=label, bg=color, fg="white", font=("Times New Roman",11,"bold"),
                      width=14, command=cmd).pack(side=tk.LEFT)
            tk.Label(row, text=desc, font=("Times New Roman",9), fg="#666",
                     justify="left", wraplength=340, anchor="w").pack(side=tk.LEFT, padx=10)
        tk.Button(frm, text="Скасувати", font=rf, command=dlg.destroy).pack(pady=(10,0))
        center_win(dlg)

    def _show_help(self):
        win = tk.Toplevel(self.win); win.title("Довідка — Конструктор схеми")
        win.geometry("720x680"); set_icon(win)
        txt = tk.Text(win, wrap="word", font=("Times New Roman",11), padx=10, pady=10)
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert("1.0", self.HELP_TEXT.strip()); txt.configure(state="disabled")
        tk.Button(win, text="Закрити", command=win.destroy,
                  font=("Times New Roman",11)).pack(pady=6)

    # ── Крок 1: фактори ─────────────────────────────────────
    def _edit_factors(self):
        dlg = tk.Toplevel(self.win); dlg.title("Фактори досліду")
        dlg.geometry("640x600"); dlg.minsize(560, 400)
        dlg.resizable(True, True); set_icon(dlg); dlg.grab_set()
        rf = ("Times New Roman",11)

        top_f = tk.Frame(dlg, padx=14, pady=10); top_f.pack(fill=tk.X)
        tk.Label(top_f, text="Кількість факторів:", font=rf).grid(row=0, column=0, sticky="w")
        n_var = tk.StringVar(value=str(max(1, len(self.factor_defs))))
        tk.Spinbox(top_f, from_=1, to=len(FACTOR_LETTERS), textvariable=n_var,
                   width=4, font=rf).grid(row=0, column=1, sticky="w", padx=6)
        tk.Label(top_f, text="Кількість повторностей у варіанті:", font=rf).grid(
            row=1, column=0, sticky="w", pady=(6,0))
        num_reps_var = tk.StringVar(value=str(getattr(self, "num_reps", 3)))
        tk.Spinbox(top_f, from_=2, to=50, textvariable=num_reps_var,
                   width=4, font=rf).grid(row=1, column=1, sticky="w", padx=6, pady=(6,0))
        tk.Label(top_f, text="Кількість рослин у повторності:", font=rf).grid(
            row=2, column=0, sticky="w", pady=(6,0))
        plot_size_var = tk.StringVar(value=str(self.plot_size))
        tk.Spinbox(top_f, from_=1, to=50, textvariable=plot_size_var,
                   width=4, font=rf).grid(row=2, column=1, sticky="w", padx=6, pady=(6,0))

        tk.Label(dlg,
                 text="Якщо дослід закладається у ВЖЕ ІСНУЮЧИЙ сад, де один фактор "
                      "фізично закріплений за рядами і його не можна змінити "
                      "(наприклад, сорти вже посаджені по окремих рядах) — позначте "
                      "цей фактор нижче як «зафіксований». Такий фактор НЕ "
                      "рандомізується — ви самі вкажете, який ряд якому рівню "
                      "відповідає (крок 2). Рандомізуються лише інші фактори.",
                 font=("Times New Roman",9), fg="#666", justify="left", wraplength=600
                 ).pack(fill=tk.X, padx=14, pady=(0,8))

        # ── Прокручувана область — Зберегти/Скасувати завжди видимі внизу,
        # незалежно від того, скільки факторів (навіть усі 10) ─────
        bf = tk.Frame(dlg); bf.pack(side=tk.BOTTOM, pady=10)

        mid = tk.Frame(dlg); mid.pack(fill=tk.BOTH, expand=True, padx=14)
        canvas = tk.Canvas(mid, highlightthickness=0)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb = ttk.Scrollbar(mid, orient="vertical", command=canvas.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.configure(yscrollcommand=vsb.set)
        rows_holder = tk.Frame(canvas)
        canvas_win = canvas.create_window((0,0), window=rows_holder, anchor="nw")
        rows_holder.bind("<Configure>",
                         lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_win, width=e.width))
        dlg.bind("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)),"units"))

        name_vars, level_vars = [], []
        fixed_choice = tk.IntVar(value=self.fixed_factor_idx if self.fixed_factor_idx is not None else -1)

        def _rebuild_rows():
            for w in rows_holder.winfo_children(): w.destroy()
            name_vars.clear(); level_vars.clear()
            try: n = int(n_var.get())
            except ValueError: n = 1
            for i in range(n):
                letter = FACTOR_LETTERS[i]
                fr = tk.LabelFrame(rows_holder, text=f"Фактор {letter}",
                                   font=("Times New Roman",10,"bold"), padx=8, pady=6)
                fr.pack(fill=tk.X, pady=4)
                tk.Label(fr, text="Назва:", font=rf).grid(row=0, column=0, sticky="w")
                default_name = (self.factor_defs[i]["name"]
                                if i < len(self.factor_defs) else "")
                nv = tk.StringVar(value=default_name)
                tk.Entry(fr, textvariable=nv, width=24, font=rf).grid(
                    row=0, column=1, sticky="w", padx=6)
                tk.Radiobutton(fr, text="🔒 Зафіксований у наявному саду",
                              variable=fixed_choice, value=i, font=("Times New Roman",9)
                              ).grid(row=0, column=2, sticky="w", padx=(14,0))
                tk.Label(fr, text="Рівні (через кому):", font=rf).grid(
                    row=1, column=0, sticky="w", pady=(4,0))
                default_levels = (", ".join(self.factor_defs[i]["levels"])
                                  if i < len(self.factor_defs) else "")
                lv = tk.StringVar(value=default_levels)
                tk.Entry(fr, textvariable=lv, width=40, font=rf).grid(
                    row=1, column=1, columnspan=2, sticky="w", padx=6, pady=(4,0))
                name_vars.append(nv); level_vars.append(lv)
            tk.Radiobutton(rows_holder, text="Жоден фактор не зафіксований (усі рандомізуються)",
                          variable=fixed_choice, value=-1, font=("Times New Roman",10)
                          ).pack(anchor="w", pady=(6,0))

        n_var.trace_add("write", lambda *a: _rebuild_rows())
        _rebuild_rows()

        def _save():
            defs = []
            for nv, lv in zip(name_vars, level_vars):
                nm = nv.get().strip()
                levels = [x.strip() for x in lv.get().split(",") if x.strip()]
                if not nm or len(levels) < 2:
                    messagebox.showwarning("",
                        "Кожен фактор потребує назви й щонайменше 2 рівнів.",
                        parent=dlg); return
                defs.append({"name": nm, "levels": levels})
            self.factor_defs = defs
            try: self.plot_size = max(1, int(plot_size_var.get()))
            except ValueError: self.plot_size = 1
            try: self.num_reps = max(2, int(num_reps_var.get()))
            except ValueError: self.num_reps = 3
            self.fixed_factor_idx = fixed_choice.get() if fixed_choice.get() >= 0 else None
            total_combos = 1
            for d in defs: total_combos *= len(d["levels"])
            fixed_txt = ""
            if self.fixed_factor_idx is not None:
                fixed_txt = f"  |  🔒 зафіксовано: {FACTOR_LETTERS[self.fixed_factor_idx]}={defs[self.fixed_factor_idx]['name']}"
            self._factors_status.configure(
                text=f"✓ {len(defs)} факт. × {total_combos} комб. × {self.plot_size} "
                     f"росл./повт. × {self.num_reps} повт. = "
                     f"{total_combos*self.plot_size*self.num_reps} рослин: " +
                     "; ".join(f"{FACTOR_LETTERS[i]}={d['name']}" for i, d in enumerate(defs)) +
                     fixed_txt,
                fg="#1a6b1a")
            self._build_legend()
            needed = total_combos * self.plot_size * self.num_reps
            messagebox.showinfo("Крок 1 завершено",
                f"Усього комбінацій факторів: {total_combos}. При {self.plot_size} "
                f"рослинах на повторність і {self.num_reps} повтореннях — "
                f"знадобиться щонайменше {needed} дослідних одиниць "
                f"(рослин/ділянок), не рахуючи захисних зон.\n\n"
                f"Тепер доступний Крок 2 — вкажіть розміри таблиці нижче.",
                parent=dlg)
            dlg.destroy()
            self._unlock_step2()

        tk.Button(bf, text="Зберегти", bg="#1a6b1a", fg="white", font=rf,
                  command=_save).pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text="Скасувати", font=rf, command=dlg.destroy).pack(side=tk.LEFT)
        center_win(dlg)

    def _build_legend(self):
        for w in self._legend_f.winfo_children(): w.destroy()
        if not self.factor_defs: return
        for i, d in enumerate(self.factor_defs):
            letter = FACTOR_LETTERS[i]
            lock = "  🔒 (зафіксовано в саду)" if i == self.fixed_factor_idx else ""
            txt = f"{letter} = {d['name']}{lock}:  " + "  ".join(
                f"{letter}{j+1}={lvl}" for j, lvl in enumerate(d["levels"]))
            tk.Label(self._legend_f, text=txt, bg="#eef3f8", fg="#1a4b8c",
                     font=("Times New Roman",10), anchor="w", justify="left"
                     ).pack(fill=tk.X)

    # ── Крок 2: розміри таблиці (той самий майстер, що й у плануванні) ──
    def _build_row_length_inputs(self):
        try:
            n = int(self._n_rows_setup_var.get())
            if n < 1: raise ValueError
        except ValueError:
            messagebox.showwarning("", "Вкажіть кількість рядів — ціле число ≥ 1."); return
        if n > 60:
            if not messagebox.askyesno("Багато рядів",
                    f"{n} рядів — це багато, таблиця буде дуже довгою. Продовжити?"):
                return
        for w in self._rowlen_holder.winfo_children(): w.destroy()
        self._rowlen_vars = []
        self._rowlen_entries = []
        grid_f = tk.Frame(self._rowlen_holder); grid_f.pack(fill=tk.X)
        rf = ("Times New Roman",10)
        PER_ROW = 6
        for i in range(n):
            r, c = divmod(i, PER_ROW)
            tk.Label(grid_f, text=f"Ряд {i+1}:", font=rf).grid(
                row=r, column=c*2, sticky="w", padx=(0 if c==0 else 10, 2), pady=2)
            v = tk.StringVar(value="")
            e = tk.Entry(grid_f, textvariable=v, width=5, font=rf)
            e.grid(row=r, column=c*2+1, sticky="w", pady=2)
            self._rowlen_vars.append(v)
            self._rowlen_entries.append(e)

        # Автоперехід у наступне поле по Enter чи стрілці вправо/вліво —
        # прискорює введення довжин для великої кількості рядів. Enter на
        # ОСТАННЬОМУ полі одразу "натискає" Побудувати таблицю.
        def _goto(idx):
            if 0 <= idx < len(self._rowlen_entries):
                w = self._rowlen_entries[idx]
                w.focus_set(); w.select_range(0, tk.END)
            return "break"
        def _on_return(idx):
            nxt = idx + 1
            if nxt < len(self._rowlen_entries):
                _goto(nxt)
            else:
                self._build_data_table()
            return "break"
        for i, e in enumerate(self._rowlen_entries):
            e.bind("<Return>", lambda ev, idx=i: _on_return(idx))
            e.bind("<Right>",  lambda ev, idx=i: _goto(idx+1))
            e.bind("<Left>",   lambda ev, idx=i: _goto(idx-1))
        if self._rowlen_entries:
            self._rowlen_entries[0].focus_set()

        btn_f = tk.Frame(self._rowlen_holder); btn_f.pack(fill=tk.X, pady=(10,0))
        tk.Button(btn_f, text="✓ Побудувати таблицю", bg="#1a6b1a", fg="white",
                  font=("Times New Roman",11), command=self._build_data_table
                  ).pack(side=tk.LEFT)

    def _build_data_table(self):
        try:
            lengths = [int(v.get()) for v in self._rowlen_vars]
            if any(L < 1 for L in lengths): raise ValueError
        except ValueError:
            messagebox.showwarning("", "Вкажіть довжину кожного ряду — додатне ціле "
                                       "число, без порожніх полів."); return
        if self._table_built and self.entries:
            if not messagebox.askyesno("Перебудувати таблицю",
                    "У таблиці вже є введені дані — перебудова видалить їх. Продовжити?"):
                return

        for w in self.inner.winfo_children(): w.destroy()
        self.row_lengths = lengths
        self.rows_n = len(lengths)
        max_len = max(lengths)
        has_fixed = self.fixed_factor_idx is not None
        pos_col0 = 2 if has_fixed else 1
        unit_hdr = "Ряд \\ Рослина"

        tk.Label(self.inner, text=unit_hdr, width=11, relief=tk.RIDGE,
                 bg="#444444", fg="white", font=("Times New Roman",10,"bold")
                 ).grid(row=0, column=0, padx=1, pady=1, sticky="nsew")
        if has_fixed:
            fname = self.factor_defs[self.fixed_factor_idx]["name"]
            tk.Label(self.inner, text=f"🔒 {fname}", width=14, relief=tk.RIDGE,
                     bg="#8c1a4a", fg="white", font=("Times New Roman",9,"bold")
                     ).grid(row=0, column=1, padx=1, pady=1, sticky="nsew")
        self.pos_labels = []
        for j in range(max_len):
            lbl = tk.Label(self.inner, text=str(j+1), width=6, relief=tk.RIDGE,
                           bg="#1a4b8c", fg="white", font=("Times New Roman",9,"bold"))
            lbl.grid(row=0, column=pos_col0+j, padx=1, pady=1, sticky="nsew")
            self.pos_labels.append(lbl)

        self.row_labels = []
        self.entries = []
        self.fixed_level_vars = []
        fixed_levels = self.factor_defs[self.fixed_factor_idx]["levels"] if has_fixed else []
        for i, L in enumerate(lengths):
            rl = tk.Label(self.inner, text=f"Ряд {i+1}", width=11, relief=tk.RIDGE,
                         bg="#444444", fg="white", font=("Times New Roman",9,"bold"))
            rl.grid(row=i+1, column=0, padx=1, pady=1, sticky="nsew")
            self.row_labels.append(rl)
            if has_fixed:
                fv = tk.StringVar(value=fixed_levels[i % len(fixed_levels)])
                ttk.Combobox(self.inner, textvariable=fv, values=fixed_levels,
                            state="readonly", width=13, font=("Times New Roman",9)
                            ).grid(row=i+1, column=1, padx=1, pady=1)
                self.fixed_level_vars.append(fv)
            row_e = []
            for j in range(L):
                e = tk.Entry(self.inner, width=6, font=("Times New Roman",10),
                            justify="center")
                e.grid(row=i+1, column=pos_col0+j, padx=1, pady=1)
                e.bind("<FocusIn>", lambda ev: self._scroll_cell_into_view(ev.widget))
                row_e.append(e)
            self.entries.append(row_e)
        _bind_nav(self.entries, self.win)
        _bind_fill_handle(self.entries, self.win)

        self._table_built = True
        self._n_rows_entry.pack_forget()
        self._rowlen_btn.pack_forget()
        for w in self._rowlen_holder.winfo_children(): w.destroy()
        self._resize_btn.pack(side=tk.LEFT, padx=10)
        self._step2_frame.configure(text="Крок 2 — Розміри таблиці ✓ завершено")
        self._unlock_step3()
        if has_fixed:
            messagebox.showinfo("Зафіксований фактор",
                f"Для кожного ряду вкажіть у стовпці «🔒 {fname}», який рівень "
                f"фактора там фізично росте (за замовчуванням підставлено по черзі — "
                f"обов'язково перевірте й виправте відповідно до реального саду). "
                f"Рандомізація потім розкидає лише РЕШТУ факторів у межах кожного ряду.")

    def _reset_table_size(self):
        if self.entries and not messagebox.askyesno("Змінити розміри таблиці",
                "Поточні дані таблиці буде втрачено при перебудові. Продовжити?"):
            return
        self._resize_btn.pack_forget()
        self._n_rows_entry.pack(side=tk.LEFT, padx=6)
        self._rowlen_btn.pack(side=tk.LEFT, padx=10)
        self._n_rows_setup_var.set(str(self.rows_n) if self.rows_n else "")
        self._step2_frame.configure(text="Крок 2 — Розміри таблиці")

    def _scroll_cell_into_view(self, entry_widget):
        """Прокручує канвас (обидва напрямки), щоб клітинка, яка щойно
        отримала фокус, була повністю видима — таблиці бувають ширші за
        вікно, і без цього легко «загубити» активну клітинку за краєм."""
        self._canvas.update_idletasks()
        bbox = self._canvas.bbox("all")
        if not bbox: return
        total_w = bbox[2] - bbox[0]; total_h = bbox[3] - bbox[1]
        if total_w <= 0 or total_h <= 0: return
        ex = entry_widget.winfo_x(); ey = entry_widget.winfo_y()
        ew = entry_widget.winfo_width(); eh = entry_widget.winfo_height()
        view_w = self._canvas.winfo_width(); view_h = self._canvas.winfo_height()
        x0 = self._canvas.canvasx(0); y0 = self._canvas.canvasy(0)
        margin = 20
        if ex < x0 + margin:
            self._canvas.xview_moveto(max(0, ex - margin) / total_w)
        elif ex + ew > x0 + view_w - margin:
            self._canvas.xview_moveto(min(1, (ex + ew + margin - view_w)) / total_w)
        if ey < y0 + margin:
            self._canvas.yview_moveto(max(0, ey - margin) / total_h)
        elif ey + eh > y0 + view_h - margin:
            self._canvas.yview_moveto(min(1, (ey + eh + margin - view_h)) / total_h)

    def _extend_row_to(self, ri, target_len):
        pos_col0 = 2 if self.fixed_factor_idx is not None else 1
        while len(self.entries[ri]) < target_len:
            j = len(self.entries[ri])
            if j >= len(self.pos_labels):
                lbl = tk.Label(self.inner, text=str(j+1), width=6, relief=tk.RIDGE,
                               bg="#1a4b8c", fg="white", font=("Times New Roman",9,"bold"))
                lbl.grid(row=0, column=pos_col0+j, padx=1, pady=1, sticky="nsew")
                self.pos_labels.append(lbl)
            e = tk.Entry(self.inner, width=6, font=("Times New Roman",10), justify="center")
            e.grid(row=ri+1, column=pos_col0+j, padx=1, pady=1)
            e.bind("<FocusIn>", lambda ev: self._scroll_cell_into_view(ev.widget))
            self.entries[ri].append(e)
        if ri < len(self.row_lengths): self.row_lengths[ri] = len(self.entries[ri])

    # ── Вставка з буфера ─────────────────────────────────────
    def _paste(self):
        if not self._table_built:
            messagebox.showinfo("", "Спочатку побудуйте таблицю."); return
        try: data = self.win.clipboard_get()
        except Exception:
            messagebox.showwarning("Буфер порожній",
                "Скопіюйте дані з Excel (Ctrl+C) і спробуйте знову."); return
        if not data.strip(): return
        pos = (0,0); w = self.win.focus_get()
        if isinstance(w, tk.Entry):
            for i, row_ in enumerate(self.entries):
                for j, e in enumerate(row_):
                    if e is w: pos = (i,j); break
        r0, c0 = pos
        for ir, line in enumerate(data.splitlines()):
            if not line.strip(): continue
            ri = r0+ir
            if ri >= len(self.entries): continue
            vals = line.split("\t")
            self._extend_row_to(ri, c0+len(vals))
            for jc, val in enumerate(vals):
                cc = c0+jc
                self.entries[ri][cc].delete(0, tk.END)
                self.entries[ri][cc].insert(0, val.strip())
        _bind_nav(self.entries, self.win)
        _bind_fill_handle(self.entries, self.win)

    # ── Крок 4: рандомізація (підказка, не диктат) ──────────
    def _open_auto_dialog(self):
        """Об'єднаний діалог «Автоматично» — заміняє колишні окремі
        «Рандомізувати» й «За однорідністю»: обидва насправді використо-
        вують той самий механізм (HPPlotBuilder + hp_apply_design), що
        одночасно формує ПОВТОРНОСТІ (не лише варіанти) з дотриманням
        захисних зон — просто «однорідність» додатково враховує вихідні
        виміри для відбору, а звичайна рандомізація — ні.

        Це вікно лише ЗБИРАЄ налаштування й закривається — саме виконання
        відбувається окремою кнопкою «▶ Виконати формування» під таблицею,
        після того як ви позначите випади/запилювачі (і, за потреби,
        виміри) у самій таблиці."""
        if not self.factor_defs:
            messagebox.showwarning("", "Спочатку задайте фактори."); return
        if not self._table_built:
            messagebox.showwarning("", "Спочатку побудуйте таблицю."); return

        total_combos = 1
        for d in self.factor_defs: total_combos *= len(d["levels"])

        dlg = tk.Toplevel(self.win); dlg.title("Автоматичне формування схеми")
        dlg.geometry("580x480"); dlg.minsize(520,380)
        dlg.resizable(True, True); set_icon(dlg); dlg.grab_set()
        rf = ("Times New Roman",11)

        tk.Label(dlg,
                 text=f"Комбінацій факторів: {total_combos}  ×  {self.plot_size} "
                      f"рослин/повторність  ×  {self.num_reps} повторень (задано "
                      f"на кроці 1)  =  {total_combos*self.plot_size*self.num_reps} "
                      f"рослин на схему.",
                 font=("Times New Roman",10,"bold"), fg="#1a4b8c", justify="left",
                 wraplength=540).pack(fill=tk.X, padx=16, pady=(12,6))

        bf = tk.Frame(dlg); bf.pack(side=tk.BOTTOM, pady=10)

        mid = tk.Frame(dlg); mid.pack(fill=tk.BOTH, expand=True, padx=16)
        canvas = tk.Canvas(mid, highlightthickness=0)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb = ttk.Scrollbar(mid, orient="vertical", command=canvas.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.configure(yscrollcommand=vsb.set)
        frm = tk.Frame(canvas)
        canvas_win = canvas.create_window((0,0), window=frm, anchor="nw")
        frm.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_win, width=e.width))
        dlg.bind("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)),"units"))

        rows_cfg = [
            ("Захисна зона на початку й кінці ряду:", "eg", "1"),
            ("Захисна зона між повторностями:", "rg", "1"),
        ]
        vv = {}
        for i, (lbl, key, dflt) in enumerate(rows_cfg):
            tk.Label(frm, text=lbl, font=rf).grid(row=i, column=0, sticky="w", pady=3)
            v = tk.StringVar(value=dflt)
            tk.Entry(frm, textvariable=v, width=8, font=rf).grid(
                row=i, column=1, sticky="w", padx=8, pady=3)
            vv[key] = v

        r0 = len(rows_cfg)
        tk.Label(frm, text="Спосіб рандомізації:", font=rf).grid(row=r0, column=0, sticky="w", pady=3)
        design_map = {"RCBD (рекомендується)": "rcbd", "CRD": "crd",
                     "Латинський квадрат": "latin"}
        design_disp = tk.StringVar(value=list(design_map.keys())[0])
        ttk.Combobox(frm, textvariable=design_disp, values=list(design_map.keys()),
                     state="readonly", width=22).grid(row=r0, column=1, sticky="w", padx=8)

        tk.Frame(frm, height=1, bg="#ccc").grid(row=r0+1, column=0, columnspan=2,
                                                sticky="ew", pady=10)

        use_cv_v = tk.BooleanVar(value=False)
        cv_extra = tk.Frame(frm)
        trait_name_v = tk.StringVar(value=getattr(self, "_trait_name", ""))
        trait_unit_v = tk.StringVar(value=getattr(self, "_trait_unit", ""))
        cv_thr_v = tk.StringVar(value="15")

        def _build_cv_extra():
            for w in cv_extra.winfo_children(): w.destroy()
            if not use_cv_v.get(): return
            tk.Label(cv_extra, text="Показник:", font=rf).grid(row=0, column=0, sticky="w", pady=3)
            tk.Entry(cv_extra, textvariable=trait_name_v, width=20, font=rf).grid(
                row=0, column=1, sticky="w", padx=8, pady=3)
            tk.Label(cv_extra, text="Одиниця:", font=rf).grid(row=1, column=0, sticky="w", pady=3)
            tk.Entry(cv_extra, textvariable=trait_unit_v, width=10, font=rf).grid(
                row=1, column=1, sticky="w", padx=8, pady=3)
            tk.Label(cv_extra, text="Поріг CV, %:", font=rf).grid(row=2, column=0, sticky="w", pady=3)
            tk.Entry(cv_extra, textvariable=cv_thr_v, width=8, font=rf).grid(
                row=2, column=1, sticky="w", padx=8, pady=3)
            tk.Label(cv_extra, text="Вихідні виміри вписуються в таблицю ПІСЛЯ закриття\n"
                                  "цього вікна, разом із позначенням випадів/запилювачів.",
                     font=("Times New Roman",9), fg="#666", justify="left"
                     ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(4,0))

        tk.Checkbutton(frm, text="Врахувати вихідні виміри показника (відбір за CV%)",
                      variable=use_cv_v, font=rf, command=_build_cv_extra
                      ).grid(row=r0+2, column=0, columnspan=2, sticky="w")
        cv_extra.grid(row=r0+3, column=0, columnspan=2, sticky="w")

        tk.Label(frm,
                 text="Після «Далі» це вікно закриється — позначте «-»/«+» (і виміри, "
                      "якщо потрібні) у таблиці, тоді натисніть «▶ Виконати формування» "
                      "під таблицею.",
                 font=("Times New Roman",9), fg="#8c5a1a", justify="left", wraplength=480
                 ).grid(row=r0+4, column=0, columnspan=2, sticky="w", pady=(10,0))

        def _go():
            try:
                eg = int(vv["eg"].get()); rg = int(vv["rg"].get())
            except ValueError:
                messagebox.showwarning("", "Перевірте числові параметри.", parent=dlg); return
            design = design_map[design_disp.get()]

            use_cv = use_cv_v.get()
            trait_name = trait_name_v.get().strip()
            trait_unit = trait_unit_v.get().strip()
            try: cv_thr = float(cv_thr_v.get()) if use_cv else 10**9
            except ValueError:
                messagebox.showwarning("", "Перевірте поріг CV.", parent=dlg); return
            if use_cv and not trait_name:
                messagebox.showwarning("", "Вкажіть назву показника.", parent=dlg); return

            self._trait_name = trait_name if use_cv else ""
            self._trait_unit = trait_unit if use_cv else ""

            status = (f"«Автоматично» — {design_disp.get()}, захист {eg}/{rg}"
                      + (f", за CV% показника «{trait_name}»" if use_cv
                         else ", без обліку вихідних вимірів"))

            dlg.destroy()
            self._set_pending_action(
                {"type":"auto", "eg":eg, "rg":rg, "design":design,
                 "use_cv":use_cv, "cv_thr":cv_thr},
                status)

        tk.Button(bf, text="Далі →", bg="#1a6b8c", fg="white", font=rf,
                  command=_go).pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text="Скасувати", font=rf, command=dlg.destroy).pack(side=tk.LEFT)
        center_win(dlg)

    def _run_homogeneous_selection(self, edge_guard, rep_guard, num_reps, cv_thr, design):
        total_combos = 1
        for d in self.factor_defs: total_combos *= len(d["levels"])

        plants = []
        for i, row in enumerate(self.entries):
            for j, e in enumerate(row):
                txt = e.get().strip()
                role, status, value, factors = self._parse_cell(txt)
                if role is None:
                    status, value = "ok", None
                p = HPPlant(i+1, j+1, value, status)
                plants.append(p)

        # HPPlotBuilder завжди рахує CV% придатних рослин для звіту, навіть
        # якщо сам поріг практично не обмежує (чиста рандомізація без
        # вихідних вимірів) — тож "ok"-рослини без числового виміру
        # потребують заповнювача 0.0, інакше обчислення впаде на None.
        for p in plants:
            if p.status == "ok" and p.value is None:
                p.value = 0.0

        builder = HPPlotBuilder(plants, self.plot_size, edge_guard, rep_guard, cv_thr,
                                max_iterations=20)
        try:
            result = builder.build()
        except Exception as ex:
            messagebox.showerror("Помилка формування", str(ex)); return

        hp_apply_design(result, design, total_combos, num_reps, seed=None)

        combos = list(itertools.product(
            *[range(1, len(d["levels"])+1) for d in self.factor_defs]))
        for p in result["plants"]:
            if p.role == HP_ROLE_RECORDED and p.variant:
                combo = combos[p.variant - 1]
                p.factors = {self.factor_defs[k]["name"]: lvl
                            for k, lvl in enumerate(combo)}

        # Записуємо результат назад у клітинки — далі можна вручну підправити
        role_txt = {HP_ROLE_GUARD_EDGE: "К", HP_ROLE_GUARD_REP: "П",
                    HP_ROLE_DEAD: "-", HP_ROLE_POLLINIZER: "+", HP_ROLE_EXTRA: "×"}
        self._cell_replication = {}
        counts = {"recorded":0, "guard":0, "dead":0, "pollinizer":0,
                  "excluded_cv":0, "extra":0, "unassigned":0}
        for p in result["plants"]:
            e = self.entries[p.row-1][p.position-1]
            e.delete(0, tk.END)
            if p.role == HP_ROLE_RECORDED and p.factors:
                code = "".join(f"{FACTOR_LETTERS[k]}{p.factors[d['name']]}"
                               for k, d in enumerate(self.factor_defs))
                e.insert(0, code)
                counts["recorded"] += 1
                if p.replication:
                    self._cell_replication[(p.row, p.position)] = p.replication
            elif p.role in role_txt:
                e.insert(0, role_txt[p.role])
                if p.role in (HP_ROLE_GUARD_EDGE, HP_ROLE_GUARD_REP): counts["guard"] += 1
                elif p.role == HP_ROLE_DEAD: counts["dead"] += 1
                elif p.role == HP_ROLE_POLLINIZER: counts["pollinizer"] += 1
                elif p.role == HP_ROLE_EXTRA: counts["extra"] += 1
            elif p.role == HP_ROLE_EXCLUDED_CV:
                counts["excluded_cv"] += 1
            else:
                counts["unassigned"] += 1

        messagebox.showinfo("Готово — результат відбору за однорідністю",
            f"Показник: {getattr(self,'_trait_name','')} "
            f"({getattr(self,'_trait_unit','') or '—'})\n"
            f"CV% фінальний: {result['final_cv_pct']:.2f}   "
            f"Ітерацій: {result['iterations_used']}\n\n"
            f"Облікових (обрано за однорідністю): {counts['recorded']}\n"
            f"Забраковано за варіацією (CV): {counts['excluded_cv']}\n"
            f"Захисних: {counts['guard']}   Випадів: {counts['dead']}   "
            f"Запилювачів: {counts['pollinizer']}\n"
            f"Поза дизайном (залишок): {counts['extra']}\n\n"
            "Результат записано в таблицю — клітинки лишаються повністю "
            "редагованими, перевірте й підправте за потреби.\n\n"
            "Зараз відкриється повний перегляд (карта саду, список "
            "облікових рослин, друкований бланк) — той самий, що й у "
            "плануванні за однорідністю.")

        self._result = result
        self._cfg = {
            "trait_name": getattr(self, "_trait_name", ""),
            "trait_unit": getattr(self, "_trait_unit", ""),
            "design_requested": design, "design_used": design,
        }
        self._show_results()

    # ── Мітки рослин для карти/списку/бланку (hook-методи mixin) ──
    def _plant_map_label(self, p):
        if p.role == HP_ROLE_RECORDED and p.factors:
            return "".join(f"{FACTOR_LETTERS[k]}{p.factors[d['name']]}"
                           for k, d in enumerate(self.factor_defs))
        return {HP_ROLE_GUARD_EDGE:"К", HP_ROLE_GUARD_REP:"П", HP_ROLE_DEAD:"-",
                HP_ROLE_POLLINIZER:"+", HP_ROLE_EXTRA:"×"}.get(p.role, "")

    def _plant_list_label(self, p):
        if not p.factors: return "-"
        code = "".join(f"{FACTOR_LETTERS[k]}{p.factors[d['name']]}"
                       for k, d in enumerate(self.factor_defs))
        names = ", ".join(f"{d['name']}={d['levels'][p.factors[d['name']]-1]}"
                          for d in self.factor_defs)
        return f"{code} ({names})"

    def _plant_form_label(self, p):
        return f"{self._plant_map_label(p)}-П{p.replication}"

    def _map_legend_note_text(self):
        return ('Позначення на клітинках: код комбінації факторів (напр. "A2B3", '
                'де кожна літера — окремий фактор)  •  '
                '"К" — захисна зона, край ряду  •  "П" — захисна зона між повтореннями  •  '
                '"-" — випад/пошкоджена рослина  •  "+" — запилювач  •  '
                '"×" — сформована повторність поза дизайном (залишок, не увійшов у жоден '
                'повний блок — не бере участі в обліку)')

    def _map_legend_decode(self, frame):
        if not self.factor_defs: return
        legend2_f = tk.Frame(frame, bg="#eef3f8", padx=8, pady=6)
        legend2_f.pack(fill=tk.X)
        for i, d in enumerate(self.factor_defs):
            letter = FACTOR_LETTERS[i]
            txt = f"{letter} = {d['name']}:  " + "  ".join(
                f"{letter}{j+1}={lvl}" for j, lvl in enumerate(d["levels"]))
            tk.Label(legend2_f, text=txt, bg="#eef3f8", fg="#1a4b8c",
                     font=("Times New Roman",10), anchor="w", justify="left"
                     ).pack(fill=tk.X)

    def _list_extra_header_lines(self):
        lines = []
        for i, d in enumerate(self.factor_defs):
            letter = FACTOR_LETTERS[i]
            lines.append(f"{letter} = {d['name']}: " + ", ".join(
                f"{letter}{j+1}={lvl}" for j, lvl in enumerate(d["levels"])))
        return lines

    def _open_split_plot_dialog(self):
        if not self.factor_defs:
            messagebox.showwarning("", "Спочатку задайте фактори."); return
        if not self._table_built:
            messagebox.showwarning("", "Спочатку побудуйте таблицю."); return
        if len(self.factor_defs) < 2:
            messagebox.showwarning("", "Split-plot потребує щонайменше 2 фактори."); return

        dlg = tk.Toplevel(self.win); dlg.title("Split-plot"); dlg.resizable(False, False)
        set_icon(dlg); dlg.grab_set()
        rf = ("Times New Roman",11)
        frm = tk.Frame(dlg, padx=16, pady=14); frm.pack()
        tk.Label(frm, text="Головний фактор — на великих суцільних ділянках у межах "
                          "кожного ряду; який САМЕ ряд/ділянка отримає який рівень "
                          "головного фактора — визначається ВИПАДКОВО (це і "
                          "відрізняє split-plot від «Існуючий сад»: там рівень "
                          "фіксований наперед і не рандomізується взагалі). Решта "
                          "факторів рандомізуються окремо всередині кожної ділянки.",
                 font=("Times New Roman",10), fg="#555", justify="left", wraplength=440
                 ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0,10))
        tk.Label(frm, text="Головний фактор (велика ділянка):", font=rf
                 ).grid(row=1, column=0, sticky="w")
        main_factor_v = tk.StringVar(value=FACTOR_LETTERS[0])
        letters = [FACTOR_LETTERS[i] for i in range(len(self.factor_defs))]
        ttk.Combobox(frm, textvariable=main_factor_v, values=letters,
                     state="readonly", width=6).grid(row=1, column=1, sticky="w", padx=8)
        tk.Label(frm, text="Після «Далі» це вікно закриється — за потреби позначте "
                          "«-»/«+» у таблиці, тоді натисніть «▶ Виконати формування».",
                 font=("Times New Roman",9), fg="#8c5a1a", justify="left", wraplength=420
                 ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(10,0))

        def _go():
            idx = FACTOR_LETTERS.index(main_factor_v.get())
            letter = FACTOR_LETTERS[idx]
            fname = self.factor_defs[idx]["name"]
            dlg.destroy()
            self._set_pending_action({"type":"split", "main_idx": idx},
                f"Split-plot — головний фактор {letter}={fname}")
        bf = tk.Frame(frm); bf.grid(row=3, column=0, columnspan=2, pady=(14,0))
        tk.Button(bf, text="Далі →", bg="#8c5a1a", fg="white", font=rf,
                  command=_go).pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text="Скасувати", font=rf, command=dlg.destroy).pack(side=tk.LEFT)
        center_win(dlg)

    def _run_fixed_factor_randomize(self):
        if self.fixed_factor_idx is None:
            messagebox.showwarning("", "Зафіксований фактор не задано (крок 1)."); return
        if not self._table_built:
            messagebox.showwarning("", "Спочатку побудуйте таблицю."); return
        fname = self.factor_defs[self.fixed_factor_idx]["name"]
        n_sub = 1
        for i, d in enumerate(self.factor_defs):
            if i != self.fixed_factor_idx: n_sub *= len(d["levels"])
        self._set_pending_action({"type":"fixed"},
            f"Існуючий сад — «{fname}» не рандомізується (задано по рядах на "
            f"кроці 2), решта факторів ({n_sub} комбінацій) — рандомізуються.")

    def _is_guard_text(self, txt):
        """Клітинки, які рандомізація НЕ повинна чіпати: захисні (К/П),
        випади (-), запилювачі (+). Вихідні числові виміри (для
        однорідності) сюди НЕ входять — вони не заважають рандомізації
        комбінацій, лише враховуються окремо в «За однорідністю»."""
        return txt.strip().upper() in ("К","K","П","P","-","+")

    def _randomize_split_plot(self, main_idx):
        """Головний фактор (main_idx) — на великих суцільних ділянках у межах
        КОЖНОГО РЯДУ (ряд = одна повторність для цілей split-plot); решта
        факторів рандомізуються окремо всередині кожної такої ділянки."""
        import random
        n_main = len(self.factor_defs[main_idx]["levels"])
        sub_idxs = [i for i in range(len(self.factor_defs)) if i != main_idx]
        sub_combos = list(itertools.product(
            *[range(1, len(self.factor_defs[i]["levels"])+1) for i in sub_idxs]))
        n_sub = len(sub_combos)

        def _code(main_lvl, sub_combo):
            parts = {main_idx: main_lvl}
            for idx, lvl in zip(sub_idxs, sub_combo): parts[idx] = lvl
            return "".join(f"{FACTOR_LETTERS[i]}{parts[i]}" for i in range(len(self.factor_defs)))

        skipped_rows = []
        for i, row in enumerate(self.entries):
            cells = [(i,j) for j in range(len(row)) if not self._is_guard_text(row[j].get())]
            if len(cells) < n_sub:
                skipped_rows.append(f"Ряд {i+1}"); continue
            main_levels_order = list(range(1, n_main+1)); random.shuffle(main_levels_order)
            idx_cell = 0
            for seg_i in range(n_main):
                if idx_cell >= len(cells): break
                seg_cells = cells[idx_cell: idx_cell+n_sub]
                idx_cell += n_sub
                main_lvl = main_levels_order[seg_i]
                sub_pool = sub_combos.copy(); random.shuffle(sub_pool)
                for (ri,rj), sub_combo in zip(seg_cells, sub_pool):
                    self.entries[ri][rj].delete(0, tk.END)
                    self.entries[ri][rj].insert(0, _code(main_lvl, sub_combo))
        msg = ("Комбінації розкидано за схемою split-plot (кожен ряд — своя "
               "повторність). Клітинки залишаються повністю редагованими — "
               "перевірте й підправте за потреби.")
        if skipped_rows:
            msg += (f"\n\n⚠ У рядах {', '.join(skipped_rows)} забракло вільних "
                    f"клітинок для жодної повної підділянки — їх пропущено.")
        messagebox.showinfo("Готово", msg)

    def _randomize_with_fixed_factor(self):
        """Фактор self.fixed_factor_idx НЕ рандомізується — його рівень для
        кожного ряду береться зі стовпця «🔒» (вже вказано користувачем,
        відповідно до реального розташування в наявному саду). Рандомізації
        підлягають лише РЕШТА факторів, окремо в межах кожного ряду —
        аналогічно до підділянки в split-plot, тільки «головна ділянка»
        тут задана не алгоритмом, а фізичною реальністю."""
        import random
        fixed_idx = self.fixed_factor_idx
        fixed_levels = self.factor_defs[fixed_idx]["levels"]
        sub_idxs = [i for i in range(len(self.factor_defs)) if i != fixed_idx]
        sub_combos = list(itertools.product(
            *[range(1, len(self.factor_defs[i]["levels"])+1) for i in sub_idxs]))
        n_sub = len(sub_combos)

        def _code(fixed_lvl_num, sub_combo):
            parts = {fixed_idx: fixed_lvl_num}
            for idx, lvl in zip(sub_idxs, sub_combo): parts[idx] = lvl
            return "".join(f"{FACTOR_LETTERS[i]}{parts[i]}" for i in range(len(self.factor_defs)))

        skipped = []
        for i, row in enumerate(self.entries):
            fixed_lvl_name = self.fixed_level_vars[i].get()
            try: fixed_lvl_num = fixed_levels.index(fixed_lvl_name) + 1
            except ValueError: fixed_lvl_num = 1
            eligible_j = [j for j in range(len(row)) if not self._is_guard_text(row[j].get())]
            if len(eligible_j) < n_sub:
                skipped.append(f"Ряд {i+1}"); continue
            reps_needed = math.ceil(len(eligible_j) / n_sub)
            pool = (sub_combos * reps_needed)[:len(eligible_j)]
            random.shuffle(pool)
            for j, sub_combo in zip(eligible_j, pool):
                row[j].delete(0, tk.END)
                row[j].insert(0, _code(fixed_lvl_num, sub_combo))

        msg = ("Решту факторів рандомізовано в межах кожного ряду, узгоджено із "
               "зафіксованим рівнем цього ряду. Клітинки залишаються повністю "
               "редагованими — перевірте й підправте за потреби.")
        if skipped:
            msg += f"\n\n⚠ У рядах {', '.join(skipped)} забракло вільних клітинок."
        messagebox.showinfo("Готово", msg)

    # ── Розбір коду клітинки на фактори ─────────────────────
    def _parse_code(self, text):
        """'A2B3' -> {'Обробка ґрунту': 2, 'Сорт': 3}. Повертає None якщо код
        не розпізнано (і це не порожньо/захисна клітинка)."""
        text = text.strip().upper()
        if not text: return None
        matches = re.findall(r'([A-Z])(\d+)', text)
        if not matches: return None
        result = {}
        for letter, num in matches:
            idx = FACTOR_LETTERS.find(letter)
            if idx < 0 or idx >= len(self.factor_defs): continue
            result[self.factor_defs[idx]["name"]] = int(num)
        return result if len(result) == len(self.factor_defs) else None

    # ── Зберегти / відкрити схему ───────────────────────────
    def _infer_replications(self):
        """Визначає номер повторності для кожної облікової клітинки БЕЗ
        окремого стовпця — рахує послідовні повні набори всіх комбінацій
        факторів у порядку сканування таблиці (ряд за рядом, зліва
        направо), не перетинаючи межу рядів. Кожен повний набір із
        total_combos облікових клітинок — одна повторність."""
        total_combos = 1
        for d in self.factor_defs: total_combos *= len(d["levels"])
        rep_map = {}
        rep_counter = 0
        for i, row in enumerate(self.entries):
            count_in_block = 0
            for j, e in enumerate(row):
                role, status, value, factors = self._parse_cell(e.get().strip())
                if role == HP_ROLE_RECORDED:
                    if count_in_block == 0:
                        rep_counter += 1
                    rep_map[(i+1, j+1)] = rep_counter
                    count_in_block += 1
                    if count_in_block >= total_combos:
                        count_in_block = 0
        return rep_map

    def _parse_cell(self, txt):
        """Розбирає вміст клітинки. Повертає (role, status, value, factors).
        Підтримує: порожньо, К/П (захист), - (випад), + (запилювач),
        число (вихідний вимір для однорідності), код факторів "A2B3"."""
        txt = txt.strip()
        if not txt:
            return HP_ROLE_UNASSIGNED, "ok", None, {}
        up = txt.upper()
        if up in ("К","K"):
            return HP_ROLE_GUARD_EDGE, "ok", None, {}
        if up in ("П","P"):
            return HP_ROLE_GUARD_REP, "ok", None, {}
        if txt == "-":
            return HP_ROLE_DEAD, "dead", None, {}
        if txt == "+":
            return HP_ROLE_POLLINIZER, "pollinizer", None, {}
        if txt == "×":
            return HP_ROLE_EXTRA, "ok", None, {}
        try:
            val = float(txt.replace(",", "."))
            return HP_ROLE_UNASSIGNED, "ok", val, {}
        except ValueError:
            pass
        factors = self._parse_code(txt)
        if factors is not None:
            return HP_ROLE_RECORDED, "ok", None, factors
        return None, None, None, None   # нерозпізнано

    def _save_scheme(self):
        if not self.factor_defs:
            messagebox.showwarning("", "Спочатку задайте фактори."); return
        if not self._table_built:
            messagebox.showwarning("", "Спочатку побудуйте таблицю."); return

        inferred_reps = self._infer_replications()
        plants = []
        bad_cells = []
        for i, row in enumerate(self.entries):
            for j, e in enumerate(row):
                txt = e.get().strip()
                role, status, value, factors = self._parse_cell(txt)
                p = HPPlant(i+1, j+1, value, status or "ok")
                if role is None:
                    bad_cells.append(f"Ряд {i+1}, поз. {j+1}: «{txt}»")
                    p.role = HP_ROLE_UNASSIGNED
                else:
                    p.role = role
                    p.factors = factors
                    if role == HP_ROLE_RECORDED:
                        p.replication = self._cell_replication.get(
                            (i+1, j+1), inferred_reps.get((i+1, j+1), 1))
                plants.append(p)

        if bad_cells:
            preview = "\n".join(bad_cells[:10])
            more = f"\n… і ще {len(bad_cells)-10}" if len(bad_cells) > 10 else ""
            if not messagebox.askyesno("Нерозпізнані клітинки",
                    f"Не вдалось розпізнати код у {len(bad_cells)} клітинках "
                    f"(мають бути коди факторів на кшталт «A2B3», «К», «П», "
                    f"«-», «+», «×», число, або порожні):\n\n{preview}{more}\n\n"
                    "Зберегти попри це (ці клітинки будуть позначені як "
                    "непризначені)?"):
                return

        d = {
            "type": "multi_factor_scheme", "version": APP_VER,
            "cfg": {"trait_name": getattr(self, "_trait_name", ""),
                    "trait_unit": getattr(self, "_trait_unit", ""),
                    "design_used": "custom"},
            "factor_defs": self.factor_defs,
            "variant_names": [],
            "plants": [{"row": p.row, "position": p.position, "value": p.value,
                       "status": p.status, "role": p.role, "plot_id": p.plot_id,
                       "variant": p.variant, "replication": p.replication,
                       "factors": p.factors} for p in plants],
        }
        default_name = "схема_" + "_".join(fd["name"] for fd in self.factor_defs)[:40] + ".sadp"
        path = filedialog.asksaveasfilename(
            parent=self.win, defaultextension=".sadp", initialfile=default_name,
            filetypes=[("SAD схема","*.sadp"),("JSON","*.json")],
            title="Зберегти схему досліду")
        if not path: return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("Збережено",
                f"Схему збережено:\n{path}\n\n"
                "Відкрийте цей файл у модулі «Польовий журнал обліків», щоб "
                "вносити виміряні значення.")
        except Exception as ex:
            messagebox.showerror("Помилка збереження", str(ex))

    def _load_scheme(self):
        path = filedialog.askopenfilename(
            parent=self.win, filetypes=[("SAD схема","*.sadp"),("JSON","*.json")],
            title="Відкрити схему")
        if not path: return
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception as ex:
            messagebox.showerror("Помилка відкриття", str(ex)); return
        if d.get("type") != "multi_factor_scheme":
            messagebox.showwarning("Не той тип файлу",
                "Цей файл не є схемою, збереженою в конструкторі схеми."); return

        self.factor_defs = d.get("factor_defs", [])
        total_combos = 1
        for fd in self.factor_defs: total_combos *= len(fd["levels"])
        self._factors_status.configure(
            text=f"✓ {len(self.factor_defs)} факт., {total_combos} комбінацій",
            fg="#1a6b1a")
        self._build_legend()
        self._unlock_step2()

        plants_by_row = {}
        for pd in d.get("plants", []):
            plants_by_row.setdefault(pd["row"], []).append(pd)
        rows_sorted = sorted(plants_by_row.keys())
        lengths = [len(plants_by_row[r]) for r in rows_sorted]

        for w in self.inner.winfo_children(): w.destroy()
        self.row_lengths = []
        self.rows_n = 0
        self.entries = []; self.row_labels = []; self.pos_labels = []
        self._rowlen_vars = [tk.StringVar(value=str(L)) for L in lengths]
        self._build_data_table()

        self._cell_replication = {}
        for ri, r in enumerate(rows_sorted):
            pdlist = sorted(plants_by_row[r], key=lambda x: x["position"])
            for pd in pdlist:
                j = pd["position"] - 1
                if j >= len(self.entries[ri]): continue
                if pd.get("replication"):
                    self._cell_replication[(r, pd["position"])] = pd["replication"]
                role = pd.get("role")
                if role == HP_ROLE_GUARD_EDGE: txt = "К"
                elif role == HP_ROLE_GUARD_REP: txt = "П"
                elif role == HP_ROLE_DEAD: txt = "-"
                elif role == HP_ROLE_POLLINIZER: txt = "+"
                elif role == HP_ROLE_EXTRA: txt = "×"
                elif role == HP_ROLE_RECORDED and pd.get("factors"):
                    codes = []
                    for i, fd in enumerate(self.factor_defs):
                        lvl = pd["factors"].get(fd["name"])
                        if lvl: codes.append(f"{FACTOR_LETTERS[i]}{lvl}")
                    txt = "".join(codes)
                elif pd.get("value") is not None:
                    txt = str(pd["value"])
                else:
                    txt = ""
                self.entries[ri][j].insert(0, txt)
        messagebox.showinfo("Відкрито", f"Схему завантажено:\n{path}")
