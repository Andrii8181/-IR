
# sad_scheme_constructor.py — Конструктор багатофакторної схеми досліду
# -*- coding: utf-8 -*-
from sad_common import *
from sad_homogeneous import (HPPlant, HP_ROLE_RECORDED, HP_ROLE_GUARD_EDGE,
    HP_ROLE_GUARD_REP, HP_ROLE_UNASSIGNED)
import itertools, re

# ═══════════════════════════════════════════════════════════════
# КОНСТРУКТОР СХЕМИ — вільна багатофакторна схема досліду
# ═══════════════════════════════════════════════════════════════
FACTOR_LETTERS = "ABCDEFGHIJ"

class SchemeConstructorWindow:
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
        self.win.geometry("1400x820"); set_icon(self.win)
        self.gs = dict(gs) if gs else {}
        self.factor_defs = []      # [{"name":..., "levels":[...]}, ...]
        self.row_lengths = []
        self.rows_n = 0
        self.entries = []
        self.row_labels = []
        self.pos_labels = []
        self.rep_vars = []         # StringVar на кожен ряд — номер повторності
        self._table_built = False
        self._build()

    # ─────────────────────────────────────────────────────
    def _build(self):
        rf = ("Times New Roman", 11)
        top = tk.Frame(self.win, padx=8, pady=6); top.pack(fill=tk.X)
        tk.Button(top, text="📝 Задати фактори", bg="#1a4b8c", fg="white",
                  font=rf, command=self._edit_factors).pack(side=tk.LEFT, padx=4)
        self._factors_status = tk.Label(top, text="(фактори не задано)",
                                        font=("Times New Roman",9), fg="#888")
        self._factors_status.pack(side=tk.LEFT, padx=(0,12))

        tk.Button(top, text="🎲 Рандомізувати", bg="#8c5a1a", fg="white",
                  font=rf, command=self._open_randomize_dialog).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Вставити з буфера", font=rf,
                  command=self._paste).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="💾 Зберегти схему", bg="#1a6b1a", fg="white",
                  font=rf, command=self._save_scheme).pack(side=tk.LEFT, padx=(12,4))
        tk.Button(top, text="📂 Відкрити схему", font=rf,
                  command=self._load_scheme).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="📚 Довідка", bg="#1a4b8c", fg="white", font=rf,
                  command=self._show_help).pack(side=tk.LEFT, padx=(12,4))
        self._resize_btn = tk.Button(top, text="🔧 Змінити розміри таблиці", font=rf,
                                     command=self._reset_table_size)

        # ── Крок: розміри таблиці ──────────────────────────
        self._setup_frame = tk.LabelFrame(self.win,
            text="Розміри таблиці — вкажіть, скільки рослин/ділянок у кожному ряду",
            font=("Times New Roman",11,"bold"), padx=10, pady=8)
        self._setup_frame.pack(fill=tk.X, padx=8, pady=(0,4))
        setup_top = tk.Frame(self._setup_frame); setup_top.pack(fill=tk.X)
        tk.Label(setup_top, text="Кількість рядів:", font=rf).pack(side=tk.LEFT)
        self._n_rows_setup_var = tk.StringVar(value="")
        tk.Entry(setup_top, textvariable=self._n_rows_setup_var, width=6, font=rf
                 ).pack(side=tk.LEFT, padx=6)
        tk.Button(setup_top, text="Задати довжину кожного ряду →", font=rf,
                  command=self._build_row_length_inputs).pack(side=tk.LEFT, padx=10)
        self._rowlen_holder = tk.Frame(self._setup_frame)
        self._rowlen_holder.pack(fill=tk.X, pady=(8,0))

        # ── Легенда факторів ────────────────────────────────
        self._legend_f = tk.Frame(self.win, bg="#eef3f8", padx=8, pady=6)
        self._legend_f.pack(fill=tk.X)

        # ── Область таблиці ─────────────────────────────────
        tbl_area = tk.Frame(self.win); tbl_area.pack(fill=tk.BOTH, expand=True, padx=8, pady=(2,4))
        self._canvas = tk.Canvas(tbl_area)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb_v = ttk.Scrollbar(tbl_area, orient="vertical", command=self._canvas.yview)
        sb_v.pack(side=tk.RIGHT, fill=tk.Y)
        sb_h = ttk.Scrollbar(self.win, orient="horizontal", command=self._canvas.xview)
        sb_h.pack(fill=tk.X, padx=8)
        self._canvas.configure(yscrollcommand=sb_v.set, xscrollcommand=sb_h.set)
        self.inner = tk.Frame(self._canvas)
        self._canvas.create_window((0,0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>",
                        lambda e: self._canvas.config(scrollregion=self._canvas.bbox("all")))
        self.win.bind("<MouseWheel>",
                      lambda e: self._canvas.yview_scroll(int(-1*(e.delta/120)),"units"))

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
        dlg.geometry("560x520"); set_icon(dlg); dlg.grab_set()
        rf = ("Times New Roman",11)

        top_f = tk.Frame(dlg, padx=14, pady=10); top_f.pack(fill=tk.X)
        tk.Label(top_f, text="Кількість факторів:", font=rf).pack(side=tk.LEFT)
        n_var = tk.StringVar(value=str(max(1, len(self.factor_defs))))
        tk.Spinbox(top_f, from_=1, to=len(FACTOR_LETTERS), textvariable=n_var,
                   width=4, font=rf).pack(side=tk.LEFT, padx=6)

        body = tk.Frame(dlg); body.pack(fill=tk.BOTH, expand=True, padx=14)
        rows_holder = tk.Frame(body); rows_holder.pack(fill=tk.BOTH, expand=True)
        name_vars, level_vars = [], []

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
                tk.Entry(fr, textvariable=nv, width=28, font=rf).grid(
                    row=0, column=1, sticky="w", padx=6)
                tk.Label(fr, text="Рівні (через кому):", font=rf).grid(
                    row=1, column=0, sticky="w", pady=(4,0))
                default_levels = (", ".join(self.factor_defs[i]["levels"])
                                  if i < len(self.factor_defs) else "")
                lv = tk.StringVar(value=default_levels)
                tk.Entry(fr, textvariable=lv, width=40, font=rf).grid(
                    row=1, column=1, sticky="w", padx=6, pady=(4,0))
                name_vars.append(nv); level_vars.append(lv)

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
            total_combos = 1
            for d in defs: total_combos *= len(d["levels"])
            self._factors_status.configure(
                text=f"✓ {len(defs)} факт., {total_combos} комбінацій: " +
                     "; ".join(f"{FACTOR_LETTERS[i]}={d['name']}" for i, d in enumerate(defs)),
                fg="#1a6b1a")
            self._build_legend()
            dlg.destroy()

        bf = tk.Frame(dlg); bf.pack(pady=10)
        tk.Button(bf, text="Зберегти", bg="#1a6b1a", fg="white", font=rf,
                  command=_save).pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text="Скасувати", font=rf, command=dlg.destroy).pack(side=tk.LEFT)
        center_win(dlg)

    def _build_legend(self):
        for w in self._legend_f.winfo_children(): w.destroy()
        if not self.factor_defs: return
        for i, d in enumerate(self.factor_defs):
            letter = FACTOR_LETTERS[i]
            txt = f"{letter} = {d['name']}:  " + "  ".join(
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
        grid_f = tk.Frame(self._rowlen_holder); grid_f.pack(fill=tk.X)
        rf = ("Times New Roman",10)
        PER_ROW = 6
        for i in range(n):
            r, c = divmod(i, PER_ROW)
            tk.Label(grid_f, text=f"Ряд {i+1}:", font=rf).grid(
                row=r, column=c*2, sticky="w", padx=(0 if c==0 else 10, 2), pady=2)
            v = tk.StringVar(value="")
            tk.Entry(grid_f, textvariable=v, width=5, font=rf).grid(
                row=r, column=c*2+1, sticky="w", pady=2)
            self._rowlen_vars.append(v)
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

        tk.Label(self.inner, text="Ряд \\ Поз.", width=9, relief=tk.RIDGE,
                 bg="#444444", fg="white", font=("Times New Roman",10,"bold")
                 ).grid(row=0, column=0, padx=1, pady=1, sticky="nsew")
        tk.Label(self.inner, text="Повт.", width=6, relief=tk.RIDGE,
                 bg="#6b4a1a", fg="white", font=("Times New Roman",9,"bold")
                 ).grid(row=0, column=1, padx=1, pady=1, sticky="nsew")
        self.pos_labels = []
        for j in range(max_len):
            lbl = tk.Label(self.inner, text=str(j+1), width=6, relief=tk.RIDGE,
                           bg="#1a4b8c", fg="white", font=("Times New Roman",9,"bold"))
            lbl.grid(row=0, column=j+2, padx=1, pady=1, sticky="nsew")
            self.pos_labels.append(lbl)

        self.row_labels = []
        self.entries = []
        self.rep_vars = []
        for i, L in enumerate(lengths):
            rl = tk.Label(self.inner, text=f"Ряд {i+1}", width=9, relief=tk.RIDGE,
                         bg="#444444", fg="white", font=("Times New Roman",9,"bold"))
            rl.grid(row=i+1, column=0, padx=1, pady=1, sticky="nsew")
            self.row_labels.append(rl)
            rv = tk.StringVar(value=str(i+1))
            tk.Entry(self.inner, textvariable=rv, width=6, font=("Times New Roman",9),
                    justify="center", bg="#fff3c4").grid(
                    row=i+1, column=1, padx=1, pady=1)
            self.rep_vars.append(rv)
            row_e = []
            for j in range(L):
                e = tk.Entry(self.inner, width=6, font=("Times New Roman",10),
                            justify="center")
                e.grid(row=i+1, column=j+2, padx=1, pady=1)
                row_e.append(e)
            self.entries.append(row_e)
        _bind_nav(self.entries, self.win)
        _bind_fill_handle(self.entries, self.win)

        self._table_built = True
        self._setup_frame.pack_forget()
        self._resize_btn.pack(side=tk.LEFT, padx=4)

    def _reset_table_size(self):
        if self.entries and not messagebox.askyesno("Змінити розміри таблиці",
                "Поточні дані таблиці буде втрачено при перебудові. Продовжити?"):
            return
        self._resize_btn.pack_forget()
        self._setup_frame.pack(fill=tk.X, padx=8, pady=(0,4), before=self._legend_f)
        self._n_rows_setup_var.set(str(self.rows_n) if self.rows_n else "")

    def _extend_row_to(self, ri, target_len):
        while len(self.entries[ri]) < target_len:
            j = len(self.entries[ri])
            if j >= len(self.pos_labels):
                lbl = tk.Label(self.inner, text=str(j+1), width=6, relief=tk.RIDGE,
                               bg="#1a4b8c", fg="white", font=("Times New Roman",9,"bold"))
                lbl.grid(row=0, column=j+2, padx=1, pady=1, sticky="nsew")
                self.pos_labels.append(lbl)
            e = tk.Entry(self.inner, width=6, font=("Times New Roman",10), justify="center")
            e.grid(row=ri+1, column=j+2, padx=1, pady=1)
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
    def _open_randomize_dialog(self):
        if not self.factor_defs:
            messagebox.showwarning("", "Спочатку задайте фактори."); return
        if not self._table_built:
            messagebox.showwarning("", "Спочатку побудуйте таблицю."); return

        total_combos = 1
        for d in self.factor_defs: total_combos *= len(d["levels"])

        dlg = tk.Toplevel(self.win); dlg.title("Рандомізація"); dlg.resizable(False, False)
        set_icon(dlg); dlg.grab_set()
        rf = ("Times New Roman",11)
        frm = tk.Frame(dlg, padx=16, pady=14); frm.pack()
        tk.Label(frm, text=f"Усього комбінацій факторів: {total_combos}",
                 font=("Times New Roman",11,"bold")).grid(row=0, column=0, columnspan=2,
                 sticky="w", pady=(0,10))

        tk.Label(frm, text="Дизайн:", font=rf).grid(row=1, column=0, sticky="w")
        design_map = {
            "CRD — повністю випадково по всій сітці": "crd",
            "RCBD — випадково в межах кожної повторності (рекомендується)": "rcbd",
            "Split-plot — один фактор на великих ділянках, інші всередині": "split",
            "Латинський квадрат — n×n, кожна комбінація раз у ряду й стовпці": "latin",
        }
        design_disp = tk.StringVar(value=list(design_map.keys())[1])
        ttk.Combobox(frm, textvariable=design_disp, values=list(design_map.keys()),
                     state="readonly", width=52).grid(row=1, column=1, sticky="w", padx=8)

        note_lbl = tk.Label(frm, font=("Times New Roman",9), fg="#666", justify="left",
                            wraplength=460)
        note_lbl.grid(row=2, column=0, columnspan=2, sticky="w", pady=(10,0))

        extra_frame = tk.Frame(frm)
        extra_frame.grid(row=3, column=0, columnspan=2, sticky="w", pady=(8,0))

        main_factor_v = tk.StringVar(value=FACTOR_LETTERS[0] if self.factor_defs else "A")
        latin_row_v = tk.StringVar(value="1")
        latin_col_v = tk.StringVar(value="1")

        def _update_extra(*_):
            for w in extra_frame.winfo_children(): w.destroy()
            design = design_map[design_disp.get()]
            if design in ("crd","rcbd"):
                note_lbl.configure(text=
                    "Клітинки, де вже вписано «К» чи «П» (захисні), не змінюються. "
                    "RCBD: кожна повторність (стовпець «Повт.») отримує рівно один "
                    "повний набір усіх комбінацій.")
            elif design == "split":
                note_lbl.configure(text=
                    "Один фактор («головний», на великих суцільних ділянках) "
                    "розподіляється по повторності, решта факторів рандомізуються "
                    "ОКРЕМО всередині кожної такої ділянки.")
                tk.Label(extra_frame, text="Головний фактор (велика ділянка):",
                         font=rf).grid(row=0, column=0, sticky="w")
                letters = [FACTOR_LETTERS[i] for i in range(len(self.factor_defs))]
                ttk.Combobox(extra_frame, textvariable=main_factor_v, values=letters,
                             state="readonly", width=6).grid(row=0, column=1, sticky="w", padx=8)
            elif design == "latin":
                n = total_combos
                note_lbl.configure(text=
                    f"Розміщує квадрат {n}×{n} (n = кількість комбінацій), де кожна "
                    f"комбінація зустрічається рівно раз у кожному ряду й кожному "
                    f"стовпці цього квадрата. Вкажіть, з якого ряду й позиції він "
                    f"починається — потрібно {n} рядів поспіль, кожен щонайменше "
                    f"{n} вільних позицій підряд від вказаної.")
                tk.Label(extra_frame, text="Починаючи з ряду №:", font=rf).grid(
                    row=0, column=0, sticky="w")
                tk.Spinbox(extra_frame, from_=1, to=max(1,self.rows_n), textvariable=latin_row_v,
                           width=5, font=rf).grid(row=0, column=1, sticky="w", padx=8)
                tk.Label(extra_frame, text="Починаючи з позиції №:", font=rf).grid(
                    row=1, column=0, sticky="w", pady=(4,0))
                max_pos = max((len(r) for r in self.entries), default=1)
                tk.Spinbox(extra_frame, from_=1, to=max_pos, textvariable=latin_col_v,
                           width=5, font=rf).grid(row=1, column=1, sticky="w", padx=8, pady=(4,0))

        design_disp.trace_add("write", _update_extra)
        _update_extra()

        def _go():
            design = design_map[design_disp.get()]
            if design in ("crd","rcbd"):
                self._randomize(design)
            elif design == "split":
                idx = FACTOR_LETTERS.index(main_factor_v.get())
                self._randomize_split_plot(idx)
            elif design == "latin":
                try:
                    r0 = int(latin_row_v.get())-1; c0 = int(latin_col_v.get())-1
                except ValueError:
                    messagebox.showwarning("", "Некоректні координати.", parent=dlg); return
                self._randomize_latin_square(r0, c0)
            dlg.destroy()
        bf = tk.Frame(frm); bf.grid(row=4, column=0, columnspan=2, pady=(14,0))
        tk.Button(bf, text="Рандомізувати", bg="#8c5a1a", fg="white", font=rf,
                  command=_go).pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text="Скасувати", font=rf, command=dlg.destroy).pack(side=tk.LEFT)
        center_win(dlg)

    def _is_guard_text(self, txt):
        return txt.strip().upper() in ("К","K","П","P")

    def _randomize(self, design):
        import random
        combos = list(itertools.product(*[range(1, len(d["levels"])+1) for d in self.factor_defs]))
        n_combos = len(combos)

        def _code(combo):
            return "".join(f"{FACTOR_LETTERS[i]}{lvl}" for i, lvl in enumerate(combo))

        if design == "crd":
            eligible = [(i,j) for i, row in enumerate(self.entries) for j in range(len(row))
                       if not self._is_guard_text(row[j].get())]
            reps_needed = math.ceil(len(eligible) / n_combos) if eligible else 0
            pool = (combos * reps_needed)[:len(eligible)]
            random.shuffle(pool)
            for (i,j), combo in zip(eligible, pool):
                self.entries[i][j].delete(0, tk.END)
                self.entries[i][j].insert(0, _code(combo))
        else:  # rcbd
            by_rep = {}
            for i, row in enumerate(self.entries):
                rep = self.rep_vars[i].get().strip() or str(i+1)
                for j in range(len(row)):
                    if self._is_guard_text(row[j].get()): continue
                    by_rep.setdefault(rep, []).append((i,j))
            for rep, cells in by_rep.items():
                reps_needed = math.ceil(len(cells) / n_combos)
                pool = (combos * reps_needed)[:len(cells)]
                random.shuffle(pool)
                for (i,j), combo in zip(cells, pool):
                    self.entries[i][j].delete(0, tk.END)
                    self.entries[i][j].insert(0, _code(combo))
        messagebox.showinfo("Готово",
            "Комбінації розкидано за обраним дизайном. Клітинки залишаються "
            "повністю редагованими — перевірте й підправте за потреби.")

    def _randomize_split_plot(self, main_idx):
        """Головний фактор (main_idx) — на великих суцільних ділянках у межах
        кожної повторності; решта факторів рандомізуються окремо всередині
        кожної такої ділянки."""
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

        by_rep = {}
        for i, row in enumerate(self.entries):
            rep = self.rep_vars[i].get().strip() or str(i+1)
            for j in range(len(row)):
                if self._is_guard_text(row[j].get()): continue
                by_rep.setdefault(rep, []).append((i,j))

        skipped_reps = []
        for rep, cells in by_rep.items():
            needed = n_main * n_sub
            if len(cells) < n_sub:
                skipped_reps.append(rep); continue
            main_levels_order = list(range(1, n_main+1)); random.shuffle(main_levels_order)
            idx_cell = 0
            for seg_i in range(n_main):
                if idx_cell >= len(cells): break
                seg_cells = cells[idx_cell: idx_cell+n_sub]
                idx_cell += n_sub
                main_lvl = main_levels_order[seg_i]
                sub_pool = sub_combos.copy(); random.shuffle(sub_pool)
                for (i,j), sub_combo in zip(seg_cells, sub_pool):
                    self.entries[i][j].delete(0, tk.END)
                    self.entries[i][j].insert(0, _code(main_lvl, sub_combo))
        msg = ("Комбінації розкидано за схемою split-plot. Клітинки залишаються "
               "повністю редагованими — перевірте й підправте за потреби.")
        if skipped_reps:
            msg += (f"\n\n⚠ У повторностях {', '.join(skipped_reps)} забракло вільних "
                    f"клітинок для жодної повної підділянки — їх пропущено.")
        messagebox.showinfo("Готово", msg)

    def _randomize_latin_square(self, r0, c0):
        """Розміщує n×n латинський квадрат (n = кількість комбінацій),
        починаючи з ряду r0 і позиції c0 (0-індексовані)."""
        import random
        combos = list(itertools.product(*[range(1, len(d["levels"])+1) for d in self.factor_defs]))
        n = len(combos)

        def _code(combo):
            return "".join(f"{FACTOR_LETTERS[i]}{lvl}" for i, lvl in enumerate(combo))

        if r0 < 0 or r0 + n > len(self.entries):
            messagebox.showwarning("", f"Потрібно {n} рядів поспіль, починаючи з "
                                       f"вказаного — у таблиці їх недостатньо."); return
        for k in range(n):
            if c0 + n > len(self.entries[r0+k]):
                messagebox.showwarning("", f"Ряд {r0+k+1} має недостатньо позицій "
                                           f"(потрібно {n} підряд від позиції {c0+1})."); return

        base = [[(i+j) % n for j in range(n)] for i in range(n)]
        random.shuffle(base)
        cols = list(range(n)); random.shuffle(cols)
        base = [[row[c] for c in cols] for row in base]
        symbols = list(range(n)); random.shuffle(symbols)
        square = [[symbols[v] for v in row] for row in base]

        for i in range(n):
            for j in range(n):
                e = self.entries[r0+i][c0+j]
                e.delete(0, tk.END)
                e.insert(0, _code(combos[square[i][j]]))
        messagebox.showinfo("Готово",
            f"Латинський квадрат {n}×{n} розміщено з ряду {r0+1}, позиції {c0+1}. "
            "Кожна комбінація зустрічається рівно раз у кожному ряду й стовпці "
            "квадрата. Клітинки залишаються повністю редагованими.")

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
    def _save_scheme(self):
        if not self.factor_defs:
            messagebox.showwarning("", "Спочатку задайте фактори."); return
        if not self._table_built:
            messagebox.showwarning("", "Спочатку побудуйте таблицю."); return

        plants = []
        bad_cells = []
        for i, row in enumerate(self.entries):
            rep = self.rep_vars[i].get().strip() or str(i+1)
            try: rep_num = int(rep)
            except ValueError: rep_num = i+1
            for j, e in enumerate(row):
                txt = e.get().strip()
                p = HPPlant(i+1, j+1, None, "ok")
                if not txt:
                    p.role = HP_ROLE_UNASSIGNED
                elif txt.upper() in ("К","K"):
                    p.role = HP_ROLE_GUARD_EDGE
                elif txt.upper() in ("П","P"):
                    p.role = HP_ROLE_GUARD_REP
                else:
                    factors = self._parse_code(txt)
                    if factors is None:
                        bad_cells.append(f"Ряд {i+1}, поз. {j+1}: «{txt}»")
                        p.role = HP_ROLE_UNASSIGNED
                    else:
                        p.role = HP_ROLE_RECORDED
                        p.factors = factors
                        p.replication = rep_num
                plants.append(p)

        if bad_cells:
            preview = "\n".join(bad_cells[:10])
            more = f"\n… і ще {len(bad_cells)-10}" if len(bad_cells) > 10 else ""
            if not messagebox.askyesno("Нерозпізнані клітинки",
                    f"Не вдалось розпізнати код у {len(bad_cells)} клітинках "
                    f"(мають бути коди факторів на кшталт «A2B3», «К» чи «П», "
                    f"або порожні):\n\n{preview}{more}\n\n"
                    "Зберегти попри це (ці клітинки будуть позначені як "
                    "непризначені)?"):
                return

        d = {
            "type": "multi_factor_scheme", "version": APP_VER,
            "cfg": {"trait_name": "", "trait_unit": "", "design_used": "custom"},
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

        plants_by_row = {}
        for pd in d.get("plants", []):
            plants_by_row.setdefault(pd["row"], []).append(pd)
        rows_sorted = sorted(plants_by_row.keys())
        lengths = [len(plants_by_row[r]) for r in rows_sorted]

        for w in self.inner.winfo_children(): w.destroy()
        self.row_lengths = []
        self.rows_n = 0
        self.entries = []; self.row_labels = []; self.pos_labels = []; self.rep_vars = []
        self._rowlen_vars = [tk.StringVar(value=str(L)) for L in lengths]
        self._build_data_table()

        for ri, r in enumerate(rows_sorted):
            pdlist = sorted(plants_by_row[r], key=lambda x: x["position"])
            reps = {pd.get("replication") for pd in pdlist if pd.get("replication")}
            if reps: self.rep_vars[ri].set(str(sorted(reps)[0]))
            for pd in pdlist:
                j = pd["position"] - 1
                if j >= len(self.entries[ri]): continue
                role = pd.get("role")
                if role == HP_ROLE_GUARD_EDGE: txt = "К"
                elif role == HP_ROLE_GUARD_REP: txt = "П"
                elif role == HP_ROLE_RECORDED and pd.get("factors"):
                    codes = []
                    for i, fd in enumerate(self.factor_defs):
                        lvl = pd["factors"].get(fd["name"])
                        if lvl: codes.append(f"{FACTOR_LETTERS[i]}{lvl}")
                    txt = "".join(codes)
                else:
                    txt = ""
                self.entries[ri][j].insert(0, txt)
        messagebox.showinfo("Відкрито", f"Схему завантажено:\n{path}")
