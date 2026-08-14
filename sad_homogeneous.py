# sad_homogeneous.py — Довідка, планування за однорідністю, польовий журнал (частина)
# -*- coding: utf-8 -*-
from sad_common import *

class HelpWindow:
    def __init__(self, parent, start_topic=None):
        self.win = tk.Toplevel(parent)
        self.win.title("S.A.D. — Довідка")
        self.win.geometry("1000x660"); set_icon(self.win)
        self.current_topic = None
        self._build()
        topic = start_topic or list(HELP_CONTENT.keys())[0]
        self._show_topic(topic)

    def _build(self):
        # Left panel
        left = tk.Frame(self.win, width=220, bg="#f5f5f5", relief=tk.RIDGE, bd=1)
        left.pack(side=tk.LEFT, fill=tk.Y); left.pack_propagate(False)
        tk.Label(left, text="Зміст довідки",
                 font=("Times New Roman",12,"bold"), bg="#1a4b8c", fg="white",
                 pady=8).pack(fill=tk.X)
        # Search
        sf = tk.Frame(left, bg="#f5f5f5"); sf.pack(fill=tk.X, padx=6, pady=6)
        tk.Label(sf, text="Пошук:", bg="#f5f5f5", font=("Times New Roman",11)).pack(side=tk.LEFT)
        self._sv = tk.StringVar(); self._sv.trace_add("write", self._on_search)
        tk.Entry(sf, textvariable=self._sv, font=("Times New Roman",11),
                 relief=tk.FLAT, bg="white").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        # Topic buttons frame (scrollable)
        self._tf = tk.Frame(left, bg="#f5f5f5"); self._tf.pack(fill=tk.BOTH, expand=True)
        self._btn = {}; self._build_list(list(HELP_CONTENT.keys()))
        # Right panel
        right = tk.Frame(self.win, bg="white"); right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._title = tk.Label(right, text="", font=("Times New Roman",14,"bold"),
                               bg="#1a4b8c", fg="white", pady=8, padx=10, anchor="w")
        self._title.pack(fill=tk.X)
        tf2 = tk.Frame(right); tf2.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        ysb = ttk.Scrollbar(tf2); ysb.pack(side=tk.RIGHT, fill=tk.Y)
        self._txt = tk.Text(tf2, wrap="word", font=("Times New Roman",12),
                            state="disabled", relief=tk.FLAT, bg="white",
                            yscrollcommand=ysb.set, padx=10, pady=8, cursor="arrow")
        self._txt.pack(fill=tk.BOTH, expand=True)
        ysb.config(command=self._txt.yview)
        self._txt.bind("<MouseWheel>",
                       lambda e: self._txt.yview_scroll(int(-1*(e.delta/120)),"units"))
        # Tags
        self._txt.tag_configure("bold", font=("Times New Roman",12,"bold"))
        self._txt.tag_configure("check", foreground="#1a6b1a", font=("Times New Roman",12))
        self._txt.tag_configure("warn",  foreground="#c62828", font=("Times New Roman",12))
        self._txt.tag_configure("normal",font=("Times New Roman",12))
        # Bottom
        bot = tk.Frame(right, bg="#f0f0f0", pady=4); bot.pack(fill=tk.X)
        tk.Button(bot, text="<- Попередня", command=self._prev,
                  font=("Times New Roman",11)).pack(side=tk.LEFT, padx=8)
        tk.Button(bot, text="Наступна ->", command=self._next,
                  font=("Times New Roman",11)).pack(side=tk.LEFT, padx=4)
        tk.Button(bot, text="Закрити", command=self.win.destroy,
                  font=("Times New Roman",11)).pack(side=tk.RIGHT, padx=8)

    def _build_list(self, topics):
        for w in self._tf.winfo_children(): w.destroy()
        self._btn = {}
        for topic in topics:
            info = HELP_CONTENT.get(topic, {})
            frm = tk.Frame(self._tf, bg="#f5f5f5", cursor="hand2")
            frm.pack(fill=tk.X, padx=3, pady=1)
            icon = info.get("icon","•")
            lbl = tk.Label(frm, text=f"{icon}  {topic}",
                           font=("Times New Roman",11,"bold"), bg="#f5f5f5",
                           anchor="w", padx=6, pady=2)
            lbl.pack(fill=tk.X)
            sub = tk.Label(frm, text=f"    {info.get('short','')}",
                           font=("Times New Roman",9), fg="#666", bg="#f5f5f5", anchor="w", padx=6)
            sub.pack(fill=tk.X)
            for w in [frm, lbl, sub]:
                w.bind("<Button-1>", lambda e, t=topic: self._show_topic(t))
                w.bind("<Enter>",  lambda e, f=frm: [c.configure(bg="#dce8ff") for c in [f]+list(f.winfo_children())])
                w.bind("<Leave>",  lambda e, f=frm, t=topic: self._set_bg(f, t))
            self._btn[topic] = frm

    def _set_bg(self, frm, topic):
        bg = "#c8d8ff" if topic == self.current_topic else "#f5f5f5"
        frm.configure(bg=bg)
        for w in frm.winfo_children(): w.configure(bg=bg)

    def _on_search(self, *_):
        q = self._sv.get().strip().lower()
        if not q: self._build_list(list(HELP_CONTENT.keys())); return
        matched = [t for t, info in HELP_CONTENT.items()
                   if q in t.lower() or q in info.get("short","").lower()
                   or q in info.get("text","").lower()]
        self._build_list(matched)

    def _show_topic(self, topic):
        if topic not in HELP_CONTENT: return
        self.current_topic = topic
        info = HELP_CONTENT[topic]
        self._title.configure(text=f"{info.get('icon','')}  {topic}")
        for t, frm in self._btn.items(): self._set_bg(frm, t)
        self._txt.configure(state="normal")
        self._txt.delete("1.0", tk.END)
        for line in info.get("text","").strip().split("\n"):
            s = line.strip()
            if s.startswith("✓") or s.startswith("✔"):
                tag = "check"
            elif s.startswith("⚠") or s.startswith("✗") or "БЛОК" in s or "КРИТ" in s:
                tag = "warn"
            elif line.isupper() and len(line) > 3:
                tag = "bold"
            else:
                tag = "normal"
            self._txt.insert(tk.END, line + "\n", tag)
        self._txt.configure(state="disabled")
        self._txt.yview_moveto(0)

    def _prev(self):
        keys = list(HELP_CONTENT.keys())
        if self.current_topic in keys:
            idx = keys.index(self.current_topic)
            if idx > 0: self._show_topic(keys[idx-1])

    def _next(self):
        keys = list(HELP_CONTENT.keys())
        if self.current_topic in keys:
            idx = keys.index(self.current_topic)
            if idx < len(keys)-1: self._show_topic(keys[idx+1])


def show_help(parent, topic=None):
    HelpWindow(parent, start_topic=topic)

# ═══════════════════════════════════════════════════════════════
# UPDATE MENU — add ANCOVA and MANOVA buttons
# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
# ПЛАНУВАННЯ ДОСЛІДУ — Конструктор схеми за однорідністю рослин
# (вставити ПІСЛЯ класу TrialDesignWindow, ПЕРЕД рядком
#  "_SADTk_orig_init = SADTk.__init__")
# ═══════════════════════════════════════════════════════════════

# ── Ролі рослин після побудови схеми ──────────────────────────
HP_ROLE_RECORDED     = "recorded"
HP_ROLE_GUARD_EDGE    = "guard_edge"    # захисна зона: край ряду (тип 1)
HP_ROLE_GUARD_REP     = "guard_rep"     # захисна зона: між повтореннями (тип 2)
HP_ROLE_EXCLUDED_CV  = "excluded_cv"
HP_ROLE_DEAD         = "dead"
HP_ROLE_POLLINIZER   = "pollinizer"
HP_ROLE_UNASSIGNED   = "unassigned"
HP_ROLE_EXTRA        = "extra_unused"   # повторність поза дизайном (залишок блоку)

HP_ROLE_COLORS = {
    HP_ROLE_RECORDED:    "#4CAF50",
    HP_ROLE_GUARD_EDGE:  "#FFC107",
    HP_ROLE_GUARD_REP:   "#FF8A65",
    HP_ROLE_EXCLUDED_CV: "#BDBDBD",
    HP_ROLE_DEAD:        "#424242",
    HP_ROLE_POLLINIZER:  "#2196F3",
    HP_ROLE_UNASSIGNED:  "#ECEFF1",
    HP_ROLE_EXTRA:       "#CFD8DC",
}
HP_ROLE_LABELS = {
    HP_ROLE_RECORDED:    "Облікова рослина (варіант)",
    HP_ROLE_GUARD_EDGE:  "Захисна — край ряду",
    HP_ROLE_GUARD_REP:   "Захисна — між повтореннями",
    HP_ROLE_EXCLUDED_CV: "Виключено за варіабельністю",
    HP_ROLE_DEAD:        "Випад / пошкоджена (-)",
    HP_ROLE_POLLINIZER:  "Запилювач (+)",
    HP_ROLE_UNASSIGNED:  "Поза межами досліду",
    HP_ROLE_EXTRA:       "Повторність поза дизайном (залишок)",
}

HP_DESIGNS = [
    ("Повна рандомізація (CRD)",         "crd"),
    ("Рандомізовані повні блоки (RCBD)", "rcbd"),
    ("Латинський квадрат",               "latin"),
]
HP_DESIGN_LABELS = dict(HP_DESIGNS)
HP_DESIGN_LABELS_REV = {v: k for k, v in HP_DESIGNS}
HP_DESIGN_LABELS_REV.update({lbl: key for lbl, key in HP_DESIGNS})


class HPPlant:
    """Одна рослина сітки ряд×позиція для конструктора однорідних ділянок."""
    __slots__ = ("row","position","value","status","role","plot_id","variant","replication","factors")
    def __init__(self, row, position, value, status="ok"):
        self.row = row; self.position = position
        self.value = value            # float або None (DEAD/POLLINIZER)
        self.status = status          # "ok" | "dead" | "pollinizer"
        self.role = HP_ROLE_UNASSIGNED
        self.plot_id = None
        self.variant = None
        self.replication = None
        self.factors = {}             # {"A": 1, "B": 3, ...} — для багатофакторних схем;
                                       # порожній словник = звичайна однофакторна схема (self.variant)

    @property
    def id(self):
        return (self.row, self.position)


def hp_cv_percent(values):
    if len(values) < 2: return 0.0
    m = float(np.mean(values))
    if m == 0: return 0.0
    sd = float(np.std(values, ddof=0))
    return sd / m * 100.0


def hp_load_plants(raw_rows):
    """raw_rows: {row_num: [(position, raw_value_str), ...]}"""
    plants = []
    for row_num, entries in raw_rows.items():
        for pos, raw_val in entries:
            v = (raw_val or "").strip()
            if v == "-":
                plants.append(HPPlant(row_num, pos, None, "dead"))
            elif v == "+":
                plants.append(HPPlant(row_num, pos, None, "pollinizer"))
            elif v:
                try:
                    plants.append(HPPlant(row_num, pos, float(v.replace(",", ".")), "ok"))
                except ValueError:
                    continue
    plants.sort(key=lambda p: (p.row, p.position))
    return plants


class HPPlotBuilder:
    """
    Ітеративний конструктор схеми.

    Термінологія: одна ПОВТОРНІСТЬ = одна група з plot_size облікових
    рослин ОДНОГО варіанту (те, що раніше в коментарях називалось
    "ділянкою"). Варіант складається з кількох повторностей (n_rep),
    розкиданих рандомізовано по полю/саду (див. hp_apply_design) —
    саме тому захист потрібен між БУДЬ-ЯКИМИ двома сусідніми
    повторностями, а не лише на межі "блоку з n_var штук".

      1. Формування повторностей уздовж ряду: набір plot_size
         облікових рослин (непридатні позиції "прозоро" пропускаються)
         -> повторити.
      2. Захисні рослини — ДВА окремих типи:
           • edge_guard_size — на початку і в кінці КОЖНОГО ряду (тип 1);
           • rep_guard_size  — після КОЖНОЇ сформованої повторності,
             тобто між будь-якими двома сусідніми повторностями (тип 2).
      3. CV% рахується лише за RECORDED-рослинами (захисні виключені).
      4. Допустимий діапазон значень звужується навколо СЕРЕДНЬОГО
         з коефіцієнтом на основі ФІКСОВАНОГО σ_start (не поточного σ) —
         це запобігає подвійному (лінійному + природному) стисненню.
      5. Зупинка: множина RECORDED не змінюється, або CV% <= порогу,
         або вичерпано max_iterations (за замовчуванням 20).
    """
    def __init__(self, plants, plot_size, edge_guard_size, rep_guard_size,
                 cv_threshold_pct, max_iterations=20,
                 count_dead_as_guard=True, count_pollinizer_as_guard=False):
        self.plants = plants
        self.plot_size = max(1, int(plot_size))
        self.edge_guard_size = max(0, int(edge_guard_size))
        self.rep_guard_size  = max(0, int(rep_guard_size))
        self.cv_threshold_pct = float(cv_threshold_pct)
        self.max_iterations = max(1, int(max_iterations))
        self.count_dead_as_guard = count_dead_as_guard
        self.count_pollinizer_as_guard = count_pollinizer_as_guard
        self.by_row = {}
        for p in plants:
            self.by_row.setdefault(p.row, []).append(p)
        for row in self.by_row:
            self.by_row[row].sort(key=lambda p: p.position)

    def _guard_eligible(self, p):
        if p.status == "dead": return self.count_dead_as_guard
        if p.status == "pollinizer": return self.count_pollinizer_as_guard
        return True

    def _take_guard(self, row_plants, i, n, needed, role):
        taken = 0
        taken_plants = []
        while i < n and taken < needed:
            gp = row_plants[i]
            if self._guard_eligible(gp):
                gp.role = role; taken += 1
                taken_plants.append(gp)
            elif gp.status == "dead":
                gp.role = HP_ROLE_DEAD
            elif gp.status == "pollinizer":
                gp.role = HP_ROLE_POLLINIZER
            i += 1
        return i, taken_plants

    def _scan_once(self, allowed_range):
        recorded = []; plot_counter = 0
        for row_num in sorted(self.by_row.keys()):
            row_plants = self.by_row[row_num]
            i = 0; n = len(row_plants)

            # Захисна зона типу 1 — початок ряду
            i, _ = self._take_guard(row_plants, i, n, self.edge_guard_size, HP_ROLE_GUARD_EDGE)

            last_rep_guard_plants = []
            ended_naturally = True
            while i < n:
                plot_members = []
                while i < n and len(plot_members) < self.plot_size:
                    p = row_plants[i]
                    eligible = (p.status == "ok" and
                                (allowed_range is None or
                                 allowed_range[0] <= p.value <= allowed_range[1]))
                    if eligible:
                        plot_members.append(p)
                    elif p.status == "dead":
                        p.role = HP_ROLE_DEAD
                    elif p.status == "pollinizer":
                        p.role = HP_ROLE_POLLINIZER
                    i += 1
                if len(plot_members) < self.plot_size:
                    # Ряд закінчився без повної повторності. Лічильник i вже
                    # пройшов повз ці рештки, тож звичайний виклик _take_guard
                    # нижче їх більше не побачить (i == n) — без цього вони
                    # губилися б без жодної ролі замість стати захисною зоною
                    # кінця ряду. Останні edge_guard_size придатних рослин із
                    # решток стають захисною зоною (тип 1), як і мало бути.
                    tail = plot_members[-self.edge_guard_size:] if self.edge_guard_size > 0 else []
                    for p in tail:
                        p.role = HP_ROLE_GUARD_EDGE
                    ended_naturally = False
                    break  # неповна повторність відкидається
                plot_counter += 1
                for p in plot_members:
                    p.role = HP_ROLE_RECORDED; p.plot_id = plot_counter
                recorded.extend(plot_members)

                # Захисна зона типу 2 — після КОЖНОЇ сформованої повторності
                # (повторності розкидаються рандомізовано по варіантах пізніше,
                # тож будь-які дві сусідні повторності потребують захисту між ними)
                i, last_rep_guard_plants = self._take_guard(
                    row_plants, i, n, self.rep_guard_size, HP_ROLE_GUARD_REP)

            # Якщо ряд закінчився РІВНО на щойно взятій захисній зоні між
            # повтореннями (без жодної рослини по тому) — вона фактично і Є
            # захисною зоною кінця ряду, а не «між повтореннями» (адже
            # наступного повторення там уже нема). Перекласифіковуємо для
            # відповідності принципу «останні edge_guard_size рослин ряду —
            # завжди захисна зона краю».
            if ended_naturally and i >= n and last_rep_guard_plants:
                for p in last_rep_guard_plants:
                    p.role = HP_ROLE_GUARD_EDGE

            # Захисна зона типу 1 — кінець ряду (додатково, якщо після break
            # лишилось ще місце, або повторностей не було взагалі)
            i, _ = self._take_guard(row_plants, i, n, self.edge_guard_size, HP_ROLE_GUARD_EDGE)
        return recorded

    def build(self):
        warnings = []; prev_ids = None; allowed_range = None
        sigma_start = None; iterations_used = 0; converged = False; final_cv = 0.0

        for it in range(1, self.max_iterations + 1):
            iterations_used = it
            for p in self.plants:
                p.role = HP_ROLE_UNASSIGNED; p.plot_id = None

            recorded = self._scan_once(allowed_range)
            ids = {p.id for p in recorded}
            values = [p.value for p in recorded]
            final_cv = hp_cv_percent(values) if values else 0.0

            for p in self.plants:
                if p.status == "ok" and p.role == HP_ROLE_UNASSIGNED:
                    p.role = HP_ROLE_EXCLUDED_CV

            if prev_ids is not None and ids == prev_ids:
                converged = True; break
            prev_ids = ids

            if not values:
                warnings.append(f"Ітерація {it}: жодної ділянки не сформовано — "
                                 "перевірте поріг CV%, розмір ділянки/захисних зон.")
                break
            if final_cv <= self.cv_threshold_pct:
                converged = True; break

            m = float(np.mean(values))
            sd = float(np.std(values, ddof=0)) if len(values) > 1 else 0.0
            if sigma_start is None: sigma_start = sd  # фіксується один раз

            shrink = 1.0 - (it / (self.max_iterations * 1.5))
            half_w = max(sigma_start * max(shrink, 0.1), 1e-6)
            allowed_range = (m - half_w, m + half_w)

        if not converged:
            warnings.append(f"Алгоритм не досяг повної збіжності за {self.max_iterations} "
                             f"ітерацій (зупинено на CV={final_cv:.2f}%).")

        plots_formed = len({p.plot_id for p in self.plants if p.role == HP_ROLE_RECORDED})
        return {
            "plants": self.plants, "plots_formed": plots_formed,
            "iterations_used": iterations_used, "converged": converged,
            "final_cv_pct": final_cv, "warnings": warnings,
        }


def hp_apply_design(result, design, num_variants, num_reps, seed=None):
    """Розподіляє plot_id -> (variant, replication) ЗГІДНО ОБРАНОГО ДИЗАЙНУ
    ЕКСПЕРИМЕНТУ. Виконується ПІСЛЯ формування ділянок за CV%, незалежно
    від порядку їх фізичного відбору.

      • CRD   — повна рандомізація: варіанти розкидані повністю випадково
                по всіх сформованих ділянках, без блокової структури.
      • RCBD  — кожні n_var послідовно сформованих ділянок = одне повне
                повторення; мітки варіантів перемішуються в межах блоку.
      • Latin — латинський квадрат порядку n_var (потрібно рівно n_var
                блоків); якщо блоків не вистачає — програма попереджає
                і автоматично переходить на RCBD.

    Повертає (реально_застосований_дизайн, чи_був_відкат_на_RCBD)."""
    import random as _random
    rng = _random.Random(seed)
    plot_ids = sorted({p.plot_id for p in result["plants"] if p.role == HP_ROLE_RECORDED})
    plants_by_plot = {}
    for p in result["plants"]:
        if p.plot_id is not None:
            plants_by_plot.setdefault(p.plot_id, []).append(p)

    k = max(1, int(num_variants))
    n_blocks = len(plot_ids) // k
    used_ids = plot_ids[:n_blocks * k]
    fell_back = False

    if design == "latin" and n_blocks != k:
        design = "rcbd"; fell_back = True

    if design == "crd":
        pool = []
        for v in range(1, k + 1):
            pool.extend([v] * max(1, int(num_reps)))
        rng.shuffle(pool)
        assign_ids = list(plot_ids); rng.shuffle(assign_ids)
        n_assign = min(len(assign_ids), len(pool))
        for idx in range(n_assign):
            plot_id = assign_ids[idx]; variant = pool[idx]
            rep_num = (idx // k) + 1
            for p in plants_by_plot[plot_id]:
                p.variant = variant; p.replication = rep_num

    elif design == "rcbd":
        for b in range(n_blocks):
            block = used_ids[b*k:(b+1)*k]
            labels = list(range(1, k+1)); rng.shuffle(labels)
            for plot_id, variant in zip(block, labels):
                for p in plants_by_plot[plot_id]:
                    p.variant = variant; p.replication = b+1

    elif design == "latin":
        base = [[(c + r) % k for c in range(k)] for r in range(k)]
        row_perm = list(range(k)); rng.shuffle(row_perm)
        col_perm = list(range(k)); rng.shuffle(col_perm)
        sym_perm = list(range(1, k+1)); rng.shuffle(sym_perm)
        square = [[sym_perm[base[row_perm[r]][col_perm[c]]] for c in range(k)]
                  for r in range(k)]
        for b in range(n_blocks):
            block = used_ids[b*k:(b+1)*k]
            for c, plot_id in enumerate(block):
                variant = square[b][c]
                for p in plants_by_plot[plot_id]:
                    p.variant = variant; p.replication = b+1

    # Повторності, що НЕ увійшли в жоден повний блок дизайну (залишок,
    # коли кількість сформованих повторностей не кратна num_variants),
    # переводимо в окрему роль — інакше вони лишались би RECORDED без
    # варіанту і показувались би на карті незрозумілою міткою "VNone".
    if design in ("rcbd", "latin"):
        leftover_ids = set(plot_ids) - set(used_ids)
        for plot_id in leftover_ids:
            for p in plants_by_plot[plot_id]:
                p.role = HP_ROLE_EXTRA
    elif design == "crd":
        unassigned_ids = set(plot_ids) - set(assign_ids[:n_assign])
        for plot_id in unassigned_ids:
            for p in plants_by_plot[plot_id]:
                p.role = HP_ROLE_EXTRA

    return design, fell_back


class HPResultsViewMixin:
    """
    Спільна логіка перегляду результату (карта саду, список облікових
    рослин, друкований бланк, збереження PNG) — використовується і
    «Плануванням за однорідністю» (однофакторне), і «Конструктором
    схеми» (багатофакторне, через «📊 За однорідністю»). Обидва класи
    викликають однаковий алгоритм відбору (HPPlotBuilder + hp_apply_design)
    — тому й перегляд результату тепер один і той самий код, а не дві
    окремі, потенційно розбіжні реалізації.

    Класи, що використовують цей mixin, повинні:
      • мати self._result, self._cfg (заповнюються після відбору)
      • реалізувати _plant_map_label(p), _plant_list_label(p),
        _plant_form_label(p), _map_legend_note_text(),
        _map_legend_decode(frame), _list_extra_header_lines()
    """

    def _show_results(self):
        win = tk.Toplevel(self.win)
        win.title("Планування досліду — Результати")
        win.geometry("1300x860"); set_icon(win)
        self._res_win = win

        main = tk.Frame(win); main.pack(fill=tk.BOTH, expand=True)
        sidebar = tk.Frame(main, width=210, bg="#2c3e50")
        sidebar.pack(side=tk.LEFT, fill=tk.Y); sidebar.pack_propagate(False)
        content = tk.Frame(main); content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(sidebar, text="ПЛАН ДОСЛІДУ", bg="#2c3e50", fg="#ecf0f1",
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

        map_frame  = tk.Frame(content)
        list_frame = tk.Frame(content)
        form_frame = tk.Frame(content)

        b_map  = _sidebar_btn("🗺 Карта саду",              "Кольорова схема ділянок")
        b_list = _sidebar_btn("📋 Список облікових рослин", "За варіантами й повтореннями")
        b_form = _sidebar_btn("🖨 Бланк обліку",            "Друкована форма для запису в полі")

        b_map.configure( command=lambda: _show_panel(map_frame, b_map))
        b_list.configure(command=lambda: _show_panel(list_frame, b_list))
        b_form.configure(command=lambda: _show_panel(form_frame, b_form))

        self._build_map_panel(map_frame)
        self._build_list_panel(list_frame)
        self._build_form_panel(form_frame)

        _show_panel(map_frame, b_map)

    # ── панель: карта саду ───────────────────────────────
    def _build_map_panel(self, frame):
        for w in frame.winfo_children(): w.destroy()
        tb = tk.Frame(frame, padx=6, pady=5); tb.pack(fill=tk.X)
        tk.Button(tb, text="💾 Зберегти PNG (друк)", font=("Times New Roman",11),
                  command=lambda: self._save_png(self._map_fig, "Зберегти карту")
                  ).pack(side=tk.LEFT, padx=4)
        tk.Button(tb, text="💾 Зберегти схему", bg="#1a4b8c", fg="white",
                  font=("Times New Roman",11),
                  command=self._save_scheme).pack(side=tk.LEFT, padx=(12,4))
        tk.Button(tb, text="📂 Відкрити схему", font=("Times New Roman",11),
                  command=self._load_scheme).pack(side=tk.LEFT, padx=4)
        cfg = self._cfg
        design_txt = HP_DESIGN_LABELS.get(cfg.get("design_used"), cfg.get("design_used",""))
        tk.Label(tb, text=f"Дизайн: {design_txt}", font=("Times New Roman",11),
                 fg="#1a4b8c").pack(side=tk.LEFT, padx=12)

        # ── Легенда кольорів + пояснення позначень (завжди повністю видимі) ──
        legend_f = tk.Frame(frame, bg="#f7f7f7", padx=8, pady=6)
        legend_f.pack(fill=tk.X)
        row1 = tk.Frame(legend_f, bg="#f7f7f7"); row1.pack(fill=tk.X)
        for role, color in HP_ROLE_COLORS.items():
            sw = tk.Frame(row1, bg=color, width=16, height=16, relief=tk.RIDGE, bd=1)
            sw.pack(side=tk.LEFT, padx=(0,4), pady=2); sw.pack_propagate(False)
            tk.Label(row1, text=HP_ROLE_LABELS[role], bg="#f7f7f7",
                     font=("Times New Roman",9)).pack(side=tk.LEFT, padx=(0,14))
        tk.Label(legend_f, text=self._map_legend_note_text(),
                 bg="#f7f7f7", fg="#444", font=("Times New Roman",9),
                 anchor="w", justify="left", wraplength=1360
                 ).pack(fill=tk.X, pady=(4,0))

        self._map_legend_decode(frame)

        map_outer = tk.Frame(frame); map_outer.pack(fill=tk.BOTH, expand=True)
        self._map_outer = map_outer
        self._draw_map()

    # ── панель: список облікових рослин ──────────────────
    def _build_list_panel(self, frame):
        for w in frame.winfo_children(): w.destroy()
        tb = tk.Frame(frame, padx=6, pady=5); tb.pack(fill=tk.X)
        tk.Button(tb, text="📋 Копіювати список", font=("Times New Roman",11),
                  command=self._copy_list).pack(side=tk.LEFT, padx=4)

        r_vsb = ttk.Scrollbar(frame, orient="vertical"); r_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.list_txt = tk.Text(frame, font=("Courier New",10),
                                yscrollcommand=r_vsb.set, state="disabled", wrap="none")
        self.list_txt.pack(fill=tk.BOTH, expand=True)
        r_vsb.config(command=self.list_txt.yview)
        self._fill_list()

    def _copy_list(self):
        self.list_txt.configure(state="normal")
        text = self.list_txt.get("1.0", tk.END)
        self.list_txt.configure(state="disabled")
        self._res_win.clipboard_clear(); self._res_win.clipboard_append(text)
        messagebox.showinfo("", "Список скопійовано у буфер обміну.")

    # ── карта саду ────────────────────────────────────────
    def _draw_map(self):
        for w in self._map_outer.winfo_children(): w.destroy()
        if not HAS_MPL or self._result is None: return
        result = self._result; cfg = self._cfg
        rows = sorted({p.row for p in result["plants"]})
        max_pos = max((p.position for p in result["plants"]), default=1)

        fig = Figure(figsize=(max(8, max_pos*0.4), max(4, len(rows)*0.55)), dpi=100)
        ax = fig.add_subplot(111)
        by_row = {}
        for p in result["plants"]:
            by_row.setdefault(p.row, {})[p.position] = p

        for ri, row_num in enumerate(rows):
            ax.text(-0.6, len(rows)-ri-0.5, f"Ряд {row_num}",
                    ha="right", va="center", fontsize=8, fontfamily="Times New Roman")
            for pos in range(1, max_pos+1):
                p = by_row.get(row_num, {}).get(pos)
                if p is None: continue
                color = HP_ROLE_COLORS.get(p.role, "#FFFFFF")
                rect = matplotlib.patches.FancyBboxPatch(
                    (pos-0.95, len(rows)-ri-0.95), 0.9, 0.9,
                    boxstyle="round,pad=0.02", facecolor=color, edgecolor="#666", linewidth=0.6)
                ax.add_patch(rect)
                label = self._plant_map_label(p)
                if label:
                    ax.text(pos-0.5, len(rows)-ri-0.5, label, ha="center", va="center",
                            fontsize=7, fontfamily="Times New Roman")
        ax.set_xlim(-1.2, max_pos+0.5); ax.set_ylim(-0.5, len(rows)+0.5)
        ax.axis("off")
        design_txt = HP_DESIGN_LABELS.get(cfg.get("design_used"), cfg.get("design_used",""))
        trait_txt = f"{cfg.get('trait_name','')} ({cfg.get('trait_unit') or '—'})  |  " \
                    if cfg.get('trait_name') else ""
        ax.set_title(
            f"{trait_txt}"
            f"CV%={result['final_cv_pct']:.2f}  |  Повторностей: {result['plots_formed']}  |  "
            f"Дизайн: {design_txt}  |  "
            f"Ітерацій: {result['iterations_used']} "
            f"({'збіжність' if result['converged'] else 'без збіжності'})",
            fontsize=9, fontfamily="Times New Roman")

        fig.tight_layout()
        self._map_fig = fig
        embed_figure(fig, self._map_outer)

    def _fill_list(self):
        if self._result is None: return
        recorded = sorted(
            [p for p in self._result["plants"] if p.role == HP_ROLE_RECORDED],
            key=lambda p: (p.variant or 0, p.replication or 0, p.row, p.position))
        cfg = self._cfg
        design_txt = HP_DESIGN_LABELS.get(cfg.get("design_used"), cfg.get("design_used",""))
        lines = []
        if cfg.get("trait_name"):
            lines.append(f"Показник: {cfg['trait_name']} ({cfg.get('trait_unit') or '—'})")
        lines += [
            f"Дизайн експерименту: {design_txt}",
            f"CV% фінальний: {self._result['final_cv_pct']:.2f}   "
            f"Повторностей: {self._result['plots_formed']}   "
            f"Ітерацій: {self._result['iterations_used']}",
        ]
        lines.extend(self._list_extra_header_lines())
        lines += [
            "-"*90,
            f"{'Варіант':<26}{'Повт.':<8}{'Ряд':<6}{'Позиція':<9}{cfg.get('trait_name','Значення')}",
            "-"*90,
        ]
        for p in recorded:
            vn = self._plant_list_label(p)
            lines.append(f"{vn:<26}{p.replication or '-':<8}"
                        f"{p.row:<6}{p.position:<9}{p.value:.2f}" if p.value is not None
                        else f"{vn:<26}{p.replication or '-':<8}{p.row:<6}{p.position:<9}—")
        self.list_txt.configure(state="normal")
        self.list_txt.delete("1.0", tk.END)
        self.list_txt.insert("1.0", "\n".join(lines))
        self.list_txt.configure(state="disabled")

    # ── панель: бланк обліку (для друку й заповнення в полі) ──
    # Скільки позицій ряду і скільки рядів вміщується на одній сторінці,
    # щоб комірки лишались достатньо великими для запису значень від руки.
    FORM_POS_PER_PAGE = 14
    FORM_ROWS_PER_PAGE = 4

    def _build_form_panel(self, frame):
        for w in frame.winfo_children(): w.destroy()
        tb = tk.Frame(frame, padx=6, pady=5); tb.pack(fill=tk.X)
        tk.Button(tb, text="💾 Зберегти PNG", font=("Times New Roman",11),
                  command=self._save_form_page).pack(side=tk.LEFT, padx=4)

        nav = tk.Frame(tb); nav.pack(side=tk.LEFT, padx=12)
        tk.Button(nav, text="◀ Попередня", font=("Times New Roman",10),
                  command=lambda: self._form_page_step(-1)).pack(side=tk.LEFT, padx=2)
        self._form_page_lbl = tk.Label(nav, text="", font=("Times New Roman",11,"bold"))
        self._form_page_lbl.pack(side=tk.LEFT, padx=8)
        tk.Button(nav, text="Наступна ▶", font=("Times New Roman",10),
                  command=lambda: self._form_page_step(1)).pack(side=tk.LEFT, padx=2)

        tk.Label(tb, text="Роздрукуйте потрібні сторінки й носіть із собою в сад — "
                          "впишіть виміряні значення прямо на бланк.",
                 font=("Times New Roman",10), fg="#555").pack(side=tk.LEFT, padx=12)

        form_outer = tk.Frame(frame); form_outer.pack(fill=tk.BOTH, expand=True)
        self._form_outer = form_outer
        self._form_pages = self._build_form_pages()
        self._form_page_idx = 0
        self._form_figs_cache = {}
        self._draw_blank_form()

    def _build_form_pages(self):
        """Розбиває фізичну карту саду на сторінки: кожен ряд ділиться на
        сегменти позицій довжиною не більше FORM_POS_PER_PAGE, а сегменти
        групуються по FORM_ROWS_PER_PAGE на одну сторінку — так комірки
        завжди лишаються великими й розбірливими для запису вручну."""
        result = self._result
        by_row = {}
        for p in result["plants"]:
            by_row.setdefault(p.row, {})[p.position] = p

        segments = []  # (row_num, [positions...])
        for row_num in sorted(by_row.keys()):
            positions = sorted(by_row[row_num].keys())
            for i in range(0, len(positions), self.FORM_POS_PER_PAGE):
                segments.append((row_num, positions[i:i+self.FORM_POS_PER_PAGE]))

        pages = []
        for i in range(0, len(segments), self.FORM_ROWS_PER_PAGE):
            pages.append(segments[i:i+self.FORM_ROWS_PER_PAGE])
        return pages or [[]]

    def _form_page_step(self, delta):
        n = len(self._form_pages)
        self._form_page_idx = max(0, min(n-1, self._form_page_idx + delta))
        self._draw_blank_form()

    def _draw_blank_form(self):
        for w in self._form_outer.winfo_children(): w.destroy()
        if not HAS_MPL or self._result is None: return
        if self._form_page_idx in self._form_figs_cache:
            fig = self._form_figs_cache[self._form_page_idx]
        else:
            fig = self._render_form_page(self._form_pages[self._form_page_idx])
            self._form_figs_cache[self._form_page_idx] = fig
        self._form_fig = fig
        n = len(self._form_pages)
        self._form_page_lbl.configure(text=f"Сторінка {self._form_page_idx+1} / {n}")
        embed_figure(fig, self._form_outer)

    def _render_form_page(self, segments):
        result = self._result
        by_row = {}
        for p in result["plants"]:
            by_row.setdefault(p.row, {})[p.position] = p

        n_seg = max(1, len(segments))
        max_len = max((len(pos_list) for _, pos_list in segments), default=1)
        fig = Figure(figsize=(max(9, max_len*0.85), max(4.5, n_seg*2.3+1.2)), dpi=100)
        ax = fig.add_subplot(111)

        ROW_H = 2.1  # висота одного сегмента ряду (мітка+значення+відступ)
        for si, (row_num, pos_list) in enumerate(segments):
            y_top = n_seg*ROW_H - si*ROW_H
            first_pos, last_pos = pos_list[0], pos_list[-1]
            ax.text(-0.7, y_top-1.0,
                    f"Ряд {row_num}\n(поз. {first_pos}-{last_pos})",
                    ha="right", va="center", fontsize=9, fontfamily="Times New Roman",
                    fontweight="bold")
            for ci, pos in enumerate(pos_list):
                p = by_row.get(row_num, {}).get(pos)
                if p is None: continue
                x = ci
                # ── порядкова нумерація позицій (для орієнтації в саду) ──
                ax.text(x+0.5, y_top+0.15, str(pos), ha="center", va="center",
                        fontsize=8, fontfamily="Times New Roman", color="#555")
                if p.role == HP_ROLE_RECORDED:
                    # верхній квадрат — мітка варіанту й повторення
                    top_r = matplotlib.patches.Rectangle(
                        (x+0.03, y_top-0.72), 0.94, 0.55,
                        facecolor="#EAF2FB", edgecolor="#333", linewidth=1.0)
                    ax.add_patch(top_r)
                    ax.text(x+0.5, y_top-0.44, self._plant_form_label(p),
                            ha="center", va="center", fontsize=8,
                            fontfamily="Times New Roman", color="#1a4b8c", fontweight="bold")
                    # нижній квадрат — порожній, для запису значення
                    bot_r = matplotlib.patches.Rectangle(
                        (x+0.03, y_top-1.55), 0.94, 0.78,
                        facecolor="white", edgecolor="#333", linewidth=1.2)
                    ax.add_patch(bot_r)
                elif p.role == HP_ROLE_EXTRA:
                    rect = matplotlib.patches.Rectangle(
                        (x+0.12, y_top-1.2), 0.76, 1.0,
                        facecolor="#eeeeee", edgecolor="#bbb", linewidth=0.6)
                    ax.add_patch(rect)
                    ax.text(x+0.5, y_top-0.7, "x", ha="center", va="center",
                            fontsize=9, color="#999")
                else:
                    label = {HP_ROLE_GUARD_EDGE:"K", HP_ROLE_GUARD_REP:"P",
                             HP_ROLE_DEAD:"-", HP_ROLE_POLLINIZER:"+"}.get(p.role, "")
                    rect = matplotlib.patches.Rectangle(
                        (x+0.12, y_top-1.2), 0.76, 1.0,
                        facecolor="#eeeeee", edgecolor="#bbb", linewidth=0.6)
                    ax.add_patch(rect)
                    if label:
                        ax.text(x+0.5, y_top-0.7, label, ha="center", va="center",
                                fontsize=8, color="#999", fontfamily="Times New Roman")

        ax.set_xlim(-1.6, max_len+0.5); ax.set_ylim(0, n_seg*ROW_H+1.3)
        ax.axis("off")
        pg_i = self._form_page_idx+1 if hasattr(self, "_form_page_idx") else 1
        pg_n = len(self._form_pages) if hasattr(self, "_form_pages") else 1
        ax.set_title(
            "БЛАНК ОБЛІКУ\n"
            "Показник: _______________________   Одиниця: _________\n"
            f"Дата: _______________     Виконав: _______________________     "
            f"Сторінка {pg_i}/{pg_n}",
            fontsize=10, fontfamily="Times New Roman", loc="left")
        fig.subplots_adjust(top=0.82, bottom=0.03, left=0.1, right=0.98)
        return fig

    def _save_form_page(self):
        if getattr(self, "_form_fig", None) is None:
            messagebox.showwarning("","Спочатку згенеруйте план."); return
        default_name = f"blank_form_page_{self._form_page_idx+1}.png"
        path = filedialog.asksaveasfilename(
            defaultextension=".png", initialfile=default_name,
            filetypes=[("PNG зображення","*.png")],
            title=f"Зберегти сторінку {self._form_page_idx+1}")
        if not path: return
        try:
            self._form_fig.savefig(path, dpi=150, bbox_inches="tight")
            messagebox.showinfo("Збережено", f"Збережено:\n{path}")
        except Exception as ex:
            messagebox.showerror("Помилка", str(ex))

    # ── збереження ────────────────────────────────────────
    def _save_png(self, fig=None, title="Зберегти зображення"):
        fig = fig if fig is not None else self._map_fig
        if self._result is None or fig is None:
            messagebox.showwarning("","Спочатку згенеруйте план."); return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                    filetypes=[("PNG зображення","*.png")], title=title)
        if not path: return
        try:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            messagebox.showinfo("Збережено", f"Збережено:\n{path}")
        except Exception as ex:
            messagebox.showerror("Помилка", str(ex))


class HomogeneousPlotWindow(HPResultsViewMixin):
    """
    Планування досліду за однорідністю рослин.

    Користувач завантажує сітку "ряд × позиція" з обраним показником
    (діаметр штамбу, врожайність минулого року тощо), позначаючи випади
    "-" і запилювачів "+". Програма ітеративно відбирає однорідні (за
    заданим CV%) рослини, формує ділянки з дотриманням ДВОХ типів
    захисних зон, застосовує обраний дизайн експерименту (CRD/RCBD/
    латинський квадрат) і будує кольорову карту саду з друком у PNG.
    """

    HELP_TEXT = """
ПЛАНУВАННЯ ДОСЛІДУ ЗА ОДНОРІДНІСТЮ РОСЛИН — ІНСТРУКЦІЯ
══════════════════════════════════════════════════════

ДЛЯ ЧОГО ЦЕЙ МОДУЛЬ?
  На відміну від «Генератора плану польового досліду» (де ділянки
  умовні й рівноцінні), цей модуль будує схему на основі РЕАЛЬНОГО
  стану наявних рослин у вже дорослому саду/насадженні — з
  урахуванням того що дерева/кущі різняться за силою розвитку.

  Це комп'ютеризована реалізація класичного агрономічного принципу
  вирівнювання дослідної ділянки.

КРОК 1. ОБРАНИЙ ПОКАЗНИК
  Показник для оцінки однорідності — БУДЬ-ЯКИЙ кількісний (не лише
  діаметр штамбу): об'єм крони, урожайність минулого року тощо.
  Вкажіть його назву та одиницю виміру.

КРОК 2. ТАБЛИЦЯ "РЯД × ПОЗИЦІЯ" (до 100 позицій у ряду)
  Рядки таблиці = ряди саду. Стовпці = позиції рослин уздовж ряду.
  У кожну клітинку введіть:
    • число — значення показника цієї рослини;
    • "-"   — випад / пошкоджена рослина;
    • "+"   — рослина-запилювач за схемою посадки.

КРОК 3. ДВА ТИПИ ЗАХИСНИХ РОСЛИН
  Повторність = група з "Повторність, рослин" облікових рослин ОДНОГО
  варіанту. Варіант складається з кількох таких повторностей (задається
  полем "Повторень"), розкиданих рандомізовано по саду.
  Тип 1 — «Захисна зона краю ряду»: 1–3 рослини на самому початку і в
          самому кінці КОЖНОГО ряду (ізолює дослід від сусідніх посадок).
  Тип 2 — «Захисна зона між повтореннями»: рослини, що вставляються
          ПІСЛЯ КОЖНОЇ сформованої повторності — тобто між будь-якими
          двома сусідніми повторностями в ряду. Оскільки повторності
          розкидаються рандомізовано (сусідами можуть опинитись як дві
          повторності одного варіанту, так і різних), захист потрібен
          між кожною парою, без винятку.
  Кількість рослин для кожного типу задає користувач окремо.

КРОК 4. АЛГОРИТМ ФОРМУВАННЯ ПОВТОРНОСТЕЙ (виконується автоматично)
  1) Пропускається захисна зона краю ряду (тип 1);
  2) уздовж ряду набирається задана кількість ОБЛІКОВИХ рослин —
     одна повторність (непридатні позиції пропускаються "прозоро");
  3) одразу після неї вставляється захисна зона між повтореннями (тип 2);
  4) цикл повторюється до кінця ряду, потім захисна зона краю ряду
     (тип 1) в кінці, далі — наступний ряд;
  5) CV% рахується лише за обліковими рослинами (захисні виключені);
  6) якщо CV% більший за поріг — допустимий діапазон значень
     звужується навколо середнього, і схема будується заново;
  7) зупинка — коли склад облікової вибірки перестає змінюватись,
     CV% досягнуто, або вичерпано ліміт ітерацій (типово 20).

КРОК 5. ДИЗАЙН ЕКСПЕРИМЕНТУ (застосовується ПІСЛЯ формування повторностей)
  Коли облікові повторності й рослини вже визначені за варіабельністю,
  програма САМА розкидає мітки варіантів по повторностях згідно обраного
  дизайну — готова польова карта формується автоматично:
    • CRD (повна рандомізація) — варіанти розкидані цілком випадково
      по всіх повторностях, без блокової структури;
    • RCBD (рандомізовані повні блоки) — кожні n_var послідовно
      сформованих повторностей групуються в один блок, і в межах
      кожного блоку варіанти рандомізуються (по одній повторності
      кожного варіанту в блоці);
    • Латинський квадрат — потребує РІВНО n_var таких блоків;
      якщо їх сформовано більше або менше — програма попередить і
      автоматично використає RCBD замість нього.
  Seed рандомізації — див. окрему довідку: однакове число = однакова
  схема (для відтворюваності), зберігайте його в документації.

КРОК 6. РЕЗУЛЬТАТ
  Натисніть «▶ Згенерувати план» — карта й список відкриються в
  окремому вікні (як і в усіх інших видах аналізу).
  Кольорова карта саду (зелений — обліково, жовтий — захисна зона
  краю ряду, помаранчевий — захисна зона між повтореннями, сірий —
  виключено за CV, темно-сірий — випад, синій — запилювач).
  Список облікових рослин за варіантами/повтореннями.
  Попередження, якщо повторностей сформовано менше ніж потрібно
  (варіантів × повторень) — послабте поріг CV% або розширте вибірку.

ЗБЕРЕЖЕННЯ
  Карту можна зберегти як зображення (PNG) — для друку чи вставки в
  звіт. Список облікових рослин можна скопіювати в буфер обміну.
"""

    def __init__(self, parent, gs=None):
        self.win = tk.Toplevel(parent)
        self.win.title("Планування досліду за однорідністю рослин")
        self.win.geometry("1400x820"); set_icon(self.win)
        self.gs = dict(gs) if gs else {}
        self._result = None
        self._map_fig = None
        self.row_lengths = []   # довжина (к-сть рослин) кожного ряду — окремо для кожного
        self.rows_n = 0
        self.entries = []
        self.row_labels = []
        self.pos_labels = []
        self._table_built = False
        self._build()

    # ─────────────────────────────────────────────────────
    def _build(self):
        rf = ("Times New Roman", 11)

        top = tk.Frame(self.win, padx=8, pady=6); top.pack(fill=tk.X)
        tk.Button(top, text="▶ Згенерувати план", bg="#c62828", fg="white",
                  font=("Times New Roman", 13),
                  command=self._run).pack(side=tk.LEFT, padx=4)

        mb2 = tk.Menubutton(top, text="⚙ Таблиця ▾", font=rf, relief=tk.RAISED, bd=2)
        mb2.pack(side=tk.LEFT, padx=4)
        sm = tk.Menu(mb2, tearoff=0)
        sm.add_command(label="Додати ряд",     command=self._add_row)
        sm.add_command(label="Видалити ряд",   command=self._del_row)
        sm.add_separator()
        sm.add_command(label="Додати позицію",   command=self._add_col)
        sm.add_command(label="Видалити позицію", command=self._del_col)
        sm.add_separator()
        sm.add_command(label="🗑 Очистити таблицю", command=self._clear_table)
        sm.add_separator()
        sm.add_command(label="💾 Зберегти проект", command=self._save_proj)
        sm.add_command(label="📂 Відкрити проект", command=self._load_proj)
        mb2["menu"] = sm

        tk.Button(top, text="Вставити з буфера", font=rf,
                  command=self._paste).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="📚 Довідка", bg="#1a4b8c", fg="white", font=rf,
                  command=self._show_help).pack(side=tk.LEFT, padx=4)
        self._resize_btn = tk.Button(top, text="🔧 Змінити розміри таблиці", font=rf,
                                     command=self._reset_table_size)
        # ще не запакована — з'явиться лише після побудови таблиці

        # ── параметри ────────────────────────────────────
        pf = tk.LabelFrame(self.win, text="Параметри", font=("Times New Roman",11,"bold"),
                           padx=8, pady=6)
        pf.pack(fill=tk.X, padx=8, pady=(0,4))

        self._v = {}
        row_defs = [
            ("Показник:",                       "trait_name", "", 12),
            ("Одиниця:",                        "trait_unit", "", 5),
            ("Поріг CV, %:",                     "cv_thr",     "", 5),
            ("Повторність, рослин:",             "plot_size",  "", 5),
            ("Захисна край ряду (1-3):",         "edge_guard", "", 4),
            ("Захисна між повтор.:",             "rep_guard",  "", 4),
            ("Варіантів:",                       "n_var",      "", 4),
            ("Повторень:",                       "n_rep",      "", 4),
            ("Макс. ітерацій:",                  "max_it",     "20", 5),
        ]
        PER_ROW = 4   # скільки пар "підпис+поле" вміщується в один рядок без переповнення
        for idx, (lbl, key, default, w) in enumerate(row_defs):
            r, c = divmod(idx, PER_ROW)
            tk.Label(pf, text=lbl, font=rf).grid(
                row=r, column=c*2, sticky="w", padx=(0 if c==0 else 14, 2), pady=3)
            v = tk.StringVar(value=default); self._v[key] = v
            tk.Entry(pf, textvariable=v, width=w, font=rf).grid(
                row=r, column=c*2+1, sticky="w", pady=3)

        next_row = (len(row_defs) - 1) // PER_ROW + 1

        tk.Label(pf, text="Дизайн експерименту:", font=rf).grid(
            row=next_row, column=0, sticky="w", padx=(0,2), pady=(6,3))
        self._design_v = tk.StringVar(value=HP_DESIGNS[1][0])  # RCBD за замовчуванням
        ttk.Combobox(pf, textvariable=self._design_v, state="readonly", width=26,
                     values=[lbl for lbl,_ in HP_DESIGNS]
                     ).grid(row=next_row, column=1, columnspan=3, sticky="w", pady=(6,3))

        tk.Label(pf, text="Seed рандомізації:", font=rf).grid(
            row=next_row, column=4, sticky="w", padx=(14,2), pady=(6,3))
        self._v["seed"] = tk.StringVar(value="1")
        tk.Entry(pf, textvariable=self._v["seed"], width=6, font=rf).grid(
            row=next_row, column=5, sticky="w", pady=(6,3))

        next_row += 1
        self._dead_guard = tk.BooleanVar(value=True)
        self._poll_guard = tk.BooleanVar(value=False)
        tk.Checkbutton(pf, text='Враховувати "-" (випади) як захисні',
                       variable=self._dead_guard, font=rf
                       ).grid(row=next_row, column=0, columnspan=4, sticky="w", pady=(3,0))
        tk.Checkbutton(pf, text='Враховувати "+" (запилювачі) як захисні',
                       variable=self._poll_guard, font=rf
                       ).grid(row=next_row, column=4, columnspan=4, sticky="w", pady=(3,0))

        next_row += 1
        self._variant_names = []
        tk.Button(pf, text="📝 Задати назви варіантів", font=rf,
                  command=self._edit_variant_names
                  ).grid(row=next_row, column=0, columnspan=3, sticky="w", pady=(8,0))
        self._varnames_status = tk.Label(pf, text="(назви не задано — на карті буде В1, В2…)",
                                         font=("Times New Roman",9), fg="#888")
        self._varnames_status.grid(row=next_row, column=3, columnspan=5, sticky="w", pady=(8,0))

        # ── таблиця даних — на всю ширину вікна ────────────
        tbl_lbl_frm = tk.Frame(self.win); tbl_lbl_frm.pack(fill=tk.X, padx=8)
        tk.Label(tbl_lbl_frm, text='Таблиця "ряд × позиція"  (число / "-" / "+")',
                 font=("Times New Roman",10,"bold")).pack(anchor="w")

        # ── Крок 1: розміри таблиці — довжина КОЖНОГО ряду окремо ──
        self._setup_frame = tk.LabelFrame(self.win,
            text="Розміри таблиці — вкажіть, скільки рослин у кожному ряду",
            font=("Times New Roman",11,"bold"), padx=10, pady=8)
        self._setup_frame.pack(fill=tk.X, padx=8, pady=(0,4))

        setup_top = tk.Frame(self._setup_frame); setup_top.pack(fill=tk.X)
        tk.Label(setup_top, text="Кількість рядів саду:", font=rf).pack(side=tk.LEFT)
        self._n_rows_setup_var = tk.StringVar(value="")
        tk.Entry(setup_top, textvariable=self._n_rows_setup_var, width=6, font=rf
                 ).pack(side=tk.LEFT, padx=6)
        tk.Button(setup_top, text="Задати довжину кожного ряду →", font=rf,
                  command=self._build_row_length_inputs).pack(side=tk.LEFT, padx=10)

        self._rowlen_holder = tk.Frame(self._setup_frame)
        self._rowlen_holder.pack(fill=tk.X, pady=(8,0))

        # ── Область таблиці — заповнюється після кроку 1 ──────
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

        self._status_lbl = tk.Label(self.win, text="", fg="#B71C1C",
                                    font=("Times New Roman",10), wraplength=1360,
                                    justify="left", anchor="w")
        self._status_lbl.pack(fill=tk.X, padx=8, pady=(0,4))

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
        PER_ROW = 8
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
        tk.Label(btn_f, text="  Довжина ряду = кількість рослин у ньому "
                             "(порожні клітинки в кінці ряду не потрібні — просто вкажіть "
                             "реальну кількість)",
                 font=("Times New Roman",9), fg="#666").pack(side=tk.LEFT, padx=8)

    def _build_data_table(self):
        try:
            lengths = [int(v.get()) for v in self._rowlen_vars]
            if any(L < 1 for L in lengths): raise ValueError
        except ValueError:
            messagebox.showwarning("", "Вкажіть довжину (кількість рослин) для кожного "
                                       "ряду — додатне ціле число, без порожніх полів."); return

        if self._table_built and self.entries:
            if not messagebox.askyesno("Перебудувати таблицю",
                    "У таблиці вже є введені дані — перебудова розмірів видалить їх. "
                    "Продовжити?"):
                return

        for w in self.inner.winfo_children(): w.destroy()
        self.row_lengths = lengths
        self.rows_n = len(lengths)
        max_len = max(lengths)

        tk.Label(self.inner, text="Ряд \\ Поз.", width=9, relief=tk.RIDGE,
                 bg="#444444", fg="white", font=("Times New Roman",10,"bold")
                 ).grid(row=0, column=0, padx=1, pady=1, sticky="nsew")
        self.pos_labels = []
        for j in range(max_len):
            lbl = tk.Label(self.inner, text=str(j+1), width=5, relief=tk.RIDGE,
                           bg="#1a4b8c", fg="white", font=("Times New Roman",9,"bold"))
            lbl.grid(row=0, column=j+1, padx=1, pady=1, sticky="nsew")
            self.pos_labels.append(lbl)

        self.row_labels = []
        self.entries = []
        for i, L in enumerate(lengths):
            rl = tk.Label(self.inner, text=f"Ряд {i+1}", width=9, relief=tk.RIDGE,
                         bg="#444444", fg="white", font=("Times New Roman",9,"bold"))
            rl.grid(row=i+1, column=0, padx=1, pady=1, sticky="nsew")
            self.row_labels.append(rl)
            row_e = []
            for j in range(L):
                e = tk.Entry(self.inner, width=5, font=("Times New Roman",10))
                e.grid(row=i+1, column=j+1, padx=1, pady=1)
                row_e.append(e)
            self.entries.append(row_e)
        _bind_nav(self.entries, self.win)
        _bind_fill_handle(self.entries, self.win)

        self._table_built = True
        self._setup_frame.pack_forget()
        self._resize_btn.pack(side=tk.LEFT, padx=4)

    def _extend_row_to(self, ri, target_len):
        """Дорощує конкретний ряд ri до target_len позицій (додає й заголовки
        позицій, якщо потрібно — інші ряди на це не впливає)."""
        while len(self.entries[ri]) < target_len:
            j = len(self.entries[ri])
            if j >= len(self.pos_labels):
                lbl = tk.Label(self.inner, text=str(j+1), width=5, relief=tk.RIDGE,
                               bg="#1a4b8c", fg="white", font=("Times New Roman",9,"bold"))
                lbl.grid(row=0, column=j+1, padx=1, pady=1, sticky="nsew")
                self.pos_labels.append(lbl)
            e = tk.Entry(self.inner, width=5, font=("Times New Roman",10))
            e.grid(row=ri+1, column=j+1, padx=1, pady=1)
            self.entries[ri].append(e)
        if ri < len(self.row_lengths):
            self.row_lengths[ri] = len(self.entries[ri])

    def _add_row_silent(self, length=1):
        """Додає новий ряд без діалогового вікна (для вставки з буфера/завантаження)."""
        i = self.rows_n
        rl = tk.Label(self.inner, text=f"Ряд {i+1}", width=9, relief=tk.RIDGE,
                     bg="#444444", fg="white", font=("Times New Roman",9,"bold"))
        rl.grid(row=i+1, column=0, padx=1, pady=1, sticky="nsew")
        self.row_labels.append(rl)
        self.entries.append([])
        self.row_lengths.append(0)
        self.rows_n += 1
        self._extend_row_to(i, length)

    def _reset_table_size(self):
        if self.entries and not messagebox.askyesno("Змінити розміри таблиці",
                "Це відкриє налаштування розмірів заново. Поточні дані таблиці "
                "будуть втрачені при побудові нової. Продовжити?"):
            return
        self._resize_btn.pack_forget()
        self._setup_frame.pack(fill=tk.X, padx=8, pady=(0,4), before=self._canvas.master)
        self._n_rows_setup_var.set(str(self.rows_n) if self.rows_n else "")

    # ─────────────────────────────────────────────────────
    def _show_help(self):
        win = tk.Toplevel(self.win); win.title("Довідка — Планування досліду")
        win.geometry("720x680"); set_icon(win)
        frm = tk.Frame(win); frm.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        vsb = ttk.Scrollbar(frm, orient="vertical"); vsb.pack(side=tk.RIGHT, fill=tk.Y)
        txt = tk.Text(frm, wrap="word", font=("Times New Roman",11),
                      yscrollcommand=vsb.set, relief=tk.FLAT, bg="#fafafa",
                      padx=10, pady=8, cursor="arrow")
        txt.pack(fill=tk.BOTH, expand=True); vsb.config(command=txt.yview)
        txt.insert("1.0", self.HELP_TEXT.strip()); txt.configure(state="disabled")
        txt.bind("<MouseWheel>", lambda e: txt.yview_scroll(int(-1*(e.delta/120)),"units"))
        tk.Button(win, text="Закрити", command=win.destroy,
                  font=("Times New Roman",11)).pack(pady=6)

    def _edit_variant_names(self):
        try:
            n_var = int(self._v["n_var"].get())
        except ValueError:
            messagebox.showwarning("", "Спочатку вкажіть кількість варіантів (число) у полі "
                                       "«Варіантів:» вище."); return
        if n_var < 2:
            messagebox.showwarning("", "Кількість варіантів має бути щонайменше 2."); return

        dlg = tk.Toplevel(self.win)
        dlg.title("Назви варіантів"); dlg.resizable(False, False)
        set_icon(dlg); dlg.grab_set()
        frm = tk.Frame(dlg, padx=16, pady=14); frm.pack()
        tk.Label(frm,
                 text="Введіть реальні назви варіантів досліду (наприклад, «Контроль»,\n"
                      "«N60P60», «Сорт Айдаред»). На карті лишаться короткі позначення\n"
                      "В1, В2… — а ці назви з'являться в легенді під картою й у списку\n"
                      "облікових рослин, щоб не заплутатись у саду.",
                 font=("Times New Roman",10), fg="#555", justify="left"
                 ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0,10))
        existing = self._variant_names
        name_vars = []
        for i in range(n_var):
            default = existing[i] if i < len(existing) else ""
            tk.Label(frm, text=f"В{i+1} =", font=("Times New Roman",11)
                     ).grid(row=i+1, column=0, sticky="w", pady=3)
            v = tk.StringVar(value=default)
            tk.Entry(frm, textvariable=v, width=32, font=("Times New Roman",11)
                     ).grid(row=i+1, column=1, sticky="w", padx=8, pady=3)
            name_vars.append(v)

        def _save():
            names = [v.get().strip() or f"Варіант {i+1}" for i, v in enumerate(name_vars)]
            self._variant_names = names
            self._varnames_status.configure(
                text="✓ " + ", ".join(f"В{i+1}={n}" for i,n in enumerate(names)),
                fg="#1a6b1a")
            dlg.destroy()
        bf = tk.Frame(frm); bf.grid(row=n_var+1, column=0, columnspan=2, pady=(12,0))
        tk.Button(bf, text="Зберегти", bg="#1a6b1a", fg="white",
                  font=("Times New Roman",11), command=_save).pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text="Скасувати", font=("Times New Roman",11),
                  command=dlg.destroy).pack(side=tk.LEFT)
        center_win(dlg)

    # ── управління таблицею ──────────────────────────────
    def _add_row(self):
        if not self._table_built:
            messagebox.showinfo("", "Спочатку побудуйте таблицю (крок 1 вгорі)."); return
        default_len = self.row_lengths[-1] if self.row_lengths else 20
        L = simpledialog.askinteger("Новий ряд", "Кількість рослин у новому ряду:",
                                    parent=self.win, initialvalue=default_len, minvalue=1)
        if not L: return
        self._add_row_silent(L)

    def _del_row(self):
        if not self.entries: return
        for e in self.entries.pop(): e.destroy()
        self.row_labels.pop().destroy()
        self.rows_n -= 1
        if self.row_lengths: self.row_lengths.pop()

    def _add_col(self):
        """Додає одну позицію в кінець того ряду, на клітинку якого зараз
        встановлено фокус (кожен ряд має свою довжину, тому глобального
        «додати стовпець одразу всім» більше немає сенсу)."""
        w = self.win.focus_get()
        ri = None
        for i, row_e in enumerate(self.entries):
            if w in row_e: ri = i; break
        if ri is None:
            messagebox.showinfo("Оберіть ряд",
                "Клацніть спочатку на будь-яку клітинку потрібного ряду — "
                "позицію буде додано саме до нього."); return
        self._extend_row_to(ri, len(self.entries[ri])+1)
        _bind_nav(self.entries, self.win)
        _bind_fill_handle(self.entries, self.win)

    def _del_col(self):
        """Видаляє останню позицію того ряду, де зараз фокус."""
        w = self.win.focus_get()
        ri = None
        for i, row_e in enumerate(self.entries):
            if w in row_e: ri = i; break
        if ri is None:
            messagebox.showinfo("Оберіть ряд",
                "Клацніть спочатку на будь-яку клітинку потрібного ряду."); return
        if len(self.entries[ri]) <= 1: return
        self.entries[ri].pop().destroy()
        self.row_lengths[ri] -= 1

    def _clear_table(self):
        if not messagebox.askyesno("Очистити", "Видалити всі дані таблиці?"): return
        for row in self.entries:
            for e in row: e.delete(0, tk.END)

    def _save_proj(self):
        generic_save_project(self.win, "homogeneous_plot", None, self.entries,
                             extra={"trait_name": self._v["trait_name"].get()
                                    if hasattr(self, "_v") and "trait_name" in self._v else "",
                                   "variant_names": self._variant_names})

    def _load_proj(self):
        d = generic_load_project(self.win)
        if d is None: return
        rd = d.get("rows_data", [])
        if not self._table_built:
            # Таблиці ще нема — створюємо одразу під розмір даних із файлу
            for w in self.inner.winfo_children(): w.destroy()
            self.entries = []; self.row_labels = []; self.pos_labels = []
            self.rows_n = 0; self.row_lengths = []
            tk.Label(self.inner, text="Ряд \\ Поз.", width=9, relief=tk.RIDGE,
                     bg="#444444", fg="white", font=("Times New Roman",10,"bold")
                     ).grid(row=0, column=0, padx=1, pady=1, sticky="nsew")
            self._table_built = True
            self._setup_frame.pack_forget()
            self._resize_btn.pack(side=tk.LEFT, padx=4)
        while len(self.entries) < len(rd): self._add_row_silent(1)
        for i, rv in enumerate(rd):
            self._extend_row_to(i, len(rv))
            for j, v in enumerate(rv):
                self.entries[i][j].delete(0, tk.END); self.entries[i][j].insert(0, v)
        vnames = d.get("variant_names")
        if vnames:
            self._variant_names = vnames
            self._varnames_status.configure(
                text="✓ " + ", ".join(f"В{i+1}={n}" for i,n in enumerate(vnames)),
                fg="#1a6b1a")
        _bind_nav(self.entries, self.win)
        _bind_fill_handle(self.entries, self.win)

    def _save_scheme(self):
        """Зберігає ЗГЕНЕРОВАНУ схему (карту рослин з роллю/варіантом/
        повторенням кожної) — саме те, що потрібно для подальшого ведення
        польового журналу обліків. Формат той самий .sadp (JSON), що й у
        решти проектів програми, з type="homogeneous_plot_scheme" — при
        відкритті програма перевіряє саме це поле, а не назву файлу, тож
        файл можна перейменувати без ризику."""
        if self._result is None:
            messagebox.showwarning("", "Спочатку згенеруйте план (▶ Згенерувати план)."); return
        plants_data = []
        for p in self._result["plants"]:
            plants_data.append({
                "row": p.row, "position": p.position, "value": p.value,
                "status": p.status, "role": p.role, "plot_id": p.plot_id,
                "variant": p.variant, "replication": p.replication,
            })
        d = {
            "type": "homogeneous_plot_scheme", "version": APP_VER,
            "cfg": self._cfg, "variant_names": self._variant_names,
            "plants": plants_data,
            "final_cv_pct": self._result.get("final_cv_pct"),
            "plots_formed": self._result.get("plots_formed"),
            "iterations_used": self._result.get("iterations_used"),
            "warnings": self._result.get("warnings", []),
        }
        default_name = "схема_" + (self._cfg.get("trait_name","план").replace(" ","_")) + ".sadp"
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
                "Цей файл можна відкрити пізніше тут само (📂 Відкрити схему), "
                "або в модулі «Польовий журнал обліків», щоб вносити фактичні "
                "виміряні значення прямо в цю схему протягом сезонів.")
        except Exception as ex:
            messagebox.showerror("Помилка збереження", str(ex))

    def _load_scheme(self):
        path = filedialog.askopenfilename(
            parent=self.win, filetypes=[("SAD схема","*.sadp"),("JSON","*.json")],
            title="Відкрити схему досліду")
        if not path: return
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception as ex:
            messagebox.showerror("Помилка відкриття", str(ex)); return
        if d.get("type") != "homogeneous_plot_scheme":
            messagebox.showwarning("Не той тип файлу",
                "Цей файл не є збереженою схемою досліду "
                "(«💾 Зберегти схема»). Можливо, це звичайний проект "
                "із вхідними даними («💾 Зберегти проект») — його слід "
                "відкривати через «📂 Відкрити проект» у меню «⚙ Таблиця»."); return

        plants = []
        for pd in d.get("plants", []):
            p = HPPlant(pd["row"], pd["position"], pd.get("value"), pd.get("status","ok"))
            p.role = pd.get("role", HP_ROLE_UNASSIGNED)
            p.plot_id = pd.get("plot_id")
            p.variant = pd.get("variant")
            p.replication = pd.get("replication")
            plants.append(p)

        self._result = {
            "plants": plants,
            "final_cv_pct": d.get("final_cv_pct", 0),
            "plots_formed": d.get("plots_formed", 0),
            "iterations_used": d.get("iterations_used", 0),
            "warnings": d.get("warnings", []),
        }
        self._cfg = d.get("cfg", {})
        self._variant_names = d.get("variant_names", [])
        self._show_results()
        messagebox.showinfo("Відкрито", f"Схему завантажено:\n{path}")

    def _paste(self):
        if not self._table_built:
            messagebox.showinfo("", "Спочатку побудуйте таблицю (крок 1 вгорі)."); return
        try: data = self.win.clipboard_get()
        except Exception:
            messagebox.showwarning("Буфер порожній",
                "Скопіюйте дані з Excel (Ctrl+C) і спробуйте знову."); return
        if not data.strip(): return
        pos = (0,0); w = self.win.focus_get()
        if isinstance(w, tk.Entry):
            for i, row_ in enumerate(self.entries):
                for j, e in enumerate(row_):
                    if e is w: pos=(i,j); break
        r0, c0 = pos
        for ir, line in enumerate(data.splitlines()):
            if line == "" and not line.strip(): continue
            ri = r0 + ir
            while ri >= len(self.entries): self._add_row_silent(1)
            vals = line.split("\t")
            self._extend_row_to(ri, c0 + len(vals))
            for jc, val in enumerate(vals):
                cc = c0+jc
                self.entries[ri][cc].delete(0, tk.END)
                self.entries[ri][cc].insert(0, val.strip())
        _bind_nav(self.entries, self.win)
        _bind_fill_handle(self.entries, self.win)

    # ── побудова схеми ────────────────────────────────────
    def _run(self):
        if not self._table_built or not self.entries:
            messagebox.showwarning("Таблиця не побудована",
                "Спочатку задайте розміри таблиці (крок 1 вгорі) і заповніть дані."); return
        trait_name = self._v["trait_name"].get().strip()
        if not trait_name:
            messagebox.showwarning("Не вказано показник",
                "Вкажіть назву показника, за яким оцінюється однорідність\n"
                "(наприклад: діаметр штамбу, об'єм крони, урожайність минулого року)."); return
        trait_unit = self._v["trait_unit"].get().strip()
        if not trait_unit:
            messagebox.showwarning("Не вказано одиницю виміру",
                "Вкажіть одиницю виміру показника (наприклад: см, кг, шт)."); return
        try:
            cv_thr      = float(self._v["cv_thr"].get())
            plot_size   = int(self._v["plot_size"].get())
            edge_guard  = int(self._v["edge_guard"].get())
            rep_guard   = int(self._v["rep_guard"].get())
            n_var       = int(self._v["n_var"].get())
            n_rep       = int(self._v["n_rep"].get())
            max_it      = int(self._v["max_it"].get())
            seed        = int(self._v["seed"].get())
        except ValueError:
            messagebox.showwarning("Не заповнено параметри",
                "Заповніть усі параметри досліду (поріг CV%, повторність, захисні "
                "зони, кількість варіантів і повторень) — жодне з них не може\n"
                "лишатися порожнім чи нечисловим."); return
        if n_var < 2:
            messagebox.showwarning("Замало варіантів",
                "Кількість варіантів має бути щонайменше 2."); return
        if n_rep < 1:
            messagebox.showwarning("Замало повторень",
                "Кількість повторень має бути щонайменше 1."); return

        design_key = HP_DESIGN_LABELS_REV.get(self._design_v.get(), "rcbd")

        raw_rows = {}
        for i, row_e in enumerate(self.entries):
            entries = [(j+1, e.get()) for j, e in enumerate(row_e) if e.get().strip()]
            if entries: raw_rows[i+1] = entries
        plants = hp_load_plants(raw_rows)
        if not plants:
            messagebox.showwarning("Немає даних", "Заповніть таблицю значеннями."); return

        builder = HPPlotBuilder(
            plants, plot_size, edge_guard, rep_guard, cv_thr, max_it,
            count_dead_as_guard=self._dead_guard.get(),
            count_pollinizer_as_guard=self._poll_guard.get())
        result = builder.build()

        needed = n_var * n_rep
        if result["plots_formed"] < needed:
            self._status_lbl.configure(text="")
            self._show_insufficient_dialog(result["plots_formed"], needed, n_var, n_rep)
            return

        design_used, fell_back = hp_apply_design(result, design_key, n_var, n_rep, seed=seed)
        self._result = result
        self._cfg = {"trait_name": trait_name, "trait_unit": trait_unit,
                     "cv_thr": cv_thr, "plot_size": plot_size,
                     "edge_guard": edge_guard, "rep_guard": rep_guard,
                     "n_var": n_var, "n_rep": n_rep,
                     "design_requested": design_key, "design_used": design_used,
                     "design_fell_back": fell_back}

        msgs = list(result["warnings"])
        if fell_back:
            msgs.append("Латинський квадрат потребує рівно n_var блоків (повторень) — "
                        "сформована кількість не збігається, застосовано RCBD замість нього.")
        self._status_lbl.configure(text=("⚠ " + " | ".join(msgs)) if msgs else "")

        self._show_results()

    def _show_insufficient_dialog(self, formed, needed, n_var, n_rep):
        """Блокуюче пояснювальне вікно (за зразком діалогу ненормальності в ANOVA):
        показується ЗАМІСТЬ звіту, коли неможливо сформувати потрібну кількість
        повторностей — карта й список у такому разі НЕ відкриваються."""
        dlg = tk.Toplevel(self.win)
        dlg.title("Недостатньо рослин для дизайну досліду")
        dlg.resizable(False, False); set_icon(dlg)
        frm = tk.Frame(dlg, padx=18, pady=16); frm.pack()

        tk.Label(frm, text="🚫 Неможливо спланувати дослід із поточними даними",
                 font=("Times New Roman",13,"bold"), fg="#B71C1C"
                 ).pack(anchor="w", pady=(0,8))
        tk.Label(frm,
                 text=f"Сформовано лише {formed} повторностей із {needed} потрібних "
                      f"({n_var} варіантів × {n_rep} повторень).",
                 font=("Times New Roman",11), justify="left", wraplength=440
                 ).pack(anchor="w", pady=(0,10))

        tk.Label(frm, text="Що можна зробити:",
                 font=("Times New Roman",11,"bold")).pack(anchor="w")
        for line in [
            "• Збільшити кількість рослин для аналізу — додати ряди/позиції в таблицю;",
            "• Послабити (збільшити) заплановане порогове значення варіації (CV%) — "
            "менш жорсткий відбір дасть більше придатних рослин;",
            "• Зменшити розмір повторності або захисних зон;",
            "• Зменшити кількість варіантів чи повторень.",
        ]:
            tk.Label(frm, text=line, font=("Times New Roman",11),
                     justify="left", wraplength=440, anchor="w"
                     ).pack(anchor="w", padx=(6,0))

        tk.Button(frm, text="Зрозуміло", bg="#c62828", fg="white",
                  font=("Times New Roman",12), width=14,
                  command=dlg.destroy).pack(pady=(14,0))
        dlg.update_idletasks(); center_win(dlg); dlg.grab_set()

    # ── Мітки рослин для карти/списку/бланку (hook-методи mixin) ──
    def _plant_map_label(self, p):
        if p.role == HP_ROLE_RECORDED: return f"V{p.variant}"
        return {HP_ROLE_GUARD_EDGE:"К", HP_ROLE_GUARD_REP:"П", HP_ROLE_DEAD:"-",
                HP_ROLE_POLLINIZER:"+", HP_ROLE_EXTRA:"×"}.get(p.role, "")

    def _plant_list_label(self, p):
        v = p.variant
        if v is None: return "-"
        vnames = self._variant_names
        if vnames and 1 <= v <= len(vnames): return f"В{v} ({vnames[v-1]})"
        return f"В{v}"

    def _plant_form_label(self, p):
        return f"В{p.variant}-П{p.replication}"

    def _map_legend_note_text(self):
        return ('Позначення на клітинках:  "V1", "V2"…  — номер ВАРІАНТУ (облікова рослина)  •  '
                '"К" — захисна зона, край ряду  •  "П" — захисна зона між повтореннями  •  '
                '"-" — випад/пошкоджена рослина  •  "+" — запилювач  •  '
                '"×" — сформована повторність поза дизайном (залишок, не увійшов у жоден '
                'повний блок — не бере участі в обліку)')

    def _map_legend_decode(self, frame):
        if not self._variant_names: return
        legend2_f = tk.Frame(frame, bg="#eef3f8", padx=8, pady=6)
        legend2_f.pack(fill=tk.X)
        names_txt = "   •   ".join(f"В{i+1} = {nm}"
                                   for i, nm in enumerate(self._variant_names))
        tk.Label(legend2_f, text="Розшифрування варіантів:  " + names_txt,
                 bg="#eef3f8", fg="#1a4b8c", font=("Times New Roman",10,"bold"),
                 anchor="w", justify="left", wraplength=1360
                 ).pack(fill=tk.X)

    def _list_extra_header_lines(self):
        vnames = self._variant_names
        if vnames:
            return ["Варіанти: " + "  •  ".join(f"В{i+1}={n}" for i,n in enumerate(vnames))]
        return []
