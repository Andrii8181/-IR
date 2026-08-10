# main.py — S.A.D. v2.1 (точка входу)
# -*- coding: utf-8 -*-
from sad_common import *
from sad_correlation import GraphSettingsDlg, HeatmapSettingsDlg, ScatterSettingsDlg, CorrelationWindow
from sad_app import SADTk
from sad_descriptive_ttest import DescriptiveWindow, TTestWindow
from sad_regression import RegressionWindow, SampleSizeWindow
from sad_cluster_pca import ClusterWindow, PCAWindow
from sad_repeated import RepeatedMeasuresWindow, MixedRepeatedWindow
from sad_stability_ancova import StabilityWindow, AncovaWindow
from sad_manova import ManovaWindow
from sad_homogeneous import HelpWindow, HPPlant, HPPlotBuilder, HomogeneousPlotWindow
from sad_journal_trial import FieldJournalWindow, TrialDesignWindow

def _SADTk_new_init(self, root):
    # Ініціалізуємо лише стан (без UI від orig_init)
    self.root = root
    self.table_win = None; self.report_win = None; self.graph_win = None
    self._graph_figs = {}
    self._active_cell = None; self._active_prev = None
    self._sel_anchor = None; self._sel_cells = set(); self._sel_orig = {}
    self._fill_drag = False; self._fill_rows = []; self._fill_cols = []
    self.factor_title_map = {}
    self.graph_settings = dict(DEF_GS)
    self._current_project_path = None
    self._lbf_cache = {}
    if not hasattr(self, '_gs_titles'): self._gs_titles = {}
    self._ordinal_mode = False

    root.geometry("1280x780")
    root.minsize(1100, 680)
    root.configure(bg="#0f1117")
    root.title("S.A.D. — Статистичний аналіз даних")
    set_icon(root)

    # ── Кольорова схема ─────────────────────────────────────
    C = {
        "bg":       "#0f1117",   # основний фон
        "sidebar":  "#161b27",   # бокова панель
        "card":     "#1e2336",   # картка
        "card_hov": "#252d45",   # картка hover
        "accent":   "#4a90d9",   # синій акцент
        "red":      "#c0392b",   # кнопка аналіз
        "text":     "#e8eaf0",   # основний текст
        "sub":      "#8892a4",   # підтекст
        "border":   "#2a3350",   # межі
        "sep":      "#1e2336",   # роздільник
        "green":    "#27ae60",
        "purple":   "#8e44ad",
        "orange":   "#d35400",
        "teal":     "#16a085",
        "brown":    "#6d4c2a",
        "olive":    "#5a7d3a",
    }

    # ── Статистика використання (зберігається між сесіями) ──
    usage_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              ".sad_usage.json")
    usage = {}
    try:
        if os.path.exists(usage_file):
            with open(usage_file, "r", encoding="utf-8") as _f:
                usage = json.load(_f)
    except Exception: pass

    def _record_usage(key):
        usage[key] = usage.get(key, 0) + 1
        try:
            with open(usage_file, "w", encoding="utf-8") as _f:
                json.dump(usage, _f)
        except Exception: pass

    # ── Визначення всіх аналізів ────────────────────────────
    # key, назва, опис, колір, клас, needs_gs, fn, ключові слова пошуку
    ANALYSES = [
        ("anova1","Однофакторний ANOVA","CRD · RCBD · ЛК",
         "#1a4b8c",None,False,lambda: self.open_table(1),
         "сила впливу нір тьюкі дункан дисперс порівняння варіантів"),
        ("anova2","Двофакторний ANOVA","CRD · RCBD · Split-plot",
         "#1a4b8c",None,False,lambda: self.open_table(2),
         "сила впливу взаємодія факторів нір тьюкі дисперс"),
        ("anova3","Трифакторний ANOVA","Латинський квадрат",
         "#1a4b8c",None,False,lambda: self.open_table(3),
         "латинський квадрат три фактори сила впливу взаємодія"),
        ("anova4","Чотирифакторний ANOVA","Складні дизайни",
         "#1a4b8c",None,False,lambda: self.open_table(4),
         "чотири фактори складний дизайн сила впливу"),
        ("desc","Описова статистика","Mean · SD · Median · CV",
         C["green"],DescriptiveWindow,True,None,
         "середнє медіана дисперсія варіація коефіцієнт cv асиметрія ексцес"),
        ("ttest","t-тест / Манн-Уітні","Порівняння двох груп",
         C["green"],TTestWindow,False,None,
         "дві групи порівняння t критерій непараметричний"),
        ("corr","Кореляційний аналіз","Пірсон · Спірмен · Heat",
         C["accent"],CorrelationWindow,True,None,
         "зв'язок залежність матриця теплова карта пірсон спірмен"),
        ("reg","Регресійний аналіз","7 моделей · R² · p",
         C["purple"],RegressionWindow,True,None,
         "регресія прогноз r квадрат лінійна нелінійна поліном"),
        ("ancova","ANCOVA","Коваріаційний аналіз",
         C["purple"],AncovaWindow,True,None,
         "коваріата контроль змінної ancova"),
        ("manova","MANOVA","Багатовимірний дисп. аналіз",
         C["purple"],ManovaWindow,True,None,
         "кілька залежних змінних wilks pillai bagатовимірний"),
        ("rm","Повторні виміри","Within-subjects ANOVA",
         C["orange"],RepeatedMeasuresWindow,True,None,
         "повторні вимірювання часові точки динаміка within"),
        ("mix","Змішаний RM","Split-plot у часі",
         C["orange"],MixedRepeatedWindow,True,None,
         "кілька варіантів динаміка дати між групами within"),
        ("cluster","Кластерний аналіз","K-means · Ієрархічний",
         C["teal"],ClusterWindow,True,None,
         "групування схожість дендрограма kmeans кластери"),
        ("pca","PCA","Головні компоненти",
         C["teal"],PCAWindow,True,None,
         "головні компоненти зменшення вимірності biplot"),
        ("stab","Аналіз стабільності","Eberhart-Russell · GGE",
         "#8c1a1a",StabilityWindow,True,None,
         "gxe стабільність адаптація сортовипробування eberhart gge"),
        ("trialdesign","Генератор плану досліду","Поле · Сад · Ягідники",
         C["brown"],TrialDesignWindow,False,None,
         "план дослід поле сад ягідник захисна зона рандомізація повторення схема генератор"),
        ("homogplot","Однорідні ділянки саду","За CV% наявних рослин",
         C["olive"],HomogeneousPlotWindow,True,None,
         "однорідні ділянки cv діаметр штамб сад дерева вирівнювання рандомізація ітеративний"),
        ("fieldjournal","Польовий журнал обліків","Внесення даних у схему досліду",
         C["olive"],FieldJournalWindow,True,None,
         "польовий журнал облік схема внесення даних сезон рік урожайність вимірювання"),
    ]

    def _open(key, cls, needs_gs, custom_fn=None):
        _record_usage(key)
        _refresh_recent()
        root.iconify()  # Згортаємо головне вікно
        if custom_fn:
            custom_fn()
        elif needs_gs:
            w = cls(root, self.graph_settings)
        else:
            w = cls(root)

    # ── HEADER ──────────────────────────────────────────────
    header = tk.Frame(root, bg="#0d1020", height=64)
    header.pack(fill=tk.X, side=tk.TOP)
    header.pack_propagate(False)


    # ── Логотип і назва ───────────────────────────────────
    logo_frm = tk.Frame(header, bg="#0d1020")
    logo_frm.pack(side=tk.LEFT, padx=12, pady=6)

    # Завантажуємо Logo.png → icon.ico → текстовий fallback
    def _load_logo(size):
        if not HAS_PIL:
            return None
        try:
            from PIL import Image, ImageTk
            p = _find_file("Logo.png", "logo.png", "SAD_logo.png", "icon.ico")
            if p:
                img = Image.open(p).convert("RGBA").resize(size, Image.LANCZOS)
                return ImageTk.PhotoImage(img)
        except Exception:
            pass
        return None

    # Іконка окремо від тексту
    _logo_img = _load_logo((44, 44))
    if _logo_img:
        root._logo_img = _logo_img
        tk.Label(logo_frm, image=_logo_img, bg="#0d1020"
                 ).pack(side=tk.LEFT, padx=(0, 10))
    else:
        # Текстовий fallback — синій квадрат
        fb = tk.Frame(logo_frm, bg="#1a4b8c", width=44, height=44)
        fb.pack(side=tk.LEFT, padx=(0, 10))
        fb.pack_propagate(False)
        tk.Label(fb, text="S", bg="#1a4b8c", fg="#ffffff",
                 font=("Arial", 22, "bold")).place(relx=0.5, rely=0.5, anchor="center")

    # Назва ОКРЕМО від логотипа
    name_f = tk.Frame(logo_frm, bg="#0d1020"); name_f.pack(side=tk.LEFT)
    tk.Label(name_f, text="S.A.D.", bg="#0d1020", fg=C["text"],
             font=("Arial", 18, "bold")).pack(anchor="w")
    tk.Label(name_f, text="Статистичний аналіз даних", bg="#0d1020",
             fg=C["sub"], font=("Arial", 9)).pack(anchor="w")

    # Права частина header — версія, розробник, підтримка
    hr = tk.Frame(header, bg="#0d1020"); hr.pack(side=tk.RIGHT, padx=16)
    def _about():
        dlg = tk.Toplevel(root); dlg.title("Про програму S.A.D.")
        dlg.geometry("480x560"); dlg.resizable(False, False)
        dlg.configure(bg=C["card"]); set_icon(dlg); dlg.grab_set()

        # Логотип у діалозі
        _li = _load_logo((120, 120))
        if _li:
            dlg._li = _li
            tk.Label(dlg, image=_li, bg=C["card"]).pack(pady=(20, 4))
        else:
            fb2 = tk.Frame(dlg, bg="#1a4b8c", width=80, height=80)
            fb2.pack(pady=(20, 4))
            fb2.pack_propagate(False)
            tk.Label(fb2, text="S", bg="#1a4b8c", fg="#ffffff",
                     font=("Arial", 36, "bold")).place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(dlg, text="S.A.D.", bg=C["card"], fg=C["text"],
                 font=("Arial", 22, "bold")).pack()
        tk.Label(dlg, text="Статистичний аналіз даних",
                 bg=C["card"], fg=C["sub"], font=("Arial", 12)).pack()

        tk.Frame(dlg, bg=C["border"], height=1).pack(fill=tk.X, padx=30, pady=10)

        info = [
            (f"Версія {APP_VER}",                        C["accent"], 11, "bold"),
            ("Розробник:",                               C["sub"],    9,  "normal"),
            ("Чаплоуцький Андрій Миколайович",           C["text"],   11, "bold"),
            ("Уманський національний університет",       C["sub"],    10, "normal"),
            ("Україна",                     C["sub"],    10, "normal"),
        ]
        for txt, col, sz, weight in info:
            tk.Label(dlg, text=txt, bg=C["card"], fg=col,
                     font=("Arial", sz, weight)).pack(pady=1)

        tk.Frame(dlg, bg=C["border"], height=1).pack(fill=tk.X, padx=30, pady=10)

        tk.Label(dlg, text="Призначення:",
                 bg=C["card"], fg=C["sub"], font=("Arial", 9)).pack()
        tk.Label(dlg,
                 text="Програма для статистичного аналізу\n"
                      "агрономічних та біологічних дослідів.\n"
                      "ANOVA, кореляція, регресія, PCA,\n"
                      "аналіз стабільності GxE та інше.",
                 bg=C["card"], fg=C["text"], font=("Arial", 10),
                 justify="center").pack(pady=4)

        tk.Label(dlg, text="© 2024 – 2025  Всі права захищені",
                 bg=C["card"], fg=C["border"], font=("Arial", 8)).pack(pady=(8, 2))

        tk.Button(dlg, text="Закрити", bg=C["accent"], fg="white",
                  font=("Arial", 11), relief=tk.FLAT, padx=24, pady=5,
                  cursor="hand2", command=dlg.destroy).pack(pady=12)
        dlg.bind("<Return>", lambda e: dlg.destroy())
        center_win(dlg)

    def _support():
        dlg = tk.Toplevel(root); dlg.title("Технічна підтримка")
        dlg.geometry("420x340"); dlg.resizable(False, False)
        dlg.configure(bg=C["card"]); set_icon(dlg); dlg.grab_set()

        tk.Label(dlg, text="📞 Технічна підтримка S.A.D.",
                 bg=C["card"], fg=C["text"],
                 font=("Arial", 13, "bold")).pack(pady=(20, 4))
        tk.Frame(dlg, bg=C["border"], height=1).pack(fill=tk.X, padx=30, pady=8)

        contacts = [
            ("✉ Email:",         "sad.stat.support@gmail.com"),
            ("🌐 Документація:", "docs.sad-stat.com"),
            ("💬 Telegram:",     "@sad_stat_support"),
        ]
        for lbl, val in contacts:
            row = tk.Frame(dlg, bg=C["card"]); row.pack(pady=3)
            tk.Label(row, text=lbl, bg=C["card"], fg=C["sub"],
                     font=("Arial", 10), width=16, anchor="e").pack(side=tk.LEFT)
            tk.Label(row, text=val, bg=C["card"], fg=C["accent"],
                     font=("Arial", 10, "bold"), anchor="w").pack(side=tk.LEFT, padx=8)

        tk.Frame(dlg, bg=C["border"], height=1).pack(fill=tk.X, padx=30, pady=8)
        tk.Label(dlg,
                 text="Ми відповімо протягом 1 робочого дня.\n"
                      "При зверненні вкажіть версію програми\n"
                      f"та опис проблеми. (Версія {APP_VER})",
                 bg=C["card"], fg=C["sub"], font=("Arial", 9),
                 justify="center").pack()
        tk.Button(dlg, text="Закрити", bg=C["accent"], fg="white",
                  font=("Arial", 11), relief=tk.FLAT, padx=24, pady=5,
                  cursor="hand2", command=dlg.destroy).pack(pady=12)
        dlg.bind("<Return>", lambda e: dlg.destroy())
        center_win(dlg)

    # ── Changelog ─────────────────────────────────────────────
    CHANGELOG = [
        (f"v{APP_VER}", "Поточна версія", [
            "Новий темний головний екран з картками аналізів",
            "Бокова панель з пошуком та категоріями",
            "Статистика використання аналізів",
            "Об'єднаний звіт ANOVA (текст + графіки в одному вікні)",
            "Адаптивний розмір графіків під вікно",
            "Генератор плану польового досліду (CRD/RCBD/Split-plot/ЛК)",
            "Латинський квадрат у дисперсійному аналізі",
            "Автовизначення бальних і відсоткових даних",
            "Формула регресії безпосередньо на графіку",
            "Кореляційний аналіз: два графіки в одному вікні",
        ]),
        ("v2.0", "Великий реліз", [
            "ANOVA 1-4 фактори: CRD, RCBD, Split-plot",
            "Кореляційний аналіз (Пірсон, Спірмен, теплова карта)",
            "Регресійний аналіз (7 моделей)",
            "MANOVA, ANCOVA, Повторні виміри",
            "PCA та кластерний аналіз",
            "Аналіз стабільності GxE (Eberhart-Russell, GGE biplot)",
            "Збереження та відкриття проектів",
            "Експорт звітів у Word та Excel",
        ]),
        ("v1.0", "Перший реліз", [
            "Описова статистика",
            "t-тест та критерій Манн-Уітні",
            "Однофакторний дисперсійний аналіз",
            "Базові графіки",
        ]),
    ]

    def _changelog():
        dlg = tk.Toplevel(root); dlg.title("Зміни версій — S.A.D.")
        dlg.geometry("560x520"); dlg.resizable(True, True)
        dlg.configure(bg=C["card"]); set_icon(dlg); dlg.grab_set()

        tk.Label(dlg, text="📋  Зміни версій", bg=C["card"], fg=C["text"],
                 font=("Arial", 14, "bold")).pack(pady=(16, 4))
        tk.Frame(dlg, bg=C["border"], height=1).pack(fill=tk.X, padx=20, pady=6)

        # Прокручуваний список
        outer = tk.Frame(dlg, bg=C["card"]); outer.pack(fill=tk.BOTH, expand=True, padx=16)
        vsb = ttk.Scrollbar(outer, orient="vertical"); vsb.pack(side=tk.RIGHT, fill=tk.Y)
        cv2 = tk.Canvas(outer, bg=C["card"], highlightthickness=0,
                        yscrollcommand=vsb.set); cv2.pack(fill=tk.BOTH, expand=True)
        vsb.config(command=cv2.yview)
        inner = tk.Frame(cv2, bg=C["card"]); cv2.create_window((0,0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: cv2.configure(scrollregion=cv2.bbox("all")))
        def _cl_mw(e):
            cv2.yview_scroll(int(-1*(e.delta/120)), "units")
            return "break"
        cv2.bind("<MouseWheel>", _cl_mw)
        inner.bind("<MouseWheel>", _cl_mw)

        for ver, tag, items in CHANGELOG:
            # Версія — заголовок
            vh = tk.Frame(inner, bg=C["card"]); vh.pack(fill=tk.X, pady=(10,2))
            tk.Label(vh, text=ver, bg=C["accent"], fg="white",
                     font=("Arial",11,"bold"), padx=10, pady=3
                     ).pack(side=tk.LEFT)
            tk.Label(vh, text=tag, bg=C["card"], fg=C["sub"],
                     font=("Arial",9), padx=8
                     ).pack(side=tk.LEFT, pady=3)
            # Пункти
            for item in items:
                tk.Label(inner, text=f"  ✓  {item}", bg=C["card"], fg=C["text"],
                         font=("Arial",9), anchor="w", justify="left"
                         ).pack(fill=tk.X, padx=8, pady=1)
            tk.Frame(inner, bg=C["border"], height=1).pack(fill=tk.X, padx=8, pady=4)

        tk.Button(dlg, text="Закрити", bg=C["accent"], fg="white",
                  font=("Arial",11), relief=tk.FLAT, padx=24, pady=5,
                  cursor="hand2", command=dlg.destroy).pack(pady=10)
        dlg.bind("<Return>", lambda e: dlg.destroy())
        center_win(dlg)

    def _license():
        dlg = tk.Toplevel(root); dlg.title("Ліцензійна угода — S.A.D.")
        dlg.geometry("600x560"); dlg.resizable(True, True)
        dlg.configure(bg=C["card"]); set_icon(dlg); dlg.grab_set()

        tk.Label(dlg, text="📄  Ліцензійна угода кінцевого користувача",
                 bg=C["card"], fg=C["text"],
                 font=("Arial", 12, "bold")).pack(pady=(16,4))
        tk.Label(dlg, text=f"S.A.D. — Статистичний аналіз даних  |  Версія {APP_VER}",
                 bg=C["card"], fg=C["sub"], font=("Arial",9)).pack()
        tk.Frame(dlg, bg=C["border"], height=1).pack(fill=tk.X, padx=20, pady=8)

        lic_text = f"""ЛІЦЕНЗІЙНА УГОДА КІНЦЕВОГО КОРИСТУВАЧА (EULA)

© 2024–2025  Чаплоуцький Андрій Миколайович
Уманський національний університет, Україна

Прочитайте цю угоду уважно перед використанням програми.
Використовуючи програму, ви погоджуєтесь з умовами цієї угоди.

──────────────────────────────────────────────────────────
1. НАДАННЯ ЛІЦЕНЗІЇ
──────────────────────────────────────────────────────────
Розробник надає вам невиключне, непередаване право на
використання програмного забезпечення S.A.D. на одному
комп'ютері (або відповідно до придбаної ліцензії).

2. ОБМЕЖЕННЯ
──────────────────────────────────────────────────────────
Вам ЗАБОРОНЕНО:
  • Копіювати, розповсюджувати або передавати програму
    третім особам без письмового дозволу розробника
  • Декомпілювати, дисасемблювати або здійснювати
    зворотну розробку програми
  • Здавати програму в оренду або субліцензувати її
  • Видаляти або змінювати повідомлення про авторські права
  • Використовувати програму для надання комерційних послуг
    без укладення окремої угоди з розробником

3. АКАДЕМІЧНЕ ТА НАУКОВЕ ВИКОРИСТАННЯ
──────────────────────────────────────────────────────────
Програма розроблена для використання в наукових
дослідженнях та навчальному процесі. Результати аналізів,
отримані за допомогою S.A.D., можуть публікуватись
у наукових роботах з посиланням на програму.

Рекомендоване посилання:
Чаплоуцький А.М. S.A.D. — Статистичний аналіз даних.
Версія {APP_VER}. Уманський НУС, Україна, 2024. [Комп'ютерна програма]

4. ІНТЕЛЕКТУАЛЬНА ВЛАСНІСТЬ
──────────────────────────────────────────────────────────
Програма та вся документація є інтелектуальною власністю
розробника і захищені законодавством України про авторське
право та міжнародними договорами.

5. ВІДМОВА ВІД ГАРАНТІЙ
──────────────────────────────────────────────────────────
Програма надається «як є» (AS IS). Розробник не гарантує
безперебійну роботу або відсутність помилок. Відповідальність
за результати статистичних аналізів лежить на користувачі.

6. КОРПОРАТИВНЕ ЛІЦЕНЗУВАННЯ
──────────────────────────────────────────────────────────
Для установ, організацій або мереж — зв'яжіться з
розробником для укладення корпоративної ліцензії.
Email: sad.stat.support@gmail.com

7. ПРИПИНЕННЯ ДІЇ ЛІЦЕНЗІЇ
──────────────────────────────────────────────────────────
Ця ліцензія діє до її розірвання. Вона автоматично
припиняється при порушенні будь-якого з умов.

© 2024–2025  Чаплоуцький А.М.  Усі права захищені."""

        outer2 = tk.Frame(dlg, bg=C["card"]); outer2.pack(fill=tk.BOTH, expand=True, padx=16)
        vsb2 = ttk.Scrollbar(outer2, orient="vertical"); vsb2.pack(side=tk.RIGHT, fill=tk.Y)
        txt = tk.Text(outer2, wrap="word", font=("Courier New", 8),
                      bg="#0f1117", fg="#c8cdd8",
                      relief=tk.FLAT, padx=12, pady=8,
                      yscrollcommand=vsb2.set, state="normal", cursor="arrow")
        txt.pack(fill=tk.BOTH, expand=True)
        vsb2.config(command=txt.yview)
        txt.insert("1.0", lic_text)
        txt.configure(state="disabled")
        def _lic_mw(e):
            txt.yview_scroll(int(-1*(e.delta/120)), "units")
            return "break"   # зупиняє поширення події на root
        txt.bind("<MouseWheel>", _lic_mw)
        dlg.bind("<MouseWheel>", _lic_mw)

        btn_f = tk.Frame(dlg, bg=C["card"]); btn_f.pack(pady=10)
        tk.Button(btn_f, text="✓ Погоджуюсь", bg=C["green"], fg="white",
                  font=("Arial",11), relief=tk.FLAT, padx=20, pady=5,
                  cursor="hand2", command=dlg.destroy).pack(side=tk.LEFT, padx=8)
        tk.Button(btn_f, text="Закрити", bg=C["card"], fg=C["sub"],
                  font=("Arial",11), relief=tk.FLAT, padx=20, pady=5,
                  cursor="hand2", command=dlg.destroy).pack(side=tk.LEFT)
        center_win(dlg)

    # ── Кнопки header ─────────────────────────────────────────
    # Роздільник
    tk.Frame(hr, bg=C["border"], width=1).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

    btn_style = dict(bg="#0d1020", fg=C["sub"], font=("Arial",9),
                     relief=tk.FLAT, cursor="hand2",
                     activebackground="#161b27", activeforeground=C["text"],
                     padx=8, pady=4)
    for txt, cmd in [
        ("ℹ  Про програму", _about),
        ("📋  Ліцензія",    _license),
        ("📞  Підтримка",   _support),
    ]:
        tk.Button(hr, text=txt, command=cmd, **btn_style
                  ).pack(side=tk.LEFT, padx=2)

    # Роздільник + версія
    tk.Frame(hr, bg=C["border"], width=1).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
    tk.Label(hr, text=f"v{APP_VER}", bg="#0d1020",
             fg=C["accent"], font=("Arial",9,"bold")).pack(side=tk.LEFT, padx=4)

    # ── MAIN AREA ────────────────────────────────────────────
    body = tk.Frame(root, bg=C["bg"]); body.pack(fill=tk.BOTH, expand=True)

    # ════════════════════════════════════════════════════════
    # БОКОВА ПАНЕЛЬ
    # ════════════════════════════════════════════════════════
    sidebar = tk.Frame(body, bg=C["sidebar"], width=260)
    sidebar.pack(side=tk.LEFT, fill=tk.Y); sidebar.pack_propagate(False)

    # Пошук
    sf = tk.Frame(sidebar, bg=C["sidebar"], pady=8, padx=10)
    sf.pack(fill=tk.X)
    search_var = tk.StringVar()
    search_entry = tk.Entry(sf, textvariable=search_var,
                            bg=C["card"], fg=C["text"], insertbackground=C["text"],
                            relief=tk.FLAT, font=("Arial",10),
                            highlightthickness=1, highlightbackground=C["border"])
    search_entry.pack(fill=tk.X, ipady=5)
    search_entry.insert(0, "🔍  Пошук аналізу...")
    search_entry.config(fg=C["sub"])
    def _search_focus_in(e):
        if search_entry.get().startswith("🔍"):
            search_entry.delete(0, tk.END); search_entry.config(fg=C["text"])
    def _search_focus_out(e):
        if not search_entry.get().strip():
            search_entry.insert(0, "🔍  Пошук аналізу..."); search_entry.config(fg=C["sub"])
    search_entry.bind("<FocusIn>", _search_focus_in)
    search_entry.bind("<FocusOut>", _search_focus_out)

    # Роздільник
    tk.Frame(sidebar, bg=C["border"], height=1).pack(fill=tk.X)

    # Список аналізів у sidebar
    sb_canvas = tk.Canvas(sidebar, bg=C["sidebar"], highlightthickness=0)
    sb_vsb = tk.Scrollbar(sidebar, orient="vertical", command=sb_canvas.yview)
    sb_vsb.pack(side=tk.RIGHT, fill=tk.Y)
    sb_canvas.pack(fill=tk.BOTH, expand=True)
    sb_canvas.configure(yscrollcommand=sb_vsb.set)
    sb_inner = tk.Frame(sb_canvas, bg=C["sidebar"])
    sb_canvas.create_window((0,0), window=sb_inner, anchor="nw")

    def _mw_sidebar(e):
        sb_canvas.yview_scroll(int(-1*(e.delta/120)), "units")
    sb_canvas.bind("<MouseWheel>", _mw_sidebar)
    sb_inner.bind("<MouseWheel>", _mw_sidebar)
    sidebar.bind("<MouseWheel>", _mw_sidebar)
    sf.bind("<MouseWheel>", _mw_sidebar)
    # Bind all sidebar children after make_sidebar
    def _bind_sb_children():
        for w in sb_inner.winfo_children():
            w.bind("<MouseWheel>", _mw_sidebar)
            for ch in w.winfo_children():
                ch.bind("<MouseWheel>", _mw_sidebar)
    sb_inner.bind("<Configure>", lambda e: (
        sb_canvas.configure(scrollregion=sb_canvas.bbox("all")),
        _bind_sb_children()))

    CATEGORIES = [
        ("ANOVA",             ["anova1","anova2","anova3","anova4"]),
        ("Базові методи",     ["desc","ttest"]),
        ("Зв'язок змінних",  ["corr","reg","ancova"]),
        ("Багатовимірні",     ["manova","rm","mix"]),
        ("Багатовимірний ML", ["cluster","pca"]),
        ("Спеціальні",        ["stab","trialdesign","homogplot"]),
    ]
    _ana_map = {a[0]: a for a in ANALYSES}
    _sb_btns = {}

    def _make_sidebar():
        for w in sb_inner.winfo_children(): w.destroy()
        q = search_var.get().lower().strip()
        if q.startswith("🔍"): q = ""
        for cat_name, keys in CATEGORIES:
            def _matches(k):
                if not q: return True
                a = _ana_map[k]
                return (q in a[1].lower() or q in a[2].lower()
                        or (len(a) > 7 and q in a[7].lower()))
            filtered = [k for k in keys if _matches(k)]
            if not filtered: continue
            tk.Label(sb_inner, text=cat_name.upper(), bg=C["sidebar"],
                     fg=C["sub"], font=("Arial",8,"bold"),
                     anchor="w", padx=12, pady=8
                     ).pack(fill=tk.X)
            for k in filtered:
                a = _ana_map[k]
                col = a[3]
                btn_f = tk.Frame(sb_inner, bg=C["sidebar"]); btn_f.pack(fill=tk.X)
                cnt = usage.get(k,0)
                lbl_txt = f"  {a[1]}"
                b = tk.Label(btn_f, text=lbl_txt, bg=C["sidebar"], fg=C["text"],
                             font=("Arial",10), anchor="w", padx=10, pady=5,
                             cursor="hand2")
                b.pack(side=tk.LEFT, fill=tk.X, expand=True)
                if cnt > 0:
                    tk.Label(btn_f, text=str(cnt), bg=C["sidebar"],
                             fg=C["sub"], font=("Arial",8),
                             padx=6).pack(side=tk.RIGHT)
                # Кольоровий лівий бордер
                border = tk.Frame(btn_f, bg=col, width=3)
                border.place(relx=0, rely=0, relheight=1)
                def _enter(e, f=btn_f, brd=border):
                    f.configure(bg=C["card_hov"])
                    for ch in f.winfo_children(): ch.configure(bg=C["card_hov"])
                    brd.configure(bg=C["accent"])
                def _leave(e, f=btn_f, brd=border, c=col):
                    f.configure(bg=C["sidebar"])
                    for ch in f.winfo_children(): ch.configure(bg=C["sidebar"])
                    brd.configure(bg=c)
                for w2 in [btn_f, b]:
                    w2.bind("<Enter>", _enter)
                    w2.bind("<Leave>", _leave)
                    w2.bind("<Button-1>", lambda e, k2=k, a2=a:
                            _open(k2, a2[4], a2[5], a2[6]))
                _sb_btns[k] = btn_f
            tk.Frame(sb_inner, bg=C["border"], height=1).pack(fill=tk.X, padx=10)

    search_var.trace_add("write", lambda *_: _make_sidebar())
    _make_sidebar()

    # ════════════════════════════════════════════════════════
    # ПРАВА ЧАСТИНА — КАРТКИ
    # ════════════════════════════════════════════════════════
    right = tk.Frame(body, bg=C["bg"]); right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Заголовок
    top_r = tk.Frame(right, bg=C["bg"], padx=24, pady=16)
    top_r.pack(fill=tk.X)
    tk.Label(top_r, text="Оберіть тип аналізу", bg=C["bg"],
             fg=C["text"], font=("Arial",18,"bold")).pack(side=tk.LEFT)

    # Кнопки проекту справа
    proj_f = tk.Frame(top_r, bg=C["bg"]); proj_f.pack(side=tk.RIGHT)
    def _load_proj_home():
        path = filedialog.askopenfilename(
            parent=root,
            filetypes=[("SAD проект","*.sadp"),("JSON","*.json"),("All","*.*")],
            title="Відкрити проект S.A.D.")
        if not path: return
        try:
            with open(path,"r",encoding="utf-8") as _f:
                d = json.load(_f)
        except Exception as ex:
            messagebox.showerror("Помилка відкриття", str(ex)); return
        ptype = d.get("type","")
        # ANOVA / головний проект — factors_count є завжди
        if not ptype or "factors_count" in d:
            project_from_dict(self, d)
            messagebox.showinfo("Проект відкрито", f"Проект завантажено:\n{path}")
            return
        # Спеціалізовані проекти
        def _after_open(win_obj):
            root.after(300, lambda w=win_obj: _fill_win(w, d, path))
        def _fill_win(w, dd, pp):
            try:
                rows = dd.get("rows_data",[])
                envs = (dd.get("env_vars") or dd.get("time_vars") or
                        dd.get("col_vars") or [])
                n_need = (1+len(envs)) if envs else len(rows[0]) if rows else 0
                while getattr(w,"cols_n",0) < n_need:
                    if hasattr(w,"_add_col"): w._add_col()
                    else: break
                for i,nm in enumerate(envs):
                    for attr in ["time_vars","col_vars","env_vars"]:
                        vl = getattr(w,attr,[])
                        if i<len(vl): vl[i].set(nm); break
                while len(getattr(w,"entries",[])) < len(rows):
                    if hasattr(w,"_add_row"): w._add_row()
                    else: break
                for i,rv in enumerate(rows):
                    for j,v in enumerate(rv):
                        ents = getattr(w,"entries",[])
                        if i<len(ents) and j<len(ents[i]):
                            ents[i][j].delete(0,tk.END)
                            ents[i][j].insert(0,v)
                messagebox.showinfo("Завантажено", f"Проект завантажено:\n{pp}")
            except Exception as ex:
                messagebox.showerror("Помилка завантаження", str(ex))
        win_map = {
            "stability":              lambda: _after_open(StabilityWindow(root,self.graph_settings)),
            "mixed_repeated_measures":lambda: _after_open(MixedRepeatedWindow(root,self.graph_settings)),
            "repeated_measures":      lambda: _after_open(RepeatedMeasuresWindow(root,self.graph_settings)),
            "correlation":            lambda: _after_open(CorrelationWindow(root,self.graph_settings)),
        }
        for key, fn in win_map.items():
            if key in ptype:
                fn(); return
        messagebox.showinfo("Проект відкрито",
            f"Тип проекту: «{ptype}»\n"
            "Відкрийте відповідний аналіз вручну і\n"
            "скористайтесь «📂 Відкрити проект» у тому вікні.")

    tk.Button(proj_f, text="📂 Відкрити проект", bg=C["card"], fg=C["text"],
              font=("Arial",10), relief=tk.FLAT, padx=12, pady=6,
              cursor="hand2", activebackground=C["card_hov"],
              command=_load_proj_home).pack(side=tk.LEFT, padx=4)

    # Прокручуваний контент
    content_canvas = tk.Canvas(right, bg=C["bg"], highlightthickness=0)
    c_vsb = tk.Scrollbar(right, orient="vertical", command=content_canvas.yview)
    c_vsb.pack(side=tk.RIGHT, fill=tk.Y)
    content_canvas.pack(fill=tk.BOTH, expand=True)
    content_canvas.configure(yscrollcommand=c_vsb.set)
    cf = tk.Frame(content_canvas, bg=C["bg"])
    cf_win = content_canvas.create_window((0,0), window=cf, anchor="nw")

    def _on_cf_configure(e):
        content_canvas.configure(scrollregion=content_canvas.bbox("all"))
        # Прокрутка для всіх нових дочірніх елементів
        def _bind_all(w):
            try: w.bind("<MouseWheel>", _mw_content)
            except Exception: pass
            for ch in w.winfo_children(): _bind_all(ch)
        _bind_all(cf)
    cf.bind("<Configure>", _on_cf_configure)

    content_canvas.bind("<Configure>",
                        lambda e: content_canvas.itemconfig(cf_win, width=e.width))

    def _mw_content(e):
        delta = int(-1*(e.delta/120))
        top, bot = content_canvas.yview()
        if delta < 0 and top <= 0.001: return
        if delta > 0 and bot >= 0.999: return
        content_canvas.yview_scroll(delta, "units")

    def _mw_sidebar_global(e):
        delta = int(-1*(e.delta/120))
        top, bot = sb_canvas.yview()
        if delta < 0 and top <= 0.001: return
        if delta > 0 and bot >= 0.999: return
        sb_canvas.yview_scroll(delta, "units")

    # Глобальне прив'язування через root — найнадійніший спосіб
    def _global_mw(e):
        wx = e.widget
        # Визначаємо чи курсор над правою частиною чи лівою
        try:
            abs_x = e.widget.winfo_rootx()
            sidebar_right = sidebar.winfo_rootx() + sidebar.winfo_width()
            if abs_x >= sidebar_right:
                _mw_content(e)
            else:
                _mw_sidebar_global(e)
        except Exception:
            _mw_content(e)

    root.bind_all("<MouseWheel>", _global_mw)
    right.bind("<MouseWheel>", _mw_content)
    content_canvas.bind("<MouseWheel>", _mw_content)
    cf.bind("<MouseWheel>", _mw_content)

    def _card(parent, key, name, desc, color, cls, needs_gs, custom_fn,
              large=False):
        """Сучасна об'ємна картка аналізу."""
        w = 280 if large else 210
        h = 120 if large else 96
        pad = 16 if large else 12
        name_sz = 13 if large else 11
        desc_sz = 9 if large else 8
        _dark  = _darken(color)
        _light = _lighten(color)

        # Зовнішня рамка — глибока тінь для об'єму
        _darker = _darken(color, 50)
        outer = tk.Frame(parent, bg=_darker,
                         width=w+4, height=h+4, cursor="hand2")
        outer.pack_propagate(False)

        # Середня рамка — бічна тінь
        mid = tk.Frame(outer, bg=_dark, cursor="hand2")
        mid.pack_propagate(False)
        mid.pack(fill=tk.BOTH, expand=True, padx=(1,3), pady=(1,3))

        # Основний фрейм
        frm = tk.Frame(mid, bg=color, cursor="hand2")
        frm.pack_propagate(False)
        frm.pack(fill=tk.BOTH, expand=True)

        # Верхня світла смужка — ефект блиску
        shine = tk.Frame(frm, bg=_light, height=3)
        shine.pack(fill=tk.X, side=tk.TOP)

        # Ліва світла смужка — бічне підсвічування
        left_shine = tk.Frame(frm, bg=_lighten(color, 20), width=2)
        left_shine.pack(side=tk.LEFT, fill=tk.Y)

        # Вміст
        inner = tk.Frame(frm, bg=color, padx=pad, pady=pad-2)
        inner.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(inner, text=name, bg=color, fg="white",
                 font=("Arial", name_sz, "bold"),
                 wraplength=w-pad*2, justify="left", anchor="w"
                 ).pack(anchor="w")
        tk.Label(inner, text=desc, bg=color,
                 font=("Arial", desc_sz),
                 wraplength=w-pad*2, justify="left", anchor="w",
                 fg="#cccccc"
                 ).pack(anchor="w", pady=(3,0))

        cnt = usage.get(key, 0)
        if cnt > 0 and large:
            tk.Label(inner, text=f"↳ використовували {cnt}×",
                     bg=color, fg="#aaaaaa",
                     font=("Arial",7)).pack(anchor="w", pady=(4,0))

        # Hover
        def _e(e):
            outer.configure(bg=_darker)
            mid.configure(bg=_dark)
            frm.configure(bg=_dark)
            shine.configure(bg=_lighten(_dark,20))
            left_shine.configure(bg=_lighten(_dark,15))
            inner.configure(bg=_dark)
            for ch in inner.winfo_children(): ch.configure(bg=_dark)
        def _l(e):
            outer.configure(bg=_darker)
            mid.configure(bg=_dark)
            frm.configure(bg=color)
            shine.configure(bg=_light)
            left_shine.configure(bg=_lighten(color,20))
            inner.configure(bg=color)
            for ch in inner.winfo_children(): ch.configure(bg=color)
        click_cmd = lambda e, k=key, cl=cls, ng=needs_gs, cf2=custom_fn:                     _open(k, cl, ng, cf2)
        for w2 in ([outer, mid, frm, shine, left_shine, inner] +
                   list(inner.winfo_children())):
            w2.bind("<Enter>", _e)
            w2.bind("<Leave>", _l)
            w2.bind("<Button-1>", click_cmd)
        return outer

    def _darken(hex_color, amt=30):
        try:
            h = hex_color.lstrip("#")
            r,g,b = int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
            return f"#{max(0,r-amt):02x}{max(0,g-amt):02x}{max(0,b-amt):02x}"
        except Exception: return hex_color

    def _lighten(hex_color, amt=40):
        try:
            h = hex_color.lstrip("#")
            r,g,b = int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
            return f"#{min(255,r+amt):02x}{min(255,g+amt):02x}{min(255,b+amt):02x}"
        except Exception: return hex_color

    def _refresh_recent():
        """Перебудовує секцію «Нещодавні»."""
        for w in cf.winfo_children(): w.destroy()

        padx = 24

        # ── Нещодавні / Часті (великі картки) ──────────────
        recent_keys = sorted(usage.keys(), key=lambda k: -usage.get(k,0))
        recent_keys = [k for k in recent_keys if k in _ana_map][:6]

        if recent_keys:
            sec1 = tk.Frame(cf, bg=C["bg"]); sec1.pack(fill=tk.X, padx=padx, pady=(8,4))
            tk.Label(sec1, text="Нещодавні та часті", bg=C["bg"],
                     fg=C["sub"], font=("Arial",10,"bold")).pack(anchor="w")
            cards_f1 = tk.Frame(cf, bg=C["bg"]); cards_f1.pack(fill=tk.X, padx=padx, pady=4)
            for k in recent_keys:
                a = _ana_map[k]
                c = _card(cards_f1, k, a[1], a[2], a[3], a[4], a[5], a[6], large=True)
                c.pack(side=tk.LEFT, padx=(0,10), pady=4)
            tk.Frame(cf, bg=C["border"], height=1).pack(fill=tk.X, padx=padx, pady=4)

        # ── Всі аналізи по категоріях ───────────────────────
        for cat_name, keys in CATEGORIES:
            sec = tk.Frame(cf, bg=C["bg"]); sec.pack(fill=tk.X, padx=padx, pady=(12,4))
            tk.Label(sec, text=cat_name, bg=C["bg"],
                     fg=C["text"], font=("Arial",12,"bold")).pack(anchor="w")
            row_f = tk.Frame(cf, bg=C["bg"]); row_f.pack(fill=tk.X, padx=padx, pady=4)
            for k in keys:
                a = _ana_map[k]
                c = _card(row_f, k, a[1], a[2], a[3], a[4], a[5], a[6])
                c.pack(side=tk.LEFT, padx=(0,8), pady=4)

        # ── Footer ──────────────────────────────────────────
        footer_f = tk.Frame(cf, bg="#0d1020"); footer_f.pack(fill=tk.X, side=tk.BOTTOM)
        tk.Frame(footer_f, bg=C["border"], height=1).pack(fill=tk.X)
        footer = tk.Frame(footer_f, bg="#0d1020", padx=24, pady=8)
        footer.pack(fill=tk.X)
        tk.Label(footer,
                 text="© 2024–2025  Чаплоуцький А.М.  |  "
                      "Уманський НУ, Україна  |  "
                      "Усі права захищені",
                 bg="#0d1020", fg=C["sub"],
                 font=("Arial", 8)).pack(side=tk.LEFT)
        tk.Label(footer,
                 text=f"S.A.D.  v{APP_VER}",
                 bg="#0d1020", fg=C["border"],
                 font=("Arial", 8)).pack(side=tk.RIGHT)

    _refresh_recent()


SADTk.__init__ = _SADTk_new_init


if __name__ == "__main__":
    root = tk.Tk()
    set_icon(root)
    app = SADTk(root)
    root.mainloop()
