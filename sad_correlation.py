# sad_correlation.py — Кореляційний аналіз
# -*- coding: utf-8 -*-
from sad_common import *
from sad_journal_trial import open_indicators_for_variant_analysis


# ═══════════════════════════════════════════════════════════════
# GRAPH SETTINGS DIALOG
# ═══════════════════════════════════════════════════════════════
class GraphSettingsDlg(tk.Toplevel):
    FONTS  = ["Times New Roman", "Arial", "Calibri", "Georgia", "Verdana", "Courier New"]
    STYLES = ["normal", "bold", "italic", "bold italic"]
    CMAPS  = ["RdYlGn", "coolwarm", "RdBu", "PiYG", "PRGn", "bwr", "seismic", "viridis", "plasma"]

    def __init__(self, parent, gs: dict, show_heatmap=False):
        super().__init__(parent)
        self.title("Налаштування графіків")
        self.resizable(False, False); set_icon(self)
        self.gs = dict(gs); self.result = None
        self._col_box = gs["box_color"]; self._col_med = gs["median_color"]
        self._col_wh  = gs["whisker_color"]; self._col_fl = gs["flier_color"]
        self._venn_fc = gs["venn_font_color"]; self._venn_cols = list(gs["venn_colors"])

        nb = ttk.Notebook(self); nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # ── Boxplot ──
        bp = tk.Frame(nb, padx=12, pady=10); nb.add(bp, text="Boxplot")
        self._ff = tk.StringVar(value=gs["font_family"])
        self._fs = tk.StringVar(value=gs["font_style"])
        self._fz = tk.IntVar(value=gs["font_size"])
        r = 0
        for lbl, var, vals in [("Шрифт:", self._ff, self.FONTS), ("Стиль:", self._fs, self.STYLES)]:
            tk.Label(bp, text=lbl).grid(row=r, column=0, sticky="w", pady=4)
            ttk.Combobox(bp, textvariable=var, values=vals, state="readonly", width=22).grid(row=r, column=1, sticky="w", padx=6); r += 1
        tk.Label(bp, text="Розмір:").grid(row=r, column=0, sticky="w", pady=4)
        tk.Spinbox(bp, from_=7, to=28, textvariable=self._fz, width=6).grid(row=r, column=1, sticky="w", padx=6); r += 1
        self._bp_btns = {}
        for lbl, attr in [("Колір коробки:", "_col_box"), ("Колір медіани:", "_col_med"),
                           ("Колір вусів:", "_col_wh"), ("Колір викидів:", "_col_fl")]:
            tk.Label(bp, text=lbl).grid(row=r, column=0, sticky="w", pady=4)
            btn = tk.Button(bp, width=6, relief=tk.SUNKEN, bg=getattr(self, attr),
                            command=lambda a=attr: self._pick(a))
            btn.grid(row=r, column=1, sticky="w", padx=6); self._bp_btns[attr] = btn; r += 1

        # ── Venn ──
        vf = tk.Frame(nb, padx=12, pady=10); nb.add(vf, text="Діаграма Венна")
        self._vff = tk.StringVar(value=gs["font_family"])
        self._vfz = tk.IntVar(value=gs["venn_font_size"])
        self._valpha = tk.DoubleVar(value=gs["venn_alpha"])
        r = 0
        tk.Label(vf, text="Шрифт:").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(vf, textvariable=self._vff, values=self.FONTS, state="readonly", width=22).grid(row=r, column=1, sticky="w", padx=6); r += 1
        tk.Label(vf, text="Розмір:").grid(row=r, column=0, sticky="w", pady=4)
        tk.Spinbox(vf, from_=7, to=28, textvariable=self._vfz, width=6).grid(row=r, column=1, sticky="w", padx=6); r += 1
        tk.Label(vf, text="Прозорість:").grid(row=r, column=0, sticky="w", pady=4)
        tk.Scale(vf, from_=0., to=1., resolution=0.05, orient="horizontal",
                 variable=self._valpha, length=180).grid(row=r, column=1, sticky="w", padx=6); r += 1
        tk.Label(vf, text="Колір тексту:").grid(row=r, column=0, sticky="w", pady=4)
        self._vfc_btn = tk.Button(vf, width=6, relief=tk.SUNKEN, bg=self._venn_fc, command=self._pick_vfc)
        self._vfc_btn.grid(row=r, column=1, sticky="w", padx=6); r += 1
        tk.Label(vf, text="Кольори кіл:").grid(row=r, column=0, sticky="w", pady=4)
        bf_ = tk.Frame(vf); bf_.grid(row=r, column=1, sticky="w", padx=6)
        self._vci_btns = []
        for idx in range(4):
            b = tk.Button(bf_, width=4, relief=tk.SUNKEN, bg=self._venn_cols[idx],
                          command=lambda i=idx: self._pick_vci(i))
            b.pack(side=tk.LEFT, padx=2); self._vci_btns.append(b)

        # ── Heatmap ──
        if show_heatmap:
            hf = tk.Frame(nb, padx=12, pady=10); nb.add(hf, text="Теплова карта")
            self._hcmap = tk.StringVar(value=gs.get("heatmap_cmap", "RdYlGn"))
            self._hfz   = tk.IntVar(value=gs.get("heatmap_font_size", 10))
            self._hannot_col = gs.get("heatmap_annot_color", "#000000")
            r = 0
            tk.Label(hf, text="Палітра:").grid(row=r, column=0, sticky="w", pady=4)
            ttk.Combobox(hf, textvariable=self._hcmap, values=self.CMAPS, state="readonly", width=18).grid(row=r, column=1, sticky="w", padx=6); r += 1
            tk.Label(hf, text="Розмір шрифту:").grid(row=r, column=0, sticky="w", pady=4)
            tk.Spinbox(hf, from_=6, to=20, textvariable=self._hfz, width=6).grid(row=r, column=1, sticky="w", padx=6); r += 1
            tk.Label(hf, text="Колір анотацій:").grid(row=r, column=0, sticky="w", pady=4)
            self._hannot_btn = tk.Button(hf, width=6, relief=tk.SUNKEN, bg=self._hannot_col, command=self._pick_hannot)
            self._hannot_btn.grid(row=r, column=1, sticky="w", padx=6)
        else:
            self._hcmap = tk.StringVar(value=gs.get("heatmap_cmap", "RdYlGn"))
            self._hfz   = tk.IntVar(value=gs.get("heatmap_font_size", 10))
            self._hannot_col = gs.get("heatmap_annot_color", "#000000")

        bf2 = tk.Frame(self); bf2.pack(fill=tk.X, padx=10, pady=(0, 10))
        tk.Button(bf2, text="OK", width=10, command=self._ok).pack(side=tk.LEFT, padx=4)
        tk.Button(bf2, text="Скасувати", width=12, command=self.destroy).pack(side=tk.LEFT)
        self.update_idletasks(); center_win(self); self.grab_set()

    def _pick(self, attr):
        c = colorchooser.askcolor(color=getattr(self, attr), parent=self, title="Виберіть колір")
        if c and c[1]:
            setattr(self, attr, c[1])
            if attr in self._bp_btns: self._bp_btns[attr].configure(bg=c[1])

    def _pick_vfc(self):
        c = colorchooser.askcolor(color=self._venn_fc, parent=self, title="Колір тексту")
        if c and c[1]: self._venn_fc = c[1]; self._vfc_btn.configure(bg=c[1])

    def _pick_vci(self, idx):
        c = colorchooser.askcolor(color=self._venn_cols[idx], parent=self, title=f"Колір кола {idx+1}")
        if c and c[1]: self._venn_cols[idx] = c[1]; self._vci_btns[idx].configure(bg=c[1])

    def _pick_hannot(self):
        c = colorchooser.askcolor(color=self._hannot_col, parent=self, title="Колір анотацій")
        if c and c[1]: self._hannot_col = c[1]; self._hannot_btn.configure(bg=c[1])

    def _ok(self):
        self.result = {
            "font_family": self._ff.get(), "font_style": self._fs.get(), "font_size": self._fz.get(),
            "box_color": self._col_box, "median_color": self._col_med,
            "whisker_color": self._col_wh, "flier_color": self._col_fl,
            "venn_colors": list(self._venn_cols), "venn_alpha": float(self._valpha.get()),
            "venn_font_size": self._vfz.get(), "venn_font_color": self._venn_fc,
            "heatmap_cmap": self._hcmap.get(), "heatmap_font_size": self._hfz.get(),
            "heatmap_annot_color": self._hannot_col,
        }
        self.destroy()

# ═══════════════════════════════════════════════════════════════
# VENN DIAGRAM  — proper overlap layout
# ═══════════════════════════════════════════════════════════════
def draw_venn(ax, factor_values, interaction_values, colors, alpha, font_size, font_color, font_family, title=""):
    """
    Proportional Venn diagram.
    factor_values: list of (label, pct) for main factors (circles)
    interaction_values: dict {frozenset_of_indices: (label, pct)} for interactions (overlaps)
    """
    ax.set_aspect("equal"); ax.axis("off")
    n = len(factor_values)
    if n == 0: return

    # Circle centres layout
    import math as _m
    if n == 1:
        centres = [(0, 0)]
    elif n == 2:
        centres = [(-0.33, 0), (0.33, 0)]
    elif n == 3:
        r_ = 0.38
        centres = [(r_ * _m.cos(_m.radians(90 + 120 * i)),
                    r_ * _m.sin(_m.radians(90 + 120 * i))) for i in range(3)]
    else:
        r_ = 0.42
        centres = [(r_ * _m.cos(_m.radians(45 + 90 * i)),
                    r_ * _m.sin(_m.radians(45 + 90 * i))) for i in range(4)]

    radius = 0.40

    # Draw circles
    for i, (cx, cy) in enumerate(centres):
        circle = mpatches.Circle((cx, cy), radius, fc=colors[i % len(colors)],
                                  alpha=alpha, ec="#444444", lw=1.2, zorder=2)
        ax.add_patch(circle)

    # Label main factors — place outside circle
    for i, ((cx, cy), (lbl, pct)) in enumerate(zip(centres, factor_values)):
        # offset outward from centre of diagram
        dx = cx - 0; dy = cy - 0
        mag = max(_m.hypot(dx, dy), 0.01)
        ox = cx + (dx / mag) * 0.55; oy = cy + (dy / mag) * 0.55
        if n == 1: ox, oy = 0, 0.65
        ax.text(ox, oy, f"{lbl}\n{fmt(pct, 1)}%",
                ha="center", va="center", fontsize=font_size,
                color=font_color, fontfamily=font_family,
                fontweight="bold", linespacing=1.3, zorder=5)

    # Label interactions at geometric intersections
    for key, (lbl, pct) in (interaction_values or {}).items():
        idxs = sorted(list(key))
        if len(idxs) < 2 or max(idxs) >= n: continue
        # mean of involved centres
        mx = sum(centres[i][0] for i in idxs) / len(idxs)
        my = sum(centres[i][1] for i in idxs) / len(idxs)
        ax.text(mx, my, f"{fmt(pct, 1)}%",
                ha="center", va="center", fontsize=max(font_size - 1, 7),
                color=font_color, fontfamily=font_family,
                fontweight="bold", zorder=6)

    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    if title:
        ax.set_title(title, fontsize=font_size + 1, fontfamily=font_family, pad=8)



# ═══════════════════════════════════════════════════════════════
# HEATMAP-ONLY SETTINGS DIALOG (для кореляційного аналізу)
# ═══════════════════════════════════════════════════════════════
class HeatmapSettingsDlg(tk.Toplevel):
    """Діалог налаштувань лише для теплової карти кореляцій."""
    CMAPS = ["RdYlGn","coolwarm","RdBu","PiYG","PRGn","bwr","seismic",
             "viridis","plasma","Blues","Reds","Greens"]

    def __init__(self, parent, gs: dict):
        super().__init__(parent)
        self.title("Налаштування теплової карти")
        self.resizable(False, False); set_icon(self)
        self.gs = dict(gs); self.result = None
        self._hannot_col = gs.get("heatmap_annot_color", "#000000")

        frm = tk.Frame(self, padx=16, pady=14); frm.pack(fill=tk.BOTH, expand=True)

        self._hcmap = tk.StringVar(value=gs.get("heatmap_cmap","RdYlGn"))
        self._hfz   = tk.IntVar(value=gs.get("heatmap_font_size", 10))
        self._ff    = tk.StringVar(value=gs.get("font_family","Times New Roman"))

        r = 0
        for lbl, wid in [("Палітра кольорів:", None),
                          ("Шрифт:", None),
                          ("Розмір шрифту:", None),
                          ("Колір тексту у клітинках:", None)]:
            tk.Label(frm, text=lbl, font=("Times New Roman",12)
                     ).grid(row=r, column=0, sticky="w", pady=6)
            if r == 0:
                ttk.Combobox(frm, textvariable=self._hcmap, values=self.CMAPS,
                             state="readonly", width=20).grid(row=r, column=1,
                             sticky="w", padx=8)
            elif r == 1:
                fonts = ["Times New Roman","Arial","Calibri","Georgia","Verdana","Courier New"]
                ttk.Combobox(frm, textvariable=self._ff, values=fonts,
                             state="readonly", width=20).grid(row=r, column=1,
                             sticky="w", padx=8)
            elif r == 2:
                tk.Spinbox(frm, from_=6, to=20, textvariable=self._hfz,
                           width=6).grid(row=r, column=1, sticky="w", padx=8)
            elif r == 3:
                self._hannot_btn = tk.Button(frm, width=6, relief=tk.SUNKEN,
                                             bg=self._hannot_col,
                                             command=self._pick_col)
                self._hannot_btn.grid(row=r, column=1, sticky="w", padx=8)
            r += 1

        bf = tk.Frame(frm); bf.grid(row=r, column=0, columnspan=2, pady=(14,0))
        tk.Button(bf, text="OK", width=10, bg="#c62828", fg="white",
                  font=("Times New Roman",12), command=self._ok).pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text="Скасувати", width=12,
                  font=("Times New Roman",12), command=self.destroy).pack(side=tk.LEFT)
        self.update_idletasks(); center_win(self); self.grab_set()

    def _pick_col(self):
        c = colorchooser.askcolor(color=self._hannot_col, parent=self, title="Колір тексту")
        if c and c[1]: self._hannot_col = c[1]; self._hannot_btn.configure(bg=c[1])

    def _ok(self):
        self.result = dict(self.gs)
        self.result.update({
            "heatmap_cmap":        self._hcmap.get(),
            "heatmap_font_size":   self._hfz.get(),
            "heatmap_annot_color": self._hannot_col,
            "font_family":         self._ff.get(),
        })
        self.destroy()


# ═══════════════════════════════════════════════════════════════
# SCATTER MATRIX SETTINGS DIALOG
# ═══════════════════════════════════════════════════════════════
class ScatterSettingsDlg(tk.Toplevel):
    """Діалог налаштувань матриці діаграм розсіювання."""
    COLORS = ["#4c72b0","#dd8452","#55a868","#c44e52","#8172b2","#1a6b1a","#c62828","#555555"]
    FONTS  = ["Times New Roman","Arial","Calibri","Georgia","Verdana","Courier New"]

    def __init__(self, parent, sc_gs: dict):
        super().__init__(parent)
        self.title("Налаштування матриці розсіювання")
        self.resizable(False, False); set_icon(self)
        self.sc_gs = dict(sc_gs); self.result = None
        self._pt_color  = sc_gs.get("sc_point_color",  "#4c72b0")
        self._tr_color  = sc_gs.get("sc_trend_color",  "#c62828")
        self._hist_col  = sc_gs.get("sc_hist_color",   "#4c72b0")

        frm = tk.Frame(self, padx=16, pady=14); frm.pack(fill=tk.BOTH, expand=True)

        self._pt_size  = tk.IntVar(value=sc_gs.get("sc_point_size",  14))
        self._pt_alpha = tk.DoubleVar(value=sc_gs.get("sc_point_alpha", 0.75))
        self._show_tr  = tk.BooleanVar(value=sc_gs.get("sc_show_trend", True))
        self._tr_width = tk.DoubleVar(value=sc_gs.get("sc_trend_width", 0.9))
        self._ff       = tk.StringVar(value=sc_gs.get("font_family","Times New Roman"))
        self._fz       = tk.IntVar(value=sc_gs.get("sc_font_size", 6))

        rows_cfg = [
            ("Шрифт:",                  "combo",  self._ff,       self.FONTS),
            ("Розмір підписів:",        "spin",   self._fz,       (5, 18)),
            ("Розмір точок:",           "spin",   self._pt_size,  (3, 50)),
            ("Прозорість точок (0-1):", "scale",  self._pt_alpha, (0.1, 1.0)),
            ("Показувати лінію тренду:","check",  self._show_tr,  None),
            ("Товщина лінії тренду:",   "scale",  self._tr_width, (0.3, 3.0)),
        ]
        self._btn_refs = {}
        r = 0
        for lbl, wtype, var, opts in rows_cfg:
            tk.Label(frm, text=lbl, font=("Times New Roman",12)
                     ).grid(row=r, column=0, sticky="w", pady=5)
            if wtype == "combo":
                ttk.Combobox(frm, textvariable=var, values=opts,
                             state="readonly", width=20).grid(row=r, column=1, sticky="w", padx=8)
            elif wtype == "spin":
                tk.Spinbox(frm, from_=opts[0], to=opts[1], textvariable=var,
                           width=7).grid(row=r, column=1, sticky="w", padx=8)
            elif wtype == "scale":
                tk.Scale(frm, from_=opts[0], to=opts[1], resolution=0.05,
                         orient="horizontal", variable=var,
                         length=160).grid(row=r, column=1, sticky="w", padx=8)
            elif wtype == "check":
                tk.Checkbutton(frm, variable=var).grid(row=r, column=1, sticky="w", padx=8)
            r += 1

        # Colour pickers
        for lbl, attr, init in [
            ("Колір точок:",       "_pt_color",  self._pt_color),
            ("Колір лінії тренду:","_tr_color",  self._tr_color),
            ("Колір гістограм:",   "_hist_col",  self._hist_col),
        ]:
            tk.Label(frm, text=lbl, font=("Times New Roman",12)
                     ).grid(row=r, column=0, sticky="w", pady=5)
            btn = tk.Button(frm, width=6, relief=tk.SUNKEN, bg=init,
                            command=lambda a=attr: self._pick(a))
            btn.grid(row=r, column=1, sticky="w", padx=8)
            self._btn_refs[attr] = btn; r += 1

        bf = tk.Frame(frm); bf.grid(row=r, column=0, columnspan=2, pady=(14,0))
        tk.Button(bf, text="OK", width=10, bg="#c62828", fg="white",
                  font=("Times New Roman",12), command=self._ok).pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text="Скасувати", width=12,
                  font=("Times New Roman",12), command=self.destroy).pack(side=tk.LEFT)
        self.update_idletasks(); center_win(self); self.grab_set()

    def _pick(self, attr):
        c = colorchooser.askcolor(color=getattr(self, attr), parent=self,
                                  title="Виберіть колір")
        if c and c[1]:
            setattr(self, attr, c[1])
            self._btn_refs[attr].configure(bg=c[1])

    def _ok(self):
        self.result = dict(self.sc_gs)
        self.result.update({
            "sc_point_color":  self._pt_color,
            "sc_trend_color":  self._tr_color,
            "sc_hist_color":   self._hist_col,
            "sc_point_size":   self._pt_size.get(),
            "sc_point_alpha":  self._pt_alpha.get(),
            "sc_show_trend":   self._show_tr.get(),
            "sc_trend_width":  self._tr_width.get(),
            "font_family":     self._ff.get(),
            "sc_font_size":    self._fz.get(),
        })
        self.destroy()

# ═══════════════════════════════════════════════════════════════
# CORRELATION ANALYSIS WINDOW
# ═══════════════════════════════════════════════════════════════
class CorrelationWindow:

    HELP_TEXT = """
КОРЕЛЯЦІЙНИЙ АНАЛІЗ — ПОКРОКОВА ІНСТРУКЦІЯ
═══════════════════════════════════════════

ЩО ТАКЕ КОРЕЛЯЦІЯ?
  Кореляція показує наявність і силу статистичного зв'язку
  між двома показниками.
  Коефіцієнт кореляції r (або ρ) від -1 до +1:
    r = +1 → ідеальний прямий зв'язок
    r = -1 → ідеальний обернений зв'язок
    r = 0  → зв'язку немає

КРОК 1. ПІДГОТОВКА ТАБЛИЦІ ДАНИХ
  Кожен стовпець таблиці = один показник (змінна).
  Перейменуйте стовпці — двічі клікніть на заголовок
  (синя клітинка вгорі) і введіть назву показника.

  Кожен рядок = одне спостереження (рослина, ділянка, рік).
  Введіть числові дані у клітинки.

  Приклад:
  | Врожайність | Висота | Маса зерна |
  |    4.2      |  98.5  |   38.2     |
  |    5.1      | 103.2  |   41.5     |

  Мінімум: 2 стовпці (показники) по ≥ 3 значення.

  Вставте дані з Excel: скопіюйте в Excel → клік на першу
  клітинку таблиці → кнопка «Вставити з Excel».

КРОК 2. ЗАПУСК АНАЛІЗУ
  Натисніть «▶ Аналіз» → з'явиться вікно параметрів.

КРОК 3. ВИБІР МЕТОДУ КОРЕЛЯЦІЇ

  Авто (рекомендовано):
    Програма перевіряє нормальність кожного показника.
    Якщо всі нормальні → Пірсон.
    Якщо хоч один ненормальний → Спірмен.

  Пірсон r:
    Для нормально розподілених, неперервних даних.
    Вимірює ЛІНІЙНИЙ зв'язок.
    ⚠ При порушенні нормальності — програма попередить!

  Спірмен ρ (rho):
    Непараметричний, для будь-якого розподілу.
    Виявляє МОНОТОННИЙ зв'язок (не лише лінійний).
    Надійніший при наявності викидів.

КРОК 4. ПОПРАВКА НА МНОЖИННІ ПОРІВНЯННЯ
  При n показниках виконується n×(n-1)/2 тестів.
  Без поправки ризик хибних результатів зростає!

  Бонферроні (строга):
    p_скор = p × кількість_пар
    Рекомендується при ≤ 10 показниках.

  Benjamini-Hochberg/FDR (ліберальніша):
    Контролює частку хибних відкриттів.
    Рекомендується при > 10 показниках.

КРОК 5. ІНТЕРПРЕТАЦІЯ ТЕПЛОВОЇ КАРТИ

  В кожній клітинці три рядки:
    r = -0.82     ← коефіцієнт кореляції
    p = 0.003     ← p-значення після поправки
    n = 15        ← кількість пар спостережень

  Значущість позначається зірочками:
    * — p < 0.05  (значущий зв'язок)
    ** — p < 0.01 (високо значущий зв'язок)
    без зірочки — p ≥ 0.05 (зв'язок незначущий)

  Колір:
    Зелений → позитивна кореляція (r > 0)
    Червоний → негативна кореляція (r < 0)
    Жовтий/білий → зв'язок відсутній

КРОК 6. СИЛА ЗВ'ЯЗКУ — ІНТЕРПРЕТАЦІЯ |r|
  0.00 – 0.19:  дуже слабкий (практично відсутній)
  0.20 – 0.39:  слабкий
  0.40 – 0.59:  помірний (середній)
  0.60 – 0.79:  сильний
  0.80 – 1.00:  дуже сильний

  Для агрономічних досліджень:
  |r| ≥ 0.60 — практично значущий зв'язок.

КРОК 7. МАТРИЦЯ ДІАГРАМ РОЗСІЮВАННЯ
  Відкривається автоматично після теплової карти.
  По діагоналі — гістограма кожного показника.
  Поза діагоналлю — точкові діаграми кожної пари.
  Червона лінія — лінія тренду.

  Якщо точки лежать уздовж прямої → сильна лінійна кореляція.
  Якщо точки хаотичні → зв'язку немає.
  Якщо є криволінійна залежність → розгляньте Спірмена.

ВАЖЛИВО:
  ⚠ Кореляція ≠ Причинно-наслідковий зв'язок!
  Навіть дуже сильна кореляція не означає що один показник
  СПРИЧИНЯЄ зміну іншого. Для висновків про причинність
  потрібен регресійний аналіз або теоретичне обґрунтування.
"""

    def __init__(self, root, graph_settings):
        self.root = root
        self.gs = dict(graph_settings)
        self._hm_fig = None
        self._sc_fig = None

        self.win = tk.Toplevel(root)
        self.win.title("Кореляційний аналіз")
        self.win.geometry("1060x660"); set_icon(self.win)

        self._build_menu()
        self._build_toolbar()
        self._build_table()

    # ── Меню ──────────────────────────────────────────────────
    def _build_menu(self):
        mb = tk.Menu(self.win)
        fm = tk.Menu(mb, tearoff=0)
        fm.add_command(label="Зберегти проект", command=self._save_proj)
        fm.add_command(label="Відкрити проект", command=self._load_proj)
        fm.add_separator()
        fm.add_command(label="Завантажити Excel", command=self._load_excel)
        mb.add_cascade(label="Файл", menu=fm)
        em = tk.Menu(mb, tearoff=0)
        em.add_command(label="Додати рядок",     command=self.add_row)
        em.add_command(label="Видалити рядок",   command=self.del_row)
        em.add_command(label="Додати стовпчик",  command=self.add_col)
        em.add_command(label="Видалити стовпчик",command=self.del_col)
        mb.add_cascade(label="Правка", menu=em)
        self.win.config(menu=mb)

    # ── Панель інструментів ───────────────────────────────────
    def _build_toolbar(self):
        tb = tk.Frame(self.win, padx=6, pady=5); tb.pack(fill=tk.X)
        tk.Button(tb, text="▶ Аналіз", bg="#c62828", fg="white",
                  font=("Times New Roman", 13),
                  command=self._run_analysis).pack(side=tk.LEFT, padx=4)
        self._settings_btn = tk.Menubutton(tb, text="⚙ Налаштування ▾",
                                           font=("Times New Roman", 11),
                                           relief=tk.RAISED, bd=2)
        self._settings_btn.pack(side=tk.LEFT, padx=4)
        sm = tk.Menu(self._settings_btn, tearoff=0)
        sm.add_command(label="Додати рядок",      command=self.add_row)
        sm.add_command(label="Видалити рядок",    command=self.del_row)
        sm.add_separator()
        sm.add_command(label="Додати стовпець",   command=self.add_col)
        sm.add_command(label="Видалити стовпець", command=self.del_col)
        sm.add_separator()
        sm.add_command(label="🗑 Очистити таблицю", command=self._clear_table)
        self._settings_btn["menu"] = sm
        tk.Button(tb, text="Вставити з буфера",
                  font=("Times New Roman", 11),
                  command=self._paste).pack(side=tk.LEFT, padx=4)
        tk.Button(tb, text="📂 Відкрити показники", bg="#1a6b8c", fg="white",
                  font=("Times New Roman", 11),
                  command=self._open_indicators_from_journal).pack(side=tk.LEFT, padx=4)
        tk.Button(tb, text="📚 Довідка", bg="#1a4b8c", fg="white",
                  font=("Times New Roman", 11),
                  command=self._show_help).pack(side=tk.LEFT, padx=4)
        tk.Label(tb,
                 text="Двічі клікніть на синій заголовок щоб перейменувати показник",
                 font=("Times New Roman", 9), fg="#666"
                 ).pack(side=tk.LEFT, padx=10)

    def _clear_table(self):
        if not messagebox.askyesno("Очистити таблицю",
                "Видалити всі числові дані?\n(Назви стовпців залишаться)"):
            return
        for row in self.entries:
            for e in row: e.delete(0, tk.END)

    # ── Таблиця даних ─────────────────────────────────────────
    def _build_table(self):
        self.rows = 14; self.cols = 6
        tbl_frm = tk.Frame(self.win); tbl_frm.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.canvas = tk.Canvas(tbl_frm)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(tbl_frm, orient="vertical", command=self.canvas.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=sb.set)
        self.inner = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>",
                        lambda e: self.canvas.config(scrollregion=self.canvas.bbox("all")))
        self.win.bind("<MouseWheel>",
                      lambda e: self.canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        self.header_labels = []
        self.header_vars   = []
        self._build_headers()
        self.entries = []
        for i in range(self.rows):
            self._add_row_widgets(i)
        _bind_nav(self.entries, self.win)
        _bind_fill_handle(self.entries, self.win)

    def _build_headers(self):
        for j in range(self.cols):
            var = tk.StringVar(value=f"Показник {j+1}")
            self.header_vars.append(var)
            lbl = tk.Label(self.inner, textvariable=var,
                           relief=tk.RIDGE, width=14, cursor="hand2",
                           bg="#1a4b8c", fg="white",
                           font=("Times New Roman", 11, "bold"))
            lbl.grid(row=0, column=j, padx=2, pady=2, sticky="nsew")
            lbl.bind("<Double-Button-1>", lambda e, idx=j: self._rename_col(idx))
            self.header_labels.append(lbl)

    def _rename_col(self, idx):
        dlg = tk.Toplevel(self.win); dlg.title("Перейменувати показник")
        dlg.resizable(False, False); dlg.grab_set(); set_icon(dlg)
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
                  font=("Times New Roman", 12), command=apply).pack(pady=(4,14))
        dlg.bind("<Return>", lambda ev: apply())
        center_win(dlg)

    def _add_row_widgets(self, i):
        row_ = []
        for j in range(self.cols):
            e = tk.Entry(self.inner, width=14, font=("Times New Roman", 11),
                         highlightthickness=1, highlightbackground="#c0c0c0")
            e.grid(row=i+1, column=j, padx=2, pady=2)
            e.bind("<Return>", self._on_enter)
            e.bind("<Tab>",    self._on_tab)
            row_.append(e)
        self.entries.append(row_)

    def add_row(self):
        i = len(self.entries)
        self._add_row_widgets(i); self.rows += 1
        _bind_nav(self.entries, self.win)
        _bind_fill_handle(self.entries, self.win)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def del_row(self):
        if not self.entries: return
        for e in self.entries.pop(): e.destroy()
        self.rows -= 1
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def add_col(self):
        ci = self.cols; self.cols += 1
        var = tk.StringVar(value=f"Показник {ci+1}")
        self.header_vars.append(var)
        lbl = tk.Label(self.inner, textvariable=var,
                       relief=tk.RIDGE, width=14, cursor="hand2",
                       bg="#1a4b8c", fg="white",
                       font=("Times New Roman", 11, "bold"))
        lbl.grid(row=0, column=ci, padx=2, pady=2, sticky="nsew")
        lbl.bind("<Double-Button-1>", lambda e, idx=ci: self._rename_col(idx))
        self.header_labels.append(lbl)
        for i, row_ in enumerate(self.entries):
            e = tk.Entry(self.inner, width=14, font=("Times New Roman", 11),
                         highlightthickness=1, highlightbackground="#c0c0c0")
            e.grid(row=i+1, column=ci, padx=2, pady=2)
            e.bind("<Return>", self._on_enter); e.bind("<Tab>", self._on_tab)
            row_.append(e)
        _bind_nav(self.entries, self.win)
        _bind_fill_handle(self.entries, self.win)

    def del_col(self):
        if self.cols <= 2: return
        self.header_labels.pop().destroy()
        self.header_vars.pop()
        for row_ in self.entries: row_.pop().destroy()
        self.cols -= 1

    # ── Навігація ─────────────────────────────────────────────
    def _on_enter(self, event):
        for i, row_ in enumerate(self.entries):
            for j, e in enumerate(row_):
                if e is event.widget:
                    if i+1 >= len(self.entries): self.add_row()
                    self.entries[i+1][j].focus_set(); return "break"
        return "break"

    def _on_tab(self, event):
        for i, row_ in enumerate(self.entries):
            for j, e in enumerate(row_):
                if e is event.widget:
                    nj = j+1; ni = i
                    if nj >= self.cols: nj = 0; ni = i+1
                    if ni >= len(self.entries): self.add_row()
                    self.entries[ni][nj].focus_set(); return "break"
        return "break"

    # ── Вставка, збереження, завантаження ────────────────────
    def _open_indicators_from_journal(self):
        """Завантажує кілька показників із журналу, усереднених до рівня
        варіанту (спершу в межах повторності, потім по повторностях) —
        кожен показник стає стовпцем, кожен варіант — рядком."""
        result = open_indicators_for_variant_analysis(self.win, multi_select=True)
        if result is None: return
        factor_cols, rows, record_names = result

        for w in self.header_labels: w.destroy()
        for row_ in self.entries:
            for e in row_: e.destroy()
        self.header_labels = []; self.header_vars = []; self.entries = []
        self.cols = len(record_names); self.rows = len(rows)
        self._build_headers()
        for rn_idx, rn in enumerate(record_names):
            self.header_vars[rn_idx].set(rn)
        for i in range(self.rows):
            self._add_row_widgets(i)
        for i, r in enumerate(rows):
            for j, rn in enumerate(record_names):
                v = r[rn]
                if v is not None:
                    self.entries[i][j].insert(0, str(v))
        _bind_nav(self.entries, self.win)
        _bind_fill_handle(self.entries, self.win)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        messagebox.showinfo("Дані перенесено",
            f"Перенесено {len(record_names)} показники по {len(rows)} варіантах "
            f"(середнє в межах повторності, потім по повторностях). Рядок — "
            f"варіант, стовпець — показник.")

    def _paste(self):
        """Вставити дані з буфера обміну.
        Якщо активна клітинка Entry — вставляємо з неї.
        Інакше — з клітинки (0,0)."""
        try: data = self.win.clipboard_get()
        except Exception:
            messagebox.showwarning("Буфер порожній",
                "Буфер обміну порожній або не містить тексту.\n"
                "Скопіюйте дані з Excel (Ctrl+C) і спробуйте знову."); return
        if not data.strip(): return
        w = self.win.focus_get()
        pos = (0, 0)
        if isinstance(w, tk.Entry):
            for i, row_ in enumerate(self.entries):
                for j, e in enumerate(row_):
                    if e is w: pos = (i, j); break
                if pos != (0, 0): break
        r0, c0 = pos
        rows_data = [rt for rt in data.splitlines() if rt.strip()]
        if not rows_data: return

        # Визначаємо потрібні розміри таблиці
        max_cols_needed = c0 + max((len(rt.split("\t")) for rt in rows_data), default=1)
        max_rows_needed = r0 + len(rows_data)

        # Додаємо стовпці якщо потрібно
        while self.cols < max_cols_needed:
            self.add_col()

        # Вставляємо дані
        for ir, rt in enumerate(rows_data):
            rr = r0 + ir
            while rr >= len(self.entries): self.add_row()
            for jc, val in enumerate(rt.split("\t")):
                cc = c0 + jc
                if cc >= self.cols: break
                self.entries[rr][cc].delete(0, tk.END)
                self.entries[rr][cc].insert(0, val.strip())

        # Інформуємо якщо таблицю розширено
        added_cols = max_cols_needed - (c0 + 1) if max_cols_needed > self.cols else 0
        added_rows = max_rows_needed - len(self.entries) if max_rows_needed > len(self.entries) else 0

        # Фокус на першу вставлену клітинку
        if self.entries and r0 < len(self.entries):
            self.entries[r0][c0].focus_set()

    def _save_proj(self):
        path = filedialog.asksaveasfilename(
            parent=self.win, defaultextension=".sadp",
            filetypes=[("SAD проект","*.sadp"),("JSON","*.json")])
        if not path: return
        d = {"type":"correlation","version":APP_VER,
             "headers":[v.get() for v in self.header_vars],
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
        headers = d.get("headers",[])
        rd      = d.get("rows_data",[])
        while self.cols < len(headers): self.add_col()
        for j, h in enumerate(headers):
            if j < len(self.header_vars): self.header_vars[j].set(h)
        while len(self.entries) < len(rd): self.add_row()
        for i, rv in enumerate(rd):
            for j, v in enumerate(rv):
                if i<len(self.entries) and j<len(self.entries[i]):
                    self.entries[i][j].delete(0,tk.END); self.entries[i][j].insert(0,v)

    def _load_excel(self):
        if not HAS_OPENPYXL: messagebox.showerror("","pip install openpyxl"); return
        path = filedialog.askopenfilename(parent=self.win,
                    filetypes=[("Excel","*.xlsx *.xlsm *.xls")])
        if not path: return
        try:
            wb = openpyxl.load_workbook(path,data_only=True,read_only=True)
            raw = [[cell for cell in row] for row in wb.active.iter_rows(values_only=True)]
            wb.close()
        except Exception as ex: messagebox.showerror("",str(ex)); return
        while raw and all(v is None for v in raw[-1]): raw.pop()
        if not raw: return
        nc = max(len(r) for r in raw)
        while self.cols < nc: self.add_col()
        while len(self.entries) < len(raw): self.add_row()
        for i,row in enumerate(raw):
            for j,v in enumerate(row):
                if j>=self.cols: break
                cv = "" if v is None else str(v).replace(",",".")
                self.entries[i][j].delete(0,tk.END); self.entries[i][j].insert(0,cv)

    def _settings(self):
        dlg = GraphSettingsDlg(self.win, self.gs, show_heatmap=True)
        self.win.wait_window(dlg)
        if dlg.result: self.gs = dlg.result

    # ── Копіювання ────────────────────────────────────────────
    def _copy_heatmap(self):
        if self._hm_fig is None:
            messagebox.showwarning("","Спочатку виконайте аналіз."); return
        ok, msg = _copy_fig_to_clipboard(self._hm_fig)
        if ok: messagebox.showinfo("","Теплову карту скопійовано.\nВставте у Word через Ctrl+V.")
        else:   messagebox.showwarning("",f"Помилка: {msg}")

    # ── Довідка ───────────────────────────────────────────────
    def _show_help(self):
        win = tk.Toplevel(self.win)
        win.title("Довідка — Кореляційний аналіз")
        win.geometry("700x640"); set_icon(win)
        frm = tk.Frame(win); frm.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        vsb = ttk.Scrollbar(frm, orient="vertical"); vsb.pack(side=tk.RIGHT, fill=tk.Y)
        txt = tk.Text(frm, wrap="word", font=("Times New Roman",11),
                      yscrollcommand=vsb.set, relief=tk.FLAT,
                      bg="#fafafa", padx=10, pady=8, cursor="arrow")
        txt.pack(fill=tk.BOTH, expand=True)
        vsb.config(command=txt.yview)
        txt.insert("1.0", self.HELP_TEXT.strip())
        txt.configure(state="disabled")
        txt.bind("<MouseWheel>",
                 lambda e: txt.yview_scroll(int(-1*(e.delta/120)), "units"))
        tk.Button(win, text="Закрити", command=win.destroy,
                  font=("Times New Roman",11)).pack(pady=6)

    # ── Діалог параметрів і запуск ────────────────────────────
    def _run_analysis(self):
        """Спрощений діалог параметрів — без блоку 'де знаходяться назви'
        (назви беруться з заголовків стовпців які користувач перейменував)."""
        dlg = tk.Toplevel(self.win)
        dlg.title("Параметри кореляційного аналізу")
        dlg.resizable(False, False); set_icon(dlg)
        frm = tk.Frame(dlg, padx=20, pady=16); frm.pack(fill=tk.BOTH, expand=True)
        rb_f = ("Times New Roman", 12)

        # ── Метод кореляції ──────────────────────────────────
        tk.Label(frm, text="Метод кореляції:",
                 font=("Times New Roman", 12, "bold")).grid(
                 row=0, column=0, columnspan=2, sticky="w", pady=0)
        meth_var = tk.StringVar(value="auto")
        methods = [
            ("auto",
             "Авто (рекомендовано) — перевіряє нормальність:\n"
             "  ✓ всі нормальні → Пірсон\n"
             "  ✓ хоч один ненормальний → Спірмен"),
            ("pearson",
             "Пірсон r — лінійний зв'язок, нормальний розподіл\n"
             "  ⚠ При ненормальних даних програма попередить"),
            ("spearman",
             "Спірмен ρ — непараметричний, будь-який розподіл,\n"
             "  монотонний зв'язок, стійкий до викидів"),
        ]
        for ri, (val, txt_) in enumerate(methods):
            tk.Radiobutton(frm, text=txt_, variable=meth_var, value=val,
                           font=rb_f, justify="left", wraplength=440
                           ).grid(row=1+ri, column=0, columnspan=2, sticky="w", pady=3)

        ttk.Separator(frm, orient="horizontal").grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=10)

        # ── Поправка на множинні порівняння ──────────────────
        tk.Label(frm, text="Поправка на множинні порівняння:",
                 font=("Times New Roman", 12, "bold")).grid(
                 row=5, column=0, columnspan=2, sticky="w", pady=0)
        corr_var = tk.StringVar(value="bonferroni")
        corrections = [
            ("bonferroni",
             "Бонферроні — строга, контролює сімейну помилку (FWER)\n"
             "  Рекомендується при ≤ 10 показниках"),
            ("bh",
             "Benjamini–Hochberg (FDR) — ліберальніша, більша потужність\n"
             "  Рекомендується при > 10 показниках"),
            ("none",
             "Без поправки — не рекомендується при > 3 показниках"),
        ]
        for ri, (val, txt_) in enumerate(corrections):
            tk.Radiobutton(frm, text=txt_, variable=corr_var, value=val,
                           font=rb_f, justify="left", wraplength=440
                           ).grid(row=6+ri, column=0, columnspan=2, sticky="w", pady=3)

        ttk.Separator(frm, orient="horizontal").grid(
            row=9, column=0, columnspan=2, sticky="ew", pady=10)

        # ── Рівень значущості ─────────────────────────────────
        tk.Label(frm, text="Рівень значущості α:",
                 font=("Times New Roman", 12, "bold")).grid(row=10, column=0, sticky="w")
        alpha_var = tk.StringVar(value="0.05")
        ttk.Combobox(frm, textvariable=alpha_var, values=["0.01","0.05","0.10"],
                     state="readonly", width=9,
                     font=("Times New Roman",12)).grid(row=10, column=1, sticky="w", padx=8)

        # ── Кнопки ───────────────────────────────────────────
        bf = tk.Frame(frm); bf.grid(row=11, column=0, columnspan=2, pady=(16,0))
        out = {"ok": False}

        def ok():
            out.update({"ok": True,
                        "method":     meth_var.get(),
                        "correction": corr_var.get(),
                        "alpha":      float(alpha_var.get())})
            dlg.destroy()

        tk.Button(bf, text="▶ Виконати аналіз", width=20,
                  bg="#c62828", fg="white",
                  font=("Times New Roman",12), command=ok).pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text="Скасувати", width=12,
                  font=("Times New Roman",12), command=dlg.destroy).pack(side=tk.LEFT, padx=4)

        dlg.update_idletasks()
        center_win(dlg)
        dlg.bind("<Return>", lambda e: ok())
        dlg.grab_set(); self.win.wait_window(dlg)
        if not out["ok"]: return

        # Назви показників — завжди з заголовків стовпців
        self._compute_and_show(out["method"], out["alpha"], out["correction"])

    # ── Обчислення кореляцій ──────────────────────────────────
    def _compute_and_show(self, method, alpha, correction="bonferroni"):
        """
        Читає дані зі стовпців таблиці.
        Назви показників = заголовки стовпців (header_vars).
        Кожен стовпець = один показник, рядки = спостереження.
        """
        # Збираємо дані по стовпцях
        labels = []; data_cols = []
        for j in range(self.cols):
            col_name = self.header_vars[j].get().strip() or f"Показник {j+1}"
            col_vals = []
            for row in self.entries:
                v = row[j].get().strip() if j < len(row) else ""
                if not v: continue
                try: col_vals.append(float(v.replace(",",".")))
                except Exception: continue
            if len(col_vals) >= 3:
                labels.append(col_name)
                data_cols.append(col_vals)

        if len(data_cols) < 2:
            messagebox.showwarning("Замало даних",
                "Потрібно ≥ 2 стовпці з даними (≥ 3 значення у кожному).\n\n"
                "Переконайтесь що дані введені у числовому форматі "
                "і кожен стовпець містить хоча б 3 числа."); return

        n = len(labels)
        arrays = [np.array(d, dtype=float) for d in data_cols]
        n_mat  = np.zeros((n, n), dtype=int)

        # ── Авто-вибір / перевірка Пірсона ───────────────────
        actual_method = method
        if method in ("auto", "pearson"):
            non_normal = []
            for i, arr in enumerate(arrays):
                if len(arr) < 3: continue
                try:
                    _, p_sw = shapiro(arr)
                    if p_sw <= 0.05: non_normal.append(labels[i])
                except Exception: pass

            if method == "auto":
                if non_normal:
                    messagebox.showinfo("Авто-вибір методу",
                        f"Показники з ненормальним розподілом:\n"
                        f"{', '.join(non_normal[:5])}\n\n"
                        "Автоматично обрано: Спірмен (непараметричний).")
                    actual_method = "spearman"
                else:
                    messagebox.showinfo("Авто-вибір методу",
                        "Всі показники відповідають нормальному розподілу.\n"
                        "Автоматично обрано: Пірсон.")
                    actual_method = "pearson"
            elif method == "pearson" and non_normal:
                ans = messagebox.askyesno(
                    "Увага: нормальність порушена",
                    f"Показники з ненормальним розподілом:\n"
                    f"{', '.join(non_normal[:5])}\n\n"
                    "Кореляція Пірсона передбачає нормальний розподіл.\n"
                    "Рекомендується: Спірмен.\n\n"
                    "Продовжити з Пірсоном попри порушення?")
                if not ans: return

        # ── Попарна кореляційна матриця ───────────────────────
        r_mat = np.full((n, n), np.nan)
        p_mat = np.full((n, n), np.nan)
        np.fill_diagonal(r_mat, 1.0)
        np.fill_diagonal(p_mat, 1.0)
        np.fill_diagonal(n_mat, [len(a) for a in arrays])

        raw_p_pairs = []
        for i in range(n):
            for j in range(i+1, n):
                a = arrays[i]; b = arrays[j]
                min_len = min(len(a), len(b))
                a2 = a[:min_len]; b2 = b[:min_len]
                mask = ~(np.isnan(a2) | np.isnan(b2))
                a2 = a2[mask]; b2 = b2[mask]
                pair_n = len(a2)
                n_mat[i,j] = n_mat[j,i] = pair_n
                if pair_n < 3: continue
                try:
                    if actual_method == "pearson":
                        r_, p_ = pearsonr(a2, b2)
                    else:
                        r_, p_ = spearmanr(a2, b2)
                    r_mat[i,j] = r_mat[j,i] = float(r_)
                    raw_p_pairs.append((i, j, float(p_)))
                except Exception: pass

        if not raw_p_pairs:
            messagebox.showwarning("Замало даних",
                "Жодної пари показників не має ≥ 3 спільних спостережень."); return

        # ── Поправка на множинні порівняння ───────────────────
        m = len(raw_p_pairs)
        if correction == "bonferroni":
            for i, j, p_raw in raw_p_pairs:
                p_mat[i,j] = p_mat[j,i] = min(1.0, p_raw * m)
        elif correction == "bh":
            sp = sorted(raw_p_pairs, key=lambda x: x[2])
            bh = np.array([p for _,_,p in sp])
            for k in range(len(bh)-1,-1,-1):
                bh[k] = min(1.0, bh[k]*len(bh)/(k+1))
                if k < len(bh)-1: bh[k] = min(bh[k], bh[k+1])
            for idx,(i,j,_) in enumerate(sp):
                p_mat[i,j] = p_mat[j,i] = float(bh[idx])
        else:
            for i,j,p_raw in raw_p_pairs:
                p_mat[i,j] = p_mat[j,i] = p_raw

        corr_label = {"bonferroni":"Бонферроні","bh":"BH/FDR","none":"без поправки"
                      }.get(correction, correction)

        # ── Показуємо результати в одному вікні з вкладками ──
        self._show_results(labels, r_mat, p_mat, n_mat, alpha,
                           actual_method, corr_label, arrays)

    # ── Об'єднане вікно результатів ───────────────────────────
    def _show_results(self, labels, r_mat, p_mat, n_mat, alpha,
                      method, corr_label, arrays):
        if not HAS_MPL:
            messagebox.showwarning("","matplotlib недоступний."); return

        win = tk.Toplevel(self.win)
        win.title("Кореляційний аналіз — результати")
        win.geometry("1100x800"); set_icon(win)
        win.resizable(True, True)

        # Зберігаємо дані для перебудови
        self._res_win    = win
        self._res_labels = labels; self._res_r = r_mat
        self._res_p      = p_mat;  self._res_n = n_mat
        self._res_alpha  = alpha;  self._res_method = method
        self._res_corr   = corr_label; self._res_arrays = arrays
        self._hm_title   = getattr(self, '_hm_title', "")
        self._sc_title   = getattr(self, '_sc_title', "")

        meth_lbl = "Пірсон" if method == "pearson" else "Спірмен"

        # ── Перемикач вкладок через кнопки (не ttk.Notebook) ─
        # ttk.Notebook на Windows приховує текст активної вкладки
        switch_f = tk.Frame(win, bg="#1a4b8c"); switch_f.pack(fill=tk.X)

        content_f = tk.Frame(win); content_f.pack(fill=tk.BOTH, expand=True)

        self._hm_frame = None
        self._sc_frame = None
        _hm_outer = tk.Frame(content_f); _hm_outer.pack(fill=tk.BOTH, expand=True)
        _sc_outer = tk.Frame(content_f); _sc_outer.pack(fill=tk.BOTH, expand=True)
        _sc_outer.pack_forget()   # ховаємо другу

        _active_tab = [0]
        _btn_hm = _btn_sc = None

        def _show_hm():
            _sc_outer.pack_forget()
            _hm_outer.pack(fill=tk.BOTH, expand=True)
            _active_tab[0] = 0
            _btn_hm.configure(bg="white", fg="#1a4b8c", relief=tk.FLAT,
                               font=("Times New Roman",11,"bold"))
            _btn_sc.configure(bg="#3d72b4", fg="white", relief=tk.FLAT,
                               font=("Times New Roman",11))
        def _show_sc():
            _hm_outer.pack_forget()
            _sc_outer.pack(fill=tk.BOTH, expand=True)
            _active_tab[0] = 1
            _btn_sc.configure(bg="white", fg="#1a4b8c", relief=tk.FLAT,
                               font=("Times New Roman",11,"bold"))
            _btn_hm.configure(bg="#3d72b4", fg="white", relief=tk.FLAT,
                               font=("Times New Roman",11))

        _btn_hm = tk.Button(switch_f, text="  🌡  Теплова карта  ",
                            bg="white", fg="#1a4b8c",
                            font=("Times New Roman",11,"bold"),
                            relief=tk.FLAT, padx=16, pady=8,
                            cursor="hand2", command=_show_hm)
        _btn_hm.pack(side=tk.LEFT)
        _btn_sc = tk.Button(switch_f, text="  ⬡  Матриця розсіювання  ",
                            bg="#3d72b4", fg="white",
                            font=("Times New Roman",11),
                            relief=tk.FLAT, padx=16, pady=8,
                            cursor="hand2", command=_show_sc)
        _btn_sc.pack(side=tk.LEFT)
        tk.Label(switch_f, bg="#1a4b8c",
                 text=f"  Метод: {meth_lbl}  |  α={alpha}  |  Поправка: {corr_label}",
                 font=("Times New Roman",9), fg="#a0b8cc").pack(side=tk.LEFT, padx=8)

        # ── Теплова карта ──────────────────────────────────────
        tb1 = tk.Frame(_hm_outer, bg="#f0f0f0", padx=6, pady=4); tb1.pack(fill=tk.X)
        for btxt, bcmd, bcol in [
            ("💾 Зберегти PNG", lambda: self._save_fig_png(self._hm_fig,"теплова_карта"), None),
            ("📋 Копіювати",    self._copy_heatmap, None),
            ("⚙ Налаштування", lambda: self._settings_heatmap(), "#1a4b8c"),
        ]:
            kw = {"bg": bcol, "fg": "white"} if bcol else {}
            tk.Button(tb1, text=btxt, font=("Times New Roman",10),
                      relief=tk.FLAT, padx=8, pady=3, cursor="hand2",
                      command=bcmd, **kw).pack(side=tk.RIGHT, padx=3)
        self._hm_frame = tk.Frame(_hm_outer)
        self._hm_frame.pack(fill=tk.BOTH, expand=True)

        # ── Матриця розсіювання ────────────────────────────────
        tb2 = tk.Frame(_sc_outer, bg="#f0f0f0", padx=6, pady=4); tb2.pack(fill=tk.X)
        for btxt, bcmd, bcol in [
            ("💾 Зберегти PNG", lambda: self._save_fig_png(self._sc_fig,"матриця_розсіювання"), None),
            ("📋 Копіювати",    lambda: self._copy_scatter(), None),
            ("⚙ Налаштування", lambda: self._settings_scatter(labels,arrays,method), "#1a4b8c"),
        ]:
            kw = {"bg": bcol, "fg": "white"} if bcol else {}
            tk.Button(tb2, text=btxt, font=("Times New Roman",10),
                      relief=tk.FLAT, padx=8, pady=3, cursor="hand2",
                      command=bcmd, **kw).pack(side=tk.RIGHT, padx=3)
        self._sc_frame = tk.Frame(_sc_outer)
        self._sc_frame.pack(fill=tk.BOTH, expand=True)

        # Будуємо обидва графіки одразу
        self._draw_heatmap(self._hm_frame, labels, r_mat, p_mat, n_mat,
                           alpha, method, corr_label, self.gs)
        self._draw_scatter(self._sc_frame, labels, arrays, method)

    def _save_fig_png(self, fig, name="графік"):
        if fig is None: messagebox.showwarning("","Спочатку виконайте аналіз."); return
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

    def _settings_heatmap(self):
        dlg = HeatmapSettingsDlg(self._res_win, self.gs)
        self._res_win.wait_window(dlg)
        if not dlg.result: return
        self.gs = dlg.result
        # Заголовок
        self._ask_title_hm()

    def _ask_title_hm(self):
        dlg = tk.Toplevel(self._res_win); dlg.title("Заголовок теплової карти")
        dlg.resizable(False, False); dlg.grab_set(); set_icon(dlg)
        tk.Label(dlg, text="Заголовок графіка:",
                 font=("Times New Roman",12)).pack(padx=16, pady=(14,4))
        tv = tk.StringVar(value=self._hm_title)
        te = tk.Entry(dlg, textvariable=tv, font=("Times New Roman",12), width=40)
        te.pack(padx=16, pady=4); te.focus_set()
        def _ok():
            self._hm_title = tv.get().strip()
            dlg.destroy()
            self._redraw_heatmap()
        tk.Button(dlg, text="Застосувати", bg="#c62828", fg="white",
                  font=("Times New Roman",12), command=_ok).pack(pady=(4,14))
        dlg.bind("<Return>", lambda e: _ok())
        center_win(dlg)

    def _redraw_heatmap(self):
        for w in self._hm_frame.winfo_children(): w.destroy()
        self._draw_heatmap(self._hm_frame,
                           self._res_labels, self._res_r, self._res_p,
                           self._res_n, self._res_alpha, self._res_method,
                           self._res_corr, self.gs)

    def _settings_scatter(self, labels, arrays, method):
        dlg = tk.Toplevel(self._res_win); dlg.title("Налаштування матриці розсіювання")
        dlg.resizable(False, False); dlg.grab_set(); set_icon(dlg)
        rf = ("Times New Roman",11)
        frm = tk.Frame(dlg, padx=16, pady=12); frm.pack()

        # Заголовок
        tk.Label(frm, text="Заголовок графіка:", font=rf
                 ).grid(row=0, column=0, sticky="w", pady=4)
        tv = tk.StringVar(value=self._sc_title)
        tk.Entry(frm, textvariable=tv, width=34, font=rf
                 ).grid(row=0, column=1, sticky="w", padx=8)

        sc = self._sc_gs
        params = [
            ("Колір точок:",  "sc_point_color", sc.get("sc_point_color","#4c72b0"), "color"),
            ("Колір тренду:", "sc_trend_color", sc.get("sc_trend_color","#c62828"), "color"),
            ("Колір гістограм:", "sc_hist_color", sc.get("sc_hist_color","#4c72b0"), "color"),
            ("Розмір точок:", "sc_point_size",  sc.get("sc_point_size",14), "spin"),
            ("Показати тренд:", "sc_show_trend", sc.get("sc_show_trend",True), "check"),
        ]
        vars_ = {}
        for ri, (lbl, key, default, wtype) in enumerate(params, 1):
            tk.Label(frm, text=lbl, font=rf).grid(row=ri, column=0, sticky="w", pady=3)
            if wtype == "color":
                v = tk.StringVar(value=default)
                vars_[key] = v
                def _pick(v2=v):
                    c = colorchooser.askcolor(color=v2.get(), parent=dlg)
                    if c and c[1]: v2.set(c[1])
                tk.Button(frm, text="Обрати", command=_pick, font=rf
                          ).grid(row=ri, column=1, sticky="w", padx=8)
            elif wtype == "spin":
                v = tk.IntVar(value=int(default))
                vars_[key] = v
                tk.Spinbox(frm, from_=2, to=30, textvariable=v, width=6, font=rf
                           ).grid(row=ri, column=1, sticky="w", padx=8)
            elif wtype == "check":
                v = tk.BooleanVar(value=bool(default))
                vars_[key] = v
                tk.Checkbutton(frm, variable=v
                               ).grid(row=ri, column=1, sticky="w", padx=8)

        def _apply():
            self._sc_title = tv.get().strip()
            for key, v in vars_.items():
                self._sc_gs[key] = v.get()
            dlg.destroy()
            self._redraw_scatter(labels, arrays, method)
        bf = tk.Frame(frm); bf.grid(row=len(params)+2, column=0, columnspan=2, pady=(12,0))
        tk.Button(bf, text="Застосувати", bg="#c62828", fg="white",
                  font=rf, command=_apply).pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text="Скасувати", font=rf,
                  command=dlg.destroy).pack(side=tk.LEFT)
        center_win(dlg)

    def _redraw_scatter(self, labels, arrays, method):
        for w in self._sc_frame.winfo_children(): w.destroy()
        self._draw_scatter(self._sc_frame, labels, arrays, method)

    def _restyle(self, win, labels, r_mat, p_mat, n_mat, alpha, method, corr_label):
        """Залишаємо для сумісності."""
        self._settings_heatmap()

    def _draw_heatmap(self, frame, labels, r_mat, p_mat, n_mat, alpha, method, corr_label, gs):
        n = len(labels)
        dpi = 96
        # Стартовий розмір — буде масштабований через resize_event
        fig  = Figure(figsize=(8, 7), dpi=dpi)
        ax   = fig.add_subplot(111)

        cmap_name = gs.get("heatmap_cmap","RdYlGn")
        fsize     = gs.get("heatmap_font_size", 9)
        acol      = gs.get("heatmap_annot_color","#000000")
        ff        = gs.get("font_family","Times New Roman")

        cmap = get_cmap_safe(cmap_name)

        masked = np.ma.array(r_mat, mask=np.isnan(r_mat))
        im = ax.imshow(masked, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

        meth_full = "Пірсон" if method=="pearson" else "Спірмен"
        custom_title = getattr(self, '_hm_title', "")
        ax.set_title(
            custom_title if custom_title else
            f"Кореляційна матриця ({meth_full}, {corr_label}, α={alpha})\n"
            f"Клітинки: r / p-скор / n   |   * p<α   ** p<α/5",
            fontsize=fsize+1, fontfamily=ff)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=fsize, fontfamily=ff)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=fsize, fontfamily=ff)

        for i in range(n):
            for j in range(n):
                r_ = r_mat[i,j]; p_ = p_mat[i,j]
                if i == j:
                    ax.text(j,i,"—",ha="center",va="center",
                            fontsize=fsize,color=acol,fontfamily=ff)
                    continue
                if math.isnan(r_): continue
                # Позначка значущості рахується саме від ОБРАНОГО користувачем
                # α (не від фіксованих 0.01/0.05) — інакше зміна α у випадаючому
                # списку ніяк не впливала б на те, що видно на карті.
                if math.isnan(p_):
                    mark = ""
                elif p_ < alpha/5:
                    mark = "**"
                elif p_ < alpha:
                    mark = "*"
                else:
                    mark = ""
                p_str  = fmt(p_,3) if not math.isnan(p_) else "н/д"
                n_ij   = int(n_mat[i,j]) if n_mat is not None else 0
                txt_   = f"{r_:.2f}{mark}\np={p_str}\nn={n_ij}"
                ax.text(j,i,txt_,ha="center",va="center",
                        fontsize=max(6,fsize-1),color=acol,fontfamily=ff,linespacing=1.3)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("r", fontsize=fsize, fontfamily=ff)
        fig.tight_layout()
        self._hm_fig = fig

        embed_figure(fig, frame, dpi=dpi)

    # ── Матриця діаграм розсіювання ───────────────────────────
    def _draw_scatter(self, frame, labels, arrays, method):
        if not HAS_MPL: return
        n = len(labels)
        if n < 2: return
        self._sc_gs = getattr(self, "_sc_gs", {
            "sc_point_color":  "#4c72b0",
            "sc_trend_color":  "#c62828",
            "sc_hist_color":   "#4c72b0",
            "sc_point_size":   14,
            "sc_point_alpha":  0.75,
            "sc_show_trend":   True,
            "sc_trend_width":  0.9,
            "font_family":     "Times New Roman",
            "sc_font_size":    8,
        })
        self._sc_labels = labels; self._sc_arrays = arrays; self._sc_method = method
        # Малюємо безпосередньо в переданий frame
        for w in frame.winfo_children(): w.destroy()
        sc = self._sc_gs
        pt_col   = sc.get("sc_point_color",  "#4c72b0")
        tr_col   = sc.get("sc_trend_color",  "#c62828")
        hi_col   = sc.get("sc_hist_color",   "#4c72b0")
        pt_size  = sc.get("sc_point_size",   14)
        pt_alpha = sc.get("sc_point_alpha",  0.75)
        show_tr  = sc.get("sc_show_trend",   True)
        tr_width = sc.get("sc_trend_width",  0.9)
        ff       = sc.get("font_family",     "Times New Roman")
        fz       = sc.get("sc_font_size",    8)

        # Статус-рядок над матрицею: скільки показників, підказка про скрол
        status = tk.Frame(frame, bg="#eef3f8"); status.pack(fill=tk.X)
        tk.Label(status,
                 text=f"Матриця {n}×{n} — усі надані показники. "
                      f"Прокручуйте по вертикалі/горизонталі (Shift+колесо — по горизонталі), "
                      f"щоб знайти потрібну пару.",
                 bg="#eef3f8", fg="#1a4b8c", font=(ff, 9), anchor="w"
                 ).pack(fill=tk.X, padx=8, pady=4)
        matrix_area = tk.Frame(frame); matrix_area.pack(fill=tk.BOTH, expand=True)

        # Фіксований розмір клітинки (у дюймах) — читабельний незалежно від n;
        # загальний розмір фігури росте разом з кількістю показників,
        # а прокрутка дозволяє знайти потрібну клітинку.
        CELL_IN = 2.1
        dpi = 96
        fig = Figure(figsize=(n*CELL_IN, n*CELL_IN), dpi=dpi)

        for i in range(n):
            for j in range(n):
                ax = fig.add_subplot(n, n, i*n+j+1)
                if i == j:
                    a = arrays[i][~np.isnan(arrays[i])]
                    if len(a) > 0:
                        ax.hist(a, bins=max(4, int(np.sqrt(len(a)))),
                                color=hi_col, alpha=0.8, edgecolor="white", linewidth=0.4)
                    ax.set_title(labels[i], fontsize=fz+1, pad=2, fontfamily=ff)
                else:
                    xi = arrays[j]; yi = arrays[i]
                    mn = min(len(xi), len(yi))
                    xi = xi[:mn]; yi = yi[:mn]
                    mask = ~(np.isnan(xi) | np.isnan(yi))
                    xi = xi[mask]; yi = yi[mask]
                    ax.scatter(xi, yi, s=pt_size, alpha=pt_alpha,
                               color=pt_col, edgecolors="none")
                    if show_tr and len(xi) >= 3:
                        try:
                            z  = np.polyfit(xi, yi, 1)
                            xl = np.linspace(xi.min(), xi.max(), 50)
                            ax.plot(xl, np.poly1d(z)(xl),
                                    color=tr_col, lw=tr_width, alpha=0.9)
                        except Exception: pass
                ax.tick_params(labelsize=fz)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

        meth_full = "Пірсон" if method == "pearson" else "Спірмен"
        custom_sc = getattr(self, '_sc_title', '')
        fig.suptitle(
            custom_sc if custom_sc else f"Матриця діаграм розсіювання ({meth_full})",
            fontsize=11, fontfamily=ff, y=0.995)
        try: fig.tight_layout(rect=[0, 0, 1, 0.98])
        except Exception: pass
        self._sc_fig = fig

        self._embed_scatter_matrix_frozen(fig, matrix_area, labels, ff)

    def _embed_scatter_matrix_frozen(self, fig, matrix_area, labels, ff):
        """Вбудовує матрицю розсіювання із ЗАКРІПЛЕНИМИ верхнім рядком і лівим
        стовпцем назв показників (аналог 'закріпити області' в Excel) — вони
        лишаються видимими під час прокрутки, тож завжди зрозуміло, якій парі
        показників відповідає клітинка, що зараз на екрані.
        Позиції заголовків беруться з РЕАЛЬНИХ координат підграфіків matplotlib
        (після tight_layout), а не з наближеного розрахунку — це гарантує
        точне вирівнювання навіть коли tight_layout нерівномірно стискає краї."""
        n = len(labels)
        HEADER_H = 40
        HEADER_W = 130

        grid = tk.Frame(matrix_area)
        grid.pack(fill=tk.BOTH, expand=True)
        grid.grid_rowconfigure(1, weight=1)
        grid.grid_columnconfigure(1, weight=1)

        corner = tk.Frame(grid, width=HEADER_W, height=HEADER_H, bg="#dfe7f2")
        corner.grid(row=0, column=0, sticky="nsew")

        top_cv  = tk.Canvas(grid, height=HEADER_H, bg="#dfe7f2", highlightthickness=0)
        top_cv.grid(row=0, column=1, sticky="ew")
        left_cv = tk.Canvas(grid, width=HEADER_W, bg="#dfe7f2", highlightthickness=0)
        left_cv.grid(row=1, column=0, sticky="ns")

        main_outer = tk.Frame(grid)
        main_outer.grid(row=1, column=1, sticky="nsew")
        vsb = tk.Scrollbar(grid, orient="vertical", width=16,
                           bg="#b0b8c4", troughcolor="#eef1f5",
                           activebackground="#1a4b8c")
        vsb.grid(row=1, column=2, sticky="ns")
        hsb = tk.Scrollbar(grid, orient="horizontal", width=16,
                           bg="#b0b8c4", troughcolor="#eef1f5",
                           activebackground="#1a4b8c")
        hsb.grid(row=2, column=1, sticky="ew")

        main_cv = tk.Canvas(main_outer, yscrollcommand=vsb.set, xscrollcommand=hsb.set,
                            highlightthickness=0, bg="white")
        main_cv.pack(fill=tk.BOTH, expand=True)

        # Віджет фігури створюємо одразу з master=main_cv, щоб коректно
        # вбудувати його через create_window нижче.
        cv = FigureCanvasTkAgg(fig, master=main_cv)
        cv.draw()
        widget = cv.get_tk_widget()
        renderer = fig.canvas.get_renderer()
        fig_w_px, fig_h_px = fig.canvas.get_width_height()

        win_id = main_cv.create_window((0, 0), window=widget, anchor="nw")
        main_cv.update_idletasks()
        # Реальний розмір відрендереного віджета (а не лише розрахункове
        # значення matplotlib) — про всяк випадок, якщо є розбіжність
        # через масштабування дисплея (той самий клас проблеми, що й у
        # embed_figure_scrollable).
        bbox0 = main_cv.bbox(win_id)
        real_w = (bbox0[2]-bbox0[0]) if bbox0 else fig_w_px
        real_h = (bbox0[3]-bbox0[1]) if bbox0 else fig_h_px
        sx = real_w / fig_w_px if fig_w_px else 1.0
        sy = real_h / fig_h_px if fig_h_px else 1.0

        # Невеликий запас (+40px) з усіх боків, щоб прокрутка завжди мала
        # видимий і відчутний простір для руху. Запас застосовано ОДНАКОВО
        # до main_cv і до відповідного header-канваса (по тій самій осі) —
        # інакше однакова частка прокрутки (fraction) відповідала б різним
        # пікселям у різних канвасах, і заголовки «розсинхронізовувались»
        # би з контентом що далі, то більше.
        PAD = 40
        main_cv.configure(scrollregion=(0, 0, real_w + PAD, real_h + PAD))

        # X-межі кожного стовпця — з підграфіків першого ряду (i=0),
        # переведені у реальні піксельні координати віджета (масштаб sx)
        col_bounds = []
        for j in range(n):
            ax = fig.axes[j]
            bb = ax.get_window_extent(renderer=renderer)
            col_bounds.append((bb.x0*sx, bb.x1*sx))
        # Y-межі кожного ряду — з підграфіків першого стовпця (j=0),
        # переведено в координати Tkinter (відлік згори) й масштаб sy
        row_bounds = []
        for i in range(n):
            ax = fig.axes[i*n]
            bb = ax.get_window_extent(renderer=renderer)
            row_bounds.append(((fig_h_px - bb.y1)*sy, (fig_h_px - bb.y0)*sy))

        for j, lbl in enumerate(labels):
            x0, x1 = col_bounds[j]
            top_cv.create_window((x0+x1)/2, HEADER_H/2, anchor="center",
                window=tk.Label(top_cv, text=lbl, bg="#dfe7f2", fg="#1a4b8c",
                                font=(ff, 9, "bold"), wraplength=max(20, int(x1-x0))))
        top_cv.configure(scrollregion=(0, 0, real_w + PAD, HEADER_H))

        for i, lbl in enumerate(labels):
            y0, y1 = row_bounds[i]
            left_cv.create_window(HEADER_W/2, (y0+y1)/2, anchor="center",
                window=tk.Label(left_cv, text=lbl, bg="#dfe7f2", fg="#1a4b8c",
                                font=(ff, 9, "bold"), wraplength=HEADER_W-14,
                                justify="center"))
        left_cv.configure(scrollregion=(0, 0, HEADER_W, real_h + PAD))

        def _sync_xview(*args):
            main_cv.xview(*args); top_cv.xview(*args)
        def _sync_yview(*args):
            main_cv.yview(*args); left_cv.yview(*args)
        hsb.config(command=_sync_xview)
        vsb.config(command=_sync_yview)

        def _on_mw(e):
            main_cv.yview_scroll(int(-1*(e.delta/120)), "units")
            left_cv.yview_scroll(int(-1*(e.delta/120)), "units")
        def _on_shift_mw(e):
            main_cv.xview_scroll(int(-1*(e.delta/120)), "units")
            top_cv.xview_scroll(int(-1*(e.delta/120)), "units")
        for w_ in (main_cv, widget):
            w_.bind("<MouseWheel>", _on_mw)
            w_.bind("<Shift-MouseWheel>", _on_shift_mw)

    def _copy_scatter(self, win=None):
        if self._sc_fig is None:
            messagebox.showwarning("","Графік ще не побудований."); return
        ok_, msg = _copy_fig_to_clipboard(self._sc_fig)
        if ok_: messagebox.showinfo("","Матрицю розсіювання скопійовано.\nВставте у Word через Ctrl+V.")
        else:   messagebox.showwarning("",f"Помилка: {msg}")

    def _restyle_scatter(self, win, labels, arrays, method):
        dlg = ScatterSettingsDlg(win, self._sc_gs)
        win.wait_window(dlg)
        if dlg.result:
            self._sc_gs = dlg.result
            self._draw_scatter(self._sc_frame, labels, arrays, method)



