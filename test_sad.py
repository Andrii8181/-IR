# -*- coding: utf-8 -*-
"""
Автоматичні тести для S.A.D. v2.1
Запуск: pip install pytest   →   pytest tests/ -v
"""

import sys, os, math, types
import numpy as np
import pytest

# ── Ізоляція від Tkinter / matplotlib / PIL ─────────────────────
# Підміняємо їх фіктивними модулями ДО імпорту main.py,
# щоб тести запускались без дисплея (сервер, CI, headless).

def _mock_tk():
    for name in ["tkinter", "tkinter.ttk", "tkinter.messagebox",
                 "tkinter.filedialog", "tkinter.colorchooser",
                 "tkinter.scrolledtext", "tkinter.font"]:
        sys.modules[name] = types.ModuleType(name)
    tk = sys.modules["tkinter"]
    for attr in ["Toplevel","Frame","Label","Button","Entry","Text",
                 "Canvas","Scrollbar","Radiobutton","Checkbutton",
                 "OptionMenu","StringVar","BooleanVar","DoubleVar","IntVar",
                 "LEFT","RIGHT","TOP","BOTTOM","BOTH","X","Y","END",
                 "NORMAL","DISABLED","FLAT","WORD","CENTER","W","E",
                 "N","S","NW","NE","SE","SW","HORIZONTAL","VERTICAL","PhotoImage"]:
        setattr(tk, attr, object)
    tk.messagebox = types.SimpleNamespace(
        showinfo=lambda *a,**k: None, showwarning=lambda *a,**k: None,
        showerror=lambda *a,**k: None, askyesno=lambda *a,**k: False)
    sys.modules["tkinter.messagebox"] = tk.messagebox
    sys.modules["tkinter.ttk"] = types.SimpleNamespace(
        Combobox=object, Treeview=object, Notebook=object, Scrollbar=object,
        Style=lambda: types.SimpleNamespace(
            configure=lambda *a,**k: None, map=lambda *a,**k: None))
    sys.modules["tkinter.filedialog"] = types.SimpleNamespace(
        asksaveasfilename=lambda *a,**k: "", askopenfilename=lambda *a,**k: "")
    sys.modules["tkinter.scrolledtext"] = types.SimpleNamespace(ScrolledText=object)
    sys.modules["tkinter.font"] = types.SimpleNamespace(Font=object, families=lambda: [])

def _mock_mpl():
    mpl = types.ModuleType("matplotlib")
    mpl.rcParams = type("RC", (), {"update": lambda self, d: None, "__setitem__": lambda self,k,v: None, "__getitem__": lambda self,k: None, "get": lambda self,k,d=None: d})()
    sys.modules["matplotlib"] = mpl
    for sub in ["matplotlib.figure","matplotlib.backends",
                "matplotlib.backends.backend_tkagg",
                "matplotlib.patches","matplotlib.colors"]:
        sys.modules[sub] = types.ModuleType(sub)
    sys.modules["matplotlib.figure"].Figure = object
    sys.modules["matplotlib.backends.backend_tkagg"].FigureCanvasTkAgg = object
    sys.modules["matplotlib.patches"].FancyBboxPatch = object
    sys.modules["matplotlib.patches"].Patch = object

def _mock_pil():
    pil = types.ModuleType("PIL"); pil.Image = types.SimpleNamespace(
        open=lambda *a,**k: None, LANCZOS=1, RGBA="RGBA")
    sys.modules["PIL"] = pil

_mock_tk(); _mock_mpl(); _mock_pil()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main as sad


# ═══════════════════════════════════════════════════════════════
# ПОМІЧНИКИ
# ═══════════════════════════════════════════════════════════════

def make_long_1f(groups: dict, factor="A", block_key=None, blocks=None):
    rows = []
    for trt, vals in groups.items():
        for bi, v in enumerate(vals):
            r = {"value": float(v), factor: trt}
            if block_key and blocks:
                r[block_key] = blocks[bi % len(blocks)]
            rows.append(r)
    return rows

def make_long_2f(data: dict, f1="A", f2="B"):
    rows = []
    for (a, b), vals in data.items():
        for v in vals:
            rows.append({"value": float(v), f1: a, f2: b})
    return rows

def make_lbf_1f(groups: dict, factor="A"):
    return {factor: list(groups.keys())}


# ═══════════════════════════════════════════════════════════════
# 1. УТИЛІТИ
# ═══════════════════════════════════════════════════════════════

class TestFmt:
    def test_ціле(self):              assert sad.fmt(3.14159, 2) == "3.14"
    def test_нуль(self):              assert sad.fmt(0, 3) == "0.000"
    def test_nan(self):               assert sad.fmt(float("nan")) == ""
    def test_none(self):              assert sad.fmt(None) == ""
    def test_рядок_числа(self):       assert sad.fmt("3.5", 1) == "3.5"
    def test_нечисловий_рядок(self):  assert sad.fmt("abc") == ""
    def test_від_ємне(self):          assert sad.fmt(-5.0, 1) == "-5.0"
    def test_6_знаків_мале_p(self):   assert sad.fmt(0.00001, 6) == "0.000010"

class TestSigMark:
    def test_дуже_значуща(self):  assert sad.sig_mark(0.001) == "**"
    def test_значуща(self):       assert sad.sig_mark(0.03)  == "*"
    def test_незначуща(self):     assert sad.sig_mark(0.10)  == ""
    def test_none(self):          assert sad.sig_mark(None)  == ""
    def test_межа_001(self):      assert sad.sig_mark(0.01)  == "*"
    def test_межа_005(self):      assert sad.sig_mark(0.05)  == ""
    def test_нуль(self):          assert sad.sig_mark(0.0)   == "**"

class TestFirstSeen:
    def test_зберігає_порядок(self):
        assert sad.first_seen([3, 1, 2, 1, 3]) == [3, 1, 2]
    def test_рядки(self):
        assert sad.first_seen(["б","а","б","в"]) == ["б","а","в"]
    def test_порожній(self):
        assert sad.first_seen([]) == []

class TestGroupsByKeys:
    def test_один_ключ(self):
        long = [{"value":1.0,"A":"к"},{"value":2.0,"A":"к"},{"value":5.0,"A":"в"}]
        g = sad.groups_by_keys(long, ["A"])
        assert g[("к",)] == [1.0, 2.0]
        assert g[("в",)] == [5.0]

    def test_пропускає_nan(self):
        long = [{"value":float("nan"),"A":"к"},{"value":3.0,"A":"к"}]
        g = sad.groups_by_keys(long, ["A"])
        assert g[("к",)] == [3.0]

    def test_два_ключі(self):
        long = [{"value":1.0,"A":"к","B":"x"},
                {"value":2.0,"A":"к","B":"y"},
                {"value":3.0,"A":"в","B":"x"}]
        g = sad.groups_by_keys(long, ["A","B"])
        assert g[("к","x")] == [1.0]
        assert g[("в","x")] == [3.0]

    def test_порожній(self):
        g = sad.groups_by_keys([], ["A"])
        assert len(g) == 0

class TestVstats:
    def test_середнє_та_sd(self):
        long = [{"value":10.0,"A":"к"},{"value":12.0,"A":"к"},{"value":14.0,"A":"к"}]
        r = sad.vstats(long, ["A"])
        m, sd, n = r[("к",)]
        assert n == 3
        assert abs(m - 12.0) < 1e-10
        assert abs(sd - 2.0) < 1e-10

    def test_одне_спостереження_sd_нуль(self):
        long = [{"value":5.0,"A":"к"}]
        m, sd, n = sad.vstats(long, ["A"])[("к",)]
        assert n == 1 and sd == 0.0

    def test_порожній(self):
        assert sad.vstats([], ["A"]) == {}


# ═══════════════════════════════════════════════════════════════
# 2. CLD
# ═══════════════════════════════════════════════════════════════

class TestCLD:
    def test_всі_різні(self):
        means = {"К":10.0,"В1":20.0,"В2":15.0}
        sig_m = {("В1","К"):True,("В1","В2"):True,("В2","К"):True}
        labels = sad.cld(["К","В1","В2"], means, sig_m)
        assert len(set(labels.values())) == 3

    def test_всі_рівні(self):
        means = {"К":10.0,"В1":10.0,"В2":10.0}
        labels = sad.cld(["К","В1","В2"], means, {})
        assert len(set(labels.values())) == 1

    def test_одна_пара_різна(self):
        means = {"К":10.0,"В1":20.0,"В2":20.0}
        sig_m = {("В1","К"):True,("В2","К"):True}
        labels = sad.cld(["К","В1","В2"], means, sig_m)
        assert labels["В1"] == labels["В2"]
        assert labels["К"] != labels["В1"]

    def test_один_варіант(self):
        labels = sad.cld(["К"], {"К":10.0}, {})
        assert "К" in labels


# ═══════════════════════════════════════════════════════════════
# 3. ОДНОФАКТОРНИЙ ANOVA — CRD
# Еталонні дані: Доспєхов, 3 варіанти × 4 повторності
# ═══════════════════════════════════════════════════════════════

DOSPEKHOV_1F = {
    "Контроль": [32.0, 34.0, 31.0, 33.0],
    "Варіант1": [41.0, 43.0, 39.0, 42.0],
    "Варіант2": [38.0, 36.0, 37.0, 39.0],
}

class TestAnovaCRD:
    def _run(self, ss_type="III"):
        long = make_long_1f(DOSPEKHOV_1F)
        lbf  = make_lbf_1f(DOSPEKHOV_1F)
        return sad.anova_crd(long, ["A"], lbf, ss_type)

    def test_ключі_результату(self):
        r = self._run()
        for k in ("table","SS_error","df_error","MS_error","SS_total","NIR05"):
            assert k in r

    def test_F_значущий(self):
        r = self._run()
        row = next(row for row in r["table"] if "Фактор" in str(row[0]))
        assert row[4] > 4.0,   f"F={row[4]:.2f}"
        assert row[5] < 0.05,  f"p={row[5]:.4f}"

    def test_df_помилки_CRD(self):
        # df_error = N - k = 12 - 3 = 9
        assert self._run()["df_error"] == 9

    def test_SS_total_адитивна_для_type_I(self):
        r = self._run("I")
        table = r["table"]
        ss_f = next(row[1] for row in table if "Фактор" in str(row[0]))
        assert abs((ss_f + r["SS_error"]) - r["SS_total"]) < 0.01

    def test_нір_позитивний(self):
        nir = self._run()["NIR05"]
        assert len(nir) > 0
        assert all(v > 0 for v in nir.values())

    def test_залишки_збігаються_з_SS_error(self):
        r = self._run()
        sse_calc = float(np.sum(np.array(r["residuals"])**2))
        assert abs(sse_calc - r["SS_error"]) < 0.001

    def test_type_I_та_III_однакові_при_рівних_n(self):
        f1 = next(row[4] for row in self._run("I")["table"]  if "Фактор" in str(row[0]))
        f3 = next(row[4] for row in self._run("III")["table"] if "Фактор" in str(row[0]))
        assert abs(f1 - f3) < 0.01

    def test_два_варіанти_мінімум(self):
        groups = {"К":[10.,12.,11.],"В":[20.,22.,21.]}
        long = make_long_1f(groups)
        lbf  = {"A":["К","В"]}
        r = sad.anova_crd(long, ["A"], lbf)
        row = next(row for row in r["table"] if "Фактор" in str(row[0]))
        assert row[5] < 0.05

    def test_без_різниці_між_варіантами(self):
        groups = {"К":[10.,10.,10.],"В":[10.,10.,10.]}
        long = make_long_1f(groups)
        r = sad.anova_crd(long, ["A"], {"A":["К","В"]})
        row = next(row for row in r["table"] if "Фактор" in str(row[0]))
        F, p = row[4], row[5]
        if not math.isnan(F):
            assert p >= 0.05 or F < 0.001


# ═══════════════════════════════════════════════════════════════
# 4. ОДНОФАКТОРНИЙ ANOVA — RCBD
# ═══════════════════════════════════════════════════════════════

class TestAnovaRCBD:
    def _make(self):
        blocks = ["П1","П2","П3","П4"]
        rows = []
        for bi, b in enumerate(blocks):
            for trt, vals in DOSPEKHOV_1F.items():
                rows.append({"value": vals[bi], "A": trt, "BLOCK": b})
        return rows

    def test_F_значущий(self):
        r = sad.anova_rcbd(self._make(), ["A"],
                           {"A": list(DOSPEKHOV_1F.keys())}, bk="BLOCK")
        row = next(row for row in r["table"] if "Фактор" in str(row[0]))
        assert row[5] < 0.05

    def test_df_помилки_RCBD(self):
        # df_error = (k-1)(b-1) = 2×3 = 6
        r = sad.anova_rcbd(self._make(), ["A"],
                           {"A": list(DOSPEKHOV_1F.keys())}, bk="BLOCK")
        assert r["df_error"] == 6

    def test_рядок_Блоки_присутній(self):
        r = sad.anova_rcbd(self._make(), ["A"],
                           {"A": list(DOSPEKHOV_1F.keys())}, bk="BLOCK")
        assert any("Блоки" in str(row[0]) for row in r["table"])

    def test_RCBD_має_менший_MSE_ніж_CRD(self):
        # При реальній блоковій мінливості RCBD дає менший MS_error
        long_crd  = make_long_1f(DOSPEKHOV_1F)
        long_rcbd = self._make()
        lbf = {"A": list(DOSPEKHOV_1F.keys())}
        r_crd  = sad.anova_crd(long_crd,  ["A"], lbf)
        r_rcbd = sad.anova_rcbd(long_rcbd, ["A"], lbf, bk="BLOCK")
        assert r_rcbd["MS_error"] <= r_crd["MS_error"]


# ═══════════════════════════════════════════════════════════════
# 5. ДВОФАКТОРНИЙ ANOVA
# ═══════════════════════════════════════════════════════════════

TWO_WAY_DATA = {
    ("N0","С1"): [20.,22.,21.],
    ("N0","С2"): [18.,19.,20.],
    ("N1","С1"): [30.,32.,31.],
    ("N1","С2"): [28.,29.,30.],
}

class TestAnova2F:
    def _run(self):
        long = make_long_2f(TWO_WAY_DATA, "A", "B")
        lbf  = {"A":["N0","N1"],"B":["С1","С2"]}
        return sad.anova_crd(long, ["A","B"], lbf, ss_type="III")

    def test_ефект_A_значущий(self):
        row = next(r for r in self._run()["table"] if r[0]=="Фактор A")
        assert row[5] < 0.05, f"p(A)={row[5]}"

    def test_ефект_B_значущий(self):
        row = next(r for r in self._run()["table"] if r[0]=="Фактор B")
        assert row[5] < 0.05, f"p(B)={row[5]}"

    def test_взаємодія_A_B_є_в_таблиці(self):
        names = [r[0] for r in self._run()["table"]]
        assert "Фактор A×B" in names

    def test_df_залишку_2F(self):
        assert self._run()["df_error"] == 8


# ═══════════════════════════════════════════════════════════════
# 6. ЛАТИНСЬКИЙ КВАДРАТ
# ═══════════════════════════════════════════════════════════════

def make_latin_3x3():
    layout = [
        ("R1","C1","А",10),("R1","C2","Б",20),("R1","C3","В",15),
        ("R2","C1","Б",22),("R2","C2","В",17),("R2","C3","А",12),
        ("R3","C1","В",18),("R3","C2","А",11),("R3","C3","Б",21),
    ]
    return [{"ROW":r,"COL":c,"A":v,"value":float(val)} for r,c,v,val in layout]

class TestAnovaLatin:
    def test_df_помилки_k3(self):
        # df_error = (k-1)(k-2) = 2
        r = sad.anova_latin_square(make_latin_3x3(), ["A"], {"A":["А","Б","В"]})
        assert r["df_error"] == 2

    def test_рядки_та_стовпці_у_таблиці(self):
        r = sad.anova_latin_square(make_latin_3x3(), ["A"], {"A":["А","Б","В"]})
        names = [row[0] for row in r["table"]]
        assert "Рядки"   in names
        assert "Стовпці" in names

    def test_latin_k(self):
        r = sad.anova_latin_square(make_latin_3x3(), ["A"], {"A":["А","Б","В"]})
        assert r["latin_k"] == 3

    def test_неправильна_структура_кидає_ValueError(self):
        with pytest.raises(ValueError):
            sad.anova_latin_square(
                make_latin_3x3()[:6], ["A"], {"A":["А","Б","В"]})


# ═══════════════════════════════════════════════════════════════
# 7. ПАРНІ ПОРІВНЯННЯ
# ═══════════════════════════════════════════════════════════════

class TestPairwise:
    MEANS = {"К":10.0,"В1":20.0,"В2":10.5}
    NS    = {"К":4,"В1":4,"В2":4}
    MSE   = 1.0; DFE = 9

    def _run(self, method):
        return sad.pairwise_param(
            list(self.MEANS.keys()), self.MEANS, self.NS,
            self.MSE, self.DFE, method)

    def _sig_pair(self, sig, a, b):
        return sig.get((a,b), False) or sig.get((b,a), False)

    def test_duncan_В1_vs_К(self):
        _, sig = self._run("duncan")
        assert self._sig_pair(sig,"В1","К"), "Duncan: В1 vs К має бути значущим"

    def test_tukey_В1_vs_К(self):
        _, sig = self._run("tukey")
        assert self._sig_pair(sig,"В1","К"), "Tukey: В1 vs К має бути значущим"

    def test_bonferroni_В1_vs_К(self):
        _, sig = self._run("bonferroni")
        assert self._sig_pair(sig,"В1","К"), "Bonferroni: В1 vs К має бути значущим"

    def test_tukey_К_vs_В2_незначуще(self):
        _, sig = self._run("tukey")
        assert not self._sig_pair(sig,"К","В2"), "К vs В2 не має бути значущим"

    def test_рядки_3_колонки(self):
        rows, _ = self._run("duncan")
        assert all(len(r)==3 for r in rows)

    def test_один_варіант_порожній_результат(self):
        rows, sig = sad.pairwise_param(["К"],{"К":10.},{"К":4},1.,9,"tukey")
        assert rows == [] and sig == {}


# ═══════════════════════════════════════════════════════════════
# 8. BUILD_EFF_ROWS
# ═══════════════════════════════════════════════════════════════

class TestBuildEffRows:
    def test_сума_100(self):
        table = [["Фактор A",80.,2,40.,10.,0.01],
                 ["Залишок", 20.,5, 4.,None,None],
                 ["Загальна",100.,7,None,None,None]]
        rows = sad.build_eff_rows(table)
        assert abs(sum(float(r[1]) for r in rows) - 100.0) < 0.01

    def test_порожня_таблиця(self):
        assert sad.build_eff_rows([]) == []

    def test_немає_загальної(self):
        table = [["Фактор A",60.,1,60.,5.,0.03],
                 ["Залишок", 40.,4,10.,None,None],
                 ["Загальна",100.,5,None,None,None]]
        rows = sad.build_eff_rows(table)
        assert all(r[0] != "Загальна" for r in rows)


# ═══════════════════════════════════════════════════════════════
# 9. РЕГРЕСІЯ
# ═══════════════════════════════════════════════════════════════

class TestRegression:
    X = np.array([1.,2.,3.,4.,5.])
    Y = np.array([3.,5.,7.,9.,11.])   # y = 2x+1

    def _win(self):
        w = object.__new__(sad.RegressionWindow)
        w.gs = {}; w.win = None
        return w

    def _fit(self, model, x=None, y=None):
        return self._win()._fit_model(model, x if x is not None else self.X,
                                      y if y is not None else self.Y, 0.05)

    def test_лінійна_R2_одиниця(self):
        r = self._fit("Лінійна:  y = a + bx")
        assert abs(r["R2"] - 1.0) < 1e-8

    def test_лінійна_коефіцієнт_a(self):
        r = self._fit("Лінійна:  y = a + bx")
        assert abs(r["params"]["a"] - 1.0) < 1e-6

    def test_лінійна_коефіцієнт_b(self):
        r = self._fit("Лінійна:  y = a + bx")
        assert abs(r["params"]["b"] - 2.0) < 1e-6

    def test_лінійна_rmse_нуль(self):
        r = self._fit("Лінійна:  y = a + bx")
        assert abs(r["RMSE"]) < 1e-8

    def test_лінійна_p_значущий(self):
        # При ідеальних даних MSE=0 → F=inf → p_F=NaN (коректна поведінка scipy).
        # Тестуємо p на реальних даних з шумом.
        np.random.seed(0)
        x = np.arange(1., 11.)
        y = 5*x + 1 + np.random.normal(0, 0.5, 10)
        r = self._fit("Лінійна:  y = a + bx", x, y)
        assert r["p_F"] < 0.05, f"p_F={r['p_F']}"

    def test_лінійна_залишки_нуль(self):
        r = self._fit("Лінійна:  y = a + bx")
        assert float(np.max(np.abs(r["residuals"]))) < 1e-8

    def test_лінійна_рядок_рівняння(self):
        r = self._fit("Лінійна:  y = a + bx")
        assert "y =" in r["equation"]

    def test_квадратична_три_параметри(self):
        x = np.array([1.,2.,3.,4.,5.])
        y = x**2 + 2*x + 3
        r = self._fit("Квадратична:  y = a + bx + cx²", x, y)
        assert r is not None and "c" in r["params"]
        assert abs(r["R2"] - 1.0) < 1e-6

    def test_реальні_дані_R2_розумний(self):
        np.random.seed(42)
        x = np.arange(1., 11.)
        y = 3*x + 2 + np.random.normal(0, 1, 10)
        r = self._fit("Лінійна:  y = a + bx", x, y)
        assert 0.8 < r["R2"] < 1.0

    def test_n_у_результаті(self):
        r = self._fit("Лінійна:  y = a + bx")
        assert r["n"] == 5


# ═══════════════════════════════════════════════════════════════
# 10. RCBD MATRIX
# ═══════════════════════════════════════════════════════════════

class TestRcbdMatrix:
    def test_базовий(self):
        long = []
        for bi, b in enumerate(["П1","П2","П3"]):
            for vi, v in enumerate(["К","В1","В2"]):
                long.append({"value":float(bi*3+vi),"VARIANT":v,"BLOCK":b})
        mat, kept = sad.rcbd_matrix(long,["К","В1","В2"],["П1","П2","П3"])
        assert len(mat)==3 and len(mat[0])==3

    def test_неповний_блок_виключається(self):
        long = [{"value":1.,"VARIANT":"К", "BLOCK":"П1"},
                {"value":2.,"VARIANT":"В1","BLOCK":"П1"},
                {"value":3.,"VARIANT":"В1","BLOCK":"П2"}]
        mat, kept = sad.rcbd_matrix(long,["К","В1"],["П1","П2"])
        assert "П2" not in kept


# ═══════════════════════════════════════════════════════════════
# 11. ЦІЛІСНІСТЬ ТАБЛИЦІ ANOVA
# ═══════════════════════════════════════════════════════════════

class TestAnovaTableIntegrity:
    def _check(self, table, min_rows):
        assert len(table) >= min_rows
        assert all(len(row)==6 for row in table), \
            [f"{r[0]}:{len(r)}" for r in table if len(r)!=6]
        assert table[-1][0] == "Загальна"
        assert "Залишок" in str(table[-2][0])

    def test_crd_1f_структура(self):
        long = make_long_1f(DOSPEKHOV_1F)
        r = sad.anova_crd(long, ["A"], make_lbf_1f(DOSPEKHOV_1F))
        self._check(r["table"], 3)

    def test_crd_2f_структура(self):
        long = make_long_2f(TWO_WAY_DATA,"A","B")
        r = sad.anova_crd(long,["A","B"],{"A":["N0","N1"],"B":["С1","С2"]})
        self._check(r["table"], 5)

    def test_rcbd_1f_структура(self):
        rows = []
        for bi, b in enumerate(["П1","П2","П3","П4"]):
            for trt, vals in DOSPEKHOV_1F.items():
                rows.append({"value":vals[bi],"A":trt,"BLOCK":b})
        r = sad.anova_rcbd(rows,["A"],{"A":list(DOSPEKHOV_1F.keys())},bk="BLOCK")
        self._check(r["table"], 4)

    def test_MS_дорівнює_SS_ділений_на_df(self):
        long = make_long_1f(DOSPEKHOV_1F)
        r = sad.anova_crd(long, ["A"], make_lbf_1f(DOSPEKHOV_1F))
        for row in r["table"]:
            nm, ss, df, ms = row[0], row[1], row[2], row[3]
            if nm == "Загальна": continue
            if ms is None or (isinstance(ms,float) and math.isnan(ms)): continue
            if df and float(df) > 0:
                expected = float(ss)/float(df)
                assert abs(expected - float(ms)) < 0.001, \
                    f"{nm}: SS/df={expected:.4f} ≠ MS={float(ms):.4f}"
