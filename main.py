# main.py
# -*- coding: utf-8 -*-

import os, sys, math
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
import tkinter.font as tkfont
from datetime import datetime
from itertools import combinations

from scipy.stats import shapiro, kruskal, mannwhitneyu, f as f_dist, t as t_dist

# matplotlib (for plots)
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image

ALPHA = 0.05
COL_W = 14  # ширина комірок вводу

# -------------------------
# Helpers (UI / path)
# -------------------------
def _script_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except Exception:
        return os.getcwd()

def set_window_icon(win):
    # залишаємо як у тебе було (пошук icon.ico, _MEIPASS тощо)
    # якщо у тебе вже є ця функція — залиш свою
    try:
        ico = os.path.join(_script_dir(), "icon.ico")
        if os.path.exists(ico):
            win.iconbitmap(ico)
    except Exception:
        pass

def center_window(win):
    win.update_idletasks()
    w = win.winfo_width()
    h = win.winfo_height()
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    x = (sw - w) // 2
    y = (sh - h) // 2
    win.geometry(f"+{x}+{y}")

def first_seen_order(seq):
    out = []
    seen = set()
    for x in seq:
        if x is None:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def fit_font_size_to_texts(texts, family="Times New Roman", start=14, min_size=9, target_px=150):
    # підбір шрифту під кнопки (як у тебе)
    f = tkfont.Font(family=family, size=start)
    for sz in range(start, min_size - 1, -1):
        f.configure(size=sz)
        ok = True
        for t in texts:
            if f.measure(t) > target_px:
                ok = False
                break
        if ok:
            return (family, sz)
    return (family, min_size)

# -------------------------
# Table formatting for report
# -------------------------
def fmt_num(x, nd=3):
    try:
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        if math.isnan(float(x)):
            return ""
        return f"{float(x):.{nd}f}"
    except Exception:
        return ""

def significance_mark(p):
    try:
        if p is None or math.isnan(float(p)):
            return ""
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""
    except Exception:
        return ""

def normality_text(p):
    try:
        if p is None or math.isnan(float(p)):
            return "н/д"
        return "нормальний розподіл" if p > 0.05 else "розподіл відрізняється від нормального"
    except Exception:
        return "н/д"

def build_table_block(headers, rows):
    # табличний блок через табуляції (для ScrolledText з tabs=...)
    out = []
    out.append("\t".join(headers) + "\n")
    out.append("\n")
    for r in rows:
        out.append("\t".join("" if v is None else str(v) for v in r) + "\n")
    out.append("\n")
    return "".join(out)

def tabs_from_table_px(font_obj, headers, rows, padding_px=32, extra_gap_after_col=None, extra_gap_px=0):
    cols = len(headers)
    maxw = [font_obj.measure(str(h)) for h in headers]
    for r in rows:
        for j in range(cols):
            if j < len(r):
                maxw[j] = max(maxw[j], font_obj.measure(str(r[j])))

    tabs = []
    acc = 0
    for j in range(cols):
        acc += maxw[j] + padding_px
        if extra_gap_after_col is not None and j == extra_gap_after_col:
            acc += extra_gap_px
        tabs.append(acc)
    return tabs

# -------------------------
# Main app class (UI stable)
# -------------------------
class SADTk:
    def __init__(self, root):
        self.root = root
        self.root.title("S.A.D. — Статистичний аналіз даних")
        self.root.geometry("980x640")
        set_window_icon(self.root)

        self.table_win = None
        self.report_win = None
        self.plot_win = None

        # стартове меню (як у тебе)
        frm = tk.Frame(root, padx=20, pady=20)
        frm.pack(fill=tk.BOTH, expand=True)

        tk.Label(frm, text="S.A.D. — Статистичний аналіз даних", font=("Times New Roman", 20, "bold")).pack(pady=(0, 12))
        tk.Label(frm, text="Оберіть кількість факторів:", font=("Times New Roman", 14)).pack(pady=(0, 12))

        btns = tk.Frame(frm)
        btns.pack()

        for k in (1, 2, 3, 4):
            tk.Button(btns, text=f"{k}-факторний аналіз", width=22,
                      command=lambda kk=k: self.open_table(kk)).grid(row=0, column=k-1, padx=8, pady=6)

    # -------------------------
    # Dialog: indicator/units/design (Split-plot main factor)
    # -------------------------
    def ask_indicator_units_design(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Параметри звіту")
        dlg.resizable(False, False)
        set_window_icon(dlg)

        frm = tk.Frame(dlg, padx=16, pady=16)
        frm.pack(fill=tk.BOTH, expand=True)

        tk.Label(frm, text="Назва показника:").grid(row=0, column=0, sticky="w", pady=6)
        e_ind = tk.Entry(frm, width=40)
        e_ind.grid(row=0, column=1, pady=6)

        tk.Label(frm, text="Одиниці виміру:").grid(row=1, column=0, sticky="w", pady=6)
        e_units = tk.Entry(frm, width=40)
        e_units.grid(row=1, column=1, pady=6)

        tk.Label(frm, text="Дизайн експерименту:").grid(row=2, column=0, sticky="w", pady=6)
        design_var = tk.StringVar(value="CRD")
        cb = ttk.Combobox(frm, textvariable=design_var, state="readonly", width=18,
                          values=["CRD", "RCBD", "Split-plot"])
        cb.grid(row=2, column=1, sticky="w", pady=6)

        split_frame = tk.Frame(frm)
        split_frame.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        split_frame.grid_remove()

        tk.Label(split_frame, text="Головний фактор (main-plot):").grid(row=0, column=0, sticky="w", padx=(0, 10))
        main_var = tk.StringVar(value="")
        cb_main = ttk.Combobox(split_frame, textvariable=main_var, state="readonly", width=10,
                               values=self.factor_keys if hasattr(self, "factor_keys") else ["A"])
        cb_main.grid(row=0, column=1, sticky="w")

        def refresh_split_ui(*_):
            if design_var.get() == "Split-plot":
                vals = self.factor_keys if hasattr(self, "factor_keys") else ["A"]
                cb_main.configure(values=vals)
                if main_var.get() not in vals:
                    main_var.set(vals[0] if vals else "")
                split_frame.grid()
            else:
                split_frame.grid_remove()
            dlg.update_idletasks()
            dlg.geometry("")
            center_window(dlg)

        cb.bind("<<ComboboxSelected>>", refresh_split_ui)

        out = {"ok": False, "indicator": "", "units": "", "design": "CRD", "main_factor": ""}

        def on_ok():
            out["indicator"] = e_ind.get().strip()
            out["units"] = e_units.get().strip()
            out["design"] = design_var.get().strip()

            if not out["indicator"] or not out["units"]:
                messagebox.showwarning("Помилка", "Заповніть назву показника та одиниці виміру.")
                return

            if out["design"] == "Split-plot":
                mf = main_var.get().strip()
                if not mf:
                    messagebox.showwarning("Помилка", "Для Split-plot потрібно обрати головний фактор.")
                    return
                out["main_factor"] = mf

            out["ok"] = True
            dlg.destroy()

        btns = tk.Frame(frm)
        btns.grid(row=4, column=0, columnspan=2, pady=(14, 0))
        tk.Button(btns, text="OK", width=10, command=on_ok).pack(side=tk.LEFT, padx=6)
        tk.Button(btns, text="Скасувати", width=12, command=lambda: dlg.destroy()).pack(side=tk.LEFT, padx=6)

        dlg.update_idletasks()
        center_window(dlg)
        e_ind.focus_set()
        dlg.bind("<Return>", lambda e: on_ok())
        dlg.grab_set()
        self.root.wait_window(dlg)
        return out

    # -------------------------
    # Dialog: choose method based on Shapiro
    # -------------------------
    def choose_method_window(self, p_norm):
        dlg = tk.Toplevel(self.root)
        dlg.title("Вибір виду аналізу")
        dlg.resizable(False, False)
        set_window_icon(dlg)

        frm = tk.Frame(dlg, padx=16, pady=14)
        frm.pack(fill=tk.BOTH, expand=True)

        normal = (p_norm is not None) and (not math.isnan(p_norm)) and (p_norm > 0.05)

        if normal:
            msg = ("Дані експерименту відповідають принципам нормального розподілу\n"
                   "за методом Шапіра-Вілка.")
            tk.Label(frm, text=msg, justify="left").pack(anchor="w", pady=(0, 10))
            options = [
                ("НІР₀₅", "lsd"),
                ("Тест Тьюкі", "tukey"),
                ("Тест Дункана", "duncan"),
                ("Тест Бонферроні", "bonferroni"),
            ]
        else:
            msg = ("Дані експерименту не відповідають принципам нормального розподілу\n"
                   "за методом Шапіра-Вілка.\n"
                   "Виберіть один з непараметричних типів аналізу.")
            tk.Label(frm, text=msg, fg="#c62828", justify="left").pack(anchor="w", pady=(0, 10))
            options = [
                ("Краскела–Уолліса", "kw"),
                ("Манна-Уітні", "mw"),
            ]

        var = tk.StringVar(value=options[0][1])
        for text, val in options:
            tk.Radiobutton(frm, text=text, variable=var, value=val).pack(anchor="w", pady=2)

        out = {"ok": False, "method": None}

        def on_ok():
            out["ok"] = True
            out["method"] = var.get()
            dlg.destroy()

        btns = tk.Frame(frm)
        btns.pack(fill=tk.X, pady=(12, 0))
        tk.Button(btns, text="OK", width=10, command=on_ok).pack(side=tk.LEFT, padx=6)
        tk.Button(btns, text="Скасувати", width=12, command=lambda: dlg.destroy()).pack(side=tk.LEFT, padx=6)

        dlg.update_idletasks()
        center_window(dlg)
        dlg.bind("<Return>", lambda e: on_ok())
        dlg.grab_set()
        self.root.wait_window(dlg)
        return out

    # -------------------------
    # Data table window (stable)
    # -------------------------
    def open_table(self, factors_count):
        if self.table_win and tk.Toplevel.winfo_exists(self.table_win):
            self.table_win.destroy()

        self.factors_count = factors_count
        self.factor_keys = ["A", "B", "C", "D"][:factors_count]

        self.table_win = tk.Toplevel(self.root)
        self.table_win.title(f"S.A.D. — {factors_count}-факторний аналіз")
        self.table_win.geometry("1280x720")
        set_window_icon(self.table_win)

        self.repeat_count = 6
        self.factor_names = [f"Фактор {self.factor_keys[i]}" for i in range(factors_count)]
        self.column_names = self.factor_names + [f"Повт.{i+1}" for i in range(self.repeat_count)]

        ctl = tk.Frame(self.table_win, padx=6, pady=6)
        ctl.pack(fill=tk.X)

        btn_texts = [
            "Додати рядок", "Видалити рядок",
            "Додати стовпчик", "Видалити стовпчик",
            "Вставити з буфера", "Аналіз даних",
            "Розробник",
        ]
        btn_font = fit_font_size_to_texts(btn_texts, family="Times New Roman", start=14, min_size=9, target_px=150)

        bw = 16
        bh = 1
        padx = 3
        pady = 2

        tk.Button(ctl, text="Додати рядок", width=bw, height=bh, font=btn_font,
                  command=self.add_row).pack(side=tk.LEFT, padx=padx, pady=pady)
        tk.Button(ctl, text="Видалити рядок", width=bw, height=bh, font=btn_font,
                  command=self.delete_row).pack(side=tk.LEFT, padx=padx, pady=pady)
        tk.Button(ctl, text="Додати стовпчик", width=bw, height=bh, font=btn_font,
                  command=self.add_column).pack(side=tk.LEFT, padx=(10, padx), pady=pady)
        tk.Button(ctl, text="Видалити стовпчик", width=bw, height=bh, font=btn_font,
                  command=self.delete_column).pack(side=tk.LEFT, padx=padx, pady=pady)
        tk.Button(ctl, text="Вставити з буфера", width=bw + 2, height=bh, font=btn_font,
                  command=self.paste_from_focus).pack(side=tk.LEFT, padx=(10, padx), pady=pady)
        tk.Button(ctl, text="Аналіз даних", width=bw, height=bh, font=btn_font,
                  bg="#c62828", fg="white", command=self.analyze).pack(side=tk.LEFT, padx=(10, padx), pady=pady)
        tk.Button(ctl, text="Розробник", width=bw, height=bh, font=btn_font,
                  command=self.show_about).pack(side=tk.RIGHT, padx=padx, pady=pady)

        self.canvas = tk.Canvas(self.table_win)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sb = ttk.Scrollbar(self.table_win, orient="vertical", command=self.canvas.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=sb.set)

        self.inner = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.rows = 12
        self.cols = len(self.column_names)
        self.entries = []
        self.header_labels = []

        for j, name in enumerate(self.column_names):
            lbl = tk.Label(self.inner, text=name, relief=tk.RIDGE, width=COL_W, bg="#f0f0f0")
            lbl.grid(row=0, column=j, padx=2, pady=2, sticky="nsew")
            self.header_labels.append(lbl)

        for i in range(self.rows):
            row_entries = []
            for j in range(self.cols):
                e = tk.Entry(self.inner, width=COL_W)
                e.grid(row=i + 1, column=j, padx=2, pady=2)
                self.bind_cell(e)
                row_entries.append(e)
            self.entries.append(row_entries)

        self.inner.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

        self.entries[0][0].focus_set()

        self.table_win.bind("<Control-v>", self.on_paste)
        self.table_win.bind("<Control-V>", self.on_paste)

    def show_about(self):
        messagebox.showinfo(
            "Розробник",
            "S.A.D. — Статистичний аналіз даних\n"
            "Версія: 1.1\n"
            "Розробик: Чаплоуцький Андрій Миколайович\n"
            "Уманський національний університет"
        )

    # cell binds / navigation
    def bind_cell(self, e: tk.Entry):
        e.bind("<Return>", self.on_enter)
        e.bind("<Up>", self.on_arrow)
        e.bind("<Down>", self.on_arrow)
        e.bind("<Left>", self.on_arrow)
        e.bind("<Right>", self.on_arrow)
        e.bind("<Control-v>", self.on_paste)
        e.bind("<Control-V>", self.on_paste)

    def add_row(self):
        i = len(self.entries)
        row_entries = []
        for j in range(self.cols):
            e = tk.Entry(self.inner, width=COL_W)
            e.grid(row=i + 1, column=j, padx=2, pady=2)
            self.bind_cell(e)
            row_entries.append(e)
        self.entries.append(row_entries)
        self.rows += 1
        self.inner.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def delete_row(self):
        if not self.entries:
            return
        last = self.entries.pop()
        for e in last:
            e.destroy()
        self.rows -= 1
        self.inner.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def add_column(self):
        self.cols += 1
        col_idx = self.cols - 1
        name = f"Повт.{col_idx - self.factors_count + 1}"

        lbl = tk.Label(self.inner, text=name, relief=tk.RIDGE, width=COL_W, bg="#f0f0f0")
        lbl.grid(row=0, column=col_idx, padx=2, pady=2, sticky="nsew")
        self.header_labels.append(lbl)

        for i, row in enumerate(self.entries):
            e = tk.Entry(self.inner, width=COL_W)
            e.grid(row=i + 1, column=col_idx, padx=2, pady=2)
            self.bind_cell(e)
            row.append(e)

        self.inner.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def delete_column(self):
        if self.cols <= self.factors_count + 1:
            return
        if self.header_labels:
            lbl = self.header_labels.pop()
            lbl.destroy()
        for row in self.entries:
            w = row.pop()
            w.destroy()
        self.cols -= 1
        self.inner.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def find_pos(self, widget):
        for i, row in enumerate(self.entries):
            for j, cell in enumerate(row):
                if cell is widget:
                    return i, j
        return None

    def on_enter(self, event=None):
        pos = self.find_pos(event.widget)
        if not pos:
            return "break"
        i, j = pos
        ni = i + 1
        if ni >= len(self.entries):
            self.add_row()
        self.entries[ni][j].focus_set()
        self.entries[ni][j].icursor(tk.END)
        return "break"

    def on_arrow(self, event=None):
        pos = self.find_pos(event.widget)
        if not pos:
            return "break"
        i, j = pos
        if event.keysym == "Up":
            i = max(0, i - 1)
        elif event.keysym == "Down":
            i = min(len(self.entries) - 1, i + 1)
        elif event.keysym == "Left":
            j = max(0, j - 1)
        elif event.keysym == "Right":
            j = min(len(self.entries[i]) - 1, j + 1)
        self.entries[i][j].focus_set()
        self.entries[i][j].icursor(tk.END)
        return "break"

    def paste_from_focus(self):
        w = self.table_win.focus_get()
        if isinstance(w, tk.Entry):
            class E: pass
            ev = E()
            ev.widget = w
            self.on_paste(ev)

    def on_paste(self, event=None):
        widget = event.widget if event else self.table_win.focus_get()
        if not isinstance(widget, tk.Entry):
            return "break"
        try:
            data = self.table_win.clipboard_get()
        except Exception:
            return "break"

        rows_text = [r for r in data.splitlines() if r != ""]
        pos = self.find_pos(widget)
        if not pos:
            return "break"
        r0, c0 = pos

        for i_r, row_text in enumerate(rows_text):
            cols = row_text.split("\t")
            for j_c, val in enumerate(cols):
                rr = r0 + i_r
                cc = c0 + j_c
                while rr >= len(self.entries):
                    self.add_row()
                if cc >= self.cols:
                    continue
                self.entries[rr][cc].delete(0, tk.END)
                self.entries[rr][cc].insert(0, val)
        return "break"

    def used_repeat_columns(self):
        rep_start = self.factors_count
        rep_cols = list(range(rep_start, self.cols))
        used = []
        for c in rep_cols:
            any_num = False
            for r in range(len(self.entries)):
                s = self.entries[r][c].get().strip()
                if not s:
                    continue
                try:
                    float(s.replace(",", "."))
                    any_num = True
                    break
                except Exception:
                    continue
            if any_num:
                used.append(c)
        return used

    def collect_long(self):
        long = []
        rep_cols = self.used_repeat_columns()
        if not rep_cols:
            return long, rep_cols

        rep_start = self.factors_count
        rep_labels = {}
        for idx, c in enumerate(range(rep_start, self.cols)):
            rep_labels[c] = f"Повт.{idx+1}"

        for i, row in enumerate(self.entries):
            levels = []
            for k in range(self.factors_count):
                v = row[k].get().strip()
                if v == "":
                    v = f"рядок{i+1}"
                levels.append(v)

            for c in rep_cols:
                s = row[c].get().strip()
                if not s:
                    continue
                try:
                    val = float(s.replace(",", "."))
                except Exception:
                    continue

                rec = {"value": val, "BLOCK": rep_labels.get(c, f"Повт.?")}
                if self.factors_count >= 1: rec["A"] = levels[0]
                if self.factors_count >= 2: rec["B"] = levels[1]
                if self.factors_count >= 3: rec["C"] = levels[2]
                if self.factors_count >= 4: rec["D"] = levels[3]
                long.append(rec)

        return long, rep_cols

    # -------------------------
    # Report window (stable)
    # -------------------------
    def show_report_segments(self, segments, plot_png_path=None, plot_meta=None):
        if self.report_win and tk.Toplevel.winfo_exists(self.report_win):
            self.report_win.destroy()

        self.report_win = tk.Toplevel(self.root)
        self.report_win.title("Звіт")
        self.report_win.geometry("1180x760")
        set_window_icon(self.report_win)

        top = tk.Frame(self.report_win, padx=8, pady=8)
        top.pack(fill=tk.X)

        xsb = ttk.Scrollbar(self.report_win, orient="horizontal")
        xsb.pack(side=tk.BOTTOM, fill=tk.X)

        txt = ScrolledText(self.report_win, width=120, height=40, wrap="none", xscrollcommand=xsb.set)
        txt.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        xsb.config(command=txt.xview)

        txt.configure(font=("Times New Roman", 14))
        font_obj = tkfont.Font(font=("Times New Roman", 14))

        table_idx = 0
        for kind, payload in segments:
            if kind == "text":
                txt.insert("end", payload)
                continue

            if isinstance(payload, dict):
                headers = payload.get("headers", [])
                rows = payload.get("rows", [])
                padding_px = int(payload.get("padding_px", 32))
                extra_after = payload.get("extra_gap_after_col", None)
                extra_px = int(payload.get("extra_gap_px", 0))
            else:
                headers, rows = payload
                padding_px = 32
                extra_after = None
                extra_px = 0

            tabs = tabs_from_table_px(
                font_obj, headers, rows,
                padding_px=padding_px,
                extra_gap_after_col=extra_after,
                extra_gap_px=extra_px
            )

            tag = f"tbl_{table_idx}"
            table_idx += 1
            txt.tag_configure(tag, tabs=tabs)

            start = txt.index("end")
            txt.insert("end", build_table_block(headers, rows))
            end = txt.index("end")
            txt.tag_add(tag, start, end)

        def copy_report():
            self.report_win.clipboard_clear()
            self.report_win.clipboard_append(txt.get("1.0", "end-1c"))
            messagebox.showinfo("Готово", "Звіт скопійовано в буфер обміну.")

        tk.Button(top, text="Копіювати звіт", command=copy_report).pack(side=tk.LEFT, padx=4)

        # графічне вікно відкриваємо з PART 2 (там змінені кнопки)
        if plot_png_path:
            self.show_plot_window(plot_png_path, plot_meta or {})

        def on_ctrl_c(event=None):
            try:
                sel = txt.get("sel.first", "sel.last")
            except Exception:
                sel = txt.get("1.0", "end-1c")
            self.report_win.clipboard_clear()
            self.report_win.clipboard_append(sel)
            return "break"

        txt.bind("<Control-c>", on_ctrl_c)
        txt.bind("<Control-C>", on_ctrl_c)

    # analyze() реалізований у PART 2
    def analyze(self):
        raise NotImplementedError("analyze() має бути визначений у PART 2")


# ===== PART 2: EDITABLE ANALYSIS LAYER =====
# (Далі вставляється ЧАСТИНА 2)
# ===== PART 2: EDITABLE ANALYSIS LAYER =====
# Повна (НЕ спрощена) реалізація CRD / RCBD / Split-plot через OLS (GLM) з Type III SS
# + коректні помилки для Split-plot (Error A = Block×Main; Error B = Residual)
# + R² після CV, таблиця НІР по факторах + загальна, розширені непараметричні звіти
# + графік: без кнопки копіювати, наукова легенда, помірна сітка

import numpy as np
import math
from itertools import combinations, product
from scipy.stats import shapiro, kruskal, mannwhitneyu, f as f_dist, t as t_dist

# -------------------------
# Numeric formatting
# -------------------------
def fmt_num(x, nd=3):
    try:
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        if math.isnan(float(x)):
            return ""
        return f"{float(x):.{nd}f}"
    except Exception:
        return ""

def significance_mark(p):
    try:
        if p is None or math.isnan(float(p)):
            return ""
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""
    except Exception:
        return ""

def normality_text(p):
    try:
        if p is None or math.isnan(float(p)):
            return "н/д"
        return "нормальний розподіл" if p > 0.05 else "розподіл відрізняється від нормального"
    except Exception:
        return "н/д"

def r2_from_ss(SS_total, SS_error):
    try:
        SS_total = float(SS_total)
        SS_error = float(SS_error)
        if SS_total <= 0:
            return np.nan
        return float(1.0 - SS_error / SS_total)
    except Exception:
        return np.nan

def cv_percent_from_values(values):
    v = np.array(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return np.nan
    m = float(np.mean(v))
    if abs(m) < 1e-12:
        return np.nan
    return float(np.std(v, ddof=1) / m * 100.0)

def cv_from_ms(ms_error, grand_mean):
    try:
        ms_error = float(ms_error)
        grand_mean = float(grand_mean)
        if ms_error <= 0 or abs(grand_mean) < 1e-12:
            return np.nan
        return float((math.sqrt(ms_error) / grand_mean) * 100.0)
    except Exception:
        return np.nan

# -------------------------
# Group utilities
# -------------------------
def first_seen_order(seq):
    out = []
    seen = set()
    for x in seq:
        if x is None:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def groups_by_keys(long, keys_tuple):
    g = {}
    for r in long:
        k = tuple(r.get(x) for x in keys_tuple)
        g.setdefault(k, []).append(r["value"])
    return g

def describe_array(arr):
    a = np.array(arr, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return dict(n=0, mean=np.nan, sd=np.nan, med=np.nan, q1=np.nan, q3=np.nan, mn=np.nan, mx=np.nan)
    return dict(
        n=int(len(a)),
        mean=float(np.mean(a)),
        sd=float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        med=float(np.median(a)),
        q1=float(np.quantile(a, 0.25)),
        q3=float(np.quantile(a, 0.75)),
        mn=float(np.min(a)),
        mx=float(np.max(a)),
    )

# -------------------------
# Robust homogeneity (Brown–Forsythe)
# -------------------------
def brown_forsythe_from_groups(groups_dict):
    try:
        zs = []
        for _, arr in groups_dict.items():
            a = np.array(arr, dtype=float)
            a = a[~np.isnan(a)]
            if len(a) == 0:
                continue
            med = np.median(a)
            z = np.abs(a - med)
            zs.append(z)
        if len(zs) < 2:
            return (np.nan, np.nan)

        allz = np.concatenate(zs)
        grand = np.mean(allz)
        k = len(zs)
        n = len(allz)

        ssb = sum(len(z) * ((np.mean(z) - grand) ** 2) for z in zs)
        ssw = sum(np.sum((z - np.mean(z)) ** 2) for z in zs)

        dfb = k - 1
        dfw = n - k
        msb = ssb / dfb if dfb > 0 else np.nan
        msw = ssw / dfw if dfw > 0 else np.nan
        F = msb / msw if msw and msw > 0 else np.nan
        p = 1 - f_dist.cdf(F, dfb, dfw) if not math.isnan(F) else np.nan
        return (float(F), float(p))
    except Exception:
        return (np.nan, np.nan)

# -------------------------
# Effect coding for categorical factors (sum-to-zero)
# -------------------------
def effect_code(levels, value):
    """
    Sum-to-zero coding for a categorical factor with L levels:
    returns length L-1 vector.
    For level i (0..L-2): e_i = 1 at i, 0 else
    For last level: all -1.
    """
    L = len(levels)
    if L <= 1:
        return np.zeros((0,), dtype=float)
    idx = levels.index(value)
    v = np.zeros((L - 1,), dtype=float)
    if idx < L - 1:
        v[idx] = 1.0
    else:
        v[:] = -1.0
    return v

def interaction_cols(cols_a, cols_b):
    if cols_a.shape[1] == 0 or cols_b.shape[1] == 0:
        return np.zeros((cols_a.shape[0], 0), dtype=float)
    out = []
    for i in range(cols_a.shape[1]):
        for j in range(cols_b.shape[1]):
            out.append((cols_a[:, i] * cols_b[:, j])[:, None])
    return np.hstack(out) if out else np.zeros((cols_a.shape[0], 0), dtype=float)

# -------------------------
# Build design matrix for Type III ANOVA
# Terms are encoded with sum-to-zero; intercept included.
# -------------------------
def build_term_matrices(long, factor_levels, include_block=False):
    """
    Returns:
      y (n,)
      base columns dict for each factor name -> matrix (n, L-1)
      block columns matrix if include_block
    """
    y = np.array([r["value"] for r in long], dtype=float)
    n = len(long)

    mats = {}
    for f, levels in factor_levels.items():
        cols = np.zeros((n, max(0, len(levels) - 1)), dtype=float)
        if cols.shape[1] == 0:
            mats[f] = cols
            continue
        for i, rec in enumerate(long):
            cols[i, :] = effect_code(levels, rec.get(f))
        mats[f] = cols

    block_mat = None
    if include_block:
        blevels = factor_levels["BLOCK"]
        block_mat = np.zeros((n, max(0, len(blevels) - 1)), dtype=float)
        for i, rec in enumerate(long):
            block_mat[i, :] = effect_code(blevels, rec.get("BLOCK"))
    return y, mats, block_mat

def build_X_from_terms(intercept=True, term_mats=None, term_order=None):
    """
    term_mats: dict term_name -> matrix
    term_order: list of term names in desired concatenation order
    returns X and slices dict term_name -> (c0,c1)
    """
    if term_mats is None:
        term_mats = {}
    if term_order is None:
        term_order = list(term_mats.keys())

    cols = []
    slices = {}
    c = 0
    if intercept:
        cols.append(np.ones((term_mats[term_order[0]].shape[0], 1), dtype=float) if term_order else np.ones((0, 1)))
        slices["Intercept"] = (0, 1)
        c = 1

    for t in term_order:
        M = term_mats[t]
        if M is None or M.shape[1] == 0:
            slices[t] = (c, c)
            continue
        cols.append(M)
        slices[t] = (c, c + M.shape[1])
        c += M.shape[1]

    X = np.hstack(cols) if cols else np.zeros((0, 0), dtype=float)
    return X, slices

# -------------------------
# OLS core and Type III SS
# -------------------------
def ols_fit_sse(y, X):
    """
    Returns: SSE, df_resid, rank
    Uses least squares with robust rank handling.
    """
    y = y.reshape(-1, 1)
    if X.size == 0:
        # model with no predictors
        resid = y - np.mean(y)
        sse = float(np.sum(resid ** 2))
        df_resid = max(1, y.shape[0] - 1)
        return sse, df_resid, 1

    beta, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    if residuals.size > 0:
        sse = float(residuals[0])
    else:
        # compute residuals explicitly
        resid = y - X @ beta
        sse = float(np.sum(resid ** 2))
    df_resid = int(y.shape[0] - rank)
    return sse, df_resid, int(rank)

def type3_anova(y, X_full, term_slices, term_testable_names):
    """
    Type III SS for each term:
      SS_term = SSE_reduced - SSE_full (dropping term columns)
    Returns dict term -> (SS, df_term)
    """
    sse_full, df_resid, rank_full = ols_fit_sse(y, X_full)
    out = {}
    for t in term_testable_names:
        c0, c1 = term_slices[t]
        if c1 <= c0:
            out[t] = (0.0, 0)
            continue
        keep = np.ones((X_full.shape[1],), dtype=bool)
        keep[c0:c1] = False
        X_red = X_full[:, keep]
        sse_red, _, _ = ols_fit_sse(y, X_red)
        ss = float(max(0.0, sse_red - sse_full))
        df = int(c1 - c0)
        out[t] = (ss, df)
    return out, sse_full, df_resid

# -------------------------
# Build full term list for CRD / RCBD / Split-plot
# -------------------------
def build_terms_crd(factor_keys):
    # all main effects + all interactions up to full order
    terms = []
    for r in range(1, len(factor_keys) + 1):
        for S in combinations(factor_keys, r):
            terms.append(("×".join(S), list(S)))
    return terms

def build_terms_rcbd(factor_keys):
    # block + all treatment terms (same as CRD)
    return build_terms_crd(factor_keys)

def build_terms_splitplot(factor_keys, main_factor):
    """
    Split-plot terms:
      Block
      Main (whole-plot)
      Block×Main (Error A term - included as model term)
      Subplot terms: any term that involves any subplot factor (i.e., factors other than main)
      including interactions among subplot factors and interactions with main
    """
    sub_factors = [f for f in factor_keys if f != main_factor]
    treatment_terms = build_terms_crd(factor_keys)

    # mark which terms are subplot-tested (Error B) vs whole-plot-tested (Error A)
    # whole-plot terms: only {main_factor}
    # everything else involving any sub factor => Error B
    term_groups = []
    for name, factors in treatment_terms:
        if factors == [main_factor]:
            term_groups.append((name, factors, "A"))  # tested vs Error A
        elif any(f in sub_factors for f in factors):
            term_groups.append((name, factors, "B"))  # tested vs Error B
        else:
            # should not happen
            term_groups.append((name, factors, "B"))

    return term_groups, sub_factors

# -------------------------
# Construct matrices for a list of term factor sets
# -------------------------
def make_term_matrix(mats, factor_list):
    if not factor_list:
        return np.zeros((next(iter(mats.values())).shape[0], 0), dtype=float)
    M = mats[factor_list[0]]
    for f in factor_list[1:]:
        M = interaction_cols(M, mats[f])
    return M

# -------------------------
# ANOVA builders: CRD / RCBD / Split-plot (FULL)
# -------------------------
def anova_crd_full(long, factor_keys, levels_by_factor):
    # Build levels dict
    factor_levels = {f: levels_by_factor[f] for f in factor_keys}
    y, mats, _ = build_term_matrices(long, factor_levels, include_block=False)

    # Build term matrices
    term_defs = build_terms_crd(factor_keys)
    term_mats = {}
    term_order = []
    for name, flist in term_defs:
        term_mats[name] = make_term_matrix(mats, flist)
        term_order.append(name)

    X, slices = build_X_from_terms(intercept=True, term_mats=term_mats, term_order=term_order)
    ss_terms, sse, df_e = type3_anova(y, X, slices, term_order)

    SS_total = float(np.sum((y - np.mean(y)) ** 2))
    SS_error = float(sse)
    MS_error = SS_error / df_e if df_e > 0 else np.nan

    rows = []
    # add each term
    for name in term_order:
        SS, df = ss_terms[name]
        MS = SS / df if df > 0 else np.nan
        F = MS / MS_error if MS_error > 0 and df > 0 else np.nan
        p = 1 - f_dist.cdf(F, df, df_e) if not math.isnan(F) else np.nan
        rows.append((name, SS, df, MS, F, p, "E"))
    rows.append(("Залишок", SS_error, df_e, MS_error, None, None, None))
    rows.append(("Загальна", SS_total, int(len(y) - 1), None, None, None, None))

    return dict(
        table=rows,
        SS_total=SS_total,
        SS_error=SS_error,
        MS_error=MS_error,
        df_error=int(df_e),
        grand_mean=float(np.mean(y)),
        ss_by_term=ss_terms
    )

def anova_rcbd_full(long, factor_keys, levels_by_factor):
    # include BLOCK
    block_levels = levels_by_factor["BLOCK"]
    factor_levels = {"BLOCK": block_levels}
    factor_levels.update({f: levels_by_factor[f] for f in factor_keys})

    y, mats, block_mat = build_term_matrices(long, factor_levels, include_block=True)
    mats["BLOCK"] = block_mat

    term_defs = build_terms_rcbd(factor_keys)

    # Order: BLOCK first, then treatment terms (Type III still ok)
    term_mats = {"Блоки": mats["BLOCK"]}
    term_order = ["Блоки"]

    for name, flist in term_defs:
        term_mats[name] = make_term_matrix(mats, flist)
        term_order.append(name)

    X, slices = build_X_from_terms(intercept=True, term_mats=term_mats, term_order=term_order)
    ss_terms, sse, df_e = type3_anova(y, X, slices, term_order)

    SS_total = float(np.sum((y - np.mean(y)) ** 2))
    SS_error = float(sse)
    MS_error = SS_error / df_e if df_e > 0 else np.nan

    rows = []
    # block row
    SSb, dfb = ss_terms["Блоки"]
    MSb = SSb / dfb if dfb > 0 else np.nan
    Fb = MSb / MS_error if MS_error > 0 and dfb > 0 else np.nan
    pb = 1 - f_dist.cdf(Fb, dfb, df_e) if not math.isnan(Fb) else np.nan
    rows.append(("Блоки", SSb, dfb, MSb, Fb, pb, "E"))

    # treatment terms
    for name in term_order[1:]:
        SS, df = ss_terms[name]
        MS = SS / df if df > 0 else np.nan
        F = MS / MS_error if MS_error > 0 and df > 0 else np.nan
        p = 1 - f_dist.cdf(F, df, df_e) if not math.isnan(F) else np.nan
        rows.append((name, SS, df, MS, F, p, "E"))

    rows.append(("Залишок", SS_error, df_e, MS_error, None, None, None))
    rows.append(("Загальна", SS_total, int(len(y) - 1), None, None, None, None))

    return dict(
        table=rows,
        SS_total=SS_total,
        SS_error=SS_error,
        MS_error=MS_error,
        df_error=int(df_e),
        grand_mean=float(np.mean(y)),
        ss_by_term=ss_terms
    )

def anova_splitplot_full(long, factor_keys, levels_by_factor, main_factor):
    # Include BLOCK and BLOCK×MAIN, all treatment terms
    block_levels = levels_by_factor["BLOCK"]
    factor_levels = {"BLOCK": block_levels}
    factor_levels.update({f: levels_by_factor[f] for f in factor_keys})

    y, mats, block_mat = build_term_matrices(long, factor_levels, include_block=True)
    mats["BLOCK"] = block_mat

    # build BLOCK×MAIN
    block_main = interaction_cols(mats["BLOCK"], mats[main_factor])

    # build treatment term groups with A/B error mapping
    term_groups, sub_factors = build_terms_splitplot(factor_keys, main_factor)

    # term mats in order:
    # Blocks, Main, Block×Main, then other treatment terms (including interactions)
    term_mats = {}
    term_order = []

    term_mats["Блоки"] = mats["BLOCK"]
    term_order.append("Блоки")

    term_mats[f"Фактор {main_factor}"] = mats[main_factor]
    term_order.append(f"Фактор {main_factor}")

    term_mats[f"Блоки×{main_factor} (Error A)"] = block_main
    term_order.append(f"Блоки×{main_factor} (Error A)")

    # Treatment terms (name with factors)
    term_error_map = {}  # term_name -> "A"/"B"
    for name, flist, which in term_groups:
        # skip main single, already added as "Фактор main"
        if flist == [main_factor]:
            continue
        tname = f"Терм {name}"
        term_mats[tname] = make_term_matrix(mats, flist)
        term_order.append(tname)
        term_error_map[tname] = which  # "B" for anything with subplot factors

    # Build full model matrix
    X_full, slices = build_X_from_terms(intercept=True, term_mats=term_mats, term_order=term_order)

    # Type III SS for all terms (including Block×Main)
    ss_terms, sse_full, df_eB = type3_anova(y, X_full, slices, term_order)

    # Error B:
    SS_error_B = float(sse_full)
    df_error_B = int(df_eB)
    MS_error_B = SS_error_B / df_error_B if df_error_B > 0 else np.nan

    # Error A comes from Block×Main term as a stratum:
    SS_error_A, df_error_A = ss_terms.get(f"Блоки×{main_factor} (Error A)", (np.nan, 0))
    SS_error_A = float(SS_error_A)
    df_error_A = int(df_error_A)
    MS_error_A = SS_error_A / df_error_A if df_error_A > 0 else np.nan

    SS_total = float(np.sum((y - np.mean(y)) ** 2))

    rows = []

    # Blocks (tested vs Error A)
    SSb, dfb = ss_terms["Блоки"]
    MSb = SSb / dfb if dfb > 0 else np.nan
    Fb = MSb / MS_error_A if MS_error_A > 0 and dfb > 0 else np.nan
    pb = 1 - f_dist.cdf(Fb, dfb, df_error_A) if not math.isnan(Fb) else np.nan
    rows.append(("Блоки", SSb, dfb, MSb, Fb, pb, "A"))

    # Main factor (tested vs Error A)
    SSm, dfm = ss_terms[f"Фактор {main_factor}"]
    MSm = SSm / dfm if dfm > 0 else np.nan
    Fm = MSm / MS_error_A if MS_error_A > 0 and dfm > 0 else np.nan
    pm = 1 - f_dist.cdf(Fm, dfm, df_error_A) if not math.isnan(Fm) else np.nan
    rows.append((f"Фактор {main_factor}", SSm, dfm, MSm, Fm, pm, "A"))

    # Error A row
    rows.append((f"Блоки×{main_factor} (Error A)", SS_error_A, df_error_A, MS_error_A, None, None, None))

    # Subplot terms (and interactions incl with main) tested vs Error B
    for tname in term_order:
        if tname in ("Блоки", f"Фактор {main_factor}", f"Блоки×{main_factor} (Error A)"):
            continue
        SS, df = ss_terms[tname]
        MS = SS / df if df > 0 else np.nan
        F = MS / MS_error_B if MS_error_B > 0 and df > 0 else np.nan
        p = 1 - f_dist.cdf(F, df, df_error_B) if not math.isnan(F) else np.nan
        rows.append((tname.replace("Терм ", ""), SS, df, MS, F, p, "B"))

    # Error B row
    rows.append(("Залишок (Error B)", SS_error_B, df_error_B, MS_error_B, None, None, None))
    rows.append(("Загальна", SS_total, int(len(y) - 1), None, None, None, None))

    return dict(
        table=rows,
        SS_total=SS_total,
        SS_error_sub=SS_error_B,
        SS_error_main=SS_error_A,
        MS_error_sub=MS_error_B,
        MS_error_main=MS_error_A,
        df_error_sub=df_error_B,
        df_error_main=df_error_A,
        grand_mean=float(np.mean(y)),
        main_factor=main_factor,
        ss_by_term=ss_terms
    )

# -------------------------
# Effect sizes: percent SS and partial eta^2
# -------------------------
def effect_size_tables(anova_rows, SS_total, SS_error_for_partial, df_error_for_partial):
    # Returns:
    # 1) %SS table (excluding total)
    # 2) partial eta^2 table for tested terms (exclude error/total)
    pct_rows = []
    eta_rows = []

    for name, SS, df, MS, F, p, strata in anova_rows:
        if name.startswith("Залишок") or name.startswith("Загальна") or "Error" in name and ("Блоки×" in name):
            continue
        if SS is None or math.isnan(float(SS)):
            continue
        pct_rows.append([name, fmt_num(100.0 * float(SS) / float(SS_total), 2)])

    # partial eta^2 = SS_term / (SS_term + SS_error_stratum)
    # Here caller should pass correct SS_error for that term's stratum.
    # For split-plot we compute per term in the report (A uses ErrorA, B uses ErrorB).
    return pct_rows, eta_rows

def partial_eta2(SS_term, SS_error):
    try:
        SS_term = float(SS_term)
        SS_error = float(SS_error)
        denom = SS_term + SS_error
        if denom <= 0:
            return np.nan
        return float(SS_term / denom)
    except Exception:
        return np.nan

# -------------------------
# LSD helpers
# -------------------------
def t_crit(alpha, df):
    try:
        return float(t_dist.ppf(1 - alpha / 2, df))
    except Exception:
        return np.nan

def harmonic_mean(ns):
    ns = [int(x) for x in ns if x is not None and int(x) > 0]
    if not ns:
        return np.nan
    return float(len(ns) / sum(1.0 / x for x in ns))

def lsd_value(alpha, df, ms_error, n_eff):
    try:
        if n_eff <= 0:
            return np.nan
        tcrit = t_crit(alpha, df)
        return float(tcrit * math.sqrt(2.0 * float(ms_error) / float(n_eff)))
    except Exception:
        return np.nan

# -------------------------
# Nonparametric pairwise + effects
# -------------------------
def cliffs_delta(a, b):
    a = np.array(a, dtype=float); a = a[~np.isnan(a)]
    b = np.array(b, dtype=float); b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    gt = 0
    lt = 0
    for x in a:
        gt += np.sum(x > b)
        lt += np.sum(x < b)
    denom = len(a) * len(b)
    return float((gt - lt) / denom)

def delta_label(d):
    try:
        if d is None or math.isnan(float(d)):
            return ""
        ad = abs(float(d))
        if ad < 0.147:
            return "незначний"
        if ad < 0.33:
            return "малий"
        if ad < 0.474:
            return "середній"
        return "великий"
    except Exception:
        return ""

def pairwise_mw_bonf_with_effect(names, groups1, alpha=0.05):
    rows = []
    sig = {}
    m = len(names)
    if m < 2:
        return rows, sig
    mtests = m * (m - 1) // 2
    for i in range(m):
        for j in range(i + 1, m):
            a = groups1[names[i]]
            b = groups1[names[j]]
            try:
                U, p = mannwhitneyu(a, b, alternative="two-sided")
                p_b = min(1.0, float(p) * mtests)
                is_sig = (p_b < alpha)
                sig[(names[i], names[j])] = is_sig
                d = cliffs_delta(a, b)
                rows.append([
                    f"{names[i]}  vs  {names[j]}",
                    fmt_num(U, 3),
                    fmt_num(p, 4),
                    fmt_num(p_b, 4),
                    ("*" if is_sig else "-"),
                    fmt_num(d, 3),
                    delta_label(d)
                ])
            except Exception:
                sig[(names[i], names[j])] = False
                rows.append([f"{names[i]}  vs  {names[j]}", "", "", "", "-", "", ""])
    return rows, sig

# -------------------------
# CLD letters (deterministic) for factor-level letters (NOT for splitplot variants)
# -------------------------
def cld_multi_letters(names, center_values, sig_matrix):
    order = sorted(names, key=lambda n: (-(center_values.get(n, -1e18) if center_values.get(n) is not None else -1e18), str(n)))
    letters = {}
    alphabet = list("abcdefghijklmnopqrstuvwxyz")
    groups_for_letter = []  # list of sets

    for g in order:
        placed = False
        for li, s in enumerate(groups_for_letter):
            ok = True
            for h in s:
                if sig_matrix.get((g, h), False) or sig_matrix.get((h, g), False):
                    ok = False
                    break
            if ok:
                s.add(g)
                letters[g] = letters.get(g, "") + alphabet[li]
                placed = True
        if not placed:
            groups_for_letter.append(set([g]))
            li = len(groups_for_letter) - 1
            letters[g] = letters.get(g, "") + alphabet[li]
    return letters

def lsd_sig_matrix(names, means, ns, ms_error, df_error, alpha=0.05):
    sig = {}
    tcrit = t_crit(alpha, df_error)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ni = ns.get(names[i], 0)
            nj = ns.get(names[j], 0)
            if ni <= 0 or nj <= 0:
                sig[(names[i], names[j])] = False
                continue
            n_eff = 2.0 / (1.0 / ni + 1.0 / nj)
            lsd = tcrit * math.sqrt(2.0 * ms_error / n_eff)
            sig[(names[i], names[j])] = (abs(means[names[i]] - means[names[j]]) > lsd)
    return sig

# -------------------------
# Plot: factor-block boxplot (as you required)
# -------------------------
from matplotlib.figure import Figure
from PIL import Image
from matplotlib.lines import Line2D

def build_factor_block_boxplot_png(long, factor_keys, levels_by_factor, letters_factor, indicator, units):
    x_positions = []
    data_list = []
    labels = []
    factor_block_ranges = []

    pos = 1
    gap = 1.2
    for f in factor_keys:
        lvls = levels_by_factor[f]
        start = pos
        for lv in lvls:
            arr = [r["value"] for r in long
                   if r.get(f) == lv and r.get("value") is not None and not math.isnan(r.get("value", np.nan))]
            if len(arr) == 0:
                arr = [np.nan]
            x_positions.append(pos)
            data_list.append(arr)
            labels.append(str(lv))
            pos += 1
        end = pos - 1
        factor_block_ranges.append((start, end, f))
        pos += gap

    fig = Figure(figsize=(12.5, 3.8), dpi=130)
    ax = fig.add_subplot(111)

    bp = ax.boxplot(
        data_list,
        positions=x_positions,
        widths=0.6,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="^", markersize=6),
        flierprops=dict(marker="o", markersize=3),
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=0, fontsize=9)
    ax.set_ylabel(f"{indicator}, {units}")

    # помірна сітка
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    ax.set_title("")

    # місце під підписи факторів
    y_min, y_max = ax.get_ylim()
    y_text = y_min - (y_max - y_min) * 0.14
    ax.set_ylim(y_text, y_max)

    # підписи блоків факторів
    for (start, end, f) in factor_block_ranges:
        mid = (start + end) / 2.0
        ax.text(mid, y_min - (y_max - y_min) * 0.09, f"Фактор {f}", ha="center", va="top", fontsize=10)
        ax.axvline(end + 0.5, linewidth=0.6, alpha=0.6)

    # літери над верхнім "вусом"
    whiskers = bp["whiskers"]
    fl_list = []
    for f in factor_keys:
        for lv in levels_by_factor[f]:
            fl_list.append((f, lv))

    for i, x in enumerate(x_positions):
        if i >= len(fl_list):
            continue
        f, lv = fl_list[i]
        letter = letters_factor.get(f, {}).get(lv, "")
        if not letter:
            continue
        try:
            wy = max(whiskers[2 * i + 1].get_ydata())
        except Exception:
            wy = np.nanmax(np.array(data_list[i], dtype=float))
        if wy is None or (isinstance(wy, float) and math.isnan(wy)):
            continue
        ax.text(x, wy + (y_max - y_min) * 0.03, letter, ha="center", va="bottom", fontsize=10)

    legend_items = [
        Line2D([0], [0], marker="^", linestyle="None", markersize=7, label="Середнє"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=5, label="Викиди"),
    ]
    ax.legend(handles=legend_items, loc="upper right", frameon=True)

    out_path = os.path.join(_script_dir(), "SAD_boxplot_factors.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return out_path

def show_plot_window(self, plot_png_path: str, factor_plot_meta: dict):
    if self.plot_win and tk.Toplevel.winfo_exists(self.plot_win):
        self.plot_win.destroy()

    self.plot_win = tk.Toplevel(self.root)
    self.plot_win.title("Графічний звіт")
    self.plot_win.geometry("1180x520")
    set_window_icon(self.plot_win)

    top = tk.Frame(self.plot_win, padx=8, pady=8)
    top.pack(fill=tk.X)

    fig = Figure(figsize=(10.8, 3.4), dpi=100)
    ax = fig.add_subplot(111)
    ax.axis("off")
    img = Image.open(plot_png_path)
    ax.imshow(img)

    canvas = FigureCanvasTkAgg(fig, master=self.plot_win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

    def save_png():
        fn = filedialog.asksaveasfilename(
            parent=self.plot_win,
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
            title="Зберегти графік"
        )
        if not fn:
            return
        try:
            img2 = Image.open(plot_png_path)
            img2.save(fn, "PNG")
            messagebox.showinfo("Готово", "Графік збережено.")
        except Exception as ex:
            messagebox.showerror("Помилка", str(ex))

    # Кнопка "копіювати графік" — ВИДАЛЕНО
    tk.Button(top, text="Зберегти графік (PNG)", command=save_png).pack(side=tk.LEFT, padx=4)

# -------------------------
# Main analyze() — FULL
# -------------------------
def analyze(self):
    created_at = datetime.now()

    params = self.ask_indicator_units_design()
    if not params["ok"]:
        return

    indicator = params["indicator"]
    units = params["units"]
    design = params["design"]
    main_factor = params.get("main_factor", "")

    long, used_rep_cols = self.collect_long()
    if len(long) == 0:
        messagebox.showwarning("Помилка", "Немає числових даних для аналізу.\nПеревірте повторності та значення.")
        return

    # levels
    levels_by_factor = {f: first_seen_order([r.get(f) for r in long]) for f in self.factor_keys}
    levels_by_factor["BLOCK"] = first_seen_order([r.get("BLOCK") for r in long])

    values = np.array([r["value"] for r in long], dtype=float)
    if len(values) < 3:
        messagebox.showinfo("Результат", "Надто мало даних для аналізу.")
        return

    # residuals for Shapiro: fit saturated cell means by OLS on all factor interactions (CRD-style),
    # using effect coding. This is correct for checking residual normality of treatment structure.
    try:
        # build full CRD model for residuals (intercept + all terms)
        tmp = anova_crd_full(long, self.factor_keys, levels_by_factor)
        # For residuals, we need actual residual vector: refit and compute
        # reconstruct full design matrix for CRD:
        factor_levels = {f: levels_by_factor[f] for f in self.factor_keys}
        y, mats, _ = build_term_matrices(long, factor_levels, include_block=False)
        term_defs = build_terms_crd(self.factor_keys)
        term_mats = {}
        term_order = []
        for name, flist in term_defs:
            term_mats[name] = make_term_matrix(mats, flist)
            term_order.append(name)
        X, _ = build_X_from_terms(intercept=True, term_mats=term_mats, term_order=term_order)
        yv = y.reshape(-1, 1)
        beta, _, _, _ = np.linalg.lstsq(X, yv, rcond=None)
        resid = (yv - X @ beta).flatten()
        if len(resid) >= 3:
            W, p_norm = shapiro(resid)
        else:
            W, p_norm = (np.nan, np.nan)
    except Exception:
        W, p_norm = (np.nan, np.nan)

    choice = self.choose_method_window(p_norm)
    if not choice["ok"]:
        return
    method = choice["method"]
    nonparam = method in ("mw", "kw")

    # variant order (by full factor combination)
    variant_order = first_seen_order([tuple(r.get(f) for f in self.factor_keys) for r in long])
    v_names = [" | ".join(map(str, k)) for k in variant_order]
    num_variants = len(variant_order)

    # groups by variants
    g_var = groups_by_keys(long, tuple(self.factor_keys))
    groups1 = {v_names[i]: g_var.get(variant_order[i], []) for i in range(len(variant_order))}

    # Brown–Forsythe for parametric only (over variants)
    bf_F, bf_p = (np.nan, np.nan)
    if not nonparam:
        bf_F, bf_p = brown_forsythe_from_groups(groups1)

    # Build model (FULL, no simplifications)
    model = None
    split_meta = None

    if not nonparam:
        if design == "CRD":
            model = anova_crd_full(long, self.factor_keys, levels_by_factor)
        elif design == "RCBD":
            model = anova_rcbd_full(long, self.factor_keys, levels_by_factor)
        else:
            if not main_factor:
                main_factor = self.factor_keys[0]
            model = anova_splitplot_full(long, self.factor_keys, levels_by_factor, main_factor)
            split_meta = {"main_factor": model["main_factor"], "df_error_main": model["df_error_main"], "df_error_sub": model["df_error_sub"]}

    # -------- Nonparametrics (expanded) --------
    kw_H = kw_p = kw_df = kw_eps2 = np.nan
    pairwise_rows = []
    if nonparam:
        if method == "kw":
            try:
                kw_samples = [groups1[name] for name in v_names if len(groups1[name]) > 0]
                if len(kw_samples) >= 2:
                    kw_res = kruskal(*kw_samples)
                    kw_H = float(kw_res.statistic)
                    kw_p = float(kw_res.pvalue)
                    kw_df = int(len(kw_samples) - 1)
                    n = len(long)
                    k = len(kw_samples)
                    kw_eps2 = float((kw_H - k + 1) / (n - k)) if (n - k) > 0 else np.nan
            except Exception:
                pass
            if not (isinstance(kw_p, float) and math.isnan(kw_p)) and kw_p < ALPHA:
                pairwise_rows, _ = pairwise_mw_bonf_with_effect(v_names, groups1, alpha=ALPHA)

        if method == "mw":
            pairwise_rows, _ = pairwise_mw_bonf_with_effect(v_names, groups1, alpha=ALPHA)

    # -------- Factor means for letters and LSD tables --------
    # Means per factor level:
    factor_groups = {f: {k[0]: v for k, v in groups_by_keys(long, (f,)).items()} for f in self.factor_keys}
    factor_means = {f: {lvl: float(np.mean(np.array(arr, dtype=float))) if len(arr) else np.nan
                        for lvl, arr in factor_groups[f].items()}
                    for f in self.factor_keys}
    factor_ns = {f: {lvl: int(len(arr)) for lvl, arr in factor_groups[f].items()} for f in self.factor_keys}

    # Letters per factor (only for LSD, and correct error stratum for split-plot)
    letters_factor = {f: {lvl: "" for lvl in levels_by_factor[f]} for f in self.factor_keys}
    if (not nonparam) and method == "lsd":
        for f in self.factor_keys:
            lvls = levels_by_factor[f]
            means_f = {str(lvl): factor_means[f].get(lvl, np.nan) for lvl in lvls}
            ns_f = {str(lvl): factor_ns[f].get(lvl, 0) for lvl in lvls}
            lvls_str = [str(lvl) for lvl in lvls]

            if design == "Split-plot":
                if f == model["main_factor"]:
                    MS_e, df_e = model["MS_error_main"], model["df_error_main"]
                else:
                    MS_e, df_e = model["MS_error_sub"], model["df_error_sub"]
            else:
                MS_e, df_e = model["MS_error"], model["df_error"]

            sig = lsd_sig_matrix(lvls_str, means_f, ns_f, MS_e, df_e, alpha=ALPHA)
            letters_map = cld_multi_letters(lvls_str, means_f, sig)
            for lvl in lvls:
                letters_factor[f][lvl] = letters_map.get(str(lvl), "")

    # Split-plot: букви істотності по варіантах НЕ подаємо
    # (методично), тому в таблиці варіантів завжди "-"
    # ------------------------------------------------------

    # =========================
    # REPORT segments
    # =========================
    seg = []
    seg.append(("text", "З В І Т   С Т А Т И С Т И Ч Н О Г О   А Н А Л І З У   Д А Н И Х\n\n"))
    seg.append(("text", f"Показник:\t{indicator}\nОдиниці виміру:\t{units}\n"))
    seg.append(("text", f"Дизайн експерименту:\t{design}\n"))
    if design == "Split-plot":
        seg.append(("text", f"Split-plot: головний фактор (main-plot):\t{main_factor}\n"))
    seg.append(("text", "\n"))

    seg.append(("text",
                f"Кількість варіантів:\t{num_variants}\n"
                f"Кількість повторностей:\t{len(used_rep_cols)}\n"
                f"Загальна кількість облікових значень:\t{len(long)}\n\n"))

    method_label = {
        "lsd": "Параметричний аналіз: Brown–Forsythe + ANOVA (Type III) + НІР₀₅ (LSD).",
        "tukey": "Параметричний аналіз: Brown–Forsythe + ANOVA (Type III) + тест Тьюкі.",
        "duncan": "Параметричний аналіз: Brown–Forsythe + ANOVA (Type III) + тест Дункана.",
        "bonferroni": "Параметричний аналіз: Brown–Forsythe + ANOVA (Type III) + корекція Бонферроні.",
        "kw": "Непараметричний аналіз: Kruskal–Wallis + (за потреби) парні Mann–Whitney з Bonferroni + ефект.",
        "mw": "Непараметричний аналіз: Mann–Whitney (парні порівняння) + Bonferroni + ефект.",
    }.get(method, "")
    if method_label:
        seg.append(("text", f"Виконуваний статистичний аналіз:\t{method_label}\n\n"))

    seg.append(("text", "Пояснення позначень істотності: ** — p<0.01; * — p<0.05.\n"))
    seg.append(("text", "У таблицях знак \"-\" свідчить що p ≥ 0.05.\n\n"))

    if not math.isnan(float(W)) if W is not None else False:
        seg.append(("text",
                    f"Перевірка нормальності залишків (Shapiro–Wilk):\t"
                    f"{normality_text(p_norm)}\t(W={fmt_num(float(W),4)}; p={fmt_num(float(p_norm),4)})\n\n"))
    else:
        seg.append(("text", "Перевірка нормальності залишків (Shapiro–Wilk):\tн/д\n\n"))

    if not nonparam:
        if not any(math.isnan(x) for x in [bf_F, bf_p]):
            bf_concl = "умова виконується" if bf_p >= ALPHA else f"умова порушена {significance_mark(bf_p)}"
            seg.append(("text",
                        f"Перевірка однорідності дисперсій (Brown–Forsythe):\t"
                        f"F={fmt_num(bf_F,4)}; p={fmt_num(bf_p,4)}\t{bf_concl}.\n\n"))
        else:
            seg.append(("text", "Перевірка однорідності дисперсій (Brown–Forsythe):\tн/д\n\n"))

        # ---------- ANOVA table ----------
        anova_rows = []
        for name, SSv, dfv, MSv, Fv, pv, strata in model["table"]:
            df_txt = str(int(dfv)) if dfv is not None and (not math.isnan(float(dfv))) else ""
            if name.startswith("Залишок") or name.startswith("Загальна") or "Error" in name:
                anova_rows.append([name, fmt_num(SSv, 2), df_txt, fmt_num(MSv, 3), "", "", ""])
            else:
                mark = significance_mark(pv) if pv is not None else ""
                concl = f"істотна різниця {mark}" if mark else "-"
                anova_rows.append([name, fmt_num(SSv, 2), df_txt, fmt_num(MSv, 3), fmt_num(Fv, 3), fmt_num(pv, 4) if pv is not None else "", concl])

        seg.append(("text", "ТАБЛИЦЯ 1. Дисперсійний аналіз (ANOVA, Type III)\n"))
        seg.append(("table", {"headers": ["Джерело", "SS", "df", "MS", "F", "p", "Висновок"],
                              "rows": anova_rows,
                              "padding_px": 32,
                              "extra_gap_after_col": 0,
                              "extra_gap_px": 60}))
        seg.append(("text", "\n"))

        # ---------- Effect sizes (percent SS + partial eta2 by correct stratum) ----------
        SS_total = model["SS_total"]

        pct_rows = []
        eta_rows = []
        for name, SSv, dfv, MSv, Fv, pv, strata in model["table"]:
            if name.startswith("Залишок") or name.startswith("Загальна") or "Error" in name:
                continue
            if SSv is None or math.isnan(float(SSv)):
                continue
            pct_rows.append([name, fmt_num(100.0 * float(SSv) / float(SS_total), 2)])

            # partial eta2 depends on stratum:
            if design == "Split-plot":
                if strata == "A":
                    eta = partial_eta2(SSv, model["SS_error_main"])
                else:
                    eta = partial_eta2(SSv, model["SS_error_sub"])
            else:
                eta = partial_eta2(SSv, model["SS_error"])
            eta_rows.append([name, fmt_num(eta, 4)])

        seg.append(("text", "ТАБЛИЦЯ 2. Сила впливу факторів і комбінацій (% від SS)\n"))
        seg.append(("table", {"headers": ["Джерело", "% від SS"], "rows": pct_rows}))
        seg.append(("text", "\n"))

        seg.append(("text", "ТАБЛИЦЯ 3. Розмір ефекту (partial η²)\n"))
        seg.append(("table", {"headers": ["Джерело", "partial η²"], "rows": eta_rows}))
        seg.append(("text", "\n"))

        # ---------- CV ----------
        gmean = model.get("grand_mean", float(np.mean(values)))
        cv_rows = []
        if design == "Split-plot":
            cv_main = cv_from_ms(model["MS_error_main"], gmean)
            cv_sub = cv_from_ms(model["MS_error_sub"], gmean)
            cv_total = cv_percent_from_values(values)
            cv_rows.append(["Main-plot (за MS Error A)", fmt_num(cv_main, 2)])
            cv_rows.append(["Sub-plot (за MS Error B)", fmt_num(cv_sub, 2)])
            cv_rows.append(["Загальний (за даними)", fmt_num(cv_total, 2)])
        else:
            cv_model = cv_from_ms(model["MS_error"], gmean)
            cv_total = cv_percent_from_values(values)
            cv_rows.append(["Модельний (за MS_error)", fmt_num(cv_model, 2)])
            cv_rows.append(["Загальний (за даними)", fmt_num(cv_total, 2)])

        seg.append(("text", "ТАБЛИЦЯ 4. Коефіцієнт варіації (CV, %)\n"))
        seg.append(("table", {"headers": ["Елемент", "CV, %"], "rows": cv_rows}))
        seg.append(("text", "\n"))

        # ---------- R² after CV ----------
        if design == "Split-plot":
            SS_err = float(model["SS_error_main"] + model["SS_error_sub"])
            R2 = r2_from_ss(model["SS_total"], SS_err)
            r2_rows = [["R² (за сумарним залишком)", fmt_num(R2, 4)]]
        else:
            R2 = r2_from_ss(model["SS_total"], model["SS_error"])
            r2_rows = [["R²", fmt_num(R2, 4)]]

        seg.append(("text", "ТАБЛИЦЯ 5. Коефіцієнт детермінації (R²)\n"))
        seg.append(("table", {"headers": ["Показник", "Значення"], "rows": r2_rows}))
        seg.append(("text", "\n"))

        # ---------- LSD table by factors + overall ----------
        if method == "lsd":
            lsd_rows = []
            if design == "Split-plot":
                for f in self.factor_keys:
                    if f == model["main_factor"]:
                        MS_e, df_e = model["MS_error_main"], model["df_error_main"]
                    else:
                        MS_e, df_e = model["MS_error_sub"], model["df_error_sub"]
                    ns = [factor_ns[f].get(lvl, 0) for lvl in levels_by_factor[f]]
                    n_eff = harmonic_mean(ns)
                    lsd = lsd_value(ALPHA, df_e, MS_e, n_eff) if not math.isnan(n_eff) else np.nan
                    lsd_rows.append([f"Фактор {f}", fmt_num(MS_e, 4), str(int(df_e)), fmt_num(n_eff, 2), fmt_num(lsd, 4)])

                # overall (variants): by Error B
                ns_v = [len(groups1[name]) for name in v_names]
                n_eff_v = harmonic_mean(ns_v)
                lsd_v = lsd_value(ALPHA, model["df_error_sub"], model["MS_error_sub"], n_eff_v) if not math.isnan(n_eff_v) else np.nan
                lsd_rows.append(["Загальна (варіанти, Error B)", fmt_num(model["MS_error_sub"], 4), str(int(model["df_error_sub"])),
                                 fmt_num(n_eff_v, 2), fmt_num(lsd_v, 4)])
            else:
                MS_e, df_e = model["MS_error"], model["df_error"]
                for f in self.factor_keys:
                    ns = [factor_ns[f].get(lvl, 0) for lvl in levels_by_factor[f]]
                    n_eff = harmonic_mean(ns)
                    lsd = lsd_value(ALPHA, df_e, MS_e, n_eff) if not math.isnan(n_eff) else np.nan
                    lsd_rows.append([f"Фактор {f}", fmt_num(MS_e, 4), str(int(df_e)), fmt_num(n_eff, 2), fmt_num(lsd, 4)])

                ns_v = [len(groups1[name]) for name in v_names]
                n_eff_v = harmonic_mean(ns_v)
                lsd_v = lsd_value(ALPHA, df_e, MS_e, n_eff_v) if not math.isnan(n_eff_v) else np.nan
                lsd_rows.append(["Загальна (варіанти)", fmt_num(MS_e, 4), str(int(df_e)), fmt_num(n_eff_v, 2), fmt_num(lsd_v, 4)])

            seg.append(("text", "ТАБЛИЦЯ 6. НІР₀₅ (LSD) — по факторах та загальна\n"))
            seg.append(("table", {"headers": ["Елемент", "MS_error", "df_error", "n_eff", "НІР₀₅"], "rows": lsd_rows}))
            seg.append(("text", "\n"))

        # ---------- Means per factor (letters for LSD) ----------
        tno = 7
        for f in self.factor_keys:
            seg.append(("text", f"ТАБЛИЦЯ {tno}. Середнє по фактору {f}\n"))
            rows_f = []
            for lvl in levels_by_factor[f]:
                m = factor_means[f].get(lvl, np.nan)
                letter = letters_factor[f].get(lvl, "")
                rows_f.append([str(lvl), fmt_num(m, 3), (letter if letter else "-")])
            seg.append(("table", {"headers": [f"Градація {f}", "Середнє", "Істотна різниця"], "rows": rows_f}))
            seg.append(("text", "\n"))
            tno += 1

        # ---------- Variants mean±sd table (split-plot: letters '-') ----------
        seg.append(("text", f"ТАБЛИЦЯ {tno}. Таблиця середніх значень варіантів\n"))
        rows_v = []
        for i, k in enumerate(variant_order):
            name = " | ".join(map(str, k))
            arr = np.array(g_var.get(k, []), dtype=float)
            arr = arr[~np.isnan(arr)]
            m = float(np.mean(arr)) if len(arr) else np.nan
            sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else (0.0 if len(arr) == 1 else np.nan)
            rows_v.append([name, fmt_num(m, 3), fmt_num(sd, 3), "-"])
        seg.append(("table", {"headers": ["Варіант", "Середнє", "± SD", "Істотна різниця"],
                              "rows": rows_v,
                              "padding_px": 32,
                              "extra_gap_after_col": 0,
                              "extra_gap_px": 80}))
        seg.append(("text", "\n"))

        if design == "Split-plot":
            seg.append(("text", "Примітка (Split-plot): буквені позначення істотності по варіантах у звіті не подаються методично.\n\n"))

    # ---------- NONPARAMETRIC (expanded, not simplified) ----------
    if nonparam:
        seg.append(("text", "НЕПАРАМЕТРИЧНИЙ ЗВІТ\n\n"))

        desc_rows = []
        for name in v_names:
            d = describe_array(groups1[name])
            desc_rows.append([
                name,
                str(d["n"]),
                fmt_num(d["med"], 3),
                fmt_num(d["q1"], 3),
                fmt_num(d["q3"], 3),
                fmt_num(d["mn"], 3),
                fmt_num(d["mx"], 3),
            ])
        seg.append(("text", "ТАБЛИЦЯ 1. Описова статистика (median, IQR, min–max)\n"))
        seg.append(("table", {"headers": ["Варіант", "n", "Median", "Q1", "Q3", "Min", "Max"],
                              "rows": desc_rows,
                              "padding_px": 32,
                              "extra_gap_after_col": 0,
                              "extra_gap_px": 80}))
        seg.append(("text", "\n"))

        if method == "kw":
            seg.append(("text",
                        f"Глобальний тест (Kruskal–Wallis):\tH={fmt_num(kw_H,4)}; df={int(kw_df) if not math.isnan(kw_df) else ''}; "
                        f"p={fmt_num(kw_p,4)}\t{('істотна різниця *' if (not math.isnan(kw_p) and kw_p<ALPHA) else '-')}\n"))
            seg.append(("text", f"Розмір ефекту (ε²):\t{fmt_num(kw_eps2,4)}\n\n"))

        if pairwise_rows:
            seg.append(("text", "ТАБЛИЦЯ 2. Парні порівняння Mann–Whitney + Bonferroni + Cliff’s δ\n"))
            seg.append(("table", {"headers": ["Комбінація", "U", "p", "p(Bonf.)", "Істотна", "δ", "Ефект"],
                                  "rows": pairwise_rows,
                                  "padding_px": 32,
                                  "extra_gap_after_col": 0,
                                  "extra_gap_px": 80}))
            seg.append(("text", "\n"))

        seg.append(("text", f"Звіт сформовано:\t{created_at.strftime('%d.%m.%Y, %H:%M')}\n"))

        # plot too (same)
        plot_png = build_factor_block_boxplot_png(long, self.factor_keys, levels_by_factor, letters_factor, indicator, units)
        self.show_report_segments(seg, plot_png_path=plot_png, plot_meta={"design": design, "split": split_meta})
        return

    # -------- Parametric: build plot and show report --------
    plot_png = build_factor_block_boxplot_png(long, self.factor_keys, levels_by_factor, letters_factor, indicator, units)
    seg.append(("text", f"Звіт сформовано:\t{created_at.strftime('%d.%m.%Y, %H:%M')}\n"))
    self.show_report_segments(seg, plot_png_path=plot_png, plot_meta={"design": design, "split": split_meta})

# -------------------------
# Bind patched methods into class (so PART 1 stays unchanged)
# -------------------------
SADTk.analyze = analyze
SADTk.show_plot_window = show_plot_window
