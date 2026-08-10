# main.py  — S.A.D. v2.1
# -*- coding: utf-8 -*-
"""
S.A.D. — Статистичний аналіз даних  v2.1
Залежності: pip install numpy scipy openpyxl matplotlib pillow
"""
import os, sys, math, json, io

# ── Визначення платформи ───────────────────────────────────────
IS_WIN   = sys.platform == "win32"
IS_MAC   = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")

if IS_MAC:
    FONT_SERIF = "Times New Roman"
    FONT_SANS  = "Helvetica Neue"
    FONT_MONO  = "Menlo"
elif IS_WIN:
    FONT_SERIF = "Times New Roman"
    FONT_SANS  = "Arial"
    FONT_MONO  = "Consolas"
else:
    FONT_SERIF = "DejaVu Serif"
    FONT_SANS  = "DejaVu Sans"
    FONT_MONO  = "DejaVu Sans Mono"


def open_file_cross(path: str):
    """Відкриває файл стандартним застосунком на будь-якій платформі."""
    try:
        if IS_WIN:
            os.startfile(path)
        elif IS_MAC:
            import subprocess; subprocess.Popen(["open", path])
        else:
            import subprocess; subprocess.Popen(["xdg-open", path])
    except Exception:
        pass


def maximize_win(win):
    """Розгортає вікно на весь екран на Windows, macOS і Linux."""
    try:
        if IS_WIN:
            win.state("zoomed")
        elif IS_MAC:
            win.update_idletasks()
            sw = win.winfo_screenwidth()
            sh = win.winfo_screenheight()
            win.geometry(f"{sw}x{sh-25}+0+25")
        else:
            try:
                win.attributes("-zoomed", True)
            except Exception:
                sw = win.winfo_screenwidth()
                sh = win.winfo_screenheight()
                win.geometry(f"{sw}x{sh}+0+0")
    except Exception:
        win.geometry("1280x800")
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser, simpledialog
from tkinter.scrolledtext import ScrolledText
import tkinter.font as tkfont
from itertools import combinations
from collections import defaultdict
from datetime import datetime

from scipy.stats import (shapiro, kruskal, mannwhitneyu, friedmanchisquare,
                         wilcoxon, levene, pearsonr, spearmanr)
from scipy.stats import f as f_dist, t as t_dist, norm
from scipy.stats import studentized_range

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    HAS_MPL = True
except Exception:
    HAS_MPL = False; Figure = None; FigureCanvasTkAgg = None

try:
    from PIL import Image as _PILImage
    HAS_PIL = True
except Exception:
    HAS_PIL = False

ALPHA   = 0.05
COL_W   = 10
APP_VER = "2.1"

if HAS_MPL:
    import matplotlib
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'font.serif':  ['Times New Roman','Times','DejaVu Serif'],
        'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'axes.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False,
    })

    def get_cmap_safe(name, fallback="RdYlGn"):
        """Повертає Colormap незалежно від версії matplotlib.

        matplotlib.cm.get_cmap() було прибрано в 3.9, тимчасово повернено
        в 3.9.1 і остаточно видалено в 3.11. Новий API — matplotlib.colormaps[name] —
        доступний з 3.5, тож пробуємо його першим і падаємо назад на старий
        виклик лише для дуже старих версій.
        """
        for nm in (name, fallback):
            try:
                return matplotlib.colormaps[nm]
            except Exception:
                pass
            try:
                return matplotlib.cm.get_cmap(nm)
            except Exception:
                pass
        # Останній рубіж — вбудована палітра, яка існує завжди
        return matplotlib.colormaps["viridis"]

# ── DPI awareness ──────────────────────────────────────────────
if IS_WIN:
    try:
        import ctypes
        try:    ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try: ctypes.windll.user32.SetProcessDPIAware()
            except Exception: pass
    except Exception: pass

# ── Icon ──────────────────────────────────────────────────────
def _app_dirs():
    """Кандидати папок, де шукати icon.ico / Logo.png: поруч зі скриптом,
    поточна робоча папка, папка самого exe (після встановлення на ПК —
    саме там реально лежать файли користувача), і тимчасова папка
    розпакування PyInstaller (де самих файлів користувача, як правило,
    немає, але лишаємо про всяк випадок)."""
    dirs = [os.getcwd()]
    try: dirs.insert(0, os.path.dirname(os.path.abspath(__file__)))
    except Exception: pass
    try:
        if getattr(sys, "frozen", False):
            dirs.insert(0, os.path.dirname(os.path.abspath(sys.executable)))
    except Exception: pass
    try:
        if hasattr(sys, "_MEIPASS"): dirs.append(sys._MEIPASS)
    except Exception: pass
    # унікалізуємо, зберігаючи порядок
    seen = set(); out = []
    for d in dirs:
        if d and d not in seen:
            seen.add(d); out.append(d)
    return out

def _find_file(*names):
    for d in _app_dirs():
        for nm in names:
            p = os.path.join(d, nm)
            if os.path.exists(p): return p
    return None

def _find_icon():
    return _find_file("icon.ico")

def set_icon(win):
    ico = _find_icon()
    if not ico: return
    try: win.iconbitmap(ico)
    except Exception:
        try: win.iconbitmap(default=ico)
        except Exception: pass

# ── Clipboard PNG → Windows ────────────────────────────────────
def embed_figure(fig, master, dpi=96):
    """Вставляє matplotlib Figure у tkinter frame."""
    cv = FigureCanvasTkAgg(fig, master=master)
    widget = cv.get_tk_widget()
    widget.pack(fill=tk.BOTH, expand=True)
    cv.draw()
    return cv


def embed_figure_scrollable(fig, master, dpi=96):
    """Вставляє matplotlib Figure у tkinter frame ЗІ СКРОЛОМ, зберігаючи її
    природний (фіксований) розмір у пікселях — на відміну від embed_figure,
    яка стискає фігуру під розмір вікна. Використовується для великих
    матриць/сіток, де кожна клітинка має лишатись читабельного розміру
    незалежно від загальної кількості клітинок."""
    outer = tk.Frame(master)
    outer.pack(fill=tk.BOTH, expand=True)
    vsb = ttk.Scrollbar(outer, orient="vertical")
    vsb.pack(side=tk.RIGHT, fill=tk.Y)
    hsb = ttk.Scrollbar(outer, orient="horizontal")
    hsb.pack(side=tk.BOTTOM, fill=tk.X)
    cv_container = tk.Canvas(outer, yscrollcommand=vsb.set, xscrollcommand=hsb.set,
                              highlightthickness=0, bg="white")
    cv_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    vsb.config(command=cv_container.yview)
    hsb.config(command=cv_container.xview)

    cv = FigureCanvasTkAgg(fig, master=cv_container)
    widget = cv.get_tk_widget()
    cv.draw()

    win_id = cv_container.create_window((0, 0), window=widget, anchor="nw")

    def _update_scrollregion(event=None):
        # Невеликий запас (+4px), щоб нижній ряд і правий стовпець точно
        # не обрізались через заокруглення розмірів рендеру.
        bbox = cv_container.bbox(win_id)
        if bbox:
            x0, y0, x1, y1 = bbox
            cv_container.configure(scrollregion=(x0, y0, x1 + 4, y1 + 4))
    widget.bind("<Configure>", _update_scrollregion)
    cv_container.after(50, _update_scrollregion)
    cv_container.after(300, _update_scrollregion)

    def _on_mw(e):
        cv_container.yview_scroll(int(-1*(e.delta/120)), "units")
    def _on_shift_mw(e):
        cv_container.xview_scroll(int(-1*(e.delta/120)), "units")
    widget.bind("<MouseWheel>", _on_mw)
    widget.bind("<Shift-MouseWheel>", _on_shift_mw)
    cv_container.bind("<MouseWheel>", _on_mw)
    cv_container.bind("<Shift-MouseWheel>", _on_shift_mw)
    return cv


def _copy_fig_to_clipboard(fig):
    if not (HAS_MPL and HAS_PIL): return False, "Потрібні matplotlib і Pillow"
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        pil = _PILImage.open(buf)
        ok, msg = _copy_pil_win(pil); buf.close(); return ok, msg
    except Exception as ex: return False, str(ex)

def _copy_pil_win(pil_img):
    try: import ctypes; from ctypes import wintypes
    except Exception: return False, "ctypes недоступний"
    if sys.platform != "win32": return False, "Лише для Windows"
    if pil_img is None: return False, "Немає зображення"
    try:
        buf = io.BytesIO(); pil_img.convert("RGB").save(buf, "BMP"); bmp = buf.getvalue()
        if len(bmp) <= 14: return False, "BMP помилка"
        data = bmp[14:]
    except Exception as ex: return False, str(ex)
    u32 = ctypes.WinDLL("user32", use_last_error=True)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    u32.OpenClipboard.argtypes  = [wintypes.HWND];           u32.OpenClipboard.restype  = wintypes.BOOL
    u32.CloseClipboard.argtypes = [];                        u32.CloseClipboard.restype  = wintypes.BOOL
    u32.EmptyClipboard.argtypes = [];                        u32.EmptyClipboard.restype  = wintypes.BOOL
    u32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]; u32.SetClipboardData.restype = wintypes.HANDLE
    k32.GlobalAlloc.argtypes  = [wintypes.UINT, ctypes.c_size_t]; k32.GlobalAlloc.restype  = wintypes.HGLOBAL
    k32.GlobalLock.argtypes   = [wintypes.HGLOBAL];          k32.GlobalLock.restype   = wintypes.LPVOID
    k32.GlobalUnlock.argtypes = [wintypes.HGLOBAL];          k32.GlobalUnlock.restype = wintypes.BOOL
    k32.GlobalFree.argtypes   = [wintypes.HGLOBAL];          k32.GlobalFree.restype   = wintypes.HGLOBAL
    if not u32.OpenClipboard(None): return False, f"OpenClipboard err {ctypes.get_last_error()}"
    try:
        u32.EmptyClipboard()
        hg = k32.GlobalAlloc(0x0042, len(data))
        if not hg: return False, "GlobalAlloc failed"
        pg = k32.GlobalLock(hg)
        if not pg: k32.GlobalFree(hg); return False, "GlobalLock failed"
        try: ctypes.memmove(pg, data, len(data))
        finally: k32.GlobalUnlock(hg)
        if not u32.SetClipboardData(8, hg): k32.GlobalFree(hg); return False, "SetClipboardData failed"
        return True, ""
    finally: u32.CloseClipboard()

# ═══════════════════════════════════════════════════════════════
# STAT HELPERS
# ═══════════════════════════════════════════════════════════════
def sig_mark(p):
    if p is None or (isinstance(p, float) and math.isnan(p)): return ""
    return "**" if p < 0.01 else ("*" if p < 0.05 else "")

def norm_txt(p):
    if p is None or (isinstance(p, float) and math.isnan(p)): return "н/д"
    return "нормальний розподіл" if p > 0.05 else "ненормальний розподіл"

def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)): return ""
    try: return f"{float(x):.{nd}f}"
    except (TypeError, ValueError): return ""

def first_seen(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen: seen.add(x); out.append(x)
    return out

def center_win(win):
    win.update_idletasks()
    w, h = win.winfo_width(), win.winfo_height()
    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    win.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

def median_q(arr):
    if not arr: return np.nan, np.nan, np.nan
    a = np.array(arr, dtype=float); a = a[~np.isnan(a)]
    if len(a) == 0: return np.nan, np.nan, np.nan
    return float(np.median(a)), float(np.percentile(a, 25)), float(np.percentile(a, 75))

def cv_vals(vals):
    a = np.array(vals, dtype=float); a = a[~np.isnan(a)]
    if len(a) < 2: return np.nan
    m = float(np.mean(a))
    return np.nan if m == 0 else float(np.std(a, ddof=1) / m * 100)

def cv_means(means):
    v = [float(x) for x in means if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(v) < 2: return np.nan
    m = float(np.mean(v))
    return np.nan if m == 0 else float(np.std(v, ddof=1) / m * 100)

def eta2_label(pe2):
    if pe2 is None or math.isnan(pe2): return ""
    if pe2 < 0.01: return "дуже слабкий"
    if pe2 < 0.06: return "слабкий"
    if pe2 < 0.14: return "середній"
    return "сильний"

def eps2_kw(H, n, k):
    if any(x is None for x in [H, n, k]) or math.isnan(H) or n <= k or k < 2: return np.nan
    return float((H - k + 1) / (n - k))

def kendalls_w(chisq, nb, kt):
    if any(x is None for x in [chisq, nb, kt]) or math.isnan(chisq) or nb <= 0 or kt <= 1: return np.nan
    return float(chisq / (nb * (kt - 1)))

def cliffs_d(x, y):
    x = np.array(x, dtype=float); y = np.array(y, dtype=float)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0: return np.nan
    gt = int(np.sum(x[:, None] > y[None, :])); lt = int(np.sum(x[:, None] < y[None, :]))
    return float((gt - lt) / (nx * ny))

def cliffs_lbl(d):
    if d is None or math.isnan(d): return ""
    if d < 0.147: return "дуже слабкий"
    if d < 0.33:  return "слабкий"
    if d < 0.474: return "середній"
    return "сильний"

def fit_font(texts, family="Times New Roman", start=13, min_s=9, target=155):
    f = tkfont.Font(family=family, size=start); sz = start
    while sz > min_s:
        if max(f.measure(t) for t in texts) <= target: break
        sz -= 1; f.configure(size=sz)
    return f

# ═══════════════════════════════════════════════════════════════
# ЗБЕРЕЖЕННЯ / ВІДКРИТТЯ ПРОЕКТУ (універсально для будь-якої таблиці)
# ═══════════════════════════════════════════════════════════════
def _get_header_texts(header_widgets):
    """Повертає список текстів заголовків незалежно від того, чи це
    StringVar, Entry, чи Label (усі три патерни зустрічаються в програмі)."""
    out = []
    for h in header_widgets:
        if hasattr(h, "get"):
            out.append(h.get())
        else:
            out.append(str(h.cget("text")))
    return out

def generic_save_project(win, proj_type, header_widgets, entries, extra=None):
    """Зберігає заголовки і вміст таблиці entries у файл проекту .sadp (JSON)."""
    path = filedialog.asksaveasfilename(
        parent=win, defaultextension=".sadp",
        filetypes=[("SAD проект","*.sadp"),("JSON","*.json")],
        title="Зберегти проект")
    if not path: return
    d = {"type": proj_type, "version": APP_VER,
         "headers": _get_header_texts(header_widgets) if header_widgets else [],
         "rows_data": [[e.get() for e in row] for row in entries]}
    if extra: d.update(extra)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        messagebox.showinfo("Збережено", f"Проект збережено:\n{path}")
    except Exception as ex:
        messagebox.showerror("Помилка збереження", str(ex))

def _set_header_text(widget, text):
    """Встановлює текст заголовка незалежно від того, чи це StringVar
    (має .set) чи Entry (має .delete/.insert)."""
    if hasattr(widget, "set"):
        widget.set(text)
    elif hasattr(widget, "delete") and hasattr(widget, "insert"):
        widget.delete(0, tk.END); widget.insert(0, text)

def generic_load_project(win):
    """Відкриває файл проекту .sadp (JSON) і повертає його вміст (dict),
    або None якщо користувач скасував/сталася помилка."""
    path = filedialog.askopenfilename(
        parent=win, filetypes=[("SAD проект","*.sadp"),("JSON","*.json")],
        title="Відкрити проект")
    if not path: return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as ex:
        messagebox.showerror("Помилка відкриття", str(ex)); return None


# ═══════════════════════════════════════════════════════════════
# TREEVIEW TABLE (fixed columns, no drift)
# ═══════════════════════════════════════════════════════════════
def make_tv(parent, headers, rows, min_col=90):
    frm = tk.Frame(parent, bd=1, relief=tk.SUNKEN)
    vsb = ttk.Scrollbar(frm, orient="vertical")
    hsb = ttk.Scrollbar(frm, orient="horizontal")
    tv = ttk.Treeview(frm, columns=headers, show="headings",
                      yscrollcommand=vsb.set, xscrollcommand=hsb.set,
                      height=min(len(rows) + 1, 22))
    vsb.config(command=tv.yview); hsb.config(command=tv.xview)
    vsb.pack(side=tk.RIGHT, fill=tk.Y)
    hsb.pack(side=tk.BOTTOM, fill=tk.X)
    tv.pack(fill=tk.BOTH, expand=True)
    fnt = tkfont.Font(family="Times New Roman", size=11)
    for i, h in enumerate(headers):
        cw = max(fnt.measure(str(h)) + 24, min_col,
                 max((fnt.measure(str(r[i]) if i < len(r) and r[i] else "") + 20) for r in rows) if rows else min_col)
        tv.heading(h, text=str(h), anchor="center")
        tv.column(h, width=cw, minwidth=50, anchor="center", stretch=True)
    for row in rows:
        tv.insert("", "end", values=[("" if v is None else str(v)) for v in row])
    style = ttk.Style()
    style.configure("Treeview", font=("Times New Roman", 11), rowheight=22)
    style.configure("Treeview.Heading", font=("Times New Roman", 11, "bold"))
    return frm, tv

# ═══════════════════════════════════════════════════════════════
# DATA HELPERS
# ═══════════════════════════════════════════════════════════════

def _nav_on_enter(event):
    e = event.widget
    i, j = getattr(e, "_nav_pos", (None, None))
    entries_2d = getattr(e, "_nav_entries", None)
    if i is None or entries_2d is None: return "break"
    ni = i + 1
    if ni >= len(entries_2d) or j >= len(entries_2d[ni]): return "break"
    target = entries_2d[ni][j]
    target.focus_set(); target.icursor(tk.END)
    return "break"

def _nav_on_arrow(event):
    e = event.widget
    i, j = getattr(e, "_nav_pos", (None, None))
    entries_2d = getattr(e, "_nav_entries", None)
    if i is None or entries_2d is None: return "break"
    if event.keysym == "Up":     i = max(0, i-1)
    elif event.keysym == "Down": i = min(len(entries_2d)-1, i+1)
    elif event.keysym == "Left": j = max(0, j-1)
    elif event.keysym == "Right":j = min(len(entries_2d[i])-1, j+1)
    if i >= len(entries_2d) or j >= len(entries_2d[i]): return "break"
    target = entries_2d[i][j]
    target.focus_set(); target.icursor(tk.END)
    return "break"

def _autofit_col(entries, col_idx, header_entries=None, min_w=8, max_w=45):
    """Розширює один стовпець таблиці (Entry-віджети в grid) під найдовший
    текст, що зараз у ньому є — заголовок і всі клітинки. Оскільки в
    tkinter.grid усі віджети одного стовпця поділяють однакову ширину
    стовпця (за найширшим серед них), достатньо виставити width всім
    коміркам цього стовпця — вирівнюються вони автоматично.
    Використовується для текстових стовпців (назви груп/варіантів/факторів),
    де довжина вмісту наперед невідома і сильно різниться.
    header_entries може містити Entry (з .get()) або Label (з .cget("text"))."""
    def _hdr_text(w):
        return w.get() if hasattr(w, "get") else str(w.cget("text"))
    lengths = []
    if header_entries is not None and col_idx < len(header_entries):
        lengths.append(len(_hdr_text(header_entries[col_idx])))
    for row in entries:
        if col_idx < len(row):
            lengths.append(len(row[col_idx].get()))
    new_w = max([min_w] + lengths)
    new_w = min(new_w, max_w)
    if header_entries is not None and col_idx < len(header_entries):
        try: header_entries[col_idx].configure(width=new_w)
        except Exception: pass
    for row in entries:
        if col_idx < len(row):
            row[col_idx].configure(width=new_w)


def _bind_nav(entries_2d, win, factors_count=0):
    """Прив'язати навігацію Enter/стрілки до двовимірного масиву Entry.

    Позиція кожної комірки кешується як атрибут самого віджета — пошук
    O(1) замість лінійного сканування всієї таблиці на кожне натискання
    Enter/стрілки (це особливо помітно на великих таблицях, на сотні
    комірок: раніше кожен keypress пересканував ВЕСЬ масив).
    Також уже прив'язані комірки не перебиндовуються повторно — виклик
    цієї функції після додавання одного рядка до великої таблиці більше
    не проходиться по ВСІХ існуючих комірках заново, а лише по нових."""
    for i, row in enumerate(entries_2d):
        for j, e in enumerate(row):
            e._nav_pos = (i, j)
            e._nav_entries = entries_2d
            if getattr(e, "_nav_bound", False):
                continue
            e.bind("<Return>", _nav_on_enter)
            e.bind("<Up>",     _nav_on_arrow)
            e.bind("<Down>",   _nav_on_arrow)
            e.bind("<Left>",   _nav_on_arrow)
            e.bind("<Right>",  _nav_on_arrow)
            e._nav_bound = True


def _bind_fill_handle(entries_2d, win):
    """Реалізує 'протягування' комірки за правий нижній кут — як в Excel.
    Клік і утримування саме в куточку комірки (кілька пікселів біля
    правого нижнього краю) починає виділення прямокутного діапазону;
    відпускання кнопки миші копіює вміст ПОЧАТКОВОЇ комірки в усі
    клітинки виділеного діапазону. Звичайний клік/виділення тексту
    всередині комірки при цьому не зачіпається."""
    HANDLE_PX = 5
    state = {"dragging": False, "origin": None, "highlighted": []}

    def _cell_pos(w):
        return getattr(w, "_nav_pos", None)

    def _clear_highlight():
        for e in state["highlighted"]:
            try: e.configure(bg="white")
            except Exception: pass
        state["highlighted"] = []

    def _on_press(event):
        w = event.widget
        if event.x >= w.winfo_width() - HANDLE_PX and event.y >= w.winfo_height() - HANDLE_PX:
            pos = _cell_pos(w)
            if pos is None: return
            state["dragging"] = True
            state["origin"] = pos
            w.configure(cursor="crosshair")
            return "break"

    def _on_motion(event):
        if not state["dragging"]: return
        oi, oj = state["origin"]
        target = win.winfo_containing(event.x_root, event.y_root)
        if not isinstance(target, tk.Entry): return "break"
        pos = _cell_pos(target)
        if pos is None: return "break"
        ti, tj = pos
        _clear_highlight()
        i0, i1 = sorted((oi, ti)); j0, j1 = sorted((oj, tj))
        for i in range(i0, i1+1):
            if i >= len(entries_2d): continue
            for j in range(j0, j1+1):
                if j >= len(entries_2d[i]): continue
                if i == oi and j == oj: continue
                e = entries_2d[i][j]
                e.configure(bg="#dbe9fb")
                state["highlighted"].append(e)
        return "break"

    def _on_release(event):
        if not state["dragging"]: return
        state["dragging"] = False
        oi, oj = state["origin"]
        origin_val = entries_2d[oi][oj].get()
        for e in state["highlighted"]:
            e.delete(0, tk.END); e.insert(0, origin_val)
        _clear_highlight()
        try: event.widget.configure(cursor="xterm")
        except Exception: pass
        state["origin"] = None
        return "break"

    for row in entries_2d:
        for e in row:
            if getattr(e, "_fill_bound", False):
                continue
            e.bind("<ButtonPress-1>", _on_press, add="+")
            e.bind("<B1-Motion>", _on_motion, add="+")
            e.bind("<ButtonRelease-1>", _on_release, add="+")
            e._fill_bound = True


def _nav_move(entries_2d, ri, ci):
    if 0 <= ri < len(entries_2d) and 0 <= ci < len(entries_2d[ri]):
        entries_2d[ri][ci].focus_set(); entries_2d[ri][ci].icursor(tk.END)
    return "break"

def _nav_down(entries_2d, ri, ci, add_row_fn=None):
    nri = ri + 1
    if nri >= len(entries_2d) and add_row_fn:
        try: add_row_fn()
        except Exception: pass
    _nav_move(entries_2d, min(nri, len(entries_2d)-1), ci)
    return "break"

def bind_nav(entries_2d, e, add_row_fn=None):
    """Прив'язати навігацію Enter/стрілки до комірки Entry у двовимірному списку."""
    def find_pos():
        for ri, row in enumerate(entries_2d):
            for ci, cell in enumerate(row):
                if cell is e: return ri, ci
        return None, None

    def move(ri, ci):
        if 0 <= ri < len(entries_2d) and 0 <= ci < len(entries_2d[ri]):
            entries_2d[ri][ci].focus_set()
            entries_2d[ri][ci].icursor(tk.END)

    def on_enter(ev):
        ri, ci = find_pos()
        if ri is None: return "break"
        nri = ri + 1
        if nri >= len(entries_2d) and add_row_fn:
            add_row_fn()
        move(min(nri, len(entries_2d)-1), ci)
        return "break"

    def on_arrow(ev):
        ri, ci = find_pos()
        if ri is None: return "break"
        k = ev.keysym
        if   k == "Up":    move(max(0, ri-1), ci)
        elif k == "Down":  move(min(len(entries_2d)-1, ri+1), ci)
        elif k == "Left":  move(ri, max(0, ci-1))
        elif k == "Right": move(ri, min(len(entries_2d[ri])-1, ci+1))
        return "break"

    e.bind("<Return>", on_enter)
    e.bind("<Up>",    on_arrow)
    e.bind("<Down>",  on_arrow)
    e.bind("<Left>",  on_arrow)
    e.bind("<Right>", on_arrow)


def groups_by_keys(long, keys):
    """Групує числові значення з long за комбінацією довільних ключів keys."""
    g = defaultdict(list)
    for r in long:
        v = r.get("value", np.nan)
        if v is None or math.isnan(v): continue
        g[tuple(r.get(x) for x in keys)].append(float(v))
    return g

def vstats(long, fkeys):
    vals = defaultdict(list)
    for r in long:
        v = r.get("value", np.nan)
        if v is None or math.isnan(v): continue
        vals[tuple(r.get(k) for k in fkeys)].append(float(v))
    out = {}
    for k, a in vals.items():
        n = len(a); m = float(np.mean(a)) if n else np.nan
        sd = float(np.std(a, ddof=1)) if n >= 2 else (0. if n == 1 else np.nan)
        out[k] = (m, sd, n)
    return out

def mean_ranks(long, keyfn):
    vals = []; ks = []
    for r in long:
        v = r.get("value", np.nan)
        if v is None or math.isnan(v): continue
        vals.append(float(v)); ks.append(keyfn(r))
    if not vals: return {}
    order = np.argsort(vals); sv = np.array(vals)[order]
    ranks = np.empty(len(vals), dtype=float)
    i = 0
    while i < len(sv):
        j = i
        while j < len(sv) and sv[j] == sv[i]: j += 1
        ar = (i + 1 + j) / 2.; ranks[order[i:j]] = ar; i = j
    agg = defaultdict(list)
    for k, rk in zip(ks, ranks): agg[k].append(float(rk))
    return {k: float(np.mean(v)) for k, v in agg.items()}

# ═══════════════════════════════════════════════════════════════
# CLD
# ═══════════════════════════════════════════════════════════════
def cld(levels_order, means_dict, sig_matrix):
    valid = [l for l in levels_order if not math.isnan(means_dict.get(l, np.nan))]
    if not valid: return {l: "" for l in levels_order}
    sl = sorted(valid, key=lambda z: means_dict[z], reverse=True)
    def sig(a, b): return bool(sig_matrix.get((a, b), False) or sig_matrix.get((b, a), False))
    groups = []
    for lv in sl:
        compat = [gi for gi, g in enumerate(groups) if all(not sig(lv, o) for o in g)]
        if not compat: groups.append({lv})
        else:
            for gi in compat: groups[gi].add(lv)
    def shared(a, b): return any(a in g and b in g for g in groups)
    for i in range(len(sl)):
        for j in range(i + 1, len(sl)):
            a, b = sl[i], sl[j]
            if sig(a, b) or shared(a, b): continue
            ng = {a, b}
            for c in sl:
                if c in ng: continue
                if not sig(c, a) and not sig(c, b) and all(not sig(c, x) for x in ng): ng.add(c)
            groups.append(ng)
    uniq = []
    for g in groups:
        if not any(g == h for h in uniq): uniq.append(g)
    cleaned = [g for g in uniq if not any(g < h for h in uniq)]
    alpha_ = "abcdefghijklmnopqrstuvwxyz"
    mapping = {lv: [] for lv in sl}
    for gi, g in enumerate(cleaned):
        lt = alpha_[gi] if gi < len(alpha_) else f"g{gi}"
        for lv in g: mapping[lv].append(lt)
    return {lv: "".join(sorted(mapping.get(lv, []))) for lv in levels_order}

# ═══════════════════════════════════════════════════════════════
# LEVENE TEST
# ═══════════════════════════════════════════════════════════════
def groups_by(long, fkeys):
    """Групує значення з long по комбінаціям рівнів факторів fkeys.
    Повертає dict: {tuple_of_levels: [values]}
    Для одного ключа fkeys=(f,) → {(level,): [values]}
    """
    result = defaultdict(list)
    for r in long:
        key = tuple(r.get(f, "") for f in fkeys)
        result[key].append(r["value"])
    return dict(result)

def levene_test(groups_dict):
    arrs = [np.array(v, dtype=float) for v in groups_dict.values() if len(v) > 0]
    if len(arrs) < 2: return np.nan, np.nan
    try:
        stat, p = levene(*arrs, center='median')
        return float(stat), float(p)
    except Exception: return np.nan, np.nan

# ═══════════════════════════════════════════════════════════════
# PAIRWISE — parametric
# ═══════════════════════════════════════════════════════════════
def lsd_sig(levels, means, ns, MS, df, alpha=ALPHA):
    sig = {}
    if MS is None or df is None or math.isnan(MS) or math.isnan(df): return sig
    df = int(df);
    if df <= 0: return sig
    tc = float(t_dist.ppf(1 - alpha / 2, df))
    for a, b in combinations(levels, 2):
        ma, mb = means.get(a, np.nan), means.get(b, np.nan)
        na, nb = ns.get(a, 0), ns.get(b, 0)
        if any(math.isnan(x) for x in [ma, mb]) or na <= 0 or nb <= 0: continue
        se = math.sqrt(MS * (1 / na + 1 / nb))
        sig[(a, b)] = (abs(ma - mb) > tc * se)
    return sig

def pairwise_param(levels, means, ns, MS, df, method, alpha=ALPHA):
    """
    Parametric pairwise comparisons.
    Duncan: true step-down procedure — critical q depends on number of
    means spanned (p-range), not total m. This is methodologically correct.
    Tukey: simultaneous, uses studentized range with m groups.
    Bonferroni: Bonferroni-adjusted t-test.
    """
    rows = []; sig = {}
    if MS is None or df is None or math.isnan(MS) or math.isnan(df): return rows, sig
    df = int(df)
    if df <= 0: return rows, sig
    lvls = [x for x in levels if not math.isnan(means.get(x, np.nan)) and ns.get(x, 0) > 0]
    m = len(lvls)
    if m < 2: return rows, sig

    # For Duncan: sort means descending and compute step ranges
    if method == "duncan":
        sorted_lvls = sorted(lvls, key=lambda x: means[x], reverse=True)
        # Build significance matrix using step-down procedure
        for i in range(m):
            for j in range(i + 1, m):
                a, b = sorted_lvls[i], sorted_lvls[j]
                ma, mb = means[a], means[b]; na, nb = ns[a], ns[b]
                se = math.sqrt(MS * (1 / na + 1 / nb))
                if se <= 0: continue
                p_range = j - i + 1      # number of means spanned (≥ 2)
                # Duncan critical value: use studentized range with p_range groups
                # Duncan alpha per step: alpha_p = 1 - (1 - alpha)^(p-1)
                alpha_p = 1.0 - (1.0 - alpha) ** (p_range - 1)
                alpha_p = min(alpha_p, 0.5)  # cap for stability
                try:
                    q_crit = float(studentized_range.ppf(1 - alpha_p, p_range, df))
                    lsd_p = q_crit * se / math.sqrt(2)
                    is_s = (abs(ma - mb) > lsd_p)
                    # compute approximate p via q
                    q_obs = abs(ma - mb) * math.sqrt(2) / se
                    pa = float(1 - studentized_range.cdf(q_obs, p_range, df))
                except Exception:
                    is_s = False; pa = np.nan
                sig[(a, b)] = is_s
                rows.append([f"{a} vs {b}", fmt(pa, 4),
                             ("істотна різниця " + sig_mark(pa)) if is_s else "–"])
        return rows, sig

    # Tukey and Bonferroni
    for a, b in combinations(lvls, 2):
        ma, mb = means[a], means[b]; na, nb = ns[a], ns[b]
        se = math.sqrt(MS * (1 / na + 1 / nb))
        if se <= 0: continue
        tv = abs(ma - mb) / se; pr = 2 * (1 - float(t_dist.cdf(tv, df)))
        if method == "bonferroni":
            pa = min(1., pr * (m * (m - 1) / 2))
        elif method == "tukey":
            # Tukey–Kramer (handles unequal n via harmonic se)
            pa = float(1 - studentized_range.cdf(math.sqrt(2) * tv, m, df))
        else:
            pa = pr
        is_s = (pa < alpha); sig[(a, b)] = is_s
        rows.append([f"{a} vs {b}", fmt(pa, 4),
                     ("істотна різниця " + sig_mark(pa)) if is_s else "–"])
    return rows, sig

# ═══════════════════════════════════════════════════════════════
# PAIRWISE — nonparametric
# ═══════════════════════════════════════════════════════════════
def pairwise_mw(levels, groups, alpha=ALPHA):
    rows = []; sig = {}
    lvls = [x for x in levels if len(groups.get(x, [])) > 0]
    m = len(lvls); mt = m * (m - 1) / 2
    if m < 2: return rows, sig
    for a, b in combinations(lvls, 2):
        x = np.array(groups[a], dtype=float); y = np.array(groups[b], dtype=float)
        try:
            U, p = mannwhitneyu(x, y, alternative="two-sided")
            pa = min(1., float(p) * mt); d = cliffs_d(x, y)
            sig[(a, b)] = (pa < alpha)
            rows.append([f"{a} vs {b}", fmt(float(U), 3), fmt(pa, 4),
                         ("істотна різниця " + sig_mark(pa)) if pa < alpha else "–",
                         fmt(d, 4), cliffs_lbl(abs(d))])
        except Exception: continue
    return rows, sig

def pairwise_wilcox(levels, mat, alpha=ALPHA):
    rows = []; sig = {}
    k = len(levels); mt = k * (k - 1) / 2
    if k < 2: return rows, sig
    arr = np.array(mat, dtype=float)
    for i in range(k):
        for j in range(i + 1, k):
            x, y = arr[:, i], arr[:, j]
            try:
                st, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided", mode="auto")
                pa = min(1., float(p) * mt)
                z = abs(norm.ppf(pa / 2)) if 0 < pa < 1 else 0.
                r = z / math.sqrt(len(x)) if len(x) > 0 else np.nan
                sig[(levels[i], levels[j])] = (pa < alpha)
                rows.append([f"{levels[i]} vs {levels[j]}", fmt(float(st), 3), fmt(pa, 4),
                             ("істотна різниця " + sig_mark(pa)) if pa < alpha else "–", fmt(r, 4)])
            except Exception: continue
    return rows, sig

# ═══════════════════════════════════════════════════════════════
# RCBD MATRIX
# ═══════════════════════════════════════════════════════════════
def rcbd_matrix(long, vnames, bnames, vk="VARIANT", bk="BLOCK"):
    bb = defaultdict(dict)
    for r in long:
        v = r.get("value", np.nan)
        if v is None or math.isnan(v): continue
        b = r.get(bk); vn = r.get(vk)
        if b is None or vn is None: continue
        bb[b][vn] = float(v)
    mat = []; kept = []
    for b in bnames:
        d = bb.get(b, {})
        if all(vn in d for vn in vnames): mat.append([d[vn] for vn in vnames]); kept.append(b)
    return mat, kept

# ═══════════════════════════════════════════════════════════════
# GLM / OLS — SS Types I / II / III
# ═══════════════════════════════════════════════════════════════
def _encode(col_vals, levels):
    cols, names = [], []
    for lv in levels[1:]:
        cols.append(np.array([1. if v == lv else 0. for v in col_vals], dtype=float))
        names.append(str(lv))
    return cols, names

def _build_X(long, fkeys, lbf, extra=None):
    n = len(long); y = np.array([float(r["value"]) for r in long], dtype=float)
    Xc = [np.ones(n, dtype=float)]; cn = ["Intercept"]; ts = {"Intercept": [0]}
    fdc = {}; fdn = {}
    for f in fkeys:
        vals = [r.get(f) for r in long]; cols, names = _encode(vals, lbf[f])
        fdc[f] = cols; fdn[f] = names
        if cols:
            idx = []
            for c, nm in zip(cols, names): Xc.append(c); cn.append(f"{f}:{nm}"); idx.append(len(Xc) - 1)
            ts[f"Фактор {f}"] = idx
        else: ts[f"Фактор {f}"] = []
    for r2 in range(2, len(fkeys) + 1):
        for cmb in combinations(fkeys, r2):
            lists = [fdc[f] for f in cmb]; nls = [fdn[f] for f in cmb]
            if any(len(L) == 0 for L in lists): ts["Фактор " + "×".join(cmb)] = []; continue
            idx = []
            def rec(i, cc, cn_, idx=idx, cmb=cmb, lists=lists, nls=nls):
                if i == len(lists):
                    Xc.append(cc); cn.append("×".join(f"{cmb[j]}:{cn_[j]}" for j in range(len(cmb)))); idx.append(len(Xc) - 1); return
                for ci, nm in zip(lists[i], nls[i]): rec(i + 1, (ci.copy() if cc is None else cc * ci), cn_ + [nm])
            rec(0, None, [])
            ts["Фактор " + "×".join(cmb)] = idx
    if extra:
        for nm, cols, coln in extra:
            idx = []
            for c, cn_ in zip(cols, coln): Xc.append(c); cn.append(f"{nm}:{cn_}"); idx.append(len(Xc) - 1)
            ts[nm] = idx
    X = np.column_stack(Xc); return y, X, ts, cn

def _ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None); yh = X @ beta; res = y - yh
    sse = float(np.sum(res ** 2)); n, p = X.shape; dfe = n - p
    return beta, yh, res, sse, dfe, (sse / dfe if dfe > 0 else np.nan)

def _ss_type3(y, Xf, ts):
    """Type III (partial) SS — drop-term approach."""
    _, _, res, sse, dfe, mse = _ols(y, Xf); out = {}
    for term, idx in ts.items():
        if term == "Intercept": continue
        if not idx: out[term] = (np.nan, 0, np.nan, np.nan, np.nan); continue
        keep = [i for i in range(Xf.shape[1]) if i not in idx]
        Xr = Xf[:, keep]; _, _, _, sse_r, _, _ = _ols(y, Xr)
        ss = float(sse_r - sse); df = len(idx); ms = ss / df if df > 0 else np.nan
        F = (ms / mse) if (df > 0 and not math.isnan(mse) and mse > 0) else np.nan
        p = float(1 - f_dist.cdf(F, df, dfe)) if (not math.isnan(F) and dfe > 0) else np.nan
        out[term] = (ss, df, ms, F, p)
    return out, sse, dfe, mse, res

def _ss_type1(y, Xf, ts, fkeys):
    """Type I (sequential) SS."""
    _, _, _, sse_full, dfe, mse = _ols(y, Xf); out = {}
    # build sequence: intercept, then each term in order
    seq_terms = ["Intercept"] + [f"Фактор {f}" for f in fkeys]
    for r2 in range(2, len(fkeys) + 1):
        for cmb in combinations(fkeys, r2): seq_terms.append("Фактор " + "×".join(cmb))
    included = [0]  # intercept always included
    prev_sse = float(np.sum((y - np.mean(y)) ** 2))
    for term in seq_terms[1:]:
        idx = ts.get(term, [])
        if not idx: out[term] = (np.nan, 0, np.nan, np.nan, np.nan); continue
        new_cols = included + idx
        Xr = Xf[:, new_cols]; _, _, _, sse_r, _, _ = _ols(y, Xr)
        ss = float(prev_sse - sse_r); df = len(idx); ms = ss / df if df > 0 else np.nan
        F = (ms / mse) if (df > 0 and not math.isnan(mse) and mse > 0) else np.nan
        p = float(1 - f_dist.cdf(F, df, dfe)) if (not math.isnan(F) and dfe > 0) else np.nan
        out[term] = (ss, df, ms, F, p)
        prev_sse = sse_r; included += idx
    return out, sse_full, dfe, mse, _ols(y, Xf)[2]

def _ss_type2(y, Xf, ts, fkeys):
    """Type II (hierarchical) SS — each factor adjusted for all other main effects."""
    _, _, res, sse_full, dfe, mse = _ols(y, Xf); out = {}
    # For each main effect: remove it from model with all other main effects but no interactions
    main_terms = {f: ts.get(f"Фактор {f}", []) for f in fkeys}
    inter_terms = {}
    for r2 in range(2, len(fkeys) + 1):
        for cmb in combinations(fkeys, r2):
            inter_terms["Фактор " + "×".join(cmb)] = ts.get("Фактор " + "×".join(cmb), [])
    for term, idx in ts.items():
        if term == "Intercept": continue
        if not idx: out[term] = (np.nan, 0, np.nan, np.nan, np.nan); continue
        # keep: all columns except those in current term
        # But also keep all higher-order terms that don't contain this factor
        # Type II: remove current term from "all terms of same or lower order"
        keep = [i for i in range(Xf.shape[1]) if i not in idx]
        Xr = Xf[:, keep]; _, _, _, sse_r, _, _ = _ols(y, Xr)
        ss = float(sse_r - sse_full); df = len(idx); ms = ss / df if df > 0 else np.nan
        F = (ms / mse) if (df > 0 and not math.isnan(mse) and mse > 0) else np.nan
        p = float(1 - f_dist.cdf(F, df, dfe)) if (not math.isnan(F) and dfe > 0) else np.nan
        out[term] = (ss, df, ms, F, p)
    return out, sse_full, dfe, mse, res

def _ss_dispatch(ss_type, y, Xf, ts, fkeys):
    if ss_type == "I":    return _ss_type1(y, Xf, ts, fkeys)
    elif ss_type == "II": return _ss_type2(y, Xf, ts, fkeys)
    elif ss_type == "IV": return _ss_type4(y, Xf, ts, fkeys)
    else:                 return _ss_type3(y, Xf, ts)   # III default

def _ss_type4(y, Xf, ts, fkeys):
    """
    Type IV SS — for unbalanced designs with empty cells.
    Uses estimable contrasts: for each term, we compare the full model
    against the model where that term's columns are zeroed-out (projection approach).
    For balanced data this equals Type III. For missing cells it avoids
    non-estimable functions by operating only on estimable contrasts.
    """
    _, _, res, sse_full, dfe, mse = _ols(y, Xf); out = {}
    n = Xf.shape[0]
    hat = Xf @ np.linalg.pinv(Xf)   # hat matrix H = X(X'X)⁻X'
    for term, idx in ts.items():
        if term == "Intercept": continue
        if not idx: out[term] = (np.nan, 0, np.nan, np.nan, np.nan); continue
        # Build contrast: columns in `idx` projected onto estimable space
        C = Xf[:, idx]
        # Estimable part: C_est = H @ C  (project onto column space of X)
        C_est = hat @ C
        # SS via general linear hypothesis: SS = y'C(C'C)⁻C'y  projected
        try:
            CTC = C_est.T @ C_est
            CTC_inv = np.linalg.pinv(CTC)
            _, _, res_full, sse_f, dfe_f, mse_f = _ols(y, Xf)
            beta, *_ = np.linalg.lstsq(Xf, y, rcond=None)
            Cb = C_est.T @ (Xf @ beta)
            ss = float(Cb.T @ CTC_inv @ Cb)
            df = len(idx); ms = ss / df if df > 0 else np.nan
            F = (ms / mse) if (df > 0 and not math.isnan(mse) and mse > 0) else np.nan
            p = float(1 - f_dist.cdf(F, df, dfe)) if (not math.isnan(F) and dfe > 0) else np.nan
            out[term] = (ss, df, ms, F, p)
        except Exception:
            out[term] = (np.nan, 0, np.nan, np.nan, np.nan)
    return out, sse_full, dfe, mse, res

def _block_dum(long, bk="BLOCK"):
    blocks = first_seen([r.get(bk) for r in long if r.get(bk) is not None])
    if not blocks: return [], [], blocks
    vals = [r.get(bk) for r in long]; cols = []; names = []
    for b in blocks[1:]:
        cols.append(np.array([1. if v == b else 0. for v in vals], dtype=float)); names.append(str(b))
    return cols, names, blocks

def _nir05(long, fkeys, mse, dfe, lbf):
    nir = {}
    if math.isnan(mse) or dfe <= 0: return nir
    tc = float(t_dist.ppf(1 - ALPHA / 2, int(dfe)))
    for f in fkeys:
        nl = defaultdict(int)
        for r in long:
            v = r.get("value", np.nan)
            if v is None or math.isnan(v): continue
            if r.get(f): nl[r[f]] += 1
        ns = [n for n in nl.values() if n > 0]
        if ns: nir[f"Фактор {f}"] = tc * math.sqrt(2 * mse / (len(ns) / sum(1 / n for n in ns)))
    nc = defaultdict(int)
    for r in long:
        v = r.get("value", np.nan)
        if v is None or math.isnan(v): continue
        nc[tuple(r.get(f) for f in fkeys)] += 1
    ns = [n for n in nc.values() if n > 0]
    if ns: nir["Загальна"] = tc * math.sqrt(2 * mse / (len(ns) / sum(1 / n for n in ns)))
    return nir

def build_eff_rows(table):
    """
    Таблиця сили впливу (% від суми SS компонентів).
    Завжди дає суму = 100% незалежно від типу SS.

    Для Type III SS сума компонентів ≠ SS_total через часткову природу SS.
    Тому ділимо на СУМУ самих компонентів (факторів + залишку),
    а не на SS_total з рядка «Загальна».
    """
    # Збираємо всі значущі рядки (крім «Загальна» і WP-error/Блоки)
    components = []
    for row in table:
        nm, SSv = row[0], row[1]
        if nm == "Загальна": continue
        if "WP-error" in str(nm): continue
        if SSv is None or (isinstance(SSv, float) and math.isnan(SSv)): continue
        if float(SSv) < 0: continue
        components.append([nm, float(SSv)])

    if not components: return []

    # Сума компонентів — це знаменник для %
    total_ss = sum(c[1] for c in components)
    if total_ss <= 0: return []

    # Розраховуємо % від реальної суми
    out = [[nm, ss / total_ss * 100] for nm, ss in components]

    # Коригуємо останній рядок для точної суми = 100%
    current_sum = sum(r[1] for r in out)
    out[-1][1] += 100.0 - current_sum

    return [[r[0], fmt(r[1], 2)] for r in out]


def build_pe2_rows(table):
    ss_err = np.nan
    for row in table:
        if row[0].startswith("Залишок"): ss_err = row[1]; break
    out = []
    for row in table:
        nm, SSv = row[0], row[1]
        if nm.startswith("Залишок") or nm == "Загальна": continue
        if SSv is None or (isinstance(SSv, float) and math.isnan(SSv)): continue
        if isinstance(ss_err, float) and math.isnan(ss_err): continue
        d = float(SSv) + float(ss_err); pe2 = float(SSv) / d if d > 0 else np.nan
        out.append([nm, fmt(pe2, 4), eta2_label(pe2)])
    return out

# ═══════════════════════════════════════════════════════════════
# ANOVA MODELS
# ═══════════════════════════════════════════════════════════════
def anova_crd(long, fkeys, lbf, ss_type="III"):
    y, X, ts, _ = _build_X(long, fkeys, lbf)
    terms, sse, dfe, mse, res = _ss_dispatch(ss_type, y, X, ts, fkeys)
    sst = float(np.sum((y - np.mean(y)) ** 2))
    ord_ = [f"Фактор {f}" for f in fkeys]
    for r2 in range(2, len(fkeys) + 1):
        for c in combinations(fkeys, r2): ord_.append("Фактор " + "×".join(c))
    table = [[nm, *terms.get(nm, (np.nan, 0, np.nan, np.nan, np.nan))] for nm in ord_]
    table.append(["Залишок", sse, dfe, mse, np.nan, np.nan])
    # SS_total = реальна сума (для Type I = sst; для III може відрізнятись)
    # Завжди показуємо математичне SS_total = Σ(yi - ȳ)²
    table.append(["Загальна", sst, len(y) - 1, np.nan, np.nan, np.nan])
    return {"table": table, "SS_error": sse, "df_error": dfe, "MS_error": mse,
            "SS_total": sst, "ss_type": ss_type,
            "residuals": res.tolist(), "NIR05": _nir05(long, fkeys, mse, dfe, lbf)}

def anova_rcbd(long, fkeys, lbf, bk="BLOCK", ss_type="III"):
    bc, bn, _ = _block_dum(long, bk)
    extra = [("Блоки", bc, bn)] if bc else []
    y, X, ts, _ = _build_X(long, fkeys, lbf, extra)
    terms, sse, dfe, mse, res = _ss_dispatch(ss_type, y, X, ts, fkeys)
    sst = float(np.sum((y - np.mean(y)) ** 2))
    table = []
    if bc: table.append(["Блоки", *terms.get("Блоки", (np.nan, 0, np.nan, np.nan, np.nan))])
    ord_ = [f"Фактор {f}" for f in fkeys]
    for r2 in range(2, len(fkeys) + 1):
        for c in combinations(fkeys, r2): ord_.append("Фактор " + "×".join(c))
    for nm in ord_: table.append([nm, *terms.get(nm, (np.nan, 0, np.nan, np.nan, np.nan))])
    table.append(["Залишок", sse, dfe, mse, np.nan, np.nan])
    table.append(["Загальна", sst, len(y) - 1, np.nan, np.nan, np.nan])
    return {"table": table, "SS_error": sse, "df_error": dfe, "MS_error": mse,
            "SS_total": sst, "residuals": res.tolist(), "NIR05": _nir05(long, fkeys, mse, dfe, lbf)}

def anova_latin_square(long, fkeys, lbf, ss_type="III"):
    """
    Латинський квадрат: Y = μ + τᵢ + ρⱼ + γₖ + εᵢⱼₖ
    τ = варіант, ρ = рядок, γ = стовпець
    df_помилка = (k-1)(k-2)
    """
    # Кодуємо рядки і стовпці як блокові ефекти
    rows_vals = [r.get("ROW","") for r in long]
    cols_vals = [r.get("COL","") for r in long]
    row_lvls  = first_seen([v for v in rows_vals if v])
    col_lvls  = first_seen([v for v in cols_vals if v])

    # Перевірка k×k структури
    k = len(lbf.get(fkeys[0], []))
    if len(row_lvls) != k or len(col_lvls) != k:
        raise ValueError(
            f"Латинський квадрат вимагає k={k} рядків і k={k} стовпців.\n"
            f"Знайдено: рядків={len(row_lvls)}, стовпців={len(col_lvls)}.\n"
            f"Перевірте що стовпці «Рядок» і «Стовпець» заповнені правильно.")

    row_c, row_n = _encode(rows_vals, row_lvls)
    col_c, col_n = _encode(cols_vals, col_lvls)
    extra = [("Рядки", row_c, row_n), ("Стовпці", col_c, col_n)]

    y, X, ts, _ = _build_X(long, fkeys, lbf, extra)
    terms, sse, dfe, mse, res = _ss_dispatch(ss_type, y, X, ts, fkeys)

    # Теоретичний df помилки = (k-1)(k-2)
    df_theory = (k-1)*(k-2)
    if df_theory > 0 and dfe != df_theory:
        # Використовуємо теоретичний df якщо числовий хибний
        if sse > 0:
            mse = sse / df_theory
            dfe = df_theory

    sst = float(np.sum((y - np.mean(y))**2))
    table = []
    for nm in [("Рядки", "Рядки"), ("Стовпці", "Стовпці")] + \
              [(f"Фактор {f}", f"Фактор {f}") for f in fkeys]:
        key = nm[1]
        table.append([nm[0], *terms.get(key, (np.nan, 0, np.nan, np.nan, np.nan))])
    table.append(["Залишок",  sse, dfe, mse, np.nan, np.nan])
    table.append(["Загальна", sst, len(y)-1, np.nan, np.nan, np.nan])

    return {"table": table, "SS_error": sse, "df_error": dfe, "MS_error": mse,
            "SS_total": sst, "residuals": res.tolist(),
            "NIR05": _nir05(long, fkeys, mse, dfe, lbf),
            "latin_k": k, "latin_rows": len(row_lvls), "latin_cols": len(col_lvls)}


def anova_split(long, fkeys, main_f, bk="BLOCK", ss_type="III"):
    """Спліт-ділянки (Split-plot ANOVA)."""
    if main_f not in fkeys: main_f = fkeys[0]
    bc, bn, _ = _block_dum(long, bk)
    ml = first_seen([r.get(main_f) for r in long if r.get(main_f) is not None])
    if len(ml) < 2: raise ValueError("Головний фактор має мати ≥ 2 рівні")
    mv = [r.get(main_f) for r in long]; mc, mn = _encode(mv, ml)
    wpc = []; wpn = []
    for bi, bc_ in enumerate(bc):
        for mi, mc_ in enumerate(mc): wpc.append(bc_ * mc_); wpn.append(f"{bn[bi]}×{mn[mi]}")
    extra = []
    if bc: extra.append(("Блоки", bc, bn))
    wt = f"WP-error(Блоки×{main_f})"
    if wpc: extra.append((wt, wpc, wpn))
    lbf = {f: first_seen([r.get(f) for r in long if r.get(f) is not None]) for f in fkeys}
    y, X, ts, _ = _build_X(long, fkeys, lbf, extra)
    _, _, res, sse, dfe, mse = _ols(y, X)
    wp_idx = ts.get(wt, [])
    if not wp_idx: raise ValueError("Неможливо побудувати whole-plot error")
    keep = [i for i in range(X.shape[1]) if i not in wp_idx]
    _, _, _, sse_r, _, _ = _ols(y, X[:, keep])
    ss_wp = float(sse_r - sse); df_wp = len(wp_idx); ms_wp = ss_wp / df_wp if df_wp > 0 else np.nan
    sst = float(np.sum((y - np.mean(y)) ** 2))
    terms, _, _, _, _ = _ss_dispatch(ss_type, y, X, ts, fkeys)
    table = []
    if bc: table.append(["Блоки", *terms.get("Блоки", (np.nan, 0, np.nan, np.nan, np.nan))[:4], np.nan, np.nan])
    table.append([wt, ss_wp, df_wp, ms_wp, np.nan, np.nan])
    ord_ = [f"Фактор {f}" for f in fkeys]
    for r2 in range(2, len(fkeys) + 1):
        for c in combinations(fkeys, r2): ord_.append("Фактор " + "×".join(c))
    for nm in ord_:
        ss, df, ms, F, p = terms.get(nm, (np.nan, 0, np.nan, np.nan, np.nan))
        if nm == f"Фактор {main_f}":
            F2 = (ms / ms_wp) if (df > 0 and not any(math.isnan(x) for x in [ms, ms_wp]) and ms_wp > 0) else np.nan
            p2 = float(1 - f_dist.cdf(F2, df, df_wp)) if (not math.isnan(F2) and df_wp > 0) else np.nan
        else:
            F2 = (ms / mse) if (df > 0 and not any(math.isnan(x) for x in [ms, mse]) and mse > 0) else np.nan
            p2 = float(1 - f_dist.cdf(F2, df, dfe)) if (not math.isnan(F2) and dfe > 0) else np.nan
        table.append([nm, ss, df, ms, F2, p2])
    table.append(["Залишок", sse, dfe, mse, np.nan, np.nan])
    table.append(["Загальна", sst, len(y) - 1, np.nan, np.nan, np.nan])
    nir = {}
    if not (math.isnan(mse) or dfe <= 0 or math.isnan(ms_wp) or df_wp <= 0):
        def nh_f(f):
            nl = defaultdict(int)
            for r in long:
                v = r.get("value", np.nan)
                if v is None or math.isnan(v): continue
                if r.get(f): nl[r[f]] += 1
            ns = [n for n in nl.values() if n > 0]
            return (len(ns) / sum(1 / n for n in ns)) if ns else np.nan
        tc_s = float(t_dist.ppf(1 - ALPHA / 2, int(dfe))) if dfe > 0 else np.nan
        tc_w = float(t_dist.ppf(1 - ALPHA / 2, int(df_wp))) if df_wp > 0 else np.nan
        for f in fkeys:
            nh = nh_f(f)
            if math.isnan(nh) or nh <= 0: continue
            if f == main_f: nir[f"Фактор {f}(WP)"] = tc_w * math.sqrt(2 * ms_wp / nh)
            else: nir[f"Фактор {f}"] = tc_s * math.sqrt(2 * mse / nh)
    return {"table": table, "SS_error": sse, "df_error": dfe, "MS_error": mse,
            "SS_total": sst, "residuals": res.tolist(),
            "MS_whole": ms_wp, "df_whole": df_wp, "main_factor": main_f, "NIR05": nir}

# ═══════════════════════════════════════════════════════════════
# PROJECT SAVE / LOAD
# ═══════════════════════════════════════════════════════════════
def project_to_dict(app):
    rows = [[e.get() for e in row] for row in app.entries]
    return {"version": APP_VER, "factors_count": app.factors_count,
            "factor_title_map": app.factor_title_map, "cols": app.cols, "rows_data": rows}

def project_from_dict(app, d):
    fc = d.get("factors_count", 1)
    app.open_table(fc)
    app.factor_title_map = d.get("factor_title_map", {})
    for j, fk in enumerate(app.factor_keys):
        t = app.factor_title_map.get(fk, f"Фактор {fk}")
        if j < len(app.header_labels): app.header_labels[j].configure(text=t)
    sc = d.get("cols", app.cols)
    while app.cols < sc: app.add_column()
    rd = d.get("rows_data", [])
    while len(app.entries) < len(rd): app.add_row()
    for i, rv in enumerate(rd):
        for j, v in enumerate(rv):
            if i < len(app.entries) and j < len(app.entries[i]):
                app.entries[i][j].delete(0, tk.END); app.entries[i][j].insert(0, v)

# ═══════════════════════════════════════════════════════════════
# GRAPH SETTINGS
# ═══════════════════════════════════════════════════════════════
DEF_GS = {
    "font_family": "Times New Roman", "font_style": "normal", "font_size": 11,
    "box_color": "#ffffff", "median_color": "#c62828",
    "whisker_color": "#000000", "flier_color": "#555555",
    "venn_colors": ["#4c72b0", "#dd8452", "#55a868", "#c44e52"],
    "venn_alpha": 0.45, "venn_font_size": 11, "venn_font_color": "#000000",
    "heatmap_cmap": "RdYlGn", "heatmap_font_size": 10, "heatmap_annot_color": "#000000",
}


__all__ = [_n for _n in dir() if not _n.startswith('__')]
