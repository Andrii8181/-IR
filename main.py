# main.py  — S.A.D. v2.0
# -*- coding: utf-8 -*-
"""
S.A.D. — Статистичний аналіз даних  v2.0
Залежності: pip install numpy scipy openpyxl matplotlib pillow
"""
import os, sys, math, json, io
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser
from tkinter.scrolledtext import ScrolledText
import tkinter.font as tkfont
from itertools import combinations
from collections import defaultdict
from datetime import datetime

from scipy.stats import (shapiro, kruskal, mannwhitneyu, friedmanchisquare,
                         wilcoxon, levene)
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
    import matplotlib.pyplot as _plt
    import matplotlib.patches as mpatches
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
APP_VER = "2.0"

# ── APA-style matplotlib defaults ────────────────────────────
if HAS_MPL:
    import matplotlib
    matplotlib.rcParams.update({
        'font.family':        'serif',
        'font.serif':         ['Times New Roman','Times','DejaVu Serif'],
        'font.size':          11,
        'axes.titlesize':     12,
        'axes.labelsize':     11,
        'xtick.labelsize':    10,
        'ytick.labelsize':    10,
        'axes.linewidth':     0.8,
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'figure.dpi':         100,
    })

# ── DPI awareness ────────────────────────────────────────────
try:
    import ctypes
    try:    ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try: ctypes.windll.user32.SetProcessDPIAware()
        except Exception: pass
except Exception: pass

# ── Icon ─────────────────────────────────────────────────────
def _find_icon():
    dirs=[os.getcwd()]
    try: dirs.insert(0,os.path.dirname(os.path.abspath(__file__)))
    except Exception: pass
    try:
        if hasattr(sys,"_MEIPASS"): dirs.append(sys._MEIPASS)
    except Exception: pass
    for d in dirs:
        p=os.path.join(d,"icon.ico")
        if os.path.exists(p): return p
    return None

def set_icon(win):
    ico=_find_icon()
    if not ico: return
    try: win.iconbitmap(ico)
    except Exception:
        try: win.iconbitmap(default=ico)
        except Exception: pass

# ── Clipboard PNG → Windows ───────────────────────────────────
def _copy_fig_to_clipboard(fig):
    if not (HAS_MPL and HAS_PIL): return False,"Потрібні matplotlib і Pillow"
    try:
        buf=io.BytesIO(); fig.savefig(buf,format="png",dpi=300,bbox_inches="tight"); buf.seek(0)
        pil=_PILImage.open(buf)
        ok,msg=_copy_pil_win(pil); buf.close(); return ok,msg
    except Exception as ex: return False,str(ex)

def _copy_pil_win(pil_img):
    try: import ctypes; from ctypes import wintypes
    except Exception: return False,"ctypes недоступний"
    if sys.platform!="win32": return False,"Лише для Windows"
    if pil_img is None: return False,"Немає зображення"
    try:
        buf=io.BytesIO(); pil_img.convert("RGB").save(buf,"BMP"); bmp=buf.getvalue()
        if len(bmp)<=14: return False,"BMP помилка"
        data=bmp[14:]
    except Exception as ex: return False,str(ex)
    u32=ctypes.WinDLL("user32",use_last_error=True)
    k32=ctypes.WinDLL("kernel32",use_last_error=True)
    u32.OpenClipboard.argtypes=[wintypes.HWND];         u32.OpenClipboard.restype=wintypes.BOOL
    u32.CloseClipboard.argtypes=[];                     u32.CloseClipboard.restype=wintypes.BOOL
    u32.EmptyClipboard.argtypes=[];                     u32.EmptyClipboard.restype=wintypes.BOOL
    u32.SetClipboardData.argtypes=[wintypes.UINT,wintypes.HANDLE]; u32.SetClipboardData.restype=wintypes.HANDLE
    k32.GlobalAlloc.argtypes=[wintypes.UINT,ctypes.c_size_t]; k32.GlobalAlloc.restype=wintypes.HGLOBAL
    k32.GlobalLock.argtypes=[wintypes.HGLOBAL];         k32.GlobalLock.restype=wintypes.LPVOID
    k32.GlobalUnlock.argtypes=[wintypes.HGLOBAL];       k32.GlobalUnlock.restype=wintypes.BOOL
    k32.GlobalFree.argtypes=[wintypes.HGLOBAL];         k32.GlobalFree.restype=wintypes.HGLOBAL
    if not u32.OpenClipboard(None): return False,f"OpenClipboard err {ctypes.get_last_error()}"
    try:
        u32.EmptyClipboard()
        hg=k32.GlobalAlloc(0x0042,len(data))
        if not hg: return False,"GlobalAlloc failed"
        pg=k32.GlobalLock(hg)
        if not pg: k32.GlobalFree(hg); return False,"GlobalLock failed"
        try: ctypes.memmove(pg,data,len(data))
        finally: k32.GlobalUnlock(hg)
        if not u32.SetClipboardData(8,hg): k32.GlobalFree(hg); return False,"SetClipboardData failed"
        return True,""
    finally: u32.CloseClipboard()

# ═══════════════════════════════════════════════════════════════
# SMALL STAT HELPERS
# ═══════════════════════════════════════════════════════════════
def sig_mark(p):
    if p is None or (isinstance(p,float) and math.isnan(p)): return ""
    return "**" if p<0.01 else ("*" if p<0.05 else "")

def norm_txt(p):
    if p is None or (isinstance(p,float) and math.isnan(p)): return "н/д"
    return "нормальний розподіл" if p>0.05 else "ненормальний розподіл"

def fmt(x,nd=3):
    if x is None or (isinstance(x,float) and math.isnan(x)): return ""
    try: return f"{float(x):.{nd}f}"
    except: return ""

def first_seen(seq):
    seen,out=set(),[]
    for x in seq:
        if x not in seen: seen.add(x); out.append(x)
    return out

def center_win(win):
    win.update_idletasks()
    w,h=win.winfo_width(),win.winfo_height()
    sw,sh=win.winfo_screenwidth(),win.winfo_screenheight()
    win.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

def median_q(arr):
    if not arr: return np.nan,np.nan,np.nan
    a=np.array(arr,dtype=float); a=a[~np.isnan(a)]
    if len(a)==0: return np.nan,np.nan,np.nan
    return float(np.median(a)),float(np.percentile(a,25)),float(np.percentile(a,75))

def cv_vals(vals):
    a=np.array(vals,dtype=float); a=a[~np.isnan(a)]
    if len(a)<2: return np.nan
    m=float(np.mean(a)); return np.nan if m==0 else float(np.std(a,ddof=1)/m*100)

def cv_means(means):
    v=[float(x) for x in means if x is not None and not (isinstance(x,float) and math.isnan(x))]
    if len(v)<2: return np.nan
    m=float(np.mean(v)); return np.nan if m==0 else float(np.std(v,ddof=1)/m*100)

def eta2_label(pe2):
    if pe2 is None or math.isnan(pe2): return ""
    if pe2<0.01: return "дуже слабкий"
    if pe2<0.06: return "слабкий"
    if pe2<0.14: return "середній"
    return "сильний"

def eps2_kw(H,n,k):
    if any(x is None for x in [H,n,k]) or math.isnan(H) or n<=k or k<2: return np.nan
    return float((H-k+1)/(n-k))

def kendalls_w(chisq,nb,kt):
    if any(x is None for x in [chisq,nb,kt]) or math.isnan(chisq) or nb<=0 or kt<=1: return np.nan
    return float(chisq/(nb*(kt-1)))

def cliffs_d(x,y):
    x=np.array(x,dtype=float); y=np.array(y,dtype=float)
    x=x[~np.isnan(x)]; y=y[~np.isnan(y)]
    nx,ny=len(x),len(y)
    if nx==0 or ny==0: return np.nan
    gt=int(np.sum(x[:,None]>y[None,:])); lt=int(np.sum(x[:,None]<y[None,:]))
    return float((gt-lt)/(nx*ny))

def cliffs_lbl(d):
    if d is None or math.isnan(d): return ""
    if d<0.147: return "дуже слабкий"
    if d<0.33:  return "слабкий"
    if d<0.474: return "середній"
    return "сильний"

# ═══════════════════════════════════════════════════════════════
# REPORT — Treeview-based tables (no column drift)
# ═══════════════════════════════════════════════════════════════
def make_tv_table(parent, headers, rows, min_col_px=90):
    """Return a Frame containing a fixed-column Treeview table."""
    frame=tk.Frame(parent, bd=1, relief=tk.SUNKEN)
    vsb=ttk.Scrollbar(frame,orient="vertical")
    hsb=ttk.Scrollbar(frame,orient="horizontal")
    tv=ttk.Treeview(frame,columns=headers,show="headings",
                    yscrollcommand=vsb.set,xscrollcommand=hsb.set,
                    height=min(len(rows)+1,22))
    vsb.config(command=tv.yview); hsb.config(command=tv.xview)
    vsb.pack(side=tk.RIGHT,fill=tk.Y)
    hsb.pack(side=tk.BOTTOM,fill=tk.X)
    tv.pack(fill=tk.BOTH,expand=True)
    fnt=tkfont.Font(family="Times New Roman",size=11)
    for i,h in enumerate(headers):
        cw=max(fnt.measure(str(h))+24, min_col_px,
               max((fnt.measure(str(r[i] if i<len(r) else ""))+20) for r in rows) if rows else min_col_px)
        tv.heading(h,text=str(h),anchor="center")
        tv.column(h,width=cw,minwidth=50,anchor="center",stretch=True)
    for row in rows:
        tv.insert("","end",values=[("" if v is None else str(v)) for v in row])
    style=ttk.Style()
    style.configure("Treeview",font=("Times New Roman",11),rowheight=22)
    style.configure("Treeview.Heading",font=("Times New Roman",11,"bold"))
    return frame, tv

# ═══════════════════════════════════════════════════════════════
# DATA HELPERS
# ═══════════════════════════════════════════════════════════════
def groups_by(long,keys):
    g=defaultdict(list)
    for r in long:
        v=r.get("value",np.nan)
        if v is None or math.isnan(v): continue
        g[tuple(r.get(x) for x in keys)].append(float(v))
    return g

def vstats(long,fkeys):
    vals=defaultdict(list)
    for r in long:
        v=r.get("value",np.nan)
        if v is None or math.isnan(v): continue
        vals[tuple(r.get(k) for k in fkeys)].append(float(v))
    out={}
    for k,a in vals.items():
        n=len(a); m=float(np.mean(a)) if n else np.nan
        sd=float(np.std(a,ddof=1)) if n>=2 else (0. if n==1 else np.nan)
        out[k]=(m,sd,n)
    return out

def mean_ranks(long,keyfn):
    vals=[]; ks=[]
    for r in long:
        v=r.get("value",np.nan)
        if v is None or math.isnan(v): continue
        vals.append(float(v)); ks.append(keyfn(r))
    if not vals: return {}
    order=np.argsort(vals); sv=np.array(vals)[order]
    ranks=np.empty(len(vals),dtype=float)
    i=0
    while i<len(sv):
        j=i
        while j<len(sv) and sv[j]==sv[i]: j+=1
        ar=(i+1+j)/2.; ranks[order[i:j]]=ar; i=j
    agg=defaultdict(list)
    for k,rk in zip(ks,ranks): agg[k].append(float(rk))
    return {k:float(np.mean(v)) for k,v in agg.items()}

# ═══════════════════════════════════════════════════════════════
# CLD
# ═══════════════════════════════════════════════════════════════
def cld(levels_order,means_dict,sig_matrix):
    valid=[l for l in levels_order if not math.isnan(means_dict.get(l,np.nan))]
    if not valid: return {l:"" for l in levels_order}
    sl=sorted(valid,key=lambda z:means_dict[z],reverse=True)
    def sig(a,b): return bool(sig_matrix.get((a,b),False) or sig_matrix.get((b,a),False))
    groups=[]
    for lv in sl:
        compat=[gi for gi,g in enumerate(groups) if all(not sig(lv,o) for o in g)]
        if not compat: groups.append({lv})
        else:
            for gi in compat: groups[gi].add(lv)
    def shared(a,b): return any(a in g and b in g for g in groups)
    for i in range(len(sl)):
        for j in range(i+1,len(sl)):
            a,b=sl[i],sl[j]
            if sig(a,b) or shared(a,b): continue
            ng={a,b}
            for c in sl:
                if c in ng: continue
                if not sig(c,a) and not sig(c,b) and all(not sig(c,x) for x in ng): ng.add(c)
            groups.append(ng)
    uniq=[]
    for g in groups:
        if not any(g==h for h in uniq): uniq.append(g)
    cleaned=[g for g in uniq if not any(g<h for h in uniq)]
    alpha_="abcdefghijklmnopqrstuvwxyz"
    mapping={lv:[] for lv in sl}
    for gi,g in enumerate(cleaned):
        lt=alpha_[gi] if gi<len(alpha_) else f"g{gi}"
        for lv in g: mapping[lv].append(lt)
    return {lv:"".join(sorted(mapping.get(lv,[]))) for lv in levels_order}

# ═══════════════════════════════════════════════════════════════
# LEVENE TEST
# ═══════════════════════════════════════════════════════════════
def levene_test(groups_dict):
    arrs=[np.array(v,dtype=float) for v in groups_dict.values() if len(v)>0]
    if len(arrs)<2: return np.nan,np.nan
    try:
        stat,p=levene(*arrs,center='median')
        return float(stat),float(p)
    except Exception: return np.nan,np.nan

# ═══════════════════════════════════════════════════════════════
# PAIRWISE — parametric
# ═══════════════════════════════════════════════════════════════
def lsd_sig(levels,means,ns,MS,df,alpha=ALPHA):
    sig={}
    if MS is None or df is None or math.isnan(MS) or math.isnan(df): return sig
    df=int(df)
    if df<=0: return sig
    tc=float(t_dist.ppf(1-alpha/2,df))
    for a,b in combinations(levels,2):
        ma,mb=means.get(a,np.nan),means.get(b,np.nan)
        na,nb=ns.get(a,0),ns.get(b,0)
        if any(math.isnan(x) for x in [ma,mb]) or na<=0 or nb<=0: continue
        se=math.sqrt(MS*(1/na+1/nb))
        sig[(a,b)]=(abs(ma-mb)>tc*se)
    return sig

def pairwise_param(levels,means,ns,MS,df,method,alpha=ALPHA):
    rows=[]; sig={}
    if MS is None or df is None or math.isnan(MS) or math.isnan(df): return rows,sig
    df=int(df)
    if df<=0: return rows,sig
    lvls=[x for x in levels if not math.isnan(means.get(x,np.nan)) and ns.get(x,0)>0]
    m=len(lvls)
    if m<2: return rows,sig
    for a,b in combinations(lvls,2):
        ma,mb=means[a],means[b]; na,nb=ns[a],ns[b]
        se=math.sqrt(MS*(1/na+1/nb))
        if se<=0: continue
        tv=abs(ma-mb)/se; pr=2*(1-float(t_dist.cdf(tv,df)))
        if method=="bonferroni":       pa=min(1.,pr*(m*(m-1)/2))
        elif method in ("tukey","duncan"): pa=float(1-studentized_range.cdf(math.sqrt(2)*tv,m,df))
        else: pa=pr
        is_s=(pa<alpha); sig[(a,b)]=is_s
        rows.append([f"{a} vs {b}",fmt(pa,4),
                     ("істотна різниця "+sig_mark(pa)) if is_s else "-"])
    return rows,sig

# ═══════════════════════════════════════════════════════════════
# PAIRWISE — nonparametric
# ═══════════════════════════════════════════════════════════════
def pairwise_mw(levels,groups,alpha=ALPHA):
    rows=[]; sig={}
    lvls=[x for x in levels if len(groups.get(x,[]))>0]
    m=len(lvls); mt=m*(m-1)/2
    if m<2: return rows,sig
    for a,b in combinations(lvls,2):
        x=np.array(groups[a],dtype=float); y=np.array(groups[b],dtype=float)
        try:
            U,p=mannwhitneyu(x,y,alternative="two-sided")
            pa=min(1.,float(p)*mt); d=cliffs_d(x,y)
            sig[(a,b)]=(pa<alpha)
            rows.append([f"{a} vs {b}",fmt(float(U),3),fmt(pa,4),
                         ("істотна різниця "+sig_mark(pa)) if pa<alpha else "-",
                         fmt(d,4),cliffs_lbl(abs(d))])
        except Exception: continue
    return rows,sig

def pairwise_wilcox(levels,mat,alpha=ALPHA):
    rows=[]; sig={}
    k=len(levels); mt=k*(k-1)/2
    if k<2: return rows,sig
    arr=np.array(mat,dtype=float)
    for i in range(k):
        for j in range(i+1,k):
            x,y=arr[:,i],arr[:,j]
            try:
                st,p=wilcoxon(x,y,zero_method="wilcox",alternative="two-sided",mode="auto")
                pa=min(1.,float(p)*mt)
                z=abs(norm.ppf(pa/2)) if 0<pa<1 else 0.
                r=z/math.sqrt(len(x)) if len(x)>0 else np.nan
                sig[(levels[i],levels[j])]=(pa<alpha)
                rows.append([f"{levels[i]} vs {levels[j]}",fmt(float(st),3),fmt(pa,4),
                             ("істотна різниця "+sig_mark(pa)) if pa<alpha else "-",fmt(r,4)])
            except Exception: continue
    return rows,sig

# ═══════════════════════════════════════════════════════════════
# RCBD MATRIX
# ═══════════════════════════════════════════════════════════════
def rcbd_matrix(long,vnames,bnames,vk="VARIANT",bk="BLOCK"):
    bb=defaultdict(dict)
    for r in long:
        v=r.get("value",np.nan)
        if v is None or math.isnan(v): continue
        b=r.get(bk); vn=r.get(vk)
        if b is None or vn is None: continue
        bb[b][vn]=float(v)
    mat=[]; kept=[]
    for b in bnames:
        d=bb.get(b,{})
        if all(vn in d for vn in vnames): mat.append([d[vn] for vn in vnames]); kept.append(b)
    return mat,kept

# ═══════════════════════════════════════════════════════════════
# GLM / OLS
# ═══════════════════════════════════════════════════════════════
def _encode(col_vals,levels):
    cols,names=[],[]
    for lv in levels[1:]:
        cols.append(np.array([1. if v==lv else 0. for v in col_vals],dtype=float))
        names.append(str(lv))
    return cols,names

def _build_X(long,fkeys,lbf,extra=None):
    n=len(long); y=np.array([float(r["value"]) for r in long],dtype=float)
    Xc=[np.ones(n,dtype=float)]; cn=["Intercept"]; ts={"Intercept":[0]}
    fdc={}; fdn={}
    for f in fkeys:
        vals=[r.get(f) for r in long]; cols,names=_encode(vals,lbf[f])
        fdc[f]=cols; fdn[f]=names
        if cols:
            idx=[]
            for c,nm in zip(cols,names): Xc.append(c); cn.append(f"{f}:{nm}"); idx.append(len(Xc)-1)
            ts[f"Фактор {f}"]=idx
        else: ts[f"Фактор {f}"]=[]
    for r2 in range(2,len(fkeys)+1):
        for cmb in combinations(fkeys,r2):
            lists=[fdc[f] for f in cmb]; nls=[fdn[f] for f in cmb]
            if any(len(L)==0 for L in lists): ts["Фактор "+"×".join(cmb)]=[]; continue
            idx=[]
            def rec(i,cc,cn_,idx=idx,cmb=cmb,lists=lists,nls=nls):
                if i==len(lists):
                    Xc.append(cc); cn.append("×".join(f"{cmb[j]}:{cn_[j]}" for j in range(len(cmb)))); idx.append(len(Xc)-1); return
                for ci,nm in zip(lists[i],nls[i]): rec(i+1,(ci.copy() if cc is None else cc*ci),cn_+[nm])
            rec(0,None,[])
            ts["Фактор "+"×".join(cmb)]=idx
    if extra:
        for nm,cols,coln in extra:
            idx=[]
            for c,cn_ in zip(cols,coln): Xc.append(c); cn.append(f"{nm}:{cn_}"); idx.append(len(Xc)-1)
            ts[nm]=idx
    X=np.column_stack(Xc); return y,X,ts,cn

def _ols(y,X):
    beta,*_=np.linalg.lstsq(X,y,rcond=None); yh=X@beta; res=y-yh; sse=float(np.sum(res**2))
    n,p=X.shape; dfe=n-p; return beta,yh,res,sse,dfe,(sse/dfe if dfe>0 else np.nan)

def _ss(y,Xf,ts):
    _,_,res,sse,dfe,mse=_ols(y,Xf); out={}
    for term,idx in ts.items():
        if term=="Intercept": continue
        if not idx: out[term]=(np.nan,0,np.nan,np.nan,np.nan); continue
        keep=[i for i in range(Xf.shape[1]) if i not in idx]
        Xr=Xf[:,keep]; _,_,_,sse_r,_,_=_ols(y,Xr)
        ss=float(sse_r-sse); df=len(idx); ms=ss/df if df>0 else np.nan
        F=(ms/mse) if (df>0 and not math.isnan(mse) and mse>0) else np.nan
        p=float(1-f_dist.cdf(F,df,dfe)) if (not math.isnan(F) and dfe>0) else np.nan
        out[term]=(ss,df,ms,F,p)
    return out,sse,dfe,mse,res

def _block_dum(long,bk="BLOCK"):
    blocks=first_seen([r.get(bk) for r in long if r.get(bk) is not None])
    if not blocks: return [],[],blocks
    vals=[r.get(bk) for r in long]; cols=[]; names=[]
    for b in blocks[1:]:
        cols.append(np.array([1. if v==b else 0. for v in vals],dtype=float)); names.append(str(b))
    return cols,names,blocks

def _nir05(long,fkeys,mse,dfe,lbf):
    nir={}
    if math.isnan(mse) or dfe<=0: return nir
    tc=float(t_dist.ppf(1-ALPHA/2,int(dfe)))
    for f in fkeys:
        nl=defaultdict(int)
        for r in long:
            v=r.get("value",np.nan)
            if v is None or math.isnan(v): continue
            if r.get(f): nl[r[f]]+=1
        ns=[n for n in nl.values() if n>0]
        if ns: nir[f"Фактор {f}"]=tc*math.sqrt(2*mse/(len(ns)/sum(1/n for n in ns)))
    nc=defaultdict(int)
    for r in long:
        v=r.get("value",np.nan)
        if v is None or math.isnan(v): continue
        nc[tuple(r.get(f) for f in fkeys)]+=1
    ns=[n for n in nc.values() if n>0]
    if ns: nir["Загальна"]=tc*math.sqrt(2*mse/(len(ns)/sum(1/n for n in ns)))
    return nir

def build_eff_rows(table):
    ss_tot=0.
    for row in table:
        if row[0]=="Загальна" and row[1] is not None and not (isinstance(row[1],float) and math.isnan(row[1])):
            ss_tot=float(row[1]); break
    if ss_tot<=0:
        ss_tot=sum(float(r[1]) for r in table if r[1] is not None
                   and not (isinstance(r[1],float) and math.isnan(r[1])) and not r[0].startswith("Залишок"))
    out=[]
    for row in table:
        nm,SSv=row[0],row[1]
        if nm.startswith("Залишок") or nm=="Загальна": continue
        if SSv is None or (isinstance(SSv,float) and math.isnan(SSv)): continue
        out.append([nm,fmt((float(SSv)/ss_tot*100) if ss_tot>0 else np.nan,2)])
    return out

def build_pe2_rows(table):
    ss_err=np.nan
    for row in table:
        if row[0].startswith("Залишок"): ss_err=row[1]; break
    out=[]
    for row in table:
        nm,SSv=row[0],row[1]
        if nm.startswith("Залишок") or nm=="Загальна": continue
        if SSv is None or (isinstance(SSv,float) and math.isnan(SSv)): continue
        if isinstance(ss_err,float) and math.isnan(ss_err): continue
        d=float(SSv)+float(ss_err); pe2=float(SSv)/d if d>0 else np.nan
        out.append([nm,fmt(pe2,4),eta2_label(pe2)])
    return out

# ═══════════════════════════════════════════════════════════════
# ANOVA MODELS
# ═══════════════════════════════════════════════════════════════
def anova_crd(long,fkeys,lbf):
    y,X,ts,_=_build_X(long,fkeys,lbf)
    terms,sse,dfe,mse,res=_ss(y,X,ts)
    sst=float(np.sum((y-np.mean(y))**2))
    ord_=[f"Фактор {f}" for f in fkeys]
    for r2 in range(2,len(fkeys)+1):
        for c in combinations(fkeys,r2): ord_.append("Фактор "+"×".join(c))
    table=[[nm,*terms.get(nm,(np.nan,0,np.nan,np.nan,np.nan))] for nm in ord_]
    table.append(["Залишок",sse,dfe,mse,np.nan,np.nan])
    table.append(["Загальна",sst,len(y)-1,np.nan,np.nan,np.nan])
    return {"table":table,"SS_error":sse,"df_error":dfe,"MS_error":mse,
            "SS_total":sst,"residuals":res.tolist(),"NIR05":_nir05(long,fkeys,mse,dfe,lbf)}

def anova_rcbd(long,fkeys,lbf,bk="BLOCK"):
    bc,bn,_=_block_dum(long,bk)
    extra=[("Блоки",bc,bn)] if bc else []
    y,X,ts,_=_build_X(long,fkeys,lbf,extra)
    terms,sse,dfe,mse,res=_ss(y,X,ts)
    sst=float(np.sum((y-np.mean(y))**2))
    table=[]
    if bc: table.append(["Блоки",*terms.get("Блоки",(np.nan,0,np.nan,np.nan,np.nan))])
    ord_=[f"Фактор {f}" for f in fkeys]
    for r2 in range(2,len(fkeys)+1):
        for c in combinations(fkeys,r2): ord_.append("Фактор "+"×".join(c))
    for nm in ord_: table.append([nm,*terms.get(nm,(np.nan,0,np.nan,np.nan,np.nan))])
    table.append(["Залишок",sse,dfe,mse,np.nan,np.nan])
    table.append(["Загальна",sst,len(y)-1,np.nan,np.nan,np.nan])
    return {"table":table,"SS_error":sse,"df_error":dfe,"MS_error":mse,
            "SS_total":sst,"residuals":res.tolist(),"NIR05":_nir05(long,fkeys,mse,dfe,lbf)}

def anova_split(long,fkeys,main_f,bk="BLOCK"):
    if main_f not in fkeys: main_f=fkeys[0]
    bc,bn,_=_block_dum(long,bk)
    ml=first_seen([r.get(main_f) for r in long if r.get(main_f) is not None])
    if len(ml)<2: raise ValueError("Головний фактор має мати ≥ 2 рівні")
    mv=[r.get(main_f) for r in long]; mc,mn=_encode(mv,ml)
    wpc=[]; wpn=[]
    for bi,bc_ in enumerate(bc):
        for mi,mc_ in enumerate(mc): wpc.append(bc_*mc_); wpn.append(f"{bn[bi]}×{mn[mi]}")
    extra=[]
    if bc: extra.append(("Блоки",bc,bn))
    wt=f"WP-error(Блоки×{main_f})"
    if wpc: extra.append((wt,wpc,wpn))
    lbf={f:first_seen([r.get(f) for r in long if r.get(f) is not None]) for f in fkeys}
    y,X,ts,_=_build_X(long,fkeys,lbf,extra)
    _,_,res,sse,dfe,mse=_ols(y,X)
    wp_idx=ts.get(wt,[])
    if not wp_idx: raise ValueError("Неможливо побудувати whole-plot error")
    keep=[i for i in range(X.shape[1]) if i not in wp_idx]
    _,_,_,sse_r,_,_=_ols(y,X[:,keep])
    ss_wp=float(sse_r-sse); df_wp=len(wp_idx); ms_wp=ss_wp/df_wp if df_wp>0 else np.nan
    sst=float(np.sum((y-np.mean(y))**2))
    terms,_,_,_,_=_ss(y,X,ts)
    table=[]
    if bc: table.append(["Блоки",*terms.get("Блоки",(np.nan,0,np.nan,np.nan,np.nan))[:4],np.nan,np.nan])
    table.append([wt,ss_wp,df_wp,ms_wp,np.nan,np.nan])
    ord_=[f"Фактор {f}" for f in fkeys]
    for r2 in range(2,len(fkeys)+1):
        for c in combinations(fkeys,r2): ord_.append("Фактор "+"×".join(c))
    for nm in ord_:
        ss,df,ms,F,p=terms.get(nm,(np.nan,0,np.nan,np.nan,np.nan))
        if nm==f"Фактор {main_f}":
            F2=(ms/ms_wp) if (df>0 and not any(math.isnan(x) for x in [ms,ms_wp]) and ms_wp>0) else np.nan
            p2=float(1-f_dist.cdf(F2,df,df_wp)) if (not math.isnan(F2) and df_wp>0) else np.nan
        else:
            F2=(ms/mse) if (df>0 and not any(math.isnan(x) for x in [ms,mse]) and mse>0) else np.nan
            p2=float(1-f_dist.cdf(F2,df,dfe)) if (not math.isnan(F2) and dfe>0) else np.nan
        table.append([nm,ss,df,ms,F2,p2])
    table.append(["Залишок",sse,dfe,mse,np.nan,np.nan])
    table.append(["Загальна",sst,len(y)-1,np.nan,np.nan,np.nan])
    nir={}
    if not (math.isnan(mse) or dfe<=0 or math.isnan(ms_wp) or df_wp<=0):
        def nh_f(f):
            nl=defaultdict(int)
            for r in long:
                v=r.get("value",np.nan)
                if v is None or math.isnan(v): continue
                if r.get(f): nl[r[f]]+=1
            ns=[n for n in nl.values() if n>0]
            return (len(ns)/sum(1/n for n in ns)) if ns else np.nan
        tc_s=float(t_dist.ppf(1-ALPHA/2,int(dfe))) if dfe>0 else np.nan
        tc_w=float(t_dist.ppf(1-ALPHA/2,int(df_wp))) if df_wp>0 else np.nan
        for f in fkeys:
            nh=nh_f(f)
            if math.isnan(nh) or nh<=0: continue
            if f==main_f: nir[f"Фактор {f}(WP)"]=tc_w*math.sqrt(2*ms_wp/nh)
            else:          nir[f"Фактор {f}"]=tc_s*math.sqrt(2*mse/nh)
    return {"table":table,"SS_error":sse,"df_error":dfe,"MS_error":mse,
            "SS_total":sst,"residuals":res.tolist(),
            "MS_whole":ms_wp,"df_whole":df_wp,"main_factor":main_f,"NIR05":nir}

# ═══════════════════════════════════════════════════════════════
# PROJECT SAVE / LOAD  (.sadp = JSON)
# ═══════════════════════════════════════════════════════════════
def project_to_dict(app):
    rows=[[e.get() for e in row] for row in app.entries]
    return {"version":APP_VER,"factors_count":app.factors_count,
            "factor_title_map":app.factor_title_map,"cols":app.cols,"rows_data":rows}

def project_from_dict(app,d):
    fc=d.get("factors_count",1)
    app.open_table(fc)
    app.factor_title_map=d.get("factor_title_map",{})
    for j,fk in enumerate(app.factor_keys):
        t=app.factor_title_map.get(fk,f"Фактор {fk}")
        if j<len(app.header_labels): app.header_labels[j].configure(text=t)
    sc=d.get("cols",app.cols)
    while app.cols<sc: app.add_column()
    rd=d.get("rows_data",[])
    while len(app.entries)<len(rd): app.add_row()
    for i,rv in enumerate(rd):
        for j,v in enumerate(rv):
            if i<len(app.entries) and j<len(app.entries[i]):
                app.entries[i][j].delete(0,tk.END); app.entries[i][j].insert(0,v)

# ═══════════════════════════════════════════════════════════════
# GRAPH SETTINGS  (APA defaults)
# ═══════════════════════════════════════════════════════════════
DEF_GS = {
    "font_family":"Times New Roman","font_style":"normal","font_size":11,
    "box_color":"#ffffff","median_color":"#c62828",
    "whisker_color":"#000000","flier_color":"#555555",
    "venn_colors":["#4c72b0","#dd8452","#55a868","#c44e52"],
    "venn_alpha":0.45,"venn_font_size":11,"venn_font_color":"#000000",
}

# ═══════════════════════════════════════════════════════════════
# FONT HELPERS
# ═══════════════════════════════════════════════════════════════
def mpl_font_props(gs):
    """Return matplotlib FontProperties-like dict from graph settings."""
    fs=gs["font_style"]
    weight="bold" if "bold" in fs else "normal"
    style="italic" if "italic" in fs else "normal"
    return {"fontfamily":gs["font_family"],"fontsize":gs["font_size"],
            "fontweight":weight,"fontstyle":style}

def fit_font(texts,family="Times New Roman",start=13,min_s=9,target=155):
    f=tkfont.Font(family=family,size=start); sz=start
    while sz>min_s:
        if max(f.measure(t) for t in texts)<=target: break
        sz-=1; f.configure(size=sz)
    return f

# ═══════════════════════════════════════════════════════════════
# GRAPH SETTINGS DIALOG
# ═══════════════════════════════════════════════════════════════
class GraphSettingsDlg(tk.Toplevel):
    FONTS=["Times New Roman","Arial","Calibri","Georgia","Verdana","Courier New"]
    STYLES=["normal","bold","italic","bold italic"]

    def __init__(self, parent, gs: dict):
        super().__init__(parent)
        self.title("Налаштування графіків")
        self.resizable(False,False)
        set_icon(self)
        self.gs=dict(gs); self.result=None

        nb=ttk.Notebook(self)
        nb.pack(fill=tk.BOTH,expand=True,padx=8,pady=8)

        # ── Boxplot tab ──
        bp=tk.Frame(nb,padx=12,pady=10); nb.add(bp,text="Boxplot")
        self._var_ff=tk.StringVar(value=gs["font_family"])
        self._var_fs=tk.StringVar(value=gs["font_style"])
        self._var_fz=tk.IntVar(value=gs["font_size"])
        self._col_box=gs["box_color"]; self._col_med=gs["median_color"]
        self._col_wh=gs["whisker_color"]; self._col_fl=gs["flier_color"]

        r=0
        tk.Label(bp,text="Шрифт:").grid(row=r,column=0,sticky="w",pady=4)
        ttk.Combobox(bp,textvariable=self._var_ff,values=self.FONTS,state="readonly",width=22).grid(row=r,column=1,sticky="w",padx=6)
        r+=1
        tk.Label(bp,text="Стиль:").grid(row=r,column=0,sticky="w",pady=4)
        ttk.Combobox(bp,textvariable=self._var_fs,values=self.STYLES,state="readonly",width=22).grid(row=r,column=1,sticky="w",padx=6)
        r+=1
        tk.Label(bp,text="Розмір:").grid(row=r,column=0,sticky="w",pady=4)
        tk.Spinbox(bp,from_=7,to=28,textvariable=self._var_fz,width=6).grid(row=r,column=1,sticky="w",padx=6)
        r+=1
        for lbl,attr in [("Колір коробки:","_col_box"),("Колір медіани:","_col_med"),
                          ("Колір вусів:","_col_wh"),("Колір викидів:","_col_fl")]:
            tk.Label(bp,text=lbl).grid(row=r,column=0,sticky="w",pady=4)
            btn=tk.Button(bp,width=6,relief=tk.SUNKEN,
                          command=lambda a=attr: self._pick_color(a,a_widget))
            btn.configure(bg=getattr(self,attr))
            btn.grid(row=r,column=1,sticky="w",padx=6)
            a_widget=btn; r+=1

        # ── Venn tab ──
        vf=tk.Frame(nb,padx=12,pady=10); nb.add(vf,text="Діаграма Венна")
        self._var_vff=tk.StringVar(value=gs["font_family"])
        self._var_vfz=tk.IntVar(value=gs["venn_font_size"])
        self._var_valpha=tk.DoubleVar(value=gs["venn_alpha"])
        self._venn_fc=gs["venn_font_color"]
        self._venn_cols=list(gs["venn_colors"])

        r=0
        tk.Label(vf,text="Шрифт:").grid(row=r,column=0,sticky="w",pady=4)
        ttk.Combobox(vf,textvariable=self._var_vff,values=self.FONTS,state="readonly",width=22).grid(row=r,column=1,sticky="w",padx=6)
        r+=1
        tk.Label(vf,text="Розмір:").grid(row=r,column=0,sticky="w",pady=4)
        tk.Spinbox(vf,from_=7,to=28,textvariable=self._var_vfz,width=6).grid(row=r,column=1,sticky="w",padx=6)
        r+=1
        tk.Label(vf,text="Прозорість (0–1):").grid(row=r,column=0,sticky="w",pady=4)
        tk.Scale(vf,from_=0.,to=1.,resolution=0.05,orient="horizontal",
                 variable=self._valpha_var() if False else self._var_valpha,length=180).grid(row=r,column=1,sticky="w",padx=6)
        r+=1
        tk.Label(vf,text="Колір тексту:").grid(row=r,column=0,sticky="w",pady=4)
        self._vfc_btn=tk.Button(vf,width=6,relief=tk.SUNKEN,bg=self._venn_fc,
                                command=self._pick_venn_fc)
        self._vfc_btn.grid(row=r,column=1,sticky="w",padx=6); r+=1
        tk.Label(vf,text="Кольори кіл:").grid(row=r,column=0,sticky="w",pady=4)
        self._vci_btns=[]
        bf=tk.Frame(vf); bf.grid(row=r,column=1,sticky="w",padx=6)
        for idx in range(4):
            b=tk.Button(bf,width=4,relief=tk.SUNKEN,bg=self._venn_cols[idx],
                        command=lambda i=idx: self._pick_venn_ci(i))
            b.pack(side=tk.LEFT,padx=2); self._vci_btns.append(b)
        r+=1

        # ── Buttons ──
        bf2=tk.Frame(self); bf2.pack(fill=tk.X,padx=10,pady=(0,10))
        tk.Button(bf2,text="OK",width=10,command=self._ok).pack(side=tk.LEFT,padx=4)
        tk.Button(bf2,text="Скасувати",width=12,command=self.destroy).pack(side=tk.LEFT)
        tk.Button(bf2,text="За замовч.",width=14,command=self._reset).pack(side=tk.RIGHT,padx=4)

        self.update_idletasks(); center_win(self)
        self.grab_set()

    def _valpha_var(self): pass  # unused placeholder

    def _pick_color(self,attr,btn_ref=None):
        cur=getattr(self,attr)
        c=colorchooser.askcolor(color=cur,parent=self,title="Виберіть колір")
        if c and c[1]:
            setattr(self,attr,c[1])
            # find button by re-scanning
            for w in self.winfo_children():
                for ww in self._all_widgets(w):
                    if isinstance(ww,tk.Button) and ww.cget("bg")==cur and ww is not self._vfc_btn:
                        ww.configure(bg=c[1])

    def _all_widgets(self,w):
        yield w
        for c in w.winfo_children():
            yield from self._all_widgets(c)

    def _pick_venn_fc(self):
        c=colorchooser.askcolor(color=self._venn_fc,parent=self,title="Колір тексту")
        if c and c[1]: self._venn_fc=c[1]; self._vfc_btn.configure(bg=c[1])

    def _pick_venn_ci(self,idx):
        c=colorchooser.askcolor(color=self._venn_cols[idx],parent=self,title=f"Колір кола {idx+1}")
        if c and c[1]: self._venn_cols[idx]=c[1]; self._vci_btns[idx].configure(bg=c[1])

    def _reset(self):
        import importlib
        # just restore defaults visually is complex; notify user
        messagebox.showinfo("За замовчуванням","Закрийте і відкрийте графік — APA стиль буде застосовано.")

    def _ok(self):
        self.result={
            "font_family":self._var_ff.get(),"font_style":self._var_fs.get(),
            "font_size":self._var_fz.get(),
            "box_color":self._col_box,"median_color":self._col_med,
            "whisker_color":self._col_wh,"flier_color":self._col_fl,
            "venn_colors":list(self._venn_cols),"venn_alpha":float(self._var_valpha.get()),
            "venn_font_size":self._var_vfz.get(),"venn_font_color":self._venn_fc,
        }
        self.destroy()

# ═══════════════════════════════════════════════════════════════
# VENN  (custom, lightweight — no matplotlib-venn dep needed)
# ═══════════════════════════════════════════════════════════════
def draw_venn(ax, labels, values, colors, alpha, font_size, font_color, font_family):
    """
    Draw a proportional Venn-like diagram (circles) for 1–4 factors.
    `values`: list of (label, percent) for each factor.
    """
    n=len(values)
    ax.set_aspect("equal"); ax.axis("off")
    if n==0: return

    import matplotlib.patches as mpatches

    # layout centres for up to 4 circles
    layouts={
        1:[(0,0)],
        2:[(-0.35,0),(0.35,0)],
        3:[(0,0.35),(-0.35,-0.2),(0.35,-0.2)],
        4:[(-0.35,0.35),(0.35,0.35),(-0.35,-0.35),(0.35,-0.35)],
    }
    centres=layouts.get(n,layouts[4])[:n]
    radius=0.42

    for i,(cx,cy) in enumerate(centres):
        circle=mpatches.Circle((cx,cy),radius,
                               fc=colors[i % len(colors)],alpha=alpha,
                               ec="#555555",lw=1.0)
        ax.add_patch(circle)
        nm,pct=values[i]
        ax.text(cx,cy+0.02,f"{nm}\n{fmt(pct,1)}%",
                ha="center",va="center",
                fontsize=font_size,color=font_color,fontfamily=font_family,
                fontweight="bold",linespacing=1.4)

    ax.set_xlim(-1,1); ax.set_ylim(-1,1)
    ax.set_title("Сила впливу факторів",
                 fontsize=font_size+1,fontfamily=font_family,pad=8)

# ═══════════════════════════════════════════════════════════════
# GUI — SADTk
# ═══════════════════════════════════════════════════════════════
class SADTk:
    SEL_BG="#cce5ff"; SEL_ANC="#99ccff"; ACT_BG="#fff3c4"

    def __init__(self, root):
        self.root=root
        root.title("S.A.D. — Статистичний аналіз даних")
        root.geometry("1000x560"); set_icon(root)
        root.option_add("*Font",("Times New Roman",14))
        root.option_add("*Foreground","#000000")
        try: ttk.Style().theme_use("clam")
        except Exception: pass

        mf=tk.Frame(root,bg="white"); mf.pack(expand=True,fill=tk.BOTH)
        tk.Label(mf,text="S.A.D. — Статистичний аналіз даних",
                 font=("Times New Roman",20,"bold"),fg="#000000",bg="white").pack(pady=18)

        bf=tk.Frame(mf,bg="white"); bf.pack(pady=10)
        for i,(txt,fc) in enumerate([("Однофакторний аналіз",1),("Двофакторний аналіз",2),
                                      ("Трифакторний аналіз",3),("Чотирифакторний аналіз",4)]):
            tk.Button(bf,text=txt,width=22,height=2,command=lambda f=fc: self.open_table(f)
                      ).grid(row=i//2,column=i%2,padx=10,pady=8)

        btm=tk.Frame(mf,bg="white"); btm.pack(pady=6)
        tk.Button(btm,text="📂 Відкрити проект",width=20,command=self.load_project).pack(side=tk.LEFT,padx=8)
        tk.Button(btm,text="💾 Зберегти проект",width=20,command=self.save_project).pack(side=tk.LEFT,padx=8)

        tk.Label(mf,text="Виберіть тип аналізу → Введіть дані → Натисніть «Аналіз даних»",
                 fg="#000000",bg="white").pack(pady=8)

        self.table_win=None; self.report_win=None; self.graph_win=None
        self._graph_fig_bp=None; self._graph_fig_vn=None
        self._active_cell=None; self._active_prev=None
        # selection
        self._sel_anchor=None; self._sel_cells=set(); self._sel_orig={}
        # fill drag
        self._fill_drag=False; self._fill_rows=[]; self._fill_cols=[]
        self.factor_title_map={}
        self.graph_settings=dict(DEF_GS)
        self._current_project_path=None

    # ── factor titles ─────────────────────────────────────────
    def ftitle(self,fk): return self.factor_title_map.get(fk,f"Фактор {fk}")
    def _set_ftitle(self,fk,t): self.factor_title_map[fk]=t.strip() or f"Фактор {fk}"

    # ── project save/load ─────────────────────────────────────
    def save_project(self):
        if not hasattr(self,"entries") or not self.entries:
            messagebox.showwarning("Проект","Спочатку відкрийте таблицю та введіть дані."); return
        path=filedialog.asksaveasfilename(
            parent=self.root,title="Зберегти проект",
            defaultextension=".sadp",
            filetypes=[("SAD проект","*.sadp"),("JSON","*.json"),("Усі файли","*.*")])
        if not path: return
        try:
            d=project_to_dict(self)
            with open(path,"w",encoding="utf-8") as f: json.dump(d,f,ensure_ascii=False,indent=2)
            self._current_project_path=path
            messagebox.showinfo("Збережено",f"Проект збережено:\n{path}")
        except Exception as ex: messagebox.showerror("Помилка",str(ex))

    def load_project(self):
        path=filedialog.askopenfilename(
            parent=self.root,title="Відкрити проект",
            filetypes=[("SAD проект","*.sadp"),("JSON","*.json"),("Усі файли","*.*")])
        if not path: return
        try:
            with open(path,"r",encoding="utf-8") as f: d=json.load(f)
            project_from_dict(self,d)
            self._current_project_path=path
            messagebox.showinfo("Відкрито",f"Проект відкрито:\n{path}")
        except Exception as ex: messagebox.showerror("Помилка",str(ex))

    # ── selection helpers ─────────────────────────────────────
    def _clear_sel(self):
        for (r,c) in list(self._sel_cells): self._restore_bg(r,c)
        self._sel_cells.clear(); self._sel_anchor=None; self._sel_orig.clear()

    def _restore_bg(self,r,c):
        try: self.entries[r][c].configure(bg=self._sel_orig.get((r,c),"white"))
        except Exception: pass

    def _apply_sel(self,cells):
        for (r,c) in cells:
            try:
                e=self.entries[r][c]
                if (r,c) not in self._sel_orig: self._sel_orig[(r,c)]=e.cget("bg")
                e.configure(bg=self.SEL_ANC if (r,c)==self._sel_anchor else self.SEL_BG)
            except Exception: pass

    def _sel_range(self,r1,c1,r2,c2):
        prev=set(self._sel_cells)
        new={(r,c) for r in range(min(r1,r2),max(r1,r2)+1)
             for c in range(min(c1,c2),max(c1,c2)+1)
             if r<len(self.entries) and c<len(self.entries[r])}
        for rc in prev-new: self._restore_bg(*rc)
        self._apply_sel(new-prev)
        if self._sel_anchor in new:
            try: self.entries[self._sel_anchor[0]][self._sel_anchor[1]].configure(bg=self.SEL_ANC)
            except Exception: pass
        self._sel_cells=new

    def _sel_bounds(self):
        if not self._sel_cells: return None
        rs=[r for r,c in self._sel_cells]; cs=[c for r,c in self._sel_cells]
        return min(rs),min(cs),max(rs),max(cs)

    def _near_br(self,w,mg=6):
        try:
            px=w.winfo_pointerx(); py=w.winfo_pointery()
            x0=w.winfo_rootx(); y0=w.winfo_rooty()
            ww=w.winfo_width(); hh=w.winfo_height()
            return (x0+ww-mg<=px<=x0+ww) and (y0+hh-mg<=py<=y0+hh)
        except Exception: return False

    def _sel_handle_cell(self):
        b=self._sel_bounds()
        if b is None: return None
        r,c=b[2],b[3]
        try: return self.entries[r][c]
        except Exception: return None

    # ── active cell ───────────────────────────────────────────
    def _set_active(self,w):
        if self._active_cell is w: return
        if isinstance(self._active_cell,tk.Entry) and self._active_prev:
            try: self._active_cell.configure(**self._active_prev)
            except Exception: pass
        self._active_cell=w
        if isinstance(w,tk.Entry):
            self._active_prev={"bg":w.cget("bg"),"highlightthickness":int(w.cget("highlightthickness")),
                               "highlightbackground":w.cget("highlightbackground"),
                               "highlightcolor":w.cget("highlightcolor"),
                               "relief":w.cget("relief"),"bd":int(w.cget("bd")) if str(w.cget("bd")).isdigit() else 1}
            try: w.configure(bg=self.ACT_BG,highlightthickness=3,highlightbackground="#c62828",highlightcolor="#c62828",relief=tk.SOLID,bd=1)
            except Exception: pass

    # ── cell binding ──────────────────────────────────────────
    def bind_cell(self,e):
        e.bind("<Return>",         self._on_enter)
        e.bind("<Up>",             self._on_arrow)
        e.bind("<Down>",           self._on_arrow)
        e.bind("<Left>",           self._on_arrow)
        e.bind("<Right>",          self._on_arrow)
        e.bind("<Control-c>",      self._on_copy_sel)
        e.bind("<Control-C>",      self._on_copy_sel)
        e.bind("<Control-v>",      self._on_paste)
        e.bind("<Control-V>",      self._on_paste)
        e.bind("<FocusIn>",        lambda ev: self._set_active(ev.widget))
        e.bind("<ButtonPress-1>",  self._on_press)
        e.bind("<B1-Motion>",      self._on_drag)
        e.bind("<ButtonRelease-1>",self._on_release)
        e.bind("<Motion>",         self._on_motion)
        e.bind("<Leave>",          lambda ev: ev.widget.configure(cursor=""))
        e.bind("<Shift-ButtonPress-1>",self._on_shift_click)

    # ── mouse events ──────────────────────────────────────────
    def _on_motion(self,event):
        w=event.widget
        if not isinstance(w,tk.Entry): return
        pos=self._pos(w)
        if not pos: return
        r,c=pos
        if c>=self.factors_count: w.configure(cursor=""); return
        # show crosshair near fill-handle of selection OR own bottom-right
        if self._sel_cells and self._sel_handle_cell() is w and self._near_br(w):
            w.configure(cursor="crosshair")
        elif not self._sel_cells and self._near_br(w):
            w.configure(cursor="crosshair")
        else:
            w.configure(cursor="")

    def _on_press(self,event):
        w=event.widget
        if not isinstance(w,tk.Entry): return
        pos=self._pos(w)
        if not pos: return
        r,c=pos
        # fill-handle?
        if c<self.factors_count:
            if self._sel_cells and self._sel_handle_cell() is w and self._near_br(w):
                self._start_fill(use_sel=True); return "break"
            if not self._sel_cells and self._near_br(w):
                self._clear_sel()
                self._sel_anchor=(r,c); self._sel_cells={(r,c)}
                self._sel_orig[(r,c)]=w.cget("bg"); self._apply_sel({(r,c)})
                self._start_fill(use_sel=False); return "break"
        self._fill_drag=False
        self._clear_sel()
        self._sel_anchor=(r,c); self._sel_cells={(r,c)}
        self._sel_orig[(r,c)]=w.cget("bg"); self._apply_sel({(r,c)})
        w.focus_set()

    def _on_shift_click(self,event):
        w=event.widget; pos=self._pos(w)
        if not pos or self._sel_anchor is None: return
        ar,ac=self._sel_anchor; r,c=pos
        self._sel_range(ar,ac,r,c); return "break"

    def _on_drag(self,event):
        w=event.widget
        if not isinstance(w,tk.Entry): return
        if self._fill_drag: self._do_fill(event); return "break"
        if self._sel_anchor is None: return
        ar,ac=self._sel_anchor
        pos=self._pos(w)
        if pos: r,c=pos
        else:
            py=w.winfo_pointery(); px=w.winfo_pointerx(); r,c=ar,ac
            for ri in range(len(self.entries)):
                for ci in range(len(self.entries[ri])):
                    cell=self.entries[ri][ci]
                    x0=cell.winfo_rootx(); y0=cell.winfo_rooty()
                    if x0<=px<=x0+cell.winfo_width() and y0<=py<=y0+cell.winfo_height():
                        r,c=ri,ci; break
        self._sel_range(ar,ac,r,c)

    def _on_release(self,event):
        if self._fill_drag:
            self._fill_drag=False; self._fill_rows=[]; self._fill_cols=[]; return "break"

    # ── fill drag ─────────────────────────────────────────────
    def _start_fill(self,use_sel):
        self._fill_drag=True
        if use_sel and self._sel_cells:
            b=self._sel_bounds()
            if b is None: self._fill_drag=False; return
            self._fill_rows=list(range(b[0],b[2]+1))
            self._fill_cols=list(range(b[1],b[3]+1))
        elif self._sel_anchor:
            self._fill_rows=[self._sel_anchor[0]]
            self._fill_cols=[self._sel_anchor[1]]
        else:
            self._fill_drag=False

    def _do_fill(self,event):
        w=event.widget
        if not isinstance(w,tk.Entry) or not self._fill_rows or not self._fill_cols: return
        last_src=self._fill_rows[-1]
        py=w.winfo_pointery(); target=last_src
        for rr in range(last_src,len(self.entries)):
            cell=self.entries[rr][self._fill_cols[0]]
            y0=cell.winfo_rooty()
            if y0<=py<=y0+cell.winfo_height(): target=rr; break
        else:
            if py>self.entries[-1][self._fill_cols[0]].winfo_rooty():
                target=len(self.entries)
        if target<=last_src: return
        n_src=len(self._fill_rows)
        dst=last_src+1
        while dst<=target:
            while dst>=len(self.entries): self.add_row()
            src_r=self._fill_rows[(dst-last_src-1)%n_src]
            for c in self._fill_cols:
                if c>=self.factors_count: break
                self.entries[dst][c].delete(0,tk.END)
                self.entries[dst][c].insert(0,self.entries[src_r][c].get())
            dst+=1
        self.inner.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    # ── copy selection (Ctrl+C) ───────────────────────────────
    def _on_copy_sel(self,event=None):
        """Copy selected rectangle to clipboard as tab-separated text."""
        if not self._sel_cells:
            # fallback: copy focused cell
            w=event.widget if event else self.table_win.focus_get()
            if isinstance(w,tk.Entry):
                try:
                    sel=w.get("sel.first","sel.last")
                except Exception:
                    sel=w.get()
                self.table_win.clipboard_clear(); self.table_win.clipboard_append(sel)
            return "break"
        b=self._sel_bounds()
        if b is None: return "break"
        r1,c1,r2,c2=b
        lines=[]
        for r in range(r1,r2+1):
            row=[]
            for c in range(c1,c2+1):
                try: row.append(self.entries[r][c].get())
                except Exception: row.append("")
            lines.append("\t".join(row))
        text="\n".join(lines)
        self.table_win.clipboard_clear(); self.table_win.clipboard_append(text)
        return "break"

    # ── navigation ────────────────────────────────────────────
    def _on_enter(self,event=None):
        pos=self._pos(event.widget)
        if not pos: return "break"
        i,j=pos; ni=i+1
        if ni>=len(self.entries): self.add_row()
        self.entries[ni][j].focus_set(); self.entries[ni][j].icursor(tk.END)
        return "break"

    def _on_arrow(self,event=None):
        pos=self._pos(event.widget)
        if not pos: return "break"
        i,j=pos
        if event.keysym=="Up":    i=max(0,i-1)
        elif event.keysym=="Down":  i=min(len(self.entries)-1,i+1)
        elif event.keysym=="Left":  j=max(0,j-1)
        elif event.keysym=="Right": j=min(len(self.entries[i])-1,j+1)
        self.entries[i][j].focus_set(); self.entries[i][j].icursor(tk.END)
        return "break"

    def _on_paste(self,event=None):
        widget=event.widget if event else self.table_win.focus_get()
        if not isinstance(widget,tk.Entry): return "break"
        try: data=self.table_win.clipboard_get()
        except Exception: return "break"
        rows=[r for r in data.splitlines() if r!=""]
        pos=self._pos(widget)
        if not pos: return "break"
        r0,c0=pos
        for ir,rt in enumerate(rows):
            cols=rt.split("\t")
            for jc,val in enumerate(cols):
                rr=r0+ir; cc=c0+jc
                while rr>=len(self.entries): self.add_row()
                if cc>=self.cols: continue
                self.entries[rr][cc].delete(0,tk.END); self.entries[rr][cc].insert(0,val)
        return "break"

    def _pos(self,widget):
        for i,row in enumerate(self.entries):
            for j,cell in enumerate(row):
                if cell is widget: return i,j
        return None

    # ── entry factory ─────────────────────────────────────────
    def _mk_entry(self,parent):
        return tk.Entry(parent,width=COL_W,fg="#000000",
                        highlightthickness=1,highlightbackground="#c0c0c0",highlightcolor="#c0c0c0")

    # ── rename factor ─────────────────────────────────────────
    def rename_factor(self,col):
        if col<0 or col>=self.factors_count: return
        fk=self.factor_keys[col]; old=self.ftitle(fk)
        dlg=tk.Toplevel(self.table_win or self.root)
        dlg.title("Перейменування"); dlg.resizable(False,False); set_icon(dlg)
        frm=tk.Frame(dlg,padx=14,pady=12); frm.pack()
        tk.Label(frm,text=f"Назва для {fk}:").grid(row=0,column=0,sticky="w")
        e=tk.Entry(frm,width=36,fg="#000000"); e.grid(row=1,column=0,pady=6)
        e.insert(0,old); e.select_range(0,"end"); e.focus_set()
        def ok():
            new=e.get().strip()
            if not new: messagebox.showwarning("","Назва не може бути порожньою."); return
            self._set_ftitle(fk,new)
            if col<len(self.header_labels): self.header_labels[col].configure(text=new)
            dlg.destroy()
        bf=tk.Frame(frm); bf.grid(row=2,column=0,sticky="w",pady=(8,0))
        tk.Button(bf,text="OK",width=10,command=ok).pack(side=tk.LEFT,padx=(0,6))
        tk.Button(bf,text="Скасувати",width=12,command=dlg.destroy).pack(side=tk.LEFT)
        dlg.update_idletasks(); center_win(dlg); dlg.bind("<Return>",lambda e:ok()); dlg.grab_set()

    # ── design help ───────────────────────────────────────────
    def show_design_help(self):
        w=tk.Toplevel(self.root); w.title("Пояснення дизайнів"); w.resizable(False,False); set_icon(w)
        frm=tk.Frame(w,padx=16,pady=14); frm.pack(fill=tk.BOTH,expand=True)
        txt=("CRD (повна рандомізація)\n• Усі варіанти розміщені випадково без блоків.\n\n"
             "RCBD (блочна рандомізація)\n• Є блоки; всередині блоку — всі варіанти.\n\n"
             "Split-plot (спліт-плот)\n• Є головний фактор і підплощі.\n"
             "• Для головного фактора використовується більша помилка (whole-plot error).")
        t=tk.Text(frm,width=60,height=12,wrap="word"); t.insert("1.0",txt); t.configure(state="disabled"); t.pack()
        tk.Button(frm,text="OK",width=10,command=w.destroy).pack(pady=(10,0))
        w.update_idletasks(); center_win(w); w.grab_set()

    # ── about ─────────────────────────────────────────────────
    def show_about(self):
        messagebox.showinfo("Розробник",
            f"S.A.D. — Статистичний аналіз даних  v{APP_VER}\n"
            "Розробник: Чаплоуцький Андрій Миколайович\n"
            "Уманський національний університет")

    # ══════════════════════════════════════════════════════════
    # OPEN TABLE
    # ══════════════════════════════════════════════════════════
    def open_table(self,fc):
        if self.table_win and tk.Toplevel.winfo_exists(self.table_win):
            self.table_win.destroy()
        self.factors_count=fc
        self.factor_keys=["A","B","C","D"][:fc]
        for fk in self.factor_keys:
            if fk not in self.factor_title_map: self._set_ftitle(fk,f"Фактор {fk}")
        self.table_win=tk.Toplevel(self.root)
        self.table_win.title(f"S.A.D. — {fc}-факторний аналіз")
        self.table_win.geometry("1280x720"); set_icon(self.table_win)

        self.repeat_count=6
        self.column_names=[self.ftitle(fk) for fk in self.factor_keys]+[f"Повт.{i+1}" for i in range(self.repeat_count)]
        ctl=tk.Frame(self.table_win,padx=6,pady=6); ctl.pack(fill=tk.X)

        btn_texts=["Додати рядок","Видалити рядок","Додати стовпчик","Видалити стовпчик",
                   "Вставити з буфера","Завантажити Excel","Аналіз даних","Розробник"]
        bf_=fit_font(btn_texts,start=14,min_s=9,target=150)
        bw,bh,px,py_=16,1,3,2

        tk.Button(ctl,text="Додати рядок",width=bw,height=bh,font=bf_,command=self.add_row).pack(side=tk.LEFT,padx=px,pady=py_)
        tk.Button(ctl,text="Видалити рядок",width=bw,height=bh,font=bf_,command=self.delete_row).pack(side=tk.LEFT,padx=px,pady=py_)
        tk.Button(ctl,text="Додати стовпчик",width=bw,height=bh,font=bf_,command=self.add_column).pack(side=tk.LEFT,padx=(10,px),pady=py_)
        tk.Button(ctl,text="Видалити стовпчик",width=bw,height=bh,font=bf_,command=self.delete_column).pack(side=tk.LEFT,padx=px,pady=py_)

        tk.Button(ctl,text="Зберегти проект",width=bw,height=bh,font=bf_,
                  command=self.save_project).pack(side=tk.LEFT,padx=(10,px),pady=py_)

        tk.Button(ctl,text="Вставити з буфера",width=bw+2,height=bh,font=bf_,
                  command=self._paste_from_focus).pack(side=tk.LEFT,padx=(6,px),pady=py_)
        xbtn=tk.Button(ctl,text="Завантажити Excel",width=bw+2,height=bh,font=bf_,
                       bg="#1a6b1a",fg="white",command=self.load_excel)
        xbtn.pack(side=tk.LEFT,padx=px,pady=py_)
        if not HAS_OPENPYXL: xbtn.configure(state="disabled",text="Excel (openpyxl!)")
        tk.Button(ctl,text="Аналіз даних",width=bw,height=bh,font=bf_,
                  bg="#c62828",fg="white",command=self.analyze).pack(side=tk.LEFT,padx=(10,px),pady=py_)
        tk.Button(ctl,text="Розробник",width=bw,height=bh,font=bf_,
                  command=self.show_about).pack(side=tk.RIGHT,padx=px,pady=py_)

        self.canvas=tk.Canvas(self.table_win); self.canvas.pack(side=tk.LEFT,fill=tk.BOTH,expand=True)
        sb=ttk.Scrollbar(self.table_win,orient="vertical",command=self.canvas.yview)
        sb.pack(side=tk.RIGHT,fill=tk.Y); self.canvas.configure(yscrollcommand=sb.set)
        self.inner=tk.Frame(self.canvas); self.canvas.create_window((0,0),window=self.inner,anchor="nw")

        self.rows=12; self.cols=len(self.column_names)
        self.entries=[]; self.header_labels=[]
        for j,nm in enumerate(self.column_names):
            lbl=tk.Label(self.inner,text=nm,relief=tk.RIDGE,width=COL_W,bg="#f0f0f0",fg="#000000")
            lbl.grid(row=0,column=j,padx=2,pady=2,sticky="nsew")
            self.header_labels.append(lbl)
            if j<self.factors_count: lbl.bind("<Double-Button-1>",lambda e,c=j:self.rename_factor(c))
        for i in range(self.rows):
            row_e=[]
            for j in range(self.cols):
                e=self._mk_entry(self.inner); e.grid(row=i+1,column=j,padx=2,pady=2)
                self.bind_cell(e); row_e.append(e)
            self.entries.append(row_e)
        self.inner.update_idletasks(); self.canvas.config(scrollregion=self.canvas.bbox("all"))
        self.entries[0][0].focus_set()
        self.table_win.bind("<Control-v>",self._on_paste)
        self.table_win.bind("<Control-V>",self._on_paste)

    # ── table editing ─────────────────────────────────────────
    def add_row(self):
        i=len(self.entries); row_e=[]
        for j in range(self.cols):
            e=self._mk_entry(self.inner); e.grid(row=i+1,column=j,padx=2,pady=2)
            self.bind_cell(e); row_e.append(e)
        self.entries.append(row_e); self.rows+=1
        self.inner.update_idletasks(); self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def delete_row(self):
        if not self.entries: return
        for e in self.entries.pop(): e.destroy()
        self.rows-=1; self.inner.update_idletasks(); self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def add_column(self):
        self.cols+=1; ci=self.cols-1
        nm=f"Повт.{ci-self.factors_count+1}"
        lbl=tk.Label(self.inner,text=nm,relief=tk.RIDGE,width=COL_W,bg="#f0f0f0",fg="#000000")
        lbl.grid(row=0,column=ci,padx=2,pady=2,sticky="nsew"); self.header_labels.append(lbl)
        for i,row in enumerate(self.entries):
            e=self._mk_entry(self.inner); e.grid(row=i+1,column=ci,padx=2,pady=2)
            self.bind_cell(e); row.append(e)
        self.inner.update_idletasks(); self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def delete_column(self):
        if self.cols<=self.factors_count+1: return
        self.header_labels.pop().destroy()
        for row in self.entries: row.pop().destroy()
        self.cols-=1; self.inner.update_idletasks(); self.canvas.config(scrollregion=self.canvas.bbox("all"))

    # ── paste from focused cell ───────────────────────────────
    def _paste_from_focus(self):
        w=self.table_win.focus_get()
        if isinstance(w,tk.Entry):
            class _E: widget=w
            self._on_paste(_E())

    # ── load Excel ────────────────────────────────────────────
    def load_excel(self):
        if not HAS_OPENPYXL: messagebox.showerror("","Встановіть openpyxl: pip install openpyxl"); return
        path=filedialog.askopenfilename(parent=self.table_win,title="Відкрити Excel",
                                        filetypes=[("Excel","*.xlsx *.xlsm *.xls"),("Усі","*.*")])
        if not path: return
        try:
            wb=openpyxl.load_workbook(path,data_only=True,read_only=True)
            snames=wb.sheetnames
        except Exception as ex: messagebox.showerror("Помилка",str(ex)); return

        sn=snames[0]
        if len(snames)>1: sn=self._ask_sheet(snames); wb.close()
        if sn is None: return

        try:
            wb2=openpyxl.load_workbook(path,data_only=True,read_only=True)
            ws=wb2[sn]
            raw=[[cell for cell in row] for row in ws.iter_rows(values_only=True)]
            wb2.close()
        except Exception as ex: messagebox.showerror("",str(ex)); return

        while raw and all(v is None or str(v).strip()=="" for v in raw[-1]): raw.pop()
        if not raw: messagebox.showwarning("","Аркуш порожній."); return
        nc=max(len(r) for r in raw); nr=len(raw)
        while len(self.entries)<nr: self.add_row()
        while self.cols<nc: self.add_column()
        for i,row in enumerate(raw):
            for j,v in enumerate(row):
                if j>=self.cols: break
                cv="" if v is None else str(v).replace(",",".")
                try: float(cv)
                except ValueError: cv="" if v is None else str(v)
                self.entries[i][j].delete(0,tk.END); self.entries[i][j].insert(0,cv)
        self.inner.update_idletasks(); self.canvas.config(scrollregion=self.canvas.bbox("all"))
        messagebox.showinfo("Завантажено",f"Аркуш «{sn}»: {nr} рядків, {nc} стовпців.")

    def _ask_sheet(self,names):
        dlg=tk.Toplevel(self.table_win); dlg.title("Вибір аркуша"); dlg.resizable(False,False); set_icon(dlg)
        frm=tk.Frame(dlg,padx=14,pady=12); frm.pack()
        tk.Label(frm,text="Виберіть аркуш:").pack(anchor="w",pady=(0,6))
        sf=tk.Frame(frm); sf.pack()
        sc=ttk.Scrollbar(sf); sc.pack(side=tk.RIGHT,fill=tk.Y)
        lb=tk.Listbox(sf,yscrollcommand=sc.set,height=min(10,len(names)),selectmode="single")
        for n in names: lb.insert(tk.END,n)
        lb.select_set(0); lb.pack(side=tk.LEFT); sc.config(command=lb.yview)
        out={"name":None}
        def ok():
            s=lb.curselection()
            if s: out["name"]=names[s[0]]
            dlg.destroy()
        bf=tk.Frame(frm); bf.pack(pady=(10,0))
        tk.Button(bf,text="OK",width=10,command=ok).pack(side=tk.LEFT,padx=(0,6))
        tk.Button(bf,text="Скасувати",width=12,command=dlg.destroy).pack(side=tk.LEFT)
        lb.bind("<Double-Button-1>",lambda e:ok()); dlg.bind("<Return>",lambda e:ok())
        dlg.update_idletasks(); center_win(dlg); dlg.grab_set(); self.root.wait_window(dlg)
        return out["name"]

    # ── collect data ──────────────────────────────────────────
    def _used_rep(self):
        rep_cols=[]
        for c in range(self.factors_count,self.cols):
            for r in range(len(self.entries)):
                s=self.entries[r][c].get().strip()
                if not s: continue
                try: float(s.replace(",",".")); rep_cols.append(c); break
                except Exception: continue
        return rep_cols

    def collect_long(self,design):
        long=[]; rep=self._used_rep()
        if not rep: return long,rep
        for i,row in enumerate(self.entries):
            lvls=[row[k].get().strip() or f"рядок{i+1}" for k in range(self.factors_count)]
            for ic,c in enumerate(rep):
                s=row[c].get().strip()
                if not s: continue
                try: val=float(s.replace(",","."))
                except Exception: continue
                rec={"value":val}
                for ki,fk in enumerate(self.factor_keys): rec[fk]=lvls[ki]
                if design in ("rcbd","split"): rec["BLOCK"]=f"Блок {ic+1}"
                long.append(rec)
        return long,rep

    # ════════════════════════════════════════════════════════════
    # DIALOGS: indicator/units, design, method
    # ════════════════════════════════════════════════════════════
    def ask_params(self):
        dlg=tk.Toplevel(self.root); dlg.title("Параметри звіту"); dlg.resizable(False,False); set_icon(dlg)
        frm=tk.Frame(dlg,padx=16,pady=16); frm.pack()
        tk.Label(frm,text="Назва показника:").grid(row=0,column=0,sticky="w",pady=5)
        e_ind=tk.Entry(frm,width=38,fg="#000000"); e_ind.grid(row=0,column=1,pady=5,padx=6)
        tk.Label(frm,text="Одиниці виміру:").grid(row=1,column=0,sticky="w",pady=5)
        e_un=tk.Entry(frm,width=38,fg="#000000"); e_un.grid(row=1,column=1,pady=5,padx=6)

        row_d=tk.Frame(frm); row_d.grid(row=2,column=0,columnspan=2,sticky="w",pady=(10,4))
        tk.Label(row_d,text="Дизайн:").pack(side=tk.LEFT)
        tk.Button(row_d,text=" ? ",width=3,command=self.show_design_help).pack(side=tk.LEFT,padx=6)
        dv=tk.StringVar(value="crd"); rf=tk.Frame(frm); rf.grid(row=2,column=1,sticky="w",pady=(10,4),padx=(180,0))
        rb_f=("Times New Roman",15)
        tk.Radiobutton(rf,text="CRD",variable=dv,value="crd",font=rb_f).pack(anchor="w")
        tk.Radiobutton(rf,text="RCBD",variable=dv,value="rcbd",font=rb_f).pack(anchor="w")
        tk.Radiobutton(rf,text="Split-plot (лише параметричний)",variable=dv,value="split",font=rb_f).pack(anchor="w")

        mfv=tk.StringVar(value="A"); sp_frm=tk.Frame(frm); sp_frm.grid(row=3,column=0,columnspan=2,sticky="w",pady=(6,0))
        tk.Label(sp_frm,text="Головний фактор:").pack(side=tk.LEFT)
        ttk.Combobox(sp_frm,textvariable=mfv,width=6,state="readonly",values=("A","B","C","D")).pack(side=tk.LEFT,padx=6)
        sp_frm.grid_remove()
        def _update_sp(*_): sp_frm.grid() if dv.get()=="split" else sp_frm.grid_remove(); dlg.update_idletasks()
        dv.trace_add("write",_update_sp)

        out={"ok":False,"indicator":"","units":"","design":"crd","split_main":"A"}
        def ok():
            out["indicator"]=e_ind.get().strip(); out["units"]=e_un.get().strip()
            out["design"]=dv.get(); out["split_main"]=mfv.get()
            if not out["indicator"] or not out["units"]:
                messagebox.showwarning("","Заповніть показник та одиниці виміру."); return
            out["ok"]=True; dlg.destroy()
        bf=tk.Frame(frm); bf.grid(row=4,column=0,columnspan=2,pady=(12,0))
        tk.Button(bf,text="OK",width=10,command=ok).pack(side=tk.LEFT,padx=4)
        tk.Button(bf,text="Скасувати",width=12,command=dlg.destroy).pack(side=tk.LEFT,padx=4)
        dlg.update_idletasks(); center_win(dlg); e_ind.focus_set()
        dlg.bind("<Return>",lambda e:ok()); dlg.grab_set(); self.root.wait_window(dlg)
        return out

    def choose_method(self,p_norm,design,n_var):
        dlg=tk.Toplevel(self.root); dlg.title("Вибір методу"); dlg.resizable(False,False); set_icon(dlg)
        frm=tk.Frame(dlg,padx=16,pady=14); frm.pack()
        normal=(p_norm is not None) and (not math.isnan(p_norm)) and (p_norm>0.05)
        rb_f=("Times New Roman",15)

        if normal:
            tk.Label(frm,text="Дані відповідають нормальному розподілу (Shapiro–Wilk).",
                     fg="#000000",justify="left").pack(anchor="w",pady=(0,8))
            options=[("НІР₀₅ (LSD)","lsd"),("Тест Тьюкі","tukey"),("Тест Дункана","duncan"),("Бонферроні","bonferroni")]
        else:
            if design=="split":
                tk.Label(frm,text=(
                    "Split-plot підтримує лише параметричний аналіз.\n"
                    "Залишки не відповідають нормальному розподілу → аналіз некоректний.\n\n"
                    "Рекомендації: трансформуйте дані або оберіть CRD/RCBD."),
                    fg="#c62828",justify="left").pack(anchor="w")
                options=[]
            else:
                tk.Label(frm,text=(
                    "Дані НЕ відповідають нормальному розподілу (Shapiro–Wilk).\n"
                    "Оберіть метод аналізу:"),fg="#c62828",justify="left").pack(anchor="w",pady=(0,8))
                if design=="crd":
                    options=[("Краскела–Уолліса","kw"),("Манна-Уітні","mw"),
                             ("🔁 Логарифмування даних + параметричний","log_param")]
                else:
                    if n_var==2: options=[("Wilcoxon (парний)","wilcoxon"),
                                          ("🔁 Логарифмування + параметричний","log_param")]
                    else:        options=[("Friedman","friedman"),
                                          ("🔁 Логарифмування + параметричний","log_param")]

        out={"ok":False,"method":None}
        if not options:
            tk.Button(frm,text="OK",width=10,command=dlg.destroy).pack(pady=(10,0))
            dlg.update_idletasks(); center_win(dlg); dlg.bind("<Return>",lambda e:dlg.destroy())
            dlg.grab_set(); self.root.wait_window(dlg); return out

        var=tk.StringVar(value=options[0][1])
        for txt,val in options: tk.Radiobutton(frm,text=txt,variable=var,value=val,font=rb_f).pack(anchor="w",pady=2)
        def ok():
            out["ok"]=True; out["method"]=var.get(); dlg.destroy()
        bf=tk.Frame(frm); bf.pack(pady=(12,0))
        tk.Button(bf,text="OK",width=10,command=ok).pack(side=tk.LEFT,padx=4)
        tk.Button(bf,text="Скасувати",width=12,command=dlg.destroy).pack(side=tk.LEFT,padx=4)
        dlg.update_idletasks(); center_win(dlg); dlg.bind("<Return>",lambda e:ok())
        dlg.grab_set(); self.root.wait_window(dlg); return out

    # ════════════════════════════════════════════════════════════
    # ANALYZE
    # ════════════════════════════════════════════════════════════
    def analyze(self):
        created=datetime.now()
        params=self.ask_params()
        if not params["ok"]: return
        indicator=params["indicator"]; units=params["units"]
        design=params["design"]; split_main=params["split_main"]

        long,used_rep=self.collect_long(design)
        if not long:
            messagebox.showwarning("","Немає числових даних."); return
        values=np.array([r["value"] for r in long],dtype=float)
        if len(values)<3: messagebox.showinfo("","Надто мало даних."); return

        lbf={f:first_seen([r.get(f) for r in long]) for f in self.factor_keys}
        var_order=first_seen([tuple(r.get(f) for f in self.factor_keys) for r in long])
        v_names=[" | ".join(map(str,k)) for k in var_order]
        n_var=len(var_order)

        # --- run model ---
        try:
            if design=="crd":   res=anova_crd(long,self.factor_keys,lbf)
            elif design=="rcbd": res=anova_rcbd(long,self.factor_keys,lbf)
            else:
                if split_main not in self.factor_keys: split_main=self.factor_keys[0]
                res=anova_split(long,self.factor_keys,split_main)
        except Exception as ex: messagebox.showerror("Помилка моделі",str(ex)); return

        residuals=np.array(res.get("residuals",[]),dtype=float)
        try: W,p_norm=shapiro(residuals) if len(residuals)>=3 else (np.nan,np.nan)
        except Exception: W,p_norm=np.nan,np.nan

        normal=(not math.isnan(p_norm)) and (p_norm>0.05)
        if design=="split" and not normal:
            messagebox.showwarning("Split-plot",
                "Залишки не відповідають нормальному розподілу.\n"
                "Split-plot аналіз некоректний. Розгляньте трансформацію або інший дизайн."); return

        choice=self.choose_method(p_norm,design,n_var)
        if not choice["ok"]: return
        method=choice["method"]

        # --- log-transform option ---
        log_applied=False
        if method=="log_param":
            if np.any(values<=0):
                messagebox.showwarning("Логарифмування","Дані містять нулі або від'ємні значення.\n"
                    "Логарифмування неможливе. Виберіть інший метод."); return
            long_orig=long; long=[dict(r,value=math.log(r["value"])) for r in long]
            values=np.array([r["value"] for r in long],dtype=float); log_applied=True
            try:
                if design=="crd":   res=anova_crd(long,self.factor_keys,lbf)
                elif design=="rcbd": res=anova_rcbd(long,self.factor_keys,lbf)
                else: res=anova_split(long,self.factor_keys,split_main)
            except Exception as ex: messagebox.showerror("",str(ex)); return
            residuals=np.array(res.get("residuals",[]),dtype=float)
            try: W,p_norm=shapiro(residuals) if len(residuals)>=3 else (np.nan,np.nan)
            except Exception: W,p_norm=np.nan,np.nan
            method="lsd"
            messagebox.showinfo("Логарифмування",
                f"Дані прологарифмовані (натуральний логарифм).\n"
                f"Shapiro–Wilk після трансформації: p={fmt(p_norm,4)}\n"
                +("✓ Нормальний розподіл" if p_norm>0.05 else "✗ Розподіл все ще не нормальний"))

        MS_err=res.get("MS_error",np.nan); df_err=res.get("df_error",np.nan)
        MS_wp=res.get("MS_whole",np.nan);  df_wp=res.get("df_whole",np.nan)
        split_mf=res.get("main_factor",split_main) if design=="split" else None

        # --- descriptive ---
        vs=vstats(long,self.factor_keys)
        v_means={k:vs[k][0] for k in vs}; v_sds={k:vs[k][1] for k in vs}; v_ns={k:vs[k][2] for k in vs}
        means1={v_names[i]:v_means.get(var_order[i],np.nan) for i in range(n_var)}
        ns1={v_names[i]:v_ns.get(var_order[i],0) for i in range(n_var)}
        gv=groups_by(long,tuple(self.factor_keys))
        groups1={v_names[i]:gv.get(var_order[i],[]) for i in range(n_var)}

        fg={f:{k[0]:v for k,v in groups_by(long,(f,)).items()} for f in self.factor_keys}
        fm={f:{lv:float(np.mean(arr)) if arr else np.nan for lv,arr in fg[f].items()} for f in self.factor_keys}
        fn={f:{lv:len(arr) for lv,arr in fg[f].items()} for f in self.factor_keys}

        fmed={f:{lv:median_q(arr)[0] for lv,arr in fg[f].items()} for f in self.factor_keys}
        fq={f:{lv:median_q(arr)[1:] for lv,arr in fg[f].items()} for f in self.factor_keys}

        vmed={var_order[i]:median_q(groups1[v_names[i]])[0] for i in range(n_var)}
        vq={var_order[i]:median_q(groups1[v_names[i]])[1:] for i in range(n_var)}

        rkv=mean_ranks(long,lambda r:" | ".join(str(r.get(f)) for f in self.factor_keys))
        rkf={f:mean_ranks(long,lambda r,ff=f:r.get(ff)) for f in self.factor_keys}

        # --- Levene ---
        lev_F,lev_p=(np.nan,np.nan)
        if method in ("lsd","tukey","duncan","bonferroni"):
            lev_F,lev_p=levene_test(groups1)

        # --- posthoc ---
        kw_H=kw_p=kw_df=kw_eps=np.nan; do_ph=True
        fr_chi=fr_p=fr_df=fr_W=np.nan
        wil_s=wil_p=np.nan
        rcbd_ph_rows=[]; rcbd_sig={}
        lf={f:{lv:"" for lv in lbf[f]} for f in self.factor_keys}
        lnamed={nm:"" for nm in v_names}
        ph_rows=[]; fpt={}

        if method=="lsd":
            for f in self.factor_keys:
                lvls=lbf[f]; MS_=(MS_wp if (design=="split" and f==split_mf) else MS_err)
                df_=(df_wp if (design=="split" and f==split_mf) else df_err)
                lf[f]=cld(lvls,fm[f],lsd_sig(lvls,fm[f],fn[f],MS_,df_))
            if design!="split":
                lnamed=cld(v_names,means1,lsd_sig(v_names,means1,ns1,MS_err,df_err))

        elif method in ("tukey","duncan","bonferroni"):
            if design!="split":
                ph_rows,sig=pairwise_param(v_names,means1,ns1,MS_err,df_err,method)
                lnamed=cld(v_names,means1,sig)
                for f in self.factor_keys:
                    r_,s_=pairwise_param(lbf[f],fm[f],fn[f],MS_err,df_err,method)
                    fpt[f]=r_; lf[f]=cld(lbf[f],fm[f],s_)
            else:
                for f in self.factor_keys:
                    MS_=MS_wp if f==split_mf else MS_err; df_=df_wp if f==split_mf else df_err
                    r_,s_=pairwise_param(lbf[f],fm[f],fn[f],MS_,df_,method)
                    fpt[f]=r_; lf[f]=cld(lbf[f],fm[f],s_)

        elif method=="kw":
            try:
                smp=[groups1[n] for n in v_names if len(groups1[n])>0]
                if len(smp)>=2:
                    kwr=kruskal(*smp); kw_H=float(kwr.statistic); kw_p=float(kwr.pvalue)
                    kw_df=len(smp)-1; kw_eps=eps2_kw(kw_H,len(long),len(smp))
            except Exception: pass
            if not math.isnan(kw_p) and kw_p>=ALPHA: do_ph=False
            if do_ph:
                ph_rows,sig=pairwise_mw(v_names,groups1)
                lnamed=cld(v_names,{n:float(np.median(groups1[n])) if groups1[n] else np.nan for n in v_names},sig)

        elif method=="mw":
            ph_rows,sig=pairwise_mw(v_names,groups1)
            lnamed=cld(v_names,{n:float(np.median(groups1[n])) if groups1[n] else np.nan for n in v_names},sig)

        elif method=="friedman":
            bnames=first_seen([f"Блок {i+1}" for i in range(len(used_rep))])
            long2=[dict(r,VARIANT=" | ".join(str(r.get(f)) for f in self.factor_keys)) for r in long]
            mat,_=rcbd_matrix(long2,v_names,bnames)
            if len(mat)<2: messagebox.showwarning("","Для Friedman потрібні ≥ 2 повних блоки."); return
            try:
                cols=list(zip(*mat)); fr=friedmanchisquare(*[np.array(c,dtype=float) for c in cols])
                fr_chi=float(fr.statistic); fr_p=float(fr.pvalue)
                fr_df=n_var-1; fr_W=kendalls_w(fr_chi,len(mat),n_var)
            except Exception: pass
            if not math.isnan(fr_p) and fr_p<ALPHA:
                rcbd_ph_rows,rcbd_sig=pairwise_wilcox(v_names,mat)
                lnamed=cld(v_names,{n:float(np.median(groups1[n])) if groups1[n] else np.nan for n in v_names},rcbd_sig)

        elif method=="wilcoxon":
            if n_var!=2: messagebox.showwarning("","Wilcoxon лише для 2 варіантів."); return
            bnames=first_seen([f"Блок {i+1}" for i in range(len(used_rep))])
            long2=[dict(r,VARIANT=" | ".join(str(r.get(f)) for f in self.factor_keys)) for r in long]
            mat,_=rcbd_matrix(long2,v_names,bnames)
            if len(mat)<2: messagebox.showwarning("","Потрібні ≥ 2 повних блоки."); return
            arr=np.array(mat,dtype=float)
            try:
                st,p=wilcoxon(arr[:,0],arr[:,1],zero_method="wilcox",alternative="two-sided",mode="auto")
                wil_s=float(st); wil_p=float(p)
            except Exception: pass
            if not math.isnan(wil_p) and wil_p<ALPHA:
                lnamed=cld(v_names,{n:float(np.median(groups1[n])) if groups1[n] else np.nan for n in v_names},
                           {(v_names[0],v_names[1]):True})

        lv_var={var_order[i]:lnamed.get(v_names[i],"") for i in range(n_var)}
        SS_tot=res.get("SS_total",np.nan); SS_err=res.get("SS_error",np.nan)
        R2=(1-(SS_err/SS_tot)) if not any(math.isnan(x) for x in [SS_tot,SS_err]) and SS_tot>0 else np.nan

        cv_r=[[self.ftitle(f),fmt(cv_means([fm[f].get(lv,np.nan) for lv in lbf[f]]),2)]
              for f in self.factor_keys]
        cv_r.append(["Загальний",fmt(cv_vals(values),2)])

        nonparam=method in ("mw","kw","friedman","wilcoxon")

        # ─ rename Фактор X → custom title in table ─
        def _rename_row(row):
            nm=row[0]
            if isinstance(nm,str) and nm.startswith("Фактор "):
                rest=nm.replace("Фактор ","")
                parts=rest.split("×")
                row=list(row); row[0]="×".join(self.ftitle(p) if p in self.factor_keys else p for p in parts)
            return row

        # ─────────────────────────────────────────────────────
        # BUILD ANOVA TABLE rows
        anova_rows=[]
        for raw_row in res["table"]:
            nm,SSv,dfv,MSv,Fv,pv=raw_row
            df_s=str(int(dfv)) if dfv is not None and not (isinstance(dfv,float) and math.isnan(dfv)) else ""
            nm2=_rename_row([nm])[0]
            if nm2.startswith("Залишок") or nm2=="Загальна" or "WP-error" in nm2 or nm2=="Блоки":
                anova_rows.append([nm2,fmt(SSv,3),df_s,fmt(MSv,3),"","",""])
            else:
                mk=sig_mark(pv); concl=f"різниця {mk}" if mk else "–"
                anova_rows.append([nm2,fmt(SSv,3),df_s,fmt(MSv,3),fmt(Fv,3),fmt(pv,4),concl])

        eff_rows=[[_rename_row([r[0]])[0],r[1]] for r in build_eff_rows(res["table"])]
        pe2_rows=[[_rename_row([r[0]])[0],r[1],r[2]] for r in build_pe2_rows(res["table"])]

        # ─────────────────────────────────────────────────────
        # SHOW REPORT
        self.show_report(
            created=created, indicator=indicator, units=units, design=design,
            method=method, log_applied=log_applied, n_var=n_var, n_rep=len(used_rep), n_obs=len(long),
            split_mf=split_mf, W=W, p_norm=p_norm,
            lev_F=lev_F, lev_p=lev_p,
            kw_H=kw_H, kw_p=kw_p, kw_df=kw_df, kw_eps=kw_eps, do_ph=do_ph,
            fr_chi=fr_chi, fr_p=fr_p, fr_df=fr_df, fr_W=fr_W,
            wil_s=wil_s, wil_p=wil_p,
            anova_rows=anova_rows, eff_rows=eff_rows, pe2_rows=pe2_rows,
            cv_r=cv_r, R2=R2,
            lf=lf, lv_var=lv_var, lbf=lbf, fm=fm, fmed=fmed, fq=fq, fn=fn,
            var_order=var_order, v_names=v_names, v_means=v_means, v_sds=v_sds, v_ns=v_ns,
            vmed=vmed, vq=vq, rkv=rkv, rkf=rkf,
            groups1=groups1,
            ph_rows=ph_rows, fpt=fpt,
            rcbd_ph_rows=rcbd_ph_rows,
            nonparam=nonparam, res=res, split_main=split_main,
        )
        self.show_graphs(long, lf, indicator, units, eff_rows)

    # ════════════════════════════════════════════════════════════
    # REPORT WINDOW  (Treeview tables)
    # ════════════════════════════════════════════════════════════
    def show_report(self, **kw):
        if self.report_win and tk.Toplevel.winfo_exists(self.report_win):
            self.report_win.destroy()
        self.report_win=rw=tk.Toplevel(self.root)
        rw.title("Звіт"); rw.geometry("1200x800"); set_icon(rw)

        top=tk.Frame(rw,padx=8,pady=6); top.pack(fill=tk.X)
        def copy_all():
            # collect text from text widget
            rw.clipboard_clear(); rw.clipboard_append(self._report_text_buf)
            messagebox.showinfo("","Текстовий звіт скопійовано.")
        tk.Button(top,text="Копіювати текст звіту",command=copy_all).pack(side=tk.LEFT,padx=4)

        main=tk.Frame(rw); main.pack(fill=tk.BOTH,expand=True)
        vsb=ttk.Scrollbar(main,orient="vertical"); vsb.pack(side=tk.RIGHT,fill=tk.Y)
        cv=tk.Canvas(main,yscrollcommand=vsb.set); cv.pack(side=tk.LEFT,fill=tk.BOTH,expand=True)
        vsb.config(command=cv.yview)
        body=tk.Frame(cv); cv.create_window((0,0),window=body,anchor="nw")
        def _on_cfg(e): cv.configure(scrollregion=cv.bbox("all"))
        body.bind("<Configure>",_on_cfg)

        txt_buf=[]
        def _txt(s):
            tk.Label(body,text=s,font=("Times New Roman",12),fg="#000000",
                     justify="left",anchor="w",wraplength=1100).pack(fill=tk.X,padx=12,pady=(1,0))
            txt_buf.append(s)

        def _head(s):
            tk.Label(body,text=s,font=("Times New Roman",13,"bold"),fg="#000000",
                     justify="left",anchor="w").pack(fill=tk.X,padx=12,pady=(8,2))
            txt_buf.append("\n"+s)

        def _table(headers,rows,cw=None):
            f,_=make_tv_table(body,headers,rows,min_col_px=90 if cw is None else cw)
            f.pack(fill=tk.X,padx=12,pady=(2,6))
            # also append to text buffer
            txt_buf.append("\t".join(headers))
            for row in rows: txt_buf.append("\t".join("" if v is None else str(v) for v in row))

        def _sep():
            ttk.Separator(body,orient="horizontal").pack(fill=tk.X,padx=12,pady=4)

        d=kw
        _txt(f"З В І Т   С Т А Т И С Т И Ч Н О Г О   А Н А Л І З У")
        _txt(f"Показник: {d['indicator']}   |   Одиниці: {d['units']}")
        _txt(f"Дата: {d['created'].strftime('%d.%m.%Y %H:%M')}")
        _sep()

        design_lbl={"crd":"CRD (повна рандомізація)","rcbd":"RCBD (блочна рандомізація)","split":"Split-plot"}[d['design']]
        _txt(f"Дизайн: {design_lbl}   |   Варіантів: {d['n_var']}   |   Повт.: {d['n_rep']}   |   Спостережень: {d['n_obs']}")
        if d['design']=="split": _txt(f"Головний фактор (WP): {d['split_mf']}")
        if d['log_applied']: _txt("⚠ Застосовано логарифмування даних (натуральний логарифм).")

        method_lbl={
            "lsd":"НІР₀₅ (LSD)","tukey":"Тест Тьюкі","duncan":"Тест Дункана","bonferroni":"Бонферроні",
            "kw":"Kruskal–Wallis","mw":"Mann–Whitney","friedman":"Friedman","wilcoxon":"Wilcoxon",
        }.get(d['method'],"")
        _txt(f"Метод: {method_lbl}")
        _txt("Позначення: ** — p<0.01; * — p<0.05; різні літери → істотна різниця.")
        _sep()

        _txt(f"Shapiro–Wilk (залишки): {norm_txt(d['p_norm'])}   W={fmt(d['W'],4)}   p={fmt(d['p_norm'],4)}")

        # global tests
        if d['method']=="kw" and not math.isnan(d['kw_p']):
            c=("різниця "+sig_mark(d['kw_p'])) if d['kw_p']<ALPHA else "–"
            _txt(f"Kruskal–Wallis:  H={fmt(d['kw_H'],4)}  df={d['kw_df']}  p={fmt(d['kw_p'],4)}  {c}   ε²={fmt(d['kw_eps'],4)}")
        if d['method']=="friedman" and not math.isnan(d['fr_p']):
            c=("різниця "+sig_mark(d['fr_p'])) if d['fr_p']<ALPHA else "–"
            _txt(f"Friedman:  χ²={fmt(d['fr_chi'],4)}  df={d['fr_df']}  p={fmt(d['fr_p'],4)}  {c}   Kendall's W={fmt(d['fr_W'],4)}")
        if d['method']=="wilcoxon" and not math.isnan(d['wil_p']):
            c=("різниця "+sig_mark(d['wil_p'])) if d['wil_p']<ALPHA else "–"
            _txt(f"Wilcoxon signed-rank:  W={fmt(d['wil_s'],4)}  p={fmt(d['wil_p'],4)}  {c}")

        if not d['nonparam']:
            if not math.isnan(d['lev_p']):
                lc="умова виконується" if d['lev_p']>=ALPHA else f"умова порушена {sig_mark(d['lev_p'])}"
                _txt(f"Тест Левена (однорідність дисперсій):  F={fmt(d['lev_F'],4)}  p={fmt(d['lev_p'],4)}  {lc}")
            _sep()
            _head("ТАБЛИЦЯ 1. Дисперсійний аналіз (ANOVA)")
            _table(["Джерело","SS","df","MS","F","p","Висновок"],d['anova_rows'])
            _head("ТАБЛИЦЯ 2. Сила впливу факторів (% від SS)")
            _table(["Джерело","%"],d['eff_rows'])
            _head("ТАБЛИЦЯ 3. Розмір ефекту (partial η²)")
            _table(["Джерело","partial η²","Сила ефекту"],d['pe2_rows'])
            _head("ТАБЛИЦЯ 4. Коефіцієнт варіації (CV, %)")
            _table(["Елемент","CV, %"],d['cv_r'])
            _txt(f"R² = {fmt(d['R2'],4)}")

            tno=5
            if d['method']=="lsd":
                nir_r=[[k,fmt(v,4)] for k,v in d['res'].get("NIR05",{}).items()]
                if nir_r:
                    _head(f"ТАБЛИЦЯ {tno}. НІР₀₅")
                    _table(["Елемент","НІР₀₅"],nir_r); tno+=1

            for f in self.factor_keys:
                _head(f"ТАБЛИЦЯ {tno}. Середні по фактору: {self.ftitle(f)}")
                rows_f=[[str(lv),fmt(d['fm'][f].get(lv,np.nan),3),d['lf'][f].get(lv,"") or "–"]
                        for lv in d['lbf'][f]]
                _table([self.ftitle(f),"Середнє","Літери CLD"],rows_f); tno+=1

            _head(f"ТАБЛИЦЯ {tno}. Середні по варіантах")
            rows_v=[[nm,fmt(d['v_means'].get(d['var_order'][i],np.nan),3),
                     fmt(d['v_sds'].get(d['var_order'][i],np.nan),3),
                     d['lv_var'].get(d['var_order'][i],"") or "–"]
                    for i,nm in enumerate(d['v_names'])]
            _table(["Варіант","Середнє","± SD","Літери CLD"],rows_v); tno+=1

            if d['design']!="split":
                if d['method'] in ("tukey","duncan","bonferroni") and d['ph_rows']:
                    _head(f"ТАБЛИЦЯ {tno}. Парні порівняння варіантів")
                    _table(["Пара","p","Висновок"],d['ph_rows']); tno+=1
            else:
                for f in self.factor_keys:
                    rr=d['fpt'].get(f,[])
                    if rr:
                        _head(f"ТАБЛИЦЯ {tno}. Парні порівняння: {self.ftitle(f)}")
                        _table(["Пара","p","Висновок"],rr); tno+=1

        else:  # nonparam
            tno=1
            for f in self.factor_keys:
                _head(f"ТАБЛИЦЯ {tno}. Описова (непараметрична): {self.ftitle(f)}")
                rows=[]
                for lv in d['lbf'][f]:
                    med=d['fmed'][f].get(lv,np.nan); q1q3=d['fq'][f].get(lv,(np.nan,np.nan))
                    rk=d['rkf'][f].get(lv,np.nan)
                    rows.append([str(lv),str(d['fn'][f].get(lv,0)),fmt(med,3),
                                 f"{fmt(q1q3[0],3)}–{fmt(q1q3[1],3)}" if not any(math.isnan(x) for x in q1q3) else "",
                                 fmt(rk,2)])
                _table([self.ftitle(f),"n","Медіана","Q1–Q3","Сер. ранг"],rows); tno+=1

            _head(f"ТАБЛИЦЯ {tno}. Описова (непараметрична): варіанти")
            rows=[]
            for i,k in enumerate(d['var_order']):
                nm=d['v_names'][i]; med=d['vmed'].get(k,np.nan); q1q3=d['vq'].get(k,(np.nan,np.nan))
                rk=d['rkv'].get(nm,np.nan)
                rows.append([nm,str(d['v_ns'].get(k,0)),fmt(med,3),
                             f"{fmt(q1q3[0],3)}–{fmt(q1q3[1],3)}" if not any(math.isnan(x) for x in q1q3) else "",
                             fmt(rk,2)])
            _table(["Варіант","n","Медіана","Q1–Q3","Сер. ранг"],rows); tno+=1

            if d['method']=="kw":
                if not d['do_ph']: _txt("Пост-хок: Kruskal–Wallis p ≥ 0.05 → порівняння не виконувалися.")
                elif d['ph_rows']:
                    _head(f"ТАБЛИЦЯ {tno}. Парні порівняння (Mann–Whitney + Bonferroni, Cliff's δ)")
                    _table(["Пара","U","p (Bonf.)","Висновок","δ","Ефект"],d['ph_rows']); tno+=1
            if d['method']=="mw" and d['ph_rows']:
                _head(f"ТАБЛИЦЯ {tno}. Парні порівняння (Mann–Whitney + Bonferroni, Cliff's δ)")
                _table(["Пара","U","p (Bonf.)","Висновок","δ","Ефект"],d['ph_rows']); tno+=1
            if d['method']=="friedman":
                if not math.isnan(d['fr_p']) and d['fr_p']>=ALPHA: _txt("Friedman p ≥ 0.05 → пост-хок не виконувався.")
                elif d['rcbd_ph_rows']:
                    _head(f"ТАБЛИЦЯ {tno}. Парні порівняння (Wilcoxon + Bonferroni)")
                    _table(["Пара","W","p (Bonf.)","Висновок","r"],d['rcbd_ph_rows']); tno+=1

        _sep()
        _txt(f"Звіт сформовано: {d['created'].strftime('%d.%m.%Y, %H:%M')}")
        self._report_text_buf="\n".join(txt_buf)

    # ════════════════════════════════════════════════════════════
    # GRAPHICAL REPORT  (Boxplot + Venn)
    # ════════════════════════════════════════════════════════════
    def show_graphs(self, long, letters_factor, indicator, units, eff_rows):
        if not HAS_MPL:
            messagebox.showwarning("","matplotlib недоступний."); return
        if self.graph_win and tk.Toplevel.winfo_exists(self.graph_win):
            self.graph_win.destroy()
        self.graph_win=gw=tk.Toplevel(self.root)
        gw.title("Графічний звіт"); gw.geometry("1300x820"); set_icon(gw)

        top=tk.Frame(gw,padx=8,pady=6); top.pack(fill=tk.X)
        tk.Button(top,text="⚙ Налаштування",command=lambda:self._open_graph_settings(gw,long,letters_factor,indicator,units,eff_rows)).pack(side=tk.LEFT,padx=4)
        tk.Button(top,text="📋 Копіювати Boxplot",command=lambda:self._copy_fig(self._graph_fig_bp)).pack(side=tk.LEFT,padx=4)
        tk.Button(top,text="📋 Копіювати Вень",command=lambda:self._copy_fig(self._graph_fig_vn)).pack(side=tk.LEFT,padx=4)
        tk.Label(top,text="Вставте у Word через Ctrl+V",anchor="w").pack(side=tk.LEFT,padx=8)

        self._graph_frame=tk.Frame(gw); self._graph_frame.pack(fill=tk.BOTH,expand=True)
        self._draw_graphs(self._graph_frame,long,letters_factor,indicator,units,eff_rows)

    def _open_graph_settings(self,gw,long,lf,indicator,units,eff_rows):
        dlg=GraphSettingsDlg(gw,self.graph_settings)
        gw.wait_window(dlg)
        if dlg.result:
            self.graph_settings=dlg.result
            # redraw
            for w in self._graph_frame.winfo_children(): w.destroy()
            self._draw_graphs(self._graph_frame,long,lf,indicator,units,eff_rows)

    def _draw_graphs(self,frame,long,letters_factor,indicator,units,eff_rows):
        gs=self.graph_settings
        fp={"fontfamily":gs["font_family"],"fontsize":gs["font_size"]}
        fs=gs["font_style"]; fw="bold" if "bold" in fs else "normal"; fst="italic" if "italic" in fs else "normal"
        fp["fontweight"]=fw; fp["fontstyle"]=fst

        nb=ttk.Notebook(frame); nb.pack(fill=tk.BOTH,expand=True)

        # ── BOXPLOT ──
        bp_frame=tk.Frame(nb); nb.add(bp_frame,text="Boxplot")
        fig_bp=Figure(figsize=(11,5.5),dpi=100); ax=fig_bp.add_subplot(111)

        positions=[]; data=[]; xlbls=[]; let_list=[]; fcentres=[]
        x=1.; gap=1.
        for f in self.factor_keys:
            lvls=self.lbf_cache.get(f,[]) if hasattr(self,"lbf_cache") else \
                 first_seen([r.get(f) for r in long if r.get(f) is not None])
            if not lvls: continue
            sx=x
            for lv in lvls:
                arr=[float(r["value"]) for r in long if r.get(f)==lv and r.get("value") is not None]
                arr=[v for v in arr if not math.isnan(v)]
                data.append(arr); positions.append(x); xlbls.append(str(lv))
                let_list.append((f,lv)); x+=1.
            fcentres.append(((sx+x-1)/2.,self.ftitle(f))); x+=gap

        if not data:
            tk.Label(bp_frame,text="Недостатньо даних").pack(pady=20)
        else:
            bp=ax.boxplot(data,positions=positions,widths=0.6,showfliers=True,patch_artist=True)
            for patch in bp["boxes"]:   patch.set(facecolor=gs["box_color"])
            for line in bp["medians"]:  line.set(color=gs["median_color"],linewidth=2)
            for line in bp["whiskers"]+bp["caps"]: line.set(color=gs["whisker_color"])
            for fl in bp["fliers"]: fl.set(markerfacecolor=gs["flier_color"],marker="o",markersize=4)

            ax.set_title(f"{indicator}, {units}",**fp)
            ax.set_ylabel(units,**fp)
            ax.set_xticks(positions); ax.set_xticklabels(xlbls,rotation=90,
                fontfamily=gs["font_family"],fontsize=max(8,gs["font_size"]-1))
            ax.yaxis.grid(True,linestyle="-",lw=0.5,alpha=0.35)
            ax.tick_params(axis="both",labelsize=max(8,gs["font_size"]-1))

            allv=[v for a in data for v in a]
            dy=max(allv)-min(allv) if len(allv)>1 else 1.
            off=0.04*dy if dy>0 else 0.5
            for i,(f,lv) in enumerate(let_list):
                lt=(letters_factor.get(f,{}) or {}).get(lv,"")
                if lt and data[i]:
                    ax.text(positions[i],max(data[i])+off,lt,ha="center",va="bottom",**fp)

            fig_bp.subplots_adjust(bottom=0.32,top=0.91,left=0.08,right=0.98)
            for cx,fnm in fcentres:
                ax.text(cx,-0.22,fnm,ha="center",va="top",transform=ax.get_xaxis_transform(),**fp)

        self._graph_fig_bp=fig_bp
        cv_bp=FigureCanvasTkAgg(fig_bp,master=bp_frame); cv_bp.draw()
        cv_bp.get_tk_widget().pack(fill=tk.BOTH,expand=True)

        # ── VENN ──
        vn_frame=tk.Frame(nb); nb.add(vn_frame,text="Діаграма Венна")
        fig_vn=Figure(figsize=(6,5),dpi=100); ax2=fig_vn.add_subplot(111)

        venn_vals=[(nm,float(pct)) for nm,pct in eff_rows
                   if pct is not None and pct!="" and not (isinstance(pct,float) and math.isnan(float(pct) if pct else np.nan))]
        venn_vals=[(nm,float(pct)) for nm,pct in venn_vals]

        if venn_vals:
            draw_venn(ax2,
                      labels=[v[0] for v in venn_vals],
                      values=venn_vals,
                      colors=gs["venn_colors"],
                      alpha=gs["venn_alpha"],
                      font_size=gs["venn_font_size"],
                      font_color=gs["venn_font_color"],
                      font_family=gs["font_family"])
        else:
            ax2.text(0.5,0.5,"Недостатньо даних",ha="center",va="center",transform=ax2.transAxes)
            ax2.axis("off")

        self._graph_fig_vn=fig_vn
        cv_vn=FigureCanvasTkAgg(fig_vn,master=vn_frame); cv_vn.draw()
        cv_vn.get_tk_widget().pack(fill=tk.BOTH,expand=True)

        # cache lbf for redraw
        self.lbf_cache={}
        for f in self.factor_keys:
            self.lbf_cache[f]=first_seen([r.get(f) for r in long if r.get(f) is not None])

    def _copy_fig(self,fig):
        if fig is None: messagebox.showwarning("","Графік відсутній."); return
        ok,msg=_copy_fig_to_clipboard(fig)
        if ok: messagebox.showinfo("","Графік скопійовано (PNG).")
        else:  messagebox.showwarning("",f"Помилка: {msg}")


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════
if __name__=="__main__":
    root=tk.Tk()
    set_icon(root)
    app=SADTk(root)
    root.mainloop()
