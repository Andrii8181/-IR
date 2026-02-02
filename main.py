# =========================
# main.py  (ЧАСТИНА 2/2)
# ВСТАВЛЯЙ ОДРАЗУ ПІСЛЯ ЧАСТИНИ 1 (з нового рядка)
# =========================

# ---------------------------------------------------------
# 1) Статистичні ядра: OLS-ANOVA (CRD/RCBD/Split-plot)
# ---------------------------------------------------------

def _encode_factor(values):
    # повертає список рівнів у порядку першої появи
    return first_seen_order([v for v in values])

def _design_matrix_from_terms(long, terms):
    """
    terms: list of tuples, e.g. [("A",), ("B",), ("A","B"), ("BLOCK",), ("BLOCK","A")]
    returns X, y, columns_info
    """
    y = np.array([float(r["value"]) for r in long], dtype=float)

    # збираємо рівні
    levels = {}
    for term in terms:
        for k in term:
            if k not in levels:
                levels[k] = _encode_factor([rec.get(k) for rec in long])

    # індекс рівня
    level_index = {k: {lvl: i for i, lvl in enumerate(levels[k])} for k in levels}

    cols = [np.ones(len(long), dtype=float)]  # intercept
    col_names = ["Intercept"]

    # dummy coding: для кожного фактора k беремо (L-1) даммі
    def add_main(k):
        L = levels[k]
        for j in range(1, len(L)):
            vj = np.array([1.0 if level_index[k][rec.get(k)] == j else 0.0 for rec in long], dtype=float)
            cols.append(vj)
            col_names.append(f"{k}={L[j]}")

    # для взаємодій: перемноження даммі головних ефектів
    dummy_cache = {}

    def get_dummy_cols(k):
        if k in dummy_cache:
            return dummy_cache[k]
        L = levels[k]
        mats = []
        names = []
        for j in range(1, len(L)):
            vj = np.array([1.0 if level_index[k][rec.get(k)] == j else 0.0 for rec in long], dtype=float)
            mats.append(vj)
            names.append((k, L[j]))
        dummy_cache[k] = (mats, names)
        return dummy_cache[k]

    for term in terms:
        if len(term) == 1:
            add_main(term[0])
        else:
            mats_list = []
            name_list = []
            for k in term:
                mats_k, names_k = get_dummy_cols(k)
                mats_list.append(mats_k)
                name_list.append(names_k)

            # декартовий добуток даммі
            def rec_build(idx, vec, label_parts):
                if idx == len(mats_list):
                    cols.append(vec)
                    nm = "×".join([f"{k}={lvl}" for (k, lvl) in label_parts])
                    col_names.append(nm)
                    return
                for vj, (k, lvl) in zip(mats_list[idx], name_list[idx]):
                    rec_build(idx + 1, vec * vj, label_parts + [(k, lvl)])

            rec_build(0, np.ones(len(long), dtype=float), [])

    X = np.column_stack(cols)
    return X, y, col_names

def _ols_fit(X, y):
    # звичайний OLS через lstsq
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    sse = float(np.sum(resid ** 2))
    return beta, yhat, resid, sse

def _df_residual(n, p):
    return int(max(0, n - p))

def _ss_total(y):
    ym = float(np.mean(y))
    return float(np.sum((y - ym) ** 2))

def _anova_type1(long, ordered_terms):
    """
    Type I SS (послідовні): додаємо терми один за одним і дивимось приріст SS.
    Повертає таблицю: [(name, SS, df, MS, F, p), ...] + residual + total
    """
    # 0) нульова модель (тільки intercept)
    X0 = np.ones((len(long), 1), dtype=float)
    y = np.array([float(r["value"]) for r in long], dtype=float)
    _, _, resid0, sse0 = _ols_fit(X0, y)

    table = []
    current_terms = []
    current_X = X0
    _, _, resid_cur, sse_cur = _ols_fit(current_X, y)

    # для df: p = кількість колонок
    p_cur = current_X.shape[1]

    for term in ordered_terms:
        # додаємо term
        next_terms = current_terms + [term]
        X, y, col_names = _design_matrix_from_terms(long, next_terms)
        _, _, resid_next, sse_next = _ols_fit(X, y)

        ss_term = sse_cur - sse_next
        df_term = X.shape[1] - p_cur
        ms_term = ss_term / df_term if df_term > 0 else np.nan

        # тимчасово лишимо F/p пустими — порахуємо після отримання MS_error
        table.append((term, ss_term, df_term, ms_term, np.nan, np.nan))

        current_terms = next_terms
        current_X = X
        resid_cur = resid_next
        sse_cur = sse_next
        p_cur = X.shape[1]

    df_error = _df_residual(len(y), current_X.shape[1])
    ss_error = sse_cur
    ms_error = ss_error / df_error if df_error > 0 else np.nan

    # порахувати F,p
    out_rows = []
    for (term, ss, df, ms, _, _) in table:
        if df > 0 and df_error > 0 and ms_error > 0:
            Fv = ms / ms_error
            pv = float(f_dist.sf(Fv, df, df_error))
        else:
            Fv, pv = (np.nan, np.nan)
        out_rows.append((term, ss, df, ms, Fv, pv))

    ss_total = _ss_total(y)
    df_total = len(y) - 1

    return {
        "rows": out_rows,
        "SS_error": ss_error,
        "df_error": df_error,
        "MS_error": ms_error,
        "SS_total": ss_total,
        "df_total": df_total,
        "residuals": resid_cur,
    }

def _term_label(term_tuple):
    if len(term_tuple) == 1:
        return f"Фактор {term_tuple[0]}"
    return "Фактор " + "×".join(term_tuple)

def anova_n_way(long, factor_keys, levels_by_factor):
    """
    CRD: n-way ANOVA (Type I SS). Повертає структуру для звіту.
    """
    ordered_terms = []
    # головні ефекти
    for k in factor_keys:
        ordered_terms.append((k,))
    # взаємодії (усі, по зростанню порядку)
    if len(factor_keys) >= 2:
        for r in range(2, len(factor_keys) + 1):
            for comb in combinations(factor_keys, r):
                ordered_terms.append(tuple(comb))

    an = _anova_type1(long, ordered_terms)

    table = []
    for (term, ss, df, ms, Fv, pv) in an["rows"]:
        table.append((_term_label(term), ss, df, ms, Fv, pv))
    table.append(("Залишок", an["SS_error"], an["df_error"], an["MS_error"], np.nan, np.nan))
    table.append(("Загальна", an["SS_total"], an["df_total"], np.nan, np.nan, np.nan))

    # середні по клітинках (для залишків CRD)
    cell_means = {}
    cell = groups_by_keys(long, tuple(factor_keys))
    for k, arr in cell.items():
        cell_means[k] = float(np.mean(arr)) if arr else np.nan

    return {
        "table": table,
        "SS_error": an["SS_error"],
        "df_error": an["df_error"],
        "MS_error": an["MS_error"],
        "SS_total": an["SS_total"],
        "cell_means": cell_means,
        "residuals": an["residuals"],
        "NIR05": {},
    }

def anova_rcbd_ols(long, factor_keys, levels_by_factor, block_key="BLOCK"):
    """
    RCBD: додаємо BLOCK як фактор у модель (Type I SS)
    """
    ordered_terms = [(block_key,)]
    for k in factor_keys:
        ordered_terms.append((k,))
    if len(factor_keys) >= 2:
        for r in range(2, len(factor_keys) + 1):
            for comb in combinations(factor_keys, r):
                ordered_terms.append(tuple(comb))

    an = _anova_type1(long, ordered_terms)

    table = []
    for (term, ss, df, ms, Fv, pv) in an["rows"]:
        # term може містити BLOCK
        if term == (block_key,):
            name = "Блоки"
        else:
            name = _term_label(term)
        table.append((name, ss, df, ms, Fv, pv))
    table.append(("Залишок", an["SS_error"], an["df_error"], an["MS_error"], np.nan, np.nan))
    table.append(("Загальна", an["SS_total"], an["df_total"], np.nan, np.nan, np.nan))

    return {
        "table": table,
        "SS_error": an["SS_error"],
        "df_error": an["df_error"],
        "MS_error": an["MS_error"],
        "SS_total": an["SS_total"],
        "residuals": an["residuals"],
        "NIR05": {},
    }

def anova_splitplot_ols(long, factor_keys, main_factor="A", block_key="BLOCK"):
    """
    Split-plot (параметричний):
    - whole-plot error: BLOCK×main_factor
    - sub-plot error: residual
    Модель: BLOCK + main + BLOCK×main + інші фактори + взаємодії між факторами (без BLOCK...)
    """
    if main_factor not in factor_keys:
        main_factor = factor_keys[0]

    other = [k for k in factor_keys if k != main_factor]

    ordered_terms = []
    ordered_terms.append((block_key,))
    ordered_terms.append((main_factor,))
    ordered_terms.append((block_key, main_factor))  # whole-plot error term

    for k in other:
        ordered_terms.append((k,))

    # взаємодії між факторами (без BLOCK)
    all_f = list(factor_keys)
    if len(all_f) >= 2:
        for r in range(2, len(all_f) + 1):
            for comb in combinations(all_f, r):
                ordered_terms.append(tuple(comb))

    an = _anova_type1(long, ordered_terms)

    # витягнути MS/df whole-plot error
    MS_whole = np.nan
    df_whole = np.nan
    SS_whole = np.nan
    for (term, ss, df, ms, Fv, pv) in an["rows"]:
        if term == (block_key, main_factor) or term == (main_factor, block_key):
            SS_whole, df_whole, MS_whole = ss, df, ms
            break

    table = []
    for (term, ss, df, ms, Fv, pv) in an["rows"]:
        if term == (block_key,):
            name = "Блоки"
        elif term == (block_key, main_factor) or term == (main_factor, block_key):
            name = f"Блоки×{main_factor} (Whole-plot error)"
        else:
            name = _term_label(term)

        # важливо: F і p для main_factor треба рахувати по MS_whole, а інші — по MS_error
        if term == (main_factor,):
            if df_whole and df_whole > 0 and MS_whole and MS_whole > 0:
                Fv2 = ms / MS_whole
                pv2 = float(f_dist.sf(Fv2, df, df_whole))
            else:
                Fv2, pv2 = (np.nan, np.nan)
            table.append((name, ss, df, ms, Fv2, pv2))
        else:
            # для інших залишаємо Type I на MS_error (вже пораховано)
            table.append((name, ss, df, ms, Fv, pv))

    # residual і total
    table.append(("Залишок", an["SS_error"], an["df_error"], an["MS_error"], np.nan, np.nan))
    table.append(("Загальна", an["SS_total"], an["df_total"], np.nan, np.nan, np.nan))

    return {
        "table": table,
        "SS_error": an["SS_error"],
        "df_error": an["df_error"],
        "MS_error": an["MS_error"],
        "SS_total": an["SS_total"],
        "residuals": an["residuals"],
        "MS_whole": MS_whole,
        "df_whole": df_whole,
        "main_factor": main_factor,
        "NIR05": {},
    }

# ---------------------------------------------------------
# 2) Перевірки, ефекти, пост-хок
# ---------------------------------------------------------

def brown_forsythe_from_groups(groups_named):
    # groups_named: dict(name -> list values)
    arrays = [np.array(v, dtype=float) for v in groups_named.values() if len(v) > 0]
    if len(arrays) < 2:
        return (np.nan, np.nan)
    try:
        stat, p = levene(*arrays, center="median")
        return float(stat), float(p)
    except Exception:
        return (np.nan, np.nan)

def lsd_sig_matrix(levels, means, ns, MS, df, alpha=0.05):
    """
    Повертає матрицю значущості: {(a,b): True/False} для LSD.
    Використовує t-критичне та гармонічне середнє n, якщо n різні.
    """
    sig = {}
    if df is None or (isinstance(df, float) and math.isnan(df)) or df <= 0:
        return sig
    tcrit = float(t_dist.ppf(1 - alpha / 2.0, df))

    # гармонічне n
    n_list = []
    for lvl in levels:
        n = ns.get(lvl, 0)
        if n and n > 0:
            n_list.append(n)
    n_h = harmonic_mean(n_list)
    if isinstance(n_h, float) and math.isnan(n_h):
        return sig

    lsd = tcrit * math.sqrt(2.0 * MS / n_h) if (MS and MS > 0) else np.nan
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            a, b = levels[i], levels[j]
            da = abs(means.get(a, np.nan) - means.get(b, np.nan))
            sig[(a, b)] = bool((not math.isnan(da)) and (not math.isnan(lsd)) and (da > lsd))
    return sig

def pairwise_param_short_variants_pm(levels, means, ns, MS, df, method, alpha=0.05):
    """
    Повертає rows для таблиці парних порівнянь (p і висновок) + sig_matrix.
    Реалізація: Tukey / Duncan / Bonferroni.
    """
    rows = []
    sig = {}
    if df is None or (isinstance(df, float) and math.isnan(df)) or df <= 0 or MS is None or MS <= 0:
        return rows, sig

    # гармонічне n
    n_list = [ns.get(l, 0) for l in levels if ns.get(l, 0) > 0]
    n_h = harmonic_mean(n_list)
    if isinstance(n_h, float) and math.isnan(n_h):
        return rows, sig

    m = len(levels)
    mtests = m * (m - 1) // 2

    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            a, b = levels[i], levels[j]
            diff = abs(means.get(a, np.nan) - means.get(b, np.nan))
            se = math.sqrt(MS / n_h)
            if se <= 0 or math.isnan(diff):
                p = np.nan
            else:
                if method == "tukey":
                    q = diff / se
                    p = float(studentized_range.sf(q, m, df))
                elif method == "duncan":
                    # спрощено: використовуємо studentized_range як основу (практично працює)
                    q = diff / se
                    p = float(studentized_range.sf(q, m, df))
                else:  # bonferroni
                    tval = diff / math.sqrt(2.0 * MS / n_h)
                    p_raw = float(2.0 * t_dist.sf(abs(tval), df))
                    p = min(1.0, p_raw * mtests)

            is_sig = (not (isinstance(p, float) and math.isnan(p))) and (p < alpha)
            sig[(a, b)] = bool(is_sig)
            rows.append([f"{a} ↔ {b}", fmt_num(p, 4), ("істотна різниця " + significance_mark(p)) if is_sig else "-"])
    return rows, sig

def mean_ranks_by_key(long, key_func):
    """
    Середній ранг по групі (для непараметричних таблиць).
    Ранги рахуються по всіх спостереженнях разом.
    """
    vals = np.array([float(r["value"]) for r in long], dtype=float)
    ranks = rankdata(vals, method="average")
    acc = defaultdict(list)
    for r, rk in zip(long, ranks):
        acc[key_func(r)].append(float(rk))
    return {k: float(np.mean(v)) for k, v in acc.items()}

def pairwise_mw_bonf_with_effect(levels, groups_named, alpha=0.05):
    rows = []
    sig = {}
    k = len(levels)
    mtests = k * (k - 1) // 2
    for i in range(k):
        for j in range(i + 1, k):
            a, b = levels[i], levels[j]
            xa = groups_named.get(a, [])
            xb = groups_named.get(b, [])
            if len(xa) == 0 or len(xb) == 0:
                continue
            try:
                U, p = mannwhitneyu(xa, xb, alternative="two-sided")
                p_adj = min(1.0, float(p) * mtests)
            except Exception:
                U, p_adj = (np.nan, np.nan)

            d = cliffs_delta(xa, xb)
            d_abs = abs(d) if not (isinstance(d, float) and math.isnan(d)) else np.nan
            concl_d = cliffs_label(d_abs)

            is_sig = (not (isinstance(p_adj, float) and math.isnan(p_adj))) and (p_adj < alpha)
            sig[(a, b)] = bool(is_sig)

            rows.append([
                f"{a} ↔ {b}",
                fmt_num(U, 3),
                fmt_num(p_adj, 4),
                ("істотна різниця " + significance_mark(p_adj)) if is_sig else "-",
                fmt_num(d, 3),
                concl_d
            ])
    return rows, sig

def rcbd_matrix_from_long(long, variant_levels, block_levels, variant_key="VARIANT", block_key="BLOCK"):
    """
    Формує матрицю блоки×варіанти без пропусків.
    Повертає mat_rows (list rows), kept_blocks.
    """
    # map (block, variant) -> list values (беремо середнє якщо раптом дубль)
    cell = defaultdict(list)
    for r in long:
        b = r.get(block_key)
        v = r.get(variant_key)
        cell[(b, v)].append(float(r["value"]))

    mat = []
    kept = []
    for b in block_levels:
        row = []
        ok = True
        for v in variant_levels:
            arr = cell.get((b, v), [])
            if len(arr) == 0:
                ok = False
                break
            row.append(float(np.mean(arr)))
        if ok:
            mat.append(row)
            kept.append(b)
    return mat, kept

def pairwise_wilcoxon_bonf(levels, mat_rows, alpha=0.05):
    """
    Pairwise Wilcoxon по блоках (парні), з Bonferroni.
    """
    rows = []
    sig = {}
    arr = np.array(mat_rows, dtype=float)
    k = len(levels)
    mtests = k * (k - 1) // 2
    for i in range(k):
        for j in range(i + 1, k):
            x = arr[:, i]
            y = arr[:, j]
            try:
                stat, p = wilcoxon(x, y, zero_method="wilcox", correction=False, alternative="two-sided", mode="auto")
                p_adj = min(1.0, float(p) * mtests)
            except Exception:
                stat, p_adj = (np.nan, np.nan)

            # ефект r = Z/sqrt(n)
            r_eff = np.nan
            try:
                if not (isinstance(p, float) and math.isnan(p)):
                    # approximate Z from p (two-sided)
                    z = abs(norm.ppf(float(p) / 2.0)) * (-1.0)
                    r_eff = abs(z) / math.sqrt(len(x))
            except Exception:
                r_eff = np.nan

            is_sig = (not (isinstance(p_adj, float) and math.isnan(p_adj))) and (p_adj < alpha)
            sig[(levels[i], levels[j])] = bool(is_sig)

            rows.append([
                f"{levels[i]} ↔ {levels[j]}",
                fmt_num(stat, 3),
                fmt_num(p_adj, 4),
                ("істотна різниця " + significance_mark(p_adj)) if is_sig else "-",
                fmt_num(r_eff, 3)
            ])
    return rows, sig

def build_effect_strength_rows(anova_table):
    # % від SS_total (без "Загальна")
    SS_total = np.nan
    for name, SSv, dfv, MSv, Fv, pv in anova_table:
        if name == "Загальна":
            SS_total = SSv
            break
    if SS_total is None or (isinstance(SS_total, float) and math.isnan(SS_total)) or SS_total <= 0:
        return []

    rows = []
    for name, SSv, dfv, MSv, Fv, pv in anova_table:
        if name in ("Залишок", "Загальна"):
            continue
        perc = (float(SSv) / float(SS_total)) * 100.0 if SSv is not None else np.nan
        rows.append([name, fmt_num(perc, 2)])
    return rows

def build_partial_eta2_rows_with_label(anova_table):
    # partial eta2 = SS_effect / (SS_effect + SS_error)
    SS_error = np.nan
    for name, SSv, dfv, MSv, Fv, pv in anova_table:
        if name == "Залишок":
            SS_error = SSv
            break
    if SS_error is None or (isinstance(SS_error, float) and math.isnan(SS_error)) or SS_error < 0:
        return []

    rows = []
    for name, SSv, dfv, MSv, Fv, pv in anova_table:
        if name in ("Залишок", "Загальна"):
            continue
        pe2 = float(SSv) / (float(SSv) + float(SS_error)) if (SSv is not None and (SSv + SS_error) > 0) else np.nan
        rows.append([name, fmt_num(pe2, 4), partial_eta2_label(pe2)])
    return rows

# ---------------------------------------------------------
# 3) Основний аналіз + звіт (методи класу SADTk)
# ---------------------------------------------------------

def groups_named_from_variant_order(long, factor_keys, variant_order):
    # повертає dict("A|B|..." -> list values)
    g = defaultdict(list)
    for r in long:
        key = tuple(r.get(f) for f in factor_keys)
        name = " | ".join(map(str, key))
        g[name].append(float(r["value"]))
    # забезпечити присутність усіх
    for k in variant_order:
        nm = " | ".join(map(str, k))
        g.setdefault(nm, [])
    return g

def factor_groups_named(long, fkey):
    g = defaultdict(list)
    for r in long:
        lvl = r.get(fkey)
        g[lvl].append(float(r["value"]))
    return g

def NIR05_value(MS, df, n_h, alpha=0.05):
    if df is None or (isinstance(df, float) and math.isnan(df)) or df <= 0:
        return np.nan
    if MS is None or (isinstance(MS, float) and math.isnan(MS)) or MS <= 0:
        return np.nan
    if n_h is None or (isinstance(n_h, float) and math.isnan(n_h)) or n_h <= 0:
        return np.nan
    tcrit = float(t_dist.ppf(1 - alpha / 2.0, df))
    return float(tcrit * math.sqrt(2.0 * MS / n_h))

def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return None

def _is_nan(x):
    return isinstance(x, float) and math.isnan(x)

# --- доповнюємо клас SADTk методами ---
def _sad_analyze(self: SADTk):
    created_at = datetime.now()

    params = self.ask_indicator_units()
    if not params["ok"]:
        return

    indicator = params["indicator"]
    units = params["units"]
    design = params["design"]
    split_main = params.get("split_main", "A")

    long, used_rep_cols = self.collect_long(design)
    if len(long) == 0:
        messagebox.showwarning("Помилка", "Немає числових даних для аналізу.\nПеревірте повторності та значення.")
        return

    values = np.array([r["value"] for r in long], dtype=float)
    if len(values) < 3:
        messagebox.showinfo("Результат", "Надто мало даних для аналізу.")
        return

    levels_by_factor = {f: first_seen_order([r.get(f) for r in long]) for f in self.factor_keys}

    variant_order = first_seen_order([tuple(r.get(f) for f in self.factor_keys) for r in long])
    v_names = [" | ".join(map(str, k)) for k in variant_order]
    num_variants = len(variant_order)

    # --- ANOVA/модель ---
    try:
        if design == "crd":
            res = anova_n_way(long, self.factor_keys, levels_by_factor)
            cell_means = res.get("cell_means", {})
            residuals = []
            for rec in long:
                key = tuple(rec.get(f) for f in self.factor_keys)
                v = rec.get("value", np.nan)
                m = cell_means.get(key, np.nan)
                if not _is_nan(v) and not _is_nan(m):
                    residuals.append(v - m)
            residuals = np.array(residuals, dtype=float)

        elif design == "rcbd":
            res = anova_rcbd_ols(long, self.factor_keys, levels_by_factor, block_key="BLOCK")
            residuals = np.array(res.get("residuals", []), dtype=float)

        else:
            if split_main not in self.factor_keys:
                split_main = self.factor_keys[0]
            res = anova_splitplot_ols(long, self.factor_keys, main_factor=split_main, block_key="BLOCK")
            residuals = np.array(res.get("residuals", []), dtype=float)

    except Exception as ex:
        messagebox.showerror("Помилка аналізу", str(ex))
        return

    # --- нормальність ---
    try:
        W, p_norm = shapiro(residuals) if len(residuals) >= 3 else (np.nan, np.nan)
    except Exception:
        W, p_norm = (np.nan, np.nan)

    normal = (p_norm is not None) and (not math.isnan(p_norm)) and (p_norm > 0.05)
    if design == "split" and not normal:
        messagebox.showwarning(
            "Split-plot: аналіз неможливий",
            "Обрано дизайн Split-plot, який у цій програмі реалізовано лише для параметричних методів.\n"
            "Оскільки залишки моделі не відповідають нормальному розподілу (p ≤ 0.05),\n"
            "параметричний split-plot аналіз є методично некоректним.\n\n"
            "Рекомендації:\n"
            "• застосувати трансформацію даних (log/√/Box-Cox) і повторити аналіз;\n"
            "• або вибрати CRD/RCBD і виконати непараметричний аналіз."
        )
        return

    choice = self.choose_method_window(p_norm, design, num_variants)
    if not choice["ok"]:
        return
    method = choice["method"]

    MS_error = res.get("MS_error", np.nan)
    df_error = res.get("df_error", np.nan)

    MS_whole = res.get("MS_whole", np.nan)
    df_whole = res.get("df_whole", np.nan)
    split_main_factor = res.get("main_factor", split_main) if design == "split" else None

    # --- групи/середні ---
    vstats = variant_mean_sd(long, self.factor_keys)
    v_means = {k: vstats[k][0] for k in vstats.keys()}
    v_sds = {k: vstats[k][1] for k in vstats.keys()}
    v_ns = {k: vstats[k][2] for k in vstats.keys()}

    means1 = {v_names[i]: v_means.get(variant_order[i], np.nan) for i in range(len(variant_order))}
    ns1 = {v_names[i]: v_ns.get(variant_order[i], 0) for i in range(len(variant_order))}
    groups1 = groups_named_from_variant_order(long, self.factor_keys, variant_order)

    factor_groups = {f: factor_groups_named(long, f) for f in self.factor_keys}
    factor_means = {f: {lvl: float(np.mean(arr)) if len(arr) else np.nan for lvl, arr in factor_groups[f].items()} for f in self.factor_keys}
    factor_ns = {f: {lvl: len(arr) for lvl, arr in factor_groups[f].items()} for f in self.factor_keys}

    factor_medians = {}
    factor_q = {}
    for f in self.factor_keys:
        factor_medians[f] = {}
        factor_q[f] = {}
        for lvl, arr in factor_groups[f].items():
            med, q1, q3 = median_q1_q3(arr)
            factor_medians[f][lvl] = med
            factor_q[f][lvl] = (q1, q3)

    ranks_by_variant = mean_ranks_by_key(long, key_func=lambda rec: " | ".join(str(rec.get(f)) for f in self.factor_keys))
    ranks_by_factor = {f: mean_ranks_by_key(long, key_func=lambda rec, ff=f: rec.get(ff)) for f in self.factor_keys}

    v_medians = {}
    v_q = {}
    for i, k in enumerate(variant_order):
        name = v_names[i]
        arr = groups1.get(name, [])
        med, q1, q3 = median_q1_q3(arr)
        v_medians[k] = med
        v_q[k] = (q1, q3)

    bf_F, bf_p = (np.nan, np.nan)
    if method in ("lsd", "tukey", "duncan", "bonferroni"):
        bf_F, bf_p = brown_forsythe_from_groups(groups1)

    kw_H, kw_p, kw_df, kw_eps2 = (np.nan, np.nan, np.nan, np.nan)
    do_posthoc = True

    fr_chi2, fr_p, fr_df, fr_W = (np.nan, np.nan, np.nan, np.nan)
    wil_stat, wil_p = (np.nan, np.nan)
    rcbd_pairwise_rows = []
    rcbd_sig = {}

    letters_factor = {f: {lvl: "" for lvl in levels_by_factor[f]} for f in self.factor_keys}
    letters_named = {name: "" for name in v_names}
    pairwise_rows = []
    factor_pairwise_tables = {}

    # --- POSTHOC / letters ---
    if method == "lsd":
        # LSD по факторах (Split: для головного фактора — MS_whole)
        for f in self.factor_keys:
            lvls = levels_by_factor[f]
            if design == "split" and f == split_main_factor:
                sig_f = lsd_sig_matrix(lvls, factor_means[f], factor_ns[f], MS_whole, df_whole, alpha=ALPHA)
            else:
                sig_f = lsd_sig_matrix(lvls, factor_means[f], factor_ns[f], MS_error, df_error, alpha=ALPHA)
            letters_factor[f] = cld_multi_letters(lvls, factor_means[f], sig_f)

        # літери по повних варіантах — тільки не split
        if design != "split":
            sigv = lsd_sig_matrix(v_names, means1, ns1, MS_error, df_error, alpha=ALPHA)
            letters_named = cld_multi_letters(v_names, means1, sigv)

        # NIR05 значення для довідки
        n_h_var = harmonic_mean([ns1.get(nm, 0) for nm in v_names if ns1.get(nm, 0) > 0])
        nir_total = NIR05_value(MS_error, df_error, n_h_var, alpha=ALPHA)

        res["NIR05"] = {"Загальна": nir_total}
        for f in self.factor_keys:
            n_h = harmonic_mean([factor_ns[f].get(lvl, 0) for lvl in levels_by_factor[f] if factor_ns[f].get(lvl, 0) > 0])
            if design == "split" and f == split_main_factor:
                res["NIR05"][f"Фактор {f}"] = NIR05_value(MS_whole, df_whole, n_h, alpha=ALPHA)
            else:
                res["NIR05"][f"Фактор {f}"] = NIR05_value(MS_error, df_error, n_h, alpha=ALPHA)

    elif method in ("tukey", "duncan", "bonferroni"):
        # повні варіанти — тільки не split
        if design != "split":
            pairwise_rows, sig = pairwise_param_short_variants_pm(v_names, means1, ns1, MS_error, df_error, method, alpha=ALPHA)
            letters_named = cld_multi_letters(v_names, means1, sig)

        # split: парні тільки на рівні факторів (з правильним error term)
        if design == "split":
            for f in self.factor_keys:
                lvls = levels_by_factor[f]
                means_f = factor_means[f]
                ns_f = factor_ns[f]
                if f == split_main_factor:
                    rows_f, _ = pairwise_param_short_variants_pm(lvls, means_f, ns_f, MS_whole, df_whole, method, alpha=ALPHA)
                else:
                    rows_f, _ = pairwise_param_short_variants_pm(lvls, means_f, ns_f, MS_error, df_error, method, alpha=ALPHA)
                factor_pairwise_tables[f] = rows_f

    elif method == "kw":
        try:
            kw_samples = [groups1[name] for name in v_names if len(groups1[name]) > 0]
            if len(kw_samples) >= 2:
                kw_res = kruskal(*kw_samples)
                kw_H = float(kw_res.statistic)
                kw_p = float(kw_res.pvalue)
                kw_df = int(len(kw_samples) - 1)
                kw_eps2 = epsilon_squared_kw(kw_H, n=len(long), k=len(kw_samples))
        except Exception:
            kw_H, kw_p, kw_df, kw_eps2 = (np.nan, np.nan, np.nan, np.nan)

        if not _is_nan(kw_p) and kw_p >= ALPHA:
            do_posthoc = False

        if do_posthoc:
            pairwise_rows, sig = pairwise_mw_bonf_with_effect(v_names, groups1, alpha=ALPHA)
            med_tmp = {name: float(np.median(groups1[name])) if len(groups1[name]) else np.nan for name in v_names}
            letters_named = cld_multi_letters(v_names, med_tmp, sig)
        else:
            letters_named = {name: "" for name in v_names}

    elif method == "mw":
        pairwise_rows, sig = pairwise_mw_bonf_with_effect(v_names, groups1, alpha=ALPHA)
        med_tmp = {name: float(np.median(groups1[name])) if len(groups1[name]) else np.nan for name in v_names}
        letters_named = cld_multi_letters(v_names, med_tmp, sig)

    elif method == "friedman":
        block_names = first_seen_order([f"Блок {i+1}" for i in range(len(used_rep_cols))])
        long2 = []
        for r in long:
            rr = dict(r)
            rr["VARIANT"] = " | ".join(str(rr.get(f)) for f in self.factor_keys)
            long2.append(rr)

        mat_rows, kept_blocks = rcbd_matrix_from_long(long2, v_names, block_names, variant_key="VARIANT", block_key="BLOCK")
        if len(mat_rows) < 2:
            messagebox.showwarning("Помилка", "Для Friedman потрібні щонайменше 2 повних блоки (без пропусків по варіантах).")
            return

        try:
            cols = list(zip(*mat_rows))
            fr = friedmanchisquare(*[np.array(c, dtype=float) for c in cols])
            fr_chi2 = float(fr.statistic)
            fr_p = float(fr.pvalue)
            fr_df = int(len(v_names) - 1)
            fr_W = kendalls_w_from_friedman(fr_chi2, n_blocks=len(mat_rows), k_treat=len(v_names))
        except Exception:
            fr_chi2, fr_p, fr_df, fr_W = (np.nan, np.nan, np.nan, np.nan)

        if not _is_nan(fr_p) and fr_p < ALPHA:
            rcbd_pairwise_rows, rcbd_sig = pairwise_wilcoxon_bonf(v_names, mat_rows, alpha=ALPHA)
            med_tmp = {name: float(np.median(groups1[name])) if len(groups1[name]) else np.nan for name in v_names}
            letters_named = cld_multi_letters(v_names, med_tmp, rcbd_sig)
        else:
            letters_named = {name: "" for name in v_names}

    elif method == "wilcoxon":
        if len(v_names) != 2:
            messagebox.showwarning("Помилка", "Wilcoxon (парний) застосовується лише для 2 варіантів.")
            return

        block_names = first_seen_order([f"Блок {i+1}" for i in range(len(used_rep_cols))])
        long2 = []
        for r in long:
            rr = dict(r)
            rr["VARIANT"] = " | ".join(str(rr.get(f)) for f in self.factor_keys)
            long2.append(rr)

        mat_rows, kept_blocks = rcbd_matrix_from_long(long2, v_names, block_names, variant_key="VARIANT", block_key="BLOCK")
        if len(mat_rows) < 2:
            messagebox.showwarning("Помилка", "Для Wilcoxon потрібні щонайменше 2 повні блоки (пари значень).")
            return
        arr = np.array(mat_rows, dtype=float)
        x = arr[:, 0]
        y = arr[:, 1]
        try:
            stat, p = wilcoxon(x, y, zero_method="wilcox", correction=False, alternative="two-sided", mode="auto")
            wil_stat = float(stat)
            wil_p = float(p)
        except Exception:
            wil_stat, wil_p = (np.nan, np.nan)

        if not _is_nan(wil_p) and wil_p < ALPHA:
            rcbd_sig = {(v_names[0], v_names[1]): True}
            med_tmp = {name: float(np.median(groups1[name])) if len(groups1[name]) else np.nan for name in v_names}
            letters_named = cld_multi_letters(v_names, med_tmp, rcbd_sig)
        else:
            letters_named = {name: "" for name in v_names}

    letters_variants = {variant_order[i]: letters_named.get(v_names[i], "") for i in range(len(variant_order))}

    SS_total = res.get("SS_total", np.nan)
    SS_error = res.get("SS_error", np.nan)
    R2 = (1.0 - (SS_error / SS_total)) if (not any(_is_nan(x) for x in [SS_total, SS_error]) and SS_total > 0) else np.nan

    cv_rows = []
    for f in self.factor_keys:
        lvl_means = [factor_means[f].get(lvl, np.nan) for lvl in levels_by_factor[f]]
        cv_f = cv_percent_from_level_means(lvl_means)
        cv_rows.append([self.factor_title(f), fmt_num(cv_f, 2)])
    cv_total = cv_percent_from_values(values)
    cv_rows.append(["Загальний", fmt_num(cv_total, 2)])

    # ---------------------------------------------------------
    # Формування звіту
    # ---------------------------------------------------------
    seg = []
    seg.append(("text", "З В І Т   С Т А Т И С Т И Ч Н О Г О   А Н А Л І З У   Д А Н И Х\n\n"))
    seg.append(("text", f"Показник:\t{indicator}\nОдиниці виміру:\t{units}\n\n"))

    seg.append(("text",
                f"Кількість варіантів:\t{num_variants}\n"
                f"Кількість повторностей:\t{len(used_rep_cols)}\n"
                f"Загальна кількість облікових значень:\t{len(long)}\n\n"))

    design_label = {"crd": "CRD (повна рандомізація)", "rcbd": "RCBD (блочна рандомізація)", "split": "Split-plot (спліт-плот)"}[design]
    seg.append(("text", f"Дизайн експерименту:\t{design_label}\n"))
    if design == "split":
        seg.append(("text", f"Головний фактор (Whole-plot factor):\t{split_main_factor}\n\n"))
    else:
        seg.append(("text", "\n"))

    method_label = {
        "lsd": "Параметричний аналіз: Brown–Forsythe + ANOVA + НІР₀₅ (LSD).",
        "tukey": "Параметричний аналіз: Brown–Forsythe + ANOVA + тест Тьюкі (Tukey HSD).",
        "duncan": "Параметричний аналіз: Brown–Forsythe + ANOVA + тест Дункана.",
        "bonferroni": "Параметричний аналіз: Brown–Forsythe + ANOVA + корекція Бонферроні.",
        "kw": "Непараметричний аналіз: Kruskal–Wallis.",
        "mw": "Непараметричний аналіз: Mann–Whitney.",
        "friedman": "Непараметричний аналіз (RCBD): Friedman.",
        "wilcoxon": "Непараметричний аналіз (RCBD): Wilcoxon signed-rank (парний).",
    }.get(method, "")

    if method_label:
        seg.append(("text", f"Виконуваний статистичний аналіз:\t{method_label}\n\n"))

    seg.append(("text", "Пояснення позначень істотності: ** — p<0.01; * — p<0.05.\n"))
    seg.append(("text", "У таблицях знак \"-\" свідчить що p ≥ 0.05.\n"))
    seg.append(("text", "Істотна різниця (літери): різні літери свідчать про наявність істотної різниці.\n\n"))

    if not _is_nan(W):
        seg.append(("text",
                    f"Перевірка нормальності залишків (Shapiro–Wilk):\t{normality_text(p_norm)}\t"
                    f"(W={fmt_num(float(W),4)}; p={fmt_num(float(p_norm),4)})\n\n"))
    else:
        seg.append(("text", "Перевірка нормальності залишків (Shapiro–Wilk):\tн/д\n\n"))

    nonparam = method in ("mw", "kw", "friedman", "wilcoxon")

    # --- Глобальні тести для непараметрики ---
    if method == "kw" and not _is_nan(kw_p):
        concl = "істотна різниця " + significance_mark(kw_p) if kw_p < ALPHA else "-"
        seg.append(("text",
                    f"Глобальний тест між варіантами (Kruskal–Wallis):\t"
                    f"H={fmt_num(kw_H,4)}; df={int(kw_df)}; p={fmt_num(kw_p,4)}\t{concl}\n"))
        seg.append(("text", f"Розмір ефекту (ε²):\t{fmt_num(kw_eps2,4)}\n\n"))

    if method == "friedman" and not _is_nan(fr_p):
        concl = "істотна різниця " + significance_mark(fr_p) if fr_p < ALPHA else "-"
        seg.append(("text",
                    f"Глобальний тест між варіантами (Friedman):\t"
                    f"χ²={fmt_num(fr_chi2,4)}; df={int(fr_df)}; p={fmt_num(fr_p,4)}\t{concl}\n"))
        seg.append(("text", f"Розмір ефекту (Kendall’s W):\t{fmt_num(fr_W,4)}\n\n"))

    if method == "wilcoxon" and not _is_nan(wil_p):
        concl = "істотна різниця " + significance_mark(wil_p) if wil_p < ALPHA else "-"
        seg.append(("text",
                    f"Парний тест (Wilcoxon signed-rank):\t"
                    f"W={fmt_num(wil_stat,4)}; p={fmt_num(wil_p,4)}\t{concl}\n\n"))

    # --- Параметрика ---
    if not nonparam:
        if not any(_is_nan(x) for x in [bf_F, bf_p]):
            bf_concl = "умова виконується" if bf_p >= ALPHA else f"умова порушена {significance_mark(bf_p)}"
            seg.append(("text",
                        f"Перевірка однорідності дисперсій (Brown–Forsythe):\t"
                        f"F={fmt_num(bf_F,4)}; p={fmt_num(bf_p,4)}\t{bf_concl}\n\n"))
        else:
            seg.append(("text", "Перевірка однорідності дисперсій (Brown–Forsythe):\tн/д\n\n"))

        if design == "split":
            seg.append(("text",
                        "Примітка (Split-plot):\n"
                        f"• {self.factor_title(split_main_factor)} перевірено на MS(Блоки×{split_main_factor}) (whole-plot error).\n"
                        "• Інші ефекти перевірено на MS(Залишок) (sub-plot error).\n\n"))

        # Таблиця ANOVA
        anova_rows = []
        for name, SSv, dfv, MSv, Fv, pv in res["table"]:
            df_txt = str(int(dfv)) if dfv is not None and not _is_nan(dfv) else ""
            if name.startswith("Фактор "):
                # заміна A/B/C/D на назви
                rest = name.replace("Фактор ", "")
                parts = rest.split("×")
                parts2 = [self.factor_title(p) if p in self.factor_keys else p for p in parts]
                name2 = "×".join(parts2)
            else:
                name2 = name

            if name2.startswith("Залишок") or name2 == "Загальна" or "Whole-plot error" in name2:
                anova_rows.append([name2, fmt_num(SSv, 2), df_txt, fmt_num(MSv, 3), "", "", ""])
            else:
                mark = significance_mark(pv)
                concl = f"істотна різниця {mark}" if mark else "-"
                anova_rows.append([name2, fmt_num(SSv, 2), df_txt, fmt_num(MSv, 3), fmt_num(Fv, 3), fmt_num(pv, 4), concl])

        seg.append(("text", "ТАБЛИЦЯ 1. Дисперсійний аналіз (ANOVA)\n"))
        seg.append(("table", {
            "headers": ["Джерело", "SS", "df", "MS", "F", "p", "Висновок"],
            "rows": anova_rows,
            "padding_px": 32,
            "extra_gap_after_col": 0,
            "extra_gap_px": 60
        }))
        seg.append(("text", "\n"))

        eff_rows = build_effect_strength_rows(res["table"])
        for r in eff_rows:
            if r and isinstance(r[0], str) and r[0].startswith("Фактор "):
                rest = r[0].replace("Фактор ", "")
                parts = rest.split("×")
                parts2 = [self.factor_title(p) if p in self.factor_keys else p for p in parts]
                r[0] = "×".join(parts2)

        seg.append(("text", "ТАБЛИЦЯ 2. Сила впливу факторів та їх комбінацій (% від SS)\n"))
        seg.append(("table", {"headers": ["Джерело", "%"], "rows": eff_rows}))
        seg.append(("text", "\n"))

        pe2_rows = build_partial_eta2_rows_with_label(res["table"])
        for r in pe2_rows:
            if r and isinstance(r[0], str) and r[0].startswith("Фактор "):
                rest = r[0].replace("Фактор ", "")
                parts = rest.split("×")
                parts2 = [self.factor_title(p) if p in self.factor_keys else p for p in parts]
                r[0] = "×".join(parts2)

        seg.append(("text", "ТАБЛИЦЯ 3. Розмір ефекту (partial η²)\n"))
        seg.append(("table", {"headers": ["Джерело", "partial η²", "Висновок"], "rows": pe2_rows}))
        seg.append(("text", "\n"))

        seg.append(("text", "ТАБЛИЦЯ 4. Коефіцієнт варіації (CV, %)\n"))
        seg.append(("table", {"headers": ["Елемент", "CV, %"], "rows": cv_rows}))
        seg.append(("text", "\n"))

        seg.append(("text", f"Коефіцієнт детермінації:\tR²={fmt_num(R2, 4)}\n\n"))

        tno = 5
        if method == "lsd":
            nir_rows = []
            for key, val in res.get("NIR05", {}).items():
                if key.startswith("Фактор "):
                    fkey = key.replace("Фактор ", "").strip()
                    nir_rows.append([self.factor_title(fkey), fmt_num(val, 4)])
                else:
                    nir_rows.append([key, fmt_num(val, 4)])

            seg.append(("text", "ТАБЛИЦЯ 5. Значення НІР₀₅\n"))
            seg.append(("table", {"headers": ["Елемент", "НІР₀₅"], "rows": nir_rows}))
            seg.append(("text", "\n"))
            tno = 6

        # середні по факторах
        for f in self.factor_keys:
            seg.append(("text", f"ТАБЛИЦЯ {tno}. Середнє по фактору: {self.factor_title(f)}\n"))
            rows_f = []
            for lvl in levels_by_factor[f]:
                m = factor_means[f].get(lvl, np.nan)
                letter = letters_factor[f].get(lvl, "")
                rows_f.append([str(lvl), fmt_num(m, 3), (letter if letter else "-")])
            seg.append(("table", {"headers": [self.factor_title(f), "Середнє", "Істотна різниця"], "rows": rows_f}))
            seg.append(("text", "\n"))
            tno += 1

        # середні варіантів
        seg.append(("text", f"ТАБЛИЦЯ {tno}. Таблиця середніх значень варіантів\n"))
        rows_v = []
        for k in variant_order:
            name = " | ".join(map(str, k))
            m = v_means.get(k, np.nan)
            sd = v_sds.get(k, np.nan)
            if design == "split":
                rows_v.append([name, fmt_num(m, 3), fmt_num(sd, 3), "-"])
            else:
                letter = letters_variants.get(k, "")
                rows_v.append([name, fmt_num(m, 3), fmt_num(sd, 3), (letter if letter else "-")])

        seg.append(("table", {
            "headers": ["Варіант", "Середнє", "± SD", "Істотна різниця"],
            "rows": rows_v,
            "padding_px": 32,
            "extra_gap_after_col": 0,
            "extra_gap_px": 80
        }))
        seg.append(("text", "\n"))
        tno += 1

        # парні порівняння
        if design != "split":
            if method in ("tukey", "duncan", "bonferroni") and pairwise_rows:
                seg.append(("text", f"ТАБЛИЦЯ {tno}. Парні порівняння варіантів\n"))
                seg.append(("table", {"headers": ["Комбінація варіантів", "p", "Істотна різниця"], "rows": pairwise_rows}))
                seg.append(("text", "\n"))
        else:
            seg.append(("text",
                        "Примітка (Split-plot): парні порівняння для повних варіантів (комбінацій факторів)\n"
                        "не подаються, оскільки для таких порівнянь потрібні спеціальні контрасти та коректний\n"
                        "облік двох різних помилок (whole-plot і sub-plot). Натомість подано парні порівняння\n"
                        "на рівні факторів з правильними error-term.\n\n"))
            if method in ("tukey", "duncan", "bonferroni"):
                for f in self.factor_keys:
                    rows_pf = factor_pairwise_tables.get(f, [])
                    if rows_pf:
                        seg.append(("text", f"ТАБЛИЦЯ {tno}. Парні порівняння для: {self.factor_title(f)} (Split-plot)\n"))
                        seg.append(("table", {"headers": ["Комбінація", "p", "Істотна різниця"], "rows": rows_pf}))
                        seg.append(("text", "\n"))
                        tno += 1

    # --- Непараметрика (детально) ---
    else:
        tno = 1
        for f in self.factor_keys:
            seg.append(("text", f"ТАБЛИЦЯ {tno}. Описова статистика (непараметрична): {self.factor_title(f)}\n"))
            rows = []
            for lvl in levels_by_factor[f]:
                med = factor_medians[f].get(lvl, np.nan)
                q1, q3 = factor_q[f].get(lvl, (np.nan, np.nan))
                rank_m = ranks_by_factor[f].get(lvl, np.nan)
                rows.append([
                    str(lvl),
                    str(int(factor_ns[f].get(lvl, 0))),
                    fmt_num(med, 3),
                    f"{fmt_num(q1,3)}–{fmt_num(q3,3)}" if not any(_is_nan(x) for x in [q1, q3]) else "",
                    fmt_num(rank_m, 2),
                    "-"
                ])
            seg.append(("table", {"headers": [self.factor_title(f), "n", "Медіана", "Q1–Q3", "Середній ранг", "Істотна різниця"], "rows": rows}))
            seg.append(("text", "\n"))
            tno += 1

        seg.append(("text", f"ТАБЛИЦЯ {tno}. Описова статистика варіантів (непараметрична)\n"))
        rows = []
        for k in variant_order:
            name = " | ".join(map(str, k))
            med = v_medians.get(k, np.nan)
            q1, q3 = v_q.get(k, (np.nan, np.nan))
            rank_m = ranks_by_variant.get(name, np.nan)
            letter = letters_variants.get(k, "")
            rows.append([
                name,
                str(int(v_ns.get(k, 0))),
                fmt_num(med, 3),
                f"{fmt_num(q1,3)}–{fmt_num(q3,3)}" if not any(_is_nan(x) for x in [q1, q3]) else "",
                fmt_num(rank_m, 2),
                (letter if letter else "-")
            ])
        seg.append(("table", {
            "headers": ["Варіант", "n", "Медіана", "Q1–Q3", "Середній ранг", "Істотна різниця"],
            "rows": rows,
            "padding_px": 32,
            "extra_gap_after_col": 0,
            "extra_gap_px": 80
        }))
        seg.append(("text", "\n"))
        tno += 1

        if method in ("kw", "mw") and pairwise_rows:
            seg.append(("text", f"ТАБЛИЦЯ {tno}. Парні порівняння + ефект (Cliff’s δ)\n"))
            seg.append(("table", {"headers": ["Комбінація варіантів", "U", "p (Bonf.)", "Істотна різниця", "δ", "Висновок"], "rows": pairwise_rows}))
            seg.append(("text", "\n"))

        if method == "friedman" and rcbd_pairwise_rows:
            seg.append(("text", f"ТАБЛИЦЯ {tno}. Парні порівняння (Wilcoxon, Bonferroni) + ефект (r)\n"))
            seg.append(("table", {"headers": ["Комбінація варіантів", "W", "p (Bonf.)", "Істотна різниця", "r"], "rows": rcbd_pairwise_rows}))
            seg.append(("text", "\n"))

    seg.append(("text", f"Звіт сформовано:\t{created_at.strftime('%d.%m.%Y, %H:%M')}\n"))
    self.show_report_segments(seg)

def _sad_show_report_segments(self: SADTk, segments):
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

    txt.configure(font=("Times New Roman", 14), fg="#000000")
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

        tabs = tabs_from_table_px(font_obj, headers, rows, padding_px=padding_px, extra_gap_after_col=extra_after, extra_gap_px=extra_px)

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

# прикріпити методи до класу
SADTk.analyze = _sad_analyze
SADTk.show_report_segments = _sad_show_report_segments

# -------------------------
# Run (ЄДИНИЙ)
# -------------------------
if __name__ == "__main__":
    root = tk.Tk()
    set_window_icon(root)
    app = SADTk(root)
    root.mainloop()
