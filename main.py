# -*- coding: utf-8 -*-
"""
DataAn Enhanced — CLI-версия
Интерактивный командный строковый статистический анализ на базе DataAnalyzer.
"""
import sys
import glob
import fnmatch
import webbrowser
import warnings
import shutil
import pandas as pd
import importlib.util
import os
warnings.filterwarnings('ignore')

def load_analyzer():
    if getattr(sys, 'frozen', False):
        # Запущено из собранного EXE — файл лежит в распакованной папке
        base_path = sys._MEIPASS
    else:
        # Обычный запуск из интерпретатора — файл рядом
        base_path = os.path.dirname(os.path.abspath(__file__))

    module_path = os.path.join(base_path, 'analyzer_enhanced.py')
    spec = importlib.util.spec_from_file_location('analyzer_enhanced', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

analyzer_enhanced = load_analyzer()
DataAnalyzer = analyzer_enhanced.DataAnalyzer

# ─────────────────── ANSI цвета для терминала ───────────────────
class C:
    """ANSI escape-коды для цветового выделения."""
    _enabled = True

    @classmethod
    def disable(cls):
        cls._enabled = False

    @staticmethod
    def _code(code):
        if not C._enabled:
            return ''
        return f'\033[{code}m'

    @staticmethod
    def bold(s):
        return f'{C._code(1)}{s}{C._code(0)}'

    @staticmethod
    def green(s):
        return f'{C._code(1)}{C._code(32)}{s}{C._code(0)}'

    @staticmethod
    def yellow(s):
        return f'{C._code(1)}{C._code(33)}{s}{C._code(0)}'

    @staticmethod
    def red(s):
        return f'{C._code(1)}{C._code(31)}{s}{C._code(0)}'

    @staticmethod
    def cyan(s):
        return f'{C._code(1)}{C._code(36)}{s}{C._code(0)}'

    @staticmethod
    def blue(s):
        return f'{C._code(1)}{C._code(34)}{s}{C._code(0)}'

    @staticmethod
    def dim(s):
        return f'{C._code(2)}{s}{C._code(0)}'

# ─────────────────────────── вспомогательные ───────────────────────────
def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


def term_width():
    try:
        return shutil.get_terminal_size((80, 24)).columns
    except Exception:
        return 80


def pick_file_from_args_or_menu():
    """Выбор файла: из аргумента CLI или интерактивно из текущего каталога."""
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isfile(path):
            return path
        print(f"  Файл не найден: {path}")
        sys.exit(1)

    xlsx = sorted(glob.glob('*.xlsx')) + sorted(glob.glob('*.xls'))
    csv = sorted(glob.glob('*.csv'))
    files = xlsx + csv

    if not files:
        print("  В текущем каталоге нет файлов .xlsx / .csv")
        sys.exit(1)

    print("\n  Доступные файлы:\n")
    for i, f in enumerate(files, 1):
        size = os.path.getsize(f)
        if size > 1_048_576:
            size_str = f"{size / 1_048_576:.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.0f} KB"
        else:
            size_str = f"{size} B"
        print(f"    {i:>3}. {f:<40s} ({size_str})")

    while True:
        try:
            choice = input("\n  Выберите номер файла (Enter — выход): ").strip()
            if not choice:
                sys.exit(0)
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                return files[idx]
            print("  Неверный номер, попробуйте снова.")
        except ValueError:
            print("  Введите число.")


def show_columns_grid(df):
    """Вывод столбцов в компактной многоколоночной сетке."""
    cols = list(df.columns)
    n = len(cols)
    tw = term_width()

    entries = []
    for i, col in enumerate(cols, 1):
        nuniq = df[col].nunique()
        miss = df[col].isna().sum()
        miss_s = f" *{miss}" if miss > 0 else ""

        if pd.api.types.is_numeric_dtype(df[col]) and nuniq >= 15:
            try:
                mn, mx = df[col].min(), df[col].max()
                tag = f"({mn:.1f}..{mx:.1f}){miss_s}"
            except Exception:
                tag = f"({nuniq} значений){miss_s}"
        else:
            vals = df[col].dropna().unique()
            if len(vals) <= 4:
                tag = f"[{', '.join(str(v) for v in vals)}]{miss_s}"
            else:
                tag = f"({nuniq} значений){miss_s}"

        entries.append(f"{i:>3}. {col} {tag}")

    max_len = max(len(e) for e in entries) + 2
    ncols = max(1, tw // max_len)
    nrows = (n + ncols - 1) // ncols

    print(f"\n  Столбцы ({n}):  * = есть пропуски\n")
    for row in range(nrows):
        parts = []
        for col_idx in range(ncols):
            idx = row + col_idx * nrows
            if idx < len(entries):
                parts.append(entries[idx].ljust(max_len))
        print("    " + "  ".join(parts))


def parse_selection(raw, total, df=None):
    """
    Разбор строки выбора в список индексов (0-based).
    Поддержка:  1,3,5 | 1-5 | 1-3,7,9-12 | * | temp* | *score
    """
    raw = raw.strip()
    if not raw:
        return None

    if raw in ('all', '*'):
        return list(range(total))

    tokens = [t.strip() for t in raw.split(',') if t.strip()]
    name_indices = []
    has_names = False

    for tok in tokens:
        # Убрать минус в начале для проверки — это может быть паттерн
        stripped = tok.lstrip('-')
        if stripped.replace('-', '').replace('.', '').isdigit():
            pass  # числовой режим
        elif df is not None:
            has_names = True
            matched = [i for i, col in enumerate(df.columns)
                       if fnmatch.fnmatch(col.lower(), tok.lower())]
            if matched:
                name_indices.extend(matched)
            else:
                print(f"  Столбцы по шаблону «{tok}» не найдены")
                return []

    if has_names:
        return sorted(set(name_indices))

    # Числовой режим: диапазоны и запятые
    indices = []
    exclude = []
    for tok in tokens:
        is_exclude = tok.startswith('-') and len(tok) > 1 and tok[1:].isdigit()
        core = tok.lstrip('-') if is_exclude else tok

        if '-' in core:
            parts = core.split('-', 1)
            try:
                a = int(parts[0].strip()) - 1
                b = int(parts[1].strip()) - 1
                if a > b:
                    a, b = b, a
                rng = list(range(a, b + 1))
                if is_exclude:
                    exclude.extend(rng)
                else:
                    indices.extend(rng)
            except ValueError:
                print(f"  Неверный диапазон: {tok}")
                return []
        else:
            try:
                idx = int(core.strip()) - 1
                if is_exclude:
                    exclude.append(idx)
                else:
                    indices.append(idx)
            except ValueError:
                print(f"  Неверное значение: {tok}")
                return []

    result = sorted(set(i for i in indices if 0 <= i < total and i not in exclude))
    invalid = [i + 1 for i in indices if i < 0 or i >= total]
    if invalid:
        print(f"  Вне диапазона: {invalid}")
    return result


def pick_columns(df, prompt, allow_empty=False, preselected=None, hint_extra="",
                 dtype_filter=None):
    """
    Интерактивный выбор столбцов. Поддерживает диапазоны, маски, запятые.
    Возвращает список имён столбцов.
    dtype_filter: 'numeric' — только числовые, 'categorical' — только категориальные.
    """
    if preselected:
        default_str = ', '.join(preselected)
        hint = f"номера/диапазоны, маска name*, Enter = [{default_str}]"
    else:
        hint = "номера/диапазоны, маска name*, Enter = пропустить"
    if allow_empty:
        hint += ", * = все, - = пропустить"
    if hint_extra:
        hint += f"\n  {hint_extra}"

    type_label = ""
    if dtype_filter == 'numeric':
        type_label = " [только числовые]"
        available_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    elif dtype_filter == 'categorical':
        type_label = " [только категориальные]"
        available_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    else:
        available_cols = df.columns.tolist()

    print(f"\n  {prompt}{type_label}")
    print(f"  (Совет: {hint})")

    while True:
        raw = input("  > ").strip()

        if not raw and preselected is not None:
            print(f"    -> {C.dim('по умолчанию:')} {', '.join(preselected)}")
            return preselected
        if not raw and allow_empty:
            return []
        if raw in ('-', 'нет', 'skip'):
            if allow_empty:
                return []
            return df.select_dtypes(include=['number']).columns.tolist()
        if raw in ('*', 'all', 'все'):
            if dtype_filter:
                print(f"    -> {C.dim('доступные:')} {', '.join(available_cols)}")
                return available_cols
            return df.columns.tolist()

        indices = parse_selection(raw, len(df.columns), df)
        if indices is None and preselected is not None:
            print(f"    -> {C.dim('по умолчанию:')} {', '.join(preselected)}")
            return preselected
        if indices is not None and len(indices) > 0:
            chosen = [df.columns[i] for i in indices]
            # Валидация типа
            if dtype_filter == 'numeric':
                invalid = [c for c in chosen if not pd.api.types.is_numeric_dtype(df[c])]
                valid = [c for c in chosen if pd.api.types.is_numeric_dtype(df[c])]
                if invalid:
                    print(f"  {C.red('ОШИБКА:')} нечисловые столбцы отброшены: {', '.join(invalid)}")
                if not valid:
                    print(f"  {C.red('ОШИБКА:')} нет корректных числовых столбцов. Попробуйте снова.")
                    continue
                print(f"    -> {C.green('выбрано:')} {', '.join(valid)}")
                return valid
            elif dtype_filter == 'categorical':
                invalid = [c for c in chosen if pd.api.types.is_numeric_dtype(df[c])]
                valid = [c for c in chosen if not pd.api.types.is_numeric_dtype(df[c])]
                if invalid:
                    print(f"  {C.red('ОШИБКА:')} числовые столбцы отброшены: {', '.join(invalid)}")
                if not valid:
                    print(f"  {C.red('ОШИБКА:')} нет корректных категориальных столбцов. Попробуйте снова.")
                    continue
                print(f"    -> {C.green('выбрано:')} {', '.join(valid)}")
                return valid
            else:
                print(f"    -> {C.green('выбрано:')} {', '.join(chosen)}")
                return chosen
        if indices is not None and len(indices) == 0:
            print(f"  {C.yellow('Ничего не выбрано.')} Попробуйте снова.")
# ─────────────────── matplotlib без отображения окон ───────────────────

def setup_no_display():
    """Matplotlib работает в фоновом режиме — графики только в буфер."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.show = lambda *a, **kw: None


# ─────────────────── проверка поддержки цветов ───────────────────
def setup_colors():
    """Включаем ANSI-цвета только для TTY."""
    if not sys.stdout.isatty():
        C.disable()


# ─────────────────────────── основная программа ───────────────────────────
def main():
    clear()
    setup_no_display()
    setup_colors()

    print("=" * 60)
    print(f"  {C.cyan('DataAn Enhanced v1.0')} — CLI-версия статистического анализа")
    print("=" * 60)

    # 1. Выбор файла
    filepath = pick_file_from_args_or_menu()
    print(f"\n  {C.dim('Загрузка:')} {filepath}")

    if filepath.lower().endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)

    print(f"  {C.green('Загружено:')} {len(df):,} строк, {len(df.columns)} столбцов")

    # Краткая сводка
    n_num = len(df.select_dtypes(include=['number']).columns)
    n_cat = len(df.select_dtypes(exclude=['number']).columns)
    total_miss = df.isnull().sum().sum()
    print(f"\n  {C.blue('Числовых:')} {n_num}  |  {C.blue('Категориальных:')} {n_cat}", end="")
    if total_miss:
        print(f"  |  {C.yellow('Пропусков:')} {total_miss} ({100 * total_miss / df.size:.1f}%)")
    else:
        print()

    show_columns_grid(df)

    # 2. Выбор переменных
    print("\n" + "=" * 60)
    print(f"  {C.cyan('ВЫБОР ПЕРЕМЕННЫХ ДЛЯ АНАЛИЗА')}")
    print("=" * 60)

    cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()

    if not cat_cols:
        print(f"\n  {C.red('ОШИБКА:')} нет категориальных столбцов. Анализ невозможен.")
        sys.exit(1)
    if not num_cols:
        print(f"\n  {C.red('ОШИБКА:')} нет числовых столбцов. Анализ невозможен.")
        sys.exit(1)

    # Восстановление последней сессии
    session_data = DataAnalyzer.load_session()
    same_file = session_data.get('last_file_path', '') == filepath
    last_params = session_data.get('last_params', {}) if same_file else {}

    def _session_preset(col_type, default):
        val = last_params.get(col_type)
        if val and isinstance(val, list):
            return [c for c in val if c in df.columns]
        return default

    # Группирующая переменная
    group_preset = [last_params.get('group')] if same_file and last_params.get('group') in cat_cols else [cat_cols[0]]
    group = pick_columns(df,
        "Группирующая переменная (категориальная) — по ней сравниваются группы",
        preselected=group_preset, dtype_filter='categorical')
    group_col = group[0]

    # Целевая переменная
    analysis_preset = [last_params.get('analysis')] if same_file and last_params.get('analysis') in num_cols else [num_cols[0]]
    analysis = pick_columns(df,
        "Целевая переменная Y (числовая) — основная мера для анализа",
        preselected=analysis_preset, dtype_filter='numeric')
    analysis_col = analysis[0]

    # Многомерные
    default_multi = [c for c in num_cols[:min(5, len(num_cols))] if c != analysis_col]
    multi_preset = _session_preset('multi', default_multi if default_multi else num_cols[:min(5, len(num_cols))])
    multi = pick_columns(df,
        "Многомерные признаки X (числовые) — для ANOVA, регрессии, кластеризации",
        preselected=multi_preset,
        dtype_filter='numeric')

    # Категориальные доп.
    cat_preset = _session_preset('cat_multi', [])
    cat_multi = pick_columns(df,
        "Доп. категориальные признаки — для MANOVA, двухфакторного ANOVA",
        allow_empty=True, preselected=cat_preset, dtype_filter='categorical')

    params = {
        'group': group_col,
        'analysis': analysis_col,
        'multi': multi,
        'cat_multi': cat_multi,
    }

    print(f"\n  {C.green('Итого:')} группировка={group_col}, Y={analysis_col}")
    print(f"         X({len(multi)}): {', '.join(multi[:5])}{'...' if len(multi) > 5 else ''}")
    if cat_multi:
        print(f"         кат.({len(cat_multi)}): {', '.join(cat_multi[:5])}")

    # 3. Разделы отчёта
    print("\n" + "=" * 60)
    print(f"  {C.cyan('РАЗДЕЛЫ ОТЧЁТА')}")
    print("=" * 60)

    section_map = [
        ('1', 'preprocessing',      'Предобработка данных'),
        ('2', 'plots',              'Визуализация (7 графиков)'),
        ('3', 'anova',              'ANOVA / Kruskal-Wallis / Категориальные'),
        ('4', 'manova',             'MANOVA / Post-hoc'),
        ('5', 'linear_regression',  'Регрессионный анализ (лин. + лог.)'),
        ('6', 'feature_selection',  'Отбор признаков (RF, RFE)'),
        ('7', 'pca',                'Метод главных компонент (PCA)'),
        ('8', 'cluster',            'Кластерный анализ (K-means)'),
        ('9', 'ml',                 'Машинное обучение (benchmark)'),
    ]

    tw = term_width()
    for i in range(0, len(section_map), 2):
        left = section_map[i]
        right = section_map[i + 1] if i + 1 < len(section_map) else None
        line = f"    {left[0]}. {left[2]:<40s}"
        if right:
            line += f"  {right[0]}. {right[2]}"
        print(line)

    raw = input("\n  Номера разделов через запятую (Enter = все): ").strip()
    if raw:
        selected = set()
        for part in raw.split(','):
            part = part.strip()
            for num, key, _ in section_map:
                if part == num:
                    selected.add(key)
        sections = {key: (key in selected) for key, _, _ in section_map}
        chosen_names = [label for num, key, label in section_map if key in selected]
        print(f"    -> выбрано: {', '.join(chosen_names)}")
    else:
        sections = {key: True for key, _, _ in section_map}
        print("    -> все разделы")

    # 4. Запуск анализа
    print("\n" + "=" * 60)
    print(f"  {C.cyan('ВЫПОЛНЕНИЕ АНАЛИЗА')}")
    print("=" * 60)

    analyzer = DataAnalyzer(df, file_name=filepath)
    analyzer.params = params

    # Предобработка
    print(f"\n  {C.blue('[1/9]')} Предобработка...")
    df_clean = analyzer.preprocess(remove_outliers=True, z_threshold=3.0, balance_groups=True)
    analyzer._current_df = df_clean
    stats = analyzer._preprocessing_stats
    print(f"    {C.dim('Строк:')} {stats.get('total_rows', 0)} -> {C.green(str(stats.get('final_analyzed', 0)))} "
          f"({C.yellow('исключено: ' + str(stats.get('total_excluded', 0)))})")
    corr_rem = stats.get('correlation_removals', [])
    if corr_rem:
        print(f"    {C.yellow('Удалено коррелированных признаков:')} {len(corr_rem)}")

    # Визуализация (без вывода окон)
    if sections.get('plots', True):
        print(f"\n  {C.blue('[2/9]')} Визуализация...")
        analyzer.plot_violin()
        analyzer.plot_boxplot_with_significance()
        analyzer.plot_histograms()
        analyzer.plot_pie_chart()
        analyzer.plot_scatter_with_regression()
        analyzer.plot_pairgrid()
        analyzer.plot_correlation_matrix()
        print(f"    {C.green('7 графиков готово')} (в HTML-отчёте)")

    # ANOVA
    if sections.get('anova', True):
        print(f"\n  {C.blue('[3/9]')} ANOVA / Категориальные...")
        anova_txt = analyzer.perform_anova_analysis()
        for line in anova_txt.strip().split('\n'):
            line = line.strip()
            if line:
                print(f"    {line}")
        tukey_txt = analyzer.perform_posthoc_tukey()
        if tukey_txt:
            sig_count = analyzer._analysis_results.get('tukey', {}).get('significant_count', 0)
            method = analyzer._analysis_results.get('anova', {}).get('method', '')
            label = 'Dunn (Holm)' if method == 'Kruskal-Wallis' else 'Тьюки'
            print(f"    {C.green(f'{label}:')} значимых пар — {sig_count}")
        two_way_txt = analyzer.perform_two_way_anova()
        if two_way_txt:
            print(f"    {C.green('Двухфакторный ANOVA:')} выполнен")
        analyzer.plot_interaction_effect()
        cat_txt = analyzer.perform_categorical_analysis()
        if cat_txt:
            print(f"    {C.green('Категориальные:')} {cat_txt}")
        analyzer.perform_frequency_analysis()

    # MANOVA
    if sections.get('manova', True):
        print(f"\n  {C.blue('[4/9]')} MANOVA...")
        manova_txt = analyzer.perform_manova()
        if manova_txt:
            for line in manova_txt.strip().split('\n'):
                if line.strip():
                    print(f"    {line.strip()}")
        ph_txt = analyzer.perform_posthoc_manova()
        if ph_txt:
            print(f"    {C.green('Post-hoc:')} {ph_txt}")

    # Регрессия
    if sections.get('linear_regression', True):
        print(f"\n  {C.blue('[5/9]')} Регрессионный анализ...")
        lr_txt = analyzer.perform_linear_regression()
        if lr_txt:
            print(f"    {lr_txt}")
        analyzer.plot_regression_diagnostics()
        logreg_txt = analyzer.perform_logistic_regression_cat()
        if analyzer._analysis_results.get('logistic_reg_cat', {}).get('html', ''):
            print(f"    {C.green('Лог. регрессия (кат.):')} выполнена")

    # Отбор признаков
    if sections.get('feature_selection', True):
        print(f"\n  {C.blue('[6/9]')} Отбор признаков...")
        analyzer.feature_selection_rf()
        rfe_txt = analyzer.rfe_selection()
        if rfe_txt:
            print(f"    {C.green('RFE:')} {rfe_txt}")

    # PCA
    if sections.get('pca', True):
        print(f"\n  {C.blue('[7/9]')} Метод главных компонент...")
        pca_txt = analyzer.pca_analysis()
        if pca_txt:
            print(f"    {pca_txt}")

    # Кластеризация
    if sections.get('cluster', True):
        print(f"\n  {C.blue('[8/9]')} Кластерный анализ...")
        elbow_txt = analyzer.determine_optimal_clusters(max_k=10)
        if elbow_txt:
            print(f"    {elbow_txt}")
        kmeans_txt = analyzer.perform_kmeans()
        if kmeans_txt:
            for line in kmeans_txt.strip().split('\n')[:3]:
                if line.strip():
                    print(f"    {line.strip()}")
        anova_cl_txt = analyzer.anova_for_clusters()
        if anova_cl_txt:
            print(f"    {C.green('ANOVA для кластеров:')} выполнен")
        analyzer.plot_cluster_boxplots()
        analyzer.plot_cluster_feature_dynamics()
        analyzer.plot_cluster_cat_frequencies()
        cluster_xlsx = analyzer.save_clusters_to_xlsx()
        if cluster_xlsx:
            print(f"    {C.green('Сохранено:')} {os.path.basename(cluster_xlsx)}")

    # Машинное обучение
    if sections.get('ml', True):
        print(f"\n  {C.blue('[9/9]')} Машинное обучение...")
        analyzer.ml_benchmark(df_clean)
        bench = analyzer._analysis_results.get('ml_benchmark', {})
        if bench.get('table'):
            print(f"    {C.cyan('Рейтинг моделей:')}")
            for i, r in enumerate(bench['table'][:3], 1):
                medal = {1: '1.', 2: '2.', 3: '3.'}.get(i, f'{i}.')
                print(f"      {C.green(medal)} {r['model']}  точность={r['accuracy']}")

    # Сохранение сессии
    analyzer.save_session(file_path=filepath)

    # 5. Генерация HTML-отчёта
    print("\n" + "=" * 60)
    print(f"  {C.cyan('ГЕНЕРАЦИЯ HTML-ОТЧЁТА')}")
    print("=" * 60)

    report_path = analyzer.generate_html_report(df_clean, sections=sections)
    abs_path = os.path.abspath(report_path)
    print(f"\n  {C.green('Отчёт:')} {abs_path}")

    # 6. Открытие в браузере
    print(f"  {C.dim('Открытие в браузере...')}")
    webbrowser.open('file://' + abs_path)

    print(f"\n  {C.green('Готово.')}")

if __name__ == '__main__':
    main()
