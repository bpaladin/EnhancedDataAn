# -*- coding: utf-8 -*-
"""
DataAn Enhanced — CLI-версия v2.1
Интерактивный командный строковый статистический анализ.
Полный отказ от Seaborn/Matplotlib в пользу интерактивного Plotly.
"""
from __future__ import annotations

import sys
import os
import glob
import fnmatch
import shutil
import logging
import warnings
import types
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%'
)
logger = logging.getLogger('DataAn')


class AnsiColor(Enum):
    RESET = '\033[0m'; BOLD = '\033[1m'; DIM = '\033[2m'
    RED = '\033[1;31m'; GREEN = '\033[1;32m'; YELLOW = '\033[1;33m'
    BLUE = '\033[1;34m'; CYAN = '\033[1;36m'


class C:
    _enabled = True
    @classmethod
    def disable(cls): cls._enabled = False
    @classmethod
    def _wrap(cls, color: AnsiColor, s: str) -> str:
        return f"{color.value}{s}{AnsiColor.RESET.value}" if cls._enabled else s
    @staticmethod
    def bold(s: str) -> str: return C._wrap(AnsiColor.BOLD, s)
    @staticmethod
    def green(s: str) -> str: return C._wrap(AnsiColor.GREEN, s)
    @staticmethod
    def yellow(s: str) -> str: return C._wrap(AnsiColor.YELLOW, s)
    @staticmethod
    def red(s: str) -> str: return C._wrap(AnsiColor.RED, s)
    @staticmethod
    def cyan(s: str) -> str: return C._wrap(AnsiColor.CYAN, s)
    @staticmethod
    def blue(s: str) -> str: return C._wrap(AnsiColor.BLUE, s)
    @staticmethod
    def dim(s: str) -> str: return C._wrap(AnsiColor.DIM, s)


class AppExit(Exception): pass


def mock_seaborn_and_mpl():
    if 'seaborn' not in sys.modules: sys.modules['seaborn'] = types.ModuleType('seaborn')
    if 'matplotlib' not in sys.modules:
        sys.modules['matplotlib'] = types.ModuleType('matplotlib')
        sys.modules['matplotlib.pyplot'] = types.ModuleType('matplotlib.pyplot')

mock_seaborn_and_mpl()


def load_analyzer():
    import importlib.util
    base_path = Path(sys._MEIPASS) if getattr(sys, 'frozen', False) else Path(__file__).parent
    module_path = base_path / 'analyzer_enhanced.py'
    if not module_path.exists():
        logger.error(f"Модуль анализа не найден: {module_path}")
        raise FileNotFoundError(f"Не найден {module_path}")
    spec = importlib.util.spec_from_file_location('analyzer_enhanced', str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clear(): os.system('cls' if os.name == 'nt' else 'clear')
def term_width() -> int:
    try: return shutil.get_terminal_size((80, 24)).columns
    except Exception: return 80

def setup_colors():
    if not sys.stdout.isatty(): C.disable()

def pick_file_from_args_or_menu() -> Path:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if path.is_file(): return path
        logger.error(f"Файл не найден: {path}"); raise AppExit(1)
    xlsx = sorted(glob.glob('*.xlsx')) + sorted(glob.glob('*.xls'))
    csv = sorted(glob.glob('*.csv')); files = xlsx + csv
    if not files: logger.error("В текущем каталоге нет файлов .xlsx / .csv"); raise AppExit(1)
    print("\n  Доступные файлы:\n")
    for i, f in enumerate(files, 1):
        size = os.path.getsize(f)
        size_str = f"{size / 1_048_576:.1f} MB" if size > 1_048_576 else f"{size / 1024:.0f} KB" if size > 1024 else f"{size} B"
        print(f"    {i:>3}. {f:<40s} ({size_str})")
    while True:
        try:
            choice = input("\n  Выберите номер файла (Enter — выход): ").strip()
            if not choice: raise AppExit(0)
            idx = int(choice) - 1
            if 0 <= idx < len(files): return Path(files[idx])
            print("  Неверный номер, попробуйте снова.")
        except ValueError: print("  Введите число.")

def show_columns_grid(df: Any):
    import pandas as pd
    cols, n, tw = list(df.columns), len(df.columns), term_width()
    entries = []
    for i, col in enumerate(cols, 1):
        nuniq = df[col].nunique(); miss = df[col].isna().sum(); miss_s = f" *{miss}" if miss > 0 else ""
        if pd.api.types.is_numeric_dtype(df[col]) and nuniq >= 15:
            try: mn, mx = df[col].min(), df[col].max(); tag = f"({mn:.1f}..{mx:.1f}){miss_s}"
            except Exception: tag = f"({nuniq} значений){miss_s}"
        else:
            vals = df[col].dropna().unique()
            tag = f"[{', '.join(str(v) for v in vals)}]{miss_s}" if len(vals) <= 4 else f"({nuniq} значений){miss_s}"
        entries.append(f"{i:>3}. {col} {tag}")
    max_len = max(len(e) for e in entries) + 2; ncols = max(1, tw // max_len); nrows = (n + ncols - 1) // ncols
    print(f"\n  Столбцы ({n}):  * = есть пропуски\n")
    for row in range(nrows):
        parts = [entries[row + c * nrows].ljust(max_len) for c in range(ncols) if row + c * nrows < len(entries)]
        print("    " + "  ".join(parts))

def parse_selection(raw: str, total: int, df: Optional[Any] = None) -> Optional[List[int]]:
    raw = raw.strip()
    if not raw: return None
    if raw in ('all', '*'): return list(range(total))
    tokens = [t.strip() for t in raw.split(',') if t.strip()]
    name_indices, has_names = [], False
    for tok in tokens:
        stripped = tok.lstrip('-')
        if not stripped.replace('-', '').replace('.', '').isdigit() and df is not None:
            has_names = True
            matched = [i for i, col in enumerate(df.columns) if fnmatch.fnmatch(col.lower(), tok.lower())]
            if matched: name_indices.extend(matched)
            else: print(f"  Столбцы по шаблону «{tok}» не найдены"); return []
    if has_names: return sorted(set(name_indices))
    indices, exclude = [], []
    for tok in tokens:
        is_exclude = tok.startswith('-') and len(tok) > 1 and tok[1:].isdigit()
        core = tok.lstrip('-') if is_exclude else tok
        if '-' in core:
            parts = core.split('-', 1)
            try: a, b = int(parts[0].strip()) - 1, int(parts[1].strip()) - 1; (exclude if is_exclude else indices).extend(range(min(a,b), max(a,b)+1))
            except ValueError: print(f"  Неверный диапазон: {tok}"); return []
        else:
            try: (exclude if is_exclude else indices).append(int(core.strip()) - 1)
            except ValueError: print(f"  Неверное значение: {tok}"); return []
    return sorted(set(i for i in indices if 0 <= i < total and i not in exclude))

def pick_columns(df: Any, prompt: str, allow_empty: bool = False, preselected: Optional[List[str]] = None, hint_extra: str = "", dtype_filter: Optional[str] = None) -> List[str]:
    import pandas as pd
    default_str = ', '.join(preselected) if preselected else ''
    hint = f"номера/диапазоны, маска name*, Enter = [{default_str}]" if preselected else "номера/диапазоны, маска name*, Enter = пропустить"
    if allow_empty: hint += ", * = все, - = пропустить"
    type_label, available_cols = "", df.columns.tolist()
    if dtype_filter == 'numeric': type_label, available_cols = " [только числовые]", [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    elif dtype_filter == 'categorical': type_label, available_cols = " [только категориальные]", [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    print(f"\n  {prompt}{type_label}\n  (Совет: {hint})")
    while True:
        raw = input("  > ").strip()
        if not raw and preselected is not None: return preselected
        if not raw and allow_empty: return []
        if raw in ('-', 'нет', 'skip'): return [] if allow_empty else df.select_dtypes(include=['number']).columns.tolist()
        if raw in ('*', 'all', 'все'): return available_cols if dtype_filter else df.columns.tolist()
        indices = parse_selection(raw, len(df.columns), df)
        if indices is None and preselected is not None: return preselected
        if indices is not None and len(indices) > 0:
            chosen = [df.columns[i] for i in indices]
            if dtype_filter in ('numeric', 'categorical'):
                valid = [c for c in chosen if (pd.api.types.is_numeric_dtype(df[c]) == (dtype_filter == 'numeric'))]
                if not valid: print(f"  {C.red('ОШИБКА:')} нет корректных столбцов."); continue
                chosen = valid
            return chosen
        if indices is not None and len(indices) == 0: print(f"  {C.yellow('Ничего не выбрано.')}")


def set_analyzer_data(analyzer: Any, df: Any):
    if hasattr(analyzer, 'set_data'): analyzer.set_data(df)
    else: analyzer._current_df = df


# ─────────────────── Основная логика ───────────────────
def select_variables(df: Any, filepath: Path) -> Dict[str, Any]:
    cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    if not cat_cols or not num_cols: logger.error("Нет категориальных или числовых столбцов."); raise AppExit(1)

    session_data = DataAnalyzer.load_session()
    same_file = session_data.get('last_file_path', '') == str(filepath)
    last_params = session_data.get('last_params', {}) if same_file else {}

    group_preset = [last_params.get('group')] if same_file and last_params.get('group') in cat_cols else [cat_cols[0]]
    group = pick_columns(df, "Группирующая переменная", preselected=group_preset, dtype_filter='categorical')

    analysis_preset = [last_params.get('analysis')] if same_file and last_params.get('analysis') in num_cols else [num_cols[0]]
    analysis = pick_columns(df, "Целевая переменная Y", preselected=analysis_preset, dtype_filter='numeric')

    default_multi = [c for c in num_cols[:5] if c != analysis[0]]
    multi_preset = last_params.get('multi', default_multi) if same_file else default_multi
    multi = pick_columns(df, "Многомерные признаки X", preselected=multi_preset, dtype_filter='numeric')

    cat_preset = last_params.get('cat_multi', []) if same_file else []
    cat_multi = pick_columns(df, "Доп. категориальные признаки", allow_empty=True, preselected=cat_preset, dtype_filter='categorical')

    return {'group': group[0], 'analysis': analysis[0], 'multi': multi, 'cat_multi': cat_multi}

def select_sections() -> Dict[str, bool]:
    section_map = [
        ('1', 'preprocessing', 'Предобработка'), ('2', 'plots', 'Визуализация'), ('3', 'anova', 'ANOVA'),
        ('4', 'manova', 'MANOVA'), ('5', 'linear_regression', 'Регрессия'), ('6', 'feature_selection', 'Отбор признаков'),
        ('7', 'pca', 'PCA'), ('8', 'cluster', 'Кластеризация'), ('9', 'ml', 'ML')
    ]
    print("\n" + "=" * 60); print(f"  {C.cyan('РАЗДЕЛЫ ОТЧЁТА')}"); print("=" * 60)
    for i in range(0, len(section_map), 2):
        left = section_map[i]; right = section_map[i + 1] if i + 1 < len(section_map) else None
        line = f"    {left[0]}. {left[2]:<30s}"
        if right: line += f"  {right[0]}. {right[2]}"
        print(line)
    raw = input("\n  Номера разделов через запятую (Enter = все): ").strip()
    if raw:
        selected_keys = set()
        for part in raw.split(','):
            for num, key, _ in section_map:
                if part.strip() == num: selected_keys.add(key)
        return {key: (key in selected_keys) for _, key, _ in section_map}
    return {key: True for _, key, _ in section_map}

def run_analysis(analyzer: Any, df_clean: Any, sections: Dict[str, bool]):
    set_analyzer_data(analyzer, df_clean)
    active_steps = [key for key, is_active in sections.items() if is_active and key != 'preprocessing']
    total_steps = len(active_steps) + 1
    print(f"\n  {C.blue(f'[1/{total_steps}]')} Предобработка...")
    stats = analyzer._preprocessing_stats
    print(f"    Строк: {stats.get('total_rows', 0)} -> {C.green(str(stats.get('final_analyzed', 0)))} ({C.yellow('искл: ' + str(stats.get('total_excluded', 0)))})")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, key in enumerate(active_steps, start=2):
            step_str = f"[{i}/{total_steps}]"
            try:
                if key == 'plots':
                    print(f"\n  {C.blue(step_str)} Визуализация (Plotly)...")
                elif key == 'anova':
                    print(f"\n  {C.blue(step_str)} ANOVA / Категориальные...")
                    analyzer.perform_anova_analysis()
                    analyzer.perform_posthoc_tukey()
                    analyzer.perform_two_way_anova()
                    analyzer.perform_categorical_analysis()
                    analyzer.perform_frequency_analysis()
                    print(f"    {C.green('Выполнено')}")
                elif key == 'manova':
                    print(f"\n  {C.blue(step_str)} MANOVA...")
                    analyzer.perform_manova()
                    analyzer.perform_posthoc_manova()
                    print(f"    {C.green('Выполнено')}")
                elif key == 'linear_regression':
                    print(f"\n  {C.blue(step_str)} Регрессионный анализ...")
                    analyzer.perform_linear_regression()
                    analyzer.perform_logistic_regression_cat()
                    print(f"    {C.green('Выполнено')}")
                elif key == 'feature_selection':
                    print(f"\n  {C.blue(step_str)} Отбор признаков...")
                    analyzer.feature_selection_rf()
                    analyzer.rfe_selection()
                    print(f"    {C.green('Выполнено')}")
                elif key == 'pca':
                    print(f"\n  {C.blue(step_str)} PCA...")
                    analyzer.pca_analysis()
                    print(f"    {C.green('Выполнено')}")
                elif key == 'cluster':
                    print(f"\n  {C.blue(step_str)} Кластеризация...")
                    analyzer.determine_optimal_clusters(max_k=10)
                    analyzer.perform_kmeans()
                    analyzer.anova_for_clusters()
                    analyzer.save_clusters_to_xlsx()
                    print(f"    {C.green('Выполнено')}")
                elif key == 'ml':
                    print(f"\n  {C.blue(step_str)} Машинное обучение...")
                    analyzer.ml_benchmark()
                    print(f"    {C.green('Выполнено')}")
            except Exception as e:
                logger.error(f"Ошибка в секции {key}: {e}", exc_info=True)
                print(f"    {C.red(f'ОШИБКА:')} {e}")

def generate_report(analyzer: Any, filepath: Path, sections: Dict[str, bool], analyzer_module: Any = None):
    import webbrowser
    print("\n" + "=" * 60); print(f"  {C.cyan('ГЕНЕРАЦИЯ ИНТЕРАКТИВНОГО HTML-ОТЧЁТА')}"); print("=" * 60)
    report_path = filepath.with_name(f"{filepath.stem}_interactive_report.html")

    analyzer.generate_html_report(sections=None, output_path=str(report_path))

    abs_path = report_path.resolve()
    print(f"\n  {C.green('Отчёт:')} {abs_path}")
    webbrowser.open(f'file://{abs_path}')

def main(analyzer_module=None):
    import pandas as pd
    clear(); setup_colors()
    print("=" * 60); print(f"  {C.cyan('DataAn Enhanced v2.1')} — CLI статистический анализ"); print("=" * 60)
    try:
        filepath = pick_file_from_args_or_menu()
        print(f"\n  {C.dim('Загрузка:')} {filepath}")
        try: df = pd.read_csv(filepath) if filepath.suffix.lower() == '.csv' else pd.read_excel(filepath)
        except Exception as e: logger.error(f"Не удалось прочитать файл: {e}"); raise AppExit(1)

        print(f"  {C.green('Загружено:')} {len(df):,} строк, {len(df.columns)} столбцов")
        show_columns_grid(df)

        params = select_variables(df, filepath); analyzer = DataAnalyzer(df, file_name=str(filepath)); analyzer.params = params
        sections = select_sections()
        print("\n" + "=" * 60); print(f"  {C.cyan('ВЫПОЛНЕНИЕ АНАЛИЗА')}"); print("=" * 60)
        df_clean = analyzer.preprocess(remove_outliers=True, z_threshold=3.0, balance_groups=True)

        run_analysis(analyzer, df_clean, sections)
        analyzer.save_session(file_path=str(filepath))
        generate_report(analyzer, filepath, sections, analyzer_module)
    except AppExit as e: sys.exit(e.code if hasattr(e, 'code') else 1)
    except KeyboardInterrupt: print(f"\n  {C.yellow('Прервано.')}"); sys.exit(130)

if __name__ == '__main__':
    analyzer_enhanced = load_analyzer(); DataAnalyzer = analyzer_enhanced.DataAnalyzer; main(analyzer_enhanced)
