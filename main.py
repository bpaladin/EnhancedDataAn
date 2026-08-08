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
import json
import numpy as np
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


# ─────────────────── Интерактивный HTML-отчёт (Plotly) ───────────────────
class InteractiveReportBuilder:
    def __init__(self, df: Any, params: Dict[str, Any], analysis_results: Dict[str, Any],
                 preprocessing_stats: Dict[str, Any] = None, analyzer_module: Any = None):
        self.df = df
        self.params = params
        self.results = analysis_results
        self.preprocessing_stats = preprocessing_stats or {}
        self.figures: Dict[str, str] = {}
        self._analyzer_module = analyzer_module
        self.stat_css = '''
        .stat-table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 0.95em; }
        .stat-table th { background: #3498db; color: white; padding: 10px 12px; border: 1px solid #2980b9; text-align: left; cursor: pointer; user-select: none; }
        .stat-table th:hover { background: #2980b9; }
        .stat-table th::after { content: " ⇅"; font-size: 0.8em; opacity: 0.6; }
        .stat-table th.sort-asc::after { content: " ▲"; opacity: 1; }
        .stat-table th.sort-desc::after { content: " ▼"; opacity: 1; }
        .stat-table td { padding: 8px 12px; border: 1px solid #d0d7de; }
        .stat-table tr:nth-child(even) { background: #f8f9fa; }
        .stat-table tr:hover { background: #eaf4fc; }
        .interp-note { background:#eef6ff; border-left:4px solid #3498db; padding:12px 16px; margin:15px 0; border-radius:0 6px 6px 0; font-size:0.95em; }
        .user-comment { background: #fffde7; border-left: 4px solid #fbc02d; padding: 12px 16px; margin: 15px 0; border-radius: 0 6px 6px 0; font-size: 0.95em; }
        '''

    def _fig_to_json(self, fig: Any) -> str:
        if self._analyzer_module is not None:
            return self._analyzer_module.fig_to_json(fig)
        import base64
        import numpy as np

        def _decode_bdata(obj):
            if isinstance(obj, dict):
                if 'bdata' in obj and 'dtype' in obj:
                    try:
                        decoded = np.frombuffer(base64.b64decode(obj['bdata']), dtype=obj.get('dtype', 'f8'))
                        return decoded.tolist()
                    except Exception:
                        return obj
                return {k: _decode_bdata(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_decode_bdata(item) for item in obj]
            return obj

        class _NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray): return obj.tolist()
                if isinstance(obj, (np.integer,)): return int(obj)
                if isinstance(obj, (np.floating,)): return float(obj)
                if isinstance(obj, (np.bool_,)): return bool(obj)
                return super().default(obj)

        d = _decode_bdata(fig.to_dict())
        raw = json.dumps(d, cls=_NumpyEncoder, ensure_ascii=False)
        return raw.replace('</script>', '<\\/script>')

    def _safe_fig(self, func, *args, **kwargs):
        try:
            fig = func(*args, **kwargs)
            return fig
        except Exception as e:
            logger.error(f"Ошибка построения графика: {e}")
            return None

    # ==================== ВИЗУАЛИЗАЦИЯ ====================
    def build_plots(self):
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from scipy import stats as sp_stats

        df = self.df
        group_col = self.params.get('group')
        analysis_col = self.params.get('analysis')
        multi = self.params.get('multi', [analysis_col])
        cat_multi = self.params.get('cat_multi', [])

        valid_multi = [c for c in multi if c in df.columns]
        num_cols_in_df = [c for c in valid_multi if pd.api.types.is_numeric_dtype(df[c])]

        # 1. Скрипичная диаграмма + ящик + swarm/strip
        fig = self._safe_fig(self._plot_violin, df, analysis_col, group_col)
        if fig:
            self.figures['violin'] = self._fig_to_json(fig)

        # 2. Ящики с усами со скобками значимости
        fig = self._safe_fig(self._boxplot_with_significance, df, group_col, analysis_col)
        if fig:
            self.figures['boxplot'] = self._fig_to_json(fig)

        # 3. Гистограммы с mean/median/mode
        fig = self._safe_fig(self._histograms_with_stats, df, num_cols_in_df, group_col)
        if fig:
            self.figures['histogram'] = self._fig_to_json(fig)

        # 4. Круговые диаграммы
        fig = self._safe_fig(self._pie_charts, df, group_col, cat_multi)
        if fig:
            self.figures['pie'] = self._fig_to_json(fig)

        # 5. Скаттерограмма с регрессией
        if len(num_cols_in_df) > 1:
            scatter_x = num_cols_in_df[0] if num_cols_in_df[0] != analysis_col else num_cols_in_df[1]
            fig = self._safe_fig(self._plot_scatter_regression, df, scatter_x, analysis_col, group_col)
            if fig:
                self.figures['scatter'] = self._fig_to_json(fig)

        # 6. PairGrid
        if len(num_cols_in_df) > 2:
            dims = num_cols_in_df[:6]
            fig = self._safe_fig(self._plot_pairgrid, df, dims, group_col)
            if fig:
                self.figures['pairgrid'] = self._fig_to_json(fig)

        # 7. Корреляционная матрица с полупрозрачными незначимыми
        if len(num_cols_in_df) > 1:
            fig = self._safe_fig(self._correlation_matrix_plotly, df, num_cols_in_df)
            if fig:
                self.figures['correlation'] = self._fig_to_json(fig)

        # 8. График взаимодействия
        fig = self._safe_fig(self._interaction_plot, df, group_col, analysis_col, cat_multi)
        if fig:
            self.figures['interaction'] = self._fig_to_json(fig)

        # 9. Диагностика регрессии
        lr_res = self.results.get('linear_regression', {})
        if lr_res.get('y_test') is not None:
            fig = self._safe_fig(self._regression_diagnostics, lr_res)
            if fig:
                self.figures['regression_diagnostics'] = self._fig_to_json(fig)

        # 10. Важность признаков (RF)
        rf_data = self.results.get('rf_importance_data', {})
        if rf_data:
            fig = self._safe_fig(self._feature_importance_plot, rf_data)
            if fig:
                self.figures['rf_importance'] = self._fig_to_json(fig)

        # 11. PCA
        pca_data = self.results.get('pca', {})
        if pca_data.get('explained_variance'):
            fig = self._safe_fig(self._pca_plot, pca_data)
            if fig:
                self.figures['pca'] = self._fig_to_json(fig)

        # 12. Elbow
        elbow = self.results.get('elbow', {})
        if elbow.get('inertias'):
            fig = self._safe_fig(self._elbow_plot, elbow)
            if fig:
                self.figures['elbow'] = self._fig_to_json(fig)

        # 12. K-Means scatter
        kmeans = self.results.get('kmeans', {})
        if kmeans.get('labels') is not None and len(num_cols_in_df) >= 2:
            fig = self._safe_fig(self._kmeans_scatter, df, num_cols_in_df, kmeans)
            if fig:
                self.figures['kmeans_scatter'] = self._fig_to_json(fig)

        # 13. Профили кластеров (динамика + тепловая карта)
        if kmeans.get('cluster_means'):
            fig = self._safe_fig(self._cluster_dynamics, kmeans)
            if fig:
                self.figures['cluster_dynamics'] = self._fig_to_json(fig)

        # 14. Boxplot признаков по кластерам
        if kmeans.get('labels') is not None and num_cols_in_df:
            fig = self._safe_fig(self._cluster_boxplots, df, num_cols_in_df, kmeans)
            if fig:
                self.figures['cluster_boxplots'] = self._fig_to_json(fig)

        # 15. Матрица ошибок
        ml = self.results.get('ml_benchmark', {})
        if ml.get('best_y_test') is not None:
            fig = self._safe_fig(self._confusion_matrix, ml)
            if fig:
                self.figures['confusion_matrix'] = self._fig_to_json(fig)

        # 16. ROC-кривая
        if ml.get('best_y_proba') is not None:
            fig = self._safe_fig(self._roc_curve, ml)
            if fig:
                self.figures['roc_curve'] = self._fig_to_json(fig)

    def _plot_violin(self, df, analysis_col, group_col):
        import plotly.express as px
        fig = px.violin(df, y=analysis_col, x=group_col, box=True,
                        points="all", color=group_col,
                        title=f"Скрипичная диаграмма: {analysis_col} по {group_col}",
                        color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(template="plotly_white", showlegend=False)
        return fig

    def _plot_scatter_regression(self, df, x_col, y_col, group_col):
        import plotly.express as px
        import numpy as np
        from scipy import stats as sp_stats
        fig = px.scatter(df, x=x_col, y=y_col, color=group_col,
                         trendline="ols",
                         title=f"Scatter + OLS: {x_col} vs {y_col}",
                         color_discrete_sequence=px.colors.qualitative.Set2)
        slope, intercept, r, p, se = sp_stats.linregress(df[x_col].dropna(), df[y_col].dropna())
        angle = np.degrees(np.arctan(slope))
        fig.update_layout(template="plotly_white")
        fig.add_annotation(
            text=(f"Угол наклона: {angle:.1f}° | Наклон (β₁): {slope:.4f}<br>"
                  f"r = {r:.3f} | R² = {r**2:.3f} | p = {p:.2e}"),
            xref="paper", yref="paper", x=0.02, y=0.98,
            showarrow=False, font=dict(size=11),
            bgcolor="rgba(255,255,255,0.85)", bordercolor="#ccc", borderwidth=1)
        return fig

    def _plot_pairgrid(self, df, dims, group_col):
        import plotly.express as px
        fig = px.scatter_matrix(df, dimensions=dims, color=group_col,
                                title="Попарные распределения",
                                height=900,
                                color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(diagonal_visible=True, showupperhalf=False)
        fig.update_layout(template="plotly_white")
        return fig

    def _boxplot_with_significance(self, df, g_col, a_col):
        import plotly.express as px
        import plotly.graph_objects as go
        from itertools import combinations
        from scipy import stats as sp_stats

        groups = sorted(df[g_col].unique(), key=str)
        fig = go.Figure()
        for g in groups:
            vals = df[df[g_col] == g][a_col].dropna()
            fig.add_trace(go.Box(y=vals, name=str(g), boxpoints='outliers',
                                 marker_color=px.colors.qualitative.Set2[groups.index(g) % len(groups)]))

        if len(groups) >= 2:
            y_max = df[a_col].max()
            y_range = df[a_col].max() - df[a_col].min()
            if y_range == 0: y_range = abs(y_max) if y_max != 0 else 1.0
            bracket_y = y_max + y_range * 0.05
            for i, j in combinations(range(len(groups)), 2):
                g1_vals = df[df[g_col] == groups[i]][a_col].dropna().values
                g2_vals = df[df[g_col] == groups[j]][a_col].dropna().values
                if len(g1_vals) < 3 or len(g2_vals) < 3: continue
                _, p_val = sp_stats.mannwhitneyu(g1_vals, g2_vals, alternative='two-sided')
                if p_val < 0.001: sig = '***'
                elif p_val < 0.01: sig = '**'
                elif p_val < 0.05: sig = '*'
                else: continue
                fig.add_shape(type="line", x0=i, x1=j, y0=bracket_y, y1=bracket_y,
                              line=dict(color="#2c3e50", width=2))
                fig.add_annotation(x=(i+j)/2, y=bracket_y + y_range*0.02, text=sig,
                                   showarrow=False, font=dict(size=14, color="#e74c3c", family="Arial Black"))
                bracket_y += y_range * 0.08

        fig.update_layout(title=f'Ящики с усами: {a_col}', yaxis_title=a_col, xaxis_title=g_col,
                          template="plotly_white", height=500)
        return fig

    def _histograms_with_stats(self, df, num_cols, group_col):
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import numpy as np

        cols = num_cols[:6]
        n = len(cols)
        if n == 0: return None
        fig = make_subplots(rows=1, cols=n, subplot_titles=cols,
                            horizontal_spacing=0.04)
        colors = px.colors.qualitative.Set2
        stat_traces = {'mean': [], 'median': [], 'mode': []}
        for idx, col in enumerate(cols):
            data = df[col].dropna()
            if data.empty: continue
            mean_v = data.mean()
            med_v = data.median()
            mode_v = data.mode().iloc[0] if not data.mode().empty else np.nan
            fig.add_trace(go.Histogram(x=data, name=col, opacity=0.75,
                                       marker_color=colors[idx % len(colors)],
                                       showlegend=False), row=1, col=idx+1)
            fig.add_vline(x=mean_v, line_dash="dash", line_color="red", row=1, col=idx+1)
            fig.add_vline(x=med_v, line_dash="dot", line_color="green", row=1, col=idx+1)
            if not np.isnan(mode_v):
                fig.add_vline(x=mode_v, line_dash="dashdot", line_color="orange", row=1, col=idx+1)
            if idx == 0:
                stat_traces['mean'].append(go.Scatter(x=[None], y=[None], mode='lines',
                    line=dict(color='red', dash='dash', width=2), name='Среднее (μ)'))
                stat_traces['median'].append(go.Scatter(x=[None], y=[None], mode='lines',
                    line=dict(color='green', dash='dot', width=2), name='Медиана (M)'))
                stat_traces['mode'].append(go.Scatter(x=[None], y=[None], mode='lines',
                    line=dict(color='orange', dash='dashdot', width=2), name='Мода (Mo)'))
            fig.add_annotation(x=mean_v, y=0, text=f'{mean_v:.2f}',
                showarrow=False, font=dict(size=9, color='red'),
                xref=f'x{idx+1}' if idx > 0 else 'x', yref='y',
                yshift=10, row=1, col=idx+1)
            fig.add_annotation(x=med_v, y=0, text=f'{med_v:.2f}',
                showarrow=False, font=dict(size=9, color='green'),
                xref=f'x{idx+1}' if idx > 0 else 'x', yref='y',
                yshift=-10, row=1, col=idx+1)

        fig.add_traces(stat_traces['mean'] + stat_traces['median'] + stat_traces['mode'])
        fig.update_layout(title="Гистограммы с статистиками", template="plotly_white",
                          height=450,
                          legend=dict(orientation='h', yanchor='bottom', y=-0.25,
                                      xanchor='center', x=0.5, font=dict(size=11)))
        fig.update_annotations(font_size=12)
        return fig

    def _pie_charts(self, df, group_col, cat_multi):
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        cats = [group_col]
        for c in cat_multi:
            if c in df.columns and c not in cats: cats.append(c)
        cats = cats[:4]
        n = len(cats)
        if n == 0: return None
        fig = make_subplots(rows=1, cols=n, specs=[[{'type': 'pie'}]*n],
                            subplot_titles=cats)
        for idx, col in enumerate(cats):
            counts = df[col].value_counts()
            fig.add_trace(go.Pie(labels=counts.index.astype(str), values=counts.values,
                                  hole=0.3, textinfo='percent',
                                  marker_colors=px.colors.qualitative.Set2[:len(counts)]), row=1, col=idx+1)
        fig.update_layout(title="Круговые диаграммы", height=400, template="plotly_white")
        return fig

    def _correlation_matrix_plotly(self, df, num_cols):
        import plotly.graph_objects as go
        from scipy import stats as sp_stats
        import pandas as pd
        import numpy as np

        corr = df[num_cols].corr(numeric_only=True)
        n = len(num_cols)
        p_vals = np.ones((n, n))
        for i, c1 in enumerate(num_cols):
            for j, c2 in enumerate(num_cols):
                if i < j:
                    pair = df[[c1, c2]].dropna()
                    _, p = sp_stats.pearsonr(pair[c1], pair[c2])
                    p_vals[i, j] = p
                    p_vals[j, i] = p

        corr_vals = corr.values

        def _marker(p):
            if p < 0.001:
                return '***'
            if p < 0.01:
                return '**'
            if p < 0.05:
                return '*'
            return '·'

        def _fmt(val, p, marker=True):
            if pd.isna(val):
                return 'N/A'
            m = _marker(p) if marker else ''
            return f'{val:.2f}{m}'

        # Значимые ячейки окрашиваются, незначимые (p>=0.05) выглядят приглушённо
        # (z=0 -> белый цвет матрицы), реальное r в подсказке приходит из customdata.
        # Один heatmap на всю матрицу — чтобы подсказки всегда показывали настоящее r,
        # а не "r=0.000" от перекрывающего слоя.
        z = np.full((n, n), np.nan)
        text = np.full((n, n), '', dtype=object)
        cd = np.full((n, n), '', dtype=object)

        for i in range(n):
            for j in range(n):
                if i == j:
                    # Диагональ: корреляция признака с самим собой = 1
                    z[i, j] = 1.0
                    cd[i, j] = '1.000'
                elif i > j:
                    r_val = corr_vals[i, j]
                    text[i, j] = _fmt(r_val, p_vals[i, j])
                    cd[i, j] = f'{r_val:.3f}'
                    z[i, j] = r_val if p_vals[i, j] < 0.05 else 0.0

        heat = go.Heatmap(
            z=z.tolist(), x=num_cols, y=num_cols,
            colorscale='RdBu_r', zmin=-1, zmax=1,
            text=text.tolist(), texttemplate='%{text}',
            textfont={'size': 11},
            customdata=cd.tolist(),
            showscale=True,
            colorbar=dict(title='Корреляция r'),
            hovertemplate='%{x} vs %{y}<br>r=%{customdata}<extra></extra>')

        fig = go.Figure(data=[heat])

        fig.update_layout(
            title='Корреляционная матрица (*** p<0.001, ** p<0.01, * p<0.05, · — незначимо, p≥0.05)',
            template='plotly_white', height=600, width=700)
        # Основа матрицы: строка з-индекса 0 сверху (как в таблицах), значения ниже главной диагонали
        fig.update_yaxes(autorange='reversed')
        return fig

    def _interaction_plot(self, df, group_col, analysis_col, cat_multi):
        import plotly.express as px
        import plotly.graph_objects as go
        import pandas as pd
        import numpy as np

        second_factor = None
        for c in cat_multi:
            if c in df.columns and c != group_col:
                second_factor = c
                break
        if second_factor is None: return None

        n_g = df[group_col].nunique()
        n_s = df[second_factor].nunique()
        if n_g >= n_s:
            x_col, hue_col = group_col, second_factor
        else:
            x_col, hue_col = second_factor, group_col

        grouped = df.groupby([x_col, hue_col])[analysis_col].agg(['median', 'count'])
        grouped.columns = ['median', 'count']
        grouped['q1'] = df.groupby([x_col, hue_col])[analysis_col].quantile(0.25)
        grouped['q3'] = df.groupby([x_col, hue_col])[analysis_col].quantile(0.75)
        grouped = grouped.reset_index()

        fig = go.Figure()
        colors = px.colors.qualitative.Set2
        hues = sorted(grouped[hue_col].unique(), key=str)
        for idx, hue_val in enumerate(hues):
            sub = grouped[grouped[hue_col] == hue_val]
            x_vals = sub[x_col].astype(str).tolist()
            medians = sub['median'].tolist()
            q1s = sub['q1'].tolist()
            q3s = sub['q3'].tolist()
            color = colors[idx % len(colors)]
            fig.add_trace(go.Scatter(x=x_vals, y=medians, mode='lines+markers',
                                      name=f'{hue_col}={hue_val}',
                                      line=dict(color=color, width=2.5),
                                      marker=dict(size=8)))
            fig.add_trace(go.Scatter(
                x=x_vals + x_vals[::-1], y=q3s + q1s[::-1],
                fill='toself', fillcolor=color.replace(')', ',0.15)').replace('rgb', 'rgba') if 'rgb' in color else color + '22',
                line=dict(color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='skip'))
        fig.update_layout(title=f"Взаимодействие: {x_col} × {hue_col} на {analysis_col}",
                          xaxis_title=x_col, yaxis_title=analysis_col,
                          template="plotly_white", height=500,
                          annotations=[dict(
                              text="Линия — медиана; область — межквартильный размах (Q1–Q3)",
                              xref="paper", yref="paper", x=0.5, y=-0.12,
                              showarrow=False, font=dict(size=11, color="gray"))])
        return fig

    def _regression_diagnostics(self, lr_res):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from scipy import stats as sp_stats

        y_test = lr_res['y_test']
        y_pred = lr_res['y_pred']
        residuals = y_test - y_pred

        fig = make_subplots(rows=1, cols=2, subplot_titles=["Остатки vs Предсказанные", "Q-Q plot остатков"])
        fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers',
                                  marker=dict(color='#3498db', opacity=0.6)),
                       row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="#e74c3c", row=1, col=1)

        sorted_res = np.sort(residuals)
        norm_quantiles = sp_stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_res)))
        fig.add_trace(go.Scatter(x=norm_quantiles, y=sorted_res, mode='markers',
                                  marker=dict(color='#3498db', opacity=0.6), name='Остатки'),
                       row=1, col=2)
        lim = max(abs(norm_quantiles.min()), abs(norm_quantiles.max()), abs(sorted_res.min()), abs(sorted_res.max()))
        fig.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode='lines',
                                  line=dict(color='#e74c3c', dash='dash'), name='Идеал'),
                       row=1, col=2)

        fig.update_xaxes(title_text="Предсказанные", row=1, col=1)
        fig.update_yaxes(title_text="Остатки", row=1, col=1)
        fig.update_xaxes(title_text="Теоретические квантили", row=1, col=2)
        fig.update_yaxes(title_text="Выборочные квантили", row=1, col=2)
        fig.update_layout(title="Диагностика регрессии", template="plotly_white", height=450, showlegend=False)
        return fig

    def _feature_importance_plot(self, rf_data):
        import plotly.graph_objects as go

        features = rf_data['features'][::-1]
        values = rf_data['values'][::-1]
        fig = go.Figure(go.Bar(x=values, y=features, orientation='h',
                                marker_color='#3498db', text=[f'{v:.3f}' for v in values],
                                textposition='outside'))
        fig.update_layout(title="Важность признаков (Random Forest)", xaxis_title="Важность",
                          template="plotly_white", height=max(300, len(features)*40))
        return fig

    def _pca_plot(self, pca_data):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import numpy as np

        ev = pca_data['explained_variance']
        cum = pca_data['cumulative_variance']
        loadings = pca_data.get('loadings', {})
        labels = [f'PC{i+1}' for i in range(len(ev))]

        has_loadings = bool(loadings)
        if has_loadings:
            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=["Объяснённая и кумулятивная дисперсия", "Нагрузки (Loadings)"],
                                column_widths=[0.5, 0.5])
        else:
            fig = make_subplots(rows=1, cols=1,
                                subplot_titles=["Объяснённая и кумулятивная дисперсия"])

        fig.add_trace(go.Bar(x=labels, y=[v*100 for v in ev], name='Доля дисперсии',
                              marker_color='#3498db', text=[f'{v*100:.1f}%' for v in ev],
                              textposition='outside'), row=1, col=1)
        fig.add_trace(go.Scatter(x=labels, y=[v*100 for v in cum], mode='lines+markers',
                                  name='Кумулятивная', marker=dict(color='#e74c3c', size=8),
                                  line=dict(color='#e74c3c', width=2)), row=1, col=1)
        fig.add_hline(y=95, line_dash="dash", line_color="#27ae60",
                      annotation_text="95%", row=1, col=1)

        if has_loadings:
            features = list(loadings.keys())
            pcs = [k for k in next(iter(loadings.values())).keys()]
            z = [[loadings[f].get(pc, 0) for pc in pcs] for f in features]
            fig.add_trace(go.Heatmap(z=z, x=pcs, y=features,
                                      colorscale='RdBu_r', zmin=-1, zmax=1,
                                      text=[[f'{v:.2f}' for v in row] for row in z],
                                      texttemplate="%{text}", textfont={"size": 10},
                                      showscale=True, name='Loadings',
                                      colorbar=dict(title='Нагрузка', thickness=12,
                                                    len=0.5, y=0.5)), row=1, col=2)

        fig.update_layout(title="Метод главных компонент (PCA)", template="plotly_white",
                          height=max(400, 100 + len(labels)*30),
                          legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5))
        fig.update_yaxes(title_text="%", row=1, col=1)
        if has_loadings:
            fig.update_xaxes(title_text="Компонента", row=1, col=2)
            fig.update_yaxes(title_text="Признак", row=1, col=2)
            fig.add_annotation(
                text='Красный — положительная нагрузка, синий — отрицательная',
                xref='paper', yref='paper', x=0.73, y=-0.15,
                showarrow=False, font=dict(size=11, color='gray'))
        return fig

    def _elbow_plot(self, elbow):
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=elbow['k_range'], y=elbow['inertias'],
                                  mode='lines+markers', marker=dict(size=10, color='#3498db')))
        optimal_k = elbow.get('optimal_k', 2)
        fig.add_vline(x=optimal_k, line_dash="dash", line_color="#e74c3c",
                      annotation_text=f"Optimal k={optimal_k}")
        fig.update_layout(title="Метод каменистой осыпи (Elbow)", xaxis_title="Число кластеров k",
                          yaxis_title="Инерция (WCSS)", template="plotly_white", height=450)
        return fig

    def _kmeans_scatter(self, df, num_cols, kmeans):
        import plotly.express as px
        import plotly.graph_objects as go
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        X = df[num_cols].dropna()
        labels = np.array(kmeans['labels'])
        if len(labels) != len(X): return None
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(StandardScaler().fit_transform(X))

        fig = px.scatter(x=X_2d[:, 0], y=X_2d[:, 1], color=labels.astype(str),
                         title=f"K-Means кластеризация (k={kmeans['k']})",
                         labels={'x': 'PC1', 'y': 'PC2', 'color': 'Кластер'},
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(template="plotly_white", height=550)
        return fig

    def _cluster_dynamics(self, kmeans):
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        features = kmeans['features']
        means = kmeans['cluster_means']
        q1_data = kmeans.get('cluster_q1', {})
        q3_data = kmeans.get('cluster_q3', {})
        clusters = list(means.keys())

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Профили кластеров (медиана + IQR)", "Тепловая карта средних"],
                            column_widths=[0.6, 0.4])

        colors = px.colors.qualitative.Set2
        for i, cl in enumerate(clusters):
            vals = [means[cl].get(f, 0) for f in features]
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(x=features, y=vals, mode='lines+markers',
                                      name=f'Кластер {cl}',
                                      line=dict(width=2.5, color=color),
                                      marker=dict(size=8, color=color)),
                           row=1, col=1)
            if cl in q1_data and cl in q3_data:
                q1_vals = [q1_data[cl].get(f, 0) for f in features]
                q3_vals = [q3_data[cl].get(f, 0) for f in features]
                fig.add_trace(go.Scatter(
                    x=features + features[::-1], y=q3_vals + q1_vals[::-1],
                    fill='toself', fillcolor=color.replace(')', ',0.12)').replace('rgb', 'rgba') if 'rgb' in color else color + '1e',
                    line=dict(color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='skip'),
                    row=1, col=1)

        heat_z = [[means[cl].get(f, 0) for cl in clusters] for f in features]
        heat_text = [[f'{means[cl].get(f, 0):.2f}' for cl in clusters] for f in features]
        fig.add_trace(go.Heatmap(z=heat_z, x=[f'Кластер {c}' for c in clusters], y=features,
                                  colorscale='YlOrRd', showscale=True,
                                  text=heat_text, texttemplate="%{text}",
                                  textfont={"size": 11}),
                       row=1, col=2)

        fig.update_layout(title="Профили кластеров", template="plotly_white", height=500)
        fig.add_annotation(
            text="Линия — среднее; область — межквартильный размах (Q1–Q3)",
            xref="paper", yref="paper", x=0.3, y=-0.12,
            showarrow=False, font=dict(size=11, color="gray"))
        return fig

    def _cluster_boxplots(self, df, num_cols, kmeans):
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        labels = kmeans['labels']
        df_plot = df[num_cols].copy()
        df_plot['Кластер'] = [str(l) for l in labels]

        fig = make_subplots(rows=1, cols=min(len(num_cols), 4),
                            subplot_titles=num_cols[:4])
        for idx, col in enumerate(num_cols[:4]):
            for cl in sorted(df_plot['Кластер'].unique()):
                vals = df_plot[df_plot['Кластер'] == cl][col].dropna()
                fig.add_trace(go.Box(y=vals, name=f'Кластер {cl}', showlegend=(idx == 0)),
                               row=1, col=idx+1)
        fig.update_layout(title="Распределение признаков по кластерам",
                          template="plotly_white", height=400)
        return fig

    def _confusion_matrix(self, ml):
        import plotly.express as px
        import plotly.graph_objects as go
        from sklearn.metrics import confusion_matrix
        import numpy as np

        y_true = ml['best_y_test']
        y_pred = ml['best_y_pred']
        class_names = ml.get('best_class_names', [])
        if class_names is None or len(class_names) == 0:
            class_names = [str(i) for i in range(len(np.unique(y_true)))]
        cm = confusion_matrix(y_true, y_pred)

        fig = go.Figure(data=go.Heatmap(z=cm, x=class_names, y=class_names,
                                         colorscale='Blues', text=cm, texttemplate="%{text}",
                                         textfont={"size": 14}))
        fig.update_layout(title=f"Матрица ошибок: {ml.get('best_model', '')}",
                          xaxis_title="Предсказанные", yaxis_title="Истинные",
                          template="plotly_white", height=450)
        return fig

    def _roc_curve(self, ml):
        import plotly.express as px
        import plotly.graph_objects as go
        from sklearn.metrics import roc_curve, roc_auc_score
        import numpy as np

        y_test = ml['best_y_test']
        y_proba = ml['best_y_proba']
        class_names = ml.get('best_class_names', [])
        if class_names is None or (hasattr(class_names, '__len__') and len(class_names) == 0):
            class_names = [str(i) for i in range(len(np.unique(y_test)))]
        nc = len(class_names)

        fig = go.Figure()
        if nc == 2:
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            auc = ml.get('best_auc_mean', 0)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'AUC={auc:.3f}',
                                      line=dict(width=2.5, color=px.colors.qualitative.Set2[0])))
        else:
            for i in range(nc):
                fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_proba[:, i])
                auc_i = roc_auc_score((y_test == i).astype(int), y_proba[:, i])
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{class_names[i]} (AUC={auc_i:.3f})',
                                          line=dict(width=2.5, color=px.colors.qualitative.Set2[i % len(px.colors.qualitative.Set2)])))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                  line=dict(dash='dash', color='gray', width=1.5), showlegend=False))
        fig.update_layout(title=f"ROC-кривая: {ml.get('best_model', '')}",
                          xaxis_title="FPR", yaxis_title="TPR",
                          template="plotly_white", height=450)
        return fig

    # ==================== HTML БЛОКИ РЕЗУЛЬТАТОВ ====================
    def _build_results_html(self) -> str:
        sections = {
            'anova': ('One-Way ANOVA / Kruskal-Wallis', 3),
            'tukey': ('Post-hoc анализ', 3),
            'two_way': ('Two-Way ANOVA', 3),
            'categorical': ('Анализ категориальных связей (χ²)', 3),
            'frequency': ('Частоты встречаемости', 3),
            'manova': ('MANOVA', 4),
            'posthoc_manova': ('Post-hoc MANOVA (Tukey HSD)', 4),
            'linear_regression': ('Линейная регрессия', 5),
            'logistic_reg_cat': ('Логистическая регрессия', 5),
            'rfe': ('Отбор признаков (RFE)', 6),
            'pca': ('Метод главных компонент (PCA)', 7),
            'elbow': ('Оптимальное число кластеров', 8),
            'kmeans': ('K-Means кластеризация', 8),
            'cluster_anova': ('ANOVA для кластеров', 8),
            'ml_benchmark': ('ML Бенчмарк', 9),
        }
        html_out = ""
        seen = set()
        for key, (title, section_num) in sections.items():
            data = self.results.get(key)
            if not data: continue
            if key in seen: continue
            seen.add(key)

            content = ""
            if key == 'ml_benchmark':
                tbl = data.get('table', [])
                if tbl:
                    rows = "".join(f"<tr><td>{r.get('model','')}</td><td>{r.get('accuracy','')}</td><td>{r.get('auc','')}</td></tr>" for r in tbl)
                    content = f"<table class='stat-table'><tr><th>Модель</th><th>Accuracy</th><th>AUC</th></tr>{rows}</table>"
                if data.get('html'): content += data['html']
            elif key == 'rf_importance':
                items = "".join(f"<li>{k}: {v:.4f}</li>" for k, v in sorted(data.items(), key=lambda x: x[1], reverse=True))
                content = f"<ul>{items}</ul>"
            else:
                content = data.get('html', '')
                if not content:
                    text = data.get('text', data.get('summary', ''))
                    if text: content = f"<pre>{text}</pre>"
                if not content:
                    content = f"<pre>{str(data)}</pre>"

            html_out += f"""
            <div class="result-card" id="result_{key}">
                <h2>{section_num}. {title}</h2>
                <div class="result-content">{content}</div>
            </div>"""
        return html_out

    # ==================== ГЕНЕРАЦИЯ HTML ====================
    def generate_html(self, output_path: Path):
        self.build_plots()
        plotly_cdn = "https://cdn.plot.ly/plotly-latest.min.js"

        # Маппинг: section_key -> list of (chart_name, chart_title)
        section_charts = {
            'plots': [
                ('violin', 'Скрипичная диаграмма'),
                ('boxplot', 'Ящики с усами (со скобками значимости)'),
                ('histogram', 'Гистограммы с расширенной статистикой'),
                ('pie', 'Круговые диаграммы'),
                ('scatter', 'Скаттерограмма с регрессией'),
                ('pairgrid', 'Попарные распределения (PairGrid)'),
                ('correlation', 'Корреляционная матрица'),
                ('interaction', 'График взаимодействия'),
            ],
            'linear_regression': [
                ('regression_diagnostics', 'Диагностика регрессии'),
            ],
            'feature_selection': [
                ('rf_importance', 'Важность признаков (Random Forest)'),
            ],
            'pca': [
                ('pca', 'Метод главных компонент (PCA)'),
            ],
            'cluster': [
                ('elbow', 'Оптимальное число кластеров (Elbow)'),
                ('kmeans_scatter', 'K-Means кластеризация (PCA 2D)'),
                ('cluster_dynamics', 'Профили кластеров (медиана + IQR)'),
                ('cluster_boxplots', 'Признаки по кластерам'),
            ],
            'ml': [
                ('confusion_matrix', 'Матрица ошибок (лучшая модель)'),
                ('roc_curve', 'ROC-кривая'),
            ],
        }

        # Секции анализа: key -> (title, subsections)
        section_defs = [
            ('preprocessing', '1. Предобработка данных', [
                ('1.1', 'Статистика предобработки', []),
                ('1.2', 'Пропуски по столбцам', []),
                ('1.3', 'Корреляционные связи', []),
                ('1.4', 'Анализ связей категориальных признаков', []),
            ]),
            ('plots', '2. Визуализация', [
                ('2.1', 'Распределения', ['violin', 'boxplot', 'histogram', 'pie']),
                ('2.2', 'Зависимости', ['scatter', 'pairgrid', 'correlation', 'interaction']),
            ]),
            ('anova', '3. Дисперсионный анализ', [
                ('3.1', 'One-Way ANOVA / Kruskal-Wallis', []),
                ('3.2', 'Post-hoc анализ', []),
                ('3.3', 'Two-Way ANOVA', []),
            ]),
            ('manova', '4. Многомерный анализ', [
                ('4.1', 'MANOVA', []),
                ('4.2', 'Post-hoc MANOVA', []),
            ]),
            ('linear_regression', '5. Регрессионный анализ', [
                ('5.1', 'Линейная регрессия', []),
                ('5.2', 'Диагностика регрессии', ['regression_diagnostics']),
                ('5.3', 'Логистическая регрессия', []),
            ]),
            ('feature_selection', '6. Отбор признаков', [
                ('6.1', 'Важность признаков (RF)', ['rf_importance']),
                ('6.2', 'Рекурсивное устранение (RFE)', []),
            ]),
            ('pca', '7. Метод главных компонент', [
                ('7.1', 'PCA', ['pca']),
            ]),
            ('cluster', '8. Кластерный анализ', [
                ('8.1', 'Оптимальное число кластеров (Elbow)', ['elbow']),
                ('8.2', 'K-Means кластеризация (PCA 2D)', ['kmeans_scatter']),
                ('8.3', 'ANOVA для кластеров', []),
                ('8.4', 'Профили и boxplot по кластерам', ['cluster_dynamics', 'cluster_boxplots']),
            ]),
            ('ml', '9. Машинное обучение', [
                ('9.1', 'Сравнение методов', []),
                ('9.2', 'Матрица ошибок и ROC', ['confusion_matrix', 'roc_curve']),
            ]),
        ]

        # Результаты анализа: key -> (title, content)
        analysis_results = {}

        # Build preprocessing results from preprocessing_stats
        if self.preprocessing_stats:
            stats = self.preprocessing_stats
            prep_items = []
            html_stats = (
                f'<table class="stat-table" style="width:60%;">'
                f'<tr><th>Показатель</th><th>Значение</th></tr>'
                f'<tr><td>Всего строк в исходных данных</td><td>{stats.get("total_rows", 0)}</td></tr>'
                f'<tr><td>Исключено строк без группирующей переменной ({stats.get("group_col", "")})</td>'
                f'<td>{stats.get("excluded_no_group", 0)}</td></tr>'
                f'<tr><td>Исключено строк с пропусками в других столбцах</td>'
                f'<td>{stats.get("excluded_other_missing", 0)}</td></tr>'
                f'<tr><td>Исключено выбросов (z-score)</td><td>{stats.get("excluded_outliers", 0)}</td></tr>'
                f'<tr><td><b>Осталось для анализа</b></td><td><b>{stats.get("final_analyzed", 0)}</b></td></tr>'
                f'<tr><td>Всего исключено</td><td>{stats.get("total_excluded", 0)}</td></tr>'
                f'</table>'
            )
            prep_items.append(('Статистика предобработки', html_stats))

            missing = stats.get('missing_per_column', {})
            if missing:
                rows = ''.join(f'<tr><td>{col}</td><td>{cnt}</td></tr>' for col, cnt in missing.items() if cnt > 0)
                if rows:
                    miss_html = (f'<table class="stat-table" style="width:50%;">'
                                 f'<tr><th>Столбец</th><th>Пропусков</th></tr>{rows}</table>')
                    prep_items.append(('Пропуски по столбцам', miss_html))

            corr_removals = stats.get('correlation_removals', [])
            corr_threshold = stats.get('correlation_threshold', 0.9)
            corr_pairs = stats.get('corr_pairs', [])
            removal_set = {(k, d) for k, d, _ in corr_removals} if corr_removals else set()
            corr_parts = []
            if corr_pairs:
                rows = ''
                for c1, c2, r in corr_pairs:
                    status = 'удалён' if (c1, c2) in removal_set or (c2, c1) in removal_set else 'оставлен'
                    rows += f'<tr><td>{c1}</td><td>{c2}</td><td>{r:.3f}</td><td>{status}</td></tr>\n'
                corr_parts.append(f'<table class="stat-table" style="width:70%;">'
                                  f'<tr><th>Признак 1</th><th>Признак 2</th><th>|r|</th><th>Статус</th></tr>{rows}</table>'
                                  f'<p>Порог: {corr_threshold}. Показаны все пары с |r| ≥ порога.</p>')
            if not corr_parts:
                corr_parts.append('<p>Высококоррелированных признаков не обнаружено — все признаки сохранены.</p>')
            prep_items.append(('Корреляционные связи', '\n'.join(corr_parts)))

            cat_html = self.results.get('categorical', {}).get('html', '')
            if not cat_html:
                cat_html = '<p>Значимых связей между категориальными признаками не обнаружено.</p>'
            prep_items.append(('Анализ связей категориальных признаков', cat_html))

            analysis_results['preprocessing'] = prep_items

        result_sections = {
            'anova': [('One-Way ANOVA / Kruskal-Wallis', 'anova'), ('Post-hoc анализ', 'tukey'),
                      ('Two-Way ANOVA', 'two_way')],
            'manova': [('MANOVA', 'manova'), ('Post-hoc MANOVA', 'posthoc_manova')],
            'linear_regression': [('Линейная регрессия', 'linear_regression'),
                                  ('Логистическая регрессия', 'logistic_reg_cat')],
            'feature_selection': [('RFE', 'rfe')],
            'pca': [('PCA', 'pca')],
            'cluster': [('ANOVA для кластеров', 'cluster_anova')],
            'ml': [('ML Бенчмарк', 'ml_benchmark')],
        }

        for sec_key, pairs in result_sections.items():
            items = []
            for title, res_key in pairs:
                data = self.results.get(res_key)
                if not data: continue
                content = self._format_result(res_key, data)
                if content:
                    items.append((title, content))
            analysis_results[sec_key] = items

        # Сборка HTML по секциям
        sections_html = ""
        chart_idx = 0
        for sec_key, sec_title, subsections in section_defs:
            # Проверяем, есть ли данные для этой секции
            has_charts = sec_key in section_charts and any(
                c in self.figures for c, _ in section_charts[sec_key])
            has_results = sec_key in analysis_results and analysis_results[sec_key]
            if not has_charts and not has_results:
                continue

            sections_html += f'<div class="section" id="section_{sec_key}">\n'
            sections_html += f'<h2>{sec_title}</h2>\n'

            for sub_num, sub_title, chart_keys in subsections:
                # Подзаголовок
                sections_html += f'<h3>{sub_num} {sub_title}</h3>\n'

                # Графики этой подсекции
                if chart_keys:
                    for ck in chart_keys:
                        if ck in self.figures:
                            chart_idx += 1
                            pid = f"plot_{ck}"
                            sections_html += f'''
            <div class="chart-container">
                <div id="{pid}"></div>
            </div>
            <script>
            (function() {{
                var fig = {self.figures[ck]};
                var el = document.getElementById('{pid}');
                Plotly.newPlot(el, fig.data, fig.layout, {{responsive: true, displayModeBar: true}});
            }})();
            </script>'''

                # Результаты этой подсекции
                if sec_key in analysis_results:
                    for ridx, (title, content) in enumerate(analysis_results[sec_key]):
                        # Проверяем, относится ли результат к текущей подсекции
                        if self._result_matches_subsection(sec_key, title, sub_num, ridx):
                            sections_html += f'''
            <div class="result-card">
                <div class="result-content">{content}</div>
            </div>'''

            sections_html += '</div>\n'

        # Оставшиеся результаты без секций
        orphan_results = ""
        for sec_key, items in analysis_results.items():
            for ridx, (title, content) in enumerate(items):
                if not self._result_placed(sec_key, title, section_defs, ridx):
                    orphan_results += f'''
            <div class="result-card">
                <h3>{title}</h3>
                <div class="result-content">{content}</div>
            </div>'''

        # Содержание
        toc_items = []
        for sec_key, sec_title, _ in section_defs:
            has_charts = sec_key in section_charts and any(
                c in self.figures for c, _ in section_charts[sec_key])
            has_results = sec_key in analysis_results and analysis_results[sec_key]
            if has_charts or has_results:
                toc_items.append(f'<a href="#section_{sec_key}">{sec_title}</a>')
        toc = f'<div class="toc"><b>Содержание:</b>{"".join(toc_items)}</div>' if toc_items else ''

        html_template = f"""
        <!DOCTYPE html><html lang="ru"><head>
            <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Интерактивный отчёт</title>
            <script src="{plotly_cdn}"></script>
            <style>
                :root {{ --bg: #f8f9fa; --card: #fff; --txt: #343a40; --brd: #dee2e6; }}
                * {{ box-sizing: border-box; }}
                body {{ font-family: 'Segoe UI', sans-serif; background: var(--bg); color: var(--txt); margin: 0; padding: 10px 20px; line-height: 1.6; }}
                h1 {{ text-align: center; color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #2980b9; border-left: 4px solid #3498db; padding-left: 10px; margin-top: 40px; }}
                h3 {{ color: #34495e; margin-top: 25px; }}
                .container {{ width: 100%; max-width: 100%; margin: 0; padding: 0 10px; }}
                .section {{ margin-bottom: 20px; }}
                .chart-container {{ background: var(--card); border: 1px solid var(--brd); border-radius: 8px; padding: 15px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); width: 100%; }}
                .result-card {{ background: var(--card); border: 1px solid var(--brd); border-radius: 8px; padding: 15px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); width: 100%; }}
                .result-content pre {{ background: #f1f3f5; padding: 15px; border-radius: 4px; overflow-x: auto; white-space: pre-wrap; }}
                .result-content ul {{ columns: 2; -webkit-columns: 2; }}
                .result-content li {{ padding: 4px 0; }}
                .toc {{ background: #f0f7ff; padding: 15px 25px; border-radius: 8px; margin-bottom: 30px; border: 1px solid #d0e3f7; }}
                .toc a {{ color: #2980b9; text-decoration: none; display: block; padding: 2px 0; }}
                .toc a:hover {{ text-decoration: underline; }}
                {self.stat_css}
            </style></head><body>
            <div class="container">
                <h1>Статистический анализ</h1>
                <p style="text-align: center; color: #6c757d;">Группировка: <b>{self.params.get('group', '')}</b> | Y: <b>{self.params.get('analysis', '')}</b></p>
                {toc}
                {sections_html}
                {orphan_results}
                <hr style="margin-top: 50px; border: 0; border-top: 1px solid #eee;">
                <p style="color: #999; font-size: 0.8em; text-align: center;">Сгенерировано DataAn Enhanced v2.1</p>
            </div>
            <script>
            document.querySelectorAll('.stat-table').forEach(function(table) {{
                var headers = table.querySelectorAll('th');
                headers.forEach(function(th, colIdx) {{
                    th.addEventListener('click', function() {{
                        var tbody = table.querySelector('tbody') || table;
                        var rows = Array.from(tbody.querySelectorAll('tr:not(:first-child)'));
                        var isAsc = th.classList.contains('sort-asc');
                        headers.forEach(function(h) {{ h.classList.remove('sort-asc', 'sort-desc'); }});
                        rows.sort(function(a, b) {{
                            var aVal = a.children[colIdx] ? a.children[colIdx].textContent.trim() : '';
                            var bVal = b.children[colIdx] ? b.children[colIdx].textContent.trim() : '';
                            var aNum = parseFloat(aVal.replace(/[±✓❌]/g, '').trim());
                            var bNum = parseFloat(bVal.replace(/[±✓❌]/g, '').trim());
                            if (!isNaN(aNum) && !isNaN(bNum)) {{
                                return isAsc ? bNum - aNum : aNum - bNum;
                            }}
                            return isAsc ? bVal.localeCompare(aVal, 'ru') : aVal.localeCompare(bVal, 'ru');
                        }});
                        th.classList.add(isAsc ? 'sort-desc' : 'sort-asc');
                        rows.forEach(function(row) {{ tbody.appendChild(row); }});
                    }});
                }});
            }});
            </script>
        </body></html>"""

        output_path.write_text(html_template, encoding='utf-8')

    def _format_result(self, key, data):
        if key == 'ml_benchmark':
            if data.get('html'): return data['html']
            tbl = data.get('table', [])
            if tbl:
                rows = "".join(f"<tr><td>{r.get('model','')}</td><td>{r.get('accuracy','')}</td><td>{r.get('auc','')}</td></tr>" for r in tbl)
                return f"<table class='stat-table'><tr><th>Модель</th><th>Accuracy</th><th>AUC</th></tr>{rows}</table>"
            return ""
        elif key == 'rf_importance':
            items_data = data
            if isinstance(items_data, dict) and 'features' in items_data:
                items = "".join(f"<li>{f}: {v:.4f}</li>" for f, v in zip(items_data['features'], items_data['values']))
            else:
                items = "".join(f"<li>{k}: {v:.4f}</li>" for k, v in sorted(items_data.items(), key=lambda x: x[1], reverse=True))
            return f"<ul>{items}</ul>"
        elif key == 'pca':
            loadings = data.get('loadings', {})
            n_95 = data.get('n_components_95', '')
            if loadings:
                features = list(loadings.keys())
                pcs = list(next(iter(loadings.values())).keys())
                header = ''.join(f'<th>{pc}</th>' for pc in pcs)
                rows_html = ''
                for f in features:
                    cells = ''.join(f'<td>{loadings[f].get(pc, 0):.3f}</td>' for pc in pcs)
                    rows_html += f'<tr><td><b>{f}</b></td>{cells}</tr>\n'
                table = (f'<p>Для 95% дисперсии необходимо {n_95} компонент(ы).</p>'
                         f'<table class="stat-table" style="width:auto;">'
                         f'<tr><th>Признак</th>{header}</tr>\n{rows_html}</table>')
                return table
            text = data.get('text', '')
            return f"<p>{text}</p>" if text else ''
        elif key == 'rfe':
            selected = data.get('selected', [])
            eliminated = data.get('eliminated', [])
            if not selected and not eliminated:
                text = data.get('text', '')
                return f"<p>{text}</p>" if text else ''
            return (f'<div style="display:flex; gap:20px; flex-wrap:wrap;">'
                    f'<div style="flex:1; min-width:300px; background:#e8f5e9; border-left:4px solid #4caf50; '
                    f'padding:12px 16px; border-radius:0 6px 6px 0;">'
                    f'<b>RFE рекомендует оставить ({len(selected)}):</b><br>{", ".join(selected)}</div>'
                    f'<div style="flex:1; min-width:300px; background:#ffeef0; border-left:4px solid #e53935; '
                    f'padding:12px 16px; border-radius:0 6px 6px 0;">'
                    f'<b>RFE рекомендует убрать ({len(eliminated)}):</b><br>{", ".join(eliminated)}</div>'
                    f'</div>')
        else:
            content = data.get('html', '')
            if not content:
                text = data.get('text', '')
                if text: content = f"<pre>{text}</pre>"
            return content or ''

    def _result_matches_subsection(self, sec_key, title, sub_num, ridx=0):
        if sec_key == 'preprocessing':
            idx_map = {'1.1': 0, '1.2': 1, '1.3': 2, '1.4': 3}
            return idx_map.get(sub_num) == ridx
        mapping = {
            'preprocessing': {'1.1': ['Статистика', 'предобработк'], '1.2': ['Пропуск'],
                              '1.3': ['Корреляц'],
                              '1.4': ['Анализ связей', 'категориальн']},
            'anova': {'3.1': ['One-Way', 'Kruskal-Wallis', 'Kruskal'], '3.2': ['Post-hoc', 'Tukey', 'Dunn'],
                      '3.3': ['Two-Way']},
            'manova': {'4.1': ['MANOVA'], '4.2': ['Post-hoc MANOVA', 'Tukey HSD']},
            'linear_regression': {'5.1': ['Линейная'], '5.2': ['Диагностика'], '5.3': ['Логистическая']},
            'feature_selection': {'6.1': [], '6.2': ['RFE']},
            'pca': {'7.1': ['PCA']},
            'cluster': {'8.3': ['ANOVA для кластеров']},
            'ml': {'9.1': ['ML Бенчмарк', 'Сравнение методов'], '9.2': ['Матрица ошибок', 'ROC']},
        }
        keywords = mapping.get(sec_key, {}).get(sub_num, [])
        if not keywords:
            return False
        return any(kw.lower() in title.lower() for kw in keywords)

    def _result_placed(self, sec_key, title, section_defs, ridx=0):
        for sk, _, subsections in section_defs:
            if sk != sec_key: continue
            for sub_num, _, _ in subsections:
                if self._result_matches_subsection(sec_key, title, sub_num, ridx):
                    return True
        return False


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

    builder = InteractiveReportBuilder(analyzer.df, analyzer.params, analyzer._analysis_results,
                                        preprocessing_stats=analyzer._preprocessing_stats,
                                        analyzer_module=analyzer_module)
    builder.generate_html(report_path)

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
