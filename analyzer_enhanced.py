# -*- coding: utf-8 -*-
"""
Enhanced DataAnalyzer v1.0
"""
import sys, gc, io, os, base64, warnings, json
import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from itertools import combinations
from scipy import stats as sp_stats
from scipy.stats import chi2_contingency, fisher_exact, shapiro, levene, kruskal
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, r2_score, mean_squared_error)
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.multivariate.manova import MANOVA

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output, HTML, FileLink
    _WIDGETS_AVAILABLE = True
except ImportError:
    _WIDGETS_AVAILABLE = False
    widgets = None
    display = None
    clear_output = None
    HTML = None
    FileLink = None


sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 10
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='seaborn')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='covariance of constraints')
warnings.filterwarnings('ignore', message='scipy.stats.shapiro')


# ====================== ДЕКОРАТОР ПРОВЕРКИ КЛАСТЕРОВ ======================
def _require_clusters(func):
    def wrapper(self, *args, **kwargs):
        if self._cluster_labels is None:
            msg = f"{func.__name__}: сначала выполните кластеризацию."
            return None if 'plot' in func.__name__ else msg
        return func(self, *args, **kwargs)
    return wrapper


# ====================== ЕДИНЫЙ СТИЛЬ ТАБЛИЦ ======================
STAT_TABLE_CSS = '''
.stat-table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 0.95em; }
.stat-table th { background: #3498db; color: white; padding: 10px 12px; border: 1px solid #2980b9; text-align: left; }
.stat-table td { padding: 8px 12px; border: 1px solid #d0d7de; }
.stat-table tr:nth-child(even) { background: #f8f9fa; }
.stat-table tr:hover { background: #eaf4fc; }
'''


class DataAnalyzer:
    """Универсальный класс для автостатического анализа данных."""

    _default_config = {
        'precision': 3,
        'correlation_threshold': 0.9,
        'z_score_threshold': 3.0,
        'bootstrap_min_size': 30,
        'bootstrap_max_ratio': 3.0,
        'remove_outliers': True,
        'balance_groups': False,
        'ml_n_repeats': 10,
        'show_boxplot_outliers': True,
    }

    def __init__(self, source, file_name=None, config_file=None):
        if isinstance(source, pd.DataFrame):
            self.df = source.copy()
            self.file_name = file_name or "Uploaded_Data"
            self._last_file_path = file_name or ''
        elif str(source).lower().endswith('.csv'):
            self.df = pd.read_csv(source)
            self.file_name = Path(source).name
            self._last_file_path = str(source)
        else:
            self.df = pd.read_excel(source)
            self.file_name = Path(source).name
            self._last_file_path = str(source)

        self.numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(exclude=['number']).columns.tolist()
        self.params = {}
        self.comments = {}
        self._analysis_results = {}
        self.correlation_removals = []
        self._current_df = None
        self._cluster_labels = None
        self._excluded_indices = pd.Index([])
        self._analyzed_indices = pd.Index([])
        self._preprocessing_stats = {}
        self._config = self._default_config.copy()
        if config_file:
            self.load_config(config_file)

    # ====================== КОНФИГУРАЦИЯ ======================
    def load_config(self, path=None):
        if path is None:
            path = os.path.join(os.getcwd(), 'analyzer_config.json')
        if not os.path.exists(path):
            return
        with open(path, 'r', encoding='utf-8') as f:
            user_cfg = json.load(f)
        for k, v in user_cfg.items():
            if k in self._default_config:
                self._config[k] = v

    def save_config(self, path='analyzer_config.json'):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, ensure_ascii=False, indent=2)

    # ====================== СЕССИЯ ======================
    @staticmethod
    def _session_path():
        return os.path.join(os.getcwd(), 'analyzer_session.json')

    def save_session(self, file_path=None, file_name=None):
        data = {
            'last_file_path': file_path or getattr(self, '_last_file_path', ''),
            'last_file_name': file_name or self.file_name,
            'last_params': self.params.copy(),
        }
        with open(self._session_path(), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_session(cls):
        path = cls._session_path()
        if not os.path.exists(path):
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # ====================== ВИДЖЕТЫ ======================
    def create_parameter_selector(self):
        if not _WIDGETS_AVAILABLE:
            return
        style = {'description_width': '200px'}
        layout = widgets.Layout(width='450px')
        self.w_group = widgets.Dropdown(options=self.categorical_cols,
                                        description='Группирующая:', style=style, layout=layout)
        self.w_analysis = widgets.Dropdown(options=self.numeric_cols,
                                           description='Для анализа (Y):', style=style, layout=layout)
        self.w_multi = widgets.SelectMultiple(
            options=self.numeric_cols,
            value=self.numeric_cols[:min(5, len(self.numeric_cols))],
            description='Многомерные (X):', rows=5, style=style, layout=layout)
        self.w_cat_multi = widgets.SelectMultiple(
            options=self.categorical_cols, value=[],
            description='Категориальные признаки:', layout=layout, style=style)
        btn_run = widgets.Button(description='Применить', button_style='success',
                                 layout=widgets.Layout(width='150px'))
        self.out = widgets.Output()

        def on_run(b):
            with self.out:
                clear_output()
                self.params = {
                    'group': self.w_group.value,
                    'analysis': self.w_analysis.value,
                    'multi': list(self.w_multi.value),
                    'cat_multi': list(self.w_cat_multi.value)
                }
                # Валидация типов
                issues = []
                g = self.params['group']
                if g and g in self.df.columns and pd.api.types.is_numeric_dtype(self.df[g]):
                    issues.append(f"'{g}' числовая — будет преобразована в строку")
                a = self.params['analysis']
                if a and a in self.df.columns and not pd.api.types.is_numeric_dtype(self.df[a]):
                    issues.append(f"'{a}' нечисловая — анализ может быть некорректным")
                for c in self.params['multi']:
                    if c in self.df.columns and not pd.api.types.is_numeric_dtype(self.df[c]):
                        issues.append(f"'{c}' нечисловая — исключена из X")
                for c in self.params['cat_multi']:
                    if c in self.df.columns and pd.api.types.is_numeric_dtype(self.df[c]):
                        issues.append(f"'{c}' числовая — исключена из категориальных")
                if issues:
                    print('Предупреждения:')
                    for iss in issues:
                        print(f'  - {iss}')
                print(f'Параметры сохранены.')

        btn_run.on_click(on_run)
        display(widgets.VBox([
            widgets.HTML('<b>Шаг 2: Параметры</b>'),
            self.w_group, self.w_analysis, self.w_multi, self.w_cat_multi,
            btn_run, self.out
        ]))

    def create_comment_widgets(self):
        if not _WIDGETS_AVAILABLE:
            return
        sections = {
            'preprocessing': '1. Предобработка',
            'plots': '2. Графики',
            'anova': '3. ANOVA / Tukey / Категориальные',
            'manova': '4. MANOVA / Post-hoc MANOVA',
            'linear_regression': '5. Линейная регрессия',
            'feature_selection': '6. Отбор признаков',
            'pca': '7. PCA',
            'cluster': '8. Кластерный анализ',
            'ml': '9. Машинное обучение',
        }
        self._comment_widgets = {}
        widgets_list = [widgets.HTML('<b>Шаг 3: Комментарии к отчёту</b>')]
        for key, label in sections.items():
            ta = widgets.Textarea(
                placeholder=f'Комментарий к блоку {label}...',
                layout=widgets.Layout(width='500px', height='70px'),
                description=f'{label}:', style={'description_width': '200px'})
            self._comment_widgets[key] = ta
            widgets_list.append(ta)

        def on_save(change):
            for key, w in self._comment_widgets.items():
                self.comments[key] = w.value

        for w in self._comment_widgets.values():
            w.observe(on_save, names='value')
        display(widgets.VBox(widgets_list))

    # ====================== ВСПОМОГАТЕЛЬНЫЕ ======================
    def _fmt(self, value):
        return f'{value:.{self._config["precision"]}f}'

    def _fig_to_base64(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return f'data:image/png;base64,{base64.b64encode(buf.read()).decode("utf-8")}'

    def _remove_highly_correlated(self, df, threshold=0.9):
        cols = self.params.get('multi', [])
        if len(cols) < 2:
            return df, []
        numeric_df = df[cols].select_dtypes(include=['number'])
        if numeric_df.shape[1] < 2:
            return df, []
        corr_matrix = numeric_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = set()
        removed_info = []
        for col in upper.columns:
            if col in to_drop:
                continue
            correlated = upper.index[upper[col] > threshold].tolist()
            if correlated:
                for corr_col in correlated:
                    if corr_col not in to_drop:
                        to_drop.add(corr_col)
                        removed_info.append((col, corr_col, upper.loc[corr_col, col]))
        if to_drop:
            df = df.drop(columns=list(to_drop))
        return df, removed_info

    def _bootstrap_balance_groups(self, df, group_col, target_size=None):
        sizes = df[group_col].value_counts()
        if target_size is None:
            target_size = int(sizes.median())
        rng = np.random.RandomState(42)
        parts = []
        for grp in sizes.index:
            sub = df[df[group_col] == grp]
            if len(sub) < target_size:
                sub = sub.sample(n=target_size, replace=True, random_state=rng)
            parts.append(sub)
        return pd.concat(parts, ignore_index=True)

    def _find_second_categorical_factor(self):
        if self._current_df is None or not self.params:
            return None
        g_col = self.params['group']
        for c in self.params.get('cat_multi', []):
            if c in self._current_df.columns and c != g_col:
                return c
        for c in self.params.get('multi', []):
            if (c in self._current_df.columns and c != g_col
                    and self._current_df[c].dtype in ['object', 'category']):
                return c
        return None

    # ====================== ПРЕДОБРАБОТКА ======================
    def _validate_params(self):
        """Проверка и корректировка типов признаков в params."""
        if not self.params:
            return
        df = self.df
        # group — категориальный
        g = self.params.get('group')
        if g and g in df.columns and pd.api.types.is_numeric_dtype(df[g]):
            print(f"  ВНИМАНИЕ: '{g}' числовая, но используется как группирующая. "
                  f"Преобразование в строку.")
        # analysis — числовая
        a = self.params.get('analysis')
        if a and a in df.columns and not pd.api.types.is_numeric_dtype(df[a]):
            # Попытка преобразовать
            try:
                df[a] = pd.to_numeric(df[a], errors='coerce')
                if df[a].isna().all():
                    print(f"  ОШИБКА: '{a}' не является числовой. Анализ невозможен.")
                else:
                    print(f"  ВНИМАНИЕ: '{a}' преобразована в числовую.")
            except Exception:
                print(f"  ОШИБКА: '{a}' не является числовой. Анализ невозможен.")
        # multi — числовые
        multi = self.params.get('multi', [])
        valid_multi = []
        for c in multi:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                valid_multi.append(c)
            elif c in df.columns:
                print(f"  ВНИМАНИЕ: '{c}' нечисловая — исключена из многомерных.")
        self.params['multi'] = valid_multi
        # cat_multi — категориальные
        cat_multi = self.params.get('cat_multi', [])
        valid_cat = []
        for c in cat_multi:
            if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
                valid_cat.append(c)
            elif c in df.columns:
                print(f"  ВНИМАНИЕ: '{c}' числовая — исключена из категориальных.")
        self.params['cat_multi'] = valid_cat

    def preprocess(self, remove_outliers=None, z_threshold=None, balance_groups=None):
        if not self.params:
            raise ValueError("Сначала выберите параметры!")
        self._validate_params()
        remove_outliers = self._config['remove_outliers'] if remove_outliers is None else remove_outliers
        z_threshold = self._config['z_score_threshold'] if z_threshold is None else z_threshold
        balance_groups = self._config['balance_groups'] if balance_groups is None else balance_groups
        corr_threshold = self._config['correlation_threshold']

        multi_cat = [c for c in self.params.get('cat_multi', []) if c in self.df.columns]
        cols = list(dict.fromkeys(
            [self.params['group'], self.params['analysis']] + self.params['multi'] + multi_cat))

        group_col = self.params['group']
        all_indices = self.df.index.copy()

        # Шаг 1: исключение строк без группирующей переменной
        group_missing_mask = self.df[group_col].isna()
        idx_without_group = all_indices[group_missing_mask]

        # Шаг 2: исключение строк с пропусками в других столбцах
        remaining_idx = all_indices[~group_missing_mask]
        non_group_cols = [c for c in cols if c != group_col]
        if non_group_cols:
            other_missing_mask = self.df.loc[remaining_idx, non_group_cols].isna().any(axis=1)
            idx_with_other_missing = remaining_idx[other_missing_mask]
        else:
            idx_with_other_missing = pd.Index([])

        self._analyzed_indices = remaining_idx.difference(idx_with_other_missing)
        self._excluded_indices = idx_without_group.append(idx_with_other_missing)

        df_work = self.df.loc[self._analyzed_indices, cols].copy()

        # Сохраняем статистику предобработки для HTML-отчёта
        self._preprocessing_stats = {
            'total_rows': len(self.df),
            'excluded_no_group': len(idx_without_group),
            'excluded_other_missing': len(idx_with_other_missing),
            'analyzed_before_outliers': len(df_work),
            'group_col': group_col,
            'missing_per_column': self.df[cols].isnull().sum().to_dict(),
        }

        df_work[group_col] = df_work[group_col].astype(str)
        for col in self.params.get('cat_multi', []):
            if col in df_work.columns:
                df_work[col] = df_work[col].astype(str)

        # Удаление выбросов
        n_outliers = 0
        if remove_outliers:
            num_cols = df_work.select_dtypes(include=['number']).columns
            stds = df_work[num_cols].std()
            valid_cols = stds[stds > 0].index
            if len(valid_cols) > 0:
                z_scores = np.abs((df_work[valid_cols] - df_work[valid_cols].mean()) / df_work[valid_cols].std())
                outlier_mask = (z_scores >= z_threshold).any(axis=1)
                n_outliers = outlier_mask.sum()
                if n_outliers > 0:
                    self._excluded_indices = self._excluded_indices.append(df_work.index[outlier_mask])
                    self._analyzed_indices = self._analyzed_indices.difference(df_work.index[outlier_mask])
                    df_work = df_work[~outlier_mask]
        self._preprocessing_stats['excluded_outliers'] = int(n_outliers)

        # Удаление высококоррелированных
        df_work, removed = self._remove_highly_correlated(df_work, threshold=corr_threshold)
        self.correlation_removals = removed
        self._preprocessing_stats['correlation_removals'] = removed

        self.params['multi'] = [c for c in self.params['multi'] if c in df_work.columns]
        self.params['cat_multi'] = [c for c in self.params.get('cat_multi', []) if c in df_work.columns]

        # Выравнивание групп
        if balance_groups:
            sizes = df_work[self.params['group']].value_counts()
            min_size = self._config['bootstrap_min_size']
            max_ratio = self._config['bootstrap_max_ratio']
            if any(s < min_size for s in sizes) or (sizes.min() > 0 and sizes.max() / sizes.min() > max_ratio):
                df_work = self._bootstrap_balance_groups(df_work, self.params['group'])

        self._preprocessing_stats['final_analyzed'] = len(df_work)
        self._preprocessing_stats['total_excluded'] = len(self._excluded_indices)
        self._current_df = df_work
        return df_work

    # ====================== АНАЛИЗ КАЧЕСТВА ДАННЫХ ======================
    def data_quality_report(self, df=None):
        if df is None:
            df = self._current_df if self._current_df is not None else self.df
        report = []
        report.append(f"Наблюдений: {len(df)}, Признаков: {len(df.columns)}")
        total_missing = df.isnull().sum().sum()
        report.append(f"Пропуски: {total_missing} ({100*total_missing/df.size:.1f}%)")
        res = "\n".join(report)
        self._analysis_results['data_quality'] = {'text': res}
        return res

    # ====================== ВИЗУАЛИЗАЦИЯ ======================
    def plot_violin(self, return_fig=False):
        fig, ax = plt.subplots(figsize=(10, 6))
        n = len(self._current_df)
        group_col = self.params['group']
        analysis_col = self.params['analysis']
        group_order = sorted(self._current_df[group_col].unique(), key=str)
        palette = sns.color_palette('Set2', n_colors=len(group_order))
        parts = sns.violinplot(x=group_col, y=analysis_col,
                               data=self._current_df, inner=None, palette=palette, ax=ax,
                               linewidth=1.5, saturation=0.85, order=group_order,
                               edgecolor='black', cut=0)
        for item in ax.collections:
            if isinstance(item, plt.matplotlib.collections.PolyCollection):
                item.set_edgecolor('black')
                item.set_linewidth(1.2)
                item.set_alpha(0.75)
        sns.boxplot(x=group_col, y=analysis_col,
                    data=self._current_df, width=0.15, order=group_order,
                    boxprops=dict(zorder=3, facecolor='white', edgecolor='black', linewidth=1.5),
                    whiskerprops=dict(color='black', linewidth=1.2),
                    capprops=dict(color='black', linewidth=1.2),
                    medianprops=dict(color='#e74c3c', linewidth=2.5),
                    showfliers=False, ax=ax)
        if n > 1000:
            sns.stripplot(x=group_col, y=analysis_col,
                          data=self._current_df, color='#2c3e50', size=2.5, ax=ax, alpha=0.3,
                          jitter=True, dodge=False, edgecolor='none')
        else:
            sns.swarmplot(x=group_col, y=analysis_col,
                          data=self._current_df, color='#2c3e50', size=3.5, ax=ax, alpha=0.65,
                          edgecolor='black', linewidth=0.4)
        ax.set_title(f'Скрипичная диаграмма: "{analysis_col}"', fontsize=14, fontweight='bold')
        ax.set_ylabel(analysis_col, fontsize=12)
        ax.set_xlabel(group_col, fontsize=12)
        ax.grid(axis='y', alpha=0.25, linestyle='-')
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor('#cccccc')
            spine.set_linewidth(0.8)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        if return_fig:
            return fig
        plt.show()

    def plot_boxplot_with_significance(self, return_fig=False):
        fig, ax = plt.subplots(figsize=(10, 6))
        g_col, a_col = self.params['group'], self.params['analysis']
        group_order = sorted(self._current_df[g_col].unique(), key=str)
        palette = sns.color_palette('Set2', n_colors=len(group_order))
        show_fliers = self._config.get('show_boxplot_outliers', True)
        sns.boxplot(x=g_col, y=a_col, data=self._current_df,
                    palette=palette, width=0.6, order=group_order, ax=ax,
                    showmeans=True, meanprops=dict(marker='D', markerfacecolor='white',
                    markeredgecolor='black', markersize=8, markeredgewidth=1.5),
                    boxprops=dict(edgecolor='black', linewidth=1.5, facecolor='white',
                                  alpha=0.85),
                    whiskerprops=dict(color='black', linewidth=1.2),
                    capprops=dict(color='black', linewidth=1.5),
                    medianprops=dict(color='#e74c3c', linewidth=2.5),
                    flierprops=dict(marker='o', markerfacecolor='#95a5a6', markersize=5,
                                    alpha=0.6, markeredgecolor='black', markeredgewidth=0.5),
                    showfliers=show_fliers)
        groups_data = [self._current_df[self._current_df[g_col] == g][a_col].dropna().values
                       for g in group_order]
        n_groups = len(group_order)
        bracket_y = None
        y_range = None
        if n_groups >= 2:
            y_max = self._current_df[a_col].max()
            y_range = self._current_df[a_col].max() - self._current_df[a_col].min()
            if y_range == 0:
                y_range = abs(y_max) if y_max != 0 else 1.0
            bracket_height = y_range * 0.04
            bracket_y = y_max + y_range * 0.05
            for i, j in combinations(range(n_groups), 2):
                g1_data, g2_data = groups_data[i], groups_data[j]
                if len(g1_data) < 3 or len(g2_data) < 3:
                    continue
                _, p_val = sp_stats.mannwhitneyu(g1_data, g2_data, alternative='two-sided')
                if p_val < 0.001:
                    sig_text = '***'
                elif p_val < 0.01:
                    sig_text = '**'
                elif p_val < 0.05:
                    sig_text = '*'
                else:
                    continue
                x1, x2 = i, j
                y = bracket_y
                ax.plot([x1, x1, x2, x2], [y, y + bracket_height * 0.3,
                                            y + bracket_height * 0.3, y],
                        lw=2.0, color='#2c3e50')
                ax.text((x1 + x2) / 2, y + bracket_height * 0.5, sig_text,
                        ha='center', va='bottom', fontsize=12, fontweight='bold',
                        color='#e74c3c')
                bracket_y += bracket_height * 1.2
        ax.set_ylim(ax.get_ylim()[0],
                    bracket_y + y_range * 0.05 if n_groups >= 2 else ax.get_ylim()[1])
        ax.set_title(f'Ящики с усами: "{a_col}"', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel(a_col, fontsize=12)
        ax.set_xlabel(g_col, fontsize=12)
        ax.grid(axis='y', alpha=0.25, linestyle='-')
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor('#cccccc')
            spine.set_linewidth(0.8)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        if return_fig:
            return fig
        plt.show()

    def plot_histograms(self, return_fig=False):
        cols = self.params.get('multi', [])[:6]
        if not cols:
            cols = self.numeric_cols[:6]
        cols = [c for c in cols if c in self._current_df.columns
                and pd.api.types.is_numeric_dtype(self._current_df[c])]
        n = len(cols)
        if n == 0:
            return None
        palette = sns.color_palette('Set2', n_colors=n)
        fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
        if n == 1:
            axes = [axes]
        for ax, col, clr in zip(axes, cols, palette):
            data = pd.to_numeric(self._current_df[col], errors='coerce').dropna()
            if data.empty:
                ax.text(0.5, 0.5, f'{col}: нет числовых данных', ha='center',
                        va='center', fontsize=10, transform=ax.transAxes)
                ax.set_title(col, fontsize=12, fontweight='bold')
                continue
            ax.hist(data, bins=30, alpha=0.85, color=clr, edgecolor='black', linewidth=1.0,
                    rwidth=0.92)
            mean_v = data.mean()
            med_v = data.median()
            mode_v = data.mode().iloc[0] if not data.mode().empty else np.nan
            for val, lbl, c in [(mean_v, f'Среднее={mean_v:.2f}', '#e74c3c'),
                                (med_v, f'Медиана={med_v:.2f}', '#27ae60'),
                                (mode_v, f'Мода={mode_v:.2f}', '#f39c12')]:
                try:
                    if not np.isnan(val):
                        ax.axvline(val, color=c, linestyle='--', linewidth=2.0, label=lbl)
                except (TypeError, ValueError):
                    continue
            ax.legend(fontsize=8, framealpha=0.95, edgecolor='gray', loc='best')
            ax.set_title(col, fontsize=13, fontweight='bold')
            ax.set_xlabel('')
            ax.grid(axis='y', alpha=0.25, linestyle='-')
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_edgecolor('#cccccc')
                spine.set_linewidth(0.8)
        plt.tight_layout()
        if return_fig:
            return fig
        plt.show()

    def plot_pie_chart(self, return_fig=False):
        cat_cols = [c for c in self.params.get('cat_multi', []) if c in self._current_df.columns]
        if not cat_cols:
            g_col = self.params['group']
            cat_cols = [g_col] if g_col in self._current_df.columns else []
        n = len(cat_cols)
        if n == 0:
            return None
        fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
        if n == 1:
            axes = [axes]
        palette = sns.color_palette('Set2')
        for ax, col in zip(axes, cat_cols):
            counts = self._current_df[col].value_counts()
            explode = [0.03] * len(counts)
            wedges, texts, autotexts = ax.pie(
                counts.values, labels=counts.index, autopct='%1.1f%%',
                startangle=90, explode=explode,
                colors=palette[:len(counts)],
                wedgeprops=dict(edgecolor='black', linewidth=1.2),
                textprops=dict(fontsize=10))
            for t in autotexts:
                t.set_fontsize(10)
                t.set_fontweight('bold')
                t.set_color('white')
            ax.set_title(f'{col}', fontsize=13, fontweight='bold')
        plt.tight_layout()
        if return_fig:
            return fig
        plt.show()

    def plot_scatter_with_regression(self, return_fig=False):
        cols = self.params.get('multi', [])
        num_cols = [c for c in cols if c in self._current_df.columns
                    and pd.api.types.is_numeric_dtype(self._current_df[c])]
        if len(num_cols) < 2:
            return None
        x, y = num_cols[0], num_cols[1]
        fig, ax = plt.subplots(figsize=(8, 6))
        data = self._current_df[[x, y]].dropna()
        sns.regplot(x=x, y=y, data=data, ax=ax, scatter_kws={'alpha': 0.55, 's': 40,
                    'edgecolor': 'black', 'linewidths': 0.5, 'color': '#3498db'},
                    line_kws={'color': '#e74c3c', 'linewidth': 2.5})
        r, p = sp_stats.pearsonr(data[x], data[y])
        ax.set_title(f'{x} vs {y}  (r={r:.3f}, p={p:.3e})', fontsize=13, fontweight='bold')
        ax.set_xlabel(x, fontsize=12)
        ax.set_ylabel(y, fontsize=12)
        ax.grid(True, alpha=0.25, linestyle='-')
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor('#cccccc')
            spine.set_linewidth(0.8)
        plt.tight_layout()
        if return_fig:
            return fig
        plt.show()

    def plot_pairgrid(self, return_fig=False):
        cols = self.params.get('multi', [])
        num_cols = [c for c in cols if c in self._current_df.columns
                    and pd.api.types.is_numeric_dtype(self._current_df[c])]
        if len(num_cols) < 3:
            return None
        cols_use = num_cols[:5]
        g = sns.PairGrid(self._current_df[cols_use], diag_sharey=False,
                         height=2.5, aspect=1.2)
        g.map_upper(sns.scatterplot, alpha=0.5, s=20, edgecolor='black', linewidth=0.3,
                    color='#3498db')
        g.map_lower(sns.kdeplot, levels=5, alpha=0.55, fill=True, linewidths=1.0,
                    cmap='Blues')
        g.map_diag(sns.histplot, kde=True, alpha=0.7, edgecolor='black', linewidth=0.8,
                    color='#2ecc71')
        for i, ax in enumerate(g.diag_axes):
            ax.set_ylabel('')
        g.fig.suptitle('Попарные зависимости признаков', fontsize=14, fontweight='bold', y=1.01)
        g.fig.tight_layout()
        fig = g.fig
        if return_fig:
            return fig
        plt.show()

    def plot_correlation_matrix(self, return_fig=False):
        fig = plt.figure(figsize=(10, 8))
        numeric_df = self._current_df[self.params['multi']].select_dtypes(include=['number'])
        if numeric_df.shape[1] < 2:
            plt.text(0.5, 0.5, 'Мало числовых признаков', ha='center', va='center', fontsize=14)
            plt.tight_layout()
            fig = plt.gcf()
            if return_fig:
                return fig
            plt.show()
            return
        corr = numeric_df.corr()
        n_vars = len(corr)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        p_vals = np.ones((n_vars, n_vars))
        for i, c1 in enumerate(numeric_df.columns):
            for j, c2 in enumerate(numeric_df.columns):
                if i < j:
                    _, p = sp_stats.pearsonr(numeric_df[c1].dropna(), numeric_df[c2].dropna())
                    p_vals[i, j] = p
                    p_vals[j, i] = p
        sig = p_vals < 0.05
        cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="light", as_cmap=True)
        ax = sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=cmap,
                         vmin=-1, vmax=1, square=True,
                         linewidths=1.5, linecolor='white',
                         cbar_kws={"shrink": 0.8, "label": "r"},
                         annot_kws={"fontsize": 10, "fontweight": "bold"})
        for i in range(n_vars):
            for j in range(n_vars):
                if i <= j:
                    continue
                if not sig[i, j]:
                    for text in ax.texts:
                        try:
                            tpos = text.get_position()
                        except Exception:
                            continue
                        if abs(tpos[0] - (j + 0.5)) < 0.1 and abs(tpos[1] - (i + 0.5)) < 0.1:
                            text.set_alpha(0.15)
                            text.set_color('#999999')
        plt.title("Матрица корреляций (серые — незначимые p≥0.05)", fontsize=13, fontweight='bold')
        plt.tight_layout()
        if return_fig:
            return fig
        plt.show()

    def plot_interaction_effect(self, return_fig=False):
        g_col = self.params['group']
        a_col = self.params['analysis']
        second_factor = self._find_second_categorical_factor()
        if second_factor is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'Нет второго категориального фактора',
                    ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_axis_off()
            plt.tight_layout()
            if return_fig:
                return fig
            plt.show()
            return None
        # Автоматический выбор: на оси X — фактор с большим числом уровней, hue — с меньшим
        n_g = self._current_df[g_col].nunique()
        n_s = self._current_df[second_factor].nunique()
        if n_g >= n_s:
            x_col, hue_col = g_col, second_factor
        else:
            x_col, hue_col = second_factor, g_col
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.pointplot(x=x_col, y=a_col, hue=hue_col, data=self._current_df,
                      dodge=True, capsize=0.1, err_kws={'linewidth': 1.8},
                      palette='Set2', ax=ax, markersize=8, linewidth=2.0)
        ax.set_title(f'Взаимодействие: {x_col} × {hue_col} на {a_col}',
                     fontsize=14, fontweight='bold')
        ax.set_ylabel(a_col, fontsize=12)
        ax.set_xlabel(x_col, fontsize=12)
        ax.grid(axis='y', alpha=0.25, linestyle='-')
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor('#cccccc')
            spine.set_linewidth(0.8)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        if return_fig:
            return fig
        plt.show()

    # ====================== ВСПОМОГАТЕЛЬНЫЕ СТАТИСТИЧЕСКИЕ МЕТОДЫ ======================
    def _dunn_test(self, data, group_col, analysis_col):
        groups_data = [(g, data.loc[data[group_col] == g, analysis_col].dropna())
                       for g in data[group_col].unique()]
        groups_data = [(g, vals) for g, vals in groups_data if len(vals) >= 2]
        if len(groups_data) < 2:
            return []
        N = sum(len(v) for _, v in groups_data)
        all_vals = np.concatenate([v.values for _, v in groups_data])
        ranks = sp_stats.rankdata(all_vals)
        pos = 0
        mean_ranks, n_per = {}, {}
        for g, vals in groups_data:
            n = len(vals)
            mean_ranks[g] = ranks[pos:pos + n].mean()
            n_per[g] = n
            pos += n
        rank_var = N * (N + 1) / 12
        pairs = []
        group_names = [g for g, _ in groups_data]
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                g1, g2 = group_names[i], group_names[j]
                z = (mean_ranks[g1] - mean_ranks[g2]) / np.sqrt(
                    rank_var * (1 / n_per[g1] + 1 / n_per[g2]))
                p = 2 * sp_stats.norm.sf(abs(z))
                pairs.append({'g1': g1, 'g2': g2, 'z': z, 'p': p})
        return pairs

    @staticmethod
    def _holm_correction(p_values):
        m = len(p_values)
        if m == 0:
            return []
        sorted_idx = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_idx]
        corrected = [min(1, p * (m - i)) for i, p in enumerate(sorted_p)]
        result = [0.0] * m
        for idx, val in zip(sorted_idx, corrected):
            result[idx] = val
        return result

    @staticmethod
    def _sidak_correction(p_values):
        m = len(p_values)
        if m == 0:
            return []
        return [min(1, 1 - (1 - p) ** m) for p in p_values]

    # ====================== СТАТИСТИЧЕСКИЙ АНАЛИЗ ======================
    def perform_anova_analysis(self):
        """ANOVA — результат в HTML-таблице с крупным форматированием."""
        g_col, a_col = self.params['group'], self.params['analysis']
        groups = [g_data[a_col].dropna() for _, g_data in self._current_df.groupby(g_col)]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='scipy.stats.shapiro')
            shapiro_p = [shapiro(g)[1] for g in groups if len(g) >= 3]
        is_normal = all(p > 0.05 for p in shapiro_p) if shapiro_p else False
        _, levene_p = levene(*groups)
        is_homogeneous = levene_p > 0.05
        
        result = {'is_normal': is_normal, 'is_homogeneous': is_homogeneous,
                  'shapiro_pvalues': shapiro_p, 'levene_pvalue': levene_p,
                  'method': '', 'text': '', 'is_significant': False}
        
        res_text = f'Проверка предпосылок:\n'
        res_text += f'  Нормальность (Shapiro-Wilk): {"Да" if is_normal else "Нет"}\n'
        res_text += f'  Гомогенность дисперсии (Levene): {"Да" if is_homogeneous else "Нет"}\n\n'
        
        if is_normal and is_homogeneous:
            model = ols(f'Q("{a_col}") ~ C(Q("{g_col}"))', data=self._current_df).fit()
            anova_table = anova_lm(model, typ=2)
            ss_effect = anova_table['sum_sq'].iloc[0]
            ss_total = anova_table['sum_sq'].sum()
            eta_sq = ss_effect / ss_total
            result['method'] = 'ANOVA'
            result['anova_table'] = anova_table
            result['eta_squared'] = eta_sq
            result['is_significant'] = anova_table['PR(>F)'].iloc[0] < 0.05
            p = anova_table['PR(>F)'].iloc[0]
            f_val = anova_table['F'].iloc[0]
            
            html_table = f'''
            <div>
                <div style="background:#e8f8e8; border-left:4px solid #27ae60; padding:10px 16px;
                    margin-bottom:15px; border-radius:0 6px 6px 0; font-size:0.95em;">
                    <b>Выбран параметрический тест</b> (ANOVA), т.к. данные распределены нормально
                    (Shapiro-Wilk p={shapiro_p[0]:.4f} > 0.05) и дисперсии гомогенны (Levene p={levene_p:.4f} > 0.05).
                </div>
                <h2 style="text-align: center; font-size: 28px;">
                    One-Way ANOVA: {a_col} по {g_col}
                </h2>
                <table class="stat-table">
                    <tr>
                        <th>Показатель</th>
                        <th>Значение</th>
                    </tr>
                    <tr>
                        <td style="padding: 15px; font-weight: bold;">F-статистика</td>
                        <td style="padding: 15px; font-size: 20px; color: #2c3e50;">{f_val:.3f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 15px; font-weight: bold;">p-value</td>
                        <td style="padding: 15px; font-size: 20px; color: #2c3e50;">{p:.4f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 15px; font-weight: bold;">η² (эта-квадрат)</td>
                        <td style="padding: 15px; font-size: 20px; color: #2c3e50;">{eta_sq:.3f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 15px; font-weight: bold;">Результат</td>
                        <td style="padding: 15px; font-size: 20px; font-weight: bold; 
                                   color: {'#27ae60' if p < 0.05 else '#c0392b'};">
                            {'✓ СТАТИСТИЧЕСКИ ЗНАЧИМО' if p < 0.05 else '✗ НЕ ЗНАЧИМО'}
                        </td>
                    </tr>
                </table>
            </div>
            '''
            result['html'] = html_table
            res_text += f"One-Way ANOVA: F={f_val:.3f}, p={p:.4f}, η²={eta_sq:.3f}\n"
            res_text += f"Интерпретация: {'статистически значимо' if p < 0.05 else 'не значимо'} (alpha = 0.05).\n"
        else:
            h_stat, p_val = kruskal(*groups)
            result['method'] = 'Kruskal-Wallis'
            result['h_statistic'] = h_stat
            result['kruskal_pvalue'] = p_val
            result['is_significant'] = p_val < 0.05
            
            # HTML-таблица с крупным форматированием
            html_table = f'''
            <div>
                <div style="background:#fff3e0; border-left:4px solid #f39c12; padding:10px 16px;
                    margin-bottom:15px; border-radius:0 6px 6px 0; font-size:0.95em;">
                    <b>Выбран непараметрический тест</b> (Kruskal-Wallis), т.к. данные распределены
                    ненормально (Shapiro-Wilk p &lt; 0.05) и/или дисперсии не гомогенны (Levene p &lt; 0.05).
                    Критерий не требует нормальности и гомогенности дисперсий.
                </div>
                <h2>
                    Kruskal-Wallis Test: {a_col} по {g_col}
                </h2>
                <table class="stat-table">
                    <tr>
                        <th style="font-size: 18px; padding: 15px;">Показатель</th>
                        <th style="font-size: 18px; padding: 15px;">Значение</th>
                    </tr>
                    <tr>
                        <td style="padding: 15px; font-weight: bold;">H-статистика</td>
                        <td style="padding: 15px; font-size: 20px; color: #2c3e50;">{h_stat:.3f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 15px; font-weight: bold;">p-value</td>
                        <td style="padding: 15px; font-size: 20px; color: #2c3e50;">{p_val:.4f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 15px; font-weight: bold;">Результат</td>
                        <td style="padding: 15px; font-size: 20px; font-weight: bold; 
                                   color: {'#27ae60' if p_val < 0.05 else '#c0392b'};">
                            {'✓ СТАТИСТИЧЕСКИ ЗНАЧИМО' if p_val < 0.05 else '✗ НЕ ЗНАЧИМО'}
                        </td>
                    </tr>
                </table>
            </div>
            '''
            result['html'] = html_table
            res_text += f"Kruskal-Wallis: H={h_stat:.3f}, p={p_val:.4f}\n"
            res_text += f"Интерпретация: {'статистически значимо' if p_val < 0.05 else 'не значимо'} (alpha = 0.05).\n"
        
        # Примечание об интерпретации
        interp_note = (
            '<div class="interp-note" style="background:#eef6ff; border-left:4px solid #3498db; '
            'padding:12px 16px; margin:15px 0; border-radius:0 6px 6px 0; font-size:0.95em;">'
            '<b>Интерпретация:</b> '
        )
        if result.get('method') == 'ANOVA':
            interp_note += (
                'p-value ниже 0.05 означает, что хотя бы одна группа значимо отличается от других. '
                'η² показывает долю дисперсии, объяснённую фактором: до 0.01 — малый эффект, '
                '0.01–0.06 — средний, 0.06–0.14 — большой, свыше 0.14 — очень большой. '
                'Для выявления конкретных различий между парами групп используйте пост-хок анализ Тьюки.'
            )
        else:
            interp_note += (
                'p-value ниже 0.05 означает, что распределения хотя бы одной группы '
                'статистически значимо различаются. Критерий непараметрический — не требует '
                'нормальности данных. Для выявления конкретных различий между парами групп '
                'выполнен пост-хок анализ Данна с поправкой Холма на множественные сравнения.'
            )
        interp_note += '</div>'
        result['html'] += interp_note

        result['text'] = res_text
        self._analysis_results['anova'] = result
        return res_text
        
    def perform_posthoc_tukey(self):
        anova_result = self._analysis_results.get('anova', {})
        if not anova_result.get('is_significant', False):
            msg = "Post-hoc анализ не выполняется: основной тест не значим (p >= 0.05)."
            self._analysis_results['tukey'] = {'text': msg, 'performed': False, 'html': ''}
            return msg

        g_col, a_col = self.params['group'], self.params['analysis']
        method = anova_result.get('method', 'ANOVA')

        if method == 'ANOVA':
            tukey = pairwise_tukeyhsd(self._current_df[a_col], self._current_df[g_col], alpha=0.05)
            df_t = tukey._results_table.data[1:]
            significant_rows = []
            for row in df_t:
                g1, g2, md, p, lo, hi, rej = row
                if rej:
                    significant_rows.append({
                        'g1': g1, 'g2': g2, 'md': float(md),
                        'p': float(p), 'lo': float(lo), 'hi': float(hi)
                    })
            method_label = 'Tukey HSD'
            method_title = 'Post-hoc Tukey HSD'
            value_label = 'Разность средних'
        else:
            pairs = self._dunn_test(self._current_df, g_col, a_col)
            if not pairs:
                msg = "Недостаточно данных для теста Данна."
                self._analysis_results['tukey'] = {'text': msg, 'performed': False, 'html': ''}
                return msg
            raw_p = [p['p'] for p in pairs]
            corrected_p = self._holm_correction(raw_p)
            significant_rows = []
            for pair, p_adj in zip(pairs, corrected_p):
                if p_adj < 0.05:
                    significant_rows.append({
                        'g1': pair['g1'], 'g2': pair['g2'],
                        'z': pair['z'], 'p': p_adj
                    })
            method_label = 'Dunn (Holm)'
            method_title = 'Post-hoc Dunn test (поправка Холма)'
            value_label = 'Z-статистика'
            # Сохраняем для справки
            self._analysis_results['dunn_details'] = {
                'pairs': pairs,
                'corrected_p': corrected_p,
                'correction': 'Holm'
            }

        if not significant_rows:
            html_table = f'<p style="font-size: 16px; color: #7f8c8d; font-style: italic;">Достоверных попарных различий не обнаружено (α = 0.05, {method_label}).</p>'
        else:
            html_rows = ''
            for r in significant_rows:
                val_str = f'{r["md"]:.3f}' if method == 'ANOVA' else f'{r["z"]:.3f}'
                html_rows += (f'<tr><td style="padding: 12px;">{r["g1"]}</td>'
                              f'<td style="padding: 12px;">{r["g2"]}</td>'
                              f'<td style="padding: 12px; font-size: 16px;">{val_str}</td>'
                              f'<td style="padding: 12px; font-size: 16px;">{r["p"]:.4f}</td>'
                              f'<td style="padding: 12px; font-size: 18px;">✅</td></tr>\n')
            html_table = (f'<p style="font-size: 16px; font-weight: bold; color: #27ae60;">'
                          f'Найдено значимых попарных различий: {len(significant_rows)} '
                          f'({method_label})</p>'
                          f'<table class="stat-table" style="font-size: 16px;">'
                          f'<tr><th style="padding: 12px; font-size: 16px;">Группа 1</th>'
                          f'<th style="padding: 12px; font-size: 16px;">Группа 2</th>'
                          f'<th style="padding: 12px; font-size: 16px;">{value_label}</th>'
                          f'<th style="padding: 12px; font-size: 16px;">p-скорр.</th>'
                          f'<th style="padding: 12px; font-size: 16px;">Значимость</th></tr>\n{html_rows}</table>')

        self._analysis_results['tukey'] = {
            'text': f'Значимых пар: {len(significant_rows)}',
            'html': html_table,
            'performed': True,
            'significant_count': len(significant_rows)
        }
        return f'Значимых пар: {len(significant_rows)} ({method_label})'

    def perform_two_way_anova(self):
        g_col = self.params['group']
        a_col = self.params['analysis']
        second_factor = self._find_second_categorical_factor()
        if second_factor is None:
            msg = "Для двухфакторного ANOVA необходим второй категориальный фактор."
            self._analysis_results['two_way'] = {'text': msg, 'html': ''}
            return msg

        try:
            model = ols(f'Q("{a_col}") ~ C(Q("{g_col}")) * C(Q("{second_factor}"))',
                        data=self._current_df).fit()
            anova_table = anova_lm(model, typ=2)
            clean_names = []
            for idx_name in anova_table.index:
                name = re.sub(r'C\(Q\("(.+?)"\)\)', r'\1', idx_name)
                name = re.sub(r'Q\("(.+?)"\)', r'\1', name)
                name = name.replace(':', ' × ')
                clean_names.append(name)
            anova_table.index = clean_names
            at = anova_table.round(3)

            # Вычисление partial eta-squared для каждого фактора
            ss_total = anova_table['sum_sq'].sum()
            eta_sq_dict = {}
            for idx_name in anova_table.index:
                ss = anova_table.loc[idx_name, 'sum_sq']
                eta_sq_dict[idx_name] = round(ss / ss_total, 3) if ss_total > 0 else 0

            html_rows = ''
            for idx_name in at.index:
                row = at.loc[idx_name]
                row_data = [f'{v:.3f}' if isinstance(v, float) else str(v) for v in row]
                sig = '✅' if row.get('PR(>F)', 1) < 0.05 else '❌'
                eta = eta_sq_dict.get(idx_name, 0)
                html_rows += (f'<tr><td>{idx_name}</td>'
                              f'{"".join(f"<td>{v}</td>" for v in row_data)}'
                              f'<td>{eta:.3f}</td><td>{sig}</td></tr>\n')
            cols = ''.join(f'<th>{c}</th>' for c in at.columns)
            html_table = (f'<table class="stat-table">'
                          f'<tr><th>Фактор</th>{cols}<th>η²</th><th>Значимость</th></tr>\n{html_rows}</table>')

            # Проверка значимости взаимодействия
            interaction_name = [n for n in at.index if '×' in n]
            interaction_sig = False
            for iname in interaction_name:
                if at.loc[iname, 'PR(>F)'] < 0.05:
                    interaction_sig = True
                    break

            # Примечание об интерпретации
            interp_note = (
                '<div class="interp-note" style="background:#eef6ff; border-left:4px solid #3498db; '
                'padding:12px 16px; margin:15px 0; border-radius:0 6px 6px 0; font-size:0.95em;">'
                '<b>Интерпретация:</b> '
                'Двухфакторный ANOVA оценивает три гипотезы одновременно: '
                '(1) основной эффект первого фактора, (2) основной эффект второго фактора, '
                '(3) взаимодействие факторов. '
                'η² показывает долю дисперсии: < 0.01 — малый, '
                '0.01–0.06 — средний, > 0.06 — большой эффект. '
            )
            simple_effects_html = ''
            if interaction_sig:
                interp_note += (
                    '<b>Взаимодействие значимо (p < 0.05)</b> — эффект каждого фактора '
                    'зависит от уровня другого. Выполнен анализ простых эффектов: '
                    'сравнения уровней первого фактора внутри каждого уровня второго фактора '
                    'с поправкой Бонферрони.'
                )
                # Анализ простых эффектов
                levels_second = sorted(self._current_df[second_factor].unique(), key=str)
                simple_rows = ''
                n_comparisons = 0
                for lev in levels_second:
                    sub = self._current_df[self._current_df[second_factor] == lev]
                    groups = [sub[sub[g_col] == g][a_col].dropna().values
                              for g in sorted(sub[g_col].unique(), key=str)
                              if len(sub[sub[g_col] == g]) >= 2]
                    valid_groups = [(g, sub[sub[g_col] == g][a_col].dropna().values)
                                    for g in sorted(sub[g_col].unique(), key=str)
                                    if len(sub[sub[g_col] == g]) >= 2]
                    if len(valid_groups) >= 2:
                        n_comparisons += 1
                        g_names = [g for g, _ in valid_groups]
                        g_vals = [v for _, v in valid_groups]
                        if len(valid_groups) == 2:
                            _, p_val = sp_stats.ttest_ind(*g_vals, equal_var=False)
                            stat_name = 't'
                            stat_val = _
                            test_name = 't-тест Уэлча'
                        else:
                            f_stat, p_val = sp_stats.f_oneway(*g_vals)
                            stat_name = 'F'
                            stat_val = f_stat
                            test_name = 'One-way ANOVA'
                        p_bonf = min(1, p_val * len(levels_second))
                        sig = '✅' if p_bonf < 0.05 else '❌'
                        simple_rows += (f'<tr><td>{second_factor}={lev}</td>'
                                        f'<td>{test_name}</td>'
                                        f'<td>{", ".join(g_names)}</td>'
                                        f'<td>{stat_val:.3f}</td>'
                                        f'<td>{p_bonf:.4f}</td><td>{sig}</td></tr>\n')
                if simple_rows:
                    simple_effects_html = (
                        f'<h4>Анализ простых эффектов (поправка Бонферрони: ×{len(levels_second)})</h4>'
                        f'<table class="stat-table">'
                        f'<tr><th>Уровень {second_factor}</th><th>Тест</th><th>Группы</th>'
                        f'<th>Статистика</th><th>p-скорр.</th><th>Значимость</th></tr>\n'
                        f'{simple_rows}</table>')
            else:
                interp_note += (
                    'Взаимодействие незначимо (p ≥ 0.05) — эффекты факторов '
                    'аддитивны и интерпретируются независимо друг от друга.'
                )
            interp_note += '</div>'
            html_table += interp_note
            if simple_effects_html:
                html_table += simple_effects_html

            self._analysis_results['two_way'] = {
                'text': at.to_string(), 'html': html_table,
                'anova_table': anova_table, 'second_factor': second_factor,
                'interaction_sig': interaction_sig
            }
        except Exception as e:
            self._analysis_results['two_way'] = {'text': f'Ошибка: {e}', 'html': ''}
        return self._analysis_results['two_way']['text']

    def perform_categorical_analysis(self):
        """Анализ связей категориальных признаков — результат в _analysis_results."""
        g_col = self.params['group']
        cat_cols = [c for c in self.params.get('cat_multi', []) if c in self._current_df.columns]
        if not cat_cols:
            cat_cols = [c for c in self.params['multi'] if c in self._current_df.columns
                        and (self._current_df[c].dtype == 'object' or self._current_df[c].nunique() < 10)]
        if len(cat_cols) < 1:
            self._analysis_results['categorical'] = {'text': 'Нет категориальных переменных.',
                                                      'html': '', 'results': []}
            return ''

        results = []
        pairs_done = set()
        for col1 in [g_col] + cat_cols:
            for col2 in [g_col] + cat_cols:
                if col1 >= col2 or (col1, col2) in pairs_done:
                    continue
                pairs_done.add((col1, col2))
                ct = pd.crosstab(self._current_df[col1], self._current_df[col2])
                chi2, p, dof, expected = chi2_contingency(ct)
                n = ct.sum().sum()
                min_dim = min(ct.shape) - 1
                cramers_v = (chi2 / (n * min_dim)) ** 0.5 if min_dim > 0 else 0
                results.append({
                    'pair': (col1, col2), 'chi2': chi2, 'p': p,
                    'cramers_v': cramers_v, 'crosstab': ct
                })

        n_sig = sum(1 for r in results if r['p'] <= 0.05)
        n_total = len(results)
        if n_sig == n_total:
            summary = f'Все {n_total} пар связаны (p ≤ 0.05).'
        elif n_sig == 0:
            summary = f'Ни одна из {n_total} пар не связана (p > 0.05).'
        else:
            summary = f'{n_sig} из {n_total} пар связаны.'

        html_rows = ''
        for r in results:
            c1, c2 = r['pair']
            icon = '✅' if r['p'] <= 0.05 else '❌'
            html_rows += (f'<tr><td>{c1} vs {c2}</td><td>{r["chi2"]:.3f}</td>'
                          f'<td>{r["p"]:.3f}</td><td>{r["cramers_v"]:.3f}</td>'
                          f'<td>{icon}</td></tr>\n')
        html_table = (f'<p><b>{summary}</b></p>'
                      f'<table class="stat-table">'
                      f'<tr><th>Пара</th><th>χ²</th><th>p-value</th>'
                      f'<th>V Крамера</th><th>Связь</th></tr>\n{html_rows}</table>')

        # Примечание об интерпретации
        interp_note = (
            '<div class="interp-note" style="background:#eef6ff; border-left:4px solid #3498db; '
            'padding:12px 16px; margin:15px 0; border-radius:0 6px 6px 0; font-size:0.95em;">'
            '<b>Интерпретация:</b> '
            'χ² проверяет нулевую гипотезу о независимости двух категориальных переменных. '
            'p ≤ 0.05 означает статистически значимую связь. V Крамера (0–1) показывает '
            'силу связи: до 0.1 — очень слабая, 0.1–0.3 — слабая, 0.3–0.5 — умеренная, '
            'свыше 0.5 — сильная. При малых ожидаемых частотах (< 5) критерий χ² может быть '
            'ненадёжным — используйте точный критерий Фишера.'
            '</div>'
        )
        html_table += interp_note

        self._analysis_results['categorical'] = {
            'text': summary, 'html': html_table, 'results': results
        }
        return summary

    def perform_frequency_analysis(self):
        """Таблицы частот — результат в _analysis_results."""
        cat_cols = [c for c in self.params.get('cat_multi', []) if c in self._current_df.columns]
        if not cat_cols and self.params['group'] in self._current_df.columns:
            cat_cols = [c for c in [self.params['group']] + self.params.get('multi', [])
                        if c in self._current_df.columns
                        and (self._current_df[c].dtype == 'object' or self._current_df[c].nunique() < 10)]
        if not cat_cols:
            self._analysis_results['frequency'] = {'html': '', 'text': 'Нет категориальных признаков.'}
            return ''

        html_parts = []
        for col in cat_cols:
            freq = self._current_df[col].value_counts().reset_index()
            freq.columns = [col, 'Частота']
            freq['%'] = (freq['Частота'] / freq['Частота'].sum() * 100).round(1)
            rows = ''
            for _, r in freq.iterrows():
                rows += f'<tr><td>{r[col]}</td><td>{r["Частота"]}</td><td>{r["%"]}</td></tr>\n'
            html_parts.append(
                f'<h4>{col}</h4><table class="stat-table" style="width:50%;">'
                f'<tr><th>Значение</th><th>Частота</th><th>%</th></tr>\n{rows}</table>')

        html_all = '\n'.join(html_parts)
        self._analysis_results['frequency'] = {'html': html_all, 'text': ''}
        return ''

    def perform_manova(self):
        """MANOVA — результат в виде HTML-таблицы + описания тестов."""
        g_col = self.params['group']
        all_numeric = [c for c in [self.params['analysis']] + self.params.get('multi', [])
                       if c in self._current_df.columns and pd.api.types.is_numeric_dtype(self._current_df[c])]
        dep_cols = list(dict.fromkeys(all_numeric))
        if len(dep_cols) < 2:
            self._analysis_results['manova'] = {
                'text': 'Для MANOVA необходимо минимум 2 зависимых переменных.',
                'html': '', 'descriptions': ''
            }
            return ''

        cat_factors = [g_col]
        for c in self.params.get('cat_multi', []):
            if c in self._current_df.columns and c != g_col:
                cat_factors.append(c)
        cat_factors = list(dict.fromkeys(cat_factors))

        dep_str = " + ".join([f'Q("{c}")' for c in dep_cols])

        test_descriptions = {
            "Pillai": ("Pillai's Trace",
                       "Наиболее устойчив к нарушениям предпосылок. Рекомендуется при небольших выборках."),
            "Wilks": ("Wilks' Lambda",
                      "Классический критерий, наиболее мощный при соблюдении всех предпосылок."),
            "Hotelling": ("Hotelling-Lawley Trace",
                          "Сумма собственных значений. Мощнее Pillai при больших эффектах."),
            "Roy": ("Roy's Greatest Root",
                    "Максимально мощный, когда эффекты сосредоточены вдоль одного направления.")
        }

        def _run_manova(formula, data):
            manova = MANOVA.from_formula(formula, data=data)
            return manova.mv_test()

        def _find_test_row(stat_df, test_key):
            test_key_lower = test_key.lower()
            for idx_name in stat_df.index:
                idx_lower = str(idx_name).lower()
                if test_key_lower in idx_lower or idx_lower in test_key_lower:
                    return idx_name, stat_df.loc[idx_name]
            return None, None

        def _extract_factor_results(mv_result, factor_key):
            """Извлекает F и p для всех 4 тестов для одного фактора."""
            available_keys = list(mv_result.results.keys())
            found_key = None
            for candidate in [f'C(Q("{factor_key}"))', f'Q("{factor_key}")', factor_key]:
                if candidate in mv_result.results:
                    found_key = candidate
                    break
            if found_key is None:
                for k in available_keys:
                    if factor_key in k:
                        found_key = k
                        break
            if found_key is None:
                return {}

            factor_data = mv_result.results[found_key]
            results = {}

            # Новая структура statsmodels
            if isinstance(factor_data, dict) and 'stat' in factor_data:
                stat_df = factor_data['stat']
                if isinstance(stat_df, pd.DataFrame):
                    for test_key, (display_name, _) in test_descriptions.items():
                        row_name, row = _find_test_row(stat_df, test_key)
                        if row is not None and 'F Value' in stat_df.columns and 'Pr > F' in stat_df.columns:
                            try:
                                results[display_name] = {
                                    'F': float(row['F Value']),
                                    'p': float(row['Pr > F']),
                                }
                            except (ValueError, TypeError):
                                continue
            else:
                # Старая структура
                tests_old = ["Pillai's Trace", "Wilks' Lambda",
                             "Hotelling-Lawley Trace", "Roy's Greatest Root"]
                for test_name in tests_old:
                    if test_name in factor_data:
                        r = factor_data[test_name]
                        if isinstance(r, pd.DataFrame) and 'F Value' in r.columns and 'Pr > F' in r.columns:
                            try:
                                results[test_name] = {
                                    'F': float(r['F Value'].iloc[0]),
                                    'p': float(r['Pr > F'].iloc[0]),
                                }
                            except (ValueError, TypeError, IndexError):
                                continue
            return results

        # Выполнение MANOVA
        all_factor_results = {}
        p_values_all = []
        try:
            formula_full = f'{dep_str} ~ {" + ".join([f"C(Q(\"{f}\"))" for f in cat_factors])}'
            manova_result = _run_manova(formula_full, self._current_df)
            for factor in cat_factors:
                all_factor_results[factor] = _extract_factor_results(manova_result, factor)
                for test_name, vals in all_factor_results[factor].items():
                    p_values_all.append(vals['p'])
        except Exception as e_multi:
            for factor in cat_factors:
                try:
                    formula_one = f'{dep_str} ~ C(Q("{factor}"))'
                    manova_one = _run_manova(formula_one, self._current_df)
                    all_factor_results[factor] = _extract_factor_results(manova_one, factor)
                    for test_name, vals in all_factor_results[factor].items():
                        p_values_all.append(vals['p'])
                except Exception:
                    all_factor_results[factor] = {}

        # Формирование HTML-таблицы
        test_names_ordered = ["Pillai's Trace", "Wilks' Lambda",
                              "Hotelling-Lawley Trace", "Roy's Greatest Root"]
        html_rows = ''
        for factor in cat_factors:
            factor_res = all_factor_results.get(factor, {})
            for test_name in test_names_ordered:
                if test_name in factor_res:
                    f_val = factor_res[test_name]['F']
                    p_val = factor_res[test_name]['p']
                    sig = '✅' if p_val <= 0.05 else '❌'
                    html_rows += (f'<tr><td>{factor}</td><td>{test_name}</td>'
                                  f'<td>{f_val:.3f}</td><td>{p_val:.4f}</td><td>{sig}</td></tr>\n')

        if html_rows:
            html_table = (f'<table class="stat-table">'
                          f'<tr><th>Фактор</th><th>Критерий</th><th>F</th>'
                          f'<th>p-value</th><th>Значимость</th></tr>\n{html_rows}</table>')
        else:
            html_table = '<p><i>Не удалось получить результаты MANOVA.</i></p>'

        # Общий вывод
        if p_values_all:
            p_min = min(p_values_all)
            any_sig = any(p < 0.05 for p in p_values_all)
            if any_sig:
                conclusion = (f'<p><b>Вывод:</b> многомерный эффект факторов '
                              f'<span style="color:green;">ЗНАЧИМ</span> '
                              f'(минимальный p = {p_min:.4f}, α = 0.05).</p>')
            else:
                conclusion = (f'<p><b>Вывод:</b> многомерный эффект факторов '
                              f'<span style="color:#c0392b;">НЕ ЗНАЧИМ</span> '
                              f'(минимальный p = {p_min:.4f}, α = 0.05).</p>')
        else:
            conclusion = '<p><i>Не удалось получить результаты ни для одного теста.</i></p>'

        # Описания критериев
        desc_html = '<h4>Особенности критериев MANOVA:</h4><ul>'
        for _, (display_name, description) in test_descriptions.items():
            desc_html += f'<li><b>{display_name}:</b> {description}</li>'
        desc_html += '</ul>'
        desc_html += ('<p><i>Примечание: MANOVA анализирует влияние факторов на совокупность '
                      'зависимых (числовых) переменных одновременно, учитывая корреляции '
                      'между этими зависимыми переменными. В отличие от отдельных ANOVA для '
                      'каждой переменной, MANOVA оценивает общий эффект, сохраняя информацию '
                      'о взаимосвязях между независимыми переменными.</i></p>')

        text_summary = f"MANOVA: {len(dep_cols)} зависимых, {len(cat_factors)} факторов.\n"
        if p_values_all:
            text_summary += f"Минимальный p = {min(p_values_all):.4f}\n"

        self._analysis_results['manova'] = {
            'text': text_summary,
            'html': html_table,
            'conclusion': conclusion,
            'descriptions': desc_html,
            'dep_cols': dep_cols,
            'cat_factors': cat_factors,
        }
        return text_summary

    def perform_posthoc_manova(self):
        """Post-hoc MANOVA — только значимые различия."""
        g_col = self.params['group']
        all_numeric = [c for c in [self.params['analysis']] + self.params.get('multi', [])
                       if c in self._current_df.columns and pd.api.types.is_numeric_dtype(self._current_df[c])]
        dep_cols = list(dict.fromkeys(all_numeric))
        if len(dep_cols) < 1:
            self._analysis_results['posthoc_manova'] = {'text': '', 'html': ''}
            return ''

        groups = self._current_df[g_col].unique()
        if len(groups) < 2:
            self._analysis_results['posthoc_manova'] = {'text': '', 'html': ''}
            return ''

        significant_rows = []
        for dv in dep_cols:
            sub = self._current_df[[g_col, dv]].dropna()
            try:
                tukey = pairwise_tukeyhsd(sub[dv], sub[g_col], alpha=0.05)
                df_t = tukey._results_table.data[1:]
                for row in df_t:
                    g1, g2, md, p, lo, hi, rej = row
                    if rej:  # ТОЛЬКО значимые различия
                        significant_rows.append({
                            'variable': dv, 'g1': g1, 'g2': g2,
                            'md': float(md), 'p': float(p),
                            'lo': float(lo), 'hi': float(hi)
                        })
            except Exception:
                continue

        if not significant_rows:
            html_table = '<p><i>Достоверных попарных различий не обнаружено (α = 0.05).</i></p>'
        else:
            # Сводка по переменным: сколько значимых попарных различий у каждой
            var_counts = {}
            for r in significant_rows:
                v = r['variable']
                var_counts.setdefault(v, []).append(r)
            sorted_vars = sorted(var_counts.items(), key=lambda x: -len(x[1]))
            show_vars = sorted_vars[:10]
            total_vars = len(sorted_vars)

            summary_html = '<h4>Резюме: значимые различия по переменным</h4>'
            summary_html += '<table class="stat-table" style="width:60%;">'
            summary_html += '<tr><th>Переменная</th><th>Значимых пар</th><th>Примечание</th></tr>\n'
            for v, rows in show_vars:
                pairs_list = [f'{r["g1"]} vs {r["g2"]}' for r in rows]
                pairs_str = ', '.join(pairs_list[:3])
                if len(pairs_list) > 3:
                    pairs_str += f' (+{len(pairs_list)-3})'
                summary_html += f'<tr><td>{v}</td><td>{len(rows)}</td><td>{pairs_str}</td></tr>\n'
            if total_vars > 10:
                summary_html += f'<tr><td colspan="3"><i>... и ещё {total_vars - 10} переменных</i></td></tr>'
            summary_html += '</table>'

            html_rows = ''
            for r in significant_rows:
                html_rows += (f'<tr><td>{r["variable"]}</td><td>{r["g1"]}</td><td>{r["g2"]}</td>'
                              f'<td>{r["md"]:.3f}</td><td>{r["p"]:.4f}</td>'
                              f'<td>{r["lo"]:.3f}</td><td>{r["hi"]:.3f}</td><td>✅</td></tr>\n')
            html_table = (f'<p><b>Найдено значимых попарных различий: {len(significant_rows)}</b></p>'
                          f'{summary_html}'
                          f'<details><summary style="cursor:pointer;font-weight:bold;">Полная таблица ({len(significant_rows)} строк)</summary>'
                          f'<table class="stat-table">'
                          f'<tr><th>Переменная</th><th>Группа 1</th><th>Группа 2</th>'
                          f'<th>Разность</th><th>p-adj</th><th>Нижняя гр.</th>'
                          f'<th>Верхняя гр.</th><th>Различие</th></tr>\n{html_rows}</table></details>')

        self._analysis_results['posthoc_manova'] = {
            'text': f'Значимых различий: {len(significant_rows)}',
            'html': html_table,
            'count': len(significant_rows)
        }
        return f'Значимых различий: {len(significant_rows)}'



    def perform_np_manova(self):
        self._analysis_results['np_manova'] = {'text': '', 'html': ''}
        return ''

    # ====================== РЕГРЕССИОННЫЙ АНАЛИЗ ======================
    def perform_linear_regression(self):
        """Линейная регрессия — результат в HTML-таблице."""
        a_col = self.params['analysis']
        predictors = [c for c in self.params.get('multi', [])
                      if c in self._current_df.columns and pd.api.types.is_numeric_dtype(self._current_df[c])
                      and c != a_col]
        if not predictors:
            self._analysis_results['linear_regression'] = {
                'text': 'Нет предикторов.', 'html': ''
            }
            return ''

        data = self._current_df[[a_col] + predictors].dropna()
        X = data[predictors].values
        y = data[a_col].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = np.mean(np.abs(y_test - y_pred))

        # HTML-таблица коэффициентов — только top-5 по |коэффициент|
        coef_df = pd.DataFrame({'Предиктор': predictors, 'Коэффициент': model.coef_})
        coef_df['abs_coef'] = coef_df['Коэффициент'].abs()
        coef_df = coef_df.sort_values('abs_coef', ascending=False)
        top5 = coef_df.head(5)
        coef_rows = ''
        for _, row in top5.iterrows():
            coef_rows += f'<tr><td>{row["Предиктор"]}</td><td>{row["Коэффициент"]:.4f}</td></tr>\n'
        coef_rows += f'<tr><td><b>Intercept</b></td><td><b>{model.intercept_:.4f}</b></td></tr>\n'
        total_pred = len(predictors)
        shown = min(5, total_pred)

        # Уравнение — всегда все предикторы
        eq_parts = [f"{c:.3f}·{p}" for p, c in zip(predictors, model.coef_)]
        eq_str = f'{a_col} = ' + ' + '.join(eq_parts) + f' + {model.intercept_:.3f}'

        # Примечание об интерпретации — после метрик, перед коэффициентами
        interp_note = (
            '<div class="interp-note" style="background:#eef6ff; border-left:4px solid #3498db; '
            'padding:12px 16px; margin:15px 0; border-radius:0 6px 6px 0; font-size:0.95em;">'
            '<b>Интерпретация:</b> '
            f'R² = {r2:.3f} означает, что модель объясняет {r2*100:.1f}% дисперсии зависимой '
            'переменной. Коэффициент показывает изменение Y при увеличении предиктора на 1 единицу '
            'при фиксированных остальных. RMSE — стандартная ошибка предсказания в тех же единицах, '
            'что и Y. MAE — средняя абсолютная ошибка. R² > 0.7 считается хорошей моделью, '
            '0.5–0.7 — удовлетворительной, ниже 0.5 — слабой.'
            '</div>'
        )

        html_table = (
            f'<h4>Метрики модели</h4>'
            f'<table class="stat-table" style="width:50%;">'
            f'<tr><th>Метрика</th><th>Значение</th></tr>'
            f'<tr><td>R²</td><td>{r2:.4f}</td></tr>'
            f'<tr><td>RMSE</td><td>{rmse:.4f}</td></tr>'
            f'<tr><td>MAE</td><td>{mae:.4f}</td></tr>'
            f'<tr><td>N (test)</td><td>{len(y_test)}</td></tr>'
            f'</table>'
            f'{interp_note}'
            f'<h4>Топ-{shown} коэффициентов (из {total_pred})</h4>'
            f'<table class="stat-table" style="width:60%;">'
            f'<tr><th>Предиктор</th><th>Коэффициент</th></tr>'
            f'{coef_rows}</table>'
            f'<p><b>Уравнение:</b></p>'
            f'<div style="overflow-x:auto; padding:10px; background:#f8f9fa; border-radius:6px; '
            f'border:1px solid #e9ecef; font-family:monospace; font-size:0.95em; margin:10px 0;">'
            f'{eq_str}</div>'
        )

        self._analysis_results['linear_regression'] = {
            'text': f'R²={r2:.3f}, RMSE={rmse:.3f}',
            'html': html_table,
            'model': model, 'r2': r2, 'rmse': rmse,
            'predictors': predictors, 'a_col': a_col
        }
        return f'R²={r2:.3f}, RMSE={rmse:.3f}'

    def plot_regression_diagnostics(self, return_fig=False):
        lr_res = self._analysis_results.get('linear_regression', {})
        if 'model' not in lr_res:
            return None
        a_col = lr_res['a_col']
        predictors = lr_res['predictors']
        data = self._current_df[[a_col] + predictors].dropna()
        X = data[predictors].values
        y = data[a_col].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = lr_res['model']
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(y_pred, residuals, alpha=0.6, s=40, edgecolor='black', linewidths=0.4,
                        color='#3498db')
        axes[0].axhline(0, color='#e74c3c', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Предсказанные', fontsize=12)
        axes[0].set_ylabel('Остатки', fontsize=12)
        axes[0].set_title('Остатки vs Предсказанные', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.25, linestyle='-')
        for spine in axes[0].spines.values():
            spine.set_edgecolor('#cccccc')
            spine.set_linewidth(0.8)
        sp_stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q plot остатков', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.25, linestyle='-')
        for spine in axes[1].spines.values():
            spine.set_edgecolor('#cccccc')
            spine.set_linewidth(0.8)
        plt.tight_layout()
        if return_fig:
            return fig
        plt.show()

    def perform_logistic_regression_cat(self):
        g_col = self.params['group']
        cat_preds = [c for c in self.params.get('cat_multi', [])
                     if c in self._current_df.columns and c != g_col]
        num_preds = [c for c in self.params.get('multi', [])
                     if c in self._current_df.columns
                     and pd.api.types.is_numeric_dtype(self._current_df[c])]
        if not cat_preds or self._current_df[g_col].nunique() < 2:
            self._analysis_results['logistic_reg_cat'] = {'text': '', 'html': ''}
            return ''
        predictors = cat_preds + num_preds
        data = self._current_df[predictors + [g_col]].dropna()
        X = data[predictors].copy()
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X = pd.concat([X, pd.get_dummies(X[col], prefix=col, drop_first=True)], axis=1)
            X.drop(col, axis=1, inplace=True)
        le = LabelEncoder()
        y = le.fit_transform(data[g_col])
        if X.shape[1] < 1 or len(np.unique(y)) < 2:
            self._analysis_results['logistic_reg_cat'] = {'text': '', 'html': ''}
            return ''
        from sklearn.base import clone
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            model = LogisticRegression(solver='lbfgs', max_iter=2000, random_state=42, class_weight='balanced')
            scores = cross_val_score(model, X, y, cv=min(5, len(np.unique(y))), scoring='accuracy')
            model.fit(X, y)
        class_names = le.classes_
        coef = model.coef_
        intercept = model.intercept_
        feature_names = X.columns.tolist()
        coef_rows = ''
        for f_idx, f_name in enumerate(feature_names):
            if len(class_names) == 2:
                coef_val = coef[0, f_idx]
                coef_rows += f'<tr><td>{f_name}</td><td>{coef_val:.4f}</td></tr>\n'
            else:
                vals = ' / '.join([f'{coef[c_idx, f_idx]:.4f}' for c_idx in range(len(class_names))])
                coef_rows += f'<tr><td>{f_name}</td><td>{vals}</td></tr>\n'
        inter_rows = ''
        for c_idx, cn in enumerate(class_names):
            inter_rows += f'<tr><td>{cn}</td><td>{intercept[c_idx]:.4f}</td></tr>\n'
        html_table = (
            f'<h4>Логистическая регрессия (целевая: {g_col})</h4>'
            f'<p><b>Точность (cross-val):</b> {scores.mean():.3f} ± {scores.std():.3f}</p>'
            f'<h4>Коэффициенты</h4>'
            f'<table class="stat-table" style="width:80%;">'
            f'<tr><th>Предиктор</th><th>{("Коэф." if len(class_names) <= 2 else "Коэф. по классам")}</th></tr>\n'
            f'{coef_rows}</table>'
            f'<h4>Свободные члены</h4>'
            f'<table class="stat-table" style="width:50%;">'
            f'<tr><th>Класс</th><th>Intercept</th></tr>\n{inter_rows}</table>'
            f'<div class="interp-note" style="background:#eef6ff; border-left:4px solid #3498db; '
            f'padding:12px 16px; margin:15px 0; border-radius:0 6px 6px 0; font-size:0.95em;">'
            f'<b>Интерпретация:</b> '
            f'Логистическая регрессия моделирует логарифм отношения шансов (log-odds) '
            f'принадлежности к целевому классу. Положительный коэффициент — предиктор увеличивает '
            f'шансы принадлежности к классу, отрицательный — уменьшает. '
            f'Точность cross-val {scores.mean():.3f} — средняя доля правильных предсказаний '
            f'при кросс-валидации.</div>'
        )
        self._analysis_results['logistic_reg_cat'] = {
            'text': f'LogReg точность: {scores.mean():.3f}',
            'html': html_table
        }
        return ''

    # ====================== МАШИННОЕ ОБУЧЕНИЕ ======================
    def _align_features(self, X_train, X_test):
        all_cols = pd.concat([X_train, X_test], axis=0).columns
        X_train = X_train.reindex(columns=all_cols, fill_value=0)
        X_test = X_test.reindex(columns=all_cols, fill_value=0)
        return X_train, X_test

    def _prepare_ml_data(self, df, test_size=0.3, random_state=42):
        X = df[self.params['multi']].copy()
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X = pd.concat([X, pd.get_dummies(X[col], prefix=col, drop_first=True)], axis=1)
            X.drop(col, axis=1, inplace=True)
        le = LabelEncoder()
        y = le.fit_transform(df[self.params['group']])
        class_names = le.classes_
        target_min = max(5, min(np.bincount(y)))
        rng = np.random.RandomState(random_state)
        X_resampled, y_resampled = [], []
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            if len(idx) < target_min:
                oversampled = rng.choice(idx, size=target_min, replace=True)
            else:
                oversampled = idx
            X_resampled.append(X.iloc[oversampled])
            y_resampled.append(np.full(len(oversampled), cls))
        X = pd.concat(X_resampled, ignore_index=True)
        y = np.concatenate(y_resampled)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y)
        X_train, X_test = self._align_features(X_train, X_test)
        return X_train, X_test, y_train, y_test, class_names

    def _check_ml_prerequisites(self, df):
        if len(self.params['multi']) == 0:
            raise ValueError("Нет многомерных признаков.")
        if df[self.params['group']].nunique() < 2:
            raise ValueError("Только один класс.")
        return True

    def _make_model(self, name):
        models = {
            'Random Forest': (RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight='balanced'), False),
            'LDA': (LinearDiscriminantAnalysis(), True),
            'SVM (RBF)': (SVC(kernel='rbf', probability=True, random_state=42,
                              class_weight='balanced'), True),
            'SVM (Poly)': (SVC(kernel='poly', degree=3, probability=True, random_state=42,
                               class_weight='balanced'), True),
            'Logistic Regression': (LogisticRegression(
                solver='lbfgs', max_iter=1000, random_state=42, class_weight='balanced'), True),
            'Decision Tree': (DecisionTreeClassifier(
                max_depth=5, random_state=42, class_weight='balanced'), False),
        }
        if _XGB_AVAILABLE:
            models['XGBoost'] = (xgb.XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=42, verbosity=0), False)
        return models.get(name)

    def _train_model(self, df, model, model_name, test_size=0.3, use_scaler=False):
        """Обучение модели с 10 повторными делениями на train/test."""
        try:
            self._check_ml_prerequisites(df)
            n_repeats = self._config.get('ml_n_repeats', 10)

            accuracies, aucs = [], []
            last_y_test, last_y_pred, last_y_proba = None, None, None
            class_names = None

            for i in range(n_repeats):
                seed = 42 + i
                X_train, X_test, y_train, y_test, class_names = self._prepare_ml_data(
                    df, test_size, random_state=seed)

                model_iter = self._make_model(model_name)[0]
                # Клонируем модель, чтобы не накапливать состояние
                from sklearn.base import clone
                model_iter = clone(model_iter)

                scaler = None
                if use_scaler:
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_train)
                    X_te = scaler.transform(X_test)
                else:
                    X_tr, X_te = X_train.values, X_test.values

                model_iter.fit(X_tr, y_train)
                y_pred = model_iter.predict(X_te)
                acc = accuracy_score(y_test, y_pred)
                accuracies.append(acc)

                if hasattr(model_iter, 'predict_proba'):
                    y_proba = model_iter.predict_proba(X_te)
                    n_cls = len(class_names)
                    if n_cls == 2:
                        auc = roc_auc_score(y_test, y_proba[:, 1])
                    else:
                        try:
                            auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
                        except Exception:
                            auc = 0.0
                    aucs.append(auc)

                # Сохраняем результаты последней итерации для визуализации
                last_y_test, last_y_pred, last_y_proba = y_test, y_pred, y_proba if hasattr(model_iter, 'predict_proba') else None

            acc_mean, acc_std = np.mean(accuracies), np.std(accuracies)
            auc_mean = np.mean(aucs) if aucs else 0.0
            auc_std = np.std(aucs) if aucs else 0.0

            key = model_name.lower().replace(' ', '_')
            self._analysis_results[key] = {
                'accuracy_mean': acc_mean,
                'accuracy_std': acc_std,
                'auc_mean': auc_mean,
                'auc_std': auc_std,
                'accuracies': accuracies,
                'n_repeats': n_repeats,
                'y_test': last_y_test,
                'y_pred': last_y_pred,
                'y_proba': last_y_proba,
                'class_names': class_names,
                'model_name': model_name,
            }
        except Exception as e:
            key = model_name.lower().replace(' ', '_')
            self._analysis_results[key] = {
                'accuracy_mean': 0, 'accuracy_std': 0,
                'auc_mean': 0, 'auc_std': 0,
                'error': str(e), 'model_name': model_name,
            }

    def train_model(self, model_name, df=None, test_size=0.3):
        if df is None:
            df = self._current_df
        spec = self._make_model(model_name)
        if spec is None:
            return
        model, use_scaler = spec
        self._train_model(df, model, model_name, test_size, use_scaler)

    # Обратная совместимость
    def train_random_forest(self, df=None, test_size=0.3):
        self.train_model('Random Forest', df, test_size)

    def train_lda(self, df=None, test_size=0.3):
        self.train_model('LDA', df, test_size)

    def train_svm_rbf(self, df=None, test_size=0.3):
        self.train_model('SVM (RBF)', df, test_size)

    def train_svm_poly(self, df=None, test_size=0.3):
        self.train_model('SVM (Poly)', df, test_size)

    def train_logistic_regression(self, df=None, test_size=0.3):
        self.train_model('Logistic Regression', df, test_size)

    def train_decision_tree(self, df=None, test_size=0.3):
        self.train_model('Decision Tree', df, test_size)

    def train_xgboost(self, df=None, test_size=0.3):
        if not _XGB_AVAILABLE:
            return
        self.train_model('XGBoost', df, test_size)

    def ml_benchmark(self, df=None, test_size=0.3):
        """Сравнение всех ML методов — результат в HTML-таблице."""
        if df is None:
            df = self._current_df
        for name in ['Random Forest', 'LDA', 'SVM (RBF)', 'SVM (Poly)',
                     'Logistic Regression', 'Decision Tree', 'XGBoost']:
            self.train_model(name, df, test_size)

        model_labels = {
            'random_forest': 'Random Forest', 'lda': 'LDA',
            'svm_(rbf)': 'SVM (RBF)', 'svm_(poly)': 'SVM (Poly)',
            'logistic_regression': 'Logistic Regression',
            'decision_tree': 'Decision Tree', 'xgboost': 'XGBoost'
        }

        rows = []
        for key, label in model_labels.items():
            if key in self._analysis_results:
                r = self._analysis_results[key]
                rows.append({
                    'model': label,
                    'accuracy': f"{r.get('accuracy_mean', 0):.3f} ± {r.get('accuracy_std', 0):.3f}",
                    'auc': f"{r.get('auc_mean', 0):.3f} ± {r.get('auc_std', 0):.3f}",
                    'n_repeats': r.get('n_repeats', 0),
                    'acc_raw': r.get('accuracy_mean', 0),
                })

        if rows:
            rows_sorted = sorted(rows, key=lambda x: x['acc_raw'], reverse=True)
            html_rows = ''
            for i, r in enumerate(rows_sorted, 1):
                medal = {1: '🥇', 2: '🥈', 3: '🥉'}.get(i, str(i))
                html_rows += (f'<tr><td>{medal}</td><td>{r["model"]}</td>'
                              f'<td>{r["accuracy"]}</td><td>{r["auc"]}</td>'
                              f'<td>{r["n_repeats"]}</td></tr>\n')
            html_table = (
                f'<p><b>Устойчивость прогноза:</b> 10 повторных делений на train/test с разными seed.</p>'
                f'<table class="stat-table">'
                f'<tr><th>Место</th><th>Модель</th><th>Accuracy (mean ± std)</th>'
                f'<th>AUC (mean ± std)</th><th>Повторений</th></tr>\n{html_rows}</table>'
            )
            self._analysis_results['ml_benchmark'] = {
                'text': f'Моделей: {len(rows)}',
                'html': html_table,
                'table': rows_sorted
            }
        else:
            self._analysis_results['ml_benchmark'] = {'text': '', 'html': ''}

    # ====================== ОТБОР ПРИЗНАКОВ ======================
    def feature_selection_rf(self):
        """Важность признаков — значения выводятся у ВСЕХ баров."""
        multi = self.params.get('multi', [])
        if not multi:
            self._analysis_results['feature_selection_rf'] = {'text': '', 'html': '', 'fig': ''}
            return ''
        X = self._current_df[multi].select_dtypes(include=['number'])
        if X.shape[1] < 2:
            self._analysis_results['feature_selection_rf'] = {'text': '', 'html': '', 'fig': ''}
            return ''
        le = LabelEncoder()
        y = le.fit_transform(self._current_df[self.params['group']])
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(X, y)
        importances = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_})
        importances = importances.sort_values('importance', ascending=True)

        fig_height = max(5, len(importances) * 0.5)
        fig, ax = plt.subplots(figsize=(10, fig_height))
        colors = sns.color_palette('Set2', n_colors=len(importances))
        bars = ax.barh(importances['feature'], importances['importance'], color=colors,
                       edgecolor='black', linewidth=0.8)
        ax.set_title('Важность признаков (Random Forest)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Важность', fontsize=12)

        # Значения у ВСЕХ баров (исправление бага)
        max_width = importances['importance'].max()
        for bar in bars.patches:
            val = bar.get_width()
            x_pos = val + max_width * 0.02
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=9, color='#333')
        ax.set_xlim(0, max_width * 1.25)
        ax.grid(axis='x', alpha=0.25, linestyle='-')
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor('#cccccc')
            spine.set_linewidth(0.8)
        plt.tight_layout()

        fig_b64 = self._fig_to_base64(fig)
        self._analysis_results['feature_selection_rf'] = {
            'text': '', 'html': '', 'fig': fig_b64, 'importances': importances
        }
        return ''

    def rfe_selection(self):
        multi = self.params.get('multi', [])
        X = self._current_df[multi].select_dtypes(include=['number'])
        if X.shape[1] < 2:
            self._analysis_results['rfe_selection'] = {'text': '', 'html': ''}
            return ''
        le = LabelEncoder()
        y = le.fit_transform(self._current_df[self.params['group']])
        n_features = max(1, X.shape[1] // 2)
        dt = DecisionTreeClassifier(random_state=42, class_weight='balanced')
        rfe = RFE(estimator=dt, n_features_to_select=n_features)
        rfe.fit(X, y)
        selected = [f for f, s in zip(X.columns, rfe.support_) if s]

        html_table = (f'<p><b>Отбрано {len(selected)} из {X.shape[1]} признаков:</b></p>'
                      f'<p>{", ".join(selected)}</p>')
        self._analysis_results['rfe_selection'] = {
            'text': f'Отбрано: {len(selected)}', 'html': html_table, 'selected': selected
        }
        return f'Отбрано: {len(selected)}'

    def pca_analysis(self):
        multi = self.params.get('multi', [])
        X = self._current_df[multi].select_dtypes(include=['number'])
        if X.shape[1] < 2:
            self._analysis_results['pca'] = {'text': '', 'html': '', 'fig': ''}
            return ''
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        n_95 = np.argmax(cum_var >= 0.95) + 1
        n_pc = min(5, X.shape[1])
        loadings = pd.DataFrame(pca.components_[:n_pc].T,
                                index=X.columns,
                                columns=[f'PC{i+1}' for i in range(n_pc)])

        fig, axes = plt.subplots(1, 2, figsize=(14, max(5, n_pc * 0.4 + 3)))
        axes[0].bar(range(1, len(cum_var) + 1), pca.explained_variance_ratio_,
                    alpha=0.7, label='Индивидуальная', color='#3498db',
                    edgecolor='black', linewidth=0.8)
        axes[0].plot(range(1, len(cum_var) + 1), cum_var, 'o-', color='#e74c3c',
                     linewidth=2.5, markersize=7, markeredgecolor='black',
                     markeredgewidth=1.0, label='Кумулятивная')
        axes[0].axhline(0.95, color='#27ae60', linestyle='--', linewidth=2, alpha=0.8,
                        label='95%')
        axes[0].set_xlabel('Компонента', fontsize=12)
        axes[0].set_ylabel('Объяснённая дисперсия', fontsize=12)
        axes[0].set_title('Метод главных компонент', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11, framealpha=0.95, edgecolor='gray')
        axes[0].grid(True, alpha=0.25, linestyle='-')
        axes[0].set_axisbelow(True)
        for spine in axes[0].spines.values():
            spine.set_edgecolor('#cccccc')
            spine.set_linewidth(0.8)
        if n_pc >= 2:
            sns.heatmap(loadings, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                        ax=axes[1], cbar_kws={'label': 'Нагрузка'})
            axes[1].set_title(f'Тепловая карта нагрузок (первые {n_pc} ГК)')
        plt.tight_layout()

        fig_b64 = self._fig_to_base64(fig)
        self._analysis_results['pca'] = {
            'text': f'Компонент для 95%: {n_95}',
            'html': '', 'fig': fig_b64, 'pca': pca, 'X_pca': X_pca
        }
        return f'Компонент для 95%: {n_95}'

    # ====================== КЛАСТЕРНЫЙ АНАЛИЗ ======================
    def determine_optimal_clusters(self, max_k=10):
        multi = self.params.get('multi', [])
        X = self._current_df[multi].select_dtypes(include=['number'])
        if X.shape[1] < 2 or len(X) < 5:
            self._analysis_results['elbow'] = {'text': '', 'fig': '', 'optimal_k': 2}
            return ''
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        inertias = []
        K_range = range(1, min(max_k + 1, len(X)))
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        diffs = np.diff(inertias)
        diff_diffs = np.diff(diffs)
        optimal_k = np.argmax(diff_diffs) + 2 if len(diff_diffs) > 0 else 2

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(K_range, inertias, 'bo-', linewidth=2.5, markersize=8, markeredgecolor='black',
                markeredgewidth=1.0)
        ax.axvline(optimal_k, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.8,
                   label=f'Elbow k={optimal_k}')
        ax.set_xlabel('Число кластеров k', fontsize=12)
        ax.set_ylabel('Инерция (WCSS)', fontsize=12)
        ax.set_title('Метод каменистой осыпи (Elbow)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.25, linestyle='-')
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor('#cccccc')
            spine.set_linewidth(0.8)
        plt.tight_layout()

        fig_b64 = self._fig_to_base64(fig)
        self._analysis_results['elbow'] = {
            'text': f'Оптимально k={optimal_k}', 'fig': fig_b64, 'optimal_k': optimal_k
        }
        return f'Оптимально k={optimal_k}'

    def perform_kmeans(self, n_clusters=None):
        multi = self.params.get('multi', [])
        X = self._current_df[multi].select_dtypes(include=['number'])
        if X.shape[1] < 2:
            self._analysis_results['kmeans'] = {'text': '', 'fig': ''}
            return ''
        if n_clusters is None:
            elbow_res = self._analysis_results.get('elbow', {})
            n_clusters = elbow_res.get('optimal_k', 3) if elbow_res else 3
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        self._cluster_labels = labels

        # Сводная таблица
        unique, counts = np.unique(labels, return_counts=True)
        html_rows = ''
        for cl, cnt in zip(unique, counts):
            html_rows += f'<tr><td>{cl}</td><td>{cnt}</td><td>{100*cnt/len(labels):.1f}</td></tr>\n'
        html_table = (f'<table class="stat-table" style="width:50%;">'
                      f'<tr><th>Кластер</th><th>Наблюдений</th><th>Доля, %</th></tr>\n{html_rows}</table>')

        fig, ax = plt.subplots(figsize=(8, 6))
        if X.shape[1] >= 2:
            pca_temp = PCA(n_components=2, random_state=42)
            X_2d = pca_temp.fit_transform(X_scaled)
            palette = sns.color_palette('Set2', n_colors=n_clusters)
            sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='Set2', alpha=0.65, s=25,
                           edgecolors='black', linewidths=0.3)
            centers_2d = pca_temp.transform(kmeans.cluster_centers_)
            ax.scatter(centers_2d[:, 0], centers_2d[:, 1], c='#e74c3c', marker='X', s=250,
                       edgecolors='black', linewidths=1.5, label='Центроиды', zorder=5)
            ax.set_xlabel('PC1', fontsize=12)
            ax.set_ylabel('PC2', fontsize=12)
            plt.colorbar(sc, ax=ax, label='Кластер')
            ax.legend(fontsize=11)
        ax.set_title(f'Кластеризация K-means (k={n_clusters})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.25, linestyle='-')
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor('#cccccc')
            spine.set_linewidth(0.8)
        plt.tight_layout()

        fig_b64 = self._fig_to_base64(fig)
        self._analysis_results['kmeans'] = {
            'text': f'k={n_clusters}', 'html': html_table, 'fig': fig_b64,
            'labels': labels, 'model': kmeans, 'k': n_clusters
        }
        return f'k={n_clusters}'

    @_require_clusters
    def anova_for_clusters(self):
        multi = self.params.get('multi', [])
        num_cols = [c for c in multi if c in self._current_df.columns
                    and pd.api.types.is_numeric_dtype(self._current_df[c])]
        if not num_cols:
            self._analysis_results['anova_clusters'] = {'text': '', 'html': ''}
            return ''
        df_cluster = self._current_df[num_cols].copy()
        df_cluster['cluster'] = self._cluster_labels

        html_rows = ''
        for col in num_cols:
            groups = [g[col].dropna() for _, g in df_cluster.groupby('cluster')]
            if len(groups) >= 2:
                f_stat, p_val = sp_stats.f_oneway(*groups)
                icon = '✅' if p_val < 0.05 else '❌'
                html_rows += (f'<tr><td>{col}</td><td>{f_stat:.3f}</td>'
                              f'<td>{p_val:.4f}</td><td>{icon}</td></tr>\n')
        html_table = (f'<table class="stat-table">'
                      f'<tr><th>Признак</th><th>F</th><th>p-value</th>'
                      f'<th>Различие</th></tr>\n{html_rows}</table>')
        self._analysis_results['anova_clusters'] = {'text': '', 'html': html_table}
        return ''

    @_require_clusters
    def plot_cluster_boxplots(self, return_fig=False, figsize=None, font_scale=1.2):
        """Боксплоты признаков по кластерам – значительно увеличенный размер."""
        multi = self.params.get('multi', [])
        num_cols = [c for c in multi if c in self._current_df.columns
                    and pd.api.types.is_numeric_dtype(self._current_df[c])]
        if not num_cols:
            return None
        n = len(num_cols)
        if figsize is None:
            figsize = (max(6, 4 * n), max(6, 8 * font_scale))
        fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        df_plot = self._current_df[num_cols].copy()
        df_plot['cluster'] = self._cluster_labels.astype(str)
        cluster_order = sorted(df_plot['cluster'].unique(), key=str)
        palette = sns.color_palette('Set2', n_colors=len(cluster_order))
        
        for ax, col in zip(axes, num_cols):
            sns.boxplot(x='cluster', y=col, data=df_plot, ax=ax, palette=palette,
                        order=cluster_order,
                        boxprops=dict(edgecolor='black', linewidth=1.5, facecolor='white',
                                      alpha=0.85),
                        whiskerprops=dict(color='black', linewidth=1.2),
                        capprops=dict(color='black', linewidth=1.5),
                        medianprops=dict(color='#e74c3c', linewidth=2.5),
                        flierprops=dict(marker='o', markerfacecolor='#95a5a6', markersize=5,
                                        alpha=0.6, markeredgecolor='black', markeredgewidth=0.5))
            ax.set_title(col, fontsize=14 * font_scale, fontweight='bold')
            ax.set_xlabel('Кластер', fontsize=12 * font_scale)
            ax.set_ylabel(col, fontsize=12 * font_scale)
            ax.tick_params(axis='both', which='major', labelsize=10 * font_scale)
            ax.grid(axis='y', alpha=0.25, linestyle='-')
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_edgecolor('#cccccc')
                spine.set_linewidth(0.8)
            if df_plot['cluster'].nunique() > 5:
                ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle('Распределение признаков по кластерам', fontsize=18 * font_scale, y=1.02, fontweight='bold')
        plt.tight_layout()
        if return_fig:
            return fig
        plt.show()

    @_require_clusters
    def plot_cluster_cat_frequencies(self, return_fig=False):
        cat_cols = self.params.get('cat_multi', [])
        if not cat_cols:
            g_col = self.params['group']
            cat_cols = [g_col] if g_col in self._current_df.columns else []
        cat_cols = [c for c in cat_cols if c in self._current_df.columns]
        if not cat_cols:
            return None
        df_plot = self._current_df[cat_cols].copy()
        df_plot['cluster'] = self._cluster_labels.astype(str)
        n = len(cat_cols)
        fig, axes = plt.subplots(1, n, figsize=(6*n, 6))
        if n == 1:
            axes = [axes]
        for ax, col in zip(axes, cat_cols):
            ct = pd.crosstab(df_plot['cluster'], df_plot[col])
            ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
            ct_pct.plot(kind='bar', stacked=True, ax=ax, colormap='Set2',
                        edgecolor='black', linewidth=0.8)
            ax.set_title(f'{col} по кластерам (%)', fontsize=13, fontweight='bold')
            ax.set_ylabel('%', fontsize=12)
            ax.set_xlabel('Кластер', fontsize=12)
            ax.legend(fontsize=10, framealpha=0.95, edgecolor='gray')
            ax.grid(axis='y', alpha=0.25, linestyle='-')
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_edgecolor('#cccccc')
                spine.set_linewidth(0.8)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.suptitle('Категориальные признаки по кластерам', fontsize=14, fontweight='bold')
        plt.tight_layout()
        if return_fig:
            return fig
        plt.show()

    @_require_clusters

    def plot_cluster_feature_dynamics(self, return_fig=False, normalize=True, figsize=None, font_scale=1.2):
        """
        Динамика средних значений признаков по кластерам.
        Левый график: каждый кластер – отдельная линия, по оси X – признаки.
        Параметр normalize=True – нормализация min-max (0-100%) каждого признака отдельно.
        Правый график: тепловая карта исходных средних.
        """
        multi = self.params.get('multi', [])
        num_cols = [c for c in multi if c in self._current_df.columns
                    and pd.api.types.is_numeric_dtype(self._current_df[c])]
        if not num_cols:
            return None

        df_plot = self._current_df[num_cols].copy()
        df_plot['cluster'] = self._cluster_labels
        cluster_means = df_plot.groupby('cluster')[num_cols].mean()
        cluster_std = df_plot.groupby('cluster')[num_cols].std()

        if figsize is None:
            figsize = (18, 8 * font_scale)
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # ---- Левый график: динамика кластеров (линии) по всем признакам ----
        n_clusters = len(cluster_means)
        if normalize:
            data_norm = cluster_means.copy()
            std_norm = cluster_std.copy()
            if n_clusters <= 2:
                for col in num_cols:
                    overall_mean = data_norm[col].mean()
                    if overall_mean != 0:
                        scale = 100.0 / overall_mean
                        data_norm[col] = data_norm[col] * scale
                        std_norm[col] = std_norm[col] * scale
                    else:
                        data_norm[col] = 100.0
                        std_norm[col] = 0.0
                data_plot = data_norm.T
                std_plot = std_norm.T
                ylabel = 'Относительное значение (% от среднего)'
            else:
                for col in num_cols:
                    cmin = data_norm[col].min()
                    cmax = data_norm[col].max()
                    if cmax > cmin:
                        scale = 100.0 / (cmax - cmin)
                        data_norm[col] = (data_norm[col] - cmin) * scale
                        std_norm[col] = std_norm[col] * scale
                    else:
                        data_norm[col] = 50.0
                        std_norm[col] = 0.0
                data_plot = data_norm.T
                std_plot = std_norm.T
                ylabel = 'Нормализованное значение (0–100%)'
        else:
            data_plot = cluster_means.T
            std_plot = cluster_std.T
            ylabel = 'Среднее значение'

        n_clusters_plot = len(data_plot.columns)
        colors = sns.color_palette("tab10", n_colors=n_clusters_plot)

        for i, cluster_id in enumerate(data_plot.columns):
            x_idx = range(len(data_plot.index))
            y_vals = data_plot[cluster_id].values
            y_err = std_plot[cluster_id].values if cluster_id in std_plot.columns else None
            axes[0].plot(x_idx, y_vals,
                         marker='o', linewidth=2.5, markersize=8,
                         label=f'Кластер {int(cluster_id)}', color=colors[i % len(colors)],
                         linestyle='-', alpha=0.85)
            if y_err is not None:
                axes[0].fill_between(x_idx, y_vals - y_err, y_vals + y_err,
                                     color=colors[i % len(colors)], alpha=0.12)

        axes[0].set_xlabel('Признак', fontsize=14 * font_scale, fontweight='bold')
        axes[0].set_ylabel(ylabel, fontsize=14 * font_scale, fontweight='bold')
        axes[0].set_title('Профили кластеров по признакам' + (' (нормализовано)' if normalize else ' (исходные средние)'),
                          fontsize=16 * font_scale, fontweight='bold')
        axes[0].grid(True, alpha=0.25, linewidth=1, linestyle='-')
        axes[0].set_xticks(range(len(data_plot.index)))
        axes[0].set_xticklabels(data_plot.index, fontsize=12 * font_scale, rotation=45, ha='right')
        axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11 * font_scale, ncol=1)
        # Устанавливаем нижнюю границу Y (для наглядности)
        if normalize and n_clusters > 2:
            axes[0].set_ylim(bottom=-5, top=105)
        elif normalize and n_clusters <= 2:
            # Для нормализации "% от среднего": добавляем отступ от краёв
            y_min = data_plot.min().min()
            y_max = data_plot.max().max()
            margin = max(5, (y_max - y_min) * 0.1)
            axes[0].set_ylim(bottom=y_min - margin, top=y_max + margin)
        else:
            # Можно оставить автоматический или задать отступ
            pass

        # ---- Правый график: тепловая карта исходных средних ----
        cluster_means_t = cluster_means.T   # строки – признаки, столбцы – кластеры
        im = axes[1].imshow(cluster_means_t.values, aspect='auto', cmap='YlOrRd')
        axes[1].set_xticks(range(len(cluster_means_t.columns)))
        axes[1].set_xticklabels([f'Кластер {int(c)}' for c in cluster_means_t.columns], fontsize=13 * font_scale)
        axes[1].set_yticks(range(len(cluster_means_t.index)))
        axes[1].set_yticklabels(cluster_means_t.index, fontsize=13 * font_scale)
        axes[1].set_title('Исходные средние значения', fontsize=16 * font_scale, fontweight='bold')
        # Подписи значений в ячейках
        for i in range(len(cluster_means_t.index)):
            for j in range(len(cluster_means_t.columns)):
                val = cluster_means_t.values[i, j]
                axes[1].text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=11 * font_scale,
                             color='white' if val > cluster_means_t.values.mean() else 'black',
                             fontweight='bold')
        plt.colorbar(im, ax=axes[1], label='Среднее значение')

        plt.suptitle('Профили кластеров', fontsize=20 * font_scale, y=1.02, fontweight='bold')
        plt.tight_layout()
        if return_fig:
            return fig
        plt.show()
                
    def save_clusters_to_xlsx(self, filename=None, use_original=True):
        """Сохранение кластеров. Исключённые строки получают NaN (пустая ячейка)."""
        if self._cluster_labels is None:
            return
        if filename is None:
            base = Path(self.file_name).stem
            filename = f"{base}_with_clusters.xlsx"
        filename = os.path.join(os.getcwd(), filename)

        if use_original:
            df_out = self.df.copy()
            df_out['cluster'] = np.nan
            analyzed_idx = getattr(self, '_analyzed_indices', None)
            if analyzed_idx is not None and len(analyzed_idx) == len(self._cluster_labels):
                df_out.loc[analyzed_idx, 'cluster'] = self._cluster_labels
            else:
                min_len = min(len(df_out), len(self._cluster_labels))
                df_out.iloc[:min_len, df_out.columns.get_loc('cluster')] = self._cluster_labels[:min_len]
        else:
            df_out = self._current_df.copy()
            df_out['cluster'] = self._cluster_labels

        df_out.to_excel(filename, index=False)
        return filename

    # ====================== ГЕНЕРАЦИЯ HTML-ОТЧЁТА ======================
    def plot_confusion_matrix(self, y_true, y_pred, class_names,
                              title="Матрица ошибок", return_fig=False):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax,
                    linewidths=1.5, linecolor='white',
                    annot_kws={"fontsize": 14, "fontweight": "bold"})
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('Истинные', fontsize=12)
        plt.xlabel('Предсказанные', fontsize=12)
        for spine in ax.spines.values():
            spine.set_edgecolor('#cccccc')
            spine.set_linewidth(0.8)
        plt.tight_layout()
        if return_fig:
            return fig
        plt.show()

    def _comment_block(self, key):
        key_map = {
            'preprocessing': 'preprocessing',
            'plots': 'plots', 'violin': 'plots', 'boxplot': 'plots', 'histograms': 'plots',
            'pie': 'plots', 'scatter': 'plots', 'pairgrid': 'plots', 'corr': 'plots',
            'interaction': 'plots',
            'anova': 'anova', 'tukey': 'anova', 'two_way': 'anova', 'categorical': 'anova',
            'manova': 'manova', 'posthoc_manova': 'manova',
            'linear_regression': 'linear_regression', 'logistic_reg_cat': 'linear_regression',
            'feature_selection': 'feature_selection', 'rfe': 'feature_selection',
            'pca': 'pca',
            'cluster': 'cluster', 'elbow': 'cluster', 'kmeans': 'cluster',
            'anova_clusters': 'cluster',
            'ml': 'ml', 'ml_benchmark': 'ml', 'random_forest': 'ml', 'lda': 'ml',
            'svm_(rbf)': 'ml', 'svm_(poly)': 'ml', 'logistic_regression': 'ml',
            'decision_tree': 'ml', 'xgboost': 'ml',
        }
        mapped = key_map.get(key, key)
        text = self.comments.get(mapped, '').strip()
        if text:
            return (f'<div class="user-comment"><b>Комментарий:</b><br>'
                    f'{text.replace(chr(10), "<br>")}</div>')
        return ''

    def _render_preprocessing_section(self):
        """Блок 'Предобработка': статистика исключений + частоты + сопряжённости + связи."""
        stats = self._preprocessing_stats
        if not stats:
            return ''

        sec = '<h2 id="preprocessing">1. Предобработка данных</h2>\n'
        sub_idx = 1

        # Статистика исключений
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
        sec += f'<h3>1.{sub_idx} Статистика предобработки</h3>\n{html_stats}\n'
        sub_idx += 1

        # Пропуски по столбцам
        missing = stats.get('missing_per_column', {})
        if missing:
            rows = ''
            for col, cnt in missing.items():
                if cnt > 0:
                    rows += f'<tr><td>{col}</td><td>{cnt}</td></tr>\n'
            if rows:
                sec += (f'<h3>1.{sub_idx} Пропуски по столбцам</h3>'
                        f'<table class="stat-table" style="width:50%;">'
                        f'<tr><th>Столбец</th><th>Пропусков</th></tr>{rows}</table>')
                sub_idx += 1

        # Удалённые высококоррелированные
        corr_removals = stats.get('correlation_removals', [])
        if corr_removals:
            rows = ''
            for kept, dropped, r in corr_removals:
                rows += f'<tr><td>{kept}</td><td>{dropped}</td><td>{r:.3f}</td></tr>\n'
            sec += (f'<h3>1.{sub_idx} Удалённые высококоррелированные признаки</h3>'
                    f'<table class="stat-table" style="width:60%;">'
                    f'<tr><th>Оставлен</th><th>Удалён</th><th>|r|</th></tr>{rows}</table>')
            sub_idx += 1

        # Частоты встречаемости (перенос из ANOVA)
        freq_html = self._analysis_results.get('frequency', {}).get('html', '')
        if freq_html:
            sec += f'<h3>1.{sub_idx} Частоты встречаемости категориальных признаков</h3>\n{freq_html}\n'
            sub_idx += 1

        # Анализ связей категориальных признаков (перенос из ANOVA)
        cat_html = self._analysis_results.get('categorical', {}).get('html', '')
        if cat_html:
            sec += f'<h3>1.{sub_idx} Анализ связей категориальных признаков</h3>\n{cat_html}\n'
            sub_idx += 1

        sec += self._comment_block('preprocessing')
        return sec

    def _render_plots_section(self, include):
        img_violin = self._fig_to_base64(self.plot_violin(return_fig=True)) if include('violin') else ''
        img_boxplot = self._fig_to_base64(self.plot_boxplot_with_significance(return_fig=True)) if include('boxplot') else ''
        fig_hist = self.plot_histograms(return_fig=True) if include('histograms') else None
        img_hist = self._fig_to_base64(fig_hist) if fig_hist is not None else ''
        fig_pie = self.plot_pie_chart(return_fig=True) if include('pie') else None
        img_pie = self._fig_to_base64(fig_pie) if fig_pie is not None else ''
        fig_scatter = self.plot_scatter_with_regression(return_fig=True) if include('scatter') else None
        img_scatter = self._fig_to_base64(fig_scatter) if fig_scatter is not None else ''
        fig_pairgrid = self.plot_pairgrid(return_fig=True) if include('pairgrid') else None
        img_pairgrid = self._fig_to_base64(fig_pairgrid) if fig_pairgrid is not None else ''
        img_corr = self._fig_to_base64(self.plot_correlation_matrix(return_fig=True)) if include('corr') else ''
        img_interaction = ''
        if include('interaction'):
            fig_interaction = self.plot_interaction_effect(return_fig=True)
            if fig_interaction is not None:
                img_interaction = self._fig_to_base64(fig_interaction)

        sec = ''
        if img_violin and img_boxplot:
            sec += f'''
            <h2 id="plots">2. Графики</h2>
            <h3>2.1 Скрипичная + Роевая диаграмма</h3>
            <div class="plot-container" onclick="this.classList.toggle('enlarged')"><img src="{img_violin}"></div>
            {self._comment_block('violin')}
            <h3>2.2 Ящики с усами</h3>
            <div class="plot-container" onclick="this.classList.toggle('enlarged')"><img src="{img_boxplot}"></div>
            {self._comment_block('boxplot')}
            '''
        if img_hist:
            sec += f'<h3>2.3 Гистограммы</h3><div class="plot-container" onclick="this.classList.toggle(\'enlarged\')"><img src="{img_hist}"></div>\n'
        if img_pie:
            sec += f'<h3>2.4 Круговая диаграмма</h3><div class="plot-container" onclick="this.classList.toggle(\'enlarged\')"><img src="{img_pie}"></div>\n'
        if img_scatter:
            sec += f'<h3>2.5 Скаттерограмма с регрессией</h3><div class="plot-container" onclick="this.classList.toggle(\'enlarged\')"><img src="{img_scatter}"></div>\n'
        if img_pairgrid:
            sec += f'<h3>2.6 PairGrid</h3><div class="plot-container" onclick="this.classList.toggle(\'enlarged\')"><img src="{img_pairgrid}"></div>\n'
        if img_corr:
            sec += f'<h3>2.7 Корреляционная матрица</h3><div class="plot-container" onclick="this.classList.toggle(\'enlarged\')"><img src="{img_corr}"></div>\n'
        if img_interaction:
            sec += f'<h3>2.8 График взаимодействия</h3><div class="plot-container" onclick="this.classList.toggle(\'enlarged\')"><img src="{img_interaction}"></div>\n'
        return sec

    def _render_anova_section(self, include):
        anova_html = self._analysis_results.get('anova', {}).get('html', '') if include('anova') else ''
        tukey_html = self._analysis_results.get('tukey', {}).get('html', '') if include('tukey') else ''
        two_way_html = self._analysis_results.get('two_way', {}).get('html', '') if include('two_way') else ''
        
        sec = ''
        if anova_html or tukey_html or two_way_html:
            sec += '<h2 id="anova">3. Дисперсионный анализ</h2>\n'
        if anova_html:
            sec += f'<h3>3.1 One-Way ANOVA / Kruskal-Wallis</h3>\n{anova_html}\n{self._comment_block("anova")}'
        if tukey_html:
            sec += f'<h3>3.2 Пост-хок тест для уточняющих сравнений (post-hoc)</h3>\n{tukey_html}\n'
        if two_way_html:
            sec += f'<h3>3.3 Two-way ANOVA</h3>\n{two_way_html}\n'
        return sec

    def _render_manova_section(self, include):
        manova_html = self._analysis_results.get('manova', {}).get('html', '') if include('manova') else ''
        manova_conclusion = self._analysis_results.get('manova', {}).get('conclusion', '') if include('manova') else ''
        manova_desc = self._analysis_results.get('manova', {}).get('descriptions', '') if include('manova') else ''
        ph_html = self._analysis_results.get('posthoc_manova', {}).get('html', '') if include('posthoc_manova') else ''

        sec = ''
        if manova_html or ph_html:
            sec += '<h2 id="manova">3. Многомерный анализ</h2>\n'
        if manova_html:
            sec += f'<h3>3.1 MANOVA</h3>\n{manova_html}\n{manova_conclusion}\n{manova_desc}\n{self._comment_block("manova")}'
        if ph_html:
            sec += f'<h3>3.2 Post-hoc MANOVA (Tukey HSD)</h3>\n{ph_html}\n'
        return sec

    def _render_regression_section(self, include):
        lr_html = self._analysis_results.get('linear_regression', {}).get('html', '') if include('linear_regression') else ''
        lr_diag_fig = ''
        if include('linear_regression'):
            fig_lr = self.plot_regression_diagnostics(return_fig=True)
            if fig_lr is not None:
                lr_diag_fig = self._fig_to_base64(fig_lr)
        logreg_html = self._analysis_results.get('logistic_reg_cat', {}).get('html', '')
        sec = ''
        if lr_html or logreg_html:
            sec += '<h2 id="regression">4. Регрессионный анализ</h2>\n'
        if lr_html:
            sec += f'<h3>4.1 Линейная регрессия</h3>\n{lr_html}\n{self._comment_block("linear_regression")}'
            if lr_diag_fig:
                diag_interp = (
                    '<div class="interp-note" style="background:#eef6ff; border-left:4px solid #3498db; '
                    'padding:12px 16px; margin:15px 0; border-radius:0 6px 6px 0; font-size:0.95em;">'
                    '<b>Интерпретация диагностики:</b> '
                    '<ul style="margin:5px 0;">'
                    '<li><b>Остатки vs Предсказанные</b> — точки должны быть равномерно распределены '
                    'вокруг нуля без видимого паттерна. Систематические отклонения (веер, парабола) '
                    'указывают на нелинейность или гетероскедастичность.</li>'
                    '<li><b>Q-Q plot</b> — точки должны лежать вдоль диагональной линии. '
                    'Отклонения вверх/вниз от линии на концах графика означают: '
                    'точки выше линии справа и ниже слева — тяжёлые хвосты (heavy tails, '
                    'больше экстремальных значений, чем в нормальном распределении); '
                    'точки ниже линии справа и выше слева — лёгкие хвосты (light tails, '
                    'меньше экстремальных значений). Изломы в середине — асимметрия.</li>'
                    '</ul></div>'
                )
                sec += f'<h3>4.1.b Диагностика</h3><div class="plot-container" onclick="this.classList.toggle(\'enlarged\')"><img src="{lr_diag_fig}"></div>{diag_interp}'
        if logreg_html:
            sec += f'<h3>4.2 Логистическая регрессия (категориальные предикторы)</h3>\n{logreg_html}\n'
        return sec

    def _render_feature_section(self, include):
        fs_rf_fig = self._analysis_results.get('feature_selection_rf', {}).get('fig', '') if include('feature_selection') else ''
        rfe_html = self._analysis_results.get('rfe_selection', {}).get('html', '') if include('feature_selection') else ''
        sec = ''
        if fs_rf_fig or rfe_html:
            sec += '<h2 id="feature_selection">5. Отбор признаков</h2>\n'
        if fs_rf_fig:
            sec += f'<h3>5.1 Важность признаков (Random Forest)</h3><div class="plot-container" onclick="this.classList.toggle(\'enlarged\')"><img src="{fs_rf_fig}"></div>'
        if rfe_html:
            rfe_interp = (
                '<div class="interp-note" style="background:#eef6ff; border-left:4px solid #3498db; '
                'padding:12px 16px; margin:15px 0; border-radius:0 6px 6px 0; font-size:0.95em;">'
                '<b>Интерпретация:</b> '
                'RFE (Recursive Feature Elimination) последовательно удаляет наименее важные '
                'признаки, обучая дерево решений на каждом шаге. Оставшиеся признаки — наиболее '
                'информативные для классификации. Список может меняться при разных данных, '
                'поэтому рекомендуется перезапуск для проверки устойчивости отбора. '
                'Для финальной модели используйте только отобранные признаки.'
                '</div>'
            )
            sec += f'<h3>5.2 Рекурсивное устранение (RFE)</h3>\n{rfe_html}\n{rfe_interp}'
        return sec

    def _render_pca_section(self, include):
        pca_text = self._analysis_results.get('pca', {}).get('text', '') if include('pca') else ''
        pca_fig = self._analysis_results.get('pca', {}).get('fig', '') if include('pca') else ''
        sec = ''
        if pca_text or pca_fig:
            pca_interp = (
                '<div class="interp-note" style="background:#eef6ff; border-left:4px solid #3498db; '
                'padding:12px 16px; margin:15px 0; border-radius:0 6px 6px 0; font-size:0.95em;">'
                '<b>Интерпретация:</b> '
                'PCA снижает размерность данных, создавая новые некоррелированные переменные '
                '(компоненты), каждая из которых объясняет максимальную оставшуюся дисперсию. '
                'Левый график: столбцы — доля дисперсии каждой компоненты, линия — кумулятивная. '
                'Компоненты за зелёной пунктирной линией (95%) можно отбросить без существенной '
                'потери информации. Тепловая карта нагруженных показывает, какие исходные '
                'признаки больше всего вкладывается в каждую компоненту (|нагрузка| > 0.3 — '
                'существенный вклад).'
                '</div>'
            )
            sec += f'<h2 id="pca">6. Метод главных компонент (PCA)</h2>\n<p>{pca_text}</p>\n'
            if pca_fig:
                sec += f'<div class="plot-container" onclick="this.classList.toggle(\'enlarged\')"><img src="{pca_fig}"></div>'
            sec += pca_interp
        return sec

    def _render_cluster_section(self, include):
        if not include('cluster'):
            return ''
        elbow_text = self._analysis_results.get('elbow', {}).get('text', '')
        elbow_fig = self._analysis_results.get('elbow', {}).get('fig', '')
        kmeans_text = self._analysis_results.get('kmeans', {}).get('text', '')
        kmeans_html = self._analysis_results.get('kmeans', {}).get('html', '')
        kmeans_fig = self._analysis_results.get('kmeans', {}).get('fig', '')
        anova_clust_html = self._analysis_results.get('anova_clusters', {}).get('html', '')

        sec = ''
        if elbow_text or kmeans_text or self._cluster_labels is not None:
            sec += '<h2 id="cluster">7. Кластерный анализ</h2>\n'
        if elbow_text:
            sec += f'<h3>7.1 Оптимальное число кластеров</h3>\n<p>{elbow_text}</p>\n'
            if elbow_fig:
                sec += f'<div class="plot-container" onclick="this.classList.toggle(\'enlarged\')"><img src="{elbow_fig}"></div>'
        if kmeans_text:
            sec += f'<h3>7.2 K-means</h3>\n<p>{kmeans_text}</p>\n{kmeans_html}\n'
            if kmeans_fig:
                sec += f'<div class="plot-container" onclick="this.classList.toggle(\'enlarged\')"><img src="{kmeans_fig}"></div>'
        if anova_clust_html:
            sec += f'<h3>7.3 ANOVA для кластеров</h3>\n{anova_clust_html}\n'

        if self._cluster_labels is not None:
            fig_cb = self.plot_cluster_boxplots(return_fig=True)
            if fig_cb is not None:
                img_cb = self._fig_to_base64(fig_cb)
                sec += f'<h3>7.4 Boxplot признаков по кластерам</h3><div class="plot-container" onclick="this.classList.toggle(\'enlarged\')"><img src="{img_cb}"></div>'
            fig_cd = self.plot_cluster_feature_dynamics(return_fig=True)
            if fig_cd is not None:
                img_cd = self._fig_to_base64(fig_cd)
                sec += f'<h3>7.5 Динамика признаков по кластерам</h3><div class="plot-container" onclick="this.classList.toggle(\'enlarged\')"><img src="{img_cd}"></div>'
            fig_cf = self.plot_cluster_cat_frequencies(return_fig=True)
            if fig_cf is not None:
                img_cf = self._fig_to_base64(fig_cf)
                sec += f'<h3>7.6 Категориальные признаки по кластерам</h3><div class="plot-container" onclick="this.classList.toggle(\'enlarged\')"><img src="{img_cf}"></div>'
        return sec

    def _render_ml_section(self, include):
        if not (include('ml') or include('ml_benchmark')):
            return ''
        ml_bench_html = self._analysis_results.get('ml_benchmark', {}).get('html', '')

        model_labels = {
            'random_forest': 'Random Forest', 'lda': 'LDA', 'svm_(rbf)': 'SVM (RBF)',
            'svm_(poly)': 'SVM (Poly)', 'logistic_regression': 'Logistic Regression',
            'decision_tree': 'Decision Tree', 'xgboost': 'XGBoost'
        }
        best_key = None
        best_acc = -1
        for mk in model_labels.keys():
            if mk in self._analysis_results and self._analysis_results[mk].get('accuracy_mean', 0) > best_acc:
                best_acc = self._analysis_results[mk]['accuracy_mean']
                best_key = mk

        sec = ''
        if ml_bench_html or best_key:
            sec += '<h2 id="ml">8. Машинное обучение</h2>\n'
        if ml_bench_html:
            sec += f'<h3>8.1 Сравнение методов (10 повторений)</h3>\n{ml_bench_html}\n{self._comment_block("ml")}'

        if include('ml') and best_key and best_key in self._analysis_results:
            br = self._analysis_results[best_key]
            if 'y_test' in br and br.get('y_test') is not None:
                img_cm = self._fig_to_base64(
                    self.plot_confusion_matrix(br['y_test'], br['y_pred'], br['class_names'],
                                               title=f'Матрица ошибок: {model_labels.get(best_key, best_key)}',
                                               return_fig=True))
                sec += f'<h3>8.2 Матрица ошибок ({model_labels.get(best_key, best_key)})</h3><div class="plot-container" onclick="this.classList.toggle(\'enlarged\')"><img src="{img_cm}"></div>'
            if 'y_proba' in br and br['y_proba'] is not None:
                fig_a, ax_a = plt.subplots(figsize=(8, 6))
                yt = br['y_test']
                yp = br['y_proba']
                cn = br['class_names']
                nc = len(cn)
                palette_roc = sns.color_palette('Set2', n_colors=nc)
                if nc == 2:
                    fpr, tpr, _ = roc_curve(yt, yp[:, 1])
                    ax_a.plot(fpr, tpr, label=f'AUC = {br["auc_mean"]:.3f}',
                              linewidth=2.5, color=palette_roc[0])
                else:
                    for i in range(nc):
                        fpr, tpr, _ = roc_curve((yt == i).astype(int), yp[:, i])
                        auc_i = roc_auc_score((yt == i).astype(int), yp[:, i])
                        ax_a.plot(fpr, tpr, label=f'{cn[i]} (AUC = {auc_i:.3f})',
                                  linewidth=2.5, color=palette_roc[i])
                ax_a.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1.5)
                ax_a.set_xlabel('FPR', fontsize=12)
                ax_a.set_ylabel('TPR', fontsize=12)
                ax_a.set_title(f'ROC-кривая: {model_labels.get(best_key, best_key)}',
                               fontsize=14, fontweight='bold')
                ax_a.legend(fontsize=11, framealpha=0.95, edgecolor='gray')
                ax_a.grid(True, alpha=0.25, linestyle='-')
                ax_a.set_axisbelow(True)
                for spine in ax_a.spines.values():
                    spine.set_edgecolor('#cccccc')
                    spine.set_linewidth(0.8)
                plt.tight_layout()
                img_auc = self._fig_to_base64(fig_a)
                # Примечание об интерпретации ROC
                roc_interp = (
                    '<div class="interp-note" style="background:#eef6ff; border-left:4px solid #3498db; '
                    'padding:12px 16px; margin:15px 0; border-radius:0 6px 6px 0; font-size:0.95em;">'
                    '<b>Интерпретация:</b> '
                    'ROC-кривая показывает соотношение истинно положительных (TPR) и ложно '
                    'положительных (FPR) результатов при различных порогах классификации. '
                    'AUC (площадь под кривой): 1.0 — идеальная модель, 0.9–1.0 — отличная, '
                    '0.8–0.9 — хорошая, 0.7–0.8 — удовлетворительная, 0.5–0.7 — слабая, '
                    '0.5 — случайное угадывание. Диагональная линия (AUC = 0.5) — случайный '
                    'классификатор. Чем выше кривая — тем лучше модель различает классы.'
                    '</div>'
                )
                sec += f'<h3>8.3 ROC-кривая</h3><div class="plot-container" onclick="this.classList.toggle(\'enlarged\')"><img src="{img_auc}"></div>{roc_interp}'
        return sec

    def generate_html_report(self, df_clean, sections=None):
        """Генерация HTML-отчёта со всеми улучшениями."""
        self._current_df = df_clean
        if sections is None:
            sections = {}

        def _include(key):
            group_map = {
                'preprocessing': ['preprocessing'],
                'plots': ['violin', 'boxplot', 'histograms', 'pie', 'scatter', 'pairgrid', 'corr', 'interaction'],
                'anova': ['anova', 'tukey', 'two_way', 'categorical'],
                'manova': ['manova', 'posthoc_manova'],
                'linear_regression': ['linear_regression'],
                'feature_selection': ['feature_selection'],
                'pca': ['pca'],
                'cluster': ['cluster'],
                'ml': ['ml', 'ml_benchmark'],
            }
            if key in group_map:
                return sections.get(key, True)
            for grp_key, members in group_map.items():
                if key in members:
                    return sections.get(grp_key, True)
            return sections.get(key, True)

        base_name = Path(self.file_name).stem
        output_filename = f"{base_name}_report.html"

        sec_preprocessing = self._render_preprocessing_section()
        sec_plots = self._render_plots_section(_include)
        sec_anova = self._render_anova_section(_include)
        sec_manova = self._render_manova_section(_include)
        sec_regression = self._render_regression_section(_include)
        sec_feature = self._render_feature_section(_include)
        sec_pca = self._render_pca_section(_include)
        sec_cluster = self._render_cluster_section(_include)
        sec_ml = self._render_ml_section(_include)

        toc_items = []
        if sec_preprocessing:
            toc_items.append('<li><a href="#preprocessing">Предобработка</a></li>')
        if sec_plots:
            toc_items.append('<li><a href="#plots">Графики</a></li>')
        if sec_anova:
            toc_items.append('<li><a href="#anova">Дисперсионный анализ</a></li>')
        if sec_manova:
            toc_items.append('<li><a href="#manova">Многомерный анализ</a></li>')
        if sec_regression:
            toc_items.append('<li><a href="#regression">Регрессионный анализ</a></li>')
        if sec_feature:
            toc_items.append('<li><a href="#feature_selection">Отбор признаков</a></li>')
        if sec_pca:
            toc_items.append('<li><a href="#pca">PCA</a></li>')
        if sec_cluster:
            toc_items.append('<li><a href="#cluster">Кластерный анализ</a></li>')
        if sec_ml:
            toc_items.append('<li><a href="#ml">Машинное обучение</a></li>')

        toc = ''
        if toc_items:
            toc = f'<div class="toc"><b>Содержание:</b><ol>{"".join(toc_items)}</ol></div>'

        html_content = f'''
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="utf-8">
            <title>Отчёт анализа: {base_name}</title>
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; line-height: 1.6; color: #333; max-width: 1200px; margin-left: auto; margin-right: auto; }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #2980b9; margin-top: 40px; border-left: 4px solid #3498db; padding-left: 10px; }}
                h3 {{ color: #34495e; }}
                .plot-container {{ text-align: center; margin: 30px 0; background: #fafafa; padding: 20px; border-radius: 8px; border: 1px solid #eee; cursor: pointer; transition: all 0.3s ease; }}
                .plot-container:hover {{ box-shadow: 0 4px 12px rgba(0,0,0,0.15); }}
                .plot-container img {{ max-width: 100%; height: auto; border-radius: 4px; transition: all 0.3s ease; }}
                .plot-container.enlarged {{ position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: 9999;
                    background: rgba(0,0,0,0.92); padding: 30px; display: flex; align-items: center; justify-content: center;
                    margin: 0; border-radius: 0; border: none; cursor: zoom-out; }}
                .plot-container.enlarged img {{ max-width: 98vw; max-height: 95vh; object-fit: contain; border-radius: 6px;
                    box-shadow: 0 0 40px rgba(255,255,255,0.1); }}
                pre {{ background: #f8f9fa; padding: 20px; border-radius: 6px; overflow-x: auto; font-size: 0.9em; border: 1px solid #e9ecef; white-space: pre-wrap; }}
                .meta {{ color: #666; font-size: 0.9em; margin-bottom: 30px; }}
                .toc {{ background: #f0f7ff; padding: 15px 25px; border-radius: 8px; margin-bottom: 30px; border: 1px solid #d0e3f7; }}
                .toc ol {{ margin: 5px 0; }}
                .toc a {{ color: #2980b9; text-decoration: none; }}
                .toc a:hover {{ text-decoration: underline; }}
                .user-comment {{ background: #fffde7; border-left: 4px solid #fbc02d; padding: 12px 16px; margin: 15px 0; border-radius: 0 6px 6px 0; font-size: 0.95em; }}
                {STAT_TABLE_CSS}
            </style>
        </head>
        <body>
            <h1>Отчёт анализа данных</h1>
            <div class="meta">
                <p><b>Файл:</b> {self.file_name} | <b>Дата:</b> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
                <p><b>Параметры:</b> Группировка: <i>{self.params.get("group","")}</i> | Целевая: <i>{self.params.get("analysis","")}</i> | Многомерные: <i>{", ".join(self.params.get("multi",[]))}</i></p>
            </div>
            {toc}
            {sec_preprocessing}
            {sec_plots}
            {sec_anova}
            {sec_manova}
            {sec_regression}
            {sec_feature}
            {sec_pca}
            {sec_cluster}
            {sec_ml}
            <hr style="margin-top: 50px; border: 0; border-top: 1px solid #eee;">
            <p style="color: #999; font-size: 0.8em; text-align: center;">Сгенерировано Python Data Analyzer v1.0</p>
        </body>
        </html>
        '''
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        return output_filename

# ====================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ======================
def create_interactive_file_uploader():
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    uploader = widgets.FileUpload(
        accept='.xlsx, .xls, .csv', multiple=False,
        description='Select File', button_style='primary',
        layout=widgets.Layout(width='250px'))
    output = widgets.Output()
    _uploaded_df = None
    _file_name = None

    def on_upload(change):
        nonlocal _uploaded_df, _file_name
        with output:
            clear_output()
            if not uploader.value:
                return
            try:
                if isinstance(uploader.value, dict):
                    file_name = list(uploader.value.keys())[0]
                    file_info = uploader.value[file_name]
                elif isinstance(uploader.value, (tuple, list)) and len(uploader.value) > 0:
                    file_info = uploader.value[0]
                    file_name = file_info.get('name', 'uploaded_file.xlsx')
                else:
                    return
                file_content = file_info['content']
                if file_name.lower().endswith('.csv'):
                    _uploaded_df = pd.read_csv(io.BytesIO(file_content))
                else:
                    _uploaded_df = pd.read_excel(io.BytesIO(file_content))
                _file_name = file_name
                print(f'✓ Файл загружен: {file_name} ({len(_uploaded_df):,} строк)')
            except Exception as e:
                print(f'Ошибка: {e}')

    uploader.observe(on_upload, names='value')
    display(widgets.VBox([widgets.HTML('<b>Step 1: Upload Data</b>'), uploader, output]))
    with output:
        print("DataAn Enhanced v1.0 загружен. Загрузите файл выше, затем запустите ячейку 2.")
    return lambda: _uploaded_df, lambda: _file_name
