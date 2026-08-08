# -*- coding: utf-8 -*-
"""
analyzer_enhanced.py — Вычислительное ядро статистического анализа v2.1
Полная версия с HTML-отображением результатов.
"""
import logging
import json
import warnings
import re
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from itertools import combinations
from scipy import stats as sp_stats
from scipy.stats import chi2_contingency, fisher_exact, shapiro, levene, kruskal
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, r2_score, mean_squared_error)
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.multivariate.manova import MANOVA
import statsmodels.api as sm

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

logger = logging.getLogger('DataAn.Core')

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='covariance of constraints')
warnings.filterwarnings('ignore', message='scipy.stats.shapiro')


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        return super().default(obj)

def decode_bdata(obj):
    import base64 as _b64
    if isinstance(obj, dict):
        if 'bdata' in obj and 'dtype' in obj:
            try:
                decoded = np.frombuffer(_b64.b64decode(obj['bdata']), dtype=obj.get('dtype', 'f8'))
                return decoded.tolist()
            except Exception:
                return obj
        return {k: decode_bdata(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decode_bdata(item) for item in obj]
    return obj


def fig_to_json(fig):
    d = decode_bdata(fig.to_dict())
    raw = json.dumps(d, cls=NumpyEncoder, ensure_ascii=False)
    return raw.replace('</script>', '<\\/script>')

STAT_TABLE_CSS = '''
.stat-table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 0.95em; }
.stat-table th { background: #3498db; color: white; padding: 10px 12px; border: 1px solid #2980b9; text-align: left; }
.stat-table td { padding: 8px 12px; border: 1px solid #d0d7de; }
.stat-table tr:nth-child(even) { background: #f8f9fa; }
.stat-table tr:hover { background: #eaf4fc; }
'''

class DataAnalyzer:
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

    def __init__(self, df: pd.DataFrame, file_name: str = ""):
        self.df = df.copy()
        self.file_name = file_name or "Uploaded_Data"
        self._last_file_path = file_name or ''
        self.numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(exclude=['number']).columns.tolist()
        self.params: Dict[str, Any] = {}
        self.comments: Dict[str, str] = {}
        self._analysis_results: Dict[str, Any] = {}
        self._preprocessing_stats: Dict[str, Any] = {}
        self.correlation_removals: List = []
        self._current_df: Optional[pd.DataFrame] = None
        self._cluster_labels: Optional[np.ndarray] = None
        self._excluded_indices = pd.Index([])
        self._analyzed_indices = pd.Index([])
        self._config = self._default_config.copy()
        logger.info(f"Инициализация анализатора для файла: {file_name}")

    def set_data(self, df: pd.DataFrame) -> None:
        self.df = df.copy()
        self.numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(exclude=['number']).columns.tolist()

    @staticmethod
    def load_session() -> Dict[str, Any]:
        path = os.path.join(os.getcwd(), 'analyzer_session.json')
        if not os.path.exists(path):
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def save_session(self, file_path: str = "") -> None:
        data = {
            'last_file_path': file_path or getattr(self, '_last_file_path', ''),
            'last_file_name': self.file_name,
            'last_params': self.params.copy(),
        }
        try:
            with open(os.path.join(os.getcwd(), 'analyzer_session.json'), 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения сессии: {e}")

    def _fig_to_json(self, fig):
        return fig_to_json(fig)

    # ====================== ВСПОМОГАТЕЛЬНЫЕ ======================
    def _fmt(self, value):
        return f'{value:.{self._config["precision"]}f}'

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
        g_col = self.params.get('group', '')
        for c in self.params.get('cat_multi', []):
            if c in self._current_df.columns and c != g_col:
                return c
        for c in self.params.get('multi', []):
            if (c in self._current_df.columns and c != g_col
                    and self._current_df[c].dtype in ['object', 'category']):
                return c
        return None

    def _validate_params(self):
        if not self.params:
            return
        df = self.df
        g = self.params.get('group')
        if g and g in df.columns and pd.api.types.is_numeric_dtype(df[g]):
            logger.warning(f"'{g}' числовая, используется как группирующая.")
        a = self.params.get('analysis')
        if a and a in df.columns and not pd.api.types.is_numeric_dtype(df[a]):
            try:
                df[a] = pd.to_numeric(df[a], errors='coerce')
            except Exception:
                pass
        multi = self.params.get('multi', [])
        self.params['multi'] = [c for c in multi if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        cat_multi = self.params.get('cat_multi', [])
        self.params['cat_multi'] = [c for c in cat_multi if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])]

    # ====================== ТЕСТЫ ДАННА И ПОПРАВКИ ======================
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

    # ====================== ПРЕДОБРАБОТКА ======================
    def preprocess(self, remove_outliers=True, z_threshold=3.0, balance_groups=True):
        if not self.params:
            raise ValueError("Сначала выберите параметры!")
        self._validate_params()
        corr_threshold = self._config['correlation_threshold']

        group_col = self.params.get('group', '')
        analysis_col = self.params.get('analysis', '')
        multi_cat = [c for c in self.params.get('cat_multi', []) if c in self.df.columns]
        cols = list(dict.fromkeys(
            [group_col, analysis_col] + self.params.get('multi', []) + multi_cat))

        all_indices = self.df.index.copy()
        group_missing_mask = self.df[group_col].isna()
        idx_without_group = all_indices[group_missing_mask]
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

        df_work, removed = self._remove_highly_correlated(df_work, threshold=corr_threshold)
        self.correlation_removals = removed
        self._preprocessing_stats['correlation_removals'] = removed
        self._preprocessing_stats['correlation_threshold'] = corr_threshold

        cols_for_corr = [c for c in self.params.get('multi', []) if c in df_work.columns
                         and pd.api.types.is_numeric_dtype(df_work[c])]
        if len(cols_for_corr) >= 2:
            corr_mat = df_work[cols_for_corr].corr().abs()
            upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
            corr_pairs = []
            for col in upper.columns:
                for idx in upper.index:
                    val = upper.loc[idx, col]
                    if pd.notna(val) and val >= corr_threshold:
                        corr_pairs.append((idx, col, float(val)))
            corr_pairs.sort(key=lambda x: -x[2])
            self._preprocessing_stats['corr_pairs'] = corr_pairs

        self.params['multi'] = [c for c in self.params.get('multi', []) if c in df_work.columns]
        self.params['cat_multi'] = [c for c in self.params.get('cat_multi', []) if c in df_work.columns]

        if balance_groups:
            sizes = df_work[group_col].value_counts()
            min_size = self._config['bootstrap_min_size']
            max_ratio = self._config['bootstrap_max_ratio']
            if any(s < min_size for s in sizes) or (sizes.min() > 0 and sizes.max() / sizes.min() > max_ratio):
                df_work = self._bootstrap_balance_groups(df_work, group_col)

        self._preprocessing_stats['final_analyzed'] = len(df_work)
        self._preprocessing_stats['total_excluded'] = len(self._excluded_indices)
        self._current_df = df_work
        self.df = df_work.copy()
        return df_work

    # ====================== КАЧЕСТВО ДАННЫХ ======================
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

    # ====================== СТАТИСТИЧЕСКИЙ АНАЛИЗ ======================
    def perform_anova_analysis(self):
        g_col = self.params['group']
        a_col = self.params['analysis']
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
            p = anova_table['PR(>F)'].iloc[0]
            f_val = anova_table['F'].iloc[0]
            result['is_significant'] = p < 0.05

            html_table = f'''
            <div>
                <div style="background:#e8f8e8; border-left:4px solid #27ae60; padding:10px 16px;
                    margin-bottom:15px; border-radius:0 6px 6px 0; font-size:0.95em;">
                    <b>Выбран параметрический тест</b> (ANOVA), т.к. данные распределены нормально
                    (Shapiro-Wilk p={shapiro_p[0]:.4f} > 0.05) и дисперсии гомогенны (Levene p={levene_p:.4f} > 0.05).
                </div>
                <h3 style="text-align:center; font-size:22px;">One-Way ANOVA: {a_col} по {g_col}</h3>
                <table class="stat-table">
                    <tr><th>Показатель</th><th>Значение</th></tr>
                    <tr><td style="padding:15px; font-weight:bold;">F-статистика</td>
                        <td style="padding:15px; font-size:20px; color:#2c3e50;">{f_val:.3f}</td></tr>
                    <tr><td style="padding:15px; font-weight:bold;">p-value</td>
                        <td style="padding:15px; font-size:20px; color:#2c3e50;">{p:.4f}</td></tr>
                    <tr><td style="padding:15px; font-weight:bold;">η² (эта-квадрат)</td>
                        <td style="padding:15px; font-size:20px; color:#2c3e50;">{eta_sq:.3f}</td></tr>
                    <tr><td style="padding:15px; font-weight:bold;">Результат</td>
                        <td style="padding:15px; font-size:20px; font-weight:bold;
                                   color:{'#27ae60' if p < 0.05 else '#c0392b'};">
                            {'✓ СТАТИСТИЧЕСКИ ЗНАЧИМО' if p < 0.05 else '✗ НЕ ЗНАЧИМО'}</td></tr>
                </table>
            </div>'''
            result['html'] = html_table
            res_text += f"One-Way ANOVA: F={f_val:.3f}, p={p:.4f}, η²={eta_sq:.3f}\n"
            res_text += f"Интерпретация: {'статистически значимо' if p < 0.05 else 'не значимо'} (alpha = 0.05).\n"
        else:
            h_stat, p_val = kruskal(*groups)
            result['method'] = 'Kruskal-Wallis'
            result['h_statistic'] = h_stat
            result['kruskal_pvalue'] = p_val
            result['is_significant'] = p_val < 0.05

            html_table = f'''
            <div>
                <div style="background:#fff3e0; border-left:4px solid #f39c12; padding:10px 16px;
                    margin-bottom:15px; border-radius:0 6px 6px 0; font-size:0.95em;">
                    <b>Выбран непараметрический тест</b> (Kruskal-Wallis), т.к. данные распределены
                    ненормально (Shapiro-Wilk p &lt; 0.05) и/или дисперсии не гомогенны (Levene p &lt; 0.05).
                </div>
                <h3>Kruskal-Wallis Test: {a_col} по {g_col}</h3>
                <table class="stat-table">
                    <tr><th style="font-size:18px; padding:15px;">Показатель</th>
                        <th style="font-size:18px; padding:15px;">Значение</th></tr>
                    <tr><td style="padding:15px; font-weight:bold;">H-статистика</td>
                        <td style="padding:15px; font-size:20px; color:#2c3e50;">{h_stat:.3f}</td></tr>
                    <tr><td style="padding:15px; font-weight:bold;">p-value</td>
                        <td style="padding:15px; font-size:20px; color:#2c3e50;">{p_val:.4f}</td></tr>
                    <tr><td style="padding:15px; font-weight:bold;">Результат</td>
                        <td style="padding:15px; font-size:20px; font-weight:bold;
                                   color:{'#27ae60' if p_val < 0.05 else '#c0392b'};">
                            {'✓ СТАТИСТИЧЕСКИ ЗНАЧИМО' if p_val < 0.05 else '✗ НЕ ЗНАЧИМО'}</td></tr>
                </table>
            </div>'''
            result['html'] = html_table
            res_text += f"Kruskal-Wallis: H={h_stat:.3f}, p={p_val:.4f}\n"
            res_text += f"Интерпретация: {'статистически значимо' if p_val < 0.05 else 'не значимо'} (alpha = 0.05).\n"

        interp_note = (
            '<div style="background:#eef6ff; border-left:4px solid #3498db; '
            'padding:12px 16px; margin:15px 0; border-radius:0 6px 6px 0; font-size:0.95em;">'
            '<b>Интерпретация:</b> '
        )
        if result.get('method') == 'ANOVA':
            interp_note += (
                'p-value ниже 0.05 означает, что хотя бы одна группа значимо отличается от других. '
                'η² показывает долю дисперсии: до 0.01 — малый, 0.01–0.06 — средний, '
                '0.06–0.14 — большой, свыше 0.14 — очень большой. '
                'Для конкретных различий используйте пост-хок анализ.'
            )
        else:
            interp_note += (
                'p-value ниже 0.05 означает, что распределения хотя бы одной группы '
                'статистически значимо различаются. Для конкретных различий '
                'выполнен пост-хок анализ Данна с поправкой Холма.'
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

        g_col = self.params['group']
        a_col = self.params['analysis']
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
            self._analysis_results['dunn_details'] = {
                'pairs': pairs, 'corrected_p': corrected_p, 'correction': 'Holm'
            }

        if not significant_rows:
            html_table = f'<p style="font-size:16px; color:#7f8c8d; font-style:italic;">Достоверных попарных различий не обнаружено (α = 0.05, {method_label}).</p>'
        else:
            html_rows = ''
            for r in significant_rows:
                val_str = f'{r["md"]:.3f}' if method == 'ANOVA' else f'{r["z"]:.3f}'
                html_rows += (f'<tr><td style="padding:12px;">{r["g1"]}</td>'
                              f'<td style="padding:12px;">{r["g2"]}</td>'
                              f'<td style="padding:12px; font-size:16px;">{val_str}</td>'
                              f'<td style="padding:12px; font-size:16px;">{r["p"]:.4f}</td>'
                              f'<td style="padding:12px; font-size:18px;">✅</td></tr>\n')
            html_table = (f'<p style="font-size:16px; font-weight:bold; color:#27ae60;">'
                          f'Найдено значимых попарных различий: {len(significant_rows)} '
                          f'({method_label})</p>'
                          f'<table class="stat-table" style="font-size:16px;">'
                          f'<tr><th style="padding:12px; font-size:16px;">Группа 1</th>'
                          f'<th style="padding:12px; font-size:16px;">Группа 2</th>'
                          f'<th style="padding:12px; font-size:16px;">{value_label}</th>'
                          f'<th style="padding:12px; font-size:16px;">p-скорр.</th>'
                          f'<th style="padding:12px; font-size:16px;">Значимость</th></tr>\n{html_rows}</table>')

        self._analysis_results['tukey'] = {
            'text': f'Значимых пар: {len(significant_rows)}',
            'html': html_table, 'performed': True,
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

            interaction_name = [n for n in at.index if '×' in n]
            interaction_sig = False
            for iname in interaction_name:
                if at.loc[iname, 'PR(>F)'] < 0.05:
                    interaction_sig = True
                    break

            simple_effects_html = ''
            interp_note = (
                '<div style="background:#eef6ff; border-left:4px solid #3498db; '
                'padding:12px 16px; margin:15px 0; border-radius:0 6px 6px 0; font-size:0.95em;">'
                '<b>Интерпретация:</b> '
                'Двухфакторный ANOVA оценивает: (1) основной эффект первого фактора, '
                '(2) основной эффект второго фактора, (3) взаимодействие факторов. '
                'η²: < 0.01 — малый, 0.01–0.06 — средний, > 0.06 — большой эффект. '
            )
            if interaction_sig:
                interp_note += (
                    '<b>Взаимодействие значимо (p < 0.05)</b> — эффект каждого фактора '
                    'зависит от уровня другого. '
                )
                levels_second = sorted(self._current_df[second_factor].unique(), key=str)
                simple_rows = ''
                for lev in levels_second:
                    sub = self._current_df[self._current_df[second_factor] == lev]
                    valid_groups = [(g, sub[sub[g_col] == g][a_col].dropna().values)
                                    for g in sorted(sub[g_col].unique(), key=str)
                                    if len(sub[sub[g_col] == g]) >= 2]
                    if len(valid_groups) >= 2:
                        g_names = [g for g, _ in valid_groups]
                        g_vals = [v for _, v in valid_groups]
                        if len(valid_groups) == 2:
                            _, p_val = sp_stats.ttest_ind(*g_vals, equal_var=False)
                            stat_name, stat_val, test_name = 't', _, 't-тест Уэлча'
                        else:
                            f_stat, p_val = sp_stats.f_oneway(*g_vals)
                            stat_name, stat_val, test_name = 'F', f_stat, 'One-way ANOVA'
                        p_bonf = min(1, p_val * len(levels_second))
                        sig = '✅' if p_bonf < 0.05 else '❌'
                        simple_rows += (f'<tr><td>{second_factor}={lev}</td>'
                                         f'<td>{test_name}</td><td>{", ".join(g_names)}</td>'
                                         f'<td>{stat_val:.3f}</td><td>{p_bonf:.4f}</td><td>{sig}</td></tr>\n')
                simple_effects_html = ''
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
                    'аддитивны и интерпретируются независимо.'
                )
            interp_note += '</div>'
            html_table += interp_note + simple_effects_html

            self._analysis_results['two_way'] = {
                'text': at.to_string(), 'html': html_table,
                'anova_table': anova_table, 'second_factor': second_factor,
                'interaction_sig': interaction_sig,
                'simple_effects_html': simple_effects_html
            }
        except Exception as e:
            self._analysis_results['two_way'] = {'text': f'Ошибка: {e}', 'html': ''}
        return self._analysis_results['two_way']['text']

    def perform_categorical_analysis(self):
        g_col = self.params['group']
        cat_cols = [c for c in self.params.get('cat_multi', []) if c in self._current_df.columns]
        if not cat_cols:
            cat_cols = [c for c in self.params.get('multi', []) if c in self._current_df.columns
                        and (self._current_df[c].dtype == 'object' or self._current_df[c].nunique() < 10)]
        if len(cat_cols) < 1:
            self._analysis_results['categorical'] = {'text': 'Нет категориальных переменных.', 'html': '', 'results': []}
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

        interp_note = (
            '<div style="background:#eef6ff; border-left:4px solid #3498db; '
            'padding:12px 16px; margin:15px 0; border-radius:0 6px 6px 0; font-size:0.95em;">'
            '<b>Интерпретация:</b> '
            'χ² проверяет независимость двух категориальных переменных. '
            'p ≤ 0.05 — значимая связь. V Крамера (0–1): до 0.1 — очень слабая, '
            '0.1–0.3 — слабая, 0.3–0.5 — умеренная, свыше 0.5 — сильная.'
            '</div>'
        )
        html_table += interp_note

        self._analysis_results['categorical'] = {
            'text': summary, 'html': html_table, 'results': results
        }
        return summary

    def perform_frequency_analysis(self):
        cat_cols = [c for c in self.params.get('cat_multi', []) if c in self._current_df.columns]
        if not cat_cols and self.params.get('group') in self._current_df.columns:
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
        g_col = self.params['group']
        all_numeric = [c for c in [self.params.get('analysis')] + self.params.get('multi', [])
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
            "Pillai": ("Pillai's Trace", "Наиболее устойчив к нарушениям предпосылок."),
            "Wilks": ("Wilks' Lambda", "Классический критерий, наиболее мощный при соблюдении всех предпосылок."),
            "Hotelling": ("Hotelling-Lawley Trace", "Сумма собственных значений."),
            "Roy": ("Roy's Greatest Root", "Максимально мощный при эффектах вдоль одного направления.")
        }

        def _run_manova(formula, data):
            manova = MANOVA.from_formula(formula, data=data)
            return manova.mv_test()

        def _extract_factor_results(mv_result, factor_key):
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
            if isinstance(factor_data, dict) and 'stat' in factor_data:
                stat_df = factor_data['stat']
                if isinstance(stat_df, pd.DataFrame):
                    for test_key, (display_name, _) in test_descriptions.items():
                        for idx_name in stat_df.index:
                            if test_key.lower() in str(idx_name).lower():
                                if 'F Value' in stat_df.columns and 'Pr > F' in stat_df.columns:
                                    try:
                                        results[display_name] = {
                                            'F': float(stat_df.loc[idx_name, 'F Value']),
                                            'p': float(stat_df.loc[idx_name, 'Pr > F']),
                                        }
                                    except (ValueError, TypeError):
                                        continue
                                break
            return results

        all_factor_results = {}
        p_values_all = []
        try:
            cat_terms = " + ".join([f'C(Q("{f}"))' for f in cat_factors])
            formula_full = f'{dep_str} ~ {cat_terms}'
            manova_result = _run_manova(formula_full, self._current_df)
            for factor in cat_factors:
                all_factor_results[factor] = _extract_factor_results(manova_result, factor)
                for test_name, vals in all_factor_results[factor].items():
                    p_values_all.append(vals['p'])
        except Exception:
            for factor in cat_factors:
                try:
                    formula_one = f'{dep_str} ~ C(Q("{factor}"))'
                    manova_one = _run_manova(formula_one, self._current_df)
                    all_factor_results[factor] = _extract_factor_results(manova_one, factor)
                    for test_name, vals in all_factor_results[factor].items():
                        p_values_all.append(vals['p'])
                except Exception:
                    all_factor_results[factor] = {}

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
            full_table = (f'<table class="stat-table">'
                          f'<tr><th>Фактор</th><th>Критерий</th><th>F</th>'
                          f'<th>p-value</th><th>Значимость</th></tr>\n{html_rows}</table>')
        else:
            full_table = '<p><i>Не удалось получить результаты MANOVA.</i></p>'

        summary_parts = []
        for factor in cat_factors:
            factor_res = all_factor_results.get(factor, {})
            sig_tests = [t for t in test_names_ordered if t in factor_res and factor_res[t]['p'] <= 0.05]
            if sig_tests:
                summary_parts.append(
                    f'<li><b>{factor}</b>: <span style="color:green;">ЗНАЧИМ</span> '
                    f'(значимо по: {", ".join(sig_tests)})</li>')
            elif any(t in factor_res for t in test_names_ordered):
                summary_parts.append(
                    f'<li><b>{factor}</b>: <span style="color:#c0392b;">НЕ ЗНАЧИМ</span></li>')
        summary_html = f'<ul style="list-style:none; padding-left:0;">{"".join(summary_parts)}</ul>' if summary_parts else ''

        if p_values_all:
            p_min = min(p_values_all)
            any_sig = any(p < 0.05 for p in p_values_all)
            if any_sig:
                conclusion = f'<p><b>Вывод:</b> многомерный эффект факторов <span style="color:green;">ЗНАЧИМ</span> (минимальный p = {p_min:.4f}).</p>'
            else:
                conclusion = f'<p><b>Вывод:</b> многомерный эффект факторов <span style="color:#c0392b;">НЕ ЗНАЧИМ</span> (минимальный p = {p_min:.4f}).</p>'
        else:
            conclusion = '<p><i>Не удалось получить результаты.</i></p>'

        desc_html = '<h4>Особенности критериев MANOVA:</h4><ul>'
        for _, (display_name, description) in test_descriptions.items():
            desc_html += f'<li><b>{display_name}:</b> {description}</li>'
        desc_html += '</ul>'
        desc_html += ('<p><i>MANOVA анализирует влияние факторов на совокупность '
                      'зависимых переменных одновременно, учитывая корреляции между ними.</i></p>')

        manova_html = (
            f'{conclusion}'
            f'{summary_html}'
            f'<details><summary style="cursor:pointer; font-weight:bold;">'
            f'Полная таблица MANOVA ({len(cat_factors)} фактор(ов), {len(dep_cols)} зависимых переменных)</summary>'
            f'{full_table}</details>'
            f'{desc_html}'
        )

        text_summary = f"MANOVA: {len(dep_cols)} зависимых, {len(cat_factors)} факторов.\n"
        if p_values_all:
            text_summary += f"Минимальный p = {min(p_values_all):.4f}\n"

        self._analysis_results['manova'] = {
            'text': text_summary, 'html': manova_html,
            'conclusion': conclusion, 'descriptions': desc_html,
            'dep_cols': dep_cols, 'cat_factors': cat_factors,
        }
        return text_summary

    def perform_posthoc_manova(self):
        g_col = self.params['group']
        all_numeric = [c for c in [self.params.get('analysis')] + self.params.get('multi', [])
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
                    if rej:
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
            var_counts = {}
            for r in significant_rows:
                v = r['variable']
                var_counts.setdefault(v, []).append(r)
            sorted_vars = sorted(var_counts.items(), key=lambda x: -len(x[1]))
            show_vars = sorted_vars[:10]

            summary_html = '<h4>Резюме: значимые различия по переменным</h4>'
            summary_html += '<table class="stat-table" style="width:60%;">'
            summary_html += '<tr><th>Переменная</th><th>Значимых пар</th><th>Примечание</th></tr>\n'
            for v, rows in show_vars:
                pairs_list = [f'{r["g1"]} vs {r["g2"]}' for r in rows]
                pairs_str = ', '.join(pairs_list[:3])
                if len(pairs_list) > 3:
                    pairs_str += f' (+{len(pairs_list)-3})'
                summary_html += f'<tr><td>{v}</td><td>{len(rows)}</td><td>{pairs_str}</td></tr>\n'
            summary_html += '</table>'

            html_rows = ''
            for r in significant_rows:
                html_rows += (f'<tr><td>{r["variable"]}</td><td>{r["g1"]}</td><td>{r["g2"]}</td>'
                              f'<td>{r["md"]:.3f}</td><td>{r["p"]:.4f}</td>'
                              f'<td>{r["lo"]:.3f}</td><td>{r["hi"]:.3f}</td><td>✅</td></tr>\n')
            html_table = (f'<p><b>Найдено значимых попарных различий: {len(significant_rows)}</b></p>'
                          f'{summary_html}'
                          f'<details><summary style="cursor:pointer; font-weight:bold;">Полная таблица ({len(significant_rows)} строк)</summary>'
                          f'<table class="stat-table">'
                          f'<tr><th>Переменная</th><th>Группа 1</th><th>Группа 2</th>'
                          f'<th>Разность</th><th>p-adj</th><th>Нижняя гр.</th>'
                          f'<th>Верхняя гр.</th><th>Различие</th></tr>\n{html_rows}</table></details>')

        self._analysis_results['posthoc_manova'] = {
            'text': f'Значимых различий: {len(significant_rows)}',
            'html': html_table, 'count': len(significant_rows)
        }
        return f'Значимых различий: {len(significant_rows)}'

    # ====================== РЕГРЕССИОННЫЙ АНАЛИЗ ======================
    def perform_linear_regression(self):
        a_col = self.params['analysis']
        predictors = [c for c in self.params.get('multi', [])
                      if c in self._current_df.columns and pd.api.types.is_numeric_dtype(self._current_df[c])
                      and c != a_col]
        if not predictors:
            self._analysis_results['linear_regression'] = {'text': 'Нет предикторов.', 'html': ''}
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

        eq_parts = [f"{c:.3f}·{p}" for p, c in zip(predictors, model.coef_)]
        eq_str = f'{a_col} = ' + ' + '.join(eq_parts) + f' + {model.intercept_:.3f}'

        interp_note = (
            '<div style="background:#eef6ff; border-left:4px solid #3498db; '
            'padding:12px 16px; margin:15px 0; border-radius:0 6px 6px 0; font-size:0.95em;">'
            '<b>Интерпретация:</b> '
            f'R² = {r2:.3f} означает, что модель объясняет {r2*100:.1f}% дисперсии Y. '
            'R² > 0.7 — хорошая модель, 0.5–0.7 — удовлетворительная, < 0.5 — слабая.'
            '</div>'
        )

        html_table = (
            f'<h4>Метрики модели</h4>'
            f'<table class="stat-table" style="width:50%;">'
            f'<tr><th>Метрика</th><th>Значение</th></tr>'
            f'<tr><td>R²</td><td>{r2:.4f}</td></tr>'
            f'<tr><td>RMSE</td><td>{rmse:.4f}</td></tr>'
            f'<tr><td>MAE</td><td>{mae:.4f}</td></tr>'
            f'<tr><td>N (test)</td><td>{len(y_test)}</td></tr></table>'
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
            'html': html_table, 'model': model, 'r2': r2, 'rmse': rmse, 'mae': mae,
            'predictors': predictors, 'a_col': a_col,
            'y_test': y_test, 'y_pred': y_pred
        }
        return f'R²={r2:.3f}, RMSE={rmse:.3f}'

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
                cells = ''.join(f'<td>{coef[c_idx, f_idx]:.4f}</td>' for c_idx in range(len(class_names)))
                coef_rows += f'<tr><td>{f_name}</td>{cells}</tr>\n'
        if len(class_names) > 2:
            header_cells = ''.join(f'<th>{cn}</th>' for cn in class_names)
            coef_table = (
                f'<table class="stat-table" style="width:80%;">'
                f'<tr><th>Предиктор</th>{header_cells}</tr>\n'
                f'{coef_rows}</table>'
            )
        else:
            coef_table = (
                f'<table class="stat-table" style="width:80%;">'
                f'<tr><th>Предиктор</th><th>Коэф.</th></tr>\n'
                f'{coef_rows}</table>'
            )
        html_table = (
            f'<h4>Логистическая регрессия (целевая: {g_col})</h4>'
            f'<p><b>Точность (cross-val):</b> {scores.mean():.3f} ± {scores.std():.3f}</p>'
            f'<h4>Коэффициенты</h4>'
            f'{coef_table}'
            f'<div style="background:#eef6ff; border-left:4px solid #3498db; '
            f'padding:12px 16px; margin:15px 0; border-radius:0 6px 6px 0; font-size:0.95em;">'
            f'<b>Интерпретация:</b> Положительный коэффициент увеличивает шансы принадлежности к классу, '
            f'отрицательный — уменьшает. Точность {scores.mean():.3f} — доля правильных предсказаний.</div>'
        )
        self._analysis_results['logistic_reg_cat'] = {
            'text': f'LogReg точность: {scores.mean():.3f}',
            'html': html_table, 'model': model, 'accuracy': scores.mean(),
            'class_names': class_names.tolist() if hasattr(class_names, 'tolist') else list(class_names),
            'feature_names': feature_names
        }
        return ''

    # ====================== ОТБОР ПРИЗНАКОВ ======================
    def feature_selection_rf(self):
        multi = self.params.get('multi', [])
        if not multi:
            self._analysis_results['rf_importance'] = {}
            return ''
        X = self._current_df[multi].select_dtypes(include=['number'])
        if X.shape[1] < 2:
            self._analysis_results['rf_importance'] = {}
            return ''
        le = LabelEncoder()
        y = le.fit_transform(self._current_df[self.params['group']])
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(X, y)
        importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        self._analysis_results['rf_importance'] = importances.to_dict()
        self._analysis_results['rf_importance_data'] = {
            'features': importances.index.tolist(),
            'values': importances.values.tolist()
        }
        return ''

    def rfe_selection(self):
        multi = self.params.get('multi', [])
        X = self._current_df[multi].select_dtypes(include=['number'])
        if X.shape[1] < 2:
            self._analysis_results['rfe'] = {'text': '', 'selected': []}
            return ''
        le = LabelEncoder()
        y = le.fit_transform(self._current_df[self.params['group']])
        n_features = max(1, X.shape[1] // 2)
        dt = DecisionTreeClassifier(random_state=42, class_weight='balanced')
        rfe = RFE(estimator=dt, n_features_to_select=n_features)
        rfe.fit(X, y)
        selected = [f for f, s in zip(X.columns, rfe.support_) if s]
        eliminated = [f for f, s in zip(X.columns, rfe.support_) if not s]
        text = (f"RFE рекомендует оставить ({len(selected)}): {', '.join(selected)}\n"
                f"RFE рекомендует убрать ({len(eliminated)}): {', '.join(eliminated)}")
        self._analysis_results['rfe'] = {'selected': selected, 'eliminated': eliminated, 'text': text}
        return text

    # ====================== PCA ======================
    def pca_analysis(self):
        multi = self.params.get('multi', [])
        X = self._current_df[multi].select_dtypes(include=['number'])
        if X.shape[1] < 2:
            self._analysis_results['pca'] = {'text': '', 'explained_variance': [], 'loadings': None}
            return ''
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        n_95 = int(np.argmax(cum_var >= 0.95) + 1)
        n_pc = min(5, X.shape[1])
        loadings = pd.DataFrame(pca.components_[:n_pc].T,
                                index=X.columns,
                                columns=[f'PC{i+1}' for i in range(n_pc)])

        self._analysis_results['pca'] = {
            'text': f'Компонент для 95%: {n_95}',
            'explained_variance': pca.explained_variance_ratio_[:n_pc].tolist(),
            'cumulative_variance': cum_var[:n_pc].tolist(),
            'loadings': loadings.to_dict(),
            'n_components_95': n_95
        }
        return f'Компонент для 95%: {n_95}'

    # ====================== КЛАСТЕРНЫЙ АНАЛИЗ ======================
    def determine_optimal_clusters(self, max_k=10):
        multi = self.params.get('multi', [])
        X = self._current_df[multi].select_dtypes(include=['number'])
        if X.shape[1] < 2 or len(X) < 5:
            self._analysis_results['elbow'] = {'text': '', 'optimal_k': 2, 'inertias': [], 'k_range': []}
            return ''
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        inertias = []
        K_range = list(range(1, min(max_k + 1, len(X))))
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        diffs = np.diff(inertias)
        diff_diffs = np.diff(diffs)
        optimal_k = int(np.argmax(diff_diffs) + 2) if len(diff_diffs) > 0 else 2

        self._analysis_results['elbow'] = {
            'text': f'Оптимально k={optimal_k}',
            'optimal_k': optimal_k,
            'inertias': inertias,
            'k_range': K_range
        }
        return f'Оптимально k={optimal_k}'

    def perform_kmeans(self, n_clusters=None):
        multi = self.params.get('multi', [])
        X = self._current_df[multi].select_dtypes(include=['number'])
        if X.shape[1] < 2:
            self._analysis_results['kmeans'] = {'text': '', 'labels': []}
            return ''
        if n_clusters is None:
            elbow_res = self._analysis_results.get('elbow', {})
            n_clusters = elbow_res.get('optimal_k', 3) if elbow_res else 3
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        self._cluster_labels = labels
        self._current_df['Cluster'] = labels

        unique, counts = np.unique(labels, return_counts=True)
        summary_rows = []
        for cl, cnt in zip(unique, counts):
            summary_rows.append({'cluster': int(cl), 'count': int(cnt), 'pct': round(100*cnt/len(labels), 1)})

        html_rows = ''
        for r in summary_rows:
            html_rows += f'<tr><td>{r["cluster"]}</td><td>{r["count"]}</td><td>{r["pct"]}</td></tr>\n'
        html_table = (f'<table class="stat-table" style="width:50%;">'
                      f'<tr><th>Кластер</th><th>Наблюдений</th><th>Доля, %</th></tr>\n{html_rows}</table>')

        cluster_means_raw = self._current_df[multi].groupby(labels).mean().to_dict()
        cluster_means = {}
        cluster_q1 = {}
        cluster_q3 = {}
        for cl_label in sorted(cluster_means_raw.get(next(iter(cluster_means_raw), ''), {}).keys()):
            cluster_means[cl_label] = {col: cluster_means_raw[col][cl_label]
                                        for col in multi if col in cluster_means_raw
                                        and cl_label in cluster_means_raw[col]}
        q1_df = self._current_df[multi].groupby(labels).quantile(0.25)
        q3_df = self._current_df[multi].groupby(labels).quantile(0.75)
        for cl_label in sorted(cluster_means.keys()):
            cluster_q1[cl_label] = {col: q1_df.loc[cl_label, col]
                                     for col in multi if col in q1_df.columns and cl_label in q1_df.index}
            cluster_q3[cl_label] = {col: q3_df.loc[cl_label, col]
                                     for col in multi if col in q3_df.columns and cl_label in q3_df.index}

        self._analysis_results['kmeans'] = {
            'text': f'k={n_clusters}',
            'html': html_table,
            'labels': labels.tolist(),
            'k': n_clusters,
            'cluster_counts': summary_rows,
            'cluster_means': cluster_means,
            'cluster_q1': cluster_q1,
            'cluster_q3': cluster_q3,
            'features': multi
        }
        return f'k={n_clusters}'

    def anova_for_clusters(self):
        multi = self.params.get('multi', [])
        if self._cluster_labels is None:
            return None
        num_cols = [c for c in multi if c in self._current_df.columns
                    and pd.api.types.is_numeric_dtype(self._current_df[c])]
        if not num_cols:
            self._analysis_results['cluster_anova'] = {'text': '', 'results': []}
            return ''

        results = []
        for col in num_cols:
            groups = [g[col].dropna() for _, g in self._current_df.groupby('Cluster')]
            if len(groups) >= 2:
                f_stat, p_val = sp_stats.f_oneway(*groups)
                results.append({'feature': col, 'f': f_stat, 'p': p_val, 'significant': p_val < 0.05})

        html_rows = ''
        for r in results:
            icon = '✅' if r['significant'] else '❌'
            html_rows += (f'<tr><td>{r["feature"]}</td><td>{r["f"]:.3f}</td>'
                          f'<td>{r["p"]:.4f}</td><td>{icon}</td></tr>\n')
        html_table = (f'<table class="stat-table">'
                      f'<tr><th>Признак</th><th>F</th><th>p-value</th>'
                      f'<th>Различие</th></tr>\n{html_rows}</table>')
        self._analysis_results['cluster_anova'] = {'text': '', 'html': html_table, 'results': results}
        return ''

    def save_clusters_to_xlsx(self, filename=None, use_original=True):
        if self._cluster_labels is None:
            return None
        if filename is None:
            base = Path(self.file_name).stem
            filename = f"{base}_with_clusters.xlsx"
        filename = os.path.join(os.getcwd(), filename)
        if use_original:
            df_out = self.df.copy() if hasattr(self, 'df') else self._current_df.copy()
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
        try:
            df_out.to_excel(filename, index=False)
            return filename
        except Exception as e:
            logger.error(f"Ошибка сохранения кластеров: {e}")
            return None

    # ====================== ML ======================
    def _make_model(self, name):
        models = {
            'Random Forest': (RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'), False),
            'LDA': (LinearDiscriminantAnalysis(), True),
            'SVM (RBF)': (SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'), True),
            'SVM (Poly)': (SVC(kernel='poly', degree=3, probability=True, random_state=42, class_weight='balanced'), True),
            'Logistic Regression': (LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42, class_weight='balanced'), True),
            'Decision Tree': (DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced'), False),
        }
        if _XGB_AVAILABLE:
            models['XGBoost'] = (xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                                     random_state=42, verbosity=0), False)
        return models.get(name)

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
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y)
        X_train, X_test = self._align_features(X_train, X_test)
        return X_train, X_test, y_train, y_test, class_names

    def _train_model(self, df, model, model_name, test_size=0.3, use_scaler=False):
        try:
            n_repeats = self._config.get('ml_n_repeats', 10)
            accuracies, aucs = [], []
            last_y_test, last_y_pred, last_y_proba = None, None, None
            class_names = None

            for i in range(n_repeats):
                seed = 42 + i
                X_train, X_test, y_train, y_test, class_names = self._prepare_ml_data(df, test_size, random_state=seed)
                from sklearn.base import clone
                model_iter = clone(model)
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
                y_proba = None
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
                last_y_test, last_y_pred, last_y_proba = y_test, y_pred, y_proba

            acc_mean, acc_std = np.mean(accuracies), np.std(accuracies)
            auc_mean = np.mean(aucs) if aucs else 0.0
            auc_std = np.std(aucs) if aucs else 0.0
            key = model_name.lower().replace(' ', '_')
            self._analysis_results[key] = {
                'accuracy_mean': acc_mean, 'accuracy_std': acc_std,
                'auc_mean': auc_mean, 'auc_std': auc_std,
                'accuracies': accuracies, 'n_repeats': n_repeats,
                'y_test': last_y_test, 'y_pred': last_y_pred, 'y_proba': last_y_proba,
                'class_names': class_names, 'model_name': model_name,
            }
        except Exception as e:
            key = model_name.lower().replace(' ', '_')
            self._analysis_results[key] = {
                'accuracy_mean': 0, 'accuracy_std': 0, 'auc_mean': 0, 'auc_std': 0,
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

    def ml_benchmark(self, df=None, test_size=0.3):
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
            best_key = max(model_labels.keys(),
                           key=lambda k: self._analysis_results.get(k, {}).get('accuracy_mean', 0)
                           if k in self._analysis_results else 0)
            best = self._analysis_results.get(best_key, {})

            self._analysis_results['ml_benchmark'] = {
                'text': f'Моделей: {len(rows)}',
                'html': html_table,
                'table': rows_sorted,
                'best_model': best.get('model_name', ''),
                'best_y_test': best.get('y_test'),
                'best_y_pred': best.get('y_pred'),
                'best_y_proba': best.get('y_proba'),
                'best_class_names': best.get('class_names'),
                'best_auc_mean': best.get('auc_mean', 0),
            }
        else:
            self._analysis_results['ml_benchmark'] = {'text': '', 'html': ''}

    # ====================== МЕЖВЫБОРОЧНЫЕ СРАВНЕНИЯ ======================
    def perform_between_sample_comparison(self):
        multi = self.params.get('multi', [])
        if not multi:
            self._analysis_results['between_sample'] = {'text': '', 'html': ''}
            return ''

        g_col = self.params['group']
        group_names = sorted(self._current_df[g_col].unique(), key=str)
        group_label = f"Группировка: {g_col} ({', '.join(str(g) for g in group_names)})"

        prefixes = {}
        for col in multi:
            if col in self._current_df.columns:
                prefix = col[:3].lower()
                prefixes.setdefault(prefix, []).append(col)

        comparable_groups = [cols for cols in prefixes.values() if len(cols) >= 2]
        if not comparable_groups:
            self._analysis_results['between_sample'] = {'text': 'Нет переменных для межвыборочного сравнения.', 'html': ''}
            return 'Нет переменных для межвыборочного сравнения.'

        all_results = []
        for var_cols in comparable_groups:
            for var_col in var_cols:
                if var_col not in self._current_df.columns:
                    continue
                try:
                    groups = [g[var_col].dropna().values
                              for _, g in self._current_df.groupby(g_col)
                              if len(g[var_col].dropna()) >= 2]
                    if len(groups) < 2:
                        continue
                    h_stat, p_val = kruskal(*groups)
                    all_results.append({'variable': var_col,
                                        'h': h_stat, 'p': p_val, 'significant': p_val < 0.05})
                except Exception:
                    continue

        if all_results:
            html_rows = ''
            for r in all_results:
                sig = '✅' if r['significant'] else '❌'
                html_rows += (f'<tr><td>{r["variable"]}</td>'
                              f'<td>{r["h"]:.3f}</td><td>{r["p"]:.4f}</td>'
                              f'<td>{sig}</td></tr>\n')
            html = (f'<p>{group_label}</p>'
                    f'<h4>Межвыборочные сравнения (критерий Краскела-Уоллиса)</h4>'
                    f'<table class="stat-table">'
                    f'<tr><th>Переменная</th><th>H</th><th>p-value</th><th>Значимость</th></tr>\n'
                    f'{html_rows}</table>')
        else:
            html = '<p>Межвыборочные сравнения не выполнены.</p>'

        self._analysis_results['between_sample'] = {'text': f'Переменных: {len(all_results)}', 'html': html}
        return f'Переменных: {len(all_results)}'

    # ====================== ВИЗУАЛИЗАЦИЯ (Plotly) ======================
    def plot_violin(self, save_html=False, filename="violin_plot.html"):
        """Скрипичная диаграмма с медианой и квартилями"""
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        g_col = self.params['group']
        a_col = self.params['analysis']
        
        fig = make_subplots(rows=1, cols=2, 
                            subplot_titles=(f'Скрипичная диаграмма: {a_col}', 
                                           f'Рояль-диаграмма: {a_col}'))
        
        groups = sorted(self._current_df[g_col].unique(), key=str)
        colors = px.colors.qualitative.Set2
        
        # Левый график - violin с квартилями
        for i, g in enumerate(groups):
            vals = self._current_df[self._current_df[g_col] == g][a_col].dropna()
            fig.add_trace(go.Violin(y=vals, name=str(g), box_visible=True,
                                    meanline_visible=True, line_color=colors[i % len(colors)],
                                    points='outliers'), row=1, col=1)
        
        # Правый график - violin с box
        for i, g in enumerate(groups):
            vals = self._current_df[self._current_df[g_col] == g][a_col].dropna()
            fig.add_trace(go.Violin(y=vals, name=str(g), box_visible=True,
                                    meanline_visible=False, line_color=colors[i % len(colors)],
                                    points=False), row=1, col=2)
        
        fig.update_layout(template='plotly_white', height=500, width=1200,
                          title_text=f'Скрипичная диаграмма: {a_col} по {g_col}')
        fig.update_xaxes(title_text=g_col, row=1, col=1)
        fig.update_xaxes(title_text=g_col, row=1, col=2)
        fig.update_yaxes(title_text=a_col, row=1, col=1)
        fig.update_yaxes(title_text=a_col, row=1, col=2)
        
        if save_html:
            fig.write_html(filename, include_plotlyjs='cdn')
            print(f"График сохранён: {filename}")
        
        try: fig.show()
        except Exception: pass
        return fig

    def plot_boxplot_with_significance(self, save_html=False, filename="boxplot_significance.html"):
        """Ящик с усами с уровнями значимости"""
        import plotly.express as px
        import plotly.graph_objects as go
        from scipy import stats as sp_stats
        from itertools import combinations
        
        g_col = self.params['group']
        a_col = self.params['analysis']
        
        fig = go.Figure()
        groups = sorted(self._current_df[g_col].unique(), key=str)
        colors = px.colors.qualitative.Set2
        
        for i, g in enumerate(groups):
            vals = self._current_df[self._current_df[g_col] == g][a_col].dropna()
            fig.add_trace(go.Box(y=vals, name=str(g), boxpoints='outliers',
                                 marker_color=colors[i % len(colors)]))
        
        # Добавление уровней значимости
        if len(groups) >= 2:
            y_max = self._current_df[a_col].max()
            y_range = self._current_df[a_col].max() - self._current_df[a_col].min()
            if y_range == 0:
                y_range = abs(y_max) if y_max != 0 else 1.0
            
            bracket_y = y_max + y_range * 0.05
            
            for i, j in combinations(range(len(groups)), 2):
                g1_vals = self._current_df[self._current_df[g_col] == groups[i]][a_col].dropna().values
                g2_vals = self._current_df[self._current_df[g_col] == groups[j]][a_col].dropna().values
                
                if len(g1_vals) < 3 or len(g2_vals) < 3:
                    continue
                
                _, p_val = sp_stats.mannwhitneyu(g1_vals, g2_vals, alternative='two-sided')
                
                if p_val < 0.001:
                    sig = '***'
                elif p_val < 0.01:
                    sig = '**'
                elif p_val < 0.05:
                    sig = '*'
                else:
                    continue
                
                # Линия-скобка
                fig.add_shape(type='line', x0=i, x1=j, y0=bracket_y, y1=bracket_y,
                              line=dict(color='#2c3e50', width=2))
                fig.add_shape(type='line', x0=i, x1=i, y0=bracket_y - y_range*0.02, 
                              y1=bracket_y, line=dict(color='#2c3e50', width=2))
                fig.add_shape(type='line', x0=j, x1=j, y0=bracket_y - y_range*0.02, 
                              y1=bracket_y, line=dict(color='#2c3e50', width=2))
                
                # Текст значимости
                fig.add_annotation(x=(i+j)/2, y=bracket_y + y_range*0.02, text=sig,
                                   showarrow=False, font=dict(size=14, color='#e74c3c', 
                                                              family='Arial Black'))
                
                bracket_y += y_range * 0.08
        
        fig.update_layout(title=f'Ящик с усами: {a_col}', yaxis_title=a_col,
                          xaxis_title=g_col, template='plotly_white', 
                          height=500, width=800)
        
        if save_html:
            fig.write_html(filename, include_plotlyjs='cdn')
            print(f"График сохранён: {filename}")
        
        try: fig.show()
        except Exception: pass
        return fig

    def plot_histograms(self, save_html=False, filename="histograms.html"):
        """Гистограммы с расширенной статистикой"""
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import numpy as np
        
        multi = self.params.get('multi', [])
        num_cols = [c for c in multi if c in self._current_df.columns
                    and pd.api.types.is_numeric_dtype(self._current_df[c])]
        
        if not num_cols:
            return None
        
        n = min(len(num_cols), 6)
        cols = num_cols[:n]
        
        fig = make_subplots(rows=1, cols=n, subplot_titles=cols, horizontal_spacing=0.05)
        colors = px.colors.qualitative.Set2
        
        for idx, col in enumerate(cols):
            data = self._current_df[col].dropna()
            
            fig.add_trace(go.Histogram(x=data, name=col, opacity=0.75,
                                       marker_color=colors[idx % len(colors)],
                                       showlegend=False, nbinsx=30), 
                          row=1, col=idx+1)
            
            mean_v = data.mean()
            med_v = data.median()
            mode_v = data.mode().iloc[0] if not data.mode().empty else np.nan
            
            fig.add_vline(x=mean_v, line_dash='dash', line_color='red', 
                          line_width=2, row=1, col=idx+1,
                          annotation_text=f'Ср.={mean_v:.2f}', annotation_position='top')
            fig.add_vline(x=med_v, line_dash='dot', line_color='green', 
                          line_width=2, row=1, col=idx+1,
                          annotation_text=f'Мед.={med_v:.2f}', annotation_position='top')
            
            if not np.isnan(mode_v):
                fig.add_vline(x=mode_v, line_dash='dashdot', line_color='orange', 
                              line_width=2, row=1, col=idx+1,
                              annotation_text=f'Мод.={mode_v:.2f}', annotation_position='bottom')
        
        fig.update_layout(title='Гистограммы с расширенной статистикой',
                          template='plotly_white', height=400, width=500*n,
                          showlegend=False)
        
        if save_html:
            fig.write_html(filename, include_plotlyjs='cdn')
            print(f"График сохранён: {filename}")
        
        try: fig.show()
        except Exception: pass
        return fig

    def plot_pie_chart(self, save_html=False, filename="pie_chart.html"):
        """Круговая диаграмма"""
        import plotly.express as px
        import plotly.graph_objects as go
        
        g_col = self.params['group']
        if g_col not in self._current_df.columns:
            return None
        
        counts = self._current_df[g_col].value_counts()
        
        fig = go.Figure(go.Pie(labels=counts.index.astype(str), values=counts.values,
                               hole=0.3, textinfo='percent+label',
                               marker_colors=px.colors.qualitative.Set2[:len(counts)],
                               textposition='outside', textfont_size=12))
        
        fig.update_layout(title=f'Распределение: {g_col}', 
                          template='plotly_white', height=500, width=600,
                          annotations=[dict(text='N', x=0.5, y=0.5, font_size=20, 
                                           showarrow=False)])
        
        if save_html:
            fig.write_html(filename, include_plotlyjs='cdn')
            print(f"График сохранён: {filename}")
        
        try: fig.show()
        except Exception: pass
        return fig

    def plot_scatter_with_regression(self, save_html=False, filename="scatter_regression.html"):
        """Scatter plot с линией регрессии"""
        import plotly.express as px
        from scipy import stats as sp_stats
        
        g_col = self.params['group']
        a_col = self.params['analysis']
        multi = self.params.get('multi', [])
        
        num_cols = [c for c in multi if c in self._current_df.columns
                    and pd.api.types.is_numeric_dtype(self._current_df[c]) and c != a_col]
        
        if not num_cols:
            return None
        
        x_col = num_cols[0]
        
        fig = px.scatter(self._current_df, x=x_col, y=a_col, color=g_col,
                         trendline='ols', title=f'Scatter + OLS: {x_col} vs {a_col}',
                         color_discrete_sequence=px.colors.qualitative.Set2)
        
        # Добавление статистики
        slope, intercept, r, p, se = sp_stats.linregress(
            self._current_df[x_col].dropna(), self._current_df[a_col].dropna())
        
        fig.add_annotation(text=f'r={r:.3f}, R²={r**2:.3f}, p={p:.2e}',
                           x=0.02, y=0.98, xref='paper', yref='paper',
                           showarrow=False, font=dict(size=12, color='black'),
                           bgcolor='rgba(255, 255, 255, 0.8)', bordercolor='black',
                           borderwidth=1, borderpad=4)
        
        fig.update_layout(template='plotly_white', height=500, width=700)
        
        if save_html:
            fig.write_html(filename, include_plotlyjs='cdn')
            print(f"График сохранён: {filename}")
        
        try: fig.show()
        except Exception: pass
        return fig

    def plot_pairgrid(self, save_html=False, filename="pairgrid.html"):
        """Парная сетка (Scatter Matrix)"""
        import plotly.express as px
        
        g_col = self.params['group']
        multi = self.params.get('multi', [])
        
        num_cols = [c for c in multi if c in self._current_df.columns
                    and pd.api.types.is_numeric_dtype(self._current_df[c])]
        
        if len(num_cols) < 2:
            return None
        
        cols = num_cols[:5]  # Ограничиваем для читаемости
        data = self._current_df[cols + [g_col]].dropna()
        
        fig = px.scatter_matrix(data, dimensions=cols, color=g_col,
                                title="Парная сетка (Scatter Matrix)",
                                color_discrete_sequence=px.colors.qualitative.Set2)
        
        fig.update_traces(diagonal_visible=True, showupperhalf=False,
                          marker=dict(size=5, opacity=0.6))
        fig.update_layout(height=800, width=800, template='plotly_white')
        
        if save_html:
            fig.write_html(filename, include_plotlyjs='cdn')
            print(f"График сохранён: {filename}")
        
        try: fig.show()
        except Exception: pass
        return fig

    def plot_correlation_matrix(self, save_html=False, filename="correlation_matrix.html"):
        """Корреляционная матрица с p-value"""
        import plotly.graph_objects as go
        import numpy as np
        from scipy import stats as sp_stats
        
        multi = self.params.get('multi', [])
        num_cols = [c for c in multi if c in self._current_df.columns
                    and pd.api.types.is_numeric_dtype(self._current_df[c])]
        
        if len(num_cols) < 2:
            return None
        
        corr = self._current_df[num_cols].corr()
        n = len(num_cols)
        
        # Вычисление p-value (попарно по общим строкам)
        p_vals = np.ones((n, n))
        for i in range(n):
            for j in range(i+1, n):
                pair = self._current_df[[num_cols[i], num_cols[j]]].dropna()
                _, p = sp_stats.pearsonr(pair[num_cols[i]], pair[num_cols[j]])
                p_vals[i, j] = p
                p_vals[j, i] = p
        
        corr_vals = corr.values
        
        def _marker(p):
            if p < 0.001: return '***'
            if p < 0.01: return '**'
            if p < 0.05: return '*'
            return '·'
        
        def _fmt(val, p):
            if pd.isna(val): return 'N/A'
            return f'{val:.2f}{_marker(p)}'
        
        # Текст: значения+маркеры только под диагональю, диагональ и выше — пусто
        text_main = np.full((n, n), '', dtype=object)
        text_faded = np.full((n, n), '', dtype=object)
        z_main = np.full((n, n), np.nan)
        z_faded = np.full((n, n), np.nan)
        
        for i in range(n):
            for j in range(n):
                if i > j:
                    if p_vals[i, j] < 0.05:
                        z_main[i, j] = corr_vals[i, j]
                        text_main[i, j] = _fmt(corr_vals[i, j], p_vals[i, j])
                    else:
                        z_faded[i, j] = corr_vals[i, j]
                        text_faded[i, j] = _fmt(corr_vals[i, j], p_vals[i, j])
        
        fig = go.Figure(data=go.Heatmap(
            z=z_main.tolist(), x=num_cols, y=num_cols,
            colorscale='RdBu_r', zmin=-1, zmax=1,
            text=text_main.tolist(), texttemplate='%{text}', textfont={'size': 12},
            showscale=True, colorbar=dict(title='Корреляция r'),
            hovertemplate='%{x} vs %{y}<br>r=%{z:.3f}<extra></extra>'))
        
        # Полупрозрачный слой для незначимых (p≥0.05) ячеек под диагональю
        if np.any(np.isfinite(z_faded)):
            fig.add_trace(go.Heatmap(
                z=z_faded.tolist(), x=num_cols, y=num_cols,
                colorscale='RdBu_r', zmin=-1, zmax=1,
                text=text_faded.tolist(), texttemplate='%{text}',
                textfont={'size': 12, 'color': '#6c757d'}, showscale=False,
                opacity=0.25,
                hovertemplate='%{x} vs %{y}<br>r=%{z:.3f} (p≥0.05)<extra></extra>',
                hoverinfo='skip'))
        
        fig.update_layout(
            title='Корреляционная матрица (*** p<0.001, ** p<0.01, * p<0.05, · — незначимо, p≥0.05)',
            template='plotly_white', height=600, width=700)
        # Основа матрицы: строка з-индекса 0 сверху, значения ниже главной диагонали
        fig.update_yaxes(autorange='reversed')
        
        if save_html:
            fig.write_html(filename, include_plotlyjs='cdn')
            print(f"График сохранён: {filename}")
        
        try: fig.show()
        except Exception: pass
        return fig

    def plot_interaction_effect(self, save_html=False, filename="interaction_effect.html"):
        """Эффект взаимодействия"""
        import plotly.graph_objects as go
        import plotly.express as px
        
        g_col = self.params['group']
        a_col = self.params['analysis']
        second = self._find_second_categorical_factor()
        
        if second is None:
            print("Нет второго категориального фактора для графика взаимодействия.")
            return None
        
        # Вычисление средних и SEM
        grouped = self._current_df.groupby([g_col, second])[a_col].agg(['median', 'count']).reset_index()
        sem = self._current_df.groupby([g_col, second])[a_col].sem().reset_index()
        grouped['sem'] = sem[a_col].values
        
        fig = go.Figure()
        
        hues = sorted(grouped[second].unique(), key=str)
        colors = px.colors.qualitative.Set2
        
        for idx, hue_val in enumerate(hues):
            sub = grouped[grouped[second] == hue_val]
            x_vals = sub[g_col].astype(str).tolist()
            y_vals = sub['median'].tolist()
            error_vals = sub['sem'].tolist()
            
            fig.add_trace(go.Bar(
                x=x_vals, y=y_vals, name=f'{second}={hue_val}',
                error_y=dict(type='data', array=error_vals, visible=True),
                marker_color=colors[idx % len(colors)]))
        
        fig.update_layout(title=f'Взаимодействие: {g_col} × {second} на {a_col}',
                          xaxis_title=g_col, yaxis_title=a_col,
                          template='plotly_white', height=500, width=700,
                          barmode='group')
        
        if save_html:
            fig.write_html(filename, include_plotlyjs='cdn')
            print(f"График сохранён: {filename}")
        
        try: fig.show()
        except Exception: pass
        return fig

    def plot_regression_diagnostics(self, save_html=False, filename="regression_diagnostics.html"):
        """Диагностика регрессии"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import scipy.stats as sp_stats
        import numpy as np
        
        lr_res = self._analysis_results.get('linear_regression', {})
        y_test = lr_res.get('y_test')
        y_pred = lr_res.get('y_pred')
        
        if y_test is None or y_pred is None:
            print("Нет данных регрессии для диагностики.")
            return None
        
        residuals = y_test - y_pred
        
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=('Остатки vs Предсказанные', 'Q-Q plot остатков'))
        
        # График остатков
        fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers',
                                 marker=dict(color='#3498db', opacity=0.6, size=8),
                                 name='Остатки', showlegend=False), row=1, col=1)
        fig.add_hline(y=0, line_dash='dash', line_color='#e74c3c', line_width=2, row=1, col=1)
        
        # Q-Q plot
        sorted_res = np.sort(residuals)
        norm_quantiles = sp_stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_res)))
        
        fig.add_trace(go.Scatter(x=norm_quantiles, y=sorted_res, mode='markers',
                                 marker=dict(color='#3498db', opacity=0.6, size=8),
                                 name='Q-Q', showlegend=False), row=1, col=2)
        
        lim = max(abs(norm_quantiles.min()), abs(norm_quantiles.max()),
                  abs(sorted_res.min()), abs(sorted_res.max()))
        fig.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode='lines',
                                 line=dict(color='#e74c3c', dash='dash', width=2),
                                 name='Идеал', showlegend=False), row=1, col=2)
        
        fig.update_layout(title='Диагностика регрессии', template='plotly_white',
                          height=450, width=1000)
        fig.update_xaxes(title_text='Предсказанные значения', row=1, col=1)
        fig.update_yaxes(title_text='Остатки', row=1, col=1)
        fig.update_xaxes(title_text='Теоретические квантили', row=1, col=2)
        fig.update_yaxes(title_text='Выборочные квантили', row=1, col=2)
        
        if save_html:
            fig.write_html(filename, include_plotlyjs='cdn')
            print(f"График сохранён: {filename}")
        
        try: fig.show()
        except Exception: pass
        return fig
        
    # ====================== WIDGETS (ipywidgets) ======================
    def create_parameter_selector(self):
        import ipywidgets as widgets
        from IPython.display import display, HTML
        cat_cols = self.categorical_cols
        num_cols = self.numeric_cols
        if not cat_cols or not num_cols:
            display(HTML('<p style="color:red;">Нет категориальных или числовых столбцов для выбора.</p>'))
            return
        self._widgets_out = widgets.Output()
        group_w = widgets.Dropdown(options=cat_cols, description='Группировка:', style={'description_width': '120px'})
        analysis_w = widgets.Dropdown(options=num_cols, description='Анализ (Y):', style={'description_width': '120px'})
        multi_w = widgets.SelectMultiple(options=num_cols, value=[c for c in num_cols[:5] if c != num_cols[0]],
                                         description='Признаки X:', rows=8, style={'description_width': '120px'})
        cat_multi_w = widgets.SelectMultiple(options=cat_cols, value=[],
                                             description='Доп. кат.:', rows=5, style={'description_width': '120px'})
        btn = widgets.Button(description='Применить', button_style='primary',
                             layout=widgets.Layout(width='150px'))

        def on_apply(b):
            self.params = {
                'group': group_w.value,
                'analysis': analysis_w.value,
                'multi': list(multi_w.value),
                'cat_multi': list(cat_multi_w.value),
            }
            self._validate_params()
            with self._widgets_out:
                from IPython.display import clear_output
                clear_output(wait=True)
                display(HTML(f'<p style="color:green; font-weight:bold;">✅ Параметры применены:</p>'
                             f'<ul><li><b>Группировка:</b> {self.params["group"]}</li>'
                             f'<li><b>Анализ (Y):</b> {self.params["analysis"]}</li>'
                             f'<li><b>Признаки X:</b> {", ".join(self.params["multi"])}</li>'
                             f'<li><b>Доп. кат.:</b> {", ".join(self.params["cat_multi"]) if self.params["cat_multi"] else "нет"}</li></ul>'))

        btn.on_click(on_apply)
        display(widgets.VBox([
            widgets.HTML('<h3>Выбор параметров анализа</h3>'),
            group_w, analysis_w, multi_w, cat_multi_w,
            btn, self._widgets_out
        ]))

    def create_comment_widgets(self):
        import ipywidgets as widgets
        from IPython.display import display, HTML
        self._comment_widgets = {}
        sections = ['Визуализация', 'ANOVA', 'MANOVA', 'Регрессия', 'Отбор признаков', 'PCA', 'Кластеризация', 'ML']
        boxes = []
        for sec in sections:
            ta = widgets.Textarea(placeholder=f'Комментарий к разделу "{sec}"...', rows=2,
                                  layout=widgets.Layout(width='90%'))
            self._comment_widgets[sec] = ta
            boxes.append(widgets.VBox([widgets.HTML(f'<b>{sec}:</b>'), ta]))
        display(widgets.VBox([widgets.HTML('<h3>Комментарии к разделам отчёта</h3>')] + boxes))

    # ====================== ГЕНЕРАЦИЯ HTML-ОТЧЁТА ======================
    def generate_html_report(self, df_clean=None, sections=None):
        import json as _json
        if df_clean is None:
            df_clean = self._current_df
        if sections is None:
            sections = {k: True for k in ['plots', 'anova', 'manova', 'linear_regression',
                                           'feature_selection', 'pca', 'cluster', 'ml']}
        comments_html = ''
        for sec, widget in getattr(self, '_comment_widgets', {}).items():
            txt = widget.value.strip()
            if txt:
                comments_html += f'<div class="user-comment"><b>{sec}:</b> {txt}</div>\n'

        plotly_figs = {}
        g_col = self.params.get('group', '')
        a_col = self.params.get('analysis', '')
        multi = self.params.get('multi', [])
        cat_multi = self.params.get('cat_multi', [])
        num_cols = [c for c in multi if c in df_clean.columns
                    and pd.api.types.is_numeric_dtype(df_clean[c])]

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.express as px
            has_plotly = True
        except ImportError:
            has_plotly = False

        if has_plotly and sections.get('plots', False):
            colors = px.colors.qualitative.Set2

            fig_methods = [
                ('violin', self.plot_violin),
                ('boxplot', self.plot_boxplot_with_significance),
                ('histogram', self.plot_histograms),
                ('pie', self.plot_pie_chart),
                ('scatter', self.plot_scatter_with_regression),
                ('interaction', self.plot_interaction_effect),
                ('regression_diagnostics', self.plot_regression_diagnostics),
            ]
            for key, method in fig_methods:
                try:
                    fig = method()
                    if fig is not None:
                        plotly_figs[key] = fig
                except Exception as e:
                    logger.warning(f"Ошибка построения графика {key}: {e}")

            # Correlation matrix (inline — uses df_clean directly)
            if len(num_cols) > 1:
                try:
                    corr = df_clean[num_cols].corr()
                    n = len(num_cols)
                    p_mat = [[0.0]*n for _ in range(n)]
                    from scipy import stats as _sp_stats
                    for i in range(n):
                        for j in range(i+1, n):
                            pair = df_clean[[num_cols[i], num_cols[j]]].dropna()
                            _, p = _sp_stats.pearsonr(pair[num_cols[i]], pair[num_cols[j]])
                            p_mat[i][j] = p; p_mat[j][i] = p

                    def _marker(p):
                        if p < 0.001: return '***'
                        if p < 0.01: return '**'
                        if p < 0.05: return '*'
                        return '·'

                    def _fmt(val, p):
                        if pd.isna(val): return 'N/A'
                        return f'{val:.2f}{_marker(p)}'

                    text_main = np.full((n, n), '', dtype=object)
                    text_faded = np.full((n, n), '', dtype=object)
                    z_main = np.full((n, n), np.nan)
                    z_faded = np.full((n, n), np.nan)

                    for i in range(n):
                        for j in range(n):
                            if i > j:
                                if p_mat[i][j] < 0.05:
                                    z_main[i, j] = corr.iloc[i, j]
                                    text_main[i, j] = _fmt(corr.iloc[i, j], p_mat[i][j])
                                else:
                                    z_faded[i, j] = corr.iloc[i, j]
                                    text_faded[i, j] = _fmt(corr.iloc[i, j], p_mat[i][j])

                    fig = go.Figure(data=go.Heatmap(
                        z=z_main.tolist(), x=num_cols, y=num_cols,
                        colorscale='RdBu_r', zmin=-1, zmax=1,
                        text=text_main.tolist(), texttemplate='%{text}',
                        textfont={'size': 12}, showscale=True,
                        colorbar=dict(title='Корреляция r'),
                        hovertemplate='%{x} vs %{y}<br>r=%{z:.3f}<extra></extra>'))

                    if np.any(np.isfinite(z_faded)):
                        fig.add_trace(go.Heatmap(
                            z=z_faded.tolist(), x=num_cols, y=num_cols,
                            colorscale='RdBu_r', zmin=-1, zmax=1,
                            text=text_faded.tolist(), texttemplate='%{text}',
                            textfont={'size': 12, 'color': '#6c757d'}, showscale=False,
                            opacity=0.25,
                            hovertemplate='%{x} vs %{y}<br>r=%{z:.3f} (p≥0.05)<extra></extra>',
                            hoverinfo='skip'))

                    fig.update_layout(
                        title='Корреляционная матрица (*** p<0.001, ** p<0.01, * p<0.05, · — незначимо, p≥0.05)',
                        template='plotly_white', height=550, width=650)
                    # Основа матрицы: строка з-индекса 0 сверху, значения ниже главной диагонали
                    fig.update_yaxes(autorange='reversed')
                    plotly_figs['correlation'] = fig
                except Exception as e:
                    logger.warning(f"Ошибка построения корреляционной матрицы: {e}")

            # RF importance
            rf_data = self._analysis_results.get('rf_importance_data', {})
            if rf_data:
                features = rf_data['features'][::-1]
                values = rf_data['values'][::-1]
                fig = go.Figure(go.Bar(x=values, y=features, orientation='h', marker_color='#3498db',
                                       text=[f'{v:.3f}' for v in values], textposition='outside'))
                fig.update_layout(title='Важность признаков (Random Forest)',
                                  xaxis_title='Важность', template='plotly_white',
                                  height=max(300, len(features)*40))
                plotly_figs['rf_importance'] = fig

            # PCA
            pca_data = self._analysis_results.get('pca', {})
            if pca_data.get('explained_variance'):
                ev = pca_data['explained_variance']
                cum = pca_data['cumulative_variance']
                labels = [f'PC{i+1}' for i in range(len(ev))]
                fig = make_subplots(rows=1, cols=1)
                fig.add_trace(go.Bar(x=labels, y=[v*100 for v in ev], name='Доля дисперсии',
                                     marker_color='#3498db', text=[f'{v*100:.1f}%' for v in ev],
                                     textposition='outside'))
                fig.add_trace(go.Scatter(x=labels, y=[v*100 for v in cum], mode='lines+markers',
                                         name='Кумулятивная', marker=dict(color='#e74c3c', size=8)))
                fig.add_hline(y=95, line_dash='dash', line_color='#27ae60', annotation_text='95%')
                fig.update_layout(title='Метод главных компонент (PCA)', template='plotly_white', height=400)
                plotly_figs['pca'] = fig

            # Elbow
            elbow = self._analysis_results.get('elbow', {})
            if elbow.get('inertias'):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=elbow['k_range'], y=elbow['inertias'],
                                         mode='lines+markers', marker=dict(size=10, color='#3498db')))
                fig.add_vline(x=elbow.get('optimal_k', 2), line_dash='dash', line_color='#e74c3c',
                              annotation_text=f"Optimal k={elbow.get('optimal_k', 2)}")
                fig.update_layout(title='Метод каменистой осыпи (Elbow)', xaxis_title='Число кластеров k',
                                  yaxis_title='Инерция (WCSS)', template='plotly_white', height=400)
                plotly_figs['elbow'] = fig

            # K-Means scatter (PCA 2D)
            kmeans = self._analysis_results.get('kmeans', {})
            if kmeans.get('labels') and len(num_cols) >= 2:
                from sklearn.decomposition import PCA as _PCA
                from sklearn.preprocessing import StandardScaler as _SS
                X_pca_data = df_clean[num_cols].dropna()
                labels_arr = np.array(kmeans['labels'])
                if len(labels_arr) == len(X_pca_data):
                    pca2 = _PCA(n_components=2, random_state=42)
                    X_2d = pca2.fit_transform(_SS().fit_transform(X_pca_data))
                    fig = px.scatter(x=X_2d[:, 0], y=X_2d[:, 1], color=labels_arr.astype(str),
                                     title=f'K-Means кластеризация (k={kmeans["k"]})',
                                     labels={'x': 'PC1', 'y': 'PC2', 'color': 'Кластер'},
                                     color_discrete_sequence=colors)
                    fig.update_layout(template='plotly_white', height=500)
                    plotly_figs['kmeans_scatter'] = fig

            # Confusion matrix
            ml = self._analysis_results.get('ml_benchmark', {})
            if ml.get('best_y_test') is not None:
                from sklearn.metrics import confusion_matrix as _cm
                y_true = ml['best_y_test']; y_pred_ml = ml['best_y_pred']
                cn = ml.get('best_class_names', [])
                if cn is None or len(cn) == 0:
                    cn = [str(i) for i in range(len(np.unique(y_true)))]
                cm = _cm(y_true, y_pred_ml)
                fig = go.Figure(data=go.Heatmap(z=cm, x=cn, y=cn, colorscale='Blues',
                                                text=cm, texttemplate='%{text}', textfont={'size': 14}))
                fig.update_layout(title=f'Матрица ошибок: {ml.get("best_model", "")}',
                                  xaxis_title='Предсказанные', yaxis_title='Истинные',
                                  template='plotly_white', height=400)
                plotly_figs['confusion_matrix'] = fig

            # ROC curve
            if ml.get('best_y_proba') is not None:
                from sklearn.metrics import roc_curve as _roc, roc_auc_score as _roc_auc
                y_test_roc = ml['best_y_test']; y_proba = ml['best_y_proba']
                cn_roc = ml.get('best_class_names', [])
                if cn_roc is None or len(cn_roc) == 0:
                    cn_roc = [str(i) for i in range(len(np.unique(y_test_roc)))]
                nc = len(cn_roc)
                fig = go.Figure()
                if nc == 2:
                    fpr, tpr, _ = _roc(y_test_roc, y_proba[:, 1])
                    auc = ml.get('best_auc_mean', 0)
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'AUC={auc:.3f}',
                                             line=dict(width=2.5, color=colors[0])))
                else:
                    for i in range(nc):
                        fpr, tpr, _ = _roc((y_test_roc == i).astype(int), y_proba[:, i])
                        auc_i = _roc_auc((y_test_roc == i).astype(int), y_proba[:, i])
                        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{cn_roc[i]} (AUC={auc_i:.3f})',
                                                 line=dict(width=2.5, color=colors[i % len(colors)])))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                         line=dict(dash='dash', color='gray', width=1.5), showlegend=False))
                fig.update_layout(title=f'ROC-кривая: {ml.get("best_model", "")}',
                                  xaxis_title='FPR', yaxis_title='TPR',
                                  template='plotly_white', height=400)
                plotly_figs['roc'] = fig

        # --- Сборка HTML ---
        section_charts = {
            'plots': ['violin', 'boxplot', 'histogram', 'pie', 'scatter', 'correlation', 'interaction'],
            'linear_regression': ['regression_diagnostics'],
            'feature_selection': ['rf_importance'],
            'pca': ['pca'],
            'cluster': ['elbow', 'kmeans_scatter'],
            'ml': ['confusion_matrix', 'roc'],
        }
        section_subsections = {
            'anova': [
                ('3.1', 'One-Way ANOVA / Kruskal-Wallis', ['anova']),
                ('3.2', 'Post-hoc анализ', ['tukey']),
                ('3.3', 'Two-Way ANOVA', ['two_way']),
            ],
            'manova': [
                ('4.1', 'MANOVA', ['manova']),
                ('4.2', 'Post-hoc MANOVA', ['posthoc_manova']),
            ],
            'linear_regression': [
                ('5.1', 'Линейная регрессия', ['linear_regression']),
                ('5.3', 'Логистическая регрессия', ['logistic_reg_cat']),
            ],
            'feature_selection': [
                ('6.2', 'Рекурсивное устранение (RFE)', ['rfe']),
            ],
            'pca': [
                ('7.1', 'PCA', ['pca']),
            ],
            'cluster': [
                ('8.2', 'K-Means кластеризация', ['kmeans']),
                ('8.3', 'ANOVA для кластеров', ['cluster_anova']),
            ],
            'ml': [
                ('9.1', 'ML Бенчмарк', ['ml_benchmark']),
            ],
        }

        sections_html = ''
        chart_idx = 0

        # === Секция предобработки ===
        stats = self._preprocessing_stats
        if stats:
            sections_html += '<div class="section"><h2>1. Предобработка данных</h2>\n'
            sections_html += (
                '<table class="stat-table" style="width:60%;">'
                '<tr><th>Показатель</th><th>Значение</th></tr>'
                f'<tr><td>Всего строк в исходных данных</td><td>{stats.get("total_rows", 0)}</td></tr>'
                f'<tr><td>Исключено строк без группирующей переменной ({stats.get("group_col", "")})</td>'
                f'<td>{stats.get("excluded_no_group", 0)}</td></tr>'
                f'<tr><td>Исключено строк с пропусками в других столбцах</td>'
                f'<td>{stats.get("excluded_other_missing", 0)}</td></tr>'
                f'<tr><td>Исключено выбросов (z-score)</td><td>{stats.get("excluded_outliers", 0)}</td></tr>'
                f'<tr><td><b>Осталось для анализа</b></td><td><b>{stats.get("final_analyzed", 0)}</b></td></tr>'
                f'<tr><td>Всего исключено</td><td>{stats.get("total_excluded", 0)}</td></tr>'
                '</table>'
            )

            missing = stats.get('missing_per_column', {})
            if missing:
                miss_rows = ''.join(f'<tr><td>{col}</td><td>{cnt}</td></tr>' for col, cnt in missing.items() if cnt > 0)
                if miss_rows:
                    sections_html += (
                        '<table class="stat-table" style="width:50%;">'
                        '<tr><th>Столбец</th><th>Пропусков</th></tr>'
                        f'{miss_rows}</table>'
                    )

            corr_removals = stats.get('correlation_removals', [])
            corr_threshold = stats.get('correlation_threshold', 0.9)
            corr_pairs = stats.get('corr_pairs', [])
            removal_set = {(k, d) for k, d, _ in corr_removals} if corr_removals else set()
            if corr_pairs:
                corr_rows = ''
                for c1, c2, r in corr_pairs:
                    status = 'удалён' if (c1, c2) in removal_set or (c2, c1) in removal_set else 'оставлен'
                    corr_rows += f'<tr><td>{c1}</td><td>{c2}</td><td>{r:.3f}</td><td>{status}</td></tr>\n'
                sections_html += (
                    f'<table class="stat-table" style="width:70%;">'
                    f'<tr><th>Признак 1</th><th>Признак 2</th><th>|r|</th><th>Статус</th></tr>{corr_rows}</table>'
                    f'<p>Порог: {corr_threshold}. Показаны все пары с |r| ≥ порога.</p>'
                )
            else:
                sections_html += '<p>Высококоррелированных признаков не обнаружено — все признаки сохранены.</p>'

            cat_data = self._analysis_results.get('categorical', {})
            cat_html = cat_data.get('html', '')
            if cat_html:
                sections_html += cat_html
            else:
                sections_html += '<p>Значимых связей между категориальными признаками не обнаружено.</p>'

            sections_html += '</div>\n'

        for sec_key in ['plots', 'anova', 'manova', 'linear_regression', 'feature_selection', 'pca', 'cluster', 'ml']:
            if not sections.get(sec_key, False):
                continue

            has_charts = sec_key in section_charts and any(c in plotly_figs for c in section_charts[sec_key])
            has_subsections = sec_key in section_subsections
            has_results = False
            if has_subsections:
                for _, _, result_keys in section_subsections[sec_key]:
                    for rk in result_keys:
                        data = self._analysis_results.get(rk)
                        if data and (data.get('html') or data.get('text')):
                            has_results = True
                            break
            if not has_charts and not has_results:
                continue

            sec_title_map = {
                'plots': 'Визуализация', 'anova': 'Дисперсионный анализ',
                'manova': 'Многомерный анализ', 'linear_regression': 'Регрессионный анализ',
                'feature_selection': 'Отбор признаков', 'pca': 'Метод главных компонент',
                'cluster': 'Кластерный анализ', 'ml': 'Машинное обучение',
            }
            sections_html += f'<div class="section"><h2>{sec_title_map[sec_key]}</h2>\n'

            if sec_key == 'plots' and sec_key in section_charts:
                for ck in section_charts[sec_key]:
                    if ck in plotly_figs:
                        chart_idx += 1
                        pid = f'plot_{chart_idx}'
                        fig_json = self._fig_to_json(plotly_figs[ck])
                        sections_html += f'''<div class="chart-container"><div id="{pid}"></div></div>
<script>(function(){{ var fig = {fig_json}; var el = document.getElementById('{pid}'); Plotly.newPlot(el, fig.data, fig.layout, {{responsive:true, displayModeBar:true}}); }})();</script>\n'''

            if has_subsections:
                for sub_num, sub_title, result_keys in section_subsections[sec_key]:
                    sub_has_content = False
                    sub_content = ''

                    if sec_key in section_charts:
                        for ck in section_charts[sec_key]:
                            if ck in plotly_figs and sub_num in ('3.1', '4.1', '5.1', '7.1', '8.2', '9.1'):
                                chart_idx += 1
                                pid = f'plot_{sub_num}_{chart_idx}'
                                fig_json = self._fig_to_json(plotly_figs[ck])
                                sub_content += f'''<div class="chart-container"><div id="{pid}"></div></div>
<script>(function(){{ var fig = {fig_json}; var el = document.getElementById('{pid}'); Plotly.newPlot(el, fig.data, fig.layout, {{responsive:true, displayModeBar:true}}); }})();</script>\n'''
                                sub_has_content = True

                    for rk in result_keys:
                        data = self._analysis_results.get(rk)
                        if not data:
                            continue
                        content = data.get('html', '')
                        if not content:
                            text = data.get('text', '')
                            if text:
                                content = f'<pre style="background:#f8f9fa; padding:12px; border-radius:6px;">{text}</pre>'
                        if content:
                            sub_content += f'<div class="result-card"><div class="result-content">{content}</div></div>\n'
                            sub_has_content = True

                    if sub_has_content:
                        sections_html += f'<h3>{sub_num} {sub_title}</h3>\n'
                        sections_html += sub_content
            else:
                if sec_key in section_charts:
                    for ck in section_charts[sec_key]:
                        if ck in plotly_figs:
                            chart_idx += 1
                            pid = f'plot_{chart_idx}'
                            fig_json = self._fig_to_json(plotly_figs[ck])
                            sections_html += f'''<div class="chart-container"><div id="{pid}"></div></div>
<script>(function(){{ var fig = {fig_json}; var el = document.getElementById('{pid}'); Plotly.newPlot(el, fig.data, fig.layout, {{responsive:true, displayModeBar:true}}); }})();</script>\n'''

            sections_html += '</div>\n'

        html = f"""<!DOCTYPE html>
<html lang="ru"><head>
<meta charset="UTF-8"><title>Отчёт — {self.file_name}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
body {{ font-family: 'Segoe UI', sans-serif; background: #f8f9fa; color: #343a40; margin: 0; padding: 20px; line-height: 1.6; }}
h1 {{ text-align: center; color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
h2 {{ color: #2980b9; border-left: 4px solid #3498db; padding-left: 10px; margin-top: 30px; }}
.section {{ margin-bottom: 20px; }}
.stat-table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 0.95em; }}
.stat-table th {{ background: #3498db; color: white; padding: 10px 12px; border: 1px solid #2980b9; text-align: left; }}
.stat-table td {{ padding: 8px 12px; border: 1px solid #d0d7de; }}
.stat-table tr:nth-child(even) {{ background: #f8f9fa; }}
.stat-table tr:hover {{ background: #eaf4fc; }}
.chart-container {{ background: #fff; border: 1px solid #dee2e6; border-radius: 8px; padding: 15px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
.result-card {{ background: #fff; border: 1px solid #dee2e6; border-radius: 8px; padding: 15px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
.user-comment {{ background: #fffde7; border-left: 4px solid #fbc02d; padding: 12px 16px; margin: 15px 0; border-radius: 0 6px 6px 0; }}
</style></head><body>
<h1>Статистический анализ: {self.file_name}</h1>
<p style="text-align:center; color:#6c757d;">Группировка: <b>{self.params.get('group','')}</b> | Y: <b>{self.params.get('analysis','')}</b></p>
{comments_html}
{sections_html}
<hr style="margin-top:50px; border:0; border-top:1px solid #eee;">
<p style="color:#999; font-size:0.8em; text-align:center;">Сгенерировано DataAn Enhanced</p>
</body></html>"""
        filename = f"{Path(self.file_name).stem}_report.html"
        filepath = os.path.join(os.getcwd(), filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f'HTML-отчёт сохранён: {filepath}')
        return filepath
