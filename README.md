# DataAn Enhanced — Automated Statistical Analysis

Automated statistical analysis tool with interactive CLI and Jupyter interfaces.  
Load an `.xlsx` / `.csv` file, select variables, and get a full HTML report with visualizations, hypothesis tests, regression, feature selection, PCA, clustering, and ML benchmark.

## Quick Start

```bash
pip install -r requirements.txt
python main.py path/to/data.xlsx
```

Or run the Jupyter notebook `DataAn_Enhanced.ipynb`.

## Interfaces

| Interface | File | Description |
|-----------|------|-------------|
| **CLI** | `main.py` | Interactive terminal. Supports `main.py file.xlsx` for direct load |
| **Jupyter** | `DataAn_Enhanced.ipynb` | Widget‑based parameter selection and report generation |
| **Library** | `analyzer_enhanced.py` | `DataAnalyzer` class — import and use programmatically |

## Data Requirements

- Format: `.xlsx` (first sheet) or `.csv`
- Minimum **1 categorical** column (grouping variable)
- Minimum **1 numerical** column (target / multivariate features)

## Pipeline

### 0. Preprocessing
- Missing value removal
- Z‑score outlier removal (configurable, default ±3σ)
- High‑correlation filtering (|r| > 0.9, configurable)
- Bootstrap group balancing (optional)
- Session persistence (`analyzer_session.json`) — restores last file params

### 1. Visualization (8 plot types)
- Violin + swarm plot with statistics
- Boxplot with significance brackets (`*`, `**`, `***`); outliers toggleable
- Histograms with mean / median / mode
- Pie chart (categorical proportions)
- Scatterplot with regression line
- PairGrid (upper scatter, lower KDE, diagonal histogram)
- Correlation matrix (non‑significant correlations semi‑transparent)
- Interaction plot (Two‑way ANOVA)

### 2. ANOVA / Categorical
- **One‑Way ANOVA** (parametric, normality + homogeneity checked) or **Kruskal‑Wallis** (non‑parametric)
- **Post‑hoc**: Tukey HSD after ANOVA; **Dunn test with Holm correction** after Kruskal‑Wallis
- **Two‑way ANOVA** with interaction term; if interaction is significant → simple effects (Welch t‑test or one‑way ANOVA within each level of the second factor, Bonferroni‑corrected)
- Chi‑squared / Fisher test + Cramér's V for categorical pairs
- Frequency tables

### 3. MANOVA
- All four criteria: Pillai, Wilks, Hotelling, Roy
- Post‑hoc MANOVA (Tukey HSD per dependent variable)

### 4. Regression Analysis
- **Linear regression** — R², RMSE, MAE, coefficients, diagnostic plots (residuals vs fitted, Q‑Q)
- **Logistic regression** — categorical + numerical predictors → group classification, cross‑validated accuracy

### 5. Feature Selection
- **Random Forest** — feature importance bar chart with values
- **RFE** (Recursive Feature Elimination) with Decision Tree

### 6. PCA
- Scree plot (individual + cumulative variance, 95% threshold)
- Loadings heatmap (first 5 components)

### 7. Cluster Analysis
- **Elbow method** — optimal k detection
- **K‑means** — clustering with PCA projection, centroids
- **ANOVA for clusters** — one‑way ANOVA on each feature across clusters
- Boxplots per cluster
- **Cluster profiles** — line plot with ±1σ shaded error bands
- Categorical feature distribution per cluster
- XLSX export with `cluster` column

### 8. Machine Learning Benchmark
- Models: Random Forest, Logistic Regression, Decision Tree, SVM (RBF), SVM (Poly), LDA, XGBoost (optional)
- 10‑fold repeated train/test split
- Accuracy ± std, AUC, ranking table
- Confusion matrix + ROC curve for the best model

## Output

An HTML report `{filename}_report.html` is generated in the working directory. Features:

- Table of contents with anchors
- All plots embedded as base64 (click to enlarge)
- Interpretation notes for every statistical test
- User comment sections
- Responsive CSS styling

## Configuration

Create `analyzer_config.json` in the working directory:

```json
{
  "precision": 3,
  "correlation_threshold": 0.9,
  "z_score_threshold": 3.0,
  "remove_outliers": true,
  "balance_groups": false,
  "bootstrap_min_size": 30,
  "bootstrap_max_ratio": 3.0,
  "ml_n_repeats": 10,
  "show_boxplot_outliers": true
}
```

## Session Persistence

The tool saves the last file path and variable selections to `analyzer_session.json`.  
When the same file is loaded again, parameters are pre‑filled automatically.

## Dependencies

Core: `pandas`, `numpy`, `scipy`, `scikit-learn`, `statsmodels`, `seaborn`, `matplotlib`  
Optional: `xgboost` (ML benchmark), `ipywidgets` (Jupyter), `openpyxl` / `xlrd` (Excel)

## Author

**black_paladin@mail.ru**  
License: MIT
