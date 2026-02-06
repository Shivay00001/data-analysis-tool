# Advanced Data Analysis Tool

A production-ready Python tool for automated data profiling, cleaning, analysis, and machine learning preprocessing.

## Features

- **Comprehensive Profiling**: Generates detailed reports on data quality, missing values, duplicates, and distributions.
- **Auto-Cleaning**: Intelligent automatic data cleaning strategies (auto, aggressive, conservative).
- **Advanced Visualization**: Generates a suite of visualizations (heatmaps, correlation matrices, boxplots, etc.) using `seaborn` and `matplotlib`.
- **Statistical Testing**: Automated normality tests (Shapiro-Wilk) and independence tests (Chi-square).
- **ML Preprocessing**: Automated encoding, scaling, imputation, and feature selection for machine learning readiness.
- **Dimensionality Reduction**: Integrated PCA analysis.
- **Anomaly Detection**: Outlier detection using Isolation Forest and statistical methods (IQR, Z-score).
- **Automated Insights**: Generates text-based insights about data quality, skewness, and correlations.

## Usage

```python
import pandas as pd
from data_analysis_tool import AdvancedDataAnalyzer

# Load your dataset
df = pd.read_csv("your_data.csv")

# Initialize analyzer
analyzer = AdvancedDataAnalyzer(df, name="My Dataset")

# Run full pipeline
analyzer.full_pipeline(auto_clean=True, visualize=True, ml_prep=True)

# Or use specific modules
report = analyzer.comprehensive_profiling()
cleaned_df = analyzer.auto_clean_data(strategy="auto")
analyzer.advanced_visualization_suite()
```

## Dependencies

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scipy`
- `scikit-learn`
