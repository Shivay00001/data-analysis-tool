"""
Advanced Data Analysis & DSA Tool
A production-ready comprehensive tool for automatic data analysis, data science, and DSA operations.
Includes: data cleaning, profiling, visualization, statistical analysis, ML preprocessing, and more.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer, KNNImputer
from datetime import datetime
import warnings
import json
import os
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
import hashlib

warnings.filterwarnings('ignore')

class AdvancedDataAnalyzer:
    """
    Production-ready comprehensive data analysis tool with security and all best practices.
    """
    
    def __init__(self, data: pd.DataFrame, name: str = "Dataset"):
        """Initialize the analyzer with a dataset."""
        self.data = data.copy()
        self.original_data = data.copy()
        self.name = name
        self.report = {}
        self.numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
        self._initialize_security()
        
    def _initialize_security(self):
        """Initialize security and tracking mechanisms."""
        self.session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
        self.operations_log = []
        self._log_operation("Initialized", "Dataset loaded successfully")
        
    def _log_operation(self, operation: str, details: str):
        """Log all operations for audit trail."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "details": details,
            "session_id": self.session_id
        }
        self.operations_log.append(log_entry)
        
    def comprehensive_profiling(self) -> Dict:
        """Generate comprehensive data profiling report."""
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE DATA PROFILING REPORT: {self.name}")
        print(f"{'='*80}\n")
        
        profile = {
            "basic_info": self._basic_info(),
            "column_analysis": self._column_analysis(),
            "missing_data": self._missing_data_analysis(),
            "duplicates": self._duplicate_analysis(),
            "statistical_summary": self._statistical_summary(),
            "correlation_analysis": self._correlation_analysis(),
            "outlier_detection": self._outlier_detection(),
            "data_quality_score": self._calculate_quality_score()
        }
        
        self._log_operation("Profiling", "Comprehensive profiling completed")
        self.report["profiling"] = profile
        return profile
    
    def _basic_info(self) -> Dict:
        """Get basic dataset information."""
        info = {
            "rows": len(self.data),
            "columns": len(self.data.columns),
            "numeric_columns": len(self.numeric_cols),
            "categorical_columns": len(self.categorical_cols),
            "datetime_columns": len(self.datetime_cols),
            "memory_usage_mb": self.data.memory_usage(deep=True).sum() / 1024**2,
            "duplicate_rows": self.data.duplicated().sum()
        }
        
        print("📊 BASIC INFORMATION")
        print(f"   Rows: {info['rows']:,}")
        print(f"   Columns: {info['columns']}")
        print(f"   Numeric: {info['numeric_columns']}, Categorical: {info['categorical_columns']}, Datetime: {info['datetime_columns']}")
        print(f"   Memory Usage: {info['memory_usage_mb']:.2f} MB")
        print(f"   Duplicate Rows: {info['duplicate_rows']}\n")
        
        return info
    
    def _column_analysis(self) -> Dict:
        """Analyze each column in detail."""
        analysis = {}
        
        print("📋 COLUMN-WISE ANALYSIS")
        for col in self.data.columns:
            col_info = {
                "dtype": str(self.data[col].dtype),
                "non_null": self.data[col].notna().sum(),
                "null_count": self.data[col].isna().sum(),
                "null_percentage": (self.data[col].isna().sum() / len(self.data)) * 100,
                "unique_values": self.data[col].nunique(),
                "unique_percentage": (self.data[col].nunique() / len(self.data)) * 100
            }
            
            if col in self.numeric_cols:
                col_info.update({
                    "mean": self.data[col].mean(),
                    "median": self.data[col].median(),
                    "std": self.data[col].std(),
                    "min": self.data[col].min(),
                    "max": self.data[col].max(),
                    "skewness": self.data[col].skew(),
                    "kurtosis": self.data[col].kurtosis()
                })
            elif col in self.categorical_cols:
                col_info["top_values"] = self.data[col].value_counts().head(5).to_dict()
                
            analysis[col] = col_info
            print(f"   {col}: {col_info['dtype']}, Null: {col_info['null_percentage']:.1f}%, Unique: {col_info['unique_values']}")
        
        print()
        return analysis
    
    def _missing_data_analysis(self) -> Dict:
        """Comprehensive missing data analysis."""
        missing = {
            "total_missing": self.data.isna().sum().sum(),
            "percentage": (self.data.isna().sum().sum() / (len(self.data) * len(self.data.columns))) * 100,
            "columns_with_missing": {},
            "rows_with_missing": self.data.isna().any(axis=1).sum()
        }
        
        print("🔍 MISSING DATA ANALYSIS")
        print(f"   Total Missing Values: {missing['total_missing']:,}")
        print(f"   Overall Missing Percentage: {missing['percentage']:.2f}%")
        print(f"   Rows with Missing Data: {missing['rows_with_missing']:,}\n")
        
        for col in self.data.columns:
            if self.data[col].isna().sum() > 0:
                missing["columns_with_missing"][col] = {
                    "count": int(self.data[col].isna().sum()),
                    "percentage": float((self.data[col].isna().sum() / len(self.data)) * 100)
                }
        
        return missing
    
    def _duplicate_analysis(self) -> Dict:
        """Analyze duplicate records."""
        duplicates = {
            "total_duplicates": self.data.duplicated().sum(),
            "percentage": (self.data.duplicated().sum() / len(self.data)) * 100,
            "duplicate_subsets": {}
        }
        
        print("🔄 DUPLICATE ANALYSIS")
        print(f"   Total Duplicates: {duplicates['total_duplicates']}")
        print(f"   Percentage: {duplicates['percentage']:.2f}%\n")
        
        return duplicates
    
    def _statistical_summary(self) -> Dict:
        """Generate statistical summary."""
        summary = {}
        
        print("📈 STATISTICAL SUMMARY")
        if self.numeric_cols:
            summary["numeric"] = self.data[self.numeric_cols].describe().to_dict()
            print("   Numeric columns statistical summary generated")
        
        if self.categorical_cols:
            summary["categorical"] = {}
            for col in self.categorical_cols[:5]:  # Top 5 categorical
                summary["categorical"][col] = self.data[col].value_counts().head(10).to_dict()
            print("   Categorical columns frequency analysis completed")
        
        print()
        return summary
    
    def _correlation_analysis(self) -> Dict:
        """Analyze correlations between numeric variables."""
        corr_data = {}
        
        if len(self.numeric_cols) > 1:
            corr_matrix = self.data[self.numeric_cols].corr()
            
            # Find high correlations
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        high_corr.append({
                            "var1": corr_matrix.columns[i],
                            "var2": corr_matrix.columns[j],
                            "correlation": float(corr_matrix.iloc[i, j])
                        })
            
            corr_data = {
                "correlation_matrix": corr_matrix.to_dict(),
                "high_correlations": high_corr
            }
            
            print("🔗 CORRELATION ANALYSIS")
            print(f"   Found {len(high_corr)} high correlations (|r| > 0.7)\n")
        
        return corr_data
    
    def _outlier_detection(self) -> Dict:
        """Detect outliers using multiple methods."""
        outliers = {}
        
        print("⚠️  OUTLIER DETECTION")
        
        for col in self.numeric_cols:
            if self.data[col].notna().sum() > 0:
                # IQR Method
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                iqr_outliers = ((self.data[col] < lower_bound) | (self.data[col] > upper_bound)).sum()
                
                # Z-score Method
                z_scores = np.abs(stats.zscore(self.data[col].dropna()))
                z_outliers = (z_scores > 3).sum()
                
                outliers[col] = {
                    "iqr_method": int(iqr_outliers),
                    "zscore_method": int(z_outliers),
                    "percentage": float((iqr_outliers / len(self.data)) * 100)
                }
                
                if iqr_outliers > 0:
                    print(f"   {col}: {iqr_outliers} outliers ({outliers[col]['percentage']:.1f}%)")
        
        print()
        return outliers
    
    def _calculate_quality_score(self) -> Dict:
        """Calculate overall data quality score."""
        scores = {
            "completeness": (1 - self.data.isna().sum().sum() / (len(self.data) * len(self.data.columns))) * 100,
            "uniqueness": (1 - self.data.duplicated().sum() / len(self.data)) * 100,
            "consistency": 100  # Placeholder for consistency checks
        }
        
        scores["overall"] = np.mean([scores["completeness"], scores["uniqueness"], scores["consistency"]])
        
        print("⭐ DATA QUALITY SCORE")
        print(f"   Completeness: {scores['completeness']:.1f}%")
        print(f"   Uniqueness: {scores['uniqueness']:.1f}%")
        print(f"   Overall Quality: {scores['overall']:.1f}%\n")
        
        return scores
    
    def auto_clean_data(self, strategy: str = "auto") -> pd.DataFrame:
        """
        Automatically clean data with comprehensive strategies.
        
        Strategies:
        - auto: Intelligent automatic cleaning
        - aggressive: Remove all problematic data
        - conservative: Minimal cleaning, preserve data
        """
        print(f"\n{'='*80}")
        print(f"AUTOMATIC DATA CLEANING: {strategy.upper()} MODE")
        print(f"{'='*80}\n")
        
        cleaned_data = self.data.copy()
        cleaning_report = []
        
        # 1. Remove duplicate rows
        initial_rows = len(cleaned_data)
        cleaned_data = cleaned_data.drop_duplicates()
        removed_dupes = initial_rows - len(cleaned_data)
        if removed_dupes > 0:
            cleaning_report.append(f"✓ Removed {removed_dupes} duplicate rows")
        
        # 2. Handle missing values
        for col in cleaned_data.columns:
            missing_pct = (cleaned_data[col].isna().sum() / len(cleaned_data)) * 100
            
            if missing_pct > 50 and strategy == "aggressive":
                cleaned_data = cleaned_data.drop(columns=[col])
                cleaning_report.append(f"✓ Dropped column '{col}' ({missing_pct:.1f}% missing)")
            elif missing_pct > 0:
                if col in self.numeric_cols:
                    if strategy == "auto" or strategy == "conservative":
                        cleaned_data[col].fillna(cleaned_data[col].median(), inplace=True)
                    else:
                        cleaned_data[col].fillna(cleaned_data[col].mean(), inplace=True)
                    cleaning_report.append(f"✓ Filled missing values in '{col}' with median/mean")
                elif col in self.categorical_cols:
                    cleaned_data[col].fillna(cleaned_data[col].mode()[0] if not cleaned_data[col].mode().empty else "Unknown", inplace=True)
                    cleaning_report.append(f"✓ Filled missing values in '{col}' with mode")
        
        # 3. Handle outliers (for aggressive mode)
        if strategy == "aggressive":
            for col in self.numeric_cols:
                if col in cleaned_data.columns:
                    Q1 = cleaned_data[col].quantile(0.25)
                    Q3 = cleaned_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    initial_len = len(cleaned_data)
                    cleaned_data = cleaned_data[(cleaned_data[col] >= lower_bound) & (cleaned_data[col] <= upper_bound)]
                    removed_outliers = initial_len - len(cleaned_data)
                    
                    if removed_outliers > 0:
                        cleaning_report.append(f"✓ Removed {removed_outliers} outliers from '{col}'")
        
        # 4. Standardize text columns
        for col in self.categorical_cols:
            if col in cleaned_data.columns:
                cleaned_data[col] = cleaned_data[col].str.strip()
                cleaned_data[col] = cleaned_data[col].str.lower()
        
        # 5. Remove constant columns
        constant_cols = [col for col in cleaned_data.columns if cleaned_data[col].nunique() == 1]
        if constant_cols:
            cleaned_data = cleaned_data.drop(columns=constant_cols)
            cleaning_report.append(f"✓ Removed {len(constant_cols)} constant columns")
        
        print("CLEANING SUMMARY:")
        for report in cleaning_report:
            print(f"   {report}")
        
        print(f"\n   Original Shape: {self.data.shape}")
        print(f"   Cleaned Shape: {cleaned_data.shape}")
        print(f"   Data Retained: {(len(cleaned_data)/len(self.data))*100:.1f}%\n")
        
        self._log_operation("Cleaning", f"Applied {strategy} cleaning strategy")
        self.data = cleaned_data
        
        return cleaned_data
    
    def advanced_visualization_suite(self, output_dir: str = "visualizations"):
        """Generate comprehensive visualization suite."""
        print(f"\n{'='*80}")
        print("ADVANCED VISUALIZATION SUITE")
        print(f"{'='*80}\n")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        viz_count = 0
        
        # 1. Missing Data Heatmap
        if self.data.isna().sum().sum() > 0:
            plt.figure(figsize=(12, 6))
            sns.heatmap(self.data.isna(), cbar=True, cmap='viridis')
            plt.title('Missing Data Heatmap', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/missing_data_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
            viz_count += 1
            print(f"✓ Generated missing data heatmap")
        
        # 2. Correlation Matrix
        if len(self.numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            corr_matrix = self.data[self.numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8})
            plt.title('Correlation Matrix', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/correlation_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
            viz_count += 1
            print(f"✓ Generated correlation matrix")
        
        # 3. Distribution plots for numeric columns
        if self.numeric_cols:
            n_cols = min(len(self.numeric_cols), 6)
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for idx, col in enumerate(self.numeric_cols[:n_cols]):
                self.data[col].hist(bins=30, ax=axes[idx], edgecolor='black', alpha=0.7)
                axes[idx].set_title(f'Distribution: {col}', fontweight='bold')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
            
            for idx in range(n_cols, 6):
                fig.delaxes(axes[idx])
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/distributions.png", dpi=300, bbox_inches='tight')
            plt.close()
            viz_count += 1
            print(f"✓ Generated distribution plots")
        
        # 4. Box plots for outlier detection
        if self.numeric_cols:
            n_cols = min(len(self.numeric_cols), 6)
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for idx, col in enumerate(self.numeric_cols[:n_cols]):
                self.data.boxplot(column=col, ax=axes[idx])
                axes[idx].set_title(f'Boxplot: {col}', fontweight='bold')
                axes[idx].set_ylabel(col)
            
            for idx in range(n_cols, 6):
                fig.delaxes(axes[idx])
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/boxplots.png", dpi=300, bbox_inches='tight')
            plt.close()
            viz_count += 1
            print(f"✓ Generated boxplots")
        
        # 5. Categorical frequency plots
        if self.categorical_cols:
            n_cols = min(len(self.categorical_cols), 4)
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            for idx, col in enumerate(self.categorical_cols[:n_cols]):
                top_categories = self.data[col].value_counts().head(10)
                top_categories.plot(kind='bar', ax=axes[idx], color='steelblue', edgecolor='black')
                axes[idx].set_title(f'Top Categories: {col}', fontweight='bold')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Count')
                axes[idx].tick_params(axis='x', rotation=45)
            
            for idx in range(n_cols, 4):
                fig.delaxes(axes[idx])
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/categorical_frequencies.png", dpi=300, bbox_inches='tight')
            plt.close()
            viz_count += 1
            print(f"✓ Generated categorical frequency plots")
        
        print(f"\n✅ Successfully generated {viz_count} visualizations in '{output_dir}/' directory\n")
        self._log_operation("Visualization", f"Generated {viz_count} visualizations")
    
    def feature_engineering(self) -> pd.DataFrame:
        """Automatic feature engineering."""
        print(f"\n{'='*80}")
        print("AUTOMATIC FEATURE ENGINEERING")
        print(f"{'='*80}\n")
        
        engineered_data = self.data.copy()
        features_created = []
        
        # 1. Create interaction features between numeric columns
        if len(self.numeric_cols) >= 2:
            for i, col1 in enumerate(self.numeric_cols[:3]):
                for col2 in self.numeric_cols[i+1:4]:
                    new_col = f"{col1}_x_{col2}"
                    engineered_data[new_col] = engineered_data[col1] * engineered_data[col2]
                    features_created.append(new_col)
        
        # 2. Create polynomial features
        for col in self.numeric_cols[:3]:
            engineered_data[f"{col}_squared"] = engineered_data[col] ** 2
            features_created.append(f"{col}_squared")
        
        # 3. Create binned features
        for col in self.numeric_cols[:3]:
            engineered_data[f"{col}_binned"] = pd.cut(engineered_data[col], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            features_created.append(f"{col}_binned")
        
        # 4. Encode categorical variables
        le = LabelEncoder()
        for col in self.categorical_cols[:5]:
            try:
                engineered_data[f"{col}_encoded"] = le.fit_transform(engineered_data[col].astype(str))
                features_created.append(f"{col}_encoded")
            except:
                pass
        
        print(f"✓ Created {len(features_created)} new features")
        print(f"   Original columns: {len(self.data.columns)}")
        print(f"   New columns: {len(engineered_data.columns)}")
        print(f"\n   New Features: {', '.join(features_created[:10])}{'...' if len(features_created) > 10 else ''}\n")
        
        self._log_operation("Feature Engineering", f"Created {len(features_created)} new features")
        
        return engineered_data
    
    def statistical_tests(self, target_column: str = None) -> Dict:
        """Perform comprehensive statistical tests."""
        print(f"\n{'='*80}")
        print("STATISTICAL HYPOTHESIS TESTING")
        print(f"{'='*80}\n")
        
        results = {}
        
        # Normality tests for numeric columns
        print("📊 NORMALITY TESTS (Shapiro-Wilk)")
        for col in self.numeric_cols[:5]:
            if self.data[col].notna().sum() > 3:
                stat, p_value = stats.shapiro(self.data[col].dropna()[:5000])  # Limit sample size
                results[f"{col}_normality"] = {
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "is_normal": p_value > 0.05
                }
                print(f"   {col}: p-value = {p_value:.4f} {'(Normal)' if p_value > 0.05 else '(Not Normal)'}")
        
        print()
        
        # Chi-square tests for categorical variables
        if len(self.categorical_cols) >= 2:
            print("🔬 CHI-SQUARE INDEPENDENCE TESTS")
            for i, col1 in enumerate(self.categorical_cols[:3]):
                for col2 in self.categorical_cols[i+1:4]:
                    contingency_table = pd.crosstab(self.data[col1], self.data[col2])
                    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                    results[f"{col1}_vs_{col2}_independence"] = {
                        "chi2": float(chi2),
                        "p_value": float(p_value),
                        "dof": int(dof),
                        "independent": p_value > 0.05
                    }
                    print(f"   {col1} vs {col2}: p-value = {p_value:.4f} {'(Independent)' if p_value > 0.05 else '(Dependent)'}")
            print()
        
        self._log_operation("Statistical Tests", f"Performed {len(results)} statistical tests")
        
        return results
    
    def ml_preprocessing(self, target_column: str = None) -> Tuple[pd.DataFrame, Dict]:
        """Prepare data for machine learning."""
        print(f"\n{'='*80}")
        print("MACHINE LEARNING PREPROCESSING")
        print(f"{'='*80}\n")
        
        ml_data = self.data.copy()
        preprocessing_info = {}
        
        # 1. Handle missing values with KNN imputation
        if ml_data.isna().sum().sum() > 0:
            print("✓ Imputing missing values with KNN...")
            numeric_imputer = KNNImputer(n_neighbors=5)
            if self.numeric_cols:
                ml_data[self.numeric_cols] = numeric_imputer.fit_transform(ml_data[self.numeric_cols])
            
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            if self.categorical_cols:
                ml_data[self.categorical_cols] = categorical_imputer.fit_transform(ml_data[self.categorical_cols])
        
        # 2. Encode categorical variables
        print("✓ Encoding categorical variables...")
        le = LabelEncoder()
        for col in self.categorical_cols:
            ml_data[f"{col}_encoded"] = le.fit_transform(ml_data[col].astype(str))
            preprocessing_info[f"{col}_encoder"] = list(le.classes_)
        
        # 3. Scale numeric features
        print("✓ Scaling numeric features...")
        scaler = StandardScaler()
        numeric_cols_to_scale = [col for col in self.numeric_cols if col != target_column]
        if numeric_cols_to_scale:
            ml_data[numeric_cols_to_scale] = scaler.fit_transform(ml_data[numeric_cols_to_scale])
            preprocessing_info['scaler_mean'] = scaler.mean_.tolist()
            preprocessing_info['scaler_std'] = scaler.scale_.tolist()
        
        # 4. Remove high correlation features
        print("✓ Removing highly correlated features...")
        if len(self.numeric_cols) > 1:
            corr_matrix = ml_data[self.numeric_cols].corr().abs()
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
            ml_data = ml_data.drop(columns=to_drop)
            preprocessing_info['dropped_high_corr'] = to_drop
        
        print(f"\n✅ Preprocessing complete!")
        print(f"   Final shape: {ml_data.shape}")
        print(f"   Features ready for ML: {ml_data.shape[1]}\n")
        
        self._log_operation("ML Preprocessing", "Data prepared for machine learning")
        
        return ml_data, preprocessing_info
    
    def generate_insights(self) -> Dict:
        """Generate automated insights from the data."""
        print(f"\n{'='*80}")
        print("AUTOMATED DATA INSIGHTS")
        print(f"{'='*80}\n")
        
        insights = []
        
        # Missing data insights
        missing_pct = (self.data.isna().sum().sum() / (len(self.data) * len(self.data.columns))) * 100
        if missing_pct > 5:
            insights.append(f"⚠️  Dataset has {missing_pct:.1f}% missing values - consider imputation strategies")
        elif missing_pct == 0:
            insights.append(f"✅ Dataset is complete with no missing values")
        
        # Duplicate insights
        dup_pct = (self.data.duplicated().sum() / len(self.data)) * 100
        if dup_pct > 1:
            insights.append(f"⚠️  Found {dup_pct:.1f}% duplicate rows - consider removing")
        
        # Cardinality insights
        for col in self.categorical_cols[:5]:
            cardinality = self.data[col].nunique()
            if cardinality > len(self.data) * 0.5:
                insights.append(f"💡 Column '{col}' has high cardinality ({cardinality} unique values) - might be an ID")
            elif cardinality < 10:
                insights.append(f"📊 Column '{col}' is ordinal/nominal with {cardinality} categories")
        
        # Imbalance insights
        for col in self.categorical_cols[:3]:
            value_counts = self.data[col].value_counts()
            if len(value_counts) > 1:
                imbalance_ratio = value_counts.iloc[0] / value_counts.iloc[1]
                if imbalance_ratio > 10:
                    insights.append(f"⚖️  Column '{col}' is highly imbalanced - consider resampling")
        
        # Skewness insights
        for col in self.numeric_cols[:5]:
            skewness = self.data[col].skew()
            if abs(skewness) > 1:
                insights.append(f"📈 Column '{col}' is {'right' if skewness > 0 else 'left'} skewed - consider transformation")
        
        # Correlation insights
        if len(self.numeric_cols) > 1:
            corr_matrix = self.data[self.numeric_cols].corr()
            high_corrs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.8:
                        high_corrs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
            
            if high_corrs:
                for var1, var2, corr in high_corrs[:3]:
                    insights.append(f"🔗 Strong correlation between '{var1}' and '{var2}' (r={corr:.2f})")
        
        # Outlier insights
        for col in self.numeric_cols[:5]:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.data[col] < Q1 - 1.5 * IQR) | (self.data[col] > Q3 + 1.5 * IQR)).sum()
            outlier_pct = (outliers / len(self.data)) * 100
            if outlier_pct > 5:
                insights.append(f"🎯 Column '{col}' has {outlier_pct:.1f}% outliers - investigate anomalies")
        
        print("KEY INSIGHTS:")
        for idx, insight in enumerate(insights, 1):
            print(f"   {idx}. {insight}")
        
        print()
        self._log_operation("Insights", f"Generated {len(insights)} insights")
        
        return {"insights": insights, "count": len(insights)}
    
    def export_report(self, filename: str = "data_analysis_report.json"):
        """Export comprehensive analysis report."""
        report = {
            "metadata": {
                "dataset_name": self.name,
                "analysis_date": datetime.now().isoformat(),
                "session_id": self.session_id,
                "original_shape": self.original_data.shape,
                "current_shape": self.data.shape
            },
            "profiling": self.report.get("profiling", {}),
            "operations_log": self.operations_log
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Report exported to '{filename}'\n")
        self._log_operation("Export", f"Report saved to {filename}")
    
    def dimensionality_reduction(self, n_components: int = 2) -> pd.DataFrame:
        """Perform PCA for dimensionality reduction."""
        print(f"\n{'='*80}")
        print(f"DIMENSIONALITY REDUCTION (PCA)")
        print(f"{'='*80}\n")
        
        if len(self.numeric_cols) < 2:
            print("⚠️  Need at least 2 numeric columns for PCA\n")
            return self.data
        
        # Prepare data
        data_for_pca = self.data[self.numeric_cols].fillna(self.data[self.numeric_cols].mean())
        
        # Standardize
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_for_pca)
        
        # Apply PCA
        pca = PCA(n_components=min(n_components, len(self.numeric_cols)))
        principal_components = pca.fit_transform(scaled_data)
        
        # Create DataFrame
        pca_df = pd.DataFrame(
            data=principal_components,
            columns=[f'PC{i+1}' for i in range(principal_components.shape[1])]
        )
        
        print(f"✓ Reduced from {len(self.numeric_cols)} to {principal_components.shape[1]} dimensions")
        print(f"✓ Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"✓ Cumulative variance explained: {sum(pca.explained_variance_ratio_):.2%}\n")
        
        self._log_operation("PCA", f"Reduced to {n_components} components")
        
        return pca_df
    
    def detect_anomalies(self, contamination: float = 0.1) -> pd.DataFrame:
        """Detect anomalies using Isolation Forest."""
        print(f"\n{'='*80}")
        print("ANOMALY DETECTION (Isolation Forest)")
        print(f"{'='*80}\n")
        
        if not self.numeric_cols:
            print("⚠️  No numeric columns for anomaly detection\n")
            return self.data
        
        # Prepare data
        data_for_anomaly = self.data[self.numeric_cols].fillna(self.data[self.numeric_cols].mean())
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = iso_forest.fit_predict(data_for_anomaly)
        
        # Add anomaly column
        result_df = self.data.copy()
        result_df['anomaly'] = anomaly_labels
        result_df['anomaly_score'] = iso_forest.score_samples(data_for_anomaly)
        
        n_anomalies = (anomaly_labels == -1).sum()
        anomaly_pct = (n_anomalies / len(self.data)) * 100
        
        print(f"✓ Detected {n_anomalies} anomalies ({anomaly_pct:.2f}%)")
        print(f"✓ Normal samples: {(anomaly_labels == 1).sum()}")
        print(f"✓ Anomaly scores range: [{result_df['anomaly_score'].min():.3f}, {result_df['anomaly_score'].max():.3f}]\n")
        
        self._log_operation("Anomaly Detection", f"Found {n_anomalies} anomalies")
        
        return result_df
    
    def time_series_analysis(self, date_column: str, value_column: str):
        """Perform time series analysis if applicable."""
        print(f"\n{'='*80}")
        print("TIME SERIES ANALYSIS")
        print(f"{'='*80}\n")
        
        if date_column not in self.data.columns or value_column not in self.data.columns:
            print("⚠️  Specified columns not found\n")
            return
        
        # Convert to datetime
        ts_data = self.data.copy()
        ts_data[date_column] = pd.to_datetime(ts_data[date_column])
        ts_data = ts_data.sort_values(date_column)
        ts_data.set_index(date_column, inplace=True)
        
        # Basic statistics
        print(f"📅 Time Range: {ts_data.index.min()} to {ts_data.index.max()}")
        print(f"📊 Total Observations: {len(ts_data)}")
        
        # Trend
        if len(ts_data) > 1:
            trend = np.polyfit(range(len(ts_data)), ts_data[value_column].fillna(method='ffill'), 1)
            print(f"📈 Trend: {'Increasing' if trend[0] > 0 else 'Decreasing'} (slope: {trend[0]:.4f})")
        
        # Seasonality check (if enough data)
        if len(ts_data) > 30:
            print("✓ Sufficient data for seasonality analysis")
        
        print()
        self._log_operation("Time Series", f"Analyzed {value_column} over time")
    
    def full_pipeline(self, auto_clean: bool = True, visualize: bool = True, 
                     feature_engineer: bool = False, ml_prep: bool = False):
        """Run complete analysis pipeline."""
        print(f"\n{'='*80}")
        print("FULL ANALYSIS PIPELINE")
        print(f"{'='*80}\n")
        
        # Step 1: Profiling
        self.comprehensive_profiling()
        
        # Step 2: Generate Insights
        self.generate_insights()
        
        # Step 3: Auto-clean if requested
        if auto_clean:
            self.auto_clean_data(strategy="auto")
        
        # Step 4: Statistical tests
        self.statistical_tests()
        
        # Step 5: Visualizations
        if visualize:
            self.advanced_visualization_suite()
        
        # Step 6: Feature Engineering
        if feature_engineer:
            engineered_data = self.feature_engineering()
        
        # Step 7: ML Preprocessing
        if ml_prep:
            ml_data, prep_info = self.ml_preprocessing()
        
        # Step 8: Anomaly Detection
        anomaly_data = self.detect_anomalies()
        
        # Step 9: Export Report
        self.export_report()
        
        print(f"{'='*80}")
        print("✅ FULL PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}\n")
        
        return self.data


# =============================================================================
# DATA STRUCTURES & ALGORITHMS (DSA) MODULE
# =============================================================================

class DSAToolkit:
    """Comprehensive DSA toolkit with common algorithms and data structures."""
    
    def __init__(self):
        self.operations_count = 0
        
    # SORTING ALGORITHMS
    def quick_sort(self, arr: List) -> List:
        """Quick Sort - O(n log n) average case."""
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return self.quick_sort(left) + middle + self.quick_sort(right)
    
    def merge_sort(self, arr: List) -> List:
        """Merge Sort - O(n log n) guaranteed."""
        if len(arr) <= 1:
            return arr
        
        mid = len(arr) // 2
        left = self.merge_sort(arr[:mid])
        right = self.merge_sort(arr[mid:])
        
        return self._merge(left, right)
    
    def _merge(self, left: List, right: List) -> List:
        """Helper for merge sort."""
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        result.extend(left[i:])
        result.extend(right[j:])
        return result
    
    def heap_sort(self, arr: List) -> List:
        """Heap Sort - O(n log n)."""
        def heapify(arr, n, i):
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2
            
            if left < n and arr[left] > arr[largest]:
                largest = left
            if right < n and arr[right] > arr[largest]:
                largest = right
            
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                heapify(arr, n, largest)
        
        arr = arr.copy()
        n = len(arr)
        
        for i in range(n // 2 - 1, -1, -1):
            heapify(arr, n, i)
        
        for i in range(n - 1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            heapify(arr, i, 0)
        
        return arr
    
    # SEARCHING ALGORITHMS
    def binary_search(self, arr: List, target) -> int:
        """Binary Search - O(log n). Returns index or -1."""
        left, right = 0, len(arr) - 1
        
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return -1
    
    def linear_search(self, arr: List, target) -> int:
        """Linear Search - O(n). Returns index or -1."""
        for i, val in enumerate(arr):
            if val == target:
                return i
        return -1
    
    # GRAPH ALGORITHMS
    def dijkstra(self, graph: Dict, start: str) -> Dict:
        """Dijkstra's shortest path algorithm."""
        distances = {node: float('inf') for node in graph}
        distances[start] = 0
        visited = set()
        
        while len(visited) < len(graph):
            current = min((node for node in graph if node not in visited),
                         key=lambda x: distances[x])
            visited.add(current)
            
            for neighbor, weight in graph[current].items():
                distance = distances[current] + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
        
        return distances
    
    def bfs(self, graph: Dict, start: str) -> List:
        """Breadth-First Search."""
        visited = []
        queue = [start]
        
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.append(node)
                queue.extend([n for n in graph.get(node, []) if n not in visited])
        
        return visited
    
    def dfs(self, graph: Dict, start: str, visited: set = None) -> List:
        """Depth-First Search."""
        if visited is None:
            visited = set()
        
        visited.add(start)
        result = [start]
        
        for neighbor in graph.get(start, []):
            if neighbor not in visited:
                result.extend(self.dfs(graph, neighbor, visited))
        
        return result
    
    # DYNAMIC PROGRAMMING
    def fibonacci(self, n: int, memo: Dict = None) -> int:
        """Fibonacci with memoization - O(n)."""
        if memo is None:
            memo = {}
        
        if n in memo:
            return memo[n]
        
        if n <= 1:
            return n
        
        memo[n] = self.fibonacci(n-1, memo) + self.fibonacci(n-2, memo)
        return memo[n]
    
    def knapsack(self, weights: List[int], values: List[int], capacity: int) -> int:
        """0/1 Knapsack problem - O(nW)."""
        n = len(weights)
        dp = [[0] * (capacity + 1) for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            for w in range(1, capacity + 1):
                if weights[i-1] <= w:
                    dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w])
                else:
                    dp[i][w] = dp[i-1][w]
        
        return dp[n][capacity]
    
    def longest_common_subsequence(self, str1: str, str2: str) -> int:
        """LCS - O(mn)."""
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    # UTILITY FUNCTIONS
    def benchmark_sorting(self, arr: List) -> Dict:
        """Benchmark different sorting algorithms."""
        import time
        
        results = {}
        algorithms = [
            ('Quick Sort', self.quick_sort),
            ('Merge Sort', self.merge_sort),
            ('Heap Sort', self.heap_sort),
            ('Python Built-in', sorted)
        ]
        
        for name, func in algorithms:
            test_arr = arr.copy()
            start = time.time()
            sorted_arr = func(test_arr)
            end = time.time()
            results[name] = {
                'time': end - start,
                'correct': sorted_arr == sorted(arr)
            }
        
        return results


# =============================================================================
# MAIN EXECUTION EXAMPLE
# =============================================================================

def main():
    """Example usage of the Advanced Data Analysis Tool."""
    
    print("\n" + "="*80)
    print("ADVANCED DATA ANALYSIS & DSA TOOL")
    print("Production-Ready Comprehensive Solution")
    print("="*80 + "\n")
    
    # Example 1: Create sample dataset
    print("Creating sample dataset for demonstration...\n")
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'age': np.random.randint(18, 80, 1000),
        'income': np.random.normal(50000, 20000, 1000),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 1000),
        'experience_years': np.random.randint(0, 40, 1000),
        'satisfaction_score': np.random.uniform(1, 10, 1000),
        'department': np.random.choice(['IT', 'HR', 'Sales', 'Marketing', 'Finance'], 1000),
        'performance_rating': np.random.uniform(1, 5, 1000)
    })
    
    # Add some missing values
    sample_data.loc[np.random.choice(sample_data.index, 50), 'income'] = np.nan
    sample_data.loc[np.random.choice(sample_data.index, 30), 'satisfaction_score'] = np.nan
    
    # Initialize analyzer
    analyzer = AdvancedDataAnalyzer(sample_data, name="Employee Data")
    
    # Run full pipeline
    analyzer.full_pipeline(
        auto_clean=True,
        visualize=True,
        feature_engineer=True,
        ml_prep=True
    )
    
    # Example 2: DSA Toolkit
    print("\n" + "="*80)
    print("DSA TOOLKIT DEMONSTRATION")
    print("="*80 + "\n")
    
    dsa = DSAToolkit()
    
    # Sorting demonstration
    test_array = [64, 34, 25, 12, 22, 11, 90, 88, 45, 50, 23, 36, 18, 77]
    print(f"Original array: {test_array}\n")
    
    print("Sorting Algorithms:")
    print(f"Quick Sort: {dsa.quick_sort(test_array)}")
    print(f"Merge Sort: {dsa.merge_sort(test_array)}")
    print(f"Heap Sort: {dsa.heap_sort(test_array)}\n")
    
    # Searching demonstration
    sorted_arr = sorted(test_array)
    target = 45
    print(f"Searching for {target} in sorted array:")
    print(f"Binary Search Index: {dsa.binary_search(sorted_arr, target)}\n")
    
    # Graph algorithms
    graph = {
        'A': {'B': 1, 'C': 4},
        'B': {'A': 1, 'C': 2, 'D': 5},
        'C': {'A': 4, 'B': 2, 'D': 1},
        'D': {'B': 5, 'C': 1}
    }
    print(f"Graph shortest paths from 'A': {dsa.dijkstra(graph, 'A')}\n")
    print(f"BFS traversal from 'A': {dsa.bfs(graph, 'A')}\n")
    
    # Dynamic Programming
    print(f"Fibonacci(10): {dsa.fibonacci(10)}")
    print(f"LCS of 'ABCDGH' and 'AEDFHR': {dsa.longest_common_subsequence('ABCDGH', 'AEDFHR')}\n")
    
    print("="*80)
    print("✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()