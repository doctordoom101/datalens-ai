import pandas as pd
import numpy as np
from typing import Dict, Any, List

class DataProfiler:
    """
    Automatic Data Profiler for CSV Datasets.
    Generates dataset-level metrics, column-level profiling, and data quality assessments.
    """

    def __init__(self, df: pd.DataFrame, dataset_name: str = "Uploaded Dataset"):
        self.df = df
        self.dataset_name = dataset_name

    def get_dataset_summary(self) -> Dict[str, Any]:
        """Calculates dataset-level summary metrics."""
        total_rows = int(len(self.df))
        total_cols = int(len(self.df.columns))
        duplicate_rows = int(self.df.duplicated().sum())
        total_cells = total_rows * total_cols
        total_missing = int(self.df.isna().sum().sum())
        missing_pct = round((total_missing / total_cells * 100), 2) if total_cells > 0 else 0.0
        
        # Memory usage
        memory_bytes = self.df.memory_usage(deep=True).sum()
        if memory_bytes < 1024 * 1024:
            memory_str = f"{round(memory_bytes / 1024, 2)} KB"
        else:
            memory_str = f"{round(memory_bytes / (1024 * 1024), 2)} MB"

        # Quality health score calculation (0 to 100)
        dup_penalty = min(20.0, (duplicate_rows / max(total_rows, 1)) * 100 * 2)
        missing_penalty = min(40.0, missing_pct * 1.5)
        health_score = max(0.0, min(100.0, round(100.0 - dup_penalty - missing_penalty, 1)))

        return {
            "dataset_name": self.dataset_name,
            "total_rows": total_rows,
            "total_columns": total_cols,
            "duplicate_rows": duplicate_rows,
            "total_missing_cells": total_missing,
            "missing_percentage": missing_pct,
            "memory_usage": memory_str,
            "health_score": health_score
        }

    def get_column_profiles(self) -> List[Dict[str, Any]]:
        """Generates detailed column-level metrics."""
        profiles = []
        total_rows = len(self.df)

        for col in self.df.columns:
            series = self.df[col]
            missing_count = int(series.isna().sum())
            missing_pct = round((missing_count / total_rows * 100), 2) if total_rows > 0 else 0.0
            unique_count = int(series.nunique(dropna=True))
            dtype = str(series.dtype)

            # Sample non-null values
            non_null_samples = series.dropna().head(5).astype(str).tolist()
            sample_str = ", ".join(non_null_samples[:3]) if non_null_samples else "None"

            profile = {
                "column_name": col,
                "data_type": dtype,
                "missing_count": missing_count,
                "missing_pct": missing_pct,
                "unique_count": unique_count,
                "sample_values": sample_str,
                "is_numeric": pd.api.types.is_numeric_dtype(series),
                "is_categorical": pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series),
                "stats": {}
            }

            # Statistical properties
            if pd.api.types.is_numeric_dtype(series) and not series.dropna().empty:
                profile["stats"] = {
                    "min": float(series.min()) if not np.isnan(series.min()) else None,
                    "max": float(series.max()) if not np.isnan(series.max()) else None,
                    "mean": float(round(series.mean(), 4)) if not np.isnan(series.mean()) else None,
                    "median": float(round(series.median(), 4)) if not np.isnan(series.median()) else None,
                    "std": float(round(series.std(), 4)) if len(series.dropna()) > 1 and not np.isnan(series.std()) else 0.0
                }
            elif (pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)) and not series.dropna().empty:
                top_counts = series.value_counts().head(3).to_dict()
                top_formatted = {str(k): int(v) for k, v in top_counts.items()}
                profile["stats"] = {
                    "top_values": top_formatted
                }

            profiles.append(profile)

        return profiles

    def get_full_profile(self) -> Dict[str, Any]:
        """Returns consolidated dataset and column profiles."""
        return {
            "summary": self.get_dataset_summary(),
            "columns": self.get_column_profiles()
        }
