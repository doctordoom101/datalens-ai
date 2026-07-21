from typing import Dict, Any, List
from langchain_core.documents import Document

class KnowledgeBaseGenerator:
    """
    Converts dataset profile metadata into structured LangChain Documents
    optimized for RAG retrieval and precise QA citations.
    """

    def __init__(self, profile_data: Dict[str, Any]):
        self.summary = profile_data["summary"]
        self.columns = profile_data["columns"]
        self.dataset_name = self.summary["dataset_name"]

    def generate_documents(self) -> List[Document]:
        """Builds a list of rich, chunked markdown documents for RAG vector index."""
        docs = []

        # 1. Dataset Overview & Health Report Document
        overview_content = f"""# Dataset Overview & Health Report
Dataset Name: {self.dataset_name}
Total Rows: {self.summary['total_rows']:,}
Total Columns: {self.summary['total_columns']}
Memory Usage: {self.summary['memory_usage']}
Duplicate Rows: {self.summary['duplicate_rows']}
Total Missing Cells: {self.summary['total_missing_cells']} ({self.summary['missing_percentage']}%)
Data Quality Health Score: {self.summary['health_score']} / 100

Summary Assessment:
The dataset '{self.dataset_name}' consists of {self.summary['total_rows']:,} records across {self.summary['total_columns']} columns.
It contains {self.summary['duplicate_rows']} duplicate row(s) and {self.summary['total_missing_cells']} missing values across all cells ({self.summary['missing_percentage']}% missing rate).
"""
        docs.append(Document(
            page_content=overview_content,
            metadata={
                "source": "Dataset Overview",
                "category": "overview",
                "dataset_name": self.dataset_name
            }
        ))

        # 2. Schema and Column Types Document
        col_list_str = []
        for c in self.columns:
            col_list_str.append(f"- Column: {c['column_name']} | Type: {c['data_type']} | Missing: {c['missing_count']} ({c['missing_pct']}%) | Unique Values: {c['unique_count']}")
        
        schema_content = f"""# Dataset Schema & Column List
Dataset: {self.dataset_name}
Columns Breakdown ({len(self.columns)} columns):

{chr(10).join(col_list_str)}
"""
        docs.append(Document(
            page_content=schema_content,
            metadata={
                "source": "Dataset Schema & Data Types",
                "category": "schema",
                "dataset_name": self.dataset_name
            }
        ))

        # 3. Data Quality & Missing Values Document
        missing_cols = [c for c in self.columns if c['missing_count'] > 0]
        if missing_cols:
            missing_lines = [f"- Column '{c['column_name']}': {c['missing_count']} missing values ({c['missing_pct']}%)" for c in missing_cols]
            quality_content = f"""# Data Quality & Missing Values Report
Dataset: {self.dataset_name}
Duplicate Rows: {self.summary['duplicate_rows']}

Columns containing missing values:
{chr(10).join(missing_lines)}
"""
        else:
            quality_content = f"""# Data Quality & Missing Values Report
Dataset: {self.dataset_name}
Duplicate Rows: {self.summary['duplicate_rows']}
Missing Values Status: Perfect! No missing values detected in any column of this dataset.
"""
        docs.append(Document(
            page_content=quality_content,
            metadata={
                "source": "Data Quality Report",
                "category": "quality",
                "dataset_name": self.dataset_name
            }
        ))

        # 4. Individual Column Detailed Documents
        for col in self.columns:
            stats = col.get("stats", {})
            stat_lines = []
            if col["is_numeric"] and stats:
                stat_lines.append(f"Minimum Value: {stats.get('min')}")
                stat_lines.append(f"Maximum Value: {stats.get('max')}")
                stat_lines.append(f"Mean (Average): {stats.get('mean')}")
                stat_lines.append(f"Median: {stats.get('median')}")
                stat_lines.append(f"Standard Deviation: {stats.get('std')}")
            elif col["is_categorical"] and stats and "top_values" in stats:
                top_str = ", ".join([f"'{k}': {v} times" for k, v in stats["top_values"].items()])
                stat_lines.append(f"Top Frequent Values: {top_str}")

            col_doc_content = f"""# Column Details: {col['column_name']}
Dataset: {self.dataset_name}
Column Name: {col['column_name']}
Data Type: {col['data_type']}
Missing Values: {col['missing_count']} ({col['missing_pct']}%)
Unique Values Count: {col['unique_count']}
Sample Values: {col['sample_values']}
{chr(10).join(stat_lines)}
"""
            docs.append(Document(
                page_content=col_doc_content,
                metadata={
                    "source": f"Column: {col['column_name']}",
                    "column_name": col['column_name'],
                    "category": "column_detail",
                    "dataset_name": self.dataset_name
                }
            ))

        # 5. Machine Learning & Predictive Analytics Suitability Document
        numeric_cols = [c['column_name'] for c in self.columns if c['is_numeric']]
        cat_cols = [c['column_name'] for c in self.columns if c['is_categorical']]
        
        ml_content = f"""# Machine Learning & Analytics Suitability
Dataset: {self.dataset_name}
Total Records: {self.summary['total_rows']}
Numerical Features Available ({len(numeric_cols)}): {', '.join(numeric_cols) if numeric_cols else 'None'}
Categorical Features Available ({len(cat_cols)}): {', '.join(cat_cols) if cat_cols else 'None'}

Potential Analytics & Modeling Scenarios:
- Regression Targets (Numerical): {', '.join([c for c in numeric_cols if 'sales' in c.lower() or 'profit' in c.lower() or 'price' in c.lower() or 'amount' in c.lower() or 'target' in c.lower()]) or ', '.join(numeric_cols[:3])}
- Classification Targets (Categorical): {', '.join([c for c in cat_cols if 'segment' in c.lower() or 'category' in c.lower() or 'status' in c.lower() or 'type' in c.lower()]) or ', '.join(cat_cols[:3])}
- Data Readiness Assessment: Total rows count is {self.summary['total_rows']}. Data contains {self.summary['duplicate_rows']} duplicates and {self.summary['missing_percentage']}% missing values.
"""
        docs.append(Document(
            page_content=ml_content,
            metadata={
                "source": "ML & Analytics Suitability",
                "category": "ml_suitability",
                "dataset_name": self.dataset_name
            }
        ))

        return docs
