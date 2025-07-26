"""
Data Loading and Processing Module

This module handles loading and processing of tabular data from various formats
including CSV, Excel, and other structured data sources.
"""

import pandas as pd
import sqlite3
import tempfile
import os
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading and processing of tabular data from various sources.
    """
    
    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.xls', '.json']
        self.data_cache = {}
        
    def load_file(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from a file into a pandas DataFrame.
        
        Args:
            file_path (str): Path to the data file
            **kwargs: Additional arguments for pandas read functions
            
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_extension = file_path.suffix.lower()
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
            
        try:
            if file_extension == '.csv':
                df = pd.read_csv(file_path, **kwargs)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, **kwargs)
            elif file_extension == '.json':
                df = pd.read_json(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported format: {file_extension}")
                
            logger.info(f"Successfully loaded {len(df)} rows from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            raise
    
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive information about the dataset.
        
        Args:
            df (pd.DataFrame): The dataset
            
        Returns:
            Dict[str, Any]: Dataset information
        """
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'sample_data': df.head().to_dict('records')
        }
        
        # Add statistical summary for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            info['numeric_summary'] = df[numeric_cols].describe().to_dict()
            
        # Add categorical information
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            info['categorical_info'] = {}
            for col in categorical_cols:
                unique_values = df[col].nunique()
                info['categorical_info'][col] = {
                    'unique_count': unique_values,
                    'top_values': df[col].value_counts().head().to_dict() if unique_values <= 100 else {}
                }
        
        return info
    
    def create_sqlite_db(self, df: pd.DataFrame, table_name: str = 'data') -> str:
        """
        Create a temporary SQLite database from the DataFrame.
        
        Args:
            df (pd.DataFrame): The dataset
            table_name (str): Name for the table in the database
            
        Returns:
            str: Path to the temporary SQLite database
        """
        # Create a temporary database file
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        
        # Connect to the database and insert data
        conn = sqlite3.connect(temp_db.name)
        df.to_sql(table_name, conn, index=False, if_exists='replace')
        conn.close()
        
        logger.info(f"Created SQLite database at {temp_db.name} with table '{table_name}'")
        return temp_db.name
    
    def get_schema_info(self, df: pd.DataFrame, table_name: str = 'data') -> str:
        """
        Generate schema information for the LLM agent.
        
        Args:
            df (pd.DataFrame): The dataset
            table_name (str): Name of the table
            
        Returns:
            str: Schema information formatted for LLM consumption
        """
        schema_info = f"Table: {table_name}\n"
        schema_info += f"Total rows: {len(df)}\n"
        schema_info += f"Total columns: {len(df.columns)}\n\n"
        
        schema_info += "Columns:\n"
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            
            schema_info += f"- {col} ({dtype}): {null_count} nulls ({null_pct:.1f}%)\n"
            
            # Add sample values for better context
            if df[col].dtype == 'object':
                unique_vals = df[col].dropna().unique()[:5]
                schema_info += f"  Sample values: {list(unique_vals)}\n"
            elif df[col].dtype in ['int64', 'float64']:
                min_val = df[col].min()
                max_val = df[col].max()
                schema_info += f"  Range: {min_val} to {max_val}\n"
        
        return schema_info
    
    def validate_data(self, df: pd.DataFrame) -> List[str]:
        """
        Validate the loaded data and return any warnings or issues.
        
        Args:
            df (pd.DataFrame): The dataset to validate
            
        Returns:
            List[str]: List of validation warnings/issues
        """
        warnings = []
        
        # Check for empty dataset
        if df.empty:
            warnings.append("Dataset is empty")
            return warnings
        
        # Check for duplicate column names
        if len(df.columns) != len(set(df.columns)):
            warnings.append("Dataset contains duplicate column names")
        
        # Check for columns with all null values
        all_null_cols = df.columns[df.isnull().all()].tolist()
        if all_null_cols:
            warnings.append(f"Columns with all null values: {all_null_cols}")
        
        # Check for very high null percentage
        high_null_cols = []
        for col in df.columns:
            null_pct = (df[col].isnull().sum() / len(df)) * 100
            if null_pct > 80:
                high_null_cols.append(f"{col} ({null_pct:.1f}%)")
        
        if high_null_cols:
            warnings.append(f"Columns with >80% null values: {high_null_cols}")
        
        # Check for very large dataset
        if len(df) > 100000:
            warnings.append(f"Large dataset ({len(df)} rows) - queries may be slow")
        
        return warnings


class DataProcessor:
    """
    Handles data processing and transformation operations.
    """
    
    def __init__(self):
        pass
    
    def clean_data(self, df: pd.DataFrame, operations: List[str] = None) -> pd.DataFrame:
        """
        Apply basic data cleaning operations.
        
        Args:
            df (pd.DataFrame): Input dataset
            operations (List[str]): List of cleaning operations to apply
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        if operations is None:
            operations = ['remove_duplicates', 'strip_whitespace']
        
        df_cleaned = df.copy()
        
        if 'remove_duplicates' in operations:
            initial_rows = len(df_cleaned)
            df_cleaned = df_cleaned.drop_duplicates()
            removed_rows = initial_rows - len(df_cleaned)
            if removed_rows > 0:
                logger.info(f"Removed {removed_rows} duplicate rows")
        
        if 'strip_whitespace' in operations:
            # Strip whitespace from string columns
            string_cols = df_cleaned.select_dtypes(include=['object']).columns
            for col in string_cols:
                df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
        
        return df_cleaned
    
    def infer_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attempt to infer and convert appropriate data types.
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with inferred types
        """
        df_typed = df.copy()
        
        for col in df_typed.columns:
            # Try to convert to numeric
            if df_typed[col].dtype == 'object':
                # Try to convert to datetime first
                try:
                    pd.to_datetime(df_typed[col], errors='raise')
                    df_typed[col] = pd.to_datetime(df_typed[col], errors='coerce')
                    logger.info(f"Converted column '{col}' to datetime")
                    continue
                except:
                    pass
                
                # Try to convert to numeric
                try:
                    numeric_series = pd.to_numeric(df_typed[col], errors='coerce')
                    if not numeric_series.isna().all():
                        df_typed[col] = numeric_series
                        logger.info(f"Converted column '{col}' to numeric")
                except:
                    pass
        
        return df_typed


# Example usage and testing
if __name__ == "__main__":
    # Create a sample dataset for testing
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'salary': [50000, 60000, 70000, 55000, 65000],
        'department': ['Engineering', 'Marketing', 'Engineering', 'Sales', 'Marketing']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Test the DataLoader
    loader = DataLoader()
    
    # Save sample data to CSV for testing
    df.to_csv('/tmp/sample_data.csv', index=False)
    
    # Load the data
    loaded_df = loader.load_file('/tmp/sample_data.csv')
    print("Loaded data:")
    print(loaded_df)
    
    # Get data info
    info = loader.get_data_info(loaded_df)
    print("\nData info:")
    print(info)
    
    # Get schema info
    schema = loader.get_schema_info(loaded_df)
    print("\nSchema info:")
    print(schema)
    
    # Create SQLite database
    db_path = loader.create_sqlite_db(loaded_df)
    print(f"\nCreated database at: {db_path}")
    
    # Clean up
    os.unlink('/tmp/sample_data.csv')
    os.unlink(db_path)

