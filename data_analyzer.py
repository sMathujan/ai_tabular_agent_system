"""
Data Analysis and Visualization Module

This module provides data analysis capabilities and generates visualizations
based on query results and user requests.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
import numpy as np
from datetime import datetime
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAnalyzer:
    """
    Analyzes data and generates insights from query results.
    """
    
    def __init__(self):
        self.analysis_cache = {}
        
    def analyze_query_result(self, data: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """
        Analyze query results and provide insights.
        
        Args:
            data (List[Dict[str, Any]]): Query result data
            query (str): Original natural language query
            
        Returns:
            Dict[str, Any]: Analysis results with insights
        """
        if not data:
            return {
                "insights": ["No data returned from the query"],
                "summary": "The query returned no results.",
                "data_type": "empty",
                "visualization_suggestions": []
            }
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(data)
        
        # Basic analysis
        analysis = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "data_types": df.dtypes.to_dict(),
            "insights": [],
            "summary": "",
            "data_type": self._classify_data_type(df),
            "visualization_suggestions": []
        }
        
        # Generate insights
        insights = self._generate_insights(df, query)
        analysis["insights"] = insights
        
        # Generate summary
        analysis["summary"] = self._generate_summary(df, query, insights)
        
        # Suggest visualizations
        analysis["visualization_suggestions"] = self._suggest_visualizations(df, query)
        
        return analysis
    
    def _classify_data_type(self, df: pd.DataFrame) -> str:
        """
        Classify the type of data returned.
        
        Args:
            df (pd.DataFrame): Data to classify
            
        Returns:
            str: Data type classification
        """
        if len(df) == 1 and len(df.columns) == 1:
            return "single_value"
        elif len(df) == 1:
            return "single_row"
        elif len(df.columns) == 1:
            return "single_column"
        elif len(df.columns) == 2:
            return "two_column"
        else:
            return "multi_column"
    
    def _generate_insights(self, df: pd.DataFrame, query: str) -> List[str]:
        """
        Generate insights from the data.
        
        Args:
            df (pd.DataFrame): Data to analyze
            query (str): Original query for context
            
        Returns:
            List[str]: List of insights
        """
        insights = []
        
        # Basic statistics
        if len(df) == 1:
            insights.append(f"Query returned a single result")
        else:
            insights.append(f"Query returned {len(df)} rows")
        
        # Analyze numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if len(df) > 1:
                mean_val = df[col].mean()
                median_val = df[col].median()
                std_val = df[col].std()
                
                insights.append(f"{col}: mean={mean_val:.2f}, median={median_val:.2f}")
                
                if std_val > mean_val * 0.5:
                    insights.append(f"{col} shows high variability (std={std_val:.2f})")
        
        # Analyze categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            unique_count = df[col].nunique()
            if unique_count < len(df):
                most_common = df[col].value_counts().iloc[0]
                most_common_value = df[col].value_counts().index[0]
                insights.append(f"{col}: {unique_count} unique values, most common is '{most_common_value}' ({most_common} times)")
        
        # Look for patterns based on query keywords
        query_lower = query.lower()
        if any(word in query_lower for word in ['trend', 'over time', 'by date', 'monthly', 'yearly']):
            date_cols = df.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                insights.append("Time-based analysis detected - consider trend visualization")
        
        if any(word in query_lower for word in ['compare', 'comparison', 'vs', 'versus']):
            insights.append("Comparison analysis detected - consider comparative visualization")
        
        if any(word in query_lower for word in ['distribution', 'spread', 'range']):
            insights.append("Distribution analysis detected - consider histogram or box plot")
        
        return insights
    
    def _generate_summary(self, df: pd.DataFrame, query: str, insights: List[str]) -> str:
        """
        Generate a natural language summary of the results.
        
        Args:
            df (pd.DataFrame): Data to summarize
            query (str): Original query
            insights (List[str]): Generated insights
            
        Returns:
            str: Natural language summary
        """
        if len(df) == 0:
            return "No data was found matching your query."
        
        summary_parts = []
        
        # Basic result description
        if len(df) == 1 and len(df.columns) == 1:
            value = df.iloc[0, 0]
            summary_parts.append(f"The result is: {value}")
        elif len(df) == 1:
            summary_parts.append(f"Found one record with {len(df.columns)} fields")
        else:
            summary_parts.append(f"Found {len(df)} records")
        
        # Add key insights
        if len(insights) > 0:
            summary_parts.append("Key insights:")
            summary_parts.extend(insights[:3])  # Top 3 insights
        
        return ". ".join(summary_parts) + "."
    
    def _suggest_visualizations(self, df: pd.DataFrame, query: str) -> List[Dict[str, str]]:
        """
        Suggest appropriate visualizations for the data.
        
        Args:
            df (pd.DataFrame): Data to visualize
            query (str): Original query for context
            
        Returns:
            List[Dict[str, str]]: List of visualization suggestions
        """
        suggestions = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Single value - no visualization needed
        if len(df) == 1 and len(df.columns) == 1:
            return suggestions
        
        # Single column analysis
        if len(df.columns) == 1:
            col = df.columns[0]
            if col in numeric_cols:
                suggestions.append({
                    "type": "histogram",
                    "description": f"Distribution of {col}",
                    "columns": [col]
                })
            else:
                suggestions.append({
                    "type": "bar",
                    "description": f"Count of {col} values",
                    "columns": [col]
                })
        
        # Two column analysis
        elif len(df.columns) == 2:
            col1, col2 = df.columns[0], df.columns[1]
            
            if col1 in numeric_cols and col2 in numeric_cols:
                suggestions.append({
                    "type": "scatter",
                    "description": f"Relationship between {col1} and {col2}",
                    "columns": [col1, col2]
                })
            elif col1 in categorical_cols and col2 in numeric_cols:
                suggestions.append({
                    "type": "bar",
                    "description": f"{col2} by {col1}",
                    "columns": [col1, col2]
                })
            elif col1 in numeric_cols and col2 in categorical_cols:
                suggestions.append({
                    "type": "bar",
                    "description": f"{col1} by {col2}",
                    "columns": [col2, col1]
                })
        
        # Multi-column analysis
        else:
            if len(numeric_cols) >= 2:
                suggestions.append({
                    "type": "correlation_heatmap",
                    "description": "Correlation between numeric variables",
                    "columns": list(numeric_cols)
                })
            
            if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
                suggestions.append({
                    "type": "grouped_bar",
                    "description": f"Numeric values grouped by categories",
                    "columns": list(categorical_cols) + list(numeric_cols)
                })
        
        # Query-specific suggestions
        query_lower = query.lower()
        if 'time' in query_lower or 'date' in query_lower:
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols and len(numeric_cols) > 0:
                suggestions.append({
                    "type": "line",
                    "description": "Time series trend",
                    "columns": date_cols + list(numeric_cols)
                })
        
        return suggestions


class DataVisualizer:
    """
    Creates visualizations from data using Plotly.
    """
    
    def __init__(self):
        # Set default theme
        pio.templates.default = "plotly_white"
        
    def create_visualization(self, df: pd.DataFrame, viz_type: str, 
                           columns: List[str] = None, title: str = None) -> Dict[str, Any]:
        """
        Create a visualization from the data.
        
        Args:
            df (pd.DataFrame): Data to visualize
            viz_type (str): Type of visualization
            columns (List[str]): Columns to use for visualization
            title (str): Title for the visualization
            
        Returns:
            Dict[str, Any]: Visualization result with figure and metadata
        """
        try:
            if columns is None:
                columns = list(df.columns)
            
            # Create the appropriate visualization
            if viz_type == "bar":
                fig = self._create_bar_chart(df, columns, title)
            elif viz_type == "line":
                fig = self._create_line_chart(df, columns, title)
            elif viz_type == "scatter":
                fig = self._create_scatter_plot(df, columns, title)
            elif viz_type == "histogram":
                fig = self._create_histogram(df, columns, title)
            elif viz_type == "box":
                fig = self._create_box_plot(df, columns, title)
            elif viz_type == "correlation_heatmap":
                fig = self._create_correlation_heatmap(df, columns, title)
            elif viz_type == "pie":
                fig = self._create_pie_chart(df, columns, title)
            else:
                raise ValueError(f"Unsupported visualization type: {viz_type}")
            
            # Convert to JSON for web display
            fig_json = fig.to_json()
            
            return {
                "success": True,
                "figure": fig,
                "figure_json": fig_json,
                "type": viz_type,
                "columns_used": columns,
                "title": title or f"{viz_type.title()} Chart"
            }
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "type": viz_type,
                "columns_used": columns
            }
    
    def _create_bar_chart(self, df: pd.DataFrame, columns: List[str], title: str) -> go.Figure:
        """Create a bar chart."""
        if len(columns) >= 2:
            x_col, y_col = columns[0], columns[1]
            fig = px.bar(df, x=x_col, y=y_col, title=title or f"{y_col} by {x_col}")
        else:
            # Single column - create value counts
            col = columns[0]
            value_counts = df[col].value_counts()
            fig = px.bar(x=value_counts.index, y=value_counts.values, 
                        title=title or f"Distribution of {col}")
            fig.update_xaxis(title=col)
            fig.update_yaxis(title="Count")
        
        return fig
    
    def _create_line_chart(self, df: pd.DataFrame, columns: List[str], title: str) -> go.Figure:
        """Create a line chart."""
        if len(columns) >= 2:
            x_col, y_col = columns[0], columns[1]
            fig = px.line(df, x=x_col, y=y_col, title=title or f"{y_col} over {x_col}")
        else:
            # Single column - plot against index
            col = columns[0]
            fig = px.line(df, y=col, title=title or f"{col} Trend")
        
        return fig
    
    def _create_scatter_plot(self, df: pd.DataFrame, columns: List[str], title: str) -> go.Figure:
        """Create a scatter plot."""
        if len(columns) >= 2:
            x_col, y_col = columns[0], columns[1]
            color_col = columns[2] if len(columns) > 2 else None
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                           title=title or f"{y_col} vs {x_col}")
        else:
            raise ValueError("Scatter plot requires at least 2 columns")
        
        return fig
    
    def _create_histogram(self, df: pd.DataFrame, columns: List[str], title: str) -> go.Figure:
        """Create a histogram."""
        col = columns[0]
        fig = px.histogram(df, x=col, title=title or f"Distribution of {col}")
        return fig
    
    def _create_box_plot(self, df: pd.DataFrame, columns: List[str], title: str) -> go.Figure:
        """Create a box plot."""
        if len(columns) >= 2:
            x_col, y_col = columns[0], columns[1]
            fig = px.box(df, x=x_col, y=y_col, title=title or f"{y_col} by {x_col}")
        else:
            col = columns[0]
            fig = px.box(df, y=col, title=title or f"Distribution of {col}")
        
        return fig
    
    def _create_correlation_heatmap(self, df: pd.DataFrame, columns: List[str], title: str) -> go.Figure:
        """Create a correlation heatmap."""
        # Select only numeric columns
        numeric_df = df[columns].select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            raise ValueError("Correlation heatmap requires at least 2 numeric columns")
        
        corr_matrix = numeric_df.corr()
        
        fig = px.imshow(corr_matrix, 
                       title=title or "Correlation Heatmap",
                       color_continuous_scale="RdBu_r",
                       aspect="auto")
        
        return fig
    
    def _create_pie_chart(self, df: pd.DataFrame, columns: List[str], title: str) -> go.Figure:
        """Create a pie chart."""
        col = columns[0]
        value_counts = df[col].value_counts()
        
        fig = px.pie(values=value_counts.values, names=value_counts.index,
                    title=title or f"Distribution of {col}")
        
        return fig
    
    def save_visualization(self, fig: go.Figure, filename: str, format: str = "html") -> str:
        """
        Save visualization to file.
        
        Args:
            fig (go.Figure): Plotly figure to save
            filename (str): Output filename
            format (str): Output format (html, png, pdf, etc.)
            
        Returns:
            str: Path to saved file
        """
        try:
            if format.lower() == "html":
                fig.write_html(filename)
            elif format.lower() == "png":
                fig.write_image(filename)
            elif format.lower() == "pdf":
                fig.write_image(filename)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Visualization saved to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving visualization: {str(e)}")
            raise


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    sample_data = {
        'department': ['Engineering', 'Marketing', 'Engineering', 'Sales', 'Marketing', 'Engineering'],
        'salary': [70000, 55000, 75000, 50000, 60000, 80000],
        'age': [28, 32, 35, 25, 30, 40],
        'experience': [3, 5, 8, 2, 4, 12]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Test analyzer
    analyzer = DataAnalyzer()
    data_list = df.to_dict('records')
    
    analysis = analyzer.analyze_query_result(data_list, "Show me salary by department")
    print("Analysis results:")
    print(json.dumps(analysis, indent=2, default=str))
    
    # Test visualizer
    visualizer = DataVisualizer()
    
    # Test different visualizations
    viz_tests = [
        ("bar", ["department", "salary"], "Salary by Department"),
        ("scatter", ["age", "salary"], "Age vs Salary"),
        ("histogram", ["salary"], "Salary Distribution")
    ]
    
    for viz_type, columns, title in viz_tests:
        print(f"\nTesting {viz_type} visualization...")
        result = visualizer.create_visualization(df, viz_type, columns, title)
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Created {result['type']} chart with title: {result['title']}")
        else:
            print(f"Error: {result['error']}")

