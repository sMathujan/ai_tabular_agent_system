"""
Agent Orchestrator

This module coordinates the interaction between different components of the AI agent system,
managing the flow from data loading to query processing to visualization.
"""

import os
import tempfile
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
from datetime import datetime
import pandas as pd

from data_loader import DataLoader, DataProcessor
from nl2sql_agent import NL2SQLAgent, QueryOptimizer
from data_analyzer import DataAnalyzer, DataVisualizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TabularDataAgent:
    """
    Main orchestrator for the tabular data AI agent system.
    """
    
    def __init__(self, openai_api_key: str, model_name: str = "gpt-4o-mini"):
        """
        Initialize the tabular data agent.
        
        Args:
            openai_api_key (str): OpenAI API key
            model_name (str): Name of the OpenAI model to use
        """
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        
        # Initialize components
        self.data_loader = DataLoader()
        self.data_processor = DataProcessor()
        self.nl2sql_agent = NL2SQLAgent(openai_api_key, model_name)
        self.query_optimizer = QueryOptimizer()
        self.data_analyzer = DataAnalyzer()
        self.data_visualizer = DataVisualizer()
        
        # State management
        self.current_dataset = None
        self.current_db_path = None
        self.dataset_info = None
        self.schema_info = ""
        self.conversation_history = []
        
    def load_dataset(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Load a dataset from a file.
        
        Args:
            file_path (str): Path to the data file
            **kwargs: Additional arguments for data loading
            
        Returns:
            Dict[str, Any]: Loading result with dataset information
        """
        try:
            logger.info(f"Loading dataset from {file_path}")
            
            # Load the data
            self.current_dataset = self.data_loader.load_file(file_path, **kwargs)
            
            # Process the data (basic cleaning)
            self.current_dataset = self.data_processor.clean_data(self.current_dataset)
            self.current_dataset = self.data_processor.infer_data_types(self.current_dataset)
            
            # Get dataset information
            self.dataset_info = self.data_loader.get_data_info(self.current_dataset)
            
            # Create SQLite database
            self.current_db_path = self.data_loader.create_sqlite_db(self.current_dataset)
            
            # Generate schema information
            self.schema_info = self.data_loader.get_schema_info(self.current_dataset)
            
            # Connect the NL2SQL agent to the database
            self.nl2sql_agent.connect_to_database(self.current_db_path, self.schema_info)
            
            # Validate the data
            warnings = self.data_loader.validate_data(self.current_dataset)
            
            # Clear conversation history for new dataset
            self.conversation_history = []
            
            result = {
                "success": True,
                "message": f"Successfully loaded dataset with {len(self.current_dataset)} rows and {len(self.current_dataset.columns)} columns",
                "dataset_info": self.dataset_info,
                "schema_info": self.schema_info,
                "warnings": warnings,
                "file_path": file_path
            }
            
            logger.info("Dataset loaded successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to load dataset from {file_path}"
            }
    
    def query_data(self, natural_language_query: str, include_visualization: bool = True) -> Dict[str, Any]:
        """
        Process a natural language query against the loaded dataset.
        
        Args:
            natural_language_query (str): User's question in natural language
            include_visualization (bool): Whether to generate visualizations
            
        Returns:
            Dict[str, Any]: Query result with data, analysis, and visualizations
        """
        if self.current_dataset is None:
            return {
                "success": False,
                "error": "No dataset loaded",
                "message": "Please load a dataset first before querying"
            }
        
        try:
            logger.info(f"Processing query: {natural_language_query}")
            
            # Record the query in conversation history
            query_timestamp = datetime.now().isoformat()
            
            # Execute the query using NL2SQL agent
            sql_result = self.nl2sql_agent.query(natural_language_query)
            
            if not sql_result["success"]:
                return {
                    "success": False,
                    "error": sql_result.get("error", "Query execution failed"),
                    "message": "Failed to execute the query",
                    "query": natural_language_query,
                    "timestamp": query_timestamp
                }
            
            # Analyze the results
            analysis = None
            if sql_result["data"]:
                analysis = self.data_analyzer.analyze_query_result(
                    sql_result["data"], 
                    natural_language_query
                )
            
            # Generate visualizations if requested and data is suitable
            visualizations = []
            if include_visualization and sql_result["data"] and analysis:
                viz_suggestions = analysis.get("visualization_suggestions", [])
                
                # Create visualizations based on suggestions
                for suggestion in viz_suggestions[:2]:  # Limit to 2 visualizations
                    try:
                        df_result = pd.DataFrame(sql_result["data"])
                        viz_result = self.data_visualizer.create_visualization(
                            df_result,
                            suggestion["type"],
                            suggestion["columns"],
                            suggestion["description"]
                        )
                        if viz_result["success"]:
                            visualizations.append(viz_result)
                    except Exception as viz_error:
                        logger.warning(f"Failed to create visualization: {str(viz_error)}")
            
            # Prepare the complete result
            result = {
                "success": True,
                "query": natural_language_query,
                "sql": sql_result.get("sql"),
                "data": sql_result["data"],
                "explanation": sql_result.get("explanation", ""),
                "analysis": analysis,
                "visualizations": visualizations,
                "timestamp": query_timestamp,
                "row_count": len(sql_result["data"]) if sql_result["data"] else 0
            }
            
            # Add to conversation history
            self.conversation_history.append({
                "query": natural_language_query,
                "result": result,
                "timestamp": query_timestamp
            })
            
            logger.info(f"Query processed successfully, returned {result['row_count']} rows")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "An error occurred while processing your query",
                "query": natural_language_query,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the currently loaded dataset.
        
        Returns:
            Dict[str, Any]: Dataset summary information
        """
        if self.current_dataset is None:
            return {
                "success": False,
                "message": "No dataset loaded"
            }
        
        try:
            # Generate a comprehensive summary
            summary = {
                "success": True,
                "basic_info": {
                    "rows": len(self.current_dataset),
                    "columns": len(self.current_dataset.columns),
                    "column_names": list(self.current_dataset.columns),
                    "memory_usage_mb": self.current_dataset.memory_usage(deep=True).sum() / 1024 / 1024
                },
                "data_types": self.current_dataset.dtypes.to_dict(),
                "missing_data": self.current_dataset.isnull().sum().to_dict(),
                "sample_data": self.current_dataset.head().to_dict('records'),
                "schema_info": self.schema_info
            }
            
            # Add statistical summary for numeric columns
            numeric_cols = self.current_dataset.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                summary["numeric_summary"] = self.current_dataset[numeric_cols].describe().to_dict()
            
            # Add categorical summary
            categorical_cols = self.current_dataset.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                summary["categorical_summary"] = {}
                for col in categorical_cols:
                    value_counts = self.current_dataset[col].value_counts().head()
                    summary["categorical_summary"][col] = value_counts.to_dict()
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating dataset summary: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to generate dataset summary"
            }
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get the conversation history for the current session.
        
        Returns:
            List[Dict[str, Any]]: List of previous queries and results
        """
        return self.conversation_history
    
    def suggest_queries(self) -> List[str]:
        """
        Suggest potential queries based on the current dataset.
        
        Returns:
            List[str]: List of suggested queries
        """
        if self.current_dataset is None:
            return []
        
        suggestions = []
        
        # Basic exploration queries
        suggestions.append("How many rows are in the dataset?")
        suggestions.append("Show me the first 10 rows")
        suggestions.append("What are the column names?")
        
        # Column-specific queries
        numeric_cols = self.current_dataset.select_dtypes(include=['number']).columns
        categorical_cols = self.current_dataset.select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            suggestions.extend([
                f"What is the average {col}?",
                f"What is the maximum {col}?",
                f"Show me the distribution of {col}"
            ])
        
        if len(categorical_cols) > 0:
            col = categorical_cols[0]
            suggestions.extend([
                f"How many unique values are in {col}?",
                f"Show me the count of each {col}",
                f"What is the most common {col}?"
            ])
        
        # Relationship queries
        if len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[0], numeric_cols[1]
            suggestions.append(f"What is the relationship between {col1} and {col2}?")
        
        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            suggestions.append(f"Show me {num_col} by {cat_col}")
        
        return suggestions[:10]  # Limit to 10 suggestions
    
    def export_results(self, query_result: Dict[str, Any], format: str = "csv") -> str:
        """
        Export query results to a file.
        
        Args:
            query_result (Dict[str, Any]): Query result to export
            format (str): Export format (csv, json, excel)
            
        Returns:
            str: Path to exported file
        """
        if not query_result.get("success") or not query_result.get("data"):
            raise ValueError("No valid data to export")
        
        # Create temporary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == "csv":
            filename = f"query_result_{timestamp}.csv"
            df = pd.DataFrame(query_result["data"])
            df.to_csv(filename, index=False)
        elif format.lower() == "json":
            filename = f"query_result_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(query_result["data"], f, indent=2, default=str)
        elif format.lower() == "excel":
            filename = f"query_result_{timestamp}.xlsx"
            df = pd.DataFrame(query_result["data"])
            df.to_excel(filename, index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Results exported to {filename}")
        return filename
    
    def cleanup(self):
        """
        Clean up temporary files and resources.
        """
        if self.current_db_path and os.path.exists(self.current_db_path):
            try:
                os.unlink(self.current_db_path)
                logger.info("Cleaned up temporary database file")
            except Exception as e:
                logger.warning(f"Failed to clean up database file: {str(e)}")
        
        # Reset state
        self.current_dataset = None
        self.current_db_path = None
        self.dataset_info = None
        self.schema_info = ""
        self.conversation_history = []


# Example usage and testing
if __name__ == "__main__":
    import tempfile
    
    # Create sample data for testing
    sample_data = {
        'employee_id': range(1, 101),
        'name': [f'Employee_{i}' for i in range(1, 101)],
        'age': [25 + (i % 40) for i in range(100)],
        'salary': [40000 + (i * 1000) + (i % 10) * 5000 for i in range(100)],
        'department': [['Engineering', 'Marketing', 'Sales', 'HR', 'Finance'][i % 5] for i in range(100)],
        'hire_date': [f'202{i%3}-{(i%12)+1:02d}-{(i%28)+1:02d}' for i in range(100)]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Save to temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_csv_path = f.name
    
    # Test the agent (requires OpenAI API key)
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        try:
            # Initialize the agent
            agent = TabularDataAgent(api_key)
            
            # Load dataset
            load_result = agent.load_dataset(temp_csv_path)
            print("Load result:")
            print(json.dumps(load_result, indent=2, default=str))
            
            # Get dataset summary
            summary = agent.get_dataset_summary()
            print("\nDataset summary:")
            print(json.dumps(summary, indent=2, default=str))
            
            # Test queries
            test_queries = [
                "How many employees are there?",
                "What is the average salary?",
                "Show me employees in the Engineering department",
                "What is the salary distribution by department?"
            ]
            
            for query in test_queries:
                print(f"\n{'='*50}")
                print(f"Query: {query}")
                result = agent.query_data(query)
                print(f"Success: {result['success']}")
                if result['success']:
                    print(f"Rows returned: {result['row_count']}")
                    print(f"SQL: {result.get('sql', 'N/A')}")
                    if result.get('analysis'):
                        print(f"Insights: {result['analysis'].get('insights', [])}")
                else:
                    print(f"Error: {result.get('error', 'Unknown error')}")
            
            # Get suggestions
            suggestions = agent.suggest_queries()
            print(f"\nSuggested queries:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"{i}. {suggestion}")
            
            # Cleanup
            agent.cleanup()
            
        except Exception as e:
            print(f"Error testing agent: {str(e)}")
    else:
        print("OPENAI_API_KEY not found in environment variables")
    
    # Clean up temporary file
    os.unlink(temp_csv_path)

