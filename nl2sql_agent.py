"""
Natural Language to SQL Agent

This module implements an LLM-based agent that converts natural language queries
into SQL statements and executes them against tabular data.
"""

import sqlite3
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from langchain_openai import ChatOpenAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.schema import AgentAction, AgentFinish
import re
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NL2SQLAgent:
    """
    An agent that converts natural language queries to SQL and executes them.
    """
    
    def __init__(self, openai_api_key: str, model_name: str = "gpt-4o-mini"):
        """
        Initialize the NL2SQL agent.
        
        Args:
            openai_api_key (str): OpenAI API key
            model_name (str): Name of the OpenAI model to use
        """
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model_name,
            temperature=0
        )
        self.db = None
        self.agent = None
        self.schema_info = ""
        
    def connect_to_database(self, db_path: str, schema_info: str = ""):
        """
        Connect to a SQLite database and create the SQL agent.
        
        Args:
            db_path (str): Path to the SQLite database
            schema_info (str): Additional schema information for context
        """
        try:
            # Create SQLDatabase instance
            self.db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
            self.schema_info = schema_info
            
            # Create SQL toolkit
            toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
            
            # Create the SQL agent with custom prompt
            self.agent = create_sql_agent(
                llm=self.llm,
                toolkit=toolkit,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                max_iterations=5,
                early_stopping_method="generate"
            )
            
            logger.info(f"Successfully connected to database: {db_path}")
            
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            raise
    
    def query(self, natural_language_query: str) -> Dict[str, Any]:
        """
        Execute a natural language query against the database.
        
        Args:
            natural_language_query (str): The user's question in natural language
            
        Returns:
            Dict[str, Any]: Query results including data, SQL, and metadata
        """
        if not self.agent:
            raise ValueError("Agent not initialized. Call connect_to_database first.")
        
        try:
            # Enhance the query with schema context
            enhanced_query = self._enhance_query_with_context(natural_language_query)
            
            # Execute the query using the agent
            result = self.agent.invoke({"input": enhanced_query})
            
            # Parse and format the result
            formatted_result = self._format_agent_result(result, natural_language_query)
            
            return formatted_result
            
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "query": natural_language_query,
                "sql": None,
                "data": None,
                "explanation": "An error occurred while processing your query."
            }
    
    def _enhance_query_with_context(self, query: str) -> str:
        """
        Enhance the user query with database schema context.
        
        Args:
            query (str): Original user query
            
        Returns:
            str: Enhanced query with context
        """
        context_prompt = f"""
You are a SQL expert helping users query their data. Here's the database schema:

{self.schema_info}

User Question: {query}

Please generate an appropriate SQL query to answer this question. Consider:
1. Use the exact column names as shown in the schema
2. Handle potential data type issues appropriately
3. If the question asks for aggregations, use appropriate SQL functions
4. If the question is ambiguous, make reasonable assumptions
5. Always return results in a user-friendly format

"""
        return context_prompt
    
    def _format_agent_result(self, agent_result: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """
        Format the agent result into a standardized response.
        
        Args:
            agent_result (Dict[str, Any]): Raw result from the agent
            original_query (str): Original user query
            
        Returns:
            Dict[str, Any]: Formatted result
        """
        try:
            output = agent_result.get("output", "")
            
            # Try to extract SQL query from the agent's reasoning
            sql_query = self._extract_sql_from_output(output)
            
            # Try to extract data if the query was successful
            data = None
            if sql_query:
                try:
                    data = self._execute_sql_directly(sql_query)
                except Exception as e:
                    logger.warning(f"Could not execute extracted SQL: {str(e)}")
            
            return {
                "success": True,
                "query": original_query,
                "sql": sql_query,
                "data": data,
                "explanation": output,
                "agent_output": agent_result
            }
            
        except Exception as e:
            logger.error(f"Error formatting agent result: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "query": original_query,
                "sql": None,
                "data": None,
                "explanation": "Error formatting the query result."
            }
    
    def _extract_sql_from_output(self, output: str) -> Optional[str]:
        """
        Extract SQL query from the agent's output.
        
        Args:
            output (str): Agent's output text
            
        Returns:
            Optional[str]: Extracted SQL query
        """
        # Look for SQL queries in the output
        sql_patterns = [
            r"```sql\n(.*?)\n```",
            r"```\n(SELECT.*?)\n```",
            r"(SELECT.*?)(?:\n|$)",
            r"Query:\s*(SELECT.*?)(?:\n|$)"
        ]
        
        for pattern in sql_patterns:
            matches = re.findall(pattern, output, re.DOTALL | re.IGNORECASE)
            if matches:
                sql = matches[0].strip()
                if sql.upper().startswith('SELECT'):
                    return sql
        
        return None
    
    def _execute_sql_directly(self, sql_query: str) -> List[Dict[str, Any]]:
        """
        Execute SQL query directly against the database.
        
        Args:
            sql_query (str): SQL query to execute
            
        Returns:
            List[Dict[str, Any]]: Query results
        """
        if not self.db:
            raise ValueError("Database not connected")
        
        try:
            # Execute the query
            result = self.db.run(sql_query)
            
            # Parse the result if it's a string representation
            if isinstance(result, str):
                # Try to parse as a list of tuples or other formats
                # This is a simplified parser - in practice, you might need more robust parsing
                return [{"result": result}]
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing SQL directly: {str(e)}")
            raise
    
    def get_table_info(self) -> str:
        """
        Get information about tables in the database.
        
        Returns:
            str: Table information
        """
        if not self.db:
            return "No database connected"
        
        try:
            return self.db.get_table_info()
        except Exception as e:
            logger.error(f"Error getting table info: {str(e)}")
            return f"Error retrieving table information: {str(e)}"
    
    def validate_sql(self, sql_query: str) -> Tuple[bool, str]:
        """
        Validate a SQL query without executing it.
        
        Args:
            sql_query (str): SQL query to validate
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        try:
            # Basic SQL validation
            sql_query = sql_query.strip()
            
            # Check for dangerous operations
            dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
            for keyword in dangerous_keywords:
                if keyword.upper() in sql_query.upper():
                    return False, f"Dangerous operation detected: {keyword}"
            
            # Check if it's a SELECT query
            if not sql_query.upper().startswith('SELECT'):
                return False, "Only SELECT queries are allowed"
            
            # Try to parse with SQLite (this will catch syntax errors)
            if self.db:
                try:
                    # Use EXPLAIN to validate without executing
                    self.db.run(f"EXPLAIN {sql_query}")
                    return True, "Valid SQL query"
                except Exception as e:
                    return False, f"SQL syntax error: {str(e)}"
            
            return True, "Basic validation passed"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"


class QueryOptimizer:
    """
    Optimizes and enhances SQL queries for better performance and results.
    """
    
    def __init__(self):
        pass
    
    def optimize_query(self, sql_query: str, table_info: str) -> str:
        """
        Optimize a SQL query for better performance.
        
        Args:
            sql_query (str): Original SQL query
            table_info (str): Information about the table structure
            
        Returns:
            str: Optimized SQL query
        """
        # Basic optimizations
        optimized = sql_query.strip()
        
        # Add LIMIT if not present and no aggregation
        if 'LIMIT' not in optimized.upper() and not any(agg in optimized.upper() for agg in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'GROUP BY']):
            optimized += " LIMIT 100"
        
        return optimized
    
    def suggest_improvements(self, sql_query: str) -> List[str]:
        """
        Suggest improvements for a SQL query.
        
        Args:
            sql_query (str): SQL query to analyze
            
        Returns:
            List[str]: List of improvement suggestions
        """
        suggestions = []
        
        query_upper = sql_query.upper()
        
        # Check for SELECT *
        if 'SELECT *' in query_upper:
            suggestions.append("Consider selecting specific columns instead of using SELECT *")
        
        # Check for missing LIMIT
        if 'LIMIT' not in query_upper and 'COUNT' not in query_upper:
            suggestions.append("Consider adding a LIMIT clause to prevent large result sets")
        
        # Check for potential performance issues
        if 'LIKE' in query_upper and '%' in sql_query:
            suggestions.append("LIKE queries with leading wildcards can be slow on large datasets")
        
        return suggestions


# Example usage and testing
if __name__ == "__main__":
    import os
    from data_loader import DataLoader
    
    # Create sample data for testing
    sample_data = {
        'employee_id': [1, 2, 3, 4, 5],
        'name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Diana Prince', 'Eve Wilson'],
        'age': [25, 30, 35, 28, 32],
        'salary': [50000, 60000, 70000, 55000, 65000],
        'department': ['Engineering', 'Marketing', 'Engineering', 'Sales', 'Marketing'],
        'hire_date': ['2020-01-15', '2019-03-20', '2018-07-10', '2021-02-28', '2020-11-05']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Create database
    loader = DataLoader()
    db_path = loader.create_sqlite_db(df, 'employees')
    schema_info = loader.get_schema_info(df, 'employees')
    
    # Test the NL2SQL agent (requires OpenAI API key)
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        try:
            agent = NL2SQLAgent(api_key)
            agent.connect_to_database(db_path, schema_info)
            
            # Test queries
            test_queries = [
                "How many employees are there?",
                "What is the average salary?",
                "Show me all employees in the Engineering department",
                "Who is the highest paid employee?"
            ]
            
            for query in test_queries:
                print(f"\nQuery: {query}")
                result = agent.query(query)
                print(f"Result: {result}")
                
        except Exception as e:
            print(f"Error testing agent: {str(e)}")
    else:
        print("OPENAI_API_KEY not found in environment variables")
    
    # Clean up
    os.unlink(db_path)

