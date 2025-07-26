# AI Tabular Data Agent

A sophisticated AI-powered system that enables natural language interaction with tabular datasets. Built using LangChain, OpenAI GPT models, and modern data processing technologies, this system allows users to upload CSV or Excel files and query their data using conversational language.

## 🌟 Features

- **Natural Language Querying**: Ask questions about your data in plain English
- **Intelligent SQL Generation**: Automatically converts natural language to SQL queries
- **Interactive Visualizations**: Generates charts and graphs based on query results
- **Multi-format Support**: Works with CSV, Excel (XLSX/XLS), and JSON files
- **Real-time Analysis**: Provides instant insights and statistical summaries
- **Web Interface**: User-friendly Streamlit-based interface
- **Sample Datasets**: Includes realistic sample data for demonstration

## 🏗️ Architecture

The system follows a modular architecture with the following components:

### Core Components

1. **Data Loader** (`data_loader.py`)
   - Handles file upload and data validation
   - Supports multiple file formats
   - Performs data cleaning and type inference
   - Creates SQLite databases for query execution

2. **NL2SQL Agent** (`nl2sql_agent.py`)
   - Converts natural language to SQL queries
   - Uses LangChain and OpenAI GPT models
   - Includes query validation and optimization
   - Handles error recovery and refinement

3. **Data Analyzer** (`data_analyzer.py`)
   - Analyzes query results and generates insights
   - Suggests appropriate visualizations
   - Provides statistical summaries
   - Creates natural language explanations

4. **Data Visualizer** (`data_analyzer.py`)
   - Creates interactive charts using Plotly
   - Supports multiple chart types (bar, line, scatter, etc.)
   - Generates publication-ready visualizations
   - Exports charts in various formats

5. **Agent Orchestrator** (`agent_orchestrator.py`)
   - Coordinates all system components
   - Manages conversation history
   - Handles session state and cleanup
   - Provides unified API interface

6. **Web Interface** (`streamlit_app.py`)
   - Streamlit-based user interface
   - File upload and management
   - Interactive chat interface
   - Real-time visualization display

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Required Python packages (see `requirements.txt`)

### Installation

1. Clone or download the project files
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

### Running the Application

1. Start the Streamlit interface:
   ```bash
   streamlit run streamlit_app.py
   ```

2. Open your browser to `http://localhost:8501`

3. Upload a CSV or Excel file

4. Start asking questions about your data!

## 📊 Sample Datasets

The system includes four realistic sample datasets:

- **Sales Data** (1000 records): Order transactions with revenue analysis
- **Employee Data** (500 records): HR information with salary and performance metrics
- **Customer Data** (800 records): Customer profiles with transaction history
- **Product Data** (200 records): Product catalog with pricing and inventory

## 💬 Example Queries

Here are some example questions you can ask:

### Basic Exploration
- "How many rows are in my dataset?"
- "What are the column names?"
- "Show me the first 10 rows"

### Statistical Analysis
- "What is the average salary by department?"
- "Which region has the highest sales?"
- "What is the correlation between price and quantity?"

### Data Visualization
- "Create a chart showing monthly sales trends"
- "Show me the distribution of customer satisfaction scores"
- "Plot revenue by product category"

### Business Intelligence
- "Who are the top 5 performing sales representatives?"
- "Which products have low inventory levels?"
- "What is the customer retention rate by industry?"

## 🔧 Technologies Used

### Core Technologies

- **LangChain**: Framework for building LLM-powered applications
- **OpenAI GPT**: Large language model for natural language understanding
- **Pandas**: Data manipulation and analysis library
- **SQLite**: Lightweight database for query execution
- **Plotly**: Interactive visualization library
- **Streamlit**: Web application framework

### Supporting Libraries

- **SQLAlchemy**: Database toolkit and ORM
- **NumPy**: Numerical computing library
- **Python-dotenv**: Environment variable management
- **OpenPyXL**: Excel file processing

## 📁 Project Structure

```
ai_tabular_agent/
├── agent_orchestrator.py      # Main orchestrator class
├── data_loader.py             # Data loading and processing
├── nl2sql_agent.py           # Natural language to SQL conversion
├── data_analyzer.py          # Data analysis and visualization
├── streamlit_app.py          # Web interface
├── sample_data.py            # Sample dataset generator
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables
├── sample_datasets/          # Sample CSV files
│   ├── sales_data.csv
│   ├── employee_data.csv
│   ├── customer_data.csv
│   ├── product_data.csv
│   └── README.md
└── README.md                 # This file
```

## 🔒 Security Considerations

- The system only executes SELECT queries for data safety
- Dangerous SQL operations (DROP, DELETE, UPDATE) are blocked
- Temporary databases are created for each session
- File uploads are validated for type and size
- API keys should be kept secure and not committed to version control

## 🎯 Use Cases

### Business Analytics
- Sales performance analysis
- Customer segmentation
- Revenue forecasting
- Market trend analysis

### Human Resources
- Salary benchmarking
- Performance evaluation
- Workforce analytics
- Diversity reporting

### Operations
- Inventory management
- Supply chain optimization
- Quality control analysis
- Process improvement

### Research & Development
- Data exploration
- Hypothesis testing
- Statistical analysis
- Report generation

## 🔧 Customization

### Adding New Data Sources
To support additional data formats, extend the `DataLoader` class:

```python
def load_custom_format(self, file_path: str) -> pd.DataFrame:
    # Implement custom loading logic
    pass
```

### Custom Visualizations
Add new chart types to the `DataVisualizer` class:

```python
def create_custom_chart(self, df: pd.DataFrame, **kwargs) -> go.Figure:
    # Implement custom visualization
    pass
```

### Different LLM Providers
Replace OpenAI with other providers by modifying the `NL2SQLAgent`:

```python
from langchain_anthropic import ChatAnthropic
# or
from langchain_community.llms import Ollama
```

## 🐛 Troubleshooting

### Common Issues

1. **API Key Error**: Ensure your OpenAI API key is set correctly
2. **File Upload Issues**: Check file format and size limits
3. **Query Failures**: Verify data quality and column names
4. **Visualization Errors**: Ensure data types are appropriate for chart type

### Debug Mode
Enable debug logging by setting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Performance Optimization

- Use smaller datasets for faster processing
- Limit query result sizes with appropriate filters
- Cache frequently accessed data
- Optimize SQL queries for better performance

## 🤝 Contributing

Contributions are welcome! Please consider:

- Adding support for new data formats
- Implementing additional visualization types
- Improving natural language understanding
- Enhancing error handling and user experience

## 📄 License

This project is provided as-is for educational and demonstration purposes.

## 🙏 Acknowledgments

- OpenAI for providing powerful language models
- LangChain community for the excellent framework
- Streamlit team for the intuitive web framework
- Plotly for beautiful interactive visualizations

---

**Built with ❤️ by Manus AI**

