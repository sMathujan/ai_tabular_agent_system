"""
Streamlit Web Interface for AI Tabular Data Agent

This module provides a user-friendly web interface for interacting with the AI agent
that can chat with tabular datasets.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import os
import tempfile
from datetime import datetime
from typing import Dict, Any, List
import logging

import dotenv
dotenv.load_dotenv()

from agent_orchestrator import TabularDataAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Tabular Data Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
        background-color: #f8f9fa;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    
    .agent-message {
        background-color: #f1f8e9;
        border-left-color: #4caf50;
    }
    
    .error-message {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
    
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 0.5rem;
        border: none;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background-color: #1565c0;
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'dataset_loaded' not in st.session_state:
        st.session_state.dataset_loaded = False
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'dataset_info' not in st.session_state:
        st.session_state.dataset_info = None
    if 'suggested_queries' not in st.session_state:
        st.session_state.suggested_queries = []

def setup_agent():
    """Setup the AI agent with API key."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        st.error("⚠️ OpenAI API key not found in environment variables. Please set OPENAI_API_KEY.")
        st.info("For this demo, the system is configured to use OpenAI's GPT models. You can modify the code to use other LLM providers.")
        return None
    
    try:
        agent = TabularDataAgent(api_key)
        return agent
    except Exception as e:
        st.error(f"Failed to initialize AI agent: {str(e)}")
        return None

def display_dataset_info(dataset_info: Dict[str, Any]):
    """Display dataset information in a nice format."""
    if not dataset_info:
        return
    
    st.subheader("📊 Dataset Overview")
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", f"{dataset_info['shape'][0]:,}")
    
    with col2:
        st.metric("Total Columns", dataset_info['shape'][1])
    
    with col3:
        memory_mb = dataset_info['memory_usage'] / (1024 * 1024)
        st.metric("Memory Usage", f"{memory_mb:.1f} MB")
    
    with col4:
        null_count = sum(dataset_info['null_counts'].values())
        st.metric("Missing Values", f"{null_count:,}")
    
    # Column information
    with st.expander("📋 Column Details", expanded=False):
        col_df = pd.DataFrame({
            'Column': list(dataset_info['columns']),
            'Data Type': [str(dataset_info['dtypes'][col]) for col in dataset_info['columns']],
            'Missing Values': [dataset_info['null_counts'][col] for col in dataset_info['columns']]
        })
        st.dataframe(col_df, use_container_width=True)
    
    # Sample data
    with st.expander("👀 Sample Data", expanded=False):
        sample_df = pd.DataFrame(dataset_info['sample_data'])
        st.dataframe(sample_df, use_container_width=True)

def display_query_result(result: Dict[str, Any]):
    """Display query result with data, analysis, and visualizations."""
    if not result['success']:
        st.error(f"❌ Query failed: {result.get('error', 'Unknown error')}")
        return
    
    # Query information
    st.info(f"🔍 **Query:** {result['query']}")
    
    if result.get('sql'):
        with st.expander("🗄️ Generated SQL", expanded=False):
            st.code(result['sql'], language='sql')
    
    # Results summary
    row_count = result.get('row_count', 0)
    if row_count > 0:
        st.success(f"✅ Found {row_count:,} result(s)")
        
        # Display data
        if result.get('data'):
            st.subheader("📋 Results")
            df = pd.DataFrame(result['data'])
            st.dataframe(df, use_container_width=True)
            
            # Analysis insights
            if result.get('analysis') and result['analysis'].get('insights'):
                st.subheader("💡 Insights")
                for insight in result['analysis']['insights']:
                    st.write(f"• {insight}")
            
            # Visualizations
            if result.get('visualizations'):
                st.subheader("📈 Visualizations")
                for viz in result['visualizations']:
                    if viz['success']:
                        st.plotly_chart(
                            go.Figure(json.loads(viz['figure_json'])),
                            use_container_width=True
                        )
                        st.caption(viz['title'])
    else:
        st.warning("⚠️ No results found for your query")
    
    # Explanation
    if result.get('explanation'):
        with st.expander("🤖 Agent Explanation", expanded=False):
            st.write(result['explanation'])

def main():
    """Main application function."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Tabular Data Agent</h1>', unsafe_allow_html=True)
    st.markdown("**Chat with your data using natural language! Upload a CSV or Excel file and ask questions about your data.**")
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ Configuration")
        
        # API Key status
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            st.success("✅ OpenAI API Key configured")
        else:
            st.error("❌ OpenAI API Key not found")
            st.info("Set the OPENAI_API_KEY environment variable")
        
        st.divider()
        
        # File upload
        st.header("📁 Upload Dataset")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload a CSV or Excel file to start chatting with your data"
        )
        
        if uploaded_file is not None and not st.session_state.dataset_loaded:
            if st.button("🚀 Load Dataset", type="primary"):
                with st.spinner("Loading dataset..."):
                    # Setup agent
                    st.session_state.agent = setup_agent()
                    
                    if st.session_state.agent:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        try:
                            # Load dataset
                            load_result = st.session_state.agent.load_dataset(tmp_file_path)
                            
                            if load_result['success']:
                                st.session_state.dataset_loaded = True
                                st.session_state.dataset_info = load_result['dataset_info']
                                st.session_state.suggested_queries = st.session_state.agent.suggest_queries()
                                st.success("✅ Dataset loaded successfully!")
                                st.rerun()
                            else:
                                st.error(f"Failed to load dataset: {load_result.get('error', 'Unknown error')}")
                        
                        except Exception as e:
                            st.error(f"Error loading dataset: {str(e)}")
                        
                        finally:
                            # Clean up temporary file
                            try:
                                os.unlink(tmp_file_path)
                            except:
                                pass
        
        # Dataset status
        if st.session_state.dataset_loaded:
            st.success("✅ Dataset loaded")
            if st.button("🔄 Load New Dataset"):
                # Reset session state
                if st.session_state.agent:
                    st.session_state.agent.cleanup()
                st.session_state.dataset_loaded = False
                st.session_state.dataset_info = None
                st.session_state.conversation_history = []
                st.session_state.suggested_queries = []
                st.session_state.agent = None
                st.rerun()
        
        st.divider()
        
        # Suggested queries
        if st.session_state.suggested_queries:
            st.header("💡 Suggested Queries")
            for i, suggestion in enumerate(st.session_state.suggested_queries[:5]):
                if st.button(f"📝 {suggestion}", key=f"suggestion_{i}"):
                    st.session_state.current_query = suggestion
                    st.rerun()
    
    # Main content area
    if not st.session_state.dataset_loaded:
        # Welcome screen
        st.markdown("""
        ## 🎯 How it works:
        
        1. **Upload your data** - CSV or Excel files supported
        2. **Ask questions** - Use natural language to query your data
        3. **Get insights** - Receive answers, visualizations, and analysis
        
        ## 🌟 Example questions you can ask:
        
        - "How many rows are in my dataset?"
        - "What is the average sales by region?"
        - "Show me the top 10 customers by revenue"
        - "What is the correlation between price and quantity?"
        - "Create a chart showing monthly trends"
        
        ## 🔧 Technologies used:
        
        - **LangChain** - For LLM agent orchestration
        - **OpenAI GPT** - For natural language understanding
        - **Pandas** - For data processing
        - **SQLite** - For query execution
        - **Plotly** - For interactive visualizations
        - **Streamlit** - For the web interface
        """)
        
    else:
        # Dataset loaded - show interface
        
        # Display dataset info
        display_dataset_info(st.session_state.dataset_info)
        
        st.divider()
        
        # Chat interface
        st.subheader("💬 Chat with your data")
        
        # Query input
        query_input = st.text_input(
            "Ask a question about your data:",
            placeholder="e.g., What is the average salary by department?",
            key="query_input"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🚀 Ask Question", type="primary", disabled=not query_input.strip()):
                if query_input.strip():
                    with st.spinner("Processing your question..."):
                        result = st.session_state.agent.query_data(query_input.strip())
                        
                        # Add to conversation history
                        st.session_state.conversation_history.append({
                            'query': query_input.strip(),
                            'result': result,
                            'timestamp': datetime.now()
                        })
                        
                        # Clear input
                        st.session_state.query_input = ""
                        st.rerun()
        
        with col2:
            if st.button("🧹 Clear History"):
                st.session_state.conversation_history = []
                st.rerun()
        
        # Display conversation history
        if st.session_state.conversation_history:
            st.subheader("📜 Conversation History")
            
            # Show conversations in reverse order (newest first)
            for i, conversation in enumerate(reversed(st.session_state.conversation_history)):
                with st.container():
                    # User query
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>🧑 You:</strong> {conversation['query']}
                        <br><small>⏰ {conversation['timestamp'].strftime('%H:%M:%S')}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Agent response
                    st.markdown(f"""
                    <div class="chat-message agent-message">
                        <strong>🤖 Agent:</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    display_query_result(conversation['result'])
                    
                    st.divider()

if __name__ == "__main__":
    main()

