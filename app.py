import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

from src.profiler import DataProfiler
from src.knowledge_base import KnowledgeBaseGenerator
from src.rag_engine import RAGEngine
from src.ui import inject_custom_css, render_header, render_metric_card, render_source_citations

# Page Configuration
st.set_page_config(
    page_title="DataLens AI — Understand Your Data. Ask Anything.",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS
inject_custom_css()

# Session State Initialization
if "df" not in st.session_state:
    st.session_state.df = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None
if "profile" not in st.session_state:
    st.session_state.profile = None
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "suggested_question" not in st.session_state:
    st.session_state.suggested_question = ""


def process_dataset(df: pd.DataFrame, name: str, api_key: str):
    """Processes uploaded dataframe: profiles data, builds knowledge base, and indexes into ChromaDB."""
    with st.spinner("⚡ Profiling dataset & indexing metadata into RAG Vector Store..."):
        st.session_state.df = df
        st.session_state.dataset_name = name
        st.session_state.chat_history = []

        # 1. Profile Data
        profiler = DataProfiler(df, name)
        st.session_state.profile = profiler.get_full_profile()

        # 2. Knowledge Base Generation
        kb_gen = KnowledgeBaseGenerator(st.session_state.profile)
        documents = kb_gen.generate_documents()

        # 3. RAG Vector DB Indexing
        if api_key:
            try:
                engine = RAGEngine(api_key=api_key)
                indexed_count = engine.index_documents(documents)
                st.session_state.rag_engine = engine
                st.sidebar.success(f"✅ Indexed {indexed_count} metadata chunks into RAG Vector Store!")
            except Exception as e:
                st.session_state.rag_engine = None
                st.sidebar.error(f"⚠️ Vector DB Indexing Warning: {str(e)}")
        else:
            st.session_state.rag_engine = None
            st.sidebar.warning("⚠️ OpenAI API Key missing. RAG chat will require API key.")


# Sidebar Section
st.sidebar.image("https://img.icons8.com/isometric-folders/100/data-configuration.png", width=64)
st.sidebar.title("DataLens AI")

# API Key Input
env_key = os.getenv("OPENAI_API_KEY", "")
openai_api_key = st.sidebar.text_input(
    "🔑 OpenAI API Key",
    value=env_key,
    type="password",
    help="Enter your OpenAI API key to enable RAG Chatbot features."
)

st.sidebar.markdown("---")
st.sidebar.subheader("📁 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="Upload your dataset in CSV format."
)

sample_button = st.sidebar.button("🚀 Load Global Superstore Sample", use_container_width=True)

# Load data based on user action
if sample_button:
    sample_path = os.path.join(os.path.dirname(__file__), "sample_data", "global_superstore_sample.csv")
    if os.path.exists(sample_path):
        sample_df = pd.read_csv(sample_path)
        process_dataset(sample_df, "Global Superstore Sample.csv", openai_api_key)
    else:
        st.sidebar.error("Sample dataset file not found.")

elif uploaded_file is not None:
    if st.session_state.dataset_name != uploaded_file.name:
        df = pd.read_csv(uploaded_file)
        process_dataset(df, uploaded_file.name, openai_api_key)

# Main Application Content
render_header()

if st.session_state.df is None:
    st.info("👈 Please upload a **CSV file** or click **'Load Global Superstore Sample'** in the sidebar to get started.")
    
    # Feature highlights box when no dataset is loaded
    st.markdown(
        """
        ### 🌟 Welcome to DataLens AI
        **DataLens AI** makes dataset exploration effortless with AI-powered Retrieval-Augmented Generation (RAG).

        #### How it works:
        1. **Upload CSV Dataset**: Load your data seamlessly.
        2. **Automatic Data Profiling**: Instantly view row counts, column metrics, missing values, duplicates, and statistical distribution.
        3. **RAG Knowledge Generation**: Dataset schema and quality statistics are automatically vectorized.
        4. **Conversational AI Assistant**: Ask questions in natural language and receive grounded answers with exact source context references!
        """
    )

else:
    summary = st.session_state.profile["summary"]
    columns_profile = st.session_state.profile["columns"]

    # Top Navigation Tabs
    tab1, tab2 = st.tabs(["📊 Dataset Overview & Profiling", "💬 AI Data Assistant (RAG Chat)"])

    # ==================== TAB 1: OVERVIEW & PROFILING ====================
    with tab1:
        st.caption(f"Currently inspecting: **{st.session_state.dataset_name}**")
        
        # 6 Metric Cards Layout
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            render_metric_card("Total Rows", f"{summary['total_rows']:,}")
        with c2:
            render_metric_card("Total Columns", f"{summary['total_columns']}")
        with c3:
            render_metric_card("Duplicate Rows", f"{summary['duplicate_rows']}")
        with c4:
            render_metric_card("Missing Cells", f"{summary['total_missing_cells']}", f"{summary['missing_percentage']}% of cells")
        with c5:
            render_metric_card("Memory Size", f"{summary['memory_usage']}")
        with c6:
            score = summary["health_score"]
            badge_class = "health-badge-good" if score >= 80 else "health-badge-warning"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Quality Score</div>
                    <div class="metric-value">{score}/100</div>
                    <div style="margin-top: 4px;"><span class="{badge_class}">Health Score</span></div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.subheader("📋 Columns Profile Overview")
            
            # Formatted table of columns
            table_data = []
            for col in columns_profile:
                table_data.append({
                    "Column Name": col["column_name"],
                    "Data Type": col["data_type"],
                    "Missing Count": f"{col['missing_count']} ({col['missing_pct']}%)",
                    "Unique Values": col["unique_count"],
                    "Sample Values": col["sample_values"]
                })
            
            df_table = pd.DataFrame(table_data)
            st.dataframe(df_table, use_container_width=True, height=350)

        with col_right:
            st.subheader("🔍 Deep Column Inspector")
            selected_col_name = st.selectbox(
                "Select column to inspect:",
                options=[col["column_name"] for col in columns_profile]
            )

            selected_col = next(col for col in columns_profile if col["column_name"] == selected_col_name)
            
            st.markdown(f"**Data Type:** `{selected_col['data_type']}`")
            st.markdown(f"**Missing Values:** `{selected_col['missing_count']} ({selected_col['missing_pct']}%)`")
            st.markdown(f"**Unique Values:** `{selected_col['unique_count']}`")
            
            if selected_col["is_numeric"] and selected_col.get("stats"):
                st.markdown("##### 📈 Numeric Statistics")
                stats = selected_col["stats"]
                st.json(stats)
            elif selected_col["is_categorical"] and selected_col.get("stats"):
                st.markdown("##### 🏷️ Top Frequent Values")
                st.json(selected_col["stats"].get("top_values", {}))

    # ==================== TAB 2: AI DATA CHAT ====================
    with tab2:
        st.caption(f"Connected dataset: **{st.session_state.dataset_name}** | RAG Engine: **{'Active ✅' if st.session_state.rag_engine else 'Inactive ❌ (API Key required)'}**")

        if not openai_api_key:
            st.warning("⚠️ Please provide an **OpenAI API Key** in the sidebar to ask questions to the RAG model.")

        st.subheader("💡 Prompt Suggestions")
        st.write("Click any suggestion to prefill your query:")

        # Quick prompt buttons
        q_cols = st.columns(4)
        with q_cols[0]:
            if st.button("📌 Explain dataset", use_container_width=True):
                st.session_state.suggested_question = "Jelaskan ringkasan dataset ini dan struktur utamanya."
        with q_cols[1]:
            if st.button("⚠️ Missing values", use_container_width=True):
                st.session_state.suggested_question = "Kolom apa saja yang paling banyak memiliki missing values?"
        with q_cols[2]:
            if st.button("🔄 Duplicate check", use_container_width=True):
                st.session_state.suggested_question = "Apakah ada duplicate rows dalam dataset ini?"
        with q_cols[3]:
            if st.button("🎯 ML suitability", use_container_width=True):
                st.session_state.suggested_question = "Apakah dataset ini cocok untuk model prediksi regresi atau klasifikasi?"

        st.markdown("<br>", unsafe_allow_html=True)

        # Render Chat History
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg["role"] == "assistant" and "sources" in msg:
                    render_source_citations(msg["sources"])

        # Chat Input
        default_val = st.session_state.suggested_question
        user_query = st.chat_input("Ask anything about your dataset...")
        
        # Override with button suggestion if clicked
        if not user_query and default_val:
            user_query = default_val
            st.session_state.suggested_question = "" # Reset suggestion

        if user_query:
            # Render user question
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.write(user_query)

            # Generate AI answer
            with st.chat_message("assistant"):
                if not st.session_state.rag_engine:
                    if not openai_api_key:
                        resp_text = "DataLens AI error: OpenAI API Key is missing. Please enter your API key in the sidebar."
                    else:
                        # Re-try initializing engine if key present now
                        try:
                            kb_gen = KnowledgeBaseGenerator(st.session_state.profile)
                            docs = kb_gen.generate_documents()
                            st.session_state.rag_engine = RAGEngine(openai_api_key)
                            st.session_state.rag_engine.index_documents(docs)
                            res = st.session_state.rag_engine.query(user_query)
                            resp_text = res["answer"]
                            sources = res["sources"]
                        except Exception as err:
                            resp_text = f"Error querying RAG engine: {str(err)}"
                            sources = []
                else:
                    with st.spinner("🧠 Retrieving dataset context & generating RAG response..."):
                        res = st.session_state.rag_engine.query(user_query)
                        resp_text = res["answer"]
                        sources = res["sources"]

                st.write(resp_text)
                if 'sources' in locals() and sources:
                    render_source_citations(sources)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": resp_text,
                        "sources": sources
                    })
                else:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": resp_text
                    })

        if st.session_state.chat_history:
            if st.button("🧹 Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
