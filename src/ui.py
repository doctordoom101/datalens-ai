import streamlit as st

def inject_custom_css():
    """Injects modern, high-end CSS styling for DataLens AI UI."""
    css = """
    <style>
    /* Global Fonts & Theme Tweaks */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Header */
    .datalens-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 24px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
    }
    
    .datalens-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }

    .datalens-tagline {
        color: #94A3B8;
        font-size: 1.05rem;
        font-weight: 400;
        margin: 0;
    }

    /* Metric Cards Grid */
    .metric-card {
        background: #1E293B;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        transform: translateY(-2px);
    }

    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #F8FAFC;
    }

    .metric-sub {
        font-size: 0.78rem;
        color: #94A3B8;
        margin-top: 4px;
    }

    /* Quality Health Score Badge */
    .health-badge-good {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10B981;
        color: #34D399;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        display: inline-block;
    }

    .health-badge-warning {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid #F59E0B;
        color: #FBBF24;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        display: inline-block;
    }

    /* Source Citation Badges */
    .source-container {
        margin-top: 12px;
        padding-top: 10px;
        border-top: 1px dashed rgba(255, 255, 255, 0.12);
    }

    .source-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #94A3B8;
        margin-bottom: 6px;
    }

    .source-badge {
        display: inline-block;
        background: #0F172A;
        border: 1px solid #334155;
        color: #38BDF8;
        font-size: 0.78rem;
        font-weight: 500;
        padding: 3px 10px;
        border-radius: 6px;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    /* Prompt Suggestion Pill */
    .suggestion-pill {
        background: #1E293B;
        border: 1px solid #334155;
        color: #E2E8F0;
        padding: 8px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        cursor: pointer;
        display: inline-block;
        margin-right: 8px;
        margin-bottom: 8px;
        transition: all 0.2s ease;
    }

    .suggestion-pill:hover {
        background: #334155;
        border-color: #38BDF8;
        color: #38BDF8;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_header():
    """Renders main application banner header."""
    st.markdown(
        """
        <div class="datalens-header">
            <h1 class="datalens-title">🔍 DataLens AI</h1>
            <p class="datalens-tagline">Understand Your Data. Ask Anything.</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_metric_card(label: str, value: str, subtext: str = ""):
    """Renders a styled metric card."""
    sub_html = f'<div class="metric-sub">{subtext}</div>' if subtext else ''
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True
    )


def render_source_citations(sources: list):
    """Renders clean source citation badges for AI responses."""
    if not sources:
        return
    
    badges_html = "".join([f'<span class="source-badge">📌 {src}</span>' for src in sources])
    st.markdown(
        f"""
        <div class="source-container">
            <div class="source-title">Context Sources Used:</div>
            <div>{badges_html}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
