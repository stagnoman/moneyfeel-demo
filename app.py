"""
moneyfeel Fundamental Analysis Engine - Interactive Demo
Streamlit dashboard per demo cliente
BRANDED VERSION con colori e logo moneyfeel
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ============================================================================
# MONEYFEEL BRANDING
# ============================================================================

MONEYFEEL_COLORS = {
    'pink': '#fb0086',
    'blue': '#4db2e6',
    'purple': '#9a4eb2',
    'dark': '#1a1a2e',
    'gray': '#f5f5f5'
}

MONEYFEEL_LOGO = "https://moneyfeel.it/favicon/moneyfeel-1200x630.png"

# Rating color map (usando colori moneyfeel)
RATING_COLORS = {
    'Strong Buy': MONEYFEEL_COLORS['blue'],
    'Buy': MONEYFEEL_COLORS['purple'],
    'Hold': '#ffd600',
    'Sell': '#ff6d00',
    'Strong Sell': MONEYFEEL_COLORS['pink']
}

# Custom CSS
CUSTOM_CSS = f"""
<style>
    /* Main colors */
    :root {{
        --moneyfeel-pink: {MONEYFEEL_COLORS['pink']};
        --moneyfeel-blue: {MONEYFEEL_COLORS['blue']};
        --moneyfeel-purple: {MONEYFEEL_COLORS['purple']};
    }}
    
    /* Header styling */
    h1 {{
        color: {MONEYFEEL_COLORS['pink']} !important;
        font-weight: 700 !important;
    }}
    
    h2, h3 {{
        color: {MONEYFEEL_COLORS['purple']} !important;
    }}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {MONEYFEEL_COLORS['purple']}15 0%, {MONEYFEEL_COLORS['blue']}15 100%);
    }}
    
    [data-testid="stSidebar"] h1 {{
        color: {MONEYFEEL_COLORS['purple']} !important;
    }}
    
    /* Metric styling */
    [data-testid="stMetricValue"] {{
        color: {MONEYFEEL_COLORS['purple']} !important;
        font-weight: 600 !important;
    }}
    
    /* Button styling */
    .stDownloadButton button {{
        background-color: {MONEYFEEL_COLORS['pink']} !important;
        color: white !important;
        border: none !important;
    }}
    
    .stDownloadButton button:hover {{
        background-color: {MONEYFEEL_COLORS['purple']} !important;
    }}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: {MONEYFEEL_COLORS['gray']};
        border-radius: 4px 4px 0 0;
        padding: 10px 20px;
        color: {MONEYFEEL_COLORS['purple']};
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {MONEYFEEL_COLORS['purple']} !important;
        color: white !important;
    }}
    
    /* Link styling */
    a {{
        color: {MONEYFEEL_COLORS['blue']} !important;
    }}
    
    a:hover {{
        color: {MONEYFEEL_COLORS['pink']} !important;
    }}
    
    /* Slider styling */
    .stSlider [data-baseweb="slider"] {{
        background: linear-gradient(90deg, {MONEYFEEL_COLORS['blue']}, {MONEYFEEL_COLORS['purple']});
    }}
    
    /* Logo in header */
    .moneyfeel-logo {{
        width: 180px;
        margin-bottom: 20px;
    }}
    
    /* Footer styling */
    .moneyfeel-footer {{
        background: linear-gradient(90deg, {MONEYFEEL_COLORS['purple']}20, {MONEYFEEL_COLORS['blue']}20);
        padding: 30px;
        border-radius: 10px;
        text-align: center;
        margin-top: 40px;
    }}
    
    .moneyfeel-footer strong {{
        color: {MONEYFEEL_COLORS['pink']};
        font-size: 1.2em;
    }}
</style>
"""

# ============================================================================
# CONFIG
# ============================================================================

st.set_page_config(
    page_title="moneyfeel Fundamental Analysis Engine - Demo",
    page_icon="https://moneyfeel.it/favicon/favicon-32x32.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_data():
    """Load demo dataset"""
    data_path = Path(__file__).parent / "data" / "demo_valuations.parquet"
    
    if not data_path.exists():
        st.error(f"❌ File demo non trovato: {data_path}")
        st.info("🔧 Esegui prima `python data_extractor.py` per generare il dataset demo")
        st.stop()
    
    df = pd.read_parquet(data_path)
    return df

# Load data
with st.spinner('📊 Caricamento dataset...'):
    df = load_data()

# ============================================================================
# SIDEBAR - FILTRI
# ============================================================================

# Logo in sidebar
st.sidebar.markdown(
    f'<img src="{MONEYFEEL_LOGO}" class="moneyfeel-logo" alt="moneyfeel logo">',
    unsafe_allow_html=True
)

st.sidebar.title("🔍 Filtri Dataset")
st.sidebar.markdown(f"**Demo Dataset:** {len(df)} titoli high-quality")
st.sidebar.markdown("---")

# Sector filter
all_sectors = ['All'] + sorted(df['sector'].dropna().unique().tolist())
selected_sectors = st.sidebar.multiselect(
    "📂 Settore",
    options=all_sectors,
    default=['All'] if 'All' in all_sectors else all_sectors[:3]
)

# Rating filter
all_ratings = ['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell']
selected_ratings = st.sidebar.multiselect(
    "⭐ Rating",
    options=all_ratings,
    default=['Strong Buy', 'Buy', 'Hold']
)

# DQS filter
dqs_min = df['dqs_score'].min()
dqs_max = df['dqs_score'].max()

# Fix: se DQS è 0-1 invece di 0-100, moltiplica per 100 per display
if dqs_max <= 10:
    dqs_display_min = int(dqs_min * 100)
    dqs_display_max = int(dqs_max * 100)
    dqs_range = st.sidebar.slider(
        "📊 DQS Score",
        min_value=0,
        max_value=100,
        value=(0, 100),
        help="Data Quality Score (0-100)"
    )
    # Converti back a scala originale
    dqs_range = (dqs_range[0] / 100, dqs_range[1] / 100)
else:
    dqs_range = st.sidebar.slider(
        "📊 DQS Score",
        min_value=int(dqs_min),
        max_value=int(dqs_max),
        value=(int(dqs_min), int(dqs_max)),
        help="Data Quality Score (0-100)"
    )

# Market cap filter
mcap_range = st.sidebar.select_slider(
    "💰 Market Cap",
    options=['All', 'Small (<5B)', 'Mid (5-50B)', 'Large (>50B)'],
    value='All'
)

st.sidebar.markdown("---")
st.sidebar.markdown("📚 **Documentazione**")
st.sidebar.markdown(f"[Technical Deep-Dive](https://moneyfeel.it/julia/analisi-fondamentale-a4.html)")
st.sidebar.markdown(f"[Overview](https://moneyfeel.it/analisi-fondamentale/)")

# ============================================================================
# APPLY FILTERS
# ============================================================================

df_filtered = df.copy()

# Sector filter
if 'All' not in selected_sectors and len(selected_sectors) > 0:
    df_filtered = df_filtered[df_filtered['sector'].isin(selected_sectors)]

# Rating filter
if len(selected_ratings) > 0:
    df_filtered = df_filtered[df_filtered['rating'].isin(selected_ratings)]

# DQS filter
df_filtered = df_filtered[
    (df_filtered['dqs_score'] >= dqs_range[0]) &
    (df_filtered['dqs_score'] <= dqs_range[1])
]

# Market cap filter
if mcap_range != 'All':
    if mcap_range == 'Small (<5B)':
        df_filtered = df_filtered[df_filtered['market_cap'] < 5e9]
    elif mcap_range == 'Mid (5-50B)':
        df_filtered = df_filtered[
            (df_filtered['market_cap'] >= 5e9) &
            (df_filtered['market_cap'] < 50e9)
        ]
    elif mcap_range == 'Large (>50B)':
        df_filtered = df_filtered[df_filtered['market_cap'] >= 50e9]

# ============================================================================
# HEADER
# ============================================================================

# Logo + Title inline
st.markdown(f"""
<div style="display: flex; align-items: center; margin-bottom: 20px;">
    <img src="{MONEYFEEL_LOGO}" style="height: 60px; margin-right: 20px;" alt="moneyfeel logo">
    <div>
        <h1 style="margin: 0; color: {MONEYFEEL_COLORS['pink']};">📊 moneyfeel Fundamental Analysis Engine</h1>
        <p style="margin: 0; color: {MONEYFEEL_COLORS['purple']}; font-size: 1.2em;">Demo Interattiva - Dual-Engine Valuation System</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Key metrics
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    pct_filtered = (len(df_filtered) / len(df) * 100) if len(df) > 0 else 0
    st.metric(
        "🎯 Titoli Filtrati",
        len(df_filtered),
        delta=f"{pct_filtered:.0f}% del totale"
    )

with col2:
    if len(df_filtered) > 0:
        avg_er = df_filtered['expected_return_12m'].mean()
        df_avg_er = df['expected_return_12m'].mean()
    else:
        avg_er = df_avg_er = 0
    
    st.metric(
        "📈 Avg Expected Return",
        f"{avg_er:.1f}%",
        delta=f"{avg_er - df_avg_er:.1f}% vs all"
    )

with col3:
    if len(df_filtered) > 0:
        avg_dqs = df_filtered['dqs_score'].mean()
        df_avg_dqs = df['dqs_score'].mean()
        # Se DQS è 0-1, moltiplica per 100 per display
        if avg_dqs < 10:
            avg_dqs *= 100
            df_avg_dqs *= 100
    else:
        avg_dqs = df_avg_dqs = 0
    
    st.metric(
        "✅ Avg DQS Score",
        f"{avg_dqs:.0f}/100",
        delta=f"{avg_dqs - df_avg_dqs:.0f} vs all"
    )

with col4:
    strong_buy_count = len(df_filtered[df_filtered['rating'] == 'Strong Buy'])
    pct_strong_buy = (strong_buy_count / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
    st.metric(
        "⭐ Strong Buy",
        strong_buy_count,
        delta=f"{pct_strong_buy:.0f}% filtered"
    )

with col5:
    avg_mcap = (df_filtered['market_cap'].mean() / 1e9) if len(df_filtered) > 0 else 0
    st.metric(
        "💰 Avg Market Cap",
        f"${avg_mcap:.1f}B"
    )

st.markdown("---")

# ============================================================================
# CHECK: Dataset vuoto dopo filtri
# ============================================================================

if len(df_filtered) == 0:
    st.warning("⚠️ Nessun titolo corrisponde ai filtri selezionati.")
    st.info("💡 Suggerimento: Riduci i criteri di filtro (es. aumenta range DQS, seleziona più settori)")
    st.stop()

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Overview Table",
    "📈 Scatter Analysis",
    "🎯 Single Stock Deep-Dive",
    "🔧 Technical Details"
])

# ============================================================================
# TAB 1: OVERVIEW TABLE
# ============================================================================

with tab1:
    st.subheader("📋 Valuation Results - Filtered Dataset")
    
    # Display configuration - EXTENDED COLUMNS
    display_cols = [
        'symbol_yahoo', 'company_name', 'sector', 'industry', 'rating',
        'current_price', 'fair_value_lagging', 'fair_value_leading',
        'expected_return_12m', 'target_12m',
        'dqs_score', 'dqs_class',
        'market_cap', 'currency',
        # Methodology weights
        'dcf_weight', 'comps_weight', 'ddm_weight', 'ri_weight',
        # Street data
        'street_analyst_count', 'street_consensus', 'street_high', 'street_low',
        # Fundamentals
        'profit_margin', 'operating_margin', 'roe', 'roa',
        'earnings_growth', 'revenue_growth', 'debt_to_equity', 'beta'
    ]
    
    # Filter only existing columns
    display_cols = [col for col in display_cols if col in df_filtered.columns]
    
    # Format dataframe
    df_display = df_filtered[display_cols].copy()
    
    # Format numeric columns
    if 'market_cap' in df_display.columns:
        df_display['market_cap'] = df_display['market_cap'].apply(lambda x: f"${x/1e9:.2f}B" if pd.notna(x) else "N/A")
    
    if 'expected_return_12m' in df_display.columns:
        df_display['expected_return_12m'] = df_display['expected_return_12m'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
    
    if 'target_12m' in df_display.columns:
        df_display['target_12m'] = df_display['target_12m'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
    
    # Fix DQS display
    if 'dqs_score' in df_display.columns:
        if df_display['dqs_score'].max() < 10:
            df_display['dqs_score'] = df_display['dqs_score'].apply(lambda x: f"{x*100:.0f}" if pd.notna(x) else "N/A")
        else:
            df_display['dqs_score'] = df_display['dqs_score'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "N/A")
    
    # Format percentages
    pct_cols = ['dcf_weight', 'comps_weight', 'ddm_weight', 'ri_weight',
                'profit_margin', 'operating_margin', 'roe', 'roa',
                'earnings_growth', 'revenue_growth']
    
    for col in pct_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A")
    
    # Format decimals
    if 'debt_to_equity' in df_display.columns:
        df_display['debt_to_equity'] = df_display['debt_to_equity'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    
    if 'beta' in df_display.columns:
        df_display['beta'] = df_display['beta'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    
    if 'street_analyst_count' in df_display.columns:
        df_display['street_analyst_count'] = df_display['street_analyst_count'].apply(lambda x: f"{int(x)}" if pd.notna(x) else "N/A")
    
    if 'street_consensus' in df_display.columns:
        df_display['street_consensus'] = df_display['street_consensus'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
    
    if 'street_high' in df_display.columns:
        df_display['street_high'] = df_display['street_high'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
    
    if 'street_low' in df_display.columns:
        df_display['street_low'] = df_display['street_low'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
    
    # Rename columns for display
    column_rename = {
        'symbol_yahoo': 'Symbol',
        'company_name': 'Company',
        'sector': 'Sector',
        'industry': 'Industry',
        'rating': 'Rating',
        'current_price': 'Price ($)',
        'fair_value_lagging': 'FV Lag ($)',
        'fair_value_leading': 'FV Lead ($)',
        'expected_return_12m': 'ER 12M (%)',
        'target_12m': 'Target ($)',
        'dqs_score': 'DQS',
        'dqs_class': 'DQS Class',
        'market_cap': 'Market Cap',
        'currency': 'Curr',
        'dcf_weight': 'DCF %',
        'comps_weight': 'Comps %',
        'ddm_weight': 'DDM %',
        'ri_weight': 'RI %',
        'street_analyst_count': 'Analysts',
        'street_consensus': 'Consensus ($)',
        'street_high': 'High ($)',
        'street_low': 'Low ($)',
        'profit_margin': 'Profit Mgn',
        'operating_margin': 'Op Mgn',
        'roe': 'ROE',
        'roa': 'ROA',
        'earnings_growth': 'Earn Gr',
        'revenue_growth': 'Rev Gr',
        'debt_to_equity': 'D/E',
        'beta': 'Beta'
    }
    
    df_display.rename(columns={k: v for k, v in column_rename.items() if k in df_display.columns}, inplace=True)
    
    # Display with color coding (moneyfeel colors)
    def color_rating(val):
        rating_bg_colors = {
            'Strong Buy': f'background-color: rgba(77, 178, 230, 0.3)',  # Blue
            'Buy': f'background-color: rgba(154, 78, 178, 0.3)',  # Purple
            'Hold': 'background-color: #fff3cd',
            'Sell': 'background-color: #f8d7da',
            'Strong Sell': f'background-color: rgba(251, 0, 134, 0.3)'  # Pink
        }
        return rating_bg_colors.get(val, '')
    
    styled_df = df_display.sort_values('ER 12M (%)', ascending=False)
    
    if 'Rating' in styled_df.columns:
        styled_df = styled_df.style.map(color_rating, subset=['Rating'])
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=500
    )
    
    # Download button
    csv = df_filtered[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f'moneyfeel_valuation_demo_{pd.Timestamp.now().strftime("%Y%m%d")}.csv',
        mime='text/csv'
    )

# ============================================================================
# TAB 2: SCATTER ANALYSIS
# ============================================================================

with tab2:
    st.subheader("📈 Fair Value vs Price Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        analysis_mode = st.radio(
            "Valuation Mode",
            options=['Leading', 'Lagging', 'Both'],
            help="Leading = predictive (NTM), Lagging = conservative (TTM)"
        )
    
    with col1:
        # Scatter plot con colori moneyfeel
        if analysis_mode in ['Leading', 'Both']:
            fig = px.scatter(
                df_filtered,
                x='current_price',
                y='fair_value_leading',
                size='market_cap',
                color='rating',
                hover_name='symbol_yahoo',
                hover_data={
                    'company_name': True,
                    'sector': True,
                    'expected_return_12m': ':.1f',
                    'dqs_score': ':.0f',
                    'current_price': ':.2f',
                    'fair_value_leading': ':.2f',
                    'market_cap': False
                },
                title="Current Price vs Fair Value (Leading Engine)",
                labels={
                    'current_price': 'Current Price ($)',
                    'fair_value_leading': 'Fair Value - Leading ($)'
                },
                color_discrete_map=RATING_COLORS,
                height=500
            )
            
            # Add diagonal line (moneyfeel purple)
            max_val = max(
                df_filtered['current_price'].max(),
                df_filtered['fair_value_leading'].max()
            )
            fig.add_trace(
                go.Scatter(
                    x=[0, max_val],
                    y=[0, max_val],
                    mode='lines',
                    line=dict(dash='dash', color=MONEYFEEL_COLORS['purple'], width=2),
                    name='Fair Value = Price',
                    showlegend=True
                )
            )
            
            fig.update_layout(
                xaxis_title="Current Price ($)",
                yaxis_title="Fair Value - Leading ($)",
                legend_title="Rating",
                plot_bgcolor=MONEYFEEL_COLORS['gray']
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Divergence analysis
    st.subheader("🔍 Lagging vs Leading Divergence")
    
    df_filtered_copy = df_filtered.copy()
    df_filtered_copy['divergence'] = (
        (df_filtered_copy['fair_value_leading'] - df_filtered_copy['fair_value_lagging']) / 
        df_filtered_copy['fair_value_lagging'] * 100
    )
    
    fig2 = px.histogram(
        df_filtered_copy,
        x='divergence',
        nbins=30,
        title="Distribution of Lagging-Leading Fair Value Divergence",
        labels={'divergence': 'Divergence (%)'},
        color='rating',
        color_discrete_map=RATING_COLORS
    )
    
    fig2.add_vline(
        x=0,
        line_dash="dash",
        line_color=MONEYFEEL_COLORS['pink'],
        annotation_text="No divergence"
    )
    
    fig2.update_layout(plot_bgcolor=MONEYFEEL_COLORS['gray'])
    
    st.plotly_chart(fig2, use_container_width=True)
    
    st.info(
        "💡 **Divergence Insight:** Divergenza positiva (Leading > Lagging) indica "
        "sentiment ottimistico su crescita futura. Divergenza negativa suggerisce cautela."
    )
    
    # Expected Return distribution
    st.subheader("💰 Expected Return 12M Distribution")
    
    fig3 = px.histogram(
        df_filtered,
        x='expected_return_12m',
        nbins=20,
        title="Expected Return 12M Distribution by Rating",
        labels={'expected_return_12m': 'Expected Return 12M (%)'},
        color='rating',
        color_discrete_map=RATING_COLORS
    )
    
    fig3.update_layout(plot_bgcolor=MONEYFEEL_COLORS['gray'])
    
    st.plotly_chart(fig3, use_container_width=True)

# ============================================================================
# TAB 3: SINGLE STOCK DEEP-DIVE
# ============================================================================

with tab3:
    st.subheader("🎯 Deep Dive - Single Stock Analysis")
    
    # Stock selector
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Safe format function
        def format_stock_option(symbol):
            try:
                match = df_filtered[df_filtered['symbol_yahoo'] == symbol]
                if len(match) > 0:
                    return f"{symbol} - {match['company_name'].iloc[0]}"
                return symbol
            except:
                return symbol
        
        selected_symbol = st.selectbox(
            "🔍 Seleziona titolo:",
            options=df_filtered['symbol_yahoo'].unique(),
            format_func=format_stock_option
        )
    
    # Safe stock selection
    stock_match = df_filtered[df_filtered['symbol_yahoo'] == selected_symbol]
    
    if len(stock_match) == 0:
        st.error(f"❌ Titolo {selected_symbol} non trovato nel dataset filtrato.")
        st.stop()
    
    stock = stock_match.iloc[0]
    
    # Header info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Company", stock['company_name'])
    with col2:
        st.metric("Sector", stock['sector'])
    with col3:
        st.metric("Market Cap", f"${stock['market_cap']/1e9:.2f}B")
    with col4:
        dqs_display = stock['dqs_score'] * 100 if stock['dqs_score'] < 10 else stock['dqs_score']
        st.metric("DQS Score", f"{dqs_display:.0f}/100")
    
    st.markdown("---")
    
    # Valuation metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Current Price",
            f"${stock['current_price']:.2f}"
        )
    
    with col2:
        fv_lag = stock['fair_value_lagging']
        upside_lag = ((fv_lag / stock['current_price']) - 1) * 100
        st.metric(
            "FV (Lagging)",
            f"${fv_lag:.2f}",
            delta=f"{upside_lag:+.1f}%"
        )
    
    with col3:
        fv_lead = stock['fair_value_leading']
        upside_lead = ((fv_lead / stock['current_price']) - 1) * 100
        st.metric(
            "FV (Leading)",
            f"${fv_lead:.2f}",
            delta=f"{upside_lead:+.1f}%"
        )
    
    with col4:
        st.metric(
            "Target 12M",
            f"${stock['target_12m']:.2f}" if pd.notna(stock['target_12m']) else "N/A"
        )
    
    with col5:
        st.metric(
            "Expected Return",
            f"{stock['expected_return_12m']:.1f}%"
        )
    
    # Rating display (moneyfeel colors)
    rating = stock['rating']
    rating_color = RATING_COLORS.get(rating, MONEYFEEL_COLORS['purple'])
    
    st.markdown(
        f"### Rating: <span style='color: {rating_color};'>"
        f"**{rating}**</span>",
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    # Methodology breakdown
    st.subheader("🔬 Valuation Methodology Breakdown")
    
    methods = ['dcf', 'comps', 'ddm', 'ri', 'sotp']
    method_labels = ['DCF', 'COMPS', 'DDM', 'RI', 'SOTP']
    method_values = [stock.get(f'{m}_fv', 0) for m in methods]
    method_weights = [stock.get(f'{m}_weight', 0) for m in methods]
    
    # Filter non-zero methods
    valid_methods = [
        (label, val, weight) 
        for label, val, weight in zip(method_labels, method_values, method_weights)
        if weight > 0 and pd.notna(val) and val > 0
    ]
    
    if len(valid_methods) > 0:
        labels, values, weights = zip(*valid_methods)
        
        # Moneyfeel colors for bars
        bar_colors = [MONEYFEEL_COLORS['blue'], MONEYFEEL_COLORS['purple'], 
                     MONEYFEEL_COLORS['pink'], '#4db2e6', '#9a4eb2']
        
        fig4 = go.Figure(data=[
            go.Bar(
                x=list(labels),
                y=list(values),
                text=[f"${v:.2f}<br>({w*100:.0f}%)" for v, w in zip(values, weights)],
                textposition='auto',
                marker_color=bar_colors[:len(labels)]
            )
        ])
        
        fig4.update_layout(
            title="Fair Value by Methodology (Leading Engine)",
            yaxis_title="Fair Value ($)",
            showlegend=False,
            height=400,
            plot_bgcolor=MONEYFEEL_COLORS['gray']
        )
        
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning("⚠️ Methodology breakdown non disponibile per questo titolo")
    
    # DQS breakdown
    dqs_display = stock['dqs_score'] * 100 if stock['dqs_score'] < 10 else stock['dqs_score']
    st.subheader(f"✅ Data Quality Score: {dqs_display:.0f}/100")
    
    dqs_components = {
        'Fundamentals': stock.get('dqs_fundamentals', dqs_display * 0.95),
        'Analyst Coverage': stock.get('dqs_analyst', 80),
        'Volatility': stock.get('dqs_volatility', 85),
        'Liquidity': stock.get('dqs_liquidity', 90),
        'Governance': stock.get('dqs_governance', 80),
        'Accounting': stock.get('dqs_accounting', 85)
    }
    
    # Fix scale if needed
    if max(dqs_components.values()) < 10:
        dqs_components = {k: v*100 for k, v in dqs_components.items()}
    
    fig5 = go.Figure(data=go.Scatterpolar(
        r=list(dqs_components.values()),
        theta=list(dqs_components.keys()),
        fill='toself',
        line_color=MONEYFEEL_COLORS['purple'],
        fillcolor='rgba(154, 78, 178, 0.3)'  # Purple with 30% opacity
    ))
    
    fig5.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False,
        title="DQS Components Breakdown",
        height=400
    )
    
    st.plotly_chart(fig5, use_container_width=True)
    
    # Fundamentals
    st.subheader("📊 Key Fundamentals")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Profitability**")
        if pd.notna(stock.get('profit_margin')):
            st.metric("Profit Margin", f"{stock['profit_margin']*100:.1f}%")
        if pd.notna(stock.get('operating_margin')):
            st.metric("Operating Margin", f"{stock['operating_margin']*100:.1f}%")
        if pd.notna(stock.get('gross_margin')):
            st.metric("Gross Margin", f"{stock['gross_margin']*100:.1f}%")
    
    with col2:
        st.markdown("**Returns**")
        if pd.notna(stock.get('roe')):
            st.metric("ROE", f"{stock['roe']*100:.1f}%")
        if pd.notna(stock.get('roa')):
            st.metric("ROA", f"{stock['roa']*100:.1f}%")
        if pd.notna(stock.get('beta')):
            st.metric("Beta", f"{stock['beta']:.2f}")
    
    with col3:
        st.markdown("**Growth**")
        if pd.notna(stock.get('earnings_growth')):
            st.metric("Earnings Growth", f"{stock['earnings_growth']*100:.1f}%")
        if pd.notna(stock.get('revenue_growth')):
            st.metric("Revenue Growth", f"{stock['revenue_growth']*100:.1f}%")
        if pd.notna(stock.get('debt_to_equity')):
            st.metric("Debt/Equity", f"{stock['debt_to_equity']:.2f}")

# ============================================================================
# TAB 4: TECHNICAL DETAILS
# ============================================================================

with tab4:
    st.subheader("🔧 Technical Implementation Details")
    
    st.markdown(f"""
    ### Dual-Engine Architecture
    
    Il sistema implementa due engine valutativi paralleli con colori distintivi moneyfeel:
    
    <div style="background: linear-gradient(90deg, {MONEYFEEL_COLORS['purple']}20, {MONEYFEEL_COLORS['blue']}20); 
                padding: 20px; border-radius: 10px; margin: 20px 0;">
    
    **🔵 LAGGING Engine (Conservative)**  
    <span style="color: {MONEYFEEL_COLORS['blue']}">■</span> Data source: TTM (Trailing Twelve Months) - dati storici confermati  
    <span style="color: {MONEYFEEL_COLORS['blue']}">■</span> WACC calculation: Country-aware con Damodaran ERP/CRP  
    <span style="color: {MONEYFEEL_COLORS['blue']}">■</span> Terminal growth: Conservativo (DM 2.0%, EM 2.25%)  
    <span style="color: {MONEYFEEL_COLORS['blue']}">■</span> Use case: IB-grade valuation, risk management floors  
    
    **🟣 LEADING Engine (Predictive)**  
    <span style="color: {MONEYFEEL_COLORS['purple']}">■</span> Data source: NTM (Next Twelve Months) - stime consensus  
    <span style="color: {MONEYFEEL_COLORS['purple']}">■</span> WACC calculation: Identico ma con terminal growth ottimistico  
    <span style="color: {MONEYFEEL_COLORS['purple']}">■</span> Terminal growth: +50-100 bps vs lagging  
    <span style="color: {MONEYFEEL_COLORS['purple']}">■</span> Street target weight: 30% (blend con analyst consensus)  
    <span style="color: {MONEYFEEL_COLORS['purple']}">■</span> Use case: Entry points, alpha signals, trading  
    
    </div>
    
    ### Valuation Methodologies
    
    5 metodologie con triangolazione YAML-driven:
    
    1. **DCF (Discounted Cash Flow)** - 2-stage Free Cash Flow to Firm
    2. **COMPS (Comparable Companies)** - Peer selection automatica multi-metric
    3. **DDM (Dividend Discount Model)** - Gordon model dividend-paying stocks
    4. **RI (Residual Income)** - Book value + PV excess returns
    5. **SOTP (Sum-of-the-Parts)** - Multi-segment valuation
    
    ### Data Quality Score (DQS)
    
    Sistema proprietario 6-dimensionale (0-100) che determina:
    - Confidence interval sul fair value finale
    - Guardrail elasticity (low DQS → tighter clamp)
    - Methodology weighting dynamic
    
    ### Coverage & Performance
    
    **Demo Dataset:** 50 high-quality stocks (DQS > 75)  
    **Full Production:** 55,000+ global equities  
    **Single query:** 300-500ms (COMPS + 5 engines + triangulation)  
    **Daily batch:** 6-8 hours (AWS ECS Fargate)  
    
    ### Update Schedule
    
    - **DAILY:** EOD prices, leading indicators, full valuation run
    - **WEEKLY:** Analyst data (estimates, ratings, upgrades/downgrades)
    - **MONTHLY:** Risk-free rates, ERP/CRP Damodaran, ESG baselines
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Documentation links
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📚 Full Documentation")
        st.markdown(f"[🔗 Technical Deep-Dive (10 pages)](https://moneyfeel.it/julia/analisi-fondamentale-a4.html)")
        st.markdown(f"[🔗 Landing Page Overview](https://moneyfeel.it/analisi-fondamentale/)")
    
    with col2:
        st.markdown("### 📧 Contact")
        st.markdown("**moneyfeel.it**")
        st.markdown("Email: luca.stagnitta@moneyfeel.it")
        st.markdown("Demo version - For evaluation purposes only")

# ============================================================================
# FOOTER BRANDIZZATO
# ============================================================================

st.markdown("---")
st.markdown(f"""
<div class="moneyfeel-footer">
    <img src="{MONEYFEEL_LOGO}" style="width: 150px; margin-bottom: 15px;" alt="moneyfeel logo"><br>
    <strong>moneyfeel Fundamental Analysis Engine</strong><br>
    Interactive Demo • Dual-Engine Valuation System<br>
    5 Methodologies • Data Quality Scoring<br>
    <br>
    <span style="color: {MONEYFEEL_COLORS['purple']};">Contact: luca.stagnitta@moneyfeel.it</span><br>
    <a href="https://moneyfeel.it/analisi-fondamentale/" style="color: {MONEYFEEL_COLORS['blue']};">
        Documentation: moneyfeel.it/analisi-fondamentale
    </a>
</div>
""", unsafe_allow_html=True)