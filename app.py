import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np                      
import plotly.graph_objects as go       
import yfinance as yf
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore
import random

# 1. 网页全局设置 (扩展为宽屏模式)
st.set_page_config(page_title="Global Macro Dashboard", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 2. Custom CSS for High-End Legal/Corporate Dark Mode
# ==========================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vollkorn:ital,wght@0,400..900;1,400..900&display=swap');
    html, body, [class*="css"] { font-family: 'Vollkorn', Georgia, 'Times New Roman', Times, serif !important; }
    h1 { color: #FFFFFF !important; font-weight: 700 !important; font-size: 3rem !important; letter-spacing: -0.01em !important; margin-bottom: 1.5rem !important; }
    div[data-testid="stTabs"] > div[role="tablist"] { display: flex !important; width: 100% !important; justify-content: space-between !important; gap: 0.5rem !important; border-bottom: none !important; padding-bottom: 1rem !important; }
    button[data-baseweb="tab"] { flex: 1 !important; display: flex !important; justify-content: center !important; align-items: center !important; padding: 0.6rem 1rem !important; background-color: transparent !important; border-radius: 12px !important; border: none !important; transition: all 0.25s ease-in-out !important; }
    button[data-baseweb="tab"] > div { font-family: 'Vollkorn', serif !important; font-size: 1.15rem !important; font-weight: 600 !important; color: #71717A !important; }
    button[data-baseweb="tab"]:hover { background-color: rgba(255, 255, 255, 0.08) !important; }
    button[data-baseweb="tab"]:hover > div { color: #D4D4D8 !important; }
    button[data-baseweb="tab"][aria-selected="true"] { background-color: rgba(212, 175, 55, 0.15) !important; }
    button[data-baseweb="tab"][aria-selected="true"] > div { color: #FFFFFF !important; font-weight: 800 !important; }
    div[data-testid="stSelectbox"] label p { font-family: 'Vollkorn', serif !important; font-size: 1.2rem !important; font-weight: 700 !important; color: #FFFFFF !important; margin-bottom: 0.5rem !important; }
    div[data-baseweb="select"] > div { background-color: #18181B !important; border: 1px solid #3F3F46 !important; }
    div[data-baseweb="select"] span { font-family: 'Vollkorn', serif !important; }
    h3 { font-family: 'Vollkorn', serif !important; color: #F4F4F5 !important; font-weight: 700 !important; margin-top: 2rem !important; padding-bottom: 0.5rem !important; border-bottom: 1px solid #27272A !important; }
    [data-testid="stSidebar"] p { font-family: 'Vollkorn', serif !important; color: #D4D4D8 !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. Sidebar & Header
# ==========================================
st.sidebar.title("Navigation Menu")
st.sidebar.markdown("Welcome to the **Global Macro Asset Allocation** Engine.")
st.sidebar.divider()
st.sidebar.info("Tip: Interact with the charts by hovering over the heatmap to see correlation values, or drag to zoom in.")

st.title("Global Macro Asset Allocation Dashboard")
st.markdown("Track the performance, risk, and correlation of global GICS sectors and macro hedges.")

# ==========================================
# 4. Load Data & Shared Resources
# ==========================================
@st.cache_data 
def load_data():
    summary = pd.read_csv("data/processed/global_universe_summary.csv", index_col=0)
    returns = pd.read_csv("data/processed/global_universe_returns.csv", index_col=0, parse_dates=True)
    return summary, returns

summary_data, returns_data = load_data()

@st.cache_resource
def load_nlp_model():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

sia = load_nlp_model()

# 统一的模拟新闻数据库 (用于替代不稳定的 yfinance 接口)
MOCK_NEWS_POOL = {
    'XLK': ["Tech giants report record earnings, surging past estimates", "New AI chips drive explosive growth in tech sector", "Regulatory concerns weigh heavily on tech valuations"],
    'XLV': ["FDA approves breakthrough Alzheimer's drug", "Healthcare sector faces headwind from new pricing regulations", "Biotech firms see massive influx of venture capital"],
    'XLF': ["Federal Reserve signals potential rate cuts, boosting bank stocks", "Major banks report lower than expected loan growth", "Financial sector rallies on strong consumer spending data"],
    'XLE': ["Oil prices plummet amid global oversupply fears", "Energy companies pivot aggressively towards renewables", "Geopolitical tensions trigger spike in crude oil futures"],
    'GLD': ["Investors flock to gold as inflation fears rise", "Strong US dollar puts downward pressure on gold prices", "Central banks continue heavy buying of gold reserves"],
    'TLT': ["Bond yields drop as investors seek safe havens", "Treasury auction sees weak demand, driving yields higher", "Fed's dovish tone sparks rally in long-term bonds"]
}
GENERIC_NEWS = [
    "Market shows cautious optimism ahead of key economic data",
    "Sector experiences mild correction following recent rally",
    "Analysts upgrade growth forecast based on solid fundamentals"
]

# ==========================================
# 5. Tabs Layout
# ==========================================
tab_pulse, tab_heatmap, tab_database, tab_opt = st.tabs([
    "Live Market Pulse",     
    "Correlation Heatmap",  
    "Risk/Return Database", 
    "Portfolio Sandbox"      
])

with tab_pulse:
    st.markdown("### 🌐 Global Market Pulse & AI Takeaway")
    
    market_tickers = {
        'Technology': 'XLK', 'Healthcare': 'XLV', 'Financials': 'XLF',
        'Consumer Discr': 'XLY', 'Consumer Staples': 'XLP', 'Energy': 'XLE',
        'Industrials': 'XLI', 'Materials': 'XLB', 'Utilities': 'XLU',
        'Real Estate': 'XLRE', 'Communications': 'XLC', 
        'Safe Haven: Gold': 'GLD', 'Safe Haven: Bonds': 'TLT'
    }
    
    @st.cache_data(ttl=3600)
    def fetch_live_performance():
        tickers = list(market_tickers.values())
        data = yf.download(tickers, period="5d")['Close']
        returns = (data.iloc[-1] - data.iloc[0]) / data.iloc[0]
        return returns

    with st.spinner("Fetching live market data..."):
        try:
            live_returns = fetch_live_performance()
            perf_df = pd.DataFrame({
                'Sector': list(market_tickers.keys()),
                'Ticker': list(market_tickers.values()),
                'Performance': [live_returns[t] for t in market_tickers.values()]
            })
            perf_df['Weight'] = 1 
            perf_df['Label'] = perf_df['Sector'] + "<br>" + perf_df['Performance'].apply(lambda x: f"{x*100:.2f}%")
            
            best_sector = perf_df.loc[perf_df['Performance'].idxmax()]
            worst_sector = perf_df.loc[perf_df['Performance'].idxmin()]
            
            st.info(f"**💡 Quant Insight:** In the trailing 5 days, **{best_sector['Sector']}** is leading the market ({best_sector['Performance']*100:.2f}%), while **{worst_sector['Sector']}** is lagging. Consider taking profits from overextended winners and exploring value in oversold sectors.")

            fig_tree = px.treemap(
                perf_df, path=[px.Constant("Global Macro Universe"), 'Label'], values='Weight',
                color='Performance', color_continuous_scale=['#FF4B4B', '#18181B', '#00C853'], color_continuous_midpoint=0
            )
            fig_tree.update_layout(margin=dict(t=20, l=0, r=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', coloraxis_showscale=False)
            fig_tree.update_traces(textfont=dict(family="Vollkorn", size=18, color="white"), textinfo="label", hovertemplate="<b>%{label}</b><br>Return: %{color:.2%}<extra></extra>")
            st.plotly_chart(fig_tree, use_container_width=True)
            
        except Exception as e:
            st.warning("Live data fetching is currently unavailable.")
            st.write(e)

    # NLP 新闻情绪 Alpha 因子分析 (首页展示版 - 使用模拟器)
    st.markdown("---")
    st.markdown("### 🧠 AI NLP Sentiment Alpha Engine")
    
    col_sel, col_space = st.columns([1, 2])
    with col_sel:
        target_sector = st.selectbox("Select Sector for Real-time News Analysis:", list(market_tickers.keys()))
        target_ticker = market_tickers[target_sector]

    with st.spinner(f"Processing NLP scores for {target_ticker}..."):
        # 使用模拟数据提取新闻
        news_titles = MOCK_NEWS_POOL.get(target_ticker, GENERIC_NEWS)
        sampled_titles = random.sample(news_titles, min(3, len(news_titles)))
        
        sentiment_scores = []
        news_items = []
        
        for title in sampled_titles:
            score = sia.polarity_scores(title)['compound']
            sentiment_scores.append(score)
            sentiment_label = "🟢 Bullish" if score > 0.05 else "🔴 Bearish" if score < -0.05 else "⚪ Neutral"
            news_items.append(f"**[Market Insight]** {title}  \n*AI Score: {score:.2f} ({sentiment_label})*")
        
        avg_score = np.mean(sentiment_scores) if sentiment_scores else 0.0
        
        col_news, col_metric = st.columns([2, 1])
        with col_news:
            st.markdown("#### 📰 Live Headlines")
            for news in news_items:
                st.write(news)
                st.markdown("<hr style='margin: 0.5em 0; border-color: #3F3F46;'>", unsafe_allow_html=True)
        
        with col_metric:
            st.markdown("#### ⚡ Alpha Factor (Sentiment)")
            delta_text = "Bullish Momentum" if avg_score > 0.05 else "Bearish Pressure" if avg_score < -0.05 else "Neutral"
            delta_color = "normal" if avg_score > 0.05 else "inverse" if avg_score < -0.05 else "off"
                
            st.metric(label=f"{target_ticker} Sentiment Score", value=f"{avg_score:.2f}", delta=delta_text, delta_color=delta_color)
            st.info(f"**How to use this Alpha?**\nA score of **{avg_score:.2f}** acts as a quantitative sentiment overlay. In our Portfolio Sandbox, this score will mathematically adjust optimal portfolio weights.")

with tab_heatmap:
    drilldown_dict = {
        "Macro Overview: 13 Core Global Assets": ['XLK', 'XLV', 'XLF', 'XLY', 'XLP', 'XLE', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC', 'TLT', 'GLD'],
        "Sector Deep Dive: Information Technology": ['XLK', 'SMH', 'IGV', 'CIBR', 'SKYY'],
        "Sector Deep Dive: Healthcare & Biotech": ['XLV', 'IBB', 'IHI', 'PPH'],
        "Sector Deep Dive: Financial Services": ['XLF', 'KRE', 'IAI', 'KIE'],
        "Sector Deep Dive: Consumer Discretionary": ['XLY', 'XRT', 'XHB', 'CARZ'],
        "Sector Deep Dive: Energy & Clean Tech": ['XLE', 'XOP', 'OIH', 'ICLN']
    }
    selected_view = st.selectbox("Select Analysis Universe:", list(drilldown_dict.keys()))
    st.markdown(f"### Correlation Matrix | {selected_view}")
    
    selected_tickers = drilldown_dict[selected_view]
    dynamic_returns = returns_data[selected_tickers]
    fig = px.imshow(dynamic_returns.corr(), text_auto=".2f", aspect="auto", color_continuous_scale="RdYlBu_r", zmin=-0.5, zmax=1)
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

with tab_database:
    st.subheader("Asset Risk & Return Summary")
    display_data = summary_data.reset_index()
    display_data = display_data[['Sector', 'Ticker', 'Ann_Return', 'Volatility', 'Sharpe_Ratio']]
    display_data.set_index('Sector', inplace=True)
    st.dataframe(display_data, use_container_width=True, height=500)

with tab_opt:
    st.markdown("### 🧪 AI-Augmented Portfolio Sandbox")
    st.markdown("Simulate optimal portfolios using historical data **overlayed with real-time NLP Sentiment Alpha**.")
    
    col_input1, col_input2 = st.columns([2, 1])
    with col_input1:
        all_tickers = returns_data.columns.tolist()
        selected_assets = st.multiselect("Select Assets for Optimization:", all_tickers, default=['XLK', 'TLT', 'GLD', 'XLE'])
    with col_input2:
        alpha_weight = st.slider("NLP Sentiment Alpha Weight", 0.0, 0.1, 0.05, 0.01)

    if len(selected_assets) >= 2:
        with st.spinner("Running Monte Carlo Simulation with AI overlay..."):
            df_selected = returns_data[selected_assets]
            base_mean_returns = df_selected.mean() * 252
            cov_matrix = df_selected.cov() * 252
            
            # 使用统一的模拟新闻数据打分
            current_sentiments = {}
            for ticker in selected_assets:
                news_titles = MOCK_NEWS_POOL.get(ticker, GENERIC_NEWS)
                sampled_titles = random.sample(news_titles, min(3, len(news_titles)))
                scores = [sia.polarity_scores(title)['compound'] for title in sampled_titles]
                current_sentiments[ticker] = np.mean(scores) if scores else 0.0
            
            adjusted_returns = base_mean_returns.copy()
            for ticker in selected_assets:
                adjusted_returns[ticker] += (current_sentiments[ticker] * alpha_weight)
            
            st.markdown("#### 📡 Real-time Sentiment Alpha Applied")
            sentiment_df = pd.DataFrame.from_dict(current_sentiments, orient='index', columns=['NLP Score'])
            sentiment_df['Alpha Boost'] = sentiment_df['NLP Score'] * alpha_weight
            sentiment_df['Alpha Boost'] = sentiment_df['Alpha Boost'].apply(lambda x: f"{x*100:+.2f}%")
            st.dataframe(sentiment_df.T, use_container_width=True)

            num_portfolios = 5000
            results = np.zeros((3, num_portfolios))
            weights_record = []
            
            for i in range(num_portfolios):
                weights = np.random.random(len(selected_assets))
                weights /= np.sum(weights)
                weights_record.append(weights)
                
                portfolio_return = np.sum(adjusted_returns * weights)
                portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                results[0,i] = portfolio_std_dev
                results[1,i] = portfolio_return
                results[2,i] = portfolio_return / portfolio_std_dev
                
            max_sharpe_idx = np.argmax(results[2])
            min_vol_idx = np.argmin(results[0])
            
            fig_opt = go.Figure()
            fig_opt.add_trace(go.Scatter(x=results[0,:], y=results[1,:], mode='markers', marker=dict(size=4, color=results[2,:], colorscale='Viridis', showscale=True), name='Simulated Portfolios', hoverinfo='none'))
            fig_opt.add_trace(go.Scatter(x=[results[0, max_sharpe_idx]], y=[results[1, max_sharpe_idx]], mode='markers+text', marker=dict(color='#D4AF37', size=16, symbol='star'), name='Max Sharpe', text=['Max Sharpe (AI-Adjusted)'], textposition="top center"))
            fig_opt.update_layout(xaxis_title="Predicted Volatility (Risk)", yaxis_title="AI-Adjusted Expected Return", height=500, margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#A1A1AA"))
            st.plotly_chart(fig_opt, use_container_width=True)
            
            col_ms, col_mv = st.columns(2)
            with col_ms:
                st.info("**🚀 AI-Max Sharpe Portfolio**")
                df_max_sharpe = pd.DataFrame({'Asset': selected_assets, 'Weight': weights_record[max_sharpe_idx]})
                df_max_sharpe['Weight'] = df_max_sharpe['Weight'].apply(lambda x: f"{x*100:.2f}%")
                st.dataframe(df_max_sharpe.set_index('Asset'), use_container_width=True)
            with col_mv:
                st.info("**🛡️ Min Volatility Portfolio**")
                df_min_vol = pd.DataFrame({'Asset': selected_assets, 'Weight': weights_record[min_vol_idx]})
                df_min_vol['Weight'] = df_min_vol['Weight'].apply(lambda x: f"{x*100:.2f}%")
                st.dataframe(df_min_vol.set_index('Asset'), use_container_width=True)

    else:
        st.warning("Please select at least 2 assets to run the optimization.")