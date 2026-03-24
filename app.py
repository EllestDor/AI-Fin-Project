import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np                      
import plotly.graph_objects as go       
import yfinance as yf
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore
import random
from datetime import datetime

# ==========================================
# 1. 网页全局设置 (必须在最前面)
# ==========================================
st.set_page_config(page_title="Global Macro Dashboard", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 2. 侧边栏与主题切换 (Light/Dark Mode)
# ==========================================
st.sidebar.title("Navigation Menu")
st.sidebar.markdown("Welcome to the **Global Macro Asset Allocation** Engine.")
st.sidebar.divider()

# 添加主题切换开关
theme_choice = st.sidebar.radio("UI Theme:", ["Dark Mode 🌙", "Light Mode ☀️"])

# 根据选择动态生成 CSS
if theme_choice == "Dark Mode 🌙":
    bg_color = "#0E1117"
    text_color = "#FFFFFF"
    sub_text_color = "#D4D4D8"
    tab_bg = "transparent"
    tab_hover = "rgba(255, 255, 255, 0.08)"
    tab_selected = "rgba(212, 175, 55, 0.15)"
    select_bg = "#18181B"
else:
    bg_color = "#F4F4F5"
    text_color = "#18181B"
    sub_text_color = "#3F3F46"
    tab_bg = "transparent"
    tab_hover = "rgba(0, 0, 0, 0.05)"
    tab_selected = "rgba(212, 175, 55, 0.2)"
    select_bg = "#FFFFFF"

# 注入动态 CSS
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vollkorn:ital,wght@0,400..900;1,400..900&display=swap');
    
    /* 强制重写 Streamlit 根背景色 */
    .stApp {{
        background-color: {bg_color} !important;
    }}
    
    html, body, [class*="css"] {{ font-family: 'Vollkorn', Georgia, 'Times New Roman', Times, serif !important; color: {text_color} !important; }}
    h1, h2, h3, h4, h5, h6 {{ color: {text_color} !important; font-family: 'Vollkorn', serif !important; }}
    
    div[data-testid="stTabs"] > div[role="tablist"] {{ display: flex !important; width: 100% !important; justify-content: space-between !important; gap: 0.5rem !important; border-bottom: none !important; padding-bottom: 1rem !important; }}
    button[data-baseweb="tab"] {{ flex: 1 !important; display: flex !important; justify-content: center !important; align-items: center !important; padding: 0.6rem 1rem !important; background-color: {tab_bg} !important; border-radius: 12px !important; border: none !important; transition: all 0.25s ease-in-out !important; }}
    button[data-baseweb="tab"] > div {{ font-size: 1.15rem !important; font-weight: 600 !important; color: {sub_text_color} !important; }}
    button[data-baseweb="tab"]:hover {{ background-color: {tab_hover} !important; }}
    button[data-baseweb="tab"]:hover > div {{ color: {text_color} !important; }}
    button[data-baseweb="tab"][aria-selected="true"] {{ background-color: {tab_selected} !important; }}
    button[data-baseweb="tab"][aria-selected="true"] > div {{ color: {text_color} !important; font-weight: 800 !important; }}
    
    div[data-testid="stSelectbox"] label p {{ font-size: 1.2rem !important; font-weight: 700 !important; color: {text_color} !important; margin-bottom: 0.5rem !important; }}
    div[data-baseweb="select"] > div {{ background-color: {select_bg} !important; border: 1px solid #3F3F46 !important; }}
    
    [data-testid="stSidebar"] p {{ color: {sub_text_color} !important; }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. 核心计算引擎：实时数据抓取与处理
# ==========================================
st.title("Global Macro Asset Allocation Dashboard")

# 定义全局宇宙资产字典
market_tickers = {
    'Technology': 'XLK', 'Healthcare': 'XLV', 'Financials': 'XLF',
    'Consumer Discr': 'XLY', 'Consumer Staples': 'XLP', 'Energy': 'XLE',
    'Industrials': 'XLI', 'Materials': 'XLB', 'Utilities': 'XLU',
    'Real Estate': 'XLRE', 'Communications': 'XLC', 
    'Gold': 'GLD', 'Long Bonds': 'TLT', 'Semiconductors': 'SMH', 'Cloud': 'SKYY'
}

@st.cache_data(ttl=3600) # 缓存1小时，防止频繁请求被封
def fetch_and_calculate_live_data():
    tickers = list(market_tickers.values())
    
    # 1. 抓取过去 1 年的历史日线数据
    raw_data = yf.download(tickers, period="1y")['Close']
    
    # 2. 实时计算日收益率
    daily_returns = raw_data.pct_change().dropna()
    
    # 3. 实时计算年化收益率和波动率 (基于 252 个交易日)
    ann_return = daily_returns.mean() * 252
    ann_volatility = daily_returns.std() * np.sqrt(252)
    
    # 4. 构建实时的 Summary DataFrame
    summary_df = pd.DataFrame({
        'Sector': list(market_tickers.keys()),
        'Ticker': tickers,
        'Ann_Return': [ann_return[t] for t in tickers],
        'Volatility': [ann_volatility[t] for t in tickers]
    })
    
    # 计算 Sharpe Ratio (假设无风险利率为 0.02)
    risk_free_rate = 0.02
    summary_df['Sharpe_Ratio'] = (summary_df['Ann_Return'] - risk_free_rate) / summary_df['Volatility']
    
    # 记录最后更新时间
    update_time = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    
    return daily_returns, summary_df, update_time

with st.spinner("Initializing live market data engine..."):
    try:
        live_returns_df, live_summary_df, last_update_str = fetch_and_calculate_live_data()
        
        # 在侧边栏显示更新时间
        st.sidebar.success(f"🟢 Live Data Active\n\nLast Updated: \n{last_update_str}")
        
    except Exception as e:
        st.error(f"Failed to fetch live data from Yahoo Finance. Please try again later. Error: {e}")
        st.stop() # 如果数据抓取失败，停止渲染后续内容

# NLP 模型加载
@st.cache_resource
def load_nlp_model():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

sia = load_nlp_model()

# ==========================================
# 4. Tabs Layout (调整顺序)
# ==========================================
tab_pulse, tab_heatmap, tab_opt, tab_database = st.tabs([
    "Live Market Pulse",     
    "Correlation Heatmap",  
    "Portfolio Sandbox",
    "Risk/Return Database"  # 移到最后
])

# ----------------- Tab 1: Live Pulse -----------------
with tab_pulse:
    st.markdown("### 🌐 Global Market Pulse")
    
    # 计算近 5 日表现用于树状图
    recent_5d_returns = (live_returns_df.iloc[-1] + 1) * (live_returns_df.iloc[-2] + 1) * (live_returns_df.iloc[-3] + 1) * (live_returns_df.iloc[-4] + 1) * (live_returns_df.iloc[-5] + 1) - 1
    
    perf_df = pd.DataFrame({
        'Sector': list(market_tickers.keys()),
        'Ticker': list(market_tickers.values()),
        'Performance': [recent_5d_returns[t] for t in market_tickers.values()]
    })
    perf_df['Weight'] = 1 
    perf_df['Label'] = perf_df['Sector'] + "<br>" + perf_df['Performance'].apply(lambda x: f"{x*100:.2f}%")
    
    best_sector = perf_df.loc[perf_df['Performance'].idxmax()]
    worst_sector = perf_df.loc[perf_df['Performance'].idxmin()]
    
    st.info(f"**💡 Quant Insight (Trailing 5D):** **{best_sector['Sector']}** is leading the market ({best_sector['Performance']*100:.2f}%), while **{worst_sector['Sector']}** is lagging.")

    fig_tree = px.treemap(
        perf_df, path=[px.Constant("Global Macro Universe"), 'Label'], values='Weight',
        color='Performance', color_continuous_scale=['#FF4B4B', '#18181B', '#00C853'], color_continuous_midpoint=0
    )
    
    # 根据主题动态调整树状图背景
    plot_bg = 'rgba(0,0,0,0)'
    fig_tree.update_layout(margin=dict(t=20, l=0, r=0, b=0), paper_bgcolor=plot_bg, plot_bgcolor=plot_bg, coloraxis_showscale=False)
    # 根据主题调整树状图字体颜色 (如果底色是红/绿/黑，白字依然适用)
    fig_tree.update_traces(textfont=dict(family="Vollkorn", size=18, color="white"), textinfo="label")
    st.plotly_chart(fig_tree, use_container_width=True)

# ----------------- Tab 2: Heatmap -----------------
with tab_heatmap:
    st.markdown("### 🔗 Real-time Correlation Matrix")
    st.markdown("Calculated dynamically using the latest 1-year daily returns.")
    
    # 使用实时计算的 daily_returns 进行相关性分析
    selected_tickers = st.multiselect("Select assets to compare:", list(market_tickers.values()), default=['XLK', 'XLV', 'XLF', 'XLE', 'TLT', 'GLD'])
    
    if len(selected_tickers) > 1:
        corr_matrix = live_returns_df[selected_tickers].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdYlBu_r", zmin=-0.5, zmax=1)
        fig_corr.update_layout(height=600, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=text_color))
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("Please select at least 2 assets.")

# ----------------- Tab 3: Portfolio Sandbox -----------------
with tab_opt:
    st.markdown("### 🧪 AI-Augmented Portfolio Sandbox")
    st.markdown("Simulate optimal portfolios using **Live Historical Data** overlayed with NLP Sentiment Alpha.")
    
    col_input1, col_input2 = st.columns([2, 1])
    with col_input1:
        selected_opt_assets = st.multiselect("Select Assets for Optimization:", list(market_tickers.values()), default=['XLK', 'TLT', 'GLD', 'XLE'], key='opt_select')
    with col_input2:
        alpha_weight = st.slider("NLP Sentiment Alpha Weight", 0.0, 0.1, 0.05, 0.01)

    if len(selected_opt_assets) >= 2:
        with st.spinner("Running Monte Carlo Simulation..."):
            # 提取选定资产的实时年化数据
            df_selected_live = live_returns_df[selected_opt_assets]
            base_mean_returns = df_selected_live.mean() * 252
            cov_matrix = df_selected_live.cov() * 252
            
            # (暂时保留之前的通用模拟新闻池用于演示 NLP)
            generic_news_pool = ["Market shows cautious optimism", "Sector experiences mild correction", "Analysts upgrade growth forecast"]
            current_sentiments = {}
            for ticker in selected_opt_assets:
                sampled_titles = random.sample(generic_news_pool, 2)
                scores = [sia.polarity_scores(title)['compound'] for title in sampled_titles]
                current_sentiments[ticker] = np.mean(scores) if scores else 0.0
            
            adjusted_returns = base_mean_returns.copy()
            for ticker in selected_opt_assets:
                adjusted_returns[ticker] += (current_sentiments[ticker] * alpha_weight)
            
            num_portfolios = 2000 # 减少模拟次数提高响应速度
            results = np.zeros((3, num_portfolios))
            weights_record = []
            
            for i in range(num_portfolios):
                weights = np.random.random(len(selected_opt_assets))
                weights /= np.sum(weights)
                weights_record.append(weights)
                
                portfolio_return = np.sum(adjusted_returns * weights)
                portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                results[0,i] = portfolio_std_dev
                results[1,i] = portfolio_return
                results[2,i] = portfolio_return / portfolio_std_dev
                
            max_sharpe_idx = np.argmax(results[2])
            
            fig_opt = go.Figure()
            fig_opt.add_trace(go.Scatter(x=results[0,:], y=results[1,:], mode='markers', marker=dict(size=4, color=results[2,:], colorscale='Viridis', showscale=True), name='Simulated Portfolios', hoverinfo='none'))
            fig_opt.add_trace(go.Scatter(x=[results[0, max_sharpe_idx]], y=[results[1, max_sharpe_idx]], mode='markers+text', marker=dict(color='#D4AF37', size=16, symbol='star'), name='Max Sharpe', text=['Max Sharpe'], textposition="top center"))
            fig_opt.update_layout(xaxis_title="Predicted Volatility (Risk)", yaxis_title="AI-Adjusted Expected Return", height=400, margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color=text_color))
            st.plotly_chart(fig_opt, use_container_width=True)

# ----------------- Tab 4: Database -----------------
with tab_database:
    st.markdown("### 📊 Live Asset Risk & Return Summary")
    st.markdown("Metrics calculated dynamically based on the trailing 1-year daily close prices.")
    
    # 格式化显示
    display_data = live_summary_df.copy()
    display_data['Ann_Return'] = display_data['Ann_Return'].apply(lambda x: f"{x*100:.2f}%")
    display_data['Volatility'] = display_data['Volatility'].apply(lambda x: f"{x*100:.2f}%")
    display_data['Sharpe_Ratio'] = display_data['Sharpe_Ratio'].apply(lambda x: f"{x:.2f}")
    
    display_data.set_index('Sector', inplace=True)
    st.dataframe(display_data, use_container_width=True, height=500)