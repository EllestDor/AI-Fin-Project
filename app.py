import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np                      # 新增：用于高效的矩阵和随机数运算
import plotly.graph_objects as go       # 新增：用于绘制极其精细的有效前沿散点图
import yfinance as yf
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore

# 1. 网页全局设置 (扩展为宽屏模式)
st.set_page_config(page_title="Global Macro Dashboard", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 2. Custom CSS for High-End Legal/Corporate Dark Mode
#    (Powered by Vollkorn Font & Pill-shaped Hover Tabs)
# ==========================================
st.markdown("""
    <style>
    /* 0. 引入 Google Fonts 中的 Vollkorn 字体 */
    @import url('https://fonts.googleapis.com/css2?family=Vollkorn:ital,wght@0,400..900;1,400..900&display=swap');

    /* 1. 全局字体设定 */
    html, body, [class*="css"] {
        font-family: 'Vollkorn', Georgia, 'Times New Roman', Times, serif !important;
    }

    /* 2. 主标题 (H1) */
    h1 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 3rem !important;
        letter-spacing: -0.01em !important;
        margin-bottom: 1.5rem !important;
    }

    /* 3. 核心修复：Tabs 容器与胶囊式圆角按钮 */
    div[data-testid="stTabs"] > div[role="tablist"] {
        display: flex !important;
        width: 100% !important;
        justify-content: space-between !important; 
        gap: 0.5rem !important; /* 设置按钮之间的间隔 */
        border-bottom: none !important; /* 去除默认的整行下划线，让按钮更独立 */
        padding-bottom: 1rem !important;
    }

    /* 每个 Tab 按钮的基础样式：增加 padding 和 圆角 */
    button[data-baseweb="tab"] {
        flex: 1 !important; 
        display: flex !important;
        justify-content: center !important; 
        align-items: center !important;
        padding: 0.6rem 1rem !important; /* 给背景留出上下左右的呼吸空间 */
        background-color: transparent !important; 
        border-radius: 12px !important; /* 关键：12px 的圆角长方形 */
        border: none !important; /* 覆盖 Streamlit 默认的底线 */
        transition: all 0.25s ease-in-out !important; /* 关键：增加 0.25秒 的平滑色彩渐变动画 */
    }

    /* Tab 内部文字 */
    button[data-baseweb="tab"] > div {
        font-family: 'Vollkorn', serif !important;
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        color: #71717A !important; /* 默认暗灰色 */
    }
    
    /* 🔥 鼠标悬停时的特效 (Hover) */
    button[data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.08) !important; /* 鼠标移上去，泛起一层极具质感的浅色微光 */
    }
    
    button[data-baseweb="tab"]:hover > div {
        color: #D4D4D8 !important; /* 字体微微提亮 */
    }

    /* 🔥 选中状态的特效 (Selected) */
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: rgba(212, 175, 55, 0.15) !important; /* 选中时，背景变成淡淡的香槟金 */
    }
    
    button[data-baseweb="tab"][aria-selected="true"] > div {
        color: #FFFFFF !important; /* 文字变成纯白 */
        font-weight: 800 !important;
    }

    /* 4. 下拉菜单的标签 (Selectbox Label) */
    div[data-testid="stSelectbox"] label p {
        font-family: 'Vollkorn', serif !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* 下拉框背景设定 */
    div[data-baseweb="select"] > div {
        background-color: #18181B !important;
        border: 1px solid #3F3F46 !important;
    }
    div[data-baseweb="select"] span {
        font-family: 'Vollkorn', serif !important;
    }

    /* 5. 子标题 (H3) */
    h3 {
        font-family: 'Vollkorn', serif !important;
        color: #F4F4F5 !important;
        font-weight: 700 !important;
        margin-top: 2rem !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 1px solid #27272A !important;
    }
    
    /* 6. 侧边栏文字设定 */
    [data-testid="stSidebar"] p {
        font-family: 'Vollkorn', serif !important;
        color: #D4D4D8 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. Sidebar
# ==========================================
st.sidebar.title("Navigation Menu")
st.sidebar.markdown("Welcome to the **Global Macro Asset Allocation** Engine.")
st.sidebar.divider()
st.sidebar.info("Tip: Interact with the charts by hovering over the heatmap to see correlation values, or drag to zoom in.")

# ==========================================
# 4. Main Header
# ==========================================
st.title("Global Macro Asset Allocation Dashboard")
st.markdown("Track the performance, risk, and correlation of global GICS sectors and macro hedges.")

# ==========================================
# 5. Load Data
# ==========================================
@st.cache_data 
def load_data():
    summary = pd.read_csv("data/processed/global_universe_summary.csv", index_col=0)
    returns = pd.read_csv("data/processed/global_universe_returns.csv", index_col=0, parse_dates=True)
    return summary, returns

summary_data, returns_data = load_data()

# ==========================================
# 6. Tabs & Layout
# ==========================================
tab_pulse, tab_heatmap, tab_database, tab_opt = st.tabs([
    "Live Market Pulse",     # main page
    "Correlation Heatmap",  
    "Risk/Return Database", 
    "Portfolio Sandbox"      # effective frontier, optimized portfolio
])

with tab_heatmap:
    # 1. 机构级分类目录 (Institutional-grade taxonomy)
    drilldown_dict = {
        "Macro Overview: 13 Core Global Assets": ['XLK', 'XLV', 'XLF', 'XLY', 'XLP', 'XLE', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC', 'TLT', 'GLD'],
        "Sector Deep Dive: Information Technology": ['XLK', 'SMH', 'IGV', 'CIBR', 'SKYY'],
        "Sector Deep Dive: Healthcare & Biotech": ['XLV', 'IBB', 'IHI', 'PPH'],
        "Sector Deep Dive: Financial Services": ['XLF', 'KRE', 'IAI', 'KIE'],
        "Sector Deep Dive: Consumer Discretionary": ['XLY', 'XRT', 'XHB', 'CARZ'],
        "Sector Deep Dive: Energy & Clean Tech": ['XLE', 'XOP', 'OIH', 'ICLN'],
        "Sector Deep Dive: Industrials & Defense": ['XLI', 'ITA', 'IYT', 'PAVE'],
        "Sector Deep Dive: Materials & Mining": ['XLB', 'GDX', 'COPX', 'VAW']
    }
    
    # 2. 下拉菜单
    selected_view = st.selectbox(
        "Select Analysis Universe:", 
        list(drilldown_dict.keys())
    )
    
    # 使用竖线分隔符
    st.markdown(f"### Correlation Matrix | {selected_view}")
    
    # 3. 动态数据提取
    selected_tickers = drilldown_dict[selected_view]
    dynamic_returns = returns_data[selected_tickers]
    
    # 计算相关性矩阵
    corr_matrix = dynamic_returns.corr()
    
    # 4. Plotly 动态热力图渲染
    fig = px.imshow(corr_matrix, 
                    text_auto=".2f", 
                    aspect="auto", 
                    color_continuous_scale="RdYlBu_r", 
                    zmin=-0.5, zmax=1,
                    labels=dict(color="Correlation"))
    
    # 留白控制
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

with tab_database:
    st.subheader("Asset Risk & Return Summary")
    
    # 创建 Sector 详细名称映射字典
    sector_mapping = {
        'Tech_0_Macro': 'Technology - Broad (XLK)',
        'Tech_1_Semi': 'Technology - Semiconductors (SMH)',
        'Tech_2_Soft': 'Technology - Software (IGV)',
        'Tech_3_Cyber': 'Technology - Cybersecurity (CIBR)',
        'Tech_4_Cloud': 'Technology - Cloud Computing (SKYY)',
        'Health_0_Macro': 'Healthcare - Broad (XLV)',
        'Health_1_Biotech': 'Healthcare - Biotech (IBB)',
        'Health_2_Device': 'Healthcare - Medical Devices (IHI)',
        'Health_3_Pharma': 'Healthcare - Pharma (PPH)',
        'Fin_0_Macro': 'Financials - Broad (XLF)',
        'Fin_1_RegBank': 'Financials - Regional Banks (KRE)',
        'Fin_2_Broker': 'Financials - Brokers (IAI)',
        'Fin_3_Insur': 'Financials - Insurance (KIE)',
        'Discr_0_Macro': 'Discretionary - Broad (XLY)',
        'Discr_1_Retail': 'Discretionary - Retail (XRT)',
        'Discr_2_Home': 'Discretionary - Homebuilding (XHB)',
        'Discr_3_Auto': 'Discretionary - Automotive (CARZ)',
        'Stapl_0_Macro': 'Staples - Broad (XLP)',
        'Stapl_1_FoodBev': 'Staples - Food & Beverage (PBJ)',
        'Stapl_2_Agri': 'Staples - Agricultural (MOO)',
        'Energy_0_Macro': 'Energy - Broad (XLE)',
        'Energy_1_OilGas': 'Energy - Oil & Gas (XOP)',
        'Energy_2_Equip': 'Energy - Equipment (OIH)',
        'Energy_3_Clean': 'Energy - Clean Energy (ICLN)',
        'Indus_0_Macro': 'Industrials - Broad (XLI)',
        'Indus_1_AeroDef': 'Industrials - Aerospace & Defense (ITA)',
        'Indus_2_Transp': 'Industrials - Transportation (IYT)',
        'Indus_3_Infra': 'Industrials - Infrastructure (PAVE)',
        'Mat_0_Macro': 'Materials - Broad (XLB)',
        'Mat_1_GoldMin': 'Materials - Gold & Mining (GDX)',
        'Mat_2_Copper': 'Materials - Copper (COPX)',
        'Mat_3_Chem': 'Materials - Chemicals (VAW)',
        'Util_0_Macro': 'Utilities - Broad (XLU)',
        'Util_1_Water': 'Utilities - Water (CGW)',
        'Util_2_Solar': 'Utilities - Solar (TAN)',
        'RE_0_Macro': 'Real Estate - Broad (XLRE)',
        'RE_1_DataCtr': 'Real Estate - Data Centers (SRVR)',
        'RE_2_Mortgage': 'Real Estate - Mortgage REITs (REM)',
        'Comm_0_Macro': 'Communications - Broad (XLC)',
        'Comm_1_Telecom': 'Communications - Telecom (IYZ)',
        'Comm_2_Media': 'Communications - Media (PBS)',
        'Macro_Gold': 'Gold (GLD)',
        'Macro_LongBond': 'Long-term Bonds (TLT)',
        'Macro_Cmdty': 'Commodities (PDBC)',
    }
    
   # 重置索引，让 Sector 成为一个普通列
    display_data = summary_data.reset_index()
    
    # 1. 映射出新的友好名称
    display_data['Sector Name'] = display_data['Sector'].map(sector_mapping)
    
    # 2. 选择展示的列并重命名一下表头
    display_data = display_data[['Sector Name', 'Ticker', 'Ann_Return', 'Volatility', 'Sharpe_Ratio']]
    
# 3. 把'Sector Name' 设为索引
    display_data.set_index('Sector Name', inplace=True)
    
    # 在网页上渲染
    st.dataframe(display_data, use_container_width=True, height=500)
    
with tab_opt:
    st.markdown("### 🧪 AI-Augmented Portfolio Sandbox")
    st.markdown("Simulate optimal portfolios using historical data **overlayed with real-time NLP Sentiment Alpha**.")
    
    # 1. 资产选择器与情绪因子开关
    col_input1, col_input2 = st.columns([2, 1])
    
    with col_input1:
        all_tickers = returns_data.columns.tolist()
        selected_assets = st.multiselect(
            "Select Assets for Optimization:", 
            all_tickers, 
            default=['XLK', 'TLT', 'GLD', 'XLE']
        )
        
    with col_input2:
        # 给用户一个滑动条，决定“情绪得分”对最终收益率的影响有多大
        alpha_weight = st.slider(
            "NLP Sentiment Alpha Weight", 
            min_value=0.0, max_value=0.1, value=0.05, step=0.01,
            help="Higher value means the optimizer relies more on today's news sentiment. Set to 0 for pure historical optimization."
        )

    if len(selected_assets) >= 2:
        with st.spinner("Running Monte Carlo Simulation with AI overlay..."):
            # 2. 获取历史基础数据
            df_selected = returns_data[selected_assets]
            base_mean_returns = df_selected.mean() * 252
            cov_matrix = df_selected.cov() * 252
            
            # 3. 动态计算选中资产的当前情绪得分 (Alpha)
            current_sentiments = {}
            for ticker in selected_assets:
                try:
                    # 获取近期新闻（为保证速度，只看前3条）
                    news = yf.Ticker(ticker).news[:3]
                    if news:
                        scores = [sia.polarity_scores(item.get('title', ''))['compound'] for item in news]
                        current_sentiments[ticker] = np.mean(scores)
                    else:
                        current_sentiments[ticker] = 0.0 # 没新闻就视为中性
                except:
                    current_sentiments[ticker] = 0.0
            
            # 4. 核心数学转换：将情绪得分转化为预期收益的 Alpha 偏置
            # 公式: 调整后收益 = 历史收益 + (情绪得分 * 权重)
            adjusted_returns = base_mean_returns.copy()
            for ticker in selected_assets:
                alpha_boost = current_sentiments[ticker] * alpha_weight
                adjusted_returns[ticker] += alpha_boost
            
            # 展示情绪得分情况
            st.markdown("#### 📡 Real-time Sentiment Alpha Applied")
            sentiment_df = pd.DataFrame.from_dict(current_sentiments, orient='index', columns=['NLP Score'])
            sentiment_df['Alpha Boost'] = sentiment_df['NLP Score'] * alpha_weight
            sentiment_df['Alpha Boost'] = sentiment_df['Alpha Boost'].apply(lambda x: f"{x*100:+.2f}%")
            
            # 使用更扁平的展示方式
            st.dataframe(sentiment_df.T, use_container_width=True)

            # 5. 蒙特卡洛模拟 (使用调整后的预期收益)
            num_portfolios = 5000
            results = np.zeros((3, num_portfolios))
            weights_record = []
            
            for i in range(num_portfolios):
                weights = np.random.random(len(selected_assets))
                weights /= np.sum(weights)
                weights_record.append(weights)
                
                # 注意这里使用的是 adjusted_returns 而不是 base_mean_returns！
                portfolio_return = np.sum(adjusted_returns * weights)
                portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                results[0,i] = portfolio_std_dev
                results[1,i] = portfolio_return
                results[2,i] = portfolio_return / portfolio_std_dev
                
            # 6. 提取最优解
            max_sharpe_idx = np.argmax(results[2])
            min_vol_idx = np.argmin(results[0])
            
            # 7. 绘制 Plotly 散点图
            fig_opt = go.Figure()
            
            fig_opt.add_trace(go.Scatter(
                x=results[0,:], y=results[1,:],
                mode='markers',
                marker=dict(size=4, color=results[2,:], colorscale='Viridis', showscale=True, colorbar=dict(title="Sharpe")),
                name='Simulated Portfolios', hoverinfo='none'
            ))
            
            fig_opt.add_trace(go.Scatter(
                x=[results[0, max_sharpe_idx]], y=[results[1, max_sharpe_idx]],
                mode='markers+text', marker=dict(color='#D4AF37', size=16, symbol='star'),
                name='Max Sharpe', text=['Max Sharpe (AI-Adjusted)'], textposition="top center"
            ))
            
            fig_opt.update_layout(
                xaxis_title="Predicted Volatility (Risk)", yaxis_title="AI-Adjusted Expected Return",
                height=500, margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#A1A1AA")
            )
            st.plotly_chart(fig_opt, use_container_width=True)
            
            # 8. 输出最优权重
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

with tab_pulse:
    st.markdown("### 🌐 Global Market Pulse & AI Takeaway")
    
    # 1. 定义要追踪的核心板块 (11大 GICS Sector 加上 黄金和长债)
    market_tickers = {
        'Technology': 'XLK', 'Healthcare': 'XLV', 'Financials': 'XLF',
        'Consumer Discr': 'XLY', 'Consumer Staples': 'XLP', 'Energy': 'XLE',
        'Industrials': 'XLI', 'Materials': 'XLB', 'Utilities': 'XLU',
        'Real Estate': 'XLRE', 'Communications': 'XLC', 
        'Safe Haven: Gold': 'GLD', 'Safe Haven: Bonds': 'TLT'
    }
    
    # 2. 抓取近期数据 (使用缓存避免每次刷新太慢)
    @st.cache_data(ttl=3600) # 缓存1小时
    def fetch_live_performance():
        tickers = list(market_tickers.values())
        # 获取最近 5 个交易日的数据来计算周涨跌幅
        data = yf.download(tickers, period="5d")['Close']
        # 计算最新的一天相对 5 天前的涨跌幅
        returns = (data.iloc[-1] - data.iloc[0]) / data.iloc[0]
        return returns

    with st.spinner("Fetching live market data..."):
        try:
            live_returns = fetch_live_performance()
            
            # 构建用于画图的 DataFrame
            perf_df = pd.DataFrame({
                'Sector': list(market_tickers.keys()),
                'Ticker': list(market_tickers.values()),
                'Performance': [live_returns[t] for t in market_tickers.values()]
            })
            
            # 为了让树状图有大小区分，我们给所有板块相同的权重，或者用绝对值
            perf_df['Weight'] = 1 
            # 格式化显示标签
            perf_df['Label'] = perf_df['Sector'] + "<br>" + perf_df['Performance'].apply(lambda x: f"{x*100:.2f}%")
            
            # 3. 动态生成 AI Takeaway 结论 (基于数据)
            best_sector = perf_df.loc[perf_df['Performance'].idxmax()]
            worst_sector = perf_df.loc[perf_df['Performance'].idxmin()]
            
            st.info(f"**💡 Quant Insight:** In the trailing 5 days, **{best_sector['Sector']}** is leading the market ({best_sector['Performance']*100:.2f}%), while **{worst_sector['Sector']}** is lagging. "
                    f"Consider taking profits from overextended winners and exploring value in oversold sectors.")

            # 4. 绘制 Bloomberg 风格的红绿 Treemap
            fig_tree = px.treemap(
                perf_df, 
                path=[px.Constant("Global Macro Universe"), 'Label'], 
                values='Weight',
                color='Performance',
                color_continuous_scale=['#FF4B4B', '#18181B', '#00C853'], # 红(跌) - 黑(平) - 绿(涨)
                color_continuous_midpoint=0
            )
            
            fig_tree.update_layout(
                margin=dict(t=20, l=0, r=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                coloraxis_showscale=False # 隐藏色条让图更纯粹
            )
            
            # 强制树状图里的文字居中和放大
            fig_tree.update_traces(
                textfont=dict(family="Vollkorn", size=18, color="white"),
                textinfo="label",
                hovertemplate="<b>%{label}</b><br>Return: %{color:.2%}<extra></extra>"
            )
            
            st.plotly_chart(fig_tree, use_container_width=True)
            
        except Exception as e:
            st.warning("Live data fetching is currently unavailable. Please check your internet connection or try again later.")
            st.write(e)

# ==========================================
# 🌟 NLP 新闻情绪 Alpha 因子分析 (NLP Sentiment Alpha)
# ==========================================
    st.markdown("---")
    st.markdown("### 🧠 AI NLP Sentiment Alpha Engine")
    
    # 1. 缓存加载 NLTK 的 VADER 词典，避免每次刷新重复下载
    @st.cache_resource
    def load_nlp_model():
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
        return SentimentIntensityAnalyzer()

    sia = load_nlp_model()

    # 2. 交互式选择器：用户想看哪个板块的新闻情绪？
    col_sel, col_space = st.columns([1, 2])
    with col_sel:
        target_sector = st.selectbox("Select Sector for Real-time News Analysis:", list(market_tickers.keys()))
        target_ticker = market_tickers[target_sector]

    # 3. 抓取新闻并进行 NLP 实时打分
    with st.spinner(f"Scraping live headlines & calculating NLP scores for {target_ticker}..."):
        try:
            ticker_obj = yf.Ticker(target_ticker)
            news_data = ticker_obj.news  # 抓取雅虎财经最新新闻
            
            if news_data:
                sentiment_scores = []
                news_items = []
                
                # 遍历最新的 5 条新闻
                for item in news_data[:5]:
                    title = item.get('title', '')
                    publisher = item.get('publisher', 'Unknown')
                    link = item.get('link', '#')
                    
                    # 核心：使用 VADER 对标题进行情绪打分 (-1 到 1)
                    score = sia.polarity_scores(title)['compound']
                    sentiment_scores.append(score)
                    
                    # 根据得分赋予视觉标签
                    if score > 0.05:
                        sentiment_label = "🟢 Bullish"
                    elif score < -0.05:
                        sentiment_label = "🔴 Bearish"
                    else:
                        sentiment_label = "⚪ Neutral"
                        
                    news_items.append(f"**[{publisher}]** [{title}]({link})  \n*AI Score: {score:.2f} ({sentiment_label})*")
                
                # 计算平均情绪得分作为 Alpha 因子
                avg_score = np.mean(sentiment_scores)
                
                # 4. 渲染到网页：左边是新闻流，右边是情绪仪表盘
                col_news, col_metric = st.columns([2, 1])
                
                with col_news:
                    st.markdown("#### 📰 Live Headlines")
                    for news in news_items:
                        st.write(news)
                        st.markdown("<hr style='margin: 0.5em 0; border-color: #3F3F46;'>", unsafe_allow_html=True)
                
                with col_metric:
                    st.markdown("#### ⚡ Alpha Factor (Sentiment)")
                    
                    # 动态设定颜色和偏好
                    if avg_score > 0.05:
                        delta_text = "Bullish Momentum"
                        delta_color = "normal"
                    elif avg_score < -0.05:
                        delta_text = "Bearish Pressure"
                        delta_color = "inverse"
                    else:
                        delta_text = "Neutral"
                        delta_color = "off"
                        
                    st.metric(label=f"{target_ticker} Sentiment Score", 
                              value=f"{avg_score:.2f}", 
                              delta=delta_text, 
                              delta_color=delta_color)
                              
                    st.info(f"**How to use this Alpha?**\nA score of **{avg_score:.2f}** (ranging from -1 to 1) acts as a quantitative sentiment overlay. In our Portfolio Sandbox, this score will be mathematically added to the historical expected return of {target_ticker} to adjust optimal portfolio weights.")
            else:
                st.warning(f"No recent news found for {target_ticker} from the feed.")
                
        except Exception as e:
            st.error(f"Failed to load NLP engine or fetch news. Error: {e}")

