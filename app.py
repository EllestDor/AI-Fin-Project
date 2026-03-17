import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np                      # 新增：用于高效的矩阵和随机数运算
import plotly.graph_objects as go       # 新增：用于绘制极其精细的有效前沿散点图

# 1. 网页全局设置 (扩展为宽屏模式)
st.set_page_config(page_title="Global Macro Dashboard v1.1", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 2. Custom CSS
# ==========================================
st.markdown("""
    <style>
    /* 0. 引入 Google Fonts 中的 Vollkorn 字体 */
    @import url('https://fonts.googleapis.com/css2?family=Vollkorn:ital,wght@0,400..900;1,400..900&display=swap');

    /* 1. 全局字体设定：优先使用 Vollkorn，如果没有则降级到其他衬线字体 */
    html, body, [class*="css"] {
        font-family: 'Vollkorn', Georgia, 'Times New Roman', Times, serif !important;
    }

    /* 2. 主标题 (H1) - 使用 Vollkorn 展现经典的高级感 */
    h1 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 3rem !important; /* 衬线字体通常可以稍微放大一点 */
        letter-spacing: -0.01em !important;
        margin-bottom: 1.5rem !important;
    }

    /* 3. 标签页 (Tabs) */
    button[data-baseweb="tab"] > div {
        font-family: 'Vollkorn', serif !important; /* 强制标签页也使用该字体 */
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        color: #71717A !important; 
        padding-bottom: 0.5rem !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] > div {
        color: #FFFFFF !important;
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
    
    /* 下拉框本身的背景设定 */
    div[data-baseweb="select"] > div {
        background-color: #18181B !important;
        border: 1px solid #3F3F46 !important;
    }
    
    /* 强制下拉框内部选中的文字也使用 Vollkorn */
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
tab1, tab2, tab3 = st.tabs(["Interactive Heatmap ", "Risk/Return Database ", "Portfolio Optimization "])

with tab1:
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

with tab2:
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
    
with tab3:
    # 1. 标题与说明
    st.markdown("### Markowitz Efficient Frontier & Portfolio Optimization")
    st.markdown("Simulate 5,000 random portfolios to find the optimal weights for Maximum Sharpe Ratio and Minimum Volatility.")
    
    # 2. 让用户自由选择资产 (默认选三个经典的不相关资产：科技、长债、黄金)
    all_tickers = returns_data.columns.tolist()
    selected_assets = st.multiselect(
        "Select Assets for Optimization:", 
        all_tickers, 
        default=['XLK', 'TLT', 'GLD']
    )
    
    # 确保用户至少选了两个资产才能计算协方差
    if len(selected_assets) >= 2:
        # 3. 提取选中资产的日收益率，并计算年化预期收益和协方差矩阵
        df_selected = returns_data[selected_assets]
        mean_returns = df_selected.mean() * 252
        cov_matrix = df_selected.cov() * 252
        
        # 4. 蒙特卡洛模拟 (Monte Carlo Simulation) 核心引擎
        num_portfolios = 5000
        # 创建空数组存放：波动率, 收益率, 夏普比率
        results = np.zeros((3, num_portfolios))
        weights_record = []
        
        for i in range(num_portfolios):
            # 生成随机权重并归一化 (让权重和为1)
            weights = np.random.random(len(selected_assets))
            weights /= np.sum(weights)
            weights_record.append(weights)
            
            # 计算组合预期收益
            portfolio_return = np.sum(mean_returns * weights)
            # 计算组合预期波动率 (核心矩阵运算: w^T * Cov * w)
            portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # 记录结果 (假设无风险利率 Risk-Free Rate 为 0)
            results[0,i] = portfolio_std_dev
            results[1,i] = portfolio_return
            results[2,i] = portfolio_return / portfolio_std_dev
            
        # 5. 找出两大核心点：最大夏普比率 (Max Sharpe) 和 最小波动率 (Min Volatility)
        max_sharpe_idx = np.argmax(results[2])
        min_vol_idx = np.argmin(results[0])
        
        # 6. 利用 Plotly Graph Objects 绘制极具科技感的散点图
        fig_opt = go.Figure()
        
        # 绘制 5000 个随机组合构成的有效前沿云图
        fig_opt.add_trace(go.Scatter(
            x=results[0,:], y=results[1,:],
            mode='markers',
            marker=dict(
                size=5,
                color=results[2,:], # 按夏普比率上色
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Sharpe Ratio")
            ),
            name='Simulated Portfolios',
            hoverinfo='none' # 背景点关闭 hover 以免杂乱
        ))
        
        # 标出 Max Sharpe 点 (红星)
        fig_opt.add_trace(go.Scatter(
            x=[results[0, max_sharpe_idx]], y=[results[1, max_sharpe_idx]],
            mode='markers+text',
            marker=dict(color='red', size=15, symbol='star'),
            name='Max Sharpe',
            text=['Max Sharpe'], textposition="top center"
        ))
        
        # 标出 Min Volatility 点 (蓝星)
        fig_opt.add_trace(go.Scatter(
            x=[results[0, min_vol_idx]], y=[results[1, min_vol_idx]],
            mode='markers+text',
            marker=dict(color='blue', size=15, symbol='star'),
            name='Min Volatility',
            text=['Min Vol'], textposition="bottom center"
        ))
        
        # 图表排版微调
        fig_opt.update_layout(
            xaxis_title="Annualized Volatility (Risk)",
            yaxis_title="Annualized Return",
            height=600,
            margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#A1A1AA")
        )
        st.plotly_chart(fig_opt, use_container_width=True)
        
        # 7. 在图表下方输出最优权重的数据表
        st.markdown("### Optimal Portfolio Weights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**Maximum Sharpe Ratio Portfolio (Highest Return/Risk)**")
            max_sharpe_weights = weights_record[max_sharpe_idx]
            df_max_sharpe = pd.DataFrame({'Asset': selected_assets, 'Weight': max_sharpe_weights})
            df_max_sharpe['Weight'] = df_max_sharpe['Weight'].apply(lambda x: f"{x*100:.2f}%")
            st.dataframe(df_max_sharpe.set_index('Asset'), use_container_width=True)
            
        with col2:
            st.info("**Minimum Volatility Portfolio (Safest)**")
            min_vol_weights = weights_record[min_vol_idx]
            df_min_vol = pd.DataFrame({'Asset': selected_assets, 'Weight': min_vol_weights})
            df_min_vol['Weight'] = df_min_vol['Weight'].apply(lambda x: f"{x*100:.2f}%")
            st.dataframe(df_min_vol.set_index('Asset'), use_container_width=True)

    else:
        st.warning("Please select at least 2 assets to run the optimization.")