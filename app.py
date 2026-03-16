import streamlit as st
import pandas as pd
import plotly.express as px  # 引入强大的动态绘图库

# 1. 网页全局设置 (扩展为宽屏模式)
st.set_page_config(page_title="Global Macro Dashboard", layout="wide", initial_sidebar_state="expanded")

# 2. 增加左侧边栏 (Sidebar) - 瞬间提升 Web App 质感
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2942/2942268.png", width=50) # 加个小图标
st.sidebar.title("Navigation")
st.sidebar.markdown("Welcome to the **Global Macro Asset Allocation** Engine.")
st.sidebar.divider()
st.sidebar.info("💡 Tip: You can interact with the charts! Hover over the heatmap to see exact correlation values, or drag to zoom in.")

# 3. 网页主标题
st.title("Global Macro Asset Allocation Dashboard")
st.markdown("Track the performance, risk, and correlation of global GICS sectors and macro hedges.")

# 4. 加载数据
@st.cache_data 
def load_data():
    summary = pd.read_csv("data/processed/global_universe_summary.csv", index_col=0)
    returns = pd.read_csv("data/processed/global_universe_returns.csv", index_col=0, parse_dates=True)
    return summary, returns

summary_data, returns_data = load_data()

# 5. 引入标签页 (Tabs) 设计 - 避免页面像长长的 PDF
tab1, tab2 = st.tabs(["Interactive Heatmap", "Risk/Return Database"])

with tab1:
    st.subheader("Macro Correlation Matrix")
    
    # 提取 13 个宏观资产
    macro_tickers = ['XLK', 'XLV', 'XLF', 'XLY', 'XLP', 'XLE', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC', 'TLT', 'GLD']
    macro_returns = returns_data[macro_tickers]
    corr_matrix = macro_returns.corr()
    
    # 🌟 魔法时刻：使用 Plotly 绘制动态热力图
    fig = px.imshow(corr_matrix, 
                    text_auto=".2f", # 显示两位小数
                    aspect="auto", 
                    color_continuous_scale="RdYlGn_r", # 红绿配色
                    zmin=-0.5, zmax=1,
                    title="Hover over the cells to view correlation details")
    
    # 微调动态图的样式
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=40, b=0))
    
    # 在网页上渲染动态图
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
    
    # 2. 重新选择我们要展示的列（直接不要 'Sector' 了），并重命名一下表头让它更专业
    display_data = display_data[['Sector Name', 'Ticker', 'Ann_Return', 'Volatility', 'Sharpe_Ratio']]
    
    # 3. 把用户友好的 'Sector Name' 设为索引，这样表格最左侧就不会有丑丑的 0, 1, 2, 3 序号了
    display_data.set_index('Sector Name', inplace=True)
    
    # 在网页上渲染
    st.dataframe(display_data, use_container_width=True, height=500)
    
    # Streamlit 自带的 dataframe 其实也是可交互的（可以点击表头排序，也可以全屏查看）
    st.dataframe(display_data, use_container_width=True, height=500)
    