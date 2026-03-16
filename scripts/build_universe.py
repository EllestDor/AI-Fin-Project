import yfinance as yf
import pandas as pd
import numpy as np

# 1. 构建我们的“全球资产大字典” (Dictionary)
# 左边是你给板块起的名字，右边是真实的 ETF 代码
# 终极全市场资产池 (涵盖 GICS 11大板块及其深度细分)
# 1. 构建纯英文的全球资产大字典 (便于代码调用和后续多语言扩展)
portfolio_universe = {
    # 1. 科技 (Information Technology)
    "Tech_0_Macro": "XLK",
    "Tech_1_Semi": "SMH",
    "Tech_2_Soft": "IGV",
    "Tech_3_Cyber": "CIBR",
    "Tech_4_Cloud": "SKYY",

    # 2. 医疗保健 (Health Care)
    "Health_0_Macro": "XLV",
    "Health_1_Biotech": "IBB",
    "Health_2_Device": "IHI",
    "Health_3_Pharma": "PPH",

    # 3. 金融 (Financials)
    "Fin_0_Macro": "XLF",
    "Fin_1_RegBank": "KRE", 
    "Fin_2_Broker": "IAI",
    "Fin_3_Insur": "KIE",

    # 4. 非必需消费 (Consumer Discretionary)
    "Discr_0_Macro": "XLY",
    "Discr_1_Retail": "XRT",
    "Discr_2_Home": "XHB",
    "Discr_3_Auto": "CARZ",

    # 5. 必需消费 (Consumer Staples)
    "Stapl_0_Macro": "XLP",
    "Stapl_1_FoodBev": "PBJ",
    "Stapl_2_Agri": "MOO",

    # 6. 能源 (Energy)
    "Energy_0_Macro": "XLE",
    "Energy_1_OilGas": "XOP",
    "Energy_2_Equip": "OIH",
    "Energy_3_Clean": "ICLN",

    # 7. 工业 (Industrials)
    "Indus_0_Macro": "XLI",
    "Indus_1_AeroDef": "ITA",
    "Indus_2_Transp": "IYT",
    "Indus_3_Infra": "PAVE",

    # 8. 材料 (Materials)
    "Mat_0_Macro": "XLB",
    "Mat_1_GoldMin": "GDX",
    "Mat_2_Copper": "COPX",
    "Mat_3_Chem": "VAW",

    # 9. 公用事业 (Utilities)
    "Util_0_Macro": "XLU",
    "Util_1_Water": "CGW",
    "Util_2_Solar": "TAN",

    # 10. 房地产 (Real Estate)
    "RE_0_Macro": "XLRE",
    "RE_1_DataCtr": "SRVR",
    "RE_2_Mortgage": "REM",

    # 11. 通信服务 (Communication Services)
    "Comm_0_Macro": "XLC",
    "Comm_1_Telecom": "IYZ",
    "Comm_2_Media": "PBS",

    # --- 跨大类宏观对冲资产 ---
    "Macro_Gold": "GLD",
    "Macro_LongBond": "TLT",
    "Macro_Cmdty": "PDBC"
}

# 提取所有的 ETF 代码去下载
tickers = list(portfolio_universe.values())
print(f"准备抓取 {len(tickers)} 个全球核心板块数据，请稍候...")

# 2. 一次性批量下载所有数据 (静默下载，只取收盘价)
data = yf.download(tickers, period="1y")['Close']

# 3. 计算每日收益率
daily_returns = data.pct_change().dropna()

# 4. 计算年化指标 (252个交易日)
annual_returns = daily_returns.mean() * 252
annual_volatility = daily_returns.std() * np.sqrt(252)

# 5. 组装成纯英文版的财务报表
summary_list = []
for name, ticker in portfolio_universe.items():
    ret = annual_returns[ticker]
    vol = annual_volatility[ticker]
    sharpe = ret / vol 
    
    summary_list.append({
        "Sector": name,             
        "Ticker": ticker,
        "Ann_Return": f"{ret:.2%}", 
        "Volatility": f"{vol:.2%}", 
        "Sharpe_Ratio": f"{sharpe:.2f}"
    })

# 转换格式并对齐
master_summary = pd.DataFrame(summary_list)
master_summary.set_index("Sector", inplace=True)

# 打印出这份极其专业的报告
print("\n" + "="*50)
print("全球大类及细分板块 风险/收益 终极体检报告")
print("="*50)
print(master_summary)

# 6. 存下这些宝贵的数据！
master_summary.to_csv("data/processed/global_universe_summary.csv")
# 注意：这行代码非常关键！我们把每天的涨跌幅存下来，用来做后面的“相关性矩阵”和“投资组合优化”
daily_returns.to_csv("data/processed/global_universe_returns.csv")

print("\n 太棒了！数据已成功保存为：")
print("1. global_universe_summary.csv (概览表)")
print("2. global_universe_returns.csv (底层收益率序列，用于后续算相关性)")