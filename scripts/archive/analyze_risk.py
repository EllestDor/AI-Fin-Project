import pandas as pd
import numpy as np

# 1. 读取我们刚才下载的“战利品”
# index_col=0 表示把第一列（日期）当作索引
print("正在读取本地数据...")
data = pd.read_csv("global_tech_etfs.csv", index_col=0, parse_dates=True)

# 2. 计算每日收益率 (Daily Returns)
# pct_change() 是 Pandas 的神仙函数，一键计算今天比昨天涨跌了百分之几
daily_returns = data.pct_change().dropna()

# 3. 计算金融界最核心的两个指标：年化收益与年化波动率
# 金融常识：一年大约有 252 个交易日
annual_returns = daily_returns.mean() * 252
annual_volatility = daily_returns.std() * np.sqrt(252)

# 4. 把结果包装成一个高级的财务分析表格
summary = pd.DataFrame({
    "年化预期收益率 (Annual Return)": annual_returns.map('{:.2%}'.format),
    "年化波动率/风险 (Volatility)": annual_volatility.map('{:.2%}'.format)
})

print("\n=== 全球科技 ETF 风险与收益体检报告 ===")
print(summary)