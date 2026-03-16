import yfinance as yf
import pandas as pd

# 1. 定义我们的全球科技 ETF 组合池
# IXN: 全球科技, CQQQ: 中国科技, IGV: 北美软件, SMH: 全球半导体
etf_tickers = ['IXN', 'CQQQ', 'IGV', 'SMH']

# 2. 下载过去 1 年的历史数据 (提取每日收盘价)
print("正在从雅虎财经下载 ETF 数据，请稍候...")
# yfinance 的 download 函数可以直接拉取多个标的的数据
data = yf.download(etf_tickers, period="1y")['Close']

# 3. 在控制台预览前 5 行数据
print("\n下载成功！数据预览：")
print(data.head())

# 4. 将数据保存到本地电脑，方便后续做收益率和风险分析
file_name = "global_tech_etfs.csv"
data.to_csv(file_name)
print(f"\n太棒了！数据已成功存入你当前文件夹下的：{file_name}")