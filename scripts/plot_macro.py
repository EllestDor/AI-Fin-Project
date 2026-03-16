import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 读取我们刚刚用大引擎跑出来的收益率“金矿”
# index_col=0 保证日期被正确识别为表格的行索引
returns_data = pd.read_csv("global_universe_returns.csv", index_col=0, parse_dates=True)

# 2. 定义我们要“降维”观察的宏观基准池 (11大 GICS 板块 + 2大避险资产)
macro_tickers = [
    'XLK',  # 科技
    'XLV',  # 医疗
    'XLF',  # 金融
    'XLY',  # 非必需消费
    'XLP',  # 必需消费
    'XLE',  # 能源
    'XLI',  # 工业
    'XLB',  # 材料
    'XLU',  # 公用事业 (防御)
    'XLRE', # 房地产
    'XLC',  # 通信服务
    'TLT',  # 20年期美债 (避险)
    'GLD'   # 黄金 (避险)
]

# 3. 从 50 列的大表中，只抽出这 13 列 （提取核心资产进行分析和展示）
macro_returns = returns_data[macro_tickers]

# 4. 计算相关性矩阵 (核心数学逻辑)
corr_matrix = macro_returns.corr()

# 5. 可视化：绘制高大上的金融热力图
plt.figure(figsize=(12, 9)) # 设置画布大小

# 画图参数解释：
# annot=True: 在格子里显示具体数字
# cmap='RdYlGn_r': 颜色映射，红(Red)代表高相关，绿(Green)代表低/负相关
# fmt=".2f": 数字保留两位小数
sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn_r', vmin=-0.5, vmax=1, center=0.5, fmt=".2f",
            linewidths=0.5, linecolor='white')

plt.title("GICS 11 Sectors & Macro Hedges Correlation Heatmap (1-Year)", fontsize=16, pad=20)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

# 弹出图表
plt.show()