import os
import pandas as pd

# 读取我们刚刚生成的对齐矩阵
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_parquet = os.path.join(base_dir, "results_archive", "06_time_alignment", "aligned_timeseries.parquet")

df = pd.read_parquet(input_parquet)

print("============== 数据分布 X光透视 ==============")
# 打印核心变量的最大值、最小值、平均值等统计信息
pd.set_option('display.max_columns', None)
print(df.describe())

print("\n============== 缺失值 (NaN) 统计 ==============")
# 查看是不是某个变量全军覆没
missing_stats = df.isna().sum()
missing_percent = (missing_stats / len(df)) * 100
stats_df = pd.DataFrame({'Missing_Count': missing_stats, 'Missing_Percent(%)': missing_percent})
print(stats_df)