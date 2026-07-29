"""
============================================================
Python 数据分析 阶段二：NumPy + Pandas 手把手教程
============================================================
我们用一个模拟的电商订单数据集来学习所有核心操作。
"""

import numpy as np
import pandas as pd

# ============================================================
# 第 1 课：NumPy 快速入门（够用就行）
# ============================================================
print("=" * 60)
print("第 1 课：NumPy 基础")
print("=" * 60)

# --- 创建数组 ---
arr = np.array([1, 2, 3, 4, 5])
print(f"数组: {arr}")
print(f"类型: {arr.dtype}, 形状: {arr.shape}")

# --- 向量化运算（这就是比 Python 循环快的原因）---
print(f"每个元素 × 2: {arr * 2}")
print(f"每个元素 + 10: {arr + 10}")
print(f"每个元素 > 3: {arr > 3}")

# --- 二维数组 ---
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\n二维数组:\n{arr2d}")
print(f"形状: {arr2d.shape}")  # (2行, 3列)

# --- 常用聚合 ---
scores = np.array([85, 92, 78, 88, 95, 72, 83])
print(f"\n考试成绩: {scores}")
print(f"均值: {scores.mean()}, 标准差: {scores.std():.2f}")
print(f"最高: {scores.max()}, 最低: {scores.min()}, 中位数: {np.median(scores)}")

# --- 生成序列 ---
print(f"\n0到1等间隔5个数: {np.linspace(0, 1, 5)}")
print(f"全零数组 (2x3):\n{np.zeros((2, 3))}")

print("\n学完 NumPy 只需要记住：数组 + 向量化运算 + 聚合函数，就够了。\n")


# ============================================================
# 第 2 课：Pandas Series 和 DataFrame
# ============================================================
print("=" * 60)
print("第 2 课：Pandas — Series 与 DataFrame")
print("=" * 60)

# --- Series：一列数据 ---
sales = pd.Series([1200, 3400, 2100, 4500], name="销售额")
print(f"Series:\n{sales}\n")

# --- DataFrame：表格 ---
df = pd.DataFrame({
    "商品": ["手机", "电脑", "耳机", "键盘"],
    "价格": [4999, 7999, 299, 599],
    "销量": [120, 45, 300, 200],
})
print(f"DataFrame:\n{df}\n")

# --- 基本查看 ---
print(f"形状: {df.shape}")          # (行数, 列数)
print(f"列名: {df.columns.tolist()}")
print(f"数据类型:\n{df.dtypes}\n")


# ============================================================
# 第 3 课：生成一个真实的电商数据集（后面的操作都用它）
# ============================================================
print("=" * 60)
print("第 3 课：生成模拟电商数据")
print("=" * 60)

np.random.seed(42)
n = 500  # 500 条订单

orders = pd.DataFrame({
    "订单ID":     [f"ORD-{i:04d}" for i in range(1, n + 1)],
    "日期":       pd.date_range("2025-01-01", periods=n, freq="h"),
    "商品品类":   np.random.choice(["电子产品", "服装", "食品", "图书", "家居"], n),
    "单价":       np.random.uniform(10, 5000, n).round(2),
    "数量":       np.random.randint(1, 6, n),
    "用户城市":   np.random.choice(["北京", "上海", "广州", "深圳", "杭州"], n),
    "用户等级":   np.random.choice(["普通", "银卡", "金卡", "钻石"], n, p=[0.5, 0.3, 0.15, 0.05]),
    "是否退货":   np.random.choice([True, False], n, p=[0.08, 0.92]),
})

# 加上一些缺失值
orders.loc[np.random.choice(n, 15, replace=False), "用户等级"] = np.nan
orders.loc[np.random.choice(n, 10, replace=False), "单价"] = np.nan

# 计算金额列
orders["金额"] = orders["单价"] * orders["数量"]

print(f"生成了 {len(orders)} 条订单数据\n")
print(orders.head(10))
print(f"\n数据概览：")
print(f"  时间范围: {orders['日期'].min().date()} ~ {orders['日期'].max().date()}")
print(f"  品类: {orders['商品品类'].nunique()} 个")
print(f"  缺失 '用户等级': {orders['用户等级'].isnull().sum()} 条")
print(f"  缺失 '单价': {orders['单价'].isnull().sum()} 条")


# ============================================================
# 第 4 课：数据查看与探索
# ============================================================
print("\n" + "=" * 60)
print("第 4 课：数据查看 — 拿到数据第一步做什么")
print("=" * 60)

# head() / tail() — 看前几行后几行
print(f">>> orders.head(3)\n{orders.head(3)}\n")
print(f">>> orders.tail(3)\n{orders.tail(3)}\n")

# info() — 看每列数据类型、缺失情况、内存占用
print(">>> orders.info()")
orders.info()

# describe() — 看数值列的统计摘要
print("\n>>> orders.describe()")
print(orders.describe())

# value_counts() — 看分类列的分布
print("\n>>> orders['商品品类'].value_counts()")
print(orders["商品品类"].value_counts())

# nunique() — 唯一值数量
print(f"\n唯一城市数: {orders['用户城市'].nunique()}")


# ============================================================
# 第 5 课：数据清洗（最常用的部分）
# ============================================================
print("\n" + "=" * 60)
print("第 5 课：数据清洗")
print("=" * 60)

# --- 查看缺失值 ---
print("每列缺失值数量:")
print(orders.isnull().sum())

print(f"\n缺失 '单价' 的行:")
print(orders[orders["单价"].isnull()][["订单ID", "单价", "金额"]])

# --- 处理缺失值 ---
# 方法1：用均值填充
mean_price = orders["单价"].mean()
print(f"\n单价均值: {mean_price:.2f}")
orders["单价"] = orders["单价"].fillna(mean_price)

# 方法2：用众数填充分类列
mode_level = orders["用户等级"].mode()[0]  # mode() 返回众数 Series
orders["用户等级"] = orders["用户等级"].fillna(mode_level)

# 方法3：直接删除（谨慎使用）
# orders.dropna()  # 删除任何有缺失的行
# orders.dropna(subset=["单价"])  # 只删除"单价"缺失的行

# 重新计算金额
orders["金额"] = orders["单价"] * orders["数量"]

print(f"填充后缺失值:")
print(orders.isnull().sum())

# --- 重复值 ---
print(f"\n重复行数: {orders.duplicated().sum()}")
# orders.drop_duplicates()  # 如果有就去重

# --- 类型转换 ---
orders["日期"] = pd.to_datetime(orders["日期"])
print(f"日期列类型: {orders['日期'].dtype}")


# ============================================================
# 第 6 课：筛选与排序
# ============================================================
print("\n" + "=" * 60)
print("第 6 课：筛选与排序")
print("=" * 60)

# --- 布尔索引 ---
high_value = orders[orders["金额"] > 5000]
print(f"金额 > 5000 的订单: {len(high_value)} 条")

# --- 多条件 (& 是且, | 是或, ~ 是非) ---
elec_beijing = orders[(orders["商品品类"] == "电子产品") & (orders["用户城市"] == "北京")]
print(f"北京 + 电子产品 的订单: {len(elec_beijing)} 条")

# --- isin() — 在一组值里面 ---
top_cities = orders[orders["用户城市"].isin(["北京", "上海"])]
print(f"北京或上海的订单: {len(top_cities)} 条")

# --- loc[] 按标签, iloc[] 按位置 ---
print(f"\n>>> orders.loc[0]  (第一行，按索引标签)")
print(orders.loc[0])
print(f"\n>>> orders.iloc[0]  (第一行，按数字位置)")
print(orders.iloc[0])

# loc 切指定行和列
print(f"\n>>> orders.loc[0:3, ['订单ID', '金额']]")
print(orders.loc[0:3, ["订单ID", "金额"]])

# --- query() — 用字符串写筛选（有时更直观）---
high_qty = orders.query("数量 >= 3 and 商品品类 == '食品'")
print(f"\n食品 + 数量>=3 的订单: {len(high_qty)} 条")

# --- 排序 ---
print(f"\n按金额降序 Top 5:")
print(orders.sort_values("金额", ascending=False).head(5))


# ============================================================
# 第 7 课：分组聚合（groupby — 最重要的操作）
# ============================================================
print("\n" + "=" * 60)
print("第 7 课：分组聚合 groupby")
print("=" * 60)

# --- 单个分组 ---
print("各品类销售额总和:")
print(orders.groupby("商品品类")["金额"].sum().sort_values(ascending=False))

# --- 多个聚合 ---
print("\n各品类销售额（总和、均值、订单数）:")
category_stats = orders.groupby("商品品类").agg(
    总销售额=("金额", "sum"),
    平均客单价=("金额", "mean"),
    订单数=("金额", "count"),
).sort_values("总销售额", ascending=False)
print(category_stats)

# --- 多维度分组 ---
print("\n各品类 × 各城市的销售额:")
city_cat = orders.groupby(["商品品类", "用户城市"])["金额"].sum().head(10)
print(city_cat)

# --- transform — 分组后保持原形状 ---
orders["品类均价"] = orders.groupby("商品品类")["金额"].transform("mean")
print(f"\n添加了 '品类均价' 列:")
print(orders[["商品品类", "金额", "品类均价"]].head(8))


# ============================================================
# 第 8 课：合并（merge / concat / join）
# ============================================================
print("\n" + "=" * 60)
print("第 8 课：合并数据")
print("=" * 60)

# 创建第二张表：品类折扣表
discount = pd.DataFrame({
    "商品品类": ["电子产品", "服装", "食品", "图书", "家居"],
    "折扣率": [0.95, 0.8, 0.9, 0.85, 0.75],
})
print("折扣表:")
print(discount)

# --- merge — 像 SQL JOIN ---
orders_with_discount = orders.merge(discount, on="商品品类", how="left")
orders_with_discount["折后金额"] = orders_with_discount["金额"] * orders_with_discount["折扣率"]
print(f"\nmerge 后:")
print(orders_with_discount[["订单ID", "商品品类", "金额", "折扣率", "折后金额"]].head(8))

# --- concat — 纵向/横向拼接 ---
# 比如把北京和上海的数据拼在一起（实际就是上面 isin 的效果）
beijing = orders[orders["用户城市"] == "北京"].head(3)
shanghai = orders[orders["用户城市"] == "上海"].head(3)
concatenated = pd.concat([beijing, shanghai], ignore_index=True)
print(f"\nconcat 北京+上海: {len(concatenated)} 行")


# ============================================================
# 第 9 课：透视表
# ============================================================
print("\n" + "=" * 60)
print("第 9 课：透视表 pivot_table")
print("=" * 60)

# --- 透视表：行=品类, 列=用户等级, 值=平均金额 ---
pivot = orders.pivot_table(
    values="金额",
    index="商品品类",
    columns="用户等级",
    aggfunc="mean",
)
print("各品类 × 各用户等级 的平均订单金额:")
print(pivot.round(0))

# --- 带边际汇总 ---
pivot2 = orders.pivot_table(
    values="金额",
    index="商品品类",
    columns="用户城市",
    aggfunc="sum",
    margins=True,      # 加合计行
    margins_name="合计",
)
print("\n各品类 × 各城市 销售额 (带合计):")
print(pivot2.round(0))


# ============================================================
# 第 10 课：时间序列操作
# ============================================================
print("\n" + "=" * 60)
print("第 10 课：时间序列")
print("=" * 60)

# 设置日期为索引
ts = orders.set_index("日期").copy()

# --- 按天聚合 ---
daily_sales = ts["金额"].resample("D").sum()
print("每日销售额:")
print(daily_sales.head(10))

# --- 提取时间特征 ---
orders["月份"] = orders["日期"].dt.month
orders["星期"] = orders["日期"].dt.day_name()
orders["小时"] = orders["日期"].dt.hour

print(f"\n各月份订单数:")
print(orders["月份"].value_counts().sort_index())

print(f"\n各星期订单数:")
print(orders["星期"].value_counts())


# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("阶段二学习总结")
print("=" * 60)

summary = """
NumPy：
  数组(np.array) × 向量化运算 × 聚合函数(mean/sum/std)

Pandas 核心操作（按使用频率排序）：
  ① head() / info() / describe()       — 拿到数据先看这三样
  ② isnull() / fillna() / dropna()     — 处理缺失值
  ③ loc[] / iloc[] / 布尔索引          — 筛选数据
  ④ groupby().agg()                    — 分组聚合（最重要的）
  ⑤ sort_values()                      — 排序
  ⑥ merge()                            — 关联表
  ⑦ pivot_table()                      — 透视表
  ⑧ value_counts()                     — 看分布
  ⑨ pd.to_datetime() + dt 属性        — 时间处理

接下来你要做的练习：
  → 运行本文件，看每部分的输出
  → 修改筛选条件、分组字段，看看结果怎么变
  → 问自己：各品类退货率是多少？哪个城市人均消费最高？
"""
print(summary)
