import pymrio
import pandas as pd
import numpy as np
import os
import uuid

# === 1. 定义区域和部门分类 ===
regions = ['Brazil', 'China', 'EU27 & UK', 'India', 'Japan', 'ROW', 'Russia', 'United States']

# EXIOBASE国家索引（1基索引）
region_country_indices = {
    'Brazil': [34],
    'China': [31],
    'EU27 & UK': list(range(1, 13)) + list(range(14, 29)) + [13],  # 28个国家
    'India': [35],
    'Japan': [30],
    'ROW': [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
    'Russia': [37],
    'United States': [29]
}

# 部门聚合
sector_sets = {
    'Aviation': [125],
    'Ground Transport': [120, 121, 122],
    'Power': [96, 112],
    'Residential & others': [119, 137, 138, 160, 161, 162],
    'Industry': list(range(20, 96)) + [113, 114]
}

# 常量
nRegions = len(regions)  # 8
nSectors = len(sector_sets)  # 5
nCountries = 49
nPerCountry = 163
nOriginal = nCountries * nPerCountry  # 7987

# 国家代码
exiobase_countries = [
    'AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GR',
    'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 'PL', 'PT', 'RO',
    'SE', 'SI', 'SK', 'GB', 'US', 'JP', 'CN', 'CA', 'KR', 'BR', 'IN', 'MX',
    'RU', 'AU', 'CH', 'TR', 'TW', 'NO', 'ID', 'ZA', 'WA', 'WL', 'WM', 'WF', 'WE'
]
country_to_region = {}
for region, indices in region_country_indices.items():
    for idx in indices:
        country_code = exiobase_countries[idx - 1]
        country_to_region[country_code] = region

# === 2. 构造S矩阵 ===
print("正在构造S矩阵...")
S = np.zeros((nOriginal, nRegions * nSectors))  # (7987, 40)
for r, region in enumerate(regions):
    region_idx = region_country_indices[region]
    for s, sector in enumerate(sector_sets):
        sector_rows = np.array(sector_sets[sector])
        col = r * nSectors + s
        for region_country in region_idx:
            idx = (region_country - 1) * nPerCountry + sector_rows
            S[idx, col] = 1
print(f"✅ S矩阵形状: {S.shape}")
eu_idx = regions.index('EU27 & UK')
eu_nonzero = np.sum(S[:, eu_idx * nSectors:(eu_idx + 1) * nSectors])
print(f"EU27 & UK 在S矩阵中的非零项数: {eu_nonzero} (预期: {28 * sum(len(sector_sets[s]) for s in sector_sets)})")
for region, indices in region_country_indices.items():
    print(f"{region} 包含国家数: {len(indices)}")

# === 3. 读取EXIOBASE 2022数据 ===
base_path = "/Users/guoguo/Desktop/碳排放科研/1970-2022consume/IOT_1995_ixi"
Z_path = f"{base_path}Z.txt"
Y_path = f"{base_path}Y.txt"
F_path = f"{base_path}satellite/F.txt"

# 读取Z矩阵
print("正在读取Z.txt...")
Z_raw = pd.read_csv(Z_path, skiprows=3, sep='\t', header=None)
Z = Z_raw.iloc[:, 2:].values
print(f"✅ Z矩阵形状: {Z.shape}")
eu_rows = []
for idx in region_country_indices['EU27 & UK']:
    eu_rows.extend(range((idx - 1) * nPerCountry, idx * nPerCountry))
print(f"Z矩阵中EU27 & UK行数据总和: {Z[eu_rows, :].sum()}")

# 读取Y矩阵
print("正在读取Y.txt...")
Y_raw = pd.read_csv(Y_path, sep="\t", header=[0, 1], skiprows=[2])
regions_for_fd = [c[0] for c in Y_raw.columns[2:]]
fd_names = [c[1] for c in Y_raw.columns[2:]]
n_fd_categories = len(set(fd_names))
print(f"最终需求类别数: {n_fd_categories}")
if not all(c in exiobase_countries for c in set(regions_for_fd)):
    print("警告：Y.txt中的国家代码与exiobase_countries不完全匹配，请检查！")
    print("Y.txt中的国家代码：", sorted(set(regions_for_fd)))
eu_countries = [exiobase_countries[idx - 1] for idx in region_country_indices['EU27 & UK']]
eu_cols = [i for i, region in enumerate(regions_for_fd) if region in eu_countries]
print(f"Y矩阵中EU27 & UK国家列数: {len(eu_cols)} (预期: {28 * n_fd_categories})")
Y = Y_raw.iloc[:, 2:].values
print(f"✅ Y矩阵形状: {Y.shape}")
print(f"Y矩阵中EU27 & UK数据总和: {Y[:, eu_cols].sum()}")

# === 修复：Y矩阵列聚合 ===
region_mapping = pd.Series(index=exiobase_countries, dtype='object')
for country, region in country_to_region.items():
    region_mapping[country] = region
region_mapping.fillna('ROW', inplace=True)
# 只对数据列设置MultiIndex
Y_data = Y_raw.iloc[:, 2:].copy()
Y_data.columns = pd.MultiIndex.from_arrays(
    [regions_for_fd, fd_names],
    names=["region", "final_demand"]
)
# 映射到聚合区域
Y_data_agg = Y_data.copy()
Y_data_agg.columns = pd.MultiIndex.from_arrays(
    [region_mapping[regions_for_fd].values, fd_names],
    names=["region", "final_demand"]
)
# 按区域和最终需求类别聚合
Y_agg_df = Y_data_agg.groupby(level=["region", "final_demand"], axis=1).sum()
new_cols = pd.MultiIndex.from_product(
    [regions, sorted(set(fd_names))],
    names=["region", "final_demand"]
)
Y_agg_df = Y_agg_df.reindex(columns=new_cols, fill_value=0)
Y_agg = S.T @ Y_agg_df.values  # (40, 56)
Y_multi_cols = new_cols
print(f"Y_agg中EU27 & UK数据总和: {Y_agg[:, eu_idx * n_fd_categories:(eu_idx + 1) * n_fd_categories].sum()}")

# 读取F矩阵
print("正在读取F.txt...")
F_raw = pd.read_csv(F_path, skiprows=2, sep='\t', header=None)
F = F_raw.iloc[:, 1:].fillna(0).values
print(f"✅ F矩阵形状: {F.shape}")
print(f"F矩阵中EU27 & UK列数据总和: {F[:, eu_rows].sum()}")

# === 4. 聚合Z、Y、F矩阵 ===
print("正在将Z、Y、F矩阵聚合到8区域×5部门...")
Z_agg = S.T @ Z @ S
F_agg = F @ S
print(f"✅ Z_agg形状: {Z_agg.shape}, Y_agg形状: {Y_agg.shape}, F_agg形状: {F_agg.shape}")
print(f"Z_agg中EU27 & UK数据总和: {Z_agg[:, eu_idx * nSectors:(eu_idx + 1) * nSectors].sum()}")
print(f"Y_agg中EU27 & UK数据总和: {Y_agg[:, eu_idx * n_fd_categories:(eu_idx + 1) * n_fd_categories].sum()}")
print(f"F_agg中EU27 & UK数据总和: {F_agg[:, eu_idx * nSectors:(eu_idx + 1) * nSectors].sum()}")

# === 5. 创建MultiIndex ===
sector_labels = [f"{r} - {s}" for r in regions for s in sector_sets.keys()]
multi_cols = pd.MultiIndex.from_tuples(
    [(r, s) for r in regions for s in sector_sets.keys()],
    names=["region", "sector"]
)

# === 6. 包装为DataFrame ===
Z_df = pd.DataFrame(Z_agg, index=multi_cols, columns=multi_cols)
Y_df = pd.DataFrame(Y_agg, index=multi_cols, columns=Y_multi_cols)
F_df = pd.DataFrame(F_agg, index=F_raw.index, columns=multi_cols)

# === 7. 创建IOSystem ===
print("正在创建2022年的IOSystem...")
io_2022 = pymrio.IOSystem(Z=Z_df, Y=Y_df)
io_2022.emissions = pymrio.Extension(name='Emissions', F=F_df)
io_2022.emissions.unit = pd.DataFrame(
    ['Kt'] * F_agg.shape[0],
    index=F_raw.index,
    columns=['unit']
)

# 保存IOSystem
output_path = '/Users/guoguo/Desktop/碳排放科研/1970-2022consume/processed_io_1995_agg'
os.makedirs(output_path, exist_ok=True)
io_2022.save_all(path=output_path)
print(f"✅ IOSystem已保存至: {output_path}")

# === 8. 处理2023年Carbon Monitor数据 ===
print("正在处理2023年Carbon Monitor数据...")
cm_path_2023 = "/Users/guoguo/Desktop/碳排放科研/1970-2022consume/carbon-monitor-GLOBAL1995.xlsx"
df_cm = pd.read_excel(cm_path_2023)
df_2023 = (
    df_cm.groupby(["country", "sector"])["MtCO2 per day"]
    .sum()
    .reset_index()
    .rename(columns={"country": "region", "sector": "sector_agg", "MtCO2 per day": "MtCO2_2023"})
)
F_2023_agg = pd.Series(0.0, index=sector_labels)
for _, row in df_2023.iterrows():
    region = row["region"]
    sec = row["sector_agg"]
    if sec == "Residential":
        sec = "Residential & others"
    col_name = f"{region} - {sec}"
    if col_name in F_2023_agg.index:
        F_2023_agg[col_name] += row["MtCO2_2023"]
F_2023_df = pd.DataFrame(F_2023_agg.values.reshape(1, -1), index=["CO2_2023"], columns=multi_cols)
print(f"✅ 2023年Carbon Monitor聚合排放量:")
print(f"2023 EU27 & UK 数据总和: {F_2023_df.iloc[:, eu_idx * nSectors:(eu_idx + 1) * nSectors].sum().sum()}")

# === 9. 计算2023年消费端排放 ===
print("正在计算2023年消费端排放...")
io_2022.emissions = pymrio.Extension(name="Emissions_2023", F=F_2023_df)
io_2022.emissions.unit = pd.DataFrame(
    ['MtCO2'] * F_2023_df.shape[0],
    index=F_2023_df.index,
    columns=['unit']
)
io_2022.calc_all()
D_cba_2023 = io_2022.emissions.D_cba
print(f"D_cba_2023中EU27 & UK数据总和: {D_cba_2023.iloc[:, eu_idx * n_fd_categories:(eu_idx + 1) * n_fd_categories].sum().sum()}")
output_dir = "/Users/guoguo/Desktop/碳排放科研/2023-2024consume/results"
os.makedirs(output_dir, exist_ok=True)
output_file = f"{output_dir}/D_cba_2023.csv"
D_cba_2023.to_csv(output_file)
print(f"✅ 2023年消费端排放已保存至: {output_file}")