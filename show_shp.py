import geopandas as gpd

# 读取 shp 文件
gdf = gpd.read_file("shp/霍城县地块.shp")

# 打印属性列表（字段名和数据类型）
print("=== 属性字段列表 ===")
print(gdf.dtypes)
print()

# 判断是面、线还是点
print("=== 几何类型 ===")
print(gdf.geom_type.unique())