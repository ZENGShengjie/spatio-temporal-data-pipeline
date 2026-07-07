# nasa_download.py 基于opengeos/NASA-Earth-Data封装
from nasa_earth_data import EarthData
import os

# =====================凭证从环境变量读取=====================
# 请先设置环境变量：NASA_USERNAME / NASA_PASSWORD
# Windows PowerShell: $env:NASA_USERNAME="xxx"; $env:NASA_PASSWORD="xxx"
# Linux/Mac: export NASA_USERNAME=xxx; export NASA_PASSWORD=xxx
USERNAME = os.getenv("NASA_USERNAME")
PASSWORD = os.getenv("NASA_PASSWORD")
if not USERNAME or not PASSWORD:
    raise ValueError("请先设置环境变量 NASA_USERNAME 和 NASA_PASSWORD")
# ==============================================================

# 初始化工具并登录NASA Earthdata
ed = EarthData()
ed.login(username=USERNAME, password=PASSWORD)

# 定义检索参数
search_params = {
    "dataset": "landsat_ot_c2_l2",  # Landsat8 OLI/TIRS C2二级产品
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "max_cloud": 20  # 云量不超过20%
}

# 1.检索北京区域 (经度min,纬度min,经度max,纬度max)
bj_bbox = [115.7, 39.4, 117.4, 41.0]
print("正在检索北京区域影像...")
bj_scenes = ed.search(
    bbox=bj_bbox,
    **search_params
)

# 2.检索洛杉矶区域
la_bbox = [-118.6, 33.7, -117.7, 34.3]
print("正在检索洛杉矶区域影像...")
la_scenes = ed.search(
    bbox=la_bbox,
    **search_params
)

# 合并全部影像
all_scenes = bj_scenes + la_scenes
print(f"总计检索到 {len(all_scenes)} 景符合条件的卫星影像")

# 导出scene_list.txt（你之前需要的影像ID清单）
with open("scene_list.txt", "w", encoding="utf-8") as f:
    for scene in all_scenes:
        f.write(scene["entity_id"] + "\n")
print("影像ID清单已保存至 scene_list.txt")

# 批量下载原始影像到当前文件夹
out_dir = "./raw_landsat"
os.makedirs(out_dir, exist_ok=True)
print("开始批量下载原始Landsat数据...")
ed.download(all_scenes, out_dir=out_dir)
print("全部任务执行完成！")