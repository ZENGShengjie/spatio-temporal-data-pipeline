import os
from usgsxplore import USGS

# ============ 凭证从环境变量读取 ============
# 请设置环境变量：GEONAMES_USERNAME / USGS_TOKEN
# 注意：usgs_token 是 NASA Earthdata M2M API 的 JWT，长期 token 会过期
username = os.getenv("NASA_USERNAME")
usgs_token = os.getenv("USGS_TOKEN")
if not username or not usgs_token:
    raise ValueError("请先设置环境变量 NASA_USERNAME 和 USGS_TOKEN")
# ==========================================

usgs = USGS(username=username, token=usgs_token)

# 北京、洛杉矶范围
bbox_beijing = (115.7, 39.4, 117.4, 41.0)
bbox_la = (-118.6, 33.7, -117.7, 34.3)

# 检索影像
res_bj = usgs.search(
    dataset="landsat_ot_c2_l2",
    bbox=bbox_beijing,
    start_date="2024-01-01",
    end_date="2024-12-31",
    max_cloud_cover=20
)
res_la = usgs.search(
    dataset="landsat_ot_c2_l2",
    bbox=bbox_la,
    start_date="2024-01-01",
    end_date="2024-12-31",
    max_cloud_cover=20
)

# 仅写入影像ID清单，不下载影像
with open("scene_list.txt", "w", encoding="utf-8") as f:
    for item in res_bj + res_la:
        f.write(item["entityId"] + "\n")

print("影像清单保存完成")