"""验证: 用 pyproj 把 h12v04 G-ring bbox 4 角点转 Sinusoidal 米制"""
from pyproj import Transformer
# MODIS Sinusoidal = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +no_defs"
tf = Transformer.from_crs("EPSG:4326", "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +no_defs", always_xy=True)
# h12v04 G-ring 角点: UL(-93.38, 50.00), UR(-77.75, 50.08), LL(-78.21, 39.79), LR(-65.08, 39.84)
print("UL:", tf.transform(-93.38, 50.00))
print("UR:", tf.transform(-77.75, 50.08))
print("LL:", tf.transform(-78.21, 39.79))
print("LR:", tf.transform(-65.08, 39.84))