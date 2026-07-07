"""Fast OSM extraction using osmium CLI for filtering, pyosmium for reading.

Strategy (much faster than full Python parsing):
  1. Use osmium tags-filter (C++) to extract:
     - amenity={restaurant,cafe,fast_food,bar,pub,food_court,biergarten,
                cinema,theatre,arts_centre,nightclub,coworking_space}
     - shop=* (any)
     - office=* (any)
     - highway={motorway,...,tertiary_link} (roads)
     into separate compact .osm files
  2. Use pyosmium to load each compact file and produce:
     - /home/ubuntu/amazon/raw_nyc/poi/nyc_poi.parquet  (5 categories)
     - /home/ubuntu/amazon/processed/osm_road_density.parquet
  3. Also use osmium tags-filter for residential building ways
"""
import os
import sys
import time
import warnings
import subprocess
import pandas as pd
import geopandas as gpd
import osmium
from shapely.geometry import LineString, box

warnings.filterwarnings("ignore")

PBF = "/home/ubuntu/amazon/raw_nyc/osm/nyc.osm.pbf"
OUT_POI = "/home/ubuntu/amazon/raw_nyc/poi"
OUT_OSM = "/home/ubuntu/amazon/processed"
TMP_DIR = "/home/ubuntu/amazon/osm_extracts"
os.makedirs(OUT_POI, exist_ok=True)
os.makedirs(OUT_OSM, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

GRID_PATH = "/home/ubuntu/amazon/grid_nyc/nyc_grid_500m.geojson"

NYC_BBOX = box(-74.26, 40.49, -73.70, 40.92)
NYC_BBOX_STR = "-74.26,40.49,-73.70,40.92"


def run(cmd, timeout=600):
    print("  $ " + " ".join(cmd))
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        print("  stderr:", r.stderr[:1000])
        raise RuntimeError("command failed")
    print("  ok (%.1fs)" % (time.time() - t0))
    return r.stdout


def extract_subsets():
    """Slice the 470MB pbf into small category-specific .osm files via osmium CLI."""
    print("[1/4] Filtering POI amenity + shop + office (osmium CLI) ...")
    poi_files = {
        "food_out": os.path.join(TMP_DIR, "poi_food.osm"),
        "shop_out": os.path.join(TMP_DIR, "poi_shop.osm"),
        "ent_out":  os.path.join(TMP_DIR, "poi_ent.osm"),
        "off_out":  os.path.join(TMP_DIR, "poi_office.osm"),
        "resid_out":os.path.join(TMP_DIR, "residential.osm"),
    }
    road_out = os.path.join(TMP_DIR, "roads.osm")
    for k, p in poi_files.items():
        if os.path.exists(p):
            print("  [skip] %s exists (%.1fMB)" % (k, os.path.getsize(p)/1e6))
    if not os.path.exists(road_out):
        print("  [filter] roads ...")
        run(["osmium", "tags-filter", PBF,
             "w/highway=motorway,trunk,primary,secondary,tertiary,"
             "unclassified,residential,living_street,"
             "motorway_link,trunk_link,primary_link,secondary_link,tertiary_link",
             "-o", road_out, "--overwrite"])

    return (poi_files["food_out"], poi_files["shop_out"],
            poi_files["ent_out"], poi_files["off_out"], road_out, poi_files["resid_out"])


def _in_bbox(x, y):
    return -74.26 <= x <= -73.70 and 40.49 <= y <= 40.92


def load_poi_nodes(path, category, default_extra=None):
    class H(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.records = []

        def node(self, n):
            if not n.location.valid():
                return
            x, y = n.location.lon, n.location.lat
            if not _in_bbox(x, y):
                return
            tags = {k: v for k, v in n.tags}
            r = {"osm_id": n.id, "lon": x, "lat": y, "category": category, "name": tags.get("name", "")}
            if default_extra:
                r.update(default_extra)
            r["amenity"] = tags.get("amenity", "")
            r["shop"]    = tags.get("shop", "")
            r["office"]  = tags.get("office", "")
            r["building"]= tags.get("building", "")
            self.records.append(r)
    h = H()
    h.apply_file(path, locations=True)
    return h.records


def extract_poi(food_p, shop_p, ent_p, off_p, resid_p):
    print("[2/4] Reading POI subsets via pyosmium ...")
    records = []
    records += load_poi_nodes(food_p, "food")
    records += load_poi_nodes(shop_p, "shopping")
    records += load_poi_nodes(ent_p, "entertainment")
    records += load_poi_nodes(off_p, "office")

    print("  reading residential ways ...")
    class RH(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.records = []
        def way(self, w):
            tags = {k: v for k, v in w.tags}
            b = tags.get("building")
            if b not in {"residential", "apartments", "house", "dormitory"}:
                return
            coords = []
            for n in w.nodes:
                if n.location.valid():
                    coords.append((n.location.lon, n.location.lat))
            if not coords:
                return
            cx = sum(c[0] for c in coords) / len(coords)
            cy = sum(c[1] for c in coords) / len(coords)
            if not _in_bbox(cx, cy):
                return
            self.records.append({
                "osm_id": w.id, "lon": cx, "lat": cy,
                "category": "residential",
                "name": tags.get("name", ""),
                "amenity": "", "shop": "", "office": "", "building": b,
            })
    h = RH()
    h.apply_file(resid_p, locations=True)
    records += h.records

    df = pd.DataFrame(records)
    if df.empty:
        print("  warning: no POI rows!")
        return df
    df = df.drop_duplicates(subset=["osm_id", "category"])
    print("  total POI: %d" % len(df))
    print("  by category:")
    print(df["category"].value_counts().to_string())
    out = os.path.join(OUT_POI, "nyc_poi.parquet")
    df.to_parquet(out, index=False)
    print("  saved: %s" % out)
    return df


def extract_roads(road_p):
    print("[3/4] Reading road ways via pyosmium ...")
    class RH(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.records = []
        def way(self, w):
            coords = []
            for n in w.nodes:
                if n.location.valid():
                    coords.append((n.location.lon, n.location.lat))
            if len(coords) < 2:
                return
            try:
                line = LineString(coords)
                bounds = line.bounds
                if bounds[2] < -74.26 or bounds[0] > -73.70 or \
                   bounds[3] < 40.49  or bounds[1] > 40.92:
                    return
                hw = w.tags.get("highway", "")
                self.records.append({
                    "osm_id": w.id, "highway": hw, "geometry": line,
                })
            except Exception:
                return
    h = RH()
    h.apply_file(road_p, locations=True)
    print("  road segments: %d" % len(h.records))
    if not h.records:
        return None

    gdf = gpd.GeoDataFrame(h.records, geometry="geometry", crs="EPSG:4326")

    print("  projecting to UTM 18N ...")
    grid = gpd.read_file(GRID_PATH)
    if "grid_id" not in grid.columns:
        grid = grid.reset_index().rename(columns={"index": "grid_id"})
    grid = grid.set_crs("EPSG:4326", allow_override=True)
    gdf_m  = gdf.to_crs(epsg=32618)
    grid_m = grid.to_crs(epsg=32618)
    gdf_m["length_m"] = gdf_m.geometry.length

    print("  spatial join (segments -> grids) ...")
    joined = gpd.sjoin(gdf_m, grid_m[["grid_id", "geometry"]], how="inner",
                       predicate="intersects")

    # total length + segment count
    seg_len = joined.groupby("grid_id")["length_m"].sum().reset_index()
    seg_len.columns = ["grid_id", "road_length_m"]
    seg_cnt = joined.groupby("grid_id")["osm_id"].nunique().reset_index()
    seg_cnt.columns = ["grid_id", "road_segment_count"]

    # road class breakdown — length per highway class per grid
    # 4 tiers: motorway/trunk (highway), primary/secondary (major), tertiary (minor), residential/service (local)
    TIER_MAP = {
        "motorway":      "tier_highway",
        "trunk":         "tier_highway",
        "primary":       "tier_major",
        "secondary":     "tier_major",
        "tertiary":      "tier_minor",
        "unclassified":  "tier_minor",
        "residential":   "tier_local",
        "living_street": "tier_local",
        "service":       "tier_local",
        "motorway_link": "tier_highway",
        "trunk_link":    "tier_highway",
        "primary_link":  "tier_major",
        "secondary_link":"tier_major",
        "tertiary_link": "tier_minor",
    }
    joined["tier"] = joined["highway"].map(TIER_MAP).fillna("tier_other")
    tier_len = joined.groupby(["grid_id", "tier"])["length_m"].sum().unstack(fill_value=0.0)
    tier_len.columns = [f"road_len_{c}_m" for c in tier_len.columns]
    tier_len = tier_len.reset_index()

    grid_m["area_m2"] = grid_m.geometry.area
    out = (grid_m[["grid_id", "area_m2"]]
           .merge(seg_len,   on="grid_id", how="left")
           .merge(seg_cnt,   on="grid_id", how="left")
           .merge(tier_len,  on="grid_id", how="left")
           )
    out["road_length_m"]        = out["road_length_m"].fillna(0)
    out["road_segment_count"]   = out["road_segment_count"].fillna(0).astype(int)
    out["road_density_km_per_km2"] = out["road_length_m"] / 1000.0 / (out["area_m2"] / 1e6)
    # ensure all tier columns exist
    for c in tier_len.columns[1:]:
        if c not in out.columns:
            out[c] = 0.0
        else:
            out[c] = out[c].fillna(0.0)

    out = grid[["grid_id"]].merge(out, on="grid_id", how="left").fillna(0)
    out_path = os.path.join(OUT_OSM, "osm_road_density.parquet")
    out.to_parquet(out_path, index=False)
    print("  saved: %s" % out_path)
    print("  road_density_km_per_km2  min=%.2f  max=%.2f  mean=%.2f" % (
        out["road_density_km_per_km2"].min(),
        out["road_density_km_per_km2"].max(),
        out["road_density_km_per_km2"].mean(),
    ))
    tier_cols = [c for c in out.columns if c.startswith("road_len_")]
    print("  road class columns:", tier_cols)
    return out


if __name__ == "__main__":
    (food_p, shop_p, ent_p, off_p, road_p, resid_p) = extract_subsets()
    poi_df = extract_poi(food_p, shop_p, ent_p, off_p, resid_p)
    roads  = extract_roads(road_p)
    print()
    print("Done.")
