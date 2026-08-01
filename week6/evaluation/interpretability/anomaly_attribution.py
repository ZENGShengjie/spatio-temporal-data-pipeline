"""Week6 任务3-3：异常事件归因"""
from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

# TaxiBJ 官方17类天气编码(来源:TaxiBJ/README.md):
# 0=Sunny, 1=Cloudy, 2=Overcast, 3=Rainy, 4=Sprinkle, 5=ModerateRain,
# 6=HeavyRain, 7=Rainstorm, 8=Thunderstorm, 9=FreezingRain, 10=Snowy,
# 11=LightSnow, 12=ModerateSnow, 13=HeavySnow, 14=Foggy, 15=Sandstorm, 16=Dusty
_WEATHER_LABELS = {
    0:"sunny",1:"cloudy",2:"overcast",3:"rainy",4:"sprinkle",
    5:"moderate_rain",6:"heavy_rain",7:"rainstorm",8:"thunderstorm",
    9:"freezing_rain",10:"snowy",11:"light_snow",12:"moderate_snow",
    13:"heavy_snow",14:"foggy",15:"sandstorm",16:"dusty",
}

VAL_END = 3288
TEST_END = 3888
START_DATE = datetime(2015, 11, 1)
TIME_STEP_MIN = 60

def load_holidays(path=None):
    candidates = [path] if path else []
    candidates += [
        Path("/home/ubuntu/data/raw_bj/BJ_Holiday.txt"),
        _REPO / "week5" / "data" / "BJ_Holiday.txt",
    ]
    p = next((c for c in candidates if c and Path(c).exists()), None)
    if p is None:
        print("[Attribution] 未找到BJ_Holiday.txt")
        return []
    print(f"[Attribution] 节假日文件: {p}")
    holidays = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        d = line.split("\t")[0].strip().split()[0]
        try:
            if "-" in d:
                dt = datetime.strptime(d, "%Y-%m-%d")
            elif len(d) == 8 and d.isdigit():
                dt = datetime.strptime(d, "%Y%m%d")
            else:
                continue
            holidays.append(dt.strftime("%Y-%m-%d"))
        except ValueError:
            continue
    return holidays

def load_weather(path=None):
    candidates = [path] if path else []
    candidates += [
        Path("/home/ubuntu/data/raw_bj/BJ_Meteorology.h5"),
        _REPO / "week5" / "data" / "BJ_Meteorology.h5",
    ]
    p = next((c for c in candidates if c and Path(c).exists()), None)
    if p is None:
        print("[Attribution] 气象文件不存在")
        return {}
    print(f"[Attribution] 气象文件: {p}")
    try:
        import h5py
        f = h5py.File(p, "r")
        dates = f["date"][:]
        w_onehot = f["Weather"][:]
        temp = f["Temperature"][:]
        per_day = {}
        for i in range(len(dates)):
            d_str = dates[i].decode() if isinstance(dates[i], bytes) else str(dates[i])
            ymd = d_str[:4] + "-" + d_str[4:6] + "-" + d_str[6:8]
            if len(ymd) != 10 or not ymd.replace("-","").isdigit():
                continue
            if ymd not in per_day:
                per_day[ymd] = {"ws":[], "ts":[]}
            wc = int(w_onehot[i].argmax())
            if wc > 0:
                per_day[ymd]["ws"].append(wc)
            per_day[ymd]["ts"].append(float(temp[i]))
        result = {}
        for day, vals in per_day.items():
            ws, ts = vals["ws"], vals["ts"]
            mode = max(set(ws), key=ws.count) if ws else 0
            result[day] = {
                "weather_class": mode,
                "weather_label": _WEATHER_LABELS.get(mode, "class_"+str(mode)),
                "temperature": sum(ts)/len(ts) if ts else None,
                "n_slots": len(ts),
            }
        print(f"[Attribution] 气象数据: {len(result)}天")
        return result
    except Exception as e:
        print(f"[Attribution] 气象加载失败: {e}")
        return {}

def load_timestamps():
    paths = [
        Path("/home/ubuntu/data/cleaned_bj/taxi_p4_4d.npz"),
        _REPO / "data" / "cleaned_bj" / "taxi_p4_4d.npz",
    ]
    for p in paths:
        if p.exists():
            try:
                d = np.load(str(p))
                if "timestamps" in d.keys():
                    return d["timestamps"]
            except Exception:
                continue
    return None

def t_to_date(t, ts=None):
    if ts is not None and 0 <= t < len(ts):
        tv = ts[t]
        if isinstance(tv, np.datetime64):
            return str(tv)[:10]
        if isinstance(tv, datetime):
            return tv.strftime("%Y-%m-%d")
    return (START_DATE + timedelta(minutes=TIME_STEP_MIN * t)).strftime("%Y-%m-%d")

def infer_level(e):
    nc = e.get("n_cells", 0)
    if nc >= 20: return 2
    if nc >= 16: return 1
    return 0

def analyze(events, holidays, weather, ts=None):
    print(f"[Attribution] 分析{len(events)}个异常事件")
    hol_set = set(holidays)
    n_hol = n_wknd = n_wkdy = 0
    hs_sum = ws_sum = wds_sum = 0.0
    hlvl = {1:0,2:0,3:0}
    wlvl = {1:0,2:0,3:0}
    for e in events:
        t_g = int(e["t_start"]) + VAL_END
        date = t_to_date(t_g, ts)
        score = e.get("avg_score", 0.0)
        lvl = infer_level(e)
        dt = datetime.strptime(date, "%Y-%m-%d")
        if date in hol_set:
            n_hol += 1; hs_sum += score
            if lvl in hlvl: hlvl[lvl] += 1
        elif dt.weekday() >= 5:
            n_wknd += 1; ws_sum += score
        else:
            n_wkdy += 1; wds_sum += score
            if lvl in wlvl: wlvl[lvl] += 1
    total = len(events)

    # TaxiBJ官方17类分组
    def wgrp(wc):
        if wc == 0: return "sunny"
        if wc in (1,2): return "cloudy_overcast"
        if wc in (3,4,5,6,7,8,9): return "rain_adverse"
        if wc == 14: return "foggy"
        if wc in (10,11,12,13): return "snow_adverse"
        return "other_weather"

    wg_keys = ["sunny","cloudy_overcast","rain_adverse","foggy","snow_adverse","other_weather","missing"]
    we = {k:0 for k in wg_keys}
    wd = {k:0 for k in wg_keys}
    test_dates = set()
    for e in events:
        t_g = int(e["t_start"]) + VAL_END
        test_dates.add(t_to_date(t_g, ts))
    for d in test_dates:
        if d in weather and weather[d].get("weather_class",0) > 0:
            wc = weather[d].get("weather_class",0)
            g = wgrp(wc); wd[g] += 1
        else:
            wd["missing"] += 1
    for e in events:
        t_g = int(e["t_start"]) + VAL_END
        date = t_to_date(t_g, ts)
        if date in weather and weather[date].get("weather_class",0) > 0:
            wc = weather[date].get("weather_class",0)
            g = wgrp(wc); we[g] += 1
        else:
            we["missing"] += 1
    n_missing = wd["missing"]
    severe = [e for e in events if infer_level(e) >= 2]
    sev_hol = sum(1 for e in severe if t_to_date(int(e["t_start"])+VAL_END,ts) in hol_set) / max(len(severe),1)
    return {
        "total_events": total,
        "holiday_distribution": {
            "n_holiday": n_hol, "n_weekend": n_wknd, "n_workday": n_wkdy,
            "holiday_ratio": n_hol/total,
            "avg_score_holiday": hs_sum/max(n_hol,1),
            "avg_score_workday": wds_sum/max(n_wkdy,1),
            "avg_score_weekend": ws_sum/max(n_wknd,1),
            "level_dist_holiday": hlvl,
            "level_dist_workday": wlvl,
        },
        "weather_distribution": {
            "events_per_class": we,
            "days_per_class": wd,
            "events_per_class_per_day": {k: we.get(k,0)/max(wd.get(k,0),1) for k in wg_keys},
        },
        "severe_events": {"total": len(severe), "holiday_ratio": sev_hol},
        "n_holidays_in_test_set": sum(1 for d in test_dates if d in hol_set),
        "n_test_days": len(test_dates),
        "n_days_missing_weather": n_missing,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="week6/data/events_test_v1.json")
    ap.add_argument("--output", default="week6.evaluation/results/interpretability/attribution/")
    args = ap.parse_args()
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    ev_path = _REPO / args.events
    if not ev_path.exists():
        print(f"[Attribution] 事件文件不存在: {ev_path}")
        (out_dir/"attribution_report.json").write_text(
            json.dumps({"error":"events not found"}, ensure_ascii=False, indent=2),
            encoding="utf-8"); return
    events = json.load(open(ev_path, encoding="utf-8"))
    ts = load_timestamps()
    if ts is not None:
        print(f"[Attribution] 时间戳: {len(ts)}步")
        print(f"[Attribution] 测试窗口: {str(ts[VAL_END])[:10]}~{str(ts[TEST_END-1])[:10]}")
    else:
        print("[Attribution] 时间戳加载失败")
    holidays = load_holidays()
    print(f"[Attribution] 节假日: {len(holidays)}个")
    weather = load_weather()
    print(f"[Attribution] 气象: {len(weather)}天")
    result = analyze(events, holidays, weather, ts)
    hol_set = set(holidays)
    top3 = sorted(events, key=lambda e: -e.get("n_cells",0))[:3]
    result["typical_cases"] = []
    for e in top3:
        t_g = int(e["t_start"]) + VAL_END
        date = t_to_date(t_g, ts)
        dt = datetime.strptime(date, "%Y-%m-%d")
        wc = weather.get(date, {}).get("weather_class", None)
        result["typical_cases"].append({
            "event_id": e.get("event_id"),
            "t_start": e.get("t_start"),
            "t_global": t_g,
            "date": date,
            "is_holiday": date in hol_set,
            "is_weekend": dt.weekday() >= 5,
            "weather_class": wc,
            "weather_label": _WEATHER_LABELS.get(wc,"unknown") if wc is not None else "unknown",
            "n_cells": e.get("n_cells"),
            "avg_score": e.get("avg_score"),
            "warning_level": infer_level(e),
        })
    (out_dir/"attribution_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[Attribution] 完成")
    print(f"  总事件: {result['total_events']}")
    print(f"  节假日占比: {result['holiday_distribution']['holiday_ratio']*100:.1f}%")
    print(f"  节假日均分: {result['holiday_distribution']['avg_score_holiday']:.3f}")
    print(f"  工作日均分: {result['holiday_distribution']['avg_score_workday']:.3f}")
    print(f"  周末均分: {result['holiday_distribution']['avg_score_weekend']:.3f}")
    print(f"  天气分布: {result['weather_distribution']['events_per_class']}")
    print(f"  无气象数据天数: {result['n_days_missing_weather']}")

if __name__ == "__main__":
    main()
