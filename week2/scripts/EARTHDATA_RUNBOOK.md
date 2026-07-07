# Earthdata Login 接入指南

## 注册账号

1. 打开 https://urs.earthdata.nasa.gov/registration
2. 填好 username + email + password, 提交 (约 2 分钟)
3. 邮箱激活链接点一下

## 在 EC2 上配 `.netrc`

注册后, 在 EC2 终端 (你已经 SSH 进去的状态):

```bash
mkdir -p ~/.ssh   # 避免 .netrc 权限错
cat > ~/.netrc <<'EOF'
machine urs.earthdata.nasa.gov login <你的USERNAME> password <你的PASSWORD>
EOF
chmod 600 ~/.netrc
```

## 安装依赖

```bash
pip install earthaccess h5py pyhdf rasterio -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 测试登录

```bash
python3 -c "import earthaccess; auth=earthaccess.login(strategy='netrc'); print('OK' if auth else 'FAIL')"
```

输出 `OK` 表示成功. 失败的话:

```
EARTHDATA_USERNAME=<USERNAME> EARTHDATA_PASSWORD=<PASSWORD> python3 -c "import earthaccess; print(earthaccess.login(strategy='environment'))"
```

## 跑 NDVI 拉取

```bash
cd /home/ubuntu/amazon
python3 scripts/step3b_ndvi.py 2>&1 | tee ndvi_run.log
```

预期耗时:
- 登录 + 搜索: 5 秒
- 下载 25 个 .hdf: 5-15 分钟 (NYC 只需要 h12v04、h12v05、h11v04)
- HDF -> GeoTIFF: 1 分钟
- 网格提取: 3 分钟
- 聚合 + 保存: 5 秒

输出:
- `cleaned_nyc/ndvi_clean.parquet` (列: grid_id, ndvi_mean, ndvi_max, ndvi_min, ndvi_count, sample_months)
- `cleaned_nyc/ndvi_fetch_log.json`

## 失败排查

| 报错 | 原因 |
|---|---|
| `401 Unauthorized` | .netrc 错或密码错 |
| `No module 'earthaccess'` | 跳了 pip install |
| `No module 'pyhdf'` | 跳了 pip install pyhdf |
| `EOFError / KeyboardInterrupt` | 进入了交互登录, 用 `strategy='netrc'` |
| 搜索返 0 结果 | tile id 是 hXXvYY, NYC 在 h11v04/h12v04/h12v05; 检查 bbox |
