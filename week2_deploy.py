"""
Week 2 — 一键部署脚本：将所有脚本上传到 EC2 t3.large 并执行

使用前提：
  1. t3.large 已启动 (i-0a0843193c1223cea)
  2. 本地 AWS profile 可用: cursor-claude-agent

用法（PowerShell）：
    python deploy.py
"""
import os
import sys
import subprocess
import time
import json
import boto3

PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # e:\amazon
EC2_IP       = None
PROFILE      = "cursor-claude-agent"
KEY_PATH     = r"C:\Users\86139\.ssh\aws-spatio-key.pem"
SSH_USER     = "ubuntu"
REMOTE_DIR   = "/home/ubuntu/amazon"
INSTANCE_ID  = "i-0a0843193c1223cea"
BUCKET       = "spatio-data"
S3_PREFIX    = "amazon"

REQUIRED_PACKAGES = [
    "numpy", "pandas", "geopandas", "pyarrow", "pyarrow-parquet",
    "scikit-learn", "scipy",
    "torch", "torch-geometric", "torch-scatter", "torch-sparse",
    "shapely", "pyproj",
]


def get_ec2_ip():
    global EC2_IP
    ec2 = boto3.client("ec2", region_name="us-east-1")
    r = ec2.describe_instances(InstanceIds=[INSTANCE_ID])
    inst = r["Reservations"][0]["Instances"][0]
    EC2_IP = inst.get("PublicIpAddress")
    if not EC2_IP:
        sys.exit(f"ERROR: EC2 {INSTANCE_ID} 没有公网 IP (状态: {inst['State']['Name']})")
    print(f"EC2 IP: {EC2_IP} (状态: {inst['State']['Name']})")
    return EC2_IP


def run(cmd, check=True):
    print(f"  $ {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=False)
    if check and r.returncode != 0:
        sys.exit(f"ERROR: 命令失败: {cmd}")
    return r.returncode == 0


def scp(local_path, remote_path):
    cmd = (
        f'scp -o StrictHostKeyChecking=no -i "{KEY_PATH}" '
        f'"{local_path}" {SSH_USER}@{EC2_IP}:{remote_path}'
    )
    run(cmd)


def ssh(cmd):
    full = (
        f'ssh -o StrictHostKeyChecking=no -i "{KEY_PATH}" '
        f'{SSH_USER}@{EC2_IP} "{cmd}"'
    )
    return run(full, check=False)


def install_packages():
    print("\n[安装 Python 依赖] ...")
    packages = " ".join(REQUIRED_PACKAGES)
    cmd = (
        f'ssh -o StrictHostKeyChecking=no -i "{KEY_PATH}" '
        f'{SSH_USER}@{EC2_IP} "'
        f'pip install {packages} --quiet 2>&1 | tail -3'
        f'"'
    )
    run(cmd)


def upload_code():
    print("\n[上传代码到 EC2] ...")

    # 上传 week 1 NYC 下载脚本
    scp(
        os.path.join(PROJECT_DIR, "week 1", "download_nyc.py"),
        f"{REMOTE_DIR}/download_nyc.py"
    )

    # 上传 week 2 所有脚本
    scripts_dir = os.path.join(PROJECT_DIR, "week 2", "scripts")
    for fname in os.listdir(scripts_dir):
        if fname.endswith(".py"):
            scp(
                os.path.join(scripts_dir, fname),
                f"{REMOTE_DIR}/week2/{fname}"
            )
            print(f"  上传: week2/{fname}")

    print("  代码上传完成!")


def upload_to_s3(local_dir, s3_path):
    print(f"\n[上传 {local_dir} → S3 {s3_path}] ...")
    cmd = (
        f'aws s3 sync "{local_dir}" "s3://{BUCKET}/{S3_path}/" '
        f'--profile {PROFILE} --exclude "*.tif" --exclude "*.tiff" '
        f'--exclude "__pycache__/*"'
    )
    run(cmd)


def download_from_s3(s3_path, local_dir):
    print(f"\n[下载 S3 {s3_path} → {local_dir}] ...")
    cmd = (
        f'aws s3 sync "s3://{BUCKET}/{S3_path}/" "{local_dir}/" '
        f'--profile {PROFILE} --exclude "*.tif" --exclude "*.tiff" '
        f'--exclude "__pycache__/*"'
    )
    run(cmd)


def run_remote(cmd, check=True):
    full = (
        f'ssh -o StrictHostKeyChecking=no -i "{KEY_PATH}" '
        f'{SSH_USER}@{EC2_IP} "{cmd}"'
    )
    r = subprocess.run(full, shell=True)
    if check and r.returncode != 0:
        print(f"WARNING: 命令失败 (exit {r.returncode}): {cmd}")
    return r.returncode == 0


def setup_ec2():
    print("\n[初始化 EC2 环境] ...")
    run_remote("mkdir -p amazon/week2/outputs amazon/raw_nyc amazon/cleaned_nyc amazon/grid_nyc amazon/features_nyc amazon/graph_nyc")
    run_remote(
        f'pip install {" ".join(REQUIRED_PACKAGES)} --quiet 2>&1 | tail -5',
        check=False
    )


def main():
    print("=" * 60)
    print("Week 2 部署脚本")
    print("=" * 60)

    # Step 1: 获取 EC2 IP
    print("\n[Step 1] 获取 EC2 IP ...")
    get_ec2_ip()

    # Step 2: 上传代码
    print("\n[Step 2] 上传代码 ...")
    upload_code()

    # Step 3: 初始化 EC2 环境
    print("\n[Step 3] 初始化 EC2 环境 ...")
    setup_ec2()

    # Step 4: 运行步骤
    print("\n[Step 4] 运行 Week 2 各步骤 ...")
    steps = [
        ("步骤1: 数据清洗",      "cd amazon && python week2/step1_clean.py"),
        ("步骤2: 网格划分",      "cd amazon && python week2/step2_grid.py"),
        ("步骤3: 特征工程",      "cd amazon && python week2/step3_features.py"),
        ("步骤4: 异构图构建",    "cd amazon && python week2/step4_graph.py"),
    ]

    for name, cmd in steps:
        print(f"\n  === {name} ===")
        run_remote(cmd, check=False)

    # Step 5: 结果推 S3
    print("\n[Step 5] 上传结果到 S3 ...")
    upload_to_s3(f"{REMOTE_DIR}/cleaned_nyc",  f"{S3_PREFIX}/cleaned")
    upload_to_s3(f"{REMOTE_DIR}/grid_nyc",      f"{S3_PREFIX}/grid")
    upload_to_s3(f"{REMOTE_DIR}/features_nyc",  f"{S3_PREFIX}/features")
    upload_to_s3(f"{REMOTE_DIR}/graph_nyc",     f"{S3_PREFIX}/graph")

    print("\n" + "=" * 60)
    print("部署完成!")
    print(f"EC2: {SSH_USER}@{EC2_IP}")


if __name__ == "__main__":
    main()
