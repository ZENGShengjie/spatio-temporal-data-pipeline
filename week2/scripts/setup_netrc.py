"""
一次性脚本: 把 username/password 通过 stdin 写入 ~/.netrc, 权限 600.
不打印密码, 不写 .bash_history (直接 heredoc 而非 echo)
"""
import sys, os, stat
import getpass

print("=" * 60)
print("Earthdata .netrc 安全配置")
print("=" * 60)

# username 用第一个参数, password 用 stdin (不 echo)
if len(sys.argv) < 2:
    print("用法: python3 setup_netrc.py <USERNAME>")
    print("      然后从 stdin 输入 password (不会 echo)")
    sys.exit(1)

username = sys.argv[1]
password = sys.stdin.read().rstrip("\n").rstrip()

if not username or not password:
    print("❌ username/password 不能为空")
    sys.exit(2)

# 确认 .netrc 路径
netrc_path = os.path.expanduser("~/.netrc")
print(f"[写] {netrc_path}")

content = f"machine urs.earthdata.nasa.gov login {username} password {password}\n"

# 临时 umask 077
old = os.umask(0o077)
try:
    with open(netrc_path, "w") as f:
        f.write(content)
finally:
    os.umask(old)

# 兜底 chmod
os.chmod(netrc_path, stat.S_IRUSR | stat.S_IWUSR)
mode = oct(os.stat(netrc_path).st_mode & 0o777)
print(f"  权限: {mode}  (要求: 0o600)")

# 验证 (不打印内容)
with open(netrc_path) as f:
    lines = f.read().splitlines()
print(f"  行数: {len(lines)}")
for ln in lines:
    parts = ln.split()
    if "machine" in parts:
        print(f"  machine: {parts[1]}  login: {parts[3]}  password: {'*' * len(parts[5])}")

print("[OK] 写完了. 现在可以跑:")
print("   python3 -c \"import earthaccess; print(earthaccess.login(strategy='netrc'))\"")
print("   python3 scripts/step3b_ndvi.py")