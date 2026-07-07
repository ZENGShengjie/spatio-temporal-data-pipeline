import os, sys

os.environ["EARTHDATA_USERNAME"] = "tony060514"
os.environ["EARTHDATA_PASSWORD"] = "SCzsj060514#"

try:
    import earthaccess
    auth = earthaccess.login(strategy="environment", persist=True)
    print("earthaccess version:", earthaccess.__version__)
    print("登录状态:", auth.authenticated)
except Exception as e:
    print("ERROR:", type(e).__name__, str(e))
