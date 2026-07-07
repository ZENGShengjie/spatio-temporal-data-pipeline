import earthaccess
auth = earthaccess.login(strategy="netrc")
print("login:", "OK" if auth else "FAIL")
import h5py, pyhdf, rasterio
print("deps: OK")
print("h5py:", h5py.__version__)
print("rasterio:", rasterio.__version__)
print("pyhdf:", pyhdf.__version__)
print("earthaccess:", earthaccess.__version__)