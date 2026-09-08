import xarray as xr

ds = xr.open_dataset("Processed_Data_ndvi.nc")

print(ds)