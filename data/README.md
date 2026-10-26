# Data

Both scripts read the SP12RTS tomography model (Koelemeijer, Ritsema,
Deuss & van Heijst 2016, *GJI* 204, 1024) as two netCDF files:

```
data/SP12RTS_dvs.nc    dlnVs, percent relative to PREM
data/SP12RTS_dvp.nc    dlnVp, percent relative to PREM
```

Each holds a variable `v` on `(depth, latitude, longitude)`, with a 1° grid
and 131 depth levels from 25 to 2891 km.

The model is distributed through the IRIS / EarthScope Earth Model
Collaboration (EMC). Download the Vs and Vp files and save them here under
the names above. The files are not tracked in git.

`pinn_degeneracy.py` finds them here automatically. Anywhere else, pass
`--vs` and `--vp`.
