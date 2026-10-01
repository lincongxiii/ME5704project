# Thin-plate spline pressure reconstruction

`tps.py` is the full TPS alternative model for Parts (a)--(c).  It reconstructs the five-sensor pressure field, evaluates the S2 leave-one-out interpolation error, finds the zero contour and constrained extrema in the sensor convex hull, and runs the 2%/5%/10% noise and lambda-sensitivity studies.

Run from the repository root:

```powershell
py -3.12 -m pip install -r tps/requirements.txt
py -3.12 tps/tps.py
```

The script shows the figures interactively.  Its diagnostic and table output can be retained as numerical evidence for the report.
