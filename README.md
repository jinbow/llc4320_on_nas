# llc4320_on_nas
Simple routine to find and read llc4320 data on NASA NAS machines like Pleiades. 

## Overview
`llc4320_on_nas.py` is a Python script designed for processing and analyzing data from the LLC4320 model, which is one of the highest resolution ocean simulations available. The script includes functions for loading, reshaping, and extracting specific data slices from the model output for specifically NAS HPC environment. 

## Features
- Load data from the LLC4320 model output.
- Reshape the loaded data into east and west hemisphere arrays.
- Extract specific data slices based on various parameters like variable name, step, and tile number.

## Installation
To use `llc4320_on_nas.py`, you need to have Python installed on your system. 

### Steps:
1. Clone the repository or download the script:

```bash 
git clone git@github.com:jinbow/llc4320_on_nas.git
```

  or just download `llc4320_on_nas.py` and put it in your working directory or put it somewhere included in $PYTHONPATH. 

### Usage Examples

```python 
import llc4320_on_nas as llc
from numpy import datetime64
sst=llc.load_llc4320_compressed_2d('Theta',k=0, #k=0 refers to the first layer, i.e., surface 
                                time=datetime(2011,10,10,1), 
                                remapping=True, 
                                retile=True)
In [5]: len(sst)
Out[5]: 2

In [6]: sst[0].shape
Out[6]: (12960, 8640)

In [7]: sst[1].shape
Out[7]: (8640, 12960)
```

```python
import llc4320_on_nas as llc 
import matplotlib.pylab as plt 
from datetime import datetime 
import numpy as np 

t0 = datetime(2011,10,10,12)
sst = llc.load_llc4320_compressed_2d('Theta',k=0,time=t0,retile=True)
display(len(sst),sst[0].shape,sst[1].shape)

# combine tiles
# Here is a layout of llc grid: https://podaac.jpl.nasa.gov/Podaac/thumbnails/ECCO_L4_OBP_LLC0090GRID_MONTHLY_V4R4.jpg
# sst[0] includes tiles 1-6, sst[1] include tiles 8-13 (rotated)
# Each tile has 4320x4320 points. sst[0] has 3x2 tiles. sst[1] has 2x3 tiles. 
# The following line provides a global coverage without the Arctic (tile 7)
# The combined array has shape (12960, 17280), the same as the two coordinates xc (lon), yc (lat) below. 
sst = np.c_[sst[0], sst[1].T[:,::-1]] 

display(sst.shape) 

xc = llc.load_llc_grid('XC') #longitude shape (12960, 17280)
yc = llc.load_llc_grid('YC') #latitude shape (12960, 17280)

display(xc.shape,yc.shape)

plt.pcolor(xc[::96,::96],yc[::96,::96],sst[::96,::96]) #skip values to speed up rendering
```