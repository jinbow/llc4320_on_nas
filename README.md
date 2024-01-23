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

  git clone git@github.com:jinbow/llc4320_on_pleiades.git

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



