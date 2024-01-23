# -*- coding: utf-8 -*-
"""
Read llc4320 data on Pleiades

"""
import pylab as plt
import numpy as np
import sys

def missing_PhiBot():
    """the following files has zeros in PhiBot"""
    a=[]

    b=open('/home1/jwang23/local/popy/popy/phibot_zeros.asc').readlines()
    for c in b:
        a.append(int(c.split('.')[1]))

    return a

def load_llc4320_compressed_2d(varn, k=0, time=None, t_index=None, step=None, remapping=True, retile=False):
    """
    Load a compressed data layer from a specified file path and return it as a numpy array.

    This function reads a layer of data from a compressed file based on the provided parameters. 
    It supports reading different variables at specified levels, times, or steps. The data can 
    optionally be remapped to a 2D lat-lon grid and retiled.

    Parameters:
    -----------
    varn : str
        The name of the variable to load (e.g., 'PhiBot', 'U', 'V', 'Theta', 'Salt', 'Eta' etc).
    k : int, optional
        The vertical level to load from the data file. Default is 0.
    time : datetime.datetime, optional
        The specific time to load the data. Used to calculate the time step for reading the file. Valid dates are between 09/13/2011 and 11/15/2012
    t_index : int, optional
        Time index to specify which time record to load. This is used by the model in output filenames.
    step : int, optional
        Direct specification of the time step for file reading. (1-13566?)
    remapping : bool, optional
        If True (default), the function returns a 2D array mapped to a lat-lon grid with shape 4320*3 by 4320*4. 
        If False, it returns the compressed unstructured 1D array.
    retile : bool, optional
        If True, the returned array is retiled. Default is False.

    Returns:
    --------
    dout : numpy.ndarray
        The loaded data as a 1D or 2D numpy array, depending on the 'remapping' parameter.

    Raises:
    -------
    ValueError
        If none of 'time', 't_index', or 'step' parameters are specified.

    Notes:
    ------
    The function computes the time step from the 'time', 't_index', or 'step' parameter and 
    uses it to determine the correct file path for reading. Special handling is implemented 
    for missing 'PhiBot' data. The data is read using memory mapping for efficiency.

    Examples:
    ---------
    # Example of loading data with a specific variable name and time
    data = load_compressed_2d('PhiBot', time=datetime.datetime(2011, 11, 1))

    # Example of loading data with a variable name, level, and time index
    data = load_compressed_2d('U', k=3, t_index=10)
    """

    import datetime
    import numpy as np

    missing=missing_PhiBot()

    if time!=None:
        t0=datetime.datetime(2011,9,13)
        dt=(time-t0).total_seconds()/3600. * 144 + 10368 #get the time step for reading the file
    elif t_index!=None:
        dt=10368+t_index*144
    elif step!=None:
        dt=step
    else:
        raise ValueError('specify time record either through datetime or t_index')

    if varn=='PhiBot' and dt in missing:
        fn='/nobackup/jwang23/llc4320_stripe/missing.PhiBot.compressed/PhiBot.%010i.data'%dt
        print("load data from /nobackup/jwang23/llc4320_stripe/missing.PhiBot.compressed for zero PhiBot")
    else:
        fn='/u/dmenemen/llc_4320/compressed/%010i/%s.%010i.data.shrunk'%(dt,varn,dt)

    """
    read a layer from compressed file.
    First find the offset, and the shape.
    Then map them to 13*4320**2

    """
    def get_offset(k):
        pth='/nobackup/jwang23/llc4320_stripe/grid/'
        if varn=='U':
            fn='data_index.hFacW.txt'
        elif varn=='V':
            fn='data_index.hFacS.txt'
        else:
            fn='data_index.hFacC.txt'
        d=np.loadtxt(f'{pth}{fn}')
        tt,offset,shape=d[k,:]
        return int(offset),int(shape)

    offset,shape=get_offset(k)
    dout=np.memmap(fn,dtype='>f4',offset=offset*4,shape=shape,mode='r')

    if remapping:
        loc='C'
        if varn=='U':loc='W'
        if varn=='V':loc='S'
        msk =load_hfac(k,loc)
        msk[msk>0]=dout
        del dout
        dout=msk
    if retile:
        dout=mds2d(dout)

    return dout

def load_hfac(k,loc='C'):
    import numpy as np
    fn0='/u/dmenemen/llc_4320/compressed/hFac%s.data'%loc
    nn=4320**2
    offset=(k*nn*13)*4
    shape=(nn*13)
    df=np.memmap(fn0,dtype='>f4',offset=offset,mode='r',shape=shape)
    dd=df.copy()
    del df
    return dd
    

class paths:
    def __init__(self,nx,tt=0,isdaily=False,run_name=''):
        if nx==4320:
            self.data_dir0='/u/dmenemen/llc_4320/MITgcm/run/'
            self.data_dir1='/u/dmenemen/llc_4320/MITgcm/run_485568/'
            if run_name!='':
                self.data_dir='/u/dmenemen/llc_4320/MITgcm/%s/'%run_name
            self.grid_dir='/u/dmenemen/llc_4320/grid/'
            self.output_dir='/u/jwang23/llc4320_stripe/'
            self.tstart=10368
            self.dt=144
            self.nt=9416
            self.nx=nx
        elif nx==2160:
            if isdaily:
                self.data_dir='/u/dmenemen/llc_2160/regions/global/'
            else:
                if tt>=(1198080-92160)/80:
                    self.data_dir='/u/dmenemen/llc_2160/MITgcm/run/'
                else:
                    self.data_dir='/u/dmenemen/llc_2160/MITgcm/run_day49_624/'
            self.grid_dir='/u/dmenemen/llc_2160/grid/'
            self.output_dir='/nobackup/jwang23/llc2160_striped/'
            self.tstart=92160
            self.dt=80
            self.nt_daily=750
            self.nt=(1586400-92160)/self.dt +1
        self.nx=nx
        self.tt=tt

    def get_fn(self,varn,tt=0,folder=0):
        if folder==0:
            self.data_dir=self.data_dir0
        else:
            self.data_dir=self.data_dir1

        if tt>self.nt-1:
            sys.exit('error: tt is larger than max')
        else:
            if self.nx==4320:
                if varn in ['Theta','Salt','U','V','W','PhiBot','Eta']:
                    fn=self.data_dir+'%s.%010i.data'%(varn,self.tstart+self.dt*tt)
                else:
                    fn=self.grid_dir+'%s.data'%(varn)
            else:
                if varn in ['Theta','Salt','U','V','W','PhiBot','Eta']:
                    fn=self.data_dir+'%s.%010i.data'%(varn,self.tstart+self.dt*tt)
                else:
                    fn=self.grid_dir+'%s.data'%(varn)
        return fn
def load_llc_grid(varn):
    """
    Load grid data for a specified variable from an HDF5 (.h5) file.

    This function is designed to load grid data related to the LLC 4320 model. It reads the specified variable 
    from a pre-defined HDF5 file containing grid data. The file path is constructed using the 'paths' function 
    from the 'popy' package. The function returns the grid data for the specified variable.

    Parameters:
    -----------
    varn : str
        The name of the variable for which grid data is to be loaded. This could be any grid variable name
        present in the HDF5 file.

    Returns:
    --------
    grid_data : Various (depending on the variable)
        The grid data associated with the specified variable. The type and structure of the returned data
        depend on the variable.

    Raises:
    -------
    FileNotFoundError
        If the HDF5 file does not exist at the specified path.

    IOError
        If the variable name does not exist in the HDF5 file.

    Examples:
    ---------
    # Load grid data for the 'Depth' variable
    depth_grid = load_llc_grid('Depth')

    # Load grid data for the 'Temperature' variable
    temp_grid = load_llc_grid('Temperature')

    Notes:
    ------
    The function assumes the existence of an HDF5 file named 'grids.h5' in the output directory
    specified by the 'paths' function of the 'popy' package. The exact location and structure
    of this file should be known and consistent for the function to work as expected.
    """
    import popy
    d = paths(4320)
    fn = d.output_dir + 'grids.h5'
    return popy.io.loadh5(fn, varn)


def load_c1440_llc2160_mitgcm(varn, step, pth='', 
                              tile_number=11, 
                              p=[0,2160,0,2160], 
                              nx=2160, 
                              k=0, 
                              return_grid=True, 
                              offset=0, 
                              shape=None):
    """
    Load data from MITgcm model output for a specified variable, step, and tile number.

    This function reads data from the MITgcm (Massachusetts Institute of Technology General Circulation Model)
    output files. It supports loading of specific variables at a given model step and from a specified tile of the model grid. 
    The function can optionally return the grid coordinates associated with the data.

    Parameters:
    -----------
    varn : str
        The name of the variable to be loaded (e.g., temperature, salinity).
    step : int
        Model step number to specify the time point of the data.
    pth : str, optional
        The path to the directory containing the MITgcm output files. Default is a predefined path.
    tile_number : int or str, optional
        The number of the tile from which to load the data. Can be a single tile number (e.g., 11)
        or a range specified as a string (e.g., '1-7'). Default is 11.
    p : list of int, optional
        List specifying the partition of the grid to load. Format is [start_x, end_x, start_y, end_y].
        Default is [0,2160,0,2160].
    nx : int, optional
        The number of grid points along one axis. Default is 2160.
    k : int, optional
        The depth level to load. Default is 0 (surface).
    return_grid : bool, optional
        If True, the function also returns the grid coordinates (longitude and latitude). Default is True.
    offset : int, optional
        Additional offset to apply when reading the file. Default is 0.
    shape : tuple, optional
        The shape of the data to be loaded. If not specified, it is inferred from the tile number.

    Returns:
    --------
    (dd, xx, yy) : tuple
        dd: numpy.ndarray
            The data array for the specified variable.
        xx: numpy.ndarray
            The longitude coordinates of the data points, returned if 'return_grid' is True.
        yy: numpy.ndarray
            The latitude coordinates of the data points, returned if 'return_grid' is True.

    Raises:
    -------
    FileNotFoundError
        If the specified file does not exist at the given path.
    IOError
        If there is an error in reading the data from the file.

    Examples:
    ---------
    # Load temperature data from tile 11 at step 100 with default grid path
    temperature, lon, lat = load_c1440_llc2160_mitgcm('temperature', 100, tile_number=11)

    # Load salinity data from the entire range of tiles 1-7 at step 50, with a custom path
    salinity, lon, lat = load_c1440_llc2160_mitgcm('salinity', 50, pth='/custom/path/', tile_number='1-7')

    Notes:
    ------
    The function is specifically designed for the output of the MITgcm model with the c1440_llc2160 configuration. 
    It assumes a specific structure and naming convention of the output files.
    """

    pth="/nobackupp11/dmenemen/DYAMOND/c1440_llc2160/mit_output/"

    fn=pth+'%s/%s.%010i.data'%(varn,varn,step*80)

    fn_xc=pth+'grid/XC.data'
    fn_yc=pth+'grid/YC.data'

    nxx=nx**2

    if tile_number in range(8,14):
        if tile_number<11:
            offset=k*nx*nx*13+7*nx*nx
        else:
            offset=k*nx*nx*13+10*nx*nx
        try:
            d=np.memmap(fn,'>f4',mode='r',offset=offset*4,shape=(nx,nx*3))
        except:
            d=np.memmap(fn,'>f4',mode='r',offset=offset*4,shape=(nx,nx*3))

        ii=np.mod(tile_number-8,3)
        i0,i1=ii+p[0],ii+p[1]
        j0,j1=p[2:]
        dd=d[j0:j1,i0:i1]
        del d
        xx,yy=0,0
        if return_grid==True:
            x=np.memmap(fn_xc,'>f4',mode='r',offset=offset*4,shape=(nx,nx*3))
            y=np.memmap(fn_yc,'>f4',mode='r',offset=offset*4,shape=(nx,nx*3))
            yy=y[j0:j1,i0:i1]
            xx=x[j0:j1,i0:i1]
    elif tile_number in range(1,7):
        offset=k*nx*nx*13+(tile_number-1)*nxx
        try:
            d=np.memmap(fn,'>f4',mode='r',offset=offset*4,shape=(nx,nx))
        except:
            d=np.memmap(fn,'>f4',mode='r',offset=offset*4,shape=(nx,nx))

        i0,i1=p[0],p[1]
        j0,j1=p[2:]
        dd=d[j0:j1,i0:i1]
        del d
        xx,yy=0,0
        if return_grid==True:
            x=np.memmap(fn_xc,'>f4',mode='r',offset=offset*4,shape=(nx,nx))
            y=np.memmap(fn_yc,'>f4',mode='r',offset=offset*4,shape=(nx,nx))
            yy=y[j0:j1,i0:i1]
            xx=x[j0:j1,i0:i1]
    elif tile_number == '1-7':
        offset=k*nx*nx*13
        try:
            d=np.memmap(fn,'>f4',mode='r',offset=offset*4,shape=(nx*7,nx))
        except:
            d=np.memmap(fn,'>f4',mode='r',offset=offset*4,shape=(nx*7,nx))

        xx,yy=0,0
        if return_grid==True:
            x=np.memmap(fn_xc,'>f4',mode='r',offset=offset*4,shape=(nx,nx))
            y=np.memmap(fn_yc,'>f4',mode='r',offset=offset*4,shape=(nx,nx))
            yy=y[j0:j1,i0:i1]
            xx=x[j0:j1,i0:i1]
        return d,xx,yy

    else:
        offset0=k*nx*nx*13+offset
        try:
            dd=np.memmap(pth.get_fn(varn,step,0),'>f4',mode='r',offset=offset0*4)
        except:
            dd=np.memmap(pth.get_fn(varn,step,1),'>f4',mode='r',offset=offset0*4)

        xx=np.memmap(pth.get_fn('XC'),'>f4',mode='r',offset=offset*4)
        yy=np.memmap(pth.get_fn('YC'),'>f4',mode='r',offset=offset*4)
        dd=mds2d(dd)
        xx=mds2d(xx)
        yy=mds2d(yy)


    return dd,xx,yy

def re_tile(d, n=4320, shift_U=False):
    """
    Re-organize and reshape a 1D array into a 2D array representing a re-tiled grid.

    This function takes a 1D numpy array representing data on a grid and re-tiles it into a 2D grid format. 
    The function is specifically designed to handle llc arrays from 
    MITgcm. 

    Parameters:
    -----------
    d : numpy.ndarray
        A 1D numpy array containing the data to be re-tiled.
    n : int, optional
        The number of grid points along one axis of the square grid. Default is 4320 for the llc4320 simulation
    shift_U : bool, optional
        If True, performs a shift operation on the data. This is typically used for shifting the U grid 
        for C-grid.

    Returns:
    --------
    xc : numpy.ndarray
        The 2D numpy array representing the re-tiled grid. The array dimensions are determined by the
        input data and the specified grid size.

    Examples:
    ---------
    # Example of re-tiling data without shift
    reshaped_data = re_tile(data_array)

    # Example of re-tiling data with shift for U grid
    reshaped_data_U = re_tile(data_array, shift_U=True)

    """

    import numpy as np
    de=d[:n**2*6].reshape(-1,n)
    de=np.c_[de[:n*3,:],de[n*3:,:]]
    dw=d[n**2*7:].reshape(n*2,-1)
    dw=np.r_[dw[:n,:],dw[n:,:]]
    if shift_U: #shift DYG, U grid on tiles 8-13
        dw=np.roll(dw,-1,axis=1)

    dw=dw.T[::-1,:]
    xc=np.c_[de,dw]
    return xc


def mds2d(dd, nx=4320):
    """
    Reshape an LLC grid data array into separate east and west hemisphere arrays.

    This function takes an array representing data on the LLC grid and separates it into two arrays, 
    one for the eastern hemisphere (tiles 1-6) and one for the western hemisphere (tiles 8-13). Tile 7 is for Arctic but not included here.
    The function does not perform any rotation on the data, meaning that the u and v components 
    (typically representing eastward and northward velocity components in oceanographic data) remain mixed for the llc grid layout

    Parameters:
    -----------
    dd : numpy.ndarray or list of numpy.ndarray
        The input data array(s) to be reshaped. Each array should be of size 13*nx**2, 
        where nx is the size of the LLC grid. The input can also be a list of such arrays.
    nx : int, optional
        The size of one side of the LLC grid. Default is 4320, suitable for the llc4320 model.

    Returns:
    --------
    (deast, dwest) : tuple
        deast : numpy.ndarray or list of numpy.ndarray
            The eastern hemisphere data array(s), each with dimensions (4320x3, 4320x2).
        dwest : numpy.ndarray or list of numpy.ndarray
            The western hemisphere data array(s), each with dimensions (4320x2, 4320x3),
            non-rotated (x is latitude, y is longitude).

    Examples:
    ---------
    # Reshaping a single data array
    east, west = mds2d(data_array)

    # Reshaping a list of data arrays
    reshaped_data_list = mds2d(list_of_data_arrays)

    Notes:
    ------
    The input data should be structured with specific tiles corresponding to different parts of 
    the global grid. The function assumes that tiles 1-6 represent the eastern hemisphere and 
    tiles 8-13 represent the western hemisphere. Tiles 7 Arctic is not included.
    """
    def rearrange(d):
        deast=np.c_[d[:nx*nx*3].reshape(3*nx,nx),
                    d[nx*nx*3:nx*nx*6].reshape(3*nx,nx)]
        dwest=d[nx*nx*7:].reshape(nx*2,nx*3)
        return deast,dwest

    if type(dd)==type([1]):
        dout=[]
        for d in dd:
            dout.append(rearrange(d))
        return dout
    else:
        return rearrange(dd)