from scipy import constants as const
import os
from subprocess import check_output
import numpy as np
import datetime as DT

# NOTE: Directories are at bottom because structure different
#       for Unix vs. Windows

# ******** SET THE NAME OF THE PROJECT & FLIGHT DATE ********
proj_name = 'PELIcoe_19'
flt_date = '20191023'
search_str = '*dataup*.data'
EM_file_tag = 'datadown' # As of 12/17/19, good EMs only in "down" files. Set to 'datadown'

# ******** SET THE TIME RANGE BOUNDARIES FOR DATA PROCESSING ********
process_start = DT.datetime(2019,10,23,16,17,31) #yr,mon,dy,hr,min,sec
process_end   = DT.datetime(2020,12,7,21,50,0)

# ******** DEFINE CONSTANTS & DATA PARAMS ********
# Speed of light in meters per second
c = const.c
pi =const.pi
# Unix Time epoch
UnixT_epoch = DT.datetime(1970,1,1)
# The number of leap seconds since 1980-01-06 00:00:00
leapsecs = 16
# Each MCS data file should contain 5 minutes of data
file_len_secs = 5 * 60
# The laser rep rate in Hertz
rep_rate = 5000.0
# Start and end altitudes of solar background region (meters)
bg_st_alt = -500.0
bg_ed_alt = -1500.0
# The bin resolution in the fixed frame (m)
vrZ_ff = 30.0
# List containing top and bottom altitudes (m) of the fixed frame
ff_bot_alt,ff_top_alt = [-15e3,22.005e3]
# The number of wavelengths
nwl = 2
# Which channels #'s are which wavelengths? (0=355,1=532,2=1064) OR (0=355,1=1064)
wl_map = [1, 1, 0, 0]
# Create a list where index # of WL corresponds to string name of WL
wl_str = ['355','1064']
# Minimum GPS week value. All < than are filtered out.
minweek = 1000
# Housekeeping record size in bytes
hk_rec_size = 269
# GPS data rate in GPS files (records per second). Please make a float!
gps_hz = 2.0
# An estimate of the number of records in 1 gps file
est_gps_recs_1file = 5*60*10 + 30
# A maximum # of IWG1 records. Data are 1 Hz and flight can be 8 hours.
est_IWG1_recs = 30000
# An estimate of the number of nav records in 1 file
est_nav_recs_1file = 5*60*10 + 30
# MCS data records per second. Please make it a float!
MCS_hz = 10.0
# IWG1 records per second. Please make it a float!
IWG1_hz = 1.0
# nav (refering to file type) records per second. (Please make it a float!) 
nav_hz = 1.0
# Set the Polarization Gain ratios for the wavelenghts [532nm, 1064nm]
PGain = [0.00,0.00]
# Set this to the maximum possible count rate. Don't set > recs in DTT file!
max_counts = 16000
# Dead time tables [list]. List in channel order OR ELSE!!!
DTT_files = ['dttable_camal_chan1_27238-022318.xdr',
    'dttable_camal_chan2_27243-022318.xdr', 'dttable_camal_chan3_27239-022318.xdr',
    'dttable_camal_chan4_27242-022318.xdr']
# Saturation values, per bin per 500 shots. List in detector order.
saturation_values = [5000.0, 5000.0, 5000.0, 5000.0]    
# The overlap file to use
overlap_file = 'OLOnes_Roscoe.xdr' #'olaptable_cpl-ccviceax_comb_iceland12.xdr'      
# The number of seconds needed to convert from the instrument's Unix time
# to UTC. Typically either 5 hours (cold season) or 4 hours (warm season).
secs_btwn_instr_UnixT_and_UTC = 0
# Offset that might be required if Unix Time between nav files and MSC
# files doesn't provide an exact match to link to other clocks (seconds).
nudge = 1.0
# Set this to 'quick' to just grab the hard-coded time offsets below
offset_choice = 'quick'
def_time_offset_UnixT = DT.timedelta(seconds=943.993)
# Roll and pitch offsets for GPS (degrees). Subtract these from read-in vals.
gps_roll_offset = 0.033
gps_pitch_offset = 0.088


# ******** CONFIGURATION PARAMETERS ********
# Averaging method - 'by_scan' or 'by_records_only'
avg_method = 'by_records_only'
# Specify the number of seconds to average
# If <= 0 & 'by_records_only,' no averaging. Set to impossibly high value
# and set avg_method = 'by_scan' to average entire scan.
secs2avg = 5.0
# Minimum number of raw profiles than can be used in an average profile
min_avg_profs = 10
# NOTE: margin variables will apply to 'by_records_only'
# If averaging 'by_scan,' specify # of records to ignore on front of angle transition
front_recs_margin = 0
# If averaging 'by_scan,' specify # of records to ignore on back of angle transition
back_recs_margin = 0
# Min # of consecutive records the scan angle needs to hold (within tolerance)
# Set to zero to ignore, i.e. not filter out on basis of scan angle
min_scan_len = 3
# default scale of color bar
CBar_max = 50.0
CBar_max_NRB = 5e13
CBar_min = 0.0
# The order of magnitude of the alt scale (help make round ticks)
scale_alt_OofM = 1e3 #order of mag.
# Default vertical range of bins to be plotted in curtain
minbin=1000
maxbin=0
# curtain plot width and height (inches)
figW = 18
figL = 10
# profile plot width and height (inches)
pp_figW = 6.5
pp_figL = 7
# "Up" or "Down?" Which direction is lidar pointed?
pointing_dir = "Down"
# default axes limits for profile plot [xmin,xmax]/[ymin,ymax]
pp_xax_bounds_raw_counts = [0,100]
pp_xax_bounds_bgsub_counts = [-15,45]
pp_xax_bounds_NRB = [-7e12,5e13]
pp_yax_bounds_bins = [1000,0]
pp_yax_bounds_alt = [-10e3,20e3]
# Y-axis bounds of the energy monitor plot
EMp_yax_bounds = [0,150]
# Channel # entries by user cannot be outside this range
min_chan = 1
max_chan = 5
# Cap on the number of profiles that can be used to generate curtain.
hori_cap = 20000
# The padding around the curtain plot in inches
CPpad = 0.1
# Set default housekeeping axis values
hk_y_max = 100
hk_y_min = -100
# For actual data processing (not the GUI), nav data source
Nav_source = 'nav' #'nav' 'gps' or 'iwg1'
# IWG1 data file
IWG1_file = "IWG1.08Dec2017-0031.txt"
# Don't process any data when below this alt (m). Doesn't apply to GUI.
alt_cutoff = 10000
# Don't process any profiles where off-nadir angle exceeds this many radians.
ONA_cutoff = 20.0 * (pi/180.0)
# The attention bar
attention_bar = '\n******************************\n'
# The number of standard deviations of EM values to keep. Can set negative to keep all.
Estds = 4

if (os.name != 'nt'): # IF UNIX-BASED MACHINE, DEFINE DIRECTORIES HERE

    # ******** DEFINE ALL DIRECTORIES ********
    # Set the directory of the raw data
    raw_dir = '/cpl3/CAMAL/Data/'+proj_name+'/'+flt_date+'/raw/'
    # Set the directory of the L0 data
    L0_dir = '/cpl3/CAMAL/Data/'+proj_name+'/'+flt_date+'/L0/'
    # Set the directory of the L1 data
    L1_dir = '/cpl3/CAMAL/Data/'+proj_name+'/'+flt_date+'/L1/'
    # Set the directory of the L2 data
    L2_dir = '/cpl3/CAMAL/Data/'+proj_name+'/'+flt_date+'/L2/'
    # Set the directory that contains configuration files
    config_dir = '/cpl3/CAMAL/Config/'
    # The source directory
    source_dir = '/cpl3/CAMAL/Source/L1A/'
    # Directory to put output
    out_dir = '/cpl3/CAMAL/Analysis/'+proj_name+'/'+flt_date+'/'
    # Directory and name of library containing C codes
    clib_path = source_dir + 'C_lib_unix/'
    clib = 'CAMAL_C_lib_v0.so'
    dtt_dir = config_dir + 'dttables/'
    # Directory containing overlap files
    olap_dir = '/home/selmer/compute_OL/' 
    # Directory and name of the DEM
    DEM_dir = config_dir + 'DEM/'
    DEM_name = 'JPL-CloudSat.dem'
    # Directory and name of C++ library with functions to read in DEM
    DEM_lib_path = DEM_dir + 'JPL_DEM_CPP_SO/'
    DEM_Cpp_name = 'JPL_DEM_CPP_FUNCTIONS.so'
    

else:                 # IF WINDOWS-BASED MACHINE, DEFINE DIRECTORIES HERE

	    # ******** DEFINE ALL DIRECTORIES ********
    # Set the directory of the raw data
    #raw_dir = 'F:\\CAMAL\\from_datakey\\'+flt_date+'\\'
    raw_dir = 'C:\\Users\\pselmer\\Documents\\Roscoe\\data\\'+proj_name+'\\'+flt_date+'\\'
    # Set the directory of the L1 data
    L1_dir = 'C:\\Users\\pselmer\\Documents\\Roscoe\\data\\'+proj_name+'\\L1\\'
    # Set the directory of the L2 data
    L2_dir = 'C:\\Users\\pselmer\\Documents\\Roscoe\\data\\'+proj_name+'\\L2\\'
    # Set the directory that contains configuration files
    config_dir = 'C:\\Users\\pselmer\\Documents\\CPL_stuff\\Config\\'
    # Directory to put output
    out_dir = 'C:\\Users\\pselmer\\Documents\\Roscoe\\data\\'+proj_name+'\\analysis\\'
    # Directory and name of library containing C codes
    clib_path = 'C:\\Users\\pselmer\\Documents\\CPL_stuff\\Source\\L1A\\C_lib_Win64\\CAMAL_C_lib_Win64\\x64\\Release\\'
    clib = 'CAMAL_C_lib_Win64.dll'
    # Directory of dead time tables
    dtt_dir = config_dir + 'CAMAL_dttables\\'
    # Directory containing overlap files
    olap_dir = config_dir
    # Directory and name of the DEM
    DEM_dir = config_dir + '\\DEM\\'
    DEM_name = 'JPL-CloudSat.dem'
    # Directory and name of C++ library with functions to read in DEM
    DEM_lib_path = 'C:\\Users\\pselmer\\Documents\\CPL_stuff\\source\\DEM_reader_code\\JPL_DEM_CPP_DLL\\x64\\Release\\'
    DEM_Cpp_name = 'JPL_DEM_CPP_DLL.dll'

