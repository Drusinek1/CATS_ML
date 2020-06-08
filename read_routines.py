# Routines to read in the various types of data files
#
# [10/9/19] **** MAJOR CHANGE ****
# Import of "initializations.py" removed. This will cause headaches for 
# a little while.
#
# [2/25/20] Most of the bumps from initializations import removal should
# be gone. But it's possible an issue will be found at some point.
#
# [3/26/20] **** Minor change ****
# Offset of gps pitch occurs before reversal of pitch orientation

import pdb
import struct as struct
import csv
import datetime as DT
import xdrlib
from functools import partial
import os
from subprocess import check_output

# Libraries I have created...
from immutable_data_structs import *  # data structs that don't change
from mutable_data_structs import *    # data structs that can change
from time_conversions import weeksecondstoutc, delta_datetime_vector


def delete_file_in_Windows(file_name):
    """ Delete a file in Windows OS. """
    
    cmd = 'del ' + file_name
    cmd_feedback = check_output(cmd, shell=True)
    
    
def delete_file_in_Unix(file_name):
    """ Delete a file in Unix-type OS. """
    
    cmd = 'rm ' + file_name
    cmd_feedback = check_output(cmd, shell=True)    


def delete_file(file_name):
    """ Calls appropriate function depending on OS. """
    if os.name != 'nt':                                # Linux/Unix
        delete_file_in_Unix(file_name)
    else:                                                # Windows
        delete_file_in_Windows(file_name)    
    

def create_a_file_list_in_Windows(file_list, search_str, raw_dir):
    """ This code  will create a file list in the current directory
        with the given input name. The other input text string is 
        so code can be reused to search for different systematically
        named types of files.
    """
    
    cmd = 'type nul > ' + file_list
    cmd_feedback = check_output(cmd, shell=True)
    cmd = 'del ' + file_list
    cmd_feedback = check_output(cmd, shell=True)
    cmd = 'dir ' + raw_dir + search_str + ' /B/S/ON > ' + file_list 
    cmd_feedback = check_output(cmd, shell=True)
    
    
def create_a_file_list_in_Unix(file_list, search_str, raw_dir):
    """ This code will create a file list in the current directory
        with the given input name. The other input text string is
        so code can be reused to search for different systematically
        named types of files.
    """

    # [8/22/18]
    # 'touch' command determined unnessessary and removed from script.
    # [9/28/18]
    # Noticed funky issue when trying to run on Bubba. Needed to add "
    # marks to the call to the make_file_list_unix script to fix.
    
    # *** NOTE *** 
    # This code relies on an external C-shell script.
    cmd = 'rm -f ' + file_list  # -f added 8/22/18, making touch unnecessary
    cmd_feedback = check_output(cmd, shell=True)
    cmd = './make_file_list_unix ' + raw_dir + ' "' + search_str + '" > ' + file_list
    cmd_feedback = check_output(cmd, shell=True)   
    
    
def create_a_file_list(file_list, search_str, raw_dir):
    """ Calls appropriate function depending on OS. """
    
    if os.name != 'nt':                                # Linux/Unix
        create_a_file_list_in_Unix(file_list, search_str, raw_dir)
    else:                                                # Windows
        create_a_file_list_in_Windows(file_list, search_str, raw_dir)


def read_in_raw_data(fname):
    """ Routine to read in raw data from a single CAMAL file 
        Takes only one input, the file name. All other constructs
        are provided from imported modules (serve like IDL's 
        common blocks)
    """
    
    # First, get nchans and nbins from the file...
    samp = np.fromfile(fname, MCS_meta_struct, count=1)
    
    # Next, use that info to form your MSC data structure...
    try:
        MCS_struct = define_MSC_structure(samp['nchans'][0], samp['nbins'][0])
    except IndexError:
        print('Size of ingested data: ', samp.shape)
        print('Bad data. Skipping file...')
        return None
    
    # Finally, read in all data from the file at once
    MCS_data = np.fromfile(fname, MCS_struct, count=-1)
    
    return(MCS_data)
  
    
def read_in_housekeeping_data(fname):
    """ Routine to read in housekeeping data from a single file.
        Takes only one input, the file name. All other constructs
        are provided from imported modules.
    """
    
    return np.fromfile(fname, hk_struct, count=-1)
  
    
def filter_out_bad_recs(MCS_struct, minweek):
    """ Currently, only works on CAMAL data [12/5/17]
        The data objects are CAMAL MCS or hk.
    """
    
    # Unix epoch time: 1970 1 1 0 0 0 (Jan 01, 00:00:00 UTC)
    
    # In the following block, it's assumed MCS_struct is 1-D array
    good_rec_mask = MCS_struct['meta']['GpsWeek'] > minweek 
    MCS_struct = MCS_struct[good_rec_mask]
    skip_flag = 0
    if MCS_struct.shape[0] == 0: skip_flag = 1
    
    return skip_flag
    
    
def read_in_gps_data(fname, file_len_secs, gps_hz, gps_pitch_offset, gps_roll_offset):
    """Routine to read in GPS files (CAMAL/ACATS).
    
       [12/15/17]
       Entire code revolves around keying in on bytes 3C and 20 (hex).
       If these bytes are found, after Python sees a "newline," the 
       start of a record is presumed. If Python raises a ValueError
       during the conversion, then that record is simply skipped.
       
       [3/9/18]
       Looking at 2017-12-06 data, it appears for whatever reason, that
       the pitch in these gps files has the opposite sign of what it's
       supposed to be. Putting a -1 factor into this code to correct it.
       
       [3/18/18]
       Pitch & roll offsets added. Set in initializations.py.
    """
    
    # Initialize a numpy structure array to hold GPS data 
    # for a single file.
    
    est_recs = int(file_len_secs * (gps_hz+1))  # add a little buffer
    gps_struct_1file = np.zeros(est_recs, dtype=gps_struct)
    
    f_obj = open(fname, 'rb')
    # Set a minimum # of bytes a single GPS record is allowed to be
    min_len = 132
    
    i = 0
    for line in f_obj:
        line_ba = bytearray(line)
        if len(line_ba) > min_len:
            if line_ba[0] == 60 and line_ba[1] == 32:
                one_rec = line.split()
                try:
                    gps_struct_1file[i]['GpsWeek'] = one_rec[1]
                    gps_struct_1file[i]['Gps_sec'] = one_rec[2]
                    gps_struct_1file[i]['lat'] = one_rec[3]
                    gps_struct_1file[i]['lon'] = one_rec[4]
                    gps_struct_1file[i]['alt'] = one_rec[5]
                    gps_struct_1file[i]['north'] = one_rec[6]
                    gps_struct_1file[i]['east'] = one_rec[7]
                    gps_struct_1file[i]['vert'] = one_rec[8]
                    gps_struct_1file[i]['pitch'] = one_rec[9] - gps_pitch_offset
                    gps_struct_1file[i]['pitch'] = gps_struct_1file[i]['pitch']*(-1.0)
                    gps_struct_1file[i]['roll'] = float(one_rec[10]) - gps_roll_offset
                    gps_struct_1file[i]['yaw'] = one_rec[11]
                    # if gps_struct_1file[i]['GpsWeek'] > 0: pdb.set_trace()
                    i += 1
                except ValueError:
                    print('------------------------------------------------')
                    print('\n\nFound the "key" in GPS file, but unable')
                    print('to convert value(s). Byte object print-out')
                    print('follows-> hex format, bytes converted to Unicode')
                    print('characters as applicable. This record will be')
                    print('skipped.\n')
                    print(line)
                    print('------------------------------------------------')
                except IndexError:
                    print('Index error in read_in_gps_data. Stopping here...')
                    pdb.set_trace()
    
    # Trim off the fat...
    
    gps_struct_1file = gps_struct_1file[0:i]
    
    if i <= 0:
        print("**** NO VALID DATA IN THIS GPS FILE ****")
        print("******** RETURNING None ********")
        return None
    
    f_obj.close()
    return gps_struct_1file
    
    
def read_in_IWG1_data(fname, est_recs):
    """ Routine to read in the IWG1 data. """
    
    IWG1_data_1file = np.zeros(est_recs, dtype=IWG1_struct)
    
    with open(fname, 'r') as csvfile:
        
        csvreader = csv.reader(csvfile)
        r = 0
        datetimeformat = "%Y-%m-%dT%H:%M:%S.%f"
        
        for row in csvreader:
            
            # Put invalids (-999.9) in blank fields
            c = 0
            for col in row:
                if col == '':
                    row[c] = -999.9
                c += 1
            
            # This invalids block above should have taken care of all the
            # invalids, but just to be extra safe (say, user passes a
            # non-IWG1 file to this routine), this try/except block will
            # safety stop code in debug mode.
            try:
                IWG1_data_1file['UTC'][r] = DT.datetime.strptime(row[1], datetimeformat)
                IWG1_data_1file['lat'][r] = row[2]
                IWG1_data_1file['lon'][r] = row[3]
                IWG1_data_1file['GPS_alt_msl'][r] = row[4]
                IWG1_data_1file['GPS_alt'][r] = row[5] 
                IWG1_data_1file['press_alt'][r] = row[6] 
                IWG1_data_1file['rad_alt'][r] = row[7] 
                IWG1_data_1file['ground_spd'][r] = row[8]
                IWG1_data_1file['true_airspd'][r] = row[9] 
                IWG1_data_1file['ind_airspd'][r] = row[10]
                IWG1_data_1file['mach_num'][r] = row[11] 
                IWG1_data_1file['vert_spd'][r] = row[12]
                IWG1_data_1file['heading'][r] = row[13]    # use as yaw
                IWG1_data_1file['track'][r] = row[14]
                IWG1_data_1file['drift'][r] = row[15] 
                IWG1_data_1file['pitch'][r] = row[16]
                IWG1_data_1file['roll'][r] = row[17]
                IWG1_data_1file['slip'][r] = row[18]
                IWG1_data_1file['attack'][r] = row[19]
                IWG1_data_1file['S_air_temp'][r] = row[20]
                IWG1_data_1file['dewp_temp'][r] = row[21] 
                IWG1_data_1file['T_air_temp'][r] = row[22]
                IWG1_data_1file['static_p'][r] = row[23] 
                IWG1_data_1file['dynmc_p'][r] = row[24]
                IWG1_data_1file['cabin_p'][r] = row[25] 
                IWG1_data_1file['wind_spd'][r] = row[26]
                IWG1_data_1file['wind_dir'][r] = row[27]
                IWG1_data_1file['vert_wind_spd'][r] = row[28]
                IWG1_data_1file['sol_zen'][r] = row[29] 
                IWG1_data_1file['air_sun_elev'][r] = row[30]
                IWG1_data_1file['so_azi'][r] = row[31]
                IWG1_data_1file['air_sun_azi'][r] = row[32]
                r += 1
            except ValueError:
                print('Something is really wrong with this IWG1 file.')
                print('Stopping in read_in_IWG1_data routine.')
                pdb.set_trace()
                
    IWG1_data_1file = IWG1_data_1file[0:r]
    return IWG1_data_1file


def decode_er2_like_wb57_nav_string(Nav_string, signs):
    """ The WB57 Nav format is different from the ER2-style format.
        Andrew Kupchock wrote code that rewrites the WB57 Nav into the
        CLS files so that's it's compatible with the old, ER2-style
        CPL processing (which decoded by byte position). He did this by 
        just leaving whitespace in the string for variables that didn't 
        match up. Unfortunately, this new CPL processing splits
        the Nav string on whitespace. Therefore, this function will handle
        these particular files (started with the REThinC 2017 campaign), since
        the existing ER2 Nav decode function cannot correctly decode these.
    """

    # INPUTS:
    #
    # Nav_string -> The Nav string (not split) in the CLS files
    # signs -> A dictionary matching various symbols to arthimetic sign
    #
    # OUTPUTS:
    #
    # CLS_decoded_nav_data -> 1-element array of dtype CLS_decoded_nav_struct

    CLS_decoded_nav_data = np.zeros(1, dtype=CLS_decoded_nav_struct)

    CLS_decoded_nav_data['UTC_Time'][0] = DT.datetime.strptime(flt_date[5:7]+'-'+Nav_string[4:16], '%y-%j:%H:%M:%S')
    CLS_decoded_nav_data['TrueHeading'][0] = float(Nav_string[37:44])
    CLS_decoded_nav_data['PitchAngle'][0] = float(Nav_string[45:53])
    CLS_decoded_nav_data['RollAngle'][0] = float(Nav_string[54:62])
    CLS_decoded_nav_data['GroundSpeed'][0] = float(Nav_string[63:70])
    CLS_decoded_nav_data['GPS_Altitude'][0] = float(Nav_string[134:141])
    CLS_decoded_nav_data['GPS_Latitude'][0] = float(Nav_string[143:151]) * signs[Nav_string[142]]
    CLS_decoded_nav_data['GPS_Longitude'][0] = float(Nav_string[153:161]) * signs[Nav_string[152]]
    CLS_decoded_nav_data['SunElevation'][0] = float(Nav_string[235:241])
    CLS_decoded_nav_data['SunAzimuth'][0] = float(Nav_string[242:248])

    return CLS_decoded_nav_data
            

def decode_er2_nav_string(Nav_fields, signs, flt_date, bad_cls_nav_time_value):
    """ This function decodes an input ER2-style CLS Nav list of strings into its
        constituent parameters. Function returns a "decoded_nav_struct"
        datatype array of length one.
    """

    # INPUTS:
    #
    # Nav_fields -> List of strings containing Nav parameters to be decoded
    # signs -> A dictionary matching various symbols to arthimetic sign
    #
    # OUTPUTS:
    #
    # CLS_decoded_nav_data -> 1-element array of dtype CLS_decoded_nav_struct

    CLS_decoded_nav_data = np.zeros(1, dtype=CLS_decoded_nav_struct)

    # --> Omitting "Header" of 'Nav' until someone tells me they need it [4/4/18]
    try:
        CLS_decoded_nav_data['UTC_Time'][0] = DT.datetime.strptime(flt_date[5:7]+'-'+Nav_fields[1], '%y-%j:%H:%M:%S')
    except:
        CLS_decoded_nav_data['UTC_Time'][0] = bad_cls_nav_time_value
    try:
        CLS_decoded_nav_data['Lat'][0] = float(Nav_fields[2][1:]) * signs[Nav_fields[2][0]]
    except:
        CLS_decoded_nav_data['Lat'][0] = -999.9
    try:
        CLS_decoded_nav_data['Lon'][0] = float(Nav_fields[3][1:]) * signs[Nav_fields[3][0]]
    except:
        CLS_decoded_nav_data['Lon'][0] = -999.9
    try:
        CLS_decoded_nav_data['TrueHeading'][0] = float(Nav_fields[4])
    except:
        CLS_decoded_nav_data['TrueHeading'][0] = -999.9
    try:
        CLS_decoded_nav_data['PitchAngle'][0] = float(Nav_fields[5][1:]) * signs[Nav_fields[5][0]]
    except:
        CLS_decoded_nav_data['PitchAngle'][0] = -999.9
    try:
        CLS_decoded_nav_data['RollAngle'][0] = float(Nav_fields[6][1:]) * signs[Nav_fields[6][0]]
    except:
        CLS_decoded_nav_data['RollAngle'][0] = -999.9
    try:
        CLS_decoded_nav_data['GroundSpeed'][0] = float(Nav_fields[7])
    except:
        CLS_decoded_nav_data['GroundSpeed'][0] = -999.9
    try:
        CLS_decoded_nav_data['TrackAngleTrue'][0] = float(Nav_fields[8])
    except:
        CLS_decoded_nav_data['TrackAngleTrue'][0] = -999.9
    try:
        CLS_decoded_nav_data['InertialWindSpeed'][0] = float(Nav_fields[9])
    except:
        CLS_decoded_nav_data['InertialWindSpeed'][0] = -999.9
    try:
        CLS_decoded_nav_data['InertialWindDirection'][0] = float(Nav_fields[10])
    except:
        CLS_decoded_nav_data['InertialWindDirection'][0] = -999.9
    try: # Just leave blank if bad.
        CLS_decoded_nav_data['BodyLongitAccl'][0] = float(Nav_fields[11][1:]) * signs[Nav_fields[11][0]]
        CLS_decoded_nav_data['BodyLateralAccl'][0] = float(Nav_fields[12][1:]) * signs[Nav_fields[12][0]]
        CLS_decoded_nav_data['BodyNormalAccl'][0] = float(Nav_fields[13][1:]) * signs[Nav_fields[13][0]]
        CLS_decoded_nav_data['TrackAngleRate'][0] = float(Nav_fields[14][1:]) * signs[Nav_fields[14][0]]
        CLS_decoded_nav_data['PitchRate'][0] = float(Nav_fields[15][1:]) * signs[Nav_fields[15][0]]
        CLS_decoded_nav_data['RollRate'][0] = float(Nav_fields[16][1:]) * signs[Nav_fields[16][0]]
        CLS_decoded_nav_data['InertialVerticalSpeed'][0] = float(Nav_fields[17][1:]) * signs[Nav_fields[17][0]]
    except:
        CLS_decoded_nav_data['BodyLongitAccl'][0] = -999.9
        CLS_decoded_nav_data['BodyLateralAccl'][0] = -999.9
        CLS_decoded_nav_data['BodyNormalAccl'][0] = -999.9
        CLS_decoded_nav_data['TrackAngleRate'][0] = -999.9
        CLS_decoded_nav_data['PitchRate'][0] = -999.9
        CLS_decoded_nav_data['RollRate'][0] = -999.9
        CLS_decoded_nav_data['InertialVerticalSpeed'][0] = -999.9
    try:
        CLS_decoded_nav_data['GPS_Altitude'][0] = float(Nav_fields[18])
    except:
        CLS_decoded_nav_data['GPS_Altitude'][0] = -999.9
    try:
        CLS_decoded_nav_data['GPS_Latitude'][0] = float(Nav_fields[19][1:]) * signs[Nav_fields[19][0]]
    except:
        CLS_decoded_nav_data['GPS_Latitude'][0] = -999.9
    try:
        CLS_decoded_nav_data['GPS_Longitude'][0] = float(Nav_fields[20][1:]) * signs[Nav_fields[20][0]]
    except:
        CLS_decoded_nav_data['GPS_Longitude'][0] = -999.9
    try:
        CLS_decoded_nav_data['StaticPressure'][0] = float(Nav_fields[21])
        CLS_decoded_nav_data['TotalPressure'][0] = float(Nav_fields[22])
        CLS_decoded_nav_data['DifferentialPressure'][0] = float(Nav_fields[23])
        CLS_decoded_nav_data['TotalTemperature'][0] = float(Nav_fields[24][1:]) * signs[Nav_fields[24][0]]
        CLS_decoded_nav_data['StaticTemperature'][0] = float(Nav_fields[25][1:]) * signs[Nav_fields[25][0]]
        CLS_decoded_nav_data['BarometricAltitude'][0] = float(Nav_fields[26])
        CLS_decoded_nav_data['MachNo'][0] = float(Nav_fields[27])
        CLS_decoded_nav_data['TrueAirSpeed'][0] = float(Nav_fields[28])
        CLS_decoded_nav_data['WindSpeed'][0] = float(Nav_fields[29])
        # --> "WindDirection" is nan in PODEX files that I used to develop this. Omitting [4/3/18]
        CLS_decoded_nav_data['SunElevation'][0] = float(Nav_fields[31][1:]) * signs[Nav_fields[31][0]]
        CLS_decoded_nav_data['SunAzimuth'][0] = float(Nav_fields[32][1:]) * signs[Nav_fields[32][0]]
    except:
        CLS_decoded_nav_data['StaticPressure'][0] = -999.9
        CLS_decoded_nav_data['TotalPressure'][0] = -999.9
        CLS_decoded_nav_data['DifferentialPressure'][0] = -999.9
        CLS_decoded_nav_data['TotalTemperature'][0] = -999.9
        CLS_decoded_nav_data['StaticTemperature'][0] = -999.9
        CLS_decoded_nav_data['BarometricAltitude'][0] = -999.9
        CLS_decoded_nav_data['MachNo'][0] = -999.9
        CLS_decoded_nav_data['TrueAirSpeed'][0] = -999.9
        CLS_decoded_nav_data['WindSpeed'][0] = -999.9
        # --> "WindDirection" is nan in Podex files that I used to develop this. Omitting [4/3/18]
        CLS_decoded_nav_data['SunElevation'][0] = -999.9
        CLS_decoded_nav_data['SunAzimuth'][0] = -999.9

    return CLS_decoded_nav_data


def decode_uav_nav_string(Nav_fields, flt_date, bad_cls_nav_time_value):
    """ This function decodes an input UAV-style CLS Nav list of strings into its
        constituent parameters. Function returns a "decoded_nav_struct"
        datatype array of length one.
    """

    # INPUTS:
    #
    # Nav_fields -> List of strings containing Nav parameters to be decoded
    #
    # OUTPUTS:
    #
    # CLS_decoded_nav_data -> 1-element array of dtype CLS_decoded_nav_struct

    datetimeformat = "%Y-%m-%dT%H:%M:%S.%f"

    CLS_decoded_nav_data = np.zeros(1, dtype=CLS_decoded_nav_struct)

    # --> Ommitting "Header" of 'Nav' until someone tells me they need it [4/4/18]
    try:
        CLS_decoded_nav_data['UTC_Time'][0] = DT.datetime.strptime(Nav_fields[1], datetimeformat)
    except:
        CLS_decoded_nav_data['UTC_Time'][0] = bad_cls_nav_time_value
    try:
        CLS_decoded_nav_data['GPS_Latitude'][0] = float(Nav_fields[2])
    except:
        CLS_decoded_nav_data['GPS_Latitude'][0] = -999.9
    try:
        CLS_decoded_nav_data['GPS_Longitude'][0] = float(Nav_fields[3])
    except:
        CLS_decoded_nav_data['GPS_Longitude'][0] = -999.9
    try:
        # Try 4 or 5. May need to put this as init file option someday [5/22/18]		
        CLS_decoded_nav_data['GPS_Altitude'][0] = float(Nav_fields[4])
    except:
        CLS_decoded_nav_data['GPS_Altitude'][0] = -999.9
    try:
        CLS_decoded_nav_data['PitchAngle'][0] = float(Nav_fields[16])
    except:
        CLS_decoded_nav_data['PitchAngle'][0] = -999.9
    try:
        CLS_decoded_nav_data['RollAngle'][0] = float(Nav_fields[17])
    except:
        CLS_decoded_nav_data['RollAngle'][0] = -999.9
    try:
        CLS_decoded_nav_data['GroundSpeed'][0] = float(Nav_fields[8])
    except:
        CLS_decoded_nav_data['GroundSpeed'][0] = -999.9
    try:
        CLS_decoded_nav_data['TrackAngleTrue'][0] = float(Nav_fields[14])
    except:
        CLS_decoded_nav_data['TrackAngleTrue'][0] = -999.9
    try:
        CLS_decoded_nav_data['WindSpeed'][0] = float(Nav_fields[26])
    except:
        CLS_decoded_nav_data['WindSpeed'][0] = -999.9
    try:
        CLS_decoded_nav_data['WindDirection'][0] = float(Nav_fields[27])
    except:
        CLS_decoded_nav_data['WindDirection'][0] = -999.9
    try:
        CLS_decoded_nav_data['SunElevation'][0] = float(Nav_fields[30])
    except:
        CLS_decoded_nav_data['SunElevation'][0] = -999.9
    try:
        CLS_decoded_nav_data['SunAzimuth'][0] = float(Nav_fields[32])
    except:
        CLS_decoded_nav_data['SunAzimuth'][0] = -999.9
    try:
        CLS_decoded_nav_data['StaticTemperature'][0] = float(Nav_fields[20])
    except:
        CLS_decoded_nav_data['StaticTemperature'][0] = -999.9
    try:
        CLS_decoded_nav_data['TotalTemperature'][0] = float(Nav_fields[22])
    except:
        CLS_decoded_nav_data['TotalTemperature'][0] = -999.9
    try:
        CLS_decoded_nav_data['StaticPressure'][0] = float(Nav_fields[23])
    except:
        CLS_decoded_nav_data['StaticPressure'][0] = -999.9

    return CLS_decoded_nav_data    
            
            
def read_in_nav_data(fname, est_recs, UTC_adj=DT.timedelta(0, 0, 0)):
    """ Routine to read in the IWG1 data in from those "nav" files 
        captured by the instrument.
    """
    
    # [6/18/18]
    # Now works for files produced by CPL, in addition to those produced
    # by CAMAL! First, code tries to read in CAMAL-style, then it tries
    # CPL. If neither works, it assumes bad file/data. The 2 file formats
    # only differ in their first column.
    #
    # [6/22/18]
    # Realized the nav text files are different for UAV/ER2 CPL. I 
    # redesigned this code so that now it determines & reads the 
    # appropriate data structure. Bottom line: This function now
    # reads CAMAL, UAV-CPL, & ER2-CPL style nav text files.
    
    nav_data_1file = np.zeros(est_recs, dtype=nav_struct)
    
    with open(fname, 'rb') as navfile:
        
        r = 0
        first_read = True	
        datetimeformatA = "%Y%m%d_%H%M%S: IWG1"
        datetimeformatB = "%Y-%m-%dT%H:%M:%S.%f"
        
        for line in navfile:
            # Split the line into an array of byte objects
            row_byte_list = line.split(b",")
            row = []  # row of strings
            
            # Determine nav text file style
            if first_read:
                first_read = False
                if len(row_byte_list) < 3:
                    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                    print('ER2 Nav-style text file detected')
                    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                    file_style = 'ER2'
                    signs = {'S':-1.0, 'N':1.0, 'W':-1.0, 'E':1.0, '-':-1.0, '+':1.0 }
                elif row_byte_list[0] == b"IWG1":
                    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                    print('UAV Nav-style text file detected')
                    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                    file_style = 'UAV'
                else:
                    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                    print('CAMAL Nav-style text file detected')
                    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                    file_style = 'CAMAL'
           
            # Put invalids (-999.9) in blank fields
            # Also, convert byte obj to str obj
            if file_style == 'ER2': 
                row_byte_list = line.split()   
                row.append('intentionally_blank')
            c = 0
            for col in row_byte_list:
                if col == b"":
                    row.append(-999.9)
                else:
                    row.append(col.decode('utf8', 'replace'))
                c += 1

            try:    
                # These 2 file_styles are identical, save the first column
                if (file_style == 'CAMAL') or (file_style == 'UAV'):
                    if file_style == 'CAMAL':
                        nav_data_1file['UnixT'][r] = DT.datetime.strptime(row[0], datetimeformatA) + UTC_adj
                    nav_data_1file['UTC'][r] = DT.datetime.strptime(row[1], datetimeformatB)
                    nav_data_1file['lat'][r] = row[2]
                    nav_data_1file['lon'][r] = row[3]
                    nav_data_1file['GPS_alt_msl'][r] = row[4]
                    nav_data_1file['GPS_alt'][r] = row[5]
                    nav_data_1file['press_alt'][r] = row[6]
                    nav_data_1file['rad_alt'][r] = row[7]
                    nav_data_1file['ground_spd'][r] = row[8]
                    nav_data_1file['true_airspd'][r] = row[9]
                    nav_data_1file['ind_airspd'][r] = row[10]
                    nav_data_1file['mach_num'][r] = row[11] 
                    nav_data_1file['vert_spd'][r] = row[12]
                    nav_data_1file['heading'][r] = row[13]    # use as yaw
                    nav_data_1file['track'][r] = row[14]
                    nav_data_1file['drift'][r] = row[15] 
                    nav_data_1file['pitch'][r] = row[16]
                    nav_data_1file['roll'][r] = row[17]
                    nav_data_1file['slip'][r] = row[18]
                    nav_data_1file['attack'][r] = row[19]
                    nav_data_1file['S_air_temp'][r] = row[20]
                    nav_data_1file['dewp_temp'][r] = row[21] 
                    nav_data_1file['T_air_temp'][r] = row[22]
                    nav_data_1file['static_p'][r] = row[23] 
                    nav_data_1file['dynmc_p'][r] = row[24]
                    nav_data_1file['cabin_p'][r] = row[25] 
                    nav_data_1file['wind_spd'][r] = row[26]
                    nav_data_1file['wind_dir'][r] = row[27]
                    nav_data_1file['vert_wind_spd'][r] = row[28]
                    nav_data_1file['sol_zen'][r] = row[29] 
                    nav_data_1file['air_sun_elev'][r] = row[30]
                    nav_data_1file['so_azi'][r] = row[31]
                    # nav_data_1file['air_sun_azi'][r] = row[32]
                elif file_style == 'ER2':
                    # You need to decode this first...
                    cls_decoded_nav = decode_er2_nav_string(row, signs, flt_date, bad_cls_nav_time_value)
                    nav_data_1file['UTC'][r] = cls_decoded_nav['UTC_Time'][0]
                    nav_data_1file['lat'][r] = cls_decoded_nav['GPS_Latitude'][0]
                    nav_data_1file['lon'][r] = cls_decoded_nav['GPS_Longitude'][0]
                    nav_data_1file['GPS_alt_msl'][r] = cls_decoded_nav['GPS_Altitude'][0]
                    nav_data_1file['GPS_alt'][r] = cls_decoded_nav['GPS_Altitude'][0]
                    pdb.set_trace()
                    nav_data_1file['press_alt'][r] = cls_decoded_nav['BarometricAltitude'][0]
                    nav_data_1file['rad_alt'][r] = -999.9
                    nav_data_1file['ground_spd'][r] = cls_decoded_nav['GroundSpeed'][0]
                    nav_data_1file['true_airspd'][r] = cls_decoded_nav['TrueAirSpeed'][0]
                    nav_data_1file['ind_airspd'][r] = -999.9
                    nav_data_1file['mach_num'][r] = cls_decoded_nav['MachNo'][0]
                    nav_data_1file['vert_spd'][r] = cls_decoded_nav['InertialVerticalSpeed'][0]
                    nav_data_1file['heading'][r] = cls_decoded_nav['TrueHeading'][0] # use as yaw
                    nav_data_1file['track'][r] = cls_decoded_nav['TrackAngleTrue'][0]
                    nav_data_1file['drift'][r] = -999.9
                    nav_data_1file['pitch'][r] = cls_decoded_nav['PitchAngle'][0]
                    nav_data_1file['roll'][r] = cls_decoded_nav['RollAngle'][0]
                    nav_data_1file['slip'][r] = -999.9
                    nav_data_1file['attack'][r] = -999.9
                    nav_data_1file['S_air_temp'][r] = cls_decoded_nav['StaticTemperature'][0]
                    nav_data_1file['dewp_temp'][r] = -999.9
                    nav_data_1file['T_air_temp'][r] = cls_decoded_nav['TotalTemperature'][0]
                    nav_data_1file['static_p'][r] = cls_decoded_nav['StaticPressure'][0]
                    nav_data_1file['dynmc_p'][r] = cls_decoded_nav['DifferentialPressure'][0]
                    nav_data_1file['cabin_p'][r] = -999.9
                    nav_data_1file['wind_spd'][r] = cls_decoded_nav['WindSpeed'][0]
                    nav_data_1file['wind_dir'][r] = cls_decoded_nav['InertialWindDirection'][0]
                    nav_data_1file['vert_wind_spd'][r] = cls_decoded_nav['InertialVerticalSpeed'][0]
                    nav_data_1file['sol_zen'][r] = -999.9
                    nav_data_1file['air_sun_elev'][r] = cls_decoded_nav['SunElevation'][0]
                    nav_data_1file['so_azi'][r] = cls_decoded_nav['SunAzimuth'][0]
                r += 1  
            except:
                print('There is a bad record in this file.')
                print('Bad record in : '+fname)
                print('row # : ' + str(r))
                continue

    nav_data_1file = nav_data_1file[0:r]
    return nav_data_1file

    
def read_in_dead_time_table(fname):
    """ Code to read in CAMAL's dead time table files and return the 
        table within as (a) numpy array(s).

        Should be able to read in dead time tables for CPL too, since
        they are the same format.
    """
    
    # NOTES:
    #
    # [3/6/18]
    # Code currently reads in files that are in CPL's XDR format, originally
    # instated by Mr. Dennis Hlavka (now retired).
    
    data_list = []
    
    with open(fname, 'rb') as f_obj:
        for block in iter(partial(f_obj.read, 4), ''):
            try:
                data_list.append(xdrlib.Unpacker(block).unpack_float())
            except EOFError:
                break
    print("Done reading in dead time table:\n"+fname)
    
    # Convert list to array then return array
    return np.asarray(data_list, dtype=np.float32)
    
    
def read_entire_gps_dataset(file_len_secs, gps_hz, raw_dir, gps_pitch_offset, gps_roll_offset, leapsecs, scrub='no'):
    """ Read in entire gps file dataset (all files). Return a gps_struct
        array of the data, as well as separate, gps_UTC datetime.datetime
        array (for convenience   ).
    """
    
    # NOTES:
    # 
    # This function is COMPLETELY RELIANT on the initializations file,
    # so no arguments are required by the calling routine in the scope of
    # CAMAL's processing. If scrub == 'scrub' bad data records will be 
    # filtered out.
    
    GPS_file_list = 'gps_file_list.txt'
    search_str = 'gps*'
    create_a_file_list(GPS_file_list, search_str, raw_dir)
    
    with open(GPS_file_list) as GPS_list_fobj:
        all_GPS_files = GPS_list_fobj.readlines()
    nGPS_files = len(all_GPS_files)
    
    j = 0
    est_gps_recs = int((file_len_secs * (gps_hz+1))*nGPS_files) # add a little buffer
    gps_data_all = np.zeros(est_gps_recs, dtype=gps_struct)
    for GPS_file in all_GPS_files:
        gps_data_1file = read_in_gps_data(GPS_file.strip(), file_len_secs, gps_hz, gps_pitch_offset, gps_roll_offset)
        try:
            gps_data_all[j:j+gps_data_1file.shape[0]] = gps_data_1file
        except ValueError:
            pdb.set_trace()
        except AttributeError:
            continue
        j += gps_data_1file.shape[0]      
    gps_data_all = gps_data_all[0:j]  # trim the fat
    
    epoch = DT.datetime.strptime("1980-01-06 00:00:00", "%Y-%m-%d %H:%M:%S")
    gps_UTC = np.zeros(gps_data_all.shape[0],dtype=DT.datetime)
    for j in range(0, gps_data_all.shape[0]):
        try:
            gps_UTC[j] = weeksecondstoutc(gps_data_all['GpsWeek'][j]*1.0,
                                          gps_data_all['Gps_sec'][j], leapsecs)
        except:
            gps_UTC[j] = epoch
            
    # Filter out bad GPS times. It seems like there will be many.
    # Try to do this in an automated way that doesn't require more
    # initialization file variables.
    if scrub == 'scrub':
        tlim0 = weeksecondstoutc(np.median(gps_data_all['GpsWeek'])-1.0, 0.0, leapsecs)
        tlim1 = weeksecondstoutc(np.median(gps_data_all['GpsWeek'])+1.0, 0.0, leapsecs)
        mask0 = gps_UTC >= tlim0
        t_stage0 = gps_UTC[mask0]
        gps_data_all = gps_data_all[mask0]
        mask1 = t_stage0 <= tlim1
        gps_UTC = t_stage0[mask1]
        gps_data_all = gps_data_all[mask1]

    return [gps_data_all, gps_UTC] 
    
    
def read_entire_nav_dataset(search_str, raw_dir, est_nav_recs_1file, secs_btwn_instr_UnixT_and_UTC):
    """ Read in entire nav file dataset (all files) """
    
    # [6/18/18] 
    # Changed search_str from 'nav*' to '*nav*' so that
    # code would be able to read the CPL-style nav files. Made this a
    # default argument to this function, as opposed to hard-coding it
    # into the body of the function so that it can easily be adapting to
    # calling code's needs.

    nav_file_list = 'nav_file_list.txt'
    create_a_file_list(nav_file_list, search_str, raw_dir)
            
    with open(nav_file_list) as nav_list_fobj:
        all_nav_files = nav_list_fobj.readlines()
    nnav_files = len(all_nav_files)
    
    est_nav_recs = est_nav_recs_1file*nnav_files
    nav_data_all = np.zeros(est_nav_recs, dtype=nav_struct)
    j = 0
    for i in range(0, nnav_files):
        nav_file = all_nav_files[i]
        nav_file = nav_file.strip()
        nav_data_1file = read_in_nav_data(nav_file, est_nav_recs_1file,
                                          DT.timedelta(seconds=secs_btwn_instr_UnixT_and_UTC))
        try:
            nav_data_all[j:j+nav_data_1file.shape[0]] = nav_data_1file
        except ValueError:
            pdb.set_trace()
        j += nav_data_1file.shape[0]
    nav_data_all = nav_data_all[0:j] # trim the fat
    
    return nav_data_all


def read_in_cls_data(fname, nbins, flt_date, bad_cls_nav_time_value, return_nav_dict=False):
    """ Routine to read in raw data from a single CPL "CLS" file. 
    """
    
    # INPUT:
    # fname -> Full-path name of CLS file.
    # nbins -> Integer # of bins, likely 833.
    # flt_date -> String like this, rep. date of flight: '06dec19'
    # bad_cls_nav_time_value -> datetime.datetime object representing invalid data.
    #                           Recommended to set to DT.datetime(1970,1,1,0,0,0).
    # return_nav_dict -> Only known application is for L1A processing.
    
    # OUTPUT:
    # CLS_decoded_data -> Numpy array structure with length equal to number
    #     of records within CLS file. Count are raw, except header and Nav
    #     data have been decoded where necessary.
    # [CLS_decoded_data, Nav_dict] -> Returned if nav_dict is True.
    
    # NOTES:
    #
    # CLS files DO NOT contain the year. This routine reads the year from the initialization file!
    #
    # This code ASSUMES that ASCII white space ALWAYS separates the fields in the 256 byte Nav record!
    #
    # Bill and Dennis used "GPS" parameters for processing. So use "GPS_Latitude" not "Lat."
    #
    # Invalid and/or missing fields set to -999.9
    #
    # Inputs added because initializations.py import was removed from this module. [2/25/20]

    # Assume an ER2 style CLS file as default
    ER2_nav_nbytes = 256
    UAV_nav_nbytes = 326
    ER2_tot_nbytes = ER2_nav_nbytes + 6912
    UAV_tot_nbytes = UAV_nav_nbytes + 6912
    cls_type = 'ER2'
    CLS_meta_struct = define_CLS_meta_struct(ER2_nav_nbytes)
    
    # First, determine what type of CLS file you're reading (ER2, UAV, other??)
    # then get nchans and nbins from the file...
    samp = np.fromfile(fname, CLS_meta_struct, count=1)
    # See which record size the file is a multiple of: ER2 or UAV
    fsize = os.path.getsize(fname)
    size_check = fsize % ER2_tot_nbytes
    if size_check != 0:
        size_type = 'UAV'
    else:
        size_type = 'ER2'
    # First, try as ER2-style ....
    num_nav_fields = len(str(samp['Nav'].view('S'+str(ER2_nav_nbytes).strip())).split())
    if (num_nav_fields < 6) & (size_type == 'UAV'):
        print('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print("UAV-style CLS file detected!")
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
        CLS_meta_struct = define_CLS_meta_struct(UAV_nav_nbytes)
        navSview = 'S'+str(UAV_nav_nbytes).strip()
        Nav_delim = ','
        cls_type = 'UAV'
    elif (num_nav_fields >= 6) & (num_nav_fields < 17):
        print('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print("WB57-to-ER2-style CLS file detected!")
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')        
        CLS_meta_struct = define_CLS_meta_struct(ER2_nav_nbytes)       
        navSview =  'S'+str(ER2_nav_nbytes).strip()
        Nav_delim = None
        cls_type = 'WB57-to-ER2'  
    else:
        print('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print("ER2-style CLS file detected!")
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')        
        CLS_meta_struct = define_CLS_meta_struct(ER2_nav_nbytes)       
        navSview =  'S'+str(ER2_nav_nbytes).strip()
        Nav_delim = None
        cls_type = 'ER2'

    # Next, use that info to form your CLS data structure...
    try:
        # NOTE: number of bins NOT included in CLS data
        CLS_raw_struct = define_CLS_structure(samp['Header']['NumChannels'][0], nbins, CLS_meta_struct)
        CLS_decoded_struct = define_CLS_decoded_structure(samp['Header']['NumChannels'][0], nbins)
    except IndexError:
        print('Size of ingested data: ', samp.shape)
        print('Bad data. Skipping file...')
        if return_nav_dict:
            return None, None
        else:
            return None
    
    # Read in all data from the file at once
    CLS_data = np.fromfile(fname, CLS_raw_struct, count=-1)
    
    # Decode the character bytes into usable data types
    CLS_decoded_data = np.zeros(CLS_data.shape[0], dtype=CLS_decoded_struct)
    signs = {'S': -1.0, 'N': 1.0, 'W': -1.0, 'E': 1.0, '-': -1.0, '+': 1.0}
    Nav_dict = {}  # Dictionary to store times as keys to Nav records
    UTC_Time_last_rec = DT.datetime(1990, 1, 1, 12, 0, 0)
    for i in range(0, CLS_data.shape[0]):

        # Conversions for the Header portion of CLS record...
        try:
            temp_string = str(CLS_data['meta']['Header']['ExactTime'][i, :].view('S27'))
            CLS_decoded_data['meta']['Header']['ExactTime'][i] = \
                DT.datetime.strptime(flt_date[5:7]+temp_string[3:28], '%yDATE__%j__TIME__%H:%M:%S')
            # As of 4/4/18, none of the other raw header parameters need to be decoded.
            CLS_decoded_data['meta']['Header']['RecordNumber'][i] = CLS_data['meta']['Header']['RecordNumber'][i]
            CLS_decoded_data['meta']['Header']['NumChannels'][i] = CLS_data['meta']['Header']['NumChannels'][i]
            CLS_decoded_data['meta']['Header']['Resolution'][i] = CLS_data['meta']['Header']['Resolution'][i]
            CLS_decoded_data['meta']['Header']['DetectorSequence'][i, :] = CLS_data['meta']['Header']['DetectorSequence'][i, :]
            CLS_decoded_data['meta']['Header']['NumberT_Probes'][i] = CLS_data['meta']['Header']['NumberT_Probes'][i]
            CLS_decoded_data['meta']['Header']['NumberV_Probes'][i] = CLS_data['meta']['Header']['NumberV_Probes'][i]
            CLS_decoded_data['meta']['Header']['Reserved'][i, :] = CLS_data['meta']['Header']['Reserved'][i, :]
        except:
            print("UNUSABLE RECORD ", i, fname)
            print(temp_string[3:28])	    
            CLS_decoded_data['meta']['Header']['ExactTime'][i] = bad_cls_nav_time_value

        # Conversions for Nav portion of CLS record...
        Nav_fields = str(CLS_data['meta']['Nav'][i].view(navSview)).split(Nav_delim)

        if cls_type == 'ER2':
            # WRITE CODE TO PARSE ER2-STYLE NAV RECORD
            CLS_decoded_data['meta']['Nav'][i] = decode_er2_nav_string(Nav_fields, signs, flt_date, bad_cls_nav_time_value)
        elif cls_type == 'UAV':
            CLS_decoded_data['meta']['Nav'][i] = decode_uav_nav_string(Nav_fields, flt_date, bad_cls_nav_time_value)
        else:
            # Don't split. We're going to decode by byte position
            Nav_fields = str(CLS_data['meta']['Nav'][i].view(navSview) )
            CLS_decoded_data['meta']['Nav'][i] = decode_er2_like_wb57_nav_string(Nav_fields, signs)

        if CLS_decoded_data['meta']['Nav']['UTC_Time'][i] != UTC_Time_last_rec: 
            Nav_dict[CLS_decoded_data['meta']['Nav']['UTC_Time'][i].strftime('%y-%j:%H:%M:%S')] = \
                CLS_decoded_data['meta']['Nav'][i]
        UTC_Time_last_rec = CLS_decoded_data['meta']['Nav']['UTC_Time'][i]
   
    # Actual photon count data can just be copied directly from "raw" to "decoded."
    CLS_decoded_data['counts'] = CLS_data['counts']
    # So can engineering data...
    CLS_decoded_data['meta']['Engineering'] = CLS_data['meta']['Engineering']
    
    if return_nav_dict:
        return [CLS_decoded_data, Nav_dict]
    else:
        return CLS_decoded_data


def remove_status_packets_from_data_edges(in_meta, bad_cls_nav_time_value, drift_plot=False):
    """ Function to eliminate status records from flight data.
        If first records of first file are invalid, this code 
        will handle that too, in the same manner.
    """

    # INPUTS:
    # in_meta -> input CLS Nav data w/o time modifications
    
    # OUTPUTS:
    # [,] -> list containing the number of records to skip from [beg.,end]

    # NOTES:
    #
    # This function assumes that valid instrument  times don't repeat themselves
    # non-consecutively. In other words, valids times don't jump backward/forwards
    # in time.

    u, ui, ncounts = np.unique(in_meta['Header']['ExactTime'], return_index=True, return_counts=True)
    u_uns = [in_meta['Header']['ExactTime'][indx] for indx in sorted(ui)]  # undo sort the unique values by their value
    u_uns = np.array(u_uns)  # convert from list to array
    sort_indx_key = np.argsort(ui)
    ncounts_uns = ncounts[sort_indx_key]
    ui_uns = ui[sort_indx_key]

    # Compute time delta between records
    delts = delta_datetime_vector(u_uns)

    # Get rid of outlier times at beginning/end that result from sending of status profile
    # that occurs when laser is not firing.
    skip_first = 0
    skip_last = 0
    i = 0
    j = delts.shape[0]
    while (abs(delts[i+1]) > 3) or (u_uns[i] == bad_cls_nav_time_value):
        skip_first = ui_uns[i+1]
        i += 1
    while (abs(delts[j-1]) > 3) or (u[j-1] == bad_cls_nav_time_value):
        skip_last += ncounts_uns[j-1]
        j -= 1

    # Analyze "expected" versus "data" time
    # Used for debugging
    if drift_plot:
        from matplotlib import pyplot as plt
        # first/last rec with ncounts >= 10
        first_rec_time = u[ncounts >= 10][0]
        last_rec_time = u[ncounts >= 10][-1]
        num_before = ncounts[0+skip_first]
        num_after = ncounts[-1-skip_last]
        computed_slop = num_before + num_after
        before_rec = ui[0+skip_first]
        after_rec = ui[-1-skip_last]
        st = first_rec_time
        ed = last_rec_time
        tott = (ed - st).total_seconds()
        pred_intrr = int(tott * 10)
        pred_recs = int(pred_intrr + computed_slop)
        pred_time = np.zeros(pred_intrr, dtype=DT.datetime)
        for i in range(0, pred_intrr, 10): pred_time[i:i+10] = st + DT.timedelta(seconds=i/10)
        ptf = np.zeros(pred_recs, dtype=DT.datetime)  # predicted time final
        ptf[0:num_before] = in_meta['UTC_Time'][before_rec]
        ptf[-1-num_after:] = in_meta['UTC_Time'][after_rec]
        ptf[num_before:-1*num_after] = pred_time
        ptford = np.arange(0, ptf.shape[0])
        at = in_meta['UTC_Time'][before_rec:after_rec]  # actual time in data
        atord = np.arange(0, at.shape[0])
        plt.plot_date(ptf, ptford, marker='o')
        plt.plot_date(at, atord, marker='x')
        ax = plt.gca()
        ax.set_xlim([st-DT.timedelta(seconds=5), ed+DT.timedelta(seconds=5)])
        plt.show()
        if ptf.shape[0] < at.shape[0]:
            diff_at_pt = ptf - at[:ptf.shape[0]]
        else:
            diff_at_pt = at - ptf[0:at.shape[0]]
        ds = np.zeros(diff_at_pt.shape, dtype=np.float64)
        for i in range(0, ds.shape[0]): ds[i] = diff_at_pt[i].total_seconds()
        plt.plot(ds, marker='x')
        plt.title('Difference between expected and actual times')    
        plt.show()
        pdb.set_trace()
    return [skip_first, skip_last]    


def read_entire_cls_dataset(file_len_recs, raw_dir, nbins, flt_date, bad_cls_nav_time_value, Fcontrol=None):
    """ This function reads in all the CLS data from the entire flight.
    """
    
    # NOTES:
    #
    # [5/24/18]
    # Fcontrol, or file control, added to ease recycling of CAMAL GUI code
    # to create CPL GUI
    #
    # [10/9/19]
    # Retooled slightly to accomodate the removal of the initializations
    # import from this module.

    cls_file_list = 'cls_file_list_for_nav_only.txt'
    search_str = '*.cls'
    create_a_file_list(cls_file_list, search_str, raw_dir)
            
    with open(cls_file_list) as cls_list_fobj:
        all_cls_files = cls_list_fobj.readlines()
    n_cls = len(all_cls_files)
    if (max(Fcontrol.sel_file_list) < n_cls) & (Fcontrol is not None):
        ncls_files = len(Fcontrol.sel_file_list)
        actual_cls_files = []
        for x in Fcontrol.sel_file_list:
            if x > len(all_cls_files)-1: continue
            actual_cls_files.append(all_cls_files[x])
        all_cls_files = actual_cls_files
        Fnumbers = Fcontrol.sel_file_list
    else:
        ncls_files = len(all_cls_files)
        Fnumbers = [x for x in range(0, ncls_files)]
    
    est_cls_recs = file_len_recs*ncls_files
    file_indx = np.zeros((ncls_files, 2), dtype=np.uint32)
    j = 0
    k = 0
    for i in Fnumbers:
        if i not in Fcontrol.sel_file_list:
            k += 1
            continue		
        cls_file = all_cls_files[k]
        cls_file = cls_file.strip()
        cls_data_1file, Nav_dict = read_in_cls_data(cls_file, nbins, flt_date, bad_cls_nav_time_value, True)
        if j == 0:
             max_chan = cls_data_1file['meta']['Header']['NumChannels'][0]
             cls_data_all = np.zeros(est_cls_recs, dtype=define_CLS_decoded_structure(max_chan, nbins))
        try:
            cls_data_all[j:j+cls_data_1file.shape[0]] = cls_data_1file
        except:
            pdb.set_trace()
        j += cls_data_1file.shape[0]
        k += 1
        print('Finshed ingesting file: '+cls_file)
    cls_data_all = cls_data_all[0:j]  # trim the fat

    return cls_data_all  # NOTE: It is decoded.


def read_entire_cls_nav_dataset():
    """ This function reads in ONLY the Nav (256-byte) record from the
        CPL CLS files.
    """

    cls_file_list = 'cls_file_list_for_nav_only.txt'
    search_str = '*.cls'
    create_a_file_list(cls_file_list, search_str)
            
    with open(cls_file_list) as cls_list_fobj:
        all_cls_files = cls_list_fobj.readlines()
    ncls_files = len(all_cls_files)
    
    est_nav_recs = file_len_recs*ncls_files
    nav_data_all = np.zeros(est_nav_recs, dtype=CLS_decoded_nav_struct)
    file_indx = np.zeros((ncls_files, 2), dtype=np.uint32)
    j = 0
    tot_Nav_dict = {}
    for i in range(0, ncls_files):
        cls_file = all_cls_files[i]
        cls_file = cls_file.strip()
        cls_data_1file, Nav_dict = read_in_cls_data(cls_file, True)
        tot_Nav_dict.update(Nav_dict)
        try:
            nav_data_all[j:j+cls_data_1file.shape[0]] = cls_data_1file['meta']['Nav']
            file_indx[i, 0] = j
            file_indx[i, 1] = j+cls_data_1file.shape[0]
        except:
            pdb.set_trace()
        j += cls_data_1file.shape[0]
        print('Finshed ingesting file: '+cls_file+' for Nav interp purposes!')
    nav_data_all = nav_data_all[0:j]  # trim the fat
    return nav_data_all


def read_entire_cls_meta_dataset(raw_dir, file_len_recs, nbins, flt_date, bad_cls_nav_time_value):
    """ This function reads in ONLY the Nav (256-byte) record from the
        CPL CLS files.
    """

    cls_file_list = 'cls_file_list_for_nav_only.txt'
    search_str = '*.cls'
    create_a_file_list(cls_file_list, search_str, raw_dir)
            
    with open(cls_file_list) as cls_list_fobj:
        all_cls_files = cls_list_fobj.readlines()
    ncls_files = len(all_cls_files)
    
    est_nav_recs = file_len_recs*ncls_files
    meta_data_all = np.zeros(est_nav_recs, dtype=CLS_decoded_meta_struct)
    file_indx = np.zeros((ncls_files, 2), dtype=np.int32)
    j = 0
    tot_Nav_dict = {}
    for i in range(0, ncls_files):
        cls_file = all_cls_files[i]
        cls_file = cls_file.strip()
        cls_data_1file, Nav_dict = read_in_cls_data(cls_file, nbins, flt_date, bad_cls_nav_time_value, True)
        if cls_data_1file is None: continue
        tot_Nav_dict.update(Nav_dict)
        try:
            meta_data_all[j:j+cls_data_1file.shape[0]] = cls_data_1file['meta']
            file_indx[i, 0] = j
            file_indx[i, 1] = j+cls_data_1file.shape[0]
        except:
            pdb.set_trace()
        j += cls_data_1file.shape[0]
        print('Finshed ingesting file: '+cls_file+' for meta purposes!')
    meta_data_all = meta_data_all[0:j]  # trim the fat

    skip_list = remove_status_packets_from_data_edges(meta_data_all, bad_cls_nav_time_value)
    meta_data_all = meta_data_all[skip_list[0]:meta_data_all.shape[0]-skip_list[1]]
    file_indx = file_indx - skip_list[0]
    file_indx[0, 0] = 0

    # Whittle down the number of files if necessary. This would occur if the values in skip_list
    # are > the number of records in some of the files at the edge.
    beg_file_indx = 0
    end_file_indx = ncls_files
    fi = 0
    while file_indx[fi, 1] <= 0:
        if fi == (ncls_files-2):
            print("Too many data are being removed. Investigate here.")
            pdb.set_trace()
        fi += 1
        if file_indx[fi, 0] > 0:
            print("Something odd is going on. A gap?")
            pdb.set_trace()
        if file_indx[fi, 0] < 0:
            skip_list[0] = abs(file_indx[fi, 0])
        else:
            skip_list[0] = 0
        file_indx[fi, 0] = 0
    if fi > 0: 
        # Don't shrink array. "beg_file_indx" will tell L1A main what to do.
        beg_file_indx = fi
        print(attention_bar)
        print("Insufficient usable records are in the first ",
             beg_file_indx, " file(s).")
        print(attention_bar)
    interim_ncls_files = file_indx.shape[0]
    fi = interim_ncls_files - 1
    file_indx[fi, 1] = file_indx[fi, 1] - skip_list[1]
    while file_indx[fi, 1] < file_indx[fi, 0]:
        leftover = file_indx[fi, 0] - file_indx[fi, 1]
        fi -= 1
        file_indx[fi, 1] = file_indx[fi, 1] - leftover
        skip_list[1] = leftover
    if fi < (interim_ncls_files - 1): 
        file_indx = file_indx[:fi+1, :]
        end_file_indx = fi+1
        print(attention_bar)
        print("Insufficient usable records are in the last ",
              ncls_files - end_file_indx, " file(s).")
        print(attention_bar)

    usable_file_range = [beg_file_indx, end_file_indx]
    return meta_data_all, file_indx, skip_list, usable_file_range


def read_in_overlap_table(fname):
    """ Code to read in an overlap table file and return the 
        table within as a numpy array.

        Should be able to read in overlap tables for both CPL
        and CAMAL, since they have the same format.
    """
    
    # NOTES:
    #
    # [4/9/18]
    # Code currently reads in files that are in CPL's XDR format, originally
    # instated by Mr. Dennis Hlavka (now retired).
    
    data_list = []
    
    with open(fname, 'rb') as f_obj:
        for block in iter(partial(f_obj.read, 4), ''):
            try:
                data_list.append(xdrlib.Unpacker(block).unpack_float())
            except EOFError:
                break
    print("Done reading in overlap table:\n"+fname)
    
    # Convert list to array then return array
    return np.asarray(data_list, dtype=np.float32)


def read_in_esrl_raob(infile):
    """ Code to read in raobs, in the format of those archived on the 
        ESRL site (https://ruc.noaa.gov/raobs/)
    """
    
    # INPUTS:
    #
    # infile -> ESRL raob file (ASCII)
    #
    # OUTPUTS:
    #
    # raob -> Numpy array structured according to dtype "raob_struct" defined
    #         in immutable_data_structs.py
    
    data_list = []
    
    with open(infile) as f_obj:
        line = f_obj.readline()
        while line:
            line_split = line.split()  # split on whitespace, no arg needed
            try:
                data_list.append([int(s) for s in line_split])
            except:
                data_list = []  # reset
            line = f_obj.readline()                
    
    raob = np.zeros(len(data_list), dtype=raob_struct)
    i = 0
    for item in data_list:
        raob['ltp'][i] = item[0]
        raob['pres'][i] = item[1]
        raob['alt'][i] = item[2]
        raob['temp'][i] = item[3]
        raob['dewp'][i] = item[4]
        raob['wdir'][i] = item[5]
        raob['wspd'][i] = item[6]
        i += 1
        
    return raob


def read_in_housekeeping_data_r(fname):
    """ This "_r" version operates on Roscoe files.
       
        Routine to read in housekeeping data from a single file.
        Takes only one input, the file name. All other constructs
        are provided from imported modules.
    """
    
    return np.fromfile(fname, hk_struct_r, count=-1)
    
    
def read_in_mcs_data_r(fname):
    """ Routine to read in raw data from a single Roscoe MCS file. 
        Takes only one input, the file name. All other constructs
        are provided from imported modules (serve like IDL's 
        common blocks)
    """
    
    # First, get nchans and nbins from the file...
    samp = np.fromfile(fname, MCS_meta_struct_r, count=1)
    
    # Next, use that info to form your MSC data structure...
    try:
        MCS_struct = define_MSC_structure_r(samp['nchans'][0], samp['nbins'][0])
    except IndexError:
        print('Size of ingested data: ', samp.shape)
        print('Bad data. Skipping file...')
        return None
    
    # Finally, read in all data from the file at once
    MCS_data = np.fromfile(fname, MCS_struct, count=-1)
    
    return MCS_data


def read_selected_mcs_files_r(MCS_file_list, rep_rate, file_len_secs, Fcontrol=None):
    """Function that will read in multiple Roscoe MCS files.
       Takes in a list of files to read as its input and returns a 
       numpy array "MCS_data" which is structured as "MCS_struct_r."
    """
    # INPUTS:
    # MSC_file_list: List of MCS files to read
    # Fcontrol:      List controlling which MCS files are read. Optional.

    with open(MCS_file_list) as MCS_list_fobj:
        all_MCS_files = MCS_list_fobj.readlines()
    nMCS_files = len(all_MCS_files)
    if Fcontrol is not None:
        n_files = nMCS_files
    else:
        n_files = len(Fcontrol)

    # Read in the MCS (science) data

    first_read = 1
    r = 0
    for i in range(0, nMCS_files):
        if i not in Fcontrol: continue
        print("File # ====== ", i)
        MCS_file = all_MCS_files[i]
        MCS_file = MCS_file.rstrip()
        MCS_data_1file = read_in_mcs_data_r(MCS_file)
        if MCS_data_1file is None:
            print('Skipping this file. Potentially corrupt data')
            continue
        # As of 10/8/19, no method to filter out bad recs for Roscoe.            
        # skip_whole_file = filter_out_bad_recs(MCS_data_1file)
        skip_whole_file = 0
        if skip_whole_file == 1:
            print('Skipping this file...') 
            continue
        if first_read:
            first_read = 0
            nc = MCS_data_1file['meta']['nchans'][0]
            nb = MCS_data_1file['meta']['nbins'][0]            
            # Put the parameters that won't change during 1 flight into variables
            nshots = MCS_data_1file['meta']['nshots'][0]
            # declare data structure to hold all data. estimate tot # recs first
            tot_est_recs = int(rep_rate/nshots)*file_len_secs*n_files
            MCS_struct_r = define_MSC_structure_r(nc, nb)
            MCS_data = np.zeros(tot_est_recs, dtype=MCS_struct_r)
        nr_1file = MCS_data_1file.shape[0]
        MCS_data[r:r+nr_1file] = MCS_data_1file
        r += nr_1file   
        # NOTE: You could put conditional break
        #      statement in this loop to read-in
        #      data from time segment only. 
    
    try:
        MCS_data = MCS_data[0:r]
    except UnboundLocalError:
        print("\n\n******************************************\n")
        print("There is no valid data in selected files. Pick other files.\n")
        print("******************************************\n")
        return False
    
    print('All MCS data are loaded.')    
    return MCS_data 


def read_selected_mcs_files(MCS_file_list, rep_rate, file_len_secs, Fcontrol=None):
    """Function that will read in multiple CAMAL MCS files.
       Takes in a list of files to read as its input and returns a 
       numpy array "MCS_data" which is structured as "MCS_struct."
    """

    # INPUTS:
    # MSC_file_list: List of MCS files to read
    # Fcontrol:      List controlling which MCS files are read. Optional.

    with open(MCS_file_list) as MCS_list_fobj:
        all_MCS_files = MCS_list_fobj.readlines()
    nMCS_files = len(all_MCS_files)
    if Fcontrol is not None:
        n_files = nMCS_files
    else:
        n_files = len(Fcontrol)	

    # Read in the MCS (science) data

    first_read = 1
    r = 0
    for i in range(0, nMCS_files):
        if i not in Fcontrol: continue
        print("File # ====== ", i)
        MCS_file = all_MCS_files[i]
        MCS_file = MCS_file.rstrip()
        MCS_data_1file = read_in_raw_data(MCS_file)
        if MCS_data_1file is None:
            print('Skipping this file. Potentially corrupt data')
            continue
        skip_whole_file = filter_out_bad_recs(MCS_data_1file)
        if skip_whole_file == 1:
            print('Skipping this file...') 
            continue
        if first_read:
            first_read = 0	
            # Put the parameters that won't change during 1 flight into variables
            nshots = MCS_data_1file['meta']['nshots'][0]
            vrT = MCS_data_1file['meta']['binwid'][0]
            vrZ = (vrT * c) / 2.0
            data_params = [nc, nb, nshots, vrT, vrZ]  # store in list
            # declare data structure to hold all data. estimate tot # recs first
            n_files = min(nMCS_files, len(Fcontrol.sel_file_list))
            tot_est_recs = int(rep_rate/nshots)*file_len_secs*n_files
            MCS_struct = define_MSC_structure(nc, nb)
            MCS_data = np.zeros(tot_est_recs, dtype=MCS_struct)
        nr_1file = MCS_data_1file.shape[0]
        MCS_data[r:r+nr_1file] = MCS_data_1file
        r += nr_1file   
        # NOTE: You could put conditional break
        #      statement in this loop to read-in
        #      data from time segment only. 
    
    try:
        MCS_data = MCS_data[0:r]
    except UnboundLocalError:
        print("\n\n******************************************\n")
        print("There is no valid data in selected files. Pick other files.\n")
        print("******************************************\n")
        return False

    print('All MCS data are loaded.')    
    return MCS_data


def read_acls_std_atm_file(std_atm_file):
    """ Read in data from those classic ACLS standard atmosphere files
        that are used for CPL.
    """

    with open(std_atm_file, 'r') as f_obj:
        Ray_data = f_obj.readlines()

    dz = 29.98
    sratm = 8.0*np.pi/3.0  # molecular lidar ratio
    Bray_alt = []
    Bray355 = []
    Bray532 = []
    Bray1064 = []
    for line in Ray_data:
        linesplit = line.split()
        if len(linesplit) > 7: continue
        Bray_alt.append(float(linesplit[0]))
        Bray355.append(float(linesplit[4]))
        Bray532.append(float(linesplit[5]))
        Bray1064.append(float(linesplit[6]))
    
    Bray_alt = np.asarray(Bray_alt)
    Bray355 = np.asarray(Bray355)
    Bray532 = np.asarray(Bray532)
    Bray1064 = np.asarray(Bray1064)
    mtsq_slant355 = np.zeros(Bray355.shape[0])
    mtsq_slant532 = np.zeros(Bray532.shape[0])
    mtsq_slant1064 = np.zeros(Bray1064.shape[0])
    
    odm355 = - 0.5 * np.log(1.0)
    odm532 = - 0.5 * np.log(1.0)
    odm1064 = - 0.5 * np.log(1.0)
    for j in range(0, Bray355.shape[0]):
        odm355 = odm355 + Bray355[j] * sratm * dz/1e3
        odm532 = odm532 + Bray532[j] * sratm * dz/1e3
        odm1064 = odm1064 + Bray1064[j] * sratm * dz/1e3
        mtsq_slant355[j] = (np.exp(-2.0*odm355))  # **(1.0/np.cos(ONA_samp))
        mtsq_slant532[j] = (np.exp(-2.0*odm532))  # **(1.0/np.cos(ONA_samp))
        mtsq_slant1064[j] = (np.exp(-2.0*odm1064))  # **(1.0/np.cos(ONA_samp))
    AMB355 = Bray355 * mtsq_slant355 
    AMB532 = Bray532 * mtsq_slant532
    AMB1064 = Bray1064 * mtsq_slant1064
    
    return [Bray_alt, AMB355, AMB532, AMB1064]


def read_in_cats_l0_data(cats_l0_fname, nchans, nbins):
    """ Read data from single CATS (ISS) file into a structured numpy array.
        'nchans' varies based on raw data mode.
    """
    
    cats_l0_structure = define_CATS_L0_struct(nchans, nbins)
    
    l0_data = np.fromfile(cats_l0_fname, dtype=cats_l0_structure)
    l0_data['chan'] = np.bitwise_and(l0_data['chan'], 32767)  # remove parity bit
    
    return l0_data

