# NOTE: I don't want this library to rely on "initialization.py" [12/19/17]
from subprocess import check_output
import numpy as np
import pdb
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, LogNorm
from matplotlib.colorbar import ColorbarBase
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math as math
from scipy import constants as const
import ctypes
import matplotlib.dates as mdates
from read_routines import read_in_raw_data
from initializations import *

# UPDATES / NOTES:

# [4/2/20] split_bins_onto_fixed_frame
# *** Andrew created new rebinning function ***
# For the re-binning software it is a little more involved then just a subtraction 
# and a multiplicative factor.
# I kept the setup very similar to the original function but I made a new 
# split_bins_onto_fixed_frame(ff, af, orig_counts) function.
# Ff is the altitude at the top of the bin in the fixed frame
# Af is the altitude at the top of the bin in the altitude frame
# Ff_bin_numbers is the index in ff where af resides between that index and the next bin
# Af_current is the percentage of af that overlaps the bin it was placed into 
# (remember af is the top of the bin). The bin of ff[ff_bin_number]
# The remainder of the percentage would then be in the next bin. That percentage 
# is stored in af_next.
# When the af bin is a subset of the ff bin then af_current will equal 1 and 
# the next afbin is likely also assigned to that ff bin. Places stores these 
# locations to remove the need for a long for loop.
# For each bin in af the following assigns the percentage of original counts 
# to the originally assigned bin. Then the remaining percentage is assigned to 
# the next bin. This does all 833 sequentially so for each bin in places, the value 
# is likely overwritten by the following bin in af.
# new_counts_frme[:, ff_bin_numbers[a]] += (orig_counts[:, a] * af_current[a])
# new_counts_frme[:, ff_bin_numbers[a] + 1] += (orig_counts[:, a] * af_next[a])
# provided places exists, and the index is not the final bin (833, cannot be 
# overwritten by 834 because it DNE) the original value of the bin is added 
# back into the ff for each af in places.
#
# [4/22/20] bug fixed in put_bins_onto_fixed_frame 
# Correction made by Andrew Kupchock. ff_bin_numbers was not being
# properly indexed in places where > 1 af bin was in single bin.


def average_lidar_data(a, ne, nd):
    """Average lidar data (or other arrays).
       INPUTS: 
               a  -> the input np array. MUST BE 2 DIMENSIONS!
               ne -> the number of elements to average
               nd -> the dimension # to be averaged (start @0)   
               
       LAST MODIFIED: 9/26/17                           
    """

    # This code will implicitly assume that the larger dimension is supposed
    # to be the x-dimension, or first dimension. In typical lidar data,
    # this corresponds to the number of records. The user should enter an
    # nd value corresponding to the "records" dimension.
    if nd == 1:
        a = a.transpose()
    elif nd != 0:
        print('Cannot subscript array with '+str(nd))
        print('nd must either be == 1 or 0')
        return None
    
    # Compute the number of elements in the output array.
    # Then initialize the output array.
    nelems = a.shape
    nx0 = nelems[0]
    ny0 = nelems[1]
    nx = int(nx0/ne)
    print(str(nx0 % ne), 'leftover')
    ny = ny0
    avgd_a = np.empty([nx, ny])
    
    i = 0
    j = 0
    while j < ny:
        i = 0
        while i < nx:
            avgd_a[i, j] = np.mean(a[i*ne:(i+1)*ne, j])
            i += 1
        j += 1            
    
    if nd == 1:
        avgd_a = avgd_a.transpose()
    
    return avgd_a
    

def compute_solar_background(rc, bg_st_bin, bg_ed_bin):
    """This code will compute the solar background of raw counts
       given a 2D counts array and a background region. It will
       return the background as a 2D array, where values are constant
       in the altitude dimension. The altitude dimension should be
       the second dimension (index 1).
    """
    
    # NOTE: This function assumes array of profs x bins [X x Y]
    bg_reg_subset = rc[:, bg_st_bin:bg_ed_bin]
    bg = np.mean(bg_reg_subset, axis=1)
    return bg
    
    
def correct_raw_counts(rc, E, r, nr, nb, bg_st_bin, bg_ed_bin, opt):
    """This function will produce NRB from raw photon counts"""
    
    # INPUTS:
    #         - rc        = raw counts      [nr,nb]
    #         - E         = energy (Joules) [nr]
    #         - r         = range (meters)  [nb]
    #         - nr        = number of records (scalar-int)
    #         - nb        = number of bins    (scalar-int)
    #         - bg_st_bin = first bin of solar background region (scalar-int)
    #         - bg_ed_bin = last bin of solar background region (scalar-int)
    #         - opt       = string = 'bg_sub' or 'NRB' or 'NRB_no_range' 
    #
    # OUTPUTS:
    #         - NRB = NRB, corrected counts [nr,nb]

    # NOTES:
    # [5/8/18] 'NRB_no_range' opt added. Select if counts are already 
    # background-subtracted and range-corrected.
    
    # Compute the solar background
    
    bg = compute_solar_background(rc, bg_st_bin, bg_ed_bin)
    
    # Compute NRB

    # Broadcast 1D arrays to 2D so computation is vectorized (read: fast)
    if opt is 'NRB': r = np.broadcast_to(r, (nr, nb))
    E = np.broadcast_to(E, (nb, nr)).transpose()
    bg = np.broadcast_to(bg, (nb, nr)).transpose()
    
    if opt == 'NRB':
        NRB = ((rc - bg) * r**2) / E
    elif opt == 'bg_sub':
        NRB = rc - bg
    elif opt == 'NRB_no_range':  
        NRB = rc / E
    else:
        print('Invalid option entered. stopping execution in lidar.py')
        pdb.set_trace()
    
    return NRB
    

def convert_raw_energy_monitor_values(raw_EMs, nwl, instr='CAMAL', e_flg=None):
    """As of 10/18/17, this routine is only specific to CAMAL"""

    # INPUTS:
    # raw_EMs -> Raw, untouched, energy monitor values. In array form.
    #            Dimension should be something like [nchans x nwl] but
    #            this could depend on instrument. I plan on put multiple
    #            instruments' conversions in this code.
    # nwl -> The number of wavelengths.
    # instr -> String identifying the instrument.
    # e_flg -> A carry-over from CPL-IDL world which tells this code which
    #          equations should be used on CPL conversions.
    #
    # OUTPUT:
    # EMall -> array containing converted energy, in micro-joules, for all
    #          wavelengths.
    
    if instr == 'CAMAL':
    
        sz = raw_EMs.shape
        nr = sz[0]
    
        EMall = np.zeros((nr, nwl), dtype=np.int32)  # nwl is in initializations.py
    
        EMall[:, 0] = raw_EMs[:, 5].astype(np.int32)*16**0 + \
            raw_EMs[:, 6].astype(np.int32)*16**2 +     \
            raw_EMs[:, 7].astype(np.int32)*16**4 +     \
            raw_EMs[:, 8].astype(np.int32)*16**6 +     \
            raw_EMs[:, 9].astype(np.int32)*16**8
        EMall[:, 1] = raw_EMs[:, 0].astype(np.int32)*16**0 + \
            raw_EMs[:, 1].astype(np.int32)*16**2 +     \
            raw_EMs[:, 2].astype(np.int32)*16**4 +     \
            raw_EMs[:, 3].astype(np.int32)*16**6 +     \
            raw_EMs[:, 4].astype(np.int32)*16**8
        EMall[:, 2] = raw_EMs[:, 10].astype(np.int32)*16**0 + \
            raw_EMs[:, 11].astype(np.int32)*16**2 +     \
            raw_EMs[:, 12].astype(np.int32)*16**4 +     \
            raw_EMs[:, 13].astype(np.int32)*16**6 +     \
            raw_EMs[:, 14].astype(np.int32)*16**8
        
        # After this line EMall is now of dtype float
        EMall = EMall.astype(np.float32) 
        # As of 1/26/18, the following 3 lines of calculations are obsolete
        # EMall[:,0] = 8.63101e-4 + 2.88557e-10*EMall[:,0] - 1.76057e-18*EMall[:,0]**2 + 0.0*EMall[:,0]**3
        # EMall[:,1] = 0.003932537 + 4.11985e-11*EMall[:,1] - 8.82687e-21*EMall[:,1]**2 + 0.0*EMall[:,1]**3
        # EMall[:,2] = 0.012758984 + 2.45103e-10*EMall[:,2] - 1.17876e-19*EMall[:,2]**2 + 0.0*EMall[:,2]**3
        # As of 1/26/18, Andrew Kupchock says convert the energy monitors as follows
        EMall[:, 2] = 1.29089e-10*EMall[:, 2] + 0.024267116
        EMall[:, 1] = 2.71388E-11*EMall[:, 1] + 0.005726336
        EMall[:, 0] = 1.36981E-10*EMall[:, 0] + 0.00206153
        # At this point all EM values are in milli-joules. Convert to micro-joules
        # in return statement.
        EMall = EMall*1e3
    
    elif instr == 'CPL':

        # NOTES:
        # This CPL Energy Monitor conversion section was copied & translated from 
        # Mr. Dennis Hlavka's "engin_corr_v3.pro" code. I even included the "e_flg"
        # and kept the meanings the same.
        #
        # CPL's EMs -> 0=355, 1=532, 2=1064

        sz = raw_EMs.shape
        nr = sz[0]   

        if e_flg == 6:
            # TC4-07 settings . Initially valid 6/19/2006
            emon_t = raw_EMs.astype(np.float64)/500.0
            emon_c = np.copy(emon_t)
            m0_355 = 0.0
            m1_355 = 0.05
            m2_355 = 0.0
            m3_355 = 0.0
            emon_c[:, 0] = m0_355 + m1_355*emon_t[:, 0] + m2_355*emon_t[:, 0]**2 + m3_355*emon_t[:, 0]**3
            m0_532 = -0.25721
            m1_532 = 0.038868
            m2_532 = -1.533e-6
            m3_532 = 0.0
            emon_c[:, 1] = m0_532 + m1_532*emon_t[:, 1] + m2_532*emon_t[:, 1]**2 + m3_532*emon_t[:, 1]**3
            m0_1064 = -26.806
            m1_1064 = 0.19851
            m2_1064 = -1.2727e-4
            m3_1064 = 0.0
            emon_c[:, 2] = m0_1064 + m1_1064*emon_t[:, 2] + m2_1064*emon_t[:, 2]**2 + m3_1064*emon_t[:, 2]**3
        elif e_flg == 7:
            # GloPac10 settings (UAV-CPL) (initially valid 02/16/2010)
            emon_t = raw_EMs.astype(np.float64)/500.0
            emon_c = np.copy(emon_t)
            m0_355 = -2.4349
            m1_355 = 0.10332
            m2_355 = 2.5793e-04
            m3_355 = 0.0
            emon_c[:, 0] = m0_355 + m1_355*emon_t[:, 0] + m2_355*emon_t[:, 0]**2 + m3_355*emon_t[:, 0]**3
            m0_532 = 0.36922
            m1_532 = 0.013169
            m2_532 = 2.4631e-07
            m3_532 = 0.0000
            emon_c[:, 1] = m0_532 + m1_532*emon_t[:, 1] + m2_532*emon_t[:, 1]**2 + m3_532*emon_t[:, 1]**3
            m0_1064 = 5.2746
            m1_1064 = 0.047994
            m2_1064 = -7.1374e-06
            m3_1064 = 0.000
            emon_c[:, 2] = m0_1064 + m1_1064*emon_t[:, 2] + m2_1064*emon_t[:, 2]**2 + m3_1064*emon_t[:, 2]**3
        elif e_flg == 9:
            # new laser box settings (ER2-CPL) 26Feb17	  
            emon_t = raw_EMs.astype(np.float64)/500.0
            emon_c = np.copy(emon_t)
            m0_355 = -3.618073187e0
            m1_355 = 0.334612636e0
            m2_355 = -0.001398674e0
            m3_355 = 0.0
            emon_c[:, 0] = m0_355 + m1_355*emon_t[:, 0] + m2_355*emon_t[:, 0]**2 + m3_355*emon_t[:, 0]**3
            lowlocs = np.where(emon_c[:, 0] < 0.001)
            emon_c[lowlocs[0], 0] = 0.001
            highlocs = np.where(emon_c[:, 0] > 20.0)
            emon_c[highlocs[0], 0] = 20.0
            m0_532 = -0.08109323e0
            m1_532 = 0.074648283e0
            m2_532 = -0.0000079241e0
            m3_532 = 0.0
            emon_c[:, 1] = m0_532 + m1_532*emon_t[:, 1] + m2_532*emon_t[:, 1]**2 + m3_532*emon_t[:, 1]**3
            m0_1064 = 2.172313321e0
            m1_1064 = 0.063209038e0
            m2_1064 = -0.0000237511e0
            m3_1064 = 0.0
            emon_c[:, 2] = m0_1064 + m1_1064*emon_t[:, 2] + m2_1064*emon_t[:, 2]**2 + m3_1064*emon_t[:, 2]**3
        elif e_flg == 10:
            # ACT-A, C-130 energy conversion
            # (updated 12/09/2016)  EMON_C in units of micro-Joules
            emon_t = raw_EMs.astype(np.float64)/500.0
            emon_c = np.copy(emon_t)
            m0_355 = -1.491632131e0
            m1_355 = 1.05116511e-1
            m2_355 = 0.0
            m3_355 = 0.0
            emon_c[:, 0] = m0_355 + m1_355*emon_t[:, 0] + m2_355*emon_t[:, 0]**2 + m3_355*emon_t[:, 0]**3
            # if (emon_c(0) lt .001D) then emon_c[:,0]= 0.001 <- IDL equiv. for ref
            emon_c[np.argwhere(emon_c[:, 0] < 0.001), 0] = 0.001
            # if (emon_c(0) gt 20.00D) then emon_c(0)= 20.00D <- IDL equiv. for ref
            emon_c[np.argwhere(emon_c[:, 0] > 20.0), 0] = 20.0
            m0_532 = 1.45462874e-1
            m1_532 = 7.142194e-3
            m2_532 = 3.1427e-8
            m3_532 = 0.0
            emon_c[:, 1] = m0_532 + m1_532*emon_t[:, 1] + m2_532*emon_t[:, 1]**2 + m3_532*emon_t[:, 1]**3
            m0_1064 = 1.090521825e0
            m1_1064 = 4.1418361e-2
            m2_1064 = -4.2348e-7
            m3_1064 = 0.0
            emon_c[:, 2] = m0_1064 + m1_1064*emon_t[:, 2] + m2_1064*emon_t[:, 2]**2 + m3_1064*emon_t[:, 2]**3
        elif e_flg == 12:
            # ER2-CPL settings (initially valid 5/1/2018)
            # Transcribed from email sent by Andrew Kupchock on 4/25/18
            # [8/22/18] UPDATE
            # According to Andrew, this EM conversion was changed in June after more
            # tinkering in the lab. The May 1, 2018 version was never used outside of the lab
            # so I am overwriting it with the June version which is currently being used for
            # the first time during the REThinC 2018 campaign. 	    
            emon_t = raw_EMs.astype(np.float64)/250.0
            emon_c = np.copy(emon_t)
            m0_355 = -0.000166416385199
            m1_355 = 0.161304322
            m2_355 = -1.086938077
            m3_355 = 0.0
            emon_c[:, 0] = m0_355*emon_t[:, 0]**2 + m1_355*emon_t[:, 0] + m2_355
            m0_532 = 0.09809502
            m1_532 = -1.017836594
            m2_532 = 0.0
            m3_532 = 0.0
            emon_c[:, 1] = m0_532*emon_t[:, 1] + m1_532
            m0_1064 = -2.45355E-06
            m1_1064 = 0.001553344
            m2_1064 = -0.018753417
            m3_1064 = 0.973099563
            emon_c[:, 2] = m0_1064*emon_t[:, 2]**3 + m1_1064*emon_t[:, 2]**2 + m2_1064*emon_t[:, 2] + m3_1064
        else:
            print(str(e_flg)+' is an invalid e_flg value. Stopping in Energy Monitor conversion function.')
            pdb.set_trace()
        EMall = emon_c

    elif instr == 'Roscoe':
    
        sz = raw_EMs.shape
        nr = sz[0]
    
    # [0,1] -> [355 nm, 1064 nm]
    # No 532 nm for Roscoe [12/17/19]
        EMall = np.zeros((nr, nwl), dtype=np.int32)  # nwl is in initializations.py
        
        EMall[:, 0] = raw_EMs[:, 5].astype(np.int32)*16**0 + \
            raw_EMs[:, 6].astype(np.int32)*16**2 +     \
            raw_EMs[:, 7].astype(np.int32)*16**4 +     \
            raw_EMs[:, 8].astype(np.int32)*16**6 +     \
            raw_EMs[:, 9].astype(np.int32)*16**8
        EMall[:, 1] = raw_EMs[:, 10].astype(np.int32)*16**0 + \
            raw_EMs[:, 11].astype(np.int32)*16**2 +     \
            raw_EMs[:, 12].astype(np.int32)*16**4 +     \
            raw_EMs[:, 13].astype(np.int32)*16**6 +     \
            raw_EMs[:, 14].astype(np.int32)*16**8
        
        # After this line EMall is now of dtype float
        EMall = EMall.astype(np.float32)
        # EMall[:,0] = EMall[:,0] / EMall[:,0].max()
        EMall[:, 1] = EMall[:, 1] / EMall[:, 1].max()
        EMall[:, 0] = EMall[:, 1]  # 355 nm has no valid values

    else:
        
        print("Instrument name "+instr+" is not valid.")
        print("Stopping in Energy Monitor conversion function.")
        pdb.set_trace()
    
    return EMall   
        


def calculate_off_nadir_angle(phi0, phi0_azi, Y, P, R, **kwargs):
    """ Calculates the off-nadir angle from given inputs.
        Equations used are derived from matrix rotations.
        Fuselage coordinates are rotated into nadir-fixed coordinates.
        Calculation follows what is shown in {G. Cai et al., Unmanned 
        Rotorcraft Systems, Advances in Industrial Control,
        DOI 10.1007/978-0-85729-635-1_2, © Springer-Verlag London 
        Limited 2011} , {Accuracy of Vertical Air Motions from Nadir-
        Viewing Doppler Airborne Radars, Gerald M. Heymsfield, Journal 
        of Atmospheric and Oceanic Technology, December 1989} , and
        {Mapping of Airborn Doppler Radar Data, Wn-Chau Lee et al, 
        Journal of Atmospheric and Oceanic Technology, 1994}.
        !!! ALL ANGLES ARE RADIANS IN CONTEXT OF THIS FUNCTION !!!
        Y, P, and R angles are assumed to be with respect to the
        instrument coordinate system. The instrument and fuselage 
        coordinate systems are assumed to be perfectly aligned.
    """
    
    # ASSUMPTIONS:
    # - All angles (input and output) are in radians.
    # - Y, P, and R angles are with respect to the instrument coordinate
    #   system.
    # - Instrument coordinate origins: +X points perpendicular to right  
    #   of motion, +Y aligns with motion, and +Z points down.
    # - Y is defined as a rotation about the z-axis.
    # - R is defined as a rotation about the y-axis.
    # - P is defined as a rotation about the x-axis.
    # - Rotation angles (Y,P,R) are positive if they move towards a
    #   positive axis origin.
    
    # INPUTS:
    # phi0 -> The off-nadir angle built into the instrument. When the
    #         instrument coordinate system is perfectly level, this is 
    #         off-nadir angle at which the beam points. phi0 should always
    #         be positive. It's not advised that phi0 be greater than or
    #         equal to 90 degrees. If it is, the code will likely run fine;
    #         however, the accuracy of the off-nadir angle calculation
    #         cannot be guaranteed. Negative values of phi0 will also produce
    #         unexpected results.
    # phi0_azi -> The counterclockwise angle from the +X instrument axis
    #             (+X defined as perpendicular and to the right of instrument
    #             motion). Counterclockwise from +X obviously defines
    #             a positive phi0_azi.
    # Y -> Yaw, in radians
    # P -> Pitch, in radians
    # R -> Roll, in radians
    
    # OUTPUTS:
    # ONA -> off-nadir angle (in radians)
    
    # Assuming an initial nadir-pointing vector with length equal to 1,
    # compute the x, y, and z components of the intial vector after it's
    # been rotated to the fuselage-ONA (phi0).
    
    x0 = 0.0
    y0 = 0.0
    z0 = 1.0
    
    # Default point angle of beam, broken into components of instrument
    # coordinate system.
    hori_xy_disp = z0 * math.sin(phi0)
    x = hori_xy_disp * math.cos(phi0_azi)  # x comp of def. pointing ang.
    y = hori_xy_disp * math.sin(phi0_azi)  # y comp of def. pointing ang.
    z = z0 * math.cos(phi0)
    
    # Now compute the nadir-fixed coordinate system components, after
    # rotating through the 3 attitude angles, yaw (Y, about the z-ax), 
    # roll (R, about the y-ax), and pitch (P, about the x-ax), in that 
    # order.
    x1 = x*math.cos(Y)*math.cos(R) + y*(math.sin(Y)*math.cos(P)+math.sin(P)*math.sin(R)*math.cos(Y)) + \
         z*(math.sin(P)*math.sin(Y)-math.sin(R)*math.cos(Y)*math.cos(P))
    y1 = x*(-1*math.sin(Y)*math.cos(R)) + y*(math.cos(Y)*math.cos(P)-math.sin(P)*math.sin(Y)*math.sin(R)) \
         + z*(math.sin(P)*math.cos(Y)+math.cos(P)*math.sin(Y)*math.sin(R))
    z1 = x*math.sin(R) + y*(-1*math.sin(P)*math.cos(R)) + z*math.cos(P)*math.cos(R)
    
    # Now use the dot-product of the initial, nadir-pointing vector, and
    # the rotated vector, to compute the off-nadir angle.
    dotproduct = x1*x0 + y1*y0 + z1*z0
    magvect = math.sqrt(x1**2 + y1**2 + z1**2)
    magnull = math.sqrt(x0**2 + y0**2 + z0**2)
    ONA = math.acos(dotproduct/(magvect*magnull))
    # print('input phi0 is ',phi0*(180./const.pi),' degrees.')
    # print('Off-nadir angle is ',ONA*(180./const.pi),' degrees.')
    
    if 'xz_ang' in kwargs:
        # Compute the angle in the x-z plane (CAMAL purposes in mind)
        dotproduct = x1*x0 + z1*z0
        magvect = math.sqrt(x1**2 + z1**2)
        magnull = math.sqrt(x0**2 + z0**2)
        xz_ang = math.acos(dotproduct/(magvect*magnull))
        return [ONA, xz_ang]
        
    if 'xy_ang' in kwargs:
        # Compute the angle in the x-y plane (CAMAL purposes in mind)

        # The next if block prevents a div-by-zero error.
        # This occurs when R == 0.0
        if x1 == 0.0: 
            if P >= 0.0: xy_ang = const.pi/2.0  # points forward
            if P < 0.0: xy_ang = -1.0*(const.pi/2.0)  # points behind
        else:
            xy_ang = np.arctan(y1/x1)
            if x1 < 0.0:
                xy_ang = xy_ang + const.pi
            elif (x1 > 0.0) and (y1 < 0.0):
                xy_ang = xy_ang + 2.0*const.pi
            elif (x1 > 0.0) and (y1 > 0.0):
                pass

        return [ONA, xy_ang]
    
    return ONA

def laser_spot_aircraft(alt0, lat0, lon0, azi, altB, ONA, H):
    """ This code will calculate the lat/lon coordinates at the point
        where the laser beam hits a specified altitude.
    """
    
    # INPUTS:
    #
    # alt0 -> The altitude of the aircraft (m)
    # lat0 -> The nadir latitude in radians
    # lon0 -> The nadir longitude in radians
    # azi -> The azimuth angle measured in the horizontal reference
    #        counterclockwise from X origin (radians)
    # altB -> The altitude at which you'd like the lat/lon coordinates (m)
    # ONA -> The off-nadir angle (radians)
    # H -> The aircraft heading (radians)

    # OUTPUT:
    #
    # A list of [lat, lon]
    
    # DESCRIPTION & NOTES:
    #
    # This code assumes the altitude (of the aircraft) is close enough to
    # the Earth, that Earth's curvature does not have to be considered.
    # There are two basic conversions used: degrees_lat-to_km and
    # degrees_lon-to_km, the fomer is constant, the latter varies with
    # latitude (distance between lines of lon decreases as you move away
    # from the equator).
    
    # Calculate horizontal distance between point at alt0 and point at altB
    
    # Define equatorial and polar radii of Earth
    a = 6378.137e3
    b = 6356.752e3
    
    D = (alt0 - altB) * np.tan(ONA)
    
    B = azi - H
    
    dx = D * np.sin(B)
    dy = D * np.cos(B)
    
    Clat = 111.0e3               # meters per degree
    Clat = Clat * (180.0/np.pi)  # meters per radian
    
    R = math.sqrt(((a**2 * np.cos(lat0))**2 + (b**2 * np.sin(lat0))**2) /
                  ((a * np.cos(lat0))**2 + (b * np.sin(lat0))**2))
               
    A = R * np.cos(lat0)  # distance from Earth's N-S axis
    
    Clon = 2.0*np.pi*A*(1.0/(np.pi*2.0))  # meters per radian
    
    latB = lat0 + dx/Clat
    lonB = lon0 + dy/Clon
    
    # print('ONA: ',ONA*(180.0/np.pi))
    # print(lat0*(180.0/np.pi),lon0*(180.0/np.pi))
    # print(latB*(180.0/np.pi),lonB*(180.0/np.pi))
    
    if abs(lat0*(180.0/np.pi)-latB*(180.0/np.pi)) > 0.5:
        print('\n************************************\n')
        print('The laserspot latitude differs from nadir')
        print('latitude by more than half a degree!')
        print(lat0*(180.0/np.pi), latB*(180.0/np.pi))
        print('\n************************************\n')

    return [latB, lonB]
               
         
def set_fixed_bin_alt_frame(bot_alt,top_alt,bwid,nb,pdir):
    """ Define  a fixed altitude frame, into which you'd like
        to interpolate data.
        Return array of top edge altitudes of each bin.
    """
    
    if pdir == "Down":
        return np.flip(np.arange(bot_alt, top_alt, bwid, dtype=np.float32), 0)
    else:
        return np.arange(bot_alt, top_alt, bwid, dtype=np.float32)


def compute_raw_bin_altitudes(nb, pdir, z0, bsize, ONA):
    """ Compute the altitudes of each raw bin (at single record) """
    
    # INPUTS:
    # nb -> the number of bins
    # pdir -> the pointing direction, must equal "Up" or "Down"
    # z0 -> the altitude of the lidar (meters)
    # bsize -> the size of one bin in meters
    # ONA -> the off-nadir angle in radians
    
    # OUTPUTS:
    # z -> altitudes of each bin. altitudes are the top edge of the
    #      bin.
    
    cosfact = math.cos(ONA)      

    if pdir == "Up":
        z = z0 + np.arange(0, nb)*bsize*cosfact
    elif pdir == "Down":
        z = z0 - np.arange(0, nb)*bsize*cosfact
    else:
        print('Invalid option entered in compute_raw_bin_altitudes')
        pdb.set_trace()
        
    return z        


def put_bins_onto_fixed_frame(ff, af, orig_counts):
    """ "Interp" counts onto fixed frame. The method of interpolation
        used here is not well-suited for much more than lidar data.
        In fact, it's more appropriately called a bin reassignment, not
        an interpolation. Created based off conversations with Dennis.
    """
    
    # *** IMPORTANT NOTE ***
    # If code crashes within this function, specifically in the innermost
    # loop, it probably means that the actual bin alts are outside the 
    # range of the fixed bin alts (as defined in the initializations.py
    # file by you!)
    #
    # Update, 10/18/19
    # Issue has been fixed, I believe.
    # The error only occurred when the fixed frame wasn't low enough and
    # the lowest bin had more than one af bins to go into it.
    # Subtracting 1 from ff_bin_numbers fixed the issue. Thinking about
    # this, I think this was a bug. The first ff bin number started at
    # subscript #1, when really, (af[1]-af[0])/2 should really have been
    # put into subscript #0. Same thing for the edge of the array, and this
    # is where the trouble occurred.
    #
    # Update, 4/10/20
    # Correction made by Andrew Kupchock. ff_bin_numbers was not being
    # properly indexed in places where > 1 af bin was in single bin.
    # From email with Andrew:
    # " For lidar.py it is not just the addition of split_bins_onto_fixed_frame() 
    #   function but also the correction of put_bins_onto_fixed_frame() 
    #   to reference ff_bin_number(ui(places(k))) and not excluding the 
    #   ui as well as that one bin offset. "
   
    nc = orig_counts.shape[0]  # number of channels, presumably

    if af.min() < ff.min():
        print('Fixed frame does not extend far enough to')
        print('accommodate actual frame.', af.min(), ff.min())

    af_midpoints = (af[:-1] + af[1:]) / 2.0
    ff_bin_numbers = np.digitize(af_midpoints, ff) - 1
    # ff_bin_numbers holds the index of the bin altitude above the af_midpoint
    u, ui, ncounts = np.unique(ff_bin_numbers, return_index=True, return_counts=True)
    # u is the unique array, ui the index of the unique ff_bin_number values
    new_counts_frme = np.zeros((nc, ff.shape[0]), dtype=np.float32)
    new_counts_frme[:, ff_bin_numbers] = orig_counts[:, :ff_bin_numbers.shape[0]]
    # Updated above to just ff_bin_numbers without the -1 for the new counts frame assignment
    places = np.where(ncounts > 1)[0]
    if places.shape[0] > 0:
        for k in range(0, places.shape[0]):
            orig_indx = np.argwhere(ff_bin_numbers == ff_bin_numbers[ui[places[k]]])
            # Updated above to be ff_bin_numbers[ui[places[k]]] instead of ff_bin_numbers[places[k]]
            for ch in range(0, nc):
                new_counts_frme[ch, ff_bin_numbers[ui[places[k]]]] = np.mean(orig_counts[ch, orig_indx])
                # Updated above to be ff_bin_numbers[ui[places[k]]] instead of ff_bin_numbers[places[k]]
    return new_counts_frme
    
def put_bins_onto_fixed_frame_C(np_clib, ff, af, orig_counts, nc):
    """ This function performs the same task as the regular
        put_bins_onto_fixed_frame function; however this version
        uses a C function contained within a library to decrease
        run time.
    """
    
    # NOTES:
    #
    # [3/5/18]
    # ff only gets created once by the calling program. ff stands for 
    # "fixed frame" after all. Therefore I've decided to only use numpy.require
    # on it if its flags aren't set properly.
    #
    #
    
    if not ff.flags['ALIGNED'] or not ff.flags['C_CONTIGUOUS'] or ff.dtype != np.float64: 
        ff = np.require(ff, float, ['ALIGNED', 'C_CONTIGUOUS'])
    if not af.flags['ALIGNED'] or not af.flags['C_CONTIGUOUS'] or af.dtype != np.float64:
        af = np.require(af, float, ['ALIGNED', 'C_CONTIGUOUS'])
    if not orig_counts.flags['ALIGNED'] or not orig_counts.flags['C_CONTIGUOUS'] or orig_counts.dtype != np.float64:
        orig_counts = np.require(orig_counts, float, ['ALIGNED', 'C_CONTIGUOUS'])
    # mult = np.zeros(ff.shape, dtype=np.float32)
    # mult = np.require(mult,float, ['ALIGNED', 'C_CONTIGUOUS'])
    new_counts_frme = np.zeros((nc, ff.shape[0]))
    new_counts_frme = np.require(new_counts_frme, float, ['ALIGNED', 'C_CONTIGUOUS'])

    np_clib.rebin_into_fixed_frame_v2.restype = None
    np_clib.rebin_into_fixed_frame_v2.argtypes = [np.ctypeslib.ndpointer(float,
        ndim=1, flags='aligned'), np.ctypeslib.ndpointer(float, ndim=1,
        flags='aligned'), np.ctypeslib.ndpointer(float, ndim=2, flags='aligned'),
        np.ctypeslib.ndpointer(float, ndim=2, flags='aligned, writeable'),
        ctypes.POINTER(np.ctypeslib.c_intp), ctypes.POINTER(np.ctypeslib.c_intp),
        ctypes.POINTER(np.ctypeslib.c_intp), ctypes.POINTER(np.ctypeslib.c_intp),
        ctypes.POINTER(np.ctypeslib.c_intp), ctypes.POINTER(np.ctypeslib.c_intp),
        ctypes.POINTER(np.ctypeslib.c_intp), ctypes.POINTER(np.ctypeslib.c_intp)]
    np_clib.rebin_into_fixed_frame_v2(ff, af, orig_counts, new_counts_frme, 
        ff.ctypes.strides,ff.ctypes.shape, af.ctypes.strides, af.ctypes.shape,
        orig_counts.ctypes.strides, orig_counts.ctypes.shape,
        new_counts_frme.ctypes.strides, new_counts_frme.ctypes.shape)
        
    '''np_clib.rebin_into_fixed_frame.restype = None
    np_clib.rebin_into_fixed_frame.argtypes = [np.ctypeslib.ndpointer(float,
        ndim=1, flags='aligned'), np.ctypeslib.ndpointer(float, ndim=1,
        flags='aligned'), np.ctypeslib.ndpointer(float, ndim=2, flags='aligned'),
        np.ctypeslib.ndpointer(float, ndim=2, flags='aligned, writeable'),
        ctypes.POINTER(np.ctypeslib.c_intp), ctypes.POINTER(np.ctypeslib.c_intp),
        ctypes.POINTER(np.ctypeslib.c_intp), ctypes.POINTER(np.ctypeslib.c_intp),
        ctypes.POINTER(np.ctypeslib.c_intp), ctypes.POINTER(np.ctypeslib.c_intp),
        ctypes.POINTER(np.ctypeslib.c_intp), ctypes.POINTER(np.ctypeslib.c_intp),
        np.ctypeslib.ndpointer(float, ndim=1, flags='aligned')]'''
        
    ''' np_clib.rebin_into_fixed_frame(ff, af, orig_counts, new_counts_frme, 
        ff.ctypes.strides,ff.ctypes.shape, af.ctypes.strides, af.ctypes.shape,
        orig_counts.ctypes.strides, orig_counts.ctypes.shape,
        new_counts_frme.ctypes.strides, new_counts_frme.ctypes.shape,mult)'''

    return new_counts_frme


def split_bins_onto_fixed_frame(ff, af, orig_counts):
    """ "Interp" counts onto fixed frame. The method of interpolation
        used here is not well-suited for much more than lidar data.
        In fact, it's more appropriately called a re-binning, not
        an interpolation. Created based off conversations with Steve and Patrick.
    """

    # Update, 4/1/20
    # Taken from put_bins now splits percentage of counts to new bin in that region

    nc = orig_counts.shape[0]  # number of channels, presumably
    if af.min() < ff.min():
        print('Fixed frame does not extend far enough to')
        print('accomodate actual frame.', af.min(), ff.min())

    af_wid = np.fabs(af[0] - af[1])
    # Assuming all altitude frame bins are the same length
    ff_bin_numbers = np.digitize(af, ff) - 1
    af_current = np.zeros((af.shape[0]))
    af_current = np.amin([np.fabs(af - ff[ff_bin_numbers + 1]) / af_wid, np.ones((af.shape[0]), dtype=np.float32)],
                         axis=0)
    # Determine the distance between the top of the altitude frame bin and the bottom of the bin it was place into
    # Then takes that distance and divides by the length of the altitude bin to determine percentage. Force maximum
    # of 1.0
    af_next = 1 - af_current
    # af_next is the remaining percentage in the next bin ff_bin_numbers_idx+1 in ff
    # The span will never be greater than across 2 bins
    new_counts_frme = np.zeros((nc, ff.shape[0]), dtype=np.float32)

    places = np.where(af_current == 1)[0]
    # Locate where altitude bins are completely inside the fixed frame
    a = np.arange(0, af.shape[0])
    new_counts_frme[:, ff_bin_numbers[a]] += (orig_counts[:, a] * af_current[a])
    new_counts_frme[:, ff_bin_numbers[a] + 1] += (orig_counts[:, a] * af_next[a])
    # Actual reassignment of bins occurs here - correction made below for overwritten bins that overlap

    if places.shape[0] > 0:
        # make sure shape is not empty
        for k in range(0, places.shape[0]):
            if places[k] < af.shape[0]-1:
                # Avoids error with final bin trying to see if following bin overwrites it
                if ff_bin_numbers[places[k]] == ff_bin_numbers[places[k] + 1]:
                    # Determine if next altitude bin also starts in the same fixed frame bin
                    new_counts_frme[:, ff_bin_numbers[places[k]]] += (orig_counts[:, places[k]] * af_current[places[k]])
                    # adds back in previously overwritten counts in new_counts_frame

    return new_counts_frme      

    
def get_a_color_map():
    """ Define or pick a color map for use with matplotlib """
    
    rainbow = [
        (0,         0,         0),
        (0.0500,     0.02745098,    0.1000),
        (0.1300,     0.02745098,    0.2000),
        (0.1800,     0.02745098,    0.3200),
        (0.2300,     0.02745098,    0.3600),
        (0.2800,     0.02745098,    0.4000),
        (0.3300,     0.02745098,    0.4400),
        (0.3800,     0.02745098,    0.4900),
        (0.3843,     0.02745098,    0.4941),  # rev to purple by here
        (0.3500,     0.02745098,    0.4941),
        (0.3400,     0.02745098,    0.5000),
        (0.3100,     0.02745098,    0.5300),
        (0.2500,     0.02745098,    0.5700),
        (0.2000,     0.02745098,    0.6200),
        (0.1500,     0.02745098,    0.7000),
        (0.1000,     0.02745098,    0.8500),
        (0.0500,     0.02745098,    0.9400),
        (0,         0,    1.0000),  # # #
        (0,    0.0500,    1.0000),
        (0,    0.1000,    1.0000),
        (0,    0.1500,    1.0000),
        (0,    0.2000,    1.0000),
        (0,    0.2500,    1.0000),
        (0,    0.3000,    1.0000),
        (0,    0.3500,    1.0000),
        (0,    0.4000,    1.0000),
        (0,    0.4500,    1.0000),
        (0,    0.5000,    1.0000),
        (0,    0.5500,    1.0000),
        (0,    0.6000,    1.0000),
        (0,    0.6500,    1.0000),
        (0,    0.7000,    1.0000),
        (0,    0.7500,    1.0000),
        (0,    0.8000,    1.0000),
        (0,    0.8500,    1.0000),
        (0,    0.9000,    1.0000),
        (0,    0.9500,    1.0000),
        (0,    1.0000,    1.0000),
        (0,    1.0000,    1.0000),
        (0,    1.0000,    0.9500),
        (0,    1.0000,    0.9000),
        (0,    1.0000,    0.8500),
        (0,    1.0000,    0.8000),
        (0,    1.0000,    0.7500),
        (0,    1.0000,    0.7000),
        (0,    1.0000,    0.6500),
        (0,    1.0000,    0.6000),
        (0,    1.0000,    0.5500),
        (0,    1.0000,    0.5000),
        (0,    1.0000,    0.4500),
        (0,    1.0000,    0.4000),
        (0,    1.0000,    0.3500),
        (0,    1.0000,    0.3000),
        (0,    1.0000,    0.2500),
        (0,    1.0000,    0.2000),
        (0,    1.0000,    0.1500),
        (0,    1.0000,    0.1000),
        (0,    1.0000,    0.0500),
        (0,    1.0000,         0),
        (0,    1.0000,         0),
        (0.0500,    1.0000,         0),
        (0.1000,    1.0000,         0),
        (0.1500,    1.0000,         0),
        (0.2000,    1.0000,         0),
        (0.2500,    1.0000,         0),
        (0.3000,    1.0000,         0),
        (0.3500,    1.0000,         0),
        (0.4000,    1.0000,         0),
        (0.4500,    1.0000,         0),
        (0.5000,    1.0000,         0),
        (0.5500,    1.0000,         0),
        (0.6000,    1.0000,         0),
        (0.6500,    1.0000,         0),
        (0.7000,    1.0000,         0),
        (0.7500,    1.0000,         0),
        (0.8000,    1.0000,         0),
        (0.8500,    1.0000,         0),
        (0.9000,    1.0000,         0),
        (0.9500,    1.0000,         0),
        (1.0000,    1.0000,         0),
        (1.0000,    1.0000,         0),
        (1.0000,    0.9500,         0),
        (1.0000,    0.9000,         0),
        (1.0000,    0.8500,         0),
        (1.0000,    0.8000,         0),
        (1.0000,    0.7500,         0),
        (1.0000,    0.7000,         0),
        (1.0000,    0.6500,         0),
        (1.0000,    0.6000,         0),
        (1.0000,    0.5500,         0),
        (1.0000,    0.5000,         0),
        (1.0000,    0.4500,         0),
        (1.0000,    0.4000,         0),
        (1.0000,    0.3500,         0),
        (1.0000,    0.3000,         0),
        (1.0000,    0.2500,         0),
        (1.0000,    0.2000,         0),
        (1.0000,    0.1500,         0),
        (1.0000,    0.1000,         0),
        (1.0000,    0.0500,         0),
        (1.0000,         0,         0),
        (1.0000,    1.0000,    1.0000)
    ]
    
    cm = LinearSegmentedColormap.from_list('rainbow', rainbow, N=len(rainbow)*2)
    
    return cm
    
    
def curtain_plot(counts_imgarr, nb, vrZ, z, cb_min, cb_max, hori_cap, pointing_dir,
                 figW, figL, CPpad, xlab, ylab, tit, yax, yax_lims, xax,
                 xax_lims, scale_alt_OofM, mpl_flg, out_dir,
                 outfile_name='new_curtain.png', cm_choose=False):
    """ Function that will create a curtain plot
        This function was basically copied from a function called
        make_curtain_plot in the GUI_function library. The name was changed
        and some things were deleted, but that's about the only 
        differences.
    """
    
    # USE THIS FUNCTION FOR IMAGE
    # The variable counts_imgarr is local to this function, and will not
    # mess up anything outside this scope. Therefore it can be sliced and
    # diced here, without crashing anything outside this scope.
    
    # INPUTS:
    #
    # counts_imgarr -> The 2D array to image. [bins x profs]
    # nb            -> The number of bins. (scalar int/float)
    # vrZ           -> Vertical resolution in meters
    # z             -> The y-axis values, corresponding to "bins" dimension of
    #                  counts_imgarr.
    # cb_min        -> The min value for the color bar.
    # cb_max        -> The max value for the color bar.
    # hori_cap      -> # of profs can't exceed this. Done to prevent code from 
    #                  burdening computer by rendering image of a super massive
    #                  gigantic array.
    # pointing_dir  -> Is the lidar looking "Up" or "Down?"
    # figW          -> The figure width
    # figL          -> The figure length
    # CPpad         -> Parameter that controls padding around the plot (inches).
    # xlab          -> xaxis title (string)
    # ylab          -> yaxis title (string)
    # tit           -> Main title (string)
    # yax           -> String identifying type of yaxis, "alt" or "bins"
    # yax_lims      -> [Bin number of lowest alt, Bin number of greatest alt]
    # xax           -> String identifying type of xaxis, "recs" or "time"
    # xax_lims      -> [record # A, record # B]
    # scale_alt_OofM-> The order of magnitude of z's scale (10 m, 1000 m, etc)
    # mpl_flg       -> If this flag == 1, interactive MPL window appears
    # out_dir       -> Directory where image will be saved; "new_curtain.png"
    # cm_choose     -> The name of the prebuilt matplotlib color map you'd
    #                  like to use. Defaults to False, which results in
    #                  get_a_color_map() module being used to define color map.
    
    # OUTPUTS:
    # 

    # expand vertical dimension of image by using np.repeat [retired Oct 2017]
    # subsetting array by user-input bins installed [12/6/17]
    counts_imgarr = counts_imgarr[int(min(yax_lims)):int(max(yax_lims)), :]
    if xax != 'time': counts_imgarr = counts_imgarr[:, int(min(xax_lims)):int(max(xax_lims))]
    img_arr_shape = counts_imgarr.shape
    print('The shape of counts_imgarr is: ', img_arr_shape)
    
    # horizontal dimension capped at certain # of profiles (in init file)
    # doing this to save memory & CPU work when rendering image
    if img_arr_shape[1] > hori_cap:
        nthin = int(img_arr_shape[1] / hori_cap)
        counts_imgarr = counts_imgarr[:, ::nthin]
        img_arr_shape = counts_imgarr.shape
        print('The shape of counts_imgarr is: ', img_arr_shape)

    # you need to manipulate altitude array in order to properly
    # label image plot
    if pointing_dir == "Up":
        alt_imgarr = np.flipud(z)
    else:
        alt_imgarr = z
    newsize = alt_imgarr.size
    alt_ind = np.linspace(0, newsize-1, newsize,dtype='uint32')
    print('The shape of alt_ind is: ', alt_ind.shape)
    # yax_lims = [ alt_ind[newsize-1],alt_ind[0] ]
    # yax_lims = [ y1,y2 ] # y1, y2 determine zoom along vertical axis

    # Actually plot the photon counts
    fig1 = plt.figure(1, figsize=(figW, figL))
    ax = plt.gca()  # ax is now the "handle" to the figure
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(tit)
    cm = get_a_color_map()
    if cm_choose: cm = plt.get_cmap(cm_choose)
    im = ax.imshow(counts_imgarr, cmap=cm, clim=(cb_min, cb_max),
                   interpolation='nearest', aspect='auto',
                   extent=xax_lims+yax_lims)                 
                   
    # Format the x-axis   
    if xax == 'time':
        ax.xaxis_date()
        time_format = mdates.DateFormatter('%H:%M:%S')
        ax.xaxis.set_major_formatter(time_format)
        fig1.autofmt_xdate()
                   
    # Format the y-axis
    locs, labels = plt.yticks()
    if yax == 'alt':
        delta_check = vrZ
        y2_ind = int(max(yax_lims))
        if y2_ind >= nb: 
            y2 = z[nb-1]
        else:
            y2 = z[y2_ind]
        y1 = z[int(min(yax_lims))]            
        ys = [y1, y2]
        min_y = math.ceil(min(ys)/scale_alt_OofM)*scale_alt_OofM
        max_y = math.floor(max(ys)/scale_alt_OofM)*scale_alt_OofM
        ndivi = int((max_y - min_y) / scale_alt_OofM)
        ytick_lab = np.linspace(min_y, max_y, ndivi+1)
    else:
        delta_check = 2
        ndivi = 20
        ytick_lab = np.linspace(yax_lims[0], yax_lims[1], ndivi+1)
    ytick_ind = np.zeros(ytick_lab.size) + 999
    k = 0
    for e in ytick_lab:
        m = abs(e - alt_imgarr) < delta_check  # where diff is smaller than 60 meters
        if np.sum(m) == 0:  # all false
            k = k + 1
            continue
        else:
            ytick_ind[k] = alt_ind[m][0]
            k = k + 1
    actual_ticks_mask = ytick_ind != 999
    ytick_lab = ytick_lab[actual_ticks_mask]
    ytick_ind = ytick_ind[actual_ticks_mask]          	    
    plt.yticks(ytick_ind, ytick_lab)
        
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.autoscale
    plt.savefig(out_dir+outfile_name, bbox_inches='tight', pad_inches=CPpad)
    if mpl_flg == 1: plt.show()
    plt.close(fig1)
    
    return None


def discrete_curtain_plot(imgarr, cmname, nb, vrZ, z, cb_min, cb_max, ndiv, hori_cap, pointing_dir,
                          figW, figL, CPpad, xlab, ylab, cblab, tit, yax, yax_lims, xax,
                          xax_lims, scale_alt_OofM, mpl_flg, outfile_name, out_dir, custom_def=None,
                          center_ticks=True, nudge=0.0):
    """ Function that will create a curtain plot of an image using discrete levels.
    """

    # USE THIS FUNCTION FOR IMAGE
    # The variable imgarr is local to this function, and will not
    # mess up anything outside this scope. Therefore it can be sliced and
    # diced here, without crashing anything outside this scope.
    
    # INPUTS:
    #
    # imgarr -> The 2D array to image. [bins x profs]
    # cmname        -> Name of one of the predefined Matplotlib color bars (string)
    # nb            -> The number of bins. (scalar int/float)
    # vrZ           -> Vertical resolution in meters
    # z             -> The y-axis values, corresponding to "bins" dimension of
    #                  imgarr.
    # cb_min        -> The min value for the color bar. Must be an int because discrete!
    # cb_max        -> The max value for the color bar. Must be an int ^!
    # ndiv          -> Length, in data units, of color bar shall be divided into 'ndiv'
    #                  equal portions from 'cb_min' to 'cb_max' with ticks being
    #                  every N * n_div. If cb_min = 0 and cb_max = 4, set ndiv = 5 to
    #                  have 0,1,2,3 as discrete color bar values. 
    # hori_cap      -> # of profs can't exceed this. Done to prevent code from 
    #                  burdening computer by rendering image of a super massive
    #                  gigantic array.
    # pointing_dir  -> Is the lidar looking "Up" or "Down?"
    # figW          -> The figure width
    # figL          -> The figure length
    # CPpad         -> Parameter that controls padding around the plot (inches).
    # xlab          -> xaxis title (string)
    # ylab          -> yaxis title (string)
    # cblab         -> label for the color bar
    # tit           -> Main title (string)
    # yax           -> String identifying type of yaxis, "alt" or "bins"
    # yax_lims      -> [Bin number of lowest alt, Bin number of greatest alt]
    # xax           -> String identifying type of xaxis, "recs" or "time"
    # xax_lims      -> [record # A, record # B]
    # scale_alt_OofM-> The order of magnitude of z's scale (10 m, 1000 m, etc)
    # mpl_flg       -> If this flag == 1, interactive MPL window appears
    # outfile_name  -> Name of the output image file. Must be a png.
    # out_dir       -> Directory where image will be saved; "new_curtain.png"
    # custom_def    -> Set to None by default. Set to list of color triplets to impliment
    #                  a custom color map, if desired. Set cmname input variable == anything
    #                  if choosing to use this option.
    # center_ticks  -> Optional, if True, places ticks in center. If you're plotting
    #                  categorical data, you'll probably want to do this. Default is
    #                  True.
    # nudge         -> Floating point numbers can be like 4.000000062, when all you really 
    #                  want is 4. Set this to a small #, like 0.1, to ensure weird floating
    #                  point boundary issues don't mess up your color mapping, particularly
    #                  when using custom_def.
    
    # OUTPUTS:
    #
    # Returns None, but code will save a png image to outdir. 
    
    # NOTES:
    #
    # [1/7/19] As of this date only discrete levels that step by integers
    # (i.e. that correspond to categories) have been tested.
    # The code should work for any discrete numerical data type.
    #
    # [6/19/19]
    # Custom color map option added via optional "custom_def" input variable.
    # Also, fixed error in call to ax.imshow where the color bar limits were
    # explicitly specified via clim. "nudge" added.

    # expand vertical dimension of image by using np.repeat [retired Oct 2017]
    # subsetting array by user-input bins installed [12/6/17]
    imgarr = imgarr[int(min(yax_lims)):int(max(yax_lims)), :]
    if xax != 'time': imgarr = imgarr[:, int(min(xax_lims)):int(max(xax_lims))]
    img_arr_shape = imgarr.shape
    print('The shape of imgarr is: ', img_arr_shape)
    
    # horizontal dimension capped at certain # of profiles (in init file)
    # doing this to save memory & CPU work when rendering image
    if img_arr_shape[1] > hori_cap:
        nthin = int(img_arr_shape[1] / hori_cap)
        imgarr = imgarr[:, ::nthin]
        img_arr_shape = imgarr.shape
        print('The shape of imgarr is: ', img_arr_shape)

    # you need to manipulate altitude array in order to properly
    # label image plot
    if pointing_dir == "Up":
        alt_imgarr = np.flipud(z)
    else:
        alt_imgarr = z
    newsize=alt_imgarr.size
    alt_ind = np.linspace(0, newsize-1, newsize, dtype='uint32')
    print('The shape of alt_ind is: ', alt_ind.shape)
    # yax_lims = [ alt_ind[newsize-1],alt_ind[0] ]
    # yax_lims = [ y1,y2 ] # y1, y2 determine zoom along vertical axis

    # Actually plot the photon counts
    # setup the plot
    fig1, ax = plt.subplots(1, 1, figsize=(14, 8))

    # define the colormap
    if custom_def is not None:
        cmap = LinearSegmentedColormap.from_list('Custom cmap', custom_def, len(custom_def))
        bounds = np.linspace(cb_min, cb_max, ndiv)
        norm = BoundaryNorm(bounds, cmap.N)
    else:
        cmap = plt.get_cmap(cmname)
        # extract all colors from the 'cmname' map
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # force the first color entry to be grey
        cmaplist[0] = (.5, .5, .5, 1.0)
        # create the new map
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
        # define the bins and normalize
        bounds = np.linspace(cb_min, cb_max, ndiv)
        norm = BoundaryNorm(bounds, cmap.N)	

    tickpos_shift = 0.0
    if center_ticks: tickpos_shift = (bounds[1] - bounds[0])/2.0

    # make the image plot
    ax.imshow(imgarr+nudge, cmap=cmap, aspect='auto', extent=xax_lims+yax_lims, clim=(cb_min, cb_max))

    # create a second axes for the colorbar
    ax2 = fig1.add_axes([0.95, 0.1, 0.02, 0.75])
    cb = ColorbarBase(ax2, cmap=cmap, norm=norm, spacing='proportional', ticks=bounds+tickpos_shift, boundaries=bounds,
                      format='%1i')
    ax.set_title(tit)
    ax.set_xlabel(xlab, size=12)
    ax.set_ylabel(ylab, size=12)
    ax2.set_ylabel(cblab, size=12)
                   
    # Format the x-axis   
    if xax == 'time':
        ax.xaxis_date()
        time_format = mdates.DateFormatter('%H:%M:%S')
        ax.xaxis.set_major_formatter(time_format)
        fig1.autofmt_xdate()
                   
    # Format the y-axis
    locs = ax.get_yticks()
    labels = ax.get_yticklabels()
    if yax == 'alt':
        delta_check = vrZ
        y2_ind = int(max(yax_lims))
        if y2_ind >= nb: 
            y2 = z[nb-1]
        else:
            y2 = z[y2_ind]
        y1 = z[int(min(yax_lims))]            
        ys = [y1, y2]
        min_y = math.ceil(min(ys)/scale_alt_OofM)*scale_alt_OofM
        max_y = math.floor(max(ys)/scale_alt_OofM)*scale_alt_OofM
        ndivi = int((max_y - min_y) / scale_alt_OofM)
        ytick_lab = np.linspace(min_y, max_y, ndivi+1)
    else:
        delta_check = 2
        ndivi = 20
        ytick_lab = np.linspace(yax_lims[0], yax_lims[1], ndivi+1)
    ytick_ind = np.zeros(ytick_lab.size) + 999
    k = 0
    for e in ytick_lab:
        m = abs(e - alt_imgarr) < delta_check  # where diff is smaller than 60 meters
        if np.sum(m) == 0:  # all false
            k = k + 1
            continue
        else:
            ytick_ind[k] = alt_ind[m][0]
            k = k + 1
    actual_ticks_mask = ytick_ind != 999
    ytick_lab = ytick_lab[actual_ticks_mask]
    ytick_ind = ytick_ind[actual_ticks_mask]          	    
    ax.set_yticks(ytick_ind)
    ax.set_yticklabels(ytick_lab)
    
    plt.savefig(out_dir+outfile_name, bbox_inches='tight', pad_inches=CPpad)
    if mpl_flg == 1: plt.show()
    plt.close(fig1)
    
    return None
  
    
def determine_look_angles(MCS_file_list, return_edges=None):
    """ 
    This function will determine all the look angles within data.
    Writtten with CAMAL in mind because CAMAL has
    programmable, continuously variable scan angles.
    
    This method will produce a list of the float values of the
    angles. 
    """
    # INPUT
    # MCS_file_list -> A list of strings containing all the MCS input full-path
    #                  file names
    #
    # OUTPUT
    # preprogrammed_scan_angle_est -> A numpy array of the scan angles determined
    #                 by histogram analysis.
    # edges (optional) -> The edges of the bins of the histogram analysis
    
    # Use entire dataset to determine scan angles...
    first_read = True
    n_files = len(MCS_file_list)
    r = 0
    for file in MCS_file_list:
        file_strip = file.strip()
        MCS_data_1file = read_in_raw_data(file_strip)
        if MCS_data_1file is None:
            # None indications corrupt or no data within file
            continue
        if first_read:
            first_read = False
            nshots = MCS_data_1file['meta']['nshots'][0]
            tot_est_recs = int(rep_rate/nshots)*file_len_secs*n_files
            scan_pos = np.zeros(tot_est_recs)
        nr_1file = MCS_data_1file.shape[0]
        scan_pos[r:r+nr_1file] = MCS_data_1file['meta']['scan_pos']
        r += nr_1file

    # Trim any excess off array      
    scan_pos = scan_pos[0:r]

    # Correct scan angles for fixed, known offset. Provided in 
    # initialization file.
    scan_pos = scan_pos + angle_offset
    
    # Histogram analysis
    # The histogram parameters here come from the initialization file
    nhbins = int((scan_pos_uplim - scan_pos_lowlim)/scan_pos_bw)
    dens, edges = np.histogram(scan_pos,
                               bins=nhbins, range=(scan_pos_lowlim, scan_pos_uplim))
    
    # Identify the 'actual' (preprogrammed) scan angles by using the median and masks
    preprogrammed_scan_angle_est = np.zeros(edges.shape[0]-1) - 999  # *
    for e in range(0, edges.shape[0]-1):
        if dens[e] > min_scan_freq:
            low_edge_mask = scan_pos >= edges[e]
            high_edge_mask = scan_pos < edges[e+1]
            edges_mask = low_edge_mask * high_edge_mask
            preprogrammed_scan_angle_est[e] = np.median(scan_pos[edges_mask])
            
    valid_mask = preprogrammed_scan_angle_est > -999               # *
    preprogrammed_scan_angle_est = preprogrammed_scan_angle_est[valid_mask]
    print('The code estimates that the following are the preprogrammed scan angles:')
    print(preprogrammed_scan_angle_est)
    
    if return_edges is not None:
        return [preprogrammed_scan_angle_est, edges]
    else:
        return preprogrammed_scan_angle_est    
      

def apply_look_angle_mask(cnt_arr, angle_arr, cs, hist_bin_width, nprofs, nhori, opt):
    """ Code to apply mask to science data based on look angle 
    """

    # This maps the full-length dataset to cnt_arr
    cnt_arr_map = np.arange(0, nprofs).astype('uint32')
    cnt_arr_map = cnt_arr_map  # could be overwritten below
        
    # Go no farther if no angle is selected
    if opt == 'no_mask': return
    # If you make it to this line, a mask is going to be applied
    print("An angle mask is being applied.")
    
    lim1 = cs - hist_bin_width
    lim2 = cs + hist_bin_width
    print('limit1 = '+str(lim1).strip()+' limit2 = '+str(lim2).strip())
    # Make an array mask to mask "where" angle fall within bin
    # First reduce array based on averaging, ang_mask then will be
    # in "averaged space."
    scan_pos_reduced = angle_arr[::nhori]
    scan_pos_reduced = scan_pos_reduced[0:nprofs]
    ang_mask = ((scan_pos_reduced >= lim1) & (scan_pos_reduced <= lim2))
    # pdb.set_trace()
        
    if opt == 'nogaps':
            
        # Here we make a sample (cnt_arr) with no gaps
        # Remember, ang_mask will be in averaged space.
        cnt_arr = cnt_arr[:, :][cnt_arr_map[ang_mask]]
        print('The shape of cnt_arr is ', cnt_arr.shape)
        # Finalize the cnt_arr_map. The numbers within this array
        # should correspond to full-rez dataset indices.
        cnt_arr_map = cnt_arr_map[ang_mask]
            
    elif opt == 'gaps':
            
        # Here we make a sample (samp chan) with gaps
        cnt_arr[:, :][cnt_arr_map[(ang_mask == False)]] = 0
        
    # Wrap up this method, overwrite nprofs
    cnt_arr_shape = cnt_arr.shape
    print('The shape of cnt_arr is ', cnt_arr_shape)
    nprofs = cnt_arr_shape[0]
    print('Length of cnt_arr: ', cnt_arr_shape[0])
    
    return cnt_arr
    
    
def vmol_interp(inputs):
    """ Port of Steve Palm's famous 'vmol_interp.pro' which interpolates atmospheric
        profile parameters between reported pressure levels (typically taken from
        a radiosonde profile) using the hypsometric equation. The original descriptive 
        header from 'vmol_interp.pro' is pasted below.
        
        pro vmol_interp, ht,pres,tmp,rel_humidity,num_levels,tempfile,stop_raob_ht
        ; ****
        ; This subroutine uses as input height, pressure, temperature and relative humidity.
        ; The vertical resolution of the input profiles is not important. However,
        ; these quantities should be defined to at least the height 'sbin', which is the
        ; height above msl of the top most bin. If they are not defined to that height,
        ; results are unpredictable. The program uses the hypsometric formula to interpolate
        ; the input met profiles to the resolution 'vres'. It then computes the molecular backscatter
        ; at 'nwl' wavelengths. The computed profiles are returned via the common block 'acls'.
        ; The equation to compute the molecular profiles takes into account the effect of water vapor
        ; on atm density and was lifted from Measures, page 42, eqn 2.134
        ;
        ; Inputs:
        ; ht - height in km
        ; pres - pressure in mb
        ; tmp = temperature in Kelvin
        ; rel_humidity = relative humidity
        ; num_levels - the number of levels for which the input profiles are defined
        ;
        ; Coded 5/9/00, S. Palm
        ; ****
    """
    
    # NOTES:
    #
    # Changed the **4 to **4.09 to match the CATS ATBD. [7/18/19]
    
    # inputs = [ht, pres, tmp, rel_humidity, num_levels, tempfile, stop_raob_ht]
    
    # Not sure if Python has common blocks. Let's just say it doesn't.
    # common acls, nwl,nbins,sbin,vres,acls_ht,acls_press,acls_temp, $
    #    acls_rel_humidity,acls_mol_back
    
    ht, pres, tmp, rel_humidity, num_levels, tempfile, stop_raob_ht, nwl, sbin, vres = inputs
    
    lambd = np.asarray([355.0, 532.0, 1064.0])  # "lambda" is a python operator
    mbs = np.zeros(nwl)
    
    f_obj = open(tempfile, 'w')
    
    dz = vres  # height increment for acls is 30 meters
    mr = 3.479108e-3    # this is really m/r which is the molecular weight of dry air
                        # divided by the universal gas constant
    g = 9.806           # gravitational constant

    i = 1
    hlast = 0.0
    h1 = 0.0
    
    for j in range(0, num_levels):
        t1 = tmp[j]
        htop = ht[j+1]
        if htop <= stop_raob_ht: p1 = pres[j]
        rh1 = rel_humidity[j]
        esat = 0.6112 * np.exp((17.67 * (t1 - 273.16)) / (t1 - 29.66))
        qsat = 0.622 * esat /(p1 / 10.0)
        q1 = rh1 * qsat / 100.0
        virtual_tbar = t1 / (1.0 - (3.0*q1/5.0))
        # n = p1*1.0e3 / (1.3806e-16*virtual_tbar)    # version 1 using virtual temperature
        n = p1*1.0e3 / (1.3806e-16*t1)               # version 2 using actual temperature 05-08-2002
        mbs[:] = n * 5.450 * (550.0 / lambd[:])**4 * 1.0e-28 * 1.0e2  # the 1.0d2 converts from 1/cm to 1/m
        
        if j == 0:
            if rh1 > 99.9: rh1 = 99.9
            formatted_line = '{0:3d}  {1:6.5f}  {2:6.1f}  {3:5.1f}  {4:4.1f}  {5:14.7e}  {6:14.7e}  {7:14.7e}\n'
            line_str = formatted_line.format(i, ht[j], p1, t1, rh1, mbs[0], mbs[1], mbs[2])
            f_obj.write(line_str)
            i += 1
            
        hbot = ht[j]
        if j > 0: hbot = hlast
        dtdz = (tmp[j+1] - tmp[j]) / (ht[j+1] - ht[j])
        drhdz = (rel_humidity[j+1] - rel_humidity[j]) / (ht[j+1] - ht[j])
        # print(' ')
        '''print('At standard level {:3d}  {:8.3f}  {:8.3f}  '
              '{:8.3f}  {:8.3f}'.format(j+1, ht[j+1], tmp[j+1], pres[j+1], rel_humidity[j+1]))'''
        # print('******************')
        for h1 in np.arange(hbot+dz, htop+dz, dz):
            t2 = t1 + (dtdz*dz)     # temp at height h1
            rh2 = rh1 + (drhdz*dz)  # RH at height h1
            tbar = (t2 + t1) / 2.0  # average temp
            esat = 0.6112 * np.exp((17.67 * (tbar - 273.16)) / (tbar - 29.66))
            qsat = 0.622*esat/(p1/10.0)
            q2 = rh2*qsat/100.0
            virtual_tbar = tbar / (1.0 - (3.0*q2/5.0))
            p2p1 = np.exp(-mr*g/virtual_tbar*dz*1000.0)
            p2 = p2p1 * p1  # pressure at height h1
            # Ideal gas law: N=pv/(kT), where N is number of molecules
            # n = p2*1.0d3 / (1.3806d-16*virtual_tbar) ;pressure from mb to dynes per cm2 using virtual temp
            n = p2*1.0e3 / (1.3806e-16*tbar)  # pressure from mb to dynes per cm2 using actual temp 05-08-2002
            
            mbs[:] = n * 5.450 * (550.0 / lambd[:])**4.09 * 1.0e-28 * 1.0e2  # the 1.0d2 converts from 1/cm to 1/m
            
            t1 = t2
            p1 = p2
            rh1 = rh2
            hlast = h1
            if rh2 > 99.9: rh2 = 99.9
            formatted_line = '{0:3d}  {1:6.5f}  {2:6.1f}  {3:5.1f}  {4:4.1f}  {5:14.7e}  {6:14.7e}  {7:14.7e}\n'
            line_str = formatted_line.format(i, h1, p2, t2, rh2, mbs[0], mbs[1], mbs[2])
            f_obj.write(line_str)
            if h1 > sbin: break
            i += 1
        if h1 > sbin: break     # This is in outer loop. Wanted to get rid of
                                # Steve's 'goto' statement

    f_obj.close()
    # print('\nnumber of interpolated points: ',i)
    n_interp_points = i
    
    hgt = np.zeros(i)
    press = np.zeros(i)
    temp = np.zeros(i)
    rel_hum = np.zeros(i)
    mol_back = np.zeros((i, 3))
    
    with open(tempfile, 'r') as f_obj:
        lines = f_obj.readlines()
        i = 0
        for line in lines:
            linesplit = line.split()
            hgt[i] = float(linesplit[1])
            press[i] = float(linesplit[2])
            temp[i] = float(linesplit[3])
            rel_hum[i] = float(linesplit[4])
            mol_back[i, :] = np.asarray([float(e) for e in linesplit[5:8]])
            i += 1
    
    f_obj.close()
    
    return [np.flip(mol_back, axis=0), n_interp_points, np.flip(hgt, axis=0),
            np.flip(press, axis=0), np.flip(temp, axis=0), np.flip(rel_hum, axis=0)]
        

def stacked_curtains_plot(counts_imgarr, nb, vrZ, z, cb_min, cb_max, hori_cap, pointing_dir,
                          figW, figL, CPpad, xlab, ylab, tit, yax, yax_lims, xax,
                          xax_lims, scale_alt_OofM, mpl_flg, out_dir, savefname,
                          im1, im2, im3):
    """ Function intended to be used to make stacked curtain plots of
        TC4, 09 Aug 07 data.
    """
    
    # INPUTS:
    #
    interp_param = 'none'  # 'nearest'
    
    # OUTPUTS:
    # 

    from matplotlib.ticker import StrMethodFormatter

    # expand vertical dimension of image by using np.repeat [retired Oct 2017]
    # subsetting array by user-input bins installed [12/6/17]
    counts_imgarr = counts_imgarr[int(min(yax_lims)):int(max(yax_lims)), :]
    if xax != 'time': counts_imgarr = counts_imgarr[:, int(min(xax_lims)):int(max(xax_lims))]
    img_arr_shape = counts_imgarr.shape
    print('The shape of counts_imgarr is: ', img_arr_shape)
    
    # horizontal dimension capped at certain # of profiles (in init file)
    # doing this to save memory & CPU work when rendering image
    if img_arr_shape[1] > hori_cap:
        nthin = int(img_arr_shape[1] / hori_cap)
        counts_imgarr = counts_imgarr[:, ::nthin]
        img_arr_shape = counts_imgarr.shape
        print('The shape of counts_imgarr is: ', img_arr_shape)

    # you need to manipulate altitude array in order to properly
    # label image plot
    if pointing_dir == "Up":
        alt_imgarr = np.flipud(z)
    else:
        alt_imgarr = z
    newsize=alt_imgarr.size
    alt_ind = np.linspace(0, newsize-1, newsize, dtype='uint32')
    print('The shape of alt_ind is: ', alt_ind.shape)
    # yax_lims = [alt_ind[newsize-1], alt_ind[0]]
    # yax_lims = [y1, y2] # y1, y2 determine zoom along vertical axis

    # Actually plot the photon counts
    fig1 = plt.figure(1, figsize=(figW, figL))
    plt.subplot(411)
    plt.title(tit)
    plt.ylabel(ylab)
    cm = get_a_color_map()
    im = plt.imshow(counts_imgarr, cmap=cm, clim=(cb_min, cb_max),
                    interpolation=interp_param, aspect='auto',
                    extent=xax_lims+yax_lims)
    ax = plt.gca()                 
                   
    # Format the x-axis   
    if xax == 'time':
        ax.xaxis_date()
        time_format = mdates.DateFormatter('%H:%M:%S')
        ax.xaxis.set_major_formatter(time_format)
        fig1.autofmt_xdate()
                   
    # Format the y-axis
    locs, labels = plt.yticks()
    if yax == 'alt':
        delta_check = vrZ
        y2_ind = int(max(yax_lims))
        if y2_ind >= nb: 
            y2 = z[nb-1]
        else:
            y2 = z[y2_ind]
        y1 = z[int(min(yax_lims))]            
        ys = [y1, y2]
        min_y = math.ceil(min(ys)/scale_alt_OofM)*scale_alt_OofM
        max_y = math.floor(max(ys)/scale_alt_OofM)*scale_alt_OofM
        ndivi = int((max_y - min_y) / scale_alt_OofM)
        ytick_lab = np.linspace(min_y, max_y, ndivi+1)
    else:
        delta_check = 2
        ndivi = 20
        ytick_lab = np.linspace(yax_lims[0], yax_lims[1], ndivi+1)
    ytick_ind = np.zeros(ytick_lab.size) + 999
    k = 0
    for e in ytick_lab:
        m = abs(e - alt_imgarr) < delta_check  # where diff is smaller than 60 meters
        if np.sum(m) == 0:  # all false
            k = k + 1
            continue
        else:
            ytick_ind[k] = alt_ind[m][0]
            k = k + 1
    actual_ticks_mask = ytick_ind != 999
    ytick_lab = ytick_lab[actual_ticks_mask].astype(np.uint32)
    ytick_ind = ytick_ind[actual_ticks_mask]          	    
    plt.yticks(ytick_ind, ytick_lab)
        
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.autoscale
    ax.tick_params(labelbottom='off')
    
    # Now that you've plotted the curtain, make other subplots...
    
    plt.subplot(412, sharex=ax, sharey=ax)
    # Averaged counts ...
    plt.imshow(im1, cmap=cm, clim=(cb_min, cb_max),
               interpolation=interp_param, aspect='auto',
               extent=xax_lims+yax_lims)
    ax1 = plt.gca()
    plt.ylabel(ylab)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="1%", pad=0.05)
    cax.set_axis_off()    
    ax1.tick_params(labelbottom='off') 
 
    # Sub-sampled counts ...
    plt.subplot(413, sharex=ax, sharey=ax)
    plt.imshow(im2, cmap=cm, clim=(cb_min, cb_max),
               interpolation=interp_param, aspect='auto',
               extent=xax_lims+yax_lims)
    ax2 = plt.gca()
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="1%", pad=0.05)
    cax.set_axis_off()
    
    # Averaged sub-sampled counts ...
    plt.subplot(414, sharex=ax, sharey=ax)
    plt.imshow(im3, cmap=cm, clim=(cb_min, cb_max),
               interpolation=interp_param, aspect='auto',
               extent=xax_lims+yax_lims)
    ax3 = plt.gca()
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="1%", pad=0.05)
    cax.set_axis_off()
    plt.tight_layout()
   
    plt.savefig(out_dir+savefname, bbox_inches='tight', pad_inches=CPpad)
    
    if mpl_flg == 1: plt.show()
    plt.close(fig1)   
    return None        


def custom_curtain_plot(counts_imgarr, nb, vrZ, z, cb_min, cb_max, hori_cap, pointing_dir,
                        figW, figL, CPpad, xlab, ylab, tit, yax, yax_lims, xax,
                        xax_lims, scale_alt_OofM, mpl_flg, out_dir,
                        outfile_name='new_curtain.png', cm_choose=False):
    """ CUSTOM FUNCTION. DON'T USE UNLESS YOU CREATED THIS!
    """
    
    # USE THIS FUNCTION FOR IMAGE
    # The variable counts_imgarr is local to this function, and will not
    # mess up anything outside this scope. Therefore it can be sliced and
    # diced here, without crashing anything outside this scope.
    
    # INPUTS:
    #
    # counts_imgarr -> The 2D array to image. [bins x profs]
    # nb            -> The number of bins. (scalar int/float)
    # vrZ           -> Vertical resolution in meters
    # z             -> The y-axis values, corresponding to "bins" dimension of
    #                  counts_imgarr.
    # cb_min        -> The min value for the color bar.
    # cb_max        -> The max value for the color bar.
    # hori_cap      -> # of profs can't exceed this. Done to prevent code from 
    #                  burdening computer by rendering image of a super massive
    #                  gigantic array.
    # pointing_dir  -> Is the lidar looking "Up" or "Down?"
    # figW          -> The figure width
    # figL          -> The figure length
    # CPpad         -> Parameter that controls padding around the plot (inches).
    # xlab          -> xaxis title (string)
    # ylab          -> yaxis title (string)
    # tit           -> Main title (string)
    # yax           -> String identifying type of yaxis, "alt" or "bins"
    # yax_lims      -> [Bin number of lowest alt, Bin number of greatest alt]
    # xax           -> String identifying type of xaxis, "recs" or "time"
    # xax_lims      -> [record # A, record # B]
    # scale_alt_OofM-> The order of magnitude of z's scale (10 m, 1000 m, etc)
    # mpl_flg       -> If this flag == 1, interactive MPL window appears
    # out_dir       -> Directory where image will be saved; "new_curtain.png"
    # cm_choose     -> The name of the prebuilt matplotlib color map you'd
    #                  like to use. Defaults to False, which results in
    #                  get_a_color_map() module being used to define color map.
    
    # OUTPUTS:
    # 

    # expand vertical dimension of image by using np.repeat [retired Oct 2017]
    # subsetting array by user-input bins installed [12/6/17]
    counts_imgarr = counts_imgarr[int(min(yax_lims)):int(max(yax_lims)), :]
    if xax != 'time': counts_imgarr = counts_imgarr[:, int(min(xax_lims)):int(max(xax_lims))]
    img_arr_shape = counts_imgarr.shape
    print('The shape of counts_imgarr is: ', img_arr_shape)
    
    # horizontal dimension capped at certain # of profiles (in init file)
    # doing this to save memory & CPU work when rendering image
    if img_arr_shape[1] > hori_cap:
        nthin = int(img_arr_shape[1] / hori_cap)
        counts_imgarr = counts_imgarr[:, ::nthin]
        img_arr_shape = counts_imgarr.shape
        print('The shape of counts_imgarr is: ', img_arr_shape)

    # you need to manipulate altitude array in order to properly
    # label image plot
    if pointing_dir == "Up":
        alt_imgarr = np.flipud(z)
    else:
        alt_imgarr = z
    newsize=alt_imgarr.size
    alt_ind = np.linspace(0, newsize-1, newsize, dtype='uint32')
    print('The shape of alt_ind is: ', alt_ind.shape)
    # yax_lims = [ alt_ind[newsize-1],alt_ind[0] ]
    # yax_lims = [ y1,y2 ] # y1, y2 determine zoom along vertical axis

    # Actually plot the photon counts
    fig1 = plt.figure(1, figsize=(figW, figL))
    ax = plt.gca()  # ax is now the "handle" to the figure
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(tit)
    cm = get_a_color_map()
    if cm_choose: cm = plt.get_cmap(cm_choose)
    im = ax.imshow(counts_imgarr, cmap=cm, clim=(cb_min, cb_max),
                   interpolation='nearest', aspect='auto',
                   extent=xax_lims+yax_lims, norm=LogNorm(vmin=cb_min, vmax=cb_max))                 
                   
    # Format the x-axis   
    if xax == 'time':
        ax.xaxis_date()
        time_format = mdates.DateFormatter('%H:%M:%S')
        ax.xaxis.set_major_formatter(time_format)
        fig1.autofmt_xdate()
                   
    # Format the y-axis
    locs, labels = plt.yticks()
    if yax == 'alt':
        delta_check = vrZ
        y2_ind = int(max(yax_lims))
        if y2_ind >= nb: 
            y2 = z[nb-1]
        else:
            y2 = z[y2_ind]
        y1 = z[int(min(yax_lims))]            
        ys = [y1, y2]
        min_y = math.ceil(min(ys)/scale_alt_OofM)*scale_alt_OofM
        max_y = math.floor(max(ys)/scale_alt_OofM)*scale_alt_OofM
        ndivi = int((max_y - min_y) / scale_alt_OofM)
        ytick_lab = np.linspace(min_y, max_y, ndivi+1)
    else:
        delta_check = 2
        ndivi = 20
        ytick_lab = np.linspace(yax_lims[0], yax_lims[1], ndivi+1)
    ytick_ind = np.zeros(ytick_lab.size) + 999
    k=0
    for e in ytick_lab:
        m = abs(e - alt_imgarr) < delta_check  # where diff is smaller than 60 meters
        if np.sum(m) == 0:  # all false
            k = k + 1
            continue
        else:
            ytick_ind[k] = alt_ind[m][0]
            k = k + 1
    actual_ticks_mask = ytick_ind != 999
    ytick_lab = ytick_lab[actual_ticks_mask]
    ytick_ind = ytick_ind[actual_ticks_mask]          	    
    plt.yticks(ytick_ind, ytick_lab)
        
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.autoscale
    plt.savefig(out_dir+outfile_name, bbox_inches='tight', pad_inches=CPpad)
    if mpl_flg == 1: plt.show()
    plt.close(fig1)
    
    return None
