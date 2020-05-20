import numpy as np
import datetime as DT
import pdb


def is_leap(yr):
    """
    Simple function to determine if input year is a leap year
    according to the Gregorian calendar. Enter a year as an interger.
    Example: 2017
    Returns a 1 if it is a leap year, 0 if not.
    """
    
    lp = 0 # leap year? 1=yes, 0=no
    
    if yr % 4 == 0:
        if ((yr % 100 == 0) and (yr % 400 == 0)):
            lp = 1
        elif ((yr % 100 == 0) and (yr % 400 != 0)):
            lp = 0
        else:
            lp = 1
    
    return(lp)


def GPSWeek2UTC(GPSW,GPSms):
    """Function to convert GPS week into UTC"""
    
    # !!!!!!!!!!! READ !!!!!!!!!!!!!!!!!!!
    # Since I found a code online that did the exact same thing,
    # I've ceased work on this code as of 11/16/17. I may want to complete
    # this code at some point in the future, because its conversion does
    # not rely on fancy Python libraries like the one I found online. 
    
    # Initialize some variables
    
    DinY = 365
    ldpmon = np.zeros((2,12))
    ldpmon[0,:] = np.array([0,31,59,90,120,151,181,212,243,273,304,334])  #normal
    ldpmon[1,:] = np.array([0,31,60,91,121,152,182,213,244,274,305,335])  #leap
    months = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
    leap = 0 # will change to 1 if leap year determined
    
    # The anchor to perform conversion
    # Converted on https://www.labsat.co.uk/index/php/en/gps-time-calculator
    
    anchorY = 2017
    anchorW = 1951 # = 28 May 2017, 00:00:00 UTC
    anchorDoY = 148 # May 28's day of year (JDay, as it were)
    anchorD2E = DinY - anchorDoY
    leaps_since_anchorY = 0
    
    # Compute total # of days since beginning of anchor year
    
    mindays = (GPSW - anchorW) * 7.0
    ndays = mindays + int( GPSms/(1e3*86400.0) )
    # The # of input secs minus the number of secs in the integer # of days
    GPSsecleft = (GPSms/1e3) - float(int( GPSms/(1e3*86400.0) ))*86400.0
    ndays_since_anchorY = ndays+(anchorDoY-1)
    
    # Compute the total number of years since beginning of anchor year
    
    years = int(ndays_since_anchorY/DinY)
    year = int(years + anchorY) #just make sure it's int for next step
    
    # Compute the number of leap years since the anchor year
    
    if years >= 1:
        print("More than 1 year since anchor year")
        for testyear in range(anchorY,year):
            testleap = is_leap(testyear)
            leaps_since_anchorY += testleap
    
    # Determine if current year is a leap year
    
    leap = is_leap(year)
    
    # Use the remaining days to figure out the month
    
    # This DoY, or "day of year" starts at zero. Zero being the first day.
    DoY = ndays_since_anchorY - (years*DinY + leaps_since_anchorY) 
    if DoY > 0:
        month_sel_mask = ldpmon[leap,:] <= DoY
        month = max(months[month_sel_mask])
    else:
        print("Retroactively correcting for intervening leap years.")
        year = year - 1
        leap = is_leap(year)
        DoY = (DinY+leap) + DoY
        pdb.set_trace()
    
    print(month,leaps_since_anchorY)
    return(None)
    

def weeksecondstoutc(gpsweek,gpsseconds,leapseconds):
    """ I, Patrick Selmer, did not write this. I found it on Github at the
        following URL:
        https://gist.github.com/jeremiahajohnson/eca97484db88bcf6b124
    """
    
    import datetime, calendar
    datetimeformat = "%Y-%m-%d %H:%M:%S"
    epoch = datetime.datetime.strptime("1980-01-06 00:00:00",datetimeformat)
    elapsed = datetime.timedelta(days=(gpsweek*7),seconds=(gpsseconds+leapseconds))
    #print(type(epoch+elapsed))
    #return datetime.datetime.strftime(epoch + elapsed,datetimeformat)
    return(epoch+elapsed)


def delta_datetime_vector(dt_vect):
    """ This function will take a numpy 1D array of datetime.datetime objects
        and return a 1D float64 array of the seconds between records.
        Returned array will have same size as input array; therefore first
        element of returned array will always be equal to zero.
    """
    
    if len(dt_vect.shape) > 1:
        print("Array must be 1D!")
        print("Input array has dimensions (shape) of ",len(dt_vect))
        pdb.set_trace()

    nr = dt_vect.shape[0]
    del_t = np.zeros(nr,dtype=DT.datetime)
    del_t[1:] = dt_vect[1:] - dt_vect[:-1]
    del_t[0] = del_t[1]*0.0
    
    del_secs = np.zeros(nr,dtype=np.float64)
    for k in range(0,nr): del_secs[k] = del_t[k].total_seconds()
    
    return del_secs
    
# Commented out code below is for testing of above conversion algorithms 
    
#ans = weeksecondstoutc(1811,164196.732,16) ## --> '2014-09-22 21:36:52'
#ans = weeksecondstoutc(2086,259200.0,16) ## --> '2020-01-01 00:00:00'
#print(ans)
#print(type(ans))

#something = GPSWeek2UTC(2086,259200000) # 01 Jan 2020, 00:00:00 UTC
#something = GPSWeek2UTC(1968,248000000)
#something = GPSWeek2UTC(1982,248000000)
    
    
    
