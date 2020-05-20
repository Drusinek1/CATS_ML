import numpy as np
import datetime as DT


# Define the CCSDS header

CCSDS_struct = np.dtype([ ('APID',np.int16), ('SeqCnt',np.int16) ,
                          ('BytCnt',np.int16), ('UnixT',np.int32) ,
                          ('SubSec',np.int16)                           ])
                          # UnixT is in seconds
                          # SubSec is in milliseconds


# Define structure of one CAMAL MCS record

MCS_meta_struct = np.dtype([ ('CCSDS',CCSDS_struct), ('nchans',np.int32) ,
                          ('nbins',np.int32), ('nshots',np.int32) ,
                          ('binwid',np.float32), ('scan_state',np.int32) ,
                          ('scan_pos',np.float32), ('GpsWeek',np.uint32) ,
                          ('Gps_msec',np.uint32), ('lat',np.float32) ,
                          ('lon',np.float32), ('GpsAlt', np.float32) ,
                          ('sync_byt1',np.uint8), ('sync_byt2',np.uint8) ,
                          ('status',np.uint8,(4,)), ('EM',np.uint8,(15,)) ])

                          
# Define structure of one CAMAL housekeeping (hk) record

hk_struct = np.dtype([ ('CCSDS',CCSDS_struct), ('GoodCmdCnt',np.int32) ,
                    ('BadCmdCnt',np.int32), ('MCS24_status',np.int8) ,
                    ('MSC24_state',np.int32), ('MCS24_npc',np.int32) ,
                    ('CtrlState',np.int32), ('CtrlSubmode',np.int32) ,
                    ('ScanState',np.int32), ('ScanAngle',np.float32) ,
                    ('ScanTemp',np.float32,(16,)), ('dis_inp',np.int8) ,
                    ('dis_outp',np.int8), ('ADVolt',np.float32,(32,)) ,
                    ('GpsWeek',np.uint32), ('Gps_msec',np.uint32) ,
                    ('lat',np.float32), ('lon',np.float32) ,
                    ('GpsAlt', np.float32), ('GPS_bytesread',np.uint32) , 
                    ('NASDAT_flags', np.uint8), ('IWG1_pkt_rec',np.uint32) ,
                    ('flags', np.uint8)                                 ])
      
      
# Define structure of one CAMAL GPS record (not IWG1)
                    
gps_struct = np.dtype([ ('GpsWeek',np.float32), ('Gps_sec',np.float32) ,
                     ('lat',np.float32), ('lon',np.float32) ,
                     ('alt',np.float32), ('north',np.float32) ,
                     ('east',np.float32), ('vert',np.float32) ,
                     ('pitch',np.float32), ('roll',np.float32) ,
                     ('yaw',np.float32)                                 ])       
                     

# Define structure of one IWG1 record

IWG1_struct = np.dtype([ ('UTC',DT.datetime), ('lat',np.float32),
                         ('lon',np.float32), ('GPS_alt_msl',np.float32), 
                         ('GPS_alt',np.float32), ('press_alt',np.float32), 
                         ('rad_alt',np.float32), ('ground_spd',np.float32),
                         ('true_airspd',np.float32), ('ind_airspd',np.float32),
                         ('mach_num',np.float32), ('vert_spd',np.float32),
                         ('heading',np.float32), ('track',np.float32),
                         ('drift',np.float32), ('pitch',np.float32),
                         ('roll',np.float32), ('slip',np.float32),
                         ('attack',np.float32), ('S_air_temp',np.float32),
                         ('dewp_temp',np.float32), ('T_air_temp',np.float32),
                         ('static_p',np.float32), ('dynmc_p',np.float32), 
                         ('cabin_p',np.float32), ('wind_spd',np.float32),
                         ('wind_dir',np.float32), ('vert_wind_spd',np.float32),
                         ('sol_zen',np.float32), ('air_sun_elev',np.float32),
                         ('so_azi',np.float32), ('air_sun_azi',np.float32)
                                                                        ])
                                                                        
# Define structure of one "nav" record. These are simply IWG1 records 
# captured by CAMAL. The only difference between this structure and the
# "IWG1_struct" is that "UnixT" is tacked on front.

nav_struct = np.dtype([ ('UnixT',DT.datetime), ('UTC',DT.datetime), ('lat',np.float32),
                         ('lon',np.float32), ('GPS_alt_msl',np.float32), 
                         ('GPS_alt',np.float32), ('press_alt',np.float32), 
                         ('rad_alt',np.float32), ('ground_spd',np.float32),
                         ('true_airspd',np.float32), ('ind_airspd',np.float32),
                         ('mach_num',np.float32), ('vert_spd',np.float32),
                         ('heading',np.float32), ('track',np.float32),
                         ('drift',np.float32), ('pitch',np.float32),
                         ('roll',np.float32), ('slip',np.float32),
                         ('attack',np.float32), ('S_air_temp',np.float32),
                         ('dewp_temp',np.float32), ('T_air_temp',np.float32),
                         ('static_p',np.float32), ('dynmc_p',np.float32), 
                         ('cabin_p',np.float32), ('wind_spd',np.float32),
                         ('wind_dir',np.float32), ('vert_wind_spd',np.float32),
                         ('sol_zen',np.float32), ('air_sun_elev',np.float32),
                         ('so_azi',np.float32), ('air_sun_azi',np.float32)
                                                                        ])  
                                                                        
# Define structure of one "Nav_save" record. This was created specifically
# for the numpy array that saves the nav data as the code loops through 
# profiles. This is exactly the same as the IWG1_struct, except that the 'UTC'
# field is a string instead of a datetime object. It would have been very
# difficult for IDL or another programming language to interpret a python
# datetime object. No objects appear in nav_save_struct.

nav_save_struct = np.dtype([ ('UTC',np.uint8,(26,)), ('lat',np.float32),
                             ('lon',np.float32), ('GPS_alt_msl',np.float32), 
                             ('GPS_alt',np.float32), ('press_alt',np.float32), 
                             ('rad_alt',np.float32), ('ground_spd',np.float32),
                             ('true_airspd',np.float32), ('ind_airspd',np.float32),
                             ('mach_num',np.float32), ('vert_spd',np.float32),
                             ('heading',np.float32), ('track',np.float32),
                             ('drift',np.float32), ('pitch',np.float32),
                             ('roll',np.float32), ('slip',np.float32),
                             ('attack',np.float32), ('S_air_temp',np.float32),
                             ('dewp_temp',np.float32), ('T_air_temp',np.float32),
                             ('static_p',np.float32), ('dynmc_p',np.float32), 
                             ('cabin_p',np.float32), ('wind_spd',np.float32),
                             ('wind_dir',np.float32), ('vert_wind_spd',np.float32),
                             ('sol_zen',np.float32), ('air_sun_elev',np.float32),
                             ('so_azi',np.float32), ('air_sun_azi',np.float32)
                                                                            ])
									
# Define the structure of one raw CPL record ("CLS").
# Just as with CAMAL, allow # channels, # bins to be flexible (not hard-coded).
# CPL record is so long, define sub-structures.

CLS_raw_header_struct = np.dtype([ ('RecordNumber',np.int32), ('ExactTime',np.uint8,(27,)), 
                            ('NumChannels',np.int8), ('Resolution',np.uint8),
                            ('DetectorSequence',np.uint8,(9,)), ('NumberT_Probes',np.uint8),
                            ('NumberV_Probes',np.uint8), ('Reserved',np.uint8,(20,))
                                                                                  ])

CLS_raw_engineering_struct = np.dtype([ ('LaserEnergyMonitors',np.int32,(3,)), ('TemperatureMeasurements',np.int32,(12,)),
                                 ('VoltageMeasurements',np.int32,(3,)), ('Pressure',np.int32), 
                                 ('Reserved',np.int32,(16,))                                                  ])

# Translation of byte-by-byte structure specified by Bill Hart's "rawheadengnav" for IDL...
# Retired 4/4/18
#CLS_raw_nav_struct = np.dtype([ ('Header',np.uint8,(2,)), ('UTC_Time',np.uint8,(13,)),
#                         ('Lat',np.uint8,(10,)), ('Lon',np.uint8,(11,)),
#                         ('TrueHeading',np.uint8,(7,)), ('PitchAngle',np.uint8,(9,)),
#                         ('RollAngle',np.uint8,(9,)), ('GroundSpeed',np.uint8,(7,)),
#                         ('TrackAngleTrue',np.uint8,(7,)), ('InertialWindSpeed',np.uint8,(5,)),
#                         ('InertialWindDirection',np.uint8,(6,)), ('BodyLongitAccl',np.uint8,(7,)),
#                         ('BodyLateralAccl',np.uint8,(7,)), ('BodyNormalAccl',np.uint8,(7,)),
#                         ('TrackAngleRate',np.uint8,(6,)), ('PitchRate',np.uint8,(6,)), 
#                         ('RollRate',np.uint8,(6,)), ('InertialVerticalSpeed',np.uint8,(7,)), 
#                         ('GPS_Altitude',np.uint8,(8,)), ('GPS_Latitude',np.uint8,(10,)), 
#                         ('GPS_Longitude',np.uint8,(11,)), ('StaticPressure',np.uint8,(9,)), 
#                         ('TotalPressure',np.uint8,(9)), ('DifferentialPressure',np.uint8,(7,)), 
#                         ('TotalTemperature',np.uint8,(7,)), ('StaticTemperature',np.uint8,(7,)),
#                         ('BarometricAltitude',np.uint8,(8,)), ('MachNo',np.uint8,(6,)),
#                         ('TrueAirSpeed',np.uint8,(7,)), ('WindSpeed',np.uint8,(5,)),
#                         ('WindDirection',np.uint8,(6+1,)), ('SunElevation',np.uint8,(7,)),
#                         ('SunAzimuth',np.uint8,(7,)), ('Pad',np.uint8,(10-1,))
#                                                                           ])

			     
CLS_decoded_header_struct = np.dtype([ ('RecordNumber',np.int32), ('ExactTime',DT.datetime), 
                            ('NumChannels',np.int8), ('Resolution',np.uint8),
                            ('DetectorSequence',np.uint8,(9,)), ('NumberT_Probes',np.uint8),
                            ('NumberV_Probes',np.uint8), ('Reserved',np.uint8,(20,))
                                                                                  ])
										  

# Define one CPL, CLS record, but this time, have those character bytes decoded into a useful format.										  
										  
CLS_decoded_nav_struct = np.dtype([ ('UTC_Time',DT.datetime),
                         ('Lat',np.float32), ('Lon',np.float32),
                         ('TrueHeading',np.float32), ('PitchAngle',np.float32),
                         ('RollAngle',np.float32), ('GroundSpeed',np.float32),
                         ('TrackAngleTrue',np.float32), ('InertialWindSpeed',np.float32),
                         ('InertialWindDirection',np.float32), ('BodyLongitAccl',np.float32),
                         ('BodyLateralAccl',np.float32), ('BodyNormalAccl',np.float32),
                         ('TrackAngleRate',np.float32), ('PitchRate',np.float32), 
                         ('RollRate',np.float32), ('InertialVerticalSpeed',np.float32), 
                         ('GPS_Altitude',np.float32), ('GPS_Latitude',np.float32), 
                         ('GPS_Longitude',np.float32), ('StaticPressure',np.float32), 
                         ('TotalPressure',np.float32), ('DifferentialPressure',np.float32), 
                         ('TotalTemperature',np.float32), ('StaticTemperature',np.float32),
                         ('BarometricAltitude',np.float32), ('MachNo',np.float32),
                         ('TrueAirSpeed',np.float32), ('WindSpeed',np.float32),
                         ('WindDirection',np.float32), ('SunElevation',np.float32),
                         ('SunAzimuth',np.float32)                              ])
									   
CLS_decoded_meta_struct = np.dtype([ ('Header',CLS_decoded_header_struct), ('Engineering',CLS_raw_engineering_struct),
                             ('Nav',CLS_decoded_nav_struct)                                   ])   									   

                                                                
# Definition of a structure to hold raob data
                                                                
raob_struct = np.dtype([ ('UTC',DT.datetime), ('lat',np.float32), ('lon',np.float32),
                  ('ltp',np.int8), ('pres',np.int32), ('alt',np.int32),
                  ('temp',np.int32), ('dewp',np.int32), ('wdir',np.int32),
                  ('wspd',np.int32)     
                                                                       ])

# Define structure of one Roscoe MCS record

MCS_meta_struct_r = np.dtype([ ('CCSDS',CCSDS_struct), ('nchans',np.int32) ,
                            ('nbins',np.int32), ('nshots',np.int32) ,
                            ('binwid',np.float32), ('voltages',np.int32,(32,)) ,
                            ('sync_byt1',np.uint8), ('sync_byt2',np.uint8) ,
                            ('status',np.uint8,(4,)), ('EM',np.uint8,(15,)) ])

                          
# Define structure of one Roscoe housekeeping (hk) record

hk_struct_r = np.dtype([ ('CCSDS',CCSDS_struct), ('GoodCmdCnt',np.int32) ,
                      ('BadCmdCnt',np.int32), ('MCS24_status',np.int8) ,
                      ('MSC24_state',np.int32), ('MCS24_npc',np.int32,(8,)) ,
                      ('CtrlState',np.int32), ('CtrlSubmode',np.int32) ,
                      ('dis_inp',np.int8) , ('dis_outp',np.int8), 
                      ('ADVolt',np.float32,(32,)) , ('NASDAT_flags', np.uint8),
                      ('IWG1_pkt_rec',np.uint32) ,('flags', np.uint8)                                
                                                                         ])
                        
                          
