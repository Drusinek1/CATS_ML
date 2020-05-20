# Define data structures that can change

# [3/27/20] CATS (ISS) data structure added

import numpy as np

from immutable_data_structs import * #data structs that don't change


def define_MSC_structure(nchans,nbins):
    """ This function will define the final MCS structure once the number
        of channels and number of bins are determined.
        
        INPUTS:
        nchans -> the number of channels
        nbins -> the number of bins	    
    """
    
    MCS_struct = np.dtype([ ('meta',MCS_meta_struct), 
                            ('counts',np.uint16,(nchans,nbins)) ])
    
    return MCS_struct
    

def define_MSC_structure_r(nchans,nbins):
    """ This function will define the final MCS structure for Roscoe
        insturment data once the number of channels and number of bins
        are determined.
        
        INPUTS:
        nchans -> the number of channels
        nbins -> the number of bins	    
    """
    
    MCS_struct_r = np.dtype([ ('meta',MCS_meta_struct_r), 
                            ('counts',np.uint16,(nchans,nbins)) ])
    
    return MCS_struct_r


 
def define_CLS_structure(nchans,nbins,CLS_meta_struct):
    """ This function will define the CLS structure once the number of
        channels and number of bins are determined.
	
        INPUTS:
        nchans -> the number of channels
        nbins -> the number of bins
	
	OUTPUT:
	A Python Numpy data structure for raw CPL "CLS" data.
    """
    
    CLS_struct = np.dtype([ ('meta',CLS_meta_struct), 
                            ('counts',np.uint16,(nchans,nbins)),
			    ('Reserved',np.uint8,(44,))      ])

    return CLS_struct
    
    
def define_CLS_decoded_structure(nchans,nbins):
    """ This function will define the CLS structure once the number of
        channels and number of bins are determined.
	
        INPUTS:
        nchans -> the number of channels
        nbins -> the number of bins
	
	OUTPUT:
	A Python Numpy data structure for CPL "CLS" data, which character
	strings converted into usable data types.
    """
    
    CLS_struct = np.dtype([ ('meta',CLS_decoded_meta_struct), 
                            ('counts',np.uint16,(nchans,nbins)),
			    ('Reserved',np.uint8,(44,))      ])

    return CLS_struct
    
    

def define_CLS_meta_struct(nbytes):
    """ This function creates the meta struct for a raw CLS file once
        CLS_raw_nav_struct is defined.
    """ 

    CLS_raw_nav_struct = np.dtype([ ('Full_rec', np.uint8, (nbytes,) ) ])

    CLS_meta_struct = np.dtype([ ('Header',CLS_raw_header_struct), ('Engineering',CLS_raw_engineering_struct),
                             ('Nav',CLS_raw_nav_struct)                                             ])    
    
    return CLS_meta_struct


def define_CATS_L0_struct(nchans, nbins):
    """ This function returns the structure of a CATS L0 file. The only
        2 things that can change are the number of channels (nchans) and
        the number of bins.
    """
    
    cats_l0_structure = np.dtype([ ('RN',(np.uint32)) , ('CM',(np.uint32)) ,
                   ('delay',(np.uint32)) , ('FT',(np.uint16)) ,
				   ('pad1',(np.uint16)) , ('CT1',(np.uint32)) ,
				   ('RDMmin',(np.uint8)) , ('RDMmaj',(np.uint8)) ,
				   ('pad2',(np.uint16)) , ('DFVR',(np.uint16)) ,
				   ('NRB',(np.uint16)) , ('DFHR',(np.uint16)) ,
				   ('NC',(np.uint16)) , ('XCTRSpos',(np.float32)) ,
				   ('YCTRSpos',(np.float32)) , ('ZCTRSpos',(np.float32)) ,
				   ('XCTRSvel',(np.float32)) , ('YCTRSvel',(np.float32)) ,
				   ('ZCTRSvel',(np.float32)) , ('QW',(np.float32)) ,
				   ('QX',(np.float32)) , ('QY',(np.float32)) ,
				   ('QZ',(np.float32)) , ('PQD',(np.uint16)) ,
				   ('pad3',(np.uint16)) , ('CT2',(np.uint32)) ,
				   ('CEC1',(np.float32)) , ('CEG1',(np.float32)) ,
				   ('CEC2',(np.float32)) , ('CEG2',(np.float32)) ,
				   ('CEC3',(np.float32)) , ('CEG3',(np.float32)) ,
				   ('AEC1',(np.float32)) , ('AEG1',(np.float32)) ,
				   ('AEC2',(np.float32)) , ('AEG2',(np.float32)) ,
				   ('AEC3',(np.float32)) , ('AEG3',(np.float32)) ,
				   ('E1a',(np.uint8)) , ('E1v',(np.uint8)) ,
				   ('E2a',(np.uint8)) , ('E2v',(np.uint8)) ,
				   ('E3a',(np.uint8)) , ('E3v',(np.uint8)) ,
				   ('E4a',(np.uint8)) , ('E4v',(np.uint8)) ,
				   ('BSM0',(np.uint32)) , ('BSM1',(np.uint32)) ,
				   ('BSM2',(np.uint32)) , ('BSM3',(np.uint32)) ,
				   ('BSM4',(np.uint32)) , ('BSM5',(np.uint32)) ,
				   ('BSM6',(np.uint32)) , ('reserved',(np.void,92)) ,
				   ('chan',np.uint16,(nchans,nbins)) ]           )
                   
    return cats_l0_structure
