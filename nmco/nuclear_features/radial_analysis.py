# -*- coding: utf-8 -*-

# Import modules
import numpy as np
import pandas as pd
import scipy.ndimage as ndi

def measure_radial_chromatin(regionmask: np.ndarray, intensity: np.ndarray):
    m1=regionmask*1
    
    m2_3=ndi.binary_erosion(m1,structure=np.ones((3,3)))
    m2_5=ndi.binary_erosion(m1,structure=np.ones((5,5)))
    m2_10=ndi.binary_erosion(m1,structure=np.ones((10,10)))
    
    m3_3=m1-m2_3
    m3_5=m1-m2_5
    m3_10=m1-m2_10
    
    feat = {
        "peripheral_chromatin_3": np.sum(m3_3*intensity)/np.sum(m1*intensity),
        "peripheral_chromatin_5": np.sum(m3_5*intensity)/np.sum(m1*intensity),
        "peripheral_chromatin_10": np.sum(m3_10*intensity)/np.sum(m1*intensity),
        }
        
    feat = pd.DataFrame([feat])

    return feat
