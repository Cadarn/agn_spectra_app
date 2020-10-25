import numpy as np
import pandas as pd

def generate_spectra(angle1, angle2, logNH):
    """"This is a place holder for a proper function"""
    _degs_to_rads = lambda x: np.pi*x/180.
    degrees = np.arange(1, 1001, 1)
    radians = np.array(list(map(_degs_to_rads, degrees)))
    linear = radians*(logNH/9.657) + 2
    ts1 = (angle1/5) * np.cos(_degs_to_rads(angle1) + radians*(logNH/1.5)) + linear
    ts2 = (angle2/10) * np.sin(_degs_to_rads(angle2) + radians*(logNH/5.)) + 5.
    combined = ts1 + ts2

    df = pd.DataFrame({"Energy": degrees,
                       "Transmitted": ts1,
                       "Reprocessed": ts2,
                       "Summed": combined})
    return df
