"""
AGN spectral model generation functions
Copyright: Adam Hill (2020)
"""

import numpy as np
import pandas as pd


def generate_spectra(angle1, angle2, logNH):
    """
    This is a place holder for a proper function
    Args:
        angle1: float, inclination angle in degrees (0-90) of the AGN view
        angle2: float, torus opening angle in degrees (0-90) of the AGN
        logNH: float, logarithm of the obscuring column density within the AGN environment

    Returns:
        dataframe: a dataframe of with columns for the energy in keV, the transmitted X-ray flux,
                   the reprocessed X-ray flux, and the total X-ray flux
    """
    _degs_to_rads = lambda x: np.pi * x / 180.0
    degrees = np.arange(1, 1001, 1)
    radians = np.array(list(map(_degs_to_rads, degrees)))
    linear_component = radians * (logNH / 9.657) + 2
    transmitted_flux = (angle1 / 5) * np.cos(
        _degs_to_rads(angle1) + radians * (logNH / 1.5)
    ) + linear_component
    reprocessed_flux = (angle2 / 10) * np.sin(
        _degs_to_rads(angle2) + radians * (logNH / 5.0)
    ) + 5.0
    total_flux = transmitted_flux + reprocessed_flux

    spectral_df = pd.DataFrame(
        {
            "Energy": degrees,
            "Transmitted": transmitted_flux,
            "Reprocessed": reprocessed_flux,
            "Summed": total_flux,
        }
    )
    return spectral_df
