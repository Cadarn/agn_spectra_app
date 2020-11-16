"""
AGN spectral model generation functions
Copyright: Adam Hill (2020)
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator

def merge_dict_dfs(d, common_column):
    """
    Main purpose:
        - merges all the dataframes collected in the
          d dictionary
        - Prints the duplicates in the merged dataframe and removes them
        
        NOTE: Each table must have the common_column to match on
    """
    d_copy = d.copy()
    merged = d_copy[0]
    if 0 in d_copy:
        del d_copy[0]
    else:
        print("No 0 dataframe found... This shouldn't have happened.")
    
    with tqdm(total = len(d_copy), position = 0, desc = "Merging tables") as pbar:
        for name, df in d_copy.items():
#             print(name)
#             print(merged.shape)
            merged = pd.merge(merged, df, how = "left", on = "E_keV")
            pbar.update(1)
    print(merged.shape)
    dupe_mask = merged.duplicated(subset = ["E_keV"], keep = "last")
    dupes = merged[dupe_mask]
    print(dupes.columns)
    print(str(len(dupes)) + " duplicates")
    print("Now removing duplicates...")
    merged = merged[~dupe_mask]
    
    for c in merged.columns:
        print(c)
    return merged

def sed(PhoIndex, Ecut, logNHtor, CFtor, thInc, A_Fe, z, factor):
    """
    Need to manually stitch together the spectra
    """
    PhoIndex_str = np.array(["p3=%.5f" %par_val for par_val in PhoIndex.ravel()])
    Ecut_str = np.array(["p4=%.5f" %par_val for par_val in Ecut.ravel()])
    logNHtor_str = np.array(["p5=%.5f" %par_val for par_val in logNHtor.ravel()])
    CFtor_str = np.array(["p6=%.5f" %par_val for par_val in CFtor.ravel()])
    thInc_str = np.array(["p7=%.5f" %par_val for par_val in thInc.ravel()])
    A_Fe_str = np.array(["p8=%.5f" %par_val for par_val in A_Fe.ravel()])
    z_str = np.array(["p9=%.5f" %par_val for par_val in z.ravel()])
    factor_str = np.array(["p17=%.5f" %par_val for par_val in factor.ravel()])
    trans = np.empty(shape = [len(PhoIndex_str, len(Ecut_str), len(logNHtor_str), len(CFtor_str), len(thInc_str), len(A_Fe_str), len(z_str), len(factor_str)), 40500])
    repro = np.empty(shape = [len(PhoIndex_str, len(Ecut_str), len(logNHtor_str), len(CFtor_str), len(thInc_str), len(A_Fe_str), len(z_str), len(factor_str)), 40500])
    scatt = np.empty(shape = [len(PhoIndex_str, len(Ecut_str), len(logNHtor_str), len(CFtor_str), len(thInc_str), len(A_Fe_str), len(z_str), len(factor_str)), 40500])

    ## note must stitch together in increasing order of length of parameter arrays
    for i, PhoIndex_val in enumerate(PhoIndex_str):
        for j, Ecut_val in enumerate(Ecut_str):
            for k, A_Fe_val in enumerate(A_Fe_str):
                for l, factor_val in enumerate(factor_str):
                    for m, z_val in enumerate(z_str):
                        for n, logNHtor_val in enumerate(logNHtor_str):
                            for o, CFtor_val in enumerate(CFtor_str):
                                for p, thInc_val in enumerate(thInc_str):
                                    df_column = "%(PhoIndex_val)s_%(Ecut_val)s_%(logNHtor_val)s_%(CFtor_val)s_%(thInc_val)s_%(A_Fe_val)s_%(z_val)s_%(factor_val)s" %locals()
                                    trans[i, j, k, l, m, n, o, p, :] = df_master["TRANS_" + df_column].values
                                    repro[i, j, k, l, m, n, o, p, :] = df_master["REPR_" + df_column].values
                                    scatt[i, j, k, l, m, n, o, p, :] = df_master["SCATT_" + df_column].values
    return trans, repro, scatt, temp["E_keV"].values



## load in the hefty dataset
## note -- need to figure out a more efficient way of storing this data
## Xspec uses a fits table, but unsure how we can generate the Python RegularGridInterpolator from that

df_dict = {}
for a, csvfile in enumerate(glob.glob("./borus_stepped/borus_step*.csv")):
    df_dict[a] = pd.read_csv(csvfile)

df_master = merge_dict_dfs(df_dict, "E_keV")
 
parLen = 5
PhoIndex = np.linspace(1.45, 2.55, 3)
Ecut = np.logspace(2., 3., 3)
logNHtor = np.linspace(22., 25.5, parLen)
CFtor = np.linspace(0.15, 0.95, parLen)
thInc = np.linspace(20., 85, parLen)
A_Fe = np.logspace(-1., 1., 3)
z = np.logspace(-3., 0., 4)
factor = np.logspace(-5., -1., 3)

params = np.meshgrid(PhoIndex, Ecut, logNHtor, CFtor, thInc, A_Fe, z, factor, indexing='ij', sparse=True)

trans_sed, repro_sed, scatt_sed, E_keV = sed(*params)
print(np.shape(SEDs))


trans_interp = RegularGridInterpolator((PhoIndex, Ecut, A_Fe, factor, z, logNHtor, CFtor, thInc), trans_sed)
repro_interp = RegularGridInterpolator((PhoIndex, Ecut, A_Fe, factor, z, logNHtor, CFtor, thInc), repro_sed)
scatt_interp = RegularGridInterpolator((PhoIndex, Ecut, A_Fe, factor, z, logNHtor, CFtor, thInc), scatt_sed)


def generate_spectra(PhoIndex, Ecut, A_Fe, factor, z, logNHtor, CFtor, thInc):
    """
    This is a place holder for a proper function
    Args:
        PhoIndex: float, powerlaw slope of the intrinsic spectrum (1.45--2.55)
        Ecut: float, high-energy exponentional cut-off of the intrinsic powerlaw (100.--1000.)
        A_Fe: float, abundance of iron in the obscurer (0.1--10.)
        factor: float, percentage of scattered emission in the warm mirror (1.e-5--1.e-1)
        z: float, redshift of the source (1.e-3--1.)
        logNHtor: float, logarithm of the column density of the obscurer (22.--22.5)
        CFtor: float, covering factor of the obscurer (0.15--0.95)
        thInc: float, inclination angle of the obscurer (20.--85.), note: edge-on = 90.
    Returns:
        dataframe: a dataframe with columns for the energy in keV, the transmitted X-ray flux,
                   the reprocessed X-ray flux, the Thomson-scattered X-ray flux, and the total X-ray flux
    """

    spectral_df = pd.DataFrame(
        {
            "Energy": E_keV,
            "Transmitted": trans_interp([PhoIndex, Ecut, A_Fe, factor, z, logNHtor, CFtor, thInc]),
            "Reprocessed": trans_interp([PhoIndex, Ecut, A_Fe, factor, z, logNHtor, CFtor, thInc]),
            "Scattered": scatt_interp([PhoIndex, Ecut, A_Fe, factor, z, logNHtor, CFtor, thInc]),
        }
    )
    spectral_df.loc[:, "Total"] = spectral_df[["Transmitted", "Reprocessed", "Scattered"]].sum()
    return spectral_df



# def generate_spectra(angle1, angle2, logNH):
#     """
#     This is a place holder for a proper function
#     Args:
#         angle1: float, inclination angle in degrees (0-90) of the AGN view
#         angle2: float, torus opening angle in degrees (0-90) of the AGN
#         logNH: float, logarithm of the obscuring column density within the AGN environment

#     Returns:
#         dataframe: a dataframe of with columns for the energy in keV, the transmitted X-ray flux,
#                    the reprocessed X-ray flux, and the total X-ray flux
#     """
#     _degs_to_rads = lambda x: np.pi * x / 180.0
#     degrees = np.arange(1, 1001, 1)
#     radians = np.array(list(map(_degs_to_rads, degrees)))
#     linear_component = radians * (logNH / 9.657) + 2
#     transmitted_flux = (angle1 / 5) * np.cos(
#         _degs_to_rads(angle1) + radians * (logNH / 1.5)
#     ) + linear_component
#     reprocessed_flux = (angle2 / 10) * np.sin(
#         _degs_to_rads(angle2) + radians * (logNH / 5.0)
#     ) + 5.0
#     total_flux = transmitted_flux + reprocessed_flux

#     spectral_df = pd.DataFrame(
#         {
#             "Energy": degrees,
#             "Transmitted": transmitted_flux,
#             "Reprocessed": reprocessed_flux,
#             "Summed": total_flux,
#         }
#     )
#     return spectral_df
