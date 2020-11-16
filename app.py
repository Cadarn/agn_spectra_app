"""
Streamlit app to host AGN spectral model generation
Copyright: Adam Hill (2020)
"""

import plotly.express as px
import streamlit as st

import pandas as pd
import numpy as np
import glob, os

"""
TODO:
- select different models
- dynamic geometry
- worth pursuing interpolator? (probably not -- just requires user to have PyXspec)
- plotly tick marks
- plotly background colour
- streamlit location of sections on screen (i.e. margins etc)
- alternative to PyXspec?
- how to do a step plot
"""
# ## NOTE adding this here instead of in separate module so that it is loaded once - is this correct?

# # from data_functions import generate_spectra
# from tqdm import tqdm
# from scipy.interpolate import RegularGridInterpolator

# def merge_dict_dfs(d, common_column = "E_keV"):
#     """
#     Main purpose:
#         - merges all the dataframes collected in the
#           d dictionary
#         - Prints the duplicates in the merged dataframe and removes them
        
#         NOTE: Each table must have the common_column to match on
#     """
#     d_copy = d.copy()
#     merged = d_copy[0]
#     if 0 in d_copy:
#         del d_copy[0]
#     else:
#         print("No 0 dataframe found... This shouldn't have happened.")
    
#     with tqdm(total = len(d_copy), position = 0, desc = "Merging spectra into one table") as pbar:
#         for name, df in d_copy.items():
# #             print(name)
# #             print(merged.shape)
#             merged = pd.merge(merged, df, how = "left", on = "E_keV")
#             pbar.update(1)
#     print(merged.shape)
#     dupe_mask = merged.duplicated(subset = ["E_keV"], keep = "last")
#     dupes = merged[dupe_mask]
#     print(dupes.columns)
#     print(str(len(dupes)) + " duplicates")
#     print("Now removing duplicates...")
#     merged = merged[~dupe_mask]
    
#     # for c in merged.columns:
#     #     print(c)
#     return merged

# def sed(PhoIndex, Ecut, A_Fe, factor, z, logNHtor, CFtor, thInc):
#     """
#     Need to manually stitch together the spectra

#     NOTE order is very important! In order of increasing grid lengths
#     """
#     PhoIndex_str = np.array(["p3=%.5f" %par_val for par_val in PhoIndex.ravel()])
#     Ecut_str = np.array(["p4=%.5f" %par_val for par_val in Ecut.ravel()])
#     A_Fe_str = np.array(["p8=%.5f" %par_val for par_val in A_Fe.ravel()])
#     factor_str = np.array(["p17=%.5f" %par_val for par_val in factor.ravel()])
#     z_str = np.array(["p9=%.5f" %par_val for par_val in z.ravel()])
#     logNHtor_str = np.array(["p5=%.5f" %par_val for par_val in logNHtor.ravel()])
#     CFtor_str = np.array(["p6=%.5f" %par_val for par_val in CFtor.ravel()])
#     thInc_str = np.array(["p7=%.5f" %par_val for par_val in thInc.ravel()])
    
#     trans = np.empty(shape = [len(PhoIndex_str), len(Ecut_str), len(A_Fe_str), len(factor_str), len(z_str), len(logNHtor_str), len(CFtor_str), len(thInc_str), 1000])
#     repro = np.empty(shape = [len(PhoIndex_str), len(Ecut_str), len(A_Fe_str), len(factor_str), len(z_str), len(logNHtor_str), len(CFtor_str), len(thInc_str), 1000])
#     scatt = np.empty(shape = [len(PhoIndex_str), len(Ecut_str), len(A_Fe_str), len(factor_str), len(z_str), len(logNHtor_str), len(CFtor_str), len(thInc_str), 1000])

#     ## note must stitch together in increasing order of length of parameter arrays
#     for i, PhoIndex_val in enumerate(PhoIndex_str):
#         for j, Ecut_val in enumerate(Ecut_str):
#             for k, A_Fe_val in enumerate(A_Fe_str):
#                 for l, factor_val in enumerate(factor_str):
#                     for m, z_val in enumerate(z_str):
#                         for n, logNHtor_val in enumerate(logNHtor_str):
#                             for o, CFtor_val in enumerate(CFtor_str):
#                                 for p, thInc_val in enumerate(thInc_str):
#                                     df_column = "%(PhoIndex_val)s_%(Ecut_val)s_%(logNHtor_val)s_%(CFtor_val)s_%(thInc_val)s_%(A_Fe_val)s_%(z_val)s_%(factor_val)s" %locals()
#                                     trans[i, j, k, l, m, n, o, p, :] = df_master["TRANS_" + df_column].values
#                                     repro[i, j, k, l, m, n, o, p, :] = df_master["REPR_" + df_column].values
#                                     scatt[i, j, k, l, m, n, o, p, :] = df_master["SCATT_" + df_column].values
#     return trans, repro, scatt



# ## load in the hefty dataset
# ## note -- need to figure out a more efficient way of storing this data
# ## Xspec uses a fits table, but unsure how we can generate the Python RegularGridInterpolator from that

# df_dict = {}
# spectra_available = sorted(glob.glob("./sim_xspec/borus_stepped/borus_step*.csv"))
# with tqdm(total = len(spectra_available), position = 0, desc = "Loading raw spectra") as pbar:
#     for a, csvfile in enumerate(spectra_available):
#         df_dict[a] = pd.read_csv(csvfile)
#         pbar.update(1)

# E_keV = df_dict[0]["E_keV"]
# df_master = merge_dict_dfs(df_dict, "E_keV")

# # df_master.to_csv("./sim_xspec/borus_all.csv", index = False)

# # df_master = pd.read_csv("./sim_xspec/borus_all.csv")

# parLen = 5
# PhoIndex_grid = np.linspace(1.45, 2.55, 3)
# Ecut_grid = np.logspace(2., 3., 3)
# A_Fe_grid = np.logspace(-1., 1., 3)
# factor_grid = np.logspace(-5., -1., 3)
# z_grid = np.logspace(-3., 0., 4)
# logNHtor_grid = np.linspace(22., 25.5, parLen)
# CFtor_grid = np.linspace(0.15, 0.95, parLen)
# thInc_grid = np.linspace(20., 85, parLen)



# params = np.meshgrid(PhoIndex_grid, Ecut_grid, A_Fe_grid, factor_grid, z_grid, logNHtor_grid, CFtor_grid, thInc_grid, indexing='ij', sparse=True)

# trans_sed, repro_sed, scatt_sed = sed(*params)
# print(np.shape(trans_sed), np.shape(repro_sed), np.shape(scatt_sed))


# trans_interp = RegularGridInterpolator((PhoIndex_grid, Ecut_grid, A_Fe_grid, factor_grid, z_grid, logNHtor_grid, CFtor_grid, thInc_grid), trans_sed)
# repro_interp = RegularGridInterpolator((PhoIndex_grid, Ecut_grid, A_Fe_grid, factor_grid, z_grid, logNHtor_grid, CFtor_grid, thInc_grid), repro_sed)
# scatt_interp = RegularGridInterpolator((PhoIndex_grid, Ecut_grid, A_Fe_grid, factor_grid, z_grid, logNHtor_grid, CFtor_grid, thInc_grid), scatt_sed)


# def generate_spectra(PhoIndex, Ecut, A_Fe, factor, z, logNHtor, CFtor, thInc):
#     """
#     This is a place holder for a proper function
#     Args:
#         PhoIndex: float, powerlaw slope of the intrinsic spectrum (1.45--2.55)
#         Ecut: float, high-energy exponentional cut-off of the intrinsic powerlaw (100.--1000.)
#         A_Fe: float, abundance of iron in the obscurer (0.1--10.)
#         factor: float, percentage of scattered emission in the warm mirror (1.e-5--1.e-1)
#         z: float, redshift of the source (1.e-3--1.)
#         logNHtor: float, logarithm of the column density of the obscurer (22.--22.5)
#         CFtor: float, covering factor of the obscurer (0.15--0.95)
#         thInc: float, inclination angle of the obscurer (20.--85.), note: edge-on = 90.
#     Returns:
#         dataframe: a dataframe with columns for the energy in keV, the transmitted X-ray flux,
#                    the reprocessed X-ray flux, the Thomson-scattered X-ray flux, and the total X-ray flux
#     """

#     print("HERE ARE THE INPUTS!")
#     print(PhoIndex, Ecut, A_Fe, factor, z, logNHtor, CFtor, thInc)
#     spectral_df = pd.DataFrame(
#         {
#             "Energy": E_keV,
#             "Transmitted": trans_interp([PhoIndex, Ecut, A_Fe, factor, z, logNHtor, CFtor, thInc]),
#             "Reprocessed": trans_interp([PhoIndex, Ecut, A_Fe, factor, z, logNHtor, CFtor, thInc]),
#             "Scattered": scatt_interp([PhoIndex, Ecut, A_Fe, factor, z, logNHtor, CFtor, thInc]),
#         }
#     )
#     spectral_df.loc[:, "Total"] = spectral_df[["Transmitted", "Reprocessed", "Scattered"]].sum()
#     return spectral_df

## trying something else as the above was too slow for now
from xspec import *
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

Fit.statMethod = 'cstat'
Xset.abund = "Wilm"
Xset.parallel.error = 4
Plot.xLog = True
Plot.yLog = True
Plot.add = True
Plot.xAxis = "keV"
Xset.cosmo = "67.3,,0.685"
Plot.device = "/null"
Fit.query = "yes"

def dummy():
    AllData.dummyrsp(lowE = 1.e-3, highE = 2.e2, nBins = 10 ** 3, scaleType = "log")
    Plot("model")
      
def generate_borus02_spectra(PhoIndex, Ecut, A_Fe, factor, logNHtor, CFtor, thInc):##, z
    
    mainModel = "constant*TBabs(atable{/Users/pboorman/Dropbox/data/xspec/models/borus/var_Fe/borus02_v170323b.fits} + zTBabs*cabs*cutoffpl + constant*cutoffpl)"
    m = Model(mainModel)
    dummy()

    ## set default values
    AllModels(1)(1).values = 1., -1.
    AllModels(1)(2).values = 0.01, -1.
    AllModels(1)(3).values = 1.9 ## VARIABLE
    AllModels(1)(4).values = 300. ## VARIABLE
    AllModels(1)(5).values = 23. ## VARIABLE
    AllModels(1)(6).values = 0.5 ## VARIABLE
    AllModels(1)(7).values = 60. ## VARIABLE
    AllModels(1)(8).values = 1. ## VARIABLE
    AllModels(1)(9).values = 0. ## redshift (VARIABLE)
    AllModels(1)(10).values = 1. ## REPR norm
    AllModels(1)(11).link = "10.^(p5 - 22.)"
    AllModels(1)(12).link = "p9"
    AllModels(1)(13).link = "p11"
    AllModels(1)(14).link = "p3"
    AllModels(1)(15).link = "p4"
    AllModels(1)(16).values = 1. ## TRANS norm
    AllModels(1)(17).values = 1.e-3 ## VARIABLE
    AllModels(1)(18).link = "p3"
    AllModels(1)(19).link = "p4"
    AllModels(1)(20).values = 1. ## SCATT norm
    AllModels.show()

    wdata = pd.DataFrame(data = {"E_keV": Plot.x()})
    
    AllModels(1)(3).values = PhoIndex
    AllModels(1)(4).values = Ecut
    AllModels(1)(5).values = logNHtor
    AllModels(1)(6).values = CFtor
    AllModels(1)(7).values = thInc
    AllModels(1)(8).values = A_Fe
    # AllModels(1)(9).values = z
    AllModels(1)(17).values = factor

    # Plot("eemodel")
    # pardf_name = "Total"
    # wdata.loc[:, pardf_name] = Plot.model()
    
    REPR_norm = AllModels(1)(10).values[0]
    TRANS_norm = AllModels(1)(16).values[0]
    SCATT_norm = AllModels(1)(20).values[0]

    AllModels(1)(10).values = 0.
    AllModels(1)(20).values = 0.
    Plot("eemodel")
    pardf_name = "Transmitted"
    wdata.loc[:, pardf_name] = Plot.model()
    AllModels(1)(10).values = REPR_norm
    AllModels(1)(20).values = SCATT_norm
    
    AllModels(1)(16).values = 0.
    AllModels(1)(20).values = 0.
    Plot("eemodel")
    pardf_name = "Reprocessed"
    wdata.loc[:, pardf_name] = Plot.model()
    AllModels(1)(16).values = TRANS_norm
    AllModels(1)(20).values = SCATT_norm
    
    AllModels(1)(10).values = 0.
    AllModels(1)(16).values = 0.
    Plot("eemodel")
    pardf_name = "Scattered"
    wdata.loc[:, pardf_name] = Plot.model()

    wdata.loc[:, "Total"] = wdata[["Transmitted", "Reprocessed", "Scattered"]].sum(axis = 1)
    ## only needed if we were going to do something else with the model...
    # AllModels(1)(10).values = REPR_norm
    # AllModels(1)(16).values = TRANS_norm
    return wdata

def generate_pexrav_spectra(PhoIndex, Ecut, A_Fe, factor, logNHtor, CFtor, thInc):##, z
    model  constant*TBabs(zTBabs*cabs*cutoffpl + pexrav + constant*cutoffpl)
              1         -1          0          0      1e+10      1e+10
           0.01         -1          0          0     100000      1e+06
              1      0.001          0          0     100000      1e+06
              0      -0.01     -0.999     -0.999         10         10
= p3
            1.9       0.01         -3         -2          9         10
            300         -1       0.01          1        500        500
          1e-05       0.01          0          0      1e+20      1e+24
= p6
= p7
           -0.1       0.01      -1000      -1000     -0.001     -0.001
              0      -0.01     -0.999     -0.999         10         10
              1      -0.01          0          0      1e+06      1e+06
              1      -0.01          0          0      1e+06      1e+06
           0.45      -0.01       0.05       0.05       0.95       0.95
= p8
          0.001       0.01      1e-05      1e-05        0.1        0.1
= p9
= p10
= p8
bayes off

    mainModel = "constant*TBabs(zTBabs*cabs*cutoffpl + pexrav + constant*cutoffpl)"
    m = Model(mainModel)
    dummy()
    AllModels(1)(1).values = 1., -1.
    AllModels(1)(2).values = 0.01, -1.
    AllModels(1)(3).values = 1. ## nHe22
    AllModels(1)(4).values = 0. ## redshift
    AllModels(1)(5).link = "p3"
    AllModels(1)(6).values = 1.9 ## VARIABLE
    AllModels(1)(7).values = 300. ## VARIABLE
    AllModels(1)(8).values = 1. ## TRANS norm
    AllModels(1)(9).values = 0. ## redshift
    AllModels(1)(10).values = 1. ## REPR norm
    AllModels(1)(11).link = "10.^(p5 - 22.)"
    AllModels(1)(12).link = "p9"
    AllModels(1)(13).link = "p11"
    AllModels(1)(14).link = "p3"
    AllModels(1)(15).link = "p4"
    AllModels(1)(16).values = 1. ## TRANS norm
    AllModels(1)(17).values = 1.e-3 ## VARIABLE
    AllModels(1)(18).link = "p3"
    AllModels(1)(19).link = "p4"
    AllModels(1)(20).values = 1. ## SCATT norm


    set_default_values(False)
    wdata = pd.DataFrame(data = {"E_keV": Plot.x()})
    
    AllModels(1)(6).values = PhoIndex
    AllModels(1)(7).values = Ecut
    AllModels(1)(3).values = 10 ** (logNHtor - 22.)
    AllModels(1)(6).values = CFtor
    AllModels(1)(7).values = thInc
    AllModels(1)(8).values = A_Fe
    # AllModels(1)(9).values = z
    AllModels(1)(17).values = factor

    # Plot("eemodel")
    # pardf_name = "Total"
    # wdata.loc[:, pardf_name] = Plot.model()
    
    REPR_norm = AllModels(1)(10).values[0]
    TRANS_norm = AllModels(1)(16).values[0]
    SCATT_norm = AllModels(1)(20).values[0]

    AllModels(1)(10).values = 0.
    AllModels(1)(20).values = 0.
    Plot("eemodel")
    pardf_name = "Transmitted"
    wdata.loc[:, pardf_name] = Plot.model()
    AllModels(1)(10).values = REPR_norm
    AllModels(1)(20).values = SCATT_norm
    
    AllModels(1)(16).values = 0.
    AllModels(1)(20).values = 0.
    Plot("eemodel")
    pardf_name = "Reprocessed"
    wdata.loc[:, pardf_name] = Plot.model()
    AllModels(1)(16).values = TRANS_norm
    AllModels(1)(20).values = SCATT_norm
    
    AllModels(1)(10).values = 0.
    AllModels(1)(16).values = 0.
    Plot("eemodel")
    pardf_name = "Scattered"
    wdata.loc[:, pardf_name] = Plot.model()

    wdata.loc[:, "Total"] = wdata[["Transmitted", "Reprocessed", "Scattered"]].sum(axis = 1)
    ## only needed if we were going to do something else with the model...
    # AllModels(1)(10).values = REPR_norm
    # AllModels(1)(16).values = TRANS_norm
    return wdata

def generate_uxclumpy_spectra(PhoIndex, Ecut, A_Fe, factor, logNHtor, CFtor, thInc):##, z
    
    mainModel = "constant*TBabs(atable{/Users/pboorman/Dropbox/data/xspec/models/borus/var_Fe/borus02_v170323b.fits} + zTBabs*cabs*cutoffpl + constant*cutoffpl)"
    m = Model(mainModel)
    dummy()

    set_default_values(False)
    wdata = pd.DataFrame(data = {"E_keV": Plot.x()})
    
    AllModels(1)(3).values = PhoIndex
    AllModels(1)(4).values = Ecut
    AllModels(1)(5).values = logNHtor
    AllModels(1)(6).values = CFtor
    AllModels(1)(7).values = thInc
    AllModels(1)(8).values = A_Fe
    # AllModels(1)(9).values = z
    AllModels(1)(17).values = factor

    # Plot("eemodel")
    # pardf_name = "Total"
    # wdata.loc[:, pardf_name] = Plot.model()
    
    REPR_norm = AllModels(1)(10).values[0]
    TRANS_norm = AllModels(1)(16).values[0]
    SCATT_norm = AllModels(1)(20).values[0]

    AllModels(1)(10).values = 0.
    AllModels(1)(20).values = 0.
    Plot("eemodel")
    pardf_name = "Transmitted"
    wdata.loc[:, pardf_name] = Plot.model()
    AllModels(1)(10).values = REPR_norm
    AllModels(1)(20).values = SCATT_norm
    
    AllModels(1)(16).values = 0.
    AllModels(1)(20).values = 0.
    Plot("eemodel")
    pardf_name = "Reprocessed"
    wdata.loc[:, pardf_name] = Plot.model()
    AllModels(1)(16).values = TRANS_norm
    AllModels(1)(20).values = SCATT_norm
    
    AllModels(1)(10).values = 0.
    AllModels(1)(16).values = 0.
    Plot("eemodel")
    pardf_name = "Scattered"
    wdata.loc[:, pardf_name] = Plot.model()

    wdata.loc[:, "Total"] = wdata[["Transmitted", "Reprocessed", "Scattered"]].sum(axis = 1)
    ## only needed if we were going to do something else with the model...
    # AllModels(1)(10).values = REPR_norm
    # AllModels(1)(16).values = TRANS_norm
    return wdata



DEGREE_SYMBOL = "\N{DEGREE SIGN}"

model_choice = st.sidebar.selectbox('Choose a model', ('borus02', 'pexrav', 'uxclumpy'))

st.sidebar.title("Parameters")

# Schematic info
# st.sidebar.subheader("AGN Geometry")
# st.sidebar.image("assets/schematic.png", use_column_width=True)



# Add some context in the main window

if model_choice == "borus02":
    st.title("${\\tt borus02}$ X-ray Simulator")
    st.subheader("${\\tt const}\\times {\\tt TBabs}({\\tt borus02\\_v170323b} + {\\tt zTBabs}\\times {\\tt cabs}\\times {\\tt cutoffpl} + {\\tt fscatt}\\times {\\tt cutoffpl})$")
    # Controllers
    PhoIndex_c = st.sidebar.slider(
        "Photon Index (PhoIndex)",
        min_value=1.45,
        max_value=2.55,
        value = 1.8,
        step=0.1,
        format="%.1f",
        key="PhoIndex",
    )
    Ecut_c = st.sidebar.slider(
        "High-energy cut-off (Ecut)",
        min_value=100.,
        max_value=1000.,
        value = 270.,
        step=10.,
        format="%.0f",
        key="Ecut",
    )
    logNHtor_c = st.sidebar.slider(
        "logNH (logNHtor)",
        min_value=22.,
        max_value=25.5,
        value = 24.,
        step=0.1,
        format="%.1f",
        key="logNHtor",
    )
    CFtor_c = st.sidebar.slider(
        "Torus Covering Factor (CFtor)",
        min_value=0.15,
        max_value=0.95,
        value = 0.5,
        step=0.1,
        format="%.1f",
        key="CFtor",
    )
    thInc_c = st.sidebar.slider(
        "Inclination Angle (thInc)",
        min_value=20.,
        max_value=85.0,
        value = 60.,
        step=1.,
        format=f"%.0f{DEGREE_SYMBOL}",
        key="thInc",
    )

    A_Fe_c = st.sidebar.slider(
        "Iron Abundance (A_Fe)",
        min_value=0.1,
        max_value=10.,
        value = 1.,
        step=0.1,
        format="%.1f",
        key="A_Fe",
    )

    # z_c = st.sidebar.slider(
    #     "Redshift",
    #     min_value=1.e-3,
    #     max_value=1.,
    #     value = 0.,
    #     step=0.001,
    #     format="%.3f",
    #     key="z",
    # )

    factor_c = st.sidebar.slider(
        "Scattered Fraction (fscatt)",
        min_value=1.e-5,
        max_value=1.e-1,
        value = 1.e-3,
        step=1.e-5,
        format="%.5f",
        key="factor",
    )
        # generate our data
    df = generate_borus02_spectra(PhoIndex_c, Ecut_c, A_Fe_c, factor_c, logNHtor_c, CFtor_c, thInc_c)##, z_c
    mod_df = df.melt(
        id_vars=["E_keV"], value_vars=["Transmitted", "Reprocessed", "Scattered", "Total"]
    )
    mod_df = mod_df.rename(columns={"variable": "Model Component", "value": "Flux"})

elif model_choice == "pexrav":
    st.title("${\\tt pexrav}$ X-ray Simulator")
    st.subheader("${\\tt const} \\times {\\tt TBabs} \\times {\\tt zTBabs}({\\tt pexrav} + {\\tt zTBabs} \\times {\\tt cabs} \\times {\\tt cutoffpl} + {\\tt fscatt} \\times {\\tt cutoffpl})$")
        # Controllers
    PhoIndex_c = st.sidebar.slider(
        "Photon Index (PhoIndex)",
        min_value=1.45,
        max_value=2.55,
        value = 1.8,
        step=0.1,
        format="%.1f",
        key="PhoIndex",
    )
    R_c = st.sidebar.slider(
        "High-energy cut-off (Ecut)",
        min_value=-100.,
        max_value=-1.e-1,
        value = -1.,
        step=0.1,
        format="%.0f",
        key="Ecut",
    )
    logNHtor_c = st.sidebar.slider(
        "logNH",
        min_value=20.,
        max_value=26.,
        value = 23.,
        step=0.1,
        format="%.1f",
        key="logNH",
    )
    thInc_c = st.sidebar.slider(
        "Inclination Angle (thInc)",
        min_value=20.,
        max_value=85.0,
        value = 60.,
        step=1.,
        format=f"%.0f{DEGREE_SYMBOL}",
        key="thInc",
    )

    factor_c = st.sidebar.slider(
        "Scattered Fraction (fscatt)",
        min_value=1.e-5,
        max_value=1.e-1,
        value = 1.e-3,
        step=1.e-5,
        format="%.5f",
        key="factor",
    )
        # generate our data
    df = generate_pexrav_spectra(PhoIndex_c, R_c, logNHtor_c, thInc_c, factor_c)##, z_c
    mod_df = df.melt(
        id_vars=["E_keV"], value_vars=["Transmitted", "Reprocessed", "Scattered", "Total"]
    )
    mod_df = mod_df.rename(columns={"variable": "Model Component", "value": "Flux"})

elif model_choice == "uxclumpy":
    st.title("${\\tt uxclumpy}$ X-ray Simulator")
    st.subheader("${\\tt const}\\times {\\tt TBabs}({\\tt uxclumpy\\_reflect} + {\\tt uxclumpy\\_transmit} + {\\tt fscatt} \\times {\\tt uxclumpy\\_omni.fits})$")
        # Controllers
    PhoIndex_c = st.sidebar.slider(
        "Photon Index (PhoIndex)",
        min_value=1.45,
        max_value=2.55,
        value = 1.8,
        step=0.1,
        format="%.1f",
        key="PhoIndex",
    )
    Ecut_c = st.sidebar.slider(
        "High-energy cut-off (Ecut)",
        min_value=100.,
        max_value=1000.,
        value = 270.,
        step=10.,
        format="%.0f",
        key="Ecut",
    )
    logNHLOS_c = st.sidebar.slider(
        "logNH (logNHLOS)",
        min_value=22.,
        max_value=25.5,
        value = 24.,
        step=0.1,
        format="%.1f",
        key="logNHLOS",
    )
    theta_tor_c = st.sidebar.slider(
        "Torus Opening Angle (theta_tor)",
        min_value=0.15,
        max_value=0.95,
        value = 0.5,
        step=0.1,
        format="%.1f",
        key="theta_tor",
    )
    thInc_c = st.sidebar.slider(
        "Inclination Angle (thInc)",
        min_value=20.,
        max_value=85.0,
        value = 60.,
        step=1.,
        format=f"%.0f{DEGREE_SYMBOL}",
        key="thInc",
    )

    CTKcover_c = st.sidebar.slider(
        "Compton-thick covering factor (CTKcover)",
        min_value=0.1,
        max_value=10.,
        value = 1.,
        step=0.1,
        format="%.1f",
        key="CTKcover",
    )

    factor_c = st.sidebar.slider(
        "Scattered Fraction (fscatt)",
        min_value=1.e-5,
        max_value=1.e-1,
        value = 1.e-3,
        step=1.e-5,
        format="%.5f",
        key="factor",
    )

    # generate our data
    df = generate_uxclumpy_spectra(PhoIndex_c, Ecut_c, logNHLOS_c, theta_tor_c, thInc_c, CTKcover_c, factor_c)##, z_c
    mod_df = df.melt(
        id_vars=["E_keV"], value_vars=["Transmitted", "Reprocessed", "Scattered", "Total"]
    )
    mod_df = mod_df.rename(columns={"variable": "Model Component", "value": "Flux"})

# Construct our plot
fig = px.line(
    mod_df,
    x="E_keV",
    y="Flux",
    color="Model Component",
    log_x=True,
    log_y=True,
    width=1000,
    height=700,
    labels=dict(Flux="EF_E / arbitrary", Energy="Energy / keV"),
)
fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), yaxis=dict(range=[-4., 1.]), xaxis=dict(range=[np.log10(1.), np.log10(200.)]))

st.plotly_chart(fig)

st.sidebar.markdown("### Model outputs")
if st.sidebar.checkbox("Show Table", False):
    st.subheader("Raw Data Table")
    st.write(df, index=False)

# Some advertising
st.sidebar.markdown("Designed by: Dr Peter Boorman & Dr Adam Hill")
