"""
Stages:
1) generate parameters
2) save csv of parameters and a unique filename
3) then go through each parameter combo and simulate the xspec model
4) save the xspec model E, model, emodel, eemodel with filename from the parameter combo csv

Considerations:
- size of one spectrum csv. This will dictate how many we can simulate on a laptop
- time taken to produce one spectrum csv. Consider parallelisation
"""
import os
from tqdm import tqdm
from xspec import *
import pandas as pd
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import functools
# Boorman_cosmo = FlatLambdaCDM(H0 = 67.3, Om0 = 0.315)

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

mainModel = "constant*TBabs(atable{/Users/pboorman/Dropbox/data/xspec/models/borus/var_Fe/borus02_v170323b.fits} + zTBabs*cabs*cutoffpl + constant*cutoffpl)"



def print_free_pars():
    print("Components:")
    print(AllModels(1).componentNames)
    for i in range(AllModels(1).startParIndex, AllModels(1).nParameters + 1, 1): 
        print() 
        print("Parameter: " + str(i)) 
        print("Parameter name: " + str(AllModels(1)(i).name)) 
        print("Parameter unit: " + str(AllModels(1)(i).unit)) 
        print("Parameter values:")
        print(AllModels(1)(i).values)


def dummy():
    AllData.dummyrsp(lowE = 1.e-3, highE = 2.e2, nBins = 10 ** 3, scaleType = "log")
    Plot("model")
        
def set_default_values(save = False):
    """
    Set default values for the model being stepped over
    """
    ## note the norms must be independent and not tied
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

    if (save == True):
        dummy()
        default_wdata = pd.DataFrame(data = {"E_keV": Plot.x(), "M": Plot.model()})
        default_filename = "./default.csv"
        default_wdata.to_csv(default_filename, index = False)
          

def step_pars():
    """
    Main purpose:
        - Generate a pandas data frame where each row is a sample of
          the default parameters of a model
        - Used to understand the effect of changing a given parameters
          on a model

    Note: every time Model() is called, the default parameters are set
    """
    m = Model(mainModel)
    set_default_values(True)
    parLen = 5
    stepparDict = {"3-PhoIndex": np.linspace(1.45, 2.55, 3),
                   "4-Ecut": np.logspace(2., 3., 3),
                   "5-logNHtor": np.linspace(22., 25.5, parLen),
                   "6-CFtor": np.linspace(0.15, 0.95, parLen),
                   "7-thInc": np.linspace(20., 85, parLen),
                   "8-A_Fe": np.logspace(-1., 1., 3),
                   "9-z": np.logspace(-3., 0., 4),
                   "17-factor": np.logspace(-5., -1., 3)}

    m = Model(mainModel)
    set_default_values(False)
    dummy()
    wdata = pd.DataFrame(data = {"E_keV": Plot.x()})
    
    par_store_dir = "./borus_stepped"
    if not os.path.isdir(par_store_dir):
        os.mkdir(par_store_dir)
    filename = par_store_dir + "/borus_step.csv"
    counter = 0

    runningTot = 1
    totSteps = functools.reduce(lambda x, y: x * y, [len(pars) * runningTot for par_names, pars in stepparDict.items()])
    with tqdm(total = totSteps, position = 0, desc = "Realisations") as pbar:
        for i, PhoIndex in enumerate(stepparDict["3-PhoIndex"]):
            for j, Ecut in enumerate(stepparDict["4-Ecut"]):
                for k, logNHtor in enumerate(stepparDict["5-logNHtor"]):
                    for l, CFtor in enumerate(stepparDict["6-CFtor"]):
                        for m, thInc in enumerate(stepparDict["7-thInc"]):
                            for n, A_Fe in enumerate(stepparDict["8-A_Fe"]):
                                for o, z in enumerate(stepparDict["9-z"]):
                                    for p, factor in enumerate(stepparDict["17-factor"]):
                                    
                                    
                                        AllModels(1)(3).values = PhoIndex
                                        AllModels(1)(4).values = Ecut
                                        AllModels(1)(5).values = logNHtor
                                        AllModels(1)(6).values = CFtor
                                        AllModels(1)(7).values = thInc
                                        AllModels(1)(8).values = A_Fe
                                        AllModels(1)(9).values = z
                                        AllModels(1)(17).values = factor

                                        Plot("model")
                                        pardf_name = "MODEL_p3=%.5f_p4=%.5f_p5=%.5f_p6=%.5f_p7=%.5f_p8=%.5f_p9=%.5f_p17=%.5f" %(PhoIndex, Ecut, logNHtor, CFtor, thInc, A_Fe, z, factor)
                                        wdata.loc[:, pardf_name] = Plot.model()
                                        
                                        REPR_norm = AllModels(1)(10).values[0]
                                        TRANS_norm = AllModels(1)(16).values[0]
                                        SCATT_norm = AllModels(1)(20).values[0]

                                        AllModels(1)(10).values = 0.
                                        AllModels(1)(20).values = 0.
                                        Plot("model")
                                        pardf_name = "TRANS_p3=%.5f_p4=%.5f_p5=%.5f_p6=%.5f_p7=%.5f_p8=%.5f_p9=%.5f_p17=%.5f" %(PhoIndex, Ecut, logNHtor, CFtor, thInc, A_Fe, z, factor)
                                        wdata.loc[:, pardf_name] = Plot.model()
                                        AllModels(1)(10).values = REPR_norm
                                        AllModels(1)(20).values = SCATT_norm
                                        
                                        AllModels(1)(16).values = 0.
                                        AllModels(1)(20).values = 0.
                                        Plot("model")
                                        pardf_name = "REPR_p3=%.5f_p4=%.5f_p5=%.5f_p6=%.5f_p7=%.5f_p8=%.5f_p9=%.5f_p17=%.5f" %(PhoIndex, Ecut, logNHtor, CFtor, thInc, A_Fe, z, factor)
                                        wdata.loc[:, pardf_name] = Plot.model()
                                        AllModels(1)(16).values = TRANS_norm
                                        AllModels(1)(20).values = SCATT_norm
                                        
                                        AllModels(1)(10).values = 0.
                                        AllModels(1)(16).values = 0.
                                        Plot("model")
                                        pardf_name = "SCATT_p3=%.5f_p4=%.5f_p5=%.5f_p6=%.5f_p7=%.5f_p8=%.5f_p9=%.5f_p17=%.5f" %(PhoIndex, Ecut, logNHtor, CFtor, thInc, A_Fe, z, factor)
                                        wdata.loc[:, pardf_name] = Plot.model()
                                        ## only needed if we were going to do something else with the model...
                                        # AllModels(1)(10).values = REPR_norm
                                        # AllModels(1)(16).values = TRANS_norm

                                        pbar.update(1)
                        wdata.to_csv(filename.replace(".csv", "%d.csv" %counter), index = False)
                        counter += 1
                        wdata = pd.DataFrame(data = {"E_keV": Plot.x()})

        
step_pars()
























