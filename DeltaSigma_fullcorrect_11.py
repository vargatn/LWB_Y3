import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
import fitsio as fio

import h5py

import copy

import xpipe.tools.catalogs as catalogs
import xpipe.paths as paths
import xpipe.xhandle.parbins as parbins
import xpipe.xhandle.xwrap as xwrap
import xpipe.tools.selector as selector
import xpipe.xhandle.shearops as shearops
import xpipe.xhandle.pzboost as pzboost
import xpipe.tools.y3_sompz as sompz
import xpipe.tools.mass as mass
import xpipe.tools.visual as visual


from importlib import reload
import pickle


import astropy.cosmology as cosmology
# this is just the default cosmology
cosmo = cosmology.FlatLambdaCDM(Om0=0.3, H0=70)

# we make sure the correct config file is loaded here, it will let us automatically now what type of files 
# were / will be produced, and where they will be placed
paths.update_params("/home/moon/vargatn/DES/PROJECTS/xpipe/settings/params_y3lwb-v02_meta.yml")


main_file_path = "/e/ocean1/users/vargatn/DESY3/Y3_mastercat_03_31_20.h5"
src = sompz.sompz_reader(main_file_path)
src.build_lookup()

Rshear = []
Rsel = []
for sbin in np.arange(4):
#     fname = "/e/ocean1/users/vargatn/DES/DES_Y3A2_cluster/data/shearcat/y3_mcal_sompz_v4_bin"+str(sbin + 1)+".h5"
#     tab = pd.read_hdf(fname, key="data")
#     _Rshear = np.average(0.5 * (tab["R11"] + tab["R22"]), weights=tab["weight"])
#     Rshear.append(_Rshear)
    print(sbin)
    fnames = [
        "/e/ocean1/users/vargatn/DES/DES_Y3A2_cluster/data/shearcat/y3_mcal_sompz_v4_unblind_bin"+str(sbin + 1)+"_1p.h5",
        "/e/ocean1/users/vargatn/DES/DES_Y3A2_cluster/data/shearcat/y3_mcal_sompz_v4_unblind_bin"+str(sbin + 1)+"_1m.h5",
        "/e/ocean1/users/vargatn/DES/DES_Y3A2_cluster/data/shearcat/y3_mcal_sompz_v4_unblind_bin"+str(sbin + 1)+"_2p.h5",
        "/e/ocean1/users/vargatn/DES/DES_Y3A2_cluster/data/shearcat/y3_mcal_sompz_v4_unblind_bin"+str(sbin + 1)+"_2m.h5",
    ]    
    tables = [pd.read_hdf(fname, key="data") for fname in fnames]  
    _R11 = (np.average(tables[0]["e1"], weights=tables[0]["weight"]) -\
            np.average(tables[1]["e1"], weights=tables[1]["weight"])) / 0.02
    _R22 = (np.average(tables[2]["e2"], weights=tables[2]["weight"]) -\
            np.average(tables[3]["e2"], weights=tables[3]["weight"])) / 0.02
    
    _Rsel = 0.5 * (_R11 + _R22)
    Rsel.append(_Rsel)
    
    
oname = "/e/ocean1/users/vargatn/DES/DES_Y3A2_cluster/data/lenscat/LWB_DESY3_ALL.fits"
allgal = fio.read(oname).byteswap().newbyteorder()


oname = "/e/ocean1/users/vargatn/DES/DES_Y3A2_cluster/data/lenscat/LWB_DESY3_ALL_rand.fits"
allrand = fio.read(oname).byteswap().newbyteorder()


# now extract the len weights, these are not used yet we can pass them on the post-processing stage
weights = pd.DataFrame()
weights["WEIGHT"] = allgal["WSYS"]
weights["ID"] = allgal["ID"]

weights_rand = pd.DataFrame()
weights_rand["WEIGHT"] = allrand["WEIGHT"]
weights_rand["ID"] = np.arange(len(allrand))

flist, flist_jk, rlist, rlist_jk = parbins.get_file_lists(paths.params, paths.dirpaths)

ms_opt=[np.array([-0.02,-0.024,-0.037]),np.array([-0.024,-0.037]),np.array([-0.024,-0.037]),np.array([-0.037,])]
optsbins=[(1,2,3),(2,3),(2,3),(3,)]

ACP_optms = []
ACP_optms_backup = []
ACP_optms_backup2 = []
for i, fname in enumerate(flist):
#     print(ms[i],optsbins[i])
    mfac_opt=1/(1+ms_opt[i])

    
    ACP = shearops.AutoCalibrateProfile([fname,], flist_jk[i], src, xlims=(0.01, 100), sbins=optsbins[i])
    ACP.get_profiles(ismeta=False, weights=weights, id_key="ID", z_key="Z", mfactor_sbins=mfac_opt, Rs_sbins=Rsel) #
    ACP_optms.append(ACP)    
    
    ACP = shearops.AutoCalibrateProfile([fname,], flist_jk[i], src, xlims=(0.01, 100), sbins=optsbins[i])
    ACP.get_profiles(ismeta=False, weights=weights, id_key="ID", z_key="Z", mfactor_sbins=mfac_opt, Rs_sbins=Rsel) #
    ACP_optms_backup.append(ACP) 
    
    ACP = shearops.AutoCalibrateProfile([fname,], flist_jk[i], src, xlims=(0.01, 100), sbins=optsbins[i])
    ACP.get_profiles(ismeta=False, weights=weights, id_key="ID", z_key="Z", mfactor_sbins=mfac_opt, Rs_sbins=Rsel) #
    ACP_optms_backup2.append(ACP)     
    
    
ACP_optm_rands2 = []
for i, rname in enumerate(rlist):
#     mfac_opt2=1/(1+ms_opt2[i])
    mfac_opt=1/(1+ms_opt[i])

    # processing randoms
    ACP = shearops.AutoCalibrateProfile([rname,], rlist_jk[i], src, xlims=(0.01, 100), sbins=optsbins[i])
    ACP.get_profiles(scinvs=ACP_optms[i].scinvs, weights=weights_rand, ismeta=False, Rs_sbins=Rsel,
                     id_key="ID", z_key="Z", mfactor_sbins=mfac_opt) #
    ACP_optm_rands2.append(ACP)
    

# This should reproduce the PZ boost wrapper
class CorrBoost(object):
    def __init__(self, corrs):
        self.boost_amps = [corrs,]

ACPs_boosted_corr = []
for i in np.arange(4): # loop over each lens bin

    profs = ACP_optms[i]._profiles # profile around LRG
    profs_ref = ACP_optm_rands2[i]._profiles # profile around randoms
    
    corrs = []
    for j in np.arange(len(profs)): # loop over each source bin
        
        corr = profs[j].snum / profs_ref[j].snum - 1
        corr = corr / (1 + corr)        
        corrs.append(corr)
        
    cb = CorrBoost(corrs)

    ACP = copy.copy(ACP_optms_backup[i])
    ACP.add_boost(cb)
    
    ACPs_boosted_corr.append(ACP)
    
    
def write_profile(prof, path):
    """saves DeltaSigma and covariance in text format"""

    # Saving profile
    profheader = "R [Mpc]\tDeltaSigma_t [M_sun / pc^2]\tDeltaSigma_t_err [M_sun / pc^2]\tDeltaSigma_x [M_sun / pc^2]\tDeltaSigma_x_err [M_sun / pc^2]"
    res = np.vstack((prof.rr, prof.dst, prof.dst_err, prof.dsx, prof.dsx_err)).T
    fname = path + "_profile.dat"
    print("saving:", fname)
    np.savetxt(fname, res, header=profheader)

    # Saving covariance
    np.savetxt(path + "_dst_cov.dat", prof.dst_cov)
    np.savetxt(path + "_dsx_cov.dat", prof.dsx_cov)
    
# calculate stack-wise subtraction
profiles_subtracted_Corr = []
for i in np.arange(4):
    prof1 = copy.copy(ACPs_boosted_corr[i].profile)
#     prof1 = copy.copy(ACP_optms[i].profile)
    prof2 = copy.copy(ACP_optm_rands2[i].profile)
#     prof = prof1.profile.composite
    prof1.composite(prof2, operation="-")
    profiles_subtracted_Corr.append(prof1)
    
    
# tmp_profs = copy.deepcopy(profiles)
fnames = []
for i in np.arange(4):
#     for s in np.arange(3):
#         scinv = scinvs[i][s+1]
#         prof = tmp_profs[i][s]
#         prof.multiply(1 / scinv)
        fname = "LWB_lowz+cmass-100Mpc_boost_rand-subtr_zbin"+str(i)+"_combined_v7_Corr"
        fnames.append(fname)
        prof = profiles_subtracted_Corr[i]
        write_profile(prof, fname)