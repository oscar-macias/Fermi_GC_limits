#  $Id$
#
#  Copyright (C) 2020
#  Ryan Keeley <rkeeley@uci.edu> and Oscar Macias <oscar.macias@ipmu.jp>
#  This example reproduces the dark matter limits in <https://arxiv.org/pdf/2003.>
#  here we show how to reproduce the DM limits in  Figs. 1 and 2.
#  Example usage, execute this file as:   $ python Fermi_GC_limits.py --channel=0 --GDE_model=0 --DM_model=1 --ignore
#
#  This file is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>
#
#

import argparse
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy import interpolate

import GCE_calcs


def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--channel', type=int, default=0,
        help='annihilation channel: bbar=0, tau=1, mu=2, W=3, Z=4, Higgs=5: default bbar')
    parser.add_argument('--GDE_model', type=int, default=0,
        help='GDE model, default baseline')
    parser.add_argument('--DM_model', type=int, default=0,
        help='DM morphology')
    parser.add_argument('--path', type=str, default='.', help='Provide absolute path to your results directory')
    parser.add_argument('--trunc', type=int, default=0,
        help='number of data points to truncate at low energies, default 0')
    parser.add_argument('--ignore', action= 'store_true',
        help='ignore divide by zero errors in log')
    args = parser.parse_args()

    rc('font',**{'family':'serif','serif':['Times New Roman']})
    plt.rcParams.update({'font.size': 16})

    channel = args.channel
    model = args.DM_model
    trunc = args.trunc
    kind = args.GDE_model
    path = args.path

    if args.ignore:
        np.seterr(divide='ignore')

    def dir_path(path):
        if os.path.isdir(path):
            return path
        else:
            raise argparse.ArgumentTypeError(r'readable_dir:{'+path+'} is not a valid path')

    # Supported self-annihilation channels
    channel_name = ['bbar','tau','mu','W','Z','higgs'][channel]
    # DM profiles: 0:NFW, 1:triaxial NFW, 2:Read, 3:triaxial Read
    # if GDE model is NFW_gamma, the code assumes GDE_baseline and the NFW gamma index runs from 0-9 spanning gamma= {0.5,1.5}

    GDE_kind = ['GDE_baseline','GDE_Dust','GDE_Gas','NFW_gamma','GDE_CSE','GDE_ICS_6rings','GDE_Gas_ICS_6rings'][kind]
    if GDE_kind == 'GDE_baseline':
        if model > 3:
            print('choose a different DM model: 0,1,2,3')
        label = ['fit352','fit353','fit354','fit355'][model]
    if GDE_kind == 'GDE_Dust':
        if model > 3:
            print('choose a different DM model: 0,1,2,3')
        label = ['fit380','fit381','fit382','fit383'][model]
    if GDE_kind == 'GDE_Gas':
        if model > 3:
            print('choose a different DM model: 0,1,2,3')
        label = ['fit384','fit385','fit386','fit387'][model]
    if GDE_kind == 'NFW_gamma':
        if model > 9:
            print('choose a different DM model: 0thru9')
        label = ['fit388','fit389','fit390','fit391','fit392','fit393','fit394','fit395','fit396','fit397'][model]
    if GDE_kind == 'GDE_CSE':
        if model > 3:
            print('choose a different DM model: 0,1,2,3')
        label = ['fit402','fit403','fit404','fit405'][model]
    if GDE_kind == 'GDE_ICS_6rings':
        if model > 3:
            print('choose a different DM model: 0,1,2,3')
        label = ['fit398','fit399','fit400','fit401'][model]

    print('Selected self-annihilation channel is %s' % channel_name)
    print('Selected GDE model is %s' % GDE_kind)
    print('Selected DM morphology is %s' % label)

    mass_table = np.loadtxt('spectra/binned_0414/new_mass_table.txt')

    if GDE_kind == 'NFW_gamma':
        if model == 5:
            J_filename = 'J_MC_limit_nomask1p0_40x40'
        elif model == 6:
            J_filename = 'J_MC_limit_nomask1p1_40x40'
        elif model == 7:
            J_filename = 'J_MC_limit_nomask1p3_40x40'
        elif model == 8:
            J_filename = 'J_MC_limit_nomask1p4_40x40'
        elif model == 9:
            J_filename = 'J_MC_limit_nomask1p5_40x40'
        else:
            print('choose another NFW gamma')
    else:
        if model == 0:
            J_filename = 'J_MC_limit_nomask1p2_40x40'
        elif model == 1:
            J_filename = 'J_MC_limit_triaxial_40x40'
        elif model == 2:
            J_filename = 'J_MC_limit_Read_40x40'
        elif model == 3:
            J_filename = 'J_MC_limit_Read_triaxial_40x40'
        else:
            print('choose another DM morphology')

    print('The J-factor data was stored in file %s' % J_filename)
    n_mass = len(mass_table)
    mass_prior_norm = np.trapz(np.ones(n_mass), x=np.log10(mass_table)) #the inverse of this norm quantities are the prior on mass

    n_sigma = 61
    sigma = np.logspace(-29., -23., num=n_sigma, endpoint=True) #flat prior in logspace (scale-invariant)
    sigma_prior_norm = np.trapz(np.ones(n_sigma), x=np.log(sigma)) #the inverse of this norm quantity is the prior on sigma

    # --------------------------------------------------------------------------
    # Read in the log-like data obtained from maximum-likelihood runs
    energies = np.loadtxt('./data/likelihood_profiles/'+GDE_kind+'_corrected/Ebands.dat')
    emin = energies[:,1]
    emax = energies[:,2]
    de = (emax-emin)/1e6
    ebin = np.sqrt(emin*emax)
    bin_center = ebin/1e6

    # --------------------------------------------------------------------------
    # Compute deltaLoglikes at 95% CL for display - not used in actual calculations!

    flux_high = np.empty(energies[trunc:].shape[0])
    for i in range(energies[trunc:].shape[0]):
        # This chunk can be used to plot the 95% CL flux upper limits as obtained from the log-like analysis
        # Finds the d log like =1.3  (95% CL) for each energy bin and saves that flux value
        dloglike,nflux = np.loadtxt('./data/likelihood_profiles/'+GDE_kind+'_corrected/UL_scan_Ebin_'+str(trunc+i+2)+'_'+label+'_V2.dat', delimiter=',').T
        f = interp1d(nflux,dloglike - 1.35,kind='linear',bounds_error=False,fill_value='extrapolate')
        flux_high[i] = brentq(f,0.,nflux[-1])

    # -----------------------------------------------------------------------------
    # Read the log-like data and compute \delta \log(like)

    def GCE_delta_log_like_limits(model_flux, GDE_kind, label, trunc):
        '''model_flux is an array of DM fluxes in the same energy bins as the binned likelihood profiles
                      has shape of (nsigma, nJ, nmass, nbin), should already be truncated
           GDE_kind and label just specify which likelihood profiles to load (GDE model plus DM morphology)
            trunc an int, ignores the first number energy bins
        '''
        energy_dat = np.loadtxt('./data/likelihood_profiles/'+GDE_kind+'_corrected/Ebands.dat')
        assert model_flux.shape[3] == energy_dat[trunc:].shape[0], 'the energy shapes are not right ' + str(
            model_flux.shape[3]) + ' and ' + str(energy_dat[trunc:].shape[0])
        delta_log_like = np.zeros(model_flux.shape)
        for i in range(model_flux.shape[3]):
            dloglike, nflux = np.loadtxt('./data/likelihood_profiles/'+GDE_kind+'_corrected/UL_scan_Ebin_'+str(trunc+i+2)+'_'+label+'_V2.dat', delimiter=',').T
            f = interpolate.interp1d(nflux, -dloglike, kind='linear', bounds_error=False, fill_value='extrapolate')
            delta_log_like[:, :, :, i] = f(model_flux[:, :, :, i])
        return delta_log_like

    # -------------------------------------------------------------------------
    # Start of Bayesian analysis
    J_prior = np.loadtxt('./data/J_factors/'+str(J_filename)+'.txt')[1] #load the prior on the J-factor, calculated from a convolution of priors on rho_local, gamma, and the scale radius

    jmax_prior = J_prior.argmax()
    J = np.loadtxt('./data/J_factors/'+str(J_filename)+'.txt')[0]
    n_J = len(J)

    binned_spectra = np.loadtxt('./spectra/binned_0414/'+channel_name+'_binned.txt')[:,trunc:]

    #This is a binned dflux/dE
    nspec_GCE = GCE_calcs.calculations.get_eflux(binned_spectra,J,sigma,mass_table) #it says get e_flux but since a binned number flux (dN/dE) is input, it returns a number spectra


    log_like_GCE_4d = GCE_delta_log_like_limits(nspec_GCE,GDE_kind,label,trunc)

    log_like_GCE_3d = np.sum(log_like_GCE_4d,axis=3) #summing the log-like along the energy bin axis

    J_prior = np.loadtxt('./data/J_factors/'+str(J_filename)+'.txt')[1] #load the prior on the J-factor, calculated from a convolution of priors on rho_local, gamma, and the scale radius

    jmax_prior = J_prior.argmax()

    J_prior = np.tile(J_prior[np.newaxis,:,np.newaxis],(n_sigma,1,n_mass)) #tiling to make the prior the same shape as the likelihood

    GCE_like_3d = np.exp(log_like_GCE_3d)*J_prior

    GCE_like_2d = np.trapz(GCE_like_3d, x=np.log10(J), axis=1) #marginalizing over the J factor

    evidence_GCE = np.trapz(np.trapz(GCE_like_2d,x = np.log(sigma),axis =0),x = np.log10(mass_table),axis=0) / (sigma_prior_norm * mass_prior_norm)

    # --------------------------------------------------------------------------
    # Plot the analysis results
    fig = plt.figure('fluxes',figsize=(6.5,5.))
    fig.subplots_adjust(left=0.15,bottom=0.15,right=0.98,
                        top=0.98,wspace=0.0,hspace=0.18)

    cmap = cm.cool

    levels = [1.35]  #[0,1,3,4.4,6,10,15,21,28,36]
    CS = plt.contour(mass_table,sigma,-np.log(GCE_like_2d) + np.log(GCE_like_2d[0,-1]), levels, cmap=cm.get_cmap(cmap, len(levels) - 1))
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Mass [GeV]')
    plt.ylim(1e-28,1e-23)
    plt.xlim(7, 1e4)
    plt.ylabel(r'Cross Section [cm$^3$ sec$^{-1}$]')
    plt.legend(loc='best',frameon=False)
    #plt.title(r'GCE $-\Delta$Log-Likelihood Contours')
    plt.savefig(dir_path(path) + '/GCE_contours_'+label+'.png')
    plt.clf()

    np.savetxt(dir_path(path) + '/sigma_mass_posterior_'+label+'.txt', (-np.log(GCE_like_2d) + np.log(GCE_like_2d.max())))
if __name__ == '__main__':
    main()
