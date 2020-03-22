#  $Id$
#
#  Copyright (C) 2020
#  Ryan Keeley <rkeeley@uci.edu> and Oscar Macias <oscar.macias@ipmu.jp>
#  This example reproduces the dark matter limits in <https://arxiv.org/pdf/1503.02641.pdf>
#  here we show how to reproduce the DM limits for the Draco dSph
#  Example usage, execute this file as:  $ python draco_test.py --channel=0 --ignore
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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

# Import a few functions
import GCE_calcs


# -----------------------------------------------------------------------------------
def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--channel', type=int, default=0,
                        help='annihilation channel: bbar=0 tau=1, default bbar')
    parser.add_argument('--path', type=str, default='.',  help='Provide absolute path to your results directory')
    parser.add_argument('--ignore', action='store_true',
                        help='ignore divide by zero errors in log')
    args = parser.parse_args()

    rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
    plt.rcParams.update({'font.size': 16})

    channel = args.channel
    path = args.path

    if args.ignore:
        np.seterr(divide='ignore')

    def dir_path(path):
        if os.path.isdir(path):
            return path
        else:
            raise argparse.ArgumentTypeError(r'readable_dir:{path} is not a valid path')

    # Supported self-annihilation channels

    channel_name = ['bbar', 'tau', 'mu', 'W', 'Z', 'higgs'][channel]

    print('Selected self-annihilation channel is %s' % (channel_name))
    mass_table = np.loadtxt("spectra/binned_0414/new_mass_table.txt")

    n_mass = len(mass_table)
    mass_prior_norm = np.trapz(np.ones(n_mass),
                               x=np.log10(mass_table))  # the inverse of this norm quantities are the prior on mass

    n_sigma = 61 # number of subdivisions for numerical integration of probability density distribution functions
    sigma = np.logspace(-29., -23., num=n_sigma, endpoint=True)  # flat prior in logspace (scale-invariant)
    sigma_prior_norm = np.trapz(np.ones(n_sigma),
                                x=np.log(sigma))  # the inverse of this norm quantities are the prior on sigma

    n_J = 300
    like_name = ['like_draco'] # dwarf considered in this example

    # Dwarf properties extracted from ArXiv:1408.0002
    dwarf_mean_J = {'like_carina': 17.92,
                    'like_draco': 19.05,
                    'like_fornax': 17.84,
                    'like_leo_I': 17.84,
                    'like_leo_II': 17.97,
                    'like_sculptor': 18.57,
                    'like_sextans': 17.92,
                    'like_ursa_minor': 18.95}

    dwarf_err_J = {'like_carina': 0.09,
                   'like_draco': 0.12,
                   'like_fornax': 0.06,
                   'like_leo_I': 0.16,
                   'like_leo_II': 0.18,
                   'like_sculptor': 0.05,
                   'like_sextans': 0.18,
                   'like_ursa_minor': 0.18}

    binned_energy_spectra_dwarf = np.loadtxt('spectra/binned_0414/' + channel_name + '_binned_dwarfs.txt')
    like_dwarf_2d = np.ones((n_sigma, n_mass))
    dwarf_levels = [1.35] # delta-loglike for 95% CL is 2.7/2.



    for i in range(len(like_name)):
        name = like_name[i]
        print('Dwarf Spheroidal included in the analysis is %s' % name)
        J_dwarf = np.logspace(dwarf_mean_J[name] - 4 * dwarf_err_J[name],
                              dwarf_mean_J[name] + 4 * dwarf_err_J[name], n_J)
        J_prior_dwarf = GCE_calcs.analysis.get_J_prior_dwarf(J_dwarf, dwarf_mean_J[name], dwarf_err_J[name])
        norm_test = np.trapz(J_prior_dwarf, x=np.log10(J_dwarf))
        assert abs(norm_test - 1) < 0.01, 'the normalization of the prior on the J-factor is off by more than 1%'
        J_prior_dwarf = np.tile(J_prior_dwarf[np.newaxis, :, np.newaxis], (
        n_sigma, 1, n_mass))  # tiling to make the prior of the J factor the same shape as the likelihood
        spec_dwarf = GCE_calcs.calculations.get_eflux(binned_energy_spectra_dwarf, J_dwarf, sigma,
                                                       mass_table)  # calculating the energy flux spectra
        log_like_dwarf_4d = GCE_calcs.analysis.dwarf_delta_log_like(spec_dwarf, name)  # likelihood
        log_like_dwarf_3d = np.sum(log_like_dwarf_4d, axis=3)  # summing over energy bins
        like_dwarf_3d = np.exp(log_like_dwarf_3d) * J_prior_dwarf
        like_ind_2d = np.trapz(like_dwarf_3d, x=np.log10(J_dwarf), axis=1)  # marginalizing over J factor

        CS = plt.contour(mass_table, sigma, -np.log(like_ind_2d) + np.log(like_ind_2d.max()), dwarf_levels)
        plt.clabel(CS, inline=False, fontsize=10,label=r'95% CL upper limits')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Mass [GeV]')
        plt.ylabel('Cross Section [cm^3 sec^-1]')
        plt.savefig(path+'/' + name + '_contours.png')
        plt.clf()
        like_dwarf_2d *= like_ind_2d

        np.savetxt(path+'/dwarf_contours.txt',
                   -np.log(like_dwarf_2d / like_dwarf_2d.max()))

    evidence_dwarf = np.trapz(np.trapz(like_dwarf_2d, x=np.log(sigma), axis=0), x=np.log10(mass_table), axis=0) / (
                sigma_prior_norm * mass_prior_norm)


    print('the dwarf evidence is ' + str(evidence_dwarf))


if __name__ == '__main__':
    main()
