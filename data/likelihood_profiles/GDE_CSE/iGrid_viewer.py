#!/usr/bin/env python

###################################################################
# Oscar Macias [Kavli IPMU]                                       #
# oscar.macias@ipmu.jp                                            #
###################################################################

from __future__ import division # confidence high

from astropy.io import fits
from astropy.wcs import WCS
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import fsolve


import sys
import os
import datetime
import numpy as np
import io
import matplotlib.pyplot as plt
plt.style.use('seaborn-ticks')


fitnum = int(sys.argv[1])


data = np.loadtxt("Ebands.dat")
emin = data[:,1]
emin = np.array(emin)*1.e-6
emax = data[:,2]
emax = np.array(emax)*1.e-6

fig = plt.figure(figsize=[29,30])
fig.subplots_adjust(hspace=0.3, wspace=0.2)


for i in range(2, 17):
    ax = fig.add_subplot(5, 4, i-1)
    ax.minorticks_on()
    ax.tick_params('both', length=10, width=1, which='major',
                   direction='in',bottom=True, top=True, left=True, right=True, pad=5, labelsize=26)
    ax.tick_params('both', length=5, width=1, which='minor',
                   direction='in',bottom=True, top=True, left=True, right=True)
    plt.ylim(-6, 10)
    plt.xlim(1e-12,1e-5)
    
    plt.axhline(y=0, ls='--',color='black')
    plt.xscale('log')
    egy_min = '%.2f' % (emin[i-2])
    egy_max = '%.2f' % (emax[i-2])
    plt.title(r''+str(egy_min)+' - '+str(egy_max)+' GeV, Ebin'+str(i)+'', fontsize=26)

    data = np.loadtxt('./UL_scan_Ebin_'+str(i)+'_fit'+str(fitnum)+'.dat', delimiter=',')
    flux =  data[:,1]
    deltalog = data[:,0]
    ax.plot(flux,deltalog, ls='-', lw=1.6,marker='v', color='red')#, marker='o'
    

    if(i==17):
        plt.ylabel(r'$\Delta \log(\rm like)$',fontsize=26)
        plt.xlabel(r'Norm [ph cm$^{-2}$ $s^{-2}$]',fontsize=26)
 
plt.plot(0,0, label=r'Fit'+str(fitnum)+'',ls='-', lw=2.0,marker='v', color='red')
#plt.plot(0,0, label=r'(NFW+benchmark GDE, manual scan)',ls='-', lw=2.0,marker='^', color='black')


ax.legend(loc='upper center', bbox_to_anchor=(1.68, 1.0), fontsize=22,  fancybox=True, shadow=True, ncol=1)
plt.savefig('grid_fit'+str(fitnum)+'.pdf', bbox_inches='tight')
    #plt.plot(1,2, label='F98-SA0')
    #lg = plt.legend(loc=2,ncol=1,fontsize=11)
    #lg.draw_frame(False)
    
