import numpy as np
from scipy import stats
from scipy import interpolate


def get_J_prior_dwarf(J,mu,sigma):
    exponent = np.random.normal(mu,sigma,100000)
    J_kde_exp = stats.gaussian_kde(exponent)
    return J_kde_exp(np.log10(J))

def GCE_delta_log_like_limits(model_flux,label_kind,label_model,trunc):
    '''model_flux is an array of DM fluxes in the same energy bins as the binned likelihood profiles
                  has shape of (nsigma, nJ, nmass, nbin), should already be truncated
       label_kind and label_model just specify whcih likelihood profiles to load
        trunc an int, ignores the first number energy bins
    '''
    energy_dat = np.loadtxt('Ebands.dat')
    assert model_flux.shape[3] == energy_dat[trunc:].shape[0], 'the energy shape are not right '+str(model_flux.shape[3])+' and '+str(energy_dat[trunc:].shape[0])
    delta_log_like = np.zeros(model_flux.shape)
    for i in range(model_flux.shape[3]):
        dloglike_load,nflux_load = np.loadtxt('data/limits_final/'+label_kind+'/UL_scan_Ebin_'+str(trunc+i+2)+'_'+label_model+'.dat', delimiter=',').T
        dloglike = np.insert(dloglike_load,0,0.0)
        nflux = np.insert(nflux_load,0,0.0)
        f = interpolate.interp1d(nflux,-dloglike,kind='linear',bounds_error=False,fill_value='extrapolate')
        delta_log_like[:,:,:,i] = f(model_flux[:,:,:,i])
    return delta_log_like

def dwarf_delta_log_like(espectra,like_name):
    data = np.loadtxt('release-01-00-00/'+like_name+'.txt', unpack=True)
    delta_log_like = np.zeros(espectra.shape)
    for i in range(espectra.shape[3]):
        istart = i*25 # This weirdness arises from the weirdness in the Fermi dwarf likelihood formats
        iend = istart+25
        # divide by 1000 to convert from MeV to GeV
        f = interpolate.interp1d(data[2,istart:iend]/1000.,data[3,istart:iend],kind='linear',bounds_error=False,fill_value='extrapolate')
        delta_log_like[:,:,:,i] = f(espectra[:,:,:,i])
    return delta_log_like
