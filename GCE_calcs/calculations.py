import numpy as np

def get_eflux(e_spec,J,sigma,mass):
    '''
    converts binned per annihilation number spectra or per annihilation energy spectra to observed number flux or energy flux
    i.e. does phi_i =  j*sigmav/8pim_x^2 (int e dN/de or int dN/de)
    #should return an array of shape (n_cross,n_J,n_mass,n_spec)
    #background, exposure should be vectors of len n_spec: the number of energy bins
    #e_spec is an array with shape n_mass,n_spec, it is the binned spectra for the various energy bins, for each mass
    #J, sigma, mass are all vectors with their corresponding length
    '''
    n_sigma = len(sigma)
    n_mass = len(mass)
    n_J = len(J)
    n_mass2,n_spec = e_spec.shape
    assert n_mass ==n_mass2, 'the mass and spectra arrays are incompatible'
    J = np.tile(J[np.newaxis,:,np.newaxis,np.newaxis],(n_sigma,1,n_mass,n_spec))
    mass = np.tile(mass[np.newaxis,np.newaxis,:,np.newaxis],(n_sigma,n_J,1,n_spec))
    sigma = np.tile(sigma[:,np.newaxis,np.newaxis,np.newaxis],(1,n_J,n_mass,n_spec))
    e_spec = np.tile(e_spec[np.newaxis,np.newaxis,:],(n_sigma,n_J,1,1))
    eflux = J*sigma*e_spec/(8.*np.pi*mass**2)
    return eflux
