__author__ = 'drjfunk'

import theano.tensor as T
from theano.ifelse import ifelse
from astropy.constants import c as sol
from astropy import cosmology
from cosmo_theano.utils.integration_routines import gauss_kronrod

cosmo = cosmology.FlatLambdaCDM(H0=67.3, Om0=.3)
Or = cosmo.Onu0 + cosmo.Ogamma0
sol = sol.value



# Flat universe with cosmological constant


def integrand_constant_flat(z, Om):
    """

    :param z: redshift
    :param Om: matter content
    :return: theano array of 1/H(z)
    """
    zp = (1 + z)
    Ode = 1 - Om - Or # Adjust cosmological constant

    return  T.power(T.pow(zp, 3) * Om + Ode, -0.5)



def lumdist_constant_flat(Om, h0, z):
    """
    Distance modulus for a flat universe with a
    cosmological constant

    :param Om: matter content
    :param h0: hubble constant
    :param z: redshift
    :return: theano array of dist. mods.
    """

    # Hubble distance
    dh = sol * 1.e-3 / h0

    # comoving distance
    dc = dh * gauss_kronrod(integrand_constant_flat, z, parameters=[Om])

    # luminosity distance
    dl = (1 + z) * dc

    return dl

