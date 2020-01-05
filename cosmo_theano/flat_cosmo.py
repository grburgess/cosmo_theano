__author__ = 'drjfunk'

import theano.tensor as T
from theano.ifelse import ifelse
from astropy.constants import c as sol
from astropy import cosmology
from cosmo_theano.utils.integration_routines import gauss_kronrod




# Flat universe with cosmological constant



class flat_cosmology(object):

    def __init__(self, Om=0.286, H0=69.3):


        astro_cosmo = cosmology.FlatLambdaCDM(H0=H0, Om0=Om)
        self._Or = cosmo.Onu0 + cosmo.Ogamma0
        self._sol = sol.value

        self._Om = Om
        self._H0 = H0

        self._Ode = 1 - self._Om - self._Or

        self._Om_Ode_sum = self._Om + self._Ode

        self._dh = self._sol * 1E-3 / self._H0
        
    def _integrand_constant_fixed(self, z):
        """

        :param z: redshift
        :param Om: matter content
        :return: theano array of 1/H(z)
        """
        zp = (1 + z)

        return  T.power(T.pow(zp, 3) * self._Om_Ode_sum, -0.5)

    def _integrand_constant(self, z, Om):
        """

        :param z: redshift
        :param Om: matter content
        :return: theano array of 1/H(z)
        """
        zp = (1 + z)
        Ode = 1 - Om - self._Or # Adjust cosmological constant
        return  T.power(T.pow(zp, 3) * Om + Ode, -0.5)

    def comoving_transverse_distance(self, z, Om):

        tmp = lambda a,b: self._integrand_constant(a,b)
        
        dc = self._dh * gauss_kronrod(tmp, z, parameters=[Om])

        return dc


    def comoving_transverse_distance_fixed(self, z):

        tmp = lambda a: self._integrand_constant_fixed(a)
        
        dc = self._dh * gauss_kronrod(tmp, z)

        return dc



    def comoving_volume(self,z,Om):

        return 4./3. * np.pi * self.comoving_transverse_distance(z,Om)

    
    def comoving_volume_fixed(self,z):

        return 4./3. * np.pi * self.comoving_transverse_distance_fixed(z)

    

    
    def luminosity_distance_fixed(self, z):
        """
        Distance modulus for a flat universe with a
        cosmological constant

        :param z: redshift
        :return: theano array of dist. mods.
        """

        # Hubble distance
        

        # comoving distance

        
        dc = self.comoving_transverse_distance_fixed(z)

        # luminosity distance
        dl = (1 + z) * dc *  3.086E24 # in centimeters

        return dl



    def luminosity_distance(self, Om, z):
        """
        Distance modulus for a flat universe with a
        cosmological constant

        :param Om: matter content
        :param h0: hubble constant
        :param z: redshift
        :return: theano array of dist. mods.
        """

        # Hubble distance


        # comoving distance
        dc = self.comoving_transverse_distance(z,Om)

        # luminosity distance
        dl = (1 + z) * dc * 3.086E24 # in cm
        return dl

