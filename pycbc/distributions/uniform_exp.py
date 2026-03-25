# Copyright (C) 2017  Christopher M. Biwer
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
""" This modules provides classes for evaluating distributions whose exponentiations
of base 10 are uniform.
"""

import numpy
from pycbc.distributions import uniform

class UniformExp10(uniform.Uniform):
    """ A uniform distribution on the exponentiation base 10 of the given parameters.
    The parameters are independent of each other. Instances of this class can
    be called like a function. By default, logpdf will be called.
    Parameters
    ----------
    \**params :
        The keyword arguments should provide the names of parameters and their
        corresponding bounds, as either tuples or a `boundaries.Bounds`
        instance.
    Attributes
    ----------
    name : "uniform_exp10"
        The name of this distribution.
    """
    name = "uniform_exp10"

    def __init__(self, **params):
        super(UniformExp10, self).__init__(**params)
        self._norm = 1./numpy.prod([10.**(bnd[1]) - 10.**(bnd[0])
                                   for bnd in self._bounds.values()])
        self._lognorm = numpy.log(self._norm)

    def _cdfinv_param(self, param, value):
        """Return the cdfinv for a single given parameter """
        lower_bound = 10.**(self._bounds[param][0])
        upper_bound = 10.**(self._bounds[param][1])
        return numpy.log10((upper_bound - lower_bound) * value + lower_bound)

    def _pdf(self, **kwargs):
        """Returns the pdf at the given values. The keyword arguments must
        contain all of parameters in self's params. Unrecognized arguments are
        ignored.
        """
        if kwargs in self:
            vals = self._norm * 10.**(numpy.array([kwargs[param] 
                                      for param in kwargs.keys()]))
            return numpy.prod(vals)
        else:
            return 0.

    def _logpdf(self, **kwargs):
        """Returns the log of the pdf at the given values. The keyword
        arguments must contain all of parameters in self's params. Unrecognized
        arguments are ignored.
        """
        if kwargs in self:
            return numpy.log(self._pdf(**kwargs))
        else:
            return -numpy.inf

__all__ = ["UniformExp10"]
