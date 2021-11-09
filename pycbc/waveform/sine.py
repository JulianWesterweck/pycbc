import numpy
from pycbc.types import TimeSeries

def td_sine(**kwargs):
    """
    Generates a sine waveform in the time domain,
    starting abruptly at tc.

    Parameters
    ----------
    freq : float
        Frequency of the sine wave.
    phi : float
        Initial phase of the sine wave. phi=0 yields
        a cosine for the plus- and a sine for the cross-polarisation
        relative to the reference time.
    amp : float
        Amplitude of the sine wave.
    t_final : float
        Duration of the sine wave.
    inclination : float
        Inclination of the source w.r.t the vector towards the observer.
    delta_t : float
        Time step for the waveform.

    Returns
    hplus : TimeSeries
        The plus polarisation of the sine waveform.
    hcross : TimeSeries
        The cross polarisation of the sine waveform.
    -------
    """
    dt = kwargs['delta_t']
    f = kwargs['freq']
    t_final = kwargs['t_final']
    amp = kwargs['amp']
    inclination = kwargs['inclination']
    phi = kwargs['phi']

    times = numpy.arange(0, t_final, dt)

    wf_real = numpy.cos(2. * numpy.pi * f * times + phi) * 0.5 \
            * (1. + numpy.cos(inclination)**2.)
    wf_imag = 1.0j * numpy.sin(2. * numpy.pi * f * times + phi) \
            * numpy.cos(inclination)

    wf = TimeSeries(amp*(wf_real+wf_imag), delta_t=dt, epoch=0)

    return wf.real(), wf.imag()

# Approximant names ###########################################################
sine_td_approximants = {
    'TdSine': td_sine
    }
