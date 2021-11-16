import numpy
from pycbc.types import TimeSeries
from pycbc.waveform.waveform import get_obj_attrs

required_args = ['freq', 'phi', 'amp', 'inclination', 't_final']
td_args = {'delta_t': None, 't_final': None, 'taper': False}

def props(obj, required, domain_args, **kwargs):
    """ Return a dictionary built from the combination of defaults, kwargs,
    and the attributes of the given object.
    """
    # Get the attributes of the template object
    pr = get_obj_attrs(obj)

    # Get the parameters to generate the waveform
    # Note that keyword arguments override values in the template object
    input_params = domain_args.copy()
    input_params.update(pr)
    input_params.update(kwargs)
    # Check if the required arguments are given
    for arg in required:
        if arg not in input_params:
            raise ValueError('Please provide ' + str(arg))

    return input_params

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

def get_td_sine(template=None, **kwargs):
    """
    Generates a sine waveform in the time domain,
    starting abruptly at tc.

    Parameters
    ----------
    template : object
        An object that has attached properties. This can be used to substitute
        for keyword arguments. A common example would be a row in an xml table.
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
    input_params = props(template, required_args, td_args, **kwargs)
    return td_sine(input_params)

# Approximant names ###########################################################
sine_td_approximants = {
    'TdSine': get_td_sine
    }
