import numpy as np


def adc2mv(data: dict, mode: str = 'channels'):
    signals = dict()
    signals['time'] = np.arange(0, data['common']['time']['length'] * data['common']['time']['interval'],
                                data['common']['time']['interval']) / 1e9  # convert in s
    channels = set(data.keys()) - {'common'}
    channels = sorted(list(channels))
    for channel in channels:
        signal = np.array(data[channel]['data'], dtype='float64')
        signal = signal * data[channel]['header']['range'] / data['common']['device']['maxADC'] / 1e3  # convert in V
        if mode == 'channels':
            signals[channel] = signal
        elif mode == 'names':
            name = f"{data['common']['name']} {data[channel]['header']['name']}"
            signals[name] = signal
        else:
            raise ValueError("mode should be 'channels' or 'names'")
    return signals
