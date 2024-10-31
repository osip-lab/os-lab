import os
import time
import re
import matplotlib.pyplot as plt
import numpy as np

from local_config import path_data_local


def press(event):
    global trig
    if event.key == 'escape':
        trig = False


if __name__ == "__main__":

    trig = True
    t_range = 120  # time interval for showing, s
    folder = 'line_shape_measurer'
    fsr0 = 150e6

    log_list = os.listdir(os.path.join(path_data_local, folder))
    log_list = list(filter(lambda x: x.endswith('log.txt'), log_list))
    log_list = sorted(log_list, key=lambda x: time.mktime(time.strptime(x[0:19], '%Y-%m-%d %H-%M-%S')))
    log = log_list[-1]

    labels = (r'T \+ L', )
    data = {lbl: np.array([], dtype='float64') for lbl in labels}
    data['time'] = np.array([], dtype='float64')

    ls1 = (r'T \+ L', )
    ls2 = ('FSR', )
    labels = ls1 + ls2
    data = {lbl: np.array([], dtype='float64') for lbl in labels}
    data['time'] = np.array([], dtype='float64')

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    cid = fig.canvas.mpl_connect('key_press_event', press)
    sax = ax.twinx()
    lines = dict()
    for lbl in ls1:
        lines[lbl] = ax.plot([], [], marker='.')[0]
    for lbl in ls2:
        lines[lbl] = sax.plot([], [], ls='--', marker='x')[0]
    ax.set_xlabel('time, s')
    ax.set_ylabel('T + L, ppm')
    sax.set_ylabel('FSR, Hz')
    ax.grid()
    sax.grid(ls='--')
    fig.tight_layout()
    plt.show(block=False)

    with open(os.path.join(path_data_local, folder, log), 'r') as f:
        while trig:
            line = f.readline()
            if line:
                print(line.strip())
                if 'measured line shape' in line:
                    # add new points to data
                    t1 = time.mktime(time.strptime(line[:19], '%Y.%m.%d %H:%M:%S'))  # seconds
                    t2 = float(line[20:23]) / 1e3  # milliseconds part
                    t = t1 + t2
                    data['time'] = np.append(data['time'], t)
                    for lbl in labels:
                        v = re.search(lbl + r' = \d{1,12}\.\d{1,12}', line).group()
                        v = float(re.search(r'\d{1,12}\.\d{1,12}', v).group())
                        offset = fsr0 if lbl == 'FSR' else 0.0
                        data[lbl] = np.append(data[lbl], v - offset)
                    m = data['time'] - np.max(data['time']) > -t_range
                    t0 = np.min(data['time'][m])
                    # set new data to lines
                    for lbl in labels:
                        lines[lbl].set_data(data['time'][m] - t0, data[lbl][m])
                    # calculate and set new limits
                    t1 = np.min(data['time'][m])
                    t2 = np.max(data['time'][m])
                    dt = t2 - t1
                    if dt < 1.0:
                        ax.set_xlim(t1 - 0.5 - t0, t2 + 0.5 - t0)
                    else:
                        ax.set_xlim(t1 - 0.1 * dt - t0, t2 + 0.1 * dt - t0)
                    for a, ls in zip((ax, sax), (ls1, ls2)):
                        v1 = np.min([np.min(data[lbl][m]) for lbl in ls])
                        v2 = np.max([np.max(data[lbl][m]) for lbl in ls])
                        dv = v2 - v1
                        if dv < 1.0:
                            a.set_ylim(v1 - 0.5, v2 + 0.5)
                        else:
                            a.set_ylim(v1 - 0.1 * dv, v2 + 0.1 * dv)
            else:
                plt.pause(0.1)