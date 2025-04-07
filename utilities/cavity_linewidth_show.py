import os
import time
import re
import matplotlib.pyplot as plt
import numpy as np

from local_config import PATH_DATA_LOCAL


trig = True
t_range = 120  # time interval for showing, s


def press(event):
    global trig
    if event.key == 'escape':
        trig = False


if __name__ == "__main__":

    log_list = os.listdir(os.path.join(PATH_DATA_LOCAL, 'linewidth_measurer'))
    log_list = list(filter(lambda x: x.endswith('log.txt'), log_list))
    log_list = sorted(log_list, key=lambda x: time.mktime(time.strptime(x[0:19], '%Y-%m-%d %H-%M-%S')))
    log = log_list[-1]

    labels = (r'T \+ L', )
    data = {lbl: np.array([], dtype='float64') for lbl in labels}
    data['time'] = np.array([], dtype='float64')

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    cid = fig.canvas.mpl_connect('key_press_event', press)
    lines = dict()
    for lbl in labels:
        lines[lbl] = ax.plot([], [], marker='.')[0]
    ax.set_xlabel('time, s')
    ax.set_ylabel('T + L, ppm')
    ax.grid()
    fig.tight_layout()
    plt.show(block=False)

    with open(os.path.join(PATH_DATA_LOCAL, 'linewidth_measurer', log), 'r') as f:
        while trig:
            line = f.readline()
            if line:
                print(line.strip())
                if 'measured linewidth successfully' in line:
                    # add new points to data
                    t1 = time.mktime(time.strptime(line[:19], '%Y.%m.%d %H:%M:%S'))  # seconds
                    t2 = float(line[20:23]) / 1e3  # milliseconds part
                    t = t1 + t2
                    data['time'] = np.append(data['time'], t)
                    for lbl in labels:
                        v = re.search(r'T \+ L' + r' = \d{1,6}\.\d{1,6}', line).group()
                        v = float(re.search(r'\d{1,6}\.\d{1,6}', v).group())
                        data[lbl] = np.append(data[lbl], v)
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
                    v1 = np.min([np.min(data[lbl][m]) for lbl in labels])
                    v2 = np.max([np.max(data[lbl][m]) for lbl in labels])
                    dv = v2 - v1
                    if dv < 1.0:
                        ax.set_ylim(v1 - 0.5, v2 + 0.5)
                    else:
                        ax.set_ylim(v1 - 0.1 * dv, v2 + 0.1 * dv)
            else:
                if not plt.fignum_exists(fig.number):
                    break
                plt.pause(0.1)
