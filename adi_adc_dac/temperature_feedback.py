import sys
import clr
import os
import shutil
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import matplotlib.gridspec as grid
from local_config import path_data_local

sys.path.append(r'C:\Program Files\Analog Devices\ACE\Client')
clr.AddReference('AnalogDevices.Csa.Remoting.Clients')  # noqa
clr.AddReference('AnalogDevices.Csa.Remoting.Contracts')  # noqa
from AnalogDevices.Csa.Remoting.Clients import ClientManager, AceOperationException  # noqa


def timestamp():
    return f"{time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())}.{int((time.time() % 1) * 1000):03d}"


def timestamp_to_float(ts):
    dt, msec = ts.split(".")
    dt = time.strptime(dt, "%Y-%m-%d %H-%M-%S")
    t = time.mktime(dt)
    t = t + int(msec) / 1000
    return t


def adc_code2volt(x):
    return float((x / 2**23 - 1) * 10 * 2.5)


def dac_code2volt(x):
    return float(x / 65535 * 2.5)


def dac_volt2code(x):
    return int(x / 2.5 * 65535)


def set_dac(cl, volt, lims=(0.0, 2.5)):
    global text_temp
    context = cl.get_ContextPath()
    client.ContextPath = r'\System\Subsystem_2\EVAL-AD5679RSDZ\AD5679R'  # noqa
    # plt.pause(0.01)
    volt = min(max(volt, lims[0]), lims[1])
    code = dac_volt2code(volt)
    client.SetBigIntParameter('Input_Value_DAC0', str(code), '-1')
    client.Run('@SoftwareLdac0')  # noqa
    m = f'DAC loaded ch0 - volt = {dac_code2volt(code):01.4f}, code = {code:d}'
    logging.info(m)
    text_temp.set_text(f'Temp voltage {v_dac * 1000:+07.1f} mV')
    client.ContextPath = context
    # plt.pause(0.01)
    return volt


def toggle_lock(event):
    global trig_lock, text_lock
    trig_lock = not trig_lock
    text_lock.set_text('Locking is on' if trig_lock else 'Locking is off')
    fig.canvas.draw_idle()

def toggle_drift(event):
    global trig_drift, text_drift
    trig_drift = not trig_drift
    text_drift.set_text('Drift is on' if trig_drift else 'Drift is off')
    fig.canvas.draw_idle()


def temp_up(event):
    global v_dac, v_st_m
    v_dac += v_st_m
    v_dac = set_dac(client, v_dac, lims=v_lim)


def temp_down(event):
    global v_dac, v_st_m
    v_dac -= v_st_m
    v_dac = set_dac(client, v_dac, lims=v_lim)

def temp_up10(event):
    global v_dac, v_st_m
    v_dac += v_st_m * 10
    v_dac = set_dac(client, v_dac, lims=v_lim)


def temp_down10(event):
    global v_dac, v_st_m
    v_dac -= v_st_m * 10
    v_dac = set_dac(client, v_dac, lims=v_lim)

def exit_app(event):
    global trig_main
    trig_main = False


if __name__ == "__main__":

    trig_main = True
    trig_lock = False
    trig_drift = False

    v_dac = 1.250  # initial voltage for DAC, V
    v_th = 1.0  # threshold voltage to react, V
    v_c = 0.0  # central voltage to react, V
    v_st = 0.001  # step voltage for DAC, V
    v_st_m = 0.001  # step voltage for manual adjusting of DAC, V
    v_lim = (0.25, 2.25)  # limitation for DAC voltage, V
    t_skip = 3.0  # time for skipping next DAC increase, s
    t_show = 600.0  # last time to show on graph, s
    t_pause = 0.5  # pause time between measurements, s

    w_dir = os.path.join(path_data_local, 'adi_adc_dac')
    data = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in zip(['ts', 't', *[f'adc_ch{i}' for i in range(8)], 'dac_ch0'], [str, *([float] * 10)])})

    adc_ls = ['adc_ch0']
    dac_ls = ['dac_ch0']
    labels = adc_ls + dac_ls

    fig = plt.figure(figsize=(10, 6))
    gs = grid.GridSpec(10, 6, figure=fig)
    ax = fig.add_subplot(gs[2:, :])
    button_lock_ax = fig.add_subplot(gs[0, 0])
    button_up10_ax = fig.add_subplot(gs[0, 4])
    button_up_ax = fig.add_subplot(gs[0, 3])
    button_down_ax = fig.add_subplot(gs[0, 2])
    button_down10_ax = fig.add_subplot(gs[0, 1])
    button_exit_ax = fig.add_subplot(gs[0, 5])
    button_drift_ax = fig.add_subplot(gs[1, 0])
    slider_drift_ax = fig.add_subplot(gs[1, 1:5])

    sax = ax.twinx()
    lines = dict()
    for lbl in adc_ls:
        lines[lbl] = ax.plot([], [], marker='.')[0]
    for lbl in dac_ls:
        lines[lbl] = sax.plot([], [], ls='--', marker='x', c='tab:orange')[0]

    button_lock = Button(button_lock_ax, 'Toggle Lock')
    button_up10 = Button(button_up10_ax, 'Temp up x10')
    button_up = Button(button_up_ax, 'Temp up')
    button_down = Button(button_down_ax, 'Temp down')
    button_down10 = Button(button_down10_ax, 'Temp down x10')
    button_exit = Button(button_exit_ax, 'Exit')
    button_drift = Button(button_drift_ax, 'Toggle Drift')
    slider_drift = Slider(slider_drift_ax, label='', valmin = -3, valmax = 3, valinit = 0)
    text_lock = ax.text(0.5, 0.95, 'Locking is off', transform=ax.transAxes, fontsize=12, ha='center', color='red')
    text_piezo = ax.text(0.2, 0.95, f'Piezo voltage {0.0 * 1000:+07.1f} mV', transform=ax.transAxes, fontsize=12, ha='center', color='k')
    text_temp = ax.text(0.8, 0.95, f'Temp voltage {v_dac * 1000:+07.1f} mV', transform=ax.transAxes, fontsize=12, ha='center', color='k')
    text_drift = ax.text(0.5, 0.9, 'Drift is off', transform=ax.transAxes, fontsize=12, ha='center', color='red')
    button_lock.on_clicked(toggle_lock)
    button_up.on_clicked(temp_up)
    button_down.on_clicked(temp_down)
    button_up10.on_clicked(temp_up10)
    button_down10.on_clicked(temp_down10)
    button_exit.on_clicked(exit_app)
    button_drift.on_clicked(toggle_drift)

    ax.set_xlabel('time, s')
    ax.set_ylabel('adc voltage, V')
    sax.set_ylabel('dac voltage, V')
    ax.grid()
    sax.grid(ls='--')
    fig.tight_layout()

    plt.show(block=False)
    plt.pause(0.1)

    start_time = time.strftime('%Y-%m-%d %H-%M-%S')
    logging.basicConfig(filename=os.path.join(w_dir, f'{start_time} log.txt'),
                        level=logging.INFO, format='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%Y.%m.%d %H:%M:%S')
    w_dir = os.path.join(w_dir, f'{start_time} data')
    os.makedirs(w_dir, exist_ok=True)

    manager = ClientManager.Create()
    client = manager.CreateRequestClient("localhost:2357")

    client.AddByComponentId("AD4111Board")
    client.NavigateToPath("Root::System")
    plt.pause(1.0)

    client.AddByComponentId("AD5679RBoard")
    client.NavigateToPath("Root::System")
    plt.pause(1.0)

    client.ContextPath = r'\System\Subsystem_1\AD4111 Board'
    client.NavigateToPath("Root::System.Subsystem_1.AD4111 Board")
    plt.pause(1.0)

    client.SetIntParameter("virtual-parameter-demo-select", "3", "-1")
    client.SetBoolParameter("virtual-parameter-demo-load", "True", "-1")
    client.SetBoolParameter("virtual-parameter-demo-load", "False", "-1")
    plt.pause(1.0)

    client.NavigateToPath("Root::System.Subsystem_2.EVAL-AD5679RSDZ.AD5679R")  # noqa
    plt.pause(1.0)

    set_dac(client, v_dac)

    client.ContextPath = r'\System\Subsystem_1\AD4111 Board\AD4111'
    plt.pause(0.1)

    t_last = 0.0

    time_stamp = timestamp()

    while trig_main:

        time_stamp2 = timestamp()
        t_drift = timestamp_to_float(time_stamp2) - timestamp_to_float(time_stamp)
        time_stamp = time_stamp2
        v_dac_old = v_dac
        print(t_drift)

        client.Run('@AsyncDataCapture("test")')
        plt.pause(0.05)

        folder = os.path.join(w_dir, time_stamp)
        os.makedirs(folder)

        # client.PullAllCaptureDataToFile(os.path.join(folder, 'Data_'), '-1', '0', '1000', 'test', 'Create')
        try:
            client.PullAllCaptureDataToFile(os.path.join(folder, 'Data_'), '-1', '0', '100', 'test', 'Create')
        except AceOperationException:
            shutil.rmtree(folder)
            plt.pause(t_pause)
            continue

        i = len(data)
        data.loc[i, 'ts'] = time_stamp
        data.loc[i, 't'] = timestamp_to_float(time_stamp)

        files = os.listdir(folder)
        files = [f for f in files if f.endswith('.csv') and os.path.isfile(os.path.join(folder, f))]
        for f in files:
            ch = f[13]
            df = pd.read_csv(os.path.join(folder, f), header=None)
            value = df[df.columns[0]].mean()
            value = adc_code2volt(value)
            data.loc[i, f'adc_ch{ch}'] = value

        data.loc[i, f'dac_ch0'] = v_dac

        msg = (f"ADC measured - ts = {data.loc[i, 'ts']}, t = {data.loc[i, 't']:09.3f} s, " +
               ", ".join(f"ch{k} = {data.loc[i, f'adc_ch{k}']:+01.4f} V" for k in range(8)))
        logging.info(msg)
        text_piezo.set_text(f"Piezo voltage {np.array(data['adc_ch0'])[-1] * 1000:+07.1f} mV")

        mask = np.max(data['t']) - data['t'] < t_show
        tdf = data[mask]
        t0 = tdf['t'].min(skipna=True)

        for lbl in labels:
            lines[lbl].set_data(tdf['t'] - t0, tdf[lbl])

        # calculate and set new limits
        t1 = np.min(tdf['t'])
        t2 = np.max(tdf['t'])
        dt = t2 - t1
        if dt < 1.0:
            ax.set_xlim(t1 - 0.5 - t0, t2 + 0.5 - t0)
        else:
            ax.set_xlim(t1 - 0.1 * dt - t0, t2 + 0.1 * dt - t0)
        for a, ls in zip((ax, sax), (adc_ls, dac_ls)):
            v1 = np.min([np.min(tdf[lbl]) for lbl in ls])
            v2 = np.max([np.max(tdf[lbl]) for lbl in ls])
            dv = v2 - v1
            if dv < 1.0:
                a.set_ylim(v1 - 0.5, v2 + 0.5)
            else:
                a.set_ylim(v1 - 0.1 * dv, v2 + 0.1 * dv)

        # here is the drift
        if trig_drift:
            v_dac += slider_drift.val * t_drift / 1000
            #v_dac = set_dac(client, v_dac, lims=v_lim)

        # here is the feedback loop
        if trig_lock and ((np.array(data['t'])[-1] - t_last) > t_skip):
            v = np.array(data['adc_ch0'])[-1]
            if v > (v_c + v_th):
                v_dac -= v_st
                #v_dac = set_dac(client, v_dac, lims=v_lim)
            elif v < (v_c - v_th):
                v_dac += v_st
                #v_dac = set_dac(client, v_dac, lims=v_lim)
            t_last = time.time()

        if v_dac != v_dac_old:
            v_dac = set_dac(client, v_dac, lims=v_lim)

        shutil.rmtree(folder)

        plt.pause(t_pause)

    client.CloseSession()
