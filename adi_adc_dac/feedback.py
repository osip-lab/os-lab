import sys
import clr
import os
import shutil
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
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
    return float(x / 65535 * 2500.0)


def dac_volt2code(x):
    return int(x / 2500.0 * 65535)


def set_dac(cl, volt, lims=(0.0, 2500.0)):
    context = cl.get_ContextPath()
    client.ContextPath = r'\System\Subsystem_2\EVAL-AD5679RSDZ\AD5679R'  # noqa
    # plt.pause(0.01)
    volt = min(max(volt, lims[0]), lims[1])
    code = dac_volt2code(volt)
    client.SetBigIntParameter('Input_Value_DAC0', str(code), '-1')
    client.Run('@SoftwareLdac0')  # noqa
    m = f'DAC loaded ch0 - volt = {dac_code2volt(code):01.4f}, code = {code:d}'
    logging.info(m)
    client.ContextPath = context
    # plt.pause(0.01)
    return volt


def toggle_lock(event):
    global trig_lock, text_lock
    textbox_DAC0.set_active(trig_lock)
    trig_lock = not trig_lock
    text_lock.set_text('Locking is on' if trig_lock else 'Locking is off')
    fig.canvas.draw_idle()

def SP0_up(event):
    global SP0
    SP0 += 1
    textbox_ADC0_setpoint.set_val(f"{SP0:+07.3f}")

def SP0_down(event):
    global SP0
    SP0 -= 1
    textbox_ADC0_setpoint.set_val(f"{SP0:+07.3f}")

def SP0_change(event):
    global SP0
    SP0 = float(textbox_ADC0_setpoint.text)
    textbox_ADC0_setpoint.set_val(f"{SP0:+07.3f}")

def DAC0_change(event):
    if trig_lock == False:
        global v_dac, trig_dac_change
        trig_dac_change = True
        v_dac = float(textbox_DAC0.text)
        textbox_DAC0.set_val(f"{v_dac:+07.3f}")

def cI_change(event):
    global cI
    cI = float(textbox_cI.text)
    textbox_cI.set_val(f"{cI:+07.3f}")

def exit_app(event):
    global trig_main
    trig_main = False


if __name__ == "__main__":

    trig_main = True
    trig_lock = False
    trig_dac_change = False

    SP0 = 0.0 # initial setpoint for ADC0, mV
    v_dac = 500  # initial voltage for DAC, mV
    cI = 10  # initial integral coefficient
    t_show = 3600.0  # last time to show on graph, s
    t_pause = 1  # pause time between measurements, s

    w_dir = os.path.join(path_data_local, 'adi_adc_dac')
    data = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in zip(['ts', 't', *[f'adc_ch{i}' for i in range(8)], 'dac_ch0'], [str, *([float] * 10)])})

    adc_ls = ['adc_ch0']
    dac_ls = ['dac_ch0']
    labels = adc_ls + dac_ls

    fig = plt.figure(figsize=(10, 6))
    gs = grid.GridSpec(10, 6, figure=fig)
    ax = fig.add_subplot(gs[2:, :])
    textbox_ADC0_ax = fig.add_subplot(gs[0, 0])
    textbox_ADC1_ax = fig.add_subplot(gs[1, 0])
    textbox_ADC0_setpoint_ax = fig.add_subplot(gs[0, 1])
    textbox_DAC0_ax = fig.add_subplot(gs[0, 3])
    textbox_cI_ax = fig.add_subplot(gs[1, 3])
    button_lock_ax = fig.add_subplot(gs[0, 4])
    button_up_ax = fig.add_subplot(gs[0, 2])
    button_down_ax = fig.add_subplot(gs[1, 2])
    button_exit_ax = fig.add_subplot(gs[0, 5])

    sax = ax.twinx()
    lines = dict()
    for lbl in adc_ls:
        lines[lbl] = ax.plot([], [], marker='.')[0]
    for lbl in dac_ls:
        lines[lbl] = sax.plot([], [], ls='--', marker='x', c='tab:orange')[0]

    textbox_ADC0 = TextBox(textbox_ADC0_ax, 'ADC0')
    textbox_ADC1 = TextBox(textbox_ADC1_ax, 'ADC1')
    textbox_ADC0_setpoint = TextBox(textbox_ADC0_setpoint_ax, 'SP0')
    textbox_DAC0 = TextBox(textbox_DAC0_ax, 'DAC0')
    textbox_cI = TextBox(textbox_cI_ax, 'cI')
    button_lock = Button(button_lock_ax, 'Toggle Lock')
    button_up = Button(button_up_ax, 'SP up')
    button_down = Button(button_down_ax, 'SP down')
    button_exit = Button(button_exit_ax, 'Exit')
    text_lock = ax.text(0.5, 0.95, 'Locking is off', transform=ax.transAxes, fontsize=12, ha='center', color='red')
    button_up.on_clicked(SP0_up)
    button_down.on_clicked(SP0_down)
    textbox_ADC0_setpoint.on_submit(SP0_change)
    textbox_DAC0.on_submit(DAC0_change)
    textbox_cI.on_submit(cI_change)
    button_lock.on_clicked(toggle_lock)
    button_exit.on_clicked(exit_app)

    ax.set_xlabel('time, s')
    ax.set_ylabel('ADC0 voltage, mV')
    sax.set_ylabel('DAC0 voltage, mV')
    ax.grid()
    sax.grid(ls='--')
    fig.tight_layout()

    textbox_ADC0.set_active(False)
    textbox_ADC1.set_active(False)
    textbox_ADC0_setpoint.set_val(f"{SP0:+07.3f}")
    textbox_DAC0.set_val(f"{v_dac:+07.3f}")
    textbox_cI.set_val(f"{cI:+07.3f}")

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

    # client.ContextPath = r'\System\Subsystem_1\AD4111 Board\AD4111'
    # client.NavigateToPath("Root::System.Subsystem_1.AD4111 Board.AD4111")
    # client.NavigateToRelativePath("Volts/Amps Analysis")
    # client.SetIntParameter("virtual-parameter-samplecount2", "100", "-1")
    # plt.pause(1.0)

    client.NavigateToPath("Root::System.Subsystem_2.EVAL-AD5679RSDZ.AD5679R")  # noqa
    plt.pause(1.0)

    set_dac(client, v_dac)

    client.ContextPath = r'\System\Subsystem_1\AD4111 Board\AD4111'
    plt.pause(0.1)

    time_stamp = timestamp()

    while trig_main:

        time_stamp2 = timestamp()
        print(timestamp_to_float(time_stamp2) - timestamp_to_float(time_stamp))
        time_stamp = time_stamp2
        v_dac_old = v_dac

        client.Run('@AsyncDataCapture("test")')
        plt.pause(0.05)

        folder = os.path.join(w_dir, time_stamp)
        os.makedirs(folder)

        # client.PullAllCaptureDataToFile(os.path.join(folder, 'Data_'), '-1', '0', '1000', 'test', 'Create')
        try:
            client.PullAllCaptureDataToFile(os.path.join(folder, 'Data_'), '-1', '0', '1000', 'test', 'Create')
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
            data.loc[i, f'adc_ch{ch}'] = value * 1000

        textbox_ADC0.set_val(f"{data.loc[i, f'adc_ch0']:+07.3f}")
        textbox_ADC1.set_val(f"{data.loc[i, f'adc_ch1']:+07.3f}")

        data.loc[i, f'dac_ch0'] = v_dac

        msg = (f"ADC measured - ts = {data.loc[i, 'ts']}, t = {data.loc[i, 't']:09.3f} s, " +
               ", ".join(f"ch{k} = {data.loc[i, f'adc_ch{k}']:+01.3f} mV" for k in range(8)))
        logging.info(msg)

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

        # here is the feedback loop
        if trig_lock:
            v = np.array(data['adc_ch0'])[-1]
            v_dac -= (v - SP0) / cI

        if v_dac != v_dac_old:
            v_dac = set_dac(client, v_dac)
            textbox_DAC0.set_val(f"{v_dac:+07.3f}")

        if trig_dac_change:
            v_dac = set_dac(client, v_dac)
            trig_dac_change = False

        shutil.rmtree(folder)

        plt.pause(t_pause - (timestamp_to_float(timestamp()) - timestamp_to_float(time_stamp)))

    client.CloseSession()
