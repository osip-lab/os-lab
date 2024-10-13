"""
TODO:
    1) margins in channel control widget
    2) is it better to transfer conversion from user units to device values from specific widgets into common controller?
    3) filter file names (by conversion to int)
    4) with a fast trigger, it can have no time to save bug files because there is no connection between the saver and controller
    5) try to look after updating events of plotter and its curves during the multithreading
"""
import sys
import os
from typing import Union
import numpy as np
import datetime
import ctypes
import json
import jsbeautifier
from itertools import cycle

from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QCheckBox, QSizePolicy, QFileDialog, QProgressBar,
                             QComboBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QEvent, QTimer
from PyQt6.QtGui import QFont, QColor

from pyqtgraph import GraphicsLayoutWidget, mkPen
from dvg_pyqtgraph_threadsafe import PlotCurve

from picosdk.ps4000a import ps4000a as ps
from picosdk.functions import assert_pico_ok
from picosdk.errors import PicoSDKCtypesError

from qt_gui.qt_ext import MyStandardWindow, QMyStandardButton, QCenteredLabel, QMyHBoxLayout, QMyVBoxLayout, \
    QMySpinBox, QMergedRadioButton, QMyComboBox, QMyLineEdit, ThreadedWorker, ThreadedWidget
from pico_scope import adc2mv

from local_config import path_data_local


class SignalConstructor(ThreadedWorker):
    send_signals = pyqtSignal(dict)

    def __init__(self, thread: QThread):
        super(SignalConstructor, self).__init__(thread)

    @pyqtSlot(dict, str)
    def adc2mv(self, data: dict, mode: str):
        signals = adc2mv(data, mode)
        self.finish(self.send_signals, signals)


class PicoControlWorker(ThreadedWorker):
    pico_scanned = pyqtSignal(dict)
    pico_connected = pyqtSignal()
    pico_disconnected = pyqtSignal()
    pico_loaded = pyqtSignal()
    pico_measured = pyqtSignal()
    send_data = pyqtSignal(dict)

    def __init__(self, thread):
        super(PicoControlWorker, self).__init__(thread)
        self.status = dict()
        self.settings = dict()
        self.data = dict()
        self.device = ctypes.c_int16()
        # self.channelNames = '12345678'
        self.channelNames = '1234'
        self.channelInputRanges = (10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000)
        self.channelCoupling = ('AC', 'DC')
        self.triggerSwitch = ('off', 'on')
        self.triggerDirection = ('above', 'below', 'rising', 'falling', 'rising_or_falling')
        self.buffer_dict_max = dict()
        self.maxADC = ctypes.c_int16(32767)
        self.maxSamples = 0
        self.cMaxSamples = ctypes.c_int32(self.maxSamples)
        self.timeIntervals = ctypes.c_float()
        self.returnedMaxSamples = ctypes.c_int32()

    @pyqtSlot()
    def ps_scan(self):
        number_of_units = ctypes.c_int16(0)
        string_pointer = ctypes.c_char_p(b'0' * 256)
        string_size = ctypes.c_int16(256)
        self.status['enumerate'] = ps.ps4000aEnumerateUnits(ctypes.byref(number_of_units), string_pointer, ctypes.byref(string_size))
        assert_pico_ok(self.status['enumerate'])
        sns = string_pointer.value.decode('utf-8').split(',')
        self.finish(self.pico_scanned, {'serial_numbers': sns})

    @pyqtSlot(str)
    def ps_connect(self, sn):
        # open the oscilloscope
        try:
            string_pointer = ctypes.c_char_p(bytes(sn, 'utf-8'))
            self.status['openUnit'] = ps.ps4000aOpenUnit(ctypes.byref(self.device), string_pointer)
            assert_pico_ok(self.status['openUnit'])
            self.finish(self.pico_connected)
        except PicoSDKCtypesError:
            power_status = self.status['openUnit']
            if power_status == 286:
                self.status['changePowerSource'] = ps.ps4000aChangePowerSource(self.device, power_status)
            else:
                raise
            assert_pico_ok(self.status['changePowerSource'])
            print('not connected')

    @pyqtSlot()
    def ps_disconnect(self):
        # close the device
        self.status['close'] = ps.ps4000aCloseUnit(self.device)
        assert_pico_ok(self.status['close'])
        self.finish(self.pico_disconnected)

    @pyqtSlot()
    def ps_load(self):
        # select channel ranges and AC/DC coupling
        for channel in self.settings['channels'].keys():
            self.status[channel] = ps.ps4000aSetChannel(self.device, *self.settings['channels'][channel])
            assert_pico_ok(self.status[channel])

        # set up the trigger
        self.status['trigger'] = ps.ps4000aSetSimpleTrigger(self.device, *self.settings['trigger'])
        assert_pico_ok(self.status['trigger'])
        pre_trigger_samples = self.settings['preTriggerSamples']
        post_trigger_samples = self.settings['postTriggerSamples']
        self.maxSamples = pre_trigger_samples + post_trigger_samples

        # select timebase
        timebase = self.settings['timebase']
        self.status['getTimebase2'] = ps.ps4000aGetTimebase2(self.device, timebase, self.maxSamples,
                                                             ctypes.byref(self.timeIntervals),
                                                             ctypes.byref(self.returnedMaxSamples), 0)
        assert_pico_ok(self.status['getTimebase2'])
        self.finish(self.pico_loaded)

    @pyqtSlot()
    def ps_start(self):
        # read serial number
        string_pointer = ctypes.c_char_p(b'0' * 256)
        string_length = ctypes.c_int16(256)
        required_size = ctypes.c_int16(0)
        pico_info = 4
        self.status['info'] = ps.ps4000aGetUnitInfo(self.device, string_pointer, string_length, ctypes.byref(required_size), pico_info)
        serial_number = string_pointer.value.decode('utf-8')

        # start the oscilloscope
        self.status['runBlock'] = ps.ps4000aRunBlock(self.device, self.settings['preTriggerSamples'],
                                                     self.settings['postTriggerSamples'],
                                                     self.settings['timebase'], None, 0, None, None)
        assert_pico_ok(self.status['runBlock'])

        # wait until the oscilloscope is ready
        ready = ctypes.c_int16(0)
        check = ctypes.c_int16(0)
        while ready.value == check.value:
            self.status['isReady'] = ps.ps4000aIsReady(self.device, ctypes.byref(ready))

        # set up memory buffer
        buffer_dict_min = dict()
        self.buffer_dict_max = dict()
        for channel in self.settings['channels'].keys():
            if self.settings['channels'][channel][1] == 1:
                source = self.settings['channels'][channel][0]
                buffer_dict_min[channel] = (ctypes.c_int16 * self.maxSamples)()
                self.buffer_dict_max[channel] = (ctypes.c_int16 * self.maxSamples)()
                self.status[channel + 'data_buffer'] = ps.ps4000aSetDataBuffers(
                    self.device, source, ctypes.byref(self.buffer_dict_max[channel]),
                    ctypes.byref(buffer_dict_min[channel]), self.maxSamples, 0, 0)
                assert_pico_ok(self.status[channel + 'data_buffer'])

        # transfer the block of data from the oscilloscope
        overflow = ctypes.c_int16()
        self.cMaxSamples = ctypes.c_int32(self.maxSamples)
        self.status['getValues'] = ps.ps4000aGetValues(self.device, 0, ctypes.byref(self.cMaxSamples), 0, 0, 0, ctypes.byref(overflow))
        assert_pico_ok(self.status['getValues'])

        # stop the oscilloscope
        self.status['stop'] = ps.ps4000aStop(self.device)
        assert_pico_ok(self.status['stop'])

        # generate output dict
        self.data:  dict[str, dict] = dict()
        self.data['common'] = dict()
        tic = datetime.datetime.now()
        self.data['common']['datetime'] = tic.strftime('%d.%m.%Y %H:%M:%S.%f')
        self.data['common']['device'] = {'maxADC': self.maxADC.value, 'serial': serial_number}
        self.data['common']['time'] = {'length': self.cMaxSamples.value, 'interval': self.timeIntervals.value}
        self.data['common']['trigger'] = {'switch': self.triggerSwitch[self.settings['trigger'][0]],
                                          'source': self.channelNames[self.settings['trigger'][1]],
                                          'threshold': self.settings['trigger'][2],
                                          'direction': self.triggerDirection[self.settings['trigger'][3]],
                                          'delay': self.settings['trigger'][4] * self.timeIntervals.value,
                                          'auto_trigger': self.settings['trigger'][5],
                                          'pre_sample': self.settings['preTriggerSamples'],
                                          'post_sample': self.settings['postTriggerSamples']}
        for channel, data in self.buffer_dict_max.items():
            header = {'range': self.channelInputRanges[self.settings['channels'][channel][3]],
                      'offset': float(self.settings['channels'][channel][4]),
                      'coupling': self.channelCoupling[self.settings['channels'][channel][2]]}
            self.data[channel] = {'header': header, 'data': np.array(data, dtype='int16')}
        self.send_data.emit(self.data)

        self.finish(self.pico_measured)


class ChannelControlWidget(ThreadedWidget):
    value_changed = pyqtSignal()
    range_changed = pyqtSignal(str, str)

    def __init__(self, ch_name: str, font_size=14, color: Union[None, str] = None):
        super(ChannelControlWidget, self).__init__(font_size=font_size)
        self.name = ch_name
        self.setTitle(self.name)

        self.switch = QCheckBox('on/off')
        self.switch.setContentsMargins(0, 0, 0, 0)
        self.switch.stateChanged.connect(lambda: self.value_changed.emit())

        self.coupling = QMergedRadioButton(option_list=('DC', 'AC'), layout='h')
        self.coupling.setToolTip('signal coupling')
        self.coupling.option_changed.connect(lambda: self.value_changed.emit())
        self.coupling.layout().setContentsMargins(0, 0, 0, 0)
        self.coupling.layout().setSpacing(0)

        self.range_dict = {'10mV': ('±10 mV', 0), '20mV': ('±20 mV', 1), '50mV': ('±50 mV', 2), '100mV': ('±100 mV', 3),
                           '200mV': ('±200 mV', 4), '500mV': ('±500 mV', 5), '1V': ('±1 V', 6), '2V': ('±2 V', 7),
                           '5V': ('±5 V', 8), '10V': ('±10 V', 9), '20V': ('±20 V', 10), '50V': ('±50 V', 11)}
        self.range_dict_reversed = {y[0]: x for x, y in self.range_dict.items()}
        self.range = QMyComboBox([x[1][0] for x in self.range_dict.items()])
        self.range.setToolTip('signal range')
        self.range.currentTextChanged.connect(lambda: self.value_changed.emit())
        self.range.currentTextChanged.connect(lambda: self.range_changed.emit(self.name, self.get_range()))
        self.range.setContentsMargins(0, 0, 0, 0)

        self.offset = QMySpinBox(v_min=-25, v_max=25, decimals=3, step=0.001, suffix=' V')
        self.offset.setToolTip('signal offset')
        self.offset.valueChanged.connect(lambda: self.value_changed.emit())
        self.offset.setContentsMargins(0, 0, 0, 0)

        self.signal = QMyLineEdit(font_size=self.font_size)
        self.signal.setToolTip('signal name')
        self.signal.textChanged.connect(lambda: self.value_changed.emit())

        self.set_switch('on')
        self.set_coupling('DC')
        self.set_range('5V')
        self.set_offset(0)
        self.set_name(f'test name {self.name}')

        if color is not None:
            self.setAutoFillBackground(True)
            p = self.palette()
            p.setColor(self.backgroundRole(), QColor(color))
            self.setPalette(p)

        self.setFlat(True)

        layout = QVBoxLayout()
        layout_small = QMyHBoxLayout(self.switch, self.coupling)
        layout_small.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(layout_small)
        layout_small = QMyHBoxLayout(self.range, self.offset)
        layout_small.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(layout_small)
        layout.addWidget(self.signal)
        self.setLayout(layout)

    def set_switch(self, state: str):
        if state == 'on':
            self.switch.setChecked(True)
        elif state == 'off':
            self.switch.setChecked(False)
        else:
            raise ValueError('switch state should be "on" or "off"')

    def get_switch_ps(self):
        if self.switch.isChecked():
            return 1
        else:
            return 0

    def get_switch(self):
        if self.switch.isChecked():
            return 'on'
        else:
            return 'off'

    def set_coupling(self, coupling: str):
        if coupling in ('DC', 'AC'):
            self.coupling.set_option(coupling)
        else:
            raise ValueError('coupling should be DC or AC')

    def get_coupling_ps(self):
        if self.coupling.get_option() == 'DC':
            return 1
        elif self.coupling.get_option() == 'AC':
            return 0
        else:
            raise ValueError('coupling should be DC or AC. Error in QMergedRadioButton')

    def get_coupling(self):
        return self.coupling.get_option()

    def set_range(self, volt_range: str):
        if volt_range in self.range_dict.keys():
            self.range.setCurrentText(self.range_dict[volt_range][0])
        else:
            raise ValueError(f'voltage range should be in: {list(self.range_dict.keys())}')

    def get_range_ps(self):
        volt_range = self.range.currentText()
        return self.range_dict[self.range_dict_reversed[volt_range]][1]

    def get_range(self):
        return self.range_dict_reversed[self.range.currentText()]

    def set_offset(self, offset: float):
        self.offset.setValue(offset)

    def get_offset(self):
        return np.round(self.offset.value(), decimals=3)

    def set_name(self, name: str):
        self.signal.setText(name)

    def get_name(self):
        return self.signal.text()


class TimeControlWidget(QWidget):
    value_changed = pyqtSignal()
    timebase_changed = pyqtSignal(float)

    def __init__(self):
        super(TimeControlWidget, self).__init__()

        self.label = QCenteredLabel('timebase')

        self.interval = QMySpinBox(v_min=12.5, v_max=1e6, v_ini=12.5, decimals=1, suffix=' ns', step=12.5)
        self.interval.setToolTip('timebase')
        self.interval.valueChanged.connect(lambda: self.value_changed.emit())
        self.interval.editingFinished.connect(lambda: self.set_interval(self.interval.value()))

        self.duration = QMySpinBox(v_min=0.0125, v_max=1e6, decimals=2, suffix=' μs')
        self.duration.setToolTip('full duration')
        self.duration.valueChanged.connect(lambda: self.value_changed.emit())
        self.duration.editingFinished.connect(lambda: self.set_duration(self.duration.value()))

        self.pre_duration = QMySpinBox(v_min=0, v_max=1e6, decimals=2, suffix=' μs')
        self.pre_duration.setToolTip('pre time')
        self.pre_duration.valueChanged.connect(lambda: self.value_changed.emit())
        self.pre_duration.editingFinished.connect(lambda: self.set_pre_duration(self.pre_duration.value()))

        self.set_interval(100.0)
        self.set_duration(1000.0)
        self.set_pre_duration(500.0)

        layout = QMyHBoxLayout(self.label, self.interval, self.duration, self.pre_duration)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch(0)
        self.setLayout(layout)

    def set_interval(self, interval: float):
        self.interval.setValue((max(np.round(interval / 12.5), 1)) * 12.5)
        self.timebase_changed.emit(interval)
        self.set_duration(self.duration.value())
        self.set_pre_duration(self.pre_duration.value())

    def get_interval(self):
        return self.interval.value()

    def get_interval_ps(self):
        return int((self.interval.value() / 12.5) - 1)

    def set_duration(self, duration):
        self.duration.setValue(np.round(duration / (self.interval.value() / 1000)) * (self.interval.value() / 1000))

    def get_duration(self):
        return self.duration.value()

    def set_pre_duration(self, pre_time):
        self.pre_duration.setValue(np.round(pre_time / (self.interval.value() / 1000)) * (self.interval.value() / 1000))

    def get_pre_duration(self):
        return self.pre_duration.value()

    def get_pre_samples_ps(self):
        return int(np.round(self.pre_duration.value() / (self.interval.value() / 1000)))

    def get_post_samples_ps(self):
        return int(np.round((self.duration.value() - self.pre_duration.value()) / (self.interval.value() / 1000)))


class TriggerControlWidget(QWidget):
    value_changed = pyqtSignal()
    source_changed = pyqtSignal(str)

    def __init__(self):
        super(TriggerControlWidget, self).__init__()

        self.label = QCenteredLabel('trigger')

        self.switch = QCheckBox('on/off')
        self.switch.stateChanged.connect(lambda: self.value_changed.emit())

        # self.source_dict = {str(i+1): i for i in range(8)}
        self.source_dict = {str(i + 1): i for i in range(4)}
        self.range = 5000.0
        self.range_dict = {'10mV': 10, '20mV': 20, '50mV': 50, '100mV': 100, '200mV': 200, '500mV': 500, '1V': 1000,
                           '2V': 2000, '5V': 5000, '10V': 10000, '20V': 20000, '50V': 50000, '100V': 100000, '200V': 200000}
        self.source = QMyComboBox(self.source_dict.keys())
        self.source.setToolTip('trigger source')
        self.source.currentTextChanged.connect(lambda: self.value_changed.emit())
        self.source.currentTextChanged.connect(lambda: self.source_changed.emit(self.source.currentText()))

        self.threshold = QMySpinBox(v_min=-self.range, v_max=self.range, v_ini=0, decimals=0, step=1, suffix=' mV')
        self.threshold.setToolTip('threshold in voltage')
        self.threshold.valueChanged.connect(lambda: self.value_changed.emit())

        self.direction_dict = {'above': 0, 'below': 1, 'rising': 2, 'falling': 3, 'rising or falling': 4}
        self.direction = QMyComboBox(self.direction_dict.keys())
        self.direction.setToolTip('trigger direction')
        self.direction.currentTextChanged.connect(lambda: self.value_changed.emit())

        self.interval = 12.5
        self.delay = QMySpinBox(v_min=0, v_max=1e6-self.interval, v_ini=0, decimals=1, step=self.interval, suffix=' ns')
        self.delay.setToolTip('delay after trigger in ns')
        self.delay.valueChanged.connect(lambda: self.value_changed.emit())
        self.delay.editingFinished.connect(lambda: self.set_delay(self.delay.value()))

        self.auto_trigger = QMySpinBox(v_min=0, v_max=2**15-1, v_ini=1, decimals=0, step=1, suffix=' ms')
        self.auto_trigger.setToolTip('auto trigger time, if 0 waits indefinitely')
        self.auto_trigger.valueChanged.connect(lambda: self.value_changed.emit())

        self.set_switch('on')
        self.set_source('1')
        self.set_threshold(100)
        self.set_direction('rising')
        self.set_delay(0)
        self.set_auto_trigger(100)

        layout = QMyHBoxLayout(self.label, self.switch, self.source, self.threshold, self.direction, self.delay,
                               self.auto_trigger, alignment=Qt.AlignLeft)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch(0)
        self.setLayout(layout)

    def set_switch(self, state: str):
        if state == 'on':
            self.switch.setChecked(True)
        elif state == 'off':
            self.switch.setChecked(False)
        else:
            raise ValueError('switch state should be "on" of "off"')

    def get_switch(self):
        if self.switch.isChecked():
            return 'on'
        else:
            return 'off'

    def get_switch_ps(self):
        if self.switch.isChecked():
            return 1
        else:
            return 0

    @pyqtSlot(str, str)
    def set_range(self, channel: str, v_range: str):
        if channel == self.source.currentText():
            self.range = self.range_dict[v_range]
            self.threshold.setMaximum(self.range)
            self.threshold.setMinimum(-self.range)

    def set_source(self, source: str):
        if source in self.source_dict.keys():
            self.source.setCurrentText(source)
        else:
            raise ValueError(f'trigger source should be in: {list(self.source_dict.keys())}')

    def get_source(self):
        return self.source.currentText()

    def get_source_ps(self):
        return self.source_dict[self.source.currentText()]

    def set_threshold(self, threshold: int):
        if (threshold >= -self.range) and (threshold <= self.range):
            self.threshold.setValue(threshold)
        else:
            raise ValueError(f'threshold should be in [-{self.range:.0f}; {self.range:.0f}]')

    def get_threshold(self):
        return float(self.threshold.value())

    def get_threshold_ps(self):
        return int(self.threshold.value() / self.range * (2**15 - 1))

    def set_direction(self, direction):
        if direction in self.direction_dict.keys():
            self.direction.setCurrentText(direction)
        else:
            raise ValueError(f'trigger source should be in: {list(self.direction_dict.keys())}')

    def get_direction(self):
        return self.direction.currentText()

    def get_direction_ps(self):
        return self.direction_dict[self.direction.currentText()]

    @pyqtSlot(float)
    def set_interval(self, interval: float):
        self.interval = interval
        self.delay.setSingleStep(interval)
        self.delay.setMaximum(1e6 - interval)
        delay = self.delay.value()
        self.set_delay(delay)

    def set_delay(self, delay: float):
        self.delay.setValue((max(np.round(delay / self.interval), 0)) * self.interval)

    def get_delay(self):
        return self.delay.value()

    def get_delay_ps(self):
        return int(self.delay.value() / self.interval)

    def set_auto_trigger(self, delay: int):
        if (delay >= 0) and (delay < 2**16):
            self.auto_trigger.setValue(delay)
        else:
            raise ValueError('auto trigger delay should be greater or equal to zero and less than 2**16')

    def get_auto_trigger(self):
        return int(self.auto_trigger.value())


class PicoControlWidget(ThreadedWidget):
    pico_scan = pyqtSignal()
    pico_connect = pyqtSignal(str)
    pico_load = pyqtSignal()
    pico_start = pyqtSignal()
    pico_disconnect = pyqtSignal()

    plot_signals = pyqtSignal(dict)
    save_data = pyqtSignal(dict)
    send_data = pyqtSignal(dict, str)
    save_settings = pyqtSignal(str, dict)
    data_received = pyqtSignal()
    settings_loaded = pyqtSignal(dict)

    def __init__(self, channels: tuple, font_size=14):
        super(PicoControlWidget, self).__init__(font_size=font_size)

        self.setTitle('Controller')

        self.trig_auto_plot = True
        self.trig_auto_save = False
        self.data = dict()

        self.btn_scan = QMyStandardButton('scan', font_size=self.font_size)
        self.btn_scan.setToolTip('scan for PicoScopes')
        self.btn_scan.clicked.connect(self.ps_scan)
        self.combobox_sn = QComboBox()
        self.combobox_sn.setToolTip('serial numbers of PicoScopes connected to PC')
        self.combobox_sn.setMinimumContentsLength(14)
        self.combobox_sn.setEnabled(False)
        self.btn_connect = QMyStandardButton('connect', font_size=self.font_size)
        self.btn_connect.setToolTip('connect to a device')
        self.btn_connect.setEnabled(False)
        self.btn_connect.clicked.connect(self.ps_connect)
        self.btn_load = QMyStandardButton('load', font_size=self.font_size)
        self.btn_load.setToolTip('load settings into the device')
        self.btn_load.setEnabled(False)
        self.btn_load.clicked.connect(self.ps_load)
        self.btn_start = QMyStandardButton('start', font_size=self.font_size)
        self.btn_start.setToolTip('start the device')
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.ps_start)
        self.btn_disconnect = QMyStandardButton('disconnect', font_size=self.font_size)
        self.btn_disconnect.setToolTip('disconnect from the device')
        self.btn_disconnect.setEnabled(False)
        self.btn_disconnect.clicked.connect(self.ps_disconnect)
        self.btn_connect.setMaximumWidth(max(self.btn_connect.width(), self.btn_disconnect.width()))
        self.btn_disconnect.setMaximumWidth(max(self.btn_connect.width(), self.btn_disconnect.width()))

        self.switch_auto_start = QCheckBox('auto_start')
        self.switch_auto_start.setChecked(False)
        self.delay_auto_start = QMySpinBox(v_min=0.1, v_max=600, v_ini=1, decimals=1, step=1, suffix=' s')
        self.delay_auto_start.setToolTip('interval for auto start')
        self.timer_auto_start = QTimer()
        self.timer_auto_start.setInterval(int(self.delay_auto_start.value() * 1e3))
        self.timer_auto_start.setSingleShot(True)
        self.timer_auto_start.timeout.connect(self.ps_auto_start)
        self.delay_auto_start.valueChanged.connect(self.auto_start_delay_changed)
        self.bar_auto_start = QProgressBar()
        self.bar_auto_start.setFormat('%v ms')
        self.bar_auto_start.setMinimum(0)
        self.bar_auto_start.setMaximum(int(self.delay_auto_start.value() * 1e3))
        self.bar_auto_start.setMinimumWidth(200)
        self.timer_progress_bar = QTimer()
        self.timer_progress_bar.setInterval(10)
        self.timer_progress_bar.timeout.connect(self.update_progress_bar)
        self.timer_progress_bar.start()

        self.time_control = TimeControlWidget()
        self.time_control.value_changed.connect(lambda: self.btn_start.setEnabled(False))

        self.trigger_control = TriggerControlWidget()
        self.trigger_control.value_changed.connect(lambda: self.btn_start.setEnabled(False))

        self.time_control.timebase_changed.connect(self.trigger_control.set_interval)
        self.trigger_control.source_changed.connect(self.set_trigger_range)

        self.channel_names = channels
        self.channel_widgets = []
        # noinspection SpellCheckingInspection
        self.color_cycle = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
        for name in self.channel_names:
            ch_widget = ChannelControlWidget(name, font_size=self.font_size, color=next(self.color_cycle))
            ch_widget.range_changed.connect(self.trigger_control.set_range)
            ch_widget.value_changed.connect(lambda: self.btn_start.setEnabled(False))
            self.channel_widgets.append(ch_widget)

        self.settings = dict()
        self.pico_settings: dict[str, [dict, list, int]] = dict()
        self.pico_settings['channels'] = dict()
        for ch, i in zip(self.channel_names, range(len(self.channel_names))):
            self.pico_settings['channels'][ch] = list(range(5))
            self.pico_settings['channels'][ch][0] = i
        self.pico_settings['trigger'] = list(range(6))

        self.pico_worker = PicoControlWorker(self.thread())
        self.pico_thread = None
        self.pico_scan.connect(self.pico_worker.ps_scan)
        self.pico_worker.pico_scanned.connect(lambda: self.btn_scan.setEnabled(True))
        self.pico_worker.pico_scanned.connect(lambda: self.btn_connect.setEnabled(True))
        self.pico_worker.pico_scanned.connect(lambda: self.combobox_sn.setEnabled(True))
        self.pico_worker.pico_scanned.connect(self.scanned)
        self.pico_worker.pico_connected.connect(lambda: self.btn_disconnect.setEnabled(True))
        self.pico_worker.pico_connected.connect(lambda: self.btn_load.setEnabled(True))
        self.pico_worker.pico_loaded.connect(lambda: self.btn_disconnect.setEnabled(True))
        self.pico_worker.pico_loaded.connect(lambda: self.btn_load.setEnabled(True))
        self.pico_worker.pico_loaded.connect(lambda: self.btn_start.setEnabled(True))
        self.pico_worker.pico_measured.connect(self.ps_measured)
        self.pico_worker.pico_disconnected.connect(lambda: self.btn_scan.setEnabled(True))
        self.pico_worker.pico_disconnected.connect(lambda: self.combobox_sn.setEnabled(True))
        self.pico_worker.pico_disconnected.connect(lambda: self.btn_connect.setEnabled(True))

        self.pico_worker.send_data.connect(self.transfer_data)
        self.pico_connect.connect(self.pico_worker.ps_connect)
        self.pico_load.connect(self.pico_worker.ps_load)
        self.pico_start.connect(self.pico_worker.ps_start)
        self.pico_disconnect.connect(self.pico_worker.ps_disconnect)

        self.plot_worker = SignalConstructor(self.thread())
        self.plot_thread = None
        self.plot_worker.send_signals.connect(self.plot_signals)
        self.send_data.connect(self.plot_worker.adc2mv)

        layout = QMyVBoxLayout()
        layout_control = QMyHBoxLayout(self.btn_scan, self.combobox_sn, self.btn_connect, self.btn_disconnect,
                                       self.btn_load, self.btn_start)
        layout_control.addStretch(0)
        layout.addLayout(layout_control)
        layout.addWidget(self.time_control, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.trigger_control, alignment=Qt.AlignmentFlag.AlignLeft)
        layout_auto = QMyHBoxLayout(self.switch_auto_start, self.delay_auto_start, self.bar_auto_start)
        layout_auto.addStretch(0)
        layout.addLayout(layout_auto)
        layout_channels = QMyHBoxLayout(*self.channel_widgets)
        layout.addLayout(layout_channels)
        self.setLayout(layout)

    def block_control(self):
        self.btn_scan.setEnabled(False)
        self.combobox_sn.setEnabled(False)
        self.btn_connect.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.btn_disconnect.setEnabled(False)

    def ps_scan(self):
        self.block_control()
        self.pico_thread = QThread()
        self.start_branch(self.pico_worker, self.pico_thread, self.pico_scan)

    @pyqtSlot(dict)
    def scanned(self, info):
        items = [self.combobox_sn.itemText(i) for i in range(self.combobox_sn.count())]
        for sn in info['serial_numbers']:
            if sn not in items:
                self.combobox_sn.addItem(sn)

    def ps_connect(self):
        self.block_control()
        self.pico_thread = QThread()
        self.start_branch(self.pico_worker, self.pico_thread, self.pico_connect, self.combobox_sn.currentText())

    def ps_disconnect(self):
        self.block_control()
        self.pico_thread = QThread()
        self.start_branch(self.pico_worker, self.pico_thread, self.pico_disconnect)

    @pyqtSlot()
    def ps_measured(self):
        self.btn_disconnect.setEnabled(True)
        self.btn_load.setEnabled(True)
        self.btn_start.setEnabled(True)
        if self.switch_auto_start.isChecked():
            self.timer_auto_start.start()

    @pyqtSlot()
    def ps_auto_start(self):
        if self.switch_auto_start.isChecked() and self.btn_start.isEnabled():
            self.ps_start()

    def refresh_settings_ps(self):
        for ch_widget in self.channel_widgets:
            self.pico_settings['channels'][ch_widget.name][1] = ch_widget.get_switch_ps()
            self.pico_settings['channels'][ch_widget.name][2] = ch_widget.get_coupling_ps()
            self.pico_settings['channels'][ch_widget.name][3] = ch_widget.get_range_ps()
            self.pico_settings['channels'][ch_widget.name][4] = ch_widget.get_offset()
        self.pico_settings['timebase'] = self.time_control.get_interval_ps()
        self.pico_settings['preTriggerSamples'] = self.time_control.get_pre_samples_ps()
        self.pico_settings['postTriggerSamples'] = self.time_control.get_post_samples_ps()
        self.pico_settings['trigger'][0] = self.trigger_control.get_switch_ps()
        self.pico_settings['trigger'][1] = self.trigger_control.get_source_ps()
        self.pico_settings['trigger'][2] = self.trigger_control.get_threshold_ps()
        self.pico_settings['trigger'][3] = self.trigger_control.get_direction_ps()
        self.pico_settings['trigger'][4] = self.trigger_control.get_delay_ps()
        self.pico_settings['trigger'][5] = self.trigger_control.get_auto_trigger()

    def refresh_settings(self):
        self.settings = dict()
        self.settings['common'] = dict()
        self.settings['common']['time'] = {'interval': self.time_control.get_interval(),
                                           'duration': self.time_control.get_duration(),
                                           'pre_duration': self.time_control.get_pre_duration()}
        self.settings['common']['trigger'] = {'switch': self.trigger_control.get_switch(),
                                              'source': self.trigger_control.get_source(),
                                              'threshold': self.trigger_control.get_threshold(),
                                              'direction': self.trigger_control.get_direction(),
                                              'delay': self.trigger_control.get_delay(),
                                              'auto_trigger': self.trigger_control.get_auto_trigger()}
        self.settings['common']['auto_start'] = self.switch_auto_start.isChecked()
        self.settings['common']['auto_start_delay'] = float(self.delay_auto_start.value())
        for channel, widget in zip(self.channel_names, self.channel_widgets):
            self.settings[channel] = {'switch': widget.get_switch(), 'coupling': widget.get_coupling(),
                                      'range': widget.get_range(), 'offset': widget.get_offset(),
                                      'name': widget.get_name()}

    @pyqtSlot(dict)
    def load_settings(self, settings):
        self.settings = settings
        self.time_control.set_interval(self.settings['common']['time']['interval'])
        self.time_control.set_duration(self.settings['common']['time']['duration'])
        self.time_control.set_pre_duration(self.settings['common']['time']['pre_duration'])
        self.trigger_control.set_interval(self.settings['common']['time']['interval'])
        self.trigger_control.set_switch(self.settings['common']['trigger']['switch'])
        self.trigger_control.set_source(self.settings['common']['trigger']['source'])
        for channel, widget in zip(self.channel_names, self.channel_widgets):
            widget.set_switch(self.settings[channel]['switch'])
            widget.set_coupling(self.settings[channel]['coupling'])
            widget.set_range(self.settings[channel]['range'])
            widget.set_offset(self.settings[channel]['offset'])
            widget.set_name(self.settings[channel]['name'])
        self.set_trigger_range(self.settings['common']['trigger']['source'])
        self.trigger_control.set_threshold(self.settings['common']['trigger']['threshold'])
        self.trigger_control.set_direction(self.settings['common']['trigger']['direction'])
        self.trigger_control.set_delay(self.settings['common']['trigger']['delay'])
        self.trigger_control.set_auto_trigger(self.settings['common']['trigger']['auto_trigger'])
        self.switch_auto_start.setChecked(self.settings['common']['auto_start'])
        self.delay_auto_start.setValue(self.settings['common']['auto_start_delay'])
        self.settings_loaded.emit(self.settings)

    def ps_load(self):
        self.block_control()
        self.refresh_settings()
        self.settings_loaded.emit(self.settings)
        self.refresh_settings_ps()
        self.pico_worker.settings = self.pico_settings
        self.pico_thread = QThread()
        self.start_branch(self.pico_worker, self.pico_thread, self.pico_load)

    def ps_start(self):
        self.block_control()
        self.pico_thread = QThread()
        self.start_branch(self.pico_worker, self.pico_thread, self.pico_start)

    @pyqtSlot(dict)
    def transfer_data(self, data: dict):
        self.data_received.emit()
        channels = set(data.keys()) - {'common'}
        channels = sorted(list(channels))
        for ch in channels:
            data[ch]['header']['name'] = self.channel_widgets[int(ch) - 1].signal.text()
        self.data = data
        if self.trig_auto_save:
            self.save_data.emit(data)
        if self.trig_auto_plot:
            self.plot_thread = QThread()
            self.start_branch(self.plot_worker, self.plot_thread, self.send_data, data, 'channels')

    @pyqtSlot()
    def provide_data(self):
        self.save_data.emit(self.data)

    @pyqtSlot(bool)
    def set_auto_save(self, state: bool):
        self.trig_auto_save = state

    @pyqtSlot(str)
    def provide_settings(self, path: str):
        self.refresh_settings()
        self.save_settings.emit(path, self.settings)

    @pyqtSlot(str)
    def set_trigger_range(self, source):
        num = int(source) - 1
        v_range = self.channel_widgets[num].get_range()
        self.trigger_control.set_range(source, v_range)

    @pyqtSlot()
    def data_saved(self):
        if self.btn_start.isEnabled() and self.switch_auto_start.isChecked():
            self.ps_start()

    @pyqtSlot()
    def update_progress_bar(self):
        if self.timer_auto_start.remainingTime() == -1:
            self.bar_auto_start.setValue(0)
        else:
            self.bar_auto_start.setValue(int(self.timer_auto_start.remainingTime()))

    @pyqtSlot(float)
    def auto_start_delay_changed(self, x):
        self.timer_auto_start.setInterval(int(x * 1e3))
        self.bar_auto_start.setMaximum(int(x * 1e3))


class PyQtPlotterWorker(ThreadedWorker):
    plot_updated = pyqtSignal()

    def __init__(self, thread: QThread):
        super(PyQtPlotterWorker, self).__init__(thread)

    @pyqtSlot(dict, dict)
    def plot_signals(self, curves: dict, signals: dict):
        channels = set(signals.keys()) - {'time'}
        channels = sorted(list(channels))
        for ch in curves.keys():
            curves[ch].setVisible(False)
        for ch in curves.keys():
            curves[ch].clear()
        for ch in channels:
            curves[ch].setData(signals['time'], signals[ch])
            curves[ch].setVisible(True)
            curves[ch].update()
        self.finish(self.plot_updated)


class PyQtPlotterWidget(ThreadedWidget):
    send_signals = pyqtSignal(dict, dict)

    def __init__(self, channels, font_size=14):
        super(PyQtPlotterWidget, self).__init__(font_size=font_size)

        self.setTitle('Plotter')

        self.channels = channels
        self.range_dict = {'10mV': 0.01, '20mV': 0.02, '50mV': 0.05, '100mV': 0.1, '200mV': 0.2, '500mV': 0.5, '1V': 1,
                           '2V': 2, '5V': 5, '10V': 10, '20V': 20, '50V': 50, '100V': 100, '200V': 200}

        self.canvas = GraphicsLayoutWidget()
        self.canvas.setBackground('w')
        self.canvas.useOpenGL(True)

        self.ax = self.canvas.addPlot()
        self.box = self.ax.getViewBox()
        self.box.installEventFilter(self)
        self.ax.setLabel('left', 'voltage', units='V', **{'color': '#FFF', 'font-size': '32px'})
        self.ax.setLabel('bottom', 'time', units='s', **{'color': '#FFF', 'font-size': '32px'})

        font = QFont('Arial', self.font_size)

        x_axis = self.ax.getAxis('bottom')
        x_axis.installEventFilter(self)
        x_axis.setTextPen('k')
        x_axis.setTickFont(font)

        y_axis = self.ax.getAxis('left')
        y_axis.installEventFilter(self)
        y_axis.setTextPen('k')
        y_axis.setTickFont(font)

        self.ax.setXRange(0, 2e-2)
        self.ax.setYRange(-5, 5)

        self.ax.showGrid(x=True, y=True, alpha=1.0)

        # noinspection SpellCheckingInspection
        self.color_cycle = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
        self.curves = dict((str(i), PlotCurve(linked_curve=self.ax.plot(pen=mkPen(next(self.color_cycle), width=3)))) for i in range(1, 9))
        for ch in self.curves.keys():
            self.curves[ch].setDownsampling(ds=True, auto=True, method='peak')
            self.curves[ch].clear()

        self.plot_worker = PyQtPlotterWorker(self.thread())
        self.plot_thread = None
        self.send_signals.connect(self.plot_worker.plot_signals)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    @pyqtSlot(dict)
    def plot_signals(self, signals):
        self.plot_thread = QThread()
        self.start_branch(self.plot_worker, self.plot_thread, self.send_signals, self.curves, signals)

    @pyqtSlot(dict)
    def load_settings(self, settings):
        self.ax.setXRange(0, settings['common']['time']['duration'] / 1e6)
        channels = set(settings.keys()) - {'common'}
        ranges = list(map(lambda x: settings[x]['range'], channels))
        ranges = list(map(lambda x: self.range_dict[x], ranges))
        max_range = max(ranges)
        self.ax.setYRange(-max_range, max_range)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.GraphicsSceneWheel:
            return True
        elif event.type() == QEvent.GraphicsSceneMouseDoubleClick:
            return True
        elif event.type() == QEvent.GraphicsSceneMousePress:
            return True
        return super().eventFilter(watched, event)


class SaverWorker(ThreadedWorker):
    saved = pyqtSignal()

    def __init__(self, thread):
        super(SaverWorker, self).__init__(thread)

    @pyqtSlot(str, dict)
    def save_signals(self, path: str, data: dict):
        np.savez_compressed(path, data=data)
        self.finish(self.saved)


class SaverWidget(ThreadedWidget):
    send_data = pyqtSignal(str, dict)
    claim_data = pyqtSignal()
    auto_save_changed = pyqtSignal(bool)
    claim_settings = pyqtSignal(str)
    load_settings = pyqtSignal(dict)
    data_saved = pyqtSignal()

    def __init__(self, font_size=14):
        super(SaverWidget, self).__init__(font_size=font_size)

        self.setTitle('Saver')

        self.btn_save_settings = QMyStandardButton('save settings', font_size=self.font_size)
        self.btn_save_settings.clicked.connect(self.save_manual_settings)
        self.btn_load_settings = QMyStandardButton('load settings', font_size=self.font_size)
        self.btn_load_settings.clicked.connect(self.load_manual_settings)

        self.btn_save_defaults = QMyStandardButton('save defaults', font_size=self.font_size)
        self.btn_save_defaults.clicked.connect(lambda: self.claim_settings.emit(
            os.path.join(path_data_local, 'pico_bank', 'settings', 'defaults.json')))
        self.btn_load_defaults = QMyStandardButton('load defaults', font_size=self.font_size)
        self.btn_load_defaults.clicked.connect(self.load_defaults)

        self.path = os.path.join(os.getcwd(), 'temp')
        self.line_path = QMyLineEdit(font_size=self.font_size)
        self.line_path.setReadOnly(True)
        self.line_path.setText(self.path)

        self.btn_choose_path = QMyStandardButton('choose path', font_size=self.font_size)
        self.btn_choose_path.clicked.connect(self.choose_path)

        self.btn_save_data = QMyStandardButton('save data', font_size=self.font_size)
        self.btn_save_data.clicked.connect(self.manual_save)

        self.switch_auto_save = QCheckBox('auto_save')
        self.switch_auto_save.setChecked(False)
        self.switch_auto_save.stateChanged.connect(lambda: self.auto_save_changed.emit(bool(self.switch_auto_save.isChecked())))

        self.spin_box_exp_num = QMySpinBox(v_min=1, v_max=99999, v_ini=1, decimals=0, step=1, prefix='#', suffix='')
        self.spin_box_exp_num.setFont((QFont('Arial', self.font_size)))

        self.location_line = QMyLineEdit(font_size=self.font_size)
        self.location_line.setToolTip('location of device')
        self.location_line.setText('test location')

        self.target_line = QMyLineEdit(font_size=self.font_size)
        self.target_line.setToolTip('target of beam')
        self.target_line.setText('test target')

        self.misc_line = QMyLineEdit(font_size=self.font_size)
        self.misc_line.setToolTip('miscellaneous comment')
        self.misc_line.setText('test comment')

        self.worker = SaverWorker(self.thread())
        self.branch = None
        self.worker.saved.connect(self.finish_saving)
        self.send_data.connect(self.worker.save_signals)

        layout = QVBoxLayout()
        small_layout = QMyHBoxLayout(self.btn_save_settings, self.btn_load_settings, self.btn_choose_path)
        small_layout.addStretch(0)
        layout.addLayout(small_layout)
        small_layout = QMyHBoxLayout(self.btn_save_defaults, self.btn_load_defaults, self.line_path)
        layout.addLayout(small_layout)
        small_layout = QMyHBoxLayout(self.spin_box_exp_num, self.btn_save_data, self.switch_auto_save)
        small_layout.addStretch(0)
        layout.addLayout(small_layout)
        small_layout = QMyHBoxLayout(self.location_line, self.target_line, self.misc_line)
        layout.addLayout(small_layout)
        layout.addStretch(0)
        self.setLayout(layout)

    @pyqtSlot(dict)
    def save_data(self, data: dict):
        if len(data) > 0:
            self.lock_saver()
            os.makedirs(self.path, exist_ok=True)
            save_path = os.path.join(self.path, f'{self.spin_box_exp_num.value():.0f}.npz')
            data['common']['name'] = f'{str(datetime.datetime.now().year)[-2:]}_{self.spin_box_exp_num.value():.0f}'
            data['common']['location'] = self.location_line.text()
            data['common']['target'] = self.target_line.text()
            data['common']['miscellaneous'] = self.misc_line.text()
            self.branch = QThread()
            self.start_branch(self.worker, self.branch, self.send_data, save_path, data)
        else:
            self.unlock_saver()

    @pyqtSlot()
    def finish_saving(self):
        self.unlock_saver()
        self.spin_box_exp_num.setValue(self.spin_box_exp_num.value() + 1)
        self.data_saved.emit()

    @pyqtSlot()
    def manual_save(self):
        self.lock_saver()
        self.claim_data.emit()

    def lock_saver(self):
        self.spin_box_exp_num.setEnabled(False)
        self.btn_save_data.setEnabled(False)
        # self.switch_auto_start.setEnabled(False)
        self.switch_auto_save.setEnabled(False)
        self.location_line.setEnabled(False)
        self.target_line.setEnabled(False)
        self.misc_line.setEnabled(False)

    def unlock_saver(self):
        self.spin_box_exp_num.setEnabled(True)
        self.btn_save_data.setEnabled(True)
        # self.switch_auto_start.setEnabled(True)
        self.switch_auto_save.setEnabled(True)
        self.location_line.setEnabled(True)
        self.target_line.setEnabled(True)
        self.misc_line.setEnabled(True)

    @pyqtSlot(str, dict)
    def save_settings(self, path: str, settings: dict):
        settings['common']['path'] = self.line_path.text()
        settings['common']['location'] = self.location_line.text()
        settings['common']['target'] = self.target_line.text()
        settings['common']['miscellaneous'] = self.misc_line.text()
        settings['common']['auto_save'] = self.switch_auto_save.isChecked()
        options = jsbeautifier.default_options()
        options.indent_size = 4
        with open(path, 'w+') as f:
            f.write(jsbeautifier.beautify(json.dumps(settings), options))

    @pyqtSlot()
    def load_defaults(self):
        with open(os.path.join(path_data_local, 'pico_bank', 'settings', 'defaults.json'), 'r') as file:
            settings = json.load(file)
        self.set_settings(settings)
        # if not os.path.isdir(self.path):  # create dir if it doesn't exist
        #     os.makedirs(self.path)

    @pyqtSlot()
    def save_manual_settings(self):
        path = QFileDialog.getSaveFileName(self, 'Choose save path', directory=os.path.join(self.path, 'settings'),
                                           filter='*.json')[0]
        if path != '':
            self.claim_settings.emit(path)

    @pyqtSlot()
    def load_manual_settings(self):
        path = QFileDialog.getOpenFileName(self, 'Choose file', directory=os.path.join(self.path, 'settings'),
                                           filter='*.json')[0]
        if path != '':
            with open(path, 'r') as file:
                settings = json.load(file)
            self.set_settings(settings)

    def set_settings(self, settings):
        self.path = settings['common']['path']
        self.line_path.setText(self.path)
        self.location_line.setText(settings['common']['location'])
        self.target_line.setText(settings['common']['target'])
        self.misc_line.setText(settings['common']['miscellaneous'])
        self.switch_auto_save.setChecked(settings['common']['auto_save'])
        self.set_next_num()
        self.load_settings.emit(settings)

    @pyqtSlot()
    def data_received(self):
        pass

    @pyqtSlot()
    def choose_path(self):
        path = QFileDialog.getExistingDirectory(self, 'Choose directory', directory=self.path)
        if path != '':
            self.path = path
            self.line_path.setText(self.path)
            self.set_next_num()

    def set_next_num(self):
        tmp_list = os.listdir(self.path)
        tmp_list = list(filter(lambda x: x.endswith('.npz'), tmp_list))
        tmp_list = list(map(lambda x: os.path.splitext(x)[0], tmp_list))
        # tmp_list = list(filter)  # filter by possibility of converting to int
        tmp_list = list(map(lambda x: int(x), tmp_list))
        tmp_list = sorted(tmp_list)
        if len(tmp_list) == 0:
            self.spin_box_exp_num.setValue(1)
        else:
            self.spin_box_exp_num.setValue(tmp_list[-1] + 1)

    @pyqtSlot(bool, name='ChangeAutoSave')
    def change_auto_save(self, s):
        self.switch_auto_save.setChecked(s)


class MainWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.font_size = 14
        # self.channels = tuple('12345678')
        self.channels = tuple('1234')

        self.controller = PicoControlWidget(self.channels, font_size=self.font_size)
        self.saver = SaverWidget(font_size=self.font_size)
        self.plotter = PyQtPlotterWidget(self.channels, font_size=self.font_size)

        self.controller.plot_signals.connect(self.plotter.plot_signals)
        self.controller.save_data.connect(self.saver.save_data)
        self.saver.claim_data.connect(self.controller.provide_data)
        self.saver.auto_save_changed.connect(self.controller.set_auto_save)
        self.saver.claim_settings.connect(self.controller.provide_settings)
        self.controller.save_settings.connect(self.saver.save_settings)
        self.saver.load_settings.connect(self.controller.load_settings)
        self.controller.data_received.connect(self.saver.data_received)
        # self.saver.data_saved.connect(self.controller.data_saved)
        self.controller.settings_loaded.connect(self.plotter.load_settings)

        self.saver.load_defaults()

        layout = QVBoxLayout()
        layout_small = QMyHBoxLayout(self.controller, self.saver)
        layout_small.addStretch(0)
        layout.addLayout(layout_small)
        layout.addWidget(self.plotter)
        self.setLayout(layout)


class MainWindow(MyStandardWindow):
    def __init__(self):
        super().__init__()
        self.pico_widget = MainWidget()
        font = self.font()
        font.setPointSize(14)
        self.pico_widget.setFont(font)
        self.pico_widget.font_size = 14
        self.appear_with_central_widget('PICO control', self.pico_widget)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec())
