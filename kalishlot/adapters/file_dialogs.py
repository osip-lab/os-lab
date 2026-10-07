"""Windows' own open / save-as windows, on THIS PC.

kalishlot's server runs on the lab PC, and a browser viewer is usually on that
same PC, so the dialogs are opened by the server (tkinter, on top of the
browser) rather than by the page, which could only hand back a file's name and
contents, never its path. The same approach as the pipeline box's folder
"browse". Each call blocks until the window is closed and returns the chosen
path, or None when it was cancelled. One window at a time.
"""

import threading
from pathlib import Path

_lock = threading.Lock()


def _dialog(kind, title, initial_dir, filetypes, default_extension, initial_name):
    import tkinter
    from tkinter import filedialog
    if not _lock.acquire(blocking=False):
        raise ValueError('a file window is already open on the lab PC')
    try:
        root = tkinter.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        try:
            start = str(Path(initial_dir).expanduser()) if initial_dir else ''
            if not start or not Path(start).is_dir():
                start = str(Path.home())
            options = dict(parent=root, title=title, initialdir=start,
                           filetypes=filetypes)
            if kind == 'save':
                chosen = filedialog.asksaveasfilename(
                    defaultextension=default_extension,
                    initialfile=initial_name or '', **options)
            else:
                chosen = filedialog.askopenfilename(**options)
        finally:
            root.destroy()
    finally:
        _lock.release()
    return str(Path(chosen)) if chosen else None


def ask_open_file(title='Open', initial_dir=None,
                  filetypes=(('All files', '*.*'),)):
    return _dialog('open', title, initial_dir, filetypes, None, None)


def ask_save_file(title='Save as', initial_dir=None,
                  filetypes=(('All files', '*.*'),), default_extension='',
                  initial_name=None):
    return _dialog('save', title, initial_dir, filetypes, default_extension,
                   initial_name)
