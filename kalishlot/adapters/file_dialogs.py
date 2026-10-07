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


CAPTURE_FILETYPES = (
    ('Scope capture or frames', '*.npz *.npy *.mkv *.avi *.mp4'),
    ('All files', '*.*'))


def ask_capture(initial_dir=None):
    """Pick a capture to show: a FOLDER (a synced video + scope capture) or a
    FILE (one scope recording, or a video / frames file). Windows' own windows
    cannot return either in one go, so a small window asks which first. Returns
    the chosen path, or None when cancelled."""
    import tkinter
    from tkinter import filedialog
    if not _lock.acquire(blocking=False):
        raise ValueError('a file window is already open on the lab PC')
    chosen = ''
    try:
        root = tkinter.Tk()
        root.title('Show a capture')
        root.attributes('-topmost', True)
        start = str(Path(initial_dir).expanduser()) if initial_dir else ''
        if not start or not Path(start).is_dir():
            start = str(Path.home())

        def pick(kind):
            nonlocal chosen
            root.withdraw()
            options = dict(parent=root, initialdir=start)
            if kind == 'folder':
                chosen = filedialog.askdirectory(
                    title='Synced capture folder', mustexist=True, **options)
            else:
                chosen = filedialog.askopenfilename(
                    title='Scope recording or video file',
                    filetypes=CAPTURE_FILETYPES, **options)
            if chosen:
                root.destroy()
            else:
                root.deiconify()        # cancelled the inner window: ask again

        tkinter.Label(root, padx=20, pady=10, justify='left', text=(
            'A folder opens the synced video + scope viewer.\n'
            'A file opens one scope recording (.npz) or a video.')).pack()
        row = tkinter.Frame(root, pady=8)
        row.pack()
        tkinter.Button(row, text='Folder…', width=14,
                       command=lambda: pick('folder')).pack(side='left', padx=6)
        tkinter.Button(row, text='File…', width=14,
                       command=lambda: pick('file')).pack(side='left', padx=6)
        tkinter.Button(row, text='Cancel', width=8,
                       command=root.destroy).pack(side='left', padx=6)
        root.mainloop()
    finally:
        _lock.release()
    return str(Path(chosen)) if chosen else None
