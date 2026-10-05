"""Save and load a mode-video frame stack as a compressed video.

A capture used to write its frames as a raw .npy - 20 MB for 120 frames of a
308x280 sensor. Two video codecs replace it, both through the ffmpeg binary:

    'h264'      8-bit H.264, the codec the Ximea app saves its .avi with.
                About 200x smaller than the raw stack, but lossy: a 12-bit
                stack is rescaled to 0-255 first (see below) and the codec adds
                a little noise on top - measured at about 8 counts rms of 3010.
    'lossless'  FFV1, bit-exact at the stack's own depth. About 3x smaller.

Files written before this existed are .npy and still load: `load_frames`
dispatches on the file's extension.

The scale. H.264 holds 8 bits, so a stack with values above 255 is divided by
`scale` = peak / 255 before encoding and multiplied back on load. The loaded
stack therefore has the original dtype and approximately the original counts,
so thresholds and saturation checks in the analysis keep their meaning. A stack
that already fits in 8 bits has scale 1 and loses nothing to rescaling.
`scale` is returned by `save_frames` and has to be stored with the capture.

The loaded stack is an ordinary in-memory array, not a memory map.
"""

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np

FORMATS = ('h264', 'lossless')
_SUFFIX = {'h264': '.mp4', 'lossless': '.mkv'}
H264_CRF = 18                    # visually transparent; 10 was 5x larger and
                                 # no more accurate (the 8-bit step dominates)
_FALLBACK_FFMPEG = Path(r'C:\Users\OsipLab\ffmpeg\bin\ffmpeg.exe')


def find_ffmpeg():
    """The ffmpeg executable: on PATH, else the lab PC's own copy."""
    found = shutil.which('ffmpeg')
    if found:
        return found
    if _FALLBACK_FFMPEG.exists():
        return str(_FALLBACK_FFMPEG)
    raise RuntimeError('ffmpeg not found. Compressed frames need it: put '
                       'ffmpeg on PATH.')


def _run(args, data, what):
    result = subprocess.run([find_ffmpeg(), '-hide_banner', '-loglevel', 'error',
                             *args], input=data, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f'ffmpeg failed to {what}: '
                           f'{result.stderr.decode(errors="replace").strip()}')
    return result.stdout


def _raw_pix_fmt(dtype):
    dtype = np.dtype(dtype)
    if dtype == np.uint8:
        return 'gray'
    if dtype == np.uint16:
        return 'gray16le'
    raise ValueError(f'frames of dtype {dtype} cannot be stored as video '
                     f'(uint8 and uint16 only)')


def save_frames(stem_path, frames, fmt='h264', fps=30.0):
    """Write `frames` (n, h, w) next to `stem_path` and return the details.

    `stem_path` has no suffix; the codec's own is added. Returns
    (path, info) where info = {'format', 'scale'} is what `load_frames` needs
    besides the file - store it in the session record. `fps` only sets the
    playback speed of the file.
    """
    if fmt not in FORMATS:
        raise ValueError(f'frame format {fmt!r} is not one of {FORMATS}')
    frames = np.ascontiguousarray(frames)
    if frames.ndim != 3:
        raise ValueError(f'expected frames shaped (n, h, w), got {frames.shape}')
    n, h, w = frames.shape
    path = Path(str(stem_path) + _SUFFIX[fmt])

    scale = 1.0
    if fmt == 'h264':
        peak = int(frames.max()) if frames.size else 0
        if peak > 255:
            scale = peak / 255.0
            frames8 = np.rint(frames.astype(np.float32) / scale).astype(np.uint8)
        else:
            frames8 = frames.astype(np.uint8)
        source, pix_fmt = frames8, 'gray'
        # yuv420p wants even dimensions, so the frame is padded and the
        # padding cropped off again on load
        codec = ['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                 '-c:v', 'libx264', '-crf', str(H264_CRF),
                 '-pix_fmt', 'yuv420p']
    else:
        source, pix_fmt = frames, _raw_pix_fmt(frames.dtype)
        codec = ['-c:v', 'ffv1', '-level', '3', '-pix_fmt', pix_fmt]

    _run(['-y', '-f', 'rawvideo', '-pix_fmt', pix_fmt, '-s', f'{w}x{h}',
          '-framerate', f'{fps:g}', '-i', 'pipe:0', *codec, str(path)],
         source.tobytes(), f'write {path.name}')
    return path, {'format': fmt, 'scale': scale}


def load_frames(path, shape, dtype, scale=1.0):
    """Read a stack written by `save_frames`, or a legacy .npy.

    `shape` and `dtype` are the original stack's, as stored in the session
    record; `scale` is the one `save_frames` returned.
    """
    path = Path(path)
    if path.suffix == '.npy':
        return np.load(path)
    dtype = np.dtype(dtype)
    n, h, w = shape
    lossy = path.suffix == '.mp4'
    pix_fmt = 'gray' if lossy else _raw_pix_fmt(dtype)
    raw = _run(['-i', str(path), '-fps_mode', 'passthrough', '-f', 'rawvideo',
                '-pix_fmt', pix_fmt, 'pipe:1'], None, f'read {path.name}')
    itemsize = 1 if pix_fmt == 'gray' else 2
    # a padded (odd-sized) frame comes back at its padded size
    ph, pw = (math.ceil(h / 2) * 2, math.ceil(w / 2) * 2) if lossy else (h, w)
    expected = n * ph * pw * itemsize
    if len(raw) != expected:
        raise RuntimeError(f'{path.name} decoded to {len(raw)} bytes, expected '
                           f'{expected} for {n} frames of {w}x{h}')
    stack = np.frombuffer(raw, np.uint8 if itemsize == 1 else '<u2')
    stack = stack.reshape(n, ph, pw)[:, :h, :w]
    if lossy and scale != 1.0:
        top = np.iinfo(dtype).max
        return np.clip(np.rint(stack.astype(np.float32) * scale), 0,
                       top).astype(dtype)
    return stack.astype(dtype)
