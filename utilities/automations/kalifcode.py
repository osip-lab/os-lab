import inspect
import queue
from typing import Optional
import sounddevice as sd
import vosk
import json
import threading
from plyer import notification
from datetime import datetime
import difflib
import inspect
import asyncio
from local_config import PATH_STT_MODEL


# Load offline Vosk model
model = vosk.Model(PATH_STT_MODEL)
log_path = "utilities/automations/voice_command_log.txt"

# Queue to receive audio data
audio_q = queue.Queue()


def log_notes(note: str):
    """Append a note to a log file with a timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"\n[{timestamp}]\n{note}\n"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)

    print(f"📝 Logged: {note}")


# Audio callback
def audio_callback(indata, frames, time, status):
    if status:
        print(f"[Audio status]: {status}")
    audio_q.put(bytes(indata))


def recognize_loop(command_map, print_speech: Optional[bool] = True, notification_speech: Optional[bool] = True):
    command_map['note'] = log_notes  # Add logging command
    rec = vosk.KaldiRecognizer(model, 16000)

    while True:
        data = audio_q.get()
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result.get("text", "").lower()
            if not text:
                continue

            if text:
                if print_speech:
                    print(f"[Recognized]: {text}")
                if notification_speech:
                    notification.notify(
                        title="Voice Command Recognized",
                        message=text,
                        timeout=0.2
                    )

                # 1. Exact match
                if text in command_map:
                    try:
                        command_map[text]()
                    except Exception as e:
                        print(f"[ERROR running command '{text}']: {e}")
                    continue

                # 2. Startswith match
                for command, func in command_map.items():
                    if text.startswith(command + " "):
                        remaining = text[len(command):].strip()
                        try:
                            sig = inspect.signature(func)
                            if len(sig.parameters) >= 1:
                                func(remaining)
                            else:
                                func()
                        except Exception as e:
                            print(f"[ERROR running command '{command}' with arg '{remaining}']: {e}")
                        return

                # 3. Fuzzy match using difflib
                best_match = difflib.get_close_matches(text, command_map.keys(), n=1, cutoff=0.8)
                if not best_match:
                    # Try matching prefix instead
                    prefix_candidates = [cmd for cmd in command_map if text.startswith(cmd[:max(6, len(cmd)//2)])]
                    best_match = difflib.get_close_matches(text.split(" ")[0], prefix_candidates, n=1, cutoff=0.7)

                if best_match:
                    cmd = best_match[0]
                    try:
                        remaining = text[len(cmd):].strip()
                        sig = inspect.signature(command_map[cmd])
                        if len(sig.parameters) >= 1:
                            command_map[cmd](remaining)
                        else:
                            command_map[cmd]()
                    except Exception as e:
                        print(f"[ERROR running fuzzy-matched command '{cmd}']: {e}")


# Entry point — pass your command_map here
def start_voice_listener(command_map: dict):
    threading.Thread(target=recognize_loop, args=(command_map,), daemon=True).start()

    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=audio_callback, device=2):
        print("🎤 Listening for commands...")
        while True:
            pass  # keep main thread alive


# %%

if __name__ == "__main__":
    # Command function map
    def foo():
        print("hello world")

    # Example command map
    start_voice_listener(command_map={
        "print": foo,
        'there are five more minutes': lambda: print("There are five more minutes left!"),
    })
