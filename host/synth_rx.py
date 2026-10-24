#!/usr/bin/env python3
"""
synth_rx.py - play the NUCLEO-F446RE synth's audio stream on the PC.

The Nucleo sends frames over the ST-LINK virtual COM port:
    0xA5 0x5A  +  256 x int16 little-endian mono samples   (514 bytes)
at 22050 Hz, 921600 baud.

Install:  pip install pyserial sounddevice numpy
Run:      python synth_rx.py                 (auto-detects the ST-LINK port)
          python synth_rx.py --port COM5     (Windows, explicit port)
          python synth_rx.py --port /dev/ttyACM0
          python synth_rx.py --device "Speakers"   (pin one output device)
          python synth_rx.py --list-devices
          python synth_rx.py --no-scope         (console stats, no window)
A scope window shows the waveform and the detected pitch. Close it to quit.
On Windows the default output follows whatever device is selected in the
taskbar volume menu, including changes made while it's running.
Linux: your user must be in the 'dialout' group to open /dev/ttyACM*.
"""
import argparse
import collections
import sys
import threading
import time

import numpy as np
import serial
import serial.tools.list_ports
import sounddevice as sd

SYNC = b"\xA5\x5A"
BLOCK = 256                 # samples per frame (must match firmware)
FRAME = 2 + 2 * BLOCK       # bytes per frame
SCOPE_BLOCKS = 32           # audio kept for the scope: 32 x 256 = ~371 ms
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def parse(buf: bytearray, stats: dict):
    """Pull complete, validated frames out of buf (modified in place).

    A frame is accepted only if the *next* frame's sync word follows it,
    so a 0xA5 0x5A pair occurring inside audio data cannot desync us.
    """
    blocks = []
    while True:
        i = buf.find(SYNC)
        if i < 0:
            del buf[:-1]            # keep a possible half sync byte
            break
        if i > 0:
            del buf[:i]
            stats["resync"] += 1
        if len(buf) < FRAME + 2:
            break
        if buf[FRAME:FRAME + 2] != SYNC:
            del buf[:1]             # false sync inside audio, keep scanning
            stats["resync"] += 1
            continue
        blocks.append(np.frombuffer(bytes(buf[2:FRAME]), dtype="<i2")
                      .astype(np.float32) / 32768.0)
        del buf[:FRAME]
    return blocks


def detect_pitch(x, rate, fmin=40.0, fmax=1200.0):
    """Fundamental in Hz by autocorrelation, or None if quiet/aperiodic."""
    x = x - x.mean()
    if np.sqrt(np.mean(x * x)) < 0.005:
        return None
    n = len(x)
    spec = np.fft.rfft(x, 2 * n)
    r = np.fft.irfft(spec * np.conj(spec))[:n]
    # normalise by the energy in each lag's overlap (McLeod's NSDF): +-1 for
    # a perfect repeat, and partial cycles at the window edges don't skew it
    sq = np.concatenate(([0.0], np.cumsum(x * x)))
    tau = np.arange(n)
    r = 2 * r / np.maximum(sq[n - tau] + sq[n] - sq[tau], 1e-12)
    lo, hi = int(rate / fmax), min(int(rate / fmin), n // 2)
    # skip the central lobe around lag 0
    neg = np.nonzero(r[:hi] < 0)[0]
    if len(neg) == 0:
        return None
    start = max(lo, neg[0])
    seg = r[start:hi + 1]
    best = seg.max()
    if best < 0.3 * r[0]:
        return None
    # take the first peak close to the best one, not the best itself: at
    # 2, 3.. periods the peak can land nearer a whole sample and win
    peaks = np.nonzero((seg[1:-1] >= seg[:-2]) & (seg[1:-1] > seg[2:]) &
                       (seg[1:-1] >= 0.9 * best))[0]
    lag = start + 1 + int(peaks[0]) if len(peaks) else start + int(seg.argmax())
    a, b, c = r[lag - 1], r[lag], r[lag + 1]     # parabolic peak refinement
    d = a - 2 * b + c
    return rate / (lag + (0.5 * (a - c) / d if d < 0 else 0.0))


def note_name(f):
    midi = 69 + 12 * np.log2(f / 440.0)
    m = int(round(midi))
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1} {100 * (midi - m):+.0f}¢"


def run_scope(scope, stats, q, rate):
    """Oscilloscope window: triggered waveform + detected pitch. Blocks."""
    import tkinter as tk
    import traceback

    root = tk.Tk()
    root.title("synth scope")
    canvas = tk.Canvas(root, width=800, height=360, bg="#000000",
                       highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    axis = canvas.create_line(0, 0, 0, 0, fill="#333333")
    trace = canvas.create_line(0, 0, 0, 0, fill="#ffffff", width=2)
    pitch = canvas.create_text(16, 12, anchor="nw", fill="#ffffff",
                               font=("Consolas", 22, "bold"))
    info = canvas.create_text(16, 52, anchor="nw", fill="#888888",
                              font=("Consolas", 10))
    hint = canvas.create_text(0, 0, anchor="se", fill="#555555",
                              font=("Consolas", 9),
                              text="↑↓/wheel gain   "
                                   "←→ cycles   R reset")
    view = {"gain": 1.0, "cycles": 3}

    def zoom(gain=1.0, cycles=0):
        view["gain"] = min(max(view["gain"] * gain, 0.25), 32.0)
        view["cycles"] = min(max(view["cycles"] + cycles, 1), 24)

    root.bind("<Up>", lambda e: zoom(gain=1.25))
    root.bind("<Down>", lambda e: zoom(gain=0.8))
    root.bind("<MouseWheel>",
              lambda e: zoom(gain=1.25 if e.delta > 0 else 0.8))
    root.bind("<Left>", lambda e: zoom(cycles=-1))
    root.bind("<Right>", lambda e: zoom(cycles=1))
    root.bind("<Key-r>", lambda e: view.update(gain=1.0, cycles=3))

    def update():
        w, h = max(canvas.winfo_width(), 2), canvas.winfo_height()
        blocks = list(scope)
        x = np.concatenate(blocks) if blocks else np.zeros(BLOCK, np.float32)
        f = detect_pitch(x[-2048:], rate)

        # show the chosen number of cycles, starting at a rising zero
        # crossing so the trace stands still instead of scrolling
        period = rate / f if f else BLOCK
        span = int(min(max(view["cycles"] * period, 128), len(x) // 2))
        s = len(x) - span
        start = s - int(period) - 1
        if start >= 0:
            seg = x[start:s]
            up = np.nonzero((seg[:-1] < 0) & (seg[1:] >= 0))[0]
            if len(up):
                s = start + up[0]
        y = x[s:s + span]

        ys = np.interp(np.linspace(0, len(y) - 1, w), np.arange(len(y)), y)
        pts = np.empty(2 * w)
        pts[0::2] = np.arange(w)
        pts[1::2] = h / 2 - ys * (0.48 * h * view["gain"])  # x1: full scale
        canvas.coords(trace, *pts.tolist())
        canvas.coords(axis, 0, h / 2, w, h / 2)
        canvas.coords(hint, w - 10, h - 8)

        canvas.itemconfigure(
            pitch, text=f"{f:6.1f} Hz   {note_name(f)}" if f else "—")
        canvas.itemconfigure(
            info, text=f"{1000 * span / rate:.1f} ms across   "
                       f"gain x{view['gain']:.2g}   "
                       f"frames {stats['frames']}  buffered {len(q)}  "
                       f"underruns {stats['under']}  drops {stats['drop']}  "
                       f"resyncs {stats['resync']}")
        root.after(33, update)                         # ~30 fps

    def on_error(exc, val, tb):                        # let Ctrl+C quit
        if issubclass(exc, KeyboardInterrupt):
            root.destroy()
        else:
            traceback.print_exception(exc, val, tb)

    root.report_callback_exception = on_error
    update()
    root.mainloop()


def find_port():
    for p in serial.tools.list_ports.comports():
        if p.vid == 0x0483:         # STMicroelectronics (ST-LINK VCP)
            return p.device
    sys.exit("No ST-LINK virtual COM port found, pass --port.")


def find_output(name):
    """Return the first output device whose name contains `name`, else None."""
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] > 0 and name.lower() in d["name"].lower():
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--rate", type=int, default=22050)
    ap.add_argument("--target", type=int, default=4,
                    help="jitter buffer depth in blocks (4 = ~46 ms)")
    ap.add_argument("--device",
                    help="output device (name substring or index). Default "
                         "follows the Windows output selection")
    ap.add_argument("--list-devices", action="store_true",
                    help="print audio devices and exit")
    ap.add_argument("--no-scope", action="store_true",
                    help="no waveform window. Print stats to the console")
    args = ap.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    # PortAudio's own default is fixed to one device when the stream opens.
    # The MME "Sound Mapper" is Windows' default-device alias, and Windows
    # reroutes it live when the output selection changes.
    if args.device is None:
        device = find_output("Microsoft Sound Mapper")    # None off Windows
    elif args.device.isdigit():
        device = int(args.device)
    else:
        device = find_output(args.device)
        if device is None:
            sys.exit(f"No output device matching '{args.device}'. "
                     "see --list-devices.")
    dev_name = sd.query_devices(device, kind="output")["name"]

    port = args.port or find_port()
    ser = serial.Serial(port, args.baud, timeout=0.05)
    print(f"Listening on {port} @ {args.baud} baud, {args.rate} Hz")
    print(f"Playing on {dev_name}")

    q = collections.deque()
    scope = collections.deque(maxlen=SCOPE_BLOCKS)   # recently played audio
    stats = {"resync": 0, "under": 0, "drop": 0, "frames": 0}
    stop = threading.Event()

    def reader():
        buf = bytearray()
        while not stop.is_set():
            buf += ser.read(4096)
            for b in parse(buf, stats):
                q.append(b)
                stats["frames"] += 1

    def callback(outdata, frames, t, status):
        # MCU and soundcard clocks differ slightly: drop a block if we fall
        # behind, output silence if we run dry.
        if len(q) > 2 * args.target:
            q.popleft()
            stats["drop"] += 1
        try:
            outdata[:, 0] = q.popleft()
        except IndexError:
            outdata.fill(0)
            stats["under"] += 1
        scope.append(outdata[:, 0].copy())

    threading.Thread(target=reader, daemon=True).start()

    while len(q) < args.target:       # prefill
        time.sleep(0.01)

    try:
        with sd.OutputStream(device=device, samplerate=args.rate,
                             channels=1, dtype="float32", latency="low",
                             blocksize=BLOCK, callback=callback):
            if not args.no_scope:
                run_scope(scope, stats, q, args.rate)
                return
            while True:
                time.sleep(1)
                print(f"\rframes {stats['frames']}  buffered {len(q)}  "
                      f"underruns {stats['under']}  drops {stats['drop']}  "
                      f"resyncs {stats['resync']}   ", end="", flush=True)
    except KeyboardInterrupt:
        print()
    finally:
        stop.set()
        ser.close()


if __name__ == "__main__":
    main()
