#!/usr/bin/env python3
"""
synth_rx.py - play the NUCLEO-F446RE synth's audio stream on the PC.

The Nucleo sends frames over the ST-LINK virtual COM port:
    0xA5 0x5A  +  256 x int16 little-endian mono samples
               +  9 x uint16 raw pot readings  +  uint8 held keys  +  uint8 wave
    (534 bytes) at 22050 Hz, 921600 baud.

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
NPOTS = 9                   # notes 1-4, cutoff, release, attack, decay, sustain
AUDIO_END = 2 + 2 * BLOCK
FRAME = AUDIO_END + 2 * NPOTS + 2   # bytes per frame
SCOPE_BLOCKS = 32           # audio kept for the scope: 32 x 256 = ~371 ms
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
WAVES = ["SAW", "SQUARE", "TRIANGLE", "SINE"]

# scope colour themes, cycled with T. None leaves that element out.
#   trace: the whole trace. With beam set, this is the dim colour of the
#          fast-moving parts, and beam (mid, bright) is drawn over the slower
#          parts, like a CRT whose brightness falls with beam speed
#   glow:  halo colours, widest first: the outer two under the mid parts,
#          the innermost under the bright parts (steps fake a soft falloff,
#          since Tk lines aren't anti-aliased)
#   grid:  graticule colour.  bezel: frame and rounded screen corners
#   panel: background of the control panel below the screen
THEMES = [
    {"bg": "#000000", "panel": "#000000", "trace": "#ffffff", "beam": None,
     "glow": None, "axis": "#333333", "grid": None, "bezel": None,
     "pitch": "#ffffff", "info": "#888888", "hint": "#555555",
     "label": "#666666", "value": "#888888", "active": "#ffffff"},
    # vintage: green P31 phosphor CRT
    {"bg": "#020b05", "panel": "#020b05", "trace": "#2b7343",
     "beam": ("#6fdc8f", "#d2ffdc"),
     "glow": ("#06200e", "#0c3519", "#1a5c2e"),
     "axis": "#1c5a2e", "grid": "#0d2e18", "bezel": "#0c0e0d",
     "pitch": "#59ff8c", "info": "#2f9a52", "hint": "#1f6b38",
     "label": "#2a8048", "value": "#3fae63", "active": "#a8ffc0"},
]
BEAM_SLOW = (4.0, 1.2)      # px of rise per px across: mid, bright below this
# canvas stacking order, bottom to top
LAYERS = ["grid", "axis", "glow_wide", "glow_mid", "glow_hi", "trace",
          "beam_mid", "beam_hi", "bezel", "text"]


def runs(mask, min_len=2):
    """(first, last) point indices of each run of >= min_len consecutive
    True segments; segment i joins points i and i+1."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    keep = ends - starts >= min_len
    return list(zip(starts[keep].tolist(), ends[keep].tolist()))


def thin(pts, step=4):
    """Every step-th point of a flat [x0, y0, x1, y1, ...] list, keeping the
    last point. For wide halos, which hide the detail anyway."""
    out = [v for i in range(0, len(pts), 2 * step) for v in pts[i:i + 2]]
    if (len(pts) - 2) % (2 * step):
        out += pts[-2:]
    return out


def parse(buf: bytearray, stats: dict, ctrl: dict):
    """Pull complete, validated frames out of buf (modified in place).

    A frame is accepted only if the *next* frame's sync word follows it,
    so a 0xA5 0x5A pair occurring inside audio data cannot desync us.
    The latest control state is stored in ctrl.
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
        blocks.append(np.frombuffer(bytes(buf[2:AUDIO_END]), dtype="<i2")
                      .astype(np.float32) / 32768.0)
        ctrl["pots"] = np.frombuffer(bytes(buf[AUDIO_END:FRAME - 2]),
                                     dtype="<u2").tolist()
        ctrl["keys"], ctrl["wave"] = buf[FRAME - 2], buf[FRAME - 1]
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


# Pot -> parameter mappings. These mirror control_update() in synth_core.c
# and must be kept in step with it.
def pot_pitch(p):
    return 55.0 * 2 ** (p * 4.0 / 4095.0)


def pot_cutoff(p, rate):
    """One-pole low-pass corner in Hz, or None when the filter is wide open."""
    k = int(32767.0 * 2 ** ((p / 4095.0 - 1.0) * 7.0)) / 32768.0
    fc = -rate * np.log(1.0 - k) / (2 * np.pi)
    return fc if fc < rate / 2 else None


def pot_release(p):
    return 0.01 * 2 ** (p * 7.6 / 4095.0)


def pot_envelope_time(p):
    return 0.002 * 2 ** (p * 9.965784 / 4095.0)


class PitchDisplay:
    """Reject pot noise for labels without changing the synth's tuning."""

    def __init__(self):
        self.history = collections.deque(maxlen=7)
        self.source = None
        self.pots = None

    def update(self, pots):
        # The receiver replaces this list for each frame. Don't count repeated
        # redraws of the same frame as additional measurements.
        if pots is self.source:
            return self.pots
        self.source = pots
        self.history.append(pots[:4])
        median = np.median(self.history, axis=0)
        if self.pots is None:
            self.pots = median.tolist()
        else:
            for i, reading in enumerate(median):
                # Four ADC counts are about five cents across the pitch range.
                if abs(reading - self.pots[i]) > 4:
                    self.pots[i] = float(reading)
        return self.pots


def control_labels(ctrl, rate, note_pots=None):
    """[(name, value text, active)] for the control panel."""
    pots, keys = ctrl["pots"], ctrl["keys"]
    cells = []
    for i in range(4):
        f = pot_pitch(pots[i] if note_pots is None else note_pots[i])
        cells.append((f"NOTE {i + 1}", f"{f:.1f} Hz\n{note_name(f)}",
                      bool(keys >> i & 1)))
    fc = pot_cutoff(pots[4], rate)
    fc_text = ("open" if fc is None else
               f"{fc / 1000:.2f} kHz" if fc >= 1000 else f"{fc:.0f} Hz")
    cells.append(("CUTOFF", f"{fc_text}\n{100 * pots[4] / 4095:.0f}%", False))
    for name, index in (("ATTACK", 6), ("DECAY", 7), ("SUSTAIN", 8), ("RELEASE", 5)):
        percent = 100 * pots[index] / 4095
        if name == "SUSTAIN":
            text = f"{percent:.0f}%\nlevel"
        else:
            t = pot_release(pots[index]) if index == 5 else pot_envelope_time(pots[index])
            t_text = f"{t:.2f} s" if t >= 1 else f"{1000 * t:.0f} ms"
            text = f"{t_text}\n{percent:.0f}%"
        cells.append((name, text, False))
    cells.append(("WAVE", f"{WAVES[ctrl['wave'] & 3]}\n ", False))
    return cells


def run_scope(scope, stats, q, rate, ctrl):
    """Oscilloscope window: triggered waveform + detected pitch, and a panel
    showing each control's current value. Blocks."""
    import tkinter as tk
    import traceback

    root = tk.Tk()
    root.title("synth scope")

    # control panel along the bottom: one cell per pot, plus the waveform
    panel = tk.Frame(root, bg="#000000")
    panel.pack(side="bottom", fill="x")
    cells = []
    for col in range(NPOTS + 1):
        panel.columnconfigure(col, weight=1, uniform="cell")
        name = tk.Label(panel, bg="#000000", fg="#666666",
                        font=("Consolas", 9))
        value = tk.Label(panel, bg="#000000", fg="#888888", justify="center",
                         font=("Consolas", 12), text="—\n ")
        name.grid(row=0, column=col, pady=(8, 0))
        value.grid(row=1, column=col, pady=(2, 10))
        cells.append((name, value))

    canvas = tk.Canvas(root, width=800, height=360, bg="#000000",
                       highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    axis = canvas.create_line(0, 0, 0, 0, fill="#333333", tags="axis")
    trace = canvas.create_line(0, 0, 0, 0, fill="#ffffff", width=2,
                               tags="trace")
    pitch = canvas.create_text(16, 12, anchor="nw", fill="#ffffff",
                               font=("Consolas", 22), tags="text")
    info = canvas.create_text(16, 48, anchor="nw", fill="#888888",
                              font=("Consolas", 10), state="hidden",
                              tags="text")
    hint = canvas.create_text(0, 0, anchor="se", fill="#555555",
                              font=("Consolas", 9), text="I - info",
                              tags="text")
    view = {"gain": 1.0, "cycles": 3, "info": False, "theme": 0,
            "grid_size": None, "restack": False}

    # line pools for the beam-speed layers: one item per run of slow trace,
    # created as needed and hidden when unused
    widths = {"glow_wide": 11, "glow_mid": 7, "glow_hi": 5, "beam_mid": 2,
              "beam_hi": 2}
    pools = {layer: [] for layer in widths}
    shown = dict.fromkeys(pools, 0)

    def show_runs(layer, lines, colour):
        pool = pools[layer]
        while len(pool) < len(lines):
            pool.append(canvas.create_line(0, 0, 0, 0, width=widths[layer],
                                           capstyle="round",
                                           joinstyle="round", tags=layer))
            view["restack"] = True
        for item, p in zip(pool, lines):
            canvas.coords(item, *p)
            canvas.itemconfigure(item, state="normal", fill=colour)
        for item in pool[len(lines):shown[layer]]:
            canvas.itemconfigure(item, state="hidden")
        shown[layer] = len(lines)

    def zoom(gain=1.0, cycles=0):
        view["gain"] = min(max(view["gain"] * gain, 0.25), 32.0)
        view["cycles"] = min(max(view["cycles"] + cycles, 1), 24)

    root.bind("<Up>", lambda e: zoom(gain=1.25))
    root.bind("<Down>", lambda e: zoom(gain=0.8))
    root.bind("<Left>", lambda e: zoom(cycles=-1))
    root.bind("<Right>", lambda e: zoom(cycles=1))
    root.bind("<Key-r>", lambda e: view.update(gain=1.0, cycles=3))

    def set_hint():
        canvas.itemconfigure(
            hint, text="↑↓ - gain   ←→ - cycles   R - reset   T - theme   "
                       "I - info" if view["info"] else "I - info")

    def toggle_info(e):
        view["info"] = not view["info"]
        canvas.itemconfigure(info, state="normal" if view["info"] else "hidden")
        set_hint()

    root.bind("<Key-i>", toggle_info)

    def draw_screen(w, h):
        """Graticule (10 x 8 divisions, minor ticks along the centre lines)
        and the bezel, for the current size and theme."""
        canvas.delete("grid", "bezel")
        view["restack"] = True
        t = THEMES[view["theme"]]
        if t["bezel"] is not None:
            r = 26                                  # screen corner radius
            for cx, cy, a0, corner in ((r, r, 180, (0, 0)),
                                       (w - r, r, 270, (w, 0)),
                                       (w - r, h - r, 0, (w, h)),
                                       (r, h - r, 90, (0, h))):
                arc = [v for a in np.radians(np.linspace(a0, a0 + 90, 12))
                       for v in (cx + r * np.cos(a), cy + r * np.sin(a))]
                canvas.create_polygon(*corner, *arc, fill=t["bezel"],
                                      outline="", tags="bezel")
            canvas.create_line(0, h, 0, 0, w, 0, w, h, fill=t["bezel"],
                               width=8, tags="bezel")
            canvas.create_line(0, h - 1, w, h - 1, fill=t["grid"],
                               width=1, tags="bezel")
        if t["grid"] is None:
            return
        for i in range(11):
            canvas.create_line(w * i / 10, 0, w * i / 10, h, tags="grid")
        for i in range(9):
            canvas.create_line(0, h * i / 8, w, h * i / 8, tags="grid")
        for i in range(51):
            x = w * i / 50
            canvas.create_line(x, h / 2 - 4, x, h / 2 + 5, tags="grid")
        for i in range(41):
            y = h * i / 40
            canvas.create_line(w / 2 - 4, y, w / 2 + 5, y, tags="grid")
        canvas.itemconfigure("grid", fill=t["grid"])

    def apply_theme():
        t = THEMES[view["theme"]]
        canvas.configure(bg=t["bg"])
        for widget in (root, panel):
            widget.configure(bg=t["panel"])
        for name, value in cells:
            name.configure(bg=t["panel"], fg=t["label"])
            value.configure(bg=t["panel"], fg=t["value"])
        for item, key in ((axis, "axis"), (trace, "trace"), (pitch, "pitch"),
                          (info, "info"), (hint, "hint")):
            canvas.itemconfigure(item, fill=t[key])
        if t["beam"] is None:
            for layer in pools:
                show_runs(layer, [], "")
        view["grid_size"] = None                    # redraw in the new colour

    def cycle_theme(e):
        view["theme"] = (view["theme"] + 1) % len(THEMES)
        apply_theme()

    root.bind("<Key-t>", cycle_theme)

    redraw_interval = 1.0 / 120.0
    next_redraw = time.perf_counter()
    pitch_display = PitchDisplay()

    def update():
        nonlocal next_redraw
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
        theme = THEMES[view["theme"]]
        # slope in px per px for each segment, smoothed so a slope sitting
        # near a threshold doesn't split a run into flickering pieces
        rise = np.abs(np.diff(pts[1::2]))
        if len(rise) >= 5:
            rise = np.convolve(rise, np.ones(5) / 5, mode="same")
        pts = pts.tolist()
        canvas.coords(trace, *pts)
        if theme["beam"]:
            mid, hi = ([pts[2 * a:2 * b + 2] for a, b in runs(rise < lim)]
                       for lim in BEAM_SLOW)
            # halos thinned: cheaper, and no knots from overlapping round joins
            mid_halo, hi_halo = [thin(p) for p in mid], [thin(p) for p in hi]
            show_runs("glow_wide", mid_halo, theme["glow"][0])
            show_runs("glow_mid", mid_halo, theme["glow"][1])
            show_runs("glow_hi", hi_halo, theme["glow"][2])
            show_runs("beam_mid", mid, theme["beam"][0])
            show_runs("beam_hi", hi, theme["beam"][1])
        if view["grid_size"] != (w, h):
            draw_screen(w, h)
            view["grid_size"] = (w, h)
        if view["restack"]:
            for layer in LAYERS:
                if canvas.find_withtag(layer):      # raise errors on no match
                    canvas.tag_raise(layer)
            view["restack"] = False
        canvas.coords(axis, 0, h / 2, w, h / 2)
        canvas.coords(hint, w - 10, h - 8)

        canvas.itemconfigure(
            pitch, text=f"{f:.1f} Hz   {note_name(f)}" if f else "—")
        canvas.itemconfigure(
            info, text=f"{1000 * span / rate:.1f} ms across   "
                       f"gain x{view['gain']:.2g}   "
                       f"frames {stats['frames']}  buffered {len(q)}  "
                       f"underruns {stats['under']}  drops {stats['drop']}  "
                       f"resyncs {stats['resync']}")

        if ctrl:
            for (name, value), (n, text, active) in zip(
                    cells, control_labels(ctrl, rate,
                                          pitch_display.update(ctrl["pots"]))):
                name.configure(text=n)
                value.configure(text=text, fg=theme["active"] if active
                                else theme["value"])
        # Keep a 120 Hz schedule independent of redraw cost and timer lateness.
        # Skip missed deadlines so a stalled UI never queues catch-up redraws.
        next_redraw += redraw_interval
        now = time.perf_counter()
        if next_redraw <= now:
            missed = int((now - next_redraw) / redraw_interval) + 1
            next_redraw += missed * redraw_interval
        delay_ms = max(1, int((next_redraw - now) * 1000 + 0.999))
        root.after(delay_ms, update)

    def on_error(exc, val, tb):                        # let Ctrl+C quit
        if issubclass(exc, KeyboardInterrupt):
            root.destroy()
        else:
            traceback.print_exception(exc, val, tb)

    root.report_callback_exception = on_error
    winmm = None
    if sys.platform == "win32":
        import ctypes
        winmm = ctypes.WinDLL("winmm")
        if winmm.timeBeginPeriod(1) != 0:
            winmm = None
    try:
        update()
        root.mainloop()
    finally:
        if winmm is not None:
            winmm.timeEndPeriod(1)


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
    ctrl = {}                                        # latest control state
    stop = threading.Event()

    def reader():
        buf = bytearray()
        while not stop.is_set():
            buf += ser.read(4096)
            for b in parse(buf, stats, ctrl):
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
                run_scope(scope, stats, q, args.rate, ctrl)
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
