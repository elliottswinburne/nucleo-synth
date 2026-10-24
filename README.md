# Synthesizer

A synthesizer running on a [NUCLEO-F446RE](https://www.st.com/en/evaluation-tools/nucleo-f446re.html). The board generates audio and streams it
over the ST-LINK virtual COM port. Python script on the PC plays it and shows
a scope view.

## Layout

| Path        | Contents                                                        |
|-------------|-----------------------------------------------------------------|
| `firmware/` | STM32CubeMX project (`synthesizer_1.ioc`), built with CMake     |
| `host/`     | `synth_rx.py`: receives the stream, plays it, draws the scope   |

---

In my setup, the NUCLEO is connected up to a breadboard with six pots fitted and some buttons.
The buttons control four notes whose pitch can be adjusted using four of the six pots.
The other two pots control the cutoff and release time.

NUCLEO pin setup:

- A0, A1, A2, A3: pitch pots for notes 1-4
- A4: cutoff pot
- A5: release time pot
- D4, D5, D6, D7: buttons for notes 1-4, wired to GND (internal pull-ups)
- B1 (on-board blue button): cycles the waveform
