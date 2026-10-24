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

The NUCLEO connects to a breadboard with nine B10K (10 kΩ linear) pots and four
note buttons. Four pots set the note pitches. Five pots control cutoff and a
full attack, decay, sustain and release (ADSR) volume envelope for each voice.

NUCLEO pin setup:

- A0, A1, A2, A3: pitch pots for notes 1-4
- A4: cutoff pot
- A5: release time pot
- PC2, CN7 pin 35: attack time pot (ADC1 channel 12)
- PC3, CN7 pin 37: decay time pot (ADC1 channel 13)
- PC4, CN10 pin 34: sustain level pot (ADC1 channel 14)
- D4, D5, D6, D7: buttons for notes 1-4, wired to GND (internal pull-ups)
- B1: cycles the waveform
