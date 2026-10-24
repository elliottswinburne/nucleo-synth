#ifndef SEQUENCER_H
#define SEQUENCER_H

#include <stdint.h>

typedef struct {
    uint32_t phase;
    uint8_t step;
    uint8_t running;
} sequencer_t;

/* One sixteenth note per step; gates occupy 75% of each step. */
static inline uint8_t sequencer_tick(sequencer_t *seq, uint8_t running,
                                     uint8_t bpm, uint8_t enabled)
{
    const uint32_t period = 22050u * 60u / 4u;
    if (!running) {
        seq->running = 0;
        seq->phase = 0;
        seq->step = 0;
        return 0;
    }
    if (!seq->running) {
        seq->running = 1;
        seq->phase = 0;
        seq->step = 0;
    } else {
        seq->phase += bpm;
        if (seq->phase >= period) {
            seq->phase -= period;
            seq->step = (seq->step + 1) & 3;
        }
    }
    return seq->phase < period * 3 / 4 ? enabled & (1u << seq->step) : 0;
}

#endif
