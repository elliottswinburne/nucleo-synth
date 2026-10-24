/*
 * synth_core.c - 4-voice polyphonic synth engine + serial audio streaming
 * Target: NUCLEO-F446RE, STM32CubeIDE / HAL.
 *
 * ---- CubeMX setup -------------------------------------------------------
 * RCC     : HSE = BYPASS clock source (8 MHz from ST-LINK MCO), SYSCLK 180 MHz,
 *           APB1 /4 (45 MHz, timers 90 MHz), APB2 /2.
 *           Use HSE, not HSI: HSI is ~1% off, which makes the PC buffer
 *           under/overrun every second or so (audible clicks).
 * USART2  : PA2/PA3 (wired to ST-LINK VCP), Asynchronous, 921600 8N1.
 *           DMA: USART2_TX, Normal mode, byte width. USART2 global IRQ on.
 * TIM6    : Prescaler 0, Counter Period 4081  -> 90 MHz/4082 = ~22.05 kHz.
 *           TIM6 global interrupt on.
 * ADC1    : Scan ON, Continuous OFF, DMA Continuous Requests OFF,
 *           9 conversions, sampling time 480 cycles each, ranks in order:
 *             1 IN0  (PA0, A0)  pitch note 1
 *             2 IN1  (PA1, A1)  pitch note 2
 *             3 IN4  (PA4, A2)  pitch note 3
 *             4 IN8  (PB0, A3)  pitch note 4
 *             5 IN11 (PC1, A4)  filter cutoff
 *             6 IN10 (PC0, A5)  release time
 *             7 IN12 (PC2, CN7-35) attack time
 *             8 IN13 (PC3, CN7-37) decay time
 *             9 IN14 (PC4, CN10-34) sustain level
 *           DMA: ADC1 (DMA2 Stream0), Normal, half-word.
 * GPIO    : PB5 (D4), PB4 (D5), PB10 (D6), PA8 (D7) = GPIO_Input, Pull-up.
 *           PC13 (blue B1) = GPIO_Input (board has its own pull-up).
 *
 * ---- Hook-up in main.c --------------------------------------------------
 *   USER CODE BEGIN 2:     synth_init();
 *   USER CODE BEGIN 3:     synth_loop();      (inside while(1))
 * and declare:  void synth_init(void); void synth_loop(void);
 */
#include "main.h"
#include <math.h>
#include <stdint.h>

extern UART_HandleTypeDef huart2;
extern TIM_HandleTypeDef  htim6;
extern ADC_HandleTypeDef  hadc1;

#define FS      22050.0f
#define BLOCK   256
#define NPOTS   9
#define NNOTES  4

/* ---------------- audio frames (double buffered) ---------------- */
typedef struct {
    uint8_t sync[2];   // two unsigned bytes
    int16_t s[BLOCK];
    uint16_t pots[NPOTS];  /* raw 12-bit ADC readings, for the host display */
    uint8_t  keys;         /* bit n set = note n+1 button held              */
    uint8_t  wave;
} __attribute__((packed)) frame_t;

_Static_assert(sizeof(frame_t) == 534, "Audio frame must match the host protocol");

static frame_t frames[2] = { { {0xA5, 0x5A}, {0} }, { {0xA5, 0x5A}, {0} } };
static volatile uint8_t  fill_buf  = 0;
static volatile uint16_t fill_idx  = 0;
static volatile int8_t   ready_buf = -1;        /* complete, waiting to send */
static volatile int8_t   tx_buf    = -1;        /* owned by UART DMA         */
static volatile uint32_t dropped_frames = 0;    /* overruns, for debugging   */

/* ---------------- parameters: control loop -> audio ISR ---------------- */
static volatile uint32_t phase_inc[NNOTES];     /* oscillator step, per voice */
static volatile uint8_t  gate[NNOTES];          /* key held, per voice      */
static volatile uint8_t  wave         = 0;      /* 0 saw 1 square 2 tri 3 sine */
static volatile int32_t  lp_k         = 32767;  /* filter coefficient Q15   */

#define ENV_MAX     (32767 * 32768)             /* Q30, converts to Q15 32767 */

static volatile int32_t attack_step = 300 * 32768;
static volatile int32_t decay_samples = 2205;
static volatile int32_t release_samples = 2205;
static volatile int32_t sustain_level = ENV_MAX;

typedef enum { ENV_IDLE, ENV_ATTACK, ENV_DECAY, ENV_SUSTAIN, ENV_RELEASE } env_stage_t;
typedef struct {
    int32_t level, step;
    env_stage_t stage;
    uint8_t held;
} envelope_t;

/* Linear Q30 ADSR. A new press attacks from the current level, even in release.
   Decay/release durations are captured when those stages begin. */
static int32_t envelope_tick(envelope_t *env, uint8_t held)
{
    if (held && !env->held) env->stage = ENV_ATTACK;
    if (!held && env->held) {
        env->stage = ENV_RELEASE;
        env->step = (env->level + release_samples - 1) / release_samples;
    }
    env->held = held;

    switch (env->stage) {
    case ENV_ATTACK:
        env->level += attack_step;
        if (env->level >= ENV_MAX) {
            env->level = ENV_MAX;
            env->stage = ENV_DECAY;
            env->step = (ENV_MAX - sustain_level + decay_samples - 1) / decay_samples;
            if (env->step < 1) env->step = 1;
        }
        break;
    case ENV_DECAY:
        env->level -= env->step;
        if (env->level <= sustain_level) {
            env->level = sustain_level;
            env->stage = ENV_SUSTAIN;
        }
        break;
    case ENV_SUSTAIN:
        env->level = sustain_level;
        break;
    case ENV_RELEASE:
        env->level -= env->step;
        if (env->level <= 0) { env->level = 0; env->stage = ENV_IDLE; }
        break;
    default: break;
    }
    return env->level;
}

static volatile uint16_t pots[NPOTS];           /* ADC DMA target           */
static volatile uint32_t adc_start_failures;    /* failed scans, for debugging */
static int16_t  sine_lut[256];

/* ---------------- audio: one sample per TIM6 tick ---------------- */
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
    if (htim->Instance != TIM6) return;

    static uint32_t phase[NNOTES];
    static envelope_t env[NNOTES];
    static int32_t lp = 0;

    int32_t mix = 0;
    for (int v = 0; v < NNOTES; v++) {
        int32_t level = envelope_tick(&env[v], gate[v]);
        if (level == 0) continue;             /* silent voice: skip */

        uint32_t p = phase[v] += phase_inc[v];

        int32_t x;
        switch (wave) {
        case 0:  x = (int32_t)(p >> 16) - 32768;                      break; /* saw    */
        case 1:  x = (p & 0x80000000u) ? 32767 : -32768;              break; /* square */
        case 2: {int32_t t = (int32_t)(p >> 15);                             /* tri    */
                 x = (t < 65536 ? t : 131071 - t) - 32768;}           break;
        default: x = sine_lut[p >> 24];                               break; /* sine   */
        }

        mix += (x * (level >> 15)) >> 15;       /* Q15 gain keeps product in int32 */
    }
    mix /= NNOTES;                             /* all voices at full scale fit */

    /* one-pole low-pass, shared by all voices */
    lp += ((mix - lp) * lp_k) >> 15;

    int32_t out = lp;

    frames[fill_buf].s[fill_idx] = (int16_t)out;
    if (++fill_idx >= BLOCK) {
        fill_idx = 0;

        /* snapshot the controls into the finished frame */
        frame_t *fr = &frames[fill_buf];
        for (int i = 0; i < NPOTS; i++) fr->pots[i] = pots[i];
        uint8_t keys = 0;
        for (int v = 0; v < NNOTES; v++) keys |= gate[v] << v;
        fr->keys = keys;
        fr->wave = wave;

        uint8_t next = fill_buf ^ 1;
        if (tx_buf == (int8_t)next) {
            /* DMA still reading the other buffer: no free buffer, so drop
               this block and refill the same one rather than tear a frame */
            dropped_frames++;
        } else {
            if (ready_buf == (int8_t)next) dropped_frames++;  /* never sent */
            ready_buf = (int8_t)fill_buf;
            fill_buf  = next;
        }
    }
}

/* ---------------- UART DMA done: release the buffer ---------------- */
void HAL_UART_TxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART2) tx_buf = -1;
}

void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart)
{
    /* a DMA error aborts the transfer without TxCplt; don't stay stuck */
    if (huart->Instance == USART2 && huart->gState == HAL_UART_STATE_READY)
        tx_buf = -1;
}

/* ---------------- controls: ~500 Hz, outside the ISR ---------------- */
static void control_update(void)
{
    /* --- buttons (active low), one voice each --- */
    static GPIO_TypeDef *const port[NNOTES] = { GPIOB, GPIOB, GPIOB, GPIOA };
    static const uint16_t      pin [NNOTES] = { GPIO_PIN_5, GPIO_PIN_4,
                                                GPIO_PIN_10, GPIO_PIN_8 };
    for (int i = 0; i < NNOTES; i++) {
        if (HAL_GPIO_ReadPin(port[i], pin[i]) == GPIO_PIN_RESET) {
            /* each note's pot sweeps 4 octaves, 55 Hz .. 880 Hz, exponential */
            float f = 55.0f * exp2f(pots[i] * (4.0f / 4095.0f));
            phase_inc[i] = (uint32_t)(f * (4294967296.0f / FS));
            gate[i] = 1;
        } else {
            gate[i] = 0;   /* phase_inc kept so the release tail keeps its pitch */
        }
    }

    /* --- B1 cycles waveform after 20 ms of stable input --- */
    static GPIO_PinState b1_last = GPIO_PIN_SET;
    static GPIO_PinState b1_stable = GPIO_PIN_SET;
    static uint32_t b1_changed;
    GPIO_PinState b1 = HAL_GPIO_ReadPin(GPIOC, GPIO_PIN_13);
    uint32_t now = HAL_GetTick();
    if (b1 != b1_last) { b1_last = b1; b1_changed = now; }
    if (b1 != b1_stable && now - b1_changed >= 20) {
        b1_stable = b1;
        if (b1 == GPIO_PIN_RESET) wave = (wave + 1) & 3;
    }

    /* --- cutoff: coefficient 1/128 .. 1, exponential --- */
    lp_k = (int32_t)(32767.0f * exp2f((pots[4] / 4095.0f - 1.0f) * 7.0f));

    /* --- ADSR: attack/decay 2 ms .. 2 s; release 10 ms .. ~2 s --- */
    float t_rel = 0.01f * exp2f(pots[5] * (7.6f / 4095.0f));
    float t_attack = 0.002f * exp2f(pots[6] * (9.965784f / 4095.0f));
    float t_decay = 0.002f * exp2f(pots[7] * (9.965784f / 4095.0f));
    attack_step = (int32_t)((float)ENV_MAX / (t_attack * FS));
    decay_samples = (int32_t)(t_decay * FS + 0.5f);
    release_samples = (int32_t)(t_rel * FS + 0.5f);
    sustain_level = (int32_t)((int64_t)pots[8] * ENV_MAX / 4095);

    /* --- start the next pot scan; results are used next update --- */
    if (HAL_ADC_Start_DMA(&hadc1, (uint32_t *)pots, NPOTS) != HAL_OK)
        adc_start_failures++;
}

void synth_init(void)
{
    for (int i = 0; i < 256; i++)
        sine_lut[i] = (int16_t)(32767.0f * sinf(i * (6.2831853f / 256.0f)));

    if (HAL_ADC_Start_DMA(&hadc1, (uint32_t *)pots, NPOTS) != HAL_OK)
        adc_start_failures++;
    HAL_TIM_Base_Start_IT(&htim6);
}

void synth_loop(void)
{
    if (tx_buf < 0 && ready_buf >= 0) {
        /* claim the frame atomically so the ISR can't swap into it between
           reading ready_buf and marking it DMA-owned */
        __disable_irq();
        int8_t r  = ready_buf;
        ready_buf = -1;
        tx_buf    = r;
        __enable_irq();

        /* 534 bytes take ~5.8 ms at 921600 baud; a block lasts ~11.6 ms */
        if (HAL_UART_Transmit_DMA(&huart2, (uint8_t *)&frames[r],
                                  sizeof(frame_t)) != HAL_OK) {
            /* not started: hand it back unless a newer frame arrived */
            __disable_irq();
            tx_buf = -1;
            if (ready_buf < 0) ready_buf = r;
            __enable_irq();
        }
    }

    static uint32_t last = 0;
    if (HAL_GetTick() - last >= 2) {
        last = HAL_GetTick();
        control_update();
    }
}
