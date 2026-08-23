/*
 * semg_classifier.c
 *
 * Created on: Aug 17, 2026
 * Author: jehyu
 */

#include "semg_classifier.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#include "semg_lda_model.h"


#define SEMG_STEP_SAMPLES 100U
#define SEMG_FEATURES_PER_CHANNEL 6U
#define SEMG_CROSSING_THRESHOLD 10.0f
#define SEMG_STABLE_WINDOWS_REQUIRED 3U


/*
 * 60 Hz IIR notch filter
 *
 * Designed for:
 *   Sampling rate = 1000 Hz
 *   Notch frequency = 60 Hz
 *   Q = 30
 *
 * These are the same coefficients we tested
 * successfully on the recorded robot-connected signal.
 */
#define SEMG_NOTCH_B0  0.99375596f
#define SEMG_NOTCH_B1 -1.84794186f
#define SEMG_NOTCH_B2  0.99375596f

#define SEMG_NOTCH_A1 -1.84794186f
#define SEMG_NOTCH_A2  0.98751193f


/*
 * State required by one second-order IIR filter.
 *
 * x1 = x[n-1]
 * x2 = x[n-2]
 * y1 = y[n-1]
 * y2 = y[n-2]
 */
typedef struct
{
    float x1;
    float x2;
    float y1;
    float y2;

    bool initialized;
} SEMG_NotchFilterState;


/*
 * These buffers now contain FILTERED floating-point
 * samples instead of raw uint16_t ADC values.
 */
static float channel_1_ring[
    SEMG_WINDOW_SAMPLES
];

static float channel_2_ring[
    SEMG_WINDOW_SAMPLES
];


/*
 * Each EMG channel requires its own independent
 * filter state.
 */
static SEMG_NotchFilterState channel_1_notch;
static SEMG_NotchFilterState channel_2_notch;


static uint32_t ring_write_index = 0U;
static uint32_t samples_in_ring = 0U;
static uint32_t samples_since_prediction = 0U;

static SEMG_Class candidate_class =
    SEMG_CLASS_UNKNOWN;

static SEMG_Class stable_class =
    SEMG_CLASS_UNKNOWN;

static uint32_t candidate_count = 0U;


/*
 * Apply one sample of the causal 60 Hz notch filter.
 *
 * Difference equation:
 *
 * y[n] =
 *      b0*x[n]
 *    + b1*x[n-1]
 *    + b2*x[n-2]
 *    - a1*y[n-1]
 *    - a2*y[n-2]
 */
static float apply_notch_filter(
    SEMG_NotchFilterState *state,
    float input_sample
)
{
    /*
     * Initialize the filter around the current ADC
     * baseline instead of zero.
     *
     * This prevents a large startup transient caused
     * by the ~2000-count DC offset of the SEN0240.
     */
    if (!state->initialized)
    {
        state->x1 = input_sample;
        state->x2 = input_sample;

        state->y1 = input_sample;
        state->y2 = input_sample;

        state->initialized = true;

        return input_sample;
    }

    float output_sample =
        SEMG_NOTCH_B0 * input_sample
        + SEMG_NOTCH_B1 * state->x1
        + SEMG_NOTCH_B2 * state->x2
        - SEMG_NOTCH_A1 * state->y1
        - SEMG_NOTCH_A2 * state->y2;


    /*
     * Shift the filter history for the next sample.
     */
    state->x2 = state->x1;
    state->x1 = input_sample;

    state->y2 = state->y1;
    state->y1 = output_sample;


    return output_sample;
}


static float get_ordered_sample(
    const float *ring_buffer,
    uint32_t ordered_index
)
{
    uint32_t ring_index =
        (
            ring_write_index
            + ordered_index
        )
        % SEMG_WINDOW_SAMPLES;

    return ring_buffer[ring_index];
}


static void extract_channel_features(
    const float *ring_buffer,
    float *features
)
{
    float signal_sum = 0.0f;

    for (
        uint32_t index = 0U;
        index < SEMG_WINDOW_SAMPLES;
        index++
    )
    {
        signal_sum += get_ordered_sample(
            ring_buffer,
            index
        );
    }


    float signal_mean =
        signal_sum
        / (float)SEMG_WINDOW_SAMPLES;


    float squared_sum = 0.0f;
    float absolute_sum = 0.0f;
    float waveform_length_sum = 0.0f;

    uint32_t zero_crossing_count = 0U;
    uint32_t slope_change_count = 0U;


    float previous_centered =
        get_ordered_sample(
            ring_buffer,
            0U
        )
        - signal_mean;


    squared_sum +=
        previous_centered
        * previous_centered;

    absolute_sum += fabsf(
        previous_centered
    );


    float previous_difference = 0.0f;


    for (
        uint32_t index = 1U;
        index < SEMG_WINDOW_SAMPLES;
        index++
    )
    {
        float centered_sample =
            get_ordered_sample(
                ring_buffer,
                index
            )
            - signal_mean;


        squared_sum +=
            centered_sample
            * centered_sample;


        absolute_sum += fabsf(
            centered_sample
        );


        float difference =
            centered_sample
            - previous_centered;


        waveform_length_sum += fabsf(
            difference
        );


        if (
            (
                previous_centered
                * centered_sample
            )
            < 0.0f
            && fabsf(difference)
                >= SEMG_CROSSING_THRESHOLD
        )
        {
            zero_crossing_count++;
        }


        if (
            index >= 2U
            && (
                previous_difference
                * difference
            )
                < 0.0f
            && fabsf(
                previous_difference
                - difference
            )
                >= SEMG_CROSSING_THRESHOLD
        )
        {
            slope_change_count++;
        }


        previous_centered =
            centered_sample;

        previous_difference =
            difference;
    }


    float mean_square =
        squared_sum
        / (float)SEMG_WINDOW_SAMPLES;


    float rms =
        sqrtf(mean_square);


    float mav =
        absolute_sum
        / (float)SEMG_WINDOW_SAMPLES;


    float waveform_length =
        waveform_length_sum
        / (float)(
            SEMG_WINDOW_SAMPLES - 1U
        );


    float zero_crossing_rate =
        (float)zero_crossing_count
        / (float)(
            SEMG_WINDOW_SAMPLES - 1U
        );


    float slope_change_rate =
        (float)slope_change_count
        / (float)(
            SEMG_WINDOW_SAMPLES - 2U
        );


    features[0] = rms;
    features[1] = mav;

    /*
     * Because the signal is mean-centered,
     * standard deviation == RMS here.
     */
    features[2] = rms;

    features[3] = waveform_length;
    features[4] = zero_crossing_rate;
    features[5] = slope_change_rate;
}


static SEMG_Class classify_current_window(void)
{
    float features[
        SEMG_FEATURE_COUNT
    ];


    extract_channel_features(
        channel_1_ring,
        &features[0]
    );


    extract_channel_features(
        channel_2_ring,
        &features[
            SEMG_FEATURES_PER_CHANNEL
        ]
    );


    int predicted_class =
        semg_lda_predict(features);


    if (
        predicted_class < 0
        || predicted_class
            >= SEMG_CLASS_COUNT
    )
    {
        return SEMG_CLASS_UNKNOWN;
    }


    return (SEMG_Class)predicted_class;
}


void SEMG_Classifier_Init(void)
{
    memset(
        channel_1_ring,
        0,
        sizeof(channel_1_ring)
    );


    memset(
        channel_2_ring,
        0,
        sizeof(channel_2_ring)
    );


    /*
     * Reset both notch filters.
     */
    memset(
        &channel_1_notch,
        0,
        sizeof(channel_1_notch)
    );


    memset(
        &channel_2_notch,
        0,
        sizeof(channel_2_notch)
    );


    ring_write_index = 0U;
    samples_in_ring = 0U;
    samples_since_prediction = 0U;


    candidate_class =
        SEMG_CLASS_UNKNOWN;


    stable_class =
        SEMG_CLASS_UNKNOWN;


    candidate_count = 0U;
}


bool SEMG_Classifier_AddSample(
    uint16_t channel_1,
    uint16_t channel_2,
    SEMG_Class *new_stable_class
)
{
    /*
     * NEW:
     *
     * Remove the dominant 60 Hz interference before
     * putting samples into the feature window.
     */
    float filtered_channel_1 =
        apply_notch_filter(
            &channel_1_notch,
            (float)channel_1
        );


    float filtered_channel_2 =
        apply_notch_filter(
            &channel_2_notch,
            (float)channel_2
        );


    /*
     * The ring buffers therefore contain filtered EMG.
     */
    channel_1_ring[
        ring_write_index
    ] = filtered_channel_1;


    channel_2_ring[
        ring_write_index
    ] = filtered_channel_2;


    ring_write_index++;


    if (
        ring_write_index
        >= SEMG_WINDOW_SAMPLES
    )
    {
        ring_write_index = 0U;
    }


    bool prediction_due = false;


    if (
        samples_in_ring
        < SEMG_WINDOW_SAMPLES
    )
    {
        samples_in_ring++;


        if (
            samples_in_ring
            == SEMG_WINDOW_SAMPLES
        )
        {
            prediction_due = true;

            samples_since_prediction = 0U;
        }
    }
    else
    {
        samples_since_prediction++;


        if (
            samples_since_prediction
            >= SEMG_STEP_SAMPLES
        )
        {
            prediction_due = true;

            samples_since_prediction = 0U;
        }
    }


    if (!prediction_due)
    {
        return false;
    }


    SEMG_Class prediction =
        classify_current_window();


    if (
        prediction
        == SEMG_CLASS_UNKNOWN
    )
    {
        candidate_class =
            SEMG_CLASS_UNKNOWN;

        candidate_count = 0U;

        return false;
    }


    if (
        prediction
        == candidate_class
    )
    {
        if (
            candidate_count
            < SEMG_STABLE_WINDOWS_REQUIRED
        )
        {
            candidate_count++;
        }
    }
    else
    {
        candidate_class = prediction;

        candidate_count = 1U;
    }


    if (
        candidate_count
            >= SEMG_STABLE_WINDOWS_REQUIRED
    )
    {
        stable_class = candidate_class;


        if (
            new_stable_class
            != NULL
        )
        {
            *new_stable_class =
                stable_class;
        }


        return true;
    }


    return false;
}


const char *SEMG_ClassName(
    SEMG_Class semg_class
)
{
    switch (semg_class)
    {
        case SEMG_CLASS_CLOSE:
            return "CLOSE";


        case SEMG_CLASS_EXTEND:
            return "EXTEND";


        case SEMG_CLASS_REST:
            return "REST";


        default:
            return "UNKNOWN";
    }
}
