#ifndef SEMG_CLASSIFIER_H
#define SEMG_CLASSIFIER_H

#include <stdbool.h>
#include <stdint.h>

typedef enum
{
    SEMG_CLASS_CLOSE = 0,
    SEMG_CLASS_EXTEND = 1,
    SEMG_CLASS_REST = 2,
    SEMG_CLASS_UNKNOWN = 255
} SEMG_Class;

void SEMG_Classifier_Init(void);

/*
 * Returns true whenever a stable classification is ready to transmit.
 * Stable states are repeated periodically so a host that connects late can
 * synchronize without waiting for the user's gesture to change.
 */
bool SEMG_Classifier_AddSample(
    uint16_t channel_1,
    uint16_t channel_2,
    SEMG_Class *new_stable_class
);

const char *SEMG_ClassName(
    SEMG_Class semg_class
);

#endif
