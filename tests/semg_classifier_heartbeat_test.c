#include <stdint.h>
#include <stdio.h>

#include "semg_classifier.h"


int main(void)
{
    SEMG_Class stable_class = SEMG_CLASS_UNKNOWN;
    uint32_t output_count = 0U;

    SEMG_Classifier_Init();

    for (uint32_t sample = 0U; sample < 700U; sample++)
    {
        if (
            SEMG_Classifier_AddSample(
                2048U,
                2048U,
                &stable_class
            )
        )
        {
            output_count++;
        }
    }

    if (stable_class != SEMG_CLASS_REST)
    {
        fprintf(stderr, "Expected REST for a constant input.\n");
        return 1;
    }

    if (output_count < 3U)
    {
        fprintf(
            stderr,
            "Stable intent was not repeated: %lu outputs.\n",
            (unsigned long)output_count
        );
        return 2;
    }

    printf(
        "Heartbeat test passed with %lu REST outputs.\n",
        (unsigned long)output_count
    );

    return 0;
}
