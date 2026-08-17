#ifndef SEMG_LDA_MODEL_H
#define SEMG_LDA_MODEL_H

#define SEMG_CLASS_COUNT 3
#define SEMG_FEATURE_COUNT 12
#define SEMG_WINDOW_SAMPLES 200

static const char *const SEMG_CLASS_NAMES[SEMG_CLASS_COUNT] = {
    "CLOSE",
    "EXTEND",
    "REST",
};

static const float SEMG_LDA_WEIGHTS[SEMG_CLASS_COUNT][SEMG_FEATURE_COUNT] = {
    {0.296989902f, 0.0142188315f, 0.296989902f, 0.164652157f, 24.2276976f, 68.0554584f, 0.0131885492f, 0.308536135f, 0.0131885492f, -0.633689459f, 61.1839852f, 16.0259393f},
    {0.000824467758f, -0.0919217506f, 0.000824467758f, -0.799997587f, 15.4408812f, 36.3756498f, 0.0206067301f, 0.0410917825f, 0.0206067301f, 0.119732714f, 79.1225754f, 33.5054553f},
    {-0.148907185f, 0.0388514595f, -0.148907185f, 0.317672715f, -19.8342894f, -52.2155541f, -0.0168976396f, -0.174813959f, -0.0168976396f, 0.256978372f, -70.1532803f, -24.7656973f},
};

static const float SEMG_LDA_BIAS[SEMG_CLASS_COUNT] = {
    -56.4632989f, -43.2513556f, 27.3790779f
};

static inline int semg_lda_predict(const float features[SEMG_FEATURE_COUNT])
{
    int best_class = 0;
    float best_score = -3.402823466e+38F;

    for (int class_index = 0; class_index < SEMG_CLASS_COUNT; class_index++) {
        float score = SEMG_LDA_BIAS[class_index];

        for (int feature_index = 0; feature_index < SEMG_FEATURE_COUNT; feature_index++) {
            score += SEMG_LDA_WEIGHTS[class_index][feature_index] * features[feature_index];
        }

        if (score > best_score) {
            best_score = score;
            best_class = class_index;
        }
    }

    return best_class;
}

#endif
