#ifndef AUDIO_BLOCK_H
#define AUDIO_BLOCK_H

typedef struct {
    float *samples;
    int count;
} AudioBlock;

void apply_gain(AudioBlock block, float gain);

#endif