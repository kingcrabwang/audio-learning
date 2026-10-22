#include "audio_block.h"

void apply_gain(AudioBlock block, float gain) {    
    for (int i = 0; i < block.count; i++) {
        block.samples[i] *= gain;// 在这里修改第 i 个样本
    }
}