#include <stdio.h>
#include "audio_block.h"

int main(void) {

    float sample[]={0.8f,0.4f,0.2f};
    AudioBlock block={sample,3};

    apply_gain(block,0.25f);

    for (int i = 0; i < 3; i++) {
        
        printf("%.2f\n", sample[i]);
    }

    return 0;
}