#ifndef DSP_FIR_H
#define DSP_FIR_H

typedef struct fir_state fir_state_t;
fir_state_t *fir_create(const float *h,int nh);

void fir_process(
    fir_state_t *s,
    const float *in,
    float *out,
    int n
);

void fir_destroy(fir_state_t *s);

#endif
