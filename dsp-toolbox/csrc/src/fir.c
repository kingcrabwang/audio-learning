#include <stdlib.h>
#include "dsp/fir.h"

struct fir_state
{
    float *h;
    float *buffer;
    int nh;
    int write_index;
};

fir_state_t *fir_create(const float *h,int nh){
    if(h==NULL || nh<=0){
        return NULL;
    }
    fir_state_t *s=malloc(sizeof(fir_state_t));
    if (s==NULL){
        return NULL;
    }
    s->nh=nh;
    s->write_index=0;
    s->h=malloc(nh*sizeof(float));
    s->buffer=calloc(nh,sizeof(float));
    if(s->h==NULL || s->buffer==NULL){
    //因为结构体s的内存已经在前面申请成功，所以如果数组申请失败了，前边的内存还是会在的，需要清理
        free(s->h);
        free(s->buffer);
        free(s);
        return NULL;
    }
    for (int i=0; i<nh;i++){
        s->h[i]=h[i];
    }
    return s;
}

void fir_destroy(fir_state_t *s){
    if(s==NULL){
        return;
    }

    free(s->h);
    free(s->buffer);
    free(s);
}

void fir_process(
    fir_state_t *s,
    const float *in,
    float *out,
    int n
) {
    // 参数无效或没有输入样本时直接返回，保留原有状态。
    if (s == NULL || in == NULL || out == NULL || n <= 0) {
        return;
    }

    // 外层循环：每读取一个输入样本，就计算一个输出样本。
    // i 是当前音频块的下标，与环形缓冲区的写入位置不同。
    for (int i = 0; i < n; i++) {

        // 把当前输入写入缓冲区，覆盖最旧的样本。
        s->buffer[s->write_index] = in[i];

        // 当前输出的累加值，每处理一个新输入都从零开始。
        float total = 0.0f;

        // 内层循环：遍历全部滤波器系数。
        // k=0 使用当前输入，k=1 使用上一输入，以此类推。
        for (int k = 0; k < s->nh; k++) {

            // 从当前写入位置向过去查找 k 个样本。
            int index = s->write_index - k;

            // 越过数组开头时，从数组末尾绕回。
            // 例如长度为 3 时，下标 -1 对应下标 2。
            if (index < 0) {
                index += s->nh;
            }

            // 第 k 个系数 × 过去 k 步的输入，
            // 将这一项的贡献累加到当前输出。
            total += s->h[k] * s->buffer[index];
        }

        // 全部系数累加完毕，保存当前输出。
        out[i] = total;

        // 写入位置向前移动一格，到达末尾后绕回 0。
        s->write_index = (s->write_index + 1) % s->nh;
    }

    // 缓冲区和写入位置保存在 s 中。
    // 下次调用继续使用这些历史，不在块边界重新清零。
}
