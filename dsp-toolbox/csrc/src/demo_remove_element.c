#include <stdio.h>

int removeElement(int *nums,int numsSize,int val){
    int write=0;

    for(int read=0;read<numsSize;read++){
        if(nums[read]!=val){
            nums[write]=nums[read];
            write++;
        }
    }
    return write;
}
int main(void) {
    int nums[] = {4, 1, 4, 6, 3, 4};

    int count = removeElement(nums, 6, 4);

    printf("有效长度：%d\n", count);

    for (int i = 0; i < count; i++) {
        printf("%d ", nums[i]);
    }
    printf("\n");

    return 0;
}