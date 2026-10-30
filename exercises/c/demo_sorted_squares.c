#include <stdio.h>
#include <assert.h>

void sortedSquares(int *nums, int numsSize, int *result) {
    int left=0;
    int right=numsSize-1;
    int index=numsSize-1;
    while (left<=right)
    {
        int left_square=nums[left]*nums[left];
        int right_square=nums[right]*nums[right];
        if(left_square<=right_square){
            result[index]=right_square;
            right--;
        }
        else{
            result[index]=left_square;
            left++;
        }
        index--;
    }
    
}

int main(void) {
    // 检查负数、零、正数，以及首尾平方相等的情况。
    int nums[] = {-7, -3, 0, 2, 7};
    int result[5];
    int expected[] = {0, 4, 9, 49, 49};

    sortedSquares(nums, 5, result);

    for (int i = 0; i < 5; i++) {
        assert(result[i] == expected[i]);
        printf("%d ", result[i]);
    }
    printf("\n");

    // 检查只剩一个元素时，是否也正确处理。
    int single[] = {-4};
    int single_result[1];

    sortedSquares(single, 1, single_result);
    assert(single_result[0] == 16);

    printf("全部测试通过\n");
    return 0;
}