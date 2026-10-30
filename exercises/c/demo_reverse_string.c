#include <stdio.h>

void reverseString(char *s, int sSize) {
    int left = 0;
    int right = sSize - 1;
    char temp;
    while (left < right) {
        // 1. 用 char temp 暂存 s[left]
        temp=s[left];
        // 2. 把 s[right] 赋给 s[left]
        s[left]=s[right];
        // 3. 把 temp 赋给 s[right]
        s[right]=temp;
        // 4. left 增加 1，right 减少 1
        left++;
        right--;
    }
}

int main(void) {
    char s[] = {'m', 'u', 's', 'i', 'c'};

    reverseString(s, 5);

    for (int i = 0; i < 5; i++) {
        printf("%c", s[i]);
    }
    printf("\n");

    return 0;
}