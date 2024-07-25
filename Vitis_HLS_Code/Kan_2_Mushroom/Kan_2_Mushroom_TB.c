#include <math.h>
#include <stdio.h>
#include "Kan_2_Mushroom_main.h"


int main() {

    float x_1 = 1;
    float x_2 = 1;
    float x_3 = 1;
    float x_4 = 1;
    float x_5 = 1;
    float x_6 = 1;
    float x_7 = 1;
    float x_8 = 1;
    float x_9;
    int i;
    KAN_2class_Mushroom(&i, &x_9, &x_1, &x_2, &x_3, &x_4, &x_5, &x_6, &x_7, &x_8);
    printf("Result: %f, %d", x_9, i);
    return 0;
}
