#include <math.h>
#include <stdio.h>
#include "KAN_3class.h"

void KAN_3class_main(int *index, float *x_14, float *x_1, float *x_2, float *x_3, float *x_4, float *x_5, float *x_6, float *x_7, float *x_8, float *x_9, float *x_10, float *x_11, float *x_12, float *x_13)
{
    float x_14_f[3];
    KAN_3class_1( &x_14_f[0],  x_1,  x_2,  x_3,  x_4,  x_5,  x_6,  x_7,  x_8,  x_9, x_10, x_11, x_12, x_13);
    KAN_3class_2( &x_14_f[1],  x_1,  x_2,  x_3,  x_4,  x_5,  x_6,  x_7,  x_8,  x_9, x_10, x_11, x_12, x_13);
    KAN_3class_3( &x_14_f[2],  x_1,  x_2,  x_3,  x_4,  x_5,  x_6,  x_7,  x_8,  x_9, x_10, x_11, x_12, x_13);
    *x_14 = x_14_f[0];
    *index = 0;
    for (int i = 1; i < 3; i++)
    {
        if (*x_14 < x_14_f[i])
        {
            *x_14 = x_14_f[i];
            *index = i;
        }
    }


}