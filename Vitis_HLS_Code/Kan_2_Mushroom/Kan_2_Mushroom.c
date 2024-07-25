#include <math.h>
#include <stdio.h>
#include "Kan_2_Mushroom.h"
void KAN_2class_Mushroom(int *index, float *x_9, float *x_1, float *x_2, float *x_3, float *x_4, float *x_5, float *x_6, float *x_7, float *x_8) 
{
    float x_17_f[2];
    KAN_2class_Mushroom_1( &x_17_f[0],  x_1,  x_2,  x_3,  x_4,  x_5,  x_6,  x_7,  x_8);
    KAN_2class_Mushroom_2( &x_17_f[1],  x_1,  x_2,  x_3,  x_4,  x_5,  x_6,  x_7,  x_8);
    *x_9 = x_17_f[0];
    *index = 0;

    if (*x_9 < x_17_f[1])
    {
        *x_9 = x_17_f[1];
        *index = 1;
    }
 
}