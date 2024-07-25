#include <math.h>
#include <stdio.h>
#include "KAN_7class.h"

void KAN_7class(int *index, float *x_17, float *x_1, float *x_2, float *x_3, float *x_4, float *x_5, float *x_6, float *x_7, float *x_8, float *x_9,float *x_10,float *x_11,float *x_12,float *x_13,float *x_14,float *x_15,float *x_16) 
{
    float x_17_f[7];
    KAN_7class_1( &x_17_f[0],  x_1,  x_2,  x_3,  x_4,  x_5,  x_6,  x_7,  x_8,  x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16) ;
    KAN_7class_2( &x_17_f[1],  x_1,  x_2,  x_3,  x_4,  x_5,  x_6,  x_7,  x_8,  x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16) ;
    KAN_7class_3( &x_17_f[2],  x_1,  x_2,  x_3,  x_4,  x_5,  x_6,  x_7,  x_8,  x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16) ;
    KAN_7class_4( &x_17_f[3],  x_1,  x_2,  x_3,  x_4,  x_5,  x_6,  x_7,  x_8,  x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16) ;
    KAN_7class_5( &x_17_f[4],  x_1,  x_2,  x_3,  x_4,  x_5,  x_6,  x_7,  x_8,  x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16) ;
    KAN_7class_6( &x_17_f[5],  x_1,  x_2,  x_3,  x_4,  x_5,  x_6,  x_7,  x_8,  x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16) ;
    KAN_7class_7( &x_17_f[6],  x_1,  x_2,  x_3,  x_4,  x_5,  x_6,  x_7,  x_8,  x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16) ;
    *x_17 = x_17_f[0];
    *index = 0;
    for (int i = 1; i < 7; i++)
    {
        if (*x_17 < x_17_f[i])
        {
            *x_17 = x_17_f[i];
            *index = i;
        }
    }





}