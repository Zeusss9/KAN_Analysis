#include <math.h>
#include <stdio.h>
#include <stdlib.h>


void Kan_1(float *x1, float *x2, float *x3)
{
    float a = 0, b = 0;
    a = -0.52 * tanhf(3.75 * sinf(1.06 * *x1 + 1.79) - 0.21 + 1.68 * sinf(1.83 * *x2 -0.02));
    b = 0.54 * tanhf(2.31 * sinf(2.16 * *x1 - 4.99) - 2.58 + 2.69 * sinf(1.56 * *x2 + 2.2));
    *x3 = a + b + 1.03;

}
