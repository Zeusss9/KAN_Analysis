#include "Dense.h"
#include <algorithm>
#include <string.h>
#include <math.h>

void CNN(float InModel[8],float &OutModel,float Weights[4866]){
	float OutDense0[64];
	float OutDense1[64];
	Dense_0(&InModel[0],OutDense0,&Weights[512],&Weights[0]);
	Dense_1(OutDense0,OutDense1,&Weights[4672],&Weights[576]);
	Dense_2(OutDense1,OutModel,&Weights[4864],&Weights[4736]);
}
