#include "Dense.h"
#include <algorithm>
#include <string.h>
#include <math.h>
void CNN(float InModel[2],float &OutModel,float Weights[65]){
	float OutDense0[8];
	float OutDense1[4];
	Dense_0(&InModel[0],OutDense0,&Weights[16],&Weights[0]);
	Dense_1(OutDense0,OutDense1,&Weights[56],&Weights[24]);
	Dense_2(OutDense1,OutModel,&Weights[64],&Weights[60]);
}
