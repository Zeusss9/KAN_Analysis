#include "Dense.h"
#include <algorithm>
#include <string.h>

void CNN(float InModel[16],float &OutModel,float Weights[892]){
	float OutDense0[20];
	float OutDense1[15];
	float OutDense2[10];
	Dense_0(&InModel[0],OutDense0,&Weights[320],&Weights[0]);
	Dense_1(OutDense0,OutDense1,&Weights[640],&Weights[340]);
	Dense_2(OutDense1,OutDense2,&Weights[805],&Weights[655]);
	Dense_3(OutDense2,OutModel,&Weights[885],&Weights[815]);
}
