#include "Dense.h"
#include <algorithm>
#include <string.h>
void CNN(float InModel[13],float &OutModel,float Weights[739]){
	float OutDense0[32];
	float OutDense1[8];
	Dense_0(&InModel[0],OutDense0,&Weights[416],&Weights[0]);
	Dense_1(OutDense0,OutDense1,&Weights[704],&Weights[448]);
	Dense_2(OutDense1,OutModel,&Weights[736],&Weights[712]);
}
