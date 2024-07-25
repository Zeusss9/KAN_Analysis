#include <math.h>
void Dense_0(float input_Dense[2],float output_Dense[8],float bias[8],float weight[16]){
	loop_for_a_Dense_0:
	for (int i = 0; i < 8; i++){
		float s=0;
		loop_for_b_Dense_0:
		for (int j = 0; j < 2; j++){
			s+=input_Dense[j]*weight[j*8+i];
		}
		if ((s+bias[i])<0) output_Dense[i]=0; else output_Dense[i]=s+bias[i];
	}
}
void Dense_1(float input_Dense[8],float output_Dense[4],float bias[4],float weight[32]){
	loop_for_a_Dense_1:
	for (int i = 0; i < 4; i++){
		float s=0;
		loop_for_b_Dense_1:
		for (int j = 0; j < 8; j++){
			s+=input_Dense[j]*weight[j*4+i];
		}
		if ((s+bias[i])<0) output_Dense[i]=0; else output_Dense[i]=s+bias[i];
	}
}
void Dense_2(float input_Dense[4],float &output_Dense,float bias[1],float weight[4]){
	float out_Dense[1];
	loop_for_a_Dense_2:
	for (int i = 0; i < 1; i++){
		float s=0;
		loop_for_b_Dense_2:
		for (int j = 0; j < 4; j++){
			s+=input_Dense[j]*weight[j*1+i];
		}
		out_Dense[i]=s+bias[i];
	}

	out_Dense[0] = 1 / (1 + expf(-1 * out_Dense[0]));



	float maxindex = out_Dense[0];
	float z = 1 - out_Dense[0];
	output_Dense = 1;
	loop_detect:
	for (int i=0; i<1; i++){
		if (z>maxindex) {
			maxindex=z;
			output_Dense=i;
		}
	}
}
