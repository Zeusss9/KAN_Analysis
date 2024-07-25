#include <math.h>
void Dense_0(float input_Dense[13],float output_Dense[32],float bias[32],float weight[416]){
	loop_for_a_Dense_0:
	for (int i = 0; i < 32; i++){
		float s=0;
		loop_for_b_Dense_0:
		for (int j = 0; j < 13; j++){
			s+=input_Dense[j]*weight[j*32+i];
		}
		if ((s+bias[i])<0) output_Dense[i]=0; else output_Dense[i]=s+bias[i];
	}
}
void Dense_1(float input_Dense[32],float output_Dense[8],float bias[8],float weight[256]){
	loop_for_a_Dense_1:
	for (int i = 0; i < 8; i++){
		float s=0;
		loop_for_b_Dense_1:
		for (int j = 0; j < 32; j++){
			s+=input_Dense[j]*weight[j*8+i];
		}
		if ((s+bias[i])<0) output_Dense[i]=0; else output_Dense[i]=s+bias[i];
	}
}
void Dense_2(float input_Dense[8],float &output_Dense,float bias[3],float weight[24]){
	float out_Dense[3];
	loop_for_a_Dense_2:
	for (int i = 0; i < 3; i++){
		float s=0;
		loop_for_b_Dense_2:
		for (int j = 0; j < 8; j++){
			s+=input_Dense[j]*weight[j*3+i];
		}
		out_Dense[i]=s+bias[i];
	}


	float sum = 0;
	for (int i = 0; i < 3; i++){

			sum += expf(out_Dense[i]);
		}
	for (int i = 0; i < 3; i++){

			out_Dense[i] = exp(out_Dense[i]) / sum;
		}


	float maxindex=out_Dense[0];
	output_Dense = 0;
	loop_detect:
	for (int i=1; i<3; i++){
		if (out_Dense[i]>maxindex) {
			maxindex=out_Dense[i];
			output_Dense=i;
		}
	}
}
