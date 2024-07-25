#include <math.h>
void Dense_0(float input_Dense[8],float output_Dense[64],float bias[64],float weight[512]){
	loop_for_a_Dense_0:
	for (int i = 0; i < 64; i++){
		float s=0;
		loop_for_b_Dense_0:
		for (int j = 0; j < 8; j++){
			s+=input_Dense[j]*weight[j*64+i];
		}
		if ((s+bias[i])<0) output_Dense[i]=0; else output_Dense[i]=s+bias[i];
	}
}
void Dense_1(float input_Dense[64],float output_Dense[64],float bias[64],float weight[4096]){
	loop_for_a_Dense_1:
	for (int i = 0; i < 64; i++){
		float s=0;
		loop_for_b_Dense_1:
		for (int j = 0; j < 64; j++){
			s+=input_Dense[j]*weight[j*64+i];
		}
		if ((s+bias[i])<0) output_Dense[i]=0; else output_Dense[i]=s+bias[i];
	}
}
void Dense_2(float input_Dense[64],float &output_Dense,float bias[2],float weight[128]){
	float out_Dense[2];
	loop_for_a_Dense_2:
	for (int i = 0; i < 2; i++){
		float s=0;
		loop_for_b_Dense_2:
		for (int j = 0; j < 64; j++){
			s+=input_Dense[j]*weight[j*2+i];
		}
		out_Dense[i]=s+bias[i];
	}

	float sum = 0;
	for (int i = 0; i < 2; i++){

			sum += expf(out_Dense[i]);
		}
	for (int i = 0; i < 2; i++){

			out_Dense[i] = exp(out_Dense[i]) / sum;
		}




	float maxindex=out_Dense[0];
	output_Dense = 0;
	loop_detect:
	for (int i=1; i<2; i++){
		if (out_Dense[i]>maxindex) {
			maxindex=out_Dense[i];
			output_Dense=i;
		}
	}
}
