#include <math.h>
void Dense_0(float input_Dense[16],float output_Dense[20],float bias[20],float weight[320]){
	loop_for_a_Dense_0:
	for (int i = 0; i < 20; i++){
		float s=0;
		loop_for_b_Dense_0:
		for (int j = 0; j < 16; j++){
			s+=input_Dense[j]*weight[j*20+i];
		}
		if ((s+bias[i])<0) output_Dense[i]=0; else output_Dense[i]=s+bias[i];
	}
}
void Dense_1(float input_Dense[20],float output_Dense[15],float bias[15],float weight[300]){
	loop_for_a_Dense_1:
	for (int i = 0; i < 15; i++){
		float s=0;
		loop_for_b_Dense_1:
		for (int j = 0; j < 20; j++){
			s+=input_Dense[j]*weight[j*15+i];
		}
		if ((s+bias[i])<0) output_Dense[i]=0; else output_Dense[i]=s+bias[i];
	}
}
void Dense_2(float input_Dense[15],float output_Dense[10],float bias[10],float weight[150]){
	loop_for_a_Dense_2:
	for (int i = 0; i < 10; i++){
		float s=0;
		loop_for_b_Dense_2:
		for (int j = 0; j < 15; j++){
			s+=input_Dense[j]*weight[j*10+i];
		}
		if ((s+bias[i])<0) output_Dense[i]=0; else output_Dense[i]=s+bias[i];
	}
}
void Dense_3(float input_Dense[10],float &output_Dense,float bias[7],float weight[70]){
	float out_Dense[7];
	loop_for_a_Dense_3:
	for (int i = 0; i < 7; i++){
		float s=0;
		loop_for_b_Dense_3:
		for (int j = 0; j < 10; j++){
			s+=input_Dense[j]*weight[j*7+i];
		}
		out_Dense[i]=s+bias[i];
	}

	float sum = 0;
	for (int i = 0; i < 7; i++){

			sum += expf(out_Dense[i]);
		}
	for (int i = 0; i < 7; i++){

			out_Dense[i] = exp(out_Dense[i]) / sum;
		}




	float maxindex=out_Dense[0];
	output_Dense = 0;
	loop_detect:
	for (int i=1; i<7; i++){
		if (out_Dense[i]>maxindex) {
			maxindex=out_Dense[i];
			output_Dense=i;
		}
	}
}
