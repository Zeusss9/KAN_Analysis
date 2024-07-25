#include <conio.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <fstream>
#include <iostream>
#include "CNN.h"
#include "Dense.h"
#define NumberOfPicture 13
#define d 1
int main(){
	float OutModel;
	float* Weights = (float*)malloc(739 * sizeof(float));
	float tmp;
	FILE* Weight = fopen("Float_Weights.txt", "r");
	for (int i = 0; i < 739; i++){
		fscanf(Weight, "%f", &tmp);
		*(Weights + i)=tmp;
	}
	fclose(Weight);
	////read Input
	float* InModel = (float*)malloc((NumberOfPicture * d * 13) * sizeof(float));
	FILE* Input = fopen("X_test.txt", "r");
	for (int i = 0; i < NumberOfPicture * d * 13; i++){
		fscanf(Input, "%f", &tmp);
		*(InModel + i)=tmp;
	}
	fclose(Input);
	//Read Label
	float*Label = (float*)malloc((NumberOfPicture) * sizeof(float));
	FILE* Output = fopen("y_test.txt", "r");
	for (int i = 0; i < NumberOfPicture ; i++)
	{
		fscanf(Output, "%f", &tmp);
		*(Label + i) = tmp;
	}
	fclose(Output);
	float OutArray[NumberOfPicture] = {};
	float Image[d * 13] = {};
	for (int i = 0; i < NumberOfPicture ; i++)
	{
		int startIndex = i * d * 13;
		for (int k = 0; k < d * 13; k++)
		{
			Image[k] = *(InModel + startIndex + k);
		}
		CNN(Image, OutModel, Weights);
		OutArray[i] = OutModel;
	}
	float countTrue = 0;
	for (int i = 0; i < NumberOfPicture; i++)
	{
		int labelValue = *(Label + i);
		if (labelValue == OutArray[i])
		{
			countTrue = countTrue + 1;
		}
	}
	float accuracy = (float)((countTrue / NumberOfPicture) * 100);
	std::cout << "accuracy of Model: " << accuracy << "%\n";
	//std::cout << "Result: " <<  OutModel <<  "\n";
	return 0;
}
