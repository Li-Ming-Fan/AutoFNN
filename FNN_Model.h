
#ifndef FNN_Model_H
#define FNN_Model_H

#include "FloatMat.h"

#include <string.h>
#include <stdio.h>

class FNN_Model
{
private:
	int NumLayers;
	int * ArrayNumNodes;
	int * ArrayActs;

public:
	FloatMat * Weights;
	FloatMat * Shifts;
	//
	//FloatMat * Connects;
	//FloatMat MatRects;

	//
	static const int ACT_RELU = 0;
	static const int ACT_LOGS = 1;
	static const int ACT_LOGB = 2;
	static const int ACT_RELB = 3;
	//

	// training samples related
	//
	float LearningPortion;    //  TrainingSamples = LearningSamples + ValidationSamples
	int SeedLearning;
	float CriteriaAssertion;
	//
	FloatMat LearningSamples;
	FloatMat LearningLabels;
	FloatMat ValidationSamples;
	FloatMat ValidationLabels;
	//
	// learning paras
	//
	int FlagErrBalance;
	int FlagLearningMethod;
	int FlagAlpha;
	int FlagMomentum;
	//
	int MaxIter;
	float error_tol;
	float gradient_tol;
	//
	float alpha_threshold;
	//
	float alpha;
	float beta;
	float delta;
	//
	float lamda;   // for momentum
	//
	float epsilon;   // for regulation
	//
	// gdd, descending alpha, no momentum
	// gda, floating alpha, no momentum
	//
	// gdm, descending alpha, exp momentum
	// gdx, floating alpha, exp momentum
	//
	static const int LEARN_GD = 0;
	static const int LEARN_LM = 1;
	//
	static const int ALPHA_PLAIN = 0;
	static const int ALPHA_DES = 1;
	static const int ALPHA_ADA = 2;
	//
	static const int MOMENTUM_NONE = 0;
	static const int MOMENTUM_PREV = 1;
	static const int MOMENTUM_EXP = 2;
	//

	// performance
	float performance[5];
	//

	//
	FNN_Model(void)
	{
		NumLayers = 1;
		ArrayNumNodes = new int[1];
		ArrayNumNodes[0] = 1;

		ArrayActs = new int[1];
		ArrayActs[0] = ACT_LOGS;

		Weights = new FloatMat[1];
		Shifts = new FloatMat[1];

		//
		LearningPortion = 0.7;
		SeedLearning = 10;
		CriteriaAssertion = 0.8;
		//
		FlagErrBalance = 0;
		FlagLearningMethod = LEARN_GD;   //
		FlagAlpha = ALPHA_PLAIN;
		FlagMomentum = MOMENTUM_NONE;
		//
		MaxIter = 100;
		error_tol = 0.0001;
		gradient_tol = 0.0001;
		//
		alpha_threshold = 0.0002;
		//
		alpha = 0.001;
		beta = 0.999;
		delta = 0.00001;
		//
		lamda = 0.6;
		//
		epsilon = 0.00001;
		//

	}
	//
	void setLearningParasDefault()
	{
		//
		//MaxIter = 1000;
		//error_tol = 0.0001;
		//gradient_tol = 0.0001;
		//
		//FlagErrBalance = 0;
		//FlagTrainingMethod = TRAIN_GDX;   //
		//
		if (FlagLearningMethod == LEARN_LM)
		{
			alpha = 0.1;
			beta = 0.1;
		}
		else
		{
			// gdd, gda, gdm, gdx,
			alpha = 0.001;
			beta = 0.999;
			//
			delta = 0.00001;
			//
			lamda = 0.6;
			//
		}

	}
	//
	~FNN_Model()
	{
		delete [] ArrayNumNodes;
		delete [] ArrayActs;
		//
		delete [] Weights;
		delete [] Shifts;
	}
	//

	//
	void setStructureFNN(int Num, int * Array)
	{
		//
		delete [] ArrayNumNodes;
		delete [] ArrayActs;

		delete [] Weights;
		delete [] Shifts;

		//
		NumLayers = Num;
		ArrayNumNodes = new int[Num];
		for (int i = 0; i < Num; i++)
		{
			ArrayNumNodes[i] = Array[i];
		}
		//
		int NumMat = Num - 1;
		//
		ArrayActs = new int[NumMat];
		ArrayActs[NumMat-1] = ACT_LOGS;
		//
		for (int i = NumMat-2; i >= 0; i--)
		{
			ArrayActs[i] = ACT_LOGB;
		}

		//最左边是输入层，最右边是输出层，
		//
		//行样本，一行是一个样本
		//DataSample * Weights[0] + Shifts[0]
		//--> ActivationFunction = 第一个隐含层的输出
		//

		//
		Weights = new FloatMat[NumMat]; // 调用无参构造函数
		Shifts = new FloatMat[NumMat];

		for (int i = 0; i < NumMat; i++)
		{
			Weights[i].setMatSize(ArrayNumNodes[i], ArrayNumNodes[i+1]);
			Shifts[i].setMatSize(1, ArrayNumNodes[i+1]);
		}
	}
	//
	void setActSingleLayer(int layer, int type)
	{
		ArrayActs[layer] = type;
	}
	void setActArray(int * Array)
	{
		for (int i = NumLayers-2; i >= 0; i--)
		{
			ArrayActs[i] = Array[i];
		}
	}
	//
	int getNumLayers()
	{
		return NumLayers;
	}
	void getArrayNumNodes(int * arr)
	{
		for (int i = 0; i < NumLayers; i++)
		{
			arr[i] = ArrayNumNodes[i];
		}
	}
	void getArrayActs(int * arr)
	{
		for (int i = NumLayers-2; i >= 0; i--)
		{
			arr[i] = ArrayActs[i];
		}
	}

	//
	void display()
	{
		//printf("\n");
		printf("NumLayers: %d\n", NumLayers);
		printf("NumNodes: ");
		for (int i = 0; i < NumLayers; i++)
		{
			printf("%d, ", ArrayNumNodes[i]);
		}
		printf("\n");
		//
		int NumMat = NumLayers -1;
		//
		printf("ActType: ");
		for (int i = 0; i < NumMat; i++)
		{
			printf("%d, ", ArrayActs[i]);
		}
		printf("\n");
		//		
		for (int i = 0; i < NumMat; i++)
		{
			printf("Weight Mat after Layer: %d\n", i);
			Weights[i].display();

			printf("Shift Mat after Layer: %d\n", i);
			Shifts[i].display();			
		}
		//
		printf("FNN_End.\n");
		//

	}
	//
	int writeToFile(char * filepath)
	{
		// return 0 for written		

		//
		FILE * fid = fopen(filepath, "w");
		//
		fprintf(fid, "NumLayers: %d\n", NumLayers);
		fprintf(fid, "NumNodes: ");
		for (int i = 0; i < NumLayers; i++)
		{
			fprintf(fid, "%d,", ArrayNumNodes[i]);
		}
		fprintf(fid, "\n");
		//
		int NumMat = NumLayers -1;
		//
		fprintf(fid, "ActType: ");
		for (int i = 0; i < NumMat; i++)
		{
			fprintf(fid, "%d,", ArrayActs[i]);
		}
		fprintf(fid, "\n");
		//		
		for (int i = 0; i < NumMat; i++)
		{
			fprintf(fid, "Mat Weight after Layer: %d\n", i);
			Weights[i].writeToFile(fid);

			fprintf(fid, "Mat Shift after Layer: %d\n", i);
			Shifts[i].writeToFile(fid);			
		}
		//
		fprintf(fid, "FNN_End.");
		//

		//
		fclose(fid);

		return 0;

	}
	int loadFromFile(char * filepath)
	{
		//return 0 for loaded,
		//return -1 for error,

		//
		FILE * fid = fopen(filepath, "r");
		if (fid == NULL) return -1;

		//
		// 2048, 4096
		int LenBuff = 4096;

		char * buff = new char[LenBuff];		
		char * str_begin;
		int curr;

		//
		int num_temp;

		fgets(buff, LenBuff, fid);  // 第一行
		str_begin = buff + 10;
		sscanf(str_begin, "%d", &num_temp);

		//
		int * array_temp = new int[num_temp];
		int Posi;

		fgets(buff, LenBuff, fid);  // 第二行
		str_begin = buff + 9;
		curr = 10;
		Posi = 0;
		while (buff[curr] != '\n')
		{
			if (buff[curr] == ',')
			{
				buff[curr] = '\0';

				sscanf(str_begin, "%d", array_temp + Posi);

				//
				Posi++;

				//
				curr++;

				str_begin = buff + curr;
			}
			else
			{
				curr++;
			}
		}//while curr

		// 修改结构，以写入
		setStructureFNN(num_temp, array_temp);

		// 激发函数
		int * array_act = new int[num_temp];

		fgets(buff, LenBuff, fid);   // 第三行
		str_begin = buff + 8;
		curr = 9;
		Posi = 0;
		while (buff[curr] != '\n')
		{
			if (buff[curr] == ',')
			{
				buff[curr] = '\0';

				sscanf(str_begin, "%d", array_act + Posi);

				//
				Posi++;

				//
				curr++;

				str_begin = buff + curr;
			}
			else
			{
				curr++;
			}
		}//while curr
		//
		setActArray(array_act);
		//

		// 读取矩阵
		int NumM1 = num_temp - 1; 
		for (int mp = 0; mp < NumM1; mp++)
		{
			//
			fgets(buff, LenBuff, fid);
			Weights[mp].loadFromFile(fid, array_temp[mp]);

			//
			fgets(buff, LenBuff, fid);
			Shifts[mp].loadFromFile(fid, 1);
		}

		//
		delete [] array_temp;
		delete [] array_act;
		delete [] buff;

		//
		fclose(fid);

		return 0;

	}// load
	//

	//
	void randomize(int a, int b)
	{
		for (int i = NumLayers - 2; i >= 0; i--)
		{
			Weights[i].randomize(a, b);
			Shifts[i].randomize(a, b);
		}
	}
	//

};


// utility functions 
int FNN_Predict(FNN_Model & fnn, FloatMat & Samples, FloatMat & Results);
int FNN_Train(FNN_Model & fnn, FloatMat & Samples, FloatMat & Labels);
int FNN_Test(FNN_Model & fnn, FloatMat & Samples, FloatMat & Labels);
//

// training functions
int FunctionFNN_Train_GD(FNN_Model & fnn, FloatMat & Samples, FloatMat & Labels);
int FunctionFNN_Train_LM(FNN_Model & fnn, FloatMat & Samples, FloatMat & Labels);
//

//
// internal functions
FloatMat Internal_Activiation_FNN(FloatMat mat, FloatMat shift, int act);
FloatMat Internal_ActDerivative_FNN(FloatMat mat, FloatMat shift, int act);
//
int Internal_DivideSamples_FNN(FNN_Model & fnn, FloatMat & Samples, FloatMat & Labels);
int Internal_MatErrBalance_FNN(FNN_Model & fnn, FloatMat & MatErrBalance);
int Internal_ValidationCheck_FNN(FNN_Model & fnn);
//


#endif

