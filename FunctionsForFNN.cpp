
//
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//
#include "FloatMat.h"
#include "FNN_Model.h"


//
char LogFileNameFNN[256];
//
void createLogTrainingFNN()
{
	FILE * fp = fopen(LogFileNameFNN, "w");
	fclose(fp);
}
void printStrToLogTrainingFNN(char * str)
{
	FILE * fp = fopen(LogFileNameFNN,"a");
	fprintf(fp,"%s",str);
	fclose(fp);
}
//

// utility functions
int FNN_Predict(FNN_Model & fnn, FloatMat & Samples, FloatMat & Results)
{
	int NumLayers = fnn.getNumLayers();
	//
	int * ArrActs = new int[NumLayers];
	fnn.getArrayActs(ArrActs);
	//
	int NumMat = NumLayers - 1;
	//
	Results = Samples;
	//
	for (int i = 0; i < NumMat; i++)
	{
		Results = Internal_Activiation_FNN(Results * fnn.Weights[i], fnn.Shifts[i], ArrActs[i]);
	}
	//
	//Result.normalizeRows();

	//
	delete [] ArrActs;

	return 1;
}
//
int FNN_Train(FNN_Model & fnn, FloatMat & Samples, FloatMat & Labels)
{
	// return positive for trained as scheduled,
	// return negative for training error,
	// return 0 for not trained,

	if (fnn.FlagLearningMethod == fnn.LEARN_GD)
	{
		return FunctionFNN_Train_GD(fnn, Samples, Labels);
	}
	else if (fnn.FlagLearningMethod == fnn.LEARN_LM)
	{
		return FunctionFNN_Train_LM(fnn, Samples, Labels);
	}
	else
	{
		printf("fnn.FlagTrainingMethod %d NOT defined.\n", fnn.FlagLearningMethod);
		//
		return 0;
	}
}
//
int FNN_Test(FNN_Model & fnn, FloatMat & Samples, FloatMat & Labels)
{
	fnn.ValidationSamples.copyFrom(Samples);
	fnn.ValidationLabels.copyFrom(Labels);

	return Internal_ValidationCheck_FNN(fnn);
}
//

// training functions
int FunctionFNN_Train_GD(FNN_Model & fnn, FloatMat & Samples, FloatMat & Labels)
{
	// return positive for trained as scheduled,
	// return negative for training error,

	// gdd, gda, gdm, gdx,
	float alpha = fnn.alpha;
	float beta = fnn.beta;
	float delta = fnn.delta;
	//
	float lamda = fnn.lamda;   // momentum
	float lamda_m = 1 - lamda;
	//
	float alpha_threshold = fnn.alpha_threshold;
	//
	int MaxIter = fnn.MaxIter;
	float error_tol = fnn.error_tol;
	float gradient_tol = fnn.gradient_tol;
	//
	//
	int FlagErrBalance = fnn.FlagErrBalance;
	//int FlagLearningMethod = fnn.FlagLearningMethod;
	int FlagAlpha = fnn.FlagAlpha;
	int FlagMomentum = fnn.FlagMomentum;
	//
	//float learning_portion = fnn.learning_portion;  // not used here
	//int SeedLearning = fnn.SeedLearning;
	//

	// log
	char StrTemp[256];
	//
	sprintf(LogFileNameFNN, "LogTrainingGD_%d_%d_%d_%.4f_%.4f_%.6f_%.2f_%d_%d.txt",
			FlagErrBalance, FlagAlpha, FlagMomentum, alpha, beta, delta, lamda, MaxIter, fnn.SeedLearning);
	//
	createLogTrainingFNN();
	//

	//
	printf("TrainingStart.\n");
	//

	// 训练样本划分
	printf("Dividing Samples ...\n");
	//
	int iResult = Internal_DivideSamples_FNN(fnn, Samples, Labels);
	if (iResult < 0)
	{
		printf("Error: too few samples, or not fit for the structure of the model.\n");
		sprintf(StrTemp, "Error: too few samples, or not fit for the structure of the model.\n");
		printStrToLogTrainingFNN(StrTemp);

		return -1;
	}
	//
	while (iResult == 0)   //
	{
		printf("Divided not properly.\n");
		printf("Dividing Samples ...\n");
		//
		iResult = Internal_DivideSamples_FNN(fnn, Samples, Labels);
	}
	printf("Samples Divided.\n");
	//

	// 步长
	float alpha_t = alpha;
	//
	// 误差
	float err;
	float gradient_length;
	//
	float err_last = 100000;
	//

	// 误差平衡
	FloatMat MatErrBalance;
	//
	int NumSamples, NumTypes;
	fnn.LearningLabels.getMatSize(NumSamples, NumTypes);
	//
	if (FlagErrBalance == 1)
	{
		if (Internal_MatErrBalance_FNN(fnn, MatErrBalance) == 0)   //
		{
			printf("Error: MatErrBalance failed to be generated.\n");
			sprintf(StrTemp, "Error: MatErrBalance failed to be generated.\n");
			printStrToLogTrainingFNN(StrTemp);

			return -3;
		}
	}
	else if (FlagErrBalance != 0)
	{
		printf("Error: FlagErrBalance != 0 or 1.\n");
		sprintf(StrTemp, "Error: FlagErrBalance != 0 or 1.\n");
		printStrToLogTrainingFNN(StrTemp);

		return -2;

	}// if FlagErrBalance


	// 网络结构
	int NumLayers = fnn.getNumLayers();
	int * ArrayNumNodes = new int[NumLayers];
	fnn.getArrayNumNodes(ArrayNumNodes);
	//
	int * ArrayActs = new int[NumLayers];
	fnn.getArrayActs(ArrayActs);
	//
	int NumMat = NumLayers - 1;
	int NumM2 = NumMat - 1;
	//

	// 中间结果，求导时要用，
	FloatMat * ActDerivative = new FloatMat[NumMat];
	FloatMat * Results = new FloatMat[NumLayers];
	//
	Results[0].copyFrom(fnn.LearningSamples);

	// 误差与回溯的矩阵，
	FloatMat ErrTrack, TrackAct;

	// 导数
	FloatMat * GradientW = new FloatMat[NumMat];
	FloatMat * GradientS = new FloatMat[NumMat];
	//
	for (int layer = 0; layer < NumMat; layer++)
	{
		GradientW[layer].setMatSize(ArrayNumNodes[layer], ArrayNumNodes[layer+1]);
		GradientS[layer].setMatSize(1, ArrayNumNodes[layer+1]);
	}

	// 惯性
	FloatMat * deltaW;
	FloatMat * deltaS;
	float * ptr_momentum_swap;
	//
	if (FlagMomentum == fnn.MOMENTUM_NONE)
	{
		deltaW = new FloatMat[1];
		deltaS = new FloatMat[1];
	}
	else
	{
		deltaW = new FloatMat[NumMat];
		deltaS = new FloatMat[NumMat];
		//
		for (int layer = 0; layer < NumMat; layer++)
		{
			deltaW[layer].setMatSize(ArrayNumNodes[layer], ArrayNumNodes[layer+1]);
			deltaS[layer].setMatSize(1, ArrayNumNodes[layer+1]);

			deltaW[layer].setMatConstant(0);
			deltaS[layer].setMatConstant(0);
		}
	}

	//
	int iRet = 1;     // MaxIter reached
	//

	// 循环优化
	int iter = 0;
	while (iter < MaxIter)
	{
		//
		printf("iter, %d, ", iter);
		sprintf(StrTemp, "iter, %d, ", iter);
		printStrToLogTrainingFNN(StrTemp);
		//

		// 前向计算
		for (int layer = 0; layer < NumMat; layer++)
		{
			Results[layer+1] = Results[layer] * fnn.Weights[layer];
			//
			ActDerivative[layer] = Internal_ActDerivative_FNN(Results[layer+1], fnn.Shifts[layer], ArrayActs[layer]);
			//
			Results[layer+1] = Internal_Activiation_FNN(Results[layer+1], fnn.Shifts[layer], ArrayActs[layer]);
		}

		// 误差计算
		ErrTrack = Results[NumMat] - fnn.LearningLabels;
		//
		if (FlagErrBalance == 1)
		{
			ErrTrack = ErrTrack.mul(MatErrBalance);
		}
		//
		err = ErrTrack.mul(ErrTrack).meanElementsAll();
		err = sqrt(err);
		//
		printf("err, %.4f, ", err);
		sprintf(StrTemp, "err, %.4f, ", err);
		printStrToLogTrainingFNN(StrTemp);
		//
		if (err < error_tol)
		{
			printf("err < error_tol = %f\n", error_tol);

			iRet = 2;  // error_tol reached
			break;
		}
		//

		// validation check
		Internal_ValidationCheck_FNN(fnn);
		//
		printf("prc, %.4f, ", fnn.performance[0]);
		sprintf(StrTemp, "prc, %.4f, ", fnn.performance[0]);
		printStrToLogTrainingFNN(StrTemp);
		//
		printf("rec, %.4f, ", fnn.performance[1]);
		sprintf(StrTemp, "rec, %.4f, ", fnn.performance[1]);
		printStrToLogTrainingFNN(StrTemp);
		//

		// 计算导数
		for (int layer = NumM2; layer >= 0; layer--)
		{
			// TrackAct
			TrackAct = ErrTrack.mul(ActDerivative[layer]);

			// 计算ErrTrack前一层
			ErrTrack = TrackAct * fnn.Weights[layer].transpose();

			// 计算Weights导数
			GradientW[layer] = Results[layer].transpose() * TrackAct; // + fnn.Weights[layer].getSigns() * epsilon;

			// 计算Shifts导数
			GradientS[layer] = TrackAct.sumCols(); // + fnn.Shifts[layer].getSigns() * epsilon;

		}//for layer

		// 计算梯度长度
		gradient_length = 0;
		for (int layer = NumM2; layer >=0; layer--)
		{
			gradient_length += (GradientW[layer].mul(GradientW[layer])).sumElementsAll();  // 可优化
			gradient_length += (GradientS[layer].mul(GradientS[layer])).sumElementsAll();
		}
		gradient_length = sqrt(gradient_length);
		//
		printf("glength, %.4f, ", gradient_length);
		sprintf(StrTemp, "glength, %.4f, ", gradient_length);
		printStrToLogTrainingFNN(StrTemp);
		//
		if (gradient_length < gradient_tol)
		{
			printf("gradient_length < gradient_tol = %f\n", gradient_tol);

			iRet = 3; // gradient_tol reached
			break;
		}
		//

		// 梯度下降
		//
		if (alpha_t > alpha_threshold)
		{
			if (FlagAlpha == fnn.ALPHA_DES)    // 步长，下降
			{
				alpha_t *= beta;
			}
			else if (FlagAlpha == fnn.ALPHA_ADA)   // 步长，自适应
			{
				if (err < err_last) alpha_t += delta;
				else alpha_t *= beta;
				//
				err_last = err;
			}
		}
		//else alpha_t = alpha_threshold;
		//
		if (FlagMomentum == fnn.MOMENTUM_EXP)    // 动量，指数平滑，
		{
			for (int layer = NumM2; layer >= 0; layer--)
			{
				deltaW[layer] = deltaW[layer] * lamda - GradientW[layer] * alpha_t;
				deltaS[layer] = deltaS[layer] * lamda - GradientS[layer] * alpha_t;
				//
				fnn.Weights[layer] = fnn.Weights[layer] + deltaW[layer];
				fnn.Shifts[layer] = fnn.Shifts[layer] + deltaS[layer];
			}
		}
		else if (FlagMomentum == fnn.MOMENTUM_PREV)    // 动量，两步合力，
		{
			for (int layer = NumM2; layer >= 0; layer--)
			{
				fnn.Weights[layer] = fnn.Weights[layer] - (GradientW[layer] * (lamda * alpha_t) + deltaW[layer] * (lamda_m * alpha_t));
				fnn.Shifts[layer] = fnn.Shifts[layer] - (GradientS[layer] * (lamda * alpha_t) + deltaS[layer] * (lamda_m * alpha_t));

				//
				ptr_momentum_swap = deltaW[layer].data;
				deltaW[layer].data = GradientW[layer].data;
				GradientW[layer].data = ptr_momentum_swap;
				//
				ptr_momentum_swap = deltaS[layer].data;
				deltaS[layer].data = GradientS[layer].data;
				GradientS[layer].data = ptr_momentum_swap;
				//
			}
		}
		else   // 无动量
		{
			for (int layer = NumM2; layer >= 0; layer--)
			{
				fnn.Weights[layer] = fnn.Weights[layer] - GradientW[layer] * alpha_t;
				fnn.Shifts[layer] = fnn.Shifts[layer] - GradientS[layer] * alpha_t;
			}
		}

		//

		//
		printf("alpha_t, %.6f\n", alpha_t);
		sprintf(StrTemp, "alpha_t, %.6f\n", alpha_t);
		printStrToLogTrainingFNN(StrTemp);
		//

		//
		iter++;

	}// while iter
	//
	if (iter >= MaxIter)
	{
		printf("iter >= MaxIter = %d\n", MaxIter);
	}
	//

	//
	delete [] ArrayNumNodes;
	delete [] ArrayActs;
	delete [] ActDerivative;
	delete [] Results;
	//
	delete [] GradientW;
	delete [] GradientS;
	delete [] deltaW;
	delete [] deltaS;
	//

	return iRet;
}
//

// problematic
int FunctionFNN_Train_LM(FNN_Model & fnn, FloatMat & Samples, FloatMat & Labels)
{
	// lm
	float alpha = fnn.alpha;
	float beta = fnn.beta;
	//
	int MaxIter = fnn.MaxIter;
	float error_tol = fnn.error_tol;
	//float gradient_tol = fnn.gradient_tol;
	//

	// 步长
	float alpha_t = alpha;
	//

	// 误差
	float err;
	float err_last = 10000;
	//

	// 记录
	char StrTemp[256];
	//
	sprintf(LogFileNameFNN, "LogTraining_%d_%.6f_%.6f.txt", MaxIter, alpha, beta);
	//
	createLogTrainingFNN();
	//

	// 网络结构
	int NumLayers = fnn.getNumLayers();
	int * ArrayNumNodes = new int[NumLayers];
	fnn.getArrayNumNodes(ArrayNumNodes);
	//
	int * ArrayActs = new int[NumLayers];
	fnn.getArrayActs(ArrayActs);
	//

	//
	int NumMat = NumLayers - 1;
	int NumM2 = NumMat - 1;
	//
	// 中间结果，求导时要用，
	FloatMat * ActDerivative = new FloatMat[NumMat];
	FloatMat * Results = new FloatMat[NumLayers];
	//
	Results[0].copyFrom(Samples);
	//

	//
	int NumSamples, NumTypes, NumVariables;
	Labels.getMatSize(NumSamples, NumTypes);
	//
	NumVariables = 0;
	for (int layer = 0; layer < NumMat; layer++)
	{
		NumVariables += (ArrayNumNodes[layer] + 1) * ArrayNumNodes[layer+1];
	}
	//
	// 误差与回溯的矩阵
	FloatMat Jacobian(NumSamples, NumVariables);
	FloatMat JacobianT; // transpose
	//
	int PosiVariable, PosiShift, PosiWeight;
	int NumShift, NumPrev, NumWeight;
	//
	FloatMat Hessian, ErrMat;
	FloatMat ErrTrack, TrackAct;
	//
	// 修正量矩阵
	FloatMat deltaMat, tempMat, tempResult;
	//
	// 存储的权重和阈值
	FloatMat * StoredW = new FloatMat[NumMat];
	FloatMat * StoredS = new FloatMat[NumMat];
	//
	for (int layer = 0; layer < NumMat; layer++)
	{
		StoredW[layer].setMatSize(ArrayNumNodes[layer], ArrayNumNodes[layer+1]);
		StoredS[layer].setMatSize(1, ArrayNumNodes[layer+1]);
	}
	//

	// 循环优化
	int iter = 0;
	while (iter < MaxIter)
	{
		//
		printf("\niter: %d, ", iter);
		sprintf(StrTemp, "\niter: %d, ", iter);
		printStrToLogTrainingFNN(StrTemp);
		//

		// 前向计算
		for (int layer = 0; layer < NumMat; layer++)
		{
			Results[layer+1] = Results[layer] * fnn.Weights[layer];
			//
			ActDerivative[layer] = Internal_ActDerivative_FNN(Results[layer+1], fnn.Shifts[layer], ArrayActs[layer]);
			//
			Results[layer+1] = Internal_Activiation_FNN(Results[layer+1], fnn.Shifts[layer], ArrayActs[layer]);
		}	
		//

		// 误差计算
		ErrTrack = Results[NumMat] - Labels;

		//
		printf("\n");
		printf("Results[NumMat]:\n");
		printf("\n");
		Results[NumMat].display();
		//
		getchar();
		//

		//
		printf("\n");
		printf("Label:\n");
		printf("\n");
		Labels.display();
		//
		getchar();
		//

		//
		printf("\n");
		printf("ErrTrack:\n");
		printf("\n");
		ErrTrack.display();
		//
		getchar();
		//


		//
		// err
		ErrMat = ErrTrack.mul(ErrTrack);
		err = sqrt(ErrMat.meanElementsAll());
		//
		printf("err: %f, ", err);
		sprintf(StrTemp, "err: %f, ", err);
		printStrToLogTrainingFNN(StrTemp);
		//		
		if (err < error_tol)
		{
			printf("err < error_tol = %f\n", error_tol);
			break;
		}
		// ErrMat
		ErrMat = ErrMat.sumRows() * (1.0 / NumTypes);
		//

		//
		if (err > err_last)
		{
			alpha_t = alpha_t / beta;

			//
			printf("err > err_last, ");
			sprintf(StrTemp, "err > err_last, ");
			printStrToLogTrainingFNN(StrTemp);
			//
		}
		else
		{
			//
			printf("err <= err_last, ");
			sprintf(StrTemp, "err <= err_last, ");
			printStrToLogTrainingFNN(StrTemp);
			//

			// 递进
			iter++;
			err_last = err;
			//
			//alpha_t = alpha_t * beta;
			//if (alpha_t > 1000000) alpha_t = alpha;
			//
			alpha_t = alpha;
			//

			// 存储自变量
			for (int layer = NumM2; layer >= 0; layer--)
			{
				//Weights
				StoredW[layer] = fnn.Weights[layer];

				//Shifts
				StoredS[layer] = fnn.Shifts[layer];

			}//for layer

			//
			// Jacobian
			PosiVariable = NumVariables;
			//
			for (int layer = NumM2; layer >= 0; layer--)
			{
				//
				printf("layer: %d, ", layer);
				sprintf(StrTemp, "layer: %d, ", layer);
				printStrToLogTrainingFNN(StrTemp);
				//

				// TrackAct 
				TrackAct = ErrTrack.mul(ActDerivative[layer]);

				//
				//printf("\n");
				//printf("Shift:\n");
				//printf("\n");
				//TrackAct.display();
				//
				//getchar();
				//

				// 计算ErrTrack前一层
				ErrTrack = TrackAct * StoredW[layer].transpose();

				// 
				// Jacobian
				NumShift = ArrayNumNodes[layer+1];
				NumPrev = ArrayNumNodes[layer];
				NumWeight = NumPrev * NumShift;
				//	
				// a = 0
				PosiShift = PosiVariable - NumShift;
				PosiWeight = PosiShift - NumWeight;
				//
				for (int a = 0; a < NumSamples; a++)
				{
					// tempMat
					tempMat.setMatSize(1, NumShift);
					memcpy(tempMat.data, TrackAct.data + a*NumShift, sizeof(float) * NumShift);
					//

					// Shifts
					memcpy(Jacobian.data + PosiShift, tempMat.data, sizeof(float) * NumShift);
					//

					// tempResult
					tempResult.setMatSize(1, NumPrev);
					memcpy(tempResult.data, Results[layer].data + a*NumPrev, sizeof(float) * NumPrev);
					//

					//Weights
					tempMat = tempResult.transpose() * tempMat;
					//
					memcpy(Jacobian.data + PosiWeight, tempMat.data, sizeof(float) * NumWeight);
					//

					//
					PosiShift += NumVariables;
					PosiWeight += NumVariables;
					//

				}// for a

				//
				//getchar();
				//

				//
				PosiVariable -= (NumShift + NumWeight);
				//

			}//for layer

			//
			printf("\njacobian ended, ");
			sprintf(StrTemp, "\njacobian ended, ");
			printStrToLogTrainingFNN(StrTemp);
			//

			//
			//printf("\n");
			//Jacobian.display();
			//
			//getchar();
			//


			//Hessian
			JacobianT = Jacobian.transpose();
			//
			Hessian = JacobianT * Jacobian;
			//

		}//if err_last

		//
		printf("hessian ended, ");
		sprintf(StrTemp, "hessian ended, ");
		printStrToLogTrainingFNN(StrTemp);
		//
		printf("alpha_t: %f, ", alpha_t);
		sprintf(StrTemp, "alpha_t: %f, ", alpha_t);
		printStrToLogTrainingFNN(StrTemp);
		//
		//printf("beta: %f, ", beta);
		//sprintf(StrTemp, "beta: %f, ", beta);
		//printStrToLogTrainingFNN(StrTemp);
		//

		//
		//printf("\n");
		//Hessian.display();
		//
		//getchar();
		//


		// deltaMat
		int iSolve = (JacobianT * ErrMat * (-1)).solveWithSymMatX(Hessian.plusWeightedIdentity(alpha_t), deltaMat);
		//

		//
		printf("\n");
		deltaMat.display();
		//
		printf("\nalpha_t: %f\n", alpha_t);
		//
		getchar();
		//

		//
		printf("iSolve: %d, ", iSolve);
		sprintf(StrTemp, "iSolve: %d, ", iSolve);
		printStrToLogTrainingFNN(StrTemp);
		//
		//
		//printf("delta_mat ended, ");
		//sprintf(StrTemp, "delta_mat ended, ");
		//printStrToLogTrainingFNN(StrTemp);
		//

		// 修改权重和阈值
		PosiVariable = 0;
		for (int layer = 0; layer < NumMat; layer++)
		{
			//Weights
			tempMat.setMatSize(ArrayNumNodes[layer], ArrayNumNodes[layer+1]);
			//
			memcpy(tempMat.data, deltaMat.data + PosiVariable, sizeof(float) * tempMat.getNumTotal());
			//
			fnn.Weights[layer] = StoredW[layer] + tempMat;
			//
			PosiVariable += tempMat.getNumTotal();
			//

			//Shifts
			tempMat.setMatSize(1, ArrayNumNodes[layer+1]);
			//
			memcpy(tempMat.data, deltaMat.data + PosiVariable, sizeof(float) * tempMat.getNumTotal());
			//
			fnn.Shifts[layer] = StoredS[layer] + tempMat;
			//
			PosiVariable += tempMat.getNumTotal();
			//	

		}//for layer

		//
		printf("variables renewed, ");
		sprintf(StrTemp, "variables renewed, ");
		printStrToLogTrainingFNN(StrTemp);
		//

	}// while iter
	//
	if (iter >= MaxIter) printf("iter >= MaxIter = %d\n", MaxIter);
	//

	//
	delete [] ArrayNumNodes;
	delete [] ArrayActs;
	//
	delete [] ActDerivative;
	delete [] Results;
	//
	delete [] StoredW;
	delete [] StoredS;	
	//

	return 0;
}
//

// internal functions
FloatMat Internal_Activiation_FNN(FloatMat mat, FloatMat shift, int act)
{
	int NumRows, NumCols;
	mat.getMatSize(NumRows, NumCols);

	FloatMat answ(NumRows, NumCols);	

	float * answ_data_p = answ.data;
	float * mat_data_p = mat.data;
	float * shift_data_p = shift.data;

	//
	FNN_Model fnn;

	//
	if (act == fnn.ACT_LOGS)
	{
		float temp;

		int Posi = 0;
		for (int i = 0; i < NumRows; i++)
		{
			for (int j = 0; j < NumCols; j++)
			{
				temp = mat_data_p[Posi] + shift_data_p[j];

				// activation function				
				//
				answ_data_p[Posi] = 1 /(1 + exp(-temp));
				//

				//
				Posi++;
			}
		}
	}
	else if (act == fnn.ACT_LOGB)
	{
		float temp;

		int Posi = 0;
		for (int i = 0; i < NumRows; i++)
		{
			for (int j = 0; j < NumCols; j++)
			{
				temp = mat_data_p[Posi] + shift_data_p[j];

				// activation function				
				//
				answ_data_p[Posi] = 2 /(1 + exp(-temp)) - 1;
				//

				//
				Posi++;
			}
		}
	}
	else if (act == fnn.ACT_RELB)
	{
		float temp;

		int Posi = 0;
		for (int i = 0; i < NumRows; i++)
		{
			for (int j = 0; j < NumCols; j++)
			{
				temp = mat_data_p[Posi] + shift_data_p[j];

				// activation function				
				//
				if (temp < -1) answ_data_p[Posi] = -1;
				else if (temp > 1) answ_data_p[Posi] = 1;
				else answ_data_p[Posi] = temp;
				//

				//
				Posi++;
			}
		}

	}
	else // FNN_Model.ACT_RELU
	{
		float temp;

		int Posi = 0;
		for (int i = 0; i < NumRows; i++)
		{
			for (int j = 0; j < NumCols; j++)
			{
				temp = mat_data_p[Posi] + shift_data_p[j];

				// activation function				
				//
				if (temp < 0) answ_data_p[Posi] = 0;
				else answ_data_p[Posi] = temp;
				//

				//
				Posi++;
			}
		}

	}// if act

	return answ;
}
//
FloatMat Internal_ActDerivative_FNN(FloatMat mat, FloatMat shift, int act)
{
	int NumRows, NumCols;
	mat.getMatSize(NumRows, NumCols);

	FloatMat answ(NumRows, NumCols);	

	float * answ_data_p = answ.data;
	float * mat_data_p = mat.data;
	float * shift_data_p = shift.data;

	//
	FNN_Model fnn;

	//
	if (act == fnn.ACT_LOGS)
	{
		float temp;

		int Posi = 0;
		for (int i = 0; i < NumRows; i++)
		{
			for (int j = 0; j < NumCols; j++)
			{
				temp = mat_data_p[Posi] + shift_data_p[j];

				// activation function	
				// answ_data_p[Posi] = 1 /(1 + exp(-temp));
				//
				temp = exp(-temp);
				answ_data_p[Posi] = temp /(1 + temp)/(1 + temp);
				//

				//
				Posi++;
			}
		}
	}
	else if (act == fnn.ACT_LOGB)
	{
		float temp;

		int Posi = 0;
		for (int i = 0; i < NumRows; i++)
		{
			for (int j = 0; j < NumCols; j++)
			{
				temp = mat_data_p[Posi] + shift_data_p[j];

				// activation function				
				// answ_data_p[Posi] = 2 /(1 + exp(-temp)) - 1;
				//
				temp = exp(-temp);
				answ_data_p[Posi] = 2 * temp /(1 + temp)/(1 + temp);
				//

				//
				Posi++;
			}
		}
	}
	else if (act == fnn.ACT_RELB)
	{
		float temp;

		int Posi = 0;
		for (int i = 0; i < NumRows; i++)
		{
			for (int j = 0; j < NumCols; j++)
			{
				temp = mat_data_p[Posi] + shift_data_p[j];

				// activation function				
				//
				if (temp < -1 || temp > 1) answ_data_p[Posi] = 0;
				else answ_data_p[Posi] = 1;
				//

				//
				Posi++;
			}
		}

	}
	else // FNN_Model.ACT_RELU
	{
		float temp;

		int Posi = 0;
		for (int i = 0; i < NumRows; i++)
		{
			for (int j = 0; j < NumCols; j++)
			{
				temp = mat_data_p[Posi] + shift_data_p[j];

				// activation function derivative			
				//
				if (temp < 0) answ_data_p[Posi] = 0;
				else answ_data_p[Posi] = 1;
				//

				//
				Posi++;
			}
		}

	}// if act

	return answ;
}
//

//
int Internal_DivideSamples_FNN(FNN_Model & fnn, FloatMat & Samples, FloatMat & Labels)
{
	// return negative for error
	// return 0 for divided not properly
	// return 1 for divided

	//
	int NumSamples, NumFeatures;
	int NumSamplesL, NumTypes;
	Samples.getMatSize(NumSamples, NumFeatures);
	Labels.getMatSize(NumSamplesL, NumTypes);
	//
	if (NumSamples < 10)   //
	{
		printf("Error: too few Samples.\n");

		return -1;
	}
	//
	if (NumSamples != NumSamplesL)
	{
		printf("Error: Samples and Labels do NOT have same number of rows.\n");

		return -2;
	}
	//
	int NumLayers = fnn.getNumLayers();
	int * ArrayNumNodes = new int[NumLayers];
	fnn.getArrayNumNodes(ArrayNumNodes);
	//
	if (NumFeatures != ArrayNumNodes[0])
	{
		printf("Error: Samples and Structure of the model do NOT match.\n");

		return -3;
	}
	//
	if (NumTypes != ArrayNumNodes[NumLayers-1])
	{
		printf("Error: Labels and Structure of the model do NOT match.\n");

		return -4;
	}

	// 随机选择
	int * FlagForLearning = new int[NumSamples];
	int NumLearning = 0;
	int NumValidation = 0;
	//
	srand(fnn.SeedLearning);
	int ThrRand = fnn.LearningPortion * 1000;  //
	//
	while (NumLearning == 0 || NumValidation == 0)
	{
		NumLearning = 0;
		NumValidation = 0;
		//
		for (int s = 0; s < NumSamples; s++)
		{
			if (rand()%1000 < ThrRand)  //
			{
				FlagForLearning[s] = 1;
				NumLearning++;
			}
			else
			{
				FlagForLearning[s] = 0;
				NumValidation++;
			}
		}// for s
	}// while 0

	// 复制
	fnn.LearningSamples.setMatSize(NumLearning, NumFeatures);
	fnn.LearningLabels.setMatSize(NumLearning, NumTypes);
	//
	fnn.ValidationSamples.setMatSize(NumValidation, NumFeatures);
	fnn.ValidationLabels.setMatSize(NumValidation, NumTypes);
	//
	float * data_learning_samples = fnn.LearningSamples.data;
	float * data_learning_labels = fnn.LearningLabels.data;
	float * data_validation_samples = fnn.ValidationSamples.data;
	float * data_validation_labels = fnn.ValidationLabels.data;
	//
	float * data_samples = Samples.data;
	float * data_labels = Labels.data;
	//
	int LenDatumSample = sizeof(float) * NumFeatures;
	int LenDatumLabel = sizeof(float) * NumTypes;
	//
	for (int s = 0; s < NumSamples; s++)
	{
		if (FlagForLearning[s] == 1)
		{
			memcpy(data_learning_samples, data_samples, LenDatumSample);
			data_learning_samples += NumFeatures;
			data_samples += NumFeatures;

			memcpy(data_learning_labels, data_labels, LenDatumLabel);
			data_learning_labels += NumTypes;
			data_labels += NumTypes;
		}
		else
		{
			memcpy(data_validation_samples, data_samples, LenDatumSample);
			data_validation_samples += NumFeatures;
			data_samples += NumFeatures;

			memcpy(data_validation_labels, data_labels, LenDatumLabel);
			data_validation_labels += NumTypes;
			data_labels += NumTypes;
		}
	}// for s

	//
	delete [] FlagForLearning;

	// 检查
	FloatMat MatCount;
	MatCount = fnn.ValidationLabels.sumCols();
	for (int t = 0; t < NumTypes; t++)
	{
		if (MatCount.data[t] == 0) return 0;
	}
	//
	MatCount = fnn.LearningLabels.sumCols();
	for (int t = 0; t < NumTypes; t++)
	{
		if (MatCount.data[t] == 0) return 0;
	}
	//

	return 1;
}
//
int Internal_MatErrBalance_FNN(FNN_Model & fnn, FloatMat & MatErrBalance)
{
	//0 for error, 1 for conducted,

	//
	FloatMat MatCount;
	MatCount = fnn.LearningLabels.sumCols();
	//

	//
	int NumSamples, NumTypes;
	fnn.LearningLabels.getMatSize(NumSamples, NumTypes);
	//
	float ratio = 1.0 * NumSamples/NumTypes;
	//
	float * data_count = MatCount.data;
	for (int i = 0; i < NumTypes; i++)
	{
		if (data_count[i] == 0)
		{
			printf("data_count[i] == 0 when generating MatErrBalance_FNN.\n");

			return 0;
		}
		//
		data_count[i] = ratio/data_count[i];
	}
	//

	//
	MatErrBalance.setMatSize(NumSamples, NumTypes);
	//
	int type = 0;
	float * data_label = fnn.LearningLabels.data;
	float * data_errbalance = MatErrBalance.data;
	//
	for (int s = 0, posi_start = 0; s < NumSamples; s++, posi_start += NumTypes)
	{
		// type
		for (int t = 0, posi = posi_start; t < NumTypes; t++, posi++)
		{
			if (data_label[posi] == 1)
			{
				type = t;
				break; // for t
			}
		}// for t

		// ratio
		ratio = data_count[type];

		// assign
		for (int t = 0, posi = posi_start; t < NumTypes; t++, posi++)
		{
			data_errbalance[posi] = ratio;
		}// for t

	}// for s

	//
	return 1;
}
//
int Internal_ValidationCheck_FNN(FNN_Model & fnn)
{
	FloatMat Results, ResultsNormalized;
	//
	FNN_Predict(fnn, fnn.ValidationSamples, Results);
	//
	ResultsNormalized.copyFrom(Results);
	ResultsNormalized.normalizeRows();
	//

	//
	int NumSamples, NumTypes;
	fnn.ValidationLabels.getMatSize(NumSamples, NumTypes);

	//
	float CriteriaAssertion = fnn.CriteriaAssertion;
	float CriteriaBasic = 1.0/NumTypes;
	//

	//
	int PositiveTotal = 0;
	int PredictedPositive = 0;
	int TruePredictedPositive = 0;
	//
	float * data_labels = fnn.ValidationLabels.data;
	float * data_results = Results.data;
	float * data_normalized = ResultsNormalized.data;
	//
	for (int s = 0; s < NumSamples; s++)
	{
		if (data_results[0] >= CriteriaBasic && data_normalized[0] >= CriteriaAssertion)
		{
			PredictedPositive++;

			if (data_labels[0] >= CriteriaAssertion)
			{
				TruePredictedPositive++;
			}
		}
		//
		if (data_labels[0] >= CriteriaAssertion)
		{
			PositiveTotal++;
		}

		//
		data_labels += NumTypes;
		data_results += NumTypes;
		data_normalized += NumTypes;

	}// for s

	//
	if (PredictedPositive > 0) fnn.performance[0] = 1.0 * TruePredictedPositive/PredictedPositive;
	else fnn.performance[0] = 0;
	//
	fnn.performance[1] = 1.0 * TruePredictedPositive/PositiveTotal;


	return 1;
}
//


