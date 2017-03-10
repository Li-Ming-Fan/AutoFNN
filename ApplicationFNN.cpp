
//
#include <direct.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

//
#include "FloatMat.h"
#include "FNN_Model.h"
//

//
void loadConfiguration();
//
int FlagLoadFromFile;
int FlagTraining;
int FlagFiles;
//
int FlagBackupNetFile;
//
int SeedLearning;
float Criteria;    // 0.95, 0.85, 0
//
int ErrBalance;
int LearningMethod;
//
float alpha, beta, delta, lamda;
//
int SeedForRandom;
int NumMaxIter;
//
int NumLayers;
int * ArrayNumNodes;
int * ArrayActs;
//

//
int main()
{  
	//printf("\n");
	printf("ApplicationFNN begin ...\n\n");
	//
	// direct
	mkdir("AutoFNN_working_direct");
	chdir("AutoFNN_working_direct");	
	//
	char WORK_DIRECT[128];
	getcwd(WORK_DIRECT, sizeof(WORK_DIRECT)); 
	//
	// configuration
	loadConfiguration();
	//
	printf("Configuration loaded.\n\n");
	//
	printf("FlagLoadFromFile: %d\n", FlagLoadFromFile);
	printf("FlagTraining: %d\n", FlagTraining);
	printf("FlagFiles: %d\n", FlagFiles);
	printf("FlagBackupNetFile: %d\n", FlagBackupNetFile);
	printf("\n");
	//
	printf("SeedLearning: %d\n", SeedLearning);
	printf("Criteria: %.2f\n", Criteria);
	printf("\n");
	//
	printf("ErrBalance: %d\n", ErrBalance);
	printf("LearningMethod: %d\n", LearningMethod);
	printf("alpha, beta, delta, lamda: %.6f, %.6f, %.6f, %.6f,\n", alpha, beta, delta, lamda);
	//printf("\n");
	//
	printf("SeedForRandom: %d\n", SeedForRandom);
	printf("MaxIter: %d\n", NumMaxIter);
	printf("\n");
	//
	printf("NumLayers: %d\n", NumLayers);
	printf("NumNodes: ");
	for (int i = 0; i < NumLayers; i++) printf("%d, ", ArrayNumNodes[i]);
	printf("\n");
	//
	printf("ActType: ");
	for (int i = 0; i < NumLayers - 1; i++) printf("%d, ", ArrayActs[i]);
	printf("\n");
	//
	printf("\n");
	//

	//
	getchar();
	//

	//
	// Structure
	//int NumLayers = 3;
	//int ArrayNumNodes[3] = {90, 60, 2};
	//
	// model
	FNN_Model fnn;
	fnn.setStructureFNN(NumLayers, ArrayNumNodes);
	//
	fnn.setActArray(ArrayActs);
	//
	//fnn.setActSingleLayer(0, fnn.ACT_LOGB);
	//fnn.setActSingleLayer(0, fnn.ACT_RELB);
	//fnn.setActSingleLayer(1, fnn.ACT_LOGS);
	//fnn.setActSingleLayer(1, fnn.ACT_RELU);
	//
	srand(SeedForRandom);
	fnn.randomize(-1, 1);
	//

	//
	char TrainingSamples_Filename[32];
	char TrainingLabels_Filename[32];
	char TestSamples_Filename[32];
	char TestLabels_Filename[32];
	//
	char FNN_Filename[32];
	//
	if (FlagFiles == 0)
	{
		strcpy(TrainingSamples_Filename, "TrainingSamples.txt");
		strcpy(TrainingLabels_Filename, "TrainingLabels.txt");
		strcpy(TestSamples_Filename, "TestSamples.txt");
		strcpy(TestLabels_Filename, "TestLabels.txt");
		//
		strcpy(FNN_Filename, "FNN_File.txt");
	}
	else if (FlagFiles == 1)
	{
		strcpy(TrainingSamples_Filename, "TrainingSamples_Ascend.txt");
		strcpy(TrainingLabels_Filename, "TrainingLabels_Ascend.txt");
		strcpy(TestSamples_Filename, "TestSamples_Ascend.txt");
		strcpy(TestLabels_Filename, "TestLabels_Ascend.txt");
		//
		strcpy(FNN_Filename, "FNN_File_Ascend.txt");
	}
	else if (FlagFiles == -1)
	{
		strcpy(TrainingSamples_Filename, "TrainingSamples_Descend.txt");
		strcpy(TrainingLabels_Filename, "TrainingLabels_Descend.txt");
		strcpy(TestSamples_Filename, "TestSamples_Descend.txt");
		strcpy(TestLabels_Filename, "TestLabels_Descend.txt");
		//
		strcpy(FNN_Filename, "FNN_File_Descend.txt");
	}
	else
	{
		strcpy(TrainingSamples_Filename, "TrainingSamples.txt");
		strcpy(TrainingLabels_Filename, "TrainingLabels.txt");
		strcpy(TestSamples_Filename, "TestSamples.txt");
		strcpy(TestLabels_Filename, "TestLabels.txt");
		//
		strcpy(FNN_Filename, "FNN_File.txt");
	}

	//
	// TrainingSamples
	FloatMat TrainingSamples(1, ArrayNumNodes[0]);
	//
	TrainingSamples.loadAllDataInFile(TrainingSamples_Filename);
	//
	printf("TrainingSamples loaded.\n");
	//
	//TrainingSamples.display();
	//
	int NumRows, NumCols;
	TrainingSamples.getMatSize(NumRows, NumCols);
	printf("TrainingSamples NumRows: %d\n", NumRows);
	//
	//getchar();
	//

	// TrainingLabels
	FloatMat TrainingLabels(1, ArrayNumNodes[NumLayers - 1]);
	//
	TrainingLabels.loadAllDataInFile(TrainingLabels_Filename);
	//
	printf("TrainingLabels loaded.\n");
	//
	//TrainingLabels.display();
	//
	TrainingLabels.getMatSize(NumRows, NumCols);
	printf("TrainingLabels NumRows: %d\n", NumRows);
	//
	getchar();	
	//

	// Load
	if (FlagLoadFromFile == 1)
	{
		// load
		int iLoad = fnn.loadFromFile(FNN_Filename);   //
		if (iLoad == 0)
		{
			//printf("\n");
			printf("Model Loaded from %s.\n", FNN_Filename);
			//fnn.display();
			//
		}
		else
		{
			//printf("\n");
			printf("Error when loading model from %s.\n", FNN_Filename);
		}
		//
		getchar();
		//
	}
	else
	{
		printf("FlagLoadFromFile == 0.\n");
		printf("\n");
	}

	// Training
	if (FlagTraining == 1)
	{
		// Training Process
		//printf("\n");
		printf("Training Process:\n");
		//
		fnn.SeedLearning = SeedLearning;
		fnn.CriteriaAssertion = Criteria;
		//
		fnn.FlagErrBalance = ErrBalance;
		fnn.FlagLearningMethod = LearningMethod;
		//
		fnn.MaxIter = NumMaxIter;
		//
		//fnn.setTrainingParasDefault();
		//
		fnn.alpha = alpha;
		fnn.beta = beta;
		fnn.delta = delta;
		fnn.lamda = lamda;
		//
		FNN_Train(fnn, TrainingSamples, TrainingLabels);
		//
		fnn.writeToFile(FNN_Filename);
		//

		//
		printf("\n");
		printf("Training Process Ended, Model saved.\n");
		//fnn.display();
		//

		//
		getchar();
		//
	}
	else
	{
		printf("FlagTraining == 0.\n\n");

		//
		// TestSamples
		FloatMat TestSamples(1, ArrayNumNodes[0]);
		//
		TestSamples.loadAllDataInFile(TestSamples_Filename);
		//
		printf("TestSamples loaded.\n");
		//
		//TestSamples.display();
		//
		int NumRows, NumCols;
		TestSamples.getMatSize(NumRows, NumCols);
		printf("TestSamples NumRows: %d\n", NumRows);
		//
		//getchar();
		//

		// TestLabels
		FloatMat TestLabels(1, ArrayNumNodes[NumLayers - 1]);
		//
		TestLabels.loadAllDataInFile(TestLabels_Filename);
		//
		printf("TestLabels loaded.\n");
		//
		//TestLabels.display();
		//
		TestLabels.getMatSize(NumRows, NumCols);
		printf("TestLabels NumRows: %d\n", NumRows);
		//
		getchar();
		//

		printf("Test Process ...\n");
		//
		SeedForRandom = 0;
		//NumMaxIter = 1;
		//
		FNN_Test(fnn, TestSamples, TestLabels);
		//
		printf("\n");
		printf("precision: %.4f\n", fnn.performance[0]);
		printf("recall: %.4f\n", fnn.performance[1]);
		printf("\n");
		//
		printf("Test Process Ended.\n");

		//
		getchar();
		//
	}
	//

	//
	if (FlagBackupNetFile == 1)
	{
		char FNN_Backup_Filename[128];
		if (FlagFiles == 1)
		{
			sprintf(FNN_Backup_Filename, "FNN_File_Ascend_%.4f_%.4f_%d_%d_%d_%d_%d_%.6f_%.6f_%.6f_%.6f.txt",
					fnn.performance[0], fnn.performance[1], SeedLearning, ErrBalance, LearningMethod, SeedForRandom, NumMaxIter,
					alpha, beta, delta, lamda);
		}
		else if (FlagFiles == -1)
		{
			sprintf(FNN_Backup_Filename, "FNN_File_Descend_%.4f_%.4f_%d_%d_%d_%d_%d_%.6f_%.6f_%.6f_%.6f.txt",
					fnn.performance[0], fnn.performance[1], SeedLearning, ErrBalance, LearningMethod, SeedForRandom, NumMaxIter,
					alpha, beta, delta, lamda);
		}
		else
		{
			sprintf(FNN_Backup_Filename, "FNN_File_%.4f_%.4f_%d_%d_%d_%d_%d_%.6f_%.6f_%.6f_%.6f.txt",
					fnn.performance[0], fnn.performance[1], SeedLearning, ErrBalance, LearningMethod, SeedForRandom, NumMaxIter,
					alpha, beta, delta, lamda);
		}

		fnn.writeToFile(FNN_Backup_Filename);
	}
	//

	//
	delete [] ArrayNumNodes;
	delete [] ArrayActs;
	//

	//
	printf("\n");
	printf("ApplicationFNN end.\n");

	getchar();
	return 0; 

}
//


//
void loadConfiguration()
{
	// Ä¬ÈÏÖµ
	FlagLoadFromFile = 0;
	FlagTraining = 0;
	FlagFiles = 0;
	//
	FlagBackupNetFile = 1;
	//
	SeedLearning = 10;
	Criteria = 0.50;
	//
	ErrBalance = 0;
	LearningMethod = 0;
	SeedForRandom = 10;
	NumMaxIter = 100;
	//
	alpha = 0.001;
	beta =0.999;
	delta = 0.00001;
	lamda = 0.6;
	//
	NumLayers = 3;
	ArrayNumNodes = new int[3]; // {2,2,2};
	ArrayActs = new int[2];   // {2,1};
	//
	ArrayNumNodes[0] = 2;
	ArrayNumNodes[1] = 2;
	ArrayNumNodes[2] = 2;
	//
	ArrayActs[0] = 2;
	ArrayActs[1] = 1;
	//

	//
	FILE * fid = fopen("AutoFNN_Configuration.txt","r");

	if (fid == NULL)
	{
		fid = fopen("AutoFNN_Configuration.txt","w");

		fprintf(fid, "FlagLoadFromFile: %d\n", FlagLoadFromFile);
		fprintf(fid, "FlagTraining: %d\n", FlagTraining);
		fprintf(fid, "FlagFiles: %d\n", FlagFiles);
		fprintf(fid, "FlagBackupNetFile: %d\n", FlagBackupNetFile);
		//
		fprintf(fid, "SeedLearning: %d\n", SeedLearning);
		fprintf(fid, "Criteria: %.2f\n", Criteria);
		//
		fprintf(fid, "ErrBalance: %d\n", ErrBalance);
		fprintf(fid, "LearningMethod: %d\n", LearningMethod);
		fprintf(fid, "SeedForRandom: %d\n", SeedForRandom);
		fprintf(fid, "MaxIter: %d\n", NumMaxIter);
		//
		fprintf(fid, "TrainingParas: %.6f, %.6f, %.6f, %.6f,\n", alpha, beta, delta, lamda);
		//
		fprintf(fid, "NumLayers: %d\n", NumLayers);
		//
		fprintf(fid, "NumNodes: ");
		for (int i = 0; i < NumLayers; i++) fprintf(fid, "%d, ", ArrayNumNodes[i]);
		fprintf(fid, "\n");
		//
		fprintf(fid, "ActType: ");
		for (int i = 0; i < NumLayers - 1; i++) fprintf(fid, "%d, ", ArrayActs[i]);
		fprintf(fid, "\n");
		//
		fprintf(fid, "End.");
		//

	}
	else
	{
		int LenBuff = 64;

		char * buff = new char[LenBuff];
		int curr;

		//
		while(fgets(buff, LenBuff, fid) != NULL)
		{
			if (strlen(buff) < 5) continue;

			//
			curr = 0;
			while (buff[curr] != ':')
			{
				curr++;
			}

			//
			buff[curr] = '\0';
			curr++;
			//
			if (strcmp(buff, "FlagLoadFromFile") == 0)         //
			{
				sscanf(buff + curr, "%d", &FlagLoadFromFile);
			}
			else if (strcmp(buff, "FlagTraining") == 0)
			{
				sscanf(buff + curr, "%d", &FlagTraining);
			}
			else if (strcmp(buff, "FlagFiles") == 0)
			{
				sscanf(buff + curr, "%d", &FlagFiles);
			}	
			else if (strcmp(buff, "FlagBackupNetFile") == 0)
			{
				sscanf(buff + curr, "%d", &FlagBackupNetFile);
			}
			else if (strcmp(buff, "SeedLearning") == 0)            //
			{
				sscanf(buff + curr, "%d", &SeedLearning);
			}
			else if (strcmp(buff, "Criteria") == 0)
			{
				sscanf(buff + curr, "%f", &Criteria);
			}	
			else if (strcmp(buff, "ErrBalance") == 0)             //
			{
				sscanf(buff + curr, "%d", &ErrBalance);
			}
			else if (strcmp(buff, "LearningMethod") == 0)
			{
				sscanf(buff + curr, "%d", &LearningMethod);
			}
			else if (strcmp(buff, "SeedForRandom") == 0)
			{
				sscanf(buff + curr, "%d", &SeedForRandom);
			}
			else if (strcmp(buff, "MaxIter") == 0)
			{
				sscanf(buff + curr, "%d", &NumMaxIter);
			}
			else if (strcmp(buff, "TrainingParas") == 0)         //
			{
				sscanf(buff + curr, "%f, %f, %f, %f,", &alpha, &beta, &delta, &lamda);
			}
			else if (strcmp(buff, "NumLayers") == 0)            //
			{
				sscanf(buff + curr, "%d", &NumLayers);
			}
			else if (strcmp(buff, "NumNodes") == 0)
			{
				//
				delete [] ArrayNumNodes;
				//
				ArrayNumNodes = new int[NumLayers];

				//
				int Posi = 0;
				char * str_begin = buff + curr;
				//
				while (buff[curr] != '\n')
				{
					if (buff[curr] == ',')
					{
						buff[curr] = '\0';

						sscanf(str_begin, "%d", ArrayNumNodes + Posi);

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
				}

			}
			else if (strcmp(buff, "ActType") == 0)
			{
				//
				delete [] ArrayActs;
				//
				ArrayActs = new int[NumLayers-1];

				//
				int Posi = 0;
				char * str_begin = buff + curr;
				//
				while (buff[curr] != '\n')
				{
					if (buff[curr] == ',')
					{
						buff[curr] = '\0';

						sscanf(str_begin, "%d", ArrayActs + Posi);

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
				}
			}

		}// while fgets

		//
		delete [] buff;
	}

	fclose(fid);
	//
}
//
