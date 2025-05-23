#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Chia 50000/10000 train/test data

#define NUMOFEXAMPLES 1000
#define FIRSTTEST_EX 50000
#define NUMOFTEST 1000

typedef struct {
    int numOfLayers;
    int *layerDescription;
    double ***W;
    double **B;
    double *X;
    double **A;
    double **Z;
    double **E;
} NeuralNetwork;

void runTest(NeuralNetwork *nn, int **data);
void initNetwork(NeuralNetwork *nn,int numOfLayers, int *layerDescription);
void forwardPropagation(NeuralNetwork *nn, int *X);
void hamHuy(NeuralNetwork * nn);
void printParametter(NeuralNetwork *p_nn);
void printOutput(NeuralNetwork *p_nn);
double sigmoid(double x);
double d_sigmoid(double x);
void train(NeuralNetwork *nn, int **data, int epochs, double lr);
double cal_loss(NeuralNetwork *nn, int **X, int **Y);
double cal_loss_test(NeuralNetwork *nn, int **X, int **Y);
void backPropagation(NeuralNetwork *nn, int *X, int *Y, double lr);
void loadTrainedModel(NeuralNetwork *nn, const char *filePath);
void guesser(NeuralNetwork *nn, int example_index, int **data);
#endif // NEURALNETWORK_H