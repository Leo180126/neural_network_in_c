#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUMOFEXAMPLES 1000

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

void initNetwork(NeuralNetwork *nn,int numOfLayers, int *layerDescription);
void forwardPropagation(NeuralNetwork *nn, int *X);
void hamHuy(NeuralNetwork * nn);
void printParametter(NeuralNetwork *p_nn);
void printOutput(NeuralNetwork *p_nn);
double sigmoid(double x);
double d_sigmoid(double x);
void train(NeuralNetwork *nn, int **data, int epochs, double lr);
double cal_loss(NeuralNetwork *nn, int **X, int **Y);
void backPropagation(NeuralNetwork *nn, int *X, int *Y, double lr);
#endif // NEURALNETWORK_H