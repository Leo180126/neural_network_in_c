#include "include/NeuralNetwork.h"
#include "include/MNIST.h"
int main(){
    NeuralNetwork nn;
    NeuralNetwork *p_nn = &nn;
    int layerDescription[4] = {784, 100, 80, 10};
    initNetwork(p_nn, 4, layerDescription);
    loadTrainedModel(p_nn, "OUTPUT");
    // for(int layerIndex = 0; layerIndex < p_nn->numOfLayers - 1; layerIndex++){
    //     for(int i = 0; i<p_nn->layerDescription[layerIndex]; i++){
    //         for(int j = 0; j<p_nn->layerDescription[layerIndex + 1]; j++){
    //             printf("%lf ", p_nn->W[layerIndex][i][j]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    //     printf("\n");
    // }
    // // Print B
    // // fprintf(pOutput, "B\n");
    // for(int layerIndex = 0; layerIndex < p_nn->numOfLayers - 1; layerIndex++){
    //     for(int j = 0; j < p_nn->layerDescription[layerIndex + 1]; j++){
    //         printf("%lf ", p_nn->B[layerIndex][j]);
    //     }
    //     printf("\n");
    // }
    int numRows;
    int **data = readCSV("include\\mnist_train.csv", &numRows);
    // int X[784];
    // for(int i=1; i<785; i++){
    //     X[i - 1] = data[0][i];
    // }
    // for(int i=0; i<784; i++){
    //     printf("%3d", X[i]);
    //     if(i % 28 == 0)printf("\n");
    // }
    // forwardPropagation(p_nn, X);
    // printOutput(p_nn);
    runTest(p_nn, data);
    // Random guesser
    guesser(p_nn, 40602, data);

    // Giai phong bo nho
    for (int i = 0; i < numRows; i++) {
        free(data[i]);
    }
    free(data);
    hamHuy(p_nn);
    return 0;
}