#include "include/NeuralNetwork.h"
#include "include/MNIST.h"

int main(){
    NeuralNetwork nn;
    NeuralNetwork *p_nn = &nn;
    int numRows;
    int layerDescription[4] = {784, 100, 80, 10};
    initNetwork(p_nn, 4, layerDescription);
    const char *path = "include\\mnist_train.csv";
    int **data = readCSV(path, &numRows);
    // printf("Before calling train\n");
    if (data == NULL) {
        printf("data is NULL\n");
        exit(1);
    }
    if (data[0] == NULL) {
        printf("data[0] is NULL\n");
        exit(1);
    }
    if (p_nn == NULL) {
        printf("Neural network pointer is NULL\n");
        exit(1);
    }
    // int X[784];
    // for(int i=1; i<785; i++){
    //     X[i - 1] = data[0][i];
    // }
    // for(int i=0; i<784; i++){
    //     printf("%3d", X[i]);
    //     if(i % 28 == 0)printf("\n");
    // }
    // printf("%d", data[0][0]);
    train(p_nn, data, 1000, 0.01);


    // Print some to the screen
    int X[3][784];
    int Y[3];
    for(int i = 0; i < 3; i++){
        for(int j=1; j < 785; j++){
            X[i][j - 1] = data[i][j];
        }
    }
    for(int i = 0; i < 3; i++){
        Y[i] = data[i][0];
    }
    for(int i = 0; i < 3; i++){
        printf("\nLabel: %d\n", Y[i]);
        for(int j = 0; j<784; j++){
            printf("%3d", X[i][j]);
            if(j % 28 == 27)printf("\n");
        }
        forwardPropagation(p_nn, X[i]);
        printOutput(p_nn);
    }
    printParametter(p_nn);
    //Giai phong bo nho
    for (int i = 0; i < numRows; i++) {
        free(data[i]);
    }
    free(data);
    // printW(p_nn);
    hamHuy(p_nn);
    return 0;
}

