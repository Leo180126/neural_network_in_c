#include "NeuralNetwork.h"

void cal_accuracy(NeuralNetwork *nn, int **X, int **data){
    int numOfCorrect = 0;
    for(int i = 0; i < NUMOFTEST; i++){
        forwardPropagation(nn, X[i]);
        // Find the largest output
        double largest = 0;
        int largestIndex = 0;
        for(int i=0; i<nn->layerDescription[nn->numOfLayers - 1]; i++){
            if(largest < nn->A[nn->numOfLayers - 2][i]){
                largest = nn->A[nn->numOfLayers - 2][i];
                largestIndex = i;
            }
        }
        if(largestIndex == data[FIRSTTEST_EX + i][0])
            numOfCorrect ++;
    }
    printf("\nAccuracy = %.2lf %%\n", ((double)numOfCorrect/NUMOFTEST)*100.0);
}

void guesser(NeuralNetwork *nn, int example_index, int **data){
    printf("\nRandom Guesser !!!!!!!!!!\n");
    int X[784];
    for(int i=1; i<785; i++){
        X[i - 1] = data[example_index][i];
    }
    forwardPropagation(nn, X);
    for(int i = 0; i<784; i++){
        printf("%3d", X[i]);
        if(i % 28 == 27)printf("\n");
    }
    printOutput(nn);
}

void runTest(NeuralNetwork *nn, int **data){
    int **X = (int **)malloc(NUMOFTEST * sizeof(int *));
    int **Y = (int **)malloc(NUMOFTEST * sizeof(int *));
    
    for (int i = 0; i < NUMOFTEST; i++) {
        X[i] = (int *)malloc(784 * sizeof(int));
        Y[i] = (int *)malloc(10 * sizeof(int));
    }

    for (int i = FIRSTTEST_EX; i < FIRSTTEST_EX + NUMOFTEST; i++) {
        int idx = i - FIRSTTEST_EX;
        int label = data[i][0];
        for (int j = 0; j < 784; j++) {
            X[idx][j] = data[i][j + 1];
        }
        for (int j = 0; j < 10; j++) {
            Y[idx][j] = (j == label) ? 1 : 0;
        }
    }

    printf("\nMSE loss = %lf\n", cal_loss_test(nn, X, Y));
    cal_accuracy(nn, X, data);

    // Print some to the screen
    for (int i = 0; i < 3; i++) {
        printf("\nLabel: ");
        for (int j = 0; j < 10; j++) {
            if (Y[i][j] == 1) printf("%d\n", j);
        }
        for (int j = 0; j < 784; j++) {
            printf("%3d", X[i][j]);
            if (j % 28 == 27) printf("\n");  // mỗi dòng 28 pixel
        }
        forwardPropagation(nn, X[i]);
        printOutput(nn);
    }
    // Free
    for (int i = 0; i < NUMOFTEST; i++) {
        free(X[i]);
        free(Y[i]);
    }
    free(X);
    free(Y);
}


void loadTrainedModel(NeuralNetwork *nn, const char *filePath){
    FILE *fp = fopen(filePath, "r");
    if (fp == NULL) {
        perror("Không thể mở file");
        exit(EXIT_FAILURE);
    }
    // Load W
    for(int layerIndex = 0; layerIndex < nn->numOfLayers - 1; layerIndex++){
        for(int i=0; i < nn->layerDescription[layerIndex]; i++){
            for(int j = 0; j < nn->layerDescription[layerIndex + 1]; j++){
                fscanf(fp, "%lf", &nn->W[layerIndex][i][j]);
            }
        }
        //Bo qua 2 dong giua cac layer
        fgetc(fp); // đọc newline
        fgetc(fp); // đọc thêm newline nữa
    }

    // Load B
    for(int layerIndex = 0; layerIndex < nn->numOfLayers - 1; layerIndex++){
        for(int j = 0; j<nn->layerDescription[layerIndex + 1]; j++){
            fscanf(fp, "%lf", &nn->B[layerIndex][j]);
        }
    }
    fclose(fp);
}

void train(NeuralNetwork *nn, int **data, int epochs, double lr) {
    // Chuẩn bị dữ liệu đầu vào X và nhãn Y
    // thêm nếu cần
    int **X = (int **)malloc(NUMOFEXAMPLES * sizeof(int *));
    int **Y = (int **)malloc(NUMOFEXAMPLES * sizeof(int *));
    for (int i = 0; i < NUMOFEXAMPLES; i++) {
        X[i] = (int *)malloc(784 * sizeof(int));
        Y[i] = (int *)malloc(10 * sizeof(int));
    }
    for (int i = 0; i < NUMOFEXAMPLES; i++) {
        int label = data[i][0];
        for (int j = 0; j < 784; j++) {
            X[i][j] = data[i][j + 1];
        }
        for (int j = 0; j < 10; j++) {
            Y[i][j] = (j == label) ? 1 : 0;
        }
    }
    // // In thử mẫu đầu tiên
    // printf("Y[0]: ");
    // for (int j = 0; j < 10; j++) {
    //     printf("%d ", Y[0][j]);
    // }
    // printf("\n\nX[0] (ảnh 28x28):\n");

    // for (int j = 0; j < 784; j++) {
    //     printf("%3d ", X[0][j]);
    //     if ((j + 1) % 28 == 0) printf("\n");
    // }
    for(int epoch = 0; epoch < epochs; epoch++){
        for(int i = 0; i<NUMOFEXAMPLES; i++){
            forwardPropagation(nn, X[i]);
            backPropagation(nn, X[i], Y[i], lr);
        }
        if(epoch % 100 == 0){
            printf("Epoch %d loss %lf\n", epoch,cal_loss(nn, X, Y));
        }
    }
    // Free 
    for (int i = 0; i < NUMOFEXAMPLES; i++) {
        free(X[i]);
        free(Y[i]);
    }
    free(X);
    free(Y);
}

double cal_loss(NeuralNetwork *nn, int **X, int **Y) {
    double loss = 0.0;
    int outputSize = nn->layerDescription[nn->numOfLayers - 1];

    for (int j = 0; j < NUMOFEXAMPLES; j++) {
        forwardPropagation(nn, X[j]);  // Gọi trực tiếp hàm forward
        for (int i = 0; i < outputSize; i++) {
            double diff = (double)Y[j][i] - nn->A[nn->numOfLayers - 2][i];
            loss += (diff * diff) / 2.0;
        }
    }

    return loss / NUMOFEXAMPLES;
}

double cal_loss_test(NeuralNetwork *nn, int **X, int **Y) {
    double loss = 0.0;
    int outputSize = nn->layerDescription[nn->numOfLayers - 1];

    for (int j = 0; j < NUMOFTEST; j++) {
        forwardPropagation(nn, X[j]);  // Gọi trực tiếp hàm forward
        for (int i = 0; i < outputSize; i++) {
            double diff = (double)Y[j][i] - nn->A[nn->numOfLayers - 2][i];
            loss += (diff * diff) / 2.0;
        }
    }

    return loss / NUMOFTEST;
}

void backPropagation(NeuralNetwork *nn, int *X, int *Y, double lr){
    for(int layerIndex = nn->numOfLayers -2; layerIndex >= 0; layerIndex--){
        for (int j = 0; j < nn->layerDescription[layerIndex + 1]; j++){ 
            if(layerIndex == nn->numOfLayers - 2)
                nn->E[layerIndex][j] = (nn->A[layerIndex][j] - Y[j])*d_sigmoid(nn->A[layerIndex][j]);
            else{ 
                double sum = 0;
                for(int n = 0; n < nn->layerDescription[layerIndex + 2]; n++){
                    sum += (nn->E[layerIndex + 1][n]*nn->W[layerIndex + 1][j][n]);
                }
                nn->E[layerIndex][j] = sum*d_sigmoid(nn->A[layerIndex][j]);
            }
        }
    }
    // Updata weight
    for(int layerIndex = 0; layerIndex < nn->numOfLayers - 1; layerIndex++){
        for(int i = 0; i < nn->layerDescription[layerIndex]; i++){
            for(int j = 0; j < nn->layerDescription[layerIndex + 1]; j++){
                if(layerIndex == 0)
                    nn->W[layerIndex][i][j] -= lr*nn->E[layerIndex][j]*X[i];
                else{
                    nn->W[layerIndex][i][j] -= lr*nn->E[layerIndex][j]*nn->A[layerIndex - 1][i];
                }
            }
        }
    }
    // Update bias
    for(int layerIndex = 0; layerIndex < nn->numOfLayers - 1; layerIndex++){
        for(int j = 0; j < nn->layerDescription[layerIndex + 1]; j++){
            nn->B[layerIndex][j] -= lr*nn->E[layerIndex][j];
        }
    }
}
void forwardPropagation(NeuralNetwork *nn, int *X){
    for (int layerIndex = 0; layerIndex < nn->numOfLayers - 1; layerIndex++){
        // Tinh Z
        for(int j = 0; j < nn->layerDescription[layerIndex + 1]; j++){
            double sum = 0;
            for(int i = 0; i < nn->layerDescription[layerIndex]; i++){
                if(layerIndex == 0)
                    sum += nn->W[layerIndex][i][j]*(double)X[i];
                else
                    sum += nn->W[layerIndex][i][j]*nn->A[layerIndex - 1][i];
            }
            nn->Z[layerIndex][j] = sum + nn->B[layerIndex][j];
            // Tinh A
            nn->A[layerIndex][j] = sigmoid(nn->Z[layerIndex][j]);
        }
    }
}

void printParametter(NeuralNetwork *nn){
    FILE *pOutput = fopen("PARAMETTER", "w");
    //Print W
    // fprintf(pOutput, "W\n");
    for(int layerIndex = 0; layerIndex < nn->numOfLayers - 1; layerIndex++){
        for(int i = 0; i<nn->layerDescription[layerIndex]; i++){
            for(int j = 0; j<nn->layerDescription[layerIndex + 1]; j++){
                fprintf(pOutput, "%lf ", nn->W[layerIndex][i][j]);
            }
            fprintf(pOutput, "\n");
        }
        fprintf(pOutput,"\n");
        fprintf(pOutput,"\n");
    }
    // Print B
    // fprintf(pOutput, "B\n");
    for(int layerIndex = 0; layerIndex < nn->numOfLayers - 1; layerIndex++){
        for(int j = 0; j < nn->layerDescription[layerIndex + 1]; j++){
            fprintf(pOutput, "%lf ", nn->B[layerIndex][j]);
        }
        fprintf(pOutput, "\n");
    }
}

void printOutput(NeuralNetwork *nn){
    for(int i = 0; i < nn->layerDescription[nn->numOfLayers - 1]; i++){
        printf("%lf ", nn->A[nn->numOfLayers - 2][i]);
    }
}

void initNetwork(NeuralNetwork *nn,int numOfLayers, int *layerDescription)
{
    srand(time(NULL)); // Seed random
    // Some nn number
    nn->numOfLayers = numOfLayers;
    nn->layerDescription = (int *)malloc(numOfLayers * sizeof(int));
    for(int i = 0; i < numOfLayers; i++){
        nn->layerDescription[i] = layerDescription[i];
    }
    // The weights
    nn->W = (double ***)malloc((numOfLayers - 1)*sizeof(double *));
    for(int layerIndex = 0; layerIndex<numOfLayers - 1; layerIndex++){
        nn->W[layerIndex] = (double **)malloc(nn->layerDescription[layerIndex]*sizeof(double *));
        for(int i = 0; i < layerDescription[layerIndex]; i++){
            nn->W[layerIndex][i] = (double *)malloc(nn->layerDescription[layerIndex + 1] * sizeof(double));
            for(int j = 0; j < layerDescription[layerIndex + 1]; j++){
                nn->W[layerIndex][i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
            }
        }
    }
    //The Bias
    nn->B = (double **)malloc((nn->numOfLayers - 1)*sizeof(double *));
    for (int layerIndex=0; layerIndex<numOfLayers - 1; layerIndex++){
        nn->B[layerIndex] = (double *)malloc((nn->layerDescription[layerIndex + 1])*sizeof(double));
        for(int i=0; i < nn->layerDescription[layerIndex + 1]; i++){
            nn->B[layerIndex][i] = 0.0;
        }
    }
    //The Z
    nn->Z = (double **)malloc((nn->numOfLayers - 1)*sizeof(double *));
    for (int layerIndex=0; layerIndex<numOfLayers - 1; layerIndex++){
        nn->Z[layerIndex] = (double *)malloc((nn->layerDescription[layerIndex + 1])*sizeof(double));
        for(int i=0; i < nn->layerDescription[layerIndex + 1]; i++){
            nn->Z[layerIndex][i] = 0;
        }
    }
    //The A
    nn->A = (double **)malloc((nn->numOfLayers - 1)*sizeof(double *));
    for (int layerIndex=0; layerIndex<numOfLayers - 1; layerIndex++){
        nn->A[layerIndex] = (double *)malloc((nn->layerDescription[layerIndex + 1])*sizeof(double));
        for(int i=0; i < nn->layerDescription[layerIndex + 1]; i++){
            nn->A[layerIndex][i] = 0;
        }
    }
    //The E
    nn->E = (double **)malloc((nn->numOfLayers - 1)*sizeof(double *));
    for (int layerIndex=0; layerIndex<numOfLayers - 1; layerIndex++){
        nn->E[layerIndex] = (double *)malloc((nn->layerDescription[layerIndex + 1])*sizeof(double));
        for(int i=0; i < nn->layerDescription[layerIndex + 1]; i++){
            nn->E[layerIndex][i] = 0;
        }
    }
    //The Input X
    nn->X = (double *)malloc(nn->layerDescription[0]*sizeof(double));
};
void hamHuy(NeuralNetwork * nn){
    // Free W
    for(int layerIndex = 0; layerIndex < nn->numOfLayers - 1; layerIndex++) {
        for(int i = 0; i < nn->layerDescription[layerIndex]; i++) {
            free(nn->W[layerIndex][i]);
        }
        free(nn->W[layerIndex]);
    }
    free(nn->W);

    // Free layerDescription
    free(nn->layerDescription);

    // Free B
    for(int layerIndex = 0; layerIndex < nn->numOfLayers - 1; layerIndex++) {
        free(nn->B[layerIndex]);
    }
    free(nn->B);

    // Free A
    for(int layerIndex = 0; layerIndex < nn->numOfLayers - 1; layerIndex++) {
        free(nn->A[layerIndex]);
    }
    free(nn->A);

    // Free Z
    for(int layerIndex = 0; layerIndex < nn->numOfLayers - 1; layerIndex++) {
        free(nn->Z[layerIndex]);
    }
    free(nn->Z);

    // Free E
    for(int layerIndex = 0; layerIndex < nn->numOfLayers - 1; layerIndex++) {
        free(nn->E[layerIndex]);
    }
    free(nn->E);

    // Free X
    free(nn->X);
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double d_sigmoid(double x){
    return x*(1 - x);
}