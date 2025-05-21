#include "MNIST.h"

int** readCSV(const char *filename, int *numRows) {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        perror("Không thể mở file");
        exit(EXIT_FAILURE);
    }

    int **data = (int **)malloc(MAX_ROWS * sizeof(int *));
    if (!data) {
        perror("Không thể cấp phát hàng");
        exit(EXIT_FAILURE);
    }

    char line[MAX_LINE_LENGTH];
    int row = 0;

    while (fgets(line, sizeof(line), fp) && row < MAX_ROWS) {
        data[row] = (int *)malloc(MAX_COLS * sizeof(int));
        if (!data[row]) {
            perror("Không thể cấp phát cột");
            exit(EXIT_FAILURE);
        }

        // line[strcspn(line, "\r\n")] = 0;

        char *token = strtok(line, ",");
        int col = 0;
        while (token != NULL && col < MAX_COLS) {
            data[row][col] = atoi(token);
            token = strtok(NULL, ",");
            col++;
        }

        row++;
    }

    fclose(fp);
    *numRows = row;  // trả lại số dòng thực tế đã đọc
    return data;
}