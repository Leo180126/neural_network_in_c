#ifndef READ_CSV_H
#define READ_CSV_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LENGTH 2048
#define MAX_ROWS 60000     // chỉnh theo số dòng bạn cần
#define MAX_COLS 785       // 1 label + 784 pixels

int** readCSV(const char *filename, int *numRows);

#endif
