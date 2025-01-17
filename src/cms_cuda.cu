#include <iostream>
#include <fstream>
#include <sstream>
#include <cstddef>

#include <sys/stat.h>
#include <unistd.h>
#include <omp.h>
#include <string.h>
#include <cuda_runtime.h>
#include <time.h>

#include "comp.h"

#define REG_SIZE 150
#define MIN_DIST 0.00001
#define DEFAULT_SHARED_SIZE 1728
#define DEFAULT_BLOCK_SIZE 8

using namespace std;

char Ref_db;

string Listfilename;
string Listprefix;
string Queryfile1;
string Queryfile2;

string Tablefilename;
string Outfilename;

int Coren = 0;

bool Is_cp_correct; //
bool Is_sim; //true: sim, false: dist;
bool Is_heatmap;
int Cluster = 2;

//bool Is_weight;
int Dist_metric = 0; //0: MS; 1: MS-uw; 2: cos 3: eu; 4: JSD; 5.Bray Curtis

int Mode = 0; //0: single, 1: multi_list, 2: multi_table
//bool Reversed_table = true;

typedef struct {
    int start_row;
    int end_row;
    int start_col;
    int end_col;
} Block;

__global__ void
CalcSimOfDiagonalSquare(float **d_Abd, float *d_sim_matrix, int *d_order_row, int *d_order_col, int num_elements, int OrderN, float *Dist_1,
                        float *Dist_2, int *Order_1, int *Order_2, int *Order_d, int sharedMemorySize) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ float sharedMemory[];
    float *s_dist_1 = sharedMemory;
    for (int count = 0; count < sharedMemorySize; count++) {
        s_dist_1[count] = Dist_1[count];
    }
    __syncthreads();

    if (tid < num_elements) {
        int row = d_order_row[tid];
        int col = d_order_col[tid];

        float *Abd_1 = d_Abd[row];
        float *Abd_2 = d_Abd[col];

        float Reg_1[REG_SIZE];
        float Reg_2[REG_SIZE];
        float total = 0;

        int i;
        //expansion loop
        for (i = 0; i <= OrderN - 10; i += 10) {
            int order_1_0 = Order_1[i];
            int order_2_0 = Order_2[i];
            int order_d_0 = Order_d[i] + REG_SIZE;

            int order_1_1 = Order_1[i + 1];
            int order_2_1 = Order_2[i + 1];
            int order_d_1 = Order_d[i + 1] + REG_SIZE;

            int order_1_2 = Order_1[i + 2];
            int order_2_2 = Order_2[i + 2];
            int order_d_2 = Order_d[i + 2] + REG_SIZE;

            int order_1_3 = Order_1[i + 3];
            int order_2_3 = Order_2[i + 3];
            int order_d_3 = Order_d[i + 3] + REG_SIZE;

            int order_1_4 = Order_1[i + 4];
            int order_2_4 = Order_2[i + 4];
            int order_d_4 = Order_d[i + 4] + REG_SIZE;

            int order_1_5 = Order_1[i + 5];
            int order_2_5 = Order_2[i + 5];
            int order_d_5 = Order_d[i + 5] + REG_SIZE;

            int order_1_6 = Order_1[i + 6];
            int order_2_6 = Order_2[i + 6];
            int order_d_6 = Order_d[i + 6] + REG_SIZE;

            int order_1_7 = Order_1[i + 7];
            int order_2_7 = Order_2[i + 7];
            int order_d_7 = Order_d[i + 7] + REG_SIZE;

            int order_1_8 = Order_1[i + 8];
            int order_2_8 = Order_2[i + 8];
            int order_d_8 = Order_d[i + 8] + REG_SIZE;

            int order_1_9 = Order_1[i + 9];
            int order_2_9 = Order_2[i + 9];
            int order_d_9 = Order_d[i + 9] + REG_SIZE;

            float dist_2_0 = 1 - Dist_2[i];
            float dist_2_1 = 1 - Dist_2[i + 1];
            float dist_2_2 = 1 - Dist_2[i + 2];
            float dist_2_3 = 1 - Dist_2[i + 3];
            float dist_2_4 = 1 - Dist_2[i + 4];
            float dist_2_5 = 1 - Dist_2[i + 5];
            float dist_2_6 = 1 - Dist_2[i + 6];
            float dist_2_7 = 1 - Dist_2[i + 7];
            float dist_2_8 = 1 - Dist_2[i + 8];
            float dist_2_9 = 1 - Dist_2[i + 9];

            float dist_1_0;
            float dist_1_1;
            float dist_1_2;
            float dist_1_3;
            float dist_1_4;
            float dist_1_5;
            float dist_1_6;
            float dist_1_7;
            float dist_1_8;
            float dist_1_9;

            if (i < sharedMemorySize) {
                dist_1_0 = 1 - s_dist_1[i];
                dist_1_1 = 1 - s_dist_1[i + 1];
                dist_1_2 = 1 - s_dist_1[i + 2];
                dist_1_3 = 1 - s_dist_1[i + 3];
                dist_1_4 = 1 - s_dist_1[i + 4];
                dist_1_5 = 1 - s_dist_1[i + 5];
                dist_1_6 = 1 - s_dist_1[i + 6];
                dist_1_7 = 1 - s_dist_1[i + 7];
                dist_1_8 = 1 - s_dist_1[i + 8];
                dist_1_9 = 1 - s_dist_1[i + 9];
            } else {
                dist_1_0 = 1 - Dist_1[i];
                dist_1_1 = 1 - Dist_1[i + 1];
                dist_1_2 = 1 - Dist_1[i + 2];
                dist_1_3 = 1 - Dist_1[i + 3];
                dist_1_4 = 1 - Dist_1[i + 4];
                dist_1_5 = 1 - Dist_1[i + 5];
                dist_1_6 = 1 - Dist_1[i + 6];
                dist_1_7 = 1 - Dist_1[i + 7];
                dist_1_8 = 1 - Dist_1[i + 8];
                dist_1_9 = 1 - Dist_1[i + 9];
            }

            float c1_1_0, c1_2_0;
            float c2_1_0, c2_2_0;

            float c1_1_1, c1_2_1;
            float c2_1_1, c2_2_1;

            float c1_1_2, c1_2_2;
            float c2_1_2, c2_2_2;

            float c1_1_3, c1_2_3;
            float c2_1_3, c2_2_3;

            float c1_1_4, c1_2_4;
            float c2_1_4, c2_2_4;

            float c1_1_5, c1_2_5;
            float c2_1_5, c2_2_5;

            float c1_1_6, c1_2_6;
            float c2_1_6, c2_2_6;

            float c1_1_7, c1_2_7;
            float c2_1_7, c2_2_7;

            float c1_1_8, c1_2_8;
            float c2_1_8, c2_2_8;

            float c1_1_9, c1_2_9;
            float c2_1_9, c2_2_9;

            //first step
            if (order_1_0 >= 0) {
                c1_1_0 = Abd_1[order_1_0];
                c1_2_0 = Abd_2[order_1_0];
            } else {
                c1_1_0 = Reg_1[order_1_0 + REG_SIZE];
                c1_2_0 = Reg_2[order_1_0 + REG_SIZE];
            }
            if (order_2_0 >= 0) {
                c2_1_0 = Abd_1[order_2_0];
                c2_2_0 = Abd_2[order_2_0];
            } else {
                c2_1_0 = Reg_1[order_2_0 + REG_SIZE];
                c2_2_0 = Reg_2[order_2_0 + REG_SIZE];
            }

            float min_1_0 = fminf(c1_1_0, c1_2_0);
            float min_2_0 = fminf(c2_1_0, c2_2_0);

            total += (min_1_0 + min_2_0);

            Reg_1[order_d_0] = (c1_1_0 - min_1_0) * dist_1_0 + (c2_1_0 - min_2_0) * dist_2_0;
            Reg_2[order_d_0] = (c1_2_0 - min_1_0) * dist_1_0 + (c2_2_0 - min_2_0) * dist_2_0;

            // second step
            if (order_1_1 >= 0) {
                c1_1_1 = Abd_1[order_1_1];
                c1_2_1 = Abd_2[order_1_1];
            } else {
                c1_1_1 = Reg_1[order_1_1 + REG_SIZE];
                c1_2_1 = Reg_2[order_1_1 + REG_SIZE];
            }

            if (order_2_1 >= 0) {
                c2_1_1 = Abd_1[order_2_1];
                c2_2_1 = Abd_2[order_2_1];
            } else {
                c2_1_1 = Reg_1[order_2_1 + REG_SIZE];
                c2_2_1 = Reg_2[order_2_1 + REG_SIZE];
            }

            float min_1_1 = fminf(c1_1_1, c1_2_1);
            float min_2_1 = fminf(c2_1_1, c2_2_1);

            total += (min_1_1 + min_2_1);

            Reg_1[order_d_1] = (c1_1_1 - min_1_1) * dist_1_1 + (c2_1_1 - min_2_1) * dist_2_1;
            Reg_2[order_d_1] = (c1_2_1 - min_1_1) * dist_1_1 + (c2_2_1 - min_2_1) * dist_2_1;

            //third step
            if (order_1_2 >= 0) {
                c1_1_2 = Abd_1[order_1_2];
                c1_2_2 = Abd_2[order_1_2];
            } else {
                c1_1_2 = Reg_1[order_1_2 + REG_SIZE];
                c1_2_2 = Reg_2[order_1_2 + REG_SIZE];
            }

            if (order_2_2 >= 0) {
                c2_1_2 = Abd_1[order_2_2];
                c2_2_2 = Abd_2[order_2_2];
            } else {
                c2_1_2 = Reg_1[order_2_2 + REG_SIZE];
                c2_2_2 = Reg_2[order_2_2 + REG_SIZE];
            }

            float min_1_2 = fminf(c1_1_2, c1_2_2);
            float min_2_2 = fminf(c2_1_2, c2_2_2);

            total += (min_1_2 + min_2_2);

            Reg_1[order_d_2] = (c1_1_2 - min_1_2) * dist_1_2 + (c2_1_2 - min_2_2) * dist_2_2;
            Reg_2[order_d_2] = (c1_2_2 - min_1_2) * dist_1_2 + (c2_2_2 - min_2_2) * dist_2_2;

            //fourth step
            if (order_1_3 >= 0) {
                c1_1_3 = Abd_1[order_1_3];
                c1_2_3 = Abd_2[order_1_3];
            } else {
                c1_1_3 = Reg_1[order_1_3 + REG_SIZE];
                c1_2_3 = Reg_2[order_1_3 + REG_SIZE];
            }

            if (order_2_3 >= 0) {
                c2_1_3 = Abd_1[order_2_3];
                c2_2_3 = Abd_2[order_2_3];
            } else {
                c2_1_3 = Reg_1[order_2_3 + REG_SIZE];
                c2_2_3 = Reg_2[order_2_3 + REG_SIZE];
            }

            float min_1_3 = fminf(c1_1_3, c1_2_3);
            float min_2_3 = fminf(c2_1_3, c2_2_3);

            total += (min_1_3 + min_2_3);

            Reg_1[order_d_3] = (c1_1_3 - min_1_3) * dist_1_3 + (c2_1_3 - min_2_3) * dist_2_3;
            Reg_2[order_d_3] = (c1_2_3 - min_1_3) * dist_1_3 + (c2_2_3 - min_2_3) * dist_2_3;

            //5th
            if (order_1_4 >= 0) {
                c1_1_4 = Abd_1[order_1_4];
                c1_2_4 = Abd_2[order_1_4];
            } else {
                c1_1_4 = Reg_1[order_1_4 + REG_SIZE];
                c1_2_4 = Reg_2[order_1_4 + REG_SIZE];
            }

            if (order_2_4 >= 0) {
                c2_1_4 = Abd_1[order_2_4];
                c2_2_4 = Abd_2[order_2_4];
            } else {
                c2_1_4 = Reg_1[order_2_4 + REG_SIZE];
                c2_2_4 = Reg_2[order_2_4 + REG_SIZE];
            }

            float min_1_4 = fminf(c1_1_4, c1_2_4);
            float min_2_4 = fminf(c2_1_4, c2_2_4);

            total += (min_1_4 + min_2_4);

            Reg_1[order_d_4] = (c1_1_4 - min_1_4) * dist_1_4 + (c2_1_4 - min_2_4) * dist_2_4;
            Reg_2[order_d_4] = (c1_2_4 - min_1_4) * dist_1_4 + (c2_2_4 - min_2_4) * dist_2_4;

            //6th
            if (order_1_5 >= 0) {
                c1_1_5 = Abd_1[order_1_5];
                c1_2_5 = Abd_2[order_1_5];
            } else {
                c1_1_5 = Reg_1[order_1_5 + REG_SIZE];
                c1_2_5 = Reg_2[order_1_5 + REG_SIZE];
            }

            if (order_2_5 >= 0) {
                c2_1_5 = Abd_1[order_2_5];
                c2_2_5 = Abd_2[order_2_5];
            } else {
                c2_1_5 = Reg_1[order_2_5 + REG_SIZE];
                c2_2_5 = Reg_2[order_2_5 + REG_SIZE];
            }

            float min_1_5 = fminf(c1_1_5, c1_2_5);
            float min_2_5 = fminf(c2_1_5, c2_2_5);

            total += (min_1_5 + min_2_5);

            Reg_1[order_d_5] = (c1_1_5 - min_1_5) * dist_1_5 + (c2_1_5 - min_2_5) * dist_2_5;
            Reg_2[order_d_5] = (c1_2_5 - min_1_5) * dist_1_5 + (c2_2_5 - min_2_5) * dist_2_5;

            //7th
            if (order_1_6 >= 0) {
                c1_1_6 = Abd_1[order_1_6];
                c1_2_6 = Abd_2[order_1_6];
            } else {
                c1_1_6 = Reg_1[order_1_6 + REG_SIZE];
                c1_2_6 = Reg_2[order_1_6 + REG_SIZE];
            }

            if (order_2_6 >= 0) {
                c2_1_6 = Abd_1[order_2_6];
                c2_2_6 = Abd_2[order_2_6];
            } else {
                c2_1_6 = Reg_1[order_2_6 + REG_SIZE];
                c2_2_6 = Reg_2[order_2_6 + REG_SIZE];
            }

            float min_1_6 = fminf(c1_1_6, c1_2_6);
            float min_2_6 = fminf(c2_1_6, c2_2_6);

            total += (min_1_6 + min_2_6);

            Reg_1[order_d_6] = (c1_1_6 - min_1_6) * dist_1_6 + (c2_1_6 - min_2_6) * dist_2_6;
            Reg_2[order_d_6] = (c1_2_6 - min_1_6) * dist_1_6 + (c2_2_6 - min_2_6) * dist_2_6;

            //8th
            if (order_1_7 >= 0) {
                c1_1_7 = Abd_1[order_1_7];
                c1_2_7 = Abd_2[order_1_7];
            } else {
                c1_1_7 = Reg_1[order_1_7 + REG_SIZE];
                c1_2_7 = Reg_2[order_1_7 + REG_SIZE];
            }

            if (order_2_7 >= 0) {
                c2_1_7 = Abd_1[order_2_7];
                c2_2_7 = Abd_2[order_2_7];
            } else {
                c2_1_7 = Reg_1[order_2_7 + REG_SIZE];
                c2_2_7 = Reg_2[order_2_7 + REG_SIZE];
            }

            float min_1_7 = fminf(c1_1_7, c1_2_7);
            float min_2_7 = fminf(c2_1_7, c2_2_7);

            total += (min_1_7 + min_2_7);

            Reg_1[order_d_7] = (c1_1_7 - min_1_7) * dist_1_7 + (c2_1_7 - min_2_7) * dist_2_7;
            Reg_2[order_d_7] = (c1_2_7 - min_1_7) * dist_1_7 + (c2_2_7 - min_2_7) * dist_2_7;

            //9th
            if (order_1_8 >= 0) {
                c1_1_8 = Abd_1[order_1_8];
                c1_2_8 = Abd_2[order_1_8];
            } else {
                c1_1_8 = Reg_1[order_1_8 + REG_SIZE];
                c1_2_8 = Reg_2[order_1_8 + REG_SIZE];
            }

            if (order_2_8 >= 0) {
                c2_1_8 = Abd_1[order_2_8];
                c2_2_8 = Abd_2[order_2_8];
            } else {
                c2_1_8 = Reg_1[order_2_8 + REG_SIZE];
                c2_2_8 = Reg_2[order_2_8 + REG_SIZE];
            }

            float min_1_8 = fminf(c1_1_8, c1_2_8);
            float min_2_8 = fminf(c2_1_8, c2_2_8);

            total += (min_1_8 + min_2_8);

            Reg_1[order_d_8] = (c1_1_8 - min_1_8) * dist_1_8 + (c2_1_8 - min_2_8) * dist_2_8;
            Reg_2[order_d_8] = (c1_2_8 - min_1_8) * dist_1_8 + (c2_2_8 - min_2_8) * dist_2_8;

            //10th
            if (order_1_9 >= 0) {
                c1_1_9 = Abd_1[order_1_9];
                c1_2_9 = Abd_2[order_1_9];
            } else {
                c1_1_9 = Reg_1[order_1_9 + REG_SIZE];
                c1_2_9 = Reg_2[order_1_9 + REG_SIZE];
            }

            if (order_2_9 >= 0) {
                c2_1_9 = Abd_1[order_2_9];
                c2_2_9 = Abd_2[order_2_9];
            } else {
                c2_1_9 = Reg_1[order_2_9 + REG_SIZE];
                c2_2_9 = Reg_2[order_2_9 + REG_SIZE];
            }

            float min_1_9 = fminf(c1_1_9, c1_2_9);
            float min_2_9 = fminf(c2_1_9, c2_2_9);

            total += (min_1_9 + min_2_9);

            Reg_1[order_d_9] = (c1_1_9 - min_1_9) * dist_1_9 + (c2_1_9 - min_2_9) * dist_2_9;
            Reg_2[order_d_9] = (c1_2_9 - min_1_9) * dist_1_9 + (c2_2_9 - min_2_9) * dist_2_9;
        }
        //Process elements haven't been divided
        for (; i < OrderN; i++) {
            int order_1 = Order_1[i];
            int order_2 = Order_2[i];
            int order_d = Order_d[i] + REG_SIZE;

            float dist_2 = 1 - Dist_2[i];
            float dist_1;

            if (i < sharedMemorySize) {
                dist_1 = 1 - s_dist_1[i];
            } else {
                dist_1 = 1 - Dist_1[i];
            }

            float c1_1, c1_2;
            float c2_1, c2_2;

            if (order_1 >= 0) {
                c1_1 = Abd_1[order_1];
                c1_2 = Abd_2[order_1];
            } else {
                c1_1 = Reg_1[order_1 + REG_SIZE];
                c1_2 = Reg_2[order_1 + REG_SIZE];
            }

            if (order_2 >= 0) {
                c2_1 = Abd_1[order_2];
                c2_2 = Abd_2[order_2];
            } else {
                c2_1 = Reg_1[order_2 + REG_SIZE];
                c2_2 = Reg_2[order_2 + REG_SIZE];
            }

            float min_1 = fminf(c1_1, c1_2);
            float min_2 = fminf(c2_1, c2_2);
            total += (min_1 + min_2);

            Reg_1[order_d] = (c1_1 - min_1) * dist_1 + (c2_1 - min_2) * dist_2;
            Reg_2[order_d] = (c1_2 - min_1) * dist_1 + (c2_2 - min_2) * dist_2;
        }

        total *= 0.01f;
        total = fminf(1.0f, fmaxf(0.0f, total));
        d_sim_matrix[tid] = total;

    } else {
        return;
    }
}

__global__ void
CalcSimOfNormalRectangle(float **Abd_row, float **Abd_col, float *d_sim_matrix, int *d_order_row, int *d_order_col, int num_elements, int OrderN,
                         float *Dist_1, float *Dist_2, int *Order_1, int *Order_2, int *Order_d, int sharedMemorySize) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ float sharedMemory[];
    float *s_dist_1 = sharedMemory;
    for (int count = 0; count < sharedMemorySize; count++) {
        s_dist_1[count] = Dist_1[count];
    }
    __syncthreads();

    if (tid < num_elements) {
        int row = d_order_row[tid];
        int col = d_order_col[tid];

        float *Abd_1 = Abd_row[row];
        float *Abd_2 = Abd_col[col];

        float Reg_1[REG_SIZE];
        float Reg_2[REG_SIZE];
        float total = 0;

        int i;
        //expansion loop
        for (i = 0; i <= OrderN - 10; i += 10) {
            int order_1_0 = Order_1[i];
            int order_2_0 = Order_2[i];
            int order_d_0 = Order_d[i] + REG_SIZE;

            int order_1_1 = Order_1[i + 1];
            int order_2_1 = Order_2[i + 1];
            int order_d_1 = Order_d[i + 1] + REG_SIZE;

            int order_1_2 = Order_1[i + 2];
            int order_2_2 = Order_2[i + 2];
            int order_d_2 = Order_d[i + 2] + REG_SIZE;

            int order_1_3 = Order_1[i + 3];
            int order_2_3 = Order_2[i + 3];
            int order_d_3 = Order_d[i + 3] + REG_SIZE;

            int order_1_4 = Order_1[i + 4];
            int order_2_4 = Order_2[i + 4];
            int order_d_4 = Order_d[i + 4] + REG_SIZE;

            int order_1_5 = Order_1[i + 5];
            int order_2_5 = Order_2[i + 5];
            int order_d_5 = Order_d[i + 5] + REG_SIZE;

            int order_1_6 = Order_1[i + 6];
            int order_2_6 = Order_2[i + 6];
            int order_d_6 = Order_d[i + 6] + REG_SIZE;

            int order_1_7 = Order_1[i + 7];
            int order_2_7 = Order_2[i + 7];
            int order_d_7 = Order_d[i + 7] + REG_SIZE;

            int order_1_8 = Order_1[i + 8];
            int order_2_8 = Order_2[i + 8];
            int order_d_8 = Order_d[i + 8] + REG_SIZE;

            int order_1_9 = Order_1[i + 9];
            int order_2_9 = Order_2[i + 9];
            int order_d_9 = Order_d[i + 9] + REG_SIZE;

            float dist_2_0 = 1 - Dist_2[i];
            float dist_2_1 = 1 - Dist_2[i + 1];
            float dist_2_2 = 1 - Dist_2[i + 2];
            float dist_2_3 = 1 - Dist_2[i + 3];
            float dist_2_4 = 1 - Dist_2[i + 4];
            float dist_2_5 = 1 - Dist_2[i + 5];
            float dist_2_6 = 1 - Dist_2[i + 6];
            float dist_2_7 = 1 - Dist_2[i + 7];
            float dist_2_8 = 1 - Dist_2[i + 8];
            float dist_2_9 = 1 - Dist_2[i + 9];

            float dist_1_0;
            float dist_1_1;
            float dist_1_2;
            float dist_1_3;
            float dist_1_4;
            float dist_1_5;
            float dist_1_6;
            float dist_1_7;
            float dist_1_8;
            float dist_1_9;

            if (i < sharedMemorySize) {
                dist_1_0 = 1 - s_dist_1[i];
                dist_1_1 = 1 - s_dist_1[i + 1];
                dist_1_2 = 1 - s_dist_1[i + 2];
                dist_1_3 = 1 - s_dist_1[i + 3];
                dist_1_4 = 1 - s_dist_1[i + 4];
                dist_1_5 = 1 - s_dist_1[i + 5];
                dist_1_6 = 1 - s_dist_1[i + 6];
                dist_1_7 = 1 - s_dist_1[i + 7];
                dist_1_8 = 1 - s_dist_1[i + 8];
                dist_1_9 = 1 - s_dist_1[i + 9];
            } else {
                dist_1_0 = 1 - Dist_1[i];
                dist_1_1 = 1 - Dist_1[i + 1];
                dist_1_2 = 1 - Dist_1[i + 2];
                dist_1_3 = 1 - Dist_1[i + 3];
                dist_1_4 = 1 - Dist_1[i + 4];
                dist_1_5 = 1 - Dist_1[i + 5];
                dist_1_6 = 1 - Dist_1[i + 6];
                dist_1_7 = 1 - Dist_1[i + 7];
                dist_1_8 = 1 - Dist_1[i + 8];
                dist_1_9 = 1 - Dist_1[i + 9];
            }

            float c1_1_0, c1_2_0;
            float c2_1_0, c2_2_0;

            float c1_1_1, c1_2_1;
            float c2_1_1, c2_2_1;

            float c1_1_2, c1_2_2;
            float c2_1_2, c2_2_2;

            float c1_1_3, c1_2_3;
            float c2_1_3, c2_2_3;

            float c1_1_4, c1_2_4;
            float c2_1_4, c2_2_4;

            float c1_1_5, c1_2_5;
            float c2_1_5, c2_2_5;

            float c1_1_6, c1_2_6;
            float c2_1_6, c2_2_6;

            float c1_1_7, c1_2_7;
            float c2_1_7, c2_2_7;

            float c1_1_8, c1_2_8;
            float c2_1_8, c2_2_8;

            float c1_1_9, c1_2_9;
            float c2_1_9, c2_2_9;

            //first step
            if (order_1_0 >= 0) {
                c1_1_0 = Abd_1[order_1_0];
                c1_2_0 = Abd_2[order_1_0];
            } else {
                c1_1_0 = Reg_1[order_1_0 + REG_SIZE];
                c1_2_0 = Reg_2[order_1_0 + REG_SIZE];
            }
            if (order_2_0 >= 0) {
                c2_1_0 = Abd_1[order_2_0];
                c2_2_0 = Abd_2[order_2_0];
            } else {
                c2_1_0 = Reg_1[order_2_0 + REG_SIZE];
                c2_2_0 = Reg_2[order_2_0 + REG_SIZE];
            }

            float min_1_0 = fminf(c1_1_0, c1_2_0);
            float min_2_0 = fminf(c2_1_0, c2_2_0);

            total += (min_1_0 + min_2_0);

            Reg_1[order_d_0] = (c1_1_0 - min_1_0) * dist_1_0 + (c2_1_0 - min_2_0) * dist_2_0;
            Reg_2[order_d_0] = (c1_2_0 - min_1_0) * dist_1_0 + (c2_2_0 - min_2_0) * dist_2_0;

            // second step
            if (order_1_1 >= 0) {
                c1_1_1 = Abd_1[order_1_1];
                c1_2_1 = Abd_2[order_1_1];
            } else {
                c1_1_1 = Reg_1[order_1_1 + REG_SIZE];
                c1_2_1 = Reg_2[order_1_1 + REG_SIZE];
            }

            if (order_2_1 >= 0) {
                c2_1_1 = Abd_1[order_2_1];
                c2_2_1 = Abd_2[order_2_1];
            } else {
                c2_1_1 = Reg_1[order_2_1 + REG_SIZE];
                c2_2_1 = Reg_2[order_2_1 + REG_SIZE];
            }

            float min_1_1 = fminf(c1_1_1, c1_2_1);
            float min_2_1 = fminf(c2_1_1, c2_2_1);

            total += (min_1_1 + min_2_1);

            Reg_1[order_d_1] = (c1_1_1 - min_1_1) * dist_1_1 + (c2_1_1 - min_2_1) * dist_2_1;
            Reg_2[order_d_1] = (c1_2_1 - min_1_1) * dist_1_1 + (c2_2_1 - min_2_1) * dist_2_1;

            //third step
            if (order_1_2 >= 0) {
                c1_1_2 = Abd_1[order_1_2];
                c1_2_2 = Abd_2[order_1_2];
            } else {
                c1_1_2 = Reg_1[order_1_2 + REG_SIZE];
                c1_2_2 = Reg_2[order_1_2 + REG_SIZE];
            }

            if (order_2_2 >= 0) {
                c2_1_2 = Abd_1[order_2_2];
                c2_2_2 = Abd_2[order_2_2];
            } else {
                c2_1_2 = Reg_1[order_2_2 + REG_SIZE];
                c2_2_2 = Reg_2[order_2_2 + REG_SIZE];
            }

            float min_1_2 = fminf(c1_1_2, c1_2_2);
            float min_2_2 = fminf(c2_1_2, c2_2_2);

            total += (min_1_2 + min_2_2);

            Reg_1[order_d_2] = (c1_1_2 - min_1_2) * dist_1_2 + (c2_1_2 - min_2_2) * dist_2_2;
            Reg_2[order_d_2] = (c1_2_2 - min_1_2) * dist_1_2 + (c2_2_2 - min_2_2) * dist_2_2;

            //fourth step
            if (order_1_3 >= 0) {
                c1_1_3 = Abd_1[order_1_3];
                c1_2_3 = Abd_2[order_1_3];
            } else {
                c1_1_3 = Reg_1[order_1_3 + REG_SIZE];
                c1_2_3 = Reg_2[order_1_3 + REG_SIZE];
            }

            if (order_2_3 >= 0) {
                c2_1_3 = Abd_1[order_2_3];
                c2_2_3 = Abd_2[order_2_3];
            } else {
                c2_1_3 = Reg_1[order_2_3 + REG_SIZE];
                c2_2_3 = Reg_2[order_2_3 + REG_SIZE];
            }

            float min_1_3 = fminf(c1_1_3, c1_2_3);
            float min_2_3 = fminf(c2_1_3, c2_2_3);

            total += (min_1_3 + min_2_3);

            Reg_1[order_d_3] = (c1_1_3 - min_1_3) * dist_1_3 + (c2_1_3 - min_2_3) * dist_2_3;
            Reg_2[order_d_3] = (c1_2_3 - min_1_3) * dist_1_3 + (c2_2_3 - min_2_3) * dist_2_3;

            //5th
            if (order_1_4 >= 0) {
                c1_1_4 = Abd_1[order_1_4];
                c1_2_4 = Abd_2[order_1_4];
            } else {
                c1_1_4 = Reg_1[order_1_4 + REG_SIZE];
                c1_2_4 = Reg_2[order_1_4 + REG_SIZE];
            }

            if (order_2_4 >= 0) {
                c2_1_4 = Abd_1[order_2_4];
                c2_2_4 = Abd_2[order_2_4];
            } else {
                c2_1_4 = Reg_1[order_2_4 + REG_SIZE];
                c2_2_4 = Reg_2[order_2_4 + REG_SIZE];
            }

            float min_1_4 = fminf(c1_1_4, c1_2_4);
            float min_2_4 = fminf(c2_1_4, c2_2_4);

            total += (min_1_4 + min_2_4);

            Reg_1[order_d_4] = (c1_1_4 - min_1_4) * dist_1_4 + (c2_1_4 - min_2_4) * dist_2_4;
            Reg_2[order_d_4] = (c1_2_4 - min_1_4) * dist_1_4 + (c2_2_4 - min_2_4) * dist_2_4;

            //6th
            if (order_1_5 >= 0) {
                c1_1_5 = Abd_1[order_1_5];
                c1_2_5 = Abd_2[order_1_5];
            } else {
                c1_1_5 = Reg_1[order_1_5 + REG_SIZE];
                c1_2_5 = Reg_2[order_1_5 + REG_SIZE];
            }

            if (order_2_5 >= 0) {
                c2_1_5 = Abd_1[order_2_5];
                c2_2_5 = Abd_2[order_2_5];
            } else {
                c2_1_5 = Reg_1[order_2_5 + REG_SIZE];
                c2_2_5 = Reg_2[order_2_5 + REG_SIZE];
            }

            float min_1_5 = fminf(c1_1_5, c1_2_5);
            float min_2_5 = fminf(c2_1_5, c2_2_5);

            total += (min_1_5 + min_2_5);

            Reg_1[order_d_5] = (c1_1_5 - min_1_5) * dist_1_5 + (c2_1_5 - min_2_5) * dist_2_5;
            Reg_2[order_d_5] = (c1_2_5 - min_1_5) * dist_1_5 + (c2_2_5 - min_2_5) * dist_2_5;

            //7th
            if (order_1_6 >= 0) {
                c1_1_6 = Abd_1[order_1_6];
                c1_2_6 = Abd_2[order_1_6];
            } else {
                c1_1_6 = Reg_1[order_1_6 + REG_SIZE];
                c1_2_6 = Reg_2[order_1_6 + REG_SIZE];
            }

            if (order_2_6 >= 0) {
                c2_1_6 = Abd_1[order_2_6];
                c2_2_6 = Abd_2[order_2_6];
            } else {
                c2_1_6 = Reg_1[order_2_6 + REG_SIZE];
                c2_2_6 = Reg_2[order_2_6 + REG_SIZE];
            }

            float min_1_6 = fminf(c1_1_6, c1_2_6);
            float min_2_6 = fminf(c2_1_6, c2_2_6);

            total += (min_1_6 + min_2_6);

            Reg_1[order_d_6] = (c1_1_6 - min_1_6) * dist_1_6 + (c2_1_6 - min_2_6) * dist_2_6;
            Reg_2[order_d_6] = (c1_2_6 - min_1_6) * dist_1_6 + (c2_2_6 - min_2_6) * dist_2_6;

            //8th
            if (order_1_7 >= 0) {
                c1_1_7 = Abd_1[order_1_7];
                c1_2_7 = Abd_2[order_1_7];
            } else {
                c1_1_7 = Reg_1[order_1_7 + REG_SIZE];
                c1_2_7 = Reg_2[order_1_7 + REG_SIZE];
            }

            if (order_2_7 >= 0) {
                c2_1_7 = Abd_1[order_2_7];
                c2_2_7 = Abd_2[order_2_7];
            } else {
                c2_1_7 = Reg_1[order_2_7 + REG_SIZE];
                c2_2_7 = Reg_2[order_2_7 + REG_SIZE];
            }

            float min_1_7 = fminf(c1_1_7, c1_2_7);
            float min_2_7 = fminf(c2_1_7, c2_2_7);

            total += (min_1_7 + min_2_7);

            Reg_1[order_d_7] = (c1_1_7 - min_1_7) * dist_1_7 + (c2_1_7 - min_2_7) * dist_2_7;
            Reg_2[order_d_7] = (c1_2_7 - min_1_7) * dist_1_7 + (c2_2_7 - min_2_7) * dist_2_7;

            //9th
            if (order_1_8 >= 0) {
                c1_1_8 = Abd_1[order_1_8];
                c1_2_8 = Abd_2[order_1_8];
            } else {
                c1_1_8 = Reg_1[order_1_8 + REG_SIZE];
                c1_2_8 = Reg_2[order_1_8 + REG_SIZE];
            }

            if (order_2_8 >= 0) {
                c2_1_8 = Abd_1[order_2_8];
                c2_2_8 = Abd_2[order_2_8];
            } else {
                c2_1_8 = Reg_1[order_2_8 + REG_SIZE];
                c2_2_8 = Reg_2[order_2_8 + REG_SIZE];
            }

            float min_1_8 = fminf(c1_1_8, c1_2_8);
            float min_2_8 = fminf(c2_1_8, c2_2_8);

            total += (min_1_8 + min_2_8);

            Reg_1[order_d_8] = (c1_1_8 - min_1_8) * dist_1_8 + (c2_1_8 - min_2_8) * dist_2_8;
            Reg_2[order_d_8] = (c1_2_8 - min_1_8) * dist_1_8 + (c2_2_8 - min_2_8) * dist_2_8;

            //10th
            if (order_1_9 >= 0) {
                c1_1_9 = Abd_1[order_1_9];
                c1_2_9 = Abd_2[order_1_9];
            } else {
                c1_1_9 = Reg_1[order_1_9 + REG_SIZE];
                c1_2_9 = Reg_2[order_1_9 + REG_SIZE];
            }

            if (order_2_9 >= 0) {
                c2_1_9 = Abd_1[order_2_9];
                c2_2_9 = Abd_2[order_2_9];
            } else {
                c2_1_9 = Reg_1[order_2_9 + REG_SIZE];
                c2_2_9 = Reg_2[order_2_9 + REG_SIZE];
            }

            float min_1_9 = fminf(c1_1_9, c1_2_9);
            float min_2_9 = fminf(c2_1_9, c2_2_9);

            total += (min_1_9 + min_2_9);

            Reg_1[order_d_9] = (c1_1_9 - min_1_9) * dist_1_9 + (c2_1_9 - min_2_9) * dist_2_9;
            Reg_2[order_d_9] = (c1_2_9 - min_1_9) * dist_1_9 + (c2_2_9 - min_2_9) * dist_2_9;
        }
        //Process elements haven't been divided
        for (; i < OrderN; i++) {
            int order_1 = Order_1[i];
            int order_2 = Order_2[i];
            int order_d = Order_d[i] + REG_SIZE;

            float dist_2 = 1 - Dist_2[i];
            float dist_1;

            if (i < sharedMemorySize) {
                dist_1 = 1 - s_dist_1[i];
            } else {
                dist_1 = 1 - Dist_1[i];
            }

            float c1_1, c1_2;
            float c2_1, c2_2;

            if (order_1 >= 0) {
                c1_1 = Abd_1[order_1];
                c1_2 = Abd_2[order_1];
            } else {
                c1_1 = Reg_1[order_1 + REG_SIZE];
                c1_2 = Reg_2[order_1 + REG_SIZE];
            }

            if (order_2 >= 0) {
                c2_1 = Abd_1[order_2];
                c2_2 = Abd_2[order_2];
            } else {
                c2_1 = Reg_1[order_2 + REG_SIZE];
                c2_2 = Reg_2[order_2 + REG_SIZE];
            }

            float min_1 = fminf(c1_1, c1_2);
            float min_2 = fminf(c2_1, c2_2);
            total += (min_1 + min_2);

            Reg_1[order_d] = (c1_1 - min_1) * dist_1 + (c2_1 - min_2) * dist_2;
            Reg_2[order_d] = (c1_2 - min_1) * dist_1 + (c2_2 - min_2) * dist_2;
        }

        total *= 0.01f;
        total = fminf(1.0f, fmaxf(0.0f, total));
        d_sim_matrix[tid] = total;

    } else {
        return;
    }
}

void calculateAbundanceSparsity(int file_count, int line, float **Abd) {
    int abd_total = 0;

    for (int i = 0; i < file_count; i++) {
        int count = 0;
        for (int j = 0; j < line; j++) {
            if (Abd[i][j] != 0.0) {
                count++;
            }
        }
        abd_total += count;
    }
    printf("\n*** total abd is %d ***\n", abd_total);

    float total_elements = static_cast<float>(file_count * line);
    float abd_avg = abd_total / total_elements;
    printf("*** average abd percentage is %.4f%% ***\n", abd_avg);

    float abd_sparsity = (1.0 - abd_avg) * 100.0;
    printf("*** abd sparsity is %.4f%% ***\n", abd_sparsity);
    printf("\n");
}

void defineOrder(int **order_row, int **order_col, int rows, int cols, long *num_elements, int flag) {
    *order_row = (int *) malloc(rows * cols * sizeof(int));
    *order_col = (int *) malloc(rows * cols * sizeof(int));

    long index = 0;
    if (1 == flag) {
        for (int i = 0; i < rows - 1; i++) {
            for (int j = i + 1; j < rows; j++) {
                (*order_row)[index] = i;
                (*order_col)[index] = j;
                index++;
            }
        }
        *num_elements = index;
    } else {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                (*order_row)[index] = i;
                (*order_col)[index] = j;
                index++;
            }
        }
        *num_elements = index;
    }
}

int showMemoryUseOfDiagonalSquare(int file_count, int orderN, double memory_ratio) {
    long long iter = (long long) file_count * (file_count - 1) / 2;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // choose default device
    size_t totalMemory = prop.totalGlobalMem;

    long long size_Abd =
            (long long) file_count * sizeof(float *) + (long long) file_count * (orderN + 1) * sizeof(float);
    size_Abd = static_cast<long long>(size_Abd * memory_ratio);
    long long size_sim_matrix = sizeof(float) * iter;
    long long size_order_m = sizeof(int) * iter;
    long long size_order_n = sizeof(int) * iter;
    long long size_Dist_1 = sizeof(float) * orderN;
    long long size_Dist_2 = sizeof(float) * orderN;
    long long size_Order_1 = sizeof(int) * orderN;
    long long size_Order_2 = sizeof(int) * orderN;
    long long size_Order_d = sizeof(int) * orderN;
    long long sum = size_Abd + size_sim_matrix + size_order_m + size_order_n + size_Dist_1 + size_Dist_2 + size_Order_1 + size_Order_2 + size_Order_d;

    long long thresholdMemory = totalMemory * 0.95;

    if (sum > thresholdMemory) {
        printf("\n*** Diagonal Square Warning: Required memory for file count %7d exceeds 95%% of total memory! ***", file_count);
        return -1;
    } else {
        printf("\n");
        printf("*** Diagonal Square ***\n");
        printf("*** Total GPU Memory size is         : %10lu KB or %9.2f MB ***\n", totalMemory / 1024, (float) totalMemory / 1048576);
        printf("*** size of Abd is                   : %10lld KB or %9.2f MB ***\n", size_Abd / 1024, (float) size_Abd / 1048576);
        printf("*** size of sim_matrix is            : %10lld KB or %9.2f MB ***\n", size_sim_matrix / 1024, (float) size_sim_matrix / 1048576);
        printf("*** size of compute order is         : %10lld KB or %9.2f MB ***\n", (size_order_m + size_order_n) / 1024,
               (float) (size_order_m + size_order_n) / 1048576);
        printf("*** size of tree dist and order is   : %10lld KB or %9.2f MB ***\n",
               (size_Dist_1 + size_Dist_2 + size_Order_1 + size_Order_2 + size_Order_d) / 1024,
               (float) (size_Dist_1 + size_Dist_2 + size_Order_1 + size_Order_2 + size_Order_d) / 1048576);
        printf("*** all allocated GPU memory size is : %10lld KB or %9.2f MB ***\n", sum / 1024, (float) sum / 1048576);
        printf("*** GPU Memory left size is about    : %10.2f MB \n", (float) (totalMemory - sum) / 1048576);
        printf("*** data split size is %d !!!\n", file_count);
        return file_count;
    }
}

int showMemoryUseOfGeneralRectangle(int file_count, int orderN, double memory_ratio) {
    long long iter = (long long) file_count * file_count;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // choose default device
    size_t totalMemory = prop.totalGlobalMem;

    long long size_Abd =
            (long long) file_count * sizeof(float *) + (long long) file_count * (orderN + 1) * sizeof(float);
    size_Abd *= 2;
    size_Abd = static_cast<long long>(size_Abd * memory_ratio);
    long long size_sim_matrix = sizeof(float) * iter;
    long long size_order_m = sizeof(int) * iter;
    long long size_order_n = sizeof(int) * iter;
    long long size_Dist_1 = sizeof(float) * orderN;
    long long size_Dist_2 = sizeof(float) * orderN;
    long long size_Order_1 = sizeof(int) * orderN;
    long long size_Order_2 = sizeof(int) * orderN;
    long long size_Order_d = sizeof(int) * orderN;
    long long sum =
            size_Abd + size_sim_matrix + size_order_m + size_order_n + size_Dist_1 + size_Dist_2 + size_Order_1 +
            size_Order_2 + size_Order_d;

    long long thresholdMemory = totalMemory * 0.95;

    if (sum > thresholdMemory) {
        printf("\n***   Normal Square Warning: Required memory for file count %7d exceeds 95%% of total memory! ***",
               file_count);
        return -1;
    } else {
        printf("\n\n");
        printf("*** Normal Square ***\n");
        printf("*** Total GPU Memory size is         : %10lu KB or %9.2f MB ***\n", totalMemory / 1024, (float) totalMemory / 1048576);
        printf("*** size of Abd is                   : %10lld KB or %9.2f MB ***\n", size_Abd / 1024, (float) size_Abd / 1048576);
        printf("*** size of sim_matrix is            : %10lld KB or %9.2f MB ***\n", size_sim_matrix / 1024, (float) size_sim_matrix / 1048576);
        printf("*** size of compute order is         : %10lld KB or %9.2f MB ***\n",
               (size_order_m + size_order_n) / 1024, (float) (size_order_m + size_order_n) / 1048576);
        printf("*** size of tree dist and order is   : %10lld KB or %9.2f MB ***\n",
               (size_Dist_1 + size_Dist_2 + size_Order_1 + size_Order_2 + size_Order_d) / 1024,
               (float) (size_Dist_1 + size_Dist_2 + size_Order_1 + size_Order_2 + size_Order_d) / 1048576);
        printf("*** all allocated GPU memory size is : %10lld KB or %9.2f MB ***\n", sum / 1024, (float) sum / 1048576);
        printf("*** GPU Memory left size is about    : %10.2f MB \n", (float) (totalMemory - sum) / 1048576);
        printf("*** data split size is %d !!!\n", file_count);
        return file_count;
    }
}

int splitSizeConfirm(int file_count, int orderN, double memory_ratio) {
    int splitFileCount = file_count;
    int result_diagonal, result_normal_square;

    //printf("memory ratio is %f\n", memory_ratio);
    result_diagonal = showMemoryUseOfDiagonalSquare(file_count, orderN, memory_ratio);
    if (result_diagonal == -1) {
        if (splitFileCount % 10000 != 0) {
            splitFileCount = (splitFileCount / 10000) * 10000;
        }
        //splitFileCount /= 2;
        while (splitFileCount > 0) {
            result_normal_square = showMemoryUseOfGeneralRectangle(splitFileCount, orderN, memory_ratio);
            if (result_normal_square == -1) {
                if (splitFileCount <= 40000) {
                    splitFileCount -= (splitFileCount > 1000) ? 1000 : splitFileCount;
                } else {
                    splitFileCount = (splitFileCount + 1) / 2;
                }
            } else {
                break;
            }
        }
        if (splitFileCount <= 0) {
            printf("GPU Memory isn't enough. Cannot proceed further.\n");
            return -1;
        }
    }
    if (splitFileCount > 40000) {
        splitFileCount = 40000;
    }
    //printf("*** final suggest split size is : %d\n\n", splitFileCount);
    return splitFileCount;
}

float
launchKernelOfDiagonalSquare(float ***d_Abd, float **d_sim_matrix, int **d_order_row, int **d_order_col, int *chunksize, int orderN,
                             float **d_Dist_1, float **d_Dist_2, int **d_Order_1, int **d_Order_2, int **d_Order_d, int blockSize, int *gridSize,
                             int sharedMemorySize, cudaStream_t *streams, int num_GPU) {
    cudaEvent_t kstart, kstop;
    float elapsedTime;
    cudaEventCreate(&kstart);
    cudaEventCreate(&kstop);
    cudaEventRecord(kstart, 0);

    printf("\n");
    for (int i = 0; i < num_GPU; i++) {
        cudaSetDevice(i);
        //kernel add offset
        CalcSimOfDiagonalSquare<<<dim3(gridSize[i]), dim3(blockSize), sharedMemorySize * 4, streams[i]>>>(d_Abd[i], d_sim_matrix[i], d_order_row[i],
                                                                                                          d_order_col[i], chunksize[i], orderN,
                                                                                                          d_Dist_1[i], d_Dist_2[i], d_Order_1[i],
                                                                                                          d_Order_2[i], d_Order_d[i],
                                                                                                          sharedMemorySize);
        printf("*** kernel %d launched with gridsize %d, blocksize %d, shared memory size %d, count %d ***\n", i + 1, gridSize[i], blockSize,
               sharedMemorySize, chunksize[i]);
    }

    cudaEventRecord(kstop, 0);
    cudaEventSynchronize(kstop);
    cudaEventElapsedTime(&elapsedTime, kstart, kstop);
    elapsedTime *= 0.001f;

    int hours = static_cast<int>(elapsedTime) / 3600;
    int minutes = (static_cast<int>(elapsedTime) % 3600) / 60;
    float seconds = elapsedTime - (hours * 3600 + minutes * 60);
    printf("\n*** kernel launch time is %.3f seconds or %d hours, %d minutes and %.3f seconds\n\n", elapsedTime, hours, minutes, seconds);
    cudaEventDestroy(kstart);
    cudaEventDestroy(kstop);

    return elapsedTime;
}

float
launchKernelOfNormalRectangle(float ***Abd_row, float ***Abd_col, float **d_sim_matrix, int **d_order_row, int **d_order_col, int *chunksize,
                              int orderN, float **d_Dist_1, float **d_Dist_2, int **d_Order_1, int **d_Order_2, int **d_Order_d, int blockSize,
                              int *gridSize, int sharedMemorySize, cudaStream_t *streams, int num_GPU) {
    cudaEvent_t kstart, kstop;
    float elapsedTime;
    cudaEventCreate(&kstart);
    cudaEventCreate(&kstop);
    cudaEventRecord(kstart, 0);

    printf("\n");

    for (int i = 0; i < num_GPU; i++) {
        cudaSetDevice(i);
        CalcSimOfNormalRectangle<<< dim3(gridSize[i]), dim3(blockSize), sharedMemorySize * 4, streams[i]>>>(Abd_row[i], Abd_col[i], d_sim_matrix[i],
                                                                                                            d_order_row[i], d_order_col[i],
                                                                                                            chunksize[i], orderN, d_Dist_1[i],
                                                                                                            d_Dist_2[i], d_Order_1[i], d_Order_2[i],
                                                                                                            d_Order_d[i], sharedMemorySize);
        printf("*** kernel %d launched with gridsize %d, blocksize %d, shared memory size %d, count %d  ***\n", i + 1, gridSize[i], blockSize,
               sharedMemorySize, chunksize[i]);
    }

    cudaEventRecord(kstop, 0);
    cudaEventSynchronize(kstop);
    cudaEventElapsedTime(&elapsedTime, kstart, kstop);
    elapsedTime *= 0.001f;

    int hours = static_cast<int>(elapsedTime) / 3600;
    int minutes = (static_cast<int>(elapsedTime) % 3600) / 60;
    float seconds = elapsedTime - (hours * 3600 + minutes * 60);
    printf("\n*** kernel launch time is %.3f seconds or %d hours, %d minutes and %.3f seconds\n\n", elapsedTime, hours, minutes, seconds);
    cudaEventDestroy(kstart);
    cudaEventDestroy(kstop);

    return elapsedTime;
}

int printhelp() {
    cout << " Welcome to Coordinated Meta Storm "<< endl;
    cout << "\tCompute the Meta-Storms distance among samples, default use rRNA copy number correction" << endl;
    cout << "Usage: " << endl;
    cout << "cuda-comp [Option] Value" << endl;
    cout << "Options: " << endl;
    cout << "\t-D (upper) ref database, " << _PMDB::Get_Args() << endl;

    cout << "\t[Input options, required]" << endl;
    //cout << "\t  -i Two samples path for single sample comparison" << endl;
    //cout << "\tor" << endl;
    //cout << "\t  -l Input files list for multi-sample comparison" << endl;
    //cout << "\t  -p List files path prefix [Optional for -l]" << endl;
    //cout << "\tor" << endl;
    cout << "\t  -T (upper) Input OTU count table (*.OTU.Count) for multi-sample comparison" << endl;
    //cout << "\t  -R If the input table is reversed, T(rue) or F(alse), default is false [Optional for -T]" << endl;

    cout << "\t[Output options]" << endl;
    cout << "\t  -o Output file, default is to output on screen" << endl;
    //cout << "\t  -d Output format, distance (T) or similarity (F), default is T" << endl;
    //cout << "\t  -P (upper) Print heatmap and clusters, T(rue) or F(alse), default is F" << endl;

    //cout << "\t[Other options]" << endl;
    //cout << "\t  -w weighted or unweighted, T(rue) or F(alse), default is T" << endl;
    //cout
            //<< "\t  -M (upper) Distance Metric, 0: Meta-Storms; 1: Meta-Storms-unweighted; 2: Cosine; 3: Euclidean; 4: Jensen-Shannon; 5: Bray-Curtis, default is 0"
            //<< endl;
    //cout << "\t  -r rRNA copy number correction, T(rue) or F(alse), default is T" << endl;
    //cout << "\t  -c Cluster number, default is 2 [Optional for -P]" << endl;
    //cout << "\t  -t Number of thread, default is auto" << endl;
    cout << "\t  -h Help" << endl;

    exit(0);
    return 0;
}

void Parse_Para(int argc, char *argv[]) {

    Ref_db = 'G';
    Coren = 0;
    Mode = 0;
    Is_cp_correct = true;
    Is_sim = false;
    Is_heatmap = false;
    Dist_metric = 0;
    int i = 1;
    if (argc == 1)
        printhelp();

    while (i < argc) {
        if (argv[i][0] != '-') {
            cerr << "Argument # " << i << " Error : Arguments must start with -" << endl;
            exit(0);
        };
        switch (argv[i][1]) {
            case 'D':
                Ref_db = argv[i + 1][0];
                break;
            case 'i':
                Queryfile1 = argv[i + 1];
                Queryfile2 = argv[i + 2];
                i++;
                Mode = 0;
                break;
            case 'l':
                Listfilename = argv[i + 1];
                Mode = 1;
                break;
            case 'p':
                Listprefix = argv[i + 1];
                break;
            case 'T':
                Tablefilename = argv[i + 1];
                Mode = 2;
                break;
            case 'o':
                Outfilename = argv[i + 1];
                break;
            case 'M':
                Dist_metric = atoi(argv[i + 1]);
                break;
            case 'r':
                if ((argv[i + 1][0] == 'f') || (argv[i + 1][0] == 'F')) Is_cp_correct = false;
                break;
            case 'd':
                if ((argv[i + 1][0] == 'f') || (argv[i + 1][0] == 'F')) Is_sim = true;
                break;
            case 'P':
                if ((argv[i + 1][0] == 't') || (argv[i + 1][0] == 'T')) Is_heatmap = true;
                break;
            case 'c':
                Cluster = atoi(argv[i + 1]);
                break;
            case 't':
                Coren = atoi(argv[i + 1]);
                break;
            case 'h':
                printhelp();
                break;
            default :
                cerr << "Error: Unrec argument " << argv[i] << endl;
                printhelp();
                break;
        }
        i += 2;
    }
    int max_core_number = sysconf(_SC_NPROCESSORS_CONF);

    if ((Coren <= 0) || (Coren > max_core_number)) {
        Coren = max_core_number;
        //printf("\nCoren:%d\n", Coren);
        Coren = 2;
    }
    if (Cluster <= 0) {
        cerr << "Warning: cluster number must be larger than 0, change to default (2)" << endl;
        //printf("Cluster:%d\n", Cluster);
    }
}

void Output_Matrix(const char *outfilename, int n, vector<float> *sim_matrix, bool is_sim, vector<string> sam_name) {

    ofstream outfile(outfilename, ofstream::out);
    if (!outfile) {
        cerr << "Error: Cannot open output file : " << outfilename << endl;
        return;
    }

    for (int i = 0; i < n; i++)
        outfile << "\t" << sam_name[i];
    outfile << endl;

    for (int i = 0; i < n; i++) {
        outfile << sam_name[i];

        for (int j = 0; j < n; j++) {
            long ii = (i <= j) ? i : j;
            long jj = (i >= j) ? i : j;
            long p = ii * (long) n + jj - (1 + ii + 1) * (ii + 1) / 2;

            if (is_sim) {
                if (ii == jj)
                    outfile << "\t" << 1.0;
                else
                    outfile << "\t" << (*sim_matrix)[p];
            } else {
                if (ii == jj)
                    outfile << "\t" << 0.0;
                else
                    outfile << "\t" << 1.0 - (*sim_matrix)[p];
            }
        }
        outfile << endl;
    }

    outfile.close();
    outfile.clear();
}

void Single_Comp() {

    _Comp_Tree comp_tree(Ref_db);

    float *Abd1 = new float[comp_tree.Get_LeafN()];
    float *Abd2 = new float[comp_tree.Get_LeafN()];

    cout << comp_tree.Load_abd(Queryfile1.c_str(), Abd1, Is_cp_correct) << " OTUs in file " << 1 << endl;
    cout << comp_tree.Load_abd(Queryfile2.c_str(), Abd2, Is_cp_correct) << " OTUs in file " << 2 << endl;

    float sim = comp_tree.Calc_sim(Abd1, Abd2, Dist_metric);

    if (Is_sim)
        cout << sim << endl;
    else
        cout << 1.0 - sim << endl;
}

void Multi_Comp() {

    _Comp_Tree comp_tree(Ref_db);
    vector<string> sam_name;
    vector<string> file_list;

    int file_count = 0;
    file_count = Load_List(Listfilename.c_str(), file_list, sam_name, Listprefix);

    float **Abd = new float *[file_count];
    for (int i = 0; i < file_count; i++) {
        Abd[i] = new float[comp_tree.Get_LeafN()];
        //cout << comp_tree.Load_abd(file_list[i].c_str(), Abd[i], Is_cp_correct) << " OTUs in file " << i + 1 << endl;
        comp_tree.Load_abd(file_list[i].c_str(), Abd[i], Is_cp_correct);
    }
    cout << file_count << " files loaded" << endl;

    vector<int> order_m;
    vector<int> order_n;
    long iter = 0;
    for (int i = 0; i < file_count - 1; i++)
        for (int j = i + 1; j < file_count; j++) {
            order_m.push_back(i);
            order_n.push_back(j);
            iter++;
        }

    vector<float> sim_matrix;
    for (long i = 0; i < iter; i++)
        sim_matrix.push_back(0);

    omp_set_num_threads(Coren);
#pragma omp parallel for schedule(dynamic, 1)
    for (long i = 0; i < iter; i++) {
        long m = order_m[i];
        long n = order_n[i];
        long p = m * (long) file_count + n - (1 + m + 1) * (m + 1) / 2;
        sim_matrix[p] = comp_tree.Calc_sim(Abd[m], Abd[n], Dist_metric);
    }

    Output_Matrix(Outfilename.c_str(), file_count, &sim_matrix, Is_sim, sam_name);
    for (int i = 0; i < file_count; i++)
        delete[] Abd[i];

//    if (Is_heatmap) {
//        char command[BUFFER_SIZE];
//        sprintf(command, "Rscript %s/Rscript/PM_Heatmap.R -d %s -o %s", Check_Env().c_str(), Outfilename.c_str(),
//                (Outfilename + ".heatmap.pdf").c_str());
//        system(command);
//        sprintf(command, "Rscript %s/Rscript/PM_Hcluster.R -d %s -o %s -c %d", Check_Env().c_str(), Outfilename.c_str(),
//                (Outfilename + ".clusters.pdf").c_str(), Cluster);
//        system(command);
//    }
}

void Multi_Comp_Table(_Table_Format abd_table) {

    _Comp_Tree comp_tree(Ref_db);
    int file_count = abd_table.Get_Sample_Size();

    //load abd
    float **Abd = new float *[file_count];
    for (int i = 0; i < file_count; i++) {
        Abd[i] = new float[comp_tree.Get_LeafN()];
        //cout << comp_tree.Load_abd(&abd_table, Abd[i], i, Is_cp_correct) << " OTUs in file " << i + 1 << endl;
        comp_tree.Load_abd(&abd_table, Abd[i], i, Is_cp_correct);
    }
    cout << file_count << " files loaded" << endl;

    //make order
    vector<int> order_m;
    vector<int> order_n;
    long iter = 0;
    for (int i = 0; i < file_count - 1; i++)
        for (int j = i + 1; j < file_count; j++) {
            order_m.push_back(i);
            order_n.push_back(j);
            iter++;
        }
    vector<float> sim_matrix;
    for (long i = 0; i < iter; i++)
        sim_matrix.push_back(0);

    omp_set_num_threads(Coren);
#pragma omp parallel for schedule(dynamic, 1)
    for (long i = 0; i < iter; i++) {

        long m = order_m[i];
        long n = order_n[i];
        long p = m * (long) file_count + n - (1 + m + 1) * (m + 1) / 2;
        sim_matrix[p] = comp_tree.Calc_sim(Abd[m], Abd[n], Dist_metric);
    }

    Output_Matrix(Outfilename.c_str(), file_count, &sim_matrix, Is_sim, abd_table.Get_Sample_Names());
    for (int i = 0; i < file_count; i++)
        delete[] Abd[i];

//    if (Is_heatmap) {
//        char command[BUFFER_SIZE];
//        sprintf(command, "Rscript %s/Rscript/PM_Heatmap.R -d %s -o %s", Check_Env().c_str(), Outfilename.c_str(),
//                (Outfilename + ".heatmap.pdf").c_str());
//        system(command);
//        sprintf(command, "Rscript %s/Rscript/PM_Hcluster.R -d %s -o %s -c %d", Check_Env().c_str(), Outfilename.c_str(),
//                (Outfilename + ".clusters.pdf").c_str(), Cluster);
//        system(command);
//    }
}

void Multi_GPU_split_MetaStorm(_Table_Format abd_table) {

    _Comp_Tree comp_tree(Ref_db);
    int file_count = abd_table.Get_Sample_Size();
    int line = comp_tree.Get_LeafN();

    int orderN = comp_tree.getOrderN();
    vector<float> Dist_1 = comp_tree.getDist_1();
    vector<float> Dist_2 = comp_tree.getDist_2();
    vector<int> Order_1 = comp_tree.getOrder_1();
    vector<int> Order_2 = comp_tree.getOrder_2();
    vector<int> Order_d = comp_tree.getOrder_d();

    long iter = (long) file_count * (file_count - 1) / 2;
    //printf("iter:%ld\n", iter);
    vector<float> sim_matrix;
    for (long i = 0; i < iter; i++)
        sim_matrix.push_back(0);
    //cout << "sim_matrix size is  " << sim_matrix.size() << "\n" << endl;

    double memory_ratio;
    switch (Ref_db) {
        case 'G':
            memory_ratio = 1.1;
            break;
        case 'R':
            memory_ratio = 1.65;
            break;
        case 'C':
            memory_ratio = 1.4;
            break;
        case 'Q':
            memory_ratio = 1.3;
            break;
        default:
            memory_ratio = 2.0;
            break;
    }

    int split_size = splitSizeConfirm(file_count, orderN, memory_ratio);

    printf("split size is : %d\n", split_size);
//	if(split_size > 20000){
//		split_size = 20000;
//	}
//	printf("modified split size is : %d\n",split_size);

    int total_blocks = ((file_count + split_size - 1) / split_size) * ((file_count + split_size - 1) / split_size);
    int block_counter = 0;
    int processed_blocks = 0;
    int skipped_blocks = 0;
    float totalKernelTime = 0.0;

    for (int block_row = 0; block_row < file_count; block_row += split_size) {
        for (int block_col = 0; block_col < file_count; block_col += split_size) {
            block_counter++;

            Block block;
            block.start_row = block_row;
            block.end_row = (block_row + split_size > file_count) ? file_count : (block_row + split_size);
            block.start_col = block_col;
            block.end_col = (block_col + split_size > file_count) ? file_count : (block_col + split_size);

            int current_block_size_row = block.end_row - block.start_row;
            int current_block_size_col = block.end_col - block.start_col;

            if (block.start_row < block.start_col && block.end_row < block.end_col) {
                skipped_blocks++;
                continue;
            } else {
                processed_blocks++;
                printf("\n***************************************************************************************************************\n");
                printf("current block : %d\n\n", block_counter);
                int *order_row, *order_col;
                long num_elements = 0;
                if (block.start_row == block.start_col && block.end_row == block.end_col) {//diagonal
                    int generateOrderFlag = 1;
                    defineOrder(&order_row, &order_col, current_block_size_row, current_block_size_col, &num_elements, generateOrderFlag);
                    printf("*** num elements : %ld\n", num_elements);

                    float **Abd = new float *[current_block_size_row];
                    for (int i = block.start_row; i < block.start_row + current_block_size_row; i++) {
                        Abd[i - block.start_row] = new float[comp_tree.Get_LeafN()];
                        //cout << comp_tree.Load_abd(&abd_table, Abd[i], i, Is_cp_correct) << " OTUs in file " << i + 1 << endl;
                        comp_tree.Load_abd(&abd_table, Abd[i - block.start_row], i, Is_cp_correct);
                    }
                    cout << "*** " << current_block_size_row << " files of diagonal square loaded" << endl;

                    //int num_GPU = 2;
                    int num_GPU;
                    cudaGetDeviceCount(&num_GPU);
                    printf("*** GPU count : %d\n", num_GPU);

                    if (file_count <= 400) {
                        num_GPU = 1;
                    }
                    //num_GPU = 2
                    printf("*** GPU count modified is : %d\n", num_GPU);

                    int *chunksize = (int *) malloc(num_GPU * sizeof(int));
                    int *chunk_start_pos = (int *) malloc(num_GPU * sizeof(int));
                    int *chunk_end_pos = (int *) malloc(num_GPU * sizeof(int));

                    int base_chunk = num_elements / num_GPU;
                    int remainder = num_elements % num_GPU;
                    //printf("base_chunk : %d\n", base_chunk);
                    //printf("remainder : %d\n", remainder);

                    for (int i = 0; i < num_GPU; i++) {
                        chunksize[i] = base_chunk;
                        if (i < remainder) {
                            chunksize[i]++;
                        }
                    }
                    long chunk_start = 0;
                    for (int i = 0; i < num_GPU; i++) {
                        chunk_start_pos[i] = chunk_start;
                        chunk_end_pos[i] = chunk_start + chunksize[i] - 1;
                        chunk_start += chunksize[i];
                    }

                    for (int i = 0; i < num_GPU; i++) {
                        printf("GPU %d chunksize = %9d, start pos = %9d, end pos = %9d\n", i, chunksize[i], chunk_start_pos[i], chunk_end_pos[i]);
                    }
                    printf("\n");

                    cudaStream_t *streams;
                    streams = new cudaStream_t[num_GPU];
                    float **d_sim_matrix = new float *[num_GPU];
                    float **d_Dist_1 = new float *[num_GPU];
                    float **d_Dist_2 = new float *[num_GPU];
                    int **d_order_row = new int *[num_GPU];
                    int **d_order_col = new int *[num_GPU];
                    int **d_Order_1 = new int *[num_GPU];
                    int **d_Order_2 = new int *[num_GPU];
                    int **d_Order_d = new int *[num_GPU];
                    float ***d_Abd = new float **[num_GPU];

                    printf("line : %d\n", line);
                    printf("orderN : %d\n", orderN);
                    printf("current_block_size_row : %d\n", current_block_size_row);
                    printf("current_block_size_col : %d\n", current_block_size_col);

                    clock_t start, end;
                    double time_used;
                    start = clock();

                    for (int i = 0; i < num_GPU; i++) {
                        cudaSetDevice(i);
                        cudaStreamCreate(&streams[i]);
                        int size = chunksize[i];
                        printf("device %d ,size : %d\n", i, size);

                        cudaMalloc((void **) &d_sim_matrix[i], sizeof(float) * size);
                        cudaMalloc((void **) &d_order_row[i], sizeof(int) * size);
                        cudaMalloc((void **) &d_order_col[i], sizeof(int) * size);
                        cudaMalloc((void **) &d_Dist_1[i], sizeof(float) * orderN);
                        cudaMalloc((void **) &d_Dist_2[i], sizeof(float) * orderN);
                        cudaMalloc((void **) &d_Order_1[i], sizeof(int) * orderN);
                        cudaMalloc((void **) &d_Order_2[i], sizeof(int) * orderN);
                        cudaMalloc((void **) &d_Order_d[i], sizeof(int) * orderN);

                        cudaMalloc((void **) &d_Abd[i], current_block_size_row * sizeof(float *));
                        for (int jj = 0; jj < current_block_size_row; jj++) {
                            float *d_row;
                            cudaError_t err = cudaMalloc((void **) &d_row, line * sizeof(float));
                            if (err != cudaSuccess) {
                                printf("cudaMalloc failed at sample : %d, error: %s\n", jj, cudaGetErrorString(err));
                                exit(EXIT_FAILURE);
                            }
                            cudaMemcpyAsync(d_row, Abd[jj], line * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
                            cudaMemcpyAsync(&d_Abd[i][jj], &d_row, sizeof(float *), cudaMemcpyHostToDevice, streams[i]);
                        }

                        int offset = chunk_start_pos[i];
                        cudaMemcpyAsync(d_order_row[i], order_row + offset, sizeof(int) * size, cudaMemcpyHostToDevice, streams[i]);
                        cudaMemcpyAsync(d_order_col[i], order_col + offset, sizeof(int) * size, cudaMemcpyHostToDevice, streams[i]);
                        cudaMemcpyAsync(d_Dist_1[i], Dist_1.data(), sizeof(float) * orderN, cudaMemcpyHostToDevice, streams[i]);
                        cudaMemcpyAsync(d_Dist_2[i], Dist_2.data(), sizeof(float) * orderN, cudaMemcpyHostToDevice, streams[i]);
                        cudaMemcpyAsync(d_Order_1[i], Order_1.data(), sizeof(int) * orderN, cudaMemcpyHostToDevice, streams[i]);
                        cudaMemcpyAsync(d_Order_2[i], Order_2.data(), sizeof(int) * orderN, cudaMemcpyHostToDevice, streams[i]);
                        cudaMemcpyAsync(d_Order_d[i], Order_d.data(), sizeof(int) * orderN, cudaMemcpyHostToDevice, streams[i]);
                    }
                    end = clock();
                    time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
                    printf("*** copy data time from host to dcu is : %f seconds ***\n", time_used);

                    for (int i = 0; i < current_block_size_row; i++)
                        delete[] Abd[i];

                    int blockSize, sharedMemorySize;
                    int *gridSize = new int[num_GPU];
                    float timeUse;
                    switch (Ref_db) {
                        case 'G':
                            printf("*** using default database GreenGenes-13-8 (16S rRNA, 97%% level)\n");
                            blockSize = 16;
                            for (int i = 0; i < num_GPU; ++i) {
                                gridSize[i] = (chunksize[i] + blockSize - 1) / blockSize;
                            }
                            sharedMemorySize = 4096;
                            timeUse = launchKernelOfDiagonalSquare(d_Abd, d_sim_matrix, d_order_row, d_order_col, chunksize, orderN, d_Dist_1,
                                                                   d_Dist_2, d_Order_1, d_Order_2, d_Order_d, blockSize, gridSize, sharedMemorySize,
                                                                   streams, num_GPU);
                            totalKernelTime += timeUse;
                            break;
                        case 'C':
                            printf("*** using reference database GreenGenes-13-8 (16S rRNA, 99%% level)\n");
                            blockSize = 8;
                            for (int i = 0; i < num_GPU; ++i) {
                                gridSize[i] = (chunksize[i] + blockSize - 1) / blockSize;
                            }
                            sharedMemorySize = 2560;
                            timeUse = launchKernelOfDiagonalSquare(d_Abd, d_sim_matrix, d_order_row, d_order_col, chunksize, orderN, d_Dist_1,
                                                                   d_Dist_2, d_Order_1, d_Order_2, d_Order_d, blockSize, gridSize, sharedMemorySize,
                                                                   streams, num_GPU);
                            totalKernelTime += timeUse;
                            break;
                        case 'R':
                            printf("*** using reference database GreenGenes-2 (16S rRNA)\n");
                            blockSize = DEFAULT_BLOCK_SIZE;
                            for (int i = 0; i < num_GPU; ++i) {
                                gridSize[i] = (chunksize[i] + blockSize - 1) / blockSize;
                            }
                            sharedMemorySize = DEFAULT_SHARED_SIZE;

                            timeUse = launchKernelOfDiagonalSquare(d_Abd, d_sim_matrix, d_order_row, d_order_col, chunksize, orderN, d_Dist_1,
                                                                   d_Dist_2, d_Order_1, d_Order_2, d_Order_d, blockSize, gridSize, sharedMemorySize,
                                                                   streams, num_GPU);
                            totalKernelTime += timeUse;
                            break;
                        case 'S':
                            printf("*** using reference database SILVA (16S rRNA, 97%% level)\n");
                            blockSize = 16;
                            for (int i = 0; i < num_GPU; ++i) {
                                gridSize[i] = (chunksize[i] + blockSize - 1) / blockSize;
                            }
                            sharedMemorySize = 3584;

                            timeUse = launchKernelOfDiagonalSquare(d_Abd, d_sim_matrix, d_order_row, d_order_col, chunksize, orderN, d_Dist_1,
                                                                   d_Dist_2, d_Order_1, d_Order_2, d_Order_d, blockSize, gridSize, sharedMemorySize,
                                                                   streams, num_GPU);
                            totalKernelTime += timeUse;
                            break;
                        case 'O':
                            printf("*** using reference database Oral_Core (16S rRNA, 97%% level)\n");
                            blockSize = DEFAULT_BLOCK_SIZE;
                            for (int i = 0; i < num_GPU; ++i) {
                                gridSize[i] = (chunksize[i] + blockSize - 1) / blockSize;
                            }
                            sharedMemorySize = DEFAULT_SHARED_SIZE;

                            timeUse = launchKernelOfDiagonalSquare(d_Abd, d_sim_matrix, d_order_row, d_order_col, chunksize, orderN, d_Dist_1,
                                                                   d_Dist_2, d_Order_1, d_Order_2, d_Order_d, blockSize, gridSize, sharedMemorySize,
                                                                   streams, num_GPU);
                            totalKernelTime += timeUse;
                            break;
                        case 'Q':
                            printf("*** using reference database Refseq (16S rRNA, 100%% level)\n");
                            blockSize = 16;
                            for (int i = 0; i < num_GPU; ++i) {
                                gridSize[i] = (chunksize[i] + blockSize - 1) / blockSize;
                            }
                            sharedMemorySize = 4096;
                            timeUse = launchKernelOfDiagonalSquare(d_Abd, d_sim_matrix, d_order_row, d_order_col, chunksize, orderN, d_Dist_1,
                                                                   d_Dist_2, d_Order_1, d_Order_2, d_Order_d, blockSize, gridSize, sharedMemorySize,
                                                                   streams, num_GPU);
                            totalKernelTime += timeUse;
                            break;
                        case 'E':
                            printf("*** using reference database SILVA (18S rRNA, 97%% level)\n");
                            blockSize = DEFAULT_BLOCK_SIZE;
                            for (int i = 0; i < num_GPU; ++i) {
                                gridSize[i] = (chunksize[i] + blockSize - 1) / blockSize;
                            }
                            sharedMemorySize = DEFAULT_SHARED_SIZE;
                            timeUse = launchKernelOfDiagonalSquare(d_Abd, d_sim_matrix, d_order_row, d_order_col, chunksize, orderN, d_Dist_1,
                                                                   d_Dist_2, d_Order_1, d_Order_2, d_Order_d, blockSize, gridSize, sharedMemorySize,
                                                                   streams, num_GPU);
                            totalKernelTime += timeUse;
                            break;
                        case 'T':
                            printf("*** using reference database ITS (ITS1, 97%% level)\n");
                            blockSize = DEFAULT_BLOCK_SIZE;
                            for (int i = 0; i < num_GPU; ++i) {
                                gridSize[i] = (chunksize[i] + blockSize - 1) / blockSize;
                            }
                            sharedMemorySize = DEFAULT_SHARED_SIZE;
                            timeUse = launchKernelOfDiagonalSquare(d_Abd, d_sim_matrix, d_order_row, d_order_col, chunksize, orderN, d_Dist_1,
                                                                   d_Dist_2, d_Order_1, d_Order_2, d_Order_d, blockSize, gridSize, sharedMemorySize,
                                                                   streams, num_GPU);
                            totalKernelTime += timeUse;
                            break;
                        default:
                            //printf("*** using default database GreenGenes-13-8 (16S rRNA, 97% level)\n");
                            blockSize = 16;
                            for (int i = 0; i < num_GPU; ++i) {
                                gridSize[i] = (chunksize[i] + blockSize - 1) / blockSize;
                            }
                            sharedMemorySize = 4096;
                            timeUse = launchKernelOfDiagonalSquare(d_Abd, d_sim_matrix, d_order_row, d_order_col, chunksize, orderN, d_Dist_1,
                                                                   d_Dist_2, d_Order_1, d_Order_2, d_Order_d, blockSize, gridSize, sharedMemorySize,
                                                                   streams, num_GPU);
                            totalKernelTime += timeUse;
                            break;
                    }

                    for (int i = 0; i < num_GPU; i++) {
                        cudaStreamSynchronize(streams[i]);
                    }

                    float *temp_result = new float[num_elements];
                    for (int i = 0; i < num_GPU; i++) {
                        cudaSetDevice(i);
                        int size = chunksize[i];
                        int offset = chunk_start_pos[i];
                        cudaMemcpyAsync(temp_result + offset, d_sim_matrix[i], sizeof(float) * size, cudaMemcpyDeviceToHost, streams[i]);
                    }
                    printf("*** copy data time from device to host finished \n");

//                    printf("\ntemp similarity matrix: \n");
//                    int tmp_print_count = num_elements < 20 ? num_elements : 20;
//                    for (int i = 0; i < tmp_print_count; i++) {
//                        printf("%f ", temp_result[i]);
//                        if ((i + 1) % 10 == 0) {
//                            printf("\n");
//                        }
//                    }
//                    printf("\n");

                    //printf("*** start to write result from temp array to similarity matrix\n");
                    clock_t f_start_ds = clock();
                    for (int i = 0; i < num_elements; i++) {
                        long m = block.start_row + order_row[i];
                        long n = block.start_col + order_col[i];
                        long p = (long) file_count * m + n - ((m + 1) * (m + 2)) / 2;
                        //printf("m:%d, n:%d, p:%d\n", m, n, p);
                        if (p > sim_matrix.size()) {
                            printf("m:%ld, n:%ld, p:%ld\n", m, n, p);
                        }
                        sim_matrix[p] = temp_result[i];
                    }
                    clock_t f_end_ds = clock();
                    double f_spent_ds = (double) (f_end_ds - f_start_ds) / CLOCKS_PER_SEC;
                    f_spent_ds = (f_spent_ds * 1000.0) / 1000.0;
                    printf("*** write result from temp array to similarity matrix finished and cost %.3f seconds \n", f_spent_ds);
                    delete[] temp_result;

                    for (int i = 0; i < num_GPU; i++) {
                        cudaSetDevice(i);
                        for (int j = 0; j < current_block_size_row; j++) {
                            float *d_row;
                            cudaMemcpy(&d_row, &d_Abd[i][j], sizeof(float *), cudaMemcpyDeviceToHost);
                            cudaFree(d_row);
                        }
                        cudaFree(d_Abd[i]);
                        cudaFree(d_sim_matrix[i]);
                        cudaFree(d_order_row[i]);
                        cudaFree(d_order_col[i]);
                        cudaFree(d_Dist_1[i]);
                        cudaFree(d_Dist_2[i]);
                        cudaFree(d_Order_1[i]);
                        cudaFree(d_Order_2[i]);
                        cudaFree(d_Order_d[i]);
                        cudaStreamDestroy(streams[i]);
                    }
                    //cout << "*** free gpu memory space finished " << endl;
                    free(chunksize);
                    free(chunk_start_pos);
                    free(chunk_end_pos);
                    delete[] streams;
                    delete[] d_sim_matrix;
                    delete[] d_Dist_1;
                    delete[] d_Dist_2;
                    delete[] d_order_row;
                    delete[] d_order_col;
                    delete[] d_Order_1;
                    delete[] d_Order_2;
                    delete[] d_Order_d;
                    delete[] d_Abd;
                    delete[] gridSize;
                    cout << "*** free gpu memory space and host temp memory space finished \n" << endl;
                } else {//general rectangle
                    int generateOrderFlag = 0;
                    defineOrder(&order_row, &order_col, current_block_size_row, current_block_size_col, &num_elements, generateOrderFlag);
                    printf("*** num elements : %ld\n", num_elements);

                    float **Abd_row = new float *[current_block_size_row];
                    for (int i = block.start_row; i < block.start_row + current_block_size_row; i++) {
                        Abd_row[i - block.start_row] = new float[comp_tree.Get_LeafN()];
                        //cout << comp_tree.Load_abd(&abd_table, Abd_row[i - block.start_row], i, Is_cp_correct) << " OTUs in file " << i + 1 << endl;
                        comp_tree.Load_abd(&abd_table, Abd_row[i - block.start_row], i, Is_cp_correct);
                    }
                    cout << "*** " << current_block_size_row << " files of normal rectangle row loaded " << endl;

                    float **Abd_col = new float *[current_block_size_col];
                    for (int i = block.start_col; i < block.start_col + current_block_size_col; i++) {
                        Abd_col[i - block.start_col] = new float[comp_tree.Get_LeafN()];
                        //cout << comp_tree.Load_abd(&abd_table, Abd_col[i - block.start_col], i, Is_cp_correct) << " OTUs in file " << i + 1 << endl;
                        comp_tree.Load_abd(&abd_table, Abd_col[i - block.start_col], i, Is_cp_correct);
                    }
                    cout << "*** " << current_block_size_col << " files of normal rectangle col loaded \n" << endl;

                    //int num_GPU = 2;
                    int num_GPU;
                    cudaGetDeviceCount(&num_GPU);
                    printf("*** GPU count : %d\n", num_GPU);

                    if (file_count <= 400) {
                        num_GPU = 1;
                    }
                    //num_GPU = 2
                    printf("*** GPU count modified is : %d\n", num_GPU);

                    int *chunksize = (int *) malloc(num_GPU * sizeof(int));
                    int *chunk_start_pos = (int *) malloc(num_GPU * sizeof(int));
                    int *chunk_end_pos = (int *) malloc(num_GPU * sizeof(int));

                    int base_chunk = num_elements / num_GPU;
                    int remainder = num_elements % num_GPU;
                    //printf("base_chunk : %d\n", base_chunk);
                    //printf("remainder : %d\n", remainder);

                    for (int i = 0; i < num_GPU; i++) {
                        chunksize[i] = base_chunk;
                        if (i < remainder) {
                            chunksize[i]++;
                        }
                    }

                    long chunk_start = 0;
                    for (int i = 0; i < num_GPU; i++) {
                        chunk_start_pos[i] = chunk_start;
                        chunk_end_pos[i] = chunk_start + chunksize[i] - 1;
                        chunk_start += chunksize[i];
                    }
                    for (int i = 0; i < num_GPU; i++) {
                        printf("GPU %d: chunksize = %9d, start pos = %9d, end pos = %9d\n", i, chunksize[i], chunk_start_pos[i], chunk_end_pos[i]);
                    }

                    cudaStream_t *streams;
                    streams = new cudaStream_t[num_GPU];
                    float **d_sim_matrix = new float *[num_GPU];
                    float **d_Dist_1 = new float *[num_GPU];
                    float **d_Dist_2 = new float *[num_GPU];
                    int **d_order_row = new int *[num_GPU];
                    int **d_order_col = new int *[num_GPU];
                    int **d_Order_1 = new int *[num_GPU];
                    int **d_Order_2 = new int *[num_GPU];
                    int **d_Order_d = new int *[num_GPU];
                    float ***d_Abd_row = new float **[num_GPU];
                    float ***d_Abd_col = new float **[num_GPU];

                    printf("line : %d\n", line);
                    printf("orderN : %d\n", orderN);
                    printf("current_block_size_row : %d\n", current_block_size_row);
                    printf("current_block_size_col : %d\n", current_block_size_col);

                    clock_t start, end;
                    double time_used;
                    start = clock();

                    for (int i = 0; i < num_GPU; i++) {
                        cudaSetDevice(i);
                        cudaStreamCreate(&streams[i]);
                        int size = chunksize[i];

                        printf("device %d ,size : %d\n", i, size);

                        cudaMalloc((void **) &d_sim_matrix[i], sizeof(float) * size);
                        cudaMalloc((void **) &d_order_row[i], sizeof(int) * size);
                        cudaMalloc((void **) &d_order_col[i], sizeof(int) * size);
                        cudaMalloc((void **) &d_Dist_1[i], sizeof(float) * orderN);
                        cudaMalloc((void **) &d_Dist_2[i], sizeof(float) * orderN);
                        cudaMalloc((void **) &d_Order_1[i], sizeof(int) * orderN);
                        cudaMalloc((void **) &d_Order_2[i], sizeof(int) * orderN);
                        cudaMalloc((void **) &d_Order_d[i], sizeof(int) * orderN);

                        cudaMalloc((void **) &d_Abd_row[i], current_block_size_row * sizeof(float *));
                        for (int j = 0; j < current_block_size_row; ++j) {
                            float *d_row_from_row;
                            cudaError_t err = cudaMalloc((void **) &d_row_from_row, line * sizeof(float));
                            if (err != cudaSuccess) {
                                printf("cudaMalloc failed at row sample : %d, error: %s\n", j, cudaGetErrorString(err));
                                exit(EXIT_FAILURE);
                            }
                            cudaMemcpyAsync(d_row_from_row, Abd_row[j], line * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
                            cudaMemcpyAsync(&d_Abd_row[i][j], &d_row_from_row, sizeof(float *), cudaMemcpyHostToDevice, streams[i]);
                        }
                        cudaMalloc((void **) &d_Abd_col[i], current_block_size_col * sizeof(float *));
                        for (int j = 0; j < current_block_size_col; ++j) {
                            float *d_row_from_col;
                            cudaError_t err = cudaMalloc((void **) &d_row_from_col, line * sizeof(float));
                            if (err != cudaSuccess) {
                                printf("cudaMalloc failed at col sample %d, error: %s\n", j, cudaGetErrorString(err));
                                exit(EXIT_FAILURE);
                            }
                            cudaMemcpy(d_row_from_col, Abd_col[j], line * sizeof(float), cudaMemcpyHostToDevice);
                            cudaMemcpy(&d_Abd_col[i][j], &d_row_from_col, sizeof(float *), cudaMemcpyHostToDevice);
                        }

                        int offset = chunk_start_pos[i];
                        cudaMemcpyAsync(d_order_row[i], order_row + offset, sizeof(int) * size, cudaMemcpyHostToDevice, streams[i]);
                        cudaMemcpyAsync(d_order_col[i], order_col + offset, sizeof(int) * size, cudaMemcpyHostToDevice, streams[i]);
                        cudaMemcpyAsync(d_Dist_1[i], Dist_1.data(), sizeof(float) * orderN, cudaMemcpyHostToDevice, streams[i]);
                        cudaMemcpyAsync(d_Dist_2[i], Dist_2.data(), sizeof(float) * orderN, cudaMemcpyHostToDevice, streams[i]);
                        cudaMemcpyAsync(d_Order_1[i], Order_1.data(), sizeof(int) * orderN, cudaMemcpyHostToDevice, streams[i]);
                        cudaMemcpyAsync(d_Order_2[i], Order_2.data(), sizeof(int) * orderN, cudaMemcpyHostToDevice, streams[i]);
                        cudaMemcpyAsync(d_Order_d[i], Order_d.data(), sizeof(int) * orderN, cudaMemcpyHostToDevice, streams[i]);
                    }
                    end = clock();
                    time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
                    printf("*** copy data time from host to dcu is : %f seconds ***\n", time_used);

                    for (int i = 0; i < current_block_size_row; i++)
                        delete[] Abd_row[i];
                    for (int i = 0; i < current_block_size_col; i++)
                        delete[] Abd_col[i];

                    int blockSize, sharedMemorySize;
                    int *gridSize = new int[num_GPU];
                    float timeUse;
                    switch (Ref_db) {
                        case 'G':
                            printf("*** using default database GreenGenes-13-8 (16S rRNA, 97%% level)\n");
                            blockSize = 16;
                            for (int i = 0; i < num_GPU; ++i) {
                                gridSize[i] = (chunksize[i] + blockSize - 1) / blockSize;
                            }
                            sharedMemorySize = 4096;
                            timeUse = launchKernelOfNormalRectangle(d_Abd_row, d_Abd_col, d_sim_matrix, d_order_row, d_order_col, chunksize, orderN,
                                                                    d_Dist_1, d_Dist_2, d_Order_1, d_Order_2, d_Order_d, blockSize, gridSize,
                                                                    sharedMemorySize, streams, num_GPU);
                            totalKernelTime += timeUse;
                            break;
                        case 'C':
                            printf("*** using reference database GreenGenes-13-8 (16S rRNA, 99%% level)\n");
                            blockSize = 8;
                            for (int i = 0; i < num_GPU; ++i) {
                                gridSize[i] = (chunksize[i] + blockSize - 1) / blockSize;
                            }
                            sharedMemorySize = 2560;
                            timeUse = launchKernelOfNormalRectangle(d_Abd_row, d_Abd_col, d_sim_matrix, d_order_row, d_order_col, chunksize, orderN,
                                                                    d_Dist_1, d_Dist_2, d_Order_1, d_Order_2, d_Order_d, blockSize, gridSize,
                                                                    sharedMemorySize, streams, num_GPU);
                            totalKernelTime += timeUse;
                            break;
                        case 'R':
                            printf("*** using reference database GreenGenes-2 (16S rRNA)\n");
                            blockSize = DEFAULT_BLOCK_SIZE;
                            for (int i = 0; i < num_GPU; ++i) {
                                gridSize[i] = (chunksize[i] + blockSize - 1) / blockSize;
                            }
                            sharedMemorySize = DEFAULT_SHARED_SIZE;

                            timeUse = launchKernelOfNormalRectangle(d_Abd_row, d_Abd_col, d_sim_matrix, d_order_row, d_order_col, chunksize, orderN,
                                                                    d_Dist_1, d_Dist_2, d_Order_1, d_Order_2, d_Order_d, blockSize, gridSize,
                                                                    sharedMemorySize, streams, num_GPU);
                            totalKernelTime += timeUse;
                            break;
                        case 'S':
                            printf("*** using reference database SILVA (16S rRNA, 97%% level)\n");
                            blockSize = 16;
                            for (int i = 0; i < num_GPU; ++i) {
                                gridSize[i] = (chunksize[i] + blockSize - 1) / blockSize;
                            }
                            sharedMemorySize = 3584;
                            timeUse = launchKernelOfNormalRectangle(d_Abd_row, d_Abd_col, d_sim_matrix, d_order_row, d_order_col, chunksize, orderN,
                                                                    d_Dist_1, d_Dist_2, d_Order_1, d_Order_2, d_Order_d, blockSize, gridSize,
                                                                    sharedMemorySize, streams, num_GPU);
                            totalKernelTime += timeUse;
                            break;
                        case 'O':
                            printf("*** using reference database Oral_Core (16S rRNA, 97%% level)\n");
                            blockSize = DEFAULT_BLOCK_SIZE;
                            for (int i = 0; i < num_GPU; ++i) {
                                gridSize[i] = (chunksize[i] + blockSize - 1) / blockSize;
                            }
                            sharedMemorySize = DEFAULT_SHARED_SIZE;
                            timeUse = launchKernelOfNormalRectangle(d_Abd_row, d_Abd_col, d_sim_matrix, d_order_row, d_order_col, chunksize, orderN,
                                                                    d_Dist_1, d_Dist_2, d_Order_1, d_Order_2, d_Order_d, blockSize, gridSize,
                                                                    sharedMemorySize, streams, num_GPU);
                            totalKernelTime += timeUse;
                            break;
                        case 'Q':
                            printf("*** using reference database Refseq (16S rRNA, 100%% level)\n");
                            blockSize = 16;
                            for (int i = 0; i < num_GPU; ++i) {
                                gridSize[i] = (chunksize[i] + blockSize - 1) / blockSize;
                            }
                            sharedMemorySize = 4096;
                            timeUse = launchKernelOfNormalRectangle(d_Abd_row, d_Abd_col, d_sim_matrix, d_order_row, d_order_col, chunksize, orderN,
                                                                    d_Dist_1, d_Dist_2, d_Order_1, d_Order_2, d_Order_d, blockSize, gridSize,
                                                                    sharedMemorySize, streams, num_GPU);
                            totalKernelTime += timeUse;
                            break;
                        case 'E':
                            printf("*** using reference database SILVA (18S rRNA, 97%% level)\n");
                            blockSize = DEFAULT_BLOCK_SIZE;
                            for (int i = 0; i < num_GPU; ++i) {
                                gridSize[i] = (chunksize[i] + blockSize - 1) / blockSize;
                            }
                            sharedMemorySize = DEFAULT_SHARED_SIZE;
                            timeUse = launchKernelOfNormalRectangle(d_Abd_row, d_Abd_col, d_sim_matrix, d_order_row, d_order_col, chunksize, orderN,
                                                                    d_Dist_1, d_Dist_2, d_Order_1, d_Order_2, d_Order_d, blockSize, gridSize,
                                                                    sharedMemorySize, streams, num_GPU);
                            totalKernelTime += timeUse;
                            break;
                        case 'T':
                            printf("*** using reference database ITS (ITS1, 97%% level)\n");
                            blockSize = DEFAULT_BLOCK_SIZE;
                            for (int i = 0; i < num_GPU; ++i) {
                                gridSize[i] = (chunksize[i] + blockSize - 1) / blockSize;
                            }
                            sharedMemorySize = DEFAULT_SHARED_SIZE;
                            timeUse = launchKernelOfNormalRectangle(d_Abd_row, d_Abd_col, d_sim_matrix, d_order_row, d_order_col, chunksize, orderN,
                                                                    d_Dist_1, d_Dist_2, d_Order_1, d_Order_2, d_Order_d, blockSize, gridSize,
                                                                    sharedMemorySize, streams, num_GPU);
                            totalKernelTime += timeUse;
                            break;
                        default:
                            //printf("*** using default database GreenGenes-13-8 (16S rRNA, 97% level)\n");
                            blockSize = 16;
                            for (int i = 0; i < num_GPU; ++i) {
                                gridSize[i] = (chunksize[i] + blockSize - 1) / blockSize;
                            }
                            sharedMemorySize = 4096;
                            timeUse = launchKernelOfNormalRectangle(d_Abd_row, d_Abd_col, d_sim_matrix, d_order_row, d_order_col, chunksize, orderN,
                                                                    d_Dist_1, d_Dist_2, d_Order_1, d_Order_2, d_Order_d, blockSize, gridSize,
                                                                    sharedMemorySize, streams, num_GPU);
                            totalKernelTime += timeUse;
                            break;
                    }

                    for (int i = 0; i < num_GPU; i++) {
                        cudaStreamSynchronize(streams[i]);
                    }

                    float *temp_result = new float[num_elements];
                    for (int i = 0; i < num_GPU; i++) {
                        cudaSetDevice(i);
                        int size = chunksize[i];
                        int offset = chunk_start_pos[i];
                        cudaMemcpyAsync(temp_result + offset, d_sim_matrix[i], sizeof(float) * size, cudaMemcpyDeviceToHost, streams[i]);
                    }
                    printf("*** copy data time from device to host finished \n");

//                    printf("\ntemp similarity matrix: \n");
//                    int tmp_print_count = num_elements < 20 ? num_elements : 20;
//                    for (int i = 0; i < tmp_print_count; i++) {
//                        printf("%f ", temp_result[i]);
//                        if ((i + 1) % 10 == 0) {
//                            printf("\n");
//                        }
//                    }
//                    printf("\n");
                    //printf("*** start to write result from temp array to similarity matrix\n");
                    clock_t f_start_gr = clock();
                    for (int i = 0; i < num_elements; i++) {
                        //special situation
                        long n = block.start_row + order_row[i];
                        long m = block.start_col + order_col[i];
                        long p = (long) file_count * m + n - ((m + 1) * (m + 2)) / 2;
                        //printf("m:%d, n:%d, p:%d\n", m, n, p);
                        if (p > sim_matrix.size()) {
                            printf("m:%ld, n:%ld, p:%ld\n", m, n, p);
                        }
                        sim_matrix[p] = temp_result[i];
                    }
                    clock_t f_end_gr = clock();
                    double f_spent_gr = (double) (f_end_gr - f_start_gr) / CLOCKS_PER_SEC;
                    f_spent_gr = (f_spent_gr * 1000.0) / 1000.0;
                    printf("*** write result from temp array to similarity matrix finished and cost %.3f seconds \n", f_spent_gr);

                    delete[] temp_result;
                    for (int i = 0; i < num_GPU; i++) {
                        cudaSetDevice(i);
                        for (int j = 0; j < current_block_size_row; j++) {
                            float *d_row_from_row;
                            cudaMemcpy(&d_row_from_row, &d_Abd_row[i][j], sizeof(float *), cudaMemcpyDeviceToHost);
                            cudaFree(d_row_from_row);
                        }
                        cudaFree(d_Abd_row);
                        for (int j = 0; j < current_block_size_col; j++) {
                            float *d_row_from_col;
                            cudaMemcpy(&d_row_from_col, &d_Abd_col[i][j], sizeof(float *), cudaMemcpyDeviceToHost);
                            cudaFree(d_row_from_col);
                        }
                        cudaFree(d_Abd_col);

                        cudaFree(d_sim_matrix[i]);
                        cudaFree(d_order_row[i]);
                        cudaFree(d_order_col[i]);
                        cudaFree(d_Dist_1[i]);
                        cudaFree(d_Dist_2[i]);
                        cudaFree(d_Order_1[i]);
                        cudaFree(d_Order_2[i]);
                        cudaFree(d_Order_d[i]);
                        cudaStreamDestroy(streams[i]);
                    }
                    //cout << "*** free gpu memory space finished " << endl;
                    free(chunksize);
                    free(chunk_start_pos);
                    free(chunk_end_pos);
                    delete[] streams;
                    delete[] d_sim_matrix;
                    delete[] d_Dist_1;
                    delete[] d_Dist_2;
                    delete[] d_order_row;
                    delete[] d_order_col;
                    delete[] d_Order_1;
                    delete[] d_Order_2;
                    delete[] d_Order_d;
                    delete[] d_Abd_row;
                    delete[] d_Abd_col;
                    delete[] gridSize;
                    cout << "*** free gpu memory space and host temp memory space finished \n" << endl;
                }
                free(order_row);
                free(order_col);
                printf("*** Processed block %3d of %3d: rows [%5d, %5d), cols [%5d, %5d), size: %5d x %5d, count:%10ld\n", block_counter,
                       total_blocks, block.start_row, block.end_row, block.start_col, block.end_col, current_block_size_row, current_block_size_col,
                       num_elements);
                printf("\n***************************************************************************************************************\n");
            }
        }
    }
    printf("\n");
    printf("*** Total blocks processed: %d\n", processed_blocks);
    printf("*** Total blocks skipped: %d\n", skipped_blocks);

    int totalHours = static_cast<int>(totalKernelTime) / 3600;
    int totalMinutes = (static_cast<int>(totalKernelTime) % 3600) / 60;
    float totalSeconds = totalKernelTime - (totalHours * 3600 + totalMinutes * 60);
    printf("*** all kernel launch time is : %.2f seconds or %d hours %d minutes and %.2f seconds \n", totalKernelTime, totalHours, totalMinutes,
           totalSeconds);

    printf("\nsimilarity matrix: \n");
    size_t size = sim_matrix.size();
    size_t print_count = size < 50 ? size : 50;
    for (size_t i = 0; i < print_count; i++) {
        printf("%f ", sim_matrix[i]);
        if ((i + 1) % 10 == 0) {
            printf("\n");
        }
    }
    printf("\n");

    clock_t timeStart = clock();
    Output_Matrix(Outfilename.c_str(), file_count, &sim_matrix, Is_sim, abd_table.Get_Sample_Names());
    clock_t timeEnd = clock();
    double time_spent = (double) (timeEnd - timeStart) / CLOCKS_PER_SEC;
    time_spent = (time_spent * 1000.0) / 1000.0;
    printf("*** write result into file took %.3f seconds to execute \n", time_spent);

//    if (Is_heatmap) {
//        char command[BUFFER_SIZE];
//        sprintf(command, "Rscript %s/Rscript/PM_Heatmap.R -d %s -o %s", Check_Env().c_str(), Outfilename.c_str(),
//                (Outfilename + ".heatmap.pdf").c_str());
//        system(command);
//
//        sprintf(command, "Rscript %s/Rscript/PM_Hcluster.R -d %s -o %s -c %d", Check_Env().c_str(), Outfilename.c_str(),
//                (Outfilename + ".clusters.pdf").c_str(), Cluster);
//        system(command);
//    }
}

int main(int argc, char *argv[]) {
    Parse_Para(argc, argv);
    switch (Mode) {
        case 0:
            Single_Comp();
            break;
        case 1:
            Multi_Comp();
            break;
        case 2: {
            //_Table_Format table(Tablefilename.c_str(), Reversed_table);
            _Table_Format table(Tablefilename.c_str());
            //printf("Dist_metric:%d\n", Dist_metric);
            if (0 != Dist_metric) {
                Multi_Comp_Table(table);
            } else {
                Multi_GPU_split_MetaStorm(table);
            }
            break;
        }
        default:
            break;
    }
    return 0;
}
