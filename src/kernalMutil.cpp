#define REG_SIZE 150
#include <hip/hip_runtime.h>
#include <cstddef>

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
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0); // choose default device
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
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0); // choose default device
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

    printf("memory ratio is %f\n", memory_ratio);
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
