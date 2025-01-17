#include <iostream>
#include <fstream>
#include <sstream>
#include <cstddef>

#include <sys/stat.h>
#include <unistd.h>
#include <omp.h>
#include <string.h>
#include <hip/hip_runtime.h>
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

extern __global__ void
CalcSimOfDiagonalSquare(float **d_Abd, float *d_sim_matrix, int *order_m, int *order_n, int num_elements, int OrderN, float *Dist_1, float *Dist_2,
                        int *Order_1, int *Order_2, int *Order_d, int sharedMemorySize);

extern __global__ void
CalcSimOfNormalRectangle(float **Abd_row, float **Abd_col, float *d_sim_matrix, int *d_order_row, int *d_order_col, int num_elements, int OrderN,
                         float *Dist_1, float *Dist_2, int *Order_1, int *Order_2, int *Order_d, int sharedMemorySize);

extern void calculateAbundanceSparsity(int file_count, int line, float **Abd);

extern void defineOrder(int **order_row, int **order_col, int rows, int cols, long *num_elements, int flag);

extern int splitSizeConfirm(int file_count, int orderN, double memory_ratio);


float
launchKernelOfDiagonalSquare(float ***d_Abd, float **d_sim_matrix, int **d_order_row, int **d_order_col, int *chunksize, int orderN,
                             float **d_Dist_1, float **d_Dist_2, int **d_Order_1, int **d_Order_2, int **d_Order_d, int blockSize, int *gridSize,
                             int sharedMemorySize, hipStream_t *streams, int num_GPU) {
    hipEvent_t kstart, kstop;
    float elapsedTime;
    hipEventCreate(&kstart);
    hipEventCreate(&kstop);
    hipEventRecord(kstart, 0);

    printf("\n");
    for (int i = 0; i < num_GPU; i++) {
        hipSetDevice(i);
        //kernel add offset
        hipLaunchKernelGGL(CalcSimOfDiagonalSquare, dim3(gridSize[i]), dim3(blockSize), sharedMemorySize * 4, streams[i], d_Abd[i], d_sim_matrix[i],
                           d_order_row[i], d_order_col[i], chunksize[i], orderN, d_Dist_1[i], d_Dist_2[i], d_Order_1[i], d_Order_2[i], d_Order_d[i],
                           sharedMemorySize);
        printf("*** kernel %d launched with gridsize %d, blocksize %d, shared memory size %d, count %d ***\n", i + 1, gridSize[i], blockSize,
               sharedMemorySize, chunksize[i]);
    }

    hipEventRecord(kstop, 0);
    hipEventSynchronize(kstop);
    hipEventElapsedTime(&elapsedTime, kstart, kstop);
    elapsedTime *= 0.001f;

    int hours = static_cast<int>(elapsedTime) / 3600;
    int minutes = (static_cast<int>(elapsedTime) % 3600) / 60;
    float seconds = elapsedTime - (hours * 3600 + minutes * 60);
    printf("\n*** kernel launch time is %.3f seconds or %d hours, %d minutes and %.3f seconds\n\n", elapsedTime, hours, minutes, seconds);
    hipEventDestroy(kstart);
    hipEventDestroy(kstop);

    return elapsedTime;
}

float
launchKernelOfNormalRectangle(float ***Abd_row, float ***Abd_col, float **d_sim_matrix, int **d_order_row, int **d_order_col, int *chunksize,
                              int orderN, float **d_Dist_1, float **d_Dist_2, int **d_Order_1, int **d_Order_2, int **d_Order_d, int blockSize,
                              int *gridSize, int sharedMemorySize, hipStream_t *streams, int num_GPU) {
    hipEvent_t kstart, kstop;
    float elapsedTime;
    hipEventCreate(&kstart);
    hipEventCreate(&kstop);
    hipEventRecord(kstart, 0);

    printf("\n");

    for (int i = 0; i < num_GPU; i++) {
        hipSetDevice(i);
        hipLaunchKernelGGL(CalcSimOfNormalRectangle, dim3(gridSize[i]), dim3(blockSize), sharedMemorySize * 4, streams[i], Abd_row[i], Abd_col[i],
                           d_sim_matrix[i], d_order_row[i], d_order_col[i], chunksize[i], orderN, d_Dist_1[i], d_Dist_2[i], d_Order_1[i],
                           d_Order_2[i], d_Order_d[i], sharedMemorySize);
        printf("*** kernel %d launched with gridsize %d, blocksize %d, shared memory size %d, count %d  ***\n", i + 1, gridSize[i], blockSize,
               sharedMemorySize, chunksize[i]);
    }

    hipEventRecord(kstop, 0);
    hipEventSynchronize(kstop);
    hipEventElapsedTime(&elapsedTime, kstart, kstop);
    elapsedTime *= 0.001f;

    int hours = static_cast<int>(elapsedTime) / 3600;
    int minutes = (static_cast<int>(elapsedTime) % 3600) / 60;
    float seconds = elapsedTime - (hours * 3600 + minutes * 60);
    printf("\n*** kernel launch time is %.3f seconds or %d hours, %d minutes and %.3f seconds\n\n", elapsedTime, hours, minutes, seconds);
    hipEventDestroy(kstart);
    hipEventDestroy(kstop);

    return elapsedTime;
}

int printhelp() {
    cout << "CMS version : " << Version << endl;
    cout << "\tCompute the Meta-Storms distance among samples, default use rRNA copy number correction" << endl;
    cout << "Usage: " << endl;
    cout << "hip-comp [Option] Value" << endl;
    cout << "Options: " << endl;
    cout << "\t-D (upper) ref database, " << _PMDB::Get_Args() << endl;

    c//out << "\t[Input options, required]" << endl;
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
                    hipGetDeviceCount(&num_GPU);
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

                    hipStream_t *streams;
                    streams = new hipStream_t[num_GPU];
                    float **d_sim_matrix = new float *[num_GPU];
                    float **d_Dist_1 = new float *[num_GPU];
                    float **d_Dist_2 = new float *[num_GPU];
                    int **d_order_row = new int *[num_GPU];
                    int **d_order_col = new int *[num_GPU];
                    int **d_Order_1 = new int *[num_GPU];
                    int **d_Order_2 = new int *[num_GPU];
                    int **d_Order_d = new int *[num_GPU];
                    float ***d_Abd = new float **[num_GPU];

//                    printf("line : %d\n", line);
//                    printf("orderN : %d\n", orderN);
//                    printf("current_block_size_row : %d\n", current_block_size_row);
//                    printf("current_block_size_col : %d\n", current_block_size_col);

                    clock_t start, end;
                    double time_used;
                    start = clock();

                    for (int i = 0; i < num_GPU; i++) {
                        hipSetDevice(i);
                        hipStreamCreate(&streams[i]);
                        int size = chunksize[i];
                        //printf("device %d ,size : %d\n", i, size);

                        hipMalloc((void **) &d_sim_matrix[i], sizeof(float) * size);
                        hipMalloc((void **) &d_order_row[i], sizeof(int) * size);
                        hipMalloc((void **) &d_order_col[i], sizeof(int) * size);
                        hipMalloc((void **) &d_Dist_1[i], sizeof(float) * orderN);
                        hipMalloc((void **) &d_Dist_2[i], sizeof(float) * orderN);
                        hipMalloc((void **) &d_Order_1[i], sizeof(int) * orderN);
                        hipMalloc((void **) &d_Order_2[i], sizeof(int) * orderN);
                        hipMalloc((void **) &d_Order_d[i], sizeof(int) * orderN);

                        hipMalloc((void **) &d_Abd[i], current_block_size_row * sizeof(float *));
                        for (int jj = 0; jj < current_block_size_row; jj++) {
                            float *d_row;
                            hipError_t err = hipMalloc((void **) &d_row, line * sizeof(float));
                            if (err != hipSuccess) {
                                printf("hipMalloc failed at sample : %d, error: %s\n", jj, hipGetErrorString(err));
                                exit(EXIT_FAILURE);
                            }
                            hipMemcpyAsync(d_row, Abd[jj], line * sizeof(float), hipMemcpyHostToDevice, streams[i]);
                            hipMemcpyAsync(&d_Abd[i][jj], &d_row, sizeof(float *), hipMemcpyHostToDevice, streams[i]);
                        }

                        int offset = chunk_start_pos[i];
                        hipMemcpyAsync(d_order_row[i], order_row + offset, sizeof(int) * size, hipMemcpyHostToDevice, streams[i]);
                        hipMemcpyAsync(d_order_col[i], order_col + offset, sizeof(int) * size, hipMemcpyHostToDevice, streams[i]);
                        hipMemcpyAsync(d_Dist_1[i], Dist_1.data(), sizeof(float) * orderN, hipMemcpyHostToDevice, streams[i]);
                        hipMemcpyAsync(d_Dist_2[i], Dist_2.data(), sizeof(float) * orderN, hipMemcpyHostToDevice, streams[i]);
                        hipMemcpyAsync(d_Order_1[i], Order_1.data(), sizeof(int) * orderN, hipMemcpyHostToDevice, streams[i]);
                        hipMemcpyAsync(d_Order_2[i], Order_2.data(), sizeof(int) * orderN, hipMemcpyHostToDevice, streams[i]);
                        hipMemcpyAsync(d_Order_d[i], Order_d.data(), sizeof(int) * orderN, hipMemcpyHostToDevice, streams[i]);
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
                        hipStreamSynchronize(streams[i]);
                    }

                    float *temp_result = new float[num_elements];
                    for (int i = 0; i < num_GPU; i++) {
                        hipSetDevice(i);
                        int size = chunksize[i];
                        int offset = chunk_start_pos[i];
                        hipMemcpyAsync(temp_result + offset, d_sim_matrix[i], sizeof(float) * size, hipMemcpyDeviceToHost, streams[i]);
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
                        hipSetDevice(i);
                        for (int j = 0; j < current_block_size_row; j++) {
                            float *d_row = d_Abd[i][j];
                            hipFree(d_row);
                        }
                        hipFree(d_Abd[i]);
                        hipFree(d_sim_matrix[i]);
                        hipFree(d_order_row[i]);
                        hipFree(d_order_col[i]);
                        hipFree(d_Dist_1[i]);
                        hipFree(d_Dist_2[i]);
                        hipFree(d_Order_1[i]);
                        hipFree(d_Order_2[i]);
                        hipFree(d_Order_d[i]);
                        hipStreamDestroy(streams[i]);
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
                    hipGetDeviceCount(&num_GPU);
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

                    hipStream_t *streams;
                    streams = new hipStream_t[num_GPU];
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

//                    printf("line : %d\n", line);
//                    printf("orderN : %d\n", orderN);
//                    printf("current_block_size_row : %d\n", current_block_size_row);
//                    printf("current_block_size_col : %d\n", current_block_size_col);

                    clock_t start, end;
                    double time_used;
                    start = clock();

                    for (int i = 0; i < num_GPU; i++) {
                        hipSetDevice(i);
                        hipStreamCreate(&streams[i]);
                        int size = chunksize[i];

                        //printf("device %d ,size : %d\n", i, size);

                        hipMalloc((void **) &d_sim_matrix[i], sizeof(float) * size);
                        hipMalloc((void **) &d_order_row[i], sizeof(int) * size);
                        hipMalloc((void **) &d_order_col[i], sizeof(int) * size);
                        hipMalloc((void **) &d_Dist_1[i], sizeof(float) * orderN);
                        hipMalloc((void **) &d_Dist_2[i], sizeof(float) * orderN);
                        hipMalloc((void **) &d_Order_1[i], sizeof(int) * orderN);
                        hipMalloc((void **) &d_Order_2[i], sizeof(int) * orderN);
                        hipMalloc((void **) &d_Order_d[i], sizeof(int) * orderN);

                        hipMalloc((void **) &d_Abd_row[i], current_block_size_row * sizeof(float *));
                        for (int j = 0; j < current_block_size_row; ++j) {
                            float *d_row_from_row;
                            hipError_t err = hipMalloc((void **) &d_row_from_row, line * sizeof(float));
                            if (err != hipSuccess) {
                                printf("hipMalloc failed at row sample : %d, error: %s\n", j, hipGetErrorString(err));
                                exit(EXIT_FAILURE);
                            }
                            hipMemcpyAsync(d_row_from_row, Abd_row[j], line * sizeof(float), hipMemcpyHostToDevice, streams[i]);
                            hipMemcpyAsync(&d_Abd_row[i][j], &d_row_from_row, sizeof(float *), hipMemcpyHostToDevice, streams[i]);
                        }
                        hipMalloc((void **) &d_Abd_col[i], current_block_size_col * sizeof(float *));
                        for (int j = 0; j < current_block_size_col; ++j) {
                            float *d_row_from_col;
                            hipError_t err = hipMalloc((void **) &d_row_from_col, line * sizeof(float));
                            if (err != hipSuccess) {
                                printf("hipMalloc failed at col sample %d, error: %s\n", j, hipGetErrorString(err));
                                exit(EXIT_FAILURE);
                            }
                            hipMemcpy(d_row_from_col, Abd_col[j], line * sizeof(float), hipMemcpyHostToDevice);
                            hipMemcpy(&d_Abd_col[i][j], &d_row_from_col, sizeof(float *), hipMemcpyHostToDevice);
                        }

                        int offset = chunk_start_pos[i];
                        hipMemcpyAsync(d_order_row[i], order_row + offset, sizeof(int) * size, hipMemcpyHostToDevice, streams[i]);
                        hipMemcpyAsync(d_order_col[i], order_col + offset, sizeof(int) * size, hipMemcpyHostToDevice, streams[i]);
                        hipMemcpyAsync(d_Dist_1[i], Dist_1.data(), sizeof(float) * orderN, hipMemcpyHostToDevice, streams[i]);
                        hipMemcpyAsync(d_Dist_2[i], Dist_2.data(), sizeof(float) * orderN, hipMemcpyHostToDevice, streams[i]);
                        hipMemcpyAsync(d_Order_1[i], Order_1.data(), sizeof(int) * orderN, hipMemcpyHostToDevice, streams[i]);
                        hipMemcpyAsync(d_Order_2[i], Order_2.data(), sizeof(int) * orderN, hipMemcpyHostToDevice, streams[i]);
                        hipMemcpyAsync(d_Order_d[i], Order_d.data(), sizeof(int) * orderN, hipMemcpyHostToDevice, streams[i]);
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
                        hipStreamSynchronize(streams[i]);
                    }

                    float *temp_result = new float[num_elements];
                    for (int i = 0; i < num_GPU; i++) {
                        hipSetDevice(i);
                        int size = chunksize[i];
                        int offset = chunk_start_pos[i];
                        hipMemcpyAsync(temp_result + offset, d_sim_matrix[i], sizeof(float) * size, hipMemcpyDeviceToHost, streams[i]);
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
                        hipSetDevice(i);
                        for (int j = 0; j < current_block_size_row; j++) {
                            float *d_row_from_row = d_Abd_row[i][j];
                            hipFree(d_row_from_row);
                        }
                        hipFree(d_Abd_row);
                        for (int j = 0; j < current_block_size_col; j++) {
                            float *d_row_from_col = d_Abd_col[i][j];
                            hipFree(d_row_from_col);
                        }
                        hipFree(d_Abd_col);

                        hipFree(d_sim_matrix[i]);
                        hipFree(d_order_row[i]);
                        hipFree(d_order_col[i]);
                        hipFree(d_Dist_1[i]);
                        hipFree(d_Dist_2[i]);
                        hipFree(d_Order_1[i]);
                        hipFree(d_Order_2[i]);
                        hipFree(d_Order_d[i]);
                        hipStreamDestroy(streams[i]);
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
