#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>
#include <string>
#include <algorithm>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include "../../Utils/helper_functions.h"

const char* main_function = "main";
const char* data_init = "data_init";
const char* comm = "comm";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* correctness_check = "correctness_check";

int compare (const void * a, const void * b)
{
    float fa = *(const float*) a;
    float fb = *(const float*) b;
    return (fa > fb) - (fa < fb);
}

void print_array(float* array, int size) {
    for (int i = 0; i < size; i++){
        printf("%0.3f,", array[i]);
    }
    printf("\n");
}

void print_iarray(int* array, int size) {
    for (int i = 0; i < size; i++){
        printf("%i,", array[i]);
    }
    printf("\n");
}

void bubbleSort(float *values, int local_data_size, int numTasks, int rankid) {
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);

    auto *temp = new float[local_data_size];
    auto *merged = new float[local_data_size *  2];

    for (int phase = 0; phase < numTasks; phase++) {
        int neighbor;
        if (phase % 2 == 0) { // Even phase
            neighbor = (rankid % 2 == 0) ? rankid + 1 : rankid - 1;
        } else { // Odd phase
            neighbor = (rankid % 2 != 0) ? rankid + 1 : rankid - 1;
        }

        CALI_MARK_BEGIN(comm);
    	CALI_MARK_BEGIN(comm_large);

        // Avoid out-of-bounds ranks
        if (neighbor >= 0 && neighbor < numTasks) {
            // Send and receive values
            if (rankid % 2 == 0) {
                MPI_Send(values, local_data_size, MPI_INT, neighbor, 0, MPI_COMM_WORLD);
                MPI_Recv(temp, local_data_size, MPI_INT, neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(temp, local_data_size, MPI_INT, neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(values, local_data_size, MPI_INT, neighbor, 0, MPI_COMM_WORLD);
            }


            CALI_MARK_END(comm_large);
    	    CALI_MARK_END(comm);

            qsort(merged, (local_data_size *  2), sizeof(int), compare);

            auto midPoint = merged + local_data_size;
            if (rankid < neighbor)
                std::copy(merged, midPoint, values);
            else
                std::copy(midPoint, merged + (local_data_size *  2), values);
        }
    }
    
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);
}

int main(int argc, char** argv) {
    CALI_CXX_MARK_FUNCTION;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <num_values> <num_processes>\n", argv[0]);
        exit(1);
    }
    int data_size = atoi(argv[1]);

    int	numTasks,
        rankid,
        rc;

    float *global_array = (float*)malloc(data_size * sizeof(float));

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rankid);
    MPI_Comm_size(MPI_COMM_WORLD,&numTasks);

    if (numTasks < 2 ) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }

    CALI_MARK_BEGIN(main_function);
    
    int local_data_size = data_size / numTasks;
    float *values = (float*)malloc(local_data_size * sizeof(float));

    CALI_MARK_BEGIN(data_init);
    array_fill_random_no_seed(values, local_data_size);
    CALI_MARK_END(data_init);

    // localBubbleSort(values, local_data_size);

    bubbleSort(values, local_data_size, numTasks, rankid);

    MPI_Gather(values, local_data_size, MPI_FLOAT, global_array, local_data_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // print_array(global_array, data_size);

    if (rankid == 0) {
        bool correct = check_sorted(global_array, data_size);
        if (correct) {
            printf("Array was sorted correctly!\n");
        } else {
            printf("Array was incorrectly sorted!\n");
        }
    }

    free(values);
    free(global_array);

    CALI_MARK_END(main_function);

    MPI_Finalize();

    return 0;
}