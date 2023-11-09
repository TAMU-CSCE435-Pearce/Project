#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

void merge(int *, int *, int, int, int);
void mergeSort(int *, int *, int, int);

void merge(int *, int *, int, int, int);
void mergeSort(int *, int *, int, int);
const char *main = "main";
const char *data_init = "data_init";
const char *correctness_check = "correctness_check";
const char *comm = "comm";
const char *MPI_Barrier = "MPI_Barrier";
const char *comm_large = "comm_large";
const char *MPI_Gather = "MPI_Gather";
const char *MPI_Scatter = "MPI_Scatter";
const char *comp = "comp";
const char *comp_large = "comp_large";
const char *comp_final = "comp_final";

int main(int argc, char **argv)
{
    CALI_MARK_BEGIN(main);
    /********** Create and populate the array **********/
    int n = atoi(argv[1]);
    int *original_array = malloc(n * sizeof(int));

    CALI_MARK_BEGIN(data_init);
    srand(time(NULL));
    for (int c = 0; c < n; c++)
    {

        original_array[c] = rand() % n;
    }
    CALI_MARK_END(data_init);

    /********** Initialize MPI **********/
    int world_rank;
    int world_size;

    MPI_INIT(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /********** Divide the array in equal-sized chunks **********/
    int size = n / world_size;

    cali::ConfigManager mgr;
    mgr.start();
    /********** Send each subarray to each process **********/
    int *sub_array = malloc(size * sizeof(int));
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(MPI_Scatter);
    MPI_Scatter(original_array, size, MPI_INT, sub_array, size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(MPI_Scatter);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
    /********** Perform the mergesort on each process **********/
    int *tmp_array = malloc(size * sizeof(int));
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    mergeSort(sub_array, tmp_array, 0, (size - 1));
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);
    /********** Gather the sorted subarrays into one **********/
    int *sorted = NULL;
    if (world_rank == 0)
    {

        sorted = malloc(n * sizeof(int));
    }
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(MPI_Gather);
    MPI_Gather(sub_array, size, MPI_INT, sorted, size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(MPI_Gather);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
    /********** Make the final mergeSort call **********/
    if (world_rank == 0)
    {

        int *other_array = malloc(n * sizeof(int));
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_final);
        mergeSort(sorted, other_array, 0, (n - 1));
        CALI_MARK_END(comp_final);
        CALI_MARK_END(comp);

        /********** Check sorted array **********/
        CALI_MARK_BEGIN(correctness_check);
        for (c = 0; c < n - 1; c++)
        {
            if (sorted[c] > sorted[c + 1])
            {
                printf("Sort Failed\n");
                break;
            }
        }
        CALI_MARK_END(correctness_check);

        printf("\n");
        printf("Sort Successful\n");

        /********** Clean up root **********/
        free(sorted);
        free(other_array);
    }

    /********** Clean up rest **********/
    free(original_array);
    free(sub_array);
    free(tmp_array);

    /********** Finalize MPI **********/
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(MPI_Barrier);
    MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END(MPI_Barrier);
    CALI_MARK_END(comm);
    adiak::init(NULL);
    adiak::launchdate();                             // launch date of the job
    adiak::libraries();                              // Libraries used
    adiak::cmdline();                                // Command line used to launch the job
    adiak::clustername();                            // Name of the cluster
    adiak::value("Algorithm", "Mergesort");          // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI");         // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int");                 // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int));     // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", n);                    // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random");             // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", world_size);           // The number of processors (MPI ranks)
    adiak::value("group_num", 18);                   // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
    CALI_MARK_END(main);
    mgr.stop();
    mgr.flush();
    MPI_Finalize();
}

/********** Merge Function **********/
void merge(int *a, int *b, int l, int m, int r)
{

    int h, i, j, k;
    h = l;
    i = l;
    j = m + 1;

    while ((h <= m) && (j <= r))
    {

        if (a[h] <= a[j])
        {

            b[i] = a[h];
            h++;
        }

        else
        {

            b[i] = a[j];
            j++;
        }

        i++;
    }

    if (m < h)
    {

        for (k = j; k <= r; k++)
        {

            b[i] = a[k];
            i++;
        }
    }

    else
    {

        for (k = h; k <= m; k++)
        {

            b[i] = a[k];
            i++;
        }
    }

    for (k = l; k <= r; k++)
    {

        a[k] = b[k];
    }
}

/********** Recursive Merge Function **********/
void mergeSort(int *a, int *b, int l, int r)
{

    int m;

    if (l < r)
    {

        m = (l + r) / 2;

        mergeSort(a, b, l, m);
        mergeSort(a, b, (m + 1), r);
        merge(a, b, l, m, r);
    }
}