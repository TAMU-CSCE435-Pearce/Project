#include "mpi.h"
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using std::string;
using std::swap;
using std::vector;

int inputSize, numProcesses;

/* Define Caliper region names */
const char *mainFunction = "main";
const char *data_init = "data_init";
const char *correctness_check = "correctness_check ";
const char *comm = "comm";
const char *comm_large = "comm_large";
const char *comm_small = "comm _small";
const char *comp = "comp";
const char *comp_large = "comp_large";
const char *comp_small = "comp_small";

/* Data generation */
void generateData(vector<int> &localData, int startingSortChoice, int amountToGenerate, int startingPosition, int my_rank)
{
    switch (startingSortChoice)
    {
    case 0:
    { // Random
        srand((my_rank + 5) * (my_rank + 12) * 1235);
        for (int i = 0; i < amountToGenerate; i++)
        {
            localData.push_back(rand() % inputSize);
        }
        break;
    }

    case 1:
    { // Sorted
        int endValue = startingPosition + amountToGenerate;
        for (int i = startingPosition; i < endValue; i++)
        {
            localData.push_back(i);
        }
        break;
    }
    case 2:
    { // Reverse sorted
        int startValue = inputSize - 1 - startingPosition;
        int endValue = inputSize - amountToGenerate - startingPosition;
        for (int i = startValue; i >= endValue; i--)
        {
            localData.push_back(i);
        }
        break;
    }
    }
}

void merge(int arr[], int size, MPI_Comm comm, int my_rank, int num_ranks)
{
    int mid = size / 2;
    int *left = arr;
    int *right = arr + mid;

    int *temp = new int[size];
    int i = 0, j = mid, k = 0;

    while (i < mid && j < size)
    {
        if (left[i] <= right[j])
        {
            temp[k++] = left[i++];
        }
        else
        {
            temp[k++] = right[j++];
        }
    }

    while (i < mid)
    {
        temp[k++] = left[i++];
    }

    while (j < size)
    {
        temp[k++] = right[j++];
    }

    // Gather size information
    int localSize = k;
    int *sizes = new int[num_ranks];
    MPI_Gather(&localSize, 1, MPI_INT, sizes, 1, MPI_INT, 0, comm);

    // Calculate displacements
    int *displacements = new int[num_ranks];
    displacements[0] = 0;
    for (int i = 1; i < num_ranks; i++)
    {
        displacements[i] = displacements[i - 1] + sizes[i - 1];
    }

    // Gather sorted data
    MPI_Gatherv(temp, localSize, MPI_INT, arr, sizes, displacements, MPI_INT, 0, comm);

    delete[] temp;
    delete[] sizes;
    delete[] displacements;
}

void mergeSort(int arr[], int size, MPI_Comm comm, int my_rank, int num_ranks)
{
    if (size <= 1)
        return;

    // Split data into two parts
    int mid = size / 2;
    int *left = arr;
    int *right = arr + mid;

    // Recursively sort each part
    mergeSort(left, mid, comm, my_rank, num_ranks);
    mergeSort(right, size - mid, comm, my_rank, num_ranks);

    // Merge the sorted parts
    merge(arr, size, comm, my_rank, num_ranks);
}

bool verifyCorrect(vector<int> &sortedData, int my_rank)
{
    // Verify local data is in order
    for (int i = 1; i < sortedData.size() - 1; i++)
    {
        if (sortedData.at(i - 1) > sortedData.at(i))
        {
            printf("Sorting error on process with rank: %d\n", my_rank);
            return false;
        }
    }

    // Verify my start and end line up
    int myDataBounds[] = {sortedData.at(0), sortedData.at(sortedData.size() - 1)};
    int boundsArraySize = 2 * numProcesses;
    int allDataBounds[boundsArraySize];
    MPI_Allgather(&myDataBounds, 2, MPI_INT, &allDataBounds, 2, MPI_INT, MPI_COMM_WORLD);

    for (int i = 1; i < boundsArraySize - 1; i++)
    {
        if (allDataBounds[i - 1] > allDataBounds[i])
        {
            printf("Sorting error on bounds regions: %d\n", my_rank);
            return false;
        }
    }

    return true;
}

int main(int argc, char *argv[])
{
    int sortingType;
    if (argc == 4)
    {
        sortingType = atoi(argv[1]);
        numProcesses = atoi(argv[2]);
        inputSize = atoi(argv[3]);
    }
    else
    {
        printf("\n Please ensure input is as follows [input sorted status (0-2)] [# processes] [size of input]");
        return 0;
    }

    string inputType;
    switch (sortingType)
    {
    case 0:
    {
        inputType = "Randomized";
        break;
    }
    case 1:
    {
        inputType = "Sorted";
        break;
    }
    case 2:
    {
        inputType = "Reverse Sorted";
        break;
    }
    }

    int my_rank,   /* rank id of my process */
        num_ranks, /* total number of ranks*/
        rc;        /* misc */

    /* MPI Setup */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    if (num_ranks < 2)
    {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }
    if (num_ranks != numProcesses)
    {
        printf("Target number of processes and actual number of ranks do not match. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }

    if (my_rank == 0)
    {
        printf("Input type: %d\n", sortingType);
        printf("Number Processes: %d\n", numProcesses);
        printf("Input Size: %d\n", inputSize);
    }

    // CALI_MARK_BEGIN(mainFunction);

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    // Data generation
    // CALI_MARK_BEGIN(data_init);
    vector<int> myLocalData;
    int amountToGenerateMyself = inputSize / numProcesses;
    int startingPos = my_rank * amountToGenerateMyself;
    generateData(myLocalData, sortingType, amountToGenerateMyself, startingPos, my_rank);
    // CALI_MARK_END(data_init);

    // Main Alg
    // CALI_MARK_BEGIN(comp);
    // CALI_MARK_BEGIN(comp_small);
    mergeSort(myLocalData.data(), myLocalData.size(), MPI_COMM_WORLD, my_rank, num_ranks);
    // CALI_MARK_END(comp_small);
    // CALI_MARK_END(comp);

    // Verification
    vector<int> sortedData(myLocalData); // Copy data for verification
    // CALI_MARK_BEGIN(correctness_check);
    bool correct = verifyCorrect(sortedData, my_rank);
    // CALI_MARK_END(correctness_check);

    // CALI_MARK_END(mainFunction);

    if (!correct)
    {
        printf("There is a problem with the sorting. Quitting...\n");
    }
    else
    {
        if (my_rank == 0)
        {
            printf("\nAll data sorted correctly!");
        }
    }

    adiak::init(NULL);
    adiak::launchdate();                                           // launch date of the job
    adiak::libraries();                                            // Libraries used
    adiak::cmdline();                                              // Command line used to launch the job
    adiak::clustername();                                          // Name of the cluster
    adiak::value("Algorithm", "SampleSort");                       // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI");                       // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int");                               // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int));                   // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", inputSize);                          // The number of elements in input dataset (1000)
    adiak::value("InputType", inputType);                          // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", numProcesses);                       // The number of processors (MPI ranks)
    adiak::value("group_num", 16);                                 // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Handwritten & Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();

    MPI_Finalize();
}
