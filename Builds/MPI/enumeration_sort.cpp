/******************************************************************************
* FILE: mpi_mm.c
* DESCRIPTION:  
*   MPI Matrix Multiply - C Version
*   In this code, the master task distributes a matrix multiply
*   operation to numtasks-1 worker tasks.
*   NOTE:  C and Fortran versions of this code differ because of the way
*   arrays are stored/passed.  C arrays are row-major order but Fortran
*   arrays are column-major order.
* AUTHOR: Blaise Barney. Adapted from Ros Leibensperger, Cornell Theory
*   Center. Converted to MPI: George L. Gusciora, MHPCC (1/95)
* LAST REVISED: 09/29/2021
******************************************************************************/
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <float.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */

int NUM_VALS;
int sorting;
/* Define Caliper region names */
const char* comm = "comm";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* mpi_bcast = "MPI_Bcast";
const char* mpi_gather = "MPI_Gather";

float random_float()
{
  return (float)rand()/(float)RAND_MAX;
}

void data_init(float *arr, int length, int sorting)
{
  CALI_CXX_MARK_FUNCTION;
  if (sorting == 0){ //random sort
    srand(time(NULL));
    int i;
    for (i = 0; i < length; ++i) {
      arr[i] = random_float();
    }
  }

  else if (sorting == 1){ //sorted
    float step = 1.0; 
    for (int i = 0; i < length; i++) {
        arr[i] = i * step;
    }
  } 

  else if (sorting == 2){ //reverse sorted
    float step = 1.0; 
    for (int i = 0; i < length; i++) {
        arr[i] = length - i * step;
    }
  }

  else if (sorting == 3){
    float step = 1.0; // Step size for the sorted array

    // Initialize the array with sorted values
    for (int i = 0; i < length; i++) {
        arr[i] = i * step;
    }

    srand((unsigned)time(NULL));

    // Calculate 1% of the array's length
    int perturbCount = length / 100;

    for (int i = 0; i < perturbCount; i++) {
        // Select two random indices
        int idx1 = rand() % length;
        int idx2 = rand() % length;

        // Swap the elements at these indices
        float temp = arr[idx1];
        arr[idx1] = arr[idx2];
        arr[idx2] = temp;
    }
  }
}

bool correctness_check(float *arr, int length){
  CALI_CXX_MARK_FUNCTION;
  bool sorted = true;
  for (int i = 0; i < length-1; ++i) {
    if(arr[i]>arr[i+1]){
      sorted = false;
      break;
    }
  }
  return sorted;
}

void parallelEnumerationSort(float *array, int size, int rank, int numprocs) {

    // Broadcast the entire array to all processes
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(mpi_bcast);
    MPI_Bcast(array, size, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
    CALI_MARK_END(mpi_bcast);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    int local_size = size / numprocs;
    int start = rank * local_size;
    int end = (rank + 1) * local_size;//(rank == numprocs - 1) ? size : start + local_size;
    int *local_sorted_indices = (int *)malloc(size/numprocs * sizeof(int));

    // Perform enumeration sort on the entire array for the assigned range
    for (int i = start; i < end; ++i) {
        int count = 0;
        for (int j = 0; j < size; ++j) {
            if (array[i] > array[j] || (array[i] == array[j] && i > j)) {
                count++;
            }
        }
        local_sorted_indices[i-start] = count; // Place in the correct position
    }
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(mpi_gather);
    int *sorted_indices = NULL;
    // Gather the sorted segments into the global sorted array
    if (rank == MASTER) {
        sorted_indices = (int *)malloc(size * sizeof(int));
    }
    MPI_Gather(local_sorted_indices, local_size, MPI_INT, sorted_indices, local_size, MPI_INT, MASTER, MPI_COMM_WORLD);
    free(local_sorted_indices); 
    CALI_MARK_END(mpi_gather);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    // Merge the sorted segments in the master
    if (rank == MASTER) {
        float *array_cpy = (float *)malloc(size * sizeof(float));
        for(int i = 0; i < size; i++){
            array_cpy[i] = array[i];
        }
        
        for(int i = 0; i < size; i++){
            array[sorted_indices[i]] = array_cpy[i];
        }
        free(sorted_indices);
        free(array_cpy);
    }
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);
}
// no need for MPI_Barrier because Bcast and Gather synchronize all processes implicitly

int main (int argc, char *argv[])
{
    CALI_CXX_MARK_FUNCTION;

    int	numtasks,              /* number of tasks */
        taskid;                /* a task identifier */
    float *global_array = NULL;
    int sizeOfArray;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
    MPI_Comm_size(MPI_COMM_WORLD,&numtasks);

    sizeOfArray = atoi(argv[1]);
    NUM_VALS = sizeOfArray;
    sorting = atoi(argv[2]);

    cali::ConfigManager mgr;
    mgr.start();

    // Master initializes the array
    global_array = (float *)malloc(sizeOfArray * sizeof(float));
    if (global_array == NULL) {
        printf("Cannot allocate enough memory for the global array.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(1);
    }

    // fill the initial global array with random values
    if (taskid == MASTER) {
        // Initialize the array with random values
        data_init(global_array, NUM_VALS, sorting);

    }

    // Perform the parallel enumeration sort
    parallelEnumerationSort(global_array, sizeOfArray, taskid, numtasks);

    if (taskid == MASTER) {

        bool sorted = correctness_check(global_array, NUM_VALS);
        printf("\nSorted is: %s\n", sorted ? "true" : "false");
    }

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "EnumerationSort"); 
    adiak::value("ProgrammingModel", "MPI"); 
    adiak::value("Datatype", "float"); 
    adiak::value("SizeOfDatatype", sizeof(float));
    adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
    switch (sorting) {
      case 0:
        adiak::value("InputType", "Random");
        break;
      case 1:
        adiak::value("InputType", "Sorted");
        break;
      case 2:
        adiak::value("InputType", "ReverseSorted");
        break;
      case 3:
        adiak::value("InputType", "1%%perturbed");
        break;
      default:
        adiak::value("InputType", "Random");
     }
    //adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", numtasks); // The number of processors (MPI ranks)
    adiak::value("num_threads", 0); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", 0); // The number of CUDA blocks 
    adiak::value("group_num", 20); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").


   // Flush Caliper output before finalizing MPI
   mgr.stop();
   mgr.flush();
   // MPI_Comm_free(&MPI_COMM_WORKER);
   MPI_Finalize();
}