#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */

int main (int argc, char *argv[])
{
CALI_CXX_MARK_FUNCTION;
    
int sizeOfMatrix;
if (argc == 2)
{
    sizeOfMatrix = atoi(argv[1]);
}
else
{
    printf("\n Please provide the size of the matrix");
    return 0;
}

int	numtasks,              /* number of tasks in partition */
	taskid,                /* a task identifier */
	numworkers,            /* number of worker tasks */
	source,                /* task id of message source */
	dest,                  /* task id of message destination */
	mtype,                 /* message type */
	rows,                  /* rows of matrix A sent to each worker */
	averow, extra, offset, /* used to determine rows sent to each worker */
	i, j, k, rc;           /* misc */

double	a[sizeOfMatrix][sizeOfMatrix],           /* matrix A to be multiplied */
	b[sizeOfMatrix][sizeOfMatrix],           /* matrix B to be multiplied */
	c[sizeOfMatrix][sizeOfMatrix]; 

MPI_Status status;    

/* Define Caliper region names */
const char* whole_computation = "whole_computation";
const char* data_init = "data_init";
const char* comm = "comm";
const char* comm_small = "comm_small";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_small = "comp_small";
const char* comp_large = "comp_large";
const char* correctness_check = "correctness_check";

MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
MPI_Comm_size(MPI_COMM_WORLD,&numtasks);

if (numtasks < 2 ) {
  printf("Need at least two MPI tasks. Quitting...\n");
  MPI_Abort(MPI_COMM_WORLD, rc);
  exit(1);
  }
numworkers = numtasks-1;

//new comm
MPI_Comm new_comm;
// MPI_Comm_split(MPI_COMM_WORLD, (taskid != MASTER), taskid, &new_comm);                             TRY IF THINGS GO WRONG
MPI_Comm_split(MPI_COMM_WORLD, (taskid == MASTER) ? MPI_UNDEFINED : 0, 0, &new_comm);

CALI_MARK_BEGIN(whole_computation);

// Create caliper ConfigManager object
cali::ConfigManager mgr;
mgr.start();

/************** MASTER ****************/

    if (taskid == MASTER){
        // INITIALIZATION PART FOR THE MASTER PROCESS STARTS HERE

        printf("Enumeration sort has started with %d tasks.\n",numtasks);
        printf("Initializing arrays...\n");

        CALI_MARK_BEGIN(data_init);

        for(int i = 0; i < sizeOfMatrix; i++) {
            b[i] = srand(1);
        }
        CALI_MARK_END(data_init);

        // Send matrix data to worker

        averow = sizeOfMatrix/numworkers;
        extra = sizeOfMatrix%numworkers;
        offset = 0;
        mtype = FROM_MASTER;

        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_large);

        for(int i = 1; i <= numworkers; i++) {
            rows = (i <= extra) ? averow+1 : averow;   	
            printf("Sending %d rows to task %d offset=%d\n",rows,i,offset);
            MPI_Send(&offset, 1, MPI_INT, i, mtype, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, i, mtype, MPI_COMM_WORLD);
            MPI_Send(&a[offset], sizeOfMatrix, MPI_DOUBLE, i, mtype,
                    MPI_COMM_WORLD);
            offset += rows;
        }
        CALI_MARK_END(comm_large);
        CALI_MARK_END(comm);

    }

    if (taskid > MASTER){

        mtype = FROM_MASTER;

        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_small);

        MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&a, rows, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
        //MPI_Recv(&b, sizeOfArray*sizeOfArray, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);

        CALI_MARK_END(comm_small);
        CALI_MARK_END(comm);

        // CALI_MARK_BEGIN(comp);
        // CALI_MARK_BEGIN(comp_small);

        //enumeration_sort()

        // CALI_MARK_END(comp_small);
        // CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_large);

        //sampleSort(a, sizeOfArray, sizeOfArray/(2*numtasks), numtasks); // function to implement                         use array b?

        CALI_MARK_END(comp_large);
        CALI_MARK_END(comp);

    }

bool sorted = true;

CALI_MARK_BEGIN(correctness_check);
//check array order

for (int i = 1; i < sizeOfArray; i++) {
    if (a[i] < a[i-1]) {
        printf("Error. Out of order sequence: %d found\n", a[i]);
        sorted = false;
    }
}
if (sorted) {
    printf("Array is in sorted order\n");
}


    CALI_MARK_END(correctness_check);
    //END COMPUTATION
    CALI_MARK_END(whole_computation);

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "EnumSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "Double"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(Double)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", sizeOfMatrix); // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", num_tasks); // The number of processors (MPI ranks)

    // adiak::value("num_threads", numworkers); // The number of CUDA or OpenMP threads
    // adiak::value("num_blocks", 0); // The number of CUDA blocks 

    adiak::value("group_num", 25); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "AI/Handwritten") // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    MPI_Comm_free(&new_comm);

   // Flush Caliper output before finalizing MPI
   mgr.stop();
   mgr.flush();

   MPI_Finalize();
   return 0;

};

int compInt(const void* a, const void* b) {
    int arg1 = *(const int*)a;
    int arg2 = *(const int*)b;

    if(arg1 == arg2) {return 0};
    else if(arg1 > arg2) {return 1};
    else {return -1;}

};


void enum_sort(int n, int *A) {

    CALI_CXX_MARK_FUNCTION;
    const int threshold = 10;

    int C[N] = {0};
    for (int j = 0; j < n; j++) {
        C[j] = 0;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if ((A[i] < A[j]) || (A[i] == A[j] && i < j)) {
                C[j] = 1;
            } else {
                C[j] = 0;
            }
        }
    }
    int B[N] = {0};
    for (int j = 0; j < n; j++) {
        A[C[j]] = A[j];
    }
}
