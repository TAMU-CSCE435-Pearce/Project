#include "mpi.h"
#include <vector>
#include <stdlib.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using std::vector;

int inputSize, numProcesses;

/* Data generation */
void generateData(vector<int> &localData, int startingSortChoice, int amountToGenerate, int startingPosition) {
    switch (startingSortChoice) {
    case 0: //Random
        srand(5304284160);
        for(int i = 0; i < amountToGenerate; i++) {
            localData.push_back(rand() % inputSize);
        }
        break;

    case 1: //Sorted
        int endValue = startingPosition + amountToGenerate;
        for(int i = startingPosition; i < endValue; i++) {
            localData.push_back(i);
        }
        break;

    case 2:  //Reverse sorted
        int startValue = inputSize - 1 - startingPosition;
        int endValue = inputSize - amountToGenerate - startingPosition;
        for(int i = startValue; i >= endValue; i--) {
            localData.push_back(i);
        }
        break;
    }
}


/* Sequential Quick Sort & Helpers 
*  quickSort and partition function from geeksforgeeks.org
*/
int partition(int arr[], int start, int end)
{
    int pivot = arr[start];
 
    int count = 0;
    for (int i = start + 1; i <= end; i++) {
        if (arr[i] <= pivot)
            count++;
    }
 
    // Giving pivot element its correct position
    int pivotIndex = start + count;
    swap(arr[pivotIndex], arr[start]);
 
    // Sorting left and right parts of the pivot element
    int i = start, j = end;
 
    while (i < pivotIndex && j > pivotIndex) {
 
        while (arr[i] <= pivot) {
            i++;
        }
 
        while (arr[j] > pivot) {
            j--;
        }
 
        if (i < pivotIndex && j > pivotIndex) {
            swap(arr[i++], arr[j--]);
        }
    }
 
    return pivotIndex;
}

void quickSort(int arr[], int start, int end)
{
 
    // base case
    if (start >= end)
        return;
 
    // partitioning the array
    int p = partition(arr, start, end);
 
    // Sorting the left part
    quickSort(arr, start, p - 1);
 
    // Sorting the right part
    quickSort(arr, p + 1, end);
}


/* Main Alg */
void sampleSort(vector<int> &localData, int my_rank) {
    /* Sample splitters */
    int numSplitters = 4;
    vector<int> sampledSplitters;
    srand(8472384065);
    for(int i = 0; i < numSplitters; i++) {
        sampledSplitters.push_back(localData.at(rand() % localData.size()));
    }


    /* Combine splitters */
    int totalSplitterArraySize = numSplitters * numProcesses;
    //vector<int> allSplitters;
    //allSplitters.resize(totalSplitterArraySize)
    int allSplitters[totalSplitterArraySize];
    
    MPI_Allgather(&sampledSplitters[0], numSplitters, MPI_INT, &allSplitters[0], numSplitters, MPI_INT, MPI_COMM_WORLD);


    /* Sort splitters & Decide cuts */
    quickSort(allSplitters, 0, totalSplitterArraySize-1); //In place sort

    vector<int> choosenSplitters;
    for(int i = 1; i < numProcesses; i++) {
        choosenSplitters.push_back(allSplitters[i*numSplitters]);
    }


    /* Eval local elements and place into buffers */
    vector<vector<int>> sendBuckets;
    for(int i = 0; i < numProcesses; i++){sendBuckets.push_back(vector<int>());}

    for(int i = 0; i < localData.size(); i++) {
        int notUsed = 1;
        for(int j = 0; j < choosenSplitters.size(); j++) {
            if(localData.at(i) < choosenSplitters.at(j)) {
                sendBuckets.at(j).push_back(localData.at(i));
                notUsed = 0;
                break;
            }
        }
        if(notUsed){sendBuckets.at(sendBuckets.size()-1).push_back(localData.at(i));}
    }


    /* Send/Receive Data */ 
    //Gather sizes
    int localBucketSizes[numProcesses];
    for(int i = 0; i < numProcesses; i++) {localBucketSizes[i] = sendBuckets.at(i).size();}

    //Communicate sizes
    int targetSizes[numProcesses];
    MPI_Gather(&localBucketSizes[my_rank], 1, MPI_INT, &targetSizes, 1, MPI_INT, my_rank, MPI_COMM_WORLD);

    //Sum and calculate displacements
    int myTotalSize = 0;
    for(int i = 0; i < numProcesses; i++) {myTotalSize += targetSizes[i];}

    int displacements[numProcesses];
    displacements[0] = 0;
    for(int i = 0; i < (numProcesses-1); i++) {displacements[i+1] = displacements[i] + targetSizes[i];}
    
    //Allocate array
    int unsortedData[myTotalSize];

    //Gather data
    MPI_Gatherv(&sendBuckets[my_rank][0], sendBuckets.at(my_rank.size()), MPI_INT, &unsortedData, &targetSizes, &displacements, MPI_INT, my_rank, MPI_COMM_WORLD);

    /* Sort */
    quickSort(unsortedData, 0, myTotalSize-1);
}



/* Verify */



/* Program Main */
int main (int argc, char *argv[])
{
    int sortingType;
    if (argc == 4) {
        sortingType = atoi(argv[1]);
        numProcesses = atoi(argv[2]);
        inputSize = atoi(argv[3]);
    }
    else {
        printf("\n Please ensure input is as follows [input sorted status (0-2)] [# processes] [size of input]");
        return 0;
    }

    int my_rank,        /* rank id of my process */
        num_ranks,      /* total number of ranks*/
        rc;             /* misc */

    
    /* Define Caliper region names */
    const char* main = "main";
    const char* data_init = "data_init";
    const char* correctness_check = "correctness_check ";
    const char* comm = "comm";
    const char* comm_large = "comm_large";
    const char* comm = "comm _small";
    const char* comp = "comp";
    const char* comp_large = "comp_large";
    const char* comp_small = "comp_small";

    /* MPI Setup */
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&num_ranks);
    if (num_ranks < 2) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }
    if(num_ranks != numProcesses) {
        printf("Target number of processes and actual number of ranks do not match. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    //Data generation
    vector<int> myLocalData;
    int amountToGenerateMyself = inputSize/numProcesses; //Should aways be based around powers of 2
    int startingPos = my_rank * (amountToGenerateMyself);
    generateData(myLocalData, sortingType, amountToGenerateMyself, startingPos);


    //Main Alg
    sampleSort(myLocalData, my_rank);



    //Verification




    // Flush Caliper output before finalizing MPI
   mgr.stop();
   mgr.flush();

   MPI_Finalize();
}
