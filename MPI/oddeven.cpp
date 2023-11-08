#include "common.h"

void edgeR(vector<float>* local_values, int i, int num_procs, int rank) {
    // Send local[rank][0] to rank-1
    // receive new local[rank][0] from rank-1
    float r = (*local_values)[0];
    MPI_Send(&r, 1, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD);
    //printf("Sending r=%.2f: i=%d, rank=%d, [0]=%.2f\n", r, i, rank, (*local_values)[0]);
    //printf("Waiting for newr: i=%d, rank=%d, [0]=%.2f\n", i, rank, (*local_values)[0]);
    float newr;
    MPI_Status status;
    MPI_Recv(&newr, 1, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    //printf("Received newr=%.2f: i=%d, rank=%d, [0]=%.2f\n", newr, i, rank, (*local_values)[0]);
    (*local_values)[0] = newr;
}

void edgeL(vector<float>* local_values, int i, int num_procs, int rank) {
    // Receive local[rank+1][0] from rank+1
    // sort local[rank][-1]
    // send new local[rank+1][0] to rank+1
    float local_n = (*local_values).size();
    float r;
    MPI_Status status;
    MPI_Recv(&r, 1, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &status);
    //printf("Waiting for r: i=%d, rank=%d, [0]=%.2f\n", i, rank, (*local_values)[0]);
    //printf("Received r=%.2f: i=%d, rank=%d, [0]=%.2f\n", r, i, rank, (*local_values)[0]);
    float newr=r;
    if(r<(*local_values)[local_n-1]) {
        newr = (*local_values)[local_n-1];
        (*local_values)[local_n-1] = r;
    }
    MPI_Send(&newr, 1, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    //printf("Sent newr=%.2f: i=%d, rank=%d, [0]=%.2f\n", newr, i, rank, (*local_values)[0]);
}

void oddeven_sort(int NUM_VALS, vector<float> *local_values, int local_size, int num_procs, int rank)
{

    //CALI_MARK_BEGIN(SAMPLE_SORT_NAME);
    if(NUM_VALS % num_procs != 0) {
        printf("ERROR: num_procs must be a multple of NUM_VALS");
        return;
    }
    int start = rank * local_values->size();
    int end = start + local_size - 1;

    /*if(rank==0) {
        printf("Master   : %d, %d\n", NUM_VALS, local_size);
    } else {
        printf("Worker %d: %d, %d\n", rank, NUM_VALS, local_size);
    }*/

    //printf("Rank %d: ", rank);
    /*for(int i=0;i<local_values->size();i++) {
        //(*local_values)[i] = NUM_VALS - start + i;
        printf("%.2f ", (*local_values)[i]);
    }
    //printf("\n");*/
    
    MPI_Barrier(MPI_COMM_WORLD);
    for(int i=0;i<NUM_VALS;i++) {
        for(int j=i%2;j<local_values->size()-1;j+=2) {
            int l = j;
            int r = l + 1;
            if((*local_values)[l] > (*local_values)[r]) {
                float tmp = (*local_values)[l];
                (*local_values)[l] = (*local_values)[r];
                (*local_values)[r] = tmp;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        //printf("Synchronized rank %d 1\n", rank);
        if(local_values->size()%2==1) { // Odd n local
            if(i%2==0) { // Even phase
                if(rank%2==0 && rank>0) { // Even rank
                    edgeR(local_values, i, num_procs, rank);
                } else if(rank%2==1 && rank<num_procs-1){ // Odd rank
                    edgeL(local_values, i, num_procs, rank);
                }
            } else { // Odd phase
                if(rank%2==0 && rank < num_procs-1) { // Even rank
                    edgeL(local_values, i, num_procs, rank);
                } else if(rank%2==1 && rank>0){ // Odd rank
                    edgeR(local_values, i, num_procs, rank);
                }
            }
        } else { // Even localsize
            if(i%2==0) { // Even phase
                
            } else { // Odd phase
                if(rank%2==0 && rank<num_procs-1) { // Even rank
                    edgeL(local_values, i, num_procs, rank);
                } else if(rank%2==1 && rank>0){ // Odd rank
                    edgeR(local_values, i, num_procs, rank);
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    /*for(int i=0;i<num_procs;i++) {
        if(rank==i) {
            printf("Rank: %d - ", rank);
            for(int j=0;j<(*local_values).size();j++) {
                printf("%.2f ", (*local_values)[j]);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }printf("\n");*/
    //CALI_MARK_END(SAMPLE_SORT_NAME);
}
