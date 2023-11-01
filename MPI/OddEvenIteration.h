#include <iostream>
#include "mpi.h"

void Odd_even_iteration( double* local_A, double* temp_B, double* temp_C, int local_n, int my_rank, MPI_comm comm, int phase, int even_partner, int odd_partner ) {
    MPI_Status status;

    /* if phase is even */
    if ( phase % 2 == 0 ) {
        /* if even partner exists */
        if ( even_partner >= 0 ) {
            MPI_Sendrecv( local_A, local_n, MPI_DOUBLE, even_partner, 0, temp_B, local_n, MPI_DOUBLE, even_partner, 0, comm, &status );
            /* if my rank is odd */
            if ( my_rank % 2 ) {
                Merge_high( local_A, temp_B, temp_C, local_n );
            }
            /* if my rank is even */
            else {
                Merge_low( local_A, temp_B, temp_C, local_n );
            }
        }
    }
    /* if phase is odd */
    else {
        /* if odd partner exists */
        if ( odd_partner >= 0 ) {
            MPI_Sendrecv( local_A, local_n, MPI_DOUBLE, odd_partner, 0, temp_B, local_n, MPI_DOUBLE, odd_partner, 0, comm, &status );
            /* if my rank is odd */
            if ( my_rank % 2 ) {
                Merge_low( local_A, temp_B, temp_C, local_n );
            }
            /* if my rank is even */
            else {
                Merge_high( local_A, temp_B, temp_C, local_n );
            }
        }
    }
}