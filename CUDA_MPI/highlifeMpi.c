// INCLUDE MAIN PACKAGES
#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>

// ----- SAVE DATA ON THIS MACHINE ----- //
// Result from last compute of world.
unsigned char *g_resultData=NULL;

// Current state of world. 
unsigned char *g_data=NULL;

// ----- SAVE RECEIVING ROWS FROM OTHER GPUS ----- //
// "Above" row
unsigned char *g_aboveRow=NULL;

// "Below" row 
unsigned char *g_belowRow=NULL;

// ----- IMPORT INIT FUNCTION ----- //
extern void HL_initMaster(int pattern, size_t worldWidth, size_t worldHeight, int myrank );

// ----- IMPORT KERNEL FUNCTION ----- //
extern void HL_kernelLaunch( unsigned char** d_data, unsigned char** d_resultData, unsigned char** d_aboveRow, unsigned char** d_belowRow, int block_count, int thread_count, unsigned int worldWidth, unsigned int worldHeight, int myrank );

// ----- IMPORT FREE MEMORY FUNCTION ----- //
extern void freeCudaArrays(int myrank);

// ----- SWAP POINTER FUNCTIONS ----- //
static inline void HL_swapPointers( unsigned char **pA, unsigned char **pB)
{
    // Create temporary holder to hold A's values
    unsigned char *temporary;
    temporary = *pA;

    // Perform the swap
    *pA = *pB;
    *pB = temporary;
}


static inline void PrintWorld( int myrank, int iter, int worldSize ){

    char buff[15];
    FILE *f;

    // ----- WRITE FILE ----- //
    if(iter == 0){
        snprintf(buff, 15, "initial_%d.txt", myrank);
        f = fopen(buff, "w");
    }
    else{
        snprintf(buff, 15, "output_%d.txt", myrank);
        f = fopen(buff, "w");
    }

    // Check if it exists
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }

    int i, j;

    fprintf(f, "\nWorld Chunk %d - Iteration %d \n", myrank, iter);
    
    for( i = 0; i < worldSize; i++)
    {
	    fprintf(f, "Row %2d: ", i);
	    for( j = 0; j < worldSize; j++)
	    {
	        fprintf(f, "%u ", (unsigned int)g_data[(i*worldSize) + j]);
	    }
	    fprintf(f, "\n");
    }

    fclose(f);
}


int main(int argc, char *argv[])
{

    // Read in user arguments
    unsigned int pattern = 0;
    unsigned int worldSize = 0;
    unsigned int iterations = 0;
    unsigned int thread_count = 0;

    // ---------- START MPI ---------- //
    MPI_Status stat;
    MPI_Request send_request, recv_request;
    MPI_Init(&argc, &argv);

    // Get MY mpi rank
    int myrank; 
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // Gent total number of GPUs
    int numranks;
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);

    if(myrank == 0){
        printf("\n\n|---------------------------------------------------------------|\n");
        printf("\tThis is the HighLife running in Parallel on %d GPUs.\n", numranks);
        printf("|---------------------------------------------------------------|\n\n");
    }

    if( argc != 5 )
    {
	printf("HighLife requires 4 arguments, 1st is pattern number, 2nd the sq size of the world, 3rd is the number of iterations, and 4th is the thread count, e.g. ./highlife 0 64 2 32 \n");
	exit(-1);
    }

    // Read in arguments
    pattern = atoi(argv[1]);
    worldSize = atoi(argv[2]);
    iterations = atoi(argv[3]);
    thread_count = atoi(argv[4]);

    // Get block count
    int block_count = (worldSize * worldSize) / thread_count;

    // ---------- INIT CUDA WORLDS MPI ---------- //
    HL_initMaster(pattern, worldSize, worldSize, myrank);

    // ----------  START THE CLOCK ---------- //
    double t0, t1;
    if(myrank == 0){
        t0 = MPI_Wtime();
    }

    // Print init world
    if(worldSize <= 64){
        PrintWorld( myrank, 0, worldSize );
    }

    // Get rank of above world and below world
    int aboveRank = ( numranks + myrank - 1 ) % numranks ;
    int belowRank = ( numranks + myrank + 1 ) % numranks ;
    
    // Make special tag for each send and receive
    int tag1 = 10;
    int tag2 = 20;

    // ----------  INIT BARRIER ---------- //
    MPI_Barrier( MPI_COMM_WORLD );

    int num_ticks;
    // ----------  PERFORM LOOP ---------- //
    for(num_ticks = 0; num_ticks < iterations; num_ticks++){

        // Send and receive to above row
        MPI_Irecv(g_belowRow, worldSize, MPI_UNSIGNED_CHAR, aboveRank, tag1, MPI_COMM_WORLD, &recv_request);
        MPI_Isend(g_belowRow, worldSize, MPI_UNSIGNED_CHAR, belowRank, tag1, MPI_COMM_WORLD, &send_request);

        // Barrier
        MPI_Wait(&send_request, &stat);
        MPI_Wait(&recv_request, &stat);

        // Send and receive to below row
        MPI_Irecv(g_aboveRow, worldSize, MPI_UNSIGNED_CHAR, belowRank, tag2, MPI_COMM_WORLD, &recv_request);
        MPI_Isend(g_aboveRow, worldSize, MPI_UNSIGNED_CHAR, aboveRank, tag2, MPI_COMM_WORLD, &send_request);

        // Barrier
        MPI_Wait(&send_request, &stat);
        MPI_Wait(&recv_request, &stat);

        // SWAP THE POINTERS
        HL_swapPointers(&g_aboveRow, &g_belowRow);
        MPI_Barrier( MPI_COMM_WORLD );

        // Call the Kernel function
        HL_kernelLaunch(&g_data, &g_resultData, &g_aboveRow, &g_belowRow, block_count, thread_count, worldSize, worldSize, myrank);
        
        // Swap the global data
        HL_swapPointers(&g_data, &g_resultData);

        // Barrier
        MPI_Barrier( MPI_COMM_WORLD );
    }

    // ----------  END THE CLOCK ---------- //
    if(myrank == 0){
        t1 = MPI_Wtime();
        printf("\n[WORLD SIZE] %d x %d\n[ITERATIONS] %d\n[EXECUTION TIME] %f\n", worldSize, worldSize, iterations, t1-t0);
    }

    // Print world
    if(worldSize <= 64){
        PrintWorld( myrank, iterations, worldSize );
    }

    // ----------  FREE CUDA ARRAYS ---------- //
    freeCudaArrays(myrank);

    // ---------- FINISH MPI ---------- //
    // Barrier
    MPI_Barrier( MPI_COMM_WORLD );
    MPI_Finalize();

    return 0;
}
