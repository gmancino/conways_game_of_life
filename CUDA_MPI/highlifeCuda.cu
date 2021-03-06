// Include packages and also CUDA packages
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Result from last compute of world.
extern unsigned char *g_resultData;

// Current state of world. 
extern unsigned char *g_data;

// ----- SAVE RECEIVING ROWS FROM OTHER GPUS ----- //
// "Above" row
extern unsigned char *g_aboveRow;

// "Below" row 
extern unsigned char *g_belowRow;

// "Above" row
extern unsigned char *g_resultAboveRow;

// "Below" row 
extern unsigned char *g_resultBelowRow;

// ----- DECLARE KERNEL ----- //
__global__ void HL_kernel(unsigned int worldWidth, unsigned int worldHeight);


// Define number of Processors
int cudaDeviceCount;
cudaError_t cE; 

static inline void HL_initAllZeros(size_t worldWidth, size_t worldHeight, int myrank, int cudaDeviceCount )
{
    size_t total_world_size = worldWidth * worldHeight;

    // Initialize the data
    cudaMallocManaged(&g_data, (total_world_size * sizeof(unsigned char)));
    cudaMemset(g_data, 0, (total_world_size * sizeof(unsigned char)));

    // Initialize the resulting data
    cudaMallocManaged(&g_resultData, (total_world_size * sizeof(unsigned char)));
    cudaMemset(g_resultData, 0, (total_world_size * sizeof(unsigned char)));

    // Initialize the above row
    cudaMallocManaged(&g_aboveRow, (worldWidth * sizeof(unsigned char)));
    cudaMemset(g_aboveRow, 0, (worldWidth * sizeof(unsigned char)));

    // Initialize the below row
    cudaMallocManaged(&g_belowRow, (worldWidth * sizeof(unsigned char)));
    cudaMemset(g_belowRow, 0, (worldWidth * sizeof(unsigned char)));
}

static inline void HL_initAllOnes(size_t worldWidth, size_t worldHeight, int myrank, int cudaDeviceCount )
{
    size_t total_world_size = worldWidth * worldHeight;

    // Initialize the data
    cudaMallocManaged(&g_data, (total_world_size * sizeof(unsigned char)));
    cudaMemset(g_data, 0, (total_world_size * sizeof(unsigned char)));

    // Initialize the resulting data
    cudaMallocManaged(&g_resultData, (total_world_size * sizeof(unsigned char)));
    cudaMemset(g_resultData, 0, (total_world_size * sizeof(unsigned char)));

    // Initialize the above row
    cudaMallocManaged(&g_aboveRow, (worldWidth * sizeof(unsigned char)));
    cudaMemset(g_aboveRow, 0, (worldWidth * sizeof(unsigned char)));

    // Initialize the below row
    cudaMallocManaged(&g_belowRow, (worldWidth * sizeof(unsigned char)));
    cudaMemset(g_belowRow, 0, (worldWidth * sizeof(unsigned char)));

    int i;

    // set all rows of world to true
    for( i = 0; i < total_world_size; i++)
    {
	    g_data[i] = 1;

        // Set above and below rows
        if (i < worldWidth){
            g_aboveRow[i] = 1;
            g_belowRow[i] = 1;
        }
    }
}

static inline void HL_initOnesInMiddle(size_t worldWidth, size_t worldHeight, int myrank, int cudaDeviceCount )
{
    size_t total_world_size = worldWidth * worldHeight;

    // Initialize the data
    cudaMallocManaged(&g_data, (total_world_size * sizeof(unsigned char)));
    cudaMemset(g_data, 0, (total_world_size * sizeof(unsigned char)));

    // Initialize the resulting data
    cudaMallocManaged(&g_resultData, (total_world_size * sizeof(unsigned char)));
    cudaMemset(g_resultData, 0, (total_world_size * sizeof(unsigned char)));

    // Initialize the above row
    cudaMallocManaged(&g_aboveRow, (worldWidth * sizeof(unsigned char)));
    cudaMemset(g_aboveRow, 0, (worldWidth * sizeof(unsigned char)));

    // Initialize the below row
    cudaMallocManaged(&g_belowRow, (worldWidth * sizeof(unsigned char)));
    cudaMemset(g_belowRow, 0, (worldWidth * sizeof(unsigned char)));

    int i;
    for(i = worldWidth * (worldHeight - 1) + 128; i < worldWidth * (worldHeight - 1) + 139; i++){
        g_data[i] = 1;
    }
}

static inline void HL_initOnesAtCorners(size_t worldWidth, size_t worldHeight, int myrank, int cudaDeviceCount )
{
    size_t total_world_size = worldWidth * worldHeight;

    // Initialize the data
    cudaMallocManaged(&g_data, (total_world_size * sizeof(unsigned char)));
    cudaMemset(g_data, 0, (total_world_size * sizeof(unsigned char)));

    // Initialize the resulting data
    cudaMallocManaged(&g_resultData, (total_world_size * sizeof(unsigned char)));
    cudaMemset(g_resultData, 0, (total_world_size * sizeof(unsigned char)));

    // Initialize the above row
    cudaMallocManaged(&g_aboveRow, (worldWidth * sizeof(unsigned char)));
    cudaMemset(g_aboveRow, 0, (worldWidth * sizeof(unsigned char)));

    // Initialize the below row
    cudaMallocManaged(&g_belowRow, (worldWidth * sizeof(unsigned char)));
    cudaMemset(g_belowRow, 0, (worldWidth * sizeof(unsigned char)));

    if(myrank == 0){
        g_data[0] = 1; // upper left
        g_data[worldWidth-1]=1; // upper right
        g_aboveRow[0] = 1; // upper left
        g_aboveRow[worldWidth-1]=1; // upper right
    }
    if(myrank == cudaDeviceCount - 1){
        g_data[(worldHeight * (worldWidth-1))]=1; // lower left
        g_data[(worldHeight * (worldWidth-1)) + worldWidth-1]=1; // lower right
        g_belowRow[0] = 1;
        g_belowRow[worldWidth - 1] = 1;
    }
}

static inline void HL_initSpinnerAtCorner(size_t worldWidth, size_t worldHeight, int myrank, int cudaDeviceCount )
{
    size_t total_world_size = worldWidth * worldHeight;

    // Initialize the data
    cudaMallocManaged(&g_data, (total_world_size * sizeof(unsigned char)));
    cudaMemset(g_data, 0, (total_world_size * sizeof(unsigned char)));

    // Initialize the resulting data
    cudaMallocManaged(&g_resultData, (total_world_size * sizeof(unsigned char)));
    cudaMemset(g_resultData, 0, (total_world_size * sizeof(unsigned char)));

    // Initialize the above row
    cudaMallocManaged(&g_aboveRow, (worldWidth * sizeof(unsigned char)));
    cudaMemset(g_aboveRow, 0, (worldWidth * sizeof(unsigned char)));

    // Initialize the below row
    cudaMallocManaged(&g_belowRow, (worldWidth * sizeof(unsigned char)));
    cudaMemset(g_belowRow, 0, (worldWidth * sizeof(unsigned char)));

    if( myrank == 0 ){
        g_data[0] = 1; // upper left
        g_data[1] = 1; // upper left +1
        g_data[worldWidth-1]=1; // upper right
        
        g_aboveRow[0] = 1; // upper left
        g_aboveRow[1] = 1; // upper left +1
        g_aboveRow[worldWidth-1]=1; // upper right
    }
}

static inline void HL_initReplicator(size_t worldWidth, size_t worldHeight, int myrank, int cudaDeviceCount )
{
    size_t total_world_size = worldWidth * worldHeight;

    // Initialize the data
    cudaMallocManaged(&g_data, (total_world_size * sizeof(unsigned char)));
    cudaMemset(g_data, 0, (total_world_size * sizeof(unsigned char)));

    // Initialize the resulting data
    cudaMallocManaged(&g_resultData, (total_world_size * sizeof(unsigned char)));
    cudaMemset(g_resultData, 0, (total_world_size * sizeof(unsigned char)));

    // Initialize the above row
    cudaMallocManaged(&g_aboveRow, (worldWidth * sizeof(unsigned char)));
    cudaMemset(g_aboveRow, 0, (worldWidth * sizeof(unsigned char)));

    // Initialize the below row
    cudaMallocManaged(&g_belowRow, (worldWidth * sizeof(unsigned char)));
    cudaMemset(g_belowRow, 0, (worldWidth * sizeof(unsigned char)));

    size_t x, y;
    x = worldWidth/2;
    y = worldHeight/2;
    
    g_data[x + y*worldWidth + 1] = 1; 
    g_data[x + y*worldWidth + 2] = 1;
    g_data[x + y*worldWidth + 3] = 1;
    g_data[x + (y+1)*worldWidth] = 1;
    g_data[x + (y+2)*worldWidth] = 1;
    g_data[x + (y+3)*worldWidth] = 1; 
}

// ---------- EXPORT TO APPROPRIATE COMPILER ---------- //
extern "C" void HL_initMaster( unsigned int pattern, size_t worldWidth, size_t worldHeight, int myrank, int cudaDeviceCount )
{
    
    // INITIALIZE THE CUDA WORLD
    if( (cE = cudaGetDeviceCount( &cudaDeviceCount)) != cudaSuccess )
    {
    printf(" Unable to determine cuda device count, error is %d, count is %d\n", cE, cudaDeviceCount );
    exit(-1);
    }
    if( (cE = cudaSetDevice( myrank % cudaDeviceCount )) != cudaSuccess )
    {
    printf(" Unable to have myrank %d set to cuda device %d, error is %d \n", myrank, (myrank % cudaDeviceCount), cE);
    exit(-1); 
    }

    // INITIALIZE THE PATTERN
    switch(pattern)
    {
    case 0:
	HL_initAllZeros( worldWidth, worldHeight, myrank, cudaDeviceCount );
	break;
	
    case 1:
	HL_initAllOnes( worldWidth, worldHeight, myrank, cudaDeviceCount  );
	break;
	
    case 2:
	HL_initOnesInMiddle( worldWidth, worldHeight, myrank, cudaDeviceCount  );
	break;
	
    case 3:
	HL_initOnesAtCorners( worldWidth, worldHeight, myrank, cudaDeviceCount  );
	break;

    case 4:
	HL_initSpinnerAtCorner( worldWidth, worldHeight, myrank, cudaDeviceCount  );
	break;

    case 5:
	HL_initReplicator( worldWidth, worldHeight, myrank, cudaDeviceCount  );
	break;
	
    default:
	printf("Pattern %u has not been implemented \n", pattern);
	exit(-1);
    }
}

// MAIN KERNEL FUNCTION THAT DOES ALL OF THE WORK
__global__ void HL_kernel( unsigned char* d_data, unsigned char* d_resultData, unsigned char* d_aboveRow, unsigned char* d_belowRow, unsigned int worldWidth, unsigned int worldHeight){
    
    // Store index value
    size_t index;

    // Loop over the threads
    for(index = blockIdx.x * blockDim.x + threadIdx.x; index < worldWidth*worldHeight; index += blockDim.x * gridDim.x){

        // Allocate space
        int y0 = ((index + worldHeight - 1) % worldHeight) * worldWidth;
        int y1 = index * worldWidth;
        int y2 = ((index + 1) % worldHeight) * worldWidth;

        // Get the current block and thread
        int x;

        // Loop over corresponding COLUMNS
	        for (x = 0; x < worldWidth; ++x){

            // Set current column, left column, and right column
            int x1 = x;
            int x0 = (x1 + worldWidth - 1) % worldWidth; 
            int x2 = (x1 + 1) % worldWidth;

            // Get the status of the current cell to determine logic of life span
            int is_alive = d_data[x1+y1];

            // Count the number of alive neighbors
            int num_alive = 0;

            // Check above and below row cases
            if (x1+y1 < worldWidth) {
                num_alive = d_aboveRow[x0] + d_aboveRow[x1] + d_aboveRow[x2] + d_data[x0+y1] + d_data[x2+y1] + d_data[x0+y2] + d_data[x1+y2] + d_data[x2+y2];
            }
            else if (x1+y1 > worldWidth*worldHeight - worldWidth - 1) {
                num_alive = d_data[x0+y0] + d_data[x1+y0] + d_data[x2+y0] + d_data[x0+y1] + d_data[x2+y1] + d_belowRow[x0] + d_belowRow[x1] + d_belowRow[x2];
            }
            else {
                num_alive = d_data[x0+y0] + d_data[x1+y0] + d_data[x2+y0] + d_data[x0+y1] + d_data[x2+y1] + d_data[x0+y2] + d_data[x1+y2] + d_data[x2+y2];
            }

            // Logic for updating values
            if (is_alive == 1){
                // Cell is alive!
                if (num_alive < 2){
                    // Underpopulated
                    d_resultData[x1+y1] = 0;
                }
                else if (num_alive == 2 || num_alive == 3){
                    // Just the right amount of neighbors
                    d_resultData[x1+y1] = 1;
                }
                else {
                    // Overpopulated
                    d_resultData[x1+y1] = 0;
                }
            }
            else {
                // Cell is dead :(
                if (num_alive == 3 || num_alive == 6) {
                    // #Resurrected
                    d_resultData[x1+y1] = 1;
                }
                else {
                    // We stay dead
                    d_resultData[x1+y1] = 0;
                }
            }// End logic for staying dead
	    } // End x loop
    } // End loop over each thread

    // ----- SWAP DATA IN ABOVE ROWS AND BELOW ROWS ----- //
    int j;
    for(j = 0; j < worldWidth; j++){
        d_aboveRow[j] = d_resultData[j];
        d_belowRow[j] = d_resultData[j + worldWidth*(worldHeight - 1)];
    }

    // Synchronize the threads?
    __syncthreads();
}

// LAUNCH KERNEL FUNCTION
extern "C" void HL_kernelLaunch( unsigned char** d_data, unsigned char** d_resultData, unsigned char** d_aboveRow, unsigned char** d_belowRow, int block_count, int thread_count, unsigned int worldWidth, unsigned int worldHeight, int myrank){
    
    // Call the kernel
    HL_kernel<<<block_count,thread_count>>>(*d_data, *d_resultData, *d_aboveRow, *d_belowRow, worldWidth, worldHeight);

    // Synchronize the CUDA devices
    cudaDeviceSynchronize();
}

// Free memory
extern "C" void freeCudaArrays(int myrank){
    cudaFree(g_data);
    cudaFree(g_resultData);
    cudaFree(g_aboveRow);
    cudaFree(g_belowRow);
}
