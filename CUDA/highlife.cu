// Include packages and also CUDA packages
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Result from last compute of world.
unsigned char *g_resultData=NULL;

// Current state of world. 
unsigned char *g_data=NULL;

// Current width of world.
size_t g_worldWidth=0;

/// Current height of world.
size_t g_worldHeight=0;

/// Current data length (product of width and height)
size_t g_dataLength=0;  // g_worldWidth * g_worldHeight

static inline void HL_initAllZeros( size_t worldWidth, size_t worldHeight )
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    // g_data = calloc( g_dataLength, sizeof(unsigned char));
    // g_resultData = calloc( g_dataLength, sizeof(unsigned char)); 

    // Set memory to CUDA
    cudaMallocManaged(&g_data, (g_dataLength * sizeof(unsigned char)));
    // Zero out the elements
    cudaMemset(g_data, 0, (g_dataLength * sizeof(unsigned char)));

    // Same for results
    cudaMallocManaged(&g_resultData, (g_dataLength * sizeof(unsigned char)));
    cudaMemset(g_resultData, 0, (g_dataLength * sizeof(unsigned char)));
}

static inline void HL_initAllOnes( size_t worldWidth, size_t worldHeight )
{
    int i;
    
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    // Set memory to CUDA
    cudaMallocManaged(&g_data, (g_dataLength * sizeof(unsigned char)));
    cudaMemset(g_data, 0, (g_dataLength * sizeof(unsigned char)));

    // set all rows of world to true
    for( i = 0; i < g_dataLength; i++)
    {
	g_data[i] = 1;
    }
    
    cudaMallocManaged(&g_resultData, (g_dataLength * sizeof(unsigned char)));
    cudaMemset(g_resultData, 0, (g_dataLength * sizeof(unsigned char)));
}

static inline void HL_initOnesInMiddle( size_t worldWidth, size_t worldHeight )
{
    int i;
    
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    cudaMallocManaged(&g_data, (g_dataLength * sizeof(unsigned char)));
    cudaMemset(g_data, 0, (g_dataLength * sizeof(unsigned char)));

    // set first 1 rows of world to true
    for( i = 10*g_worldWidth; i < 11*g_worldWidth; i++)
    {
	if( (i >= ( 10*g_worldWidth + 10)) && (i < (10*g_worldWidth + 20)))
	{
	    g_data[i] = 1;
	}
    }
    
    cudaMallocManaged(&g_resultData, (g_dataLength * sizeof(unsigned char)));
    cudaMemset(g_resultData, 0, (g_dataLength * sizeof(unsigned char)));
}

static inline void HL_initOnesAtCorners( size_t worldWidth, size_t worldHeight )
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    cudaMallocManaged(&g_data, (g_dataLength * sizeof(unsigned char)));
    cudaMemset(g_data, 0, (g_dataLength * sizeof(unsigned char)));

    g_data[0] = 1; // upper left
    g_data[worldWidth-1]=1; // upper right
    g_data[(worldHeight * (worldWidth-1))]=1; // lower left
    g_data[(worldHeight * (worldWidth-1)) + worldWidth-1]=1; // lower right
    
    cudaMallocManaged(&g_resultData, (g_dataLength * sizeof(unsigned char)));
    cudaMemset(g_resultData, 0, (g_dataLength * sizeof(unsigned char)));
}

static inline void HL_initSpinnerAtCorner( size_t worldWidth, size_t worldHeight )
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    cudaMallocManaged(&g_data, (g_dataLength * sizeof(unsigned char)));
    cudaMemset(g_data, 0, (g_dataLength * sizeof(unsigned char)));

    g_data[0] = 1; // upper left
    g_data[1] = 1; // upper left +1
    g_data[worldWidth-1]=1; // upper right
    
    cudaMallocManaged(&g_resultData, (g_dataLength * sizeof(unsigned char)));
    cudaMemset(g_resultData, 0, (g_dataLength * sizeof(unsigned char)));
}

static inline void HL_initReplicator( size_t worldWidth, size_t worldHeight )
{
    size_t x, y;
    
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    cudaMallocManaged(&g_data, (g_dataLength * sizeof(unsigned char)));
    cudaMemset(g_data, 0, (g_dataLength * sizeof(unsigned char)));

    x = worldWidth/2;
    y = worldHeight/2;
    
    g_data[x + y*worldWidth + 1] = 1; 
    g_data[x + y*worldWidth + 2] = 1;
    g_data[x + y*worldWidth + 3] = 1;
    g_data[x + (y+1)*worldWidth] = 1;
    g_data[x + (y+2)*worldWidth] = 1;
    g_data[x + (y+3)*worldWidth] = 1; 
    
    cudaMallocManaged(&g_resultData, (g_dataLength * sizeof(unsigned char)));
    cudaMemset(g_resultData, 0, (g_dataLength * sizeof(unsigned char))); 
}

static inline void HL_initMaster( unsigned int pattern, size_t worldWidth, size_t worldHeight )
{
    switch(pattern)
    {
    case 0:
	HL_initAllZeros( worldWidth, worldHeight );
	break;
	
    case 1:
	HL_initAllOnes( worldWidth, worldHeight );
	break;
	
    case 2:
	HL_initOnesInMiddle( worldWidth, worldHeight );
	break;
	
    case 3:
	HL_initOnesAtCorners( worldWidth, worldHeight );
	break;

    case 4:
	HL_initSpinnerAtCorner( worldWidth, worldHeight );
	break;

    case 5:
	HL_initReplicator( worldWidth, worldHeight );
	break;
	
    default:
	printf("Pattern %u has not been implemented \n", pattern);
	exit(-1);
    }
}

static inline void HL_swap( unsigned char **pA, unsigned char **pB)
{
    // Create temporay holder to hold A's values
    unsigned char *temporary;
    temporary = *pA;

    // Perform the swap
    *pA = *pB;
    *pB = temporary;
}
 
/*
// Don't Modify this function or your submitty autograding will not work
static inline void HL_printWorld(size_t iteration)
{
    int i, j;

    printf("Print World - Iteration %lu \n", iteration);
    
    for( i = 0; i < g_worldHeight; i++)
    {
	printf("Row %2d: ", i);
	for( j = 0; j < g_worldWidth; j++)
	{
	    printf("%u ", (unsigned int)g_data[(i*g_worldWidth) + j]);
	}
	printf("\n");
    }

    printf("\n\n");
}*/

// MAIN KERNEL FUNCTION THAT DOES ALL OF THE WORK
__global__ void HL_kernel(unsigned char* d_data, unsigned int worldWidth, unsigned int worldHeight, unsigned char* d_resultData){
    // Store index value
    size_t index;

    // Loop over the threads
    for(index = blockIdx.x * blockDim.x + threadIdx.x; index < worldWidth*worldHeight; index += blockDim.x * gridDim.x){

        // Grab the current y
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
            num_alive = d_data[x0+y0] + d_data[x1+y0] + d_data[x2+y0] + d_data[x0+y1] + d_data[x2+y1] + d_data[x0+y2] + d_data[x1+y2] + d_data[x2+y2];

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
            }
	    } // End x loop

    } // End loop over each thread

    // Synchronize the threads?
    __syncthreads();
}

// LAUNCH KERNEL FUNCTION
bool HL_kernelLaunch(unsigned char** d_data, unsigned char** d_resultData, size_t worldWidth, size_t worldHeight, size_t iterationsCount, ushort threadsCount){
    // Delcare iteration variable
    int i;

    // Declare number of blocks
    int block_count = (worldHeight * worldWidth) / threadsCount;

    // Loop over the iterations
    for(i = 0; i < iterationsCount; i++){

        // Perform kernel operations in parallel over the threads
        HL_kernel<<<block_count,threadsCount>>>(*d_data, worldWidth, worldHeight, *d_resultData);

        // Synchronize the CUDA devices
        cudaDeviceSynchronize();

        // Swap the pointers
        HL_swap(d_data, d_resultData);

    } // End iterations loop

    // Synchronize the device again?
    cudaDeviceSynchronize();

    return true;
}

int main(int argc, char *argv[])
{
    unsigned int pattern = 0;
    unsigned int worldSize = 0;
    unsigned int iterations = 0;
    unsigned int thread_count = 0;

    printf("This is the HighLife running in Parallel on a GPU.\n");

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
    
    // Initialize the world
    HL_initMaster(pattern, worldSize, worldSize);

    // Launch the kernel
    HL_kernelLaunch(&g_data, &g_resultData, worldSize, worldSize, iterations, thread_count);

    // Free memory
    cudaFree(g_data);
    cudaFree(g_resultData);
    
    return true;
}
