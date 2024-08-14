#include <curand_kernel.h>
#include <iostream>

#define ROUNDS 1000000000
#define ROLLS 231
#define BLOCKSIZE 1024

/*
*
	This is made to simulate the number of times a 1 is rolled on a d4 per round of 231 rolls, similar to Austin Hourigan's graveler.py
	I've taken some liberties to optimize it further, in addition to making it run in CUDA.
	The kernel, __global__ void sim_rolls, does the heavy lifting, as it runs on the CUDA cores, reporting back to the system after completing.
	Unnecessary arrays were dropped, instead keeping track of the fact that the rolls were done and how many times in a round a one was rolled, AKA graveler was too paralyzed to move.
	The unnecessary check for 177 ones has been dropped. This is running too fast to stop at a precise target.
	In this version, the number of roll sessions can overflow, but should not go below target. This wastes valuable time on excess roll sessions, but is faster than my other code because it's not constructing and destructing threads over and over.
	Now even more unhinged, trying out bitwise optimization. 
	The separate killswitch thread has been removed. I am now using shared memory to speed up atomic operations even further. While the killswitch did give me tighter control over the number of sessions rolled, it was wasting precious time now that I'm in such a small time scale.
*
*/

__global__ void sim_rolls(int *d_maxOnes, unsigned long long *d_rolls, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ROUNDS) return;
    __shared__ int sharedMax[BLOCKSIZE];
    curandState state;
    curand_init(seed, idx, 0, &state);
    while(*d_rolls < ROUNDS - (idx * 4)){
        int ones = 0;
        int curRolls = 0;
        for (int i = 0; i < 15; i++){
            unsigned int roll = curand(&state);
            for (int j = 0; j < 16; j++){
                if(curRolls < 231){
                    int shift = (j * 2);
                    unsigned int currentRoll = (roll >> shift) & 0x03;
                    if(currentRoll == 0) ones++;
                    curRolls++;
                }
            }
        }
        sharedMax[threadIdx.x] = max(sharedMax[threadIdx.x], ones);
        atomicAdd(d_rolls, 1);
    }
    int newMax = 0;
    for (int x=0; x<BLOCKSIZE; x++){
        newMax = max(sharedMax[x], newMax);
    }
    atomicMax(d_maxOnes, newMax);
}

int main() {
    int maxOnes = 0;
    int *d_maxOnes;
    unsigned long long rolls = 0;
    unsigned long long *d_rolls;
    cudaDeviceProp prop;
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&prop, deviceId);
    int smCount = prop.multiProcessorCount;
    // Allocate memory for CUDA copies of maxOnes and rolls
    cudaMalloc(&d_maxOnes, sizeof(int));
    cudaMalloc(&d_rolls, sizeof(unsigned long long));

    // Initialize these values to 0 on the card
    cudaMemcpy(d_maxOnes, &maxOnes, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rolls, &rolls, sizeof(unsigned long long), cudaMemcpyHostToDevice);

    // Set block size, allocate blocks, run kernel
    int blockSize = BLOCKSIZE;
    // The code now polls the CUDA device for how many blocks it can run. The kernel will kill itself thread by thread when the target number of sessions has been reached.
    int maxActiveBlocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, sim_rolls, blockSize, 0);
    int numBlocks = smCount * maxActiveBlocks;
    //New timing method using cuda events
    float totalTime=0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    sim_rolls<<<numBlocks, blockSize, 0>>>(d_maxOnes, d_rolls, time(NULL));
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&totalTime, start, stop);
    
    // Copy the result back to system memory, now that the CUDA program is over
    cudaMemcpy(&maxOnes, d_maxOnes, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&rolls, d_rolls, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // Report the important numbers
    std::cout << "Highest Ones Roll: " << maxOnes << std::endl;
    std::cout << "Number of Roll Sessions: " << rolls << std::endl;
    std::cout << totalTime << "ms" << std::endl;
    // Never malloc without a free
    cudaFree(d_maxOnes);
    cudaFree(d_rolls);
    return 0;
}
