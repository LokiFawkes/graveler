#include <curand_kernel.h>
#include <iostream>
#include <stdio.h>

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
	Two kills are included - One within the stream that runs the simulation, one in a dedicated stream for stopping the simulation.
	Now even more unhinged, trying out bitwise optimization. 
*
*/

__global__ void sim_stop(unsigned long long *d_rolls, bool *d_kill, int *d_blocks) {
    while(*d_rolls < ROUNDS - (*d_blocks * BLOCKSIZE)) {
        if(*d_rolls >= ROUNDS - *d_blocks) *d_kill = true; return;
    }
    *d_kill = true;
    return;
}

__global__ void sim_rolls(int *d_maxOnes, unsigned long long *d_rolls, int seed, bool *d_kill) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ROUNDS) return;

    curandState state;
    curand_init(seed, idx, 0, &state);
    while(*d_rolls < ROUNDS - (idx * 2) && !*d_kill){
        int ones = 0;
        int curRolls = 0;
        for (int i = 0; i < 16; i++){
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
        if(*d_kill) return;
        // atomicMax writes the number of ones rolled this session to d_maxOnes if it is greater, atomicAdd increments d_rolls by one each session
        atomicMax(d_maxOnes, ones);
        atomicAdd(d_rolls, 1);
        
    }
}

int main() {
    int maxOnes = 0;
    int *d_maxOnes;
    unsigned long long rolls = 0;
    unsigned long long *d_rolls;
    bool kill = false;
    bool *d_kill;
    cudaDeviceProp prop;
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&prop, deviceId);
    int smCount = prop.multiProcessorCount;
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1); cudaStreamCreate(&stream2);
    // Allocate memory for CUDA copies of maxOnes and rolls
    cudaMalloc(&d_maxOnes, sizeof(int));
    cudaMalloc(&d_rolls, sizeof(unsigned long long));
    cudaMalloc(&d_kill, sizeof(bool));

    // Initialize these values to 0 on the card
    cudaMemcpy(d_maxOnes, &maxOnes, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rolls, &rolls, sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kill, &kill, sizeof(bool), cudaMemcpyHostToDevice);

    // Set block size, allocate blocks, run kernel
    int blockSize = BLOCKSIZE;
    // The code now polls the CUDA device for how many blocks it can run. The kernel will kill itself thread by thread when the target number of sessions has been reached.
    int maxActiveBlocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, sim_rolls, blockSize, 0);
    int numBlocks = smCount * maxActiveBlocks;
    // Passing number of blocks to device for the dedicated killswitch kernel
    int *d_blocks;
    cudaMalloc(&d_blocks, sizeof(int));
    cudaMemcpy(d_blocks, &numBlocks, sizeof(int), cudaMemcpyHostToDevice);
    //New timing method using cuda events
    float totalTime=0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    sim_stop<<<1, 1, 0, stream1>>>(d_rolls, d_kill, d_blocks);
    sim_rolls<<<numBlocks, blockSize, 0, stream2>>>(d_maxOnes, d_rolls, time(NULL), d_kill);
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
    cudaFree(d_kill);
    cudaFree(d_blocks);
    return 0;
}
