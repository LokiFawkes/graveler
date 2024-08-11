#include <curand_kernel.h>
#include <iostream>

#define ROUNDS 1000000000
#define ROLLS 231

/*
*
	This is made to simulate the number of times a 1 is rolled on a d4 per round of 231 rolls, similar to Austin Hourigan's graveler.py
	I've taken some liberties to optimize it further, in addition to making it run in CUDA.
	The kernel, __global__ void sim_rolls, does the heavy lifting, as it runs on the CUDA cores, reporting back to the system after completing.
	Unnecessary arrays were dropped, instead keeping track of the fact that the rolls were done and how many times in a round a one was rolled, AKA graveler was too paralyzed to move.
	Despite the virtual impossibility of actually getting 177 ones, I kept the condition and as such still keep track of how many roll sessions occurred.
*
*/

__global__ void sim_rolls(int *d_maxOnes, int *d_rolls, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ROUNDS || *d_maxOnes >= 177) return;

    curandState state;
    curand_init(seed, idx, 0, &state);
    while(*d_rolls < ROUNDS - idx){
        int ones = 0;
        if(*d_rolls >= ROUNDS - idx) return;
        for (int i = 0; i < ROLLS; i++) {
            int roll = curand(&state) % 4 + 1;
            if (roll == 1) ones++;
        }

        // atomicMax writes the number of ones rolled this session to d_maxOnes if it is greater, atomicAdd increments d_rolls by one each session
        atomicMax(d_maxOnes, ones);
        atomicAdd(d_rolls, 1);
    }
}

int main() {
    int maxOnes = 0;
    int *d_maxOnes;
    int rolls = 0;
    int *d_rolls;
    cudaDeviceProp prop;
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&prop, deviceId);
    int smCount = prop.multiProcessorCount;

    // Allocate memory for CUDA copies of maxOnes and rolls
    cudaMalloc(&d_maxOnes, sizeof(int));
    cudaMalloc(&d_rolls, sizeof(int));

    // Initialize these values to 0 on the card
    cudaMemcpy(d_maxOnes, &maxOnes, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rolls, &rolls, sizeof(int), cudaMemcpyHostToDevice);

    // Set block size, allocate blocks, run kernel
    int blockSize = 1024;
    
    int maxActiveBlocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, sim_rolls, blockSize, 0);
    int numBlocks = smCount * maxActiveBlocks;
    //New timing method using cuda events
    float totalTime=0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    sim_rolls<<<numBlocks, blockSize>>>(d_maxOnes, d_rolls, time(NULL));
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&totalTime, start, stop);
    
    // Copy the result back to system memory, now that the CUDA program is over
    cudaMemcpy(&maxOnes, d_maxOnes, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&rolls, d_rolls, sizeof(int), cudaMemcpyDeviceToHost);

    // Report the important numbers
    std::cout << "Highest Ones Roll: " << maxOnes << std::endl;
    std::cout << "Number of Roll Sessions: " << rolls << std::endl;
    std::cout << totalTime << "ms" << std::endl;
    // Never malloc without a free
    cudaFree(d_maxOnes);
    cudaFree(d_rolls);

    return 0;
}
