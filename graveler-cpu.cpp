#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <cmath>
#include <random>
#include <iostream>
#include <unistd.h>
#include <chrono>
#include <future>
#include <vector>
//Requires libpcg-cpp-dev (pcg-random.org)
#include "pcg_random.hpp"

#define ROUNDS 1000000
#define ROLLS 231
uint64_t setRounds = ROUNDS;
using namespace std;
int cores = thread::hardware_concurrency();
pcg_extras::seed_seq_from<random_device> seed_source;
vector<pcg128_once_insecure> rng(cores);

uint64_t rolls = 0;
uint8_t maxOnes = 0;
vector<uint8_t> sharedMax(cores);
vector<future<void>> m_Futures;
void sim_rand(int thr){
    sharedMax[thr] = 0;
    while(rolls < setRounds - thr){
        uint8_t ones = 0;
        uint8_t curRolls = 0;
        unsigned __int128 roll = 0;
        for (uint8_t i = 0; i < 4; ++i){
            roll = rng[thr]();
            for (uint8_t j = 0; j < 64; j++){
                if(curRolls < 231){
                    uint8_t shift = (j * 2);
                    uint64_t currentRoll = (roll >> shift) & 0x03;
                    if(currentRoll == 0) ones++;
                    curRolls++;
                }
            }
        }
        sharedMax[thr] = max(ones, sharedMax[thr]);
        rolls++;
    }
    if(thr == 0){
        for(int n = 0; n < cores; n++){
            maxOnes = max(sharedMax[n], maxOnes);
        }
    }
}

int main(int argc, char **argv){
    switch(getopt(argc, argv, "n:")){
        case 'n':
            setRounds = atoll(optarg);
        default:
            break;
    }
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();

    for(int i=0; i < cores; i++){
        rng[i].seed(seed_source);
        m_Futures.push_back(async(launch::async, sim_rand, i));
    }
    while(rolls < setRounds){
        usleep(1);
    }
    auto t2 = high_resolution_clock::now();
    printf("Highest Ones Roll: %d\nNumber of Roll Sessions: %d\n", maxOnes, rolls);
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    printf("Time: %dms\n", ms_int.count());
    return 0;
}
