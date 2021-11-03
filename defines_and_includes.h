#ifndef MGOMEZ4_DEFINES_H
#define MGOMEZ4_DEFINES_H

#include <vector>
#include <math.h>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <algorithm>
#include <assert.h>
#include <string.h>
#include <stdlib.h>


using namespace std;
/*
* CONSTANTS AND GLOBALS
*/
#define EPOCHS      20
#define N_NEURONS   1024
#define PIXELS      784
#define BATCHSIZE   100
#define LEARNINGRATE    0.001
#define N_IMAGES 60000
#define N_TEST_IMAGES 10000
#define CHECKPOINT 0
#define NANCHECK 1
#define TIMER 1
#define index_flattened(array, i, j,M) array[i*M+j]
default_random_engine gen;
uniform_real_distribution<float> d;

// loss is mean square error
#endif