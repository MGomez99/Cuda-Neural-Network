#ifndef NN_NN_H_MGOMEZ4
#define NN_NN_H_MGOMEZ4
#include "defines_and_includes.h"
#include "util.h"
#include "cuda_fn.h"



void forwardpass(float *X, int start, float *W1, float *W2, float *B1, float *B2, float *Z1, float *Z2, float *A1, float *A2, int SIZE)
{
    /*
    X is SIZE x PIXELS (we ignore last column)
    =>X_T is PIXELS x SIZE
    W1 is N_NEURONS * PIXELS
    B1 is N_NEURONS * 1 
    W2 is 10 * N_NEURONS [10 is number of output layer neurons]
    B2 is 10 * 1 
    Z1 is N_NEURONS * SIZE
    A1 is N_NEURONS * SIZE
    Z2 is 10 * SIZE
    A2 is 10 * SIZE

    */
    if (CHECKPOINT)
        cout << "\t\tSTARTING FORWARD PASS..." << endl;
    /* CALCULATING Z1 = W1@X + B1 */
    float *g_X = array_to_device_PTR(X, SIZE, PIXELS);
    float *g_X_T = generate_empty_device_PTR(PIXELS, SIZE);
    TRANSPOSE_MATRIX(g_X, g_X_T, PIXELS, SIZE);
    float *g_W1 = array_to_device_PTR(W1, N_NEURONS, PIXELS);
    // we use XT for calculations
    float *g_W1T_DOT_X_T = generate_empty_device_PTR(N_NEURONS, SIZE);
    // dot product first (matrix product)
    MULT_MATRIX(g_W1, g_X_T, g_W1T_DOT_X_T, N_NEURONS, PIXELS, SIZE);
    // add B1
    float *g_B1 = array_to_device_PTR(B1, N_NEURONS, 1);
    float *g_Z1 = generate_empty_device_PTR(N_NEURONS, SIZE);
    ADD_VECTOR_MATRIX(g_W1T_DOT_X_T, g_B1, g_Z1, N_NEURONS, SIZE);
    copyFromDevice(Z1, g_Z1, N_NEURONS, SIZE);

    /* CALCULATING A1 = RELU (Z1)*/
    float *g_A1 = generate_empty_device_PTR(N_NEURONS, SIZE);
    RELU_MATRIX(g_Z1, g_A1, N_NEURONS, SIZE);
    copyFromDevice(A1, g_A1, N_NEURONS, SIZE); // copy A1 to host

    /* CALCULATING Z2 = W2@A1 + B2*/
    float *g_W2 = array_to_device_PTR(W2, 10, N_NEURONS);
    // dot product first (matrix product)
    float *g_W2_DOT_A1 = generate_empty_device_PTR(10, SIZE);
    MULT_MATRIX(g_W2, g_A1, g_W2_DOT_A1, 10, N_NEURONS, SIZE);
    // add B2
    float *g_B2 = array_to_device_PTR(B2, 10, 1);
    float *g_Z2 = generate_empty_device_PTR(10, SIZE);
    ADD_VECTOR_MATRIX(g_W2_DOT_A1, g_B2, g_Z2, 10, SIZE);
    copyFromDevice(Z2, g_Z2, 10, SIZE); // copy Z2 to host

    /* CALCULATING A2 = SOFTMAX (Z2) */
    float *g_A2 = generate_empty_device_PTR(10, SIZE);
    SOFTMAX_MATRIX(g_Z2, g_A2, 10, SIZE);
    copyFromDevice(A2, g_A2, 10, SIZE); // copy A1 to host
    cudaFree(g_X);
    cudaFree(g_X_T);
    cudaFree(g_W1);
    cudaFree(g_W1T_DOT_X_T);
    cudaFree(g_B1);
    cudaFree(g_Z1);
    cudaFree(g_A1);
    cudaFree(g_W2);
    cudaFree(g_W2_DOT_A1);
    cudaFree(g_B2);
    cudaFree(g_Z2);
    cudaFree(g_A2);

    if (CHECKPOINT)
        cout << "...DONE!" << endl;
}

void backprop(float *X, float *Ys, int start, float *Z1, float *A1, float *Z2, float *A2, float *W1, float *B1, float *W2, float *B22,
              float *dW1, float *dW2, float *dB1, float *dB2, int SIZE)
{
    /*
    X is SIZE x PIXELS (we ignore last column)
    W1 is N_NEURONS * PIXELS
    B1 is N_NEURONS * 1
    W2 is 10 * N_NEURONS [10 is number of output layer neurons]
    B2 is 10 x 1
    Z1 is N_NEURONS * SIZE
    A1 is N_NEURONS * SIZE
    Z2 is 10 * SIZE
    A2 is 10 * SIZE
    */
    if (CHECKPOINT)
        cout << "\t\tSTARTING BACKPROP..." << endl;
    // CALCULATE dZ2 = A2-Ys for later computations
    float *g_dZ2 = generate_empty_device_PTR(10, SIZE); 
    float *g_A2 = array_to_device_PTR(A2, 10, SIZE);
    float *g_Ys = array_to_device_PTR(Ys, 10, SIZE);
    SUB_MATRIX(g_A2, g_Ys, g_dZ2, 10, SIZE);
    /* 1. CALCULATE dW2*/

    // CALCULATE dW2 = 1/SIZE * dZ2@A1.T
    float *g_dW2 = generate_empty_device_PTR(10, N_NEURONS);
    float *g_A1 = array_to_device_PTR(A1, N_NEURONS, SIZE);
    float *g_A1_T = generate_empty_device_PTR(SIZE, N_NEURONS);
    TRANSPOSE_MATRIX(g_A1, g_A1_T, SIZE, N_NEURONS); // we now have A1.T
    float *g_dZ2_DOT_A1_T = generate_empty_device_PTR(10, N_NEURONS);
    MULT_MATRIX(g_dZ2, g_A1_T, g_dZ2_DOT_A1_T, 10, SIZE, N_NEURONS);
    SCALE_MATRIX(g_dZ2_DOT_A1_T, g_dW2,((1.f)/((float)SIZE)), 10, N_NEURONS); //scale by 1/BATCHSIZE
    copyFromDevice(dW2, g_dW2, 10, N_NEURONS);

    /* 2. CALCULATE dB2 */
    float *g_dZ2_ROW_SUM = generate_empty_device_PTR(10, 1);
    ROW_SUM_MATRIX(g_dZ2, g_dZ2_ROW_SUM, 10, SIZE); // sum along the rows (along nubmer of images)
    float * g_dB2 = generate_empty_device_PTR(10, 1); 
    SCALE_MATRIX(g_dZ2_ROW_SUM, g_dB2, ((1.f)/((float)SIZE)), 10, 1);
    copyFromDevice(dB2, g_dB2, 10, 1);
    /* 3. dW1 = 1/size  * dZ1@X.T*/
    // CALCULATE dZ1
    float *g_dZ1 = generate_empty_device_PTR(N_NEURONS, SIZE); 
    float *g_W2 = array_to_device_PTR(W2, 10, N_NEURONS);
    float *g_W2_T = generate_empty_device_PTR(N_NEURONS, 10);
    TRANSPOSE_MATRIX(g_W2, g_W2_T, N_NEURONS, 10);
    float *g_W2_T_DOT_dZ2 = generate_empty_device_PTR(N_NEURONS, SIZE);
    MULT_MATRIX(g_W2_T, g_dZ2, g_W2_T_DOT_dZ2, N_NEURONS, 10, SIZE);
    float* g_DERIV_RELU_Z1 = generate_empty_device_PTR(N_NEURONS, SIZE);
    float * g_Z1 = array_to_device_PTR(Z1, N_NEURONS, SIZE);
    DERIV_RELU_MATRIX(g_Z1, g_DERIV_RELU_Z1, N_NEURONS, SIZE);
    HADAMARD_PRODUCT_MATRIX(g_W2_T_DOT_dZ2, g_DERIV_RELU_Z1, g_dZ1, N_NEURONS, SIZE); // g_dZ1 is size N_NEURONS x SIZE
    float * g_X = array_to_device_PTR(X, SIZE, PIXELS);
    float * g_dZ1_DOT_X = generate_empty_device_PTR(N_NEURONS, PIXELS);
    MULT_MATRIX(g_dZ1, g_X, g_dZ1_DOT_X, N_NEURONS, SIZE, PIXELS);
    float * g_dW1 = generate_empty_device_PTR(N_NEURONS, PIXELS);
    SCALE_MATRIX(g_dZ1_DOT_X, g_dW1, ((1.f)/((float)SIZE)), N_NEURONS, PIXELS); // g_dW1 is N_NEURONS x PIXELS
    copyFromDevice(dW1, g_dW1, N_NEURONS, PIXELS);
    /* 4. CALCULATE dB1*/
    float * g_dB1 = generate_empty_device_PTR(N_NEURONS, 1);
    float * g_dZ1_ROW_SUM = generate_empty_device_PTR(N_NEURONS, 1);
    ROW_SUM_MATRIX(g_dZ1, g_dZ1_ROW_SUM, N_NEURONS, SIZE);
    SCALE_MATRIX(g_dZ1_ROW_SUM, g_dB1, ((1.f)/((float)SIZE)), N_NEURONS, 1);
    copyFromDevice(dB1, g_dB1, N_NEURONS, 1);
    // Free used memory

    cudaFree(g_dZ2);
    cudaFree(g_A2);
    cudaFree(g_Ys);
    cudaFree(g_dW2);
    cudaFree(g_A1_T);
    cudaFree(g_A1);
    cudaFree(g_dZ2_DOT_A1_T);
    cudaFree(g_dZ2_ROW_SUM);
    cudaFree(g_dB2);
    cudaFree(g_dZ1);
    cudaFree(g_W2_T);
    cudaFree(g_W2);
    cudaFree(g_W2_T_DOT_dZ2);
    cudaFree(g_DERIV_RELU_Z1);
    cudaFree(g_Z1);
    cudaFree(g_X);
    cudaFree(g_dZ1_DOT_X);
    cudaFree(g_dW1);
    cudaFree(g_dB1);
    cudaFree(g_dZ1_ROW_SUM);
    if (CHECKPOINT)
        cout << "...DONE!" << endl;
}


void update_params(float *W1, float *W2, float *B1, float *B2, float *dW1, float *dW2, float *dB1, float *dB2)
{
    float *g_W1 = array_to_device_PTR(W1, N_NEURONS, PIXELS);
    float *g_W2 = array_to_device_PTR(W2, 10, N_NEURONS);
    float *g_B1 = array_to_device_PTR(B1, N_NEURONS, 1);
    float *g_B2 = array_to_device_PTR(B2, 10, 1);

    float *g_dW1 = array_to_device_PTR(dW1, N_NEURONS, PIXELS);
    float *g_dW2 = array_to_device_PTR(dW2, 10, N_NEURONS);
    float *g_dB1 = array_to_device_PTR(dB1, N_NEURONS, 1);
    float *g_dB2 = array_to_device_PTR(dB2, 10, 1);

    float *g_dW1_scaled = generate_empty_device_PTR(N_NEURONS, PIXELS);
    float *g_dW2_scaled = generate_empty_device_PTR(10, N_NEURONS);
    float *g_dB1_scaled = generate_empty_device_PTR(N_NEURONS, 1);
    float *g_dB2_scaled = generate_empty_device_PTR(10, 1);

    SCALE_MATRIX(g_dW1, g_dW1_scaled, LEARNINGRATE, N_NEURONS, PIXELS);
    SCALE_MATRIX(g_dW2, g_dW2_scaled, LEARNINGRATE, 10, N_NEURONS);
    SCALE_MATRIX(g_dB1, g_dB1_scaled, LEARNINGRATE, N_NEURONS, 1);
    SCALE_MATRIX(g_dB2, g_dB2_scaled, LEARNINGRATE, 10, 1);

    float *g_W1_RESULT = array_to_device_PTR(W1, N_NEURONS, PIXELS);
    float *g_W2_RESULT = array_to_device_PTR(W2, 10, N_NEURONS);
    float *g_B1_RESULT = array_to_device_PTR(B1, N_NEURONS, 1);
    float *g_B2_RESULT = array_to_device_PTR(B2, 10, 1);

    SUB_MATRIX(g_W1, g_dW1_scaled, g_W1_RESULT, N_NEURONS, PIXELS);
    SUB_MATRIX(g_W2, g_dW2_scaled, g_W2_RESULT, 10, N_NEURONS);
    SUB_MATRIX(g_B1, g_dB1_scaled, g_B1_RESULT, N_NEURONS, 1);
    SUB_MATRIX(g_B2, g_dB2_scaled, g_B2_RESULT, 10, 1);

    copyFromDevice(W1, g_W1_RESULT, N_NEURONS, PIXELS);
    copyFromDevice(W2, g_W2_RESULT, 10, N_NEURONS);
    copyFromDevice(B1, g_B1_RESULT, N_NEURONS, 1);
    copyFromDevice(B2, g_B2_RESULT, 10, 1);

    cudaFree(g_W1);
    cudaFree(g_W2);
    cudaFree(g_B1);
    cudaFree(g_B2);
    cudaFree(g_dW1);
    cudaFree(g_dW2);
    cudaFree(g_dB1);
    cudaFree(g_dB2);
    cudaFree(g_dW1_scaled);
    cudaFree(g_dW2_scaled);
    cudaFree(g_dB1_scaled);
    cudaFree(g_dB2_scaled);
    cudaFree(g_W1_RESULT);
    cudaFree(g_W2_RESULT);
    cudaFree(g_B1_RESULT);
    cudaFree(g_B2_RESULT);
}

float getAccuracy(float *X, float *X_WITH_SOLN, float *W1, float *W2, float *B1, float *B2, int SIZE)
{
    // X is SIZE x PIXELS
    // X_WITH_SOLN is SIZE x PIXELS + 1
    float *Z1 = (float *)malloc(N_NEURONS * SIZE * sizeof(float));
    float *A1 = (float *)malloc(N_NEURONS * SIZE * sizeof(float));
    float *Z2 = (float *)malloc(10 * SIZE * sizeof(float));
    float *A2 = (float *)malloc(10 * SIZE * sizeof(float));

    forwardpass(X, 0, W1, W2, B1, B2, Z1, Z2, A1, A2, SIZE);
    int correct_guesses = 0;

    for (int image = 0; image < SIZE; image++)
    { // for each image
        bool is_correct = true;
        int index = (int)index_flattened(X_WITH_SOLN, image, PIXELS, (PIXELS + 1));

        float max = index_flattened(A2, index, image, SIZE);


        for (int i = 0; i < 10; i++)
        {


            if (NANCHECK && (index_flattened(A2, i, image, SIZE) < 0.f ||  index_flattened(A2, i, image, SIZE) > 1.f))
            {
                cout << "invalid probability = " << index_flattened(A2, i, image, SIZE) << endl;
                for (int j = 0; j < 10; j++)
                {
                    cout << index_flattened(Z2, j, image, SIZE) << "\t\t\t" << index_flattened(A2, j, image, SIZE) << endl;
                }
                assert(false);
            }
            assert(index_flattened(A2, i, image, SIZE) >= 0.f && index_flattened(A2, i, image, SIZE) <= 1.f);

            // check for max
            if (i != index && index_flattened(A2, i, image, SIZE) > max)
            {
                is_correct = false;
            }
        }
        if (is_correct)
            ++correct_guesses;
    }
    float acc = ((float)(correct_guesses) )/ ((float)SIZE);
    free(Z1);
    free(Z2);
    free(A1);
    free(A2);
    return acc;
}



#endif