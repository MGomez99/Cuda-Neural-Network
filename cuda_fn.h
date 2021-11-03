#ifndef NN_NN_cuda_H_MGOMEZ4
#define NN_NN_cuda_H_MGOMEZ4


#define gpu_assert(rv) gpu_assert_h((rv), __FILE__, __LINE__)
void
gpu_assert_h(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
static_assert(N_NEURONS%1024 == 0, "N_NEURONS must be multiple of 1024");

/*
* DEVICE
*/

__global__ void g_TRANSPOSE_MATRIX(float *A, float* AT, int N, int M)
{
    // height and width correspond to the resulting matrix
    int ROW = blockIdx.y * 32 + threadIdx.y;
    int COL = blockIdx.x * 32 + threadIdx.x;
   
   if (ROW < N && COL < M)
   {
       //printf("for j,i = %d,%d\t%f is moving to %d, %d\n", COL, ROW, A[(COL*N)+ROW], ROW, COL);
       AT[(ROW*M)+COL] = A[(COL*N)+ROW]; 
   }
}

__global__ void g_MULT_MATRIX(float* A, float* B, float* C, int ARows, int BCols, int commonDim)
{
    // N = Rows of A and Rows of C
    // M = Cols of B and Cols of C
    // K is Cols of A and Rows of B

    float value = 0;

    int ROW = blockIdx.y*32 + threadIdx.y;
    int COL = blockIdx.x*32 + threadIdx.x;

    for (int k = 0; k < (32 + commonDim - 1)/32; k++) {

        for (int n = 0; n < 32; ++n)
            if ((k*32 + n < commonDim && ROW < ARows) && (k*32 + n < commonDim && COL < BCols))
                value += A[ROW*commonDim + k*32 + n] * B[(k*32 + n)*BCols + COL];

    }

    if (ROW < ARows && COL < BCols){
        int index = ((blockIdx.y * blockDim.y + threadIdx.y)*BCols)+(blockIdx.x*blockDim.x)+threadIdx.x;
        C[index]=value;
    }



}__global__ void g_HADAMARD_PROD_MATRIX(float* A, float* B, float* C, int N, int M)
{


    int ROW = blockIdx.y * 32 + threadIdx.y;
    int COL = blockIdx.x * 32 + threadIdx.x;

    if (ROW < N && COL < M)
    {
        C[ROW * M + COL] =  A[ROW * M + COL] *  B[ROW * M + COL];
    }


}
__global__ void g_SCALE_MATRIX(float *A, float *RESULT, float alpha, int N, int M)
{
    // A AND RESULT ARE N x M
    int ROW = blockIdx.y * 32 + threadIdx.y;
    int COL = blockIdx.x * 32 + threadIdx.x;
    if (ROW < N && COL < M)
    {
        RESULT[ROW * M + COL] = alpha * A[ROW * M + COL];
        
    }
}
__global__ void g_RELU_MATRIX(float *A, float *RESULT, int N, int M)
{
    int ROW = blockIdx.y * 32 + threadIdx.y;
    int COL = blockIdx.x * 32 + threadIdx.x;
    if (ROW < N && COL < M)
    {
        
        if (A[ROW * M + COL] < 0.f) {
            RESULT[ROW * M + COL] = 0.f;
        }
        else{
            RESULT[ROW * M + COL] =  A[ROW * M + COL];
        }
    }
}

__global__ void g_DERIV_RELU_MATRIX(float *A, float *RESULT, int N, int M)
{
    int ROW = blockIdx.y * 32 + threadIdx.y;
    int COL = blockIdx.x * 32 + threadIdx.x;
    if (ROW < N && COL < M)
    {
        if (A[ROW * M + COL] <= 0) RESULT[ROW * M + COL] = 0;
        else RESULT[ROW * M + COL] = 1;
    }
}

__global__ void g_SOFTMAX_MATRIX(float *A, float *RESULT, int N, int M)
{
    int ROW = blockIdx.y * 32 + threadIdx.y;
    int COL = blockIdx.x * 32 + threadIdx.x;

    if (ROW < N && COL < M)
    {
        
        float max = A[0 * M + COL]; // set the max as the first element
        for(int i = 1; i < N; i++){
            if(A[i * M + COL] > max) max = A[i * M + COL]; // find actual max
        }

        float sum = 0.f;
        for(int i = 0; i < N; i++){
            sum += expf(A[i * M + COL] - max); 
        }
        RESULT[ROW * M + COL] = expf(A[ROW * M + COL] - max)/sum;

        // float sum = 0.f;
        // for(int i = 0; i < N; i++){
        //     sum += expf(A[i * M + COL]); 
        // }
        // RESULT[ROW * M + COL] = expf(A[ROW * M + COL])/sum;


    }
}

__global__ void g_ROW_SUM_MATRIX(float *A, float *RESULT,int N, int M)
{
    int ROW = blockIdx.y * 32 + threadIdx.y;
    int COL = blockIdx.x * 32 + threadIdx.x;

    if (ROW < N && COL < M)
    {
        
        float sum = 0.f; // set the max as the first element
        for(int i = 0; i < M; i++){
            sum += A[ROW*M + i]; // assumes A and Result will have same columns
        }

        RESULT[ROW] = sum; // RESULT is always N x 1

    }
}

__global__ void g_ADD_SCALAR_MATRIX(float *A, float *RESULT, float alpha, int N, int M)
{
    int ROW = blockIdx.y * 32 + threadIdx.y;
    int COL = blockIdx.x * 32 + threadIdx.x;
    
    if (ROW < N && COL < M)
    {
        RESULT[ROW * M + COL] = alpha + A[ROW * M + COL];
    }
}
__global__ void g_ADD_VECTOR_MATRIX(float *A,  float* B, float *RESULT, int N, int M)
{
    // B MUST BE A N * 1 MATRIX 
    int ROW = blockIdx.y * 32 + threadIdx.y;
    int COL = blockIdx.x * 32 + threadIdx.x;
    
    if (ROW < N && COL < M)
    {
        RESULT[ROW * M + COL] = B[ROW] + A[ROW * M + COL];
    }
}

__global__ void g_SUB_MATRIX(float *A,  float* B, float *RESULT, int N, int M)
{
    // A - B
    int ROW = blockIdx.y * 32 + threadIdx.y;
    int COL = blockIdx.x * 32 + threadIdx.x;
    
    if (ROW < N && COL < M)
    {
        RESULT[ROW * M + COL] = A[ROW * M + COL] - B[ROW * M + COL];
    }
}

/*
HOST
*/
void TRANSPOSE_MATRIX(float *A, float* AT, int N, int M)
{
    // A is M by N AT IS N by M,
    int BLOCK_SIZE = 32;

    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dim_grid;
    dim_grid.x = (M + dim_block.x - 1)/dim_block.x;
    dim_grid.y = (N + dim_block.y - 1)/dim_block.y;
    g_TRANSPOSE_MATRIX<<<dim_grid, dim_block>>>(A, AT, N, M);
    cudaDeviceSynchronize();
}

void MULT_MATRIX(float *A, float *B, float *C, int N, int K, int M)
{
    // N = Rows of A and Rows of C
    // K is Cols of A and Rows of B
    // M = Cols of B and Cols of C
    // FOR DOT PROD, A MUST BE ROW, B IS COL
    int BLOCK_SIZE = 32;

    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dim_grid;
    dim_grid.x = (M + dim_block.x - 1)/dim_block.x;
    dim_grid.y = (N + dim_block.y - 1)/dim_block.y;

    g_MULT_MATRIX<<<dim_grid, dim_block>>>(A, B, C, N, M, K);
    cudaDeviceSynchronize();
}
void HADAMARD_PRODUCT_MATRIX(float *A, float *B, float *C, int N, int M)
{
    // N = Rows of A and Rows of C
    // M = Cols of B and Cols of C
    // K is Cols of A and Rows of B
    // FOR DOT PROD, A MUST BE ROW, B IS COL
    int BLOCK_SIZE = 32;

    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dim_grid;
    dim_grid.x = (M + dim_block.x - 1)/dim_block.x;
    dim_grid.y = (N + dim_block.y - 1)/dim_block.y;

    g_HADAMARD_PROD_MATRIX<<<dim_grid, dim_block>>>(A, B, C, N, M);
    cudaDeviceSynchronize();
}

void SCALE_MATRIX(float *A, float *RESULT, float alpha, int N, int M)
{
    // A AND RESULT ARE N x M
    int BLOCK_SIZE = 32;

    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dim_grid;
    dim_grid.x = (M + dim_block.x - 1)/dim_block.x;
    dim_grid.y = (N + dim_block.y - 1)/dim_block.y;

    g_SCALE_MATRIX<<<dim_grid, dim_block>>>(A, RESULT, alpha, N, M);
    cudaDeviceSynchronize();
}


void RELU_MATRIX(float *A, float *RESULT,int N, int M)
{
    int BLOCK_SIZE = 32;

    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dim_grid;
    dim_grid.x = (M + dim_block.x - 1)/dim_block.x;
    dim_grid.y = (N + dim_block.y - 1)/dim_block.y;

    g_RELU_MATRIX<<<dim_grid, dim_block>>>(A, RESULT, N, M);
    cudaDeviceSynchronize();
}

void DERIV_RELU_MATRIX(float *A, float *RESULT,int N, int M)
{
    int BLOCK_SIZE = 32;

    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dim_grid;
    dim_grid.x = (M + dim_block.x - 1)/dim_block.x;
    dim_grid.y = (N + dim_block.y - 1)/dim_block.y;

    g_DERIV_RELU_MATRIX<<<dim_grid, dim_block>>>(A, RESULT, N, M);
    cudaDeviceSynchronize();
}

void SOFTMAX_MATRIX(float *A, float *RESULT, int N, int M)
{
    int BLOCK_SIZE = 32;

    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dim_grid;
    dim_grid.x = (M + dim_block.x - 1)/dim_block.x;
    dim_grid.y = (N + dim_block.y - 1)/dim_block.y;
    g_SOFTMAX_MATRIX<<<dim_grid, dim_block>>>(A, RESULT,N, M);
    cudaDeviceSynchronize();
}

void ROW_SUM_MATRIX(float *A, float *RESULT, int N, int M)
{
    // M is the number of columns of A
    // RESULT is N x 1
    int BLOCK_SIZE = 32;

    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dim_grid;
    dim_grid.x = (M + dim_block.x - 1)/dim_block.x;
    dim_grid.y = (N + dim_block.y - 1)/dim_block.y;
    g_ROW_SUM_MATRIX<<<dim_grid, dim_block>>>(A, RESULT, N, M);
    cudaDeviceSynchronize();
}

void ADD_SCALAR_MATRIX(float *A, float *RESULT, float alpha, int N, int M)
{
    int BLOCK_SIZE = 32;

    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dim_grid;
    dim_grid.x = (M + dim_block.x - 1)/dim_block.x;
    dim_grid.y = (N + dim_block.y - 1)/dim_block.y;

    g_ADD_SCALAR_MATRIX<<<dim_grid, dim_block>>>(A, RESULT, alpha, N, M);
    cudaDeviceSynchronize();
}


void ADD_VECTOR_MATRIX(float *A,  float* B, float *RESULT, int N, int M)
{
    int BLOCK_SIZE = 32;

    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dim_grid;
    dim_grid.x = (M + dim_block.x - 1)/dim_block.x;
    dim_grid.y = (N + dim_block.y - 1)/dim_block.y;

    g_ADD_VECTOR_MATRIX<<<dim_grid, dim_block>>>(A, B, RESULT, N, M);
    cudaDeviceSynchronize();
}

void SUB_MATRIX(float *A,  float* B, float *RESULT, int N, int M)
{ 
    // RESULT = A - B
    // ALL ARRAYS MUST BE N x M
    int BLOCK_SIZE = 32;

    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dim_grid;
    dim_grid.x = (M + dim_block.x - 1)/dim_block.x;
    dim_grid.y = (N + dim_block.y - 1)/dim_block.y;

    g_SUB_MATRIX<<<dim_grid, dim_block>>>(A, B, RESULT, N, M);
    cudaDeviceSynchronize();
}


/* CUDA util */

float* flatten_vector(vector<vector<float>>  v, int N, int M)
{
    float *rv = (float *)malloc(N * M * sizeof(float)); // Assuming all rows have the same size
    for (unsigned i = 0; i < N; i++)
        memcpy(rv + (M * i), v[i].data(), M * sizeof(float));
    return rv;
}
float* flatten_vector_range_cols(vector<vector<float>>  v, int start, int end, int N, int M)
{
    // copies over a range of columns from v to an array on the heap
    assert((end - start) == M);
    float *rv = (float *)malloc(N * M * sizeof(float)); // Assuming all rows have the same size
    cout<<endl;
    int ptr = 0;
    for(int i = 0; i < N; i++){
        for (int j = start; j < end; j++){
            rv[ptr] = v[i][j];
            ptr++;
            
        }
    }
    cout<<endl;
    return rv;
}
float* flatten_vector_range_rows(vector<vector<float>>  v, int start, int end, int N, int M)
{
    // copies over a range of rows from v to an array on the heap
    assert((end-start)==N);
    float *rv = (float *)malloc(N * M * sizeof(float)); // Assuming all rows have the same size
    for (unsigned i = 0; i < N; i++)
        memcpy(rv + (M * i), v[i+start].data(), M * sizeof(float));
    return rv;
}
float* host_vec_to_device_PTR(vector<vector<float>> v, int N, int M){
    /* flattens the 2D vector v of given size and copies it to the device */
    //assumes equal row length

    float *g_ptr;
    cudaError_t rv_ce = cudaMalloc(&g_ptr, N * M * sizeof(float));
    for (int i = 0; i < N; i++) // copy row by row
    {
        rv_ce = cudaMemcpy(g_ptr + (i * M), v[i].data(), M * sizeof(float), cudaMemcpyHostToDevice); // 
        gpu_assert(rv_ce);
    }
    return g_ptr;
}
float * generate_empty_device_PTR(int rows, int cols){
    /* allocates an empty pointer of the given size on the device*/
    float *g_ptr;
    cudaError_t rv_ce = cudaMalloc(&g_ptr, rows * cols * sizeof(float));
    gpu_assert(rv_ce);
    rv_ce=cudaMemset(g_ptr, 0, rows * cols * sizeof(float)); 
    gpu_assert(rv_ce);
    return g_ptr;
}

float*  device_to_host_PTR(float * g_ptr, int rows, int cols){
    /* makes and returns a new pointer on the host from the given device pointer */
    float * retval = (float*) malloc(sizeof(float)*rows*cols);
    cudaError_t rv_ce = cudaMemcpy(retval, g_ptr, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost);
    gpu_assert(rv_ce);
    return retval;

}
void copyFromDevice(float * h_ptr, float * g_ptr, int N, int M){
    /* similar to device_to_host_pointer, except it requires a host pointer that is already allocated*/
    cudaError_t rv_ce  = cudaMemcpy(h_ptr, g_ptr, sizeof(float) * N * M, cudaMemcpyDeviceToHost);
    gpu_assert(rv_ce);
}
float *array_to_device_PTR(float *h_ptr, int N, int M)
{
    /* generates a device pointer with memory initalized to that of the provided host pointer*/
    float* g_ptr;
    cudaError_t rv_ce = cudaMalloc(&g_ptr, N * M * sizeof(float));
    gpu_assert(rv_ce);
    rv_ce = cudaMemcpy(g_ptr, h_ptr, sizeof(float) * N * M, cudaMemcpyHostToDevice);
    gpu_assert(rv_ce);
    return g_ptr;
}



#endif