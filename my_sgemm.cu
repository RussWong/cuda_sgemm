#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define OFFSET(row, col, width) (row * width + col)
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&pointer)[0])
template<
    const int BLOCK_SIZE_M,//128
    const int BLOCK_SIZE_K,//8,every BLOCK_SIZE_M x BLOCK_SIZE_K load into shared mem
    const int BLOCK_SIZE_N,//128
    const int THREAD_SIZE_Y,//8
    const int THREAD_SIZE_X,//8
    const bool USE_PREFETCH>
__global__ void Sgemm(
    float * __restrict__ A,
    float * __restrict__ B,
    float * __restrict__ C, 
    const int M,
    const int N,
    const int K
){
    // block
    int bx = blockIdx.x;
    int by = blockIdx.y;
    // thread
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // other params
    /// num of theads 
    const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;
    /// thread id in cur block
    const int tid = ty * THREAD_X_PER_BLOCK + tx;
    /// shared mem
    __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];
    /// reg for C A B
    float a_reg[2][THREAD_SIZE_Y];
    float b_reg[2][THREAD_SIZE_X];
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
    /// tmp reg for A B
    const int num_glob_tmp_a = BLOCK_SIZE_M * BLOCK_SIZE_K / (THREAD_NUM_PER_BLOCK * 4);
    const int num_glob_tmp_b = BLOCK_SIZE_K * BLOCK_SIZE_N / (THREAD_NUM_PER_BLOCK * 4);
    float tmp_a_reg[4*num_glob_tmp_a];
    float tmp_b_reg[4*num_glob_tmp_b];
    // params for prefetch to shared (key!!!)
    /// num of threads per row
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;
    /// row number and col number that needs to be loaded by this thread(key!!!!)
    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;
    const int A_TILE_COL = (tid % A_TILE_THREAD_PER_ROW) * 4;
    const int B_TILE_COL = (tid % B_TILE_THREAD_PER_ROW) * 4;
    /// row stride that thread use to load mutiple rows of a tile, finally totally prefetch BLOCK_SIZE_M rows and BLOCK_SIZE_K rows
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;
    // prefetch first big(block_size_k stride K) and small(1 stride block_size_k) iter's data before big iter, including global -> tmp reg -> shared mem -> reg
    /// A global->shared, including transpose, every thread here fetch two times totally 8 floats into As
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE){
        int ldg_index = i/A_TILE_ROW_STRIDE * 4
        FETCH_FLOAT4(tmp_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
            by * BLOCK_SIZE_M + A_TILE_ROW_START +  i,
            A_TILE_COL,
            K
        )]);
        As[0][A_TILE_COL][A_TILE_ROW_START + i]=tmp_a_reg[ldg_index];
        As[0][A_TILE_COL+1][A_TILE_ROW_START + i]=tmp_a_reg[ldg_index+1];
        As[0][A_TILE_COL+2][A_TILE_ROW_START + i]=tmp_a_reg[ldg_index+2];
        As[0][A_TILE_COL+3][A_TILE_ROW_START + i]=tmp_a_reg[ldg_index+3];
    }
    /// B global->shared
    // for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE){
    //     FETCH_FLOAT4(Bs[0][B_TILE_ROW_START+i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
    //         B_TILE_ROW_START +  i,
    //         B_TILE_COL + bx * BLOCK_SIZE_N,
    //         N
    //     )]);
    // }
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE){
        int ldg_index = i/B_TILE_ROW_STRIDE * 4;
        FETCH_FLOAT4(tmp_b_reg[ldg_index]) = FETCH_FLOAT4(B[OFFSET(
            B_TILE_ROW_START + i,
            BLOCK_SIZE_N * bx + B_TILE_COL,
            N)]);
        Bs[0][B_TILE_COL][B_TILE_ROW_START + i]=tmp_b_reg[ldg_index];
    }
    __syncthreads();
    /// prefetch first!! small iter, A and B shared -> reg
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4){
        FETCH_FLOAT4(a_reg[0][thread_y]) = FETCH_FLOAT4(As[0][0][ty*THREAD_SIZE_Y+thread_y]);
    }
    #pragma unroll
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4){
        FETCH_FLOAT4(b_reg[0][thread_x]) = FETCH_FLOAT4(Bs[0][0][tx*THREAD_SIZE_X+thread_x]);
    }


    int tile_idx = 0;
    //position that write_stage_idx locates represent the corresponding shared mem need be prefetch,
    //position that load_stage_idx locates represent the corresponding shared mem need be read,
    int write_stage_idx = 1;
    // big iter
    do{
        // prefetch next big iter, global -> tmp reg
        tile_idx += BLOCK_SIZE_K;
        if (tile_idx < K){
            for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE){
                int ldg_index = i/A_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(tmp_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
                    by * BLOCK_SIZE_M + A_TILE_ROW_START + i,
                    tile_idx + A_TILE_COL,
                    K)]);
            }
            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE){
                int ldg_index = i/B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(tmp_b_reg[ldg_index]) = FETCH_FLOAT4(B[OFFSET(
                    tile_idx + B_TILE_ROW_START + i,
                    BLOCK_SIZE_N * bx + B_TILE_COL,
                    N)]);
            }
        }
        // 7 small iters
        int load_stage_idx = write_stage_idx ^ 1;//0
        #pragma unroll
        for (int j = 0; j < BLOCK_SIZE_K - 1; ++j){
            // prefetch next small iter, shared mem -> reg
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4){
                FETCH_FLOAT4(a_reg[(j+1)%2][thread_y]) = FETCH_FLOAT4(As[load_stage_idx][j+1][ty*THREAD_SIZE_Y+thread_y])//because are prefetch next smll iter,so the index should be j+1
            }
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4){
                FETCH_FLOAT4(b_reg[(j+1)%2][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx][j+1][tx*THREAD_SIZE_X+thread_x])
            }
            // unroll loop, compute cur thread 8x8
            #pragma unroll
            for (thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y++){
                #pragma unroll
                for (thread_x = 0; thread_x < THREAD_SIZE_X; thread_x++){
                    accum[thread_y][thread_x] += a_reg[j%2][thread_y] * b_reg[j%2][thread_x]
                }
            }
        }
        // prefetch next big iter, tmp reg -> shared mem
        if (tile_idx < K){
            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE){
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                As[write_stage_idx][A_TILE_COL][A_TILE_ROW_START+i] = tmp_a_reg[ldg_index]
                As[write_stage_idx][A_TILE_COL+1][A_TILE_ROW_START+i] = tmp_a_reg[ldg_index+1]
                As[write_stage_idx][A_TILE_COL+2][A_TILE_ROW_START+i] = tmp_a_reg[ldg_index+2]
                As[write_stage_idx][A_TILE_COL+3][A_TILE_ROW_START+i] = tmp_a_reg[ldg_index+3]
            }
            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE){
                int ldg_index = i / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(tmp_b_reg[ldg_index]);
            }
            __syncthreads();
            write_stage_idx ^= 1;
        }
        // 8th small iter, do same thing as what we do before big iter
            // prefetch next big iter, shared to reg
        #pragma unroll
        for (int thread_y=0; thread_y < THREAD_SIZE_Y; thread_y+=4) {
            FETCH_FLOAT4(a_reg[0][thread_y]) = FETCH_FLOAT4(As[load_stage_idx^1][0][THREAD_SIZE_Y * ty + thread_y]);
        }
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
            FETCH_FLOAT4(b_reg[0][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx^1][0][THREAD_SIZE_X * tx + thread_x]);
        }
            // unroll loop, compute cur thread 8x8
        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                accum[thread_y][thread_x] += a_reg[1][thread_y] * b_reg[1][thread_x];
            }
        }
    }while (tile_idx < K);
    // store back to C， accum to C, thread level
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y++)
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x+=4){
            FETCH_FLOAT4(C[OFFSET(
                by * BLOCK_SIZE_M + ty * THREAD_SIZE_Y + thread_y,
                bx * BLOCK_SIZE_N + tx * THREAD_SIZE_X + thread_x,
                N
            )]) = FETCH_FLOAT4(accum[thread_y][thread_x])
        }

}

int main(int argc, char** argv){
    size_t M = 2048;
    size_t K = 2048;
    size_t N = 2048;
    size_t A_bytes = sizeof(float) * M * K;
    size_t B_bytes = sizeof(float) * N * K;
    size_t C_bytes = sizeof(float) * M * N;

    float* GPU_A;
    float* GPU_B;
    float* GPU_C;

    float* CPU_A = (float*)malloc(A_bytes);
    float* CPU_B = (float*)malloc(B_bytes);
    float* CPU_C = (float*)malloc(C_bytes);

    cudaMalloc((void**)&GPU_A, A_bytes);//global mem
    cudaMalloc((void**)&GPU_B, B_bytes);
    cudaMalloc((void**)&GPU_C, C_bytes);
    double avg_ms[2] = {0, 0};//two eles to compare with cublas
    double FLOPS[2] = {0, 0};
    double GEMM_FLOPs = 2.0 * M * N * K;
    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_K = 8;
    const int BLOCK_SIZE_N = 128;
    const int THREAD_SIZE_X = 8;
    const int THREAD_SIZE_Y = 8;
    const bool USE_PREFETCH = false;

    //generate data
    for( int i = 0; i < M * K; i++ ) {
        int row = (i / K);
        int col = (i % K);
        int row_block = row / BLOCK_SIZE_M;
        int col_block = col / BLOCK_SIZE_K;
        if ((row_block * k_block + col_block) % stride == 0) CPU_A[i] = 1;
        else {
            CPU_A[i] = 0;
        }
    } 

    for( int i = 0; i < K * N; i++ ) {
        if ( i >= K * N / 2) CPU_B[i] = 2;
        else {
            CPU_B[i] = 0;
        }
    }
    cudaMemcpy( GPU_A, CPU_A, A_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy( GPU_B, CPU_B, B_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy( GPU_C, CPU_C, C_bytes, cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0;
    int nIter = 200;
    cudaEventRecord(start);
    for (int run = 0 ; run < nIter; run ++ ) {
        dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
        dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
        Sgemm<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X, USE_PREFETCH> 
        <<< dimGrid, dimBlock >>>(GPU_A, GPU_B, GPU_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy( CPU_C, GPU_C, C_bytes, cudaMemcpyDeviceToHost);
    avg_ms[0] = ms / nIter;
    FLOPS[0] = (GEMM_FLOPs * 1.0e-9f) / (avg_ms[0] / 1000.0f);
    printf( "My gemm Performance= %.2f GFLOPS, Time= %.3f msec, Size= %.0f Ops,\n",
        FLOPS[0],
        avg_ms[0],
        GEMM_FLOPs);
        
    cudaFree(GPU_A);
    cudaFree(GPU_B);
    cudaFree(GPU_C);
    
    free(CPU_A);
    free(CPU_B);
    free(CPU_C);
}
