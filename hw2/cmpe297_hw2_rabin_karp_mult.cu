// CMPE297-6 HW2
// CUDA version Rabin-Karp

#include<stdio.h>
#include<iostream>
#include <cuda_runtime.h>


/*ADD CODE HERE: Implement the parallel version of the sequential Rabin-Karp*/
__global__ void 
findIfExistsCu(char* input, int input_length, char* pattern, int pattern_length, int patHash, int* result, unsigned long long* runtime)
{ 
    unsigned long long start_time = clock64();
    int tid = threadIdx.x;  // tid is where the search index will start
    int input_hash = 0;
    for(int k = tid; k < (pattern_length + tid); k++) {
        input_hash = (input_hash * 256 + input[k]) % 997;
    }

    // unable to call "host" functions in gpu (memcmp)
    // if(input_hash == patHash && memcmp(input + tid*sizeof(char), pattern, pattern_length) == 0) {

    if(input_hash == patHash) {
        printf("+++ match at: %i\n", tid);
        result[tid] = 1;
    }
    else { 
        result[tid] = 0; 
    } 
    unsigned long long stop_time = clock64();
    runtime[tid] = (unsigned long long)(stop_time - start_time);
}

int main()
{
    // host variables
    char input[] = "HEABAL AND SOME MORE CHARACTERS";    /*Sample Input*/
    char pattern[] = "AB";      /*Sample Pattern*/
    int patHash = 0;            /*hash for the pattern*/
    int* result;                /*Result array*/
    unsigned long long* runtime;               /*Exection cycles*/
    int pattern_length = 2;     /*Pattern Length*/
    int input_length = 32;       /*Input Length*/

    // device variables
    char* d_input;
    char* d_pattern;
    int* d_result;
    unsigned long long* d_runtime;

    // measure the execution time by using clock() api in the kernel as we did in Lab3
    int runtime_size = (input_length-pattern_length)*sizeof(unsigned long long);
    result = (int*)malloc((input_length-pattern_length)*sizeof(int));
    runtime = (unsigned long long *) malloc(runtime_size);
    memset(runtime, 0, runtime_size);
    
    /*Calculate the hash of the pattern*/
    for (int i = 0; i < pattern_length; i++)
    {
        patHash = (patHash * 256 + pattern[i]) % 997;
    }

    /*ADD CODE HERE: Allocate memory on the GPU and copy or set the appropriate values from the HOST*/
    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void**)&d_input, input_length*sizeof(char));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device input string. (ERR: %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMalloc((void**)&d_pattern, pattern_length*sizeof(char));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device input string. (ERR: %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void**)&d_result, input_length*sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device results. (ERR: %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void**)&d_runtime, runtime_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device runtime variable. (ERR: %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // copying host memory to device now
    err = cudaMemcpy(d_input, input, input_length*sizeof(char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy input string from host to device. (ERR: %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_pattern, pattern, pattern_length*sizeof(char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy pattern from host to device. (ERR: %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /*ADD CODE HERE: Launch the kernel and pass the arguments*/
    int blocksPerGrid = 1;
    int threadsPerBlock = input_length - pattern_length;
    printf("Running kernel functions blocks: %i and threads: %i\n", blocksPerGrid, threadsPerBlock); 
    findIfExistsCu<<<blocksPerGrid, threadsPerBlock>>>(d_input, input_length, d_pattern, pattern_length, patHash, d_result, d_runtime);
        
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch kernel code (ERR: %s)\n", cudaGetErrorString(err));
    }

    /*ADD CODE HERE: Copy the execution times from the GPU memory to HOST Code*/        
    err = cudaMemcpy(runtime, d_runtime, runtime_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy runtime results from device to host. (ERR: %s)\n", cudaGetErrorString(err));
    }
    
    
    /*RUN TIME calculation*/
    unsigned long long elapsed_time = 0;
    for(int i = 0; i < input_length-pattern_length; i++)
        if(elapsed_time < runtime[i])
            elapsed_time = runtime[i];

    printf("Kernel Execution Time: %llu cycles\n", elapsed_time);
    printf("Total cycles: %d \n", elapsed_time);
    printf("Kernel Execution Time: %d cycles\n", elapsed_time);

    
    /*ADD CODE HERE: COPY the result and print the result as in the HW description*/
    err = cudaMemcpy(result, d_result, input_length*sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to retrieve result from device to host. (ERR: %s)\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }

    printf("\nRESULTS\n"); 
    printf("-----------\n"); 
    printf("Input string = %s\n", input); 
    printf("Pattern      = %s\n", pattern); 
    for (int i = 0; i < input_length; i++) { 
        printf("Pos: %i Result: %i\n", i, result[i]);
    }

    cudaFree(d_input);
    cudaFree(d_pattern);
    cudaFree(d_runtime);
    
    return 0;
}

