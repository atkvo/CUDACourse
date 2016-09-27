// CMPE297-6 HW2
// CUDA version Rabin-Karp

#include<stdio.h>
#include<iostream>
#include <cuda_runtime.h>


__global__ void 
findIfExistsCu(char* input, int input_length, char* pattern, int numberOfPatterns, int* patternLengths, int *startstops, int* patHash, int* result, unsigned long long* runtime)
{ 
    unsigned long long start_time = clock64();
    int patIndex = blockIdx.x;   // use this to index the pattern 
    int tid = threadIdx.x;  // tid is where the search index will start
    int input_hash = 0;

    if (tid < (input_length - patternLengths[patIndex])) {
        for(int k = tid; k < (patternLengths[patIndex] + tid); k++) {
            input_hash = (input_hash * 256 + input[k]) % 997;
        }
        if(input_hash == patHash[patIndex]) {
            result[patIndex] |= 1;
        }
        else { 
            result[patIndex] |= 0; 
        } 
        unsigned long long stop_time = clock64();
        runtime[tid] = (unsigned long long)(stop_time - start_time);
    }
}

int main()
{
    // host variables
    char input[] = "HEABAL AND SOME MORE CHARACTERS";    /*Sample Input*/
    int* patHash;                /*hash for the pattern*/
    int* result;                 /*Result array*/
    unsigned long long* runtime; /*Exection cycles*/
    int input_length = 31;       /*Input Length*/

    const int numberOfPatterns = 5;
    char* patterns2d[] = {          /*2D array patterns*/
        "ABR", "OME", "DNA", "RACT", "AND SOME"
    };
    int lengths[numberOfPatterns] = { 3, 3, 3, 4, 8};
    int startstops[numberOfPatterns*2] = { 
        0, 2,
        3, 5, 
        6, 8,
        9, 12,
        13, 20
    };
    patHash = (int*)malloc(sizeof(int)*numberOfPatterns);

    // flatten the 2d character array
    int totalLengths = 0;
    for(int i = 0; i < numberOfPatterns; i++) {
        totalLengths += lengths[i];
    }
    char * patternsFlat;
    patternsFlat  = (char*)malloc(sizeof(char)*totalLengths);
    for(int i = 0; i < numberOfPatterns; i++) {
        for(int j = 0; j < lengths[i]; j++) {
            patternsFlat[startstops[i*2] + j] = patterns2d[i][j];
            // printf("%c", patterns2d[i][j]);
        }
        printf("\n");
    }
    // printf("Done flattening 2d array\n");
    // done flattening 2d array into patternsFlat
    

    // device variables
    char* d_input;
    char* d_pattern;
    int* d_result;
    int* d_startstops;
    int* d_patternLengths;
    int* d_patHash;
    unsigned long long* d_runtime;

    // measure the execution time by using clock() api in the kernel as we did in Lab3
    int runtime_size = (input_length*numberOfPatterns)*sizeof(unsigned long long);
    result = (int*)malloc((input_length*numberOfPatterns)*sizeof(int));
    runtime = (unsigned long long *) malloc(runtime_size);
    memset(runtime, 0, runtime_size);
    
    /*Calculate the hash of the pattern*/
    for (int i = 0; i < numberOfPatterns; i++) {
        for (int j = 0; j < lengths[i]; j++)
        {
            patHash[i] = (patHash[i] * 256 + patterns2d[i][j]) % 997;
        }
    }

    /*ADD CODE HERE: Allocate memory on the GPU and copy or set the appropriate values from the HOST*/
    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void**)&d_input, input_length*sizeof(char));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device input string. (ERR: %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMalloc((void**)&d_pattern, totalLengths*sizeof(char));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device pattern 1d string. (ERR: %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void**)&d_patternLengths, numberOfPatterns*sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device pattern len array. (ERR: %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void**)&d_startstops, numberOfPatterns*2*sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device pattern start stops array. (ERR: %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void**)&d_patHash, numberOfPatterns*sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device pattern hash array. (ERR: %s)\n", cudaGetErrorString(err));
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

    err = cudaMemcpy(d_pattern, patternsFlat, totalLengths*sizeof(char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy pattern from host to device. (ERR: %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_patternLengths, lengths, numberOfPatterns*sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy pattern lengths from host to device. (ERR: %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_startstops, startstops, numberOfPatterns*2*sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy pattern start stops from host to device. (ERR: %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_patHash, patHash, numberOfPatterns*sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy pattern hashes from host to device. (ERR: %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /*ADD CODE HERE: Launch the kernel and pass the arguments*/
    int blocksPerGrid = numberOfPatterns;
    // int threadsPerBlock = input_length - pattern_length;
    int threadsPerBlock = input_length; // use the maximum amount of threads needed per pattern just in case
    printf("Running kernel functions blocks: %i and threads: %i\n", blocksPerGrid, threadsPerBlock); 
    // findIfExistsCu<<<blocksPerGrid, threadsPerBlock>>>(d_input, input_length, d_pattern, pattern_length, patHash, d_result, d_runtime);
    findIfExistsCu<<<blocksPerGrid, threadsPerBlock>>>(d_input, input_length, d_pattern, numberOfPatterns, d_patternLengths, d_startstops, d_patHash, d_result, d_runtime);
        
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
    for(int i = 0; i < input_length*numberOfPatterns; i++)
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
    for (int i = 0; i < numberOfPatterns; i++) { 
        if(result[i] == 1) {
            printf("Pattern FOUND:    %s\n", patterns2d[i]);
        }
        else {
            printf("Pattern MISSING:  %s\n", patterns2d[i]);
        }
    }

    cudaFree(d_input);
    cudaFree(d_pattern);
    cudaFree(d_runtime);
    cudaFree(d_patternLengths);
    cudaFree(d_patHash);
    cudaFree(d_startstops);
    cudaFree(d_result);
    
    return 0;
}

