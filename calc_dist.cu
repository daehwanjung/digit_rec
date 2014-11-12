/*
 * Proj 3-2 SKELETON
 */

#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include "utils.h"

/* Transposes the square array ARR. */
__global__ void transposeKernel(float *out, float *arr, int width) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i < width && j < width)
		out[(i * width) + j] = arr[(j * width) + i];
}

/* Rotates the square array ARR by 90 degrees counterclockwise. */
__global__ void rotateKernel(float *out, float *arr, int width) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i < width && j < width)
		out[(i * width) + j] = arr[(j * width) + width - i - 1];
}

__global__ void calc_min_dist_kernel(float *scratch, float *image, int i_width, float *temp, int t_width, int k, int l) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i < t_width && j < t_width) {
		float x = temp[(i * t_width) + j] - image[(i + k) * i_width + j + l];
		scratch[(i * t_width) + j] = x * x;
	}
}

__global__ void reduceKernel(float *arr, unsigned int len, unsigned int level) {
    unsigned int threadIndex = (level * 2) * (blockIdx.x * blockDim.x + threadIdx.x);
    if (threadIndex + level < len)
    	arr[threadIndex] += arr[threadIndex + level];
}

/* Returns the squared Euclidean distance between TEMPLATE and IMAGE. The size of IMAGE
 * is I_WIDTH * I_HEIGHT, while TEMPLATE is square with side length T_WIDTH. The template
 * image should be flipped, rotated, and translated across IMAGE.
 */
float calc_min_dist(float *image, int i_width, int i_height, float *temp, int t_width) {
    // float* image and float* temp are pointers to GPU addressible memory
    // You MAY NOT copy this data back to CPU addressible memory and you MAY 
    // NOT perform any computation using values from image or temp on the CPU.
    // The only computation you may perform on the CPU directly derived from distance
    // values is selecting the minimum distance value given a calculated distance and a 
    // "min so far"

    float *scratch;
    float seq = FLT_MAX;
    float min_dist = FLT_MAX;
    unsigned int len = t_width * t_width;
    size_t size = t_width * t_width * sizeof(float);
    CUDA_SAFE_CALL(cudaMalloc(&scratch, size));

    int threads_per_block = 512;
    int gridx = t_width / 256;
    int gridy = t_width / 2;
    if (t_width < threads_per_block) {
    	gridx = 2;
    	gridy = 256;
    }
    dim3 dim_blocks_per_grid(gridx, gridy);
    dim3 dim_threads_per_block(256, 2, 1);

    for (int i = 0; i < 8; i++) {

    	if (i == 4) {
    		transposeKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>(scratch, temp, t_width);
    		cudaThreadSynchronize();
    		CUT_CHECK_ERROR("");
			CUDA_SAFE_CALL(cudaMemcpy(temp, scratch, size, cudaMemcpyDeviceToDevice));
    	} else if (i > 0) {
    		rotateKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>(scratch, temp, t_width);
    		cudaThreadSynchronize();
    		CUT_CHECK_ERROR("");
			CUDA_SAFE_CALL(cudaMemcpy(temp, scratch, size, cudaMemcpyDeviceToDevice));
    	}

    	for (int j = 0; j <= i_width - t_width; j++) {
    		for (int k = 0; k <= i_height - t_width; k++) {

    			calc_min_dist_kernel<<<dim_blocks_per_grid, dim_threads_per_block>>>(scratch, image, i_width, temp, t_width, j, k);
    			cudaThreadSynchronize();
    			CUT_CHECK_ERROR("");

				int blocks_per_grid = (len/threads_per_block)/2;
    			unsigned int level = 1;

    			while (level != len) { 

					dim3 dim_blocks_per_grid(blocks_per_grid, 1);
    				dim3 dim_threads_per_block(threads_per_block, 1, 1);

					reduceKernel<<<dim_blocks_per_grid, dim_threads_per_block>>>(scratch, len, level);
					cudaThreadSynchronize();
        			CUT_CHECK_ERROR("");

					level *= 2;

					blocks_per_grid = blocks_per_grid / 2;
					if (blocks_per_grid == 0) blocks_per_grid = 1;

    			}

    			CUDA_SAFE_CALL(cudaMemcpy(&seq, scratch, sizeof(float), cudaMemcpyDeviceToHost));

    			if (min_dist > seq) min_dist = seq;
    		}
    	}

    }

    CUDA_SAFE_CALL(cudaFree(scratch));

    return min_dist;
}
