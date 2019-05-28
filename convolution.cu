#include<cstdlib>
#include<cstdio>
#include "cudnn.h"
#include "implicit_gemm_kernel.h"

#define ErrChk(code) { Assert((code), __FILE__, __LINE__); }
inline void Assert(cudaError_t  code, const char *file, int line){
	if(code!=cudaSuccess) {
		printf("CUDA Runtime Error: %s:%d:'%s'\n", file, line, cudaGetErrorString(code));
		exit(EXIT_FAILURE);
	}
}
inline void Assert(cudnnStatus_t code, const char *file, int line){
    if (code!=CUDNN_STATUS_SUCCESS){
		printf("cuDNN API Error: %s:%d:'%s'\n", file, line, cudnnGetErrorString(code));
        exit(EXIT_FAILURE);
    }
}

#define KernelErrChk(){\
		cudaError_t errSync  = cudaGetLastError();\
		cudaError_t errAsync = cudaDeviceSynchronize();\
		if (errSync != cudaSuccess) {\
			  printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));\
			  exit(EXIT_FAILURE);\
		}\
		if (errAsync != cudaSuccess){\
			printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));\
			exit(EXIT_FAILURE);\
		}\
}


int main(int argc, char *argv[]){
	
	//convolution parameters
	int N = 32; //batch size
	int C = 1024; //channel
	int H = 24;
	int W = 24;
	
	int K = 16; //number of filter
	int R = 1;
	int S = 1;
	int U = 1;
	int V = 1;

	int pad_h = 0;
	int pad_w = 0;
	
	int dilation = 1;

	int P = (H + 2*pad_h - (((R-1)*dilation) + 1) )/U + 1;
	int Q = (W + 2*pad_w - (((S-1)*dilation) + 1) )/U + 1;

/*
	if (argc != 12){
		printf("Usage: You need to type 11 arguments: N C H W K R S pad_h U P Q\n");
		exit(EXIT_FAILURE);
	}

	int	N = atoi(argv[1]);
	int C = atoi(argv[2]);
	int H = atoi(argv[3]);
	int W = atoi(argv[4]);
	int K = atoi(argv[5]);
	int R = atoi(argv[6]);
	int S = atoi(argv[7]);
	int pad_h = atoi(argv[8]);
	int pad_w = atoi(argv[8]);
	int U = atoi(argv[9]);
	int V = atoi(argv[9]);
	int P = atoi(argv[10]);
	int Q = atoi(argv[11]);

	if (!(R==1 && pad_h==0 && U==1))
		return 1;
*/
	
	//int dilation = 1;
	//prepare data
	float *h_input = (float*) malloc(N*C*H*W*sizeof(float));
	for (int j=0; j<N*C*H*W; ++j)
		h_input[j] = 1.f;

	float *h_filter = (float*) malloc(K*C*R*S*sizeof(float));
	for (int j=0; j<K*C*R*S; ++j)
		h_filter[j] = 1.f;

	float *h_result_cudnn = (float*) malloc(K*P*Q*N*sizeof(float));
	float *h_result_our = (float*) malloc(K*P*Q*N*sizeof(float));

	//cuDNN
	//prepare data
	float *input; //input data
	float *filter; //filter
	float *result_cudnn; //result

	ErrChk(cudaMalloc(&input, N*C*H*W*sizeof(float)));
	ErrChk(cudaMalloc(&filter, K*C*R*S*sizeof(float)));
	ErrChk(cudaMalloc(&result_cudnn, N*K*P*Q*sizeof(float)));
	
	ErrChk(cudaMemcpy(input, h_input, N*C*H*W*sizeof(float), cudaMemcpyHostToDevice));
	ErrChk(cudaMemcpy(filter, h_filter, K*C*R*S*sizeof(float), cudaMemcpyHostToDevice));
	
	float one = 1.0, zero = 0.0;
	size_t size;

	cudnnHandle_t handle;
	ErrChk(cudnnCreate(&handle));


	cudnnTensorDescriptor_t xDesc, yDesc;
	cudnnFilterDescriptor_t filterDesc; // CUDNN_TENSOR_NHWC, CUDNN_TENSOR_NCHW
	ErrChk(cudnnCreateTensorDescriptor(&xDesc));
	ErrChk(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));

	ErrChk(cudnnCreateTensorDescriptor(&yDesc));
	ErrChk(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, K, P, Q));

	ErrChk(cudnnCreateFilterDescriptor(&filterDesc));
	ErrChk(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C, R, S));

	cudnnConvolutionDescriptor_t convDesc;
	ErrChk(cudnnCreateConvolutionDescriptor(&convDesc));
	ErrChk(cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, U, V, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

	cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

	ErrChk(cudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, filterDesc, convDesc, yDesc, algo, (size_t *)&(size)));

	float *extra;
	ErrChk(cudaMalloc((void **) &extra, size));

	//  3. Computing
	ErrChk(cudnnConvolutionForward(handle, &one, xDesc, input, filterDesc, filter, convDesc, algo, extra, size, &zero, yDesc, result_cudnn));
	
	ErrChk(cudaMemcpy(h_result_cudnn, result_cudnn, sizeof(float)*N*K*P*Q, cudaMemcpyDeviceToHost));



	//Our implementation
	//matrix parameters, because the matrix is stored in Row-Major style and MM is Column-Major, A*B -> BT * AT
		
	float *result_our;
	ErrChk(cudaMalloc((void**)&result_our, N*K*P*Q*sizeof(float)));
	

	//gemm	1101
	dim3 block_size;
	block_size.x = 64;
	block_size.y = 1;
	block_size.z = 1;
	
	dim3 grid_size;
	grid_size.x = K/16;
	grid_size.y = (Q*P-1)/16 + 1;
	grid_size.z = N;

	if (H*W%2)
		implicit_gemm_1101_1<<<grid_size, block_size>>>(input, filter, result_our, N, C, H, W, K);
	else if (H*W%16)
		implicit_gemm_1101_2<<<grid_size, block_size>>>(input, filter, result_our, N, C, H, W, K);
	else
		implicit_gemm_1101_16<<<grid_size, block_size>>>(input, filter, result_our, N, C, H, W, K);
	
 	KernelErrChk();

	ErrChk(cudaMemcpy(h_result_our, result_our, sizeof(float)*N*K*P*Q, cudaMemcpyDeviceToHost));
	
	//Result Test
	for (int j=0; j<N*K; ++j){
		for (int i=0; i<P*Q; ++i)
			printf("%.f ", h_result_cudnn[j*P*Q+i]);
		printf("\n");
	}
	printf("\n");

	printf("----------------------------------\n");
	for (int j=0; j<N*K; ++j){
		for (int i=0; i<P*Q; ++i)
			printf("%.f ", h_result_our[j*P*Q+i]);
		printf("\n");
	}
	printf("\n");

	for (int j=0; j<N*K*P*Q; ++j){
		if (abs(h_result_cudnn[j] - h_result_our[j]) > 10e-2){
			printf("Rejected @ %d\n", j);
			exit(EXIT_FAILURE);
		}
	}
	printf("Passed\n");

	
	ErrChk(cudnnDestroy(handle));
	ErrChk(cudnnDestroyTensorDescriptor(xDesc));
	ErrChk(cudnnDestroyTensorDescriptor(yDesc));
	ErrChk(cudnnDestroyFilterDescriptor(filterDesc));
	ErrChk(cudnnDestroyConvolutionDescriptor(convDesc));

	ErrChk(cudaFree(input));
	ErrChk(cudaFree(filter));
	ErrChk(cudaFree(result_our));
	ErrChk(cudaFree(result_cudnn));
	ErrChk(cudaFree(extra));
	
	free(h_input);
	free(h_filter);
	free(h_result_our);
	free(h_result_cudnn);


	return 0;
}
