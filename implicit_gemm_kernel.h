__global__ void implicit_gemm_1101_16(float *input, float *filter, float *output, int N, int C, int H, int W, int K){

	float reg_C[4];
	float reg_A[8];
	float reg_B[2];

	__shared__ float sh_A[2*16*8];
	__shared__ float sh_B[2*16*8];

	reg_C[0] = 0.f;
	reg_C[1] = 0.f;
	reg_C[2] = 0.f;
	reg_C[3] = 0.f;

	//load A from input feature map to shared memory
	float2 *A = (float2 *)(filter + blockIdx.x*16*C + (threadIdx.x/16)*2 + (threadIdx.x%16)*C);
	*((float2 *)(sh_A + 2*threadIdx.x)) = *A;

	float2 *B = (float2 *)(input + blockIdx.z*C*H*W + blockIdx.y*16 + (threadIdx.x%8)*2 + (threadIdx.x/8)*H*W);
	*((float2 *)(sh_B + 2*threadIdx.x)) = *B;

	int double_buffer = 0;

#pragma unroll
	for (int c=0; c<C; c+=8){
	
		__syncthreads();

		int A_offset = double_buffer + (threadIdx.x%4)*8;
		int B_offset = double_buffer + (threadIdx.x/4);

#pragma unroll
		
		for (int i=0; i<8; i+=2){
			
			reg_A[0] = sh_A[A_offset];
			reg_A[1] = sh_A[A_offset+2];
			reg_A[2] = sh_A[A_offset+4];
			reg_A[3] = sh_A[A_offset+6];
			reg_A[4] = sh_A[A_offset+1];
			reg_A[5] = sh_A[A_offset+3];
			reg_A[6] = sh_A[A_offset+5];
			reg_A[7] = sh_A[A_offset+7];
		
			reg_B[0] = sh_B[B_offset];
			reg_B[1] = sh_B[B_offset+16];
			
			reg_C[0] = fma(reg_A[0], reg_B[0], reg_C[0]);
			reg_C[1] = fma(reg_A[1], reg_B[0], reg_C[1]);
			reg_C[2] = fma(reg_A[2], reg_B[0], reg_C[2]);
			reg_C[3] = fma(reg_A[3], reg_B[0], reg_C[3]);
			reg_C[0] = fma(reg_A[4], reg_B[1], reg_C[0]);
			reg_C[1] = fma(reg_A[5], reg_B[1], reg_C[1]);
			reg_C[2] = fma(reg_A[6], reg_B[1], reg_C[2]);
			reg_C[3] = fma(reg_A[7], reg_B[1], reg_C[3]);

			A_offset += 32;
			B_offset += 32;
		}
		
		double_buffer ^= 128;
		
		if (c+8<C){
			A += 4;
			B += 4*H*W;
			*((float2 *)(sh_A + double_buffer + 2*threadIdx.x)) = *A;
			*((float2 *)(sh_B + double_buffer + 2*threadIdx.x)) = *B;
		}
	}
	
	int C_offset = blockIdx.z*K*H*W + blockIdx.x*16*H*W + blockIdx.y*16 + (threadIdx.x/4) + (threadIdx.x%4)*4*H*W;
	output[C_offset] = reg_C[0];
	output[C_offset+H*W] = reg_C[1];
	output[C_offset+2*H*W] = reg_C[2];
	output[C_offset+3*H*W] = reg_C[3];
}

__global__ void implicit_gemm_1101_2(float *input, float *filter, float *output, int N, int C, int H, int W, int K){

	if (blockIdx.y < gridDim.y-1){
		float reg_C[4];
		float reg_A[8];
		float reg_B[2];

		__shared__ float sh_A[2*16*8];
		__shared__ float sh_B[2*16*8];

		reg_C[0] = 0.f;
		reg_C[1] = 0.f;
		reg_C[2] = 0.f;
		reg_C[3] = 0.f;

		//load filter to shared memory
		float2 *A = (float2 *)(filter + blockIdx.x*16*C + (threadIdx.x/16)*2 + (threadIdx.x%16)*C);
		*((float2 *)(sh_A + 2*threadIdx.x)) = *A;

		//load input feature map to shared memory
		float2 *B = (float2 *)(input + blockIdx.z*C*H*W + blockIdx.y*16 + (threadIdx.x%8)*2 + (threadIdx.x/8)*H*W);
		*((float2 *)(sh_B + 2*threadIdx.x)) = *B;

		int double_buffer = 0;

#pragma unroll
		for (int c=0; c<C; c+=8){
		
			__syncthreads();

			int A_offset = double_buffer + (threadIdx.x%4)*8;
			int B_offset = double_buffer + (threadIdx.x/4);

#pragma unroll
			
			for (int i=0; i<8; i+=2){
				
				reg_A[0] = sh_A[A_offset];
				reg_A[1] = sh_A[A_offset+2];
				reg_A[2] = sh_A[A_offset+4];
				reg_A[3] = sh_A[A_offset+6];
				reg_A[4] = sh_A[A_offset+1];
				reg_A[5] = sh_A[A_offset+3];
				reg_A[6] = sh_A[A_offset+5];
				reg_A[7] = sh_A[A_offset+7];
			
				reg_B[0] = sh_B[B_offset];
				reg_B[1] = sh_B[B_offset+16];
				
				reg_C[0] = fma(reg_A[0], reg_B[0], reg_C[0]);
				reg_C[1] = fma(reg_A[1], reg_B[0], reg_C[1]);
				reg_C[2] = fma(reg_A[2], reg_B[0], reg_C[2]);
				reg_C[3] = fma(reg_A[3], reg_B[0], reg_C[3]);
				reg_C[0] = fma(reg_A[4], reg_B[1], reg_C[0]);
				reg_C[1] = fma(reg_A[5], reg_B[1], reg_C[1]);
				reg_C[2] = fma(reg_A[6], reg_B[1], reg_C[2]);
				reg_C[3] = fma(reg_A[7], reg_B[1], reg_C[3]);

				A_offset += 32;
				B_offset += 32;
			}
			
			double_buffer ^= 128;
			
			if (c+8<C){
				A += 4;
				B += 4*H*W;
				*((float2 *)(sh_A + double_buffer + 2*threadIdx.x)) = *A;
				*((float2 *)(sh_B + double_buffer + 2*threadIdx.x)) = *B;
			}
		}
		
		int C_offset = blockIdx.z*K*H*W + blockIdx.x*16*H*W + blockIdx.y*16 + (threadIdx.x/4) + (threadIdx.x%4)*4*H*W;
		output[C_offset] = reg_C[0];
		output[C_offset+H*W] = reg_C[1];
		output[C_offset+2*H*W] = reg_C[2];
		output[C_offset+3*H*W] = reg_C[3];
	}

	else{
		float reg_C[4];
		float reg_A[8];
		float reg_B[2];

		__shared__ float sh_A[2*16*8];
		__shared__ float sh_B[2*16*8];

		reg_C[0] = 0.f;
		reg_C[1] = 0.f;
		reg_C[2] = 0.f;
		reg_C[3] = 0.f;

		//load filter to shared memory
		float2 *A = (float2 *)(filter + blockIdx.x*16*C + (threadIdx.x/16)*2 + (threadIdx.x%16)*C);
		*((float2 *)(sh_A + 2*threadIdx.x)) = *A;

		//load input feature map to shared memory

		float2 *B = (float2 *)(input + blockIdx.z*C*H*W + blockIdx.y*16 + (threadIdx.x%8)*2 + (threadIdx.x/8)*H*W);
		if (threadIdx.x%8 < (H*W%16)/2 ){
			*((float2 *)(sh_B + 2*threadIdx.x)) = *B;
		}

		int double_buffer = 0;

#pragma unroll
		for (int c=0; c<C; c+=8){
		
			__syncthreads();

			int A_offset = double_buffer + (threadIdx.x%4)*8;
			int B_offset = double_buffer + (threadIdx.x/4);

			if (threadIdx.x < (H*W%16)*4)
#pragma unroll
				for (int i=0; i<8; i+=2){
					
					reg_A[0] = sh_A[A_offset];
					reg_A[1] = sh_A[A_offset+2];
					reg_A[2] = sh_A[A_offset+4];
					reg_A[3] = sh_A[A_offset+6];
					reg_A[4] = sh_A[A_offset+1];
					reg_A[5] = sh_A[A_offset+3];
					reg_A[6] = sh_A[A_offset+5];
					reg_A[7] = sh_A[A_offset+7];
				
					reg_B[0] = sh_B[B_offset];
					reg_B[1] = sh_B[B_offset+16];
					
					reg_C[0] = fma(reg_A[0], reg_B[0], reg_C[0]);
					reg_C[1] = fma(reg_A[1], reg_B[0], reg_C[1]);
					reg_C[2] = fma(reg_A[2], reg_B[0], reg_C[2]);
					reg_C[3] = fma(reg_A[3], reg_B[0], reg_C[3]);
					reg_C[0] = fma(reg_A[4], reg_B[1], reg_C[0]);
					reg_C[1] = fma(reg_A[5], reg_B[1], reg_C[1]);
					reg_C[2] = fma(reg_A[6], reg_B[1], reg_C[2]);
					reg_C[3] = fma(reg_A[7], reg_B[1], reg_C[3]);

					A_offset += 32;
					B_offset += 32;
				}
				
			double_buffer ^= 128;
			
			if (c+8<C){
				A += 4;
				B += 4*H*W;
				*((float2 *)(sh_A + double_buffer + 2*threadIdx.x)) = *A;
				if (threadIdx.x%8 < (H*W%16)/2 )
					*((float2 *)(sh_B + double_buffer + 2*threadIdx.x)) = *B;
			}
		}
		
		if (threadIdx.x < (H*W%16)*4){
		int C_offset = blockIdx.z*K*H*W + blockIdx.x*16*H*W + blockIdx.y*16 + (threadIdx.x/4) + (threadIdx.x%4)*4*H*W;
		output[C_offset] = reg_C[0];
		output[C_offset+H*W] = reg_C[1];
		output[C_offset+2*H*W] = reg_C[2];
		output[C_offset+3*H*W] = reg_C[3];
		}
	 }

}


__global__ void implicit_gemm_1101_1(float *input, float *filter, float *output, int N, int C, int H, int W, int K){

	if (blockIdx.y < gridDim.y-1){
		float reg_C[4];
		float reg_A[8];
		float reg_B[2];

		__shared__ float sh_A[2*16*8];
		__shared__ float sh_B[2*16*8];

		reg_C[0] = 0.f;
		reg_C[1] = 0.f;
		reg_C[2] = 0.f;
		reg_C[3] = 0.f;

		//load filter to shared memory
		float2 *A = (float2 *)(filter + blockIdx.x*16*C + (threadIdx.x/16)*2 + (threadIdx.x%16)*C);
		*((float2 *)(sh_A + 2*threadIdx.x)) = *A;

		//load input feature map to shared memory
		float *B = (input + blockIdx.z*C*H*W + blockIdx.y*16 + (threadIdx.x%8)*2 + (threadIdx.x/8)*H*W);
		*(sh_B + 2*threadIdx.x) = *B;
		*(sh_B + 2*threadIdx.x + 1) = *(B+1);

		int double_buffer = 0;

#pragma unroll
		for (int c=0; c<C; c+=8){
		
			__syncthreads();

			int A_offset = double_buffer + (threadIdx.x%4)*8;
			int B_offset = double_buffer + (threadIdx.x/4);

#pragma unroll
			
			for (int i=0; i<8; i+=2){
				
				reg_A[0] = sh_A[A_offset];
				reg_A[1] = sh_A[A_offset+2];
				reg_A[2] = sh_A[A_offset+4];
				reg_A[3] = sh_A[A_offset+6];
				reg_A[4] = sh_A[A_offset+1];
				reg_A[5] = sh_A[A_offset+3];
				reg_A[6] = sh_A[A_offset+5];
				reg_A[7] = sh_A[A_offset+7];
			
				reg_B[0] = sh_B[B_offset];
				reg_B[1] = sh_B[B_offset+16];
				
				reg_C[0] = fma(reg_A[0], reg_B[0], reg_C[0]);
				reg_C[1] = fma(reg_A[1], reg_B[0], reg_C[1]);
				reg_C[2] = fma(reg_A[2], reg_B[0], reg_C[2]);
				reg_C[3] = fma(reg_A[3], reg_B[0], reg_C[3]);
				reg_C[0] = fma(reg_A[4], reg_B[1], reg_C[0]);
				reg_C[1] = fma(reg_A[5], reg_B[1], reg_C[1]);
				reg_C[2] = fma(reg_A[6], reg_B[1], reg_C[2]);
				reg_C[3] = fma(reg_A[7], reg_B[1], reg_C[3]);

				A_offset += 32;
				B_offset += 32;
			}
			
			double_buffer ^= 128;
			
			if (c+8<C){
				A += 4;
				B += 4*H*W;
				*((float2 *)(sh_A + double_buffer + 2*threadIdx.x)) = *A;
				*(sh_B + double_buffer + 2*threadIdx.x) = *B;
				*(sh_B + double_buffer + 2*threadIdx.x + 1) = *(B+1);
			}
		}
		
		int C_offset = blockIdx.z*K*H*W + blockIdx.x*16*H*W + blockIdx.y*16 + (threadIdx.x/4) + (threadIdx.x%4)*4*H*W;
		output[C_offset] = reg_C[0];
		output[C_offset+H*W] = reg_C[1];
		output[C_offset+2*H*W] = reg_C[2];
		output[C_offset+3*H*W] = reg_C[3];
	}

	else{
		float reg_C[4];
		float reg_A[8];
		float reg_B[2];

		__shared__ float sh_A[2*16*8];
		__shared__ float sh_B[2*16*8];

		reg_C[0] = 0.f;
		reg_C[1] = 0.f;
		reg_C[2] = 0.f;
		reg_C[3] = 0.f;

		//load filter to shared memory
		float2 *A = (float2 *)(filter + blockIdx.x*16*C + (threadIdx.x/16)*2 + (threadIdx.x%16)*C);
		*((float2 *)(sh_A + 2*threadIdx.x)) = *A;

		//load input feature map to shared memory

		int ruler = (H*W)%16;
		float *B = input + blockIdx.z*C*H*W + blockIdx.y*16 + (threadIdx.x%4)*2*H*W + (threadIdx.x/4);
		if (threadIdx.x < ruler*4 ){
			*(sh_B + 2*threadIdx.x) = *B;
			*(sh_B + 2*threadIdx.x + 1) = *(B+H*W);
		}

		int double_buffer = 0;

#pragma unroll
		for (int c=0; c<C; c+=8){
		
			__syncthreads();

			int A_offset = double_buffer + (threadIdx.x%4)*8;
			int B_offset = double_buffer + (threadIdx.x/4)*8;

			if (threadIdx.x < ruler*4)
#pragma unroll
				for (int i=0; i<8; i+=2){
					
					reg_A[0] = sh_A[A_offset];
					reg_A[1] = sh_A[A_offset+2];
					reg_A[2] = sh_A[A_offset+4];
					reg_A[3] = sh_A[A_offset+6];
					reg_A[4] = sh_A[A_offset+1];
					reg_A[5] = sh_A[A_offset+3];
					reg_A[6] = sh_A[A_offset+5];
					reg_A[7] = sh_A[A_offset+7];
				
					reg_B[0] = sh_B[B_offset];
					reg_B[1] = sh_B[B_offset+1];
					
					reg_C[0] = fma(reg_A[0], reg_B[0], reg_C[0]);
					reg_C[1] = fma(reg_A[1], reg_B[0], reg_C[1]);
					reg_C[2] = fma(reg_A[2], reg_B[0], reg_C[2]);
					reg_C[3] = fma(reg_A[3], reg_B[0], reg_C[3]);
					reg_C[0] = fma(reg_A[4], reg_B[1], reg_C[0]);
					reg_C[1] = fma(reg_A[5], reg_B[1], reg_C[1]);
					reg_C[2] = fma(reg_A[6], reg_B[1], reg_C[2]);
					reg_C[3] = fma(reg_A[7], reg_B[1], reg_C[3]);

					A_offset += 32;
					B_offset += 2;
				}
				
			double_buffer ^= 128;
			
			if (c+8<C){
				A += 4;
				B += 8*H*W;
				*((float2 *)(sh_A + double_buffer + 2*threadIdx.x)) = *A;
				if (threadIdx.x < ruler*4 ){
					*(sh_B + 2*threadIdx.x + double_buffer) = *B;
					*(sh_B + 2*threadIdx.x + double_buffer + 1) = *(B+H*W);
				}
			}
		}
		
		if (threadIdx.x < ruler*4){
		int C_offset = blockIdx.z*K*H*W + blockIdx.x*16*H*W + blockIdx.y*16 + (threadIdx.x/4) + (threadIdx.x%4)*4*H*W;
		output[C_offset] = reg_C[0];
		output[C_offset+H*W] = reg_C[1];
		output[C_offset+2*H*W] = reg_C[2];
		output[C_offset+3*H*W] = reg_C[3];
		}
	 }

}
