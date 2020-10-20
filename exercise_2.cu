#include <stdio.h>
#include <sys/time.h>

double mysecond(){
  struct timeval tp;
  struct timezone tzp;
  int i = gettimeofday(&tp, &tzp);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void SAXPY_CPU(int N, float A, float *X, float *Y, float *R){
  for(int i=0; i<N; i++){
    R[i] = A * X[i] + Y[i];
  }
}

__global__ void SAXPY_GPU(float A, float *X, float *Y){
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  Y[i] = A * X[i] + Y[i];
}

int main(int argc, char *argv[]){

  int ARRAY_SIZE = 10000;
  if(argc>0) ARRAY_SIZE = atoi(argv[1]);
  printf("ARRAY_SIZE: %d\n", ARRAY_SIZE);

  float A = 10;
  float *X_CPU = (float *) malloc(ARRAY_SIZE*sizeof(float));
  float *Y_CPU = (float *) malloc(ARRAY_SIZE*sizeof(float));
  float *R_CPU = (float *) malloc(ARRAY_SIZE*sizeof(float));
  for(int i=0; i<ARRAY_SIZE; i++){
    X_CPU[i] = float(i)/ARRAY_SIZE * 2;
    Y_CPU[i] = float(i)/ARRAY_SIZE * 4;
  }

  // CPU Part:
  double T_CPU = mysecond();
  SAXPY_CPU(ARRAY_SIZE, A, X_CPU, Y_CPU, R_CPU);
  T_CPU = mysecond() - T_CPU;
  printf("Computing SAXPY on the CPU... Done! Time: %f\n", T_CPU);

  // GPU Part:
  float *X_GPU = 0;
  float *Y_GPU = 0;
  float *R_GPU = (float *) malloc(ARRAY_SIZE*sizeof(float));
  cudaMalloc(&X_GPU, ARRAY_SIZE*sizeof(float));
  cudaMalloc(&Y_GPU, ARRAY_SIZE*sizeof(float));
  cudaMemcpy(X_GPU, X_CPU, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(Y_GPU, Y_CPU, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);
  int TPB = 256;
  int BPG = (ARRAY_SIZE+TPB-1)/TPB;
  double T_GPU = mysecond();
  SAXPY_GPU<<<BPG, TPB>>>(A, X_GPU, Y_GPU);
  cudaMemcpy(R_GPU, Y_GPU, ARRAY_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  T_GPU = mysecond() - T_GPU;
  printf("Computing SAXPY on the GPU... Done! Time: %f\n", T_GPU);


  // Comparison Part:
  float maxError = -10.0;
  for(int i=0; i<ARRAY_SIZE; i++){
    maxError = fmax(maxError, fabs(R_CPU[i]-R_GPU[i]));
  }
  if(maxError<0.00001){
   printf("Comparing the output for each implementation... Correct! Max Error: %e\n", maxError);
  }
  else{
   printf("Not Correct! Max Error: %e\n", maxError);
  }

  //printf("%d, %f, %f, %e\n", ARRAY_SIZE, T_CPU, T_GPU, maxError);

  free(X_CPU);
  free(Y_CPU);
  cudaFree(X_GPU);
  cudaFree(Y_GPU);
  return 0;

}


