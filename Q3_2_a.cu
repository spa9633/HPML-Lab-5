#include <iostream>
#include <math.h>
#include <iomanip>
// Kernel function to add the elements of two arrays
__global__
void add(long n, float* x, float* y)
{   int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; index+=stride)
        y[i] = x[i] + y[i];
}

int main(int argc, char** argv)
{
  long K = atoi(argv[1]) * 1000000;
  size_t size = K * sizeof(float);

  //Initialize host side arrays and allocate memory
  float *h_x, *h_y;

  h_x = (float*)malloc(K * sizeof(float));
  h_y = (float*)malloc(K * sizeof(float));

  for (int i = 0; i < K; i++) {
    h_x[i] = 1.0f;
    h_y[i] = 2.0f;
  }
  
  //Initialize device side arrays and allocate memory in CUDA
  float *d_x, *d_y;
  cudaMalloc((void**)&d_x, size);
  cudaMalloc((void**)&d_y, size);

  cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

  // Run kernel on 1M elements on the GPU
  add<<<1, 1>>>(K, d_x, d_y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  
  //Copy results to host side array
  cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < K; i++)
    maxError = fmax(maxError, fabs(h_y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(d_x);
  cudaFree(d_y);
  delete [] h_x;
  delete [] h_y;
  
  return 0;
}