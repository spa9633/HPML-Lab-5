#include <iostream>
#include <math.h>
#include <iomanip>
// Kernel function to add the elements of two arrays
__global__
void add(long n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}


int main(int argc, char** argv)
{
  long K = atoi(argv[1]) * 1000000;
  size_t size = K * sizeof(float);

  //Initialize host side arrays and allocate memory

  float *h_x = (float*)malloc(K * sizeof(float));
  float *h_y = (float*)malloc(K * sizeof(float));

  for (int i = 0; i < K; i++) {
    h_x[i] = 1.0f;
    h_y[i] = 2.0f;
  }
  
  //Initialize device side arrays and allocate memory in CUDA
  float *d_x, *d_y;
  cudaMalloc((void**)&d_x, size);
  cudaMalloc((void**)&d_y, size);
  
  //Copy arrays to device
  cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

  // Run kernel on KM elements on the GPU
  add<<<1, 256>>>(K, d_x, d_y);

 //Copy results to host side array
  cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

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