#include <iostream>
#include <math.h>

// function to add the elements of two arrays
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
      y[i] = x[i] + y[i];
}

int main(int argc, char** argv)
{
  int K = atoi(argv[1]) * 1000000; // K*M elements

  float *x = new float[K];
  float *y = new float[K];

  // initialize x and y arrays on the host
  for (int i = 0; i < K; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the CPU
  add(K, x, y);

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < K; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  free(x);
  free(y);

  return 0;
}