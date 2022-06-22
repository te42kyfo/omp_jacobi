#include <iostream>
#include <sys/time.h>


#define GPU_ERROR(ans)                          \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    std::cerr << "GPUassert: \"" << cudaGetErrorString(code) << "\"  in "
              << file << ": " << line << "\n";
    if (abort) exit(code);
  }
}
using namespace std;

double dtime() {
  double tseconds = 0;
  struct timeval t;
  gettimeofday(&t, NULL);
  tseconds = (double)t.tv_sec + (double)t.tv_usec * 1.0e-6;
  return tseconds;
}


__global__ void jacobi_kernel(double const * __restrict__ gridA,
                              double  * __restrict__ gridB, int width,
                              int height) {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x + 1;
  int tidy = threadIdx.y + blockIdx.y * blockDim.y + 1;

  if (tidx >= width - 1 || tidy >= height - 1)
    return;

  gridB[tidy * width + tidx] =
      0.25 *
      (gridA[tidy * width + tidx + 1] + gridA[tidy * width + tidx - 1] +
       gridA[(tidy + 1) * width + tidx] + gridA[(tidy - 1) * width + tidx]);
}

int main(int argc, char **argv) {
  const int width = 20000;
  const int height = 20000;

  const int iters = 10;
  double* gridA = ((double *)malloc(width * height * sizeof(double) + 64)) + 3;
  double* gridB = ((double *)malloc(width * height * sizeof(double) + 64)) + 3;


#pragma omp parallel for
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      if (y != 0) {
        gridA[y * width + x] = 0.0;
        gridB[y * width + x] = 0.0;
      } else {
        gridA[y * width + x] = 1.0;
        gridB[y * width + x] = 1.0;
      }
    }
  }

  double* DgridA;
  double* DgridB;
  cudaMalloc(&DgridA, width * height * sizeof(double));
  cudaMalloc(&DgridB, width * height * sizeof(double));
  cudaMemcpy(DgridA, gridA, width*height*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(DgridB, gridB, width*height*sizeof(double), cudaMemcpyHostToDevice);


  cudaDeviceSynchronize();
  double t1 = dtime();

  dim3 threadBlockSize = dim3(128, 4);
  dim3 gridSize = dim3(width / threadBlockSize.x + 1, height / threadBlockSize.y + 1);
  for (int it = 0; it < iters; it++) {
    jacobi_kernel<<<gridSize, threadBlockSize>>>(DgridA, DgridB, width, height);
    std::swap(DgridA, DgridB);
  }

  GPU_ERROR(cudaDeviceSynchronize());

  double t2 = dtime();
  double dt = t2 - t1;
  cout << dt * 1000 << " ms   "
       << 4 * (int64_t)iters * width * height / dt / 1e9 << " GLup/s   "
       << (int64_t)iters * width * height * sizeof(double) * 2.0 / dt / 1e9
       << " GB/s\n";

  free(gridA - 3);
  free(gridB - 3);

  return 0;
}
