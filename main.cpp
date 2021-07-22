#include <iostream>
#include <sys/time.h>

using namespace std;

double dtime() {
  double tseconds = 0;
  struct timeval t;
  gettimeofday(&t, NULL);
  tseconds = (double)t.tv_sec + (double)t.tv_usec * 1.0e-6;
  return tseconds;
}

int main(int argc, char **argv) {
  const int width = 20000;

  const int height = 20000;
  const int iters = 4;
  double *gridA;
  double *gridB;
  gridA = (double *)malloc(width * height * sizeof(double));
  gridB = (double *)malloc(width * height * sizeof(double));


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

  double t1 = dtime();
  for (int it = 0; it < iters; it++) {
#pragma omp target teams distribute parallel for collapse(2)
    for (int y = 1; y < height - 1; y++) {
      for (int x = 1; x < width - 1; x++) {
        gridA[y * width + x] =
            0.25 * (gridB[y * width + x + 1] + gridB[y * width + x - 1] +
                    gridB[(y + 1) * width + x] + gridB[(y - 1) * width + x]);
      }
    }

    double residual = 0.0;

#pragma omp target teams distribute parallel for reduction(+:residual) collapse(2)
    for (int y = 1; y < height - 1; y++) {
      for (int x = 1; x < width - 1; x++) {
        residual += -4.0 * gridB[y * width + x] +
                    (gridB[(y + 1) * width + x] + gridB[(y - 1) * width + x] +
                     gridB[y * width + x - 1] + gridB[y * width + x + 1]);
      }
    }

    cout << it << " " << residual << "\n";
    swap(gridA, gridB);
  }

  double t2 = dtime();
  double dt = t2 - t1;
  cout << dt << " ms   " << 4 * (int64_t)iters * width * height * 1.0 / dt / 1e9
       << " GLup/s   "
       << (int64_t)iters * width * height * sizeof(double) * 3 * 1.0 / dt / 1e9
       << " GB/s\n";

  // GPU_ERROR(cudaFree(gridA));
  // GPU_ERROR(cudaFree(gridB));

  return 0;
}
