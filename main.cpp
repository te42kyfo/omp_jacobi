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
  const int width = 16002;
  const int height = 16002;

  const int iters = 10;
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

#pragma omp target enter data map(to                                           \
                                  : gridA[:width * height], gridB              \
                                  [:width * height])

  double t1 = dtime();
  for (int it = 0; it < iters; it += 2) {

#pragma omp target teams distribute parallel for collapse(2)
    for (int oy = 1; oy < height - 1; oy += 4) {
      for (int x = 1; x < width - 1; x++) {
        for (int iy = 0; iy < 4; iy++) {
          int y = oy + iy;
          gridA[y * width + x] =
              0.25 * (gridB[y * width + x + 1] + gridB[y * width + x - 1] +
                      gridB[(y + 1) * width + x] + gridB[(y - 1) * width + x]);
        }
      }
    }
    #pragma omp target teams distribute parallel for collapse(2)
    for (int oy = 1; oy < height - 1; oy += 4) {
      for (int x = 1; x < width - 1; x++) {
        for (int iy = 0; iy < 4; iy++) {
          int y = oy + iy;
          gridB[y * width + x] =
              0.25 * (gridA[y * width + x + 1] + gridA[y * width + x - 1] +
                      gridA[(y + 1) * width + x] + gridA[(y - 1) * width + x]);
        }
      }
    }
  }
  double t2 = dtime();
  double dt = t2 - t1;
  cout << dt << " ms   "
       << (int64_t)iters * width * height * sizeof(double) * 2  / dt / 1e9
       << " GB/s\n";


  free(gridA);
  free(gridB);
  return 0;
}
