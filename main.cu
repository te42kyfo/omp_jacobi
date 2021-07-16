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
typedef double DTYPE;


int main(int argc, char **argv) {
  const int width = 10000;
  const int height = 10000;
  const int iters = 10;
  DTYPE *gridA;
  gridA = (DTYPE*) malloc( width * height*sizeof(DTYPE) );
  //GPU_ERROR(cudaMallocManaged(&gridA, width * height * sizeof(DTYPE)));
  DTYPE *gridB;
  gridB = (DTYPE*) malloc( width * height*sizeof(DTYPE) );
  //GPU_ERROR(cudaMallocManaged(&gridB, width * height * sizeof(DTYPE)));

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


  #pragma acc enter data  copyin(gridA[:width*height]) copyin(gridB[:width*height])

  double t1 = dtime();
  for (int it = 0; it < iters; it++) {

      DTYPE residual = 0.0;
#pragma acc data present(gridA[:width*height]), present(gridB[:width*height]), copy(residual)
    {

#pragma acc parallel loop independent
        for (int y = 1; y < height - 1; y++) {
            #pragma acc loop independent
            for (int x = 1; x < width - 1; x++) {
                gridA[y*width + x] = 0.25 * (gridB[y*width + x + 1] + gridB[y*width + x - 1] + gridB[(y+1)*width + x] + gridB[(y-1)*width + x]);
            }
        }
#pragma acc parallel loop independent
        for (int y = 1; y < height - 1; y++) {
            #pragma acc loop independent
            for (int x = 1; x < width - 1; x++) {
                gridB[y*width + x] = 0.25 * (gridA[y*width + x + 1] + gridA[y*width + x - 1] + gridA[(y+1)*width + x] + gridA[(y-1)*width + x]);
            }
        }
#pragma acc parallel loop independent reduction(+:residual)
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                residual += -4.0 * gridB[y * width + x] +
                    (gridB[(y + 1) * width + x] + gridB[(y - 1) * width + x] +
                     gridB[y * width + x - 1] + gridB[y * width + x + 1]);
            }
        }
    }

    cout << it << " " << residual << "\n";

    //swap(gridA, gridB);
  }

  double t2 = dtime();
  double dt = t2 - t1;
  cout << dt << " ms   " << 2*(int64_t)iters * width * height * 1.0 / dt / 1e9
       << " GLup/s   "
       << (int64_t)iters * width * height * sizeof(DTYPE) * 6 * 1.0 / dt / 1e9
       << " GB/s\n";

  //GPU_ERROR(cudaFree(gridA));
  //GPU_ERROR(cudaFree(gridB));

  return 0;
}
