#include <iostream>

using namespace std;

typedef double DTYPE;

void jacobi_iteration(DTYPE *gridA, DTYPE *gridB, int width, int height) {
#pragma omp parallel for
  for (int y = 1; y < height - 1; y++) {
    for (int x = 1; x < width - 1; x++) {
      gridA[y * width + x] =
          0.25 * (gridB[(y + 1) * width + x] + gridB[(y - 1) * width + x] +
                  gridB[y * width + x - 1] + gridB[y * width + x + 1]);
    }
  }
}

DTYPE compute_residual(DTYPE *gridA, DTYPE *gridB, int width, int height) {
  DTYPE residual = 0.0;
#pragma omp parallel for reduction(+ : residual)
  for (int y = 1; y < height - 1; y++) {
    for (int x = 1; x < width - 1; x++) {
      residual += -4.0 * gridB[y * width + x] +
                  (gridB[(y + 1) * width + x] + gridB[(y - 1) * width + x] +
                   gridB[y * width + x - 1] + gridB[y * width + x + 1]);
    }
  }
  return residual;
}

int main(int argc, char **argv) {
  const int width = 16000;
  const int height = 16000;
  const int iters = 100;
  DTYPE *gridA = new DTYPE[width * height];
  DTYPE *gridB = new DTYPE[width * height];

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

  for (int it = 0; it < iters; it++) {

    if (it % 10 == 0)
      cout << it << " " << compute_residual(gridA, gridB, width, height)
           << "\n";

    jacobi_iteration(gridA, gridB, width, height);

    swap(gridA, gridB);
  }

  delete gridA;
  delete gridB;

  return 0;
}
