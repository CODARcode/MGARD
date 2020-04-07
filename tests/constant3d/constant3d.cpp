#include "mgard_api.h"
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

int main(int argc, char **argv) {
  double step, result, tol = 0.001;
  int out_size;
  const int npoints = 4096;
  double v[npoints];
  unsigned char *compressed_data = 0;

  for (int i = 0; i < npoints; i++) {
    v[i] = 10.0;
  }

  compressed_data = mgard_compress(v, out_size, 16, 16, npoints / 256, tol);
  cout << "Original size = " << npoints * 8 << " out_size = " << out_size
       << " CR = " << npoints * 8.0 / out_size << endl;

  double *decompressed_data = mgard_decompress<double>(
      compressed_data, out_size, 16, 16, npoints / 256);

  double abserr = 0.0;
  double max_abserr = 0.0;
  double max = v[0];
  for (int i = 0; i < npoints; i++) {
    abserr = abs(decompressed_data[i] - v[i]);
    if (max_abserr < abserr) {
      max_abserr = abserr;
    }

    if (max < abs(v[i])) {
      max = abs(v[i]);
    }
  }

  cout << "The prescribed error = " << tol
       << " The achieved error = " << max_abserr / max << endl;

  if (tol >= max_abserr / max) {
    return 0;
  } else {
    return 1;
  }
}
