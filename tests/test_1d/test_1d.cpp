#include "mgard_api.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#ifdef MGARD_TIMING
#include <chrono>
#endif

using namespace std;

int main(int argc, char **argv) {
  double tol, result;
  int out_size;
  unsigned char *compressed_data = 0;

  if (argc != 3) {
    cerr << "Wrong arugments!\n";
    return 1;
  }

  ifstream datafile(argv[1], ios::in | ios::binary);
  if (!datafile)
    cerr << "Can't open input file!";

  tol = stod(argv[2]);

  datafile.seekg(0, ios::end);
  size_t num_elements = datafile.tellg() / sizeof(double);
  datafile.seekg(0, ios::beg);

  std::vector<double> data(num_elements);
  datafile.read(reinterpret_cast<char *>(&data[0]),
                num_elements * sizeof(double));
#ifdef MGARD_TIMING
  auto start = chrono::high_resolution_clock::now();
#endif
  compressed_data =
      mgard_compress(data.data(), out_size, 1, num_elements, 1, tol);
#ifdef MGARD_TIMING
  auto stop = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
  cout << "Original size = " << num_elements * 8 << " out_size = " << out_size
       << " CR = " << num_elements * 8.0 / out_size
       << " Time = " << (double) duration.count() / 1000000<< endl;
#endif 

  double *decompressed_data =
      mgard_decompress<double>(compressed_data, out_size, 1, num_elements, 1);

  double abserr = 0.0;
  double max_abserr = 0.0;
  double max = abs(data[0]);
  for (int i = 0; i < num_elements; i++) {
    abserr = abs(decompressed_data[i] - data[i]);
    if (max_abserr < abserr) {
      max_abserr = abserr;
    }

    if (max < abs(data[i])) {
      max = abs(data[i]);
    }
  }

  cout << "The prescribed error = " << tol
       << " The achieved error = " << max_abserr / max << endl;

  free(compressed_data);
  compressed_data = 0;

  free(decompressed_data);
  decompressed_data = 0;

  if (tol >= max_abserr / max) {
    return 0;
  } else {
    return 1;
  }
}
