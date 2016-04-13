#include <iostream>
#include <armadillo>
#include <omp.h>
#include <cblas.h>

using namespace std;
using namespace arma;

int main(int argc, char** argv) 
{
  int tid;
  time_t start, stop;
  struct timeval tv;
  int n = 1000;
  mat I = eye(n,n);
  vec d = zeros<vec>(8);

  gettimeofday(&tv, NULL);
  start = (tv.tv_sec)*1000 + (tv.tv_usec)/1000;

  omp_set_num_threads(8);
  openblas_set_num_threads(1);
#pragma omp parallel private(tid)
{
  tid = omp_get_thread_num();
  d[tid] = det(I+tid);
}

  cout << d << endl;
  gettimeofday(&tv, NULL);
  stop = (tv.tv_sec) * 1000 + (tv.tv_usec)/1000;
  cout << (stop - start) / 1000.0 << "seconds" << endl;

  return 0;
}
