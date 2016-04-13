// http://bisqwit.iki.fi/story/howto/openmp/
// https://computing.llnl.gov/tutorials/openMP/#CFormat
// http://luiarthur.github.io/assets/ams250/notes/notes5.pdf
#include <iostream>
#include <armadillo>
#include <omp.h>
#include <cblas.h>

using namespace std;
using namespace arma;

mat mvrnorm(mat M, mat S) {
  // in R:  mvrnorm <- function(M,S,n=nrow(S)) M + t(chol(S)) %*% rnorm(n)
  int n = M.n_rows;
  mat e = randn(n);
  return M + chol(S).t()*e;
}


int main(int argc, char** argv) {
  // timer //////////////////
  time_t start,stop;       //
  struct timeval tv;       //
  gettimeofday(&tv, NULL); //
  ///////////////////////////

  // set number of threads ////////////////////////
  int num_threads_openblas = atoi(argv[1]);      //
  int num_threads_openmp = atoi(argv[2]);        //
  openblas_set_num_threads(num_threads_openblas);//
  omp_set_num_threads(num_threads_openmp);       //
  /////////////////////////////////////////////////

  int j;
  int J = 1000;
  int n = 1000;
  vec m = linspace(0,n-1,n);
  mat S = eye(n,n);
  mat out = zeros<mat>(n,J);

  start = (tv.tv_sec)*1000 + (tv.tv_usec)/1000;

  #pragma omp parallel shared(out) private(j)
  {
    #pragma omp for 
    for (j=0; j<J; j++) {
      out.col(j) = mvrnorm(m,S);
    }
  }

  cout << mean(out,1);

  // timer //////////////////////////////////////////////////////
  gettimeofday(&tv, NULL);                                     //
  stop = (tv.tv_sec) * 1000 + (tv.tv_usec)/1000;               //
  cout << endl << (stop - start) / 1000.0 << "seconds" << endl;//
  ///////////////////////////////////////////////////////////////
  return 0;
}
