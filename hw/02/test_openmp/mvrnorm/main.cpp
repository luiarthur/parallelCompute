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

  int n = 10;
  vec m = linspace(0,n-1,n);
  mat S = eye(n,n);

  cout << mvrnorm(m,S);

  return 0;
}
