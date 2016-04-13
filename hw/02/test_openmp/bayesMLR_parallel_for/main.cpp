// http://bisqwit.iki.fi/story/howto/openmp/
// https://computing.llnl.gov/tutorials/openMP/#CFormat
// http://luiarthur.github.io/assets/ams250/notes/notes5.pdf
#include <iostream>
#include <armadillo>
#include <omp.h>
#include <cblas.h>

using namespace std;
using namespace arma;

const double pi  =3.141592653589793238462;
int n,k;  // nrow(X),ncol(X)
mat z,X,y,XXi,Xt;
const double a=1;
const double b=1;
const int B=2*pow(10,5);
const double s2 = 10;

double ll(mat be, double sig2) {
  mat c(1,k), out(k,1);

  c = y-X*be;
  out = (c.t()*c / sig2 + n*log(sig2))/-2;
  
  return as_scalar(out);
}

double lpb(mat be) {
  return as_scalar(-be.t()*XXi*be/(2*s2));
}

double lps(double sig2) {
  return (a-1) * log(sig2) - sig2/b;
}

mat mvrnorm(mat M, mat S) {
  int n = M.n_rows;
  mat e = randn(n);
  return M + chol(S).t()*e;
}
// mvrnorm <- function(M,S,n=nrow(S)) M + t(chol(S)) %*% rnorm(n)


vec bayesMLR() {
  mat mle;

  //posteriors:
  mat bb; 
  mat ss;

  // candidate sigma:
  mat csb;
  const double css = 1;

  //candidate values:
  mat candb;
  double cands;
  
  //acceptance rates:
  int accb = 0;
  int accs = 0;

  //metropolis ratio:
  double q;

  n = z.n_rows;
  k = z.n_cols-1;

  y = z.col(0);
  X = z.cols(1,k); // columns 2 to k+1 of z
  Xt = X.t();
  XXi = (Xt*X).i();
  csb = 4*XXi;
  mle = XXi * Xt * y;
  
  bb.set_size(B,k);
  ss.set_size(B,1);
  bb.zeros();
  ss.ones();

  //current vals:
  mat bc = bb.row(0);
  double sc = 1.0;

  //cout << "Starting Metropolis:" <<endl;

  for (int i=1; i<B; i++) {
    // Set Initial Values:
    bb.row(i) = bb.row(i-1);
    ss.at(i,0) = sc;
    bc = bb.row(i).t();

    //Update Beta:
    candb = mvrnorm(bc,csb);
    q = ll(candb,sc)+lpb(candb) -ll(bc,sc)-lpb(bc);
    if (q>log(randu())) {
      bc = candb;
      bb.row(i) = bc.t();
      accb++;
    }

    //Update sigma2:
    cands = randn()*sqrt(css)+sc;
    if (cands>0){
      q = ll(bc,cands)+lps(cands) -ll(bc,sc)-lps(sc);
      if (q>log(randu())) {
        sc = cands;
        accs++;
      }
    }
    
    //if (i % (B/10) == 0 || i == 1) fprintf(stderr,"\r%d%s",i*100/B,"%");
  }

  return mean(bb.rows(90000,100000-1)).t();
}


int main() {
  z.load("dat.txt");
  time_t start,stop;
  struct timeval tv;
  gettimeofday(&tv, NULL);
  start = (tv.tv_sec)*1000 + (tv.tv_usec)/1000;

  int tid;
  int th = 1;
  omp_set_num_threads(th);
  openblas_set_num_threads(1);
  mat out = zeros<mat>(10,th);

  cout << "COMPUTING..." << endl;
  #pragma omp parallel private(tid)
  {
    tid = omp_get_thread_num();
    cout << tid << " ";

    out.col(tid) = bayesMLR();
  }

  cout << endl << out ;
  gettimeofday(&tv, NULL);
  stop = (tv.tv_sec) * 1000 + (tv.tv_usec)/1000;
  cout << (stop - start) / 1000.0 << "seconds" << endl;

  return 0;
}
