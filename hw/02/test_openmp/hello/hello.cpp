#include <stdio.h>
#include <omp.h>

int main() 
{
  int nthreads, tid;
  #pragma omp parallel private(tid) 
  {
    tid = omp_get_thread_num();
    printf("Hello, world! I am thread %d\n", tid);
    #pragma omp barrier
    if (tid == 0) 
    {
      nthreads = omp_get_num_threads();
      printf("Number of threads = %d\n",nthreads);
    }
  }
  return 0;
}
