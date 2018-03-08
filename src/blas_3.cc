#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>

#ifdef LIKWID_PERFMON
    #include <likwid.h>
#endif

#include "../includes/blislab/bl_dgemm.h"

#define USE_SET_DIFF 1
#define TOLERANCE 1E-10
void computeError( int ldc, int ldc_ref, int m, int n, double *C, double *C_ref ){
    for (int i = 0; i < m; i ++ ) 
        for (int j = 0; j < n; j ++ ) 
            if ( fabs( C( i, j ) - C_ref( i, j ) ) > TOLERANCE ) {
                printf( "C[ %d ][ %d ] != C_ref, %E, %E\n", i, j, C( i, j ), C_ref( i, j ) );
                break;
            }

}

void run_dgemm( int m, int n, int k, int nrepeats, bool check) {
    int    i, j, p;
    double blis_dgemm_time = std::numeric_limits<double>::max(); 
    double ref_dgemm_time  = blis_dgemm_time;

    double* A    = (double*)malloc( sizeof(double) * m * k );
    double* B    = (double*)malloc( sizeof(double) * k * n );

    const int lda = m;
    const int ldb = k;
    const int ldc =
#ifdef DGEMM_MR
              ( ( m - 1 ) / DGEMM_MR + 1 ) * DGEMM_MR;
#else
              m;
#endif
    int ldc_ref = m;
    double* C     = bl_malloc_aligned( ldc, n + 4, sizeof(double) );

    srand48 (time(NULL));

    // Randonly generate points in [ 0, 1 ].
    for ( p = 0; p < k; p ++ ) 
        for ( i = 0; i < m; i ++ ) 
            A( i, p ) = static_cast<double>(rand())/RAND_MAX;	
            
    for ( j = 0; j < n; j ++ )
        for ( p = 0; p < k; p ++ )
            B( p, j ) = static_cast<double>(rand())/RAND_MAX;

    for ( j = 0; j < n; j ++ )
        for ( i = 0; i < m; i ++ )
            C( i, j ) = static_cast<double>(rand())/RAND_MAX;

    double* C_ref;
    if(check){
        C_ref = (double*)malloc( sizeof(double) * m * n );
        for ( j = 0; j < n; j ++ )
            for ( i = 0; i < m; i ++ )
                C_ref( i, j ) = static_cast<double>(rand())/RAND_MAX;	
    }

    
#ifdef LIKWID_PERFMON
        char name[256];
        sprintf ( name, "size-%d-%d-%d", m, n, k );
        LIKWID_MARKER_START(name);
#endif
    
    for ( i = 0; i < nrepeats; i ++ ) {
        auto start = std::chrono::high_resolution_clock::now();
        bl_dgemm( m, n, k, A, lda, B, ldb, C, ldc );
        blis_dgemm_time = std::min(
                blis_dgemm_time,
                std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-start).count());
    }
        
#ifdef LIKWID_PERFMON
        LIKWID_MARKER_STOP(name);
#endif

    // Compute overall floating point operations.
    const double flops = ( m * n / ( 1000.0 * 1000.0 * 1000.0 ) ) * ( 2 * k );
    
    if(check){
        for ( i = 0; i < nrepeats; i ++ ) {
            auto start = std::chrono::high_resolution_clock::now();
            bl_dgemm_ref( m, n, k, A, lda, B, ldb, C_ref, ldc_ref );
            ref_dgemm_time = std::min(
                    ref_dgemm_time,
                    std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-start).count());
        }

        computeError( ldc, ldc_ref, m, n, C, C_ref );
        free(C_ref);
        printf( ">>> (0) %5d %5d %5d %5.2lf %5.2lf\n", 
                m, n, k, flops / blis_dgemm_time, flops / ref_dgemm_time );
    } else {
        printf("Test C=A*B with A=%dX%d B=%dX%d\n", m,k,k,n);
        printf( "Serial  1 thread %15.5lf GFLOPs/s\n", 
                flops / blis_dgemm_time );
    }

    // clean up
    free(A); free(B); free(C);
}

int main( int argc, char *argv[] ) {
    
  // process input arguments: size m:
  int m = 100;
  if (argc > 1) m = std::atoi(argv[1]);
  
  // ... size n:
  int n = m;
  if (argc > 2) n = std::atoi(argv[2]);
  
  // ... size k:
  int k = m;
  if (argc > 3) k = std::atoi(argv[3]);
  
  // ... number of repetitions:
  std::size_t n_repetitions = 3;
  if (argc > 4) n_repetitions = std::atoi(argv[4]);
    

  // initialize LIKWID as profiler
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
#pragma omp parallel
  {
    LIKWID_MARKER_THREADINIT;
  }
#endif

  // run dgemm 
  run_dgemm( m, n, k , n_repetitions, argc > 5);
    
  // finalize LIKWID
#ifdef LIKWID_PERFMON
    LIKWID_MARKER_CLOSE;
#endif

    return 0;
}
