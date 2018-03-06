#ifdef LIKWID_PERFMON
    #include <likwid.h>
#endif

#include "../includes/blislab/bl_dgemm.h"

#define USE_SET_DIFF 1
#define TOLERANCE 1E-10
void computeError( int ldc, int ldc_ref, int m, int n, double *C, double *C_ref ){
    int i, j;
    for ( i = 0; i < m; i ++ ) {
        for ( j = 0; j < n; j ++ ) {
            if ( fabs( C( i, j ) - C_ref( i, j ) ) > TOLERANCE ) {
                printf( "C[ %d ][ %d ] != C_ref, %E, %E\n", i, j, C( i, j ), C_ref( i, j ) );
                break;
            }
        }
    }

}

void run_dgemm( int m, int n, int k, int nrepeats) {
    int    i, j, p, nx;
    double *A, *B, *C, *C_ref;
    double tmp, error, flops;
    double ref_beg, ref_time, bl_dgemm_beg, bl_dgemm_time;
    int    lda, ldb, ldc, ldc_ref;
    double ref_rectime, bl_dgemm_rectime;

    A    = (double*)malloc( sizeof(double) * m * k );
    B    = (double*)malloc( sizeof(double) * k * n );

    lda = m;
    ldb = k;
#ifdef DGEMM_MR
    ldc = ( ( m - 1 ) / DGEMM_MR + 1 ) * DGEMM_MR;
#else
    ldc     = m;
#endif
    ldc_ref = m;
    C     = bl_malloc_aligned( ldc, n + 4, sizeof(double) );
    C_ref = (double*)malloc( sizeof(double) * m * n );

    srand48 (time(NULL));

    // Randonly generate points in [ 0, 1 ].
    for ( p = 0; p < k; p ++ ) {
        for ( i = 0; i < m; i ++ ) {
            A( i, p ) = (double)( drand48() );	
        }
    }
    for ( j = 0; j < n; j ++ ) {
        for ( p = 0; p < k; p ++ ) {
            B( p, j ) = (double)( drand48() );
        }
    }

    for ( j = 0; j < n; j ++ ) {
        for ( i = 0; i < m; i ++ ) {
            C_ref( i, j ) = (double)( 0.0 );	
                C( i, j ) = (double)( 0.0 );	
        }
    }

    
#ifdef LIKWID_PERFMON
        char name[256];
        sprintf ( name, "size-%d-%d-%d", m, n, k );
        LIKWID_MARKER_START(name);
#endif
    
    for ( i = 0; i < nrepeats; i ++ ) {
        bl_dgemm_beg = bl_clock();
        bl_dgemm( m, n, k, A, lda, B, ldb, C, ldc );
        bl_dgemm_time = bl_clock() - bl_dgemm_beg;

        if ( i == 0 ) {
            bl_dgemm_rectime = bl_dgemm_time;
        } else {
            bl_dgemm_rectime = bl_dgemm_time < bl_dgemm_rectime ? bl_dgemm_time : bl_dgemm_rectime;
        }
    }
        
#ifdef LIKWID_PERFMON
        LIKWID_MARKER_STOP(name);
#endif

    for ( i = 0; i < nrepeats; i ++ ) {
        ref_beg = bl_clock();
        bl_dgemm_ref( m, n, k, A, lda, B, ldb, C_ref, ldc_ref );
        ref_time = bl_clock() - ref_beg;

        if ( i == 0 ) {
            ref_rectime = ref_time;
        } else {
            ref_rectime = ref_time < ref_rectime ? ref_time : ref_rectime;
        }
    }

    computeError( ldc, ldc_ref, m, n, C, C_ref );

    // Compute overall floating point operations.
    flops = ( m * n / ( 1000.0 * 1000.0 * 1000.0 ) ) * ( 2 * k );

    printf( ">>> (0) %5d %5d %5d %5.2lf %5.2lf\n", 
            m, n, k, flops / bl_dgemm_rectime, flops / ref_rectime );

    free( A     );
    free( B     );
    free( C     );
    free( C_ref );
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
  run_dgemm( m, n, k , n_repetitions);
    
  // finalize LIKWID
#ifdef LIKWID_PERFMON
    LIKWID_MARKER_CLOSE;
#endif

    return 0;
}
