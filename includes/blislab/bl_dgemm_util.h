#ifndef BLISLAB_DGEMM_H
#define BLISLAB_DGEMM_H

#include <math.h>
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include "bl_config.h"

#define min( i, j ) ( (i)<(j) ? (i): (j) )

#define A( i, j )     A[ (j)*lda + (i) ]
#define B( i, j )     B[ (j)*ldb + (i) ]
#define C( i, j )     C[ (j)*ldc + (i) ]
#define C_ref( i, j ) C_ref[ (j)*ldc_ref + (i) ]

double *bl_malloc_aligned( int m, int n, int size){
    double *ptr;
    int    err;

    err = posix_memalign( (void**)&ptr, (size_t)GEMM_SIMD_ALIGN_SIZE, size * m * n );

    if ( err ) {
        printf( "bl_malloc_aligned(): posix_memalign() failures" );
        exit( 1 );    
    }

    return ptr;
}

void bl_printmatrix( double *A, int lda, int m, int n){
    int    i, j;
    for ( i = 0; i < m; i ++ ) {
        for ( j = 0; j < n; j ++ ) {
            printf("%lf\t", A[j * lda + i]);
        }
        printf("\n");
    }
}

static double gtod_ref_time_sec = 0.0;

double bl_clock_helper(){
    double the_time, norm_sec;
    struct timespec ts;

    clock_gettime( CLOCK_MONOTONIC, &ts );

    if ( gtod_ref_time_sec == 0.0 )
        gtod_ref_time_sec = ( double ) ts.tv_sec;

    norm_sec = ( double ) ts.tv_sec - gtod_ref_time_sec;

    the_time = norm_sec + ts.tv_nsec * 1.0e-9;

    return the_time;
}

double bl_clock( void ){
    return bl_clock_helper();
}

//#include <bl_dgemm.h>

#ifdef USE_BLAS
/* 
 * dgemm prototype
 *
 */ 
extern void dgemm_(char*, char*, int*, int*, int*, double*, double*, 
        int*, double*, int*, double*, double*, int*);
#endif

void bl_dgemm_ref( int m, int n, int k, double *XA, int lda, double *XB, int ldb, double *XC, int ldc){
    // Local variables.
    int    i, j, p;
    double alpha = 1.0, beta = 1.0;

    // Sanity check for early return.
    if ( m == 0 || n == 0 || k == 0 ) return;

    // Reference GEMM implementation.

#ifdef USE_BLAS
    dgemm_( "N", "N", &m, &n, &k, &alpha,
            XA, &lda, XB, &ldb, &beta, XC, &ldc );
#else
    for ( i = 0; i < m; i ++ ) {
        for ( j = 0; j < n; j ++ ) {
            for ( p = 0; p < k; p ++ ) {
                XC[ j * ldc + i ] += XA[ p * lda + i ] * XB[ j * ldb + p ];
            }
        }
    }
#endif

}

#endif
