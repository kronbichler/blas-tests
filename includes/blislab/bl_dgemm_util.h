/*
 * --------------------------------------------------------------------------
 * BLISLAB 
 * --------------------------------------------------------------------------
 * Copyright (C) 2016, The University of Texas at Austin
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name of The University of Texas nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * bl_dgemm.h
 *
 *
 * Purpose:
 * this header file contains all function prototypes.
 *
 * Todo:
 *
 *
 * Modification:
 *
 * 

#ifndef BLISLAB_DGEMM_H
#define BLISLAB_DGEMM_H

#include <math.h>
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include "bl_config.h"

#define MIN( i, j ) ( (i)<(j) ? (i): (j) )

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
