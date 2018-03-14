

#ifndef BLISLAB_DGEMM_SIMPLE_H
#define BLISLAB_DGEMM_SIMPLE_H

void bl_dgemm(int m, int n, int k, double *A, int lda, double *B, int ldb, double *C, int ldc){
    
  int    i, j, p;

  // Early return if possible
  if ( m == 0 || n == 0 || k == 0 ) {
    printf( "bl_dgemm(): early return\n" );
    return;
  }

  for ( j = 0; j < n; j ++ ) {              // Start 2-nd loop
      for ( p = 0; p < k; p ++ ) {          // Start 1-st loop
          for ( i = 0; i < m; i ++ ) {      // Start 0-th loop

              //C[ j * ldc + i ] += A[ p * lda + i ] * B[ j * ldb + p ];
              C( i, j ) += A( i, p ) * B( p, j ); //Each operand is a MACRO defined in bl_dgemm() function.

          }                                 // End   0-th loop
      }                                     // End   1-st loop
  }                                         // End   2-nd loop

}

#endif