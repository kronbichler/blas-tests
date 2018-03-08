#ifndef BLISLAB_CONFIG_H
#define BLISLAB_CONFIG_H

#define GEMM_SIMD_ALIGN_SIZE 32

#if BL_TEST == 0
    // Simple 3 loop version
#elif BL_TEST == 1
    // with micro kernel
    #define DGEMM_MC 4080
    #define DGEMM_NC 4080
    #define DGEMM_KC 4080
    #define DGEMM_MR 8
    #define DGEMM_NR 6
    #define BL_MICRO_KERNEL bl_dgemm_asm_8x6
#else
    // with micro kernel and blocking
    #define DGEMM_MC 72
    #define DGEMM_NC 4080
    #define DGEMM_KC 256
    #define DGEMM_MR 8
    #define DGEMM_NR 6
    #define BL_MICRO_KERNEL bl_dgemm_asm_8x6
#endif



#endif
