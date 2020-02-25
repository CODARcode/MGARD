/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */
#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include "mgard_cuda_helper.h"
#include "scanLargeArray_kernel.cu"
#include <assert.h>

inline bool 
isPowerOfTwo(int n)
{
    return ((n&(n-1))==0) ;
}

inline int 
floorPow2(int n)
{
#ifdef WIN32
    // method 2
    return 1 << (int)logb((double)n);
#else
    // method 1
    // double nf = (double)n;
    // return 1 << (((*(int*)&nf) >> 23) - 127); 
    int exp;
    frexp((double)n, &exp);
    return 1 << (exp - 1);
#endif
}

#define BLOCK_SIZE 256

double** g_scanBlockSums;
unsigned int g_numEltsAllocated = 0;
unsigned int g_numLevelsAllocated = 0;

void preallocBlockSums(unsigned int maxNumElements)
{
    assert(g_numEltsAllocated == 0); // shouldn't be called 

    g_numEltsAllocated = maxNumElements;

    unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
    unsigned int numElts = maxNumElements;

    int level = 0;

    do
    {       
        unsigned int numBlocks = 
            max(1, (int)ceil((double)numElts / (2.f * blockSize)));
        if (numBlocks > 1)
        {
            level++;
        }
        numElts = numBlocks;
    } while (numElts > 1);

    g_scanBlockSums = (double**) malloc(level * sizeof(double*));
    g_numLevelsAllocated = level;
    
    numElts = maxNumElements;
    level = 0;
    
    do
    {       
        unsigned int numBlocks = 
            max(1, (int)ceil((double)numElts / (2.f * blockSize)));
        if (numBlocks > 1) 
        {
            cudaMalloc((void**) &g_scanBlockSums[level++],  
                                      numBlocks * sizeof(double));
        }
        numElts = numBlocks;
    } while (numElts > 1);

    //CUT_CHECK_ERROR("preallocBlockSums");
    gpuErrchk(cudaGetLastError ());
}

void deallocBlockSums()
{
    for (int i = 0; i < g_numLevelsAllocated; i++)
    {
        cudaFree(g_scanBlockSums[i]);
    }

    //CUT_CHECK_ERROR("deallocBlockSums");
    gpuErrchk(cudaGetLastError ());

    free((void**)g_scanBlockSums);

    g_scanBlockSums = 0;
    g_numEltsAllocated = 0;
    g_numLevelsAllocated = 0;
}


void postscanArrayRecursive(double *outArray, 
                           const double *inArray, 
                           int numElements, 
                           int level)
{
    unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
    unsigned int numBlocks = 
        max(1, (int)ceil((double)numElements / (2.f * blockSize)));
    unsigned int numThreads;

    if (numBlocks > 1)
        numThreads = blockSize;
    else if (isPowerOfTwo(numElements))
        numThreads = numElements / 2;
    else
        numThreads = floorPow2(numElements);

    unsigned int numEltsPerBlock = numThreads * 2;

    // if this is a non-power-of-2 array, the last block will be non-full
    // compute the smallest power of 2 able to compute its scan.
    unsigned int numEltsLastBlock = 
        numElements - (numBlocks-1) * numEltsPerBlock;
    unsigned int numThreadsLastBlock = max(1, numEltsLastBlock / 2);
    unsigned int np2LastBlock = 0;
    unsigned int sharedMemLastBlock = 0;
    
    if (numEltsLastBlock != numEltsPerBlock)
    {
        np2LastBlock = 1;

        if(!isPowerOfTwo(numEltsLastBlock))
            numThreadsLastBlock = floorPow2(numEltsLastBlock);    
        
        unsigned int extraSpace = (2 * numThreadsLastBlock) / NUM_BANKS;
        sharedMemLastBlock = 
            sizeof(double) * (2 * numThreadsLastBlock + extraSpace);
    }

    // padding space is used to avoid shared memory bank conflicts
    unsigned int extraSpace = numEltsPerBlock / NUM_BANKS;
    unsigned int sharedMemSize = 
        sizeof(double) * (numEltsPerBlock + extraSpace);

#ifdef DEBUG
    if (numBlocks > 1)
    {
        assert(g_numEltsAllocated >= numElements);
    }
#endif

    // setup execution parameters
    // if NP2, we process the last block separately
    dim3  grid(max(1, numBlocks - np2LastBlock), 1, 1); 
    dim3  threads(numThreads, 1, 1);

    // make sure there are no CUDA errors before we start
    //CUT_CHECK_ERROR("prescanArrayRecursive before kernels");
    gpuErrchk(cudaGetLastError ());

    // execute the scan
    if (numBlocks > 1)
    {
        postscan<true, false><<< grid, threads, sharedMemSize >>>(outArray, 
                                                                 inArray, 
                                                                 g_scanBlockSums[level],
                                                                 numThreads * 2, 0, 0);
        //CUT_CHECK_ERROR("prescanWithBlockSums");
        gpuErrchk(cudaGetLastError ());
        if (np2LastBlock)
        {
            postscan<true, true><<< 1, numThreadsLastBlock, sharedMemLastBlock >>>
                (outArray, inArray, g_scanBlockSums[level], numEltsLastBlock, 
                 numBlocks - 1, numElements - numEltsLastBlock);
            //CUT_CHECK_ERROR("prescanNP2WithBlockSums");
            gpuErrchk(cudaGetLastError ());

        }

        // After scanning all the sub-blocks, we are mostly done.  But now we 
        // need to take all of the last values of the sub-blocks and scan those.  
        // This will give us a new value that must be sdded to each block to 
        // get the final results.
        // recursive (CPU) call
        postscanArrayRecursive(g_scanBlockSums[level], 
                              g_scanBlockSums[level], 
                              numBlocks, 
                              level+1);

        uniformMultiply<<< grid, threads >>>(outArray, 
                                        g_scanBlockSums[level], 
                                        numElements - numEltsLastBlock, 
                                        0, 0);
        //CUT_CHECK_ERROR("uniformAdd");
        gpuErrchk(cudaGetLastError ());
        if (np2LastBlock)
        {
            uniformMultiply<<< 1, numThreadsLastBlock >>>(outArray, 
                                                     g_scanBlockSums[level], 
                                                     numEltsLastBlock, 
                                                     numBlocks - 1, 
                                                     numElements - numEltsLastBlock);
            //CUT_CHECK_ERROR("uniformAdd");
            gpuErrchk(cudaGetLastError ());
        }
    }
    else if (isPowerOfTwo(numElements))
    {
        postscan<false, false><<< grid, threads, sharedMemSize >>>(outArray, inArray,
                                                                  0, numThreads * 2, 0, 0);
        //CUT_CHECK_ERROR("prescan");
        gpuErrchk(cudaGetLastError ());
    }
    else
    {
         postscan<false, true><<< grid, threads, sharedMemSize >>>(outArray, inArray, 
                                                                  0, numElements, 0, 0);
         //CUT_CHECK_ERROR("prescanNP2");
         gpuErrchk(cudaGetLastError ());
    }
}

void postscanArray(double *outArray, double *inArray, int numElements)
{
    postscanArrayRecursive(outArray, inArray, numElements, 0);
}


#endif // _PRESCAN_CU_
