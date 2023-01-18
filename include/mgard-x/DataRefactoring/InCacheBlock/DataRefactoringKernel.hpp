/*
 * Copyright 2023, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: Jan. 15, 2023
 */

#ifndef MGARD_X_IN_CACHE_BLOCK_DATA_REFACTORING_KERNEL_TEMPLATE
#define MGARD_X_IN_CACHE_BLOCK_DATA_REFACTORING_KERNEL_TEMPLATE

#include "../../RuntimeX/RuntimeX.h"

#include "IndexTable8x8x8.hpp"

#define DECOMPOSE 0
#define RECOMPOSE 1

namespace mgard_x {

namespace data_refactoring {

namespace in_cache_block {

template <DIM D, typename T, SIZE Z, SIZE Y, SIZE X, OPTION OP,
          typename DeviceType>
class DataRefactoringFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT DataRefactoringFunctor() {}
  MGARDX_CONT DataRefactoringFunctor(SubArray<D, T, DeviceType> v) : v(v) {
    Functor<DeviceType>();
  }

  // MGARDX_EXEC bool Mask1D(SIZE z, SIZE y, SIZE x) {
  //   // if (edgeCase >= 256) { printf("GetNumberOfPrimitives out of range\n");
  //   // edgeCase = 0; }
  //   static constexpr bool mask[5][5][5] = {
  //                   {{false,true, false, true, false},
  //                    {true, false, true, false, true},
  //                    {false,true, false, true, false},
  //                    {true, false, true, false, true},
  //                    {false,true, false, true, false}},

  //                   {{true,false, true, false, true},
  //                    {false, false, false, false, false},
  //                    {true,false, true, false, true},
  //                    {false, false, false, false, false},
  //                    {true,false, true, false, true}},

  //                   {{false,true, false, true, false},
  //                    {true, false, true, false, true},
  //                    {false,true, false, true, false},
  //                    {true, false, true, false, true},
  //                    {false,true, false, true, false}},

  //                   {{true,false, true, false, true},
  //                    {false, false, false, false, false},
  //                    {true,false, true, false, true},
  //                    {false, false, false, false, false},
  //                    {true,false, true, false, true}},

  //                   {{false,true, false, true, false},
  //                    {true, false, true, false, true},
  //                    {false,true, false, true, false},
  //                    {true, false, true, false, true},
  //                    {false,true, false, true, false}}
  //                   };

  //   return mask[z][y][x];
  // }

  // MGARDX_EXEC int8_t const* LeftIndex1D(SIZE z, SIZE y, SIZE x) {
  //   // if (edgeCase >= 256) { printf("GetNumberOfPrimitives out of range\n");
  //   // edgeCase = 0; }
  //   static constexpr int8_t index[5][5][5][3] = {
  //                   {{{0, 0, 0},{0, 0, -1}, {0, 0, 0}, {0, 0, -1}, {0, 0,
  //                   0}},
  //                    {{0, -1, 0}, {0, 0, 0}, {0, -1, 0}, {0, 0, 0}, {0, -1,
  //                    0}},
  //                    {{0, 0, 0},{0, 0, -1}, {0, 0, 0}, {0, 0, -1}, {0, 0,
  //                    0}},
  //                    {{0, -1, 0}, {0, 0, 0}, {0, -1, 0}, {0, 0, 0}, {0, -1,
  //                    0}},
  //                    {{0, 0, 0},{0, 0, -1}, {0, 0, 0}, {0, 0, -1}, {0, 0,
  //                    0}}},

  //                   {{{-1, 0, 0},{0, 0, 0}, {-1, 0, 0}, {0, 0, 0}, {-1, 0,
  //                   0}},
  //                    {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{-1, 0, 0},{0, 0, 0}, {-1, 0, 0}, {0, 0, 0}, {-1, 0,
  //                    0}},
  //                    {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{-1, 0, 0},{0, 0, 0}, {-1, 0, 0}, {0, 0, 0}, {-1, 0,
  //                    0}}},

  //                   {{{0, 0, 0},{0, 0, -1}, {0, 0, 0}, {0, 0, -1}, {0, 0,
  //                   0}},
  //                    {{0, -1, 0}, {0, 0, 0}, {0, -1, 0}, {0, 0, 0}, {0, -1,
  //                    0}},
  //                    {{0, 0, 0},{0, 0, -1}, {0, 0, 0}, {0, 0, -1}, {0, 0,
  //                    0}},
  //                    {{0, -1, 0}, {0, 0, 0}, {0, -1, 0}, {0, 0, 0}, {0, -1,
  //                    0}},
  //                    {{0, 0, 0},{0, 0, -1}, {0, 0, 0}, {0, 0, -1}, {0, 0,
  //                    0}}},

  //                   {{{-1, 0, 0},{0, 0, 0}, {-1, 0, 0}, {0, 0, 0}, {-1, 0,
  //                   0}},
  //                    {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{-1, 0, 0},{0, 0, 0}, {-1, 0, 0}, {0, 0, 0}, {-1, 0,
  //                    0}},
  //                    {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{-1, 0, 0},{0, 0, 0}, {-1, 0, 0}, {0, 0, 0}, {-1, 0,
  //                    0}}},

  //                   {{{0, 0, 0},{0, 0, -1}, {0, 0, 0}, {0, 0, -1}, {0, 0,
  //                   0}},
  //                    {{0, -1, 0}, {0, 0, 0}, {0, -1, 0}, {0, 0, 0}, {0, -1,
  //                    0}},
  //                    {{0, 0, 0},{0, 0, -1}, {0, 0, 0}, {0, 0, -1}, {0, 0,
  //                    0}},
  //                    {{0, -1, 0}, {0, 0, 0}, {0, -1, 0}, {0, 0, 0}, {0, -1,
  //                    0}},
  //                    {{0, 0, 0},{0, 0, -1}, {0, 0, 0}, {0, 0, -1}, {0, 0,
  //                    0}}}
  //                   };

  //   return index[z][y][x];
  // }

  // MGARDX_EXEC int8_t const* RightIndex1D(SIZE z, SIZE y, SIZE x) {
  //   // if (edgeCase >= 256) { printf("GetNumberOfPrimitives out of range\n");
  //   // edgeCase = 0; }
  //   static constexpr int8_t index[5][5][5][3] = {
  //                   {{{0, 0, 0},{0, 0, 1}, {0, 0, 0}, {0, 0, 1}, {0, 0, 0}},
  //                    {{0, 1, 0}, {0, 0, 0}, {0, 1, 0}, {0, 0, 0}, {0, 1, 0}},
  //                    {{0, 0, 0},{0, 0, 1}, {0, 0, 0}, {0, 0, 1}, {0, 0, 0}},
  //                    {{0, 1, 0}, {0, 0, 0}, {0, 1, 0}, {0, 0, 0}, {0, 1, 0}},
  //                    {{0, 0, 0},{0, 0, 1}, {0, 0, 0}, {0, 0, 1}, {0, 0, 0}}},

  //                   {{{1, 0, 0},{0, 0, 0}, {1, 0, 0}, {0, 0, 0}, {1, 0, 0}},
  //                    {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{1, 0, 0},{0, 0, 0}, {1, 0, 0}, {0, 0, 0}, {1, 0, 0}},
  //                    {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{1, 0, 0},{0, 0, 0}, {1, 0, 0}, {0, 0, 0}, {1, 0, 0}}},

  //                   {{{0, 0, 0},{0, 0, 1}, {0, 0, 0}, {0, 0, 1}, {0, 0, 0}},
  //                    {{0, 1, 0}, {0, 0, 0}, {0, 1, 0}, {0, 0, 0}, {0, 1, 0}},
  //                    {{0, 0, 0},{0, 0, 1}, {0, 0, 0}, {0, 0, 1}, {0, 0, 0}},
  //                    {{0, 1, 0}, {0, 0, 0}, {0, 1, 0}, {0, 0, 0}, {0, 1, 0}},
  //                    {{0, 0, 0},{0, 0, 1}, {0, 0, 0}, {0, 0, 1}, {0, 0, 0}}},

  //                   {{{1, 0, 0},{0, 0, 0}, {1, 0, 0}, {0, 0, 0}, {1, 0, 0}},
  //                    {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{1, 0, 0},{0, 0, 0}, {1, 0, 0}, {0, 0, 0}, {1, 0, 0}},
  //                    {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{1, 0, 0},{0, 0, 0}, {1, 0, 0}, {0, 0, 0}, {1, 0, 0}}},

  //                   {{{0, 0, 0},{0, 0, 1}, {0, 0, 0}, {0, 0, 1}, {0, 0, 0}},
  //                    {{0, 1, 0}, {0, 0, 0}, {0, 1, 0}, {0, 0, 0}, {0, 1, 0}},
  //                    {{0, 0, 0},{0, 0, 1}, {0, 0, 0}, {0, 0, 1}, {0, 0, 0}},
  //                    {{0, 1, 0}, {0, 0, 0}, {0, 1, 0}, {0, 0, 0}, {0, 1, 0}},
  //                    {{0, 0, 0},{0, 0, 1}, {0, 0, 0}, {0, 0, 1}, {0, 0, 0}}}
  //                   };

  //   return index[z][y][x];
  // }

  // MGARDX_EXEC bool Mask2D(SIZE z, SIZE y, SIZE x) {
  //   // if (edgeCase >= 256) { printf("GetNumberOfPrimitives out of range\n");
  //   // edgeCase = 0; }
  //   static constexpr bool mask[5][5][5] = {
  //                   {{false,false, false, false, false},
  //                    {false, true, false, true, false},
  //                    {false,false, false, false, false},
  //                    {false, true, false, true, false},
  //                    {false,false, false, false, false}},

  //                   {{false,true, false, true, false},
  //                    {true, false, true, false, true},
  //                    {false,true, false, true, false},
  //                    {true, false, true, false, true},
  //                    {false,true, false, true, false}},

  //                   {{false,false, false, false, false},
  //                    {false, true, false, true, false},
  //                    {false,false, false, false, false},
  //                    {false, true, false, true, false},
  //                    {false,false, false, false, false}},

  //                   {{false,true, false, true, false},
  //                    {true, false, true, false, true},
  //                    {false,true, false, true, false},
  //                    {true, false, true, false, true},
  //                    {false,true, false, true, false}},

  //                   {{false,false, false, false, false},
  //                    {false, true, false, true, false},
  //                    {false,false, false, false, false},
  //                    {false, true, false, true, false},
  //                    {false,false, false, false, false}}
  //                   };

  //   return mask[z][y][x];
  // }

  // MGARDX_EXEC int8_t const* LeftIndex2D(SIZE z, SIZE y, SIZE x) {
  //   // if (edgeCase >= 256) { printf("GetNumberOfPrimitives out of range\n");
  //   // edgeCase = 0; }
  //   static constexpr int8_t index[5][5][5][3] = {
  //                   {{{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0}, {0, -1, 0}, {0, 0, 0}, {0, -1, 0}, {0, 0,
  //                    0}},
  //                    {{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0}, {0, -1, 0}, {0, 0, 0}, {0, -1, 0}, {0, 0,
  //                    0}},
  //                    {{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},

  //                   {{{0, 0, 0},{-1, 0, 0}, {0, 0, 0}, {-1, 0, 0}, {0, 0,
  //                   0}},
  //                    {{-1, 0, 0}, {0, 0, 0}, {-1, 0, 0}, {0, 0, 0}, {-1, 0,
  //                    0}},
  //                    {{0, 0, 0},{-1, 0, 0}, {0, 0, 0}, {-1, 0, 0}, {0, 0,
  //                    0}},
  //                    {{-1, 0, 0}, {0, 0, 0}, {-1, 0, 0}, {0, 0, 0}, {-1, 0,
  //                    0}},
  //                    {{0, 0, 0},{-1, 0, 0}, {0, 0, 0}, {-1, 0, 0}, {0, 0,
  //                    0}}},

  //                   {{{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0}, {0, -1, 0}, {0, 0, 0}, {0, -1, 0}, {0, 0,
  //                    0}},
  //                    {{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0}, {0, -1, 0}, {0, 0, 0}, {0, -1, 0}, {0, 0,
  //                    0}},
  //                    {{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},

  //                   {{{0, 0, 0},{-1, 0, 0}, {0, 0, 0}, {-1, 0, 0}, {0, 0,
  //                   0}},
  //                    {{-1, 0, 0}, {0, 0, 0}, {-1, 0, 0}, {0, 0, 0}, {-1, 0,
  //                    0}},
  //                    {{0, 0, 0},{-1, 0, 0}, {0, 0, 0}, {-1, 0, 0}, {0, 0,
  //                    0}},
  //                    {{-1, 0, 0}, {0, 0, 0}, {-1, 0, 0}, {0, 0, 0}, {-1, 0,
  //                    0}},
  //                    {{0, 0, 0},{-1, 0, 0}, {0, 0, 0}, {-1, 0, 0}, {0, 0,
  //                    0}}},

  //                   {{{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0}, {0, -1, 0}, {0, 0, 0}, {0, -1, 0}, {0, 0,
  //                    0}},
  //                    {{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0}, {0, -1, 0}, {0, 0, 0}, {0, -1, 0}, {0, 0,
  //                    0}},
  //                    {{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}}
  //                   };

  //   return index[z][y][x];
  // }

  // MGARDX_EXEC int8_t const* RightIndex2D(SIZE z, SIZE y, SIZE x) {
  //   // if (edgeCase >= 256) { printf("GetNumberOfPrimitives out of range\n");
  //   // edgeCase = 0; }
  //   static constexpr int8_t index[5][5][5][3] = {
  //                   {{{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0}, {0, 1, 0}, {0, 0, 0}, {0, 1, 0}, {0, 0, 0}},
  //                    {{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0}, {0, 1, 0}, {0, 0, 0}, {0, 1, 0}, {0, 0, 0}},
  //                    {{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},

  //                   {{{0, 0, 0},{1, 0, 0}, {0, 0, 0}, {1, 0, 0}, {0, 0, 0}},
  //                    {{1, 0, 0}, {0, 0, 0}, {1, 0, 0}, {0, 0, 0}, {1, 0, 0}},
  //                    {{0, 0, 0},{1, 0, 0}, {0, 0, 0}, {1, 0, 0}, {0, 0, 0}},
  //                    {{1, 0, 0}, {0, 0, 0}, {1, 0, 0}, {0, 0, 0}, {1, 0, 0}},
  //                    {{0, 0, 0},{1, 0, 0}, {0, 0, 0}, {1, 0, 0}, {0, 0, 0}}},

  //                   {{{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0}, {0, 1, 0}, {0, 0, 0}, {0, 1, 0}, {0, 0, 0}},
  //                    {{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0}, {0, 1, 0}, {0, 0, 0}, {0, 1, 0}, {0, 0, 0}},
  //                    {{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},

  //                   {{{0, 0, 0},{1, 0, 0}, {0, 0, 0}, {1, 0, 0}, {0, 0, 0}},
  //                    {{1, 0, 0}, {0, 0, 0}, {1, 0, 0}, {0, 0, 0}, {1, 0, 0}},
  //                    {{0, 0, 0},{1, 0, 0}, {0, 0, 0}, {1, 0, 0}, {0, 0, 0}},
  //                    {{1, 0, 0}, {0, 0, 0}, {1, 0, 0}, {0, 0, 0}, {1, 0, 0}},
  //                    {{0, 0, 0},{1, 0, 0}, {0, 0, 0}, {1, 0, 0}, {0, 0, 0}}},

  //                   {{{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0}, {0, 1, 0}, {0, 0, 0}, {0, 1, 0}, {0, 0, 0}},
  //                    {{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0}, {0, 1, 0}, {0, 0, 0}, {0, 1, 0}, {0, 0, 0}},
  //                    {{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}}
  //                   };

  //   return index[z][y][x];
  // }

  // MGARDX_EXEC bool Mask3D(SIZE z, SIZE y, SIZE x) {
  //   static constexpr bool mask[5][5][5] = {
  //                   {{false,false, false, false, false},
  //                    {false, false, false, false, false},
  //                    {false,false, false, false, false},
  //                    {false, false, false, false, false},
  //                    {false,false, false, false, false}},

  //                   {{false,false, false, false, false},
  //                    {false, true, false, true, false},
  //                    {false,false, false, false, false},
  //                    {false, true, false, true, false},
  //                    {false,false, false, false, false}},

  //                   {{false,false, false, false, false},
  //                    {false, false, false, false, false},
  //                    {false,false, false, false, false},
  //                    {false, false, false, false, false},
  //                    {false,false, false, false, false}},

  //                   {{false,false, false, false, false},
  //                    {false, true, false, true, false},
  //                    {false,false, false, false, false},
  //                    {false, true, false, true, false},
  //                    {false,false, false, false, false}},

  //                   {{false,false, false, false, false},
  //                    {false, false, false, false, false},
  //                    {false,false, false, false, false},
  //                    {false, false, false, false, false},
  //                    {false,false, false, false, false}}
  //                   };

  //   return mask[z][y][x];
  // }

  // MGARDX_EXEC int8_t const* LeftIndex3D(SIZE z, SIZE y, SIZE x) {
  //   // if (edgeCase >= 256) { printf("GetNumberOfPrimitives out of range\n");
  //   // edgeCase = 0; }
  //   static constexpr int8_t index[5][5][5][3] = {
  //                   {{{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},

  //                   {{{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0}, {-1, 0, 0}, {0, 0, 0}, {-1, 0, 0}, {0, 0,
  //                    0}},
  //                    {{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0}, {-1, 0, 0}, {0, 0, 0}, {-1, 0, 0}, {0, 0,
  //                    0}},
  //                    {{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},

  //                   {{{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},

  //                   {{{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0}, {-1, 0, 0}, {0, 0, 0}, {-1, 0, 0}, {0, 0,
  //                    0}},
  //                    {{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0}, {-1, 0, 0}, {0, 0, 0}, {-1, 0, 0}, {0, 0,
  //                    0}},
  //                    {{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},

  //                   {{{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}}
  //                   };

  //   return index[z][y][x];
  // }

  // MGARDX_EXEC int8_t const* RightIndex3D(SIZE z, SIZE y, SIZE x) {
  //   static constexpr int8_t index[5][5][5][3] = {
  //                   {{{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},

  //                   {{{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0}, {1, 0, 0}, {0, 0, 0}, {1, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0}, {1, 0, 0}, {0, 0, 0}, {1, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},

  //                   {{{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},

  //                   {{{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0}, {1, 0, 0}, {0, 0, 0}, {1, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0}, {1, 0, 0}, {0, 0, 0}, {1, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},

  //                   {{{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
  //                    {{0, 0, 0},{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}}
  //                   };

  //   return index[z][y][x];
  // }

  MGARDX_EXEC void Operation1() {
    sm = (T *)FunctorBase<DeviceType>::GetSharedMemory();

    x = FunctorBase<DeviceType>::GetThreadIdX();
    y = FunctorBase<DeviceType>::GetThreadIdY();
    z = FunctorBase<DeviceType>::GetThreadIdZ();
    x_gl = FunctorBase<DeviceType>::GetBlockDimX() *
               FunctorBase<DeviceType>::GetBlockIdX() +
           x;
    y_gl = FunctorBase<DeviceType>::GetBlockDimY() *
               FunctorBase<DeviceType>::GetBlockIdY() +
           y;
    z_gl = FunctorBase<DeviceType>::GetBlockDimZ() *
               FunctorBase<DeviceType>::GetBlockIdZ() +
           z;
    // sm[get_idx(B, B, z, y, x)]= *v(z_gl, y_gl, x_gl);

    sm[get_idx(X, Y, z, y, x)] = *v(z_gl, y_gl, x_gl);
    int8_t const *left_index = LeftIndex1D8x8x8(z, y, x);
    int8_t const *right_index = RightIndex1D8x8x8(z, y, x);
    left = sm[get_idx(X, Y, z + left_index[0], y + left_index[1],
                      x + left_index[2])];
    right = sm[get_idx(X, Y, z + right_index[0], y + right_index[1],
                       x + right_index[2])];
    middle = sm[get_idx(X, Y, z, y, x)];
    middle = middle - (left + right) * (T)0.5;
    sm[get_idx(X, Y, z, y, x)] = middle;
  }

  MGARDX_EXEC void Operation2() {
    int8_t const *left_index = LeftIndex2D8x8x8(z, y, x);
    int8_t const *right_index = LeftIndex2D8x8x8(z, y, x);
    left = sm[get_idx(X, Y, z + left_index[0], y + left_index[1],
                      x + left_index[2])];
    right = sm[get_idx(X, Y, z + right_index[0], y + right_index[1],
                       x + right_index[2])];
    middle = sm[get_idx(X, Y, z, y, x)];
    middle = middle - (left + right) * (T)0.5;
  }

  MGARDX_EXEC void Operation3() {
    int8_t const *left_index = LeftIndex3D8x8x8(z, y, x);
    int8_t const *right_index = LeftIndex3D8x8x8(z, y, x);
    left = sm[get_idx(X, Y, z + left_index[0], y + left_index[1],
                      x + left_index[2])];
    right = sm[get_idx(X, Y, z + right_index[0], y + right_index[1],
                       x + right_index[2])];
    middle = sm[get_idx(X, Y, z, y, x)];
    middle = middle - (left + right) * (T)0.5;
  }

  MGARDX_EXEC void Operation4() {
    T a = sm[x];
    T b = sm[x];
    T c = sm[x];
    const T h1 = 1.0 / 6.0, h2 = 1.0 / 6.0;
    T tb = a * h1 + b * (h1 + h2) + c * h2;
    *v(z_gl, y_gl, x_gl) = tb;
  }

  MGARDX_EXEC void Operation5() {
    T ta = sm[x];
    T tb = sm[x];
    T tc = sm[x];
    T r1 = 0.5, r4 = 0.5;
    tb += ta * r1 + tc * r4;
    *v(z_gl, y_gl, x_gl) = tb;
  }

  MGARDX_EXEC void Operation6() {
    T a = sm[x];
    T b = sm[x];
    T c = sm[x];
    const T h1 = 1.0 / 6.0, h2 = 1.0 / 6.0;
    T tb = a * h1 + b * (h1 + h2) + c * h2;
    *v(z_gl, y_gl, x_gl) = tb;
  }

  MGARDX_EXEC void Operation7() {
    T ta = sm[x];
    T tb = sm[x];
    T tc = sm[x];
    T r1 = 0.5, r4 = 0.5;
    tb += ta * r1 + tc * r4;
    *v(z_gl, y_gl, x_gl) = tb;
  }

  MGARDX_EXEC void Operation8() {
    T a = sm[x];
    T b = sm[x];
    T c = sm[x];
    const T h1 = 1.0 / 6.0, h2 = 1.0 / 6.0;
    T tb = a * h1 + b * (h1 + h2) + c * h2;
    *v(z_gl, y_gl, x_gl) = tb;
  }

  MGARDX_EXEC void Operation9() {
    T ta = sm[x];
    T tb = sm[x];
    T tc = sm[x];
    T r1 = 0.5, r4 = 0.5;
    tb += ta * r1 + tc * r4;
    *v(z_gl, y_gl, x_gl) = tb;
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = Z * Y * X * sizeof(T);
    return size;
  }

private:
  SubArray<D, T, DeviceType> v;
  T *sm;
  int z, y, x, z_gl, y_gl, x_gl;
  T left, right, middle;
};

template <DIM D, typename T, SIZE Z, SIZE Y, SIZE X, OPTION OP,
          typename DeviceType>
class DataRefactoringKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "lwpk";
  MGARDX_CONT
  DataRefactoringKernel(SubArray<D, T, DeviceType> v) : v(v) {}

  MGARDX_CONT Task<DataRefactoringFunctor<D, T, Z, Y, X, OP, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = DataRefactoringFunctor<D, T, Z, Y, X, OP, DeviceType>;
    FunctorType functor(v);

    SIZE total_thread_z = 1;
    SIZE total_thread_y = 1;
    SIZE total_thread_x = 1;
    if (D >= 3)
      total_thread_z = v.shape(D - 3);
    if (D >= 2)
      total_thread_y = v.shape(D - 2);
    total_thread_x = v.shape(D - 1);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = Z;
    tby = Y;
    tbx = X;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);
    for (int d = D - 4; d >= 0; d--) {
      gridx *= v.shape(d);
    }
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SubArray<D, T, DeviceType> v;
};

} // namespace in_cache_block

} // namespace data_refactoring

} // namespace mgard_x

#endif