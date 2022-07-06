#ifndef MGARD_X_DEVICE_ADAPTER_SERIAL_H
#define MGARD_X_DEVICE_ADAPTER_SERIAL_H

#include "DeviceAdapter.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <utility>

namespace mgard_x {

template <> struct SyncBlock<SERIAL> {
  MGARDX_EXEC static void Sync() {
    // do nothing
  }
};

template <> struct SyncGrid<SERIAL> {
  MGARDX_EXEC static void Sync() {
    // do nothing
  }
};

template <typename T, OPTION MemoryType, OPTION Scope>
struct Atomic<T, MemoryType, Scope, SERIAL> {
  MGARDX_EXEC static T Min(T *result, T value) {
    T old = *result;
    *result = std::min(*result, value);
    return old;
  }
  MGARDX_EXEC static T Max(T *result, T value) {
    T old = *result;
    *result = std::max(*result, value);
    return old;
  }
  MGARDX_EXEC static T Add(T *result, T value) {
    T old = *result;
    *result += value;
    return old;
  }
};

// based on de Bruijn sequence:
// 0000001000011000101000111001001011001101001111010101110110111111
// = 151050438420815295
static const int DeBruijnSubString[64] = {
    0,  1,  2,  4,  8,  16, 33, 3,  6,  12, 24, 49, 34, 5,  10, 20,
    40, 17, 35, 7,  14, 28, 57, 50, 36, 9,  18, 37, 11, 22, 44, 25,

    51, 38, 13, 26, 52, 41, 19, 39, 15, 30, 61, 58, 53, 42, 21, 43,

    23, 46, 29, 59, 54, 45, 27, 55, 47, 31, 63, 62, 60, 56, 48, 32};

static const int MultiplyDeBruijnBitPosition[64] = {
    0,  1,  2,  7,  3,  13, 8,  19, 4,  25, 14, 28, 9,  34, 20, 40,

    5,  17, 26, 38, 15, 46, 29, 48, 10, 31, 35, 54, 21, 50, 41, 57,

    63, 6,  12, 18, 24, 27, 33, 39, 16, 37, 45, 47, 30, 53, 49, 56,

    62, 11, 23, 32, 36, 44, 52, 55, 61, 22, 43, 51, 60, 42, 59, 58};

template <> struct Math<SERIAL> {
  template <typename T> MGARDX_EXEC static T Min(T a, T b) {
    return std::min(a, b);
  }
  template <typename T> MGARDX_EXEC static T Max(T a, T b) {
    return std::max(a, b);
  }

  MGARDX_EXEC static int ffs(unsigned int a) { return ffs(a); }
  MGARDX_EXEC static int ffsll(long long unsigned int a) {
    // return ffsll(a);
    int pos = 0;
    if (a == 0)
      return pos;
    pos = MultiplyDeBruijnBitPosition
        [((long long unsigned int)((a & -a) * 151050438420815295)) >> 58];
    // while (!(a & 1))
    // {
    //   a >>= 1;
    //   ++pos;
    // }
    // if (pos2 != pos) {printf("mismatch: %d %d", pos2, pos);}
    return pos + 1;
  }
  MGARDX_EXEC
  static uint64_t binary2negabinary(const int64_t x) {
    return (x + (uint64_t)0xaaaaaaaaaaaaaaaaull) ^
           (uint64_t)0xaaaaaaaaaaaaaaaaull;
  }

  MGARDX_EXEC
  static uint32_t binary2negabinary(const int32_t x) {
    return (x + (uint32_t)0xaaaaaaaau) ^ (uint32_t)0xaaaaaaaau;
  }

  MGARDX_EXEC
  static int64_t negabinary2binary(const uint64_t x) {
    return (x ^ 0xaaaaaaaaaaaaaaaaull) - 0xaaaaaaaaaaaaaaaaull;
  }

  MGARDX_EXEC
  static int32_t negabinary2binary(const uint32_t x) {
    return (x ^ 0xaaaaaaaau) - 0xaaaaaaaau;
  }
};

#define ALLOC_BLOCK                                                            \
  Byte *shared_memory = new Byte[task.GetSharedMemorySize()];                  \
  TaskType ***threads = new TaskType **[task.GetBlockDimZ()];                  \
  for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {           \
    threads[threadz] = new TaskType *[task.GetBlockDimY()];                    \
    for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {         \
      threads[threadz][thready] = new TaskType[task.GetBlockDimX()];           \
      for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) {       \
        TaskType &thread = threads[threadz][thready][threadx];                 \
        thread = task;                                                         \
        thread.GetFunctor().InitConfig(                                        \
            task.GetGridDimZ(), task.GetGridDimY(), task.GetGridDimX(),        \
            task.GetBlockDimZ(), task.GetBlockDimY(), task.GetBlockDimX());    \
        thread.GetFunctor().InitSharedMemory(shared_memory);                   \
        thread.GetFunctor().InitThreadId(threadz, thready, threadx);           \
      }                                                                        \
    }                                                                          \
  }

#define DEALLOC_BLOCK                                                          \
  for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {           \
    for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {         \
      delete[] threads[threadz][thready];                                      \
    }                                                                          \
    delete[] threads[threadz];                                                 \
  }                                                                            \
  delete[] threads;                                                            \
  delete[] shared_memory;

// #define INIT_BLOCK \
//   for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {\
//     for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {\
//       for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) {\
//         TaskType &thread = threads[threadz][thready][threadx];\
//         thread = task;\
//         threads[threadz][thready][threadx].GetFunctor().Init(\
//                  task.GetGridDimZ(), task.GetGridDimY(), task.GetGridDimX(),\
//                  task.GetBlockDimZ(), task.GetBlockDimY(), task.GetBlockDimX(),\
//                  blockz, blocky, blockx, threadz, thready, threadx,\
//                  shared_memory);\
//       }\
//     }\
//   }

#define INIT_BLOCK                                                             \
  for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {           \
    for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {         \
      for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) {       \
        threads[threadz][thready][threadx].GetFunctor().InitBlockId(           \
            blockz, blocky, blockx);                                           \
      }                                                                        \
    }                                                                          \
  }

#define COMPUTE_BLOCK(OPERATION)                                               \
  for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {           \
    for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {         \
      for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) {       \
        threads[threadz][thready][threadx].GetFunctor().OPERATION();           \
      }                                                                        \
    }                                                                          \
  }

template <typename TaskType> MGARDX_KERL void SerialKernel(TaskType task) {
  // Timer timer_op, timer_op1, timer_op2, timer_op3, timer_alloc, timer_init,
  // timer_dealloc; timer_op.clear();
  // timer_alloc.clear();timer_init.clear();timer_dealloc.clear();
  // timer_op1.clear(); timer_op2.clear(); timer_op3.clear();
  // timer_alloc.start();
  ALLOC_BLOCK;
  // timer_alloc.end();
  for (SIZE blockz = 0; blockz < task.GetGridDimZ(); blockz++) {
    for (SIZE blocky = 0; blocky < task.GetGridDimY(); blocky++) {
      for (SIZE blockx = 0; blockx < task.GetGridDimX(); blockx++) {
        // timer_init.start();
        INIT_BLOCK;
        // timer_init.end();
        // timer_op.start();
        // timer_op1.start();
        COMPUTE_BLOCK(Operation1);
        // timer_op1.end();
        // timer_op2.start();
        COMPUTE_BLOCK(Operation2);
        // timer_op2.end();
        // timer_op3.start();
        COMPUTE_BLOCK(Operation3);
        // timer_op3.end();
        COMPUTE_BLOCK(Operation4);
        COMPUTE_BLOCK(Operation5);
        COMPUTE_BLOCK(Operation6);
        COMPUTE_BLOCK(Operation7);
        COMPUTE_BLOCK(Operation8);
        COMPUTE_BLOCK(Operation9);
        COMPUTE_BLOCK(Operation10);
        // timer_op.end();
      }
    }
  }
  // timer_dealloc.start();
  DEALLOC_BLOCK;
  // timer_dealloc.end();

  // timer_op.print(task.GetFunctorName() + "_Op");
  // timer_op1.print(task.GetFunctorName() + "_Op1");
  // timer_op2.print(task.GetFunctorName() + "_Op2");
  // timer_op3.print(task.GetFunctorName() + "_Op3");
  // timer_alloc.print(task.GetFunctorName() + "_Alloc");
  // timer_init.print(task.GetFunctorName() + "_Init");
  // timer_dealloc.print(task.GetFunctorName() + "_Dealloc");
  // timer_op.clear();
  // timer_alloc.clear();timer_init.clear();timer_dealloc.clear();
}

#define ALLOC_BLOCK_CONDITION                                                  \
  Byte *shared_memory = new Byte[task.GetSharedMemorySize()];                  \
  TaskType ***threads = new TaskType **[task.GetBlockDimZ()];                  \
  bool ***active = new bool **[task.GetBlockDimZ()];                           \
  for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {           \
    threads[threadz] = new TaskType *[task.GetBlockDimY()];                    \
    active[threadz] = new bool *[task.GetBlockDimY()];                         \
    for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {         \
      threads[threadz][thready] = new TaskType[task.GetBlockDimX()];           \
      active[threadz][thready] = new bool[task.GetBlockDimX()];                \
      for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) {       \
        active[threadz][thready][threadx] = true;                              \
        TaskType &thread = threads[threadz][thready][threadx];                 \
        thread = task;                                                         \
        thread.GetFunctor().InitConfig(                                        \
            task.GetGridDimZ(), task.GetGridDimY(), task.GetGridDimX(),        \
            task.GetBlockDimZ(), task.GetBlockDimY(), task.GetBlockDimX());    \
        thread.GetFunctor().InitThreadId(threadz, thready, threadx);           \
        thread.GetFunctor().InitSharedMemory(shared_memory);                   \
      }                                                                        \
    }                                                                          \
  }

#define DEALLOC_BLOCK_CONDITION                                                \
  for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {           \
    for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {         \
      delete[] threads[threadz][thready];                                      \
      delete[] active[threadz][thready];                                       \
    }                                                                          \
    delete[] threads[threadz];                                                 \
    delete[] active[threadz];                                                  \
  }                                                                            \
  delete[] threads;                                                            \
  delete[] active;                                                             \
  delete[] shared_memory;

// #define INIT_BLOCK_CONDITION \
//   for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {\
//     for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {\
//       for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) {\
//         TaskType &thread = threads[threadz][thready][threadx];\
//         thread = task;\
//         thread.GetFunctor().InitConfig(task.GetGridDimZ(), task.GetGridDimY(), task.GetGridDimX(),\
//                                  task.GetBlockDimZ(), task.GetBlockDimY(), task.GetBlockDimX());\
//         thread.GetFunctor().InitSharedMemory(shared_memory);\
//         thread.GetFunctor().InitBlockId(blockz, blocky, blockx);\
//         thread.GetFunctor().InitThreadId(threadz, thready, threadx);\
//         active[threadz][thready][threadx] = true;\
//       }\
//     }\
//   }

#define INIT_BLOCK_CONDITION                                                   \
  for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {           \
    for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {         \
      for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) {       \
        TaskType &thread = threads[threadz][thready][threadx];                 \
        thread.GetFunctor().InitBlockId(blockz, blocky, blockx);               \
        active[threadz][thready][threadx] = true;                              \
      }                                                                        \
    }                                                                          \
  }

#define RESET_BLOCK_CONDITION                                                  \
  for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {           \
    for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {         \
      for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) {       \
        active[threadz][thready][threadx] = true;                              \
      }                                                                        \
    }                                                                          \
  }

#define EVALUATE_BLOCK_CONDITION(CONDITION_VAR, CONDITION_OP)                  \
  CONDITION_VAR = false;                                                       \
  for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {           \
    for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {         \
      for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) {       \
        TaskType &thread = threads[threadz][thready][threadx];                 \
        bool thread_active = thread.GetFunctor().CONDITION_OP();               \
        active[threadz][thready][threadx] = thread_active;                     \
        CONDITION_VAR = CONDITION_VAR | thread_active;                         \
      }                                                                        \
    }                                                                          \
  }

#define COMPUTE_BLOCK_CONDITION(OPERATION)                                     \
  for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {           \
    for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {         \
      for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) {       \
        TaskType &thread = threads[threadz][thready][threadx];                 \
        if (active[threadz][thready][threadx]) {                               \
          thread.GetFunctor().OPERATION();                                     \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

template <typename TaskType> MGARDX_KERL void SerialIterKernel(TaskType task) {
  ALLOC_BLOCK_CONDITION;
  bool condition1 = false;
  bool condition2 = false;
  for (SIZE blockz = 0; blockz < task.GetGridDimZ(); blockz++) {
    for (SIZE blocky = 0; blocky < task.GetGridDimY(); blocky++) {
      for (SIZE blockx = 0; blockx < task.GetGridDimX(); blockx++) {
        INIT_BLOCK_CONDITION;
        COMPUTE_BLOCK_CONDITION(Operation1);
        COMPUTE_BLOCK_CONDITION(Operation2);
        EVALUATE_BLOCK_CONDITION(condition1, LoopCondition1);
        while (condition1) {
          COMPUTE_BLOCK_CONDITION(Operation3);
          COMPUTE_BLOCK_CONDITION(Operation4);
          COMPUTE_BLOCK_CONDITION(Operation5);
          COMPUTE_BLOCK_CONDITION(Operation6);
          EVALUATE_BLOCK_CONDITION(condition1, LoopCondition1);
        }
        RESET_BLOCK_CONDITION;
        COMPUTE_BLOCK_CONDITION(Operation7);
        COMPUTE_BLOCK_CONDITION(Operation8);
        COMPUTE_BLOCK_CONDITION(Operation9);
        COMPUTE_BLOCK_CONDITION(Operation10);
        EVALUATE_BLOCK_CONDITION(condition2, LoopCondition2);
        while (condition2) {
          COMPUTE_BLOCK_CONDITION(Operation11);
          COMPUTE_BLOCK_CONDITION(Operation12);
          COMPUTE_BLOCK_CONDITION(Operation13);
          COMPUTE_BLOCK_CONDITION(Operation14);
          EVALUATE_BLOCK_CONDITION(condition2, LoopCondition2);
        }
        RESET_BLOCK_CONDITION;
        COMPUTE_BLOCK_CONDITION(Operation15);
        COMPUTE_BLOCK_CONDITION(Operation16);
        COMPUTE_BLOCK_CONDITION(Operation17);
      }
    }
  }
  DEALLOC_BLOCK_CONDITION;
}

#define ALLOC_GRID                                                             \
  TaskType ******threads = new TaskType *****[task.GetGridDimZ()];             \
  Byte ****shared_memory = new Byte ***[task.GetGridDimZ()];                   \
  for (SIZE blockz = 0; blockz < task.GetGridDimZ(); blockz++) {               \
    threads[blockz] = new TaskType ****[task.GetGridDimY()];                   \
    shared_memory[blockz] = new Byte **[task.GetGridDimY()];                   \
    for (SIZE blocky = 0; blocky < task.GetGridDimY(); blocky++) {             \
      threads[blockz][blocky] = new TaskType ***[task.GetGridDimX()];          \
      shared_memory[blockz][blocky] = new Byte *[task.GetGridDimX()];          \
      for (SIZE blockx = 0; blockx < task.GetGridDimX(); blockx++) {           \
        threads[blockz][blocky][blockx] =                                      \
            new TaskType **[task.GetBlockDimZ()];                              \
        shared_memory[blockz][blocky][blockx] =                                \
            new Byte[task.GetSharedMemorySize()];                              \
        for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {     \
          threads[blockz][blocky][blockx][threadz] =                           \
              new TaskType *[task.GetBlockDimY()];                             \
          for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {   \
            threads[blockz][blocky][blockx][threadz][thready] =                \
                new TaskType[task.GetBlockDimX()];                             \
            for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) { \
              threads[blockz][blocky][blockx][threadz][thready][threadx] =     \
                  task;                                                        \
              threads[blockz][blocky][blockx][threadz][thready][threadx]       \
                  .GetFunctor()                                                \
                  .Init(task.GetGridDimZ(), task.GetGridDimY(),                \
                        task.GetGridDimX(), task.GetBlockDimZ(),               \
                        task.GetBlockDimY(), task.GetBlockDimX(), blockz,      \
                        blocky, blockx, threadz, thready, threadx,             \
                        shared_memory[blockz][blocky][blockx]);                \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

#define ALLOC_ACTIVE_GRID(ACTIVE_VAR)                                          \
  bool ******ACTIVE_VAR = new bool *****[task.GetGridDimZ()];                  \
  for (SIZE blockz = 0; blockz < task.GetGridDimZ(); blockz++) {               \
    ACTIVE_VAR[blockz] = new bool ****[task.GetGridDimY()];                    \
    for (SIZE blocky = 0; blocky < task.GetGridDimY(); blocky++) {             \
      ACTIVE_VAR[blockz][blocky] = new bool ***[task.GetGridDimX()];           \
      for (SIZE blockx = 0; blockx < task.GetGridDimX(); blockx++) {           \
        ACTIVE_VAR[blockz][blocky][blockx] = new bool **[task.GetBlockDimZ()]; \
        for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {     \
          ACTIVE_VAR[blockz][blocky][blockx][threadz] =                        \
              new bool *[task.GetBlockDimY()];                                 \
          for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {   \
            ACTIVE_VAR[blockz][blocky][blockx][threadz][thready] =             \
                new bool[task.GetBlockDimX()];                                 \
            for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) { \
              ACTIVE_VAR[blockz][blocky][blockx][threadz][thready][threadx] =  \
                  true;                                                        \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

#define DEALLOC_GRID                                                           \
  for (SIZE blockz = 0; blockz < task.GetGridDimZ(); blockz++) {               \
    for (SIZE blocky = 0; blocky < task.GetGridDimY(); blocky++) {             \
      for (SIZE blockx = 0; blockx < task.GetGridDimX(); blockx++) {           \
        for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {     \
          for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {   \
            for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) { \
            }                                                                  \
            delete[] threads[blockz][blocky][blockx][threadz][thready];        \
          }                                                                    \
          delete[] threads[blockz][blocky][blockx][threadz];                   \
        }                                                                      \
        delete[] threads[blockz][blocky][blockx];                              \
        delete[] shared_memory[blockz][blocky][blockx];                        \
      }                                                                        \
      delete[] threads[blockz][blocky];                                        \
      delete[] shared_memory[blockz][blocky];                                  \
    }                                                                          \
    delete[] threads[blockz];                                                  \
    delete[] shared_memory[blockz];                                            \
  }                                                                            \
  delete[] threads;                                                            \
  delete[] shared_memory;

#define DEALLOC_ACTIVE_GRID(ACTIVE_VAR)                                        \
  for (SIZE blockz = 0; blockz < task.GetGridDimZ(); blockz++) {               \
    for (SIZE blocky = 0; blocky < task.GetGridDimY(); blocky++) {             \
      for (SIZE blockx = 0; blockx < task.GetGridDimX(); blockx++) {           \
        for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {     \
          for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {   \
            for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) { \
            }                                                                  \
            delete[] ACTIVE_VAR[blockz][blocky][blockx][threadz][thready];     \
          }                                                                    \
          delete[] ACTIVE_VAR[blockz][blocky][blockx][threadz];                \
        }                                                                      \
        delete[] ACTIVE_VAR[blockz][blocky][blockx];                           \
      }                                                                        \
      delete[] ACTIVE_VAR[blockz][blocky];                                     \
    }                                                                          \
    delete[] ACTIVE_VAR[blockz];                                               \
  }                                                                            \
  delete[] ACTIVE_VAR;

#define COMPUTE_GRID(OPERATION, ACTIVE_VAR)                                    \
  for (SIZE blockz = 0; blockz < task.GetGridDimZ(); blockz++) {               \
    for (SIZE blocky = 0; blocky < task.GetGridDimY(); blocky++) {             \
      for (SIZE blockx = 0; blockx < task.GetGridDimX(); blockx++) {           \
        for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {     \
          for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {   \
            for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) { \
              TaskType &thread =                                               \
                  threads[blockz][blocky][blockx][threadz][thready][threadx];  \
              if (ACTIVE_VAR[blockz][blocky][blockx][threadz][thready]         \
                            [threadx]) {                                       \
                thread.GetFunctor().OPERATION();                               \
              }                                                                \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

#define RESER_CONDITION_GRID(ACTIVE_VAR)                                       \
  for (SIZE blockz = 0; blockz < task.GetGridDimZ(); blockz++) {               \
    for (SIZE blocky = 0; blocky < task.GetGridDimY(); blocky++) {             \
      for (SIZE blockx = 0; blockx < task.GetGridDimX(); blockx++) {           \
        for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {     \
          for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {   \
            for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) { \
              ACTIVE_VAR[blockz][blocky][blockx][threadz][thready][threadx] =  \
                  true;                                                        \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

#define INHERENT_CONDITION_GRID(ACTIVE_VAR1, ACTIVE_VAR2)                      \
  for (SIZE blockz = 0; blockz < task.GetGridDimZ(); blockz++) {               \
    for (SIZE blocky = 0; blocky < task.GetGridDimY(); blocky++) {             \
      for (SIZE blockx = 0; blockx < task.GetGridDimX(); blockx++) {           \
        for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {     \
          for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {   \
            for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) { \
              ACTIVE_VAR2[blockz][blocky][blockx][threadz][thready][threadx] = \
                  ACTIVE_VAR1[blockz][blocky][blockx][threadz][thready]        \
                             [threadx];                                        \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

#define EVALUATE_CONDITION_GRID(CONDITION_VAR, ACTIVE_VAR, CONDITION_OP)       \
  CONDITION_VAR = false;                                                       \
  for (SIZE blockz = 0; blockz < task.GetGridDimZ(); blockz++) {               \
    for (SIZE blocky = 0; blocky < task.GetGridDimY(); blocky++) {             \
      for (SIZE blockx = 0; blockx < task.GetGridDimX(); blockx++) {           \
        for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {     \
          for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {   \
            for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) { \
              TaskType &thread =                                               \
                  threads[blockz][blocky][blockx][threadz][thready][threadx];  \
              bool thread_active = thread.GetFunctor().CONDITION_OP();         \
              ACTIVE_VAR[blockz][blocky][blockx][threadz][thready][threadx] =  \
                  thread_active;                                               \
              CONDITION_VAR = CONDITION_VAR | thread_active;                   \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

template <typename TaskType>
MGARDX_KERL void SerialHuffmanCLCustomizedKernel(TaskType task) {

  bool loop_condition1 = false;
  bool loop_condition2 = false;
  bool branch_condition1 = false;

  ALLOC_GRID;
  ALLOC_ACTIVE_GRID(all_active);
  ALLOC_ACTIVE_GRID(loop1_active);
  ALLOC_ACTIVE_GRID(loop2_active);
  ALLOC_ACTIVE_GRID(branch1_active);

  COMPUTE_GRID(Operation1, all_active);
  INHERENT_CONDITION_GRID(all_active, loop1_active);
  EVALUATE_CONDITION_GRID(loop_condition1, loop1_active, LoopCondition1);
  while (loop_condition1) {
    COMPUTE_GRID(Operation2, loop1_active);
    COMPUTE_GRID(Operation3, loop1_active);
    COMPUTE_GRID(Operation4, loop1_active);
    COMPUTE_GRID(Operation5, loop1_active);
    INHERENT_CONDITION_GRID(loop1_active, branch1_active);
    EVALUATE_CONDITION_GRID(branch_condition1, branch1_active,
                            BranchCondition1);
    if (branch_condition1) {
      INHERENT_CONDITION_GRID(branch1_active, loop2_active);
      EVALUATE_CONDITION_GRID(loop_condition2, loop2_active, LoopCondition2);
      while (loop_condition2) {
        COMPUTE_GRID(Operation6, loop2_active);
        COMPUTE_GRID(Operation7, loop2_active);
        COMPUTE_GRID(Operation8, loop2_active);
        EVALUATE_CONDITION_GRID(loop_condition2, loop2_active, LoopCondition2);
      }
      COMPUTE_GRID(Operation9, branch1_active)
      COMPUTE_GRID(Operation10, branch1_active);
    }
    COMPUTE_GRID(Operation11, loop1_active);
    COMPUTE_GRID(Operation12, loop1_active);
    COMPUTE_GRID(Operation13, loop1_active);
    COMPUTE_GRID(Operation14, loop1_active);
    EVALUATE_CONDITION_GRID(loop_condition1, loop1_active, LoopCondition1);
  }

  DEALLOC_GRID;
  DEALLOC_ACTIVE_GRID(all_active);
  DEALLOC_ACTIVE_GRID(loop1_active);
  DEALLOC_ACTIVE_GRID(loop2_active);
  DEALLOC_ACTIVE_GRID(branch1_active);
}

template <typename TaskType>
MGARDX_KERL void SerialHuffmanCWCustomizedKernel(TaskType task) {

  bool loop_condition1 = false;

  ALLOC_GRID;
  ALLOC_ACTIVE_GRID(all_active);
  ALLOC_ACTIVE_GRID(loop1_active);
  COMPUTE_GRID(Operation1, all_active);
  COMPUTE_GRID(Operation2, all_active);
  COMPUTE_GRID(Operation3, all_active);
  INHERENT_CONDITION_GRID(all_active, loop1_active);
  EVALUATE_CONDITION_GRID(loop_condition1, loop1_active, LoopCondition1);
  while (loop_condition1) {
    COMPUTE_GRID(Operation4, loop1_active);
    COMPUTE_GRID(Operation5, loop1_active);
    COMPUTE_GRID(Operation6, loop1_active);
    COMPUTE_GRID(Operation7, loop1_active);
    COMPUTE_GRID(Operation8, loop1_active);
    EVALUATE_CONDITION_GRID(loop_condition1, loop1_active, LoopCondition1);
  }
  COMPUTE_GRID(Operation9, all_active);
  COMPUTE_GRID(Operation10, all_active);

  DEALLOC_GRID;
  DEALLOC_ACTIVE_GRID(all_active);
  DEALLOC_ACTIVE_GRID(loop1_active);
}

template <> class DeviceSpecification<SERIAL> {
public:
  MGARDX_CONT
  DeviceSpecification() {
    NumDevices = 1;
    MaxSharedMemorySize = new int[NumDevices];
    WarpSize = new int[NumDevices];
    NumSMs = new int[NumDevices];
    ArchitectureGeneration = new int[NumDevices];
    MaxNumThreadsPerSM = new int[NumDevices];
    MaxNumThreadsPerTB = new int[NumDevices];
    AvailableMemory = new size_t[NumDevices];
    SupportCooperativeGroups = new bool[NumDevices];
    DeviceNames = new std::string[NumDevices];

    for (int d = 0; d < NumDevices; d++) {
      MaxSharedMemorySize[d] = 1e6;
      WarpSize[d] = 32;
      NumSMs[d] = 80;
      MaxNumThreadsPerSM[d] = 1024;
      MaxNumThreadsPerTB[d] = 1024;
      ArchitectureGeneration[d] = 1;
      SupportCooperativeGroups[d] = true;
      DeviceNames[d] = "CPU";
    }
  }

  MGARDX_CONT int GetNumDevices() { return NumDevices; }

  MGARDX_CONT int GetMaxSharedMemorySize(int dev_id) {
    return MaxSharedMemorySize[dev_id];
  }

  MGARDX_CONT int GetWarpSize(int dev_id) { return WarpSize[dev_id]; }

  MGARDX_CONT int GetNumSMs(int dev_id) { return NumSMs[dev_id]; }

  MGARDX_CONT int GetArchitectureGeneration(int dev_id) {
    return ArchitectureGeneration[dev_id];
  }

  MGARDX_CONT int GetMaxNumThreadsPerSM(int dev_id) {
    return MaxNumThreadsPerSM[dev_id];
  }

  MGARDX_CONT int GetMaxNumThreadsPerTB(int dev_id) {
    return MaxNumThreadsPerTB[dev_id];
  }

  MGARDX_CONT int GetAvailableMemory(int dev_id) {
    AvailableMemory[dev_id] = 8e9;
    return AvailableMemory[dev_id];
  }

  MGARDX_CONT bool SupportCG(int dev_id) {
    return SupportCooperativeGroups[dev_id];
  }

  MGARDX_CONT std::string GetDeviceName(int dev_id) {
    return DeviceNames[dev_id];
  }

  MGARDX_CONT
  ~DeviceSpecification() {
    delete[] MaxSharedMemorySize;
    delete[] WarpSize;
    delete[] NumSMs;
    delete[] ArchitectureGeneration;
    delete[] MaxNumThreadsPerSM;
    delete[] MaxNumThreadsPerTB;
    delete[] AvailableMemory;
    delete[] SupportCooperativeGroups;
    delete[] DeviceNames;
  }

  int NumDevices;
  int *MaxSharedMemorySize;
  int *WarpSize;
  int *NumSMs;
  int *ArchitectureGeneration;
  int *MaxNumThreadsPerSM;
  int *MaxNumThreadsPerTB;
  size_t *AvailableMemory;
  bool *SupportCooperativeGroups;
  std::string *DeviceNames;
};

template <> class DeviceQueues<SERIAL> {
public:
  MGARDX_CONT
  DeviceQueues() {
    // do nothing
  }

  MGARDX_CONT int GetQueue(int dev_id, SIZE queue_id) { return 0; }

  MGARDX_CONT void SyncQueue(int dev_id, SIZE queue_id) {
    // do nothing
  }

  MGARDX_CONT void SyncAllQueues(int dev_id) {
    // do nothing
  }

  MGARDX_CONT
  ~DeviceQueues() {
    // do nothing
  }
};

template <> class DeviceRuntime<SERIAL> {
public:
  MGARDX_CONT
  DeviceRuntime() {}

  MGARDX_CONT static int GetDeviceCount() { return DeviceSpecs.NumDevices; }

  MGARDX_CONT static void SelectDevice(SIZE dev_id) {
    // do nothing
  }

  MGARDX_CONT static int GetQueue(SIZE queue_id) { return 0; }

  MGARDX_CONT static void SyncQueue(SIZE queue_id) {
    // do nothing
  }

  MGARDX_CONT static void SyncAllQueues() {
    // do nothing
  }

  MGARDX_CONT static void SyncDevice() {
    // do nothing
  }

  MGARDX_CONT static std::string GetDeviceName() {
    return DeviceSpecs.GetDeviceName(curr_dev_id);
  }

  MGARDX_CONT static int GetMaxSharedMemorySize() {
    return DeviceSpecs.GetMaxSharedMemorySize(curr_dev_id);
  }

  MGARDX_CONT static int GetWarpSize() {
    return DeviceSpecs.GetWarpSize(curr_dev_id);
  }

  MGARDX_CONT static int GetNumSMs() {
    return DeviceSpecs.GetNumSMs(curr_dev_id);
  }

  MGARDX_CONT static int GetArchitectureGeneration() {
    return DeviceSpecs.GetArchitectureGeneration(curr_dev_id);
  }

  MGARDX_CONT static int GetMaxNumThreadsPerSM() {
    return DeviceSpecs.GetMaxNumThreadsPerSM(curr_dev_id);
  }

  MGARDX_CONT static int GetMaxNumThreadsPerTB() {
    return DeviceSpecs.GetMaxNumThreadsPerTB(curr_dev_id);
  }

  MGARDX_CONT static size_t GetAvailableMemory() {
    return DeviceSpecs.GetAvailableMemory(curr_dev_id);
  }

  MGARDX_CONT static bool SupportCG() {
    return DeviceSpecs.SupportCG(curr_dev_id);
  }

  template <typename FunctorType>
  MGARDX_CONT static int
  GetOccupancyMaxActiveBlocksPerSM(FunctorType functor, int blockSize,
                                   size_t dynamicSMemSize) {
    return 32;
  }

  template <typename FunctorType>
  MGARDX_CONT static void SetMaxDynamicSharedMemorySize(FunctorType functor,
                                                        int maxbytes) {
    // do nothing
  }

  MGARDX_CONT
  ~DeviceRuntime() {}

  static int curr_dev_id;
  static DeviceQueues<SERIAL> queues;
  static bool SyncAllKernelsAndCheckErrors;
  static DeviceSpecification<SERIAL> DeviceSpecs;
  static bool TimingAllKernels;
  static bool PrintKernelConfig;
};

template <> class MemoryManager<SERIAL> {
public:
  MGARDX_CONT
  MemoryManager(){};

  template <typename T>
  MGARDX_CONT static void Malloc1D(T *&ptr, SIZE n, int queue_idx) {
    using converted_T =
        typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    ptr = (T *)std::malloc(n * sizeof(converted_T));
    if (ptr == NULL) {
      std::cout << log::log_err << "MemoryManager<SERIAL>::Malloc1D error.\n";
    }
  }

  template <typename T>
  MGARDX_CONT static void MallocND(T *&ptr, SIZE n1, SIZE n2, SIZE &ld,
                                   int queue_idx) {
    using converted_T =
        typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    ptr = (T *)std::malloc(n1 * n2 * sizeof(converted_T));
    ld = n1;
    if (ptr == NULL) {
      std::cout << log::log_err << "MemoryManager<SERIAL>::Malloc1D error.\n";
    }
  }

  template <typename T>
  MGARDX_CONT static void MallocManaged1D(T *&ptr, SIZE n, int queue_idx) {
    using converted_T =
        typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    ptr = (T *)std::malloc(n * sizeof(converted_T));
    if (ptr == NULL) {
      std::cout << log::log_err
                << "MemoryManager<SERIAL>::MallocManaged1D error.\n";
    }
  }

  template <typename T> MGARDX_CONT static void Free(T *ptr) {
    if (ptr == NULL)
      return;
    std::free(ptr);
  }

  template <typename T>
  MGARDX_CONT static void Copy1D(T *dst_ptr, const T *src_ptr, SIZE n,
                                 int queue_idx) {
    using converted_T =
        typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    std::memcpy(dst_ptr, src_ptr, sizeof(converted_T) * n);
  }

  template <typename T>
  MGARDX_CONT static void CopyND(T *dst_ptr, SIZE dst_ld, const T *src_ptr,
                                 SIZE src_ld, SIZE n1, SIZE n2, int queue_idx) {
    using converted_T =
        typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    std::memcpy(dst_ptr, src_ptr, sizeof(converted_T) * n1 * n2);
  }

  template <typename T>
  MGARDX_CONT static void MallocHost(T *&ptr, SIZE n, int queue_idx) {
    using converted_T =
        typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    ptr = (T *)std::malloc(n * sizeof(converted_T));
    if (ptr == NULL) {
      std::cout << log::log_err << "MemoryManager<SERIAL>::Malloc1D error.\n";
    }
  }

  template <typename T> MGARDX_CONT static void FreeHost(T *ptr) {
    if (ptr == NULL)
      return;
    std::free(ptr);
  }

  template <typename T>
  MGARDX_CONT static void Memset1D(T *ptr, SIZE n, int value, int queue_idx) {
    using converted_T =
        typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    memset(ptr, value, n * sizeof(converted_T));
  }

  template <typename T>
  MGARDX_CONT static void MemsetND(T *ptr, SIZE ld, SIZE n1, SIZE n2, int value,
                                   int queue_idx) {
    using converted_T =
        typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    memset(ptr, value, n1 * n2 * sizeof(converted_T));
  }

  template <typename T> MGARDX_CONT static bool IsDevicePointer(T *ptr) {
    return true;
  }
  static bool ReduceMemoryFootprint;
};

#define ALIGN_LEFT 0  // for encoding
#define ALIGN_RIGHT 1 // for decoding

#define Sign_Encoding_Atomic 0
#define Sign_Encoding_Reduce 1
#define Sign_Encoding_Ballot 2

#define Sign_Decoding_Parallel 0

#define BINARY 0
#define NEGABINARY 1

#define Bit_Transpose_Serial_All 0
#define Bit_Transpose_Parallel_B_Serial_b 1
#define Bit_Transpose_Parallel_B_Atomic_b 2
#define Bit_Transpose_Parallel_B_Reduce_b 3
#define Bit_Transpose_Parallel_B_Ballot_b 4
#define Bit_Transpose_TCU 5

#define Warp_Bit_Transpose_Serial_All 0
#define Warp_Bit_Transpose_Parallel_B_Serial_b 1
#define Warp_Bit_Transpose_Serial_B_Atomic_b 2
#define Warp_Bit_Transpose_Serial_B_Reduce_b 3
#define Warp_Bit_Transpose_Serial_B_Ballot_b 4
#define Warp_Bit_Transpose_TCU 5

#define Error_Collecting_Disable -1
#define Error_Collecting_Serial_All 0
#define Error_Collecting_Parallel_Bitplanes_Serial_Error 1
#define Error_Collecting_Parallel_Bitplanes_Atomic_Error 2
#define Error_Collecting_Parallel_Bitplanes_Reduce_Error 3

#define Warp_Error_Collecting_Disable -1
#define Warp_Error_Collecting_Serial_All 0
#define Warp_Error_Collecting_Parallel_Bitplanes_Serial_Error 1
#define Warp_Error_Collecting_Serial_Bitplanes_Atomic_Error 2
#define Warp_Error_Collecting_Serial_Bitplanes_Reduce_Error 3

typedef unsigned long long int uint64_cu;

template <typename T_org, typename T_trans, SIZE nblockx, SIZE nblocky,
          SIZE nblockz, OPTION ALIGN, OPTION METHOD>
struct BlockBitTranspose<T_org, T_trans, nblockx, nblocky, nblockz, ALIGN,
                         METHOD, SERIAL> {

  MGARDX_EXEC
  static void Serial_All(T_org *v, T_trans *tv, SIZE b, SIZE B, SIZE IdX,
                         SIZE IdY) {
    if (IdX == 0 && IdY == 0) {
      // printf("add-in: %llu %u\n", v, v[0]);
      // for (int i = 0; i < b; i++) {
      //   printf("v: %u\n", v[i]);
      // }
      for (SIZE B_idx = 0; B_idx < B; B_idx++) {
        T_trans buffer = 0;
        for (SIZE b_idx = 0; b_idx < b; b_idx++) {
          T_trans bit = (v[b_idx] >> (sizeof(T_org) * 8 - 1 - B_idx)) & 1u;
          if (ALIGN == ALIGN_LEFT) {
            buffer += bit << (sizeof(T_trans) * 8 - 1 - b_idx);
            // if (B_idx == 0) {
            // printf("%u %u %u\n", B_idx, b_idx, bit);
            // print_bits(buffer, sizeof(T_trans)*8, false);
            // printf("\n");
            // }
          } else if (ALIGN == ALIGN_RIGHT) {
            buffer += bit << (b - 1 - b_idx);
            // if (b_idx == 0) printf("%u %u %u\n", B_idx, b_idx, bit);
          } else {
          }
          // if (j == 0 ) {printf("i %u j %u shift %u bit %u\n", i,j,b-1-j,
          // bit); }
        }

        // printf("buffer: %u\n", buffer);

        tv[B_idx] = buffer;
      }
    }
  }

  MGARDX_EXEC
  static void Transpose(T_org *v, T_trans *tv, SIZE b, SIZE B, SIZE IdX,
                        SIZE IdY) {
    Serial_All(v, tv, b, B, IdX, IdY);
  }
};

template <typename T, typename T_fp, typename T_sfp, typename T_error,
          SIZE nblockx, SIZE nblocky, SIZE nblockz, OPTION METHOD,
          OPTION BinaryType>
struct BlockErrorCollect<T, T_fp, T_sfp, T_error, nblockx, nblocky, nblockz,
                         METHOD, BinaryType, SERIAL> {

  MGARDX_EXEC
  static void Serial_All(T *v, T_error *temp, T_error *errors, SIZE num_elems,
                         SIZE num_bitplanes, SIZE IdX, SIZE IdY) {
    if (IdX == 0 && IdY == 0) {
      for (SIZE elem_idx = 0; elem_idx < num_elems; elem_idx++) {
        T data = v[elem_idx];
        T_fp fp_data = (T_fp)fabs(data);
        T_sfp fps_data = (T_sfp)data;
        T_fp ngb_data = Math<SERIAL>::binary2negabinary(fps_data);
        T_error mantissa;
        if (BinaryType == BINARY) {
          mantissa = fabs(data) - fp_data;
        } else if (BinaryType == NEGABINARY) {
          mantissa = data - fps_data;
        }
        for (SIZE bitplane_idx = 0; bitplane_idx < num_bitplanes;
             bitplane_idx++) {
          uint64_t mask = (1 << bitplane_idx) - 1;
          T_error diff = 0;
          if (BinaryType == BINARY) {
            diff = (T_error)(fp_data & mask) + mantissa;
          } else if (BinaryType == NEGABINARY) {
            diff = (T_error)Math<SERIAL>::negabinary2binary(ngb_data & mask) +
                   mantissa;
          }
          errors[num_bitplanes - bitplane_idx] += diff * diff;
          // if (blockIdx.x == 0 && num_bitplanes-bitplane_idx == 2) {
          //   printf("elem error[%u]: %f\n", elem_idx, diff * diff);
          // }
        }
        errors[0] += data * data;
      }
    }
  }

  MGARDX_EXEC
  static void Collect(T *v, T_error *temp, T_error *errors, SIZE num_elems,
                      SIZE num_bitplanes, SIZE IdX, SIZE IdY) {
    Serial_All(v, temp, errors, num_elems, num_bitplanes, IdX, IdY);
  }
};

template <typename TaskType> class DeviceAdapter<TaskType, SERIAL> {
public:
  MGARDX_CONT
  DeviceAdapter(){};

  MGARDX_CONT
  int IsResourceEnough(TaskType &task) {
    if (task.GetBlockDimX() * task.GetBlockDimY() * task.GetBlockDimZ() >
        DeviceRuntime<SERIAL>::GetMaxNumThreadsPerTB()) {
      return THREADBLOCK_TOO_LARGE;
    }
    if (task.GetSharedMemorySize() >
        DeviceRuntime<SERIAL>::GetMaxSharedMemorySize()) {
      return SHARED_MEMORY_TOO_LARGE;
    }
    return RESOURCE_ENOUGH;
  }

  MGARDX_CONT
  ExecutionReturn Execute(TaskType &task) {

    if (DeviceRuntime<SERIAL>::PrintKernelConfig) {
      std::cout << log::log_info << task.GetFunctorName() << ": <"
                << task.GetBlockDimX() << ", " << task.GetBlockDimY() << ", "
                << task.GetBlockDimZ() << "> <" << task.GetGridDimX() << ", "
                << task.GetGridDimY() << ", " << task.GetGridDimZ() << ">\n";
    }

    ExecutionReturn ret;
    if (IsResourceEnough(task) != RESOURCE_ENOUGH) {
      if (DeviceRuntime<SERIAL>::PrintKernelConfig) {
        if (IsResourceEnough(task) == THREADBLOCK_TOO_LARGE) {
          std::cout << log::log_info << "threadblock too large.\n";
        }
        if (IsResourceEnough(task) == SHARED_MEMORY_TOO_LARGE) {
          std::cout << log::log_info << "shared memory too large.\n";
        }
      }
      ret.success = false;
      ret.execution_time = std::numeric_limits<double>::max();
      return ret;
    }

    Timer timer;
    if (DeviceRuntime<SERIAL>::TimingAllKernels ||
        AutoTuner<SERIAL>::ProfileKernels) {
      DeviceRuntime<SERIAL>::SyncDevice();
      timer.start();
    }
    // if constexpr evalute at compile time otherwise this does not compile
    if constexpr (std::is_base_of<Functor<SERIAL>,
                                  typename TaskType::Functor>::value) {
      SerialKernel(task);
    } else if constexpr (std::is_base_of<IterFunctor<SERIAL>,
                                         typename TaskType::Functor>::value) {
      SerialIterKernel(task);
    } else if constexpr (std::is_base_of<HuffmanCLCustomizedFunctor<SERIAL>,
                                         typename TaskType::Functor>::value) {
      SerialHuffmanCLCustomizedKernel(task);
    } else if constexpr (std::is_base_of<HuffmanCWCustomizedFunctor<SERIAL>,
                                         typename TaskType::Functor>::value) {
      SerialHuffmanCWCustomizedKernel(task);
    }
    // timer.end();
    // timer.print(task.GetFunctorName());
    // timer.clear();
    if (DeviceRuntime<SERIAL>::TimingAllKernels ||
        AutoTuner<SERIAL>::ProfileKernels) {
      DeviceRuntime<SERIAL>::SyncDevice();
      timer.end();
      if (DeviceRuntime<SERIAL>::TimingAllKernels) {
        timer.print(task.GetFunctorName());
      }
      if (AutoTuner<SERIAL>::ProfileKernels) {
        ret.success = true;
        ret.execution_time = timer.get();
      }
    }
    return ret;
  }
};

template <> class DeviceCollective<SERIAL> {
public:
  MGARDX_CONT
  DeviceCollective(){};

  template <typename T>
  MGARDX_CONT static void Sum(SIZE n, SubArray<1, T, SERIAL> &v,
                              SubArray<1, T, SERIAL> &result, int queue_idx) {
    *result((IDX)0) = std::accumulate(v((IDX)0), v((IDX)n), 0);
  }

  template <typename T>
  MGARDX_CONT static void AbsMax(SIZE n, SubArray<1, T, SERIAL> &v,
                                 SubArray<1, T, SERIAL> &result,
                                 int queue_idx) {
    T max_result = 0;
    for (SIZE i = 0; i < n; ++i) {
      max_result = std::max((T)fabs(*v(i)), max_result);
    }
    *result((IDX)0) = max_result;
  }

  template <typename T>
  MGARDX_CONT static void SquareSum(SIZE n, SubArray<1, T, SERIAL> &v,
                                    SubArray<1, T, SERIAL> &result,
                                    int queue_idx) {
    T sum_result = 0;
    for (SIZE i = 0; i < n; ++i) {
      T tmp = *v(i);
      sum_result += tmp * tmp;
    }
    *result((IDX)0) = sum_result;
  }

  template <typename T>
  MGARDX_CONT static void ScanSumInclusive(SIZE n, SubArray<1, T, SERIAL> &v,
                                           SubArray<1, T, SERIAL> &result,
                                           int queue_idx) {
    // std::inclusive_scan(v(0), v(n), result(0));
    std::cout << log::log_err << "ScanSumInclusive<SERIAL> not implemented.\n";
  }

  template <typename T>
  MGARDX_CONT static void ScanSumExclusive(SIZE n, SubArray<1, T, SERIAL> &v,
                                           SubArray<1, T, SERIAL> &result,
                                           int queue_idx) {
    // std::exclusive_scan(v(0), v(n), result(0));
    std::cout << log::log_err << "ScanSumExclusive<SERIAL> not implemented.\n";
  }

  template <typename T>
  MGARDX_CONT static void ScanSumExtended(SIZE n, SubArray<1, T, SERIAL> &v,
                                          SubArray<1, T, SERIAL> &result,
                                          int queue_idx) {
    // std::inclusive_scan(v(0), v(n), result(1));
    // result(0) = 0;
    std::cout << log::log_err << "ScanSumExtended<SERIAL> not implemented.\n";
  }

  template <typename KeyT, typename ValueT>
  MGARDX_CONT static void SortByKey(SIZE n, SubArray<1, KeyT, SERIAL> &keys,
                                    SubArray<1, ValueT, SERIAL> &values,
                                    int queue_idx) {
    std::vector<std::pair<KeyT, ValueT>> data(n);
    for (SIZE i = 0; i < n; ++i) {
      data[i] = std::pair<KeyT, ValueT>(*keys(i), *values(i));
    }
    std::stable_sort(data.begin(), data.end(),
                     KeyValueComparator<KeyT, ValueT>{});
    for (SIZE i = 0; i < n; ++i) {
      *keys(i) = data[i].first;
      *values(i) = data[i].second;
    }
  }
};

} // namespace mgard_x

#endif