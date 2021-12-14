#ifndef MGARD_X_DEVICE_ADAPTER_SERIAL_H
#define MGARD_X_DEVICE_ADAPTER_SERIAL_H

#include "DeviceAdapter.h"
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <utility>
#include <algorithm>
#include <climits>
#include <cmath>

namespace mgard_x {

template<>
struct SyncBlock<Serial> {
  MGARDX_EXEC static void Sync() {
    // do nothing
  }
};

template<>
struct SyncGrid<Serial> {
  MGARDX_EXEC static void Sync() {
    // do nothing
  }
};

template <> 
struct Atomic<Serial> {
  template <typename T>
  MGARDX_EXEC static
  T Min(T * result, T value) {
    T old = *result;
    *result = std::min(*result, value);
    return old;
  }
  template <typename T>
  MGARDX_EXEC static
  T Max(T * result, T value) {
    T old = *result;
    *result = std::max(*result, value);
    return old;
  }
  template <typename T>
  MGARDX_EXEC static
  T Add(T * result, T value) {
    T old = *result;
    * result += value;
    return old;
  }
};

template <> 
struct Math<Serial> {
  template <typename T>
  MGARDX_EXEC static
  T Min(T a, T b) {
    return std::min(a, b);
  }
  template <typename T>
  MGARDX_EXEC static
  T Max(T a, T b) {
    return std::max(a, b);
  }

  MGARDX_EXEC static
  int ffs(unsigned int a) {
    return ffs(a);
  }
  MGARDX_EXEC static
  int ffsll(long long unsigned int a) {
    // return ffsll(a);
    unsigned pos = 0;
    if (a == 0) return pos;
    while (!(a & 1))
    {
      a >>= 1;
      ++pos;
    }
    return pos + 1;
  }
};


#define ALLOC_BLOCK \
  Byte *shared_memory = new Byte[task.GetSharedMemorySize()];\
  TaskType *** threads = new TaskType**[task.GetBlockDimZ()];\
  for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {\
    threads[threadz] = new TaskType*[task.GetBlockDimY()];\
    for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {\
      threads[threadz][thready] = new TaskType[task.GetBlockDimX()];\
      for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) {\
        TaskType &thread = threads[threadz][thready][threadx];\
        thread = task;\
        thread.GetFunctor().InitConfig(task.GetGridDimZ(), task.GetGridDimY(), task.GetGridDimX(),\
                                 task.GetBlockDimZ(), task.GetBlockDimY(), task.GetBlockDimX());\
        thread.GetFunctor().InitSharedMemory(shared_memory);\
        thread.GetFunctor().InitThreadId(threadz, thready, threadx);\
      }\
    }\
  }

#define DEALLOC_BLOCK \
  for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {\
    for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {\
      delete [] threads[threadz][thready];\
    }\
    delete [] threads[threadz];\
  }\
  delete [] threads;\
  delete [] shared_memory;

#define INIT_BLOCK \
  for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {\
    for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {\
      for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) {\
        TaskType &thread = threads[threadz][thready][threadx];\
        thread = task;\
        threads[threadz][thready][threadx].GetFunctor().Init(\
                 task.GetGridDimZ(), task.GetGridDimY(), task.GetGridDimX(),\
                 task.GetBlockDimZ(), task.GetBlockDimY(), task.GetBlockDimX(),\
                 blockz, blocky, blockx, threadz, thready, threadx,\
                 shared_memory);\
      }\
    }\
  }

#define COMPUTE_BLOCK(OPERATION) \
  for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {\
    for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {\
      for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) {\
        TaskType &thread = threads[threadz][thready][threadx];\
        thread.GetFunctor().OPERATION();\
      }\
    }\
  }

template <typename TaskType>
MGARDX_KERL void SerialKernel(TaskType task) {
  // Timer timer_op, timer_alloc, timer_init, timer_dealloc;
  // timer_op.clear(); timer_alloc.clear();timer_init.clear();timer_dealloc.clear();
    
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
        COMPUTE_BLOCK(Operation1);
        COMPUTE_BLOCK(Operation2);
        COMPUTE_BLOCK(Operation3);
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
  // timer_alloc.print(task.GetFunctorName() + "_Alloc");
  // timer_init.print(task.GetFunctorName() + "_Init");
  // timer_dealloc.print(task.GetFunctorName() + "_Dealloc");
  // timer_op.clear(); timer_alloc.clear();timer_init.clear();timer_dealloc.clear();
}


#define ALLOC_BLOCK_CONDITION \
  Byte *shared_memory = new Byte[task.GetSharedMemorySize()];\
  TaskType *** threads = new TaskType**[task.GetBlockDimZ()];\
  bool *** active = new bool**[task.GetBlockDimZ()];\
  for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {\
    threads[threadz] = new TaskType*[task.GetBlockDimY()];\
    active[threadz] = new bool*[task.GetBlockDimY()];\
    for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {\
      threads[threadz][thready] = new TaskType[task.GetBlockDimX()];\
      active[threadz][thready] = new bool[task.GetBlockDimX()];\
      for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) {\
        active[threadz][thready][threadx] = true;\
        TaskType &thread = threads[threadz][thready][threadx];\
        thread = task;\
        thread.GetFunctor().InitConfig(task.GetGridDimZ(), task.GetGridDimY(), task.GetGridDimX(),\
                                 task.GetBlockDimZ(), task.GetBlockDimY(), task.GetBlockDimX());\
        thread.GetFunctor().InitSharedMemory(shared_memory);\
      }\
    }\
  }

#define DEALLOC_BLOCK_CONDITION \
  for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {\
    for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {\
      delete [] threads[threadz][thready];\
      delete [] active[threadz][thready];\
    }\
    delete [] threads[threadz];\
    delete [] active[threadz];\
  }\
  delete [] threads;\
  delete [] active;\
  delete [] shared_memory;

#define INIT_BLOCK_CONDITION \
  for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {\
    for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {\
      for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) {\
        TaskType &thread = threads[threadz][thready][threadx];\
        thread = task;\
        thread.GetFunctor().InitConfig(task.GetGridDimZ(), task.GetGridDimY(), task.GetGridDimX(),\
                                 task.GetBlockDimZ(), task.GetBlockDimY(), task.GetBlockDimX());\
        thread.GetFunctor().InitSharedMemory(shared_memory);\
        thread.GetFunctor().InitBlockId(blockz, blocky, blockx);\
        thread.GetFunctor().InitThreadId(threadz, thready, threadx);\
        active[threadz][thready][threadx] = true;\
      }\
    }\
  }

#define RESET_BLOCK_CONDITION \
  for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {\
    for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {\
      for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) {\
        active[threadz][thready][threadx] = true;\
      } \
    }\
  }

#define EVALUATE_BLOCK_CONDITION(CONDITION_VAR, CONDITION_OP) \
  CONDITION_VAR = false;\
  for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {\
    for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {\
      for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) {\
        TaskType &thread = threads[threadz][thready][threadx];\
        bool thread_active = thread.GetFunctor().CONDITION_OP();\
        active[threadz][thready][threadx] = thread_active;\
        CONDITION_VAR = CONDITION_VAR | thread_active;\
      } \
    }\
  }

#define COMPUTE_BLOCK_CONDITION(OPERATION) \
  for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {\
    for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {\
      for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) {\
        TaskType &thread = threads[threadz][thready][threadx];\
        if (active[threadz][thready][threadx]) {\
          thread.GetFunctor().OPERATION();\
        }\
      }\
    }\
  }


template <typename TaskType>
MGARDX_KERL void SerialIterKernel(TaskType task) {
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


#define ALLOC_GRID \
  TaskType ****** threads = new TaskType*****[task.GetGridDimZ()];\
  Byte **** shared_memory = new Byte***[task.GetGridDimZ()];\
  for (SIZE blockz = 0; blockz < task.GetGridDimZ(); blockz++) {\
    threads[blockz] = new TaskType****[task.GetGridDimY()];\
    shared_memory[blockz] = new Byte**[task.GetGridDimY()];\
    for (SIZE blocky = 0; blocky < task.GetGridDimY(); blocky++) {\
      threads[blockz][blocky] = new TaskType***[task.GetGridDimX()];\
      shared_memory[blockz][blocky] = new Byte*[task.GetGridDimX()];\
      for (SIZE blockx = 0; blockx < task.GetGridDimX(); blockx++) {\
        threads[blockz][blocky][blockx] = new TaskType**[task.GetBlockDimZ()];\
        shared_memory[blockz][blocky][blockx] = new Byte[task.GetSharedMemorySize()];\
        for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {\
          threads[blockz][blocky][blockx][threadz] = new TaskType*[task.GetBlockDimY()];\
          for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {\
            threads[blockz][blocky][blockx][threadz][thready] = new TaskType[task.GetBlockDimX()];\
            for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) {\
              threads[blockz][blocky][blockx][threadz][thready][threadx] = task;\
              threads[blockz][blocky][blockx][threadz][thready][threadx].GetFunctor().Init(\
                                 task.GetGridDimZ(), task.GetGridDimY(), task.GetGridDimX(),\
                                 task.GetBlockDimZ(), task.GetBlockDimY(), task.GetBlockDimX(),\
                                 blockz, blocky, blockx, threadz, thready, threadx,\
                                 shared_memory[blockz][blocky][blockx]);\
            }\
          }\
        }\
      }\
    }\
  }

#define ALLOC_ACTIVE_GRID(ACTIVE_VAR) \
  bool ****** ACTIVE_VAR = new bool*****[task.GetGridDimZ()];\
  for (SIZE blockz = 0; blockz < task.GetGridDimZ(); blockz++) {\
    ACTIVE_VAR[blockz] = new bool****[task.GetGridDimY()];\
    for (SIZE blocky = 0; blocky < task.GetGridDimY(); blocky++) {\
      ACTIVE_VAR[blockz][blocky] = new bool***[task.GetGridDimX()];\
      for (SIZE blockx = 0; blockx < task.GetGridDimX(); blockx++) {\
        ACTIVE_VAR[blockz][blocky][blockx] = new bool**[task.GetBlockDimZ()];\
        for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {\
          ACTIVE_VAR[blockz][blocky][blockx][threadz] = new bool*[task.GetBlockDimY()];\
          for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {\
            ACTIVE_VAR[blockz][blocky][blockx][threadz][thready] = new bool[task.GetBlockDimX()];\
            for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) {\
              ACTIVE_VAR[blockz][blocky][blockx][threadz][thready][threadx] = true;\
            }\
          }\
        }\
      }\
    }\
  }


#define DEALLOC_GRID \
  for (SIZE blockz = 0; blockz < task.GetGridDimZ(); blockz++) {\
    for (SIZE blocky = 0; blocky < task.GetGridDimY(); blocky++) {\
      for (SIZE blockx = 0; blockx < task.GetGridDimX(); blockx++) {\
        for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {\
          for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {\
            for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) {\
            }\
            delete [] threads[blockz][blocky][blockx][threadz][thready];\
          }\
          delete [] threads[blockz][blocky][blockx][threadz];\
        }\
        delete [] threads[blockz][blocky][blockx];\
        delete [] shared_memory[blockz][blocky][blockx];\
      }\
      delete [] threads[blockz][blocky];\
      delete [] shared_memory[blockz][blocky];\
    }\
    delete [] threads[blockz];\
    delete [] shared_memory[blockz];\
  }\
  delete [] threads;\
  delete [] shared_memory;


#define DEALLOC_ACTIVE_GRID(ACTIVE_VAR) \
  for (SIZE blockz = 0; blockz < task.GetGridDimZ(); blockz++) {\
    for (SIZE blocky = 0; blocky < task.GetGridDimY(); blocky++) {\
      for (SIZE blockx = 0; blockx < task.GetGridDimX(); blockx++) {\
        for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {\
          for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {\
            for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) {\
            }\
            delete [] ACTIVE_VAR[blockz][blocky][blockx][threadz][thready];\
          }\
          delete [] ACTIVE_VAR[blockz][blocky][blockx][threadz];\
        }\
        delete [] ACTIVE_VAR[blockz][blocky][blockx];\
      }\
      delete [] ACTIVE_VAR[blockz][blocky];\
    }\
    delete [] ACTIVE_VAR[blockz];\
  }\
  delete [] ACTIVE_VAR;


#define COMPUTE_GRID(OPERATION, ACTIVE_VAR) \
  for (SIZE blockz = 0; blockz < task.GetGridDimZ(); blockz++) {\
    for (SIZE blocky = 0; blocky < task.GetGridDimY(); blocky++) {\
      for (SIZE blockx = 0; blockx < task.GetGridDimX(); blockx++) {\
        for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {\
          for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {\
            for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) {\
              TaskType &thread = threads[blockz][blocky][blockx][threadz][thready][threadx];\
              if (ACTIVE_VAR[blockz][blocky][blockx][threadz][thready][threadx]) {\
                thread.GetFunctor().OPERATION();\
              }\
            }\
          }\
        }\
      }\
    }\
  }

#define RESER_CONDITION_GRID(ACTIVE_VAR) \
  for (SIZE blockz = 0; blockz < task.GetGridDimZ(); blockz++) {\
    for (SIZE blocky = 0; blocky < task.GetGridDimY(); blocky++) {\
      for (SIZE blockx = 0; blockx < task.GetGridDimX(); blockx++) {\
        for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {\
          for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {\
            for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) {\
              ACTIVE_VAR[blockz][blocky][blockx][threadz][thready][threadx] = true;\
            }\
          }\
        }\
      }\
    }\
  }


#define INHERENT_CONDITION_GRID(ACTIVE_VAR1, ACTIVE_VAR2) \
  for (SIZE blockz = 0; blockz < task.GetGridDimZ(); blockz++) {\
    for (SIZE blocky = 0; blocky < task.GetGridDimY(); blocky++) {\
      for (SIZE blockx = 0; blockx < task.GetGridDimX(); blockx++) {\
        for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {\
          for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {\
            for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) {\
              ACTIVE_VAR2[blockz][blocky][blockx][threadz][thready][threadx] = \
              ACTIVE_VAR1[blockz][blocky][blockx][threadz][thready][threadx];\
            }\
          }\
        }\
      }\
    }\
  }


#define EVALUATE_CONDITION_GRID(CONDITION_VAR, ACTIVE_VAR, CONDITION_OP) \
  CONDITION_VAR = false; \
  for (SIZE blockz = 0; blockz < task.GetGridDimZ(); blockz++) {\
    for (SIZE blocky = 0; blocky < task.GetGridDimY(); blocky++) {\
      for (SIZE blockx = 0; blockx < task.GetGridDimX(); blockx++) {\
        for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {\
          for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {\
            for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) {\
              TaskType &thread = threads[blockz][blocky][blockx][threadz][thready][threadx];\
              bool thread_active = thread.GetFunctor().CONDITION_OP();\
              ACTIVE_VAR[blockz][blocky][blockx][threadz][thready][threadx] = thread_active;\
              CONDITION_VAR = CONDITION_VAR | thread_active;\
            }\
          }\
        }\
      }\
    }\
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
    EVALUATE_CONDITION_GRID(branch_condition1, branch1_active, BranchCondition1);
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




template <>
class DeviceSpecification<Serial> {
  public:
  MGARDX_CONT
  DeviceSpecification(){
    NumDevices = 1;
    MaxSharedMemorySize = new int[NumDevices];
    WarpSize = new int[NumDevices];
    NumSMs = new int[NumDevices];
    ArchitectureGeneration = new int[NumDevices];
    MaxNumThreadsPerSM = new int[NumDevices];

    for (int d = 0; d < NumDevices; d++) {
      MaxSharedMemorySize[d] = 1e6;
      WarpSize[d] = 32;
      NumSMs[d] = 80;
      MaxNumThreadsPerSM[d] = 1024;
      ArchitectureGeneration[d] = 1;
    }
  }

  MGARDX_CONT int
  GetNumDevices() {
    return NumDevices;
  }

  MGARDX_CONT int
  GetMaxSharedMemorySize(int dev_id) {
    return MaxSharedMemorySize[dev_id];
  }

  MGARDX_CONT int
  GetWarpSize(int dev_id) {
    return WarpSize[dev_id];
  }

  MGARDX_CONT int
  GetNumSMs(int dev_id) {
    return NumSMs[dev_id];
  }

  MGARDX_CONT int
  GetArchitectureGeneration(int dev_id) {
    return ArchitectureGeneration[dev_id];
  }

  MGARDX_CONT int
  GetMaxNumThreadsPerSM(int dev_id) {
    return MaxNumThreadsPerSM[dev_id];
  }

  MGARDX_CONT
  ~DeviceSpecification() {
    delete [] MaxSharedMemorySize;
    delete [] WarpSize;
    delete [] NumSMs;
    delete [] ArchitectureGeneration;
  }

  int NumDevices;
  int* MaxSharedMemorySize;
  int* WarpSize;
  int* NumSMs;
  int* ArchitectureGeneration;
  int* MaxNumThreadsPerSM;
};

template <>
class DeviceQueues<Serial> {
  public:
  MGARDX_CONT
  DeviceQueues(){
    // do nothing
  }

  MGARDX_CONT int 
  GetQueue(int dev_id, SIZE queue_id){
    return 0;
  }

  MGARDX_CONT void 
  SyncQueue(int dev_id, SIZE queue_id){
    // do nothing
  }

  MGARDX_CONT void 
  SyncAllQueues(int dev_id){
    // do nothing
  }

  MGARDX_CONT
  ~DeviceQueues(){
    // do nothing
  }
};

template <>
class DeviceRuntime<Serial> {
  public:
  MGARDX_CONT
  DeviceRuntime(){}

  MGARDX_CONT static void 
  SelectDevice(SIZE dev_id){
    // do nothing
  }

  MGARDX_CONT static int 
  GetQueue(SIZE queue_id){
    return 0;
  }

  MGARDX_CONT static void 
  SyncQueue(SIZE queue_id){
    // do nothing
  }

  MGARDX_CONT static void 
  SyncAllQueues(){
    // do nothing
  }

  MGARDX_CONT static void
  SyncDevice(){
    // do nothing
  }

  MGARDX_CONT static int
  GetMaxSharedMemorySize() {
    return DeviceSpecs.GetMaxSharedMemorySize(curr_dev_id);
  }

  MGARDX_CONT static int
  GetWarpSize() {
    return DeviceSpecs.GetWarpSize(curr_dev_id);
  }

  MGARDX_CONT static int
  GetNumSMs() {
    return DeviceSpecs.GetNumSMs(curr_dev_id);
  }

  MGARDX_CONT static int
  GetArchitectureGeneration() {
    return DeviceSpecs.GetArchitectureGeneration(curr_dev_id);
  }

  MGARDX_CONT static int
  GetMaxNumThreadsPerSM() {
    return DeviceSpecs.GetMaxNumThreadsPerSM(curr_dev_id);
  }

  template <typename FunctorType>
  MGARDX_CONT static int
  GetOccupancyMaxActiveBlocksPerSM(FunctorType functor, int blockSize, size_t dynamicSMemSize) {
    return 32;
  }

  template <typename FunctorType>
  MGARDX_CONT static void
  SetMaxDynamicSharedMemorySize(FunctorType functor, int maxbytes) {
    // do nothing
  }


  MGARDX_CONT
  ~DeviceRuntime(){
  }

  static int curr_dev_id;
  static DeviceQueues<Serial> queues;
  static bool SyncAllKernelsAndCheckErrors;
  static DeviceSpecification<Serial> DeviceSpecs;
  static bool TimingAllKernels;
  static bool PrintKernelConfig;
};

template <>
class MemoryManager<Serial> {
  public:
  MGARDX_CONT
  MemoryManager(){};

  template <typename T>
  MGARDX_CONT static
  void Malloc1D(T *& ptr, SIZE n, int queue_idx) {
    using converted_T = typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    ptr = (T*)std::malloc(n * sizeof(converted_T));
    if (ptr == NULL) {
      std::cout << log::log_err << "MemoryManager<Serial>::Malloc1D error.\n";
    }
  }

  template <typename T>
  MGARDX_CONT static
  void MallocND(T *& ptr, SIZE n1, SIZE n2, SIZE &ld, int queue_idx) {
    using converted_T = typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    ptr = (T*)std::malloc(n1 * n2 * sizeof(converted_T));
    ld = n1;
    if (ptr == NULL) {
      std::cout << log::log_err << "MemoryManager<Serial>::Malloc1D error.\n";
    }
  }

  template <typename T>
  MGARDX_CONT static
  void Free(T * ptr) {
    if (ptr == NULL) return;
    std::free(ptr);
  }

  template <typename T>
  MGARDX_CONT static
  void Copy1D(T * dst_ptr, const T * src_ptr, SIZE n, int queue_idx) {
    using converted_T = typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    std::memcpy(dst_ptr, src_ptr, sizeof(converted_T) * n);
  }

  template <typename T>
  MGARDX_CONT static
  void CopyND(T * dst_ptr, SIZE dst_ld, const T * src_ptr, SIZE src_ld, SIZE n1, SIZE n2, int queue_idx) {
    using converted_T = typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    std::memcpy(dst_ptr, src_ptr, sizeof(converted_T) * n1 * n2);
  }

  template <typename T>
  MGARDX_CONT static
  void MallocHost(T *& ptr, SIZE n, int queue_idx) {
    using converted_T = typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    ptr = (T*)std::malloc(n * sizeof(converted_T));
    if (ptr == NULL) {
      std::cout << log::log_err << "MemoryManager<Serial>::Malloc1D error.\n";
    }
  }

  template <typename T>
  MGARDX_CONT static
  void FreeHost(T * ptr) {
    if (ptr == NULL) return;
    std::free(ptr);
  }

  template <typename T>
  MGARDX_CONT static
  void Memset1D(T * ptr, SIZE n, int value) {
    using converted_T = typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    memset(ptr, value, n * sizeof(converted_T));
  }

  template <typename T>
  MGARDX_CONT static
  void MemsetND(T * ptr, SIZE ld, SIZE n1, SIZE n2, int value) {
    using converted_T = typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    memset(ptr, value, n1 * n2 * sizeof(converted_T));
  }

  template <typename T>
  MGARDX_CONT static
  bool IsDevicePointer(T * ptr) {
    return true;
  }
  static bool ReduceMemoryFootprint;
};


template <typename TaskTypeType>
class DeviceAdapter<TaskTypeType, Serial> {
public:
  MGARDX_CONT
  DeviceAdapter(){};

  MGARDX_CONT
  ExecutionReturn Execute(TaskTypeType& task) {

    if (DeviceRuntime<Serial>::PrintKernelConfig) {
      std::cout << log::log_info << task.GetFunctorName() << ": <" <<
                task.GetBlockDimX() << ", " <<
                task.GetBlockDimY() << ", " <<
                task.GetBlockDimZ() << "> <" <<
                task.GetGridDimX() << ", " <<
                task.GetGridDimY() << ", " <<
                task.GetGridDimZ() << ">\n";
    }


    Timer timer;
    if (DeviceRuntime<Serial>::TimingAllKernels || AutoTuner<Serial>::ProfileKernels) {
      DeviceRuntime<Serial>::SyncDevice();
      timer.start();
    }
    // if constexpr evalute at compile time otherwise this does not compile
    if constexpr (std::is_base_of<Functor<Serial>, typename TaskTypeType::Functor>::value) {
      SerialKernel(task);
    } else if constexpr (std::is_base_of<IterFunctor<Serial>, typename TaskTypeType::Functor>::value) {
      SerialIterKernel(task);
    } else if constexpr (std::is_base_of<HuffmanCLCustomizedFunctor<Serial>, typename TaskTypeType::Functor>::value) {
      SerialHuffmanCLCustomizedKernel(task);
    } else if constexpr (std::is_base_of<HuffmanCWCustomizedFunctor<Serial>, typename TaskTypeType::Functor>::value) {
      SerialHuffmanCWCustomizedKernel(task);
    }
    // timer.end();
    // timer.print(task.GetFunctorName());
    // timer.clear();
    ExecutionReturn ret;
    if (DeviceRuntime<Serial>::TimingAllKernels || AutoTuner<Serial>::ProfileKernels) {
      DeviceRuntime<Serial>::SyncDevice();
      timer.end();
      if (DeviceRuntime<Serial>::TimingAllKernels) {
        timer.print(task.GetFunctorName());
      }
      if (AutoTuner<Serial>::ProfileKernels) {
        ret.execution_time = timer.get();
      }
    }
    return ret;
  }
};

template <typename KeyT, typename ValueT>
struct KeyValueComparator{
    bool operator()(std::pair<KeyT, ValueT> a, std::pair<KeyT, ValueT> b) const { return a.first < b.first; }
};

template <>
class DeviceCollective<Serial>{
public:
  MGARDX_CONT
  DeviceCollective(){};

  template <typename T> MGARDX_CONT static
  void Sum(SIZE n, SubArray<1, T, Serial>& v, SubArray<1, T, Serial>& result, int queue_idx) {
    *result((IDX)0) = std::accumulate(v(0), v(n), 0);
  }

  template <typename T> MGARDX_CONT static
  void AbsMax(SIZE n, SubArray<1, T, Serial>& v, SubArray<1, T, Serial>& result, int queue_idx) {
    T max_result = 0;
    for (SIZE i = 0; i < n; ++i) {
      max_result = std::max((T)fabs(*v(i)), max_result);
    }
    *result((IDX)0) = max_result;
  }

  template <typename T> MGARDX_CONT static
  void SquareSum(SIZE n, SubArray<1, T, Serial>& v, SubArray<1, T, Serial>& result, int queue_idx) {
    T sum_result = 0;
    for (SIZE i = 0; i < n; ++i) {
      T tmp = *v(i);
      sum_result += tmp * tmp; 
    }
    *result((IDX)0) = sum_result;
  }
 
 
 template <typename T> MGARDX_CONT static
  void ScanSumInclusive(SIZE n, SubArray<1, T, Serial>& v, SubArray<1, T, Serial>& result, int queue_idx) {
    // std::inclusive_scan(v(0), v(n), result(0));
    std::cout << log::log_err << "ScanSumInclusive<Serial> not implemented.\n";
  }

  template <typename T> MGARDX_CONT static
  void ScanSumExclusive(SIZE n, SubArray<1, T, Serial>& v, SubArray<1, T, Serial>& result, int queue_idx) {
    // std::exclusive_scan(v(0), v(n), result(0));
    std::cout << log::log_err << "ScanSumExclusive<Serial> not implemented.\n";
  }

  template <typename T> MGARDX_CONT static
  void ScanSumExtended(SIZE n, SubArray<1, T, Serial>& v, SubArray<1, T, Serial>& result, int queue_idx) {
    // std::inclusive_scan(v(0), v(n), result(1));
    // result(0) = 0;
    std::cout << log::log_err << "ScanSumExtended<Serial> not implemented.\n";
  }

  template <typename KeyT, typename ValueT> MGARDX_CONT static
  void SortByKey(SIZE n, SubArray<1, KeyT, Serial>& keys, SubArray<1, ValueT, Serial>& values, 
                 int queue_idx) {
    std::vector<std::pair<KeyT, ValueT>> data(n);
    for (SIZE i = 0; i < n; ++i) {
      data[i] = std::pair<KeyT, ValueT>(*keys(i), *values(i));
    }
    std::stable_sort(data.begin(), data.end(), KeyValueComparator<KeyT, ValueT>{});
    for (SIZE i = 0; i < n; ++i) {
      *keys(i) = data[i].first;
      *values(i) = data[i].second;
    }
  }
};


}

#endif