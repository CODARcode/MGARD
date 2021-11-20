#ifndef MGARD_CUDA_DEVICE_ADAPTER_SERIAL_H
#define MGARD_CUDA_DEVICE_ADAPTER_SERIAL_H

namespace mgard_cuda {

template<>
struct SyncBlock<Serial> {
  MGARDm_EXEC static void Sync() {
    // do nothing
  }
};

template<>
struct SyncGrid<Serial> {
  MGARDm_EXEC static void Sync() {
    // do nothing
  }
};

template <> 
struct Atomic<Serial> {
  template <typename T>
  MGARDm_EXEC static
  void Min(T * result, T value) {
    * result = value;
  }
};

#define COMPUTE_BLOCK(OPERATION) \
  for (SIZE threadz = 0; threadz < blockDim_z; threadz++) { \
    for (SIZE thready = 0; thready < blockDim_y; thread++) { \
      for (SIZE threadx = 0; threadx < blockDim_x;threadx++) { \
        task.get_functor().Init_thread_id(threadz, thready, threadx); \
        task.get_functor().OPERATION(); \
      } \
    } \
  } 

#define COMPUTE_CONDITION(CONDITION_VAR, CONDITION_OP) \
  task.get_functor().Init_thread_id(0, 0, 0); \
  CONDITION_VAR = task.get_functor().CONDITION_OP(); \
  for (SIZE threadz = 0; threadz < blockDim_z; threadz++) { \
    for (SIZE thready = 0; thready < blockDim_y; thread++) { \
      for (SIZE threadx = 0; threadx < blockDim_x;threadx++) { \
        task.get_functor().Init_thread_id(threadz, thready, threadx); \
        if(CONDITION_VAR != task.get_functor().CONDITION_OP()) { \
          std::cout << log::log_err << "IterKernel<Serial> inconsistant condition."; \
          exit(-1); \
        } \
      } \
    } \
  }

#define COMPUTE_BLOCK(OPERATION) \
  for (SIZE gridz = 0; gridz < gridDim_z; gridz++) { \
    for (SIZE gridy = 0; gridy < gridDim_y; gridy++) { \
      for (SIZE gridx = 0; gridx < gridDim_x; gridx++) { \
        task.get_functor().Init_block_id(blockz, blocky, blockx); \
        for (SIZE blockz = 0; blockz < blockDim_z; gridz++) { \
          for (SIZE blocky = 0; blocky < blockDim_y; gridy++) { \
            for (SIZE blockx = 0; blockx < blockDim_x; gridx++) { \
              task.get_functor().Init_thread_id(threadz, thready, threadx); \
              task.get_functor().OPERATION(); \
            } \
          } \
        } \
      } \
    } \
  }



template <typename Task>
MGARDm_KERL void Kernel(Task task, 
                        SIZE gridDim_z, SIZE gridDim_y, SIZE gridDim_x,
                        SIZE blockDim_z, SIZE blockDim_y, SIZE blockDim_x,
                        size_t shared_memory_size) {
  Byte * shared_memory = new Byte[shared_memory_size];
  task.get_functor().Init_shared_memory(shared_memory);
  task.get_functor().Init_config(gridDim_z, gridDim_y, gridDim_x, 
                                 blockDim_z, blockDim_y, blockDim_x);

  for (SIZE blockz = 0; blockz < gridDim_z; blockz++) {
    for (SIZE blocky = 0; blocky < gridDim_y; blocky++) {
      for (SIZE blockx = 0; blockx < gridDim_x; blockx++) {
        task.get_functor().Init_block_id(blockz, blocky, blockx);
        COMPUTE_BLOCK(Operation1);
        SyncBlock<Serial>::Sync();
        COMPUTE_BLOCK(Operation2);
        SyncBlock<Serial>::Sync();
        COMPUTE_BLOCK(Operation3);
        SyncBlock<Serial>::Sync();
        COMPUTE_BLOCK(Operation4);
        SyncBlock<Serial>::Sync();
        COMPUTE_BLOCK(Operation5);
        SyncBlock<Serial>::Sync();
      } //blockx
    } //blocky
  } //blockz

  delete[] shared_memory;
}

template <typename Task>
MGARDm_KERL void IterKernel(Task task,
                            SIZE gridDim_z, SIZE gridDim_y, SIZE gridDim_x,
                            SIZE blockDim_z, SIZE blockDim_y, SIZE blockDim_x,
                            size_t shared_memory_size) {

  Byte * shared_memory = new Byte[shared_memory_size];
  bool condition1, condition2;

  task.get_functor().Init_shared_memory(shared_memory);
  task.get_functor().Init_config(gridDim_z, gridDim_y, gridDim_x, 
                                 blockDim_z, blockDim_y, blockDim_x);



  bool * loop_condition = new bool[blockDim_z * blockDim_y * blockDim_x];

  for (SIZE blockz = 0; blockz < gridDim_z; blockz++) {
    for (SIZE blocky = 0; blocky < gridDim_y; blocky++) {
      for (SIZE blockx = 0; blockx < gridDim_x; blockx++) {
        task.get_functor().Init_block_id(blockz, blocky, blockx);

        COMPUTE_BLOCK(Operation1);
        SyncBlock<Serial>::Sync();

        COMPUTE_CONDITION(condition1, LoopCondition1);
        while (condition1) {
          COMPUTE_BLOCK(Operation2);
          SyncBlock<Serial>::Sync();
          COMPUTE_BLOCK(Operation3);
          SyncBlock<Serial>::Sync();
          COMPUTE_BLOCK(Operation4);
          SyncBlock<Serial>::Sync();
          COMPUTE_BLOCK(Operation5);
          SyncBlock<Serial>::Sync();
          COMPUTE_CONDITION(condition1, LoopCondition1);
          SyncBlock<Serial>::Sync();
        } // loop1

        COMPUTE_BLOCK(Operation6);
        SyncBlock<Serial>::Sync();
        COMPUTE_BLOCK(Operation7);
        SyncBlock<Serial>::Sync();
        COMPUTE_BLOCK(Operation8);
        SyncBlock<Serial>::Sync();
        COMPUTE_BLOCK(Operation9);
        SyncBlock<Serial>::Sync();

        COMPUTE_CONDITION(condition2, LoopCondition2); 
        while (condition2) {
          COMPUTE_BLOCK(Operation10);
          SyncBlock<Serial>::Sync();
          COMPUTE_BLOCK(Operation11);
          SyncBlock<Serial>::Sync();
          COMPUTE_BLOCK(Operation12);
          SyncBlock<Serial>::Sync();
          COMPUTE_BLOCK(Operation13);
          SyncBlock<Serial>::Sync();
          COMPUTE_CONDITION(condition2, LoopCondition2);
          SyncBlock<Serial>::Sync();
        }

        COMPUTE_BLOCK(Operation14);
        SyncBlock<Serial>::Sync();
        COMPUTE_BLOCK(Operation15);
        SyncBlock<Serial>::Sync();
        COMPUTE_BLOCK(Operation16);
        SyncBlock<Serial>::Sync();
      }
    }
  }

  delete [] shared_memory;
}

template <typename Task>
MGARDm_KERL void HuffmanCLCustomizedKernel(Task task,
                                           SIZE gridDim_z, SIZE gridDim_y, SIZE gridDim_x,
                                           SIZE blockDim_z, SIZE blockDim_y, SIZE blockDim_x,
                                           size_t shared_memory_size) {
  Byte * shared_memory = new Byte[shared_memory_size];

  task.get_functor().Init_shared_memory(shared_memory);
  task.get_functor().Init_config(gridDim_z, gridDim_y, gridDim_x, 
                                 blockDim_z, blockDim_y, blockDim_x);


  for (SIZE gridz = 0; gridz < gridDim_z; gridz++) {
    for (SIZE gridy = 0; gridy < gridDim_y; gridy++) {
      for (SIZE gridx = 0; gridx < gridDim_x; gridx++) {
        task.get_functor().Init_block_id(blockz, blocky, blockx);
        for (SIZE blockz = 0; blockz < blockDim_z; gridz++) {
          for (SIZE blocky = 0; blocky < blockDim_y; gridy++) {
            for (SIZE blockx = 0; blockx < blockDim_x; gridx++) {
              task.get_functor().Init_thread_id(threadz, thready, threadx);
              task.get_functor().Operation1();
            }
          }
        }
      }
    }
  }




              task.get_functor().Init(gridDim_z, gridDim_y, gridDim_x, 
                                      blockDim_z, blockDim_y, blockDim_x,
                                      gridz, gridy, gridx, 
                                      blockz, blocky, blockx,
                                      shared_memory);

              task.get_functor().Operation1();
              SyncGrid<Serial>::Sync();
              while (task.get_functor().LoopCondition1()) {
                task.get_functor().Operation2();
                SyncGrid<Serial>::Sync();
                task.get_functor().Operation3();
                SyncGrid<Serial>::Sync();
                task.get_functor().Operation4();
                SyncGrid<Serial>::Sync();
                task.get_functor().Operation5();
                SyncBlock<Serial>::Sync();
                if (task.get_functor().BranchCondition1()) {
                  while (task.get_functor().LoopCondition2()) {
                    task.get_functor().Operation6();
                    SyncBlock<Serial>::Sync();
                    task.get_functor().Operation7();
                    SyncBlock<Serial>::Sync();
                    task.get_functor().Operation8();
                    SyncBlock<Serial>::Sync();
                  }
                  task.get_functor().Operation9();
                  SyncGrid<Serial>::Sync();
                  task.get_functor().Operation10();
                  SyncGrid<Serial>::Sync();
                }
                task.get_functor().Operation11();
                SyncGrid<Serial>::Sync();
                task.get_functor().Operation12();
                SyncGrid<Serial>::Sync();
                task.get_functor().Operation13();
                SyncGrid<Serial>::Sync();
                task.get_functor().Operation14();
                SyncGrid<Serial>::Sync();
              }
            }
          }
        }
      }
    }
  }
  delete [] shared_memory;
}


template <typename Task>
MGARDm_KERL void HuffmanCWCustomizedKernel(Task task) {
  Byte * shared_memory = new Byte[shared_memory_size];

  for (SIZE gridz = 0; gridz < gridDim_z; gridz++) {
    for (SIZE gridy = 0; gridy < gridDim_y; gridy++) {
      for (SIZE gridx = 0; gridx < gridDim_x; gridx++) {
        for (SIZE blockz = 0; blockz < blockDim_z; gridz++) {
          for (SIZE blocky = 0; blocky < blockDim_y; gridy++) {
            for (SIZE blockx = 0; blockx < blockDim_x; gridx++) {

              task.get_functor().Init(gridDim_z, gridDim_y, gridDim_x, 
                                      blockDim_z, blockDim_y, blockDim_x,
                                      gridz, gridy, gridx, 
                                      blockz, blocky, blockx,
                                      shared_memory);

              task.get_functor().Operation1();
              SyncGrid<Serial>::Sync();
              task.get_functor().Operation2();
              SyncGrid<Serial>::Sync();
              task.get_functor().Operation3();
              SyncGrid<Serial>::Sync();

              while (task.get_functor().LoopCondition1()) {
                task.get_functor().Operation4();
                SyncGrid<Serial>::Sync();
                task.get_functor().Operation5();
                SyncGrid<Serial>::Sync();
                task.get_functor().Operation6();
                SyncGrid<Serial>::Sync();
                task.get_functor().Operation7();
                SyncGrid<Serial>::Sync();
                task.get_functor().Operation8();
                SyncGrid<Serial>::Sync();
              }
              task.get_functor().Operation9();
              SyncGrid<Serial>::Sync();
              task.get_functor().Operation10();
              SyncGrid<Serial>::Sync();
            }
          }
        }
      }
    }
  }
  delete [] shared_memory;
}


template <>
class DeviceSpecification<Serial> {
  public:
  MGARDm_CONT
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
      NumSMs[d] = 1;
      MaxNumThreadsPerSM[d] = 1024;
      ArchitectureGeneration[d] = 1;
    }
  }

  MGARDm_CONT int
  GetNumDevices() {
    return NumDevices;
  }

  MGARDm_CONT int
  GetMaxSharedMemorySize(int dev_id) {
    return MaxSharedMemorySize[dev_id];
  }

  MGARDm_CONT int
  GetWarpSize(int dev_id) {
    return WarpSize[dev_id];
  }

  MGARDm_CONT int
  GetNumSMs(int dev_id) {
    return NumSMs[dev_id];
  }

  MGARDm_CONT int
  GetArchitectureGeneration(int dev_id) {
    return ArchitectureGeneration[dev_id];
  }

  MGARDm_CONT int
  GetMaxNumThreadsPerSM(int dev_id) {
    return MaxNumThreadsPerSM[dev_id];
  }

  MGARDm_CONT
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


}

#endif