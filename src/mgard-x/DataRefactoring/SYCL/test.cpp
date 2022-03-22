/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */
#include "mgard-x/RuntimeX/RuntimeX.h"

namespace mgard_x {


}

// #include "mgard-x/DataRefactoring/MultiDimension/DataRefactoring.hpp"
// #include "mgard-x/DataRefactoring/SingleDimension/DataRefactoring.hpp"





#include <iostream>

#include <chrono>
namespace mgard_x {




template <DIM D, typename T, SIZE R, SIZE C, SIZE F, OPTION OP, typename DeviceType>
class TestFunctor : public Functor<DeviceType> {
public:
  // TestFunctor() {}
  TestFunctor(SubArray<D, T, DeviceType> v): v(v){
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    *v((IDX)0) = 1.0;
  }
private:
  SubArray<D, T, DeviceType> v;
};

template <DIM D, typename T, typename DeviceType>
class Test {
public:
  Test(){}

  void Execute() {
    using FunctorType = TestFunctor<D, T, 1, 1, 1, COPY, DeviceType>;
    using TaskType = Task<FunctorType>;   
    SubArray<D, T, DeviceType> v;
    FunctorType functor(v);
    Task task(functor, 1, 1, 1, 1, 1, 1, 0, 0, "Test");
    DeviceAdapter<TaskType, DeviceType> adapter;
    ExecutionReturn ret = adapter.Execute(task); 
  }
};

template class Test<1, double, SYCL>;
// 

// template <> inline constexpr bool sycl::is_device_copyable_v<Task<TestFunctor<1, double, 1, 1, 1, COPY, SYCL>>> = true;


} // namespace mgard_x

template <> struct sycl::is_device_copyable<mgard_x::Task<mgard_x::TestFunctor<1, double, 1, 1, 1, COPY, mgard_x::SYCL>>> : std::true_type {};