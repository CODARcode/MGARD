#ifndef MGARD_X_160_BLOCK_LOCAL_HIERARCHY_DATA_REFACTOR_INTERFACE_HPP
#define MGARD_X_160_BLOCK_LOCAL_HIERARCHY_DATA_REFACTOR_INTERFACE_HPP
namespace mgard_x_160 {

namespace data_refactoring {

template <DIM D, typename T, typename DeviceType>
class BlockLocalHierarchyDataRefactor {
  virtual void Decompose(SubArray<D, T, DeviceType> data, int queue_idx) = 0;
  virtual void Recompose(SubArray<D, T, DeviceType> data, int queue_idx) = 0;
};

} // namespace data_refactoring

} // namespace mgard_x_160

#endif