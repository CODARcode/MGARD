namespace mgard {

// Public member functions.

template <typename T>
HierarchyCoefficients<T>::HierarchyCoefficients(T *const data) : data(data) {}

template <typename T>
T *begin(const MeshHierarchy &hierarchy, const HierarchyCoefficients<T> u) {
  return u.data;
}

template <typename T>
T *end(const MeshHierarchy &hierarchy, const HierarchyCoefficients<T> u) {
  return u.data + hierarchy.ndof();
}

#define DEFINE_HC_SUBCLASS_CONVERSION(name)                                    \
  template <typename T>                                                        \
  template <typename U>                                                        \
  name<T>::operator name<U>() const {                                          \
    return name<U>(static_cast<U *>(this->data));                              \
  }

DEFINE_HC_SUBCLASS_CONVERSION(HierarchyCoefficients)

#define DEFINE_HC_SUBCLASS_CONSTRUCTOR(name)                                   \
  template <typename T>                                                        \
  name<T>::name(T *const data) : HierarchyCoefficients<T>(data) {}

#define DEFINE_HC_SUBCLASS(name)                                               \
  DEFINE_HC_SUBCLASS_CONSTRUCTOR(name)                                         \
  DEFINE_HC_SUBCLASS_CONVERSION(name)

DEFINE_HC_SUBCLASS(NodalCoefficients)

DEFINE_HC_SUBCLASS(MultilevelCoefficients)

DEFINE_HC_SUBCLASS(IndicatorCoefficients)

#undef DEFINE_HC_SUBCLASS

#undef DEFINE_HC_SUBCLASS_CONSTRUCTOR

#undef DEFINE_HC_SUBCLASS_CONVERSION

} // namespace mgard
