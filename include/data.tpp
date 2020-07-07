namespace mgard {

// Public member functions.

template <typename T>
HierarchyCoefficients<T>::HierarchyCoefficients(T *const data) : data(data) {}

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

#define DEFINE_HC_SUBCLASS_COMPARISONS(name)                                   \
  template <typename T> bool operator==(const name<T> &a, const name<T> &b) {  \
    return a.data == b.data;                                                   \
  }                                                                            \
                                                                               \
  template <typename T> bool operator!=(const name<T> &a, const name<T> &b) {  \
    return !operator==(a, b);                                                  \
  }

#define DEFINE_HC_SUBCLASS(name)                                               \
  DEFINE_HC_SUBCLASS_CONSTRUCTOR(name)                                         \
  DEFINE_HC_SUBCLASS_CONVERSION(name)                                          \
  DEFINE_HC_SUBCLASS_COMPARISONS(name)

DEFINE_HC_SUBCLASS(NodalCoefficients)

DEFINE_HC_SUBCLASS(MultilevelCoefficients)

#undef DEFINE_HC_SUBCLASS

#undef DEFINE_HC_SUBCLASS_CONSTRUCTOR

#undef DEFINE_HC_SUBCLASS_CONVERSION

#undef DEFINE_HC_SUBCLASS_COMPARISONS

} // namespace mgard
