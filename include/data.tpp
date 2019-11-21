namespace mgard {

//Public member functions.

template <typename T>
HierarchyCoefficients<T>::HierarchyCoefficients(T * const data):
    data(data)
{
}

#define DEFINE_HC_SUBCLASS_CONVERSION(name) \
template <typename T>\
template <typename U>\
name<T>::operator name<U>() const {\
    return name<U>(static_cast<U *>(this->data));\
}

DEFINE_HC_SUBCLASS_CONVERSION(HierarchyCoefficients)

#define DEFINE_HC_SUBCLASS_CONSTRUCTOR(name) \
template <typename T>\
name<T>::name(T * const data):\
    HierarchyCoefficients<T>(data)\
{\
}\

#define DEFINE_HC_SUBCLASS(name) \
DEFINE_HC_SUBCLASS_CONSTRUCTOR(name)\
DEFINE_HC_SUBCLASS_CONVERSION(name)

DEFINE_HC_SUBCLASS(NodalCoefficients)

DEFINE_HC_SUBCLASS(MultilevelCoefficients)

DEFINE_HC_SUBCLASS(IndicatorCoefficients)

#undef DEFINE_HC_SUBCLASS

#undef DEFINE_HC_SUBCLASS_CONSTRUCTOR

#undef DEFINE_HC_SUBCLASS_CONVERSION

}
