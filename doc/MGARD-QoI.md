# MGARD-QOI

The QoI preservation feature of MGARD is designed on top of MGARD to enable error control on linear quantities derived from the reduced data. Based on the input mesh hierarchy and QoI functions, the API computes an operator norm $R_s$ . By using $\tau/R_s$  as the error tolerance for primary data compression, MGARD can guarantee that the resulted errors in QoIs derived from the reduced primary data will be strictly bounded by $\tau$. Please refer [1] for detailed math and implementation. 

### Supporting features
* Linearly derived quantities 
* Dimensions: N-Dim
* Error-bound type: s-norm, L-inf, L2 error
* Data structure: Cartesian grid (uniform, non-uniform)
* Portability: Same as `MGARD-CPU`


### Configure and build:
The QoI functionality is automatically built together with MGARD.

### APIs
The API of MGARD quantities of interest compression is contained in mgard/TensorQuantityOfInterest.hpp. The API has been designed in a way to give users a flexible way of defining their own QoI functions. The output is an operator norm representing the mapping between the error (bounds) of the primary data and the derived QoI.  

* To build the function of operator norm: `mgard::TensorQuantityOfInterest (const TensorMeshHierarchy<N, Real> &hierarchy, const Functional &functional)`

    * [In] hierarchy: tensor mesh hierarchy used by MGARD (see TensorMeshHierarchy.hpp).
    * [In] functional: function object of QoIs.
    * [Return] Q: function for operator norm calculation.

* To calculate the operator norm: `mgard::TensorQuantityOfInterest::norm(const Real s)`
    * [In] s: smoothness parameter.
    * [Return] $R_s$: operator norm; use $\tau/R_s$ as the input to MGARD compression API to guarantee the error of the QoI will be bounded by $\tau$.

### Example Code
MGARD-QoI example code can be found in [here][qoi-example].

[qoi-example]: ../examples/qoi 


[1] Ainsworth, Mark, et al. "Multilevel techniques for compression and reduction of scientific data-quantitative control of accuracy in derived quantities." SIAM Journal on Scientific Computing41.4 (2019): A2146-A2171.
