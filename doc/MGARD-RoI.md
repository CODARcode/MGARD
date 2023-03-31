# MGARD-ROI

The region-adaptive feature of MGARD is designed on top of MGARD to enable regionally varied error control. The API accepts RoI masks as the input. When RoI masks are not provided, the API will perform critical region detection based on an intermediate outcome of MGARD compression and a modified Adaptive Mesh Refinement (AMR) algorithm. The critical region detection returns regions where variational features may exist so they can be compressed with a different error bound than the rest. 

The error bounds used by different regions/data points are encoded in the quantization bins. It can write out the masks outputted from critical region detection, but the masks are not required and will not be used by reconstruction. The region-adaptive compressed data will be reconstructed using the regular MGARD decompression, the same API used by uniformly compressed data. Please refer [1] for detailed algorithms. 

Note: The region-adaptive feature is an experimental component of the MGARD software. The internal algorithms and API designs are subject to change in future release of MGARD. Currently, this feature is only available with CPU implementation. 

#### Supporting features

*	User-defined RoI (RoI mask as the input) and auto-detected RoI (based on variational features)
*	Dimensions: 1D-3D
*	Error-bound type: L_2 error
*	Data structure: Cartesian grid (uniform, non-uniform)
*	Portability: Same as `MGARD-CPU`  

#### Configure and build
The region-adaptive functionality is automatically built together with MGARD-CPU. 

#### APIs 
The API of region-adaptive compression is contained in the mgard/compress.hpp. The API has been designed in a way such that the RoI setup and detection are highly customizable to meet user’s requirements. The decompression API is the same as MGARD. Here lists the key components of MGARD region-adaptive compression. 

* For compression: `mgard::compress_roi (const TensorMeshHierarchy<N, Real> &hierarchy, Real *const v, const Real s, const Real tol, const std::vector<Real> thresh, const std::vector<size_t> init_bw, const std::vector<size_t> bw_ratio, const size_t l_th, const char *roi_mask, bool wr)`

    * [In] hierarchy: tensor mesh hierarchy used by MGARD (see TensorMeshHierarchy.hpp).
    * [In] v: Data to be compressed.
    * [In] s: Smoothness parameter.
    * [In] tol: Error tolerance. 
    * [In] thresh: RoIs as the percentage of data to be kept at each level of refinement.
    * [In] init_bw: Initial bin width used for refinement. 
    * [In] bw_ratio: Ratios of bin width used before and after each level of refinement.
    * [In] l_th: The coarsest level at which MGARD multilevel coefficients will be used for RoI detection 
    * [In] roi_mask: Filename of the RoI mask; use NULL when RoIs are not pre-defiend. The API will the invoke region detection.
    * [In] wr: Indicator of whether to write out the mask of detected RoIs. 
    * [Out][Optional]: the masks of detected RoIs
    * [Return] Compressed data

* For decompression: mgard::decompress(const CompressedDataset<N, Real> &compressed)
    * Please refer MGARD documents

#### Example Code

MGARD-RoI example code can found in [here][roi-example].

[roi-example]: ../examples/roi/

[1] Gong, Qian, et al. "Region-adaptive, Error-controlled Scientific Data Compression using Multilevel Decomposition." Proceedings of the 34th International Conference on Scientific and Statistical Database Management. 2022.

