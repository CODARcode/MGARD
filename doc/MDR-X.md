
# MDR/MDR-X

Multi-precision Data Refactoring designed on top of MGARD for enabling fine-grain progressive data reconstruction with error control. Currently, there are two designs available:

* ***MGARD-DR (MDR):*** Full-featured CPU serial implementaion of multi-precision data refactoring
* ***MGARD-XDR (MDR-X):*** GPU acceleratoed portable implementation of MDR. Key features are implemented with other features under development.

***Note: Both MDR and MDR-X are experimenal compoments of the MGARD software. Their internal algorithms and API designs are subject to change in future releases of MGARD.***

## Supporting features
* **Data type:** Double and single precision floating-point data
* **Dimensions:** 1D-5D
* **Error-bound type:** L\_Inf error and L\_2 error
* **Data structure:** Uniform spaced Cartisan gird
* **Portability:** Same as MGARD-X (MDR-X only)

## Configure and build

Both MDR and MDR-X are automatically built together with MGARD-X. Please follow the [instruction of MGARD-X][mgard-x-build] to build MDR and MDR-X.

[mgard-x-build]: MGARD-X.md

## Use
### Header files
Inculde the follow header files to use MDR or MDR-X:

* MDR: `mgard/mdr.hpp`
* MDR-X: `mgard/mdr-x.hpp`


### APIs

The APIs of MDR and MDR-X are designed in a way such that the refactoring the reconstruction process are highly customizable to satisify users needs. Here lists the key components of MDR and MDR-X.  

* **Decomposer:** Responsible for transforming original multidimensional data to multilevel coefficients (decompose) and the other way around (recompose). This could be any external decorrelation algorithm such as MGARD and wavelet transforms. Currently MDR supports MGARD decomposer (multilinear interpolation with L2 projection, see `MGARDOrthoganalDecomposer`) and hierarchical decomposer (multilinear interpolation, see `MGARDHierarchicalDecomposer`).
 
* **ErrorCollector:** Responsible for collecting error information that is required for error estimation during retrieval. Currently MDR implements a max error collector (collecting the maximum coefficients in each level, see `MaxErrorCollector`) and squared error collector (collecting the sum of squared error in each level, see `SquaredErrorCollector`).
 
* **ErrorEstimator:** Responsible for estimating the error for each precision fragment based on the collected error information. Currently MDR implements a max error estimator (see `MaxErrorEstimator`) and a L2 error estimator (see `SquaredErrorEstimator`) for the two decomposition methods supported.
 
* **Interleaver:** Responsible for linearizing the multidimensional level coefficients to 1D for precision encoding. Currently MDR supports a direct interleaver (linearizing coefficients one by one, see `DirectInterleaver`), a blocked based interleaver (linearizing coefficients in blocks, see `BlockedInterleaver`), and a space-filling-curve based one (linearizing coefficients using specific space filling curves, see `SFCInterleaver`).
 
* **LosslessCompressor:** Responsible for lossless compressing the encoding bit-planes (using ZSTD in the implementation). Currently MDR implements a null compressor (performing no lossless compression, see `NullLevelCompressor`), a default compressor (losslessly compressing each bit-plane, see `DefaultLevelCompressor`),  and an adaptive compressor (compressing the first a few bit-planes based on the characteristics, see `AdaptiveLevelCompressor`).

* **SizeInterpreter:** Responsible for interpreting which precision fragment to fetch upon retrieval. Currently MDR implements an in-order size interpreter (fetching from coarse level to fine level based on bit-plane order, see `InorderSizeInterpreter`), a round-robin size interpreter (fetching one bit-plane per level, see `RoundRobinSizeInterpreter`), and three greedy-based size interpreters (fetching based on the error impact, or efficiency defined in the paper, see `GreedyBasedSizeInterpreter`, `SignExcludeGreedyBasedSizeInterpreter`, `NegaBinaryGreedyBasedSizeInterpreter`).
 
* **Writer:** Responsible for writing precision fragments to files. Users are suggested to implement a derived class to write with their preferred formats and I/O libraries. Currently MDR implements a concatenated writer (writing each level in one file, see `ConcatLevelFileWriter`) and a fragment writer (writing aggregated precision fragments in one file, see `HPSSFileWriter`).
 
* **Retriever:**  Responsible for reading precision fragments (inverse operations of the writer). Will be merged into the writer class in future release.
 
* **Refactor:** Responsible for constructing a data refactor using the components above. Users are encouraged to pick any component according to their needs.
 
* **Reconstructor:** Responsible for constructing a data reconstructor (inverse operations of the Refactor). Will be merged into the Refactor class in future release.

## Example Code
* MDR example code can be found in [here][mdr-example].
* MDR-X example code can be found in [here][mdr-x-example].

[mdr-example]: ../examples/mgard-x/MDR
[mdr-x-example]: ../examples/mgard-x/MDR-X
