
# MDR-X (Experimental)

MRD-X is a portable implementation of the Multi-precision Data Refactoring (MDR) designed on top of MGARD-X for enabling fine-grain progressive data reconstruction with error control. 

***Note: MDR-X is currently an experimenal compoment of MGARD. Its algorithm design, implementation, and APIs are subject to change in future releases of MGARD.***

## Supporting features
* **Data type:** Double and single precision floating-point data
* **Dimensions:** 1D-5D
* **Error-bound type:** L\_Inf error and L\_2 error
* **Data structure:** Uniform spaced Cartisan gird
* **Portability:** Same as MGARD-X

## Configure and build

MDR-X is automatically built together with MGARD-X. Please follow the [instruction of MGARD-X][mgard-x-build] to build MDR-X.

[mgard-x-build]: MGARD-X.md

## Interfaces
To do

## Example Code
* Example code can be found in [here][example].

[example]: ../examples/mgard-x/MDR
