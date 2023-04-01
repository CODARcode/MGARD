#ifndef LAGRANGE_OPTIMIZER_HPP
#define LAGRANGE_OPTIMIZER_HPP

#include "adios2/core/Operator.h"

class LagrangeOptimizer {
public:
  // Constructor
  LagrangeOptimizer(const char *species, const char *precision);
  LagrangeOptimizer(size_t planeOffset, size_t nodeOffset, size_t p, size_t n,
                    size_t vx, size_t vy, const uint8_t species,
                    const uint8_t precision);

  // Destructor
  ~LagrangeOptimizer();
  // Compute mesh parameters and QoIs
  void computeParamsAndQoIs(const std::string meshFile, adios2::Dims blockStart,
                            adios2::Dims blockCount, const double *dataIn);
  void writeOutput(const char *varname, std::vector<double> &tensor);
  // Compute Lagrange Parameters
  double *computeLagrangeParameters(const double *reconstructedData);
  // Get the particular species
  const uint8_t getSpecies();
  const uint8_t getPrecision();
  // Get the plane offset
  size_t getPlaneOffset();
  // Get the node offset
  size_t getNodeOffset();
  // Get the number of planes
  size_t getPlaneCount();
  // Get the number of nodes
  size_t getNodeCount();
  // Get the vx dimensions
  size_t getVxCount();
  // Get the vy dimensions
  size_t getVyCount();
  // Get the number of bytes needed to store the Lagrange parameters
  size_t getParameterSize();
  // Get the number of bytes needed to store the PQ table
  size_t getTableSize();
  size_t putResultNoPQ(char *&bufferOut, size_t &bufferOutOffset);
  size_t putResult(char *&bufferOut, size_t &bufferOutOffset,
                   const char *precision);
  size_t putResultV1(char *&bufferOut, size_t &bufferOutOffset);
  size_t putResultV2(char *&bufferOut, size_t &bufferOutOffset);
  void setDataFromCharBufferV1(double *&dataOut, const double *bufferIn,
                               const char *filename);
  void setData(const double *dataOut, const double *bufferIn);
  char *setDataFromCharBuffer(double *&dataOut, const char *bufferIn,
                              size_t bufferTotalSize);
  void setDataFromCharBuffer2(double *&dataOut, const char *bufferIn,
                              size_t bufferOffset, size_t bufferSize);
  void readCharBuffer(const char *bufferIn, size_t bufferOffset,
                      size_t bufferSize);
  void compareQoIs(const double *reconData, const double *bregData);

protected:
  // APIs
  void readF0Params(const std::string meshFile);
  void setVolume();
  void setVolume(std::vector<double> &vol);
  void setVp();
  void setVp(std::vector<double> &vp);
  void setMuQoi();
  void setMuQoi(std::vector<double> &muqoi);
  void setVth2();
  void setVth2(std::vector<double> &vth, std::vector<double> &vth2);
  void compute_C_qois(int iphi, std::vector<double> &density,
                      std::vector<double> &upara, std::vector<double> &tperp,
                      std::vector<double> &tpara, std::vector<double> &n0,
                      std::vector<double> &t0, const double *dataIn);
  bool isConverged(std::vector<double> difflist, double eB, int count);
  void compareErrorsPD(const double *reconData, const double *bregData,
                       int rank);
  void compareErrorsQoI(std::vector<double> &x, std::vector<double> &y,
                        std::vector<double> &z, const char *qoi, int rank);
  double rmseErrorPD(const double *y, double &e, double &maxv, double &minv,
                     double &ysize);
  double rmseError(std::vector<double> &rqoi, std::vector<double> &bqoi,
                   double &e, double &maxv, double &minv, int &ysize);
  double determinant(double a[4][4], double k);
  double **cofactor(double num[4][4], double f);
  double **transpose(double num[4][4], double fac[4][4], double r);
  void initializeClusterCenters(double *&clusters, double *lagarray,
                                int numObjs);
  void quantizeLagranges(int offset, int *&membership, double *&cluster);
  void initializeClusterCentersMPI(double *&clusters, int numP, int myRank,
                                   double *lagarray, int numObjs);
  void quantizeLagrangesMPI(int offset, int *&membership, double *&cluster);
  size_t putPQIndexes(char *&bufferOut, size_t &bufferOutOffset);
  size_t putLagrangeParameters(char *&bufferOut, size_t &bufferOutOffset);
  size_t getPQIndexes(const char *bufferIn);

  // Members
  // Actual data being compressed and related parameters
  uint8_t mySpecies;
  uint8_t myPrecision;
  std::vector<double> myDataIn;
  long unsigned int myPlaneOffset;
  long unsigned int myNodeOffset;
  long unsigned int myNodeCount;
  long unsigned int myPlaneCount;
  long unsigned int myVxCount;
  long unsigned int myVyCount;
  long unsigned int myLocalElements;
  double myMaxValue;
  // Constant physics parameters
  static const int myNumSpecies = 2;
  const char *mySpeciesList[myNumSpecies] = {"electron", "ion"};
  double mySpeciesCharge[myNumSpecies] = {1.6022e-19, 1.6022e-19};
  // double mySpeciesMass[myNumSpecies] = {9.1093E-31, 1.6720E-27};
  double mySpeciesMass[myNumSpecies] = {3.344e-29, 3.344e-27};
  double mySmallElectronCharge;
  double myParticleMass;
  // Mesh Parameters
  std::string myMeshFile;
  std::vector<double> myGridVolume;
  std::vector<int> myF0Nvp;
  std::vector<int> myF0Nmu;
  std::vector<double> myF0Dvp;
  std::vector<double> myF0Dsmu;
  std::vector<double> myF0TEv;
  std::vector<double> myVolume;
  std::vector<double> myVp;
  std::vector<double> myMuQoi;
  std::vector<double> myVth2;
  std::vector<double> myVth;
  // Original QoIs
  std::vector<double> myDensity;
  std::vector<double> myUpara;
  std::vector<double> myTperp;
  std::vector<double> myTpara;
  std::vector<double> myN0;
  std::vector<double> myT0;
  // Lagrange Parameters
  double *myLagranges;
  // PQ parameters
  int myNumClusters;
  int *myLagrangeIndexesDensity;
  int *myLagrangeIndexesUpara;
  int *myLagrangeIndexesTperp;
  int *myLagrangeIndexesRpara;
  double *myDensityTable;
  double *myUparaTable;
  double *myTperpTable;
  double *myRparaTable;
  std::vector<double> myTable;
  double myEpsilon;
  int useKMeansMPI;
};

#endif
