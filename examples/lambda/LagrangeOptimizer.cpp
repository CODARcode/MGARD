#include <sched.h>

#include "KmeansMPI.h"
#include "LagrangeOptimizer.hpp"
#include "adios2.h"
#include <assert.h>
#include <iostream>
#include <map>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <string.h>
#include <time.h>
#define GET4D(d0, d1, d2, d3, i, j, k, l)                                      \
  ((d1 * d2 * d3) * i + (d2 * d3) * j + d3 * k + l)

// int mpi_kmeans(double*, int, int, int, float, int*&, double*&);

LagrangeOptimizer::LagrangeOptimizer(const char *species,
                                     const char *precision) {
  // Initialize charge and mass variables
  for (int i = 0; i < myNumSpecies; ++i) {
    if (!strcmp(species, mySpeciesList[i])) {
      mySmallElectronCharge = mySpeciesCharge[i];
      myParticleMass = mySpeciesMass[i];
      mySpecies = i;
    }
  }
  myPrecision = 0;
  if (!strcmp(precision, "single")) {
    myPrecision = 1;
  } else if (!strcmp(precision, "half")) {
    myPrecision = 2;
  }
  myNumClusters = 256;
  myEpsilon = 100;
  useKMeansMPI = 0;
}

LagrangeOptimizer::LagrangeOptimizer(size_t planeOffset, size_t nodeOffset,
                                     size_t p, size_t n, size_t vx, size_t vy,
                                     const uint8_t species,
                                     const uint8_t precision) {
  // Initialize charge and mass variables
  mySmallElectronCharge = mySpeciesCharge[species];
  myParticleMass = mySpeciesMass[species];
  mySpecies = species;
  myPrecision = precision;
  // Change it if it is for electrons
  myPlaneOffset = planeOffset;
  myNodeOffset = nodeOffset;
  myPlaneCount = p;
  myNodeCount = n;
  myVxCount = vx;
  myVyCount = vy;
  myNumClusters = 256;
  myLagrangeIndexesDensity = NULL;
  myLagrangeIndexesUpara = NULL;
  myLagrangeIndexesTperp = NULL;
  myLagrangeIndexesRpara = NULL;
  myDensityTable = NULL;
  myUparaTable = NULL;
  myTperpTable = NULL;
  myRparaTable = NULL;
  myEpsilon = 100;
  useKMeansMPI = 0;
}

LagrangeOptimizer::~LagrangeOptimizer() {}

void LagrangeOptimizer::computeParamsAndQoIs(const std::string meshFile,
                                             adios2::Dims blockStart,
                                             adios2::Dims blockCount,
                                             const double *dataIn) {
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  myMeshFile = meshFile;
  double start, end;
  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();
  int planeIndex = 0;
  int nodeIndex = 2;
  int velXIndex = 1;
  int velYIndex = 3;
  int iphi = 0;
  myPlaneOffset = blockStart[planeIndex];
  myNodeOffset = blockStart[nodeIndex];
  myNodeCount = blockCount[nodeIndex];
  myPlaneCount = blockCount[planeIndex];
  myVxCount = blockCount[velXIndex];
  myVyCount = blockCount[velYIndex];
#ifdef UF_DEBUG
  printf("#planes: %d, #nodes: %d, #vx: %d, #vy: %d\n", myPlaneCount,
         myNodeCount, myVxCount, myVyCount);
#endif
  myLocalElements = myNodeCount * myPlaneCount * myVxCount * myVyCount;
  myDataIn.reserve(myLocalElements);
  int lindex, rindex;
  for (int i = 0; i < myPlaneCount; i++) {
    for (int j = 0; j < myVxCount; j++) {
      for (int k = 0; k < myNodeCount; k++) {
        for (int l = 0; l < myVyCount; l++) {
          lindex = int(GET4D(myPlaneCount, myNodeCount, myVxCount, myVyCount, i,
                             k, j, l));
          rindex = int(GET4D(myPlaneCount, myVxCount, myNodeCount, myVyCount, i,
                             j, k, l));
          myDataIn[lindex] = dataIn[rindex];
          // GET4D(myDataIn, myPlaneCount, myNodeCount, myVxCount,
          // myVyCount, i, k, j, l) =
          // GET4D(dataIn, myPlaneCount, myVxCount,
          // myNodeCount, myVyCount, i, j, k, l);
        }
      }
    }
  }
  readF0Params(meshFile);
  setVolume();
  setVp();
  setMuQoi();
  setVth2();
#ifdef UF_DEBUG
  printf("volume: gv %d f0_nvp %d f0_nmu %d, vp: %d, vth: %d, vth2: %d, "
         "mu_qoi: %d\n",
         myGridVolume.size(), myF0Nvp.size(), myF0Nmu.size(), myVp.size(),
         myVth.size(), myVth2.size(), myMuQoi.size());
  for (iphi = 0; iphi < myPlaneCount; ++iphi) {
    compute_C_qois(iphi, myDensity, myUpara, myTperp, myTpara, myN0, myT0,
                   myDataIn.data());
  }
#endif
  myMaxValue = 0;
  for (size_t i = 0; i < myLocalElements; ++i) {
    myMaxValue = (myMaxValue > myDataIn[i]) ? myMaxValue : myDataIn[i];
  }
  MPI_Barrier(MPI_COMM_WORLD);
  end = MPI_Wtime();
  if (my_rank == 0) {
    printf("%d Time Taken for QoI Computation: %f\n", mySpecies, (end - start));
  }
}

double *LagrangeOptimizer::computeLagrangeParameters(const double *reconData) {
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  double start, end, start1;
  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();
  int ii, i, j, k, l, m;
  for (ii = 0; ii < myLocalElements; ++ii) {
    if (!(reconData[ii] > 0)) {
      ((double *)reconData)[ii] = myEpsilon;
    }
  }
  std::vector<double> i_g(myLocalElements);
  int lindex, rindex;
  for (int i = 0; i < myPlaneCount; i++) {
#pragma omp parallel for default(none)                                         \
    shared(i, myPlaneCount, myNodeCount, myVxCount, myVyCount, reconData,      \
           i_g) private(lindex, rindex)
    for (int k = 0; k < myNodeCount; k++) {
      for (int j = 0; j < myVxCount; j++) {
        for (int l = 0; l < myVyCount; l++) {
          lindex = int(GET4D(myPlaneCount, myNodeCount, myVxCount, myVyCount, i,
                             k, j, l));
          rindex = int(GET4D(myPlaneCount, myVxCount, myNodeCount, myVyCount, i,
                             j, k, l));
          i_g[lindex] = reconData[rindex];
          // GET4D(i_g, myPlaneCount, myNodeCount, myVxCount,
          // myVyCount, i, k, j, l) =
          // GET4D(reconData, myPlaneCount, myVxCount,
          // myNodeCount, myVyCount, i, j, k, l);
        }
      }
    }
  }
  reconData = i_g.data();
  myLagranges = new double[4 * myPlaneCount * myNodeCount];
  std::vector<double> V2(myNodeCount * myVxCount * myVyCount, 0);
  std::vector<double> V3(myNodeCount * myVxCount * myVyCount, 0);
  std::vector<double> V4(myNodeCount * myVxCount * myVyCount, 0);
#pragma omp parallel for default(none)                                         \
    shared(myNodeCount, myVxCount, myVyCount, myVolume, myVth, myVp, myMuQoi,  \
           myVth2, myParticleMass, V2, V3, V4) private(i, j, l, m)
  for (k = 0; k < myNodeCount * myVxCount * myVyCount; ++k) {
    i = int(k / (myVxCount * myVyCount));
    j = int(k % myVyCount);
    l = int(k % (myVxCount * myVyCount));
    m = int(l / myVyCount);
    V2[k] = myVolume[k] * myVth[i] * myVp[j];
    V3[k] = myVolume[k] * 0.5 * myMuQoi[m] * myVth2[i] * myParticleMass;
    V4[k] = myVolume[k] * pow(myVp[j], 2) * myVth2[i] * myParticleMass;
  }
  int breg_index = 0;
  int iphi, idx;
  double *breg_recon = new double[myLocalElements];
  for (iphi = 0; iphi < myPlaneCount; ++iphi) {
    const double *f0_f = &myDataIn[iphi * myNodeCount * myVxCount * myVyCount];
    std::vector<double> D(myNodeCount, 0);
#pragma omp parallel for default(none)                                         \
    shared(myNodeCount, myVxCount, myVyCount, myVolume, f0_f, D) private(i)
    for (k = 0; k < myNodeCount * myVxCount * myVyCount; ++k) {
      i = int(k / (myVxCount * myVyCount));
      D[i] += f0_f[k] * myVolume[k];
      // if (i > 300) {
      // printf ("Node %d F0F[%d]=%5.3g Volume=%5.3g\n", i, k, f0_f[k],
      // myVolume[k]);
      // }
    }
    std::vector<double> U(myNodeCount, 0);
    std::vector<double> Tperp(myNodeCount, 0);
#pragma omp parallel for default(none)                                         \
    shared(myNodeCount, myVxCount, myVyCount, myVolume, myVth, myVp, f0_f,     \
           myMuQoi, myVth2, myParticleMass, mySmallElectronCharge, D, U,       \
           Tperp) private(i, j, l, m)
    for (k = 0; k < myNodeCount * myVxCount * myVyCount; ++k) {
      i = int(k / (myVxCount * myVyCount));
      j = int(k % myVyCount);
      l = int(k % (myVxCount * myVyCount));
      m = int(l / myVyCount);
      U[i] += (f0_f[k] * myVolume[k] * myVth[i] * myVp[j]) / D[i];
      Tperp[i] += (f0_f[k] * myVolume[k] * 0.5 * myMuQoi[m] * myVth2[i] *
                   myParticleMass) /
                  D[i] / mySmallElectronCharge;
    }
    std::vector<double> Tpara(myNodeCount, 0);
    std::vector<double> Rpara(myNodeCount, 0);
    double en;
#pragma omp parallel for default(none) shared(                                 \
    myNodeCount, myVxCount, myVyCount, myVolume, myVth, myVp, f0_f, myVth2,    \
    myParticleMass, mySmallElectronCharge, D, U, Tpara) private(i, j, en)
    for (k = 0; k < myNodeCount * myVxCount * myVyCount; ++k) {
      i = int(k / (myVxCount * myVyCount));
      j = int(k % myVyCount);
      en = 0.5 * pow((myVp[j] - U[i] / myVth[i]), 2);
      Tpara[i] += 2 *
                  (f0_f[k] * myVolume[k] * en * myVth2[i] * myParticleMass) /
                  D[i] / mySmallElectronCharge;
    }
#pragma omp parallel for default(none)                                         \
    shared(myNodeCount, myVxCount, myVyCount, myVolume, myVth, myVth2,         \
           myParticleMass, mySmallElectronCharge, U, Tpara, Rpara) private(i)
    for (k = 0; k < myNodeCount * myVxCount * myVyCount; ++k) {
      i = int(k / (myVxCount * myVyCount));
      Rpara[i] = mySmallElectronCharge * Tpara[i] +
                 myVth2[i] * myParticleMass * pow((U[i] / myVth[i]), 2);
    }
    int count_unLag = 0;
    std::vector<int> node_unconv;
    double maxD = -99999;
    double maxU = -99999;
    double maxTperp = -99999;
    double maxTpara = -99999;
#pragma omp parallel for default(none)                                         \
    shared(myNodeCount, maxD, maxU, maxTperp, maxTpara, D, U, Tperp, Rpara)
    for (i = 0; i < myNodeCount; ++i) {
      if (D[i] > maxD) {
        maxD = D[i];
      }
      if (U[i] > maxU) {
        maxU = U[i];
      }
      if (Tperp[i] > maxTperp) {
        maxTperp = Tperp[i];
      }
      if (Rpara[i] > maxTpara) {
        maxTpara = Rpara[i];
      }
    }
    double DeB = pow(maxD * 1e-09, 2);
    double UeB = pow(maxU * 1e-09, 2);
    double TperpEB = pow(maxTperp * 1e-09, 2);
    double TparaEB = pow(maxTpara * 1e-09, 2);
    double PDeB = pow(myMaxValue * 1e-09, 2);
#if UF_DEBUG
    if (my_rank == 0) {
      printf("Max: D %f U %d Tperp %f Tpara %f PD %f\n", maxD, maxU, maxTperp,
             maxTpara, myMaxValue);
      printf("Bounds: D %f U %d Tperp %f Tpara %f PD %f\n", DeB, UeB, TperpEB,
             TparaEB, PDeB);
    }
#endif
    int maxIter = 50;
#pragma omp parallel for default(none)                                         \
    shared(reconData, iphi, D, U, V2, V3, V4, f0_f, Tperp, Rpara, DeB, UeB,    \
           TperpEB, TparaEB, PDeB, maxIter, node_unconv,                       \
           my_rank) private(count_unLag, breg_recon)
    for (idx = 0; idx < myNodeCount; ++idx) {
      int count = 0;
      double gradients[4] = {0.0, 0.0, 0.0, 0.0};
      double hessians[4][4] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      double K[myVxCount * myVyCount];
      double breg_result[myVxCount * myVyCount];
      memset(K, 0, myVxCount * myVyCount * sizeof(double));
      std::vector<double> L2_den(maxIter, 0);
      std::vector<double> L2_upara(maxIter, 0);
      std::vector<double> L2_tperp(maxIter, 0);
      std::vector<double> L2_tpara(maxIter, 0);
      std::vector<double> L2_PD(maxIter, 0);
      std::fill(L2_den.begin(), L2_den.end(), 0);
      std::fill(L2_upara.begin(), L2_upara.end(), 0);
      std::fill(L2_tperp.begin(), L2_tperp.end(), 0);
      std::fill(L2_tpara.begin(), L2_tpara.end(), 0);
      std::fill(L2_PD.begin(), L2_PD.end(), 0);
      const double *recon_one =
          &reconData[myNodeCount * myVxCount * myVyCount * iphi +
                     myVxCount * myVyCount * idx];
      double lambdas[4] = {0.0, 0.0, 0.0, 0.0};
      count = 0;
      double aD = D[idx] * mySmallElectronCharge;
      int i;
      while (1) {
        for (i = 0; i < myVxCount * myVyCount; ++i) {
          K[i] = lambdas[0] * myVolume[myVxCount * myVyCount * idx + i] +
                 lambdas[1] * V2[myVxCount * myVyCount * idx + i] +
                 lambdas[2] * V3[myVxCount * myVyCount * idx + i] +
                 lambdas[3] * V4[myVxCount * myVyCount * idx + i];
        }
#ifdef UF_DEBUG
        printf("Iteration: %d L1 %g, L2 %g L3 %g, L4 %g K[0] %g\n", count,
               lambdas[0], lambdas[1], lambdas[2], lambdas[3], exp(-K[0]));
#endif
        double update_D = 0, update_U = 0, update_Tperp = 0, update_Tpara = 0,
               rmse_pd = 0;
        if (count > 0) {
          for (i = 0; i < myVxCount * myVyCount; ++i) {
            breg_result[i] = recon_one[i] * exp(-K[i]);
            update_D +=
                breg_result[i] * myVolume[myVxCount * myVyCount * idx + i];
            update_U +=
                breg_result[i] * V2[myVxCount * myVyCount * idx + i] / D[idx];
            update_Tperp +=
                breg_result[i] * V3[myVxCount * myVyCount * idx + i] / aD;
            update_Tpara +=
                breg_result[i] * V4[myVxCount * myVyCount * idx + i] / D[idx];
            rmse_pd += pow(
                (breg_result[i] - f0_f[myVxCount * myVyCount * idx + i]), 2);
          }
#ifdef UF_DEBUG
          if (my_rank == 0) {
            printf("updated D %5.3g, U %5.3g, Tperp %5.3g, Tpara %5.3g, PD "
                   "%5.3g\n",
                   update_D, update_U, update_Tperp, update_Tpara, rmse_pd);
          }
#endif
          L2_den[count] = pow((update_D - D[idx]), 2);
          L2_upara[count] = pow((update_U - U[idx]), 2);
          L2_tperp[count] = pow((update_Tperp - Tperp[idx]), 2);
          L2_tpara[count] = pow((update_Tpara - Rpara[idx]), 2);
          L2_PD[count] = sqrt(rmse_pd);
#ifdef UF_DEBUG
          if (my_rank == 0) {
            printf("Errors count %d D %5.3g, U %5.3g, Tperp %5.3g, Tpara "
                   "%5.3g, PD %5.3g\n",
                   count, L2_den[count], L2_upara[count], L2_tperp[count],
                   L2_tpara[count], L2_PD[count]);
          }
#endif
          bool c1, c2, c3, c4;
          bool converged = (isConverged(L2_den, DeB, count) &&
                            isConverged(L2_upara, UeB, count) &&
                            isConverged(L2_tpara, TparaEB, count) &&
                            isConverged(L2_tperp, TperpEB, count)) &&
                           isConverged(L2_PD, PDeB, count);
          if (converged) {
            for (i = 0; i < myVxCount * myVyCount; ++i) {
              breg_recon[breg_index++] = breg_result[i];
            }
            /*
            double mytperp[1521];
            for (i=0; i<myVxCount*myVyCount; ++i) {
                mytperp[i] = breg_result[i]*V3[
                    myVxCount*myVyCount*idx + i]/aD;
            }
            printf ("Mytperp %d\n", mytperp[0]);
            */
            int lagIndex = myNodeCount * iphi + idx;
            myLagranges[lagIndex * 4] = lambdas[0];
            myLagranges[lagIndex * 4 + 1] = lambdas[1];
            myLagranges[lagIndex * 4 + 2] = lambdas[2];
            myLagranges[lagIndex * 4 + 3] = lambdas[3];
#ifdef UF_DEBUG
            if (my_rank == 0) {
              printf("Node: %d, Dpred %f Dact %f, Upred %f Uact %f, Tperp-pred "
                     "%f Tperp-act %f, Tapara-pred %f Tpara-act %f\n",
                     idx, update_D, D[idx], update_U, U[idx], update_Tperp,
                     Tperp[idx], update_Tpara, Rpara[idx]);
              for (i = 0; i < myVxCount * myVyCount; ++i) {
                printf("Node %d, pre nK %f, x %d, y %d, n %d, %g\n", idx, K[i],
                       i / 37, i % 37, idx, breg_result[i]);
              }
            }
#endif
            break;
          } else if (count == maxIter && !converged) {
            for (i = 0; i < myVxCount * myVyCount; ++i) {
              breg_recon[breg_index++] = recon_one[i];
            }
            converged = (isConverged(L2_den, DeB, count) &&
                         isConverged(L2_upara, UeB, count) &&
                         isConverged(L2_tpara, TparaEB, count) &&
                         isConverged(L2_tperp, TperpEB, count) &&
                         isConverged(L2_PD, PDeB, count));
            int lagIndex = myNodeCount * iphi + idx;
            myLagranges[lagIndex * 4] = 0;
            myLagranges[lagIndex * 4 + 1] = 0;
            myLagranges[lagIndex * 4 + 2] = 0;
            myLagranges[lagIndex * 4 + 3] = 0;
            printf("Node %d did not converge\n", idx);
            count_unLag = count_unLag + 1;
#pragma omp critical
            { node_unconv.push_back(idx); }
            break;
          }
        }
        double gvalue1 = D[idx], gvalue2 = U[idx] * D[idx];
        double gvalue3 = Tperp[idx] * aD, gvalue4 = Rpara[idx] * D[idx];
        double hvalue1 = 0, hvalue2 = 0, hvalue3 = 0, hvalue4 = 0;
        double hvalue5 = 0, hvalue6 = 0, hvalue7 = 0;
        double hvalue8 = 0, hvalue9 = 0, hvalue10 = 0;

        for (i = 0; i < myVxCount * myVyCount; ++i) {
          gvalue1 += recon_one[i] * myVolume[myVxCount * myVyCount * idx + i] *
                     exp(-K[i]) * -1.0;
          gvalue2 += recon_one[i] * V2[myVxCount * myVyCount * idx + i] *
                     exp(-K[i]) * -1.0;
          gvalue3 += recon_one[i] * V3[myVxCount * myVyCount * idx + i] *
                     exp(-K[i]) * -1.0;
          gvalue4 += recon_one[i] * V4[myVxCount * myVyCount * idx + i] *
                     exp(-K[i]) * -1.0;

          hvalue1 += recon_one[i] *
                     pow(myVolume[myVxCount * myVyCount * idx + i], 2) *
                     exp(-K[i]);
          hvalue2 += recon_one[i] * myVolume[myVxCount * myVyCount * idx + i] *
                     V2[myVxCount * myVyCount * idx + i] * exp(-K[i]);
          hvalue3 += recon_one[i] * myVolume[myVxCount * myVyCount * idx + i] *
                     V3[myVxCount * myVyCount * idx + i] * exp(-K[i]);
          hvalue4 += recon_one[i] * myVolume[myVxCount * myVyCount * idx + i] *
                     V4[myVxCount * myVyCount * idx + i] * exp(-K[i]);
          hvalue5 += recon_one[i] *
                     pow(V2[myVxCount * myVyCount * idx + i], 2) * exp(-K[i]);
          hvalue6 += recon_one[i] * V2[myVxCount * myVyCount * idx + i] *
                     V3[myVxCount * myVyCount * idx + i] * exp(-K[i]);
          hvalue7 += recon_one[i] * V2[myVxCount * myVyCount * idx + i] *
                     V4[myVxCount * myVyCount * idx + i] * exp(-K[i]);
          hvalue8 += recon_one[i] *
                     pow(V3[myVxCount * myVyCount * idx + i], 2) * exp(-K[i]);
          hvalue9 += recon_one[i] * V3[myVxCount * myVyCount * idx + i] *
                     V4[myVxCount * myVyCount * idx + i] * exp(-K[i]);
          hvalue10 += recon_one[i] *
                      pow(V4[myVxCount * myVyCount * idx + i], 2) * exp(-K[i]);
        }
        gradients[0] = gvalue1;
        gradients[1] = gvalue2;
        gradients[2] = gvalue3;
        gradients[3] = gvalue4;
#ifdef UF_DEBUG
        if (idx == 5427) {
          printf("Element %d, Iter %d, grad0 = %f, grad1 = %f, grad2 = %f, "
                 "grad3 = %f\n",
                 idx, count, gvalue1, gvalue2, gvalue3, gvalue4);
        }
#endif
        hessians[0][0] = hvalue1;
        hessians[0][1] = hvalue2;
        hessians[0][2] = hvalue3;
        hessians[0][3] = hvalue4;
        hessians[1][0] = hvalue2;
        hessians[1][1] = hvalue5;
        hessians[1][2] = hvalue6;
        hessians[1][3] = hvalue7;
        hessians[2][0] = hvalue3;
        hessians[2][1] = hvalue6;
        hessians[2][2] = hvalue8;
        hessians[2][3] = hvalue9;
        hessians[3][0] = hvalue4;
        hessians[3][1] = hvalue7;
        hessians[3][2] = hvalue9;
        hessians[3][3] = hvalue10;
        // compute lambdas
        int order = 4;
        int k;
        double d = determinant(hessians, order);
        if (d == 0) {
          printf("Need to define pesudoinverse for matrix in node %d\n", idx);
          printf("%5.3g %5.3g %5.3g %5.3g\n", hessians[0][0], hessians[0][1],
                 hessians[0][2], hessians[0][3]);
          printf("%5.3g %5.3g %5.3g %5.3g\n", hessians[1][0], hessians[1][1],
                 hessians[1][2], hessians[1][3]);
          printf("%5.3g %5.3g %5.3g %5.3g\n", hessians[2][0], hessians[2][1],
                 hessians[2][2], hessians[2][3]);
          printf("%5.3g %5.3g %5.3g %5.3g\n", hessians[3][0], hessians[3][1],
                 hessians[3][2], hessians[3][3]);
          break;
        } else {
          double **inverse = cofactor(hessians, order);
#if UF_DEBUG
          printf("Hessians: %g, %g, %g, %g\n", hessians[0][0], hessians[0][1],
                 hessians[0][2], hessians[0][3]);
          printf("Inverse: %g, %g, %g, %g\n", inverse[0][0], inverse[0][1],
                 inverse[0][2], inverse[0][3]);
#endif
          double matmul[4] = {0, 0, 0, 0};
          for (i = 0; i < 4; ++i) {
            matmul[i] = 0;
            for (k = 0; k < 4; ++k) {
              matmul[i] += inverse[i][k] * gradients[k];
            }
          }
          lambdas[0] = lambdas[0] - matmul[0];
          lambdas[1] = lambdas[1] - matmul[1];
          lambdas[2] = lambdas[2] - matmul[2];
          lambdas[3] = lambdas[3] - matmul[3];
        }
        count = count + 1;
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  end = MPI_Wtime();
  if (my_rank == 0) {
    printf("%d Time Taken for Lagrange Computations: %f %f\n", mySpecies,
           end - start, end - start1);
  }
  // double* breg_recon = new double[myLocalElements];
  // memset(breg_recon, 0, myLocalElements*sizeof(double));
  double *new_recon = new double[myLocalElements];
  ;
  const char *precision = "double";
  double nK[myVxCount * myVyCount];
  for (iphi = 0; iphi < myPlaneCount; ++iphi) {
    for (idx = 0; idx < myNodeCount; ++idx) {
      const double *recon_one =
          &reconData[myNodeCount * myVxCount * myVyCount * iphi +
                     myVxCount * myVyCount * idx];
      double *new_recon_one =
          &new_recon[myNodeCount * myVxCount * myVyCount * iphi +
                     myVxCount * myVyCount * idx];
      int x = 4 * (iphi * myNodeCount + idx);
      for (i = 0; i < myVxCount * myVyCount; ++i) {
        if (!strcmp(precision, "float")) {
          nK[i] =
              float(myLagranges[x]) *
                  myVolume[myVxCount * myVyCount * idx + i] +
              float(myLagranges[x + 1]) * V2[myVxCount * myVyCount * idx + i] +
              float(myLagranges[x + 2]) * V3[myVxCount * myVyCount * idx + i] +
              float(myLagranges[x + 3]) * V4[myVxCount * myVyCount * idx + i];
        } else if (!strcmp(precision, "double")) {
          nK[i] = (myLagranges[x]) * myVolume[myVxCount * myVyCount * idx + i] +
                  (myLagranges[x + 1]) * V2[myVxCount * myVyCount * idx + i] +
                  (myLagranges[x + 2]) * V3[myVxCount * myVyCount * idx + i] +
                  (myLagranges[x + 3]) * V4[myVxCount * myVyCount * idx + i];
        }
        new_recon_one[i] = recon_one[i] * exp(-nK[i]);
#if UF_DEBUG
        if (my_rank == 0) {
          printf("Node %d, post nK %f, x %d, y %d, n %d %g\n", idx, nK[i],
                 i / 37, i % 37, idx, new_recon_one[i]);
        }
#endif
      }
    }
  }
  compareQoIs(reconData, new_recon);
  return myLagranges;
}

void LagrangeOptimizer::initializeClusterCenters(double *&clusters,
                                                 double *lagarray,
                                                 int numObjs) {
  clusters = new double[myNumClusters];
  assert(clusters != NULL);

  srand(time(NULL));
  double *myNumbers = new double[myNumClusters];
  std::map<int, int> mymap;
  for (int i = 0; i < myNumClusters; ++i) {
    int index = rand() % numObjs;
    while (mymap.find(index) != mymap.end()) {
      index = rand() % numObjs;
    }
    clusters[i] = lagarray[index];
    mymap[index] = i;
  }
}

void LagrangeOptimizer::quantizeLagranges(int offset, int *&membership,
                                          double *&clusters) {
  int numObjs = myPlaneCount * myNodeCount;
  float threshold = 0.0001;
  double *lagarray = new double[myNodeCount];
  for (int iphi = 0; iphi < myPlaneCount; ++iphi) {
    for (int idx = 0; idx < myNodeCount; ++idx) {
      lagarray[iphi * myNodeCount + idx] =
          myLagranges[iphi * myNodeCount + 4 * idx + offset];
    }
  }

  initializeClusterCenters(clusters, lagarray, numObjs);
  membership = new int[numObjs];
  memset(membership, 0, numObjs * sizeof(int));
  kmeans(lagarray, numObjs, myNumClusters, threshold, membership, clusters);
  return;
}

void LagrangeOptimizer::initializeClusterCentersMPI(double *&clusters, int numP,
                                                    int myRank,
                                                    double *lagarray,
                                                    int numObjs) {
  clusters = new double[myNumClusters];
  assert(clusters != NULL);
  int *counts = new int[numP];
  int *disps = new int[numP];

  int pertask = myNumClusters / numP;
  for (int i = 0; i < numP - 1; i++) {
    counts[i] = pertask;
  }
  counts[numP - 1] = myNumClusters - pertask * (numP - 1);

  disps[0] = 0;
  for (int i = 1; i < numP; i++) {
    disps[i] = disps[i - 1] + counts[i - 1];
  }

  srand(time(NULL));
  int myNumClusters = counts[myRank];
  double *myNumbers = new double[myNumClusters];
  std::map<int, int> mymap;
  for (int i = 0; i < myNumClusters; ++i) {
    int index = rand() % numObjs;
    while (mymap.find(index) != mymap.end()) {
      index = rand() % numObjs;
    }
    myNumbers[i] = lagarray[index];
    mymap[index] = i;
  }
  MPI_Allgatherv(myNumbers, myNumClusters, MPI_DOUBLE, clusters, counts, disps,
                 MPI_DOUBLE, MPI_COMM_WORLD);
}

void LagrangeOptimizer::quantizeLagrangesMPI(int offset, int *&membership,
                                             double *&clusters) {
  int numObjs = myPlaneCount * myNodeCount;
  float threshold = 0.01;
  int num_procs;
  int my_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  double *lagarray = new double[myNodeCount];
  for (int iphi = 0; iphi < myPlaneCount; ++iphi) {
    for (int idx = 0; idx < myNodeCount; ++idx) {
      lagarray[iphi * myNodeCount + idx] =
          myLagranges[iphi * myNodeCount + 4 * idx + offset];
    }
  }

  initializeClusterCentersMPI(clusters, num_procs, my_rank, lagarray, numObjs);
  membership = new int[numObjs];
  memset(membership, 0, numObjs * sizeof(int));
  mpi_kmeans(lagarray, numObjs, myNumClusters, threshold, membership, clusters);
  return;
}

const uint8_t LagrangeOptimizer::getSpecies() { return mySpecies; }

const uint8_t LagrangeOptimizer::getPrecision() { return myPrecision; }

size_t LagrangeOptimizer::getPlaneOffset() { return myPlaneOffset; }

size_t LagrangeOptimizer::getNodeOffset() { return myNodeOffset; }

size_t LagrangeOptimizer::getPlaneCount() { return myPlaneCount; }

size_t LagrangeOptimizer::getNodeCount() { return myNodeCount; }

size_t LagrangeOptimizer::getVxCount() { return myVxCount; }

size_t LagrangeOptimizer::getVyCount() { return myVyCount; }

size_t LagrangeOptimizer::getParameterSize() {
  return myNodeCount * 4 * sizeof(double);
}

// Get the number of bytes needed to store the PQ table
size_t LagrangeOptimizer::getTableSize() { return 0; }

size_t LagrangeOptimizer::putLagrangeParameters(char *&bufferOut,
                                                size_t &bufferOutOffset) {
#if 0
    int i, count = 0;
    int numObjs = myPlaneCount*myNodeCount;
    for (i=0; i<numObjs*4; i+=4) {
        *reinterpret_cast<float*>(
              bufferOut+bufferOutOffset+(count++)*sizeof(float)) =
                  myLagranges[i];
    }
    for (i=0; i<numObjs; ++i) {
        *reinterpret_cast<float*>(
              bufferOut+bufferOutOffset+(count++)*sizeof(float)) =
                  myLagranges[i+1];
    }
    for (i=0; i<numObjs; ++i) {
        *reinterpret_cast<float*>(
              bufferOut+bufferOutOffset+(count++)*sizeof(float)) =
                  myLagranges[i+2];
    }
    for (i=0; i<numObjs; ++i) {
        *reinterpret_cast<float*>(
              bufferOut+bufferOutOffset+(count++)*sizeof(float)) =
                  myLagranges[i+3];
    }
    return count * sizeof(float);
#else
  int i, count = 0;
  int numObjs = myPlaneCount * myNodeCount;
  for (i = 0; i < numObjs * 4; i += 4) {
    *reinterpret_cast<double *>(bufferOut + bufferOutOffset +
                                (count++) * sizeof(double)) = myLagranges[i];
  }
  for (i = 0; i < numObjs; ++i) {
    *reinterpret_cast<double *>(bufferOut + bufferOutOffset +
                                (count++) * sizeof(double)) =
        myLagranges[i + 1];
  }
  for (i = 0; i < numObjs; ++i) {
    *reinterpret_cast<double *>(bufferOut + bufferOutOffset +
                                (count++) * sizeof(double)) =
        myLagranges[i + 2];
  }
  for (i = 0; i < numObjs; ++i) {
    *reinterpret_cast<double *>(bufferOut + bufferOutOffset +
                                (count++) * sizeof(double)) =
        myLagranges[i + 3];
  }
  return count * sizeof(double);
#endif
}

size_t LagrangeOptimizer::putResultV2(char *&bufferOut,
                                      size_t &bufferOutOffset) {
  // TODO: after your algorithm is done, put the result into
  // *reinterpret_cast<double*>(bufferOut+bufferOutOffset) for your       first
  // double number *reinterpret_cast<double*>(bufferOut+bufferOutOff      set+8)
  // for your second double number and so on
  int intbytes = putLagrangeParameters(bufferOut, bufferOutOffset);
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  if (my_rank == 0) {
    FILE *fp = fopen("PqMeshInfo.bin", "wb");
    int str_length = myMeshFile.length();
    printf("Mesh file %d %s\n", str_length, myMeshFile.c_str());
    fwrite(&str_length, sizeof(int), 1, fp);
    fwrite(myMeshFile.c_str(), sizeof(char), str_length, fp);
    fclose(fp);
  }
  return intbytes;
}

size_t LagrangeOptimizer::putPQIndexes(char *&bufferOut,
                                       size_t &bufferOutOffset) {
  int i, intcount = 0, count = 0, singlecount = 0, bufferbytes = 0;
  int numObjs = myPlaneCount * myNodeCount;
  for (i = 0; i < numObjs; ++i) {
    *reinterpret_cast<int *>(bufferOut + bufferOutOffset +
                             (intcount++) * sizeof(uint8_t)) =
        uint8_t(myLagrangeIndexesDensity[i]);
  }
  for (i = 0; i < numObjs; ++i) {
    *reinterpret_cast<int *>(bufferOut + bufferOutOffset +
                             (intcount++) * sizeof(uint8_t)) =
        uint8_t(myLagrangeIndexesUpara[i]);
  }
  for (i = 0; i < numObjs; ++i) {
    *reinterpret_cast<int *>(bufferOut + bufferOutOffset +
                             (intcount++) * sizeof(uint8_t)) =
        uint8_t(myLagrangeIndexesTperp[i]);
  }
  for (i = 0; i < numObjs; ++i) {
    *reinterpret_cast<int *>(bufferOut + bufferOutOffset +
                             (intcount++) * sizeof(uint8_t)) =
        uint8_t(myLagrangeIndexesRpara[i]);
  }
  bufferbytes = intcount * sizeof(uint8_t);
  if (useKMeansMPI == 0) {
    singlecount = 1;
    int intbytes = intcount * sizeof(uint8_t);
    *reinterpret_cast<int *>(bufferOut + bufferOutOffset + intbytes) =
        myNumClusters;
    intbytes += singlecount * sizeof(int);
    for (i = 0; i < myNumClusters; ++i) {
      *reinterpret_cast<int *>(bufferOut + bufferOutOffset + intbytes +
                               (count++) * sizeof(double)) = myDensityTable[i];
    }
    for (i = 0; i < myNumClusters; ++i) {
      *reinterpret_cast<int *>(bufferOut + bufferOutOffset + intbytes +
                               (count++) * sizeof(double)) = myUparaTable[i];
    }
    for (i = 0; i < myNumClusters; ++i) {
      *reinterpret_cast<int *>(bufferOut + bufferOutOffset + intbytes +
                               (count++) * sizeof(double)) = myTperpTable[i];
    }
    for (i = 0; i < myNumClusters; ++i) {
      *reinterpret_cast<int *>(bufferOut + bufferOutOffset + intbytes +
                               (count++) * sizeof(double)) = myRparaTable[i];
    }
    bufferbytes = intbytes + count * sizeof(double);
  }
  return bufferbytes;
}

size_t LagrangeOptimizer::getPQIndexes(const char *bufferIn) {
  int i, intcount = 0;
  for (i = 0; i < myNodeCount; ++i) {
    myLagrangeIndexesDensity[i] = *(reinterpret_cast<const uint8_t *>(
        bufferIn + (intcount++) * sizeof(uint8_t)));
  }
  for (i = 0; i < myNodeCount; ++i) {
    myLagrangeIndexesUpara[i] = (*reinterpret_cast<const uint8_t *>(
        bufferIn + (intcount++) * sizeof(uint8_t)));
  }
  for (i = 0; i < myNodeCount; ++i) {
    myLagrangeIndexesTperp[i] = (*reinterpret_cast<const uint8_t *>(
        bufferIn + (intcount++) * sizeof(uint8_t)));
  }
  for (i = 0; i < myNodeCount; ++i) {
    myLagrangeIndexesRpara[i] = (*reinterpret_cast<const uint8_t *>(
        bufferIn + (intcount++) * sizeof(uint8_t)));
  }
  return intcount * sizeof(uint8_t);
}

size_t LagrangeOptimizer::putResultV1(char *&bufferOut,
                                      size_t &bufferOutOffset) {
  // TODO: after your algorithm is done, put the result into
  // *reinterpret_cast<double*>(bufferOut+bufferOutOffset) for your       first
  // double number *reinterpret_cast<double*>(bufferOut+bufferOutOff      set+8)
  // for your second double number and so on
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  int intbytes = putPQIndexes(bufferOut, bufferOutOffset);
  // printf ("Rank %d numObjs %d numBytes %d\n", my_rank,
  // myPlaneCount*myNodeCount, intbytes);
  if (my_rank == 0 && useKMeansMPI == 1) {
    FILE *fp = fopen("PqMeshInfo.bin", "wb");
    // write out the PQ table and the mesh parameters
    fwrite(&myNumClusters, sizeof(int), 1, fp);
    fwrite(myDensityTable, sizeof(double), myNumClusters, fp);
    fwrite(myUparaTable, sizeof(double), myNumClusters, fp);
    fwrite(myTperpTable, sizeof(double), myNumClusters, fp);
    fwrite(myRparaTable, sizeof(double), myNumClusters, fp);
    int str_length = myMeshFile.length();
    printf("Mesh file %d %s\n", str_length, myMeshFile.c_str());
    fwrite(&str_length, sizeof(int), 1, fp);
    fwrite(myMeshFile.c_str(), sizeof(char), str_length, fp);
    fclose(fp);
  }
  return intbytes;
}

char *LagrangeOptimizer::setDataFromCharBuffer(double *&reconData,
                                               const char *bufferIn,
                                               size_t sizeOut) {
  if (myLagrangeIndexesDensity == NULL) {
    myLagrangeIndexesDensity = new int[myNodeCount];
    myLagrangeIndexesUpara = new int[myNodeCount];
    myLagrangeIndexesTperp = new int[myNodeCount];
    myLagrangeIndexesRpara = new int[myNodeCount];
    myDensityTable = new double[myNumClusters];
    myUparaTable = new double[myNumClusters];
    myTperpTable = new double[myNumClusters];
    myRparaTable = new double[myNumClusters];
  }
  // size_t bufferOffset = getPQIndexes(bufferIn);
  FILE *fp = fopen("PqMeshInfo.bin", "rb");
  fread(&myNumClusters, sizeof(int), 1, fp);
  fread(myDensityTable, sizeof(double), myNumClusters, fp);
  fread(myUparaTable, sizeof(double), myNumClusters, fp);
  fread(myTperpTable, sizeof(double), myNumClusters, fp);
  fread(myRparaTable, sizeof(double), myNumClusters, fp);
  int str_length = 0;
  fread(&str_length, sizeof(int), 1, fp);
  char meshFile[str_length];
  fread(meshFile, sizeof(char), str_length, fp);
  fclose(fp);
  readF0Params(std::string(meshFile, 0, str_length));
  std::vector<double> V2(myNodeCount * myVxCount * myVyCount, 0);
  std::vector<double> V3(myNodeCount * myVxCount * myVyCount, 0);
  std::vector<double> V4(myNodeCount * myVxCount * myVyCount, 0);
  double nK[myVxCount * myVyCount];
  int i, j, k, l, m;
  myLocalElements = myNodeCount * myPlaneCount * myVxCount * myVyCount;
  for (i = 0; i < myLocalElements; ++i) {
    if (!(reconData[i] > 0)) {
      ((double *)reconData)[i] = myEpsilon;
    }
  }
  setVolume();
  setVp();
  setMuQoi();
  setVth2();
  for (k = 0; k < myNodeCount * myVxCount * myVyCount; ++k) {
    i = int(k / (myVxCount * myVyCount));
    j = int(k % myVxCount);
    l = int(k % (myVxCount * myVyCount));
    m = int(l / myVyCount);
    V2[k] = myVolume[k] * myVth[i] * myVp[j];
    V3[k] = myVolume[k] * 0.5 * myMuQoi[m] * myVth2[i] * myParticleMass;
    V4[k] = myVolume[k] * pow(myVp[j], 2) * myVth2[i] * myParticleMass;
  }
  int iphi, idx;
  for (iphi = 0; iphi < myPlaneCount; ++iphi) {
    for (idx = 0; idx < myNodeCount; ++idx) {
      const double *recon_one =
          &reconData[myNodeCount * myVxCount * myVyCount * iphi +
                     myVxCount * myVyCount * idx];
      int m1 = myLagrangeIndexesDensity[iphi * myNodeCount + idx];
      int m2 = myLagrangeIndexesUpara[iphi * myNodeCount + idx];
      int m3 = myLagrangeIndexesTperp[iphi * myNodeCount + idx];
      int m4 = myLagrangeIndexesRpara[iphi * myNodeCount + idx];
      double c1 = myDensityTable[m1];
      double c2 = myUparaTable[m2];
      double c3 = myTperpTable[m3];
      double c4 = myRparaTable[m4];
      for (i = 0; i < myVxCount * myVyCount; ++i) {
        nK[i] = (c1)*myVolume[myVxCount * myVyCount * idx + i] +
                (c2)*V2[myVxCount * myVyCount * idx + i] +
                (c3)*V3[myVxCount * myVyCount * idx + i] +
                (c4)*V4[myVxCount * myVyCount * idx + i];
        ((double *)recon_one)[i] = recon_one[i] * exp(-nK[i]);
      }
    }
  }
  return reinterpret_cast<char *>(reconData);
}

void LagrangeOptimizer::setData(const double *reconData,
                                const double *bufferIn) {
  // size_t bufferOffset = getPQIndexes(bufferIn);
  // FILE* fp = fopen("PqMeshInfo.bin", "rb");
  // fread(&myNumClusters, sizeof(int), 1, fp);
  // fread(myDensityTable, sizeof(double), myNumClusters, fp);
  // fread(myUparaTable, sizeof(double), myNumClusters, fp);
  // fread(myTperpTable, sizeof(double), myNumClusters, fp);
  // fread(myRparaTable, sizeof(double), myNumClusters, fp);
  // int str_length = 0;
  // fread(&str_length, sizeof(int), 1, fp);
  // char meshFile[str_length];
  // fread(meshFile, sizeof(char), str_length, fp);
  // fclose(fp);
  std::vector<double> V2(myNodeCount * myVxCount * myVyCount, 0);
  std::vector<double> V3(myNodeCount * myVxCount * myVyCount, 0);
  std::vector<double> V4(myNodeCount * myVxCount * myVyCount, 0);
  int i, j, k, l, m;
  myLocalElements = myNodeCount * myPlaneCount * myVxCount * myVyCount;
  for (i = 0; i < myLocalElements; ++i) {
    if (!(reconData[i] > 0)) {
      ((double *)reconData)[i] = myEpsilon;
    }
  }
  for (k = 0; k < myNodeCount * myVxCount * myVyCount; ++k) {
    i = int(k / (myVxCount * myVyCount));
    j = int(k % myVxCount);
    l = int(k % (myVxCount * myVyCount));
    m = int(l / myVyCount);
    V2[k] = myVolume[k] * myVth[i] * myVp[j];
    V3[k] = myVolume[k] * 0.5 * myMuQoi[m] * myVth2[i] * myParticleMass;
    V4[k] = myVolume[k] * pow(myVp[j], 2) * myVth2[i] * myParticleMass;
  }
  int numLagrangeParameters = myPlaneCount * myNodeCount * 4;
  for (i = 0; i < numLagrangeParameters; ++i) {
    myLagranges[i] = bufferIn[i];
  }
  double *new_recon = new double[myLocalElements];
  ;
  const char *precision = "double";
  double nK[myVxCount * myVyCount];
  for (int iphi = 0; iphi < myPlaneCount; ++iphi) {
    for (int idx = 0; idx < myNodeCount; ++idx) {
      const double *recon_one =
          &reconData[myNodeCount * myVxCount * myVyCount * iphi +
                     myVxCount * myVyCount * idx];
      double *new_recon_one =
          &new_recon[myNodeCount * myVxCount * myVyCount * iphi +
                     myVxCount * myVyCount * idx];
      int x = 4 * (iphi * myNodeCount + idx);
      for (i = 0; i < myVxCount * myVyCount; ++i) {
        if (!strcmp(precision, "float")) {
          nK[i] =
              float(myLagranges[x]) *
                  myVolume[myVxCount * myVyCount * idx + i] +
              float(myLagranges[x + 1]) * V2[myVxCount * myVyCount * idx + i] +
              float(myLagranges[x + 2]) * V3[myVxCount * myVyCount * idx + i] +
              float(myLagranges[x + 3]) * V4[myVxCount * myVyCount * idx + i];
        } else if (!strcmp(precision, "double")) {
          nK[i] = (myLagranges[x]) * myVolume[myVxCount * myVyCount * idx + i] +
                  (myLagranges[x + 1]) * V2[myVxCount * myVyCount * idx + i] +
                  (myLagranges[x + 2]) * V3[myVxCount * myVyCount * idx + i] +
                  (myLagranges[x + 3]) * V4[myVxCount * myVyCount * idx + i];
        }
        new_recon_one[i] = recon_one[i] * exp(-nK[i]);
#if UF_DEBUG
        if (my_rank == 0) {
          printf("Node %d, post nK %f, x %d, y %d, n %d %g\n", idx, nK[i],
                 i / 37, i % 37, idx, new_recon_one[i]);
        }
#endif
      }
    }
  }
  compareQoIs(reconData, new_recon);
  return;
}

void LagrangeOptimizer::setDataFromCharBufferV1(double *&reconData,
                                                const double *bufferIn,
                                                const char *meshFile) {
  // size_t bufferOffset = getPQIndexes(bufferIn);
  // FILE* fp = fopen("PqMeshInfo.bin", "rb");
  // fread(&myNumClusters, sizeof(int), 1, fp);
  // fread(myDensityTable, sizeof(double), myNumClusters, fp);
  // fread(myUparaTable, sizeof(double), myNumClusters, fp);
  // fread(myTperpTable, sizeof(double), myNumClusters, fp);
  // fread(myRparaTable, sizeof(double), myNumClusters, fp);
  // int str_length = 0;
  // fread(&str_length, sizeof(int), 1, fp);
  // char meshFile[str_length];
  // fread(meshFile, sizeof(char), str_length, fp);
  // fclose(fp);
  readF0Params(meshFile);
#if 0
    std::vector <double> V2 (myNodeCount*myVxCount*myVyCount, 0);
    std::vector <double> V3 (myNodeCount*myVxCount*myVyCount, 0);
    std::vector <double> V4 (myNodeCount*myVxCount*myVyCount, 0);
    double nK[myVxCount*myVyCount];
    int i, j, k, l, m;
    myLocalElements = myNodeCount*myPlaneCount*myVxCount*myVyCount;
    for (i=0; i<myLocalElements; ++i) {
        if (!(reconData[i] > 0)) {
            ((double*)reconData)[i] = myEpsilon;
        }
    }
    setVolume();
    setVp();
    setMuQoi();
    setVth2();
    for (k=0; k<myNodeCount*myVxCount*myVyCount; ++k) {
        i = int(k/(myVxCount*myVyCount));
        j = int (k%myVxCount);
        l = int(k%(myVxCount*myVyCount));
        m = int(l/myVyCount);
        V2[k] = myVolume[k] * myVth[i] * myVp[j];
        V3[k] = myVolume[k] * 0.5 * myMuQoi[m] * myVth2[i] * myParticleMass;
        V4[k] = myVolume[k] * pow(myVp[j],2) * myVth2[i] * myParticleMass;
    }
    int numLagrangeParameters = myNodeCount*4;
    for (i=0; i<numLagrangeParameters; ++i) {
        myLagranges[i] = bufferIn[i];
    }
    double K[myVxCount*myVyCount];
    int iphi, idx;
    for (iphi=0; iphi<myPlaneCount; ++iphi) {
        for (idx = 0; idx<myNodeCount; ++idx) {
            double* recon_one = &reconData[myNodeCount*myVxCount*
                  myVyCount*iphi + myVxCount*myVyCount*idx];
            int x = 4*idx;
            for (i=0; i<myVxCount * myVyCount; ++i) {
                K[i] = myLagranges[x]*myVolume[myVxCount*myVyCount*idx+i]+
                       myLagranges[x+1]*V2[myVxCount*myVyCount*idx+i] +
                       myLagranges[x+2]*V3[myVxCount*myVyCount*idx+i] +
                       myLagranges[x+3]*V4[myVxCount*myVyCount*idx+i];
                ((double*)recon_one)[i] = recon_one[i] * exp(-K[i]);
            }
        }
    }
    // return reinterpret_cast<char*>(reconData);
#endif
  return;
}

size_t LagrangeOptimizer::putResult(char *&bufferOut, size_t &bufferOutOffset,
                                    const char *precision) {
  // TODO: after your algorithm is done, put the result into
  // *reinterpret_cast<double*>(bufferOut+bufferOutOffset) for your       first
  // double number *reinterpret_cast<double*>(bufferOut+bufferOutOff      set+8)
  // for your second double number and so on
  int i, intcount = 0, count = 0;
  int numObjs = myPlaneCount * myNodeCount;
  for (i = 0; i < numObjs; ++i) {
    *reinterpret_cast<int *>(bufferOut + bufferOutOffset +
                             (intcount++) * sizeof(int)) =
        myLagrangeIndexesDensity[i];
  }
  for (i = 0; i < numObjs; ++i) {
    *reinterpret_cast<int *>(bufferOut + bufferOutOffset +
                             (intcount++) * sizeof(int)) =
        myLagrangeIndexesUpara[i];
  }
  for (i = 0; i < numObjs; ++i) {
    *reinterpret_cast<int *>(bufferOut + bufferOutOffset +
                             (intcount++) * sizeof(int)) =
        myLagrangeIndexesTperp[i];
  }
  for (i = 0; i < numObjs; ++i) {
    *reinterpret_cast<int *>(bufferOut + bufferOutOffset +
                             (intcount++) * sizeof(int)) =
        myLagrangeIndexesRpara[i];
  }
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  if (my_rank == 0) {
    FILE *fp = fopen("PqMeshInfo.bin", "wb");
    // write out the PQ table and the mesh parameters
    fwrite(&myNumClusters, sizeof(int), 1, fp);
    fwrite(myDensityTable, sizeof(double), myNumClusters, fp);
    fwrite(myUparaTable, sizeof(double), myNumClusters, fp);
    fwrite(myTperpTable, sizeof(double), myNumClusters, fp);
    fwrite(myRparaTable, sizeof(double), myNumClusters, fp);
    fwrite(&myNodeCount, sizeof(int), 1, fp);
    fwrite(myGridVolume.data() + myNodeCount, sizeof(double), myNodeCount, fp);
    fwrite(myF0TEv.data() + myNodeCount, sizeof(double), myNodeCount, fp);
    fwrite(&myF0Dvp, sizeof(double), 1, fp);
    fwrite(&myF0Dsmu, sizeof(double), 1, fp);
    fwrite(&myF0Nvp, sizeof(int), 1, fp);
    fwrite(&myF0Nmu, sizeof(int), 1, fp);
    fclose(fp);
#if 0
        // for (i=0; i<myNumClusters; ++i) {
            //*reinterpret_cast<double*>(
               // bufferOut+bufferOutOffset+(count++)*sizeof(double)) =
               // myDensityTable[i];
            // fwrite(myDensityTable[i], sizeof(double), 1, fp);
        // }
        double* gridVolume = new double[myNodeCount];
        double* f0tev = new double[myNodeCount];
        int elements = 0;
        i = 0;
        for (double d : myGridVolume) {
            if (elements < myNodeCount) {
                elements++;
                continue;
            }
            // *reinterpret_cast<double*>(
                  // bufferOut+bufferOutOffset+(count++)*sizeof(double)) = d;
            gridVolume[i++] = d;
        }
        // Access f0_t_ev with an offset of nodes to get to the electrons
        elements = 0;
        for (double d : myF0TEv) {
            if (elements < myNodeCount) {
                elements++;
                continue;
            }
            *reinterpret_cast<double*>(
                  bufferOut+bufferOutOffset+(count++)*sizeof(double)) = d;
        }
        *reinterpret_cast<double*>(
            bufferOut+bufferOutOffset+(count++)*sizeof(double)) = myF0Dvp[0];
        *reinterpret_cast<double*>(
            bufferOut+bufferOutOffset+(count++)*sizeof(double)) = myF0Dsmu[0];

        int offset = count*sizeof(double) + intcount*sizeof(int);
        *reinterpret_cast<int*>(
            bufferOut+bufferOutOffset+offset) = myF0Nvp[0];
        *reinterpret_cast<int*>(
            bufferOut+bufferOutOffset+offset+sizeof(int)) = myF0Nmu[0];
        intcount += 2;
#endif
  }
  return intcount * sizeof(int);
}

#if 0
void LagrangeOptimizer::setDataFromCharBuffer(double* &reconData,
    const char* bufferIn, size_t bufferTotalSize)
{
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    int i, j, k, l, m, intcount = 0, doublecount = 0;
    for (i=0; i<myNodeCount; ++i) {
        myLagrangeIndexesDensity[i] = *(reinterpret_cast<const int*>(bufferIn+(intcount++)*sizeof(int)));
    }
    for (i=0; i<myNodeCount; ++i) {
        myLagrangeIndexesUpara[i] = (*reinterpret_cast<const int*>(bufferIn+(intcount++)*sizeof(int)));
    }
    for (i=0; i<myNodeCount; ++i) {
        myLagrangeIndexesTperp[i] = (*reinterpret_cast<const int*>(bufferIn+(intcount++)*sizeof(int)));
    }
    for (i=0; i<myNodeCount; ++i) {
        myLagrangeIndexesRpara[i] = (*reinterpret_cast<const int*>(bufferIn+(intcount++)*sizeof(int)));
    }
    FILE* fp = fopen("PqMeshInfo.bin", "rb");
    fread(&myNumClusters, sizeof(int), 1, fp);
    fread(myDensityTable, sizeof(double), myNumClusters, fp);
    fread(myUparaTable, sizeof(double), myNumClusters, fp);
    fread(myTperpTable, sizeof(double), myNumClusters, fp);
    fread(myRparaTable, sizeof(double), myNumClusters, fp);
    fread(&myNodeCount, sizeof(int), 1, fp);
    double* gridVolume = new double[myNodeCount];
    double* f0TEv = new double[myNodeCount];
    fread(gridVolume, sizeof(double), myNodeCount, fp);
    fread(f0TEv, sizeof(double), myNodeCount, fp);
    fread(&myF0Dvp, sizeof(double), 1, fp);
    fread(&myF0Dsmu, sizeof(double), 1, fp);
    fread(&myF0Nvp, sizeof(int), 1, fp);
    fread(&myF0Nmu, sizeof(int), 1, fp);
    fclose(fp);
    for (i=0; i<myNodeCount; ++i) {
        myGridVolume.push_back(0.0);
        myF0TEv.push_back(0.0);
    }
    for (i=0; i<myNodeCount; ++i) {
        myGridVolume.push_back(gridVolume[i]);
        myF0TEv.push_back(f0TEv[i]);
    }
#if 0
    double* gridVolume = new double[myNodeCount];
    double* f0TEv = new double[myNodeCount];
    int* nvp = new int [1];
    int* nmu = new int [1];
    double* dvp = new double[1];
    double* dsmu = new double[1];
    int bufferOffset = intcount*sizeof(int);
    int bcast_rank = 0;
    int local_rank = 0;
    // if ((bufferTotalSize-bufferOffset) > 0) {
        for (i=0; i<myNumClusters; ++i) {
            myDensityTable[i] = (*reinterpret_cast<const double*>(bufferIn+bufferOffset+(doublecount++)*sizeof(double)));
        }
        // local_rank = my_rank;
    // }
    // MPI_Allreduce(&local_rank, &bcast_rank, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    // MPI_Bcast(myDensityTable, myNumClusters, MPI_DOUBLE, bcast_rank, MPI_COMM_WORLD);
        for (i=0; i<myNumClusters; ++i) {
            myUparaTable[i] = (*reinterpret_cast<const double*>(bufferIn+bufferOffset+(doublecount++)*sizeof(double)));
        }
        for (i=0; i<myNumClusters; ++i) {
            myTperpTable[i] = (*reinterpret_cast<const double*>(bufferIn+bufferOffset+(doublecount++)*sizeof(double)));
        }
        for (i=0; i<myNumClusters; ++i) {
            myRparaTable[i] = (*reinterpret_cast<const double*>(bufferIn+bufferOffset+(doublecount++)*sizeof(double)));
        }
    // printf ("My rank %d density %5.3g upara %5.3g tperp %5.3g rpara %5.3g\n", my_rank, myDensityTable[0], myUparaTable[0], myTperpTable[0], myRparaTable[0]);
        bufferOffset += doublecount*sizeof(double);
        doublecount = 0;
        for (i=0; i<myNodeCount; ++i) {
            gridVolume[i] = (*reinterpret_cast<const double*>(bufferIn+bufferOffset+(doublecount++)*sizeof(double)));
        }
        bufferOffset += i*sizeof(double);
        for (i=0; i<myNodeCount; ++i) {
            f0TEv[i] = (*reinterpret_cast<const double*>(bufferIn+bufferOffset+(doublecount++)*sizeof(double)));
        }
        bufferOffset += doublecount*sizeof(double);
        dvp[0] = (*reinterpret_cast<const double*>(bufferIn+bufferOffset));
        dsmu[0] = (*reinterpret_cast<const double*>(bufferIn+bufferOffset+sizeof(double)));
        nvp[0] = (*reinterpret_cast<const int*>(bufferIn+bufferOffset+2*sizeof(double)));
        nmu[0] = (*reinterpret_cast<const int*>(bufferIn+bufferOffset+2*sizeof(double)+sizeof(int)));
    MPI_Bcast(myDensityTable, myNumClusters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(myUparaTable, myNumClusters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(myTperpTable, myNumClusters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(myRparaTable, myNumClusters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(gridVolume, myNodeCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    myGridVolume.insert(myGridVolume.begin(), gridVolume, gridVolume+myNodeCount);
    MPI_Bcast(f0TEv, myNodeCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    myF0TEv.insert(myF0TEv.begin(), f0TEv, f0TEv+myNodeCount);
    MPI_Bcast(dvp, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    myF0Dvp.push_back(dvp[0]);
    MPI_Bcast(dsmu, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    myF0Dsmu.push_back(dsmu[0]);
    MPI_Bcast(nvp, 1, MPI_INT, 0, MPI_COMM_WORLD);
    myF0Nvp.push_back(nvp[0]);
    MPI_Bcast(nmu, 1, MPI_INT, 0, MPI_COMM_WORLD);
    myF0Nmu.push_back(nmu[0]);

    setVolume();
    setVp();
    setMuQoi();
    setVth2();
    std::vector <double> V2 (myNodeCount*myVxCount*myVyCount, 0);
    std::vector <double> V3 (myNodeCount*myVxCount*myVyCount, 0);
    std::vector <double> V4 (myNodeCount*myVxCount*myVyCount, 0);
    for (k=0; k<myNodeCount*myVxCount*myVyCount; ++k) {
        i = int(k/(myVxCount*myVyCount));
        j = int (k%myVxCount);
        V2[k] = myVolume[k] * myVth[i] * myVp[j];
        V3[k] = myVolume[k] * 0.5 * myMuQoi[m] * myVth2[i] * myParticleMass;
        V4[k] = myVolume[k] * pow(myVp[j],2) * myVth2[i] * myParticleMass;
    }
    double K[myVxCount*myVyCount];
    int iphi, idx;
    for (iphi=0; iphi<myPlaneCount; ++iphi) {
        for (idx = 0; idx<myNodeCount; ++idx) {
            double* recon_one = &reconData[myNodeCount*myVxCount*
                  myVyCount*iphi + myVxCount*myVyCount*idx];
            int x = 4*idx;
            for (i=0; i<myVxCount * myVyCount; ++i) {
                K[i] = myDensityTable[myLagrangeIndexesDensity[idx]]*myVolume[myVxCount*myVyCount*idx+i]+
                       myUparaTable[myLagrangeIndexesUpara[idx]]*V2[myVxCount*myVyCount*idx+i] +
                       myTperpTable[myLagrangeIndexesTperp[idx]]*V3[myVxCount*myVyCount*idx+i] +
                       myRparaTable[myLagrangeIndexesRpara[idx]]*V4[myVxCount*myVyCount*idx+i];
                recon_one[i] = recon_one[i] * exp(-K[i]);
            }
        }
    }
#endif
}
#endif

size_t LagrangeOptimizer::putResultNoPQ(char *&bufferOut,
                                        size_t &bufferOutOffset) {
  // TODO: after your algorithm is done, put the result into
  // *reinterpret_cast<double*>(bufferOut+bufferOutOffset) for your       first
  // double number *reinterpret_cast<double*>(bufferOut+bufferOutOff      set+8)
  // for your second double number and so on

  int i, count = 0;
  for (i = 0; i < 4 * myNodeCount; ++i) {
    *reinterpret_cast<double *>(bufferOut + bufferOutOffset +
                                (count++) * sizeof(double)) = myLagranges[i];
  }
  int lagrangeCount = count;
  // Access grid_vol with an offset of nodes to get to the electrons
  int elements = 0;
  for (double d : myGridVolume) {
    if (elements < myNodeCount) {
      elements++;
      continue;
    }
    *reinterpret_cast<double *>(bufferOut + bufferOutOffset +
                                (count++) * sizeof(double)) = d;
#ifdef UF_DEBUG
    if (count == lagrangeCount + myNodeCount) {
      printf("Grid vol element %f\n", d);
    }
#endif
  }
  // Access f0_t_ev with an offset of nodes to get to the electrons
  elements = 0;
  for (double d : myF0TEv) {
    if (elements < myNodeCount) {
      elements++;
      continue;
    }
    *reinterpret_cast<double *>(bufferOut + bufferOutOffset +
                                (count++) * sizeof(double)) = d;
  }
  *reinterpret_cast<double *>(bufferOut + bufferOutOffset +
                              (count++) * sizeof(double)) = myF0Dvp[0];
  *reinterpret_cast<double *>(bufferOut + bufferOutOffset +
                              (count++) * sizeof(double)) = myF0Dsmu[0];
  int offset = count * sizeof(double);
  *reinterpret_cast<int *>(bufferOut + bufferOutOffset + offset) = myF0Nvp[0];
  *reinterpret_cast<int *>(bufferOut + bufferOutOffset + offset + sizeof(int)) =
      myF0Nmu[0];
  return count * sizeof(double) + 2 * sizeof(int);
}

void LagrangeOptimizer::setDataFromCharBuffer2(double *&reconData,
                                               const char *bufferIn,
                                               size_t bufferOffset,
                                               size_t totalSize) {
  size_t bufferSize = totalSize - bufferOffset;
  int i, j, k, l, m;
  // Assuming the Lagrange parameters are stored as double numbers
  // This will change as we add quantization
  // Find node size
  myNodeCount = int((bufferSize - 2 * sizeof(double) - 2 * sizeof(int)) /
                    (6 * sizeof(double)));
  int numLagrangeParameters = myNodeCount * 4;
  for (i = 0; i < numLagrangeParameters; ++i) {
    myLagranges[i] = (*reinterpret_cast<const double *>(
        bufferIn + bufferOffset + i * sizeof(double)));
  }
  bufferOffset += i * sizeof(double);
  for (i = 0; i < myNodeCount; ++i) {
    myGridVolume.push_back(*reinterpret_cast<const double *>(
        bufferIn + bufferOffset + i * sizeof(double)));
  }
  bufferOffset += i * sizeof(double);
  for (i = 0; i < myNodeCount; ++i) {
    myF0TEv.push_back(*reinterpret_cast<const double *>(
        bufferIn + bufferOffset + i * sizeof(double)));
  }
  bufferOffset += i * sizeof(double);
  myF0Dvp.push_back(*reinterpret_cast<const double *>(bufferIn + bufferOffset));
  myF0Dsmu.push_back(*reinterpret_cast<const double *>(bufferIn + bufferOffset +
                                                       sizeof(double)));
  myF0Nvp.push_back(*reinterpret_cast<const int *>(bufferIn + bufferOffset +
                                                   2 * sizeof(double)));
  myF0Nmu.push_back(*reinterpret_cast<const int *>(
      bufferIn + bufferOffset + 2 * sizeof(double) + sizeof(int)));
  setVolume();
  setVp();
  setMuQoi();
  setVth2();
  std::vector<double> V2(myNodeCount * myVxCount * myVyCount, 0);
  std::vector<double> V3(myNodeCount * myVxCount * myVyCount, 0);
  std::vector<double> V4(myNodeCount * myVxCount * myVyCount, 0);
  for (k = 0; k < myNodeCount * myVxCount * myVyCount; ++k) {
    i = int(k / (myVxCount * myVyCount));
    j = int(k % myVxCount);
    V2[k] = myVolume[k] * myVth[i] * myVp[j];
    V3[k] = myVolume[k] * 0.5 * myMuQoi[m] * myVth2[i] * myParticleMass;
    V4[k] = myVolume[k] * pow(myVp[j], 2) * myVth2[i] * myParticleMass;
  }
  double K[myVxCount * myVyCount];
  int iphi, idx;
  for (iphi = 0; iphi < myPlaneCount; ++iphi) {
    for (idx = 0; idx < myNodeCount; ++idx) {
      double *recon_one =
          &reconData[myNodeCount * myVxCount * myVyCount * iphi +
                     myVxCount * myVyCount * idx];
      int x = 4 * idx;
      for (i = 0; i < myVxCount * myVyCount; ++i) {
        K[i] = myLagranges[x] * myVolume[myVxCount * myVyCount * idx + i] +
               myLagranges[x + 1] * V2[myVxCount * myVyCount * idx + i] +
               myLagranges[x + 2] * V3[myVxCount * myVyCount * idx + i] +
               myLagranges[x + 3] * V4[myVxCount * myVyCount * idx + i];
        recon_one[i] = recon_one[i] * exp(-K[i]);
      }
    }
  }
}

#if 0
void LagrangeOptimizer::readCharBuffer(const char* bufferIn, size_t bufferOffset, size_t bufferSize)
{
    int i;
    // Assuming the Lagrange parameters are stored as double numbers
    // This will change as we add quantization
    // Find node size
    myNodeCount = int((bufferSize-2*sizeof(double)-2*sizeof(int))/(6*sizeof(double)));
    int numLagrangeParameters = myNodeCount*4;
    std::vector <double> lagranges;
    std::vector <double> gridVol;
    std::vector <double> f0TEv;
    double nvp, dvp, nmu, dsmu;
    for (i=0; i<numLagrangeParameters; ++i) {
        lagranges.push_back(*reinterpret_cast<const double*>(bufferIn+bufferOffset+i*sizeof(double)));
    }
    bufferOffset += i*sizeof(double);
    for (i=0; i<myNodeCount; ++i) {
        gridVol.push_back(*reinterpret_cast<const double*>(bufferIn+bufferOffset+i*sizeof(double)));
        if (i == myNodeCount-1) {
             printf("Grid vol element %f\n", *reinterpret_cast<const double*>(bufferIn+bufferOffset+i*sizeof(double)));
        }
    }
    bufferOffset += i*sizeof(double);
    for (i=0; i<myNodeCount; ++i) {
        f0TEv.push_back(*reinterpret_cast<const double*>(bufferIn+bufferOffset+i*sizeof(double)));
    }
    bufferOffset += i*sizeof(double);
    dvp = *reinterpret_cast<const double*>(bufferIn+bufferOffset);
    dsmu = *reinterpret_cast<const double*>(bufferIn+bufferOffset+sizeof(double));
    nvp = *reinterpret_cast<const int*>(bufferIn+bufferOffset+2*sizeof(double));
    nmu = *reinterpret_cast<const int*>(bufferIn+bufferOffset+2*sizeof(double)+sizeof(int));
    double error = rmseError(myLagranges, lagranges);
    printf ("Lagrange error %f\n", error);
    error = rmseError2(myGridVolume, gridVol, myNodeCount);
    printf ("Grid volume error %f\n", error);
    error = rmseError2(myF0TEv, f0TEv, myNodeCount);
    printf ("f0_T_ev error %f\n", error);
    printf ("Nvp error %f\n", myF0Nvp[0] - nvp);
    printf ("Dvp error %f\n", myF0Dvp[0] - dvp);
    printf ("Nmu error %f\n", myF0Nmu[0] - nmu);
    printf ("Dsmu error %f\n", myF0Dsmu[0] - dsmu);
}
#endif

// Get all variables from mesh file pertaining to ions and electrons
void LagrangeOptimizer::readF0Params(const std::string meshFile) {
  // Read ADIOS2 files from here
  adios2::ADIOS adios(MPI_COMM_WORLD);
  adios2::IO io = adios.DeclareIO("SubIO");
  adios2::Engine engine = io.Open(meshFile, adios2::Mode::Read);

  // Get grid_vol
  auto var = io.InquireVariable<double>("f0_grid_vol_vonly");
  std::vector<std::size_t> volumeShape = var.Shape();

  var.SetSelection(adios2::Box<adios2::Dims>({0, myNodeOffset},
                                             {volumeShape[0], myNodeCount}));
  engine.Get<double>(var, myGridVolume);

  // Get myF0Nvp
  auto var_nvp = io.InquireVariable<int>("f0_nvp");
  engine.Get<int>(var_nvp, myF0Nvp);

  // Get f0_nmu
  auto var_nmu = io.InquireVariable<int>("f0_nmu");
  engine.Get<int>(var_nmu, myF0Nmu);

  // Get f0_dvp
  auto var_dvp = io.InquireVariable<double>("f0_dvp");
  engine.Get<double>(var_dvp, myF0Dvp);

  // Get f0_dsmu
  auto var_dsmu = io.InquireVariable<double>("f0_dsmu");
  engine.Get<double>(var_dsmu, myF0Dsmu);

  // Get f0_T_ev
  auto var_ev = io.InquireVariable<double>("f0_T_ev");
  std::vector<std::size_t> evShape = var_ev.Shape();

  var_ev.SetSelection(
      adios2::Box<adios2::Dims>({0, myNodeOffset}, {evShape[0], myNodeCount}));
  engine.Get<double>(var_ev, myF0TEv);
  engine.Close();
}

void LagrangeOptimizer::writeOutput(const char *varname,
                                    std::vector<double> &tensor) {
  // Write ADIOS2 files from here
  adios2::ADIOS adios("C++");
  adios2::IO io = adios.DeclareIO("SubIO");
  adios2::Engine engine = io.Open("out.bp", adios2::Mode::Write);

  // Get grid_vol
  auto var = io.InquireVariable<double>(varname);
  engine.Put<double>(var, tensor.data());
  engine.Close();
}

void LagrangeOptimizer::setVolume(std::vector<double> &volume) {
  int vvsize = myF0Nvp[0] * 2 + 1;
  int mvsize = myF0Nmu[0] + 1;
  std::vector<double> vp_vol(vvsize, 1.0);
  vp_vol[0] = 0.5;
  vp_vol[myF0Nvp[0] * 2] = 0.5;

  std::vector<double> mu_vol(mvsize, 1.0);
  mu_vol[0] = 0.5;
  mu_vol[myF0Nmu[0]] = 0.5;

  std::vector<double> mu_vp_vol(vvsize * mvsize, 0);
  int k, i, j;
  for (k = 0; k < vvsize * mvsize; ++k) {
    i = int(k / mvsize);
    j = int(k % vvsize);
    mu_vp_vol[k] = mu_vol[i] * vp_vol[j];
  }
  int indexOffset = mySpecies == 1 ? myNodeCount : 0;
  for (k = 0; k < myNodeCount * vvsize * mvsize; ++k) {
    i = int(k / (vvsize * mvsize));
    j = int(k % (vvsize * mvsize));
    volume[k] = (myGridVolume[indexOffset + i] * mu_vp_vol[j]);
  }
  return;
}

void LagrangeOptimizer::setVolume() {
  std::vector<double> vp_vol;
  vp_vol.push_back(0.5);
  for (int ii = 1; ii < myF0Nvp[0] * 2; ++ii) {
    vp_vol.push_back(1.0);
  }
  vp_vol.push_back(0.5);

  std::vector<double> mu_vol;
  mu_vol.push_back(0.5);
  for (int ii = 1; ii < myF0Nmu[0]; ++ii) {
    mu_vol.push_back(1.0);
  }
  mu_vol.push_back(0.5);

  std::vector<double> mu_vp_vol;
  for (int ii = 0; ii < mu_vol.size(); ++ii) {
    for (int jj = 0; jj < vp_vol.size(); ++jj) {
      mu_vp_vol.push_back(mu_vol[ii] * vp_vol[jj]);
    }
  }
  int indexOffset = mySpecies == 1 ? myNodeCount : 0;
  for (int ii = 0; ii < myNodeCount; ++ii) {
    for (int jj = 0; jj < mu_vp_vol.size(); ++jj) {
      myVolume.push_back(myGridVolume[indexOffset + ii] * mu_vp_vol[jj]);
    }
  }
  return;
}

void LagrangeOptimizer::setVp(std::vector<double> &vp) {
  for (int ii = -myF0Nvp[0]; ii < myF0Nvp[0] + 1; ++ii) {
    vp[ii + myF0Nvp[0]] = (ii * myF0Dvp[0]);
  }
  return;
}

void LagrangeOptimizer::setVp() {
  for (int ii = -myF0Nvp[0]; ii < myF0Nvp[0] + 1; ++ii) {
    myVp.push_back(ii * myF0Dvp[0]);
  }
  return;
}

void LagrangeOptimizer::setMuQoi(std::vector<double> &muqoi) {
  for (int ii = 0; ii < myF0Nmu[0] + 1; ++ii) {
    muqoi[ii] = (pow(ii * myF0Dsmu[0], 2));
  }
  return;
}

void LagrangeOptimizer::setMuQoi() {
  for (int ii = 0; ii < myF0Nmu[0] + 1; ++ii) {
    myMuQoi.push_back(pow(ii * myF0Dsmu[0], 2));
  }
  return;
}

void LagrangeOptimizer::setVth2(std::vector<double> &vth,
                                std::vector<double> &vth2) {
  double value;
  int indexOffset = mySpecies == 1 ? myNodeCount : 0;
  for (int i = 0; i < myNodeCount; ++i) {
    value = myF0TEv[indexOffset + i] * mySmallElectronCharge / myParticleMass;
    vth2[i] = (value);
    vth[i] = (sqrt(value));
  }
  return;
}

void LagrangeOptimizer::setVth2() {
  double value = 0;
  int indexOffset = mySpecies == 1 ? myNodeCount : 0;
  for (int ii = 0; ii < myNodeCount; ++ii) {
    // Access f0_T_ev with an offset of myNodeCount to get to the electrons
    value = myF0TEv[indexOffset + ii] * mySmallElectronCharge / myParticleMass;
    myVth2.push_back(value);
    myVth.push_back(sqrt(value));
  }
  return;
}

void LagrangeOptimizer::compute_C_qois(
    int iphi, std::vector<double> &density, std::vector<double> &upara,
    std::vector<double> &tperp, std::vector<double> &tpara,
    std::vector<double> &n0, std::vector<double> &t0, const double *dataIn) {
  std::vector<double> den;
  std::vector<double> upar;
  std::vector<double> upar_;
  std::vector<double> tper;
  std::vector<double> en;
  std::vector<double> T_par;
  int i, j, k;
  const double *f0_f = &dataIn[iphi * myNodeCount * myVxCount * myVyCount];
  int den_index = iphi * myNodeCount;

  for (i = 0; i < myNodeCount * myVxCount * myVyCount; ++i) {
    den.push_back(f0_f[i] * myVolume[i]);
  }

  double value = 0;
  for (i = 0; i < myNodeCount; ++i) {
    value = 0;
    for (j = 0; j < myVxCount * myVyCount; ++j) {
      value += den[myVxCount * myVyCount * i + j];
    }
    density.push_back(value);
  }
  for (i = 0; i < myNodeCount; ++i) {
    for (j = 0; j < myVxCount; ++j)
      for (k = 0; k < myVyCount; ++k) {
        upar.push_back(f0_f[myVxCount * myVyCount * i + myVyCount * j + k] *
                       myVolume[myVxCount * myVyCount * i + myVyCount * j + k] *
                       myVth[i] * myVp[k]);
      }
  }
  for (i = 0; i < myNodeCount; ++i) {
    double value = 0;
    for (j = 0; j < myVxCount * myVyCount; ++j) {
      value += upar[myVxCount * myVyCount * i + j];
    }
    upara.push_back(value / density[den_index + i]);
  }
  for (i = 0; i < myNodeCount; ++i) {
    upar_.push_back(upara[den_index + i] / myVth[i]);
  }
  for (i = 0; i < myNodeCount; ++i) {
    for (j = 0; j < myVxCount; ++j)
      for (k = 0; k < myVyCount; ++k) {
        tper.push_back(f0_f[myVxCount * myVyCount * i + myVyCount * j + k] *
                       myVolume[myVxCount * myVyCount * i + myVyCount * j + k] *
                       0.5 * myMuQoi[j] * myVth2[i] * myParticleMass);
      }
  }
  for (i = 0; i < myNodeCount; ++i) {
    double value = 0;
    for (j = 0; j < myVxCount * myVyCount; ++j) {
      value += tper[myVxCount * myVyCount * i + j];
    }
    tperp.push_back(value / density[den_index + i] / mySmallElectronCharge);
    // printf ("Tperp %g, %g, %g, %g\n", value/density[den_index +
    // i]/mySmallElectronCharge, value, myParticleMass, mySmallElectronCharge);
  }
  for (i = 0; i < myNodeCount; ++i) {
    for (j = 0; j < myVyCount; ++j)
      en.push_back(0.5 * pow((myVp[j] - upar_[i]), 2));
  }
  for (i = 0; i < myNodeCount; ++i) {
    for (j = 0; j < myVxCount; ++j)
      for (k = 0; k < myVyCount; ++k)
        T_par.push_back(
            f0_f[myVxCount * myVyCount * i + myVyCount * j + k] *
            myVolume[myVxCount * myVyCount * i + myVyCount * j + k] *
            en[myVyCount * i + k] * myVth2[i] * myParticleMass);
  }
  for (i = 0; i < myNodeCount; ++i) {
    double value = 0;
    for (j = 0; j < myVxCount * myVyCount; ++j) {
      value += T_par[myVxCount * myVyCount * i + j];
    }
    tpara.push_back(2.0 * value / density[den_index + i] /
                    mySmallElectronCharge);
  }
  for (i = 0; i < myNodeCount; ++i) {
    n0.push_back(density[i]);
    t0.push_back((2.0 * tperp[i] + tpara[i]) / 3.0);
  }
  return;
}

#if 0
void LagrangeOptimizer::compute_C_qois(int iphi,
      std::vector <double> &density, std::vector <double> &upara,
      std::vector <double> &tperp, std::vector <double> &tpara,
      std::vector <double> &n0, std::vector <double> &t0,
      const double* dataIn)
{
    std::vector <double> den;
    std::vector <double> upar;
    std::vector <double> upar_;
    std::vector <double> tper;
    std::vector <double> en;
    std::vector <double> T_par;
    int i, j, k;
    const double* f0_f = &dataIn[iphi*myNodeCount*myVxCount*myVyCount];
    int den_index = iphi*myNodeCount;

    double value = 0;
    for (i=0; i<myNodeCount; ++i) {
        value = 0;
        for (k=0; k<myVxCount; ++k) {
            int offset = k*myNodeCount*myVyCount + i*myVyCount;
            for (j=offset; j < myVyCount+offset; j++) {
                value += f0_f[j] * myVolume[i*myVxCount*myVyCount + k*myVyCount + (j-offset)];
            }
        }
        density.push_back(value);
    }
    for (i=0; i<myNodeCount; ++i) {
        value = 0;
        for (k=0; k<myVxCount; ++k) {
            int offset = k*myNodeCount*myVyCount + i*myVyCount;
            for (j=offset; j < myVyCount+offset; j++) {
                value +=f0_f[j] * myVolume[i*myVxCount*myVyCount + k*myVyCount + (j-offset)] * myVth[i]*myVp[(j-offset)];
            }
        }
        upara.push_back(value/density[den_index + i]);
    }
    for (i=0; i<myNodeCount; ++i) {
        value = 0;
        for (k=0; k<myVxCount; ++k) {
            int offset = k*myNodeCount*myVyCount + i*myVyCount;
            for (j=offset; j < myVyCount+offset; j++) {
                value +=f0_f[j] * myVolume[i*myVxCount*myVyCount + k*myVyCount + (j-offset)] * 0.5 * myMuQoi[k] * myVth2[i] * myParticleMass;
            }
        }
        tperp.push_back(value/density[den_index + i]/mySmallElectronCharge);
    }
    for (i=0; i<myNodeCount; ++i) {
        upar_.push_back(upara[den_index + i]/myVth[i]);
    }
    for (i=0; i<myNodeCount; ++i) {
        for (j=0; j<myVyCount; ++j)
            en.push_back(0.5*pow((myVp[j]-upar_[i]),2));
    }
    for (i=0; i<myNodeCount; ++i) {
        value = 0;
        for (k=0; k<myVxCount; ++k) {
            int offset = k*myNodeCount*myVyCount + i*myVyCount;
            for (j=offset; j < myVyCount+offset; j++) {
                value += f0_f[j] * myVolume[i*myVxCount*myVyCount + k*myVyCount + (j-offset)] * en[myVyCount*i+(j-offset)] * myVth2[i] * myParticleMass;
            }
        }
        tpara.push_back(2.0*value/density[den_index + i]/mySmallElectronCharge);
    }
#if 0
    if (myPlaneOffset == 0 && myNodeOffset == 0) {
        FILE* fp = fopen ("density.txt", "w");
        for (i=0; i<myNodeCount; ++i) {
            fprintf(fp, "%f\n", density[i]);
        }
        fclose(fp);
        fp = fopen ("upara.txt", "w");
        for (i=0; i<myNodeCount; ++i) {
            fprintf(fp, "%f\n", upara[i]);
        }
        fclose(fp);
        fp = fopen ("tperp.txt", "w");
        for (i=0; i<myNodeCount; ++i) {
            fprintf(fp, "%f\n", tperp[i]);
        }
        fclose(fp);
        fp = fopen ("tpara.txt", "w");
        for (i=0; i<myNodeCount; ++i) {
            fprintf(fp, "%f\n", tpara[i]);
        }
        fclose(fp);
    }
#endif
    for (i=0; i<myNodeCount; ++i) {
        n0.push_back(density[i]);
        t0.push_back((2.0*tperp[i]+tpara[i])/3.0);
    }
    return;
}
#endif

#if 0
void compute_C_qois_new(int iphi,
      std::vector <double> &density, std::vector <double> &upara,
      std::vector <double> &tperp, std::vector <double> &tpara,
      std::vector <double> &n0, std::vector <double> &t0,
      const double* dataIn)
{
    std::vector <double> myden(myNodeCount, 0);
    std::vector <double> myupara(myNodeCount, 0);
    std::vector <double> mytperp(myNodeCount, 0);
    std::vector <double> upar_;
    std::vector <double> tper;
    std::vector <double> en;
    std::vector <double> T_par;
    // std::vector <double> upar_(myNodeCount, 0);
    // std::vector <double> tper(myNodeCount*myVxCount*myVyCount, 0);
    // std::vector <double> en (myNodeCount*myVxCount, 0);
    // std::vector <double> T_par(myNodeCount*myVxCount*myVyCount, 0);
    int i, j, k, l;

    const double* f0_f = &dataIn[iphi*myNodeCount*myVxCount*myVyCount];
    int den_index = iphi*myNodeCount;

#pragma omp parallel for
    for (k=0; k<myNodeCount*myVxCount*myVyCount; ++k) {
        i = int(k/(myVxCount*myVyCount));
        myden[i] += f0_f[k] * myVolume[k];
    }
    for (k=0; k<myNodeCount*myVxCount*myVyCount; ++k) {
        i = int(k/(myVxCount*myVyCount));
        j = int(k%myVyCount);
        l = int(k/myVyCount);
        myupara[i] += (f0_f[k] * myVolume[k] * myVth[i] * myVp[j])/myden[i];
        // mytperp[i] = (f0_f[k] * myVolume[k] * 0.5 * myMuQoi[l] * myVth2[i] * myParticleMass)/myden[i]/mySmallElectronCharge;
    }
    density.assign(myden.begin(), myden.end());
    upara.assign(myupara.begin(), myupara.end());
    // tperp.assign(mytperp.begin(), mytperp.end());
    for (i=0; i<myNodeCount; ++i) {
        upar_.push_back(upara[den_index + i]/myVth[i]);
    }
    for (i=0; i<myNodeCount; ++i) {
        for (j=0; j<myVxCount; ++j)
            for (k=0; k<myVyCount; ++k) {
                tper.push_back(f0_f[myVxCount*myVyCount*i + myVyCount*j + k] *
                    myVolume[myVxCount*myVyCount*i + myVyCount*j + k] * 0.5 *
                    myMuQoi[j] * myVth2[i] * myParticleMass);
            }
    }
    for (i=0; i<myNodeCount; ++i) {
        double value = 0;
        for (j=0; j<myVxCount*myVyCount; ++j) {
            value += tper[myVxCount*myVyCount*i + j];
        }
        tperp.push_back(value/density[den_index + i]/mySmallElectronCharge);
        // printf ("Tperp %g, %g, %g, %g\n", value/density[den_index + i]/mySmallElectronCharge, value, myParticleMass, mySmallElectronCharge);
    }
#if 0
    for (k=0; k<myNodeCount*myVxCount*myVyCount; ++k) {
        i = int(k/(myVxCount*myVyCount));
        j = int(k%myVyCount);
        l = int(k/myVyCount);
        myupara[i] += (f0_f[k] * myVolume[k] * myVth[i] * myVp[j])/myden[i];
        mytperp[i] = (f0_f[k] * myVolume[k] * 0.5 * myMuQoi[l] * myVth2[i] * myParticleMass)/myden[i]/mySmallElectronCharge;
    }
#endif
    for (i=0; i<myNodeCount; ++i) {
        for (j=0; j<myVxCount; ++j)
            en.push_back(0.5*pow((myVp[j]-upar_[i]),2));
    }
    for (i=0; i<myNodeCount; ++i) {
        for (j=0; j<myVxCount; ++j)
            for (k=0; k<myVyCount; ++k)
                T_par.push_back(f0_f[myVxCount*myVyCount*i + myVyCount*j + k] *
                    myVolume[myVxCount*myVyCount*i + myVyCount*j + k] *
                    en[myVxCount*i+k] * myVth2[i] * myParticleMass);
    }
    for (i=0; i<myNodeCount; ++i) {
        double value = 0;
        for (j=0; j<myVxCount*myVyCount; ++j) {
            value += T_par[myVxCount*myVyCount*i + j];
        }
        tpara.push_back(2.0*value/density[den_index + i]/mySmallElectronCharge);
    }
    for (i=0; i<myNodeCount; ++i) {
        n0.push_back(myDensity[i]);
        t0.push_back((2.0*myTperp[i]+myTpara[i])/3.0);
    }
    return;
}
#endif

bool LagrangeOptimizer::isConverged(std::vector<double> difflist, double eB,
                                    int count) {
  bool status = false;
  if (count < 2) {
    return status;
  }
  double last2Val = difflist[count - 2];
  double last1Val = difflist[count - 1];
  if (abs(last2Val - last1Val) < eB) {
    status = true;
  }
  return status;
}

#if 0
void LagrangeOptimizer::compareErrorsPD(const double* reconData, const double* bregData, int rank)
{
    double pd_b;
    size_t pd_size_b;
    double pd_min_b;
    double pd_max_b;
    double pd_a;
    size_t pd_size_a;
    double pd_min_a;
    double pd_max_a;
    double pd_error_b = rmseErrorPD(reconData, pd_b, pd_max_b, pd_min_b, pd_size_b);
    double pd_error_a = rmseErrorPD(bregData, pd_a, pd_max_a, pd_min_a, pd_size_a);
    // get total error for recon
    double pd_e_b;
    size_t pd_s_b;
    double pd_omin_b;
    double pd_omax_b;
    MPI_Allreduce(&pd_b, &pd_e_b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&pd_size_b, &pd_s_b, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&pd_min_b, &pd_omin_b, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&pd_max_b, &pd_omax_b, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    double pd_e_a;
    size_t pd_s_a;
    double pd_omin_a;
    double pd_omax_a;
    MPI_Allreduce(&pd_a, &pd_e_a, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    printf ("Rank %d Num elements %d\n", rank, pd_size_a);
    MPI_Allreduce(&pd_size_a, &pd_s_a, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&pd_min_a, &pd_omin_a, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&pd_max_a, &pd_omax_a, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
        printf ("%d Overall PD Error: %f %f\n", mySpecies, sqrt(pd_e_b/pd_s_b)/(pd_omax_b-pd_omin_b), sqrt(pd_e_a/pd_s_a)/(pd_omax_a-pd_omin_a));
        // printf ("PD Error stats: %f %d %f %f %f %d %f %f\n", pd_e_b, pd_s_b,pd_omax_b,pd_omin_b, pd_e_a,pd_s_a,pd_omax_a,pd_omin_a);
    }
}
#endif

void LagrangeOptimizer::compareErrorsPD(const double *reconData,
                                        const double *bregData, int rank) {
  double pd_b;
  double pd_size_b;
  double pd_min_b;
  double pd_max_b;
  double pd_a;
  double pd_size_a;
  double pd_min_a;
  double pd_max_a;
  double pd_error_b =
      rmseErrorPD(reconData, pd_b, pd_max_b, pd_min_b, pd_size_b);
  double pd_error_a =
      rmseErrorPD(bregData, pd_a, pd_max_a, pd_min_a, pd_size_a);
  // get total error for recon
  double pd_e_b;
  double pd_s_b;
  double pd_omin_b;
  double pd_omax_b;
  MPI_Allreduce(&pd_b, &pd_e_b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&pd_size_b, &pd_s_b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&pd_min_b, &pd_omin_b, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&pd_max_b, &pd_omax_b, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  double pd_e_a;
  double pd_s_a;
  double pd_omin_a;
  double pd_omax_a;
  MPI_Allreduce(&pd_a, &pd_e_a, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&pd_size_a, &pd_s_a, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&pd_min_a, &pd_omin_a, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&pd_max_a, &pd_omax_a, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  if (rank == 0) {
    printf("%d Overall PD Error: %f %f\n", mySpecies,
           sqrt(pd_e_b / pd_s_b) / (pd_omax_b - pd_omin_b),
           sqrt(pd_e_a / pd_s_a) / (pd_omax_a - pd_omin_a));
    // printf ("PD Error stats: %f %f %f %f %f %f %f %f\n", pd_e_b,
    // pd_s_b,pd_omax_b,pd_omin_b, pd_e_a,pd_s_a,pd_omax_a,pd_omin_a);
  }
}

void LagrangeOptimizer::compareErrorsQoI(std::vector<double> &refqoi,
                                         std::vector<double> &rqoi,
                                         std::vector<double> &bqoi,
                                         const char *qoi, int rank) {
  double pd_b;
  int pd_size_b;
  double pd_min_b;
  double pd_max_b;
  double pd_a;
  int pd_size_a;
  double pd_min_a;
  double pd_max_a;
  double pd_error_b =
      rmseError(refqoi, rqoi, pd_b, pd_max_b, pd_min_b, pd_size_b);
  double pd_error_a =
      rmseError(refqoi, bqoi, pd_a, pd_max_a, pd_min_a, pd_size_a);
  // get total error for recon
  double pd_e_b;
  int pd_s_b;
  double pd_omin_b;
  double pd_omax_b;
  MPI_Allreduce(&pd_b, &pd_e_b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&pd_size_b, &pd_s_b, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&pd_min_b, &pd_omin_b, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&pd_max_b, &pd_omax_b, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  double pd_e_a;
  int pd_s_a;
  double pd_omin_a;
  double pd_omax_a;
  MPI_Allreduce(&pd_a, &pd_e_a, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&pd_size_a, &pd_s_a, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&pd_min_a, &pd_omin_a, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&pd_max_a, &pd_omax_a, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  if (rank == 0) {
    printf("%d Overall %s Error: %f %10.5g\n", mySpecies, qoi,
           sqrt(pd_e_b / pd_s_b) / (pd_omax_b - pd_omin_b),
           sqrt(pd_e_a / pd_s_a) / (pd_omax_a - pd_omin_a));
  }
}

void LagrangeOptimizer::compareQoIs(const double *reconData,
                                    const double *bregData) {
  int iphi;
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  std::vector<double> rdensity;
  std::vector<double> rupara;
  std::vector<double> rtperp;
  std::vector<double> rtpara;
  std::vector<double> rn0;
  std::vector<double> rt0;
  for (iphi = 0; iphi < myPlaneCount; ++iphi) {
    compute_C_qois(iphi, rdensity, rupara, rtperp, rtpara, rn0, rt0, reconData);
#if 0
        if (my_rank == 0) {
            FILE* fp = fopen("PartialOrigQoI.txt", "w");
            for (int i=0; i<rdensity.size(); ++i) {
                fprintf (fp, "(%d %d) density %5.3g upara %5.3g tperp %5.3g tpara %5.3g n0 %5.3g t0 %5.3g vth %5.3g vp %5.3g\n fof %5.3g", myPlaneOffset, i, rdensity[i], rupara[i], rtperp[i], rtpara[i], rn0[i], rt0[0], myVth[i], myVp[i%39], reconData[i]);
                if (i==0) {
                    for (int j=0; j<1521; ++j) {
                        // fprintf (fp, "(%d %d) f0f %5.3g volume %5.3g\n", myPlaneOffset, j, reconData[j], myVolume[j]);
                        fprintf (fp, "(%d %d) volume %5.3g grid volume %5.3g nvp %d nmu %d\n", myPlaneOffset, j, myVolume[j], myGridVolume[j+myNodeCount], myF0Nvp[0], myF0Nmu[0]);
                    }
                }
            }
            fclose (fp);
            fp = fopen("PartialOrigF0F.txt", "w");
            for (int i=0; i<10*39*39; ++i) {
                fprintf (fp, "(%d %d) %5.3g\n", i/1521, i%1521, reconData[i]);
            }
            fclose (fp);
        }
#endif
  }
  std::vector<double> bdensity;
  std::vector<double> bupara;
  std::vector<double> btperp;
  std::vector<double> btpara;
  std::vector<double> bn0;
  std::vector<double> bt0;
  for (iphi = 0; iphi < myPlaneCount; ++iphi) {
    compute_C_qois(iphi, bdensity, bupara, btperp, btpara, bn0, bt0, bregData);
#if 0
        if (my_rank == 0) {
            FILE* fp = fopen("PartialBregQoI.txt", "w");
            for (int i=0; i<bdensity.size(); ++i) {
                fprintf (fp, "(%d %d) density %5.3g upara %5.3g tperp %5.3g tpara %5.3g n0 %5.3g t0 %5.3g vth %5.3g vp %5.3g fof %5.3g\n", myPlaneOffset, i, bdensity[i], bupara[i], btperp[i], btpara[i], bn0[i], bt0[0], myVth[i], myVp[i%39], bregData[i]);
                if (i==0) {
                    for (int j=0; j<1521; ++j) {
                        // fprintf (fp, "(%d %d) f0f %5.3g volume %5.3g\n", myPlaneOffset, j, bregData[j], myVolume[j]);
                        fprintf (fp, "(%d %d) volume %5.3g grid volume %5.3g nvp %d nmu %d\n", myPlaneOffset, j, myVolume[j], myGridVolume[j+myNodeCount], myF0Nvp[0], myF0Nmu[0]);
                    }
                }
            }
            fclose (fp);
            fp = fopen("PartialBregF0F.txt", "w");
            for (int i=0; i<10*39*39; ++i) {
                fprintf (fp, "(%d %d) %5.3g\n", i/1521, i%1521, bregData[i]);
            }
            fclose (fp);
        }
#endif
  }
  std::vector<double> refdensity;
  std::vector<double> refupara;
  std::vector<double> reftperp;
  std::vector<double> reftpara;
  std::vector<double> refn0;
  std::vector<double> reft0;
  for (iphi = 0; iphi < myPlaneCount; ++iphi) {
    compute_C_qois(iphi, refdensity, refupara, reftperp, reftpara, refn0, reft0,
                   myDataIn.data());
#if 0
        if (my_rank == 0) {
            FILE* fp = fopen("PartialRefQoI.txt", "w");
            for (int i=0; i<bdensity.size(); ++i) {
                fprintf (fp, "(%d %d) density %5.3g upara %5.3g tperp %5.3g tpara %5.3g n0 %5.3g t0 %5.3g vth %5.3g vp %5.3g fof %5.3g\n", myPlaneOffset, i, bdensity[i], bupara[i], btperp[i], btpara[i], bn0[i], bt0[0], myVth[i], myVp[i%39], bregData[i]);
                if (i==0) {
                    for (int j=0; j<1521; ++j) {
                        // fprintf (fp, "(%d %d) f0f %5.3g volume %5.3g\n", myPlaneOffset, j, bregData[j], myVolume[j]);
                        fprintf (fp, "(%d %d) volume %5.3g grid volume %5.3g nvp %d nmu %d\n", myPlaneOffset, j, myVolume[j], myGridVolume[j+myNodeCount], myF0Nvp[0], myF0Nmu[0]);
                    }
                }
            }
            fclose (fp);
            fp = fopen("PartialRefF0F.txt", "w");
            for (int i=0; i<10*39*39; ++i) {
                fprintf (fp, "(%d %d) %5.3g\n", i/1521, i%1521, bregData[i]);
            }
            fclose (fp);
        }
#endif
  }
  if (my_rank == 0) {
    std::cout << "Compare errors for reconstructed data (first number) and "
                 "post processed data (second number)"
              << std::endl;
  }
  compareErrorsPD(reconData, bregData, my_rank);
  compareErrorsQoI(refdensity, rdensity, bdensity, "density", my_rank);
  compareErrorsQoI(refupara, rupara, bupara, "upara", my_rank);
  compareErrorsQoI(reftperp, rtperp, btperp, "tperp", my_rank);
  compareErrorsQoI(reftpara, rtpara, btpara, "tpara", my_rank);
  compareErrorsQoI(refn0, rn0, bn0, "n0", my_rank);
  compareErrorsQoI(reft0, rt0, bt0, "T0", my_rank);
#if 0
    if (isnan(pd_error_a)) {
        for (int i=0; i<myLocalElements; ++i) {
            printf ("Breg data: %d %f\n", i, bregData[i]);
        }
    }
#endif
  return;
}

double LagrangeOptimizer::rmseErrorPD(const double *y, double &e, double &maxv,
                                      double &minv, double &nsize) {
  e = 0;
  maxv = -99999;
  minv = 99999;
  const double *x = myDataIn.data();
  nsize = double(myLocalElements);
  for (int i = 0; i < nsize; ++i) {
    e += pow((x[i] - y[i]), 2);
    if (x[i] < minv) {
      minv = x[i];
    }
    if (x[i] > maxv) {
      maxv = x[i];
    }
  }
  return sqrt(e / nsize) / (maxv - minv);
}

double LagrangeOptimizer::rmseError(std::vector<double> &x,
                                    std::vector<double> &y, double &e,
                                    double &maxv, double &minv, int &ysize) {
  int xsize = x.size();
  ysize = y.size();
  assert(xsize == ysize);
  e = 0;
  maxv = -99999;
  minv = 99999;
  for (int i = 0; i < xsize; ++i) {
    e += pow((x[i] - y[i]), 2);
    if (x[i] < minv) {
      minv = x[i];
    }
    if (x[i] > maxv) {
      maxv = x[i];
    }
  }
  return sqrt(e / xsize) / (maxv - minv);
}

double LagrangeOptimizer::determinant(double a[4][4], double k) {
  double s = 1, det = 0;
  double b[4][4] = {{0.0, 0.0, 0.0, 0.0},
                    {0.0, 0.0, 0.0, 0.0},
                    {0.0, 0.0, 0.0, 0.0},
                    {0.0, 0.0, 0.0, 0.0}};
  int i, j, m, n, c;
  if (k == 1) {
    return (a[0][0]);
  } else {
    det = 0;
    for (c = 0; c < k; c++) {
      m = 0;
      n = 0;
      for (i = 0; i < k; i++) {
        for (j = 0; j < k; j++) {
          b[i][j] = 0;
          if (i != 0 && j != c) {
            b[m][n] = a[i][j];
            if (n < (k - 2))
              n++;
            else {
              n = 0;
              m++;
            }
          }
        }
      }
      det = det + s * (a[0][c] * determinant(b, k - 1));
      s = -1 * s;
    }
  }

  return (det);
}

double **LagrangeOptimizer::cofactor(double num[4][4], double f) {
  double b[4][4], fac[4][4];
  int p, q, m, n, i, j;
  for (q = 0; q < f; q++) {
    for (p = 0; p < f; p++) {
      m = 0;
      n = 0;
      for (i = 0; i < f; i++) {
        for (j = 0; j < f; j++) {
          if (i != q && j != p) {
            b[m][n] = num[i][j];
            if (n < (f - 2))
              n++;
            else {
              n = 0;
              m++;
            }
          }
        }
      }
      fac[q][p] = pow(-1, q + p) * determinant(b, f - 1);
    }
  }
  return transpose(num, fac, f);
}
/*Finding transpose of matrix*/
double **LagrangeOptimizer::transpose(double num[4][4], double fac[4][4],
                                      double r) {
  int i, j;
  double b[4][4], d;
  double **inverse = new double *[4];
  inverse[0] = new double[4];
  inverse[1] = new double[4];
  inverse[2] = new double[4];
  inverse[3] = new double[4];

  for (i = 0; i < r; i++) {
    for (j = 0; j < r; j++) {
      b[i][j] = fac[j][i];
    }
  }
  d = determinant(num, r);
  for (i = 0; i < r; i++) {
    for (j = 0; j < r; j++) {
      inverse[i][j] = b[i][j] / d;
    }
  }
  return inverse;
}
