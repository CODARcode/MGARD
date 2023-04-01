#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>
#include "KmeansMPI.h"


// C++ Implementation of the Quick Sort Algorithm.
#include <iostream>
using namespace std;

int partition(double arr[], int start, int end)
{

	double pivot = arr[start];

	int count = 0;
	for (int i = start + 1; i <= end; i++) {
		if (arr[i] <= pivot)
			count++;
	}

	// Giving pivot element its correct position
	int pivotIndex = start + count;
  swap(arr[pivotIndex], arr[start]);

	// Sorting left and right parts of the pivot element
	int i = start, j = end;

	while (i < pivotIndex && j > pivotIndex) {

		while (arr[i] <= pivot) {
			i++;
		}

		while (arr[j] > pivot) {
			j--;
		}

		if (i < pivotIndex && j > pivotIndex) {
      swap(arr[i++], arr[j--]);
		}
	}

	return pivotIndex;
}

int partitionFloat(float arr[], int start, int end)
{

	float pivot = arr[start];

	int count = 0;
	for (int i = start + 1; i <= end; i++) {
		if (arr[i] <= pivot)
			count++;
	}

	// Giving pivot element its correct position
	int pivotIndex = start + count;
  swap(arr[pivotIndex], arr[start]);

	// Sorting left and right parts of the pivot element
	int i = start, j = end;

	while (i < pivotIndex && j > pivotIndex) {

		while (arr[i] <= pivot) {
			i++;
		}

		while (arr[j] > pivot) {
			j--;
		}

		if (i < pivotIndex && j > pivotIndex) {
      swap(arr[i++], arr[j--]);
		}
	}

	return pivotIndex;
}

// https://www.geeksforgeeks.org/cpp-program-for-quicksort/
void quickSort(double arr[], int start, int end)
{

	// base case
	if (start >= end)
		return;

	// partitioning the array
	int p = partition(arr, start, end);

	// Sorting the left part
	quickSort(arr, start, p - 1);

	// Sorting the right part
	quickSort(arr, p + 1, end);
}

void quickSortFloat(float arr[], int start, int end)
{

	// base case
	if (start >= end)
		return;

	// partitioning the array
	int p = partitionFloat(arr, start, end);

	// Sorting the left part
	quickSortFloat(arr, start, p - 1);

	// Sorting the right part
	quickSortFloat(arr, p + 1, end);
}

// Method to compare which one is the more close.
// We find the closest by taking the difference
// between the target and both values. It assumes
// that val2 is greater than val1 and target lies
// between these two.
int getClosest(double val1, double val2,
			double target, int index1, int index2)
{
	if (target - val1 >= val2 - target)
		return index2;
	else
		return index1;
}

// https://www.geeksforgeeks.org/find-closest-number-array/
// Returns element closest to target in arr[]
int findClosest(double arr[], int n, double target)
{
	// Corner cases
	if (target <= arr[0])
		return 0;
	if (target >= arr[n - 1])
		return n - 1;

	// Doing binary search
	int i = 0, j = n, mid = 0;
	while (i < j) {
		mid = (i + j) / 2;

		if (arr[mid] == target)
			return mid;

		/* If target is less than array element,
			then search in left */
		if (target < arr[mid]) {

			// If target is greater than previous
			// to mid, return closest of two
			if (mid > 0 && target > arr[mid - 1])
				return getClosest(arr[mid - 1],
								arr[mid], target, mid-1, mid);

			/* Repeat for left half */
			j = mid;
		}

		// If target is greater than mid
		else {
			if (mid < n - 1 && target < arr[mid + 1])
				return getClosest(arr[mid],
								arr[mid + 1], target, mid, mid+1);
			// update i
			i = mid + 1;
		}
	}

	// Only single element left after search
	return mid;
}


int findClosestFloat(float arr[], int n, float target)
{
	// Corner cases
	if (target <= arr[0])
		return 0;
	if (target >= arr[n - 1])
		return n - 1;

	// Doing binary search
	int i = 0, j = n, mid = 0;
	while (i < j) {
		mid = (i + j) / 2;

		if (arr[mid] == target)
			return mid;

		/* If target is less than array element,
			then search in left */
		if (target < arr[mid]) {

			// If target is greater than previous
			// to mid, return closest of two
			if (mid > 0 && target > arr[mid - 1])
				return getClosest(arr[mid - 1],
								arr[mid], target, mid-1, mid);

			/* Repeat for left half */
			j = mid;
		}

		// If target is greater than mid
		else {
			if (mid < n - 1 && target < arr[mid + 1])
				return getClosest(arr[mid],
								arr[mid + 1], target, mid, mid+1);
			// update i
			i = mid + 1;
		}
	}

	// Only single element left after search
	return mid;
}

/*----< mpi_kmeans() >-------------------------------------------------------*/
int mpi_kmeans(double     *objects,     /* in: [numObjs][numCoords] */
               int        numObjs,     /* no. objects */
               int        numClusters, /* no. clusters */
               float      threshold,   /* % objects change membership */
               int       *&membership,  /* out: [numObjs] */
               double    *&clusters)    /* out: [numClusters][numCoords] */
               // MPI_Comm   comm)        /* MPI communicator */
{
    int      i, j, rank, index, loop=0, total_numObjs;
    int     *newClusterSize; /* [numClusters]: no. objects assigned in each
                                new cluster */
    int     *clusterSize;    /* [numClusters]: temp buffer for Allreduce */
    float    delta;          /* % of objects change their clusters */
    float    delta_tmp;
    double  *newClusters;    /* [numClusters][numCoords] where numCords==1*/
    double  *origClusters;    /* [numClusters][numCoords] where numCords==1*/
    int _debug = 0;

    if (_debug) MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* initialize membership[] */
    for (i=0; i<numObjs; i++) membership[i] = -1;

    /* need to initialize newClusterSize and newClusters[0] to all 0 */
    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);
    clusterSize    = (int*) calloc(numClusters, sizeof(int));
    assert(clusterSize != NULL);

    // newClusters    = (float**) malloc(numClusters *            sizeof(float*));
    newClusters    = new double[numClusters];
    origClusters    = new double[numClusters];
    assert(newClusters != NULL);
    for (i=0; i<numClusters; i++) {
        newClusters[i] = 0.0;
        origClusters[i] = clusters[i];
    }

    MPI_Allreduce(&numObjs, &total_numObjs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (_debug) printf("%2d: numObjs=%d total_numObjs=%d numClusters=%d \n",rank,numObjs,total_numObjs,numClusters);

    do {
        double curT = MPI_Wtime();
        delta = 0.0;
	      quickSort(clusters, 0, numClusters-1);
        for (i=0; i<numObjs; i++) {
            /* find the array index of nestest cluster center */
	          index = findClosest(clusters, numClusters, objects[i]);

            /* if membership changes, increase delta by 1 */
            if (membership[i] != index) delta += 1.0;

            /* assign the membership to object i */
            membership[i] = index;

            /* update new cluster centers : sum of objects located within */
            newClusterSize[index]++;
            newClusters[index] += objects[i];
        }

        /* sum all data objects in newClusters */
        MPI_Allreduce(newClusters, clusters, numClusters,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(newClusterSize, clusterSize, numClusters, MPI_INT,
                      MPI_SUM, MPI_COMM_WORLD);

        /* average the sum and replace old cluster centers with newClusters */
        for (i=0; i<numClusters; i++) {
            if (clusterSize[i] > 1)
                clusters[i] /= clusterSize[i];
            newClusters[i] = 0.0;  /* set back to 0 */
            newClusterSize[i] = 0;   /* set back to 0 */
        }
        MPI_Allreduce(&delta, &delta_tmp, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        delta = delta_tmp / total_numObjs;

        if (_debug) {
            double maxTime;
            curT = MPI_Wtime() - curT;
            MPI_Reduce(&curT, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0)
                printf("%2d: loop=%d time=%f sec\n",rank,loop,curT);
        }
    } while (delta > threshold && loop++ < 5);
    if (_debug && rank == 0)
        printf("%2d: delta=%f threshold=%f loop=%d\n",rank,delta,threshold,loop);

    free(newClusters);
    free(newClusterSize);
    free(clusterSize);

    return 1;
}

/*----< kmeans() - on a single process >-------------------------------------------------------*/
int kmeans(double     *objects,     /* in: [numObjs][numCoords] */
               int        numObjs,     /* no. objects */
               int        numClusters, /* no. clusters */
               float      threshold,   /* % objects change membership */
               int       *&membership,  /* out: [numObjs] */
               double    *&clusters)    /* out: [numClusters][numCoords] */
               // MPI_Comm   comm)        /* MPI communicator */
{
    int      i, j, rank, index, loop=0, total_numObjs;
    int     *newClusterSize; /* [numClusters]: no. objects assigned in each
                                new cluster */
    int     *clusterSize;    /* [numClusters]: temp buffer for Allreduce */
    float    delta;          /* % of objects change their clusters */
    float    delta_tmp;
    double  *newClusters;    /* [numClusters][numCoords] where numCords==1*/
    double  *origClusters;    /* [numClusters][numCoords] where numCords==1*/
    int _debug = 0;

    /* initialize membership[] */
    for (i=0; i<numObjs; i++) membership[i] = -1;

    /* need to initialize newClusterSize and newClusters[0] to all 0 */
    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);
    clusterSize    = (int*) calloc(numClusters, sizeof(int));
    assert(clusterSize != NULL);

    newClusters    = new double[numClusters];
    origClusters    = new double[numClusters];
    assert(newClusters != NULL);
    for (i=0; i<numClusters; i++) {
        newClusters[i] = 0.0;
        origClusters[i] = clusters[i];
    }

    do {
        double curT = MPI_Wtime();
        delta = 0.0;
	      quickSort(clusters, 0, numClusters-1);
        for (i=0; i<numObjs; i++) {
            /* find the array index of nestest cluster center */
	          index = findClosest(clusters, numClusters, objects[i]);

            /* if membership changes, increase delta by 1 */
            if (membership[i] != index) delta += 1.0;

            /* assign the membership to object i */
            membership[i] = index;

            /* update new cluster centers : sum of objects located within */
            newClusterSize[index]++;
            newClusters[index] += objects[i];
        }

        /* average the sum and replace old cluster centers with newClusters */
        for (i=0; i<numClusters; i++) {
            if (clusterSize[i] > 1)
                clusters[i] /= clusterSize[i];
            newClusters[i] = 0.0;  /* set back to 0 */
            newClusterSize[i] = 0;   /* set back to 0 */
        }
        delta = delta / numObjs;
    } while (delta > threshold && loop++ < 1000);

    free(newClusters);
    free(newClusterSize);
    free(clusterSize);

    return 1;
}
/*----< kmeans() - on a single process >-------------------------------------------------------*/
int kmeans_float(float     *objects,     /* in: [numObjs][numCoords] */
               int        numObjs,     /* no. objects */
               int        numClusters, /* no. clusters */
               float      threshold,   /* % objects change membership */
               int       *&membership,  /* out: [numObjs] */
               float    *&clusters)    /* out: [numClusters][numCoords] */
               // MPI_Comm   comm)        /* MPI communicator */
{
    int      i, j, rank, index, loop=0, total_numObjs;
    int     *newClusterSize; /* [numClusters]: no. objects assigned in each
                                new cluster */
    int     *clusterSize;    /* [numClusters]: temp buffer for Allreduce */
    float    delta;          /* % of objects change their clusters */
    float    delta_tmp;
    float  *newClusters;    /* [numClusters][numCoords] where numCords==1*/
    float  *origClusters;    /* [numClusters][numCoords] where numCords==1*/
    int _debug = 0;

    /* initialize membership[] */
    for (i=0; i<numObjs; i++) membership[i] = -1;

    /* need to initialize newClusterSize and newClusters[0] to all 0 */
    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);
    clusterSize    = (int*) calloc(numClusters, sizeof(int));
    assert(clusterSize != NULL);

    newClusters    = new float[numClusters];
    origClusters    = new float[numClusters];
    assert(newClusters != NULL);
    for (i=0; i<numClusters; i++) {
        newClusters[i] = 0.0;
        origClusters[i] = clusters[i];
    }

    do {
        double curT = MPI_Wtime();
        delta = 0.0;
	      quickSortFloat(clusters, 0, numClusters-1);
        for (i=0; i<numObjs; i++) {
            /* find the array index of nestest cluster center */
	          index = findClosestFloat(clusters, numClusters, objects[i]);

            /* if membership changes, increase delta by 1 */
            if (membership[i] != index) delta += 1.0;

            /* assign the membership to object i */
            membership[i] = index;

            /* update new cluster centers : sum of objects located within */
            newClusterSize[index]++;
            newClusters[index] += objects[i];
        }

        /* average the sum and replace old cluster centers with newClusters */
        for (i=0; i<numClusters; i++) {
            if (clusterSize[i] > 1)
                clusters[i] /= clusterSize[i];
            newClusters[i] = 0.0;  /* set back to 0 */
            newClusterSize[i] = 0;   /* set back to 0 */
        }
        delta = delta / numObjs;
    } while (delta > threshold && loop++ < 1000);

    free(newClusters);
    free(newClusterSize);
    free(clusterSize);

    return 1;
}
