#ifndef _H_KMEANS
#define _H_KMEANS

#include <assert.h>

int mpi_kmeans(double*, int, int, float, int*&, double*&);
int kmeans(double*, int, int, float, int*&, double*&);
int kmeans_float(float*, int, int, float, int*&, float*&);

double  wtime(void);

extern int _debug;

/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__inline static
double euclid_dist_2(double coord1,   /* [numdims] */
                    double coord2)   /* [numdims] */
{
    return (coord1-coord2) * (coord1-coord2);
}

/*----< find_nearest_cluster() >---------------------------------------------*/
__inline static
int find_nearest_cluster(int     numClusters, /* no. clusters */
                         double  object,      /* [numCoords] */
                         double *clusters)    /* [numClusters] */
{
    int   index, i;
    double dist, min_dist;

    /* find the cluster id that has min distance to object */
    index    = 0;
    min_dist = euclid_dist_2(object, clusters[0]);

    for (i=1; i<numClusters; i++) {
        dist = euclid_dist_2(object, clusters[i]);
        /* no need square root */
        if (dist < min_dist) { /* find the min and its array index */
            min_dist = dist;
            index    = i;
        }
    }
    return(index);
}

#endif
