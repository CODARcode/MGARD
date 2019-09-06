#ifndef MEASURE_HPP
#define MEASURE_HPP
//!\file
//!\brief Functions to compute measures of elements.
//!
//!See §3 of [these notes][LNoGR] for more on the formulae used here.
//!
//![LNoGR]: https://people.eecs.berkeley.edu/~jrs/meshpapers/robnotes.pdf

namespace helpers {

//!Find the signed area of a parallelogram.
//!
//!\param [in] a, b, c Coordinates, of size 2, of three vertices of the
//!parallelogram, with the third lying between the first and second.
//!
//!\return Signed area of the parallelogram.
double orient_2d(
    double const * const a,
    double const * const b,
    double const * const c
);

//!Find the signed area of a parallelepiped.
//!
//!\param [in] a, b, c, d Coordinates, of size 3, of four vertices of the
//!parallelepiped, with the fourth sharing edges with each of the first three.
//!
//!\return Signed volume of the parallelepiped.
double orient_3d(
    double const * const a,
    double const * const b,
    double const * const c,
    double const * const d
);

//!Find the length of an edge.
//!
//!\param [in] p Coordinates of the vertices of the edge (six values in total,
//!grouped by vertex).
//!
//!\return Length of the edge.
double edge_measure(double const * const p);

//!Find the area of a triangle.
//!
//!\param[in] p Coordinates of the vertices of the triangle (nine values in
//!total, grouped by vertex).
//!
//!\return Area of the triangle.
double tri_measure(double const * const p);

//!Find the volume of a tetrahedron.
//!
//!\param [in] p Coordinates of the vertices of the tetrahedron (twelve values
//!in total, grouped by vertex).
//!
//!\return Volume of the tetrahedron.
double tet_measure(double const * const p);

}

#endif
