#ifndef UNIFORMEDGEFAMILIES_HPP
#define UNIFORMEDGEFAMILIES_HPP
//!\file
//!\brief Edges, their endpoints, and their midpoints associated with uniform
//!mesh hierarchies.

#include <array>
#include <iterator>

#include "moab/EntityHandle.hpp"

#include "MeshLevel.hpp"

namespace mgard {

//Forward declaration.
template <typename T>
class EdgeFamilyIterable;

//!An edge in a coarse mesh, its endpoints in the coarse mesh, and its midpoint
//!in the fine mesh.
struct EdgeFamily {
    //!Constructor.
    //!
    //!\param iterable Group of edges to which this family is associated.
    //!\param edge Edge in the coarse mesh of the iterable.
    template <typename T>
    EdgeFamily(
        const EdgeFamilyIterable<T> &iterable,
        const moab::EntityHandle edge
    );

    //!Edge being refined.
    moab::EntityHandle edge;

    //!Endpoints (in the coarse mesh) of the edge.
    std::array<moab::EntityHandle, 2> endpoints;

    //!Midpoint (in the fine mesh) of the edge.
    moab::EntityHandle midpoint;
};

//!Object allowing a user to iterate over a group of edges. `T` should be an
//!iterator that dereferences to `moab::EntityHandle` (an edge).
template <typename T>
class EdgeFamilyIterable {
    public:
        //!Constructor.
        //!
        //!\param mesh Mesh containing the edges in question.
        //!\param MESH Mesh produced by refining the first mesh.
        //!\param begin Beginning of edge range.
        //!\param end End of edge range.
        EdgeFamilyIterable(
            const MeshLevel &mesh,
            const MeshLevel &MESH,
            const T begin,
            const T end
        );

        //!Iterator over a group of edges.
        class iterator: std::iterator<std::input_iterator_tag, EdgeFamily> {
            public:
                //!Constructor.
                //!
                //!\param iterable Group of edges to which this iterator is
                //!associated.
                //!\param inner Iterator that dereferences to edges.
                iterator(
                    const EdgeFamilyIterable &iterable,
                    const T inner
                );

                //!Equality comparison.
                //!
                //!\param other Iterator to compare with.
                bool operator==(const iterator &other) const;

                //!Inequality comparison.
                //!
                //!\param other Iterator to compare with.
                bool operator!=(const iterator &other) const;

                //!Preincrement.
                iterator & operator++();

                //!Postincrement.
                iterator operator++(int);

                //!Dereference.
                EdgeFamily operator*() const;

            private:
                const EdgeFamilyIterable &iterable;
                T inner;
        };

        //!Return an iterator to the beginning of the iterable.
        iterator begin() const;

        //!Return an iterator to the end of the iterable.
        iterator end() const;


        //!Mesh containing the edges in question.
        const MeshLevel &mesh;

        //!Mesh produced by refining the first mesh.
        const MeshLevel &MESH;

    private:
        //!Beginning of edge range.
        const T begin_;

        //!End of edge range.
        const T end_;
};

}

#include "UniformEdgeFamilies.tpp"
#endif
