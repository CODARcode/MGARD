#ifndef MESHLEVEL_HPP
#define MESHLEVEL_HPP
//!\file
//!\brief Class for representing a level in a mesh hierarchy.

#include <cstddef>

#include "moab/Core.hpp"

namespace mgard {

class MeshLevel {
    public:
        //!Constructor.
        //!
        //!\param impl MOAB interface.
        //!\param nodes Nodes of the mesh.
        //!\param edges Edges of the mesh.
        //!\param elements Elements of the mesh.
        MeshLevel(
            moab::Interface * const impl,
            const moab::Range nodes,
            const moab::Range edges,
            const moab::Range elements
        );

        //!Constructor.
        //!
        //!\overload
        //!
        //!\param impl MOAB interface.
        //!\param mesh_set Meshset from which the nodes, edges, and elements of
        //!the mesh will be extracted.
        MeshLevel(
            moab::Interface * const impl,
            const moab::EntityHandle mesh_set
        );

        //!Find the measure of an edge of element of the mesh.
        //!
        //!\param handle Handle of an edge or element of the mesh.
        //!
        //!\return Measure of the edge or element.
        double measure(const moab::EntityHandle handle) const;

        //!Compute and store the measures of the elements of the mesh.
        void precompute_element_measures();

        //!Apply the mass matrix to a vector of nodal values.
        //!
        //!\param [in] x Vector of nodal values.
        //!\param [out] b Vector of integrals against hat functions.
        void mass_matrix_matvec(double const * const x, double * const b);

        //!MOAB interface to which the mesh is associated.
        moab::Interface *impl;
        //!Nodes of the mesh.
        moab::Range nodes;
        //!Edges of the mesh.
        moab::Range edges;
        //!Elements of the mesh.
        moab::Range elements;

    private:
        std::vector<double> element_measures;

        std::size_t node_index(const moab::EntityHandle node) const;
        std::size_t edge_index(const moab::EntityHandle edge) const;
        std::size_t element_index(const moab::EntityHandle element) const;
};

}

#endif
