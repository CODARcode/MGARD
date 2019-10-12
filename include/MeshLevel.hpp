#ifndef MESHLEVEL_HPP
#define MESHLEVEL_HPP
//!\file
//!\brief Class for representing a level in a mesh hierarchy.

#include <cstddef>

#include "moab/EntityType.hpp"
#include "moab/Interface.hpp"

#include "utilities.hpp"

namespace mgard {

//!Triangular or tetrahedral mesh (either freestanding or part of a hierarchy).
class MeshLevel {
    public:
        //!Constructor.
        //!
        //!\param impl MOAB interface.
        //!\param nodes Nodes of the mesh.
        //!\param edges Edges of the mesh.
        //!\param elements Elements of the mesh.
        MeshLevel(
            moab::Interface &impl,
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
        //!the mesh will be extracted. Defaults to the root set.
        //!
        //!The edges will be created if they are missing. If elements are
        //!provided, the edges will be recalculated from them.
        MeshLevel(
            moab::Interface &impl,
            const moab::EntityHandle mesh_set = 0
        );

        //!Report the number of degrees of freedom of the mesh.
        std::size_t ndof() const;

        //!Find the measure of an entity of the mesh.
        //!
        //!Note that only edges, triangles, and tetrahedra are supported here.
        //!
        //!\param handle Handle of the entity.
        //!
        //!\return Measure of the entity.
        double measure(const moab::EntityHandle handle) const;

        //!Find the measure of the union of the elements containing a node.
        //!
        //!This function exists to give the mass matrix preconditioner access to
        //!`preconditioner_divisors`.
        //!
        //!\param node Handle of the node.
        double containing_elements_measure(const moab::EntityHandle node) const;

        //!Compute and store the measures of the elements of the mesh.
        //!
        //!Additionally, store for each node the sum of the measures of the
        //!elements containing that node. The reciprocals of these values are
        //the diagonal entries of the mass matrix preconditioner.
        moab::ErrorCode precompute_element_measures() const;

        //!Find the index of a mesh entity.
        //!
        //!\param handle Handle of the entity.
        //!
        //!\return Index of the entity.
        std::size_t index(const moab::EntityHandle handle) const;

        //!Find the connectivity of an edge or element.
        //!
        //!\param handle Handle of the edge or element.
        PseudoArray<const moab::EntityHandle> connectivity(
            const moab::EntityHandle handle
        ) const;

        //!MOAB interface to which the mesh is associated.
        moab::Interface &impl;

        //!MOAB entities composing the mesh.
        moab::Range entities[moab::MBMAXTYPE];

        //!Entity type of elements (currently `moab::MBTRI` or `moab::MBTET`).
        moab::EntityType element_type;

        //!Topological dimension of the mesh.
        std::size_t topological_dimension;

        //!Number of nodes associated to each element of the mesh.
        std::size_t num_nodes_per_element;

    protected:
        //!Ensure that a system has one degree of freedom per node.
        //!
        //!\param N Expected system size.
        void check_system_size(const std::size_t N) const;

    private:
        mutable std::vector<double> measures[moab::MBMAXTYPE];
        //!For each node, the sum of the measures of the elements containing
        //that node.
        mutable std::vector<double> preconditioner_divisors;

        //!Initialize `topological_dimension` and `num_nodes_per_element` from
        //!`element_type`.
        void populate_from_element_type();

        //!Fill out the edges list from the elements.
        void get_edges_from_elements();

        virtual std::size_t do_ndof() const;

        //!Compute the measures of entities of a certain type.
        //!
        //!This function fills in `measures[type]`. If `type` is `element_type`,
        //!it will also fill in `preconditioner_divisors`.
        //!
        //!\param type Type of entities to measure.
        moab::ErrorCode precompute_measures(const moab::EntityType type) const;
};

}

#endif
