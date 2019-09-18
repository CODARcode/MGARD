#ifndef MESHLEVEL_HPP
#define MESHLEVEL_HPP
//!\file
//!\brief Class for representing a level in a mesh hierarchy.

#include <cstddef>

#include "moab/EntityType.hpp"
#include "moab/Interface.hpp"

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

        //!Find the measure of an entity of the mesh.
        //!
        //!Note that only edges, triangles, and tetrahedra are supported here.
        //!
        //!\param handle Handle of the entity.
        //!
        //!\return Measure of the entity.
        double measure(const moab::EntityHandle handle);

        //!Compute and store the measures of the elements of the mesh.
        //!
        //!Additionally, store for each node the sum of the measures of the
        //!elements containing that node. The reciprocals of these values are
        //the diagonal entries of the mass matrix preconditioner.
        moab::ErrorCode precompute_element_measures();

        //!Find the index of a mesh entity.
        //!
        //!\param handle Handle of the entity.
        //!
        //!\return Index of the entity.
        std::size_t index(const moab::EntityHandle handle) const;

        //!Apply the mass matrix to a vector of nodal values.
        //!
        //!\param [in] v Vector of nodal values.
        //!\param [out] b Vector of integrals against hat functions.
        void mass_matrix_matvec(double const * const v, double * const b);

        //!Apply the mass matrix to a vector of nodal values.
        //!
        //!\overload
        //!
        //!\param [in] N Size of the system.
        //!\param [in] v Vector of nodal values.
        //!\param [out] b Vector of integrals against hat functions.
        void mass_matrix_matvec(
            const std::size_t N, double const * const v, double * const b
        );

        //!MOAB interface to which the mesh is associated.
        moab::Interface *impl;

        //MOAB entities composing the mesh.
        moab::Range entities[moab::MBMAXTYPE];

        //Entity type of elements (currently `moab::MBTRI` or `moab::MBTET`).
        moab::EntityType element_type;

        //Topological dimension of the mesh.
        std::size_t topological_dimension;

        //Number of nodes associated to each element of the mesh.
        std::size_t num_nodes_per_element;

    protected:
        //!Ensure that a system has one degree of freedom per node.
        //!
        //!\param N Expected system size.
        void check_system_size(const std::size_t N) const;

    private:
        std::vector<double> measures[moab::MBMAXTYPE];
        //!For each node, the sum of the measures of the elements containing
        //that node.
        std::vector<double> preconditioner_divisors;

        void populate_from_element_type();

        //!Compute the measures of entities of a certain type.
        //!
        //!This function fills in `measures[type]`. If `type` is `element_type`,
        //!it will also fill in `preconditioner_divisors`.
        //!
        //!\param type Type of entities to measure.
        moab::ErrorCode precompute_measures(const moab::EntityType type);
};

}

#endif
