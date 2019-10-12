namespace mgard {

template <typename T>
EdgeFamily::EdgeFamily(
    const EdgeFamilyIterable<T> &iterable,
    const moab::EntityHandle edge
):
    edge(edge)
{
    const MeshLevel &mesh = iterable.mesh;
    const MeshLevel &MESH = iterable.MESH;

    std::array<moab::EntityHandle, 2>::iterator p = endpoints.begin();
    for (const moab::EntityHandle node : mesh.connectivity(edge)) {
        *p++ = node;
    }

    midpoint = MESH.entities[moab::MBVERTEX][mesh.ndof() + mesh.index(edge)];
}

template <typename T>
EdgeFamilyIterable<T>::EdgeFamilyIterable(
    const MeshLevel &mesh,
    const MeshLevel &MESH,
    const T begin,
    const T end
):
    mesh(mesh),
    MESH(MESH),
    begin_(begin),
    end_(end)
{
}

template <typename T>
EdgeFamilyIterable<T>::iterator::iterator(
    const EdgeFamilyIterable<T> &iterable, const T inner
):
    iterable(iterable),
    inner(inner)
{
}

//Could compare the `iterable`s here, but then we'd want to compare the meshes
//in the `iterable`s, and I don't want to define `MeshLevel::operator==`. At
//least as of this writing, edges are unique to the mesh, so it should be OK to
//just compare the iterators.
template <typename T>
bool EdgeFamilyIterable<T>::iterator::operator==(
    const EdgeFamilyIterable<T>::iterator &other
) const {
    return inner == other.inner;
}

template <typename T>
bool EdgeFamilyIterable<T>::iterator::operator!=(
    const EdgeFamilyIterable<T>::iterator &other
) const {
    return !(this->operator==(other));
}

template <typename T>
typename EdgeFamilyIterable<T>::iterator &
EdgeFamilyIterable<T>::iterator::operator++() {
    ++inner;
    return *this;
}

template <typename T>
typename EdgeFamilyIterable<T>::iterator
EdgeFamilyIterable<T>::iterator::operator++(int) {
    const EdgeFamilyIterable<T>::iterator tmp = *this;
    this->operator++();
    return tmp;
}

template <typename T>
EdgeFamily
EdgeFamilyIterable<T>::iterator::operator*() const {
    return EdgeFamily(iterable, *inner);
}

template <typename T>
typename EdgeFamilyIterable<T>::iterator
EdgeFamilyIterable<T>::begin() const {
    return EdgeFamilyIterable<T>::iterator(*this, begin_);
}

template <typename T>
typename EdgeFamilyIterable<T>::iterator
EdgeFamilyIterable<T>::end() const {
    return EdgeFamilyIterable<T>::iterator(*this, end_);
}

}
