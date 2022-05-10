#ifndef HUFFMAN_HPP
#define HUFFMAN_HPP
//!\file
//!\brief Huffman trees for quantized multilevel coefficients.

namespace mgard {

void huffman_encoding(long int *quantized_data, const std::size_t n,
                      unsigned char **out_data_hit, size_t *out_data_hit_size,
                      unsigned char **out_data_miss, size_t *out_data_miss_size,
                      unsigned char **out_tree, size_t *out_tree_size);

void huffman_decoding(long int *quantized_data,
                      const std::size_t quantized_data_size,
                      unsigned char *out_data_hit, size_t out_data_hit_size,
                      unsigned char *out_data_miss, size_t out_data_miss_size,
                      unsigned char *out_tree, size_t out_tree_size);

} // namespace mgard

#endif
