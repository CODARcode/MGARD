#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>

#include <queue>

#include "huffman.hpp"

namespace mgard {

const int nql = 32768 * 4;

struct htree_node {
  int q;
  size_t cnt;
  unsigned int code;
  size_t len;
  htree_node *left;
  htree_node *right;
};

struct huffman_codec {
  int q;
  unsigned int code;
  size_t len;
};

htree_node *new_htree_node(int q, size_t cnt) {
  htree_node *new_node = new htree_node;
  new_node->q = q;
  new_node->cnt = cnt;
  new_node->code = 0;
  new_node->len = 0;
  new_node->left = 0;
  new_node->right = 0;

  return new_node;
}

struct LessThanByCnt {
  bool operator()(const htree_node *lhs, const htree_node *rhs) const {
    return lhs->cnt > rhs->cnt;
  }
};

template <class T>
using my_priority_queue =
    std::priority_queue<T *, std::vector<T *>, LessThanByCnt>;

void build_codec(htree_node *root, unsigned int code, size_t len,
                 huffman_codec *codec) {

  root->len = len;
  root->code = code;

  if (!root->left && !root->right) {
    codec[root->q].q = root->q;
    codec[root->q].code = code;
    codec[root->q].len = len;
  }

  if (root->left) {
    build_codec(root->left, code << 1, len + 1, codec);
  }

  if (root->right) {
    build_codec(root->right, code << 1 | 0x1, len + 1, codec);
  }
}

my_priority_queue<htree_node> *build_tree(size_t *cnt) {
  my_priority_queue<htree_node> *phtree;
  phtree = new my_priority_queue<htree_node>;
#if 1
  for (int i = 0; i < nql; i++) {
    if (cnt[i] != 0) {
      htree_node *new_node = new_htree_node(i, cnt[i]);
      phtree->push(new_node);
    }
  }

  while (phtree->size() > 1) {
    htree_node *top_node1 = phtree->top();
    phtree->pop();
    htree_node *top_node2 = phtree->top();
    phtree->pop();

    htree_node *new_node = new_htree_node(-1, top_node1->cnt + top_node2->cnt);
    new_node->left = top_node1;
    new_node->right = top_node2;
    phtree->push(new_node);
  }
#endif
  return phtree;
}

void free_htree_node(htree_node *node) {
  if (node->left) {
    free_htree_node(node->left);
    node->left = 0;
  }

  if (node->right) {
    free_htree_node(node->right);
    node->right = 0;
  }

  delete node;
}

void free_tree(my_priority_queue<htree_node> *phtree) {
  if (phtree) {
    free_htree_node(phtree->top());

    phtree->pop();

    delete phtree;
  }
}

// Note this function will change the quantized data.
size_t *build_ft(long int *quantized_data, const std::size_t n,
                 size_t &num_outliers) {
  size_t *cnt = (size_t *)malloc(nql * sizeof(size_t));
  std::memset(cnt, 0, nql * sizeof(size_t));

  for (std::size_t i = 0; i < n; i++) {
    // Convert quantization level to positive so that counting freq can be
    // easily done. Level 0 is reserved a out-of-range flag.
    quantized_data[i] = quantized_data[i] + nql / 2;
    if (quantized_data[i] > 0 && quantized_data[i] < nql) {
      cnt[quantized_data[i]]++;
    } else {
      cnt[0]++;
    }
  }

  num_outliers = cnt[0];

  return cnt;
}

huffman_codec *build_huffman_codec(long int *quantized_data, size_t **ft,
                                   const std::size_t n, size_t &num_outliers) {
  size_t *cnt;

  cnt = build_ft(quantized_data, n, num_outliers);
  *ft = cnt;

  my_priority_queue<htree_node> *phtree = build_tree(cnt);

  huffman_codec *codec = (huffman_codec *)malloc(sizeof(huffman_codec) * nql);
  std::memset(codec, 0, sizeof(huffman_codec) * nql);

  build_codec(phtree->top(), 0, 0, codec);

  free_tree(phtree);
  phtree = 0;

  return codec;
}

void huffman_encoding(long int *quantized_data, const std::size_t n,
                      unsigned char **out_data_hit, size_t *out_data_hit_size,
                      unsigned char **out_data_miss, size_t *out_data_miss_size,
                      unsigned char **out_tree, size_t *out_tree_size) {
  size_t num_miss = 0;
  size_t *ft = 0;

  huffman_codec *codec = build_huffman_codec(quantized_data, &ft, n, num_miss);

  assert(n >= num_miss);

  /* For those miss points, we still need to maintain a flag (q = 0),
   * and therefore we need to allocate space for n numbers.
   */
  unsigned char *p_hit = (unsigned char *)malloc(n * sizeof(int));
  std::memset(p_hit, 0, n * sizeof(int));

  int *p_miss = 0;
  if (num_miss > 0) {
    p_miss = (int *)malloc(num_miss * sizeof(int));
    std::memset(p_miss, 0, num_miss * sizeof(int));
  }

  *out_data_hit = p_hit;
  *out_data_miss = (unsigned char *)p_miss;
  *out_data_hit_size = 0;
  *out_data_miss_size = 0;

  size_t start_bit = 0;
  unsigned int *cur = (unsigned int *)p_hit;
  size_t cnt_missed = 0;
  for (std::size_t i = 0; i < n; i++) {
    int q = quantized_data[i];
    unsigned int code;
    size_t len;

    if (q > 0 && q < nql) {
      // for those that are within the range
      code = codec[q].code;
      len = codec[q].len;
    } else {
      // for those that are out of the range, q is set to 0
      code = codec[0].code;
      len = codec[0].len;

      *p_miss = q;
      p_miss++;
      cnt_missed++;
    }

    // Note that if len == 0, then that means that either the data is all the
    // same number or (more likely) all data are outside the quantization
    // range. Either way, the code contains no information and is therefore 0
    // bits.

    if (32 - start_bit % 32 < len) {
      // current unsigned int cannot hold the code
      // copy 32 - start_bit % 32 bits to the current int
      // and copy  the rest len - (32 - start_bit % 32) to the next int
      size_t rshift = len - (32 - start_bit % 32);
      size_t lshift = 32 - rshift;
      *(cur + start_bit / 32) = (*(cur + start_bit / 32)) | (code >> rshift);
      *(cur + start_bit / 32 + 1) =
          (*(cur + start_bit / 32 + 1)) | (code << lshift);
      start_bit += len;
    } else if (len > 0) {
      code = code << (32 - start_bit % 32 - len);
      *(cur + start_bit / 32) = (*(cur + start_bit / 32)) | code;
      start_bit += len;
    } else {
      // Sequence is empty (everything must be the same). Do nothing.
    }
  }

  // Note: hit size is in bits, while miss size is in bytes.
  *out_data_hit_size = start_bit;
  *out_data_miss_size = num_miss * sizeof(int);

  // write frequency table to buffer
  int nonZeros = 0;
  for (int i = 0; i < nql; i++) {
    if (ft[i] > 0) {
      nonZeros++;
    }
  }

  size_t *cft = (size_t *)malloc(2 * nonZeros * sizeof(size_t));
  int off = 0;
  for (int i = 0; i < nql; i++) {
    if (ft[i] > 0) {
      cft[2 * off] = i;
      cft[2 * off + 1] = ft[i];
      off++;
    }
  }

  *out_tree = (unsigned char *)cft;
  *out_tree_size = 2 * nonZeros * sizeof(size_t);
  free(ft);
  ft = 0;

  free(codec);
  codec = 0;
}

void huffman_decoding(long int *quantized_data,
                      const std::size_t quantized_data_size,
                      unsigned char *out_data_hit, size_t out_data_hit_size,
                      unsigned char *out_data_miss, size_t out_data_miss_size,
                      unsigned char *out_tree, size_t out_tree_size) {
  size_t *cft = (size_t *)out_tree;
  int nonZeros = out_tree_size / (2 * sizeof(size_t));
  size_t *ft = (size_t *)malloc(nql * sizeof(size_t));

  std::memset(ft, 0, nql * sizeof(size_t));

  for (int j = 0; j < nonZeros; j++) {
    ft[cft[2 * j]] = cft[2 * j + 1];
  }

  my_priority_queue<htree_node> *phtree = build_tree(ft);

  unsigned int *buf = (unsigned int *)out_data_hit;

  // The out_data_miss may not be aligned. Therefore, the code
  // here makes a new buffer.
  int *miss_buf = (int *)malloc(out_data_miss_size);
  if (out_data_miss_size) {
    std::memcpy(miss_buf, out_data_miss, out_data_miss_size);
  }

  int *miss_bufp = miss_buf;

  size_t start_bit = 0;
  unsigned int mask = 0x80000000;

  long int *q = quantized_data;
  size_t i = 0;
  size_t num_missed = 0;
  while (q < (quantized_data + (quantized_data_size / sizeof(*q)))) {
    htree_node *root = phtree->top();
    assert(root);

    size_t len = 0;
    int offset = 0;
    while (root->left) {
      int flag = *(buf + start_bit / 32 + offset) & mask;
      if (!flag) {
        root = root->left;
      } else {
        root = root->right;
      }

      len++;

      mask >>= 1;
      if (!mask) {
        mask = 0x80000000;
        offset = 1;
      } else {
        //        offset = 0;
      }
    }

    if (root->q != 0) {
      *q = root->q - nql / 2;

    } else {
      *q = *miss_buf - nql / 2;

      miss_buf++;
      num_missed++;
    }

    q++;
    i++;

    start_bit += len;
  }

  assert(start_bit == out_data_hit_size);
  assert(sizeof(int) * num_missed == out_data_miss_size);

  // Avoid unused argument warning. If NDEBUG is defined, then the assert
  // becomes empty and out_data_hit_size is unused. Tell the compiler that
  // is OK and expected.
  (void)out_data_hit_size;

  free(miss_bufp);
  miss_bufp = 0;
  free_tree(phtree);
  phtree = 0;
  free(ft);
  ft = 0;
}

} // namespace mgard
