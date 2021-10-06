#include "compressors.hpp"

#include <algorithm>
#include <bitset>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <queue>
#include <vector>

#ifdef MGARD_TIMING
#include <chrono>
#include <iostream>
#endif

#include <zlib.h>

#ifdef MGARD_ZSTD
#include <zstd.h>
#endif

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

bool myfunction(htree_node i, htree_node j) { return (i.cnt < j.cnt); }

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

namespace {

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

} // namespace

void decompress_memory_huffman(unsigned char *data, const std::size_t data_len,
                               long int *out_data, const std::size_t out_size) {
  unsigned char *out_data_hit = 0;
  size_t out_data_hit_size;
  unsigned char *out_data_miss = 0;
  size_t out_data_miss_size;
  unsigned char *out_tree = 0;
  size_t out_tree_size;

  unsigned char *buf = data;

  out_tree_size = *(size_t *)buf;
  buf += sizeof(size_t);

  out_data_hit_size = *(size_t *)buf;
  buf += sizeof(size_t);

  out_data_miss_size = *(size_t *)buf;
  buf += sizeof(size_t);
#if 0
std::cout << "decompress total len = " << data_len << " out_tree_size = " << out_tree_size << " out_data_hit_size = " << out_data_hit_size << " out_data_miss_size = " << out_data_miss_size << "\n";
#endif
  size_t total_huffman_size =
      out_tree_size + out_data_hit_size / 8 + 4 + out_data_miss_size;
  unsigned char *huffman_encoding_p =
      (unsigned char *)malloc(total_huffman_size);
#ifndef MGARD_ZSTD
  decompress_memory_z(buf, data_len - 3 * sizeof(size_t), huffman_encoding_p,
                      total_huffman_size);
#else
  decompress_memory_zstd(buf, data_len - 3 * sizeof(size_t), huffman_encoding_p,
                         total_huffman_size);
#endif
  out_tree = huffman_encoding_p;
  out_data_hit = huffman_encoding_p + out_tree_size;
  out_data_miss =
      huffman_encoding_p + out_tree_size + out_data_hit_size / 8 + 4;

  huffman_decoding(out_data, out_size, out_data_hit, out_data_hit_size,
                   out_data_miss, out_data_miss_size, out_tree, out_tree_size);

  free(huffman_encoding_p);
}

namespace {

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
    assert(len >= 0);

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

} // namespace

unsigned char *compress_memory_huffman(const std::vector<long int> &qv,
                                       std::size_t &outsize) {
  unsigned char *out_data_hit = 0;
  size_t out_data_hit_size;
  unsigned char *out_data_miss = 0;
  size_t out_data_miss_size;
  unsigned char *out_tree = 0;
  size_t out_tree_size;
#ifdef MGARD_TIMING
  auto huff_time1 = std::chrono::high_resolution_clock::now();
#endif
  huffman_encoding(const_cast<long int *>(qv.data()), qv.size(), &out_data_hit,
                   &out_data_hit_size, &out_data_miss, &out_data_miss_size,
                   &out_tree, &out_tree_size);
#ifdef MGARD_TIMING
  auto huff_time2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      huff_time2 - huff_time1);
  std::cout << "Huffman tree time = " << (double)duration.count() / 1000000
            << "\n";
#endif
  const size_t total_size =
      out_data_hit_size / 8 + 4 + out_data_miss_size + out_tree_size;
  unsigned char *payload = (unsigned char *)malloc(total_size);
  unsigned char *bufp = payload;

  if (out_tree_size) {
    std::memcpy(bufp, out_tree, out_tree_size);
    bufp += out_tree_size;
  }

  std::memcpy(bufp, out_data_hit, out_data_hit_size / 8 + 4);
  bufp += out_data_hit_size / 8 + 4;

  if (out_data_miss_size) {
    std::memcpy(bufp, out_data_miss, out_data_miss_size);
    bufp += out_data_miss_size;
  }

  free(out_tree);
  free(out_data_hit);
  free(out_data_miss);

  std::vector<unsigned char> out_data;
#ifndef MGARD_ZSTD
#ifdef MGARD_TIMING
  auto z_time1 = std::chrono::high_resolution_clock::now();
#endif
  compress_memory_z(payload, total_size, out_data);
#ifdef MGARD_TIMING
  auto z_time2 = std::chrono::high_resolution_clock::now();
  auto z_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(z_time2 - z_time1);
  std::cout << "ZLIB compression time = "
            << (double)z_duration.count() / 1000000 << "\n";
#endif
#else
#ifdef MGARD_TIMING
  auto zstd_time1 = std::chrono::high_resolution_clock::now();
#endif
  compress_memory_zstd(payload, total_size, out_data);
#ifdef MGARD_TIMING
  auto zstd_time2 = std::chrono::high_resolution_clock::now();
  auto zstd_duration = std::chrono::duration_cast<std::chrono::microseconds>(
      zstd_time2 - zstd_time1);
  std::cout << "ZSTD compression time = "
            << (double)zstd_duration.count() / 1000000 << "\n";
#endif
#endif
  free(payload);
  payload = 0;

  outsize = out_data.size() + 3 * sizeof(size_t);
  unsigned char *const buffer = new unsigned char[outsize];

  bufp = buffer;
  *(size_t *)bufp = out_tree_size;
  bufp += sizeof(size_t);

  *(size_t *)bufp = out_data_hit_size;
  bufp += sizeof(size_t);

  *(size_t *)bufp = out_data_miss_size;
  bufp += sizeof(size_t);

  std::copy(out_data.begin(), out_data.end(), bufp);
#if 0
std::cout << "outsize = " << outsize << " out_tree_size = " <<
out_tree_size << " out_data_hit_size = " << out_data_hit_size << " out_data_miss_size = " << out_data_miss_size << "\n";
#endif
  return buffer;
}

#ifdef MGARD_ZSTD
/*! CHECK
 * Check that the condition holds. If it doesn't print a message and die.
 */
#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "%s:%d CHECK(%s) failed: ", __FILE__, __LINE__, #cond);  \
      fprintf(stderr, "" __VA_ARGS__);                                         \
      fprintf(stderr, "\n");                                                   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

/*! CHECK_ZSTD
 * Check the zstd error code and die if an error occurred after printing a
 * message.
 */
/*! CHECK_ZSTD
 * Check the zstd error code and die if an error occurred after printing a
 * message.
 */
#define CHECK_ZSTD(fn, ...)                                                    \
  do {                                                                         \
    size_t const err = (fn);                                                   \
    CHECK(!ZSTD_isError(err), "%s", ZSTD_getErrorName(err));                   \
  } while (0)

void compress_memory_zstd(void *const in_data, const std::size_t in_data_size,
                          std::vector<std::uint8_t> &out_data) {
  size_t const cBuffSize = ZSTD_compressBound(in_data_size);
  uint8_t *cBuff = (uint8_t *)malloc(cBuffSize);

  assert(cBuff);

  size_t const cSize =
      ZSTD_compress(cBuff, cBuffSize, in_data, in_data_size, 1);
  CHECK_ZSTD(cSize);

  std::copy(cBuff, cBuff + cSize, back_inserter(out_data));

  free(cBuff);
}
#endif

void compress_memory_z(void *const in_data, const std::size_t in_data_size,
                       std::vector<std::uint8_t> &out_data) {
  std::vector<std::uint8_t> buffer;

  const std::size_t BUFSIZE = 2048 * 1024;
  std::uint8_t temp_buffer[BUFSIZE];

  z_stream strm;
  strm.zalloc = Z_NULL;
  strm.zfree = Z_NULL;
  strm.next_in = static_cast<std::uint8_t *>(in_data);
  strm.avail_in = in_data_size;
  strm.next_out = temp_buffer;
  strm.avail_out = BUFSIZE;

  deflateInit(&strm, Z_BEST_COMPRESSION);

  while (strm.avail_in != 0) {
    [[maybe_unused]] const int res = deflate(&strm, Z_NO_FLUSH);
    assert(res == Z_OK);
    if (strm.avail_out == 0) {
      buffer.insert(buffer.end(), temp_buffer, temp_buffer + BUFSIZE);
      strm.next_out = temp_buffer;
      strm.avail_out = BUFSIZE;
    }
  }

  int res = Z_OK;
  while (res == Z_OK) {
    if (strm.avail_out == 0) {
      buffer.insert(buffer.end(), temp_buffer, temp_buffer + BUFSIZE);
      strm.next_out = temp_buffer;
      strm.avail_out = BUFSIZE;
    }
    res = deflate(&strm, Z_FINISH);
  }

  assert(res == Z_STREAM_END);
  buffer.insert(buffer.end(), temp_buffer,
                temp_buffer + BUFSIZE - strm.avail_out);
  deflateEnd(&strm);

  out_data.swap(buffer);
}

void decompress_memory_z(void *const src, const std::size_t srcLen,
                         int *const dst, const std::size_t dstLen) {
  z_stream strm = {};
  strm.total_in = strm.avail_in = srcLen;
  strm.total_out = strm.avail_out = dstLen;
  strm.next_in = static_cast<Bytef *>(src);
  strm.next_out = reinterpret_cast<Bytef *>(dst);

  strm.zalloc = Z_NULL;
  strm.zfree = Z_NULL;
  strm.opaque = Z_NULL;

  [[maybe_unused]] int res;
  res = inflateInit2(&strm, (15 + 32)); // 15 window bits, and the +32 tells
                                        // zlib to to detect if using gzip or
                                        // zlib
  assert(res == Z_OK);
  res = inflate(&strm, Z_FINISH);
  assert(res == Z_STREAM_END);
  res = inflateEnd(&strm);
  assert(res == Z_OK);
}

#ifdef MGARD_ZSTD
void decompress_memory_zstd(void *const src, const std::size_t srcLen,
                            unsigned char *const dst,
                            const std::size_t dstLen) {
  size_t const dSize = ZSTD_decompress(dst, dstLen, src, srcLen);
  CHECK_ZSTD(dSize);

  /* When zstd knows the content size, it will error if it doesn't match. */
  CHECK(dstLen == dSize, "Impossible because zstd will check this condition!");
}
#endif

} // namespace mgard
