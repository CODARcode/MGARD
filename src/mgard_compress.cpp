#include "mgard_compress.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>
#include <queue>
#include <bitset>

#include <zlib.h>

namespace mgard {
//  int nql = 32768;
  int nql = 8;
struct htree_node {
  int q;
  size_t cnt;
  unsigned int code;
  size_t len;
  htree_node * left;
  htree_node * right;
};

struct huffman_codec {
  int q;
  unsigned int code;
  size_t len;
};

bool myfunction (htree_node i, htree_node j) { return (i.cnt < j.cnt); }

htree_node * new_htree_node(int q, size_t cnt) {
  htree_node * new_node = new htree_node;
  new_node->q = q;
  new_node->cnt = cnt;
  new_node->code = 0;
  new_node->len = 0;
  new_node->left = 0;
  new_node->right = 0;

  return new_node;
}

struct LessThanByCnt
{
  bool operator()(const htree_node * lhs, const htree_node * rhs) const
  {
    return lhs->cnt > rhs->cnt;
  }
};

void print_huffman_tree(htree_node * r) {

}

void build_codec(htree_node * root, unsigned int code, size_t len, huffman_codec * codec) {

  root->len = len;
  root->code = code;

  if (!root->left && !root->right) {
    codec[root->q].q = root->q; 
    codec[root->q].code = code; 
    codec[root->q].len = len; 
    std::cout << "code = " << std::bitset<32>(code)
    	      << " len = " << len 
	      << " count = " << root->cnt << "\n";
  }

  if (root->left) {
    build_codec(root->left, code << 1, len + 1, codec);
  }

  if (root->right) {
    build_codec(root->right, code << 1 | 0x1, len + 1, codec);
  }
}

huffman_codec * build_huffman_tree(int * quantized_data, const std::size_t n) {
  htree_node * root = 0;
  size_t * cnt = (size_t *) malloc (nql * sizeof (size_t));
  memset (cnt, 0, nql * sizeof (size_t));

  for (int i = 0; i < n; i++) {
    quantized_data[i] = quantized_data[i] + nql;
    if (quantized_data[i] >= 0 && quantized_data[i] < nql) {
      cnt[quantized_data[i]]++;
    }
  }

  std::priority_queue<htree_node *, std::vector<htree_node *>, LessThanByCnt> htree;
  for (int i = 0; i < nql; i++) {
    if (cnt[i] != 0) {
      htree_node * new_node = new_htree_node(i, cnt[i]);
      htree.push(new_node);
    }
  }

  while (htree.size() > 1) {
    htree_node * top_node1 = htree.top();
    htree.pop();
    htree_node * top_node2 = htree.top();
    htree.pop();

    htree_node * new_node = new_htree_node(-1, top_node1->cnt + top_node2->cnt);
    new_node->left = top_node1;
    new_node->right = top_node2;
    htree.push(new_node);
  }

  huffman_codec * codec = (huffman_codec *) malloc(sizeof (huffman_codec) * nql);
  memset (codec, 0, sizeof (huffman_codec) * nql);

  build_codec(htree.top(), 0 , 0, codec);

  for (int i = 0; i < nql; i++) {
    if (codec[i].len != 0) {
      std::cout << "codec: i = " << i << " len = " << codec[i].len << " code = " << std::bitset<32>(codec[i].code) << "\n";
    }
  }


  return codec;
}

void huffman_encoding(int *const quantized_data, const std::size_t n,
                      char * out_data) {
  std::cout << "huffman_encoding\n";

  huffman_codec * codec = build_huffman_tree(quantized_data, n);

  char * p = (char *)malloc (n * sizeof(int));
  memset (p, 0, n * sizeof (int));

  int start_bit = 0;
  unsigned int * cur = (unsigned int *) p;

  for (int i = 0; i < n; i++) {
    int q = quantized_data[i];

    if (q >= 0 && q < nql) {
      // for those that are within the range
      unsigned int code = codec[q].code;
      size_t len = codec[q].len;

      assert (len > 0);

      if (32 - start_bit % 32 >= len) {
        code = code << (32 - start_bit % 32);
	*(cur + start_bit / 32) = (*(cur + start_bit / 32)) | code;
	start_bit += len;
      } else {
        // current unsigned int cannot hold the code
	// copy 32 - start_bit % 32 bits to the current int
	// and copy  the rest len - (32 - start_bit % 32) to the next int
	size_t rshift = len - (32 - start_bit % 32);
	size_t lshift = 32 - rshift;
	*(cur + start_bit / 32) = (*(cur + start_bit / 32)) | (code >> rshift); 
	*(cur + start_bit / 32 + 1) = (*(cur + start_bit / 32 + 1)) | (code << lshift);
	start_bit += len;
      }
      
    } else {
      // for those that are out of the range
      unsigned int code = (unsigned int) q;
      size_t len = 32;

      assert (len > 0);

      if (32 - start_bit % 32 >= len) {
        code = code << (32 - start_bit % 32);
        *(cur + start_bit / 32) = (*(cur + start_bit / 32)) | code;
        start_bit += len;
      } else {
        // current unsigned int cannot hold the code
        // copy 32 - start_bit % 32 bits to the current int
        // and copy  the rest len - (32 - start_bit % 32) to the next int
        size_t rshift = len - (32 - start_bit % 32);
        size_t lshift = 32 - rshift;
        *(cur + start_bit / 32) = (*(cur + start_bit / 32)) | (code >> rshift);
        *(cur + start_bit / 32 + 1) = (*(cur + start_bit / 32 + 1)) | (code << lshift);
        start_bit += len;
      }
    }
  }

  std::cout << "huffman_encoding over\n";
}

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
    const int res = deflate(&strm, Z_NO_FLUSH);
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

void decompress_memory_z(void *const src, const int srcLen, int *const dst,
                         const int dstLen) {
  z_stream strm = {0};
  strm.total_in = strm.avail_in = srcLen;
  strm.total_out = strm.avail_out = dstLen;
  strm.next_in = static_cast<Bytef *>(src);
  strm.next_out = reinterpret_cast<Bytef *>(dst);

  strm.zalloc = Z_NULL;
  strm.zfree = Z_NULL;
  strm.opaque = Z_NULL;

  int res;
  res = inflateInit2(&strm, (15 + 32)); // 15 window bits, and the +32 tells
                                        // zlib to to detect if using gzip or
                                        // zlib
  assert(res == Z_OK);
  res = inflate(&strm, Z_FINISH);
  assert(res == Z_STREAM_END);
  res = inflateEnd(&strm);
  assert(res == Z_OK);
}

} // namespace mgard
