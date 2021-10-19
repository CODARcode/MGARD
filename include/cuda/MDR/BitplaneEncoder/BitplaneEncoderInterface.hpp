#ifndef _MDR_BITPLANE_ENCODER_INTERFACE_HPP
#define _MDR_BITPLANE_ENCODER_INTERFACE_HPP

#include <cassert>
namespace mgard_cuda {
namespace MDR {
    namespace concepts {
        #define UINT8_BITS 8 
        // concept of encoder which encodes T type data into bitstreams
        template<DIM D, typename T>
        class BitplaneEncoderInterface {
        public:

            virtual ~BitplaneEncoderInterface() = default;

            virtual std::vector<uint8_t *> encode(T const * data, SIZE n, int32_t exp, uint8_t num_bitplanes, std::vector<SIZE>& streams_sizes) const = 0;

            virtual T * decode(const std::vector<uint8_t const *>& streams, SIZE n, int exp, uint8_t num_bitplanes) = 0;

            virtual T * progressive_decode(const std::vector<uint8_t const *>& streams, SIZE n, int exp, uint8_t starting_bitplane, uint8_t num_bitplanes, int level) = 0;

            virtual void print() const = 0;

        };
    }
}
}

namespace mgard_m {
namespace MDR {
    namespace concepts {
        // concept of encoder which encodes T type data into bitstreams
        template<typename HandleType, mgard_cuda::DIM D, typename T_data, typename T_bitplane, typename T_error>
        class BitplaneEncoderInterface {
        public:

            virtual ~BitplaneEncoderInterface() = default;

            virtual mgard_cuda::Array<2, T_bitplane, mgard_cuda::CUDA> encode(mgard_cuda::SIZE n, mgard_cuda::SIZE num_bitplanes, int32_t exp, 
                                                            mgard_cuda::SubArray<1, T_data, mgard_cuda::CUDA> v,
                                                            mgard_cuda::SubArray<1, T_error, mgard_cuda::CUDA> level_errors,
                                                            std::vector<mgard_cuda::SIZE>& streams_sizes, int queue_idx) const = 0;

            virtual mgard_cuda::Array<1, T_data, mgard_cuda::CUDA> decode(mgard_cuda::SIZE n, mgard_cuda::SIZE num_bitplanes, int32_t exp, 
                                                        mgard_cuda::SubArray<2, T_bitplane, mgard_cuda::CUDA> encoded_bitplanes, int level,
                                                        int queue_idx) = 0;

            virtual mgard_cuda::Array<1, T_data, mgard_cuda::CUDA> progressive_decode(mgard_cuda::SIZE n, mgard_cuda::SIZE starting_bitplanes, mgard_cuda::SIZE num_bitplanes, int32_t exp, 
                                                                   mgard_cuda::SubArray<2, T_bitplane, mgard_cuda::CUDA> encoded_bitplanes,  int level,
                                                                   int queue_idx) = 0;

            virtual mgard_cuda::SIZE buffer_size(mgard_cuda::SIZE n) const = 0;

            virtual void print() const = 0;

        };
    }
}
}
#endif
