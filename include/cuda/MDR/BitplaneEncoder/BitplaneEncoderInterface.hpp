#ifndef _MDR_BITPLANE_ENCODER_INTERFACE_HPP
#define _MDR_BITPLANE_ENCODER_INTERFACE_HPP

#include <cassert>
namespace mgard_x {
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
        template<typename HandleType, mgard_x::DIM D, typename T_data, typename T_bitplane, typename T_error>
        class BitplaneEncoderInterface {
        public:

            virtual ~BitplaneEncoderInterface() = default;

            virtual mgard_x::Array<2, T_bitplane, mgard_x::CUDA> encode(mgard_x::SIZE n, mgard_x::SIZE num_bitplanes, int32_t exp, 
                                                            mgard_x::SubArray<1, T_data, mgard_x::CUDA> v,
                                                            mgard_x::SubArray<1, T_error, mgard_x::CUDA> level_errors,
                                                            std::vector<mgard_x::SIZE>& streams_sizes, int queue_idx) const = 0;

            virtual mgard_x::Array<1, T_data, mgard_x::CUDA> decode(mgard_x::SIZE n, mgard_x::SIZE num_bitplanes, int32_t exp, 
                                                        mgard_x::SubArray<2, T_bitplane, mgard_x::CUDA> encoded_bitplanes, int level,
                                                        int queue_idx) = 0;

            virtual mgard_x::Array<1, T_data, mgard_x::CUDA> progressive_decode(mgard_x::SIZE n, mgard_x::SIZE starting_bitplanes, mgard_x::SIZE num_bitplanes, int32_t exp, 
                                                                   mgard_x::SubArray<2, T_bitplane, mgard_x::CUDA> encoded_bitplanes,  int level,
                                                                   int queue_idx) = 0;

            virtual mgard_x::SIZE buffer_size(mgard_x::SIZE n) const = 0;

            virtual void print() const = 0;

        };
    }
}
}
#endif
