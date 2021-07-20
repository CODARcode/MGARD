#ifndef _MDR_GROUPED_BP_ENCODER_HPP
#define _MDR_GROUPED_BP_ENCODER_HPP

#include "BitplaneEncoderInterface.hpp"

namespace MDR {
    // general bitplane encoder that encodes data by block using T_stream type buffer
    template<class T_data, class T_stream>
    class GroupedBPEncoder : public concepts::BitplaneEncoderInterface<T_data> {
    public:
        GroupedBPEncoder(){
            static_assert(std::is_floating_point<T_data>::value, "GeneralBPEncoder: input data must be floating points.");
            static_assert(!std::is_same<T_data, long double>::value, "GeneralBPEncoder: long double is not supported.");
            static_assert(std::is_unsigned<T_stream>::value, "GroupedBPBlockEncoder: streams must be unsigned integers.");
            static_assert(std::is_integral<T_stream>::value, "GroupedBPBlockEncoder: streams must be unsigned integers.");
        }

        std::vector<uint8_t *> encode(T_data const * data, int32_t n, int32_t exp, uint8_t num_bitplanes, std::vector<uint32_t>& stream_sizes) const {
            assert(num_bitplanes > 0);
            // determine block size based on bitplane integer type
            uint32_t block_size = block_size_based_on_bitplane_int_type<T_stream>();
            std::vector<uint8_t> starting_bitplanes = std::vector<uint8_t>((n - 1)/block_size + 1, 0);
            stream_sizes = std::vector<uint32_t>(num_bitplanes, 0);
            // define fixed point type
            using T_fp = typename std::conditional<std::is_same<T_data, double>::value, uint64_t, uint32_t>::type;
            std::vector<uint8_t *> streams;
            for(int i=0; i<num_bitplanes; i++){
                streams.push_back((uint8_t *) malloc(2 * n / UINT8_BITS + sizeof(T_stream)));
            }
            std::vector<T_fp> int_data_buffer(block_size, 0);
            std::vector<T_stream *> streams_pos(streams.size());
            for(int i=0; i<streams.size(); i++){
                streams_pos[i] = reinterpret_cast<T_stream*>(streams[i]);
            }
            T_data const * data_pos = data;
            int block_id=0;
            for(int i=0; i<n - block_size; i+=block_size){
                T_stream sign_bitplane = 0;
                for(int j=0; j<block_size; j++){
                    T_data cur_data = *(data_pos++);
                    T_data shifted_data = ldexp(cur_data, num_bitplanes - exp);
                    int64_t fix_point = (int64_t) shifted_data;
                    T_stream sign = cur_data < 0;
                    int_data_buffer[j] = sign ? -fix_point : +fix_point;
                    sign_bitplane += sign << j;
                }
                starting_bitplanes[block_id ++] = encode_block(int_data_buffer.data(), block_size, num_bitplanes, sign_bitplane, streams_pos);
            }
            // leftover
            {
                int rest_size = n - block_size * block_id;
                T_stream sign_bitplane = 0;
                for(int j=0; j<rest_size; j++){
                    T_data cur_data = *(data_pos++);
                    T_data shifted_data = ldexp(cur_data, num_bitplanes - exp);
                    int64_t fix_point = (int64_t) shifted_data;
                    T_stream sign = cur_data < 0;
                    int_data_buffer[j] = sign ? -fix_point : +fix_point;
                    sign_bitplane += sign << j;
                }
                starting_bitplanes[block_id ++] = encode_block(int_data_buffer.data(), rest_size, num_bitplanes, sign_bitplane, streams_pos);
            }
            for(int i=0; i<num_bitplanes; i++){
                stream_sizes[i] = reinterpret_cast<uint8_t*>(streams_pos[i]) - streams[i];
            }
            // merge starting_bitplane with the first bitplane
            uint32_t merged_size = 0;
            uint8_t * merged = merge_arrays(reinterpret_cast<uint8_t const*>(starting_bitplanes.data()), starting_bitplanes.size() * sizeof(uint8_t), reinterpret_cast<uint8_t*>(streams[0]), stream_sizes[0], merged_size);
            free(streams[0]);
            streams[0] = merged;
            stream_sizes[0] = merged_size;
            return streams;
        }

        // only differs in error collection
        std::vector<uint8_t *> encode(T_data const * data, int32_t n, int32_t exp, uint8_t num_bitplanes, std::vector<uint32_t>& stream_sizes, std::vector<double>& level_errors) const {
            assert(num_bitplanes > 0);
            // determine block size based on bitplane integer type
            uint32_t block_size = block_size_based_on_bitplane_int_type<T_stream>();
            std::vector<uint8_t> starting_bitplanes = std::vector<uint8_t>((n - 1)/block_size + 1, 0);
            stream_sizes = std::vector<uint32_t>(num_bitplanes, 0);
            // define fixed point type
            using T_fp = typename std::conditional<std::is_same<T_data, double>::value, uint64_t, uint32_t>::type;
            std::vector<uint8_t *> streams;
            for(int i=0; i<num_bitplanes; i++){
                streams.push_back((uint8_t *) malloc(2 * n / UINT8_BITS + sizeof(T_stream)));
            }
            std::vector<T_fp> int_data_buffer(block_size, 0);
            std::vector<T_stream *> streams_pos(streams.size());
            for(int i=0; i<streams.size(); i++){
                streams_pos[i] = reinterpret_cast<T_stream*>(streams[i]);
            }
            // init level errors
            level_errors.clear();
            level_errors.resize(num_bitplanes + 1);
            for(int i=0; i<level_errors.size(); i++){
                level_errors[i] = 0;
            }
            T_data const * data_pos = data;
            int block_id=0;
            for(int i=0; i<n - block_size; i+=block_size){
                T_stream sign_bitplane = 0;
                for(int j=0; j<block_size; j++){
                    T_data cur_data = *(data_pos++);
                    T_data shifted_data = ldexp(cur_data, num_bitplanes - exp);
                    // compute level errors
                    collect_level_errors(level_errors, fabs(shifted_data), num_bitplanes);
                    int64_t fix_point = (int64_t) shifted_data;
                    T_stream sign = cur_data < 0;
                    int_data_buffer[j] = sign ? -fix_point : +fix_point;
                    sign_bitplane += sign << j;
                }
                starting_bitplanes[block_id ++] = encode_block(int_data_buffer.data(), block_size, num_bitplanes, sign_bitplane, streams_pos);
            }
            // leftover
            {
                int rest_size = n - block_size * block_id;
                T_stream sign_bitplane = 0;
                for(int j=0; j<rest_size; j++){
                    T_data cur_data = *(data_pos++);
                    T_data shifted_data = ldexp(cur_data, num_bitplanes - exp);
                    // compute level errors
                    collect_level_errors(level_errors, fabs(shifted_data), num_bitplanes);
                    int64_t fix_point = (int64_t) shifted_data;
                    T_stream sign = cur_data < 0;
                    int_data_buffer[j] = sign ? -fix_point : +fix_point;
                    sign_bitplane += sign << j;
                }
                starting_bitplanes[block_id ++] = encode_block(int_data_buffer.data(), rest_size, num_bitplanes, sign_bitplane, streams_pos);
            }
            for(int i=0; i<num_bitplanes; i++){
                stream_sizes[i] = reinterpret_cast<uint8_t*>(streams_pos[i]) - streams[i];
            }
            // merge starting_bitplane with the first bitplane
            uint32_t merged_size = 0;
            uint8_t * merged = merge_arrays(reinterpret_cast<uint8_t const*>(starting_bitplanes.data()), starting_bitplanes.size() * sizeof(uint8_t), reinterpret_cast<uint8_t*>(streams[0]), stream_sizes[0], merged_size);
            free(streams[0]);
            streams[0] = merged;
            stream_sizes[0] = merged_size;
            // translate level errors
            for(int i=0; i<level_errors.size(); i++){
                level_errors[i] = ldexp(level_errors[i], 2*(- num_bitplanes + exp));
            }
            return streams;
        }

        T_data * decode(const std::vector<uint8_t const *>& streams, int32_t n, int exp, uint8_t num_bitplanes) {
            uint32_t block_size = block_size_based_on_bitplane_int_type<T_stream>();
            // define fixed point type
            using T_fp = typename std::conditional<std::is_same<T_data, double>::value, uint64_t, uint32_t>::type;
            T_data * data = (T_data *) malloc(n * sizeof(T_data));
            if(num_bitplanes == 0){
                memset(data, 0, n * sizeof(T_data));
                return data;
            }
            std::vector<T_stream const *> streams_pos(streams.size());
            for(int i=0; i<streams.size(); i++){
                streams_pos[i] = reinterpret_cast<T_stream const *>(streams[i]);
            }
            // deinterleave the first bitplane
            uint32_t recording_bitplane_size = *reinterpret_cast<int32_t const*>(streams_pos[0]);
            uint8_t const * recording_bitplanes = reinterpret_cast<uint8_t const*>(streams_pos[0]) + sizeof(uint32_t);
            streams_pos[0] = reinterpret_cast<T_stream const *>(recording_bitplanes + recording_bitplane_size);

            std::vector<T_fp> int_data_buffer(block_size, 0);
            // decode
            T_data * data_pos = data;
            int block_id = 0;
            for(int i=0; i<n - block_size; i+=block_size){
                uint8_t recording_bitplane = recording_bitplanes[block_id ++];
                if(recording_bitplane < num_bitplanes){
                    memset(int_data_buffer.data(), 0, block_size * sizeof(T_fp));
                    T_stream sign_bitplane = *(streams_pos[recording_bitplane] ++);
                    decode_block(streams_pos, block_size, recording_bitplane, num_bitplanes - recording_bitplane, int_data_buffer.data());
                    for(int j=0; j<block_size; j++, sign_bitplane >>= 1){
                        T_data cur_data = ldexp((T_data)int_data_buffer[j], - num_bitplanes + exp);
                        *(data_pos++) = (sign_bitplane & 1u) ? -cur_data : cur_data;
                    }
                }
                else{
                    for(int j=0; j<block_size; j++){
                        *(data_pos ++) = 0;
                    }
                }
            }
            // leftover
            {
                int rest_size = n - block_size * block_id;
                int recording_bitplane = recording_bitplanes[block_id];
                T_stream sign_bitplane = 0;
                if(recording_bitplane < num_bitplanes){
                    memset(int_data_buffer.data(), 0, block_size * sizeof(T_fp));
                    sign_bitplane = *(streams_pos[recording_bitplane] ++);
                    decode_block(streams_pos, block_size, recording_bitplane, num_bitplanes - recording_bitplane, int_data_buffer.data());
                    for(int j=0; j<rest_size; j++, sign_bitplane >>= 1){
                        T_data cur_data = ldexp((T_data)int_data_buffer[j], - num_bitplanes + exp);
                        *(data_pos++) = (sign_bitplane & 1u) ? -cur_data : cur_data;
                    }
                }
                else{
                    for(int j=0; j<block_size; j++){
                        *(data_pos ++) = 0;
                    }
                }
            }
            return data;
        }

        // decode the data and record necessary information for progressiveness
        T_data * progressive_decode(const std::vector<uint8_t const *>& streams, int32_t n, int exp, uint8_t starting_bitplane, uint8_t num_bitplanes, int level) {
            uint32_t block_size = block_size_based_on_bitplane_int_type<T_stream>();
            // define fixed point type
            using T_fp = typename std::conditional<std::is_same<T_data, double>::value, uint64_t, uint32_t>::type;
            T_data * data = (T_data *) malloc(n * sizeof(T_data));
            if(num_bitplanes == 0){
                memset(data, 0, n * sizeof(T_data));
                return data;
            }
            std::vector<T_stream const *> streams_pos(streams.size());
            for(int i=0; i<streams.size(); i++){
                streams_pos[i] = reinterpret_cast<T_stream const *>(streams[i]);
            }
            if(level_recording_bitplanes.size() == level){
                // deinterleave the first bitplane
                uint32_t recording_bitplane_size = *reinterpret_cast<int32_t const*>(streams_pos[0]);
                uint8_t const * recording_bitplanes_pos = reinterpret_cast<uint8_t const*>(streams_pos[0]) + sizeof(uint32_t);
                auto recording_bitplanes = std::vector<uint8_t>(recording_bitplanes_pos, recording_bitplanes_pos + recording_bitplane_size);
                level_recording_bitplanes.push_back(recording_bitplanes);
                streams_pos[0] = reinterpret_cast<T_stream const *>(recording_bitplanes_pos + recording_bitplane_size);
            }

            std::vector<T_fp> int_data_buffer(block_size, 0);
            if(level_signs.size() == level){
                level_signs.push_back(std::vector<bool>(n, false));
            }
            const std::vector<uint8_t>& recording_bitplanes = level_recording_bitplanes[level];
            std::vector<bool>& signs = level_signs[level];
            const uint8_t ending_bitplane = starting_bitplane + num_bitplanes;
            // decode
            T_data * data_pos = data;
            int block_id = 0;
            for(int i=0; i<n - block_size; i+=block_size){
                uint8_t recording_bitplane = recording_bitplanes[block_id ++];
                if(recording_bitplane < ending_bitplane){
                    memset(int_data_buffer.data(), 0, block_size * sizeof(T_fp));
                    if(recording_bitplane >= starting_bitplane){
                        // have not recorded signs for this block
                        T_stream sign_bitplane = *(streams_pos[recording_bitplane - starting_bitplane] ++);
                        for(int j=0; j<block_size; j++, sign_bitplane >>= 1){
                            signs[i + j] = sign_bitplane & 1u;
                        }
                        decode_block(streams_pos, block_size, recording_bitplane - starting_bitplane, ending_bitplane - recording_bitplane, int_data_buffer.data());
                    }                    
                    else{
                        decode_block(streams_pos, block_size, 0, num_bitplanes, int_data_buffer.data());                    
                    }
                    for(int j=0; j<block_size; j++){
                        T_data cur_data = ldexp((T_data)int_data_buffer[j], - ending_bitplane + exp);
                        *(data_pos++) = signs[i + j] ? -cur_data : cur_data;
                    }
                }
                else{
                    for(int j=0; j<block_size; j++){
                        *(data_pos ++) = 0;
                    }
                }
            }
            // leftover
            {
                int rest_size = n - block_size * block_id;
                uint8_t recording_bitplane = recording_bitplanes[block_id];
                if(recording_bitplane < ending_bitplane){
                    memset(int_data_buffer.data(), 0, block_size * sizeof(T_fp));
                    if(recording_bitplane >= starting_bitplane){
                        // have not recorded signs for this block
                        T_stream sign_bitplane = *(streams_pos[recording_bitplane - starting_bitplane] ++);
                        for(int j=0; j<rest_size; j++, sign_bitplane >>= 1){
                            signs[block_size * block_id + j] = sign_bitplane & 1u;
                        }
                        decode_block(streams_pos, rest_size, recording_bitplane - starting_bitplane, ending_bitplane - recording_bitplane, int_data_buffer.data());
                    }
                    else{
                        decode_block(streams_pos, rest_size, 0, num_bitplanes, int_data_buffer.data());                    
                    }
                    for(int j=0; j<rest_size; j++){
                        T_data cur_data = ldexp((T_data)int_data_buffer[j], - ending_bitplane + exp);
                        *(data_pos++) = signs[block_size * block_id + j] ? -cur_data : cur_data;
                    }
                }
                else{
                    for(int j=0; j<rest_size; j++){
                        *(data_pos ++) = 0;
                    }
                }
            }
            return data;
        }

        void print() const {
            std::cout << "Grouped bitplane encoder" << std::endl;
        }
    private:
        template<class T>
        uint32_t block_size_based_on_bitplane_int_type() const {
            uint32_t block_size = 0;
            if(std::is_same<T, uint64_t>::value){
                block_size = 64;
            }
            else if(std::is_same<T, uint32_t>::value){
                block_size = 32;
            }
            else if(std::is_same<T, uint16_t>::value){
                block_size = 16;
            }
            else if(std::is_same<T, uint8_t>::value){
                block_size = 8;
            }
            else{
                std::cerr << "Integer type not supported." << std::endl;
                exit(0);
            }
            return block_size;
        }
        inline void collect_level_errors(std::vector<double>& level_errors, float data, int num_bitplanes) const {
            uint32_t fp_data = (uint32_t) data;
            double mantissa = data - (uint32_t) data;
            level_errors[num_bitplanes] += mantissa * mantissa;
            for(int k=1; k<num_bitplanes; k++){
                uint32_t mask = (1 << k) - 1;
                double diff = (double) (fp_data & mask) + mantissa;
                level_errors[num_bitplanes - k] += diff * diff;
            }
            double diff = fp_data + mantissa;
            level_errors[0] += data * data;
        }

        template <class T_int>
        inline uint8_t encode_block(T_int const * data, size_t n, uint8_t num_bitplanes, T_stream sign, std::vector<T_stream *>& streams_pos) const {
            bool recorded = false;
            uint8_t recording_bitplane = num_bitplanes;
            for(int k=num_bitplanes - 1; k>=0; k--){
                T_stream bitplane_value = 0;
                T_stream bitplane_index = num_bitplanes - 1 - k;
                for (int i=0; i<n; i++){
                    bitplane_value += (T_stream)((data[i] >> k) & 1u) << i;
                }
                if(bitplane_value || recorded){
                    if(!recorded){
                        recorded = true;
                        recording_bitplane = bitplane_index;
                        *(streams_pos[bitplane_index] ++) = sign;
                    }
                    *(streams_pos[bitplane_index] ++) = bitplane_value;
                }
            }
            return recording_bitplane;
        }

        template <class T_int>
        inline void decode_block(std::vector<T_stream const *>& streams_pos, size_t n, uint8_t recording_bitplane, uint8_t num_bitplanes, T_int * data) const {
            for(int k=num_bitplanes - 1; k>=0; k--){
                T_stream bitplane_index = recording_bitplane + num_bitplanes - 1 - k;
                T_stream bitplane_value = *(streams_pos[bitplane_index] ++);
                for (int i=0; i<n; i++){
                    data[i] += ((bitplane_value >> i) & 1u) << k;
                }
            }
        }

        uint8_t * merge_arrays(uint8_t const * array1, uint32_t size1, uint8_t const * array2, uint32_t size2, uint32_t& merged_size) const {
            merged_size = sizeof(uint32_t) + size1 + size2;
            uint8_t * merged_array = (uint8_t *) malloc(merged_size);
            *reinterpret_cast<uint32_t*>(merged_array) = size1;
            memcpy(merged_array + sizeof(uint32_t), array1, size1);
            memcpy(merged_array + sizeof(uint32_t) + size1, array2, size2);
            return merged_array;
        }

        std::vector<std::vector<bool>> level_signs;
        std::vector<std::vector<uint8_t>> level_recording_bitplanes;
    };
}
#endif
