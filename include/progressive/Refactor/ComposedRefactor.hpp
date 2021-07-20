#ifndef _MDR_COMPOSED_REFACTOR_HPP
#define _MDR_COMPOSED_REFACTOR_HPP

#include "RefactorInterface.hpp"
#include "Decomposer/Decomposer.hpp"
#include "Interleaver/Interleaver.hpp"
#include "BitplaneEncoder/BitplaneEncoder.hpp"
#include "ErrorCollector/ErrorCollector.hpp"
#include "LosslessCompressor/LevelCompressor.hpp"
#include "Writer/Writer.hpp"
#include "RefactorUtils.hpp"

namespace MDR {
    // a decomposition-based scientific data refactor: compose a refactor using decomposer, interleaver, encoder, and error collector
    template<class T, class Decomposer, class Interleaver, class Encoder, class Compressor, class ErrorCollector, class Writer>
    class ComposedRefactor : public concepts::RefactorInterface<T> {
    public:
        ComposedRefactor(Decomposer decomposer, Interleaver interleaver, Encoder encoder, Compressor compressor, ErrorCollector collector, Writer writer)
            : decomposer(decomposer), interleaver(interleaver), encoder(encoder), compressor(compressor), collector(collector), writer(writer) {}

        void refactor(T const * data_, const std::vector<uint32_t>& dims, uint8_t target_level, uint8_t num_bitplanes){
            Timer timer;
            timer.start();
            dimensions = dims;
            uint32_t num_elements = 1;
            for(const auto& dim:dimensions){
                num_elements *= dim;
            }
            data = std::vector<T>(data_, data_ + num_elements);
            // if refactor successfully
            if(refactor(target_level, num_bitplanes)){
                timer.end();
                timer.print("Refactor");
                timer.start();
                level_num = writer.write_level_components(level_components, level_sizes);
                timer.end();
                timer.print("Write");                
            }

            write_metadata();
            for(int i=0; i<level_components.size(); i++){
                for(int j=0; j<level_components[i].size(); j++){
                    free(level_components[i][j]);                    
                }
            }
        }

        void write_metadata() const {
            uint32_t metadata_size = sizeof(uint8_t) + get_size(dimensions) // dimensions
                            + sizeof(uint8_t) + get_size(level_error_bounds) + get_size(level_squared_errors) + get_size(level_sizes) // level information
                            + get_size(stopping_indices) + get_size(level_num);
            uint8_t * metadata = (uint8_t *) malloc(metadata_size);
            uint8_t * metadata_pos = metadata;
            *(metadata_pos ++) = (uint8_t) dimensions.size();
            serialize(dimensions, metadata_pos);
            *(metadata_pos ++) = (uint8_t) level_error_bounds.size();
            serialize(level_error_bounds, metadata_pos);
            serialize(level_squared_errors, metadata_pos);
            serialize(level_sizes, metadata_pos);
            serialize(stopping_indices, metadata_pos);
            serialize(level_num, metadata_pos);
            writer.write_metadata(metadata, metadata_size);
            free(metadata);
        }

        ~ComposedRefactor(){}

        void print() const {
            std::cout << "Composed refactor with the following components." << std::endl;
            std::cout << "Decomposer: "; decomposer.print();
            std::cout << "Interleaver: "; interleaver.print();
            std::cout << "Encoder: "; encoder.print();
        }
    private:
        bool refactor(uint8_t target_level, uint8_t num_bitplanes){
            uint8_t max_level = log2(*min_element(dimensions.begin(), dimensions.end())) - 1;
            if(target_level > max_level){
                std::cerr << "Target level is higher than " << max_level << std::endl;
                return false;
            }
            Timer timer;
            // decompose data hierarchically
            timer.start();
            decomposer.decompose(data.data(), dimensions, target_level);
            timer.end();
            timer.print("Decompose");

            // encode level by level
            level_error_bounds.clear();
            level_squared_errors.clear();
            level_components.clear();
            level_sizes.clear();
            auto level_dims = compute_level_dims(dimensions, target_level);
            auto level_elements = compute_level_elements(level_dims, target_level);
            std::vector<uint32_t> dims_dummy(dimensions.size(), 0);
            SquaredErrorCollector<T> s_collector = SquaredErrorCollector<T>();
            for(int i=0; i<=target_level; i++){
                timer.start();
                const std::vector<uint32_t>& prev_dims = (i == 0) ? dims_dummy : level_dims[i - 1];
                T * buffer = (T *) malloc(level_elements[i] * sizeof(T));
                // extract level i component
                interleaver.interleave(data.data(), dimensions, level_dims[i], prev_dims, reinterpret_cast<T*>(buffer));
                // compute max coefficient as level error bound
                T level_max_error = compute_max_abs_value(reinterpret_cast<T*>(buffer), level_elements[i]);
                level_error_bounds.push_back(level_max_error);
                timer.end();
                timer.print("Interleave");
                // collect errors
                // auto collected_error = s_collector.collect_level_error(buffer, level_elements[i], num_bitplanes, level_max_error);
                // level_squared_errors.push_back(collected_error);
                // encode level data
                timer.start();
                int level_exp = 0;
                frexp(level_max_error, &level_exp);
                std::vector<uint32_t> stream_sizes;
                std::vector<double> level_sq_err;
                auto streams = encoder.encode(buffer, level_elements[i], level_exp, num_bitplanes, stream_sizes, level_sq_err);
                free(buffer);
                level_squared_errors.push_back(level_sq_err);
                timer.end();
                timer.print("Encoding");
                timer.start();
                // lossless compression
                uint8_t stopping_index = compressor.compress_level(streams, stream_sizes);
                stopping_indices.push_back(stopping_index);
                // record encoded level data and size
                level_components.push_back(streams);
                level_sizes.push_back(stream_sizes);
                timer.end();
                timer.print("Lossless time");
            }
            print_vec("level sizes", level_sizes);
            return true;
        }

        Decomposer decomposer;
        Interleaver interleaver;
        Encoder encoder;
        Compressor compressor;
        ErrorCollector collector;
        Writer writer;
        std::vector<T> data;
        std::vector<uint32_t> dimensions;
        std::vector<T> level_error_bounds;
        std::vector<uint8_t> stopping_indices;
        std::vector<std::vector<uint8_t*>> level_components;
        std::vector<std::vector<uint32_t>> level_sizes;
        std::vector<uint32_t> level_num;
        std::vector<std::vector<double>> level_squared_errors;
    };
}
#endif

