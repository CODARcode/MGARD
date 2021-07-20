#ifndef _MDR_COMPOSED_RECONSTRUCTOR_HPP
#define _MDR_COMPOSED_RECONSTRUCTOR_HPP

#include "ReconstructorInterface.hpp"
#include "Decomposer/Decomposer.hpp"
#include "Interleaver/Interleaver.hpp"
#include "BitplaneEncoder/BitplaneEncoder.hpp"
#include "Retriever/Retriever.hpp"
#include "ErrorEstimator/ErrorEstimator.hpp"
#include "ErrorCollector/ErrorCollector.hpp"
#include "SizeInterpreter/SizeInterpreter.hpp"
#include "LosslessCompressor/LevelCompressor.hpp"
#include "RefactorUtils.hpp"

namespace MDR {
    // a decomposition-based scientific data reconstructor: inverse operator of composed refactor
    template<class T, class Decomposer, class Interleaver, class Encoder, class Compressor, class SizeInterpreter, class ErrorEstimator, class Retriever>
    class ComposedReconstructor : public concepts::ReconstructorInterface<T> {
    public:
        ComposedReconstructor(Decomposer decomposer, Interleaver interleaver, Encoder encoder, Compressor compressor, SizeInterpreter interpreter, Retriever retriever)
            : decomposer(decomposer), interleaver(interleaver), encoder(encoder), compressor(compressor), interpreter(interpreter), retriever(retriever){}

        // reconstruct data from encoded streams
        T * reconstruct(double tolerance){
            Timer timer;
            timer.start();
            std::vector<std::vector<double>> level_abs_errors;
            uint8_t target_level = level_error_bounds.size() - 1;
            std::vector<std::vector<double>>& level_errors = level_squared_errors;
            if(std::is_base_of<MaxErrorEstimator<T>, ErrorEstimator>::value){
                std::cout << "ErrorEstimator is base of MaxErrorEstimator, computing absolute error" << std::endl;
                MaxErrorCollector<T> collector = MaxErrorCollector<T>();
                for(int i=0; i<=target_level; i++){
                    auto collected_error = collector.collect_level_error(NULL, 0, level_squared_errors[i].size(), level_error_bounds[i]);
                    level_abs_errors.push_back(collected_error);
                }
                level_errors = level_abs_errors;
            }
            else if(std::is_base_of<SquaredErrorEstimator<T>, ErrorEstimator>::value){
                std::cout << "ErrorEstimator is base of SquaredErrorEstimator, using level squared error directly" << std::endl;
            }
            else{
                std::cerr << "Customized error estimator not supported yet" << std::endl;
                exit(-1);
            }
            timer.end();
            timer.print("Preprocessing");            

            timer.start();
            auto prev_level_num_bitplanes(level_num_bitplanes);
            auto retrieve_sizes = interpreter.interpret_retrieve_size(level_sizes, level_errors, tolerance, level_num_bitplanes);
            // retrieve data
            level_components = retriever.retrieve_level_components(level_sizes, retrieve_sizes, prev_level_num_bitplanes, level_num_bitplanes);
            // check whether to reconstruct to full resolution
            int skipped_level = 0;
            for(int i=0; i<=target_level; i++){
                if(level_num_bitplanes[target_level - i] != 0){
                    skipped_level = i;
                    break;
                }
            }
            // TODO: uncomment skip level to reconstruct low resolution data
            // target_level -= skipped_level;
            timer.end();
            timer.print("Interpret and retrieval");

            bool success = reconstruct(target_level, prev_level_num_bitplanes);
            retriever.release();
            if(success) return data.data();
            else{
                std::cerr << "Reconstruct unsuccessful, return NULL pointer" << std::endl;
                return NULL;
            }
        }

        // reconstruct progressively based on available data
        T * progressive_reconstruct(double tolerance){
            std::vector<T> cur_data(data);
            reconstruct(tolerance);
            // TODO: add resolution changes
            if(cur_data.size() == data.size()){
                for(int i=0; i<data.size(); i++){
                    data[i] += cur_data[i];
                }                
            }
            else if(cur_data.size()){
                std::cerr << "Reconstruct size changes, not supported yet." << std::endl;
                std::cerr << "Sizes before reconstruction: " << cur_data.size() << std::endl;
                std::cerr << "Sizes after reconstruction: " << data.size() << std::endl;
                exit(0);
            }
            return data.data();
        }

        void load_metadata(){
            uint8_t * metadata = retriever.load_metadata();
            uint8_t const * metadata_pos = metadata;
            uint8_t num_dims = *(metadata_pos ++);
            deserialize(metadata_pos, num_dims, dimensions);
            uint8_t num_levels = *(metadata_pos ++);
            deserialize(metadata_pos, num_levels, level_error_bounds);
            deserialize(metadata_pos, num_levels, level_squared_errors);
            deserialize(metadata_pos, num_levels, level_sizes);
            deserialize(metadata_pos, num_levels, stopping_indices);
            deserialize(metadata_pos, num_levels, level_num);
            level_num_bitplanes = std::vector<uint8_t>(num_levels, 0);
            free(metadata);
        }

        const std::vector<uint32_t>& get_dimensions(){
            return dimensions;
        }

        ~ComposedReconstructor(){}

        void print() const {
            std::cout << "Composed reconstructor with the following components." << std::endl;
            std::cout << "Decomposer: "; decomposer.print();
            std::cout << "Interleaver: "; interleaver.print();
            std::cout << "Encoder: "; encoder.print();
            std::cout << "SizeInterpreter: "; interpreter.print();
            std::cout << "Retriever: "; retriever.print();
        }
    private:
        bool reconstruct(uint8_t target_level, const std::vector<uint8_t>& prev_level_num_bitplanes, bool progressive=true){
            Timer timer;
            timer.start();
            auto level_dims = compute_level_dims(dimensions, target_level);
            auto reconstruct_dimensions = level_dims[target_level];
            uint32_t num_elements = 1;
            for(const auto& dim:reconstruct_dimensions){
                num_elements *= dim;
            }
            data.clear();
            data = std::vector<T>(num_elements, 0);
            timer.end();
            timer.print("Reconstruct Preprocessing");            

            auto level_elements = compute_level_elements(level_dims, target_level);
            std::vector<uint32_t> dims_dummy(reconstruct_dimensions.size(), 0);
            for(int i=0; i<=target_level; i++){
                timer.start();
                compressor.decompress_level(level_components[i], level_sizes[i], prev_level_num_bitplanes[i], level_num_bitplanes[i] - prev_level_num_bitplanes[i], stopping_indices[i]);
                timer.end();
                timer.print("Lossless");            
                timer.start();
                int level_exp = 0;
                frexp(level_error_bounds[i], &level_exp);
                auto level_decoded_data = encoder.progressive_decode(level_components[i], level_elements[i], level_exp, prev_level_num_bitplanes[i], level_num_bitplanes[i] - prev_level_num_bitplanes[i], i);
                compressor.decompress_release();
                timer.end();
                timer.print("Decoding");            

                timer.start();
                const std::vector<uint32_t>& prev_dims = (i == 0) ? dims_dummy : level_dims[i - 1];
                interleaver.reposition(level_decoded_data, reconstruct_dimensions, level_dims[i], prev_dims, data.data());
                free(level_decoded_data);
                timer.end();
                timer.print("Reposition");            
            }
            timer.start();
            decomposer.recompose(data.data(), reconstruct_dimensions, target_level);
            timer.end();
            timer.print("Recomposing");            
            return true;
        }

        Decomposer decomposer;
        Interleaver interleaver;
        Encoder encoder;
        SizeInterpreter interpreter;
        Retriever retriever;
        Compressor compressor;
        std::vector<T> data;
        std::vector<uint32_t> dimensions;
        std::vector<T> level_error_bounds;
        std::vector<uint8_t> level_num_bitplanes;
        std::vector<uint8_t> stopping_indices;
        std::vector<std::vector<const uint8_t*>> level_components;
        std::vector<std::vector<uint32_t>> level_sizes;
        std::vector<uint32_t> level_num;
        std::vector<std::vector<double>> level_squared_errors;
    };
}
#endif

