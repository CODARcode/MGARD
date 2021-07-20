#ifndef _MDR_REFACTOR_UTILS_HPP
#define _MDR_REFACTOR_UTILS_HPP

#include <cassert>
#include <vector>
#include <cmath>
#include <ctime>
#include <fstream>

namespace MDR {

    // MDR utility functions
    template<typename Type>
    std::vector<Type> readfile(const char *file, size_t &num) {
        std::ifstream fin(file, std::ios::binary);
        if (!fin) {
            std::cout << " Error, Couldn't find the file" << "\n";
            return std::vector<Type>();
        }
        fin.seekg(0, std::ios::end);
        const size_t num_elements = fin.tellg() / sizeof(Type);
        fin.seekg(0, std::ios::beg);
        auto data = std::vector<Type>(num_elements);
        fin.read(reinterpret_cast<char *>(&data[0]), num_elements * sizeof(Type));
        fin.close();
        num = num_elements;
        return data;
    }
    template<typename Type>
    void writefile(const char *file, Type *data, size_t num_elements) {
        std::ofstream fout(file, std::ios::binary);
        fout.write(reinterpret_cast<const char *>(&data[0]), num_elements * sizeof(Type));
        fout.close();
    }
    template <class T>
    void print_statistics(const T * data_ori, const T * data_dec, size_t data_size){
        double max_val = data_ori[0];
        double min_val = data_ori[0];
        double max_abs = fabs(data_ori[0]);
        for(int i=0; i<data_size; i++){
            if(data_ori[i] > max_val) max_val = data_ori[i];
            if(data_ori[i] < min_val) min_val = data_ori[i];
            if(fabs(data_ori[i]) > max_abs) max_abs = fabs(data_ori[i]);
        }
        double max_err = 0;
        int pos = 0;
        double mse = 0;
        for(int i=0; i<data_size; i++){
            double err = data_ori[i] - data_dec[i];
            mse += err * err;
            if(fabs(err) > max_err){
                pos = i;
                max_err = fabs(err);
            }
        }
        mse /= data_size;
        double psnr = 20 * log10((max_val - min_val) / sqrt(mse));
        std::cout << "Max value = " << max_val << ", min value = " << min_val << std::endl;
        std::cout << "Max error = " << max_err << ", pos = " << pos << std::endl;
        std::cout << "MSE = " << mse << ", PSNR = " << psnr << std::endl;
    }
    template <class T>
    void print_statistics(const T * data_ori, const T * data_dec, size_t data_size, size_t compressed_size){
        print_statistics(data_ori, data_dec, data_size);
        std::cout << "Compression ratio = " << data_size * sizeof(T) * 1.0 / compressed_size << std::endl;
    }
    // MGARD related
    // TODO: put API in MGARD

    // compute level dimensions
    /*
        @params dims: input dimensions
        @params target_level: the target decomposition level
    */
    /* Different from MGARDx in the computation of level dims 
        MGARD master: 13 -> 9 -> 5 -> 3 -> 2
        MGARDx:       13 -> 7 -> 4 -> 3 -> 2
    */
    std::vector<std::vector<uint32_t>> compute_level_dims(const std::vector<uint32_t>& dims, uint32_t target_level){
        std::vector<std::vector<uint32_t>> level_dims;
        for(int i=0; i<=target_level; i++){
            level_dims.push_back(std::vector<uint32_t>(dims.size()));
        }
        for(int i=0; i<dims.size(); i++){
            level_dims[target_level][i] = dims[i];
            uint32_t p = log2(dims[i] - 1);
            int n = (1u << p) + 1;
            for(int j=1; j<=target_level; j++){
                level_dims[target_level - j][i] = n;
                n = (n >> 1) + 1;
            }
        }
        std::cout << "Dimensions:" << std::endl;
        for(int i=0; i<=target_level; i++){
            for(int j=0; j<dims.size(); j++)
                std::cout << level_dims[i][j] << " ";
            std::cout << std::endl;
        }
        std::cout << "Dimensions end" << std::endl;
        return level_dims;
    }

    // compute level elements
    /*
        @params level_dims: dimensions for all levels
        @params target_level: the target decomposition level
    */
    std::vector<uint32_t> compute_level_elements(const std::vector<std::vector<uint32_t>>& level_dims, int target_level){
        assert(level_dims.size());
        uint8_t num_dims = level_dims[0].size();
        std::vector<uint32_t> level_elements(level_dims.size());
        level_elements[0] = 1;
        for(int j=0; j<num_dims; j++){
            level_elements[0] *= level_dims[0][j];
        }
        uint32_t pre_num_elements = level_elements[0];
        for(int i=1; i<=target_level; i++){
            uint32_t num_elements = 1;
            for(int j=0; j<num_dims; j++){
                num_elements *= level_dims[i][j];
            }
            level_elements[i] = num_elements - pre_num_elements;
            pre_num_elements = num_elements;
        }
        return level_elements;
    }

    // Simple utility functions

    // compute maximum value in level
    /*
    @params data: level data
    @params n: number of level data points
    */
    template <class T>
    T compute_max_abs_value(const T * data, uint32_t n){
        T max_val = 0;
        for(int i=0; i<n; i++){
            T val = fabs(data[i]);
            if(val > max_val) max_val = val;
        }
        return max_val;
    }

    // Get size of vector
    template <class T>
    inline uint32_t get_size(const std::vector<T>& vec){
        return vec.size() * sizeof(T);
    }
    template <class T>
    uint32_t get_size(const std::vector<std::vector<T>>& vec){
        uint32_t size = 0;
        for(int i=0; i<vec.size(); i++){
            size += sizeof(uint32_t) + vec[i].size() * sizeof(T);
        }
        return size;
    }

    // Serialize/deserialize vectors
    // Auto-increment buffer position
    template <class T>
    inline void serialize(const std::vector<T>& vec, uint8_t *& buffer_pos){
        memcpy(buffer_pos, vec.data(), vec.size() * sizeof(T));
        buffer_pos += vec.size() * sizeof(T);
    }
    template <class T>
    void serialize(const std::vector<std::vector<T>>& vec, uint8_t *& buffer_pos){
        uint8_t const * const start = buffer_pos;
        for(int i=0; i<vec.size(); i++){
            *reinterpret_cast<uint32_t*>(buffer_pos) = vec[i].size();
            buffer_pos += sizeof(uint32_t);
            memcpy(buffer_pos, vec[i].data(), vec[i].size() * sizeof(T));
            buffer_pos += vec[i].size() * sizeof(T);
        }
    }
    template <class T>
    inline void deserialize(uint8_t const *& buffer_pos, uint32_t size, std::vector<T>& vec){
        vec.clear();
        vec = std::vector<T>(reinterpret_cast<const T*>(buffer_pos), reinterpret_cast<const T*>(buffer_pos) + size);
        buffer_pos += size * sizeof(T);
    }
    template <class T>
    void deserialize(uint8_t const *& buffer_pos, uint32_t num_levels, std::vector<std::vector<T>>& vec){
        vec.clear();
        for(int i=0; i<num_levels; i++){
            uint32_t num = *reinterpret_cast<const uint32_t*>(buffer_pos);
            buffer_pos += sizeof(uint32_t);
            std::vector<T> level_vec = std::vector<T>(reinterpret_cast<const T *>(buffer_pos), reinterpret_cast<const T *>(buffer_pos) + num);
            vec.push_back(level_vec);
            buffer_pos += num * sizeof(T);
        }
    }

    // print vector
    template <class T>
    void print_vec(const std::vector<T>& vec){
        for(int i=0; i<vec.size(); i++){
            std::cout << vec[i] << " ";
        }
        std::cout << std::endl;
    }
    // print nested vector
    template <class T>
    void print_vec(const std::string& name, const std::vector<std::vector<T>>& vec){
        std::cout << name << std::endl;
        for(int i=0; i<vec.size(); i++){
            print_vec(vec[i]);
        }
        std::cout << std::endl;
    }

    class Timer{
    public:
        void start(){
            err = clock_gettime(CLOCK_REALTIME, &start_time);
        }
        void end(){
            err = clock_gettime(CLOCK_REALTIME, &end_time);
            total_time += (double)(end_time.tv_sec - start_time.tv_sec) + (double)(end_time.tv_nsec - start_time.tv_nsec)/(double)1000000000;
        }
        double get(){
            double time = (double)(end_time.tv_sec - start_time.tv_sec) + (double)(end_time.tv_nsec - start_time.tv_nsec)/(double)1000000000;
            clear();
            return time;
        }
        void clear(){
            total_time = 0;
        }
        void print(std::string s){
            std::cout << s << " time: " << total_time << "s" << std::endl;
            clear();
        }
    private:
        int err = 0;
        double total_time = 0;
        struct timespec start_time, end_time;
    };

}
#endif
