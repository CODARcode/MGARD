#ifndef _MDR_SFC_INTERLEAVER_HPP
#define _MDR_SFC_INTERLEAVER_HPP

#include "InterleaverInterface.hpp"

namespace MDR {
    // direct interleaver with in-order recording
    template<class T>
    class SFCInterleaver : public concepts::InterleaverInterface<T> {
    public:
        SFCInterleaver(){}
        void interleave(T const * data, const std::vector<uint32_t>& dims, const std::vector<uint32_t>& dims_fine, const std::vector<uint32_t>& dims_coasre, T * buffer) const {
            size_t n1_nodal = dims_coasre[0];
            size_t n2_nodal = dims_coasre[1];
            size_t n3_nodal = dims_coasre[2];
            size_t n1_coeff = dims_fine[0] - n1_nodal;
            size_t n2_coeff = dims_fine[1] - n2_nodal;
            size_t n3_coeff = dims_fine[2] - n3_nodal;
            size_t dim0_offset = dims[1] * dims[2];
            size_t dim1_offset = dims[2];
            const int block_size = 1;
            if(n1_nodal * n2_nodal * n3_nodal == 0){
                collect_data_3d_blocked(data, n1_coeff, n2_coeff, n3_coeff, dim0_offset, dim1_offset, block_size, buffer);
            }
            else{
                const T * nodal_nodal_coeff_pos = data + n3_nodal;
                const T * nodal_coeff_nodal_pos = data + n2_nodal * dim1_offset;
                const T * nodal_coeff_coeff_pos = nodal_coeff_nodal_pos + n3_nodal;
                const T * coeff_nodal_nodal_pos = data + n1_nodal * dim0_offset;
                const T * coeff_nodal_coeff_pos = coeff_nodal_nodal_pos + n3_nodal;
                const T * coeff_coeff_nodal_pos = coeff_nodal_nodal_pos + n2_nodal * dim1_offset;
                const T * coeff_coeff_coeff_pos = coeff_coeff_nodal_pos + n3_nodal;
                T * tmp_buffer = (T *) malloc(dims_fine[0] * dims_fine[1] * dims_fine[2] * sizeof(T));
                T * buffer_pos = tmp_buffer;
                const T * pos[7];
                pos[0] = buffer_pos;
                buffer_pos += collect_data_3d_blocked(nodal_nodal_coeff_pos, n1_nodal, n2_nodal, n3_coeff, dim0_offset, dim1_offset, block_size, buffer_pos);
                pos[1] = buffer_pos;
                buffer_pos += collect_data_3d_blocked(nodal_coeff_nodal_pos, n1_nodal, n2_coeff, n3_nodal, dim0_offset, dim1_offset, block_size, buffer_pos);
                pos[2] = buffer_pos;
                buffer_pos += collect_data_3d_blocked(nodal_coeff_coeff_pos, n1_nodal, n2_coeff, n3_coeff, dim0_offset, dim1_offset, block_size, buffer_pos);
                pos[3] = buffer_pos;
                buffer_pos += collect_data_3d_blocked(coeff_nodal_nodal_pos, n1_coeff, n2_nodal, n3_nodal, dim0_offset, dim1_offset, block_size, buffer_pos);
                pos[4] = buffer_pos;
                buffer_pos += collect_data_3d_blocked(coeff_nodal_coeff_pos, n1_coeff, n2_nodal, n3_coeff, dim0_offset, dim1_offset, block_size, buffer_pos);
                pos[5] = buffer_pos;
                buffer_pos += collect_data_3d_blocked(coeff_coeff_nodal_pos, n1_coeff, n2_coeff, n3_nodal, dim0_offset, dim1_offset, block_size, buffer_pos);
                pos[6] = buffer_pos;
                buffer_pos += collect_data_3d_blocked(coeff_coeff_coeff_pos, n1_coeff, n2_coeff, n3_coeff, dim0_offset, dim1_offset, block_size, buffer_pos);
                // z_order_data_collection(pos, buffer, n1_nodal, n1_coeff, n2_nodal, n2_coeff, n3_nodal, n3_coeff);
                skip_one_data_collection(pos, buffer, n1_nodal, n1_coeff, n2_nodal, n2_coeff, n3_nodal, n3_coeff);
                free(tmp_buffer);
            }
        }
        void reposition(T const * buffer, const std::vector<uint32_t>& dims, const std::vector<uint32_t>& dims_fine, const std::vector<uint32_t>& dims_coasre, T * data) const {
            size_t n1_nodal = dims_coasre[0];
            size_t n2_nodal = dims_coasre[1];
            size_t n3_nodal = dims_coasre[2];
            size_t n1_coeff = dims_fine[0] - n1_nodal;
            size_t n2_coeff = dims_fine[1] - n2_nodal;
            size_t n3_coeff = dims_fine[2] - n3_nodal;
            size_t dim0_offset = dims[1] * dims[2];
            size_t dim1_offset = dims[2];
            const int block_size = 1;
            if(n1_nodal * n2_nodal * n3_nodal == 0){
                reposition_data_3d_blocked(buffer, n1_coeff, n2_coeff, n3_coeff, dim0_offset, dim1_offset, block_size, data);
            }
            else{
                T * nodal_nodal_coeff_pos = data + n3_nodal;
                T * nodal_coeff_nodal_pos = data + n2_nodal * dim1_offset;
                T * nodal_coeff_coeff_pos = nodal_coeff_nodal_pos + n3_nodal;
                T * coeff_nodal_nodal_pos = data + n1_nodal * dim0_offset;
                T * coeff_nodal_coeff_pos = coeff_nodal_nodal_pos + n3_nodal;
                T * coeff_coeff_nodal_pos = coeff_nodal_nodal_pos + n2_nodal * dim1_offset;
                T * coeff_coeff_coeff_pos = coeff_coeff_nodal_pos + n3_nodal;
                T * tmp_buffer = (T *) malloc(dims_fine[0] * dims_fine[1] * dims_fine[2] * sizeof(T));        
                T * pos[7];
                pos[0] = tmp_buffer;
                pos[1] = pos[0] + n1_nodal * n2_nodal * n3_coeff;
                pos[2] = pos[1] + n1_nodal * n2_coeff * n3_nodal;
                pos[3] = pos[2] + n1_nodal * n2_coeff * n3_coeff;
                pos[4] = pos[3] + n1_coeff * n2_nodal * n3_nodal;
                pos[5] = pos[4] + n1_coeff * n2_nodal * n3_coeff;
                pos[6] = pos[5] + n1_coeff * n2_coeff * n3_nodal;
                // z_order_data_reposition(buffer, pos, n1_nodal, n1_coeff, n2_nodal, n2_coeff, n3_nodal, n3_coeff);
                skip_one_data_reposition(buffer, pos, n1_nodal, n1_coeff, n2_nodal, n2_coeff, n3_nodal, n3_coeff);
                // note that pos[p] has been increased to the position of next segment
                reposition_data_3d_blocked(tmp_buffer, n1_nodal, n2_nodal, n3_coeff, dim0_offset, dim1_offset, block_size, nodal_nodal_coeff_pos);
                reposition_data_3d_blocked(pos[0], n1_nodal, n2_coeff, n3_nodal, dim0_offset, dim1_offset, block_size, nodal_coeff_nodal_pos);
                reposition_data_3d_blocked(pos[1], n1_nodal, n2_coeff, n3_coeff, dim0_offset, dim1_offset, block_size, nodal_coeff_coeff_pos);
                reposition_data_3d_blocked(pos[2], n1_coeff, n2_nodal, n3_nodal, dim0_offset, dim1_offset, block_size, coeff_nodal_nodal_pos);
                reposition_data_3d_blocked(pos[3], n1_coeff, n2_nodal, n3_coeff, dim0_offset, dim1_offset, block_size, coeff_nodal_coeff_pos);
                reposition_data_3d_blocked(pos[4], n1_coeff, n2_coeff, n3_nodal, dim0_offset, dim1_offset, block_size, coeff_coeff_nodal_pos);
                reposition_data_3d_blocked(pos[5], n1_coeff, n2_coeff, n3_coeff, dim0_offset, dim1_offset, block_size, coeff_coeff_coeff_pos);
                free(tmp_buffer);
            }
        }
        void print() const {
            std::cout << "Space filling curve interleaver" << std::endl;
        }
    private:
        size_t collect_data_3d_blocked(const T * data, const size_t n1, const size_t n2, const size_t n3, const size_t dim0_offset, const size_t dim1_offset, const int block_size, T * buffer) const{
            size_t num_block_1 = (n1 - 1) / block_size + 1;
            size_t num_block_2 = (n2 - 1) / block_size + 1;
            size_t num_block_3 = (n3 - 1) / block_size + 1;
            size_t index = 0;
            const T * data_x_pos = data;
            for(int i=0; i<num_block_1; i++){
                int size_1 = (i == num_block_1 - 1) ? n1 - i * block_size : block_size;
                const T * data_y_pos = data_x_pos;
                for(int j=0; j<num_block_2; j++){
                    int size_2 = (j == num_block_2 - 1) ? n2 - j * block_size : block_size;
                    const T * data_z_pos = data_y_pos;
                    for(int k=0; k<num_block_3; k++){
                        int size_3 = (k == num_block_3 - 1) ? n3 - k * block_size : block_size;
                        const T * cur_data_pos = data_z_pos;
                        for(int ii=0; ii<size_1; ii++){
                            for(int jj=0; jj<size_2; jj++){
                                for(int kk=0; kk<size_3; kk++){
                                    buffer[index ++] = *cur_data_pos;
                                    cur_data_pos ++;
                                }
                                cur_data_pos += dim1_offset - size_3;
                            }
                            cur_data_pos += dim0_offset - size_2 * dim1_offset;
                        }
                        data_z_pos += size_3;
                    }
                    data_y_pos += dim1_offset * size_2;
                }
                data_x_pos += dim0_offset * size_1;
            }
            return index;
        }
        size_t reposition_data_3d_blocked(const T * buffer, const size_t n1, const size_t n2, const size_t n3, const size_t dim0_offset, const size_t dim1_offset, const int block_size, T * data) const{
            size_t num_block_1 = (n1 - 1) / block_size + 1;
            size_t num_block_2 = (n2 - 1) / block_size + 1;
            size_t num_block_3 = (n3 - 1) / block_size + 1;
            size_t index = 0;
            T * data_x_pos = data;
            for(int i=0; i<num_block_1; i++){
                int size_1 = (i == num_block_1 - 1) ? n1 - i * block_size : block_size;
                T * data_y_pos = data_x_pos;
                for(int j=0; j<num_block_2; j++){
                    int size_2 = (j == num_block_2 - 1) ? n2 - j * block_size : block_size;
                    T * data_z_pos = data_y_pos;
                    for(int k=0; k<num_block_3; k++){
                        int size_3 = (k == num_block_3 - 1) ? n3 - k * block_size : block_size;
                        T * cur_data_pos = data_z_pos;
                        for(int ii=0; ii<size_1; ii++){
                            for(int jj=0; jj<size_2; jj++){
                                for(int kk=0; kk<size_3; kk++){
                                    *cur_data_pos = buffer[index ++];
                                    cur_data_pos ++;
                                }
                                cur_data_pos += dim1_offset - size_3;
                            }
                            cur_data_pos += dim0_offset - size_2 * dim1_offset;
                        }
                        data_z_pos += size_3;
                    }
                    data_y_pos += dim1_offset * size_2;
                }
                data_x_pos += dim0_offset * size_1;
            }
            return index;
        }
        // collect data in self-defined order
        /*
            0 1 2 3
            4 5 6 7  => 1-4-5-6-3-7

            3d 0-7 => 2-1-3-6-4-5-7
        */
        void skip_one_data_collection(const T * pos[7], T * buffer, size_t n1_nodal, size_t n1_coeff, size_t n2_nodal, size_t n2_coeff, size_t n3_nodal, size_t n3_coeff) const{
            int index = 0;
            for(int i=0; i<n1_coeff; i++){
                for(int j=0; j<n2_coeff; j++){
                    for(int k=0; k<n3_coeff; k++){
                        buffer[index ++] = *pos[1];
                        pos[1] ++;
                        buffer[index ++] = *pos[0];
                        pos[0] ++;
                        buffer[index ++] = *pos[2];
                        pos[2] ++;
                        buffer[index ++] = *pos[5];
                        pos[5] ++;
                        buffer[index ++] = *pos[3];
                        pos[3] ++;
                        buffer[index ++] = *pos[4];
                        pos[4] ++;
                        buffer[index ++] = *pos[6];
                        pos[6] ++;
                    }
                    for(int k=n3_coeff; k<n3_nodal; k++){
                        buffer[index ++] = *pos[1];
                        pos[1] ++;
                        buffer[index ++] = *pos[5];
                        pos[5] ++;
                        buffer[index ++] = *pos[3];
                        pos[3] ++;
                    }
                }
                for(int j=n2_coeff; j<n2_nodal; j++){
                    for(int k=0; k<n3_coeff; k++){
                        buffer[index ++] = *pos[0];
                        pos[0] ++;
                        buffer[index ++] = *pos[3];
                        pos[3] ++;
                        buffer[index ++] = *pos[4];
                        pos[4] ++;
                    }
                    for(int k=n3_coeff; k<n3_nodal; k++){
                        buffer[index ++] = *pos[3];
                        pos[3] ++;
                    }
                }
            }
            for(int i=n1_coeff; i<n1_nodal; i++){
                for(int j=0; j<n2_coeff; j++){
                    for(int k=0; k<n3_coeff; k++){
                        buffer[index ++] = *pos[1];
                        pos[1] ++;
                        buffer[index ++] = *pos[0];
                        pos[0] ++;
                        buffer[index ++] = *pos[2];
                        pos[2] ++;            
                    }
                    for(int k=n3_coeff; k<n3_nodal; k++){
                        buffer[index ++] = *pos[1];
                        pos[1] ++;
                    }
                }
                for(int j=n2_coeff; j<n2_nodal; j++){
                    for(int k=0; k<n3_coeff; k++){
                        buffer[index ++] = *pos[0];
                        pos[0] ++;
                    }                
                }
            }    
        }        
        void skip_one_data_reposition(const T * buffer, T * pos[7], size_t n1_nodal, size_t n1_coeff, size_t n2_nodal, size_t n2_coeff, size_t n3_nodal, size_t n3_coeff) const{
            int index = 0;
            for(int i=0; i<n1_coeff; i++){
                for(int j=0; j<n2_coeff; j++){
                    for(int k=0; k<n3_coeff; k++){
                        *pos[1] = buffer[index ++];
                        pos[1] ++;
                        *pos[0] = buffer[index ++];
                        pos[0] ++;
                        *pos[2] = buffer[index ++];
                        pos[2] ++;
                        *pos[5] = buffer[index ++];
                        pos[5] ++;
                        *pos[3] = buffer[index ++];
                        pos[3] ++;
                        *pos[4] = buffer[index ++];
                        pos[4] ++;
                        *pos[6] = buffer[index ++];
                        pos[6] ++;
                    }
                    for(int k=n3_coeff; k<n3_nodal; k++){
                        *pos[1] = buffer[index ++];
                        pos[1] ++;
                        *pos[5] = buffer[index ++];
                        pos[5] ++;
                        *pos[3] = buffer[index ++];
                        pos[3] ++;
                    }
                }
                for(int j=n2_coeff; j<n2_nodal; j++){
                    for(int k=0; k<n3_coeff; k++){
                        *pos[0] = buffer[index ++];
                        pos[0] ++;
                        *pos[3] = buffer[index ++];
                        pos[3] ++;
                        *pos[4] = buffer[index ++];
                        pos[4] ++;
                    }
                    for(int k=n3_coeff; k<n3_nodal; k++){
                        *pos[3] = buffer[index ++];
                        pos[3] ++;
                    }
                }
            }
            for(int i=n1_coeff; i<n1_nodal; i++){
                for(int j=0; j<n2_coeff; j++){
                    for(int k=0; k<n3_coeff; k++){
                        *pos[1] = buffer[index ++];
                        pos[1] ++;
                        *pos[0] = buffer[index ++];
                        pos[0] ++;
                        *pos[2] = buffer[index ++];
                        pos[2] ++;            
                    }
                    for(int k=n3_coeff; k<n3_nodal; k++){
                        *pos[1] = buffer[index ++];
                        pos[1] ++;
                    }
                }
                for(int j=n2_coeff; j<n2_nodal; j++){
                    for(int k=0; k<n3_coeff; k++){
                        *pos[0] = buffer[index ++];
                        pos[0] ++;
                    }                
                }
            }  
        }
    };
}
#endif
