// MGARD: MultiGrid Adaptive Reduction of Data
// Authors: Mark Ainsworth, Ozan Tugluk, Ben Whitney
// Corresponding Author: Ozan Tugluk

// version: 0.0.0.1

// This file is part of MGARD.

// MGARD is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// Foobar is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public License
// along with MGARD.  If not, see <http://www.gnu.org/licenses/>.


#ifndef MGARD_CAPI_H
#define MGARD_CAPI_H

extern unsigned char *mgard_compress(int itype_flag, void *data, int *out_size, int nrow, int ncol, void* tol);

extern unsigned char *mgard_decompress(int itype_flag, unsigned char *data, int data_len, int nrow, int ncol);


#endif
