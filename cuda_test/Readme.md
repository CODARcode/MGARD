### 1. Install dependecies
#### a. ADIOS2: https://github.com/ornladios/ADIOS2
#### b. Zlib: https://zlib.net/
### 2. Compile and install MAGRD
#### a. ```cd``` into the root of this repository.
#### b. Create a ```build``` directory and ```cd``` into that directory
#### c. Use ```CMake``` to build the MGARD and install it
### 3. Testing 
#### a. ```cd``` into the ```cuda_test```  directory.
#### b. Generate Gray-Scott simulation data
##### b.1 ```cd``` into the ```gray-scott``` directory.
##### b.2 Create a ```build``` directory and ```cd``` into that directory
##### b.3 Use ```CMake``` to build Gray-Scott simulation code
##### b.4 Use ```cd```  back to the ```gray-scott``` directory.
##### b.5 Run ```generate_data.sh``` bash script to run Gray-Scott simulation. Multiple runs with different problem sizes will run and dump simulation data in ADIOS BP4 format to the ```gs_data``` directory. 
#### c. Convert ADIOS BP4 format to plain binary format
##### c.1 ```cd``` into the ```bp2bin``` directory
##### c.2 Create a ```build``` directory and ```cd``` into that directory
##### c.3 Use ```CMake``` to build Gray-Scott simulation code
##### c.4 Use ```cd```  back to the ```bp2bin``` directory
##### c.5 Run ```convert_data.sh``` bash script to run convert data format. Converted binary data will be in ```gs_bin_data``` directory.
#### d. Run tests
##### d1. Run 'test_mgard.py' Python script to automically run our GPU-accelerated multlgrid-based data refactoring and the original CPU version (baseline). Multiple input sizes will be test. For each size, the test will be done multiple time and the average time takes for each kernel inside data refactoring routines (decomposition and recomposition) will be save in CSV files under ```<PLATFORM>``` directory. ```<PLATFORM>``` is configured near the top of the Python run script. The test will also have output in the terminal including the profiling information when our data refactoring routines is used in lossy compressors (e.g., compression ratio, GPU-CPU memory copy time, entropy encoding time, quantization, whether or not the error tolerance is met). 



