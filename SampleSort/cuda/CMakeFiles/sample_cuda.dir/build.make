# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /sw/eb/sw/CMake/3.12.1-GCCcore-7.3.0/bin/cmake

# The command to remove a file.
RM = /sw/eb/sw/CMake/3.12.1-GCCcore-7.3.0/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /scratch/user/aidan.heffron/CSCE435GroupProject/SampleSort/cuda

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /scratch/user/aidan.heffron/CSCE435GroupProject/SampleSort/cuda

# Include any dependencies generated for this target.
include CMakeFiles/sample_cuda.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/sample_cuda.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sample_cuda.dir/flags.make

CMakeFiles/sample_cuda.dir/sample_cuda.cu.o: CMakeFiles/sample_cuda.dir/flags.make
CMakeFiles/sample_cuda.dir/sample_cuda.cu.o: sample_cuda.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/scratch/user/aidan.heffron/CSCE435GroupProject/SampleSort/cuda/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/sample_cuda.dir/sample_cuda.cu.o"
	/sw/eb/sw/CUDA/9.2.88/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /scratch/user/aidan.heffron/CSCE435GroupProject/SampleSort/cuda/sample_cuda.cu -o CMakeFiles/sample_cuda.dir/sample_cuda.cu.o

CMakeFiles/sample_cuda.dir/sample_cuda.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/sample_cuda.dir/sample_cuda.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/sample_cuda.dir/sample_cuda.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/sample_cuda.dir/sample_cuda.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/sample_cuda.dir/scratch/user/aidan.heffron/CSCE435GroupProject/Utils/helper_functions.cpp.o: CMakeFiles/sample_cuda.dir/flags.make
CMakeFiles/sample_cuda.dir/scratch/user/aidan.heffron/CSCE435GroupProject/Utils/helper_functions.cpp.o: /scratch/user/aidan.heffron/CSCE435GroupProject/Utils/helper_functions.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/scratch/user/aidan.heffron/CSCE435GroupProject/SampleSort/cuda/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/sample_cuda.dir/scratch/user/aidan.heffron/CSCE435GroupProject/Utils/helper_functions.cpp.o"
	/sw/eb/sw/GCCcore/7.3.0/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sample_cuda.dir/scratch/user/aidan.heffron/CSCE435GroupProject/Utils/helper_functions.cpp.o -c /scratch/user/aidan.heffron/CSCE435GroupProject/Utils/helper_functions.cpp

CMakeFiles/sample_cuda.dir/scratch/user/aidan.heffron/CSCE435GroupProject/Utils/helper_functions.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sample_cuda.dir/scratch/user/aidan.heffron/CSCE435GroupProject/Utils/helper_functions.cpp.i"
	/sw/eb/sw/GCCcore/7.3.0/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /scratch/user/aidan.heffron/CSCE435GroupProject/Utils/helper_functions.cpp > CMakeFiles/sample_cuda.dir/scratch/user/aidan.heffron/CSCE435GroupProject/Utils/helper_functions.cpp.i

CMakeFiles/sample_cuda.dir/scratch/user/aidan.heffron/CSCE435GroupProject/Utils/helper_functions.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sample_cuda.dir/scratch/user/aidan.heffron/CSCE435GroupProject/Utils/helper_functions.cpp.s"
	/sw/eb/sw/GCCcore/7.3.0/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /scratch/user/aidan.heffron/CSCE435GroupProject/Utils/helper_functions.cpp -o CMakeFiles/sample_cuda.dir/scratch/user/aidan.heffron/CSCE435GroupProject/Utils/helper_functions.cpp.s

# Object files for target sample_cuda
sample_cuda_OBJECTS = \
"CMakeFiles/sample_cuda.dir/sample_cuda.cu.o" \
"CMakeFiles/sample_cuda.dir/scratch/user/aidan.heffron/CSCE435GroupProject/Utils/helper_functions.cpp.o"

# External object files for target sample_cuda
sample_cuda_EXTERNAL_OBJECTS =

CMakeFiles/sample_cuda.dir/cmake_device_link.o: CMakeFiles/sample_cuda.dir/sample_cuda.cu.o
CMakeFiles/sample_cuda.dir/cmake_device_link.o: CMakeFiles/sample_cuda.dir/scratch/user/aidan.heffron/CSCE435GroupProject/Utils/helper_functions.cpp.o
CMakeFiles/sample_cuda.dir/cmake_device_link.o: CMakeFiles/sample_cuda.dir/build.make
CMakeFiles/sample_cuda.dir/cmake_device_link.o: /scratch/group/csce435-f23/Caliper/caliper/lib64/libcaliper.so.2.11.0-dev
CMakeFiles/sample_cuda.dir/cmake_device_link.o: /sw/eb/sw/impi/2019.9.304-iccifort-2020.4.304/intel64/lib/libmpicxx.so
CMakeFiles/sample_cuda.dir/cmake_device_link.o: /sw/eb/sw/impi/2019.9.304-iccifort-2020.4.304/intel64/lib/release/libmpi.so
CMakeFiles/sample_cuda.dir/cmake_device_link.o: /lib64/librt.so
CMakeFiles/sample_cuda.dir/cmake_device_link.o: /lib64/libpthread.so
CMakeFiles/sample_cuda.dir/cmake_device_link.o: /lib64/libdl.so
CMakeFiles/sample_cuda.dir/cmake_device_link.o: CMakeFiles/sample_cuda.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/scratch/user/aidan.heffron/CSCE435GroupProject/SampleSort/cuda/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA device code CMakeFiles/sample_cuda.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sample_cuda.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sample_cuda.dir/build: CMakeFiles/sample_cuda.dir/cmake_device_link.o

.PHONY : CMakeFiles/sample_cuda.dir/build

# Object files for target sample_cuda
sample_cuda_OBJECTS = \
"CMakeFiles/sample_cuda.dir/sample_cuda.cu.o" \
"CMakeFiles/sample_cuda.dir/scratch/user/aidan.heffron/CSCE435GroupProject/Utils/helper_functions.cpp.o"

# External object files for target sample_cuda
sample_cuda_EXTERNAL_OBJECTS =

sample_cuda: CMakeFiles/sample_cuda.dir/sample_cuda.cu.o
sample_cuda: CMakeFiles/sample_cuda.dir/scratch/user/aidan.heffron/CSCE435GroupProject/Utils/helper_functions.cpp.o
sample_cuda: CMakeFiles/sample_cuda.dir/build.make
sample_cuda: /scratch/group/csce435-f23/Caliper/caliper/lib64/libcaliper.so.2.11.0-dev
sample_cuda: /sw/eb/sw/impi/2019.9.304-iccifort-2020.4.304/intel64/lib/libmpicxx.so
sample_cuda: /sw/eb/sw/impi/2019.9.304-iccifort-2020.4.304/intel64/lib/release/libmpi.so
sample_cuda: /lib64/librt.so
sample_cuda: /lib64/libpthread.so
sample_cuda: /lib64/libdl.so
sample_cuda: CMakeFiles/sample_cuda.dir/cmake_device_link.o
sample_cuda: CMakeFiles/sample_cuda.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/scratch/user/aidan.heffron/CSCE435GroupProject/SampleSort/cuda/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable sample_cuda"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sample_cuda.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sample_cuda.dir/build: sample_cuda

.PHONY : CMakeFiles/sample_cuda.dir/build

CMakeFiles/sample_cuda.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sample_cuda.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sample_cuda.dir/clean

CMakeFiles/sample_cuda.dir/depend:
	cd /scratch/user/aidan.heffron/CSCE435GroupProject/SampleSort/cuda && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /scratch/user/aidan.heffron/CSCE435GroupProject/SampleSort/cuda /scratch/user/aidan.heffron/CSCE435GroupProject/SampleSort/cuda /scratch/user/aidan.heffron/CSCE435GroupProject/SampleSort/cuda /scratch/user/aidan.heffron/CSCE435GroupProject/SampleSort/cuda /scratch/user/aidan.heffron/CSCE435GroupProject/SampleSort/cuda/CMakeFiles/sample_cuda.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/sample_cuda.dir/depend

