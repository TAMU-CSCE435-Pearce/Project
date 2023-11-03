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
CMAKE_SOURCE_DIR = /home/miguelgi347/Final_project/CSCE435GroupProject/MergeSort/cuda

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/miguelgi347/Final_project/CSCE435GroupProject/MergeSort/cuda

# Include any dependencies generated for this target.
include CMakeFiles/merge_cuda.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/merge_cuda.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/merge_cuda.dir/flags.make

CMakeFiles/merge_cuda.dir/merge_cuda.cu.o: CMakeFiles/merge_cuda.dir/flags.make
CMakeFiles/merge_cuda.dir/merge_cuda.cu.o: merge_cuda.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/miguelgi347/Final_project/CSCE435GroupProject/MergeSort/cuda/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/merge_cuda.dir/merge_cuda.cu.o"
	/sw/eb/sw/CUDA/9.2.88/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/miguelgi347/Final_project/CSCE435GroupProject/MergeSort/cuda/merge_cuda.cu -o CMakeFiles/merge_cuda.dir/merge_cuda.cu.o

CMakeFiles/merge_cuda.dir/merge_cuda.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/merge_cuda.dir/merge_cuda.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/merge_cuda.dir/merge_cuda.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/merge_cuda.dir/merge_cuda.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/merge_cuda.dir/home/miguelgi347/Final_project/CSCE435GroupProject/Utils/helper_functions.cpp.o: CMakeFiles/merge_cuda.dir/flags.make
CMakeFiles/merge_cuda.dir/home/miguelgi347/Final_project/CSCE435GroupProject/Utils/helper_functions.cpp.o: /home/miguelgi347/Final_project/CSCE435GroupProject/Utils/helper_functions.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/miguelgi347/Final_project/CSCE435GroupProject/MergeSort/cuda/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/merge_cuda.dir/home/miguelgi347/Final_project/CSCE435GroupProject/Utils/helper_functions.cpp.o"
	/sw/eb/sw/GCCcore/7.3.0/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/merge_cuda.dir/home/miguelgi347/Final_project/CSCE435GroupProject/Utils/helper_functions.cpp.o -c /home/miguelgi347/Final_project/CSCE435GroupProject/Utils/helper_functions.cpp

CMakeFiles/merge_cuda.dir/home/miguelgi347/Final_project/CSCE435GroupProject/Utils/helper_functions.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/merge_cuda.dir/home/miguelgi347/Final_project/CSCE435GroupProject/Utils/helper_functions.cpp.i"
	/sw/eb/sw/GCCcore/7.3.0/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/miguelgi347/Final_project/CSCE435GroupProject/Utils/helper_functions.cpp > CMakeFiles/merge_cuda.dir/home/miguelgi347/Final_project/CSCE435GroupProject/Utils/helper_functions.cpp.i

CMakeFiles/merge_cuda.dir/home/miguelgi347/Final_project/CSCE435GroupProject/Utils/helper_functions.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/merge_cuda.dir/home/miguelgi347/Final_project/CSCE435GroupProject/Utils/helper_functions.cpp.s"
	/sw/eb/sw/GCCcore/7.3.0/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/miguelgi347/Final_project/CSCE435GroupProject/Utils/helper_functions.cpp -o CMakeFiles/merge_cuda.dir/home/miguelgi347/Final_project/CSCE435GroupProject/Utils/helper_functions.cpp.s

# Object files for target merge_cuda
merge_cuda_OBJECTS = \
"CMakeFiles/merge_cuda.dir/merge_cuda.cu.o" \
"CMakeFiles/merge_cuda.dir/home/miguelgi347/Final_project/CSCE435GroupProject/Utils/helper_functions.cpp.o"

# External object files for target merge_cuda
merge_cuda_EXTERNAL_OBJECTS =

CMakeFiles/merge_cuda.dir/cmake_device_link.o: CMakeFiles/merge_cuda.dir/merge_cuda.cu.o
CMakeFiles/merge_cuda.dir/cmake_device_link.o: CMakeFiles/merge_cuda.dir/home/miguelgi347/Final_project/CSCE435GroupProject/Utils/helper_functions.cpp.o
CMakeFiles/merge_cuda.dir/cmake_device_link.o: CMakeFiles/merge_cuda.dir/build.make
CMakeFiles/merge_cuda.dir/cmake_device_link.o: /scratch/group/csce435-f23/Caliper/caliper/lib64/libcaliper.so.2.11.0-dev
CMakeFiles/merge_cuda.dir/cmake_device_link.o: /sw/eb/sw/impi/2019.9.304-iccifort-2020.4.304/intel64/lib/libmpicxx.so
CMakeFiles/merge_cuda.dir/cmake_device_link.o: /sw/eb/sw/impi/2019.9.304-iccifort-2020.4.304/intel64/lib/release/libmpi.so
CMakeFiles/merge_cuda.dir/cmake_device_link.o: /lib64/librt.so
CMakeFiles/merge_cuda.dir/cmake_device_link.o: /lib64/libpthread.so
CMakeFiles/merge_cuda.dir/cmake_device_link.o: /lib64/libdl.so
CMakeFiles/merge_cuda.dir/cmake_device_link.o: CMakeFiles/merge_cuda.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/miguelgi347/Final_project/CSCE435GroupProject/MergeSort/cuda/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA device code CMakeFiles/merge_cuda.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/merge_cuda.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/merge_cuda.dir/build: CMakeFiles/merge_cuda.dir/cmake_device_link.o

.PHONY : CMakeFiles/merge_cuda.dir/build

# Object files for target merge_cuda
merge_cuda_OBJECTS = \
"CMakeFiles/merge_cuda.dir/merge_cuda.cu.o" \
"CMakeFiles/merge_cuda.dir/home/miguelgi347/Final_project/CSCE435GroupProject/Utils/helper_functions.cpp.o"

# External object files for target merge_cuda
merge_cuda_EXTERNAL_OBJECTS =

merge_cuda: CMakeFiles/merge_cuda.dir/merge_cuda.cu.o
merge_cuda: CMakeFiles/merge_cuda.dir/home/miguelgi347/Final_project/CSCE435GroupProject/Utils/helper_functions.cpp.o
merge_cuda: CMakeFiles/merge_cuda.dir/build.make
merge_cuda: /scratch/group/csce435-f23/Caliper/caliper/lib64/libcaliper.so.2.11.0-dev
merge_cuda: /sw/eb/sw/impi/2019.9.304-iccifort-2020.4.304/intel64/lib/libmpicxx.so
merge_cuda: /sw/eb/sw/impi/2019.9.304-iccifort-2020.4.304/intel64/lib/release/libmpi.so
merge_cuda: /lib64/librt.so
merge_cuda: /lib64/libpthread.so
merge_cuda: /lib64/libdl.so
merge_cuda: CMakeFiles/merge_cuda.dir/cmake_device_link.o
merge_cuda: CMakeFiles/merge_cuda.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/miguelgi347/Final_project/CSCE435GroupProject/MergeSort/cuda/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable merge_cuda"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/merge_cuda.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/merge_cuda.dir/build: merge_cuda

.PHONY : CMakeFiles/merge_cuda.dir/build

CMakeFiles/merge_cuda.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/merge_cuda.dir/cmake_clean.cmake
.PHONY : CMakeFiles/merge_cuda.dir/clean

CMakeFiles/merge_cuda.dir/depend:
	cd /home/miguelgi347/Final_project/CSCE435GroupProject/MergeSort/cuda && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/miguelgi347/Final_project/CSCE435GroupProject/MergeSort/cuda /home/miguelgi347/Final_project/CSCE435GroupProject/MergeSort/cuda /home/miguelgi347/Final_project/CSCE435GroupProject/MergeSort/cuda /home/miguelgi347/Final_project/CSCE435GroupProject/MergeSort/cuda /home/miguelgi347/Final_project/CSCE435GroupProject/MergeSort/cuda/CMakeFiles/merge_cuda.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/merge_cuda.dir/depend

