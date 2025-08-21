ROOT_DIR:=$(shell pwd)

# Flags for compiling with NVCC
CU = nvcc
#CUFLAGS = -std=c++17 -O0 -g -ccbin g++-11 -I "/usr/include/eigen3" -I "/usr/include/nlohmann/" -I "$(ROOT_DIR)/ "
CUFLAGS = -std=c++17 -O0 -g -ccbin mpic++ \
          -I "/usr/include/eigen3" \
          -I "/usr/include/nlohmann/" \
          -I "$(ROOT_DIR)/" \
          -I "/usr/local/lib/python3.10/dist-packages/finitediff/include/"


LDFLAGS =  -lgomp

# List the .cpp files required for compiling to .o objects, note we exlcude the .tpp template files 
LIST_HELPERS_CODE = 	memetico/helpers/rng memetico/helpers/distance
LIST_HELPERS_TEST = 	# To be implemented
LIST_MODEL_BASE_CODE = 	memetico/model_base/model
LIST_MODEL_BASE_TEST = 	memetico/model_base/element.test memetico/model_base/model.test
LIST_MODELS_CODE = 		# All code via template classes, so effectively all code is in header files
## Had to remove the regression test when -O is > 0, specifically at the time we introduced lenz
#LIST_MODELS_TEST = 		memetico/models/regression.test memetico/models/cont_frac.test memetico/models/branch_cont_frac_dd.test
LIST_MODELS_TEST = 		memetico/models/cont_frac.test memetico/models/branch_cont_frac_dd.test
#LIST_POP_CODE =			# working.. may all be in tpp/header files
#LIST_POP_TEST =			memetico/population/agent.test memetico/population/pop.test
LIST_DATA_CODE =		memetico/data/data_set
LIST_DATA_TEST =		memetico/data/data_set.test memetico/data/multiV.test
LIST_GPU_CODE =			memetico/gpu/cuda
LIST_GPU_TEST =			memetico/gpu/cuda.test
LIST_OPTIMISE_CODE =	
LIST_OPTIMISE_TEST =	memetico/optimise/objective.test memetico/optimise/local_search.test
# Aggregates
LIST_TEST = $(LIST_HELPERS_TEST) $(LIST_MODEL_BASE_TEST) $(LIST_MODELS_TEST) $(LIST_POP_TEST) $(LIST_DATA_TEST) $(LIST_GPU_TEST) $(LIST_OPTIMISE_TEST)
LIST_CODE = $(LIST_HELPERS_CODE) $(LIST_MODEL_BASE_CODE) $(LIST_MODELS_CODE) $(LIST_POP_CODE) $(LIST_DATA_CODE) $(LIST_GPU_CODE) $(LIST_OPTIMISE_CODE)
# Final list 
LIST = main.test $(LIST_TEST) $(LIST_CODE)

# List the .cu cuda files. We can technically compile these files with NVCC and all others with g++ 
# However there are no impacts in debugging or optimisation that work differently with NVCC and
# when comparing -O3 optimisation it was 3 seconds faster on a 50second run
CULIST = 

# Append suffix to files above
SRC = $(addsuffix .cpp, $(LIST)) $(addsuffix .cu, $(CULIST)) 

# Generate object files based on the -o filenames.
# This assumes that the directories that appear in LIST or CULIST have corresponding directories in ./bin/
OBJ = $(addprefix bin/, $(addsuffix .o, $(LIST)))  $(addprefix bin/, $(addsuffix .o, $(CULIST)))

all: main

# Compile .cpp files
bin/%.o : %.cpp
	$(CU) -c $< $(CUFLAGS) -o $@

# Compile .cu files (same as .cpp)
bin/%.o : %.cu
	$(CU) -c $< $(CUFLAGS) -o $@

# Compiile main executable.
main: $(OBJ)
	$(CU) $^ $(LDFLAGS) -o bin/main

# Clean bin directories to ensure recompilation
clean:
	rm -f bin/memetico/* bin/memetico/helpers/* bin/memetico/models/* bin/memetico/model_base/* bin/memetico/population/* bin/memetico/data/* bin/memetico/gpu/* bin/memetico/optimise/*  bin/*  
