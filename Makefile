CC:=g++    
NVCC := nvcc
HIPCC := hipcc

OMPFLG=-fopenmp        
HASHFLG=-Wno-deprecated      
BUILDFLG=-w -ffunction-sections -fdata-sections -fmodulo-sched 
OBJ_EXT=src/ExtractRNA.o    

EXE_CMP=bin/comp
EXE_CMP_CUDA=bin/cuda-comp     
EXE_CMP_HIP := bin/hip-comp

MODE ?=

all:
	@echo "Building with MODE=$(MODE)"
	$(CC) -o $(EXE_CMP) src/comp_sam.cpp $(HASHFLG) $(BUILDFLG) $(OMPFLG)

ifeq ($(MODE), hip)
	$(HIPCC) -o $(EXE_CMP_HIP) src/cms_hip.cpp -lgomp
else
	$(NVCC) -w -o $(EXE_CMP_CUDA) src/cms_cuda.cu -lgomp -Xcompiler -w
endif

clean:
	rm -rf bin/*comp src/*.o
