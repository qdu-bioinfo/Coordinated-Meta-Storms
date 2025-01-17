CC:=g++    

OMPFLG=-fopenmp        
HASHFLG=-Wno-deprecated      
BUILDFLG=-w -ffunction-sections -fdata-sections -fmodulo-sched 
OBJ_EXT=src/ExtractRNA.o    

EXE_CMP=bin/comp
EXE_CMP_CUDA=bin/cuda-comp     

tax:$(OBJ_TAX) src/frame.cpp  
	$(CC) -o $(EXE_CMP) src/comp_sam.cpp $(HASHFLG) $(BUILDFLG) $(OMPFLG)   
	nvcc -w -o $(EXE_CMP_CUDA) src/cms_cuda.cu -lgomp -Xcompiler -w 

clean:
	rm -rf bin/*comp src/*.o   
