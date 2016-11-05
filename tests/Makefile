CFLAGS=-Wall -Wextra -std=c99 -fopenmp -I../include -O3
LDLIBS=../build/libSpTOL_s.a -lm
NVCC=nvcc
NVCCFLAGS=-gencode arch=compute_35,code=sm_35 -I../include -Xcompiler -fopenmp -g

TARGETS=reset loadstore add kronecker khatri-rao convert ttm sttm ttm_new ttm_test ttm_omptest mttkrp dmul

.PHONY: all clean

all: ../build/libSpTOL_s.a $(TARGETS)

clean:
	make -C ../build clean
	rm -f $(TARGETS)

%: %.c ../build/libSpTOL_s.a
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

%: %.cu ../build/libSpTOL_s.a
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDLIBS)

../build/libSpTOL_s.a:
	mkdir -p ../build
	cd ../build && \
	cmake .. -DCMAKE_C_FLAGS="-Wall -Wextra -std=c99 -fopenmp -g -fPIC" -DCMAKE_CXX_FLAGS="-Wall -Wextra -fopenmp -g -fPIC" -DCUDA_NVCC_FLAGS="-gencode arch=compute_35,code=sm_35 -g" && \
	$(MAKE) #libSpTOL_s.a
