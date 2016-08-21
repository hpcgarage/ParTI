CFLAGS=-Wall -Wextra -std=c99 -fopenmp -g -I../include
LDLIBS=../build/libSpTOL_s.a
NVCC=nvcc
NVCCFLAGS=-I../include

TARGETS=loadstore add kronecker khatri-rao convert ttm

.PHONY: all clean

all: ../build/libSpTOL_s.a $(TARGETS)

clean:
	make -C ../build clean
	rm -f $(TARGETS)

%: %.c ../build/libSpTOL_s.a

%: %.cu ../build/libSpTOL_s.a
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDLIBS)

../build/libSpTOL_s.a:
	mkdir -p ../build
	cd ../build && \
	cmake .. -DCMAKE_C_FLAGS="-Wall -Wextra -std=c99 -fopenmp -g -fPIC" -DCMAKE_CXX_FLAGS="-Wall -Wextra -fopenmp -g -fPIC" -DCUDA_NVCC_FLAGS="-g" && \
	$(MAKE)
