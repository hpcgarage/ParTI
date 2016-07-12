CFLAGS=-Wall -Werror -std=c99 -g -I../include
LDLIBS=../build/libSpTOL_s.a

TARGETS=loadstore add kronecker khatri-rao

.PHONY: all clean

all: ../build/libSpTOL_s.a $(TARGETS)

clean:
	make -C ../build clean
	rm -f $(TARGETS)

%: %.c ../build/libSpTOL_s.a

../build/libSpTOL_s.a:
	mkdir -p ../build
	cd ../build && \
	cmake .. -DCMAKE_C_FLAGS="-Wall -Werror -std=c99 -fopenmp -g" && \
	$(MAKE)
