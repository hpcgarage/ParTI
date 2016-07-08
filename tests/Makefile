CFLAGS=-Wall -Werror -g -I../include
LIBS=../build/libSpTOL_s.a

.PHONY: all clean ../build/libSpTOL_s.a

all: loadstore

clean:
	make -C ../build clean
	rm loadstore

loadstore: loadstore.c ../build/libSpTOL_s.a

../build/libSpTOL_s.a:
	mkdir -p ../build
	cmake -E chdir ../build cmake .. -DCMAKE_C_FLAGS="-Wall -Werror -g"
	make -C ../build
