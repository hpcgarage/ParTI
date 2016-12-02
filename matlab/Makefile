.PHONY: all clean all.mexa64 all.mexmaci64 all.mexw64

all: all.mexa64

clean:
	find -name '*.mex*' -exec rm '{}' +

all.mexa64:
	find -name '*.c' | sed 's/\.c$$/.mexa64/' | xargs $(MAKE)

all.mexmaci64:
	find -name '*.c' | sed 's/\.c$$/.mexmaci64/' | xargs $(MAKE)

all.mexw64:
	find -name '*.c' | sed 's/\.c$$/.mexw64/' | xargs $(MAKE)

%.mexa64: %.c
	mex $(MEXFLAGS) CFLAGS='$$CFLAGS -std=gnu99 -Wall' -I../include -L../build -outdir "$$(dirname "$<")" "$<" -lParTI

%.mexmaci64: %.c
	mex $(MEXFLAGS) CFLAGS='$$CFLAGS -std=gnu99 -Wall' -I../include -L../build -outdir "$$(dirname "$<")" "$<" -lParTI

%.mexw64: %.c
	mex $(MEXFLAGS) CFLAGS='$$CFLAGS -std=gnu99 -Wall' -I../include -L../build -outdir "$$(dirname "$<")" "$<" -lParTI
