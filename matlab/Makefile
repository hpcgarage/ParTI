all:
	find . -name '*.cu' -print -execdir nvcc --ptx $(NVCCFLAGS) '{}' ';'
