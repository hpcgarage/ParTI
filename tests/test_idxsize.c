#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ParTI.h>

int main(int argc, char ** argv)
{
	int niters = 30;
	uint8_t len_bits = atoi(argv[1]);
	uint64_t len = (uint64_t)pow(2, len_bits);
	printf("len: %lu\n", len);
	float * vec = (float*)malloc(len * sizeof(*vec));
	float * vec_res = (float*)malloc(len * sizeof(*vec_res));
	for(uint64_t i=0; i<len; ++i) {
		vec[i] = 1;
		vec_res[i] = 0;
	}
	float tmp;
	sptTimer timer;

	uint64_t nblocks = len / 128;
	printf("nblocks: %lu\n", nblocks);
	float * block_vec, * block_vec_res;

	/* 8-bit index */
  sptNewTimer(&timer, 0);
  sptStartTimer(timer);
	
	for(int it=0; it<niters; ++it) {
		for(uint32_t b=0; b<nblocks; ++b) {
			block_vec = vec + b * 128;
			block_vec_res = vec_res + b * 128;
			for(uint8_t i=0; i<128; ++i) {
				block_vec_res[i] = block_vec[i];
			}
		}
	}

  sptStopTimer(timer);
  sptPrintAverageElapsedTime(timer, niters, "Vec Uint8");
  sptFreeTimer(timer);

  for(uint64_t i=0; i<len; ++i) {
  	if(vec_res[i] != vec[i]) {
  		printf("[Vec Uint8] Wrong results.\n");
  		break;
  	}
  }
	for(uint64_t i=0; i<len; ++i) {
		vec_res[i] = 0;
	}


  /* 16-bit index */
	nblocks = len / 32768;
	printf("nblocks: %lu\n", nblocks);

  sptNewTimer(&timer, 0);
  sptStartTimer(timer);
	
	for(int it=0; it<niters; ++it) {
		for(uint32_t b=0; b<nblocks; ++b) {
			block_vec = vec + b * 32768;
			block_vec_res = vec_res + b * 32768;
			for(uint16_t i=0; i<32768; ++i) {
				block_vec_res[i] = block_vec[i];
			}
		}
	}

  sptStopTimer(timer);
  sptPrintAverageElapsedTime(timer, niters, "Vec Uint16");
  sptFreeTimer(timer);

  for(uint64_t i=0; i<len; ++i) {
  	if(vec_res[i] != vec[i]) {
  		printf("[Vec Uint8] Wrong results.\n");
  		break;
  	}
  }
	for(uint64_t i=0; i<len; ++i) {
		vec_res[i] = 0;
	}
  
  /* 32-bit index */
  sptNewTimer(&timer, 0);
  sptStartTimer(timer);
	
	for(int it=0; it<niters; ++it) {
		for(uint32_t i=0; i<len; ++i) {
			vec_res[i] = vec[i];
		}
	}

  sptStopTimer(timer);
  sptPrintAverageElapsedTime(timer, niters, "Vec Uint32");
  sptFreeTimer(timer);

  /* 64-bit index */
  for(uint64_t i=0; i<len; ++i) {
  	if(vec_res[i] != vec[i]) {
  		printf("[Vec Uint32] Wrong results.\n");
  		break;
  	}
  }
	for(uint64_t i=0; i<len; ++i) {
		vec_res[i] = 0;
	}

  sptNewTimer(&timer, 0);
  sptStartTimer(timer);
	
	for(int it=0; it<niters; ++it) {
		for(uint64_t i=0; i<len; ++i) {
			vec_res[i] = vec[i];
		}
	}

  sptStopTimer(timer);
  sptPrintAverageElapsedTime(timer, niters, "Vec Uint64");
  sptFreeTimer(timer);

  for(uint64_t i=0; i<len; ++i) {
  	if(vec_res[i] != vec[i]) {
  		printf("[Vec Uint64] Wrong results.\n");
  		break;
  	}
  }

	return 0;

}