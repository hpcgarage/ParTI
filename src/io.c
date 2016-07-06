SparseTensorCOO * ReadSparseTensorFromFile(char const * const fname);
SparseTensorCOO * ReadSparseTensorFromFileStream(FILE * fin);


SparseTensorCOO * ReadSparseTensorFromFileStream(FILE * fin) {
  char * ptr = NULL;

  /* first count nnz in tensor */
  IndexType nnz = 0;
  IndexType nmodes = 0;

  IndexType dims[MAX_NMODES];
  GetSparseTensorShape(fin, &nmodes, &nnz, dims);

  if(nmodes > MAX_NMODES) {
    fprintf(stderr, "ERROR: maximum %"PF_INDEX" modes supported. "
                    "Found %"PF_INDEX". Please recompile with "
                    "MAX_NMODES=%"PF_INDEX".\n", MAX_NMODES, nmodes, nmodes);
    return NULL;
  }

  SparseTensorCOO * spten = AllocateSparseTensor(nnz, nmodes);
  memcpy(spten->dims, dims, nmodes * sizeof(*dims));

  char * line = NULL;
  int64_t read;
  size_t len = 0;

  /* fill in tensor data */
  rewind(fin);
  nnz = 0;
  while((read = getline(&line, &len, fin)) != -1) {
    /* skip empty and commented lines */
    if(read > 1 && line[0] != '#') {
      ptr = line;
      for(IndexType m=0; m < nmodes; ++m) {
        spten->inds[m][nnz] = strtoull(ptr, &ptr, 10) - 1;
      }
      spten->vals[nnz++] = strtod(ptr, &ptr);
    }
  }

  free(line);

  return spten;
}


SparseTensorCOO * ReadSparseTensorFromFile(char const * const fname) {
  FILE * fin;
  if((fin = fopen(fname, "r")) == NULL) {
    fprintf(stderr, "ERROR: failed to open '%s'\n", fname);
    return NULL;
  }

  timer_start(&g_timers[TIMER_IO]);
  SparseTensorCOO * spten = ReadSparseTensorFromFileStream(fin);
  timer_stop(&g_timers[TIMER_IO]);
  fclose(fin);
  return spten;
}