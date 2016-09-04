#include <SpTOL.h>
#include <stdio.h>

void spt_ComplainError(int errcode, const char *name, const char *file, unsigned line) {
    fprintf(stderr, "[%s] error 0x%08x at %s:%u\n", name, errcode, file, line);
}
