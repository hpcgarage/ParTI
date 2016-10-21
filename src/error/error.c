#include <SpTOL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "error.h"

static __thread struct {
    const char *module;
    int errcode;
    const char *file;
    unsigned line;
    char *reason;
} g_last_error = { NULL, 0, NULL, 0, NULL };

void spt_ComplainError(const char *module, int errcode, const char *file, unsigned line, const char *reason) {
    g_last_error.module = module;
    g_last_error.errcode = errcode;
    g_last_error.file = file;
    g_last_error.line = line;
    free(g_last_error.reason);
    if(reason) {
        size_t len = strlen(reason);
        g_last_error.reason = malloc(len+1);
        if(!g_last_error.reason) {
            abort();
        }
        memcpy(g_last_error.reason, reason, len+1);
        fprintf(stderr, "[%s] error 0x%08x at %s:%u, %s\n", module, errcode, file, line, reason);
    } else {
        fprintf(stderr, "[%s] error 0x%08x at %s:%u\n", module, errcode, file, line);
    }
}

int sptGetLastError(const char **module, const char **file, unsigned *line, const char **reason) {
    if(module) {
        *module = g_last_error.module;
    }
    if(file) {
        *file = g_last_error.file;
    }
    if(line) {
        *line = g_last_error.line;
    }
    if(reason) {
        *reason = g_last_error.reason;
    }
    return g_last_error.errcode;
}

void sptClearLastError(void) {
    g_last_error.module = NULL;
    g_last_error.errcode = 0;
    g_last_error.file = NULL;
    g_last_error.line = 0;
    free(g_last_error.reason);
    g_last_error.reason = NULL;
}
