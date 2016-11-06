/*
    This file is part of SpTOL.

    SpTOL is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    SpTOL is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with SpTOL.
    If not, see <http://www.gnu.org/licenses/>.
*/

#include <SpTOL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "error.h"

/**
 * Global variables to store last error code and information
 */
static __thread struct {
    const char *module;
    int errcode;
    const char *file;
    unsigned line;
    char *reason;
} g_last_error = { NULL, 0, NULL, 0, NULL };

/**
 * Set last error information as specified and print the error message.
 * Should not be called directly, use the macro `spt_CheckError`.
 */
void spt_ComplainError(const char *module, int errcode, const char *file, unsigned line, const char *reason) {
    g_last_error.errcode = errcode;
    g_last_error.module = module;
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
    }
    if(g_last_error.reason && g_last_error.reason[0] != '\0') {
        fprintf(stderr, "[%s] error 0x%08x at %s:%u, %s\n",
            g_last_error.module,
            g_last_error.errcode,
            g_last_error.file,
            g_last_error.line,
            g_last_error.reason
        );
    } else {
        fprintf(stderr, "[%s] error 0x%08x at %s:%u\n",
            g_last_error.module,
            g_last_error.errcode,
            g_last_error.file,
            g_last_error.line
        );
    }
}

/**
 * Get the last error code and message.
 * @param[out] module store the module name of the last error
 * @param[out] file   store the C source name of the last error
 * @param[out] line   store the line number of the last error
 * @param[out] reason store the human readable error reason
 * @return the error code of the last error
 */
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


/**
 * Clear the information of the last error.
 * Usually called before a procedure.
 */
void sptClearLastError(void) {
    g_last_error.module = NULL;
    g_last_error.errcode = 0;
    g_last_error.file = NULL;
    g_last_error.line = 0;
    free(g_last_error.reason);
    g_last_error.reason = NULL;
}
