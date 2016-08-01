#include <SpTOL.h>

const char *sptExplainError(int errcode) {
    // TODO: currently we only have "return -1" to indicate error.
    // Add other error codes and explain them here.
    static const char *explainations[] = {
        "Unknown error"
    };
    (void) errcode; // prevent -Wunused-parameter
    return explainations[0];
}
