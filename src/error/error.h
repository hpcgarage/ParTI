#ifndef SPTOL_ERROR_H_INCLUDED
#define SPTOL_ERROR_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

#ifndef NDEBUG
#define spt_CheckError(errcode, module, reason) \
    if((errcode) != 0) { \
        spt_ComplainError(module, (errcode), __FILE__, __LINE__, (reason)); \
        return (errcode); \
    }
#else
#define spt_CheckError(errcode, module, reason) \
    if((errcode) != 0) { \
        return (errcode); \
    }
#endif

void spt_ComplainError(const char *module, int errcode, const char *file, unsigned line, const char *reason);

#ifdef __cplusplus
}
#endif

#endif
