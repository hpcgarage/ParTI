#include <SpTOL.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

struct sptTagTimer {
    int use_cuda;
    struct timespec start_timespec;
    struct timespec stop_timespec;
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
};

int sptNewTimer(sptTimer *timer, int use_cuda) {
    *timer = (sptTimer) malloc(sizeof **timer);
    (*timer)->use_cuda = use_cuda;
    if(use_cuda) {
        int result;
        result = cudaEventCreate(&(*timer)->start_event);
        if(result) {
            return result;
        }
        result = cudaEventCreate(&(*timer)->stop_event);
        if(result) {
            return result;
        }
    }
    return 0;
}

int sptStartTimer(sptTimer timer) {
    if(timer->use_cuda) {
        int result;
        result = cudaEventRecord(timer->start_event);
        if(result) {
            return result;
        }
        result = cudaEventSynchronize(timer->start_event);
        if(result) {
            return result;
        }
    } else {
        return clock_gettime(CLOCK_MONOTONIC, &timer->start_timespec);
    }
    return 0;
}

int sptStopTimer(sptTimer timer) {
    if(timer->use_cuda) {
        int result;
        result = cudaEventRecord(timer->stop_event);
        if(result) {
            return result;
        }
        result = cudaEventSynchronize(timer->stop_event);
        if(result) {
            return result;
        }
    } else {
        return clock_gettime(CLOCK_MONOTONIC, &timer->stop_timespec);
    }
    return 0;
}

double sptElapsedTime(const sptTimer timer) {
    if(timer->use_cuda) {
        float elapsed;
        if(cudaEventElapsedTime(&elapsed, timer->start_event, timer->stop_event) != 0) {
            return NAN;
        }
        return elapsed * 1e-3;
    } else {
        return timer->stop_timespec.tv_sec - timer->start_timespec.tv_sec
            + (timer->stop_timespec.tv_nsec - timer->start_timespec.tv_nsec) * 1e-9;
    }
}

double sptPrintElapsedTime(const sptTimer timer, const char *name) {
    double elapsed_time = sptElapsedTime(timer);
    fprintf(stderr, "[%s] operation took %.9lf s\n", name, elapsed_time);
    return elapsed_time;
}

int sptFreeTimer(sptTimer timer) {
    if(timer->use_cuda) {
        int result;
        result = cudaEventDestroy(timer->start_event);
        if(result) {
            return result;
        }
        result = cudaEventDestroy(timer->stop_event);
        if(result) {
            return result;
        }
    }
    free(timer);
    return 0;
}
