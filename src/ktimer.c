#include "timer.h"

void timer_reset(Timer * const kTimer) {
  kTimer->running       = 0;
  kTimer->seconds       = 0;
  kTimer->Start.tv_sec  = 0;
  kTimer->Start.tv_usec = 0;
  kTimer->Stop.tv_sec   = 0;
  kTimer->Stop.tv_usec  = 0;
}


void timer_start(Timer * const kTimer) {
  kTimer->running = 1;
  gettimeofday(&(kTimer->Start), NULL);
}


void timer_stop(Timer * const kTimer) {
  kTimer->running = 0;
  gettimeofday(&(kTimer->Stop), NULL);
  kTimer->seconds += (double)(kTimer->Stop.tv_sec - kTimer->Start.tv_sec);
  kTimer->seconds += 1e-6 * (kTimer->Stop.tv_usec - kTimer->Start.tv_usec);
}


void timer_fstart(Timer * const kTimer) {
  timer_reset(kTimer);
  timer_start(kTimer);
}