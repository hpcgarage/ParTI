#include <sys/time.h>
#include <stddef.h>
#include <stdio.h>


typedef struct {
  int running;
  double seconds;
  struct timeval Start;
  struct timeval Stop;
} Timer;


void timer_reset(Timer * const kTimer);

void timer_start(Timer * const kTimer);

void timer_stop(Timer * const kTimer);

void timer_fstart(Timer * const kTimer);


