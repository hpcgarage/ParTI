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