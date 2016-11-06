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


