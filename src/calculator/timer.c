#include <stdlib.h>
#include <stdint.h>

#include <Arduino.h>


typedef struct
{
	unsigned long start;
	unsigned long stop;
}
Timer;


static Timer timer = { 0 };


void timer_init()
{
}


void timer_start()
{
	timer.start = micros();
}


void timer_stop()
{
	timer.stop = micros();
}

unsigned long timer_value_us()
{
	return timer.stop - timer.start;
}
