#ifndef TIMER_H
#define TIMER_H


#ifdef __cplusplus
extern "C" {
#endif

void timer_init();
void timer_start();
void timer_stop();
unsigned long timer_value_us();

#ifdef __cplusplus
}
#endif

#endif // TIMER_H
