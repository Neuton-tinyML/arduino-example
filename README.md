# How to integrate Neuton into your firmware project 

##	Extract files from archive

Copy all files from archive from platform to `src/calculator/neuton`.

## Make inference

To make prediction you should call two functions:
* `neuton_model_set_inputs` to pass input features;
* `neuton_model_run_inference` to run inference.

It is important that features are in same order as on model training.

``` C
float sample[] = {
  846,    // PT08.S3(NOx)
  1638,   // PT08.S4(NO2)
  991,    // PT08.S5(O3)
  11.3,   // T
  78.1,   // RH
  1.0411, // AH
};
	
if (neuton_model_set_inputs(sample) == 0)
{
  uint16_t predictedClass;
  float* probabilities;
  
  if (neuton_model_run_inference(&predictedClass, &probabilities) == 0)
  {
    if (neuton_model_task_type() == TASK_BINARY_CLASSIFICATION && neuton_model_outputs_count() >= 2)
    {
      if (probabilities[predictedClass] > 0.5)
      {
        if (predictedClass == 0)
        {
          led_green(1);
          led_red(0);
        }
        else
        {
          led_green(0);
          led_red(1);
        }
      }
      else
      {
        led_green(0);
        led_red(0);
      }
    }
  }
}
```
