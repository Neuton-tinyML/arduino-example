#ifndef NEUTON_H
#define NEUTON_H

#include <stdint.h>
#include "model/model.h"

#ifdef __cplusplus
extern "C"
{
#endif


#if (NEUTON_MODEL_HEADER_VERSION < 3)
typedef float input_t;
#endif

///
/// \brief Get element count of array that you should pass to neuton_model_set_inputs() function
/// \return Array elements count
///
uint16_t neuton_model_inputs_count();

///
/// \brief Set input values
/// \param inputs - input_t[] array of neuton_model_inputs_count() elements
/// \return Zero if model ready for prediction. Result < 0 indicates error, result > 0 - model not ready for prediction.
///
int8_t neuton_model_set_inputs(input_t* inputs);

///
/// \brief Reset input values
///
void neuton_model_reset_inputs();

///
/// \brief Get element count of array that neuton_model_run_inference() returns
/// \return Array elements count
///
uint16_t neuton_model_outputs_count();

///
/// \brief Make a prediction
/// \param index - pointer to predicted class variable (binary/multi classification). Can be NULL.
/// \param outputs - float[] array of neuton_model_outputs_count() elements, contains predicted target variable
///                  (for regression task) or probabilities of each class (binary/multi classification).
/// \return Zero on successful prediction. Result > 0 - model not ready for prediction.
///
int8_t neuton_model_run_inference(uint16_t* index, float** outputs);

///
/// \brief Task types
///
typedef enum
{
	TASK_MULTICLASS_CLASSIFICATION = 0,
	TASK_BINARY_CLASSIFICATION     = 1,
	TASK_REGRESSION                = 2
}
TaskType;

///
/// \brief Get task type
/// \return Task type value
///
TaskType neuton_model_task_type();

///
/// \brief Get model quantization level
/// \return Quantization level (possible values: 8, 16, 32)
///
uint8_t neuton_model_quantization_level();

///
/// \brief Get float support flag
/// \return Flag value (possible values: 0, 1)
///
uint8_t neuton_model_float_calculations();

///
/// \brief Get model neurons count
/// \return Neurons count
///
uint16_t neuton_model_neurons_count();

///
/// \brief Get model weights count
/// \return Weights count
///
uint32_t neuton_model_weights_count();

///
/// \brief Get element count of input normalization array
/// \return Array elements count
///
uint16_t neuton_model_inputs_limits_count();

///
/// \brief Get window size
/// \return Window size
///
uint16_t neuton_model_window_size();

///
/// \brief Get model RAM usage
/// \return RAM usage in bytes
///
uint32_t neuton_model_ram_usage();

///
/// \brief Get model size
/// \return Model size without meta information
///
uint32_t neuton_model_size();

///
/// \brief Get model & meta information size
/// \return Model size with meta information
///
uint32_t neuton_model_size_with_meta();

#ifdef __cplusplus
}
#endif

#endif // NEUTON_H
