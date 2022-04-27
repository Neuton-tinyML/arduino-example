#ifndef NEUTON_MODEL_MODEL_H
#define NEUTON_MODEL_MODEL_H

#ifdef __cplusplus
extern "C"
{
#endif

/* Model info */
#define NEUTON_MODEL_HEADER_VERSION 3
#define NEUTON_MODEL_QLEVEL 8
#define NEUTON_MODEL_FLOAT_SUPPORT 0
#define NEUTON_MODEL_TASK_TYPE 1  // binary classification
#define NEUTON_MODEL_NEURONS_COUNT 4
#define NEUTON_MODEL_WEIGHTS_COUNT 16
#define NEUTON_MODEL_INPUTS_COUNT 7
#define NEUTON_MODEL_INPUT_LIMITS_COUNT 7
#define NEUTON_MODEL_OUTPUTS_COUNT 2
#define NEUTON_MODEL_LOG_SCALE_OUTPUTS 0
#define NEUTON_MODEL_HAS_CLASSES_RATIO 0
#define NEUTON_MODEL_HAS_NEGPOS_RATIO 0

/* Preprocessing */
#define NEUTON_PREPROCESSING_ENABLED 0
#define NEUTON_DROP_ORIGINAL_FEATURES 0
#define NEUTON_BITMASK_ENABLED 1
#define NEUTON_INPUTS_IS_INTEGER 0
#define NEUTON_MODEL_SA_PRECISION 24

/* Types */
typedef float input_t;
typedef float extracted_feature_t;
typedef uint8_t coeff_t;
typedef int8_t weight_t;
typedef int32_t acc_signed_t;
typedef uint32_t acc_unsigned_t;
typedef uint8_t sources_size_t;
typedef uint8_t weights_size_t;
typedef uint8_t neurons_size_t;

/* Limits */
static const input_t modelInputMin[] = {
	4.4200001, 1.74, 2.24, 1.1799999, 2.0599999, 4.1700001, 55.099998 };
static const input_t modelInputMax[] = {
	27.549999, 42.560001, 29.870001, 49.380001, 100.52, 17.120001, 83.5 };

static const uint8_t modelUsedInputsMask[] = { 0x7f };

/* Structure */
static const weight_t modelWeights[] = {
	-1, 126, -128, -92, 29, -5, -128, 31, 124, 61, -2, -65, 61, -65, 125, -52 };

static const sources_size_t modelLinks[] = {
	0, 1, 2, 3, 4, 7, 0, 7, 0, 0, 3, 5, 6, 7, 2, 7 };

static const weights_size_t modelIntLinksBoundaries[] = { 0, 7, 9, 15 };
static const weights_size_t modelExtLinksBoundaries[] = { 6, 8, 14, 16 };

static const coeff_t modelFuncCoeffs[] = { 160, 155, 158, 148 };
static const uint8_t modelFuncTypes[] = { 0, 0, 0, 0 };

static const neurons_size_t modelOutputNeurons[] = { 1, 3 };

#ifdef __cplusplus
}
#endif

#endif // NEUTON_MODEL_MODEL_H

