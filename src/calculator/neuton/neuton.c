#include "neuton.h"
#include <stdlib.h>

#include "preprocessing/blocks/normalize/normalize.h"

#if (NEUTON_MODEL_FLOAT_SUPPORT == 1)
	#include <math.h>
#endif


#define N_ELEMENTS(arr) (sizeof(arr) / sizeof(arr[0]))
#define MAX_INPUT_FLOAT	0.9999999f

#if (NEUTON_MODEL_HEADER_VERSION < 3)
#define NEUTON_INPUTS_IS_INTEGER 0
#endif


static float modelOutput[NEUTON_MODEL_OUTPUTS_COUNT];
static coeff_t modelAccumulators[NEUTON_MODEL_NEURONS_COUNT];
static uint8_t modelIsReadyForInference = 0;

#if (NEUTON_MODEL_QLEVEL < 32)
static coeff_t ctMax = (1u << NEUTON_MODEL_QLEVEL) - 1;
#endif

#define NEUTON_MODEL_INPUT_TYPE_SIZE ((sizeof(input_t) > sizeof(coeff_t)) ? sizeof(input_t) : sizeof(coeff_t))

#if (NEUTON_PREPROCESSING_ENABLED == 0)

	static uint8_t modelInputBuffer[NEUTON_MODEL_INPUTS_COUNT * NEUTON_MODEL_INPUT_TYPE_SIZE] __attribute__ ((aligned));

#else

	static uint8_t modelInputBuffer[NEUTON_MODEL_INPUTS_COUNT_ORIGINAL * NEUTON_MODEL_WINDOW_SIZE * NEUTON_MODEL_INPUT_TYPE_SIZE] __attribute__ ((aligned));
	static uint16_t modelWindowFill = 0;

	#ifndef NEUTON_MODEL_SA_PRECISION
		#define NEUTON_MODEL_SA_PRECISION 24
	#endif

	#define SA_PRECISION NEUTON_MODEL_SA_PRECISION

	#include "fe/statistical/Common.h"
	#include "fe/statistical/DSP.h"

	#define NEUTON_MODEL_EXTRACTED_FEATURES_TYPE_SIZE ((sizeof(extracted_feature_t) > sizeof(coeff_t)) ? sizeof(extracted_feature_t) : sizeof(coeff_t))
	static uint8_t extractedFeaturesBuffer[EF_COUNT * NEUTON_MODEL_INPUTS_COUNT_ORIGINAL * NEUTON_MODEL_EXTRACTED_FEATURES_TYPE_SIZE] __attribute__ ((aligned));

#endif


inline uint8_t neuton_model_quantization_level()
{
	return NEUTON_MODEL_QLEVEL;
}

inline uint8_t neuton_model_float_calculations()
{
	return NEUTON_MODEL_FLOAT_SUPPORT;
}

inline TaskType neuton_model_task_type()
{
	return (TaskType) NEUTON_MODEL_TASK_TYPE;
}

inline uint16_t neuton_model_outputs_count()
{
	return NEUTON_MODEL_OUTPUTS_COUNT;
}

inline uint16_t neuton_model_neurons_count()
{
	return NEUTON_MODEL_NEURONS_COUNT;
}

inline uint32_t neuton_model_weights_count()
{
	return NEUTON_MODEL_WEIGHTS_COUNT;
}

inline uint16_t neuton_model_inputs_limits_count()
{
	return NEUTON_MODEL_INPUT_LIMITS_COUNT;
}

inline uint16_t neuton_model_inputs_count()
{
#if (NEUTON_PREPROCESSING_ENABLED == 0)

	return NEUTON_MODEL_INPUTS_COUNT;

#else

	return NEUTON_MODEL_INPUTS_COUNT_ORIGINAL;

#endif
}

inline uint16_t neuton_model_window_size()
{
#if (NEUTON_PREPROCESSING_ENABLED == 1)

	return NEUTON_MODEL_WINDOW_SIZE;

#else

	return 1;

#endif
}

uint32_t neuton_model_ram_usage()
{
	return sizeof(modelOutput) + sizeof(modelAccumulators) + sizeof(modelIsReadyForInference)
			+ sizeof(modelInputBuffer)

#if (NEUTON_PREPROCESSING_ENABLED == 1)

			+ sizeof(extractedFeaturesBuffer)
			+ sizeof(modelWindowFill)

#endif
#if (NEUTON_MODEL_QLEVEL < 32)
			+ sizeof(ctMax)
#endif
			;
}

uint32_t neuton_model_size()
{
	return sizeof(modelWeights)	+ sizeof(modelLinks) + sizeof(modelFuncCoeffs)
			+ sizeof(modelIntLinksBoundaries) + sizeof(modelExtLinksBoundaries)
			+ sizeof(modelOutputNeurons)

#if (NEUTON_MODEL_HEADER_VERSION > 1)

			+ sizeof(modelFuncTypes)

#endif
			;
}

uint32_t neuton_model_size_with_meta()
{
	return neuton_model_size()

#if (NEUTON_PREPROCESSING_ENABLED == 0)

			+ sizeof(modelInputMin)	+ sizeof(modelInputMax)

#endif
#if (NEUTON_PREPROCESSING_ENABLED == 1)

	#if (NEUTON_DROP_ORIGINAL_FEATURES != 1)
			+ sizeof(modelInputScaleMin) + sizeof(modelInputScaleMax)
	#endif
			+ sizeof(extractedFeaturesScaleMin) + sizeof(extractedFeaturesScaleMax)

#endif
#if (NEUTON_BITMASK_ENABLED == 1)

			+ sizeof(modelUsedInputsMask)

#endif
#if (NEUTON_MODEL_TASK_TYPE == 2)

			+ sizeof(modelOutputMin) + sizeof(modelOutputMax)

#endif
#if (NEUTON_MODEL_LOG_SCALE_OUTPUTS == 1)

			+ sizeof(modelOutputLogFlag) + sizeof(modelOutputLogScale)

#endif
			;
}

static void denormalize_outputs()
{
#if (NEUTON_MODEL_TASK_TYPE == 0) || (NEUTON_MODEL_TASK_TYPE == 1)

	float sum = 0;

	for (uint16_t i = 0; i < NEUTON_MODEL_OUTPUTS_COUNT; ++i)
		sum += modelOutput[i];

	for (uint16_t i = 0; i < NEUTON_MODEL_OUTPUTS_COUNT; ++i)
		modelOutput[i] = (sum != 0) ? modelOutput[i] / sum: 0;

#endif

#if (NEUTON_MODEL_TASK_TYPE == 2)

	for (uint16_t i = 0; i < NEUTON_MODEL_OUTPUTS_COUNT; ++i)
	{
		modelOutput[i] = modelOutput[i] * (modelOutputMax[i] - modelOutputMin[i]) + modelOutputMin[i];

#if (NEUTON_MODEL_LOG_SCALE_OUTPUTS == 1)

		if (modelOutputLogFlag[i])
			modelOutput[i] = exp(modelOutput[i]) - modelOutputLogScale[i];

#endif

	}
#endif
}

void neuton_model_reset_inputs()
{
#if (NEUTON_PREPROCESSING_ENABLED != 0)

	modelWindowFill = 0;

#endif

	modelIsReadyForInference = 0;
}

int8_t neuton_model_set_inputs(input_t *inputs)
{
	if (!inputs)
		return -1;

	input_t* buffer = (input_t*)modelInputBuffer;

#if (NEUTON_PREPROCESSING_ENABLED == 0)

	for (uint16_t i = 0; i < neuton_model_inputs_count(); ++i)
		buffer[i] = inputs[i];

	modelIsReadyForInference = 1;

	return 0;

#else

	for (uint16_t i = 0; i < NEUTON_MODEL_INPUTS_COUNT_ORIGINAL; ++i)
		buffer[i * NEUTON_MODEL_WINDOW_SIZE + modelWindowFill] = inputs[i];

	if (++modelWindowFill >= NEUTON_MODEL_WINDOW_SIZE)
	{
		modelWindowFill = 0;
		modelIsReadyForInference = 1;

		return 0;
	}

	return 1;

#endif
}

#if (NEUTON_MODEL_QLEVEL == 32)

static inline coeff_t neuton_activation_fn(neurons_size_t neuronIndex, acc_signed_t summ)
{
	return 1.0f / (1.0f + exp((acc_signed_t) ((acc_signed_t) -modelFuncCoeffs[neuronIndex]) * summ));
}

#else // (NEUTON_MODEL_QLEVEL == 32)

#if (NEUTON_MODEL_FLOAT_SUPPORT == 0)

	static coeff_t accurate_fast_sigmoid(acc_signed_t arg)
	{
		coeff_t qResult = 0;
		coeff_t secondPointY = 0;
		coeff_t firstPointY = 0;

		static const uint8_t QLVL = NEUTON_MODEL_QLEVEL;
		static const acc_signed_t CT_MAX_VALUE = 1u << QLVL;
		const acc_signed_t intPart = abs(arg) / (1u << QLVL);
		const acc_signed_t realPart = abs(arg) - (intPart << QLVL);

		if (intPart == 0 && realPart == 0)
		{
			return 1u << (QLVL - 1);
		}

		int s = arg > 0 ? 0 : 1;
		if (realPart == 0)
		{
			for (int i = 0; i < QLVL; i++)
			{
				const uint8_t bit = ((i / intPart + s) % 2);
				qResult = qResult | (bit << (QLVL - i - 1));
			}
			return qResult;
		}

		const acc_signed_t secondPointX = intPart + 1;
		if (intPart == 0)
		{
			firstPointY = 1u << (QLVL - 1);
			for (int i = 0; i < QLVL; i++)
			{
				const uint8_t bit = ((i / secondPointX) % 2);
				secondPointY = secondPointY | (bit << (QLVL - i - 1));
			}
		}
		else
		{
			if (secondPointX == 0)
			{
				for (int i = 0; i < QLVL; i++)
				{
					const uint8_t bit = ((i / intPart) % 2);
					firstPointY = firstPointY | (bit << (QLVL - i - 1));
				}
				secondPointY = 1u << (QLVL - 1);
			}
			else
			{
				for (int i = 0; i < QLVL; i++)
				{
					uint8_t bit = ((i / intPart) % 2);
					firstPointY = firstPointY | (bit << (QLVL - i - 1));
					bit = ((i / secondPointX) % 2);
					secondPointY = secondPointY | (bit << (QLVL - i - 1));
				}
			}
		}

		const acc_signed_t res = (CT_MAX_VALUE - realPart) * firstPointY + realPart * secondPointY;
		if (arg > 0)
		{
			return res >> QLVL;
		}
		else
		{
			qResult = res >> QLVL;
			return qResult == 0 ? CT_MAX_VALUE - 1 : CT_MAX_VALUE - qResult;
		}
	}

#endif // (NEUTON_MODEL_FLOAT_SUPPORT == 0)

static inline float neuton_deqantize_value(coeff_t value)
{
	return (float) value / (float) (1u << NEUTON_MODEL_QLEVEL);
}

#if (NEUTON_MODEL_QLEVEL == 8)
	#define KSHIFT 2
#endif

#if (NEUTON_MODEL_QLEVEL == 16)
	#define KSHIFT 10
#endif

static coeff_t neuton_activation_fn(neurons_size_t neuronIndex, acc_signed_t summ)
{

#if (NEUTON_MODEL_FLOAT_SUPPORT == 1)

	const float qs = (float) (((acc_signed_t) modelFuncCoeffs[neuronIndex] * summ)
			>> (NEUTON_MODEL_QLEVEL + KSHIFT - 1)) / (float) (1u << (NEUTON_MODEL_QLEVEL));

	const float tmpValue = 1.0f / (1.0f + expf(-qs));

	return (tmpValue > MAX_INPUT_FLOAT ? MAX_INPUT_FLOAT : tmpValue) * (float) (1u << NEUTON_MODEL_QLEVEL);

#else // (NEUTON_MODEL_FLOAT_SUPPORT == 1)

	return accurate_fast_sigmoid(
		-(((acc_signed_t) modelFuncCoeffs[neuronIndex] * summ) >> (NEUTON_MODEL_QLEVEL + KSHIFT - 1))
	);

#endif // (NEUTON_MODEL_FLOAT_SUPPORT == 1)

}
#endif // (NEUTON_MODEL_QLEVEL == 32)


#if (NEUTON_BITMASK_ENABLED == 1)
static inline uint8_t is_input_used(uint32_t pos)
{
	return (modelUsedInputsMask[pos / 8] & (1u << (pos % 8)));
}
#endif

#if (NEUTON_DROP_ORIGINAL_FEATURES != 1)

static coeff_t prepare_model_input(input_t value, uint16_t index)
{
	input_t min, max;

#if (NEUTON_PREPROCESSING_ENABLED == 1)

	min = modelInputScaleMin[index / NEUTON_MODEL_WINDOW_SIZE];
	max = modelInputScaleMax[index / NEUTON_MODEL_WINDOW_SIZE];

#else

	#if (NEUTON_MODEL_INPUT_LIMITS_COUNT == 1)

		min = modelInputMin[0];
		max = modelInputMax[0];

	#else

		min = modelInputMin[index];
		max = modelInputMax[index];

	#endif

#endif

#if (NEUTON_MODEL_QLEVEL == 32)

	coeff_t ct = value;
	neuton_preprocessing_block_normalize(&ct, &ct, 1, min, max);
	return ct;

#else

	if (value < min)
		value = min;
	if (value > max)
		value = max;

	#if (NEUTON_INPUTS_IS_INTEGER == 1)

		uint64_t tmp = value - min;

		if ((max - min) == 0)
		{
			return tmp * ctMax;
		}
		else
		{
			return tmp * ctMax / (max - min);
		}

	#else

		if ((max - min) == 0)
		{
			return (value - min) * ctMax;
		}
		else
		{
			return (value - min) * ctMax / (max - min);
		}

	#endif

#endif
}

static void prepare_model_inputs()
{
	input_t* src = (input_t*)modelInputBuffer;
	coeff_t* dst = (coeff_t*)modelInputBuffer;

#if (NEUTON_PREPROCESSING_ENABLED == 0)
	uint16_t count = NEUTON_MODEL_INPUTS_COUNT;
#else
	uint16_t count = NEUTON_MODEL_INPUTS_COUNT_ORIGINAL * NEUTON_MODEL_WINDOW_SIZE;
#endif

	if (sizeof(coeff_t) > sizeof(input_t))
	{
		for (uint16_t i = count-1; ; --i)
		{
		#if (NEUTON_BITMASK_ENABLED == 1)
			if (is_input_used(i))
		#endif
				dst[i] = prepare_model_input(src[i], i);

			if (i == 0) break;
		}
	}
	else
	{
		for (uint16_t i = 0; i < count; ++i)
		{
		#if (NEUTON_BITMASK_ENABLED == 1)
			if (is_input_used(i))
		#endif
				dst[i] = prepare_model_input(src[i], i);
		}
	}
}

#endif // (NEUTON_DROP_ORIGINAL_FEATURES != 1)

#if (NEUTON_PREPROCESSING_ENABLED == 1)

static void extract_features()
{
	input_t* src = (input_t*)modelInputBuffer;
	extracted_feature_t* dst = (extracted_feature_t*)extractedFeaturesBuffer;

	uint16_t windowSize = NEUTON_MODEL_WINDOW_SIZE;
	uint16_t extractedFeaturesCount = EF_COUNT;

#if (NEUTON_BITMASK_ENABLED == 1)
	uint32_t shift = NEUTON_MODEL_INPUTS_COUNT_ORIGINAL * NEUTON_MODEL_WINDOW_SIZE;
	uint32_t pos = 0;

	#if (NEUTON_DROP_ORIGINAL_FEATURES == 1)
		shift = 0;
	#endif
#endif

	for (uint16_t i = 0; i < NEUTON_MODEL_INPUTS_COUNT_ORIGINAL; ++i)
	{
	#if (NEUTON_BITMASK_ENABLED == 1)
		pos = shift + i * extractedFeaturesCount;
		InitDataBuffer(modelUsedInputsMask, pos, src, windowSize);
	#else
		InitDataBuffer(NULL, 0, src, windowSize);
	#endif


	#if (NEUTON_BITMASK_ENABLED == 1)
		if (is_input_used(pos + EF_MIN))
	#endif
		{
			dst[EF_MIN] = Min();
		}
	#if (NEUTON_BITMASK_ENABLED == 1)
		if (is_input_used(pos + EF_MAX))
	#endif
		{
			dst[EF_MAX] = Max();
		}
	#if (NEUTON_BITMASK_ENABLED == 1)
		if (is_input_used(pos + EF_MEAN))
	#endif
		{
			dst[EF_MEAN] = Mean();
		}
	#if (NEUTON_BITMASK_ENABLED == 1)
		if (is_input_used(pos + EF_RMS))
	#endif
		{
			dst[EF_RMS] = RootMeanSquare();
		}
	#if (NEUTON_BITMASK_ENABLED == 1)
		if (is_input_used(pos + EF_SIGN_CHANGES))
	#endif
		{
			dst[EF_SIGN_CHANGES] = CountSignChanges();
		}
	#if (NEUTON_BITMASK_ENABLED == 1)
		if (is_input_used(pos + EF_VARIANCE))
	#endif
		{
			dst[EF_VARIANCE] = Variance();
		}
	#if (NEUTON_BITMASK_ENABLED == 1)
		if (is_input_used(pos + EF_PFD))
	#endif
		{
			dst[EF_PFD] = PetrosianFractalDimension();
		}
	#if (NEUTON_BITMASK_ENABLED == 1)
		if (is_input_used(pos + EF_SKEWNESS))
	#endif
		{
			dst[EF_SKEWNESS] = Skewness();
		}
	#if (NEUTON_BITMASK_ENABLED == 1)
		if (is_input_used(pos + EF_KURTOSIS))
	#endif
		{
			dst[EF_KURTOSIS] = Kurtosis();
		}

		src += windowSize;
		dst += extractedFeaturesCount;
	}
}

static coeff_t prepare_extracted_feature(extracted_feature_t value, uint16_t index)
{
	extracted_feature_t min = extractedFeaturesScaleMin[index];
	extracted_feature_t max = extractedFeaturesScaleMax[index];

#if (NEUTON_MODEL_QLEVEL == 32)

	coeff_t ct = value;
	neuton_preprocessing_block_normalize(&ct, &ct, 1, min, max);
	return ct;

#else

	if (value < min)
		value = min;
	if (value > max)
		value = max;

	#if (NEUTON_INPUTS_IS_INTEGER == 1)

		uint64_t tmp = value - min;

		if ((max - min) == 0)
		{
			return tmp * ctMax;
		}
		else
		{
			return tmp * ctMax / (max - min);
		}

	#else

		if ((max - min) == 0)
		{
			return (value - min) * ctMax;
		}
		else
		{
			return (value - min) * ctMax / (max - min);
		}

	#endif

#endif
}

static void prepare_extracted_features()
{
	extracted_feature_t* src = (extracted_feature_t*)extractedFeaturesBuffer;
	coeff_t* dst = (coeff_t*)extractedFeaturesBuffer;
	uint16_t count = EF_COUNT * NEUTON_MODEL_INPUTS_COUNT_ORIGINAL;

#if (NEUTON_BITMASK_ENABLED == 1)
	uint32_t shift = NEUTON_MODEL_INPUTS_COUNT_ORIGINAL * NEUTON_MODEL_WINDOW_SIZE;

	#if (NEUTON_DROP_ORIGINAL_FEATURES == 1)
		shift = 0;
	#endif
#endif

	if (sizeof(coeff_t) > sizeof(input_t))
	{
		for (uint16_t i = count-1; ; --i)
		{
		#if (NEUTON_BITMASK_ENABLED == 1)
			if (is_input_used(i + shift))
		#endif
				dst[i] = prepare_extracted_feature(src[i], i);

			if (i == 0) break;
		}
	}
	else
	{
		for (uint16_t i = 0; i < count; ++i)
		{
		#if (NEUTON_BITMASK_ENABLED == 1)
			if (is_input_used(i + shift))
		#endif
				dst[i] = prepare_extracted_feature(src[i], i);
		}
	}
}

#endif // (NEUTON_PREPROCESSING_ENABLED == 1)


static inline coeff_t get_model_input(uint16_t index)
{
#if (NEUTON_PREPROCESSING_ENABLED == 0)

	const uint16_t inputsCount = NEUTON_MODEL_INPUTS_COUNT;

#else

	const uint16_t inputsCount = NEUTON_MODEL_INPUTS_COUNT_ORIGINAL * NEUTON_MODEL_WINDOW_SIZE;

#endif

#if (NEUTON_DROP_ORIGINAL_FEATURES == 1)

	index += inputsCount;

#endif

	coeff_t* coeff;


	if (index < inputsCount)
	{
		coeff = (coeff_t*)modelInputBuffer;
		return coeff[index];
	}

#if (NEUTON_PREPROCESSING_ENABLED == 1)

	index -= inputsCount;
	if (index < (EF_COUNT * NEUTON_MODEL_INPUTS_COUNT_ORIGINAL))
	{
		coeff = (coeff_t*)extractedFeaturesBuffer;
		return coeff[index];
	}

#endif

#if (NEUTON_MODEL_QLEVEL == 32)

	return 1.0f;

#else

	return ctMax;

#endif
}


int8_t neuton_model_run_inference(uint16_t *index, float **outputs)
{
	if (!modelIsReadyForInference)
		return 1;

#if (NEUTON_PREPROCESSING_ENABLED == 1)

	extract_features();
	prepare_extracted_features();

#endif

#if (NEUTON_DROP_ORIGINAL_FEATURES != 1)
	prepare_model_inputs();
#endif

	for (neurons_size_t i = 0; i < N_ELEMENTS(modelAccumulators); ++i)
		modelAccumulators[i] = 0;

	weights_size_t weightIndex = 0;
	for (neurons_size_t neuronIndex = 0; neuronIndex < NEUTON_MODEL_NEURONS_COUNT; ++neuronIndex)
	{
		acc_signed_t summ = 0;

		for (; weightIndex < modelIntLinksBoundaries[neuronIndex]; ++weightIndex)
		{
			const acc_signed_t firstValue  = (acc_signed_t) modelWeights[weightIndex];
			const acc_signed_t secondValue = (acc_signed_t) modelAccumulators[modelLinks[weightIndex]];
			summ += firstValue * secondValue;
		}

		for (; weightIndex < modelExtLinksBoundaries[neuronIndex]; ++weightIndex)
		{
			const acc_signed_t firstValue  = (acc_signed_t) modelWeights[weightIndex];
			const acc_signed_t secondValue = (acc_signed_t) get_model_input(modelLinks[weightIndex]);
			summ += firstValue * secondValue;
		}

		modelAccumulators[neuronIndex] = neuton_activation_fn(neuronIndex, summ);
	}

	for (neurons_size_t i = 0; i < N_ELEMENTS(modelOutputNeurons); ++i)
	{
#if (NEUTON_MODEL_QLEVEL == 32)

		modelOutput[i] = modelAccumulators[modelOutputNeurons[i]];

#else

		modelOutput[i] = neuton_deqantize_value(modelAccumulators[modelOutputNeurons[i]]);

#endif

	}


	denormalize_outputs();


#if (NEUTON_MODEL_OUTPUTS_COUNT == 1)

	if (index)
		*index = 0;

#else

	if (index)
	{
		uint16_t target = 0;
		float max = 0.0;

		for (uint16_t i = 0; i < N_ELEMENTS(modelOutput); ++i)
			if (max < modelOutput[i])
			{
				max = modelOutput[i];
				target = i;
			}

		*index = target;
	}

#endif

	if (outputs)
		*outputs = modelOutput;

	return 0;
}
