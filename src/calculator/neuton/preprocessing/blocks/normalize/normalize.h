#ifndef NEUTON_PREPROCESSING_BLOCKS_NORMALIZE_H
#define NEUTON_PREPROCESSING_BLOCKS_NORMALIZE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

///
/// \brief Normalize float array to range 0..1
/// \param src - array to be normalized
/// \param dst - destination array to store normalized values (may be same as @src)
/// \param count - elements count in array
/// \param min - minimum value
/// \param max - maximum value
///
void neuton_preprocessing_block_normalize(const float* src, float *dst, size_t count, float min, float max);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // NEUTON_PREPROCESSING_BLOCKS_NORMALIZE_H
