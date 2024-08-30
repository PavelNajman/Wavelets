#ifndef DWT_H
#define DWT_H

#include "common.h"

void cdf53_separable_lifting_generic_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC cdf53_separable_lifting_generic (size_t step);

void cdf53_non_separable_lifting_generic_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC cdf53_non_separable_lifting_generic (size_t step);

void cdf53_non_separable_convolution_at_generic_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC cdf53_non_separable_convolution_at_generic(size_t step);

void cdf53_non_separable_convolution_star_generic_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC cdf53_non_separable_convolution_star_generic(size_t step);

void cdf97_separable_lifting_generic_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC cdf97_separable_lifting_generic (size_t step);

void cdf97_non_separable_lifting_generic_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC cdf97_non_separable_lifting_generic (size_t step);

void cdf97_non_separable_convolution_at_generic_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC cdf97_non_separable_convolution_at_generic (size_t step);

void cdf97_non_separable_convolution_star_generic_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC cdf97_non_separable_convolution_star_generic (size_t step);

void cdf97_non_separable_polyconvolution_generic_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC cdf97_non_separable_polyconvolution_generic (size_t step);

void dd137_separable_lifting_generic_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC dd137_separable_lifting_generic (size_t step);

void dd137_non_separable_lifting_generic_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC dd137_non_separable_lifting_generic (size_t step);

void dd137_non_separable_convolution_at_generic_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC dd137_non_separable_convolution_at_generic (size_t step);

void dd137_non_separable_convolution_star_generic_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC dd137_non_separable_convolution_star_generic (size_t step);

void NO_TREE_VECTORIZE haar_single_loop_generic_transform_tile(const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC NO_TREE_VECTORIZE haar_single_loop_generic(size_t step);

#ifdef USE_SSE
void cdf53_separable_lifting_amd64_sse_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC cdf53_separable_lifting_amd64_sse (size_t step);

void cdf53_non_separable_lifting_amd64_sse_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC cdf53_non_separable_lifting_amd64_sse (size_t step);

void cdf53_non_separable_convolution_at_amd64_sse_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC cdf53_non_separable_convolution_at_amd64_sse(size_t step);

void cdf53_non_separable_convolution_star_amd64_sse_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC cdf53_non_separable_convolution_star_amd64_sse(size_t step);

void cdf97_separable_lifting_amd64_sse_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC cdf97_separable_lifting_amd64_sse (size_t step);

void cdf97_non_separable_lifting_amd64_sse_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC cdf97_non_separable_lifting_amd64_sse (size_t step);

void cdf97_non_separable_convolution_at_amd64_sse_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC cdf97_non_separable_convolution_at_amd64_sse (size_t step);

void cdf97_non_separable_convolution_star_amd64_sse_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC cdf97_non_separable_convolution_star_amd64_sse (size_t step);

void cdf97_non_separable_polyconvolution_amd64_sse_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC cdf97_non_separable_polyconvolution_amd64_sse (size_t step);

void dd137_separable_lifting_amd64_sse_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC dd137_separable_lifting_amd64_sse (size_t step);

void dd137_non_separable_lifting_amd64_sse_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC dd137_non_separable_lifting_amd64_sse (size_t step);

void dd137_non_separable_convolution_at_amd64_sse_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC dd137_non_separable_convolution_at_amd64_sse (size_t step);

void dd137_non_separable_convolution_star_amd64_sse_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC dd137_non_separable_convolution_star_amd64_sse (size_t step);

void NO_TREE_VECTORIZE haar_single_loop_amd64_sse_transform_tile(const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC NO_TREE_VECTORIZE haar_single_loop_amd64_sse(size_t step);
#endif

#ifdef USE_AVX_512
void cdf53_separable_lifting_avx_512_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC cdf53_separable_lifting_avx_512 (size_t step);

void cdf53_non_separable_lifting_avx_512_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC cdf53_non_separable_lifting_avx_512 (size_t step);

void cdf53_non_separable_convolution_at_avx_512_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC cdf53_non_separable_convolution_at_avx_512(size_t step);

void cdf53_non_separable_convolution_star_avx_512_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC cdf53_non_separable_convolution_star_avx_512(size_t step);

void cdf97_separable_lifting_avx_512_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC cdf97_separable_lifting_avx_512 (size_t step);

void cdf97_non_separable_lifting_avx_512_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC cdf97_non_separable_lifting_avx_512 (size_t step);

void cdf97_non_separable_convolution_at_avx_512_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC cdf97_non_separable_convolution_at_avx_512(size_t step);

void cdf97_non_separable_convolution_star_avx_512_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC cdf97_non_separable_convolution_star_avx_512(size_t step);

void cdf97_non_separable_polyconvolution_avx_512_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC cdf97_non_separable_polyconvolution_avx_512(size_t step);

void dd137_separable_lifting_avx_512_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC dd137_separable_lifting_avx_512 (size_t step);

void dd137_non_separable_lifting_avx_512_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC dd137_non_separable_lifting_avx_512 (size_t step);

void dd137_non_separable_convolution_at_avx_512_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC dd137_non_separable_convolution_at_avx_512(size_t step);

void dd137_non_separable_convolution_star_avx_512_transform_tile (const TransformStepArguments * tsa);
TRANSFORM_STEP_FUNC dd137_non_separable_convolution_star_avx_512(size_t step);
#endif

#endif	// DWT_H
