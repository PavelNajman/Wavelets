#ifndef COMMON_AMD64_SSE_H
#define COMMON_AMD64_SSE_H
#include <stdio.h>

#ifdef USE_SSE

#include <xmmintrin.h>
#include <smmintrin.h>

#ifdef USE_FMA
    #include <immintrin.h>
#endif

/*
 | ----------- | ----------- | ----------- | --------- | --------- | --------- | --------- |
 | LX03L HX03L | LX02L HX02L | LX01L HX01L | LX00 HX00 | LX01 HX01 | LX02 HX02 | LX03 HX03 |
 | LX03L HX03L | LX02L HX02L | LX01L HX01L | LX00 HX00 | LX01 HX01 | LX02 HX02 | LX03 HX03 |
 | ----------- | ----------- | ----------- | --------- | --------- | --------- | --------- |
 | LX13L HX13L | LX12L HX12L | LX11L HX11L | LX10 HX10 | LX11 HX11 | LX12 HX12 | LX13 HX13 |
 | LX13L HX13L | LX12L HX12L | LX11L HX11L | LX10 HX10 | LX11 HX11 | LX12 HX12 | LX13 HX13 |
 | ----------- | ----------- | ----------- | --------- | --------- | --------- | --------- |
 */

__m128 mul_add(__m128 m1, __m128 m2, __m128 m3);

__m128 reverse(__m128 MM);

__m128 rotate_right(__m128 MM);

__m128 rotate_left(__m128 MM);

__m128 load_packed_LX00(__m128 C0, __m128 C1);

__m128 load_packed_HX00(__m128 C0, __m128 C1);

__m128 load_packed_LX01(__m128 LX00, __m128 C1);

__m128 load_packed_LX02(__m128 LX01, __m128 C1);

__m128 load_packed_LX03(__m128 LX02, __m128 C2);

__m128 load_packed_HX01(__m128 HX00, __m128 C1);

__m128 load_packed_HX02(__m128 HX01, __m128 C1);

__m128 load_packed_HX03(__m128 HX02, __m128 C2);

__m128 load_packed_LX01L(__m128 LX00, __m128 C1L);

__m128 load_packed_LX02L(__m128 LX01, __m128 C1L);

__m128 load_packed_LX03L(__m128 LX02, __m128 C2L);

__m128 load_packed_HX01L(__m128 HX00, __m128 C1L);

__m128 load_packed_HX02L(__m128 HX01, __m128 C1L);

__m128 load_packed_HX03L(__m128 HX02, __m128 C2L);

__m128 load_unpacked_XX01L(__m128 XX00, __m128 XX01L);

__m128 load_unpacked_XX02L(__m128 XX01L_, __m128 XX01L);

__m128 load_unpacked_XX03L(__m128 XX02L, __m128 XX01L);

__m128 load_unpacked_XX01(__m128 XX00, __m128 XX01);

__m128 load_unpacked_XX02(__m128 XX01_, __m128 XX01);

__m128 load_unpacked_XX03(__m128 XX02, __m128 XX01);

__m128 load_unpacked_HX01LS(__m128 XX00);

__m128 load_unpacked_HX02LS(__m128 XX00);

__m128 load_unpacked_HX03LS(__m128 XX00);

__m128 load_unpacked_LX01LS(__m128 XX00);

__m128 load_unpacked_LX02LS(__m128 XX00);

__m128 load_unpacked_LX03LS(__m128 XX00);

__m128 load_unpacked_LX01S(__m128 XX00);

__m128 load_unpacked_LX02S(__m128 XX00);

__m128 load_unpacked_LX03S(__m128 XX00);

__m128 load_unpacked_HX01S(__m128 XX00);

__m128 load_unpacked_HX02S(__m128 XX00);

__m128 load_unpacked_HX03S(__m128 XX00);

void print__m128(__m128 A);

#endif /* COMMON_AMD64_SSE_H */
#endif
