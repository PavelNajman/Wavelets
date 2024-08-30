#include "common-amd64-sse.h"

#ifdef USE_SSE

extern inline __m128 mul_add(__m128 m1, __m128 m2, __m128 m3)
{
#ifdef USE_FMA
    return _mm_fmadd_ps(m1, m2, m3);
#else
    return _mm_add_ps(_mm_mul_ps(m1, m2), m3);
#endif
}

extern inline __m128 reverse(__m128 MM)
{
    return _mm_shuffle_ps(MM, MM, _MM_SHUFFLE(0, 1, 2, 3));
}

extern inline __m128 rotate_right(__m128 MM)
{
    return _mm_shuffle_ps(MM, MM, _MM_SHUFFLE(0, 3, 2, 1));
}

extern inline __m128 rotate_left(__m128 MM)
{
    return _mm_shuffle_ps(MM, MM, _MM_SHUFFLE(2, 1, 0, 3));
}

// unpacks LL00 or LH00
extern inline __m128 load_packed_LX00(__m128 C0, __m128 C1)
{
    return _mm_shuffle_ps(C0, C1, _MM_SHUFFLE(2, 0, 2, 0));
}

// unpacks HL00 or HH00
extern inline __m128 load_packed_HX00(__m128 C0, __m128 C1)
{
    return _mm_shuffle_ps(C0, C1, _MM_SHUFFLE(3, 1, 3, 1));
}

// unpacks LL01 or LH01
extern inline __m128 load_packed_LX01(__m128 LX00, __m128 C1)
{
    return _mm_insert_ps(rotate_right(LX00), C1, 0x30);
}

// unpacks LL02 or LH02
extern inline __m128 load_packed_LX02(__m128 LX01, __m128 C1)
{
    return _mm_insert_ps(rotate_right(LX01), C1, 0xB0);
}

// unpacks LL03 or LH03
extern inline __m128 load_packed_LX03(__m128 LX02, __m128 C2)
{
    return _mm_insert_ps(rotate_right(LX02), C2, 0x30);
}

// unpacks HL01 or HH01
extern inline __m128 load_packed_HX01(__m128 HX00, __m128 C1)
{
    return _mm_insert_ps(rotate_right(HX00), C1, 0x70);
}

// unpacks HL02 or HH02
extern inline __m128 load_packed_HX02(__m128 HX01, __m128 C1)
{
    return _mm_insert_ps(rotate_right(HX01), C1, 0xF0);
}

// unpacks HL03 or HH03
extern inline __m128 load_packed_HX03(__m128 HX02, __m128 C2)
{
    return _mm_insert_ps(rotate_right(HX02), C2, 0x70);
}

// unpacks LL01L or LH01L
extern inline __m128 load_packed_LX01L(__m128 LX00, __m128 C1L)
{
    return _mm_insert_ps(rotate_left(LX00), C1L, 0x80);
}

// unpacks LL02L or LH02L
extern inline __m128 load_packed_LX02L(__m128 LX01, __m128 C1L)
{
    return _mm_insert_ps(rotate_left(LX01), C1L, 0x00);
}

// unpacks LL03 or LH03L
extern inline __m128 load_packed_LX03L(__m128 LX02, __m128 C2L)
{
    return _mm_insert_ps(rotate_left(LX02), C2L, 0x80);
}

// unpacks HL01L or HH01L
extern inline __m128 load_packed_HX01L(__m128 HX00, __m128 C1L)
{
    return _mm_insert_ps(rotate_left(HX00), C1L, 0xC0);
}

// unpacks HL02L or HH02L
extern inline __m128 load_packed_HX02L(__m128 HX01, __m128 C1L)
{
    return _mm_insert_ps(rotate_left(HX01), C1L, 0x40);
}

// unpacks HL03L or HH03L
extern inline __m128 load_packed_HX03L(__m128 HX02, __m128 C2L)
{
    return _mm_insert_ps(rotate_left(HX02), C2L, 0xC0);
}

extern inline __m128 load_unpacked_XX01L(__m128 XX00, __m128 XX01L)
{
    return _mm_insert_ps(rotate_left(XX00), XX01L, 0xC0);
}

extern inline __m128 load_unpacked_XX02L(__m128 XX01L_, __m128 XX01L)
{
    return _mm_insert_ps(rotate_left(XX01L_), XX01L, 0x80);
}

extern inline __m128 load_unpacked_XX03L(__m128 XX02L, __m128 XX01L)
{
    return _mm_insert_ps(rotate_left(XX02L), XX01L, 0x40);
}

extern inline __m128 load_unpacked_XX01(__m128 XX00, __m128 XX01)
{
    return _mm_insert_ps(rotate_right(XX00), XX01, 0x30);
}

extern inline __m128 load_unpacked_XX02(__m128 XX01_, __m128 XX01)
{
    return _mm_insert_ps(rotate_right(XX01_), XX01, 0x70);
}

extern inline __m128 load_unpacked_XX03(__m128 XX02, __m128 XX01)
{
    return _mm_insert_ps(rotate_right(XX02), XX01, 0xB0);
}

extern inline __m128 load_unpacked_HX01LS(__m128 XX00)
{
    return _mm_shuffle_ps(XX00, XX00, _MM_SHUFFLE(2, 1, 0, 0));
}

extern inline __m128 load_unpacked_HX02LS(__m128 XX00)
{
    return _mm_shuffle_ps(XX00, XX00, _MM_SHUFFLE(1, 0, 0, 1));
}

extern inline __m128 load_unpacked_HX03LS(__m128 XX00)
{
    return _mm_shuffle_ps(XX00, XX00, _MM_SHUFFLE(0, 0, 1, 2));
}

extern inline __m128 load_unpacked_LX01LS(__m128 XX00)
{
    return _mm_shuffle_ps(XX00, XX00, _MM_SHUFFLE(2, 1, 0, 1));
}

extern inline __m128 load_unpacked_LX02LS(__m128 XX00)
{
    return _mm_shuffle_ps(XX00, XX00, _MM_SHUFFLE(1, 0, 1, 2));
}

extern inline __m128 load_unpacked_LX03LS(__m128 XX00)
{
    return _mm_shuffle_ps(XX00, XX00, _MM_SHUFFLE(0, 1, 2, 3));
}

extern inline __m128 load_unpacked_LX01S(__m128 XX00)
{
    return _mm_shuffle_ps(XX00, XX00, _MM_SHUFFLE(3, 3, 2, 1));
}

extern inline __m128 load_unpacked_LX02S(__m128 XX00)
{
    return _mm_shuffle_ps(XX00, XX00, _MM_SHUFFLE(2, 3, 3, 2));
}

extern inline __m128 load_unpacked_LX03S(__m128 XX00)
{
    return _mm_shuffle_ps(XX00, XX00, _MM_SHUFFLE(1, 2, 3, 3));
}

extern inline __m128 load_unpacked_HX01S(__m128 XX00)
{
    return _mm_shuffle_ps(XX00, XX00, _MM_SHUFFLE(2, 3, 2, 1));
}

extern inline __m128 load_unpacked_HX02S(__m128 XX00)
{
    return _mm_shuffle_ps(XX00, XX00, _MM_SHUFFLE(1, 2, 3, 2));
}

extern inline __m128 load_unpacked_HX03S(__m128 XX00)
{
    return _mm_shuffle_ps(XX00, XX00, _MM_SHUFFLE(0, 1, 2, 3));
}

extern inline void print__m128(__m128 A){
    float * i = (float *) &A;
    fprintf(stderr, "%f %f %f %f\n", i[0], i[1], i[2], i[3]);
}
#endif
