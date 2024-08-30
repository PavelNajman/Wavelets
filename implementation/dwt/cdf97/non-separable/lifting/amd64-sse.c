#include <common.h>
#include <common-amd64-sse.h>

#ifdef USE_SSE

#ifdef _OPENMP
    #include <omp.h>
#else
    #define omp_get_thread_num() 0
    #define omp_get_max_threads() 1
    #define omp_get_num_threads() 1
    #define omp_set_num_threads(num_threads)
#endif

static inline void FORCE_INLINE predict_with_unpack_kernel(__m128 R0C0, __m128 R0C1, __m128 R0C2, __m128 R1C0, __m128 R1C1, __m128 R1C2, __m128 R2C0, __m128 R2C1, __m128 R2C2, float * ll, float * lh, float *hl, float * hh)
{
    const __m128 ALPHA = _mm_set1_ps(-1.58613434342059f);
    const __m128 ALPHA2 = _mm_set1_ps(+2.51582f);

    __m128 LL00, LL01, LL11, LL10, LH00, LH01, HH00, HL00, HL10, RES, RES1, RES2;

    LL00 = load_packed_LX00(R0C0, R0C1);
    HL00 = load_packed_HX00(R0C0, R0C1);
    LH00 = load_packed_LX00(R1C0, R1C1);
    HH00 = load_packed_HX00(R1C0, R1C1);

    LL10 = load_packed_LX00(R2C0, R2C1);
    HL10 = load_packed_HX00(R2C0, R2C1);

    LL01 = load_packed_LX01(LL00, R0C2);
    LH01 = load_packed_LX01(LH00, R1C2);
    LL11 = load_packed_LX01(LL10, R2C2);

    RES = _mm_add_ps(LL00, LL01);
#ifdef USE_FMA
    HL00 = _mm_fmadd_ps(RES, ALPHA, HL00);
    LH00 = _mm_fmadd_ps(LL10, ALPHA, LH00);
#else
    RES = _mm_mul_ps(RES, ALPHA);
    HL00 = _mm_add_ps(RES, HL00);
    LL10 = _mm_mul_ps(LL10, ALPHA);
    LH00 = _mm_add_ps(LL10, LH00);
#endif

    RES1 = _mm_mul_ps(LL11, ALPHA2);
    RES = _mm_add_ps(HL00, LH00);
    RES2 = _mm_add_ps(HL10, LH01);
    RES = _mm_add_ps(RES, RES2);

#ifndef USE_FMA
    RES = _mm_mul_ps(RES, ALPHA);
    RES = _mm_add_ps(RES, RES1);
#else
    RES = _mm_fmadd_ps(RES, ALPHA, RES1);
#endif
    HH00 = _mm_add_ps(HH00, RES);

#ifndef USE_FMA
    RES2 = _mm_mul_ps(LL00, ALPHA);
    LH00 = _mm_add_ps(RES2, LH00);
#else
    LH00 = _mm_fmadd_ps(LL00, ALPHA, LH00);
#endif

    _mm_store_ps(ll, LL00);
    _mm_store_ps(hl, HL00);
    _mm_store_ps(hh, HH00);
    _mm_store_ps(lh, LH00);
}

static inline void FORCE_INLINE predict_with_unpack_symmetric_extension_kernel(__m128 R0C0, __m128 R0C1, __m128 R1C0, __m128 R1C1, __m128 R2C0, __m128 R2C1, float * ll, float * lh, float *hl, float * hh)
{
    const __m128 ALPHA = _mm_set1_ps(-1.58613434342059f);
    const __m128 ALPHA2 = _mm_set1_ps(+2.51582f);

    __m128 LL00, LL01, LL11, LL10, LH00, LH01, HH00, HL00, HL10, RES, RES1, RES2;

    LL00 = load_packed_LX00(R0C0, R0C1);
    HL00 = load_packed_HX00(R0C0, R0C1);
    LH00 = load_packed_LX00(R1C0, R1C1);
    HH00 = load_packed_HX00(R1C0, R1C1);

    LL10 = load_packed_LX00(R2C0, R2C1);
    HL10 = load_packed_HX00(R2C0, R2C1);

    LL01 = load_unpacked_LX01S(LL00);
    LH01 = load_unpacked_LX01S(LH00);
    LL11 = load_unpacked_LX01S(LL10);

    RES = _mm_add_ps(LL00, LL01);
#ifndef USE_FMA
    RES = _mm_mul_ps(RES, ALPHA);
    HL00 = _mm_add_ps(RES, HL00);
    LL10 = _mm_mul_ps(LL10, ALPHA);
    LH00 = _mm_add_ps(LL10, LH00);
#else
    HL00 = _mm_fmadd_ps(RES, ALPHA, HL00);
    LH00 = _mm_fmadd_ps(LL10, ALPHA, LH00);
#endif

    RES1 = _mm_mul_ps(LL11, ALPHA2);
    RES = _mm_add_ps(HL00, LH00);
    RES2 = _mm_add_ps(HL10, LH01);
    RES = _mm_add_ps(RES, RES2);
#ifndef USE_FMA
    RES = _mm_mul_ps(RES, ALPHA);
    RES = _mm_add_ps(RES, RES1);
#else
    RES = _mm_fmadd_ps(RES, ALPHA, RES1);
#endif
    HH00 = _mm_add_ps(HH00, RES);

#ifndef USE_FMA
    RES2 = _mm_mul_ps(LL00, ALPHA);
    LH00 = _mm_add_ps(RES2, LH00);
#else
    LH00 = _mm_fmadd_ps(LL00, ALPHA, LH00);
#endif

    _mm_store_ps(ll, LL00);
    _mm_store_ps(hl, HL00);
    _mm_store_ps(hh, HH00);
    _mm_store_ps(lh, LH00);
}

static inline void FORCE_INLINE predict_unpacked_kernel(const float * ll, const float * lh, const float *hl, float * hh, float * tmp_lh, float *tmp_hl, const float *ll10, const float *hl10)
{
    const __m128 ALPHA = _mm_set1_ps(+0.88291107553090f);
    const __m128 ALPHA2 = _mm_set1_ps(+0.779532f);

    __m128 LL00, LL01, LL11, LL10, LH00, LH01, HH00, HL00, HL10, RES, RES1, RES2;

    LL00 = _mm_load_ps(ll);
    HL00 = _mm_load_ps(hl);

    HH00 = _mm_load_ps(hh);
    LH00 = _mm_load_ps(lh);

    LL10 = _mm_load_ps(ll10);
    HL10 = _mm_load_ps(hl10);

    LL01 = load_unpacked_XX01(LL00, _mm_load_ps(ll+4));
    LH01 = load_unpacked_XX01(LH00, _mm_load_ps(lh+4));
    LL11 = load_unpacked_XX01(LL10, _mm_load_ps(ll10+4));

    RES = _mm_add_ps(LL00, LL01);
#ifdef USE_FMA
    HL00 = _mm_fmadd_ps(RES, ALPHA, HL00);
    LH00 = _mm_fmadd_ps(LL10, ALPHA, LH00);
#else
    RES = _mm_mul_ps(RES, ALPHA);
    HL00 = _mm_add_ps(RES, HL00);
    LL10 = _mm_mul_ps(LL10, ALPHA);
    LH00 = _mm_add_ps(LL10, LH00);
#endif

    RES1 = _mm_mul_ps(LL11, ALPHA2);
    RES = _mm_add_ps(HL00, LH00);
    RES2 = _mm_add_ps(HL10, LH01);
    RES = _mm_add_ps(RES, RES2);

#ifndef USE_FMA
    RES = _mm_mul_ps(RES, ALPHA);
    RES = _mm_add_ps(RES, RES1);
#else
    RES = _mm_fmadd_ps(RES, ALPHA, RES1);
#endif
    HH00 = _mm_add_ps(HH00, RES);

#ifndef USE_FMA
    RES2 = _mm_mul_ps(LL00, ALPHA);
    LH00 = _mm_add_ps(RES2, LH00);
#else
    LH00 = _mm_fmadd_ps(LL00, ALPHA, LH00);
#endif

    _mm_store_ps(tmp_hl, HL00);
    _mm_store_ps(hh, HH00);
    _mm_store_ps(tmp_lh, LH00);
}

static inline void FORCE_INLINE predict_unpacked_symmetric_extension_kernel(const float * ll, const float * lh, const float *hl, float * hh, float * tmp_lh, float *tmp_hl, const float *ll1, const float *hl1)
{
    const __m128 ALPHA = _mm_set1_ps(+0.88291107553090f);
    const __m128 ALPHA2 = _mm_set1_ps(+0.779532f);

    __m128 LL00, LL01, LL11, LL10, LH00, LH01, HH00, HL00, HL10, RES, RES1, RES2;

    LL00 = _mm_load_ps(ll);
    HL00 = _mm_load_ps(hl);

    HH00 = _mm_load_ps(hh);
    LH00 = _mm_load_ps(lh);

    LL10 = _mm_load_ps(ll1);
    HL10 = _mm_load_ps(hl1);

    LL01 = load_unpacked_LX01S(LL00);
    LH01 = load_unpacked_LX01S(LH00);
    LL11 = load_unpacked_LX01S(LL10);

    RES = _mm_add_ps(LL00, LL01);
#ifndef USE_FMA
    RES = _mm_mul_ps(RES, ALPHA);
    HL00 = _mm_add_ps(RES, HL00);
    LL10 = _mm_mul_ps(LL10, ALPHA);
    LH00 = _mm_add_ps(LL10, LH00);
#else
    HL00 = _mm_fmadd_ps(RES, ALPHA, HL00);
    LH00 = _mm_fmadd_ps(LL10, ALPHA, LH00);
#endif

    RES1 = _mm_mul_ps(LL11, ALPHA2);
    RES = _mm_add_ps(HL00, LH00);
    RES2 = _mm_add_ps(HL10, LH01);
    RES = _mm_add_ps(RES, RES2);
#ifndef USE_FMA
    RES = _mm_mul_ps(RES, ALPHA);
    RES = _mm_add_ps(RES, RES1);
#else
    RES = _mm_fmadd_ps(RES, ALPHA, RES1);
#endif
    HH00 = _mm_add_ps(HH00, RES);

#ifndef USE_FMA
    RES2 = _mm_mul_ps(LL00, ALPHA);
    LH00 = _mm_add_ps(RES2, LH00);
#else
    LH00 = _mm_fmadd_ps(LL00, ALPHA, LH00);
#endif

    _mm_store_ps(tmp_hl, HL00);
    _mm_store_ps(hh, HH00);
    _mm_store_ps(tmp_lh, LH00);
}

static inline void FORCE_INLINE update_unpacked_kernel(float * ll, float * lh, float * hl, const float * hh, const float *tmp_lh, const float *tmp_hl, const float *tmp_hl1, const float *tmp_lh1, const float *hh1)
{
    const __m128 BETA = _mm_set1_ps(-0.05298011857290f);
    const __m128 BETA2 = _mm_set1_ps(+0.00280689f);

    __m128 LL00, HL01, HH11, LH10, LH00, HH01, HH00, HL00, HH10, RES, RES1, RES2;

    LL00 = _mm_load_ps(ll);
    HL00 = _mm_load_ps(tmp_hl);
    LH00 = _mm_load_ps(tmp_lh);
    HH00 = _mm_load_ps(hh);

    LH10 = _mm_load_ps(tmp_lh1);
    HH10 = _mm_load_ps(hh1);

    HL01 = load_unpacked_XX01L(HL00, _mm_load_ps(tmp_hl1-4));
    HH01 = load_unpacked_XX01L(HH00, _mm_load_ps(hh - 4));
    HH11 = load_unpacked_XX01L(HH10, _mm_load_ps(hh1 - 4));

    RES = _mm_add_ps(HH01, HH00);
#ifndef USE_FMA
    RES = _mm_mul_ps(RES, BETA);
    LH00 = _mm_add_ps(RES, LH00);
    HH10 = _mm_mul_ps(HH10, BETA);
    HL00 = _mm_add_ps(HH10, HL00);
#else
    LH00 = _mm_fmadd_ps(RES, BETA, LH00);
    HL00 = _mm_fmadd_ps(HH10, BETA, HL00);
#endif

    RES1 = _mm_mul_ps(HH11, BETA2);
    RES = _mm_add_ps(LH00, HL00);
    RES2 = _mm_add_ps(LH10, HL01);
    RES = _mm_add_ps(RES, RES2);
#ifndef USE_FMA
    RES = _mm_mul_ps(RES, BETA);
    RES = _mm_add_ps(RES, RES1);
#else
    RES = _mm_fmadd_ps(RES, BETA, RES1);
#endif
    LL00 = _mm_add_ps(LL00, RES);

#ifndef USE_FMA
    RES = _mm_mul_ps(HH00, BETA);
    HL00 = _mm_add_ps(RES, HL00);
#else
    HL00 = _mm_fmadd_ps(HH00, BETA, HL00);
#endif

    _mm_store_ps(lh, LH00);
    _mm_store_ps(ll, LL00);
    _mm_store_ps(hl, HL00);
}

static inline void FORCE_INLINE update_unpacked_symmetric_extension_kernel(float * ll, float * lh, float * hl, const float * hh, const float * tmp_lh, const float * tmp_hl, const float *tmp_lh1, const float *hh1)
{
    const __m128 BETA = _mm_set1_ps(-0.05298011857290f);
    const __m128 BETA2 = _mm_set1_ps(+0.00280689f);

    __m128 LL00, HL01, HH11, LH10, LH00, HH01, HH00, HL00, HH10, RES, RES1, RES2;

    LL00 = _mm_load_ps(ll);
    HL00 = _mm_load_ps(tmp_hl);
    LH00 = _mm_load_ps(tmp_lh);
    HH00 = _mm_load_ps(hh);

    LH10 = _mm_load_ps(tmp_lh1);
    HH10 = _mm_load_ps(hh1);

    HL01 = load_unpacked_HX01LS(HL00);
    HH01 = load_unpacked_HX01LS(HH00);
    HH11 = load_unpacked_HX01LS(HH10);

    RES = _mm_add_ps(HH01, HH00);
#ifndef USE_FMA
    RES = _mm_mul_ps(RES, BETA);
    LH00 = _mm_add_ps(RES, LH00);
    HH10 = _mm_mul_ps(HH10, BETA);
    HL00 = _mm_add_ps(HH10, HL00);
#else
    LH00 = _mm_fmadd_ps(RES, BETA, LH00);
    HL00 = _mm_fmadd_ps(HH10, BETA, HL00);
#endif

    RES1 = _mm_mul_ps(HH11, BETA2);
    RES = _mm_add_ps(LH00, HL00);
    RES2 = _mm_add_ps(LH10, HL01);
    RES = _mm_add_ps(RES, RES2);
#ifndef USE_FMA
    RES = _mm_mul_ps(RES, BETA);
    RES = _mm_add_ps(RES, RES1);
#else
    RES = _mm_fmadd_ps(RES, BETA, RES1);
#endif
    LL00 = _mm_add_ps(LL00, RES);

#ifndef USE_FMA
    RES = _mm_mul_ps(HH00, BETA);
    HL00 = _mm_add_ps(RES, HL00);
#else
    HL00 = _mm_fmadd_ps(HH00, BETA, HL00);
#endif

    _mm_store_ps(lh, LH00);
    _mm_store_ps(ll, LL00);
    _mm_store_ps(hl, HL00);
}

static inline void FORCE_INLINE update_unpacked_kernel2(float * ll, float * lh, float * hl, const float * hh, const float *tmp_lh, const float *tmp_hl, const float *tmp_hl1, const float *tmp_lh1, const float *hh1)
{
    const __m128 BETA = _mm_set1_ps(+0.44350685204390f);
    const __m128 BETA2 = _mm_set1_ps(+0.196698f);

    __m128 LL00, HL01, HH11, LH10, LH00, HH01, HH00, HL00, HH10, RES, RES1, RES2;

    LL00 = _mm_load_ps(ll);
    HL00 = _mm_load_ps(tmp_hl);
    LH00 = _mm_load_ps(tmp_lh);
    HH00 = _mm_load_ps(hh);

    LH10 = _mm_load_ps(tmp_lh1);
    HH10 = _mm_load_ps(hh1);

    HL01 = load_unpacked_XX01L(HL00, _mm_load_ps(tmp_hl1 - 4));
    HH01 = load_unpacked_XX01L(HH00, _mm_load_ps(hh - 4));
    HH11 = load_unpacked_XX01L(HH10, _mm_load_ps(hh1 - 4));

    RES = _mm_add_ps(HH01, HH00);
#ifndef USE_FMA
    RES = _mm_mul_ps(RES, BETA);
    LH00 = _mm_add_ps(RES, LH00);
    HH10 = _mm_mul_ps(HH10, BETA);
    HL00 = _mm_add_ps(HH10, HL00);
#else
    LH00 = _mm_fmadd_ps(RES, BETA, LH00);
    HL00 = _mm_fmadd_ps(HH10, BETA, HL00);
#endif

    RES1 = _mm_mul_ps(HH11, BETA2);
    RES = _mm_add_ps(LH00, HL00);
    RES2 = _mm_add_ps(LH10, HL01);
    RES = _mm_add_ps(RES, RES2);
#ifndef USE_FMA
    RES = _mm_mul_ps(RES, BETA);
    RES = _mm_add_ps(RES, RES1);
#else
    RES = _mm_fmadd_ps(RES, BETA, RES1);
#endif
    LL00 = _mm_add_ps(LL00, RES);

#ifndef USE_FMA
    RES = _mm_mul_ps(HH00, BETA);
    HL00 = _mm_add_ps(RES, HL00);
#else
    HL00 = _mm_fmadd_ps(HH00, BETA, HL00);
#endif

    _mm_store_ps(lh, LH00);
    _mm_store_ps(ll, LL00);
    _mm_store_ps(hl, HL00);
}

static inline void FORCE_INLINE update_unpacked_symmetric_extension_kernel2(float * ll, float * lh, float * hl, const float * hh, const float * tmp_lh, const float * tmp_hl, const float *tmp_lh1, const float *hh1)
{
    const __m128 BETA = _mm_set1_ps(+0.44350685204390f);
    const __m128 BETA2 = _mm_set1_ps(+0.196698f);

    __m128 LL00, HL01, HH11, LH10, LH00, HH01, HH00, HL00, HH10, RES, RES1, RES2;

    LL00 = _mm_load_ps(ll);
    HL00 = _mm_load_ps(tmp_hl);
    LH00 = _mm_load_ps(tmp_lh);
    HH00 = _mm_load_ps(hh);

    LH10 = _mm_load_ps(tmp_lh1);
    HH10 = _mm_load_ps(hh1);

    HL01 = load_unpacked_HX01LS(HL00);
    HH01 = load_unpacked_HX01LS(HH00);
    HH11 = load_unpacked_HX01LS(HH10);

    RES = _mm_add_ps(HH01, HH00);
#ifndef USE_FMA
    RES = _mm_mul_ps(RES, BETA);
    LH00 = _mm_add_ps(RES, LH00);
    HH10 = _mm_mul_ps(HH10, BETA);
    HL00 = _mm_add_ps(HH10, HL00);
#else
    LH00 = _mm_fmadd_ps(RES, BETA, LH00);
    HL00 = _mm_fmadd_ps(HH10, BETA, HL00);
#endif

    RES1 = _mm_mul_ps(HH11, BETA2);
    RES = _mm_add_ps(LH00, HL00);
    RES2 = _mm_add_ps(LH10, HL01);
    RES = _mm_add_ps(RES, RES2);
#ifndef USE_FMA
    RES = _mm_mul_ps(RES, BETA);
    RES = _mm_add_ps(RES, RES1);
#else
    RES = _mm_fmadd_ps(RES, BETA, RES1);
#endif
    LL00 = _mm_add_ps(LL00, RES);

#ifndef USE_FMA
    RES = _mm_mul_ps(HH00, BETA);
    HL00 = _mm_add_ps(RES, HL00);
#else
    HL00 = _mm_fmadd_ps(HH00, BETA, HL00);
#endif

    _mm_store_ps(lh, LH00);
    _mm_store_ps(ll, LL00);
    _mm_store_ps(hl, HL00);
}

static void predict_with_unpack(const TransformStepArguments * tsa)
{
    const size_t size_x = tsa->tile.size_x;
    const size_t size_y = tsa->tile.size_y;
    const size_t stride_y = tsa->tile.stride_y;
    float * mem = tsa->tile.data;

    const size_t band_stride_y = tsa->tile_bands.stride_y;

    const size_t tid = (size_t)omp_get_thread_num();

    const size_t band_start_y = tsa->threading_info->band_start_y[tid];
    const size_t band_end_y = tsa->threading_info->band_end_y[tid];
    const size_t band_start_x = tsa->threading_info->band_start_x[tid];
    const size_t band_end_x = tsa->threading_info->band_end_x[tid];

    const size_t img_start_y = band_start_y << 1;
    const size_t img_end_y = band_end_y << 1;
    const size_t img_start_x = band_start_x << 1;
    const size_t img_end_x = band_end_x << 1;

    const size_t next_tile_y = 2 * stride_y  - (img_end_x - img_start_x) + 8;
    const size_t next_band_y = band_stride_y - (band_end_x - band_start_x) + 4;

    float * m0 = mem + img_start_x + img_start_y * stride_y;
    float * m1 = mem + img_start_x + (img_start_y+1) * stride_y;
    float * m2 = mem + img_start_x + (img_start_y+2) * stride_y;

    float * ll = tsa->tile_bands.LL + band_start_x + band_start_y * band_stride_y;
    float * hh = tsa->tile_bands.HH + band_start_x + band_start_y * band_stride_y;
    float * hl = tsa->tile_bands.HL + band_start_x + band_start_y * band_stride_y;
    float * lh = tsa->tile_bands.LH + band_start_x + band_start_y * band_stride_y;

    const size_t num_tmp_cache_lines = ((tsa->tmp.size_y * tsa->tmp.size_x) * sizeof(float)) / CACHE_LINE_SIZE;
    const size_t tmp_offset = tid * (num_tmp_cache_lines / (size_t) omp_get_num_threads()) * (CACHE_LINE_SIZE / sizeof(float));

    float * tmp_hl = tsa->tmp.HL + tmp_offset;
    float * tmp_lh = tsa->tmp.LH + tmp_offset;

    __m128 R0C0, R0C1, R0C2, R1C0, R1C1, R1C2, R2C0, R2C1, R2C2;

    if(img_end_x == size_x){
        for(size_t y = img_start_y; y < img_end_y - 2; y += 2){
            R0C0 = _mm_load_ps(m0); R1C0 = _mm_load_ps(m1); R2C0 = _mm_load_ps(m2);
            for(size_t x = img_start_x; x < img_end_x - 8; x += 8){
                R0C1 = _mm_load_ps(m0 + 4); R0C2 = _mm_load_ps(m0 + 8);
                R1C1 = _mm_load_ps(m1 + 4); R1C2 = _mm_load_ps(m1 + 8);
                R2C1 = _mm_load_ps(m2 + 4); R2C2 = _mm_load_ps(m2 + 8);

                predict_with_unpack_kernel(R0C0, R0C1, R0C2, R1C0, R1C1, R1C2, R2C0, R2C1, R2C2, ll, lh, hl, hh);

                R0C0 = R0C2; R1C0 = R1C2; R2C0 = R2C2;

                ll += 4; hh += 4; hl += 4; lh +=4;
                m0 += 8; m1 += 8; m2 += 8;

            }
            R0C1 = _mm_load_ps(m0 + 4); R1C1 = _mm_load_ps(m1 + 4); R2C1 = _mm_load_ps(m2 + 4);

            predict_with_unpack_symmetric_extension_kernel(R0C0, R0C1, R1C0, R1C1, R2C0, R2C1, ll, lh, hl, hh);

            ll += next_band_y; hh += next_band_y; hl += next_band_y; lh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y;
        }

        m2 = img_end_y == size_y ? m2 - 2 * stride_y : m2;
        R0C0 = _mm_load_ps(m0); R1C0 = _mm_load_ps(m1); R2C0 = _mm_load_ps(m2);
        for(size_t x = img_start_x; x < img_end_x - 8; x += 8){
            R0C1 = _mm_load_ps(m0 + 4); R0C2 = _mm_load_ps(m0 + 8);
            R1C1 = _mm_load_ps(m1 + 4);R1C2 = _mm_load_ps(m1 + 8);
            R2C1 = _mm_load_ps(m2 + 4); R2C2 = _mm_load_ps(m2 + 8);

            predict_with_unpack_kernel(R0C0, R0C1, R0C2, R1C0, R1C1, R1C2, R2C0, R2C1, R2C2, ll, tmp_lh, hl, hh);

            R0C0 = R0C2; R1C0 = R1C2; R2C0 = R2C2;
            ll += 4; hh += 4; hl += 4; tmp_lh += 4;
            m0 += 8; m1 += 8; m2 += 8;
        }
        R0C1 = _mm_load_ps(m0 + 4); R1C1 = _mm_load_ps(m1 + 4); R2C1 = _mm_load_ps(m2 + 4);

        predict_with_unpack_symmetric_extension_kernel(R0C0, R0C1, R1C0, R1C1, R2C0, R2C1, ll, tmp_lh, hl, hh);
    } else {
        for(size_t y = img_start_y; y < img_end_y - 2; y += 2){
            R0C0 = _mm_load_ps(m0); R1C0 = _mm_load_ps(m1); R2C0 = _mm_load_ps(m2);
            for(size_t x = img_start_x; x < img_end_x - 8; x += 8){
                R0C1 = _mm_load_ps(m0 + 4); R0C2 = _mm_load_ps(m0 + 8);
                R1C1 = _mm_load_ps(m1 + 4); R1C2 = _mm_load_ps(m1 + 8);
                R2C1 = _mm_load_ps(m2 + 4); R2C2 = _mm_load_ps(m2 + 8);

                predict_with_unpack_kernel(R0C0, R0C1, R0C2, R1C0, R1C1, R1C2, R2C0, R2C1, R2C2, ll, lh, hl, hh);

                R0C0 = R0C2; R1C0 = R1C2; R2C0 = R2C2;

                ll += 4; hh += 4; hl += 4; lh +=4;
                m0 += 8; m1 += 8; m2 += 8;
            }
            R0C1 = _mm_load_ps(m0 + 4); R1C1 = _mm_load_ps(m1 + 4); R2C1 = _mm_load_ps(m2 + 4);
            R0C2 = _mm_load_ps(m0 + 8); R1C2 = _mm_load_ps(m1 + 8); R2C2 = _mm_load_ps(m2 + 8);
            predict_with_unpack_kernel(R0C0, R0C1, R0C2, R1C0, R1C1, R1C2, R2C0, R2C1, R2C2, ll, lh, tmp_hl, hh);

            ll += next_band_y; hh += next_band_y; hl += next_band_y; lh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y;
            tmp_hl += 4;
        }

        m2 = img_end_y == size_y ? m2 - 2 * stride_y : m2;
        R0C0 = _mm_load_ps(m0); R1C0 = _mm_load_ps(m1); R2C0 = _mm_load_ps(m2);
        for(size_t x = img_start_x; x < img_end_x - 8; x += 8){
            R0C1 = _mm_load_ps(m0 + 4); R0C2 = _mm_load_ps(m0 + 8);
            R1C1 = _mm_load_ps(m1 + 4);R1C2 = _mm_load_ps(m1 + 8);
            R2C1 = _mm_load_ps(m2 + 4); R2C2 = _mm_load_ps(m2 + 8);

            predict_with_unpack_kernel(R0C0, R0C1, R0C2, R1C0, R1C1, R1C2, R2C0, R2C1, R2C2, ll, tmp_lh, hl, hh);

            R0C0 = R0C2; R1C0 = R1C2; R2C0 = R2C2;
            ll += 4; hh += 4; hl += 4; tmp_lh += 4;
            m0 += 8; m1 += 8; m2 += 8;
        }

        R0C1 = _mm_load_ps(m0 + 4); R0C2 = _mm_load_ps(m0 + 8);
        R1C1 = _mm_load_ps(m1 + 4);R1C2 = _mm_load_ps(m1 + 8);
        R2C1 = _mm_load_ps(m2 + 4); R2C2 = _mm_load_ps(m2 + 8);

        predict_with_unpack_kernel(R0C0, R0C1, R0C2, R1C0, R1C1, R1C2, R2C0, R2C1, R2C2, ll, tmp_lh, tmp_hl, hh);
    }

    # pragma omp barrier
}

static void update_unpacked(const TransformStepArguments * tsa)
{
    const size_t band_stride_y = tsa->tile_bands.stride_y;

    const size_t tid = (size_t)omp_get_thread_num();

    const size_t band_size_x = tsa->tile_bands.size_x;

    const size_t band_start_y = tsa->threading_info->band_start_y[tid];
    const size_t band_end_y = tsa->threading_info->band_end_y[tid];
    const size_t band_start_x = tsa->threading_info->band_start_x[tid];
    const size_t band_end_x = tsa->threading_info->band_end_x[tid];

    const size_t next_band_y = ((band_end_x - band_start_x) - 4 - band_stride_y);

    float * ll = tsa->tile_bands.LL + (band_end_y - 1) * band_stride_y + band_end_x - 4;
    float * hl = tsa->tile_bands.HL + (band_end_y - 1) * band_stride_y + band_end_x - 4;
    float * lh = tsa->tile_bands.LH + (band_end_y - 1) * band_stride_y + band_end_x - 4;
    float * hh = tsa->tile_bands.HH + (band_end_y - 1) * band_stride_y + band_end_x - 4;

    const size_t num_tmp_cache_lines = ((tsa->tmp.size_y * tsa->tmp.size_x) * sizeof(float)) / CACHE_LINE_SIZE;
    const size_t num_tmp_elements_per_thread = (num_tmp_cache_lines / (size_t) omp_get_num_threads()) * (CACHE_LINE_SIZE / sizeof(float));
    const size_t tmp_offset = tid * num_tmp_elements_per_thread;
    const size_t tmp_offset_left = (tid - 1) * num_tmp_elements_per_thread;
    const size_t tmp_offset_upper = (tid - tsa->threading_info->thread_cols) * num_tmp_elements_per_thread;

    float * tmp_lh = tsa->tmp.LH + tmp_offset + (band_end_x - band_start_x) - 4;
    float * tmp_lh1 = tsa->tmp.LH + tmp_offset_upper + (band_end_x - band_start_x) - 4;
    float * hh1 = tsa->tile_bands.HH + (band_start_y - 1) * band_stride_y + band_end_x - 4;
    if(band_start_y == 0){
        hh1 = tsa->tile_bands.HH + band_end_x - 4;
        if(band_end_y - band_start_y == 1){
            tmp_lh1 = band_start_y == 0 ? tmp_lh : tmp_lh1;
        } else {
            tmp_lh1 = band_start_y == 0 ? tsa->tile_bands.LH + band_end_x - 4 : tmp_lh1;
        }
    }

    if(band_start_x == 0 && band_end_x == band_size_x){
        if(band_end_y - band_start_y > 1){
            for(size_t x = band_end_x - 4; x >= band_start_x + 4; x -= 4){ // main area
                update_unpacked_kernel(ll, lh, hl, hh, tmp_lh, hl, hl, lh - band_stride_y, hh - band_stride_y);

                lh -= 4; ll -= 4; hl -= 4; hh -= 4; tmp_lh -= 4;
            }
            // left extension
            update_unpacked_symmetric_extension_kernel(ll, lh, hl, hh, tmp_lh, hl, lh - band_stride_y, hh - band_stride_y);

            ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y;
        }

        if(band_end_y - band_start_y > 2){
            for(size_t y = band_end_y - 2; y > band_start_y; --y){
                for(size_t x = band_end_x - 4; x >= band_start_x + 4; x -= 4){ // main area
                    update_unpacked_kernel(ll, lh, hl, hh, lh, hl, hl, lh - band_stride_y, hh - band_stride_y);

                    lh -= 4; ll -= 4; hl -= 4; hh -= 4;
                }
                // left extension
                update_unpacked_symmetric_extension_kernel(ll, lh, hl, hh, lh, hl, lh - band_stride_y, hh - band_stride_y);

                ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y;
            }
        }

        tmp_lh = band_end_y - band_start_y == 1 ? tmp_lh : lh;
        for(size_t x = band_end_x - 4; x >= band_start_x + 4; x -= 4){ // top row
            update_unpacked_kernel(ll, lh, hl, hh, tmp_lh, hl, hl, tmp_lh1, hh1);

            lh -= 4; ll -= 4; hl-= 4; hh -= 4; hh1 -= 4; tmp_lh1 -= 4; tmp_lh -= 4;
        }
        // left extension
        update_unpacked_symmetric_extension_kernel(ll, lh, hl, hh, tmp_lh, hl, tmp_lh1, hh1);
    } else if(band_start_x == 0){
        float * tmp_hl = tsa->tmp.HL + tmp_offset + 4 * (band_end_y - band_start_y) - 4;

        if(band_end_y - band_start_y > 1){
            update_unpacked_kernel(ll, lh, hl, hh, tmp_lh, tmp_hl, hl, lh - band_stride_y, hh - band_stride_y);

            lh -= 4; ll -= 4; hl -= 4; hh -= 4; tmp_lh -= 4; tmp_hl -= 4;

            for(size_t x = band_end_x - 8; x >= band_start_x + 4; x -= 4){ // main area
                update_unpacked_kernel(ll, lh, hl, hh, tmp_lh, hl, hl, lh - band_stride_y, hh - band_stride_y);

                lh -= 4; ll -= 4; hl -= 4; hh -= 4; tmp_lh -= 4;
            }
            // left extension
            update_unpacked_symmetric_extension_kernel(ll, lh, hl, hh, tmp_lh, hl, lh - band_stride_y, hh - band_stride_y);

            ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y;
        }

        if(band_end_y - band_start_y > 2){
            for(size_t y = band_end_y - 2; y > band_start_y; --y){
                update_unpacked_kernel(ll, lh, hl, hh, lh, tmp_hl, hl, lh - band_stride_y, hh - band_stride_y);

                lh -= 4; ll -= 4; hl -= 4; hh -= 4; tmp_hl -= 4;

                for(size_t x = band_end_x - 8; x >= band_start_x + 4; x -= 4){ // main area
                    update_unpacked_kernel(ll, lh, hl, hh, lh, hl, hl, lh - band_stride_y, hh - band_stride_y);

                    lh -= 4; ll -= 4; hl -= 4; hh -= 4;
                }
                // left extension
                update_unpacked_symmetric_extension_kernel(ll, lh, hl, hh, lh, hl, lh - band_stride_y, hh - band_stride_y);

                ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y;
            }
        }

        tmp_lh = band_end_y - band_start_y == 1 ? tmp_lh : lh;
        update_unpacked_kernel(ll, lh, hl, hh, tmp_lh, tmp_hl, hl, tmp_lh1, hh1);

        lh -= 4; ll -= 4; hl-= 4; hh -= 4; hh1 -= 4; tmp_lh1 -= 4; tmp_lh -= 4;

        for(size_t x = band_end_x - 8; x >= band_start_x + 4; x -= 4){ // top row
            update_unpacked_kernel(ll, lh, hl, hh, tmp_lh, hl, hl, tmp_lh1, hh1);

            lh -= 4; ll -= 4; hl-= 4; hh -= 4; hh1 -= 4; tmp_lh1 -= 4; tmp_lh -= 4;
        }
        // left extension
        update_unpacked_symmetric_extension_kernel(ll, lh, hl, hh, tmp_lh, hl, tmp_lh1, hh1);
    } else if(band_end_x == band_size_x){
        float * tmp_hl = tsa->tmp.HL + tmp_offset_left + 4 * (tsa->threading_info->band_end_y[tid-1] - tsa->threading_info->band_start_y[tid-1]);

        if(band_end_y - band_start_y > 1){
            for(size_t x = band_end_x - 4; x >= band_start_x + 4; x -= 4){ // main area
                update_unpacked_kernel(ll, lh, hl, hh, tmp_lh, hl, hl, lh - band_stride_y, hh - band_stride_y);

                lh -= 4; ll -= 4; hl -= 4; hh -= 4; tmp_lh -= 4;
            }
            // left extension
            update_unpacked_kernel(ll, lh, hl, hh, tmp_lh, hl, tmp_hl, lh - band_stride_y, hh - band_stride_y);

            ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y; tmp_hl -= 4;
        }

        if(band_end_y - band_start_y > 2){
            for(size_t y = band_end_y - 2; y > band_start_y; --y){
                for(size_t x = band_end_x - 4; x >= band_start_x + 4; x -= 4){ // main area
                    update_unpacked_kernel(ll, lh, hl, hh, lh, hl, hl, lh - band_stride_y, hh - band_stride_y);

                    lh -= 4; ll -= 4; hl -= 4; hh -= 4;
                }
                // left extension
                update_unpacked_kernel(ll, lh, hl, hh, lh, hl, tmp_hl, lh - band_stride_y, hh - band_stride_y);

                ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y; tmp_hl -= 4;
            }
        }

        tmp_lh = band_end_y - band_start_y == 1 ? tmp_lh : lh;
        for(size_t x = band_end_x - 4; x >= band_start_x + 4; x -= 4){ // top row
            update_unpacked_kernel(ll, lh, hl, hh, tmp_lh, hl, hl, tmp_lh1, hh1);

            lh -= 4; ll -= 4; hl-= 4; hh -= 4; hh1 -= 4; tmp_lh1 -= 4; tmp_lh -= 4;
        }
        // left extension
        update_unpacked_kernel(ll, lh, hl, hh, tmp_lh, hl, tmp_hl, tmp_lh1, hh1);
    } else {
        float * tmp_hl = tsa->tmp.HL + tmp_offset + 4 * (band_end_y - band_start_y) - 4;
        float * tmp_hl1 = tsa->tmp.HL + tmp_offset_left + 4 * (tsa->threading_info->band_end_y[tid-1] - tsa->threading_info->band_start_y[tid-1]);

        if(band_end_y - band_start_y > 1){
            update_unpacked_kernel(ll, lh, hl, hh, tmp_lh, tmp_hl, hl, lh - band_stride_y, hh - band_stride_y);

            lh -= 4; ll -= 4; hl -= 4; hh -= 4; tmp_hl -= 4; tmp_lh -= 4;

            for(size_t x = band_end_x - 8; x >= band_start_x + 4; x -= 4){ // main area
                update_unpacked_kernel(ll, lh, hl, hh, tmp_lh, hl, hl, lh - band_stride_y, hh - band_stride_y);

                lh -= 4; ll -= 4; hl -= 4; hh -= 4; tmp_lh -= 4;
            }
            // left extension
            update_unpacked_kernel(ll, lh, hl, hh, tmp_lh, hl, tmp_hl1, lh - band_stride_y, hh  - band_stride_y);

            ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y; tmp_hl1 -= 4;
        }

        if(band_end_y - band_start_y > 2){
            for(size_t y = band_end_y - 2; y > band_start_y; --y){
                update_unpacked_kernel(ll, lh, hl, hh, lh, tmp_hl, hl, lh - band_stride_y, hh - band_stride_y);

                lh -= 4; ll -= 4; hl -= 4; hh -= 4; tmp_hl -= 4;

                for(size_t x = band_end_x - 8; x >= band_start_x + 4; x -= 4){ // main area
                    update_unpacked_kernel(ll, lh, hl, hh, lh, hl, hl, lh - band_stride_y, hh - band_stride_y);

                    lh -= 4; ll -= 4; hl -= 4; hh -= 4;
                }
                // left extension
                update_unpacked_kernel(ll, lh, hl, hh, lh, hl, tmp_hl1, lh - band_stride_y, hh  - band_stride_y);

                ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y; tmp_hl1 -= 4;
            }
        }

        tmp_lh = band_end_y - band_start_y == 1 ? tmp_lh : lh;
        update_unpacked_kernel(ll, lh, hl, hh, tmp_lh, tmp_hl, hl, tmp_lh1, hh1);

        lh -= 4; ll -= 4; hl-= 4; hh -= 4; hh1 -= 4; tmp_lh1 -= 4; tmp_lh -= 4;

        for(size_t x = band_end_x - 8; x >= band_start_x + 4; x -= 4){ // top row
            update_unpacked_kernel(ll, lh, hl, hh, tmp_lh, hl, hl, tmp_lh1, hh1);

            lh -= 4; ll -= 4; hl-= 4; hh -= 4; hh1 -= 4; tmp_lh1 -= 4; tmp_lh -= 4;
        }

        update_unpacked_kernel(ll, lh, hl, hh, tmp_lh, hl, tmp_hl1, tmp_lh1, hh1);
    }

    #pragma omp barrier
}

static void predict_unpacked(const TransformStepArguments * tsa)
{
    const size_t band_size_y = tsa->tile_bands.size_y;
    const size_t band_size_x = tsa->tile_bands.size_x;
    const size_t band_stride_y = tsa->tile_bands.stride_y;

    const size_t tid = (size_t)omp_get_thread_num();

    const size_t band_start_y = tsa->threading_info->band_start_y[tid];
    const size_t band_end_y = tsa->threading_info->band_end_y[tid];
    const size_t band_start_x = tsa->threading_info->band_start_x[tid];
    const size_t band_end_x = tsa->threading_info->band_end_x[tid];

    const size_t next_band_y = band_stride_y - (band_end_x - band_start_x) + 4;

    float * ll = tsa->tile_bands.LL + band_start_x + band_start_y * band_stride_y;
    float * hl = tsa->tile_bands.HL + band_start_x + band_start_y * band_stride_y;
    float * lh = tsa->tile_bands.LH + band_start_x + band_start_y * band_stride_y;
    float * hh = tsa->tile_bands.HH + band_start_x + band_start_y * band_stride_y;

    const size_t num_tmp_cache_lines = ((tsa->tmp.size_y * tsa->tmp.size_x) * sizeof(float)) / CACHE_LINE_SIZE;
    const size_t tmp_offset = tid * (num_tmp_cache_lines / (size_t) omp_get_num_threads()) * (CACHE_LINE_SIZE / sizeof(float));

    float * tmp_hl = tsa->tmp.HL + tmp_offset;
    float * tmp_lh = tsa->tmp.LH + tmp_offset;

    if(band_start_x == 0 && band_end_x == band_size_x) {
        if(band_end_y - band_start_y > 1){
            for(size_t x = band_start_x; x < band_end_x - 4; x += 4){
                predict_unpacked_kernel(ll, lh, hl, hh, lh, tmp_hl, ll + band_stride_y, hl + band_stride_y);

                ll += 4; hl += 4; hh += 4; lh += 4; tmp_hl += 4;
            }

            // right
            predict_unpacked_symmetric_extension_kernel(ll, lh, hl, hh, lh, tmp_hl, ll + band_stride_y, hl + band_stride_y);

            ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y;
        }

        if(band_end_y - band_start_y > 2){
            for(size_t y = band_start_y + 1; y < band_end_y - 1; ++y){
                for(size_t x = band_start_x; x < band_end_x - 4; x += 4){
                    predict_unpacked_kernel(ll, lh, hl, hh, lh, hl, ll + band_stride_y, hl + band_stride_y);

                    ll += 4; hl += 4; hh += 4; lh += 4;
                }

                // right
                predict_unpacked_symmetric_extension_kernel(ll, lh, hl, hh, lh, hl, ll + band_stride_y, hl + band_stride_y);

                ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y;
            }
        }

        // bottom row
        float * ll10 = band_end_y == band_size_y ? ll : ll + band_stride_y;
        float * hl10 = band_end_y == band_size_y ? hl : hl + band_stride_y;
        float * hl1 = band_end_y - band_start_y == 1 ? tmp_hl : hl;
        for(size_t x = band_start_x; x < band_end_x - 4; x += 4){
            predict_unpacked_kernel(ll, lh, hl, hh, tmp_lh, hl1, ll10, hl10);

            ll += 4; hl += 4; hh += 4; lh += 4; ll10 += 4; hl10 += 4; tmp_lh += 4; hl1 += 4;
        }

        // botom right
        predict_unpacked_symmetric_extension_kernel(ll, lh, hl, hh, tmp_lh, hl1, ll10, hl10);
    } else if(band_start_x == 0){
        if(band_end_y - band_start_y > 1){
            for(size_t x = band_start_x; x < band_end_x - 4; x += 4){
                predict_unpacked_kernel(ll, lh, hl, hh, lh, tmp_hl, ll + band_stride_y, hl + band_stride_y);

                ll += 4; hl += 4; hh += 4; lh += 4; tmp_hl += 4;
            }

            // right
            predict_unpacked_kernel(ll, lh, hl, hh, lh, tmp_hl, ll + band_stride_y, hl + band_stride_y);

            ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y; tmp_hl += 4;
        }

        if(band_end_y - band_start_y > 2){
            for(size_t y = band_start_y + 1; y < band_end_y - 1; ++y){
                for(size_t x = band_start_x; x < band_end_x - 4; x += 4){
                    predict_unpacked_kernel(ll, lh, hl, hh, lh, hl, ll + band_stride_y, hl + band_stride_y);

                    ll += 4; hl += 4; hh += 4; lh += 4;
                }

                // right
                predict_unpacked_kernel(ll, lh, hl, hh, lh, tmp_hl, ll + band_stride_y, hl + band_stride_y);

                ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y;
                tmp_hl += 4;
            }
        }

        // bottom row
        float * ll10 = band_end_y == band_size_y ? ll : ll + band_stride_y;
        float * hl10 = band_end_y == band_size_y ? hl : hl + band_stride_y;
        float * hl1 = band_end_y - band_start_y == 1 ? tmp_hl : hl;
        for(size_t x = band_start_x; x < band_end_x - 4; x += 4){
            predict_unpacked_kernel(ll, lh, hl, hh, tmp_lh, hl1, ll10, hl10);

            ll += 4; hl += 4; hh += 4; lh += 4; ll10 += 4; hl10 += 4; tmp_lh += 4; hl1 += 4;
        }
        predict_unpacked_kernel(ll, lh, hl, hh, tmp_lh, band_end_y - band_start_y == 1 ? hl1 : tmp_hl, ll10, hl10);
    } else if(band_end_x == band_size_x){
        if(band_end_y - band_start_y > 1){
            predict_unpacked_kernel(ll, lh, hl, hh, tmp_lh, tmp_hl, ll + band_stride_y, hl + band_stride_y);

            ll += 4; hl += 4; hh += 4; lh += 4; tmp_hl += 4; tmp_lh += 4;

            for(size_t x = band_start_x + 4; x < band_end_x - 4; x += 4){
                predict_unpacked_kernel(ll, lh, hl, hh, lh, tmp_hl, ll + band_stride_y, hl + band_stride_y);

                ll += 4; hl += 4; hh += 4; lh += 4; tmp_hl += 4;
            }

            // right
            predict_unpacked_symmetric_extension_kernel(ll, lh, hl, hh, lh, tmp_hl, ll + band_stride_y, hl + band_stride_y);

            ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y; tmp_hl += 4;
        }

        if(band_end_y - band_start_y > 2){
            for(size_t y = band_start_y + 1; y < band_end_y - 1; ++y){
                predict_unpacked_kernel(ll, lh, hl, hh, tmp_lh, hl, ll + band_stride_y, hl + band_stride_y);

                ll += 4; hl += 4; hh += 4; lh += 4; tmp_lh += 4;

                for(size_t x = band_start_x + 4; x < band_end_x - 4; x += 4){
                    predict_unpacked_kernel(ll, lh, hl, hh, lh, hl, ll + band_stride_y, hl + band_stride_y);

                    ll += 4; hl += 4; hh += 4; lh += 4;
                }

                // right
                predict_unpacked_symmetric_extension_kernel(ll, lh, hl, hh, lh, hl, ll + band_stride_y, hl + band_stride_y);

                ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y;
            }
        }

        // bottom row
        float * ll10 = band_end_y == band_size_y ? ll : ll + band_stride_y;
        float * hl10 = band_end_y == band_size_y ? hl : hl + band_stride_y;
        float * hl1 = band_end_y - band_start_y == 1 ? tmp_hl : hl;
        for(size_t x = band_start_x; x < band_end_x - 4; x += 4){
            predict_unpacked_kernel(ll, lh, hl, hh, tmp_lh, hl1, ll10, hl10);

            ll += 4; hl += 4; hh += 4; lh += 4; ll10 += 4; hl10 += 4; tmp_lh += 4; hl1 += 4;
        }

        // botom right
        predict_unpacked_symmetric_extension_kernel(ll, lh, hl, hh, tmp_lh, hl1, ll10, hl10);
    } else {
        if(band_end_y - band_start_y > 1){
            predict_unpacked_kernel(ll, lh, hl, hh, tmp_lh, tmp_hl, ll + band_stride_y, hl + band_stride_y);

            ll += 4; hl += 4; hh += 4; lh += 4; tmp_hl += 4; tmp_lh += 4;

            for(size_t x = band_start_x + 4; x < band_end_x - 4; x += 4){
                predict_unpacked_kernel(ll, lh, hl, hh, lh, tmp_hl, ll + band_stride_y, hl + band_stride_y);

                ll += 4; hl += 4; hh += 4; lh += 4; tmp_hl += 4;
            }
            // right
            predict_unpacked_kernel(ll, lh, hl, hh, lh, tmp_hl, ll + band_stride_y, hl + band_stride_y);

            ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y; tmp_hl += 4;
        }

        if(band_end_y - band_start_y > 2){
            for(size_t y = band_start_y + 1; y < band_end_y - 1; ++y){
                predict_unpacked_kernel(ll, lh, hl, hh, tmp_lh, hl, ll + band_stride_y, hl + band_stride_y);

                ll += 4; hl += 4; hh += 4; lh += 4; tmp_lh += 4;

                for(size_t x = band_start_x + 4; x < band_end_x - 4; x += 4){
                    predict_unpacked_kernel(ll, lh, hl, hh, lh, hl, ll + band_stride_y, hl + band_stride_y);

                    ll += 4; hl += 4; hh += 4; lh += 4;
                }

                // right
                predict_unpacked_kernel(ll, lh, hl, hh, lh, tmp_hl, ll + band_stride_y, hl + band_stride_y);

                ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y; tmp_hl += 4;
            }
        }

        // bottom row
        float * ll10 = band_end_y == band_size_y ? ll : ll + band_stride_y;
        float * hl10 = band_end_y == band_size_y ? hl : hl + band_stride_y;
        float * hl1 = band_end_y - band_start_y == 1 ? tmp_hl : hl;
        for(size_t x = band_start_x; x < band_end_x - 4; x += 4){
            predict_unpacked_kernel(ll, lh, hl, hh, tmp_lh, hl1, ll10, hl10);

            ll += 4; hl += 4; hh += 4; lh += 4; ll10 += 4; hl10 += 4; tmp_lh += 4; hl1 += 4;
        }

        predict_unpacked_kernel(ll, lh, hl, hh, tmp_lh, band_end_y - band_start_y == 1 ? hl1 : tmp_hl, ll10, hl10);
    }

    #pragma omp barrier
}

static void update_unpacked_2(const TransformStepArguments * tsa)
{
    const size_t band_stride_y = tsa->tile_bands.stride_y;
    const size_t band_size_x = tsa->tile_bands.size_x;

    const size_t tid = (size_t)omp_get_thread_num();

    const size_t band_start_y = tsa->threading_info->band_start_y[tid];
    const size_t band_end_y = tsa->threading_info->band_end_y[tid];
    const size_t band_start_x = tsa->threading_info->band_start_x[tid];
    const size_t band_end_x = tsa->threading_info->band_end_x[tid];

    const size_t next_band_y = ((band_end_x - band_start_x) - 4 - band_stride_y);

    float * ll = tsa->tile_bands.LL + (band_end_y - 1) * band_stride_y + band_end_x - 4;
    float * hl = tsa->tile_bands.HL + (band_end_y - 1) * band_stride_y + band_end_x - 4;
    float * lh = tsa->tile_bands.LH + (band_end_y - 1) * band_stride_y + band_end_x - 4;
    float * hh = tsa->tile_bands.HH + (band_end_y - 1) * band_stride_y + band_end_x - 4;

    const size_t num_tmp_cache_lines = ((tsa->tmp.size_y * tsa->tmp.size_x) * sizeof(float)) / CACHE_LINE_SIZE;
    const size_t num_tmp_elements_per_thread = (num_tmp_cache_lines / (size_t) omp_get_num_threads()) * (CACHE_LINE_SIZE / sizeof(float));
    const size_t tmp_offset = tid * num_tmp_elements_per_thread;
    const size_t tmp_offset_left = (tid - 1) * num_tmp_elements_per_thread;
    const size_t tmp_offset_upper = (tid - tsa->threading_info->thread_cols) * num_tmp_elements_per_thread;

    if(band_start_x == 0 && band_end_x == band_size_x){
        float * tmp_hl = tsa->tmp.HL + tmp_offset + (band_end_x - band_start_x) - 4;
        float * tmp_lh = tsa->tmp.LH + tmp_offset + (band_end_x - band_start_x) - 4;
        float * tmp_lh1 = tsa->tmp.LH + tmp_offset_upper + (band_end_x - band_start_x) - 4;

        if(band_end_y - band_start_y > 1){
            for(size_t x = band_end_x - 4; x >= band_start_x + 4; x-=4){ // main area
                update_unpacked_kernel2(ll, lh, hl, hh, tmp_lh, hl, hl, lh - band_stride_y, hh - band_stride_y);

                lh -= 4; ll -= 4; hl-= 4; hh -= 4; tmp_lh -= 4;
            }
            // left extension
            update_unpacked_symmetric_extension_kernel2(ll, lh, hl, hh, tmp_lh, hl, lh - band_stride_y, hh - band_stride_y);

            ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y;
        }

        if(band_end_y - band_start_y > 2){
            for(size_t y = band_end_y - 2; y > band_start_y; --y){
                for(size_t x = band_end_x - 4; x >= band_start_x + 4; x-=4){ // main area
                    update_unpacked_kernel2(ll, lh, hl, hh, lh, hl, hl, lh - band_stride_y, hh - band_stride_y);

                    lh -= 4; ll -= 4; hl-= 4; hh -= 4;
                }
                // left extension
                update_unpacked_symmetric_extension_kernel2(ll, lh, hl, hh, lh, hl, lh - band_stride_y, hh - band_stride_y);

                ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y;
            }
        }

        if(band_end_y - band_start_y == 1)
            tmp_lh1 = band_start_y == 0 ? tmp_lh : tmp_lh1;
        else
            tmp_lh1 = band_start_y == 0 ? lh : tmp_lh1;

        float * hh1 = band_start_y == 0 ? hh : hh - band_stride_y;
        float * tmp_lh2 = band_end_y - band_start_y == 1 ? tmp_lh : lh;
        for(size_t x = band_end_x - 4; x >= band_start_x + 4; x-=4){ // top row
            update_unpacked_kernel2(ll, lh, hl, hh, tmp_lh2, tmp_hl, tmp_hl, tmp_lh1, hh1);

            lh -= 4; ll -= 4; hl-= 4; hh -= 4; hh1 -= 4; tmp_lh1 -= 4; tmp_hl -= 4; tmp_lh2 -= 4;
        }
        // left extension
        update_unpacked_symmetric_extension_kernel2(ll, lh, hl, hh, tmp_lh2, tmp_hl, tmp_lh1, hh1);
    } else if(band_start_x == 0){
        float * tmp_hl = tsa->tmp.HL + tmp_offset + (band_end_x - band_start_x) - 4 + 4 * (band_end_y - band_start_y) - 4;
        float * tmp_lh = tsa->tmp.LH + tmp_offset + (band_end_x - band_start_x) - 4;
        float * tmp_lh1 = tsa->tmp.LH + tmp_offset_upper + (band_end_x - band_start_x) - 4;

        if(band_end_y - band_start_y > 1){
            update_unpacked_kernel2(ll, lh, hl, hh, tmp_lh, tmp_hl, hl, lh - band_stride_y, hh - band_stride_y);

            lh -= 4; ll -= 4; hl -= 4; hh -= 4; tmp_hl -= 4; tmp_lh -= 4;

            for(size_t x = band_end_x - 8; x >= band_start_x + 4; x-=4){ // main area
                update_unpacked_kernel2(ll, lh, hl, hh, tmp_lh, hl, hl, lh - band_stride_y, hh - band_stride_y);

                lh -= 4; ll -= 4; hl-= 4; hh -= 4; tmp_lh -= 4;
            }
            // left extension
            update_unpacked_symmetric_extension_kernel2(ll, lh, hl, hh, tmp_lh, hl, lh - band_stride_y, hh - band_stride_y);

            ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y;
        }

        if(band_end_y - band_start_y > 2){
            for(size_t y = band_end_y - 2; y > band_start_y; --y){
                update_unpacked_kernel2(ll, lh, hl, hh, lh, tmp_hl, hl, lh - band_stride_y, hh - band_stride_y);

                lh -= 4; ll -= 4; hl-= 4; hh -= 4; tmp_hl -= 4;

                for(size_t x = band_end_x - 8; x >= band_start_x + 4; x-=4){ // main area
                    update_unpacked_kernel2(ll, lh, hl, hh, lh, hl, hl, lh - band_stride_y, hh - band_stride_y);

                    lh -= 4; ll -= 4; hl-= 4; hh -= 4;
                }
                // left extension
                update_unpacked_symmetric_extension_kernel2(ll, lh, hl, hh, lh, hl, lh - band_stride_y, hh - band_stride_y);

                ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y;
            }
        }

        if(band_end_y - band_start_y == 1)
            tmp_lh1 = band_start_y == 0 ? tmp_lh : tmp_lh1;
        else
            tmp_lh1 = band_start_y == 0 ? tsa->tile_bands.LH + band_end_x - 4 : tmp_lh1;
        float * tmp_lh2 = band_end_y - band_start_y == 1 ? tmp_lh : lh;
        float * hh1 = band_start_y == 0 ? tsa->tile_bands.HH + band_end_x - 4 : hh - band_stride_y;
        for(size_t x = band_end_x - 4; x >= band_start_x + 4; x-=4){ // top row
            update_unpacked_kernel2(ll, lh, hl, hh, tmp_lh2, tmp_hl, tmp_hl, tmp_lh1, hh1);

            lh -= 4; ll -= 4; hl-= 4; hh -= 4; hh1 -= 4; tmp_lh1 -= 4; tmp_hl -= 4; tmp_lh2 -= 4;
        }
        // left extension
        update_unpacked_symmetric_extension_kernel2(ll, lh, hl, hh, tmp_lh2, tmp_hl, tmp_lh1, hh1);
    } else if(band_end_x == band_size_x){
        float * tmp_hl = tsa->tmp.HL + tmp_offset + (band_end_x - band_start_x) - 4;
        float * tmp_hl1 = tsa->tmp.HL + tmp_offset_left + (band_end_x - band_start_x) - 4 + 4 * (tsa->threading_info->band_end_y[tid-1] - tsa->threading_info->band_start_y[tid-1]);
        float * tmp_lh = tsa->tmp.LH + tmp_offset + (band_end_x - band_start_x) - 4 + 4 * (band_end_y - band_start_y) - 4;
        float * tmp_lh1;

        if(band_end_y - band_start_y > 1){
            for(size_t x = band_end_x - 4; x >= band_start_x + 4; x-=4){ // main area
                update_unpacked_kernel2(ll, lh, hl, hh, tmp_lh, hl, hl, lh - band_stride_y, hh - band_stride_y);

                lh -= 4; ll -= 4; hl-= 4; hh -= 4; tmp_lh -= 4;
            }
            // left extension
            update_unpacked_kernel2(ll, lh, hl, hh, tmp_lh, hl, tmp_hl1, tmp_lh - 4, hh - band_stride_y);

            ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y; tmp_hl1 -= 4; tmp_lh -= 4;
        }

        if(band_end_y - band_start_y > 2){
            for(size_t y = band_end_y - 2; y > band_start_y; --y){
                for(size_t x = band_end_x - 4; x >= band_start_x + 4; x-=4){ // main area
                    update_unpacked_kernel2(ll, lh, hl, hh, lh, hl, hl, lh - band_stride_y, hh - band_stride_y);

                    lh -= 4; ll -= 4; hl-= 4; hh -= 4;
                }
                // left extension
                update_unpacked_kernel2(ll, lh, hl, hh, tmp_lh, hl, tmp_hl1, tmp_lh - 4, hh - band_stride_y);

                ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y;
                tmp_hl1 -= 4; tmp_lh -= 4;
            }
        }

        if(band_end_y - band_start_y == 1)
            tmp_lh1 = band_start_y == 0 ? tmp_lh : tsa->tmp.LH + tmp_offset_upper + (band_end_x - band_start_x) - 4 + 4 * (tsa->threading_info->band_end_y[tid-tsa->threading_info->thread_cols] - tsa->threading_info->band_start_y[tid-tsa->threading_info->thread_cols]) - 4;
        else
            tmp_lh1 = band_start_y == 0 ? tsa->tile_bands.LH + band_end_x - 4 : tsa->tmp.LH + tmp_offset_upper + (band_end_x - band_start_x) - 4 + 4 * (tsa->threading_info->band_end_y[tid-tsa->threading_info->thread_cols] - tsa->threading_info->band_start_y[tid-tsa->threading_info->thread_cols]) - 4;
        float * tmp_lh2 = band_end_y - band_start_y == 1 ? tmp_lh : lh;
        float * hh1 = band_start_y == 0 ? tsa->tile_bands.HH + band_end_x - 4 : hh - band_stride_y;
        for(size_t x = band_end_x - 4; x >= band_start_x + 4; x-=4){ // top row
            update_unpacked_kernel2(ll, lh, hl, hh, tmp_lh2, tmp_hl, tmp_hl, tmp_lh1, hh1);

            lh -= 4; ll -= 4; hl-= 4; hh -= 4; hh1 -= 4; tmp_lh1 -= 4; tmp_hl -= 4; tmp_lh2 -= 4;
        }
        if(band_end_y - band_start_y == 1)
            tmp_lh1 = band_start_y == 0 ? tmp_lh2 : tmp_lh1;
        else
            tmp_lh1 = band_start_y == 0 ? tmp_lh : tmp_lh1;
        update_unpacked_kernel2(ll, lh, hl, hh, band_end_y - band_start_y == 1 ? tmp_lh2 : tmp_lh, tmp_hl, tmp_hl1, tmp_lh1, hh1);
    } else {
        float * tmp_hl = tsa->tmp.HL + tmp_offset + (band_end_x - band_start_x) - 4 + 4 * (band_end_y - band_start_y) - 4;
        float * tmp_hl1 = tsa->tmp.HL + tmp_offset_left + (band_end_x - band_start_x) - 4 + 4 * (tsa->threading_info->band_end_y[tid-1] - tsa->threading_info->band_start_y[tid-1]);
        float * tmp_lh = tsa->tmp.LH + tmp_offset + (band_end_x - band_start_x) - 4 + 4 * (band_end_y - band_start_y) - 4;
        float * tmp_lh1;

        if(band_end_y - band_start_y > 1){
            update_unpacked_kernel2(ll, lh, hl, hh, tmp_lh, tmp_hl, hl, lh - band_stride_y, hh - band_stride_y);
            lh -= 4; ll -= 4; hl -= 4; hh -= 4; tmp_hl -= 4; tmp_lh -= 4;

            for(size_t x = band_end_x - 8; x >= band_start_x + 4; x -= 4){ // main area
                update_unpacked_kernel2(ll, lh, hl, hh, tmp_lh, hl, hl, lh - band_stride_y, hh - band_stride_y);

                lh -= 4; ll -= 4; hl -= 4; hh -= 4; tmp_lh -= 4;
            }
            // left extension
            update_unpacked_kernel2(ll, lh, hl, hh, tmp_lh, hl, tmp_hl1, tmp_lh - 4, hh - band_stride_y);

            ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y; tmp_hl1 -= 4; tmp_lh -= 4;
        }

        if(band_end_y - band_start_y > 2){
            for(size_t y = band_end_y - 2; y > band_start_y; --y){
                update_unpacked_kernel2(ll, lh, hl, hh, lh, tmp_hl, hl, lh - band_stride_y, hh - band_stride_y);
                lh -= 4; ll -= 4; hl -= 4; hh -= 4; tmp_hl -= 4;

                for(size_t x = band_end_x - 8; x >= band_start_x + 4; x -= 4){ // main area
                    update_unpacked_kernel2(ll, lh, hl, hh, lh, hl, hl, lh - band_stride_y, hh - band_stride_y);

                    lh -= 4; ll -= 4; hl -= 4; hh -= 4;
                }
                // left extension
                update_unpacked_kernel2(ll, lh, hl, hh, tmp_lh, hl, tmp_hl1, tmp_lh - 4, hh - band_stride_y);

                ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y; tmp_hl1 -= 4; tmp_lh -= 4;
            }
        }

        if(band_end_y - band_start_y == 1)
            tmp_lh1 = band_start_y == 0 ? tmp_lh : tsa->tmp.LH + tmp_offset_upper + (band_end_x - band_start_x) - 4 + 4 * (tsa->threading_info->band_end_y[tid-tsa->threading_info->thread_cols] - tsa->threading_info->band_start_y[tid-tsa->threading_info->thread_cols]) - 4;
        else
            tmp_lh1 = band_start_y == 0 ? tsa->tile_bands.LH + band_end_x - 4 : tsa->tmp.LH + tmp_offset_upper + (band_end_x - band_start_x) - 4 + 4 * (tsa->threading_info->band_end_y[tid-tsa->threading_info->thread_cols] - tsa->threading_info->band_start_y[tid-tsa->threading_info->thread_cols]) - 4;
        float * tmp_lh2 = band_end_y - band_start_y == 1 ? tmp_lh : lh;
        float * hh1 = band_start_y == 0 ? tsa->tile_bands.HH + band_end_x - 4 : hh - band_stride_y;
        update_unpacked_kernel2(ll, lh, hl, hh, tmp_lh2, tmp_hl, tmp_hl, tmp_lh1, hh1);

        lh -= 4; ll -= 4; hl -= 4; hh -= 4; hh1 -= 4; tmp_lh1 -= 4; tmp_hl -= 4; tmp_lh2 -= 4;

        for(size_t x = band_end_x - 8; x >= band_start_x + 4; x -= 4){ // top row
            update_unpacked_kernel2(ll, lh, hl, hh, tmp_lh2, tmp_hl, tmp_hl, tmp_lh1, hh1);

            lh -= 4; ll -= 4; hl -= 4; hh -= 4; hh1 -= 4; tmp_lh1 -= 4; tmp_hl -= 4; tmp_lh2 -= 4;
        }

        // left extension
        if(band_end_y - band_start_y == 1)
            tmp_lh1 = band_start_y == 0 ? tmp_lh2 : tmp_lh1;
        else
            tmp_lh1 = band_start_y == 0 ? tmp_lh : tmp_lh1;
        update_unpacked_kernel2(ll, lh, hl, hh, band_end_y - band_start_y == 1 ? tmp_lh2 : tmp_lh, tmp_hl, tmp_hl1, tmp_lh1, hh1);
    }
}

TRANSFORM_STEP_FUNC NO_TREE_VECTORIZE cdf97_non_separable_lifting_amd64_sse(size_t step)
{
    switch(step){
        case 0:
            return predict_with_unpack;
        case 1:
            return update_unpacked;
        case 2:
            return predict_unpacked;
        case 3:
            return update_unpacked_2;
        default:
            return NULL;
    }
}

void NO_TREE_VECTORIZE cdf97_non_separable_lifting_amd64_sse_transform_tile(const TransformStepArguments * tsa)
{
    predict_with_unpack(tsa);
    update_unpacked(tsa);
    predict_unpacked(tsa);
    update_unpacked_2(tsa);
}
#endif
