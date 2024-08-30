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

static inline void FORCE_INLINE H_predict_with_unpack_kernel(__m128 R0C0, __m128 R0C1, __m128 R0C2, __m128 R1C0, __m128 R1C1, __m128 R1C2, float * ll, float * hl, float * lh, float * hh)
{
    const __m128 ALPHA = _mm_set_ps1(-0.5f);

    __m128 LL00, HL00, LH00, HH00, TMP, RES;

    LL00 = load_packed_LX00(R0C0, R0C1);
    HL00 = load_packed_HX00(R0C0, R0C1);
    LH00 = load_packed_LX00(R1C0, R1C1);
    HH00 = load_packed_HX00(R1C0, R1C1);

    TMP = load_packed_LX01(LL00, R0C2);

    RES = _mm_add_ps(LL00, TMP);
#ifdef USE_FMA
    HL00 = _mm_fmadd_ps(RES, ALPHA, HL00);
#else
    RES = _mm_mul_ps(RES, ALPHA);
    HL00 = _mm_add_ps(RES, HL00);
#endif

    TMP = load_packed_LX01(LH00, R1C2);

    RES = _mm_add_ps(LH00, TMP);
#ifdef USE_FMA
    HH00 = _mm_fmadd_ps(RES, ALPHA, HH00);
#else
    RES = _mm_mul_ps(RES, ALPHA);
    HH00 = _mm_add_ps(RES, HH00);
#endif

    _mm_store_ps(ll, LL00);
    _mm_store_ps(hl, HL00);
    _mm_store_ps(lh, LH00);
    _mm_store_ps(hh, HH00);
}

static inline void FORCE_INLINE H_predict_with_unpack_symmetric_extension_kernel(__m128 R0C0, __m128 R0C1, __m128 R1C0, __m128 R1C1, float * ll, float * hl, float * lh, float * hh)
{
    const __m128 ALPHA = _mm_set_ps1(-0.5f);

    __m128 LL00, HL00, LH00, HH00, TMP, RES;

    LL00 = load_packed_LX00(R0C0, R0C1);
    HL00 = load_packed_HX00(R0C0, R0C1);
    LH00 = load_packed_LX00(R1C0, R1C1);
    HH00 = load_packed_HX00(R1C0, R1C1);

    TMP = load_unpacked_LX01S(LL00);

    RES = _mm_add_ps(LL00, TMP);
#ifdef USE_FMA
    HL00 = _mm_fmadd_ps(RES, ALPHA, HL00);
#else
    RES = _mm_mul_ps(RES, ALPHA);
    HL00 = _mm_add_ps(RES, HL00);
#endif

    TMP = load_unpacked_LX01S(LH00);

    RES = _mm_add_ps(LH00, TMP);
#ifdef USE_FMA
    HH00 = _mm_fmadd_ps(RES, ALPHA, HH00);
#else
    RES = _mm_mul_ps(RES, ALPHA);
    HH00 = _mm_add_ps(RES, HH00);
#endif

    _mm_store_ps(ll, LL00);
    _mm_store_ps(hl, HL00);
    _mm_store_ps(lh, LH00);
    _mm_store_ps(hh, HH00);
}

static inline void FORCE_INLINE H_update_unpacked_kernel(__m128 C0, __m128 X0, __m128 C1, __m128 X1, float * ll, float *lh)
{
    const __m128 BETA = _mm_set1_ps(0.25f);

    __m128 RES1, RES2, TMP1, TMP2;

    RES1 = _mm_load_ps(ll);
    RES2 = _mm_load_ps(lh);

    TMP1 = load_unpacked_XX01L(C0, C1);
    TMP2 = load_unpacked_XX01L(X0, X1);

    C1 = _mm_add_ps(C0, TMP1);
    X1 = _mm_add_ps(X0, TMP2);

#ifdef USE_FMA
    RES1 = _mm_fmadd_ps(C1, BETA, RES1);
    RES2 = _mm_fmadd_ps(X1, BETA, RES2);
#else
    C1 = _mm_mul_ps(C1, BETA);
    RES1 = _mm_add_ps(C1, RES1);
    X1 = _mm_mul_ps(X1, BETA);
    RES2 = _mm_add_ps(X1, RES2);
#endif
    _mm_store_ps(ll, RES1);
    _mm_store_ps(lh, RES2);
}

static inline void FORCE_INLINE H_update_unpacked_symmetric_extension_kernel(__m128 C0, __m128 X0, float * ll, float *lh)
{
    const __m128 BETA = _mm_set1_ps(0.25f);

    __m128 C1, X1, RES1, RES2, TMP1, TMP2;

    RES1 = _mm_load_ps(ll);
    RES2 = _mm_load_ps(lh);

    TMP1 = load_unpacked_HX01LS(C0);
    TMP2 = load_unpacked_HX01LS(X0);

    C1 = _mm_add_ps(C0, TMP1);
    X1 = _mm_add_ps(X0, TMP2);
#ifdef USE_FMA
    RES1 = _mm_fmadd_ps(C1, BETA, RES1);
    RES2 = _mm_fmadd_ps(X1, BETA, RES2);
#else
    C1 = _mm_mul_ps(C1, BETA);
    RES1 = _mm_add_ps(C1, RES1);
    X1 = _mm_mul_ps(X1, BETA);
    RES2 = _mm_add_ps(X1, RES2);
#endif
    _mm_store_ps(ll, RES1);
    _mm_store_ps(lh, RES2);
}

static inline void FORCE_INLINE V_unpacked_kernel(float * A, float * B, const float * C, const float * D, const float * E, const float * F, const float ALPHA)
{
    const __m128 COEF = _mm_set1_ps(ALPHA);

    __m128 R1 = _mm_add_ps(_mm_load_ps(C), _mm_load_ps(E));
    __m128 R2 = _mm_add_ps(_mm_load_ps(D), _mm_load_ps(F));
#ifdef USE_FMA
    R1 = _mm_fmadd_ps(R1, COEF, _mm_load_ps(A));
    R2 = _mm_fmadd_ps(R2, COEF, _mm_load_ps(B));
#else
    R1 = _mm_mul_ps(R1, COEF);
    R1 = _mm_add_ps(R1, _mm_load_ps(A));
    R2 = _mm_mul_ps(R2, COEF);
    R2 = _mm_add_ps(R2, _mm_load_ps(B));
#endif

    _mm_store_ps(A, R1);
    _mm_store_ps(B, R2);
}

static void H_predict_with_unpack(const TransformStepArguments * tsa)
{
    const size_t size_x = tsa->tile.size_x;
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

    const size_t next_band_y = band_stride_y - (band_end_x - band_start_x) + 4;
    const size_t next_tile_y = 2 * stride_y - (img_end_x - img_start_x) + 8;

    float * m0 = mem + img_start_x + img_start_y * stride_y;
    float * m1 = mem + img_start_x + (img_start_y+1) * stride_y;

    float * ll = tsa->tile_bands.LL + band_start_x + band_start_y * band_stride_y;
    float * hl = tsa->tile_bands.HL + band_start_x + band_start_y * band_stride_y;
    float * lh = tsa->tile_bands.LH + band_start_x + band_start_y * band_stride_y;
    float * hh = tsa->tile_bands.HH + band_start_x + band_start_y * band_stride_y;

    __m128 R0C0, R0C1, R0C2, R1C0, R1C1, R1C2;

    if(img_end_x == size_x){
        for(size_t y = img_start_y; y < img_end_y; y += 2){
            R0C0 = _mm_load_ps(m0); R1C0 = _mm_load_ps(m1);
            for(size_t x = img_start_x; x < img_end_x-8; x+=8){
                R0C1 = _mm_load_ps(m0+4); R0C2 = _mm_load_ps(m0+8);
                R1C1 = _mm_load_ps(m1+4); R1C2 = _mm_load_ps(m1+8);

                H_predict_with_unpack_kernel(R0C0, R0C1, R0C2, R1C0, R1C1, R1C2, ll, hl, lh, hh);

                R0C0 = R0C2; R1C0 = R1C2;

                ll += 4; hl += 4; hh += 4; lh += 4;
                m0 += 8; m1 += 8;
            }

            R0C1 = _mm_load_ps(m0+4); R1C1 = _mm_load_ps(m1+4);
            H_predict_with_unpack_symmetric_extension_kernel(R0C0, R0C1, R1C0, R1C1, ll, hl, lh, hh);

            ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y;
        }
    } else {
        for(size_t y = img_start_y; y < img_end_y; y += 2){
            R0C0 = _mm_load_ps(m0); R1C0 = _mm_load_ps(m1);
            for(size_t x = img_start_x; x < img_end_x-8; x+=8){
                R0C1 = _mm_load_ps(m0+4); R0C2 = _mm_load_ps(m0+8);
                R1C1 = _mm_load_ps(m1+4); R1C2 = _mm_load_ps(m1+8);

                H_predict_with_unpack_kernel(R0C0, R0C1, R0C2, R1C0, R1C1, R1C2, ll, hl, lh, hh);

                R0C0 = R0C2; R1C0 = R1C2;

                ll += 4; hl += 4; hh += 4; lh += 4;
                m0 += 8; m1 += 8;
            }

            R0C1 = _mm_load_ps(m0+4); R1C1 = _mm_load_ps(m1+4);
            R0C2 = _mm_load_ps(m0+8); R1C2 = _mm_load_ps(m1+8);
            H_predict_with_unpack_kernel(R0C0, R0C1, R0C2, R1C0, R1C1, R1C2, ll, hl, lh, hh);

            ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y;
        }
    }

    #pragma omp barrier
}

static void H_update_unpacked(const TransformStepArguments * tsa)
{
    const size_t band_stride_y = tsa->tile_bands.stride_y;

    const size_t tid = (size_t)omp_get_thread_num();

    const size_t band_start_y = tsa->threading_info->band_start_y[tid];
    const size_t band_end_y = tsa->threading_info->band_end_y[tid];
    const size_t band_start_x = tsa->threading_info->band_start_x[tid];
    const size_t band_end_x = tsa->threading_info->band_end_x[tid];

    const size_t next_y = band_stride_y - (band_end_x - band_start_x);

    __m128 C0, C1, X0, X1;

    float * ll = tsa->tile_bands.LL + band_start_x + band_start_y * band_stride_y;
    float * hl = tsa->tile_bands.HL + band_start_x + band_start_y * band_stride_y;
    float * lh = tsa->tile_bands.LH + band_start_x + band_start_y * band_stride_y;
    float * hh = tsa->tile_bands.HH + band_start_x + band_start_y * band_stride_y;

    if(band_start_x == 0){
        for(size_t y = band_start_y; y < band_end_y; ++y){
            // first 4 columns
            C0 = _mm_load_ps(hl); X0 = _mm_load_ps(hh);

            H_update_unpacked_symmetric_extension_kernel(C0, X0, ll, lh);

            C1 = C0; X1 = X0;

            ll += 4; hl += 4; lh += 4; hh += 4;

            //main area
            for(size_t x = band_start_x + 4; x < band_end_x; x += 4){

                C0 = _mm_load_ps(hl); X0 = _mm_load_ps(hh);

                H_update_unpacked_kernel(C0, X0, C1, X1, ll, lh);

                C1 = C0; X1 = X0;

                ll += 4; hl += 4; lh += 4; hh += 4;
            }

            ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        }
    } else {
        for(size_t y = band_start_y; y < band_end_y; ++y){
            // first 4 columns
            C0 = _mm_load_ps(hl); X0 = _mm_load_ps(hh);
            C1 = _mm_load_ps(hl-4); X1 = _mm_load_ps(hh-4);
            H_update_unpacked_kernel(C0, X0, C1, X1, ll, lh);

            C1 = C0; X1 = X0;

            ll += 4; hl += 4; lh += 4; hh += 4;

            //main area
            for(size_t x = band_start_x + 4; x < band_end_x; x += 4){

                C0 = _mm_load_ps(hl); X0 = _mm_load_ps(hh);

                H_update_unpacked_kernel(C0, X0, C1, X1, ll, lh);

                C1 = C0; X1 = X0;

                ll += 4; hl += 4; lh += 4; hh += 4;
            }


            ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        }
    }

    #pragma omp barrier
}

static void V_predict_unpacked(const TransformStepArguments * tsa)
{
    const size_t band_size_y = tsa->tile_bands.size_y;
    const size_t band_stride_y = tsa->tile_bands.stride_y;

    const size_t tid = (size_t)omp_get_thread_num();

    const size_t band_start_y = tsa->threading_info->band_start_y[tid];
    const size_t band_end_y = tsa->threading_info->band_end_y[tid];
    const size_t band_start_x = tsa->threading_info->band_start_x[tid];
    const size_t band_end_x = tsa->threading_info->band_end_x[tid];

    const size_t next_y = band_stride_y - (band_end_x - band_start_x);

    float * ll = tsa->tile_bands.LL + band_start_x + band_start_y * band_stride_y;
    float * hl = tsa->tile_bands.HL + band_start_x + band_start_y * band_stride_y;
    float * lh = tsa->tile_bands.LH + band_start_x + band_start_y * band_stride_y;
    float * hh = tsa->tile_bands.HH + band_start_x + band_start_y * band_stride_y;

    float * ll1 = tsa->tile_bands.LL + band_start_x + (band_start_y+1) * band_stride_y;
    float * hl1 = tsa->tile_bands.HL + band_start_x + (band_start_y+1) * band_stride_y;

    if(band_end_y != band_size_y){
        for(size_t y = band_start_y; y < band_end_y; ++y){
            for(size_t x = band_start_x; x < band_end_x; x += 4){

                V_unpacked_kernel(lh, hh, ll, hl, ll1, hl1, -0.5f);

                ll+=4; hl+=4; lh+=4; hh+=4; ll1+=4; hl1+=4;
            }

            ll += next_y; hl += next_y; lh += next_y; hh += next_y; ll1+=next_y; hl1+=next_y;
        }
    } else {
        for(size_t y = band_start_y; y < band_end_y-1; ++y){
            for(size_t x = band_start_x; x < band_end_x; x += 4){
                V_unpacked_kernel(lh, hh, ll, hl, ll1, hl1, -0.5f);

                ll+=4; hl+=4; lh+=4; hh+=4; ll1+=4; hl1+=4;
            }

            ll += next_y; hl += next_y; lh += next_y; hh += next_y; ll1+=next_y; hl1+=next_y;
        }
        // last row
        for(size_t x = band_start_x; x < band_end_x; x += 4){
            V_unpacked_kernel(lh, hh, ll, hl, ll, hl, -0.5f);

            ll+=4; hl+=4; lh+=4; hh+=4;
        }
    }

    #pragma omp barrier
}

static void V_update_unpacked(const TransformStepArguments * tsa)
{
    const size_t band_stride_y = tsa->tile_bands.stride_y;
    float * LH = tsa->tile_bands.LH;
    float * HH = tsa->tile_bands.HH;

    const size_t tid = (size_t)omp_get_thread_num();

    const size_t band_start_y = tsa->threading_info->band_start_y[tid];
    const size_t band_end_y = tsa->threading_info->band_end_y[tid];
    const size_t band_start_x = tsa->threading_info->band_start_x[tid];
    const size_t band_end_x = tsa->threading_info->band_end_x[tid];

    const size_t next_y = band_stride_y - (band_end_x - band_start_x);

    float * lh1, * hh1;

    float * ll = tsa->tile_bands.LL + band_start_x + band_start_y * band_stride_y;
    float * hl = tsa->tile_bands.HL + band_start_x + band_start_y * band_stride_y;
    float * lh = tsa->tile_bands.LH + band_start_x + band_start_y * band_stride_y;
    float * hh = tsa->tile_bands.HH + band_start_x + band_start_y * band_stride_y;

    if(band_start_y != 0){
        lh1 = LH + band_start_x + (band_start_y-1) * band_stride_y;
        hh1 = HH + band_start_x + (band_start_y-1) * band_stride_y;

        for(size_t y = band_start_y; y < band_end_y; ++y){
            for(size_t x = band_start_x; x < band_end_x; x += 4){

                V_unpacked_kernel(ll, hl, lh, hh, lh1, hh1, 0.25f);

                ll+=4; hl+=4; lh+=4; hh+=4; lh1+=4; hh1+=4;
            }

            ll += next_y; hl += next_y; lh += next_y; hh += next_y; lh1+=next_y; hh1+=next_y;
        }
    }else{
        // first row
        for(size_t x = band_start_x; x < band_end_x; x += 4){
            V_unpacked_kernel(ll, hl, lh, hh, lh, hh, 0.25f);
            ll+=4; hl+=4; lh+=4; hh+=4;
        }

        ll += next_y; hl += next_y; lh += next_y; hh += next_y;

        lh1 = LH + band_start_x + (band_start_y) * band_stride_y;
        hh1 = HH + band_start_x + (band_start_y) * band_stride_y;

        for(size_t y = band_start_y+1; y < band_end_y; ++y){
            for(size_t x = band_start_x; x < band_end_x; x += 4){
                V_unpacked_kernel(ll, hl, lh, hh, lh1, hh1, 0.25f);
                ll+=4; hl+=4; lh+=4; hh+=4; lh1+=4; hh1+=4;
            }
            ll += next_y; hl += next_y; lh += next_y; hh += next_y; lh1+=next_y; hh1+=next_y;
        }
    }
}

TRANSFORM_STEP_FUNC NO_TREE_VECTORIZE cdf53_separable_lifting_amd64_sse(size_t step)
{
    switch(step){
        case 0:
            return H_predict_with_unpack;
        case 1:
            return V_predict_unpacked;
        case 2:
            return H_update_unpacked;
        case 3:
            return V_update_unpacked;
        default:
            return NULL;
    }
}

void NO_TREE_VECTORIZE cdf53_separable_lifting_amd64_sse_transform_tile(const TransformStepArguments * tsa)
{
    H_predict_with_unpack(tsa);
    V_predict_unpacked(tsa);
    H_update_unpacked(tsa);
    V_update_unpacked(tsa);
}
#endif
