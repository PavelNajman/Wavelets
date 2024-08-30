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

static inline void FORCE_INLINE unpack_kernel(__m128 R0C0, __m128 R0C1, __m128 R1C0, __m128 R1C1, float * ll, float * hl, float * lh, float * hh)
{
    const __m128 ALPHA = _mm_set_ps1(0.5f);

    __m128 LL00, HL00, LH00, HH00, TMP;

    LL00 = load_packed_LX00(R0C0, R0C1);
    HL00 = load_packed_HX00(R0C0, R0C1); TMP = HL00;
    LH00 = load_packed_LX00(R1C0, R1C1);
    HH00 = load_packed_HX00(R1C0, R1C1);
    
    HH00 = _mm_add_ps(_mm_sub_ps(HH00, LH00), _mm_sub_ps(LL00, HL00));
    LH00 = _mm_sub_ps(mul_add(ALPHA, HH00, LH00), LL00);
    HL00 = _mm_sub_ps(mul_add(ALPHA, HH00, HL00), LL00);
    LL00 = mul_add(ALPHA, _mm_add_ps(_mm_sub_ps(TMP, LL00), LH00), LL00);
    
    _mm_store_ps(ll, LL00);
    _mm_store_ps(hl, HL00);
    _mm_store_ps(lh, LH00);
    _mm_store_ps(hh, HH00);
}

static void single_loop_with_unpack(const TransformStepArguments * tsa)
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

    const size_t next_band_y = band_stride_y - (band_end_x - band_start_x);
    const size_t next_tile_y = 2 * stride_y - (img_end_x - img_start_x);

    float * m0 = mem + img_start_x + img_start_y * stride_y;
    float * m1 = mem + img_start_x + (img_start_y+1) * stride_y;

    float * ll = tsa->tile_bands.LL + band_start_x + band_start_y * band_stride_y;
    float * hl = tsa->tile_bands.HL + band_start_x + band_start_y * band_stride_y;
    float * lh = tsa->tile_bands.LH + band_start_x + band_start_y * band_stride_y;
    float * hh = tsa->tile_bands.HH + band_start_x + band_start_y * band_stride_y;

    __m128 R0C0, R0C1, R1C0, R1C1;

    if(img_end_x == size_x){
        for(size_t y = img_start_y; y < img_end_y; y += 2){
            for(size_t x = img_start_x; x < img_end_x; x+=8){
                R0C0 = _mm_load_ps(m0);   R1C0 = _mm_load_ps(m1);
                R0C1 = _mm_load_ps(m0+4); R1C1 = _mm_load_ps(m1+4);

                unpack_kernel(R0C0, R0C1, R1C0, R1C1, ll, hl, lh, hh);

                ll += 4; hl += 4; hh += 4; lh += 4;
                m0 += 8; m1 += 8;
            }

            ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y;
        }
    } else {
        for(size_t y = img_start_y; y < img_end_y; y += 2){            
            for(size_t x = img_start_x; x < img_end_x; x+=8){
                R0C0 = _mm_load_ps(m0);   R1C0 = _mm_load_ps(m1);
                R0C1 = _mm_load_ps(m0+4); R1C1 = _mm_load_ps(m1+4);

                unpack_kernel(R0C0, R0C1, R1C0, R1C1, ll, hl, lh, hh);

                ll += 4; hl += 4; hh += 4; lh += 4;
                m0 += 8; m1 += 8;
            }

            ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y;
        }
    }

    #pragma omp barrier
}

TRANSFORM_STEP_FUNC NO_TREE_VECTORIZE haar_single_loop_amd64_sse(size_t step)
{
    switch(step){
        case 0:
            return single_loop_with_unpack;
        default:
            return NULL;
    }
}

void NO_TREE_VECTORIZE haar_single_loop_amd64_sse_transform_tile(const TransformStepArguments * tsa)
{
    single_loop_with_unpack(tsa);
}
#endif
