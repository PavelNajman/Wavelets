#ifdef _OPENMP
    #include <omp.h>
#endif

#include "common.h"

static inline void FORCE_INLINE NO_TREE_VECTORIZE unpack_kernel(const float * m0, const float * m1, float * ll, float * hl, float * lh, float * hh)
{  
    float LL00 = *m0;
    float HL00 = *(m0+1);
    float LH00 = *m1;
    float HH00 = *(m1+1);

    HL00 -= LL00;
    HH00 -= LH00;
    
    LH00 -= LL00;
    HH00 -= HL00;
    
    LL00 += 0.5f * (HL00);
    LH00 += 0.5f * (HH00);
    
    LL00 += 0.5f * (LH00);
    HL00 += 0.5f * (HH00);
    
    *hh = HH00;
    *lh = LH00;
    *hl = HL00;
    *ll = LL00;
}

static void NO_TREE_VECTORIZE single_loop_with_unpack(const TransformStepArguments * tsa)
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
    
    if(img_end_x == size_x){
        for(size_t y = img_start_y; y < img_end_y; y += 2){
            for(size_t x = img_start_x; x < img_end_x; x += 2){
                unpack_kernel(m0, m1, ll, hl, lh, hh);

                ++ll; ++hl; ++hh; ++lh;
                m0 += 2; m1 += 2;
            }

            ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y;
        }
    } else { 
        for(size_t y = img_start_y; y < img_end_y; y += 2){
            for(size_t x = img_start_x; x < img_end_x; x += 2){
                unpack_kernel(m0, m1, ll, hl, lh, hh);

                ++ll; ++hl; ++hh; ++lh;
                m0 += 2; m1 += 2;
            }

            ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y;
        }
    }
    #pragma omp barrier
}

TRANSFORM_STEP_FUNC NO_TREE_VECTORIZE haar_single_loop_generic(size_t step)
{
    switch(step){
        case 0: 
            return single_loop_with_unpack;
        default:
            return NULL;
    }
}

void NO_TREE_VECTORIZE haar_single_loop_generic_transform_tile(const TransformStepArguments * tsa)
{
    single_loop_with_unpack(tsa);    
}
