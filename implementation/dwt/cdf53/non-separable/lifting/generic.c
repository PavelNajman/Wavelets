#ifdef _OPENMP
    #include <omp.h>
#endif

#include "common.h"

static inline void FORCE_INLINE NO_TREE_VECTORIZE predict_with_unpack_kernel(const float * m0, const float * m1, const float * m2, float * ll, float * hl, float *lh, float * hh)
{
    float LL00, LL01, LL11, LL10, LH00, LH01, HH00, HL00, HL10;

    LL00 = *m0; LL01 = *(m0+2); LL10 = *m2; LL11 = *(m2+2);
    LH00 = *m1; LH01 = *(m1+2);
    HH00 = *(m1+1);
    HL00 = *(m0+1); HL10 = *(m2+1);

    HL00 += -0.5f * (LL01 + LL00);

    LH00 += -0.5f * (LL10);
    HH00 += 0.25f * LL11 + -0.5f * (HL10 + LH01 + LH00);
    LH00 += -0.5f * LL00;

    HH00 += -0.5f * HL00;

    *ll = LL00; *hl = HL00;
    *hh = HH00; *lh = LH00;
}

static inline void FORCE_INLINE NO_TREE_VECTORIZE predict_with_unpack_symmetric_extension_kernel(const float * m0, const float * m1, const float * m2, float * ll, float * hl, float *lh, float * hh)
{
    float LL00, LL01, LL11, LL10, LH00, LH01, HH00, HL00, HL10;

    LL00 = *m0; LL01 = *m0; LL10 = *m2; LL11 = *m2;
    LH00 = *m1; LH01 = *m1;
    HH00 = *(m1+1);
    HL00 = *(m0+1); HL10 = *(m2+1);

    HL00 += -0.5f * (LL01 + LL00);

    LH00 += -0.5f * (LL10);
    HH00 += 0.25f * LL11 + -0.5f * (HL10 + LH01 + LH00);
    LH00 += -0.5f * LL00;

    HH00 += -0.5f * HL00;

    *ll = LL00; *hl = HL00;
    *hh = HH00; *lh = LH00;
}

static inline void FORCE_INLINE NO_TREE_VECTORIZE update_unpacked_kernel(float * ll, float * lh, float * hl, const float * hh, const float *tmp_lh, const float *tmp_hl, const float *tmp_hl1, const float *tmp_lh1, const float *hh1)
{
    float LL00, HL01, HH11, LH10, LH00, HH01, HH00, HL00, HH10;

    LL00 = *ll; HL00 = *tmp_hl; LH00 = *tmp_lh; HH00 = *hh;
    HL01 = *(tmp_hl1 - 1); HH01 = *(hh - 1);
    LH10 = *(tmp_lh1); HH10 = *(hh1);
    HH11 = *(hh1 - 1);

    LL00 += 0.0625f * (HH11) + 0.25f * (LH10 + HL01);
    HL00 += 0.25f * HH10;
    LH00 += 0.25f * (HH01 + HH00);
    LL00 += 0.25f * (HL00 + LH00);
    HL00 += 0.25f * HH00;

    *lh = LH00;
    *ll = LL00;
    *hl = HL00;
}

static inline void FORCE_INLINE NO_TREE_VECTORIZE update_unpacked_symmetric_extension_kernel(float * ll, float * lh, float * hl, const float * hh, const float *tmp_lh, const float *tmp_hl, const float *tmp_lh1, const float *hh1)
{
    float LL00, HL01, HH11, LH10, LH00, HH01, HH00, HL00, HH10;

    LL00 = *ll; HL00 = *tmp_hl; LH00 = *tmp_lh; HH00 = *hh;
    HL01 = *(tmp_hl); HH01 = *(hh);
    LH10 = *(tmp_lh1); HH10 = *(hh1);
    HH11 = *(hh1);

    LL00 += 0.0625f * (HH11) + 0.25f * (LH10 + HL01);
    HL00 += 0.25f * HH10;
    LH00 += 0.25f * (HH01 + HH00);
    LL00 += 0.25f * (HL00 + LH00);
    HL00 += 0.25f * HH00;

    *lh = LH00;
    *ll = LL00;
    *hl = HL00;
}

static void NO_TREE_VECTORIZE predict_with_unpack(const TransformStepArguments * tsa)
{
    const size_t size_y = tsa->tile.size_y;
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

    const size_t next_band_y = band_stride_y - (band_end_x - band_start_x) + 1;
    const size_t next_tile_y = 2 * stride_y - (img_end_x - img_start_x) + 2;

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
    
    if(img_end_x == size_x){
        for(size_t y = img_start_y; y < img_end_y - 2; y += 2){
            for(size_t x = img_start_x; x < img_end_x - 2; x += 2){
                predict_with_unpack_kernel(m0, m1, m2, ll, hl, lh, hh);

                ll++; hh++; hl++; lh++;
                m0 += 2; m1 += 2; m2 += 2;
            }

            predict_with_unpack_symmetric_extension_kernel(m0, m1, m2, ll, hl, lh, hh);

            ll += next_band_y; hh += next_band_y; hl += next_band_y; lh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y;
        }

        m2 = img_end_y == size_y ? m2 - 2 * stride_y : m2;
        for(size_t x = img_start_x; x < img_end_x - 2; x += 2){
            predict_with_unpack_kernel(m0, m1, m2, ll, hl, tmp_lh, hh);

            ll++; hh++; hl++; tmp_lh++;
            m0 += 2; m1 += 2; m2 += 2;
        }

        predict_with_unpack_symmetric_extension_kernel(m0, m1, m2, ll, hl, tmp_lh, hh);
    } else {
        for(size_t y = img_start_y; y < img_end_y - 2; y += 2){
            for(size_t x = img_start_x; x < img_end_x - 2; x += 2){
                predict_with_unpack_kernel(m0, m1, m2, ll, hl, lh, hh);

                ll++; hh++; hl++; lh++;
                m0 += 2; m1 += 2; m2 += 2;
            }

            predict_with_unpack_kernel(m0, m1, m2, ll, tmp_hl, lh, hh);
            
            ll += next_band_y; hh += next_band_y; hl += next_band_y; lh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y;
            tmp_hl++;
        }

        m2 = img_end_y == size_y ? m2 - 2 * stride_y : m2;
        for(size_t x = img_start_x; x < img_end_x - 2; x += 2){
            predict_with_unpack_kernel(m0, m1, m2, ll, hl, tmp_lh, hh);

            ll++; hh++; hl++; tmp_lh++;
            m0 += 2; m1 += 2; m2 += 2;
        }
        
        predict_with_unpack_kernel(m0, m1, m2, ll, tmp_hl, tmp_lh, hh);
    }
    # pragma omp barrier
}

static void NO_TREE_VECTORIZE update_unpacked(const TransformStepArguments * tsa)
{
    const size_t band_stride_y = tsa->tile_bands.stride_y;

    const size_t tid = (size_t)omp_get_thread_num();

    const size_t band_size_x = tsa->tile_bands.size_x;
    
    const size_t band_start_y = tsa->threading_info->band_start_y[tid];
    const size_t band_end_y = tsa->threading_info->band_end_y[tid];
    const size_t band_start_x = tsa->threading_info->band_start_x[tid];
    const size_t band_end_x = tsa->threading_info->band_end_x[tid];

    const size_t next_band_y = ((band_end_x - band_start_x) - 1 - band_stride_y);

    float * ll = tsa->tile_bands.LL + (band_end_y-1) * band_stride_y + band_end_x - 1;
    float * hl = tsa->tile_bands.HL + (band_end_y-1) * band_stride_y + band_end_x - 1;
    float * lh = tsa->tile_bands.LH + (band_end_y-1) * band_stride_y + band_end_x - 1;
    float * hh = tsa->tile_bands.HH + (band_end_y-1) * band_stride_y + band_end_x - 1;
   
    const size_t num_tmp_cache_lines = ((tsa->tmp.size_y * tsa->tmp.size_x) * sizeof(float)) / CACHE_LINE_SIZE;
    const size_t num_tmp_elements_per_thread = (num_tmp_cache_lines / (size_t) omp_get_num_threads()) * (CACHE_LINE_SIZE / sizeof(float));
    const size_t tmp_offset = tid * num_tmp_elements_per_thread;
    const size_t tmp_offset_left = (tid - 1) * num_tmp_elements_per_thread;
    const size_t tmp_offset_upper = (tid - tsa->threading_info->thread_cols) * num_tmp_elements_per_thread;
    
    float * tmp_lh = tsa->tmp.LH + tmp_offset + (band_end_x - band_start_x) - 1;
    float * tmp_lh1 = tsa->tmp.LH + tmp_offset_upper + (band_end_x - band_start_x) - 1;
    float * hh1 = tsa->tile_bands.HH + (band_start_y - 1) * band_stride_y + band_end_x - 1;
    if(band_start_y == 0){
        hh1 = tsa->tile_bands.HH + band_end_x - 1;
        if(band_end_y - band_start_y == 1){
            tmp_lh1 = band_start_y == 0 ? tmp_lh : tmp_lh1;
        } else {
            tmp_lh1 = band_start_y == 0 ? tsa->tile_bands.LH + band_end_x - 1 : tmp_lh1;
        }
    }
    
    if(band_start_x == 0 && band_end_x == band_size_x){   
        if(band_end_y - band_start_y > 1){
            for(size_t x = band_end_x - 1; x >= band_start_x + 1; --x){
                update_unpacked_kernel(ll, lh, hl, hh, tmp_lh, hl, hl, lh - band_stride_y, hh  - band_stride_y);

                lh--; ll--; hl--; hh--; tmp_lh--;
            }

            // left extension
            update_unpacked_symmetric_extension_kernel(ll, lh, hl, hh, tmp_lh, hl, lh - band_stride_y, hh  - band_stride_y);

            ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y;
        }
        
        if(band_end_y - band_start_y > 2){
            for(size_t y = band_end_y-2; y > band_start_y; --y){
                for(size_t x = band_end_x - 1; x >= band_start_x + 1; --x){ // main area
                    update_unpacked_kernel(ll, lh, hl, hh, lh, hl, hl, lh - band_stride_y, hh  - band_stride_y);

                    lh--; ll--; hl--; hh--;
                }

                // left extension
                update_unpacked_symmetric_extension_kernel(ll, lh, hl, hh, lh, hl, lh - band_stride_y, hh  - band_stride_y);

                ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y;
            }
        }

        tmp_lh = band_end_y - band_start_y == 1 ? tmp_lh : lh;                    
        for(size_t x = band_end_x - 1; x >= band_start_x + 1; --x){ // top row
            update_unpacked_kernel(ll, lh, hl, hh, tmp_lh, hl, hl, tmp_lh1, hh1);

            lh--; ll--; hl--; hh--; hh1--; tmp_lh1--; tmp_lh--;
        }

        // left extension
        update_unpacked_symmetric_extension_kernel(ll, lh, hl, hh, tmp_lh, hl, tmp_lh1, hh1);
    } else if(band_start_x == 0){
        float * tmp_hl = tsa->tmp.HL + tmp_offset + (band_end_y - band_start_y) - 1;
        
        if(band_end_y - band_start_y > 1){
            update_unpacked_kernel(ll, lh, hl, hh, tmp_lh, tmp_hl, hl, lh - band_stride_y, hh  - band_stride_y);

            lh--; ll--; hl--; hh--; tmp_hl--; tmp_lh--;

            for(size_t x = band_end_x - 2; x >= band_start_x + 1; --x){ // main area
                update_unpacked_kernel(ll, lh, hl, hh, tmp_lh, hl, hl, lh - band_stride_y, hh  - band_stride_y);

                lh--; ll--; hl--; hh--; tmp_lh--;
            }

            // left extension
            update_unpacked_symmetric_extension_kernel(ll, lh, hl, hh, tmp_lh, hl, lh - band_stride_y, hh  - band_stride_y);

            ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y;
        }
        
        if(band_end_y - band_start_y > 2){
            for(size_t y = band_end_y-2; y > band_start_y; --y){
                update_unpacked_kernel(ll, lh, hl, hh, lh, tmp_hl, hl, lh - band_stride_y, hh  - band_stride_y);

                lh--; ll--; hl--; hh--; tmp_hl--;

                for(size_t x = band_end_x - 2; x >= band_start_x + 1; --x){ // main area
                    update_unpacked_kernel(ll, lh, hl, hh, lh, hl, hl, lh - band_stride_y, hh  - band_stride_y);

                    lh--; ll--; hl--; hh--;
                }

                // left extension
                update_unpacked_symmetric_extension_kernel(ll, lh, hl, hh, lh, hl, lh - band_stride_y, hh  - band_stride_y);

                ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y;
            }
        }

        tmp_lh = band_end_y - band_start_y == 1 ? tmp_lh : lh;
        update_unpacked_kernel(ll, lh, hl, hh, tmp_lh, tmp_hl, hl, tmp_lh1, hh1);

        lh--; ll--; hl--; hh--; hh1--; tmp_lh1--; tmp_hl--; tmp_lh--;
        
        for(size_t x = band_end_x - 2; x >= band_start_x + 1; --x){ // top row
            update_unpacked_kernel(ll, lh, hl, hh, tmp_lh, hl, hl, tmp_lh1, hh1);

            lh--; ll--; hl--; hh--; hh1--; tmp_lh1--; tmp_lh--;
        }

        // left extension
        update_unpacked_symmetric_extension_kernel(ll, lh, hl, hh,  tmp_lh, hl, tmp_lh1, hh1);
    } else if(band_end_x == band_size_x){
        float * tmp_hl = tsa->tmp.HL + tmp_offset_left + (tsa->threading_info->band_end_y[tid-1] - tsa->threading_info->band_start_y[tid-1]);
                
        if(band_end_y - band_start_y > 1){
            for(size_t x = band_end_x - 1; x >= band_start_x + 1; --x){ // main area
                update_unpacked_kernel(ll, lh, hl, hh, tmp_lh, hl, hl, lh - band_stride_y, hh  - band_stride_y);

                lh--; ll--; hl--; hh--; tmp_lh--;
            }

            // left extension
            update_unpacked_kernel(ll, lh, hl, hh, tmp_lh, hl, tmp_hl, lh - band_stride_y, hh  - band_stride_y);

            ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y; tmp_hl--;
        }
        
        if(band_end_y - band_start_y > 2){
            for(size_t y = band_end_y-2; y > band_start_y; --y){
                for(size_t x = band_end_x - 1; x >= band_start_x + 1; --x){ // main area
                    update_unpacked_kernel(ll, lh, hl, hh, lh, hl, hl, lh - band_stride_y, hh  - band_stride_y);

                    lh--; ll--; hl--; hh--;
                }

                // left extension
                update_unpacked_kernel(ll, lh, hl, hh, lh, hl, tmp_hl, lh - band_stride_y, hh  - band_stride_y);

                ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y; tmp_hl--;
            }
        }

        tmp_lh = band_end_y - band_start_y == 1 ? tmp_lh : lh;
        for(size_t x = band_end_x - 1; x >= band_start_x + 1; --x){ // top row
            update_unpacked_kernel(ll, lh, hl, hh, tmp_lh, hl, hl, tmp_lh1, hh1);

            lh--; ll--; hl--; hh--; hh1--; tmp_lh1--; tmp_lh--;
        }
        
        // left extension
        update_unpacked_kernel(ll, lh, hl, hh, tmp_lh, hl, tmp_hl, tmp_lh1, hh1);
    } else {
        float * tmp_hl = tsa->tmp.HL + tmp_offset + (band_end_y - band_start_y) - 1;
        float * tmp_hl1 = tsa->tmp.HL + tmp_offset_left + (tsa->threading_info->band_end_y[tid-1] - tsa->threading_info->band_start_y[tid-1]);
  
        if(band_end_y - band_start_y > 1){
            update_unpacked_kernel(ll, lh, hl, hh, tmp_lh, tmp_hl, hl, lh - band_stride_y, hh - band_stride_y);

            lh--; ll--; hl--; hh--; tmp_hl--; tmp_lh--;

            for(size_t x = band_end_x - 2; x >= band_start_x + 1; --x){ // main area
                update_unpacked_kernel(ll, lh, hl, hh, tmp_lh, hl, hl, lh - band_stride_y, hh - band_stride_y);

                lh--; ll--; hl--; hh--; tmp_lh--;
            }

            // left extension
            update_unpacked_kernel(ll, lh, hl, hh, tmp_lh, hl, tmp_hl1, lh - band_stride_y, hh - band_stride_y);

            ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y; tmp_hl1--;
        }
        
        if(band_end_y - band_start_y > 2){
            for(size_t y = band_end_y-2; y > band_start_y; --y){
                update_unpacked_kernel(ll, lh, hl, hh, lh, tmp_hl, hl, lh - band_stride_y, hh - band_stride_y);

                lh--; ll--; hl--; hh--; tmp_hl--;

                for(size_t x = band_end_x - 2; x >= band_start_x + 1; --x){ // main area
                    update_unpacked_kernel(ll, lh, hl, hh, lh, hl, hl, lh - band_stride_y, hh - band_stride_y);

                    lh--; ll--; hl--; hh--;
                }

                // left extension
                update_unpacked_kernel(ll, lh, hl, hh, lh, hl, tmp_hl1, lh - band_stride_y, hh - band_stride_y);

                ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y; tmp_hl1--;
            }
        }

        tmp_lh = band_end_y - band_start_y == 1 ? tmp_lh : lh;
        update_unpacked_kernel(ll, lh, hl, hh, tmp_lh, tmp_hl, hl, tmp_lh1, hh1);

        lh--; ll--; hl--; hh--; hh1--; tmp_lh1--; tmp_lh--;
        
        for(size_t x = band_end_x - 2; x >= band_start_x + 1; --x){ // top row
            update_unpacked_kernel(ll, lh, hl, hh, tmp_lh, hl, hl, tmp_lh1, hh1);

            lh--; ll--; hl--; hh--; hh1--; tmp_lh1--; tmp_lh--;
        }
            
        update_unpacked_kernel(ll, lh, hl, hh, tmp_lh, hl, tmp_hl1, tmp_lh1, hh1);
    }
}

TRANSFORM_STEP_FUNC NO_TREE_VECTORIZE cdf53_non_separable_lifting_generic(size_t step)
{
    switch(step){
        case 0:
            return predict_with_unpack;
        case 1:
            return update_unpacked;
        default:
            return NULL;
    }
}

void NO_TREE_VECTORIZE cdf53_non_separable_lifting_generic_transform_tile(const TransformStepArguments * tsa)
{
    predict_with_unpack(tsa);
    update_unpacked(tsa);
}
