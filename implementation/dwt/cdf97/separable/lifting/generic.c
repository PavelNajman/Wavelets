#ifdef _OPENMP
    #include <omp.h>
#endif

#include "common.h"

static inline void FORCE_INLINE NO_TREE_VECTORIZE unpack_kernel(const float * m0, const float * m1, const float * m2, const float * m3, float * ll, float * hl, float * lh, float * hh)
{
    *hl = *(m0+1) + -1.58613434342059f * (*(m0) + *(m2));
    *hh = *(m1+1) + -1.58613434342059f * (*(m1) + *(m3));

    *ll = *(m0); *lh = *(m1);
}

static inline void FORCE_INLINE NO_TREE_VECTORIZE unpacked_kernel(const float * A, const float * B, const float * C, const float * D, const float ALPHA, float * E, float * F)
{
    *(E) += ALPHA * (*(A) + *(C));
    *(F) += ALPHA * (*(B) + *(D));
}

static void NO_TREE_VECTORIZE H_predict_with_unpack(const TransformStepArguments * tsa)
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

    const size_t next_band_y = band_stride_y - (band_end_x - band_start_x) + 1;
    const size_t next_tile_y = 2 * stride_y - (img_end_x - img_start_x) + 2;
    
    float * m0 = mem + img_start_x + img_start_y * stride_y;
    float * m1 = mem + img_start_x + (img_start_y+1) * stride_y;

    float * ll = tsa->tile_bands.LL + band_start_x + band_start_y * band_stride_y;
    float * hl = tsa->tile_bands.HL + band_start_x + band_start_y * band_stride_y;
    float * lh = tsa->tile_bands.LH + band_start_x + band_start_y * band_stride_y;
    float * hh = tsa->tile_bands.HH + band_start_x + band_start_y * band_stride_y;
    
    if(img_end_x == size_x){
        for(size_t y = img_start_y; y < img_end_y; y += 2){
            for(size_t x = img_start_x; x < img_end_x-2; x += 2){
                unpack_kernel(m0, m1, m0+2, m1+2, ll, hl, lh, hh);

                ++ll; ++hl; ++hh; ++lh;
                m0 += 2; m1 += 2;
            }

            unpack_kernel(m0, m1, m0, m1, ll, hl, lh, hh);

            ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y;
        }
    } else { 
        for(size_t y = img_start_y; y < img_end_y; y += 2){
            for(size_t x = img_start_x; x < img_end_x-2; x += 2){
                unpack_kernel(m0, m1, m0+2, m1+2, ll, hl, lh, hh);

                ++ll; ++hl; ++hh; ++lh;
                m0 += 2; m1 += 2;
            }

            unpack_kernel(m0, m1, m0 + 2, m1 + 2, ll, hl, lh, hh);

            ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y;
        }
    }
    #pragma omp barrier
}

static void NO_TREE_VECTORIZE H_update_unpacked(const TransformStepArguments * tsa)
{
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

    if(band_start_x == 0){
        for(size_t y = band_start_y; y < band_end_y; ++y){
            // first 4 columns
            unpacked_kernel(hl, hh, hl, hh, -0.05298011857290f, ll, lh);

            ++ll; ++hl; ++lh; ++hh;
            for(size_t x = band_start_x + 1; x < band_end_x; x += 1){ // main area

                unpacked_kernel(hl, hh, hl-1, hh-1, -0.05298011857290f, ll, lh);

                ++ll; ++hl; ++lh; ++hh;
            }

            ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        }
    } else {
        for(size_t y = band_start_y; y < band_end_y; ++y){
            for(size_t x = band_start_x; x < band_end_x; x += 1){ // main area

                unpacked_kernel(hl, hh, hl-1, hh-1, -0.05298011857290f, ll, lh);

                ++ll; ++hl; ++lh; ++hh;
            }

            ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        }
    }
    #pragma omp barrier
}

static void NO_TREE_VECTORIZE V_predict_unpacked(const TransformStepArguments * tsa)
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
            for(size_t x = band_start_x; x < band_end_x; ++x){
                
                unpacked_kernel(ll, hl, ll1, hl1, -1.58613434342059f, lh, hh);

                ++ll; ++hl; ++lh; ++hh; ++ll1; ++hl1;
            }
            
            ll += next_y; hl += next_y; lh += next_y; hh += next_y; ll1+=next_y; hl1+=next_y;
        }
    } else {
        for(size_t y = band_start_y; y < band_end_y-1; ++y){
            for(size_t x = band_start_x; x < band_end_x; ++x){
                unpacked_kernel(ll, hl, ll1, hl1, -1.58613434342059f, lh, hh);

                ++ll; ++hl; ++lh; ++hh; ++ll1; ++hl1;
            }

            ll += next_y; hl += next_y; lh += next_y; hh += next_y; ll1+=next_y; hl1+=next_y;
        }
        // last row
        for(size_t x = band_start_x; x < band_end_x; ++x){
            unpacked_kernel(ll, hl, ll, hl, -1.58613434342059f, lh, hh);

            ++ll; ++hl; ++lh; ++hh;
        }
    }

    #pragma omp barrier
}

static void NO_TREE_VECTORIZE V_update_unpacked(const TransformStepArguments * tsa)
{
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

    if(band_start_y != 0){
        float * lh1 = tsa->tile_bands.LH + band_start_x + (band_start_y-1) * band_stride_y;
        float * hh1 = tsa->tile_bands.HH + band_start_x + (band_start_y-1) * band_stride_y;

        for(size_t y = band_start_y; y < band_end_y; ++y){
            for(size_t x = band_start_x; x < band_end_x; ++x){
                
                unpacked_kernel(lh, hh, lh1, hh1, -0.05298011857290f, ll, hl);

                ++ll; ++hl; ++lh; ++hh; ++lh1; ++hh1;
            }
            
            ll += next_y; hl += next_y; lh += next_y; hh += next_y; lh1+=next_y; hh1+=next_y;
        }
    }else{
        // first row
        for(size_t x = band_start_x; x < band_end_x; ++x){
            unpacked_kernel(lh, hh, lh, hh, -0.05298011857290f , ll, hl);
            ++ll; ++hl; ++lh; ++hh;
        }

        ll += next_y; hl += next_y; lh += next_y; hh += next_y;

        float * lh1 = tsa->tile_bands.LH + band_start_x + (band_start_y) * band_stride_y;
        float * hh1 = tsa->tile_bands.HH + band_start_x + (band_start_y) * band_stride_y;

        for(size_t y = band_start_y+1; y < band_end_y; ++y){
            for(size_t x = band_start_x; x < band_end_x; ++x){
                unpacked_kernel(lh, hh, lh1, hh1, -0.05298011857290f , ll, hl);
                ++ll; ++hl; ++lh; ++hh; ++lh1; ++hh1;
            }
            ll += next_y; hl += next_y; lh += next_y; hh += next_y; lh1+=next_y; hh1+=next_y;
        }
    }
    #pragma omp barrier
}

static void NO_TREE_VECTORIZE H_predict_unpacked(const TransformStepArguments * tsa)
{
    const size_t band_size_x = tsa->tile_bands.size_x;
    const size_t band_stride_y = tsa->tile_bands.stride_y;

    const size_t tid = (size_t)omp_get_thread_num();

    const size_t band_start_y = tsa->threading_info->band_start_y[tid];
    const size_t band_end_y = tsa->threading_info->band_end_y[tid];
    const size_t band_start_x = tsa->threading_info->band_start_x[tid];
    const size_t band_end_x = tsa->threading_info->band_end_x[tid];

    const size_t next_y = band_stride_y - (band_end_x - band_start_x) + 1;

    float * ll = tsa->tile_bands.LL + band_start_x + band_start_y * band_stride_y;
    float * hl = tsa->tile_bands.HL + band_start_x + band_start_y * band_stride_y;
    float * lh = tsa->tile_bands.LH + band_start_x + band_start_y * band_stride_y;
    float * hh = tsa->tile_bands.HH + band_start_x + band_start_y * band_stride_y;

    if(band_end_x == band_size_x) {
        for(size_t y = band_start_y; y < band_end_y; ++y){
            for(size_t x = band_start_x; x < band_end_x - 1; x += 1){ // main area    
                unpacked_kernel(ll, lh, ll+1, lh+1, +0.88291107553090f, hl, hh);
                ++ll; ++hl; ++lh; ++hh;
            }
            unpacked_kernel(ll, lh, ll, lh, +0.88291107553090f, hl, hh);
            ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        }
    } else {
        for(size_t y = band_start_y; y < band_end_y; ++y){
            for(size_t x = band_start_x; x < band_end_x - 1; x += 1){ // main area    
                unpacked_kernel(ll, lh, ll+1, lh+1, +0.88291107553090f, hl, hh);
                ++ll; ++hl; ++lh; ++hh;
            }
            unpacked_kernel(ll, lh, ll+1, lh+1, +0.88291107553090f, hl, hh);
            ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        }
    }
    
    #pragma omp barrier
}

static void NO_TREE_VECTORIZE H_update_unpacked_2(const TransformStepArguments * tsa)
{
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

    if(band_start_x == 0){
        for(size_t y = band_start_y; y < band_end_y; ++y){
            // first 4 columns
            unpacked_kernel(hl, hh, hl, hh, +0.44350685204390f, ll, lh);
            ++ll; ++hl; ++lh; ++hh;
            for(size_t x = band_start_x + 1; x < band_end_x; x += 1){ // main area
                unpacked_kernel(hl, hh, hl-1, hh-1, +0.44350685204390f, ll, lh);
                ++ll; ++hl; ++lh; ++hh;
            }
            ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        }
    } else {
        for(size_t y = band_start_y; y < band_end_y; ++y){
            for(size_t x = band_start_x; x < band_end_x; x += 1){ // main area
                unpacked_kernel(hl, hh, hl-1, hh-1, +0.44350685204390f, ll, lh);
                ++ll; ++hl; ++lh; ++hh;
            }
            ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        }
    }
    #pragma omp barrier
}

static void NO_TREE_VECTORIZE V_predict_unpacked_2(const TransformStepArguments * tsa)
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
            for(size_t x = band_start_x; x < band_end_x; ++x){
                unpacked_kernel(ll, hl, ll1, hl1, +0.88291107553090f, lh, hh);
                ++ll; ++hl; ++lh; ++hh; ++ll1; ++hl1;
            }
            
            ll += next_y; hl += next_y; lh += next_y; hh += next_y; ll1+=next_y; hl1+=next_y;
        }
    } else {
        for(size_t y = band_start_y; y < band_end_y-1; ++y){
            for(size_t x = band_start_x; x < band_end_x; ++x){
                unpacked_kernel(ll, hl, ll1, hl1, +0.88291107553090f, lh, hh);
                ++ll; ++hl; ++lh; ++hh; ++ll1; ++hl1;
            }

            ll += next_y; hl += next_y; lh += next_y; hh += next_y; ll1+=next_y; hl1+=next_y;
        }
        // last row
        for(size_t x = band_start_x; x < band_end_x; ++x){
            unpacked_kernel(ll, hl, ll, hl, +0.88291107553090f, lh, hh);
            ++ll; ++hl; ++lh; ++hh;
        }
    }

    #pragma omp barrier
}

static void NO_TREE_VECTORIZE V_update_unpacked_2(const TransformStepArguments * tsa)
{
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

    if(band_start_y != 0){
        float * lh1 = tsa->tile_bands.LH + band_start_x + (band_start_y-1) * band_stride_y;
        float * hh1 = tsa->tile_bands.HH + band_start_x + (band_start_y-1) * band_stride_y;

        for(size_t y = band_start_y; y < band_end_y; ++y){
            for(size_t x = band_start_x; x < band_end_x; ++x){
                
                unpacked_kernel(lh, hh, lh1, hh1, +0.44350685204390f, ll, hl);

                ++ll; ++hl; ++lh; ++hh; ++lh1; ++hh1;
            }
            
            ll += next_y; hl += next_y; lh += next_y; hh += next_y; lh1+=next_y; hh1+=next_y;
        }
    }else{
        // first row
        for(size_t x = band_start_x; x < band_end_x; ++x){
            unpacked_kernel(lh, hh, lh, hh, +0.44350685204390f, ll, hl);
            ++ll; ++hl; ++lh; ++hh;
        }

        ll += next_y; hl += next_y; lh += next_y; hh += next_y;

        float * lh1 = tsa->tile_bands.LH + band_start_x + (band_start_y) * band_stride_y;
        float * hh1 = tsa->tile_bands.HH + band_start_x + (band_start_y) * band_stride_y;

        for(size_t y = band_start_y+1; y < band_end_y; ++y){
            for(size_t x = band_start_x; x < band_end_x; ++x){
                unpacked_kernel(lh, hh, lh1, hh1, +0.44350685204390f, ll, hl);
                ++ll; ++hl; ++lh; ++hh; ++lh1; ++hh1;
            }
            ll += next_y; hl += next_y; lh += next_y; hh += next_y; lh1+=next_y; hh1+=next_y;
        }
    }
}

TRANSFORM_STEP_FUNC NO_TREE_VECTORIZE cdf97_separable_lifting_generic(size_t step)
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
        case 4: 
            return H_predict_unpacked;
        case 5:
            return V_predict_unpacked_2;
        case 6:
            return H_update_unpacked_2;
        case 7:
            return V_update_unpacked_2;
        default:
            return NULL;
    }
}

void NO_TREE_VECTORIZE cdf97_separable_lifting_generic_transform_tile(const TransformStepArguments * tsa)
{
    H_predict_with_unpack(tsa);
    V_predict_unpacked(tsa);
    H_update_unpacked(tsa);
    V_update_unpacked(tsa);
    H_predict_unpacked(tsa);
    V_predict_unpacked_2(tsa);
    H_update_unpacked_2(tsa);
    V_update_unpacked_2(tsa);
}
