#ifdef _OPENMP
    #include <omp.h>
#endif

#include "common.h"

static inline float NO_TREE_VECTORIZE kernel(float A, float B, float C, float D, float ALPHA, float ALPHA2)
{    
    return  ALPHA * (A + B) + ALPHA2 * (C + D);
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
    
    float LL01_L, LL00, HL00, LL01, LL02;
    float LH01_L, LH00, HH00, LH01, LH02;
    
    if(img_start_x == 0 && img_end_x == size_x){
        for(size_t y = img_start_y; y < img_end_y; y += 2){        
            LL01_L = *(m0+2); LL00 = *m0; HL00 = *(m0+1); LL01 = *(m0+2); LL02 = *(m0+4);   
            LH01_L = *(m1+2); LH00 = *m1; HH00 = *(m1+1); LH01 = *(m1+2); LH02 = *(m1+4);

            HL00 += kernel(LL00, LL01, LL02, LL01_L, (-9.f/16), (+1.f/16));
            HH00 += kernel(LH00, LH01, LH02, LH01_L, (-9.f/16), (+1.f/16));
            *ll = LL00; *hl = HL00; 
            *lh = LH00; *hh = HH00;

            ++ll; ++hl; ++hh; ++lh;
            m0 += 2; m1 += 2;

            for(size_t x = img_start_x + 2; x < img_end_x-4; x += 2){
                LL01_L = *(m0-2); LL00 = *m0; HL00 = *(m0+1); LL01 = *(m0+2); LL02 = *(m0+4);   
                LH01_L = *(m1-2); LH00 = *m1; HH00 = *(m1+1); LH01 = *(m1+2); LH02 = *(m1+4);

                HL00 += kernel(LL00, LL01, LL02, LL01_L, (-9.f/16), (+1.f/16));
                HH00 += kernel(LH00, LH01, LH02, LH01_L, (-9.f/16), (+1.f/16));
                *ll = LL00; *hl = HL00; 
                *lh = LH00; *hh = HH00;

                ++ll; ++hl; ++hh; ++lh;
                m0 += 2; m1 += 2;
            }

            LL01_L = *(m0-2); LL00 = *m0; HL00 = *(m0+1); LL01 = *(m0+2); LL02 = *(m0+2);   
            LH01_L = *(m1-2); LH00 = *m1; HH00 = *(m1+1); LH01 = *(m1+2); LH02 = *(m1+2);

            HL00 += kernel(LL00, LL01, LL02, LL01_L, (-9.f/16), (+1.f/16));
            HH00 += kernel(LH00, LH01, LH02, LH01_L, (-9.f/16), (+1.f/16));
            *ll = LL00; *hl = HL00; 
            *lh = LH00; *hh = HH00;

            ++ll; ++hl; ++hh; ++lh;
            m0 += 2; m1 += 2;

            LL01_L = *(m0-2); LL00 = *m0; HL00 = *(m0+1); LL01 = *(m0); LL02 = *(m0-2);   
            LH01_L = *(m1-2); LH00 = *m1; HH00 = *(m1+1); LH01 = *(m1); LH02 = *(m1-2);

            HL00 += kernel(LL00, LL01, LL02, LL01_L, (-9.f/16), (+1.f/16));
            HH00 += kernel(LH00, LH01, LH02, LH01_L, (-9.f/16), (+1.f/16));
            *ll = LL00; *hl = HL00; 
            *lh = LH00; *hh = HH00;

            ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y;
        }
    } else if(img_start_x == 0){
        for(size_t y = img_start_y; y < img_end_y; y += 2){        
            LL01_L = *(m0+2); LL00 = *m0; HL00 = *(m0+1); LL01 = *(m0+2); LL02 = *(m0+4);   
            LH01_L = *(m1+2); LH00 = *m1; HH00 = *(m1+1); LH01 = *(m1+2); LH02 = *(m1+4);

            HL00 += kernel(LL00, LL01, LL02, LL01_L, (-9.f/16), (+1.f/16));
            HH00 += kernel(LH00, LH01, LH02, LH01_L, (-9.f/16), (+1.f/16));
            *ll = LL00; *hl = HL00; 
            *lh = LH00; *hh = HH00;

            ++ll; ++hl; ++hh; ++lh;
            m0 += 2; m1 += 2;

            for(size_t x = img_start_x + 2; x < img_end_x-4; x += 2){
                LL01_L = *(m0-2); LL00 = *m0; HL00 = *(m0+1); LL01 = *(m0+2); LL02 = *(m0+4);   
                LH01_L = *(m1-2); LH00 = *m1; HH00 = *(m1+1); LH01 = *(m1+2); LH02 = *(m1+4);

                HL00 += kernel(LL00, LL01, LL02, LL01_L, (-9.f/16), (+1.f/16));
                HH00 += kernel(LH00, LH01, LH02, LH01_L, (-9.f/16), (+1.f/16));
                *ll = LL00; *hl = HL00; 
                *lh = LH00; *hh = HH00;

                ++ll; ++hl; ++hh; ++lh;
                m0 += 2; m1 += 2;
            }

            LL01_L = *(m0-2); LL00 = *m0; HL00 = *(m0+1); LL01 = *(m0+2); LL02 = *(m0+4);   
            LH01_L = *(m1-2); LH00 = *m1; HH00 = *(m1+1); LH01 = *(m1+2); LH02 = *(m1+4);

            HL00 += kernel(LL00, LL01, LL02, LL01_L, (-9.f/16), (+1.f/16));
            HH00 += kernel(LH00, LH01, LH02, LH01_L, (-9.f/16), (+1.f/16));
            *ll = LL00; *hl = HL00; 
            *lh = LH00; *hh = HH00;

            ++ll; ++hl; ++hh; ++lh;
            m0 += 2; m1 += 2;

            LL01_L = *(m0-2); LL00 = *m0; HL00 = *(m0+1); LL01 = *(m0+2); LL02 = *(m0+4);   
            LH01_L = *(m1-2); LH00 = *m1; HH00 = *(m1+1); LH01 = *(m1+2); LH02 = *(m1+4);

            HL00 += kernel(LL00, LL01, LL02, LL01_L, (-9.f/16), (+1.f/16));
            HH00 += kernel(LH00, LH01, LH02, LH01_L, (-9.f/16), (+1.f/16));
            *ll = LL00; *hl = HL00;
            *lh = LH00; *hh = HH00;

            ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y;
        }
    } else if(img_end_x == size_x){
        for(size_t y = img_start_y; y < img_end_y; y += 2){                  
            LL01_L = *(m0-2); LL00 = *m0; HL00 = *(m0+1); LL01 = *(m0+2); LL02 = *(m0+4);   
            LH01_L = *(m1-2); LH00 = *m1; HH00 = *(m1+1); LH01 = *(m1+2); LH02 = *(m1+4);

            HL00 += kernel(LL00, LL01, LL02, LL01_L, (-9.f/16), (+1.f/16));
            HH00 += kernel(LH00, LH01, LH02, LH01_L, (-9.f/16), (+1.f/16));
            *ll = LL00; *hl = HL00; 
            *lh = LH00; *hh = HH00;

            ++ll; ++hl; ++hh; ++lh;
            m0 += 2; m1 += 2;

            for(size_t x = img_start_x + 2; x < img_end_x-4; x += 2){
                LL01_L = *(m0-2); LL00 = *m0; HL00 = *(m0+1); LL01 = *(m0+2); LL02 = *(m0+4);   
                LH01_L = *(m1-2); LH00 = *m1; HH00 = *(m1+1); LH01 = *(m1+2); LH02 = *(m1+4);

                HL00 += kernel(LL00, LL01, LL02, LL01_L, (-9.f/16), (+1.f/16));
                HH00 += kernel(LH00, LH01, LH02, LH01_L, (-9.f/16), (+1.f/16));
                *ll = LL00; *hl = HL00; 
                *lh = LH00; *hh = HH00;

                ++ll; ++hl; ++hh; ++lh;
                m0 += 2; m1 += 2;
            }

            LL01_L = *(m0-2); LL00 = *m0; HL00 = *(m0+1); LL01 = *(m0+2); LL02 = *(m0+2);   
            LH01_L = *(m1-2); LH00 = *m1; HH00 = *(m1+1); LH01 = *(m1+2); LH02 = *(m1+2);

            HL00 += kernel(LL00, LL01, LL02, LL01_L, (-9.f/16), (+1.f/16));
            HH00 += kernel(LH00, LH01, LH02, LH01_L, (-9.f/16), (+1.f/16));
            *ll = LL00; *hl = HL00; 
            *lh = LH00; *hh = HH00;

            ++ll; ++hl; ++hh; ++lh;
            m0 += 2; m1 += 2;

            LL01_L = *(m0-2); LL00 = *m0; HL00 = *(m0+1); LL01 = *(m0); LL02 = *(m0-2);   
            LH01_L = *(m1-2); LH00 = *m1; HH00 = *(m1+1); LH01 = *(m1); LH02 = *(m1-2);

            HL00 += kernel(LL00, LL01, LL02, LL01_L, (-9.f/16), (+1.f/16));
            HH00 += kernel(LH00, LH01, LH02, LH01_L, (-9.f/16), (+1.f/16));
            *ll = LL00; *hl = HL00; 
            *lh = LH00; *hh = HH00;

            ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y;
        }
    } else {
        for(size_t y = img_start_y; y < img_end_y; y += 2){                 
            LL01_L = *(m0-2); LL00 = *m0; HL00 = *(m0+1); LL01 = *(m0+2); LL02 = *(m0+4);   
            LH01_L = *(m1-2); LH00 = *m1; HH00 = *(m1+1); LH01 = *(m1+2); LH02 = *(m1+4);

            HL00 += kernel(LL00, LL01, LL02, LL01_L, (-9.f/16), (+1.f/16));
            HH00 += kernel(LH00, LH01, LH02, LH01_L, (-9.f/16), (+1.f/16));
            *ll = LL00; *hl = HL00; 
            *lh = LH00; *hh = HH00;

            ++ll; ++hl; ++hh; ++lh;
            m0 += 2; m1 += 2;

            for(size_t x = img_start_x + 2; x < img_end_x-4; x += 2){
                LL01_L = *(m0-2); LL00 = *m0; HL00 = *(m0+1); LL01 = *(m0+2); LL02 = *(m0+4);   
                LH01_L = *(m1-2); LH00 = *m1; HH00 = *(m1+1); LH01 = *(m1+2); LH02 = *(m1+4);

                HL00 += kernel(LL00, LL01, LL02, LL01_L, (-9.f/16), (+1.f/16));
                HH00 += kernel(LH00, LH01, LH02, LH01_L, (-9.f/16), (+1.f/16));
                *ll = LL00; *hl = HL00; 
                *lh = LH00; *hh = HH00;

                ++ll; ++hl; ++hh; ++lh;
                m0 += 2; m1 += 2;
            }

            LL01_L = *(m0-2); LL00 = *m0; HL00 = *(m0+1); LL01 = *(m0+2); LL02 = *(m0+4);   
            LH01_L = *(m1-2); LH00 = *m1; HH00 = *(m1+1); LH01 = *(m1+2); LH02 = *(m1+4);

            HL00 += kernel(LL00, LL01, LL02, LL01_L, (-9.f/16), (+1.f/16));
            HH00 += kernel(LH00, LH01, LH02, LH01_L, (-9.f/16), (+1.f/16));
            *ll = LL00; *hl = HL00; 
            *lh = LH00; *hh = HH00;

            ++ll; ++hl; ++hh; ++lh;
            m0 += 2; m1 += 2;

            LL01_L = *(m0-2); LL00 = *m0; HL00 = *(m0+1); LL01 = *(m0+2); LL02 = *(m0+4);   
            LH01_L = *(m1-2); LH00 = *m1; HH00 = *(m1+1); LH01 = *(m1+2); LH02 = *(m1+4);

            HL00 += kernel(LL00, LL01, LL02, LL01_L, (-9.f/16), (+1.f/16));
            HH00 += kernel(LH00, LH01, LH02, LH01_L, (-9.f/16), (+1.f/16));
            *ll = LL00; *hl = HL00; 
            *lh = LH00; *hh = HH00;                        

            ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y;
        }
    }
    #pragma omp barrier
}

static void NO_TREE_VECTORIZE H_update_unpacked(const TransformStepArguments * tsa)
{
    const size_t band_size_x = tsa->tile_bands.size_x;
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

    float HL02_L, HL01_L, LL00, HL00, HL01;     
    float HH02_L, HH01_L, LH00, HH00, HH01;
    
    if(band_start_x == 0 && band_end_x == band_size_x){
        for(size_t y = band_start_y; y < band_end_y; ++y){
            // first column
            HL02_L = *(hl+1); HL01_L = *hl; LL00 = *ll; HL00 = *hl; HL01 = *(hl+1);     
            HH02_L = *(hh+1); HH01_L = *hh; LH00 = *lh; HH00 = *hh; HH01 = *(hh+1);

            LL00 += kernel(HL01_L, HL00, HL01, HL02_L, (+9.f/32), (-1.f/32));
            LH00 += kernel(HH01_L, HH00, HH01, HH02_L, (+9.f/32), (-1.f/32));

            *ll = LL00;
            *lh = LH00;            

            ++ll; ++hl; ++lh; ++hh;

            //second column
            HL02_L = *(hl-1); HL01_L = *(hl-1); LL00 = *ll; HL00 = *hl; HL01 = *(hl+1);     
            HH02_L = *(hh-1); HH01_L = *(hh-1); LH00 = *lh; HH00 = *hh; HH01 = *(hh+1);

            LL00 += kernel(HL01_L, HL00, HL01, HL02_L, (+9.f/32), (-1.f/32));
            LH00 += kernel(HH01_L, HH00, HH01, HH02_L, (+9.f/32), (-1.f/32));

            *ll = LL00;
            *lh = LH00;

            ++ll; ++hl; ++lh; ++hh;

            for(size_t x = band_start_x + 2; x < band_end_x-1; x += 1){ // main area
                HL02_L = *(hl-2); HL01_L = *(hl-1); LL00 = *ll; HL00 = *hl; HL01 = *(hl+1);     
                HH02_L = *(hh-2); HH01_L = *(hh-1); LH00 = *lh; HH00 = *hh; HH01 = *(hh+1);
                LL00 += kernel(HL01_L, HL00, HL01, HL02_L, (+9.f/32), (-1.f/32));
                LH00 += kernel(HH01_L, HH00, HH01, HH02_L, (+9.f/32), (-1.f/32));

                *ll = LL00;
                *lh = LH00;
                ++ll; ++hl; ++lh; ++hh;
            }

            // last column
            HL02_L = *(hl-2); HL01_L = *(hl-1); LL00 = *ll; HL00 = *hl; HL01 = *(hl-1);     
            HH02_L = *(hh-2); HH01_L = *(hh-1); LH00 = *lh; HH00 = *hh; HH01 = *(hh-1);

            LL00 += kernel(HL01_L, HL00, HL01, HL02_L, (+9.f/32), (-1.f/32));
            LH00 += kernel(HH01_L, HH00, HH01, HH02_L, (+9.f/32), (-1.f/32));

            *ll = LL00;
            *lh = LH00;

            ++ll; ++hl; ++lh; ++hh;

            ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        }
    } else if(band_start_x == 0) {
        for(size_t y = band_start_y; y < band_end_y; ++y){
            // first column
            HL02_L = *(hl+1); HL01_L = *hl; LL00 = *ll; HL00 = *hl; HL01 = *(hl+1);     
            HH02_L = *(hh+1); HH01_L = *hh; LH00 = *lh; HH00 = *hh; HH01 = *(hh+1);

            LL00 += kernel(HL01_L, HL00, HL01, HL02_L, (+9.f/32), (-1.f/32));
            LH00 += kernel(HH01_L, HH00, HH01, HH02_L, (+9.f/32), (-1.f/32));

            *ll = LL00;
            *lh = LH00;            

            ++ll; ++hl; ++lh; ++hh;

            //second column
            HL02_L = *(hl-1); HL01_L = *(hl-1); LL00 = *ll; HL00 = *hl; HL01 = *(hl+1);     
            HH02_L = *(hh-1); HH01_L = *(hh-1); LH00 = *lh; HH00 = *hh; HH01 = *(hh+1);

            LL00 += kernel(HL01_L, HL00, HL01, HL02_L, (+9.f/32), (-1.f/32));
            LH00 += kernel(HH01_L, HH00, HH01, HH02_L, (+9.f/32), (-1.f/32));

            *ll = LL00;
            *lh = LH00;

            ++ll; ++hl; ++lh; ++hh;

            for(size_t x = band_start_x + 2; x < band_end_x-1; x += 1){ // main area
                HL02_L = *(hl-2); HL01_L = *(hl-1); LL00 = *ll; HL00 = *hl; HL01 = *(hl+1);     
                HH02_L = *(hh-2); HH01_L = *(hh-1); LH00 = *lh; HH00 = *hh; HH01 = *(hh+1);
                LL00 += kernel(HL01_L, HL00, HL01, HL02_L, (+9.f/32), (-1.f/32));
                LH00 += kernel(HH01_L, HH00, HH01, HH02_L, (+9.f/32), (-1.f/32));

                *ll = LL00;
                *lh = LH00;
                ++ll; ++hl; ++lh; ++hh;
            }

            // last column
            HL02_L = *(hl-2); HL01_L = *(hl-1); LL00 = *ll; HL00 = *hl; HL01 = *(hl+1);     
            HH02_L = *(hh-2); HH01_L = *(hh-1); LH00 = *lh; HH00 = *hh; HH01 = *(hh+1);

            LL00 += kernel(HL01_L, HL00, HL01, HL02_L, (+9.f/32), (-1.f/32));
            LH00 += kernel(HH01_L, HH00, HH01, HH02_L, (+9.f/32), (-1.f/32));

            *ll = LL00;
            *lh = LH00;

            ++ll; ++hl; ++lh; ++hh;

            ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        }
    } else if(band_end_x == band_size_x) {
        for(size_t y = band_start_y; y < band_end_y; ++y){
            // first column
            HL02_L = *(hl-2); HL01_L = *(hl-1); LL00 = *ll; HL00 = *hl; HL01 = *(hl+1);     
            HH02_L = *(hh-2); HH01_L = *(hh-1); LH00 = *lh; HH00 = *hh; HH01 = *(hh+1);

            LL00 += kernel(HL01_L, HL00, HL01, HL02_L, (+9.f/32), (-1.f/32));
            LH00 += kernel(HH01_L, HH00, HH01, HH02_L, (+9.f/32), (-1.f/32));

            *ll = LL00;
            *lh = LH00;            

            ++ll; ++hl; ++lh; ++hh;

            //second column
            HL02_L = *(hl-2); HL01_L = *(hl-1); LL00 = *ll; HL00 = *hl; HL01 = *(hl+1);     
            HH02_L = *(hh-2); HH01_L = *(hh-1); LH00 = *lh; HH00 = *hh; HH01 = *(hh+1);

            LL00 += kernel(HL01_L, HL00, HL01, HL02_L, (+9.f/32), (-1.f/32));
            LH00 += kernel(HH01_L, HH00, HH01, HH02_L, (+9.f/32), (-1.f/32));

            *ll = LL00;
            *lh = LH00;

            ++ll; ++hl; ++lh; ++hh;

            for(size_t x = band_start_x + 2; x < band_end_x-1; x += 1){ // main area
                HL02_L = *(hl-2); HL01_L = *(hl-1); LL00 = *ll; HL00 = *hl; HL01 = *(hl+1);     
                HH02_L = *(hh-2); HH01_L = *(hh-1); LH00 = *lh; HH00 = *hh; HH01 = *(hh+1);
                LL00 += kernel(HL01_L, HL00, HL01, HL02_L, (+9.f/32), (-1.f/32));
                LH00 += kernel(HH01_L, HH00, HH01, HH02_L, (+9.f/32), (-1.f/32));

                *ll = LL00;
                *lh = LH00;
                ++ll; ++hl; ++lh; ++hh;
            }

            // last column
            HL02_L = *(hl-2); HL01_L = *(hl-1); LL00 = *ll; HL00 = *hl; HL01 = *(hl-1);     
            HH02_L = *(hh-2); HH01_L = *(hh-1); LH00 = *lh; HH00 = *hh; HH01 = *(hh-1);

            LL00 += kernel(HL01_L, HL00, HL01, HL02_L, (+9.f/32), (-1.f/32));
            LH00 += kernel(HH01_L, HH00, HH01, HH02_L, (+9.f/32), (-1.f/32));

            *ll = LL00;
            *lh = LH00;

            ++ll; ++hl; ++lh; ++hh;

            ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        }
    } else {
        for(size_t y = band_start_y; y < band_end_y; ++y){
            // first column
            HL02_L = *(hl-2); HL01_L = *(hl-1); LL00 = *ll; HL00 = *hl; HL01 = *(hl+1);     
            HH02_L = *(hh-2); HH01_L = *(hh-1); LH00 = *lh; HH00 = *hh; HH01 = *(hh+1);

            LL00 += kernel(HL01_L, HL00, HL01, HL02_L, (+9.f/32), (-1.f/32));
            LH00 += kernel(HH01_L, HH00, HH01, HH02_L, (+9.f/32), (-1.f/32));

            *ll = LL00;
            *lh = LH00;            

            ++ll; ++hl; ++lh; ++hh;

            //second column
            HL02_L = *(hl-2); HL01_L = *(hl-1); LL00 = *ll; HL00 = *hl; HL01 = *(hl+1);     
            HH02_L = *(hh-2); HH01_L = *(hh-1); LH00 = *lh; HH00 = *hh; HH01 = *(hh+1);

            LL00 += kernel(HL01_L, HL00, HL01, HL02_L, (+9.f/32), (-1.f/32));
            LH00 += kernel(HH01_L, HH00, HH01, HH02_L, (+9.f/32), (-1.f/32));

            *ll = LL00;
            *lh = LH00;

            ++ll; ++hl; ++lh; ++hh;

            for(size_t x = band_start_x + 2; x < band_end_x-1; x += 1){ // main area
                HL02_L = *(hl-2); HL01_L = *(hl-1); LL00 = *ll; HL00 = *hl; HL01 = *(hl+1);     
                HH02_L = *(hh-2); HH01_L = *(hh-1); LH00 = *lh; HH00 = *hh; HH01 = *(hh+1);
                LL00 += kernel(HL01_L, HL00, HL01, HL02_L, (+9.f/32), (-1.f/32));
                LH00 += kernel(HH01_L, HH00, HH01, HH02_L, (+9.f/32), (-1.f/32));

                *ll = LL00;
                *lh = LH00;
                ++ll; ++hl; ++lh; ++hh;
            }

            // last column
            HL02_L = *(hl-2); HL01_L = *(hl-1); LL00 = *ll; HL00 = *hl; HL01 = *(hl+1);     
            HH02_L = *(hh-2); HH01_L = *(hh-1); LH00 = *lh; HH00 = *hh; HH01 = *(hh+1);

            LL00 += kernel(HL01_L, HL00, HL01, HL02_L, (+9.f/32), (-1.f/32));
            LH00 += kernel(HH01_L, HH00, HH01, HH02_L, (+9.f/32), (-1.f/32));

            *ll = LL00;
            *lh = LH00;

            ++ll; ++hl; ++lh; ++hh;

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
    
    float LL10_A, LL00, LH00, LL10, LL20;
    float HL10_A, HL00, HH00, HL10, HL20;
    
    if(band_start_y == 0 && band_end_y == band_size_y){        
        for(size_t x = band_start_x; x < band_end_x; ++x){
            LL10_A = *(ll+band_stride_y); LL00 = *ll; LH00 = *lh; LL10 = *(ll+band_stride_y); LL20 = *(ll+2*band_stride_y);
            HL10_A = *(hl+band_stride_y); HL00 = *hl; HH00 = *hh; HL10 = *(hl+band_stride_y); HL20 = *(hl+2*band_stride_y);
            
            LH00 += kernel(LL00, LL10, LL10_A, LL20, (-9.f/16), (+1.f/16));
            HH00 += kernel(HL00, HL10, HL10_A, HL20, (-9.f/16), (+1.f/16));
            
            *lh = LH00;
            *hh = HH00;           
            
            ++ll; ++hl; ++lh; ++hh;
        }
        ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        
        for(size_t y = band_start_y+1; y < band_end_y - 2; ++y){
            for(size_t x = band_start_x; x < band_end_x; ++x){
                LL10_A = *(ll-band_stride_y); LL00 = *ll; LH00 = *lh; LL10 = *(ll+band_stride_y); LL20 = *(ll+2*band_stride_y);
                HL10_A = *(hl-band_stride_y); HL00 = *hl; HH00 = *hh; HL10 = *(hl+band_stride_y); HL20 = *(hl+2*band_stride_y);

                LH00 += kernel(LL00, LL10, LL10_A, LL20, (-9.f/16), (+1.f/16));
                HH00 += kernel(HL00, HL10, HL10_A, HL20, (-9.f/16), (+1.f/16));

                *lh = LH00;
                *hh = HH00;
                
                ++ll; ++hl; ++lh; ++hh;
            }
            ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        }  
        
        for(size_t x = band_start_x; x < band_end_x; ++x){
            LL10_A = *(ll-band_stride_y); LL00 = *ll; LH00 = *lh; LL10 = *(ll+band_stride_y); LL20 = *(ll+band_stride_y);
            HL10_A = *(hl-band_stride_y); HL00 = *hl; HH00 = *hh; HL10 = *(hl+band_stride_y); HL20 = *(hl+band_stride_y);

            LH00 += kernel(LL00, LL10, LL10_A, LL20, (-9.f/16), (+1.f/16));
            HH00 += kernel(HL00, HL10, HL10_A, HL20, (-9.f/16), (+1.f/16));
            
            *lh = LH00;
            *hh = HH00;
            
            ++ll; ++hl; ++lh; ++hh;
        }
        ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        
        for(size_t x = band_start_x; x < band_end_x; ++x){
            LL10_A = *(ll-band_stride_y); LL00 = *ll; LH00 = *lh; LL10 = *(ll); LL20 = *(ll-band_stride_y);
            HL10_A = *(hl-band_stride_y); HL00 = *hl; HH00 = *hh; HL10 = *(hl); HL20 = *(hl-band_stride_y);

            LH00 += kernel(LL00, LL10, LL10_A, LL20, (-9.f/16), (+1.f/16));
            HH00 += kernel(HL00, HL10, HL10_A, HL20, (-9.f/16), (+1.f/16));
            
            *lh = LH00;
            *hh = HH00;

            ++ll; ++hl; ++lh; ++hh;
        } 
    } else if(band_start_y == 0) {         
        for(size_t x = band_start_x; x < band_end_x; ++x){
            LL10_A = *(ll+band_stride_y); LL00 = *ll; LH00 = *lh; LL10 = *(ll+band_stride_y); LL20 = *(ll+2*band_stride_y);
            HL10_A = *(hl+band_stride_y); HL00 = *hl; HH00 = *hh; HL10 = *(hl+band_stride_y); HL20 = *(hl+2*band_stride_y);            
            
            LH00 += kernel(LL00, LL10, LL10_A, LL20, (-9.f/16), (+1.f/16));
            HH00 += kernel(HL00, HL10, HL10_A, HL20, (-9.f/16), (+1.f/16));
            
            *lh = LH00;
            *hh = HH00;

            ++ll; ++hl; ++lh; ++hh;
        }
        ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        
        for(size_t y = band_start_y+1; y < band_end_y; ++y){
            for(size_t x = band_start_x; x < band_end_x; ++x){
                LL10_A = *(ll-band_stride_y); LL00 = *ll; LH00 = *lh; LL10 = *(ll+band_stride_y); LL20 = *(ll+2*band_stride_y);
                HL10_A = *(hl-band_stride_y); HL00 = *hl; HH00 = *hh; HL10 = *(hl+band_stride_y); HL20 = *(hl+2*band_stride_y);

                LH00 += kernel(LL00, LL10, LL10_A, LL20, (-9.f/16), (+1.f/16));
                HH00 += kernel(HL00, HL10, HL10_A, HL20, (-9.f/16), (+1.f/16));

                *lh = LH00;
                *hh = HH00;
                
                ++ll; ++hl; ++lh; ++hh;
            }
            ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        }
    
    } else if(band_end_y == band_size_y){
        if(band_start_y < band_size_y-1){
            for(size_t y = band_start_y; y < band_end_y - 2; ++y){
                for(size_t x = band_start_x; x < band_end_x; ++x){
                    LL10_A = *(ll-band_stride_y); LL00 = *ll; LH00 = *lh; LL10 = *(ll+band_stride_y); LL20 = *(ll+2*band_stride_y);
                    HL10_A = *(hl-band_stride_y); HL00 = *hl; HH00 = *hh; HL10 = *(hl+band_stride_y); HL20 = *(hl+2*band_stride_y);

                    LH00 += kernel(LL00, LL10, LL10_A, LL20, (-9.f/16), (+1.f/16));
                    HH00 += kernel(HL00, HL10, HL10_A, HL20, (-9.f/16), (+1.f/16));

                    *lh = LH00;
                    *hh = HH00;

                    ++ll; ++hl; ++lh; ++hh;
                }
                ll += next_y; hl += next_y; lh += next_y; hh += next_y;
            }
        
            for(size_t x = band_start_x; x < band_end_x; ++x){
                LL10_A = *(ll-band_stride_y); LL00 = *ll; LH00 = *lh; LL10 = *(ll+band_stride_y); LL20 = *(ll+band_stride_y);
                HL10_A = *(hl-band_stride_y); HL00 = *hl; HH00 = *hh; HL10 = *(hl+band_stride_y); HL20 = *(hl+band_stride_y);

                LH00 += kernel(LL00, LL10, LL10_A, LL20, (-9.f/16), (+1.f/16));
                HH00 += kernel(HL00, HL10, HL10_A, HL20, (-9.f/16), (+1.f/16));

                *lh = LH00;
                *hh = HH00;

                ++ll; ++hl; ++lh; ++hh;
            }
            ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        }
                
        for(size_t x = band_start_x; x < band_end_x; ++x){
            LL10_A = *(ll-band_stride_y); LL00 = *ll; LH00 = *lh; LL10 = *(ll); LL20 = *(ll-band_stride_y);
            HL10_A = *(hl-band_stride_y); HL00 = *hl; HH00 = *hh; HL10 = *(hl); HL20 = *(hl-band_stride_y);

            LH00 += kernel(LL00, LL10, LL10_A, LL20, (-9.f/16), (+1.f/16));
            HH00 += kernel(HL00, HL10, HL10_A, HL20, (-9.f/16), (+1.f/16));

            *lh = LH00;
            *hh = HH00;
            
            ++ll; ++hl; ++lh; ++hh;
        } 
    } else if(band_end_y == band_size_y-1){
        for(size_t y = band_start_y; y < band_end_y - 2; ++y){
            for(size_t x = band_start_x; x < band_end_x; ++x){
                LL10_A = *(ll-band_stride_y); LL00 = *ll; LH00 = *lh; LL10 = *(ll+band_stride_y); LL20 = *(ll+2*band_stride_y);
                HL10_A = *(hl-band_stride_y); HL00 = *hl; HH00 = *hh; HL10 = *(hl+band_stride_y); HL20 = *(hl+2*band_stride_y);

                LH00 += kernel(LL00, LL10, LL10_A, LL20, (-9.f/16), (+1.f/16));
                HH00 += kernel(HL00, HL10, HL10_A, HL20, (-9.f/16), (+1.f/16));

                *lh = LH00;
                *hh = HH00;

                ++ll; ++hl; ++lh; ++hh;
            }
            ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        }
        
        for(size_t x = band_start_x; x < band_end_x; ++x){
            LL10_A = *(ll-band_stride_y); LL00 = *ll; LH00 = *lh; LL10 = *(ll+band_stride_y); LL20 = *(ll+band_stride_y);
            HL10_A = *(hl-band_stride_y); HL00 = *hl; HH00 = *hh; HL10 = *(hl+band_stride_y); HL20 = *(hl+band_stride_y);

            LH00 += kernel(LL00, LL10, LL10_A, LL20, (-9.f/16), (+1.f/16));
            HH00 += kernel(HL00, HL10, HL10_A, HL20, (-9.f/16), (+1.f/16));

            *lh = LH00;
            *hh = HH00;

            ++ll; ++hl; ++lh; ++hh;
        }        
    }else {
        for(size_t y = band_start_y; y < band_end_y; ++y){
            for(size_t x = band_start_x; x < band_end_x; ++x){
                LL10_A = *(ll-band_stride_y); LL00 = *ll; LH00 = *lh; LL10 = *(ll+band_stride_y); LL20 = *(ll+2*band_stride_y);
                HL10_A = *(hl-band_stride_y); HL00 = *hl; HH00 = *hh; HL10 = *(hl+band_stride_y); HL20 = *(hl+2*band_stride_y);

                LH00 += kernel(LL00, LL10, LL10_A, LL20, (-9.f/16), (+1.f/16));
                HH00 += kernel(HL00, HL10, HL10_A, HL20, (-9.f/16), (+1.f/16));

                *lh = LH00;
                *hh = HH00;
                
                ++ll; ++hl; ++lh; ++hh;
            }
            ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        }
    }        

    #pragma omp barrier
}

static void NO_TREE_VECTORIZE V_update_unpacked(const TransformStepArguments * tsa)
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

    float LH20_A, LH10_A, LL00, LH00, LH10;
    float HH20_A, HH10_A, HL00, HH00, HH10;
    
    if(band_start_y == 0 && band_end_y == band_size_y){
        //first row
        for(size_t x = band_start_x; x < band_end_x; ++x){
            LH20_A = *(lh+band_stride_y); LH10_A = *(lh); LL00 = *(ll); LH00 = *(lh); LH10 = *(lh+band_stride_y);
            HH20_A = *(hh+band_stride_y); HH10_A = *(hh); HL00 = *(hl); HH00 = *(hh); HH10 = *(hh+band_stride_y);
            LL00 += kernel(LH00, LH10_A, LH20_A, LH10, (+9.f/32), (-1.f/32));
            HL00 += kernel(HH00, HH10_A, HH20_A, HH10, (+9.f/32), (-1.f/32));

            *ll = LL00;
            *hl = HL00;
                        
            ++ll; ++hl; ++lh; ++hh;
        }
        ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        
        //second row
        for(size_t x = band_start_x; x < band_end_x; ++x){
            LH20_A = *(lh-band_stride_y); LH10_A = *(lh-band_stride_y); LL00 = *(ll); LH00 = *(lh); LH10 = *(lh+band_stride_y);
            HH20_A = *(hh-band_stride_y); HH10_A = *(hh-band_stride_y); HL00 = *(hl); HH00 = *(hh); HH10 = *(hh+band_stride_y);
            LL00 += kernel(LH00, LH10_A, LH20_A, LH10, (+9.f/32), (-1.f/32));
            HL00 += kernel(HH00, HH10_A, HH20_A, HH10, (+9.f/32), (-1.f/32));

            *ll = LL00;
            *hl = HL00;

            ++ll; ++hl; ++lh; ++hh;
        }
        ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        
        //main area
        for(size_t y = band_start_y + 2; y < band_end_y - 1; ++y){
            for(size_t x = band_start_x; x < band_end_x; ++x){
                LH20_A = *(lh-2*band_stride_y); LH10_A = *(lh-band_stride_y); LL00 = *(ll); LH00 = *(lh); LH10 = *(lh+band_stride_y);
                HH20_A = *(hh-2*band_stride_y); HH10_A = *(hh-band_stride_y); HL00 = *(hl); HH00 = *(hh); HH10 = *(hh+band_stride_y);
                LL00 += kernel(LH00, LH10_A, LH20_A, LH10, (+9.f/32), (-1.f/32));
                HL00 += kernel(HH00, HH10_A, HH20_A, HH10, (+9.f/32), (-1.f/32));

                *ll = LL00;
                *hl = HL00;
                
                ++ll; ++hl; ++lh; ++hh;
            }
            ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        }
        
        //last row
        for(size_t x = band_start_x; x < band_end_x; ++x){
            LH20_A = *(lh-2*band_stride_y); LH10_A = *(lh-band_stride_y); LL00 = *(ll); LH00 = *(lh); LH10 = *(lh-band_stride_y);
            HH20_A = *(hh-2*band_stride_y); HH10_A = *(hh-band_stride_y); HL00 = *(hl); HH00 = *(hh); HH10 = *(hh-band_stride_y);
            LL00 += kernel(LH00, LH10_A, LH20_A, LH10, (+9.f/32), (-1.f/32));
            HL00 += kernel(HH00, HH10_A, HH20_A, HH10, (+9.f/32), (-1.f/32));

            *ll = LL00;
            *hl = HL00;

            ++ll; ++hl; ++lh; ++hh;
        }
        
    } else if(band_start_y == 0) {
        //first row
        for(size_t x = band_start_x; x < band_end_x; ++x){
            LH20_A = *(lh+band_stride_y); LH10_A = *(lh); LL00 = *(ll); LH00 = *(lh); LH10 = *(lh+band_stride_y);
            HH20_A = *(hh+band_stride_y); HH10_A = *(hh); HL00 = *(hl); HH00 = *(hh); HH10 = *(hh+band_stride_y);
            LL00 += kernel(LH00, LH10_A, LH20_A, LH10, (+9.f/32), (-1.f/32));
            HL00 += kernel(HH00, HH10_A, HH20_A, HH10, (+9.f/32), (-1.f/32));

            *ll = LL00;
            *hl = HL00;
            
            ++ll; ++hl; ++lh; ++hh;
        }
        ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        
        //second row
        if(band_end_y > 1){
            for(size_t x = band_start_x; x < band_end_x; ++x){
                LH20_A = *(lh-band_stride_y); LH10_A = *(lh-band_stride_y); LL00 = *(ll); LH00 = *(lh); LH10 = *(lh+band_stride_y);
                HH20_A = *(hh-band_stride_y); HH10_A = *(hh-band_stride_y); HL00 = *(hl); HH00 = *(hh); HH10 = *(hh+band_stride_y);
                LL00 += kernel(LH00, LH10_A, LH20_A, LH10, (+9.f/32), (-1.f/32));
                HL00 += kernel(HH00, HH10_A, HH20_A, HH10, (+9.f/32), (-1.f/32));

                *ll = LL00;
                *hl = HL00;

                ++ll; ++hl; ++lh; ++hh;
            }
            ll += next_y; hl += next_y; lh += next_y; hh += next_y;
            
            //main area
            for(size_t y = band_start_y + 2; y < band_end_y; ++y){
                for(size_t x = band_start_x; x < band_end_x; ++x){
                    LH20_A = *(lh-2*band_stride_y); LH10_A = *(lh-band_stride_y); LL00 = *(ll); LH00 = *(lh); LH10 = *(lh+band_stride_y);
                    HH20_A = *(hh-2*band_stride_y); HH10_A = *(hh-band_stride_y); HL00 = *(hl); HH00 = *(hh); HH10 = *(hh+band_stride_y);
                    LL00 += kernel(LH00, LH10_A, LH20_A, LH10, (+9.f/32), (-1.f/32));
                    HL00 += kernel(HH00, HH10_A, HH20_A, HH10, (+9.f/32), (-1.f/32));

                    *ll = LL00;
                    *hl = HL00;

                    ++ll; ++hl; ++lh; ++hh;
                }
                ll += next_y; hl += next_y; lh += next_y; hh += next_y;
            }
        }    
    } else if(band_start_y == 1){
        //second row
        for(size_t x = band_start_x; x < band_end_x; ++x){
            LH20_A = *(lh-band_stride_y); LH10_A = *(lh-band_stride_y); LL00 = *(ll); LH00 = *(lh); LH10 = *(lh+band_stride_y);
            HH20_A = *(hh-band_stride_y); HH10_A = *(hh-band_stride_y); HL00 = *(hl); HH00 = *(hh); HH10 = *(hh+band_stride_y);
            LL00 += kernel(LH00, LH10_A, LH20_A, LH10, (+9.f/32), (-1.f/32));
            HL00 += kernel(HH00, HH10_A, HH20_A, HH10, (+9.f/32), (-1.f/32));

            *ll = LL00;
            *hl = HL00;
            
            ++ll; ++hl; ++lh; ++hh;
        }
        ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        //main area
        for(size_t y = band_start_y + 2; y < band_end_y; ++y){
            for(size_t x = band_start_x; x < band_end_x; ++x){
                LH20_A = *(lh-2*band_stride_y); LH10_A = *(lh-band_stride_y); LL00 = *(ll); LH00 = *(lh); LH10 = *(lh+band_stride_y);
                HH20_A = *(hh-2*band_stride_y); HH10_A = *(hh-band_stride_y); HL00 = *(hl); HH00 = *(hh); HH10 = *(hh+band_stride_y);
                LL00 += kernel(LH00, LH10_A, LH20_A, LH10, (+9.f/32), (-1.f/32));
                HL00 += kernel(HH00, HH10_A, HH20_A, HH10, (+9.f/32), (-1.f/32));

                *ll = LL00;
                *hl = HL00;
                
                ++ll; ++hl; ++lh; ++hh;
            }
            ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        }
    }else if(band_end_y == band_size_y){
        //main area
        for(size_t y = band_start_y; y < band_end_y - 1; ++y){
            for(size_t x = band_start_x; x < band_end_x; ++x){
                LH20_A = *(lh-2*band_stride_y); LH10_A = *(lh-band_stride_y); LL00 = *(ll); LH00 = *(lh); LH10 = *(lh+band_stride_y);
                HH20_A = *(hh-2*band_stride_y); HH10_A = *(hh-band_stride_y); HL00 = *(hl); HH00 = *(hh); HH10 = *(hh+band_stride_y);
                LL00 += kernel(LH00, LH10_A, LH20_A, LH10, (+9.f/32), (-1.f/32));
                HL00 += kernel(HH00, HH10_A, HH20_A, HH10, (+9.f/32), (-1.f/32));

                *ll = LL00;
                *hl = HL00;
                
                ++ll; ++hl; ++lh; ++hh;
            }
            ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        }
        
        //last row
        for(size_t x = band_start_x; x < band_end_x; ++x){
            LH20_A = *(lh-2*band_stride_y); LH10_A = *(lh-band_stride_y); LL00 = *(ll); LH00 = *(lh); LH10 = *(lh-band_stride_y);
            HH20_A = *(hh-2*band_stride_y); HH10_A = *(hh-band_stride_y); HL00 = *(hl); HH00 = *(hh); HH10 = *(hh-band_stride_y);
            LL00 += kernel(LH00, LH10_A, LH20_A, LH10, (+9.f/32), (-1.f/32));
            HL00 += kernel(HH00, HH10_A, HH20_A, HH10, (+9.f/32), (-1.f/32));

            *ll = LL00;
            *hl = HL00;
            
            ++ll; ++hl; ++lh; ++hh;
        }
    } else {
        for(size_t y = band_start_y; y < band_end_y; ++y){
            for(size_t x = band_start_x; x < band_end_x; ++x){
                LH20_A = *(lh-2*band_stride_y); LH10_A = *(lh-band_stride_y); LL00 = *(ll); LH00 = *(lh); LH10 = *(lh+band_stride_y);
                HH20_A = *(hh-2*band_stride_y); HH10_A = *(hh-band_stride_y); HL00 = *(hl); HH00 = *(hh); HH10 = *(hh+band_stride_y);
                LL00 += kernel(LH00, LH10_A, LH20_A, LH10, (+9.f/32), (-1.f/32));
                HL00 += kernel(HH00, HH10_A, HH20_A, HH10, (+9.f/32), (-1.f/32));

                *ll = LL00;
                *hl = HL00;
                
                ++ll; ++hl; ++lh; ++hh;
            }
            ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        }
    }
}

TRANSFORM_STEP_FUNC NO_TREE_VECTORIZE dd137_separable_lifting_generic(size_t step)
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

void NO_TREE_VECTORIZE dd137_separable_lifting_generic_transform_tile(const TransformStepArguments * tsa)
{
    H_predict_with_unpack(tsa);
    V_predict_unpacked(tsa);
    H_update_unpacked(tsa);
    V_update_unpacked(tsa);
}
