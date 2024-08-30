#ifdef _OPENMP
    #include <omp.h>
#endif

#include "common.h"

static inline float NO_TREE_VECTORIZE kernel_0(float A, float B, float C, float ALPHA, float ALPHA2)
{ 
    return (A + B) * ALPHA2 + C * ALPHA;
}

static inline float NO_TREE_VECTORIZE kernel_1(float A, float B, float C, float D, float ALPHA, float ALPHA2)
{
    return (A + B) * ALPHA2 + (C + D) * ALPHA;
}

static inline float NO_TREE_VECTORIZE kernel_2(float A, float B, float C, float D, float ALPHA)
{
    return (A + B + C + D) * ALPHA;
}

#define UPDATE_UNPACKED_KERNEL \
    HL00 += kernel_0(HH10, HH20_A, HH10_A, (+9.f/32), (-1.f/32));\
    LH00 += kernel_1(HH02_L, HH01, HH01_L, HH00, (+9.f/32), (-1.f/32));\
    LL00 += kernel_2(LH10_A, HL01_L, HL00, LH00, (+9.f/32));\
    LL00 += kernel_2(LH10, LH20_A, HL01, HL02_L, (-1.f/32));\
    LL00 += kernel_2(HH11, HH21_A, HH12_L, HH22_LA, +0.000976562f);\
    LL00 += kernel_2(HH11_L, HH01_A, HH12_LA, HH21_LA, -0.00878906f);\
    LL00 += +0.0791016f * HH11_LA;\
    HL00 += (+9.f/32) * HH00;\
\
    *lh = LH00; *ll = LL00; *hl = HL00;

#define PREDICT_WITH_UNPACK_KERNEL \
    LH00 += kernel_0(LL10_A, LL20, LL10, (-9.f/16), (+1.f/16));\
    HL00 += kernel_1(LL02, LL01_L, LL00, LL01, (-9.f/16), (+1.f/16));\
    HH00 += kernel_2(LH00, LH01, HL00, HL10, (-9.f/16));\
    HH00 += kernel_2(LH01_L, LH02, HL10_A, HL20, (+1.f/16));\
    HH00 += kernel_2(LL11_LA, LL22, LL12_A, LL21_L, +0.00390625f);\
    HH00 += kernel_2(LL11_A, LL12, LL21, LL11_L, -0.0351562f);\
    HH00 += +0.316406f * LL11;\
    LH00 += (-9.f/16) * LL00;\
\
    *ll = LL00; *hl = HL00; *hh = HH00; *lh = LH00;

static inline void NO_TREE_VECTORIZE predict_line_with_unpack_0(
        const float * m0, const float * m1, const float * m2, const float * m3, const float * m4, 
        float * ll, float * hl, float * lh, float * hh,
        int o0, size_t img_start_x, size_t img_end_x)
{
    const int o1 = 2; 
    const int o2 = 4; 
    
    float LL11_LA, LL10_A, HL10_A, LL11_A, LL12_A;
    float LL01_L, LL00, HL00, LL01, LL02;
    float LH01_L, LH00, HH00, LH01, LH02;
    float LL11_L, LL10, HL10, LL11, LL12;
    float LL21_L, LL20, HL20, LL21, LL22;
    float TMP;
    
    LL11_LA = *(m0+o0); LL10_A = *(m0); HL10_A = *(m0+1);   LL11_A = *(m0+o1);  LL12_A = *(m0+o2);
    LL01_L = *(m1+o0);  LL00 = *(m1);   HL00 = *(m1+1);     LL01 = *(m1+o1);    LL02 = *(m1+o2);
    LH01_L = *(m2+o0);  LH00 = *(m2);   HH00 = *(m2+1);     LH01 = *(m2+o1);    LH02 = *(m2+o2);
    LL11_L = *(m3+o0);  LL10 = *(m3);   HL10 = *(m3+1);     LL11 = *(m3+o1);    LL12 = *(m3+o2);
    LL21_L = *(m4+o0);  LL20 = *(m4);   HL20 = *(m4+1);     LL21 = *(m4+o1);    LL22 = *(m4+o2);
    TMP = LH00;

    PREDICT_WITH_UNPACK_KERNEL

    ll++; hh++; hl++; lh++;
    m0 += 2; m1 += 2; m2 += 2; m3 += 2; m4 += 2;

    LL11_LA = LL10_A; LL10_A = LL11_A; LL11_A = LL12_A;
    LL01_L = LL00; LL00 = LL01; LL01 = LL02;
    LH01_L = TMP; LH00 = LH01; LH01 = LH02;
    LL11_L = LL10; LL10 = LL11; LL11 = LL12;
    LL21_L = LL20; LL20 = LL21; LL21 = LL22;

    for(size_t x = img_start_x+2; x < img_end_x - 4; x += 2){
        HL10_A = *(m0+1);   LL12_A = *(m0+o2);
        HL00 = *(m1+1);     LL02 = *(m1+o2);
        HH00 = *(m2+1);     LH02 = *(m2+o2);
        HL10 = *(m3+1);     LL12 = *(m3+o2);
        HL20 = *(m4+1);     LL22 = *(m4+o2);
        TMP = LH00;

        PREDICT_WITH_UNPACK_KERNEL

        ll++; hh++; hl++; lh++;
        m0 += 2; m1 += 2; m2 += 2; m3 += 2; m4 += 2;

        LL11_LA = LL10_A; LL10_A = LL11_A; LL11_A = LL12_A;
        LL01_L = LL00; LL00 = LL01; LL01 = LL02;
        LH01_L = TMP; LH00 = LH01; LH01 = LH02;
        LL11_L = LL10; LL10 = LL11; LL11 = LL12;
        LL21_L = LL20; LL20 = LL21; LL21 = LL22;
    }

    HL10_A = *(m0+1);
    HL00 = *(m1+1);
    HH00 = *(m2+1);
    HL10 = *(m3+1);
    HL20 = *(m4+1);
    TMP = LH00;

    PREDICT_WITH_UNPACK_KERNEL

    ll++; hh++; hl++; lh++;
    m0 += 2; m1 += 2; m2 += 2; m3 += 2; m4 += 2;

    LL11_LA = LL10_A; LL10_A = LL11_A; LL11_A = LL12_A;
    LL01_L = LL00; LL00 = LL01; LL01 = LL02;
    LH01_L = TMP; LH00 = LH01; LH01 = LH02;
    LL11_L = LL10; LL10 = LL11; LL11 = LL12;
    LL21_L = LL20; LL20 = LL21; LL21 = LL22;

    HL10_A = *(m0+1);   LL12_A = LL11_LA;
    HL00 = *(m1+1);     LL02 = LL01_L;
    HH00 = *(m2+1);     LH02 = LH01_L;
    HL10 = *(m3+1);     LL12 = LL11_L;
    HL20 = *(m4+1);     LL22 = LL21_L;
    TMP = LH00;

    PREDICT_WITH_UNPACK_KERNEL

    ll++; hh++; hl++; lh++;
    m0 += 2; m1 += 2; m2 += 2; m3 += 2; m4 += 2;
}

static inline void NO_TREE_VECTORIZE predict_line_with_unpack_1(
        const float * m0, const float * m1, const float * m2, const float * m3, const float * m4, 
        float * ll, float * hl, float * lh, float * hh,
        int o0, size_t img_start_x, size_t img_end_x
)
{
    const int o1 = 2; 
    const int o2 = 4;
    
    float LL11_LA, LL10_A, HL10_A, LL11_A, LL12_A;
    float LL01_L, LL00, HL00, LL01, LL02;
    float LH01_L, LH00, HH00, LH01, LH02;
    float LL11_L, LL10, HL10, LL11, LL12;
    float LL21_L, LL20, HL20, LL21, LL22;
    float TMP;
    
    LL11_LA = *(m0+o0); LL10_A = *(m0); HL10_A = *(m0+1);   LL11_A = *(m0+o1);  LL12_A = *(m0+o2);
    LL01_L = *(m1+o0);  LL00 = *(m1);   HL00 = *(m1+1);     LL01 = *(m1+o1);    LL02 = *(m1+o2);
    LH01_L = *(m2+o0);  LH00 = *(m2);   HH00 = *(m2+1);     LH01 = *(m2+o1);    LH02 = *(m2+o2);
    LL11_L = *(m3+o0);  LL10 = *(m3);   HL10 = *(m3+1);     LL11 = *(m3+o1);    LL12 = *(m3+o2);
    LL21_L = *(m4+o0);  LL20 = *(m4);   HL20 = *(m4+1);     LL21 = *(m4+o1);    LL22 = *(m4+o2);
    TMP = LH00;

    PREDICT_WITH_UNPACK_KERNEL

    ll++; hh++; hl++; lh++;
    m0 += 2; m1 += 2; m2 += 2; m3 += 2; m4 += 2;

    LL11_LA = LL10_A; LL10_A = LL11_A; LL11_A = LL12_A;
    LL01_L = LL00; LL00 = LL01; LL01 = LL02;
    LH01_L = TMP; LH00 = LH01; LH01 = LH02;
    LL11_L = LL10; LL10 = LL11; LL11 = LL12;
    LL21_L = LL20; LL20 = LL21; LL21 = LL22;

    for(size_t x = img_start_x+2; x < img_end_x - 2; x += 2){
        HL10_A = *(m0+1);   LL12_A = *(m0+o2);
        HL00 = *(m1+1);     LL02 = *(m1+o2);
        HH00 = *(m2+1);     LH02 = *(m2+o2);
        HL10 = *(m3+1);     LL12 = *(m3+o2);
        HL20 = *(m4+1);     LL22 = *(m4+o2);
        TMP = LH00;

        PREDICT_WITH_UNPACK_KERNEL

        ll++; hh++; hl++; lh++;
        m0 += 2; m1 += 2; m2 += 2; m3 += 2; m4 += 2;

        LL11_LA = LL10_A; LL10_A = LL11_A; LL11_A = LL12_A;
        LL01_L = LL00; LL00 = LL01; LL01 = LL02;
        LH01_L = TMP; LH00 = LH01; LH01 = LH02;
        LL11_L = LL10; LL10 = LL11; LL11 = LL12;
        LL21_L = LL20; LL20 = LL21; LL21 = LL22;
    }

    HL10_A = *(m0+1);   LL12_A = *(m0+o2);
    HL00 = *(m1+1);     LL02 = *(m1+o2);
    HH00 = *(m2+1);     LH02 = *(m2+o2);
    HL10 = *(m3+1);     LL12 = *(m3+o2);
    HL20 = *(m4+1);     LL22 = *(m4+o2);
    TMP = LH00;

    PREDICT_WITH_UNPACK_KERNEL

    ll++; hh++; hl++; lh++;
    m0 += 2; m1 += 2; m2 += 2; m3 += 2; m4 += 2;
}

static inline void NO_TREE_VECTORIZE update_line_unpacked_0(
        float * ll, float * hl, float * lh, const float * hh,
        const float * tmp_hl, const float * tmp_lh,
        long long v0, long long v1, long long v2,
        int h2, size_t band_start_x, size_t band_end_x)
{
    const int h0 = -2;
    const int h1 = -1;
    
    float HH22_LA, HH21_LA, LH20_A, HH20_A, HH21_A;
    float HH12_LA, HH11_LA, LH10_A, HH10_A, HH01_A;
    float HL02_L, HL01_L, LL00, HL00, HL01;
    float HH02_L, HH01_L, LH00, HH00, HH01;
    float HH12_L, HH11_L, LH10, HH10, HH11;
    float TMP;
    
    HH22_LA = *(hh+v0+h0);  HH21_LA = *(hh+v0+h1);  LH20_A = *(tmp_lh+v0);  HH20_A = *(hh+v0);  HH21_A = *(hh+v0+h2);
    HH12_LA = *(hh+v1+h0);  HH11_LA = *(hh+v1+h1);  LH10_A = *(tmp_lh+v1);  HH10_A = *(hh+v1);  HH01_A = *(hh+v1+h2);
    HL02_L = *(tmp_hl+h0);  HL01_L = *(tmp_hl+h1);  LL00 = *(ll);           HL00 = *(tmp_hl);   HL01 = *(tmp_hl+h2);
    HH02_L = *(hh+h0);      HH01_L = *(hh+h1);      LH00 = *(tmp_lh);       HH00 = *(hh);       HH01 = *(hh+h2);
    HH12_L = *(hh+v2+h0);   HH11_L = *(hh+v2+h1);   LH10 = *(tmp_lh+v2);    HH10 = *(hh+v2);    HH11 = *(hh+v2+h2);
    TMP = HL00;

    UPDATE_UNPACKED_KERNEL 

    lh--; ll--; hl--; hh--; tmp_hl--; tmp_lh--;

    HH21_A = HH20_A;    HH20_A = HH21_LA;   HH21_LA = HH22_LA;
    HH01_A = HH10_A;    HH10_A = HH11_LA;   HH11_LA = HH12_LA;
    HL01 = TMP;         HL00 = HL01_L;      HL01_L = HL02_L;
    HH01 = HH00;        HH00 = HH01_L;      HH01_L = HH02_L;
    HH11 = HH10;        HH10 = HH11_L;      HH11_L = HH12_L;

    for(size_t x = band_end_x - 2; x >= band_start_x + 2; --x){
        HH22_LA = *(hh+v0+h0);  LH20_A = *(tmp_lh+v0);
        HH12_LA = *(hh+v1+h0);  LH10_A = *(tmp_lh+v1);
        HL02_L = *(tmp_hl+h0);  LL00 = *(ll);
        HH02_L = *(hh+h0);      LH00 = *(tmp_lh);
        HH12_L = *(hh+v2+h0);   LH10 = *(tmp_lh+v2);
        TMP = HL00;

        UPDATE_UNPACKED_KERNEL 

        lh--; ll--; hl--; hh--; tmp_hl--; tmp_lh--;

        HH21_A = HH20_A;    HH20_A = HH21_LA;   HH21_LA = HH22_LA;
        HH01_A = HH10_A;    HH10_A = HH11_LA;   HH11_LA = HH12_LA;
        HL01 = TMP;         HL00 = HL01_L;      HL01_L = HL02_L;
        HH01 = HH00;        HH00 = HH01_L;      HH01_L = HH02_L;
        HH11 = HH10;        HH10 = HH11_L;      HH11_L = HH12_L;
    }

    LH20_A = *(tmp_lh+v0);
    LH10_A = *(tmp_lh+v1);
    LL00 = *(ll);
    LH00 = *(tmp_lh);
    LH10 = *(tmp_lh+v2);
    TMP = HL00;

    UPDATE_UNPACKED_KERNEL 

    lh--; ll--; hl--; hh--; tmp_hl--; tmp_lh--;

    HH21_A = HH20_A;    HH20_A = HH21_LA;   HH21_LA = HH22_LA;
    HH01_A = HH10_A;    HH10_A = HH11_LA;   HH11_LA = HH12_LA;
    HL01 = TMP;         HL00 = HL01_L;      HL01_L = HL02_L;
    HH01 = HH00;        HH00 = HH01_L;      HH01_L = HH02_L;
    HH11 = HH10;        HH10 = HH11_L;      HH11_L = HH12_L;

    HH22_LA = HH21_A;   LH20_A = *(tmp_lh+v0);
    HH12_LA = HH01_A;   LH10_A = *(tmp_lh+v1);
    HL02_L = HL01;      LL00 = *(ll);
    HH02_L = HH01;      LH00 = *(tmp_lh);
    HH12_L = HH11;      LH10 = *(tmp_lh+v2);
    TMP = HL00;

    UPDATE_UNPACKED_KERNEL 

    lh--; ll--; hl--; hh--; tmp_hl--; tmp_lh--;
}

static inline void NO_TREE_VECTORIZE update_line_unpacked_1(
        float * ll, float * hl, float * lh, const float * hh,
        const float * tmp_hl, const float * tmp_lh,
        long long v0, long long v1, long long v2,
        int h2, size_t band_start_x, size_t band_end_x)
{
    const int h0 = -2;
    const int h1 = -1;
    
    float HH22_LA, HH21_LA, LH20_A, HH20_A, HH21_A;
    float HH12_LA, HH11_LA, LH10_A, HH10_A, HH01_A;
    float HL02_L, HL01_L, LL00, HL00, HL01;
    float HH02_L, HH01_L, LH00, HH00, HH01;
    float HH12_L, HH11_L, LH10, HH10, HH11;
    float TMP;
    
    HH22_LA = *(hh+v0+h0);  HH21_LA = *(hh+v0+h1);  LH20_A = *(tmp_lh+v0);  HH20_A = *(hh+v0);  HH21_A = *(hh+v0+h2);
    HH12_LA = *(hh+v1+h0);  HH11_LA = *(hh+v1+h1);  LH10_A = *(tmp_lh+v1);  HH10_A = *(hh+v1);  HH01_A = *(hh+v1+h2);
    HL02_L = *(tmp_hl+h0);  HL01_L = *(tmp_hl+h1);  LL00 = *(ll);           HL00 = *(tmp_hl);   HL01 = *(tmp_hl+h2);
    HH02_L = *(hh+h0);      HH01_L = *(hh+h1);      LH00 = *(tmp_lh);       HH00 = *(hh);       HH01 = *(hh+h2);
    HH12_L = *(hh+v2+h0);   HH11_L = *(hh+v2+h1);   LH10 = *(tmp_lh+v2);    HH10 = *(hh+v2);    HH11 = *(hh+v2+h2);
    TMP = HL00;

    UPDATE_UNPACKED_KERNEL 

    lh--; ll--; hl--; hh--; tmp_hl--; tmp_lh--;

    HH21_A = HH20_A;    HH20_A = HH21_LA;   HH21_LA = HH22_LA;
    HH01_A = HH10_A;    HH10_A = HH11_LA;   HH11_LA = HH12_LA;
    HL01 = TMP;         HL00 = HL01_L;      HL01_L = HL02_L;
    HH01 = HH00;        HH00 = HH01_L;      HH01_L = HH02_L;
    HH11 = HH10;        HH10 = HH11_L;      HH11_L = HH12_L;

    for(size_t x = band_end_x - 2; x >= band_start_x + 1; --x){
        HH22_LA = *(hh+v0+h0);  LH20_A = *(tmp_lh+v0);
        HH12_LA = *(hh+v1+h0);  LH10_A = *(tmp_lh+v1);
        HL02_L = *(tmp_hl+h0);  LL00 = *(ll);
        HH02_L = *(hh+h0);      LH00 = *(tmp_lh);
        HH12_L = *(hh+v2+h0);   LH10 = *(tmp_lh+v2);
        TMP = HL00;

        UPDATE_UNPACKED_KERNEL 

        lh--; ll--; hl--; hh--; tmp_hl--; tmp_lh--;

        HH21_A = HH20_A;    HH20_A = HH21_LA;   HH21_LA = HH22_LA;
        HH01_A = HH10_A;    HH10_A = HH11_LA;   HH11_LA = HH12_LA;
        HL01 = TMP;         HL00 = HL01_L;      HL01_L = HL02_L;
        HH01 = HH00;        HH00 = HH01_L;      HH01_L = HH02_L;
        HH11 = HH10;        HH10 = HH11_L;      HH11_L = HH12_L;
    }

    HH22_LA = *(hh+v0+h0);  LH20_A = *(tmp_lh+v0);
    HH12_LA = *(hh+v1+h0);  LH10_A = *(tmp_lh+v1);
    HL02_L = *(tmp_hl+h0);  LL00 = *(ll);
    HH02_L = *(hh+h0);      LH00 = *(tmp_lh);
    HH12_L = *(hh+v2+h0);   LH10 = *(tmp_lh+v2);
    TMP = HL00;

    UPDATE_UNPACKED_KERNEL 

    lh--; ll--; hl--; hh--; tmp_hl--; tmp_lh--;
}

static void predict_with_unpack(const TransformStepArguments * tsa)
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

    const size_t next_band_y = band_stride_y;
    const size_t next_tile_y = 2 * stride_y;
    
    float * m0 = mem + img_start_x + (img_start_y-2) * stride_y;
    float * m1 = mem + img_start_x + img_start_y * stride_y;
    float * m2 = mem + img_start_x + (img_start_y+1) * stride_y;
    float * m3 = mem + img_start_x + (img_start_y+2) * stride_y;
    float * m4 = mem + img_start_x + (img_start_y+4) * stride_y;
        
    float * ll = tsa->tile_bands.LL + band_start_x + band_start_y * band_stride_y;
    float * hh = tsa->tile_bands.HH + band_start_x + band_start_y * band_stride_y;
    
    float * tmp_hl = tsa->tmp.HL + band_start_x + band_start_y * band_stride_y;
    float * tmp_lh = tsa->tmp.LH + band_start_x + band_start_y * band_stride_y;

    int o0;   
    
    if(img_start_y == 0 && img_end_y == size_y){
        if(img_end_x == size_x && img_start_x == 0){
            o0 = 2;
            m0 = m3; 
            predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

            ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
            m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;

            m0 = m1 - 2 * stride_y;    
            for(size_t y = img_start_y + 2; y < img_end_y - 4; y += 2){                
                predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            m4 = m3;
            predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

            ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;

            m3 = m1;
            m4 = m0;
            predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);
        } else if(img_end_x == size_x){
            o0 = -2;
            m0 = m3;
            predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

            ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
            m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;

            m0 = m1 - 2 * stride_y;    
            for(size_t y = img_start_y + 2; y < img_end_y - 4; y += 2){                            
                predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            m4 = m3;
            predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

            ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;

            m3 = m1;
            m4 = m0;
            predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);
        } else if(img_start_x == 0){
            o0 = 2;
            m0 = m3; 
            predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

            ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
            m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;

            m0 = m1 - 2 * stride_y;    
            for(size_t y = img_start_y + 2; y < img_end_y - 4; y += 2){
                predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            m4 = m3;
            predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

            ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;

            m3 = m1;
            m4 = m0;

            predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);
        } else {
            o0 = -2;
            m0 = m3; 
            predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

            ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
            m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;

            m0 = m1 - 2 * stride_y;    
            for(size_t y = img_start_y + 2; y < img_end_y - 4; y += 2){                            
                predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }
            
            m4 = m3;
            predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

            ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;

            m3 = m1;
            m4 = m0;
            predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);
        }
    } else if(img_start_y == 0 && img_end_y == size_y - 2){
        if(img_end_x == size_x && img_start_x == 0){
            o0 = 2;
            m0 = m3; 
            predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

            ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
            m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;

            m0 = m1 - 2 * stride_y;    
            for(size_t y = img_start_y + 2; y < img_end_y - 2; y += 2){                
                predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }            
            m4 = m3;
            predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);
        } else if(img_end_x == size_x){
            o0 = -2;
            m0 = m3;
            predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

            ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
            m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;

            m0 = m1 - 2 * stride_y;    
            for(size_t y = img_start_y + 2; y < img_end_y - 2; y += 2){                            
                predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            m4 = m3;
            predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);
        } else if(img_start_x == 0){
            o0 = 2;
            m0 = m3; 
            predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

            ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
            m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;

            m0 = m1 - 2 * stride_y;    
            for(size_t y = img_start_y + 2; y < img_end_y - 2; y += 2){
                predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            m4 = m3;
            predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);
        } else {
            o0 = -2;
            m0 = m3; 
            predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

            ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
            m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;

            m0 = m1 - 2 * stride_y;    
            for(size_t y = img_start_y + 2; y < img_end_y - 2; y += 2){
                predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            m4 = m3;
            predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);
        }
    } else if(img_start_y == 0){
        if(img_end_x == size_x && img_start_x == 0){
            o0 = 2;
            m0 = m3; 
            predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

            ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
            m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;

            m0 = m1 - 2 * stride_y;    
            for(size_t y = img_start_y + 2; y < img_end_y; y += 2){
                predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }
        } else if(img_end_x == size_x){
            o0 = -2;
            m0 = m3;
            predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

            ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
            m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;

            m0 = m1 - 2 * stride_y;    
            for(size_t y = img_start_y + 2; y < img_end_y; y += 2){
                predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }
        } else if(img_start_x == 0){
            o0 = 2;
            m0 = m3; 
            predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

            ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
            m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;

            m0 = m1 - 2 * stride_y;    
            for(size_t y = img_start_y + 2; y < img_end_y; y += 2){
                predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }
        } else {
            o0 = -2;
            m0 = m3; 
            predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

            ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
            m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;

            m0 = m1 - 2 * stride_y;    
            for(size_t y = img_start_y + 2; y < img_end_y; y += 2){                            
                predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }
        }
    } else if(img_end_y == size_y){
        if(img_end_x == size_x && img_start_x == 0){
            o0 = 2;   
            for(size_t y = img_start_y; y < img_end_y - 4; y += 2){                
                predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            if(img_start_y  < size_y -2){
                m4 = m3;
                predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            m3 = m1;
            m4 = m0;
            predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);
        } else if(img_end_x == size_x){
            o0 = -2;   
            for(size_t y = img_start_y; y < img_end_y - 4; y += 2){                            
                predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            if(img_start_y  < size_y -2){
                m4 = m3;
                predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            m3 = m1;
            m4 = m0;
            predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);
        } else if(img_start_x == 0){
            o0 = 2;    
            for(size_t y = img_start_y; y < img_end_y - 4; y += 2){
                predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            if(img_start_y  < size_y -2){
                m4 = m3;
                predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            m3 = m1;
            m4 = m0;
            predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);
        } else {
            o0 = -2;   
            for(size_t y = img_start_y; y < img_end_y - 4; y += 2){                            
                predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }
            
            if(img_start_y  < size_y -2){
                m4 = m3;
                predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            m3 = m1;
            m4 = m0;
            predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);
        }
    } else if(img_end_y == size_y - 2){
        if(img_end_x == size_x && img_start_x == 0){
            o0 = 2;    
            for(size_t y = img_start_y; y < img_end_y - 2; y += 2){                
                predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            m4 = m3;
            predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

        } else if(img_end_x == size_x){
            o0 = -2;
            for(size_t y = img_start_y; y < img_end_y - 2; y += 2){                            
                predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            m4 = m3;
            predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);
        } else if(img_start_x == 0){
            o0 = 2;   
            for(size_t y = img_start_y; y < img_end_y - 2; y += 2){
                predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            m4 = m3;
            predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);
        } else {
            o0 = -2;  
            for(size_t y = img_start_y; y < img_end_y - 2; y += 2){                            
                predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            m4 = m3;
            predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);
        }
    } else {
        if(img_end_x == size_x && img_start_x == 0){
            o0 = 2; 
            for(size_t y = img_start_y; y < img_end_y; y += 2){                
                predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }
        } else if(img_end_x == size_x){
            o0 = -2;  
            for(size_t y = img_start_y; y < img_end_y; y += 2){                            
                predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }
        } else if(img_start_x == 0){
            o0 = 2; 
            for(size_t y = img_start_y; y < img_end_y; y += 2){
                predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }
        } else {
            o0 = -2;  
            for(size_t y = img_start_y; y < img_end_y; y += 2){                            
                predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, o0, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }
        }
    }
    # pragma omp barrier
}

static void update_unpacked(const TransformStepArguments * tsa)
{
    const size_t band_size_y = tsa->tile_bands.size_y;
    const size_t band_size_x = tsa->tile_bands.size_x;
    const size_t band_stride_y = tsa->tile_bands.stride_y;
    
    const size_t tid = (size_t)omp_get_thread_num();

    const size_t band_start_y = tsa->threading_info->band_start_y[tid];
    const size_t band_end_y = tsa->threading_info->band_end_y[tid];
    const size_t band_start_x = tsa->threading_info->band_start_x[tid];
    const size_t band_end_x = tsa->threading_info->band_end_x[tid];     

    const size_t next_band_y = band_stride_y;

    float * ll = tsa->tile_bands.LL + (band_end_y - 1) * band_stride_y + band_end_x - 1;
    float * hl = tsa->tile_bands.HL + (band_end_y - 1) * band_stride_y + band_end_x - 1;
    float * lh = tsa->tile_bands.LH + (band_end_y - 1) * band_stride_y + band_end_x - 1;
    float * hh = tsa->tile_bands.HH + (band_end_y - 1) * band_stride_y + band_end_x - 1;
    
    float * tmp_hl = tsa->tmp.HL + (band_end_y - 1) * band_stride_y + band_end_x - 1;
    float * tmp_lh = tsa->tmp.LH + (band_end_y - 1) * band_stride_y + band_end_x - 1;
    
    long long v0 = -2 * (long long ) band_stride_y, v1 = - (long long) band_stride_y, v2 = (long long) band_stride_y;
    int h2 = 1;
    
    if(band_end_y == band_size_y && band_start_y == 0){
        if(band_end_x == band_size_x && band_start_x == 0){
            h2 = -1;
            v2 = -(long long) band_stride_y;
            update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);

            ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
            tmp_hl -= next_band_y; tmp_lh -= next_band_y;

            v2 = (long long) band_stride_y;
            for(size_t y = band_end_y-2; y >= band_start_y + 2; --y){
                update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }   

            v0 = -(long long) band_stride_y;
            update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);

            ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
            tmp_hl -= next_band_y; tmp_lh -= next_band_y;

            v0 = (long long) band_stride_y;
            v1 = 0;
            update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);
        } else if(band_start_x == 0){
            v2 = -(long long) band_stride_y;
            update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);                        

            ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
            tmp_hl -= next_band_y; tmp_lh -= next_band_y;

            v2 = (long long) band_stride_y;
            for(size_t y = band_end_y-2; y >= band_start_y + 2; --y){
                update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);                        

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }   

            v0 = -(long long) band_stride_y;
            update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);                        

            ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
            tmp_hl -= next_band_y; tmp_lh -= next_band_y;

            v0 = (long long) band_stride_y;
            v1 = 0;
            update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);                           
        } else if(band_end_x == band_size_x){
            h2 = -1;
            v2 = -(long long) band_stride_y;
            update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);

            ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
            tmp_hl -= next_band_y; tmp_lh -= next_band_y;

            v2 = (long long) band_stride_y;
            for(size_t y = band_end_y-2; y >= band_start_y + 2; --y){
                update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }   

            v0 = -(long long) band_stride_y;
            update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);

            ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
            tmp_hl -= next_band_y; tmp_lh -= next_band_y;

            v0 = (long long) band_stride_y;
            v1 = 0;
            update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);
        } else {
            v2 = -(long long) band_stride_y;

            update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);

            ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
            tmp_hl -= next_band_y; tmp_lh -= next_band_y;

            v2 = (long long) band_stride_y;
            for(size_t y = band_end_y-2; y >= band_start_y + 2; --y){
                update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }   

            v0 = -(long long) band_stride_y;
            update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);

            ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
            tmp_hl -= next_band_y; tmp_lh -= next_band_y;

            v0 = (long long) band_stride_y;
            v1 = 0;
            update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);
        }
    } else if(band_end_y == band_size_y && band_start_y == 1){
        if(band_end_x == band_size_x && band_start_x == 0){
            h2 = -1;
            v2 = -(long long) band_stride_y;
            update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);

            ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
            tmp_hl -= next_band_y; tmp_lh -= next_band_y;

            v2 = (long long) band_stride_y;
            for(size_t y = band_end_y-2; y >= band_start_y + 1; --y){
                update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }   

            v0 = -(long long) band_stride_y;
            update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);
        } else if(band_start_x == 0){
            v2 = -(long long) band_stride_y;
            update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);                        

            ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
            tmp_hl -= next_band_y; tmp_lh -= next_band_y;

            v2 = (long long) band_stride_y;
            for(size_t y = band_end_y-2; y >= band_start_y + 1; --y){
                update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);                        

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }   

            v0 = -(long long) band_stride_y;
            update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);                                              
        } else if(band_end_x == band_size_x){
            h2 = -1;
            v2 = -(long long) band_stride_y;
            update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);

            ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
            tmp_hl -= next_band_y; tmp_lh -= next_band_y;

            v2 = (long long) band_stride_y;
            for(size_t y = band_end_y-2; y >= band_start_y + 1; --y){
                update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }   

            v0 = -(long long) band_stride_y;
            update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);
        } else {
            v2 = -(long long) band_stride_y;

            update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);

            ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
            tmp_hl -= next_band_y; tmp_lh -= next_band_y;

            v2 = (long long) band_stride_y;
            for(size_t y = band_end_y-2; y >= band_start_y + 1; --y){
                update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }   

            v0 = -(long long) band_stride_y;
            update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);
        }
    } else if(band_end_y == band_size_y){
        if(band_end_x == band_size_x && band_start_x == 0){
            h2 = -1;
            v2 = -(long long) band_stride_y;
            update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);

            ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
            tmp_hl -= next_band_y; tmp_lh -= next_band_y;

            v2 = (long long) band_stride_y;
            for(size_t y = band_end_y-2; y >= band_start_y; --y){
                update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }   
        } else if(band_start_x == 0){
            v2 = -(long long) band_stride_y;
            update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);                        

            ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
            tmp_hl -= next_band_y; tmp_lh -= next_band_y;

            v2 = (long long) band_stride_y;
            for(size_t y = band_end_y-2; y >= band_start_y; --y){
                update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);                        

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }                                              
        } else if(band_end_x == band_size_x){
            h2 = -1;
            v2 = -(long long) band_stride_y;
            update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);

            ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
            tmp_hl -= next_band_y; tmp_lh -= next_band_y;

            v2 = (long long) band_stride_y;
            for(size_t y = band_end_y-2; y >= band_start_y; --y){
                update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }   
        } else {
            v2 = -(long long) band_stride_y;

            update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);

            ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
            tmp_hl -= next_band_y; tmp_lh -= next_band_y;

            v2 = (long long) band_stride_y;
            for(size_t y = band_end_y-2; y >= band_start_y; --y){
                update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }
        }
    } else if(band_start_y == 0) {
        if(band_end_x == band_size_x && band_start_x == 0){
            h2 = -1;
            for(size_t y = band_end_y-1; y >= band_start_y + 2; --y){
                update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }   

            if(band_end_y > 1){
                v0 = -(long long) band_stride_y;
                update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }

            v0 = (long long) band_stride_y;
            v1 = 0;
            update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);
        } else if(band_start_x == 0){
            for(size_t y = band_end_y-1; y >= band_start_y + 2; --y){
                update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);                        

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }   

            if(band_end_y > 1){
                v0 = -(long long) band_stride_y;
                update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);                        

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }

            v0 = (long long) band_stride_y;
            v1 = 0;
            update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);                           
        } else if(band_end_x == band_size_x){
            h2 = -1;
            for(size_t y = band_end_y-1; y >= band_start_y + 2; --y){
                update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }   

            if(band_end_y > 1){
                v0 = -(long long) band_stride_y;
                update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }

            v0 = (long long) band_stride_y;
            v1 = 0;
            update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);
        } else {
            for(size_t y = band_end_y-1; y >= band_start_y + 2; --y){
                update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }   

            if(band_end_y > 1){
                v0 = -(long long) band_stride_y;
                update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }

            v0 = (long long) band_stride_y;
            v1 = 0;
            update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);
        }
    } else if(band_start_y == 1) {
        if(band_end_x == band_size_x && band_start_x == 0){
            h2 = -1;
            for(size_t y = band_end_y-1; y >= band_start_y + 1; --y){
                update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }   

            v0 = -(long long) band_stride_y;
            update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);
        } else if(band_start_x == 0){
            for(size_t y = band_end_y-1; y >= band_start_y + 1; --y){
                update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);                        

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }   

            v0 = -(long long) band_stride_y;
            update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);                                                 
        } else if(band_end_x == band_size_x){
            h2 = -1;
            for(size_t y = band_end_y-1; y >= band_start_y + 1; --y){
                update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }   

            v0 = -(long long) band_stride_y;
            update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);
        } else {
            for(size_t y = band_end_y-1; y >= band_start_y + 1; --y){
                update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }   

            v0 = -(long long) band_stride_y;
            update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);
        }
    } else {
        if(band_end_x == band_size_x && band_start_x == 0){
            h2 = -1;
            for(size_t y = band_end_y-1; y >= band_start_y; --y){
                update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }   
        } else if(band_start_x == 0){
            for(size_t y = band_end_y-1; y >= band_start_y; --y){
                update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);                        

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }                                                  
        } else if(band_end_x == band_size_x){
            h2 = -1;
            for(size_t y = band_end_y-1; y >= band_start_y; --y){
                update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }   
        } else {
            for(size_t y = band_end_y-1; y >= band_start_y; --y){
                update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, h2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y; 
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }   
        }
    }
}

TRANSFORM_STEP_FUNC NO_TREE_VECTORIZE dd137_non_separable_lifting_generic(size_t step)
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

void NO_TREE_VECTORIZE dd137_non_separable_lifting_generic_transform_tile(const TransformStepArguments * tsa)
{
    predict_with_unpack(tsa);
    update_unpacked(tsa);
}
