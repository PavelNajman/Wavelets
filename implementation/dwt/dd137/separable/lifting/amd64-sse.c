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

static inline void NO_TREE_VECTORIZE H_predict_with_unpack_kernel(__m128 C1_L, __m128 C0, __m128 C1, __m128 C2, float * l_, float * h_)
{
    const __m128 ALPHA = _mm_set_ps1(-9.f/16);
    const __m128 ALPHA2 = _mm_set_ps1(+1.f/16);

    __m128 L_00, L_01, L_02, L_02_L, H_00;

    L_00 = load_packed_LX00(C0, C1);
    H_00 = load_packed_HX00(C0, C1);

    L_01 = load_packed_LX01(L_00, C2);
    L_02 = load_packed_LX02(L_01, C2);
    L_02_L = load_packed_LX01L(L_00, C1_L);

    L_01 = _mm_add_ps(L_00, L_01);
    L_02_L = _mm_add_ps(L_02_L, L_02);

#ifdef USE_FMA
    H_00 = _mm_fmadd_ps(L_01, ALPHA, H_00);
    H_00 = _mm_fmadd_ps(L_02_L, ALPHA2, H_00);
#else
    L_01 = _mm_mul_ps(L_01, ALPHA);
    L_02_L = _mm_mul_ps(L_02_L, ALPHA2);
    H_00 = _mm_add_ps(L_01, H_00);
    H_00 = _mm_add_ps(L_02_L, H_00);
#endif

    _mm_store_ps(l_, L_00);
    _mm_store_ps(h_, H_00);
}

static inline void NO_TREE_VECTORIZE V_predict_unpacked_kernel(const float * r0, const float * r1, float * r2, const float * r3, const float * r4)
{
    const __m128 ALPHA = _mm_set_ps1(-9.f/16);
    const __m128 ALPHA2 = _mm_set_ps1(+1.f/16);

    __m128 _L10_A = _mm_load_ps(r0);
    __m128 _L00 = _mm_load_ps(r1);
    __m128 _H00 = _mm_load_ps(r2);
    __m128 _L10 = _mm_load_ps(r3);
    __m128 _L20 = _mm_load_ps(r4);

    _L00 = _mm_add_ps(_L00, _L10);
    _L20 = _mm_add_ps(_L10_A, _L20);
#ifdef USE_FMA
    _H00 = _mm_fmadd_ps(_L00, ALPHA, _H00);
    _H00 = _mm_fmadd_ps(_L20, ALPHA2, _H00);
#else
    _L00 = _mm_mul_ps(_L00, ALPHA);
    _L20 = _mm_mul_ps(_L20, ALPHA2);
    _H00 = _mm_add_ps(_L00, _H00);
    _H00 = _mm_add_ps(_L20, _H00);
#endif

    _mm_store_ps(r2, _H00);
}

static inline void NO_TREE_VECTORIZE H_update_unpacked_kernel(__m128 c0, __m128 c1, __m128 c2, __m128 c3, float * dst)
{
    const __m128 BETA = _mm_set_ps1(+9.f/32);
    const __m128 BETA2 = _mm_set_ps1(-1.f/32);

    __m128 TMP;

    __m128 HL02_L;
    __m128 HL01_L = c0;
    __m128 LL00 = c1;
    __m128 HL00 = c2;
    __m128 HL02 = c3;

    TMP = load_unpacked_XX01L(HL00, HL01_L);

    HL02_L = load_unpacked_XX02L(TMP, HL01_L);

    HL01_L = TMP;

    HL02 = load_unpacked_XX01(HL00, HL02);

    HL01_L = _mm_add_ps(HL01_L, HL00);
    HL02_L = _mm_add_ps(HL02_L, HL02);

#ifdef USE_FMA
    LL00 = _mm_fmadd_ps(HL01_L, BETA, LL00);
    LL00 = _mm_fmadd_ps(HL02_L, BETA2, LL00);
#else
    HL01_L = _mm_mul_ps(HL01_L, BETA);
    HL02_L = _mm_mul_ps(HL02_L, BETA2);
    LL00 = _mm_add_ps(HL01_L, LL00);
    LL00 = _mm_add_ps(HL02_L, LL00);
#endif
    _mm_store_ps(dst, LL00);
}

static inline void NO_TREE_VECTORIZE V_update_unpacked_kernel(const float * r0, const float * r1, float * r2, const float * r3, const float * r4)
{
    const __m128 BETA = _mm_set_ps1(+9.f/32);
    const __m128 BETA2 = _mm_set_ps1(-1.f/32);

    __m128 LH20_A = _mm_load_ps(r0);
    __m128 LH10_A = _mm_load_ps(r1);
    __m128 LL00 = _mm_load_ps(r2);
    __m128 LH00 = _mm_load_ps(r3);
    __m128 LH20 = _mm_load_ps(r4);

    LH00 = _mm_add_ps(LH00, LH10_A);
    LH20 = _mm_add_ps(LH20, LH20_A);
#ifdef USE_FMA
    LL00 = _mm_fmadd_ps(LH00, BETA, LL00);
    LL00 = _mm_fmadd_ps(LH20, BETA2, LL00);
#else
    LH00 = _mm_mul_ps(LH00, BETA);
    LH20 = _mm_mul_ps(LH20, BETA2);
    LL00 = _mm_add_ps(LH00, LL00);
    LL00 = _mm_add_ps(LH20, LL00);
#endif

    _mm_store_ps(r2, LL00);
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

    __m128 R0C1_L, R0C0, R0C1, R0C2, R1C1_L, R1C0, R1C1, R1C2;

    if(img_start_x == 0 && img_end_x == size_x){
        for(size_t y = img_start_y; y < img_end_y; y += 2){
            R0C0 = _mm_load_ps(m0); R0C1 = _mm_load_ps(m0+4); R0C2 = _mm_load_ps(m0+8);
            R1C0 = _mm_load_ps(m1); R1C1 = _mm_load_ps(m1+4); R1C2 = _mm_load_ps(m1+8);
            H_predict_with_unpack_kernel(R0C0, R0C0, R0C1, R0C2, ll, hl);
            H_predict_with_unpack_kernel(R1C0, R1C0, R1C1, R1C2, lh, hh);

            R0C1_L = R0C1; R0C0 = R0C2;
            R1C1_L = R1C1; R1C0 = R1C2;
            ll += 4; hl += 4; hh += 4; lh += 4;
            m0 += 8; m1 += 8;

            for(size_t x = img_start_x + 8; x < img_end_x-8; x += 8){
                R0C1 = _mm_load_ps(m0+4); R0C2 = _mm_load_ps(m0+8);
                R1C1 = _mm_load_ps(m1+4); R1C2 = _mm_load_ps(m1+8);

                H_predict_with_unpack_kernel(R0C1_L, R0C0, R0C1, R0C2, ll, hl);
                H_predict_with_unpack_kernel(R1C1_L, R1C0, R1C1, R1C2, lh, hh);

                R0C1_L = R0C1; R0C0 = R0C2;
                R1C1_L = R1C1; R1C0 = R1C2;
                ll += 4; hl += 4; hh += 4; lh += 4;
                m0 += 8; m1 += 8;
            }

            R0C1 = _mm_load_ps(m0+4);
            R0C2 = _mm_loadr_ps(m0+4);
            R0C2 = rotate_right(R0C2);

            R1C1 = _mm_load_ps(m1+4);
            R1C2 = _mm_loadr_ps(m1+4);
            R1C2 = rotate_right(R1C2);

            H_predict_with_unpack_kernel(R0C1_L, R0C0, R0C1, R0C2, ll, hl);
            H_predict_with_unpack_kernel(R1C1_L, R1C0, R1C1, R1C2, lh, hh);

            ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y;
        }
    } else if(img_start_x == 0){
        for(size_t y = img_start_y; y < img_end_y; y += 2){
            R0C0 = _mm_load_ps(m0); R0C1 = _mm_load_ps(m0+4); R0C2 = _mm_load_ps(m0+8);
            R1C0 = _mm_load_ps(m1); R1C1 = _mm_load_ps(m1+4); R1C2 = _mm_load_ps(m1+8);
            H_predict_with_unpack_kernel(R0C0, R0C0, R0C1, R0C2, ll, hl);
            H_predict_with_unpack_kernel(R1C0, R1C0, R1C1, R1C2, lh, hh);

            R0C1_L = R0C1; R0C0 = R0C2;
            R1C1_L = R1C1; R1C0 = R1C2;
            ll += 4; hl += 4; hh += 4; lh += 4;
            m0 += 8; m1 += 8;

            for(size_t x = img_start_x + 8; x < img_end_x-8; x += 8){
                R0C1 = _mm_load_ps(m0+4); R0C2 = _mm_load_ps(m0+8);
                R1C1 = _mm_load_ps(m1+4); R1C2 = _mm_load_ps(m1+8);

                H_predict_with_unpack_kernel(R0C1_L, R0C0, R0C1, R0C2, ll, hl);
                H_predict_with_unpack_kernel(R1C1_L, R1C0, R1C1, R1C2, lh, hh);

                R0C1_L = R0C1; R0C0 = R0C2;
                R1C1_L = R1C1; R1C0 = R1C2;
                ll += 4; hl += 4; hh += 4; lh += 4;
                m0 += 8; m1 += 8;
            }

            R0C1 = _mm_load_ps(m0+4); R0C2 = _mm_load_ps(m0+8);
            R1C1 = _mm_load_ps(m1+4); R1C2 = _mm_load_ps(m1+8);
            H_predict_with_unpack_kernel(R0C1_L, R0C0, R0C1, R0C2, ll, hl);
            H_predict_with_unpack_kernel(R1C1_L, R1C0, R1C1, R1C2, lh, hh);

            ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y;
        }
    } else if(img_end_x == size_x){
        for(size_t y = img_start_y; y < img_end_y; y += 2){
            R0C1_L = _mm_load_ps(m0-4); R0C0 = _mm_load_ps(m0);
            R1C1_L = _mm_load_ps(m1-4); R1C0 = _mm_load_ps(m1);

            for(size_t x = img_start_x; x < img_end_x-8; x += 8){
                R0C1 = _mm_load_ps(m0+4); R0C2 = _mm_load_ps(m0+8);
                R1C1 = _mm_load_ps(m1+4); R1C2 = _mm_load_ps(m1+8);

                H_predict_with_unpack_kernel(R0C1_L, R0C0, R0C1, R0C2, ll, hl);
                H_predict_with_unpack_kernel(R1C1_L, R1C0, R1C1, R1C2, lh, hh);

                R0C1_L = R0C1; R0C0 = R0C2;
                R1C1_L = R1C1; R1C0 = R1C2;
                ll += 4; hl += 4; hh += 4; lh += 4;
                m0 += 8; m1 += 8;
            }

            R0C1 = _mm_load_ps(m0+4);
            R0C2 = _mm_loadr_ps(m0+4);
            R0C2 = rotate_right(R0C2);

            R1C1 = _mm_load_ps(m1+4);
            R1C2 = _mm_loadr_ps(m1+4);
            R1C2 = rotate_right(R1C2);
            H_predict_with_unpack_kernel(R0C1_L, R0C0, R0C1, R0C2, ll, hl);
            H_predict_with_unpack_kernel(R1C1_L, R1C0, R1C1, R1C2, lh, hh);

            ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y;
        }
    } else {
        for(size_t y = img_start_y; y < img_end_y; y += 2){
            R0C1_L = _mm_load_ps(m0-4); R0C0 = _mm_load_ps(m0);
            R1C1_L = _mm_load_ps(m1-4); R1C0 = _mm_load_ps(m1);
            for(size_t x = img_start_x; x < img_end_x-8; x += 8){
                R0C1 = _mm_load_ps(m0+4); R0C2 = _mm_load_ps(m0+8);
                R1C1 = _mm_load_ps(m1+4); R1C2 = _mm_load_ps(m1+8);

                H_predict_with_unpack_kernel(R0C1_L, R0C0, R0C1, R0C2, ll, hl);
                H_predict_with_unpack_kernel(R1C1_L, R1C0, R1C1, R1C2, lh, hh);

                R0C1_L = R0C1; R0C0 = R0C2;
                R1C1_L = R1C1; R1C0 = R1C2;
                ll += 4; hl += 4; hh += 4; lh += 4;
                m0 += 8; m1 += 8;
            }

            R0C1 = _mm_load_ps(m0+4); R0C2 = _mm_load_ps(m0+8);
            R1C1 = _mm_load_ps(m1+4); R1C2 = _mm_load_ps(m1+8);
            H_predict_with_unpack_kernel(R0C1_L, R0C0, R0C1, R0C2, ll, hl);
            H_predict_with_unpack_kernel(R1C1_L, R1C0, R1C1, R1C2, lh, hh);

            ll += next_band_y; hl += next_band_y; lh += next_band_y; hh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y;
        }
    }
    #pragma omp barrier
}

static void H_update_unpacked(const TransformStepArguments * tsa)
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

    __m128 HL01_L, HL00, HL01, LL00, HH01_L, HH00, HH01, LH00;

    if(band_start_x == 0 && band_end_x == band_size_x){
        for(size_t y = band_start_y; y < band_end_y; ++y){
            // first column
            HL01_L = _mm_loadr_ps(hl); HL00 = _mm_load_ps(hl); HL01 = _mm_load_ps(hl+4);
            HH01_L = _mm_loadr_ps(hh); HH00 = _mm_load_ps(hh); HH01 = _mm_load_ps(hh+4);
            LL00 = _mm_load_ps(ll); LH00 = _mm_load_ps(lh);

            H_update_unpacked_kernel(HL01_L, LL00, HL00, HL01, ll);
            H_update_unpacked_kernel(HH01_L, LH00, HH00, HH01, lh);

            ll += 4; hl += 4; lh += 4; hh += 4;

            HL01_L = HL00; HL00 = HL01;
            HH01_L = HH00; HH00 = HH01;

            for(size_t x = band_start_x + 4; x < band_end_x-4; x += 4){ // main area
                HL01 = _mm_load_ps(hl+4);
                HH01 = _mm_load_ps(hh+4);
                LL00 = _mm_load_ps(ll); LH00 = _mm_load_ps(lh);

                H_update_unpacked_kernel(HL01_L, LL00, HL00, HL01, ll);
                H_update_unpacked_kernel(HH01_L, LH00, HH00, HH01, lh);

                ll += 4; hl += 4; lh += 4; hh += 4;

                HL01_L = HL00; HL00 = HL01;
                HH01_L = HH00; HH00 = HH01;
            }

            // last column
            HL01 = _mm_loadr_ps(hl);
            HL01 = rotate_right(HL01);
            HH01 = _mm_loadr_ps(hh);
            HH01 = rotate_right(HH01);
            LL00 = _mm_load_ps(ll); LH00 = _mm_load_ps(lh);

            H_update_unpacked_kernel(HL01_L, LL00, HL00, HL01, ll);
            H_update_unpacked_kernel(HH01_L, LH00, HH00, HH01, lh);

            ll += 4; hl += 4; lh += 4; hh += 4;

            ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        }
    } else if(band_start_x == 0){
        for(size_t y = band_start_y; y < band_end_y; ++y){
            // first column
            HL01_L = _mm_loadr_ps(hl); HL00 = _mm_load_ps(hl); HL01 = _mm_load_ps(hl+4);
            HH01_L = _mm_loadr_ps(hh); HH00 = _mm_load_ps(hh); HH01 = _mm_load_ps(hh+4);
            LL00 = _mm_load_ps(ll); LH00 = _mm_load_ps(lh);

            H_update_unpacked_kernel(HL01_L, LL00, HL00, HL01, ll);
            H_update_unpacked_kernel(HH01_L, LH00, HH00, HH01, lh);

            ll += 4; hl += 4; lh += 4; hh += 4;

            HL01_L = HL00; HL00 = HL01;
            HH01_L = HH00; HH00 = HH01;

            for(size_t x = band_start_x + 4; x < band_end_x-4; x += 4){ // main area
                HL01 = _mm_load_ps(hl+4);
                HH01 = _mm_load_ps(hh+4);
                LL00 = _mm_load_ps(ll); LH00 = _mm_load_ps(lh);

                H_update_unpacked_kernel(HL01_L, LL00, HL00, HL01, ll);
                H_update_unpacked_kernel(HH01_L, LH00, HH00, HH01, lh);

                ll += 4; hl += 4; lh += 4; hh += 4;

                HL01_L = HL00; HL00 = HL01;
                HH01_L = HH00; HH00 = HH01;
            }

            // last column
            HL01 = _mm_load_ps(hl+4);
            HH01 = _mm_load_ps(hh+4);
            LL00 = _mm_load_ps(ll); LH00 = _mm_load_ps(lh);

            H_update_unpacked_kernel(HL01_L, LL00, HL00, HL01, ll);
            H_update_unpacked_kernel(HH01_L, LH00, HH00, HH01, lh);

            ll += 4; hl += 4; lh += 4; hh += 4;

            ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        }
    } else if(band_end_x == band_size_x){
        for(size_t y = band_start_y; y < band_end_y; ++y){
            // first column
            HL01_L = _mm_load_ps(hl-4); HL00 = _mm_load_ps(hl);
            HH01_L = _mm_load_ps(hh-4); HH00 = _mm_load_ps(hh);

            for(size_t x = band_start_x; x < band_end_x-4; x += 4){ // main area
                HL01 = _mm_load_ps(hl+4);
                HH01 = _mm_load_ps(hh+4);
                LL00 = _mm_load_ps(ll); LH00 = _mm_load_ps(lh);

                H_update_unpacked_kernel(HL01_L, LL00, HL00, HL01, ll);
                H_update_unpacked_kernel(HH01_L, LH00, HH00, HH01, lh);

                ll += 4; hl += 4; lh += 4; hh += 4;

                HL01_L = HL00; HL00 = HL01;
                HH01_L = HH00; HH00 = HH01;
            }

            // last column
            HL01 = _mm_loadr_ps(hl);
            HL01 = rotate_right(HL01);
            HH01 = _mm_loadr_ps(hh);
            HH01 = rotate_right(HH01);
            LL00 = _mm_load_ps(ll); LH00 = _mm_load_ps(lh);

            H_update_unpacked_kernel(HL01_L, LL00, HL00, HL01, ll);
            H_update_unpacked_kernel(HH01_L, LH00, HH00, HH01, lh);

            ll += 4; hl += 4; lh += 4; hh += 4;

            ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        }
    } else {
        for(size_t y = band_start_y; y < band_end_y; ++y){
            // first column
            HL01_L = _mm_load_ps(hl-4); HL00 = _mm_load_ps(hl);
            HH01_L = _mm_load_ps(hh-4); HH00 = _mm_load_ps(hh);

            for(size_t x = band_start_x; x < band_end_x-4; x += 4){ // main area
                HL01 = _mm_load_ps(hl+4);
                HH01 = _mm_load_ps(hh+4);
                LL00 = _mm_load_ps(ll); LH00 = _mm_load_ps(lh);

                H_update_unpacked_kernel(HL01_L, LL00, HL00, HL01, ll);
                H_update_unpacked_kernel(HH01_L, LH00, HH00, HH01, lh);

                ll += 4; hl += 4; lh += 4; hh += 4;

                HL01_L = HL00; HL00 = HL01;
                HH01_L = HH00; HH00 = HH01;
            }

            // last column
            HL01 = _mm_load_ps(hl+4);
            HH01 = _mm_load_ps(hh+4);
            LL00 = _mm_load_ps(ll); LH00 = _mm_load_ps(lh);

            H_update_unpacked_kernel(HL01_L, LL00, HL00, HL01, ll);
            H_update_unpacked_kernel(HH01_L, LH00, HH00, HH01, lh);

            ll += 4; hl += 4; lh += 4; hh += 4;

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

    if(band_start_y == 0 && band_end_y == band_size_y){
        for(size_t x = band_start_x; x < band_end_x; x += 4){
            V_predict_unpacked_kernel(ll+band_stride_y, ll, lh, ll+band_stride_y, ll+2*band_stride_y);
            V_predict_unpacked_kernel(hl+band_stride_y, hl, hh, hl+band_stride_y, hl+2*band_stride_y);

            ll += 4; hl += 4; lh += 4; hh += 4;
        }
        ll += next_y; hl += next_y; lh += next_y; hh += next_y;

        for(size_t y = band_start_y+1; y < band_end_y - 2; ++y){
            for(size_t x = band_start_x; x < band_end_x;  x += 4){
                V_predict_unpacked_kernel(ll-band_stride_y, ll, lh, ll+band_stride_y, ll+2*band_stride_y);
                V_predict_unpacked_kernel(hl-band_stride_y, hl, hh, hl+band_stride_y, hl+2*band_stride_y);

                ll += 4; hl += 4; lh += 4; hh += 4;
            }
            ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        }

        for(size_t x = band_start_x; x < band_end_x;  x += 4){
            V_predict_unpacked_kernel(ll-band_stride_y, ll, lh, ll+band_stride_y, ll+band_stride_y);
            V_predict_unpacked_kernel(hl-band_stride_y, hl, hh, hl+band_stride_y, hl+band_stride_y);

            ll += 4; hl += 4; lh += 4; hh += 4;
        }
        ll += next_y; hl += next_y; lh += next_y; hh += next_y;

        for(size_t x = band_start_x; x < band_end_x;  x += 4){
            V_predict_unpacked_kernel(ll-band_stride_y, ll, lh, ll, ll-band_stride_y);
            V_predict_unpacked_kernel(hl-band_stride_y, hl, hh, hl, hl-band_stride_y);

            ll += 4; hl += 4; lh += 4; hh += 4;
        }
    } else if(band_start_y == 0) {
        for(size_t x = band_start_x; x < band_end_x;  x += 4){
            V_predict_unpacked_kernel(ll+band_stride_y, ll, lh, ll+band_stride_y, ll+2*band_stride_y);
            V_predict_unpacked_kernel(hl+band_stride_y, hl, hh, hl+band_stride_y, hl+2*band_stride_y);

            ll += 4; hl += 4; lh += 4; hh += 4;
        }
        ll += next_y; hl += next_y; lh += next_y; hh += next_y;

        for(size_t y = band_start_y+1; y < band_end_y; ++y){
            for(size_t x = band_start_x; x < band_end_x; x += 4){
                V_predict_unpacked_kernel(ll-band_stride_y, ll, lh, ll+band_stride_y, ll+2*band_stride_y);
                V_predict_unpacked_kernel(hl-band_stride_y, hl, hh, hl+band_stride_y, hl+2*band_stride_y);

                ll += 4; hl += 4; lh += 4; hh += 4;
            }
            ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        }
    } else if(band_end_y == band_size_y){
        if(band_start_y < band_size_y-1){
            for(size_t y = band_start_y; y < band_end_y - 2; ++y){
                for(size_t x = band_start_x; x < band_end_x;  x += 4){
                    V_predict_unpacked_kernel(ll-band_stride_y, ll, lh, ll+band_stride_y, ll+2*band_stride_y);
                    V_predict_unpacked_kernel(hl-band_stride_y, hl, hh, hl+band_stride_y, hl+2*band_stride_y);

                    ll += 4; hl += 4; lh += 4; hh += 4;
                }
                ll += next_y; hl += next_y; lh += next_y; hh += next_y;
            }


            for(size_t x = band_start_x; x < band_end_x; x += 4){
                V_predict_unpacked_kernel(ll-band_stride_y, ll, lh, ll+band_stride_y, ll+band_stride_y);
                V_predict_unpacked_kernel(hl-band_stride_y, hl, hh, hl+band_stride_y, hl+band_stride_y);

                ll += 4; hl += 4; lh += 4; hh += 4;
            }
            ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        }

        for(size_t x = band_start_x; x < band_end_x;  x += 4){
            V_predict_unpacked_kernel(ll-band_stride_y, ll, lh, ll, ll-band_stride_y);
            V_predict_unpacked_kernel(hl-band_stride_y, hl, hh, hl, hl-band_stride_y);

            ll += 4; hl += 4; lh += 4; hh += 4;
        }
    } else if(band_end_y == band_size_y-1){
        for(size_t y = band_start_y; y < band_end_y - 2; ++y){
            for(size_t x = band_start_x; x < band_end_x; x += 4){
                V_predict_unpacked_kernel(ll-band_stride_y, ll, lh, ll+band_stride_y, ll+2*band_stride_y);
                V_predict_unpacked_kernel(hl-band_stride_y, hl, hh, hl+band_stride_y, hl+2*band_stride_y);

                ll += 4; hl += 4; lh += 4; hh += 4;
            }
            ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        }

        for(size_t x = band_start_x; x < band_end_x; x += 4){
            V_predict_unpacked_kernel(ll-band_stride_y, ll, lh, ll+band_stride_y, ll+band_stride_y);
            V_predict_unpacked_kernel(hl-band_stride_y, hl, hh, hl+band_stride_y, hl+band_stride_y);

            ll += 4; hl += 4; lh += 4; hh += 4;
        }
    } else {
        for(size_t y = band_start_y; y < band_end_y; ++y){
            for(size_t x = band_start_x; x < band_end_x;  x += 4){
                V_predict_unpacked_kernel(ll-band_stride_y, ll, lh, ll+band_stride_y, ll+2*band_stride_y);
                V_predict_unpacked_kernel(hl-band_stride_y, hl, hh, hl+band_stride_y, hl+2*band_stride_y);

                ll += 4; hl += 4; lh += 4; hh += 4;
            }
            ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        }
    }
    #pragma omp barrier
}

static void V_update_unpacked(const TransformStepArguments * tsa)
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

    if(band_start_y == 0 && band_end_y == band_size_y){
        //first row
        for(size_t x = band_start_x; x < band_end_x; x += 4){
            V_update_unpacked_kernel(lh+band_stride_y, lh, ll, lh, lh+band_stride_y);
            V_update_unpacked_kernel(hh+band_stride_y, hh, hl, hh, hh+band_stride_y);

            ll += 4; hl += 4; lh += 4; hh += 4;
        }
        ll += next_y; hl += next_y; lh += next_y; hh += next_y;

        //second row
        for(size_t x = band_start_x; x < band_end_x; x += 4){
            V_update_unpacked_kernel(lh-band_stride_y, lh-band_stride_y, ll, lh, lh+band_stride_y);
            V_update_unpacked_kernel(hh-band_stride_y, hh-band_stride_y, hl, hh, hh+band_stride_y);

            ll += 4; hl += 4; lh += 4; hh += 4;
        }
        ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        //main area
        for(size_t y = band_start_y + 2; y < band_end_y - 1; ++y){
            for(size_t x = band_start_x; x < band_end_x; x += 4){
                V_update_unpacked_kernel(lh-2*band_stride_y, lh-band_stride_y, ll, lh, lh+band_stride_y);
                V_update_unpacked_kernel(hh-2*band_stride_y, hh-band_stride_y, hl, hh, hh+band_stride_y);

                ll += 4; hl += 4; lh += 4; hh += 4;
            }
            ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        }

        //last row
        for(size_t x = band_start_x; x < band_end_x; x += 4){
            V_update_unpacked_kernel(lh-2*band_stride_y, lh-band_stride_y, ll, lh, lh-band_stride_y);
            V_update_unpacked_kernel(hh-2*band_stride_y, hh-band_stride_y, hl, hh, hh-band_stride_y);

            ll += 4; hl += 4; lh += 4; hh += 4;
        }
    } else if(band_start_y == 0) {
        //first row
        for(size_t x = band_start_x; x < band_end_x; x += 4){
            V_update_unpacked_kernel(lh+band_stride_y, lh, ll, lh, lh+band_stride_y);
            V_update_unpacked_kernel(hh+band_stride_y, hh, hl, hh, hh+band_stride_y);

            ll += 4; hl += 4; lh += 4; hh += 4;
        }
        ll += next_y; hl += next_y; lh += next_y; hh += next_y;

        //second row
        if(band_end_y > 1){
            for(size_t x = band_start_x; x < band_end_x; x += 4){
                V_update_unpacked_kernel(lh-band_stride_y, lh-band_stride_y, ll, lh, lh+band_stride_y);
                V_update_unpacked_kernel(hh-band_stride_y, hh-band_stride_y, hl, hh, hh+band_stride_y);

                ll += 4; hl += 4; lh += 4; hh += 4;
            }
            ll += next_y; hl += next_y; lh += next_y; hh += next_y;
            //main area
            for(size_t y = band_start_y + 2; y < band_end_y; ++y){
                for(size_t x = band_start_x; x < band_end_x; x += 4){
                    V_update_unpacked_kernel(lh-2*band_stride_y, lh-band_stride_y, ll, lh, lh+band_stride_y);
                    V_update_unpacked_kernel(hh-2*band_stride_y, hh-band_stride_y, hl, hh, hh+band_stride_y);

                    ll += 4; hl += 4; lh += 4; hh += 4;
                }
                ll += next_y; hl += next_y; lh += next_y; hh += next_y;
            }
        }
    } else if(band_start_y == 1){
        for(size_t x = band_start_x; x < band_end_x; x += 4){
            V_update_unpacked_kernel(lh-band_stride_y, lh-band_stride_y, ll, lh, lh+band_stride_y);
            V_update_unpacked_kernel(hh-band_stride_y, hh-band_stride_y, hl, hh, hh+band_stride_y);

            ll += 4; hl += 4; lh += 4; hh += 4;
        }
        ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        //main area
        for(size_t y = band_start_y + 2; y < band_end_y; ++y){
            for(size_t x = band_start_x; x < band_end_x; x += 4){
                V_update_unpacked_kernel(lh-2*band_stride_y, lh-band_stride_y, ll, lh, lh+band_stride_y);
                V_update_unpacked_kernel(hh-2*band_stride_y, hh-band_stride_y, hl, hh, hh+band_stride_y);

                ll += 4; hl += 4; lh += 4; hh += 4;
            }
            ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        }
    } else if(band_end_y == band_size_y){
        //main area
        for(size_t y = band_start_y; y < band_end_y - 1; ++y){
            for(size_t x = band_start_x; x < band_end_x; x += 4){
                V_update_unpacked_kernel(lh-2*band_stride_y, lh-band_stride_y, ll, lh, lh+band_stride_y);
                V_update_unpacked_kernel(hh-2*band_stride_y, hh-band_stride_y, hl, hh, hh+band_stride_y);

                ll += 4; hl += 4; lh += 4; hh += 4;
            }
            ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        }

        //last row
        for(size_t x = band_start_x; x < band_end_x; x += 4){
            V_update_unpacked_kernel(lh-2*band_stride_y, lh-band_stride_y, ll, lh, lh-band_stride_y);
            V_update_unpacked_kernel(hh-2*band_stride_y, hh-band_stride_y, hl, hh, hh-band_stride_y);

            ll += 4; hl += 4; lh += 4; hh += 4;
        }
    } else {
        for(size_t y = band_start_y; y < band_end_y; ++y){
            for(size_t x = band_start_x; x < band_end_x; x += 4){
                V_update_unpacked_kernel(lh-2*band_stride_y, lh-band_stride_y, ll, lh, lh+band_stride_y);
                V_update_unpacked_kernel(hh-2*band_stride_y, hh-band_stride_y, hl, hh, hh+band_stride_y);

                ll += 4; hl += 4; lh += 4; hh += 4;
            }
            ll += next_y; hl += next_y; lh += next_y; hh += next_y;
        }
    }
}

TRANSFORM_STEP_FUNC NO_TREE_VECTORIZE dd137_separable_lifting_amd64_sse(size_t step)
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

void NO_TREE_VECTORIZE dd137_separable_lifting_amd64_sse_transform_tile(const TransformStepArguments * tsa)
{
    H_predict_with_unpack(tsa);
    V_predict_unpacked(tsa);
    H_update_unpacked(tsa);
    V_update_unpacked(tsa);
}
#endif
