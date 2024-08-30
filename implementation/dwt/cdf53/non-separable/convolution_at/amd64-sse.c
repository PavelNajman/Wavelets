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

#define LOCA(y, x) ((float *)(y + x))

static void NO_TREE_VECTORIZE cdf53_non_separable_convolution_at_amd64_sse_all(const TransformStepArguments * tsa)
{
    __m128 alpha     = _mm_set1_ps(-0.5f);
    __m128 alpha2    = _mm_set1_ps(+0.25f);
    __m128 beta      = _mm_set1_ps(+0.875f);
    __m128 alpha3    = _mm_set1_ps(-0.125f);
    __m128 gamma     = _mm_set1_ps(+0.765625f);
    __m128 delta     = _mm_set1_ps(-0.109375f);
    __m128 epsilon   = _mm_set1_ps(+0.015625f);
    __m128 zeta      = _mm_set1_ps(+0.21875f);
    __m128 eta       = _mm_set1_ps(-0.03125f);
    __m128 theta     = _mm_set1_ps(-0.4375f);
    __m128 alpha4    = _mm_set1_ps(+0.0625f);

    float *tile = tsa->tile.data;

    const size_t tid = (size_t)omp_get_thread_num();

    const size_t bandStrideY = tsa->tile_bands.stride_y;

    const size_t tileSizeX = tsa->tile.size_x;
    const size_t tileSizeY = tsa->tile.size_y;
    const size_t tileStrideY = tsa->tile.stride_y;

    const size_t bandStartX = tsa->threading_info->band_start_x[tid];
    const size_t bandStartY = tsa->threading_info->band_start_y[tid];
    const size_t bandEndX = tsa->threading_info->band_end_x[tid];
    const size_t bandEndY = tsa->threading_info->band_end_y[tid];

    const size_t tileStartX = bandStartX << 1; // * 2
    const size_t tileStartY = bandStartY << 1;
    const size_t tileEndX = bandEndX << 1;
    const size_t tileEndY = bandEndY << 1;

    float *outLL = tsa->tile_bands.LL + bandStartY * bandStrideY + bandStartX;
    float *outHL = tsa->tile_bands.HL + bandStartY * bandStrideY + bandStartX;
    float *outLH = tsa->tile_bands.LH + bandStartY * bandStrideY + bandStartX;
    float *outHH = tsa->tile_bands.HH + bandStartY * bandStrideY + bandStartX;

    const size_t nextBandY = bandStrideY - (bandEndX - bandStartX);

    float *tileYL = tile + tileStartY * tileStrideY;
    float *tileYH = tile + (tileStartY + 1) * tileStrideY;

    const size_t tileStrideY2 = 2 * tileStrideY;

    const size_t stepX = 8;

    // N_{P,U1}, S^H_{U0}, S^V_{U0}
    {
        for (size_t ly = tileStartY; ly < tileEndY; ly += 2) {
            size_t ly1L = mirr((long)ly + 2, tileSizeY);
            size_t ly_1L = mirr((long)ly - 2, tileSizeY);
            size_t ly_1H = mirr((long)ly - 1, tileSizeY);

            float *tileY1L = tile + ly1L * tileStrideY;
            float *tileY_1L = tile + ly_1L * tileStrideY;
            float *tileY_1H = tile + ly_1H * tileStrideY;

            for (size_t lx = tileStartX; lx < tileEndX; lx += stepX) {
                __m128 mLLYX = load_packed_LX00(_mm_load_ps(LOCA(tileYL, lx)), _mm_load_ps(LOCA(tileYL, lx + 4)));
                __m128 mHLYX = load_packed_HX00(_mm_load_ps(LOCA(tileYL, lx)), _mm_load_ps(LOCA(tileYL, lx + 4)));
                __m128 mLHYX = load_packed_LX00(_mm_load_ps(LOCA(tileYH, lx)), _mm_load_ps(LOCA(tileYH, lx + 4)));
                __m128 mHHYX = load_packed_HX00(_mm_load_ps(LOCA(tileYH, lx)), _mm_load_ps(LOCA(tileYH, lx + 4)));

                __m128 mLLY1X = load_packed_LX00(_mm_load_ps(LOCA(tileY1L, lx)), _mm_load_ps(LOCA(tileY1L, lx + 4)));
                __m128 mHLY1X = load_packed_HX00(_mm_load_ps(LOCA(tileY1L, lx)), _mm_load_ps(LOCA(tileY1L, lx + 4)));

                __m128 mLLY_1X = load_packed_LX00(_mm_load_ps(LOCA(tileY_1L, lx)), _mm_load_ps(LOCA(tileY_1L, lx + 4)));
                __m128 mHLY_1X = load_packed_HX00(_mm_load_ps(LOCA(tileY_1L, lx)), _mm_load_ps(LOCA(tileY_1L, lx + 4)));
                __m128 mLHY_1X = load_packed_LX00(_mm_load_ps(LOCA(tileY_1H, lx)), _mm_load_ps(LOCA(tileY_1H, lx + 4)));
                __m128 mHHY_1X = load_packed_HX00(_mm_load_ps(LOCA(tileY_1H, lx)), _mm_load_ps(LOCA(tileY_1H, lx + 4)));

                __m128 mLLYX1, mLHYX1, mLLY_1X1, mLHY_1X1, mLLY1X1;
                if (lx < tileSizeX - stepX) {
                    mLLYX1 = load_packed_LX01(mLLYX, _mm_load_ps(LOCA(tileYL, lx + 8)));
                    mLHYX1 = load_packed_LX01(mLHYX, _mm_load_ps(LOCA(tileYH, lx + 8)));

                    mLLY1X1 = load_packed_LX01(mLLY1X, _mm_load_ps(LOCA(tileY1L, lx + 8)));

                    mLLY_1X1 = load_packed_LX01(mLLY_1X, _mm_load_ps(LOCA(tileY_1L, lx + 8)));
                    mLHY_1X1 = load_packed_LX01(mLHY_1X, _mm_load_ps(LOCA(tileY_1H, lx + 8)));
                } else {
                    mLLYX1 = load_unpacked_LX01S(mLLYX);
                    mLHYX1 = load_unpacked_LX01S(mLHYX);

                    mLLY1X1 = load_unpacked_LX01S(mLLY1X);

                    mLLY_1X1 = load_unpacked_LX01S(mLLY_1X);
                    mLHY_1X1 = load_unpacked_LX01S(mLHY_1X);
                }

                __m128 mLLYX_1, mHLYX_1, mLHYX_1, mHHYX_1, mLLY_1X_1, mHLY_1X_1, mLHY_1X_1, mHHY_1X_1, mLLY1X_1, mHLY1X_1;
                if (lx > 0) {
                    mLLYX_1 = load_packed_LX01L(mLLYX, _mm_load_ps(LOCA(tileYL, lx - 4)));
                    mHLYX_1 = load_packed_HX01L(mHLYX, _mm_load_ps(LOCA(tileYL, lx - 4)));
                    mLHYX_1 = load_packed_LX01L(mLHYX, _mm_load_ps(LOCA(tileYH, lx - 4)));
                    mHHYX_1 = load_packed_HX01L(mHHYX, _mm_load_ps(LOCA(tileYH, lx - 4)));

                    mLLY1X_1 = load_packed_LX01L(mLLY1X, _mm_load_ps(LOCA(tileY1L, lx - 4)));
                    mHLY1X_1 = load_packed_HX01L(mHLY1X, _mm_load_ps(LOCA(tileY1L, lx - 4)));

                    mLLY_1X_1 = load_packed_LX01L(mLLY_1X, _mm_load_ps(LOCA(tileY_1L, lx - 4)));
                    mHLY_1X_1 = load_packed_HX01L(mHLY_1X, _mm_load_ps(LOCA(tileY_1L, lx - 4)));
                    mLHY_1X_1 = load_packed_LX01L(mLHY_1X, _mm_load_ps(LOCA(tileY_1H, lx - 4)));
                    mHHY_1X_1 = load_packed_HX01L(mHHY_1X, _mm_load_ps(LOCA(tileY_1H, lx - 4)));
                } else {
                    mLLYX_1 = load_unpacked_LX01LS(mLLYX);
                    mHLYX_1 = load_unpacked_HX01LS(mHLYX);
                    mLHYX_1 = load_unpacked_LX01LS(mLHYX);
                    mHHYX_1 = load_unpacked_HX01LS(mHHYX);

                    mLLY1X_1 = load_unpacked_LX01LS(mLLY1X);
                    mHLY1X_1 = load_unpacked_HX01LS(mHLY1X);

                    mLLY_1X_1 = load_unpacked_LX01LS(mLLY_1X);
                    mHLY_1X_1 = load_unpacked_HX01LS(mHLY_1X);
                    mLHY_1X_1 = load_unpacked_LX01LS(mLHY_1X);
                    mHHY_1X_1 = load_unpacked_HX01LS(mHHY_1X);
                }

                // N_{P,U1}
                // LL = V@*V@(LL) + V@*U1(HL) + U1*V@(LH) + U1*U1(HH)
                __m128 mOutLL;
                mOutLL = _mm_mul_ps(gamma, mLLYX);
                mOutLL = mul_add(delta, _mm_add_ps(mLLY_1X, mLLYX_1), mOutLL);
                mOutLL = mul_add(epsilon, mLLY_1X_1, mOutLL);
                mOutLL = mul_add(zeta, _mm_add_ps(mHLYX_1, mLHY_1X), mOutLL);
                mOutLL = mul_add(eta, _mm_add_ps(mHLY_1X_1, mLHY_1X_1), mOutLL);
                mOutLL = mul_add(alpha4, mHHY_1X_1, mOutLL);

                // HL = V@*P(LL) + V@*(HL) + U1*P(LH) + U1*(HH)
                __m128 mOutHL;
                mOutHL = _mm_mul_ps(theta, _mm_add_ps(mLLYX1, mLLYX));
                mOutHL = mul_add(alpha4, _mm_add_ps(mLLY_1X1, mLLY_1X), mOutHL);
                mOutHL = mul_add(beta, mHLYX, mOutHL);
                mOutHL = mul_add(alpha3, _mm_add_ps(_mm_add_ps(mLHY_1X, mLHY_1X1), mHLY_1X), mOutHL);
                mOutHL = mul_add(alpha2, mHHY_1X, mOutHL);

                // LH = P*V@(LL) + P*U1(HL) + V@(LH) + U1(HH)
                __m128 mOutLH;
                mOutLH = _mm_mul_ps(theta, _mm_add_ps(mLLY1X, mLLYX));
                mOutLH = mul_add(alpha4, _mm_add_ps(mLLY1X_1, mLLYX_1), mOutLH);
                mOutLH = mul_add(alpha3, _mm_add_ps(_mm_add_ps(mHLYX_1, mHLY1X_1), mLHYX_1), mOutLH);
                mOutLH = mul_add(beta, mLHYX, mOutLH);
                mOutLH = mul_add(alpha2, mHHYX_1, mOutLH);

                // HH = P*P(LL) + P*(HL) + P(LH) + 1(HH)
                __m128 mOutHH;
                mOutHH = _mm_mul_ps(alpha2, _mm_add_ps(_mm_add_ps(mLLY1X1, mLLYX1), _mm_add_ps(mLLY1X, mLLYX)));
                mOutHH = mul_add(alpha, _mm_add_ps(_mm_add_ps(mHLYX, mHLY1X), _mm_add_ps(mLHYX, mLHYX1)), mOutHH);
                mOutHH = _mm_add_ps(mHHYX, mOutHH);

                // S^H_{U0}
                // LL = 1(LL) + U0(HL) + 0(LH) + 0(HH)
                //mOutLL = mul_add(alpha2, mOutHL), mOutLL);
                // HL = 0(LL) + 1(HL) + 0(LH) + 0(HH)

                // LH = 0(LL) + 0(HL) + 1(LH) + U0(HH)
                mOutLH = mul_add(alpha2, mOutHH, mOutLH);
                // HH = 0(LL) + 0(HL) + 0(LH) +  1(HH)

                // S^V_{U0}
                // LL = 1(LL) + 0(HL) + U0*(LH) + 0(HH)
                mOutLL = mul_add(alpha2, _mm_add_ps(mOutLH, mOutHL), mOutLL);
                // HL = 0(LL) + 1(HL) + 0(LH) + U0*(HH)
                mOutHL = mul_add(alpha2, mOutHH, mOutHL);
                // LH = 0(LL) + 0(HL) + 1(LH) + 0(HH)

                // HH = 0(LL) + 0(HL) + 0(LH) + 1(HH)

                _mm_store_ps(outLL, mOutLL);
                _mm_store_ps(outHL, mOutHL);
                _mm_store_ps(outLH, mOutLH);
                _mm_store_ps(outHH, mOutHH);

                outLL += 4;
                outHL += 4;
                outLH += 4;
                outHH += 4;
            }

            tileYL += tileStrideY2;
            tileYH += tileStrideY2;

            outLL += nextBandY;
            outHL += nextBandY;
            outLH += nextBandY;
            outHH += nextBandY;
        }
    }
    //#pragma omp barrier
}

TRANSFORM_STEP_FUNC NO_TREE_VECTORIZE cdf53_non_separable_convolution_at_amd64_sse(size_t step)
{
    (void)(step);
    return cdf53_non_separable_convolution_at_amd64_sse_all;
}

void NO_TREE_VECTORIZE cdf53_non_separable_convolution_at_amd64_sse_transform_tile(const TransformStepArguments * tsa)
{
    cdf53_non_separable_convolution_at_amd64_sse_all(tsa);
}
#endif
