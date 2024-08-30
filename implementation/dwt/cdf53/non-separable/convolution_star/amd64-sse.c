#include <assert.h>

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

static void NO_TREE_VECTORIZE cdf53_non_separable_convolution_star_amd64_sse_all(const TransformStepArguments * tsa)
{
    __m128 alpha     = _mm_set1_ps(-0.5f);
    __m128 alpha2    = _mm_set1_ps(+0.25f);
    __m128 alpha3    = _mm_set1_ps(-0.125f);
    __m128 alpha4    = _mm_set1_ps(+0.0625f);
    __m128 beta      = _mm_set1_ps(+0.875f);
    __m128 gamma     = _mm_set1_ps(-0.4375f);
    __m128 delta     = _mm_set1_ps(+0.21875f);
    __m128 epsilon   = _mm_set1_ps(+0.765625f);

    float *tile = tsa->tile.data;

    const size_t tid = (size_t)omp_get_thread_num();

    const size_t bandStrideY = tsa->tile_bands.stride_y;
    //const size_t bandSizeX = tsa->tile_bands.size_x;
    //const size_t bandSizeY = tsa->tile_bands.size_y;

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

    // 4 for SSE
    const size_t tmpBandsMargin = 4;//tsa->tmp.margin;
    const size_t tmpBandsStrideY = bandStrideY + 8;//tsa->tmp.stride_y;
    assert(tsa->tmp.margin >= tmpBandsMargin);

    long tileStartYM = tileStartY == 0 ? (long)(tileStartY - 2 * tmpBandsMargin) : (long)tileStartY;
    long tileEndYM = tileEndY == tileSizeY ? (long)(tileEndY + 2 * tmpBandsMargin) : (long)tileEndY;
    long tileStartXM = tileStartX == 0 ? (long)(tileStartX - 2 * tmpBandsMargin) : (long)tileStartX;
    long tileEndXM = tileEndX == tileSizeX ? (long)(tileEndX + 2 * tmpBandsMargin) : (long)tileEndX;
    long bandStartXM = tileStartX == 0 ? (long)bandStartX : (long)(bandStartX + tmpBandsMargin);
    long bandEndXM = tileEndX == tileSizeX ? (long)(bandEndX + 2 * tmpBandsMargin) : (long)(bandEndX + tmpBandsMargin);
    long bandStartYM = tileStartY == 0 ? (long)bandStartY : (long)(bandStartY + tmpBandsMargin);
    //long bandEndYM = tileEndY == tileSizeY ? (long)(bandEndY + 2 * tmpBandsMargin) : (long)(bandEndY + tmpBandsMargin);

    float *tmpLL = tsa->tmp.LL + bandStartYM * (long)tmpBandsStrideY + bandStartXM;
    float *tmpHL = tsa->tmp.HL + bandStartYM * (long)tmpBandsStrideY + bandStartXM;
    float *tmpLH = tsa->tmp.LH + bandStartYM * (long)tmpBandsStrideY + bandStartXM;
    float *tmpHH = tsa->tmp.HH + bandStartYM * (long)tmpBandsStrideY + bandStartXM;

    const size_t nextBandY = bandStrideY - (bandEndX - bandStartX);
    const size_t nextTmpBandY =  nextBandY + 2 * tmpBandsMargin;
    const size_t nextBandYM = tmpBandsStrideY - (size_t)(bandEndXM - bandStartXM);

    const size_t bandStartXMargin = bandStartX + tmpBandsMargin;

    const size_t stepX = 8;

    const size_t tileSizeX_stepX = tileSizeX - stepX;

    // T^H_{P0}, T^V_{P0}
    {
        for (long ly = tileStartYM; ly < tileEndYM; ly += 2) {
            size_t lyL = mirr((long)ly, tileSizeY);
            size_t lyH = mirr((long)ly + 1, tileSizeY);

            float *tileYL = tile + lyL * tileStrideY;
            float *tileYH = tile + lyH * tileStrideY;

            for (long lx = tileStartXM; lx < tileEndXM; lx += (long)stepX) {

                __m128 mLLYX, mHLYX, mLHYX, mHHYX;
                if (lx < 0) {
                    mLLYX = reverse(load_packed_LX00(_mm_loadu_ps(LOCA(tileYL, 2)), _mm_loadu_ps(LOCA(tileYL, 6))));
                    mHLYX = reverse(load_packed_HX00(_mm_load_ps((tileYL)), _mm_load_ps(LOCA(tileYL, 4))));
                    mLHYX = reverse(load_packed_LX00(_mm_loadu_ps(LOCA(tileYH, 2)), _mm_loadu_ps(LOCA(tileYH, 6))));
                    mHHYX = reverse(load_packed_HX00(_mm_load_ps((tileYH)), _mm_load_ps(LOCA(tileYH, 4))));
                } else if (lx > (long)(tileSizeX_stepX)) {
                    mLLYX = reverse(load_packed_LX00(_mm_load_ps(LOCA(tileYL, tileSizeX_stepX)), _mm_load_ps(LOCA(tileYL, tileSizeX - 4))));
                    mHLYX = reverse(load_packed_HX00(_mm_loadu_ps(LOCA(tileYL, tileSizeX_stepX - 2)), _mm_loadu_ps(LOCA(tileYL, tileSizeX - 6))));
                    mLHYX = reverse(load_packed_LX00(_mm_load_ps(LOCA(tileYH, tileSizeX_stepX)), _mm_load_ps(LOCA(tileYH, tileSizeX - 4))));
                    mHHYX = reverse(load_packed_HX00(_mm_loadu_ps(LOCA(tileYH, tileSizeX_stepX - 2)), _mm_loadu_ps(LOCA(tileYH, tileSizeX - 6))));
                } else {
                    mLLYX = load_packed_LX00(_mm_load_ps(LOCA(tileYL, lx)), _mm_load_ps(LOCA(tileYL, lx + 4)));
                    mHLYX = load_packed_HX00(_mm_load_ps(LOCA(tileYL, lx)), _mm_load_ps(LOCA(tileYL, lx + 4)));
                    mLHYX = load_packed_LX00(_mm_load_ps(LOCA(tileYH, lx)), _mm_load_ps(LOCA(tileYH, lx + 4)));
                    mHHYX = load_packed_HX00(_mm_load_ps(LOCA(tileYH, lx)), _mm_load_ps(LOCA(tileYH, lx + 4)));
                }

                // T^H_{P0}:
                // LL = 1(LL) + 0(HL) + 0(LH) + 0(HH)
                _mm_store_ps(tmpLL, mLLYX);
                // HL = P0(LL) + 1(HL) + 0(LH) + 0(HH)
                mHLYX = mul_add(alpha, mLLYX, mHLYX);
                _mm_store_ps(tmpHL, mHLYX);
                // LH = 0(LL) + 0(HL) + 1(LH) + 0(HH)

                // HH = 0(LL) + 0(HL) + P0(LH) + 0(HH)

                // T^V_{P0}:
                // LL = 1(LL) + 0(HL) + 0(LH) + 0(HH)

                // HL = 0(LL) + 1(HL) + 0(LH) + 0(HH)

                // LH = P0*(LL) + 0(HL) + 1(LH) + 0(HH)
                _mm_store_ps(tmpLH, mul_add(alpha, mLLYX, mLHYX));
                // HH = 0(LL) + P0*(HL) + 0(LH) + 1(HH)
                _mm_store_ps(tmpHH, mul_add(alpha, _mm_add_ps(mHLYX, mLHYX), mHHYX));


                tmpLL += 4;
                tmpHL += 4;
                tmpLH += 4;
                tmpHH += 4;
            }

            tmpLL += nextBandYM;
            tmpHL += nextBandYM;
            tmpLH += nextBandYM;
            tmpHH += nextBandYM;
        }
    }

    // sync threads
    #pragma omp barrier

    tmpLL = tsa->tmp.LL + (bandStartY + tmpBandsMargin) * tmpBandsStrideY + bandStartXMargin;
    tmpHL = tsa->tmp.HL + (bandStartY + tmpBandsMargin) * tmpBandsStrideY + bandStartXMargin;
    tmpLH = tsa->tmp.LH + (bandStartY + tmpBandsMargin) * tmpBandsStrideY + bandStartXMargin;
    tmpHH = tsa->tmp.HH + (bandStartY + tmpBandsMargin) * tmpBandsStrideY + bandStartXMargin;

    float *bandLLY1 = tsa->tmp.LL + (bandStartY + 1 + tmpBandsMargin) * tmpBandsStrideY + bandStartXMargin;
    float *bandHLY1 = tsa->tmp.HL + (bandStartY + 1 + tmpBandsMargin) * tmpBandsStrideY + bandStartXMargin;
    float *bandLHY_1 = tsa->tmp.LH + (bandStartY - 1 + tmpBandsMargin) * tmpBandsStrideY + bandStartXMargin;
    float *bandHHY_1 = tsa->tmp.HH + (bandStartY - 1 + tmpBandsMargin) * tmpBandsStrideY + bandStartXMargin;

    // N_{P1,U1}, S^H_{U0}, S^V_{U0}
    {
        for (size_t ly = bandStartY; ly < bandEndY; ly++) {
            for (size_t lx = bandStartX; lx < bandEndX; lx += 4) {
                __m128 mLLYX = _mm_load_ps((tmpLL));
                __m128 mHLYX = _mm_load_ps((tmpHL));
                __m128 mLHYX = _mm_load_ps((tmpLH));
                __m128 mHHYX = _mm_load_ps((tmpHH));

                __m128 mLLY1X = _mm_load_ps((bandLLY1));
                __m128 mHLY1X = _mm_load_ps((bandHLY1));

                __m128 mLHY_1X = _mm_load_ps((bandLHY_1));
                __m128 mHHY_1X = _mm_load_ps((bandHHY_1));

                __m128 mLLYX1 = load_unpacked_XX01(mLLYX, _mm_load_ps(tmpLL + 4));

                __m128 mLHYX1 = load_unpacked_XX01(mLHYX, _mm_load_ps(tmpLH + 4));

                __m128 mLLY1X1 = load_unpacked_XX01(mLLY1X, _mm_load_ps(bandLLY1 + 4));

                __m128 mLHY_1X1 = load_unpacked_XX01(mLHY_1X, _mm_load_ps(bandLHY_1 + 4));

                __m128 mHLYX_1 = load_unpacked_XX01L(mHLYX, _mm_load_ps(tmpHL - 4));

                __m128 mHHYX_1 = load_unpacked_XX01L(mHHYX, _mm_load_ps(tmpHH - 4));

                __m128 mHLY1X_1 = load_unpacked_XX01L(mHLY1X, _mm_load_ps(bandHLY1 - 4));

                __m128 mHHY_1X_1 = load_unpacked_XX01L(mHHY_1X, _mm_load_ps(bandHHY_1 - 4));

                // N_{P1,U1}
                // LL = V1*V1(LL) + V1*U1(HL) + U1*V1(LH) + U1*U1(HH)
                __m128 mOutLL;
                mOutLL = _mm_mul_ps(epsilon, mLLYX);
                mOutLL = mul_add(delta, _mm_add_ps(mHLYX_1, mLHY_1X), mOutLL);
                mOutLL = mul_add(alpha4, mHHY_1X_1, mOutLL);

                // HL = V1*P1(LL) + V1*(HL) + U1*P1(LH) + U1*(HH)
                __m128 mOutHL;
                mOutHL = _mm_mul_ps(gamma, mLLYX1);
                mOutHL = mul_add(beta, mHLYX, mOutHL);
                mOutHL = mul_add(alpha3, mLHY_1X1, mOutHL);
                mOutHL = mul_add(alpha2, mHHY_1X, mOutHL);

                // LH = P1*V1(LL) + P1*U1(HL) + V1(LH) + U1(HH)
                __m128 mOutLH;
                mOutLH = _mm_mul_ps(gamma, mLLY1X);
                mOutLH = mul_add(alpha3, mHLY1X_1, mOutLH);
                mOutLH = mul_add(beta, mLHYX, mOutLH);
                mOutLH = mul_add(alpha2, mHHYX_1, mOutLH);

                // HH = P1*P1(LL) + P1*(HL) + P1(LH) + 1(HH)
                __m128 mOutHH;
                mOutHH = _mm_mul_ps(alpha2, mLLY1X1);
                mOutHH = mul_add(alpha, _mm_add_ps(mHLY1X, mLHYX1), mOutHH);
                mOutHH = _mm_add_ps(mHHYX, mOutHH);

                // S^H_{U0}
                // LL = 1(LL) + U0(HL) + 0(LH) + 0(HH)
                //mOutLL = mul_add(alpha2, mOutHL, mOutLL);
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

                tmpLL += 4;
                tmpHL += 4;
                tmpLH += 4;
                tmpHH += 4;

                bandLLY1 += 4;
                bandHLY1 += 4;
                bandLHY_1 += 4;
                bandHHY_1 += 4;
            }

            outLL += nextBandY;
            outHL += nextBandY;
            outLH += nextBandY;
            outHH += nextBandY;

            tmpLL += nextTmpBandY;
            tmpHL += nextTmpBandY;
            tmpLH += nextTmpBandY;
            tmpHH += nextTmpBandY;

            bandLLY1 += nextTmpBandY;
            bandHLY1 += nextTmpBandY;
            bandLHY_1 += nextTmpBandY;
            bandHHY_1 += nextTmpBandY;
        }
    }
    //#pragma omp barrier
}

TRANSFORM_STEP_FUNC NO_TREE_VECTORIZE cdf53_non_separable_convolution_star_amd64_sse(size_t step)
{
    (void)(step);
    return cdf53_non_separable_convolution_star_amd64_sse_all;
}

void NO_TREE_VECTORIZE cdf53_non_separable_convolution_star_amd64_sse_transform_tile(const TransformStepArguments * tsa)
{
    cdf53_non_separable_convolution_star_amd64_sse_all(tsa);
}
#endif
