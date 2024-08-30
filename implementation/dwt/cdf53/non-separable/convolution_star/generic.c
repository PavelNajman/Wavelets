#include <assert.h>

#include <common.h>

#ifdef _OPENMP
    #include <omp.h>
#else
    #define omp_get_thread_num() 0
    #define omp_get_max_threads() 1
    #define omp_get_num_threads() 1
    #define omp_set_num_threads(num_threads)
#endif

#define LOCA_LL(y, x) (*(float *)(y + x))
#define LOCA_HL(y, x) (*(float *)(y + x + 1))
#define LOCA_LH(y, x) (*(float *)(y + x))
#define LOCA_HH(y, x) (*(float *)(y + x + 1))

#define LOCA_TMP(y, x) (*(float *)(y + x))

static void NO_TREE_VECTORIZE cdf53_non_separable_convolution_star_generic_all(const TransformStepArguments * tsa)
{
    float alpha     = -0.5f;
    float alpha2    = +0.25f;
    float alpha3    = -0.125f;
    float alpha4    = +0.0625f;
    float beta      = +0.875f;
    float gamma     = -0.4375f;
    float delta     = +0.21875f;
    float epsilon   = +0.765625f;

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

    const size_t tmpBandsMargin = 1;//tsa->tmp.margin;
    const size_t tmpBandsStrideY = bandStrideY + 2;//tsa->tmp.stride_y;
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
    const size_t nextTmpBandY = tmpBandsStrideY - (bandEndX - bandStartX);
    const size_t nextBandYM = tmpBandsStrideY - (size_t)(bandEndXM - bandStartXM);

    const size_t bandStartXMargin = bandStartX + tmpBandsMargin;


    // T^H_{P0}, T^V_{P0}
    {
        for (long ly = tileStartYM; ly < tileEndYM; ly += 2) {
            size_t lyL = mirr((long)ly, tileSizeY);
            size_t lyH = mirr((long)ly + 1, tileSizeY);
            float *tileYL = tile + lyL * tileStrideY;
            float *tileYH = tile + lyH * tileStrideY;
            for (long lx = tileStartXM; lx < tileEndXM; lx += 2) {
                size_t lxL = mirr((long)lx, tileSizeX);
                size_t lxH = mirr((long)lx + 1, tileSizeX) - 1;

                float mLLYX = LOCA_LL(tileYL, lxL);
                float mHLYX = LOCA_HL(tileYL, lxH);
                float mLHYX = LOCA_LH(tileYH, lxL);
                float mHHYX = LOCA_HH(tileYH, lxH);

                // T^H_{P0}:
                // LL = 1(LL) + 0(HL) + 0(LH) + 0(HH)
                (*tmpLL) = mLLYX;
                // HL = P0(LL) + 1(HL) + 0(LH) + 0(HH)
                (*tmpHL) = alpha * mLLYX + mHLYX;
                // LH = 0(LL) + 0(HL) + 1(LH) + 0(HH)

                // HH = 0(LL) + 0(HL) + P0(LH) + 0(HH)
                //(*tmpHH) = alpha * mLHYX + mHHYX;

                // T^V_{P0}:
                // LL = 1(LL) + 0(HL) + 0(LH) + 0(HH)

                // HL = 0(LL) + 1(HL) + 0(LH) + 0(HH)

                // LH = P0*(LL) + 0(HL) + 1(LH) + 0(HH)
                (*tmpLH) = alpha * mLLYX + mLHYX;
                // HH = 0(LL) + P0*(HL) + 0(LH) + 1(HH)
                (*tmpHH) = alpha * ((*tmpHL) + mLHYX) + mHHYX;

                tmpLL++;
                tmpHL++;
                tmpLH++;
                tmpHH++;
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

    long lx1 = +1;
    long lx_1 = -1;

    // N_{P1,U1}, S^H_{U0}, S^V_{U0}
    {
        for (size_t ly = bandStartY; ly < bandEndY; ly++) {
            for (size_t lx = bandStartX; lx < bandEndX; lx++) {

                // N_{P1,U1}
                // LL = V1*V1(LL) + V1*U1(HL) + U1*V1(LH) + U1*U1(HH)
                (*outLL) = epsilon * (*tmpLL)
                        + delta * (LOCA_TMP(tmpHL, lx_1) + (*bandLHY_1))
                        + alpha4 * LOCA_TMP(bandHHY_1, lx_1);

                // HL = V1*P1(LL) + V1*(HL) + U1*P1(LH) + U1*(HH)
                (*outHL) = gamma * LOCA_TMP(tmpLL, lx1)
                        + beta * (*tmpHL)
                        + alpha3 * LOCA_TMP(bandLHY_1, lx1)
                        + alpha2 * (*bandHHY_1);

                // LH = P1*V1(LL) + P1*U1(HL) + V1(LH) + U1(HH)
                (*outLH) = gamma * (*bandLLY1)
                        + alpha3 * LOCA_TMP(bandHLY1, lx_1)
                        + beta * (*tmpLH)
                        + alpha2 * LOCA_TMP(tmpHH, lx_1);

                // HH = P1*P1(LL) + P1*(HL) + P1(LH) + 1(HH)
                (*outHH) = alpha2 * LOCA_TMP(bandLLY1, lx1)
                        + alpha * ((*bandHLY1) + LOCA_TMP(tmpLH, lx1)) + (*tmpHH);

                // S^H_{U0}
                // LL = 1(LL) + U0(HL) + 0(LH) + 0(HH)
                //(*outLL) += alpha2 * (*outHL);
                // HL = 0(LL) + 1(HL) + 0(LH) + 0(HH)

                // LH = 0(LL) + 0(HL) + 1(LH) + U0(HH)
                (*outLH) += alpha2 * (*outHH);
                // HH = 0(LL) + 0(HL) + 0(LH) +  1(HH)


                // S^V_{U0}
                // LL = 1(LL) + 0(HL) + U0*(LH) + 0(HH)
                (*outLL) += alpha2 * ((*outHL) + (*outLH));
                // HL = 0(LL) + 1(HL) + 0(LH) + U0*(HH)
                (*outHL) += alpha2 * (*outHH);
                // LH = 0(LL) + 0(HL) + 1(LH) + 0(HH)

                // HH = 0(LL) + 0(HL) + 0(LH) + 1(HH)

                outLL++;
                outHL++;
                outLH++;
                outHH++;

                tmpLL++;
                tmpHL++;
                tmpLH++;
                tmpHH++;

                bandLLY1++;
                bandHLY1++;
                bandLHY_1++;
                bandHHY_1++;
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

TRANSFORM_STEP_FUNC NO_TREE_VECTORIZE cdf53_non_separable_convolution_star_generic(size_t step)
{
    (void)(step);
    return cdf53_non_separable_convolution_star_generic_all;
}

void NO_TREE_VECTORIZE cdf53_non_separable_convolution_star_generic_transform_tile(const TransformStepArguments * tsa)
{
    cdf53_non_separable_convolution_star_generic_all(tsa);
}
