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

static void NO_TREE_VECTORIZE cdf97_non_separable_polyconvolution_generic_all(const TransformStepArguments * tsa)
{
    float alpha     = -1.58613f;
    float alpha2    = +2.51582f;
    float alpha22   = -0.0529801f;
    float beta      = +1.08403f;
    float alpha3    = +0.0840336f;
    float gamma     = +1.17513f;
    float delta     = +0.0910952f;
    float epsilon   = +0.00706164f;
    float zeta      = -0.0574322f;
    float eta       = -0.00445211f;
    float theta     = -1.71942f;
    float alpha4    = +0.00280689f;
    float alpha44   = -0.133289f;

    float alpha4P   = +0.196698f;
    float betaP     = +1.39158f;
    float alpha3P   = +0.391577f;
    float alpha2P   = +0.443507f;
    float alpha5P   = +0.779532f;
    float alphaP    = +0.882911f;
    float gammaP    = +1.22864f;
    float deltaP    = +0.617174f;
    float epsilonP  = +1.93649f;

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

    const size_t tmpBandsMargin = 2;//tsa->tmp.margin;
    const size_t tmpBandsStrideY = bandStrideY + 4;//tsa->tmp.stride_y;
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

    // N_{P,U1}, S^H_{U0}, S^V_{U0}
    {
        for (long ly = tileStartYM; ly < tileEndYM; ly += 2) {
            size_t lyL = mirr((long)ly, tileSizeY);
            size_t lyH = mirr((long)ly + 1, tileSizeY);
            size_t ly1L = mirr((long)ly + 2, tileSizeY);
            size_t ly_1L = mirr((long)ly - 2, tileSizeY);
            size_t ly_1H = mirr((long)ly - 1, tileSizeY);

            float *tileYL = tile + lyL * tileStrideY;
            float *tileYH = tile + lyH * tileStrideY;
            float *tileY1L = tile + ly1L * tileStrideY;
            float *tileY_1L = tile + ly_1L * tileStrideY;
            float *tileY_1H = tile + ly_1H * tileStrideY;

            for (long lx = tileStartXM; lx < tileEndXM; lx += 2) {
                size_t lxL = mirr((long)lx, tileSizeX);
                size_t lxH = mirr((long)lx + 1, tileSizeX) - 1;
                size_t lx1L = mirr((long)lx + 2, tileSizeX);
                size_t lx_1L = mirr((long)lx - 2, tileSizeX);
                size_t lx_1H = mirr((long)lx - 1, tileSizeX) - 1;

                float mLLYX = LOCA_LL(tileYL, lxL);
                float mHLYX = LOCA_HL(tileYL, lxH);
                float mLHYX = LOCA_LH(tileYH, lxL);
                float mHHYX = LOCA_HH(tileYH, lxH);

                float mLLY1X = LOCA_LL(tileY1L, lxL);
                float mHLY1X = LOCA_HL(tileY1L, lxH);

                float mLLY_1X = LOCA_LL(tileY_1L, lxL);
                float mHLY_1X = LOCA_HL(tileY_1L, lxH);
                float mLHY_1X = LOCA_LH(tileY_1H, lxL);
                float mHHY_1X = LOCA_HH(tileY_1H, lxH);

                float mLLYX1 = LOCA_LL(tileYL, lx1L);

                float mLHYX1 = LOCA_LH(tileYH, lx1L);

                float mLLY1X1 = LOCA_LL(tileY1L, lx1L);

                float mLLY_1X1 = LOCA_LL(tileY_1L, lx1L);

                float mLHY_1X1 = LOCA_LH(tileY_1H, lx1L);

                float mLLYX_1 = LOCA_LL(tileYL, lx_1L);
                float mHLYX_1 = LOCA_HL(tileYL, lx_1H);
                float mLHYX_1 = LOCA_LH(tileYH, lx_1L);
                float mHHYX_1 = LOCA_HH(tileYH, lx_1H);

                float mLLY1X_1 = LOCA_LL(tileY1L, lx_1L);
                float mHLY1X_1 = LOCA_HL(tileY1L, lx_1H);

                float mLLY_1X_1 = LOCA_LL(tileY_1L, lx_1L);
                float mHLY_1X_1 = LOCA_HL(tileY_1L, lx_1H);
                float mLHY_1X_1 = LOCA_LH(tileY_1H, lx_1L);
                float mHHY_1X_1 = LOCA_HH(tileY_1H, lx_1H);

                // N_{P,U1}
                // LL = V@*V@(LL) + V@*U1(HL) + U1*V@(LH) + U1*U1(HH)
                (*tmpLL) = gamma * mLLYX
                        + delta * (mLLY_1X + mLLYX_1)
                        + epsilon * mLLY_1X_1
                        + zeta * (mHLYX_1 + mLHY_1X)
                        + eta * (mHLY_1X_1 + mLHY_1X_1)
                        + alpha4 * mHHY_1X_1;

                // HL = V@*P(LL) + V@*(HL) + U1*P(LH) + U1*(HH)
                (*tmpHL) = theta * (mLLYX1 + mLLYX)
                        + alpha44 * (mLLY_1X1 + mLLY_1X)
                        + beta * mHLYX
                        + alpha3 * (mHLY_1X + mLHY_1X + mLHY_1X1)
                        + alpha22 * mHHY_1X;

                // LH = P*V@(LL) + P*U1(HL) + V@(LH) + U1(HH)
                (*tmpLH) = theta * (mLLY1X + mLLYX)
                        + alpha44 * (mLLY1X_1 + mLLYX_1)
                        + alpha3 * (mHLYX_1 + mHLY1X_1 + mLHYX_1)
                        + beta * mLHYX
                        + alpha22 * mHHYX_1;

                // HH = P*P(LL) + P*(HL) + P(LH) + 1(HH)
                (*tmpHH) = alpha2 * (mLLY1X1 + mLLYX1 + mLLY1X + mLLYX)
                        + alpha * (mHLYX + mHLY1X + mLHYX + mLHYX1)
                        + mHHYX;

                // S^H_{U0}
                // LL = 1(LL) + U0(HL) + 0(LH) + 0(HH)
                (*tmpLL) += alpha22 * (*tmpHL);
                // HL = 0(LL) + 1(HL) + 0(LH) + 0(HH)

                // LH = 0(LL) + 0(HL) + 1(LH) + U0(HH)
                (*tmpLH) += alpha22 * (*tmpHH);
                // HH = 0(LL) + 0(HL) + 0(LH) +  1(HH)


                // S^V_{U0}
                // LL = 1(LL) + 0(HL) + U0*(LH) + 0(HH)
                (*tmpLL) += alpha22 * (*tmpLH);
                // HL = 0(LL) + 1(HL) + 0(LH) + U0*(HH)
                (*tmpHL) += alpha22 * (*tmpHH);
                // LH = 0(LL) + 0(HL) + 1(LH) + 0(HH)

                // HH = 0(LL) + 0(HL) + 0(LH) + 1(HH)

                // T^H_{P0}:
                // LL = 1(LL) + 0(HL) + 0(LH) + 0(HH)

                // HL = P0(LL) + 1(HL) + 0(LH) + 0(HH)
                (*tmpHL) += alphaP * (*tmpLL);
                // LH = 0(LL) + 0(HL) + 1(LH) + 0(HH)

                // HH = 0(LL) + 0(HL) + P0(LH) + 0(HH)
                (*tmpHH) += alphaP * (*tmpLH);

                // T^V_{P0}:
                // LL = 1(LL) + 0(HL) + 0(LH) + 0(HH)

                // HL = 0(LL) + 1(HL) + 0(LH) + 0(HH)

                // LH = P0*(LL) + 0(HL) + 1(LH) + 0(HH)
                (*tmpLH) += alphaP * (*tmpLL);
                // HH = 0(LL) + P0*(HL) + 0(LH) + 1(HH)
                (*tmpHH) += alphaP * (*tmpHL);


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
                (*outLL) = epsilonP * (*tmpLL)
                        + deltaP * (LOCA_TMP(tmpHL, lx_1) + (*bandLHY_1))
                        + alpha4P * LOCA_TMP(bandHHY_1, lx_1);

                // HL = V1*P1(LL) + V1*(HL) + U1*P1(LH) + U1*(HH)
                (*outHL) = gammaP * LOCA_TMP(tmpLL, lx1)
                        + betaP * (*tmpHL)
                        + alpha3P * LOCA_TMP(bandLHY_1, lx1)
                        + alpha2P * (*bandHHY_1);

                // LH = P1*V1(LL) + P1*U1(HL) + V1(LH) + U1(HH)
                (*outLH) = gammaP * (*bandLLY1)
                        + alpha3P * LOCA_TMP(bandHLY1, lx_1)
                        + betaP * (*tmpLH)
                        + alpha2P * LOCA_TMP(tmpHH, lx_1);

                // HH = P1*P1(LL) + P1*(HL) + P1(LH) + 1(HH)
                (*outHH) = alpha5P * LOCA_TMP(bandLLY1, lx1)
                        + alphaP * ((*bandHLY1) + LOCA_TMP(tmpLH, lx1))
                        + (*tmpHH);

                // S^H_{U0}
                // LL = 1(LL) + U0(HL) + 0(LH) + 0(HH)
                //(*outLL) += alpha2P * (*outHL);
                // HL = 0(LL) + 1(HL) + 0(LH) + 0(HH)

                // LH = 0(LL) + 0(HL) + 1(LH) + U0(HH)
                (*outLH) += alpha2P * (*outHH);
                // HH = 0(LL) + 0(HL) + 0(LH) +  1(HH)


                // S^V_{U0}
                // LL = 1(LL) + 0(HL) + U0*(LH) + 0(HH)
                (*outLL) += alpha2P * ((*outHL) + (*outLH));
                // HL = 0(LL) + 1(HL) + 0(LH) + U0*(HH)
                (*outHL) += alpha2P * (*outHH);
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

TRANSFORM_STEP_FUNC NO_TREE_VECTORIZE cdf97_non_separable_polyconvolution_generic(size_t step)
{
    (void)(step);
    return cdf97_non_separable_polyconvolution_generic_all;
}

void NO_TREE_VECTORIZE cdf97_non_separable_polyconvolution_generic_transform_tile(const TransformStepArguments * tsa)
{
    cdf97_non_separable_polyconvolution_generic_all(tsa);
}
