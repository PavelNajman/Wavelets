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

static void NO_TREE_VECTORIZE cdf97_non_separable_convolution_star_generic_all(const TransformStepArguments * tsa)
{
    float alpha     = -1.58613434342059f;
    float alpha2    = +0.4435068520439f;

    float beta      = +0.0136748f;
    float gamma     = +0.0979904f;
    float delta     = +0.0496387f;
    float epsilon   = +0.702178f;
    float zeta      = +0.3557f;
    float eta       = +0.180186f;
    float theta     = -0.00862145f;
    float iota      = -0.0617794f;
    float kappa     = -0.0312954f;
    float lambda    = +0.0408159f;
    float mu        = +0.292478f;
    float nu        = +0.148159f;
    float xi        = -0.002426f;
    float omicron   = -0.0173842f;
    float pi        = -0.00880624f;
    float rho       = +0.00543551f;
    float sigma     = -0.0257329f;
    float tau       = +0.0015295f;
    float upsilon   = +0.121826f;
    float phi       = -0.00724101f;
    float chi       = +0.000430388f;
    float alphaN    = +0.00867621f;
    float betaN     = +0.0621718f;
    float gammaN    = +0.0314941f;
    float epsilonN  = -0.064882f;
    float zetaN     = -0.46493f;
    float etaN      = -0.235518f;
    float thetaN    = +0.111923f;
    float iotaN     = +0.802016f;
    float kappaN    = +0.406275f;
    float lambdaN   = -0.00547003f;
    float muN       = -0.0391971f;
    float nuN       = -0.0198559f;
    float xiN       = +0.105999f;
    float omicronN  = +0.759566f;
    float piN       = +0.384771f;
    float rhoN      = +0.0258964f;
    float sigmaN    = -0.00153922f;
    float tauN      = +0.0409057f;
    float upsilonN  = -0.193657f;
    float phiN      = +0.0115105f;
    float chiN      = -0.0705635f;
    float alphaM    = +0.334063f;
    float betaM     = +0.00344866f;
    float gammaM    = -0.0163267f;
    float epsilonM  = +0.000970421f;
    float zetaM     = -0.0668286f;
    float etaM      = +0.316382f;
    float thetaM    = -0.018805f;
    float iotaM     = +0.00550478f;
    float kappaM    = -0.0411655f;
    float lambdaM   = +0.0710116f;
    float muM       = +0.307842f;
    float nuM       = -0.531035f;
    float xiM       = +0.916051f;
    float omicronM  = -0.00347056f;
    float piM       = +0.0259534f;
    float rhoM      = -0.0447703f;
    float sigmaM    = +0.0672531f;
    float tauM      = -0.502928f;
    float upsilonM  = +0.867565f;
    float phiM      = +0.00218806f;
    float chiM      = -0.0424006f;
    float alphaO    = +0.821645f;

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
    float *bandLHY1 = tsa->tmp.LH + (bandStartY + 1 + tmpBandsMargin) * tmpBandsStrideY + bandStartXMargin;
    float *bandHHY1 = tsa->tmp.HH + (bandStartY + 1 + tmpBandsMargin) * tmpBandsStrideY + bandStartXMargin;
    float *bandLLY_1 = tsa->tmp.LL + (bandStartY - 1 + tmpBandsMargin) * tmpBandsStrideY + bandStartXMargin;
    float *bandHLY_1 = tsa->tmp.HL + (bandStartY - 1 + tmpBandsMargin) * tmpBandsStrideY + bandStartXMargin;
    float *bandLHY_1 = tsa->tmp.LH + (bandStartY - 1 + tmpBandsMargin) * tmpBandsStrideY + bandStartXMargin;
    float *bandHHY_1 = tsa->tmp.HH + (bandStartY - 1 + tmpBandsMargin) * tmpBandsStrideY + bandStartXMargin;

    float *bandLLY2 = tsa->tmp.LL + (bandStartY + 2 + tmpBandsMargin) * tmpBandsStrideY + bandStartXMargin;
    float *bandHLY2 = tsa->tmp.HL + (bandStartY + 2 + tmpBandsMargin) * tmpBandsStrideY + bandStartXMargin;
    float *bandLHY_2 = tsa->tmp.LH + (bandStartY - 2 + tmpBandsMargin) * tmpBandsStrideY + bandStartXMargin;
    float *bandHHY_2 = tsa->tmp.HH + (bandStartY - 2 + tmpBandsMargin) * tmpBandsStrideY + bandStartXMargin;

    long lx1 = +1;
    long lx_1 = -1;

    long lx2 = +2;
    long lx_2 = -2;

    // N_{P1,U1}, S^H_{U0}, S^V_{U0}
    {
        for (size_t ly = bandStartY; ly < bandEndY; ly++) {
            for (size_t lx = bandStartX; lx < bandEndX; lx++) {

                // M
                // LL = M00(LL) + M01(HL) + M02(LH) + M03(HH)
                (*outLL) = beta * LOCA_TMP(bandLLY1, lx1)
                        + gamma * (LOCA_TMP(tmpLL, lx1) + (*bandLLY1))
                        + delta * (LOCA_TMP(bandLLY_1, lx1) + LOCA_TMP(bandLLY1, lx_1))
                        + epsilon * (*tmpLL)
                        + zeta * ((*bandLLY_1) + LOCA_TMP(tmpLL, lx_1))
                        + eta * LOCA_TMP(bandLLY_1, lx_1)
                        + theta * ((*bandHLY1) + LOCA_TMP(tmpLH, lx1))
                        + iota * ((*tmpHL) + (*tmpLH))
                        + kappa * ((*bandHLY_1) + LOCA_TMP(tmpLH, lx_1))
                        + lambda * (LOCA_TMP(bandHLY1, lx_1) + LOCA_TMP(bandLHY_1, lx1))
                        + mu * (LOCA_TMP(tmpHL, lx_1) + (*bandLHY_1))
                        + nu * (LOCA_TMP(bandHLY_1, lx_1) + LOCA_TMP(bandLHY_1, lx_1))
                        + xi * (LOCA_TMP(bandHLY1, lx_2) + LOCA_TMP(bandLHY_2, lx1))
                        + omicron * (LOCA_TMP(tmpHL, lx_2) + (*bandLHY_2))
                        + pi * (LOCA_TMP(bandHLY_1, lx_2) + LOCA_TMP(bandLHY_2, lx_1))
                        + rho * (*tmpHH)
                        + sigma * ((*bandHHY_1) + LOCA_TMP(tmpHH, lx_1))
                        + tau * ((*bandHHY_2) + LOCA_TMP(tmpHH, lx_2))
                        + upsilon * LOCA_TMP(bandHHY_1, lx_1)
                        + phi * (LOCA_TMP(bandHHY_2, lx_1) + LOCA_TMP(bandHHY_1, lx_2))
                        + chi * LOCA_TMP(bandHHY_2, lx_2);

                // HL = M10(LL) + M11(HL) + M12(LH) + M13(HH)
                (*outHL) = alphaN * LOCA_TMP(bandLLY1, lx2)
                        + betaN * LOCA_TMP(tmpLL, lx2)
                        + gammaN * LOCA_TMP(bandLLY_1, lx2)
                        + epsilonN * LOCA_TMP(bandLLY1, lx1)
                        + zetaN * LOCA_TMP(tmpLL, lx1)
                        + etaN * LOCA_TMP(bandLLY_1, lx1)
                        + thetaN * (*bandLLY1)
                        + iotaN * (*tmpLL)
                        + kappaN * (*bandLLY_1)
                        + lambdaN * (LOCA_TMP(bandHLY1, lx1) + LOCA_TMP(bandHLY1, lx_1) + LOCA_TMP(tmpLH, lx2))
                        + muN * (LOCA_TMP(tmpHL, lx1) + LOCA_TMP(tmpHL, lx_1))
                        + nuN * (LOCA_TMP(bandHLY_1, lx1) + LOCA_TMP(bandHLY_1, lx_1) + (*bandLHY_2))
                        + xiN * (*bandHLY1)
                        + omicronN * (*tmpHL)
                        + piN * (*bandHLY_1)
                        + rhoN * LOCA_TMP(bandLHY_1, lx2)
                        + sigmaN * LOCA_TMP(bandLHY_2, lx2)
                        + tauN * LOCA_TMP(tmpLH, lx1)
                        + upsilonN * LOCA_TMP(bandLHY_1, lx1)
                        + phiN * LOCA_TMP(bandLHY_2, lx1)
                        + chiN * (*tmpLH)
                        + alphaM * (*bandLHY_1)
                        + betaM * (LOCA_TMP(tmpHH, lx1) + LOCA_TMP(tmpHH, lx_1))
                        + gammaM * (LOCA_TMP(bandHHY_1, lx1) + LOCA_TMP(bandHHY_1, lx_1))
                        + epsilonM * (LOCA_TMP(bandHHY_2, lx1) + LOCA_TMP(bandHHY_2, lx_1))
                        + zetaM * (*tmpHH)
                        + etaM * (*bandHHY_1)
                        + thetaM * (*bandHHY_2);

                // LH = M20(LL) + M21(HL) + M22(LH) + M23(HH)
                (*outLH) = alphaN * LOCA_TMP(bandLLY2, lx1)
                        + epsilonN * LOCA_TMP(bandLLY1, lx1)
                        + thetaN * LOCA_TMP(tmpLL, lx1)
                        + betaN * (*bandLLY2)
                        + zetaN * (*bandLLY1)
                        + iotaN * (*tmpLL)
                        + gammaN * LOCA_TMP(bandLLY2, lx_1)
                        + etaN * LOCA_TMP(bandLLY1, lx_1)
                        + kappaN * LOCA_TMP(tmpLL, lx_1)
                        + lambdaN * ((*bandHLY2) + LOCA_TMP(bandLHY1, lx1) + LOCA_TMP(bandLHY_1, lx1))
                        + tauN * (*bandHLY1)
                        + chiN * (*tmpHL)
                        + rhoN * LOCA_TMP(bandHLY2, lx_1)
                        + upsilonN * LOCA_TMP(bandHLY1, lx_1)
                        + alphaM * LOCA_TMP(tmpHL, lx_1)
                        + sigmaN * LOCA_TMP(bandHLY2, lx_2)
                        + phiN * LOCA_TMP(bandHLY1, lx_2)
                        + nuN * (LOCA_TMP(tmpHL, lx_2) + LOCA_TMP(bandLHY1, lx_1) + LOCA_TMP(bandLHY_1, lx_1))
                        + xiN * LOCA_TMP(tmpLH, lx1)
                        + muN * ((*bandLHY1) + (*bandLHY_1))
                        + omicronN * (*tmpLH)
                        + piN * LOCA_TMP(tmpLH, lx_1)
                        + betaM * ((*bandHHY1) + (*bandHHY_1))
                        + zetaM * (*tmpHH)
                        + gammaM * (LOCA_TMP(bandHHY1, lx_1) + LOCA_TMP(bandHHY_1, lx_1))
                        + etaM * LOCA_TMP(tmpHH, lx_1)
                        + epsilonM * (LOCA_TMP(bandHHY1, lx_2) + LOCA_TMP(bandHHY_1, lx_2))
                        + thetaM * LOCA_TMP(tmpHH, lx_2);

                // HH = M30(LL) + M31(HL) + M32(LH) + M33(HH)
                (*outHH) = iotaM * LOCA_TMP(bandLLY2, lx2)
                        + kappaM * (LOCA_TMP(bandLLY1, lx2) + LOCA_TMP(bandLLY2, lx1))
                        + lambdaM * (LOCA_TMP(tmpLL, lx2) + (*bandLLY2))
                        + muM * LOCA_TMP(bandLLY1, lx1)
                        + nuM * (LOCA_TMP(tmpLL, lx1) + (*bandLLY1))
                        + xiM * (*tmpLL)
                        + omicronM * (LOCA_TMP(bandHLY2, lx1) + LOCA_TMP(bandHLY2, lx_1) + LOCA_TMP(bandLHY1, lx2) + LOCA_TMP(bandLHY_1, lx2))
                        + piM * (LOCA_TMP(bandHLY1, lx1) + LOCA_TMP(bandHLY1, lx_1) + LOCA_TMP(bandLHY1, lx1) + LOCA_TMP(bandLHY_1, lx1))
                        + rhoM * (LOCA_TMP(tmpHL, lx1) + LOCA_TMP(tmpHL, lx_1) + (*bandLHY1) + (*bandLHY_1))
                        + sigmaM * ((*bandHLY2) + LOCA_TMP(tmpLH, lx2))
                        + tauM * ((*bandHLY1) + LOCA_TMP(tmpLH, lx1))
                        + upsilonM * ((*tmpHL) + (*tmpLH))
                        + phiM * (LOCA_TMP(bandHHY1, lx1) + LOCA_TMP(bandHHY_1, lx1) + LOCA_TMP(bandHHY1, lx_1) + LOCA_TMP(bandHHY_1, lx_1))
                        + chiM * (LOCA_TMP(tmpHH, lx1) + (*bandHHY1) + (*bandHHY_1) + LOCA_TMP(tmpHH, lx_1))
                        + alphaO * (*tmpHH);

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
                bandLHY1++;
                bandHHY1++;
                bandLLY_1++;
                bandHLY_1++;
                bandLHY_1++;
                bandHHY_1++;

                bandLLY2++;
                bandHLY2++;
                bandLHY_2++;
                bandHHY_2++;
            }

            outLL += nextBandY;
            outHL += nextBandY;
            outLH += nextBandY;
            outHH += nextBandY;

            tmpLL += nextTmpBandY;
            tmpHL += nextTmpBandY;
            tmpLH += nextTmpBandY;
            tmpHH += nextTmpBandY;

            bandLLY1 += nextTmpBandY;;
            bandHLY1 += nextTmpBandY;;
            bandLHY1 += nextTmpBandY;;
            bandHHY1 += nextTmpBandY;;
            bandLLY_1 += nextTmpBandY;;
            bandHLY_1 += nextTmpBandY;;
            bandLHY_1 += nextTmpBandY;;
            bandHHY_1 += nextTmpBandY;;

            bandLLY2 += nextTmpBandY;;
            bandHLY2 += nextTmpBandY;;
            bandLHY_2 += nextTmpBandY;;
            bandHHY_2 += nextTmpBandY;;
        }
    }
    //#pragma omp barrier
}

TRANSFORM_STEP_FUNC NO_TREE_VECTORIZE cdf97_non_separable_convolution_star_generic(size_t step)
{
    (void)(step);
    return cdf97_non_separable_convolution_star_generic_all;
}

void NO_TREE_VECTORIZE cdf97_non_separable_convolution_star_generic_transform_tile(const TransformStepArguments * tsa)
{
    cdf97_non_separable_convolution_star_generic_all(tsa);
}
