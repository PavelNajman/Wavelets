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

static void NO_TREE_VECTORIZE cdf97_non_separable_convolution_star_amd64_sse_all(const TransformStepArguments * tsa)
{
    __m128 alpha     = _mm_set1_ps(-1.58613434342059f);
    __m128 alpha2    = _mm_set1_ps(+0.4435068520439f);

    __m128 beta      = _mm_set1_ps(+0.0136748f);
    __m128 gamma     = _mm_set1_ps(+0.0979904f);
    __m128 delta     = _mm_set1_ps(+0.0496387f);
    __m128 epsilon   = _mm_set1_ps(+0.702178f);
    __m128 zeta      = _mm_set1_ps(+0.3557f);
    __m128 eta       = _mm_set1_ps(+0.180186f);
    __m128 theta     = _mm_set1_ps(-0.00862145f);
    __m128 iota      = _mm_set1_ps(-0.0617794f);
    __m128 kappa     = _mm_set1_ps(-0.0312954f);
    __m128 lambda    = _mm_set1_ps(+0.0408159f);
    __m128 mu        = _mm_set1_ps(+0.292478f);
    __m128 nu        = _mm_set1_ps(+0.148159f);
    __m128 xi        = _mm_set1_ps(-0.002426f);
    __m128 omicron   = _mm_set1_ps(-0.0173842f);
    __m128 pi        = _mm_set1_ps(-0.00880624f);
    __m128 rho       = _mm_set1_ps(+0.00543551f);
    __m128 sigma     = _mm_set1_ps(-0.0257329f);
    __m128 tau       = _mm_set1_ps(+0.0015295f);
    __m128 upsilon   = _mm_set1_ps(+0.121826f);
    __m128 phi       = _mm_set1_ps(-0.00724101f);
    __m128 chi       = _mm_set1_ps(+0.000430388f);
    __m128 alphaN    = _mm_set1_ps(+0.00867621f);
    __m128 betaN     = _mm_set1_ps(+0.0621718f);
    __m128 gammaN    = _mm_set1_ps(+0.0314941f);
    __m128 epsilonN  = _mm_set1_ps(-0.064882f);
    __m128 zetaN     = _mm_set1_ps(-0.46493f);
    __m128 etaN      = _mm_set1_ps(-0.235518f);
    __m128 thetaN    = _mm_set1_ps(+0.111923f);
    __m128 iotaN     = _mm_set1_ps(+0.802016f);
    __m128 kappaN    = _mm_set1_ps(+0.406275f);
    __m128 lambdaN   = _mm_set1_ps(-0.00547003f);
    __m128 muN       = _mm_set1_ps(-0.0391971f);
    __m128 nuN       = _mm_set1_ps(-0.0198559f);
    __m128 xiN       = _mm_set1_ps(+0.105999f);
    __m128 omicronN  = _mm_set1_ps(+0.759566f);
    __m128 piN       = _mm_set1_ps(+0.384771f);
    __m128 rhoN      = _mm_set1_ps(+0.0258964f);
    __m128 sigmaN    = _mm_set1_ps(-0.00153922f);
    __m128 tauN      = _mm_set1_ps(+0.0409057f);
    __m128 upsilonN  = _mm_set1_ps(-0.193657f);
    __m128 phiN      = _mm_set1_ps(+0.0115105f);
    __m128 chiN      = _mm_set1_ps(-0.0705635f);
    __m128 alphaM    = _mm_set1_ps(+0.334063f);
    __m128 betaM     = _mm_set1_ps(+0.00344866f);
    __m128 gammaM    = _mm_set1_ps(-0.0163267f);
    __m128 epsilonM  = _mm_set1_ps(+0.000970421f);
    __m128 zetaM     = _mm_set1_ps(-0.0668286f);
    __m128 etaM      = _mm_set1_ps(+0.316382f);
    __m128 thetaM    = _mm_set1_ps(-0.018805f);
    __m128 iotaM     = _mm_set1_ps(+0.00550478f);
    __m128 kappaM    = _mm_set1_ps(-0.0411655f);
    __m128 lambdaM   = _mm_set1_ps(+0.0710116f);
    __m128 muM       = _mm_set1_ps(+0.307842f);
    __m128 nuM       = _mm_set1_ps(-0.531035f);
    __m128 xiM       = _mm_set1_ps(+0.916051f);
    __m128 omicronM  = _mm_set1_ps(-0.00347056f);
    __m128 piM       = _mm_set1_ps(+0.0259534f);
    __m128 rhoM      = _mm_set1_ps(-0.0447703f);
    __m128 sigmaM    = _mm_set1_ps(+0.0672531f);
    __m128 tauM      = _mm_set1_ps(-0.502928f);
    __m128 upsilonM  = _mm_set1_ps(+0.867565f);
    __m128 phiM      = _mm_set1_ps(+0.00218806f);
    __m128 chiM      = _mm_set1_ps(-0.0424006f);
    __m128 alphaO    = _mm_set1_ps(+0.821645f);

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
                __m128 mLHY1X = _mm_load_ps((bandLHY1));
                __m128 mHHY1X = _mm_load_ps((bandHHY1));

                __m128 mLLY_1X = _mm_load_ps((bandLLY_1));
                __m128 mHLY_1X = _mm_load_ps((bandHLY_1));
                __m128 mLHY_1X = _mm_load_ps((bandLHY_1));
                __m128 mHHY_1X = _mm_load_ps((bandHHY_1));

                __m128 mLLY2X = _mm_load_ps((bandLLY2));
                __m128 mHLY2X = _mm_load_ps((bandHLY2));

                __m128 mLHY_2X = _mm_load_ps((bandLHY_2));
                __m128 mHHY_2X = _mm_load_ps((bandHHY_2));

                __m128 mLLYX1 = load_unpacked_XX01(mLLYX, _mm_load_ps(tmpLL + 4));
                __m128 mHLYX1 = load_unpacked_XX01(mHLYX, _mm_load_ps(tmpHL + 4));
                __m128 mLHYX1 = load_unpacked_XX01(mLHYX, _mm_load_ps(tmpLH + 4));
                __m128 mHHYX1 = load_unpacked_XX01(mHHYX, _mm_load_ps(tmpHH + 4));

                __m128 mLLY1X1 = load_unpacked_XX01(mLLY1X, _mm_load_ps(bandLLY1 + 4));
                __m128 mHLY1X1 = load_unpacked_XX01(mHLY1X, _mm_load_ps(bandHLY1 + 4));
                __m128 mLHY1X1 = load_unpacked_XX01(mLHY1X, _mm_load_ps(bandLHY1 + 4));
                __m128 mHHY1X1 = load_unpacked_XX01(mHHY1X, _mm_load_ps(bandHHY1 + 4));

                __m128 mLLY_1X1 = load_unpacked_XX01(mLLY_1X, _mm_load_ps(bandLLY_1 + 4));
                __m128 mHLY_1X1 = load_unpacked_XX01(mHLY_1X, _mm_load_ps(bandHLY_1 + 4));
                __m128 mLHY_1X1 = load_unpacked_XX01(mLHY_1X, _mm_load_ps(bandLHY_1 + 4));
                __m128 mHHY_1X1 = load_unpacked_XX01(mHHY_1X, _mm_load_ps(bandHHY_1 + 4));

                __m128 mLLY2X1 = load_unpacked_XX01(mLLY2X, _mm_load_ps(bandLLY2 + 4));
                __m128 mHLY2X1 = load_unpacked_XX01(mHLY2X, _mm_load_ps(bandHLY2 + 4));

                __m128 mLHY_2X1 = load_unpacked_XX01(mLHY_2X, _mm_load_ps(bandLHY_2 + 4));
                __m128 mHHY_2X1 = load_unpacked_XX01(mHHY_2X, _mm_load_ps(bandHHY_2 + 4));

                __m128 mLLYX2 = load_unpacked_XX02(mLLYX1, _mm_load_ps(tmpLL + 4));

                __m128 mLHYX2 = load_unpacked_XX02(mLHYX1, _mm_load_ps(tmpLH + 4));

                __m128 mLLY1X2 = load_unpacked_XX02(mLLY1X1, _mm_load_ps(bandLLY1 + 4));

                __m128 mLHY1X2 = load_unpacked_XX02(mLHY1X1, _mm_load_ps(bandLHY1 + 4));

                __m128 mLLY_1X2 = load_unpacked_XX02(mLLY_1X1, _mm_load_ps(bandLLY_1 + 4));

                __m128 mLHY_1X2 = load_unpacked_XX02(mLHY_1X1, _mm_load_ps(bandLHY_1 + 4));

                __m128 mLLY2X2 = load_unpacked_XX02(mLLY2X1, _mm_load_ps(bandLLY2 + 4));

                __m128 mLHY_2X2 = load_unpacked_XX02(mLHY_2X1, _mm_load_ps(bandLHY_2 + 4));

                __m128 mLLYX_1 = load_unpacked_XX01L(mLLYX, _mm_load_ps(tmpLL - 4));
                __m128 mHLYX_1 = load_unpacked_XX01L(mHLYX, _mm_load_ps(tmpHL - 4));
                __m128 mLHYX_1 = load_unpacked_XX01L(mLHYX, _mm_load_ps(tmpLH - 4));
                __m128 mHHYX_1 = load_unpacked_XX01L(mHHYX, _mm_load_ps(tmpHH - 4));

                __m128 mLLY1X_1 = load_unpacked_XX01L(mLLY1X, _mm_load_ps(bandLLY1 - 4));
                __m128 mHLY1X_1 = load_unpacked_XX01L(mHLY1X, _mm_load_ps(bandHLY1 - 4));
                __m128 mLHY1X_1 = load_unpacked_XX01L(mLHY1X, _mm_load_ps(bandLHY1 - 4));
                __m128 mHHY1X_1 = load_unpacked_XX01L(mHHY1X, _mm_load_ps(bandHHY1 - 4));

                __m128 mLLY_1X_1 = load_unpacked_XX01L(mLLY_1X, _mm_load_ps(bandLLY_1 - 4));
                __m128 mHLY_1X_1 = load_unpacked_XX01L(mHLY_1X, _mm_load_ps(bandHLY_1 - 4));
                __m128 mLHY_1X_1 = load_unpacked_XX01L(mLHY_1X, _mm_load_ps(bandLHY_1 - 4));
                __m128 mHHY_1X_1 = load_unpacked_XX01L(mHHY_1X, _mm_load_ps(bandHHY_1 - 4));

                __m128 mLLY2X_1 = load_unpacked_XX01L(mLLY2X, _mm_load_ps(bandLLY2 - 4));
                __m128 mHLY2X_1 = load_unpacked_XX01L(mHLY2X, _mm_load_ps(bandHLY2 - 4));

                __m128 mLHY_2X_1 = load_unpacked_XX01L(mLHY_2X, _mm_load_ps(bandLHY_2 - 4));
                __m128 mHHY_2X_1 = load_unpacked_XX01L(mHHY_2X, _mm_load_ps(bandHHY_2 - 4));

                __m128 mHLYX_2 = load_unpacked_XX02L(mHLYX_1, _mm_load_ps(tmpHL - 4));

                __m128 mHHYX_2 = load_unpacked_XX02L(mHHYX_1, _mm_load_ps(tmpHH - 4));

                __m128 mHLY1X_2 = load_unpacked_XX02L(mHLY1X_1, _mm_load_ps(bandHLY1 - 4));

                __m128 mHHY1X_2 = load_unpacked_XX02L(mHHY1X_1, _mm_load_ps(bandHHY1 - 4));

                __m128 mHLY_1X_2 = load_unpacked_XX02L(mHLY_1X_1, _mm_load_ps(bandHLY_1 - 4));

                __m128 mHHY_1X_2 = load_unpacked_XX02L(mHHY_1X_1, _mm_load_ps(bandHHY_1 - 4));

                __m128 mHLY2X_2 = load_unpacked_XX02L(mHLY2X_1, _mm_load_ps(bandHLY2 - 4));

                __m128 mHHY_2X_2 = load_unpacked_XX02L(mHHY_2X_1, _mm_load_ps(bandHHY_2 - 4));

                // M
                // LL = M00(LL) + M01(HL) + M02(LH) + M03(HH)
                __m128 mOutLL;
                mOutLL = _mm_mul_ps(beta, mLLY1X1);
                mOutLL = mul_add(gamma, _mm_add_ps(mLLYX1, mLLY1X), mOutLL);
                mOutLL = mul_add(delta, _mm_add_ps(mLLY_1X1, mLLY1X_1), mOutLL);
                mOutLL = mul_add(epsilon, mLLYX, mOutLL);
                mOutLL = mul_add(zeta, _mm_add_ps(mLLY_1X, mLLYX_1), mOutLL);
                mOutLL = mul_add(eta, mLLY_1X_1, mOutLL);
                mOutLL = mul_add(theta, _mm_add_ps(mHLY1X, mLHYX1), mOutLL);
                mOutLL = mul_add(iota, _mm_add_ps(mHLYX, mLHYX), mOutLL);
                mOutLL = mul_add(kappa, _mm_add_ps(mHLY_1X, mLHYX_1), mOutLL);
                mOutLL = mul_add(lambda, _mm_add_ps(mHLY1X_1, mLHY_1X1), mOutLL);
                mOutLL = mul_add(mu, _mm_add_ps(mHLYX_1, mLHY_1X), mOutLL);
                mOutLL = mul_add(nu, _mm_add_ps(mHLY_1X_1, mLHY_1X_1), mOutLL);
                mOutLL = mul_add(xi, _mm_add_ps(mHLY1X_2, mLHY_2X1), mOutLL);
                mOutLL = mul_add(omicron, _mm_add_ps(mHLYX_2, mLHY_2X), mOutLL);
                mOutLL = mul_add(pi, _mm_add_ps(mHLY_1X_2, mLHY_2X_1), mOutLL);
                mOutLL = mul_add(rho, mHHYX, mOutLL);
                mOutLL = mul_add(sigma, _mm_add_ps(mHHY_1X, mHHYX_1), mOutLL);
                mOutLL = mul_add(tau, _mm_add_ps(mHHY_2X, mHHYX_2), mOutLL);
                mOutLL = mul_add(upsilon, mHHY_1X_1, mOutLL);
                mOutLL = mul_add(phi, _mm_add_ps(mHHY_2X_1, mHHY_1X_2), mOutLL);
                mOutLL = mul_add(chi, mHHY_2X_2, mOutLL);

                // HL = M10(LL) + M11(HL) + M12(LH) + M13(HH)
                __m128 mOutHL;
                mOutHL = _mm_mul_ps(alphaN, mLLY1X2);
                mOutHL = mul_add(betaN, mLLYX2, mOutHL);
                mOutHL = mul_add(gammaN, mLLY_1X2, mOutHL);
                mOutHL = mul_add(epsilonN, mLLY1X1, mOutHL);
                mOutHL = mul_add(zetaN, mLLYX1, mOutHL);
                mOutHL = mul_add(etaN, mLLY_1X1, mOutHL);
                mOutHL = mul_add(thetaN, mLLY1X, mOutHL);
                mOutHL = mul_add(iotaN, mLLYX, mOutHL);
                mOutHL = mul_add(kappaN, mLLY_1X, mOutHL);
                mOutHL = mul_add(lambdaN, _mm_add_ps(_mm_add_ps(mHLY1X1, mHLY1X_1), mLHYX2), mOutHL);
                mOutHL = mul_add(muN, _mm_add_ps(mHLYX1, mHLYX_1), mOutHL);
                mOutHL = mul_add(nuN, _mm_add_ps(_mm_add_ps(mHLY_1X1, mHLY_1X_1), mLHY_2X), mOutHL);
                mOutHL = mul_add(xiN, mHLY1X, mOutHL);
                mOutHL = mul_add(omicronN, mHLYX, mOutHL);
                mOutHL = mul_add(piN, mHLY_1X, mOutHL);
                mOutHL = mul_add(rhoN, mLHY_1X2, mOutHL);
                mOutHL = mul_add(sigmaN, mLHY_2X2, mOutHL);
                mOutHL = mul_add(tauN, mLHYX1, mOutHL);
                mOutHL = mul_add(upsilonN, mLHY_1X1, mOutHL);
                mOutHL = mul_add(phiN, mLHY_2X1, mOutHL);
                mOutHL = mul_add(chiN, mLHYX, mOutHL);
                mOutHL = mul_add(alphaM, mLHY_1X, mOutHL);
                mOutHL = mul_add(betaM, _mm_add_ps(mHHYX1, mHHYX_1), mOutHL);
                mOutHL = mul_add(gammaM, _mm_add_ps(mHHY_1X1, mHHY_1X_1), mOutHL);
                mOutHL = mul_add(epsilonM, _mm_add_ps(mHHY_2X1, mHHY_2X_1), mOutHL);
                mOutHL = mul_add(zetaM, mHHYX, mOutHL);
                mOutHL = mul_add(etaM, mHHY_1X, mOutHL);
                mOutHL = mul_add(thetaM, mHHY_2X, mOutHL);

                // LH = M20(LL) + M21(HL) + M22(LH) + M23(HH)
                __m128 mOutLH;
                mOutLH = _mm_mul_ps(alphaN, mLLY2X1);
                mOutLH = mul_add(epsilonN, mLLY1X1, mOutLH);
                mOutLH = mul_add(thetaN, mLLYX1, mOutLH);
                mOutLH = mul_add(betaN, mLLY2X, mOutLH);
                mOutLH = mul_add(zetaN, mLLY1X, mOutLH);
                mOutLH = mul_add(iotaN, mLLYX, mOutLH);
                mOutLH = mul_add(gammaN, mLLY2X_1, mOutLH);
                mOutLH = mul_add(etaN, mLLY1X_1, mOutLH);
                mOutLH = mul_add(kappaN, mLLYX_1, mOutLH);
                mOutLH = mul_add(lambdaN, _mm_add_ps(_mm_add_ps(mHLY2X, mLHY1X1), mLHY_1X1), mOutLH);
                mOutLH = mul_add(tauN, mHLY1X, mOutLH);
                mOutLH = mul_add(chiN, mHLYX, mOutLH);
                mOutLH = mul_add(rhoN, mHLY2X_1, mOutLH);
                mOutLH = mul_add(upsilonN, mHLY1X_1, mOutLH);
                mOutLH = mul_add(alphaM, mHLYX_1, mOutLH);
                mOutLH = mul_add(sigmaN, mHLY2X_2, mOutLH);
                mOutLH = mul_add(phiN, mHLY1X_2, mOutLH);
                mOutLH = mul_add(nuN, _mm_add_ps(_mm_add_ps(mHLYX_2, mLHY1X_1), mLHY_1X_1), mOutLH);
                mOutLH = mul_add(xiN, mLHYX1, mOutLH);
                mOutLH = mul_add(muN, _mm_add_ps(mLHY1X, mLHY_1X), mOutLH);
                mOutLH = mul_add(omicronN, mLHYX, mOutLH);
                mOutLH = mul_add(piN, mLHYX_1, mOutLH);
                mOutLH = mul_add(betaM, _mm_add_ps(mHHY1X, mHHY_1X), mOutLH);
                mOutLH = mul_add(zetaM, mHHYX, mOutLH);
                mOutLH = mul_add(gammaM, _mm_add_ps(mHHY1X_1, mHHY_1X_1), mOutLH);
                mOutLH = mul_add(etaM, mHHYX_1, mOutLH);
                mOutLH = mul_add(epsilonM, _mm_add_ps(mHHY1X_2, mHHY_1X_2), mOutLH);
                mOutLH = mul_add(thetaM, mHHYX_2, mOutLH);

                // HH = M30(LL) + M31(HL) + M32(LH) + M33(HH)
                __m128 mOutHH;
                mOutHH = _mm_mul_ps(iotaM, mLLY2X2);
                mOutHH = mul_add(kappaM, _mm_add_ps(mLLY1X2, mLLY2X1), mOutHH);
                mOutHH = mul_add(lambdaM, _mm_add_ps(mLLYX2, mLLY2X), mOutHH);
                mOutHH = mul_add(muM, mLLY1X1, mOutHH);
                mOutHH = mul_add(nuM, _mm_add_ps(mLLYX1, mLLY1X), mOutHH);
                mOutHH = mul_add(xiM, mLLYX, mOutHH);
                mOutHH = mul_add(omicronM, _mm_add_ps(_mm_add_ps(_mm_add_ps(mHLY2X1, mHLY2X_1), mLHY1X2), mLHY_1X2), mOutHH);
                mOutHH = mul_add(piM, _mm_add_ps(_mm_add_ps(_mm_add_ps(mHLY1X1, mHLY1X_1), mLHY1X1), mLHY_1X1), mOutHH);
                mOutHH = mul_add(rhoM, _mm_add_ps(_mm_add_ps(_mm_add_ps(mHLYX1, mHLYX_1), mLHY1X), mLHY_1X), mOutHH);
                mOutHH = mul_add(sigmaM, _mm_add_ps(mHLY2X, mLHYX2), mOutHH);
                mOutHH = mul_add(tauM, _mm_add_ps(mHLY1X, mLHYX1), mOutHH);
                mOutHH = mul_add(upsilonM, _mm_add_ps(mHLYX, mLHYX), mOutHH);
                mOutHH = mul_add(phiM, _mm_add_ps(_mm_add_ps(_mm_add_ps(mHHY1X1, mHHY_1X1), mHHY1X_1), mHHY_1X_1), mOutHH);
                mOutHH = mul_add(chiM, _mm_add_ps(_mm_add_ps(_mm_add_ps(mHHYX1, mHHY1X), mHHY_1X), mHHYX_1), mOutHH);
                mOutHH = mul_add(alphaO, mHHYX, mOutHH);

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
                bandLHY1 += 4;
                bandHHY1 += 4;
                bandLLY_1 += 4;
                bandHLY_1 += 4;
                bandLHY_1 += 4;
                bandHHY_1 += 4;

                bandLLY2 += 4;
                bandHLY2 += 4;
                bandLHY_2 += 4;
                bandHHY_2 += 4;
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
            bandLHY1 += nextTmpBandY;
            bandHHY1 += nextTmpBandY;
            bandLLY_1 += nextTmpBandY;
            bandHLY_1 += nextTmpBandY;
            bandLHY_1 += nextTmpBandY;
            bandHHY_1 += nextTmpBandY;

            bandLLY2 += nextTmpBandY;
            bandHLY2 += nextTmpBandY;
            bandLHY_2 += nextTmpBandY;
            bandHHY_2 += nextTmpBandY;
        }
    }
    //#pragma omp barrier
}

TRANSFORM_STEP_FUNC NO_TREE_VECTORIZE cdf97_non_separable_convolution_star_amd64_sse(size_t step)
{
    (void)(step);
    return cdf97_non_separable_convolution_star_amd64_sse_all;
}

void NO_TREE_VECTORIZE cdf97_non_separable_convolution_star_amd64_sse_transform_tile(const TransformStepArguments * tsa)
{
    cdf97_non_separable_convolution_star_amd64_sse_all(tsa);
}
#endif
