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

static void NO_TREE_VECTORIZE dd137_non_separable_convolution_star_amd64_sse_all(const TransformStepArguments * tsa)
{
    __m128 alpha     = _mm_set1_ps(-0.5625f);
    __m128 alpha2    = _mm_set1_ps(+0.28125f);
    __m128 beta      = _mm_set1_ps(+3.8147e-6f);
    __m128 gamma     = _mm_set1_ps(-3.43323e-5f);
    __m128 delta     = _mm_set1_ps(-0.00163651f);
    __m128 epsilon   = _mm_set1_ps(+0.00030899f);
    __m128 zeta      = _mm_set1_ps(+0.0147285f);
    __m128 eta       = _mm_set1_ps(+0.702061f);
    __m128 theta     = _mm_set1_ps(+6.10352e-5f);
    __m128 iota      = _mm_set1_ps(-0.000549316f);
    __m128 kappa     = _mm_set1_ps(+0.00494385f);
    __m128 lambda    = _mm_set1_ps(-0.0261841f);
    __m128 mu        = _mm_set1_ps(+0.235657f);
    __m128 nu        = _mm_set1_ps(+0.000976562f);
    __m128 xi        = _mm_set1_ps(-0.00878906f);
    __m128 omicron   = _mm_set1_ps(+0.0791016f);
    __m128 pi        = _mm_set1_ps(-0.00012207f);
    __m128 rho       = _mm_set1_ps(+0.00109863f);
    __m128 sigma     = _mm_set1_ps(-0.0098877f);
    __m128 tau       = _mm_set1_ps(+0.0523682f);
    __m128 upsilon   = _mm_set1_ps(-0.471313f);
    __m128 phi       = _mm_set1_ps(-0.00195312f);
    __m128 chi       = _mm_set1_ps(+0.0175781f);
    __m128 alphaN    = _mm_set1_ps(+0.837891f);
    __m128 betaN     = _mm_set1_ps(-0.158203f);
    __m128 gammaN    = _mm_set1_ps(-0.03125f);
    __m128 epsilonN  = _mm_set1_ps(+0.00390625f);
    __m128 zetaN     = _mm_set1_ps(-0.0351562f);
    __m128 etaN      = _mm_set1_ps(+0.316406f);
    __m128 thetaN    = _mm_set1_ps(+0.0625f);

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
    float *bandLLY_2 = tsa->tmp.LL + (bandStartY - 2 + tmpBandsMargin) * tmpBandsStrideY + bandStartXMargin;
    float *bandHLY_2 = tsa->tmp.HL + (bandStartY - 2 + tmpBandsMargin) * tmpBandsStrideY + bandStartXMargin;
    float *bandLHY_2 = tsa->tmp.LH + (bandStartY - 2 + tmpBandsMargin) * tmpBandsStrideY + bandStartXMargin;
    float *bandHHY_2 = tsa->tmp.HH + (bandStartY - 2 + tmpBandsMargin) * tmpBandsStrideY + bandStartXMargin;

    float *bandLLY3 = tsa->tmp.LL + (bandStartY + 3 + tmpBandsMargin) * tmpBandsStrideY + bandStartXMargin;
    float *bandHLY3 = tsa->tmp.HL + (bandStartY + 3 + tmpBandsMargin) * tmpBandsStrideY + bandStartXMargin;
    float *bandLLY_3 = tsa->tmp.LL + (bandStartY - 3 + tmpBandsMargin) * tmpBandsStrideY + bandStartXMargin;
    float *bandHLY_3 = tsa->tmp.HL + (bandStartY - 3 + tmpBandsMargin) * tmpBandsStrideY + bandStartXMargin;

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

                __m128 mLLY_2X = _mm_load_ps((bandLLY_2));
                __m128 mHLY_2X = _mm_load_ps((bandHLY_2));
                __m128 mLHY_2X = _mm_load_ps((bandLHY_2));
                __m128 mHHY_2X = _mm_load_ps((bandHHY_2));

                __m128 mLLY3X = _mm_load_ps((bandLLY3));
                __m128 mHLY3X = _mm_load_ps((bandHLY3));

                __m128 mLLY_3X = _mm_load_ps((bandLLY_3));
                __m128 mHLY_3X = _mm_load_ps((bandHLY_3));

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

                __m128 mLLY_2X1 = load_unpacked_XX01(mLLY_2X, _mm_load_ps(bandLLY_2 + 4));
                __m128 mHLY_2X1 = load_unpacked_XX01(mHLY_2X, _mm_load_ps(bandHLY_2 + 4));
                __m128 mLHY_2X1 = load_unpacked_XX01(mLHY_2X, _mm_load_ps(bandLHY_2 + 4));
                __m128 mHHY_2X1 = load_unpacked_XX01(mHHY_2X, _mm_load_ps(bandHHY_2 + 4));

                __m128 mLLY3X1 = load_unpacked_XX01(mLLY3X, _mm_load_ps(bandLLY3 + 4));
                __m128 mHLY3X1 = load_unpacked_XX01(mHLY3X, _mm_load_ps(bandHLY3 + 4));

                __m128 mLLY_3X1 = load_unpacked_XX01(mLLY_3X, _mm_load_ps(bandLLY_3 + 4));
                __m128 mHLY_3X1 = load_unpacked_XX01(mHLY_3X, _mm_load_ps(bandHLY_3 + 4));

                __m128 mLLYX2 = load_unpacked_XX02(mLLYX1, _mm_load_ps(tmpLL + 4));

                __m128 mLHYX2 = load_unpacked_XX02(mLHYX1, _mm_load_ps(tmpLH + 4));

                __m128 mLLY1X2 = load_unpacked_XX02(mLLY1X1, _mm_load_ps(bandLLY1 + 4));

                __m128 mLHY1X2 = load_unpacked_XX02(mLHY1X1, _mm_load_ps(bandLHY1 + 4));

                __m128 mLLY_1X2 = load_unpacked_XX02(mLLY_1X1, _mm_load_ps(bandLLY_1 + 4));

                __m128 mLHY_1X2 = load_unpacked_XX02(mLHY_1X1, _mm_load_ps(bandLHY_1 + 4));

                __m128 mLLY2X2 = load_unpacked_XX02(mLLY2X1, _mm_load_ps(bandLLY2 + 4));

                __m128 mLLY_2X2 = load_unpacked_XX02(mLLY_2X1, _mm_load_ps(bandLLY_2 + 4));

                __m128 mLHY_2X2 = load_unpacked_XX02(mLHY_2X1, _mm_load_ps(bandLHY_2 + 4));

                __m128 mLLY3X2 = load_unpacked_XX02(mLLY3X1, _mm_load_ps(bandLLY3 + 4));

                __m128 mLLY_3X2 = load_unpacked_XX02(mLLY_3X1, _mm_load_ps(bandLLY_3 + 4));

                __m128 mLLYX3 = load_unpacked_XX03(mLLYX2, _mm_load_ps(tmpLL + 4));

                __m128 mLHYX3 = load_unpacked_XX03(mLHYX2, _mm_load_ps(tmpLH + 4));

                __m128 mLLY1X3 = load_unpacked_XX03(mLLY1X2, _mm_load_ps(bandLLY1 + 4));

                __m128 mLHY1X3 = load_unpacked_XX03(mLHY1X2, _mm_load_ps(bandLHY1 + 4));

                __m128 mLLY_1X3 = load_unpacked_XX03(mLLY_1X2, _mm_load_ps(bandLLY_1 + 4));

                __m128 mLHY_1X3 = load_unpacked_XX03(mLHY_1X2, _mm_load_ps(bandLHY_1 + 4));

                __m128 mLLY2X3 = load_unpacked_XX03(mLLY2X2, _mm_load_ps(bandLLY2 + 4));

                __m128 mLLY_2X3 = load_unpacked_XX03(mLLY_2X2, _mm_load_ps(bandLLY_2 + 4));

                __m128 mLHY_2X3 = load_unpacked_XX03(mLHY_2X2, _mm_load_ps(bandLHY_2 + 4));

                __m128 mLLY3X3 = load_unpacked_XX03(mLLY3X2, _mm_load_ps(bandLLY3 + 4));

                __m128 mLLY_3X3 = load_unpacked_XX03(mLLY_3X2, _mm_load_ps(bandLLY_3 + 4));

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

                __m128 mLLY_2X_1 = load_unpacked_XX01L(mLLY_2X, _mm_load_ps(bandLLY_2 - 4));
                __m128 mHLY_2X_1 = load_unpacked_XX01L(mHLY_2X, _mm_load_ps(bandHLY_2 - 4));
                __m128 mLHY_2X_1 = load_unpacked_XX01L(mLHY_2X, _mm_load_ps(bandLHY_2 - 4));
                __m128 mHHY_2X_1 = load_unpacked_XX01L(mHHY_2X, _mm_load_ps(bandHHY_2 - 4));

                __m128 mLLY3X_1 = load_unpacked_XX01L(mLLY3X, _mm_load_ps(bandLLY3 - 4));
                __m128 mHLY3X_1 = load_unpacked_XX01L(mHLY3X, _mm_load_ps(bandHLY3 - 4));

                __m128 mLLY_3X_1 = load_unpacked_XX01L(mLLY_3X, _mm_load_ps(bandLLY_3 - 4));
                __m128 mHLY_3X_1 = load_unpacked_XX01L(mHLY_3X, _mm_load_ps(bandHLY_3 - 4));

                __m128 mLLYX_2 = load_unpacked_XX02L(mLLYX_1, _mm_load_ps(tmpLL - 4));
                __m128 mHLYX_2 = load_unpacked_XX02L(mHLYX_1, _mm_load_ps(tmpHL - 4));
                __m128 mLHYX_2 = load_unpacked_XX02L(mLHYX_1, _mm_load_ps(tmpLH - 4));
                __m128 mHHYX_2 = load_unpacked_XX02L(mHHYX_1, _mm_load_ps(tmpHH - 4));

                __m128 mLLY1X_2 = load_unpacked_XX02L(mLLY1X_1, _mm_load_ps(bandLLY1 - 4));
                __m128 mHLY1X_2 = load_unpacked_XX02L(mHLY1X_1, _mm_load_ps(bandHLY1 - 4));
                __m128 mLHY1X_2 = load_unpacked_XX02L(mLHY1X_1, _mm_load_ps(bandLHY1 - 4));
                __m128 mHHY1X_2 = load_unpacked_XX02L(mHHY1X_1, _mm_load_ps(bandHHY1 - 4));

                __m128 mLLY_1X_2 = load_unpacked_XX02L(mLLY_1X_1, _mm_load_ps(bandLLY_1 - 4));
                __m128 mHLY_1X_2 = load_unpacked_XX02L(mHLY_1X_1, _mm_load_ps(bandHLY_1 - 4));
                __m128 mLHY_1X_2 = load_unpacked_XX02L(mLHY_1X_1, _mm_load_ps(bandLHY_1 - 4));
                __m128 mHHY_1X_2 = load_unpacked_XX02L(mHHY_1X_1, _mm_load_ps(bandHHY_1 - 4));

                __m128 mLLY2X_2 = load_unpacked_XX02L(mLLY2X_1, _mm_load_ps(bandLLY2 - 4));
                __m128 mHLY2X_2 = load_unpacked_XX02L(mHLY2X_1, _mm_load_ps(bandHLY2 - 4));

                __m128 mLLY_2X_2 = load_unpacked_XX02L(mLLY_2X_1, _mm_load_ps(bandLLY_2 - 4));
                __m128 mHLY_2X_2 = load_unpacked_XX02L(mHLY_2X_1, _mm_load_ps(bandHLY_2 - 4));
                __m128 mLHY_2X_2 = load_unpacked_XX02L(mLHY_2X_1, _mm_load_ps(bandLHY_2 - 4));
                __m128 mHHY_2X_2 = load_unpacked_XX02L(mHHY_2X_1, _mm_load_ps(bandHHY_2 - 4));

                __m128 mLLY3X_2 = load_unpacked_XX02L(mLLY3X_1, _mm_load_ps(bandLLY3 - 4));
                __m128 mHLY3X_2 = load_unpacked_XX02L(mHLY3X_1, _mm_load_ps(bandHLY3 - 4));

                __m128 mLLY_3X_2 = load_unpacked_XX02L(mLLY_3X_1, _mm_load_ps(bandLLY_3 - 4));
                __m128 mHLY_3X_2 = load_unpacked_XX02L(mHLY_3X_1, _mm_load_ps(bandHLY_3 - 4));

                __m128 mLLYX_3 = load_unpacked_XX03L(mLLYX_2, _mm_load_ps(tmpLL - 4));
                __m128 mLHYX_3 = load_unpacked_XX03L(mLHYX_2, _mm_load_ps(tmpLH - 4));

                __m128 mLLY1X_3 = load_unpacked_XX03L(mLLY1X_2, _mm_load_ps(bandLLY1 - 4));
                __m128 mLHY1X_3 = load_unpacked_XX03L(mLHY1X_2, _mm_load_ps(bandLHY1 - 4));

                __m128 mLLY_1X_3 = load_unpacked_XX03L(mLLY_1X_2, _mm_load_ps(bandLLY_1 - 4));
                __m128 mLHY_1X_3 = load_unpacked_XX03L(mLHY_1X_2, _mm_load_ps(bandLHY_1 - 4));

                __m128 mLLY2X_3 = load_unpacked_XX03L(mLLY2X_2, _mm_load_ps(bandLLY2 - 4));

                __m128 mLLY_2X_3 = load_unpacked_XX03L(mLLY_2X_2, _mm_load_ps(bandLLY_2 - 4));
                __m128 mLHY_2X_3 = load_unpacked_XX03L(mLHY_2X_2, _mm_load_ps(bandLHY_2 - 4));

                __m128 mLLY3X_3 = load_unpacked_XX03L(mLLY3X_2, _mm_load_ps(bandLLY3 - 4));

                __m128 mLLY_3X_3 = load_unpacked_XX03L(mLLY_3X_2, _mm_load_ps(bandLLY_3 - 4));

                // N_{P1,U1}
                __m128 mOutLL;
                mOutLL = _mm_mul_ps(beta, _mm_add_ps(_mm_add_ps(_mm_add_ps(mLLY3X3, mLLY3X_3), mLLY_3X3), mLLY_3X_3));
                mOutLL = mul_add(gamma, _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(mLLY3X2, mLLY3X1), mLLY3X_1), mLLY3X_2), mLLY2X3), mLLY2X_3), mLLY1X3), mLLY1X_3), mLLY_1X3), mLLY_1X_3), mLLY_2X3), mLLY_2X_3), mLLY_3X2), mLLY_3X1), mLLY_3X_1), mLLY_3X_2), mOutLL);
                mOutLL = mul_add(delta, _mm_add_ps(_mm_add_ps(_mm_add_ps(mLLY3X, mLLYX3), mLLYX_3), mLLY_3X), mOutLL);
                mOutLL = mul_add(epsilon, _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(mLLY2X2, mLLY2X1), mLLY2X_1), mLLY2X_2), mLLY1X2), mLLY1X1), mLLY1X_1), mLLY1X_2), mLLY_1X2), mLLY_1X1), mLLY_1X_1), mLLY_1X_2), mLLY_2X2), mLLY_2X1), mLLY_2X_1), mLLY_2X_2), mOutLL);
                mOutLL = mul_add(zeta, _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(mLLY2X, mLLY1X), mLLYX2), mLLYX1), mLLYX_1), mLLYX_2), mLLY_1X), mLLY_2X), mOutLL);
                mOutLL = mul_add(eta, mLLYX, mOutLL);
                mOutLL = mul_add(theta, _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(mHLY3X1, mHLY3X_2), mHLY_3X1), mHLY_3X_2), mLHY1X3), mLHY1X_3), mLHY_2X3), mLHY_2X_3), mOutLL);
                mOutLL = mul_add(iota, _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(mHLY3X_1, mHLY2X1), mHLY2X_2), mHLY1X1), mHLY1X_2), mHLY_1X1), mHLY_1X_2), mHLY_2X1), mHLY_2X_2), mHLY_3X_1), mLHY1X2), mLHY1X1), mLHY1X_1), mLHY1X_2), mLHY_1X3), mLHY_1X_3), mLHY_2X2), mLHY_2X1), mLHY_2X_1), mLHY_2X_2), mOutLL);
                mOutLL = mul_add(kappa, _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(mHLY2X_1, mHLY1X_1), mHLY_1X_1), mHLY_2X_1), mLHY_1X2), mLHY_1X1), mLHY_1X_1), mLHY_1X_2), mOutLL);
                mOutLL = mul_add(lambda, _mm_add_ps(_mm_add_ps(_mm_add_ps(mHLYX1, mHLYX_2), mLHY1X), mLHY_2X), mOutLL);
                mOutLL = mul_add(mu, _mm_add_ps(mHLYX_1, mLHY_1X), mOutLL);
                mOutLL = mul_add(nu, _mm_add_ps(_mm_add_ps(_mm_add_ps(mHHY1X1, mHHY1X_2), mHHY_2X1), mHHY_2X_2), mOutLL);
                mOutLL = mul_add(xi, _mm_add_ps(_mm_add_ps(_mm_add_ps(mHHY1X_1, mHHY_1X1), mHHY_1X_2), mHHY_2X_1), mOutLL);
                mOutLL = mul_add(omicron, mHHY_1X_1, mOutLL);

                // HL = V1*P1(LL) + V1*(HL) + U1*P1(LH) + U1*(HH)
                __m128 mOutHL;
                mOutHL = _mm_mul_ps(pi, _mm_add_ps(_mm_add_ps(_mm_add_ps(mLLY3X2, mLLY3X_1), mLLY_3X2), mLLY_3X_1));
                mOutHL = mul_add(rho, _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(mLLY3X1, mLLY2X2), mLLY2X_1), mLLY1X2), mLLY1X_1), mLLY_1X2), mLLY_1X_1), mLLY_2X2), mLLY_2X_1), mLLY_3X1), mOutHL);
                mOutHL = mul_add(sigma, _mm_add_ps(_mm_add_ps(_mm_add_ps(mLLY2X1, mLLY1X1), mLLY_1X1), mLLY_2X1), mOutHL);
                mOutHL = mul_add(tau, _mm_add_ps(mLLYX2, mLLYX_1), mOutHL);
                mOutHL = mul_add(upsilon, mLLYX1, mOutHL);
                mOutHL = mul_add(phi, _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(mHLY3X, mHLY_3X), mLHY1X2), mLHY1X_1), mLHY_2X2), mLHY_2X_1), mOutHL);
                mOutHL = mul_add(chi, _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(mHLY2X, mHLY1X), mHLY_1X), mHLY_2X), mLHY1X1), mLHY_1X2), mLHY_1X_1), mLHY_2X1), mOutHL);
                mOutHL = mul_add(alphaN, mHLYX, mOutHL);
                mOutHL = mul_add(betaN, mLHY_1X1, mOutHL);
                mOutHL = mul_add(gammaN, _mm_add_ps(mHHY1X, mHHY_2X), mOutHL);
                mOutHL = mul_add(alpha2, mHHY_1X, mOutHL);

                // LH = P1*V1(LL) + P1*U1(HL) + V1(LH) + U1(HH)
                __m128 mOutLH;
                mOutLH = _mm_mul_ps(pi, _mm_add_ps(_mm_add_ps(_mm_add_ps(mLLY2X3, mLLY2X_3), mLLY_1X3), mLLY_1X_3));
                mOutLH = mul_add(rho, _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(mLLY2X2, mLLY2X1), mLLY2X_1), mLLY2X_2), mLLY1X3), mLLY1X_3), mLLY_1X2), mLLY_1X1), mLLY_1X_1), mLLY_1X_2), mOutLH);
                mOutLH = mul_add(tau, _mm_add_ps(mLLY2X, mLLY_1X), mOutLH);
                mOutLH = mul_add(sigma, _mm_add_ps(_mm_add_ps(_mm_add_ps(mLLY1X2, mLLY1X1), mLLY1X_1), mLLY1X_2), mOutLH);
                mOutLH = mul_add(upsilon, mLLY1X, mOutLH);
                mOutLH = mul_add(phi, _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(mHLY2X1, mHLY2X_2), mHLY_1X1), mHLY_1X_2), mLHYX3), mLHYX_3), mOutLH);
                mOutLH = mul_add(chi, _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(mHLY2X_1, mHLY1X1), mHLY1X_2), mHLY_1X_1), mLHYX2), mLHYX1), mLHYX_1), mLHYX_2), mOutLH);
                mOutLH = mul_add(betaN, mHLY1X_1, mOutLH);
                mOutLH = mul_add(alphaN, mLHYX, mOutLH);
                mOutLH = mul_add(gammaN, _mm_add_ps(mHHYX1, mHHYX_2), mOutLH);
                mOutLH = mul_add(alpha2, mHHYX_1, mOutLH);

                // HH = P1*P1(LL) + P1*(HL) + P1(LH) + 1(HH)
                __m128 mOutHH;
                mOutHH = _mm_mul_ps(epsilonN, _mm_add_ps(_mm_add_ps(_mm_add_ps(mLLY2X2, mLLY2X_1), mLLY_1X2), mLLY_1X_1));
                mOutHH = mul_add(zetaN, _mm_add_ps(_mm_add_ps(_mm_add_ps(mLLY2X1, mLLY1X2), mLLY1X_1), mLLY_1X1), mOutHH);
                mOutHH = mul_add(etaN, mLLY1X1, mOutHH);
                mOutHH = mul_add(thetaN, _mm_add_ps(_mm_add_ps(_mm_add_ps(mHLY2X, mHLY_1X), mLHYX2), mLHYX_1), mOutHH);
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
                bandLHY1 += 4;
                bandHHY1 += 4;
                bandLLY_1 += 4;
                bandHLY_1 += 4;
                bandLHY_1 += 4;
                bandHHY_1 += 4;

                bandLLY2 += 4;
                bandHLY2 += 4;
                bandLLY_2 += 4;
                bandHLY_2 += 4;
                bandLHY_2 += 4;
                bandHHY_2 += 4;

                bandLLY3 += 4;
                bandHLY3 += 4;
                bandLLY_3 += 4;
                bandHLY_3 += 4;
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
            bandLLY_2 += nextTmpBandY;
            bandHLY_2 += nextTmpBandY;
            bandLHY_2 += nextTmpBandY;
            bandHHY_2 += nextTmpBandY;

            bandLLY3 += nextTmpBandY;
            bandHLY3 += nextTmpBandY;
            bandLLY_3 += nextTmpBandY;
            bandHLY_3 += nextTmpBandY;
        }
    }
    //#pragma omp barrier
}

TRANSFORM_STEP_FUNC NO_TREE_VECTORIZE dd137_non_separable_convolution_star_amd64_sse(size_t step)
{
    (void)(step);
    return dd137_non_separable_convolution_star_amd64_sse_all;
}

void NO_TREE_VECTORIZE dd137_non_separable_convolution_star_amd64_sse_transform_tile(const TransformStepArguments * tsa)
{
    dd137_non_separable_convolution_star_amd64_sse_all(tsa);
}
#endif
