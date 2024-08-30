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

#define mLL(y, x) (*(float *)(y + x))
#define mHL(y, x) (*(float *)(y + x + 1))
#define mLH(y, x) (*(float *)(y + x))
#define mHH(y, x) (*(float *)(y + x + 1))

#define LOCA_TMP(y, x) (*(float *)(y + x))

#define LOCA(y, x) ((float *)(y + x))

static void NO_TREE_VECTORIZE cdf97_non_separable_polyconvolution_amd64_sse_all(const TransformStepArguments * tsa)
{
    __m128 alpha     = _mm_set1_ps(-1.58613f);
    __m128 alpha2    = _mm_set1_ps(+2.51582f);
    __m128 alpha22   = _mm_set1_ps(-0.0529801f);
    __m128 beta      = _mm_set1_ps(+1.08403f);
    __m128 alpha3    = _mm_set1_ps(+0.0840336f);
    __m128 gamma     = _mm_set1_ps(+1.17513f);
    __m128 delta     = _mm_set1_ps(+0.0910952f);
    __m128 epsilon   = _mm_set1_ps(+0.00706164f);
    __m128 zeta      = _mm_set1_ps(-0.0574322f);
    __m128 eta       = _mm_set1_ps(-0.00445211f);
    __m128 theta     = _mm_set1_ps(-1.71942f);
    __m128 alpha4    = _mm_set1_ps(+0.00280689f);
    __m128 alpha44   = _mm_set1_ps(-0.133289f);

    __m128 alpha4P   = _mm_set1_ps(+0.196698f);
    __m128 betaP     = _mm_set1_ps(+1.39158f);
    __m128 alpha3P   = _mm_set1_ps(+0.391577f);
    __m128 alpha2P   = _mm_set1_ps(+0.443507f);
    __m128 alpha5P   = _mm_set1_ps(+0.779532f);
    __m128 alphaP    = _mm_set1_ps(+0.882911f);
    __m128 gammaP    = _mm_set1_ps(+1.22864f);
    __m128 deltaP    = _mm_set1_ps(+0.617174f);
    __m128 epsilonP  = _mm_set1_ps(+1.93649f);

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

            for (long lx = tileStartXM; lx < tileEndXM; lx += (long)stepX) {

                __m128 mLLYX, mHLYX, mLHYX, mHHYX;
                __m128 mLLY1X, mHLY1X;
                __m128 mLLY_1X, mHLY_1X, mLHY_1X, mHHY_1X;
                if (lx < 0) {
                    mLLYX = reverse(load_packed_LX00(_mm_loadu_ps(LOCA(tileYL, 2)), _mm_loadu_ps(LOCA(tileYL, 6))));
                    mHLYX = reverse(load_packed_HX00(_mm_load_ps((tileYL)), _mm_load_ps(LOCA(tileYL, 4))));
                    mLHYX = reverse(load_packed_LX00(_mm_loadu_ps(LOCA(tileYH, 2)), _mm_loadu_ps(LOCA(tileYH, 6))));
                    mHHYX = reverse(load_packed_HX00(_mm_load_ps((tileYH)), _mm_load_ps(LOCA(tileYH, 4))));

                    mLLY1X = reverse(load_packed_LX00(_mm_loadu_ps(LOCA(tileY1L, 2)), _mm_loadu_ps(LOCA(tileY1L, 6))));
                    mHLY1X = reverse(load_packed_HX00(_mm_load_ps((tileY1L)), _mm_load_ps(LOCA(tileY1L, 4))));

                    mLLY_1X = reverse(load_packed_LX00(_mm_loadu_ps(LOCA(tileY_1L, 2)), _mm_loadu_ps(LOCA(tileY_1L, 6))));
                    mHLY_1X = reverse(load_packed_HX00(_mm_load_ps((tileY_1L)), _mm_load_ps(LOCA(tileY_1L, 4))));
                    mLHY_1X = reverse(load_packed_LX00(_mm_loadu_ps(LOCA(tileY_1H, 2)), _mm_loadu_ps(LOCA(tileY_1H, 6))));
                    mHHY_1X = reverse(load_packed_HX00(_mm_load_ps((tileY_1H)), _mm_load_ps(LOCA(tileY_1H, 4))));
                } else if (lx > (long)(tileSizeX_stepX)) {
                    mLLYX = reverse(load_packed_LX00(_mm_load_ps(LOCA(tileYL, tileSizeX_stepX)), _mm_load_ps(LOCA(tileYL, tileSizeX - 4))));
                    mHLYX = reverse(load_packed_HX00(_mm_loadu_ps(LOCA(tileYL, tileSizeX_stepX - 2)), _mm_loadu_ps(LOCA(tileYL, tileSizeX - 6))));
                    mLHYX = reverse(load_packed_LX00(_mm_load_ps(LOCA(tileYH, tileSizeX_stepX)), _mm_load_ps(LOCA(tileYH, tileSizeX - 4))));
                    mHHYX = reverse(load_packed_HX00(_mm_loadu_ps(LOCA(tileYH, tileSizeX_stepX - 2)), _mm_loadu_ps(LOCA(tileYH, tileSizeX - 6))));

                    mLLY1X = reverse(load_packed_LX00(_mm_load_ps(LOCA(tileY1L, tileSizeX_stepX)), _mm_load_ps(LOCA(tileY1L, tileSizeX - 4))));
                    mHLY1X = reverse(load_packed_HX00(_mm_loadu_ps(LOCA(tileY1L, tileSizeX_stepX - 2)), _mm_loadu_ps(LOCA(tileY1L, tileSizeX - 6))));

                    mLLY_1X = reverse(load_packed_LX00(_mm_load_ps(LOCA(tileY_1L, tileSizeX_stepX)), _mm_load_ps(LOCA(tileY_1L, tileSizeX - 4))));
                    mHLY_1X = reverse(load_packed_HX00(_mm_loadu_ps(LOCA(tileY_1L, tileSizeX_stepX - 2)), _mm_loadu_ps(LOCA(tileY_1L, tileSizeX - 6))));
                    mLHY_1X = reverse(load_packed_LX00(_mm_load_ps(LOCA(tileY_1H, tileSizeX_stepX)), _mm_load_ps(LOCA(tileY_1H, tileSizeX - 4))));
                    mHHY_1X = reverse(load_packed_HX00(_mm_loadu_ps(LOCA(tileY_1H, tileSizeX_stepX - 2)), _mm_loadu_ps(LOCA(tileY_1H, tileSizeX - 6))));
                } else {
                    mLLYX = load_packed_LX00(_mm_load_ps(LOCA(tileYL, lx)), _mm_load_ps(LOCA(tileYL, lx + 4)));
                    mHLYX = load_packed_HX00(_mm_load_ps(LOCA(tileYL, lx)), _mm_load_ps(LOCA(tileYL, lx + 4)));
                    mLHYX = load_packed_LX00(_mm_load_ps(LOCA(tileYH, lx)), _mm_load_ps(LOCA(tileYH, lx + 4)));
                    mHHYX = load_packed_HX00(_mm_load_ps(LOCA(tileYH, lx)), _mm_load_ps(LOCA(tileYH, lx + 4)));

                    mLLY1X = load_packed_LX00(_mm_load_ps(LOCA(tileY1L, lx)), _mm_load_ps(LOCA(tileY1L, lx + 4)));
                    mHLY1X = load_packed_HX00(_mm_load_ps(LOCA(tileY1L, lx)), _mm_load_ps(LOCA(tileY1L, lx + 4)));

                    mLLY_1X = load_packed_LX00(_mm_load_ps(LOCA(tileY_1L, lx)), _mm_load_ps(LOCA(tileY_1L, lx + 4)));
                    mHLY_1X = load_packed_HX00(_mm_load_ps(LOCA(tileY_1L, lx)), _mm_load_ps(LOCA(tileY_1L, lx + 4)));
                    mLHY_1X = load_packed_LX00(_mm_load_ps(LOCA(tileY_1H, lx)), _mm_load_ps(LOCA(tileY_1H, lx + 4)));
                    mHHY_1X = load_packed_HX00(_mm_load_ps(LOCA(tileY_1H, lx)), _mm_load_ps(LOCA(tileY_1H, lx + 4)));
                }

                __m128 mLLYX1, mLHYX1, mLLY_1X1, mLHY_1X1, mLLY1X1;
                if (lx < (long)(tileSizeX_stepX)) {
                    mLLYX1 = load_packed_LX01(mLLYX, _mm_load_ps(LOCA(tileYL, lx + 8)));
                    mLHYX1 = load_packed_LX01(mLHYX, _mm_load_ps(LOCA(tileYH, lx + 8)));

                    mLLY1X1 = load_packed_LX01(mLLY1X, _mm_load_ps(LOCA(tileY1L, lx + 8)));

                    mLLY_1X1 = load_packed_LX01(mLLY_1X, _mm_load_ps(LOCA(tileY_1L, lx + 8)));
                    mLHY_1X1 = load_packed_LX01(mLHY_1X, _mm_load_ps(LOCA(tileY_1H, lx + 8)));
                }  else {
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
                __m128 mOutLL = mLLYX;
                mOutLL = _mm_mul_ps(gamma, mLLYX);
                mOutLL = mul_add(delta, _mm_add_ps(mLLY_1X, mLLYX_1), mOutLL);
                mOutLL = mul_add(epsilon, mLLY_1X_1, mOutLL);
                mOutLL = mul_add(zeta, _mm_add_ps(mHLYX_1, mLHY_1X), mOutLL);
                mOutLL = mul_add(eta, _mm_add_ps(mHLY_1X_1, mLHY_1X_1), mOutLL);
                mOutLL = mul_add(alpha4, mHHY_1X_1, mOutLL);

                // HL = V@*P(LL) + V@*(HL) + U1*P(LH) + U1*(HH)
                __m128 mOutHL = mHLYX;
                mOutHL = _mm_mul_ps(theta, _mm_add_ps(mLLYX1, mLLYX));
                mOutHL = mul_add(alpha44, _mm_add_ps(mLLY_1X1, mLLY_1X), mOutHL);
                mOutHL = mul_add(beta, mHLYX, mOutHL);
                mOutHL = mul_add(alpha3, _mm_add_ps(_mm_add_ps(mLHY_1X, mLHY_1X1), mHLY_1X), mOutHL);
                mOutHL = mul_add(alpha22, mHHY_1X, mOutHL);

                // LH = P*V@(LL) + P*U1(HL) + V@(LH) + U1(HH)
                __m128 mOutLH = mLHYX;
                mOutLH = _mm_mul_ps(theta, _mm_add_ps(mLLY1X, mLLYX));
                mOutLH = mul_add(alpha44, _mm_add_ps(mLLY1X_1, mLLYX_1), mOutLH);
                mOutLH = mul_add(alpha3, _mm_add_ps(_mm_add_ps(mHLYX_1, mHLY1X_1), mLHYX_1), mOutLH);
                mOutLH = mul_add(beta, mLHYX, mOutLH);
                mOutLH = mul_add(alpha22, mHHYX_1, mOutLH);

                // HH = P*P(LL) + P*(HL) + P(LH) + 1(HH)
                __m128 mOutHH = mHHYX;
                mOutHH = _mm_mul_ps(alpha2, _mm_add_ps(_mm_add_ps(mLLY1X1, mLLYX1), _mm_add_ps(mLLY1X, mLLYX)));
                mOutHH = mul_add(alpha, _mm_add_ps(_mm_add_ps(mHLYX, mHLY1X), _mm_add_ps(mLHYX, mLHYX1)), mOutHH);
                mOutHH = _mm_add_ps(mHHYX, mOutHH);

                // S^H_{U0}
                // LL = 1(LL) + U0(HL) + 0(LH) + 0(HH)
                mOutLL = mul_add(alpha22, mOutHL, mOutLL);
                // HL = 0(LL) + 1(HL) + 0(LH) + 0(HH)

                // LH = 0(LL) + 0(HL) + 1(LH) + U0(HH)
                mOutLH = mul_add(alpha22, mOutHH, mOutLH);
                // HH = 0(LL) + 0(HL) + 0(LH) +  1(HH)

                // S^V_{U0}
                // LL = 1(LL) + 0(HL) + U0*(LH) + 0(HH)
                mOutLL = mul_add(alpha22, mOutLH, mOutLL);
                // HL = 0(LL) + 1(HL) + 0(LH) + U0*(HH)
                mOutHL = mul_add(alpha22, mOutHH, mOutHL);
                // LH = 0(LL) + 0(HL) + 1(LH) + 0(HH)

                // HH = 0(LL) + 0(HL) + 0(LH) + 1(HH)

                // T^H_{P0}:
                // LL = 1(LL) + 0(HL) + 0(LH) + 0(HH)

                // HL = P0(LL) + 1(HL) + 0(LH) + 0(HH)
                mOutHL = mul_add(alphaP, mOutLL, mOutHL);
                // LH = 0(LL) + 0(HL) + 1(LH) + 0(HH)

                // HH = 0(LL) + 0(HL) + P0(LH) + 0(HH)
                mOutHH = mul_add(alphaP, mOutLH, mOutHH);

                // T^V_{P0}:
                // LL = 1(LL) + 0(HL) + 0(LH) + 0(HH)

                // HL = 0(LL) + 1(HL) + 0(LH) + 0(HH)

                // LH = P0*(LL) + 0(HL) + 1(LH) + 0(HH)
                mOutLH = mul_add(alphaP, mOutLL, mOutLH);
                // HH = 0(LL) + P0*(HL) + 0(LH) + 1(HH)
                mOutHH = mul_add(alphaP, mOutHL, mOutHH);
                //}

                _mm_store_ps(tmpLL, mOutLL);
                _mm_store_ps(tmpHL, mOutHL);
                _mm_store_ps(tmpLH, mOutLH);
                _mm_store_ps(tmpHH, mOutHH);


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

                __m128 mLLYX1 = _mm_loadu_ps((tmpLL + 1));
                __m128 mLHYX1 = _mm_loadu_ps((tmpLH + 1));

                __m128 mLLY1X1 = _mm_loadu_ps((bandLLY1 + 1));

                __m128 mLHY_1X1 = _mm_loadu_ps((bandLHY_1 + 1));

                __m128 mHLYX_1 = _mm_loadu_ps((tmpHL - 1));
                __m128 mHHYX_1 = _mm_loadu_ps((tmpHH - 1));

                __m128 mHLY1X_1 = _mm_loadu_ps((bandHLY1 - 1));

                __m128 mHHY_1X_1 = _mm_loadu_ps((bandHHY_1 - 1));

                // N_{P1,U1}
                // LL = V1*V1(LL) + V1*U1(HL) + U1*V1(LH) + U1*U1(HH)
                __m128 mOutLL;
                mOutLL = _mm_mul_ps(epsilonP, mLLYX);
                mOutLL = mul_add(deltaP, _mm_add_ps(mHLYX_1, mLHY_1X), mOutLL);
                mOutLL = mul_add(alpha4P, mHHY_1X_1, mOutLL);

                // HL = V1*P1(LL) + V1*(HL) + U1*P1(LH) + U1*(HH)
                __m128 mOutHL;
                mOutHL = _mm_mul_ps(gammaP, mLLYX1);
                mOutHL = mul_add(betaP, mHLYX, mOutHL);
                mOutHL = mul_add(alpha3P, mLHY_1X1, mOutHL);
                mOutHL = mul_add(alpha2P, mHHY_1X, mOutHL);

                // LH = P1*V1(LL) + P1*U1(HL) + V1(LH) + U1(HH)
                __m128 mOutLH;
                mOutLH = _mm_mul_ps(gammaP, mLLY1X);
                mOutLH = mul_add(alpha3P, mHLY1X_1, mOutLH);
                mOutLH = mul_add(betaP, mLHYX, mOutLH);
                mOutLH = mul_add(alpha2P, mHHYX_1, mOutLH);

                // HH = P1*P1(LL) + P1*(HL) + P1(LH) + 1(HH)
                __m128 mOutHH;
                mOutHH = _mm_mul_ps(alpha5P, mLLY1X1);
                mOutHH = mul_add(alphaP, _mm_add_ps(mHLY1X, mLHYX1), mOutHH);
                mOutHH = _mm_add_ps(mHHYX, mOutHH);

                // S^H_{U0}
                // LL = 1(LL) + U0(HL) + 0(LH) + 0(HH)
                //mOutLL = mul_add(alpha2P, mOutHL, mOutLL);
                // HL = 0(LL) + 1(HL) + 0(LH) + 0(HH)

                // LH = 0(LL) + 0(HL) + 1(LH) + U0(HH)
                mOutLH = mul_add(alpha2P, mOutHH, mOutLH);
                // HH = 0(LL) + 0(HL) + 0(LH) +  1(HH)

                // S^V_{U0}
                // LL = 1(LL) + 0(HL) + U0*(LH) + 0(HH)
                mOutLL = mul_add(alpha2P, _mm_add_ps(mOutLH, mOutHL), mOutLL);
                // HL = 0(LL) + 1(HL) + 0(LH) + U0*(HH)
                mOutHL = mul_add(alpha2P, mOutHH, mOutHL);
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

TRANSFORM_STEP_FUNC NO_TREE_VECTORIZE cdf97_non_separable_polyconvolution_amd64_sse(size_t step)
{
    (void)(step);
    return cdf97_non_separable_polyconvolution_amd64_sse_all;
}

void NO_TREE_VECTORIZE cdf97_non_separable_polyconvolution_amd64_sse_transform_tile(const TransformStepArguments * tsa)
{
    cdf97_non_separable_polyconvolution_amd64_sse_all(tsa);
}
#endif
