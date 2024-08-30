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

static void NO_TREE_VECTORIZE cdf97_non_separable_convolution_at_amd64_sse_all(const TransformStepArguments * tsa)
{
    __m128 alpha     = _mm_set1_ps(+0.0136748f);
    __m128 beta      = _mm_set1_ps(+0.111665f);
    __m128 gamma     = _mm_set1_ps(-0.0151008f);
    __m128 delta     = _mm_set1_ps(+0.00384796f);
    __m128 epsilon   = _mm_set1_ps(+0.911834f);
    __m128 zeta      = _mm_set1_ps(+0.0314216f);
    __m128 eta       = _mm_set1_ps(-0.12331f);
    __m128 theta     = _mm_set1_ps(+0.0166756f);
    __m128 iota      = _mm_set1_ps(-0.00424923f);
    __m128 kappa     = _mm_set1_ps(+0.00108278f);
    __m128 lambda    = _mm_set1_ps(-0.00862145f);
    __m128 mu        = _mm_set1_ps(-0.0704009f);
    __m128 nu        = _mm_set1_ps(+0.00952051f);
    __m128 xi        = _mm_set1_ps(-0.002426f);
    __m128 omicron   = _mm_set1_ps(+0.0408159f);
    __m128 pi        = _mm_set1_ps(+0.333294f);
    __m128 rho       = _mm_set1_ps(-0.0198102f);
    __m128 sigma     = _mm_set1_ps(-0.0450723f);
    __m128 tau       = _mm_set1_ps(+0.00267898f);
    __m128 upsilon   = _mm_set1_ps(+0.0114852f);
    __m128 phi       = _mm_set1_ps(-0.000682654f);
    __m128 chi       = _mm_set1_ps(+0.00543551f);

    __m128 alpha2    = _mm_set1_ps(-0.0257329f);
    __m128 beta2     = _mm_set1_ps(+0.0015295f);
    __m128 gamma2    = _mm_set1_ps(+0.121826f);
    __m128 delta2    = _mm_set1_ps(-0.00724101f);
    __m128 epsilon2  = _mm_set1_ps(+0.000430388f);
    __m128 zeta2     = _mm_set1_ps(+0.00867621f);
    __m128 eta2      = _mm_set1_ps(+0.070848f);
    __m128 theta2    = _mm_set1_ps(-0.00958098f);
    __m128 iota2     = _mm_set1_ps(+0.00244141f);
    __m128 kappa2    = _mm_set1_ps(-0.0562057f);
    __m128 lambda2   = _mm_set1_ps(-0.458963f);
    __m128 mu2       = _mm_set1_ps(+0.062067f);
    __m128 nu2       = _mm_set1_ps(-0.0158158f);
    __m128 xi2       = _mm_set1_ps(-0.00547003f);
    __m128 omicron2  = _mm_set1_ps(-0.0446671f);
    __m128 pi2       = _mm_set1_ps(+0.00604046f);
    __m128 rho2      = _mm_set1_ps(-0.00153922f);
    __m128 sigma2    = _mm_set1_ps(+0.105999f);
    __m128 tau2      = _mm_set1_ps(+0.865566f);
    __m128 upsilon2  = _mm_set1_ps(-0.117053f);
    __m128 phi2      = _mm_set1_ps(+0.0298272f);
    __m128 chi2      = _mm_set1_ps(+0.0258964f);

    __m128 alpha3    = _mm_set1_ps(+0.0354357f);
    __m128 beta3     = _mm_set1_ps(-0.16776f);
    __m128 gamma3    = _mm_set1_ps(+0.00997127f);
    __m128 delta3    = _mm_set1_ps(+0.00344866f);
    __m128 epsilon3  = _mm_set1_ps(-0.0163267f);
    __m128 zeta3     = _mm_set1_ps(+0.000970421f);
    __m128 eta3      = _mm_set1_ps(-0.0668286f);
    __m128 theta3    = _mm_set1_ps(+0.316382f);
    __m128 iota3     = _mm_set1_ps(-0.018805f);
    __m128 kappa3    = _mm_set1_ps(+0.00550478f);
    __m128 lambda3   = _mm_set1_ps(-0.0356607f);
    __m128 mu3       = _mm_set1_ps(+0.231015f);
    __m128 nu3       = _mm_set1_ps(-0.00347056f);
    __m128 xi3       = _mm_set1_ps(+0.0224828f);
    __m128 omicron3  = _mm_set1_ps(+0.0672531f);
    __m128 pi3       = _mm_set1_ps(-0.435675f);
    __m128 rho3      = _mm_set1_ps(+0.00218806f);
    __m128 sigma3    = _mm_set1_ps(-0.0424006f);
    __m128 tau3      = _mm_set1_ps(+0.821645f);

    __m128 upsilon3  = _mm_set1_ps(+0.4435068520439f);

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

    // M, S^H_{U0}, S^V_{U0}
    {
        for (size_t ly = tileStartY; ly < tileEndY; ly += 2) {
            size_t ly1L = mirr((long)ly + 2, tileSizeY);
            size_t ly1H = mirr((long)ly + 3, tileSizeY);
            size_t ly_1L = mirr((long)ly - 2, tileSizeY);
            size_t ly_1H = mirr((long)ly - 1, tileSizeY);

            size_t ly2L = mirr((long)ly + 4, tileSizeY);
            size_t ly_2L = mirr((long)ly - 4, tileSizeY);
            size_t ly_2H = mirr((long)ly - 3, tileSizeY);

            float *tileY1L = tile + ly1L * tileStrideY;
            float *tileY1H = tile + ly1H * tileStrideY;
            float *tileY_1L = tile + ly_1L * tileStrideY;
            float *tileY_1H = tile + ly_1H * tileStrideY;

            float *tileY2L = tile + ly2L * tileStrideY;
            float *tileY_2L = tile + ly_2L * tileStrideY;
            float *tileY_2H = tile + ly_2H * tileStrideY;

            for (size_t lx = tileStartX; lx < tileEndX; lx += stepX) {
                __m128 mLLYX = load_packed_LX00(_mm_load_ps(LOCA(tileYL, lx)), _mm_load_ps(LOCA(tileYL, lx + 4)));
                __m128 mHLYX = load_packed_HX00(_mm_load_ps(LOCA(tileYL, lx)), _mm_load_ps(LOCA(tileYL, lx + 4)));
                __m128 mLHYX = load_packed_LX00(_mm_load_ps(LOCA(tileYH, lx)), _mm_load_ps(LOCA(tileYH, lx + 4)));
                __m128 mHHYX = load_packed_HX00(_mm_load_ps(LOCA(tileYH, lx)), _mm_load_ps(LOCA(tileYH, lx + 4)));

                __m128 mLLY1X = load_packed_LX00(_mm_load_ps(LOCA(tileY1L, lx)), _mm_load_ps(LOCA(tileY1L, lx + 4)));
                __m128 mHLY1X = load_packed_HX00(_mm_load_ps(LOCA(tileY1L, lx)), _mm_load_ps(LOCA(tileY1L, lx + 4)));
                __m128 mLHY1X = load_packed_LX00(_mm_load_ps(LOCA(tileY1H, lx)), _mm_load_ps(LOCA(tileY1H, lx + 4)));
                __m128 mHHY1X = load_packed_HX00(_mm_load_ps(LOCA(tileY1H, lx)), _mm_load_ps(LOCA(tileY1H, lx + 4)));

                __m128 mLLY_1X = load_packed_LX00(_mm_load_ps(LOCA(tileY_1L, lx)), _mm_load_ps(LOCA(tileY_1L, lx + 4)));
                __m128 mHLY_1X = load_packed_HX00(_mm_load_ps(LOCA(tileY_1L, lx)), _mm_load_ps(LOCA(tileY_1L, lx + 4)));
                __m128 mLHY_1X = load_packed_LX00(_mm_load_ps(LOCA(tileY_1H, lx)), _mm_load_ps(LOCA(tileY_1H, lx + 4)));
                __m128 mHHY_1X = load_packed_HX00(_mm_load_ps(LOCA(tileY_1H, lx)), _mm_load_ps(LOCA(tileY_1H, lx + 4)));

                __m128 mLLY2X = load_packed_LX00(_mm_load_ps(LOCA(tileY2L, lx)), _mm_load_ps(LOCA(tileY2L, lx + 4)));
                __m128 mHLY2X = load_packed_HX00(_mm_load_ps(LOCA(tileY2L, lx)), _mm_load_ps(LOCA(tileY2L, lx + 4)));

                __m128 mLLY_2X = load_packed_LX00(_mm_load_ps(LOCA(tileY_2L, lx)), _mm_load_ps(LOCA(tileY_2L, lx + 4)));
                __m128 mHLY_2X = load_packed_HX00(_mm_load_ps(LOCA(tileY_2L, lx)), _mm_load_ps(LOCA(tileY_2L, lx + 4)));
                __m128 mLHY_2X = load_packed_LX00(_mm_load_ps(LOCA(tileY_2H, lx)), _mm_load_ps(LOCA(tileY_2H, lx + 4)));
                __m128 mHHY_2X = load_packed_HX00(_mm_load_ps(LOCA(tileY_2H, lx)), _mm_load_ps(LOCA(tileY_2H, lx + 4)));

                __m128 mLLYX1, mHLYX1, mLHYX1, mHHYX1;
                __m128 mLLY1X1, mHLY1X1, mLHY1X1, mHHY1X1;
                __m128 mLLY_1X1, mHLY_1X1, mLHY_1X1, mHHY_1X1;
                __m128 mLLY2X1, mHLY2X1;
                __m128 mLLY_2X1, mHLY_2X1, mLHY_2X1, mHHY_2X1;
                __m128 mLLYX2, mLHYX2;
                __m128 mLLY1X2, mLHY1X2;
                __m128 mLLY_1X2, mLHY_1X2;
                __m128 mLLY2X2;
                __m128 mLLY_2X2, mLHY_2X2;

                if (lx < tileSizeX - stepX) {
                    mLLYX1 = load_packed_LX01(mLLYX, _mm_load_ps(LOCA(tileYL, lx + 8)));
                    mHLYX1 = load_packed_HX01(mHLYX, _mm_load_ps(LOCA(tileYL, lx + 8)));
                    mLHYX1 = load_packed_LX01(mLHYX, _mm_load_ps(LOCA(tileYH, lx + 8)));
                    mHHYX1 = load_packed_HX01(mHHYX, _mm_load_ps(LOCA(tileYH, lx + 8)));

                    mLLY1X1 = load_packed_LX01(mLLY1X, _mm_load_ps(LOCA(tileY1L, lx + 8)));
                    mHLY1X1 = load_packed_HX01(mHLY1X, _mm_load_ps(LOCA(tileY1L, lx + 8)));
                    mLHY1X1 = load_packed_LX01(mLHY1X, _mm_load_ps(LOCA(tileY1H, lx + 8)));
                    mHHY1X1 = load_packed_HX01(mHHY1X, _mm_load_ps(LOCA(tileY1H, lx + 8)));

                    mLLY_1X1 = load_packed_LX01(mLLY_1X, _mm_load_ps(LOCA(tileY_1L, lx + 8)));
                    mHLY_1X1 = load_packed_HX01(mHLY_1X, _mm_load_ps(LOCA(tileY_1L, lx + 8)));
                    mLHY_1X1 = load_packed_LX01(mLHY_1X, _mm_load_ps(LOCA(tileY_1H, lx + 8)));
                    mHHY_1X1 = load_packed_HX01(mHHY_1X, _mm_load_ps(LOCA(tileY_1H, lx + 8)));

                    mLLY2X1 = load_packed_LX01(mLLY2X, _mm_load_ps(LOCA(tileY2L, lx + 8)));
                    mHLY2X1 = load_packed_HX01(mHLY2X, _mm_load_ps(LOCA(tileY2L, lx + 8)));

                    mLLY_2X1 = load_packed_LX01(mLLY_2X, _mm_load_ps(LOCA(tileY_2L, lx + 8)));
                    mHLY_2X1 = load_packed_HX01(mHLY_2X, _mm_load_ps(LOCA(tileY_2L, lx + 8)));
                    mLHY_2X1 = load_packed_LX01(mLHY_2X, _mm_load_ps(LOCA(tileY_2H, lx + 8)));
                    mHHY_2X1 = load_packed_HX01(mHHY_2X, _mm_load_ps(LOCA(tileY_2H, lx + 8)));

                    mLLYX2 = load_packed_LX02(mLLYX1, _mm_load_ps(LOCA(tileYL, lx + 8)));

                    mLHYX2 = load_packed_LX02(mLHYX1, _mm_load_ps(LOCA(tileYH, lx + 8)));

                    mLLY1X2 = load_packed_LX02(mLLY1X1, _mm_load_ps(LOCA(tileY1L, lx + 8)));

                    mLHY1X2 = load_packed_LX02(mLHY1X1, _mm_load_ps(LOCA(tileY1H, lx + 8)));

                    mLLY_1X2 = load_packed_LX02(mLLY_1X1, _mm_load_ps(LOCA(tileY_1L, lx + 8)));

                    mLHY_1X2 = load_packed_LX02(mLHY_1X1, _mm_load_ps(LOCA(tileY_1H, lx + 8)));

                    mLLY2X2 = load_packed_LX02(mLLY2X1, _mm_load_ps(LOCA(tileY2L, lx + 8)));

                    mLLY_2X2 = load_packed_LX02(mLLY_2X1, _mm_load_ps(LOCA(tileY_2L, lx + 8)));

                    mLHY_2X2 = load_packed_LX02(mLHY_2X1, _mm_load_ps(LOCA(tileY_2H, lx + 8)));
                } else {
                    mLLYX1 = load_unpacked_LX01S(mLLYX);
                    mHLYX1 = load_unpacked_HX01S(mHLYX);
                    mLHYX1 = load_unpacked_LX01S(mLHYX);
                    mHHYX1 = load_unpacked_HX01S(mHHYX);

                    mLLY1X1 = load_unpacked_LX01S(mLLY1X);
                    mHLY1X1 = load_unpacked_HX01S(mHLY1X);
                    mLHY1X1 = load_unpacked_LX01S(mLHY1X);
                    mHHY1X1 = load_unpacked_HX01S(mHHY1X);

                    mLLY_1X1 = load_unpacked_LX01S(mLLY_1X);
                    mHLY_1X1 = load_unpacked_HX01S(mHLY_1X);
                    mLHY_1X1 = load_unpacked_LX01S(mLHY_1X);
                    mHHY_1X1 = load_unpacked_HX01S(mHHY_1X);

                    mLLY2X1 = load_unpacked_LX01S(mLLY2X);
                    mHLY2X1 = load_unpacked_HX01S(mHLY2X);

                    mLLY_2X1 = load_unpacked_LX01S(mLLY_2X);
                    mHLY_2X1 = load_unpacked_HX01S(mHLY_2X);
                    mLHY_2X1 = load_unpacked_LX01S(mLHY_2X);
                    mHHY_2X1 = load_unpacked_HX01S(mHHY_2X);

                    mLLYX2 = load_unpacked_LX02S(mLLYX);

                    mLHYX2 = load_unpacked_LX02S(mLHYX);

                    mLLY1X2 = load_unpacked_LX02S(mLLY1X);

                    mLHY1X2 = load_unpacked_LX02S(mLHY1X);

                    mLLY_1X2 = load_unpacked_LX02S(mLLY_1X);

                    mLHY_1X2 = load_unpacked_LX02S(mLHY_1X);

                    mLLY2X2 = load_unpacked_LX02S(mLLY2X);

                    mLLY_2X2 = load_unpacked_LX02S(mLLY_2X);

                    mLHY_2X2 = load_unpacked_LX02S(mLHY_2X);
                }

                __m128 mLLYX_1, mHLYX_1, mLHYX_1, mHHYX_1;
                __m128 mLLY1X_1, mHLY1X_1, mLHY1X_1, mHHY1X_1;
                __m128 mLLY_1X_1, mHLY_1X_1, mLHY_1X_1, mHHY_1X_1;
                __m128 mLLY2X_1, mHLY2X_1;
                __m128 mLLY_2X_1, mHLY_2X_1, mLHY_2X_1, mHHY_2X_1;
                __m128 mLLYX_2, mHLYX_2, mLHYX_2, mHHYX_2;
                __m128 mLLY1X_2, mHLY1X_2, mLHY1X_2, mHHY1X_2;
                __m128 mLLY_1X_2, mHLY_1X_2, mLHY_1X_2, mHHY_1X_2;
                __m128 mLLY2X_2, mHLY2X_2;
                __m128 mLLY_2X_2, mHLY_2X_2, mLHY_2X_2, mHHY_2X_2;

                if (lx > 0) {
                    mLLYX_1 = load_packed_LX01L(mLLYX, _mm_load_ps(LOCA(tileYL, lx - 4)));
                    mHLYX_1 = load_packed_HX01L(mHLYX, _mm_load_ps(LOCA(tileYL, lx - 4)));
                    mLHYX_1 = load_packed_LX01L(mLHYX, _mm_load_ps(LOCA(tileYH, lx - 4)));
                    mHHYX_1 = load_packed_HX01L(mHHYX, _mm_load_ps(LOCA(tileYH, lx - 4)));

                    mLLY1X_1 = load_packed_LX01L(mLLY1X, _mm_load_ps(LOCA(tileY1L, lx - 4)));
                    mHLY1X_1 = load_packed_HX01L(mHLY1X, _mm_load_ps(LOCA(tileY1L, lx - 4)));
                    mLHY1X_1 = load_packed_LX01L(mLHY1X, _mm_load_ps(LOCA(tileY1H, lx - 4)));
                    mHHY1X_1 = load_packed_HX01L(mHHY1X, _mm_load_ps(LOCA(tileY1H, lx - 4)));

                    mLLY_1X_1 = load_packed_LX01L(mLLY_1X, _mm_load_ps(LOCA(tileY_1L, lx - 4)));
                    mHLY_1X_1 = load_packed_HX01L(mHLY_1X, _mm_load_ps(LOCA(tileY_1L, lx - 4)));
                    mLHY_1X_1 = load_packed_LX01L(mLHY_1X, _mm_load_ps(LOCA(tileY_1H, lx - 4)));
                    mHHY_1X_1 = load_packed_HX01L(mHHY_1X, _mm_load_ps(LOCA(tileY_1H, lx - 4)));

                    mLLY2X_1 = load_packed_LX01L(mLLY2X, _mm_load_ps(LOCA(tileY2L, lx - 4)));
                    mHLY2X_1 = load_packed_HX01L(mHLY2X, _mm_load_ps(LOCA(tileY2L, lx - 4)));

                    mLLY_2X_1 = load_packed_LX01L(mLLY_2X, _mm_load_ps(LOCA(tileY_2L, lx - 4)));
                    mHLY_2X_1 = load_packed_HX01L(mHLY_2X, _mm_load_ps(LOCA(tileY_2L, lx - 4)));
                    mLHY_2X_1 = load_packed_LX01L(mLHY_2X, _mm_load_ps(LOCA(tileY_2H, lx - 4)));
                    mHHY_2X_1 = load_packed_HX01L(mHHY_2X, _mm_load_ps(LOCA(tileY_2H, lx - 4)));

                    mLLYX_2 = load_packed_LX02L(mLLYX_1, _mm_load_ps(LOCA(tileYL, lx - 4)));
                    mHLYX_2 = load_packed_HX02L(mHLYX_1, _mm_load_ps(LOCA(tileYL, lx - 4)));
                    mLHYX_2 = load_packed_LX02L(mLHYX_1, _mm_load_ps(LOCA(tileYH, lx - 4)));
                    mHHYX_2 = load_packed_HX02L(mHHYX_1, _mm_load_ps(LOCA(tileYH, lx - 4)));

                    mLLY1X_2 = load_packed_LX02L(mLLY1X_1, _mm_load_ps(LOCA(tileY1L, lx - 4)));
                    mHLY1X_2 = load_packed_HX02L(mHLY1X_1, _mm_load_ps(LOCA(tileY1L, lx - 4)));
                    mLHY1X_2 = load_packed_LX02L(mLHY1X_1, _mm_load_ps(LOCA(tileY1H, lx - 4)));
                    mHHY1X_2 = load_packed_HX02L(mHHY1X_1, _mm_load_ps(LOCA(tileY1H, lx - 4)));

                    mLLY_1X_2 = load_packed_LX02L(mLLY_1X_1, _mm_load_ps(LOCA(tileY_1L, lx - 4)));
                    mHLY_1X_2 = load_packed_HX02L(mHLY_1X_1, _mm_load_ps(LOCA(tileY_1L, lx - 4)));
                    mLHY_1X_2 = load_packed_LX02L(mLHY_1X_1, _mm_load_ps(LOCA(tileY_1H, lx - 4)));
                    mHHY_1X_2 = load_packed_HX02L(mHHY_1X_1, _mm_load_ps(LOCA(tileY_1H, lx - 4)));

                    mLLY2X_2 = load_packed_LX02L(mLLY2X_1, _mm_load_ps(LOCA(tileY2L, lx - 4)));
                    mHLY2X_2 = load_packed_HX02L(mHLY2X_1, _mm_load_ps(LOCA(tileY2L, lx - 4)));

                    mLLY_2X_2 = load_packed_LX02L(mLLY_2X_1, _mm_load_ps(LOCA(tileY_2L, lx - 4)));
                    mHLY_2X_2 = load_packed_HX02L(mHLY_2X_1, _mm_load_ps(LOCA(tileY_2L, lx - 4)));
                    mLHY_2X_2 = load_packed_LX02L(mLHY_2X_1, _mm_load_ps(LOCA(tileY_2H, lx - 4)));
                    mHHY_2X_2 = load_packed_HX02L(mHHY_2X_1, _mm_load_ps(LOCA(tileY_2H, lx - 4)));
                } else {
                    mLLYX_1 = load_unpacked_LX01LS(mLLYX);
                    mHLYX_1 = load_unpacked_HX01LS(mHLYX);
                    mLHYX_1 = load_unpacked_LX01LS(mLHYX);
                    mHHYX_1 = load_unpacked_HX01LS(mHHYX);

                    mLLY1X_1 = load_unpacked_LX01LS(mLLY1X);
                    mHLY1X_1 = load_unpacked_HX01LS(mHLY1X);
                    mLHY1X_1 = load_unpacked_LX01LS(mLHY1X);
                    mHHY1X_1 = load_unpacked_HX01LS(mHHY1X);

                    mLLY_1X_1 = load_unpacked_LX01LS(mLLY_1X);
                    mHLY_1X_1 = load_unpacked_HX01LS(mHLY_1X);
                    mLHY_1X_1 = load_unpacked_LX01LS(mLHY_1X);
                    mHHY_1X_1 = load_unpacked_HX01LS(mHHY_1X);

                    mLLY2X_1 = load_unpacked_LX01LS(mLLY2X);
                    mHLY2X_1 = load_unpacked_HX01LS(mHLY2X);

                    mLLY_2X_1 = load_unpacked_LX01LS(mLLY_2X);
                    mHLY_2X_1 = load_unpacked_HX01LS(mHLY_2X);
                    mLHY_2X_1 = load_unpacked_LX01LS(mLHY_2X);
                    mHHY_2X_1 = load_unpacked_HX01LS(mHHY_2X);

                    mLLYX_2 = load_unpacked_LX02LS(mLLYX);
                    mHLYX_2 = load_unpacked_HX02LS(mHLYX);
                    mLHYX_2 = load_unpacked_LX02LS(mLHYX);
                    mHHYX_2 = load_unpacked_HX02LS(mHHYX);

                    mLLY1X_2 = load_unpacked_LX02LS(mLLY1X);
                    mHLY1X_2 = load_unpacked_HX02LS(mHLY1X);
                    mLHY1X_2 = load_unpacked_LX02LS(mLHY1X);
                    mHHY1X_2 = load_unpacked_HX02LS(mHHY1X);

                    mLLY_1X_2 = load_unpacked_LX02LS(mLLY_1X);
                    mHLY_1X_2 = load_unpacked_HX02LS(mHLY_1X);
                    mLHY_1X_2 = load_unpacked_LX02LS(mLHY_1X);
                    mHHY_1X_2 = load_unpacked_HX02LS(mHHY_1X);

                    mLLY2X_2 = load_unpacked_LX02LS(mLLY2X);
                    mHLY2X_2 = load_unpacked_HX02LS(mHLY2X);

                    mLLY_2X_2 = load_unpacked_LX02LS(mLLY_2X);
                    mHLY_2X_2 = load_unpacked_HX02LS(mHLY_2X);
                    mLHY_2X_2 = load_unpacked_LX02LS(mLHY_2X);
                    mHHY_2X_2 = load_unpacked_HX02LS(mHHY_2X);
                }


                // M
                // LL = M00(LL) + M01(HL) + M02(LH) + M03(HH)
                __m128 mOutLL;
                mOutLL = _mm_mul_ps(alpha, mLLY1X1);
                mOutLL = mul_add(beta, _mm_add_ps(mLLYX1, mLLY1X), mOutLL);
                mOutLL = mul_add(gamma, _mm_add_ps(mLLY_1X1, mLLY1X_1), mOutLL);
                mOutLL = mul_add(delta, _mm_add_ps(mLLY_2X1, mLLY1X_2), mOutLL);
                mOutLL = mul_add(epsilon, mLLYX, mOutLL);
                mOutLL = mul_add(eta, _mm_add_ps(mLLY_1X, mLLYX_1), mOutLL);
                mOutLL = mul_add(zeta, _mm_add_ps(mLLY_2X, mLLYX_2), mOutLL);
                mOutLL = mul_add(theta, mLLY_1X_1, mOutLL);
                mOutLL = mul_add(iota, _mm_add_ps(mLLY_2X_1, mLLY_1X_2), mOutLL);
                mOutLL = mul_add(kappa, mLLY_2X_2, mOutLL);
                mOutLL = mul_add(lambda, _mm_add_ps(mHLY1X, mLHYX1), mOutLL);
                mOutLL = mul_add(mu, _mm_add_ps(mHLYX, mLHYX), mOutLL);
                mOutLL = mul_add(nu, _mm_add_ps(mHLY_1X, mLHYX_1), mOutLL);
                mOutLL = mul_add(xi, _mm_add_ps(_mm_add_ps(_mm_add_ps(mHLY_2X, mHLY1X_2), mLHY_2X1), mLHYX_2), mOutLL);
                mOutLL = mul_add(omicron, _mm_add_ps(mHLY1X_1, mLHY_1X1), mOutLL);
                mOutLL = mul_add(pi, _mm_add_ps(mHLYX_1, mLHY_1X), mOutLL);
                mOutLL = mul_add(sigma, _mm_add_ps(mHLY_1X_1, mLHY_1X_1), mOutLL);
                mOutLL = mul_add(upsilon, _mm_add_ps(mHLY_2X_1, mLHY_1X_2), mOutLL);
                mOutLL = mul_add(rho, _mm_add_ps(mHLYX_2, mLHY_2X), mOutLL);
                mOutLL = mul_add(tau, _mm_add_ps(mHLY_1X_2, mLHY_2X_1), mOutLL);
                mOutLL = mul_add(phi, _mm_add_ps(mHLY_2X_2, mLHY_2X_2), mOutLL);
                mOutLL = mul_add(chi, mHHYX, mOutLL);
                mOutLL = mul_add(alpha2, _mm_add_ps(mHHY_1X, mHHYX_1), mOutLL);
                mOutLL = mul_add(beta2, _mm_add_ps(mHHY_2X, mHHYX_2), mOutLL);
                mOutLL = mul_add(gamma2, mHHY_1X_1, mOutLL);
                mOutLL = mul_add(delta2, _mm_add_ps(mHHY_2X_1, mHHY_1X_2), mOutLL);
                mOutLL = mul_add(epsilon2, mHHY_2X_2, mOutLL);

                // HL = M10(LL) + M11(HL) + M12(LH) + M13(HH)
                __m128 mOutHL;
                mOutHL = _mm_mul_ps(zeta2, _mm_add_ps(mLLY1X2, mLLY1X_1));
                mOutHL = mul_add(eta2, _mm_add_ps(mLLYX2, mLLYX_1), mOutHL);
                mOutHL = mul_add(theta2, _mm_add_ps(mLLY_1X2, mLLY_1X_1), mOutHL);
                mOutHL = mul_add(iota2, _mm_add_ps(mLLY_2X2, mLLY_2X_1), mOutHL);
                mOutHL = mul_add(kappa2, _mm_add_ps(mLLY1X1, mLLY1X), mOutHL);
                mOutHL = mul_add(lambda2, _mm_add_ps(mLLYX1, mLLYX), mOutHL);
                mOutHL = mul_add(mu2, _mm_add_ps(mLLY_1X1, mLLY_1X), mOutHL);
                mOutHL = mul_add(nu2, _mm_add_ps(mLLY_2X1, mLLY_2X), mOutHL);
                mOutHL = mul_add(xi2, _mm_add_ps(_mm_add_ps(_mm_add_ps(mHLY1X1, mHLY1X_1), mLHYX2), mLHYX_1), mOutHL);
                mOutHL = mul_add(omicron2, _mm_add_ps(mHLYX1, mHLYX_1), mOutHL);
                mOutHL = mul_add(pi2, _mm_add_ps(mHLY_1X1, mHLY_1X_1), mOutHL);
                mOutHL = mul_add(rho2, _mm_add_ps(_mm_add_ps(_mm_add_ps(mHLY_2X1, mHLY_2X_1), mLHY_2X2), mLHY_2X_1), mOutHL);
                mOutHL = mul_add(sigma2, mHLY1X, mOutHL);
                mOutHL = mul_add(tau2, mHLYX, mOutHL);
                mOutHL = mul_add(upsilon2, mHLY_1X, mOutHL);
                mOutHL = mul_add(phi2, mHLY_2X, mOutHL);
                mOutHL = mul_add(chi2, _mm_add_ps(mLHY_1X2, mLHY_1X_1), mOutHL);
                mOutHL = mul_add(alpha3, _mm_add_ps(mLHYX1, mLHYX), mOutHL);
                mOutHL = mul_add(beta3, _mm_add_ps(mLHY_1X1, mLHY_1X), mOutHL);
                mOutHL = mul_add(gamma3, _mm_add_ps(mLHY_2X1, mLHY_2X), mOutHL);
                mOutHL = mul_add(delta3, _mm_add_ps(mHHYX1, mHHYX_1), mOutHL);
                mOutHL = mul_add(epsilon3, _mm_add_ps(mHHY_1X1, mHHY_1X_1), mOutHL);
                mOutHL = mul_add(zeta3, _mm_add_ps(mHHY_2X1, mHHY_2X_1), mOutHL);
                mOutHL = mul_add(eta3, mHHYX, mOutHL);
                mOutHL = mul_add(theta3, mHHY_1X, mOutHL);
                mOutHL = mul_add(iota3, mHHY_2X, mOutHL);

                // LH = M20(LL) + M21(HL) + M22(LH) + M23(HH)
                __m128 mOutLH;
                mOutLH = _mm_mul_ps(zeta2, _mm_add_ps(mLLY2X1, mLLY_1X1));
                mOutLH = mul_add(kappa2, _mm_add_ps(mLLY1X1, mLLYX1), mOutLH);
                mOutLH = mul_add(eta2, _mm_add_ps(mLLY2X, mLLY_1X), mOutLH);
                mOutLH = mul_add(lambda2, _mm_add_ps(mLLY1X, mLLYX), mOutLH);
                mOutLH = mul_add(theta2, _mm_add_ps(mLLY2X_1, mLLY_1X_1), mOutLH);
                mOutLH = mul_add(mu2, _mm_add_ps(mLLY1X_1, mLLYX_1), mOutLH);
                mOutLH = mul_add(iota2, _mm_add_ps(mLLY2X_2, mLLY_1X_2), mOutLH);
                mOutLH = mul_add(nu2, _mm_add_ps(mLLY1X_2, mLLYX_2), mOutLH);
                mOutLH = mul_add(xi2, _mm_add_ps(_mm_add_ps(_mm_add_ps(mHLY2X, mHLY_1X), mLHY1X1), mLHY_1X1), mOutLH);
                mOutLH = mul_add(alpha3, _mm_add_ps(mHLY1X, mHLYX), mOutLH);
                mOutLH = mul_add(chi2, _mm_add_ps(mHLY2X_1, mHLY_1X_1), mOutLH);
                mOutLH = mul_add(beta3, _mm_add_ps(mHLY1X_1, mHLYX_1), mOutLH);
                mOutLH = mul_add(rho2, _mm_add_ps(_mm_add_ps(_mm_add_ps(mHLY2X_2, mHLY_1X_2), mLHY1X_2), mLHY_1X_2), mOutLH);
                mOutLH = mul_add(gamma3, _mm_add_ps(mHLY1X_2, mHLYX_2), mOutLH);
                mOutLH = mul_add(sigma2, mLHYX1, mOutLH);
                mOutLH = mul_add(omicron2, _mm_add_ps(mLHY1X, mLHY_1X), mOutLH);
                mOutLH = mul_add(tau2, mLHYX, mOutLH);
                mOutLH = mul_add(pi2, _mm_add_ps(mLHY1X_1, mLHY_1X_1), mOutLH);
                mOutLH = mul_add(upsilon2, mLHYX_1, mOutLH);
                mOutLH = mul_add(phi2, mLHYX_2, mOutLH);
                mOutLH = mul_add(delta3, _mm_add_ps(mHHY1X, mHHY_1X), mOutLH);
                mOutLH = mul_add(eta3, mHHYX, mOutLH);
                mOutLH = mul_add(epsilon3, _mm_add_ps( mHHY1X_1, mHHY_1X_1), mOutLH);
                mOutLH = mul_add(theta3, mHHYX_1, mOutLH);
                mOutLH = mul_add(zeta3, _mm_add_ps(mHHY1X_2, mHHY_1X_2), mOutLH);
                mOutLH = mul_add(iota3, mHHYX_2, mOutLH);

                // HH = M30(LL) + M31(HL) + M32(LH) + M33(HH)
                __m128 mOutHH;
                mOutHH = _mm_mul_ps(kappa3, _mm_add_ps(_mm_add_ps(_mm_add_ps(mLLY2X2, mLLY_1X2), mLLY2X_1), mLLY_1X_1));
                mOutHH = mul_add(lambda3, _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(mLLY1X2, mLLYX2), mLLY2X1), mLLY_1X1), mLLY2X), mLLY_1X), mLLY1X_1), mLLYX_1), mOutHH);
                mOutHH = mul_add(mu3, _mm_add_ps(_mm_add_ps(_mm_add_ps(mLLY1X1, mLLYX1), mLLY1X), mLLYX), mOutHH);
                mOutHH = mul_add(nu3, _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(mHLY2X1, mHLY_1X1), mHLY2X_1), mHLY_1X_1), mLHY1X2), mLHY_1X2), mLHY1X_1), mLHY_1X_1), mOutHH);
                mOutHH = mul_add(xi3, _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(mHLY1X1, mHLYX1), mHLY1X_1), mHLYX_1), mLHY1X1), mLHY_1X1), mLHY1X), mLHY_1X), mOutHH);
                mOutHH = mul_add(omicron3, _mm_add_ps(_mm_add_ps(_mm_add_ps(mHLY2X, mHLY_1X), mLHYX2), mLHYX_1), mOutHH);
                mOutHH = mul_add(pi3, _mm_add_ps(_mm_add_ps(_mm_add_ps(mHLY1X, mLHYX1), mHLYX), mLHYX), mOutHH);
                mOutHH = mul_add(rho3, _mm_add_ps(_mm_add_ps(_mm_add_ps(mHHY1X1, mHHY_1X1), mHHY1X_1), mHHY_1X_1), mOutHH);
                mOutHH = mul_add(sigma3, _mm_add_ps(_mm_add_ps(_mm_add_ps(mHHYX1, mHHY1X), mHHY_1X), mHHYX_1), mOutHH);
                mOutHH = mul_add(tau3, mHHYX, mOutHH);

                // S^H_{U0}
                // LL = 1(LL) + U0(HL) + 0(LH) + 0(HH)
                //mOutLL = mul_add(upsilon3, mOutHL, mOutLL);
                // HL = 0(LL) + 1(HL) + 0(LH) + 0(HH)

                // LH = 0(LL) + 0(HL) + 1(LH) + U0(HH)
                mOutLH = mul_add(upsilon3, mOutHH, mOutLH);
                // HH = 0(LL) + 0(HL) + 0(LH) +  1(HH)

                // S^V_{U0}
                // LL = 1(LL) + 0(HL) + U0*(LH) + 0(HH)
                mOutLL = mul_add(upsilon3, _mm_add_ps(mOutLH, mOutHL), mOutLL);
                // HL = 0(LL) + 1(HL) + 0(LH) + U0*(HH)
                mOutHL = mul_add(upsilon3, mOutHH, mOutHL);
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

TRANSFORM_STEP_FUNC NO_TREE_VECTORIZE cdf97_non_separable_convolution_at_amd64_sse(size_t step)
{
    (void)(step);
    return cdf97_non_separable_convolution_at_amd64_sse_all;
}

void NO_TREE_VECTORIZE cdf97_non_separable_convolution_at_amd64_sse_transform_tile(const TransformStepArguments * tsa)
{
    cdf97_non_separable_convolution_at_amd64_sse_all(tsa);
}
#endif
