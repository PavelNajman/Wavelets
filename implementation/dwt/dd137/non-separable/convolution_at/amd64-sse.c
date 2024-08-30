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

static void NO_TREE_VECTORIZE dd137_non_separable_convolution_at_amd64_sse_all(const TransformStepArguments * tsa)
{
    __m128 alpha1    = _mm_set1_ps(+0.0625f);
    __m128 alpha2    = _mm_set1_ps(-0.5625f);
    __m128 alpha3    = _mm_set1_ps(+0.00390625f);
    __m128 alpha4    = _mm_set1_ps(+0.316406f);
    __m128 alpha5    = _mm_set1_ps(-0.03125f);
    __m128 alpha6    = _mm_set1_ps(+0.28125f);
    __m128 alpha7    = _mm_set1_ps(+0.000976562f);
    __m128 beta      = _mm_set1_ps(-0.00195312f);
    __m128 gamma     = _mm_set1_ps(+0.0175781f);
    __m128 delta     = _mm_set1_ps(-0.158203f);
    __m128 epsilon   = _mm_set1_ps(+0.0351562f);
    __m128 epsilon_  = _mm_set1_ps(-0.0351562f);
    __m128 zeta      = _mm_set1_ps(+0.837891f);
    __m128 eta       = _mm_set1_ps(-0.140625f);
    __m128 theta     = _mm_set1_ps(+3.8147e-6f);
    __m128 iota      = _mm_set1_ps(-3.43323e-5f);
    __m128 kappa     = _mm_set1_ps(-6.86646e-5f);
    __m128 lambda    = _mm_set1_ps(-0.00163651f);
    __m128 mu        = _mm_set1_ps(+0.000274658f);
    __m128 nu        = _mm_set1_ps(+0.00030899f);
    __m128 xi        = _mm_set1_ps(+0.000617981f);
    __m128 omicron   = _mm_set1_ps(+0.0147285f);
    __m128 pi        = _mm_set1_ps(-0.00247192f);
    __m128 rho       = _mm_set1_ps(+0.00123596f);
    __m128 sigma     = _mm_set1_ps(+0.0294571f);
    __m128 tau_      = _mm_set1_ps(-0.00494385f);
    __m128 tau       = _mm_set1_ps(+0.00494385f);
    __m128 upsilon   = _mm_set1_ps(+0.702061f);
    __m128 phi       = _mm_set1_ps(-0.117828f);
    __m128 chi       = _mm_set1_ps(+0.0197754f);
    __m128 chi_      = _mm_set1_ps(-0.0197754f);
    __m128 alphaN    = _mm_set1_ps(+6.10352e-5f);
    __m128 betaN     = _mm_set1_ps(-0.000549316f);
    __m128 gammaN    = _mm_set1_ps(+0.00109863f);
    __m128 gammaN_   = _mm_set1_ps(-0.00109863f);
    __m128 deltaN    = _mm_set1_ps(-0.0261841f);
    __m128 epsilonN  = _mm_set1_ps(+0.00439453f);
    __m128 zetaN     = _mm_set1_ps(+0.0098877f);
    __m128 zetaN_    = _mm_set1_ps(-0.0098877f);
    __m128 etaN      = _mm_set1_ps(+0.235657f);
    __m128 thetaN    = _mm_set1_ps(-0.0395508f);
    __m128 iotaN     = _mm_set1_ps(-0.00012207f);
    __m128 kappaN    = _mm_set1_ps(+0.00219727f);
    __m128 lambdaN   = _mm_set1_ps(+0.0523682f);
    __m128 muN       = _mm_set1_ps(-0.00878906f);
    __m128 nuN       = _mm_set1_ps(-0.471313f);
    __m128 xiN       = _mm_set1_ps(+0.0791016f);

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

    size_t tileStrideY2 = 2 * tileStrideY;

    const size_t stepX = 8;

    // N_{P,U1}, S^H_{U0}, S^V_{U0}
    {
        for (size_t ly = tileStartY; ly < tileEndY; ly += 2) {
            size_t ly1L = mirr((long)ly + 2, tileSizeY);
            size_t ly1H = mirr((long)ly + 3, tileSizeY);
            size_t ly_1L = mirr((long)ly - 2, tileSizeY);
            size_t ly_1H = mirr((long)ly - 1, tileSizeY);

            size_t ly2L = mirr((long)ly + 4, tileSizeY);
            size_t ly_2L = mirr((long)ly - 4, tileSizeY);
            size_t ly_2H = mirr((long)ly - 3, tileSizeY);

            size_t ly3L = mirr((long)ly + 6, tileSizeY);
            size_t ly_3L = mirr((long)ly - 6, tileSizeY);

            float *tileY1L = tile + ly1L * tileStrideY;
            float *tileY1H = tile + ly1H * tileStrideY;
            float *tileY_1L = tile + ly_1L * tileStrideY;
            float *tileY_1H = tile + ly_1H * tileStrideY;

            float *tileY2L = tile + ly2L * tileStrideY;
            float *tileY_2L = tile + ly_2L * tileStrideY;
            float *tileY_2H = tile + ly_2H * tileStrideY;

            float *tileY3L = tile + ly3L * tileStrideY;
            float *tileY_3L = tile + ly_3L * tileStrideY;

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

                __m128 mLLY3X = load_packed_LX00(_mm_load_ps(LOCA(tileY3L, lx)), _mm_load_ps(LOCA(tileY3L, lx + 4)));
                __m128 mHLY3X = load_packed_HX00(_mm_load_ps(LOCA(tileY3L, lx)), _mm_load_ps(LOCA(tileY3L, lx + 4)));

                __m128 mLLY_3X = load_packed_LX00(_mm_load_ps(LOCA(tileY_3L, lx)), _mm_load_ps(LOCA(tileY_3L, lx + 4)));
                __m128 mHLY_3X = load_packed_HX00(_mm_load_ps(LOCA(tileY_3L, lx)), _mm_load_ps(LOCA(tileY_3L, lx + 4)));

                __m128 mLLYX1, mHLYX1, mLHYX1, mHHYX1;
                __m128 mLLY1X1, mHLY1X1, mLHY1X1, mHHY1X1;
                __m128 mLLY_1X1, mHLY_1X1, mLHY_1X1, mHHY_1X1;
                __m128 mLLY2X1, mHLY2X1;
                __m128 mLLY_2X1, mHLY_2X1, mLHY_2X1, mHHY_2X1;
                __m128 mLLY3X1, mHLY3X1;
                __m128 mLLY_3X1, mHLY_3X1;
                __m128 mLLYX2, mLHYX2;
                __m128 mLLY1X2, mLHY1X2;
                __m128 mLLY_1X2, mLHY_1X2;
                __m128 mLLY2X2;
                __m128 mLLY_2X2, mLHY_2X2;
                __m128 mLLY3X2;
                __m128 mLLY_3X2;
                __m128 mLLYX3, mLHYX3;
                __m128 mLLY1X3, mLHY1X3;
                __m128 mLLY_1X3, mLHY_1X3;
                __m128 mLLY2X3;
                __m128 mLLY_2X3, mLHY_2X3;
                __m128 mLLY3X3;
                __m128 mLLY_3X3;

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

                    mLLY3X1 = load_packed_LX01(mLLY3X, _mm_load_ps(LOCA(tileY3L, lx + 8)));
                    mHLY3X1 = load_packed_HX01(mHLY3X, _mm_load_ps(LOCA(tileY3L, lx + 8)));

                    mLLY_3X1 = load_packed_LX01(mLLY_3X, _mm_load_ps(LOCA(tileY_3L, lx + 8)));
                    mHLY_3X1 = load_packed_HX01(mHLY_3X, _mm_load_ps(LOCA(tileY_3L, lx + 8)));
                    //
                    mLLYX2 = load_packed_LX02(mLLYX1, _mm_load_ps(LOCA(tileYL, lx + 8)));

                    mLHYX2 = load_packed_LX02(mLHYX1, _mm_load_ps(LOCA(tileYH, lx + 8)));

                    mLLY1X2 = load_packed_LX02(mLLY1X1, _mm_load_ps(LOCA(tileY1L, lx + 8)));

                    mLHY1X2 = load_packed_LX02(mLHY1X1, _mm_load_ps(LOCA(tileY1H, lx + 8)));

                    mLLY_1X2 = load_packed_LX02(mLLY_1X1, _mm_load_ps(LOCA(tileY_1L, lx + 8)));

                    mLHY_1X2 = load_packed_LX02(mLHY_1X1, _mm_load_ps(LOCA(tileY_1H, lx + 8)));

                    mLLY2X2 = load_packed_LX02(mLLY2X1, _mm_load_ps(LOCA(tileY2L, lx + 8)));

                    mLLY_2X2 = load_packed_LX02(mLLY_2X1, _mm_load_ps(LOCA(tileY_2L, lx + 8)));

                    mLHY_2X2 = load_packed_LX02(mLHY_2X1, _mm_load_ps(LOCA(tileY_2H, lx + 8)));

                    mLLY3X2 = load_packed_LX02(mLLY3X1, _mm_load_ps(LOCA(tileY3L, lx + 8)));

                    mLLY_3X2 = load_packed_LX02(mLLY_3X1, _mm_load_ps(LOCA(tileY_3L, lx + 8)));
                    //
                    mLLYX3 = load_packed_LX03(mLLYX2, _mm_load_ps(LOCA(tileYL, lx + 12)));

                    mLHYX3 = load_packed_LX03(mLHYX2, _mm_load_ps(LOCA(tileYH, lx + 12)));

                    mLLY1X3 = load_packed_LX03(mLLY1X2, _mm_load_ps(LOCA(tileY1L, lx + 12)));

                    mLHY1X3 = load_packed_LX03(mLHY1X2, _mm_load_ps(LOCA(tileY1H, lx + 12)));

                    mLLY_1X3 = load_packed_LX03(mLLY_1X2, _mm_load_ps(LOCA(tileY_1L, lx + 12)));

                    mLHY_1X3 = load_packed_LX03(mLHY_1X2, _mm_load_ps(LOCA(tileY_1H, lx + 12)));

                    mLLY2X3 = load_packed_LX03(mLLY2X2, _mm_load_ps(LOCA(tileY2L, lx + 12)));

                    mLLY_2X3 = load_packed_LX03(mLLY_2X2, _mm_load_ps(LOCA(tileY_2L, lx + 12)));

                    mLHY_2X3 = load_packed_LX03(mLHY_2X2, _mm_load_ps(LOCA(tileY_2H, lx + 12)));

                    mLLY3X3 = load_packed_LX03(mLLY3X2, _mm_load_ps(LOCA(tileY3L, lx + 12)));

                    mLLY_3X3 = load_packed_LX03(mLLY_3X2, _mm_load_ps(LOCA(tileY_3L, lx + 12)));
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

                    mLLY3X1 = load_unpacked_LX01S(mLLY3X);
                    mHLY3X1 = load_unpacked_HX01S(mHLY3X);

                    mLLY_3X1 = load_unpacked_LX01S(mLLY_3X);
                    mHLY_3X1 = load_unpacked_HX01S(mHLY_3X);
                    //
                    mLLYX2 = load_unpacked_LX02S(mLLYX);

                    mLHYX2 = load_unpacked_LX02S(mLHYX);

                    mLLY1X2 = load_unpacked_LX02S(mLLY1X);

                    mLHY1X2 = load_unpacked_LX02S(mLHY1X);

                    mLLY_1X2 = load_unpacked_LX02S(mLLY_1X);

                    mLHY_1X2 = load_unpacked_LX02S(mLHY_1X);

                    mLLY2X2 = load_unpacked_LX02S(mLLY2X);

                    mLLY_2X2 = load_unpacked_LX02S(mLLY_2X);

                    mLHY_2X2 = load_unpacked_LX02S(mLHY_2X);

                    mLLY3X2 = load_unpacked_LX02S(mLLY3X);

                    mLLY_3X2 = load_unpacked_LX02S(mLLY_3X);
                    //
                    mLLYX3 = load_unpacked_LX03S(mLLYX);

                    mLHYX3 = load_unpacked_LX03S(mLHYX);

                    mLLY1X3 = load_unpacked_LX03S(mLLY1X);

                    mLHY1X3 = load_unpacked_LX03S(mLHY1X);

                    mLLY_1X3 = load_unpacked_LX03S(mLLY_1X);

                    mLHY_1X3 = load_unpacked_LX03S(mLHY_1X);

                    mLLY2X3 = load_unpacked_LX03S(mLLY2X);

                    mLLY_2X3 = load_unpacked_LX03S(mLLY_2X);

                    mLHY_2X3 = load_unpacked_LX03S(mLHY_2X);

                    mLLY3X3 = load_unpacked_LX03S(mLLY3X);

                    mLLY_3X3 = load_unpacked_LX03S(mLLY_3X);
                }

                __m128 mLLYX_1, mHLYX_1, mLHYX_1, mHHYX_1;
                __m128 mLLY1X_1, mHLY1X_1, mLHY1X_1, mHHY1X_1;
                __m128 mLLY_1X_1, mHLY_1X_1, mLHY_1X_1, mHHY_1X_1;
                __m128 mLLY2X_1, mHLY2X_1;
                __m128 mLLY_2X_1, mHLY_2X_1, mLHY_2X_1, mHHY_2X_1;
                __m128 mLLY3X_1, mHLY3X_1;
                __m128 mLLY_3X_1, mHLY_3X_1;
                __m128 mLLYX_2, mHLYX_2, mLHYX_2, mHHYX_2;
                __m128 mLLY1X_2, mHLY1X_2, mLHY1X_2, mHHY1X_2;
                __m128 mLLY_1X_2, mHLY_1X_2, mLHY_1X_2, mHHY_1X_2;
                __m128 mLLY2X_2, mHLY2X_2;
                __m128 mLLY_2X_2, mHLY_2X_2, mLHY_2X_2, mHHY_2X_2;
                __m128 mLLY3X_2, mHLY3X_2;
                __m128 mLLY_3X_2, mHLY_3X_2;
                __m128 mLLYX_3;
                __m128 mLHYX_3;
                __m128 mLLY1X_3;
                __m128 mLHY1X_3;
                __m128 mLLY_1X_3;
                __m128 mLHY_1X_3;
                __m128 mLLY2X_3;
                __m128 mLLY_2X_3;
                __m128 mLHY_2X_3;
                __m128 mLLY3X_3;
                __m128 mLLY_3X_3;

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

                    mLLY3X_1 = load_packed_LX01L(mLLY3X, _mm_load_ps(LOCA(tileY3L, lx - 4)));
                    mHLY3X_1 = load_packed_HX01L(mHLY3X, _mm_load_ps(LOCA(tileY3L, lx - 4)));

                    mLLY_3X_1 = load_packed_LX01L(mLLY_3X, _mm_load_ps(LOCA(tileY_3L, lx - 4)));
                    mHLY_3X_1 = load_packed_HX01L(mHLY_3X, _mm_load_ps(LOCA(tileY_3L, lx - 4)));

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

                    mLLY3X_2 = load_packed_LX02L(mLLY3X_1, _mm_load_ps(LOCA(tileY3L, lx - 4)));
                    mHLY3X_2 = load_packed_HX02L(mHLY3X_1, _mm_load_ps(LOCA(tileY3L, lx - 4)));

                    mLLY_3X_2 = load_packed_LX02L(mLLY_3X_1, _mm_load_ps(LOCA(tileY_3L, lx - 4)));
                    mHLY_3X_2 = load_packed_HX02L(mHLY_3X_1, _mm_load_ps(LOCA(tileY_3L, lx - 4)));

                    mLLYX_3 = load_packed_LX03L(mLLYX_2, _mm_load_ps(LOCA(tileYL, lx - 8)));

                    mLHYX_3 = load_packed_LX03L(mLHYX_2, _mm_load_ps(LOCA(tileYH, lx - 8)));

                    mLLY1X_3 = load_packed_LX03L(mLLY1X_2, _mm_load_ps(LOCA(tileY1L, lx - 8)));

                    mLHY1X_3 = load_packed_LX03L(mLHY1X_2, _mm_load_ps(LOCA(tileY1H, lx - 8)));

                    mLLY_1X_3 = load_packed_LX03L(mLLY_1X_2, _mm_load_ps(LOCA(tileY_1L, lx - 8)));

                    mLHY_1X_3 = load_packed_LX03L(mLHY_1X_2, _mm_load_ps(LOCA(tileY_1H, lx - 8)));

                    mLLY2X_3 = load_packed_LX03L(mLLY2X_2, _mm_load_ps(LOCA(tileY2L, lx - 8)));

                    mLLY_2X_3 = load_packed_LX03L(mLLY_2X_2, _mm_load_ps(LOCA(tileY_2L, lx - 8)));

                    mLHY_2X_3 = load_packed_LX03L(mLHY_2X_2, _mm_load_ps(LOCA(tileY_2H, lx - 8)));

                    mLLY3X_3 = load_packed_LX03L(mLLY3X_2, _mm_load_ps(LOCA(tileY3L, lx - 8)));

                    mLLY_3X_3 = load_packed_LX03L(mLLY_3X_2, _mm_load_ps(LOCA(tileY_3L, lx - 8)));
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

                    mLLY3X_1 = load_unpacked_LX01LS(mLLY3X);
                    mHLY3X_1 = load_unpacked_HX01LS(mHLY3X);

                    mLLY_3X_1 = load_unpacked_LX01LS(mLLY_3X);
                    mHLY_3X_1 = load_unpacked_HX01LS(mHLY_3X);

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

                    mLLY3X_2 = load_unpacked_LX02LS(mLLY3X);
                    mHLY3X_2 = load_unpacked_HX02LS(mHLY3X);

                    mLLY_3X_2 = load_unpacked_LX02LS(mLLY_3X);
                    mHLY_3X_2 = load_unpacked_HX02LS(mHLY_3X);

                    mLLYX_3 = load_unpacked_LX03LS(mLLYX);

                    mLHYX_3 = load_unpacked_LX03LS(mLHYX);

                    mLLY1X_3 = load_unpacked_LX03LS(mLLY1X);

                    mLHY1X_3 = load_unpacked_LX03LS(mLHY1X);

                    mLLY_1X_3 = load_unpacked_LX03LS(mLLY_1X);

                    mLHY_1X_3 = load_unpacked_LX03LS(mLHY_1X);

                    mLLY2X_3 = load_unpacked_LX03LS(mLLY2X);

                    mLLY_2X_3 = load_unpacked_LX03LS(mLLY_2X);

                    mLHY_2X_3 = load_unpacked_LX03LS(mLHY_2X);

                    mLLY3X_3 = load_unpacked_LX03LS(mLLY3X);

                    mLLY_3X_3 = load_unpacked_LX03LS(mLLY_3X);
                }

                // N_{P,U1}
                // LL = V@*V@(LL) + V@*U1(HL) + U1*V@(LH) + U1*U1(HH)
                __m128 mOutLL;
                mOutLL = _mm_mul_ps(theta, _mm_add_ps(_mm_add_ps(_mm_add_ps(mLLY3X3, mLLY_3X3), mLLY3X_3), mLLY_3X_3));
                mOutLL = mul_add(iota, _mm_add_ps(_mm_add_ps(_mm_add_ps(mLLY2X3, mLLY3X2), mLLY_3X2), mLLY2X_3), mOutLL);
                mOutLL = mul_add(kappa, _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(mLLY1X3, mLLY_2X3), mLLY3X1), mLLY_3X1), mLLY3X_2), mLLY_3X_2), mLLY1X_3), mLLY_2X_3), mOutLL);
                mOutLL = mul_add(lambda, _mm_add_ps(_mm_add_ps(_mm_add_ps(mLLYX3, mLLY3X), mLLY_3X), mLLYX_3), mOutLL);
                mOutLL = mul_add(mu, _mm_add_ps(_mm_add_ps(_mm_add_ps(mLLY_1X3, mLLY3X_1), mLLY_3X_1), mLLY_1X_3), mOutLL);
                mOutLL = mul_add(nu, mLLY2X2, mOutLL);
                mOutLL = mul_add(xi, _mm_add_ps(_mm_add_ps(_mm_add_ps(mLLY1X2, mLLY_2X2), mLLY2X1), mLLY2X_2), mOutLL);
                mOutLL = mul_add(omicron, _mm_add_ps(mLLYX2, mLLY2X), mOutLL);
                mOutLL = mul_add(pi, _mm_add_ps(mLLY_1X2, mLLY2X_1), mOutLL);
                mOutLL = mul_add(rho, _mm_add_ps(_mm_add_ps(_mm_add_ps(mLLY1X1, mLLY_2X1), mLLY1X_2), mLLY_2X_2), mOutLL);
                mOutLL = mul_add(sigma, _mm_add_ps(_mm_add_ps(_mm_add_ps(mLLYX1, mLLY1X), mLLY_2X), mLLYX_2), mOutLL);
                mOutLL = mul_add(tau_, _mm_add_ps(_mm_add_ps(_mm_add_ps(mLLY_1X1, mLLY1X_1), mLLY_2X_1), mLLY_1X_2), mOutLL);
                mOutLL = mul_add(upsilon, mLLYX, mOutLL);
                mOutLL = mul_add(phi, _mm_add_ps(mLLY_1X, mLLYX_1), mOutLL);
                mOutLL = mul_add(chi, mLLY_1X_1, mOutLL);
                mOutLL = mul_add(alphaN, _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(mHLY3X1, mHLY_3X1), mHLY3X_2), mHLY_3X_2), mLHY1X3), mLHY_2X3), mLHY1X_3), mLHY_2X_3), mOutLL);
                mOutLL = mul_add(betaN, _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(mHLY2X1, mHLY3X_1), mHLY_3X_1), mHLY2X_2), mLHY_1X3), mLHY1X2), mLHY_2X2), mLHY_1X_3), mOutLL);
                mOutLL = mul_add(gammaN_, _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(mHLY1X1, mHLY_2X1), mHLY1X_2), mHLY_2X_2), mLHY1X1), mLHY_2X1), mLHY1X_2), mLHY_2X_2), mOutLL);
                mOutLL = mul_add(deltaN, _mm_add_ps(_mm_add_ps(_mm_add_ps(mHLYX1, mHLYX_2), mLHY1X), mLHY_2X), mOutLL);
                mOutLL = mul_add(epsilonN, _mm_add_ps(_mm_add_ps(_mm_add_ps(mHLY_1X1, mHLY_1X_2), mLHY1X_1), mLHY_2X_1), mOutLL);
                mOutLL = mul_add(tau, _mm_add_ps(mHLY2X_1, mLHY_1X2), mOutLL);
                mOutLL = mul_add(zetaN, _mm_add_ps(_mm_add_ps(_mm_add_ps(mHLY1X_1, mHLY_2X_1), mLHY_1X1), mLHY_1X_2), mOutLL);
                mOutLL = mul_add(etaN, _mm_add_ps(mHLYX_1, mLHY_1X), mOutLL);
                mOutLL = mul_add(thetaN, _mm_add_ps(mHLY_1X_1, mLHY_1X_1), mOutLL);
                mOutLL = mul_add(alpha7, _mm_add_ps(_mm_add_ps(_mm_add_ps(mHHY1X1, mHHY1X_2), mHHY_2X1), mHHY_2X_2), mOutLL);
                mOutLL = mul_add(muN, _mm_add_ps(_mm_add_ps(_mm_add_ps(mHHY1X_1, mHHY_1X1), mHHY_1X_2), mHHY_2X_1), mOutLL);
                mOutLL = mul_add(xiN, mHHY_1X_1, mOutLL);

                // HL = V@*P(LL) + V@*(HL) + U1*P(LH) + U1*(HH)
                __m128 mOutHL;
                mOutHL = _mm_mul_ps(iotaN, _mm_add_ps(_mm_add_ps(_mm_add_ps(mLLY3X2, mLLY_3X2), mLLY3X_1), mLLY_3X_1));
                mOutHL = mul_add(gammaN, _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(mLLY2X2, mLLY3X1), mLLY_3X1), mLLY3X), mLLY_3X), mLLY2X_1), mOutHL);
                mOutHL = mul_add(kappaN, _mm_add_ps(_mm_add_ps(_mm_add_ps(mLLY1X2, mLLY_2X2), mLLY1X_1), mLLY_2X_1), mOutHL);
                mOutHL = mul_add(lambdaN, _mm_add_ps(mLLYX2, mLLYX_1), mOutHL);
                mOutHL = mul_add(muN, _mm_add_ps(mLLY_1X2, mLLY_1X_1), mOutHL);
                mOutHL = mul_add(zetaN_, _mm_add_ps(mLLY2X1, mLLY2X), mOutHL);
                mOutHL = mul_add(chi_, _mm_add_ps(_mm_add_ps(_mm_add_ps(mLLY1X1, mLLY_2X1), mLLY1X), mLLY_2X), mOutHL);
                mOutHL = mul_add(nuN, _mm_add_ps(mLLYX1, mLLYX), mOutHL);
                mOutHL = mul_add(xiN, _mm_add_ps(mLLY_1X1, mLLY_1X), mOutHL);
                mOutHL = mul_add(beta, _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(mHLY3X, mHLY_3X), mLHY1X2), mLHY_2X2), mLHY1X_1), mLHY_2X_1), mOutHL);
                mOutHL = mul_add(gamma, _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(mHLY2X, mLHY_1X2), mLHY1X1), mLHY_2X1), mLHY1X), mLHY_2X), mLHY_1X_1), mOutHL);
                mOutHL = mul_add(epsilon, _mm_add_ps(mHLY1X, mHLY_2X), mOutHL);
                mOutHL = mul_add(zeta, mHLYX, mOutHL);
                mOutHL = mul_add(eta, mHLY_1X, mOutHL);
                mOutHL = mul_add(delta, _mm_add_ps(mLHY_1X1, mLHY_1X), mOutHL);
                mOutHL = mul_add(alpha5, _mm_add_ps(mHHY1X, mHHY_2X), mOutHL);
                mOutHL = mul_add(alpha6, mHHY_1X, mOutHL);

                // LH = P*V@(LL) + P*U1(HL) + V@(LH) + U1(HH)
                __m128 mOutLH;
                mOutLH = _mm_mul_ps(iotaN, _mm_add_ps(_mm_add_ps(_mm_add_ps(mLLY2X3, mLLY_1X3), mLLY2X_3), mLLY_1X_3));
                mOutLH = mul_add(gammaN, _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(mLLY1X3, mLLYX3), mLLY2X2), mLLY_1X2), mLLY1X_3), mLLYX_3), mOutLH);
                mOutLH = mul_add(zetaN_, _mm_add_ps(mLLY1X2, mLLYX2), mOutLH);
                mOutLH = mul_add(kappaN, _mm_add_ps(_mm_add_ps(_mm_add_ps(mLLY2X1, mLLY_1X1), mLLY2X_2), mLLY_1X_2), mOutLH);
                mOutLH = mul_add(chi_, _mm_add_ps(_mm_add_ps(_mm_add_ps(mLLY1X1, mLLYX1), mLLY1X_2), mLLYX_2), mOutLH);
                mOutLH = mul_add(lambdaN, _mm_add_ps(mLLY2X, mLLY_1X), mOutLH);
                mOutLH = mul_add(nuN, _mm_add_ps(mLLY1X, mLLYX), mOutLH);
                mOutLH = mul_add(muN, _mm_add_ps(mLLY2X_1, mLLY_1X_1), mOutLH);
                mOutLH = mul_add(xiN, _mm_add_ps(mLLY1X_1, mLLYX_1), mOutLH);
                mOutLH = mul_add(beta, _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(mHLY2X1, mHLY_1X1), mHLY2X_2), mHLY_1X_2), mLHYX3), mLHYX_3), mOutLH);
                mOutLH = mul_add(gamma, _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(mHLY1X1, mHLYX1), mHLY2X_1), mHLY_1X_1), mHLY1X_2), mHLYX_2), mLHYX2), mOutLH);
                mOutLH = mul_add(delta, _mm_add_ps(mHLY1X_1, mHLYX_1), mOutLH);
                mOutLH = mul_add(epsilon, _mm_add_ps(mLHYX1, mLHYX_2), mOutLH);
                mOutLH = mul_add(zeta, mLHYX, mOutLH);
                mOutLH = mul_add(eta, mLHYX_1, mOutLH);
                mOutLH = mul_add(alpha5, _mm_add_ps(mHHYX1, mHHYX_2), mOutLH);
                mOutLH = mul_add(alpha6, mHHYX_1, mOutLH);

                // HH = P*P(LL) + P*(HL) + P(LH) + 1(HH)
                __m128 mOutHH;
                mOutHH = _mm_mul_ps(alpha3, _mm_add_ps(_mm_add_ps(_mm_add_ps(mLLY2X2, mLLY2X_1), mLLY_1X2), mLLY_1X_1));
                mOutHH = mul_add(epsilon_, _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(mLLY2X1, mLLY2X), mLLY1X2), mLLY1X_1), mLLYX2), mLLYX_1), mLLY_1X1), mLLY_1X), mOutHH);
                mOutHH = mul_add(alpha4, _mm_add_ps(_mm_add_ps(_mm_add_ps(mLLY1X1, mLLY1X), mLLYX1), mLLYX), mOutHH);
                mOutHH = mul_add(alpha1, _mm_add_ps(_mm_add_ps(_mm_add_ps(mHLY2X, mHLY_1X), mLHYX2), mLHYX_1), mOutHH);
                mOutHH = mul_add(alpha2, _mm_add_ps(_mm_add_ps(_mm_add_ps(mHLY1X, mHLYX), mLHYX1), mLHYX), mOutHH);
                mOutHH = _mm_add_ps(mHHYX, mOutHH);

                // S^H_{U0}
                // LL = 1(LL) + U0(HL) + 0(LH) + 0(HH)
                //mOutLL = mul_add(alpha6, mOutHL, mOutLL);
                // HL = 0(LL) + 1(HL) + 0(LH) + 0(HH)

                // LH = 0(LL) + 0(HL) + 1(LH) + U0(HH)
                mOutLH = mul_add(alpha6, mOutHH, mOutLH);
                // HH = 0(LL) + 0(HL) + 0(LH) +  1(HH)

                // S^V_{U0}
                // LL = 1(LL) + 0(HL) + U0*(LH) + 0(HH)
                mOutLL = mul_add(alpha6, _mm_add_ps(mOutLH, mOutHL), mOutLL);
                // HL = 0(LL) + 1(HL) + 0(LH) + U0*(HH)
                mOutHL = mul_add(alpha6, mOutHH, mOutHL);
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

TRANSFORM_STEP_FUNC NO_TREE_VECTORIZE dd137_non_separable_convolution_at_amd64_sse(size_t step)
{
    (void)(step);
    return dd137_non_separable_convolution_at_amd64_sse_all;
}

void NO_TREE_VECTORIZE dd137_non_separable_convolution_at_amd64_sse_transform_tile(const TransformStepArguments * tsa)
{
    dd137_non_separable_convolution_at_amd64_sse_all(tsa);
}
#endif
