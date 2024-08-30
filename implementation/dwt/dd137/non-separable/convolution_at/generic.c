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

static void NO_TREE_VECTORIZE dd137_non_separable_convolution_at_generic_all(const TransformStepArguments * tsa)
{
    float alpha1    = +0.0625f;
    float alpha2    = -0.5625f;
    float alpha3    = +0.00390625f;
    float alpha4    = +0.316406f;
    float alpha5    = -0.03125f;
    float alpha6    = +0.28125f;
    float alpha7    = +0.000976562f;
    float beta      = -0.00195312f;
    float gamma     = +0.0175781f;
    float delta     = -0.158203f;
    float epsilon   = +0.0351562f;
    float epsilon_  = -0.0351562f;
    float zeta      = +0.837891f;
    float eta       = -0.140625f;
    float theta     = +3.8147e-6f;
    float iota      = -3.43323e-5f;
    float kappa     = -6.86646e-5f;
    float lambda    = -0.00163651f;
    float mu        = +0.000274658f;
    float nu        = +0.00030899f;
    float xi        = +0.000617981f;
    float omicron   = +0.0147285f;
    float pi        = -0.00247192f;
    float rho       = +0.00123596f;
    float sigma     = +0.0294571f;
    float tau_      = -0.00494385f;
    float tau       = +0.00494385f;
    float upsilon   = +0.702061f;
    float phi       = -0.117828f;
    float chi       = +0.0197754f;
    float chi_      = -0.0197754f;
    float alphaN    = +6.10352e-5f;
    float betaN     = -0.000549316f;
    float gammaN    = +0.00109863f;
    float gammaN_   = -0.00109863f;
    float deltaN    = -0.0261841f;
    float epsilonN  = +0.00439453f;
    float zetaN     = +0.0098877f;
    float zetaN_    = -0.0098877f;
    float etaN      = +0.235657f;
    float thetaN    = -0.0395508f;
    float iotaN     = -0.00012207f;
    float kappaN    = +0.00219727f;
    float lambdaN   = +0.0523682f;
    float muN       = -0.00878906f;
    float nuN       = -0.471313f;
    float xiN       = +0.0791016f;

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

            for (size_t lx = tileStartX; lx < tileEndX; lx += 2) {
                size_t lx1L = mirr((long)lx + 2, tileSizeX);
                size_t lx1H = mirr((long)lx + 3, tileSizeX) - 1;
                size_t lx_1L = mirr((long)lx - 2, tileSizeX);
                size_t lx_1H = mirr((long)lx - 1, tileSizeX) - 1;

                size_t lx2L = mirr((long)lx + 4, tileSizeX);
                size_t lx_2L = mirr((long)lx - 4, tileSizeX);
                size_t lx_2H = mirr((long)lx - 3, tileSizeX) - 1;

                size_t lx3L = mirr((long)lx + 6, tileSizeX);
                size_t lx_3L = mirr((long)lx - 6, tileSizeX);

                float mLLYX = LOCA_LL(tileYL, lx);
                float mHLYX = LOCA_HL(tileYL, lx);
                float mLHYX = LOCA_LH(tileYH, lx);
                float mHHYX = LOCA_HH(tileYH, lx);

                float mLLY1X = LOCA_LL(tileY1L, lx);
                float mHLY1X = LOCA_HL(tileY1L, lx);
                float mLHY1X = LOCA_LH(tileY1H, lx);
                float mHHY1X = LOCA_HH(tileY1H, lx);

                float mLLY_1X = LOCA_LL(tileY_1L, lx);
                float mHLY_1X = LOCA_HL(tileY_1L, lx);
                float mLHY_1X = LOCA_LH(tileY_1H, lx);
                float mHHY_1X = LOCA_HH(tileY_1H, lx);

                float mLLY2X = LOCA_LL(tileY2L, lx);
                float mHLY2X = LOCA_HL(tileY2L, lx);

                float mLLY_2X = LOCA_LL(tileY_2L, lx);
                float mHLY_2X = LOCA_HL(tileY_2L, lx);
                float mLHY_2X = LOCA_LH(tileY_2H, lx);
                float mHHY_2X = LOCA_HH(tileY_2H, lx);

                float mLLY3X = LOCA_LL(tileY3L, lx);
                float mHLY3X = LOCA_HL(tileY3L, lx);

                float mLLY_3X = LOCA_LL(tileY_3L, lx);
                float mHLY_3X = LOCA_HL(tileY_3L, lx);

                float mLLYX1 = LOCA_LL(tileYL, lx1L);
                float mHLYX1 = LOCA_HL(tileYL, lx1H);
                float mLHYX1 = LOCA_LH(tileYH, lx1L);
                float mHHYX1 = LOCA_HH(tileYH, lx1H);

                float mLLY1X1 = LOCA_LL(tileY1L, lx1L);
                float mHLY1X1 = LOCA_HL(tileY1L, lx1H);
                float mLHY1X1 = LOCA_LH(tileY1H, lx1L);
                float mHHY1X1 = LOCA_HH(tileY1H, lx1H);

                float mLLY_1X1 = LOCA_LL(tileY_1L, lx1L);
                float mHLY_1X1 = LOCA_HL(tileY_1L, lx1H);
                float mLHY_1X1 = LOCA_LH(tileY_1H, lx1L);
                float mHHY_1X1 = LOCA_HH(tileY_1H, lx1H);

                float mLLY2X1 = LOCA_LL(tileY2L, lx1L);
                float mHLY2X1 = LOCA_HL(tileY2L, lx1H);

                float mLLY_2X1 = LOCA_LL(tileY_2L, lx1L);
                float mHLY_2X1 = LOCA_HL(tileY_2L, lx1H);
                float mLHY_2X1 = LOCA_LH(tileY_2H, lx1L);
                float mHHY_2X1 = LOCA_HH(tileY_2H, lx1H);

                float mLLY3X1 = LOCA_LL(tileY3L, lx1L);
                float mHLY3X1 = LOCA_HL(tileY3L, lx1H);

                float mLLY_3X1 = LOCA_LL(tileY_3L, lx1L);
                float mHLY_3X1 = LOCA_HL(tileY_3L, lx1H);

                float mLLYX2 = LOCA_LL(tileYL, lx2L);

                float mLHYX2 = LOCA_LH(tileYH, lx2L);

                float mLLY1X2 = LOCA_LL(tileY1L, lx2L);

                float mLHY1X2 = LOCA_LH(tileY1H, lx2L);

                float mLLY_1X2 = LOCA_LL(tileY_1L, lx2L);

                float mLHY_1X2 = LOCA_LH(tileY_1H, lx2L);

                float mLLY2X2 = LOCA_LL(tileY2L, lx2L);

                float mLLY_2X2 = LOCA_LL(tileY_2L, lx2L);

                float mLHY_2X2 = LOCA_LH(tileY_2H, lx2L);

                float mLLY3X2 = LOCA_LL(tileY3L, lx2L);

                float mLLY_3X2 = LOCA_LL(tileY_3L, lx2L);

                float mLLYX3 = LOCA_LL(tileYL, lx3L);

                float mLHYX3 = LOCA_LH(tileYH, lx3L);

                float mLLY1X3 = LOCA_LL(tileY1L, lx3L);

                float mLHY1X3 = LOCA_LH(tileY1H, lx3L);

                float mLLY_1X3 = LOCA_LL(tileY_1L, lx3L);

                float mLHY_1X3 = LOCA_LH(tileY_1H, lx3L);

                float mLLY2X3 = LOCA_LL(tileY2L, lx3L);

                float mLLY_2X3 = LOCA_LL(tileY_2L, lx3L);

                float mLHY_2X3 = LOCA_LH(tileY_2H, lx3L);

                float mLLY3X3 = LOCA_LL(tileY3L, lx3L);

                float mLLY_3X3 = LOCA_LL(tileY_3L, lx3L);

                float mLLYX_1 = LOCA_LL(tileYL, lx_1L);
                float mHLYX_1 = LOCA_HL(tileYL, lx_1H);
                float mLHYX_1 = LOCA_LH(tileYH, lx_1L);
                float mHHYX_1 = LOCA_HH(tileYH, lx_1H);

                float mLLY1X_1 = LOCA_LL(tileY1L, lx_1L);
                float mHLY1X_1 = LOCA_HL(tileY1L, lx_1H);
                float mLHY1X_1 = LOCA_LH(tileY1H, lx_1L);
                float mHHY1X_1 = LOCA_HH(tileY1H, lx_1H);

                float mLLY_1X_1 = LOCA_LL(tileY_1L, lx_1L);
                float mHLY_1X_1 = LOCA_HL(tileY_1L, lx_1H);
                float mLHY_1X_1 = LOCA_LH(tileY_1H, lx_1L);
                float mHHY_1X_1 = LOCA_HH(tileY_1H, lx_1H);

                float mLLY2X_1 = LOCA_LL(tileY2L, lx_1L);
                float mHLY2X_1 = LOCA_HL(tileY2L, lx_1H);

                float mLLY_2X_1 = LOCA_LL(tileY_2L, lx_1L);
                float mHLY_2X_1 = LOCA_HL(tileY_2L, lx_1H);
                float mLHY_2X_1 = LOCA_LH(tileY_2H, lx_1L);
                float mHHY_2X_1 = LOCA_HH(tileY_2H, lx_1H);

                float mLLY3X_1 = LOCA_LL(tileY3L, lx_1L);
                float mHLY3X_1 = LOCA_HL(tileY3L, lx_1H);

                float mLLY_3X_1 = LOCA_LL(tileY_3L, lx_1L);
                float mHLY_3X_1 = LOCA_HL(tileY_3L, lx_1H);

                float mLLYX_2 = LOCA_LL(tileYL, lx_2L);
                float mHLYX_2 = LOCA_HL(tileYL, lx_2H);
                float mLHYX_2 = LOCA_LH(tileYH, lx_2L);
                float mHHYX_2 = LOCA_HH(tileYH, lx_2H);

                float mLLY1X_2 = LOCA_LL(tileY1L, lx_2L);
                float mHLY1X_2 = LOCA_HL(tileY1L, lx_2H);
                float mLHY1X_2 = LOCA_LH(tileY1H, lx_2L);
                float mHHY1X_2 = LOCA_HH(tileY1H, lx_2H);

                float mLLY_1X_2 = LOCA_LL(tileY_1L, lx_2L);
                float mHLY_1X_2 = LOCA_HL(tileY_1L, lx_2H);
                float mLHY_1X_2 = LOCA_LH(tileY_1H, lx_2L);
                float mHHY_1X_2 = LOCA_HH(tileY_1H, lx_2H);

                float mLLY2X_2 = LOCA_LL(tileY2L, lx_2L);
                float mHLY2X_2 = LOCA_HL(tileY2L, lx_2H);

                float mLLY_2X_2 = LOCA_LL(tileY_2L, lx_2L);
                float mHLY_2X_2 = LOCA_HL(tileY_2L, lx_2H);
                float mLHY_2X_2 = LOCA_LH(tileY_2H, lx_2L);
                float mHHY_2X_2 = LOCA_HH(tileY_2H, lx_2H);

                float mLLY3X_2 = LOCA_LL(tileY3L, lx_2L);
                float mHLY3X_2 = LOCA_HL(tileY3L, lx_2H);

                float mLLY_3X_2 = LOCA_LL(tileY_3L, lx_2L);
                float mHLY_3X_2 = LOCA_HL(tileY_3L, lx_2H);

                float mLLYX_3 = LOCA_LL(tileYL, lx_3L);

                float mLHYX_3 = LOCA_LH(tileYH, lx_3L);

                float mLLY1X_3 = LOCA_LL(tileY1L, lx_3L);

                float mLHY1X_3 = LOCA_LH(tileY1H, lx_3L);

                float mLLY_1X_3 = LOCA_LL(tileY_1L, lx_3L);

                float mLHY_1X_3 = LOCA_LH(tileY_1H, lx_3L);

                float mLLY2X_3 = LOCA_LL(tileY2L, lx_3L);

                float mLLY_2X_3 = LOCA_LL(tileY_2L, lx_3L);

                float mLHY_2X_3 = LOCA_LH(tileY_2H, lx_3L);

                float mLLY3X_3 = LOCA_LL(tileY3L, lx_3L);

                float mLLY_3X_3 = LOCA_LL(tileY_3L, lx_3L);

                // N_{P,U1}
                // LL = V@*V@(LL) + V@*U1(HL) + U1*V@(LH) + U1*U1(HH)
                (*outLL) = theta * (mLLY3X3 + mLLY_3X3 + mLLY3X_3 + mLLY_3X_3)
                        + iota * (mLLY2X3 + mLLY3X2 + mLLY_3X2 + mLLY2X_3)
                        + kappa * (mLLY1X3 + mLLY_2X3 + mLLY3X1 + mLLY_3X1 + mLLY3X_2 + mLLY_3X_2 + mLLY1X_3 + mLLY_2X_3)
                        + lambda * (mLLYX3 + mLLY3X + mLLY_3X + mLLYX_3)
                        + mu * (mLLY_1X3 + mLLY3X_1 + mLLY_3X_1 + mLLY_1X_3)
                        + nu * mLLY2X2
                        + xi * (mLLY1X2 + mLLY_2X2 + mLLY2X1 + mLLY2X_2)
                        + omicron * (mLLYX2 + mLLY2X)
                        + pi * (mLLY_1X2 + mLLY2X_1)
                        + rho * (mLLY1X1 + mLLY_2X1 + mLLY1X_2 + mLLY_2X_2)
                        + sigma * (mLLYX1 + mLLY1X + mLLY_2X + mLLYX_2)
                        + tau_ * (mLLY_1X1 + mLLY1X_1 + mLLY_2X_1 + mLLY_1X_2)
                        + upsilon * mLLYX
                        + phi * (mLLY_1X + mLLYX_1)
                        + chi * mLLY_1X_1
                        + alphaN * (mHLY3X1 + mHLY_3X1 + mHLY3X_2 + mHLY_3X_2 + mLHY1X3 + mLHY_2X3 + mLHY1X_3 + mLHY_2X_3)
                        + betaN * (mHLY2X1 + mHLY3X_1 + mHLY_3X_1 + mHLY2X_2 + mLHY_1X3 + mLHY1X2 + mLHY_2X2 + mLHY_1X_3)
                        + gammaN_ * (mHLY1X1 + mHLY_2X1 + mHLY1X_2 + mHLY_2X_2 + mLHY1X1 + mLHY_2X1 + mLHY1X_2 + mLHY_2X_2)
                        + deltaN * (mHLYX1 + mHLYX_2 + mLHY1X + mLHY_2X)
                        + epsilonN * (mHLY_1X1 + mHLY_1X_2 + mLHY1X_1 + mLHY_2X_1)
                        + tau * (mHLY2X_1 + mLHY_1X2)
                        + zetaN * (mHLY1X_1 + mHLY_2X_1 + mLHY_1X1 + mLHY_1X_2)
                        + etaN * (mHLYX_1 + mLHY_1X)
                        + thetaN * (mHLY_1X_1 + mLHY_1X_1)
                        + alpha7 * (mHHY1X1 + mHHY1X_2 + mHHY_2X1 + mHHY_2X_2)
                        + muN * (mHHY1X_1 + mHHY_1X1 + mHHY_1X_2 + mHHY_2X_1)
                        + xiN * mHHY_1X_1;

                // HL = V@*P(LL) + V@*(HL) + U1*P(LH) + U1*(HH)
                (*outHL) = iotaN * (mLLY3X2 + mLLY_3X2 + mLLY3X_1 + mLLY_3X_1)
                        + gammaN * (mLLY2X2 + mLLY3X1 + mLLY_3X1 + mLLY3X + mLLY_3X + mLLY2X_1)
                        + kappaN * (mLLY1X2 + mLLY_2X2 + mLLY1X_1 + mLLY_2X_1)
                        + lambdaN * (mLLYX2 + mLLYX_1)
                        + muN * (mLLY_1X2 + mLLY_1X_1)
                        + zetaN_ * (mLLY2X1 + mLLY2X)
                        + chi_ * (mLLY1X1 + mLLY_2X1 + mLLY1X + mLLY_2X)
                        + nuN * (mLLYX1 + mLLYX)
                        + xiN * (mLLY_1X1 + mLLY_1X)
                        + beta * (mHLY3X + mHLY_3X + mLHY1X2 + mLHY_2X2 + mLHY1X_1 + mLHY_2X_1)
                        + gamma * (mHLY2X + mLHY_1X2 + mLHY1X1 + mLHY_2X1 + mLHY1X + mLHY_2X + mLHY_1X_1)
                        + epsilon * (mHLY1X + mHLY_2X)
                        + zeta * mHLYX
                        + eta * mHLY_1X
                        + delta * (mLHY_1X1 + mLHY_1X)
                        + alpha5 * (mHHY1X + mHHY_2X)
                        + alpha6 * mHHY_1X;

                // LH = P*V@(LL) + P*U1(HL) + V@(LH) + U1(HH)
                (*outLH) = iotaN * (mLLY2X3 + mLLY_1X3 + mLLY2X_3 + mLLY_1X_3)
                        + gammaN * (mLLY1X3 + mLLYX3 + mLLY2X2 + mLLY_1X2 + mLLY1X_3 + mLLYX_3)
                        + zetaN_ * (mLLY1X2 + mLLYX2)
                        + kappaN * (mLLY2X1 + mLLY_1X1 + mLLY2X_2 + mLLY_1X_2)
                        + chi_ * (mLLY1X1 + mLLYX1 + mLLY1X_2 + mLLYX_2)
                        + lambdaN * (mLLY2X + mLLY_1X)
                        + nuN * (mLLY1X + mLLYX)
                        + muN * (mLLY2X_1 + mLLY_1X_1)
                        + xiN * (mLLY1X_1 + mLLYX_1)
                        + beta * (mHLY2X1 + mHLY_1X1 + mHLY2X_2 + mHLY_1X_2 + mLHYX3 + mLHYX_3)
                        + gamma * (mHLY1X1 + mHLYX1 + mHLY2X_1 + mHLY_1X_1 + mHLY1X_2 + mHLYX_2 + mLHYX2)
                        + delta * (mHLY1X_1 + mHLYX_1)
                        + epsilon * (mLHYX1 + mLHYX_2)
                        + zeta * mLHYX
                        + eta * mLHYX_1
                        + alpha5 * (mHHYX1 + mHHYX_2)
                        + alpha6 * mHHYX_1;

                // HH = P*P(LL) + P*(HL) + P(LH) + 1(HH)
                (*outHH) = alpha3 * (mLLY2X2 + mLLY2X_1 + mLLY_1X2 + mLLY_1X_1)
                        + epsilon_ * (mLLY2X1 + mLLY2X + mLLY1X2 + mLLY1X_1 + mLLYX2 + mLLYX_1 + mLLY_1X1 + mLLY_1X)
                        + alpha4 * (mLLY1X1 + mLLY1X + mLLYX1 + mLLYX)
                        + alpha1 * (mHLY2X + mHLY_1X + mLHYX2 + mLHYX_1)
                        + alpha2 * (mHLY1X + mHLYX + mLHYX1 + mLHYX)
                        + mHHYX;

                // S^H_{U0}
                // LL = 1(LL) + U0(HL) + 0(LH) + 0(HH)
                //(*outLL) += alpha6 * (*outHL);
                // HL = 0(LL) + 1(HL) + 0(LH) + 0(HH)

                // LH = 0(LL) + 0(HL) + 1(LH) + U0(HH)
                (*outLH) += alpha6 * (*outHH);
                // HH = 0(LL) + 0(HL) + 0(LH) +  1(HH)


                // S^V_{U0}
                // LL = 1(LL) + 0(HL) + U0*(LH) + 0(HH)
                (*outLL) += alpha6 * ((*outHL) + (*outLH));
                // HL = 0(LL) + 1(HL) + 0(LH) + U0*(HH)
                (*outHL) += alpha6 * (*outHH);
                // LH = 0(LL) + 0(HL) + 1(LH) + 0(HH)

                // HH = 0(LL) + 0(HL) + 0(LH) + 1(HH)

                outLL++;
                outHL++;
                outLH++;
                outHH++;
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

TRANSFORM_STEP_FUNC NO_TREE_VECTORIZE dd137_non_separable_convolution_at_generic(size_t step)
{
    (void)(step);
    return dd137_non_separable_convolution_at_generic_all;
}

void NO_TREE_VECTORIZE dd137_non_separable_convolution_at_generic_transform_tile(const TransformStepArguments * tsa)
{
    dd137_non_separable_convolution_at_generic_all(tsa);
}
