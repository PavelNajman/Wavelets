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

static void NO_TREE_VECTORIZE cdf97_non_separable_convolution_at_generic_all(const TransformStepArguments * tsa)
{
    float alpha     = +0.0136748f;
    float beta      = +0.111665f;
    float gamma     = -0.0151008f;
    float delta     = +0.00384796f;
    float epsilon   = +0.911834f;
    float zeta      = +0.0314216f;
    float eta       = -0.12331f;
    float theta     = +0.0166756f;
    float iota      = -0.00424923f;
    float kappa     = +0.00108278f;
    float lambda    = -0.00862145f;
    float mu        = -0.0704009f;
    float nu        = +0.00952051f;
    float xi        = -0.002426f;
    float omicron   = +0.0408159f;
    float pi        = +0.333294f;
    float rho       = -0.0198102f;
    float sigma     = -0.0450723f;
    float tau       = +0.00267898f;
    float upsilon   = +0.0114852f;
    float phi       = -0.000682654f;
    float chi       = +0.00543551f;

    float alpha2    = -0.0257329f;
    float beta2     = +0.0015295f;
    float gamma2    = +0.121826f;
    float delta2    = -0.00724101f;
    float epsilon2  = +0.000430388f;
    float zeta2     = +0.00867621f;
    float eta2      = +0.070848f;
    float theta2    = -0.00958098f;
    float iota2     = +0.00244141f;
    float kappa2    = -0.0562057f;
    float lambda2   = -0.458963f;
    float mu2       = +0.062067f;
    float nu2       = -0.0158158f;
    float xi2       = -0.00547003f;
    float omicron2  = -0.0446671f;
    float pi2       = +0.00604046f;
    float rho2      = -0.00153922f;
    float sigma2    = +0.105999f;
    float tau2      = +0.865566f;
    float upsilon2  = -0.117053f;
    float phi2      = +0.0298272f;
    float chi2      = +0.0258964f;

    float alpha3    = +0.0354357f;
    float beta3     = -0.16776f;
    float gamma3    = +0.00997127f;
    float delta3    = +0.00344866f;
    float epsilon3  = -0.0163267f;
    float zeta3     = +0.000970421f;
    float eta3      = -0.0668286f;
    float theta3    = +0.316382f;
    float iota3     = -0.018805f;
    float kappa3    = +0.00550478f;
    float lambda3   = -0.0356607f;
    float mu3       = +0.231015f;
    float nu3       = -0.00347056f;
    float xi3       = +0.0224828f;
    float omicron3  = +0.0672531f;
    float pi3       = -0.435675f;
    float rho3      = +0.00218806f;
    float sigma3    = -0.0424006f;
    float tau3      = +0.821645f;

    float upsilon3  = +0.4435068520439f;

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

            for (size_t lx = tileStartX; lx < tileEndX; lx += 2) {
                size_t lx1L = mirr((long)lx + 2, tileSizeX);
                size_t lx1H = mirr((long)lx + 3, tileSizeX) - 1;
                size_t lx_1L = mirr((long)lx - 2, tileSizeX);
                size_t lx_1H = mirr((long)lx - 1, tileSizeX) - 1;

                size_t lx2L = mirr((long)lx + 4, tileSizeX);
                size_t lx_2L = mirr((long)lx - 4, tileSizeX);
                size_t lx_2H = mirr((long)lx - 3, tileSizeX) - 1;

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

                float mLLYX2 = LOCA_LL(tileYL, lx2L);

                float mLHYX2 = LOCA_LH(tileYH, lx2L);

                float mLLY1X2 = LOCA_LL(tileY1L, lx2L);

                float mLHY1X2 = LOCA_LH(tileY1H, lx2L);

                float mLLY_1X2 = LOCA_LL(tileY_1L, lx2L);

                float mLHY_1X2 = LOCA_LH(tileY_1H, lx2L);

                float mLLY2X2 = LOCA_LL(tileY2L, lx2L);

                float mLLY_2X2 = LOCA_LL(tileY_2L, lx2L);

                float mLHY_2X2 = LOCA_LH(tileY_2H, lx2L);

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

                // M
                // LL = M00(LL) + M01(HL) + M02(LH) + M03(HH)
                (*outLL) = alpha * mLLY1X1
                        + beta * (mLLYX1 + mLLY1X)
                        + gamma * (mLLY_1X1 + mLLY1X_1)
                        + delta * (mLLY_2X1 + mLLY1X_2)
                        + epsilon * mLLYX
                        + eta * (mLLY_1X + mLLYX_1)
                        + zeta * (mLLY_2X + mLLYX_2)
                        + theta * mLLY_1X_1
                        + iota * (mLLY_2X_1 + mLLY_1X_2)
                        + kappa * mLLY_2X_2
                        + lambda * (mHLY1X + mLHYX1)
                        + mu * (mHLYX + mLHYX)
                        + nu * (mHLY_1X + mLHYX_1)
                        + xi * (mHLY_2X + mHLY1X_2 + mLHY_2X1 + mLHYX_2)
                        + omicron * (mHLY1X_1 + mLHY_1X1)
                        + pi * (mHLYX_1 + mLHY_1X)
                        + sigma * (mHLY_1X_1 + mLHY_1X_1)
                        + upsilon * (mHLY_2X_1 + mLHY_1X_2)
                        + rho * (mHLYX_2 + mLHY_2X)
                        + tau * (mHLY_1X_2 + mLHY_2X_1)
                        + phi * (mHLY_2X_2 + mLHY_2X_2)
                        + chi * mHHYX
                        + alpha2 * (mHHY_1X + mHHYX_1)
                        + beta2 * (mHHY_2X + mHHYX_2)
                        + gamma2 * mHHY_1X_1
                        + delta2 * (mHHY_2X_1 + mHHY_1X_2)
                        + epsilon2 * mHHY_2X_2;

                // HL = M10(LL) + M11(HL) + M12(LH) + M13(HH)
                (*outHL) = zeta2 * (mLLY1X2 + mLLY1X_1)
                        + eta2 * (mLLYX2 + mLLYX_1)
                        + theta2 * (mLLY_1X2 + mLLY_1X_1)
                        + iota2 * (mLLY_2X2 + mLLY_2X_1)
                        + kappa2 * (mLLY1X1 + mLLY1X)
                        + lambda2 * (mLLYX1 + mLLYX)
                        + mu2 * (mLLY_1X1 + mLLY_1X)
                        + nu2 * (mLLY_2X1 + mLLY_2X)
                        + xi2 * (mHLY1X1 + mHLY1X_1 + mLHYX2 + mLHYX_1)
                        + omicron2 * (mHLYX1 + mHLYX_1)
                        + pi2 * (mHLY_1X1 + mHLY_1X_1)
                        + rho2 * (mHLY_2X1 + mHLY_2X_1 + mLHY_2X2 + mLHY_2X_1)
                        + sigma2 * mHLY1X
                        + tau2 * mHLYX
                        + upsilon2 * mHLY_1X
                        + phi2 * mHLY_2X
                        + chi2 * (mLHY_1X2 + mLHY_1X_1)
                        + alpha3 * (mLHYX1 + mLHYX)
                        + beta3 * (mLHY_1X1 + mLHY_1X)
                        + gamma3 * (mLHY_2X1 + mLHY_2X)
                        + delta3 * (mHHYX1 + mHHYX_1)
                        + epsilon3 * (mHHY_1X1 + mHHY_1X_1)
                        + zeta3 * (mHHY_2X1 + mHHY_2X_1)
                        + eta3 * mHHYX
                        + theta3 * mHHY_1X
                        + iota3 * mHHY_2X;

                // LH = M20(LL) + M21(HL) + M22(LH) + M23(HH)
                (*outLH) = zeta2 * (mLLY2X1 + mLLY_1X1)
                        + kappa2 * (mLLY1X1 + mLLYX1)
                        + eta2 * (mLLY2X + mLLY_1X)
                        + lambda2 * (mLLY1X + mLLYX)
                        + theta2 * (mLLY2X_1 + mLLY_1X_1)
                        + mu2 * (mLLY1X_1 + mLLYX_1)
                        + iota2 * (mLLY2X_2 + mLLY_1X_2)
                        + nu2 * (mLLY1X_2 + mLLYX_2)
                        + xi2 * (mHLY2X + mHLY_1X + mLHY1X1 + mLHY_1X1)
                        + alpha3 * (mHLY1X + mHLYX)
                        + chi2 * (mHLY2X_1 + mHLY_1X_1)
                        + beta3 * (mHLY1X_1 + mHLYX_1)
                        + rho2 * (mHLY2X_2 + mHLY_1X_2 + mLHY1X_2 + mLHY_1X_2)
                        + gamma3 * (mHLY1X_2 + mHLYX_2)
                        + sigma2 * mLHYX1
                        + omicron2 * (mLHY1X + mLHY_1X)
                        + tau2 * mLHYX
                        + pi2 * (mLHY1X_1 + mLHY_1X_1)
                        + upsilon2 * mLHYX_1
                        + phi2 * mLHYX_2
                        + delta3 * (mHHY1X + mHHY_1X)
                        + eta3 * mHHYX
                        + epsilon3 * (mHHY1X_1 + mHHY_1X_1)
                        + theta3 * mHHYX_1
                        + zeta3 * (mHHY1X_2 + mHHY_1X_2)
                        + iota3 * mHHYX_2;

                // HH = M30(LL) + M31(HL) + M32(LH) + M33(HH)
                (*outHH) = kappa3 * (mLLY2X2 + mLLY_1X2 + mLLY2X_1 + mLLY_1X_1)
                        + lambda3 * (mLLY1X2 + mLLYX2 + mLLY2X1 + mLLY_1X1 + mLLY2X + mLLY_1X + mLLY1X_1 + mLLYX_1)
                        + mu3 * (mLLY1X1 + mLLYX1 + mLLY1X + mLLYX)
                        + nu3 * (mHLY2X1 + mHLY_1X1 + mHLY2X_1 + mHLY_1X_1 + mLHY1X2 + mLHY_1X2 + mLHY1X_1 + mLHY_1X_1)
                        + xi3 * (mHLY1X1 + mHLYX1 + mHLY1X_1 + mHLYX_1 + mLHY1X1 + mLHY_1X1 + mLHY1X + mLHY_1X)
                        + omicron3 * (mHLY2X + mHLY_1X + mLHYX2 + mLHYX_1)
                        + pi3 * (mHLY1X + mLHYX1 + mHLYX + mLHYX)
                        + rho3 * (mHHY1X1 + mHHY_1X1 + mHHY1X_1 + mHHY_1X_1)
                        + sigma3 * (mHHYX1 + mHHY1X + mHHY_1X + mHHYX_1)
                        + tau3 * mHHYX;

                // S^H_{U0}
                // LL = 1(LL) + U0(HL) + 0(LH) + 0(HH)
                //(*outLL) += upsilon3 * (*outHL);
                // HL = 0(LL) + 1(HL) + 0(LH) + 0(HH)

                // LH = 0(LL) + 0(HL) + 1(LH) + U0(HH)
                (*outLH) += upsilon3 * (*outHH);
                // HH = 0(LL) + 0(HL) + 0(LH) +  1(HH)


                // S^V_{U0}
                // LL = 1(LL) + 0(HL) + U0*(LH) + 0(HH)
                (*outLL) += upsilon3 * ((*outHL) + (*outLH));
                // HL = 0(LL) + 1(HL) + 0(LH) + U0*(HH)
                (*outHL) += upsilon3 * (*outHH);
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

TRANSFORM_STEP_FUNC NO_TREE_VECTORIZE cdf97_non_separable_convolution_at_generic(size_t step)
{
    (void)(step);
    return cdf97_non_separable_convolution_at_generic_all;
}

void NO_TREE_VECTORIZE cdf97_non_separable_convolution_at_generic_transform_tile(const TransformStepArguments * tsa)
{
    cdf97_non_separable_convolution_at_generic_all(tsa);
}
