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

static void NO_TREE_VECTORIZE cdf53_non_separable_convolution_at_generic_all(const TransformStepArguments * tsa)
{
    float alpha     = -0.5f;
    float alpha2    = +0.25f;
    float beta      = +0.875f;
    float alpha3    = -0.125f;
    float gamma     = +0.765625f;
    float delta     = -0.109375f;
    float epsilon   = +0.015625f;
    float zeta      = +0.21875f;
    float eta       = -0.03125f;
    float theta     = -0.4375f;
    float alpha4    = +0.0625f;

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
            size_t ly_1L = mirr((long)ly - 2, tileSizeY);
            size_t ly_1H = mirr((long)ly - 1, tileSizeY);

            float *tileY1L = tile + ly1L * tileStrideY;
            float *tileY_1L = tile + ly_1L * tileStrideY;
            float *tileY_1H = tile + ly_1H * tileStrideY;

            for (size_t lx = tileStartX; lx < tileEndX; lx += 2) {
                size_t lx1L = mirr((long)lx + 2, tileSizeX);
                size_t lx_1L = mirr((long)lx - 2, tileSizeX);
                size_t lx_1H = mirr((long)lx - 1, tileSizeX) - 1;

                float mLLYX = LOCA_LL(tileYL, lx);
                float mHLYX = LOCA_HL(tileYL, lx);
                float mLHYX = LOCA_LH(tileYH, lx);
                float mHHYX = LOCA_HH(tileYH, lx);

                float mLLY1X = LOCA_LL(tileY1L, lx);
                float mHLY1X = LOCA_HL(tileY1L, lx);

                float mLLY_1X = LOCA_LL(tileY_1L, lx);
                float mHLY_1X = LOCA_HL(tileY_1L, lx);
                float mLHY_1X = LOCA_LH(tileY_1H, lx);
                float mHHY_1X = LOCA_HH(tileY_1H, lx);

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
                (*outLL) = gamma * mLLYX
                        + delta * (mLLY_1X + mLLYX_1)
                        + epsilon * mLLY_1X_1
                        + zeta * (mHLYX_1 + mLHY_1X)
                        + eta * (mHLY_1X_1 + mLHY_1X_1)
                        + alpha4 * mHHY_1X_1;

                // HL = V@*P(LL) + V@*(HL) + U1*P(LH) + U1*(HH)
                (*outHL) = theta * (mLLYX1 + mLLYX)
                        + alpha4 * (mLLY_1X1 + mLLY_1X)
                        + beta * mHLYX
                        + alpha3 * (mLHY_1X + mLHY_1X1 + mHLY_1X)
                        + alpha2 * mHHY_1X;

                // LH = P*V@(LL) + P*U1(HL) + V@(LH) + U1(HH)
                (*outLH) = theta * (mLLY1X + mLLYX)
                        + alpha4 * (mLLY1X_1 + mLLYX_1)
                        + alpha3 * (mHLYX_1 + mHLY1X_1 + mLHYX_1)
                        + beta * mLHYX
                        + alpha2 * mHHYX_1;

                // HH = P*P(LL) + P*(HL) + P(LH) + 1(HH)
                (*outHH) = alpha2 * (mLLY1X1 + mLLYX1 + mLLY1X + mLLYX)
                        + alpha * (mHLYX + mHLY1X + mLHYX + mLHYX1)
                        + mHHYX;

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

TRANSFORM_STEP_FUNC NO_TREE_VECTORIZE cdf53_non_separable_convolution_at_generic(size_t step)
{
    (void)(step);
    return cdf53_non_separable_convolution_at_generic_all;
}

void NO_TREE_VECTORIZE cdf53_non_separable_convolution_at_generic_transform_tile(const TransformStepArguments * tsa)
{
    cdf53_non_separable_convolution_at_generic_all(tsa);
}
