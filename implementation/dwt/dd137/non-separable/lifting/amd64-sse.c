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

// (A+B) * ALPHA2 + C * ALPHA
static inline __m128 kernel_0(__m128 A, __m128 B, __m128 C, __m128 ALPHA, __m128 ALPHA2)
{
    __m128 RESULT;
    RESULT = _mm_add_ps(A, B);
    RESULT = _mm_mul_ps(RESULT, ALPHA2);

#ifdef USE_FMA
    return _mm_fmadd_ps(C, ALPHA, RESULT);
#else
    return _mm_add_ps(RESULT, _mm_mul_ps(C, ALPHA));
#endif
}

// (A+B) * ALPHA2 + (C+D) * ALPHA
static inline __m128 kernel_1(__m128 A, __m128 B, __m128 C, __m128 D, __m128 ALPHA, __m128 ALPHA2)
{
    __m128 TMP1, TMP2;
    TMP1 = _mm_add_ps(A, B);
    TMP1 = _mm_mul_ps(TMP1, ALPHA2);

    TMP2 = _mm_add_ps(C, D);

#ifdef USE_FMA
    return _mm_fmadd_ps(TMP2, ALPHA, TMP1);
#else
    TMP2 = _mm_mul_ps(TMP2, ALPHA);
    return _mm_add_ps(TMP1, TMP2);
#endif
}

// E + (A+B+C+D) * ALPHA
static inline __m128 kernel_2(__m128 A, __m128 B, __m128 C, __m128 D, __m128 E, __m128 ALPHA)
{
    __m128 TMP1, TMP2;
    TMP1 = _mm_add_ps(A, B);
    TMP2 = _mm_add_ps(C, D);
    TMP1 = _mm_add_ps(TMP1, TMP2);

#ifdef USE_FMA
    return _mm_fmadd_ps(TMP1, ALPHA, E);
#else
    return _mm_add_ps(E, _mm_mul_ps(TMP1, ALPHA));
#endif
}

#ifdef USE_FMA
// A * ALPHA  + B
// C * ALPHA2 + D
#define PREDICT_CORE_0 \
    HH00 = _mm_fmadd_ps(LL11, _mm_set_ps1(+0.316406f), HH00); \
    LH00 = _mm_fmadd_ps(LL00, _mm_set_ps1(-9.f/16), LH00);
#define UPDATE_CORE_0 \
    LL00 = _mm_fmadd_ps(HH11_LA, _mm_set_ps1(+0.0791016f), LL00); \
    HL00 = _mm_fmadd_ps(HH00, _mm_set_ps1(+9.f/32), HL00);
#else
#define PREDICT_CORE_0 \
    HH00 = _mm_add_ps(HH00, _mm_mul_ps(LL11, _mm_set_ps1(+0.316406f))); \
    LH00 = _mm_add_ps(LH00, _mm_mul_ps(LL00, _mm_set_ps1(-9.f/16)));
#define UPDATE_CORE_0 \
    LL00 = _mm_add_ps(LL00, _mm_mul_ps(HH11_LA, _mm_set_ps1(+0.0791016f))); \
    HL00 = _mm_add_ps(HL00, _mm_mul_ps(HH00, _mm_set_ps1(+9.f/32)));
#endif

#define PREDICT_CORE \
    LL10_A = load_packed_LX00(R0C1, R0C2); \
    LL20   = load_packed_LX00(R4C1, R4C2); \
    LL10   = load_packed_LX00(R3C1, R3C2); \
    LH00   = load_packed_LX00(R2C1, R2C2); \
\
    LH01_L = load_packed_LX01L(LH00, R2C0); \
    LH01   = load_packed_LX01(LH00, R2C3); \
    LH02   = load_packed_LX02(LH01, R2C3); \
\
    LH00 = _mm_add_ps(LH00, kernel_0(LL10_A, LL20, LL10, _mm_set_ps1(-9.f/16), _mm_set_ps1(+1.f/16))); \
\
    HL00   = load_packed_HX00(R1C1, R1C2); \
    LL00   = load_packed_LX00(R1C1, R1C2); \
\
    LL01_L = load_packed_LX01L(LL00, R1C0); \
    LL01   = load_packed_LX01(LL00, R1C3); \
    LL02   = load_packed_LX02(LL01, R1C3); \
\
    HL00 = _mm_add_ps(HL00, kernel_1(LL02, LL01_L, LL00, LL01, _mm_set_ps1(-9.f/16), _mm_set_ps1(+1.f/16))); \
\
    HH00 = load_packed_HX00(R2C1, R2C2); \
    HL10_A = load_packed_HX00(R0C1, R0C2); \
    HL10   = load_packed_HX00(R3C1, R3C2); \
    HL20   = load_packed_HX00(R4C1, R4C2); \
\
    HH00 = kernel_2(HL00, LH00, LH01, HL10, HH00, _mm_set_ps1(-9.f/16)); \
    HH00 = kernel_2(LH01_L, LH02, HL10_A, HL20, HH00, _mm_set_ps1(+1.f/16)); \
\
    LL11_LA  = load_packed_LX01L(LL10_A, R0C0); \
    LL11_A   = load_packed_LX01(LL10_A, R0C3); \
    LL12_A   = load_packed_LX02(LL11_A, R0C3); \
\
    LL11_L   = load_packed_LX01L(LL10, R3C0); \
    LL11     = load_packed_LX01(LL10, R3C3); \
    LL12     = load_packed_LX02(LL11, R3C3); \
\
    LL21_L   = load_packed_LX01L(LL20, R4C0); \
    LL21     = load_packed_LX01(LL20, R4C3); \
    LL22     = load_packed_LX02(LL21, R4C3); \
\
    HH00 = kernel_2(LL11_LA, LL22, LL12_A, LL21_L, HH00, _mm_set_ps1(+0.00390625f)); \
    HH00 = kernel_2(LL11_A, LL12, LL21, LL11_L, HH00, _mm_set_ps1(-0.0351562f)); \
\
    PREDICT_CORE_0 \
\
    _mm_store_ps(ll, LL00); \
    _mm_store_ps(tmp_hl, HL00); \
    _mm_store_ps(hh, HH00); \
    _mm_store_ps(tmp_lh , LH00); \
\
    ll += 4; hh += 4; tmp_hl += 4; tmp_lh += 4; \
    m0 += 8; m1 += 8; m2 += 8; m3 += 8; m4 += 8;

#define PREDICT_MAIN_AREA \
    for(size_t x = img_start_x + 8; x < img_end_x - 8; x += 8){\
        R0C2 = _mm_load_ps(m0+o1); R0C3 = _mm_load_ps(m0+o2); \
        R1C2 = _mm_load_ps(m1+o1); R1C3 = _mm_load_ps(m1+o2); \
        R2C2 = _mm_load_ps(m2+o1); R2C3 = _mm_load_ps(m2+o2); \
        R3C2 = _mm_load_ps(m3+o1); R3C3 = _mm_load_ps(m3+o2); \
        R4C2 = _mm_load_ps(m4+o1); R4C3 = _mm_load_ps(m4+o2); \
\
        PREDICT_CORE \
\
        R0C0 = R0C2; R0C1 = R0C3; \
        R1C0 = R1C2; R1C1 = R1C3; \
        R2C0 = R2C2; R2C1 = R2C3; \
        R3C0 = R3C2; R3C1 = R3C3; \
        R4C0 = R4C2; R4C1 = R4C3; \
    }

#define UPDATE_CORE \
    LH00 = _mm_load_ps(tmp_lh); \
    HH00 = R3C1; \
    HH01_L = load_unpacked_XX01L(HH00, R3C0); \
    HH02_L = load_unpacked_XX02L(HH01_L, R3C0); \
    HH01   = load_unpacked_XX01(HH00, R3C2); \
\
    LH00 = _mm_add_ps(LH00, kernel_1(HH02_L, HH01, HH01_L, HH00, _mm_set_ps1(+9.f/32), _mm_set_ps1(-1.f/32))); \
\
    HL00 = R2C1; \
    HH20_A = R0C1; \
    HH10_A = R1C1; \
    HH10 = R4C1; \
\
    HL01_L = load_unpacked_XX01L(HL00, R2C0); \
    HL02_L = load_unpacked_XX02L(HL01_L, R2C0); \
    HL01   = load_unpacked_XX01(HL00, R2C2); \
\
    HL00 = _mm_add_ps(HL00, kernel_0(HH10, HH20_A, HH10_A, _mm_set_ps1(+9.f/32), _mm_set_ps1(-1.f/32))); \
\
    LL00 = _mm_load_ps(ll); \
    LH10_A = _mm_load_ps(tmp_lh+v1); \
\
    LL00 = kernel_2(LH10_A, HL01_L, HL00, LH00, LL00, _mm_set_ps1(+9.f/32)); \
\
    LH10   = _mm_load_ps(tmp_lh+v2); \
    LH20_A = _mm_load_ps(tmp_lh+v0); \
\
    LL00 = kernel_2(LH10, LH20_A, HL01, HL02_L, LL00, _mm_set_ps1(-1.f/32)); \
\
    HH11_L = load_unpacked_XX01L(HH10, R4C0); \
    HH12_L = load_unpacked_XX02L(HH11_L, R4C0); \
    HH11   = load_unpacked_XX01(HH10, R4C2); \
\
    HH21_LA = load_unpacked_XX01L(HH20_A, R0C0); \
    HH22_LA = load_unpacked_XX02L(HH21_LA, R0C0); \
    HH21_A  = load_unpacked_XX01(HH20_A, R0C2); \
\
    LL00 = kernel_2(HH11, HH21_A, HH12_L, HH22_LA, LL00, _mm_set_ps1(+0.000976562f)); \
\
    HH11_LA = load_unpacked_XX01L(HH10_A, R1C0); \
    HH12_LA = load_unpacked_XX02L(HH11_LA, R1C0); \
    HH11_A  = load_unpacked_XX01(HH10_A, R1C2); \
\
    LL00 = kernel_2(HH11_L, HH11_A, HH12_LA, HH21_LA, LL00, _mm_set_ps1(-0.00878906f)); \
\
    UPDATE_CORE_0 \
\
    _mm_store_ps(ll, LL00); \
    _mm_store_ps(lh, LH00); \
    _mm_store_ps(hl, HL00); \
\
    lh -= 4; ll -= 4; hl -= 4; hh -= 4; \
    tmp_hl -= 4; tmp_lh -= 4;

#define UPDATE_MAIN_AREA \
    for(size_t x = band_end_x - 8; x >= band_start_x+4; x -= 4){ \
        R0C0 = _mm_load_ps(hh+v0+h0); \
        R1C0 = _mm_load_ps(hh+v1+h0); \
        R2C0 = _mm_load_ps(tmp_hl+h0); \
        R3C0 = _mm_load_ps(hh+h0); \
        R4C0 = _mm_load_ps(hh+v2+h0); \
\
        UPDATE_CORE \
\
        R1C2 = R1C1; R1C1 = R1C0; \
        R0C2 = R0C1; R0C1 = R0C0; \
        R2C2 = R2C1; R2C1 = R2C0; \
        R3C2 = R3C1; R3C1 = R3C0; \
        R4C2 = R4C1; R4C1 = R4C0; \
    }

static inline void predict_line_with_unpack_0(
        const float * m0, const float * m1, const float * m2, const float * m3, const float * m4,
        float * ll, float * tmp_hl, float * tmp_lh, float * hh,
        size_t img_start_x, size_t img_end_x)
{
    const int o1 = 4, o2 = 8;
    __m128 R0C0, R0C1, R0C2, R0C3;
    __m128 R1C0, R1C1, R1C2, R1C3;
    __m128 R2C0, R2C1, R2C2, R2C3;
    __m128 R3C0, R3C1, R3C2, R3C3;
    __m128 R4C0, R4C1, R4C2, R4C3;

    __m128 LL10_A, LL20, LL10, LH00, LH01_L, LH01, LH02;
    __m128 HL00, LL00, LL01_L, LL01, LL02, HH00, HL10_A;
    __m128 HL10, HL20, LL11_LA, LL11_A, LL12_A, LL11_L;
    __m128 LL11, LL12, LL21_L, LL21, LL22;

    R0C1 = _mm_load_ps(m0); R0C2 = _mm_load_ps(m0+o1); R0C3 = _mm_load_ps(m0+o2);
    R1C1 = _mm_load_ps(m1); R1C2 = _mm_load_ps(m1+o1); R1C3 = _mm_load_ps(m1+o2);
    R2C1 = _mm_load_ps(m2); R2C2 = _mm_load_ps(m2+o1); R2C3 = _mm_load_ps(m2+o2);
    R3C1 = _mm_load_ps(m3); R3C2 = _mm_load_ps(m3+o1); R3C3 = _mm_load_ps(m3+o2);
    R4C1 = _mm_load_ps(m4); R4C2 = _mm_load_ps(m4+o1); R4C3 = _mm_load_ps(m4+o2);

    R0C0 = R0C1;
    R1C0 = R1C1;
    R2C0 = R2C1;
    R3C0 = R3C1;
    R4C0 = R4C1;

    PREDICT_CORE

    R0C0 = R0C2; R0C1 = R0C3;
    R1C0 = R1C2; R1C1 = R1C3;
    R2C0 = R2C2; R2C1 = R2C3;
    R3C0 = R3C2; R3C1 = R3C3;
    R4C0 = R4C2; R4C1 = R4C3;

    PREDICT_MAIN_AREA

    R0C2 = _mm_load_ps(m0+o1); R0C3 = _mm_loadr_ps(m0+4); R0C3 = rotate_right(R0C3);
    R1C2 = _mm_load_ps(m1+o1); R1C3 = _mm_loadr_ps(m1+4); R1C3 = rotate_right(R1C3);
    R2C2 = _mm_load_ps(m2+o1); R2C3 = _mm_loadr_ps(m2+4); R2C3 = rotate_right(R2C3);
    R3C2 = _mm_load_ps(m3+o1); R3C3 = _mm_loadr_ps(m3+4); R3C3 = rotate_right(R3C3);
    R4C2 = _mm_load_ps(m4+o1); R4C3 = _mm_loadr_ps(m4+4); R4C3 = rotate_right(R4C3);

    PREDICT_CORE
}

static inline void predict_line_with_unpack_1(
        const float * m0, const float * m1, const float * m2, const float * m3, const float * m4,
        float * ll, float * tmp_hl, float * tmp_lh, float * hh,
        size_t img_start_x, size_t img_end_x)
{
    const int o0 = -4, o1 = 4, o2 = 8;
    __m128 R0C0, R0C1, R0C2, R0C3;
    __m128 R1C0, R1C1, R1C2, R1C3;
    __m128 R2C0, R2C1, R2C2, R2C3;
    __m128 R3C0, R3C1, R3C2, R3C3;
    __m128 R4C0, R4C1, R4C2, R4C3;

    __m128 LL10_A, LL20, LL10, LH00, LH01_L, LH01, LH02;
    __m128 HL00, LL00, LL01_L, LL01, LL02, HH00, HL10_A;
    __m128 HL10, HL20, LL11_LA, LL11_A, LL12_A, LL11_L;
    __m128 LL11, LL12, LL21_L, LL21, LL22;

    R0C1 = _mm_load_ps(m0); R0C2 = _mm_load_ps(m0+o1); R0C3 = _mm_load_ps(m0+o2);
    R1C1 = _mm_load_ps(m1); R1C2 = _mm_load_ps(m1+o1); R1C3 = _mm_load_ps(m1+o2);
    R2C1 = _mm_load_ps(m2); R2C2 = _mm_load_ps(m2+o1); R2C3 = _mm_load_ps(m2+o2);
    R3C1 = _mm_load_ps(m3); R3C2 = _mm_load_ps(m3+o1); R3C3 = _mm_load_ps(m3+o2);
    R4C1 = _mm_load_ps(m4); R4C2 = _mm_load_ps(m4+o1); R4C3 = _mm_load_ps(m4+o2);

    R0C0 = _mm_load_ps(m0+o0);
    R1C0 = _mm_load_ps(m1+o0);
    R2C0 = _mm_load_ps(m2+o0);
    R3C0 = _mm_load_ps(m3+o0);
    R4C0 = _mm_load_ps(m4+o0);

    PREDICT_CORE

    R0C0 = R0C2; R0C1 = R0C3;
    R1C0 = R1C2; R1C1 = R1C3;
    R2C0 = R2C2; R2C1 = R2C3;
    R3C0 = R3C2; R3C1 = R3C3;
    R4C0 = R4C2; R4C1 = R4C3;

    PREDICT_MAIN_AREA

    R0C2 = _mm_load_ps(m0+o1);
    R1C2 = _mm_load_ps(m1+o1);
    R2C2 = _mm_load_ps(m2+o1);
    R3C2 = _mm_load_ps(m3+o1);
    R4C2 = _mm_load_ps(m4+o1);

    R0C3 = _mm_loadr_ps(m0+4); R0C3 = rotate_right(R0C3);
    R1C3 = _mm_loadr_ps(m1+4); R1C3 = rotate_right(R1C3);
    R2C3 = _mm_loadr_ps(m2+4); R2C3 = rotate_right(R2C3);
    R3C3 = _mm_loadr_ps(m3+4); R3C3 = rotate_right(R3C3);
    R4C3 = _mm_loadr_ps(m4+4); R4C3 = rotate_right(R4C3);

    PREDICT_CORE
}

static inline void predict_line_with_unpack_2(
        const float * m0, const float * m1, const float * m2, const float * m3, const float * m4,
        float * ll, float * tmp_hl, float * tmp_lh, float * hh,
        size_t img_start_x, size_t img_end_x)
{
    const int o1 = 4, o2 = 8;
    __m128 R0C0, R0C1, R0C2, R0C3;
    __m128 R1C0, R1C1, R1C2, R1C3;
    __m128 R2C0, R2C1, R2C2, R2C3;
    __m128 R3C0, R3C1, R3C2, R3C3;
    __m128 R4C0, R4C1, R4C2, R4C3;

    __m128 LL10_A, LL20, LL10, LH00, LH01_L, LH01, LH02;
    __m128 HL00, LL00, LL01_L, LL01, LL02, HH00, HL10_A;
    __m128 HL10, HL20, LL11_LA, LL11_A, LL12_A, LL11_L;
    __m128 LL11, LL12, LL21_L, LL21, LL22;

    R0C1 = _mm_load_ps(m0); R0C2 = _mm_load_ps(m0+o1); R0C3 = _mm_load_ps(m0+o2);
    R1C1 = _mm_load_ps(m1); R1C2 = _mm_load_ps(m1+o1); R1C3 = _mm_load_ps(m1+o2);
    R2C1 = _mm_load_ps(m2); R2C2 = _mm_load_ps(m2+o1); R2C3 = _mm_load_ps(m2+o2);
    R3C1 = _mm_load_ps(m3); R3C2 = _mm_load_ps(m3+o1); R3C3 = _mm_load_ps(m3+o2);
    R4C1 = _mm_load_ps(m4); R4C2 = _mm_load_ps(m4+o1); R4C3 = _mm_load_ps(m4+o2);

    R0C0 = R0C1;
    R1C0 = R1C1;
    R2C0 = R2C1;
    R3C0 = R3C1;
    R4C0 = R4C1;

    PREDICT_CORE

    R0C0 = R0C2; R0C1 = R0C3;
    R1C0 = R1C2; R1C1 = R1C3;
    R2C0 = R2C2; R2C1 = R2C3;
    R3C0 = R3C2; R3C1 = R3C3;
    R4C0 = R4C2; R4C1 = R4C3;

    PREDICT_MAIN_AREA

    R0C2 = _mm_load_ps(m0+o1); R0C3 = _mm_load_ps(m0+o2);
    R1C2 = _mm_load_ps(m1+o1); R1C3 = _mm_load_ps(m1+o2);
    R2C2 = _mm_load_ps(m2+o1); R2C3 = _mm_load_ps(m2+o2);
    R3C2 = _mm_load_ps(m3+o1); R3C3 = _mm_load_ps(m3+o2);
    R4C2 = _mm_load_ps(m4+o1); R4C3 = _mm_load_ps(m4+o2);

    PREDICT_CORE
}

static inline void predict_line_with_unpack_3(
        const float * m0, const float * m1, const float * m2, const float * m3, const float * m4,
        float * ll, float * tmp_hl, float * tmp_lh, float * hh,
        size_t img_start_x, size_t img_end_x)
{
    const int o0 = -4, o1 = 4, o2 = 8;
    __m128 R0C0, R0C1, R0C2, R0C3;
    __m128 R1C0, R1C1, R1C2, R1C3;
    __m128 R2C0, R2C1, R2C2, R2C3;
    __m128 R3C0, R3C1, R3C2, R3C3;
    __m128 R4C0, R4C1, R4C2, R4C3;

    __m128 LL10_A, LL20, LL10, LH00, LH01_L, LH01, LH02;
    __m128 HL00, LL00, LL01_L, LL01, LL02, HH00, HL10_A;
    __m128 HL10, HL20, LL11_LA, LL11_A, LL12_A, LL11_L;
    __m128 LL11, LL12, LL21_L, LL21, LL22;

    R0C1 = _mm_load_ps(m0); R0C2 = _mm_load_ps(m0+o1); R0C3 = _mm_load_ps(m0+o2);
    R1C1 = _mm_load_ps(m1); R1C2 = _mm_load_ps(m1+o1); R1C3 = _mm_load_ps(m1+o2);
    R2C1 = _mm_load_ps(m2); R2C2 = _mm_load_ps(m2+o1); R2C3 = _mm_load_ps(m2+o2);
    R3C1 = _mm_load_ps(m3); R3C2 = _mm_load_ps(m3+o1); R3C3 = _mm_load_ps(m3+o2);
    R4C1 = _mm_load_ps(m4); R4C2 = _mm_load_ps(m4+o1); R4C3 = _mm_load_ps(m4+o2);

    R0C0 = _mm_load_ps(m0+o0);
    R1C0 = _mm_load_ps(m1+o0);
    R2C0 = _mm_load_ps(m2+o0);
    R3C0 = _mm_load_ps(m3+o0);
    R4C0 = _mm_load_ps(m4+o0);

    PREDICT_CORE

    R0C0 = R0C2; R0C1 = R0C3;
    R1C0 = R1C2; R1C1 = R1C3;
    R2C0 = R2C2; R2C1 = R2C3;
    R3C0 = R3C2; R3C1 = R3C3;
    R4C0 = R4C2; R4C1 = R4C3;

    PREDICT_MAIN_AREA

    R0C2 = _mm_load_ps(m0+o1); R0C3 = _mm_load_ps(m0+o2);
    R1C2 = _mm_load_ps(m1+o1); R1C3 = _mm_load_ps(m1+o2);
    R2C2 = _mm_load_ps(m2+o1); R2C3 = _mm_load_ps(m2+o2);
    R3C2 = _mm_load_ps(m3+o1); R3C3 = _mm_load_ps(m3+o2);
    R4C2 = _mm_load_ps(m4+o1); R4C3 = _mm_load_ps(m4+o2);

    PREDICT_CORE
}

static inline void update_line_unpacked_0(
        float * ll, float * hl, float * lh, const float * hh,
        const float * tmp_hl, const float * tmp_lh,
        long long v0, long long v1, long long v2,
        size_t band_start_x, size_t band_end_x)
{
    const int h0 = -4;

    __m128 HH22_LA; __m128 HH21_LA; __m128 LH20_A;  __m128 HH20_A;  __m128 HH21_A;
    __m128 HH12_LA; __m128 HH11_LA; __m128 LH10_A;  __m128 HH10_A;  __m128 HH11_A;
    __m128 HL02_L;  __m128 HL01_L;  __m128 LL00;    __m128 HL00;    __m128 HL01;
    __m128 HH02_L;  __m128 HH01_L;  __m128 LH00;    __m128 HH00;    __m128 HH01;
    __m128 HH12_L;  __m128 HH11_L;  __m128 LH10;    __m128 HH10;    __m128 HH11;

    __m128 R0C0, R0C1, R0C2;
    __m128 R1C0, R1C1, R1C2;
    __m128 R2C0, R2C1, R2C2;
    __m128 R3C0, R3C1, R3C2;
    __m128 R4C0, R4C1, R4C2;

    R0C0 = _mm_load_ps(hh+v0+h0);   R0C1 = _mm_load_ps(hh+v0);
    R1C0 = _mm_load_ps(hh+v1+h0);   R1C1 = _mm_load_ps(hh+v1);
    R2C0 = _mm_load_ps(tmp_hl+h0);  R2C1 = _mm_load_ps(tmp_hl);
    R3C0 = _mm_load_ps(hh+h0);      R3C1 = _mm_load_ps(hh);
    R4C0 = _mm_load_ps(hh+v2+h0);   R4C1 = _mm_load_ps(hh+v2);

    R0C2 = _mm_loadr_ps(hh+v0);  R0C2 = rotate_right(R0C2);
    R1C2 = _mm_loadr_ps(hh+v1);  R1C2 = rotate_right(R1C2);
    R2C2 = _mm_loadr_ps(tmp_hl); R2C2 = rotate_right(R2C2);
    R3C2 = _mm_loadr_ps(hh);     R3C2 = rotate_right(R3C2);
    R4C2 = _mm_loadr_ps(hh+v2);  R4C2 = rotate_right(R4C2);

    UPDATE_CORE

    R1C2 = R1C1; R1C1 = R1C0;
    R0C2 = R0C1; R0C1 = R0C0;
    R2C2 = R2C1; R2C1 = R2C0;
    R3C2 = R3C1; R3C1 = R3C0;
    R4C2 = R4C1; R4C1 = R4C0;

    UPDATE_MAIN_AREA

    R0C0 = _mm_loadr_ps(hh+v0);
    R1C0 = _mm_loadr_ps(hh+v1);
    R2C0 = _mm_loadr_ps(tmp_hl);
    R3C0 = _mm_loadr_ps(hh);
    R4C0 = _mm_loadr_ps(hh+v2);

    UPDATE_CORE
}

static inline void update_line_unpacked_1(
        float * ll, float * hl, float * lh, const float * hh,
        const float * tmp_hl, const float * tmp_lh,
        long long v0, long long v1, long long v2,
        size_t band_start_x, size_t band_end_x)
{
    const int h0 = -4;

    __m128 HH22_LA; __m128 HH21_LA; __m128 LH20_A;  __m128 HH20_A;  __m128 HH21_A;
    __m128 HH12_LA; __m128 HH11_LA; __m128 LH10_A;  __m128 HH10_A;  __m128 HH11_A;
    __m128 HL02_L;  __m128 HL01_L;  __m128 LL00;    __m128 HL00;    __m128 HL01;
    __m128 HH02_L;  __m128 HH01_L;  __m128 LH00;    __m128 HH00;    __m128 HH01;
    __m128 HH12_L;  __m128 HH11_L;  __m128 LH10;    __m128 HH10;    __m128 HH11;

    __m128 R0C0, R0C1, R0C2;
    __m128 R1C0, R1C1, R1C2;
    __m128 R2C0, R2C1, R2C2;
    __m128 R3C0, R3C1, R3C2;
    __m128 R4C0, R4C1, R4C2;

    R0C0 = _mm_load_ps(hh+v0+h0);   R0C1 = _mm_load_ps(hh+v0);
    R1C0 = _mm_load_ps(hh+v1+h0);   R1C1 = _mm_load_ps(hh+v1);
    R2C0 = _mm_load_ps(tmp_hl+h0);  R2C1 = _mm_load_ps(tmp_hl);
    R3C0 = _mm_load_ps(hh+h0);      R3C1 = _mm_load_ps(hh);
    R4C0 = _mm_load_ps(hh+v2+h0);   R4C1 = _mm_load_ps(hh+v2);

    R0C2 = _mm_loadr_ps(hh+v0);  R0C2 = rotate_right(R0C2);
    R1C2 = _mm_loadr_ps(hh+v1);  R1C2 = rotate_right(R1C2);
    R2C2 = _mm_loadr_ps(tmp_hl); R2C2 = rotate_right(R2C2);
    R3C2 = _mm_loadr_ps(hh);     R3C2 = rotate_right(R3C2);
    R4C2 = _mm_loadr_ps(hh+v2);  R4C2 = rotate_right(R4C2);

    UPDATE_CORE

    R1C2 = R1C1; R1C1 = R1C0;
    R0C2 = R0C1; R0C1 = R0C0;
    R2C2 = R2C1; R2C1 = R2C0;
    R3C2 = R3C1; R3C1 = R3C0;
    R4C2 = R4C1; R4C1 = R4C0;

    UPDATE_MAIN_AREA

    R0C0 = _mm_load_ps(hh+v0+h0);
    R1C0 = _mm_load_ps(hh+v1+h0);
    R2C0 = _mm_load_ps(tmp_hl+h0);
    R3C0 = _mm_load_ps(hh+h0);
    R4C0 = _mm_load_ps(hh+v2+h0);

    UPDATE_CORE
}

static inline void update_line_unpacked_2(
        float * ll, float * hl, float * lh, const float * hh,
        const float * tmp_hl, const float * tmp_lh,
        long long v0, long long v1, long long v2,
        size_t band_start_x, size_t band_end_x)
{
    const int h0 = -4, h1 = 4;

    __m128 HH22_LA; __m128 HH21_LA; __m128 LH20_A;  __m128 HH20_A;  __m128 HH21_A;
    __m128 HH12_LA; __m128 HH11_LA; __m128 LH10_A;  __m128 HH10_A;  __m128 HH11_A;
    __m128 HL02_L;  __m128 HL01_L;  __m128 LL00;    __m128 HL00;    __m128 HL01;
    __m128 HH02_L;  __m128 HH01_L;  __m128 LH00;    __m128 HH00;    __m128 HH01;
    __m128 HH12_L;  __m128 HH11_L;  __m128 LH10;    __m128 HH10;    __m128 HH11;

    __m128 R0C0, R0C1, R0C2;
    __m128 R1C0, R1C1, R1C2;
    __m128 R2C0, R2C1, R2C2;
    __m128 R3C0, R3C1, R3C2;
    __m128 R4C0, R4C1, R4C2;

    R0C0 = _mm_load_ps(hh+v0+h0);   R0C1 = _mm_load_ps(hh+v0);
    R1C0 = _mm_load_ps(hh+v1+h0);   R1C1 = _mm_load_ps(hh+v1);
    R2C0 = _mm_load_ps(tmp_hl+h0);  R2C1 = _mm_load_ps(tmp_hl);
    R3C0 = _mm_load_ps(hh+h0);      R3C1 = _mm_load_ps(hh);
    R4C0 = _mm_load_ps(hh+v2+h0);   R4C1 = _mm_load_ps(hh+v2);

    R0C2 = _mm_load_ps(hh+v0+h1);
    R1C2 = _mm_load_ps(hh+v1+h1);
    R2C2 = _mm_load_ps(tmp_hl+h1);
    R3C2 = _mm_load_ps(hh+h1);
    R4C2 = _mm_load_ps(hh+v2+h1);

    UPDATE_CORE

    R1C2 = R1C1; R1C1 = R1C0;
    R0C2 = R0C1; R0C1 = R0C0;
    R2C2 = R2C1; R2C1 = R2C0;
    R3C2 = R3C1; R3C1 = R3C0;
    R4C2 = R4C1; R4C1 = R4C0;

    UPDATE_MAIN_AREA

    R0C0 = _mm_loadr_ps(hh+v0);
    R1C0 = _mm_loadr_ps(hh+v1);
    R2C0 = _mm_loadr_ps(tmp_hl);
    R3C0 = _mm_loadr_ps(hh);
    R4C0 = _mm_loadr_ps(hh+v2);

    UPDATE_CORE
}

static inline void update_line_unpacked_3(
        float * ll, float * hl, float * lh, const float * hh,
        const float * tmp_hl, const float * tmp_lh,
        long long v0, long long v1, long long v2,
        size_t band_start_x, size_t band_end_x)
{
    const int h0 = -4, h1 = 4;

    __m128 HH22_LA; __m128 HH21_LA; __m128 LH20_A;  __m128 HH20_A;  __m128 HH21_A;
    __m128 HH12_LA; __m128 HH11_LA; __m128 LH10_A;  __m128 HH10_A;  __m128 HH11_A;
    __m128 HL02_L;  __m128 HL01_L;  __m128 LL00;    __m128 HL00;    __m128 HL01;
    __m128 HH02_L;  __m128 HH01_L;  __m128 LH00;    __m128 HH00;    __m128 HH01;
    __m128 HH12_L;  __m128 HH11_L;  __m128 LH10;    __m128 HH10;    __m128 HH11;

    __m128 R0C0, R0C1, R0C2;
    __m128 R1C0, R1C1, R1C2;
    __m128 R2C0, R2C1, R2C2;
    __m128 R3C0, R3C1, R3C2;
    __m128 R4C0, R4C1, R4C2;

    R0C0 = _mm_load_ps(hh+v0+h0);   R0C1 = _mm_load_ps(hh+v0);
    R1C0 = _mm_load_ps(hh+v1+h0);   R1C1 = _mm_load_ps(hh+v1);
    R2C0 = _mm_load_ps(tmp_hl+h0);  R2C1 = _mm_load_ps(tmp_hl);
    R3C0 = _mm_load_ps(hh+h0);      R3C1 = _mm_load_ps(hh);
    R4C0 = _mm_load_ps(hh+v2+h0);   R4C1 = _mm_load_ps(hh+v2);

    R0C2 = _mm_load_ps(hh+v0+h1);
    R1C2 = _mm_load_ps(hh+v1+h1);
    R2C2 = _mm_load_ps(tmp_hl+h1);
    R3C2 = _mm_load_ps(hh+h1);
    R4C2 = _mm_load_ps(hh+v2+h1);

    UPDATE_CORE

    R1C2 = R1C1; R1C1 = R1C0;
    R0C2 = R0C1; R0C1 = R0C0;
    R2C2 = R2C1; R2C1 = R2C0;
    R3C2 = R3C1; R3C1 = R3C0;
    R4C2 = R4C1; R4C1 = R4C0;

    UPDATE_MAIN_AREA

    R0C0 = _mm_load_ps(hh+v0+h0);
    R1C0 = _mm_load_ps(hh+v1+h0);
    R2C0 = _mm_load_ps(tmp_hl+h0);
    R3C0 = _mm_load_ps(hh+h0);
    R4C0 = _mm_load_ps(hh+v2+h0);

    UPDATE_CORE
}

static void predict_with_unpack(const TransformStepArguments * tsa)
{
    const size_t size_y = tsa->tile.size_y;
    const size_t size_x = tsa->tile.size_x;
    const size_t stride_y = tsa->tile.stride_y;
    float * mem = tsa->tile.data;

    const size_t band_stride_y = tsa->tile_bands.stride_y;

    const size_t tid = (size_t)omp_get_thread_num();

    const size_t band_start_y = tsa->threading_info->band_start_y[tid];
    const size_t band_end_y = tsa->threading_info->band_end_y[tid];
    const size_t band_start_x = tsa->threading_info->band_start_x[tid];
    const size_t band_end_x = tsa->threading_info->band_end_x[tid];

    const size_t img_start_y = band_start_y << 1;
    const size_t img_end_y = band_end_y << 1;
    const size_t img_start_x = band_start_x << 1;
    const size_t img_end_x = band_end_x << 1;

    const size_t next_band_y = band_stride_y;
    const size_t next_tile_y = 2 * stride_y;

    float * m0 = mem + img_start_x + (img_start_y-2) * stride_y;
    float * m1 = mem + img_start_x + img_start_y * stride_y;
    float * m2 = mem + img_start_x + (img_start_y+1) * stride_y;
    float * m3 = mem + img_start_x + (img_start_y+2) * stride_y;
    float * m4 = mem + img_start_x + (img_start_y+4) * stride_y;

    float * ll = tsa->tile_bands.LL + band_start_x + band_start_y * band_stride_y;
    float * hh = tsa->tile_bands.HH + band_start_x + band_start_y * band_stride_y;

    float * tmp_hl = tsa->tmp.HL + band_start_x + band_start_y * band_stride_y;
    float * tmp_lh = tsa->tmp.LH + band_start_x + band_start_y * band_stride_y;

    if(img_start_y == 0 && img_end_y == size_y){
        if(img_start_x == 0 && img_end_x == size_x){
            m0 = m3;
            predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

            ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;

            m0 = m1 - 2 * stride_y;
            for(size_t y = img_start_y + 2; y < img_end_y - 4; y += 2){
                predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            m4 = m3;
            predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

            ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;

            m3 = m1;
            m4 = m0;
            predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);
        } else if(img_end_x == size_x){
            m0 = m3;
            predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

            ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;

            m0 = m1 - 2 * stride_y;
            for(size_t y = img_start_y + 2; y < img_end_y - 4; y += 2){
                predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            m4 = m3;
            predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

            ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;

            m3 = m1;
            m4 = m0;
            predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);
        } else if(img_start_x == 0){
            m0 = m3;
            predict_line_with_unpack_2(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

            ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;

            m0 = m1 - 2 * stride_y;
            for(size_t y = img_start_y + 2; y < img_end_y - 4; y += 2){
                predict_line_with_unpack_2(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            m4 = m3;
            predict_line_with_unpack_2(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

            ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;

            m3 = m1;
            m4 = m0;
            predict_line_with_unpack_2(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);
        } else {
            m0 = m3;
            predict_line_with_unpack_3(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

            ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;

            m0 = m1 - 2 * stride_y;
            for(size_t y = img_start_y + 2; y < img_end_y - 4; y += 2){
                predict_line_with_unpack_3(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            m4 = m3;
            predict_line_with_unpack_3(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

            ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;

            m3 = m1;
            m4 = m0;
            predict_line_with_unpack_3(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);
        }
    } else if(img_start_y == 0 && img_end_y == size_y - 2){
        if(img_start_x == 0 && img_end_x == size_x){
            m0 = m3;
            predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

            ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;

            m0 = m1 - 2 * stride_y;
            for(size_t y = img_start_y + 2; y < img_end_y - 2; y += 2){
                predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            m4 = m3;
            predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);
        } else if(img_end_x == size_x){
            m0 = m3;
            predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

            ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;

            m0 = m1 - 2 * stride_y;
            for(size_t y = img_start_y + 2; y < img_end_y - 2; y += 2){
                predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            m4 = m3;
            predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);
        } else if(img_start_x == 0){
            m0 = m3;
            predict_line_with_unpack_2(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

            ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;

            m0 = m1 - 2 * stride_y;
            for(size_t y = img_start_y + 2; y < img_end_y - 2; y += 2){
                predict_line_with_unpack_2(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            m4 = m3;
            predict_line_with_unpack_2(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);
        } else {
            m0 = m3;
            predict_line_with_unpack_3(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

            ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;

            m0 = m1 - 2 * stride_y;
            for(size_t y = img_start_y + 2; y < img_end_y - 2; y += 2){
                predict_line_with_unpack_3(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            m4 = m3;
            predict_line_with_unpack_3(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);
        }
    } else if(img_start_y == 0){
        if(img_start_x == 0 && img_end_x == size_x){
            m0 = m3;
            predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

            ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;

            m0 = m1 - 2 * stride_y;
            for(size_t y = img_start_y + 2; y < img_end_y; y += 2){
                predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }
        } else if(img_end_x == size_x){
            m0 = m3;
            predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

            ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;

            m0 = m1 - 2 * stride_y;
            for(size_t y = img_start_y + 2; y < img_end_y; y += 2){
                predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }
        } else if(img_start_x == 0){
            m0 = m3;
            predict_line_with_unpack_2(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

            ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;

            m0 = m1 - 2 * stride_y;
            for(size_t y = img_start_y + 2; y < img_end_y; y += 2){
                predict_line_with_unpack_2(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }
        } else {
            m0 = m3;
            predict_line_with_unpack_3(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

            ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
            m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;

            m0 = m1 - 2 * stride_y;
            for(size_t y = img_start_y + 2; y < img_end_y; y += 2){
                predict_line_with_unpack_3(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }
        }

    } else if(img_end_y == size_y){
        if(img_start_x == 0 && img_end_x == size_x){
            for(size_t y = img_start_y; y < img_end_y - 4; y += 2){
                predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            if(img_start_y  < size_y -2){
                m4 = m3;
                predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            m3 = m1;
            m4 = m0;
            predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);
        } else if(img_end_x == size_x){
            for(size_t y = img_start_y; y < img_end_y - 4; y += 2){
                predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            if(img_start_y  < size_y -2){
                m4 = m3;
                predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            m3 = m1;
            m4 = m0;
            predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);
        } else if(img_start_x == 0){
            for(size_t y = img_start_y; y < img_end_y - 4; y += 2){
                predict_line_with_unpack_2(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            if(img_start_y  < size_y -2){
                m4 = m3;
                predict_line_with_unpack_2(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            m3 = m1;
            m4 = m0;
            predict_line_with_unpack_2(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);
        } else {
            for(size_t y = img_start_y; y < img_end_y - 4; y += 2){
                predict_line_with_unpack_3(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            if(img_start_y  < size_y -2){
                m4 = m3;
                predict_line_with_unpack_3(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            m3 = m1;
            m4 = m0;
            predict_line_with_unpack_3(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);
        }
    } else if(img_end_y == size_y - 2){
        if(img_start_x == 0 && img_end_x == size_x){
            for(size_t y = img_start_y; y < img_end_y - 2; y += 2){
                predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            m4 = m3;
            predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);
        } else if(img_end_x == size_x){
            for(size_t y = img_start_y; y < img_end_y - 2; y += 2){
                predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            m4 = m3;
            predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);
        } else if(img_start_x == 0){
            for(size_t y = img_start_y; y < img_end_y - 2; y += 2){
                predict_line_with_unpack_2(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            m4 = m3;
            predict_line_with_unpack_2(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);
        } else {
            for(size_t y = img_start_y; y < img_end_y - 2; y += 2){
                predict_line_with_unpack_3(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }

            m4 = m3;
            predict_line_with_unpack_3(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);
        }
    } else {
        if(img_start_x == 0 && img_end_x == size_x){
            for(size_t y = img_start_y; y < img_end_y; y += 2){
                predict_line_with_unpack_0(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }
        } else if(img_end_x == size_x){
            for(size_t y = img_start_y; y < img_end_y; y += 2){
                predict_line_with_unpack_1(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }
        } else if(img_start_x == 0){
            for(size_t y = img_start_y; y < img_end_y; y += 2){
                predict_line_with_unpack_2(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }
        } else {
            for(size_t y = img_start_y; y < img_end_y; y += 2){
                predict_line_with_unpack_3(m0, m1, m2, m3, m4, ll, tmp_hl, tmp_lh, hh, img_start_x, img_end_x);

                ll += next_band_y; hh += next_band_y; tmp_hl += next_band_y; tmp_lh += next_band_y;
                m0 += next_tile_y; m1 += next_tile_y; m2 += next_tile_y; m3 += next_tile_y; m4 += next_tile_y;
            }
        }
    }

    # pragma omp barrier
}

static void update_unpacked(const TransformStepArguments * tsa)
{
    const size_t band_size_y = tsa->tile_bands.size_y;
    const size_t band_size_x = tsa->tile_bands.size_x;
    const size_t band_stride_y = tsa->tile_bands.stride_y;

    const size_t tid = (size_t)omp_get_thread_num();

    const size_t band_start_y = tsa->threading_info->band_start_y[tid];
    const size_t band_end_y = tsa->threading_info->band_end_y[tid];
    const size_t band_start_x = tsa->threading_info->band_start_x[tid];
    const size_t band_end_x = tsa->threading_info->band_end_x[tid];

    const size_t next_band_y = band_stride_y;

    float * ll = tsa->tile_bands.LL + (band_end_y - 1) * band_stride_y + band_end_x - 4;
    float * hl = tsa->tile_bands.HL + (band_end_y - 1) * band_stride_y + band_end_x - 4;
    float * lh = tsa->tile_bands.LH + (band_end_y - 1) * band_stride_y + band_end_x - 4;
    float * hh = tsa->tile_bands.HH + (band_end_y - 1) * band_stride_y + band_end_x - 4;

    float * tmp_hl = tsa->tmp.HL + (band_end_y - 1) * band_stride_y + band_end_x - 4;
    float * tmp_lh = tsa->tmp.LH + (band_end_y - 1) * band_stride_y + band_end_x - 4;

    long long v0 = -2 * (long long ) band_stride_y, v1 = - (long long) band_stride_y, v2 = (long long) band_stride_y;

    if(band_end_y == band_size_y && band_start_y == 0){
        if(band_end_x == band_size_x && band_start_x == 0){
            v2 = -(long long) band_stride_y;
            update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

            ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
            tmp_hl -= next_band_y; tmp_lh -= next_band_y;

            v2 = (long long) band_stride_y;
            for(size_t y = band_end_y-2; y >= band_start_y + 2; --y){
                update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }

            v0 = -(long long) band_stride_y;
            update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

            ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
            tmp_hl -= next_band_y; tmp_lh -= next_band_y;

            v0 = (long long) band_stride_y;
            v1 = 0;
            update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);
        } else if(band_end_x == band_size_x){
            v2 = -(long long) band_stride_y;
            update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

            ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
            tmp_hl -= next_band_y; tmp_lh -= next_band_y;

            v2 = (long long) band_stride_y;
            for(size_t y = band_end_y-2; y >= band_start_y + 2; --y){
                update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }

            v0 = -(long long) band_stride_y;
            update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

            ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
            tmp_hl -= next_band_y; tmp_lh -= next_band_y;

            v0 = (long long) band_stride_y;
            v1 = 0;
            update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);
        } else if(band_start_x == 0){
            v2 = -(long long) band_stride_y;
            update_line_unpacked_2(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

            ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
            tmp_hl -= next_band_y; tmp_lh -= next_band_y;

            v2 = (long long) band_stride_y;
            for(size_t y = band_end_y-2; y >= band_start_y + 2; --y){
                update_line_unpacked_2(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }

            v0 = -(long long) band_stride_y;
            update_line_unpacked_2(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

            ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
            tmp_hl -= next_band_y; tmp_lh -= next_band_y;

            v0 = (long long) band_stride_y;
            v1 = 0;
            update_line_unpacked_2(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);
        } else {
            v2 = -(long long) band_stride_y;
            update_line_unpacked_3(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

            ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
            tmp_hl -= next_band_y; tmp_lh -= next_band_y;

            v2 = (long long) band_stride_y;
            for(size_t y = band_end_y-2; y >= band_start_y + 2; --y){
                update_line_unpacked_3(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }

            v0 = -(long long) band_stride_y;
            update_line_unpacked_3(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

            ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
            tmp_hl -= next_band_y; tmp_lh -= next_band_y;

            v0 = (long long) band_stride_y;
            v1 = 0;
            update_line_unpacked_3(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);
        }
    } else if(band_end_y == band_size_y && band_start_y == 1){
        if(band_end_x == band_size_x && band_start_x == 0){
            v2 = -(long long) band_stride_y;
            update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

            ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
            tmp_hl -= next_band_y; tmp_lh -= next_band_y;

            v2 = (long long) band_stride_y;
            for(size_t y = band_end_y-2; y >= band_start_y + 1; --y){
                update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }

            v0 = -(long long) band_stride_y;
            update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);
        } else if(band_end_x == band_size_x){
            v2 = -(long long) band_stride_y;
            update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

            ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
            tmp_hl -= next_band_y; tmp_lh -= next_band_y;

            v2 = (long long) band_stride_y;
            for(size_t y = band_end_y-2; y >= band_start_y + 1; --y){
                update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }

            v0 = -(long long) band_stride_y;
            update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);
        } else if(band_start_x == 0){
            v2 = -(long long) band_stride_y;
            update_line_unpacked_2(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

            ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
            tmp_hl -= next_band_y; tmp_lh -= next_band_y;

            v2 = (long long) band_stride_y;
            for(size_t y = band_end_y-2; y >= band_start_y + 1; --y){
                update_line_unpacked_2(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }

            v0 = -(long long) band_stride_y;
            update_line_unpacked_2(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);
        } else {
            v2 = -(long long) band_stride_y;
            update_line_unpacked_3(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

            ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
            tmp_hl -= next_band_y; tmp_lh -= next_band_y;

            v2 = (long long) band_stride_y;
            for(size_t y = band_end_y-2; y >= band_start_y + 1; --y){
                update_line_unpacked_3(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }

            v0 = -(long long) band_stride_y;
            update_line_unpacked_3(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);
        }
    } else if(band_end_y == band_size_y){
        if(band_end_x == band_size_x && band_start_x == 0){
            v2 = -(long long) band_stride_y;
            update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

            ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
            tmp_hl -= next_band_y; tmp_lh -= next_band_y;

            v2 = (long long) band_stride_y;
            for(size_t y = band_end_y-2; y >= band_start_y; --y){
                update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }
        } else if(band_end_x == band_size_x){
            v2 = -(long long) band_stride_y;
            update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

            ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
            tmp_hl -= next_band_y; tmp_lh -= next_band_y;

            v2 = (long long) band_stride_y;
            for(size_t y = band_end_y-2; y >= band_start_y; --y){
                update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }
        } else if(band_start_x == 0){
            v2 = -(long long) band_stride_y;
            update_line_unpacked_2(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

            ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
            tmp_hl -= next_band_y; tmp_lh -= next_band_y;

            v2 = (long long) band_stride_y;
            for(size_t y = band_end_y-2; y >= band_start_y; --y){
                update_line_unpacked_2(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }
        } else {
            v2 = -(long long) band_stride_y;
            update_line_unpacked_3(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

            ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
            tmp_hl -= next_band_y; tmp_lh -= next_band_y;

            v2 = (long long) band_stride_y;
            for(size_t y = band_end_y-2; y >= band_start_y; --y){
                update_line_unpacked_3(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }
        }
    } else if(band_start_y == 0) {
        if(band_end_x == band_size_x && band_start_x == 0){
            for(size_t y = band_end_y-1; y >= band_start_y + 2; --y){
                update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }

            if(band_end_y > 1){
                v0 = -(long long) band_stride_y;
                update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }

            v0 = (long long) band_stride_y;
            v1 = 0;
            update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);
        } else if(band_end_x == band_size_x){
            for(size_t y = band_end_y-1; y >= band_start_y + 2; --y){
                update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }

            if(band_end_y > 1){
                v0 = -(long long) band_stride_y;
                update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }

            v0 = (long long) band_stride_y;
            v1 = 0;
            update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);
        } else if(band_start_x == 0){
            for(size_t y = band_end_y-1; y >= band_start_y + 2; --y){
                update_line_unpacked_2(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }

            if(band_end_y > 1){
                v0 = -(long long) band_stride_y;
                update_line_unpacked_2(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }

            v0 = (long long) band_stride_y;
            v1 = 0;
            update_line_unpacked_2(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);
        } else {
            for(size_t y = band_end_y-1; y >= band_start_y + 2; --y){
                update_line_unpacked_3(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }

            if(band_end_y > 1){
                v0 = -(long long) band_stride_y;
                update_line_unpacked_3(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }

            v0 = (long long) band_stride_y;
            v1 = 0;
            update_line_unpacked_3(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);
        }
    } else if(band_start_y == 1) {
        if(band_end_x == band_size_x && band_start_x == 0){
            for(size_t y = band_end_y-1; y >= band_start_y + 1; --y){
                update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }

            v0 = -(long long) band_stride_y;
            update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);
        } else if(band_end_x == band_size_x){
            for(size_t y = band_end_y-1; y >= band_start_y + 1; --y){
                update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }

            v0 = -(long long) band_stride_y;
            update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);
        } else if(band_start_x == 0){
            for(size_t y = band_end_y-1; y >= band_start_y + 1; --y){
                update_line_unpacked_2(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }

            v0 = -(long long) band_stride_y;
            update_line_unpacked_2(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);
        } else {
            for(size_t y = band_end_y-1; y >= band_start_y + 1; --y){
                update_line_unpacked_3(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }

            v0 = -(long long) band_stride_y;
            update_line_unpacked_3(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);
        }
    } else {
        if(band_end_x == band_size_x && band_start_x == 0){
            for(size_t y = band_end_y-1; y >= band_start_y; --y){
                update_line_unpacked_0(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }
        } else if(band_end_x == band_size_x){
            for(size_t y = band_end_y-1; y >= band_start_y; --y){
                update_line_unpacked_1(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }
        } else if(band_start_x == 0){
            for(size_t y = band_end_y-1; y >= band_start_y; --y){
                update_line_unpacked_2(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }
        } else {
            for(size_t y = band_end_y-1; y >= band_start_y; --y){
                update_line_unpacked_3(ll, hl, lh, hh, tmp_hl, tmp_lh, v0, v1, v2, band_start_x, band_end_x);

                ll -= next_band_y; hl -= next_band_y; lh -= next_band_y; hh -= next_band_y;
                tmp_hl -= next_band_y; tmp_lh -= next_band_y;
            }
        }
    }
}

TRANSFORM_STEP_FUNC NO_TREE_VECTORIZE dd137_non_separable_lifting_amd64_sse(size_t step)
{
    switch(step){
        case 0:
            return predict_with_unpack;
        case 1:
            return update_unpacked;
        default:
            return NULL;
    }
}

void NO_TREE_VECTORIZE dd137_non_separable_lifting_amd64_sse_transform_tile(const TransformStepArguments * tsa)
{
    predict_with_unpack(tsa);
    update_unpacked(tsa);
}
#endif
