cmake_minimum_required(VERSION 3.2.2)

project("dwt" C)

set (CMAKE_C_FLAGS "-m64 -pedantic -Wextra -Wconversion -march=native -D_POSIX_C_SOURCE=199309L -O3 -Wall -I. -std=c99")

# Check if OpenMP is supported.
find_package(OpenMP)
if (OPENMP_FOUND)
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif()

# Check for SSE and FMA support.
include (CheckCSourceCompiles)
CHECK_C_SOURCE_COMPILES(
    "#include <smmintrin.h>
    void main(){__m128 A; _mm_insert_ps(A, A, 0x30);}" 
    HAVE_SSE-4.1 )

CHECK_C_SOURCE_COMPILES(
    "#include <immintrin.h>
    void main(){
        __m128 A;
        _mm_fmadd_ps(A,A,A);
    }"
    HAVE_FMA )

CHECK_C_SOURCE_COMPILES(
    "#include <immintrin.h>
    void main(){
        __m512 A;
        _mm512_permute4f128_ps(A, _MM_PERM_ABCD);
        _mm512_swizzle_ps(A, _MM_SWIZ_REG_BADC);
        _mm512_i32gather_ps(A, NULL, 0);
    }"
    HAVE_AVX-512 )

# Add SSE, FMA and AVX 512 options.
#option (USE_AVX_512 "Use AVX 512 instructions." OFF)

include (CMakeDependentOption)
CMAKE_DEPENDENT_OPTION(USE_SSE "Use SSE instructions." ON "HAVE_SSE-4.1" OFF)
CMAKE_DEPENDENT_OPTION(USE_FMA "Use FMA extensions." ON "USE_SSE;HAVE_FMA" OFF)
CMAKE_DEPENDENT_OPTION(USE_AVX-512 "Use AVX 512 instructions." ON "HAVE_AVX-512" OFF)

# Set SSE, FMA and AVX 512 sources and defines.
if (USE_SSE)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DUSE_SSE")
    set (AMD64-SSE-SOURCES "cdf53/separable/lifting/amd64-sse.c" "cdf53/non-separable/lifting/amd64-sse.c" "cdf53/non-separable/convolution_at/amd64-sse.c" "cdf53/non-separable/convolution_star/amd64-sse.c" "cdf97/separable/lifting/amd64-sse.c" "cdf97/non-separable/lifting/amd64-sse.c" "cdf97/non-separable/convolution_at/amd64-sse.c" "cdf97/non-separable/convolution_star/amd64-sse.c" "cdf97/non-separable/polyconvolution/amd64-sse.c" "dd137/separable/lifting/amd64-sse.c" "dd137/non-separable/lifting/amd64-sse.c" "dd137/non-separable/convolution_at/amd64-sse.c" "dd137/non-separable/convolution_star/amd64-sse.c" "haar/amd64-sse.c" "common-amd64-sse.c")
endif ()

if (USE_FMA)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DUSE_FMA")
endif ()

if (USE_AVX-512)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DUSE_AVX_512")
    set (AVX-512-SOURCES "cdf53/separable/lifting/avx-512.c cdf53/non-separable/lifting/avx-512.c cdf53/non-separable/convolution_at/avx-512.c cdf53/non-separable/convolution_star/avx-512.c cdf97/separable/lifting/avx-512.c cdf97/non-separable/lifting/avx-512.c cdf97/non-separable/convolution_at/avx-512.c cdf97/non-separable/convolution_star/avx-512.c cdf97/non-separable/polyconvolution/avx-512.c dd137/separable/lifting/avx-512.c dd137/non-separable/lifting/avx-512.c dd137/non-separable/convolution_at/avx-512.c dd137/non-separable/convolution_star/avx-512.c common-avx-512.c")
endif ()

include_directories(.)
set (GENERIC-SOURCES "cdf53/separable/lifting/generic.c" "cdf53/non-separable/lifting/generic.c" "cdf53/non-separable/convolution_at/generic.c" "cdf53/non-separable/convolution_star/generic.c" "cdf97/separable/lifting/generic.c" "cdf97/non-separable/lifting/generic.c" "cdf97/non-separable/convolution_at/generic.c" "cdf97/non-separable/convolution_star/generic.c" "cdf97/non-separable/polyconvolution/generic.c" "dd137/separable/lifting/generic.c" "dd137/non-separable/lifting/generic.c" "dd137/non-separable/convolution_at/generic.c" "dd137/non-separable/convolution_star/generic.c" "haar/generic.c")

# Create dwt-benchmark target
add_executable(dwt-benchmark main.c common.c benchmark.c ${GENERIC-SOURCES} ${AMD64-SSE-SOURCES} ${AVX-512-SOURCES})
set_target_properties(dwt-benchmark PROPERTIES COMPILE_FLAGS "${CMAKE_C_FLAGS}")
target_link_libraries(dwt-benchmark m)
target_link_libraries(dwt-benchmark rt)

# Create dwt-test target
add_executable(dwt-test main.c common.c test.c ${GENERIC-SOURCES} ${AMD64-SSE-SOURCES} ${AVX-512-SOURCES})
set_target_properties(dwt-test PROPERTIES COMPILE_FLAGS "${CMAKE_C_FLAGS} -DDEBUG")
target_link_libraries(dwt-test m)
target_link_libraries(dwt-test rt)

# Create libdwt target
add_library(dwt SHARED common.c ${GENERIC-SOURCES} ${AMD64-SSE-SOURCES} ${AVX-512-SOURCES})
set_target_properties(dwt PROPERTIES COMPILE_FLAGS "${CMAKE_C_FLAGS}")
