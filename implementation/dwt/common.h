#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>

// assuming cache line size = 64 B
#define CACHE_LINE_SIZE 64

// local memory layout:
// 4 memory areas (one for each subband) due to next transform level
// coefficient is float32 (thus 4 B)
// i.e. 16 coefficients in the cache line

#define LOCA_SPARSE_BANDS
//#define LOCA_SPARSE_ROWS

// NOTE: the +1 term would break down the maping to cache lines
// NOTE: the +CACHE_LINE_SIZE term skews the maping to cache sets
#ifdef LOCA_SPARSE_BANDS
    #define LOCA_BAND_PADDING CACHE_LINE_SIZE
#else
    #define LOCA_BAND_PADDING 0
#endif

// NOTE: the +1 term would break down the maping to cache lines
// NOTE: the +CACHE_LINE_SIZE term skews the maping to cache sets
#ifdef LOCA_SPARSE_ROWS // unused
    #define LOCA_ROW_PADDING CACHE_LINE_SIZE
#else
    #define LOCA_ROW_PADDING 0
#endif

#define BAND_CHUNK_Y(num_threads) ((band_size_y + (num_threads) - 1) / (num_threads))

#ifdef __GNUC__
    #ifdef __INTEL_COMPILER
        #pragma diag_suppress boolean_controlling_expr_is_constant
        #define NO_TREE_VECTORIZE
        #define FORCE_INLINE __attribute__((always_inline))
        #ifdef __MIC__
            #define PROC_BIND_CLOSE
            //#define MEM_ALIGN_BYTES 64
        #else
            #define PROC_BIND_CLOSE proc_bind(close)
            //#define MEM_ALIGN_BYTES 16
        #endif
    #else
        #define NO_TREE_VECTORIZE __attribute__((optimize("no-tree-vectorize")))
        #define PROC_BIND_CLOSE proc_bind(close)
        #define FORCE_INLINE __attribute__((always_inline))
        //#define MEM_ALIGN_BYTES 16
    #endif
#else
    #define NO_TREE_VECTORIZE
    #define FORCE_INLINE
    #ifdef __INTEL_COMPILER
        #pragma diag_suppress boolean_controlling_expr_is_constant
        #ifdef __MIC__
            #define PROC_BIND_CLOSE
            //#define MEM_ALIGN_BYTES 64
        #else
            #define PROC_BIND_CLOSE proc_bind(close)
            //#define MEM_ALIGN_BYTES 16
        #endif
    #else
        #define PROC_BIND_CLOSE
        //#define MEM_ALIGN_BYTES 16
    #endif
#endif

#define NUM_TRANSFORMS_GENERIC 14

#ifdef USE_SSE
    #define NUM_TRANSFORMS_SSE 14
#else
    #define NUM_TRANSFORMS_SSE 0
#endif

#ifdef USE_AVX_512
    #define MEM_ALIGN_BYTES 64
    #define NUM_TRANSFORMS_AVX_512 13
#else
    #define MEM_ALIGN_BYTES 16
    #define NUM_TRANSFORMS_AVX_512 0
#endif

#define NUM_TRANSFORMS (NUM_TRANSFORMS_GENERIC + NUM_TRANSFORMS_SSE + NUM_TRANSFORMS_AVX_512)

#define TMP_BANDS_MARGIN 32

typedef enum {CDF_5_3, CDF_9_7, DD_13_7, HAAR} Wavelet;
typedef enum {SEP_LIFTING, NSP_LIFTING, NSP_CONVOLUTION_AT, NSP_CONVOLUTION_STAR, NSP_POLYCONVOLUTION, SINGLE_LOOP} Scheme;
typedef enum {GENERIC, SSE, AVX_512} Instruction_set;

extern const char * WaveletNames[];
extern const char * SchemeNames[];
extern const char * InstructionSetNames[];

void *memalloc(size_t alignment, size_t size);
void memfree(void *data);

typedef struct
{
    size_t size_x, size_y, stride_y;
    float * data;
} Tile;

#define IMAGE_ROW_PADDING CACHE_LINE_SIZE

typedef struct
{
    size_t size_x, size_y, stride_y;
    size_t tile_size_x, tile_size_y;
    size_t tiles_per_width, tiles_per_height;
    size_t num_tiles;
    float *data;
} Image;

void allocate_image(Image * img, size_t size_x, size_t size_y);

void init_image(Image * img, size_t tile_size_x, size_t tile_size_y);

void load_image(const char * filename, Image * img);

void set_tile_size(Image * img, size_t tile_size_x, size_t tile_size_y);

void get_tile(const Image * img, Tile * tile, size_t x, size_t y);

void free_image(Image * img);

typedef struct
{
    size_t margin;
    size_t size_x, size_y, stride_y;
    float * LL, * HL, * LH, * HH;
} TileBands;

#define BANDS_ROW_PADDING CACHE_LINE_SIZE

typedef struct
{
    size_t margin;
    size_t size_x, size_y, stride_y;
    size_t band_size_x, band_size_y;
    float * LL, * HL, * LH, * HH;
} Bands;

void allocate_bands(Bands * bands, const Image * img, size_t margin);

void clear_bands(Bands * bands);

void get_tile_bands(const Bands * bands, TileBands * tile_bands, size_t x, size_t y);

void free_bands(Bands * bands);

typedef struct
{
    size_t blocks;
    size_t num_threads;
    size_t image_size_x, image_size_y;
    size_t tile_size_x, tile_size_y;
} InputConfig;

void init_input_config(InputConfig * input_config);

typedef struct
{
    size_t num_threads;

    size_t thread_cols;
    size_t thread_rows;

    size_t * band_start_y;
    size_t * band_end_y;

    size_t * band_start_x;
    size_t * band_end_x;
} BandsThreadingInfo;

void allocate_bands_threading_info(BandsThreadingInfo * info, size_t num_threads);

void init_bands_threading_info(BandsThreadingInfo * info, const Bands * bands, const InputConfig * ic);

void free_bands_threading_info(BandsThreadingInfo * info);

typedef struct
{
    Image * input;
    Bands * output, * tmp;
    BandsThreadingInfo * threading_info;
} TransformArguments;

void init_transform_arguments(TransformArguments * transform_arguments, const InputConfig * input_config);

void free_transform_arguments(TransformArguments * transform_arguments);

typedef struct
{
    Tile tile;
    TileBands tile_bands;
    TileBands tmp;
    BandsThreadingInfo * threading_info;
} TransformStepArguments;

typedef void (*TRANSFORM_STEP_FUNC) (const TransformStepArguments *ta);
typedef void (*TRANSFORM_TILE_FUNC) (const TransformStepArguments *ta);

typedef struct
{
    Wavelet wavelet;
    Scheme scheme;
    Instruction_set instruction_set;

    TRANSFORM_TILE_FUNC transform_tile;

    size_t num_steps;
    TRANSFORM_STEP_FUNC * steps;
} Transform;

void init_transform(Transform * t, Wavelet wavelet, Scheme scheme, Instruction_set instruction_set);

void free_transform(Transform * t);

void init_transform_step_arguments(TransformStepArguments *tsa, const TransformArguments * ta, size_t i, size_t j);

Transform * init_all_transforms();

void free_all_transforms(Transform * transforms);

typedef struct
{
    size_t num_tiles;
    size_t start_index, end_index;
} Chunk;

void init_chunk(Chunk * chunk, const Image * img, size_t tid);

size_t mirr(long pos, size_t size);

size_t mirr_band(long pos, size_t size, int lFlag);

void dwt(const Transform * t, const TransformArguments * ta);

#endif	// COMMON_H

