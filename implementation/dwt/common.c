#include <common.h>

#include <malloc.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdarg.h>
#include <assert.h>

#ifdef _OPENMP
    #include <omp.h>
#endif

#if defined(USE_SSE) || defined(USE_AVX_512)
    #include <xmmintrin.h>
#endif

#include "dwt.h"

const char * WaveletNames[] = {"CDF 5/3", "CDF 9/7", "DD 13/7", "HAAR"};
const char * SchemeNames[] = {"SEPARABLE LIFTING", "NON-SEPARABLE LIFTING", "NON-SEPARABLE CONVOLUTION @", "NON-SEPARABLE CONVOLUTION *", "NON-SEPARABLE POLYCONVOLUTION *@", "SINGLE LOOP"};
const char * InstructionSetNames[] = {"GENERIC", "SSE 4.1", "AVX-512"};

//#define round(x) ((x)>=0?(long)((x)+0.5):(long)((x)-0.5))

#ifdef _WIN32
    inline void *memalloc(size_t alignment, size_t size)
    {
#if defined(USE_SSE) || defined(USE_AVX_512)
        return _mm_malloc(size, alignment);
#else
        return _aligned_malloc(size, alignment);
#endif
    }

    inline void memfree(void *data)
    {
#if defined(USE_SSE) || defined(USE_AVX_512)
        _mm_free(data);
#else
        _aligned_free(data);
#endif
    }
#else
    void *memalloc(size_t alignment, size_t size)
    {
#if defined(USE_SSE) || defined(USE_AVX_512)
        return _mm_malloc(size, alignment);
#else
        return memalign(alignment, size);
#endif
    }

    void memfree(void *data)
    {
#if defined(USE_SSE) || defined(USE_AVX_512)
        _mm_free(data);
#else
        free(data);
#endif
    }
#endif

void allocate_image(Image * img, size_t size_x, size_t size_y)
{
    img->size_x = size_x;
    img->size_y = size_y;
    img->stride_y = size_x + (IMAGE_ROW_PADDING / sizeof(float));
    img->data = memalloc(MEM_ALIGN_BYTES, img->stride_y * size_y * sizeof(float));
    assert(img->data != NULL);
}

void init_image(Image * img, size_t tile_size_x, size_t tile_size_y)
{
    srand(0);
    set_tile_size(img, tile_size_x, tile_size_y);

    for(size_t y = 0; y < img->size_y; ++y){
        for(size_t x = 0; x < img->size_x; ++x){
            img->data[y*img->stride_y + x] = (float)(rand()) / (float)(RAND_MAX);
        }
    }
}

void load_image(const char * filename, Image * img)
{
    FILE * f = fopen(filename, "r");
    assert(f != NULL);

    float n1 = 0;
    for (size_t i = 0; i < img->size_y; ++i) {
        for (size_t j = 0; j < img->size_x; ++j) {
            assert(fscanf(f, "%f,\t", &n1) != EOF);
            *(img->data + i * img->stride_y + j) = n1;
        }
    }
    fclose(f);
}

void set_tile_size(Image * img, size_t tile_size_x, size_t tile_size_y)
{
    assert(tile_size_x <= img->size_x && "tile width <= image width");
    assert(tile_size_y <= img->size_y && "tile height <= image height");
    assert(img->size_x % tile_size_x == 0 && "image width % tile width == 0");
    assert(img->size_y % tile_size_y == 0 && "image height % tile height == 0");

    img->tile_size_x = tile_size_x;
    img->tile_size_y = tile_size_y;

    img->tiles_per_width = img->size_x / img->tile_size_x;
    img->tiles_per_height = img->size_y / img->tile_size_y;

    img->num_tiles = img->tiles_per_width * img->tiles_per_height;
}

void get_tile(const Image * img, Tile * tile, size_t x, size_t y)
{
    tile->size_x = img->tile_size_x;
    tile->size_y = img->tile_size_y;
    tile->stride_y = img->stride_y;
    tile->data = img->data + y * tile->size_y * tile->stride_y + x * tile->size_x;
}

void free_image(Image * img)
{
    memfree(img->data);
}

void init_chunk(Chunk * chunk, const Image * img, size_t tid)
{
    chunk->num_tiles = img->num_tiles;
    chunk->start_index = tid * chunk->num_tiles;
    chunk->end_index = (tid + 1) * chunk->num_tiles;
        if(chunk->end_index > img->num_tiles)
            chunk->end_index = img->num_tiles;
}

void allocate_bands(Bands * bands, const Image * img, size_t margin)
{
    bands->margin = margin;
    bands->size_x = img->size_x >> 1;
    bands->size_y = img->size_y >> 1;

    bands->stride_y = bands->size_x + 2*bands->margin + (BANDS_ROW_PADDING / sizeof(float));
    size_t byte_size = (bands->size_y + (BANDS_ROW_PADDING / sizeof(float)) + 2*bands->margin) * bands->stride_y * sizeof(float) + LOCA_BAND_PADDING;

    bands->band_size_x = img->tile_size_x >> 1;
    bands->band_size_y = img->tile_size_y >> 1;

    float *mem = memalloc(MEM_ALIGN_BYTES, byte_size * 4);    
    assert(mem != NULL);

    bands->LL = mem + 0 * byte_size / sizeof(float);
    bands->HL = mem + 1 * byte_size / sizeof(float);
    bands->LH = mem + 2 * byte_size / sizeof(float);
    bands->HH = mem + 3 * byte_size / sizeof(float);

    clear_bands(bands);
}

void clear_bands(Bands * bands)
{
    memset(bands->LL, 0, (bands->size_y + 2*bands->margin) * bands->stride_y * 4 * sizeof(float));
}

void get_tile_bands(const Bands * bands, TileBands * tile_bands, size_t x, size_t y)
{
    tile_bands->margin = bands->margin;

    tile_bands->size_x = bands->band_size_x;
    tile_bands->size_y = bands->band_size_y;
    tile_bands->stride_y = bands->stride_y;

    tile_bands->LL = bands->LL + y * bands->band_size_y * bands->stride_y + x * bands->band_size_x;
    tile_bands->HL = bands->HL + y * bands->band_size_y * bands->stride_y + x * bands->band_size_x;
    tile_bands->LH = bands->LH + y * bands->band_size_y * bands->stride_y + x * bands->band_size_x;
    tile_bands->HH = bands->HH + y * bands->band_size_y * bands->stride_y + x * bands->band_size_x;
}

void free_bands(Bands * bands)
{
    memfree(bands->LL);
}

void allocate_bands_threading_info(BandsThreadingInfo * info, size_t num_threads)
{
    size_t *mem = memalloc(MEM_ALIGN_BYTES, 4 * num_threads * sizeof(size_t));
    assert(mem != NULL);

    info->band_start_y = mem;
    info->band_end_y   = mem + num_threads;
    info->band_start_x = mem + 2 * num_threads;
    info->band_end_x   = mem + 3 * num_threads;
}

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

size_t find_factors(size_t n, size_t * factors)
{
    size_t num_factors = 0;
    for(size_t f = 1; f < (n >> 1); ++f){
        if(!(n % f))
            factors[num_factors] = f;
    }

    return num_factors;
}

/* We are looking for a factor of num_threads f that satisfies (row_size / f) % (CACHE_LINE_SIZE / sizeof(float)) == 0 */
size_t find_num_threads_per_row(size_t row_size, size_t num_threads)
{
    size_t result = 1;
    for(size_t f = 1; f <= (num_threads >> 1); ++f){
        if(!(num_threads % f)
                && num_threads / f >= num_threads / (num_threads / f)
                && !(row_size % f)
                && !((row_size / f) % (CACHE_LINE_SIZE / sizeof(float))))
            result = f;
    }
    return result;
}

void init_bands_threading_info(BandsThreadingInfo * info, const Bands * bands, const InputConfig * ic)
{
    assert(ic->image_size_x != 0 && ic->tile_size_x != 0 && ic->image_size_y != 0 && ic->tile_size_y != 0 && "Image and tile area have to be greater than 0.");
    assert(ic->image_size_x >= ic->tile_size_x && ic->image_size_y >= ic->tile_size_y && "Image size has to be greater than or equal to tile size.");
    assert(ic->image_size_x % ic->tile_size_x == 0 && ic->image_size_y % ic->tile_size_y == 0 && "Image size has to be divisible by tile size.");
    assert(ic->tile_size_x % 2 == 0 && ic->tile_size_y % 2 == 0 && "Tile width and height have to be even.");
    
    info->num_threads = ic->num_threads;

    assert((bands->band_size_x % (CACHE_LINE_SIZE / sizeof(float))) == 0 && "Band width has to be divisible by cache line size");

    if(!ic->blocks)
        info->thread_cols = 1;
    else
        info->thread_cols = find_num_threads_per_row(bands->band_size_x, info->num_threads);
    info->thread_rows = info->num_threads / info->thread_cols;
    size_t px_per_thread_per_row = bands->band_size_x / info->thread_cols;
    double rows_per_thread = (double) bands->band_size_y / (double)(info->thread_rows);

    if(!ic->blocks){
        assert(bands->band_size_y >= info->num_threads && "Too much threads! There has to be at least one band row for each thread.");
    } else {
        size_t cache_lines_per_row = bands->band_size_x / (CACHE_LINE_SIZE / sizeof(float));
        size_t num_cache_lines = cache_lines_per_row * bands->band_size_y;
        assert(num_cache_lines >= info->num_threads && "Too much threads! There has to be at least one cache line for each thread.");
        assert(bands->band_size_y * info->thread_cols >= info->num_threads && "Too much threads! There has to be at least one block for each thread.");
    }

    //printf("%lu %lu %lu %f %f\n", cache_lines_per_row, threads_per_row, px_per_thread_per_row, threads_per_column, rows_per_thread);
    for(size_t tid = 0; tid < info->num_threads; ++tid){
        size_t c_tid = tid % info->thread_cols;
        size_t r_tid = tid / info->thread_cols;

        info->band_start_x[tid] = MIN(bands->band_size_x, (size_t) floor((double) (c_tid * px_per_thread_per_row)));
        info->band_end_x[tid] = MIN(bands->band_size_x, (size_t)  floor((double)((c_tid + 1) * px_per_thread_per_row)));

        info->band_start_y[tid] = MIN(bands->band_size_y, (size_t) floor((double)r_tid * rows_per_thread));
        info->band_end_y[tid] = MIN(bands->band_size_y, (size_t) floor((double)(r_tid+1) * rows_per_thread));

        if(tid == info->num_threads - 1){
            info->band_end_x[tid] = bands->band_size_x;
            info->band_end_y[tid] = bands->band_size_y;
        }
    }
}

#undef MIN
#undef MAX

void free_bands_threading_info(BandsThreadingInfo * info)
{
    memfree(info->band_start_y);
}

void init_input_config(InputConfig * input_config)
{
    input_config->blocks = 0;
    input_config->num_threads = 1;
    input_config->image_size_x = 512;
    input_config->image_size_y = 512;
    input_config->tile_size_x = 512;
    input_config->tile_size_y = 512;
}

void init_transform_arguments(TransformArguments * ta, const InputConfig * ic)
{
    ta->input = memalloc(MEM_ALIGN_BYTES, sizeof(Image));
    ta->output = memalloc(MEM_ALIGN_BYTES, sizeof(Bands));
    ta->tmp = memalloc(MEM_ALIGN_BYTES, sizeof(Bands));
    ta->threading_info = memalloc(MEM_ALIGN_BYTES, sizeof(BandsThreadingInfo));

    allocate_image(ta->input, ic->image_size_x, ic->image_size_y);

    set_tile_size(ta->input, ic->tile_size_x, ic->tile_size_y);

    allocate_bands(ta->output, ta->input, 0);

    allocate_bands(ta->tmp, ta->input, TMP_BANDS_MARGIN);

    allocate_bands_threading_info(ta->threading_info, ic->num_threads);

    init_image(ta->input, ic->tile_size_x, ic->tile_size_y);
    init_bands_threading_info(ta->threading_info, ta->output, ic);
}

void free_transform_arguments(TransformArguments * transform_arguments)
{
    free_image(transform_arguments->input);
    free_bands(transform_arguments->output);
    free_bands(transform_arguments->tmp);
    free_bands_threading_info(transform_arguments->threading_info);

    memfree(transform_arguments->input);
    memfree(transform_arguments->output);
    memfree(transform_arguments->tmp);
    memfree(transform_arguments->threading_info);
}

void init_transform_step_arguments(TransformStepArguments *tsa, const TransformArguments * ta, size_t i, size_t j)
{
    tsa->threading_info = ta->threading_info;
    get_tile(ta->input, &(tsa->tile), j, i);
    get_tile_bands(ta->output, &(tsa->tile_bands), j, i);
    get_tile_bands(ta->tmp, &(tsa->tmp), 0, 0);
}

void init_transform_scheme(Transform * t, size_t num_steps, ...)
{
    t->num_steps = num_steps;
    t->steps = (TRANSFORM_STEP_FUNC *) memalloc(MEM_ALIGN_BYTES, sizeof(TRANSFORM_STEP_FUNC) * num_steps);

    va_list valist;
    va_start(valist, num_steps);

    for (size_t i = 0; i < num_steps; ++i)
      t->steps[i] = va_arg(valist, TRANSFORM_STEP_FUNC);

    va_end(valist);
}

void init_transform(Transform * t, Wavelet wavelet, Scheme scheme, Instruction_set instruction_set)
{
    t->wavelet = wavelet;
    t->scheme = scheme;
    t->instruction_set = instruction_set;

    switch(scheme){
        case SEP_LIFTING:
            switch(wavelet){
                case CDF_5_3:
                    switch(instruction_set){
                        case GENERIC:
                            t->transform_tile = cdf53_separable_lifting_generic_transform_tile;
                            init_transform_scheme(t, 4,
                                cdf53_separable_lifting_generic(0),
                                cdf53_separable_lifting_generic(1),
                                cdf53_separable_lifting_generic(2),
                                cdf53_separable_lifting_generic(3));
                            break;
                        case SSE:
#ifdef USE_SSE
                            t->transform_tile = cdf53_separable_lifting_amd64_sse_transform_tile;
                            init_transform_scheme(t, 4,
                                cdf53_separable_lifting_amd64_sse(0),
                                cdf53_separable_lifting_amd64_sse(1),
                                cdf53_separable_lifting_amd64_sse(2),
                                cdf53_separable_lifting_amd64_sse(3));
#endif
                            break;
                        case AVX_512:
#ifdef USE_AVX_512
                            t->transform_tile = cdf53_separable_lifting_avx_512_transform_tile;
                            init_transform_scheme(t, 4,
                                cdf53_separable_lifting_avx_512(0),
                                cdf53_separable_lifting_avx_512(1),
                                cdf53_separable_lifting_avx_512(2),
                                cdf53_separable_lifting_avx_512(3));
#endif
                            break;
                    }
                    break;
                case CDF_9_7:
                    switch(instruction_set){
                        case GENERIC:
                            t->transform_tile = cdf97_separable_lifting_generic_transform_tile;
                            init_transform_scheme(t, 8,
                                cdf97_separable_lifting_generic(0),
                                cdf97_separable_lifting_generic(1),
                                cdf97_separable_lifting_generic(2),
                                cdf97_separable_lifting_generic(3),
                                cdf97_separable_lifting_generic(4),
                                cdf97_separable_lifting_generic(5),
                                cdf97_separable_lifting_generic(6),
                                cdf97_separable_lifting_generic(7));
                            break;
                        case SSE:
#ifdef USE_SSE
                            t->transform_tile = cdf97_separable_lifting_amd64_sse_transform_tile;
                            init_transform_scheme(t, 8,
                                cdf97_separable_lifting_amd64_sse(0),
                                cdf97_separable_lifting_amd64_sse(1),
                                cdf97_separable_lifting_amd64_sse(2),
                                cdf97_separable_lifting_amd64_sse(3),
                                cdf97_separable_lifting_amd64_sse(4),
                                cdf97_separable_lifting_amd64_sse(5),
                                cdf97_separable_lifting_amd64_sse(6),
                                cdf97_separable_lifting_amd64_sse(7));
#endif
                            break;
                        case AVX_512:
#ifdef USE_AVX_512
                            t->transform_tile = cdf97_separable_lifting_avx_512_transform_tile;
                            init_transform_scheme(t, 8,
                                cdf97_separable_lifting_avx_512(0),
                                cdf97_separable_lifting_avx_512(1),
                                cdf97_separable_lifting_avx_512(2),
                                cdf97_separable_lifting_avx_512(3),
                                cdf97_separable_lifting_avx_512(4),
                                cdf97_separable_lifting_avx_512(5),
                                cdf97_separable_lifting_avx_512(6),
                                cdf97_separable_lifting_avx_512(7));
#endif
                            break;
                    }
                    break;
                case DD_13_7:
                    switch(instruction_set){
                        case GENERIC:
                            t->transform_tile = dd137_separable_lifting_generic_transform_tile;
                            init_transform_scheme(t, 4,
                                dd137_separable_lifting_generic(0),
                                dd137_separable_lifting_generic(1),
                                dd137_separable_lifting_generic(2),
                                dd137_separable_lifting_generic(3));
                            break;
                        case SSE:
#ifdef USE_SSE
                            t->transform_tile = dd137_separable_lifting_amd64_sse_transform_tile;
                            init_transform_scheme(t, 4,
                                dd137_separable_lifting_amd64_sse(0),
                                dd137_separable_lifting_amd64_sse(1),
                                dd137_separable_lifting_amd64_sse(2),
                                dd137_separable_lifting_amd64_sse(3));
#endif
                            break;
                        case AVX_512:
#ifdef USE_AVX_512
                            t->transform_tile = dd137_separable_lifting_avx_512_transform_tile;
                            init_transform_scheme(t, 4,
                                dd137_separable_lifting_avx_512(0),
                                dd137_separable_lifting_avx_512(1),
                                dd137_separable_lifting_avx_512(2),
                                dd137_separable_lifting_avx_512(3));
#endif
                            break;
                    }
                    break;
                default:
                    break;
            }
            break;
        case NSP_LIFTING:
            switch(wavelet){
                case CDF_5_3:
                    switch(instruction_set){
                        case GENERIC:
                            t->transform_tile = cdf53_non_separable_lifting_generic_transform_tile;
                            init_transform_scheme(t, 2,
                                cdf53_non_separable_lifting_generic(0),
                                cdf53_non_separable_lifting_generic(1));
                            break;
                        case SSE:
#ifdef USE_SSE
                            t->transform_tile = cdf53_non_separable_lifting_amd64_sse_transform_tile;
                            init_transform_scheme(t, 2,
                                cdf53_non_separable_lifting_amd64_sse(0),
                                cdf53_non_separable_lifting_amd64_sse(1));
#endif
                            break;
                        case AVX_512:
#ifdef USE_AVX_512
                            t->transform_tile = cdf53_non_separable_lifting_avx_512_transform_tile;
                            init_transform_scheme(t, 2,
                                cdf53_non_separable_lifting_avx_512(0),
                                cdf53_non_separable_lifting_avx_512(1));
#endif
                            break;
                    }
                    break;
                case CDF_9_7:
                    switch(instruction_set){
                        case GENERIC:
                            t->transform_tile = cdf97_non_separable_lifting_generic_transform_tile;
                            init_transform_scheme(t, 4,
                                cdf97_non_separable_lifting_generic(0),
                                cdf97_non_separable_lifting_generic(1),
                                cdf97_non_separable_lifting_generic(2),
                                cdf97_non_separable_lifting_generic(3));
                            break;
                        case SSE:
#ifdef USE_SSE
                            t->transform_tile = cdf97_non_separable_lifting_amd64_sse_transform_tile;
                            init_transform_scheme(t, 4,
                                cdf97_non_separable_lifting_amd64_sse(0),
                                cdf97_non_separable_lifting_amd64_sse(1),
                                cdf97_non_separable_lifting_amd64_sse(2),
                                cdf97_non_separable_lifting_amd64_sse(3));
#endif
                            break;
                        case AVX_512:
#ifdef USE_AVX_512
                            t->transform_tile = cdf97_non_separable_lifting_avx_512_transform_tile;
                            init_transform_scheme(t, 4,
                                cdf97_non_separable_lifting_avx_512(0),
                                cdf97_non_separable_lifting_avx_512(1),
                                cdf97_non_separable_lifting_avx_512(2),
                                cdf97_non_separable_lifting_avx_512(3));
#endif
                            break;
                    }
                    break;
                case DD_13_7:
                    switch(instruction_set){
                        case GENERIC:
                            t->transform_tile = dd137_non_separable_lifting_generic_transform_tile;
                            init_transform_scheme(t, 2,
                                dd137_non_separable_lifting_generic(0),
                                dd137_non_separable_lifting_generic(1));
                            break;
                        case SSE:
#ifdef USE_SSE
                            t->transform_tile = dd137_non_separable_lifting_amd64_sse_transform_tile;
                            init_transform_scheme(t, 2,
                                dd137_non_separable_lifting_amd64_sse(0),
                                dd137_non_separable_lifting_amd64_sse(1));
#endif
                            break;
                        case AVX_512:
#ifdef USE_AVX_512
                            t->transform_tile = dd137_non_separable_lifting_avx_512_transform_tile;
                            init_transform_scheme(t, 2,
                                dd137_non_separable_lifting_avx_512(0),
                                dd137_non_separable_lifting_avx_512(1));
#endif
                            break;
                    }
                    break;
                default:
                    break;
            }
            break;
        case NSP_CONVOLUTION_AT:
            switch(wavelet) {
                case CDF_5_3:
                    switch(instruction_set) {
                        case GENERIC:
                            t->transform_tile = cdf53_non_separable_convolution_at_generic_transform_tile;
                            init_transform_scheme(t, 1, cdf53_non_separable_convolution_at_generic(0));
                            break;
                        case SSE:
#ifdef USE_SSE
                            t->transform_tile = cdf53_non_separable_convolution_at_amd64_sse_transform_tile;
                            init_transform_scheme(t, 1, cdf53_non_separable_convolution_at_amd64_sse(0));
#endif
                            break;
                        case AVX_512:
#ifdef USE_AVX_512
                            t->transform_tile = cdf53_non_separable_convolution_at_avx_512_transform_tile;
                            init_transform_scheme(t, 1, cdf53_non_separable_convolution_at_avx_512(0));
#endif
                            break;
                    }
                    break;
                case CDF_9_7:
                    switch(instruction_set) {
                        case GENERIC:
                            t->transform_tile = cdf97_non_separable_convolution_at_generic_transform_tile;
                            init_transform_scheme(t, 1, cdf97_non_separable_convolution_at_generic(0));
                            break;
                        case SSE:
#ifdef USE_SSE
                            t->transform_tile = cdf97_non_separable_convolution_at_amd64_sse_transform_tile;
                            init_transform_scheme(t, 1, cdf97_non_separable_convolution_at_amd64_sse(0));
#endif
                            break;
                        case AVX_512:
#ifdef USE_AVX_512
                            t->transform_tile = cdf97_non_separable_convolution_at_avx_512_transform_tile;
                            init_transform_scheme(t, 1, cdf97_non_separable_convolution_at_avx_512(0));
#endif
                            break;
                    }
                    break;
                case DD_13_7:
                    switch(instruction_set) {
                        case GENERIC:
                            t->transform_tile = dd137_non_separable_convolution_at_generic_transform_tile;
                            init_transform_scheme(t, 1, dd137_non_separable_convolution_at_generic(0));
                            break;
                        case SSE:
#ifdef USE_SSE
                            t->transform_tile = dd137_non_separable_convolution_at_amd64_sse_transform_tile;
                            init_transform_scheme(t, 1, dd137_non_separable_convolution_at_amd64_sse(0));
#endif
                            break;
                case AVX_512:
#ifdef USE_AVX_512
                            t->transform_tile = dd137_non_separable_convolution_at_avx_512_transform_tile;
                            init_transform_scheme(t, 1, dd137_non_separable_convolution_at_avx_512(0));
#endif
                            break;
                    }
                    break;
                default:
                    break;
            }
            break;
        case NSP_CONVOLUTION_STAR:
            switch(wavelet) {
                case CDF_5_3:
                    switch(instruction_set) {
                        case GENERIC:
                            t->transform_tile = cdf53_non_separable_convolution_star_generic_transform_tile;
                            init_transform_scheme(t, 1, cdf53_non_separable_convolution_star_generic(0));
                            break;
                        case SSE:
#ifdef USE_SSE
                            t->transform_tile = cdf53_non_separable_convolution_star_amd64_sse_transform_tile;
                            init_transform_scheme(t, 1, cdf53_non_separable_convolution_star_amd64_sse(0));
#endif
                            break;
                        case AVX_512:
#ifdef USE_AVX_512
                            t->transform_tile = cdf53_non_separable_convolution_star_avx_512_transform_tile;
                            init_transform_scheme(t, 1, cdf53_non_separable_convolution_star_avx_512(0));
#endif
                            break;
                    }
                    break;
                case CDF_9_7:
                    switch(instruction_set) {
                        case GENERIC:
                            t->transform_tile = cdf97_non_separable_convolution_star_generic_transform_tile;
                            init_transform_scheme(t, 1, cdf97_non_separable_convolution_star_generic(0));
                            break;
                        case SSE:
#ifdef USE_SSE
                            t->transform_tile = cdf97_non_separable_convolution_star_amd64_sse_transform_tile;
                            init_transform_scheme(t, 1, cdf97_non_separable_convolution_star_amd64_sse(0));
#endif
                            break;
                        case AVX_512:
#ifdef USE_AVX_512
                            t->transform_tile = cdf97_non_separable_convolution_star_avx_512_transform_tile;
                            init_transform_scheme(t, 1, cdf97_non_separable_convolution_star_avx_512(0));
#endif
                            break;
                    }
                    break;
                case DD_13_7:
                    switch(instruction_set) {
                        case GENERIC:
                            t->transform_tile = dd137_non_separable_convolution_star_generic_transform_tile;
                            init_transform_scheme(t, 1, dd137_non_separable_convolution_star_generic(0));
                            break;
                        case SSE:
#ifdef USE_SSE
                            t->transform_tile = dd137_non_separable_convolution_star_amd64_sse_transform_tile;
                            init_transform_scheme(t, 1, dd137_non_separable_convolution_star_amd64_sse(0));
#endif
                            break;
                        case AVX_512:
#ifdef USE_AVX_512
                            t->transform_tile = dd137_non_separable_convolution_star_avx_512_transform_tile;
                            init_transform_scheme(t, 1, dd137_non_separable_convolution_star_avx_512(0));
#endif
                            break;
                    }
                    break;
                default:
                    break;
            }
            break;
        case NSP_POLYCONVOLUTION:
            switch(wavelet) {
                case CDF_9_7:
                    switch(instruction_set) {
                        case GENERIC:
                            t->transform_tile = cdf97_non_separable_polyconvolution_generic_transform_tile;
                            init_transform_scheme(t, 1, cdf97_non_separable_polyconvolution_generic(0));
                            break;
                        case SSE:
#ifdef USE_SSE
                            t->transform_tile = cdf97_non_separable_polyconvolution_amd64_sse_transform_tile;
                            init_transform_scheme(t, 1, cdf97_non_separable_polyconvolution_amd64_sse(0));
#endif
                            break;
                        case AVX_512:
#ifdef USE_AVX_512
                            t->transform_tile = cdf97_non_separable_polyconvolution_avx_512_transform_tile;
                            init_transform_scheme(t, 1, cdf97_non_separable_polyconvolution_avx_512(0));
#endif
                            break;
                    }
                    break;
                default:
                    break;
            }
            break;
        case SINGLE_LOOP:
            switch(wavelet) {
                case HAAR:
                    switch(instruction_set) {
                        case GENERIC:
                            t->transform_tile = haar_single_loop_generic_transform_tile;
                            init_transform_scheme(t, 1, haar_single_loop_generic(0));
                            break;
                        case SSE:
#ifdef USE_SSE
                            t->transform_tile = haar_single_loop_amd64_sse_transform_tile;
                            init_transform_scheme(t, 1, haar_single_loop_amd64_sse(0));
#endif
                            break;
                        case AVX_512:
                            break;
                    }
                    break;
                default:
                    break;
            }
            break;
            
    }
}

void free_transform(Transform * t)
{
    memfree(t->steps);
}

Transform * init_all_transforms()
{
    Transform * transforms = (Transform *) memalloc(MEM_ALIGN_BYTES, sizeof(Transform) * NUM_TRANSFORMS);

    size_t i = 0;
    init_transform(&(transforms[i++]), CDF_5_3, SEP_LIFTING, GENERIC);
    init_transform(&(transforms[i++]), CDF_5_3, NSP_LIFTING, GENERIC);
    init_transform(&(transforms[i++]), CDF_5_3, NSP_CONVOLUTION_AT, GENERIC);
    init_transform(&(transforms[i++]), CDF_5_3, NSP_CONVOLUTION_STAR, GENERIC);
    init_transform(&(transforms[i++]), CDF_9_7, SEP_LIFTING, GENERIC);
    init_transform(&(transforms[i++]), CDF_9_7, NSP_LIFTING, GENERIC);
    init_transform(&(transforms[i++]), CDF_9_7, NSP_CONVOLUTION_AT, GENERIC);
    init_transform(&(transforms[i++]), CDF_9_7, NSP_CONVOLUTION_STAR, GENERIC);
    init_transform(&(transforms[i++]), CDF_9_7, NSP_POLYCONVOLUTION, GENERIC);
    init_transform(&(transforms[i++]), DD_13_7, SEP_LIFTING, GENERIC);
    init_transform(&(transforms[i++]), DD_13_7, NSP_LIFTING, GENERIC);
    init_transform(&(transforms[i++]), DD_13_7, NSP_CONVOLUTION_AT, GENERIC);
    init_transform(&(transforms[i++]), DD_13_7, NSP_CONVOLUTION_STAR, GENERIC);
    init_transform(&(transforms[i++]), HAAR, SINGLE_LOOP, GENERIC);

#ifdef USE_SSE
    init_transform(&(transforms[i++]), CDF_5_3, SEP_LIFTING, SSE);
    init_transform(&(transforms[i++]), CDF_5_3, NSP_LIFTING, SSE);
    init_transform(&(transforms[i++]), CDF_5_3, NSP_CONVOLUTION_AT, SSE);
    init_transform(&(transforms[i++]), CDF_5_3, NSP_CONVOLUTION_STAR, SSE);
    init_transform(&(transforms[i++]), CDF_9_7, SEP_LIFTING, SSE);
    init_transform(&(transforms[i++]), CDF_9_7, NSP_LIFTING, SSE);
    init_transform(&(transforms[i++]), CDF_9_7, NSP_CONVOLUTION_AT, SSE);
    init_transform(&(transforms[i++]), CDF_9_7, NSP_CONVOLUTION_STAR, SSE);
    init_transform(&(transforms[i++]), CDF_9_7, NSP_POLYCONVOLUTION, SSE);
    init_transform(&(transforms[i++]), DD_13_7, SEP_LIFTING, SSE);
    init_transform(&(transforms[i++]), DD_13_7, NSP_LIFTING, SSE);
    init_transform(&(transforms[i++]), DD_13_7, NSP_CONVOLUTION_AT, SSE);
    init_transform(&(transforms[i++]), DD_13_7, NSP_CONVOLUTION_STAR, SSE);
    init_transform(&(transforms[i++]), HAAR, SINGLE_LOOP, SSE);
#endif

#ifdef USE_AVX_512
    init_transform(&(transforms[i++]), CDF_5_3, SEP_LIFTING, AVX_512);
    init_transform(&(transforms[i++]), CDF_5_3, NSP_LIFTING, AVX_512);
    init_transform(&(transforms[i++]), CDF_5_3, NSP_CONVOLUTION_AT, AVX_512);
    init_transform(&(transforms[i++]), CDF_5_3, NSP_CONVOLUTION_STAR, AVX_512);
    init_transform(&(transforms[i++]), CDF_9_7, SEP_LIFTING, AVX_512);
    init_transform(&(transforms[i++]), CDF_9_7, NSP_LIFTING, AVX_512);
    init_transform(&(transforms[i++]), CDF_9_7, NSP_CONVOLUTION_AT, AVX_512);
    init_transform(&(transforms[i++]), CDF_9_7, NSP_CONVOLUTION_STAR, AVX_512);
    init_transform(&(transforms[i++]), CDF_9_7, NSP_POLYCONVOLUTION, AVX_512);
    init_transform(&(transforms[i++]), DD_13_7, SEP_LIFTING, AVX_512);
    init_transform(&(transforms[i++]), DD_13_7, NSP_LIFTING, AVX_512);
    init_transform(&(transforms[i++]), DD_13_7, NSP_CONVOLUTION_AT, AVX_512);
    init_transform(&(transforms[i++]), DD_13_7, NSP_CONVOLUTION_STAR, AVX_512);
#endif

    return transforms;
}

void free_all_transforms(Transform * transforms)
{
    for(size_t t = 0; t < NUM_TRANSFORMS; ++t){
        free_transform(&(transforms[t]));
    }
    memfree(transforms);
}

extern inline size_t mirr(long pos, size_t size) {
    if (pos < 0) {
        pos *= -1;
    }
    if (pos > (long)size - 1) {
        pos = 2 * ((long)size - 1) - pos;
    }
    return (size_t)pos;
}

extern inline size_t mirr_band(long pos, size_t size, int lFlag) {
    if (pos < 0) {
        pos *= -1;
        if (!lFlag)
            pos--;
    }
    if (pos > (long)size - 1) {
        pos = 2 * ((long)size - 1) - pos;
        if (lFlag)
            pos++;
    }
    return (size_t)pos;
}

void dwt(const Transform * t, const TransformArguments * ta)
{
    Chunk chunk;
    init_chunk(&chunk, ta->input, 0);

    #pragma omp parallel PROC_BIND_CLOSE num_threads(ta->threading_info->num_threads) if (ta->threading_info->num_threads > 1)
    {
        for(size_t idx = chunk.start_index; idx < chunk.end_index; ++idx){
            size_t i = idx / ta->input->tiles_per_width;
            size_t j = idx % ta->input->tiles_per_width;

            TransformStepArguments tsa;
            init_transform_step_arguments(&tsa, ta, i, j);

            t->transform_tile(&tsa);
        }
    }
}
