#include <benchmark.h>

#include <time.h>
#include <assert.h>
#include <malloc.h>
#include <stdlib.h>

#ifdef _WIN32
#include <Windows.h>
#define CLOCK_MONOTONIC_RAW 0
LARGE_INTEGER getFILETIMEoffset()
{
    SYSTEMTIME s;
    FILETIME f;
    LARGE_INTEGER t;
    s.wYear = 1970;
    s.wMonth = 1;
    s.wDay = 1;
    s.wHour = 0;
    s.wMinute = 0;
    s.wSecond = 0;
    s.wMilliseconds = 0;
    SystemTimeToFileTime(&s, &f);
    t.QuadPart = f.dwHighDateTime;
    t.QuadPart <<= 32;
    t.QuadPart |= f.dwLowDateTime;
    return (t);
}

int clock_gettime(int x, struct timespec *tv)
{
    (void)(x);
    LARGE_INTEGER           t;
    FILETIME                f;
    double                  microseconds;
    double                  nanoseconds;
    static LARGE_INTEGER    offset;
    static double           frequencyToMicroseconds;
    static int              initialized = 0;
    static BOOL             usePerformanceCounter = 0;

    if (!initialized) {
        LARGE_INTEGER performanceFrequency;
        initialized = 1;
        usePerformanceCounter = QueryPerformanceFrequency(&performanceFrequency);
        if (usePerformanceCounter) {
            QueryPerformanceCounter(&offset);
            frequencyToMicroseconds = (double)performanceFrequency.QuadPart / 1000000.;
        } else {
            offset = getFILETIMEoffset();
            frequencyToMicroseconds = 10.;
        }
    }

    if (usePerformanceCounter) {
        QueryPerformanceCounter(&t);
    } else {
        GetSystemTimeAsFileTime(&f);
        t.QuadPart = f.dwHighDateTime;
        t.QuadPart <<= 32;
        t.QuadPart |= f.dwLowDateTime;
    }

    t.QuadPart -= offset.QuadPart;
    microseconds = (double)t.QuadPart / frequencyToMicroseconds;
    nanoseconds = microseconds * 1000;
    t.QuadPart = (long)(microseconds);
    tv->tv_sec = (time_t)(microseconds / 1000000);
    tv->tv_nsec = (long)(nanoseconds - tv->tv_sec * 1000000000LL);
    return (0);
}
#endif

static long long gettimer()
{
    struct timespec t;
    if( -1 == clock_gettime(CLOCK_MONOTONIC_RAW, &t) )
        abort();
    return t.tv_sec * 1000000000LL + t.tv_nsec;
}

static int compare_times(const void *p1, const void * p2)
{
    long long t1 = *(long long*) p1;
    long long t2 = *(long long*) p2;

    long long diff = t1 - t2;
    if(diff < 0)
        return -1;
    if(diff > 0)
        return 1;
    return 0;
}

static void measure_image(const Transform * t, const TransformArguments * ta, const BenchmarkConfig * bc, double * results)
{
    const size_t repetitions = bc->repetitions;
    const size_t attempts = bc->attempts;

    for(size_t a = 0; a < attempts; ++a)
    {
        long long t0 = gettimer(); // ns

        for(size_t r = 0; r < repetitions; ++r){
            dwt(t, ta);
        }

        long long t1 = gettimer();

        long long t_ = t1 - t0;

        *(results + a) = 1000.0*(double)t_/(double)(ta->input->size_x * ta->input->size_y * repetitions);
    }
}

static void measure_tile(const Transform * t, const TransformArguments *ta, const BenchmarkConfig * bc, double * results)
{
    assert(ta->input->num_tiles == 1);

    const size_t repetitions = bc->repetitions;
    const size_t attempts = bc->attempts;

    TransformStepArguments tsa;
    init_transform_step_arguments(&tsa, ta, 0, 0);

    for(size_t a = 0; a < attempts; ++a)
    {
        long long t0 = gettimer(); // ns

        for(size_t r = 0; r < repetitions; ++r){
            #   pragma omp parallel PROC_BIND_CLOSE num_threads(ta->threading_info->num_threads) if (ta->threading_info->num_threads > 1)
            {
                t->transform_tile(&tsa);
            }
        }

        long long t1 = gettimer();

        long long t_ = t1 - t0;

        *(results + a) = 1000.0*(double)t_/(double)((tsa.tile.size_x * tsa.tile.size_y) * repetitions);
    }
}

static void measure_tiles_in_image(const Transform * t, const TransformArguments *ta, const BenchmarkConfig * bc, double * results)
{
    const size_t repetitions = bc->repetitions;
    const size_t attempts = bc->attempts;

    TransformStepArguments tsa;
    init_transform_step_arguments(&tsa, ta, 0, 0);

    for(size_t i = 0; i < ta->input->tiles_per_height; ++i){
        for(size_t j = 0; j < ta->input->tiles_per_width; ++j){
            Tile tile;
            TileBands tile_bands;

            get_tile(ta->input, &tile, j, i);
            get_tile_bands(ta->output, &tile_bands, j, i);

            for(size_t a = 0; a < attempts; ++a){
                long long t0 = gettimer(); // ns

                for(size_t r = 0; r < repetitions; ++r){
                    #   pragma omp parallel PROC_BIND_CLOSE num_threads(ta->threading_info->num_threads) if(ta->threading_info->num_threads > 1)
                    {
                        t->transform_tile(&tsa);
                    }
                }

                long long t1 = gettimer();

                long long t_ = t1 - t0;

                size_t idx = i * ta->input->tiles_per_width + j;
                *(results + idx * attempts + a) = 1000.0*(double)t_/(double)((tsa.tile.size_x * tsa.tile.size_y) * repetitions);
            }
        }
    }
}

static void benchmark(const Transform * t, const TransformArguments *ta, const BenchmarkConfig * bc, double * results)
{
    switch(bc->type){
        case IMAGE:
            measure_image(t, ta, bc, results);
            break;
        case TILE:
            measure_tile(t, ta, bc, results);
            break;
        case TILES_IN_IMAGE:
            measure_tiles_in_image(t, ta, bc, results);
            break;
    }
}

static void print_benchmark_results(const Transform * transforms, const TransformArguments * ta, const BenchmarkConfig * bc, double * results[NUM_TRANSFORMS])
{
    size_t num_results_per_transform = bc->tests * (bc->type == TILES_IN_IMAGE ? bc->attempts * ta->input->num_tiles : bc->attempts);

    for(size_t t = 0; t < NUM_TRANSFORMS; ++t){
        if(bc->human_readable){
            printf("%s\t%-32s\t%s\t\t", WaveletNames[transforms[t].wavelet], SchemeNames[transforms[t].scheme], InstructionSetNames[transforms[t].instruction_set]);
        }
        printf("%f", results[t][num_results_per_transform/2]/1000.0);
        ((t == NUM_TRANSFORMS - 1) || (bc->human_readable)) ? printf("\n") : printf("\t");
    }
}

void init_benchmark_config(BenchmarkConfig * benchmark_config)
{
    benchmark_config->human_readable = 0;
    benchmark_config->type = IMAGE;
    benchmark_config->threads = 0;
    benchmark_config->tests = 10;
    benchmark_config->attempts = 10;
    benchmark_config->repetitions = 10;
}

void benchmark_all_transforms(const Transform * transforms, const TransformArguments * ta, const BenchmarkConfig * bc)
{
    if(bc->threads > 0){
        InputConfig ic;
        ic.num_threads = ta->threading_info->num_threads;
        ic.image_size_x = ta->input->size_x;
        ic.image_size_y = ta->input->size_y;
        ic.tile_size_x = ta->input->tile_size_x;
        ic.tile_size_y = ta->input->tile_size_y;

        const size_t threads = ic.num_threads;
        double **results = memalloc(16, sizeof(double) * bc->threads * threads * NUM_TRANSFORMS);

        size_t *results_sizes = memalloc(16, sizeof(size_t) * threads);

        for (size_t threadTest = 0; threadTest < bc->threads; ++threadTest) {
            for (size_t thread = 0; thread < threads; ++thread) {
                ic.num_threads = thread + 1;

                TransformArguments ta;
                init_transform_arguments(&ta, &ic);

                Transform *transforms = init_all_transforms();

                if(bc->human_readable){
                    if (threadTest == 0 && thread == 0) {
                        for (size_t transform = 0; transform < NUM_TRANSFORMS; ++transform) {
                            printf("\t%s %s %s", WaveletNames[transforms[transform].wavelet], SchemeNames[transforms[transform].scheme], InstructionSetNames[transforms[transform].instruction_set]);
                        }
                        printf("\n");
                    }
                }

                if (threadTest == 0)
                    results_sizes[thread] = bc->type == TILES_IN_IMAGE ? bc->attempts * ta.input->num_tiles : bc->attempts;

                for (size_t transform = 0; transform < NUM_TRANSFORMS; ++transform) {
                    results[bc->threads * threads * transform + bc->threads * thread + threadTest] = memalloc(16, sizeof(double) * bc->tests * results_sizes[thread]);
                }

                for (size_t test = 0; test < bc->tests; ++test) {
                    for(size_t transform = 0; transform < NUM_TRANSFORMS; ++transform) {
                        benchmark(&(transforms[transform]), &ta, bc, &(results[bc->threads * threads * transform + bc->threads * thread + threadTest][test * results_sizes[thread]]));
                        fprintf(stderr, "\rDone %d%%", (int)(100 * ((float)(NUM_TRANSFORMS * bc->tests * threads * threadTest + NUM_TRANSFORMS * bc->tests * thread + NUM_TRANSFORMS * test + transform + 1) / (float)(NUM_TRANSFORMS * bc->tests * threads * bc->threads))));
                    }
                }

                /*for (size_t transform = 0; transform < NUM_TRANSFORMS; ++transform) {
                    qsort(results[threads * THREADS_TESTS * transform + THREADS_TESTS * thread + threadTest], TESTS * results_sizes[thread], sizeof(double), compare_times);
                }*/

                free_all_transforms(transforms);

                free_transform_arguments(&ta);
            }
        }

        fprintf(stderr, "\n");

        for (size_t thread = 0; thread < threads; ++thread) {
            printf("%zu", thread + 1);
            for (size_t transform = 0; transform < NUM_TRANSFORMS; ++transform) {
                double *threadTestResults = memalloc(16, sizeof(double) * bc->tests * results_sizes[thread] * bc->threads);
                for (size_t threadTest = 0; threadTest < bc->threads; ++threadTest) {
                    for (size_t test = 0; test < bc->tests; ++test) {
                        for (size_t size = 0; size < results_sizes[thread]; ++size) {
                            threadTestResults[bc->tests * results_sizes[thread] * threadTest + results_sizes[thread] * test + size] = results[threads * bc->threads * transform + bc->threads * thread + threadTest][test * results_sizes[thread] + size];
                        }
                    }
                }
                qsort(threadTestResults, bc->tests * results_sizes[thread] * bc->threads, sizeof(double), compare_times);
                // Minimum
                //printf("\t%f", threadTestResults[0] / 1000.0);
                // Median
                printf("\t%f", threadTestResults[(bc->tests * results_sizes[thread] * bc->threads) / 2] / 1000.0);
                // Maximum
                //printf("\t%f", threadTestResults[TESTS * results_sizes[thread] * THREADS_TESTS - 1] / 1000.0);
                memfree(threadTestResults);
            }
            printf("\n");
        }

        for (size_t result = 0; result < bc->threads * threads * NUM_TRANSFORMS; ++result) {
            memfree(results[result]);
        }

        memfree(results_sizes);
        memfree(results);
    } else {

        double * results[NUM_TRANSFORMS];

        size_t results_size = bc->type == TILES_IN_IMAGE ? bc->attempts * ta->input->num_tiles : bc->attempts;
        for(size_t t = 0; t < NUM_TRANSFORMS; ++t)
            results[t] = memalloc(MEM_ALIGN_BYTES, sizeof(double) * bc->tests * results_size);

        for(size_t ti = 0; ti < bc->tests; ++ti){
            for(size_t t = 0; t < NUM_TRANSFORMS; ++t){
                benchmark(&(transforms[t]), ta, bc, &(results[t][ti * results_size]));
            }
        }

        for(size_t t = 0; t < NUM_TRANSFORMS; ++t){
            qsort(results[t], bc->tests * results_size, sizeof(double), compare_times);
        }

        print_benchmark_results(transforms, ta, bc, results);

        for(size_t t = 0; t < NUM_TRANSFORMS; ++t)
            memfree(results[t]);
    }
}
