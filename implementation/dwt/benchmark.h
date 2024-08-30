#ifndef MEASURE_H
#define MEASURE_H

#include "common.h"

typedef enum {IMAGE, TILE, TILES_IN_IMAGE} BenchmarkType;

typedef struct
{
    size_t human_readable;
    BenchmarkType type;
    size_t threads, tests, attempts, repetitions;   
} BenchmarkConfig;

void init_benchmark_config(BenchmarkConfig * benchmark_config);

void benchmark_all_transforms(const Transform * transforms, const TransformArguments * ta, const BenchmarkConfig * bc);

#endif // MEASURE_H
