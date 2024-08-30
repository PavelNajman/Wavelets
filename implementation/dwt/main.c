#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <common.h>
#include <test.h>
#include <benchmark.h>
#include <string.h>

typedef struct
{
    InputConfig input_config;
    #ifndef DEBUG
        BenchmarkConfig benchmark_config;
    #endif
} ProgramArguments;

static void process_program_arguments(ProgramArguments * arguments, int argc, char **argv);

int main(int argc, char** argv) {
    ProgramArguments pa;
    process_program_arguments(&pa, argc, argv);
    
    TransformArguments ta;
    init_transform_arguments(&ta, &pa.input_config);

    Transform * transforms = init_all_transforms();

    #if defined(DEBUG)
        test_all_transforms(transforms, &ta);
    #endif

    #ifndef DEBUG
        benchmark_all_transforms(transforms, &ta, &pa.benchmark_config);
    #endif

    free_all_transforms(transforms);

    free_transform_arguments(&ta);
    
    return EXIT_SUCCESS;
}

static void print_help()
{
    fprintf(stderr, "\nMAIN OPTIONS\n");
    fprintf(stderr, "--help\n");
    fprintf(stderr, "\nINPUT CONFIGURATION OPTIONS\n");
    fprintf(stderr, "--blocks\n");
    fprintf(stderr, "--threads\n");
    fprintf(stderr, "--image-width\n");
    fprintf(stderr, "--image-height\n");
    fprintf(stderr, "--tile-width\n");
    fprintf(stderr, "--tile-height\n");
    
    #ifndef DEBUG
        fprintf(stderr, "\nBENCHMARK CONFIGURATION OPTIONS\n");
        fprintf(stderr, "-h\n");
        fprintf(stderr, "--benchmark-type\n");
        fprintf(stderr, "--benchmark-threads\n");
        fprintf(stderr, "--benchmark-tests\n");
        fprintf(stderr, "--benchmark-attempts\n");
        fprintf(stderr, "--benchmark-repetitions\n");  
    #endif
}

static int process_main_options(char *arg)
{
    int result = 0;
    const size_t num_main_arguments = 1;
    const char* main_arguments[] = {"--help"};
    for(size_t j = 0; j < num_main_arguments; j++){
        if(strncmp(main_arguments[j], arg, strlen(main_arguments[j])) == 0){
            result = 1;
            switch(j){
                case 0:
                    print_help();
                    exit(EXIT_SUCCESS);
                    break;
                default:
                    break;
            }
        }
    }
    return result;
}

static int process_input_configuration_options(ProgramArguments * pa, char *arg)
{
    int result = 0;
    const size_t num_input_config_arguments = 6;
    const char* input_config_arguments[] = {"--blocks", "--threads=", "--image-width=", "--image-height=", "--tile-width=", "--tile-height="};
    
    for(size_t j = 0; j < num_input_config_arguments; j++){
        size_t n = strlen(input_config_arguments[j]);
        if(strncmp(input_config_arguments[j], arg, n) == 0){
            result = 1;
            switch(j){
                case 0:
                    //stripes
                    pa->input_config.blocks = 1;
                    break;
                case 1:
                    //threads
                    pa->input_config.num_threads = (size_t) atoi(&(arg[n]));
                    break;
                case 2:
                    //image-width
                    pa->input_config.image_size_x = (size_t) atoi(&(arg[n]));
                    break;
                case 3:
                    //image-height
                    pa->input_config.image_size_y = (size_t) atoi(&(arg[n]));
                    break;
                case 4:
                    //tile-width
                    pa->input_config.tile_size_x = (size_t) atoi(&(arg[n]));
                    break;
                case 5:
                    //tile-height
                    pa->input_config.tile_size_y = (size_t) atoi(&(arg[n]));
                    break;
                default:
                    break;
            }
        }
    }
    return result;
}

#ifndef DEBUG

static int process_benchmark_configuration_options(ProgramArguments * pa, char *arg)
{
    int result = 0;
    const size_t num_benchmark_config_arguments = 6;
    const char* benchmark_config_arguments[] = {"-h", "--benchmark-type=", "--benchmark-threads=", "--benchmark-tests=", "--benchmark-attempts=", "--benchmark-repetitions="};
    
    for(size_t j = 0; j < num_benchmark_config_arguments; j++){
        size_t n = strlen(benchmark_config_arguments[j]);
        if(strncmp(benchmark_config_arguments[j], arg, n) == 0){
            result = 1;
            switch(j){
                case 0:
                    //h
                    pa->benchmark_config.human_readable = 1;
                    break;
                case 1:
                    //benchmark-type
                    pa->benchmark_config.type = (BenchmarkType) atoi(&(arg[n]));
                    break;
                case 2:
                    //benchmark-threads
                    pa->benchmark_config.threads = (size_t) atoi(&(arg[n]));
                    break;
                case 3:
                    //benchmark-tests
                    pa->benchmark_config.tests = (size_t) atoi(&(arg[n]));
                    break;
                case 4:
                    //benchmark-attempts
                    pa->benchmark_config.attempts = (size_t) atoi(&(arg[n]));
                    break;
                case 5:
                    //benchmark-repetitions
                    pa->benchmark_config.repetitions = (size_t) atoi(&(arg[n]));
                    break;
                default:
                    break;
            }
        }
    }
    return result;
}

#endif

static void process_program_arguments(ProgramArguments * pa, int argc, char **argv)
{
    init_input_config(&(pa->input_config));
    #ifndef DEBUG
        init_benchmark_config(&(pa->benchmark_config));
    #endif
    
    for(int i = 1; i < argc; ++i){
        int processed = 0;
        processed |= process_main_options(argv[i]);
        processed |= process_input_configuration_options(pa, argv[i]);
        
        #ifndef DEBUG
            processed |= process_benchmark_configuration_options(pa, argv[i]);
        #endif
        if(!processed){
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            exit(EXIT_FAILURE);
        }
    }
}