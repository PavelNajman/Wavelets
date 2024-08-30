#include <test.h>

#include <math.h>
#include <string.h>
#include <assert.h>

#ifdef _OPENMP
    #include <omp.h>
#endif

static void compare_tiles(const TileBands tile_bands, const Tile reference)
{
    float n1, n2;
    float *ll, *hl, *lh, *hh;

    const float EPS = 1e-4f;
    for (size_t i = 0; i < reference.size_y; ++i) {
        for (size_t j = 0; j < reference.size_x; j += 2) {
            n1 = *(reference.data + i * reference.stride_y + j);
            n2 = *(reference.data + i * reference.stride_y + j + 1);

            if (i % 2 == 0) {
                ll = tile_bands.LL + (i>>1) * tile_bands.stride_y + (j >> 1);
                hl = tile_bands.HL + (i>>1) * tile_bands.stride_y + (j >> 1);
                if(fabsf(*ll - n1) >= EPS)
                    fprintf(stderr,"ll %zu %zu %f %f\n", j, i, (double)*ll, (double)n1);
                if(fabsf(*hl - n2) >= EPS)
                    fprintf(stderr,"hl %zu %zu %f %f\n", j, i, (double)*hl, (double)n2);
                assert(fabsf(*ll - n1) < EPS);
                assert(fabsf(*hl - n2) < EPS);
            } else {
                lh = tile_bands.LH + (i>>1) * tile_bands.stride_y + (j >> 1);
                hh = tile_bands.HH + (i>>1) * tile_bands.stride_y + (j >> 1);
                if(fabsf(*lh - n1) >= EPS)
                    fprintf(stderr,"lh %zu %zu %f %f\n", j, i, (double)*lh, (double)n1);
                if(fabsf(*hh - n2) >= EPS)
                    fprintf(stderr,"hh %zu %zu %f %f\n", j, i, (double)*hh, (double)n2);
                assert(fabsf(*lh - n1) < EPS);
                assert(fabsf(*hh - n2) < EPS);
            }
        }
    }
}

static void compare_images(const Image reference, const Bands * bands)
{
    Tile tile;
    TileBands tile_bands;

    size_t num_tiles = reference.tiles_per_height * reference.tiles_per_width;
    size_t chunk = num_tiles;

    for (size_t y = 0; y < reference.tiles_per_height; ++y) {
        for (size_t x = 0; x < reference.tiles_per_width; ++x) {
            get_tile(&reference, &tile, x, y);
            size_t idx = (y * reference.tiles_per_width + x) / chunk;
            get_tile_bands(&bands[idx], &tile_bands, x, y);
            //fprintf(stderr, "%zu %zu\n", x, y);
            compare_tiles(tile_bands, tile);
        }
    }
}

static void test(const Transform * t, const TransformArguments * ta)
{
    char filename[50];
    sprintf(filename, "%d_%zu_%zu.mat", (int) t->wavelet, ta->input->size_x, ta->input->tile_size_x);

    FILE * f = fopen(filename, "r");
    if(!f)
        return;
    fclose(f);

    fprintf(stderr, "Overall test (%s) ... ", filename);

    Image test;
    allocate_image(&test, ta->input->size_x, ta->input->size_y);
    set_tile_size(&test, ta->input->tile_size_x, ta->input->tile_size_y);

    load_image(filename, &test);

    clear_bands(ta->output);
    clear_bands(ta->tmp);

    dwt(t, ta);

    char path_buff[4096];
    sprintf(path_buff, "out_%d_%zu_%zu.mat", (int) t->wavelet, ta->input->size_x, ta->input->tile_size_x);
    FILE *file = fopen(path_buff, "w");
    assert(file);
    for (size_t ly = 0; ly < ta->output->size_y; ly++) {
        for (size_t lx = 0; lx < ta->output->size_x; lx++) {
            fprintf(file, "%+f%s", (double)ta->output->LL[ly * ta->output->stride_y + lx], ",\t");
            fprintf(file, "%+f%s", (double)ta->output->HL[ly * ta->output->stride_y + lx], (lx == ta->output->size_x - 1) ? "\n" : ",\t");
        }
        for (size_t lx = 0; lx < ta->output->size_x; lx++) {
            fprintf(file, "%+f%s", (double)ta->output->LH[ly * ta->output->stride_y + lx], ",\t");
            fprintf(file, "%+f%s", (double)ta->output->HH[ly * ta->output->stride_y + lx], (lx == ta->output->size_x - 1) ? "\n" : ",\t");
        }
    }
    fclose(file);

    compare_images(test, ta->output);

    free_image(&test);

    fprintf(stderr, "OK\n");
}

static void test_steps(const Transform * t, const TransformArguments * ta)
{
    int stop = 0;
    char filename[50];

    Image test;
    allocate_image(&test, ta->input->tile_size_x, ta->input->tile_size_y);
    set_tile_size(&test, ta->input->tile_size_x, ta->input->tile_size_y);

    Tile test_tile;
    get_tile(&test, &test_tile, 0, 0);

    size_t tid = 0;

    Chunk chunk;
    init_chunk(&chunk, ta->input, tid);

    #   pragma omp parallel num_threads(ta->threading_info->num_threads) if(ta->threading_info->num_threads > 1)
    {
        for(size_t idx = chunk.start_index; idx < chunk.end_index; ++idx){
            size_t i = idx / ta->input->tiles_per_width;
            size_t j = idx % ta->input->tiles_per_width;

            TransformStepArguments tsa;
            init_transform_step_arguments(&tsa, ta, i, j);

            for(size_t s = 0; s < t->num_steps; ++s){
                t->steps[s](&tsa);

                #pragma omp barrier
                #pragma omp master
                {

                    sprintf(filename, "%d_%d_%zu_%zu_%zu_%zu.mat", (int) t->scheme, (int) t->wavelet, ta->input->size_x, ta->input->tile_size_x, idx, s);
                    FILE * f = fopen(filename, "r");
                    if(!f){
                        stop = 1;
                    }
                    else{
                        fprintf(stderr, "Step %zu (%s) ... ", s, filename);

                        fclose(f);
                        load_image(filename, &test);

                        if(s == 0 && t->scheme == NSP_LIFTING){
                            const size_t num_tmp_cache_lines = ((tsa.tmp.size_y * tsa.tmp.size_x) * sizeof(float)) / CACHE_LINE_SIZE;
                            if(t->wavelet == CDF_5_3 && t->instruction_set == GENERIC){
                                for(size_t i = 0; i < ta->threading_info->num_threads; ++i){
                                    const size_t tmp_offset = i * (num_tmp_cache_lines / (size_t) omp_get_num_threads()) * (CACHE_LINE_SIZE / sizeof(float));
                                    if(ta->threading_info->band_end_x[i] != tsa.tile_bands.size_x){
                                        float * tmp_hl = tsa.tmp.HL + tmp_offset;
                                        float * hl = tsa.tile_bands.HL + ta->threading_info->band_start_y[i] * tsa.tile_bands.stride_y + ta->threading_info->band_end_x[i] - 1;
                                        for(size_t y = ta->threading_info->band_start_y[i]; y < ta->threading_info->band_end_y[i]; ++y){
                                            *hl = *tmp_hl;
                                            tmp_hl++;
                                            hl += tsa.tile_bands.stride_y;
                                        }
                                    }
                                    float * tmp_lh = tsa.tmp.LH + tmp_offset;
                                    memcpy(tsa.tile_bands.LH + (ta->threading_info->band_end_y[i] - 1) * tsa.tile_bands.stride_y + ta->threading_info->band_start_x[i],
                                            tmp_lh,
                                            (ta->threading_info->band_end_x[i] - ta->threading_info->band_start_x[i]) * sizeof(float));
                                }
                            } else if(t->wavelet == CDF_5_3 && t->instruction_set == SSE){
                                for(size_t i = 0; i < ta->threading_info->num_threads; ++i){
                                    const size_t tmp_offset = i * (num_tmp_cache_lines / (size_t) omp_get_num_threads()) * (CACHE_LINE_SIZE / sizeof(float));
                                    if(ta->threading_info->band_end_x[i] != tsa.tile_bands.size_x){
                                        float * tmp_hl = tsa.tmp.HL + tmp_offset;
                                        float * hl = tsa.tile_bands.HL + ta->threading_info->band_start_y[i] * tsa.tile_bands.stride_y + ta->threading_info->band_end_x[i] - 4;
                                        for(size_t y = ta->threading_info->band_start_y[i]; y < ta->threading_info->band_end_y[i]; ++y){
                                            hl[0] = tmp_hl[0];
                                            hl[1] = tmp_hl[1];
                                            hl[2] = tmp_hl[2];
                                            hl[3] = tmp_hl[3];
                                            tmp_hl += 4;
                                            hl += tsa.tile_bands.stride_y;
                                        }
                                    }
                                    float * tmp_lh = tsa.tmp.LH + tmp_offset;
                                    memcpy(tsa.tile_bands.LH + (ta->threading_info->band_end_y[i] - 1) * tsa.tile_bands.stride_y + ta->threading_info->band_start_x[i],
                                            tmp_lh,
                                            (ta->threading_info->band_end_x[i] - ta->threading_info->band_start_x[i]) * sizeof(float));
                                }
                            }
                        }

                        compare_tiles(tsa.tile_bands, test_tile);

                        fprintf(stderr, "OK\n");
                    }
                }
                #pragma omp barrier
                if(stop) break;
            }
        }
    }
    free_image(&test);
}

void test_all_transforms(const Transform * transforms, const TransformArguments * ta)
{
    fprintf(stderr, "===================== THREADING INFO =====================\n");
    fprintf(stderr, "Threads (width x height): %zu x %zu\n", ta->threading_info->thread_cols, ta->threading_info->thread_rows);
    for(size_t tid = 0; tid < ta->threading_info->num_threads; ++tid){
        fprintf(stderr, "Thread %zu - X: %zu - %zu Y: %zu - %zu\n", tid, ta->threading_info->band_start_x[tid], ta->threading_info->band_end_x[tid], ta->threading_info->band_start_y[tid], ta->threading_info->band_end_y[tid]);
    }
    fprintf(stderr, "==========================================================\n\n");
    
    for(size_t t = 0; t < NUM_TRANSFORMS; ++t){
        fprintf(stderr, "========================== TEST ==========================\n");
        fprintf(stderr, "WAWELET: %s\nSCHEME: %s\nINSTRUCTION SET: %s\n",
                WaveletNames[transforms[t].wavelet],
                SchemeNames[transforms[t].scheme],
                InstructionSetNames[transforms[t].instruction_set]);

        test_steps(transforms + t, ta);

        test(transforms + t, ta);

        fprintf(stderr, "==========================================================\n\n");
    }
}
