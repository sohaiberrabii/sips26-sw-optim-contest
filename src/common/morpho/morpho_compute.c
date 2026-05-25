#include <stdlib.h>
#include <assert.h>
#include <nrc2.h>

#include "motion/macros.h"
#include "motion/morpho/morpho_compute.h"

morpho_data_t* morpho_alloc_data(const int i0, const int i1, const int j0, const int j1) {
    morpho_data_t* morpho_data = (morpho_data_t*)malloc(sizeof(morpho_data_t));
    morpho_data->i0 = i0;
    morpho_data->i1 = i1;
    morpho_data->j0 = j0;
    morpho_data->j1 = j1;
    morpho_data->IB = ui8matrix(morpho_data->i0, morpho_data->i1, morpho_data->j0, morpho_data->j1);
    return morpho_data;
}

void morpho_init_data(morpho_data_t* morpho_data) {
    zero_ui8matrix(morpho_data->IB , morpho_data->i0, morpho_data->i1, morpho_data->j0, morpho_data->j1);
}

void morpho_free_data(morpho_data_t* morpho_data) {
    free_ui8matrix(morpho_data->IB, morpho_data->i0, morpho_data->i1, morpho_data->j0, morpho_data->j1);
    free(morpho_data);
}

static void morpho_copy_borders(const uint8_t** img_in, uint8_t** img_out, const int i0, const int i1,
                                const int j0, const int j1) {
    for (int i = i0; i <= i1; i++) {
        img_out[i][j0] = img_in[i][j0];
        img_out[i][j1] = img_in[i][j1];
    }
    for (int j = j0; j <= j1; j++) {
        img_out[i0][j] = img_in[i0][j];
        img_out[i1][j] = img_in[i1][j];
    }
}

static void morpho_compute_erosion3_horizontal(const uint8_t** img_in, uint8_t** img_out,
                                               const int i0, const int i1, const int j0, const int j1) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = i0; i <= i1; i++) {
        const uint8_t* in = img_in[i];
        uint8_t* out = img_out[i];
        out[j0] = in[j0];
        out[j1] = in[j1];
        for (int j = j0 + 1; j <= j1 - 1; j++)
            out[j] = in[j - 1] & in[j] & in[j + 1];
    }
}

static void morpho_compute_erosion3_vertical(const uint8_t** img_in, uint8_t** img_out, const uint8_t** borders,
                                             const int i0, const int i1, const int j0, const int j1) {
    morpho_copy_borders(borders, img_out, i0, i1, j0, j1);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = i0 + 1; i <= i1 - 1; i++) {
        const uint8_t* prev = img_in[i - 1];
        const uint8_t* cur = img_in[i];
        const uint8_t* next = img_in[i + 1];
        uint8_t* out = img_out[i];
        for (int j = j0 + 1; j <= j1 - 1; j++)
            out[j] = prev[j] & cur[j] & next[j];
    }
}

static void morpho_compute_dilation3_horizontal(const uint8_t** img_in, uint8_t** img_out,
                                                const int i0, const int i1, const int j0, const int j1) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = i0; i <= i1; i++) {
        const uint8_t* in = img_in[i];
        uint8_t* out = img_out[i];
        out[j0] = in[j0];
        out[j1] = in[j1];
        for (int j = j0 + 1; j <= j1 - 1; j++)
            out[j] = in[j - 1] | in[j] | in[j + 1];
    }
}

static void morpho_compute_dilation3_vertical(const uint8_t** img_in, uint8_t** img_out, const uint8_t** borders,
                                              const int i0, const int i1, const int j0, const int j1) {
    morpho_copy_borders(borders, img_out, i0, i1, j0, j1);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = i0 + 1; i <= i1 - 1; i++) {
        const uint8_t* prev = img_in[i - 1];
        const uint8_t* cur = img_in[i];
        const uint8_t* next = img_in[i + 1];
        uint8_t* out = img_out[i];
        for (int j = j0 + 1; j <= j1 - 1; j++)
            out[j] = prev[j] | cur[j] | next[j];
    }
}

static void morpho_compute_erosion3_separable(const uint8_t** img_in, uint8_t** tmp, uint8_t** img_out,
                                              const int i0, const int i1, const int j0, const int j1) {
    morpho_compute_erosion3_horizontal(img_in, tmp, i0, i1, j0, j1);
    morpho_compute_erosion3_vertical((const uint8_t**)tmp, img_out, img_in, i0, i1, j0, j1);
}

static void morpho_compute_dilation3_separable(const uint8_t** img_in, uint8_t** tmp, uint8_t** img_out,
                                               const int i0, const int i1, const int j0, const int j1) {
    morpho_compute_dilation3_horizontal(img_in, tmp, i0, i1, j0, j1);
    morpho_compute_dilation3_vertical((const uint8_t**)tmp, img_out, img_in, i0, i1, j0, j1);
}

void morpho_compute_erosion3(const uint8_t** img_in, uint8_t** img_out, const int i0, const int i1, const int j0,
                             const int j1) {
    assert(img_in != NULL);
    assert(img_out != NULL);
    assert(img_in != (const uint8_t**)img_out);

    morpho_copy_borders(img_in, img_out, i0, i1, j0, j1);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = i0 + 1; i <= i1 - 1; i++) {
        const uint8_t* prev = img_in[i - 1];
        const uint8_t* cur = img_in[i + 0];
        const uint8_t* next = img_in[i + 1];
        uint8_t* out = img_out[i];
        for (int j = j0 + 1; j <= j1 - 1; j++) {
            const uint8_t c0 = prev[j - 1] & prev[j] & prev[j + 1];
            const uint8_t c1 = cur[j - 1] & cur[j] & cur[j + 1];
            const uint8_t c2 = next[j - 1] & next[j] & next[j + 1];
            out[j] = c0 & c1 & c2;
        }
    }
}

void morpho_compute_dilation3(const uint8_t** img_in, uint8_t** img_out, const int i0, const int i1, const int j0,
                              const int j1) {
    assert(img_in != NULL);
    assert(img_out != NULL);
    assert(img_in != (const uint8_t**)img_out);

    morpho_copy_borders(img_in, img_out, i0, i1, j0, j1);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = i0 + 1; i <= i1 - 1; i++) {
        const uint8_t* prev = img_in[i - 1];
        const uint8_t* cur = img_in[i + 0];
        const uint8_t* next = img_in[i + 1];
        uint8_t* out = img_out[i];
        for (int j = j0 + 1; j <= j1 - 1; j++) {
            const uint8_t c0 = prev[j - 1] | prev[j] | prev[j + 1];
            const uint8_t c1 = cur[j - 1] | cur[j] | cur[j + 1];
            const uint8_t c2 = next[j - 1] | next[j] | next[j + 1];
            out[j] = c0 | c1 | c2;
        }
    }
}

void morpho_compute_opening3(morpho_data_t* morpho_data, const uint8_t** img_in, uint8_t** img_out, const int i0,
                             const int i1, const int j0, const int j1) {
    assert(img_in != NULL);
    assert(img_out != NULL);
    morpho_compute_erosion3_separable(img_in, morpho_data->IB, img_out, i0, i1, j0, j1);
    morpho_compute_dilation3_separable((const uint8_t**)img_out, morpho_data->IB, img_out, i0, i1, j0, j1);
}

void morpho_compute_closing3(morpho_data_t* morpho_data, const uint8_t** img_in, uint8_t** img_out, const int i0,
                             const int i1, const int j0, const int j1) {
    assert(img_in != NULL);
    assert(img_out != NULL);
    morpho_compute_dilation3_separable(img_in, morpho_data->IB, img_out, i0, i1, j0, j1);
    morpho_compute_erosion3_separable((const uint8_t**)img_out, morpho_data->IB, img_out, i0, i1, j0, j1);
}
