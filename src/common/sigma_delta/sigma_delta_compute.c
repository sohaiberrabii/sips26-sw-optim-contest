#include <stdlib.h>
#include <nrc2.h>

#include "motion/macros.h"
#include "motion/sigma_delta/sigma_delta_compute.h"

sigma_delta_data_t* sigma_delta_alloc_data(const int i0, const int i1, const int j0, const int j1, const uint8_t vmin,
                                           const uint8_t vmax) {
    sigma_delta_data_t* sd_data = (sigma_delta_data_t*)malloc(sizeof(sigma_delta_data_t));
    sd_data->i0 = i0;
    sd_data->i1 = i1;
    sd_data->j0 = j0;
    sd_data->j1 = j1;
    sd_data->vmin = vmin;
    sd_data->vmax = vmax;
    sd_data->M = ui8matrix(sd_data->i0, sd_data->i1, sd_data->j0, sd_data->j1);
    sd_data->O = ui8matrix(sd_data->i0, sd_data->i1, sd_data->j0, sd_data->j1);
    sd_data->V = ui8matrix(sd_data->i0, sd_data->i1, sd_data->j0, sd_data->j1);
    return sd_data;
}

void sigma_delta_init_data(sigma_delta_data_t* sd_data, const uint8_t** img_in, const int i0, const int i1,
                           const int j0, const int j1) {
    for (int i = i0; i <= i1; i++) {
        for (int j = j0; j <= j1; j++) {
            sd_data->M[i][j] = img_in != NULL ? img_in[i][j] : sd_data->vmax;
            sd_data->V[i][j] = sd_data->vmin;
        }
    }
}

void sigma_delta_free_data(sigma_delta_data_t* sd_data) {
    free_ui8matrix(sd_data->M, sd_data->i0, sd_data->i1, sd_data->j0, sd_data->j1);
    free_ui8matrix(sd_data->O, sd_data->i0, sd_data->i1, sd_data->j0, sd_data->j1);
    free_ui8matrix(sd_data->V, sd_data->i0, sd_data->i1, sd_data->j0, sd_data->j1);
    free(sd_data);
}

void sigma_delta_compute(sigma_delta_data_t *sd_data, const uint8_t** img_in, uint8_t** img_out, const int i0,
                         const int i1, const int j0, const int j1, const uint8_t N) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = i0; i <= i1; i++) {
        const uint8_t* in = img_in[i];
        uint8_t* out = img_out[i];
        uint8_t* M = sd_data->M[i];
        uint8_t* O = sd_data->O[i];
        uint8_t* V = sd_data->V[i];
        for (int j = j0; j <= j1; j++) {
            const uint8_t pixel = in[j];
            uint8_t mean = M[j];
            mean += mean < pixel;
            mean -= mean > pixel;
            M[j] = mean;

            const uint8_t delta = mean > pixel ? mean - pixel : pixel - mean;
            O[j] = delta;

            const uint16_t target = (uint16_t)N * delta;
            uint8_t variance = V[j];
            variance += variance < target;
            variance -= variance > target;
            variance = MAX(MIN(variance, sd_data->vmax), sd_data->vmin);
            V[j] = variance;

            out[j] = delta < variance ? 0 : 255;
        }
    }
}
