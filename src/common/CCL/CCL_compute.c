#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <nrc2.h>

#include "motion/macros.h"
#include "motion/CCL/CCL_compute.h"

CCL_data_t* CCL_LSL_alloc_data(int i0, int i1, int j0, int j1) {
    CCL_data_t* CCL_data = (CCL_data_t*)malloc(sizeof(CCL_data_t));
    CCL_data->i0 = i0;
    CCL_data->i1 = i1;
    CCL_data->j0 = j0;
    CCL_data->j1 = j1;
    long n = (CCL_data->i1 - CCL_data->i0 + 1) * (CCL_data->j1 - CCL_data->j0 + 1);
    CCL_data->er = ui32matrix(CCL_data->i0, CCL_data->i1, CCL_data->j0, CCL_data->j1);
    //CCL_data->ea = ui32matrix(CCL_data->i0, CCL_data->i1, CCL_data->j0, CCL_data->j1);
    CCL_data->era = ui32matrix(CCL_data->i0, CCL_data->i1, CCL_data->j0, CCL_data->j1);
    CCL_data->rlc = ui32matrix(CCL_data->i0, CCL_data->i1, CCL_data->j0, CCL_data->j1);
    CCL_data->eq = ui32vector(0, n);
    CCL_data->ner = ui32vector(CCL_data->i0, CCL_data->i1);
    return CCL_data;
}

void CCL_LSL_init_data(CCL_data_t* CCL_data) {
    zero_ui32matrix(CCL_data->er , CCL_data->i0, CCL_data->i1, CCL_data->j0, CCL_data->j1);
    zero_ui32matrix(CCL_data->era , CCL_data->i0, CCL_data->i1, CCL_data->j0, CCL_data->j1);
    zero_ui32matrix(CCL_data->rlc , CCL_data->i0, CCL_data->i1, CCL_data->j0, CCL_data->j1);
    long n = (CCL_data->i1 - CCL_data->i0 + 1) * (CCL_data->j1 - CCL_data->j0 + 1);
    zero_ui32vector(CCL_data->eq, 0, n);
    zero_ui32vector(CCL_data->ner, CCL_data->i0, CCL_data->i1);
}

void CCL_LSL_free_data(CCL_data_t* CCL_data) {
    long n = (CCL_data->i1 - CCL_data->i0 + 1) * (CCL_data->j1 - CCL_data->j0 + 1);
    free_ui32matrix(CCL_data->er, CCL_data->i0, CCL_data->i1, CCL_data->j0, CCL_data->j1);
    //free_ui32matrix(CCL_data->ea, CCL_data->i0, CCL_data->i1, CCL_data->j0, CCL_data->j1);
    free_ui32matrix(CCL_data->era, CCL_data->i0, CCL_data->i1, CCL_data->j0, CCL_data->j1);
    free_ui32matrix(CCL_data->rlc, CCL_data->i0, CCL_data->i1, CCL_data->j0, CCL_data->j1);
    free_ui32vector(CCL_data->eq, 0, n);
    free_ui32vector(CCL_data->ner, CCL_data->i0, CCL_data->i1);
    free(CCL_data);
}

void _LSL_segment_detection(uint32_t* line_er, uint32_t* line_rlc, uint32_t* line_ner, const uint8_t* img_line,
                            const int j0, const int j1) {
    uint32_t j_curr;
    uint32_t j_prev = 0;
    uint32_t f = 0; // Front detection
    uint32_t b = 0;
    uint32_t er = 0;

    for (int j = j0; j <= j1; j++) {
        j_curr = (uint32_t)img_line[j];
        f = j_curr ^ j_prev;        // Xor: Front detection
        line_rlc[er] = j - (b & 1); // Begin/End of segment
        b ^= f;                     // Xor: End of segment correction
        er += (f & 1);              // Increment label if front detected
        line_er[j] = er;
        j_prev = j_curr; // Save one load
    }
    j_curr = 0;
    f = j_curr ^ j_prev;
    line_rlc[er] = j1 + 1 - (b & 1);
    er += (f & 1);
    *line_ner = er;
}

void _LSL_segment_detection_threshold(uint32_t* line_er, uint32_t* line_rlc, uint32_t* line_ner,
                                      const uint8_t* img_line, const int j0, const int j1, const uint8_t threshold) {
    uint32_t j_curr;
    uint32_t j_prev = 0;
    uint32_t f = 0; // Front detection
    uint32_t b = 0;
    uint32_t er = 0;

    for (int j = j0; j <= j1; j++) {
        j_curr = (uint32_t)img_line[j] >= threshold;
        f = j_curr ^ j_prev;        // Xor: Front detection
        line_rlc[er] = j - (b & 1); // Begin/End of segment
        b ^= f;                     // Xor: End of segment correction
        er += (f & 1);              // Increment label if front detected
        line_er[j] = er;
        j_prev = j_curr; // Save one load
    }
    j_curr = 0;
    f = j_curr ^ j_prev;
    line_rlc[er] = j1 + 1 - (b & 1);
    er += (f & 1);
    *line_ner = er;
}

void _LSL_equivalence_construction(uint32_t* CCL_data_eq, const uint32_t* line_rlc, uint32_t* line_era,
                                   const uint32_t* prevline_er, const uint32_t* prevline_era, const int n, const int x0,
                                   const int x1, uint32_t* nea) {
    int k, er, j0, j1, er0, er1, ea, erk, eak;
    for (k = 0; k < n; k += 2) {
        er = k + 1;

        j0 = line_rlc[k];  // Segment begin
        j1 = line_rlc[er]; // Segment end (k+1)

        // Extends for 8-connected
        if (j0 > x0)
            j0 -= 1;
        if (j1 < x1)
            j1 += 1;

        er0 = prevline_er[j0];
        er1 = prevline_er[j1];

        if ((er0 & 1) == 0) // er0 is even
            er0 += 1;
        if ((er1 & 1) == 0) // er1 is even
            er1 -= 1;

        if (er1 >= er0) { // Adjacency -> connect components
            ea = prevline_era[er0];
            uint32_t a = CCL_data_eq[ea];
            for (erk = er0 + 2; erk <= er1; erk += 2) {
                eak = prevline_era[erk];
                uint32_t ak = CCL_data_eq[eak];
                while (ak != CCL_data_eq[ak]) {
                    ak = CCL_data_eq[ak];
                }
                if (a < ak) {
                    CCL_data_eq[ak] = a; // Minimum propagation
                }

                if (a > ak) {
                    CCL_data_eq[a] = ak;
                    a = ak;
                    ea = eak;
                }
            }
            line_era[er] = a; // Global minimum
        } else {              // No adjacency -> new label
            line_era[er] = *nea;
            CCL_data_eq[*nea] = *nea;
            (*nea)++;
        }
    }
}

void _LSL_compute_final_image_labeling(const uint32_t** CCL_data_er, const  uint32_t** CCL_data_era,
                                       const uint32_t** CCL_data_rlc, const uint32_t* CCL_data_eq,
                                       const uint32_t* CCL_data_ner, uint32_t** labels, const int i0,
                                       const int i1) {
    // Step #5 - Final image labeling
    for (int i = i0; i <= i1; i++) {
        uint32_t n = CCL_data_ner[i];
        for (uint32_t k = 0; k < n; k += 2) {
            int a = CCL_data_rlc[i][k];
            int b = CCL_data_rlc[i][k + 1];

            // Step #3 merged with step #5
            uint32_t val = CCL_data_era[i][CCL_data_er[i][a]];
            val = CCL_data_eq[val] + 1;

            for (int j = a; j <= b; j++) {
                labels[i][j] = (uint32_t)val;
            }
        }
    }
}

static uint64_t sum_inclusive(const int a, const int b) {
    const uint64_t width = (uint64_t)(b - a + 1);
    return (uint64_t)(a + b) * width / 2;
}

static void _LSL_compute_final_image_labeling_RoIs(const uint32_t** CCL_data_er, const uint32_t** CCL_data_era,
                                                  const uint32_t** CCL_data_rlc, const uint32_t* CCL_data_eq,
                                                  const uint32_t* CCL_data_ner, uint32_t** labels,
                                                  const int i0, const int i1, const int j0, const int j1,
                                                  RoI_t* RoIs, uint64_t* RoIs_sum_x, uint64_t* RoIs_sum_y,
                                                  const size_t n_RoIs) {
    for (size_t i = 0; i < n_RoIs; i++) {
        RoIs[i].id = 0;
        RoIs[i].xmin = j1;
        RoIs[i].xmax = j0;
        RoIs[i].ymin = i1;
        RoIs[i].ymax = i0;
        RoIs[i].S = 0;
        RoIs[i].x = 0.f;
        RoIs[i].y = 0.f;
        RoIs[i].prev_id = 0;
        RoIs[i].next_id = 0;
        RoIs_sum_x[i] = 0;
        RoIs_sum_y[i] = 0;
    }

    for (int i = i0; i <= i1; i++) {
        const uint32_t n = CCL_data_ner[i];
        for (uint32_t k = 0; k < n; k += 2) {
            const int a = CCL_data_rlc[i][k];
            const int b = CCL_data_rlc[i][k + 1];
            const uint32_t label = CCL_data_eq[CCL_data_era[i][CCL_data_er[i][a]]] + 1;
            RoI_t* roi = &RoIs[label - 1];
            const uint32_t width = (uint32_t)(b - a + 1);

            if (labels) {
                uint32_t* line = labels[i];
                for (int j = a; j <= b; j++)
                    line[j] = label;
            }

            roi->id = label;
            roi->S += width;
            roi->xmin = MIN(roi->xmin, (uint32_t)a);
            roi->xmax = MAX(roi->xmax, (uint32_t)b);
            roi->ymin = MIN(roi->ymin, (uint32_t)i);
            roi->ymax = MAX(roi->ymax, (uint32_t)i);
            RoIs_sum_x[label - 1] += sum_inclusive(a, b);
            RoIs_sum_y[label - 1] += (uint64_t)i * width;
        }
    }

    for (size_t i = 0; i < n_RoIs; i++) {
        RoIs[i].x = (float)RoIs_sum_x[i] / (float)RoIs[i].S;
        RoIs[i].y = (float)RoIs_sum_y[i] / (float)RoIs[i].S;
    }
}

uint32_t __CCL_LSL_apply(uint32_t** CCL_data_er, uint32_t** CCL_data_era, uint32_t** CCL_data_rlc,
                         uint32_t* CCL_data_eq, uint32_t* CCL_data_ner, const uint8_t** img, const int i0, const int i1,
                         const int j0, const int j1) {
    // Step #2 - Equivalence construction
    uint32_t nea = i0;
    uint32_t n = CCL_data_ner[i0];
    for (uint32_t k = 0; k < n; k += 2) {
        CCL_data_eq[nea] = nea;
        CCL_data_era[i0][k + 1] = nea++;
    }
    for (int i = i0 + 1; i <= i1; i++) {
        _LSL_equivalence_construction(CCL_data_eq, CCL_data_rlc[i], CCL_data_era[i], CCL_data_er[i - 1],
                                      CCL_data_era[i - 1], CCL_data_ner[i], j0, j1, &nea);
    }

    // Step #3 - Relative to Absolute label conversion

    // Step #4 - Resolution of equivalence classes
    uint32_t trueN = 0;
    for (uint32_t i = 0; i < nea; i++) {
        if (i != CCL_data_eq[i]) {
            CCL_data_eq[i] = CCL_data_eq[CCL_data_eq[i]];
        } else {
            CCL_data_eq[i] = trueN++;
        }
    }

    return trueN;
}

uint32_t _CCL_LSL_apply(uint32_t** CCL_data_er, uint32_t** CCL_data_era, uint32_t** CCL_data_rlc, uint32_t* CCL_data_eq,
                        uint32_t* CCL_data_ner, const uint8_t** img, uint32_t** labels, const int i0, const int i1,
                        const int j0, const int j1, const uint8_t no_init_labels) {
    if (!no_init_labels) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int i = i0; i <= i1; i++)
            memset(labels[i], 0, sizeof(uint32_t) * ((j1 - j0) + 1));
    }

    // Step #1 - Segment detection
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = i0; i <= i1; i++)
        _LSL_segment_detection(CCL_data_er[i], CCL_data_rlc[i], &CCL_data_ner[i], img[i], j0, j1);

    uint32_t trueN = __CCL_LSL_apply(CCL_data_er, CCL_data_era, CCL_data_rlc, CCL_data_eq, CCL_data_ner, img, i0, i1,
                                     j0, j1);

    _LSL_compute_final_image_labeling((const uint32_t**)CCL_data_er, (const uint32_t**)CCL_data_era,
                                      (const uint32_t**)CCL_data_rlc, (const uint32_t*)CCL_data_eq,
                                      (const uint32_t*)CCL_data_ner, labels, i0, i1);
    return trueN;
}

uint32_t CCL_LSL_apply(CCL_data_t* CCL_data, const uint8_t** img, uint32_t** labels, const uint8_t no_init_labels) {
    return _CCL_LSL_apply(CCL_data->er, CCL_data->era, CCL_data->rlc, CCL_data->eq, CCL_data->ner, img, labels,
                          CCL_data->i0, CCL_data->i1, CCL_data->j0, CCL_data->j1, no_init_labels);
}

uint32_t CCL_LSL_apply_extract(CCL_data_t* CCL_data, const uint8_t** img, uint32_t** labels,
                               const uint8_t no_init_labels, RoI_t* RoIs, uint64_t* RoIs_sum_x,
                               uint64_t* RoIs_sum_y, const size_t max_RoIs) {
    assert(RoIs != NULL);
    assert(RoIs_sum_x != NULL);
    assert(RoIs_sum_y != NULL);

    if (labels != NULL && !no_init_labels) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int i = CCL_data->i0; i <= CCL_data->i1; i++)
            memset(labels[i], 0, sizeof(uint32_t) * ((CCL_data->j1 - CCL_data->j0) + 1));
    }

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = CCL_data->i0; i <= CCL_data->i1; i++)
        _LSL_segment_detection(CCL_data->er[i], CCL_data->rlc[i], &CCL_data->ner[i], img[i], CCL_data->j0,
                               CCL_data->j1);

    const uint32_t n_RoIs = __CCL_LSL_apply(CCL_data->er, CCL_data->era, CCL_data->rlc, CCL_data->eq, CCL_data->ner,
                                           img, CCL_data->i0, CCL_data->i1, CCL_data->j0, CCL_data->j1);
    assert(n_RoIs <= max_RoIs);

    _LSL_compute_final_image_labeling_RoIs((const uint32_t**)CCL_data->er, (const uint32_t**)CCL_data->era,
                                           (const uint32_t**)CCL_data->rlc, (const uint32_t*)CCL_data->eq,
                                           (const uint32_t*)CCL_data->ner, labels, CCL_data->i0, CCL_data->i1,
                                           CCL_data->j0, CCL_data->j1, RoIs, RoIs_sum_x, RoIs_sum_y, n_RoIs);
    return n_RoIs;
}
