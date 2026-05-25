#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <nrc2.h>

#include "vec.h"

#include "motion/args.h"
#include "motion/tools.h"
#include "motion/macros.h"

#include "motion/CCL.h"
#include "motion/features.h"
#include "motion/kNN.h"
#include "motion/tracking.h"
#include "motion/video.h"
#include "motion/image.h"
#include "motion/sigma_delta.h"
#include "motion/morpho.h"
#include "motion/visu.h"

typedef struct {
    double dec_us;
    double sd_us;
    double morpho_us;
    double ccl_us;
    double filter_us;
    double knn_us;
    double tracking_us;
    double log_us;
    double visu_us;
} stage_timings_t;

typedef struct {
    int i0;
    int i1;
    int j0;
    int j1;
    size_t max_candidate_RoIs;
    size_t max_filtered_RoIs;
    sigma_delta_data_t* sigma_delta;
    morpho_data_t* morphology;
    CCL_data_t* ccl;
    RoI_t* candidate_RoIs;
    uint64_t* candidate_sum_x;
    uint64_t* candidate_sum_y;
    uint8_t** mask;
    uint32_t** labels;
    uint32_t** filtered_labels;
} detection_data_t;

static detection_data_t detection_alloc(const int i0, const int i1, const int j0, const int j1,
                                        const size_t max_candidate_RoIs, const size_t max_filtered_RoIs,
                                        const int keep_filtered_labels) {
    detection_data_t data;
    data.i0 = i0;
    data.i1 = i1;
    data.j0 = j0;
    data.j1 = j1;
    data.max_candidate_RoIs = max_candidate_RoIs;
    data.max_filtered_RoIs = max_filtered_RoIs;
    data.sigma_delta = sigma_delta_alloc_data(i0, i1, j0, j1, 1, 254);
    data.morphology = morpho_alloc_data(i0, i1, j0, j1);
    data.ccl = CCL_LSL_alloc_data(i0, i1, j0, j1);
    data.candidate_RoIs = features_alloc_RoIs(max_candidate_RoIs);
    data.candidate_sum_x = (uint64_t*)malloc(max_candidate_RoIs * sizeof(uint64_t));
    data.candidate_sum_y = (uint64_t*)malloc(max_candidate_RoIs * sizeof(uint64_t));
    assert(data.candidate_sum_x != NULL);
    assert(data.candidate_sum_y != NULL);
    data.mask = ui8matrix(i0, i1, j0, j1);
    data.labels = keep_filtered_labels ? ui32matrix(i0, i1, j0, j1) : NULL;
    data.filtered_labels = keep_filtered_labels ? ui32matrix(i0, i1, j0, j1) : NULL;
    return data;
}

static void detection_init(detection_data_t* data, const uint8_t** first_frame) {
    sigma_delta_init_data(data->sigma_delta, first_frame, data->i0, data->i1, data->j0, data->j1);
    morpho_init_data(data->morphology);
    CCL_LSL_init_data(data->ccl);
    features_init_RoIs(data->candidate_RoIs, data->max_candidate_RoIs);
    zero_ui8matrix(data->mask, data->i0, data->i1, data->j0, data->j1);
    if (data->labels)
        zero_ui32matrix(data->labels, data->i0, data->i1, data->j0, data->j1);
    if (data->filtered_labels)
        zero_ui32matrix(data->filtered_labels, data->i0, data->i1, data->j0, data->j1);
}

static uint32_t detection_process(detection_data_t* data, const uint8_t** gray, RoI_t* RoIs,
                                  const uint8_t sigma_delta_n, const uint32_t min_surface,
                                  const uint32_t max_surface, stage_timings_t* timings) {
    TIME_POINT(sd_b);
    sigma_delta_compute(data->sigma_delta, gray, data->mask, data->i0, data->i1, data->j0, data->j1, sigma_delta_n);
    TIME_POINT(sd_e);
    timings->sd_us += TIME_ELAPSED2_US(sd_b, sd_e);

    TIME_POINT(morpho_b);
    morpho_compute_opening3(data->morphology, (const uint8_t**)data->mask, data->mask,
                            data->i0, data->i1, data->j0, data->j1);
    morpho_compute_closing3(data->morphology, (const uint8_t**)data->mask, data->mask,
                            data->i0, data->i1, data->j0, data->j1);
    TIME_POINT(morpho_e);
    timings->morpho_us += TIME_ELAPSED2_US(morpho_b, morpho_e);

    TIME_POINT(ccl_b);
    const uint32_t n_candidate_RoIs = CCL_LSL_apply_extract(data->ccl, (const uint8_t**)data->mask, data->labels, 0,
                                                            data->candidate_RoIs, data->candidate_sum_x,
                                                            data->candidate_sum_y, data->max_candidate_RoIs);
    TIME_POINT(ccl_e);
    timings->ccl_us += TIME_ELAPSED2_US(ccl_b, ccl_e);

    TIME_POINT(filter_b);
    const uint32_t n_RoIs = features_filter_surface((const uint32_t**)data->labels, data->filtered_labels,
                                                    data->i0, data->i1, data->j0, data->j1,
                                                    data->candidate_RoIs, n_candidate_RoIs, min_surface, max_surface);
    assert(n_RoIs <= data->max_filtered_RoIs);
    features_shrink_basic(data->candidate_RoIs, n_candidate_RoIs, RoIs);
    for (uint32_t i = 0; i < n_RoIs; i++) {
        RoIs[i].prev_id = 0;
        RoIs[i].next_id = 0;
    }
    TIME_POINT(filter_e);
    timings->filter_us += TIME_ELAPSED2_US(filter_b, filter_e);

    return n_RoIs;
}

static void detection_free(detection_data_t* data) {
    sigma_delta_free_data(data->sigma_delta);
    morpho_free_data(data->morphology);
    CCL_LSL_free_data(data->ccl);
    features_free_RoIs(data->candidate_RoIs);
    free(data->candidate_sum_x);
    free(data->candidate_sum_y);
    free_ui8matrix(data->mask, data->i0, data->i1, data->j0, data->j1);
    if (data->labels)
        free_ui32matrix(data->labels, data->i0, data->i1, data->j0, data->j1);
    if (data->filtered_labels)
        free_ui32matrix(data->filtered_labels, data->i0, data->i1, data->j0, data->j1);
}

static double avg_ms(const double elapsed_us, const size_t n_frames) {
    return n_frames ? elapsed_us * 1e-3 / (double)n_frames : 0.;
}

int main(int argc, char** argv) {

    // ---------------------------------- //
    // -- DEFAULT VALUES OF PARAMETERS -- //
    // ---------------------------------- //

    char* def_p_vid_in_path = NULL;
    int def_p_vid_in_start = 0;
    int def_p_vid_in_stop = 0;
    int def_p_vid_in_skip = 0;
    int def_p_vid_in_loop = 1;
    int def_p_vid_in_threads = 0;
    char def_p_vid_in_dec_hw[16] = "NONE";
    int def_p_sd_n = 2;
    char* def_p_ccl_fra_path = NULL;
    int def_p_flt_s_min = 50;
    int def_p_flt_s_max = 100000;
    int def_p_knn_k = 3;
    int def_p_knn_d = 10;
    float def_p_knn_s = 0.125f;
    int def_p_trk_ext_d = 5;
    int def_p_trk_ext_o = 3;
    int def_p_trk_obj_min = 2;
    char* def_p_trk_roi_path = NULL;
    char* def_p_log_path = NULL;
    int def_p_cca_roi_max1 = 65536; // Maximum number of RoIs
    int def_p_cca_roi_max2 = 8192; // Maximum number of RoIs after filtering
    char* def_p_vid_out_path = NULL;

    // ------------------------ //
    // -- CMD LINE ARGS HELP -- //
    // ------------------------ //

    if (args_find(argc, argv, "--help,-h")) {
        fprintf(stderr,
                "  --vid-in-path     Path to video file or to an images sequence                            [%s]\n",
                def_p_vid_in_path ? def_p_vid_in_path : "NULL");
        fprintf(stderr,
                "  --vid-in-start    Start frame id (included) in the video                                 [%d]\n",
                def_p_vid_in_start);
        fprintf(stderr,
                "  --vid-in-stop     Stop frame id (included) in the video (if set to 0, read entire video) [%d]\n",
                def_p_vid_in_stop);
        fprintf(stderr,
                "  --vid-in-skip     Number of frames to skip                                               [%d]\n",
                def_p_vid_in_skip);
        fprintf(stderr,
                "  --vid-in-buff     Bufferize all the video in global memory before executing the chain        \n");
        fprintf(stderr,
                "  --vid-in-loop     Number of times the video is read in loop                              [%d]\n",
                def_p_vid_in_loop);
        fprintf(stderr,
                "  --vid-in-threads  Select the number of threads to use to decode video input (in ffmpeg)  [%d]\n",
                def_p_vid_in_threads);
        fprintf(stderr,
                "  --vid-in-dec-hw   Select video decoder hardware acceleration ('NONE', 'NVDEC', 'VIDTB')  [%s]\n",
                def_p_vid_in_dec_hw);
        fprintf(stderr,
                "  --sd-n            Value of the N parameter in the Sigma-Delta algorithm                  [%d]\n",
                def_p_sd_n);
        fprintf(stderr,
                "  --ccl-fra-path    Path of the files for CC debug frames                                  [%s]\n",
                def_p_ccl_fra_path ? def_p_ccl_fra_path : "NULL");
#ifdef MOTION_OPENCV_LINK
        fprintf(stderr,
                "  --ccl-fra-id      Show the RoI/CC ids on the ouptut CC frames                                \n");
#endif
        fprintf(stderr,
                "  --cca-roi-max1    Maximum number of RoIs after CCA                                       [%d]\n",
                def_p_cca_roi_max1);
        fprintf(stderr,
                "  --cca-roi-max2    Maximum number of RoIs after surface filtering                         [%d]\n",
                def_p_cca_roi_max2);
        fprintf(stderr,
                "  --flt-s-min       Minimum surface of the CCs in pixels                                   [%d]\n",
                def_p_flt_s_min);
        fprintf(stderr,
                "  --flt-s-max       Maxumum surface of the CCs in pixels                                   [%d]\n",
                def_p_flt_s_max);
        fprintf(stderr,
                "  --knn-k           Maximum number of neighbors considered in k-NN algorithm               [%d]\n",
                def_p_knn_k);
        fprintf(stderr,
                "  --knn-d           Maximum distance in pixels between two images (in k-NN)                [%d]\n",
                def_p_knn_d);
        fprintf(stderr,
                "  --knn-s           Minimum surface ratio to match two CCs in k-NN                         [%f]\n",
                def_p_knn_s);
        fprintf(stderr,
                "  --trk-ext-d       Search radius in pixels for CC extrapolation (piece-wise tracking)     [%d]\n",
                def_p_trk_ext_d);
        fprintf(stderr,
                "  --trk-ext-o       Maximum number of frames to extrapolate (linear) for lost objects      [%d]\n",
                def_p_trk_ext_o);
        fprintf(stderr,
                "  --trk-obj-min     Minimum number of frames required to track an object                   [%d]\n",
                def_p_trk_obj_min);
        fprintf(stderr,
                "  --trk-roi-path    Path to the file containing the RoI ids for each track                 [%s]\n",
                def_p_trk_roi_path ? def_p_trk_roi_path : "NULL");
        fprintf(stderr,
                "  --log-path        Path of the output statistics, only required for debugging purpose     [%s]\n",
                def_p_log_path ? def_p_log_path : "NULL");
        fprintf(stderr,
                "  --vid-out-path    Path to video file or to an images sequence to write the output        [%s]\n",
                def_p_vid_out_path ? def_p_vid_out_path : "NULL");
        fprintf(stderr,
                "  --vid-out-play    Show the output video in a SDL window                                      \n");
#ifdef MOTION_OPENCV_LINK
        fprintf(stderr,
                "  --vid-out-id      Draw the track ids on the ouptut video                                     \n");
#endif
        fprintf(stderr,
                "  --stats           Show the average latency of each task                                      \n");
        fprintf(stderr,
                "  --help, -h        This help                                                                  \n");
        exit(1);
    }

    // ------------------------- //
    // -- PARSE CMD LINE ARGS -- //
    // ------------------------- //

    const char* p_vid_in_path = args_find_char(argc, argv, "--vid-in-path", def_p_vid_in_path);
    const int p_vid_in_start = args_find_int_min(argc, argv, "--vid-in-start", def_p_vid_in_start, 0);
    const int p_vid_in_stop = args_find_int_min(argc, argv, "--vid-in-stop", def_p_vid_in_stop, 0);
    const int p_vid_in_skip = args_find_int_min(argc, argv, "--vid-in-skip", def_p_vid_in_skip, 0);
    const int p_vid_in_buff = args_find(argc, argv, "--vid-in-buff");
    const int p_vid_in_loop = args_find_int_min(argc, argv, "--vid-in-loop", def_p_vid_in_loop, 1);
    const int p_vid_in_threads = args_find_int_min(argc, argv, "--vid-in-threads", def_p_vid_in_threads, 0);
    const char* p_vid_in_dec_hw = args_find_char(argc, argv, "--vid-in-dec-hw", def_p_vid_in_dec_hw);
    const int p_sd_n = args_find_int_min(argc, argv, "--sd-n", def_p_sd_n, 0);
    const char* p_ccl_fra_path = args_find_char(argc, argv, "--ccl-fra-path", def_p_ccl_fra_path);
#ifdef MOTION_OPENCV_LINK
    const int p_ccl_fra_id = args_find(argc, argv, "--ccl-fra-id,--show-id");
#else
    const int p_ccl_fra_id = 0;
#endif
    const int p_cca_roi_max1 = args_find_int_min(argc, argv, "--cca-roi-max1", def_p_cca_roi_max1, 0);
    const int p_cca_roi_max2 = args_find_int_min(argc, argv, "--cca-roi-max2", def_p_cca_roi_max2, 0);
    const int p_flt_s_min = args_find_int_min(argc, argv, "--flt-s-min", def_p_flt_s_min, 0);
    const int p_flt_s_max = args_find_int_min(argc, argv, "--flt-s-max", def_p_flt_s_max, 0);
    const int p_knn_k = args_find_int_min(argc, argv, "--knn-k", def_p_knn_k, 0);
    const int p_knn_d = args_find_int_min(argc, argv, "--knn-d", def_p_knn_d, 0);
    const float p_knn_s = args_find_float_min_max(argc, argv, "--knn-s", def_p_knn_s, 0.f, 1.f);
    const int p_trk_ext_d = args_find_int_min(argc, argv, "--trk-ext-d", def_p_trk_ext_d, 0);
    const int p_trk_ext_o = args_find_int_min_max(argc, argv, "--trk-ext-o", def_p_trk_ext_o, 0, 255);
    const int p_trk_obj_min = args_find_int_min(argc, argv, "--trk-obj-min", def_p_trk_obj_min, 2);
    const char* p_trk_roi_path = args_find_char(argc, argv, "--trk-roi-path", def_p_trk_roi_path);
    const char* p_log_path = args_find_char(argc, argv, "--log-path", def_p_log_path);
    const char* p_vid_out_path = args_find_char(argc, argv, "--vid-out-path", def_p_vid_out_path);
    const int p_vid_out_play = args_find(argc, argv, "--vid-out-play");
#ifdef MOTION_OPENCV_LINK
    const int p_vid_out_id = args_find(argc, argv, "--vid-out-id");
#else
    const int p_vid_out_id = 0;
#endif
    const int p_stats = args_find(argc, argv, "--stats");

    // --------------------- //
    // -- HEADING DISPLAY -- //
    // --------------------- //

    printf("#  ---------- \n");
    printf("# |  MOTION  |\n");
    printf("#  ---------- \n");
    printf("#\n");
    printf("# Parameters:\n");
    printf("# -----------\n");
    printf("#  * vid-in-path    = %s\n", p_vid_in_path);
    printf("#  * vid-in-start   = %d\n", p_vid_in_start);
    printf("#  * vid-in-stop    = %d\n", p_vid_in_stop);
    printf("#  * vid-in-skip    = %d\n", p_vid_in_skip);
    printf("#  * vid-in-buff    = %d\n", p_vid_in_buff);
    printf("#  * vid-in-loop    = %d\n", p_vid_in_loop);
    printf("#  * vid-in-threads = %d\n", p_vid_in_threads);
    printf("#  * vid-in-dec-hw  = %s\n", p_vid_in_dec_hw);
    printf("#  * sd-n           = %d\n", p_sd_n);
    printf("#  * ccl-fra-path   = %s\n", p_ccl_fra_path);
#ifdef MOTION_OPENCV_LINK
    printf("#  * ccl-fra-id     = %d\n", p_ccl_fra_id);
#endif
    printf("#  * cca-roi-max1   = %d\n", p_cca_roi_max1);
    printf("#  * cca-roi-max2   = %d\n", p_cca_roi_max2);
    printf("#  * flt-s-min      = %d\n", p_flt_s_min);
    printf("#  * flt-s-max      = %d\n", p_flt_s_max);
    printf("#  * knn-k          = %d\n", p_knn_k);
    printf("#  * knn-d          = %d\n", p_knn_d);
    printf("#  * knn-s          = %1.3f\n", p_knn_s);
    printf("#  * trk-ext-d      = %d\n", p_trk_ext_d);
    printf("#  * trk-ext-o      = %d\n", p_trk_ext_o);
    printf("#  * trk-obj-min    = %d\n", p_trk_obj_min);
    printf("#  * trk-roi-path   = %s\n", p_trk_roi_path);
    printf("#  * log-path       = %s\n", p_log_path);
    printf("#  * vid-out-path   = %s\n", p_vid_out_path);
    printf("#  * vid-out-play   = %d\n", p_vid_out_play);
#ifdef MOTION_OPENCV_LINK
    printf("#  * vid-out-id     = %d\n", p_vid_out_id);
#endif
    printf("#  * stats          = %d\n", p_stats);

    printf("#\n");

    // -------------------------- //
    // -- CMD LINE ARGS CHECKS -- //
    // -------------------------- //

    if (!p_vid_in_path) {
        fprintf(stderr, "(EE) '--vid-in-path' is missing\n");
        exit(1);
    }
    if (p_vid_in_stop && p_vid_in_stop < p_vid_in_start) {
        fprintf(stderr, "(EE) '--vid-in-stop' has to be higher than '--vid-in-start'\n");
        exit(1);
    }
#ifdef MOTION_OPENCV_LINK
    if (p_ccl_fra_id && !p_ccl_fra_path)
        fprintf(stderr, "(WW) '--ccl-fra-id' has to be combined with the '--ccl-fra-path' parameter\n");
#endif
    if (p_vid_out_path && p_vid_out_play)
        fprintf(stderr, "(WW) '--vid-out-path' will be ignore because '--vid-out-play' is set\n");
#ifdef MOTION_OPENCV_LINK
    if (p_vid_out_id && !p_vid_out_path && !p_vid_out_play)
        fprintf(stderr,
                "(WW) '--vid-out-id' will be ignore because neither '--vid-out-play' nor 'p_vid_out_path' are set\n");
#endif

    // --------------------------------------- //
    // -- VIDEO ALLOCATION & INITIALISATION -- //
    // --------------------------------------- //

    TIME_POINT(start_alloc_init);
    int i0, i1, j0, j1; // image dimension (i0 = y_min, i1 = y_max, j0 = x_min, j1 = x_max)
    video_reader_t* video = video_reader_alloc_init(p_vid_in_path, p_vid_in_start, p_vid_in_stop, p_vid_in_skip,
                                                    p_vid_in_buff, p_vid_in_threads, VCDC_FFMPEG_IO,
                                                    video_hwaccel_str_to_enum(p_vid_in_dec_hw), &i0, &i1, &j0, &j1);
    video->loop_size = (size_t)(p_vid_in_loop);
    video_writer_t* video_writer = NULL;
    img_data_t* img_data = NULL;
    if (p_ccl_fra_path) {
        img_data = image_gs_alloc((i1 - i0) + 1, (j1 - j0) + 1);
        const size_t n_threads = 1;
        video_writer = video_writer_alloc_init(p_ccl_fra_path, p_vid_in_start, n_threads, (i1 - i0) + 1, (j1 - j0) + 1,
                                               PIXFMT_GRAY, VCDC_FFMPEG_IO, 0);
    }
    visu_data_t *visu_data = NULL;
    if (p_vid_out_play || p_vid_out_path) {
        const uint8_t n_threads = 1;
        visu_data = visu_alloc_init(p_vid_out_path, p_vid_in_start, n_threads, (i1 - i0) + 1, (j1 - j0) + 1,
                                    PIXFMT_RGB24, VCDC_FFMPEG_IO, p_vid_out_id, p_vid_out_play, p_trk_obj_min,
                                    p_cca_roi_max2, p_vid_in_skip);
    }

    detection_data_t detection = detection_alloc(i0, i1, j0, j1, p_cca_roi_max1, p_cca_roi_max2,
                                                 p_ccl_fra_path != NULL);
    RoI_t* previous_RoIs = features_alloc_RoIs(p_cca_roi_max2);
    RoI_t* current_RoIs = features_alloc_RoIs(p_cca_roi_max2);
    kNN_data_t* knn_data = kNN_alloc_data(p_cca_roi_max2);
    tracking_data_t* tracking_data = tracking_alloc_data(MAX(p_trk_obj_min, p_trk_ext_o) + 1, p_cca_roi_max2);
    uint8_t** frame = ui8matrix(i0, i1, j0, j1);

    int cur_fra;
    if ((cur_fra = video_reader_get_frame(video, frame)) != -1) {
        detection_init(&detection, (const uint8_t**)frame);
    } else {
        fprintf(stderr, "(EE) Something is not working well with the input video.\n");
        exit(1);
    }
    features_init_RoIs(previous_RoIs, p_cca_roi_max2);
    features_init_RoIs(current_RoIs, p_cca_roi_max2);
    kNN_init_data(knn_data);
    tracking_init_data(tracking_data);
    if (visu_data)
        visu_display(visu_data, (const uint8_t**)frame, current_RoIs, 0, tracking_data->tracks, cur_fra);

    TIME_POINT(stop_alloc_init);
    printf("# Allocations and initialisations took %6.3f sec\n", TIME_ELAPSED2_SEC(start_alloc_init, stop_alloc_init));

    // --------------------- //
    // -- PROCESSING LOOP -- //
    // --------------------- //

    printf("# The program is running...\n");
    size_t n_moving_objs = 0;
    size_t n_processed_frames = 0;
    uint32_t n_previous_RoIs = 0;
    stage_timings_t timings = {0};
    TIME_POINT(start_compute);
    while (1) {
        TIME_POINT(dec_b);
        cur_fra = video_reader_get_frame(video, frame);
        TIME_POINT(dec_e);
        timings.dec_us += TIME_ELAPSED2_US(dec_b, dec_e);
        if (cur_fra == -1)
            break;

        fprintf(stderr, "(II) Frame n°%4d", cur_fra);

        const uint32_t n_current_RoIs = detection_process(&detection, (const uint8_t**)frame, current_RoIs,
                                                          (uint8_t)p_sd_n, p_flt_s_min, p_flt_s_max, &timings);

        TIME_POINT(knn_b);
        kNN_match(knn_data, previous_RoIs, n_previous_RoIs, current_RoIs, n_current_RoIs,
                  p_knn_k, p_knn_d, p_knn_s);
        TIME_POINT(knn_e);
        timings.knn_us += TIME_ELAPSED2_US(knn_b, knn_e);

        TIME_POINT(trk_b);
        tracking_perform(tracking_data, current_RoIs, n_current_RoIs, cur_fra, p_trk_ext_d, p_trk_obj_min,
                         p_trk_roi_path != NULL || visu_data, p_trk_ext_o, p_knn_s);
        TIME_POINT(trk_e);
        timings.tracking_us += TIME_ELAPSED2_US(trk_b, trk_e);

        TIME_POINT(log_b);
        if (img_data) {
            image_gs_draw_labels(img_data, (const uint32_t**)detection.filtered_labels,
                                 current_RoIs, n_current_RoIs, p_ccl_fra_id);
            video_writer_save_frame(video_writer, (const uint8_t**)image_gs_get_pixels_2d(img_data));
        }
        if (p_log_path) {
            tools_create_folder(p_log_path);
            char filename[1024];
            snprintf(filename, sizeof(filename), "%s/%05d.txt", p_log_path, cur_fra);
            FILE* f = fopen(filename, "w");
            if (f == NULL) {
                fprintf(stderr, "(EE) error while opening '%s'\n", filename);
                exit(1);
            }
            const int prev_fra = cur_fra > p_vid_in_start ? cur_fra - (p_vid_in_skip + 1) : -1;
            features_RoIs0_RoIs1_write(f, prev_fra, cur_fra, previous_RoIs, n_previous_RoIs,
                                        current_RoIs, n_current_RoIs, tracking_data->tracks);
            if (cur_fra > p_vid_in_start) {
                fprintf(f, "#\n");
                kNN_asso_conflicts_write(f, knn_data, previous_RoIs, n_previous_RoIs,
                                         current_RoIs, n_current_RoIs);
                fprintf(f, "#\n");
                tracking_tracks_write_full(f, tracking_data->tracks);
            }
            fclose(f);
        }
        TIME_POINT(log_e);
        timings.log_us += TIME_ELAPSED2_US(log_b, log_e);

        TIME_POINT(vis_b);
        if (visu_data)
            visu_display(visu_data, (const uint8_t**)frame, current_RoIs, n_current_RoIs,
                         tracking_data->tracks, cur_fra);
        TIME_POINT(vis_e);
        timings.visu_us += TIME_ELAPSED2_US(vis_b, vis_e);

        RoI_t* tmp_RoIs = previous_RoIs;
        previous_RoIs = current_RoIs;
        current_RoIs = tmp_RoIs;
        n_previous_RoIs = n_current_RoIs;

        n_processed_frames++;
        n_moving_objs = tracking_count_objects(tracking_data->tracks);

        TIME_POINT(stop_compute);
        fprintf(stderr, " -- Time = %6.3f sec", TIME_ELAPSED2_SEC(start_compute, stop_compute));
        fprintf(stderr, " -- FPS = %4d", (int)(n_processed_frames / (TIME_ELAPSED2_SEC(start_compute, stop_compute))));
        fprintf(stderr, " -- Tracks = %3lu\r", (unsigned long)n_moving_objs);
        fflush(stderr);
    }
    TIME_POINT(stop_compute);
    fprintf(stderr, "\n");

    if (p_trk_roi_path) {
        FILE* f = fopen(p_trk_roi_path, "w");
        if (f == NULL) {
            fprintf(stderr, "(EE) error while opening '%s'\n", p_trk_roi_path);
            exit(1);
        }
        tracking_tracks_RoIs_id_write(f, tracking_data->tracks);
        fclose(f);
    }
    tracking_tracks_write(stdout, tracking_data->tracks);

    printf("# Tracks statistics:\n");
    printf("# -> Processed frames = %4u\n", (unsigned)n_processed_frames);
    printf("# -> Detected tracks  = %4lu\n", (unsigned long)n_moving_objs);
    printf("# -> Took %6.3f seconds (avg %d FPS)\n", TIME_ELAPSED2_SEC(start_compute, stop_compute),
           (int)(n_processed_frames / (TIME_ELAPSED2_SEC(start_compute, stop_compute))));
    if (p_stats) {
        printf("#\n");
        printf("# Average latencies: \n");
        printf("# -> Video decoding = %8.3f ms\n", avg_ms(timings.dec_us, n_processed_frames));
        printf("# -> Sigma-Delta    = %8.3f ms\n", avg_ms(timings.sd_us, n_processed_frames));
        printf("# -> Morphology     = %8.3f ms\n", avg_ms(timings.morpho_us, n_processed_frames));
        printf("# -> CCL + RoIs     = %8.3f ms\n", avg_ms(timings.ccl_us, n_processed_frames));
        printf("# -> Filtering      = %8.3f ms\n", avg_ms(timings.filter_us, n_processed_frames));
        printf("# -> k-NN           = %8.3f ms\n", avg_ms(timings.knn_us, n_processed_frames));
        printf("# -> Tracking       = %8.3f ms\n", avg_ms(timings.tracking_us, n_processed_frames));
        printf("# -> *Logs*         = %8.3f ms\n", avg_ms(timings.log_us, n_processed_frames));
        printf("# -> *Visu*         = %8.3f ms\n", avg_ms(timings.visu_us, n_processed_frames));
        const double total = avg_ms(timings.dec_us + timings.sd_us + timings.morpho_us + timings.ccl_us +
                                    timings.filter_us + timings.knn_us + timings.tracking_us + timings.log_us +
                                    timings.visu_us, n_processed_frames);
        printf("# => Total          = %8.3f ms [~%5.2f FPS]\n", total, 1000. / total);
    }

    // some frames have been buffered for the visualization, display or write these frames here
    if (visu_data)
        visu_flush(visu_data, tracking_data->tracks);

    detection_free(&detection);
    free_ui8matrix(frame, i0, i1, j0, j1);
    features_free_RoIs(previous_RoIs);
    features_free_RoIs(current_RoIs);
    video_reader_free(video);
    if (img_data) {
        image_gs_free(img_data);
        video_writer_free(video_writer);
    }
    if (visu_data)
        visu_free(visu_data);
    kNN_free_data(knn_data);
    tracking_free_data(tracking_data);

    printf("#\n");
    printf("# End of the program, exiting.\n");

    return EXIT_SUCCESS;
}
