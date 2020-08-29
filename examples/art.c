<<<<<<< HEAD:examples/art.c
#include "darknet.h"
=======
#include "network.h"
#include "utils.h"
#include "parser.h"
#include "option_list.h"
#include "blas.h"
#include "classifier.h"
#ifdef WIN32
#include <time.h>
#include "gettimeofday.h"
#else
#include <sys/time.h>
#endif
>>>>>>> 05dee78fa3c41d92eb322d8d57fb065ddebc00b4:src/art.c

#include <sys/time.h>

void demo_art(char *cfgfile, char *weightfile, int cam_index)
{
#ifdef OPENCV
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);

    srand(2222222);
<<<<<<< HEAD:examples/art.c

    void * cap = open_video_stream(0, cam_index, 0,0,0);

    char *window = "ArtJudgementBot9000!!!";
    if(!cap) error("Couldn't connect to webcam.\n");
=======
    cap_cv * cap;

    cap = get_capture_webcam(cam_index);

    char *window = "ArtJudgementBot9000!!!";
    if(!cap) error("Couldn't connect to webcam.\n");
    create_window_cv(window, 0, 512, 512);
>>>>>>> 05dee78fa3c41d92eb322d8d57fb065ddebc00b4:src/art.c
    int i;
    int idx[] = {37, 401, 434};
    int n = sizeof(idx)/sizeof(idx[0]);

    while(1){
<<<<<<< HEAD:examples/art.c
        image in = get_image_from_stream(cap);
        image in_s = resize_image(in, net->w, net->h);
=======
        image in = get_image_from_stream_cpp(cap);
        image in_s = resize_image(in, net.w, net.h);
        show_image(in, window);
>>>>>>> 05dee78fa3c41d92eb322d8d57fb065ddebc00b4:src/art.c

        float *p = network_predict(net, in_s.data);

        printf("\033[2J");
        printf("\033[1;1H");

        float score = 0;
        for(i = 0; i < n; ++i){
            float s = p[idx[i]];
            if (s > score) score = s;
        }
        score = score;
        printf("I APPRECIATE THIS ARTWORK: %10.7f%%\n", score*100);
        printf("[");
	int upper = 30;
        for(i = 0; i < upper; ++i){
            printf("%c", ((i+.5) < score*upper) ? 219 : ' ');
        }
        printf("]\n");

        show_image(in, window, 1);
        free_image(in_s);
        free_image(in);
<<<<<<< HEAD:examples/art.c
=======

        wait_key_cv(1);
>>>>>>> 05dee78fa3c41d92eb322d8d57fb065ddebc00b4:src/art.c
    }
#endif
}


void run_art(int argc, char **argv)
{
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    char *cfg = argv[2];
    char *weights = argv[3];
    demo_art(cfg, weights, cam_index);
}
