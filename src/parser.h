#ifndef PARSER_H
#define PARSER_H
#include "darknet.h"
#include "network.h"

<<<<<<< HEAD
=======
#ifdef __cplusplus
extern "C" {
#endif
network parse_network_cfg(char *filename);
network parse_network_cfg_custom(char *filename, int batch, int time_steps);
>>>>>>> 05dee78fa3c41d92eb322d8d57fb065ddebc00b4
void save_network(network net, char *filename);
void save_weights_double(network net, char *filename);

#ifdef __cplusplus
}
#endif
#endif
