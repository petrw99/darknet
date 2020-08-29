#ifndef LIST_H
#define LIST_H
#include "darknet.h"

#ifdef __cplusplus
extern "C" {
#endif
list *make_list();
int list_find(list *l, void *val);

void list_insert(list *, void *);


<<<<<<< HEAD
=======
void free_list_val(list *l);
void free_list(list *l);
>>>>>>> 05dee78fa3c41d92eb322d8d57fb065ddebc00b4
void free_list_contents(list *l);
void free_list_contents_kvp(list *l);

#ifdef __cplusplus
}
#endif
#endif
