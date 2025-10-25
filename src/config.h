#ifndef CONFIG_H
#define CONFIG_H

#include "utils.h"

/* Activation-related tunables (defaults in config.c) */
extern mat_t ACT_PARAM_MIN;   /* min allowed activation param value */
extern mat_t ACT_PARAM_MAX;   /* max allowed activation param value */
extern mat_t ACT_Z_CLIP_B;    /* clip pre-activation z to [-B, B] in forward */

/* Default clipping for activation gradients (L2 norm) */
extern mat_t ACT_GRAD_CLIP_NORM;

/* Global weight/bias gradient clipping (default) */
extern mat_t GRAD_CLIP_NORM;

/* Utility to set these at runtime if desired */
void config_set_act_bounds(mat_t pmin, mat_t pmax);
void config_set_z_clip(mat_t B);
void config_set_act_grad_clip(mat_t norm);

#endif
