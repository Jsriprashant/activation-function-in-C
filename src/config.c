#include "config.h"

/* Default values — can be overridden at runtime via setter functions */
mat_t ACT_PARAM_MIN = -10.0;
mat_t ACT_PARAM_MAX = 10.0;
mat_t ACT_Z_CLIP_B = 5.0;
mat_t ACT_GRAD_CLIP_NORM = 1.0;
mat_t GRAD_CLIP_NORM = 1.0;

void config_set_act_bounds(mat_t pmin, mat_t pmax)
{
    ACT_PARAM_MIN = pmin;
    ACT_PARAM_MAX = pmax;
}

void config_set_z_clip(mat_t B)
{
    ACT_Z_CLIP_B = B;
}

void config_set_act_grad_clip(mat_t norm)
{
    ACT_GRAD_CLIP_NORM = norm;
}
