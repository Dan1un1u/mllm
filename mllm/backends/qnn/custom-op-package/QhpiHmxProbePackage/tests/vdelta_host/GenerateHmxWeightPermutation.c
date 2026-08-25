#include <stdlib.h>

#include "vdelta_helper.h"

#define LN 128
#define logLN 7

int gsc[LN][4 * logLN];
int mux[2 * LN];
int tmux[2 * LN];
path top[2 * LN];
path bot[2 * LN];
int insb[2 * LN];
int outsb[2 * LN];
int inperm[2 * LN];
int result[2 * LN];
int hperm[LN];
int outperm[LN];

int main(void) {
  enum { logN = 7, vec_len = 128 };
  int tperm[vec_len];
  for (int destination = 0; destination < vec_len; ++destination) {
    const int output_channel = destination / 4;
    const int input_channel = destination % 4;
    tperm[destination] = input_channel * 32 + output_channel;
  }

  if (!tryvdelta(tperm, vec_len) || !tryvrdelta(tperm, vec_len)) return 0;
  for (int index = 0; index < vec_len; ++index) inperm[index] = index;
  if (check_benes(tperm, vec_len)) return 1;
  invert_permute(tperm, vec_len);
  for (int index = 0; index < vec_len; ++index) outperm[index] = tperm[index];
  gen_switch_cntrl(logN, top, bot, vec_len, inperm, outperm, insb, outsb, 0, 0);
  collect_switches(vec_len, logN);
  convert_const_geo(vec_len, logN);
  convert_rev_butterfly(gsc, vec_len, logN);
  print_hvx_bits(gsc, vec_len, logN);
  check_perm(gsc, mux, vec_len, logN);

  for (int index = 0; index < vec_len; ++index) inperm[index] = index;
  for (int index = 0; index < vec_len; ++index) result[tperm[index]] = index;
  for (int index = 0; index < vec_len; ++index) {
    if (mux[index] != result[index]) return 2;
  }
  return 0;
}
