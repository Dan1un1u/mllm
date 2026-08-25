#pragma once

static inline unsigned Q6_R_brev_R(unsigned value) {
  value = ((value & 0x55555555u) << 1) | ((value >> 1) & 0x55555555u);
  value = ((value & 0x33333333u) << 2) | ((value >> 2) & 0x33333333u);
  value = ((value & 0x0f0f0f0fu) << 4) | ((value >> 4) & 0x0f0f0f0fu);
  value = (value << 24) | ((value & 0xff00u) << 8) | ((value >> 8) & 0xff00u) | (value >> 24);
  return value;
}
