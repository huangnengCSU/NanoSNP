#ifndef __TENSOR_HPP
#define __TENSOR_HPP

#include <array>

typedef enum {
    eChannel_A = 0,
    eChannel_C,
    eChannel_G,
    eChannel_T,
    eChannel_I,
    eChannel_I1,
    eChannel_D,
    eChannel_D1,
    eChannel_Star, // *
    eChannel_a,
    eChannel_c,
    eChannel_g,
    eChannel_t,
    eChannel_i,
    eChannel_i1,
    eChannel_d,
    eChannel_d1,
    eChannel_pound, // #
    eChannel_Size
} EChannelIndex;

EChannelIndex
string_to_channel_idx(const char p1, const char p2 = 'x');

typedef int tensor_cnt_type;

typedef struct {
    int x, y, z;
} TensorShape;

#define ONT_TENSOR_DEPTH 89

#endif // __TENSOR_HPP