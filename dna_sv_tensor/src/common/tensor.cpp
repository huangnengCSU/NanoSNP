#include "tensor.hpp"

EChannelIndex
string_to_channel_idx(const char p1, const char p2)
{
    switch(p1) {
        case 'A':
            return eChannel_A;
            break;
        case 'a':
            return eChannel_a;
            break;
        case 'C':
            return eChannel_C;
            break;
        case 'c':
            return eChannel_c;
            break;
        case 'G':
            return eChannel_G;
            break;
        case 'g':
            return eChannel_g;
            break;
        case 'T':
            return eChannel_T;
            break;
        case 't':
            return eChannel_t;
            break;      
        case 'I':
            return (p2 == '1') ? eChannel_I1 : eChannel_I;
            break;
        case 'i':
            return (p2 == '1') ? eChannel_i1 : eChannel_i;
            break;  
        case 'D':
            return (p2 == '1') ? eChannel_D1 : eChannel_D;
            break;
        case 'd':
            return (p2 == '1') ? eChannel_d1 : eChannel_d;
            break;  
        case '*':
            return eChannel_Star;
            break;
        case '#':
            return eChannel_pound;
            break;
        default:
            return eChannel_Size;
            break;
    }
}