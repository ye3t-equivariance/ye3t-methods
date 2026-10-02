# yaml-cpp

This directory contains the needed `include/` and `src/` trees from the upstream
[yaml-cpp 0.8.0 release](https://github.com/jbeder/yaml-cpp/releases/tag/0.8.0).
The downloaded release archive had SHA-256
`fbe74bbdcee21d656715688706da3c8becfd946d92cd44705cc6098bb23b3a16`.

The unused `contrib/` directories were omitted. Source adjustments are
`#include <cstdint>` in `src/emitterutils.cpp` for compilers that do not
provide fixed-width integers transitively, and trailing-space removal in
`src/singledocparser.cpp`. yaml-cpp is distributed under the MIT license in
`yaml-cpp/LICENSE`.
