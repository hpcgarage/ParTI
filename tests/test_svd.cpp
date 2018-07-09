/*
    This file is part of ParTI!.

    ParTI! is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    ParTI! is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with ParTI!.
    If not, see <http://www.gnu.org/licenses/>.
*/

#include <memory>
#include <ParTI/algorithm.hpp>
#include <ParTI/argparse.hpp>
#include <ParTI/cfile.hpp>
#include <ParTI/errcode.hpp>
#include <ParTI/error.hpp>
#include <ParTI/timer.hpp>
#include <ParTI/session.hpp>
#include <ParTI/sptensor.hpp>
#include <ParTI/tensor.hpp>
#include <ParTI/utils.hpp>

using namespace pti;

int main(int argc, char const* argv[]) {
    size_t limit = 10;
    int device = 0;
    bool no_u = false, no_v = false;
    bool min_u = false, min_v = false;
    ParamDefinition defs[] = {
        { "-l",             PARAM_SIZET,  { &limit } },
        { "--limit",        PARAM_SIZET,  { &limit } },
        { "--dev",          PARAM_INT,    { &device } },
        { "--no-u",         PARAM_BOOL,   { &no_u } },
        { "--no-v",         PARAM_BOOL,   { &no_v } },
        { "--min-u",        PARAM_BOOL,   { &min_u } },
        { "--min-v",        PARAM_BOOL,   { &min_v } },
        { ptiEndParamDefinition }
    };
    std::vector<char const*> args = parse_args(argc, argv, defs);

    if(args.size() != 1) {
        std::printf("Usage: %s [OPTIONS] X\n\n", argv[0]);
        std::printf("Options:\n");
        std::printf("\t-l, --limit\t\tLimit the number of elements to print [Default: 10].\n");
        std::printf("\t--dev\t\tComputing device\n");
        std::printf("\t--no-u\t\tDo not calculate U\n");
        std::printf("\t--no-v\t\tDo not calculate V\n");
        std::printf("\t--min-u\t\tOnly calculate minimal part of U\n");
        std::printf("\t--min-v\t\tOnly calculate minimal part of V\n");
        std::printf("\n");
        return 1;
    }

    session.print_devices();
    Device* dev = session.devices[device];
    if(dynamic_cast<CudaDevice*>(dev) != nullptr) {
        std::printf("Using CUDA for calculation.\n");
    } else if(dynamic_cast<CpuDevice*>(dev) != nullptr) {
        std::printf("Using CPU for calculation.\n");
    } else {
        std::printf("Unknown device type.\n");
        return 1;
    }

    CFile fX(args[0], "r");
    Tensor X = Tensor::load(fX);
    fX.fclose();

    std::printf("X = %s\n\n", X.to_string(limit).c_str());

    Tensor U, S, V;
    svd(
        no_u ? nullptr : &U, false, min_u,
        S,
        no_v ? nullptr : &V, false, min_v,
        X, dev
    );

    if(!no_u) {
        std::printf("U = %s\n\n", U.to_string(limit).c_str());
    }
    std::printf("S = %s\n\n", S.to_string(limit).c_str());
    if(!no_v) {
        std::printf("V = %s\n\n", V.to_string(limit).c_str());
    }

    return 0;
}
