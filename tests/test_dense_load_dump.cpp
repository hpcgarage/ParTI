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

#include <ParTI/argparse.hpp>
#include <ParTI/cfile.hpp>
#include <ParTI/tensor.hpp>

using namespace pti;

int main(int argc, char const* argv[]) {
    size_t limit = 10;
    ParamDefinition defs[] = {
        { "-l",             PARAM_SIZET, { &limit } },
        { "--limit",        PARAM_SIZET, { &limit } },
        { ptiEndParamDefinition }
    };
    std::vector<char const*> args = parse_args(argc, argv, defs);

    if(args.size() != 1 && args.size() != 2) {
        std::printf("Usage: %s [OPTIONS] input_tensor [output_tensor]\n\n", argv[0]);
        std::printf("Options:\n");
        std::printf("\t-l, --limit\t\tLimit the number of elements to print [Default: 10].\n");
        std::printf("\n");
        return 1;
    }

    CFile fi(args[0], "r");
    Tensor tsr = Tensor::load(fi);
    fi.fclose();

    std::printf("tsr = %s\n", tsr.to_string(limit).c_str());

    if(args.size() == 2) {
        CFile fo(args[1], "w");
        tsr.dump(fo);
    }

    return 0;
}
