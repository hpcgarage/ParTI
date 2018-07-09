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

#include <ParTI/algorithm.hpp>
#include <ParTI/argparse.hpp>
#include <ParTI/cfile.hpp>
#include <ParTI/timer.hpp>
#include <ParTI/session.hpp>
#include <ParTI/sptensor.hpp>
#include <ParTI/tensor.hpp>

using namespace pti;

int main(int argc, char const* argv[]) {
    bool dense_format = false;
    size_t limit = 10;
    std::string output;
    ParamDefinition defs[] = {
        { "-d",             PARAM_BOOL,   { &dense_format } },
        { "--dense-format", PARAM_BOOL,   { &dense_format } },
        { "-l",             PARAM_SIZET,  { &limit } },
        { "--limit",        PARAM_SIZET,  { &limit } },
        { "-o",             PARAM_STRING, { &output } },
        { "--output",       PARAM_STRING, { &output } },
        { ptiEndParamDefinition }
    };
    std::vector<char const*> args = parse_args(argc, argv, defs);

    if(args.size() < 2) {
        std::printf("Usage: %s [OPTIONS] X U1 U2 ...\n\n", argv[0]);
        std::printf("Options:\n");
        std::printf("\t-d, --dense-format\tPrint tensor in dense format instead of sparse format.\n");
        std::printf("\t-l, --limit\t\tLimit the number of elements to print [Default: 10].\n");
        std::printf("\t-o, --output\t\tWrite the result to a file\n");
        std::printf("\n");
        return 1;
    }

    session.print_devices();

    CFile fX(args[0], "r");
    SparseTensor X = SparseTensor::load(fX, 1);
    fX.fclose();

    std::printf("X = %s\n", X.to_string(!dense_format, limit).c_str());

    for(size_t argi = 1; argi < args.size(); ++argi) {

        if(X.nmodes < argi) {
            break;
        }
        size_t mode = X.nmodes - argi;

        CFile fU(args[argi], "r");
        Tensor U = Tensor::load(fU);
        fU.fclose();

        std::printf("\nU[%zu] = %s\n", argi, U.to_string(limit).c_str());

        Timer timer(cpu);
        timer.start();
        SparseTensor Y = tensor_times_matrix(X, U, mode, session.devices[cpu]);
        timer.stop();

        timer.print_elapsed_time("TTM");
        std::printf("Result: Y[%zu] = %s\n", argi, Y.to_string(!dense_format, limit).c_str());

        X = std::move(Y);

    }

    if(output.length() != 0) {
        CFile fY(output.c_str(), "w");
        X.dump(fY, 1);
    }

    return 0;
}
