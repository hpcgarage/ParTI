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

#ifndef PTI_ARGPARSE_INCLUDED
#define PTI_ARGPARSE_INCLUDED

#include <vector>
#include <stdexcept>
#include <string>
#include <ParTI/scalar.hpp>

namespace pti {

enum ParamType {
    PARAM_BOOL,

    PARAM_STRING,
    PARAM_INT,
    PARAM_SIZET,
    PARAM_SCALAR,

    PARAM_STRINGS,
    PARAM_INTS,
    PARAM_SIZETS,
    PARAM_SCALARS
};

struct ParamDefinition {
    char const* option;
    ParamType   type;
    union {
        void*                     value;

        bool*                     vbool;

        std::string*              vstr;
        int*                      vint;
        size_t*                   vsizet;
        Scalar*                   vscalar;

        std::vector<std::string>* vstrs;
        std::vector<int>*         vints;
        std::vector<size_t>*      vsizets;
        std::vector<Scalar>*      vscalars;
    };
};

#define ptiEndParamDefinition nullptr, PARAM_BOOL, { nullptr }

std::vector<char const*> parse_args(
    int argc,
    char const* argv[],
    ParamDefinition const defs[]
);

}

#endif
