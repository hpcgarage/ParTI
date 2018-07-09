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
#include <ParTI/utils.hpp>
#include <cstdlib>
#include <cstring>
#include <string>

namespace pti {

namespace {

std::vector<std::string> split_array(char const* s) {
    std::vector<std::string> result;
    size_t last_pos = 0;
    while(char const* c = std::strchr(s + last_pos, ',')) {
        size_t next_pos = c - s;
        result.push_back(std::string(s, last_pos, next_pos - last_pos));
        last_pos = next_pos;
    }
    result.push_back(std::string(s, last_pos));
    return result;
}

}

std::vector<char const*> parse_args(int argc, char const* argv[], ParamDefinition const defs[]) {
    std::vector<char const*> remaining;
    bool meet_dashdash = false;

    for(int argi = 1; argi < argc; ++argi) {
        char const* arg = argv[argi];
        if(!meet_dashdash) {
            if(std::strcmp(arg, "--") == 0) {
                meet_dashdash = true;
                continue;
            }
            bool def_found = false;
            for(ParamDefinition const* def = defs; def && def->option; ++def) {
                if(std::strcmp(arg, def->option) == 0) {
                    def_found = true;
                    switch(def->type) {
                    case PARAM_BOOL:
                        *def->vbool = true;
                        break;
                    case PARAM_STRING:
                        ++argi;
                        if(argi >= argc) {
                            throw std::invalid_argument((std::string("No argument for option") + arg).c_str());
                        }
                        *def->vstr = argv[argi];
                        break;
                    case PARAM_INT:
                        ++argi;
                        if(argi >= argc) {
                            throw std::invalid_argument((std::string("No argument for option") + arg).c_str());
                        }
                        *def->vint = strtonum(std::strtol, argv[argi], 0);
                        break;
                    case PARAM_SIZET:
                        ++argi;
                        if(argi >= argc) {
                            throw std::invalid_argument((std::string("No argument for option") + arg).c_str());
                        }
                        *def->vint = strtonum(std::strtoull, argv[argi], 0);
                        break;
                    case PARAM_SCALAR:
                        ++argi;
                        if(argi >= argc) {
                            throw std::invalid_argument((std::string("No argument for option") + arg).c_str());
                        }
                        *def->vscalar = strtonum(std::strtod, argv[argi]);
                        break;
                    case PARAM_STRINGS:
                        ++argi;
                        if(argi >= argc) {
                            throw std::invalid_argument((std::string("No argument for option") + arg).c_str());
                        }
                        def->vstrs->push_back(argv[argi]);
                        break;
                    case PARAM_INTS:
                        ++argi;
                        if(argi >= argc) {
                            throw std::invalid_argument((std::string("No argument for option") + arg).c_str());
                        }
                        {
                            std::vector<std::string> vints = split_array(argv[argi]);
                            for(std::string const& vint : vints) {
                                def->vints->push_back(strtonum(std::strtol, vint.c_str(), 0));
                            }
                        }
                        break;
                    case PARAM_SIZETS:
                        ++argi;
                        if(argi >= argc) {
                            throw std::invalid_argument((std::string("No argument for option") + arg).c_str());
                        }
                        {
                            std::vector<std::string> vsizets = split_array(argv[argi]);
                            for(std::string const& vsizet : vsizets) {
                                def->vsizets->push_back(strtonum(std::strtoull, vsizet.c_str(), 0));
                            }
                        }
                        break;
                    case PARAM_SCALARS:
                        ++argi;
                        if(argi >= argc) {
                            throw std::invalid_argument((std::string("No argument for option") + arg).c_str());
                        }
                        {
                            std::vector<std::string> vscalars = split_array(argv[argi]);
                            for(std::string const& vscalar : vscalars) {
                                def->vscalars->push_back(strtonum(std::strtod, vscalar.c_str()));
                            }
                        }
                        break;
                    default:
                        std::abort();
                    }
                    break;
                }
            }
            if(def_found) {
                continue;
            }
        }
        remaining.push_back(arg);
    }

    return remaining;
}

}
