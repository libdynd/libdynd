//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/functional.hpp>
#include <dynd/callables/string_concat_callable.hpp>
#include <dynd/callables/string_count_callable.hpp>
#include <dynd/callables/string_find_callable.hpp>
#include <dynd/callables/string_rfind_callable.hpp>
#include <dynd/callables/string_replace_callable.hpp>
#include <dynd/callables/string_split_callable.hpp>
#include <dynd/string.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::string_concatenation =
    nd::functional::elwise(nd::make_callable<nd::string_concat_callable>());

DYND_API nd::callable nd::string_count = nd::functional::elwise(nd::make_callable<nd::string_count_callable>());

DYND_API nd::callable nd::string_find = nd::functional::elwise(nd::make_callable<nd::string_find_callable>());

DYND_API nd::callable nd::string_rfind = nd::functional::elwise(nd::make_callable<nd::string_rfind_callable>());

DYND_API nd::callable nd::string_replace = nd::functional::elwise(nd::make_callable<nd::string_replace_callable>());

DYND_API nd::callable nd::string_split = nd::functional::elwise(nd::make_callable<nd::string_split_callable>());
