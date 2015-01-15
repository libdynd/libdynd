#include <chrono>

#include <dynd/func/elwise.hpp>
#include <dynd/func/random.hpp>

using namespace std;
using namespace dynd;

nd::arrfunc nd::decl::random::uniform::as_arrfunc()
{
  unsigned int seed =
      std::chrono::system_clock::now().time_since_epoch().count();

  return nd::functional::elwise(nd::as_arrfunc<
      kernels::uniform_ck, std::default_random_engine, numeric_types>(
      ndt::type("(a: ?R, b: ?R, dst_tp: type) -> R"),
      std::shared_ptr<std::default_random_engine>(
          new std::default_random_engine(seed))));
}

nd::decl::random::uniform nd::random::uniform;