#include <chrono>

#include <dynd/func/elwise.hpp>
#include <dynd/func/random.hpp>

using namespace std;
using namespace dynd;

nd::arrfunc nd::decl::random::uniform::as_arrfunc()
{
  unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();

  nd::arrfunc host_self = nd::as_arrfunc<
      kernels::uniform_ck, std::default_random_engine, numeric_types>(
      ndt::type("(a: ?R, b: ?R, tp: type | R) -> R"),
      std::shared_ptr<std::default_random_engine>(
          new std::default_random_engine(seed)));

  return nd::functional::elwise(
      ndt::type("(a: ?R, b: ?R, tp: type | Dims... * R) -> Dims... * R"),
      host_self);
}

nd::decl::random::uniform nd::random::uniform;