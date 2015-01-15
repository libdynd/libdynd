#include <chrono>

#include <dynd/func/elwise.hpp>
#include <dynd/func/random.hpp>

using namespace std;
using namespace dynd;

nd::arrfunc nd::decl::random::uniform::as_arrfunc()
{
  std::random_device random_device;

  return nd::functional::elwise(nd::as_arrfunc<
      kernels::uniform_ck, std::default_random_engine, numeric_types>(
      ndt::type("(a: ?R, b: ?R, dst_tp: type) -> R"),
      std::shared_ptr<std::default_random_engine>(
          new std::default_random_engine(random_device()))));
}

nd::decl::random::uniform nd::random::uniform;