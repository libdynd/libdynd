#include <dynd/func/apply.hpp>
#include <dynd/func/random.hpp>

using namespace std;
using namespace dynd;

nd::arrfunc nd::random::default_random_engine()
{
  return functional::apply(new std::default_random_engine(),
                           &delete_wrapper<std::default_random_engine>);
}

nd::arrfunc nd::random::minstd_rand()
{
  return functional::apply(new std::minstd_rand(),
                           &delete_wrapper<std::minstd_rand>);
}

nd::arrfunc nd::random::minstd_rand(std::minstd_rand::result_type seed)
{
  return functional::apply(new std::minstd_rand(seed),
                           &delete_wrapper<std::minstd_rand>);
}

decl::nd::uniform nd::uniform;