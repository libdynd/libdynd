int drand();
double exp(), frexp(), ldexp();
volatile double x, y, z;

main()
{
int i, e;

for( i=0; i<100000; i++ )
  {
    drand(&x);
    x = exp( 10.0*(x - 1.5) );
    y = frexp( x, &e );
    z = ldexp( y, e );
    if( z != x )
      abort();
  }
}
