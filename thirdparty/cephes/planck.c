/*							planck.c
 *
 *	Integral of Planck's black body radiation formula
 *
 *
 *
 * SYNOPSIS:
 *
 * double lambda, T, y, plancki();
 *
 * y = plancki( lambda, T );
 *
 *
 *
 * DESCRIPTION:
 *
 *  Evaluates the definite integral, from wavelength 0 to lambda,
 *  of Planck's radiation formula
 *                      -5
 *            c1  lambda
 *     E =  ------------------
 *            c2/(lambda T)
 *           e             - 1
 *
 * Physical constants c1 and c2 (see below) are built in
 * to the function program.  They are scaled to provide a result
 * in watts per square meter.  Argument T represents temperature in degrees
 * Kelvin; lambda is wavelength in meters.
 *
 * The integral is expressed in closed form, in terms of polylogarithms
 * (see polylog.c).
 *
 * The total area under the curve is
 *      (-1/8) (42 zeta(4) - 12 pi^2 zeta(2) + pi^4 ) c1 (T/c2)^4
 *       = (pi^4 / 15)  c1 (T/c2)^4
 *       =  sigma T^4
 * 
 *
 * CONSTANTS:
 *
 * First radiation constant c1 = 2 pi h c^2 = 3.741 771 53 (17) e-16 W m2
 * Second radiation constant c2 = h c / k  = 0.014 387 770 (13) m K
 * Stefan-Boltzmann constant sigma = 5.670 373 (21) e-8 W m^-2 K^-4
 * Wien wavelength displacement law constant  wien = 2.8977721 (26) e-3 m K
 * These are NIST values as of 2010.
 *
 *
 * ACCURACY:
 *
 * The left tail of the function experiences some relative error
 * amplification in computing the dominant term exp(-c2/(lambda T)).
 * For the right-hand tail see planckc, below.  These error estimates
 * do not count uncertainty in the physical constants.
 *
 *                      Relative error.
 *   The domain refers to lambda T / c2.
 * arithmetic   domain     # trials      peak         rms
 *    IEEE      0.1, 10      50000      7.1e-15     5.4e-16
 *
 */


/*
Cephes Math Library Release 2.8:  July, 1999
Copyright 1999 by Stephen L. Moshier
*/

#include "mconf.h"
#ifdef ANSIPROT
extern double polylog (int, double);
extern double exp (double);
extern double log1p (double); /* log(1+x) */
extern double expm1 (double); /* exp(x) - 1 */
double planckc(double, double);
double plancki(double, double);
#else
double polylog(), exp(), log1p(), expm1();
double planckc(), plancki();
#endif

double planck_c1 = 3.74177153e-16;
double planck_c2 = 0.014387770;
static double wien = 2.8977721e-3;



double
plancki(w, T)
  double w, T;
{
  double b, h, y, bw;

  b = T / planck_c2;
  bw = b * w;

  if (bw > 0.59375)
    {
      y = b * b;
      h = y * y;
      /* Right tail.  */
      y = planckc (w, T);
      /* pi^4 / 15  */
      y =  6.493939402266829149096 * planck_c1 * h  -  y;
      return y;
    }

  h = exp(-planck_c2/(w*T));
  y =      6. * polylog (4, h)  * bw;
  y = (y + 6. * polylog (3, h)) * bw;
  y = (y + 3. * polylog (2, h)) * bw;
  y = (y          - log1p (-h)) * bw;
  h = w * w;
  h = h * h;
  y = y * (planck_c1 / h);
  return y;
}

/*							planckc
 *
 *	Complemented Planck radiation integral
 *
 *
 *
 * SYNOPSIS:
 *
 * double lambda, T, y, planckc();
 *
 * y = planckc( lambda, T );
 *
 *
 *
 * DESCRIPTION:
 *
 *  Integral from w to infinity (area under right hand tail)
 *  of Planck's radiation formula.
 *
 *  The program for large lambda uses an asymptotic series in inverse
 *  powers of the wavelength.
 *
 * ACCURACY:
 *
 *                      Relative error.
 *   The domain refers to lambda T / c2.
 * arithmetic   domain     # trials      peak         rms
 *    IEEE      0.6, 10      50000      1.1e-15     2.2e-16
 *
 */

double
planckc (w, T)
     double w;
     double T;
{
  double b, d, p, u, y;

  b = T / planck_c2;
  d = b*w;
  if (d <= 0.59375)
    {
      y =  6.493939402266829149096 * planck_c1 * b*b*b*b;
      return (y - plancki(w,T));
    }
  u = 1.0/d;
  p = u * u;
#if 0
  y = 236364091.*p/365866013534056632601804800000.;
  y = (y - 15458917./475677107995483570176000000.)*p;
  y = (y + 174611./123104841613737984000000.)*p;
  y = (y - 43867./643745871363538944000.)*p;
  y = ((y + 3617./1081289781411840000.)*p - 1./5928123801600.)*p;
  y = ((y + 691./78460462080000.)*p - 1./2075673600.)*p;
  y = ((((y + 1./35481600.)*p - 1.0/544320.)*p + 1.0/6720.)*p -  1./40.)*p;
  y = y + log(d * expm1(u));
  y = y - 5.*u/8. + 1./3.;
#else
  y = -236364091.*p/45733251691757079075225600000.;
  y = (y + 77683./352527500984795136000000.)*p;
  y = (y - 174611./18465726242060697600000.)*p;
  y = (y + 43867./107290978560589824000.)*p;
  y = ((y - 3617./202741834014720000.)*p + 1./1270312243200.)*p;
  y = ((y - 691./19615115520000.)*p + 1./622702080.)*p;
  y = ((((y - 1./13305600.)*p + 1./272160.)*p - 1./5040.)*p + 1./60.)*p;
  y = y - 0.125*u + 1./3.;
#endif
  y = y * planck_c1 * b / (w*w*w);
  return y;
}


/*							planckd
 *
 *	Planck's black body radiation formula
 *
 *
 *
 * SYNOPSIS:
 *
 * double lambda, T, y, planckd();
 *
 * y = planckd( lambda, T );
 *
 *
 *
 * DESCRIPTION:
 *
 *  Evaluates Planck's radiation formula
 *                      -5
 *            c1  lambda
 *     E =  ------------------
 *            c2/(lambda T)
 *           e             - 1
 *
 */

double
planckd(w, T)
  double w, T;
{
   return (planck_c1 / ((w*w*w*w*w) * (exp(planck_c2/(w*T)) - 1.0)));
}


/* Wavelength, w, of maximum radiation at given temperature T.
   c2/wT = constant
   Wein displacement law.
  */
double
planckw(T)
  double T;
{
  return (wien / T);
}
