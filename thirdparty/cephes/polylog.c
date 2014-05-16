/*							polylog.c
 *
 *	Polylogarithms
 *
 *
 *
 * SYNOPSIS:
 *
 * double x, y, polylog();
 * int n;
 *
 * y = polylog( n, x );
 *
 *
 * The polylogarithm of order n is defined by the series
 *
 *
 *              inf   k
 *               -   x
 *  Li (x)  =    >   ---  .
 *    n          -     n
 *              k=1   k
 *
 *
 *  For x = 1,
 *
 *               inf
 *                -    1
 *   Li (1)  =    >   ---   =  Riemann zeta function (n)  .
 *     n          -     n
 *               k=1   k
 *
 *
 *  When n = 2, the function is the dilogarithm, related to Spence's integral:
 *
 *                 x                      1-x
 *                 -                        -
 *                | |  -ln(1-t)            | |  ln t
 *   Li (x)  =    |    -------- dt    =    |    ------ dt    =   spence(1-x) .
 *     2        | |       t              | |    1 - t
 *               -                        -
 *                0                        1
 *
 *
 *  See also the program cpolylog.c for the complex polylogarithm,
 *  whose definition is extended to x > 1.
 *
 *  References:
 *
 *  Lewin, L., _Polylogarithms and Associated Functions_,
 *  North Holland, 1981.
 *
 *  Lewin, L., ed., _Structural Properties of Polylogarithms_,
 *  American Mathematical Society, 1991.
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain   n   # trials      peak         rms
 *    IEEE      0, 1     2     50000      6.2e-16     8.0e-17
 *    IEEE      0, 1     3    100000      2.5e-16     6.6e-17
 *    IEEE      0, 1     4     30000      1.7e-16     4.9e-17
 *    IEEE      0, 1     5     30000      5.1e-16     7.8e-17
 *
 */

/*
Cephes Math Library Release 2.8:  July, 1999
Copyright 1999 by Stephen L. Moshier
*/

#include "mconf.h"
extern double PI;

/* polylog(4, 1-x) = zeta(4) - x zeta(3) + x^2 A4(x)/B4(x)
   0 <= x <= 0.125
   Theoretical peak absolute error 4.5e-18  */
#if UNK
static double A4[13] = {
 3.056144922089490701751E-2,
 3.243086484162581557457E-1,
 2.877847281461875922565E-1,
 7.091267785886180663385E-2,
 6.466460072456621248630E-3,
 2.450233019296542883275E-4,
 4.031655364627704957049E-6,
 2.884169163909467997099E-8,
 8.680067002466594858347E-11,
 1.025983405866370985438E-13,
 4.233468313538272640380E-17,
 4.959422035066206902317E-21,
 1.059365867585275714599E-25,
};
static double B4[12] = {
  /* 1.000000000000000000000E0, */
 2.821262403600310974875E0,
 1.780221124881327022033E0,
 3.778888211867875721773E-1,
 3.193887040074337940323E-2,
 1.161252418498096498304E-3,
 1.867362374829870620091E-5,
 1.319022779715294371091E-7,
 3.942755256555603046095E-10,
 4.644326968986396928092E-13,
 1.913336021014307074861E-16,
 2.240041814626069927477E-20,
 4.784036597230791011855E-25,
};
#endif
#if DEC
static short A4[52] = {
0036772,0056001,0016601,0164507,
0037646,0005710,0076603,0176456,
0037623,0054205,0013532,0026476,
0037221,0035252,0101064,0065407,
0036323,0162231,0042033,0107244,
0035200,0073170,0106141,0136543,
0033607,0043647,0163672,0055340,
0031767,0137614,0173376,0072313,
0027676,0160156,0161276,0034203,
0025347,0003752,0123106,0064266,
0022503,0035770,0160173,0177501,
0017273,0056226,0033704,0132530,
0013403,0022244,0175205,0052161,
};
static short B4[48] = {
  /*0040200,0000000,0000000,0000000, */
0040464,0107620,0027471,0071672,
0040343,0157111,0025601,0137255,
0037701,0075244,0140412,0160220,
0037002,0151125,0036572,0057163,
0035630,0032452,0050727,0161653,
0034234,0122515,0034323,0172615,
0032415,0120405,0123660,0003160,
0030330,0140530,0161045,0150177,
0026002,0134747,0014542,0002510,
0023134,0113666,0035730,0035732,
0017723,0110343,0041217,0007764,
0014024,0007412,0175575,0160230,
};
#endif
#if IBMPC
static short A4[52] = {
0x3d29,0x23b0,0x4b80,0x3f9f,
0x7fa6,0x0fb0,0xc179,0x3fd4,
0x45a8,0xa2eb,0x6b10,0x3fd2,
0x8d61,0x5046,0x2755,0x3fb2,
0x71d4,0x2883,0x7c93,0x3f7a,
0x37ac,0x118c,0x0ecf,0x3f30,
0x4b5c,0xfcf7,0xe8f4,0x3ed0,
0xce99,0x9edf,0xf7f1,0x3e5e,
0xc710,0xdc57,0xdc0d,0x3dd7,
0xcd17,0x54c8,0xe0fd,0x3d3c,
0x7fe8,0x1c0f,0x677f,0x3c88,
0x96ab,0xc6f8,0x6b92,0x3bb7,
0xaa8e,0x9f50,0x6494,0x3ac0,
};
static short B4[48] = {
  /*0x0000,0x0000,0x0000,0x3ff0,*/
0x2e77,0x05e7,0x91f2,0x4006,
0x37d6,0x2570,0x7bc9,0x3ffc,
0x5c12,0x9821,0x2f54,0x3fd8,
0x4bce,0xa7af,0x5a4a,0x3fa0,
0xfc75,0x4a3a,0x06a5,0x3f53,
0x7eb2,0xa71a,0x94a9,0x3ef3,
0x00ce,0xb4f6,0xb420,0x3e81,
0xba10,0x1c44,0x182b,0x3dfb,
0x40a9,0xe32c,0x573c,0x3d60,
0x077b,0xc77b,0x92f6,0x3cab,
0xe1fe,0x6851,0x721c,0x3bda,
0xbc13,0x5f6f,0x81e1,0x3ae2,
};
#endif
#if MIEEE
static short A4[52] = {
0x3f9f,0x4b80,0x23b0,0x3d29,
0x3fd4,0xc179,0x0fb0,0x7fa6,
0x3fd2,0x6b10,0xa2eb,0x45a8,
0x3fb2,0x2755,0x5046,0x8d61,
0x3f7a,0x7c93,0x2883,0x71d4,
0x3f30,0x0ecf,0x118c,0x37ac,
0x3ed0,0xe8f4,0xfcf7,0x4b5c,
0x3e5e,0xf7f1,0x9edf,0xce99,
0x3dd7,0xdc0d,0xdc57,0xc710,
0x3d3c,0xe0fd,0x54c8,0xcd17,
0x3c88,0x677f,0x1c0f,0x7fe8,
0x3bb7,0x6b92,0xc6f8,0x96ab,
0x3ac0,0x6494,0x9f50,0xaa8e,
};
static short B4[48] = {
  /*0x3ff0,0x0000,0x0000,0x0000,*/
0x4006,0x91f2,0x05e7,0x2e77,
0x3ffc,0x7bc9,0x2570,0x37d6,
0x3fd8,0x2f54,0x9821,0x5c12,
0x3fa0,0x5a4a,0xa7af,0x4bce,
0x3f53,0x06a5,0x4a3a,0xfc75,
0x3ef3,0x94a9,0xa71a,0x7eb2,
0x3e81,0xb420,0xb4f6,0x00ce,
0x3dfb,0x182b,0x1c44,0xba10,
0x3d60,0x573c,0xe32c,0x40a9,
0x3cab,0x92f6,0xc77b,0x077b,
0x3bda,0x721c,0x6851,0xe1fe,
0x3ae2,0x81e1,0x5f6f,0xbc13,
};
#endif

#ifdef ANSIPROT
extern double spence ( double );
extern double polevl ( double, void *, int );
extern double p1evl ( double, void *, int );
extern double zetac ( double );
extern double pow ( double, double );
extern double powi ( double, int );
extern double log ( double );
extern double fac ( int i );
extern double fabs (double);
double polylog (int, double);
#else
extern double spence(), polevl(), p1evl(), zetac();
extern double pow(), powi(), log();
extern double fac(); /* factorial */
extern double fabs();
double polylog();
#endif
extern double MACHEP;

double
polylog (n, x)
     int n;
     double x;
{
  double h, k, p, s, t, u, xc, z;
  int i, j;

/*  This recurrence provides formulas for n < 2.

    d                 1
    --   Li (x)  =   ---  Li   (x)  .
    dx     n          x     n-1

*/

  if (n == -1)
    {
      p  = 1.0 - x;
      u = x / p;
      s = u * u + u;
      return s;
    }

  if (n == 0)
    {
      s = x / (1.0 - x);
      return s;
    }

  /* Not implemented for n < -1.
     Not defined for x > 1.  Use cpolylog if you need that.  */
  if (x > 1.0 || n < -1)
    {
      mtherr("polylog", DOMAIN);
      return 0.0;
    }

  if (n == 1)
    {
      s = -log (1.0 - x);
      return s;
    }

  /* Argument +1 */
  if (x == 1.0 && n > 1)
    {
      s = zetac ((double) n) + 1.0;
      return s;
    }

  /* Argument -1.
                        1-n
     Li (-z)  = - (1 - 2   ) Li (z)
       n                       n
   */
  if (x == -1.0 && n > 1)
    {
      /* Li_n(1) = zeta(n) */
      s = zetac ((double) n) + 1.0;
      s = s * (powi (2.0, 1 - n) - 1.0);
      return s;
    }

/*  Inversion formula:
 *                                                   [n/2]   n-2r
 *                n                  1     n           -  log    (z)
 *  Li (-z) + (-1)  Li (-1/z)  =  - --- log (z)  +  2  >  ----------- Li  (-1)
 *    n               n              n!                -   (n - 2r)!    2r
 *                                                    r=1
 */
  if (x < -1.0 && n > 1)
    {
      double q, w;
      int r;

      w = log (-x);
      s = 0.0;
      for (r = 1; r <= n / 2; r++)
	{
	  j = 2 * r;
	  p = polylog (j, -1.0);
	  j = n - j;
	  if (j == 0)
	    {
	      s = s + p;
	      break;
	    }
	  q = (double) j;
	  q = pow (w, q) * p / fac (j);
	  s = s + q;
	}
      s = 2.0 * s;
      q = polylog (n, 1.0 / x);
      if (n & 1)
	q = -q;
      s = s - q;
      s = s - pow (w, (double) n) / fac (n);
      return s;
    }

  if (n == 2)
    {
      if (x < 0.0 || x > 1.0)
	return (spence (1.0 - x));
    }



  /*  The power series converges slowly when x is near 1.  For n = 3, this
      identity helps:

      Li (-x/(1-x)) + Li (1-x) + Li (x)
        3               3          3
                     2                               2                 3
       = Li (1) + (pi /6) log(1-x) - (1/2) log(x) log (1-x) + (1/6) log (1-x)
           3
  */

  if (n == 3)
    {
      p = x * x * x;
      if (x > 0.8)
	{
	  /* Thanks to Oscar van Vlijmen for detecting an error here.  */
	  u = log(x);
	  s = u * u * u / 6.0;
	  xc = 1.0 - x;
	  s = s - 0.5 * u * u * log(xc);
          s = s + PI * PI * u / 6.0;
          s = s - polylog (3, -xc/x);
	  s = s - polylog (3, xc);
	  s = s + zetac(3.0);
	  s = s + 1.0;
	  return s;
	}
      /* Power series  */
      t = p / 27.0;
      t = t + .125 * x * x;
      t = t + x;

      s = 0.0;
      k = 4.0;
      do
	{
	  p = p * x;
	  h = p / (k * k * k);
	  s = s + h;
	  k += 1.0;
	}
      while (fabs(h/s) > 1.1e-16);
      return (s + t);
    }

if (n == 4)
  {
    if (x >= 0.875)
      {
	u = 1.0 - x;
	s = polevl(u, A4, 12) / p1evl(u, B4, 12);
	s =  s * u * u - 1.202056903159594285400 * u;
	s +=  1.0823232337111381915160;
	return s;
      }
    goto pseries;
  }


  if (x < 0.75)
    goto pseries;


/*  This expansion in powers of log(x) is especially useful when
    x is near 1.

    See also the pari gp calculator.

                      inf                  j
                       -    z(n-j) (log(x))
    polylog(n,x)  =    >   -----------------
                       -           j!
                      j=0

      where

      z(j) = Riemann zeta function (j), j != 1

                              n-1
                               -
      z(1) =  -log(-log(x)) +  >  1/k
                               -
                              k=1
  */

  z = log(x);
  h = -log(-z);
  for (i = 1; i < n; i++)
    h = h + 1.0/i;
  p = 1.0;
  s = zetac((double)n) + 1.0;
  for (j=1; j<=n+1; j++)
  {
    p = p * z / j;
    if (j == n-1)
      s = s + h * p;
    else
      s = s + (zetac((double)(n-j)) + 1.0) * p;
  }
  j = n + 3;
  z = z * z;
  for(;;)
    {
      p = p * z / ((j-1)*j);
      h = (zetac((double)(n-j)) + 1.0);
      h = h * p;
      s = s + h;
      if (fabs(h/s) < MACHEP)
	break;
      j += 2;
    }
  return s;


pseries:

  p = x * x * x;
  k = 3.0;
  s = 0.0;
  do
    {
      p = p * x;
      k += 1.0;
      h = p / powi(k, n);
      s = s + h;
    }
  while (fabs(h/s) > MACHEP);
  s += x * x * x / powi(3.0,n);
  s += x * x / powi(2.0,n);
  s += x;
  return s;
}
