/*							arcdot.c
 *
 *	Angle between two vectors
 *
 *
 *
 *
 * SYNOPSIS:
 *
 * double p[3], q[3], arcdot();
 *
 * y = arcdot( p, q );
 *
 *
 *
 * DESCRIPTION:
 *
 * For two vectors p, q, the angle A between them is given by
 *
 *      p.q / (|p| |q|)  = cos A  .
 *
 * where "." represents inner product, "|x|" the length of vector x.
 * If the angle is small, an expression in sin A is preferred.
 * Set r = q - p.  Then
 *
 *     p.q = p.p + p.r ,
 *
 *     |p|^2 = p.p ,
 *
 *     |q|^2 = p.p + 2 p.r + r.r ,
 *
 *                  p.p^2 + 2 p.p p.r + p.r^2
 *     cos^2 A  =  ----------------------------
 *                    p.p (p.p + 2 p.r + r.r)
 *
 *                  p.p + 2 p.r + p.r^2 / p.p
 *              =  --------------------------- ,
 *                     p.p + 2 p.r + r.r
 *
 *     sin^2 A  =  1 - cos^2 A
 *
 *                   r.r - p.r^2 / p.p
 *              =  --------------------
 *                  p.p + 2 p.r + r.r
 *
 *              =   (r.r - p.r^2 / p.p) / q.q  .
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE      -1, 1        10^6       1.7e-16     4.2e-17
 *
 */

/*
Cephes Math Library Release 2.3:  November, 1995
Copyright 1995 by Stephen L. Moshier
*/

#include "mconf.h"
#ifdef ANSIPROT
extern double sqrt ( double );
extern double acos ( double );
extern double asin ( double );
extern double atan ( double );
#else
double sqrt(), acos(), asin(), atan();
#endif
extern double PI;

double arcdot(p,q)
double p[], q[];
{
double pp, pr, qq, rr, rt, pt, qt, pq;
int i;

pq = 0.0;
qq = 0.0;
pp = 0.0;
pr = 0.0;
rr = 0.0;
for (i=0; i<3; i++)
  {
    pt = p[i];
    qt = q[i];
    pq += pt * qt;
    qq += qt * qt;
    pp += pt * pt;
    rt = qt - pt;
    pr += pt * rt;
    rr += rt * rt;
  }
if (rr == 0.0 || pp == 0.0 || qq == 0.0)
  return 0.0;
rt = (rr - (pr * pr) / pp) / qq;
if (rt <= 0.75)
  {
    rt = sqrt(rt);
    qt = asin(rt);
    if (pq < 0.0)
      qt = PI - qt;
  }
else
  {
    pt = pq / sqrt(pp*qq);
    qt = acos(pt);
  }
return qt;
}
