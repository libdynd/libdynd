/*							ltstd.c		*/
/*  Function test routine.
 *  Requires long double type check routine and double precision function
 *  under test.  Indicate function name and range in #define statements
 *  below.  Modifications for two argument functions and absolute
 *  rather than relative accuracy report are indicated.
 */

#include <stdio.h>
/* int printf(), gets(), sscanf(); */

#include "mconf.h"
#ifdef ANSIPROT
int drand ( double * );
int dprec ( void );
int ldprec ( void );
double exp ( double );
double sqrt ( double );
double fabs ( double );
double floor ( double );
long double sqrtl ( long double );
long double fabsl ( long double );
#else
int drand();
int dprec(), ldprec();
double exp(), sqrt(), fabs(), floor();
long double sqrtl(), fabsl();
#endif

#define RELERR 1
#define ONEARG 0
#define ONEINT 0
#define TWOARG 0
#define TWOINT 0
#define THREEARG 1
#define THREEINT 0
#define FOURARG 0
#define VECARG 0
#define FOURANS 0
#define TWOANS 0
#define PROB 0
#define EXPSCALE 0
#define EXPSC2 0
/* insert function to be tested here: */
#define FUNC hyperg
double FUNC();
#define QFUNC hypergl
long double QFUNC();
/*extern int aiconf;*/

extern double MAXLOG;
extern double MINLOG;
extern double MAXNUM;
#define LTS 3.258096538
/* insert low end and width of test interval */
#define LOW 0.0
#define WIDTH 30.0
#define LOWA  0.0
#define WIDTHA 30.0
/* 1.073741824e9 */
/* 2.147483648e9 */
long double qone = 1.0L;
static long double q1, q2, q3, qa, qb, qc, qz, qy1, qy2, qy3, qy4;
static double y2, y3, y4, a, b, c, x, y, z, e;
static long double qe, qmax, qrmsa, qave;
volatile double v;
static long double lp[3], lq[3];
static double dp[3], dq[3];

char strave[20];
char strrms[20];
char strmax[20];
double underthresh =  2.22507385850720138309E-308; /* 2^-1022 */

void main()
{
char s[80];
int i, j, k;
long m, n;

merror = 0;
ldprec();   /* set up coprocessor.  */
/*aiconf = -1;*/	/* configure Airy function */
x = 1.0;
z = x * x;
qmax = 0.0L;
sprintf(strmax, "%.4Le", qmax );
qrmsa = 0.0L;
qave = 0.0L;

#if 1
printf(" Start at random number #:" );
gets( s );
sscanf( s, "%ld", &n );
printf("%ld\n", n );
#else
n = 0;
#endif

for( m=0; m<n; m++ )
	drand( &x );
n = 0;
m = 0;
x = floor( x );

loop:

for( i=0; i<500; i++ )
{
n++;
m++;

#if ONEARG || TWOARG || THREEARG || FOURARG
/*ldprec();*/	/* set up floating point coprocessor */
/* make random number in desired range */
drand( &x );
x = WIDTH *  ( x - 1.0 )  +  LOW;
#if EXPSCALE
x = exp(x);
drand( &a );
a = 1.0e-13 * x * a;
if( x > 0.0 )
	x -= a;
else
	x += a;
#endif
#if ONEINT
k = x;
x = k;
#endif
v = x;
q1 = v;		/* double number to q type */
#endif

/* do again if second argument required */

#if TWOARG || THREEARG || FOURARG
drand( &a );
a = WIDTHA *  ( a - 1.0 )  +  LOWA;
/*a /= 50.0;*/
#if EXPSC2
a = exp(a);
drand( &y2 );
y2 = 1.0e-13 * y2 * a;
if( a > 0.0 )
	a -= y2;
else
	a += y2;
#endif
#if TWOINT || THREEINT
k = a + 0.25;
a = k;
#endif
v = a;
qy4 = v;
#endif

#if THREEARG || FOURARG
drand( &b );
#if PROB
/*
b = b - 1.0;
b = a * b;
*/
#if 1
/* This makes b <= a, for bdtr.  */
b = (a - LOWA) *  ( b - 1.0 )  +  LOWA;
if( b > 1.0 && a > 1.0 )
  b -= 1.0;
else
  {
    a += 1.0;
    k = a;
    a = k;
    v = a;
    qy4 = v;
  }
#else
b = WIDTHA *  ( b - 1.0 )  +  LOWA;
#endif

/* Half-integer a and b */
/*
a = 0.5*floor(2.0*a+1.0);
b = 0.5*floor(2.0*b+1.0);
*/
v = a;
qy4 = v;
/*x = (a / (a+b));*/

#else
b = WIDTHA *  ( b - 1.0 )  +  LOWA;
#endif
#if THREEINT
j = b + 0.25;
b = j;
#endif
v = b;
qb = v;
#endif

#if FOURARG
drand( &c );
c = WIDTHA *  ( c - 1.0 )  +  LOWA;
/* for hyp2f1 to ensure c-a-b > -1 */
/*
z = c-a-b;
if( z < -1.0 )
	c -= 1.6 * z;
*/
v = c;
qc = v;
#endif

#if VECARG
for( j=0; j<3; j++)
  {
    drand( &x );
    x = WIDTH *  ( x - 1.0 )  +  LOW;
    v = x;
    dp[j] = v;
    q1 = v;		/* double number to q type */
    lp[j] = q1;
    drand( &x );
    x = WIDTH *  ( x - 1.0 )  +  LOW;
    v = x;
    dq[j] = v;
    q1 = v;		/* double number to q type */
    lq[j] = q1;
  }
#endif /* VECARG */

/*printf("%.16E %.16E\n", a, x);*/
/* compute function under test */
/* Set to double precision */
/*dprec();*/
#if ONEARG
#if FOURANS
/*FUNC( x, &z, &y2, &y3, &y4 );*/
FUNC( x, &y4, &y2, &y3, &z );
#else
#if TWOANS
FUNC( x, &z, &y2 );
/*FUNC( x, &y2, &z );*/
#else
#if ONEINT
z = FUNC( k );
#else
z = FUNC( x );
#endif
#endif
#endif
#endif

#if TWOARG
#if TWOINT
z = FUNC( k, x );
/*z = FUNC( x, k );*/
/*z = FUNC( a, x );*/
#else
#if FOURANS
FUNC( a, x, &z, &y2, &y3, &y4 );
#else
z = FUNC( a, x );
#endif
#endif
#endif

#if THREEARG
#if THREEINT
z = FUNC( j, k, x );
#else
z = FUNC( a, b, x );
#endif
#endif

#if FOURARG
z = FUNC( a, b, c, x );
#endif

#if VECARG
z = FUNC( dp, dq );
#endif

q2 = z;
/* handle detected overflow */
if( (z == MAXNUM) || (z == -MAXNUM) )
	{
	printf("detected overflow ");
#if FOURARG
	printf("%.4E %.4E %.4E %.4E %.4E %6ld \n",
		a, b, c, x, y, n);
#else
	printf("%.16E %.4E %.4E %6ld \n", x, a, z, n);
#endif
	e = 0.0;
	m -= 1;
	goto endlup;
	}
/* Skip high precision if underflow.  */
if( merror == UNDERFLOW )
  goto underf;

/* compute high precision function */
/*ldprec();*/
#if ONEARG
#if FOURANS
/*qy4 = QFUNC( q1, qz, qy2, qy3 );*/
qz = QFUNC( q1, qy4, qy2, qy3 );
#else
#if TWOANS
qy2 = QFUNC( q1, qz );
/*qz = QFUNC( q1, qy2 );*/
#else
/* qy4 = 0.0L;*/
/* qy4 = 1.0L;*/
/*qz = QFUNC( qy4, q1 );*/
/*qz = QFUNC( 1, q1 );*/
qz = QFUNC( q1 );  /* normal */
#endif
#endif
#endif

#if TWOARG
#if TWOINT
qz = QFUNC( k, q1 );
/*qz = QFUNC( q1, qy4 );*/
/*qz = QFUNC( qy4, q1 );*/
#else
#if FOURANS
qc = QFUNC( qy4, q1, qz, qy2, qy3 );
#else
/*qy4 = 0.0L;;*/
/*qy4 = 1.0L );*/
qz = QFUNC( qy4, q1 );
#endif
#endif
#endif

#if THREEARG
#if THREEINT
qz = QFUNC( j, k, q1 );
#else
qz = QFUNC( qy4, qb, q1 );
#endif
#endif

#if FOURARG
qz = QFUNC( qy4, qb, qc, q1 );
#endif

#if VECARG
qz = QFUNC( lp, lq );
#endif

y = qz; /* correct answer, in double precision */

/* get absolute error, in extended precision */
qe = q2 - qz;
e = qe; /* the error in double precision */

/*  handle function result equal to zero
    or underflowed. */
if( qz == 0.0L || merror == UNDERFLOW || fabs(z) < underthresh )
	{
underf:
	  merror = 0;
/* Don't bother to print anything.  */
#if 0
	printf("ans 0 ");
#if ONEARG
	printf("%.8E %.8E %.4E %6ld \n", x, y, e, n);
#endif

#if TWOARG
#if TWOINT
	printf("%d %.8E %.8E %.4E %6ld \n", k, x, y, e, n);
#else
	printf("%.6E %.6E %.6E %.4E %6ld \n", a, x, y, e, n);
#endif
#endif

#if THREEARG
	printf("%.6E %.6E %.6E %.6E %.4E %6ld \n", a, b, x, y, e, n);
#endif

#if FOURARG
	printf("%.4E %.4E %.4E %.4E %.4E %.4E %6ld \n",
		a, b, c, x, y, e, n);
#endif
#endif /* 0 */
	  qe = 0.0L;
	e = 0.0;
	m -= 1;
	goto endlup;
	}

else

/*	relative error	*/

/* comment out the following two lines if absolute accuracy report */

#if RELERR
  qe = qe / qz;
#else
	{
	  q2 = qz;
	  q2 = fabsl(q2);
	  if( q2 > 1.0L )
	    qe = qe / qz;
	}
#endif

qave = qave + qe;
/* absolute value of error */
qe = fabs(qe);

/* peak detect the error */
if( qe > qmax )
	{
	  qmax = qe;
	  sprintf(strmax, "%.4Le", qmax );
#if ONEARG
	printf("%.8E %.8E %s %6ld \n", x, y, strmax, n);
#endif
#if TWOARG
#if TWOINT
	printf("%d %.8E %.8E %s %6ld \n", k, x, y, strmax, n);
#else
	printf("%.6E %.6E %.6E %s %6ld \n", a, x, y, strmax, n);
#endif
#endif
#if THREEARG
	printf("%.6E %.6E %.6E %.6E %s %6ld \n", a, b, x, y, strmax, n);
#endif
#if FOURARG
	printf("%.4E %.4E %.4E %.4E %.4E %s %6ld \n",
		a, b, c, x, y, strmax, n);
#endif
#if VECARG
	printf("%.8E %s %6ld \n", y, strmax, n);
#endif
	}

/* accumulate rms error	*/
/* rmsa += e * e;  accumulate the square of the error */
q2 = qe * qe;
qrmsa = qrmsa + q2;
endlup:   ;
/*ldprec();*/
}

/* report every 500 trials */
/* rms = sqrt( rmsa/m ); */
q1 = m;
q2 = qrmsa / q1;
q2 = sqrtl(q2);
sprintf(strrms, "%.4Le", q2 );

q2 = qave / q1;
sprintf(strave, "%.4Le", q2 );
/*
printf("%6ld   max = %s   rms = %s  ave = %s \n", m, strmax, strrms, strave );
*/
printf("%6ld   max = %s   rms = %s  ave = %s \r", m, strmax, strrms, strave );
fflush(stdout);
goto loop;
}
