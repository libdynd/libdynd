/*							lrand.c
 *
 *	Pseudorandom number generator
 *
 *
 *
 * SYNOPSIS:
 *
 * long y, drand();
 *
 * drand( &y );
 *
 *
 *
 * DESCRIPTION:
 *
 * Yields a long integer random number.
 *
 * The three-generator congruential algorithm by Brian
 * Wichmann and David Hill (BYTE magazine, March, 1987,
 * pp 127-8) is used. The period, given by them, is
 * 6953607871644.
 *
 *
 */



#include "mconf.h"


/*  Three-generator random number algorithm
 * of Brian Wichmann and David Hill
 * BYTE magazine, March, 1987 pp 127-8
 *
 * The period, given by them, is (p-1)(q-1)(r-1)/4 = 6.95e12.
 */

static int sx = 1;
static int sy = 10000;
static int sz = 3000;

/* This function implements the three
 * congruential generators.
 */
 
long lrand()
{
int r, s;
unsigned long ans;

/*
if( arg )
	{
	sx = 1;
	sy = 10000;
	sz = 3000;
	}
*/

/*  sx = sx * 171 mod 30269 */
r = sx/177;
s = sx - 177 * r;
sx = 171 * s - 2 * r;
if( sx < 0 )
	sx += 30269;


/* sy = sy * 172 mod 30307 */
r = sy/176;
s = sy - 176 * r;
sy = 172 * s - 35 * r;
if( sy < 0 )
	sy += 30307;

/* sz = 170 * sz mod 30323 */
r = sz/178;
s = sz - 178 * r;
sz = 170 * s - 63 * r;
if( sz < 0 )
	sz += 30323;

ans = sx * sy * sz;
return(ans);
}

