/*							minv.c
 *
 *	Matrix inversion
 *
 *
 *
 * SYNOPSIS:
 *
 * int n, errcod;
 * double A[n*n], X[n*n];
 * double B[n];
 * int IPS[n];
 * int minv();
 *
 * errcod = minv( A, X, n, B, IPS );
 *
 *
 *
 * DESCRIPTION:
 *
 * Finds the inverse of the n by n matrix A.  The result goes
 * to X.   B and IPS are scratch pad arrays of length n.
 * The contents of matrix A are destroyed.
 *
 * The routine returns nonzero on error; error messages are printed
 * by subroutine simq().
 *
 */

minv( A, X, n, B, IPS )
double A[], X[];
int n;
double B[];
int IPS[];
{
double *pX;
int i, j, k;

for( i=1; i<n; i++ )
	B[i] = 0.0;
B[0] = 1.0;
/* Reduce the matrix and solve for first right hand side vector */
pX = X;
k = simq( A, B, pX, n, 1, IPS );
if( k )
	return(-1);
/* Solve for the remaining right hand side vectors */
for( i=1; i<n; i++ )
	{
	B[i-1] = 0.0;
	B[i] = 1.0;
	pX += n;
	k = simq( A, B, pX, n, -1, IPS );
	if( k )
		return(-1);
	}
/* Transpose the array of solution vectors */
mtransp( n, X, X );
return(0);
}

