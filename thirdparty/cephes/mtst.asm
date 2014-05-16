;	Static Name Aliases
;
;	$S192_y4	EQU	y4
;	$S193_a	EQU	a
;	$S194_x	EQU	x
;	$S195_y	EQU	y
;	$S196_z	EQU	z
;	$S197_e	EQU	e
;	$S198_max	EQU	max
;	$S199_rmsa	EQU	rmsa
;	$S200_rms	EQU	rms
;	$S201_ave	EQU	ave
;	$S181_headrs	EQU	headrs
;	$S189_y1	EQU	y1
;	$S190_y2	EQU	y2
;	$S191_y3	EQU	y3
	TITLE   mtst.c
	.8087
INCLUDELIB	SLIBCE
_TEXT	SEGMENT  WORD PUBLIC 'CODE'
_TEXT	ENDS
_DATA	SEGMENT  WORD PUBLIC 'DATA'
_DATA	ENDS
CONST	SEGMENT  WORD PUBLIC 'CONST'
CONST	ENDS
_BSS	SEGMENT  WORD PUBLIC 'BSS'
_BSS	ENDS
DGROUP	GROUP	CONST, _BSS, _DATA
	ASSUME DS: DGROUP, SS: DGROUP
PUBLIC  _defs
EXTRN	__acrtused:ABS
EXTRN	_yn:NEAR
EXTRN	_iv:NEAR
EXTRN	_kn:NEAR
EXTRN	_exp2:NEAR
EXTRN	_log2:NEAR
EXTRN	_exp10:NEAR
EXTRN	_log10:NEAR
EXTRN	_exit:NEAR
EXTRN	_dprec:NEAR
EXTRN	_printf:NEAR
EXTRN	__aNchkstk:NEAR
EXTRN	_ndtr:NEAR
EXTRN	_ndtri:NEAR
EXTRN	__aNftol:NEAR
EXTRN	_drand:NEAR
EXTRN	_ellpe:NEAR
EXTRN	_ellpk:NEAR
EXTRN	_gamma:NEAR
EXTRN	_lgam:NEAR
EXTRN	_fabs:NEAR
EXTRN	_sqrt:NEAR
EXTRN	_cbrt:NEAR
EXTRN	_exp:NEAR
EXTRN	_log:NEAR
EXTRN	_tan:NEAR
EXTRN	_atan:NEAR
EXTRN	_sin:NEAR
EXTRN	_asin:NEAR
EXTRN	_cos:NEAR
EXTRN	_acos:NEAR
EXTRN	_pow:NEAR
EXTRN	_tanh:NEAR
EXTRN	_atanh:NEAR
EXTRN	_sinh:NEAR
EXTRN	_asinh:NEAR
EXTRN	_cosh:NEAR
EXTRN	_acosh:NEAR
EXTRN	_jn:NEAR
EXTRN	_MAXLOG:QWORD
EXTRN	_PI:QWORD
EXTRN	_PIO2:QWORD
EXTRN	__fac:QWORD
EXTRN	__fltused:NEAR
_DATA      SEGMENT
$SG147	DB	'  cube',  00H
$SG148	DB	'  cbrt',  00H
$SG149	DB	'   tan',  00H
$SG150	DB	'  atan',  00H
$SG151	DB	'  asin',  00H
$SG152	DB	'   sin',  00H
$SG153	DB	'square',  00H
$SG154	DB	'  sqrt',  00H
$SG155	DB	'   exp',  00H
$SG156	DB	'   log',  00H
$SG157	DB	' atanh',  00H
$SG158	DB	'  tanh',  00H
$SG159	DB	'  sinh',  00H
$SG160	DB	' asinh',  00H
$SG161	DB	'  cosh',  00H
$SG162	DB	' acosh',  00H
$SG163	DB	'  exp2',  00H
$SG164	DB	'  log2',  00H
$SG165	DB	' exp10',  00H
$SG166	DB	' log10',  00H
$SG167	DB	'pow',  00H
$SG168	DB	'pow',  00H
$SG169	DB	' ellpe',  00H
$SG170	DB	' ellpk',  00H
$SG171	DB	'  ndtr',  00H
$SG172	DB	' ndtri',  00H
$SG173	DB	'gamma',  00H
$SG174	DB	'lgam',  00H
$SG175	DB	'  acos',  00H
$SG176	DB	'   cos',  00H
$SG177	DB	'  Jn',  00H
$SG178	DB	'  Yn',  00H
$SG179	DB	'  Iv',  00H
$SG180	DB	'  Kn',  00H
$SG182	DB	'x = %s( %s(x) ): ',  00H
$SG183	DB	'x = %s( %s(x,a),1/a ): ',  00H
$SG184	DB	'Legendre %s, %s: ',  00H
$SG185	DB	'%s(x) = log(%s(x)): ',  00H
$SG186	DB	'Wronksian of %s, %s: ',  00H
$SG187	DB	'Wronksian of %s, %s: ',  00H
$SG188	DB	'Wronksian of %s, %s: ',  00H
$SG214	DB	'Consistency test of math functions.',  0aH,  00H
$SG215	DB	'Max and rms relative errors for %d random arguments.',  0aH,  00H
$SG220	DB	'Absolute error criterion (but relative if >1):',  0aH,  00H
$SG222	DB	'Absolute error and only %d trials:',  0aH,  00H
$SG263	DB	'Illegal nargs= %d',  00H
$SG279	DB	'x %.6E z %.6E y %.6E max %.4E',  0aH,  00H
$SG281	DB	'a %.6E',  0aH,  00H
$SG283	DB	'y1 %.4E y2 %.4E y3 %.4E y4 %.4E k %d x %.4E',  0aH,  00H
$SG284	DB	' max = %.2E   rms = %.2E',  0aH,  00H
	ORG	$+1
_defs	DW	DGROUP:$SG147
	DW	_cube
	DW	DGROUP:$SG148
	DW	_cbrt
	DW	01H
	DW	00H
	DD	01H
	DQ	0409f400000000000r    ;	2000.000000000000
	DQ	0c08f400000000000r    ;	-1000.000000000000
	DD	00H
	DQ	00000000000000000r    ;	.0000000000000000
	DQ	00000000000000000r    ;	.0000000000000000
	DD	00H
	DW	DGROUP:$SG149
	DW	_tan
	DW	DGROUP:$SG150
	DW	_atan
	DW	01H
	DW	00H
	DD	01H
	DQ	00000000000000000r    ;	.0000000000000000
	DQ	00000000000000000r    ;	.0000000000000000
	DD	00H
	DQ	00000000000000000r    ;	.0000000000000000
	DQ	00000000000000000r    ;	.0000000000000000
	DD	00H
	DW	DGROUP:$SG151
	DW	_asin
	DW	DGROUP:$SG152
	DW	_sin
	DW	01H
	DW	00H
	DD	01H
	DQ	04000000000000000r    ;	2.000000000000000
	DQ	0bff0000000000000r    ;	-1.000000000000000
	DD	00H
	DQ	00000000000000000r    ;	.0000000000000000
	DQ	00000000000000000r    ;	.0000000000000000
	DD	00H
	DW	DGROUP:$SG153
	DW	_square
	DW	DGROUP:$SG154
	DW	_sqrt
	DW	01H
	DW	00H
	DD	01H
	DQ	04065400000000000r    ;	170.0000000000000
	DQ	0c055400000000000r    ;	-85.00000000000000
	DD	04H
	DQ	00000000000000000r    ;	.0000000000000000
	DQ	00000000000000000r    ;	.0000000000000000
	DD	00H
	DW	DGROUP:$SG155
	DW	_exp
	DW	DGROUP:$SG156
	DW	_log
	DW	01H
	DW	00H
	DD	01H
	DQ	04075400000000000r    ;	340.0000000000000
	DQ	0c065400000000000r    ;	-170.0000000000000
	DD	00H
	DQ	00000000000000000r    ;	.0000000000000000
	DQ	00000000000000000r    ;	.0000000000000000
	DD	00H
	DW	DGROUP:$SG157
	DW	_atanh
	DW	DGROUP:$SG158
	DW	_tanh
	DW	01H
	DW	00H
	DD	01H
	DQ	04000000000000000r    ;	2.000000000000000
	DQ	0bff0000000000000r    ;	-1.000000000000000
	DD	00H
	DQ	00000000000000000r    ;	.0000000000000000
	DQ	00000000000000000r    ;	.0000000000000000
	DD	00H
	DW	DGROUP:$SG159
	DW	_sinh
	DW	DGROUP:$SG160
	DW	_asinh
	DW	01H
	DW	00H
	DD	01H
	DQ	04075400000000000r    ;	340.0000000000000
	DQ	00000000000000000r    ;	.0000000000000000
	DD	00H
	DQ	00000000000000000r    ;	.0000000000000000
	DQ	00000000000000000r    ;	.0000000000000000
	DD	00H
	DW	DGROUP:$SG161
	DW	_cosh
	DW	DGROUP:$SG162
	DW	_acosh
	DW	01H
	DW	00H
	DD	01H
	DQ	04075400000000000r    ;	340.0000000000000
	DQ	00000000000000000r    ;	.0000000000000000
	DD	00H
	DQ	00000000000000000r    ;	.0000000000000000
	DQ	00000000000000000r    ;	.0000000000000000
	DD	00H
	DW	DGROUP:$SG163
	DW	_exp2
	DW	DGROUP:$SG164
	DW	_log2
	DW	01H
	DW	00H
	DD	01H
	DQ	04075400000000000r    ;	340.0000000000000
	DQ	0c065400000000000r    ;	-170.0000000000000
	DD	00H
	DQ	00000000000000000r    ;	.0000000000000000
	DQ	00000000000000000r    ;	.0000000000000000
	DD	00H
	DW	DGROUP:$SG165
	DW	_exp10
	DW	DGROUP:$SG166
	DW	_log10
	DW	01H
	DW	00H
	DD	01H
	DQ	04075400000000000r    ;	340.0000000000000
	DQ	0c065400000000000r    ;	-170.0000000000000
	DD	00H
	DQ	00000000000000000r    ;	.0000000000000000
	DQ	00000000000000000r    ;	.0000000000000000
	DD	00H
	DW	DGROUP:$SG167
	DW	_pow
	DW	DGROUP:$SG168
	DW	_pow
	DW	02H
	DW	01H
	DD	01H
	DQ	04035000000000000r    ;	21.00000000000000
	DQ	00000000000000000r    ;	.0000000000000000
	DD	00H
	DQ	04045000000000000r    ;	42.00000000000000
	DQ	0c035000000000000r    ;	-21.00000000000000
	DD	00H
	DW	DGROUP:$SG169
	DW	_ellpe
	DW	DGROUP:$SG170
	DW	_ellpk
	DW	01H
	DW	02H
	DD	01H
	DQ	03ff0000000000000r    ;	1.000000000000000
	DQ	00000000000000000r    ;	.0000000000000000
	DD	00H
	DQ	00000000000000000r    ;	.0000000000000000
	DQ	00000000000000000r    ;	.0000000000000000
	DD	00H
	DW	DGROUP:$SG171
	DW	_ndtr
	DW	DGROUP:$SG172
	DW	_ndtri
	DW	01H
	DW	00H
	DD	01H
	DQ	04024000000000000r    ;	10.00000000000000
	DQ	0c024000000000000r    ;	-10.00000000000000
	DD	00H
	DQ	00000000000000000r    ;	.0000000000000000
	DQ	00000000000000000r    ;	.0000000000000000
	DD	00H
	DW	DGROUP:$SG173
	DW	_gamma
	DW	DGROUP:$SG174
	DW	_lgam
	DW	01H
	DW	03H
	DD	00H
	DQ	04026000000000000r    ;	11.00000000000000
	DQ	03ff0000000000000r    ;	1.000000000000000
	DD	00H
	DQ	00000000000000000r    ;	.0000000000000000
	DQ	00000000000000000r    ;	.0000000000000000
	DD	00H
	DW	DGROUP:$SG175
	DW	_acos
	DW	DGROUP:$SG176
	DW	_cos
	DW	01H
	DW	00H
	DD	00H
	DQ	04000000000000000r    ;	2.000000000000000
	DQ	0bff0000000000000r    ;	-1.000000000000000
	DD	00H
	DQ	00000000000000000r    ;	.0000000000000000
	DQ	00000000000000000r    ;	.0000000000000000
	DD	00H
	DW	DGROUP:$SG177
	DW	_jn
	DW	DGROUP:$SG178
	DW	_yn
	DW	02H
	DW	04H
	DD	00H
	DQ	0403e000000000000r    ;	30.00000000000000
	DQ	03fb999999999999ar    ;	.1000000000000000
	DD	00H
	DQ	04044000000000000r    ;	40.00000000000000
	DQ	0c034000000000000r    ;	-20.00000000000000
	DD	02H
	DW	DGROUP:$SG179
	DW	_iv
	DW	DGROUP:$SG180
	DW	_kn
	DW	02H
	DW	05H
	DD	00H
	DQ	0403e000000000000r    ;	30.00000000000000
	DQ	03fb999999999999ar    ;	.1000000000000000
	DD	00H
	DQ	04034000000000000r    ;	20.00000000000000
	DQ	00000000000000000r    ;	.0000000000000000
	DD	02H
$S181_headrs	DW	DGROUP:$SG182
	DW	DGROUP:$SG183
	DW	DGROUP:$SG184
	DW	DGROUP:$SG185
	DW	DGROUP:$SG186
	DW	DGROUP:$SG187
	DW	DGROUP:$SG188
$S189_y1	DQ	00000000000000000r    ;	.0000000000000000
$S190_y2	DQ	00000000000000000r    ;	.0000000000000000
$S191_y3	DQ	00000000000000000r    ;	.0000000000000000
$S192_y4	DQ	00000000000000000r    ;	.0000000000000000
$S193_a	DQ	00000000000000000r    ;	.0000000000000000
$S194_x	DQ	00000000000000000r    ;	.0000000000000000
$S195_y	DQ	00000000000000000r    ;	.0000000000000000
$S196_z	DQ	00000000000000000r    ;	.0000000000000000
$S197_e	DQ	00000000000000000r    ;	.0000000000000000
$S198_max	DQ	00000000000000000r    ;	.0000000000000000
$S199_rmsa	DQ	00000000000000000r    ;	.0000000000000000
$S200_rms	DQ	00000000000000000r    ;	.0000000000000000
$S201_ave	DQ	00000000000000000r    ;	.0000000000000000
_DATA      ENDS
_TEXT      SEGMENT
	ASSUME	CS: _TEXT
; Line 1
; Line 37
; Line 52
	PUBLIC	_square
_square	PROC NEAR
	push	bp
	mov	bp,sp
	xor	ax,ax
	call	__aNchkstk
;	x = 4
; Line 54
	fld	QWORD PTR [bp+4]	;x
	fmul	QWORD PTR [bp+4]	;x
	fstp	QWORD PTR __fac
	fwait	
	mov	ax,OFFSET __fac
; Line 55
	mov	sp,bp
	pop	bp
	ret	

_square	ENDP
; Line 58
	PUBLIC	_cube
_cube	PROC NEAR
	push	bp
	mov	bp,sp
	xor	ax,ax
	call	__aNchkstk
;	x = 4
; Line 60
	fld	QWORD PTR [bp+4]	;x
	fmul	QWORD PTR [bp+4]	;x
	fmul	QWORD PTR [bp+4]	;x
	fstp	QWORD PTR __fac
	fwait	
	mov	ax,OFFSET __fac
; Line 61
	mov	sp,bp
	pop	bp
	ret	

_cube	ENDP
_TEXT      ENDS
CONST      SEGMENT
$T20002	DQ	0bfe0000000000000r    ;	-.5000000000000000
$T20003	DQ	04000000000000000r    ;	2.000000000000000
$T20004	DQ	00000000000000000r    ;	.0000000000000000
CONST      ENDS
_TEXT      SEGMENT
	ASSUME	CS: _TEXT
; Line 182
_TEXT      ENDS
CONST      SEGMENT
$T20005	DQ	03ff0000000000000r    ;	1.000000000000000
$T20006	DQ	0bd3c25c268497682r    ;	-1.000000000000000E-13
$T20007	DQ	03fd0000000000000r    ;	.2500000000000000
$T20009	DQ	03d3c25c268497682r    ;	1.000000000000000E-13
$T20010	DQ	03ddb7cdfd9d7bdbbr    ;	1.000000000000000E-10
$T20011	DQ	04341c37937e08000r    ;	1.000000000000000E+16
$T20012	DQ	03c9cd2b297d889bcr    ;	1.000000000000000E-16
CONST      ENDS
_TEXT      SEGMENT
	ASSUME	CS: _TEXT
	PUBLIC	_main
_main	PROC NEAR
	push	bp
	mov	bp,sp
	mov	ax,32
	call	__aNchkstk
	push	di
	push	si
;	fun = -12
;	ifun = -2
;	d = -8
;	i = -10
;	k = -14
;	itst = -6
;	m = -16
;	ntr = -4
; Line 190
	call	_dprec
; Line 193
	mov	ax,OFFSET DGROUP:$SG214
	push	ax
	call	_printf
	add	sp,2
; Line 195
	mov	ax,10000
	mov	WORD PTR [bp-4],ax	;ntr
	push	ax
	mov	ax,OFFSET DGROUP:$SG215
	push	ax
	call	_printf
	add	sp,4
; Line 199
	fld	QWORD PTR _PI
	fst	QWORD PTR _defs+72
	fmul	QWORD PTR $T20002
	fstp	QWORD PTR _defs+80
	fwait	
; Line 200
	mov	ax,OFFSET DGROUP:_defs+184
	mov	di,ax
	mov	si,OFFSET _MAXLOG
	push	ds
	pop	es
	movsw
	movsw
	movsw
	movsw
; Line 201
	fld	QWORD PTR _MAXLOG
	fmul	QWORD PTR $T20002
	fstp	QWORD PTR _defs+192
; Line 202
	fld	QWORD PTR $T20003
	fmul	QWORD PTR _MAXLOG
	fst	QWORD PTR [bp-24]
	fstp	QWORD PTR _defs+240
; Line 203
	fld	QWORD PTR _MAXLOG
	fchs	
	fst	QWORD PTR [bp-32]
	fstp	QWORD PTR _defs+248
; Line 204
	fld	QWORD PTR [bp-24]
	fstp	QWORD PTR _defs+352
; Line 205
	fld	QWORD PTR [bp-32]
	fstp	QWORD PTR _defs+360
	fwait	
; Line 206
	mov	ax,OFFSET DGROUP:_defs+408
	mov	di,ax
	mov	si,OFFSET _MAXLOG
	movsw
	movsw
	movsw
	movsw
; Line 207
	mov	ax,OFFSET DGROUP:_defs+416
	mov	di,ax
	mov	si,OFFSET DGROUP:$T20004
	movsw
	movsw
	movsw
	movsw
; Line 212
	mov	WORD PTR [bp-6],0	;itst
	jmp	$F216
$FC224:
; Line 240
; Line 241
	inc	WORD PTR [bp-16]	;m
; Line 244
	mov	bx,WORD PTR [bp-8]	;d
	mov	ax,WORD PTR [bx+8]
	dec	ax
	jne	$JCC254
	jmp	$I235
$JCC254:
	dec	ax
	je	$SC232
; Line 245
; Line 248
$SD262:
; Line 343
	push	WORD PTR [bx+8]
	mov	ax,OFFSET DGROUP:$SG263
	push	ax
	call	_printf
	add	sp,4
; Line 344
	mov	ax,1
	push	ax
	call	_exit
	add	sp,2
; Line 345
	jmp	$I259
; Line 250
$SC232:
; Line 251
	mov	ax,OFFSET DGROUP:$S193_a
	push	ax
	call	_drand
	add	sp,2
; Line 252
	fld	QWORD PTR $S193_a
	fsub	QWORD PTR $T20005
	mov	bx,WORD PTR [bp-8]	;d
	fmul	QWORD PTR [bx+36]
	fadd	QWORD PTR [bx+44]
	fstp	QWORD PTR $S193_a
	fwait	
; Line 253
	test	BYTE PTR [bx+52],4
	je	$I234
; Line 254
; Line 255
	push	WORD PTR $S193_a+6
	push	WORD PTR $S193_a+4
	push	WORD PTR $S193_a+2
	push	WORD PTR $S193_a
	call	_exp
	add	sp,8
	mov	dx,OFFSET DGROUP:$S193_a
	mov	di,dx
	mov	si,ax
	push	ds
	pop	es
	movsw
	movsw
	movsw
	movsw
; Line 256
	mov	ax,OFFSET DGROUP:$S190_y2
	push	ax
	call	_drand
	add	sp,2
; Line 257
	fld	QWORD PTR $T20006
	fmul	QWORD PTR $S190_y2
	fadd	QWORD PTR $T20005
	fmul	QWORD PTR $S193_a
	fstp	QWORD PTR $S193_a
	fwait	
; Line 258
; Line 259
$I234:
	mov	bx,WORD PTR [bp-8]	;d
	test	BYTE PTR [bx+52],2
	je	$I235
; Line 260
; Line 262
	fld	QWORD PTR $T20007
	fadd	QWORD PTR $S193_a
	call	__aNftol
	mov	WORD PTR [bp-14],ax	;k
	fild	WORD PTR [bp-14]	;k
	fstp	QWORD PTR $S193_a
	fwait	
; Line 263
; Line 265
$I235:
; Line 266
	mov	ax,OFFSET DGROUP:$S194_x
	push	ax
	call	_drand
	add	sp,2
; Line 267
	fld	QWORD PTR $S194_x
	fsub	QWORD PTR $T20005
	mov	bx,WORD PTR [bp-8]	;d
	fmul	QWORD PTR [bx+16]
	fadd	QWORD PTR [bx+24]
	fstp	QWORD PTR $S194_x
	fwait	
; Line 268
	test	BYTE PTR [bx+32],4
	je	$SB227
; Line 269
; Line 270
	push	WORD PTR $S194_x+6
	push	WORD PTR $S194_x+4
	push	WORD PTR $S194_x+2
	push	WORD PTR $S194_x
	call	_exp
	add	sp,8
	mov	dx,OFFSET DGROUP:$S194_x
	mov	di,dx
	mov	si,ax
	push	ds
	pop	es
	movsw
	movsw
	movsw
	movsw
; Line 271
	mov	ax,OFFSET DGROUP:$S193_a
	push	ax
	call	_drand
	add	sp,2
; Line 272
	fld	QWORD PTR $S193_a
	fmul	QWORD PTR $T20009
	fadd	QWORD PTR $T20005
	fmul	QWORD PTR $S194_x
	fstp	QWORD PTR $S194_x
	fwait	
; Line 273
; Line 274
$SB227:
; Line 278
	mov	bx,WORD PTR [bp-8]	;d
	mov	ax,WORD PTR [bx+8]
	dec	ax
	je	$SC242
	dec	ax
	jne	$JCC566
	jmp	$SC250
$JCC566:
	jmp	$SD262
; Line 279
; Line 280
$SC242:
; Line 281
	mov	ax,WORD PTR [bx+10]
	dec	ax
	dec	ax
	je	$SC247
	dec	ax
	jne	$JCC582
	jmp	$SC248
$JCC582:
; Line 296
	push	WORD PTR $S194_x+6
	push	WORD PTR $S194_x+4
	push	WORD PTR $S194_x+2
	push	WORD PTR $S194_x
	call	WORD PTR [bp-12]	;fun
	add	sp,8
	mov	dx,OFFSET DGROUP:$S196_z
	mov	di,dx
	mov	si,ax
	push	ds
	pop	es
	movsw
	movsw
	movsw
	movsw
; Line 297
	push	WORD PTR $S196_z+6
	push	WORD PTR $S196_z+4
	push	WORD PTR $S196_z+2
	push	WORD PTR $S196_z
	call	WORD PTR [bp-2]	;ifun
	add	sp,8
	jmp	$L20024
; Line 282
	nop	
; Line 283
$SC247:
; Line 284
	push	WORD PTR $S194_x+6
	push	WORD PTR $S194_x+4
	push	WORD PTR $S194_x+2
	push	WORD PTR $S194_x
	call	WORD PTR [bp-12]	;fun
	add	sp,8
	mov	dx,OFFSET DGROUP:$S189_y1
	mov	di,dx
	mov	si,ax
	push	ds
	pop	es
	movsw
	movsw
	movsw
	movsw
; Line 285
	fld1	
	fsub	QWORD PTR $S194_x
	sub	sp,8
	mov	bx,sp
	fstp	QWORD PTR [bx]
	fwait	
	call	WORD PTR [bp-12]	;fun
	add	sp,8
	mov	dx,OFFSET DGROUP:$S190_y2
	mov	di,dx
	mov	si,ax
	push	ds
	pop	es
	movsw
	movsw
	movsw
	movsw
; Line 286
	push	WORD PTR $S194_x+6
	push	WORD PTR $S194_x+4
	push	WORD PTR $S194_x+2
	push	WORD PTR $S194_x
	call	WORD PTR [bp-2]	;ifun
	add	sp,8
	mov	dx,OFFSET DGROUP:$S191_y3
	mov	di,dx
	mov	si,ax
	push	ds
	pop	es
	movsw
	movsw
	movsw
	movsw
; Line 287
	fld1	
	fsub	QWORD PTR $S194_x
	sub	sp,8
	mov	bx,sp
	fstp	QWORD PTR [bx]
	fwait	
	call	WORD PTR [bp-2]	;ifun
	add	sp,8
	jmp	$L20021
; Line 290
$SC248:
; Line 291
	push	WORD PTR $S194_x+6
	push	WORD PTR $S194_x+4
	push	WORD PTR $S194_x+2
	push	WORD PTR $S194_x
	call	_lgam
	add	sp,8
	mov	dx,OFFSET DGROUP:$S195_y
	mov	di,dx
	mov	si,ax
	push	ds
	pop	es
	movsw
	movsw
	movsw
	movsw
; Line 292
	push	WORD PTR $S194_x+6
	push	WORD PTR $S194_x+4
	push	WORD PTR $S194_x+2
	push	WORD PTR $S194_x
	call	_gamma
	add	sp,8
	mov	bx,ax
	push	WORD PTR [bx+6]
	push	WORD PTR [bx+4]
	push	WORD PTR [bx+2]
	push	WORD PTR [bx]
	call	_log
	add	sp,8
	mov	dx,OFFSET DGROUP:$S194_x
	jmp	$L20020
; Line 301
$SC250:
; Line 302
	test	BYTE PTR [bx+52],2
	jne	$JCC868
	jmp	$I251
$JCC868:
; Line 303
; Line 304
	mov	ax,WORD PTR [bx+10]
	sub	ax,4
	je	$SC256
	dec	ax
	jne	$JCC882
	jmp	$SC257
$JCC882:
; Line 321
	push	WORD PTR $S194_x+6
	push	WORD PTR $S194_x+4
	push	WORD PTR $S194_x+2
	push	WORD PTR $S194_x
	push	WORD PTR [bp-14]	;k
	call	WORD PTR [bp-12]	;fun
	add	sp,10
	mov	dx,OFFSET DGROUP:$S196_z
	mov	di,dx
	mov	si,ax
	push	ds
	pop	es
	movsw
	movsw
	movsw
	movsw
; Line 322
	push	WORD PTR $S196_z+6
	push	WORD PTR $S196_z+4
	push	WORD PTR $S196_z+2
	push	WORD PTR $S196_z
	push	WORD PTR [bp-14]	;k
	call	WORD PTR [bp-2]	;ifun
	add	sp,10
	jmp	$L20024
; Line 305
	nop	
; Line 306
$SC256:
; Line 307
	push	WORD PTR $S194_x+6
	push	WORD PTR $S194_x+4
	push	WORD PTR $S194_x+2
	push	WORD PTR $S194_x
	push	WORD PTR [bp-14]	;k
	call	WORD PTR [bp-12]	;fun
	add	sp,10
	mov	dx,OFFSET DGROUP:$S189_y1
	mov	di,dx
	mov	si,ax
	push	ds
	pop	es
	movsw
	movsw
	movsw
	movsw
; Line 308
	push	WORD PTR $S194_x+6
	push	WORD PTR $S194_x+4
	push	WORD PTR $S194_x+2
	push	WORD PTR $S194_x
	mov	ax,WORD PTR [bp-14]	;k
	inc	ax
	push	ax
	mov	si,ax
	call	WORD PTR [bp-12]	;fun
	add	sp,10
	mov	dx,OFFSET DGROUP:$S190_y2
	push	si
	mov	di,dx
	mov	si,ax
	push	ds
	pop	es
	movsw
	movsw
	movsw
	movsw
	pop	si
; Line 309
	push	WORD PTR $S194_x+6
	push	WORD PTR $S194_x+4
	push	WORD PTR $S194_x+2
	push	WORD PTR $S194_x
	push	WORD PTR [bp-14]	;k
	call	WORD PTR [bp-2]	;ifun
	add	sp,10
	mov	dx,OFFSET DGROUP:$S191_y3
	push	si
	mov	di,dx
	mov	si,ax
	push	ds
	pop	es
	movsw
	movsw
	movsw
	movsw
	pop	si
; Line 310
	push	WORD PTR $S194_x+6
	push	WORD PTR $S194_x+4
	push	WORD PTR $S194_x+2
	push	WORD PTR $S194_x
	push	si
$L20023:
	call	WORD PTR [bp-2]	;ifun
	add	sp,10
$L20021:
	mov	dx,OFFSET DGROUP:$S192_y4
	jmp	$L20020
	nop	
; Line 313
$SC257:
; Line 314
	push	WORD PTR $S194_x+6
	push	WORD PTR $S194_x+4
	push	WORD PTR $S194_x+2
	push	WORD PTR $S194_x
	push	WORD PTR $S193_a+6
	push	WORD PTR $S193_a+4
	push	WORD PTR $S193_a+2
	push	WORD PTR $S193_a
	call	WORD PTR [bp-12]	;fun
	add	sp,16
	mov	dx,OFFSET DGROUP:$S189_y1
	mov	di,dx
	mov	si,ax
	push	ds
	pop	es
	movsw
	movsw
	movsw
	movsw
; Line 315
	push	WORD PTR $S194_x+6
	push	WORD PTR $S194_x+4
	push	WORD PTR $S194_x+2
	push	WORD PTR $S194_x
	fld1	
	fadd	QWORD PTR $S193_a
	sub	sp,8
	mov	bx,sp
	fstp	QWORD PTR [bx]
	fwait	
	call	WORD PTR [bp-12]	;fun
	add	sp,16
	mov	dx,OFFSET DGROUP:$S190_y2
	mov	di,dx
	mov	si,ax
	push	ds
	pop	es
	movsw
	movsw
	movsw
	movsw
; Line 316
	push	WORD PTR $S194_x+6
	push	WORD PTR $S194_x+4
	push	WORD PTR $S194_x+2
	push	WORD PTR $S194_x
	push	WORD PTR [bp-14]	;k
	call	WORD PTR [bp-2]	;ifun
	add	sp,10
	mov	dx,OFFSET DGROUP:$S191_y3
	mov	di,dx
	mov	si,ax
	push	ds
	pop	es
	movsw
	movsw
	movsw
	movsw
; Line 317
	push	WORD PTR $S194_x+6
	push	WORD PTR $S194_x+4
	push	WORD PTR $S194_x+2
	push	WORD PTR $S194_x
	mov	ax,WORD PTR [bp-14]	;k
	inc	ax
	push	ax
	jmp	$L20023
; Line 320
$I251:
; Line 326
; Line 327
	cmp	WORD PTR [bx+10],1
	jne	$I260
; Line 328
; Line 329
	push	WORD PTR $S193_a+6
	push	WORD PTR $S193_a+4
	push	WORD PTR $S193_a+2
	push	WORD PTR $S193_a
	push	WORD PTR $S194_x+6
	push	WORD PTR $S194_x+4
	push	WORD PTR $S194_x+2
	push	WORD PTR $S194_x
	call	WORD PTR [bp-12]	;fun
	add	sp,16
	mov	dx,OFFSET DGROUP:$S196_z
	mov	di,dx
	mov	si,ax
	push	ds
	pop	es
	movsw
	movsw
	movsw
	movsw
; Line 330
	fld	QWORD PTR $S193_a
	fld1	
	fdivr	
	sub	sp,8
	mov	bx,sp
	fstp	QWORD PTR [bx]
	fwait	
	push	WORD PTR $S196_z+6
	push	WORD PTR $S196_z+4
	push	WORD PTR $S196_z+2
	push	WORD PTR $S196_z
	jmp	SHORT $L20016
$I260:
; Line 333
; Line 334
	push	WORD PTR $S194_x+6
	push	WORD PTR $S194_x+4
	push	WORD PTR $S194_x+2
	push	WORD PTR $S194_x
	push	WORD PTR $S193_a+6
	push	WORD PTR $S193_a+4
	push	WORD PTR $S193_a+2
	push	WORD PTR $S193_a
	call	WORD PTR [bp-12]	;fun
	add	sp,16
	mov	dx,OFFSET DGROUP:$S196_z
	mov	di,dx
	mov	si,ax
	push	ds
	pop	es
	movsw
	movsw
	movsw
	movsw
; Line 335
	push	WORD PTR $S196_z+6
	push	WORD PTR $S196_z+4
	push	WORD PTR $S196_z+2
	push	WORD PTR $S196_z
	push	WORD PTR $S193_a+6
	push	WORD PTR $S193_a+4
	push	WORD PTR $S193_a+2
	push	WORD PTR $S193_a
$L20016:
	call	WORD PTR [bp-2]	;ifun
	add	sp,16
$L20024:
	mov	dx,OFFSET DGROUP:$S195_y
$L20020:
	mov	di,dx
	mov	si,ax
	push	ds
	pop	es
	movsw
	movsw
	movsw
	movsw
; Line 336
; Line 337
$I259:
; Line 347
	mov	bx,WORD PTR [bp-8]	;d
	mov	ax,WORD PTR [bx+10]
	dec	ax
	dec	ax
	jne	$JCC1478
	jmp	$SC271
$JCC1478:
	dec	ax
	dec	ax
	je	$SC269
	dec	ax
	je	$SC270
; Line 362
	fld	QWORD PTR $S195_y
	fsub	QWORD PTR $S194_x
	jmp	SHORT $L20018
; Line 348
; Line 349
$SC269:
; Line 350
	fld	QWORD PTR $S191_y3
	fmul	QWORD PTR $S190_y2
	fld	QWORD PTR $S192_y4
	fmul	QWORD PTR $S189_y1
	fsub	
	fld	QWORD PTR _PI
	fmul	QWORD PTR $S194_x
	fdivr	QWORD PTR $T20003
$L20017:
	fsub	
$L20018:
	fstp	QWORD PTR $S197_e
	fwait	
; Line 366
	mov	bx,WORD PTR [bp-8]	;d
	test	BYTE PTR [bx+12],1
	je	$I273
; Line 367
	fld	QWORD PTR $S194_x
	fdivr	QWORD PTR $S197_e
	fstp	QWORD PTR $S197_e
	fwait	
; Line 368
	jmp	$I275
	nop	
; Line 353
$SC270:
; Line 354
	fld	QWORD PTR $S191_y3
	fmul	QWORD PTR $S190_y2
	fld	QWORD PTR $S192_y4
	fmul	QWORD PTR $S189_y1
	fadd	
	fld	QWORD PTR $S194_x
	fld1	
	fdivr	
	jmp	SHORT $L20017
; Line 357
$SC271:
; Line 358
	fld	QWORD PTR $S189_y1
	fsub	QWORD PTR $S191_y3
	fmul	QWORD PTR $S192_y4
	fld	QWORD PTR $S191_y3
	fmul	QWORD PTR $S190_y2
	fadd	
	fsub	QWORD PTR _PIO2
	jmp	SHORT $L20018
; Line 361
	nop	
$I273:
; Line 369
; Line 370
	push	WORD PTR $S194_x+6
	push	WORD PTR $S194_x+4
	push	WORD PTR $S194_x+2
	push	WORD PTR $S194_x
	call	_fabs
	add	sp,8
	mov	bx,ax
	fld	QWORD PTR [bx]
	fcom	QWORD PTR $T20005
	fstp	ST(0)
	fstsw	WORD PTR [bp-32]
	fwait	
	mov	ah,BYTE PTR [bp-31]
	sahf	
	jbe	$I275
; Line 371
	fld	QWORD PTR $S194_x
	fdivr	QWORD PTR $S197_e
	fstp	QWORD PTR $S197_e
; Line 372
$I275:
; Line 374
	fld	QWORD PTR $S197_e
	fadd	QWORD PTR $S201_ave
	fstp	QWORD PTR $S201_ave
; Line 376
	fld	QWORD PTR $S197_e
	fcom	QWORD PTR $T20004
	fstp	ST(0)
	fstsw	WORD PTR [bp-32]
	fwait	
	mov	ah,BYTE PTR [bp-31]
	sahf	
	jae	$I276
; Line 377
	fld	QWORD PTR $S197_e
	fchs	
	fstp	QWORD PTR $S197_e
; Line 380
$I276:
	fld	QWORD PTR $S198_max
	fcom	QWORD PTR $S197_e
	fstp	ST(0)
	fstsw	WORD PTR [bp-32]
	fwait	
	mov	ah,BYTE PTR [bp-31]
	sahf	
	jb	$JCC1790
	jmp	$I282
$JCC1790:
; Line 381
; Line 382
	mov	ax,OFFSET DGROUP:$S198_max
	mov	di,ax
	mov	si,OFFSET DGROUP:$S197_e
	push	ds
	pop	es
	movsw
	movsw
	movsw
	movsw
; Line 384
	fld	QWORD PTR $S197_e
	fcom	QWORD PTR $T20010
	fstp	ST(0)
	fstsw	WORD PTR [bp-32]
	fwait	
	mov	ah,BYTE PTR [bp-31]
	sahf	
	ja	$JCC1832
	jmp	$I282
$JCC1832:
; Line 385
; Line 387
	push	WORD PTR $S197_e+6
	push	WORD PTR $S197_e+4
	push	WORD PTR $S197_e+2
	push	WORD PTR $S197_e
	push	WORD PTR $S195_y+6
	push	WORD PTR $S195_y+4
	push	WORD PTR $S195_y+2
	push	WORD PTR $S195_y
	push	WORD PTR $S196_z+6
	push	WORD PTR $S196_z+4
	push	WORD PTR $S196_z+2
	push	WORD PTR $S196_z
	push	WORD PTR $S194_x+6
	push	WORD PTR $S194_x+4
	push	WORD PTR $S194_x+2
	push	WORD PTR $S194_x
	mov	ax,OFFSET DGROUP:$SG279
	push	ax
	call	_printf
	add	sp,34
; Line 388
	mov	bx,WORD PTR [bp-8]	;d
	cmp	WORD PTR [bx+10],1
	jne	$I280
; Line 389
; Line 390
	push	WORD PTR $S193_a+6
	push	WORD PTR $S193_a+4
	push	WORD PTR $S193_a+2
	push	WORD PTR $S193_a
	mov	ax,OFFSET DGROUP:$SG281
	push	ax
	call	_printf
	add	sp,10
; Line 391
; Line 392
$I280:
	mov	bx,WORD PTR [bp-8]	;d
	cmp	WORD PTR [bx+10],4
	jl	$I282
; Line 393
; Line 395
	push	WORD PTR $S194_x+6
	push	WORD PTR $S194_x+4
	push	WORD PTR $S194_x+2
	push	WORD PTR $S194_x
	push	WORD PTR [bp-14]	;k
	push	WORD PTR $S192_y4+6
	push	WORD PTR $S192_y4+4
	push	WORD PTR $S192_y4+2
	push	WORD PTR $S192_y4
	push	WORD PTR $S191_y3+6
	push	WORD PTR $S191_y3+4
	push	WORD PTR $S191_y3+2
	push	WORD PTR $S191_y3
	push	WORD PTR $S190_y2+6
	push	WORD PTR $S190_y2+4
	push	WORD PTR $S190_y2+2
	push	WORD PTR $S190_y2
	push	WORD PTR $S189_y1+6
	push	WORD PTR $S189_y1+4
	push	WORD PTR $S189_y1+2
	push	WORD PTR $S189_y1
	mov	ax,OFFSET DGROUP:$SG283
	push	ax
	call	_printf
	add	sp,44
; Line 396
; Line 397
$I282:
; Line 407
; Line 410
	fld	QWORD PTR $T20011
	fmul	QWORD PTR $S197_e
	fst	QWORD PTR $S197_e
; Line 411
	fmul	QWORD PTR $S197_e
	fadd	QWORD PTR $S199_rmsa
	fstp	QWORD PTR $S199_rmsa
	fwait	
; Line 412
	inc	WORD PTR [bp-10]	;i
$F223:
	mov	ax,WORD PTR [bp-10]	;i
	cmp	WORD PTR [bp-4],ax	;ntr
	jle	$JCC2089
	jmp	$FC224
$JCC2089:
; Line 415
	fld	QWORD PTR $S199_rmsa
	fidiv	WORD PTR [bp-16]	;m
	sub	sp,8
	mov	bx,sp
	fstp	QWORD PTR [bx]
	fwait	
	call	_sqrt
	add	sp,8
	mov	bx,ax
	fld	QWORD PTR [bx]
	fmul	QWORD PTR $T20012
	fstp	QWORD PTR $S200_rms
	fwait	
; Line 416
	push	WORD PTR $S200_rms+6
	push	WORD PTR $S200_rms+4
	push	WORD PTR $S200_rms+2
	push	WORD PTR $S200_rms
	push	WORD PTR $S198_max+6
	push	WORD PTR $S198_max+4
	push	WORD PTR $S198_max+2
	push	WORD PTR $S198_max
	mov	ax,OFFSET DGROUP:$SG284
	push	ax
	call	_printf
	add	sp,18
; Line 417
	inc	WORD PTR [bp-6]	;itst
$F216:
	cmp	WORD PTR [bp-6],17	;itst
	jl	$JCC2185
	jmp	$FB218
$JCC2185:
; Line 213
; Line 215
	mov	WORD PTR [bp-16],0	;m
; Line 216
	mov	ax,OFFSET DGROUP:$S198_max
	mov	di,ax
	mov	si,OFFSET DGROUP:$T20004
	push	ds
	pop	es
	movsw
	movsw
	movsw
	movsw
; Line 217
	mov	ax,OFFSET DGROUP:$S199_rmsa
	mov	di,ax
	mov	si,OFFSET DGROUP:$T20004
	movsw
	movsw
	movsw
	movsw
; Line 218
	mov	ax,OFFSET DGROUP:$S201_ave
	mov	di,ax
	mov	si,OFFSET DGROUP:$T20004
	movsw
	movsw
	movsw
	movsw
; Line 219
	mov	ax,56
	imul	WORD PTR [bp-6]	;itst
	mov	bx,ax
	add	bx,OFFSET DGROUP:_defs
	mov	WORD PTR [bp-8],bx	;d
	mov	ax,WORD PTR [bx+2]
	mov	WORD PTR [bp-12],ax	;fun
; Line 220
	mov	ax,WORD PTR [bx+6]
	mov	WORD PTR [bp-2],ax	;ifun
; Line 225
	cmp	WORD PTR [bx+10],3
	jne	$I219
; Line 226
	mov	ax,OFFSET DGROUP:$SG220
	push	ax
	call	_printf
	add	sp,2
; Line 231
$I219:
	mov	bx,WORD PTR [bp-8]	;d
	cmp	WORD PTR [bx+10],4
	jne	$I221
; Line 232
; Line 234
	mov	ax,2000
	mov	WORD PTR [bp-4],ax	;ntr
	push	ax
	mov	ax,OFFSET DGROUP:$SG222
	push	ax
	call	_printf
	add	sp,4
; Line 235
; Line 237
$I221:
	mov	bx,WORD PTR [bp-8]	;d
	push	WORD PTR [bx]
	push	WORD PTR [bx+4]
	mov	bx,WORD PTR [bx+10]
	shl	bx,1
	push	WORD PTR $S181_headrs[bx]
	call	_printf
	add	sp,6
; Line 239
	mov	WORD PTR [bp-10],0	;i
	jmp	$F223
	nop	
$FB218:
; Line 419
	pop	si
	pop	di
	mov	sp,bp
	pop	bp
	ret	

_main	ENDP
_TEXT	ENDS
END
