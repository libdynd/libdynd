;	Static Name Aliases
;
;	$S167_bmask	EQU	bmask
	TITLE   floorx.c
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
EXTRN	_MAXNUM:QWORD
EXTRN	__fac:QWORD
EXTRN	__fltused:NEAR
_DATA      SEGMENT
$S167_bmask	DW	0ffffH
	DW	0fffeH
	DW	0fffcH
	DW	0fff8H
	DW	0fff0H
	DW	0ffe0H
	DW	0ffc0H
	DW	0ff80H
	DW	0ff00H
	DW	0fe00H
	DW	0fc00H
	DW	0f800H
	DW	0f000H
	DW	0e000H
	DW	0c000H
	DW	08000H
	DW	00H
_DATA      ENDS
CONST      SEGMENT
$T20001	DQ	03ff0000000000000r    ;	1.000000000000000
CONST      ENDS
_TEXT      SEGMENT
	ASSUME	CS: _TEXT
; Line 1
; Line 59
; Line 93
	PUBLIC	_ceil
_ceil	PROC NEAR
	push	bp
	mov	bp,sp
	sub	sp,10
	push	di
	push	si
;	x = 4
;	y = -8
; Line 104
	push	WORD PTR [bp+10]
	push	WORD PTR [bp+8]
	push	WORD PTR [bp+6]
	push	WORD PTR [bp+4]	;x
	call	_floor
	add	sp,8
	mov	bx,ax
	fld	QWORD PTR [bx]
	fst	QWORD PTR [bp-8]	;y
	fcom	QWORD PTR [bp+4]	;x
	fstp	ST(0)
	fstsw	WORD PTR [bp-10]
	fwait	
	mov	ah,BYTE PTR [bp-9]
	sahf	
	jae	$I166
; Line 105
	fld1	
	fadd	QWORD PTR [bp-8]	;y
	fstp	QWORD PTR [bp-8]	;y
	fwait	
; Line 106
$I166:
	mov	ax,OFFSET __fac
	mov	di,ax
	lea	si,WORD PTR [bp-8]	;y
	push	ds
	pop	es
	movsw
	movsw
	movsw
	movsw
; Line 107
	pop	si
	pop	di
	mov	sp,bp
	pop	bp
	ret	
	nop	

_ceil	ENDP
_TEXT      ENDS
CONST      SEGMENT
$T20004	DQ	00000000000000000r    ;	.0000000000000000
$T20005	DQ	0bff0000000000000r    ;	-1.000000000000000
CONST      ENDS
_TEXT      SEGMENT
	ASSUME	CS: _TEXT
; Line 138
	PUBLIC	_floor
_floor	PROC NEAR
	push	bp
	mov	bp,sp
	sub	sp,12
	push	di
	push	si
;	x = 4
;	register bx = p
;	y = -10
;	e = -2
	fld	QWORD PTR [bp+4]	;x
; Line 149
	fst	QWORD PTR [bp-10]	;y
	fwait	
; Line 159
	mov	cl,4
	mov	di,WORD PTR [bp-4]
	and	di,32752
	shr	di,cl
	sub	di,1023
; Line 160
	lea	bx,WORD PTR [bp-10]	;y
; Line 169
	or	di,di
	jge	$I173
	fstp	ST(0)
; Line 171
	fldz	
	fcom	QWORD PTR [bp-10]	;y
	fstp	ST(0)
	fstsw	WORD PTR [bp-12]
	fwait	
	mov	ah,BYTE PTR [bp-11]
	sahf	
	jbe	$I174
; Line 172
	mov	ax,OFFSET __fac
	mov	di,ax
	mov	si,OFFSET DGROUP:$T20005
	jmp	$L20026
; Line 173
$I174:
; Line 174
	mov	ax,OFFSET __fac
	mov	di,ax
	mov	si,OFFSET DGROUP:$T20004
	jmp	SHORT $L20026
	nop	
	nop	
; Line 177
$I173:
; Line 179
	mov	ax,52
	sub	ax,di
	mov	di,ax
	cmp	di,16
	jl	$FB178
	mov	cl,4
	shr	ax,cl
	mov	dx,ax
	shl	ax,cl
	sub	di,ax
	mov	WORD PTR [bp-2],di	;e
$FC177:
; Line 182
	inc	bx
	inc	bx
	mov	WORD PTR [bx-2],0
; Line 193
	dec	dx
	jne	$FC177
$FB178:
; Line 196
	or	di,di
	jle	$I179
; Line 197
	shl	di,1
	mov	ax,WORD PTR $S167_bmask[di]
	and	WORD PTR [bx],ax
; Line 199
$I179:
	fldz	
	fcom	ST(1)
	fstp	ST(0)
	fstsw	WORD PTR [bp-12]
	fwait	
	mov	ah,BYTE PTR [bp-11]
	sahf	
	ja	$L20008
	fstp	ST(0)
	jmp	SHORT $I180
	nop	
$L20008:
	fld	ST(0)
	fcom	QWORD PTR [bp-10]	;y
	fstp	ST(0)
	fstsw	WORD PTR [bp-12]
	fwait	
	mov	ah,BYTE PTR [bp-11]
	sahf	
	fstp	ST(0)
	je	$I180
; Line 200
	fld1	
	fsubr	QWORD PTR [bp-10]	;y
	fstp	QWORD PTR [bp-10]	;y
	fwait	
; Line 202
$I180:
	mov	ax,OFFSET __fac
	mov	di,ax
	lea	si,WORD PTR [bp-10]	;y
$L20026:
	push	ds
	pop	es
	movsw
	movsw
	movsw
	movsw
; Line 203
	pop	si
	pop	di
	mov	sp,bp
	pop	bp
	ret	

_floor	ENDP
_TEXT      ENDS
CONST      SEGMENT
$T20010	DQ	04000000000000000r    ;	2.000000000000000
CONST      ENDS
_TEXT      SEGMENT
	ASSUME	CS: _TEXT
; Line 208
	PUBLIC	_frexp
_frexp	PROC NEAR
	push	bp
	mov	bp,sp
	sub	sp,14
	push	di
	push	si
;	x = 4
;	pw2 = 12
;	y = -12
;	i = -4
;	register bx = k
;	q = -2
; Line 215
	lea	di,WORD PTR [bp-12]	;y
	lea	si,WORD PTR [bp+4]	;x
	mov	ax,ss
	mov	es,ax
	movsw
	movsw
	movsw
	movsw
; Line 223
	lea	ax,WORD PTR [bp-6]
	mov	WORD PTR [bp-2],ax	;q
; Line 250
	mov	cl,4
	mov	dx,WORD PTR [bp-6]
	and	dx,32752
	sar	dx,cl
	mov	cx,dx
; Line 251
	or	dx,dx
	jne	$ieeedon190
; Line 273
	fldz	
	fcom	QWORD PTR [bp-12]	;y
	fstp	ST(0)
	fstsw	WORD PTR [bp-14]
	fwait	
	mov	ah,BYTE PTR [bp-13]
	sahf	
	jne	$L20009
; Line 275
	mov	bx,WORD PTR [bp+12]	;pw2
	mov	WORD PTR [bx],dx
; Line 276
	mov	ax,OFFSET __fac
	mov	di,ax
	mov	si,OFFSET DGROUP:$T20004
	jmp	SHORT $L20027
$L20009:
	fld	QWORD PTR $T20010
	fwait	
	mov	WORD PTR [bp-4],dx	;i
	mov	si,dx
; Line 281
$D192:
; Line 283
	fld	ST(0)
	fmul	QWORD PTR [bp-12]	;y
	fstp	QWORD PTR [bp-12]	;y
	fwait	
; Line 284
	dec	si
; Line 285
	mov	bx,WORD PTR [bp-6]
	and	bx,32752
	mov	cl,4
	sar	bx,cl
; Line 287
	or	bx,bx
	je	$D192
	mov	WORD PTR [bp-4],si	;i
	fstp	ST(0)
	mov	dx,si
; Line 288
	add	dx,bx
; Line 291
$ieeedon190:
; Line 294
	sub	dx,1022
	mov	bx,WORD PTR [bp+12]	;pw2
	mov	WORD PTR [bx],dx
; Line 295
	and	WORD PTR [bp-6],-32753
; Line 296
	or	WORD PTR [bp-6],16352
; Line 297
	mov	ax,OFFSET __fac
	mov	di,ax
	lea	si,WORD PTR [bp-12]	;y
$L20027:
	push	ds
	pop	es
	movsw
	movsw
	movsw
	movsw
; Line 299
	pop	si
	pop	di
	mov	sp,bp
	pop	bp
	ret	

_frexp	ENDP
_TEXT      ENDS
CONST      SEGMENT
$T20023	DQ	03fe0000000000000r    ;	.5000000000000000
CONST      ENDS
_TEXT      SEGMENT
	ASSUME	CS: _TEXT
; Line 307
	PUBLIC	_ldexp
_ldexp	PROC NEAR
	push	bp
	mov	bp,sp
	sub	sp,12
	push	di
	push	si
;	x = 4
;	pw2 = 12
;	y = -10
;	q = -2
;	e = -2
; Line 319
	lea	di,WORD PTR [bp-10]	;y
	lea	si,WORD PTR [bp+4]	;x
	mov	ax,ss
	mov	es,ax
	movsw
	movsw
	movsw
	movsw
; Line 328
	lea	ax,WORD PTR [bp-4]
	mov	WORD PTR [bp-2],ax	;q
	fld	QWORD PTR $T20023
	fld	QWORD PTR $T20010
	fldz	
	fwait	
	mov	dx,WORD PTR [bp+12]	;pw2
; Line 333
$FC203:
	mov	cl,4
	mov	si,WORD PTR [bp-4]
	and	si,32752
	sar	si,cl
	or	si,si
	jne	$FB204
; Line 335
	fld	QWORD PTR [bp-10]	;y
	fcom	ST(1)
	fstp	ST(0)
	fstsw	WORD PTR [bp-12]
	fwait	
	mov	ah,BYTE PTR [bp-11]
	sahf	
	je	$L20018
; Line 340
	or	dx,dx
	jle	$I206
; Line 342
	fld	ST(1)
	fmul	QWORD PTR [bp-10]	;y
	fstp	QWORD PTR [bp-10]	;y
	fwait	
; Line 343
	dec	dx
; Line 345
$I206:
	or	dx,dx
	jge	$I207
; Line 347
	cmp	dx,-53
	jl	$L20018
; Line 349
	fld	ST(2)
	fmul	QWORD PTR [bp-10]	;y
	fstp	QWORD PTR [bp-10]	;y
	fwait	
; Line 350
	inc	dx
; Line 352
$I207:
	or	dx,dx
	jne	$FC203
	fstp	ST(0)
	fstp	ST(0)
$L20032:
	fstp	ST(0)
	jmp	$L20031
$L20018:
	fstp	ST(0)
	fstp	ST(0)
	jmp	SHORT $L20029
; Line 354
$FB204:
	fstp	ST(0)
; Line 364
	add	si,dx
	cmp	si,2047
	jl	$I210
; Line 365
	fld	ST(0)
	fmul	QWORD PTR _MAXNUM
	fstp	QWORD PTR __fac
	fwait	
	mov	ax,OFFSET __fac
	fstp	ST(0)
	fstp	ST(0)
	pop	si
	pop	di
	mov	sp,bp
	pop	bp
	ret	
	nop	
; Line 369
$I210:
	fstp	ST(0)
	cmp	si,1
	jge	$L20024
; Line 372
	cmp	si,-53
	jge	$I212
$L20029:
	fstp	ST(0)
; Line 373
	mov	ax,OFFSET __fac
	mov	di,ax
	mov	si,OFFSET DGROUP:$T20004
	jmp	SHORT $L20028
; Line 374
$I212:
	and	WORD PTR [bp-4],-32753
; Line 375
	or	BYTE PTR [bp-4],16
; Line 376
	cmp	si,1
	jge	$L20032
	mov	dx,1
	sub	dx,si
	mov	WORD PTR [bp-2],si	;q
$FC214:
; Line 378
	fld	ST(0)
	fmul	QWORD PTR [bp-10]	;y
	fstp	QWORD PTR [bp-10]	;y
	fwait	
; Line 380
	dec	dx
	jne	$FC214
	jmp	SHORT $L20032
$L20024:
	mov	WORD PTR [bp-2],si	;q
; Line 386
	fstp	ST(0)
; Line 392
	and	WORD PTR [bp-4],-32753
	mov	dx,si
; Line 393
	mov	cl,4
	and	dh,7
	shl	dx,cl
	or	WORD PTR [bp-4],dx
; Line 395
$L20031:
	mov	ax,OFFSET __fac
	mov	di,ax
	lea	si,WORD PTR [bp-10]	;y
$L20028:
	push	ds
	pop	es
	movsw
	movsw
	movsw
	movsw
; Line 397
	pop	si
	pop	di
	mov	sp,bp
	pop	bp
	ret	
	nop	

_ldexp	ENDP
_TEXT	ENDS
END
