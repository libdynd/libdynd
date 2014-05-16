; This subroutine sets the 68882 coprocessor
; for true IEEE double precision.  In this mode
; the PARANOIA test program returns a rating of "excellent!"
; when the Cephes library pow() function is used in place
; of the Silicon Valley one.
;
; Note, this is not a complete fix for the Silicon Valley
; run time library, since at least some of the routines in it
; depend on the coprocessor being set to extended precision.
; Each of these routines would have to be modified to change the
; precision and then set it back to the default value.
;
; Reference: MC68881/MC68882 Floating-Point Coprocessor
; User's Manual, Motorola, Prentice-Hall, 1987 (First Edition)
; Pages 1-14, 2-3, 4-68.
;
; -- Steve Moshier


; FPcr code $80 sets the 68882 coprocessor to
;    rounding precision = 53 bits
;    rounding mode = nearest or even

	GLOBAL einit
einit
;	FMOVE.L	#$80,Fcr
; The dumb assembler doesn't seem to know this instruction,
; so it is written out here in binary:
	DATA.W	$f23c,$9000,$0000,$0080 ; 53 bit rounding
;	DATA.W	$f23c,$9000,$0000,$0040	; 24 bit rounding
	RTS

; set to single precision
	GLOBAL einits
einits
	DATA.W	$f23c,$9000,$0000,$0040	; 24 bit rounding
	RTS

; set to double precision
	GLOBAL einitd
einitd
	DATA.W	$f23c,$9000,$0000,$0080 ; 53 bit rounding
	RTS

	END

