/* Math coprocessor precision settings
   This version works with Microsoft Visual C++ version 6. */

int
dprec()
{
  __asm
    {
	push eax
	sub esp,4
	fstcw [esp]
	fwait
	mov eax,[esp]
	and eax,0xfcf7
	or eax,0x200
	mov [esp],eax
	fldcw [esp]
	pop eax
	pop eax
    }
  return 0;
}

int
sprec()
{
  __asm
    {
	push eax
	sub esp,4
	fstcw [esp]
	fwait
	mov eax,[esp]
	and eax,0xfcff
	mov [esp],eax
	fldcw [esp]
	pop eax
	pop eax
    }
return 0;
}

int
ldprec()
{
  __asm
    {
	push eax
	sub esp,4
	fstcw [esp]
	fwait
	mov eax,[esp]
	or eax,0x300
	mov [esp],eax
	fldcw [esp]
	pop eax
	pop eax
    }
  return 0;
}
