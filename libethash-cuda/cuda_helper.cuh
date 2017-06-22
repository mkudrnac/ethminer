#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <cuda.h>
#include <cuda_runtime.h>

//MARK: swab64
// Input:       77665544 33221100
// Output:      00112233 44556677
__device__ __forceinline__
uint64_t cuda_swab64(uint64_t x)
{
	uint64_t result;
	uint2 t;
	asm("mov.b64 {%0,%1},%2; \n\t"
		: "=r"(t.x), "=r"(t.y) : "l"(x));
	t.x = __byte_perm(t.x, 0, 0x0123);
	t.y = __byte_perm(t.y, 0, 0x0123);
	asm("mov.b64 %0,{%1,%2}; \n\t"
		: "=l"(result) : "r"(t.y), "r"(t.x));
	return result;
}

//MARK: uint2 ROTATE LEFT
__device__ __forceinline__
uint2 ROL2(const uint2 a, const int offset)
{
    uint2 result;
    if(offset >= 32)
    {        
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.x), "r"(a.y), "r"(offset));
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));
    }
    else
    {
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.y), "r"(a.x), "r"(offset));
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
    }
    return result;
}

//MARK: vectorize/devectorize
__device__ __forceinline__
uint64_t devectorize(const uint2 x)
{
	uint64_t result;
	asm("mov.b64 %0,{%1,%2}; \n\t"
		: "=l"(result) : "r"(x.x), "r"(x.y));
	return result;
}

__device__ __forceinline__
uint2 vectorize(const uint64_t x)
{
	uint2 result;
	asm("mov.b64 {%0,%1},%2; \n\t"
		: "=r"(result.x), "=r"(result.y) : "l"(x));
	return result;
}

__device__ __forceinline__
void devectorize2(const uint4 inn, uint2 &x, uint2 &y)
{
	x.x = inn.x;
	x.y = inn.y;
	y.x = inn.z;
	y.y = inn.w;
}

__device__ __forceinline__
uint4 vectorize2(const uint2 x, const uint2 y)
{
	uint4 result;
	result.x = x.x;
	result.y = x.y;
	result.z = y.x;
	result.w = y.y;
	return result;
}

__device__ __forceinline__
uint4 vectorize2(const uint2 x)
{
	uint4 result;
	result.x = x.x;
	result.y = x.y;
	result.z = x.x;
	result.w = x.y;
	return result;
}

__device__ __forceinline__
void devectorize4(const uint4 inn, uint64_t &x, uint64_t &y)
{
	asm("mov.b64 %0,{%1,%2}; \n\t"
		: "=l"(x) : "r"(inn.x), "r"(inn.y));
	asm("mov.b64 %0,{%1,%2}; \n\t"
		: "=l"(y) : "r"(inn.z), "r"(inn.w));
}

//MARK: uint2 operators
__device__ __forceinline__
uint2 operator^ (uint2 a, uint32_t b)
{
    return make_uint2(a.x ^ b, a.y);
}

__device__ __forceinline__
uint2 operator^ (uint2 a, uint2 b)
{
    return make_uint2(a.x ^ b.x, a.y ^ b.y);
}

__device__ __forceinline__
uint2 operator& (uint2 a, uint2 b)
{
    return make_uint2(a.x & b.x, a.y & b.y);
}

__device__ __forceinline__
uint2 operator| (uint2 a, uint2 b)
{
    return make_uint2(a.x | b.x, a.y | b.y);
}

__device__ __forceinline__
uint2 operator~ (uint2 a)
{
    return make_uint2(~a.x, ~a.y);
}

__device__ __forceinline__
void operator^= (uint2 &a, uint2 b)
{
    a = a ^ b;
}

__device__ __forceinline__
uint2 operator+ (uint2 a, uint2 b)
{
	uint2 result;
	asm("{\n\t"
		"add.cc.u32 %0,%2,%4; \n\t"
		"addc.u32 %1,%3,%5;   \n\t"
	"}\n\t"
		: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
	return result;
}

__device__ __forceinline__
uint2 operator+ (uint2 a, uint32_t b)
{
	uint2 result;
	asm("{\n\t"
		"add.cc.u32 %0,%2,%4; \n\t"
		"addc.u32 %1,%3,%5;   \n\t"
		"}\n\t"
		: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(b), "r"(0));
	return result;
}

__device__ __forceinline__
uint2 operator- (uint2 a, uint32_t b)
{
	uint2 result;
	asm("{\n\t"
		"sub.cc.u32 %0,%2,%4; \n\t"
		"subc.u32 %1,%3,%5;   \n\t"
		"}\n\t"
		: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(b), "r"(0));
	return result;
}

__device__ __forceinline__
uint2 operator- (uint2 a, uint2 b)
{
	uint2 result;
	asm("{\n\t"
		"sub.cc.u32 %0,%2,%4; \n\t"
		"subc.u32 %1,%3,%5;   \n\t"
		"}\n\t"
		: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
	return result;
}

__device__ __forceinline__
void operator+= (uint2 &a, uint2 b)
{
    a = a + b;
}

/**
 * basic multiplication between 64bit no carry outside that range (ie mul.lo.b64(a*b))
 * (what does uint64 "*" operator)
 */
__device__ __forceinline__
uint2 operator* (uint2 a, uint2 b)
{
	uint2 result;
	asm("{\n\t"
		"mul.lo.u32        %0,%2,%4;  \n\t"
		"mul.hi.u32        %1,%2,%4;  \n\t"
		"mad.lo.cc.u32    %1,%3,%4,%1; \n\t"
		"madc.lo.u32      %1,%3,%5,%1; \n\t"
	"}\n\t"
		: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
	return result;
}

//MARK: uint4 operators
__device__ __forceinline__
uint4 operator^ (uint4 a, uint4 b)
{
    return make_uint4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w);
}

__device__ __forceinline__
uint4 operator& (uint4 a, uint4 b)
{
    return make_uint4(a.x & b.x, a.y & b.y, a.z & b.z, a.w & b.w);
}

__device__ __forceinline__
uint4 operator| (uint4 a, uint4 b)
{
    return make_uint4(a.x | b.x, a.y | b.y, a.z | b.z, a.w | b.w);
}

__device__ __forceinline__
uint4 operator~ (uint4 a)
{
    return make_uint4(~a.x, ~a.y, ~a.z, ~a.w);
}

__device__ __forceinline__
void operator^= (uint4 &a, uint4 b)
{
    a = a ^ b;
}

__device__ __forceinline__
uint4 operator^ (uint4 a, uint2 b)
{
    return make_uint4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.x, a.w ^ b.y);
}

__device__ __forceinline__
uint4 operator*(uint4 a, uint32_t b)
{
    return make_uint4(a.x * b, a.y * b, a.z * b,  a.w * b);
}

//MARK: misc
__device__ __forceinline__
uint32_t bfe(const uint32_t x, const uint32_t bit, const uint32_t numBits)
{
	uint32_t ret;
	asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(x), "r"(bit), "r"(numBits));
	return ret;

}

//MARK: Macros to catch CUDA errors in CUDA runtime calls
#define CUDA_SAFE_CALL(call)                                          \
do {                                                                  \
	cudaError_t err = call;                                           \
	if (cudaSuccess != err) {                                         \
		fprintf(stderr, "Cuda error in func '%s' at line %i : %s.\n", \
		         __FUNCTION__, __LINE__, cudaGetErrorString(err) );   \
		exit(EXIT_FAILURE);                                           \
	}                                                                 \
} while (0)

#endif // #ifndef CUDA_HELPER_H
