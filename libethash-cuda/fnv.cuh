/*
 *  https://en.wikipedia.org/wiki/Fowler–Noll–Vo_hash_function
 */


#define FNV_PRIME	0x01000193

__device__ __forceinline__
const uint32_t fnv(const uint32_t x, const uint32_t y)
{
    return x * FNV_PRIME ^ y;
}

__device__ __forceinline__
const uint4 fnv4(const uint4 a, const uint4 b)
{
	uint4 c;
	c.x = a.x * FNV_PRIME ^ b.x;
	c.y = a.y * FNV_PRIME ^ b.y;
	c.z = a.z * FNV_PRIME ^ b.z;
	c.w = a.w * FNV_PRIME ^ b.w;
	return c;
}

__device__ __forceinline__
const uint32_t fnv_reduce(uint4 v)
{
	return fnv(fnv(fnv(v.x, v.y), v.z), v.w);
}
