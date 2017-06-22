#include "ethash_cuda_miner_kernel_globals.cuh"
#include "ethash_cuda_miner_kernel.h"
#include "cuda_helper.cuh"

//__device__
//uint64_t compute_hash(uint64_t nonce)
//{
//    // sha3_512(header .. nonce)
//    uint2 state[25];
//
//    state[4] = vectorize(nonce);
//
//    keccak_f1600_init(state);
//
//    // Threads work together in this phase in groups of 8.
//    const int thread_id     = threadIdx.x & (THREADS_PER_HASH - 1);
//    const int mix_idx       = thread_id & 3;
//    const int shuffle_idx_1 = mix_idx * 2;
//    const int shuffle_idx_2 = shuffle_idx_1 + 1;
//
//    uint4 mix;
//    uint2 shuffle[8];
//
//    for(int i = 0;i < THREADS_PER_HASH;++i)
//    {
//        //share init among threads
//        for(int j = 0;j < 8;++j)
//        {
//            shuffle[j].x = __shfl(state[j].x, i, THREADS_PER_HASH);
//            shuffle[j].y = __shfl(state[j].y, i, THREADS_PER_HASH);
//        }
//        
//        //prepare mix
//        mix = vectorize2(shuffle[shuffle_idx_1], shuffle[shuffle_idx_2]);
//        
//        //init0
//        uint32_t init0 = __shfl(shuffle[0].x, 0, THREADS_PER_HASH);
//        for(uint32_t a = 0;a < ACCESSES;a += 4)
//        {
//            int t = bfe(a, 2u, 3u);
//            
//            for(uint32_t b = 0;b < 4;++b)
//            {
//                if(thread_id == t)
//                {
//                    shuffle[0].x = fnv(init0 ^ (a + b), ((uint32_t*)&mix)[b]) % d_dag_size;
//                }
//                
//                shuffle[0].x = __shfl(shuffle[0].x, t, THREADS_PER_HASH);
//                mix = fnv4(mix, d_dag[shuffle[0].x].uint4s[thread_id]);
//            }
//        }
//        
//        //update mix accross threads
//        uint32_t thread_mix = fnv_reduce(mix);
//        shuffle[0].x = __shfl(thread_mix, 0, THREADS_PER_HASH);
//        shuffle[0].y = __shfl(thread_mix, 1, THREADS_PER_HASH);
//        shuffle[1].x = __shfl(thread_mix, 2, THREADS_PER_HASH);
//        shuffle[1].y = __shfl(thread_mix, 3, THREADS_PER_HASH);
//        shuffle[2].x = __shfl(thread_mix, 4, THREADS_PER_HASH);
//        shuffle[2].y = __shfl(thread_mix, 5, THREADS_PER_HASH);
//        shuffle[3].x = __shfl(thread_mix, 6, THREADS_PER_HASH);
//        shuffle[3].y = __shfl(thread_mix, 7, THREADS_PER_HASH);
//        
//        if(i == thread_id)
//        {
//            //move mix into state:
//            state[8] = shuffle[0];
//            state[9] = shuffle[1];
//            state[10] = shuffle[2];
//            state[11] = shuffle[3];
//        }
//    }
//
//    // keccak_256(keccak_512(header..nonce) .. mix);
//    return keccak_f1600_final(state);
//}



#define PARALLEL_HASH 4

__device__
uint64_t compute_hash(uint64_t nonce)
{
	// sha3_512(header .. nonce)
	uint2 state[25];
	
	state[4] = vectorize(nonce);

	keccak_f1600_init(state);
	
	// Threads work together in this phase in groups of 8.
	const int thread_id     = threadIdx.x & (THREADS_PER_HASH - 1);
	const int mix_idx       = thread_id & 3;
    const int shuffle_idx_1 = mix_idx * 2;
    const int shuffle_idx_2 = shuffle_idx_1 + 1;

	for(int i = 0;i < THREADS_PER_HASH;i += PARALLEL_HASH)
	{
		uint4 mix[PARALLEL_HASH];
		uint32_t offset[PARALLEL_HASH];
		uint32_t init0[PARALLEL_HASH];
	
		//share init among threads
		for(int p = 0;p < PARALLEL_HASH;++p)
		{
			uint2 shuffle[8];
			for(int j = 0;j < 8;++j)
			{
				shuffle[j].x = __shfl(state[j].x, i+p, THREADS_PER_HASH);
				shuffle[j].y = __shfl(state[j].y, i+p, THREADS_PER_HASH);
			}
            
            mix[p] = vectorize2(shuffle[shuffle_idx_1], shuffle[shuffle_idx_2]);
            
			init0[p] = __shfl(shuffle[0].x, 0, THREADS_PER_HASH);
		}

		for(uint32_t a = 0;a < ACCESSES;a += 4)
		{
			int t = bfe(a, 2u, 3u);

			for(uint32_t b = 0;b < 4;++b)
			{
				for(int p = 0;p < PARALLEL_HASH;++p)
				{
					offset[p] = fnv(init0[p] ^ (a + b), ((uint32_t *)&mix[p])[b]) % d_dag_size;
					offset[p] = __shfl(offset[p], t, THREADS_PER_HASH);
				}
                
				#pragma unroll
				for(int p = 0;p < PARALLEL_HASH;++p)
				{
					mix[p] = fnv4(mix[p], d_dag[offset[p]].uint4s[thread_id]);
				}
			}
		}

		for(int p = 0;p < PARALLEL_HASH;++p)
		{
			uint2 shuffle[4];
			uint32_t thread_mix = fnv_reduce(mix[p]);

			// update mix accross threads

			shuffle[0].x = __shfl(thread_mix, 0, THREADS_PER_HASH);
			shuffle[0].y = __shfl(thread_mix, 1, THREADS_PER_HASH);
			shuffle[1].x = __shfl(thread_mix, 2, THREADS_PER_HASH);
			shuffle[1].y = __shfl(thread_mix, 3, THREADS_PER_HASH);
			shuffle[2].x = __shfl(thread_mix, 4, THREADS_PER_HASH);
			shuffle[2].y = __shfl(thread_mix, 5, THREADS_PER_HASH);
			shuffle[3].x = __shfl(thread_mix, 6, THREADS_PER_HASH);
			shuffle[3].y = __shfl(thread_mix, 7, THREADS_PER_HASH);

			if((i + p) == thread_id)
            {
				//move mix into state:
				state[8] = shuffle[0];
				state[9] = shuffle[1];
				state[10] = shuffle[2];
				state[11] = shuffle[3];
			}
		}
	}
	
	// keccak_256(keccak_512(header..nonce) .. mix);
	return keccak_f1600_final(state);
}



