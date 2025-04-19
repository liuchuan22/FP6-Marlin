#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <vector>
#include <cuda_fp16.h>

void Extract_segments_from_8_padded_fp6(unsigned char Seg_6bit[], unsigned char Padded_8_FP6[], int bit_width, int bit_offset){
    for(int i=0; i< bit_width; i++)
        Seg_6bit[i] = 0;
    for(int i=0; i<8; i++){
        unsigned int seg = (Padded_8_FP6[i] << bit_offset) & 0x000000ff;
        int mask = 0xffffff00;
        seg &= mask >> bit_width;
        //
        int Seg_idx = (i * bit_width) / 8;
        int Seg_off = (i * bit_width) % 8;
        Seg_6bit[Seg_idx] |= seg >> Seg_off;
    }
}

unsigned char Extract_6_Bits_To_A_Byte(unsigned char* Bytes, int ByteOffset, int BitOffset){
    assert (sizeof(unsigned int)==4);
    unsigned int tmp_int32_word=0;
    unsigned char* uchar_ptr = reinterpret_cast<unsigned char*>(&tmp_int32_word);
    uchar_ptr[3] = Bytes[ByteOffset+0];
    uchar_ptr[2] = Bytes[ByteOffset+1];
    tmp_int32_word = tmp_int32_word << BitOffset;

    signed int mask = 0x80000000;
    mask = mask >> (5);
    tmp_int32_word &= mask;

    unsigned char out = uchar_ptr[3];
    return out;
}

void Assign_32_FP6_To_4_Thread(
    std::vector<unsigned char> Vec_Seg_2bit[], 
    std::vector<unsigned char> Vec_Seg_4bit[], 
    unsigned char* PTR[]) 
{
    constexpr int BIT_WIDTH = 6;
    constexpr int nTHREADS = 4;
    constexpr int FPx_PER_THREAD = 8;
    unsigned char Padded_8_FP6[nTHREADS][FPx_PER_THREAD];
    for(int i=0; i<nTHREADS; i++){                             // 4 threads
        for(int j=0; j<FPx_PER_THREAD; j++){                   // 8 FPx per thread
            int offset = (i*2 + j%2) * BIT_WIDTH;
            int ByteOffset = offset / 8;
            int BitOffset  = offset % 8;
            Padded_8_FP6[i][j] = Extract_6_Bits_To_A_Byte(PTR[j/2], ByteOffset, BitOffset);
        }
    }

    unsigned char Seg_2bit[nTHREADS][2];
    unsigned char Seg_4bit[nTHREADS][4];
    for(int t=0; t<nTHREADS; t++){
        Extract_segments_from_8_padded_fp6(Seg_2bit[t], Padded_8_FP6[t], 2, int(BIT_WIDTH & 1));
        Extract_segments_from_8_padded_fp6(Seg_4bit[t], Padded_8_FP6[t], 4, int(BIT_WIDTH & 3));
    }

    for(int t=0; t<4; t++)
    {
        Vec_Seg_2bit[t].push_back(Seg_2bit[t][0]);
        Vec_Seg_2bit[t].push_back(Seg_2bit[t][1]);
        Vec_Seg_4bit[t].push_back(Seg_4bit[t][0]);
        Vec_Seg_4bit[t].push_back(Seg_4bit[t][1]);
        Vec_Seg_4bit[t].push_back(Seg_4bit[t][2]);
        Vec_Seg_4bit[t].push_back(Seg_4bit[t][3]);
    }
}

template<int BIT_WIDTH>
void BitInterleaving_x_bit(unsigned char* PTR_4Bytes)
{
    unsigned int *PTR_UINT = reinterpret_cast<unsigned int*>(PTR_4Bytes);
    unsigned int input  = *PTR_UINT;
    //
    int* order = NULL;
    int order_1bit[32] = {2,6,10,14,18,22,26,30,
                          4,8,12,16,20,24,28,32,
                          1,5,9, 13,17,21,25,29,
                          3,7,11,15,19,23,27,31};  // pre-defined order for bit-interleaving in FP6-LLM
    int order_2bit[16] = {2,6,10,14,4,8,12,16,1,5,9,13,3,7,11,15};  // pre-defined order for bit-interleaving in FP6-LLM
    int order_4bit[8] = {2,6,4,8,1,5,3,7};  // pre-defined order for bit-interleaving in FP6-LLM
    if(BIT_WIDTH==1) order = order_1bit;
    if(BIT_WIDTH==2) order = order_2bit;
    if(BIT_WIDTH==4) order = order_4bit;
    assert(order);
    //
    int mask = 0x80000000;
    assert(BIT_WIDTH>=1);
    mask = mask >> (BIT_WIDTH-1);
    //
    unsigned int output = 0x00000000;
    for(int i=0; i<32/BIT_WIDTH; i++){
        unsigned int Frag_xbit = ( input << BIT_WIDTH*(order[i]-1) ) & mask;    // The highest x bits are used to store the extracted fragments.
        output |= Frag_xbit >> (i*BIT_WIDTH);
    }
    //
    *PTR_UINT = output;
}

void weight_matrix_prepacking(int* packed_weights, int* FP6Weights, size_t K, size_t N) {
    assert(K % 64 == 0);
    assert(N % 64 == 0);
    constexpr int BIT_WIDTH = 6;
    // pointers
    unsigned char* Weight_6bit = reinterpret_cast<unsigned char*>(FP6Weights);
    unsigned char* Weight_2bit = reinterpret_cast<unsigned char*>(packed_weights);
    unsigned char* Weight_4bit = Weight_2bit + K*N*2/8;
    // segments
    std::vector<unsigned char> A_Segment_2bit[32];
    std::vector<unsigned char> A_Segment_4bit[32];
    size_t BytesPerRow = N*BIT_WIDTH/8;
    // Modification: 2+4 split in row-major order instead of column-major. 
    for (size_t i=0; i< N / 16; i++) {
        for (size_t j=0; j < K / 16; j++) {
            // for (int k=0; k < 64 / 16; k++) {
                size_t row = j*16;
                size_t col = i*16;
                // B matrix is transposed therefore we transpose the pointer to get the correct address
                unsigned char* StartPTR_1 = Weight_6bit + row*BytesPerRow + col*(BIT_WIDTH)/8;
                unsigned char* StartPTR_3 = StartPTR_1 + 8*BytesPerRow;
                unsigned char* StartPTR_2 = StartPTR_1 + 8*(BIT_WIDTH)/8;
                unsigned char* StartPTR_4 = StartPTR_3 + 8*(BIT_WIDTH)/8;
                // Dealing with each 16*16 blocks then...
                for(int l=0; l<8; l++) {
                    unsigned char* PTR[4]={StartPTR_1+l*BytesPerRow, StartPTR_2+l*BytesPerRow, StartPTR_3+l*BytesPerRow, StartPTR_4+l*BytesPerRow};
                    Assign_32_FP6_To_4_Thread(&A_Segment_2bit[l*4], &A_Segment_4bit[l*4], PTR);
                }
            // }
        }
    }
    // Verifying the length of 2/4_bit segments.
    size_t BytesPerThread_2bit = K*N*2/8/32;
    size_t BytesPerThread_4bit = K*N*4/8/32;
    for(int i=0; i<32; i++){
        assert(A_Segment_2bit[i].size()==BytesPerThread_2bit);
        assert(A_Segment_4bit[i].size()==BytesPerThread_4bit);
    }
    // Optimizing coleasced global memory access
    for(size_t i=0; i<BytesPerThread_2bit/8; i++) 
        for(int t=0; t<32; t++)
            for(int a = 0; a<2; a++) // why a: loading 8 bytes of 1 thread at a time (e.g., b0&b1 t0, b2&b3 t0)
                for(int b=0; b<4; b++) // why (3-b): special byte order within a register
                    Weight_2bit[i*256+t*8+a*4+(3-b)] = A_Segment_2bit[t][i*8+a*4+b];
    for(size_t i=0; i<BytesPerThread_4bit/16; i++)
        for(int t=0; t<32; t++)
            for(int a=0; a<4; a++) // e.g., b0t0, b1t0, b2t0, b3t0
                for(int b=0; b<4; b++) // why (3-b): special byte order within a register
                    Weight_4bit[i*512+t*16+a*4+(3-b)] = A_Segment_4bit[t][i*16+a*4+b];
    // Bit-level interleaving
    for(size_t i=0; i<BytesPerThread_2bit*32/4; i++)
        BitInterleaving_x_bit<2>(Weight_2bit+4*i);
    for(size_t i=0; i<BytesPerThread_4bit*32/4; i++)
        BitInterleaving_x_bit<4>(Weight_4bit+4*i);
}

// Dequantization
void dequantMatrix_fp6_to_fp16(half* A_16bit_h, unsigned char* A_x_bit_h, size_t M, size_t K, half* scale) {
    //
    assert(M%64==0);
    assert(K%64==0);
    constexpr int BIT_WIDTH = 6;
    size_t TotalSizeInByte = M * K * BIT_WIDTH / 8;
    //
    half* OutPTR = A_16bit_h;
    for(size_t i=0; i<TotalSizeInByte/BIT_WIDTH; i++) {    // Processing BIT_WIDTH Bytes for each Loop, generating 8 FP16.
        unsigned char Bytes[BIT_WIDTH];
        for(int x=0; x<BIT_WIDTH; x++)  Bytes[x] = A_x_bit_h[i*BIT_WIDTH+x];
        unsigned char OUT[8];
        // Prepare Initial memory layout for Dequant
        for(int x=0; x<8; x++) {
            int ByteOffset  = BIT_WIDTH * x / 8;
            int BitOffset   = BIT_WIDTH * x % 8;
            OUT[x] = Extract_6_Bits_To_A_Byte(Bytes, ByteOffset, BitOffset);
        }
        // Dequant
        constexpr int MASK1 = 0x80000000;
        constexpr int MASK2 = MASK1 >> 5;
        constexpr int MASK  = MASK2 & 0x7fffffff;
        constexpr int RIGHT_SHIFT = 2;
        constexpr int BIAS = 4096;
        for(int x=0; x<8; x++) {
            unsigned int OUT_fp16;        // Storing fp16 in the high 16 bits.
            OUT_fp16 = int(OUT[x]) << 24;
            OUT_fp16 = (OUT_fp16 & 0x80000000) | ( (OUT_fp16 & MASK) >> RIGHT_SHIFT );
            OUT_fp16 = OUT_fp16 >> 16;
            //
            half* OUT_FP16_PTR = reinterpret_cast<half*>(&OUT_fp16);
            // OutPTR[x] = __float2half_rn( __half2float(*OUT_FP16_PTR) * (1.0f*BIAS) * __half2float(scale[(8*i)/K]));
            OutPTR[x] = __hmul(__hmul(*OUT_FP16_PTR, __float2half_rn(1.0f*BIAS)), scale[(8*i)/K]);
        }   
        //
        OutPTR +=8;
    }
}
