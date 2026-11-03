/* Sparse C14 decoding. Preserve branch order and Python double arithmetic. */
#include <stdint.h>
#include <string.h>
#include <math.h>
static uint64_t u64(const unsigned char *p) { uint64_t x; memcpy(&x,p,8); return x; }
static unsigned u16(const unsigned char *p) { return p[0] | ((unsigned)p[1]<<8); }
int c14_decode(const unsigned char *p, size_t n, float *out,
               uint64_t *ids, uint16_t *lengths, size_t capacity) {
    size_t at=11, w=0;
    if(n<11 || memcmp(p,"C14SPARSE1\n",11)) return -1;
    while(at<n) {
        if(w>=capacity || n-at<18) return -2;
        ids[2*w]=u64(p+at); ids[2*w+1]=u64(p+at+8);
        unsigned len=u16(p+at+16); at+=18;
        if(!len || len>128 || ids[2*w+1]%128) return -3;
        lengths[w]=len;
        for(unsigned j=0;j<len;j++) {
            if(n-at<5) return -4;
            unsigned k=p[at], censored=p[at+1], eof=p[at+2], size=u16(p+at+3); at+=5;
            if(k>j+1 || censored>1 || eof>1 || size>256 || (censored&&!k) || (!k&&(size||eof)) || n-at<size*9) return -5;
            /* Sum in double: real cache counts fit exactly within 53 bits. */
            uint64_t total=0; unsigned distinct=0; unsigned char seen[256]={0};
            for(unsigned b=0;b<size;b++) {
                unsigned byte=p[at+9*b]; uint64_t count=u64(p+at+9*b+1);
                if(seen[byte] || !count) return -6;
                seen[byte]=1;
                if(byte!=30) { if(UINT64_MAX-total<count) return -7; total+=count; distinct++; }
            }
            if(total>9007199254740992ULL) return -8;
            float *f=out+(w*128+j)*260;
            if(total) {
                double entropy=0.;
                for(unsigned b=0;b<size;b++) {
                    unsigned byte=p[at+9*b]; uint64_t count=u64(p+at+9*b+1);
                    if(byte!=30) { double prob=(double)count/(double)total; f[byte]=(float)prob; entropy+=prob*log(prob); }
                }
                f[256]=(float)log1p((double)total); f[257]=(float)(-entropy);
                f[258]=(float)(distinct>1); f[259]=(float)log1p((double)k);
            }
            at+=9*size;
        }
        w++;
    }
    return (int)w;
}
