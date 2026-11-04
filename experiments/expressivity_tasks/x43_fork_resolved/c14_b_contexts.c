/* Raw cached query metadata only. Answer arithmetic stays in the frozen decoder. */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
static unsigned u16(const unsigned char *p) { return p[0] | ((unsigned)p[1]<<8); }
int c14_contexts(const unsigned char *p, size_t n, int64_t *out, size_t capacity) {
    size_t at=11,w=0;
    if(n<11 || memcmp(p,"C14SPARSE1\n",11)) return -1;
    while(at<n) {
        if(w>=capacity || n-at<18) return -2;
        unsigned len=u16(p+at+16);at+=18;
        if(!len || len>128) return -3;
        for(unsigned j=0;j<len;j++) {
            if(n-at<5) return -4;
            unsigned k=p[at],censored=p[at+1],eof=p[at+2],size=u16(p+at+3);at+=5;
            if(k>j+1 || censored>1 || eof>1 || size>256 || (censored&&!k) || (!k&&(size||eof)) || n-at<size*9) return -5;
            out[w*128+j]=k;
            at+=9*size;
        }
        w++;
    }
    return (int)w;
}

/* Exact window-local metadata equality/expiry. Never used as an address key. */
int c14_owner_metadata(const unsigned char *raw, const int64_t *contexts,
                       const int32_t *valid, int32_t *owners, int32_t *expiry,
                       size_t batches, size_t width) {
    for(size_t b=0;b<batches;b++) {
        const unsigned char *r=raw+b*width;
        int next=0,n=valid[b];
        if(n<0 || (size_t)n>width) return -1;
        for(size_t i=0;i<width;i++) {
            size_t at=b*width+i;
            owners[at]=-1;expiry[at]=n;
            if((int)i>=n) continue;
            int64_t k=contexts[at];
            if(k<0 || k>(int64_t)i+1) return -2;
            if(!k) continue;
            int id=-1;
            for(size_t prev=0;prev<i;prev++) {
                if(contexts[b*width+prev]==k && !memcmp(r+i+1-k,r+prev+1-k,(size_t)k)) {
                    id=owners[b*width+prev];break;
                }
            }
            if(id<0) id=next++;
            owners[at]=id;
            for(int j=(int)i+1;j<n;j++) {
                if(memcmp(r+i+1-k,r+j+1-k,(size_t)k)) { expiry[at]=j;break; }
            }
        }
    }
    return 0;
}
