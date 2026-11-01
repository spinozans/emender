"""Additional documented query formats and refusals, with verbatim logs."""
import json
from index_pipeline import OUT, XSA, run

def main():
    container=OUT/'build/corpus.sxi2'; corpus=(OUT/'corpus/corpus.bin').read_bytes()
    summary={}
    for name,data,flags in [
        ('missing_terminal',b'First Citizen:',[]),
        ('fasta_auto',b'>one\nFirst \nCitizen:\n',[]),
        ('fasta_forced',b'>one\nFirst \nCitizen:\n',['--patterns-fasta']),
    ]:
        file=OUT/(name+'.bin'); file.write_bytes(data)
        selector=flags or ['--patterns']
        p=run(name,[XSA,'query',container,*selector,file,'-j','1'])
        assert p.returncode==0
        expected=[]; start=0
        while (start:=corpus.find(b'First Citizen:',start))>=0: expected.append(start); start+=1
        actual=[int(line.split(b'\t')[1]) for line in p.stdout.splitlines()]
        assert sorted(actual)==expected
        summary[name]=dict(status='passed',occurrences=len(actual))
    for name,flags in [('counts_json_conflict',['--counts','--output','jsonl']),('removed_mode',['--mode','text']),('removed_ri4',['--ri4'])]:
        p=run(name,[XSA,'query',container,'--patterns',OUT/'patterns.bin','-j','1',*flags]); assert p.returncode!=0
        summary[name]=dict(status='passed',behavior='refused as documented',returncode=p.returncode)
    (OUT/'extra_query_edges.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__': main()
