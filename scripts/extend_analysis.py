# Prepares source-disjoint evaluation samples, scores frozen checkpoints and produces grouped statistical summaries, plots and optional OCR-audit outputs.
"""Prepare, score and analyse the held-out samples from the project root."""
import argparse
import csv
import hashlib
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path('results/heldout')
DATA = Path('data/heldout')

def read(path):
    with open(path, encoding='utf-8-sig') as f:
        return [json.loads(l) for l in f if l.strip()]

def write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

def norm(s):
    return ' '.join(unicodedata.normalize('NFC', s).split())

def key(s):
    return hashlib.sha256(norm(s).encode()).hexdigest()

def issue(file):

    return re.sub(r'_p\d+(?:\.txt)?$', '', Path(file).name)

def metadata(r, corpus):
    text = r['text']
    if corpus == 'historical':
        group = issue(r['file'])
        m = re.match(r'([A-Za-z]+)_(\d{4})', Path(r['file']).name)
        if not m:
            raise ValueError('Unrecognised historical filename: ' + r['file'])
        source, year = m.group(1), int(m.group(2))
    else:
        if not r.get('doc'):
            raise ValueError('Modern records must retain their original doc identifier.')
        group, source, year = str(r['doc']), 'NCHLT', None
    return dict(r, id=key(text), corpus=corpus, group=group,
                source=source, year=year, chars=len(text), words=len(text.split()))

def prepare(args):
    if DATA.exists():
        raise SystemExit('data/heldout exists. Preserve it; do not regenerate a test after seeing its results.')
    old = {}
    old_keys = set()
    for corpus, filename in [('historical', 'hist_sample'), ('modern', 'modern_sample')]:
        old[corpus] = [metadata(r, corpus) for r in read(f'data/samples/{filename}.jsonl')]
        old_keys.update(r['id'] for r in old[corpus])
    hist = []
    for r in read('data/historical_clean/historical_clean.jsonl'):
        hist.extend(metadata({'text': s, 'file': r['file']}, 'historical') for s in r['sentences'])
    modern = [metadata(r, 'modern') for r in read('data/modern_corpus/modern_sentences.jsonl')]


    # Remove exact cross-corpus matches before splitting.
    overlap = {r['id'] for r in hist} & {r['id'] for r in modern}
    all_rows, counts = [], {}
    rng = random.Random(args.seed)
    for corpus, pool in [('historical', hist), ('modern', modern)]:
        seen, clean = set(), []
        for r in pool:
            if r['id'] not in seen and r['id'] not in overlap:
                seen.add(r['id'])
                clean.append(r)
        old_groups = {r['group'] for r in old[corpus]}
        groups = sorted({r['group'] for r in clean} - old_groups)
        rng.shuffle(groups)


        # Previously evaluated sources stay in development.
        n_all = len({r['group'] for r in clean})
        n_hold = max(1, round(.15 * n_all))
        if len(groups) < 2 * n_hold + 1:
            raise ValueError(f'{corpus}: too few unseen groups for requested split.')
        val, test = set(groups[:n_hold]), set(groups[n_hold:2*n_hold])
        for r in clean:
            r['split'] = 'validation' if r['group'] in val else 'test' if r['group'] in test else 'development_train'

            if r['split'] != 'development_train' and r['id'] in old_keys:
                continue
            all_rows.append(r)
        counts[corpus] = {'input_rows': len(pool), 'deduplicated_rows': len(clean),
                          'old_source_groups': len(old_groups), 'all_source_groups': n_all}
    dev_hist = [r for r in all_rows if r['corpus'] == 'historical' and r['split'] == 'development_train']

    # Define length bins from development data only.
    edges = np.quantile([r['chars'] for r in dev_hist], [0, .2, .4, .6, .8, 1])
    edges[0], edges[-1] = 0, np.inf
    if np.any(np.diff(edges) <= 0):
        raise ValueError('Length bin edges not unique.')
    for r in all_rows:
        r['length_bin'] = int(np.searchsorted(edges, r['chars'], side='right') - 1)
    write(DATA / 'split_manifest.jsonl', all_rows)
    for split in ['validation', 'test']:
        bins = {(c, b): [r for r in all_rows if r['corpus'] == c and r['split'] == split and r['length_bin'] == b]
                for c in ['historical', 'modern'] for b in range(5)}
        per_bin = args.n // 5
        if args.n % 5 or min(map(len, bins.values())) < per_bin:
            raise ValueError('Requested n must be divisible by 5 and fit all bins. Preserve manifest; inspect availability.')
        sample = []
        for c in ['historical', 'modern']:
            for b in range(5):
                sample.extend(rng.sample(bins[c, b], per_bin))
        rng.shuffle(sample)
        write(DATA / f'{split}.jsonl', sample)
    protocol = {'seed': args.seed, 'n_per_corpus_per_split': args.n,
                'length_edges': [None if not np.isfinite(x) else float(x) for x in edges],
                'counts': counts, 'cross_corpus_overlap_removed': len(overlap),
                'primary_metric': 'median historical minus modern NLL per character, within model',
                'secondary_metrics': ['NLL per token', 'tokens per character', 'tokens per whitespace word'],
                'inference': 'independent source-group bootstrap; Holm across models for each metric',
                'scope': 'Frozen pretrained checkpoints. development_train is unused for training.',
                'test_policy': 'Freeze checkpoints, filters, scoring and reporting before scoring test.'}
    (DATA / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.glob('*') if p.is_file()}
    (DATA / 'sha256.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    print(pd.DataFrame(all_rows).groupby(['corpus','split']).agg(sentences=('id','size'), groups=('group','nunique')))
    print('Prepared immutable samples. Archive protocol.json and sha256.json before evaluation.')

def imports():
    sys.path.insert(0, str(Path('scripts').resolve()))
    from load_models import MODELS, load
    return MODELS, load

def measure(text, tok, model, kind):
    import torch
    encoded = tok(text, return_tensors='pt', return_special_tokens_mask=True, truncation=False)
    special = encoded.pop('special_tokens_mask')[0].bool()
    ids = encoded['input_ids'][0]
    limits = [getattr(tok, 'model_max_length', 10**9), getattr(model.config, 'max_position_embeddings', 10**9)]

    if len(ids) > min(limits):
        raise ValueError(f'Context too long ({len(ids)}); no truncation or silent exclusion allowed.')
    encoded = {k:v.to(model.device) for k,v in encoded.items()}
    content_n = int((~special).sum())
    model.eval()
    with torch.no_grad():
        if kind == 'mlm':
            total, n = 0., 0
            for i in range(len(ids)):
                if special[i]:
                    continue
                x = {k:v.clone() for k,v in encoded.items()}
                target = int(x['input_ids'][0,i])
                x['input_ids'][0,i] = tok.mask_token_id
                total -= float(torch.log_softmax(model(**x).logits[0,i].float(), -1)[target])
                n += 1
        else:
            labels = encoded['input_ids'].clone()


            # The shifted loss includes any tokenizer-added end token.
            n = labels.shape[1] - 1
            total = float(model(**encoded, labels=labels).loss) * n
    if n < 1 or not np.isfinite(total):
        raise ValueError('Invalid score')
    return {'nll_total': total, 'n_scored_tokens': n, 'nll_token': total/n,
            'nll_char': total/len(text), 'content_tokens': content_n,
            'tokens_per_char': content_n/len(text), 'tokens_per_word': content_n/len(text.split())}

def score(args):
    hashes = json.loads((DATA / 'sha256.json').read_text(encoding='utf-8'))
    for name, expected_hash in hashes.items():
        if hashlib.sha256((DATA/name).read_bytes()).hexdigest() != expected_hash:
            raise ValueError('Frozen sample/protocol changed: '+name)
    MODELS, load = imports()
    rows = read(DATA / f'{args.split}.jsonl')
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f'{args.split}_scores.jsonl'
    if dest.exists():
        raise SystemExit('Score output already exists; preserve it instead of overwriting.')
    config = {k: MODELS[k] for k in MODELS}
    (OUT / f'{args.split}_models.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    import torch
    for name in MODELS:
        tok, model, kind = load(name)
        details = {'model': name, 'checkpoint': MODELS[name],
                   'resolved_commit': getattr(model.config, '_commit_hash', None),
                   'tokenizer_commit': tok.init_kwargs.get('_commit_hash'),
                   'model_type': kind}
        with (OUT / f'{args.split}_resolved_models.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(details)+'\n')
        for r in rows:
            result = dict(r, model=name, model_type=kind, **measure(r['text'], tok, model, kind))
            with dest.open('a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False)+'\n')
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(name, 'complete', flush=True)

def holm(values):
    p = np.array(values, float)
    ans = np.empty(len(p))
    running = 0.
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p)-rank)*p[i])
        ans[i] = min(running, 1.)
    return ans

def cluster_draw(d, column, rng):
    arrays = [g[column].to_numpy() for _,g in d.groupby('group', sort=True)]
    if len(arrays) < 2:
        raise ValueError('Need multiple source groups for bootstrap.')
    return np.concatenate([arrays[i] for i in rng.integers(len(arrays), size=len(arrays))])

def contrast(a, b, metric, rng, boots):
    av, bv = a[metric].to_numpy(), b[metric].to_numpy()
    delta = float(np.median(av)-np.median(bv))
    sims = np.array([np.median(cluster_draw(a, metric, rng))-np.median(cluster_draw(b, metric, rng)) for _ in range(boots)])

    cliffs = float(np.sign(av[:,None]-bv[None,:]).mean())


    # This centred-bootstrap p-value is approximate.
    boot_p = (1+np.count_nonzero(np.abs(sims-delta) >= abs(delta)))/(boots+1)
    return {'gap': delta, 'ci_low': float(np.quantile(sims,.025)), 'ci_high': float(np.quantile(sims,.975)),
            'cliffs_delta': cliffs,
            'cluster_boot_p_approx': boot_p,
            'n_a':len(a),'n_b':len(b),'groups_a':a.group.nunique(),'groups_b':b.group.nunique()}

def analyze(args):
    d = pd.DataFrame(read(OUT / f'{args.split}_scores.jsonl'))
    expected = pd.DataFrame(read(DATA / f'{args.split}.jsonl'))
    MODELS, _ = imports()
    if set(d.model) != set(MODELS):
        raise ValueError('Incomplete model set; finish scoring before analysis.')
    for name,g in d.groupby('model'):
        if len(g) != len(expected) or set(g.id) != set(expected.id) or g.id.duplicated().any():
            raise ValueError('Missing or duplicate scores: '+name)
    rng = np.random.default_rng(args.seed)
    summary = []
    for metric in ['nll_char','nll_token','tokens_per_char','tokens_per_word']:
        block=[]
        for name,g in d.groupby('model'):
            a,b = g[g.corpus=='historical'],g[g.corpus=='modern']
            block.append(dict(model=name, metric=metric, **contrast(a,b,metric,rng,args.boots)))
        for r,p in zip(block,holm([r['cluster_boot_p_approx'] for r in block])):
            r['cluster_boot_p_holm_approx']=p
        summary.extend(block)
    pd.DataFrame(summary).to_csv(OUT/f'{args.split}_summary.csv',index=False)
    d.groupby(['model','corpus']).agg(n=('id','size'), groups=('group','nunique'),
        mean_chars=('chars','mean'), median_chars=('chars','median')).to_csv(OUT/f'{args.split}_coverage.csv')


    # Save random passages alongside score extremes.
    audits=[]
    for (name,corp),g in d.groupby(['model','corpus']):
        for reason,part in [('lowest_nll',g.nsmallest(5,'nll_char')),('highest_nll',g.nlargest(5,'nll_char')),('random',g.sample(min(10,len(g)),random_state=args.seed))]:
            for r in part.to_dict('records'):
                audits.append(dict(r,selection=reason,visible_damage='',fragment='',proper_names='',notes=''))
    pd.DataFrame(audits).to_csv(OUT/f'{args.split}_qualitative_audit.csv',index=False)
    h=d[d.corpus=='historical'].copy()
    h['period']=h['source']+' '+np.where(h.year<1884,'before1884',np.where(h.year<=1888,'1884-1888','after1888'))
    h.groupby(['model','period']).agg(n=('id','size'),groups=('group','nunique'),median_nll_char=('nll_char','median')).to_csv(OUT/f'{args.split}_periods.csv')


    # Restrict this contrast to years represented in both newspapers.
    years = [set(g.year) for _,g in h.groupby('source')]
    common = set.intersection(*years) if len(years)==2 else set()
    comparisons=[]
    for name,g in h[h.year.isin(common)].groupby('model'):
        a,b=g[g.source=='IMVO'],g[g.source=='ISIG']
        if a.group.nunique()>1 and b.group.nunique()>1:
            comparisons.append(dict(model=name,years=','.join(map(str,sorted(common))),
                **contrast(a,b,'nll_char',rng,args.boots)))
    pd.DataFrame(comparisons).to_csv(OUT/f'{args.split}_newspapers_shared_years.csv',index=False)
    print(pd.DataFrame(summary).query("metric=='nll_char'").to_string(index=False))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('Tables complete. Install matplotlib to also produce the distribution plot.')
        return
    fig,axes=plt.subplots(1,len(MODELS),figsize=(3*len(MODELS),3.5),squeeze=False)
    for ax,(name,g) in zip(axes[0],d.groupby('model')):
        ax.boxplot([g[g.corpus==c].nll_char for c in ['historical','modern']],tick_labels=['Historical','Modern'],showfliers=False)
        ax.set_title(name); ax.set_ylabel('Negative (pseudo) log score / character')
    fig.tight_layout(); fig.savefig(OUT/f'{args.split}_scores.png',dpi=200); plt.close(fig)

def ocr_template(args):
    rows=read(DATA/'validation.jsonl')
    h=[r for r in rows if r['corpus']=='historical']
    random.Random(args.seed).shuffle(h)


    # Take at most one randomly encountered sentence per issue.
    selected=[]; used=set()
    for r in h:
        if r['group'] not in used:
            selected.append(dict(id=r['id'],file=r['file'],group=r['group'],observed=r['text'],
                                 reference='',verified='no',scan_locator='',notes=''))
            used.add(r['group'])
        if len(selected)==args.n:
            break
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'ocr_pairs.csv'
    if path.exists():
        raise SystemExit('OCR audit already exists; preserve annotations.')
    pd.DataFrame(selected).to_csv(path,index=False)
    print('Created',path,'with',len(selected),'distinct issues. Transcribe scans faithfully; preserve historical spelling.')

def editdistance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):
            cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]

def ocr(args):
    rows=pd.read_csv(args.pairs,keep_default_na=False).to_dict('records')
    verified=[r for r in rows if str(r['verified']).lower()=='yes' and r['reference'].strip() and r['scan_locator'].strip()]
    if not verified:
        raise ValueError('No scan-verified references. Do not invent corrections.')
    errors=[]
    for r in verified:
        a,b=norm(r['reference']),norm(r['observed'])
        errors.append(dict(r,char_edits=editdistance(a,b),reference_chars=len(a),
                          word_edits=editdistance(a.split(),b.split()),reference_words=len(a.split())))
    OUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(errors).to_csv(OUT/'ocr_errors.csv',index=False)
    report={'sampled':len(rows),'verified':len(verified),'unverified':len(rows)-len(verified),
            'CER':sum(r['char_edits'] for r in errors)/sum(r['reference_chars'] for r in errors),
            'WER':sum(r['word_edits'] for r in errors)/sum(r['reference_words'] for r in errors),
            'normalisation':'NFC and whitespace collapse; case and punctuation preserved',
            'scope':'Residual transcription discrepancies in selected cleaned validation sentences; not raw OCR CER.'}
    (OUT/'ocr_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(report)
    if args.score_pairs:
        MODELS,load=imports(); scores=[]; summary=[]
        rng=np.random.default_rng(args.seed)
        for name in MODELS:
            tok,model,kind=load(name)
            for r in verified:
                observed=measure(r['observed'],tok,model,kind)
                reference=measure(r['reference'],tok,model,kind)
                scores.append(dict(id=r['id'],group=r['group'],model=name,
                    observed_nll_char=observed['nll_char'],reference_nll_char=reference['nll_char'],
                    delta=observed['nll_char']-reference['nll_char']))
            del model
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            g=pd.DataFrame([r for r in scores if r['model']==name])
            if g.group.nunique()>1:
                sims=[np.mean(cluster_draw(g,'delta',rng)) for _ in range(args.boots)]
                summary.append(dict(model=name,mean_paired_delta=g.delta.mean(),
                    ci_low=float(np.quantile(sims,.025)),ci_high=float(np.quantile(sims,.975)),n=len(g)))
        write(OUT/'ocr_paired_scores.jsonl',scores)
        pd.DataFrame(summary).to_csv(OUT/'ocr_score_effect.csv',index=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    for name in ['prepare','score','analyze','ocr-template','ocr']:
        s=sub.add_parser(name)
        s.add_argument('--seed',type=int,default=20261031)
        s.add_argument('--boots',type=int,default=2000)
        if name in ['prepare','ocr-template']: s.add_argument('--n',type=int,default=300 if name=='prepare' else 50)
        if name in ['score','analyze']: s.add_argument('--split',choices=['validation','test'],required=True)
        if name=='ocr':
            s.add_argument('--pairs',default='results/heldout/ocr_pairs.csv')
            s.add_argument('--score-pairs',action='store_true')
    args=p.parse_args()
    {'prepare':prepare,'score':score,'analyze':analyze,'ocr-template':ocr_template,'ocr':ocr}[args.command](args)

if __name__=='__main__': main()
