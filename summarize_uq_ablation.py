from pathlib import Path
import argparse,json
p=argparse.ArgumentParser()
p.add_argument('--tag',default='v1');p.add_argument('--seed',type=int,default=42)
a=p.parse_args()
root=Path(__file__).resolve().parent/'uq_diagnostics/Vaihingen/DeepKANSeg'/f's{a.seed}_{a.tag}'
print('stage         epochs    best mIoU    last mIoU    baseline')
for path in sorted((root/'runs').rglob('history.jsonl')):
    rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if not rows: continue
    key='miou_5';base=rows[-1].get('baseline_miou_5')
    print(f'{path.parent.name:12} {len(rows):6} {max(x[key] for x in rows):12.6f} {rows[-1][key]:12.6f} {str(base):>12}')
print('Compare candidate metrics, not fallback-selected baseline scores. Partial runs are provisional.')
