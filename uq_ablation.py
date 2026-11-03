"""Controlled continuation experiments; never publish into production results."""
from pathlib import Path
import argparse, json
import uq_config as cfg
import run_tri_uq as engine

STAGES=('continue','adapter','smooth','kl','data','dist','model','full')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=STAGES,required=True)
    p.add_argument('--dataset',choices=('Vaihingen','Potsdam','LoveDA'),default='Vaihingen')
    p.add_argument('--model',choices=cfg.BASELINE_MODELS,default='DeepKANSeg')
    p.add_argument('--baseline-checkpoint')
    p.add_argument('--epochs',type=int,default=25)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--tag',default='v1')
    p.add_argument('--device',default='cuda')
    p.add_argument('--data-root')
    p.add_argument('--plan-only',action='store_true')
    a=p.parse_args()
    if a.dataset!='Vaihingen' or a.model!='DeepKANSeg':
        raise ValueError('Initial controlled experiment is restricted to Vaihingen DeepKANSeg legacy; other recipes need matched controls')
    checkpoint=cfg.resolve_baseline_checkpoint(a.model,a.baseline_checkpoint,a.dataset)
    # A manifest pins all stages to the exact same native baseline.
    root=cfg.ROOT/'uq_diagnostics'/a.dataset/a.model/f's{a.seed}_{a.tag}'
    manifest=root/'baseline.json'
    if manifest.is_file():
        recorded=json.loads(manifest.read_text())
        if Path(recorded['checkpoint']).resolve()!=checkpoint.resolve():
            raise ValueError('Baseline changed since first stage; pass the original --baseline-checkpoint or use a new --tag')
        if recorded['epochs']!=a.epochs:
            raise ValueError('Use identical epochs within an experiment tag')
    root.mkdir(parents=True,exist_ok=True)
    manifest.write_text(json.dumps({'checkpoint':str(checkpoint.resolve()),'epochs':a.epochs},indent=2))
    method='baseline' if a.stage=='continue' else 'uq'
    options=['train','--dataset',a.dataset,'--model',a.model,'--method',method,
             '--checkpoint',str(checkpoint),'--recipe','legacy','--output',str(root/'runs'),
             '--results-uq',str(root/'weights'),'--run-name',a.stage,'--seed',str(a.seed),
             '--epochs',str(a.epochs),'--steps-per-epoch','1000','--batch','10',
             '--eval-batch','10','--eval-stride','32','--lr',str(1e-4 if method=='baseline' else 1e-3),
             '--backbone-lr','0.0001','--decoder-lr','0.0001','--head-warmup-epochs','0',
             '--preview-every','0','--device',a.device,'--samples','16',
             '--lambda-id',str(.01 if a.stage in ('kl','data','dist','model','full') else 0),
             '--lambda-ood','0','--eta-max',str(.5 if a.stage in ('data','dist','model','full') else 0),
             '--dist-focus-weight',str(.5 if a.stage in ('dist','full') else 0),
             '--model-focus-weight',str(.25 if a.stage in ('model','full') else 0)]
    if a.stage=='adapter': options+=['--plain-ce']
    # For focus-only controls keep eta ramp but neutralize data attenuation.
    if a.stage in ('dist','model'): options+=['--gamma','0']
    if a.data_root: options+=['--data-root',a.data_root]
    (root/f'{a.stage}_command.json').write_text(json.dumps(options,indent=2))
    if a.plan_only:
        print(json.dumps(options,indent=2)); return
    engine.UQ_VISUALS=root/'visuals'
    engine.main(options)

if __name__=='__main__': main()
