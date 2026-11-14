"""Import measured baseline artifacts into local MLflow; this does not retrain models."""
import argparse, hashlib, json
from pathlib import Path

def sha256(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def run(args):
    import mlflow
    root = args.run.resolve()
    selection = json.loads((root/'selection.json').read_text())
    config = json.loads((root/'config.json').read_text())
    metrics = json.loads((root/'evaluation/metrics.json').read_text())
    for kind, expected in selection['checkpoint_sha256'].items():
        if sha256(root/f'{kind}.pt') != expected:
            raise ValueError('Checkpoint integrity mismatch: '+kind)
    if metrics['manifest_sha256'] != selection['manifest_sha256']:
        raise ValueError('Evaluation and selection manifests differ')
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment('visionguard-measured-baseline-import')
    records=[]
    for kind in ['cnn','frozen','finetuned']:
        digest=selection['checkpoint_sha256'][kind]
        existing=mlflow.search_runs(filter_string=f"tags.checkpoint_sha256 = '{digest}'",output_format='list')
        if existing:
            records.append(dict(model=kind,run_id=existing[0].info.run_id,reused=True));continue
        with mlflow.start_run(run_name=kind+' - imported measured run') as tracking:
            mlflow.set_tags({'provenance':'historical_artifact_import_not_retraining','checkpoint_sha256':digest,'selected':str(kind==selection['selected_model'])})
            mlflow.log_params({'seed':config['seed'],'manifest_sha256':selection['manifest_sha256'],'model':kind,'selection_basis':'validation macro-F1'})
            mlflow.log_metrics({'test_macro_f1':metrics['models'][kind]['macro_f1'],'test_accuracy':metrics['models'][kind]['accuracy'],
                                'validation_macro_f1':selection['models'][kind]['validation']['macro_f1']})
            for name in ['config.json','selection.json',f'{kind}-history.json',f'{kind}.pt']:
                mlflow.log_artifact(str(root/name))
            mlflow.log_artifact(str(root/'evaluation/metrics.json'),'evaluation')
            if kind==selection['selected_model']:
                mlflow.log_artifact(str(root/'evaluation/confusion_matrix.png'),'evaluation')
            records.append(dict(model=kind,run_id=tracking.info.run_id,reused=False))
    print(json.dumps(records,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--tracking-uri',required=True);run(p.parse_args())
