"""One CPU job: source-matched statistical controls and HDTF-only neural heads."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gad import hdtf_followup as followup, source_sensitivity as sensitivity, segment_training as original
from gad import bayesian_svdd as bayes, bayesian_kl_ablation as ablation, svdd_controls as point, old_metrics as metrics
from gad.diagnostic_protocol import sha256_file
from run_segment_controls import load_reference
from run_segment_kl_ablation import verify_controls
from run_covariance_controls import verify_geometry
from run_source_sensitivity import verify_completed_covariance
from run_geometry_diagnostics import wilson

RECIPE = {"subsample_seeds":list(range(10)),"talkinghead_matched_seed":0,
    "subsampling":"Match HDTF source count; proportional HDTF/TalkingHead stratified source sampling, retain all original clips inside selected sources",
    "betas":list(followup.BASE_BETAS),"neural_seed":42,
    "KL_scaling":["native beta/G_HDTF","same beta/G_mixed coefficient, implemented beta_HDTF=beta*G_HDTF/G_mixed"],
    "early_stop":"Only original HDTF early_stop sources; original role assignment unchanged",
    "calibration":"Keep original mixed genuine calibration, 95%/99% linear quantiles; no target-domain threshold tuning",
    "neural":"Four HDTF Bayesian fits and one no-prior deterministic SVDD per encoder; same architecture/optimizer/stopping as archived algorithms",
    "references":"Replay mixed beta1e-3, beta1e-5 and deterministic heads without retraining",
    "bootstrap_draws":500,"final_test":"forbidden",
    "limitations":["Exploratory repeated development analysis, not independent test evidence.",
        "Source sampling does not establish actor independence; ten subsets overlap and are not ten independent test datasets.",
        "Matched source count does not perfectly isolate domain, image quality, within-source variation or early-stop composition.",
        "Matching a KL coefficient does not match gradients or optimizer trajectories.",
        "Retaining mixed calibration is a comparison condition, not proof of cross-domain FPR control."]}


def verify_sensitivity(directory,config,covariance_hash):
    directory = Path(directory)
    reg = original.read_json(directory/"sensitivity_registration.json")
    complete = original.read_json(directory/"completion.json")
    if (not complete.get("completed") or complete.get("phase") != "development_source_sensitivity"
            or complete["registration_sha256"] != sha256_file(directory/"sensitivity_registration.json")
            or reg["reference_configuration"] != config or reg["covariance_registration_sha256"] != covariance_hash):
        raise ValueError("Sensitivity reference does not match the completed inputs")
    for name,expected in complete["artifacts_sha256"].items():
        if sha256_file(directory/name) != expected: raise ValueError("Sensitivity artifact changed: "+name)
    for name,expected in reg["algorithm_sha256"].items():
        base = Path(__file__).parent if name.startswith("run_") else Path(followup.__file__).parent
        if sha256_file(base/name) != expected: raise ValueError("Sensitivity algorithm changed: "+name)
    return sha256_file(directory/"sensitivity_registration.json")


def load_kl_reference(directory,config,controls_hash):
    directory = Path(directory)
    reg_path = directory/"kl_registration.json"
    reg = original.read_json(reg_path)
    if (original.read_json(directory/"completion.json") != {"completed":True,"phase":"development_kl_ablation",
            "encoders":["official","finetuned"],"kl_weights":[1e-4,1e-5]}
            or reg["reference_configuration"] != config or reg["controls_registration_sha256"] != controls_hash):
        raise ValueError("Use the matching completed KL reference 317")
    for name,expected in reg["algorithm_sha256"].items():
        base = Path(__file__).parent if name.startswith("run_") else Path(ablation.__file__).parent
        if sha256_file(base/name) != expected: raise ValueError("KL reference algorithm changed: "+name)
    heads = {}
    for enc in ("official","finetuned"):
        folder = directory/enc/"kl_1e-5"
        lock = original.read_json(folder/"kl_lock.json")
        if (lock["registration_sha256"] != sha256_file(reg_path) or lock["head_sha256"] != sha256_file(folder/"head.npz")
                or lock["report_sha256"] != sha256_file(folder/"report.json")):
            raise ValueError("KL reference head/report changed")
        with np.load(folder/"head.npz",allow_pickle=False) as saved: heads[enc] = dict(saved)
    return heads,sha256_file(reg_path)


def evaluate(folder,units,score_map,cal_mask,dev_mask,encoder,profile,summary):
    scored = units.copy()
    reports = {}
    dev = units.loc[dev_mask].reset_index(drop=True)
    for name,values in score_map.items():
        values = np.asarray(values,float)
        if values.shape != (len(units),) or not np.isfinite(values).all(): raise ValueError("Invalid scores")
        scored[name] = values
        cal,development = values[cal_mask],values[dev_mask]
        interval = metrics.clustered_auroc_interval(dev.label,development,dev.source_clip,draws=RECIPE["bootstrap_draws"])
        reports[name] = {}
        for alpha in (.05,.01):
            threshold = float(np.quantile(cal,1-alpha))
            point_metrics = metrics.detection_metrics((dev.label==0).astype(int),development,threshold,alpha)
            point_metrics["roc_interpolated_tpr_at_nominal_fpr"] = point_metrics.pop("tpr_at_target_fpr")
            point_metrics["calibration_observed_fpr"] = float((cal>=threshold).mean())
            real = dev.label.eq(0).to_numpy()
            point_metrics["dev_real_fpr_wilson95"] = wilson(int((development[real]>=threshold).sum()),int(real.sum()))
            reports[name][str(alpha)] = point_metrics
            summary.append({"encoder":encoder,"profile":profile,"head":name,"nominal_fpr":alpha,
                "auc_ci_low":interval["low"],"auc_ci_high":interval["high"],
                **{k:v for k,v in point_metrics.items() if np.isscalar(v) or v is None}})
    scored.to_csv(folder/"scores.csv",index=False)
    original.write_json(folder/"report.json",{"heads":reports})


def replay_neural(directory,enc,scored,mask,filename,column,name):
    previous = pd.read_csv(Path(directory)/enc/filename,keep_default_na=False).set_index("video_key")
    np.testing.assert_allclose(scored[mask],previous.loc[name,column],rtol=1e-5,atol=1e-7)


def run(reference,controls,kl_run,geometry_run,covariance_run,sensitivity_run,data,cache_run,selection,out):
    out = Path(out)
    if out.exists(): raise FileExistsError("Use a new HDTF follow-up output directory")
    units,caches,config,old_heads,_,_ = load_reference(reference,data,cache_run,selection)
    control_hash = verify_controls(controls,reference,config)
    covariance_hash = verify_completed_covariance(covariance_run,config)
    sensitivity_hash = verify_sensitivity(sensitivity_run,config,covariance_hash)
    _,directions,_ = verify_geometry(geometry_run,config,units,control_hash)
    cov_reg = original.read_json(Path(covariance_run)/"covariance_registration.json")
    if cov_reg["geometry_registration_sha256"] != sha256_file(Path(geometry_run)/"geometry_registration.json"):
        raise ValueError("Classifier directions are from a different geometry run")
    old_low,kl_hash = load_kl_reference(kl_run,config,control_hash)
    control_reg = original.read_json(Path(controls)/"control_registration.json")
    if control_reg["algorithm_sha256"]["svdd_controls.py"] != sha256_file(point.__file__):
        raise ValueError("Deterministic control algorithm changed")
    sensitivity.selection_masks(units)
    profiles,early,counts = followup.source_profiles(units,RECIPE["subsample_seeds"])
    variants = followup.kl_variants(counts["mixed_sources"],counts["hdtf_sources"])
    cal,dev = units.head_role.eq("calibration").to_numpy(),units.head_role.eq("dev").to_numpy()
    evaluation = cal|dev
    eval_units = units.loc[evaluation].reset_index(drop=True)
    ecal,edev = eval_units.head_role.eq("calibration").to_numpy(),eval_units.head_role.eq("dev").to_numpy()
    selection_rows = []
    for name,mask in profiles.items():
        part = units.loc[mask].copy();part["profile"] = name;selection_rows.append(part)
    part = units.loc[early].copy();part["profile"] = "hdtf_neural_early_stop";selection_rows.append(part)
    out.mkdir(parents=True)
    units.to_csv(out/"training_units.csv",index=False)
    pd.concat(selection_rows,ignore_index=True).to_csv(out/"selected_units.csv",index=False)
    profile_counts = {name:{"clips":int(mask.sum()),"sources":units.loc[mask,"source_clip"].nunique(),
        "datasets":units.loc[mask].groupby("dataset").source_clip.nunique().to_dict()} for name,mask in profiles.items()}
    registration = {"phase":"development_hdtf_followup","recipe":RECIPE,"reference_configuration":config,
        "controls_registration_sha256":control_hash,"KL_registration_sha256":kl_hash,
        "covariance_registration_sha256":covariance_hash,"sensitivity_registration_sha256":sensitivity_hash,
        "geometry_registration_sha256":sha256_file(Path(geometry_run)/"geometry_registration.json"),
        "training_units_sha256":sha256_file(out/"training_units.csv"),"selected_units_sha256":sha256_file(out/"selected_units.csv"),
        "source_counts":counts,"profile_counts":profile_counts,"neural_variants":variants,
        "early_stop":{"clips":int(early.sum()),"sources":units.loc[early,"source_clip"].nunique()},
        "algorithm_sha256":{Path(p).name:sha256_file(p) for p in [__file__,followup.__file__,sensitivity.__file__,ablation.__file__,point.__file__,bayes.__file__]},
        "libraries":{"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__,"torch":torch.__version__},
        "statistical_parameters":"Deterministic source selections, scores and mechanism statistics saved; covariance matrices can be refit from locked features and are not duplicated for random subsets",
        "limitations":config["limitations"]+RECIPE["limitations"]}
    original.write_json(out/"followup_registration.json",registration)
    stats_summary,neural_summary,mechanisms = [],[],[]
    for enc in ("official","finetuned"):
        print("=== HDTF follow-up: "+enc+" ===",flush=True)
        folder = out/enc;folder.mkdir()
        cache = caches[enc];lookup = {k:i for i,k in enumerate(cache["video_keys"])}
        order = [lookup[k] for k in units.video_key]
        x,probs = cache["features"][order],cache["fake_prob"][order]
        for name,mask in profiles.items():
            print("Statistics: "+name,flush=True)
            target = folder/"statistics"/name;target.mkdir(parents=True)
            groups = units.loc[mask,"source_clip"].tolist()
            heads,_ = sensitivity.fit_family(x[mask],groups)
            scores = sensitivity.score_family(x,heads,probs)
            if name in ("mixed_reference","hdtf_reference"):
                old_profile = "mixed" if name=="mixed_reference" else "hdtf_only"
                previous = pd.read_csv(Path(sensitivity_run)/enc/(old_profile+"_all_unit_scores.csv"),keep_default_na=False)
                if previous.video_key.tolist()!=units.video_key.tolist(): raise ValueError("Sensitivity score order changed")
                for method,values in scores.items(): np.testing.assert_allclose(values,previous[method],rtol=1e-6,atol=1e-7)
            evaluate(target,eval_units,{k:v[evaluation] for k,v in scores.items()},ecal,edev,enc,name,stats_summary)
            mechanism = followup.direction_mechanism(x[mask],groups,units.loc[mask,"dataset"].tolist(),directions[enc+"/direction"],heads)
            mechanism.update(encoder=enc,profile=name);mechanisms.append(mechanism)
        with np.load(Path(controls)/enc/"control_heads.npz",allow_pickle=False) as values:
            old_point = {k.split('/',1)[1]:values[k] for k in values.files if k.startswith('deterministic_svdd/')}
        reference_scores = {"mixed_bayes_b1e3":bayes.score(x[evaluation],old_heads[enc],"cpu")[0],
            "mixed_bayes_b1e5":bayes.score(x[evaluation],old_low[enc],"cpu")[0],
            "mixed_deterministic_svdd":point.score_deterministic(x[evaluation],old_point)}
        for role,mask,filename in (("calibration",ecal,"calibration_scores.csv"),("dev",edev,"development_scores.csv")):
            keys = eval_units.loc[mask,"video_key"]
            replay_neural(reference,enc,reference_scores["mixed_bayes_b1e3"],mask,filename,"bayesian_deep_svdd_score",keys)
            replay_neural(Path(kl_run)/enc/"kl_1e-5","",reference_scores["mixed_bayes_b1e5"],mask,filename,"score",keys)
            replay_neural(controls,enc,reference_scores["mixed_deterministic_svdd"],mask,filename,"deterministic_svdd",keys)
        ref_folder = folder/"neural_references";ref_folder.mkdir()
        evaluate(ref_folder,eval_units,reference_scores,ecal,edev,enc,"mixed_neural_references",neural_summary)
        fit = profiles["hdtf_reference"]
        groups,early_groups = units.loc[fit,"source_clip"].tolist(),units.loc[early,"source_clip"].tolist()
        for name,variant in list(variants.items())+[("hdtf_deterministic_svdd",None)]:
            print("Neural fit: "+name,flush=True)
            target = folder/"neural"/name;target.mkdir(parents=True)
            def progress(event):
                with (target/"progress.jsonl").open('a',encoding='utf-8') as f: f.write(json.dumps(event)+'\n')
                print("epoch={epoch}, best={best_epoch}, early={early_stop_distance:.6g}".format(**event),flush=True)
            if variant is None:
                head = point.fit_deterministic(x[fit],x[early],groups,early_groups,progress=progress)
                values = point.score_deterministic(x[evaluation],head)
                posterior = None
            else:
                head = ablation.fit(x[fit],x[early],groups,early_groups,variant["actual_beta"],progress=progress)
                np.testing.assert_allclose(float(head["kl_weight"])/int(head["effective_train_groups"]),variant["kl_coefficient"],rtol=1e-12)
                values,extras = bayes.score(x[evaluation],head,"cpu")
                posterior = extras["posterior_distance_variance"]
            np.savez_compressed(target/"head.npz",**head)
            np.savetxt(target/"training_history.csv",head["training_history"],delimiter=',',
                header=','.join(ablation.HISTORY_COLUMNS) if variant else 'epoch,total_loss,data_loss,regularizer,early_stop_distance,elapsed_seconds',comments='')
            evaluate(target,eval_units,{name:values},ecal,edev,enc,"hdtf_neural",neural_summary)
            if posterior is not None:
                extra = eval_units[["video_key","head_role"]].copy();extra["posterior_distance_variance"] = posterior
                extra.to_csv(target/"posterior_variance.csv",index=False)
            diagnostic = {k:np.asarray(head[k]).item() for k in ("best_epoch","epochs_run","validation_distance","training_seconds",
                "effective_train_groups","early_stop_source_groups","train_score_relative_std","train_embedding_variance_ratio",
                "near_constant_train_scores","low_train_embedding_variance")}
            original.write_json(target/"training_diagnostics.json",{"KL_variant":variant,"training":diagnostic})
    stats_table = pd.DataFrame(stats_summary)
    stats_table.to_csv(out/"statistical_comparison.csv",index=False)
    repeated = stats_table.loc[stats_table.profile.str.startswith("mixed_matched_seed")]
    aggregates = []
    for keys,part in repeated.groupby(["encoder","head","nominal_fpr"]):
        record = dict(zip(("encoder","head","nominal_fpr"),keys),subsets=len(part))
        for column in ("auroc","observed_false_positive_rate","observed_true_positive_rate"):
            record.update({column+"_mean":float(part[column].mean()),column+"_std_across_subsets":float(part[column].std(ddof=1)),
                           column+"_min":float(part[column].min()),column+"_max":float(part[column].max())})
        aggregates.append(record)
    pd.DataFrame(aggregates).to_csv(out/"subsample_summary.csv",index=False)
    pd.DataFrame(neural_summary).to_csv(out/"neural_comparison.csv",index=False)
    pd.concat([pd.DataFrame(stats_summary).assign(stage="statistics"),pd.DataFrame(neural_summary).assign(stage="neural")],ignore_index=True).to_csv(out/"comparison.csv",index=False)
    original.write_json(out/"direction_mechanisms.json",{"records":mechanisms})
    artifacts = {p.relative_to(out).as_posix():sha256_file(p) for p in out.rglob('*') if p.is_file() and p.name!='followup_registration.json'}
    original.write_json(out/"completion.json",{"completed":True,"phase":"development_hdtf_followup",
        "registration_sha256":sha256_file(out/"followup_registration.json"),"artifacts_sha256":artifacts})
    print("All fixed source and neural controls completed. Final test remains unopened.",flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference","controls","kl-run","geometry-run","covariance-run","sensitivity-run","data","cache-run","selection","out"):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    run(args.reference,args.controls,args.kl_run,args.geometry_run,args.covariance_run,args.sensitivity_run,args.data,args.cache_run,args.selection,args.out)
