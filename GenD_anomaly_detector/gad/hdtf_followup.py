"""Predeclared source-matched controls and KL-coefficient bookkeeping."""
import numpy as np
from scipy.linalg import solve_triangular

from .feature_geometry import matrix, source_means


BASE_BETAS = (1e-3, 1e-5)


def source_profiles(rows, seeds):
    fit = rows.head_role.eq("gradient_fit")
    group_datasets = rows.loc[fit].groupby("source_clip").dataset.agg(lambda v: tuple(sorted(set(v))))
    if any(len(v) != 1 or v[0] not in ("HDTF", "TalkingHead-1KH") for v in group_datasets):
        raise ValueError("Fitting source has inconsistent dataset metadata")
    h = sorted(g for g, v in group_datasets.items() if v == ("HDTF",))
    t = sorted(g for g, v in group_datasets.items() if v == ("TalkingHead-1KH",))
    target, total = len(h), len(h)+len(t)
    if target < 3 or len(t) < target or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Need at least three HDTF sources and enough TalkingHead sources for matching")
    h_count = int(round(target*len(h)/total))
    t_count = target-h_count
    if min(h_count,t_count) < 1:
        raise ValueError("Matched mixture must retain both datasets")
    selected = {"mixed_reference": h+t, "hdtf_reference": h,
                "talkinghead_matched": sorted(np.random.default_rng(0).choice(t,target,replace=False).tolist())}
    for seed in seeds:
        if type(seed) is not int or seed < 0:
            raise ValueError("Use distinct nonnegative integer source-sampling seeds")
        rng = np.random.default_rng(seed)
        selected["mixed_matched_seed%d" % seed] = sorted(
            rng.choice(h,h_count,replace=False).tolist()+rng.choice(t,t_count,replace=False).tolist())
    masks = {name: (fit & rows.source_clip.isin(groups)).to_numpy() for name,groups in selected.items()}
    early = (rows.head_role.eq("early_stop") & rows.dataset.eq("HDTF")).to_numpy()
    if not early.any() or set(rows.loc[early,"source_clip"]) & set(h):
        raise ValueError("Need original independent HDTF early-stop sources")
    return masks, early, {"mixed_sources": total, "hdtf_sources": target,
        "talkinghead_pool_sources": len(t), "matched_mixture_HDTF_sources": h_count,
        "matched_mixture_TalkingHead_sources": t_count}


def kl_variants(mixed_sources, hdtf_sources):
    if not 0 < hdtf_sources < mixed_sources:
        raise ValueError("Expected fewer positive HDTF fitting sources")
    variants = {}
    for base in BASE_BETAS:
        for scaling in ("native", "matched_coefficient"):
            actual = base if scaling == "native" else base*hdtf_sources/mixed_sources
            name = "hdtf_bayes_b%s_%s" % ("1e3" if base == 1e-3 else "1e5", scaling)
            variants[name] = {"base_beta": base, "scaling": scaling, "actual_beta": actual,
                              "kl_denominator": hdtf_sources, "kl_coefficient": actual/hdtf_sources}
    return variants


def direction_mechanism(features, groups, datasets, direction, heads):
    x = matrix(features)
    names, means = source_means(x, groups)
    u = np.asarray(direction,float)
    if u.shape != (x.shape[1],) or not np.isclose(np.linalg.norm(u),1):
        raise ValueError("Expected matching frozen unit classifier direction")
    labels = {}
    for group,dataset in zip(groups,datasets):
        if group in labels and labels[group] != dataset:
            raise ValueError("Source crosses datasets")
        labels[group] = dataset
    coordinate = means@u
    overall = float(np.var(coordinate))
    average = float(np.mean(np.var(means,axis=0)))
    components, subgroup_means = {}, {}
    for dataset in ("HDTF","TalkingHead-1KH"):
        subset = coordinate[[labels[g] == dataset for g in names]]
        components[dataset+"_weight"] = len(subset)/len(coordinate)
        components[dataset+"_weighted_within_variance"] = float(np.var(subset))*len(subset)/len(coordinate) if len(subset) else 0.
        subgroup_means[dataset] = float(subset.mean()) if len(subset) else 0.
    between = (components["HDTF_weight"]*components["TalkingHead-1KH_weight"]
               *(subgroup_means["HDTF"]-subgroup_means["TalkingHead-1KH"])**2)
    np.testing.assert_allclose(overall, between+components["HDTF_weighted_within_variance"]
                               +components["TalkingHead-1KH_weighted_within_variance"],atol=1e-12)
    precision = {name:float(np.sum(solve_triangular(head["cholesky"],u,lower=True)**2))
                 for name,head in heads.items() if "cholesky" in head}
    return {"covariance_scope":"One mean per source, biased between-source covariance; matches source-mean Mahalanobis scope",
        "variance_along_classifier":overall,"average_direction_variance":average,
        "variance_ratio":overall/average if average > 0 else None,
        "between_dataset_means_variance":between, "components":components,
        "precision_quadratic_u_P_u":precision,
        "limitation":"Descriptive mechanism, not causal proof; variance ratio and inverse-covariance weight are different quantities"}
