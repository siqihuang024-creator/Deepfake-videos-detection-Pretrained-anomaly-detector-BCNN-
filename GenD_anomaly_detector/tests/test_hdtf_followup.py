"""Source-matched sampling, KL coefficients, mechanism algebra and full replay."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'scripts')]
from gad import hdtf_followup as f, source_sensitivity as sensitivity, segment_training, bayesian_svdd as bayes
from gad.feature_cache import sha256_file,model_state_hash,signature_hash
import run_hdtf_followup as pipeline
import run_source_sensitivity as source_run
import run_covariance_controls as covariance_run
import run_geometry_diagnostics as geometry_run
import run_segment_kl_ablation as kl_run
from run_segment_controls import run as controls_run
from test_feature_geometry import FrozenModel
import test_segment_training as fixtures


class HDTFFollowupTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def test_source_count_matching_is_order_stable_retains_clips_and_ignores_scores(self):
        records=[]
        for i in range(20):
            for k in range(1+i%2):
                records.append(dict(video_key='v%d_%d'%(i,k),source_clip='g%d'%i,head_role='gradient_fit',
                                    dataset='HDTF' if i<8 else 'TalkingHead-1KH',label=0))
        records.append(dict(video_key='early',source_clip='early',head_role='early_stop',dataset='HDTF',label=0))
        rows=pd.DataFrame(records)
        a,early,counts=f.source_profiles(rows,list(range(10)))
        self.assertEqual(len(a),13)
        self.assertEqual(counts['hdtf_sources'],8)
        for name,mask in a.items():
            if name!='mixed_reference': self.assertEqual(rows.loc[mask,'source_clip'].nunique(),8)
            for group in rows.loc[mask,'source_clip'].unique():
                self.assertEqual(int((mask & rows.source_clip.eq(group).to_numpy()).sum()),int(rows.source_clip.eq(group).sum()))
        self.assertEqual(int(early.sum()),1)
        shuffled=rows.sample(frac=1,random_state=9).reset_index(drop=True)
        b,_,_=f.source_profiles(shuffled,list(range(10)))
        for name in a:self.assertEqual(set(rows.loc[a[name],'video_key']),set(shuffled.loc[b[name],'video_key']))
        with self.assertRaises(ValueError):f.source_profiles(rows,[0,0])

    def test_kl_coefficient_matching_is_exact_and_native_factor_is_explicit(self):
        variants=f.kl_variants(352,162)
        self.assertEqual(len(variants),4)
        for v in variants.values():
            expected=v['base_beta']/(352 if v['scaling']=='matched_coefficient' else 162)
            self.assertAlmostEqual(v['kl_coefficient'],expected,places=16)
        self.assertAlmostEqual(variants['hdtf_bayes_b1e5_matched_coefficient']['actual_beta'],4.6022727272727275e-6)
        with self.assertRaises(ValueError):f.kl_variants(162,352)

    def test_direction_variance_decomposes_dataset_means_and_precision_is_direct(self):
        rng=np.random.default_rng(4)
        x=rng.normal(size=(12,5));groups=['g%d'%(i//2) for i in range(12)]
        datasets=['HDTF']*6+['TalkingHead-1KH']*6
        x[6:,0]+=4
        u=np.array([1.,0,0,0,0])
        heads,_=sensitivity.fit_family(x,groups)
        m=f.direction_mechanism(x,groups,datasets,u,heads)
        self.assertGreater(m['between_dataset_means_variance'],1)
        np.testing.assert_allclose(m['variance_along_classifier'],m['between_dataset_means_variance']+
            m['components']['HDTF_weighted_within_variance']+m['components']['TalkingHead-1KH_weighted_within_variance'])
        L=heads['mahalanobis_source_mean']['cholesky']
        self.assertAlmostEqual(m['precision_quadratic_u_P_u']['mahalanobis_source_mean'],float(u@np.linalg.inv(L@L.T)@u))

    def test_full_cached_pipeline_trains_all_fixed_heads_and_replays_saved_references(self):
        fixture=fixtures.SegmentTrainingTests();fixture.setUp();self.addCleanup(fixture.tearDown)
        base=fixture.rows.loc[fixture.rows.source_clip.eq('train0')].copy()
        additions=[]
        for i in range(6,20):
            part=base.copy()
            for key in ('source_clip','parent_video_key','video_key','rel_path'):
                part[key]=part[key].str.replace('train0','train%d'%i,regex=False).str.replace('train_0_','train_%d_'%i,regex=False)
            additions.append(part)
        fixture.rows=pd.concat([fixture.rows]+additions,ignore_index=True)
        # Find a metadata-only assignment with both source families and independent HDTF early stop.
        for offset in range(20):
            h={'train%d'%((i+offset)%20) for i in range(8)}
            mask=fixture.rows.split.eq('train')
            fixture.rows.loc[mask,'dataset']=np.where(fixture.rows.loc[mask,'source_clip'].isin(h),'HDTF','TalkingHead-1KH')
            fixture.rows.loc[fixture.rows.source_clip.eq('calibration0'),'dataset']='TalkingHead-1KH'
            units,_,_=fixture.choose()
            try:
                f.source_profiles(units,[0,1])
                break
            except ValueError:pass
        else:self.fail('Fixture cannot form valid fitting and early-stop sources')
        inputs=fixture.make_inputs();cache=inputs[1]
        models={'official':FrozenModel(7),'finetuned':FrozenModel(8)}
        for enc,model in models.items():
            receipt_path=cache/'cache_receipts'/(enc+'.json');receipt=json.loads(receipt_path.read_text())
            path=Path(receipt['cache'])/'metadata.json';metadata=json.loads(path.read_text())
            metadata['signature'].update(weights_sha256=model_state_hash(model),head_configuration='LinearNorm',
                official_revision='pinned' if enc=='official' else None)
            metadata['signature_sha256']=signature_hash(metadata['signature'])
            path.write_text(json.dumps(metadata),encoding='utf-8');receipt['metadata_sha256']=sha256_file(path)
            receipt_path.write_text(json.dumps(receipt),encoding='utf-8')
        ref,controls,kl,geometry,covariance,source,out=[fixture.root/n for n in ('ref','controls','kl','geometry','covariance','source','out')]
        with ExitStack() as stack:
            recipe=dict(segment_training.RECIPE)
            for name,value in {'WIDTHS':(4,2),'MAX_EPOCHS':2,'PATIENCE':2,'BATCH_SIZE':4,'VALIDATION_DRAWS':3,'POSTERIOR_DRAWS':4}.items():
                stack.enter_context(patch.object(bayes,name,value));recipe[name.lower()]=list(value) if name=='WIDTHS' else value
            stack.enter_context(patch.dict(segment_training.RECIPE,recipe))
            for module in (geometry_run,covariance_run,source_run,pipeline):stack.enter_context(patch.dict(module.RECIPE,bootstrap_draws=5))
            stack.enter_context(patch.dict(pipeline.RECIPE,subsample_seeds=[0,1]))
            def loader(model,device,revision):return models['official' if model=='official' else 'finetuned'],None
            with redirect_stdout(io.StringIO()):
                segment_training.run(*inputs,ref)
                controls_run(ref,*inputs,controls)
                kl_run.run(ref,controls,*inputs,kl)
                geometry_run.run(ref,controls,*inputs,geometry,loader=loader)
                covariance_run.run(ref,controls,geometry,*inputs,covariance)
                source_run.run(ref,controls,covariance,*inputs,source)
                pipeline.run(ref,controls,kl,geometry,covariance,source,*inputs,out)
            completion=json.loads((out/'completion.json').read_text());self.assertTrue(completion['completed'])
            for name,value in completion['artifacts_sha256'].items():self.assertEqual(sha256_file(out/name),value)
            reg=json.loads((out/'followup_registration.json').read_text())
            self.assertEqual(len(reg['profile_counts']),5)
            self.assertEqual(len(pd.read_csv(out/'statistical_comparison.csv')),100)
            self.assertEqual(len(pd.read_csv(out/'neural_comparison.csv')),32)
            for enc in models:
                folders=list((out/enc/'neural').iterdir());self.assertEqual(len(folders),5)
                for folder in folders:
                    with np.load(folder/'head.npz') as head:
                        self.assertEqual(head['first_mean'].shape[1],1024)
                        self.assertEqual(int(head['effective_train_groups']),reg['source_counts']['hdtf_sources'])
                        if folder.name in reg['neural_variants']:
                            self.assertAlmostEqual(float(head['kl_weight'])/int(head['effective_train_groups']),
                                reg['neural_variants'][folder.name]['kl_coefficient'],places=16)
            with self.assertRaises(FileExistsError):pipeline.run(ref,controls,kl,geometry,covariance,source,*inputs,out)
            (source/'comparison.csv').write_text('changed',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Sensitivity artifact changed'):
                pipeline.run(ref,controls,kl,geometry,covariance,source,*inputs,fixture.root/'reject')
            self.assertFalse((fixture.root/'reject').exists())
