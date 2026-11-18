#!/usr/bin/env python3
"""Development-only excess-adjacency evaluation and stability analysis."""
import argparse
from datetime import datetime,timezone,timedelta
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
RESULTS=PROJECT/'results'


def load_module(name,path):
    """Import a Python module from an explicit file path."""
    import importlib.util
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sign(value):
    """Return the ordering sign used for pairwise transitions."""
    return (value>0)-(value<0)


def transitions(baseline,modified,quality,pairs):
    """Count corrections, damage and ties separately."""
    counts={a+'_to_'+b:0 for a in ('correct','wrong','tied') for b in ('correct','wrong','tied')}
    changed=0
    for i,j in pairs:
        old,new,truth=sign(baseline[i]-baseline[j]),sign(modified[i]-modified[j]),sign(quality[i]-quality[j])
        if truth==0:
            raise ValueError('Quality ties require an explicit additional outcome state')
        old_state='tied' if old==0 else 'correct' if old==truth else 'wrong'
        new_state='tied' if new==0 else 'correct' if new==truth else 'wrong'
        counts[old_state+'_to_'+new_state]+=1
        changed+=old!=new
    corrected=counts['wrong_to_correct'];damaged=counts['correct_to_wrong']
    old_wrong=sum(counts['wrong_to_'+b] for b in ('correct','wrong','tied'))
    old_correct=sum(counts['correct_to_'+b] for b in ('correct','wrong','tied'))
    new_correct=sum(counts[a+'_to_correct'] for a in ('correct','wrong','tied'))
    return dict(pairs=len(pairs),changed_pairs=changed,corrected=corrected,damaged=damaged,
                net_corrections=corrected-damaged,
                correction_precision=corrected/changed if changed else None,
                original_wrong_pairs=old_wrong,
                original_error_recovery_fraction=corrected/old_wrong if old_wrong else None,
                original_pair_accuracy=old_correct/len(pairs) if pairs else None,
                modified_pair_accuracy=new_correct/len(pairs) if pairs else None,**counts)


def check(script,starter):
    """Run an isolated check against the supplied course starter."""
    start=time.perf_counter()
    process=subprocess.run([sys.executable,'-B',str(script),'--starter',str(starter)],
                           capture_output=True,text=True,encoding='utf-8',errors='replace')
    return dict(script=str(script),returncode=process.returncode,
                elapsed_seconds=time.perf_counter()-start,stdout=process.stdout,stderr=process.stderr)


def main():
    """Evaluate the excess penalty, ablations and development stability."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--starter',type=Path,default=PROJECT/'course_starter')
    parser.add_argument('--output',type=Path,default=PROJECT/'outputs/manual')
    args=parser.parse_args();starter=args.starter.resolve()
    global RESULTS
    RESULTS=args.output.resolve()/'results'
    utils=load_module('v1_utilities',HERE/'run_adjacent.py')
    sys.addaudithook(utils.deny_final_labels)
    sys.path.append(str(starter))
    from nsc import layer_capacities,original_nsc,position_example,resource_estimate
    evaluator=load_module('v2_course_evaluator',starter/'evaluate.py')
    v1=load_module('v1_frozen_score',PROJECT/'src/adjacent_score.py')
    v2=load_module('v2_candidate_score',PROJECT/'src/excess_score.py')
    protocol_path=PROJECT/'protocols/excess.json'
    protocol_hash=hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    prior_result_hash=hashlib.sha256((RESULTS/'adjacent/development_metrics.json').read_bytes()).hexdigest()
    safe_before=utils.safe_input_hashes(starter)
    start=time.perf_counter()
    read=lambda name:json.loads((starter/'data'/name).read_text(encoding='utf-8'))
    configs,resources,splits=read('architectures.json'),read('resources.json'),read('splits.json')
    labels=read('development_labels.json');keys=splits['development']
    assert len(keys)==100 and set(keys)==set(labels) and not set(keys)&set(splits['final'])
    ppl=[labels[key]['valid_ppl'] for key in keys];quality=[-v for v in ppl]
    assert len(set(ppl))==len(ppl),'Transition diagnostics currently require distinct quality values'
    totals=[resources[key]['total'] for key in keys]
    pairs=[(i,j) for i in range(len(keys)) for j in range(i)]
    same_width=[(i,j) for i,j in pairs if configs[keys[i]]['d_model']==configs[keys[j]]['d_model']]
    matched=[(i,j) for i,j in same_width if min(totals[i],totals[j])/max(totals[i],totals[j])>=.90]
    baseline=[evaluator.canonical(original_nsc(configs[key])) for key in keys]
    margins={(i,j):abs(math.log(baseline[i]/baseline[j])) for i,j in pairs}
    subsets=dict(all=pairs,same_hidden_width=same_width,
                 different_hidden_width=[(i,j) for i,j in pairs if (i,j) not in set(same_width)],
                 size_controlled=matched,
                 near_1pct=[p for p in pairs if margins[p]<.01],
                 near_5pct=[p for p in pairs if margins[p]<.05],
                 near_1pct_and_size_controlled=[p for p in matched if margins[p]<.01])

    def ranking(values,indices=None):
        """Compute full-panel and size-controlled ranking metrics."""
        indices=list(range(len(keys))) if indices is None else list(indices)
        local_pairs=[(i,j) for pos,i in enumerate(indices) for j in indices[:pos]]
        local_matched=[(i,j) for i,j in local_pairs
                       if configs[keys[i]]['d_model']==configs[keys[j]]['d_model']
                       and min(totals[i],totals[j])/max(totals[i],totals[j])>=.90]
        return dict(spearman_rho=evaluator.correlation(evaluator.ranks([values[i] for i in indices]),
                                                      evaluator.ranks([quality[i] for i in indices])),
                    all_pairs=evaluator.pair_summary(values,quality,local_pairs),
                    size_controlled_pairs=evaluator.pair_summary(values,quality,local_matched))

    functions=dict(v1=v1.adjacent_score,global_cv2=v1.global_score,v2=v2.excess_score)
    grid_scores={};grid_rows=[]
    for mechanism,function in functions.items():
        for coefficient in protocol['grid']:
            values=[evaluator.canonical(function(configs[key],coefficient)) for key in keys]
            grid_scores[mechanism,coefficient]=values
            rank=ranking(values);effect=transitions(baseline,values,quality,pairs)
            grid_rows.append(dict(mechanism=mechanism,coefficient=coefficient,
                spearman_rho=rank['spearman_rho'],tau_b=rank['all_pairs']['tau_b'],
                controlled_tau_b=rank['size_controlled_pairs']['tau_b'],
                changed_pairs=effect['changed_pairs'],corrected=effect['corrected'],
                damaged=effect['damaged'],net_corrections=effect['net_corrections']))
    selected={name:max((row for row in grid_rows if row['mechanism']==name),
                      key=lambda row:(row['spearman_rho'],-row['coefficient']))['coefficient']
              for name in functions}
    assert selected['v1']==v1.SELECTED_LAMBDA
    prior=json.loads((RESULTS/'adjacent/development_metrics.json').read_text(encoding='utf-8'))
    assert selected['global_cv2']==prior['selected']['independently_selected_global_lambda']
    path=PROJECT/'src/excess_score.py'
    assert float(selected['v2']) == v2.SELECTED_LAMBDA, 'Selected coefficient differs from the frozen method'
    v2=load_module('v2_selected_score',path)
    methods=dict(original_nsc=original_nsc,position_example=position_example,v1_frozen=v1.score,
                 global_v1_lambda=v1.ablation,
                 global_tuned=lambda cfg:v1.global_score(cfg,selected['global_cv2']),
                 v2=v2.score,uncentered_v2_lambda=v2.ablation,
                 global_v2_lambda=lambda cfg:v1.global_score(cfg,selected['v2']))
    scores={name:[evaluator.canonical(function(configs[key])) for key in keys]
            for name,function in methods.items()}
    scores['reported_parameters']=totals
    scores['reference_flops']=[resource_estimate(configs[key])['reference_flops_per_token'] for key in keys]
    assert scores['v2']==grid_scores['v2',selected['v2']]
    rankings={name:ranking(values) for name,values in scores.items()}
    effects={name:{group:transitions(baseline,values,quality,group_pairs)
                   for group,group_pairs in subsets.items()} for name,values in scores.items()}
    bands=[]
    edges=protocol['margin_band_edges']
    for low,high in zip(edges,edges[1:]):
        upper=math.inf if high=='infinity' else high
        group=[p for p in pairs if low<=margins[p]<upper]
        for name,values in scores.items():
            bands.append(dict(method=name,log_margin_low=low,log_margin_high=high,
                              **transitions(baseline,values,quality,group)))

    official=evaluator.evaluate('development',path)
    mapping=dict(original_nsc='original_nsc',position_example='position_example',student='v2',
                 ablation='uncentered_v2_lambda',reported_parameters='reported_parameters',reference_flops='reference_flops')
    for off,name in mapping.items():assert official['ranking'][off]==rankings[name]
    for old,new in [('original_nsc','original_nsc'),('position_example','position_example'),
                    ('adjacent_nsc','v1_frozen'),('global_same_lambda','global_v1_lambda'),
                    ('global_tuned','global_tuned'),('reported_parameters','reported_parameters'),('reference_flops','reference_flops')]:
        assert prior['ranking'][old]==rankings[new]

    selections=[];trial_rows=[];summary_rows=[]
    for budget in evaluator.BUDGETS:
        pool=[i for i,total in enumerate(totals) if total<=budget];oracle=min(ppl[i] for i in pool)
        for k in evaluator.K_VALUES:
            if len(pool)<k:continue
            seed=202627+budget+k;stats={}
            for name in list(scores)+['random']:
                values=None if name=='random' else scores[name]
                result=evaluator.select_trials(pool,values,k,ppl,seed);stats[name]=result
                summary_rows.append(dict(parameter_cap=budget,k=k,eligible=len(pool),oracle_ppl=oracle,method=name,**result))
                rng=random.Random(seed);observed=[]
                for trial in range(evaluator.REPEATS):
                    if values is None:chosen=rng.sample(pool,k)
                    else:
                        chosen=list(pool);rng.shuffle(chosen);chosen.sort(key=lambda i:values[i],reverse=True);chosen=chosen[:k]
                    best=min(ppl[i] for i in chosen);observed.append(best)
                    trial_rows.append(dict(parameter_cap=budget,k=k,method=name,trial=trial,seed=seed,
                        chosen_ids=json.dumps([keys[i] for i in chosen]),chosen_ppl=json.dumps([ppl[i] for i in chosen]),
                        best_ppl=best,regret=best-oracle))
                assert statistics.mean(observed)==result['mean_best_ppl']
                assert statistics.pstdev(observed)==result['sd_best_ppl']
            selections.append(dict(parameter_cap=budget,k=k,eligible=len(pool),oracle_ppl=oracle,methods=stats))
    for actual,off in zip(selections,official['selection']):
        assert (actual['parameter_cap'],actual['k'])==(off['parameter_cap'],off['k'])
        for a,b in mapping.items():assert actual['methods'][b]==off['methods'][a]
        assert actual['methods']['random']==off['methods']['random']

    rng=random.Random(protocol['stability_analysis']['seed']);folds=[[] for _ in range(5)]
    for width in sorted({configs[key]['d_model'] for key in keys}):
        group=[i for i,key in enumerate(keys) if configs[key]['d_model']==width]
        rng.shuffle(group)
        for j,index in enumerate(group):folds[j%5].append(index)
    fold_rows=[];fold_grid=[];assignment=[]
    for fold,test in enumerate(folds):
        test=sorted(test);train=[i for i in range(len(keys)) if i not in set(test)]
        test_pairs=[(i,j) for pos,i in enumerate(test) for j in test[:pos]]
        assignment.extend(dict(architecture_id=keys[i],fold=fold,d_model=configs[keys[i]]['d_model']) for i in test)
        for name in ['original_nsc']+list(functions):
            coefficient=0
            if name=='original_nsc':values=baseline
            else:
                candidates=[]
                for coefficient in protocol['grid']:
                    values=grid_scores[name,coefficient]
                    rho=evaluator.correlation(evaluator.ranks([values[i] for i in train]),
                                             evaluator.ranks([quality[i] for i in train]))
                    candidates.append((rho,-coefficient,coefficient))
                    fold_grid.append(dict(fold=fold,mechanism=name,coefficient=coefficient,train_spearman_rho=rho))
                coefficient=max(candidates)[2];values=grid_scores[name,coefficient]
            rank=ranking(values,test);effect=transitions(baseline,values,quality,test_pairs)
            fold_rows.append(dict(fold=fold,method=name,train_architectures=len(train),test_architectures=len(test),
                selected_lambda=coefficient,spearman_rho=rank['spearman_rho'],tau_b=rank['all_pairs']['tau_b'],
                controlled_tau_b=rank['size_controlled_pairs']['tau_b'],controlled_pairs=rank['size_controlled_pairs']['pairs'],
                corrected=effect['corrected'],damaged=effect['damaged'],net_corrections=effect['net_corrections'],
                changed_pairs=effect['changed_pairs']))
    fold_summary={}
    for name in ['original_nsc']+list(functions):
        rows=[row for row in fold_rows if row['method']==name]
        rhos=[row['spearman_rho'] for row in rows];taus=[row['tau_b'] for row in rows]
        fold_summary[name]=dict(mean_fold_spearman=statistics.mean(rhos),sd_fold_spearman=statistics.pstdev(rhos),
            mean_fold_tau=statistics.mean(taus),selected_lambdas=[row['selected_lambda'] for row in rows],
            within_fold_corrected=sum(row['corrected'] for row in rows),within_fold_damaged=sum(row['damaged'] for row in rows))

    panel=[]
    for i,key in enumerate(keys):
        values=layer_capacities(configs[key]);stats=v2.excess_statistics(values)
        panel.append(dict(architecture_id=key,d_model=configs[key]['d_model'],n_layer=configs[key]['n_layer'],
            reported_parameters=totals[i],valid_ppl=ppl[i],layer_capacities=json.dumps(values),
            adjacent_mismatch=stats['adjacent'],global_cv2=stats['global_cv2'],
            random_order_reference=stats['random_order_reference'],excess_mismatch=stats['excess'],
            **{name:values_[i] for name,values_ in scores.items() if name!='reported_parameters'}))
    changed=[]
    for name in ('v1_frozen','v2','uncentered_v2_lambda','global_tuned'):
        values=scores[name]
        for i,j in pairs:
            old,new,truth=sign(baseline[i]-baseline[j]),sign(values[i]-values[j]),sign(quality[i]-quality[j])
            if old==new:continue
            a,b=panel[i],panel[j]
            changed.append(dict(method=name,id_a=keys[i],id_b=keys[j],width_a=a['d_model'],width_b=b['d_model'],
                original_log_margin=margins[i,j],size_controlled=(i,j) in set(matched),
                original_a=baseline[i],original_b=baseline[j],modified_a=values[i],modified_b=values[j],
                excess_a=a['excess_mismatch'],excess_b=b['excess_mismatch'],
                ppl_a=ppl[i],ppl_b=ppl[j],corrected=old!=truth and new==truth,damaged=old==truth and new!=truth))
    original_six=[]
    for row in changed:
        if row['method']!='v1_frozen':
            continue
        i,j=keys.index(row['id_a']),keys.index(row['id_b'])
        truth=sign(quality[i]-quality[j]);old=sign(baseline[i]-baseline[j])
        first=sign(scores['v1_frozen'][i]-scores['v1_frozen'][j])
        second=sign(scores['v2'][i]-scores['v2'][j])
        category=('保留纠正' if second==truth else '放弃纠正') if first==truth else ('避免原误伤' if second==truth else '仍然误伤')
        original_six.append(dict(id_a=keys[i],id_b=keys[j],v1_corrected=first==truth,
            original_correct=old==truth,v2_correct=second==truth,v2_changed_original=old!=second,
            v2_action=category,width_a=panel[i]['d_model'],width_b=panel[j]['d_model'],
            original_log_margin=margins[i,j],excess_a=panel[i]['excess_mismatch'],excess_b=panel[j]['excess_mismatch']))
    assert len(original_six)==6
    toys=[]
    for inner in ([512,512,2048,2048],[512,2048,512,2048],[2048,2048,512,512]):
        cfg=dict(d_model=128,n_layer=4,n_head=[4]*4,d_inner=inner)
        toys.append(dict(config=cfg,**v2.excess_statistics(layer_capacities(cfg)),
                         v1_lambda1=v1.adjacent_score(cfg,1),v2_lambda1=v2.excess_score(cfg,1),
                         v2_selected=v2.score(cfg),measured_perplexity=None))
    checks=[check(PROJECT/'tests/test_excess.py',starter),check(PROJECT/'tests/test_adjacent.py',starter),
            check(HERE/'check_starter_safe.py',starter)]
    output=RESULTS/'excess';output.mkdir(parents=True,exist_ok=True)
    utils.write_json(output/'checks.json',checks)
    assert all(c['returncode']==0 for c in checks),checks
    assert safe_before==utils.safe_input_hashes(starter)
    assert prior_result_hash==hashlib.sha256((RESULTS/'adjacent/development_metrics.json').read_bytes()).hexdigest()
    assert protocol_hash==hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    excess_values=[row['excess_mismatch'] for row in panel]
    result=dict(version='v2_exploratory',split='development',architectures=len(keys),selected=selected,
        protocol=protocol,ranking=rankings,pair_effects=effects,selection=selections,
        stability_summary=fold_summary,stability_folds=fold_rows,synthetic_controls=toys,
        original_six_followup=original_six,
        excess_diagnostics=dict(positive=sum(v>0 for v in excess_values),zero=sum(v==0 for v in excess_values),
            min=min(excess_values),max=max(excess_values),mean=statistics.mean(excess_values)),
        verification=dict(official_harness_exact_match=True,initial_results_unchanged=True,v1_baselines_reproduced=True,
            original_starter_unchanged=True,protocol_unchanged=True,controlled_tests_passed=True,final_label_open_guard=True),
        environment=utils.environment(),elapsed_seconds=time.perf_counter()-start,
        timestamp_local=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        interpretation='Development exploration after v1 outcomes; folds are stability checks, not fresh validation; no pooled cross-fold ranking.')
    utils.write_json(output/'development_metrics.json',result)
    utils.write_json(output/'official_development.json',official)
    utils.write_csv(output/'grid_search.csv',grid_rows)
    utils.write_csv(output/'development_scores.csv',panel)
    utils.write_csv(output/'pair_effects.csv',[dict(method=name,subgroup=group,**effect)
        for name,groups in effects.items() for group,effect in groups.items()])
    utils.write_csv(output/'margin_bands.csv',bands)
    utils.write_csv(output/'changed_pairs.csv',changed)
    utils.write_csv(output/'original_six_followup.csv',original_six)
    utils.write_csv(output/'selection_summary.csv',summary_rows)
    utils.write_csv(output/'selection_trials.csv',trial_rows)
    utils.write_csv(output/'fold_assignments.csv',assignment)
    utils.write_csv(output/'fold_grid.csv',fold_grid)
    utils.write_csv(output/'fold_results.csv',fold_rows)
    utils.write_csv(output/'ranking.csv',[dict(method=name,spearman_rho=rank['spearman_rho'],
        **{'all_'+key:value for key,value in rank['all_pairs'].items()},
        **{'controlled_'+key:value for key,value in rank['size_controlled_pairs'].items()}) for name,rank in rankings.items()])
    print('Selected:',selected)
    for name in ('original_nsc','v1_frozen','v2','global_tuned'):
        rank=rankings[name];effect=effects[name]['all']
        print(name,'rho',rank['spearman_rho'],'tau',rank['all_pairs']['tau_b'],
              'controlled_tau',rank['size_controlled_pairs']['tau_b'],
              'changed',effect['changed_pairs'],'corrected',effect['corrected'],'damaged',effect['damaged'])
    print('Excess diagnostic:',result['excess_diagnostics'])
    print('Stability:',fold_summary)
    print('Excess results:',output)
    print('Original starter and initial results preserved. Final labels were not opened.')


if __name__=='__main__':
    main()
