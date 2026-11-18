#!/usr/bin/env python3
"""Development-only tuning and evaluation; final labels are blocked."""
import argparse
import copy
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT/'results'


def write_json(path, value):
    """Write finite JSON results in a reproducible UTF-8 representation."""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                    encoding='utf-8')


def write_csv(path, rows):
    """Export rows with a stable header and UTF-8 encoding."""
    rows = list(rows)
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_module(name, path):
    """Import a Python module from an explicit file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_input_hashes(starter):
    # Enumerate inputs to avoid opening final labels.
    """Hash permitted starter inputs without accessing final labels."""
    names = ['nsc.py', 'student_score.py', 'evaluate.py', 'check_starter.py',
             'README.md', 'expected_development.json', 'data/architectures.json',
             'data/resources.json', 'data/splits.json',
             'data/development_labels.json', 'data/provenance.json']
    return {name: hashlib.sha256((starter / name).read_bytes()).hexdigest() for name in names}


def deny_final_labels(event, args):
    """Reject final-label file access during development."""
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        name = os.fsdecode(args[0]).replace('\\', '/').rsplit('/', 1)[-1].lower()
        if name == 'final_labels.json':
            raise PermissionError('Final-label access is forbidden during development')


def environment():
    """Record Python, operating system and available hardware information."""
    result = dict(python=sys.version, os=platform.platform(), machine=platform.machine(),
                  logical_processors=os.cpu_count(), processor=platform.processor())
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            result['processor_model'] = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
    except (ImportError, OSError):
        result['processor_model'] = 'Unavailable'
    try:
        import ctypes
        class Memory(ctypes.Structure):
            """Represent Windows physical-memory fields for environment reporting."""
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ('total_phys', 'avail_phys', 'total_page', 'avail_page',
                 'total_virtual', 'avail_virtual', 'avail_extended')]
        memory = Memory()
        memory.length = ctypes.sizeof(memory)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            result['physical_memory_bytes'] = memory.total_phys
    except (AttributeError, OSError):
        pass
    return result


def run_check(script, starter):
    """Run a development check and capture its status and output."""
    start = time.perf_counter()
    process = subprocess.run([sys.executable, '-B', str(PROJECT / ('tests' if script.startswith('test_') else 'scripts') / script),
                              '--starter', str(starter)], cwd=PROJECT,
                             capture_output=True, text=True, encoding='utf-8', errors='replace')
    return dict(script=script, returncode=process.returncode,
                elapsed_seconds=time.perf_counter() - start,
                stdout=process.stdout, stderr=process.stderr)


def main():
    """Reproduce development tuning, ranking, selection and frozen artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--starter', type=Path, default=PROJECT / 'course_starter')
    parser.add_argument('--output', type=Path, default=PROJECT/'outputs/manual')
    args = parser.parse_args()
    global RESULTS
    RESULTS = args.output.resolve()/'results'
    starter = args.starter.resolve()
    if starter == PROJECT:
        raise ValueError('Starter and project must be separate directories')
    sys.addaudithook(deny_final_labels)
    sys.path.append(str(starter))
    from nsc import layer_capacities, original_nsc, position_example, resource_estimate
    evaluator = load_module('course_evaluator', starter / 'evaluate.py')
    student = load_module('project_student', PROJECT / 'src/adjacent_score.py')
    read = lambda name: json.loads((starter / 'data' / name).read_text(encoding='utf-8'))
    protocol = json.loads((PROJECT / 'protocols/adjacent.json').read_text(encoding='utf-8'))
    before_hashes = safe_input_hashes(starter)
    wall_start = time.perf_counter()
    configs, resources, splits = read('architectures.json'), read('resources.json'), read('splits.json')
    labels = read('development_labels.json')
    keys = splits['development']
    assert len(keys) == len(splits['final']) == 100
    assert not set(keys) & set(splits['final'])
    assert set(labels) == set(keys)
    assert set(keys) | set(splits['final']) == set(configs)
    ppl = [labels[key]['valid_ppl'] for key in keys]
    assert all(math.isfinite(value) and value > 0 for value in ppl)
    quality = [-value for value in ppl]
    total = [resources[key]['total'] for key in keys]
    pairs = [(i, j) for i in range(len(keys)) for j in range(i)]
    matched = [(i, j) for i, j in pairs
               if configs[keys[i]]['d_model'] == configs[keys[j]]['d_model']
               and min(total[i], total[j]) / max(total[i], total[j]) >= .90]

    def ranking(values):
        """Compute full-panel and size-controlled ranking metrics."""
        return dict(spearman_rho=evaluator.correlation(evaluator.ranks(values), evaluator.ranks(quality)),
                    all_pairs=evaluator.pair_summary(values, quality, pairs),
                    size_controlled_pairs=evaluator.pair_summary(values, quality, matched))

    grid_rows, grid_scores = [], {}
    for mechanism, function in [('adjacent', student.adjacent_score), ('global', student.global_score)]:
        for coefficient in protocol['grid']:
            values = [evaluator.canonical(function(copy.deepcopy(configs[key]), coefficient)) for key in keys]
            stats = ranking(values)
            grid_scores[mechanism, coefficient] = values
            grid_rows.append(dict(mechanism=mechanism, coefficient=coefficient,
                                  spearman_rho=stats['spearman_rho'],
                                  tau_b=stats['all_pairs']['tau_b'],
                                  controlled_tau_b=stats['size_controlled_pairs']['tau_b'],
                                  score_tied_pairs=stats['all_pairs']['score_only_ties'] + stats['all_pairs']['both_ties']))
    def select_coefficient(mechanism):
        """Select the maximum-Spearman grid value, preferring smaller ties."""
        candidates = [row for row in grid_rows if row['mechanism'] == mechanism]
        return max(candidates, key=lambda row: (row['spearman_rho'], -row['coefficient']))['coefficient']
    adjacent_lambda, global_lambda = select_coefficient('adjacent'), select_coefficient('global')
    score_path = PROJECT / 'src/adjacent_score.py'
    assert float(adjacent_lambda) == student.SELECTED_LAMBDA, 'Selected coefficient differs from the frozen method'
    student = load_module('frozen_project_student', score_path)
    selected = dict(adjacent_lambda=adjacent_lambda, independently_selected_global_lambda=global_lambda,
                    ablation_lambda=adjacent_lambda, objective=protocol['selection_objective'],
                    tie_break=protocol['tie_break'], grid=protocol['grid'])

    checks = [run_check('test_adjacent.py', starter), run_check('check_starter_safe.py', starter)]
    if any(check['returncode'] != 0 for check in checks):
        write_json(args.output / 'failed_checks.json', checks)
        raise RuntimeError('Checks failed; see failed_checks.json')

    methods = dict(original_nsc=original_nsc, position_example=position_example,
                   adjacent_nsc=student.score, global_same_lambda=student.ablation,
                   global_tuned=lambda cfg: student.global_score(cfg, global_lambda))
    scores, timings = {}, {}
    for name, function in methods.items():
        start = time.perf_counter()
        scores[name] = [evaluator.canonical(function(copy.deepcopy(configs[key]))) for key in keys]
        timings[name] = time.perf_counter() - start
    scores['reported_parameters'] = total
    scores['reference_flops'] = [resource_estimate(configs[key])['reference_flops_per_token'] for key in keys]
    rankings = {name: ranking(values) for name, values in scores.items()}
    assert scores['adjacent_nsc'] == grid_scores['adjacent', adjacent_lambda]
    assert scores['global_same_lambda'] == grid_scores['global', adjacent_lambda]
    assert scores['global_tuned'] == grid_scores['global', global_lambda]

    official = evaluator.evaluate('development', score_path)
    names = dict(original_nsc='original_nsc', position_example='position_example',
                 student='adjacent_nsc', ablation='global_same_lambda',
                 reported_parameters='reported_parameters', reference_flops='reference_flops')
    for official_name, name in names.items():
        assert official['ranking'][official_name] == rankings[name]

    selections, trial_rows, summary_rows = [], [], []
    for budget in evaluator.BUDGETS:
        pool = [i for i, size in enumerate(total) if size <= budget]
        oracle = min(ppl[i] for i in pool)
        for k in evaluator.K_VALUES:
            if len(pool) < k:
                continue
            seed = 202627 + budget + k
            results = {}
            for name in list(scores) + ['random']:
                values = None if name == 'random' else scores[name]
                stats = evaluator.select_trials(pool, values, k, ppl, seed)
                results[name] = stats
                summary_rows.append(dict(parameter_cap=budget, k=k, eligible=len(pool), oracle_ppl=oracle,
                                         method=name, **stats))
                rng = random.Random(seed)
                observed = []
                for trial in range(evaluator.REPEATS):
                    if values is None:
                        chosen = rng.sample(pool, k)
                    else:
                        chosen = list(pool)
                        rng.shuffle(chosen)
                        chosen.sort(key=lambda i: values[i], reverse=True)
                        chosen = chosen[:k]
                    best = min(ppl[i] for i in chosen)
                    observed.append(best)
                    trial_rows.append(dict(parameter_cap=budget, k=k, method=name, trial=trial,
                                           seed=seed, chosen_ids=json.dumps([keys[i] for i in chosen]),
                                           chosen_ppl=json.dumps([ppl[i] for i in chosen]),
                                           best_ppl=best, regret=best-oracle))
                assert statistics.mean(observed) == stats['mean_best_ppl']
                assert statistics.pstdev(observed) == stats['sd_best_ppl']
            selections.append(dict(parameter_cap=budget, k=k, eligible=len(pool), oracle_ppl=oracle, methods=results))
    for entry, off in zip(selections, official['selection']):
        assert (entry['parameter_cap'], entry['k'], entry['eligible'], entry['oracle_ppl']) == (
            off['parameter_cap'], off['k'], off['eligible'], off['oracle_ppl'])
        for official_name, name in names.items():
            assert entry['methods'][name] == off['methods'][official_name]
        assert entry['methods']['random'] == off['methods']['random']

    expected = json.loads((starter / 'expected_development.json').read_text())
    for name in ('original_nsc', 'position_example', 'reported_parameters', 'reference_flops'):
        assert rankings[name] == expected['ranking'][name]
    reference_selection_matches = True
    for entry, ref in zip(selections, expected['selection']):
        for name in ('original_nsc', 'position_example', 'reported_parameters', 'reference_flops', 'random'):
            for metric, value in entry['methods'][name].items():
                reference_selection_matches &= math.isclose(value, ref['methods'][name][metric], rel_tol=1e-12, abs_tol=1e-12)
    assert reference_selection_matches

    panel, diagnostics = [], []
    for i, key in enumerate(keys):
        capacities = layer_capacities(configs[key])
        imbalance = student.capacity_statistics(capacities)
        diagnostics.append(imbalance)
        panel.append(dict(architecture_id=key, d_model=configs[key]['d_model'], n_layer=configs[key]['n_layer'],
                          n_head=json.dumps(configs[key]['n_head']), d_inner=json.dumps(configs[key]['d_inner']),
                          reported_parameters=total[i], reference_flops=scores['reference_flops'][i],
                          layer_capacities=json.dumps(capacities), adjacent_mismatch=imbalance['adjacent'],
                          global_cv2=imbalance['global_cv2'], valid_ppl=ppl[i],
                          **{name: scores[name][i] for name in methods}))
    def describe(values):
        """Summarize a development statistic without changing the scoring rule."""
        return dict(min=min(values), median=statistics.median(values), max=max(values),
                    mean=statistics.mean(values), sd=statistics.pstdev(values),
                    positive=sum(value > 0 for value in values),
                    distinct_12sig=len(set(evaluator.canonical(value) for value in values)))
    d_adj = [item['adjacent'] for item in diagnostics]
    cv2 = [item['global_cv2'] for item in diagnostics]
    sensitivity = dict(adjacent=describe(d_adj), global_cv2=describe(cv2),
                       adjacent_global_spearman=evaluator.correlation(evaluator.ranks(d_adj), evaluator.ranks(cv2)),
                       by_hidden_width={str(d): describe([d_adj[i] for i, key in enumerate(keys) if configs[key]['d_model']==d])
                                        for d in sorted({configs[key]['d_model'] for key in keys})},
                       selected_vs_original=evaluator.pair_summary(scores['adjacent_nsc'], scores['original_nsc'], pairs))
    toys = []
    for inner in ([512, 512, 2048, 2048], [512, 2048, 512, 2048], [2048, 2048, 512, 512]):
        cfg = dict(d_model=128, n_layer=4, n_head=[4]*4, d_inner=inner)
        values = layer_capacities(cfg)
        toys.append(dict(config=cfg, layer_capacities=values, **student.capacity_statistics(values),
                         adjacent_lambda1=student.adjacent_score(cfg, 1),
                         global_lambda1=student.global_score(cfg, 1),
                         adjacent_selected=student.score(cfg), ablation_selected=student.ablation(cfg),
                         resources=resource_estimate(cfg), measured_perplexity=None))
    after_hashes = safe_input_hashes(starter)
    assert before_hashes == after_hashes, 'Original starter inputs changed'
    results = dict(split='development', label='valid_ppl', architectures=len(keys),
                   selected=selected, protocol=protocol, ranking=rankings, selection=selections,
                   sensitivity=sensitivity, synthetic_controls=toys,
                   verification=dict(controlled_tests_passed=True, starter_safe_checks_passed=True,
                                     official_harness_exact_match=True, expected_baselines_match=True,
                                     expected_selection_match_tolerance=1e-12, starter_unchanged=True,
                                     final_label_hash_check_skipped=True, final_label_open_guard=True),
                   scoring_seconds=timings,
                   timing_note='Diagnostic only: MP cache is shared and warmed by tuning; not a cold-start comparison.',
                   environment=environment(), development_pipeline_seconds=time.perf_counter()-wall_start,
                   timestamp_local=datetime.now(timezone(timedelta(hours=8))).isoformat())
    output = RESULTS / 'adjacent'
    output.mkdir(parents=True,exist_ok=True)
    write_json(output / 'development_metrics.json', results)
    write_json(output / 'grid_search.json', dict(protocol=protocol, selected=selected, candidates=grid_rows))
    write_json(output / 'official_development.json', official)
    write_json(output / 'checks.json', checks)
    write_csv(output / 'grid_search.csv', grid_rows)
    write_csv(output / 'development_scores.csv', panel)
    write_csv(output / 'selection_summary.csv', summary_rows)
    write_csv(output / 'selection_trials.csv', trial_rows)
    rank_rows = []
    for name, value in rankings.items():
        rank_rows.append(dict(method=name, spearman_rho=value['spearman_rho'],
                              **{'all_'+k: v for k,v in value['all_pairs'].items()},
                              **{'controlled_'+k: v for k,v in value['size_controlled_pairs'].items()}))
    write_csv(output / 'ranking.csv', rank_rows)
    print('Selected coefficients:', selected)
    for name, value in rankings.items():
        print(f"{name:22s} rho={value['spearman_rho']:.9f} tau={value['all_pairs']['tau_b']:.9f} controlled={value['size_controlled_pairs']['tau_b']:.9f}")
    print('Controlled tests and safe starter checks passed; official harness agrees.')
    print('Artifacts:', output)
    print('Milestone development evaluation complete. Final labels were not opened.')


if __name__ == '__main__':
    main()
