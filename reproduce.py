#!/usr/bin/env python3
"""Reproduce the NSC development experiments without changing reference results."""
import argparse
import ast
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent
STARTER=ROOT/'course_starter'
REFERENCE=ROOT/'results'

def deny_final_labels(event,args):
    """Block final-label access in the orchestration process."""
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        if Path(os.fsdecode(args[0])).name.lower()=='final_labels.json':
            raise PermissionError('Final evaluation is excluded from this reproduction')

def read_json(path):
    """Read a UTF-8 result or protocol file."""
    return json.loads(path.read_text(encoding='utf-8-sig'))

def write_json(path,value):
    """Write a finite, human-readable audit record."""
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def csv_rows(path):
    """Read a CSV table while accepting the exports' UTF-8 byte-order mark."""
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))

def equivalent(expected,actual,location='value'):
    """Compare deterministic outputs with tolerance for floating-point roundoff."""
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or expected.keys()!=actual.keys():
            raise AssertionError('Different keys at '+location)
        for key in expected:equivalent(expected[key],actual[key],location+'.'+str(key))
    elif isinstance(expected,list):
        if not isinstance(actual,list) or len(expected)!=len(actual):
            raise AssertionError('Different row counts at '+location)
        for index,(left,right) in enumerate(zip(expected,actual)):
            equivalent(left,right,location+'['+str(index)+']')
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
        if not isinstance(actual,(int,float)) or not math.isclose(expected,actual,rel_tol=1e-10,abs_tol=1e-12):
            raise AssertionError('Different numbers at '+location)
    elif expected!=actual:
        if isinstance(expected,str) and isinstance(actual,str):
            try:
                equivalent(json.loads(expected),json.loads(actual),location)
                return
            except (ValueError,TypeError):
                pass
        raise AssertionError('Different values at '+location+': '+repr(expected)+' != '+repr(actual))


def check_layout():
    """Check the permitted development data and concise source documentation."""
    if sys.version_info<(3,9):raise RuntimeError('Python 3.9 or later is required')
    if any(p.name.lower()=='final_labels.json' for p in ROOT.rglob('*')):
        raise RuntimeError('Final outcomes must remain outside this development repository')
    configs=read_json(STARTER/'data/architectures.json')
    splits=read_json(STARTER/'data/splits.json')
    labels=read_json(STARTER/'data/development_labels.json')
    assert len(configs)==200 and len(splits['development'])==len(splits['final'])==100
    assert set(splits['development'])==set(labels)
    assert not set(splits['development'])&set(splits['final'])
    sources=[Path(__file__)]
    for directory in ('src','scripts','tests'):sources+=list((ROOT/directory).glob('*.py'))
    for path in sources:
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            if isinstance(node,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                assert ast.get_docstring(node), (path,getattr(node,'name','module'))


def run_step(name,relative,arguments,output,steps):
    """Run a repository entry point and save its output and elapsed time."""
    command=[sys.executable,'-B',str(ROOT/relative)]+[str(v) for v in arguments]
    env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1')
    start=time.perf_counter()
    process=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,
                           encoding='utf-8',errors='replace')
    steps.append(dict(name=name,command=command,returncode=process.returncode,
                      elapsed_seconds=time.perf_counter()-start))
    (output/(name+'.log')).write_text(process.stdout+process.stderr,encoding='utf-8')
    print(('PASS' if process.returncode==0 else 'FAIL')+': '+name,flush=True)
    if process.returncode:raise RuntimeError('Inspect '+str(output/(name+'.log')))

def compare_results(summary):
    """Compare all six milestone summaries with the included preliminary results."""
    checked=[]
    for name in ('ranking.csv','lambda_grid.csv','budget_selection.csv','pairwise_changes.csv','fold_results.csv'):
        equivalent(csv_rows(REFERENCE/name),csv_rows(summary/name),name)
        checked.append(name)
    equivalent(read_json(REFERENCE/'preliminary_summary.json'),read_json(summary/'preliminary_summary.json'),
               'preliminary_summary.json')
    checked.append('preliminary_summary.json')
    return checked


def main():
    """Run the milestone controls or reproduce the draft's preliminary results."""
    sys.dont_write_bytecode=True
    sys.addaudithook(deny_final_labels)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true',help='Run controlled tests and course sanity checks only')
    parser.add_argument('--output',type=Path,help='A new output directory')
    args=parser.parse_args()
    output=(args.output or ROOT/'outputs'/datetime.now().strftime('run-%Y%m%d-%H%M%S')).resolve()
    for source in [ROOT]+[ROOT/name for name in ('course_starter','src','scripts','tests','protocols','results')]:
        if output==source or (source!=ROOT and source in output.parents):
            raise ValueError('Output must be outside code, data and reference results')
    if output.exists() and any(output.iterdir()):raise ValueError('Choose a new or empty output directory')
    output.mkdir(parents=True,exist_ok=True)
    steps=[];start=time.perf_counter()
    result=dict(mode='controlled-tests' if args.check_only else 'milestone-development',python=sys.version,
                output=str(output),steps=steps,final_labels_opened=False)
    try:
        check_layout()
        if args.check_only:
            run_step('controlled_adjacent','tests/test_adjacent.py',[],output,steps)
            run_step('controlled_excess','tests/test_excess.py',[],output,steps)
            run_step('safe_starter','scripts/check_starter_safe.py',[],output,steps)
        else:
            run_step('adjacent_experiment','scripts/run_adjacent.py',['--output',output],output,steps)
            run_step('pair_diagnostics','scripts/analyze_rank_changes.py',['--results',output/'results'],output,steps)
            run_step('excess_experiment','scripts/run_excess.py',['--output',output],output,steps)
            run_step('interpretation','scripts/analyze_interpretation.py',['--results',output/'results'],output,steps)
            run_step('summary','scripts/export_results.py',['--results',output/'results','--output',output/'summary'],output,steps)
            result['reference_comparisons']=compare_results(output/'summary')
        result['status']='passed'
    except Exception as exc:
        result.update(status='failed',error=str(exc))
        raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-start
        write_json(output/'validation.json',result)
    print('PASS: '+result['mode']+'; '+str(round(result['elapsed_seconds'],2))+' seconds')
    print('Output: '+str(output))

if __name__=='__main__':
    main()
