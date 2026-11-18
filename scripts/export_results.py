"""Export English tables directly from recorded development results."""
import argparse
import csv
import json
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def read_json(path):
    """Read a UTF-8 result or protocol file."""
    return json.loads(path.read_text(encoding='utf-8-sig'))

def csv_rows(path):
    """Read a CSV table while accepting the exports' UTF-8 byte-order mark."""
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))

def markdown_table(headers,rows):
    """Format results as a Markdown table for the milestone report."""
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])

def report_tables(work,output):
    """Export English tables covering every numeric claim in the milestone draft."""
    metrics=read_json(work/'excess/development_metrics.json')
    initial=read_json(work/'adjacent/development_metrics.json')
    interpretation=read_json(work/'diagnostics/interpretation/interpretation_audit.json')
    labels={'original_nsc':'Original NSC','position_example':'Position example',
            'v1_frozen':'Adjacent penalty','global_v1_lambda':'Global at adjacent coefficient',
            'global_tuned':'Global independently tuned','v2':'Excess penalty',
            'uncentered_v2_lambda':'Adjacent at excess coefficient',
            'global_v2_lambda':'Global at excess coefficient',
            'reported_parameters':'Parameter count','reference_flops':'Reference FLOPs','random':'Uniform random'}
    ranking=csv_rows(work/'excess/ranking.csv')
    lines=['# Reproduced milestone tables','',
           'All results use development validation perplexity only. No final outcomes are accessed.','',
           '## Ranking','',markdown_table(['Method','Spearman rho','Kendall tau-b','Size-controlled tau-b'],
           [[labels[r['method']]]+[f"{float(r[k]):.9f}" for k in ('spearman_rho','all_tau_b','controlled_tau_b')]
            for r in ranking]),'', '## Coefficient grid','']
    grid=csv_rows(work/'excess/grid_search.csv')
    lines+=[markdown_table(list(grid[0]),[[r[k] for k in grid[0]] for r in grid]),'',
            '## Pairwise corrections','']
    effects=metrics['pair_effects']
    lines+=[markdown_table(['Method','Pairs','Changed','Corrected','Harmed','Net'],
        [[labels[name]]+[effects[name]['all'][k] for k in ('pairs','changed_pairs','corrected','damaged','net_corrections')]
         for name in ('original_nsc','position_example','v1_frozen','v2','global_tuned')]),'',
        '## All parameter-budget and selection results','']
    selection=csv_rows(work/'excess/selection_summary.csv')
    lines+=[markdown_table(['Budget','k','Method','Oracle PPL','Best selected PPL','Regret','PPL SD'],
        [[str(int(r['parameter_cap'])//1000000)+'M',r['k'],labels[r['method']]]+
         [f"{float(r[k]):.6f}" for k in ('oracle_ppl','mean_best_ppl','mean_regret','sd_best_ppl')]
         for r in selection]),'', '## Synthetic controls','',
        'Synthetic permutations have no measured perplexity.','',
        markdown_table(['FFN widths','Total capacity','Adjacent mismatch','Random reference','Excess'],
         [[str(r['config']['d_inner'])]+[f"{r[k]:.6f}" for k in
          ('total','adjacent','random_order_reference','excess')] for r in metrics['synthetic_controls']]),'',
        '## Supplementary evidence','',
        'Initial adjacent/CV-squared Spearman: '+str(initial['sensitivity']['adjacent_global_spearman']),
        '', 'Mean fractional score reductions and fold coverage:', '```json',
        json.dumps(interpretation,indent=2,ensure_ascii=False),'```','',
        'Five-fold development stability (not independent validation):','```json',
        json.dumps(metrics['stability_summary'],indent=2,ensure_ascii=False),'```','']
    (output/'milestone_tables.md').write_text('\n'.join(lines),encoding='utf-8')


def preliminary_summary(results):
    """Collect the draft's remaining claims from the computed development outputs."""
    initial=read_json(results/'adjacent/development_metrics.json')
    refined=read_json(results/'excess/development_metrics.json')
    return dict(
        stage='milestone_development_only',architectures=refined['architectures'],
        selected_coefficients=dict(adjacent=initial['selected']['adjacent_lambda'],
            global_cv2=refined['selected']['global_cv2'],excess=refined['selected']['v2']),
        size_controlled_pairs=refined['ranking']['v2']['size_controlled_pairs']['pairs'],
        initial_pair_diagnostics=read_json(results/'diagnostics/ranking_changes/summary.json'),
        excess_distribution=refined['excess_diagnostics'],
        adjacent_global_spearman=initial['sensitivity']['adjacent_global_spearman'],
        original_six_followup=refined['original_six_followup'],
        synthetic_controls=refined['synthetic_controls'],
        stability_summary=refined['stability_summary'],
        interpretation=read_json(results/'diagnostics/interpretation/interpretation_audit.json'),
        final_evaluation_performed=False)


def export_summary(results,output):
    """Write only the compact tables and supplementary evidence needed at the milestone."""
    output.mkdir(parents=True,exist_ok=True)
    names={'ranking.csv':'excess/ranking.csv',
           'lambda_grid.csv':'excess/grid_search.csv',
           'budget_selection.csv':'excess/selection_summary.csv',
           'pairwise_changes.csv':'excess/pair_effects.csv',
           'fold_results.csv':'excess/fold_results.csv'}
    for target,source in names.items():shutil.copy2(results/source,output/target)
    (output/'preliminary_summary.json').write_text(
        json.dumps(preliminary_summary(results),indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    report_tables(results,output)


def main():
    """Export the compact milestone results and readable report tables."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True,help='Recomputed detailed development results')
    parser.add_argument('--output',type=Path,required=True,help='Directory for milestone summaries')
    args=parser.parse_args()
    export_summary(args.results.resolve(),args.output.resolve())
    print('Milestone summaries:',args.output.resolve())

if __name__=='__main__':
    main()
