"""Post-hoc ranking diagnostics from development exports."""
import argparse
import csv
import json
import math
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
HERE=PROJECT/'results/diagnostics/ranking_changes'


def write_csv(name, rows):
    """Export rows with a stable header and UTF-8 encoding."""
    with (HERE/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def table(headers, rows):
    """Format column headers and records as a Markdown table."""
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    """Audit corrections and damage in the initial development rankings."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=PROJECT/'results')
    args=parser.parse_args()
    global HERE
    HERE=args.results.resolve()/'diagnostics/ranking_changes'
    HERE.mkdir(parents=True,exist_ok=True)
    adjacent=args.results.resolve()/'adjacent'
    with (adjacent/'development_scores.csv').open(encoding='utf-8-sig',newline='') as f:
        panel=list(csv.DictReader(f))
    results=json.loads((adjacent/'development_metrics.json').read_text(encoding='utf-8'))
    pairs=[]
    for i,a in enumerate(panel):
        for b in panel[:i]:
            x,y=float(a['original_nsc']),float(b['original_nsc'])
            u,v=float(a['adjacent_nsc']),float(b['adjacent_nsc'])
            old=(x>y)-(x<y); new=(u>v)-(u<v)
            qa,qb=-float(a['valid_ppl']),-float(b['valid_ppl'])
            truth=(qa>qb)-(qa<qb)
            ratio=min(float(a['reported_parameters']),float(b['reported_parameters']))/max(float(a['reported_parameters']),float(b['reported_parameters']))
            same_width=int(a['d_model'])==int(b['d_model'])
            pairs.append(dict(id_a=a['architecture_id'],id_b=b['architecture_id'],
                width_a=int(a['d_model']),width_b=int(b['d_model']),
                depth_a=int(a['n_layer']),depth_b=int(b['n_layer']),
                parameter_ratio=ratio,same_width=same_width,size_controlled=same_width and ratio>=.9,
                original_a=x,original_b=y,modified_a=u,modified_b=v,
                original_log_margin=abs(math.log(x/y)),relative_margin=abs(x-y)/max(x,y),
                mismatch_a=float(a['adjacent_mismatch']),mismatch_b=float(b['adjacent_mismatch']),
                global_cv2_a=float(a['global_cv2']),global_cv2_b=float(b['global_cv2']),
                ppl_a=-qa,ppl_b=-qb,original_correct=old==truth,modified_correct=new==truth,
                ordering_changed=old!=new,corrected=old!=truth and new==truth,
                damaged=old==truth and new!=truth))
    changed=[p for p in pairs if p['ordering_changed']]
    bands=[]
    for lo,hi in [(0,.005),(.005,.01),(.01,.02),(.02,.05),(.05,.1),(.1,float('inf'))]:
        members=[p for p in pairs if lo<=p['original_log_margin']<hi]
        bands.append(dict(log_margin_low=lo,log_margin_high=hi if math.isfinite(hi) else 'infinity',
            pairs=len(members),original_correct=sum(p['original_correct'] for p in members),
            modified_correct=sum(p['modified_correct'] for p in members),
            original_pair_accuracy=sum(p['original_correct'] for p in members)/len(members),
            modified_pair_accuracy=sum(p['modified_correct'] for p in members)/len(members),
            corrected=sum(p['corrected'] for p in members),damaged=sum(p['damaged'] for p in members)))
    corrected=sum(p['corrected'] for p in changed); damaged=sum(p['damaged'] for p in changed)
    old_errors=sum(not p['original_correct'] for p in pairs)
    summary=dict(split='development',architectures=len(panel),pairs=len(pairs),changed=len(changed),
        corrected=corrected,damaged=damaged,net_corrections=corrected-damaged,
        correction_precision=corrected/len(changed),original_errors=old_errors,
        error_recovery_fraction=corrected/old_errors,
        maximum_flipped_relative_margin=max(p['relative_margin'] for p in changed),
        margin_definition='abs(log(original_a/original_b))',
        label_use='Post-hoc diagnostic only. No scoring rule, coefficient or threshold selected.',
        note='Architecture pairs overlap and are not independent observations. No final labels used.')
    assert corrected==damaged==3 and len(changed)==6 and old_errors==343
    write_csv('all_pair_diagnostics.csv',pairs)
    write_csv('changed_pairs.csv',changed)
    write_csv('margin_bands.csv',bands)
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# 原始 NSC 的局部排序修正诊断（开发集）','',
        '本分析仅读取第一版已导出的开发集分数，没有修改评分公式、系数、冻结记录或原始 starter；没有读取最终标签。','',
        '## 1. 已确认的现象','',
        f'在 {len(pairs)} 对架构中，λ=0.1 改变 {len(changed)} 对排序，纠正 {corrected} 对、误伤 {damaged} 对，净纠错为零。'
        '变动对中的纠错比例为 50%，纠正了原始 NSC 的 343 对错误中的 3 对（约 0.875%）。'
        '这些都是开发集描述性结果，不能把六对样本当成独立验证。','',
        '六对的原始 NSC 相对差距都小于 0.85%。因此，第一版实际上已经只扰动了接近分数的架构；'
        '进一步限制“只改近分数”本身，并没有解决该改哪些对的问题。','',
        table(['架构 A','架构 B','宽度 A/B','原分数相对差距','结果','满足既定规模控制'],[
            [p['id_a'],p['id_b'],str(p['width_a'])+'/'+str(p['width_b']),f"{p['relative_margin']*100:.4f}%",
             '纠正' if p['corrected'] else '误伤','是' if p['size_controlled'] else '否'] for p in changed]),'',
        '三对纠正中有两对跨隐藏宽度；三对误伤均发生在同宽架构间。唯一符合既定规模控制的变动对是误伤。'
        '样本太少，不能据此制定“跨宽才修正”等规则。这个模式提示需要警惕宽度效应，并在更完整的受控比较中检验。','',
        '## 2. 原始分数差距与排名难度','',
        '下列区间是看到第一版结果后提出的探索性诊断，未用于选择模型或改变既定选参目标。'
        '差距采用 |log(S0_a/S0_b)|，接近零时约等于相对差距。不能把某一个事后表现好的区间当作确认性结果。','',
        table(['log 差距下限','上限','配对数','原始正确率','修正后正确率','纠正','误伤'],[
            [b['log_margin_low'],b['log_margin_high'],b['pairs'],f"{b['original_pair_accuracy']:.4f}",
             f"{b['modified_pair_accuracy']:.4f}",b['corrected'],b['damaged']] for b in bands]),'',
        '配对相互重叠，不能把配对数当成独立样本量，也未进行配对独立性的显著性检验。'
        '上述接近分数群体不是相同宽度且参数匹配的群体，两类控制需要分别报告。','',
        '## 3. 可研究的问题','',
        '研究问题可以聚焦为：在原始 NSC 难以区分的候选中，相邻容量结构是否提供额外的排序信息？'
        '相应诊断指标为纠正数、误伤数、净纠错、变动对中的纠错比例和原始错误的恢复比例。'
        '这些指标补充预先规定的 Spearman、Kendall、规模控制和预算选择，不替换或隐藏原来的指标。','',
        '任何新评分都应是能够对单个配置计算的统一有限标量，仍可进入官方 student-score 接口。'
        '不要把只有比较两个候选才能产生、可能出现循环的两两判断，直接当成该接口的评分。','',
        '## 4. 下一轮探索边界','',
        '保留原方法作为第一版。新候选需在运行前写清公式、小网格、选择指标和比较方式。'
        '随机排列参照可作为一个候选信号，用来测试更纯粹的排列效应；它并不自动识别正确的三对。'
        '不要根据六对样本的 ID、宽度或具体困惑度编写特殊规则。','',
        '如果用开发集内部交叉验证，可降低后续系数选择对单次划分的依赖；但因本次研究已看过全部开发标签，'
        '它仍属于开发阶段的稳健性分析，最终独立证据来自保留的最终集。'
        '公式和系数选择完成后再冻结下一版本；当前不读取最终集。','']
    (HERE/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print('Diagnostic report:',HERE/'REPORT.md')


if __name__=='__main__':
    main()
