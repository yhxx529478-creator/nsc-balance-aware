"""Read existing frozen v2 exports to audit interpretation; no new tuning."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'results/diagnostics/interpretation'
V2=ROOT/'results/excess'


def main():
    """Audit penalty strength and coverage using frozen development exports."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'results')
    args=parser.parse_args()
    global HERE,V2
    HERE=args.results.resolve()/'diagnostics/interpretation'
    HERE.mkdir(parents=True,exist_ok=True)
    V2=args.results.resolve()/'excess'
    def rows(name):
        """Read a frozen development export for the interpretation audit."""
        with (V2/name).open(encoding='utf-8-sig',newline='') as file:
            return list(csv.DictReader(file))
    metrics=json.loads((V2/'development_metrics.json').read_text(encoding='utf-8'))
    panel=rows('development_scores.csv')
    folds={row['architecture_id']:int(row['fold']) for row in rows('fold_assignments.csv')}
    penalty={}
    for name in ('v1_frozen','v2','uncentered_v2_lambda','global_tuned'):
        discount=[1-float(row[name])/float(row['original_nsc']) for row in panel]
        penalty[name]=dict(mean_fractional_discount=statistics.mean(discount),
                           median_fractional_discount=statistics.median(discount),
                           zero_discount_architectures=sum(value==0 for value in discount))
    coverage=[]
    for row in rows('changed_pairs.csv'):
        if row['method']!='v2':continue
        a,b=row['id_a'],row['id_b']
        coverage.append(dict(id_a=a,id_b=b,fold_a=folds[a],fold_b=folds[b],
                             compared_within_heldout_fold=folds[a]==folds[b],
                             corrected=row['corrected']=='True',damaged=row['damaged']=='True'))
    summary=dict(analysis_type='post-hoc interpretation audit; no method or hyperparameter changes',
                 penalty=penalty,changed_pair_fold_coverage=coverage,
                 within_fold_covered_v2_changes=sum(row['compared_within_heldout_fold'] for row in coverage),
                 total_v2_changes=len(coverage),
                 original_pair_errors=metrics['pair_effects']['v2']['all']['original_wrong_pairs'],
                 v2_pair_errors=metrics['pair_effects']['v2']['all']['original_wrong_pairs']-
                                metrics['pair_effects']['v2']['all']['net_corrections'])
    (HERE/'interpretation_audit.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# 第二版结果的进一步解释','',
        '本补充分析只读取冻结的开发结果，不改变公式、参数、网格、评分代码或原结果。没有读取最终标签。','',
        '## 1. 可以确认的改善','',
        '固定开发网格选出 λ=0.5。相对第一版，改变对数从 6 减至 5，纠正从 3 增至 4，误伤从 3 减至 1。'
        '相对原始 NSC，净纠错为 +3，总错误配对从 343 降至 340，净相对错误减少约 0.875%。'
        '变化对中的 80% 是 4/5，不能说全部排名准确率为 80% 或提升了 80%。','',
        '在约 1% 原始分数差距的 23 对中，第二版纠正 3 对、误伤 0 对，正确率从 9/23 到 12/23。'
        '该群体的阈值在看过第一版后确定，属于探索性诊断，而且配对重叠。','',
        '## 2. 不能宣称全面优于基线','',
        '全局独立调参的 Spearman 为 0.970309，高于第二版 0.969925。位置示例的 Spearman 为 0.970069、'
        'Kendall 为 0.865051，也都高于第二版。近 1% 群体中位置示例达到 17/23 正确，高于第二版 12/23。'
        '第二版的亮点是相对第一版减少误伤，不是成为本档案上整体最强的方法。','',
        '180 个既定规模控制配对中，第二版纠正 1 对、误伤 1 对，τ_b 与原始 NSC 相同。'
        '全部八个预算/k 组合的最佳所选 PPL 与第一版相同，因此局部排名变化尚未带来额外的资源选择收益。','',
        '## 3. 需要排除的解释：修正变得更保守','',
        f"第一版平均分数扣减为 {penalty['v1_frozen']['mean_fractional_discount']*100:.4f}%，"
        f"第二版为 {penalty['v2']['mean_fractional_discount']*100:.4f}%。"
        f"第二版有 {penalty['v2']['zero_discount_architectures']} 个架构不扣分，第一版为零个。",'',
        f"即使固定相同 λ=0.5，未减参照的相邻惩罚平均扣减为 {penalty['uncentered_v2_lambda']['mean_fractional_discount']*100:.4f}%，"
        f"第二版仍只有 {penalty['v2']['mean_fractional_discount']*100:.4f}%。"
        '所以同系数消融改变了惩罚形状，也明显改变了有效强度。现有结果不能单独证明改善来自更纯粹的排列信号。','',
        '需要的进一步证据是：在不使用标签的前提下校准相同平均扣减强度，再比较两种惩罚。'
        '这样的附加消融应先记录校准规则，然后统一报告，不能根据结果选择有利的强度。','',
        '## 4. 五折结果的检验覆盖问题','',
        f"五折内没有观察到第二版相对原始 NSC 的排序改动。但完整开发集上改变的 {len(coverage)} 对架构中，"
        f"只有 {summary['within_fold_covered_v2_changes']} 对两端处于同一个折。",'',
        '因此该五折折内指标没有直接覆盖这五对变化。不能把“折内没有提升”简单解释为这五次修正已经被证伪，'
        '也不能据此说它们已经通过稳健性检验。它暴露出稀疏排序变化在小规模单次划分下难以检验的问题。','',
        '若进一步检查，应事先固定多个重复划分并完整汇报，不挑选有利划分；系数只在每次训练部分选择。'
        '此前已查看全部开发标签，所以这些仍是开发阶段稳健性分析，不能替代保留的最终集。','',
        '## 5. 对参照与系数的解释','',
        '随机排列期望的组合推导经过穷举控制验证，但“高于随机平均就应该扣分”仍然是性能假设。'
        'D_excess 含总体 CV² 的尺度信息，减去条件期望并不意味着与全局不平衡统计独立。'
        '51 个零惩罚体现了正部截断的稀疏性，也可能是减少误伤的原因。','',
        'λ 从 0.1、0.25 到 0.5 时净纠错分别为 +1、+2、+3；到 1 和 2 时变为 −2、−9。'
        '这支持“只能作为温和修正”的开发观察，不能外推成越强越好。没有根据该趋势改变已定网格或选参规则。','',
        '## 6. 中期可支持的表述','',
        '原相邻惩罚在开发集产生数量相当的纠正与误伤。减去同容量集合的随机排列期望并截断负值后，'
        '所选第二版表现为更稀疏的修正，误伤减少并获得少量净纠错。'
        '整体收益有限，尚未超越全部基线或产生额外预算收益；有效强度与稀疏性混杂，以及稀疏配对的验证覆盖不足仍需进一步检查。','']
    (HERE/'INTERPRETATION.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Saved supplementary analysis:',HERE/'INTERPRETATION.md')
    print('V2 changed pairs covered within held-out folds:',summary['within_fold_covered_v2_changes'],'/',len(coverage))


if __name__=='__main__':main()
