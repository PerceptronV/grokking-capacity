"""Generate the sensitivity table without rewriting author prose."""


def threshold_table(rows):
    lines=[r'\begin{table}[t]',r'\centering\small',r'\caption{Sensitivity on the same ten-prime primary cohort. Timescale means exclude unobserved events. All variants refind actual crossings and retain unresolved delays. Median residual is $\log_{10}(P_{\mathrm{onset}}/\widehat{P}_{\mathrm{cross}})$.}',r'\label{tab:threshold-sensitivity}',r'\begin{tabular}{lcccc}',r'\toprule',r'Train / gen / delay-val (\%) & Delay statistic & Pairs / 10 & Median residual & $\rho$ \\',r'\midrule']
    for r in rows:
        lines.append(f'{r["train_threshold"]} / {r["gen_threshold"]} / {r["delay_val_threshold"]} & {r["delay_aggregation"]} & {r["valid"]}/10 & ${r["median_log_ratio"]:.3f}$ & ${r["rho"]:.3f}$'+r' \\')
    return '\n'.join(lines+[r'\bottomrule',r'\end{tabular}',r'\end{table}'])
