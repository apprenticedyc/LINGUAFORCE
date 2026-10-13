"""Export compact LaTeX tables from completed, scored prediction files."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MODELS=[('gpt','GPT-6 Astra'),('qwen','Qwen3.8-Max'),('deepseek','DeepSeek-Flash'),
        ('tfidf_word','Word TF--IDF'),('tfidf_char','Character TF--IDF'),('tfidf_constant','Constant')]


def fmt(v):return 'n/a' if v is None else f'{v:.3f}'


def main():
    results=json.loads((ROOT/'results/human_llm/metrics.json').read_text())
    common=json.loads((ROOT/'results/human_llm/common_subset_metrics.json').read_text())
    lines=[]
    def table(label,header,rows,caption):
        lines.extend(['\\begin{table}[t]','\\centering\\small','\\setlength{\\tabcolsep}{3pt}',
                      '\\begin{tabular}{@{}l'+'r'*(len(header)-1)+'@{}}','\\toprule',
                      ' & '.join(header)+r'\\','\\midrule'])
        lines.extend(' & '.join(row)+r'\\' for row in rows)
        lines.extend(['\\bottomrule','\\end{tabular}',r'\caption{'+caption+'}',r'\label{'+label+'}','\\end{table}',''])
    table('tab:pressure_results',['Model','Valid $n$','AUROC','F1'],
          [[name,str(results[m]['pressure']['n']),fmt(results[m]['pressure']['auroc']),fmt(results[m]['pressure']['f1'])] for m,name in MODELS],
          'Pressure detection. GPT, Qwen, and DeepSeek are zero-shot; TF--IDF models use human training labels. Constant predicts the training-majority label. Valid counts are out of 634; AUROC additionally requires a valid pressure score (see released metrics).')
    table('tab:intensity_results',['Model','Valid $n$','MAE $\\downarrow$','QWK $\\uparrow$'],
          [[name,str(results[m]['intensity']['n']),fmt(results[m]['intensity']['mae']),fmt(results[m]['intensity']['qwk'])] for m,name in MODELS],
          'Overall intensity prediction on valid 0--5 outputs. Counts are out of 634. The constant predictor uses the training median.')
    table('tab:strategy_results',['Model','Valid $n$','Macro-F1','Micro-F1'],
          [[name,str(results[m]['strategies']['n']),fmt(results[m]['strategies']['macro_f1']),fmt(results[m]['strategies']['micro_f1'])] for m,name in MODELS],
          'Fifteen-label strategy recognition. A task-valid response supplies all fifteen binary labels. One unambiguous key spelling is normalized without changing its value.')
    table('tab:dimension_results',['Model','Valid $n$','MAE $\\downarrow$','QWK $\\uparrow$'],
          [[name,str(results[m]['dimension_profile_n']),fmt(results[m].get('dimension_profile_mae')),fmt(results[m].get('dimension_profile_qwk'))] for m,name in MODELS],
          'Dimension prediction on complete, range-valid seven-dimensional outputs. MAE and QWK are averaged over dimensions. Different coverage prevents a direct ranking; Table~\\ref{tab:common_dimensions} uses identical dialogues.')
    table('tab:common_dimensions',['Model','MAE $\\downarrow$','QWK $\\uparrow$','All exact $\\uparrow$'],
          [[name,fmt(common['dimensions']['models'][m]['mae']),fmt(common['dimensions']['models'][m]['qwk']),fmt(results[m]['dimension_exact_accuracy_all'])] for m,name in MODELS],
          f"Aligned dimension comparison. MAE/QWK use the same {common['dimensions']['n']} dialogues with valid profiles from every model. All exact instead uses all 634 dialogues and seven ratings each, counting missing or out-of-range ratings as incorrect. The common subset is selected by model output validity and may be easier than the full test set.")
    path=ROOT/'results/human_llm/result_tables.tex'
    path.write_text('\n'.join(lines))
    print(path)


if __name__=='__main__':main()
