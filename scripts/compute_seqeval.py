from seqeval.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from seqeval.scheme import IOB2
import numpy as np
import json

def pretty_print_dict(d, indent):
    res = ""
    for k, v in d.items():
        res += "\t"*indent + str(k) + "\n"
        if isinstance(v, dict):
            res += pretty_print_dict(v, indent+1)
        else:
            res += "\t"*(indent+1) + str(v) + "\n"
    print(res)
    return res
    
def compute_seqeval_jsonl(references_jsonl, predictions_jsonl, ref_col='ner_tags', pred_col='pred_ner_tags'):
    '''
    Computes the seqeval scores between two datasets loaded from jsonl (list of dicts with same keys).
    Sorts the datasets by 'unique_id' and verifies that the tokens match.
    '''
    # extract the tags and reverse the dict
    ref_dict = {k:[e[k] for e in references_jsonl] for k in references_jsonl[0].keys()}
    pred_dict = {k:[e[k] for e in predictions_jsonl] for k in predictions_jsonl[0].keys()}
        
    # sort by unique_id
    ref_idx = np.argsort(ref_dict['unique_id'])
    pred_idx = np.argsort(pred_dict['unique_id'])
    ref_ner_tags = np.array(ref_dict[ref_col], dtype=object)[ref_idx]
    pred_ner_tags = np.array(pred_dict[pred_col], dtype=object)[pred_idx]
    ref_tokens = np.array(ref_dict['tokens'], dtype=object)[ref_idx]
    pred_tokens = np.array(pred_dict['tokens'], dtype=object)[pred_idx]

    # check that tokens match
    #assert((ref_tokens==pred_tokens).all())
    
    
    # get report
    report = classification_report(y_true=ref_ner_tags, y_pred=pred_ner_tags, 
                                   scheme=IOB2, output_dict=True,
                                  )
    
    # extract values we care about
    report.pop("macro avg")
    report.pop("weighted avg")
    overall_score = report.pop("micro avg")

    seqeval_results = {
        type_name: {
            "precision": score["precision"],
            "recall": score["recall"],
            "f1": score["f1-score"],
            "suport": score["support"],
        }
        for type_name, score in report.items()
    }
    seqeval_results["overall_precision"] = overall_score["precision"]
    seqeval_results["overall_recall"] = overall_score["recall"]
    seqeval_results["overall_f1"] = overall_score["f1-score"]
    seqeval_results["overall_accuracy"] = accuracy_score(y_true=ref_ner_tags, y_pred=pred_ner_tags)    
    
    return(seqeval_results)


if __name__ == '__main__':

    # Pour les étudiants : indiquer le chemin vers le fichier NER-VALIDATION
    with open("data/NER-VALIDATION.jsonlines", 'r') as f:
        references_jsonl = [json.loads(l) for l in list(f)]

    # Pour les étudiants : indiquer ici le chemin vers votre fichier de prédiction sur le jeu de validation
    with open("predictions_output/predictions_output_1.jsonlines", 'r') as f:
        pred_jsonl = [json.loads(l) for l in list(f)]


    res = compute_seqeval_jsonl(references_jsonl, pred_jsonl, ref_col = 'ner_tags', pred_col='ner_tags')
    pretty_print_dict(res, 0)

