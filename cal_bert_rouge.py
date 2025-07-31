import gspread
from oauth2client.service_account import ServiceAccountCredentials
from rouge_score import rouge_scorer
from bert_score import score
import pandas as pd

def compute_rouge_l_char_precision(reference, prediction):
    reference_chars = reference.replace(" ", "")
    prediction_chars = prediction.replace(" ", "")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = scorer.score(reference_chars, prediction_chars)
    return round(scores["rougeL"].precision, 4)

def compute_bert_score_f1(reference, prediction):
    try:
        _, _, f1 = score([prediction], [reference], lang="th", model_type="bert-base-multilingual-cased")
        return round(f1[0].item(), 4)
    except:
        return 0.0

def get_real_context(raw_df, question):
    for _, row in raw_df.iterrows():
        for q in str(row["คำถาม"]).split(";"):
            if q.strip() == question.strip():
                return row["ข้อมูลอ้างอิง"]
    return ""

def main():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
    client = gspread.authorize(creds)

    sheet = client.open("sheetdata")
    raw_data = sheet.worksheet("RawData")
    toask = sheet.worksheet("ToAsk")

    raw_df = pd.DataFrame(raw_data.get_all_records())
    ask_df = pd.DataFrame(toask.get_all_records())

    rouge_real = []
    rouge_mixed = []
    bert_real = []
    bert_mixed = []

    for i, row in ask_df.iterrows():
        question = row["คำถาม"]
        mixed_context = row["ข้อมูลอ้างอิง"]
        answer = row["answer"]

        real_context = get_real_context(raw_df, question)

        rouge_real.append(compute_rouge_l_char_precision(real_context, answer))
        rouge_mixed.append(compute_rouge_l_char_precision(mixed_context, answer))
        bert_real.append(compute_bert_score_f1(real_context, answer))
        bert_mixed.append(compute_bert_score_f1(mixed_context, answer))

        print(f"Processed row {i+1}")

    toask.update_acell("E1", "ROUGE-L Precision (True Context)")
    toask.update_acell("F1", "ROUGE-L Precision (Mixed Context)")
    toask.update_acell("G1", "BERTScore F1 (True Context)")
    toask.update_acell("H1", "BERTScore F1 (Mixed Context)")

    toask.update(f"E2:E{len(rouge_real)+1}", [[s] for s in rouge_real])
    toask.update(f"F2:F{len(rouge_mixed)+1}", [[s] for s in rouge_mixed])
    toask.update(f"G2:G{len(bert_real)+1}", [[s] for s in bert_real])
    toask.update(f"H2:H{len(bert_mixed)+1}", [[s] for s in bert_mixed])

    print("ROUGE and BERTScore successfully written to columns E–H.")

if __name__ == "__main__":
    main()
