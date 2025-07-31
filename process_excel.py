import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import random
import boto3
from dotenv import load_dotenv
import os

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
REGION_NAME = os.getenv("REGION_NAME")

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=REGION_NAME,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)


PROMPT_TEMPLATE = """You are a legal assistant specialized in Thai government and law. Your only source of knowledge for answering the user's questions is the CONTENT. 

The USER, whom you are responding to does NOT want to know anything else about the topic aside from what they are asking so DO NOT elaborate or provide any extra knowledge outside of the retrieved content. 

RULES:
- Provide ACCURATE, and RELEVANT responses to the user's query using ONLY the CONTENT
- If the answer is not found, respond with: “Sorry, there is no information regarding your query.” Then stop immediately
- The following expressions are strictly banned:
    * "ตามที่ระบุไว้ในบริบท"
    * "จากข้อมูลที่ให้มา"
    * "จากบริบทข้างต้น"
    * Any variation that repeats or refers back to the existence of the context itself
- Do not include examples (e.g. “เช่น”) unless the question explicitly asks for them. Avoid naming committees, subpoints, or illustrative cases unless required

Instruction: Responses must
- Be written with Markdown.
- Be organized into clear sections.
- Answer in the language of the question

FINAL CHECK:
Did you hallucinate? If so, remove the part which contained hallucinations

Question: {Question}

CONTENT: {Content}"""

class GoogleSheetPromptProcessor:

    def __init__(self, sheet_name):
        # set up Google Sheets API client
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
        client = gspread.authorize(creds)

   
        self.raw_sheet = client.open(sheet_name).worksheet("RawData")
        self.ask_sheet = client.open(sheet_name).worksheet("ToAsk")
  
        self.ask_df = None


    def expand_questions(self):
        raw = self.raw_sheet.get_all_records()
        rows = []
        for i, r in enumerate(raw): 
            content_true = r["ข้อมูลอ้างอิง"]
            questions = r["คำถาม"]
            for q in questions.split(";"):
                q = q.strip()
                if not q:
                    continue
                
                # Select 2 irrelevant chunks
                other_chunks = [x["ข้อมูลอ้างอิง"] for j, x in enumerate(raw) if j != i]
                irrelevant_chunks = random.sample(other_chunks, k=2) if len(other_chunks) >= 2 else other_chunks

                # Merge the real + irrelevant ones
                mixed_context = [content_true] + irrelevant_chunks
                random.shuffle(mixed_context)

                rows.append({
                    "prompt_template": PROMPT_TEMPLATE,
                    "content": "\n\n".join(mixed_context),
                    "question": q,
                    "mock_response": ""
                })

        self.ask_df = pd.DataFrame(rows)
        print(f"Expanded to {len(self.ask_df)} question‐rows.")


    def save_expanded(self):
        """
        Overwrite the 'ToAsk' sheet with the expanded table (prompt, content, question).
        """
        # clear existing
        self.ask_sheet.clear()

        # prepare rows: header + all data rows
        data = [["คำถาม", "ข้อมูลอ้างอิง", "system prompt", "answer"]] + \
       self.ask_df[["question", "content", "prompt_template", "mock_response"]].values.tolist()

        self.ask_sheet.update(data)
        print("Wrote expanded questions to 'ToAsk' tab.")

    def mock_ai_response(self, prompt, content, question):
        # for now just fill the prompt in place of a real call
        return self.get_claude_response(question, content)

    def process_rows(self):
        """
        Fill mock_response for each row in the DataFrame.
        """
        self.ask_df["mock_response"] = self.ask_df.apply(
            lambda row: self.mock_ai_response(
                row["prompt_template"],
                row["content"],
                row["question"]
            ),
            axis=1
        )
        print("Generated mock_response for all rows.")

    def save_responses(self):
        """
        Write back only the mock_response column into the 'ToAsk' sheet,
        leaving the first three columns intact.
        """
        # batch‐update the entire column D (4th column)
        # build a list of lists for cells D2:Dn
        responses = [[r] for r in self.ask_df["mock_response"].tolist()]
        cell_range = f"D2:D{len(responses)+1}"
        self.ask_sheet.update(cell_range, responses)
        print("Written answer to sheet.")
    
    def get_claude_response(self, question, content):
        system_prompt = """You are a legal assistant specialized in Thai government and law. Your only source of knowledge for answering the user's QUESTIONS is the CONTENT.
    The USER, whom you are responding to does NOT want to know anything else about the topic aside from what they are asking so DO NOT elaborate or provide any extra knowledge outside of the retrieved CONTENT.
    RULES:
    - Provide ACCURATE, and RELEVANT responses to the user's query using ONLY the CONTENT 
    - If the answer is not found, respond with:  “Sorry, there is no information regarding your query.” Then stop immediately
    - The following expressions are strictly banned:
    *"ตามที่ระบุไว้ในบริบท"
    *"จากข้อมูลที่ให้มา"
    *"จากบริบทข้างต้น"
    *Any variation that repeats or refers back to the existence of the context itself
    - Do not include examples (e.g. “เช่น”) unless the QUESTION explicitly asks for them. Avoid naming committees, subpoints, or illustrative cases unless required
    Instruction: Responses must 
    - Be written with Markdown. 
    - Be organized into clear sections. 
    - Answer in the language of the QUESTION
    FINAL CHECK:
    Did you hallucinate? If so, remove the part which contained hallucinations"""

        response = bedrock_runtime.converse(
            modelId="anthropic.claude-3-5-haiku-20241022-v1:0",
            messages=[
                {
                    "role": "user",
                    "content": [{
                        "text": f"QUESTION {question}\nCONTENT: {content}"
                    }]
                }
            ],
            system=[{"text": system_prompt}],
            inferenceConfig={
                "maxTokens": 1500,
                "temperature": 0.1,
                "topP": 0.1
            }
        )
        usage = response.get("usage", {})
        input_toks = usage.get("inputTokens", 0)
        output_toks = usage.get("outputTokens", 0)
        cost_usd = (input_toks / 1000) * 0.001 + (output_toks / 1000) * 0.005
        USD_TO_THB = 35.0
        cost_baht = cost_usd * USD_TO_THB


        print(f"Cost: {cost_baht:.2f} THB \nOutput tokens: {output_toks:.2f}")
        answer = response["output"]["message"]["content"][0]["text"]
        return answer


if __name__ == "__main__":
    proc = GoogleSheetPromptProcessor("sheetdata")
    proc.expand_questions()      # expands questions & contexts with 3 chunks (your current code)
    proc.save_expanded()         # writes to ToAsk sheet
    proc.process_rows()          # calls Claude API
    proc.save_responses()        # writes AI answers back to ToAsk sheet

