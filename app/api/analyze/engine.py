import os
import openai
import pandas as pd
import numpy as np
import json
import re
import sys
import asyncio
from llama_index.core import VectorStoreIndex
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


openai.api_key = process.env.OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = process.env.OPENAI_API_KEY


# ---------- PROMPT TEMPLATES ----------

PROMPT_TEMPLATE_GENERAL = """
You are tasked with the role of a climate scientist, assigned to analyze a company's sustainability report. Based on the following extracted parts from the sustainability report, answer the given QUESTIONS.
If you don't know the answer, just say that you don't know by answering "NA". Don't try to make up an answer.

Given are the following sources:
--------------------- [BEGIN OF SOURCES]
{sources}
--------------------- [END OF SOURCES]

QUESTIONS:
1. What is the company of the report?
2. What sector does the company belong to?
3. Where is the company located?

Format your answers in JSON format with the following keys: COMPANY_NAME, COMPANY_SECTOR, COMPANY_LOCATION.
Your FINAL_ANSWER in JSON (ensure there's no format error):
"""

PROMPT_TEMPLATE_YEAR = """
You are tasked with the role of a climate scientist, assigned to analyze a company's sustainability report. Based on the following extracted parts from the sustainability report, answer the given QUESTION.
If you don't know the answer, just say that you don't know by answering "NA". Don't try to make up an answer.

Given are the following sources:
--------------------- [BEGIN OF SOURCES]
{sources}
--------------------- [END OF SOURCES]

QUESTION:
In which year was the report published?

Format your answer in JSON format with the key: YEAR
Your FINAL_ANSWER in JSON (ensure there's no format error):
"""

PROMPT_TEMPLATE_QA = """
You are a senior sustainability analyst evaluating a company's climate-related transition plan and strategy.

This is basic information to the company:
{basic_info}

You are presented with the following sources from the company's report:
--------------------- [BEGIN OF SOURCES]
{sources}
--------------------- [END OF SOURCES]

Given the sources and no prior knowledge, your main task is to respond to the posed question encapsulated in "||".
Question: ||{question}||

Please consider the following explanation to the question:
+++++ [BEGIN OF EXPLANATION]
{explanation}
+++++ [END OF EXPLANATION]

Guidelines:
1. Your response must be grounded in the text.
2. If unsure, say so.
3. Stay under {answer_length} words.
4. Be skeptical of greenwashing.
5. Watch out for “cheap talk”.
6. Acknowledge that info is from the company.
7. Highlight if the report uses vague language.
8. Start your answer with "[[YES]]" or "[[NO]]" followed by explanation.

Return JSON with keys: ANSWER and SOURCES.
Your FINAL_ANSWER in JSON (no format errors):
"""

# ---------- HELPER FUNCTIONS ----------

import sys
import fitz  # PyMuPDF

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    return "\n".join(page.get_text() for page in doc)

if __name__ == "__main__":
    file_path = sys.argv[1]
    text = extract_text_from_pdf(file_path)
    print(text)




def pull_text_for_analysis(pdf_name: str) -> str:
    """
    Replaces the old createRetriever code.
    Use Pinecone to pull the doc's full text.
    """
    return pull_text_from_pinecone(pdf_name)



def extract_sources(retrieved_nodes):
    sources = []
    for i in retrieved_nodes:
        page = i.metadata.get("page_label", "N/A")
        clean_text = i.get_content().replace("\n", "")
        sources.append(f"PAGE {page}: {clean_text}")
    return "\n\n\n".join(sources)

def get_basic_info(retriever, model):
    nodes = retriever.retrieve("What is the name of the company, the sector it operates in and location of headquarters?")
    prompt = PromptTemplate(PROMPT_TEMPLATE_GENERAL).format(sources=extract_sources(nodes))
    response = OpenAI(temperature=0, model=model).complete(prompt)
    json_str = response.text.replace("```json", "").replace("```", "")
    parsed = json.loads(json_str)
    info_text = f" - Company name: {parsed['COMPANY_NAME']}\n - Industry: {parsed['COMPANY_SECTOR']}\n - Location: {parsed['COMPANY_LOCATION']}"
    return info_text, parsed

def get_year_info(retriever, model):
    nodes = retriever.retrieve("In which year was the report published?")
    prompt = PromptTemplate(PROMPT_TEMPLATE_YEAR).format(sources=extract_sources(nodes))
    response = OpenAI(temperature=0, model=model).complete(prompt)
    json_str = response.text.replace("```json", "").replace("```", "")
    return json.loads(json_str)

def create_prompt(retriever, basic_info, question, explanation, answer_length):
    nodes = retriever.retrieve(question)
    sources = extract_sources(nodes)
    return PromptTemplate(PROMPT_TEMPLATE_QA).format(
        basic_info=basic_info,
        sources=sources,
        question=question,
        explanation=explanation,
        answer_length=answer_length
    )

async def answer_async(prompts, model):
    llm = OpenAI(temperature=0, model=model)
    tasks = [llm.acomplete(p) for p in prompts]
    return await asyncio.gather(*tasks)

def write_to_excel(answers, questions, prompts, report_name, df, model, output_folder, option):
    verdicts, texts, pages, used_sources = [], [], [], []
    subcategories = [i.split("_")[1] for i in df["identifier"]]

    for i, a in enumerate(answers):
        try:
            json_str = a.text.replace("```json", "").replace("```", "")
            answer_dict = json.loads(json_str)
        except Exception:
            answer_dict = {"ANSWER": "ERROR", "SOURCES": []}
        verdict = re.search(r"\[\[([^]]+)\]\]", answer_dict["ANSWER"])
        verdicts.append(verdict.group(1) if verdict else "NA")
        texts.append(answer_dict["ANSWER"])
        pages.append(", ".join(map(str, answer_dict["SOURCES"])))
        used_sources.append(prompts[i].split("[BEGIN OF SOURCES]")[1])

    df_out = pd.DataFrame({
        "subcategory": subcategories,
        "question": questions,
        "decision": verdicts,
        "answer": texts,
        "source_pages": pages,
        "source_texts": used_sources
    })
    output_path = os.path.join(output_folder, f"{os.path.basename(report_name).split('.')[0]}_{model}{option}.xlsx")
    df_out.to_excel(output_path)
    return output_path



from pinecone import Pinecone

def pull_text_from_pinecone(pdf_name: str) -> str:
    """
    Query Pinecone for all chunks belonging to pdf_name,
    then return one big string of text.
    """
    # 1) Initialize Pinecone
    pc = Pinecone(
        api_key= process.env.PINECONE_API_KEY,
        # Optional: specify environment as well
        environment="us-east-1"
    )
    index = pc.Index("resonate-report-rag")

    # Dummy vector query with filter
    dummy_vector = [0.0] * 1536
    query_response = index.query(
        vector=[0.0]*1536,
        top_k=100,
        include_metadata=True,
        filter={"pdf_name": pdf_name}
    )

    matches = query_response.matches
    if not matches:
        raise FileNotFoundError(f"No Pinecone chunks found for doc: {pdf_name}")

    # 3) Sort by page_number (or chunk_id)
    sorted_matches = sorted(
        matches,
        key=lambda m: m.metadata.get("page_number", 0)
    )

    # 4) Join all chunk texts
    # We assume 'text' was the field used at ingestion
    full_text = "\n\n".join(
        m.metadata["text"] for m in sorted_matches if "text" in m.metadata
    )
    return full_text

def chunk_local_text(
    full_text: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int
):
    """
    Take a big text, re-chunk it with LlamaIndex, and
    return a local retriever that can run .retrieve().
    """
    documents = [Document(text=full_text)]
    parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = parser.get_nodes_from_documents(documents)
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    index_lgi = VectorStoreIndex(nodes, embed_model=embed_model)
    retriever = VectorIndexRetriever(index=index_lgi, similarity_top_k=top_k)
    return retriever


# ---------- MAIN ----------
# ============ <NEW> MAIN FUNCTION: run_engine() ============

async def run_engine(report_path: str, model: str = "gpt-4", num_questions: int = None) -> dict:
    # 1) Load the questions
    questions_xlsx = "/Users/achalme2/Desktop/reports_rag/app/api/transition-audit/data/Questions.xlsx"
    df = pd.read_excel(questions_xlsx)
    if num_questions:
        df = df.head(num_questions)

    # 2) Pull text from Pinecone, then re-chunk with LlamaIndex
    full_text = pull_text_from_pinecone(report_path)
    retriever = chunk_local_text(full_text, 350, 50, 8)

    # 3) Grab basic info + year
    basic_info, metadata = get_basic_info(retriever, model)
    year_info = get_year_info(retriever, model)
    metadata.update(year_info)
    metadata["REPORT"] = report_path

    # 4) Build prompts for each question
    prompts, questions = [], []
    for i in range(len(df)):
        q = df.iloc[i]["question"]
        exp = df.iloc[i]["question definitions"]
        prompts.append(create_prompt(retriever, basic_info, q, exp, 200))
        questions.append(q)

    # 5) Run asynchronously in batches of 5 (like your main code)
    answers = []
    batch_size = 5
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        res = await answer_async(batch, model)
        answers.extend(res)

    # 6) (Optional) Write to Excel or skip
    # output_folder can be a param, or we skip
    output_folder = "/Users/achalme2/Desktop/reports_rag/transition_outputs"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    excel_path = write_to_excel(answers, questions, prompts, report_path, df, model,
                                output_folder, f"_topk8_params{num_questions or 'all'}")

    # 7) Return a dictionary with summary info
    return {
        "metadata": metadata,
        "excel_path": excel_path,
        "num_questions": len(questions),
    }


# ============ <OLD> CLI Entry Point ============

if __name__ == "__main__":
    # usage: python engine.py <report_path> <model> <num_questions?>
    if len(sys.argv) < 3:
        print("Usage: python engine.py <report_path> <model> [num_questions]")
        sys.exit(1)

    report_path_cli = sys.argv[1]
    model_cli = sys.argv[2]
    num_q_cli = None if len(sys.argv) < 4 else int(sys.argv[3])

    # Use 'run_engine' programmatically
    final_result = asyncio.run(run_engine(report_path_cli, model_cli, num_q_cli))
    print(json.dumps(final_result, indent=2))