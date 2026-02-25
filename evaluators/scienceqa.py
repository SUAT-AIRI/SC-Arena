import re
from typing import List
from openai import OpenAI
from base import EvaluateEngine
from registry import register_evaluator
import json
from pathlib import Path
import random
import pronto
import networkx as nx
from collections import deque
import io
import openai
import os
import pandas as pd


@register_evaluator("scienceqa")
class ScienceqaEvaluator(EvaluateEngine):
    def __init__(self, datasetname, base_url,api_key,model_name, evaluated_model,obo_path="cl_minimal_clean5.obo"):
        self.datasetname = datasetname

        self.api_key = api_key
        self.base_url = base_url
        self.model_name=model_name
        self.evaluated_model=evaluated_model
        


    def load_prompts(self) -> List[str]:
        """Load all prompt templates."""
        current_file = Path(__file__)
        prompt_path = current_file.parent.parent / "prompts" / "scienceqa.jsonl"

        prompts = []
        with open(prompt_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                prompts.append(data["prompt"])
        return prompts

    def init_data(self) -> List[str]:
        """Convert input data into diversified prompts."""
        results = []
        prompts = self.load_prompts()

        
        with open(self.datasetname, "r", encoding="utf-8") as infile:
            data_list = json.load(infile)  
            for data in data_list:
                for question in data["response"]:

        
                    prompt_template = random.choice(prompts)

                    # Fill template variables.
                    filled_prompt = prompt_template.format(
                        question_type=question["type"],
                        question=question["question"]

                    )
                    results.append(filled_prompt)
        return results

    def evaluate(self, answers: List[str]) -> dict:
        correct = 0
        total = 0
        idx=-1
        with open(self.datasetname, "r", encoding="utf-8") as infile:
            data_list = json.load(infile)  
            for data in data_list:
                paper_title=data["title"]
                paper_abstract=data["abstract"]
                for question in data["response"]:
                    idx+=1
                    if idx >= len(answers):
                        raise ValueError("Answers list is shorter than dataset, some answers are missing.")

                    answer = answers[idx]
                    pattern = r"\[Answer:\s*([^\]]+?)\]"
                    match = re.search(pattern, answer)

                    if match:
                        model_answer = match.group(1).strip()
                    else:
                        model_answer = answer

                    message_template=f"""
                    You are a domain expert in single-cell biology and scientific reasoning.  
                    Your task is to evaluate whether the answer provided by a language model ("Evaluated Model") to a scientific question is accurate, well-reasoned, and biologically sound.

                    ---

                    ## You will be given:

                    - **[Question Type]**: A label indicating the type of knowledge or reasoning required (e.g., Marker-Based Reasoning, Pathway Logic, Experimental Design, etc.).
                    - **[Original Question]**: The actual question that was posed to the model.
                    - **[Ground Truth Answer]**: A reliable, expert-verified reference answer.
                    - **[Model Answer]**: The output from the Evaluated Model.
                    - **[Reference Paper Title]**: The title of the scientific paper from which the question is derived.
                    - **[Reference Paper Abstract]**: The abstract of that paper, provided as external knowledge to help you assess correctness.
                    - **[Relevant Passage]**: The specific section of the paper most closely related to this question (may include results, figures, or methods). Use this passage as the primary reference for correctness.

                    ---

                    ## Instructions

                    Carefully analyze whether the Model Answer is:

                    1. **Scientifically correct** (check facts, terminology, biological mechanisms).  
                    2. **Logically consistent** with the Original Question.  
                    3. **Well-aligned** with the Ground Truth Answer.  
                    4. **Appropriate** to the Question Type, showing the right reasoning depth and domain relevance.  
                    5. **Consistent with and supported by the Reference Paper and Relevant Passage** (do not copy text verbatim, but use them to check correctness).  

                    ---

                    ## Important Evaluation Rules

                    - Any **factual or scientific error** (e.g., misclassifying cytokines, incorrect pathway direction, or wrong biological effect) must lower the score.  
                    - If such an error exists, the score **cannot be 5**.  
                    - **Conceptual or mechanistic errors** that undermine reasoning (e.g., mixing up immune stimulatory vs suppressive roles) should be considered major flaws, scored **≤3**.  
                    - If the answer is largely correct but contains **minor imprecision** (e.g., vague wording, lack of detail without scientific contradiction), it may be scored **4**.  
                    - Only if the answer is **fully correct, with no scientific errors and strong alignment**, may it receive a **5**.  

                    ---

                    ## Your Evaluation Should Include:

                    - **Strengths**: What the Model Answer did well.  
                    - **Weaknesses / Errors**: Be explicit about what is wrong or misleading.  
                    - **Impact of Errors**: How they affect correctness and scoring.  

                    ---

                    ## Scoring Rubric

                    | Score | Description |
                    |-------|-------------|
                    | **5** | Fully correct, scientifically accurate, no errors, insightful, and well-aligned with the ground truth. |
                    | **4** | Mostly correct, but with minor flaws or imprecisions; no major scientific errors. |
                    | **3** | Partially correct, contains at least one clear scientific error or noticeable gap, though some correct reasoning is present. |
                    | **2** | Largely incorrect or incomplete; multiple scientific errors or major misunderstanding. |
                    | **1** | Minimally relevant, deeply flawed, or mostly wrong. |
                    | **0** | Completely incorrect, irrelevant, or nonsensical. |

                    At the end of your response, you must include the final score in this exact format:

                    `[Score: X]`

                    ---

                    ## Input

                    1. **Question Type**:  
                    `{question["type"]}`

                    2. **Original Question**:  
                    `{question["question"]}`

                    3. **Ground Truth Answer**:  
                    `{question["answer"]}`

                    4. **Model Answer**:  
                    `{model_answer}`

                    5. **Reference Paper Title**:  
                    `{paper_title}`

                    6. **Reference Paper Abstract**:  
                    `{paper_abstract}`

                    7. **Relevant Passage**:  
                    `{question["relevant_passage"]}`

                    """
                    client = OpenAI(api_key=self.api_key,
                    base_url=self.base_url)

                    response = client.chat.completions.create(
                        model=self.model_name, 
                        messages=[{"role": "system", "content": "You are a scientific expert in single-cell biology. Your task is to evaluate whether a language model's answer to a given question is scientifically correct, well-reasoned, and aligned with the ground truth. At the end, assign a score from 0 to 5 using [Score: X]."},
                            {"role": "user", "content": message_template }])
                    match = re.search(r'\[Score:\s*(\d+)\]', response.choices[0].message.content)
                    if match:
                        current_dir = Path(__file__).parent

                        
                        out_path = current_dir / "scienceqa" / f"{self.evaluated_model}.json"
                        filename = out_path
                        new_data = {
                            "Question Type": question["type"],
                            "Original Question": question["question"],
                            "Ground Truth Answer": question["answer"],
                            "Model Answer": model_answer,
                            "LLM Reasoning Trace":answer,
                            "Score": int(match.group(1)),
                            "response": response.choices[0].message.content

                        }

                        # If file exists, load old data; otherwise initialize an empty list.
                        if os.path.exists(filename):
                            with open(filename, 'r', encoding='utf-8') as f:
                                try:
                                    data = json.load(f)
                                except json.JSONDecodeError:
                                    data = []
                        else:
                            data = []
                            

                        # Append new data.
                        data.append(new_data)
                        

                        # Write back to file.
                        with open(filename, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=4, ensure_ascii=False)

                        correct+= 0.2* int(match.group(1))

                    else:
                        current_dir = Path(__file__).parent

                        
                        out_path = current_dir /"scienceqa" / f"{self.evaluated_model}.json"
                        filename = out_path
                        new_data = {
                            "Question Type": question["type"],
                            "Original Question": question["question"],
                            "Ground Truth Answer": question["answer"],
                            "Model Answer": model_answer,
                            "LLM Reasoning Trace":answer,
                            "response": response.choices[0].message.content

                        }

                        # If file exists, load old data; otherwise initialize an empty list.
                        if os.path.exists(filename):
                            with open(filename, 'r', encoding='utf-8') as f:
                                try:
                                    data = json.load(f)
                                except json.JSONDecodeError:
                                    data = []
                        else:
                            data = []
                            

                        # Append new data.
                        data.append(new_data)
                        

                        # Write back to file.
                        with open(filename, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=4, ensure_ascii=False)
                        correct += 0
                    total += 1

        accuracy = correct / total if total > 0 else 0
        return {
            "accuracy": accuracy,
            "total": total,
            "correct": correct,
        }
