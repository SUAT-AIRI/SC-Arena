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


@register_evaluator("generation")
class GenerationEvaluator(EvaluateEngine):
    def __init__(self, datasetname, base_url,api_key,model_name, evaluated_model,obo_path="cl_minimal_clean5.obo"):
        self.datasetname = datasetname
        # Get the directory of the current script.
        current_dir = Path(__file__).parent
        
        obo_path = current_dir / "cl_nl6.obo"
        cell_marker_path = current_dir / "cell_marker.txt"

        cell_types = set()

        with open(self.datasetname, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                cell_type = record.get("cell_type")
                if cell_type:
                    cell_types.add(cell_type)
        self.cell_types = cell_types

        # Load ontology.
        self.ontology = pronto.Ontology(str(obo_path))

        self.cell_marker_list = pd.read_csv(cell_marker_path, sep="\t", encoding="utf-8")

        self.api_key = api_key
        self.base_url = base_url
        self.model_name=model_name
        self.evaluated_model=evaluated_model
        

        


    def load_prompts(self) -> List[str]:
        """Load all prompt templates."""
        current_file = Path(__file__)
        prompt_path = current_file.parent.parent / "prompts" / "generation.jsonl"

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

        
        for celltype in self.cell_types:
            
            matching_term = [t for t in self.ontology.terms() if t.name == celltype][0]
            
            caption=matching_term.definition
            

            # Randomly choose one template.
            prompt_template = random.choice(prompts)

            # Fill template variables.
            filled_prompt = prompt_template.format(
                cell_type=celltype,
                cell_description=caption,
            )
            results.append(filled_prompt)
        return results

    def evaluate(self, answers: List[str]) -> dict:
        correct = 0
        total = 0
        if not len(self.cell_types) == len(answers):
                raise ValueError("Answers list is shorter than dataset, some answers are missing.")
        for idx, celltype in enumerate(self.cell_types):
            matching_term = [t for t in self.ontology.terms() if t.name == celltype][0]
            caption=matching_term.definition
            matched_sentences = []
            target_id=matching_term.id
            df=self.cell_marker_list
            cell_markers = ','.join(df[df["Specific_Cell_Ontology_ID"] == target_id]["Cell_Marker"].tolist())
            if not cell_markers:
                cell_markers = "Not Found For Now"
                

            with open(self.datasetname, "r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    if record.get("cell_type") == celltype:
                        sentence = record.get("cell_sentence")
                        if sentence:
                            matched_sentences.append(sentence)

            # Randomly sample up to 3 entries.
            sampled = random.sample(matched_sentences, min(3, len(matched_sentences)))
            # Build output string.
            sentence_str = ", ".join([f"cell{i+1}: {s}" for i, s in enumerate(sampled)])

            
        
            if idx >= len(answers):
                raise ValueError("Answers list is shorter than dataset, some answers are missing.")

            answer = answers[idx]
            pattern = r"\[Cell_Sentence:\s*([^\]]+?)\]"
            match = re.search(pattern, answer)

            if match:
                predicted_cell = match.group(1).strip()
            else:
                predicted_cell = answer

            message_template=f"""
            You are a biomedical expert in single-cell transcriptomics.

            Your task is to **evaluate whether the generated *cell sentence* (a ranked list of genes) matches the given cell type and natural language description**, using the provided **reference cell sentence** and **known marker genes** as benchmarks for comparison.

            ---

            **You are provided with:**

            - **Cell Type:**  
            The specific type of cell, such as “T cell,” “hepatocyte,” or “neuron.”

            - **Cell Description:**  
            A natural language description summarizing the key biological features, functions, or markers of the cell.

            - **Generated Cell Sentence (Ranked Genes):**  
            A gene list (cell sentence) generated by another model, ordered by predicted expression level.

            - **Reference Cell Sentence:**  
            A known, biologically accurate gene list that reflects the cell type and description, serving as a reference for evaluation.

            - **Known Marker Genes:**  
            A list of well-established marker genes for the given cell type. These genes are expected to appear with relatively high ranking if the generated sentence is accurate.

            ---

            **Please provide your evaluation as follows:**

            ### Reasoning:  
            Justify your evaluation considering:  
            - Are known marker genes present and appropriately ranked in the generated list?  
            - Does the gene list reflect the biological functions described?  
            - Are there key similarities or discrepancies with the reference cell sentence?  
            - Are there unexpected genes or patterns that reduce biological plausibility?

            ### Plausibility Score (0–5):  
            Provide a score from **0 to 5**, where:  
            - **5** = Highly realistic, strongly aligned with the cell type, description, marker genes, and reference  
            - **3** = Moderately plausible with some inconsistencies  
            - **0** = Not aligned or biologically implausible

            ---

            **Your answer should include a score in the following format:**  
            `[Score: X]`

            ---

            ### Input:

            1. **Cell Type**:  
            `{celltype}`
            

            2. **Cell Description**:  
            `{caption}`


            3. **Known Marker Genes**:  
            `{cell_markers}`


            4. **Generated Cell Sentence**:  
            `{predicted_cell}`


            5. **Ground Truth**:  
            - **Reference Cell Sentence**:  
                `{sentence_str}`


            """
            client = OpenAI(api_key=self.api_key,
            base_url=self.base_url)

            response = client.chat.completions.create(
                model=self.model_name, 
                messages=[{"role": "system", "content": "You are an expert biomedical evaluator with specialized knowledge in single-cell transcriptomics, cell marker gene interpretation, and Cell Ontology standards."},
                    {"role": "user", "content": message_template }])
            match = re.search(r'\[Score:\s*(\d+)\]', response.choices[0].message.content)
            if match:
                current_dir = Path(__file__).parent

                
                out_path = current_dir / "generation" / f"{self.evaluated_model}.json"
                filename = out_path
                new_data = {
                    "Cell Type": celltype,
                    "Cell Description": caption,
                    "Generated Cell Sentence": predicted_cell,
                    "Reference Cell Sentence": sentence_str,
                    "Marker Genes":cell_markers,
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

                
                out_path = current_dir /"generation" / f"{self.evaluated_model}.json"
                filename = out_path
                new_data = {
                    "Cell Type": celltype,
                    "Cell Description": caption,
                    "Generated Cell Sentence": predicted_cell,
                    "Reference Cell Sentence": sentence_str,
                    "Marker Genes":cell_markers,
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
