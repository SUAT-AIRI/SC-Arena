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


@register_evaluator("captioning")
class CaptioningEvaluator(EvaluateEngine):
    def __init__(self, datasetname, base_url,api_key,model_name, evaluated_model,obo_path="cl_minimal_clean5.obo"):
        self.datasetname = datasetname
        # Get the directory of the current script.
        current_dir = Path(__file__).parent
        
        obo_path = current_dir / "cl_nl6.obo"

        # Load ontology.
        self.ontology = pronto.Ontology(str(obo_path))
        self.graph = self.build_is_a_graph(self.ontology)

        self.api_key = api_key
        self.base_url = base_url
        self.model_name=model_name
        self.evaluated_model=evaluated_model
        


    def load_prompts(self) -> List[str]:
        """Load all prompt templates."""
        current_file = Path(__file__)
        prompt_path = current_file.parent.parent / "prompts" / "captioning.jsonl"

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
            for line in infile:
                data = json.loads(line.strip())
                cell_sentence = data["cell_sentence"]
                organism = data["tissue"]
                num_genes = len(cell_sentence.split())

                # Randomly choose one template.
                prompt_template = random.choice(prompts)

                # Fill template variables.
                filled_prompt = prompt_template.format(
                    organism=organism,
                    cell_sentence=cell_sentence,
                    num_genes=num_genes
                )
                results.append(filled_prompt)
        return results
    

    def build_is_a_graph(self, ontology):
        G = nx.DiGraph()
        for term in ontology.terms():
            if not term.id.startswith("CL:"):
                continue
            parents = term.superclasses(distance=1).to_set() - {term}
            for parent in parents:
                if parent.id.startswith("CL:"):
                    G.add_edge(term.id, parent.id, label="is_a")
        return G

    def get_paths_to_roots(self, graph, start_node):
        paths = []
        queue = deque()
        queue.append([start_node])
        while queue:
            path = queue.popleft()
            current = path[-1]
            parents = list(graph.successors(current))
            if not parents:
                paths.append(path)
            else:
                for p in parents:
                    queue.append(path + [p])
        return paths

    
    def get_cell_paths_str(self, cell_name):
        matching_terms = [t for t in self.ontology.terms() if t.name == cell_name]
        if not matching_terms:
            return f"No term found with name '{cell_name}'"

        import io
        output = io.StringIO()
        for term_idx, term in enumerate(matching_terms, start=1):
            output.write(f"\n=== {term.name} Definition Path  ===\n")
            paths = self.get_paths_to_roots(self.graph, term.id)
            if not paths:
                output.write("No paths to roots found.\n")
                continue
            seen = set()
            unique_display_paths = []

            for path in paths:
                trimmed = tuple(path[:5])  # Use first 5 nodes as de-duplication key.
                if trimmed not in seen:
                    seen.add(trimmed)
                    unique_display_paths.append(trimmed)

            # Print each unique path.
            for i, display_path in enumerate(unique_display_paths, 1):
                output.write(f"\nPath {i}:\n")
                for j, pid in enumerate(reversed(display_path)):
                    node = self.ontology[pid]
                    indent = "  " * j
                    output.write(f"{indent}└─ {node.name}: {node.definition}\n")
                    
        result_str = output.getvalue()
        output.close()
        return result_str

    def evaluate(self, answers: List[str]) -> dict:
        correct = 0
        total = 0

        with open(self.datasetname, "r", encoding="utf-8") as infile:
            for idx, line in enumerate(infile):
                data = json.loads(line.strip())
                cell_sentence = data["cell_sentence"]
                true_cell_type = data["cell_type"]
                cell_path_chain=self.get_cell_paths_str(true_cell_type)
                print(cell_path_chain)
                

                if idx >= len(answers):
                    raise ValueError("Answers list is shorter than dataset, some answers are missing.")

                answer = answers[idx]
                pattern = r"\[Captioning:\s*([^\]]+?)\]"
                match = re.search(pattern, answer)

                if match:
                    predicted_caption = match.group(1).strip()
                else:
                    predicted_caption = answer

                message_template=f"""
                You are a biomedical expert in single-cell transcriptomics and cell type classification, with deep expertise in the Cell Ontology and its hierarchical structure.  

                Your task is to **evaluate a model-generated description of a single cell** using three clearly separated inputs:  

                ---

                ### 1. **Gene expression profile**
                A ranked list of genes from most to least expressed.  

                {cell_sentence}


                ---

                ### 2. **Cell Ontology definition path**
                A hierarchical lineage from a broad parent concept down to a specific, fine-grained cell type. Each level contains a name and definition.  

                {cell_path_chain}


                ---

                ### 3. **Cell description (to be evaluated)**
                **IMPORTANT:** This is the only text produced by the model that you should score.  
                If this section is empty, contains only whitespace, or does not describe a cell type, you must assign **[Score: 0]** without further analysis.  

                {predicted_caption}

                ---

                ## **Evaluation Objective**
                Assess whether the description in **Section 3** accurately and specifically reflects the **target cell type** as situated in the ontology path, while considering **gene expression evidence** from Section 1.  

                You must:
                - Check **ontology match** (terminal node or appropriate ancestor).  
                - Check **gene expression support** for claimed specificity.  

                ---

                ## **Key Principles**
                - **5 points** — Description exactly matches the **terminal node**, supported by clear marker genes.  
                - **4 points** — Matches terminal node but with minor omissions; gene evidence mostly supportive.  
                - **3 points** — Correct broader parent type or plausible sibling type, supported by gene data; **does not** name terminal node.  
                - **2 points** — Overly broad or vague description with limited evidence.  
                - **1 point** — Barely relevant or generic tissue/system reference.  
                - **0 points** — Empty, unrelated, incoherent, or wrong cell type.  

                **Special rule:** If Section 3 is empty or generic (e.g., "unknown cell" / "this is a cell"), assign **0** immediately.  

                ---

                **Your answer should include a score in the following format:**  
                [Score: X]  
                Then add a brief justification (2–4 sentences) explaining the reasoning behind your score.

                ---

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

                    
                    out_path = current_dir / "captioning_new" / f"{self.evaluated_model}.json"
                    filename = out_path
                    new_data = {
                        "Cell sentence": cell_sentence,
                        "Cell description": predicted_caption,
                        "Cell Ontology definition": cell_path_chain,
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

                    
                    out_path = current_dir /"captioning_new" / f"{self.evaluated_model}.json"
                    filename = out_path
                    new_data = {
                        "Cell sentence": cell_sentence,
                        "Cell description": predicted_caption,
                        "Cell Ontology definition": cell_path_chain,
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
