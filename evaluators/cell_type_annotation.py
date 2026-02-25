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

@register_evaluator("celltype")
class CellTypeEvaluator(EvaluateEngine):
    def __init__(self, datasetname, base_url,api_key,model_name, evaluated_model,obo_path="cl_minimal_clean5.obo"):
        self.datasetname = datasetname
        # Get the directory of the current script.
        current_dir = Path(__file__).parent

        # Build the OBO file path.
        obo_path = current_dir / "cl_nl6.obo"

        # Load ontology.
        self.ontology = pronto.Ontology(str(obo_path))
        # Build graph.
        self.graph = self.build_is_a_graph(self.ontology)
        self.api_key = api_key
        self.base_url = base_url
        self.model_name=model_name
        self.evaluated_model=evaluated_model
        


    def load_prompts(self) -> List[str]:
        """Load all prompt templates."""
        current_file = Path(__file__)
        prompt_path = current_file.parent.parent / "prompts" / "test_prompt.jsonl"

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

    def evaluate(self, answers: List[str]) -> dict:
        correct = 0
        total = 0

        with open(self.datasetname, "r", encoding="utf-8") as infile:
            for idx, line in enumerate(infile):
                data = json.loads(line.strip())
                true_cell_type = data["cell_type"]

                if idx >= len(answers):
                    raise ValueError("Answers list is shorter than dataset, some answers are missing.")

                answer = answers[idx]
                pattern = r"\[Predicted_Cell_Type:\s*([^\]]+?)\]"
                match = re.search(pattern, answer)

                if match:
                    predicted_cell_type = match.group(1).strip()
                else:
                    predicted_cell_type = "Unknown"

                if predicted_cell_type.lower() == true_cell_type.lower():
                    correct += 1
                else:
                    correct+=self.llm_eval(predicted_cell_type,true_cell_type)*0.2

                total += 1

        accuracy = correct / total if total > 0 else 0
        return {
            "accuracy": accuracy,
            "total": total,
            "correct": correct,
        }

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
            output.write(f"\n=== Term {term_idx}: {term.id} - {term.name} ===\n")
            paths = self.get_paths_to_roots(self.graph, term.id)
            if not paths:
                output.write("No paths to roots found.\n")
                continue
            for i, path in enumerate(paths, 1):
                output.write(f"\nPath {i}:\n")
                for j, pid in enumerate(reversed(path)):
                    node = self.ontology[pid]
                    indent = "  " * j
                    output.write(f"{indent}└─ {node.id} - {node.name}\n")
        result_str = output.getvalue()
        output.close()
        return result_str
    
    def llm_eval(self, predicted_cell_type, true_cell_type):

        message_template="""
        ## Input Provided:

        - **Exact Cell Type**: The manually annotated ground truth cell type  
        - **Cell Ontology Chain(s)**: Hierarchical paths from the root to the exact cell type  
        - **Predicted Cell Type**: The machine-generated cell type to evaluate  

        ---

        ## Task:

        Score the predicted cell type on a scale of **0–5**, based on both:

        1. Its biological and functional proximity to the exact cell type  
        2. Its position in the provided ontology chain(s), if applicable  

        Use your knowledge of cell biology and immunology to make a judgment that prioritizes **semantic and lineage closeness**, using the ontology chains as a **supporting reference**, not a rigid constraint.

        ---

        ## Scoring Rules:

        ###  Score: 5
        - The predicted type **exactly matches** the provided ground truth cell type.

        ###  Score: 4
        - The predicted type is a **recognized synonym**, **functionally equivalent**, or a **direct parent/ancestor** in the ontology path.
        - Alternatively, it is a highly specific **sibling subtype** of the exact cell (e.g., CD4+ vs. CD8+ T cells).

        ###  Score: 3
        - The predicted type is a **reasonable sibling** or **subtype within the same lineage**, but is **less specific** or **1–2 levels away**.
        - It may appear in the ontology path, or be biologically close based on known immunology or developmental origin.

        ###  Score: 2
        - The predicted type belongs to the **same broad functional category or lineage**, but is **general or distant** in semantic meaning.

        ###  Score: 1
        - The predicted type is **vaguely related**, such as belonging to the **same germ layer** or **broader immune context**, but lacks clear lineage relevance.

        ###  Score: 0
        - The predicted type is from a **different cell lineage**, **functionally unrelated**, or has **no biologically plausible relationship** to the exact cell type.
        - Also assign 0 for nonsensical, ambiguous, or non-cell-type predictions.

        ---

        ## Additional Principles for Scoring:

        - **Ontology Chains as Reference**:  
        Use the ontology path(s) to help identify possible matches and hierarchy positions. However, do **not rely solely on ontology inclusion**—prioritize functional and lineage reasoning.

        - **Biological Reasoning Encouraged**:  
        Even if a term is missing from the ontology chain, consider whether the predicted type is biologically plausible and reasonably related.

        - **Lineage First, Specificity Second**:  
        Prioritize whether the prediction belongs to the correct cell lineage before judging how specific or distant it is.

        - **Use the Best-Matching Chain**:  
        If multiple ontology chains are provided, use the one that leads to the best possible valid score for the predicted cell type.

        ---

        ## Your Output:

        Provide only the final score in the following format:

        [Score: X]
        Where `X` is an integer from 0 to 5.
 
        Input:
        Exact Cell Type:%s
        Cell Ontology Chain(s):%s
        Predicted Cell Type:%s
    """
        



        client = OpenAI(api_key=self.api_key,
                base_url=self.base_url)
        cell_path_chain=self.get_cell_paths_str(true_cell_type)
        

        
        response = client.chat.completions.create(
            model=self.model_name, 
            messages=[{"role": "system", "content": "You are a computational biologist specializing in cell type ontology analysis. You are tasked with evaluating the accuracy of predicted cell types by comparing them to expert-annotated ground truth labels in the context of the Cell Ontology (CL) hierarchy."},
                  {"role": "user", "content": message_template % (true_cell_type,cell_path_chain,predicted_cell_type)}])
        
        
        match = re.search(r'\[Score:\s*(\d+)\]', response.choices[0].message.content)
        if match:
            current_dir = Path(__file__).parent

            
            out_path = current_dir / "celltype" / f"{self.evaluated_model}.json"
            filename = out_path
            new_data = {
                "Exact_Cell_Type": true_cell_type,
                "Predicted_Cell_Type": predicted_cell_type,
                "Cell_Ontology_Chain(s)":cell_path_chain,
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

            return int(match.group(1))

        else:
            current_dir = Path(__file__).parent

            
            out_path = current_dir / "celltype" / f"{self.evaluated_model}.json"
            filename = out_path
            new_data = {
                "Exact_Cell_Type": true_cell_type,
                "Predicted_Cell_Type": predicted_cell_type,
                "Cell_Ontology_Chain(s)":cell_path_chain,
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
            return 0

        






        
