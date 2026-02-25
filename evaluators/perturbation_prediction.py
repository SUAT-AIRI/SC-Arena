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


@register_evaluator("perturbation")
class PerturbationEvaluator(EvaluateEngine):
    def __init__(self, datasetname, base_url,api_key,model_name, evaluated_model,obo_path="cl_minimal_clean5.obo"):
        self.datasetname = datasetname
        # Get the directory of the current script.
        current_dir = Path(__file__).parent
        #GO-C Cellular Component Annotations
        self.GO_C_annotations_df = pd.read_csv(current_dir / 'gene_ontology_C.csv')
        #GO-P Biological Process Annotations
        self.GO_P_annotations_df = pd.read_csv(current_dir / 'gene_ontology_P.csv')
        #GO-F Molecular Function Annotations
        self.GO_F_annotations_df = pd.read_csv(current_dir / 'gene_ontology_F.csv')
        # NCBI Gene Card Annotations
        self.NCBI_gene_card_summaries = json.load(open(current_dir / 'NCBI_summary_of_genes.json', 'rb'))
        # NCBI Gene Card + UniProt protein summaries Annotations
        self.NCBI_UniProt_gene_card_protein_summaries = json.load(open(current_dir / 'NCBI_UniProt_summary_of_genes.json', 'rb'))
        
        self.api_key = api_key
        self.base_url = base_url
        self.model_name=model_name
        self.evaluated_model=evaluated_model
        


    def load_prompts(self) -> List[str]:
        """Load all prompt templates."""
        current_file = Path(__file__)
        prompt_path = current_file.parent.parent / "prompts" / "perturbation_prompt.jsonl"

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
                cell_sentence = data["ctrl_sentence"]
                
                num_genes = len(cell_sentence.split())
                candidate_genes = data["up_genes"] + data["down_genes"]

                random.seed(42)
                random.shuffle(candidate_genes)

                
                candidate_deg_list = ", ".join(candidate_genes)

                # Randomly choose one template.
                prompt_template = random.choice(prompts)
                raw_condition = data["condition"]  # e.g., "AHR+FEV" or "ZC3HAV1+ctrl"
                genes = raw_condition.split("+")

                if "ctrl" in genes:
                    # Single knockdown + control.
                    target_gene = genes[0] if genes[1] == "ctrl" else genes[1]
                    perturbation_description = f"The cell was perturbed by CRISPRi knockdown of {target_gene} (with a non-targeting control sgRNA)."
                    
                else:
                    # Double knockdown.
                    joined_genes = " and ".join(genes)
                    perturbation_description = f"The cell was perturbed by simultaneous CRISPRi knockdown of {joined_genes}."

                # Fill template variables.
                filled_prompt = prompt_template.format(
                    cell_sentence=cell_sentence,
                    perturbation_description=perturbation_description,
                    candidate_deg_list=candidate_deg_list,
                    num_genes=num_genes
                )
                results.append(filled_prompt)
        return results

    def evaluate(self, answers: List[str]) -> dict:
        correct = 0
        total = 0
        with open(self.datasetname, "r", encoding="utf-8") as infile:
            data_list = json.load(infile)

            for idx, data in enumerate(data_list):
                pert_sentence = data["pert_sentence"]
                ctrl_sentence = data["ctrl_sentence"]
                down_genes_str = ", ".join(data["down_genes"])
                up_genes_str = ", ".join(data["up_genes"])
                raw_condition = data["condition"]  # e.g., "AHR+FEV" or "ZC3HAV1+ctrl"
                genes = raw_condition.split("+")
                if "ctrl" in genes:
                    # Single knockdown + control.
                    target_gene = genes[0] if genes[1] == "ctrl" else genes[1]
                    perturbation_description = f"The cell was perturbed by CRISPRi knockdown of {target_gene} (with a non-targeting control sgRNA)."
                    
                else:
                    # Double knockdown.
                    joined_genes = " and ".join(genes)
                    perturbation_description = f"The cell was perturbed by simultaneous CRISPRi knockdown of {joined_genes}."

                NCBI_gene_card_UniProt_summaries = ''
                GO_C_description=''
                GO_P_description=''
                GO_F_description=''

                for gene in genes:
                    if gene.lower() == 'ctrl':
                        continue

                    NCBI_gene_card_UniProt_summaries += f"Annotations for GENE: {gene}\n"

                    # Safe fetch with .get or try-except
                    ncbi_summary = self.NCBI_gene_card_summaries.get(gene, "No NCBI Gene Card summary found.")
                    NCBI_gene_card_UniProt_summaries += f"NCBI Gene Card Summary: {ncbi_summary}\n"

                    combined_summary = self.NCBI_UniProt_gene_card_protein_summaries.get(gene, "No NCBI + UniProt summary found.")
                    NCBI_gene_card_UniProt_summaries += f"NCBI Gene Card + UniProt Protein Summary: {combined_summary}\n"

                    # GO_C
                    GO_C_description += f"GENE: {gene}\n"
                    go_c_labels = self.GO_C_annotations_df[self.GO_C_annotations_df['gene'] == gene]['direct_class_label'].dropna().unique()
                    GO_C_description += ', '.join(go_c_labels) + "\n" if len(go_c_labels) > 0 else "No GO_C annotation found.\n"

                    # GO_P
                    GO_P_description += f"GENE: {gene}\n"
                    go_p_labels = self.GO_P_annotations_df[self.GO_P_annotations_df['gene'] == gene]['direct_class_label'].dropna().unique()
                    GO_P_description += ', '.join(go_p_labels) + "\n" if len(go_p_labels) > 0 else "No GO_P annotation found.\n"

                    # GO_F
                    GO_F_description += f"GENE: {gene}\n"
                    go_f_labels = self.GO_F_annotations_df[self.GO_F_annotations_df['gene'] == gene]['direct_class_label'].dropna().unique()
                    GO_F_description += ', '.join(go_f_labels) + "\n" if len(go_f_labels) > 0 else "No GO_F annotation found.\n"


                if idx >= len(answers):
                    raise ValueError("Answers list is shorter than dataset, some answers are missing.")

                answer = answers[idx]+"]"
                pattern1 = r"\[Up:\s*([^\]]+?)\]"
                pattern2 = r"\[Down:\s*([^\]]+?)\]"
                pattern3 = r"\[Cell_Sentence:\s*([^\]]+?)\]"
                match = re.search(pattern1, answer)
                if match:
                    up_genes = match.group(1).strip()
                else:
                    up_genes = "Unknown"
                match = re.search(pattern2, answer)
                if match:
                    down_genes = match.group(1).strip()
                else:
                    down_genes = "Unknown"
                match = re.search(pattern3, answer)
                if match:
                    cell_sentence = match.group(1).strip()
                else:
                    cell_sentence = "Unknown"
                
                message_template=f"""
                You are an expert in single-cell transcriptomics and gene regulation. Your task is to evaluate the **biological plausibility and accuracy** of a predicted gene expression perturbation in response to a specific condition. You will be given the following information:

                1. **Unperturbed Cell Expression (cell sentence)** – A description of the gene expression profile before any perturbation.  
                2. **Perturbation Condition** – The experimental factor applied to perturb the cell.  
                3. **Ground Truth**:  
                - Perturbed Cell Expression (cell sentence)  
                - List of significantly **up-regulated genes**  
                - List of significantly **down-regulated genes**  
                4. **Predicted Result** by a language model:  
                - Predicted perturbed Cell Expression (cell sentence)  
                - Predicted significantly **up-regulated genes**  
                - Predicted significantly **down-regulated genes**  
                5. **Reference Knowledge**:  
                - Brief summaries from NCBI, GeneCards, and UniProt for the involved genes  
                - Gene Ontology (GO) information:  
                    - Cellular Component (GO_C_description)  
                    - Biological Process (GO_P_description)  
                    - Molecular Function (GO_F_description)  

                ---

                Your job is to compare the predicted response against the ground truth and reference knowledge, considering the following criteria:

                - Are the **predicted expression changes** consistent with the true perturbation pattern?  
                - Are the **up/down-regulated genes** biologically plausible given the perturbation and consistent with known gene functions?  
                - Does the **predicted cell sentence** qualitatively resemble the real one in terms of key gene activity shifts?  
                - Are the predictions **supported or contradicted** by the provided reference knowledge?  

                Give a score from **0 to 5** based on overall plausibility and alignment with both ground truth and known biology, where:  
                - 0 = Completely incorrect and biologically implausible  
                - 1 = Poor prediction and unconvincing  
                - 2 = Somewhat plausible but with major gaps or errors  
                - 3 = Reasonable prediction with some soundness  
                - 4 = Mostly correct with minor inconsistencies  
                - 5 = Highly accurate and biologically consistent  

                **Your answer should include a score in the following format**:  
                `[Score: X]`

                ---

                ### Input:

                1. **Unperturbed Cell Expression (cell sentence)**:  
                {ctrl_sentence}

                2. **Perturbation Condition**:  
                {perturbation_description}

                3. **Ground Truth**:  
                - **Perturbed Cell Expression (cell sentence)**:  
                {pert_sentence}  
                - **Up-regulated Genes**:  
                {up_genes_str}  
                - **Down-regulated Genes**:  
                {down_genes_str}

                4. **Predicted Result** by the language model:  
                - **Predicted Perturbed Cell Expression (cell sentence)**:  
                {cell_sentence}  
                - **Predicted Up-regulated Genes**:  
                {up_genes}  
                - **Predicted Down-regulated Genes**:  
                {down_genes}

                5. **Reference Knowledge**:  
                - **Gene Summaries (NCBI / GeneCards / UniProt)**:  
                {NCBI_gene_card_UniProt_summaries}  
                - **Gene Ontology Descriptions**:  
                - **Cellular Component**: {GO_C_description}  
                - **Biological Process**: {GO_P_description}  
                - **Molecular Function**: {GO_F_description}

                """
                client = OpenAI(api_key=self.api_key,
                base_url=self.base_url)

                response = client.chat.completions.create(
                    model=self.model_name, 
                    messages=[{"role": "system", "content": "You are a domain expert in single-cell transcriptomics, gene regulation, and biological data interpretation. Your role is to critically evaluate gene expression perturbation predictions made by another AI model, based on comprehensive input data including baseline expression, perturbation conditions, ground truth data, predicted results, and relevant gene functional annotations. Focus on assessing biological plausibility, consistency with experimental data, and alignment with known gene functions and ontology terms. Provide a concise, clear final score from 0 to 5 reflecting overall prediction quality and validity."},
                        {"role": "user", "content": message_template }],
                    
                        )
                match = re.search(r'\[Score:\s*(\d+)\]', response.choices[0].message.content)
                if match:
                    current_dir = Path(__file__).parent

                    
                    out_path = current_dir / "perturbation_new" / f"{self.evaluated_model}.json"
                    filename = out_path
                    new_data = {
                        "Unperturbed Cell Expression": ctrl_sentence,
                        "Perturbation Condition": perturbation_description,
                        "Perturbed Cell Expression":pert_sentence,
                        "Up-regulated Genes":up_genes_str,
                        "Down-regulated Genes":down_genes_str,
                        "Predicted Perturbed Cell Expression":cell_sentence,
                        "Predicted Up-regulated Genes":up_genes,
                        "Predicted Down-regulated Genes":down_genes,
                        "Gene Summaries":NCBI_gene_card_UniProt_summaries,
                        "Cellular Component":GO_C_description,
                        "Biological Process":GO_P_description,
                        "Molecular Function":GO_F_description,
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

                    
                    out_path = current_dir /"perturbation_new" / f"{self.evaluated_model}.json"
                    filename = out_path
                    new_data = {
                        "Unperturbed Cell Expression": ctrl_sentence,
                        "Perturbation Condition": perturbation_description,
                        "Perturbed Cell Expression":pert_sentence,
                        "Up-regulated Genes":up_genes_str,
                        "Down-regulated Genes":down_genes_str,
                        "Predicted Perturbed Cell Expression":cell_sentence,
                        "Predicted Up-regulated Genes":up_genes,
                        "Predicted Down-regulated Genes":down_genes,
                        "Gene Summaries":NCBI_gene_card_UniProt_summaries,
                        "Cellular Component":GO_C_description,
                        "Biological Process":GO_P_description,
                        "Molecular Function":GO_F_description,
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
