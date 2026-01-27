import os
from langchain_community.llms import Ollama

class ComplianceReporter:
    def __init__(self):
        # Service name 'ollama' works because of your Docker network
        self.llm = Ollama(model="llama3", base_url="http://ollama:11434")

    def generate_report(self, reasoning_list: list):
        # Pointing to YOUR existing file
        kb_path = "data/compliance/regulations.txt"
        
        try:
            with open(kb_path, "r") as f:
                regulations = f.read()
        except FileNotFoundError:
            regulations = "Standard AML protocols apply."

        factors = ", ".join(reasoning_list)
        
        prompt = f"""
        [INST]
        You are a Financial Compliance Auditor. 
        Referencing these regulations: {regulations}
        
        A transaction was flagged for the following reasons: {factors}
        
        TASK:
        Provide a concise, professional audit justification (max 2 sentences).
        1. Only reference the specific transaction type and Rule mentioned in the reasons.
        2. Do not include rules or transaction types from the framework that are not relevant to this specific flag.
        3. Explicitly cite the Section or Rule ID (e.g., Rule-101) in your response.
        [/INST]
        """
        
        try:
            return self.llm.invoke(prompt)
        except Exception as e:
            # Fallback if the AI is still waking up
            return f"Flagged due to: {factors}. (Detailed report pending AI sync)"