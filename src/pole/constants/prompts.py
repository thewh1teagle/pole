"""Prompt templates for agent-based mutation and verification."""

MUTATOR_INSTRUCTION_TEMPLATE = """You are a prompt optimization agent. Generate {num_variations} improved versions of the given prompt.

CURRENT PROMPT:
{prompt}{loss_info}

CRITICAL RULES:
1. Keep the EXACT same task/goal as the original
2. Keep the EXACT same output format (if original expects one word, variations must too)
3. Make SMALL improvements only (clearer wording, better structure)
4. NO explanations, NO meta-text, NO labels, NO commentary
5. Output ONLY raw prompts separated by "---"

FORMAT REQUIREMENT:
Your entire response must be ONLY the prompts themselves, separated by "---" on its own line.

EXAMPLE OF CORRECT OUTPUT:
Classify the sentiment as positive, negative, or neutral. Be concise.

---

Determine sentiment: positive, negative, or neutral. One word only.

---

Analyze text sentiment. Reply with: positive, negative, or neutral.

EXAMPLE OF WRONG OUTPUT (DO NOT DO THIS):
Here are three improved prompts:
1. First prompt...
2. Second prompt...

YOUR RESPONSE (only prompts, separated by "---"):"""


VERIFIER_INSTRUCTION_TEMPLATE = """You are a quality control agent. Verify if this prompt variation is valid and makes sense.

Original Prompt:
"{original}"

Proposed Variation:
"{variation}"

Is this variation:
1. A valid prompt (not gibberish)?
2. Related to the original task?
3. Potentially useful?

Answer ONLY with: YES or NO"""


FALLBACK_VARIATION = "{prompt}\n\nBe more precise."
