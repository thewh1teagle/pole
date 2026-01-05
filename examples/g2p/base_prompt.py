"""
Base prompt for Hebrew G2P (Grapheme-to-Phoneme) task.

This is the initial prompt that will be optimized by the pole library.
"""

BASE_PROMPT = """
You will receive Hebrew text. Convert it into IPA-like phonemes using ONLY the following symbols:

Vowels: a, e, i, o, u  
Consonants: b, v, d, h, z, χ, t, j, k, l, m, n, s, f, p, ts, tʃ, w, ʔ, ɡ, ʁ, ʃ, ʒ, dʒ

Rules:
1. Every Hebrew word must include a stress mark.
2. Place the stress mark (ˈ) immediately **before the stressed vowel**, not before the whole syllable.
   Example: shalom → ʃalˈom
3. Keep punctuation exactly as in the input.
4. Output ONLY the phonemes (no explanations, no slashes).
5. Use ʔ for א / ע.
6. Don't add vowels or consonants that aren't written.

Examples:
שלום עולם → ʃalˈom ʔolˈam
מה קורה? → mˈa koʁˈe?
אתה יודע → ʔatˈa jodˈeʔa

Now wait for the text.
"""

