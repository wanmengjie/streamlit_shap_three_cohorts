import re
from pathlib import Path

text = Path("PAPER_Manuscript_Submission_Ready.md").read_text(encoding="utf-8")
pat = r"[A-Za-z0-9]+(?:'[A-Za-z]+)?"
m = re.search(r"# ABSTRACT\n\n(.+?)\n\n---\n\n# MAIN TEXT", text, re.S)
abs_body = m.group(1) if m else ""
abs_words = len(re.findall(pat, abs_body))
m2 = re.search(r"# MAIN TEXT\n\n(.+?)\n\n---\n\n# FIGURE LEGENDS", text, re.S)
main = m2.group(1) if m2 else ""
main_lines = [ln for ln in main.splitlines() if not ln.strip().startswith("|")]
main_narrative = "\n".join(main_lines)
main_words = len(re.findall(pat, main_narrative))
# Structured abstract paragraphs (file lines 44–50)
lines = text.splitlines()
abs_lines = lines[43:51]  # 0-based: 44-50 in file = indices 43-50 inclusive -> 43:51
abs_join = "\n".join(abs_lines)
abs_struct_words = len(re.findall(pat, abs_join))
abs_plain = re.sub(r"\*+|#+", "", abs_join)
abs_wp = len([w for w in re.split(r"\s+", abs_plain.strip()) if w])
print("Abstract_block_words_full", abs_words)
print("Abstract_structured_BMRC_words", abs_struct_words)
print("Abstract_whitespace_word_count", abs_wp)
main_plain = re.sub(r"\*+", "", main_narrative)
main_wp = len([w for w in re.split(r"\s+", main_plain) if w and not w.startswith("|")])
print("Main_narrative_words_excl_table_rows", main_words)
print("Main_whitespace_word_count", main_wp)
