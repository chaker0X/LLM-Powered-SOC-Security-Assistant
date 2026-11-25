from soc_assistant.llm.citations import check_citations, cited_ids, split_sentences
from soc_assistant.llm.prompts import NO_CONTEXT


def test_citation_forms_are_parsed_in_order_without_duplicates():
    text = "A [S2]. B [S1][S3]. C [S1, S4]. D [s5; S2]."
    assert cited_ids(text) == ["S2", "S1", "S3", "S4", "S5"]


def test_valid_answer():
    answer = ("Encoded PowerShell often hides malicious scripts [S1].\n"
              "1. Isolate the host from the network [S3]\n"
              "2. Decode the base64 command to see what it ran [S1][S2]")
    r = check_citations(answer, n_sources=4)
    assert r.valid and r.invalid == [] and r.problems() == []
    assert r.sentences == 3 and r.coverage == 1.0


def test_hallucinated_source_is_flagged():
    r = check_citations("PowerShell can download payloads from the internet [S7].", n_sources=4)
    assert not r.valid
    assert r.invalid == ["S7"]
    assert "[S7]" in r.problems()[0]


def test_answer_without_citations_is_flagged():
    r = check_citations("Isolate the machine and reset all the passwords of the user.", n_sources=4)
    assert not r.valid
    assert r.problems() == ["no source is cited"]


def test_refusal_is_valid():
    assert check_citations(NO_CONTEXT, n_sources=4).valid
    assert check_citations(f"  {NO_CONTEXT}\n", n_sources=0).refused


def test_citation_after_full_stop_belongs_to_previous_sentence():
    assert split_sentences("PowerShell runs code in memory without disk. [S1] Then it spreads to servers.") == [
        "PowerShell runs code in memory without disk [S1].", "Then it spreads to servers."]


def test_low_coverage_is_reported():
    answer = ("PowerShell is a scripting environment on Windows [S1]. "
              "Attackers like it because it is installed everywhere. "
              "It can run code directly in memory to avoid antivirus.")
    r = check_citations(answer, n_sources=2)
    assert r.valid                                   # ids are fine...
    assert r.problems() == ["only 1 of 3 sentences cite a source"]   # ...but most claims are unsourced
