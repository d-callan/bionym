from idresolver.questions import s0_classify


def test_ncbi_gene_numeric():
    assert s0_classify.regex_hints("672") == ["ncbi_gene"]


def test_veupathdb_plasmodium():
    assert "veupathdb" in s0_classify.regex_hints("PF3D7_0710100")


def test_veupathdb_tritryp_dot_style():
    assert "veupathdb" in s0_classify.regex_hints("Tb927.10.1230")


def test_ensembl():
    assert "ensembl" in s0_classify.regex_hints("ENSG00000139618")


def test_uniprot():
    assert "uniprot" in s0_classify.regex_hints("P12345")


def test_refseq():
    assert "refseq" in s0_classify.regex_hints("NM_000000")


def test_locus_tag_dropped_when_specific_match():
    hints = s0_classify.regex_hints("PF3D7_0710100")
    assert "locus_tag" not in hints


def test_generic_falls_back_to_locus_tag():
    assert s0_classify.regex_hints("someGene42") == ["locus_tag"]


def test_choice_question_has_other_escape():
    q = s0_classify.build_questions(["ncbi_gene", "veupathdb"])
    assert "other" in q["id_type"]["criteria"]
