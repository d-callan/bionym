from bionym.evidence import Evidence
from bionym.graph import NodeType
from bionym.jev import JevClient
from bionym.llm import LlmClient
from bionym.resolver import Resolver

_EV = [Evidence(source="fake", endpoint="fake://x", summary="fake", payload={})]


class FakeNcbi:
    def gene_by_id(self, gene_id):
        return [
            {
                "source": "ncbi_datasets",
                "gene_id": gene_id,
                "symbol": "BRCA2",
                "description": "BRCA2 DNA repair associated",
                "locus_tag": None,
                "tax_id": 9606,
                "organism": "Homo sapiens",
                "assembly_accession": "GCF_000001405.40",
                "assembly_accessions": ["GCF_000001405.40", "GCF_009914755.1"],
                "genomic_accessions": ["NC_000013.11"],
                "raw": {},
            }
        ], _EV

    def find_gene(self, term):
        return [
            {
                "source": "ncbi_eutils",
                "gene_id": "8107544",
                "symbol": "PF3D7_0710100",
                "description": "hypothetical protein",
                "locus_tag": "PF3D7_0710100",
                "tax_id": 5833,
                "organism": "Plasmodium falciparum 3D7",
                "assembly_accession": None,
                "genomic_accessions": [],
                "raw": {},
            }
        ], []

    def taxon(self, tax_id):
        return {
            "tax_id": tax_id,
            "name": "Homo sapiens" if tax_id == 9606 else "Plasmodium falciparum",
            "rank": "species",
            "species_tax_id": tax_id,
            "species_name": "x",
            "lineage": [],
            "raw": {},
        }, []

    def assemblies_for_taxon(self, tax_id):
        return [
            {
                "accession": "GCF_999999999.1",
                "name": "ASM999v1",
                "organism": "Homo sapiens",
                "tax_id": tax_id,
                "level": "Complete Genome",
                "refseq": True,
                "submission_date": "2025-01-01",
                "raw": {},
            }
        ], []

    def gene_products(self, gene_id):
        return {
            "transcripts": ["XM_001348563.4"],
            "proteins": ["XP_001348599.1"],
        }, _EV

    def gene_pubmed(self, gene_id):
        return [{"pubmed_id": "26289816", "kind": "gene_pubmed"}], _EV

    def pubmed_details(self, pmids):
        return {
            "26289816": {
                "title": "BRCA1 loss alters the DNA damage response",
                "abstract": "BRCA1 is required for homologous recombination.",
                "journal": "Nature",
                "authors": ["Smith J"],
                "year": "2016",
            }
        }, _EV

    def geo_datasets_for_gene(self, symbol, organism=None):
        return [
            {
                "source": "geo",
                "accession": "GSE12345",
                "title": "Expression profiling of BRCA1",
                "taxon": "Homo sapiens",
                "n_samples": 20,
                "gds_type": "Expression profiling by array",
                "tech_type": "in situ oligonucleotide",
                "pubmed_ids": [],
                "summary": "",
                "raw": {},
            },
            {
                "source": "geo",
                "accession": "GDS9999",
                "title": "Curated BRCA2 dataset",
                "taxon": "Homo sapiens",
                "n_samples": 10,
                "gds_type": "Expression profiling by array",
                "tech_type": "in situ oligonucleotide",
                "pubmed_ids": [],
                "summary": "",
                "raw": {},
            },
        ], []

    def geo_dataset_values(self, accession, symbol):
        if not accession.startswith("GDS"):
            return [], []
        return [
            {"sample": "control", "value": 2.0, "percentile": 10.0},
            {"sample": "treated", "value": 8.5, "percentile": 95.0},
        ], _EV


class FakeVeuPathDB:
    def lookup_gene(self, gene_id):
        if gene_id != "PF3D7_1477700":
            return [], []
        return [
            {
                "source": "veupathdb",
                "project": "plasmodb",
                "gene_id": gene_id,
                "symbol": "MAL13P1.100",
                "tax_id": 5833,
                "organism": "Plasmodium falciparum 3D7",
                "url": "https://plasmodb.org/x",
                "aliases": [
                    {
                        "db": "Uniprot/SWISSPROT",
                        "alias": "Q8I5A7",
                        "type": "protein",
                    },
                    {"db": "GeneDB", "alias": "PFF0740w", "type": "previous ID"},
                    # duplicate + self-reference: both must be skipped
                    {"db": "GeneDB", "alias": "PFF0740w", "type": "previous ID"},
                    {"db": "PlasmoDB", "alias": gene_id, "type": "primary key"},
                ],
            }
        ], _EV

    def gene_pubmed(self, gene_id):
        return [], _EV

    def gene_datasets(self, gene_id):
        return {
            "project": "plasmodb",
            "datasets": [
                {
                    "dataset_id": "DS_TEST",
                    "url": "https://plasmodb.org/x",
                    "sample_names": ["ring", "trophozoite"],
                    "values": [
                        {"sample": "ring", "value": "10.0", "percentile": "12.0"},
                        {
                            "sample": "trophozoite",
                            "value": "900.0",
                            "percentile": "99.1",
                        },
                    ],
                }
            ],
            "dataset_names": ["Test dataset"],
        }, _EV

    def dataset_records(self, dataset_ids):
        return {
            dsid: {
                "display_name": "Test dataset",
                "type": "rna seq",
                "summary": "test",
                "pmid": None,
            }
            for dsid in dataset_ids
        }


class FakeOma:
    def protein_info(self, identifier):
        return {
            "source": "oma",
            "entry_nr": 12345,
            "omaid": "HUMAN00000",
            "canonical_id": "BRCA2_HUMAN",
            "species": "Homo sapiens",
            "tax_id": 9606,
            "raw": {},
        }, _EV

    def orthologs(self, identifier):
        return [
            {
                "source": "oma",
                "omaid": "HUMAN00001",
                "canonical_id": "BRCA1_HUMAN",
                "species": "Mus musculus",
                "tax_id": 10090,
                "rel_type": "1:1",
                "score": 0.9,
                "raw": {},
            }
        ], []


class FakeUniProt:
    def entry_by_accession(self, accession):
        return None, _EV

    def search_gene(self, symbol, tax_id=None):
        return [
            {
                "source": "uniprot",
                "accession": "P38398",
                "uniprot_id": "BRCA1_HUMAN",
                "gene_name": "BRCA1",
                "protein_name": "Breast cancer type 1 susceptibility protein",
                "organism": "Homo sapiens",
                "tax_id": 9606,
                "go_terms": [
                    {"id": "GO:0003677", "term": "DNA binding", "aspect": "F", "evidence": "IDA"},
                    {"id": "GO:0006281", "term": "DNA repair", "aspect": "P", "evidence": "IEA"},
                ],
                "kegg": ["hsa:672"],
                "interpro": [{"id": "IPR001357", "name": "BRCT"}],
                "pfam": [{"id": "PF00533", "name": "BRCT"}],
                "keywords": ["DNA damage"],
                "raw": {},
            }
        ], []


class FakeKegg:
    def pathways_for_gene(self, kegg_gene_id):
        return [
            {"pathway_id": "hsa03440", "name": "Homologous recombination"},
        ], []


class FakeGxa:
    def experiments_for_gene(self, symbol, organism=None):
        return [
            {
                "source": "expression_atlas",
                "accession": "E-MTAB-0000",
                "title": "RNA-seq of human tissues",
                "type": "Baseline",
                "species": "Homo sapiens",
                "factors": ["organism part"],
                "n_assays": 100,
                "technology": ["RNA-Seq mRNA"],
                "raw": {},
            }
        ], []

    def experiment_gene_values(self, accession, symbol, differential=False):
        return [
            {"sample": "liver", "value": 45.0, "percentile": 90.0},
            {"sample": "brain", "value": 3.0, "percentile": 15.0},
        ], _EV


def _resolver():
    return Resolver(
        jev=JevClient(mock=True),
        ncbi=FakeNcbi(),
        veupathdb=FakeVeuPathDB(),
        oma=FakeOma(),
        uniprot=FakeUniProt(),
        gxa=FakeGxa(),
        kegg=FakeKegg(),
        llm=LlmClient(mock=True),
    )


def test_resolve_numeric_id_mock_jev():
    r = _resolver()
    g = r.resolve("672", depth=1)
    assert "idtype:ncbi_gene" in g.nodes
    assert "taxon:9606" in g.nodes
    assert "assembly:GCF_000001405.40" in g.nodes
    predicates = {e.predicate for e in g.edges}
    assert {"is_a", "same_as", "in_organism", "in_assembly"} <= predicates
    # query node is a pure input: only is_a + same_as edges
    q_edges = [e for e in g.edges if e.subject == "672"]
    assert {e.predicate for e in q_edges} <= {"is_a", "same_as"}
    # the anchor carries the resource data
    assert any(
        e.subject == "ncbigene:672" and e.predicate == "in_organism"
        for e in g.edges
    )
    assert all(e.evidence or e.jev_question_id for e in g.edges)
    # s0 + s1 + proposals_triage + proposals (pubmed node exists at depth 1)
    assert g.metadata["jev_usage"]["total"]["calls"] == 4


def test_resolve_veupathdb_id_falls_back_to_ncbi():
    r = _resolver()
    g = r.resolve("PF3D7_0710100", depth=1)
    assert "idtype:veupathdb" in g.nodes
    assert "taxon:5833" in g.nodes


def test_veupathdb_aliases_become_idtype_nodes():
    r = _resolver()
    g = r.resolve("PF3D7_1477700", depth=1)
    assert "veupathdb:PF3D7_1477700" in g.nodes
    # alias list → IdType nodes; duplicate and self-reference skipped
    assert g.nodes["alias:Q8I5A7"].type == NodeType.ID_TYPE
    assert g.nodes["alias:Q8I5A7"].id_namespace == "uniprot/swissprot"
    edges = [e for e in g.edges if e.predicate == "has_alias"]
    assert {e.object for e in edges} == {"alias:Q8I5A7", "alias:PFF0740w"}
    assert all(e.subject == "veupathdb:PF3D7_1477700" for e in edges)
    assert all(e.confidence == 1.0 for e in edges)


def test_dataset_values_produce_data_claims():
    r = _resolver()
    g = r.resolve("PF3D7_1477700", depth=5)
    assert "dataset:DS_TEST" in g.nodes
    assert g.nodes["dataset:DS_TEST"].attrs["kind"] == "expression"
    # the mock LLM's canned proposal becomes a claim node off the dataset
    # via the data-claims pass (text pass would make a condition: node)
    edge = next(
        e
        for e in g.edges
        if e.subject == "dataset:DS_TEST" and e.predicate == "reports"
    )
    assert edge.object == "claim:mock proposal"
    assert edge.evidence[0].payload["n_rows"] == 2
    # GEO GDS + GXA datasets lazy-fetch values and get the same treatment;
    # the GSE series is skipped (series-matrix layout, not fetched)
    reports_from = {
        e.subject for e in g.edges if e.predicate == "reports"
    }
    assert "dataset:GDS9999" in reports_from
    assert "dataset:E-MTAB-0000" in reports_from
    assert "dataset:GSE12345" not in reports_from


def test_propose_false_skips_llm_passes():
    # quick mode: deterministic stages + JEV edge scoring still run, but
    # no LLM-proposed nodes/edges and no summary. (condition: nodes from
    # GXA factors are deterministic and still expected.)
    g = _resolver().resolve("672", depth=6, propose=False)
    assert "condition:mock proposal" not in g.nodes
    assert "claim:mock proposal" not in g.nodes
    assert not any(
        e.predicate in ("mentions", "reports") for e in g.edges
    )
    assert "summary" not in g.metadata
    # deterministic edges are unaffected
    assert g.edges


def test_summarize_populates_metadata():
    r = _resolver()
    g = r.resolve("672", depth=1, summarize=True)
    summary = g.metadata.get("summary")
    assert summary
    assert summary[0]["claim"] == "mock proposal"
    assert summary[0]["confidence"] == 0.5
    # off by default
    g2 = r.resolve("672", depth=1)
    assert "summary" not in g2.metadata


def test_summarize_on_existing_graph():
    # the /api/summarize path: summarize a graph built without it
    r = _resolver()
    g = r.resolve("672", depth=1)
    assert "summary" not in g.metadata
    r.summarize(g, {"symbol": "BRCA2", "organism": "Homo sapiens"})
    assert g.metadata["summary"][0]["claim"] == "mock proposal"


def test_serialize_rows_sorts_caps_tolerates():
    from bionym import proposals

    rows = [
        {"sample": "low", "value": "1", "percentile": "5.0"},
        {"sample": "high", "value": "9", "percentile": "99.0"},
        {"sample": "bad", "value": "x", "percentile": None},
    ]
    lines = proposals.serialize_rows(rows).split("\n")
    assert lines[0] == "sample | value | percentile"
    assert "high" in lines[1]  # highest percentile first
    many = [
        {"sample": f"s{i}", "value": "1", "percentile": "1"}
        for i in range(proposals.DATA_ROWS_MAX + 10)
    ]
    assert "more rows" in proposals.serialize_rows(many)


def test_depth_zero_classifies_only():
    r = _resolver()
    g = r.resolve("672", depth=0)
    assert "idtype:ncbi_gene" in g.nodes
    assert "taxon:9606" not in g.nodes


def test_depth_two_adds_related_assembly():
    r = _resolver()
    g = r.resolve("672", depth=2)
    assert "assembly:GCF_999999999.1" in g.nodes
    predicates = {e.predicate for e in g.edges}
    assert "related_assembly" in predicates
    assert "superseded_by" in predicates


def test_depth_three_adds_ortholog():
    r = _resolver()
    g = r.resolve("672", depth=3)
    assert "gene:BRCA1_HUMAN" in g.nodes
    predicates = {e.predicate for e in g.edges}
    assert "ortholog_of" in predicates


def test_depth_four_adds_annotation():
    r = _resolver()
    g = r.resolve("672", depth=4)
    assert "uniprot:P38398" in g.nodes
    assert "go:GO:0003677" in g.nodes
    assert "kegg:hsa:672" in g.nodes
    assert "kegg:hsa03440" in g.nodes
    assert g.nodes["kegg:hsa03440"].type == NodeType.PATHWAY
    assert "domain:IPR001357" in g.nodes
    predicates = {e.predicate for e in g.edges}
    assert {"same_as", "has_go_term", "in_pathway", "has_domain"} <= predicates
    # IDA evidence should outscore IEA
    conf = {e.object: e.confidence for e in g.edges if e.predicate == "has_go_term"}
    assert conf["go:GO:0003677"] > conf["go:GO:0006281"]


def test_depth_five_adds_expression():
    r = _resolver()
    g = r.resolve("672", depth=5)
    assert "dataset:GSE12345" in g.nodes
    assert "dataset:E-MTAB-0000" in g.nodes
    assert "condition:organism part" in g.nodes
    predicates = {e.predicate for e in g.edges}
    assert {"measured_in", "has_factor"} <= predicates


def test_depth_six_adds_remap():
    r = _resolver()
    g = r.resolve("672", depth=6)
    predicates = {e.predicate for e in g.edges}
    # GCF_000001405.40 + GCF_009914755.1 are in the record's annotations
    assert "annotated_in" in predicates
    # GCF_999999999.1 (S2 fake) is not annotated -> likely_present
    assert "likely_present" in predicates
    lp = [e for e in g.edges if e.predicate == "likely_present"]
    assert lp[0].object == "assembly:GCF_999999999.1"


def test_llm_proposals_add_gated_claims():
    r = _resolver()
    g = r.resolve("672", depth=1)
    # mock LLM proposes one claim per text-bearing node; mock JEV scores 0.5
    reports = [e for e in g.edges if e.predicate == "reports"]
    assert reports
    assert reports[0].subject == "pubmed:26289816"
    assert reports[0].object == "claim:mock proposal"
    assert reports[0].confidence == 0.5
    assert reports[0].jev_question_id.startswith("prop_")
    assert reports[0].evidence[0].source == "llm_proposal"


def test_parse_normalized():
    from bionym import proposals
    ok = '{"normalized": ["time series post infection", "  drug  treatment "]}'
    assert proposals.parse_normalized(ok, 2) == [
        "time series post infection", "drug treatment",
    ]
    # wrong length / missing key / garbage -> None (caller falls back)
    assert proposals.parse_normalized('{"normalized": ["only one"]}', 2) is None
    assert proposals.parse_normalized('{"proposals": []}', 1) is None
    assert proposals.parse_normalized("not json", 1) is None


def test_jev_ask_splits_on_max_tokens(monkeypatch):
    from bionym.jev import JevClient, JevError
    jev = JevClient(model="m", api_key="x")
    calls = []

    def fake_call(body):
        qs = body["questions"]
        calls.append(len(qs))
        if len(qs) > 2:
            raise JevError(
                'JEV API error 400: {"detail":{"error_type":"max_tokens_exceeded"}}'
            )
        return {k: {"noul": 0.5} for k in qs}, {"input_tokens": 1, "output_tokens": 1}

    monkeypatch.setattr(jev, "_call", fake_call)
    questions = {f"q{i}": {"type": "noul", "instructions": "x"} for i in range(4)}
    answers = jev.ask({"s": 1}, questions, stage="t")
    assert set(answers) == set(questions)
    assert calls == [4, 2, 2]
    # a single oversized question can't be split -> propagates
    import pytest

    def always_fail(body):
        raise JevError(
            'JEV API error 400: {"detail":{"error_type":"max_tokens_exceeded"}}'
        )

    monkeypatch.setattr(jev, "_call", always_fail)
    with pytest.raises(JevError):
        jev.ask({"s": 1}, {"q0": questions["q0"]}, stage="t")


def test_filter_by_confidence_and_url():
    from bionym.graph import KnowledgeGraph, Node, NodeType, Edge
    g = KnowledgeGraph(metadata={"query": "Q"})
    g.add_node(Node(id="Q", type=NodeType.GENE, attrs={"gene_id": "672"}))
    g.add_node(Node(id="kegg:hsa05290", type=NodeType.PATHWAY, id_namespace="kegg"))
    g.add_node(Node(id="dataset:GSE1", type=NodeType.DATASET))
    g.add_edge(Edge(subject="Q", predicate="in_pathway", object="kegg:hsa05290", confidence=0.8))
    g.add_edge(Edge(subject="Q", predicate="measured_in", object="dataset:GSE1", confidence=0.2))
    # urls
    assert g.nodes["Q"].url == "https://www.ncbi.nlm.nih.gov/gene/672"
    assert g.nodes["kegg:hsa05290"].url == "https://www.kegg.jp/entry/hsa05290"
    assert "geo/query/acc.cgi?acc=GSE1" in g.nodes["dataset:GSE1"].url
    # filter
    g.filter_by_confidence(0.5)
    assert len(g.edges) == 1 and g.edges[0].predicate == "in_pathway"
    assert "dataset:GSE1" not in g.nodes and "Q" in g.nodes
