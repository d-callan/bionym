from idresolver.jev import JevClient
from idresolver.resolver import Resolver


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
                "genomic_accessions": ["NC_000013.11"],
                "raw": {},
            }
        ], []

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
            }
        ], []


class FakeVeuPathDB:
    def lookup_gene(self, gene_id):
        return [], []


class FakeOma:
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
                "kegg": ["hsa03440"],
                "interpro": [{"id": "IPR001357", "name": "BRCT"}],
                "pfam": [{"id": "PF00533", "name": "BRCT"}],
                "keywords": ["DNA damage"],
                "raw": {},
            }
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


def _resolver():
    return Resolver(
        jev=JevClient(mock=True),
        ncbi=FakeNcbi(),
        veupathdb=FakeVeuPathDB(),
        oma=FakeOma(),
        uniprot=FakeUniProt(),
        gxa=FakeGxa(),
    )


def test_resolve_numeric_id_mock_jev():
    r = _resolver()
    g = r.resolve("672", depth=1)
    assert "idtype:ncbi_gene" in g.nodes
    assert "taxon:9606" in g.nodes
    assert "assembly:GCF_000001405.40" in g.nodes
    predicates = {e.predicate for e in g.edges}
    assert {"is_a", "resolved_to", "in_organism", "in_assembly"} <= predicates
    assert all(e.evidence or e.jev_question_id for e in g.edges)
    assert g.metadata["jev_usage"]["total"]["calls"] == 2


def test_resolve_veupathdb_id_falls_back_to_ncbi():
    r = _resolver()
    g = r.resolve("PF3D7_0710100", depth=1)
    assert "idtype:veupathdb" in g.nodes
    assert "taxon:5833" in g.nodes


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
    assert "kegg:hsa03440" in g.nodes
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
