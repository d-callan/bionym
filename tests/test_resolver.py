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


class FakeVeuPathDB:
    def lookup_gene(self, gene_id):
        return [], []


def test_resolve_numeric_id_mock_jev():
    r = Resolver(jev=JevClient(mock=True), ncbi=FakeNcbi(), veupathdb=FakeVeuPathDB())
    g = r.resolve("672", depth=1)
    assert "idtype:ncbi_gene" in g.nodes
    assert "taxon:9606" in g.nodes
    assert "assembly:GCF_000001405.40" in g.nodes
    predicates = {e.predicate for e in g.edges}
    assert {"is_a", "resolved_to", "in_organism", "in_assembly"} <= predicates
    assert all(e.evidence or e.jev_question_id for e in g.edges)
    assert g.metadata["jev_usage"]["total"]["calls"] == 2


def test_resolve_veupathdb_id_falls_back_to_ncbi():
    r = Resolver(jev=JevClient(mock=True), ncbi=FakeNcbi(), veupathdb=FakeVeuPathDB())
    g = r.resolve("PF3D7_0710100", depth=1)
    assert "idtype:veupathdb" in g.nodes
    assert "taxon:5833" in g.nodes


def test_depth_zero_classifies_only():
    r = Resolver(jev=JevClient(mock=True), ncbi=FakeNcbi(), veupathdb=FakeVeuPathDB())
    g = r.resolve("672", depth=0)
    assert "idtype:ncbi_gene" in g.nodes
    assert "taxon:9606" not in g.nodes
