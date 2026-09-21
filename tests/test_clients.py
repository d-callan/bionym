"""Fixture tests for client normalizers — the parsing layer where real
bugs have come from (Datasets v2 nesting, GEO esummary fields, OMA xrefs).
"""

from bionym.clients.ncbi import NcbiClient
from bionym.clients.oma import OmaClient
from bionym.clients.uniprot import UniProtClient
from bionym.clients.expression import ExpressionAtlasClient
from bionym.jev import JevClient
from bionym.graph import KnowledgeGraph, Node, NodeType, Edge
from bionym.report import render_html


# -- NCBI -----------------------------------------------------------------

def test_normalize_gene_report_datasets_v2():
    # Datasets v2 nests gene fields under report["gene"]; assemblies under
    # report["gene"]["annotations"].
    rec = NcbiClient._normalize_gene_report({
        "gene": {
            "gene_id": 672,
            "symbol": "BRCA1",
            "description": "BRCA1 DNA repair associated",
            "locus_tag": None,
            "tax_id": 9606,
            "taxname": "Homo sapiens",
            "annotations": [
                {"assembly_accession": "GCF_000001405.40",
                 "genomic_locations": [{"genomic_accession_version": "NC_000017.11"}]},
                {"assembly_accession": "GCF_009914755.1",
                 "genomic_locations": [{"genomic_accession_version": "NC_060941.1"}]},
            ],
        }
    })
    assert rec["gene_id"] == "672"
    assert rec["symbol"] == "BRCA1"
    assert rec["organism"] == "Homo sapiens"
    assert rec["assembly_accessions"] == ["GCF_000001405.40", "GCF_009914755.1"]
    assert rec["assembly_accession"] == "GCF_000001405.40"
    assert "NC_000017.11" in rec["genomic_accessions"]


def test_normalize_gene_summary_eutils():
    rec = NcbiClient._normalize_gene_summary({
        "uid": "672",
        "name": "BRCA1",
        "description": "BRCA1 DNA repair associated",
        "organism": {"taxid": 9606, "scientificname": "Homo sapiens"},
        "genomicinfo": [{"chraccver": "NC_000017.11"}],
    })
    assert rec["gene_id"] == "672"
    assert rec["symbol"] == "BRCA1"
    assert rec["tax_id"] == 9606
    assert rec["genomic_accessions"] == ["NC_000017.11"]


def test_normalize_gds_summary_geo():
    rec = NcbiClient._normalize_gds_summary({
        "accession": "GSE308090",
        "title": "ISG15 promotes PARP inhibitor resistance",
        "taxon": "Homo sapiens",
        "n_samples": 12,
        "gdstype": "Expression profiling by high throughput sequencing",
        "ptechtype": "RNA-seq",
    })
    assert rec["source"] == "geo"
    assert rec["accession"] == "GSE308090"
    assert rec["n_samples"] == 12


def test_normalize_assembly_report():
    rec = NcbiClient._normalize_assembly_report({
        "accession": "GCF_000001405.40",
        "assembly_info": {"assembly_name": "GRCh38.p14",
                          "assembly_level": "Complete Genome",
                          "refseq": True, "submission_date": "2022-02-03"},
        "organism": {"organism_name": "Homo sapiens", "tax_id": 9606},
    })
    assert rec["accession"] == "GCF_000001405.40"
    assert rec["name"] == "GRCh38.p14"
    assert rec["level"] == "Complete Genome"
    assert rec["refseq"] is True


def test_parse_taxon_xml_species_rank():
    xml = """<TaxaSet><Taxon><TaxId>5835</TaxId>
      <ScientificName>Plasmodium falciparum Camp</ScientificName><Rank>no rank</Rank>
      <LineageEx>
        <Taxon><TaxId>5833</TaxId><ScientificName>Plasmodium falciparum</ScientificName><Rank>species</Rank></Taxon>
        <Taxon><TaxId>5820</TaxId><ScientificName>Plasmodium</ScientificName><Rank>genus</Rank></Taxon>
      </LineageEx></Taxon></TaxaSet>"""
    rec = NcbiClient._parse_taxon_xml(xml)
    assert rec["tax_id"] == 5835
    assert rec["species_tax_id"] == 5833          # normalized to species rank
    assert rec["species_name"] == "Plasmodium falciparum"
    assert rec["rank"] == "no rank"


# -- OMA ------------------------------------------------------------------

def test_normalize_ortholog_captures_entry_nr():
    rec = OmaClient._normalize_ortholog({
        "entry_nr": 25667275,
        "omaid": "PLABA00208",
        "canonicalid": "A0A122IJL9",
        "species": {"name": "Plasmodium berghei", "taxon_id": 5823},
        "rel_type": "1:1",
    })
    assert rec["entry_nr"] == 25667275   # needed for the xref endpoint
    assert rec["canonical_id"] == "A0A122IJL9"
    assert rec["rel_type"] == "1:1"
    assert rec["tax_id"] == 5823


# -- UniProt --------------------------------------------------------------

def test_normalize_uniprot_entry_xrefs():
    rec = UniProtClient._normalize_entry({
        "primaryAccession": "P38398",
        "uniProtkbId": "BRCA1_HUMAN",
        "genes": [{"geneName": {"value": "BRCA1"}}],
        "proteinDescription": {"recommendedName": {"fullName": {"value": "Breast cancer type 1"}}},
        "organism": {"scientificName": "Homo sapiens", "taxonId": 9606},
        "uniProtKBCrossReferences": [
            {"database": "GO", "id": "GO:0000155",
             "properties": [{"key": "GoTerm", "value": "P:phosphoprotein binding"},
                            {"key": "GoEvidenceType", "value": "IDA"}]},
            {"database": "KEGG", "id": "hsa05290"},
            {"database": "Pfam", "id": "PF00504",
             "properties": [{"key": "EntryName", "value": "BRCA1"}]},
            {"database": "InterPro", "id": "IPR015257",
             "properties": [{"key": "EntryName", "value": "Brca-1"}]},
        ],
    })
    assert rec["accession"] == "P38398"
    assert rec["gene_name"] == "BRCA1"
    assert rec["tax_id"] == 9606
    assert rec["go_terms"][0]["id"] == "GO:0000155"
    assert rec["go_terms"][0]["aspect"] == "P"          # stripped from "P:..."
    assert rec["go_terms"][0]["term"] == "phosphoprotein binding"
    assert rec["go_terms"][0]["evidence"] == "IDA"
    assert rec["kegg"] == ["hsa05290"]
    assert rec["pfam"] == [{"id": "PF00504", "name": "BRCA1"}]
    assert rec["interpro"] == [{"id": "IPR015257", "name": "Brca-1"}]


# -- Expression Atlas -----------------------------------------------------

def test_normalize_gxa_experiment():
    rec = ExpressionAtlasClient._normalize({
        "experimentAccession": "E-MTAB-2770",
        "experimentDescription": "RNA-seq of 1019 human cancer cell lines",
        "experimentType": "Baseline",
        "species": "Homo sapiens",
        "experimentalFactors": [{"name": "organism part"}],
        "numberOfAssays": 1019,
        "technologyType": ["rna-seq"],
    })
    assert rec["accession"] == "E-MTAB-2770"
    assert rec["type"] == "Baseline"
    assert rec["n_assays"] == 1019
    assert rec["factors"][0]["name"] == "organism part"


# -- JEV mock -------------------------------------------------------------

def test_jev_mock_answers_and_usage():
    jev = JevClient(mock=True)
    answers = jev.ask(
        {"x": 1},
        {
            "pick": {"type": "choice", "criteria": {"a": "A", "b": "B"}},
            "rate": {"type": "score", "criteria": ["low", "mid", "high"]},
            "flag": {"type": "noul"},
        },
        stage="test",
    )
    assert answers["pick"]["choice"] == "a"          # first option
    assert answers["rate"]["score"] == 1.0           # midpoint of 3 levels
    assert "noul" in answers["flag"]
    assert jev.total_usage()["calls"] == 1


# -- report ---------------------------------------------------------------

def test_render_html_embeds_graph():
    g = KnowledgeGraph(metadata={"input": "X", "query": "X", "stages": [], "jev_usage": {}})
    g.add_node(Node(id="X", type=NodeType.GENE, label="X", attrs={"gene_id": "1"}))
    g.add_node(Node(id="kegg:k1", type=NodeType.PATHWAY, id_namespace="kegg"))
    g.add_edge(Edge(subject="X", predicate="in_pathway", object="kegg:k1", confidence=0.8))
    html = render_html(g)
    assert "BioNym" in html
    assert '"nodes"' in html and '"edges"' in html
    assert "kegg:k1" in html


# -- CLI ------------------------------------------------------------------

def test_cli_resolve_mock(tmp_path):
    from typer.testing import CliRunner
    from bionym.cli import app
    out = tmp_path / "g.json"
    r = CliRunner().invoke(app, ["resolve", "672", "-o", str(out), "--mock-jev", "--depth", "1"])
    assert r.exit_code == 0, r.output
    assert out.exists()
    import json
    g = json.loads(out.read_text())
    assert "nodes" in g and "edges" in g
