from delphi_demo import icd10_pl
from delphi_demo.slownik import wczytaj_nazwy_oficjalne

MINI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hcd xmlns="http://rsk.rejestrymedyczne.csioz.gov.pl" hcd="00CD10">
  <node code="A00-A09"><name>Choroby zakaźne jelit</name>
    <node code="A00"><name>Cholera</name>
      <node code="A00.0"><name>Cholera wywołana przez Vibrio cholerae 01</name></node>
    </node>
  </node>
  <node code="F00*"><name>Otępienie w chorobie Alzheimera (G30.–†)</name></node>
  <node code="A17†"><name>Gruźlicze zapalenie opon mózgowych</name></node>
</hcd>
"""


def test_parsuj_tylko_kody_3_znakowe_bez_znacznikow(tmp_path):
    xml = tmp_path / "mini.xml"
    xml.write_text(MINI_XML, encoding="utf-8")
    assert icd10_pl.parsuj(xml) == {"A00": "Cholera",
                                    "F00": "Otępienie w chorobie Alzheimera (G30.–†)",
                                    "A17": "Gruźlicze zapalenie opon mózgowych"}


def test_przygotuj_i_wczytaj_bez_sieci(tmp_path):
    xml, csv = tmp_path / "mini.xml", tmp_path / "icd10_pl.csv"
    xml.write_text(MINI_XML, encoding="utf-8")
    assert icd10_pl.przygotuj(xml, csv, min_kategorii=1) == 3        # XML jest, więc nic nie pobiera
    assert csv.read_text(encoding="utf-8").startswith("# Polskie nazwy kategorii ICD-10")
    assert wczytaj_nazwy_oficjalne(csv) == {"A00": "cholera",
                                            "F00": "otępienie w chorobie Alzheimera",   # bez odsyłacza
                                            "A17": "gruźlicze zapalenie opon mózgowych"}
