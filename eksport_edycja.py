#!/usr/bin/env python3
"""Buduje eksport/prezentacja_edycja.html: ten sam deck co prezentacja.html,
plus przycisk „Edytuj” do poprawiania słów na slajdach w Artifact na claude.ai.

Poprawki zapisują się w bazie danych Artifactu (capability `db`, kolekcja
`poprawki`), a nie w samym HTML. Źródłem prawdy zostają slides/*.html:
poprawki z Artifactu Claude wczytuje (ArtifactData → list „poprawki”)
i nanosi na slides/*.html, po czym składa deck od nowa (assemble.py).

Każdy dokument poprawki: {slajd (data-label), nr_slajdu, idx, przed_tekst,
przed_html, html, tekst, kiedy}. Dopasowanie przy nanoszeniu: po etykiecie
slajdu i tekście „przed” (idx jest tylko podpowiedzią).

Uruchomienie: python3 assemble.py && python3 eksport_edycja.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "prezentacja.html"
OUT = ROOT / "eksport" / "prezentacja_edycja.html"

TYTUL = "Trafny po 17 lekarzach"

DODATEK = r"""
<style id="edycja-styl">
  .ed-pasek {
    position: fixed; z-index: 2147483000;
    left: calc(16px + env(safe-area-inset-left, 0px));
    bottom: calc(16px + env(safe-area-inset-bottom, 0px));
    display: flex; gap: 8px; align-items: center;
    font: 600 14px/1.2 Inter, system-ui, sans-serif; color: #f4f1ff;
  }
  .ed-pasek > * { position: static !important; margin: 0 !important; flex: none; transform: none !important; }
  .ed-pasek button {
    font: inherit; color: inherit; cursor: pointer; white-space: nowrap;
    background: rgba(20, 16, 40, 0.88); border: 1px solid rgba(255, 175, 123, 0.55);
    border-radius: 999px; padding: 9px 16px; backdrop-filter: blur(6px);
  }
  .ed-pasek button:hover { background: rgba(58, 28, 125, 0.95); }
  .ed-pasek button:focus-visible { outline: 2px solid #00bcde; outline-offset: 2px; }
  .ed-pasek button[disabled] { opacity: 0.45; cursor: not-allowed; }
  .ed-pasek button[aria-pressed="true"] { background: #ffaf7b; color: #1a1030; border-color: #ffaf7b; }
  .ed-stan {
    background: rgba(20, 16, 40, 0.88); border-radius: 999px; padding: 8px 14px;
    color: rgba(244, 241, 255, 0.8); font-weight: 500;
  }
  .ed-stan:empty { display: none; }
  .ed-on [data-ed] { outline: 1px dashed rgba(255, 175, 123, 0.35); outline-offset: 4px; border-radius: 4px; cursor: text; }
  .ed-on [data-ed]:hover { outline-color: rgba(255, 175, 123, 0.8); }
  .ed-on [data-ed]:focus { outline: 2px solid #ffaf7b; background: rgba(255, 175, 123, 0.06); }
  .ed-on .build { opacity: 1 !important; transform: none !important; }
  /* znacznik „poprawione” tylko w trybie edycji: cienka kreska z lewej, bez zmiany układu */
  .ed-on [data-ed][data-ed-zmienione]:not(:focus) { box-shadow: -10px 0 0 -7px #00bcde; }
  @media print { .ed-pasek { display: none !important; } }
</style>
<div class="ed-pasek no-print" id="ed-pasek">
  <button type="button" id="ed-przelacz" aria-pressed="false" disabled title="Łączę z zapisem poprawek…">✏️ Edytuj</button>
  <button type="button" id="ed-przywroc" hidden title="Przywróć oryginalny tekst zaznaczonego fragmentu">↺ Oryginał</button>
  <span class="ed-stan" id="ed-stan" role="status" aria-live="polite"></span>
</div>
<script id="edycja-skrypt">
(function () {
  "use strict";
  var BLOKI = "div,p,ul,ol,li,table,section,h1,h2,h3,h4,h5,h6,svg,img,canvas,button,pre,input,textarea,select,video";
  var pasek = document.getElementById("ed-pasek");
  var btn = document.getElementById("ed-przelacz");
  var btnOryg = document.getElementById("ed-przywroc");
  var stan = document.getElementById("ed-stan");
  var db = null, wlaczona = false, aktywny = null;
  var elementy = [];            // {el, slajd, nr, idx, przedTekst, przedHtml, id}
  var zapisane = {};            // id -> html zapisany w bazie
  var kolejka = {};             // id -> Promise (jeden zapis naraz na dokument)

  function tekst(el) { return (el.textContent || "").replace(/\s+/g, " ").trim(); }
  function pokaz(msg) { stan.textContent = msg || ""; }

  function liscBlok(el) {
    if (!el || el.nodeType !== 1) return false;
    if (/^(SCRIPT|STYLE|TEMPLATE|SPAN|STRONG|EM|B|I|A|SUP|SUB|CODE|BR|SMALL)$/.test(el.tagName)) return false;
    if (el.closest("#inference-anim, .ed-pasek, [id^='qr-'], .page-num")) return false;
    if (el.querySelector(BLOKI)) return false;
    return /[0-9A-Za-zÀ-ž]/.test(tekst(el));   // same strzałki i ikonki nie są do edycji
  }

  function zbierz() {
    var sekcje = document.querySelectorAll("deck-stage > section");
    sekcje.forEach(function (sek, nr) {
      var slajd = sek.getAttribute("data-label") || ("slajd-" + (nr + 1));
      var idx = 0;
      sek.querySelectorAll("*").forEach(function (el) {
        if (!liscBlok(el) || liscBlok(el.parentElement)) return;
        var id = "s" + (nr + 1) + "-e" + idx;
        el.setAttribute("data-ed", id);
        elementy.push({ el: el, slajd: slajd, nr: nr + 1, idx: idx, id: id,
                        przedTekst: tekst(el), przedHtml: el.innerHTML.trim() });
        idx++;
      });
    });
  }

  function znajdz(d) {
    var naSlajdzie = elementy.filter(function (e) { return e.slajd === d.slajd; });
    var poIdx = naSlajdzie.find(function (e) { return e.idx === d.idx && e.przedTekst === d.przed_tekst; });
    return poIdx || naSlajdzie.find(function (e) { return e.przedTekst === d.przed_tekst; }) || null;
  }

  function oznacz(e, zmienione) {
    if (zmienione) e.el.setAttribute("data-ed-zmienione", ""); else e.el.removeAttribute("data-ed-zmienione");
  }

  function zastosuj(snap) {
    var nieDopasowane = 0, ile = 0;
    snap.docs.forEach(function (doc) {
      var d = doc.data(); if (!d) return;
      var e = znajdz(d);
      if (!e) { nieDopasowane++; return; }
      ile++;
      zapisane[e.id] = d.html;
      if (e.el !== document.activeElement && e.el.innerHTML.trim() !== d.html) e.el.innerHTML = d.html;
      oznacz(e, true);
    });
    snap.docChanges().forEach(function (ch) {
      if (ch.type !== "removed") return;
      var d = ch.doc.data(); var e = d && znajdz(d);
      if (e && e.el !== document.activeElement) { e.el.innerHTML = e.przedHtml; delete zapisane[e.id]; oznacz(e, false); }
    });
    if (!wlaczona) pokaz(ile ? ("Poprawek: " + ile + (nieDopasowane ? (", niedopasowanych: " + nieDopasowane) : "")) : "");
  }

  function zapisz(e) {
    var html = e.el.innerHTML.trim();
    var poprzedni = Object.prototype.hasOwnProperty.call(zapisane, e.id) ? zapisane[e.id] : e.przedHtml;
    if (html === poprzedni) return;
    var ref = db.collection("poprawki").doc(e.id);
    var wroc = html === e.przedHtml;
    var krok = function () {
      pokaz("Zapisuję…");
      var p = wroc ? ref.delete() : ref.set({
        slajd: e.slajd, nr_slajdu: e.nr, idx: e.idx,
        przed_tekst: e.przedTekst, przed_html: e.przedHtml,
        html: html, tekst: tekst(e.el), kiedy: new Date().toISOString()
      });
      return p.then(function () {
        if (wroc) delete zapisane[e.id]; else zapisane[e.id] = html;
        oznacz(e, !wroc);
        pokaz(wroc ? "Przywrócono oryginał" : "Zapisano");
      }, function (err) {
        var code = err && err.code;
        pokaz(code === "invalid_argument" ? "Brak uprawnień do zapisu poprawek" :
              code === "quota_exceeded" ? "Baza poprawek jest pełna" :
              "Nie zapisano. Sprawdź połączenie i edytuj ten fragment jeszcze raz.");
      });
    };
    kolejka[e.id] = (kolejka[e.id] || Promise.resolve()).then(krok, krok);
  }

  function elementDla(node) {
    var el = node && node.closest ? node.closest("[data-ed]") : null;
    return el ? elementy.find(function (e) { return e.el === el; }) : null;
  }

  function ustawTryb(on) {
    wlaczona = on;
    document.documentElement.classList.toggle("ed-on", on);
    btn.setAttribute("aria-pressed", on ? "true" : "false");
    btn.textContent = on ? "✓ Zakończ edycję" : "✏️ Edytuj";
    elementy.forEach(function (e) {
      if (on) { e.el.setAttribute("contenteditable", "true"); e.el.setAttribute("spellcheck", "true"); }
      else { e.el.removeAttribute("contenteditable"); e.el.removeAttribute("spellcheck"); }
    });
    var stage = document.querySelector("deck-stage");
    var tz = stage && stage.shadowRoot && stage.shadowRoot.getElementById("ed-tapzones");
    if (stage && stage.shadowRoot && !tz) {
      tz = document.createElement("style"); tz.id = "ed-tapzones";
      stage.shadowRoot.appendChild(tz);
    }
    if (tz) tz.textContent = on ? ".tapzones { display: none !important; }" : "";
    btnOryg.hidden = !on;
    if (!on && document.activeElement && document.activeElement.blur) document.activeElement.blur();
    pokaz(on ? "Kliknij tekst na slajdzie i pisz. Zapis po kliknięciu obok." : "");
  }

  // Cyfry przeskakują do slajdu, a strzałki zmieniają slajd: w edytowanym tekście
  // klawisze mają tylko pisać. stopPropagation nie blokuje wpisywania znaków.
  window.addEventListener("keydown", function (ev) {
    var t = ev.target;
    if (!t || !t.isContentEditable) return;
    if (ev.key === "Escape") { t.blur(); return; }
    ev.stopPropagation();
  }, true);

  // Wklejamy sam tekst, bez formatowania z innych stron.
  document.addEventListener("paste", function (ev) {
    var t = ev.target && ev.target.closest && ev.target.closest("[data-ed]");
    if (!t || !wlaczona) return;
    ev.preventDefault();
    var txt = (ev.clipboardData || window.clipboardData).getData("text/plain");
    document.execCommand("insertText", false, txt);
  });

  document.addEventListener("focusin", function (ev) { var e = elementDla(ev.target); if (e) aktywny = e; });
  document.addEventListener("focusout", function (ev) {
    var e = elementDla(ev.target);
    if (e && db && wlaczona) zapisz(e);
  });

  btn.addEventListener("click", function () { ustawTryb(!wlaczona); });
  btnOryg.addEventListener("mousedown", function (ev) { ev.preventDefault(); });
  btnOryg.addEventListener("click", function () {
    if (!aktywny) { pokaz("Najpierw kliknij fragment tekstu"); return; }
    aktywny.el.innerHTML = aktywny.przedHtml;
    zapisz(aktywny);
  });

  function start() {
    zbierz();
    if (!window.claude || typeof window.claude.use !== "function") {
      btn.title = "Edycja działa tylko w Artifact na claude.ai"; return;
    }
    window.claude.use("db").then(function (d) {
      if (!d) { btn.title = "Zapis poprawek niedostępny w tym widoku"; return; }
      db = d;
      var gotowe = false;
      var timer = setTimeout(function () { if (!gotowe) pokaz("Nie mogę jeszcze wczytać poprawek"); }, 10000);
      db.collection("poprawki").onSnapshot(function (snap) {
        if (!snap.metadata.fromCache && !gotowe) { gotowe = true; clearTimeout(timer); }
        zastosuj(snap);
      }, function () { clearTimeout(timer); pokaz("Poprawki niedostępne"); });
      btn.disabled = false;
      btn.title = "Włącz edycję słów na slajdach";
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
</script>
"""


def main():
    h = SRC.read_text(encoding="utf-8")
    m = re.search(r"</head>\s*<body[^>]*>", h)
    if not m or not h.rstrip().endswith("</body></html>"):
        sys.exit("Nie rozpoznaję struktury prezentacja.html (head/body)")
    head = h[: m.start()]
    head = re.sub(r"^\s*<!DOCTYPE html>\s*<html[^>]*>\s*<head>\s*", "", head, flags=re.I)
    head = re.sub(r"<title>.*?</title>", f"<title>{TYTUL}</title>", head, count=1, flags=re.S)
    body = h[m.end():].rstrip()[: -len("</body></html>")]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(head + "\n" + body + DODATEK, encoding="utf-8")
    print(f"OK: {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
