// GENERATION_STEPS dla slajdu „Krok po kroku" (o10_inference.html).
// Źródło: ślad nazwy „angioma cenine" z komórki GENEROWANIE notebooka (przebieg Kamila, 31.08.2026).
// Format = eksport z komórki „ŚLAD GENEROWANIA → JSON POD SLAJD"; pole desc dopisane ręcznie.
// ␣ = spacja (na slajdzie zwykła spacja byłaby niewidoczna). „…" = suma pozostałych liter.
// assemble.py wstawia tę tablicę w miejsce `const GENERATION_STEPS = [...]` w skrypcie decku.
[
  {
    context: '%',
    probs: [['c', 0.19], ['a', 0.16], ['s', 0.16], ['m', 0.13], ['p', 0.12], ['…', 0.24]],
    chosen: 'a', chosen_p: 0.16, top: 'c',
    desc: '🎲 Start: faworytem było „c" (19%), ale wylosowaliśmy „a" (16%). Na początku pasuje wiele liter — rozkład jest prawie płaski.'
  },
  {
    context: '%a',
    probs: [['n', 0.38], ['c', 0.23], ['s', 0.10], ['p', 0.08], ['l', 0.05], ['…', 0.16]],
    chosen: 'n', chosen_p: 0.38, top: 'n',
    desc: 'Losujemy — pada „n" (38%), najbardziej prawdopodobna litera.'
  },
  {
    context: '%an',
    probs: [['g', 0.38], ['a', 0.33], ['u', 0.07], ['t', 0.06], ['e', 0.05], ['…', 0.11]],
    chosen: 'g', chosen_p: 0.38, top: 'g',
    desc: '„g" (38%) minimalnie wygrywa z „a" (33%). „ang-" to częsty początek: angina, angioma, angiopathia…'
  },
  {
    context: '%ang',
    probs: [['i', 0.74], ['r', 0.15], ['e', 0.08], ['l', 0.02], ['u', 0.00], ['…', 0.01]],
    chosen: 'i', chosen_p: 0.74, top: 'i',
    desc: 'Rozkład się wyostrza: „i" ma 74%. Im dłuższy kontekst, tym mniej wątpliwości.'
  },
  {
    context: '%angi',
    probs: [['n', 0.52], ['o', 0.31], ['t', 0.09], ['c', 0.03], ['a', 0.02], ['…', 0.03]],
    chosen: 'o', chosen_p: 0.31, top: 'n',
    desc: '🎲 Faworytem było „n" (52% — w stronę „angina"), wylosowaliśmy „o" (31%). Jedno losowanie zmienia kierunek całej nazwy.'
  },
  {
    context: '%angio',
    probs: [['m', 0.83], ['s', 0.11], ['n', 0.04], ['c', 0.01], ['p', 0.01], ['…', 0.00]],
    chosen: 'm', chosen_p: 0.83, top: 'm',
    desc: '„m" (83%) — model rozpoznaje znany rdzeń „angiom-".'
  },
  {
    context: '%angiom',
    probs: [['a', 1.00], ['y', 0.00], ['e', 0.00], ['i', 0.00], ['u', 0.00], ['…', 0.00]],
    chosen: 'a', chosen_p: 1.00, top: 'a',
    desc: 'Praktycznie pewność: „a" ma 100%. „angioma" to prawdziwa nazwa ze zbioru treningowego.'
  },
  {
    context: '%angioma',
    probs: [['␣', 0.55], ['!', 0.44], ['t', 0.01], ['e', 0.00], ['r', 0.00], ['…', 0.00]],
    chosen: '␣', chosen_p: 0.55, top: '␣',
    desc: 'Rozstaje: koniec nazwy („!" 44%) czy ciąg dalszy (spacja ␣ 55%)? Wypadła spacja — nazwa będzie dwuczłonowa.'
  },
  {
    context: '%angioma␣',
    probs: [['p', 0.16], ['c', 0.11], ['a', 0.10], ['s', 0.09], ['l', 0.08], ['…', 0.46]],
    chosen: 'c', chosen_p: 0.11, top: 'p',
    desc: '🎲 Drugi człon może zacząć się prawie od wszystkiego — faworyt „p" ma tylko 16%. Wylosowaliśmy „c" (11%).'
  },
  {
    context: '%angioma␣c',
    probs: [['o', 0.77], ['a', 0.07], ['h', 0.07], ['e', 0.05], ['r', 0.03], ['…', 0.01]],
    chosen: 'e', chosen_p: 0.05, top: 'o',
    desc: '🎲 Model stawiał na „o" (77%), a wylosowaliśmy „e" — tylko 5%. Od tego miejsca nazwa schodzi z utartych ścieżek.'
  },
  {
    context: '%angioma␣ce',
    probs: [['r', 0.85], ['n', 0.12], ['s', 0.01], ['l', 0.01], ['m', 0.00], ['…', 0.01]],
    chosen: 'n', chosen_p: 0.12, top: 'r',
    desc: '🎲 Model dawał „r" aż 85% (jak w „cerebri"), ale losowanie wybrało mniej prawdopodobne „n" (12%). Słabszy kandydat też czasem wygrywa.'
  },
  {
    context: '%angioma␣cen',
    probs: [['i', 0.44], ['t', 0.15], ['s', 0.15], ['e', 0.06], ['␣', 0.06], ['…', 0.14]],
    chosen: 'i', chosen_p: 0.44, top: 'i',
    desc: 'Losujemy — pada „i" (44%), najbardziej prawdopodobna litera.'
  },
  {
    context: '%angioma␣ceni',
    probs: [['s', 0.30], ['c', 0.22], ['!', 0.21], ['␣', 0.07], ['t', 0.06], ['n', 0.02], ['…', 0.12]],
    chosen: 'n', chosen_p: 0.02, top: 's',
    desc: '🎲 „n" miało tylko 2% — spoza pierwszej piątki — a jednak wypadło. Model nigdy nie mówi „nie wiem": zawsze coś wylosuje.'
  },
  {
    context: '%angioma␣cenin',
    probs: [['e', 0.78], ['a', 0.12], ['!', 0.03], ['g', 0.02], ['i', 0.02], ['…', 0.03]],
    chosen: 'e', chosen_p: 0.78, top: 'e',
    desc: 'Wracamy na utarty szlak: „e" (78%).'
  },
  {
    context: '%angioma␣cenine',
    probs: [['u', 0.32], ['!', 0.22], ['s', 0.22], ['r', 0.07], ['t', 0.06], ['…', 0.11]],
    chosen: '!', chosen_p: 0.22, top: 'u',
    desc: '🎲 STOP (!) wylosowany z 22% (faworyt: „u", 32%). Wynik: „angioma cenine" — brzmi łacińsko, a nie istnieje. Halucynacja w miniaturze.'
  }
]
