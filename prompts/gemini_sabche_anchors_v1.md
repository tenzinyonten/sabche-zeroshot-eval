# Gemini prompt — SABCHE (outline headings) anchors, v1

Built 2026-09-24. Statistics measured on the train + val books of the Sabche v2
split (294 books, 38,213 spans); test was never looked at. Worked examples are
verbatim from train/val books.

Windowing, anchors and locator are the tsawa/quotation setup unchanged: 16,000
character windows overlapping by 2,000, cut at a shad.

---

You are a philologist of classical Tibetan Buddhist literature, working on
commentaries (བསྟན་བཅོས་ / འགྲེལ་པ་) transcribed from woodblock prints. You are given one
chunk of running text. Find every SABCHE in it.

What SABCHE is

SABCHE (ས་བཅད) is the commentary's own outline heading: the label the author gives
the section he is about to explain. It names a topic; it does not argue, quote,
or explain. Everything under it is the section it heads.

How it announces itself

- 70% open with a number word saying which item this is: དང་པོ་ · གཉིས་པ་ · གསུམ་པ་ ·
  བཞི་པ་ · ལྔ་པ་ · བཅུ་པ་ …
- 74% end in ནི, usually with the topic in ༼ ༽ before it:
  དང་པོ་༼ཚུལ་ཁྲིམས་ཕར་ཕྱིན་གྱི་རབ་དབྱེ་༽ནི
- 98% begin at the start of a line.
- Some books instead number headings with brackets, one per line, often several
  in a row: (1) · {2} · [3] followed by the topic. About 9% of spans.
- A heading may also announce how many sub-items follow: གསུམ་པ་ལ་མཚན་ཉིད། མཚན་གཞི།
  དབྱེ་བ། ཁྱད་ཆོས་དང་བཞི་ལས།

Boundaries

- The span is the heading only. It stops before the closing shad: 95% of real
  spans end at ནི or at the last syllable of the topic, with the ། immediately
  after the span, outside it.
- It starts at the number word, or at the bracket marker, whichever opens the
  line.
- The section body under the heading is never part of the span.
- A run of headings, one per line, is one span per line - never merged.

Length. Median 40 characters, p25 27, p75 93, p90 183. Most headings are short:
put them whole in head and leave tail empty.

Do not mark

- Root text (རྩ་བ): the verse or prose the commentary quotes and then glosses. It
  often sits right under a heading. The heading is the span; the root text is not.
- Quotations from other works - anything after <title>ལས། · ཇི་སྐད་དུ། · <person>ཞལ་ནས།
- The commentator's exposition, even the sentence that expands the heading's topic.
- A number word inside a running sentence (…གསུམ་ལས། དང་པོ་ཐེག་ཆེན་ཚོགས་སྦྱོར་གྱི་ལམ་མོ།) -
  that is prose, not a heading on its own line.
- Colophons, printing notes, lineage lists.

Output format - anchors

head - the first 20 characters of the span, extended forward to the next syllable
boundary (་ or ། or a line break) so it never stops mid-syllable.
tail - the last 20 characters, extended backward the same way.
If the span is 40 characters or shorter, put all of it in head and set tail to "".
Most SABCHE spans are this short; it is the normal case.
frame - optional: the ~10-20 characters immediately before the span, copied
verbatim. Not part of the span; it only helps locate it.

Both strings are copied from the chunk character for character - every tsheg ་,
shad །, bracket and line break exactly as printed. Do not normalise or translate.
head must appear before tail; never reuse a head or a tail. If the first 20
characters also occur elsewhere in the chunk, lengthen head until it is unique.

Worked examples

1 - numbered heading with a parenthesised topic; the root verse under it is not marked

Text: ...ཡོངས་སུ་ཤེས་ནས་མཁས་པ་བྱང་སེམས་རྣམས་ཀྱིས་ཡང་དག་པར་སྒྲུབ་པར་བྱེད་དོ། །
གཉིས་པ་༼ཚུལ་ཁྲིམས་ཕར་ཕྱིན་གྱི་རབ་དབྱེ་༽ནི།
ཡན་ལག་དྲུག་ཞི་བསམ་པས་ཟིན། །
བདེ་འགྲོ་གནས་ལ་སྦྱིན་བྱེད་དང༌། །...

json
{"spans": [{"label": "SABCHE", "head": "གཉིས་པ་༼ཚུལ་ཁྲིམས་ཕར་ཕྱིན་གྱི་རབ་དབྱེ་༽ནི", "tail": ""}]}

41 characters, so it goes whole in head. The span stops at ནི - the ། after it is
outside. The verse lines below are root text and get no span.

2 - bracket-numbered headings, one per line, several in a row

Text: ...(2) སྨོན་ལམ་གྱི་མཐའ་བསྟན་པ།
[2] རྗེས་ཕན་ཡོན་བསྟན་པས་མཇུག་བསྡུ་བ།
(1) སྨོན་ལམ་གྱི་ཕན་ཡོན།
{2} ཐ་མར་དགེ་བ་མཇུག་གི་དོན།
[1] སྨོན་ལམ་བཏོན་ཅིང་དགེ་རྩ་བསྔོ་བ།...

json
{"spans": [{"label": "SABCHE", "head": "(2) སྨོན་ལམ་གྱི་མཐའ་བསྟན་པ", "tail": ""}, {"label": "SABCHE", "head": "[2] རྗེས་ཕན་ཡོན་བསྟན་པས་མཇུག་བསྡུ་བ", "tail": ""}, {"label": "SABCHE", "head": "(1) སྨོན་ལམ་གྱི་ཕན་ཡོན", "tail": ""}, {"label": "SABCHE", "head": "{2} ཐ་མར་དགེ་བ་མཇུག་གི་དོན", "tail": ""}, {"label": "SABCHE", "head": "[1] སྨོན་ལམ་བཏོན་ཅིང་དགེ་རྩ་བསྔོ་བ", "tail": ""}]}

One span per line, marker included, closing ། excluded. Never merge a run.

3 - a long heading that lists its sub-items

Text: ...གཉིས་པ་ནི་སྒོམ་ལམ་དང་མི་སློབ་པའི་ལམ་གཉིས་སོ། །གསུམ་པ་ལ་མཚན་ཉིད། མཚན་གཞི། དབྱེ་བ། ཁྱད་ཆོས་དང་བཞི་ལས། དང་པོ་༼གསུམ་པ་ལ་མཚན་ཉིད་༽ནི།
ཐེག་པ་ཆེན་པོའི་ཆོས་བཟོད་དང་ཆོས་ཤེས།...

json
{"spans": [{"label": "SABCHE", "head": "གཉིས་པ་ནི་སྒོམ་ལམ་དང་མི་སློབ་", "tail": "༼གསུམ་པ་ལ་མཚན་ཉིད་༽ནི"}]}

127 characters, so head and tail are both given. It runs from the number word to
the ནི that opens the section being explained; the prose after it is the body.

4 - a heading followed by a quotation - only the heading is marked

Text: ...གསུམ་པ་༼མ་གྲོལ་གྱི་ལམ་ལ་འདུན་པས་ཀུན་ནས་ཡོངས་སུ་བསླང་བའི་ཐུགས་བསྐྱེད་པ༽ནི།
མ་སྐྱེས་དགྲའི་འགྱོད་པ་བསལ་བ་ལས། བསྐལ་པ་དཔག་ཏུ་མེད་ཅིང་གྲངས་མེད་པའི་གོང་རོལ་དུ་...

json
{"spans": [{"label": "SABCHE", "head": "གསུམ་པ་༼མ་གྲོལ་གྱི་ལམ་ལ་འདུན་", "tail": "བའི་ཐུགས་བསྐྱེད་པ༽ནི"}]}

The passage after མ་སྐྱེས་དགྲའི་འགྱོད་པ་བསལ་བ་ལས། is a quotation from a named work and
gets no span.

Output

Return the spans in the order they appear in the chunk. No offsets, no indices,
no translations, no explanations, no markdown fences.

Reply with JSON only: {"spans":[{"label":"SABCHE","frame":"...","head":"...","tail":"..."}]}
frame is optional; head and tail are copied character for character. If the span
is 40 characters or shorter, put it all in head and leave tail empty. If none:
{"spans": []}

Text:

---
