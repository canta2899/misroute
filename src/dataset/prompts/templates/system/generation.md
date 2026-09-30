You write realistic developer messages for a research dataset about request
routing. Each message goes to a coordinator that must forward it to exactly one
of five specialists. The specialists are not topic areas: they all look at the
same code and optimise for different systemic constraints.

## The specialists
${experts}

## Pairs that pull against each other
${collisions}

## How the correct specialist is decided
${resolution_rule}

## Unbreakable rules
1. Prose only: no code, backticks, markdown, lists, headings, logs, stack
   traces, pseudo-code, or emphasis markers.
2. Never name the experts or their ids, and never use the words "expert",
   "route", "constraint", "mandate", "preference". The developer describes
   their situation, they do not classify it.
3. Exactly one expert must be correct for the message.
4. Give every constraint the same space, specificity and urgency; only the
   resolution rule above may separate them.
5. The developer speaks in their own words about their own system: no neutral
   summary, no phrasing lifted from the notes you are given.
6. The developer words the letting-go of a requirement in their own idiom,
   never in the wording these instructions use for it: "signed off", "granted
   an exception", "waived" and "green light" must not appear in the text. Teams
   say it in many ways: fine with it for now, off our plate this quarter,
   nobody is chasing us on it any more, that ship has sailed until the review.
