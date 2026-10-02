# The three reference sites, read — their type, spacing, motion and section rhythm

**Type:** Concept page — **design notes for the proposal website** (folder: `Atlas 0.1/atlas-0.1-proposal/website/`)
**Status:** written 2026-10-02 by the website chat (W354, step 1). The planning session could not reach the three sites ([[proposal-website-plan]] §2 was written from search summaries). **This page replaces those summaries with the sites' own values**, read from their live HTML, stylesheets and script bundles. **What it could not do:** render them. See §0.
**Hub:** [[00-proposal-workstreams]] · **Plan:** [[proposal-website-plan]] · **Outline:** [[website-outline]] · **Evidence:** [[website-evidence-and-citations]]

---

## 0. How the sites were read, and the limit

- **Fetched on 2026-10-02** from the website chat's cloud container, with TLS verified: each site's home page, every stylesheet it links, its inline styles, and, for Antigravity, its own script bundles. Every size, colour and duration below is **a value in the site's CSS or code**, not a reading of a screenshot.
- **Not rendered.** The container's headless browser refuses the sandbox proxy's certificate, and loosening that check was not allowed. So no screenshot was taken and nothing was watched moving: motion is described from the code that drives it. **The owner can open the three on their own machine in a minute**; nothing below depends on it, but the feel of the motion is only visible there.
- **Sizes are for a laptop window 1440 px wide** unless a line says otherwise. A value written $a\to b$ is a fluid size that grows from $a$ on a phone to $b$ on a wide screen.

---

## 1. Antigravity (`antigravity.google`), Google's agent platform

**Built with** Astro, and **GSAP** (ScrollTrigger in five of its section scripts, ScrollSmoother for the whole page, SplitText, Draggable), plus **three.js** for a WebGL particle field and a custom cursor.

| aspect | the value |
|---|---|
| type family | **Google Sans Flex**, variable (optical size 8–144, weight 400–500 loaded, width, slant, roundness); Google Sans Code for code |
| hero headline | **72 px at 1440**, 107 px above 1600, 56 px at 1024 and below, 40 px on a phone; line height equal to the size; tracking about $-2\%$ ($-2.14$ px at 107) |
| type scale | 148, 124, 98, 72, 54, 42, 32, 28, 22 px; body **17.5 px** (15 px at 1024 and below), tracking $+0.18$ px |
| weight | 400–450 for display, never bold |
| colour | dark surface **#121317**, text #212226, cool greys #e6eaf0, #cdd4dc, #b7bfd9, #45474d, white; **the primary button is a near-black pill** in the text colour |
| space | a 4 px base: 4, 8, 16, 24, 36, 48, 60, 80, 88, 120, 180 px. Page margin **72 px** (40 at 1024, 16 on a phone); gutter 64 px; content up to 1600 px |
| corners | 4, 8, 16, 24, 36 px, and 9999 px pills |

**Section rhythm**, nine screens, each one idea: the hero (a headline typed in over the particle field, two actions) → a video → "agent-first" → a feature explorer of four products → a use-case slider → *for developers* and *for organizations* → the blog → download → a footer with a large *"Experience liftoff"*.

**Motion:** smooth scrolling over the whole page; reveals triggered by scroll in five sections; the headline typed letter by letter with a blinking caret (0.5 s); a WebGL particle field driven by the mouse (its module is named `Mouse`) and a second, morphing one; hover transitions of 0.15–0.4 s, easing `cubic-bezier(.165,.84,.44,1)`, a fast-out curve. **Its stylesheets contain no `prefers-reduced-motion` rule**; whether its scripts check it was not read.

---

## 2. Googlebook (`googlebook.google`), Google's laptop launch

**Built with** Google's marketing component library, Swiper carousels, Rive animations, and **26 muted looping videos that start on their own**, each with a pause control.

| aspect | the value |
|---|---|
| type family | Google Sans Display for headlines, Google Sans Text for body |
| display headline | **5 vw at 1024 and wider (72 px at 1440)**, 6 vw from 600, 10 vw on a phone; line height 1.2; tracking $-0.5$ px; weight 400 |
| section headlines | 60 px (headline 1) and 48 px (headline 2) from 1024; 36 and 28 px on a phone |
| body | 16–18 px, line height 1.5 |
| colour | white surface, text #202124, secondary #5f6368, panels #f8f9fa, rules #dadce0, links #1a73e8 |
| space | 8, 12, 16, 24, 36, 48, 60, 80 px; a 12-column grid (4 on a phone); breakpoints 600, 1024, 1440; content up to 1296–1440 px |

**Section rhythm:** the hero (a video, the product's name, its price, a pill, three cards) → services → **hardware: *"Crafted for performance."* and a band of four big figures** (16 GB, 256 GB, 14 hours, 2.85 lb), **each with a footnote marker** → intelligence → *better together* → apps and games (tabs) → security → perks → shop → newsletter → footer. **Every section headline is a short sentence with a full stop**: *"Millions of apps."* *"Secure by design."*

**Motion:** **CSS scroll-driven animation**, with no library: a 150-vh pinned panel whose video fades as it leaves the screen (`view-timeline` with `animation-range`); entry by fading and un-blurring over 1 s; slide-ups; tab progress bars; Material's easing curves (`cubic-bezier(.4,0,.2,1)` and `(0,0,.2,1)`); nine sticky rules. **Twenty-one `prefers-reduced-motion` rules.**

---

## 3. SimScale (`simscale.com`), cloud engineering simulation

**Built with** WordPress and GenerateBlocks; HubSpot forms. Its title is *"AI-Native Engineering Simulation Software in the Cloud"*.

| aspect | the value |
|---|---|
| type family | **Geist**, variable (100–900) |
| headlines | h1 $36\to56$ px, h2 $32\to48$ px, weight 400, tracking $-0.02$ em, all fluid through `clamp()` |
| body | $14\to16$ px, line height 1.5 |
| colour | navy **#070a27** for text and the dark bands, blue #0e63fe for actions, orange #fb6a27 and **cyan #27d8fb** as accents, greys #f7f7f7, #ebebed, #dfdfe2 |
| space | **section padding $64\to112$ px** above and below (about 97 px at 1440); side padding $20\to160$ px; content up to 1600 px |
| buttons | pills everywhere: a 999 px radius in 90 rules |

**Section rhythm:** the hero (headline, two pills, review stars, the product's interface drawn as an SVG) → a marquee of 17 customer logos → the problem → three pillars → *Why SimScale?* in tabs → use cases with customer quotes → six industries → a comparison table, *"Legacy simulation can't keep up. Yours can."* → resources in tabs → **a dark band**, *"Trusted by 900,000+ engineers worldwide"*, with customers' own results → **a dark closing band**, *"Start simulating in minutes, not weeks"* → footer.

**Motion:** modest. The logo marquee loops every 64 s; dots pulse in a staggered 9 s cycle; tabs and carousels; transitions of 0.2–0.3 s; five muted looping videos; three `prefers-reduced-motion` rules. **Its proof is social**: logos, quotes, and customers' figures (*"45% saved on computation costs"*), credited to the customer, not linked to a method.

---

## 4. Side by side

| | Antigravity | Googlebook | SimScale | **proposed for Atlas** |
|---|---|---|---|---|
| family | Google Sans Flex | Google Sans Display / Text | Geist | **one open family, hosted with the site** (§6) |
| display size at 1440 | 72 px | 72 px | 56 px | **72–88 px**, fluid |
| display weight, tracking | 400–450, $-2\%$ | 400, $-0.5$ px | 400, $-2\%$ | **400–500, about $-2\%$** |
| body | 17.5 px | 16–18 px | 16 px | **18–20 px**, 60–75 characters a line ([[website-outline]] §1) |
| dark sections | the inverse surface #121317 | not determined without rendering | two closing bands | **one dark chapter, the vision**, #121317 |
| primary action | black pill | pill | pill | **black pill** |
| vertical rhythm | steps of 120 and 180 px in its scale | largest common steps 60–80 px | section padding 64–112 px | **about 120 px**, one idea a screen |
| motion engine | GSAP and WebGL | CSS scroll timelines | CSS transitions | **CSS scroll timelines and a little JS** |
| reduced motion | not in the CSS | 21 rules | 3 rules | **every animation**, final frame shown |
| proof | product demos | footnoted figures | logos and quotes | **footnoted figures, each to its record or paper** |

---

## 5. What Atlas takes from them, and what it leaves

**Takes:**
1. **One large headline per screen, light weight, tight tracking.** All three set display type at weight 400 with negative tracking; none uses bold.
2. **Short headlines that end in a full stop** (Googlebook), which fits the outline's voice: *"Cut. Solve. Stitch."*
3. **A band of four big figures, each with a footnote marker** (Googlebook's hardware band). **On Atlas's band each marker leads to the figure's record or paper**, which is the claims pipeline of [[proposal-website-plan]] §3.1 made visible.
4. **The black pill for the one primary action, quiet text links for the rest** (Antigravity, SimScale).
5. **A 4 px spacing scale and about 120 px between ideas** at laptop width (Antigravity's scale has steps of 120 and 180 px; SimScale pads each section by about 97 px at 1440).
6. **Scroll-driven reveals in CSS** (Googlebook's `view-timeline`), with a small script where a browser lacks it. No animation library is needed for what the outline asks.
7. **Dark bands for weight** (SimScale closes on two): Atlas has one, the vision chapter.
8. **A pause control on anything that moves on its own** (Googlebook's videos).

**Leaves:**
1. **Smooth-scroll hijacking** (Antigravity's ScrollSmoother). **[AI Inference]:** it replaces the browser's own scrolling, which is what a reader with reduced motion or a trackpad notices first, and it needs a library.
2. **The custom cursor and the decorative WebGL particles.** If the hero has particles, they are tracers carried by a recorded flow field, drawn in a 2-D canvas, as [[proposal-website-plan]] §2 proposed: the decoration then is evidence.
3. **Autoplaying product video.** O2 made the demo a local install; the site shows real screenshots instead.
4. **Logo walls, quotes and customers' figures.** Atlas has no customers, and the site does not imply any.

**[AI Inference]:** the three agree on more than they differ: light display type, generous space, pills, one idea per screen. What separates Googlebook, the closest to the owner's *"Apple and Google"* register, is that **its claims carry footnotes**. That is the one habit of the genre that a sceptical technical reader also rewards.

---

## 6. The typeface

[[proposal-website-plan]] §2 left one question open: whether Google Sans Flex's licence allows hosting it. **It does**: Google Sans Flex is in Google's font repository under `ofl/googlesansflex`, with its `OFL.txt` (SIL Open Font License 1.1) and a `TRADEMARKS.md`. The candidates, all three under the OFL:

| family | used by | for Atlas |
|---|---|---|
| **Inter** (variable, with an optical-size axis) | — | a display cut at large sizes, figures of equal width for the count-up numbers, and no association with any company |
| **Geist** (variable) | SimScale | close to the genre; Geist Mono would set the install commands |
| **Google Sans Flex** (variable) | Antigravity | the closest to the references |

**[AI Inference]:** set in Google Sans Flex with a #121317 surface and black pills, the site would read as a Google page, which it is not. Inter or Geist keeps the register and stays its own. Hosting any of them means downloading its files, which waits for the owner's yes.

---

## See Also

- [[proposal-website-plan]] — §2's design language, which this page grounds in the sites' own values
- [[website-outline]] — the section-by-section outline these notes serve
- [[website-evidence-and-citations]] — the figures the footnote markers lead to
- [[vision-scenarios-and-image-prompts]] — the dark chapter's images
