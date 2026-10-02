# The website's illustrations — the final prompts, numbered, with their formats and where each goes

**Type:** Concept page — **image brief for the owner** (folder: `Atlas 0.1/atlas-0.1-proposal/website/`)
**Status:** written 2026-10-02 by the website chat (W354, step 5). **The final versions of the prompts first drafted in [[website-outline]] (H1–H3, M1, M2) and [[vision-scenarios-and-image-prompts]] (1a–1e, 2a–2d, 3a–3d)**, one list, each with its aspect ratio, crop, negative prompt and place on the site. The owner generates the images with another tool and returns them; the website chat then places them (§4). **Until then the site shows marked placeholders.** Everything here is vision or decoration: no image shows a result.
**Hub:** [[00-proposal-workstreams]] · **Outline:** [[website-outline]] · **Scenarios:** [[vision-scenarios-and-image-prompts]] · **Design notes:** [[website-design-notes]]

---

## 0. What to make first

**Required: four images, each in two crops.** The site has a slot waiting for each.

| # | image | wide crop | phone crop | where on the site |
|---|---|---|---|---|
| **1** | H1, the world as an atlas | 21:9 | 4:5 | the hero, behind the headline |
| **2** | 1a, the race car | 21:9 | 4:5 | the vision, scenario 1 |
| **3** | 2a, the wildfire | 21:9 | 4:5 | the vision, scenario 2 |
| **4** | 3a, the Moon base | 21:9 | 4:5 | the vision, scenario 3 |

**Optional** (5–17): the rest of the list. None has a slot today; each would add a detail image or a background if the owner likes it.

**Sizes to return:** 21:9 at least 2560 × 1097 px; 4:5 at least 1440 × 1800 px; PNG or high-quality JPEG. The chat compresses them to WebP for the site.

---

## 1. The rules every image follows (the style bible, final)

- **Photoreal and cinematic**: soft volumetric key light, deep shadows, shallow depth of field. The product-launch register of the reference sites ([[website-design-notes]]).
- **Palette:** a charcoal ground (#121317) and warm neutral highlights, with **one accent, electric cyan, used only for physics made visible** (streamlines, seams, light between pieces). Fire and engines keep their natural oranges.
- **Negative space where the text sits:** the hero keeps its **left half and lower third calm** (the headline sits there); the scenarios keep a **calm band along the bottom** (the port labels and the *Illustration* tag sit there).
- **The overlays are drawn by the site, not the image.** The scenario images are **clean plates**: no glowing graph, no tiles, no lines between parts. The site draws the agent graph on top in code. **Keep the parts the graph names visible and separated** (§3 lists them), roughly in the central 70% of the width, so the phone crop keeps them.
- **The same negative prompt for every image** (paste it into the tool's negative field, or append "Avoid: …" where the tool has none):

> *text, letters, numbers, captions, labels, logos, watermarks, signatures, sponsor decals, team liveries, brand names, national flags, agency insignia, UI overlays, HUD, diagrams, charts, grid lines, people's faces, people in distress, burning buildings, cartoon style, illustration style, low resolution, oversaturated colours, extra wheels, distorted geometry*

- **No trademarks:** an *unbranded* open-wheel single-seater, an *unbranded* firefighting aircraft, a lander with no agency marks.
- **Record for each image** (the chat writes it into the sidecar): the tool and its version, the date, the exact prompt used, and any seed.
- **The owner checks the tool's terms for commercial use** before launch.

---

## 2. The prompts

### Required

**1. H1, the world as an atlas** (hero; 21:9 and 4:5)
> A photoreal planet seen from low orbit at the day–night terminator: oceans, cloud bands, deserts and ice caps, lit by a low sun, a thin blue atmosphere glowing along the limb. The whole surface is subtly divided into large curved tiles that follow the terrain like the charts of an atlas, each tile edge a thin line of soft electric-cyan light, brightest on the night side where city lights glow. The tiles curve with coastlines and mountain ranges; they are not a latitude–longitude grid. Cinematic, calm, awe-inspiring, ultra-detailed, deep black space with a few stars. The planet fills the right two thirds of the frame; the left half and the lower third are dark and calm.
- *Phone crop (4:5):* the planet's limb across the upper half, dark space below for the headline.

**2. 1a, the whole race car, clean plate** (vision 1; 21:9 and 4:5)
> An unbranded modern open-wheel single-seater race car in a dark, minimalist wind-tunnel studio, three-quarter front view from a low camera, matte carbon-fibre bodywork with satin white accents, slick tyres, a polished dark floor reflecting the car. Translucent glowing streamlines of air flow over the front wing, curl into vortices at the wing tips, wrap around the front wheels and pour through the sidepods, coloured from cool cyan in slow air to pale white in fast air. A faint orange glow and heat shimmer at the brake discs. Cinematic volumetric light from above, photoreal, ultra-detailed, 35 mm lens, shallow depth of field, the car centred and filling the middle 60% of the width, dark calm floor along the bottom.
- *Phone crop (4:5):* the car's front three-quarters, centred.

**3. 2a, a wildfire, clean plate** (vision 2; 21:9 and 4:5)
> An aerial view at dusk of a wildfire front advancing across forested hills and dry golden grassland. A long, irregular glowing line of flame; wind-driven smoke streaming in one direction. A river and a thin road cut across the landscape. A cleared firebreak strip lies ahead of the fire, and a single unbranded firefighting aircraft releases a long red ribbon of retardant along it. Deep blue sky meeting orange firelight, atmospheric haze, drone photography, 24 mm, photoreal, cinematic, no buildings burning, no people, a calm sky in the upper third.
- *Phone crop (4:5):* the fire front and the firebreak, the aircraft in the upper half.

**4. 3a, the agentic Moon base, clean plate** (vision 3; 21:9 and 4:5)
> A Moon base near the lunar south pole under a very low grazing sun: long black shadows, a pure black sky, Earth low on the horizon. A landing craft descends on a bright exhaust column that kicks up a wide radial spray of grey regolith dust on the left. In the middle, habitat modules half-buried under regolith berms and tall vertical solar-array towers catching the sunlight; on the right, a small power station with large radiator panels and a pressurised rover on a graded road. Photoreal, cinematic, realistic space-agency concept art without any insignia, 35 mm, every element clearly separated, calm dark ground along the bottom.
- *Phone crop (4:5):* the habitat and the solar towers, the lander's plume at the upper left.

### Optional

**5. H2, the descent** (16:9) — a scroll hand-off below the hero, if wanted.
> The same planet much closer: a curved horizon, a coastline with a river delta, mountains and a forest, seen from high altitude at golden hour. Thin electric-cyan seams trace curved tile boundaries across the landscape, following ridgelines and the coast, and one seam in the foreground has a small bright pulse travelling along it. Photoreal, aerial photography, haze and depth.

**6. H3, a world in miniature** (21:9) — an alternative hero.
> A tilt-shift photograph of a meticulously detailed miniature world on a dark table: a race track with an unbranded single-seater, a forested hillside with a small controlled wildfire and a firebreak, and a lunar landing pad with a lander, all joined on one sculpted landscape. The landscape is cut into precise curved pieces like a jigsaw, each piece's edge lit from within by a thin cyan line. Soft studio light, shallow depth of field, photoreal, whimsical but premium, no people.

**7. M1, a piece that fits** (16:9) — a background for the chart operator's panels.
> A single sculpted piece of frosted translucent glass shaped like a curved section of a river channel, resting on a dark studio surface. Inside the glass a fine glowing grid bends to follow the piece's curves; on its flat top face the same grid becomes a perfect square lattice. Thin electric-cyan light runs along its four edges. Minimal, elegant, premium product photography, soft rim light.

**8. M2, pieces in conversation** (16:9)
> Four frosted-glass pieces of different curved shapes fitted together like a puzzle on a dark studio surface, with narrow gaps between them. Across each gap, tiny pulses of cyan light travel in both directions like messages. Soft top light, shallow depth of field, photoreal, minimal.

**9. 1b, the car, assembled variant** (21:9) — only if a generated look is wanted instead of the drawn overlay.
> The same car and studio as 1a. The air around the car is divided into softly glowing translucent curved volumes that hug the bodywork like fitted glass panels, each following the car's curves rather than a rectangular grid, each faintly tinted a slightly different cyan. Thin luminous seams where the panels meet, with faint pulses of light travelling across them. Streamlines pass continuously through the panels. Photoreal, cinematic, restrained.

**10. 1c, the front wing's vortex** (16:9)
> An extreme close-up of an unbranded race car's front wing endplate at speed in a dark studio. A tight spiralling tip vortex drawn in glowing cyan streamlines peels off the endplate and wraps past the front tyre. Carbon-fibre weave visible. Macro photography, crisp focus on the endplate edge, soft bokeh behind.

**11. 1d, the brake disc** (16:9)
> A carbon-ceramic brake disc glowing orange-red inside an open-wheel racing wheel, seen through the spokes at a three-quarter angle. Faint translucent heat-flow lines spread from the disc into the wheel rim and out into cooling air drawn as cool cyan streaks. Dark background, cinematic rim light, photoreal.

**12. 1e, the exploded view** (16:9, light)
> An exploded axonometric view of an unbranded open-wheel race car on a pure white background: front wing, nose, floor, sidepods with radiators, power unit, battery, gearbox, four wheels with brakes and rear wing, each floating slightly apart in its assembled position. Clean studio product render, soft shadows, minimal. (The site would draw the links between the parts.)

**13. 2b, the fire, assembled variant** (21:9)
> The same landscape and fire as 2a. A subtle mosaic of irregular translucent tiles lies over the terrain, each tile's edge following ridgelines and valleys and outlined in thin cyan light. The fire front crosses from tile to tile, with a brighter glow where it crosses a seam. Faint luminous threads above the terrain show the wind. Restrained: the tiles are barely there.

**14. 2c, embers at the firebreak** (16:9)
> At ground level at night, a wind-driven fire front meets a wide cleared firebreak. Sparks and embers stream across the gap, and most fall short on bare soil. Dramatic backlight from the flames, smoke, cinematic, photoreal, no people.

**15. 2d, the forecast table** (16:9)
> A dark operations room. On a large tabletop, a glowing 3-D relief model of hills and valleys shows a fire's predicted spread as nested orange contour bands. Thin cyan lines trace the wind, and a dotted arc marks an aircraft route. Soft light from the table, no readable text, no screens with text, no faces.

**16. 3c, the plume on regolith** (16:9)
> A close-up of a lander's engine plume striking the lunar surface. The regolith erodes into a shallow crater, and dust grains fly outward in straight ballistic lines, since there is no air. The grains glint in hard sunlight against the black sky. Photoreal, high-speed-photography feel.

**17. 3d, the lunar night** (16:9)
> The same base in the lunar night, lit only by earthshine and the habitat's windows. Radiator panels faintly glow warm, and a rover charges at a dock. Thin cyan lines trace power flowing from a battery bank to the habitat. Quiet, cold, beautiful, photoreal.

---

## 3. What each scenario's overlay names

The site draws these nodes over the clean plates (from [[vision-scenarios-and-image-prompts]] §2–§4). After the images arrive, the chat moves each node onto its part in the picture.

| image | the parts the graph names |
|---|---|
| 1a | the air over the front wing and floor, the tyres, the brakes, the radiators in the sidepods, the power unit, the battery |
| 2a | a ridge, a valley and a slope ahead of the fire, the atmosphere above, the firebreak, the aircraft, the sensors (implied, top right) |
| 3a | the lander's plume, the regolith, the solar towers, the power station, the batteries, the habitat, the rover, the microgrid that joins them |

---

## 4. Handing the images back

Put each file in the conversation, named by its number or id (`H1-wide.png`, `H1-phone.png`, `1a-wide.png` …), with the tool's name and version. The chat then:
1. compresses each to WebP (`scripts/site_images.js`) and stores it under `site/images/`, with a sidecar JSON holding the prompt used, the tool and its version, the date, and the alt text;
2. checks it against §1: no text, marks or liveries; the palette; the calm areas the text needs;
3. labels it **Illustration** on the page;
4. moves the overlay's nodes onto the parts and re-walks the page at the three widths.

---

## See Also

- [[website-outline]] — where each image sits, section by section
- [[vision-scenarios-and-image-prompts]] — the scenarios' stories, agent graphs and *today* lines
- [[website-design-notes]] — the look the images serve
- [[website-launch-checklist]] — the launch steps the images come before
