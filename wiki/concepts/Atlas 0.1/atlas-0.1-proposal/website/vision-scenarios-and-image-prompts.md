# The vision — three scenarios, their agent graphs, and the prompts for their images

**Type:** Concept page — **scenario design and image-generation brief** (folder: `Atlas 0.1/atlas-0.1-proposal/website/`)
**Status:** written 2026-09-30. **Everything on this page is vision, not result.** The *today* line under each scenario says what exists in the vault toward it, from its measured pages. The prompts are a first draft for the owner to run in an image generator. The website chat refines them once the first images come back.
**Revised 2026-09-30:** the owner chose the headline *"A full world. Decomposed and simulated by experts."* (O7). The hero is now a whole world cut into charts, and its prompts (H1–H3) are in [[website-outline]] section 1, with the prompts for the architecture section (M1, M2). This page keeps the three scenarios.
**Hub:** [[00-proposal-workstreams]] · **Used by:** [[proposal-website-plan]] §1, section 7 · [[website-outline]] section 7 · **The end goal they illustrate:** [[f1-pathmap-and-end-goal]] · [[physics-foundation-models]]

---

## 0. What the three scenarios have to say

The owner asked for three: a fully simulated 3-D race car, a wildfire's spread and its control, and an agentic Moon base, where *"each interaction that takes place is using the agentic 'communication' from learnt experts."*

**One idea, three scales:**
- **the car:** one machine, many physics, split into experts that trade forces and heat at their seams;
- **the fire:** a landscape, where the pieces are terrain charts and the seams carry heat, embers and wind, fast enough to forecast and steer;
- **the base:** a whole settlement, where every subsystem is an expert and every cable, pipe and plume is a typed port.

**"Agentic communication" made precise.** In Atlas an agent is a local expert, and what agents say to each other is fixed: a **typed port message**, an effort and a flow whose product is power, carried as incoming and outgoing waves ([[chart-operator-architecture]] §4). Two consequences are worth saying on the site:
- **the messages are physical quantities**, so every exchange can be audited: power in equals power out;
- **the compiler checks every connection before anything runs.**

That is what makes these experts cooperate, rather than merely run side by side.

---

## 1. The style bible

**One visual world for all three**, so they read as one product:
- **Rendering:** photoreal and cinematic, the product-launch register. Soft volumetric key light, deep shadows, a shallow depth of field.
- **Palette:** a charcoal ground ($\#121317$) and warm neutral highlights, with **one accent, electric cyan**, reserved for the physics made visible: streamlines, seams, waves between agents. Fire and engines keep their natural oranges.
- **Composition:** generous negative space where the headline will sit (upper third for 21:9, left third for 16:9).
- **Formats:** each hero at **21:9** (desktop) with a **4:5** crop (phone); each detail at **16:9**.
- **Negative prompt, every image:** *text, letters, numbers, logos, watermarks, sponsor decals, team liveries, national flags, agency insignia, UI overlays, people in distress.*
- **No trademarks.** Describe *an unbranded open-wheel single-seater*, not a team's car. **[AI Inference]:** "F1" in the site's copy is descriptive use, but the images must carry no team or series marks. The owner checks the image generator's terms for commercial use.
- **Overlays are drawn in code, not generated.** Generators render tiles, graphs and seams unreliably. So each scene gets a **clean plate**, and the site draws the agent graph and the seams over it in SVG, where they are exact and can animate (the seam motion of [[proposal-website-plan]] §2). An **"assembled" variant** prompt is given too, for a hero where a generated look is wanted.

---

## 2. Scenario 1 — the whole car, assembled

**The story.** *A race car is air, heat, rubber, metal and electricity at once. Today each is simulated by its own tool, stitched together by hand. Atlas gives each part its own expert, lets them talk through the forces and heat they share, and solves the whole car together, fast enough to try a design in minutes.*

**The agent graph:**

| agent | physics | talks to, through |
|---|---|---|
| air charts hugging the body: front wing, floor and diffuser, sidepods, rear wing | incompressible, then compressible flow (learned chart operators) | the bodywork: `MECH` (pressure and shear); the radiators: `ADVEC` (air mass and enthalpy) |
| the four tyres and wheels | contact mechanics, tyre heating | the road and suspension: `MECH`; the brakes and the air: `THERM`; the drivetrain: `ROT` |
| the four brakes | transient conduction, heat by friction | the wheels and cooling air: `THERM`; the axle: `ROT` |
| the radiators | a heat exchanger | the air: `ADVEC`; the coolant loop: `ADVEC`, `THERM` |
| the power unit and energy recovery | combustion, machine and circuit | the drivetrain: `ROT`; the battery: `ELEC`; the coolant: `THERM` |
| the battery | electrochemistry and heat | the power unit: `ELEC`; the cooling: `THERM` |

**Today** (measured): a 2-D car in 14 windows on a rolling road, and a 3-D half-car in a box with a symmetry plane, both classical ([[poc3-racelab-3d]]); an 18-agent vehicle union across five families with a rotor, a radiator and a powertrain circuit ([[case-study-vehicle-march-atlas-0.1]]). **What does not work yet:** the learned column at the car's cadence ran at $0.067\times$ classical speed and left its envelope ([[outcome-c5-requirements-for-dd-native-experts]] S2).

**Prompts:**

> **1a, hero, clean plate (21:9).** An unbranded modern open-wheel single-seater race car in a dark, minimalist wind-tunnel studio, three-quarter front view from a low camera, matte carbon-fibre bodywork with satin white accents, slick tyres, a polished dark floor reflecting the car. Translucent glowing streamlines of air flow over the front wing, curl into vortices at the wing tips, wrap around the front wheels and pour through the sidepods, coloured from cool cyan in slow air to pale white in fast air. A faint orange glow and heat shimmer at the brake discs. Cinematic volumetric light from above, photoreal, ultra-detailed, 35mm lens, shallow depth of field, large empty dark space above the car.

> **1b, hero, assembled variant (21:9).** The same car and studio as 1a. The air around the car is divided into softly glowing translucent curved volumes that hug the bodywork like fitted glass panels, each panel following the car's curves rather than a rectangular grid, each faintly tinted a slightly different cyan. Thin luminous seams where the panels meet, with faint pulses of light travelling across the seams, as if the pieces were talking to each other. Streamlines pass continuously through the panels. Photoreal, cinematic, elegant, restrained.

> **1c, detail (16:9).** An extreme close-up of an unbranded race car's front wing endplate at speed, in a dark studio. A tight spiralling tip vortex drawn in glowing cyan streamlines peels off the endplate and wraps past the front tyre. Carbon-fibre weave visible. Macro photography feel, crisp focus on the endplate edge, soft bokeh behind.

> **1d, detail (16:9).** A carbon-ceramic brake disc glowing orange-red inside an open-wheel racing wheel, seen through the spokes at a three-quarter angle. Faint translucent heat-flow lines spread from the disc into the wheel rim and out into cooling air drawn as cool cyan streaks. Dark background, cinematic rim light, photoreal.

> **1e, optional, the exploded view (16:9, light).** An exploded axonometric view of an unbranded open-wheel race car on a pure white background: front wing, nose, floor, sidepods with radiators, power unit, battery, gearbox, four wheels with brakes and rear wing, each floating slightly apart in its assembled position. Thin cyan lines connect the neighbouring parts like a network diagram. Clean studio product render, soft shadows, minimal.

---

## 3. Scenario 2 — a wildfire, forecast and steered

**The story.** *A wildfire is fire, fuel, terrain and wind feeding each other. Atlas splits the landscape into pieces that follow the ridges and valleys, gives each piece an expert for how fire spreads through its fuel, and lets the pieces pass heat, embers and wind across their edges. It is fast enough to forecast hours ahead, and to test where a firebreak or an air drop would do the most good before anyone is sent.*

**The agent graph:**

| agent | physics | talks to, through |
|---|---|---|
| terrain charts following ridgelines and valleys | fire spread through fuel: reaction, advection and diffusion (learned chart operators) | neighbouring charts: `THERM` (radiant and convective heat), `ADVEC` (embers); the air above: `MECH`, `ADVEC` |
| the atmosphere over the terrain | wind over hills, the fire's own plume | every chart below it: `MECH`, `ADVEC` (smoke and heat) |
| rivers, roads, firebreaks | boundary conditions: no fuel | the charts they cross |
| suppression: aircraft drops, crews | control agents changing fuel and moisture | the charts they act on: a source through a declared port |
| sensors and satellites | observation agents | the forecast, which corrects the state |

**Today:** nothing fire-specific is built. The nearest pieces are the workbench's pollutant plume carried by a solved flow ([[showcase-gallery]]), flow over drawn terrain (`farm-hill`), and the vault's page on learning to control flows ([[rl-for-flow-control-and-coordination]]). **This scenario is the furthest from today's record, and the site says so.**

**Prompts:**

> **2a, hero, clean plate (21:9).** An aerial view at dusk of a wildfire front advancing across forested hills and dry golden grassland. A long, irregular glowing line of flame, wind-driven smoke streaming in one direction. A river and a thin road cut across the landscape. A cleared firebreak strip lies ahead of the fire, and a single firefighting aircraft releases a long red ribbon of retardant along it. Deep blue sky meets orange firelight, atmospheric haze, drone photography, 24mm, photoreal, cinematic, no buildings burning, no people, large calm sky in the upper third.

> **2b, hero, assembled variant (21:9).** The same landscape and fire as 2a. A subtle mosaic of irregular translucent tiles lies over the terrain, each tile's edge following ridgelines and valleys and outlined in thin cyan light. The fire front crosses from tile to tile, with a brighter glow where it crosses a seam. Faint luminous threads above the terrain show the wind's direction. Restrained and elegant: the tiles are barely there, like a heads-up display made of light.

> **2c, detail (16:9).** At ground level at night, a wind-driven fire front meets a wide cleared firebreak. Sparks and embers stream across the gap, and most fall short on bare soil. Dramatic backlight from the flames, smoke, cinematic, photoreal, no people.

> **2d, detail (16:9).** A dark operations room. On a large tabletop, a glowing 3-D relief model of hills and valleys shows the fire's predicted spread as nested orange contour bands reaching hours ahead. Thin cyan lines trace the wind, and a dotted arc marks an aircraft route. Soft light from the table on the room, no readable text, no screens with text, no people's faces.

---

## 4. Scenario 3 — the agentic Moon base

**The story.** *A Moon base is a machine the size of a town where every system leans on the others. A landing blasts dust at the solar arrays; the arrays feed the batteries; the batteries keep the habitat warm through the two-week night; the radiators shed the heat. Atlas gives each system its own expert, and every link between them, every cable, pipe and plume, becomes a typed conversation that carries power, heat or force. The whole base can be run forward in time, and every one of those conversations is checked.*

**The agent graph:**

| agent | physics | talks to, through |
|---|---|---|
| the lander's descent plume | compressible jet in vacuum | the regolith: `MECH` (pressure, shear), `ADVEC` (ejected dust) |
| the regolith surface | granular erosion and ejecta | the plume; the arrays and habitat it sprays: `ADVEC` |
| the solar towers | photovoltaics and heat, sun tracking | the microgrid: `ELEC`; the tracking drive: `ROT`; their radiators: `THERM` |
| the fission power unit and radiators | a reactor and its heat rejection | the microgrid: `ELEC`; the radiators: `THERM` |
| batteries and fuel cells | storage | the microgrid: `ELEC`; thermal control: `THERM` |
| the habitat modules | heat, air, pressure | power: `ELEC`; thermal loops: `THERM`, `ADVEC`; the structure: `MECH` |
| the rover | drivetrain and wheel–soil contact | the road: `MECH`; its motors: `ROT`; charging: `ELEC` |
| the microgrid | a lumped circuit, nodal analysis | every electrical agent: `ELEC`. This is the workbench's style D |

**Today:** a seven-agent 2-D rocket ascent, coupled and conserving ([[case-study-rocket-ascent-2d-atlas-0.1]]); a powertrain from rotor shaft to battery ([[case-study-powertrain-atlas-0.1]]); a closed coolant loop ([[case-study-cooling-loop-atlas-0.1]]); and fields coupled to lumped circuits in the workbench (style D). Plume–surface interaction, regolith and the lunar environment are not built.

**Prompts:**

> **3a, hero, clean plate (21:9).** A Moon base near the lunar south pole under a very low grazing sun: long black shadows, a pure black sky, Earth low on the horizon. A landing craft descends on a bright exhaust column that kicks up a wide radial spray of grey regolith dust. Habitat modules half-buried under regolith berms, tall vertical solar array towers catching the sunlight, a small power station with large radiator panels in the distance, a pressurised rover on a graded road. Photoreal, cinematic, realistic space-agency concept art, 35mm, no text, no flags, no insignia.

> **3b, hero, assembled variant (21:9).** The same base as 3a. Thin glowing cyan lines link the lander, the dust plume, the solar towers, the batteries, the power station, the habitat and the rover, like a living network. Small pulses of light travel along each link in both directions. A faint translucent tiling surrounds the dust plume and the habitat, showing the simulation pieces. Restrained, elegant, photoreal.

> **3c, detail (16:9).** A close-up of a lander's engine plume striking the lunar surface. The regolith erodes into a shallow crater, and dust grains fly outward in straight ballistic lines, since there is no air. The grains glint in hard sunlight against the black sky. Photoreal, high-speed-photography feel.

> **3d, detail (16:9).** The lunar night at the same base, lit only by earthshine and the habitat's windows. Radiator panels faintly glow warm, and a rover charges at a dock. Thin cyan lines trace power flowing from a battery bank to the habitat. Quiet, cold, beautiful, photoreal.

---

## 5. Handing the images back

When the owner returns images, the website chat:
1. stores each under `site/images/vision/` with its prompt, the generator's name and version, and the date, in a sidecar file;
2. checks each against §1: no text, no marks, the palette, the negative space;
3. draws the agent graph over each clean plate in SVG, from the tables above;
4. labels every one of them **Illustration** on the page.

---

## See Also

- [[proposal-website-plan]] — where these go
- [[f1-pathmap-and-end-goal]] — the vault's end goal, of which scenario 1 is the picture
- [[port-algebra-atlas-0.1]] — the five port types every table above uses
- [[chart-operator-architecture]] — what "learned expert" means in each scenario
