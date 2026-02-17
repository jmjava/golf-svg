# Golf Putt Illustration Generator

Generate SVG illustrations of different putt conditions to assist in green reading practice.
Uses realistic shading to represent visual cues that golfers actually see on putting greens.

## Features

- Creates 120x660 pixel SVG images with a 120px annotation sidebar
- Illustrates putt break in 3 sections (start/ball, middle, near hole)
- **Realistic shading** based on actual green reading visual cues:
  - **Grain shading**: Lighter/shiny = with grain (faster), Darker/dull = against grain (slower)
  - **Elevation shading**: Darker = lower/uphill areas, Lighter = higher/downhill areas
  - **Break shading**: Darker on high side, lighter on low side
  - **Hole edge shading**: Shows grain direction with gradient effect
- Physics-based trajectory using scipy ODE integration
- AimPoint methodology for slope reading and finger-count calculation

## Visual Cues Explained

The shading in these illustrations mimics what golfers actually observe when reading greens:

1. **Grain Direction**:
   - When putting **with the grain**, grass appears **lighter and shiny** because you're looking at the reflective tops of grass blades
   - When putting **against the grain**, grass appears **darker and duller** because you're looking into the grass blades
   - The hole edge will show a "burned" or ragged appearance on the side where grain is growing

2. **Elevation Changes**:
   - **Uphill putts**: The area around the ball appears darker (lower areas hold more moisture and appear darker)
   - **Downhill/flat putts**: The area appears lighter (better drainage, higher areas appear lighter)

3. **Break Direction**:
   - The **high side** (where ball breaks from) appears **darker**
   - The **low side** (where ball breaks toward) appears **lighter**
   - The intensity of shading corresponds to the degree of break

### Using Adjustable Tint Sunglasses to Read Grain

One of the most effective tools for seeing grain on putting greens is a pair of
**adjustable tint sunglasses** — lenses where you can dial the darkness level up
or down in real time (e.g.
[OUANGANC Adjustable Sunglasses](https://www.amazon.com/OUANGANC-Adjustable-Sunglasses-Protection-Universal/dp/B0F2J4Z91K)).

**Why they work:**
Grain is visible because grass blades reflect light differently depending on
direction. With the grain (downgrain) the surface looks **shiny and lighter**;
against the grain it looks **dark and matte**. Ordinary sunglasses reduce all
light equally, which can wash out this subtle contrast. Adjustable tint lenses
let you find the *exact* darkness level that maximizes the shiny-vs-dull
difference on that day's light conditions.

**How to use them on the green:**

1. **Start with a medium tint** — stand behind your ball and look toward the
   hole. Note which areas look lighter (downgrain / with grain) and which look
   darker (into the grain / against grain).
2. **Slowly increase the tint** — as you darken the lens, the shiny (with-grain)
   areas will "pop" more against the darker (against-grain) areas. Stop at the
   tint level where the contrast is strongest.
3. **Check from multiple angles** — walk to the side and behind the hole. Grain
   direction becomes especially obvious when you look across the green at a low
   angle with the optimized tint.
4. **Look at the hole edge** — with the right tint you can clearly see the
   "burned" or ragged lip on the side where grain grows over the cup edge.
   Grain grows *toward* the ragged side.

**Tips:**
- Bermuda grass (warm-climate courses) has the strongest grain effect — the
  adjustable tint is most useful here
- Bent grass (cool-climate courses) has subtler grain, but the adjustable tint
  can still reveal it in afternoon light when shadows are longer
- The optimal tint changes throughout the day as the sun angle shifts — being
  able to adjust on the fly is the key advantage over fixed-tint sunglasses
- Polarized adjustable lenses work even better because polarization suppresses
  glare while the tint control enhances the grain contrast

## Usage

### Basic Usage

```python
from create_putt_illustration import PuttIllustrationGenerator

generator = PuttIllustrationGenerator()

# Generate an SVG
svg, section_slopes = generator.generate_svg(
    with_grain=True,            # True = with grain (faster), False = against grain (slower)
    uphill=True,                # True = uphill (slower), False = downhill/flat
    break_direction="right",    # "left" or "right"
    slope_percent=1.5,          # Slope percentage (0.5-5.0%)
    output_path="my_putt.svg"   # Optional: save to file
)
```

### Advanced Usage (Break Changes & Ridge Putts)

```python
# Double break: slope changes mid-putt
svg, section_slopes = generator.generate_svg(
    with_grain=True,
    uphill=False,
    break_direction="right",
    slope_percent=1.0,
    break_change_points=[(5.0, 2.5)],  # At 5ft, slope changes to 2.5%
    output_path="double_break.svg"
)

# Ridge putt: break reverses direction
svg, section_slopes = generator.generate_svg(
    with_grain=True,
    uphill=False,
    break_direction="right",
    slope_percent=2.0,
    break_change_points=[(4.0, -1.8)],  # Negative = direction reversal
    ridge_putt=True,
    output_path="ridge_putt.svg"
)
```

### Parameters

- **with_grain** (bool): Whether the putt is with the grain (faster) or against it (slower)
- **uphill** (bool): Whether the putt is uphill (slower) or downhill/flat
- **break_direction** (str): `"left"` or `"right"` — the direction the ball will break
- **slope_percent** (float): Slope percentage (clamped to 0.5–5.0%). Typical greens: 1–3%, max realistic: 4–5%
- **break_change_points** (list, optional): List of `(distance_ft, new_slope)` tuples for slope changes mid-putt. Negative slopes reverse break direction (ridge putts)
- **ridge_putt** (bool, optional): Set to `True` when break_change_points include direction reversals
- **output_path** (str, optional): File path to save the SVG

### Running All Scenarios

```bash
python create_putt_illustration.py
```

This generates SVGs in the `output/` directory organized into:
- `output/` — 17 main scenarios (simple, double, ridge, complex breaks)
- `output/aimpoint_1/` — Gentle break scenarios (AimPoint 1 finger)
- `output/aimpoint_2/` — Moderate break scenarios (AimPoint 2 fingers)
- `output/aimpoint_3/` — Strong break scenarios (AimPoint 3+ fingers)

## AimPoint Green Reading Method

AimPoint is a physics-based green reading system used by many PGA Tour professionals
(Collin Morikawa, Viktor Hovland, Adam Scott, and others) to measure slope and
calculate the correct aim point for every putt.

### The 3-Step Process

**Step 1 — Feel the Slope with Your Feet**

Stand at the midpoint between your ball and the hole, straddling the putt line with
feet shoulder-width apart. Close your eyes and feel which foot bears more weight.
The heavier foot is on the low side — the ball will break **toward** it.

**Step 2 — Rate the Slope (1–5 Fingers)**

Assign a severity number based on how much tilt you felt:

| Fingers | Slope % | Green Feel | Typical Context |
|---------|---------|------------|-----------------|
| 1 | ~1% | Barely noticeable | Flat-ish municipal greens |
| 2 | ~2% | Moderate tilt | Average well-maintained course |
| 3 | ~3% | Clearly tilted | Championship / tournament greens |
| 4 | ~4% | Very steep | Severe undulations (Augusta-style) |
| 5 | ~5% | Extreme | Maximum playable — very rare |

> **Realistic ranges:** Most putts on well-maintained courses are 1–3 fingers.
> Slopes above 5% (roughly 3°) are not found on playable putting surfaces.

**Step 3 — Aim Using Your Fingers**

Stand directly behind your ball, close one eye, and extend your arm toward the hole
at eye level. Place your pointer finger just outside the edge of the cup on the
**high side** (the side the ball breaks from). Then hold up the number of fingers
matching your slope rating.

- **1 finger** → aim 1 finger-width to the high side of the hole
- **2 fingers** → aim 2 finger-widths to the high side
- **3 fingers** → aim 3 finger-widths to the high side
- etc.

Your top finger marks your **aim point**. Align your putter face and stroke toward
that point — gravity and the slope will curve the ball back to the hole.

### AimPoint Feel — What You Sense at the Midpoint

When you stand at the midpoint of the putt straddling the target line, your body
gives you all the information AimPoint needs:

| What You Feel | What It Means |
|---------------|---------------|
| Weight shifts to **left foot** | Left side is lower — ball breaks **right to left** |
| Weight shifts to **right foot** | Right side is lower — ball breaks **left to right** |
| **Subtle** pressure shift | ~1% slope — 1 finger read |
| **Clear** pressure shift | ~2% slope — 2 finger read |
| **Heavy** pressure on one foot | ~3%+ slope — 3+ finger read |
| Leaning **back** (uphill putt) | Hit firmer — ball will break **less** |
| Leaning **forward** (downhill putt) | Hit softer — ball will break **more** |

**Examples of what you'd feel:**

- **Gentle right break, downhill (1.5% slope):** You feel clear pressure on your
  **right** foot with a slight forward lean. Ball will fall to the right (right is
  the low side). Use 1 finger.
- **Strong left break, uphill (3.0% slope):** You feel heavy pressure on your
  **left** foot with a backward lean. Ball will fall to the left (left is the low
  side), but less than expected because you'll hit it harder uphill. Use 2–3 fingers.
- **Ridge putt (break reverses at midpoint):** You may feel the slope shift
  under your feet. Walk both halves separately — the first half may tilt one way
  while the second half tilts the opposite.

### Why Downhill Putts Break More Than Uphill

This is one of the most important concepts in green reading. The same side-slope
will produce **more break on a downhill putt** and **less break on an uphill putt**.

**The physics:**
- **Downhill** → You hit the ball softer (less initial speed needed). The ball
  rolls slower across the side-slope, giving gravity more time to pull it sideways.
  Effective slope is ~15% greater (e.g., 2.0% plays like **2.3%**).
- **Uphill** → You hit the ball harder (more speed to fight the hill). The ball
  rolls faster across the side-slope, giving gravity less time to deflect it.
  Effective slope is ~15% less (e.g., 2.0% plays like **1.7%**).

**Concrete examples:**

| Actual Slope | Uphill Effective | Downhill Effective | Difference |
|-------------|-----------------|-------------------|------------|
| 1.0% | 0.9% | 1.2% | 0.3% |
| 2.0% | 1.7% | 2.3% | 0.6% |
| 3.0% | 2.6% | 3.5% | 0.9% |
| 4.0% | 3.4% | 4.6% | 1.2% |

> **Practical tip:** On a 2% side-slope, a downhill putt will break roughly
> **35% more** than the same putt uphill. Always add break on downhill reads
> and subtract it on uphill reads.

### Key Principles

- **Always aim on the high side** — the ball breaks from dark (high) to light (low)
- **Uphill putts break less** — the ball has more speed fighting gravity, reducing
  the time slope can act on it (~15% less effective break)
- **Downhill putts break more** — the ball is slower, exposing it to lateral slope
  longer (~15% more effective break)
- **Grain matters** — putting with the grain (faster) amplifies break; against
  the grain (slower) dampens it
- **Speed controls break** — a firmly struck putt breaks less than a dying putt

### How This Tool Uses AimPoint

Each SVG illustration in this project includes:
- **AimPoint finger count** — calculated from the physics simulation's aim offset
- **Aim point markers** (pink circles) showing where to aim on the high side
- **Section-by-section slope analysis** — how slope changes across the 3 zones
- **Net slope and direction** — the effective break after accounting for reversals
- **Condition adjustments** — uphill/downhill and grain effects on the read

The scenarios are organized into AimPoint categories:
- `aimpoint_1/` — Gentle breaks (1 finger, ~1–1.5% slopes)
- `aimpoint_2/` — Moderate breaks (2 fingers, ~2–2.5% slopes)
- `aimpoint_3/` — Strong breaks (3+ fingers, ~3–5% slopes)

## Visual Elements

Each SVG includes:
- **Layered shading gradients** that combine grain, elevation, and break effects
- **Grass texture pattern** for realistic appearance
- **Gold dashed curve** showing the actual putt path with break
- **White dashed line** showing the straight target line (for reference)
- **Ball** at the bottom (starting position)
- **Hole** at the top (target) with grain-direction shading on the edge
- **Section dividers** separating the three zones (start/ball, middle, near hole)
- **AimPoint markers** (pink circles) showing the aim line
- **Right sidebar** with detailed AimPoint analysis, section slopes, conditions, and reading tips

## Creating Multiple Variations

```python
generator = PuttIllustrationGenerator()

conditions = [
    (True, True, "right", 1.5),
    (True, False, "right", 2.0),
    (False, False, "left", 2.5),
    # ... more combinations
]

for i, (grain, up, direction, slope) in enumerate(conditions):
    generator.generate_svg(
        with_grain=grain,
        uphill=up,
        break_direction=direction,
        slope_percent=slope,
        output_path=f"putt_{i+1}.svg"
    )
```
