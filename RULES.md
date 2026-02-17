# GOLF PUTT ILLUSTRATION RULES

## CRITICAL INSTRUCTION
**BEFORE MAKING ANY CODE CHANGE:**
1. Read this rules document completely
2. Verify your change against ALL rules in the checklist below
3. Test the change validates correctly
4. If user requests a fix, ADD IT TO THESE RULES immediately

## FUNDAMENTAL RULES (NEVER VIOLATE)

### 1. SHADING RULE
- **Dark = HIGH (higher elevation)**
- **Light = LOW (lower elevation)**
- **Ball ALWAYS breaks from DARK (high) to LIGHT (low)**

### 2. SLOPE DIRECTION NAMING
- **"L to R"** means: **L (Left) is HIGHER** → Dark on LEFT, Light on RIGHT → Ball breaks RIGHT
- **"R to L"** means: **R (Right) is HIGHER** → Dark on RIGHT, Light on LEFT → Ball breaks LEFT

### 3. BREAK DIRECTION LOGIC
- If breaking RIGHT: Dark on LEFT, Light on RIGHT (L to R slope)
- If breaking LEFT: Dark on RIGHT, Light on LEFT (R to L slope)

### 4. PATH GENERATION - CRITICAL - YELLOW LINE CURVATURE
**THE YELLOW LINE (PUTT PATH) MUST VISUALLY SHOW THE ACTUAL BREAK DIRECTION:**

**CRITICAL RULE - PATH MUST START AT THE BALL:**
- **The path MUST start at the ball position (center, bottom)**
- The ball is positioned at the center horizontally, at the bottom of the green
- The path starts at this exact ball position

**CURVATURE RULES:**
- **"R to L"** means: The yellow line curves from **RIGHT to LEFT** as it moves to the hole
  - You **AIM RIGHT** (on the high/dark side) and the ball **CURVES BACK LEFT** (toward the low/light side)
  - Dark on RIGHT (high), light on LEFT (low)
  - Path starts at ball (center), curves RIGHT initially (toward high/dark side), then curves LEFT back toward center (hole)
  
- **"L to R"** means: The yellow line curves from **LEFT to RIGHT** as it moves to the hole
  - You **AIM LEFT** (on the high/dark side) and the ball **CURVES BACK RIGHT** (toward the low/light side)
  - Dark on LEFT (high), light on RIGHT (low)
  - Path starts at ball (center), curves LEFT initially (toward high/dark side), then curves RIGHT back toward center (hole)

**UNIVERSAL RULE:**
- **Ball ALWAYS breaks from DARK (high) to LIGHT (low)** - This applies to BOTH "R to L" and "L to R"
- The yellow line must visually show this movement from dark side toward light side

**VALIDATION RULES:**
- **Path MUST start at the ball position (center x, bottom y)**
- If section label is "R to L": Path starts at ball (center), curves RIGHT (toward high/dark side), then LEFT back to hole
- If section label is "L to R": Path starts at ball (center), curves LEFT (toward high/dark side), then RIGHT back to hole
- Path must always show movement from dark (high) side toward light (low) side
- Text description must match what the yellow line actually shows

### 5. SECTION INDEPENDENCE
- Each section (1, 2, 3) has its OWN slope and direction
- Each section's shading is INDEPENDENT based on its own direction
- Section 1 (start): 0-3.3ft
- Section 2 (middle): 3.3-6.6ft  
- Section 3 (hole): 6.6-10ft

### 6. SLOPE DIRECTION CALCULATION - CRITICAL
**EACH SECTION MUST BE CALCULATED INDEPENDENTLY:**

For each section:
1. Determine if THIS SECTION breaks RIGHT or LEFT:
   - If NO break_change_points in this section: use base break_direction
   - If break_change_points occur in this section: use the NEW slope's direction
   - For ridge putts: if section slope is negative, REVERSE the direction
   
2. Convert to label string:
   - If section breaks RIGHT → label is "L to R" (L is higher, dark on left)
   - If section breaks LEFT → label is "R to L" (R is higher, dark on right)

**EXAMPLE:**
- break_direction="left", slope_percent=0.8, break_change=[(4.0, 2.2)]
- Section 1 (0-3.3ft): 0.8% breaking LEFT → "R to L" ✓
- Section 2 (3.3-6.6ft): Changes to 2.2% at 4ft, still breaking LEFT → "R to L" ✓
- Section 3 (6.6-10ft): Continues 2.2% breaking LEFT → "R to L" ✓

**NEVER use base direction for all sections - each must be independent!**

### 7. AIM POINT
- Based on WEIGHTED AVERAGE of section slopes: Hole 40%, Middle 30%, Begin 20%
- Position: On the HIGH (dark) side, opposite where ball breaks to
- If breaking RIGHT (L to R): Aim on LEFT (dark/high side)
- If breaking LEFT (R to L): Aim on RIGHT (dark/high side)

### 8. PATH CURVATURE
- Curvature builds up over distance
- Steeper slopes over longer distances = more curve
- Path is ONE SMOOTH CONTINUOUS curve (ball has momentum, doesn't step)

## VALIDATION CHECKLIST

Before making ANY change, verify:
- [ ] Dark is always on the high side
- [ ] Light is always on the low side  
- [ ] "L to R" label = L is higher, breaks right
- [ ] "R to L" label = R is higher, breaks left
- [ ] **Path MUST start at the ball position (center x, bottom y)**
- [ ] **"R to L" yellow line curves from RIGHT to LEFT (aim right, curves back left)**
- [ ] **"L to R" yellow line curves from LEFT to RIGHT (aim left, curves back right)**
- [ ] Path always shows movement from DARK (high) to LIGHT (low)
- [ ] Path starts at ball (center), curves toward high (dark) side, then curves back to hole (center)
- [ ] Each section uses its OWN direction (not base direction)
- [ ] Shading matches the section's direction label
- [ ] **Text description matches what the yellow line actually shows**
- [ ] Aim point is on the high (dark) side

