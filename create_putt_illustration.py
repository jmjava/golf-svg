"""
Generate SVG illustrations of putt conditions for green reading practice.
Based on AimPoint green reading methodology with realistic slopes (typically 1-3%,
up to 10% for ridge/reversal putts).
Creates 120x600 pixel images with detailed annotations.

Refactored to use clean physics model with scipy ODE integration.
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path
from enum import Enum
import numpy as np
from scipy.integrate import solve_ivp


# =============================================================================
# DATA MODELS
# =============================================================================

class BreakDirection(Enum):
    """Direction the ball breaks (toward low side)."""
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class LayoutConfig:
    """
    Shared layout dimensions for the SVG canvas.
    
    Single source of truth for all positional constants used by
    the renderer, generator, and coordinate mapping.
    """
    width: int = 120
    height: int = 600
    text_column_width: int = 120
    putt_length_ft: float = 10.0
    margin: int = 30  # Top/bottom margin for ball and hole positions
    
    @property
    def total_width(self) -> int:
        return self.width + self.text_column_width
    
    @property
    def center_x(self) -> float:
        return self.width / 2
    
    @property
    def ball_y(self) -> float:
        return self.height - self.margin
    
    @property
    def hole_y(self) -> float:
        return float(self.margin)
    
    @property
    def total_distance_pixels(self) -> float:
        """Vertical pixel distance from ball to hole."""
        return self.ball_y - self.hole_y
    
    @property
    def section_height(self) -> float:
        return self.height / 3


class Colors:
    """
    Centralized color palette for SVG rendering.
    
    Grouped by semantic meaning so the theme can be adjusted in one place.
    """
    # --- Green terrain tones (dark to light) ---
    GREEN_DARKEST = "#0D1F0A"
    GREEN_VERY_DARK = "#1B3D14"
    GREEN_DARK = "#2E5A1F"
    GREEN_DARK_MED = "#3E7B2F"
    GREEN_DARK_ARROW = "#3A6B1F"   # Against-grain arrow color
    GREEN_MED_DARK = "#4A7C27"
    GREEN_MEDIUM = "#689F38"
    GREEN_BASE_DARK = "#558B2F"
    GREEN_LIGHT_MED = "#8BC34A"
    GREEN_LIGHT = "#AED581"
    GREEN_LIGHTER = "#C5E1A5"
    GREEN_PALE_MED = "#A5D6A7"
    GREEN_PALE = "#C8E6C9"
    GREEN_PALEST = "#E8F5E9"
    GREEN_BRIGHT = "#81C784"
    GREEN_TEXTURE = "#5A9B3F"
    
    # --- UI accent colors ---
    GOLD = "#FFD700"           # Putt path, labels, headings
    PINK = "#FF6B6B"           # AimPoint markers, break change markers
    CYAN = "#00CED1"           # Section divider lines
    
    # --- Neutral tones ---
    WHITE = "#FFFFFF"
    OFF_WHITE = "#F5F5F5"
    LIGHT_GRAY = "#CCCCCC"
    MED_GRAY = "#999999"
    DARK_GRAY = "#2C2C2C"
    NEAR_BLACK = "#1A1A1A"
    BLACK = "#000000"


@dataclass
class SlopeSection:
    """
    Represents a slope section on the green.
    
    Attributes:
        start_ft: Starting distance from ball (feet)
        end_ft: Ending distance from ball (feet)
        slope_percent: Slope percentage (positive = same as base direction, negative = reversed)
        base_direction: The base break direction for this putt
    """
    start_ft: float
    end_ft: float
    slope_percent: float
    base_direction: BreakDirection
    
    @property
    def breaks_right(self) -> bool:
        """Returns True if ball breaks right in this section."""
        # Negative slope reverses direction (ridge putts)
        if self.slope_percent < 0:
            return self.base_direction == BreakDirection.LEFT
        return self.base_direction == BreakDirection.RIGHT
    
    @property
    def direction_label(self) -> str:
        """Returns 'L to R' or 'R to L' based on break direction."""
        return "L to R" if self.breaks_right else "R to L"
    
    @property
    def section_number(self) -> int:
        """Returns section number (1, 2, or 3) based on position."""
        if self.end_ft <= 3.33:
            return 1
        if self.end_ft <= 6.67:
            return 2
        return 3
    
    def description(self) -> str:
        """Generate human-readable description."""
        section_names = {1: "Start section (0-3.3ft)", 2: "Middle section (3.3-6.6ft)", 3: "Hole section (6.6-10ft)"}
        return f"{section_names[self.section_number]}: {abs(self.slope_percent):.1f}% {self.direction_label} slope"


@dataclass
class PuttConfig:
    """Configuration for a putt scenario."""
    with_grain: bool
    uphill: bool
    break_direction: BreakDirection
    slope_percent: float
    break_change_points: List[Tuple[float, float]] = field(default_factory=list)
    ridge_putt: bool = False
    putt_length_ft: float = 10.0
    
    def get_sections(self) -> List[SlopeSection]:
        """
        Build list of slope sections from configuration.
        Returns 3 sections corresponding to start/middle/hole.
        
        Each section gets the slope from the last break_change_point
        that falls before the section's end boundary. If no change
        applies, it uses the base slope_percent.
        """
        changes = sorted(self.break_change_points, key=lambda x: x[0])
        boundaries = [(0.0, 3.33), (3.33, 6.67), (6.67, 10.0)]
        
        sections = []
        for start_ft, end_ft in boundaries:
            # Start with base slope, apply latest change before this section ends
            slope = self.slope_percent
            for dist, new_slope in changes:
                if dist < end_ft:
                    slope = new_slope
            sections.append(SlopeSection(start_ft, end_ft, slope, self.break_direction))
        
        return sections


# =============================================================================
# PHYSICS ENGINE
# =============================================================================

class PuttingPhysics:
    """
    Physics engine for golf ball putting simulation.
    
    Uses scipy.integrate.solve_ivp for accurate ODE integration.
    Works in normalized coordinates: t_param (0=ball, 1=hole) and
    lateral offset (abstract units proportional to slope effect).
    
    Physics model based on real putting mechanics:
    - Ball rolls with deceleration from friction
    - Lateral acceleration from gravity on slope: a = g * sin(slope_angle)
    - Break increases as ball slows (more time exposed to lateral force)
    - Typical break: 1-2 inches per foot for 1% slope
    
    References:
    - Pelz, Dave. "Dave Pelz's Putting Bible" - break estimation formulas
    - AimPoint methodology for slope reading
    """
    
    def __init__(self, config: PuttConfig):
        """
        Initialize physics engine.
        
        Args:
            config: Putt configuration
        """
        self.config = config
        self.sections = config.get_sections()
    
    def get_slope_at_distance(self, dist_ft: float) -> Tuple[float, bool]:
        """
        Get slope percentage and break direction at a given distance from ball.
        
        Returns:
            Tuple of (slope_percent, breaks_right)
        """
        for section in self.sections:
            if section.start_ft <= dist_ft < section.end_ft:
                return abs(section.slope_percent), section.breaks_right
        
        # Default to last section
        last = self.sections[-1]
        return abs(last.slope_percent), last.breaks_right
    
    def compute_trajectory(self) -> List[Tuple[float, float]]:
        """
        Compute realistic ball trajectory in normalized coordinates.
        
        RULES (from rules.md):
        - "L to R" (breaks right): aim LEFT (high/dark side), ball curves back RIGHT
        - "R to L" (breaks left): aim RIGHT (high/dark side), ball curves back LEFT
        - Path starts at ball (center), curves toward high/dark side, then back to hole
        
        Returns:
            List of (t_param, x_offset) normalized coordinates where:
            - t_param: 0.0 (ball) to 1.0 (hole)
            - x_offset: lateral offset (positive = right, negative = left)
        """
        # Determine overall break direction from hole section
        overall_breaks_right = self.sections[-1].breaks_right
        
        # Use ODE integration for physically correct trajectory
        return self._generate_smooth_trajectory(overall_breaks_right)
    
    def _weighted_average_slope(self) -> float:
        """Calculate weighted average slope (hole section weighted more)."""
        s1 = abs(self.sections[0].slope_percent)  # Start
        s2 = abs(self.sections[1].slope_percent)  # Middle
        s3 = abs(self.sections[2].slope_percent)  # Hole
        return (s1 * 0.2 + s2 * 0.3 + s3 * 0.5)
    
    def _generate_smooth_trajectory(self, breaks_right: bool) -> List[Tuple[float, float]]:
        """
        Generate physically correct trajectory.
        
        IMPORTANT: The ball curves in ONE direction only (the break direction).
        There is NO S-curve for a consistent slope.
        
        For "L to R" (breaks_right=True):
        - Aim LEFT (toward high/dark side)
        - Ball curves RIGHT continuously toward hole (the break)
        
        For "R to L" (breaks_right=False):
        - Aim RIGHT (toward high/dark side)  
        - Ball curves LEFT continuously toward hole (the break)
        
        The trajectory is a parabolic arc, not an S-curve.
        """
        # Use ODE integration for physically correct curve
        return self._solve_trajectory_ode(breaks_right)
    
    def _solve_trajectory_ode(self, breaks_right: bool) -> List[Tuple[float, float]]:
        """
        Solve trajectory using scipy ODE integration.
        
        CORRECT PHYSICS for putting:
        - Gravity on slope creates CONSTANT lateral acceleration toward low side
        - Ball is aimed to compensate - initial velocity toward high side
        - Ball curves continuously in ONE direction (the break direction)
        - Result: parabolic arc, NOT an S-curve
        
        For "R to L" (breaks left): ball curves LEFT the entire path
        For "L to R" (breaks right): ball curves RIGHT the entire path
        
        Kinematics: x(t) = v0*t + 0.5*a*t²
        For ball to end at center (x=0 at t=1): v0 = -0.5*a
        """
        
        # Scale factor for visual appearance
        # For 1% slope: ~10px deviation, for 3% slope: ~30px deviation
        BASE_BREAK_SCALE = 40.0  # Pixels per slope-percent-squared
        
        # Elevation affects break amount:
        # - Uphill: ball slows down -> less time to break -> LESS break (0.8x)
        # - Downhill: ball speeds up then slows near hole -> MORE break (1.2x)
        if self.config.uphill:
            BREAK_SCALE = BASE_BREAK_SCALE * 0.8  # Less break on uphill
        else:
            BREAK_SCALE = BASE_BREAK_SCALE * 1.2  # More break on downhill
        
        def derivatives(t_param: float, state: np.ndarray) -> np.ndarray:
            """
            State: [x_offset, x_velocity]
            t_param: progress from 0 (ball) to 1 (hole)
            
            Acceleration is ALWAYS toward low side (break direction).
            Elevation affects how much the ball breaks.
            """
            x, vx = state
            
            # Get section-specific slope and direction
            dist_ft = t_param * self.config.putt_length_ft
            slope_pct, section_breaks_right = self.get_slope_at_distance(dist_ft)
            
            # Acceleration toward low side: positive = right, negative = left
            accel_dir = 1.0 if section_breaks_right else -1.0
            
            # Acceleration magnitude proportional to slope
            accel = slope_pct * BREAK_SCALE * accel_dir
            
            return np.array([vx, accel])
        
        def simulate_with_v0(v0: float) -> float:
            """Simulate and return final x position."""
            y0_test = np.array([0.0, v0])
            sol = solve_ivp(derivatives, (0.0, 1.0), y0_test, dense_output=True,
                           rtol=1e-6, atol=1e-8, max_step=0.01)
            return sol.sol(1.0)[0]
        
        # Start with estimate from average slope
        weighted_slope = self._weighted_average_slope()
        avg_accel_dir = 1.0 if breaks_right else -1.0
        avg_acceleration = weighted_slope * BREAK_SCALE * avg_accel_dir
        initial_velocity = -0.5 * avg_acceleration
        
        # Iterate to find v0 that lands ball at center (x=0)
        # Newton-Raphson style iteration
        for _ in range(5):
            final_x = simulate_with_v0(initial_velocity)
            if abs(final_x) < 0.1:  # Close enough
                break
            # Adjust v0: if ball ends left (negative), need more rightward initial velocity
            initial_velocity -= final_x * 0.8  # Correction factor
        
        # Initial conditions: start at center with calculated velocity toward aim
        y0 = np.array([0.0, initial_velocity])
        
        # Solve the ODE - this is deterministic physics, no fallback needed
        solution = solve_ivp(
            derivatives, (0.0, 1.0), y0,
            dense_output=True,
            rtol=1e-6,
            atol=1e-8,
            max_step=0.01
        )
        
        # Sample points evenly along the path
        num_points = 80
        t_eval = np.linspace(0, 1, num_points)
        sol = solution.sol(t_eval)
        
        # Return normalized coordinates: (t_param, x_offset)
        points = [(t_eval[i], sol[0, i]) for i in range(num_points)]
        
        # Ensure final point lands exactly at center (x=0)
        points[-1] = (1.0, 0.0)
        
        return points


# =============================================================================
# SVG PATH GENERATOR
# =============================================================================

class SVGPathGenerator:
    """Generates smooth SVG paths from trajectory points."""
    
    @staticmethod
    def points_to_svg_path(points: List[Tuple[float, float]]) -> str:
        """
        Convert trajectory points to SVG path string using cubic bezier curves.
        """
        if len(points) < 2:
            return ""
        
        if len(points) == 2:
            return f"M {points[0][0]:.1f} {points[0][1]:.1f} L {points[1][0]:.1f} {points[1][1]:.1f}"
        
        # Sample key points for smoother curve
        key_points = SVGPathGenerator._sample_key_points(points, max_points=15)
        
        # Build path with cubic bezier curves
        path_parts = [f"M {key_points[0][0]:.1f} {key_points[0][1]:.1f}"]
        
        for i in range(len(key_points) - 1):
            p1 = key_points[i]
            p2 = key_points[i + 1]
            
            # Calculate control points (1/3 of distance)
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            c1_x = p1[0] + dx * 0.33
            c1_y = p1[1] + dy * 0.33
            c2_x = p2[0] - dx * 0.33
            c2_y = p2[1] - dy * 0.33
            
            path_parts.append(
                f"C {c1_x:.1f} {c1_y:.1f}, {c2_x:.1f} {c2_y:.1f}, {p2[0]:.1f} {p2[1]:.1f}"
            )
        
        return " ".join(path_parts)
    
    @staticmethod
    def _sample_key_points(points: List[Tuple[float, float]], 
                          max_points: int) -> List[Tuple[float, float]]:
        """Sample evenly-spaced key points from trajectory."""
        if len(points) <= max_points:
            return points
        
        indices = [int(i * (len(points) - 1) / (max_points - 1)) for i in range(max_points)]
        indices[0] = 0
        indices[-1] = len(points) - 1
        
        return [points[i] for i in sorted(set(indices))]
    
    @staticmethod
    def calculate_aim_point(points: List[Tuple[float, float]], 
                           hole_y: float, center_x: float) -> Tuple[float, float]:
        """
        Calculate the AIM point - the initial starting line of the putt.
        
        This represents where the putter face is aimed at the start,
        approximately 2 feet from the ball along the initial trajectory.
        
        Returns the x-position of the aim point (y is set by caller).
        """
        if len(points) < 10:
            return (center_x, hole_y)
        
        # Ball position
        ball_x, ball_y = points[0]
        
        # Find the point approximately 2 feet from ball
        # Total path is ~10ft (540 pixels from ball to hole)
        # 2 feet = 20% of path = about index 16 in 80-point trajectory
        aim_idx = min(len(points) // 5, 16)  # ~20% of path = 2 feet
        
        aim_x = points[aim_idx][0]
        
        # Return the aim x position
        return (aim_x, hole_y)
    
    @staticmethod
    def find_apex(points: List[Tuple[float, float]], center_x: float) -> Tuple[float, float]:
        """Find the point of maximum deviation from center (for reference)."""
        max_deviation = 0
        apex = points[0]
        
        for point in points:
            deviation = abs(point[0] - center_x)
            if deviation > max_deviation:
                max_deviation = deviation
                apex = point
        
        return apex


# =============================================================================
# SVG RENDERER
# =============================================================================

class SVGRenderer:
    """Renders SVG elements for putt illustrations."""
    
    def __init__(self, layout: LayoutConfig):
        self.layout = layout
        # Convenience aliases for frequently accessed values
        self.width = layout.width
        self.height = layout.height
        self.text_column_width = layout.text_column_width
        self.total_width = layout.total_width
        self.section_height = layout.section_height
        self.putt_length_ft = layout.putt_length_ft
        self.center_x = layout.center_x
        self.ball_y = layout.ball_y
        self.hole_y = layout.hole_y
    
    def render_gradient_defs(self, config: PuttConfig, sections: List[SlopeSection]) -> str:
        """Generate all gradient definitions."""
        return f"""
        {self._grain_gradient(config.with_grain)}
        {self._elevation_gradient(config.uphill)}
        {self._break_gradients(sections)}
        {self._hole_grain_gradient(config.with_grain)}
        """
    
    def _grain_gradient(self, with_grain: bool) -> str:
        """Create grain shading gradient - lighter/shiny for with grain, darker for against."""
        C = Colors
        if with_grain:
            # With grain = shiny, lighter appearance (ball at bottom looking toward hole)
            return f'''<linearGradient id="grainGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" style="stop-color:{C.GREEN_PALEST};stop-opacity:0.6" />
            <stop offset="30%" style="stop-color:{C.GREEN_PALE};stop-opacity:0.5" />
            <stop offset="60%" style="stop-color:{C.GREEN_PALE_MED};stop-opacity:0.35" />
            <stop offset="100%" style="stop-color:{C.GREEN_BRIGHT};stop-opacity:0.2" />
        </linearGradient>'''
        # Against grain = dark, matte appearance
        return f'''<linearGradient id="grainGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" style="stop-color:{C.GREEN_DARKEST};stop-opacity:0.7" />
            <stop offset="30%" style="stop-color:{C.GREEN_VERY_DARK};stop-opacity:0.6" />
            <stop offset="60%" style="stop-color:{C.GREEN_DARK};stop-opacity:0.45" />
            <stop offset="100%" style="stop-color:{C.GREEN_DARK_MED};stop-opacity:0.3" />
        </linearGradient>'''
    
    def _elevation_gradient(self, uphill: bool) -> str:
        """Create elevation shading gradient."""
        C = Colors
        if uphill:
            return f'''<linearGradient id="elevationGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" style="stop-color:{C.GREEN_LIGHT};stop-opacity:0.3" />
            <stop offset="20%" style="stop-color:{C.GREEN_LIGHT_MED};stop-opacity:0.4" />
            <stop offset="50%" style="stop-color:{C.GREEN_MEDIUM};stop-opacity:0.6" />
            <stop offset="80%" style="stop-color:{C.GREEN_MED_DARK};stop-opacity:0.85" />
            <stop offset="100%" style="stop-color:{C.GREEN_VERY_DARK};stop-opacity:1.0" />
        </linearGradient>'''
        return f'''<linearGradient id="elevationGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" style="stop-color:{C.GREEN_VERY_DARK};stop-opacity:1.0" />
            <stop offset="20%" style="stop-color:{C.GREEN_MED_DARK};stop-opacity:0.85" />
            <stop offset="50%" style="stop-color:{C.GREEN_MEDIUM};stop-opacity:0.6" />
            <stop offset="80%" style="stop-color:{C.GREEN_LIGHT_MED};stop-opacity:0.4" />
            <stop offset="100%" style="stop-color:{C.GREEN_LIGHT};stop-opacity:0.3" />
        </linearGradient>'''
    
    def _break_gradients(self, sections: List[SlopeSection]) -> str:
        """Create break shading gradients for each section."""
        C = Colors
        gradients = []
        
        for section in sections:
            section_num = section.section_number
            gradient_id = f"breakGradientSec{section_num}"
            
            # Calculate intensity from slope
            abs_slope = abs(section.slope_percent)
            intensity = 0.3 + (abs_slope - 1.0) * 0.35
            intensity = max(0.3, min(intensity, 1.0))
            
            dark_opacity = 0.6 + intensity * 0.4
            light_opacity = 0.5 - intensity * 0.3
            
            # Dark = high side, Light = low side
            # Breaking right: dark LEFT, light RIGHT
            # Breaking left: dark RIGHT, light LEFT
            if section.breaks_right:
                gradients.append(f'''<linearGradient id="{gradient_id}" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" style="stop-color:{C.GREEN_VERY_DARK};stop-opacity:{dark_opacity}" />
            <stop offset="25%" style="stop-color:{C.GREEN_MEDIUM};stop-opacity:{dark_opacity - (dark_opacity - light_opacity) * 0.25}" />
            <stop offset="50%" style="stop-color:{C.GREEN_LIGHT_MED};stop-opacity:{dark_opacity - (dark_opacity - light_opacity) * 0.5}" />
            <stop offset="75%" style="stop-color:{C.GREEN_LIGHT};stop-opacity:{dark_opacity - (dark_opacity - light_opacity) * 0.75}" />
            <stop offset="100%" style="stop-color:{C.GREEN_LIGHTER};stop-opacity:{light_opacity}" />
        </linearGradient>''')
            else:
                gradients.append(f'''<linearGradient id="{gradient_id}" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" style="stop-color:{C.GREEN_LIGHTER};stop-opacity:{light_opacity}" />
            <stop offset="25%" style="stop-color:{C.GREEN_LIGHT};stop-opacity:{light_opacity + (dark_opacity - light_opacity) * 0.2}" />
            <stop offset="50%" style="stop-color:{C.GREEN_LIGHT_MED};stop-opacity:{light_opacity + (dark_opacity - light_opacity) * 0.5}" />
            <stop offset="75%" style="stop-color:{C.GREEN_MEDIUM};stop-opacity:{light_opacity + (dark_opacity - light_opacity) * 0.75}" />
            <stop offset="100%" style="stop-color:{C.GREEN_VERY_DARK};stop-opacity:{dark_opacity}" />
        </linearGradient>''')
        
        return '\n        '.join(gradients)
    
    def _hole_grain_gradient(self, with_grain: bool) -> str:
        """Create gradient for hole edge."""
        C = Colors
        if with_grain:
            return f'''<linearGradient id="holeGrainGradient" x1="50%" y1="0%" x2="50%" y2="100%">
            <stop offset="0%" style="stop-color:{C.WHITE};stop-opacity:0.3" />
            <stop offset="50%" style="stop-color:{C.LIGHT_GRAY};stop-opacity:0.5" />
            <stop offset="100%" style="stop-color:{C.MED_GRAY};stop-opacity:0.6" />
        </linearGradient>'''
        return f'''<linearGradient id="holeGrainGradient" x1="50%" y1="0%" x2="50%" y2="100%">
            <stop offset="0%" style="stop-color:{C.MED_GRAY};stop-opacity:0.6" />
            <stop offset="50%" style="stop-color:{C.LIGHT_GRAY};stop-opacity:0.5" />
            <stop offset="100%" style="stop-color:{C.WHITE};stop-opacity:0.3" />
        </linearGradient>'''
    
    def render_break_shading_rects(self, sections: List[SlopeSection]) -> str:
        """Generate rectangles for each section's break shading."""
        rects = []
        section_height = (self.height - 60) / 3
        
        # Section positions (y increases downward in SVG)
        positions = [
            (self.height - 30 - section_height, section_height),  # Section 1 (bottom)
            (self.height - 30 - 2 * section_height, section_height),  # Section 2 (middle)
            (30, section_height),  # Section 3 (top/hole)
        ]
        
        for section in sections:
            idx = section.section_number - 1
            y_top, height = positions[idx]
            gradient_id = f"breakGradientSec{section.section_number}"
            rects.append(
                f'<rect x="0" y="{y_top}" width="{self.width}" height="{height}" '
                f'fill="url(#{gradient_id})" opacity="1.0" />'
            )
        
        return '\n    '.join(rects)
    
    def render_section_dividers(self, sections: List[SlopeSection]) -> str:
        """Create clear dividing lines between sections with break labels."""
        C = Colors
        elements = []
        
        # Section positions (from ball to hole)
        section_boundaries = [
            self.ball_y,  # Bottom of section 1
            self.ball_y - self.section_height,  # Top of section 1 / bottom of section 2
            self.ball_y - 2 * self.section_height,  # Top of section 2 / bottom of section 3
            self.hole_y  # Top of section 3
        ]
        
        # Draw section divider lines in cyan (different from gold break labels)
        for i in range(1, 3):
            y_pos = section_boundaries[i]
            elements.append(
                f'<line x1="0" y1="{y_pos}" x2="{self.width}" y2="{y_pos}" '
                f'stroke="{C.CYAN}" stroke-width="2" opacity="0.8" stroke-dasharray="8,4" />'
            )
        
        # Add section labels with break info (AFTER lines so they layer on top)
        section_names = ["SEC 1", "SEC 2", "SEC 3"]
        for i, section in enumerate(sections):
            y_top = section_boundaries[i + 1]
            y_bottom = section_boundaries[i]
            y_center = (y_top + y_bottom) / 2
            
            # Break direction and percentage
            direction = "L->R" if section.breaks_right else "R->L"
            slope_text = f"{abs(section.slope_percent):.1f}%"
            
            # Section number badge (left side) - fully opaque background
            elements.append(
                f'<rect x="3" y="{y_center - 22}" width="28" height="14" '
                f'fill="{C.NEAR_BLACK}" rx="2" />'
            )
            elements.append(
                f'<text x="17" y="{y_center - 12}" font-family="Arial, sans-serif" '
                f'font-size="8" fill="{C.WHITE}" text-anchor="middle" font-weight="bold">'
                f'{section_names[i]}</text>'
            )
            
            # Break info badge (center of section) - fully opaque, covers any lines
            badge_width = 54
            badge_x = (self.width - badge_width) / 2
            elements.append(
                f'<rect x="{badge_x}" y="{y_center - 8}" width="{badge_width}" height="16" '
                f'fill="{C.NEAR_BLACK}" rx="3" stroke="{C.GOLD}" stroke-width="1.5" />'
            )
            elements.append(
                f'<text x="{self.width / 2}" y="{y_center + 4}" font-family="Arial, sans-serif" '
                f'font-size="9" fill="{C.GOLD}" text-anchor="middle" font-weight="bold">'
                f'{slope_text} {direction}</text>'
            )
        
        return '\n    '.join(elements)
    
    def render_break_change_markers(self, break_change_points: List[Tuple[float, float]]) -> str:
        """Create visual markers where break changes."""
        C = Colors
        markers = []
        for dist_ft, new_slope in break_change_points:
            y_pos = self.ball_y - (dist_ft / self.putt_length_ft) * (self.ball_y - self.hole_y)
            
            markers.append(
                f'<line x1="0" y1="{y_pos}" x2="{self.width}" y2="{y_pos}" '
                f'stroke="{C.PINK}" stroke-width="2" opacity="0.8" stroke-dasharray="4,2" />'
            )
            markers.append(
                f'<rect x="{self.width - 50}" y="{y_pos - 8}" width="47" height="16" '
                f'fill="{C.PINK}" opacity="0.9" rx="2" />'
            )
            markers.append(
                f'<text x="{self.width - 26.5}" y="{y_pos + 5}" font-family="Arial, sans-serif" '
                f'font-size="7" fill="{C.WHITE}" text-anchor="middle" font-weight="bold">'
                f'BREAK: {new_slope:.1f}%</text>'
            )
        
        return '\n    '.join(markers) if markers else ''
    
    def render_grain_indicators(self, with_grain: bool) -> str:
        """Create visual arrows showing grass grain direction."""
        C = Colors
        indicators = []
        arrow_spacing = 80
        arrow_length = 15
        
        for y_start in range(100, int(self.height - 50), arrow_spacing):
            if with_grain:
                # Arrows point up (toward hole)
                indicators.append(
                    f'<line x1="{self.center_x}" y1="{y_start}" x2="{self.center_x}" y2="{y_start - arrow_length}" '
                    f'stroke="{C.GREEN_LIGHTER}" stroke-width="2" opacity="0.7" marker-end="url(#arrowhead-grain)" />'
                )
            else:
                # Arrows point down (toward ball)
                indicators.append(
                    f'<line x1="{self.center_x}" y1="{y_start}" x2="{self.center_x}" y2="{y_start + arrow_length}" '
                    f'stroke="{C.GREEN_DARK_ARROW}" stroke-width="2" opacity="0.7" marker-end="url(#arrowhead-grain)" />'
                )
        
        return '\n    '.join(indicators) if indicators else ''
    
    def render_explanation_text(self, config: PuttConfig, sections: List[SlopeSection], 
                               aim_fingers: int) -> str:
        """Generate AimPoint explanation text for right column."""
        C = Colors
        x_offset = self.width + 10
        y_start = 25
        line_height = 16
        max_text_width = self.text_column_width - 20
        
        # Calculate NET slope adjusted for elevation
        # SIGNED sum: L->R is positive, R->L is negative (or vice versa)
        # This way opposite breaks cancel out (e.g., ridge putts)
        signed_sum = 0.0
        for s in sections:
            # Use positive for one direction, negative for other
            sign = 1.0 if s.breaks_right else -1.0
            signed_sum += sign * abs(s.slope_percent)
        
        # Average the signed sum
        net_slope = signed_sum / len(sections)
        
        # Take absolute value for display (direction shown separately)
        avg_slope = abs(net_slope)
        
        # Adjust for elevation: uphill = less break, downhill = more break
        if config.uphill:
            adj_slope = avg_slope * 0.85  # Less effective break uphill
        else:
            adj_slope = avg_slope * 1.15  # More effective break downhill
        
        # Direction based on NET slope (positive = L->R, negative = R->L)
        direction_str = "L->R" if net_slope >= 0 else "R->L"
        
        parts = [
            f'<text x="{x_offset}" y="{y_start}" font-family="Arial, sans-serif" font-size="10" fill="{C.GOLD}" font-weight="bold">AIMPOINT READ</text>',
            f'<text x="{x_offset}" y="{y_start + 12}" font-family="Arial, sans-serif" font-size="10" fill="{C.GOLD}" font-weight="bold">ANALYSIS</text>',
        ]
        
        y_pos = y_start + 35
        
        # Slope info - adjusted average
        parts.append(f'<text x="{x_offset}" y="{y_pos}" font-family="Arial, sans-serif" font-size="9" fill="{C.WHITE}"><tspan font-weight="bold">Slope (adj):</tspan> {adj_slope:.1f}%</text>')
        y_pos += line_height
        
        parts.append(f'<text x="{x_offset}" y="{y_pos}" font-family="Arial, sans-serif" font-size="9" fill="{C.WHITE}"><tspan font-weight="bold">Direction:</tspan> {direction_str}</text>')
        y_pos += line_height
        
        parts.append(f'<text x="{x_offset}" y="{y_pos}" font-family="Arial, sans-serif" font-size="9" fill="{C.WHITE}"><tspan font-weight="bold">AimPoint:</tspan> {aim_fingers} finger{"s" if aim_fingers != 1 else ""}</text>')
        y_pos += 13
        
        parts.append(f'<text x="{x_offset + 5}" y="{y_pos}" font-family="Arial, sans-serif" font-size="8" fill="{C.LIGHT_GRAY}">({direction_str} break)</text>')
        y_pos += 15
        
        # Section slopes
        parts.append(f'<text x="{x_offset}" y="{y_pos}" font-family="Arial, sans-serif" font-size="9" fill="{C.GOLD}" font-weight="bold">SECTION SLOPES:</text>')
        y_pos += line_height
        
        for section in sections:
            slope_text = f'Sec {section.section_number}: {abs(section.slope_percent):.1f}% {section.direction_label}'
            parts.append(f'<text x="{x_offset}" y="{y_pos}" font-family="Arial, sans-serif" font-size="8" fill="{C.WHITE}">{slope_text}</text>')
            y_pos += 13
            
            # Wrap description
            desc = section.description()
            for line in self._wrap_text(desc, max_text_width - 5, 7):
                parts.append(f'<text x="{x_offset + 5}" y="{y_pos}" font-family="Arial, sans-serif" font-size="7" fill="{C.LIGHT_GRAY}">{line}</text>')
                y_pos += 11
            y_pos += 2
        
        y_pos += 10
        
        # Conditions
        parts.append(f'<text x="{x_offset}" y="{y_pos}" font-family="Arial, sans-serif" font-size="9" fill="{C.GOLD}" font-weight="bold">CONDITIONS:</text>')
        y_pos += line_height
        parts.append(f'<text x="{x_offset}" y="{y_pos}" font-family="Arial, sans-serif" font-size="8" fill="{C.WHITE}">Grain: {"WITH" if config.with_grain else "AGAINST"}</text>')
        y_pos += 12
        grain_desc = "fast, shiny" if config.with_grain else "slow, dark"
        parts.append(f'<text x="{x_offset}" y="{y_pos}" font-family="Arial, sans-serif" font-size="7" fill="{C.LIGHT_GRAY}">({grain_desc})</text>')
        y_pos += 12
        
        parts.append(f'<text x="{x_offset}" y="{y_pos}" font-family="Arial, sans-serif" font-size="8" fill="{C.WHITE}">Elevation: {"UPHILL" if config.uphill else "FLAT/DOWN"}</text>')
        y_pos += 16
        
        # Reading tips
        parts.append(f'<text x="{x_offset}" y="{y_pos}" font-family="Arial, sans-serif" font-size="9" fill="{C.GOLD}" font-weight="bold">READING TIPS:</text>')
        y_pos += line_height
        
        tips = [
            "1. Feel slope at midpoint",
            f"2. Use {aim_fingers} finger{'s' if aim_fingers != 1 else ''} for aim",
            "3. " + ("Account for grain" if config.with_grain else "Allow slower roll")
        ]
        if config.break_change_points:
            tips.append("4. Break changes marked!")
        
        for tip in tips:
            color = C.GOLD if "4." in tip else C.WHITE
            weight = "bold" if "4." in tip else "normal"
            parts.append(f'<text x="{x_offset}" y="{y_pos}" font-family="Arial, sans-serif" font-size="8" fill="{color}" font-weight="{weight}">{tip}</text>')
            y_pos += 13
        
        return '\n    '.join(parts)
    
    def _wrap_text(self, text: str, max_width: int, font_size: int) -> List[str]:
        """Wrap text to fit within max_width."""
        char_width = font_size * 0.6
        max_chars = int(max_width / char_width)
        
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1
            if current_length + word_length <= max_chars:
                current_line.append(word)
                current_length += word_length
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_length
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines


# =============================================================================
# MAIN GENERATOR
# =============================================================================

class PuttIllustrationGenerator:
    """
    Generate SVG illustrations of golf putt conditions using AimPoint methodology.
    
    Uses realistic putting green slopes based on golf course design standards:
      - 1-2%: Typical greens (gentle to moderate break)
      - 2-3%: Tournament/championship greens (challenging break)
      - 3-4%: Severe undulations, e.g. Augusta National (steep break)
      - 4-5%: Absolute maximum for a playable putting surface (extreme)
    
    Slopes above 5% are not realistic for putting greens and are clamped.
    Includes detailed AimPoint annotations explaining the read and break analysis.
    """
    
    def __init__(self, width: int = 120, height: int = 600, text_column_width: int = 120):
        self.layout = LayoutConfig(width=width, height=height, text_column_width=text_column_width)
        self.renderer = SVGRenderer(self.layout)
    
    def generate_svg(self,
                    with_grain: bool,
                    uphill: bool,
                    break_direction: str,
                    slope_percent: float,
                    break_change_points: Optional[List[Tuple[float, float]]] = None,
                    ridge_putt: bool = False,
                    output_path: Optional[str] = None) -> Tuple[str, List]:
        """
        Generate SVG string for a putt illustration with AimPoint annotations.
        """
        # Validate and clamp inputs
        if break_direction not in ["left", "right"]:
            raise ValueError("break_direction must be 'left' or 'right'")
        
        # Clamp slope to realistic putting green range:
        #   0.5% minimum to avoid degenerate cases
        #   5.0% maximum — real putting greens rarely exceed 3-4%
        #   (USGA design standards: 1-3% typical, 4% challenging, 5% absolute max)
        slope_percent = max(0.5, min(5.0, slope_percent))
        
        # Create configuration
        config = PuttConfig(
            with_grain=with_grain,
            uphill=uphill,
            break_direction=BreakDirection(break_direction),
            slope_percent=slope_percent,
            break_change_points=break_change_points or [],
            ridge_putt=ridge_putt
        )
        
        # Get sections
        sections = config.get_sections()
        
        # Calculate physics trajectory (in normalized coordinates)
        physics = PuttingPhysics(config)
        normalized_trajectory = physics.compute_trajectory()
        
        # Convert normalized trajectory to pixel coordinates
        r = self.renderer
        trajectory = self._trajectory_to_pixels(normalized_trajectory, self.layout)
        
        # Generate path and calculate aim point (where you aim at the start)
        putt_path = SVGPathGenerator.points_to_svg_path(trajectory)
        apex_point = SVGPathGenerator.calculate_aim_point(trajectory, r.hole_y, r.center_x)
        
        # Calculate AimPoint fingers from ACTUAL aim offset
        # 1 finger ≈ 1 inch ≈ ~4.5 pixels (hole is ~4.25" = 18px diameter)
        aim_offset_pixels = abs(apex_point[0] - r.center_x)
        aim_fingers = int(aim_offset_pixels / 4.5 + 0.5)  # Round to nearest finger
        
        # Build section_slopes list for compatibility
        section_slopes = [
            (s.section_number, s.slope_percent, s.description())
            for s in sections
        ]
        
        # Generate SVG
        svg = self._build_svg(config, sections, putt_path, apex_point, aim_fingers)
        
        # Save if output path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(svg)
        
        return svg, section_slopes
    
    @staticmethod
    def _trajectory_to_pixels(
        normalized: List[Tuple[float, float]], layout: LayoutConfig
    ) -> List[Tuple[float, float]]:
        """
        Convert normalized trajectory to pixel coordinates.
        
        Args:
            normalized: List of (t_param, x_offset) from physics engine
            layout: Layout configuration with canvas dimensions
            
        Returns:
            List of (px, py) pixel coordinates
        """
        margin = 10  # Pixel margin to keep path within green bounds
        
        points = []
        for t_param, x_offset in normalized:
            px = layout.center_x + x_offset
            py = layout.ball_y - t_param * layout.total_distance_pixels
            
            # Clamp x to stay within green bounds
            px = max(margin, min(layout.width - margin, px))
            points.append((px, py))
        
        # Ensure final point is exactly at hole center
        points[-1] = (layout.center_x, layout.hole_y)
        return points
    
    def _build_svg(self, config: PuttConfig, sections: List[SlopeSection],
                   putt_path: str, apex_point: Tuple[float, float], aim_fingers: int) -> str:
        """Assemble the complete SVG document."""
        
        r = self.renderer
        L = self.layout
        C = Colors
        
        return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{L.total_width}" height="{L.height}" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <!-- Base green gradient -->
        <linearGradient id="baseGreen" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" style="stop-color:{C.GREEN_MEDIUM};stop-opacity:1" />
            <stop offset="50%" style="stop-color:{C.GREEN_BASE_DARK};stop-opacity:1" />
            <stop offset="100%" style="stop-color:{C.GREEN_MED_DARK};stop-opacity:1" />
        </linearGradient>
        
        {r.render_gradient_defs(config, sections)}
        
        <filter id="shadow">
            <feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.3"/>
        </filter>
        
        <!-- Arrow marker for grain direction -->
        <marker id="arrowhead-grain" markerWidth="10" markerHeight="10" refX="5" refY="5" orient="auto">
            <polygon points="0,0 10,5 0,10" fill="{C.GREEN_LIGHT}" opacity="0.7" />
        </marker>
        
        <!-- Grass texture pattern -->
        <pattern id="grassTexture" x="0" y="0" width="4" height="4" patternUnits="userSpaceOnUse">
            <rect width="4" height="4" fill="url(#baseGreen)" opacity="0.95"/>
            <line x1="0" y1="2" x2="4" y2="2" stroke="{C.GREEN_TEXTURE}" stroke-width="0.5" opacity="0.3"/>
        </pattern>
    </defs>
    
    <!-- Base green background with texture -->
    <rect width="{L.width}" height="{L.height}" fill="url(#baseGreen)" />
    <rect width="{L.width}" height="{L.height}" fill="url(#grassTexture)" opacity="0.3" />
    
    <!-- Apply elevation shading (uphill/downhill effect) -->
    <rect width="{L.width}" height="{L.height}" fill="url(#elevationGradient)" opacity="0.7" />
    
    <!-- Apply break shading (side-to-side slope effect) - section by section -->
    {r.render_break_shading_rects(sections)}
    
    <!-- Apply grain shading (direction of grass blades) -->
    <rect width="{L.width}" height="{L.height}" fill="url(#grainGradient)" opacity="0.6" />
    
    <!-- Visual grain direction indicators -->
    {r.render_grain_indicators(config.with_grain)}
    
    <!-- Target line (straight line to hole) -->
    <line x1="{L.center_x}" 
          y1="{L.ball_y}" 
          x2="{L.center_x}" 
          y2="{L.hole_y}" 
          stroke="{C.WHITE}" 
          stroke-width="1.5" 
          stroke-dasharray="3,3"
          opacity="0.5" />
    
    <!-- Putt path (break line) -->
    <path d="{putt_path}" 
          stroke="{C.GOLD}" 
          stroke-width="3" 
          fill="none" 
          stroke-dasharray="5,3"
          opacity="0.95" />
    
    <!-- Section dividers with break info (rendered last to cover lines) -->
    {r.render_section_dividers(sections)}
    
    <!-- AimPoint marker - pink circle showing starting aim line (~2ft from ball) -->
    <circle cx="{apex_point[0]}" 
            cy="{L.height - 138}" 
            r="4" 
            fill="{C.PINK}" 
            opacity="0.9" 
            stroke="{C.WHITE}" 
            stroke-width="1" />
    
    <!-- AimPoint marker - pink circle showing aim point near hole (same x position) -->
    <circle cx="{apex_point[0]}" 
            cy="50" 
            r="4" 
            fill="{C.PINK}" 
            opacity="0.9" 
            stroke="{C.WHITE}" 
            stroke-width="1" />
    
    <!-- AimPoint display - above hole -->
    <text x="{L.center_x}" y="12" font-family="Arial, sans-serif" font-size="11" fill="{C.GOLD}" text-anchor="middle" font-weight="bold">AimPoint: {aim_fingers}</text>
    
    <!-- Hole with grain shading on edge -->
    <circle cx="{L.center_x}" 
            cy="{L.hole_y}" 
            r="9" 
            fill="{C.NEAR_BLACK}" 
            filter="url(#shadow)" />
    <circle cx="{L.center_x}" 
            cy="{L.hole_y}" 
            r="7" 
            fill="{C.DARK_GRAY}" />
    <circle cx="{L.center_x}" 
            cy="{L.hole_y}" 
            r="8.5" 
            fill="none" 
            stroke="url(#holeGrainGradient)" 
            stroke-width="1.5"
            opacity="0.8" />
    
    <!-- Ball starting position -->
    <circle cx="{L.center_x}" 
            cy="{L.ball_y}" 
            r="6" 
            fill="{C.WHITE}" 
            stroke="{C.LIGHT_GRAY}" 
            stroke-width="1"
            filter="url(#shadow)" />
    <circle cx="{L.center_x}" 
            cy="{L.ball_y}" 
            r="4" 
            fill="{C.OFF_WHITE}" />
    
    <!-- Ball label -->
    <text x="{L.center_x}" y="{L.height - 15}" font-family="Arial, sans-serif" font-size="9" fill="{C.WHITE}" text-anchor="middle" font-weight="bold" opacity="0.95">START (BALL)</text>
    
    <!-- Right column for explanation text -->
    <rect x="{L.width}" y="0" width="{L.text_column_width}" height="{L.height}" fill="{C.NEAR_BLACK}" opacity="0.95" />
    <rect x="{L.width + 3}" y="3" width="{L.text_column_width - 6}" height="{L.height - 6}" fill="{C.BLACK}" opacity="0.9" rx="4" stroke="{C.GOLD}" stroke-width="2" />
    {r.render_explanation_text(config, sections, aim_fingers)}
</svg>'''


# =============================================================================
# VALIDATION
# =============================================================================

def validate_svg(svg_content: str, break_direction: str, slope_percent: float,
                 break_change_points: List, ridge_putt: bool, section_slopes: List) -> List[str]:
    """
    Validate SVG against all rules. Returns list of error messages (empty if valid).
    """
    errors = []
    
    # Parse section slopes
    for section_num, section_slope, section_explanation in section_slopes:
        abs_slope = abs(section_slope)
        
        # Determine expected direction
        if ridge_putt and section_slope < 0:
            section_breaks_right = (break_direction == "left")
        else:
            section_breaks_right = (break_direction == "right")
        
        expected_dir = "L to R" if section_breaks_right else "R to L"
        
        # Extract actual direction from explanation
        actual_dir = None
        if "L to R" in section_explanation:
            actual_dir = "L to R"
        elif "R to L" in section_explanation:
            actual_dir = "R to L"
        
        if actual_dir != expected_dir:
            errors.append(f"Section {section_num}: Direction mismatch. Expected '{expected_dir}', got '{actual_dir}'")
    
    # Validate path starts at ball
    path_match = '<path d="M '
    if path_match in svg_content:
        path_start = svg_content.find(path_match) + len(path_match)
        path_end = svg_content.find('"', path_start)
        path_data = svg_content[path_start:path_end]
        
        parts = path_data.replace(',', ' ').split()
        if len(parts) >= 2:
            try:
                path_start_x = float(parts[0])
                path_start_y = float(parts[1])
                
                ball_x = 60.0  # Center
                ball_y = 570.0  # Bottom
                
                if abs(path_start_x - ball_x) > 0.5:
                    errors.append(f"Path must start at ball x={ball_x}, got {path_start_x:.1f}")
                if abs(path_start_y - ball_y) > 0.5:
                    errors.append(f"Path must start at ball y={ball_y}, got {path_start_y:.1f}")
            except ValueError:
                pass
    
    return errors


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Generate comprehensive set of realistic 10ft putt scenarios (1-3% slopes)."""
    generator = PuttIllustrationGenerator(width=120, height=600, text_column_width=120)
    
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Create AimPoint category folders
    aimpoint_dirs = {
        1: output_dir / "aimpoint_1",
        2: output_dir / "aimpoint_2", 
        3: output_dir / "aimpoint_3",
    }
    for d in aimpoint_dirs.values():
        d.mkdir(exist_ok=True)
    
    # Original scenarios (to main output folder)
    scenarios = [
        # SIMPLE BREAKS
        ("01_simple_right_with_grain_downhill", {
            "with_grain": True, "uphill": False, "break_direction": "right",
            "slope_percent": 1.5, "description": "Simple right break, with grain, downhill"
        }),
        ("02_simple_left_against_grain_uphill", {
            "with_grain": False, "uphill": True, "break_direction": "left",
            "slope_percent": 1.8, "description": "Simple left break, against grain, uphill"
        }),
        ("03_simple_right_with_grain_flat", {
            "with_grain": True, "uphill": False, "break_direction": "right",
            "slope_percent": 1.2, "description": "Simple right break, with grain, flat"
        }),
        ("04_simple_left_strong_against_grain", {
            "with_grain": False, "uphill": False, "break_direction": "left",
            "slope_percent": 2.5, "description": "Strong left break, against grain"
        }),
        ("05_simple_right_strong_with_grain_uphill", {
            "with_grain": True, "uphill": True, "break_direction": "right",
            "slope_percent": 2.8, "description": "Strong right break, with grain, uphill"
        }),
        ("06_simple_left_gentle_against_grain_downhill", {
            "with_grain": False, "uphill": False, "break_direction": "left",
            "slope_percent": 1.0, "description": "Gentle left break, against grain, downhill"
        }),
        
        # DOUBLE BREAKS
        ("07_double_right_increasing_with_grain", {
            "with_grain": True, "uphill": False, "break_direction": "right",
            "slope_percent": 1.0, "break_change_points": [(5.0, 2.5)],
            "description": "Double break - right, increasing from 1% to 2.5%, with grain"
        }),
        ("08_double_left_increasing_against_grain_uphill", {
            "with_grain": False, "uphill": True, "break_direction": "left",
            "slope_percent": 0.8, "break_change_points": [(4.0, 2.2)],
            "description": "Double break - left, increasing from 0.8% to 2.2%, against grain, uphill"
        }),
        ("09_double_right_increasing_near_hole", {
            "with_grain": True, "uphill": False, "break_direction": "right",
            "slope_percent": 1.2, "break_change_points": [(7.0, 2.8)],
            "description": "Double break - right, increases near hole from 1.2% to 2.8%"
        }),
        ("10_double_left_decreasing_with_grain", {
            "with_grain": True, "uphill": False, "break_direction": "left",
            "slope_percent": 2.5, "break_change_points": [(5.0, 1.2)],
            "description": "Double break - left, decreasing from 2.5% to 1.2%, with grain"
        }),
        ("11_double_right_decreasing_against_grain_uphill", {
            "with_grain": False, "uphill": True, "break_direction": "right",
            "slope_percent": 2.8, "break_change_points": [(4.0, 1.0)],
            "description": "Double break - right, decreasing from 2.8% to 1.0%, against grain, uphill"
        }),
        
        # RIDGE PUTTS
        ("12_ridge_right_reverses_with_grain", {
            "with_grain": True, "uphill": False, "break_direction": "right",
            "slope_percent": 2.0, "break_change_points": [(4.0, -1.8)], "ridge_putt": True,
            "description": "Ridge putt - right break reverses to left 1.8% at 4ft, with grain"
        }),
        ("13_ridge_left_reverses_against_grain", {
            "with_grain": False, "uphill": False, "break_direction": "left",
            "slope_percent": 1.5, "break_change_points": [(5.0, -2.0)], "ridge_putt": True,
            "description": "Ridge putt - left break reverses to right 2.0% at 5ft, against grain"
        }),
        ("14_ridge_right_reverses_uphill", {
            "with_grain": True, "uphill": True, "break_direction": "right",
            "slope_percent": 2.2, "break_change_points": [(3.5, -1.5)], "ridge_putt": True,
            "description": "Ridge putt - right break reverses to left 1.5% at 3.5ft, uphill"
        }),
        
        # COMPLEX BREAKS
        ("15_triple_right_complex_with_grain", {
            "with_grain": True, "uphill": False, "break_direction": "right",
            "slope_percent": 0.8, "break_change_points": [(3.0, 2.2), (7.0, 1.0)],
            "description": "Complex triple break - starts 0.8%, increases to 2.2% at 3ft, decreases to 1.0% at 7ft"
        }),
        ("16_triple_left_complex_against_grain", {
            "with_grain": False, "uphill": False, "break_direction": "left",
            "slope_percent": 1.0, "break_change_points": [(2.5, 2.5), (6.5, 1.5)],
            "description": "Complex triple break - starts 1.0%, increases to 2.5% at 2.5ft, decreases to 1.5% at 6.5ft"
        }),
        ("17_triple_right_complex_uphill", {
            "with_grain": True, "uphill": True, "break_direction": "right",
            "slope_percent": 1.2, "break_change_points": [(4.0, 2.8), (8.0, 1.8)],
            "description": "Complex triple break uphill - starts 1.2%, peaks at 2.8%, ends at 1.8%"
        }),
    ]
    
    # =========================================================================
    # AIMPOINT CATEGORIZED SCENARIOS
    # Tuned to produce specific AimPoint finger readings
    # =========================================================================
    
    # AIMPOINT 1 (gentle breaks ~1.0-1.5%)
    aimpoint_1_scenarios = [
        # Simple breaks
        ("ap1_gentle_right_downhill", {
            "with_grain": True, "uphill": False, "break_direction": "right",
            "slope_percent": 1.2, "description": "AimPoint 1: Gentle right break, with grain, downhill"
        }),
        ("ap1_gentle_left_flat", {
            "with_grain": True, "uphill": False, "break_direction": "left",
            "slope_percent": 1.3, "description": "AimPoint 1: Gentle left break, with grain, flat"
        }),
        ("ap1_gentle_right_uphill", {
            "with_grain": False, "uphill": True, "break_direction": "right",
            "slope_percent": 1.5, "description": "AimPoint 1: Gentle right break, against grain, uphill"
        }),
        ("ap1_gentle_left_against_grain", {
            "with_grain": False, "uphill": False, "break_direction": "left",
            "slope_percent": 1.2, "description": "AimPoint 1: Gentle left break, against grain"
        }),
        # Double breaks
        ("ap1_double_right_slight", {
            "with_grain": True, "uphill": False, "break_direction": "right",
            "slope_percent": 1.0, "break_change_points": [(5.0, 1.5)],
            "description": "AimPoint 1: Double break, slight right increase"
        }),
        ("ap1_double_left_decreasing", {
            "with_grain": False, "uphill": True, "break_direction": "left",
            "slope_percent": 1.8, "break_change_points": [(6.0, 1.0)],
            "description": "AimPoint 1: Double break left, decreasing uphill"
        }),
        # Double breaks with REVERSAL (direction change mid-putt)
        ("ap1_double_right_to_left", {
            "with_grain": True, "uphill": False, "break_direction": "right",
            "slope_percent": 2.0, "break_change_points": [(5.0, -0.8)], "ridge_putt": True,
            "description": "AimPoint 1: Double break - R->L reversal at 5ft"
        }),
        ("ap1_double_left_to_right", {
            "with_grain": False, "uphill": False, "break_direction": "left",
            "slope_percent": 1.8, "break_change_points": [(5.0, -0.6)], "ridge_putt": True,
            "description": "AimPoint 1: Double break - L->R reversal at 5ft"
        }),
        # Triple breaks with 2 REVERSALS (R->L->R or L->R->L)
        ("ap1_triple_right_left_right", {
            "with_grain": True, "uphill": False, "break_direction": "right",
            "slope_percent": 1.5, "break_change_points": [(3.5, -0.8), (7.0, 0.6)], "ridge_putt": True,
            "description": "AimPoint 1: Triple - R->L->R (2 reversals)"
        }),
        ("ap1_triple_left_right_left", {
            "with_grain": False, "uphill": False, "break_direction": "left",
            "slope_percent": 1.3, "break_change_points": [(4.0, -0.7), (7.5, 0.5)], "ridge_putt": True,
            "description": "AimPoint 1: Triple - L->R->L (2 reversals)"
        }),
    ]
    
    # AIMPOINT 2 (moderate breaks ~2.0-2.5%)
    aimpoint_2_scenarios = [
        # Simple breaks
        ("ap2_moderate_right_downhill", {
            "with_grain": True, "uphill": False, "break_direction": "right",
            "slope_percent": 2.0, "description": "AimPoint 2: Moderate right break, with grain, downhill"
        }),
        ("ap2_moderate_left_flat", {
            "with_grain": True, "uphill": False, "break_direction": "left",
            "slope_percent": 2.2, "description": "AimPoint 2: Moderate left break, with grain"
        }),
        ("ap2_moderate_right_against_grain", {
            "with_grain": False, "uphill": False, "break_direction": "right",
            "slope_percent": 2.3, "description": "AimPoint 2: Moderate right break, against grain"
        }),
        ("ap2_moderate_left_uphill", {
            "with_grain": True, "uphill": True, "break_direction": "left",
            "slope_percent": 2.5, "description": "AimPoint 2: Moderate left break, with grain, uphill"
        }),
        # Double breaks
        ("ap2_double_right_building", {
            "with_grain": True, "uphill": False, "break_direction": "right",
            "slope_percent": 1.5, "break_change_points": [(5.0, 2.5)],
            "description": "AimPoint 2: Double break right, building slope"
        }),
        ("ap2_double_left_consistent", {
            "with_grain": False, "uphill": False, "break_direction": "left",
            "slope_percent": 2.0, "break_change_points": [(5.0, 2.3)],
            "description": "AimPoint 2: Double break left, consistent moderate"
        }),
        # Double breaks with REVERSAL - STRONG asymmetry for AimPoint 2
        ("ap2_double_right_to_left", {
            "with_grain": True, "uphill": False, "break_direction": "right",
            "slope_percent": 4.5, "break_change_points": [(7.0, -0.5)], "ridge_putt": True,
            "description": "AimPoint 2: Double R->L - strong initial, late tiny reversal"
        }),
        ("ap2_double_left_to_right", {
            "with_grain": True, "uphill": False, "break_direction": "left",
            "slope_percent": 4.2, "break_change_points": [(7.0, -0.4)], "ridge_putt": True,
            "description": "AimPoint 2: Double L->R - strong initial, late tiny reversal"
        }),
        # Triple breaks with 2 REVERSALS - asymmetric for AimPoint 2
        ("ap2_triple_right_left_right", {
            "with_grain": True, "uphill": False, "break_direction": "right",
            "slope_percent": 4.0, "break_change_points": [(5.0, -0.5), (8.0, 0.3)], "ridge_putt": True,
            "description": "AimPoint 2: Triple R->L->R - dominant right"
        }),
        ("ap2_triple_left_right_left", {
            "with_grain": True, "uphill": False, "break_direction": "left",
            "slope_percent": 3.8, "break_change_points": [(5.0, -0.4), (8.0, 0.2)], "ridge_putt": True,
            "description": "AimPoint 2: Triple L->R->L - dominant left"
        }),
    ]
    
    # AIMPOINT 3 (strong breaks ~3.0-5.0%)
    aimpoint_3_scenarios = [
        # Simple breaks
        ("ap3_strong_right_downhill", {
            "with_grain": True, "uphill": False, "break_direction": "right",
            "slope_percent": 3.2, "description": "AimPoint 3: Strong right break, with grain, downhill"
        }),
        ("ap3_strong_left_flat", {
            "with_grain": True, "uphill": False, "break_direction": "left",
            "slope_percent": 3.5, "description": "AimPoint 3: Strong left break, with grain"
        }),
        ("ap3_strong_right_against_grain", {
            "with_grain": False, "uphill": False, "break_direction": "right",
            "slope_percent": 3.8, "description": "AimPoint 3: Strong right break, against grain"
        }),
        ("ap3_strong_left_uphill", {
            "with_grain": False, "uphill": True, "break_direction": "left",
            "slope_percent": 4.0, "description": "AimPoint 3: Strong left break, against grain, uphill"
        }),
        # Double breaks
        ("ap3_double_right_steep", {
            "with_grain": True, "uphill": False, "break_direction": "right",
            "slope_percent": 2.5, "break_change_points": [(4.0, 4.0)],
            "description": "AimPoint 3: Double break right, steep increase"
        }),
        ("ap3_double_left_heavy", {
            "with_grain": True, "uphill": False, "break_direction": "left",
            "slope_percent": 3.0, "break_change_points": [(5.0, 3.5)],
            "description": "AimPoint 3: Double break left, heavy consistent"
        }),
        # Double breaks with REVERSAL - steep but realistic for AimPoint 3
        ("ap3_double_right_to_left", {
            "with_grain": True, "uphill": False, "break_direction": "right",
            "slope_percent": 5.0, "break_change_points": [(8.5, -0.3)], "ridge_putt": True,
            "description": "AimPoint 3: Double R->L - 5% steep initial, small late reversal"
        }),
        ("ap3_double_left_to_right", {
            "with_grain": True, "uphill": False, "break_direction": "left",
            "slope_percent": 4.8, "break_change_points": [(8.5, -0.3)], "ridge_putt": True,
            "description": "AimPoint 3: Double L->R - 4.8% steep initial, small late reversal"
        }),
        # Triple breaks with 2 REVERSALS - realistic slopes for AimPoint 3
        ("ap3_triple_right_left_right", {
            "with_grain": True, "uphill": False, "break_direction": "right",
            "slope_percent": 5.0, "break_change_points": [(7.0, -0.3), (9.0, 0.2)], "ridge_putt": True,
            "description": "AimPoint 3: Triple R->L->R - 5% dominant with small reversals"
        }),
        ("ap3_triple_left_right_left", {
            "with_grain": True, "uphill": False, "break_direction": "left",
            "slope_percent": 4.8, "break_change_points": [(7.0, -0.3), (9.0, 0.2)], "ridge_putt": True,
            "description": "AimPoint 3: Triple L->R->L - 4.8% dominant with small reversals"
        }),
        # Triple breaks
        ("ap3_triple_right_aggressive", {
            "with_grain": True, "uphill": False, "break_direction": "right",
            "slope_percent": 2.5, "break_change_points": [(3.0, 4.0), (7.0, 3.2)],
            "description": "AimPoint 3: Triple break - aggressive right with high peak"
        }),
        ("ap3_triple_left_intense", {
            "with_grain": False, "uphill": False, "break_direction": "left",
            "slope_percent": 2.8, "break_change_points": [(3.5, 4.2), (7.0, 3.5)],
            "description": "AimPoint 3: Triple break - intense left variations"
        }),
    ]
    
    print("=" * 80)
    print("GENERATING PUTT ILLUSTRATIONS WITH REFACTORED PHYSICS ENGINE")
    print("=" * 80)
    
    import re
    
    successful_count = 0
    validation_errors = []
    aimpoint_counts = {0: 0, 1: 0, 2: 0, 3: 0, "4+": 0}
    
    def extract_aimpoint_from_svg(svg_content: str) -> int:
        """Extract the actual AimPoint value from generated SVG."""
        match = re.search(r'AimPoint: (\d+)', svg_content)
        if match:
            return int(match.group(1))
        return 0
    
    def generate_main_scenarios(scenario_list, target_dir):
        """Generate main scenarios to a fixed directory."""
        nonlocal successful_count, validation_errors
        
        for idx, (filename, params) in enumerate(scenario_list, 1):
            output_path = str(target_dir / f"{filename}.svg")
            
            print(f"\n[MAIN] [{idx}/{len(scenario_list)}] Generating: {filename}")
            print(f"   Description: {params.get('description', 'N/A')}")
            print(f"   Parameters: break={params['break_direction']}, slope={params['slope_percent']}%")
            
            if params.get('break_change_points'):
                print(f"   Break changes: {params['break_change_points']}")
            if params.get('ridge_putt'):
                print(f"   Ridge putt: Yes")
            
            try:
                svg_content, section_slopes = generator.generate_svg(
                    with_grain=params["with_grain"],
                    uphill=params["uphill"],
                    break_direction=params["break_direction"],
                    slope_percent=params["slope_percent"],
                    break_change_points=params.get("break_change_points"),
                    ridge_putt=params.get("ridge_putt", False),
                    output_path=output_path
                )
                
                aimpoint_val = extract_aimpoint_from_svg(svg_content)
                print(f"   [OK] SVG generated (AimPoint: {aimpoint_val})")
                
                errors = validate_svg(
                    svg_content, params["break_direction"], params["slope_percent"],
                    params.get("break_change_points", []), params.get("ridge_putt", False),
                    section_slopes
                )
                
                if errors:
                    validation_errors.append((filename, errors))
                    print(f"\n   [FAIL] VALIDATION FAILED for {filename}:")
                    for error in errors:
                        print(f"      - {error}")
                else:
                    successful_count += 1
                    print(f"   [OK] Saved to: {output_path}")
                    
            except Exception as e:
                print(f"\n   [ERROR] ERROR generating {filename}: {e}")
                import traceback
                traceback.print_exc()
        
        return len(scenario_list)
    
    def generate_and_classify_scenarios(scenario_list, label=""):
        """Generate scenarios and classify by ACTUAL AimPoint value."""
        nonlocal successful_count, validation_errors, aimpoint_counts
        
        for idx, (filename, params) in enumerate(scenario_list, 1):
            prefix = f"[{label}] " if label else ""
            print(f"\n{prefix}[{idx}/{len(scenario_list)}] Generating: {filename}")
            print(f"   Description: {params.get('description', 'N/A')}")
            print(f"   Parameters: break={params['break_direction']}, slope={params['slope_percent']}%")
            
            if params.get('break_change_points'):
                print(f"   Break changes: {params['break_change_points']}")
            if params.get('ridge_putt'):
                print(f"   Ridge putt: Yes")
            
            try:
                # Generate WITHOUT saving first to get actual AimPoint
                svg_content, section_slopes = generator.generate_svg(
                    with_grain=params["with_grain"],
                    uphill=params["uphill"],
                    break_direction=params["break_direction"],
                    slope_percent=params["slope_percent"],
                    break_change_points=params.get("break_change_points"),
                    ridge_putt=params.get("ridge_putt", False),
                    output_path=None  # Don't save yet
                )
                
                # Extract actual AimPoint value
                actual_aimpoint = extract_aimpoint_from_svg(svg_content)
                
                # Classify into correct folder based on ACTUAL value
                if actual_aimpoint == 0:
                    # Skip AimPoint 0 - not useful for training
                    print(f"   [SKIP] AimPoint: 0 - skipping (breaks cancel out)")
                    aimpoint_counts[0] += 1
                    continue
                elif actual_aimpoint == 1:
                    target_folder = aimpoint_dirs[1]
                    aimpoint_counts[1] += 1
                elif actual_aimpoint == 2:
                    target_folder = aimpoint_dirs[2]
                    aimpoint_counts[2] += 1
                elif actual_aimpoint == 3:
                    target_folder = aimpoint_dirs[3]
                    aimpoint_counts[3] += 1
                else:
                    # AimPoint 4+ goes to aimpoint_3 folder
                    target_folder = aimpoint_dirs[3]
                    aimpoint_counts["4+"] += 1
                
                # Save to classified folder
                output_path = str(target_folder / f"{filename}.svg")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(svg_content)
                
                print(f"   [OK] Actual AimPoint: {actual_aimpoint} -> Saved to: aimpoint_{min(actual_aimpoint, 3)}/")
                
                errors = validate_svg(
                    svg_content, params["break_direction"], params["slope_percent"],
                    params.get("break_change_points", []), params.get("ridge_putt", False),
                    section_slopes
                )
                
                if errors:
                    validation_errors.append((filename, errors))
                    print(f"   [WARN] Validation issues: {errors}")
                else:
                    successful_count += 1
                    
            except Exception as e:
                print(f"\n   [ERROR] ERROR generating {filename}: {e}")
                import traceback
                traceback.print_exc()
        
        return len(scenario_list)
    
    # Generate main scenarios (fixed location)
    print("\n" + "-" * 40)
    print("MAIN SCENARIOS")
    print("-" * 40)
    total_main = generate_main_scenarios(scenarios, output_dir)
    
    # Combine all AimPoint scenarios into one pool for classification
    all_aimpoint_scenarios = (
        aimpoint_1_scenarios + 
        aimpoint_2_scenarios + 
        aimpoint_3_scenarios
    )
    
    # Generate and classify by ACTUAL AimPoint value
    print("\n" + "-" * 40)
    print("AIMPOINT SCENARIOS (Classifying by ACTUAL value)")
    print("-" * 40)
    total_ap = generate_and_classify_scenarios(all_aimpoint_scenarios, "AP")
    
    total_all = total_main + total_ap - aimpoint_counts[0]  # Subtract skipped
    
    print("\n" + "=" * 80)
    print(f"GENERATION SUMMARY")
    print("=" * 80)
    print(f"Main scenarios: {total_main}")
    print(f"\nAimPoint Classification (by ACTUAL calculated value):")
    print(f"  AimPoint 0 (skipped): {aimpoint_counts[0]}")
    print(f"  AimPoint 1: {aimpoint_counts[1]}")
    print(f"  AimPoint 2: {aimpoint_counts[2]}")
    print(f"  AimPoint 3: {aimpoint_counts[3]}")
    print(f"  AimPoint 4+: {aimpoint_counts['4+']} (saved to aimpoint_3/)")
    print(f"\nTotal generated and validated: {successful_count}/{total_all}")
    
    if validation_errors:
        print(f"\n[WARNING] VALIDATION ERRORS: {len(validation_errors)} file(s) failed validation")
    elif successful_count == total_all:
        print("\n[SUCCESS] All files generated and validated successfully!")
    
    print(f"\nOutput folders:")
    print(f"  Main: {output_dir}")
    print(f"  AimPoint 1: {aimpoint_dirs[1]}")
    print(f"  AimPoint 2: {aimpoint_dirs[2]}")
    print(f"  AimPoint 3: {aimpoint_dirs[3]}")


if __name__ == "__main__":
    main()
