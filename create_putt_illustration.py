"""
Generate SVG illustrations of putt conditions for green reading practice.
Based on AimPoint green reading methodology with realistic putting green slopes
(typically 1-3%, up to 5% absolute max for playable surfaces).
Creates 120x600 pixel images with detailed annotations.

Refactored to use clean physics model with scipy ODE integration.
"""

import math
import re
import traceback
import warnings
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
    height: int = 660
    text_column_width: int = 200
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
    
    # --- Elevation indicator colors ---
    ELEV_UPHILL = "#FFA726"        # Warm amber — uphill triangles
    ELEV_DOWNHILL = "#4FC3F7"      # Cool light blue — downhill triangles
    
    # --- Neutral tones ---
    WHITE = "#FFFFFF"
    OFF_WHITE = "#F5F5F5"
    LIGHT_GRAY = "#CCCCCC"
    MED_GRAY = "#999999"
    DARK_GRAY = "#2C2C2C"
    NEAR_BLACK = "#1A1A1A"
    BLACK = "#000000"


# Elevation adjustment: scales effective break based on how steep the
# uphill/downhill grade is.  At 0% elevation -> factor 1.0 (no change).
# Each 1% of uphill reduces effective break by ~5%, each 1% downhill
# increases it by ~5%.  Capped so the factor stays within [0.70, 1.30].
ELEVATION_FACTOR_PER_PERCENT = 0.05

def elevation_factor(elevation_pct: float) -> float:
    """Return the break-adjustment factor for a given elevation percent.
    
    Positive elevation_pct = uphill -> factor < 1 (less break).
    Negative elevation_pct = downhill -> factor > 1 (more break).
    """
    raw = 1.0 - elevation_pct * ELEVATION_FACTOR_PER_PERCENT
    return max(0.70, min(1.30, raw))


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
    """Configuration for a putt scenario.
    
    elevation_percent: positive = uphill, negative = downhill, 0 = flat.
    Typical range: -3.0 to +3.0 (putting greens rarely exceed 3% grade).
    """
    with_grain: bool
    elevation_percent: float  # +uphill / −downhill / 0 flat
    break_direction: BreakDirection
    slope_percent: float
    break_change_points: List[Tuple[float, float]] = field(default_factory=list)
    ridge_putt: bool = False
    putt_length_ft: float = 10.0
    
    @property
    def uphill(self) -> bool:
        """Backward-compatible property: True if elevation > 0."""
        return self.elevation_percent > 0
    
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
    
    # Visual break scaling
    BASE_BREAK_SCALE = 40.0    # Pixels of lateral deviation per slope-percent-squared
    # Elevation break scaling uses elevation_factor() based on config.elevation_percent
    
    # Newton-Raphson trajectory solver
    NR_MAX_ITERATIONS = 5       # Max iterations to find initial velocity
    NR_CONVERGENCE_THRESHOLD = 0.1  # Acceptable final x offset (pixels)
    NR_CORRECTION_FACTOR = 0.8  # Damping factor for velocity correction
    
    # Trajectory sampling
    TRAJECTORY_POINTS = 80      # Number of points to sample along the path
    
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
        
        # Elevation affects break amount:
        # - Uphill: ball has more speed fighting gravity -> less break
        # - Downhill: ball is slower near hole -> more time for slope to act
        elev_fac = elevation_factor(self.config.elevation_percent)
        BREAK_SCALE = self.BASE_BREAK_SCALE * elev_fac
        
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
        converged = False
        for _ in range(self.NR_MAX_ITERATIONS):
            final_x = simulate_with_v0(initial_velocity)
            if abs(final_x) < self.NR_CONVERGENCE_THRESHOLD:
                converged = True
                break
            # Adjust v0: if ball ends left (negative), need more rightward initial velocity
            initial_velocity -= final_x * self.NR_CORRECTION_FACTOR
        
        if not converged:
            warnings.warn(
                f"Trajectory solver did not converge after {self.NR_MAX_ITERATIONS} "
                f"iterations (final offset: {final_x:.2f}px). "
                f"Slope config: {self.config.slope_percent}%, "
                f"direction: {self.config.break_direction.value}",
                RuntimeWarning,
                stacklevel=2,
            )
        
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
        num_points = self.TRAJECTORY_POINTS
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
        {self._elevation_gradient(config.elevation_percent)}
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
    
    def _elevation_gradient(self, elevation_percent: float) -> str:
        """Create elevation shading gradient, scaled by elevation magnitude."""
        C = Colors
        # Scale opacity by how steep the elevation is (0% = faint, 3% = full)
        intensity = min(abs(elevation_percent) / 3.0, 1.0)
        lo = 0.15 + 0.15 * intensity   # faint end: 0.15–0.30
        hi = 0.5 + 0.5 * intensity     # strong end: 0.50–1.00
        mid = (lo + hi) / 2
        
        if elevation_percent > 0:
            # Uphill: dark at bottom (ball), lighter at top (hole is higher)
            return f'''<linearGradient id="elevationGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" style="stop-color:{C.GREEN_LIGHT};stop-opacity:{lo:.2f}" />
            <stop offset="50%" style="stop-color:{C.GREEN_MEDIUM};stop-opacity:{mid:.2f}" />
            <stop offset="100%" style="stop-color:{C.GREEN_VERY_DARK};stop-opacity:{hi:.2f}" />
        </linearGradient>'''
        elif elevation_percent < 0:
            # Downhill: dark at top (hole is lower), lighter at bottom
            return f'''<linearGradient id="elevationGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" style="stop-color:{C.GREEN_VERY_DARK};stop-opacity:{hi:.2f}" />
            <stop offset="50%" style="stop-color:{C.GREEN_MEDIUM};stop-opacity:{mid:.2f}" />
            <stop offset="100%" style="stop-color:{C.GREEN_LIGHT};stop-opacity:{lo:.2f}" />
        </linearGradient>'''
        # Flat: very subtle uniform gradient
        return f'''<linearGradient id="elevationGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" style="stop-color:{C.GREEN_MEDIUM};stop-opacity:0.15" />
            <stop offset="100%" style="stop-color:{C.GREEN_MEDIUM};stop-opacity:0.15" />
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
    
    def render_elevation_indicators(self, elevation_percent: float) -> str:
        """Create visual triangles showing uphill/downhill elevation along left side of green.
        
        Args:
            elevation_percent: Positive = uphill, negative = downhill, 0 = flat.
        """
        C = Colors
        indicators = []
        x_pos = self.center_x - 20  # Offset left of center to avoid grain arrows
        spacing = 120  # Wider spacing than grain arrows (80) to keep it clean
        tri_size = 10  # Line length for marker attachment
        
        # Scale opacity with elevation magnitude (subtle at 0.5%, strong at 3%)
        opacity = min(0.5 + abs(elevation_percent) * 0.15, 0.95)
        
        if elevation_percent > 0:
            # Uphill: triangles pointing UP (toward hole)
            marker = "url(#arrowhead-elev-up)"
            color = C.ELEV_UPHILL
            for y_mid in range(140, int(self.height - 80), spacing):
                indicators.append(
                    f'<line x1="{x_pos}" y1="{y_mid + tri_size}" x2="{x_pos}" y2="{y_mid}" '
                    f'stroke="{color}" stroke-width="1.5" opacity="{opacity:.2f}" '
                    f'marker-end="{marker}" />'
                )
            label = f"▲ {abs(elevation_percent):.1f}%"
        elif elevation_percent < 0:
            # Downhill: triangles pointing DOWN (toward ball)
            marker = "url(#arrowhead-elev-down)"
            color = C.ELEV_DOWNHILL
            for y_mid in range(140, int(self.height - 80), spacing):
                indicators.append(
                    f'<line x1="{x_pos}" y1="{y_mid}" x2="{x_pos}" y2="{y_mid + tri_size}" '
                    f'stroke="{color}" stroke-width="1.5" opacity="{opacity:.2f}" '
                    f'marker-end="{marker}" />'
                )
            label = f"▼ {abs(elevation_percent):.1f}%"
        else:
            # Flat: just a label, no arrows
            color = C.MED_GRAY
            label = "FLAT"
        
        # Label near the top
        indicators.append(
            f'<text x="{x_pos}" y="125" font-family="Arial, sans-serif" font-size="6" '
            f'fill="{color}" text-anchor="middle" font-weight="bold" opacity="0.9">{label}</text>'
        )
        
        return '\n    '.join(indicators) if indicators else ''
    
    def render_explanation_text(self, config: PuttConfig, sections: List[SlopeSection], 
                               aim_fingers: int) -> str:
        """Generate AimPoint explanation text for right column.
        
        All text uses the full column width via _wrap_text.
        Font sizes: titles 12, headings 10, body 9, detail 8.
        """
        C = Colors
        x = self.width + 8            # left margin inside black column
        w = self.text_column_width - 16  # usable text width (both margins)
        
        # ----- compute values -----
        signed_sum = sum(
            (1.0 if s.breaks_right else -1.0) * abs(s.slope_percent)
            for s in sections
        )
        net_slope = signed_sum / len(sections)
        avg_slope = abs(net_slope)
        elev_fac = elevation_factor(config.elevation_percent)
        adj_slope = avg_slope * elev_fac
        direction_str = "L->R" if net_slope >= 0 else "R->L"
        breaks_right = (net_slope >= 0)
        ep = config.elevation_percent
        
        # ----- helpers -----
        def heading(y, text):
            return (y + 19, f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" '
                    f'font-size="11" fill="{C.GOLD}" font-weight="bold">{text}</text>')
        
        def body_line(y, text, color=C.WHITE, size=9, indent=0):
            return (y + 15, f'<text x="{x + indent}" y="{y}" font-family="Arial, sans-serif" '
                    f'font-size="{size}" fill="{color}">{text}</text>')
        
        def body_wrap(y, text, color=C.LIGHT_GRAY, size=9, indent=4):
            """Wrap long text to fit column, return (new_y, list_of_svg)."""
            lines = self._wrap_text(text, w - indent, size)
            svgs = []
            cy = y
            for ln in lines:
                svgs.append(f'<text x="{x + indent}" y="{cy}" font-family="Arial, sans-serif" '
                           f'font-size="{size}" fill="{color}">{ln}</text>')
                cy += 14
            return (cy, svgs)
        
        parts = []
        y = 22
        
        # ═══ TITLE ═══
        parts.append(f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" '
                     f'font-size="12" fill="{C.GOLD}" font-weight="bold">AIMPOINT ANALYSIS</text>')
        y += 22
        
        # ═══ SUMMARY ═══
        y, t = body_line(y, f'Slope (adj): {adj_slope:.1f}%  |  Dir: {direction_str}')
        parts.append(t)
        y, t = body_line(y, f'AimPoint: {aim_fingers} finger{"s" if aim_fingers != 1 else ""} ({direction_str} break)')
        parts.append(t)
        
        # elevation one-liner
        if ep > 0:
            elev_label = f"Elevation: UPHILL {ep:.1f}%"
        elif ep < 0:
            elev_label = f"Elevation: DOWNHILL {abs(ep):.1f}%"
        else:
            elev_label = "Elevation: FLAT (0%)"
        y, t = body_line(y, elev_label)
        parts.append(t)
        
        grain_label = "Grain: WITH (fast/shiny)" if config.with_grain else "Grain: AGAINST (slow/dark)"
        y, t = body_line(y, grain_label)
        parts.append(t)
        y += 2
        
        # ═══ SECTION SLOPES ═══
        y, t = heading(y, "SECTION SLOPES:")
        parts.append(t)
        for section in sections:
            slope_text = f'Sec {section.section_number}: {abs(section.slope_percent):.1f}% {section.direction_label}'
            y, t = body_line(y, slope_text, size=9)
            parts.append(t)
            desc = section.description()
            y, svgs = body_wrap(y, desc, size=8, indent=6)
            parts.extend(svgs)
            y += 1
        y += 2
        
        # ═══ GRAIN & BREAK ═══
        y, t = heading(y, "GRAIN &amp; BREAK:")
        parts.append(t)
        
        if config.with_grain:
            grain_text = (
                "WITH grain the ball rolls faster and looks shiny "
                "(you see blade tops reflecting light). The ball holds "
                "speed longer so it breaks LESS in the first half. "
                "But as it dies near the hole, break increases sharply "
                "- the slow-rolling ball curves hard at the end. "
                "Play MORE break than the slope suggests."
            )
        else:
            grain_text = (
                "AGAINST grain the ball rolls slower and looks dark/dull "
                "(you are looking into the blades). The ball loses speed "
                "faster so the slope pulls it sideways sooner. Break "
                "happens throughout the entire putt, not just at the end. "
                "Hit FIRMER to maintain speed and play LESS break "
                "than the slope suggests."
            )
        
        y, svgs = body_wrap(y, grain_text)
        parts.extend(svgs)
        y += 4
        
        # ═══ AIMPOINT FEEL ═══
        y, t = heading(y, "AIMPOINT FEEL:")
        parts.append(t)
        
        # AimPoint: heavy foot is on the LOW side (where the ball breaks toward)
        heavy_foot = "RIGHT" if breaks_right else "LEFT"
        low_side = "right" if breaks_right else "left"
        high_side = "left" if breaks_right else "right"
        
        if avg_slope < 1.0:
            feel_word, weight_word = "subtle", "slight"
        elif avg_slope < 2.5:
            feel_word, weight_word = "moderate", "clear"
        else:
            feel_word, weight_word = "strong", "heavy"
        
        feel_text = (
            f"Stand at the midpoint facing the hole. "
            f"You will feel {weight_word} pressure on your "
            f"{heavy_foot} foot ({feel_word} tilt). "
            f"High side: {high_side}. Ball falls to the {low_side}."
        )
        y, svgs = body_wrap(y, feel_text, color=C.WHITE)
        parts.extend(svgs)
        
        # elevation feel
        if ep > 0:
            elev_feel = f"Lean BACK slightly ({ep:.1f}% uphill). Hit firmer - ball breaks less because it moves faster across the slope."
        elif ep < 0:
            elev_feel = f"Lean FORWARD slightly ({abs(ep):.1f}% downhill). Hit softer - ball breaks more because it moves slower across the slope."
        else:
            elev_feel = "Level stance (flat). Normal pace - no elevation adjustment needed."
        y, svgs = body_wrap(y, elev_feel, color=C.LIGHT_GRAY, size=8)
        parts.extend(svgs)
        y += 4
        
        # ═══ ELEVATION & BREAK ═══
        y, t = heading(y, "ELEVATION &amp; BREAK:")
        parts.append(t)
        
        this_adj = avg_slope * elev_fac
        
        if ep > 0:
            elev_text = (
                f"Uphill {ep:.1f}% means LESS break. You must hit harder "
                f"to reach the hole, so the ball moves faster across "
                f"the side slope, giving gravity less time to pull it "
                f"sideways. This {avg_slope:.1f}% slope plays as "
                f"{this_adj:.1f}% effective (vs {avg_slope:.1f}% on flat)."
            )
        elif ep < 0:
            elev_text = (
                f"Downhill {abs(ep):.1f}% means MORE break. You hit softer "
                f"so the ball rolls slower across the side slope, "
                f"giving gravity more time to pull it sideways. "
                f"This {avg_slope:.1f}% slope plays as "
                f"{this_adj:.1f}% effective (vs {avg_slope:.1f}% on flat)."
            )
        else:
            elev_text = (
                f"Flat putt - no elevation adjustment. "
                f"The {avg_slope:.1f}% slope plays at face value."
            )
        
        y, svgs = body_wrap(y, elev_text, color=C.LIGHT_GRAY, size=8)
        parts.extend(svgs)
        
        # ═══ QUICK TIPS ═══
        y += 4
        y, t = heading(y, "QUICK TIPS:")
        parts.append(t)
        
        tips = [
            f"1. Feel the slope with your feet at the midpoint.",
            f"2. Aim {aim_fingers} finger{'s' if aim_fingers != 1 else ''} to the {high_side} (high side).",
        ]
        if config.with_grain:
            tips.append("3. With grain: expect late break near the hole.")
        else:
            tips.append("3. Against grain: hit firmer, less total break.")
        if config.break_change_points:
            tips.append("4. Watch for break changes (marked on green)!")
        
        for tip in tips:
            y, svgs = body_wrap(y, tip, color=C.WHITE, size=8)
            parts.extend(svgs)
        
        return '\n    '.join(parts)
    
    def _wrap_text(self, text: str, max_width: int, font_size: int) -> List[str]:
        """Wrap text to fit within max_width pixels.
        
        Arial average character width is ~0.47× the font size (measured).
        """
        char_width = font_size * 0.47
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
    
    # Slope clamping range (realistic putting green limits)
    MIN_SLOPE_PERCENT = 0.5   # Below this, break is negligible
    MAX_SLOPE_PERCENT = 5.0   # USGA: 1-3% typical, 4% challenging, 5% absolute max
    
    # AimPoint finger calculation
    PIXELS_PER_FINGER = 4.5   # 1 finger ≈ 1 inch ≈ ~4.5px (hole is ~4.25" = 18px)
    
    
    # Trajectory pixel margin
    PATH_MARGIN_PX = 10  # Keep putt path within green bounds
    
    def __init__(self, width: int = 120, height: int = 660, text_column_width: int = 200):
        self.layout = LayoutConfig(width=width, height=height, text_column_width=text_column_width)
        self.renderer = SVGRenderer(self.layout)
    
    def generate_svg(self,
                    with_grain: bool,
                    elevation_percent: float,
                    break_direction: str,
                    slope_percent: float,
                    break_change_points: Optional[List[Tuple[float, float]]] = None,
                    ridge_putt: bool = False,
                    output_path: Optional[str] = None) -> Tuple[str, List]:
        """
        Generate SVG string for a putt illustration with AimPoint annotations.
        
        Args:
            elevation_percent: Positive = uphill, negative = downhill, 0 = flat.
        """
        # Validate and clamp inputs
        if break_direction not in ["left", "right"]:
            raise ValueError("break_direction must be 'left' or 'right'")
        
        # Clamp slope to realistic putting green range
        slope_percent = max(self.MIN_SLOPE_PERCENT, min(self.MAX_SLOPE_PERCENT, slope_percent))
        
        # Validate break_change_points
        if break_change_points:
            for i, (dist, slope) in enumerate(break_change_points):
                if not (0.0 < dist < 10.0):
                    raise ValueError(
                        f"break_change_points[{i}]: distance {dist} must be between "
                        f"0 and {10.0} (putt length in feet)"
                    )
                if abs(slope) > self.MAX_SLOPE_PERCENT:
                    warnings.warn(
                        f"break_change_points[{i}]: slope {slope}% exceeds realistic "
                        f"max ({self.MAX_SLOPE_PERCENT}%), will be used as-is in sections",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        
        # Create configuration
        config = PuttConfig(
            with_grain=with_grain,
            elevation_percent=elevation_percent,
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
        aim_offset_pixels = abs(apex_point[0] - r.center_x)
        aim_fingers = int(aim_offset_pixels / self.PIXELS_PER_FINGER + 0.5)
        
        # Generate SVG
        svg = self._build_svg(config, sections, putt_path, apex_point, aim_fingers)
        
        # Save if output path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(svg)
        
        return svg, sections
    
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
        margin = PuttIllustrationGenerator.PATH_MARGIN_PX
        
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
        
        <!-- Elevation direction markers (distinct from grain) -->
        <marker id="arrowhead-elev-up" markerWidth="8" markerHeight="8" refX="4" refY="8" orient="0">
            <polygon points="4,0 8,8 0,8" fill="{C.ELEV_UPHILL}" opacity="0.85" />
        </marker>
        <marker id="arrowhead-elev-down" markerWidth="8" markerHeight="8" refX="4" refY="0" orient="0">
            <polygon points="0,0 8,0 4,8" fill="{C.ELEV_DOWNHILL}" opacity="0.85" />
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
    
    <!-- Visual elevation direction indicators (uphill/downhill) -->
    {r.render_elevation_indicators(config.elevation_percent)}
    
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
                 break_change_points: List, ridge_putt: bool,
                 sections: List[SlopeSection]) -> List[str]:
    """
    Validate SVG against all rules. Returns list of error messages (empty if valid).
    
    Args:
        svg_content: Generated SVG string
        break_direction: Base break direction ("left" or "right")
        slope_percent: Base slope percentage
        break_change_points: List of (distance_ft, new_slope) changes
        ridge_putt: Whether this is a ridge putt with direction reversals
        sections: List of SlopeSection objects from the putt config
    """
    errors = []
    
    # Validate section directions
    for section in sections:
        expected_dir = "L to R" if section.breaks_right else "R to L"
        desc = section.description()
        
        # Extract actual direction from description
        actual_dir = None
        if "L to R" in desc:
            actual_dir = "L to R"
        elif "R to L" in desc:
            actual_dir = "R to L"
        
        if actual_dir != expected_dir:
            errors.append(
                f"Section {section.section_number}: Direction mismatch. "
                f"Expected '{expected_dir}', got '{actual_dir}'"
            )
    
    # Validate path starts at ball using XML parsing
    from xml.etree import ElementTree as ET
    try:
        root = ET.fromstring(svg_content)
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        
        # Find all <path> elements (try with and without namespace)
        paths = root.findall('.//svg:path', ns) or root.findall('.//path')
        
        for path_el in paths:
            d = path_el.get('d', '')
            if d.startswith('M '):
                # Parse the M (moveto) command coordinates
                parts = d[2:].replace(',', ' ').split()
                if len(parts) >= 2:
                    try:
                        path_start_x = float(parts[0])
                        path_start_y = float(parts[1])
                        
                        layout = LayoutConfig()
                        if abs(path_start_x - layout.center_x) > 0.5:
                            errors.append(
                                f"Path must start at ball x={layout.center_x}, "
                                f"got {path_start_x:.1f}"
                            )
                        if abs(path_start_y - layout.ball_y) > 0.5:
                            errors.append(
                                f"Path must start at ball y={layout.ball_y}, "
                                f"got {path_start_y:.1f}"
                            )
                    except ValueError:
                        pass
                break  # Only validate the first path (putt trajectory)
    except ET.ParseError as e:
        errors.append(f"SVG is not valid XML: {e}")
    
    return errors


# =============================================================================
# SCENARIO DATA
# =============================================================================

MAIN_SCENARIOS = [
    # SIMPLE BREAKS — variety of elevation: downhill, uphill, flat, steep down, gentle up
    ("01_simple_right_with_grain_downhill", {
        "with_grain": True, "elevation_percent": -1.5, "break_direction": "right",
        "slope_percent": 1.5, "description": "Simple right break, with grain, 1.5% downhill"
    }),
    ("02_simple_left_against_grain_uphill", {
        "with_grain": False, "elevation_percent": 2.0, "break_direction": "left",
        "slope_percent": 1.8, "description": "Simple left break, against grain, 2% uphill"
    }),
    ("03_simple_right_with_grain_flat", {
        "with_grain": True, "elevation_percent": 0.0, "break_direction": "right",
        "slope_percent": 1.2, "description": "Simple right break, with grain, flat"
    }),
    ("04_simple_left_strong_against_grain", {
        "with_grain": False, "elevation_percent": -0.5, "break_direction": "left",
        "slope_percent": 2.5, "description": "Strong left break, against grain, slight downhill"
    }),
    ("05_simple_right_strong_with_grain_uphill", {
        "with_grain": True, "elevation_percent": 1.0, "break_direction": "right",
        "slope_percent": 2.8, "description": "Strong right break, with grain, 1% uphill"
    }),
    ("06_simple_left_gentle_against_grain_downhill", {
        "with_grain": False, "elevation_percent": -2.5, "break_direction": "left",
        "slope_percent": 1.0, "description": "Gentle left break, against grain, 2.5% downhill"
    }),

    # DOUBLE BREAKS
    ("07_double_right_increasing_with_grain", {
        "with_grain": True, "elevation_percent": -1.0, "break_direction": "right",
        "slope_percent": 1.0, "break_change_points": [(5.0, 2.5)],
        "description": "Double break - right, increasing 1->2.5%, with grain, 1% downhill"
    }),
    ("08_double_left_increasing_against_grain_uphill", {
        "with_grain": False, "elevation_percent": 1.5, "break_direction": "left",
        "slope_percent": 0.8, "break_change_points": [(4.0, 2.2)],
        "description": "Double break - left, increasing 0.8->2.2%, against grain, 1.5% uphill"
    }),
    ("09_double_right_increasing_near_hole", {
        "with_grain": True, "elevation_percent": 0.0, "break_direction": "right",
        "slope_percent": 1.2, "break_change_points": [(7.0, 2.8)],
        "description": "Double break - right, increases near hole 1.2->2.8%, flat"
    }),
    ("10_double_left_decreasing_with_grain", {
        "with_grain": True, "elevation_percent": -0.8, "break_direction": "left",
        "slope_percent": 2.5, "break_change_points": [(5.0, 1.2)],
        "description": "Double break - left, decreasing 2.5->1.2%, slight downhill"
    }),
    ("11_double_right_decreasing_against_grain_uphill", {
        "with_grain": False, "elevation_percent": 2.5, "break_direction": "right",
        "slope_percent": 2.8, "break_change_points": [(4.0, 1.0)],
        "description": "Double break - right, decreasing 2.8->1%, 2.5% uphill"
    }),

    # RIDGE PUTTS
    ("12_ridge_right_reverses_with_grain", {
        "with_grain": True, "elevation_percent": -1.2, "break_direction": "right",
        "slope_percent": 2.0, "break_change_points": [(4.0, -1.8)], "ridge_putt": True,
        "description": "Ridge putt - right reverses to left at 4ft, 1.2% downhill"
    }),
    ("13_ridge_left_reverses_against_grain", {
        "with_grain": False, "elevation_percent": 0.0, "break_direction": "left",
        "slope_percent": 1.5, "break_change_points": [(5.0, -2.0)], "ridge_putt": True,
        "description": "Ridge putt - left reverses to right at 5ft, flat"
    }),
    ("14_ridge_right_reverses_uphill", {
        "with_grain": True, "elevation_percent": 1.8, "break_direction": "right",
        "slope_percent": 2.2, "break_change_points": [(3.5, -1.5)], "ridge_putt": True,
        "description": "Ridge putt - right reverses at 3.5ft, 1.8% uphill"
    }),

    # COMPLEX BREAKS
    ("15_triple_right_complex_with_grain", {
        "with_grain": True, "elevation_percent": -0.5, "break_direction": "right",
        "slope_percent": 0.8, "break_change_points": [(3.0, 2.2), (7.0, 1.0)],
        "description": "Triple break right, slight downhill"
    }),
    ("16_triple_left_complex_against_grain", {
        "with_grain": False, "elevation_percent": -2.0, "break_direction": "left",
        "slope_percent": 1.0, "break_change_points": [(2.5, 2.5), (6.5, 1.5)],
        "description": "Triple break left, 2% downhill"
    }),
    ("17_triple_right_complex_uphill", {
        "with_grain": True, "elevation_percent": 1.2, "break_direction": "right",
        "slope_percent": 1.2, "break_change_points": [(4.0, 2.8), (8.0, 1.8)],
        "description": "Triple break right, 1.2% uphill"
    }),
]

# =========================================================================
# AIMPOINT CATEGORIZED SCENARIOS
# Tuned to produce specific AimPoint finger readings
# =========================================================================

AIMPOINT_1_SCENARIOS = [
    # Simple breaks — mix of elevations
    ("ap1_gentle_right_downhill", {
        "with_grain": True, "elevation_percent": -1.0, "break_direction": "right",
        "slope_percent": 1.2, "description": "AimPoint 1: Gentle right, 1% downhill"
    }),
    ("ap1_gentle_left_flat", {
        "with_grain": True, "elevation_percent": 0.0, "break_direction": "left",
        "slope_percent": 1.3, "description": "AimPoint 1: Gentle left, flat"
    }),
    ("ap1_gentle_right_uphill", {
        "with_grain": False, "elevation_percent": 1.5, "break_direction": "right",
        "slope_percent": 1.5, "description": "AimPoint 1: Gentle right, 1.5% uphill"
    }),
    ("ap1_gentle_left_against_grain", {
        "with_grain": False, "elevation_percent": -0.3, "break_direction": "left",
        "slope_percent": 1.2, "description": "AimPoint 1: Gentle left, slight downhill"
    }),
    # Double breaks
    ("ap1_double_right_slight", {
        "with_grain": True, "elevation_percent": -0.5, "break_direction": "right",
        "slope_percent": 1.0, "break_change_points": [(5.0, 1.5)],
        "description": "AimPoint 1: Double right, slight downhill"
    }),
    ("ap1_double_left_decreasing", {
        "with_grain": False, "elevation_percent": 2.0, "break_direction": "left",
        "slope_percent": 1.8, "break_change_points": [(6.0, 1.0)],
        "description": "AimPoint 1: Double left decreasing, 2% uphill"
    }),
    # Double breaks with REVERSAL
    ("ap1_double_right_to_left", {
        "with_grain": True, "elevation_percent": 0.0, "break_direction": "right",
        "slope_percent": 2.0, "break_change_points": [(5.0, -0.8)], "ridge_putt": True,
        "description": "AimPoint 1: R->L reversal at 5ft, flat"
    }),
    ("ap1_double_left_to_right", {
        "with_grain": False, "elevation_percent": -1.5, "break_direction": "left",
        "slope_percent": 1.8, "break_change_points": [(5.0, -0.6)], "ridge_putt": True,
        "description": "AimPoint 1: L->R reversal at 5ft, 1.5% downhill"
    }),
    # Triple breaks with 2 REVERSALS
    ("ap1_triple_right_left_right", {
        "with_grain": True, "elevation_percent": 0.5, "break_direction": "right",
        "slope_percent": 1.5, "break_change_points": [(3.5, -0.8), (7.0, 0.6)], "ridge_putt": True,
        "description": "AimPoint 1: Triple R->L->R, gentle uphill"
    }),
    ("ap1_triple_left_right_left", {
        "with_grain": False, "elevation_percent": -0.8, "break_direction": "left",
        "slope_percent": 1.3, "break_change_points": [(4.0, -0.7), (7.5, 0.5)], "ridge_putt": True,
        "description": "AimPoint 1: Triple L->R->L, slight downhill"
    }),
]

AIMPOINT_2_SCENARIOS = [
    # Simple breaks — varied elevations
    ("ap2_moderate_right_downhill", {
        "with_grain": True, "elevation_percent": -1.8, "break_direction": "right",
        "slope_percent": 2.0, "description": "AimPoint 2: Moderate right, 1.8% downhill"
    }),
    ("ap2_moderate_left_flat", {
        "with_grain": True, "elevation_percent": 0.0, "break_direction": "left",
        "slope_percent": 2.2, "description": "AimPoint 2: Moderate left, flat"
    }),
    ("ap2_moderate_right_against_grain", {
        "with_grain": False, "elevation_percent": -0.5, "break_direction": "right",
        "slope_percent": 2.3, "description": "AimPoint 2: Moderate right, slight downhill"
    }),
    ("ap2_moderate_left_uphill", {
        "with_grain": True, "elevation_percent": 1.5, "break_direction": "left",
        "slope_percent": 2.5, "description": "AimPoint 2: Moderate left, 1.5% uphill"
    }),
    # Double breaks
    ("ap2_double_right_building", {
        "with_grain": True, "elevation_percent": -1.0, "break_direction": "right",
        "slope_percent": 1.5, "break_change_points": [(5.0, 2.5)],
        "description": "AimPoint 2: Double right building, 1% downhill"
    }),
    ("ap2_double_left_consistent", {
        "with_grain": False, "elevation_percent": 0.8, "break_direction": "left",
        "slope_percent": 2.0, "break_change_points": [(5.0, 2.3)],
        "description": "AimPoint 2: Double left consistent, gentle uphill"
    }),
    # Double breaks with REVERSAL
    ("ap2_double_right_to_left", {
        "with_grain": True, "elevation_percent": 0.0, "break_direction": "right",
        "slope_percent": 4.5, "break_change_points": [(7.0, -0.5)], "ridge_putt": True,
        "description": "AimPoint 2: R->L reversal, flat"
    }),
    ("ap2_double_left_to_right", {
        "with_grain": True, "elevation_percent": -2.0, "break_direction": "left",
        "slope_percent": 4.2, "break_change_points": [(7.0, -0.4)], "ridge_putt": True,
        "description": "AimPoint 2: L->R reversal, 2% downhill"
    }),
    # Triple breaks with 2 REVERSALS
    ("ap2_triple_right_left_right", {
        "with_grain": True, "elevation_percent": 1.0, "break_direction": "right",
        "slope_percent": 4.0, "break_change_points": [(5.0, -0.5), (8.0, 0.3)], "ridge_putt": True,
        "description": "AimPoint 2: Triple R->L->R, 1% uphill"
    }),
    ("ap2_triple_left_right_left", {
        "with_grain": True, "elevation_percent": -0.5, "break_direction": "left",
        "slope_percent": 3.8, "break_change_points": [(5.0, -0.4), (8.0, 0.2)], "ridge_putt": True,
        "description": "AimPoint 2: Triple L->R->L, slight downhill"
    }),
]

AIMPOINT_3_SCENARIOS = [
    # Simple breaks — varied elevations
    ("ap3_strong_right_downhill", {
        "with_grain": True, "elevation_percent": -2.5, "break_direction": "right",
        "slope_percent": 3.2, "description": "AimPoint 3: Strong right, 2.5% downhill"
    }),
    ("ap3_strong_left_flat", {
        "with_grain": True, "elevation_percent": 0.0, "break_direction": "left",
        "slope_percent": 3.5, "description": "AimPoint 3: Strong left, flat"
    }),
    ("ap3_strong_right_against_grain", {
        "with_grain": False, "elevation_percent": -1.0, "break_direction": "right",
        "slope_percent": 3.8, "description": "AimPoint 3: Strong right, 1% downhill"
    }),
    ("ap3_strong_left_uphill", {
        "with_grain": False, "elevation_percent": 2.5, "break_direction": "left",
        "slope_percent": 4.0, "description": "AimPoint 3: Strong left, 2.5% uphill"
    }),
    # Double breaks
    ("ap3_double_right_steep", {
        "with_grain": True, "elevation_percent": -1.5, "break_direction": "right",
        "slope_percent": 2.5, "break_change_points": [(4.0, 4.0)],
        "description": "AimPoint 3: Double right steep, 1.5% downhill"
    }),
    ("ap3_double_left_heavy", {
        "with_grain": True, "elevation_percent": 0.5, "break_direction": "left",
        "slope_percent": 3.0, "break_change_points": [(5.0, 3.5)],
        "description": "AimPoint 3: Double left heavy, gentle uphill"
    }),
    # Double breaks with REVERSAL
    ("ap3_double_right_to_left", {
        "with_grain": True, "elevation_percent": 0.0, "break_direction": "right",
        "slope_percent": 5.0, "break_change_points": [(8.5, -0.3)], "ridge_putt": True,
        "description": "AimPoint 3: R->L reversal, flat"
    }),
    ("ap3_double_left_to_right", {
        "with_grain": True, "elevation_percent": -2.0, "break_direction": "left",
        "slope_percent": 4.8, "break_change_points": [(8.5, -0.3)], "ridge_putt": True,
        "description": "AimPoint 3: L->R reversal, 2% downhill"
    }),
    # Triple breaks with 2 REVERSALS
    ("ap3_triple_right_left_right", {
        "with_grain": True, "elevation_percent": 1.2, "break_direction": "right",
        "slope_percent": 5.0, "break_change_points": [(7.0, -0.3), (9.0, 0.2)], "ridge_putt": True,
        "description": "AimPoint 3: Triple R->L->R, 1.2% uphill"
    }),
    ("ap3_triple_left_right_left", {
        "with_grain": True, "elevation_percent": -0.8, "break_direction": "left",
        "slope_percent": 4.8, "break_change_points": [(7.0, -0.3), (9.0, 0.2)], "ridge_putt": True,
        "description": "AimPoint 3: Triple L->R->L, slight downhill"
    }),
    # Triple breaks
    ("ap3_triple_right_aggressive", {
        "with_grain": True, "elevation_percent": -3.0, "break_direction": "right",
        "slope_percent": 2.5, "break_change_points": [(3.0, 4.0), (7.0, 3.2)],
        "description": "AimPoint 3: Triple right aggressive, 3% downhill"
    }),
    ("ap3_triple_left_intense", {
        "with_grain": False, "elevation_percent": 1.8, "break_direction": "left",
        "slope_percent": 2.8, "break_change_points": [(3.5, 4.2), (7.0, 3.5)],
        "description": "AimPoint 3: Triple left intense, 1.8% uphill"
    }),
]


# =============================================================================
# MAIN
# =============================================================================

def _extract_aimpoint_from_svg(svg_content: str) -> int:
    """Extract the actual AimPoint value from generated SVG content."""
    match = re.search(r'AimPoint: (\d+)', svg_content)
    return int(match.group(1)) if match else 0


def _generate_scenario(generator, filename, params, output_path=None):
    """
    Generate a single scenario SVG, validate it, and return results.
    
    Args:
        generator: PuttIllustrationGenerator instance
        filename: Scenario name for logging
        params: Dict of scenario parameters
        output_path: If provided, save directly to this path
        
    Returns:
        Tuple of (svg_content, sections, aimpoint_val, errors)
    """
    svg_content, sections = generator.generate_svg(
        with_grain=params["with_grain"],
        elevation_percent=params.get("elevation_percent", 0.0),
        break_direction=params["break_direction"],
        slope_percent=params["slope_percent"],
        break_change_points=params.get("break_change_points"),
        ridge_putt=params.get("ridge_putt", False),
        output_path=output_path
    )
    
    aimpoint_val = _extract_aimpoint_from_svg(svg_content)
    
    errors = validate_svg(
        svg_content, params["break_direction"], params["slope_percent"],
        params.get("break_change_points", []), params.get("ridge_putt", False),
        sections
    )
    
    return svg_content, sections, aimpoint_val, errors


def _log_scenario_header(prefix, idx, total, filename, params):
    """Print scenario generation header."""
    print(f"\n{prefix}[{idx}/{total}] Generating: {filename}")
    print(f"   Description: {params.get('description', 'N/A')}")
    print(f"   Parameters: break={params['break_direction']}, slope={params['slope_percent']}%")
    if params.get('break_change_points'):
        print(f"   Break changes: {params['break_change_points']}")
    if params.get('ridge_putt'):
        print(f"   Ridge putt: Yes")


def main():
    """Generate comprehensive set of realistic 10ft putt scenarios."""
    generator = PuttIllustrationGenerator(width=120, height=660, text_column_width=200)
    
    output_dir = Path(__file__).parent / "output"
    
    # Clean stale SVGs from previous runs before regenerating
    if output_dir.exists():
        for old_svg in output_dir.rglob("*.svg"):
            old_svg.unlink()
    
    output_dir.mkdir(exist_ok=True)
    
    aimpoint_dirs = {
        1: output_dir / "aimpoint_1",
        2: output_dir / "aimpoint_2",
        3: output_dir / "aimpoint_3",
    }
    for d in aimpoint_dirs.values():
        d.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("GENERATING PUTT ILLUSTRATIONS WITH REFACTORED PHYSICS ENGINE")
    print("=" * 80)
    
    successful_count = 0
    validation_errors = []
    aimpoint_counts = {0: 0, 1: 0, 2: 0, 3: 0, "4+": 0}
    
    # --- Generate main scenarios (fixed output directory) ---
    print("\n" + "-" * 40)
    print("MAIN SCENARIOS")
    print("-" * 40)
    
    for idx, (filename, params) in enumerate(MAIN_SCENARIOS, 1):
        _log_scenario_header("[MAIN] ", idx, len(MAIN_SCENARIOS), filename, params)
        try:
            output_path = str(output_dir / f"{filename}.svg")
            result = _generate_scenario(generator, filename, params, output_path)
            svg_content, sections, aimpoint_val, errors = result
            
            print(f"   [OK] SVG generated (AimPoint: {aimpoint_val})")
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
            traceback.print_exc()
    
    # --- Generate and classify AimPoint scenarios ---
    print("\n" + "-" * 40)
    print("AIMPOINT SCENARIOS (Classifying by ACTUAL value)")
    print("-" * 40)
    
    all_aimpoint_scenarios = (
        AIMPOINT_1_SCENARIOS +
        AIMPOINT_2_SCENARIOS +
        AIMPOINT_3_SCENARIOS
    )
    
    for idx, (filename, params) in enumerate(all_aimpoint_scenarios, 1):
        _log_scenario_header("[AP] ", idx, len(all_aimpoint_scenarios), filename, params)
        try:
            result = _generate_scenario(generator, filename, params)
            svg_content, sections, actual_aimpoint, errors = result
            
            # Classify into folder by actual AimPoint value
            if actual_aimpoint == 0:
                print(f"   [SKIP] AimPoint: 0 - skipping (breaks cancel out)")
                aimpoint_counts[0] += 1
                continue
            
            if actual_aimpoint >= 4:
                target_folder = aimpoint_dirs[3]
                aimpoint_counts["4+"] += 1
            else:
                target_folder = aimpoint_dirs[actual_aimpoint]
                aimpoint_counts[actual_aimpoint] += 1
            
            output_path = str(target_folder / f"{filename}.svg")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(svg_content)
            
            print(f"   [OK] Actual AimPoint: {actual_aimpoint} -> Saved to: aimpoint_{min(actual_aimpoint, 3)}/")
            if errors:
                validation_errors.append((filename, errors))
                print(f"   [WARN] Validation issues: {errors}")
            else:
                successful_count += 1
        except Exception as e:
            print(f"\n   [ERROR] ERROR generating {filename}: {e}")
            traceback.print_exc()
    
    # --- Summary ---
    total_all = len(MAIN_SCENARIOS) + len(all_aimpoint_scenarios) - aimpoint_counts[0]
    
    print("\n" + "=" * 80)
    print("GENERATION SUMMARY")
    print("=" * 80)
    print(f"Main scenarios: {len(MAIN_SCENARIOS)}")
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
