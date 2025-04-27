import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import time
import random

@dataclass
class Pin:
    absolute_position: Tuple[float, float]
    parent_cell: 'Cell'
    parent_net: List['Net']

@dataclass
class Cell:
    name: str
    pins: dict

@dataclass
class Net:
    pins: List[Pin]

class WirelengthGradientCalculator:
    @staticmethod
    def calculate_wa_gradient(cell: Cell, gamma: float = 1.0) -> Tuple[float, float]:
        """Calculate wirelength gradient using Wirelength Approximation (WA) model."""
        grad_x, grad_y = 0.0, 0.0
        related_nets = WirelengthGradientCalculator._get_related_nets(cell)

        for net in related_nets:
            if len(net.pins) <= 1:
                continue

            positions = [pin.absolute_position for pin in net.pins]
            xs, ys = zip(*positions)
            cell_pins = [pin for pin in net.pins if pin.parent_cell.name == cell.name]

            for pin in cell_pins:
                x_pos, y_pos = pin.absolute_position
                x_grad = WirelengthGradientCalculator._compute_wa_component(x_pos, xs, gamma)
                y_grad = WirelengthGradientCalculator._compute_wa_component(y_pos, ys, gamma)
                
                scaling_factor = len(xs) * gamma
                grad_x += x_grad / scaling_factor
                grad_y += y_grad / scaling_factor

        return grad_x, grad_y

    @staticmethod
    def calculate_lse_gradient(cell: Cell, alpha_scale: float = 0.01) -> Tuple[float, float]:
        """Calculate wirelength gradient using Log-Sum-Exp (LSE) model."""
        grad_x, grad_y = 0.0, 0.0
        related_nets = WirelengthGradientCalculator._get_related_nets(cell)

        for net in related_nets:
            if len(net.pins) <= 1:
                continue

            positions = [pin.absolute_position for pin in net.pins]
            xs, ys = zip(*positions)
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            alpha = alpha_scale * max(max_x - min_x, max_y - min_y, 1e-6)

            for pin in net.pins:
                if pin.parent_cell.name == cell.name:
                    x_pos, y_pos = pin.absolute_position
                    grad_x += np.tanh((x_pos - min_x) / alpha) - np.tanh((max_x - x_pos) / alpha)
                    grad_y += np.tanh((y_pos - min_y) / alpha) - np.tanh((max_y - y_pos) / alpha)

        return grad_x, grad_y

    @staticmethod
    def _get_related_nets(cell: Cell) -> List[Net]:
        """Extract all nets connected to the cell's pins."""
        related_nets = []
        for pin in cell.pins.values():
            if pin.parent_net:
                related_nets.extend(pin.parent_net)
        return related_nets

    @staticmethod
    def _compute_wa_component(pos: float, positions: Tuple[float, ...], gamma: float) -> float:
        """Compute WA gradient component for a single coordinate."""
        grad = 0.0
        for other_pos in positions:
            if abs(pos - other_pos) > 1e-6:
                dx_plus = gamma + pos - other_pos
                dx_minus = gamma - pos + other_pos
                grad += (1.0 / max(dx_plus, 1e-6)) - (1.0 / max(dx_minus, 1e-6))
        return grad

def create_synthetic_circuit(num_cells: int, num_nets: int, pins_per_net: int) -> List[Cell]:
    """Create a synthetic circuit for benchmarking."""
    cells = []
    nets = []

    # Create cells
    for i in range(num_cells):
        pins = {
            f"pin_{j}": Pin(
                absolute_position=(random.uniform(0, 100), random.uniform(0, 100)),
                parent_cell=None,  # Will be set later
                parent_net=[]
            ) for j in range(random.randint(1, 5))
        }
        cell = Cell(name=f"cell_{i}", pins=pins)
        for pin in pins.values():
            pin.parent_cell = cell
        cells.append(cell)

    # Create nets and connect to cell pins
    all_pins = [pin for cell in cells for pin in cell.pins.values()]
    for i in range(num_nets):
        if len(all_pins) < pins_per_net:
            break
        net_pins = random.sample(all_pins, min(pins_per_net, len(all_pins)))
        net = Net(pins=net_pins)
        for pin in net_pins:
            pin.parent_net.append(net)
        nets.append(net)
        all_pins = [p for p in all_pins if p not in net_pins]  # Remove used pins

    return cells

def benchmark_gradients(num_cells: int = 100, num_nets: int = 50, pins_per_net: int = 4):
    """Benchmark WA and LSE gradient models for correctness and performance."""
    print(f"Creating synthetic circuit with {num_cells} cells, {num_nets} nets...")
    cells = create_synthetic_circuit(num_cells, num_nets, pins_per_net)
    
    wa_times, lse_times = [], []
    wa_grads, lse_grads = [], []
    
    calculator = WirelengthGradientCalculator()
    
    print("Running benchmarks...")
    for cell in cells:
        # Benchmark WA model
        start_time = time.time()
        wa_grad = calculator.calculate_wa_gradient(cell, gamma=1.0)
        wa_times.append(time.time() - start_time)
        wa_grads.append(wa_grad)
        
        # Benchmark LSE model
        start_time = time.time()
        lse_grad = calculator.calculate_lse_gradient(cell, alpha_scale=0.01)
        lse_times.append(time.time() - start_time)
        lse_grads.append(lse_grad)
    
    # Analyze results
    avg_wa_time = sum(wa_times) / len(wa_times)
    avg_lse_time = sum(lse_times) / len(lse_times)
    
    # Compare gradients for correctness (relative difference)
    grad_diffs = [
        np.sqrt((wa[0] - lse[0])**2 + (wa[1] - lse[1])**2) / max(np.sqrt(wa[0]**2 + wa[1]**2), 1e-6)
        for wa, lse in zip(wa_grads, lse_grads)
    ]
    avg_grad_diff = sum(grad_diffs) / len(grad_diffs)
    
    print("\nBenchmark Results:")
    print(f"Average WA model time: {avg_wa_time:.6f} seconds")
    print(f"Average LSE model time: {avg_lse_time:.6f} seconds")
    print(f"Average relative gradient difference: {avg_grad_diff:.6f}")
    print(f"Max relative gradient difference: {max(grad_diffs):.6f}")

if __name__ == "__main__":
    benchmark_gradients()