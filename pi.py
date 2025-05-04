import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.patches import Circle
from tqdm import tqdm

def monte_carlo_pi(num_points):
    x = np.random.random(num_points)
    y = np.random.random(num_points)
    distances = (x - 0.5)**2 + (y - 0.5)**2
    inside_circle = distances <= 0.25
    points_inside_x = x[inside_circle]
    points_inside_y = y[inside_circle]
    points_outside_x = x[~inside_circle]
    points_outside_y = y[~inside_circle]
    pi_approximation = 4 * np.sum(inside_circle) / num_points
    return (
        pi_approximation,
        points_inside_x, points_inside_y,
        points_outside_x, points_outside_y
    )

def analyze_convergence(max_points, num_steps=20):
    point_counts = np.logspace(1, np.log10(max_points), num_steps).astype(int)
    approximations = []
    errors = []
    for n in tqdm(point_counts):
        pi_approx, _, _, _, _ = monte_carlo_pi(n)
        approximations.append(pi_approx)
        errors.append(abs(pi_approx - np.pi))
    return point_counts, approximations, errors

def plot_simulation(inside_x, inside_y, outside_x, outside_y, pi_approx, num_points):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(outside_x, outside_y, s=1, color='red', alpha=0.5, label='Fora do círculo')
    plt.scatter(inside_x, inside_y, s=1, color='blue', alpha=0.5, label='Dentro do círculo')
    circle = Circle((0.5, 0.5), 0.5, fill=False, color='black', linewidth=2)
    plt.gca().add_patch(circle)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal')
    plt.grid(True, alpha=0.3)
    plt.title(f'Simulação de Monte Carlo com {num_points:,} pontos\n'
              f'π ≈ {pi_approx:.8f} (real: {np.pi:.8f}, erro: {abs(pi_approx - np.pi):.8f})')
    plt.legend()
    return plt

def plot_convergence(point_counts, approximations, errors):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.semilogx(point_counts, approximations, 'o-', markersize=4)
    plt.axhline(y=np.pi, color='r', linestyle='-', alpha=0.7, label='Valor real de π')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Número de pontos (escala log)')
    plt.ylabel('Aproximação de π')
    plt.title('Convergência da aproximação de π')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.loglog(point_counts, errors, 'o-', markersize=4)
    reference_line = np.pi / np.sqrt(point_counts)
    plt.loglog(point_counts, reference_line, 'r--', alpha=0.7, 
              label='Tendência teórica (1/√n)')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Número de pontos (escala log)')
    plt.ylabel('Erro absoluto (escala log)')
    plt.title('Erro da aproximação vs. número de pontos')
    plt.legend()
    plt.tight_layout()
    return plt

def run_full_demo():
    print("=== Demonstração do Método de Monte Carlo para calcular π ===")
    num_points = 100000
    print(f"\nCalculando π com {num_points:,} pontos...")
    start_time = time.time()
    pi_approx, inside_x, inside_y, outside_x, outside_y = monte_carlo_pi(num_points)
    execution_time = time.time() - start_time
    print(f"Aproximação de π: {pi_approx:.10f}")
    print(f"Valor real de π:  {np.pi:.10f}")
    print(f"Erro absoluto:    {abs(pi_approx - np.pi):.10f}")
    print(f"Erro relativo:    {abs(pi_approx - np.pi) / np.pi * 100:.8f}%")
    print(f"Tempo de execução: {execution_time:.4f} segundos")
    plt1 = plot_simulation(inside_x, inside_y, outside_x, outside_y, pi_approx, num_points)
    print("\nAnalisando convergência...")
    max_points = 1000000
    point_counts, approximations, errors = analyze_convergence(max_points)
    plt2 = plot_convergence(point_counts, approximations, errors)
    plt1.tight_layout()
    plt2.tight_layout()
    plt1.show()
    plt2.show()
    print("\nDemonstração concluída!")

if __name__ == "__main__":
    np.random.seed(42)
    run_full_demo()
