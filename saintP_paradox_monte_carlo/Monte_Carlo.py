import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_simulation(A0, B0, k1, k2, dt, t_max):
    A = A0
    B = B0
    N = A + B  # общее количество частиц 
    
    # временные точки для сохранения результатов
    time_points = np.arange(0, t_max + dt, dt)
    A_history = np.zeros(len(time_points))
    B_history = np.zeros(len(time_points))
    
    A_history[0] = A
    B_history[0] = B
    
    t = 0
    time_idx = 1
    
    while t < t_max:
        # шаг 2: выбираем случайную частицу N раз
        for _ in range(N):
            # выбираем A или B пропорционально их количеству
            if np.random.rand() < A / (A + B):
                # выбрана частица A
                if np.random.rand() < k1 * dt:
                    # A превращается в B
                    A -= 1
                    B += 1
            else:
                # выбрана частица B
                if np.random.rand() < k2 * dt:
                    # B превращается в A
                    A += 1
                    B -= 1
        
        # увеличиваем время
        t += dt
        
        # сохраняем текущие значения
        if time_idx < len(time_points):
            A_history[time_idx] = A
            B_history[time_idx] = B
            time_idx += 1
    
    return time_points, A_history, B_history

def analytical_solution(A0, B0, k1, k2, t_points):

    N_total = A0 + B0
    k_sum = k1 + k2
    
    #предэкспоненциальные коэффициенты
    C1 = (k2 * N_total) / k_sum
    C2 = (A0 * k1 - B0 * k2) / k_sum
    
    A_analytical = C1 + C2 * np.exp(-k_sum * t_points)
    B_analytical = N_total - A_analytical
    
    return A_analytical, B_analytical

def main():
    A0 = 500
    B0 = 200
    k1 = 0.1
    k2 = 0.8
    N_total = A0 + B0
    
    # параметры симуляции
    dt = 0.01  # шаг по времени
    t_max = 30  # максимальное время
    n_simulations = 100  # количество симуляций для усреднения
    
    #массивы для усредненных результатов
    time_points = np.arange(0, t_max + dt, dt)
    A_all_simulations = np.zeros((n_simulations, len(time_points)))
    B_all_simulations = np.zeros((n_simulations, len(time_points)))
    
    print(f"Запуск {n_simulations} симуляций Монте-Карло...")
    print(f"Параметры: A0={A0}, B0={B0}, k1={k1}, k2={k2}, dt={dt}, t_max={t_max}")
    
    #ыыполняем симуляции
    for i in range(n_simulations):
        if (i + 1) % 10 == 0:
            print(f"  Симуляция {i + 1}/{n_simulations}")
        t_points, A_hist, B_hist = monte_carlo_simulation(A0, B0, k1, k2, dt, t_max)
        A_all_simulations[i] = A_hist
        B_all_simulations[i] = B_hist
    
    # ссредняем результаты
    A_avg = np.mean(A_all_simulations, axis=0)
    B_avg = np.mean(B_all_simulations, axis=0)
    
    # стандартное отклонение
    A_std = np.std(A_all_simulations, axis=0)
    B_std = np.std(B_all_simulations, axis=0)
    
    # аналитическое решение
    A_analytical, B_analytical = analytical_solution(A0, B0, k1, k2, time_points)
    
    # предельные значения при t -> бесконечность
    A_inf = (k2 * N_total) / (k1 + k2)
    B_inf = (k1 * N_total) / (k1 + k2)
    
    print(f"\nРезультаты:")
    print(f"Аналитическое решение при t→∞: A∞={A_inf:.2f}, B∞={B_inf:.2f}")
    print(f"Усредненный результат при t={t_max}: A={A_avg[-1]:.2f}±{A_std[-1]:.2f}, B={B_avg[-1]:.2f}±{B_std[-1]:.2f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # график 1: усредненные значения A и B
    ax1 = axes[0, 0]
    ax1.plot(time_points, A_avg, 'b-', linewidth=2, label=f'A (MC, среднее по {n_simulations} симуляциям)')
    ax1.plot(time_points, B_avg, 'r-', linewidth=2, label=f'B (MC, среднее по {n_simulations} симуляциям)')
    ax1.plot(time_points, A_analytical, 'b--', linewidth=2, alpha=0.7, label='A (аналитическое)')
    ax1.plot(time_points, B_analytical, 'r--', linewidth=2, alpha=0.7, label='B (аналитическое)')
    ax1.axhline(y=A_inf, color='b', linestyle=':', alpha=0.5, label=f'A∞ = {A_inf:.1f}')
    ax1.axhline(y=B_inf, color='r', linestyle=':', alpha=0.5, label=f'B∞ = {B_inf:.1f}')
    ax1.set_xlabel('Время')
    ax1.set_ylabel('Количество частиц')
    ax1.set_title('Усредненные результаты Монте-Карло vs Аналитическое решение')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # график 2: флуктуации
    ax2 = axes[0, 1]
    ax2.plot(time_points, A_std, 'b-', linewidth=2, label='Стандартное отклонение A')
    ax2.plot(time_points, B_std, 'r-', linewidth=2, label='Стандартное отклонение B')
    ax2.set_xlabel('Время')
    ax2.set_ylabel('Стандартное отклонение')
    ax2.set_title('Флуктуации в симуляциях Монте-Карло')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # график 3: несколько отдельных симуляций
    ax3 = axes[1, 0]
    for i in range(min(5, n_simulations)):
        ax3.plot(time_points, A_all_simulations[i], alpha=0.6, linewidth=1, label=f'Симуляция {i+1}' if i < 3 else "")
    ax3.plot(time_points, A_analytical, 'k--', linewidth=2, label='Аналитическое решение')
    ax3.set_xlabel('Время')
    ax3.set_ylabel('Количество A')
    ax3.set_title('Отдельные симуляции Монте-Карло (частицы A)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # график 4: распределение в конце симуляции
    ax4 = axes[1, 1]
    final_A_values = A_all_simulations[:, -1]
    ax4.hist(final_A_values, bins=20, alpha=0.7, edgecolor='black', color='blue')
    ax4.axvline(x=A_inf, color='red', linestyle='--', linewidth=2, label=f'Аналитическое A∞ = {A_inf:.1f}')
    ax4.axvline(x=np.mean(final_A_values), color='green', linestyle='-', linewidth=2, 
                label=f'Среднее MC = {np.mean(final_A_values):.1f}')
    ax4.set_xlabel('Количество A в конце симуляции')
    ax4.set_ylabel('Частота')
    ax4.set_title(f'Распределение A(t={t_max}) по {n_simulations} симуляциям')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    #дополнительная статистика
    print(f"\nСтатистика по {n_simulations} симуляциям:")
    print(f"Среднее A в конце: {np.mean(final_A_values):.2f} ± {np.std(final_A_values):.2f}")
    print(f"Медиана A в конце: {np.median(final_A_values):.2f}")
    print(f"Минимальное A в конце: {np.min(final_A_values):.2f}")
    print(f"Максимальное A в конце: {np.max(final_A_values):.2f}")
    print(f"Разброс (max-min): {np.max(final_A_values) - np.min(final_A_values):.2f}")

if __name__ == "__main__":
    main()