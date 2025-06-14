
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# 1. link2에 input되는 함수(modified sine function)인 y_theta 정의

def y_theta(theta):
    h = 280.03
    if theta < 1/8:
        y = 216.57 - (h / (4 + np.pi)) * (np.pi * theta - (1/4) * np.sin(4 * np.pi * theta))
    elif theta < 7/8:
        y = 216.57 - (h / (4 + np.pi)) * (2 + np.pi * theta - (9/4) * np.sin((4 * np.pi * theta)/3 + np.pi / 3))
    elif theta <= 1:
        y = 216.57 - (h / (4 + np.pi)) * (4 + np.pi * theta - (1/4) * np.sin(4 * np.pi * theta))
    else:
        y = 0
    return y


# 2. 시간 및 r2 계산

t_vals = np.linspace(0.01, 1, 200)
r2_vals = np.array([y_theta(t) for t in t_vals])
r6_vals = []


# 3. closed loop equaition 세우기

delta_theta = np.radians(130)
initial_guess = [np.radians(270), np.radians(0), np.radians(0), 500.0]

def loop_eqs(vars, r2, delta_theta):
    theta3, theta4a, theta5, r6 = vars
    theta4b = theta4a + delta_theta

    eq1 = r2 + 140 * np.cos(theta3) - 250 * np.cos(theta4a) - 52 * np.cos(theta4b)
    eq2 = 179.83 + 140 * np.sin(theta3) - 250 * np.sin(theta4a) - 52 * np.sin(theta4b)
    eq3 = 250 * np.cos(theta4a) + 250 * np.cos(theta5) - r6
    eq4 = 250 * np.sin(theta4a) + 250 * np.sin(theta5)
    return [eq1, eq2, eq3, eq4]

# 4. 시간 루프 시뮬레이션

for r2 in r2_vals:
    try:
        sol = fsolve(loop_eqs, initial_guess, args=(r2, delta_theta))
        theta3, theta4a, theta5, r6 = sol
        r6_vals.append(r6)
        initial_guess = sol
    except:
        r6_vals.append(np.nan)

r6_vals = np.array(r6_vals)


# 5. link 6의 속도 및 가속도 계산

dr6_dt = np.gradient(r6_vals, t_vals)
d2r6_dt2 = np.gradient(dr6_dt, t_vals)




# 속도 극값 표시
min_val = np.min(dr6_dt)
max_val = np.max(dr6_dt)
min_idx = np.argmin(dr6_dt)
max_idx = np.argmax(dr6_dt)
# 가속도 극값도 표시
min_acc_idx = np.argmin(d2r6_dt2)
max_acc_idx = np.argmax(d2r6_dt2)

# 6. 결과 그래프 출력 (개별 그래프)

fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# 위치 그래프
axs[0].plot(t_vals, r6_vals, color='blue')
axs[0].set_title("link6 - Position")
axs[0].set_ylabel("x (mm)")
axs[0].grid(True)

# 위치 극값
min_pos_idx = np.argmin(r6_vals)
max_pos_idx = np.argmax(r6_vals)
axs[0].plot(t_vals[min_pos_idx], r6_vals[min_pos_idx], 'ko')
axs[0].text(t_vals[min_pos_idx], r6_vals[min_pos_idx], f"Min: {r6_vals[min_pos_idx]:.2f}", ha='right', va='top')
axs[0].plot(t_vals[max_pos_idx], r6_vals[max_pos_idx], 'ko')
axs[0].text(t_vals[max_pos_idx], r6_vals[max_pos_idx], f"Max: {r6_vals[max_pos_idx]:.2f}", ha='left', va='bottom')
axs[0].set_xlabel("Time (s)")

# 속도 그래프
axs[1].plot(t_vals, dr6_dt, color='red')
axs[1].set_title("link6 - Velocity")
axs[1].set_ylabel("v (mm/s)")
axs[1].grid(True)
axs[1].set_xlabel("Time (s)")
# 속도 극값
axs[1].plot(t_vals[min_idx], dr6_dt[min_idx], 'ko')
axs[1].text(t_vals[min_idx], dr6_dt[min_idx], f"Min: {dr6_dt[min_idx]:.2f}", ha='right', va='top')
axs[1].plot(t_vals[max_idx], dr6_dt[max_idx], 'ko')
axs[1].text(t_vals[max_idx], dr6_dt[max_idx], f"Max: {dr6_dt[max_idx]:.2f}", ha='left', va='bottom')

# 가속도 그래프
axs[2].plot(t_vals, d2r6_dt2, color='green')
axs[2].set_title("link6 - Acceleration")
axs[2].set_xlabel("time (s)")
axs[2].set_ylabel("a (mm/s²)")
axs[2].grid(True)

# 가속도 극값
axs[2].plot(t_vals[min_acc_idx], d2r6_dt2[min_acc_idx], 'ko')
axs[2].text(t_vals[min_acc_idx], d2r6_dt2[min_acc_idx], f"Min: {d2r6_dt2[min_acc_idx]:.2f}", ha='right', va='top')
axs[2].plot(t_vals[max_acc_idx], d2r6_dt2[max_acc_idx], 'ko')
axs[2].text(t_vals[max_acc_idx], d2r6_dt2[max_acc_idx], f"Max: {d2r6_dt2[max_acc_idx]:.2f}", ha='left', va='bottom')

plt.tight_layout()
plt.show()


# kinematic advantage 구하기
threshold = 1e-3  # v6이 0에 가까운 경우 제외
v2_vals = np.gradient(r2_vals, t_vals)
v6_vals = dr6_dt
valid_idx = np.abs(v6_vals) > threshold
kinematic_advantage = np.full_like(v6_vals, np.nan)
kinematic_advantage[valid_idx] = np.abs(v6_vals[valid_idx] / v2_vals[valid_idx])

# 최대값 계산
max_ma_idx = np.nanargmax(kinematic_advantage)
max_ma = kinematic_advantage[max_ma_idx]
max_time = t_vals[max_ma_idx]

print(f"최대 Kinematic Advantage: {max_ma:.3f} at t = {max_time:.3f} s")


# 그래프 출력

plt.figure(figsize=(10, 5))
plt.plot(t_vals, kinematic_advantage, color='darkgreen', label="Kinematic Advantage (|v6/v2|)")
plt.plot(max_time, max_ma, 'ro', label=f"Max: {max_ma:.2f}")
plt.text(max_time, max_ma, f"  Max: {max_ma:.2f}", va='bottom', ha='left', fontsize=10)
plt.title("Kinematic Advantage over Time")
plt.xlabel("Time (s)")
plt.ylabel("Kinematic Advantage")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
