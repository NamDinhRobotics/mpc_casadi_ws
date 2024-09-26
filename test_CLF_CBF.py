import matplotlib.pyplot as plt
import numpy as np
import casadi as ca

# Khởi tạo các tham số robot
L = 2.0  # chiều dài cơ sở của xe
v_max = 5.0  # vận tốc tối đa
phi_max = np.pi / 6  # góc lái tối đa (radians)

# Các tham số cho bài toán
x_goal = 10.0  # tọa độ x của đích
y_goal = 10.0  # tọa độ y của đích
x_obs = 5.0  # tọa độ x của chướng ngại vật
y_obs = 5.0  # tọa độ y của chướng ngại vật
r_safe = 1.5  # bán kính an toàn xung quanh chướng ngại vật
delta_max = 0.1  # biên bảo thủ thêm vào bán kính an toàn

# Các hệ số CLF và CBF
c1 = 1.0  # hệ số CLF
gamma = 1.0  # hệ số CBF
p_relax = 1000  # hệ số phạt biến thư giãn

# Khởi tạo CasADi variables
x = ca.MX.sym('x')  # vị trí x của robot
y = ca.MX.sym('y')  # vị trí y của robot
theta = ca.MX.sym('theta')  # góc định hướng của robot
v = ca.MX.sym('v')  # vận tốc
a = ca.MX.sym('a')  # gia tốc (biến tối ưu)
phi = ca.MX.sym('phi')  # góc lái (biến tối ưu)
delta = ca.MX.sym('delta')  # biến thư giãn (biến tối ưu)

# Động lực học robot
x_dot = v * ca.cos(theta)
y_dot = v * ca.sin(theta)
theta_dot = (v / L) * ca.tan(phi)

# Hàm Lyapunov V(x, y) để tiến về đích
V = 0.5 * ((x - x_goal) ** 2 + (y - y_goal) ** 2)

# Đạo hàm Lyapunov theo thời gian
V_dot = (x - x_goal) * x_dot + (y - y_goal) * y_dot

# Ràng buộc từ aCLF (tiến về đích)
clf_constraint = V_dot + c1 * V

# Hàm rào chắn an toàn CBF
h = (x - x_obs) ** 2 + (y - y_obs) ** 2 - (r_safe + delta_max) ** 2

# Đạo hàm CBF theo thời gian
h_dot = 2 * (x - x_obs) * x_dot + 2 * (y - y_obs) * y_dot

# Ràng buộc từ CBF (tránh va chạm với chướng ngại vật, có thêm biến thư giãn delta)
cbf_constraint = h_dot + gamma * h + delta

# Hàm mục tiêu tối thiểu hóa gia tốc, góc lái và biến thư giãn
objective = 0.5 * (a ** 2 + phi ** 2 + p_relax * delta ** 2)

# Tạo bài toán tối ưu hóa CasADi
opti = ca.Opti()

# Thêm biến tối ưu
a_var = opti.variable()  # gia tốc tối ưu
phi_var = opti.variable()  # góc lái tối ưu
delta_var = opti.variable()  # biến thư giãn tối ưu

# Các biến trạng thái của robot
x_var = opti.parameter()
y_var = opti.parameter()
theta_var = opti.parameter()
v_var = opti.parameter()

# Ràng buộc tối thiểu hóa
opti.minimize(objective)

# Ràng buộc động học robot
opti.subject_to(phi_var <= phi_max)
opti.subject_to(phi_var >= -phi_max)
opti.subject_to(v_var <= v_max)

# Ràng buộc aCLF
opti.subject_to(clf_constraint <= 0)

# Ràng buộc CBF
opti.subject_to(cbf_constraint >= 0)

# Giới hạn biến thư giãn delta
opti.subject_to(delta_var >= 0)

# Cài đặt solver
opti.solver('ipopt')

# Các thông số mô phỏng
T = 20  # số bước thời gian
dt = 0.1  # thời gian bước

# Trạng thái ban đầu
x0, y0, theta0, v0 = 0.0, 0.0, np.pi / 4, 1.0
x_traj = [x0]
y_traj = [y0]

# Mô phỏng hệ thống qua các bước thời gian
for t in range(T):
    # Gán giá trị cho các tham số trạng thái
    opti.set_value(x_var, x0)
    opti.set_value(y_var, y0)
    opti.set_value(theta_var, theta0)
    opti.set_value(v_var, v0)

    # Giải bài toán tối ưu hóa
    sol = opti.solve()

    # Lấy kết quả tối ưu
    a_opt = sol.value(a_var)
    phi_opt = sol.value(phi_var)
    delta_opt = sol.value(delta_var)

    # Cập nhật trạng thái robot
    x0 = x0 + v0 * np.cos(theta0) * dt
    y0 = y0 + v0 * np.sin(theta0) * dt
    theta0 = theta0 + (v0 / L) * np.tan(phi_opt) * dt
    v0 = v0 + a_opt * dt

    # Lưu lại quỹ đạo
    x_traj.append(x0)
    y_traj.append(y0)

# Vẽ quỹ đạo của robot và chướng ngại vật
fig, ax = plt.subplots()
ax.plot(x_traj, y_traj, label="Quỹ đạo robot", marker="o")
obs = plt.Circle((x_obs, y_obs), r_safe + delta_max, color="r", alpha=0.5)
ax.add_artist(obs)
ax.plot(x_goal, y_goal, 'gx', label="Vị trí đích", markersize=10)
ax.set_xlim(-2, 12)
ax.set_ylim(-2, 12)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Quỹ đạo robot với tránh chướng ngại vật")
ax.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.show()

