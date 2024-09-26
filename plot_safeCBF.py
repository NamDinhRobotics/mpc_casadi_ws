import numpy as np
import matplotlib.pyplot as plt

# Tạo các giá trị x và tính f(x)
x_vals = np.linspace(-10, 10, 400)
f_vals = x_vals * np.sin(x_vals)

# Tính sự thay đổi giữa hai điểm gần nhau
dx = 0.5  # Khoảng cách nhỏ giữa các điểm
x1_vals = x_vals[:-int(dx * 80)]  # Loại bỏ các điểm cuối
x2_vals = x1_vals + dx
f1_vals = x1_vals * np.sin(x1_vals)
f2_vals = x2_vals * np.sin(x2_vals)

# Độ dốc tuyệt đối giữa các điểm x1 và x2
lipschitz_approx = np.abs(f2_vals - f1_vals) / dx

# Vẽ đồ thị
plt.figure(figsize=(10, 6))

# Đồ thị hàm f(x)
plt.plot(x_vals, f_vals, label=r'$f(x) = x \cdot \sin(x)$', color='b')

# Đồ thị độ dốc (Lipschitz Approximation)
plt.plot(x1_vals, lipschitz_approx, label='Lipschitz Approximation', color='r', linestyle='--')

plt.title('Chứng minh Lipschitz cho hàm $f(x) = x \cdot \sin(x)$')
plt.xlabel('x')
plt.ylabel('f(x) và độ dốc')
plt.grid(True)
plt.legend()

# Hiển thị đồ thị
plt.show()
