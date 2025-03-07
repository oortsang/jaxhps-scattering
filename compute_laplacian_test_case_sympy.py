from sympy import symbols, diff, sqrt, atan

##########################################
# 2D Example

print("##########################################")
print("# 2D Example")

# Define symbolic variables
x, y = symbols("x y")

# Define the function
r = sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2)
u = atan(10 * (r - 0.7))

# Calcuate first partial derivatives
u_x = diff(u, x)  # ∂u/∂x
u_y = diff(u, y)  # ∂u/∂y

# Print the results
print("∂u/∂x =")
print(u_x)
print("\n∂u/∂y =")
print(u_y)


# Calculate second partial derivatives
u_xx = diff(u, x, 2)  # ∂²u/∂x²
u_yy = diff(u, y, 2)  # ∂²u/∂y²

# Print the results
print("∂²u/∂x² =")
print(u_xx)
print("\n∂²u/∂y² =")
print(u_yy)

##########################################
# 3D Example

print("##########################################")
print("# 3D Example")

# Define symbolic variables
x, y, z = symbols("x y z")

# Define the function
r = sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2)
u = atan(10 * (r - 0.7))

# Calcuate first partial derivatives
u_x = diff(u, x)  # ∂u/∂x
u_y = diff(u, y)  # ∂u/∂y
u_z = diff(u, z)  # ∂u/∂z

# Print the results
print("∂u/∂x =")
print(u_x)
print("\n∂u/∂y =")
print(u_y)
print("\n∂u/∂z =")
print(u_z)


# Calculate second partial derivatives
u_xx = diff(u, x, 2)  # ∂²u/∂x²
u_yy = diff(u, y, 2)  # ∂²u/∂y²
u_zz = diff(u, z, 2)  # ∂²u/∂z²

# Print the results
print("∂²u/∂x² =")
print(u_xx)
print("\n∂²u/∂y² =")
print(u_yy)
print("\n∂²u/∂z² =")
print(u_zz)
