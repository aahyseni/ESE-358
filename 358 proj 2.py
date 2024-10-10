import sys
import numpy as np
import cv2

# Function to map 3D points to 2D using intrinsic, rotation, and translation matrices
def Map2Da(K, R, T, Vi):
    T_transpose = np.transpose(np.atleast_2d(T))
    V_transpose = np.transpose(np.atleast_2d(np.append(Vi, [1])))
    RandTappended = np.append(R, T_transpose, axis=1)
    P = K @ RandTappended @ V_transpose
    P = np.asarray(P).flatten()

    w1 = P[2]
    v = [P[0] / w1, P[1] / w1]  # Return the 2D point (x, y)
    return v

# Function to map mm coordinates to image pixel indices
def MapIndex(u, c0, r0, p):
    v = [None] * 2
    v[0] = round(r0 - u[1] / p)  # Row index
    v[1] = round(c0 + u[0] / p)  # Column index
    return v

# Custom line-drawing function to replace cv2.line()
def draw_line_custom(image, point1, point2, value=255, thickness=1):
    x1, y1 = point1
    x2, y2 = point2
    dx = x2 - x1
    dy = y2 - y1
    d = np.sqrt(dx**2 + dy**2)
    if d == 0:
        return
    unit_vector = (dx / d, dy / d)
    for c in np.arange(0, d, 0.5):
        qx = int(round(x1 + c * unit_vector[0]))
        qy = int(round(y1 + c * unit_vector[1]))
        if 0 <= qx < image.shape[1] and 0 <= qy < image.shape[0]:
            image[qy, qx] = value
        if thickness > 1:
            for t in range(1, thickness):
                offset_x = t * unit_vector[1]
                offset_y = t * -unit_vector[0]
                px = int(round(qx + offset_x))
                py = int(round(qy + offset_y))
                if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
                    image[py, px] = value
                nx = int(round(qx - offset_x))
                ny = int(round(qy - offset_y))
                if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0]:
                    image[ny, nx] = value

# Function to calculate the rotation matrix
def calculate_rotation_matrix(V1, V8, theta):
    N0 = np.array(V8) - np.array(V1)
    N0 = N0 / np.linalg.norm(N0)  # Unit vector N0
    Nx, Ny, Nz = N0
    N = np.array([
        [0, -Nz, Ny],
        [Nz, 0, -Nx],
        [-Ny, Nx, 0]
    ])
    R = np.eye(3) + np.sin(theta) * N + (1 - np.cos(theta)) * (N @ N)
    return R

# Function to calculate unit vectors u21 and u41
def calculate_unit_vectors(V1, V2, V4):
    u21 = np.array(V2) - np.array(V1)
    u21 = u21 / np.linalg.norm(u21)  # Unit vector u21
    u41 = np.array(V4) - np.array(V1)
    u41 = u41 / np.linalg.norm(u41)  # Unit vector u41
    return u21, u41

# Load and use the image provided
background_image = cv2.imread('/Users/aahyseni/Downloads/background.jpg', cv2.IMREAD_GRAYSCALE)
if background_image is None:
    print("Failed to load image.")
else:
    print("Image loaded successfully.")

# Parameters
f = 40  # Focal length in mm
K = np.array([
    [f, 0, 0],
    [0, f, 0],
    [0, 0, 1]
])

V1 = np.array([0, 0, 0])  # Vertex coordinates
V2 = np.array([1, 0, 0])
V4 = np.array([0, 1, 0])
V8 = np.array([1, 1, 1])

# Compute rotation matrix R
theta = np.radians(45)  # Rotation angle in radians
R = calculate_rotation_matrix(V1, V8, theta)

# Translation vector
T = np.array([0, 0, 0])  # No translation

# Compute unit vectors u21 and u41
u21, u41 = calculate_unit_vectors(V1, V2, V4)

# Generate 3D points on the face to be textured
texture_points = []
grid_size = 50
for i in range(grid_size):
    for j in range(grid_size):
        X = V1[0] + i * u21[0] + j * u41[0]
        Y = V1[1] + i * u21[1] + j * u41[1]
        Z = V1[2] + i * u21[2] + j * u41[2]
        texture_points.append([X, Y, Z])

# Image properties
image = np.zeros_like(background_image) if background_image is not None else np.zeros((600, 600), dtype=np.uint8)
c0, r0 = image.shape[1] // 2, image.shape[0] // 2  # Image center
p = 0.1  # Pixel size in mm

# Draw cube edges
cube_edges = [
    (V1, V2), (V2, V8), (V8, V4), (V4, V1),
    # (Add remaining edges for a complete cube)
]
for v1, v2 in cube_edges:
    v1_2d = Map2Da(K, R, T, v1)
    v2_2d = Map2Da(K, R, T, v2)
    point1 = MapIndex(v1_2d, c0, r0, p)
    point2 = MapIndex(v2_2d, c0, r0, p)
    draw_line_custom(image, point1, point2, value=255, thickness=1)

# Overlay the cube on the background image
if background_image is not None:
    combined_image = cv2.addWeighted(background_image, 0.7, image, 0.3, 0)
else:
    combined_image = image

# Display the result
cv2.imshow("Cube Animation", combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
