# CSCE 435 Group project

## 1. Group members:
1. Joseph Buskmiller
2. Fredy Medawar
3. Shreeman Jayaram
4. Ahsan Yahya 

---

## 2. _due 10/25_ Project topic
Performance of different implementations of parallel ray tracing algorithms, including bounding volume hierarchy (BVH) optimizations.

## 2. _due 10/25_ Brief project description (what algorithms will you be comparing and on what architectures)
- 1a Rendering spheres + reflections (CUDA)
- 1b Rendering spheres + reflections (MPI on each core)
- 2a Rendering custom geometry with BVH (CUDA) (tentative)
- 2b Rendering custom geometry with BVH (MPI on each core) (tentative)

## 2. Pseudocode

if not rendering custom mesh
    scene = InitializeScene (including basic shapes and any lights)
else
    scene = LoadModel

camera = InitializeCamera
Allocate memory for frameBuffer
camera.TakePicture(scene)
output image from frameBuffer

function:
Camera TakePicture(scene)
    initialize origin rays
    for i in width
        for j in height
            ComputerRayColor()
            write color to rendered image
