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


## function 1a:
    Camera ComputeRayColor(scene, origin, direction, t0, t1, bounces, current)
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if ray intersects scene
            for light in lights
                calculate color of pixel

        write color to frameBuffer

    Camera TakePicture(scene)
        initialize origin rays

        num_pixels = width * height
        num_blocks = num_pixels / num_threads

        iniialize block size based on number of threads and expected image size

        copy scene, camera values, and framebuffer to GPU

        ComputeRayColor<<<blocks, threads>>>(scene, origin, direction, 0, inf, bounces, 0)

        copy framebuffer from GPU

## function 1b:


## function 2a:


## function 2b:
