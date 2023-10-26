# CSCE 435 Group project

## 1. Group members:
1. Joseph Buskmiller
2. Fredy Medawar
3. Shreeman Jayaram
4. Ahsan Yahya

Communication will be by discord.

---

## 2. _due 10/25_ Project topic
Performance of different implementations of parallel ray tracing algorithms, including bounding volume hierarchy (BVH) optimizations.

## 2. _due 10/25_ Brief project description (what algorithms will you be comparing and on what architectures)
- Rendering spheres + reflections (CUDA)
- Rendering spheres + reflections (MPI on each core)
- Rendering custom geometry with BVH (CUDA)
- Rendering custom geometry with BVH (MPI on each core)

## 2. Pseudocode
    if not rendering custom mesh
        scene = InitializeScene (including basic shapes and any lights)
    else
        scene = LoadModel

    camera = InitializeCamera
    Allocate memory for frameBuffer
    camera.TakePicture(scene)
    output image from frameBuffer





## Rendering spheres + reflections (CUDA):
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

## Rendering spheres + reflections (MPI on each core):
        DEFINE CHUNKSIZE = 16
        
        //maps a worker to a rectangle on the image
        class workRecord
          int worker;
          int x;
          int y;
          int chunk[CHUNKSIZE * CHUNKSIZE * 3];
          
        
        takePicture(scene, numWorkers)
          initMessagePassing()
        
          vector<workRecord> workRecords;
        
          if(this.rank == 0) 
            ... //slice the image into chunks of size CHUNKSIZE and add elements to workRecords to remember
            
            //since workRecord is really a plain C struct, this should be possible with reinterpret_cast<char*>
            
            for (workRecord r : workRecords)
              sendMessage(r->worker, r)
          
          
          messagePassingBarrier(world) //wait for master to assign work
        
          while(messagesInInbox) 
            recvMessage(master, buf)
            workRecord r = buf;
            workRecords.pushBack(workRecord)
              
          messagePassingBarrier(world) //wait for workers to get their work
        
          //workers do work
          if(this.rank != 0)
            ... //for every pixel of every workrecord, calculate the color and store that back in the chunk field of the workRecord
        
            for(workRecord r : workRecords)
              sendMessage(0, r) 
        
          messagePassingBarrier(world) //wait for workers to finish their work and send it back
        
          //receive messages with finished work from workers
          if(this.rank == 0)
            vector<workRecord> doneRecords;
            while(messagesInInbox)
              recvMessage(null, buf)
              workRecord r = buf;
              doneRecords.pushBack(workRecord)
              
              //great time for a sanity check between workRecords and doneRecords
              
              ... //stitch together image from doneRecords
      


## Rendering custom geometry with BVH (CUDA):
    
      BVH ConstructBVH(scene, bvh, iterations)
        sort morton codes for scene objects
        initialize BVH with N-1 internal nodes and N leaf nodes
        calculate split for internal nodes and assign children <<<blocks, threads>>>
        synchronize()
        return BVH
    
      object BVHIntersect(scene, bvh, origin, direction)
        while hit not detected:
          cast ray against current bvh level
          descend to next bvh level if hit, otherwise return null
    
        return leaf at bvh level
          
    
      Camera ComputeRayColor(scene, bvh, origin, direction, t0, t1, bounces, current)
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            
            for i in range(bounces):
              hit = BVHIntersect(scene, bvh, origin, direction)
              origin = hit.position
              direction = hit.normal
              recalculate color
    
            write color to frameBuffer[x][y]
    
        Camera TakePicture(scene)
            initialize origin rays
    
            num_pixels = width * height
            num_blocks = num_pixels / num_threads
    
            iniialize block size based on number of threads and expected image size
    
            copy scene, camera values, and framebuffer to GPU
    
            ComputeRayColor<<<blocks, threads>>>(scene, origin, direction, 0, inf, bounces, 0)
    
            copy framebuffer from GPU

## Rendering custom geometry with BVH (MPI on each core):
 DEFINE CHUNKSIZE = 16
        
        //maps a worker to a rectangle on the image
        class workRecord
          int worker;
          int x;
          int y;
          int chunk[CHUNKSIZE * CHUNKSIZE * 3];
          
        
        takePicture(scene, numWorkers)
          initMessagePassing()
        
          vector<workRecord> workRecords;
        
          if(this.rank == 0) 
            ... //slice the image into chunks of size CHUNKSIZE and add elements to workRecords to remember
            
            //since workRecord is really a plain C struct, this should be possible with reinterpret_cast<char*>
            
            for (workRecord r : workRecords)
              sendMessage(r->worker, r)
          
          
          messagePassingBarrier(world) //wait for master to assign work
        
          while(messagesInInbox) 
            recvMessage(master, buf)
            workRecord r = buf;
            workRecords.pushBack(workRecord)
              
          messagePassingBarrier(world) //wait for workers to get their work
        
          //workers do work
          if(this.rank != 0)
            triangle res = BVHCollisionTest() //do a test for collision
            ... //for every pixel of every workrecord, calculate the color and store that back in the chunk field of the workRecord
        
            for(workRecord r : workRecords)
              sendMessage(0, r) 
        
          messagePassingBarrier(world) //wait for workers to finish their work and send it back
        
          //receive messages with finished work from workers
          if(this.rank == 0)
            vector<workRecord> doneRecords;
            while(messagesInInbox)
              recvMessage(null, buf)
              workRecord r = buf;
              doneRecords.pushBack(workRecord)
              
              //great time for a sanity check between workRecords and doneRecords
              
              ... //stitch together image from doneRecords
