// WebGPU Particle System
// High-performance GPU-accelerated particle simulation

let canvas, context, device, format;
let computePipeline, renderPipeline;
let particleBuffers, uniformBuffer, renderBindGroup, computeBindGroups;
let audioContext = null;
let analyser = null;
let audioData = null;
let audioFreq = null;
let audioSource = null;
let audioStream = null;
let audioReady = false;

// Detect device type
const isIOS = /iPhone|iPad|iPod/i.test(navigator.userAgent);
const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) || window.innerWidth < 768;

// Very conservative particle counts - iOS WebGPU is still experimental
let numParticles = isIOS ? 3000 : (isMobile ? 8000 : 50000);
let currentBuffer = 0;
let isRunning = true;

// Simulation parameters
let params = {
    mouseX: 0,
    mouseY: 0,
    mouseDown: 0,
    deltaTime: 0.016,
    force: 0.5,
    speed: 1.0,
    effectMode: 0, // 0: attract, 1: repel, 2: vortex, 3: gravity, 4: chaos
    colorMode: 0,  // 0: rainbow, 1: fire, 2: ice, 3: neon, 4: galaxy, 5: matrix
    particleSize: 3.0,
    trailFade: 0.95,
    audioMode: 0,
    audioBoost: 1.0,
    audioLevel: 0.0,
    audioBass: 0.0,
    audioMid: 0.0,
    audioTreble: 0.0
};

// FPS tracking
let lastTime = performance.now();
let frameCount = 0;
let fps = 0;

// Check WebGPU support
async function initWebGPU() {
    if (!navigator.gpu) {
        document.getElementById('error-message').style.display = 'block';
        document.getElementById('controls').style.display = 'none';
        document.getElementById('stats').style.display = 'none';
        throw new Error('WebGPU not supported');
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        document.getElementById('error-message').style.display = 'block';
        throw new Error('No GPU adapter found');
    }

    device = await adapter.requestDevice();
    canvas = document.getElementById('canvas');
    context = canvas.getContext('webgpu');

    format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device,
        format,
        alphaMode: 'premultiplied'
    });

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    await createPipelines();
    createParticleBuffers();
    setupEventListeners();
    setupControls();

    requestAnimationFrame(render);
}

function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = window.innerWidth * dpr;
    canvas.height = window.innerHeight * dpr;
}

async function createPipelines() {
    // Compute shader for particle simulation
    const computeShaderCode = `
        struct Particle {
            pos: vec2f,
            vel: vec2f,
            life: f32,
            seed: f32
        }

        struct Params {
            mouse: vec2f,
            resolution: vec2f,
            deltaTime: f32,
            force: f32,
            speed: f32,
            effectMode: f32,
            colorMode: f32,
            particleSize: f32,
            mouseDown: f32,
            time: f32,
            audioLevel: f32,
            audioBass: f32,
            audioMid: f32,
            audioTreble: f32
        }

        @group(0) @binding(0) var<storage, read> particlesIn: array<Particle>;
        @group(0) @binding(1) var<storage, read_write> particlesOut: array<Particle>;
        @group(0) @binding(2) var<uniform> params: Params;

        fn hash(n: f32) -> f32 {
            return fract(sin(n) * 43758.5453123);
        }

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            let i = id.x;
            if (i >= arrayLength(&particlesIn)) {
                return;
            }

            var p = particlesIn[i];
            let mouse = params.mouse;
            let res = params.resolution;
            let dt = params.deltaTime * params.speed;
            let audio = params.audioLevel;
            let bass = params.audioBass;
            let mid = params.audioMid;
            let treble = params.audioTreble;
            let audioPush = 1.0 + audio * 2.5;
            let force = params.force;
            let effectMode = i32(params.effectMode);
            
            // Calculate direction to audio attractor
            let toMouse = mouse - p.pos;
            let dist = length(toMouse);
            let dir = normalize(toMouse);
            
            // Apply force based on effect mode
            var accel = vec2f(0.0);
            
            if (effectMode == 0) {
                // Attract
                if (params.mouseDown > 0.5) {
                    accel = dir * force * 500.0 / (dist + 50.0);
                }
            } else if (effectMode == 1) {
                // Repel
                if (params.mouseDown > 0.5) {
                    accel = -dir * force * 800.0 / (dist + 30.0);
                }
            } else if (effectMode == 2) {
                // Vortex
                let perpDir = vec2f(-dir.y, dir.x);
                if (params.mouseDown > 0.5) {
                    accel = perpDir * force * 600.0 / (dist + 40.0);
                    accel += dir * force * 100.0 / (dist + 100.0);
                }
            } else if (effectMode == 3) {
                // Gravity
                accel.y = force * 200.0;
                if (params.mouseDown > 0.5) {
                    accel += dir * force * 300.0 / (dist + 50.0);
                }
            } else if (effectMode == 4) {
                // Chaos
                let noise = hash(p.seed + params.time * 0.01);
                accel = vec2f(
                    sin(noise * 6.28 + params.time) * force * 200.0,
                    cos(noise * 6.28 + params.time * 1.3) * force * 200.0
                );
                if (params.mouseDown > 0.5) {
                    accel += dir * force * 400.0 / (dist + 50.0);
                }
            }

            // Audio-driven swirl and pulse
            let swirl = vec2f(-dir.y, dir.x) * (treble * 320.0 + audio * 140.0);
            accel += swirl;
            accel += dir * (bass * 620.0);

            // Global flow field to prevent collapse
            let n1 = hash(p.seed * 91.7 + params.time * 0.13);
            let n2 = hash(p.seed * 37.1 + params.time * 0.17 + 3.1);
            let flow = vec2f(sin(n1 * 6.283 + params.time), cos(n2 * 6.283 + params.time * 1.2));
            accel += flow * (120.0 + audio * 220.0);

            // Soft repulsion from center for visual spread
            let center = res * 0.5;
            let fromCenter = p.pos - center;
            let centerDist = length(fromCenter) + 0.001;
            accel += normalize(fromCenter) * (20.0 + audio * 80.0) / (centerDist * 0.02 + 1.0);

            // Update velocity with damping
            p.vel += accel * dt * audioPush;
            p.vel *= mix(0.987, 0.95, audio);
            
            // Limit velocity
            let speed = length(p.vel);
            if (speed > 500.0) {
                p.vel = normalize(p.vel) * 500.0;
            }

            // Update position
            p.pos += p.vel * dt;

            // Boundary wrapping
            if (p.pos.x < 0.0) { p.pos.x += res.x; }
            if (p.pos.x > res.x) { p.pos.x -= res.x; }
            if (p.pos.y < 0.0) { p.pos.y += res.y; }
            if (p.pos.y > res.y) { p.pos.y -= res.y; }

            // Update life based on speed
            p.life = min(1.0, length(p.vel) / 200.0 + 0.3 + audio * 0.4);

            particlesOut[i] = p;
        }
    `;

    // Render shader for particles
    const renderShaderCode = `
        struct Params {
            mouse: vec2f,
            resolution: vec2f,
            deltaTime: f32,
            force: f32,
            speed: f32,
            effectMode: f32,
            colorMode: f32,
            particleSize: f32,
            mouseDown: f32,
            time: f32,
            audioLevel: f32,
            audioBass: f32,
            audioMid: f32,
            audioTreble: f32
        }

        struct Particle {
            pos: vec2f,
            vel: vec2f,
            life: f32,
            seed: f32
        }

        struct VertexOutput {
            @builtin(position) position: vec4f,
            @location(0) color: vec4f,
            @location(1) pointCoord: vec2f
        }

        @group(0) @binding(0) var<storage, read> particles: array<Particle>;
        @group(0) @binding(1) var<uniform> params: Params;

        fn hue2rgb(h: f32) -> vec3f {
            let r = abs(h * 6.0 - 3.0) - 1.0;
            let g = 2.0 - abs(h * 6.0 - 2.0);
            let b = 2.0 - abs(h * 6.0 - 4.0);
            return clamp(vec3f(r, g, b), vec3f(0.0), vec3f(1.0));
        }

        fn getColor(seed: f32, life: f32, colorMode: i32) -> vec3f {
            if (colorMode == 0) {
                // Rainbow
                return hue2rgb(fract(seed + life * 0.5));
            } else if (colorMode == 1) {
                // Fire
                let t = life;
                return vec3f(1.0, t * 0.6, t * t * 0.2);
            } else if (colorMode == 2) {
                // Ice
                let t = life;
                return vec3f(0.5 + t * 0.5, 0.8 + t * 0.2, 1.0);
            } else if (colorMode == 3) {
                // Neon
                let h = fract(seed * 3.0);
                if (h < 0.33) {
                    return vec3f(1.0, 0.0, 0.8) * (0.5 + life * 0.5);
                } else if (h < 0.66) {
                    return vec3f(0.0, 1.0, 0.8) * (0.5 + life * 0.5);
                } else {
                    return vec3f(0.8, 0.0, 1.0) * (0.5 + life * 0.5);
                }
            } else if (colorMode == 4) {
                // Galaxy
                let t = fract(seed * 2.0);
                if (t < 0.3) {
                    return vec3f(1.0, 0.9, 0.7) * life; // Stars
                } else if (t < 0.6) {
                    return vec3f(0.5, 0.3, 0.8) * life; // Purple nebula
                } else {
                    return vec3f(0.2, 0.4, 0.9) * life; // Blue
                }
            } else {
                // Matrix
                return vec3f(0.0, life, 0.0);
            }
        }

        @vertex
        fn vertexMain(
            @builtin(vertex_index) vertexIndex: u32,
            @builtin(instance_index) instanceIndex: u32
        ) -> VertexOutput {
            let particle = particles[instanceIndex];
            let pos = particle.pos;
            let life = particle.life;
            let seed = particle.seed;

            // Quad vertices
            let quadPos = array<vec2f, 6>(
                vec2f(-1.0, -1.0),
                vec2f(1.0, -1.0),
                vec2f(-1.0, 1.0),
                vec2f(-1.0, 1.0),
                vec2f(1.0, -1.0),
                vec2f(1.0, 1.0)
            );

            let size = params.particleSize * (0.5 + life * 0.5 + params.audioLevel * 0.8);
            let vertPos = quadPos[vertexIndex] * size;

            var output: VertexOutput;
            output.position = vec4f(
                (pos.x + vertPos.x) / params.resolution.x * 2.0 - 1.0,
                1.0 - (pos.y + vertPos.y) / params.resolution.y * 2.0,
                0.0,
                1.0
            );

            let colorMode = i32(params.colorMode);
            let color = getColor(seed, life, colorMode);
            let pulse = 0.6 + params.audioLevel * 0.8 + params.audioTreble * 0.6;
            output.color = vec4f(color * pulse, life * (0.7 + params.audioLevel * 0.6));
            output.pointCoord = quadPos[vertexIndex] * 0.5 + 0.5;

            return output;
        }

        @fragment
        fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
            // Circular particle with soft edges
            let dist = length(input.pointCoord - vec2f(0.5));
            if (dist > 0.5) {
                discard;
            }
            let alpha = input.color.a * (1.0 - dist * 2.0);
            return vec4f(input.color.rgb * alpha, alpha);
        }
    `;

    // Create compute pipeline
    const computeShaderModule = device.createShaderModule({ code: computeShaderCode });
    computePipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: computeShaderModule,
            entryPoint: 'main'
        }
    });

    // Create render pipeline
    const renderShaderModule = device.createShaderModule({ code: renderShaderCode });
    renderPipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: {
            module: renderShaderModule,
            entryPoint: 'vertexMain'
        },
        fragment: {
            module: renderShaderModule,
            entryPoint: 'fragmentMain',
            targets: [{
                format,
                blend: {
                    color: {
                        srcFactor: 'src-alpha',
                        dstFactor: 'one',
                        operation: 'add'
                    },
                    alpha: {
                        srcFactor: 'one',
                        dstFactor: 'one',
                        operation: 'add'
                    }
                }
            }]
        },
        primitive: {
            topology: 'triangle-list'
        }
    });
}

function createParticleBuffers() {
    // Initialize particle data
    const particleData = new Float32Array(numParticles * 6); // pos(2) + vel(2) + life(1) + seed(1)
    
    for (let i = 0; i < numParticles; i++) {
        const idx = i * 6;
        particleData[idx] = Math.random() * canvas.width;     // pos.x
        particleData[idx + 1] = Math.random() * canvas.height; // pos.y
        particleData[idx + 2] = (Math.random() - 0.5) * 100;   // vel.x
        particleData[idx + 3] = (Math.random() - 0.5) * 100;   // vel.y
        particleData[idx + 4] = Math.random();                 // life
        particleData[idx + 5] = Math.random();                 // seed
    }

    // Create double buffers for ping-pong
    particleBuffers = [
        device.createBuffer({
            size: particleData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        }),
        device.createBuffer({
            size: particleData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        })
    ];

    new Float32Array(particleBuffers[0].getMappedRange()).set(particleData);
    particleBuffers[0].unmap();

    // Create uniform buffer
    uniformBuffer = device.createBuffer({
        size: 64, // 16 floats * 4 bytes
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    // Create bind groups for compute
    computeBindGroups = [
        device.createBindGroup({
            layout: computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: particleBuffers[0] } },
                { binding: 1, resource: { buffer: particleBuffers[1] } },
                { binding: 2, resource: { buffer: uniformBuffer } }
            ]
        }),
        device.createBindGroup({
            layout: computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: particleBuffers[1] } },
                { binding: 1, resource: { buffer: particleBuffers[0] } },
                { binding: 2, resource: { buffer: uniformBuffer } }
            ]
        })
    ];

    updateRenderBindGroup();
}

function updateRenderBindGroup() {
    renderBindGroup = device.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: particleBuffers[currentBuffer] } },
            { binding: 1, resource: { buffer: uniformBuffer } }
        ]
    });
}

function setupEventListeners() {
    // Mouse/touch input intentionally disabled for audio-driven mode
}

function updateMousePosition(x, y) {
    const dpr = window.devicePixelRatio || 1;
    params.mouseX = x * dpr;
    params.mouseY = y * dpr;
}

function setupControls() {
    // Particle count - adjust for device
    const countSlider = document.getElementById('particle-count');
    const countValue = document.getElementById('count-value');
    
    if (isIOS) {
        countSlider.max = 10000;
        countSlider.min = 500;
        countSlider.step = 500;
        countSlider.value = numParticles;
        countValue.textContent = numParticles;
    } else if (isMobile) {
        countSlider.max = 30000;
        countSlider.value = numParticles;
        countValue.textContent = numParticles;
    }
    
    countSlider.addEventListener('input', (e) => {
        countValue.textContent = e.target.value;
    });
    countSlider.addEventListener('change', (e) => {
        numParticles = parseInt(e.target.value);
        createParticleBuffers();
    });

    // Color mode
    document.getElementById('color-mode').addEventListener('change', (e) => {
        const modes = { rainbow: 0, fire: 1, ice: 2, neon: 3, galaxy: 4, matrix: 5 };
        params.colorMode = modes[e.target.value];
    });

    // Effect mode
    document.getElementById('effect-mode').addEventListener('change', (e) => {
        const modes = { attract: 0, repel: 1, vortex: 2, gravity: 3, chaos: 4 };
        params.effectMode = modes[e.target.value];
    });

    // Force
    const forceSlider = document.getElementById('force');
    const forceValue = document.getElementById('force-value');
    forceSlider.addEventListener('input', (e) => {
        params.force = parseFloat(e.target.value);
        forceValue.textContent = params.force.toFixed(2);
    });

    // Particle size
    const sizeSlider = document.getElementById('particle-size');
    const sizeValue = document.getElementById('size-value');
    sizeSlider.addEventListener('input', (e) => {
        params.particleSize = parseFloat(e.target.value);
        sizeValue.textContent = params.particleSize.toFixed(1);
    });

    // Speed
    const speedSlider = document.getElementById('speed');
    const speedValue = document.getElementById('speed-value');
    speedSlider.addEventListener('input', (e) => {
        params.speed = parseFloat(e.target.value);
        speedValue.textContent = params.speed.toFixed(1);
    });

    // Trail
    const trailSlider = document.getElementById('trail');
    const trailValue = document.getElementById('trail-value');
    trailSlider.addEventListener('input', (e) => {
        params.trailFade = parseFloat(e.target.value);
        trailValue.textContent = params.trailFade.toFixed(2);
    });

    // Audio mode
    const audioMode = document.getElementById('audio-mode');
    audioMode.addEventListener('change', async (e) => {
        params.audioMode = e.target.value === 'on' ? 1 : 0;
        if (params.audioMode === 1) {
            await startAudio();
        } else {
            stopAudio();
        }
    });

    // Audio boost
    const audioBoost = document.getElementById('audio-boost');
    const audioBoostValue = document.getElementById('audio-boost-value');
    audioBoost.addEventListener('input', (e) => {
        params.audioBoost = parseFloat(e.target.value);
        audioBoostValue.textContent = params.audioBoost.toFixed(1);
    });
}

let time = 0;

async function startAudio() {
    if (!window.isSecureContext) {
        alert('Microphone requires HTTPS or localhost.');
        params.audioMode = 0;
        document.getElementById('audio-mode').value = 'off';
        return;
    }
    if (audioReady) return;
    try {
        audioStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 2048;
        analyser.smoothingTimeConstant = 0.8;
        audioData = new Uint8Array(analyser.fftSize);
        audioFreq = new Uint8Array(analyser.frequencyBinCount);
        audioSource = audioContext.createMediaStreamSource(audioStream);
        audioSource.connect(analyser);
        audioReady = true;
    } catch (error) {
        console.error('Audio init error:', error);
        params.audioMode = 0;
        document.getElementById('audio-mode').value = 'off';
        alert('Microphone permission denied or unavailable.');
    }
}

function stopAudio() {
    audioReady = false;
    if (audioSource) {
        audioSource.disconnect();
        audioSource = null;
    }
    if (audioStream) {
        audioStream.getTracks().forEach((t) => t.stop());
        audioStream = null;
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
    analyser = null;
    audioData = null;
    audioFreq = null;
    params.audioLevel = 0;
    params.audioBass = 0;
    params.audioMid = 0;
    params.audioTreble = 0;
}

function updateAudio() {
    if (!audioReady || !analyser) {
        params.audioLevel = 0;
        params.audioBass = 0;
        params.audioMid = 0;
        params.audioTreble = 0;
        return;
    }

    analyser.getByteTimeDomainData(audioData);
    analyser.getByteFrequencyData(audioFreq);

    // RMS volume
    let sum = 0;
    for (let i = 0; i < audioData.length; i++) {
        const v = (audioData[i] - 128) / 128;
        sum += v * v;
    }
    let level = Math.sqrt(sum / audioData.length);

    // Frequency bands
    const bandCount = audioFreq.length;
    const bassEnd = Math.floor(bandCount * 0.1);
    const midEnd = Math.floor(bandCount * 0.4);
    let bass = 0, mid = 0, treble = 0;

    for (let i = 0; i < bassEnd; i++) bass += audioFreq[i];
    for (let i = bassEnd; i < midEnd; i++) mid += audioFreq[i];
    for (let i = midEnd; i < bandCount; i++) treble += audioFreq[i];

    bass /= Math.max(1, bassEnd);
    mid /= Math.max(1, midEnd - bassEnd);
    treble /= Math.max(1, bandCount - midEnd);

    const boost = params.audioBoost;
    params.audioLevel = Math.min(1, level * 2.5 * boost);
    params.audioBass = Math.min(1, (bass / 255) * 2.2 * boost);
    params.audioMid = Math.min(1, (mid / 255) * 1.8 * boost);
    params.audioTreble = Math.min(1, (treble / 255) * 1.6 * boost);
}

function render() {
    if (!isRunning) return;
    
    try {
        const now = performance.now();
        const deltaTime = (now - lastTime) / 1000;
        lastTime = now;
        time += deltaTime;

        // FPS calculation
        frameCount++;
        if (frameCount >= 30) {
            fps = Math.round(frameCount / (deltaTime * 30));
            frameCount = 0;
            document.getElementById('fps').textContent = fps;
            document.getElementById('particle-stat').textContent = numParticles.toLocaleString();
        }

    // Audio
    updateAudio();

    // Audio-driven virtual attractor (always on)
    let mouseXForSim = canvas.width * 0.5;
    let mouseYForSim = canvas.height * 0.5;
    let mouseDownForSim = 0.0;
    if (params.audioMode === 1) {
        const cx = canvas.width * 0.5;
        const cy = canvas.height * 0.5;
        const wobbleX = (params.audioTreble - 0.5) * 260 + Math.sin(time * 1.7) * 160;
        const wobbleY = (params.audioBass - 0.5) * 260 + Math.cos(time * 1.3) * 160;
        mouseXForSim = cx + wobbleX * (0.5 + params.audioLevel);
        mouseYForSim = cy + wobbleY * (0.5 + params.audioLevel);
        mouseDownForSim = 1.0;
    }

    // Update uniforms
    const uniformData = new Float32Array([
        mouseXForSim, mouseYForSim,
        canvas.width, canvas.height,
        Math.min(deltaTime, 0.05),
        params.force,
        params.speed,
        params.effectMode,
        params.colorMode,
        params.particleSize,
        mouseDownForSim,
        time,
        params.audioLevel,
        params.audioBass,
        params.audioMid,
        params.audioTreble
    ]);
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const commandEncoder = device.createCommandEncoder();

    // Compute pass
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(computePipeline);
    computePass.setBindGroup(0, computeBindGroups[currentBuffer]);
    computePass.dispatchWorkgroups(Math.ceil(numParticles / 256));
    computePass.end();

    // Swap buffers
    currentBuffer = 1 - currentBuffer;
    updateRenderBindGroup();

    // Render pass
    const textureView = context.getCurrentTexture().createView();
    const renderPass = commandEncoder.beginRenderPass({
        colorAttachments: [{
            view: textureView,
            clearValue: { r: 0, g: 0, b: 0.02, a: 1 },
            loadOp: 'clear',
            storeOp: 'store'
        }]
    });

    renderPass.setPipeline(renderPipeline);
    renderPass.setBindGroup(0, renderBindGroup);
    renderPass.draw(6, numParticles);
    renderPass.end();

    device.queue.submit([commandEncoder.finish()]);

    requestAnimationFrame(render);
    } catch (error) {
        console.error('Render error:', error);
        isRunning = false;
        document.getElementById('error-message').innerHTML = `
            <h2>⚠️ Rendering Error</h2>
            <p>${error.message}</p>
            <p style="margin-top: 15px;"><button onclick="location.reload()">Reload Page</button></p>
        `;
        document.getElementById('error-message').style.display = 'block';
    }
}

// Global functions for buttons
function resetParticles() {
    createParticleBuffers();
}

function explode() {
    // Create explosion by setting high velocities
    const particleData = new Float32Array(numParticles * 6);
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;

    for (let i = 0; i < numParticles; i++) {
        const idx = i * 6;
        const angle = Math.random() * Math.PI * 2;
        const speed = 200 + Math.random() * 300;
        
        particleData[idx] = centerX + (Math.random() - 0.5) * 50;
        particleData[idx + 1] = centerY + (Math.random() - 0.5) * 50;
        particleData[idx + 2] = Math.cos(angle) * speed;
        particleData[idx + 3] = Math.sin(angle) * speed;
        particleData[idx + 4] = 1.0;
        particleData[idx + 5] = Math.random();
    }

    device.queue.writeBuffer(particleBuffers[currentBuffer], 0, particleData);
}

function toggleControls() {
    const controls = document.getElementById('controls');
    controls.classList.toggle('hidden');
}

function stopSimulation() {
    isRunning = false;
    document.getElementById('fps').textContent = 'Stopped';
}

// Mobile: toggle controls on drag handle
document.addEventListener('DOMContentLoaded', () => {
    const dragHandle = document.querySelector('.drag-handle');
    if (dragHandle) {
        dragHandle.addEventListener('click', toggleControls);
    }
});

// Initialize
initWebGPU().catch(console.error);
