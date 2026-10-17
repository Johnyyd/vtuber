// VTuber 3D Avatar Viewport & Motion Interpolation Engine

(function () {
  "use strict";

  let scene, camera, renderer;
  let currentVrm = null;
  let mouthMesh = null;
  let headBone = null;
  let neckBone = null;
  let spineBone = null;
  let chestBone = null;
  let upperChestBone = null;
  let hipsBone = null;
  let eyeLeftBone = null;
  let eyeRightBone = null;
  let leftUpperArmBone = null;
  let rightUpperArmBone = null;
  let leftLowerArmBone = null;
  let rightLowerArmBone = null;
  const clock = new THREE.Clock();

  const morphMeshes = [];

  // Motion target and current interpolated values
  const targetMotion = {
    vrm: {
      neutral: 1.0,
      a: 0.0,
      i: 0.0,
      u: 0.0,
      e: 0.0,
      o: 0.0,
      blink: 0.0,
      blink_l: 0.0,
      blink_r: 0.0,
      joy: 0.0,
      angry: 0.0,
      sorrow: 0.0,
      fun: 0.0,
      surprised: 0.0,
    },
    vrc: {
      vrc_blink: 0.0,
      vrc_v_aa: 0.0,
      vrc_v_oh: 0.0,
      vrc_v_ou: 0.0,
      vrc_v_ee: 0.0,
      vrc_v_ih: 0.0,
      vrc_v_sil: 1.0,
      vrc_v_ch: 0.0,
      vrc_v_dd: 0.0,
      vrc_v_ff: 0.0,
      vrc_v_kk: 0.0,
      vrc_v_nn: 0.0,
      vrc_v_pp: 0.0,
      vrc_v_rr: 0.0,
      vrc_v_ss: 0.0,
      vrc_v_th: 0.0,
    },
    rotation: {
      pitch: 0.0,
      yaw: 0.0,
      roll: 0.0,
    },
    gaze: {
      x: 0.0,
      y: 0.0,
    },
    body: {
      sway_gain: 1.0,
      breath_gain: 1.0,
      hair_gravity: 0.15,
      hair_stiffness: 0.95,
      hair_drag: 0.50,
    },
  };

  const currentMotion = {
    vrm: { ...targetMotion.vrm },
    vrc: { ...targetMotion.vrc },
    rotation: { ...targetMotion.rotation },
    gaze: { ...targetMotion.gaze },
    body: { ...targetMotion.body },
  };

  let lastMotionTimestamp = 0;
  let frameCount = 0;
  let lastFpsTime = performance.now();

  // Blink State Machine
  let blinkState = "IDLE"; // IDLE, HOLD, COOLDOWN
  let blinkTimer = 0;
  let activeBlinkType = null;

  function init() {
    const container = document.getElementById("canvas-container");
    const canvas = document.getElementById("avatar-canvas");

    // 1. Scene setup
    scene = new THREE.Scene();

    // 2. Camera setup - focused on avatar upper body with ample headroom for hair/forehead
    const aspect = window.innerWidth / window.innerHeight;
    camera = new THREE.PerspectiveCamera(30.0, aspect, 0.1, 20.0);
    camera.position.set(0.0, 1.40, 1.25);
    camera.lookAt(0.0, 1.34, 0.0);

    // 3. Renderer setup
    renderer = new THREE.WebGLRenderer({
      canvas: canvas,
      antialias: true,
      alpha: true,
      powerPreference: "high-performance",
    });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2.0));
    renderer.outputEncoding = THREE.sRGBEncoding;

    // 4. Lighting setup
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xffffff, 0.9);
    keyLight.position.set(1.0, 2.0, 1.5);
    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0xb0c4de, 0.45);
    fillLight.position.set(-1.5, 1.0, 1.0);
    scene.add(fillLight);

    const rimLight = new THREE.DirectionalLight(0xfff5ee, 0.55);
    rimLight.position.set(0.0, 2.0, -1.5);
    scene.add(rimLight);

    window.addEventListener("resize", onWindowResize);

    // 5. Setup window size customization controls
    setupWindowSizeControls();

    // 6. Load VRM Model (Arisa)
    loadModel(["Arisa.vrm", "../assets/Arisa.vrm", "assets/Arisa.vrm"]);

    // 7. Start Render Loop
    requestAnimationFrame(animate);
  }

  // Window size customization & IPC
  function requestWindowResize(width, height) {
    const w = Math.max(300, Math.min(3840, parseInt(width, 10) || 1024));
    const h = Math.max(200, Math.min(2160, parseInt(height, 10) || 768));
    document.title = `vtuber:resize:${w}:${h}:${Date.now()}`;
  }

  function setupWindowSizeControls() {
    const presetSelect = document.getElementById("win-preset-select");
    const widthInput = document.getElementById("win-width-input");
    const heightInput = document.getElementById("win-height-input");
    const applyBtn = document.getElementById("win-apply-btn");
    const resizeHandle = document.getElementById("win-resize-handle");

    if (!presetSelect || !widthInput || !heightInput) return;

    function syncInputsToCurrentSize() {
      const curW = window.innerWidth;
      const curH = window.innerHeight;
      widthInput.value = curW;
      heightInput.value = curH;

      const matchingPreset = `${curW}x${curH}`;
      let hasMatch = false;
      for (let i = 0; i < presetSelect.options.length; i++) {
        if (presetSelect.options[i].value === matchingPreset) {
          presetSelect.selectedIndex = i;
          hasMatch = true;
          break;
        }
      }
      if (!hasMatch) {
        presetSelect.value = "custom";
      }
    }

    // Initialize with current window dimensions
    syncInputsToCurrentSize();

    // Handle preset selection
    presetSelect.addEventListener("change", () => {
      const val = presetSelect.value;
      if (val === "custom") {
        widthInput.focus();
        widthInput.select();
        return;
      }
      const [w, h] = val.split("x").map((n) => parseInt(n, 10));
      if (w && h) {
        widthInput.value = w;
        heightInput.value = h;
        requestWindowResize(w, h);
      }
    });

    // Handle manual input apply
    function applyCustomSize() {
      const w = parseInt(widthInput.value, 10);
      const h = parseInt(heightInput.value, 10);
      if (w && h) {
        requestWindowResize(w, h);
        const matchingPreset = `${w}x${h}`;
        let hasMatch = false;
        for (let i = 0; i < presetSelect.options.length; i++) {
          if (presetSelect.options[i].value === matchingPreset) {
            presetSelect.selectedIndex = i;
            hasMatch = true;
            break;
          }
        }
        if (!hasMatch) {
          presetSelect.value = "custom";
        }
      }
    }

    if (applyBtn) {
      applyBtn.addEventListener("click", applyCustomSize);
    }

    widthInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") applyCustomSize();
    });
    heightInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") applyCustomSize();
    });

    // Handle drag resize handle
    if (resizeHandle) {
      let isDragging = false;
      let startX = 0;
      let startY = 0;
      let startW = 0;
      let startH = 0;

      resizeHandle.addEventListener("mousedown", (e) => {
        isDragging = true;
        startX = e.screenX;
        startY = e.screenY;
        startW = window.innerWidth;
        startH = window.innerHeight;
        e.preventDefault();
      });

      window.addEventListener("mousemove", (e) => {
        if (!isDragging) return;
        const deltaX = e.screenX - startX;
        const deltaY = e.screenY - startY;
        const newW = Math.max(300, Math.min(3840, startW + deltaX));
        const newH = Math.max(200, Math.min(2160, startH + deltaY));
        widthInput.value = newW;
        heightInput.value = newH;
        requestWindowResize(newW, newH);
      });

      window.addEventListener("mouseup", () => {
        if (isDragging) {
          isDragging = false;
          syncInputsToCurrentSize();
        }
      });
    }

    // Expose callback for Python resizeEvent
    window.onWindowResized = function (w, h) {
      widthInput.value = w;
      heightInput.value = h;
      const matchingPreset = `${w}x${h}`;
      let hasMatch = false;
      for (let i = 0; i < presetSelect.options.length; i++) {
        if (presetSelect.options[i].value === matchingPreset) {
          presetSelect.selectedIndex = i;
          hasMatch = true;
          break;
        }
      }
      if (!hasMatch) {
        presetSelect.value = "custom";
      }
    };
  }

  function hideLoading() {
    const overlay = document.getElementById("loading-overlay");
    if (overlay) {
      overlay.style.opacity = "0";
      setTimeout(() => {
        overlay.style.display = "none";
      }, 350);
    }
  }

  function loadModel(vrmUrls) {
    const urls = Array.isArray(vrmUrls) ? vrmUrls : [vrmUrls];
    const loader = new THREE.GLTFLoader();

    function tryLoad(index) {
      if (index >= urls.length) {
        console.error("[VTuber] All candidate model URLs failed to load.");
        const txt = document.getElementById("loading-text");
        if (txt) txt.innerText = "Error loading model. Check assets/Arisa.vrm";
        return;
      }

      const currentUrl = urls[index];
      console.log(`[VTuber] Loading VRM model from: ${currentUrl}`);

      loader.load(
        currentUrl,
        (gltf) => {
          try {
            const vrmCreator =
              (window.THREE_VRM && window.THREE_VRM.VRM && window.THREE_VRM.VRM.from)
                ? window.THREE_VRM.VRM.from.bind(window.THREE_VRM.VRM)
                : (window.THREE && window.THREE.VRM && window.THREE.VRM.fromModel)
                  ? window.THREE.VRM.fromModel.bind(window.THREE.VRM)
                  : null;

            if (vrmCreator) {
              vrmCreator(gltf)
                .then((vrm) => {
                  currentVrm = vrm;
                  window.currentVrm = vrm;
                  scene.add(vrm.scene);
                  setupModelReferences(vrm.scene);
                  hideLoading();
                  console.log(`[VTuber] VRM model loaded successfully from ${currentUrl} via THREE_VRM.VRM.from!`);
                })
                .catch((err) => {
                  console.warn(`[VTuber] VRM.from parser error on ${currentUrl}, fallback to gltf.scene:`, err);
                  scene.add(gltf.scene);
                  setupModelReferences(gltf.scene);
                  hideLoading();
                });
            } else {
              console.warn(`[VTuber] VRM parser not found, using gltf.scene directly for ${currentUrl}`);
              scene.add(gltf.scene);
              setupModelReferences(gltf.scene);
              hideLoading();
            }
          } catch (e) {
            console.error(`[VTuber] Error parsing model from ${currentUrl}, using gltf.scene fallback:`, e);
            scene.add(gltf.scene);
            setupModelReferences(gltf.scene);
            hideLoading();
          }
        },
        (progress) => {
          if (progress.total > 0) {
            const pct = Math.round((progress.loaded / progress.total) * 100);
            const txt = document.getElementById("loading-text");
            if (txt) txt.innerText = `Loading 3D Model (${pct}%)...`;
          }
        },
        (err) => {
          console.warn(`[VTuber] Failed loading ${currentUrl}, trying fallback...`, err);
          tryLoad(index + 1);
        }
      );
    }

    tryLoad(0);
  }

  function setupModelReferences(root) {
    // Face the camera directly (rotate 180 degrees around Y axis)
    root.rotation.y = Math.PI;
    // Lower model slightly so head and forehead/hair have comfortable headroom
    root.position.y = -0.10;

    // Resolve humanoid bones via THREE_VRM humanoid if available
    if (currentVrm && currentVrm.humanoid) {
      try {
        headBone = currentVrm.humanoid.getBoneNode("head") ||
          (window.THREE_VRM && window.THREE_VRM.VRMSchema &&
            currentVrm.humanoid.getBoneNode(window.THREE_VRM.VRMSchema.HumanoidBoneName.Head));
        neckBone = currentVrm.humanoid.getBoneNode("neck") ||
          (window.THREE_VRM && window.THREE_VRM.VRMSchema &&
            currentVrm.humanoid.getBoneNode(window.THREE_VRM.VRMSchema.HumanoidBoneName.Neck));
        spineBone = currentVrm.humanoid.getBoneNode("spine") ||
          (window.THREE_VRM && window.THREE_VRM.VRMSchema &&
            currentVrm.humanoid.getBoneNode(window.THREE_VRM.VRMSchema.HumanoidBoneName.Spine));
        chestBone = currentVrm.humanoid.getBoneNode("chest") ||
          (window.THREE_VRM && window.THREE_VRM.VRMSchema &&
            currentVrm.humanoid.getBoneNode(window.THREE_VRM.VRMSchema.HumanoidBoneName.Chest));
        upperChestBone = currentVrm.humanoid.getBoneNode("upperChest") ||
          (window.THREE_VRM && window.THREE_VRM.VRMSchema &&
            currentVrm.humanoid.getBoneNode(window.THREE_VRM.VRMSchema.HumanoidBoneName.UpperChest));
        hipsBone = currentVrm.humanoid.getBoneNode("hips") ||
          (window.THREE_VRM && window.THREE_VRM.VRMSchema &&
            currentVrm.humanoid.getBoneNode(window.THREE_VRM.VRMSchema.HumanoidBoneName.Hips));
        eyeLeftBone = currentVrm.humanoid.getBoneNode("leftEye") ||
          (window.THREE_VRM && window.THREE_VRM.VRMSchema &&
            currentVrm.humanoid.getBoneNode(window.THREE_VRM.VRMSchema.HumanoidBoneName.LeftEye));
        eyeRightBone = currentVrm.humanoid.getBoneNode("rightEye") ||
          (window.THREE_VRM && window.THREE_VRM.VRMSchema &&
            currentVrm.humanoid.getBoneNode(window.THREE_VRM.VRMSchema.HumanoidBoneName.RightEye));
        leftUpperArmBone = currentVrm.humanoid.getBoneNode("leftUpperArm") ||
          (window.THREE_VRM && window.THREE_VRM.VRMSchema &&
            currentVrm.humanoid.getBoneNode(window.THREE_VRM.VRMSchema.HumanoidBoneName.LeftUpperArm));
        rightUpperArmBone = currentVrm.humanoid.getBoneNode("rightUpperArm") ||
          (window.THREE_VRM && window.THREE_VRM.VRMSchema &&
            currentVrm.humanoid.getBoneNode(window.THREE_VRM.VRMSchema.HumanoidBoneName.RightUpperArm));
        leftLowerArmBone = currentVrm.humanoid.getBoneNode("leftLowerArm") ||
          (window.THREE_VRM && window.THREE_VRM.VRMSchema &&
            currentVrm.humanoid.getBoneNode(window.THREE_VRM.VRMSchema.HumanoidBoneName.LeftLowerArm));
        rightLowerArmBone = currentVrm.humanoid.getBoneNode("rightLowerArm") ||
          (window.THREE_VRM && window.THREE_VRM.VRMSchema &&
            currentVrm.humanoid.getBoneNode(window.THREE_VRM.VRMSchema.HumanoidBoneName.RightLowerArm));
      } catch (e) {
        console.warn("[VTuber] Humanoid bone lookup notice:", e);
      }
    }

    // Traverse root to find mesh morph targets and fallback bones
    root.traverse((obj) => {
      if (obj.isMesh && obj.morphTargetDictionary) {
        mouthMesh = obj;
        morphMeshes.push(obj);
      }

      if (!headBone && (obj.name === "J_Bip_C_Head" || obj.name === "mixamorig:Head" || obj.name.endsWith("Head"))) {
        headBone = obj;
      }
      if (!neckBone && (obj.name === "J_Bip_C_Neck" || obj.name === "mixamorig:Neck" || obj.name.endsWith("Neck"))) {
        neckBone = obj;
      }
      if (!spineBone && (obj.name === "J_Bip_C_Spine" || obj.name === "mixamorig:Spine" || obj.name.endsWith("Spine"))) {
        spineBone = obj;
      }
      if (!chestBone && (obj.name === "J_Bip_C_Chest" || obj.name === "mixamorig:Spine1" || obj.name.endsWith("Chest"))) {
        chestBone = obj;
      }
      if (!upperChestBone && (obj.name === "J_Bip_C_UpperChest" || obj.name === "mixamorig:Spine2" || obj.name.endsWith("UpperChest"))) {
        upperChestBone = obj;
      }
      if (!hipsBone && (obj.name === "J_Bip_C_Hips" || obj.name === "mixamorig:Hips" || obj.name.endsWith("Hips"))) {
        hipsBone = obj;
      }
      if (!eyeLeftBone && (obj.name.includes("Eye_L") || obj.name.includes("leftEye") || obj.name.includes("Eye.L"))) {
        eyeLeftBone = obj;
      }
      if (!eyeRightBone && (obj.name.includes("Eye_R") || obj.name.includes("rightEye") || obj.name.includes("Eye.R"))) {
        eyeRightBone = obj;
      }
      if (!leftUpperArmBone && (obj.name === "J_Bip_L_UpperArm" || obj.name.includes("LeftUpperArm") || obj.name.includes("leftUpperArm"))) {
        leftUpperArmBone = obj;
      }
      if (!rightUpperArmBone && (obj.name === "J_Bip_R_UpperArm" || obj.name.includes("RightUpperArm") || obj.name.includes("rightUpperArm"))) {
        rightUpperArmBone = obj;
      }
      if (!leftLowerArmBone && (obj.name === "J_Bip_L_LowerArm" || obj.name.includes("LeftLowerArm") || obj.name.includes("leftLowerArm"))) {
        leftLowerArmBone = obj;
      }
      if (!rightLowerArmBone && (obj.name === "J_Bip_R_LowerArm" || obj.name.includes("RightLowerArm") || obj.name.includes("rightLowerArm"))) {
        rightLowerArmBone = obj;
      }
    });

    // Disable VRM 1.0 override behaviors so Joy doesn't block A, E, I, O, U
    if (currentVrm && currentVrm.expressionManager && currentVrm.expressionManager.expressions) {
      for (const [groupName, group] of Object.entries(currentVrm.expressionManager.expressions)) {
        group.overrideMouth = 'none';
        group.overrideBlink = 'none';
        group.overrideLookAt = 'none';
      }
    }

    if (currentVrm && currentVrm.lookAt) {
      currentVrm.lookAt.autoUpdate = false;
    }

    applyHairPhysics(
      targetMotion.body.hair_gravity,
      targetMotion.body.hair_stiffness,
      targetMotion.body.hair_drag
    );

    console.log("[VTuber] Model setup complete. Head:", headBone ? headBone.name : "None", "Neck:", neckBone ? neckBone.name : "None", "Eyes:", eyeLeftBone ? eyeLeftBone.name : "None", eyeRightBone ? eyeRightBone.name : "None");
  }

  let lastAppliedHairParams = { gravity: null, stiffness: null, drag: null };

  function isHairBone(bone) {
    if (!bone) return false;
    const name = bone.name || "";
    if (/hair|bang|ahoge|tail|pigtail|braid/i.test(name)) {
      return true;
    }
    let curr = bone.parent;
    while (curr) {
      if (curr === headBone) return true;
      if (curr.name === "J_Bip_C_Head" || /head/i.test(curr.name)) return true;
      curr = curr.parent;
    }
    return false;
  }

  function applyHairPhysics(gravityPower, stiffnessForce, dragForce) {
    if (!currentVrm || !currentVrm.springBoneManager) return;
    const mgr = currentVrm.springBoneManager;
    if (!mgr.springBoneGroupList) return;

    const grav = typeof gravityPower === "number" ? gravityPower : 0.15;
    const stiff = typeof stiffnessForce === "number" ? stiffnessForce : 0.95;
    const drag = typeof dragForce === "number" ? dragForce : 0.50;

    if (
      lastAppliedHairParams.gravity === grav &&
      lastAppliedHairParams.stiffness === stiff &&
      lastAppliedHairParams.drag === drag
    ) {
      return;
    }

    lastAppliedHairParams.gravity = grav;
    lastAppliedHairParams.stiffness = stiff;
    lastAppliedHairParams.drag = drag;

    let hairBonesCount = 0;
    mgr.springBoneGroupList.forEach((group) => {
      group.forEach((spring) => {
        if (isHairBone(spring.bone)) {
          spring.gravityPower = grav;
          spring.gravityDir.set(0, -1, 0); // Always pull downwards along Y axis
          spring.stiffnessForce = stiff;
          spring.dragForce = drag;
          hairBonesCount++;
        }
      });
    });
    console.log(`[VTuber] Applied hair physics: ${hairBonesCount} bones updated (gravity: ${grav}, stiffness: ${stiff}, drag: ${drag})`);
  }

  function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  }

  // Global API called directly from Python via page.runJavaScript
  window.updateMotion = function (data) {
    if (!data) return;

    const now = performance.now();

    if (data.vrm) {
      for (const [k, v] of Object.entries(data.vrm)) {
        if (targetMotion.vrm.hasOwnProperty(k)) {
          targetMotion.vrm[k] = v;
        }
      }
    }

    if (data.vrc) {
      for (const [k, v] of Object.entries(data.vrc)) {
        if (targetMotion.vrc.hasOwnProperty(k)) {
          targetMotion.vrc[k] = v;
        }
      }
    }

    if (data.rotation) {
      targetMotion.rotation.pitch = data.rotation.pitch || 0.0;
      targetMotion.rotation.yaw = data.rotation.yaw || 0.0;
      targetMotion.rotation.roll = data.rotation.roll || 0.0;
    }

    if (data.gaze && blinkState !== "HOLD") {
      targetMotion.gaze.x = data.gaze.x || 0.0;
      targetMotion.gaze.y = data.gaze.y || 0.0;
    }

    if (data.body) {
      if (typeof data.body.sway_gain === "number") {
        targetMotion.body.sway_gain = data.body.sway_gain;
      }
      if (typeof data.body.breath_gain === "number") {
        targetMotion.body.breath_gain = data.body.breath_gain;
      }
      if (typeof data.body.hair_gravity === "number") {
        targetMotion.body.hair_gravity = data.body.hair_gravity;
      }
      if (typeof data.body.hair_stiffness === "number") {
        targetMotion.body.hair_stiffness = data.body.hair_stiffness;
      }
      if (typeof data.body.hair_drag === "number") {
        targetMotion.body.hair_drag = data.body.hair_drag;
      }
      if (typeof data.body.backend === "string") {
        const hudBackend = document.getElementById("hud-backend");
        if (hudBackend && hudBackend.innerText !== data.body.backend) {
          hudBackend.innerText = data.body.backend;
        }
      }
      applyHairPhysics(
        targetMotion.body.hair_gravity,
        targetMotion.body.hair_stiffness,
        targetMotion.body.hair_drag
      );
    }

    lastMotionTimestamp = performance.now();
  };

  window.currentMotion = currentMotion;
  window.targetMotion = targetMotion;
  window.threeRenderer = () => renderer;
  window.threeScene = () => scene;
  window.threeCamera = () => camera;

  function animate(now) {
    requestAnimationFrame(animate);

    // Calculate FPS
    frameCount++;
    if (now - lastFpsTime >= 1000) {
      const fpsEl = document.getElementById("hud-fps");
      if (fpsEl) fpsEl.innerText = `${frameCount} FPS`;
      frameCount = 0;
      lastFpsTime = now;
    }

    // Auto-decay to rest pose if no face detected for > 1.0s
    const isTrackingActive = now - lastMotionTimestamp < 1000;
    const hudStatus = document.getElementById("hud-status");
    if (hudStatus) {
      hudStatus.innerText = isTrackingActive ? "Tracking Active" : "No Face Detected";
    }
    const hudStatusDot = document.getElementById("hud-status-dot");
    if (hudStatusDot) {
      if (isTrackingActive) {
        hudStatusDot.classList.remove("inactive");
        hudStatusDot.title = "Tracking Active (Đang nhận diện)";
      } else {
        hudStatusDot.classList.add("inactive");
        hudStatusDot.title = "No Face Detected (Không thấy khuôn mặt)";
      }
    }

    if (!isTrackingActive) {
      // Gently return to neutral
      targetMotion.rotation.pitch = 0.0;
      targetMotion.rotation.yaw = 0.0;
      targetMotion.rotation.roll = 0.0;
      targetMotion.gaze.x = 0.0;
      targetMotion.gaze.y = 0.0;
      for (const k of Object.keys(targetMotion.vrm)) {
        targetMotion.vrm[k] = k === "neutral" ? 1.0 : 0.0;
      }
      for (const k of Object.keys(targetMotion.vrc)) {
        targetMotion.vrc[k] = k === "vrc_v_sil" ? 1.0 : 0.0;
      }
    }

    // LERP interpolate head rotation with snappy, responsive factor
    const lerpRot = 0.35;
    currentMotion.rotation.pitch +=
      (targetMotion.rotation.pitch - currentMotion.rotation.pitch) * lerpRot;
    currentMotion.rotation.yaw +=
      (targetMotion.rotation.yaw - currentMotion.rotation.yaw) * lerpRot;
    currentMotion.rotation.roll +=
      (targetMotion.rotation.roll - currentMotion.rotation.roll) * lerpRot;

    // Smooth LERP for body parameters
    currentMotion.body.sway_gain += (targetMotion.body.sway_gain - currentMotion.body.sway_gain) * 0.1;
    currentMotion.body.breath_gain += (targetMotion.body.breath_gain - currentMotion.body.breath_gain) * 0.1;

    const swayGain = currentMotion.body.sway_gain;
    const breathGain = currentMotion.body.breath_gain;

    // Apply head, neck, and torso bone rotations (Euler: X=pitch, Y=yaw, Z=roll)
    // Pitch: inverted so looking up tilts head backward
    // Yaw: follows natural user gaze
    // Roll: inverted to correct left/right tilt inversion
    // Clamp to anatomical cervical limits to prevent any anomalous displacement
    // headLevelOffset (-0.08 rad / ~4.5 deg) lowers the resting chin so face looks horizontally level at user
    const headLevelOffset = -0.08;
    const p = Math.max(-0.60, Math.min(0.60, -currentMotion.rotation.pitch + headLevelOffset));
    const y = Math.max(-0.70, Math.min(0.70, -currentMotion.rotation.yaw));
    const r = Math.max(-0.50, Math.min(0.50, currentMotion.rotation.roll));

    // 1. Procedural breathing and gentle idle sway
    const time = clock.getElapsedTime();
    // Breathing cycle (~3.6s per cycle, 1.75 rad/s)
    const breathCycle = Math.sin(time * 1.75);
    const breathPitch = breathCycle * 0.012 * breathGain;
    const breathY = breathCycle * 0.003 * breathGain;

    // Harmonic idle swaying (two asynchronous sine waves preventing robotic loop)
    const idleRoll = (Math.sin(time * 0.7) * 0.012 + Math.cos(time * 1.1) * 0.006) * breathGain;
    const idleYaw = Math.sin(time * 0.5) * 0.008 * breathGain;
    const idleX = (Math.sin(time * 0.7) * 0.005) * breathGain;

    // 2. Kinematic Torso Sway (distribute pitch, yaw, roll across spine, chest, neck, head)
    const spinePitch = (p * 0.12 * swayGain) + (breathPitch * 0.5);
    const spineYaw = (y * 0.10 * swayGain) + (idleYaw * 0.4);
    const spineRoll = (r * 0.18 * swayGain) + (idleRoll * 0.6);

    const chestPitch = (p * 0.14 * swayGain) + breathPitch;
    const chestYaw = (y * 0.14 * swayGain) + (idleYaw * 0.6);
    const chestRoll = (r * 0.18 * swayGain) + (idleRoll * 0.4);

    // Neck and Head take remaining rotation for accurate gaze
    const neckPitch = p * 0.25;
    const neckYaw = y * 0.25;
    const neckRoll = r * 0.25;

    const headPitch = p * 0.49;
    const headYaw = y * 0.51;
    const headRoll = r * 0.39;

    if (spineBone) {
      spineBone.rotation.set(spinePitch, spineYaw, spineRoll);
    }
    if (chestBone) {
      chestBone.rotation.set(chestPitch, chestYaw, chestRoll);
    } else if (upperChestBone) {
      upperChestBone.rotation.set(chestPitch, chestYaw, chestRoll);
    }
    if (neckBone) {
      neckBone.rotation.set(neckPitch, neckYaw, neckRoll);
    }
    if (headBone) {
      headBone.rotation.set(headPitch, headYaw, headRoll);
    }

    // Lateral translation when leaning + vertical breathing motion on root
    const rootBaseY = -0.10;
    const lateralSway = Math.max(-0.15, Math.min(0.15, (r * 0.04 * swayGain) + idleX));
    if (currentVrm && currentVrm.scene) {
      currentVrm.scene.position.x = lateralSway;
      currentVrm.scene.position.y = rootBaseY + breathY;
    }

    // Apply natural lowered resting pose for arms (hands rested comfortably in front of hips)
    if (leftUpperArmBone) {
      leftUpperArmBone.rotation.set(-0.12, 0.0, 1.25);
    }
    if (rightUpperArmBone) {
      rightUpperArmBone.rotation.set(-0.12, 0.0, -1.25);
    }
    if (leftLowerArmBone) {
      leftLowerArmBone.rotation.set(0.0, 0.20, 0.15);
    }
    if (rightLowerArmBone) {
      rightLowerArmBone.rotation.set(0.0, -0.20, -0.15);
    }

    // Blink State Machine (Runs every frame)
    const isPhysicalBlink = targetMotion.vrm.blink >= 0.15 || targetMotion.vrm.blink_l >= 0.25 || targetMotion.vrm.blink_r >= 0.25;

    if (blinkState === "IDLE") {
      if (isPhysicalBlink) {
        blinkState = "HOLD";
        blinkTimer = now + 140; // Hold for at least 140ms
        if (targetMotion.vrm.blink >= 0.15) activeBlinkType = "blink";
        else if (targetMotion.vrm.blink_l >= 0.25) activeBlinkType = "blink_l";
        else activeBlinkType = "blink_r";
      }
    } else if (blinkState === "HOLD") {
      if (isPhysicalBlink) {
        // If both eyes close while previously in single-eye wink, upgrade immediately to synchronized blink
        if (activeBlinkType !== "blink" && targetMotion.vrm.blink >= 0.15) {
          activeBlinkType = "blink";
        }
        blinkTimer = Math.max(blinkTimer, now + 80);
      }
      if (now >= blinkTimer) {
        blinkState = "COOLDOWN";
        blinkTimer = now + 160; // Cooldown for 160ms
      }
    } else if (blinkState === "COOLDOWN") {
      if (now >= blinkTimer && !isPhysicalBlink) {
        blinkState = "IDLE";
        activeBlinkType = null;
      }
    }

    // Mutual exclusivity enforcement across blink targets:
    // Prevents Blink and Blink_L/R from ever stacking additively (which causes > 100% vertex displacement)
    if (activeBlinkType === "blink" || targetMotion.vrm.blink > 0.10) {
      targetMotion.vrm.blink_l = 0.0;
      targetMotion.vrm.blink_r = 0.0;
      currentMotion.vrm.blink_l = 0.0;
      currentMotion.vrm.blink_r = 0.0;
    } else if (activeBlinkType === "blink_l") {
      targetMotion.vrm.blink = 0.0;
      targetMotion.vrm.blink_r = 0.0;
      currentMotion.vrm.blink = 0.0;
      currentMotion.vrm.blink_r = 0.0;
    } else if (activeBlinkType === "blink_r") {
      targetMotion.vrm.blink = 0.0;
      targetMotion.vrm.blink_l = 0.0;
      currentMotion.vrm.blink = 0.0;
      currentMotion.vrm.blink_l = 0.0;
    }

    // Check if eyes are blinking or closing
    const isBlinkActive = blinkState === "HOLD";
    const isBothBlink = (
      currentMotion.vrm.blink > 0.10 ||
      targetMotion.vrm.blink > 0.15 ||
      (isBlinkActive && activeBlinkType === "blink")
    );
    const isLeftClosing = isBothBlink || currentMotion.vrm.blink_l > 0.15 || (isBlinkActive && activeBlinkType === "blink_l");
    const isRightClosing = isBothBlink || currentMotion.vrm.blink_r > 0.15 || (isBlinkActive && activeBlinkType === "blink_r");

    // LERP interpolate iris gaze smoothly when eyes are open; freeze during blink
    if (!isBothBlink) {
      const lerpGaze = 0.35;
      currentMotion.gaze.x +=
        (targetMotion.gaze.x - currentMotion.gaze.x) * lerpGaze;
      currentMotion.gaze.y +=
        (targetMotion.gaze.y - currentMotion.gaze.y) * lerpGaze;
    }

    // Rotate eyeball bones: max ~0.30 radians (~17 degrees)
    // Eye pitch offset lowers the resting pupil position so it is comfortably centered
    const maxEyeAngle = 0.30;
    const eyePitchOffset = -0.07; // Downward pitch offset (~4 deg) to lower the pupil to natural height

    // When eyes are closed/closing, smoothly return eyeballs to resting center
    // This completely prevents frozen tilted eyeball bones twisting the eyelid mesh!
    const eyeRotX = -currentMotion.gaze.y * maxEyeAngle + eyePitchOffset;
    const eyeRotY = -currentMotion.gaze.x * maxEyeAngle;

    if (eyeLeftBone) {
      const rotXL = isLeftClosing ? eyePitchOffset : eyeRotX;
      const rotYL = isLeftClosing ? 0.0 : eyeRotY;
      eyeLeftBone.rotation.set(rotXL, rotYL, 0.0);
    }
    if (eyeRightBone) {
      const rotXR = isRightClosing ? eyePitchOffset : eyeRotX;
      const rotYR = isRightClosing ? 0.0 : eyeRotY;
      eyeRightBone.rotation.set(rotXR, rotYR, 0.0);
    }

    // LERP interpolate expressions and blendshapes
    let topMouth = "Rest";
    let maxMouthWeight = 0.0;

    let topEye = "Open";
    let maxEyeWeight = 0.0;

    const eyeKeys = ["blink", "blink_l", "blink_r", "joy", "fun", "sorrow", "angry", "surprised"];
    const mouthKeys = ["a", "i", "u", "e", "o", "joy", "angry", "sorrow", "fun", "surprised"];

    // Mutual Exclusivity for Facial Emotions: Only 1 dominant emotion allowed at any time
    const emotionKeys = ["joy", "surprised", "sorrow", "angry", "fun"];
    let dominantEmotion = null;
    let maxEmotionVal = 0.0;
    for (const emo of emotionKeys) {
      const val = targetMotion.vrm[emo] || 0.0;
      if (val > maxEmotionVal) {
        maxEmotionVal = val;
        dominantEmotion = emo;
      }
    }
    const vowelKeys = ["a", "i", "u", "e", "o"];

    // Deliberate emotion detection threshold (>= 0.25)
    const hasActiveEmotion = maxEmotionVal >= 0.25;

    // Zero out all conflicting non-dominant emotions
    for (const emo of emotionKeys) {
      if (emo !== dominantEmotion || !hasActiveEmotion) {
        targetMotion.vrm[emo] = 0.0;
        currentMotion.vrm[emo] = 0.0;
      }
    }

    // Only strong deliberate emotions (maxEmotionVal >= 0.60) override mouth visemes.
    // When user is speaking (any vowel > 0.05), speech vowels drive the mouth in real time!
    const isSpeaking = vowelKeys.some(k => (targetMotion.vrm[k] || 0) > 0.05);
    const shouldOverrideVowels = hasActiveEmotion && (maxEmotionVal >= 0.60 || !isSpeaking);

    if (shouldOverrideVowels) {
      for (const k of vowelKeys) {
        targetMotion.vrm[k] = 0.0;
      }
      targetMotion.vrm["neutral"] = 0.0;
    }

    for (const [key, rawTargetVal] of Object.entries(targetMotion.vrm)) {
      const isBlink = (key === "blink" || key === "blink_l" || key === "blink_r");

      let effectiveTarget = rawTargetVal;
      if (isBlink) {
        if (blinkState === "HOLD" && activeBlinkType === key) {
          const eyeSquint = Math.max(currentMotion.vrm.joy || 0, currentMotion.vrm.fun || 0, currentMotion.vrm.sorrow || 0);
          effectiveTarget = Math.max(0.0, 1.0 - eyeSquint);
        } else if (blinkState === "COOLDOWN") {
          effectiveTarget = 0.0;
        }
      }

      // Responsive LERP interpolation:
      // Blinks: fast close (0.95), smooth open (0.35)
      // Speech vowels (A, E, I, O, U): snappy open (0.75) for real-time lip sync, smooth close (0.55)
      // Other emotions: 0.45
      const isSpeechVowel = vowelKeys.includes(key);
      const isTargetHigher = effectiveTarget > currentMotion.vrm[key];
      let lerpFactor = 0.45;
      if (isBlink) {
        lerpFactor = isTargetHigher ? 0.95 : 0.35;
      } else if (isSpeechVowel) {
        lerpFactor = isTargetHigher ? 0.75 : 0.55;
      }

      currentMotion.vrm[key] +=
        (effectiveTarget - currentMotion.vrm[key]) * lerpFactor;

      // Clean snap to 0.0 when target is 0 and residual value is tiny, preventing eyelid droop or stuck expressions
      if (effectiveTarget === 0.0 && currentMotion.vrm[key] < 0.015) {
        currentMotion.vrm[key] = 0.0;
      }

      if (currentVrm) {
        if (currentVrm.blendShapeProxy) {
          // VRM 0.0 expects specific PascalCase enum strings
          let vrm0Key = key;
          if (key === "a") vrm0Key = "A";
          if (key === "i") vrm0Key = "I";
          if (key === "u") vrm0Key = "U";
          if (key === "e") vrm0Key = "E";
          if (key === "o") vrm0Key = "O";
          if (key === "blink") vrm0Key = "Blink";
          if (key === "blink_l") vrm0Key = "Blink_L";
          if (key === "blink_r") vrm0Key = "Blink_R";
          if (key === "joy") vrm0Key = "Joy";
          if (key === "angry") vrm0Key = "Angry";
          if (key === "sorrow") vrm0Key = "Sorrow";
          if (key === "fun") vrm0Key = "Fun";
          if (key === "surprised") vrm0Key = "Surprised";
          if (key === "lookup") vrm0Key = "LookUp";
          if (key === "lookdown") vrm0Key = "LookDown";
          if (key === "lookleft") vrm0Key = "LookLeft";
          if (key === "lookright") vrm0Key = "LookRight";
          if (key === "neutral") vrm0Key = "Neutral";

          let vrm0Val = currentMotion.vrm[key];
          // VRoid models' MTH_U target pinches mouth vertices across the center line if > 0.60
          if (vrm0Key === "U") {
            vrm0Val = Math.min(0.60, vrm0Val);
          }
          // When mouth is opening for speech (a > 0.05), attenuate Joy and Surprised so their mouth shapes don't lock or fight with A
          if ((vrm0Key === "Joy" || vrm0Key === "Surprised") && currentMotion.vrm.a > 0.05) {
            const speechAtten = Math.max(0.0, 1.0 - currentMotion.vrm.a * 1.8);
            vrm0Val *= speechAtten;
          }

          // Strict vertex displacement safety clamp for eyelids:
          // In VRoid models, EYE_Close (Blink), EYE_Close_L, EYE_Close_R, and emotion presets (Joy, Sorrow, Angry)
          // affect the exact same eyelid vertices additively. Total eyelid closure MUST NEVER exceed 1.0 (100%),
          // otherwise vertices penetrate the cheek and invert normals, causing 90-degree eyelid flip/collapse.
          const emotionEyeClosure = Math.max(
            currentMotion.vrm.joy || 0,
            currentMotion.vrm.sorrow || 0,
            currentMotion.vrm.angry || 0
          );

          if (vrm0Key === "Blink") {
            vrm0Val = Math.min(vrm0Val, Math.max(0.0, 1.0 - emotionEyeClosure));
          } else if (vrm0Key === "Blink_L") {
            const currentBlink = (currentMotion.vrm.blink || 0);
            vrm0Val = Math.min(vrm0Val, Math.max(0.0, 1.0 - currentBlink - emotionEyeClosure));
          } else if (vrm0Key === "Blink_R") {
            const currentBlink = (currentMotion.vrm.blink || 0);
            vrm0Val = Math.min(vrm0Val, Math.max(0.0, 1.0 - currentBlink - emotionEyeClosure));
          }

          try { currentVrm.blendShapeProxy.setValue(vrm0Key, vrm0Val); } catch (_) { }
        } else if (currentVrm.expressionManager) {
          // VRM 1.0 expects specific expression names
          let vrm1Key = key;
          if (key === "a") vrm1Key = "aa";
          if (key === "i") vrm1Key = "ih";
          if (key === "u") vrm1Key = "ou";
          if (key === "e") vrm1Key = "ee";
          if (key === "o") vrm1Key = "oh";
          if (key === "blink_l") vrm1Key = "blinkLeft";
          if (key === "blink_r") vrm1Key = "blinkRight";
          if (key === "joy") vrm1Key = "happy";
          if (key === "angry") vrm1Key = "angry";
          if (key === "sorrow") vrm1Key = "sad";
          if (key === "fun") vrm1Key = "relaxed";
          if (key === "surprised") vrm1Key = "surprised";

          let vrm1Val = currentMotion.vrm[key];
          if (vrm1Key === "ou") {
            vrm1Val = Math.min(0.60, vrm1Val);
          }
          if ((vrm1Key === "happy" || vrm1Key === "surprised") && currentMotion.vrm.a > 0.05) {
            const speechAtten = Math.max(0.0, 1.0 - currentMotion.vrm.a * 1.8);
            vrm1Val *= speechAtten;
          }
          if (vrm1Key === "blink" || vrm1Key === "blinkLeft" || vrm1Key === "blinkRight") {
            const emotionEyeClosure = Math.max(
              currentMotion.vrm.joy || 0,
              currentMotion.vrm.sorrow || 0,
              currentMotion.vrm.angry || 0
            );
            vrm1Val = Math.min(vrm1Val, Math.max(0.0, 1.0 - emotionEyeClosure));
          }
          try { currentVrm.expressionManager.setValue(vrm1Key, vrm1Val); } catch (_) { }
        }
      }

      if (key !== "neutral") {
        if (eyeKeys.includes(key) && currentMotion.vrm[key] > maxEyeWeight) {
          maxEyeWeight = currentMotion.vrm[key];
          topEye = key.toUpperCase();
        }
        if (mouthKeys.includes(key)) {
          // Priority order for mouth viseme: A > E > I > U > O
          const vowelPriorityBonus = { "a": 0.05, "e": 0.04, "i": 0.03, "u": 0.02, "o": 0.01 };
          const bonus = vowelPriorityBonus[key] || 0.0;
          const weightedVal = currentMotion.vrm[key] + bonus;
          if (weightedVal > maxMouthWeight) {
            maxMouthWeight = currentMotion.vrm[key];
            topMouth = key.toUpperCase();
          }
        }
      }
    }

    // Immediately flush blendShapeProxy values to mesh morph targets
    if (currentVrm) {
      if (currentVrm.blendShapeProxy) currentVrm.blendShapeProxy.update();
      if (currentVrm.expressionManager) currentVrm.expressionManager.update();
    }
    // No fallback needed for VRM models, expressionManager handles it natively.
    // Raw GLTF fallback is removed for simplicity, as this project focuses on VRM.

    const hudEye = document.getElementById("hud-eye");
    const hudMouth = document.getElementById("hud-mouth");
    if (hudEye) {
      hudEye.innerText = "Eye: " + (maxEyeWeight > 0.15 ? topEye : "Open");
    }
    if (hudMouth) {
      hudMouth.innerText = "Mouth: " + (maxMouthWeight > 0.15 ? topMouth : "Rest");
    }

    // Update VRM internal components (spring bones, physics, and blendshapes)
    if (currentVrm) {
      const delta = Math.min(clock.getDelta(), 0.04);
      currentVrm.update(delta);
    }

    renderer.render(scene, camera);
  }

  // Initialize once DOM is ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
