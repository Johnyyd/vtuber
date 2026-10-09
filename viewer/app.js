// VTuber 3D Avatar Viewport & Motion Interpolation Engine

(function () {
  "use strict";

  let scene, camera, renderer;
  let currentVrm = null;
  let mouthMesh = null;
  let headBone = null;
  let neckBone = null;
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
  };

  const currentMotion = {
    vrm: { ...targetMotion.vrm },
    vrc: { ...targetMotion.vrc },
    rotation: { ...targetMotion.rotation },
    gaze: { ...targetMotion.gaze },
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

    // 6. Load VRM Model (Yong)
    loadModel("../assets/Arisa.vrm");

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

  function loadModel(vrmUrl) {
    const loader = new THREE.GLTFLoader();

    loader.load(
      vrmUrl,
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
                console.log("[VTuber] VRM model loaded successfully via THREE_VRM.VRM.from!");
              })
              .catch((err) => {
                console.warn("[VTuber] VRM.from parser error, fallback to gltf.scene:", err);
                scene.add(gltf.scene);
                setupModelReferences(gltf.scene);
                hideLoading();
              });
          } else {
            console.warn("[VTuber] VRM parser not found, using gltf.scene directly");
            scene.add(gltf.scene);
            setupModelReferences(gltf.scene);
            hideLoading();
          }
        } catch (e) {
          console.error("[VTuber] Error parsing model, using gltf.scene fallback:", e);
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
        console.error("[VTuber] GLTF loader network error:", err);
        const txt = document.getElementById("loading-text");
        if (txt) txt.innerText = "Error loading model. Check assets/Yong.vrm";
      }
    );
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

    console.log("[VTuber] Model setup complete. Head:", headBone ? headBone.name : "None", "Neck:", neckBone ? neckBone.name : "None", "Eyes:", eyeLeftBone ? eyeLeftBone.name : "None", eyeRightBone ? eyeRightBone.name : "None");
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

    // Apply head and neck bone rotations (Euler: X=pitch, Y=yaw, Z=roll)
    // Pitch: inverted so looking up tilts head backward
    // Yaw: follows natural user gaze
    // Roll: inverted to correct left/right tilt inversion
    const p = -currentMotion.rotation.pitch;
    const y = -currentMotion.rotation.yaw;
    const r = currentMotion.rotation.roll;

    if (neckBone) {
      neckBone.rotation.set(p * 0.3, y * 0.3, r * 0.3);
    }
    if (headBone) {
      headBone.rotation.set(p * 0.7, y * 0.7, r * 0.7);
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
    const physicalBlink = targetMotion.vrm.blink >= 0.15 || targetMotion.vrm.blink_l >= 0.25 || targetMotion.vrm.blink_r >= 0.25;

    if (blinkState === "IDLE") {
      if (physicalBlink) {
        blinkState = "HOLD";
        blinkTimer = now + 150; // Hold for at least 150ms
        if (targetMotion.vrm.blink >= 0.15) activeBlinkType = "blink";
        else if (targetMotion.vrm.blink_l >= 0.25) activeBlinkType = "blink_l";
        else activeBlinkType = "blink_r";
      }
    } else if (blinkState === "HOLD") {
      if (physicalBlink) {
        // Extend the hold if they are still physically blinking
        blinkTimer = Math.max(blinkTimer, now + 100);
      }
      if (now >= blinkTimer) {
        blinkState = "COOLDOWN";
        blinkTimer = now + 200; // Cooldown for 200ms
      }
    } else if (blinkState === "COOLDOWN") {
      if (now >= blinkTimer && !physicalBlink) {
        // Must wait for cooldown to expire AND physically stop blinking to release
        blinkState = "IDLE";
        activeBlinkType = null;
      }
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

    const eyeRotX = -currentMotion.gaze.y * maxEyeAngle + eyePitchOffset;
    const eyeRotY = -currentMotion.gaze.x * maxEyeAngle;

    // Freeze eyeball bone rotation while eyelids are closing, closed, or opening!
    // This completely eliminates any jumping, snapping, or twitching of pupils during blinks.
    if (eyeLeftBone && !isLeftClosing) {
      eyeLeftBone.rotation.set(eyeRotX, eyeRotY, 0.0);
    }
    if (eyeRightBone && !isRightClosing) {
      eyeRightBone.rotation.set(eyeRotX, eyeRotY, 0.0);
    }

    // LERP interpolate expressions and blendshapes
    let topMouth = "Rest";
    let maxMouthWeight = 0.0;

    let topEye = "Open";
    let maxEyeWeight = 0.0;

    const eyeKeys = ["blink", "blink_l", "blink_r", "joy", "fun", "sorrow", "angry", "surprised"];
    const mouthKeys = ["a", "i", "u", "e", "o", "joy", "angry", "sorrow", "fun", "surprised"];

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

      // Asymmetric LERP: snap close instantly (0.95), open smoothly (0.35)
      const isClosing = effectiveTarget > currentMotion.vrm[key];
      const lerpFactor = isBlink ? (isClosing ? 0.95 : 0.35) : 0.45;

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
          try { currentVrm.expressionManager.setValue(vrm1Key, vrm1Val); } catch (_) { }
        }
      }

      if (key !== "neutral") {
        if (eyeKeys.includes(key) && currentMotion.vrm[key] > maxEyeWeight) {
          maxEyeWeight = currentMotion.vrm[key];
          topEye = key.toUpperCase();
        }
        if (mouthKeys.includes(key) && currentMotion.vrm[key] > maxMouthWeight) {
          maxMouthWeight = currentMotion.vrm[key];
          topMouth = key.toUpperCase();
        }
      }
    }

    // Immediately flush blendShapeProxy values to mesh morph targets
    if (currentVrm) {
      if (currentVrm.blendShapeProxy) currentVrm.blendShapeProxy.update();
      if (currentVrm.expressionManager) currentVrm.expressionManager.update();

      // If user is speaking while smiling, preserve happy eye squint via EYE_Joy
      if (mouthMesh && mouthMesh.morphTargetDictionary && mouthMesh.morphTargetInfluences) {
        const eyeJoyIdx = mouthMesh.morphTargetDictionary["Face.M_F00_000_00_Fcl_EYE_Joy"];
        if (eyeJoyIdx !== undefined) {
          if (currentMotion.vrm.joy > 0.05 && currentMotion.vrm.a > 0.05) {
            const eyeSmileWeight = currentMotion.vrm.joy * Math.min(1.0, currentMotion.vrm.a * 2.0);
            mouthMesh.morphTargetInfluences[eyeJoyIdx] = eyeSmileWeight;
          } else {
            mouthMesh.morphTargetInfluences[eyeJoyIdx] = 0.0;
          }
        }

        // If user is speaking while surprised, preserve surprised brow & eye widening
        const brwSurpIdx = mouthMesh.morphTargetDictionary["Face.M_F00_000_00_Fcl_BRW_Surprised"];
        const eyeSurpIdx = mouthMesh.morphTargetDictionary["Face.M_F00_000_00_Fcl_EYE_Surprised"];
        if (brwSurpIdx !== undefined) {
          if (currentMotion.vrm.surprised > 0.05 && currentMotion.vrm.a > 0.05) {
            mouthMesh.morphTargetInfluences[brwSurpIdx] = currentMotion.vrm.surprised;
          } else {
            mouthMesh.morphTargetInfluences[brwSurpIdx] = 0.0;
          }
        }
        if (eyeSurpIdx !== undefined) {
          if (currentMotion.vrm.surprised > 0.05 && currentMotion.vrm.a > 0.05) {
            mouthMesh.morphTargetInfluences[eyeSurpIdx] = currentMotion.vrm.surprised;
          } else {
            mouthMesh.morphTargetInfluences[eyeSurpIdx] = 0.0;
          }
        }
      }
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
      const delta = clock.getDelta();
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
