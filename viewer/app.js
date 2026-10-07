// VTuber 3D Avatar Viewport & Motion Interpolation Engine

(function () {
  "use strict";

  let scene, camera, renderer;
  let currentVrm = null;
  let mouthMesh = null;
  let headBone = null;
  let neckBone = null;
  const clock = new THREE.Clock();

  // Dictionary mapping target keys to mesh morph target indices
  const morphIndexMap = {};
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
  };

  const currentMotion = {
    vrm: { ...targetMotion.vrm },
    vrc: { ...targetMotion.vrc },
    rotation: { ...targetMotion.rotation },
  };

  let lastMotionTimestamp = 0;
  let frameCount = 0;
  let lastFpsTime = performance.now();

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

    // 5. Load VRM Model (Yong)
    loadModel("../assets/Yong.vrm");

    // 6. Start Render Loop
    requestAnimationFrame(animate);
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
      } catch (e) {
        console.warn("[VTuber] Humanoid bone lookup notice:", e);
      }
    }

    // Traverse root to find mesh morph targets and fallback bones
    root.traverse((obj) => {
      if (obj.isMesh && obj.morphTargetDictionary) {
        mouthMesh = obj;
        morphMeshes.push(obj);
        // Build mapping from target keys to morph target indices
        for (const [targetName, idx] of Object.entries(obj.morphTargetDictionary)) {
          for (const key of Object.keys(targetMotion.vrc)) {
            if (targetName.includes(key)) {
              morphIndexMap[key] = idx;
            }
          }
          for (const key of Object.keys(targetMotion.vrm)) {
            if (targetName.toLowerCase() === key.toLowerCase() || targetName.endsWith(key)) {
              morphIndexMap[key] = idx;
            }
          }
        }
      }

      if (!headBone && (obj.name === "J_Bip_C_Head" || obj.name === "mixamorig:Head" || obj.name.endsWith("Head"))) {
        headBone = obj;
      }
      if (!neckBone && (obj.name === "J_Bip_C_Neck" || obj.name === "mixamorig:Neck" || obj.name.endsWith("Neck"))) {
        neckBone = obj;
      }
    });

    console.log("[VTuber] Model setup complete. Head:", headBone ? headBone.name : "None", "Neck:", neckBone ? neckBone.name : "None");
  }

  function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  }

  // Global API called directly from Python via page.runJavaScript
  window.updateMotion = function (data) {
    if (!data) return;

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

    lastMotionTimestamp = performance.now();
  };

  window.currentMotion = currentMotion;
  window.targetMotion = targetMotion;

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
    const y = currentMotion.rotation.yaw;
    const r = -currentMotion.rotation.roll;

    if (neckBone) {
      neckBone.rotation.set(p * 0.3, y * 0.3, r * 0.3);
    }
    if (headBone) {
      headBone.rotation.set(p * 0.7, y * 0.7, r * 0.7);
    }

    // LERP interpolate expressions and blendshapes
    let topExpression = "Neutral";
    let maxWeight = 0.0;

    for (const [key, targetVal] of Object.entries(targetMotion.vrm)) {
      const isBlink = (key === "blink" || key === "blink_l" || key === "blink_r");
      const lerpFactor = isBlink ? 0.85 : 0.45;
      currentMotion.vrm[key] +=
        (targetVal - currentMotion.vrm[key]) * lerpFactor;

      // Clean snap to 0.0 when target is 0 and residual value is tiny, preventing eyelid droop
      if (isBlink && targetVal === 0.0 && currentMotion.vrm[key] < 0.01) {
        currentMotion.vrm[key] = 0.0;
      }

      if (currentVrm && currentVrm.blendShapeProxy) {
        currentVrm.blendShapeProxy.setValue(key, currentMotion.vrm[key]);
      }

      if (key !== "neutral" && currentMotion.vrm[key] > maxWeight) {
        maxWeight = currentMotion.vrm[key];
        topExpression = key.toUpperCase();
      }
    }

    // Fallback direct morph target update only when VRM blendShapeProxy is not available
    if ((!currentVrm || !currentVrm.blendShapeProxy) && morphMeshes.length > 0) {
      for (const mesh of morphMeshes) {
        if (mesh.morphTargetInfluences) {
          for (const [k, v] of Object.entries(currentMotion.vrm)) {
            if (morphIndexMap[k] !== undefined) {
              mesh.morphTargetInfluences[morphIndexMap[k]] = v;
            }
          }
        }
      }
    }

    const expEl = document.getElementById("hud-expression");
    if (expEl) {
      expEl.innerText = maxWeight > 0.2 ? topExpression : "Neutral";
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
