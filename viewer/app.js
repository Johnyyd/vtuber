// VTuber 3D Avatar Viewport & Motion Interpolation Engine

(function () {
  "use strict";

  let scene, camera, renderer;
  let currentVrm = null;
  let mouthMesh = null;
  let headBone = null;
  let neckBone = null;

  // Dictionary mapping our VRC target keys to mesh morph target indices
  const morphIndexMap = {};

  // Motion target and current interpolated values
  const targetMotion = {
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

    // 2. Camera setup - focused on avatar upper body / face
    const aspect = window.innerWidth / window.innerHeight;
    camera = new THREE.PerspectiveCamera(28.0, aspect, 0.1, 20.0);
    camera.position.set(0.0, 1.48, 0.85);
    camera.lookAt(0.0, 1.44, 0.0);

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

    // 5. Load VRM Model
    loadModel("../assets/character.vrm");

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
                console.log("[VTuber] VRM model loaded successfully!");
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
        if (txt) txt.innerText = "Error loading model. Check assets/character.vrm";
      }
    );
  }

  function setupModelReferences(root) {
    // Face the camera directly (rotate 180 degrees around Y axis)
    root.rotation.y = Math.PI;

    // Traverse to find Mesh_InteriorMouth2 and bone nodes
    root.traverse((obj) => {
      if (obj.isMesh && obj.morphTargetDictionary) {
        mouthMesh = obj;
        // Build mapping from VRC morph names to morph target indices
        for (const [targetName, idx] of Object.entries(obj.morphTargetDictionary)) {
          // Check if target name ends with vrc_* (e.g. blendShape3.vrc_blink)
          for (const key of Object.keys(targetMotion.vrc)) {
            if (targetName.includes(key)) {
              morphIndexMap[key] = idx;
            }
          }
        }
        console.log("[VTuber] Mapped morph targets:", morphIndexMap);
      }

      if (obj.name === "mixamorig:Head" || obj.name.endsWith("Head")) {
        headBone = obj;
      }
      if (obj.name === "mixamorig:Neck" || obj.name.endsWith("Neck")) {
        neckBone = obj;
      }
    });
  }

  function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  }

  // Global API called directly from Python via page.runJavaScript
  window.updateMotion = function (data) {
    if (!data) return;

    if (data.vrc) {
      for (const [k, v] of Object.entries(data.vrc)) {
        targetMotion.vrc[k] = v;
      }
    }

    if (data.rotation) {
      targetMotion.rotation.pitch = data.rotation.pitch || 0.0;
      targetMotion.rotation.yaw = data.rotation.yaw || 0.0;
      targetMotion.rotation.roll = data.rotation.roll || 0.0;
    }

    lastMotionTimestamp = performance.now();
  };

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
      for (const k of Object.keys(targetMotion.vrc)) {
        targetMotion.vrc[k] = k === "vrc_v_sil" ? 1.0 : 0.0;
      }
    }

    // LERP interpolate head rotation
    const lerpRot = 0.22;
    currentMotion.rotation.pitch +=
      (targetMotion.rotation.pitch - currentMotion.rotation.pitch) * lerpRot;
    currentMotion.rotation.yaw +=
      (targetMotion.rotation.yaw - currentMotion.rotation.yaw) * lerpRot;
    currentMotion.rotation.roll +=
      (targetMotion.rotation.roll - currentMotion.rotation.roll) * lerpRot;

    // Apply head and neck bone rotations (Euler: X=pitch, Y=yaw, Z=roll)
    // Pitch is inverted so looking up tilts head backward, bowing down tilts head forward
    const p = -currentMotion.rotation.pitch;
    const y = currentMotion.rotation.yaw;
    const r = -currentMotion.rotation.roll;

    if (neckBone) {
      neckBone.rotation.set(p * 0.3, y * 0.3, r * 0.3);
    }
    if (headBone) {
      headBone.rotation.set(p * 0.7, y * 0.7, r * 0.7);
    }

    // LERP interpolate morph targets
    let topExpression = "Rest";
    let maxWeight = 0.0;

    if (mouthMesh && mouthMesh.morphTargetInfluences) {
      for (const [key, targetVal] of Object.entries(targetMotion.vrc)) {
        const lerpFactor = key === "vrc_blink" ? 0.55 : 0.35;
        currentMotion.vrc[key] +=
          (targetVal - currentMotion.vrc[key]) * lerpFactor;

        const morphIdx = morphIndexMap[key];
        if (morphIdx !== undefined) {
          mouthMesh.morphTargetInfluences[morphIdx] = currentMotion.vrc[key];
        }

        if (key !== "vrc_v_sil" && currentMotion.vrc[key] > maxWeight) {
          maxWeight = currentMotion.vrc[key];
          topExpression = key.replace("vrc_", "");
        }
      }
    }

    const expEl = document.getElementById("hud-expression");
    if (expEl) {
      expEl.innerText = maxWeight > 0.2 ? topExpression : "Neutral";
    }

    // Update VRM internal components if present
    if (currentVrm) {
      const delta = 1.0 / 60.0;
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
