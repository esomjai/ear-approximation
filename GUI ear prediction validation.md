```python
import os
import re
import urllib.request
import tempfile
import numpy as np
import vtk
import qt
import slicer


# ========================================================================
# Repository raw URLs
# ========================================================================
REPO_RAW = "https://raw.githubusercontent.com/esomjai/ear-approximation/main"
REPO_FILES = {
    "fh4":               REPO_RAW + "/FH4_landmarks.json",
    "hard_tissue":       REPO_RAW + "/hard%20tissue.json",
    "soft_tissue":       REPO_RAW + "/soft%20tissue.mrk.json",
    "lateral_template":  REPO_RAW + "/for%20L%20mastoid%20lateral%20angle.mrk.json",
}


class EarApproximationGUI(qt.QWidget):

    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.setWindowTitle("Ear Approximation — Step-by-step protocol assistant")
        self.setMinimumSize(600, 520)

        self.loadedLandmarks = {}
        self.planeNodes = {}
        self.realignment_already_done = False

        outer = qt.QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        header = qt.QHBoxLayout()
        title = qt.QLabel("🧠 Ear Approximation — protocol assistant")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        header.addWidget(title)
        header.addStretch(1)
        outer.addLayout(header)

        self.scrollArea = qt.QScrollArea()
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setFrameShape(qt.QFrame.NoFrame)
        self.scrollArea.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
        self.scrollArea.setVerticalScrollBarPolicy(qt.Qt.ScrollBarAsNeeded)
        self.scrollArea.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding)
        self.scrollArea.setMinimumHeight(160)

        scrollContent = qt.QWidget()
        scrollLayout = qt.QVBoxLayout(scrollContent)
        scrollLayout.setContentsMargins(0, 0, 0, 0)
        scrollLayout.setSpacing(0)

        self.stepStack = qt.QStackedWidget()
        self.stepStack.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding)
        scrollLayout.addWidget(self.stepStack)

        self.scrollArea.setWidget(scrollContent)
        outer.addWidget(self.scrollArea, 1)

        self.statusLabel = qt.QLabel("Ready")
        self.statusLabel.setWordWrap(True)
        self.statusLabel.setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Maximum)
        self.statusLabel.setStyleSheet(
            "padding: 8px; background-color: #f0f0f0; border-radius: 5px; font-weight: bold;"
        )
        outer.addWidget(self.statusLabel)

        nav = qt.QHBoxLayout()
        self.prevButton = qt.QPushButton("◀ Previous")
        self.prevButton.setMaximumWidth(130)
        self.prevButton.clicked.connect(self.onPrevButtonClicked)
        nav.addWidget(self.prevButton)

        self.stepLabel = qt.QLabel("Step 1 / 7")
        self.stepLabel.setAlignment(qt.Qt.AlignCenter)
        self.stepLabel.setStyleSheet("font-weight: bold; font-size: 14px;")
        nav.addWidget(self.stepLabel, 1)

        self.nextButton = qt.QPushButton("Next ▶")
        self.nextButton.setMaximumWidth(130)
        self.nextButton.clicked.connect(self.onNextButtonClicked)
        nav.addWidget(self.nextButton)

        outer.addLayout(nav)

        try:
            screen = qt.QApplication.desktop().availableGeometry(self)
            max_h = max(500, screen.height() - 80)
            max_w = max(600, screen.width() - 80)
        except Exception:
            max_h, max_w = 1000, 1200
        self.setMaximumSize(max_w, max_h)
        self.resize(min(700, max_w), min(780, max_h))

        self.createStep1_Setup()
        self.createStep2_FHPAlignment()
        self.createStep3_PlaneP()
        self.createStep4_HardTissueAndAxes()
        self.createStep5_SoftTissueAndPlanes()
        self.createStep6_LinearMeasurements()
        self.createStep7_AnglesAndExport()
        self.createStep8_MetadataAndExport()
        # Pre-flight: check for a pre-existing FHP plane
        preflight = self.preflight_check_plane_and_landmarks()
        if preflight is not None:
            if preflight:
                msg = ("Pre-existing FHP plane detected: "
                       + ", ".join(preflight)
                       + ". Realignment will be skipped.")
            else:
                msg = ("Pre-existing FHP plane 'P' detected. "
                       "Realignment will be skipped.")
            self.step1Status.setText(msg)
            self.step2Status.setText(
                msg + " Step 2 will be skipped — the scene is already aligned."
            )
        self.currentStep = 0
        self.updateStepUI()
        self.show()

    # ==================================================================
    # Helpers
    # ==================================================================
    def _add_step(self, widget):
        self.stepStack.addWidget(widget)

    def _find_node(self, name):
        try:
            return slicer.util.getNode(name)
        except slicer.util.MRMLNodeNotFoundException:
            return None

    def _landmark_position(self, node, *label_candidates, world=True):
        if node is None:
            raise ValueError("Node is None")
        n = node.GetNumberOfControlPoints()
        labels_lower = [(node.GetNthControlPointLabel(i) or "").strip().lower()
                        for i in range(n)]
        for cand in label_candidates:
            cand_l = cand.strip().lower()
            for i, lbl in enumerate(labels_lower):
                if lbl == cand_l:
                    p = [0.0, 0.0, 0.0]
                    if world:
                        node.GetNthControlPointPositionWorld(i, p)
                    else:
                        node.GetNthControlPointPosition(i, p)
                    return np.array(p)
        avail = [node.GetNthControlPointLabel(i) for i in range(n)]
        raise ValueError(
            f"None of {label_candidates} found in '{node.GetName()}'. "
            f"Available labels: {avail}"
        )

    def _download_json(self, url, node_name):
        old = self._find_node(node_name)
        if old is not None:
            slicer.mrmlScene.RemoveNode(old)
        req = urllib.request.Request(url, headers={"User-Agent": "Slicer/ear-approx"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mrk.json")
        try:
            tmp.write(data)
            tmp.close()
            node = slicer.util.loadMarkups(tmp.name)
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
        if not node:
            raise RuntimeError(f"loadMarkups returned None for {url}")
        node.SetName(node_name)
        return node
    def preflight_check_plane_and_landmarks(self):
        """Scan the scene for a pre-existing FHP plane. If found:
          - rename it to 'P' if it is called anything else
          - rename any FH4-style landmark labels (por1, orb1, ...) to their
            coded equivalents (poR, poL, zyoL, zyoR) in every fiducial node
          - set self.realignment_already_done = True so Step 2 can be
            skipped

        Returns a list of change strings describing what was modified, or
        None if no FHP plane was found.
        """
        # 1. Look for the plane
        plane_found = None
        for candidate in self.FHP_PLANE_NAMES:
            node = self._find_node(candidate)
            if node is not None and node.GetClassName() == "vtkMRMLMarkupsPlaneNode":
                plane_found = node
                break

        if plane_found is None:
            self.realignment_already_done = False
            return None

        changed = []

        # 2. Rename the plane to 'P'
        if plane_found.GetName() != "P":
            changed.append(f"plane '{plane_found.GetName()}' -> 'P'")
            plane_found.SetName("P")

        # 3. Rename landmarks
        rename_lower = {k.lower(): v for k, v in self.FHP_LANDMARK_RENAMES.items()}
        for fid in slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode"):
            for i in range(fid.GetNumberOfControlPoints()):
                lbl = (fid.GetNthControlPointLabel(i) or "").strip()
                if lbl.lower() in rename_lower:
                    new_lbl = rename_lower[lbl.lower()]
                    if lbl != new_lbl:
                        fid.SetNthControlPointLabel(i, new_lbl)
                        changed.append(
                            f"{lbl} -> {new_lbl} (in '{fid.GetName()}')"
                        )

        self.realignment_already_done = True
        return changed

    # ==================================================================
    # STEP 1
    # ==================================================================
    def createStep1_Setup(self):
        w = qt.QWidget()
        layout = qt.QVBoxLayout(w)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        t = qt.QLabel("Step 1 — Load Frankfort landmarks")
        t.setStyleSheet("font-weight: bold; font-size: 15px;")
        layout.addWidget(t)

        info = qt.QLabel(
            "The FHP alignment script needs <b>three</b> landmarks: "
            "<b>poR</b> (right porion), <b>poL</b> (left porion), "
            "<b>zyoL</b> (left orbitale).<br><br>"
            "This button downloads <code>FH4_landmarks.json</code> from the "
            "repository and renames its four control points to "
            "<code>poR / poL / zyoL / zyoR</code>. The order in the file "
            "does not matter — the script finds them by label."
        )
        info.setTextFormat(qt.Qt.RichText)
        info.setWordWrap(True)
        layout.addWidget(info)

        btn = qt.QPushButton("Download FH4 landmarks and rename labels")
        btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        btn.clicked.connect(self.onDownloadFH4Clicked)
        layout.addWidget(btn)

        btn_check = qt.QPushButton("Show current labels (diagnostic)")
        btn_check.setStyleSheet("background-color: #607D8B; color: white; padding: 6px;")
        btn_check.clicked.connect(self.onShowFH4LabelsClicked)
        layout.addWidget(btn_check)
        
        btn_preflight = qt.QPushButton("Check scene for an existing FHP plane")
        btn_preflight.setStyleSheet(
            "background-color: #FF9800; color: white; padding: 6px;"
        )
        btn_preflight.setToolTip(
            "Scans the scene for a plane named P / FHP / Frankfort. If one "
            "is found, renames it to 'P', renames FH4-style landmarks to "
            "their coded equivalents, and marks Step 2 as skipped."
        )
        btn_preflight.clicked.connect(self.onPreflightClicked)
        layout.addWidget(btn_preflight)
        self.step1Status = qt.QLabel("Status: waiting.")
        self.step1Status.setWordWrap(True)
        self.step1Status.setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Maximum)
        self.step1Status.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        layout.addWidget(self.step1Status)

        layout.addStretch(1)
        self._add_step(w)

    def onPreflightClicked(self):
        changed = self.preflight_check_plane_and_landmarks()
        if changed is None:
            self.step1Status.setText(
                "No pre-existing FHP plane found in the scene. Run the "
                "Frankfort alignment in Step 2 as usual."
            )
        else:
            self.step1Status.setText(
                "FHP plane found and prepared: " + ", ".join(changed)
                + ". Realignment will be skipped."
            )
            self.step2Status.setText(
                "Realignment skipped — a pre-existing plane P is present. "
                "Proceed to Step 3."
            )
        self.statusLabel.setText(self.step1Status.text)

    def onDownloadFH4Clicked(self):
        self.step1Status.setText("Downloading FH4_landmarks.json ...")
        slicer.app.processEvents()
        try:
            node = self._download_json(REPO_FILES["fh4"], "FH4_landmarks")
            rename = {"por1": "poR", "por2": "poL", "orb1": "zyoL", "orb2": "zyoR"}
            renamed = []
            for i in range(node.GetNumberOfControlPoints()):
                lbl = node.GetNthControlPointLabel(i)
                if lbl in rename:
                    node.SetNthControlPointLabel(i, rename[lbl])
                    renamed.append(f"{lbl} -> {rename[lbl]}")
            self.loadedLandmarks["fh4"] = node
            self.step1Status.setText(
                f"Loaded '{node.GetName()}' with {node.GetNumberOfControlPoints()} points. "
                f"Renamed: {', '.join(renamed)}."
            )
        except Exception as e:
            self.step1Status.setText(f"Error: {e}")
        self.statusLabel.setText(self.step1Status.text)

    def onShowFH4LabelsClicked(self):
        node = self._find_node("FH4_landmarks")
        if not node:
            self.step1Status.setText("FH4_landmarks not loaded.")
            return
        labels = [node.GetNthControlPointLabel(i) for i in range(node.GetNumberOfControlPoints())]
        self.step1Status.setText(f"FH4_landmarks labels: {labels}")
        self.statusLabel.setText(self.step1Status.text)

    # ==================================================================
    # STEP 2
    # ==================================================================
    def createStep2_FHPAlignment(self):
        w = qt.QWidget()
        layout = qt.QVBoxLayout(w)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        t = qt.QLabel("Step 2 — Frankfort Horizontal Plane alignment")
        t.setStyleSheet("font-weight: bold; font-size: 15px;")
        layout.addWidget(t)

        info = qt.QLabel(
            "Applies three successive transforms to align the volume and all "
            "fiducial nodes to the Frankfort Horizontal Plane:<br>"
            "<b>1.</b> Yaw about Z (aligns the porion pair with the X axis)<br>"
            "<b>2.</b> Roll about Y<br>"
            "<b>3.</b> Pitch about X (aligns zyoL with the horizontal)<br><br>"
            "All three transforms are <b>hardened</b> onto the volume at the end, "
            "so subsequent measurements are in the aligned space. If you have "
            "more than one volume in the scene, all of them are transformed."
        )
        info.setTextFormat(qt.Qt.RichText)
        info.setWordWrap(True)
        layout.addWidget(info)

        btn = qt.QPushButton("Run Frankfort alignment")
        btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        btn.clicked.connect(self.onRunFHPAlignmentClicked)
        layout.addWidget(btn)

        self.step2Status = qt.QLabel("Status: idle.")
        self.step2Status.setWordWrap(True)
        self.step2Status.setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Maximum)
        self.step2Status.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        layout.addWidget(self.step2Status)

        layout.addStretch(1)
        self._add_step(w)

    def onRunFHPAlignmentClicked(self):
        F = self.loadedLandmarks.get("fh4") or self._find_node("FH4_landmarks")
        
        if not F:
            self.step2Status.setText("Error: FH4_landmarks not loaded. Run Step 1 first.")
            return
        if getattr(self, "realignment_already_done", False):
            self.step2Status.setText(
                "Skipped: a pre-existing plane 'P' was found in the scene, "
                "so realignment is unnecessary. Proceed to Step 3."
            )
            self.statusLabel.setText(self.step2Status.text)
            return
        try:
            poR = self._landmark_position(F, "poR")
            poL = self._landmark_position(F, "poL")
            zyoL = self._landmark_position(F, "zyoL")
        except ValueError as e:
            self.step2Status.setText(f"Error: {e}")
            return

        scene = slicer.mrmlScene

        # Yaw about Z
        v = poR - poL
        t1 = vtk.vtkTransform()
        t1.RotateZ(-np.degrees(np.arctan2(v[1], v[0])))
        tn1 = scene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "FHP_yaw")
        tn1.SetMatrixTransformToParent(t1.GetMatrix())

        volumes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        for vol in volumes:
            vol.SetAndObserveTransformNodeID(tn1.GetID())
        for fid in slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode"):
            fid.SetAndObserveTransformNodeID(tn1.GetID())

        m1 = t1.GetMatrix()
        def apply_m(m, p):
            q = m.MultiplyPoint([p[0], p[1], p[2], 0.0])
            return np.array(q[:3])

        poR_1 = apply_m(m1, poR)
        poL_1 = apply_m(m1, poL)
        v1 = poR_1 - poL_1

        # Roll about Y
        t2 = vtk.vtkTransform()
        t2.RotateY(np.degrees(np.arctan2(v1[2], v1[0])))
        tn2 = scene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "FHP_roll")
        tn2.SetMatrixTransformToParent(t2.GetMatrix())
        tn1.SetAndObserveTransformNodeID(tn2.GetID())

        m2 = t2.GetMatrix()
        zyoL_2 = apply_m(m2, apply_m(m1, zyoL))
        poR_2 = apply_m(m2, poR_1)
        poL_2 = apply_m(m2, poL_1)
        mid_2 = (poR_2 + poL_2) / 2.0
        v2 = zyoL_2 - mid_2

        # Pitch about X
        t3 = vtk.vtkTransform()
        t3.RotateX(-np.degrees(np.arctan2(v2[2], v2[1])))
        tn3 = scene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "FHP_pitch")
        tn3.SetMatrixTransformToParent(t3.GetMatrix())
        tn2.SetAndObserveTransformNodeID(tn3.GetID())

        logic = slicer.vtkSlicerTransformLogic()
        for vol in volumes:
            logic.hardenTransform(vol)

        self.step2Status.setText(
            f"FHP alignment complete. {len(volumes)} volume(s) hardened. "
            "Proceed to Step 3 to review the landmark positions and create plane P."
        )
        self.statusLabel.setText(self.step2Status.text)

    # ==================================================================
    # STEP 3 — Optional landmark review, then create plane P
    # ==================================================================
    def createStep3_PlaneP(self):
        w = qt.QWidget()
        layout = qt.QVBoxLayout(w)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        t = qt.QLabel("Step 3 — Create plane 'P' (the FHP plane)")
        t.setStyleSheet("font-weight: bold; font-size: 15px;")
        layout.addWidget(t)

        info = qt.QLabel(
            "The four Frankfort landmarks (poR, poL, zyoL, zyoR) were placed "
            "<b>before</b> the realignment in Step 2, so their current positions "
            "are in the aligned frame — but the alignment may have introduced "
            "small errors.<br><br>"
            "<b>Do you want to re-check the landmarks?</b><br>"
            "&nbsp;&nbsp;• <b>No</b> — the landmarks look fine. Just click the "
            "button below.<br>"
            "&nbsp;&nbsp;• <b>Yes</b> — adjust the positions of poR / poL / zyoL / zyoR "
            "manually in the 3D view first, then come back and click the button.<br><br>"
            "The plane is fitted through the four <i>current</i> positions, so "
            "whatever state the landmarks are in when you click the button is "
            "what gets used. If the FH4 node is still parented to the FHP "
            "transforms, harden it first (right-click the node in the Data "
            "module, choose <i>Harden transform</i>)."
        )
        info.setTextFormat(qt.Qt.RichText)
        info.setWordWrap(True)
        layout.addWidget(info)

        btn = qt.QPushButton("Create plane 'P' from the 4 Frankfort landmarks")
        btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; padding: 10px;")
        btn.clicked.connect(self.onCreatePlanePClicked)
        layout.addWidget(btn)

        self.step3Status = qt.QLabel("Status: idle.")
        self.step3Status.setWordWrap(True)
        self.step3Status.setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Maximum)
        self.step3Status.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        layout.addWidget(self.step3Status)

        layout.addStretch(1)
        self._add_step(w)

    def onCreatePlanePClicked(self):
        F = self.loadedLandmarks.get("fh4") or self._find_node("FH4_landmarks")
        if not F:
            self.step3Status.setText("Error: FH4_landmarks not loaded. Run Step 1 first.")
            return

        try:
            pts = np.array([
                self._landmark_position(F, "poR"),
                self._landmark_position(F, "poL"),
                self._landmark_position(F, "zyoL"),
                self._landmark_position(F, "zyoR"),
            ])
        except ValueError as e:
            self.step3Status.setText(f"Error: {e}")
            return

        centroid = pts.mean(axis=0)
        centred = pts - centroid
        _, _, Vt = np.linalg.svd(centred, full_matrices=False)
        normal = Vt[-1, :]
        normal /= np.linalg.norm(normal)
        if normal[1] < 0:
            normal = -normal

        old = self._find_node("P")
        if old is not None:
            slicer.mrmlScene.RemoveNode(old)

        plane = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsPlaneNode", "P")
        plane.SetOrigin(centroid)
        plane.SetNormal(normal)
        plane.SetSize(250, 250)
        dn = plane.GetDisplayNode()
        dn.SetColor(0.0, 0.6, 1.0)
        dn.SetSelectedColor(0.0, 0.6, 1.0)
        dn.SetOpacity(0.7)

        self.planeNodes["P"] = plane
        self.step3Status.setText(
            f"Plane 'P' created. Origin: "
            f"[{centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}]  "
            f"Normal: [{normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f}]."
        )
        self.statusLabel.setText(self.step3Status.text)
    # ==================================================================
    # STEP 4 — Hard tissue and mastoid axes
    # ==================================================================
    # Axis definitions from Table 7 of the protocol. Displayed next to each
    # button so the user knows what to draw while adjusting.
    AXIS_DEFINITIONS = {
        "main axis of right mastoid L":
            "General axis passing through the central aspect of the RIGHT "
            "mastoid process in LATERAL view, starting from maiR.",
        "main axis of left mastoid L":
            "General axis passing through the central aspect of the LEFT "
            "mastoid process in LATERAL view, starting from maiL.",
        "main axis of right mastoid I":
            "General axis passing through the central aspect of the RIGHT "
            "mastoid process in INFERIOR view, starting from maiR.",
        "main axis of left mastoid I":
            "General axis passing through the central aspect of the LEFT "
            "mastoid process in INFERIOR view, starting from maiL.",
        "mandibular ramus axis R":
            "Axis connecting the most superior part of the RIGHT condylar "
            "process to the RIGHT gonion of the mandible.",
        "mandibular ramus axis L":
            "Axis connecting the most superior part of the LEFT condylar "
            "process to the LEFT gonion of the mandible.",
    }

    def createStep4_HardTissueAndAxes(self):
        w = qt.QWidget()
        layout = qt.QVBoxLayout(w)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        t = qt.QLabel("Step 4 — Hard tissue landmarks and mastoid axes")
        t.setStyleSheet("font-weight: bold; font-size: 15px;")
        layout.addWidget(t)

        info = qt.QLabel(
            "Loads the hard tissue landmark set and creates six reference "
            "axes as line nodes, each pre-populated with <b>one</b> anchor "
            "control point. You place the second control point manually in "
            "the 3D view.<br><br>"
            "The <b>mastoid heights</b> are computed automatically: they are "
            "the orthogonal projections of <code>msR</code> and "
            "<code>msL</code> onto plane <code>P</code>. No manual point "
            "placement is needed for them."
        )
        info.setTextFormat(qt.Qt.RichText)
        info.setWordWrap(True)
        layout.addWidget(info)

        # --- Load button ---
        btn_load = qt.QPushButton("1. Load hard tissue landmarks")
        btn_load.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;"
        )
        btn_load.clicked.connect(self.onLoadHardTissueClicked)
        layout.addWidget(btn_load)

        # --- Create axes button ---
        btn_axes = qt.QPushButton("2. Create the six axes (one anchor point each)")
        btn_axes.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold; padding: 10px;"
        )
        btn_axes.clicked.connect(self.onCreateAxesClicked)
        layout.addWidget(btn_axes)

        # --- Project mastoid heights button (no manual adjustment needed) ---
        btn_proj = qt.QPushButton("3. Project mastoid heights (msR, msL) onto plane P")
        btn_proj.setStyleSheet(
            "background-color: #9C27B0; color: white; font-weight: bold; padding: 10px;"
        )
        btn_proj.setToolTip(
            "Computes the shortest distance from msR and msL to plane P. "
            "No manual adjustment is required."
        )
        btn_proj.clicked.connect(self.onProjectMastoidOntoPClicked)
        layout.addWidget(btn_proj)

        # --- Axis definitions (visible while the user adjusts) ---
        defs_group = qt.QGroupBox("Axis definitions (Table 7 of the protocol)")
        defs_layout = qt.QVBoxLayout(defs_group)
        for name, definition in self.AXIS_DEFINITIONS.items():
            row = qt.QLabel(
                f"<b>{name}</b><br>"
                f"<span style='color: #555;'>{definition}</span>"
            )
            row.setTextFormat(qt.Qt.RichText)
            row.setWordWrap(True)
            row.setContentsMargins(5, 4, 5, 4)
            defs_layout.addWidget(row)
        layout.addWidget(defs_group)

        self.step4Status = qt.QLabel("Status: idle.")
        self.step4Status.setWordWrap(True)
        self.step4Status.setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Maximum)
        self.step4Status.setStyleSheet(
            "padding: 10px; background-color: #f0f0f0; border-radius: 5px;"
        )
        layout.addWidget(self.step4Status)

        layout.addStretch(1)
        self._add_step(w)

    def onCreateAxesClicked(self):
        """Create the six axes as line nodes, each with a single anchor
        control point. The mastoid heights are NOT created here — they are
        handled entirely by the projection button."""
        hard = self.loadedLandmarks.get("hard_tissue") or self._find_node("hard tissue")
        if not hard:
            self.step4Status.setText(
                "Error: 'hard tissue' not loaded. Click button 1 first."
            )
            return

        specs = [
            ("main axis of right mastoid L", "maiR"),
            ("main axis of left mastoid L",  "maiL"),
            ("main axis of right mastoid I", "maiR"),
            ("main axis of left mastoid I",  "maiL"),
            ("mandibular ramus axis R",      "goR"),
            ("mandibular ramus axis L",      "goL"),
        ]

        created, missing = [], []
        for line_name, anchor_label in specs:
            try:
                self._make_empty_line_from_landmark(hard, line_name, anchor_label)
                created.append(line_name)
            except ValueError as e:
                missing.append(f"{line_name} ({e})")

        if missing:
            self.step4Status.setText(
                f"Created {len(created)} axes, {len(missing)} failed: "
                + " | ".join(missing)
            )
        else:
            self.step4Status.setText(
                f"Created {len(created)} axes, each with one anchor point. "
                "Place the second control point of each line in the 3D view, "
                "using the definitions above as a guide. Then run the "
                "mastoid height projection."
            )
        self.statusLabel.setText(self.step4Status.text)

    def onProjectMastoidOntoPClicked(self):
        """Create the two mastoid height lines by projecting msR and msL
        orthogonally onto plane P. No user interaction required."""
        plane = self._find_node("P")
        if not plane:
            self.step4Status.setText("Error: plane 'P' not found. Finish Step 3 first.")
            return

        hard = self.loadedLandmarks.get("hard_tissue") or self._find_node("hard tissue")
        if not hard:
            self.step4Status.setText("Error: 'hard tissue' not loaded.")
            return

        planeOrigin = np.zeros(3); plane.GetOriginWorld(planeOrigin)
        planeNormal = np.zeros(3); plane.GetNormalWorld(planeNormal)
        planeNormal = planeNormal / np.linalg.norm(planeNormal)

        results = []
        for label, anchor in [("mastoid height R", "msR"),
                              ("mastoid height L", "msL")]:
            try:
                anchor_pt = self._landmark_position(hard, anchor)
            except ValueError as e:
                results.append(f"{label}: landmark not found ({e})")
                continue

            # Orthogonal projection of the anchor onto plane P
            dist = np.dot(anchor_pt - planeOrigin, planeNormal)
            proj_pt = anchor_pt - planeNormal * dist

            # Replace any old node of this name
            old = self._find_node(label)
            if old is not None:
                slicer.mrmlScene.RemoveNode(old)

            # Create the line with two control points: anchor and its
            # projection onto P. The distance between them is the mastoid
            # height.
            line = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", label)
            dn = line.GetDisplayNode()
            if dn is not None:
                dn.SetColor(1.0, 0.8, 0.0)
                dn.SetSelectedColor(1.0, 0.5, 0.0)
                dn.SetPointLabelsVisibility(True)
            line.AddControlPoint(anchor_pt)
            line.AddControlPoint(proj_pt)

            results.append(
                f"{label}: distance = {abs(dist):.2f} mm "
                f"(anchor {anchor} projected onto P)"
            )

        self.step4Status.setText(
            "Mastoid heights computed:\n  " + "\n  ".join(results)
        )
        self.statusLabel.setText(self.step4Status.text)

    # ==================================================================
    # STEP 5 — Soft tissue and reference planes
    # ==================================================================
    def createStep5_SoftTissueAndPlanes(self):
        w = qt.QWidget()
        layout = qt.QVBoxLayout(w)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        t = qt.QLabel("Step 5 — Soft tissue landmarks and reference planes")
        t.setStyleSheet("font-weight: bold; font-size: 15px;")
        layout.addWidget(t)

        info = qt.QLabel(
            "<b>5a. Load soft tissue landmarks.</b> This downloads "
            "<code>soft tissue.mrk.json</code>. After loading, you may want to "
            "toggle the volume rendering on or off to see the skin surface.<br><br>"
            "<b>5b. Create the four reference planes.</b> The button below "
            "clones plane <code>P</code> into <code>SagR</code>, "
            "<code>SagL</code>, <code>CorR</code>, <code>CorL</code> and "
            "applies the two rotations described in the protocol "
            "(90 degrees in LR first, then 90 degrees in IS on the sagittal "
            "pair). The rotations are applied to temporary transform nodes "
            "that are hardened automatically.<br><br>"
            "<b>Manual work after creating the planes:</b><br>"
            "&nbsp;&nbsp;• Enable <i>Interaction handles</i> in the Markups "
            "module: Display -> Interaction -> Visibility and Translate.<br>"
            "&nbsp;&nbsp;• Move <b>SagR</b> until it passes through "
            "<b>obsR</b>, and <b>SagL</b> through <b>obsL</b>.<br>"
            "&nbsp;&nbsp;• Move <b>CorR</b> until it passes through the "
            "manual endpoint of <b>main axis of right mastoid I</b>, and "
            "<b>CorL</b> through the endpoint of <b>main axis of left mastoid I</b>.<br>"
            "&nbsp;&nbsp;• If a plane refuses to move, its control points are "
            "locked — unlock them in the Markups module."
        )
        info.setTextFormat(qt.Qt.RichText)
        info.setWordWrap(True)
        layout.addWidget(info)

        btn_load = qt.QPushButton("5a. Load soft tissue landmarks")
        btn_load.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        btn_load.clicked.connect(self.onLoadSoftTissueClicked)
        layout.addWidget(btn_load)

        btn_planes = qt.QPushButton("5b. Clone P into SagR, SagL, CorR, CorL and rotate")
        btn_planes.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; padding: 10px;")
        btn_planes.clicked.connect(self.onCreateReferencePlanesClicked)
        layout.addWidget(btn_planes)

        self.step5Status = qt.QLabel("Status: idle.")
        self.step5Status.setWordWrap(True)
        self.step5Status.setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Maximum)
        self.step5Status.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        layout.addWidget(self.step5Status)

        layout.addStretch(1)
        self._add_step(w)

    def onLoadSoftTissueClicked(self):
        self.step5Status.setText("Downloading soft tissue landmarks ...")
        slicer.app.processEvents()
        try:
            node = self._download_json(REPO_FILES["soft_tissue"], "soft tissue")
            labels = [node.GetNthControlPointLabel(i)
                      for i in range(node.GetNumberOfControlPoints())]
            self.loadedLandmarks["soft_tissue"] = node
            self.step5Status.setText(
                f"Loaded 'soft tissue' with {len(labels)} points. "
                f"Labels: {labels}. You may now toggle the volume rendering "
                "to see the skin surface."
            )
        except Exception as e:
            self.step5Status.setText(f"Error: {e}")
        self.statusLabel.setText(self.step5Status.text)

    # ----------------------------------------------------------------
    # Missing methods that the paste dropped — needed by Step 4 and Step 5
    # ----------------------------------------------------------------
    def onLoadHardTissueClicked(self):
        self.step4Status.setText("Downloading hard tissue landmarks ...")
        slicer.app.processEvents()
        try:
            node = self._download_json(REPO_FILES["hard_tissue"], "hard tissue")
             # Defensive: unlock any points that came in locked
            for i in range(node.GetNumberOfControlPoints()):
                if node.GetNthControlPointLocked(i):
                    node.SetNthControlPointLocked(i, False)
            labels = [node.GetNthControlPointLabel(i)
                      for i in range(node.GetNumberOfControlPoints())]
            self.loadedLandmarks["hard_tissue"] = node
            self.step4Status.setText(
                f"Loaded 'hard tissue' with {len(labels)} points. "
                f"Labels: {labels}."
            )
        except Exception as e:
            self.step4Status.setText(f"Error: {e}")
        self.statusLabel.setText(self.step4Status.text)

    def _make_empty_line_from_landmark(self, source_node, label, anchor_label):
        """Create an empty line node with one control point taken from
        `source_node` at the landmark called `anchor_label`."""
        old = self._find_node(label)
        if old is not None:
            slicer.mrmlScene.RemoveNode(old)
        line = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", label)
        dn = line.GetDisplayNode()
        if dn is not None:
            dn.SetColor(0.4, 1.0, 1.0)
            dn.SetSelectedColor(0.6, 0.0, 0.5)
            dn.SetPointLabelsVisibility(True)
        anchor = self._landmark_position(source_node, anchor_label)
        line.AddControlPoint(anchor)
        return line

    # ----------------------------------------------------------------
    # Step 5b — reference planes
    # ----------------------------------------------------------------
    def _create_perpendicular_plane(self, name, origin, normal, colour):
        """Create (or replace) a plane node with the given origin and normal."""
        old = self._find_node(name)
        if old is not None:
            slicer.mrmlScene.RemoveNode(old)
        plane = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsPlaneNode", name)
        n = normal / np.linalg.norm(normal)
        plane.SetOrigin(origin)
        plane.SetNormal(n)
        plane.SetSize(250, 250)
        dn = plane.GetDisplayNode()
        if dn is not None:
            dn.SetColor(*colour)
            dn.SetSelectedColor(*colour)
            dn.SetOpacity(0.7)
        return plane

    def onCreateReferencePlanesClicked(self):
        """Create SagR, SagL, CorR, CorL as planes perpendicular to P and
        passing through the specified landmarks, per the protocol:

          - SagR / SagL: PARASAGITTAL planes (thin left-right), with normal
            along the LR axis, passing through obsR / obsL.
          - CorR / CorL: CORONAL planes (thin front-back), with normal
            along the AP axis, passing through the manual endpoint of the
            right / left mastoid I line respectively.
        """
        p = self._find_node("P")
        if not p:
            self.step5Status.setText("Error: plane 'P' not found. Finish Step 3 first.")
            return

        hard = self.loadedLandmarks.get("hard_tissue") or self._find_node("hard tissue")
        soft = self.loadedLandmarks.get("soft_tissue") or self._find_node("soft tissue")
        if not hard or not soft:
            self.step5Status.setText(
                "Error: 'hard tissue' or 'soft tissue' not loaded. "
                "Run Steps 4 and 5a first."
            )
            return

        nP = np.zeros(3); p.GetNormalWorld(nP)
        nP = nP / np.linalg.norm(nP)

        # Left-right axis (RAS +X)
        LR = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(LR, nP)) > 0.95:
            LR = np.array([0.0, 1.0, 0.0])

        # Anterior-posterior axis: perpendicular to both nP and LR.
        # (This is the vector that points front-to-back in the FHP frame.)
        AP = np.cross(nP, LR)
        AP = AP / np.linalg.norm(AP)
        # Make AP actually point anterior (+Y) for consistency
        if AP[1] < 0:
            AP = -AP

        try:
            # ---- Origins ----
            obsR_origin = self._landmark_position(soft, "obsR")
            obsL_origin = self._landmark_position(soft, "obsL")

            def _manual_endpoint(line_name):
                node = self._find_node(line_name)
                if node is None:
                    raise RuntimeError(
                        f"Axis '{line_name}' not found. Did you create the "
                        "mastoid axes in Step 4 and place their second points?"
                    )
                if node.GetNumberOfDefinedControlPoints() < 2:
                    raise RuntimeError(
                        f"Axis '{line_name}' has only "
                        f"{node.GetNumberOfDefinedControlPoints()} defined "
                        "point(s). Place its second control point in Step 4."
                    )
                return np.array(node.GetNthControlPointPositionWorld(1))

            maiR_endpoint = _manual_endpoint("main axis of right mastoid I")
            maiL_endpoint = _manual_endpoint("main axis of left mastoid I")

            # ---- SagR / SagL: parasagittal, normal = LR, through obsR / obsL ----
            sagR = self._create_perpendicular_plane(
                "SagR", obsR_origin, LR, (1.0, 0.4, 0.4))
            sagL = self._create_perpendicular_plane(
                "SagL", obsL_origin, LR, (1.0, 0.6, 0.6))

            # ---- CorR / CorL: coronal, normal = AP, through mastoid endpoints ----
            corR = self._create_perpendicular_plane(
                "CorR", maiR_endpoint, AP, (0.4, 1.0, 0.4))
            corL = self._create_perpendicular_plane(
                "CorL", maiL_endpoint, AP, (0.6, 1.0, 0.6))

            self.planeNodes["SagR"] = sagR
            self.planeNodes["SagL"] = sagL
            self.planeNodes["CorR"] = corR
            self.planeNodes["CorL"] = corL

            self.step5Status.setText(
                "Created four planes:\n"
                f"  SagR (parasagittal, normal={LR}) through obsR\n"
                f"  SagL (parasagittal, normal={LR}) through obsL\n"
                f"  CorR (coronal, normal={AP}) through mastoid-R endpoint\n"
                f"  CorL (coronal, normal={AP}) through mastoid-L endpoint\n"
                "All four are perpendicular to plane P."
            )
        except ValueError as e:
            self.step5Status.setText(f"Error locating landmark: {e}")
        except Exception as e:
            self.step5Status.setText(f"Error creating planes: {e}")
        self.statusLabel.setText(self.step5Status.text)

    # ==================================================================
    # STEP 6 — Linear measurements
    # ==================================================================
    # Linear distances from Table 9 of the protocol. Each entry is
    # (line_name, source_node, label_A, label_B). The line is created from
    # the current positions of A and B in the named source node.
    LINEAR_MEASUREMENTS = [
        ("ear height L",  "soft", "saL",  "sbaL"),
        ("ear height R",  "soft", "saR",  "sbaR"),
        ("ear width R",               "soft", "praR", "paR"),
        ("ear width L",               "soft", "praL", "paL"),
        ("ear insertion height R",    "soft", "obsR", "obiR"),
        ("ear insertion height L",    "soft", "obsL", "obiL"),
        ("nasal dorsum length",       "soft", "se",   "prn"),
        ("soft nose height 1",        "soft", "se",   "sn"),
        ("soft nose height 2",        "soft", "g'",   "sn"),
        ("hard nose height 1",        "hard", "n",    "ss"),
        ("hard nose height 2",        "hard", "g",    "ss"),
        ("facial height",             "hard", "n",    "gn"),
        ("ear protrusion R obs-hx",   "soft", "obsR", "hxR"),
        ("ear protrusion L obs-hx",   "soft", "obsL", "hxL"),
        ("for hard nose angle",       "hard", "n",    "rhi"),
    ]

    # Final export column order (36 columns).
    # Each entry is (header_name, node_name_or_None).
    # None means the value comes from the metadata form.
    EXPORT_COLUMNS = [
        ("ID",                       None),
        ("age",                      None),
        ("sex",                      None),
        ("source",                   None),
        ("ear angle L",              "ear angle L"),
        ("ear angle R",              "ear angle R"),
        ("ear height L",             "ear height L"),
        ("ear height R",             "ear height R"),
        ("ear insertion angle L",    "ear insertion angle L"),
        ("ear insertion angle R",    "ear insertion angle R"),
        ("ear insertion height L",   "ear insertion height L"),
        ("ear insertion height R",   "ear insertion height R"),
        ("ear protrusion L",         "ear protrusion L obs-hx"),
        ("ear protrusion R",         "ear protrusion R obs-hx"),
        ("ear width L",              "ear width L"),
        ("ear width R",              "ear width R"),
        ("facial height",            "facial height"),
        ("hard nose angle",          "hard nose angle"),
        ("hard nose height 1",       "hard nose height 1"),
        ("hard nose height 2",       "hard nose height 2"),
        ("mandible ramus angle L",   "mandible ramus angle L"),
        ("mandible ramus angle R",   "mandible ramus angle R"),
        ("mastoid anterior angle L", "mastoid anterior angle L"),
        ("mastoid anterior angle R", "mastoid anterior angle R"),
        ("mastoid height L",         "mastoid height L"),
        ("mastoid height R",         "mastoid height R"),
        ("mastoid lateral angle L",  "mastoid lateral angle L"),
        ("mastoid lateral angle R",  "mastoid lateral angle R"),
        ("nasal dorsum length",      "nasal dorsum length"),
        ("nasal root angle",         "nasal root angle"),
        ("soft nose height 1",       "soft nose height 1"),
        ("soft nose height 2",       "soft nose height 2"),
        ("R earlobe",                None),
        ("L earlobe",                None),
        ("L supramastoid crest",     None),
        ("R supramastoid crest",     None),
    ]
    # Names of an existing FHP plane that trigger the "already aligned" path
    FHP_PLANE_NAMES = ("P", "FHP", "Frankfort", "Frankfort horizontal")

    # Landmark renames applied when a pre-existing FHP plane is detected.
    # Key = current label (lowercased for lookup), value = coded label.
    # Includes both the raw FH4 file labels and the already-coded forms
    # so the operation is idempotent.
    FHP_LANDMARK_RENAMES = {
        "por1": "poR",
        "por2": "poL",
        "orb1": "zyoL",
        "orb2": "zyoR",
        "por":  "poR",
        "pol":  "poL",
        "zyol": "zyoL",
        "zyor": "zyoR",
    }

    def createStep6_LinearMeasurements(self):
        w = qt.QWidget()
        layout = qt.QVBoxLayout(w)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        t = qt.QLabel("Step 6 — Linear measurements")
        t.setStyleSheet("font-weight: bold; font-size: 15px;")
        layout.addWidget(t)

        info = qt.QLabel(
            "Creates 15 linear distance nodes from the soft and hard tissue "
            "landmarks, following Table 9 of the protocol. The three "
            "reference lines <code>ear protrusion R obs-hx</code>, "
            "<code>ear protrusion L obs-hx</code>, and "
            "<code>for hard nose angle</code> are also created — they will "
            "feed the angle measurements in Step 7.<br><br>"
            "<b>Manual work after running:</b><br>"
            "&nbsp;&nbsp;• Enable <i>Interaction handles</i> in the Markups "
            "module and adjust the line <b>for hard nose angle</b> so it "
            "crosses plane <b>P</b>. Use only the red arrow to preserve the "
            "line's orientation.<br>"
            "&nbsp;&nbsp;• The three reference lines (two ear protrusions, "
            "one for hard nose angle) are <b>excluded from the database</b> — "
            "they exist only to feed the angle measurements."
        )
        info.setTextFormat(qt.Qt.RichText)
        info.setWordWrap(True)
        layout.addWidget(info)

        btn = qt.QPushButton("Create the 15 linear measurement lines")
        btn.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold; padding: 10px;"
        )
        btn.clicked.connect(self.onCreateLinearMeasurementsClicked)
        layout.addWidget(btn)

        btn_extend = qt.QPushButton("Extend 'for hard nose angle' to cross plane P")
        btn_extend.setStyleSheet(
            "background-color: #FF9800; color: white; font-weight: bold; padding: 10px;"
        )
        btn_extend.setToolTip(
            "Lengthens the line along its current n -> rhi direction "
            "so it crosses plane P. The direction is preserved; only the "
            "endpoints move outward."
        )
        btn_extend.clicked.connect(self.onExtendHardNoseLineClicked)
        layout.addWidget(btn_extend)
        btn_copy = qt.QPushButton("Copy all complete line measurements to clipboard")
        btn_copy.setStyleSheet(
            "background-color: #27AE60; color: white; font-weight: bold; padding: 10px;"
        )
        btn_copy.setToolTip(
            "Same as Ctrl+M. Paste the result into Excel or a database."
        )
        btn_copy.clicked.connect(self.onCopyLineMeasurementsClicked)
        layout.addWidget(btn_copy)

        self.step6Status = qt.QLabel("Status: idle.")
        self.step6Status.setWordWrap(True)
        self.step6Status.setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Maximum)
        self.step6Status.setStyleSheet(
            "padding: 10px; background-color: #f0f0f0; border-radius: 5px;"
        )
        layout.addWidget(self.step6Status)

        layout.addStretch(1)
        self._add_step(w)

    def onExtendHardNoseLineClicked(self):
        """Extend the 'for hard nose angle' line along its own direction
        until both endpoints lie clearly beyond plane P. The direction
        defined by the n -> rhi landmark pair is preserved exactly."""
        line = self._find_node("for hard nose angle")
        if line is None:
            self.step6Status.setText(
                "Error: 'for hard nose angle' not found. Create the linear "
                "measurements first."
            )
            return

        plane = self._find_node("P")
        if plane is None:
            self.step6Status.setText(
                "Error: plane 'P' not found. Finish Step 3 first."
            )
            return

        if line.GetNumberOfDefinedControlPoints() < 2:
            self.step6Status.setText(
                "Error: 'for hard nose angle' needs two defined control points."
            )
            return

        # Endpoints in world RAS
        p0 = np.array(line.GetNthControlPointPositionWorld(0))
        p1 = np.array(line.GetNthControlPointPositionWorld(1))

        # Direction of the line as created (p0 -> p1); this is n -> rhi.
        direction = p1 - p0
        length = np.linalg.norm(direction)
        if length < 1e-9:
            self.step6Status.setText(
                "Error: 'for hard nose angle' has zero length; cannot extend."
            )
            return
        unit_dir = direction / length

        # Plane P geometry
        plane_origin = np.zeros(3); plane.GetOriginWorld(plane_origin)
        plane_normal = np.zeros(3); plane.GetNormalWorld(plane_normal)
        n_norm = np.linalg.norm(plane_normal)
        if n_norm < 1e-9:
            self.step6Status.setText("Error: plane P has a zero-length normal.")
            return
        plane_normal = plane_normal / n_norm

        # Signed distances of the two endpoints from the plane
        s0 = float(np.dot(p0 - plane_origin, plane_normal))
        s1 = float(np.dot(p1 - plane_origin, plane_normal))

        # We want to extend in both directions until the endpoints are
        # comfortably beyond the plane. Use a healthy margin so the line
        # visibly crosses P rather than just touching it.
        margin = 30.0  # mm beyond the plane on each side

        # Extend backward from p0 along -unit_dir until s0 = -margin
        # (i.e. on the negative side of the plane by at least `margin`).
        # Extend forward from p1 along +unit_dir until s1 = +margin.

        # The plane crossing parameter t where s(t) = 0:
        #   s0 + t * (s1 - s0) = 0   ->  t = -s0 / (s1 - s0)
        denom = (s1 - s0)
        if abs(denom) < 1e-9:
            self.step6Status.setText(
                "Error: 'for hard nose angle' is parallel to plane P; "
                "cannot force it to cross. Check the n and rhi landmarks."
            )
            return
        t_cross = -s0 / denom           # in [0, 1] if the segment crosses P
        cross_pt = p0 + t_cross * (p1 - p0)

        # Distance from cross_pt to the nearest existing endpoint, plus
        # margin, gives us how far to extend each side.
        new_p0 = cross_pt - unit_dir * margin
        new_p1 = cross_pt + unit_dir * margin

        # Write the new endpoint positions back
        was_modified = line.StartModify()
        line.SetNthControlPointPositionWorld(0, *new_p0)
        line.SetNthControlPointPositionWorld(1, *new_p1)
        line.EndModify(was_modified)

        # Report
        self.step6Status.setText(
            "Extended 'for hard nose angle' along its n -> rhi direction.\n"
            f"  Plane crossing at: "
            f"[{cross_pt[0]:.1f}, {cross_pt[1]:.1f}, {cross_pt[2]:.1f}]\n"
            f"  New endpoints: "
            f"[{new_p0[0]:.1f}, {new_p0[1]:.1f}, {new_p0[2]:.1f}] and "
            f"[{new_p1[0]:.1f}, {new_p1[1]:.1f}, {new_p1[2]:.1f}]\n"
            f"  Direction preserved; each side extended {margin:.0f} mm "
            "beyond plane P."
        )
        self.statusLabel.setText(self.step6Status.text)

    def onCreateLinearMeasurementsClicked(self):
        soft = self.loadedLandmarks.get("soft_tissue") or self._find_node("soft tissue")
        hard = self.loadedLandmarks.get("hard_tissue") or self._find_node("hard tissue")
        if not soft or not hard:
            self.step6Status.setText(
                "Error: need both 'soft tissue' and 'hard tissue' loaded first."
            )
            return

        created, failed = [], []
        for line_name, source, a_label, b_label in self.LINEAR_MEASUREMENTS:
            src = soft if source == "soft" else hard
            try:
                pa = self._landmark_position(src, a_label)
                pb = self._landmark_position(src, b_label)
            except ValueError as e:
                failed.append(f"{line_name} ({e})")
                continue

            old = self._find_node(line_name)
            if old is not None:
                slicer.mrmlScene.RemoveNode(old)
            line = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", line_name)
            dn = line.GetDisplayNode()
            if dn is not None:
                dn.SetColor(0.4, 1.0, 1.0)
                dn.SetSelectedColor(0.6, 0.0, 0.5)
                dn.SetPointLabelsVisibility(False)
            line.AddControlPoint(pa)
            line.AddControlPoint(pb)
            created.append(line_name)

        msg = f"Created {len(created)} of {len(self.LINEAR_MEASUREMENTS)} linear measurement lines."
        if failed:
            msg += " Failed: " + " | ".join(failed)
        msg += (
            "\n\nNext: adjust 'for hard nose angle' so it crosses plane P, "
            "then use the copy button."
        )
        self.step6Status.setText(msg)
        self.statusLabel.setText(msg)

    def onCopyLineMeasurementsClicked(self):
        """Same behaviour as the protocol's Ctrl+M shortcut: copy all
        complete line node measurements to the clipboard."""
        measurements = []
        lineNodes = slicer.util.getNodesByClass("vtkMRMLMarkupsLineNode")
        for lineNode in lineNodes:
            if lineNode.GetNumberOfDefinedControlPoints() < 2:
                continue
            try:
                length = lineNode.GetMeasurement("length").GetValue()
            except Exception:
                continue
            measurements.append(f"{lineNode.GetName()}\t{length:.2f}")

        text = "\n".join(measurements) + ("\n" if measurements else "")
        qt.QApplication.clipboard().setText(text)
        msg = f"Copied {len(measurements)} line measurements to the clipboard."
        self.step6Status.setText(msg)
        self.statusLabel.setText(msg)

    # ==================================================================
    # STEP 7 — Intersections, lateral angle lines, projected lines, angles, export
    # ==================================================================
    # Mapping of line -> plane for the intersection script.
    # Source: Table 10 of the protocol.
    INTERSECTION_TARGETS = [
        # (intersection_node_name, line_name, plane_name)
        ("EA R intersection",  "ear height R",           "P"),
        ("EA L intersection",  "ear height L",           "P"),
        ("EIA R intersection", "ear insertion height R", "P"),
        ("EIA L intersection", "ear insertion height L", "P"),
        ("MAA R intersection", "main axis of right mastoid L", "P"),
        ("MAA L intersection", "main axis of left mastoid L",  "P"),
        ("MLA R intersection", "main axis of right mastoid I", "CorR"),
        ("MLA L intersection", "main axis of left mastoid I",  "CorL"),
        ("MRA R intersection", "mandibular ramus axis R", "P"),
        ("MRA L intersection", "mandibular ramus axis L", "P"),
        ("NRA intersection",   "nasal dorsum length",     "P"),
        ("HNA intersection",   "for hard nose angle",     "P"),
    ]

    # Angle specifications from Table 11 of the protocol.
    # (angle_name, A_source, A_label, vertex_source, vertex_label)
    ANGLE_SPECS = [
        # (angle_name, A_source, A_label, vertex_source, vertex_label, b_rule)
        ("ear angle R",             "soft", "sbaR", "intersection", "EA R intersection",
            ("projected", "ear height R projected", "anterior")),
        ("ear angle L",             "soft", "sbaL", "intersection", "EA L intersection",
            ("projected", "ear height L projected", "anterior")),
        ("ear insertion angle R",   "soft", "obiR", "intersection", "EIA R intersection",
            ("projected", "ear insertion height R projected", "anterior")),
        ("ear insertion angle L",   "soft", "obiL", "intersection", "EIA L intersection",
            ("projected", "ear insertion height L projected", "anterior")),
        ("ear protrusion R",        "soft", "hxR",  "landmark", "obsR",
            ("projected", "ear protrusion R obs-hx projected", "posterior")),
        ("ear protrusion L",        "soft", "hxL",  "landmark", "obsL",
            ("projected", "ear protrusion L obs-hx projected", "posterior")),
        ("mastoid anterior angle R","hard", "maiR", "intersection", "MAA R intersection",
            ("projected", "main axis of right mastoid L projected", "anterior")),
        ("mastoid anterior angle L","hard", "maiL", "intersection", "MAA L intersection",
            ("projected", "main axis of left mastoid L projected", "anterior")),
        ("mastoid lateral angle R", "hard", "maiR", "intersection", "MLA R intersection",
            ("projected", "for R mastoid lateral angle projected", "lateralmost")),
        ("mastoid lateral angle L", "hard", "maiL", "intersection", "MLA L intersection",
            ("projected", "for L mastoid lateral angle projected", "lateralmost")),
        ("mandible ramus angle R",  "hard", "goR",  "intersection", "MRA R intersection",
            ("projected", "mandibular ramus axis R projected", "medial")),
        ("mandible ramus angle L",  "hard", "goL",  "intersection", "MRA L intersection",
            ("projected", "mandibular ramus axis L projected", "medial")),
        ("nasal root angle",        "soft", "prn",  "intersection", "NRA intersection",
            ("projected", "nasal dorsum length projected", "anterior")),
        ("hard nose angle",         "hard", "n",    "intersection", "HNA intersection",
            ("projected", "for hard nose angle projected", "anterior")),
    ]

    def createStep7_AnglesAndExport(self):
        w = qt.QWidget()
        layout = qt.QVBoxLayout(w)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        t = qt.QLabel("Step 7 — Intersections, angles, and export")
        t.setStyleSheet("font-weight: bold; font-size: 15px;")
        layout.addWidget(t)

        info = qt.QLabel(

            "This step runs five sub-operations, all triggered by buttons. "
            "Do them in order; each one depends on the previous one.<br><br>"
            "<b>7a.</b> Compute the 12 intersection points where the "
            "measurement lines meet their respective planes (Table 10).<br>"
            "<b>7b.</b> Create the two mastoid lateral angle lines "
            "<code>for L mastoid lateral angle</code> and "
            "<code>for R mastoid lateral angle</code>, both collinear with "
            "the MLA R - MLA L axis.<br>"
            "<b>7c.</b> Project the reference lines onto their planes "
            "(produces &lt;name&gt; projected nodes).<br>"
            "<b>7d.</b> Create the 14 angle nodes with all three control "
            "points placed automatically.<br>"
            "<b>7e.</b> Copy all angle measurements to the clipboard."
        )
        
        info.setTextFormat(qt.Qt.RichText)
        info.setWordWrap(True)
        layout.addWidget(info)

        btn_ints = qt.QPushButton("7a. Compute the 12 intersection points")
        btn_ints.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 8px;")
        btn_ints.clicked.connect(self.onComputeIntersectionsClicked)
        layout.addWidget(btn_ints)

        btn_lat = qt.QPushButton("7b. Populate the two mastoid lateral angle lines from MLA R and MLA L")
        btn_lat.setStyleSheet("background-color: #9966CC; color: white; font-weight: bold; padding: 8px;")
        btn_lat.clicked.connect(self.onPopulateLateralAngleLinesClicked)
        layout.addWidget(btn_lat)

        btn_proj = qt.QPushButton("7c. Project reference lines onto their planes")
        btn_proj.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; padding: 8px;")
        btn_proj.clicked.connect(self.onProjectLinesClicked)
        layout.addWidget(btn_proj)

        btn_ang = qt.QPushButton("7d. Create the 14 angle nodes (all three points)")
        btn_ang.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        btn_ang.clicked.connect(self.onPrepopulateAnglesClicked)
        layout.addWidget(btn_ang)

        btn_copy = qt.QPushButton("7e. Copy all angle measurements to clipboard")
        btn_copy.setStyleSheet("background-color: #27AE60; color: white; font-weight: bold; padding: 8px;")
        btn_copy.clicked.connect(self.onCopyAngleMeasurementsClicked)
        layout.addWidget(btn_copy)

        self.step7Status = qt.QLabel("Status: idle.")
        self.step7Status.setWordWrap(True)
        self.step7Status.setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Maximum)
        self.step7Status.setStyleSheet(
            "padding: 10px; background-color: #f0f0f0; border-radius: 5px;"
        )
        layout.addWidget(self.step7Status)

        layout.addStretch(1)
        self._add_step(w)

    # ------------------------------------------------------------------
    # 7a — intersections
    # ------------------------------------------------------------------
    def _find_intersection_of_line_and_plane(self, line_node, plane_node):
        """Return the world-space intersection point of a line node and a
        plane node, or None if they are parallel."""
        line_dir = np.array(line_node.GetLineStartPositionWorld()) - \
                   np.array(line_node.GetLineEndPositionWorld())
        norm = np.linalg.norm(line_dir)
        if norm < 1e-9:
            return None
        line_dir = line_dir / norm
        line_pt = np.array(line_node.GetLineStartPositionWorld())

        plane_norm = np.array(plane_node.GetNormalWorld())
        plane_norm = plane_norm / np.linalg.norm(plane_norm)
        plane_pt = np.array(plane_node.GetOriginWorld())

        denom = float(np.dot(line_dir, plane_norm))
        if abs(denom) < 1e-9:
            return None
        t = float(np.dot(plane_pt - line_pt, plane_norm)) / denom
        return line_pt + t * line_dir

    def onComputeIntersectionsClicked(self):
        """Compute the 12 intersection points and store them as fiducial
        nodes (one point each)."""
        created, failed = [], []
        for node_name, line_name, plane_name in self.INTERSECTION_TARGETS:
            line = self._find_node(line_name)
            plane = self._find_node(plane_name)
            if line is None:
                failed.append(f"{node_name}: line '{line_name}' missing")
                continue
            if plane is None:
                failed.append(f"{node_name}: plane '{plane_name}' missing")
                continue
            if line.GetNumberOfDefinedControlPoints() < 2:
                failed.append(f"{node_name}: line '{line_name}' has fewer than 2 defined points")
                continue

            pt = self._find_intersection_of_line_and_plane(line, plane)
            if pt is None:
                failed.append(f"{node_name}: line and plane are parallel")
                continue

            old = self._find_node(node_name)
            if old is not None:
                slicer.mrmlScene.RemoveNode(old)
            node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode", node_name)
            node.AddControlPointWorld(*pt)
            dn = node.GetDisplayNode()
            if dn is not None:
                dn.SetColor(1.0, 0.4, 0.0)
                dn.SetSelectedColor(1.0, 0.0, 0.0)
            created.append(node_name)

        msg = f"Created {len(created)} of {len(self.INTERSECTION_TARGETS)} intersections."
        if failed:
            msg += "\nFailures:\n  " + "\n  ".join(failed)
        self.step7Status.setText(msg)
        self.statusLabel.setText(msg)
    # ------------------------------------------------------------------
    # FHP-aligned direction helpers
    # ------------------------------------------------------------------
    def _get_fhp_frame(self):
        """Return (ap_axis, lr_axis, is_axis, midsagittal_x) in world RAS.

        - ap_axis: unit vector pointing anteriorly (from plane P's normal)
        - lr_axis: unit vector pointing to the patient's LEFT
        - is_axis: unit vector pointing superiorly (world +Z)
        - midsagittal_x: X coordinate of the midsagittal plane, derived from
          the soft tissue midline landmarks (se, prn, sn, g').
        Raises RuntimeError if plane P is missing or the midline landmarks
        cannot be found.
        """
        p = self._find_node("P")
        if not p:
            raise RuntimeError("Plane 'P' not found — cannot determine AP axis.")

        ap = np.zeros(3); p.GetNormalWorld(ap)
        n = np.linalg.norm(ap)
        if n < 1e-9:
            raise RuntimeError("Plane 'P' has a zero-length normal.")
        ap = ap / n
        # Force anterior to be +Y (the flip in Step 3 normally ensures this,
        # but be defensive).
        if ap[1] < 0:
            ap = -ap

        is_axis = np.array([0.0, 0.0, 1.0])
        lr_axis = np.cross(ap, is_axis)
        ln = np.linalg.norm(lr_axis)
        if ln < 1e-9:
            # Degenerate: plane P happens to be horizontal. Fall back to
            # world X for LR.
            lr_axis = np.array([1.0, 0.0, 0.0])
        else:
            lr_axis = lr_axis / ln
        # The cross product ap x is may point left or right depending on
        # the AP orientation. Make it point to the patient's LEFT (+X).
        if lr_axis[0] < 0:
            lr_axis = -lr_axis

        # Midsagittal X from the midline soft-tissue landmarks
        soft = self.loadedLandmarks.get("soft_tissue") or self._find_node("soft tissue")
        mid_x = 0.0
        if soft is not None:
            try:
                mids = [
                    self._landmark_position(soft, "se"),
                    self._landmark_position(soft, "prn"),
                    self._landmark_position(soft, "sn"),
                    self._landmark_position(soft, "g'"),
                ]
                mid_x = float(np.median([m[0] for m in mids]))
            except ValueError:
                mid_x = 0.0
        return ap, lr_axis, is_axis, mid_x

    def _endpoint_along_axis(self, line_node, axis):
        """Return the endpoint of line_node that lies furthest along `axis`
        (unit vector). Returns a numpy array of shape (3,)."""
        p0 = np.array(line_node.GetNthControlPointPositionWorld(0))
        p1 = np.array(line_node.GetNthControlPointPositionWorld(1))
        # Positive projection = further along axis
        if np.dot(p0, axis) >= np.dot(p1, axis):
            return p0
        return p1
    def _endpoint_lateralmost(self, line_node, mid_x):
        """Return the endpoint of line_node that is FURTHEST from the
        midsagittal plane, i.e. the endpoint with the largest
        |X - mid_x|. This is the definition of 'most lateral' used by
        the mastoid lateral angle B point."""
        p0 = np.array(line_node.GetNthControlPointPositionWorld(0))
        p1 = np.array(line_node.GetNthControlPointPositionWorld(1))
        if abs(p0[0] - mid_x) >= abs(p1[0] - mid_x):
            return p0
        return p1
    def _endpoint_medial(self, line_node, mid_x):
        """Return the endpoint of line_node closer to the midsagittal plane
        (|X - mid_x| is smaller)."""
        p0 = np.array(line_node.GetNthControlPointPositionWorld(0))
        p1 = np.array(line_node.GetNthControlPointPositionWorld(1))
        if abs(p0[0] - mid_x) <= abs(p1[0] - mid_x):
            return p0
        return p1
    # ------------------------------------------------------------------
    # 7b — lateral angle lines
    # ------------------------------------------------------------------
    def onPopulateLateralAngleLinesClicked(self):
        """Create two mastoid lateral angle lines, one per side, both
        collinear with the mediolateral axis defined by MLA R and MLA L:

          - 'for R mastoid lateral angle': starts at MLA R and extends
            EXTENSION mm further outward (rightward).
          - 'for L mastoid lateral angle': starts at MLA L and extends
            EXTENSION mm further outward (leftward).

        The direction is taken from (MLA R - MLA L), so both side lines
        share the same axis. No manual toggling is required.
        """
        mla_r = self._find_node("MLA R intersection")
        mla_l = self._find_node("MLA L intersection")
        if not mla_r or not mla_l:
            self.step7Status.setText(
                "Error: 'MLA R intersection' or 'MLA L intersection' not found. "
                "Run 7a first."
            )
            return

        p_r = np.array(mla_r.GetNthControlPointPositionWorld(0))
        p_l = np.array(mla_l.GetNthControlPointPositionWorld(0))

        # Mediolateral direction (L -> R). Both side lines use this axis.
        axis = p_r - p_l
        axis_len = np.linalg.norm(axis)
        if axis_len < 1e-6:
            self.step7Status.setText(
                "Error: MLA R and MLA L coincide; cannot define axis."
            )
            return
        unit_axis = axis / axis_len

        EXTENSION = 60.0   # mm of lateral extension beyond each MLA point

        # Optional grey reference line so the user sees the shared axis
        ref_name = "mastoid lateral axis (reference)"
        old = self._find_node(ref_name)
        if old is not None:
            slicer.mrmlScene.RemoveNode(old)
        ref = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsLineNode", ref_name)
        dn = ref.GetDisplayNode()
        if dn is not None:
            dn.SetColor(0.55, 0.55, 0.55)
            dn.SetSelectedColor(0.55, 0.55, 0.55)
            dn.SetPointLabelsVisibility(False)
        ref.AddControlPoint(p_l)
        ref.AddControlPoint(p_r)

        # Two side lines.
        #   for R: start at MLA R, extend outward along +unit_axis
        #   for L: start at MLA L, extend outward along -unit_axis
        sides = [
            ("for R mastoid lateral angle", p_r, +1.0),
            ("for L mastoid lateral angle", p_l, -1.0),
        ]
        for line_name, mla_pt, sign in sides:
            old = self._find_node(line_name)
            if old is not None:
                slicer.mrmlScene.RemoveNode(old)
            line = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsLineNode", line_name)
            dn = line.GetDisplayNode()
            if dn is not None:
                dn.SetColor(1.0, 0.6, 0.9)
                dn.SetSelectedColor(1.0, 0.0, 0.5)
                dn.SetPointLabelsVisibility(True)
            line.AddControlPoint(mla_pt)
            line.AddControlPoint(mla_pt + sign * EXTENSION * unit_axis)

        self.step7Status.setText(
            "Created two side lines, collinear with the MLA R - MLA L axis:\n"
            f"  'for R mastoid lateral angle': MLA R + {EXTENSION:.0f} mm outward\n"
            f"  'for L mastoid lateral angle': MLA L + {EXTENSION:.0f} mm outward\n"
            f"  'mastoid lateral axis (reference)' created for visual check.\n\n"
            "Run 7c next."
        )
        self.statusLabel.setText(self.step7Status.text)


    # ------------------------------------------------------------------
    # 7c — projected reference lines
    # ------------------------------------------------------------------
    # (line_name, plane_name) — lines to project, and onto which plane.
    PROJECTION_TARGETS = [
        ("ear height R",              "P"),
        ("ear height L",              "P"),
        ("ear insertion height R",    "P"),
        ("ear insertion height L",    "P"),
        ("nasal dorsum length",       "P"),
        ("for hard nose angle",       "P"),
        ("ear protrusion R obs-hx",   "SagR"),
        ("ear protrusion L obs-hx",   "SagL"),
        ("main axis of right mastoid L", "P"),
        ("main axis of left mastoid L",  "P"),
        ("mandibular ramus axis R",   "P"),
        ("mandibular ramus axis L",   "P"),
        ("for L mastoid lateral angle", "CorL"),
        ("for R mastoid lateral angle", "CorR"),
    ]

    def onProjectLinesClicked(self):
        """For each target line, produce a projected copy onto its plane.
        The projected node is named '<original> projected'."""
        created, failed = [], []
        for line_name, plane_name in self.PROJECTION_TARGETS:
            line = self._find_node(line_name)
            plane = self._find_node(plane_name)
            if line is None:
                failed.append(f"{line_name}: line missing")
                continue
            if plane is None:
                failed.append(f"{line_name}: plane '{plane_name}' missing")
                continue
            if line.GetNumberOfDefinedControlPoints() < 2:
                failed.append(f"{line_name}: fewer than 2 defined points")
                continue

            # Get plane origin/normal
            planeOrigin = np.zeros(3); plane.GetOriginWorld(planeOrigin)
            planeNormal = np.zeros(3); plane.GetNormalWorld(planeNormal)
            planeNormal = planeNormal / np.linalg.norm(planeNormal)

            proj_name = line_name + " projected"
            old = self._find_node(proj_name)
            if old is not None:
                slicer.mrmlScene.RemoveNode(old)
            proj = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsLineNode", proj_name)
            dn = proj.GetDisplayNode()
            if dn is not None:
                dn.SetColor(0.4, 0.9, 0.4)
                dn.SetSelectedColor(0.0, 1.0, 0.0)
                dn.SetPointLabelsVisibility(False)

            for i in range(2):
                p = np.array(line.GetNthControlPointPositionWorld(i))
                dist = float(np.dot(p - planeOrigin, planeNormal))
                proj.AddControlPointWorld(*(p - planeNormal * dist))

            created.append(proj_name)

        msg = f"Created {len(created)} projected lines."
        if failed:
            msg += "\nSkipped:\n  " + "\n  ".join(failed)
        self.step7Status.setText(msg)
        self.statusLabel.setText(msg)

    # ------------------------------------------------------------------
    # 7d — pre-populate angle nodes
    # ------------------------------------------------------------------
    def onPrepopulateAnglesClicked(self):
        """Create the 14 angle nodes, each with A, vertex, and B
        pre-populated automatically.

        B is chosen from the projected line associated with the angle,
        picking the endpoint at the 'anterior', 'posterior', 'medial',
        'lateral_left' or 'lateral_right' end according to Table 11.
        """
        soft = self.loadedLandmarks.get("soft_tissue") or self._find_node("soft tissue")
        hard = self.loadedLandmarks.get("hard_tissue") or self._find_node("hard tissue")
        if not soft or not hard:
            self.step7Status.setText("Error: soft and hard tissue must be loaded.")
            return

        # FHP frame for direction picking
        try:
            ap_axis, lr_axis, is_axis, mid_x = self._get_fhp_frame()
        except RuntimeError as e:
            self.step7Status.setText(f"Error computing FHP frame: {e}")
            return

        created, failed = [], []
        for spec in self.ANGLE_SPECS:
            angle_name, a_source, a_label, v_source, v_label, b_rule = spec
            src_a = soft if a_source == "soft" else hard

            # ---- A point ----
            try:
                a_pt = self._landmark_position(src_a, a_label)
            except ValueError as e:
                failed.append(f"{angle_name}: A ({a_label}) not found ({e})")
                continue

            # ---- Vertex point ----
            try:
                if v_source == "intersection":
                    v_node = self._find_node(v_label)
                    if not v_node:
                        failed.append(f"{angle_name}: vertex '{v_label}' missing")
                        continue
                    vertex_pt = np.array(v_node.GetNthControlPointPositionWorld(0))
                elif v_source == "landmark":
                    src_v = soft if a_source == "soft" else hard
                    vertex_pt = self._landmark_position(src_v, v_label)
                else:
                    failed.append(f"{angle_name}: unknown vertex source {v_source}")
                    continue
            except ValueError as e:
                failed.append(f"{angle_name}: vertex ({v_label}) not found ({e})")
                continue

            # ---- B point (auto, from projected line endpoint) ----
            b_pt = None
            if b_rule is not None:
                kind, target_line_name, end_spec = b_rule
                if kind != "projected":
                    failed.append(f"{angle_name}: unknown b_rule kind {kind}")
                    continue
                proj_line = self._find_node(target_line_name)
                if not proj_line:
                    failed.append(
                        f"{angle_name}: projected line '{target_line_name}' missing "
                        "(did you run 7c?)"
                    )
                    continue
                if proj_line.GetNumberOfDefinedControlPoints() < 2:
                    failed.append(
                        f"{angle_name}: '{target_line_name}' has < 2 defined points"
                    )
                    continue

                if end_spec == "anterior":
                    b_pt = self._endpoint_along_axis(proj_line, ap_axis)
                elif end_spec == "posterior":
                    b_pt = self._endpoint_along_axis(proj_line, -ap_axis)
                elif end_spec == "lateral_left":
                    b_pt = self._endpoint_along_axis(proj_line, lr_axis)
                elif end_spec == "lateral_right":
                    b_pt = self._endpoint_along_axis(proj_line, -lr_axis)
                elif end_spec == "lateralmost":
                    b_pt = self._endpoint_lateralmost(proj_line, mid_x)
                elif end_spec == "medial":
                    b_pt = self._endpoint_medial(proj_line, mid_x)
                else:
                    failed.append(f"{angle_name}: unknown end_spec {end_spec}")
                    continue

            # ---- Guard against degenerate angles ----
            if b_pt is not None:
                MIN_ARM = 0.5
                dAV = float(np.linalg.norm(a_pt - vertex_pt))
                dBV = float(np.linalg.norm(b_pt - vertex_pt))
                if dAV < MIN_ARM or dBV < MIN_ARM:
                    failed.append(
                        f"{angle_name}: arm too short "
                        f"(A-V={dAV:.2f} mm, B-V={dBV:.2f} mm) — "
                        "check the vertex and the projected line"
                    )
                    continue

            # ---- Create the angle node ----
            old = self._find_node(angle_name)
            if old is not None:
                slicer.mrmlScene.RemoveNode(old)
            ang = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsAngleNode", angle_name)
            dn = ang.GetDisplayNode()
            if dn is not None:
                dn.SetColor(1.0, 0.8, 0.0)
                dn.SetSelectedColor(1.0, 0.5, 0.0)
                dn.SetPointLabelsVisibility(True)

            # Add order: A, vertex, B — matches the protocol's
            # (initial arm, vertex, terminal arm) convention.
            ang.AddControlPoint(a_pt)
            ang.AddControlPoint(vertex_pt)
            if b_pt is not None:
                ang.AddControlPoint(b_pt)
            created.append(angle_name)

        msg = (
            f"Created {len(created)} of {len(self.ANGLE_SPECS)} angle nodes, "
            "each with all three control points placed automatically.\n\n"
            "Verify visually in the 3D view. If any angle looks wrong, "
            "drag the corresponding endpoint to fix it — the measurement "
            "updates live."
        )
        if failed:
            msg += "\n\nSome angles could not be created:\n  " + "\n  ".join(failed)
        self.step7Status.setText(msg)
        self.statusLabel.setText(msg)
    # ------------------------------------------------------------------
    # 7e — angle export
    # ------------------------------------------------------------------
    def onCopyAngleMeasurementsClicked(self):
        """Copy all complete angle node measurements to the clipboard."""
        measurements = []
        angleNodes = slicer.util.getNodesByClass("vtkMRMLMarkupsAngleNode")
        for angleNode in angleNodes:
            if angleNode.GetNumberOfDefinedControlPoints() < 3:
                continue
            try:
                val = angleNode.GetMeasurement("angle").GetValue()
            except Exception:
                continue
            measurements.append(f"{angleNode.GetName()}\t{val:.2f}")

        text = "\n".join(measurements) + ("\n" if measurements else "")
        qt.QApplication.clipboard().setText(text)
        msg = f"Copied {len(measurements)} complete angle measurements to the clipboard."
        self.step7Status.setText(msg)
        self.statusLabel.setText(msg)
    # ==================================================================
    # STEP 8 — Metadata and final export
    # ==================================================================
    def createStep8_MetadataAndExport(self):
        w = qt.QWidget()
        layout = qt.QVBoxLayout(w)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        t = qt.QLabel("Step 8 — Subject metadata and final export")
        t.setStyleSheet("font-weight: bold; font-size: 15px;")
        layout.addWidget(t)

        info = qt.QLabel(
            "Fill in the subject metadata and observations below, then click "
            "the copy button at the bottom of this step. The button copies "
            "a single-row, tab-separated table containing one column per "
            "measurement, with the metadata and observations in the "
            "remaining columns. Paste directly into Excel or the database."
        )
        info.setTextFormat(qt.Qt.RichText)
        info.setWordWrap(True)
        layout.addWidget(info)

        # ---------- Subject block ----------
        subj_group = qt.QGroupBox("Subject")
        subj_form = qt.QFormLayout(subj_group)

        self.meta_id = qt.QLineEdit()
        subj_form.addRow("ID:", self.meta_id)

        self.meta_age = qt.QSpinBox()
        self.meta_age.setRange(0, 130)
        self.meta_age.setSuffix(" years")
        subj_form.addRow("Age:", self.meta_age)

        self.meta_sex = qt.QComboBox()
        self.meta_sex.addItems(["", "Male", "Female", "Unknown"])
        subj_form.addRow("Biological sex:", self.meta_sex)

        self.meta_source = qt.QLineEdit()
        subj_form.addRow("Source:", self.meta_source)

        layout.addWidget(subj_group)

        # ---------- Observations block ----------
        obs_group = qt.QGroupBox("Observations")
        obs_form = qt.QFormLayout(obs_group)

        self.meta_earlobe_R = qt.QComboBox()
        self.meta_earlobe_R.addItems(["", "Free", "Attached"])
        obs_form.addRow("Right earlobe morphology:", self.meta_earlobe_R)

        self.meta_earlobe_L = qt.QComboBox()
        self.meta_earlobe_L.addItems(["", "Free", "Attached"])
        obs_form.addRow("Left earlobe morphology:", self.meta_earlobe_L)

        self.meta_crest_L = qt.QComboBox()
        self.meta_crest_L.addItems(["", "Well-developed", "Smooth"])
        obs_form.addRow("Left supramastoid crest:", self.meta_crest_L)

        self.meta_crest_R = qt.QComboBox()
        self.meta_crest_R.addItems(["", "Well-developed", "Smooth"])
        obs_form.addRow("Right supramastoid crest:", self.meta_crest_R)

        layout.addWidget(obs_group)
        # ---------- Copy options ----------
        self.include_header_checkbox = qt.QCheckBox(
            "Include column header row in the copied text"
        )
        self.include_header_checkbox.setChecked(True)
        self.include_header_checkbox.setToolTip(
            "When checked, the clipboard contains two rows: a header row "
            "and a data row. When unchecked, only the data row is copied — "
            "useful when pasting underneath an existing header."
        )
        layout.addWidget(self.include_header_checkbox)

        # ---------- Copy button ----------
        btn_copy = qt.QPushButton(
            "Copy all results with metadata to clipboard"
        )
        btn_copy.setStyleSheet(
            "background-color: #27AE60; color: white; font-weight: bold; "
            "padding: 12px; font-size: 14px;"
        )
        btn_copy.setToolTip(
            "Produces a tab-separated table: header row plus one data row, "
            "one column per measurement."
        )
        btn_copy.clicked.connect(self.onCopyAllWithMetadataClicked)
        layout.addWidget(btn_copy)

        self.step8Status = qt.QLabel("Status: idle.")
        self.step8Status.setWordWrap(True)
        self.step8Status.setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Maximum)
        self.step8Status.setStyleSheet(
            "padding: 10px; background-color: #f0f0f0; border-radius: 5px;"
        )
        layout.addWidget(self.step8Status)

        layout.addStretch(1)
        self._add_step(w)

    def _measurement_value(self, node_name):
        """Return the numeric measurement (length or angle) of the named
        node as a formatted string, or '' if the node is missing or
        incomplete."""
        n = self._find_node(node_name)
        if not n:
            return ""
        try:
            cls = n.GetClassName()
            if cls == "vtkMRMLMarkupsLineNode":
                if n.GetNumberOfDefinedControlPoints() < 2:
                    return ""
                return f"{n.GetMeasurement('length').GetValue():.2f}"
            if cls == "vtkMRMLMarkupsAngleNode":
                if n.GetNumberOfDefinedControlPoints() < 3:
                    return ""
                return f"{n.GetMeasurement('angle').GetValue():.2f}"
        except Exception:
            return ""
        return ""

    def onCopyAllWithMetadataClicked(self):
        """Copy a single-row, one-column-per-measurement table to the
        clipboard. Column order is defined by EXPORT_COLUMNS. Missing or
        incomplete nodes produce an empty cell so column alignment is
        preserved."""
        # ---- Read metadata ----
        subject_id = self.meta_id.text.strip()
        age = str(self.meta_age.value)
        sex = self.meta_sex.currentText
        source = self.meta_source.text.strip()

        ear_r = self.meta_earlobe_R.currentText
        ear_l = self.meta_earlobe_L.currentText
        crest_l = self.meta_crest_L.currentText
        crest_r = self.meta_crest_R.currentText

        meta_by_col = {
            "ID": subject_id,
            "age": age,
            "sex": sex,
            "source": source,
            "R earlobe": ear_r,
            "L earlobe": ear_l,
            "L supramastoid crest": crest_l,
            "R supramastoid crest": crest_r,
        }

        # ---- Build header and data row ----
        header_cells = [col for col, _ in self.EXPORT_COLUMNS]
        row_cells = []
        n_filled = 0
        n_expected = 0

        for col, node_name in self.EXPORT_COLUMNS:
            if node_name is None:
                row_cells.append(meta_by_col.get(col, ""))
            else:
                n_expected += 1
                val = self._measurement_value(node_name)
                row_cells.append(val)
                if val:
                    n_filled += 1
        if self.include_header_checkbox.isChecked():
            text = "\t".join(header_cells) + "\n" + "\t".join(row_cells) + "\n"
        else:
            text = "\t".join(row_cells) + "\n"
        qt.QApplication.clipboard().setText(text)

        # ---- Report ----
        header_mode = "with header" if self.include_header_checkbox.isChecked() else "data only"
        self.step8Status.setText(
            f"Copied 1 row with {len(header_cells)} columns ({header_mode}).\n"
            f"Measurements filled: {n_filled} / {n_expected}.\n"
            f"Subject: ID='{subject_id}', Age='{age}', Sex='{sex}', "
            f"Source='{source}'."
        )
        self.statusLabel.setText(self.step8Status.text)
    # ==================================================================
    # PLACEHOLDERS
    # ==================================================================
    def _placeholder_step(self, title, note):
        w = qt.QWidget()
        layout = qt.QVBoxLayout(w)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)
        t = qt.QLabel(title)
        t.setStyleSheet("font-weight: bold; font-size: 15px;")
        layout.addWidget(t)
        info = qt.QLabel(note)
        info.setTextFormat(qt.Qt.RichText)
        info.setWordWrap(True)
        layout.addWidget(info)
        layout.addStretch(1)
        self._add_step(w)

    

    # ==================================================================
    # Navigation
    # ==================================================================
    def onPrevButtonClicked(self):
        if self.currentStep > 0:
            self.currentStep -= 1
            self.updateStepUI()

    def onNextButtonClicked(self):
        if self.currentStep < self.stepStack.count - 1:
            self.currentStep += 1
            self.updateStepUI()

    def updateStepUI(self):
        self.stepStack.setCurrentIndex(self.currentStep)
        self.stepLabel.setText(f"Step {self.currentStep + 1} / {self.stepStack.count}")
        self.prevButton.setEnabled(self.currentStep > 0)
        self.nextButton.setEnabled(self.currentStep < self.stepStack.count - 1)
        page = self.stepStack.currentWidget()
        if page is not None:
            page.updateGeometry()
        self.layout().activate()


if hasattr(slicer, "EarApproximationGUIWidget"):
    try:
        w = slicer.EarApproximationGUIWidget
        if w and not w.isHidden():
            w.raise_()
            w.activateWindow()
            w.show()
        else:
            raise AttributeError
    except Exception:
        slicer.EarApproximationGUIWidget = EarApproximationGUI()
else:
    slicer.EarApproximationGUIWidget = EarApproximationGUI()

```
