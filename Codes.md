1. Script for Frankfort horizontal plane alignment

``` python

import numpy
scene = slicer.mrmlScene
F = getNodesByClass('vtkMRMLMarkupsFiducialNode')
F = F[0]

# Get the fiducial IDs of porions and zygomatico - orbitale  suture from the fiducial list by name
po1_id = -1; po2_id = -1; zyo_id = -1;

for i in range(0, F.GetNumberOfControlPoints()):
	if F.GetNthControlPointLabel(i) == 'poR' :
		po1_id = i
	if F.GetNthControlPointLabel(i) == 'poL' :
		po2_id = i
	if F.GetNthControlPointLabel(i) == 'zyoL' :
		zyo_id = i


# Get the coordinates of landmarks
po1 =[0, 0, 0]
po2 =[0, 0, 0]
zyo =[0, 0, 0]

F.GetNthControlPointPosition(po1_id,po1)
F.GetNthControlPointPosition(po2_id,po2)
F.GetNthControlPointPosition(zyo_id,zyo)

# The vector between two porions that we will align to LR axis by calculating the yaw angle
po =[po1[0] - po2[0], po1[1] -po2[1], po1[2]-po2[2]]
vTransform = vtk.vtkTransform()
vTransform.RotateZ(-numpy.arctan2(po[1], po[0])*180/numpy.pi)
transform = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLinearTransformNode')
transform.SetMatrixTransformToParent(vTransform.GetMatrix())

# Apply the transform to the fiducials and the volume
transform = slicer.vtkMRMLLinearTransformNode()

scene.AddNode(transform) 
transform.SetMatrixTransformToParent(vTransform.GetMatrix())
V = getNodesByClass('vtkMRMLScalarVolumeNode')
V = V[0]
V.SetAndObserveTransformNodeID(transform.GetID())
F.SetAndObserveTransformNodeID(transform.GetID())

# Get the updated (transformed) coordinates from the list
po1 =[0, 0, 0]
po2 =[0, 0, 0]
zyo =[0, 0, 0]

F.GetNthControlPointPosition(po1_id,po1)
F.GetNthControlPointPosition(po2_id,po2)
F.GetNthControlPointPosition(zyo_id,zyo)

# Apply the transform to the coordinates
po1 = vTransform.GetMatrix().MultiplyPoint([po1[0], po1[1], po1[2], 0])
po2 = vTransform.GetMatrix().MultiplyPoint([po2[0], po2[1], po2[2], 0])
zyo = vTransform.GetMatrix().MultiplyPoint([zyo[0], zyo[1], zyo[2], 0])
po =[po1[0]-po2[0], po1[1]-po2[1], po1[2]-po2[2]]

# Calculate the rotation for the roll 
vTransform2 = vtk.vtkTransform()

vTransform2.RotateY(numpy.arctan2(po[2], po[0])*180/numpy.pi)

# Apply the transform to previous transform hierarchy
transform2 = slicer.vtkMRMLLinearTransformNode()
scene.AddNode(transform2) 
transform2.SetMatrixTransformToParent(vTransform2.GetMatrix())
transform.SetAndObserveTransformNodeID(transform2.GetID())

# Get the coordinates again
po1 =[0, 0, 0]
po2 =[0, 0, 0]
zyo =[0, 0, 0]

F.GetNthControlPointPosition(po1_id,po1)
F.GetNthControlPointPosition(po2_id,po2)
F.GetNthControlPointPosition(zyo_id,zyo)

# Apply transforms to points to get updated coordinates
po1 = vTransform.GetMatrix().MultiplyPoint([po1[0], po1[1], po1[2], 0])
po2 = vTransform.GetMatrix().MultiplyPoint([po2[0], po2[1], po2[2], 0])
zyo = vTransform.GetMatrix().MultiplyPoint([zyo[0], zyo[1], zyo[2], 0])
po1 = vTransform2.GetMatrix().MultiplyPoint([po1[0], po1[1], po1[2], 0])
po2 = vTransform2.GetMatrix().MultiplyPoint([po2[0], po2[1], po2[2], 0])
zyo = vTransform2.GetMatrix().MultiplyPoint([zyo[0], zyo[1], zyo[2], 0])

# The vector for pitch angle
po_zyo =[zyo[0]-(po1[0]+po2[0])/2, zyo[1]-(po1[1]+po2[1])/2, zyo[2]-(po1[2]+po2[2])/2]

# Calculate the transform for aligning pitch
vTransform3 = vtk.vtkTransform()
vTransform3.RotateX(-numpy.arctan2(po_zyo[2], po_zyo[1])*180/numpy.pi)

# Apply the transform to both fiducial list and the volume
transform3 = slicer.vtkMRMLLinearTransformNode()
scene.AddNode(transform3) 
transform3.SetMatrixTransformToParent(vTransform3.GetMatrix())
transform2.SetAndObserveTransformNodeID(transform3.GetID())

slicer.vtkSlicerTransformLogic().hardenTransform(V)
```

2. Script to pre-populate manual axes with already established landmarks 

```python
G=getNode('hard tissue')
L=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsLineNode')
firstPoint = G.GetNthControlPointPositionVector(4)
L.AddControlPoint(firstPoint)
L.SetName('mastoid height R')


G=getNode('hard tissue')
L=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsLineNode')
firstPoint = G.GetNthControlPointPositionVector(5)
L.AddControlPoint(firstPoint)
L.SetName('mastoid height L')


G=getNode('hard tissue')
L=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsLineNode')
firstPoint = G.GetNthControlPointPositionVector(6)
L.AddControlPoint(firstPoint)
L.SetName('main axis of left mastoid I')

G=getNode('hard tissue')
L=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsLineNode')
firstPoint = G.GetNthControlPointPositionVector(6)
L.AddControlPoint(firstPoint)
L.SetName('main axis of left mastoid L')


G=getNode('hard tissue')
L=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsLineNode')
firstPoint = G.GetNthControlPointPositionVector(7)
L.AddControlPoint(firstPoint)
L.SetName('main axis of right mastoid I')


G=getNode('hard tissue')
L=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsLineNode')
firstPoint = G.GetNthControlPointPositionVector(7)
L.AddControlPoint(firstPoint)
L.SetName('main axis of right mastoid L')


G=getNode('hard tissue')
L=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsLineNode')
firstPoint = G.GetNthControlPointPositionVector(9)
L.AddControlPoint(firstPoint)
L.SetName('mandibular ramus axis R') 

G=getNode('hard tissue')
L=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsLineNode')
firstPoint = G.GetNthControlPointPositionVector(10)
L.AddControlPoint(firstPoint)
L.SetName('mandibular ramus axis L')
```

3. Script for orthogonal projection of msR and msL for mastoid height measurement 

```python
plane = getNode('P')
line = getNode('mastoid height L')
import numpy as np
point = np.zeros(3)
planeOrigin = np.zeros(3)
planeNormal = np.zeros(3)
plane.GetOriginWorld(planeOrigin)
plane.GetNormalWorld(planeNormal)
line.GetNthControlPointPositionWorld(0, point)
distanceFromPlane = np.dot(np.subtract(point, planeOrigin), planeNormal)
line.SetNthControlPointPositionWorld(1, *(point-planeNormal*distanceFromPlane))


plane = getNode('P')
line = getNode('mastoid height R')
import numpy as np
point = np.zeros(3)
planeOrigin = np.zeros(3)
planeNormal = np.zeros(3)
plane.GetOriginWorld(planeOrigin)
plane.GetNormalWorld(planeNormal)
line.GetNthControlPointPositionWorld(0, point)
distanceFromPlane = np.dot(np.subtract(point, planeOrigin), planeNormal)
line.SetNthControlPointPositionWorld(1, *(point-planeNormal*distanceFromPlane))line.SetNthControlPointPositionWorld(1, *(point-planeNormal*distanceFromPlane))
```

4. Script to establish linear measurements and additional supporting linear distances via the allocated landmarks
   
```python
F=getNode('soft tissue')
L=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsLineNode')
firstPoint = F.GetNthControlPointPositionVector(4)
L.AddControlPoint(firstPoint)
secondPoint = F.GetNthControlPointPositionVector(5)
L.AddControlPoint(secondPoint)
L.SetName('ear height R')

F=getNode('soft tissue')
L=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsLineNode')
firstPoint = F.GetNthControlPointPositionVector(6)
L.AddControlPoint(firstPoint)
secondPoint = F.GetNthControlPointPositionVector(7)
L.AddControlPoint(secondPoint)
L.SetName('ear height L')

F=getNode('soft tissue')
L=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsLineNode')
firstPoint = F.GetNthControlPointPositionVector(9)
L.AddControlPoint(firstPoint)
secondPoint = F.GetNthControlPointPositionVector(10)
L.AddControlPoint(secondPoint)
L.SetName('ear width R')

F=getNode('soft tissue')
L=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsLineNode')
firstPoint = F.GetNthControlPointPositionVector(8)
L.AddControlPoint(firstPoint)
secondPoint = F.GetNthControlPointPositionVector(11)
L.AddControlPoint(secondPoint)
L.SetName('ear width L')

F=getNode('soft tissue')
L=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsLineNode')
firstPoint = F.GetNthControlPointPositionVector(2)
L.AddControlPoint(firstPoint)
secondPoint = F.GetNthControlPointPositionVector(3)
L.AddControlPoint(secondPoint)
L.SetName('ear insertion height R')


F=getNode('soft tissue')
L=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsLineNode')
firstPoint = F.GetNthControlPointPositionVector(0)
L.AddControlPoint(firstPoint)
secondPoint = F.GetNthControlPointPositionVector(1)
L.AddControlPoint(secondPoint)
L.SetName('ear insertion height L')

F=getNode('soft tissue')
L=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsLineNode')
firstPoint = F.GetNthControlPointPositionVector(14)
L.AddControlPoint(firstPoint)
secondPoint = F.GetNthControlPointPositionVector(15)
L.AddControlPoint(secondPoint)
L.SetName('nasal dorsum length')

F=getNode('soft tissue')
L=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsLineNode')
firstPoint = F.GetNthControlPointPositionVector(14)
L.AddControlPoint(firstPoint)
secondPoint = F.GetNthControlPointPositionVector(17)
L.AddControlPoint(secondPoint)
L.SetName('soft nose height 1')

F=getNode('soft tissue')
L=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsLineNode')
firstPoint = F.GetNthControlPointPositionVector(16)
L.AddControlPoint(firstPoint)
secondPoint = F.GetNthControlPointPositionVector(17)
L.AddControlPoint(secondPoint)
L.SetName('soft nose height 2')

G=getNode('hard tissue')
L=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsLineNode')
firstPoint = G.GetNthControlPointPositionVector(1)
L.AddControlPoint(firstPoint)
secondPoint = G.GetNthControlPointPositionVector(3)
L.AddControlPoint(secondPoint)
L.SetName('hard nose height 1')


G=getNode('hard tissue')
L=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsLineNode')
firstPoint = G.GetNthControlPointPositionVector(0)
L.AddControlPoint(firstPoint)
secondPoint = G.GetNthControlPointPositionVector(3)
L.AddControlPoint(secondPoint)
L.SetName('hard nose height 2')

G=getNode('hard tissue')
L=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsLineNode')
firstPoint = G.GetNthControlPointPositionVector(1)
L.AddControlPoint(firstPoint)
secondPoint = G.GetNthControlPointPositionVector(8)
L.AddControlPoint(secondPoint)
L.SetName('facial height')

F=getNode('soft tissue')
L=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsLineNode')
firstPoint = F.GetNthControlPointPositionVector(2)
L.AddControlPoint(firstPoint)
secondPoint = F.GetNthControlPointPositionVector(12)
L.AddControlPoint(secondPoint)
L.SetName('ear protrusion R obs-hx')


F=getNode('soft tissue')
L=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsLineNode')
firstPoint = F.GetNthControlPointPositionVector(0)
L.AddControlPoint(firstPoint)
secondPoint = F.GetNthControlPointPositionVector(13)
L.AddControlPoint(secondPoint)
L.SetName('ear protrusion L obs-hx')


G=getNode('hard tissue')
L=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsLineNode')
firstPoint = G.GetNthControlPointPositionVector(1)
L.AddControlPoint(firstPoint)
secondPoint = G.GetNthControlPointPositionVector(2)
L.AddControlPoint(secondPoint)
L.SetName('for hard nose angle')
```

5. Script to copy linear measurements to clipboard via shortcut (Ctrl+M) 
```python
def createMeasurements():
  for nodeName in[]:
    lineNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", nodeName)
    lineNode.CreateDefaultDisplayNodes()
    dn = lineNode.GetDisplayNode()
    # Use crosshair glyph to allow more accurate point placement
    dn.SetGlyphTypeFromString("CrossDot2D")
    # Hide measurement result while markup up
    lineNode.GetMeasurement('length').SetEnabled(False)

shortcut1 = qt.QShortcut(slicer.util.mainWindow())
shortcut1.setKey(qt.QKeySequence("Ctrl+m"))
shortcut1.connect( 'activated()', createMeasurements)
```

```python
def copyLineMeasurementsToClipboard():
  measurements =[]
  # Collect all line measurements from the scene
  lineNodes = getNodesByClass('vtkMRMLMarkupsLineNode')
  for lineNode in lineNodes:
    if lineNode.GetNumberOfDefinedControlPoints() < 2:
      # incomplete line, skip it
      continue
    # Get node filename that the length was measured on
    try:
      volumeNode = slicer.mrmlScene.GetNodeByID(lineNode.GetNthControlPointAssociatedNodeID(0))
      imagePath = volumeNode.GetStorageNode().GetFileName()
    except:
      imagePath = '(unknown)'
    # Get line node n
    measurementName = lineNode.GetName()
    # Get length measurement
    lineNode.GetMeasurement('length').SetEnabled(True)
    length = str(lineNode.GetMeasurement('length').GetValue())
    # Add fields to results
    measurements.append('\t'.join([imagePath, measurementName, length]))
  # Copy all measurements to clipboard (to be pasted into Excel)
  outputText = "\n".join(measurements) + "\n"
  slicer.app.clipboard().setText(outputText)
  slicer.util.delayDisplay(f"Copied {len(measurements)} length measurements to the clipboard.")

shortcut2 = qt.QShortcut(slicer.util.mainWindow())
shortcut2.setKey(qt.QKeySequence("Ctrl+m"))
shortcut2.connect( 'activated()', copyLineMeasurementsToClipboard)
```

6. Script for generating intersection points (finding the coordinates of points where respective planes are intersected by linear structures)

``` python

# Get inputs from nodes
lineNode = getNode('ear height R')
planeNode = getNode('P')
# Create output node
intersectionPointNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', "EA R intersection")

import numpy as np

# Define a line by its direction vector and a point on it
line_dir = np.array(lineNode.GetLineStartPositionWorld())-np.array(lineNode.GetLineEndPositionWorld())
line_dir /= np.linalg.norm(line_dir)
line_pt = lineNode.GetLineStartPositionWorld()

# Define a plane by its normal vector and a point on it
plane_norm = planeNode.GetNormalWorld()
plane_pt = planeNode.GetOriginWorld()

###

# Compute the dot product of the line direction and the plane normal
dot_prod = sum([a*b for a,b in zip(line_dir, plane_norm)])

# Check if the dot product is zero, which means the line is parallel to the plane
if dot_prod == 0:
    print("The line is parallel to the plane. No intersection point.")
else:
    # Compute the parameter t that gives the intersection point
    t = sum([(a-b)*c for a,b,c in zip(plane_pt, line_pt, plane_norm)]) / dot_prod

    # Compute the intersection point by plugging t into the line equation
    inter_pt =[a + b*t for a,b in zip(line_pt, line_dir)]

    # Print the intersection point
    print("The intersection point is", inter_pt)

###

# Save result into the output node
intersectionPointNode.AddControlPointWorld(inter_pt)


lineNode = getNode('ear height L')
planeNode = getNode('P')
# Create output node
intersectionPointNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', "EA L intersection")

import numpy as np

# Define a line by its direction vector and a point on it
line_dir = np.array(lineNode.GetLineStartPositionWorld())-np.array(lineNode.GetLineEndPositionWorld())
line_dir /= np.linalg.norm(line_dir)
line_pt = lineNode.GetLineStartPositionWorld()

# Define a plane by its normal vector and a point on it
plane_norm = planeNode.GetNormalWorld()
plane_pt = planeNode.GetOriginWorld()

###

# Compute the dot product of the line direction and the plane normal
dot_prod = sum([a*b for a,b in zip(line_dir, plane_norm)])

# Check if the dot product is zero, which means the line is parallel to the plane
if dot_prod == 0:
    print("The line is parallel to the plane. No intersection point.")
else:
    # Compute the parameter t that gives the intersection point
    t = sum([(a-b)*c for a,b,c in zip(plane_pt, line_pt, plane_norm)]) / dot_prod

    # Compute the intersection point by plugging t into the line equation
    inter_pt =[a + b*t for a,b in zip(line_pt, line_dir)]

    # Print the intersection point
    print("The intersection point is", inter_pt)

###

# Save result into the output node
intersectionPointNode.AddControlPointWorld(inter_pt)


lineNode = getNode('ear insertion height R')
planeNode = getNode('P')
# Create output node
intersectionPointNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', "EIA R intersection")

import numpy as np

# Define a line by its direction vector and a point on it
line_dir = np.array(lineNode.GetLineStartPositionWorld())-np.array(lineNode.GetLineEndPositionWorld())
line_dir /= np.linalg.norm(line_dir)
line_pt = lineNode.GetLineStartPositionWorld()

# Define a plane by its normal vector and a point on it
plane_norm = planeNode.GetNormalWorld()
plane_pt = planeNode.GetOriginWorld()

###

# Compute the dot product of the line direction and the plane normal
dot_prod = sum([a*b for a,b in zip(line_dir, plane_norm)])

# Check if the dot product is zero, which means the line is parallel to the plane
if dot_prod == 0:
    print("The line is parallel to the plane. No intersection point.")
else:
    # Compute the parameter t that gives the intersection point
    t = sum([(a-b)*c for a,b,c in zip(plane_pt, line_pt, plane_norm)]) / dot_prod

    # Compute the intersection point by plugging t into the line equation
    inter_pt =[a + b*t for a,b in zip(line_pt, line_dir)]

    # Print the intersection point
    print("The intersection point is", inter_pt)

###

# Save result into the output node
intersectionPointNode.AddControlPointWorld(inter_pt)


lineNode = getNode('ear insertion height L')
planeNode = getNode('P')
# Create output node
intersectionPointNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', "EIA L intersection")

import numpy as np

# Define a line by its direction vector and a point on it
line_dir = np.array(lineNode.GetLineStartPositionWorld())-np.array(lineNode.GetLineEndPositionWorld())
line_dir /= np.linalg.norm(line_dir)
line_pt = lineNode.GetLineStartPositionWorld()

# Define a plane by its normal vector and a point on it
plane_norm = planeNode.GetNormalWorld()
plane_pt = planeNode.GetOriginWorld()

###

# Compute the dot product of the line direction and the plane normal
dot_prod = sum([a*b for a,b in zip(line_dir, plane_norm)])

# Check if the dot product is zero, which means the line is parallel to the plane
if dot_prod == 0:
    print("The line is parallel to the plane. No intersection point.")
else:
    # Compute the parameter t that gives the intersection point
    t = sum([(a-b)*c for a,b,c in zip(plane_pt, line_pt, plane_norm)]) / dot_prod

    # Compute the intersection point by plugging t into the line equation
    inter_pt =[a + b*t for a,b in zip(line_pt, line_dir)]

    # Print the intersection point
    print("The intersection point is", inter_pt)

###

# Save result into the output node
intersectionPointNode.AddControlPointWorld(inter_pt)



# Get inputs from nodes
lineNode = getNode('main axis of right mastoid L')
planeNode = getNode('P')
# Create output node
intersectionPointNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', "MAA R intersection")

import numpy as np

# Define a line by its direction vector and a point on it
line_dir = np.array(lineNode.GetLineStartPositionWorld())-np.array(lineNode.GetLineEndPositionWorld())
line_dir /= np.linalg.norm(line_dir)
line_pt = lineNode.GetLineStartPositionWorld()

# Define a plane by its normal vector and a point on it
plane_norm = planeNode.GetNormalWorld()
plane_pt = planeNode.GetOriginWorld()

###

# Compute the dot product of the line direction and the plane normal
dot_prod = sum([a*b for a,b in zip(line_dir, plane_norm)])

# Check if the dot product is zero, which means the line is parallel to the plane
if dot_prod == 0:
    print("The line is parallel to the plane. No intersection point.")
else:
    # Compute the parameter t that gives the intersection point
    t = sum([(a-b)*c for a,b,c in zip(plane_pt, line_pt, plane_norm)]) / dot_prod

    # Compute the intersection point by plugging t into the line equation
    inter_pt =[a + b*t for a,b in zip(line_pt, line_dir)]

    # Print the intersection point
    print("The intersection point is", inter_pt)

###

# Save result into the output node
intersectionPointNode.AddControlPointWorld(inter_pt)


# Get inputs from nodes
lineNode = getNode('main axis of left mastoid L')
planeNode = getNode('P')
# Create output node
intersectionPointNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', "MAA L intersection")

import numpy as np

# Define a line by its direction vector and a point on it
line_dir = np.array(lineNode.GetLineStartPositionWorld())-np.array(lineNode.GetLineEndPositionWorld())
line_dir /= np.linalg.norm(line_dir)
line_pt = lineNode.GetLineStartPositionWorld()

# Define a plane by its normal vector and a point on it
plane_norm = planeNode.GetNormalWorld()
plane_pt = planeNode.GetOriginWorld()

###

# Compute the dot product of the line direction and the plane normal
dot_prod = sum([a*b for a,b in zip(line_dir, plane_norm)])

# Check if the dot product is zero, which means the line is parallel to the plane
if dot_prod == 0:
    print("The line is parallel to the plane. No intersection point.")
else:
    # Compute the parameter t that gives the intersection point
    t = sum([(a-b)*c for a,b,c in zip(plane_pt, line_pt, plane_norm)]) / dot_prod

    # Compute the intersection point by plugging t into the line equation
    inter_pt =[a + b*t for a,b in zip(line_pt, line_dir)]

    # Print the intersection point
    print("The intersection point is", inter_pt)

###

# Save result into the output node
intersectionPointNode.AddControlPointWorld(inter_pt)


# Get inputs from nodes
lineNode = getNode('main axis of right mastoid I')
planeNode = getNode('CorR')
# Create output node
intersectionPointNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', "MLA R intersection")

import numpy as np

# Define a line by its direction vector and a point on it
line_dir = np.array(lineNode.GetLineStartPositionWorld())-np.array(lineNode.GetLineEndPositionWorld())
line_dir /= np.linalg.norm(line_dir)
line_pt = lineNode.GetLineStartPositionWorld()

# Define a plane by its normal vector and a point on it
plane_norm = planeNode.GetNormalWorld()
plane_pt = planeNode.GetOriginWorld()

###

# Compute the dot product of the line direction and the plane normal
dot_prod = sum([a*b for a,b in zip(line_dir, plane_norm)])

# Check if the dot product is zero, which means the line is parallel to the plane
if dot_prod == 0:
    print("The line is parallel to the plane. No intersection point.")
else:
    # Compute the parameter t that gives the intersection point
    t = sum([(a-b)*c for a,b,c in zip(plane_pt, line_pt, plane_norm)]) / dot_prod

    # Compute the intersection point by plugging t into the line equation
    inter_pt =[a + b*t for a,b in zip(line_pt, line_dir)]

    # Print the intersection point
    print("The intersection point is", inter_pt)

###

# Save result into the output node
intersectionPointNode.AddControlPointWorld(inter_pt)


# Get inputs from nodes
lineNode = getNode('main axis of left mastoid I')
planeNode = getNode('CorL')
# Create output node
intersectionPointNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', "MLA L intersection")

import numpy as np

# Define a line by its direction vector and a point on it
line_dir = np.array(lineNode.GetLineStartPositionWorld())-np.array(lineNode.GetLineEndPositionWorld())
line_dir /= np.linalg.norm(line_dir)
line_pt = lineNode.GetLineStartPositionWorld()

# Define a plane by its normal vector and a point on it
plane_norm = planeNode.GetNormalWorld()
plane_pt = planeNode.GetOriginWorld()

###

# Compute the dot product of the line direction and the plane normal
dot_prod = sum([a*b for a,b in zip(line_dir, plane_norm)])

# Check if the dot product is zero, which means the line is parallel to the plane
if dot_prod == 0:
    print("The line is parallel to the plane. No intersection point.")
else:
    # Compute the parameter t that gives the intersection point
    t = sum([(a-b)*c for a,b,c in zip(plane_pt, line_pt, plane_norm)]) / dot_prod

    # Compute the intersection point by plugging t into the line equation
    inter_pt =[a + b*t for a,b in zip(line_pt, line_dir)]

    # Print the intersection point
    print("The intersection point is", inter_pt)

###

# Save result into the output node
intersectionPointNode.AddControlPointWorld(inter_pt)


# Get inputs from nodes
lineNode = getNode('mandibular ramus axis R')
planeNode = getNode('P')
# Create output node
intersectionPointNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', "MRA R intersection")

import numpy as np

# Define a line by its direction vector and a point on it
line_dir = np.array(lineNode.GetLineStartPositionWorld())-np.array(lineNode.GetLineEndPositionWorld())
line_dir /= np.linalg.norm(line_dir)
line_pt = lineNode.GetLineStartPositionWorld()

# Define a plane by its normal vector and a point on it
plane_norm = planeNode.GetNormalWorld()
plane_pt = planeNode.GetOriginWorld()

###

# Compute the dot product of the line direction and the plane normal
dot_prod = sum([a*b for a,b in zip(line_dir, plane_norm)])

# Check if the dot product is zero, which means the line is parallel to the plane
if dot_prod == 0:
    print("The line is parallel to the plane. No intersection point.")
else:
    # Compute the parameter t that gives the intersection point
    t = sum([(a-b)*c for a,b,c in zip(plane_pt, line_pt, plane_norm)]) / dot_prod

    # Compute the intersection point by plugging t into the line equation
    inter_pt =[a + b*t for a,b in zip(line_pt, line_dir)]

    # Print the intersection point
    print("The intersection point is", inter_pt)

###

# Save result into the output node
intersectionPointNode.AddControlPointWorld(inter_pt)


# Get inputs from nodes
lineNode = getNode('mandibular ramus axis L')
planeNode = getNode('P')
# Create output node
intersectionPointNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', "MRA L intersection")

import numpy as np

# Define a line by its direction vector and a point on it
line_dir = np.array(lineNode.GetLineStartPositionWorld())-np.array(lineNode.GetLineEndPositionWorld())
line_dir /= np.linalg.norm(line_dir)
line_pt = lineNode.GetLineStartPositionWorld()

# Define a plane by its normal vector and a point on it
plane_norm = planeNode.GetNormalWorld()
plane_pt = planeNode.GetOriginWorld()

###

# Compute the dot product of the line direction and the plane normal
dot_prod = sum([a*b for a,b in zip(line_dir, plane_norm)])

# Check if the dot product is zero, which means the line is parallel to the plane
if dot_prod == 0:
    print("The line is parallel to the plane. No intersection point.")
else:
    # Compute the parameter t that gives the intersection point
    t = sum([(a-b)*c for a,b,c in zip(plane_pt, line_pt, plane_norm)]) / dot_prod

    # Compute the intersection point by plugging t into the line equation
    inter_pt =[a + b*t for a,b in zip(line_pt, line_dir)]

    # Print the intersection point
    print("The intersection point is", inter_pt)

###

# Save result into the output node
intersectionPointNode.AddControlPointWorld(inter_pt)


# Get inputs from nodes
lineNode = getNode('for hard nose angle')
planeNode = getNode('P')
# Create output node
intersectionPointNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', "HNA intersection")

import numpy as np

# Define a line by its direction vector and a point on it
line_dir = np.array(lineNode.GetLineStartPositionWorld())-np.array(lineNode.GetLineEndPositionWorld())
line_dir /= np.linalg.norm(line_dir)
line_pt = lineNode.GetLineStartPositionWorld()

# Define a plane by its normal vector and a point on it
plane_norm = planeNode.GetNormalWorld()
plane_pt = planeNode.GetOriginWorld()

###

# Compute the dot product of the line direction and the plane normal
dot_prod = sum([a*b for a,b in zip(line_dir, plane_norm)])

# Check if the dot product is zero, which means the line is parallel to the plane
if dot_prod == 0:
    print("The line is parallel to the plane. No intersection point.")
else:
    # Compute the parameter t that gives the intersection point
    t = sum([(a-b)*c for a,b,c in zip(plane_pt, line_pt, plane_norm)]) / dot_prod

    # Compute the intersection point by plugging t into the line equation
    inter_pt =[a + b*t for a,b in zip(line_pt, line_dir)]

    # Print the intersection point
    print("The intersection point is", inter_pt)

###

# Save result into the output node
intersectionPointNode.AddControlPointWorld(inter_pt)


# Get inputs from nodes
lineNode = getNode('nasal dorsum length')
planeNode = getNode('P')
# Create output node
intersectionPointNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', "NRA intersection")

import numpy as np

# Define a line by its direction vector and a point on it
line_dir = np.array(lineNode.GetLineStartPositionWorld())-np.array(lineNode.GetLineEndPositionWorld())
line_dir /= np.linalg.norm(line_dir)
line_pt = lineNode.GetLineStartPositionWorld()

# Define a plane by its normal vector and a point on it
plane_norm = planeNode.GetNormalWorld()
plane_pt = planeNode.GetOriginWorld()

###

# Compute the dot product of the line direction and the plane normal
dot_prod = sum([a*b for a,b in zip(line_dir, plane_norm)])

# Check if the dot product is zero, which means the line is parallel to the plane
if dot_prod == 0:
    print("The line is parallel to the plane. No intersection point.")
else:
    # Compute the parameter t that gives the intersection point
    t = sum([(a-b)*c for a,b,c in zip(plane_pt, line_pt, plane_norm)]) / dot_prod

    # Compute the intersection point by plugging t into the line equation
    inter_pt =[a + b*t for a,b in zip(line_pt, line_dir)]

    # Print the intersection point
    print("The intersection point is", inter_pt)



# Save result into the output node
intersectionPointNode.AddControlPointWorld(inter_pt)

```

7. Script to generate projected lines for angle measurements

```python

# Get plane projected line

lineNode = getNode('ear height R')
planeNode = getNode('P')

# Create new node for storing the projected line node
projectedLineNode = slicer.mrmlScene.AddNewNodeByClass(lineNode.GetClassName(), lineNode.GetName()+" projected")

# Get transforms
planeToWorld = vtk.vtkMatrix4x4()
planeNode.GetObjectToWorldMatrix(planeToWorld)
worldToPlane = vtk.vtkMatrix4x4()
vtk.vtkMatrix4x4.Invert(planeToWorld, worldToPlane)

# Project each point
for pointIndex in range(2):
    point_World =[*lineNode.GetNthControlPointPositionWorld(pointIndex), 1.0]
    point_Plane = worldToPlane.MultiplyPoint(point_World)
    projectedPoint_Plane =[point_Plane[0], point_Plane[1], 0.0, 1.0]
    projectedPoint_World = planeToWorld.MultiplyPoint(projectedPoint_Plane)
    projectedLineNode.AddControlPoint(projectedPoint_World[0:3])

# Get plane projected line
lineNode = getNode('ear height L')
planeNode = getNode('P')

# Create new node for storing the projected line node
projectedLineNode = slicer.mrmlScene.AddNewNodeByClass(lineNode.GetClassName(), lineNode.GetName()+" projected")

# Get transforms
planeToWorld = vtk.vtkMatrix4x4()
planeNode.GetObjectToWorldMatrix(planeToWorld)
worldToPlane = vtk.vtkMatrix4x4()
vtk.vtkMatrix4x4.Invert(planeToWorld, worldToPlane)

# Project each point
for pointIndex in range(2):
    point_World =[*lineNode.GetNthControlPointPositionWorld(pointIndex), 1.0]
    point_Plane = worldToPlane.MultiplyPoint(point_World)
    projectedPoint_Plane =[point_Plane[0], point_Plane[1], 0.0, 1.0]
    projectedPoint_World = planeToWorld.MultiplyPoint(projectedPoint_Plane)
    projectedLineNode.AddControlPoint(projectedPoint_World[0:3])


# Get plane projected line
lineNode = getNode('ear insertion height R')
planeNode = getNode('P')

# Create new node for storing the projected line node
projectedLineNode = slicer.mrmlScene.AddNewNodeByClass(lineNode.GetClassName(), lineNode.GetName()+" projected")

# Get transforms
planeToWorld = vtk.vtkMatrix4x4()
planeNode.GetObjectToWorldMatrix(planeToWorld)
worldToPlane = vtk.vtkMatrix4x4()
vtk.vtkMatrix4x4.Invert(planeToWorld, worldToPlane)

# Project each point
for pointIndex in range(2):
    point_World =[*lineNode.GetNthControlPointPositionWorld(pointIndex), 1.0]
    point_Plane = worldToPlane.MultiplyPoint(point_World)
    projectedPoint_Plane =[point_Plane[0], point_Plane[1], 0.0, 1.0]
    projectedPoint_World = planeToWorld.MultiplyPoint(projectedPoint_Plane)
    projectedLineNode.AddControlPoint(projectedPoint_World[0:3])

# Get plane projected line
lineNode = getNode('ear insertion height L')
planeNode = getNode('P')

# Create new node for storing the projected line node
projectedLineNode = slicer.mrmlScene.AddNewNodeByClass(lineNode.GetClassName(), lineNode.GetName()+" projected")

# Get transforms
planeToWorld = vtk.vtkMatrix4x4()
planeNode.GetObjectToWorldMatrix(planeToWorld)
worldToPlane = vtk.vtkMatrix4x4()
vtk.vtkMatrix4x4.Invert(planeToWorld, worldToPlane)

# Project each point
for pointIndex in range(2):
    point_World =[*lineNode.GetNthControlPointPositionWorld(pointIndex), 1.0]
    point_Plane = worldToPlane.MultiplyPoint(point_World)
    projectedPoint_Plane =[point_Plane[0], point_Plane[1], 0.0, 1.0]
    projectedPoint_World = planeToWorld.MultiplyPoint(projectedPoint_Plane)
    projectedLineNode.AddControlPoint(projectedPoint_World[0:3])

lineNode = getNode('nasal dorsum length')
planeNode = getNode('P')

# Create new node for storing the projected line node
projectedLineNode = slicer.mrmlScene.AddNewNodeByClass(lineNode.GetClassName(), lineNode.GetName()+" projected")

# Get transforms
planeToWorld = vtk.vtkMatrix4x4()
planeNode.GetObjectToWorldMatrix(planeToWorld)
worldToPlane = vtk.vtkMatrix4x4()
vtk.vtkMatrix4x4.Invert(planeToWorld, worldToPlane)

# Project each point
for pointIndex in range(2):
    point_World =[*lineNode.GetNthControlPointPositionWorld(pointIndex), 1.0]
    point_Plane = worldToPlane.MultiplyPoint(point_World)
    projectedPoint_Plane =[point_Plane[0], point_Plane[1], 0.0, 1.0]
    projectedPoint_World = planeToWorld.MultiplyPoint(projectedPoint_Plane)
    projectedLineNode.AddControlPoint(projectedPoint_World[0:3])

lineNode = getNode('for hard nose angle')
planeNode = getNode('P')

# Create new node for storing the projected line node
projectedLineNode = slicer.mrmlScene.AddNewNodeByClass(lineNode.GetClassName(), lineNode.GetName()+" projected")

# Get transforms
planeToWorld = vtk.vtkMatrix4x4()
planeNode.GetObjectToWorldMatrix(planeToWorld)
worldToPlane = vtk.vtkMatrix4x4()
vtk.vtkMatrix4x4.Invert(planeToWorld, worldToPlane)

# Project each point
for pointIndex in range(2):
    point_World =[*lineNode.GetNthControlPointPositionWorld(pointIndex), 1.0]
    point_Plane = worldToPlane.MultiplyPoint(point_World)
    projectedPoint_Plane =[point_Plane[0], point_Plane[1], 0.0, 1.0]
    projectedPoint_World = planeToWorld.MultiplyPoint(projectedPoint_Plane)
    projectedLineNode.AddControlPoint(projectedPoint_World[0:3])

lineNode = getNode('ear protrusion R obs-hx')
planeNode = getNode('SagR')

# Create new node for storing the projected line node
projectedLineNode = slicer.mrmlScene.AddNewNodeByClass(lineNode.GetClassName(), lineNode.GetName()+" projected")

# Get transforms
planeToWorld = vtk.vtkMatrix4x4()
planeNode.GetObjectToWorldMatrix(planeToWorld)
worldToPlane = vtk.vtkMatrix4x4()
vtk.vtkMatrix4x4.Invert(planeToWorld, worldToPlane)

# Project each point
for pointIndex in range(2):
    point_World =[*lineNode.GetNthControlPointPositionWorld(pointIndex), 1.0]
    point_Plane = worldToPlane.MultiplyPoint(point_World)
    projectedPoint_Plane =[point_Plane[0], point_Plane[1], 0.0, 1.0]
    projectedPoint_World = planeToWorld.MultiplyPoint(projectedPoint_Plane)
    projectedLineNode.AddControlPoint(projectedPoint_World[0:3])

lineNode = getNode('ear protrusion L obs-hx')
planeNode = getNode('SagL')

# Create new node for storing the projected line node
projectedLineNode = slicer.mrmlScene.AddNewNodeByClass(lineNode.GetClassName(), lineNode.GetName()+" projected")

# Get transforms
planeToWorld = vtk.vtkMatrix4x4()
planeNode.GetObjectToWorldMatrix(planeToWorld)
worldToPlane = vtk.vtkMatrix4x4()
vtk.vtkMatrix4x4.Invert(planeToWorld, worldToPlane)

# Project each point
for pointIndex in range(2):
    point_World =[*lineNode.GetNthControlPointPositionWorld(pointIndex), 1.0]
    point_Plane = worldToPlane.MultiplyPoint(point_World)
    projectedPoint_Plane =[point_Plane[0], point_Plane[1], 0.0, 1.0]
    projectedPoint_World = planeToWorld.MultiplyPoint(projectedPoint_Plane)
    projectedLineNode.AddControlPoint(projectedPoint_World[0:3])

lineNode = getNode('main axis of right mastoid L')
planeNode = getNode('P')

# Create new node for storing the projected line node
projectedLineNode = slicer.mrmlScene.AddNewNodeByClass(lineNode.GetClassName(), lineNode.GetName()+" projected")

# Get transforms
planeToWorld = vtk.vtkMatrix4x4()
planeNode.GetObjectToWorldMatrix(planeToWorld)
worldToPlane = vtk.vtkMatrix4x4()
vtk.vtkMatrix4x4.Invert(planeToWorld, worldToPlane)

# Project each point
for pointIndex in range(2):
    point_World =[*lineNode.GetNthControlPointPositionWorld(pointIndex), 1.0]
    point_Plane = worldToPlane.MultiplyPoint(point_World)
    projectedPoint_Plane =[point_Plane[0], point_Plane[1], 0.0, 1.0]
    projectedPoint_World = planeToWorld.MultiplyPoint(projectedPoint_Plane)
    projectedLineNode.AddControlPoint(projectedPoint_World[0:3])

lineNode = getNode('main axis of left mastoid L')
planeNode = getNode('P')

# Create new node for storing the projected line node
projectedLineNode = slicer.mrmlScene.AddNewNodeByClass(lineNode.GetClassName(), lineNode.GetName()+" projected")

# Get transforms
planeToWorld = vtk.vtkMatrix4x4()
planeNode.GetObjectToWorldMatrix(planeToWorld)
worldToPlane = vtk.vtkMatrix4x4()
vtk.vtkMatrix4x4.Invert(planeToWorld, worldToPlane)

# Project each point
for pointIndex in range(2):
    point_World =[*lineNode.GetNthControlPointPositionWorld(pointIndex), 1.0]
    point_Plane = worldToPlane.MultiplyPoint(point_World)
    projectedPoint_Plane =[point_Plane[0], point_Plane[1], 0.0, 1.0]
    projectedPoint_World = planeToWorld.MultiplyPoint(projectedPoint_Plane)
    projectedLineNode.AddControlPoint(projectedPoint_World[0:3])


lineNode = getNode('mandibular ramus axis R')
planeNode = getNode('P')

# Create new node for storing the projected line node
projectedLineNode = slicer.mrmlScene.AddNewNodeByClass(lineNode.GetClassName(), lineNode.GetName()+" projected")

# Get transforms
planeToWorld = vtk.vtkMatrix4x4()
planeNode.GetObjectToWorldMatrix(planeToWorld)
worldToPlane = vtk.vtkMatrix4x4()
vtk.vtkMatrix4x4.Invert(planeToWorld, worldToPlane)

# Project each point
for pointIndex in range(2):
    point_World =[*lineNode.GetNthControlPointPositionWorld(pointIndex), 1.0]
    point_Plane = worldToPlane.MultiplyPoint(point_World)
    projectedPoint_Plane =[point_Plane[0], point_Plane[1], 0.0, 1.0]
    projectedPoint_World = planeToWorld.MultiplyPoint(projectedPoint_Plane)
    projectedLineNode.AddControlPoint(projectedPoint_World[0:3])

lineNode = getNode('mandibular ramus axis L')
planeNode = getNode('P')

# Create new node for storing the projected line node
projectedLineNode = slicer.mrmlScene.AddNewNodeByClass(lineNode.GetClassName(), lineNode.GetName()+" projected")

# Get transforms
planeToWorld = vtk.vtkMatrix4x4()
planeNode.GetObjectToWorldMatrix(planeToWorld)
worldToPlane = vtk.vtkMatrix4x4()
vtk.vtkMatrix4x4.Invert(planeToWorld, worldToPlane)

# Project each point
for pointIndex in range(2):
    point_World =[*lineNode.GetNthControlPointPositionWorld(pointIndex), 1.0]
    point_Plane = worldToPlane.MultiplyPoint(point_World)
    projectedPoint_Plane =[point_Plane[0], point_Plane[1], 0.0, 1.0]
    projectedPoint_World = planeToWorld.MultiplyPoint(projectedPoint_Plane)
    projectedLineNode.AddControlPoint(projectedPoint_World[0:3])


lineNode = getNode('for L mastoid lateral angle')
planeNode = getNode('CorL')

# Create new node for storing the projected line node
projectedLineNode = slicer.mrmlScene.AddNewNodeByClass(lineNode.GetClassName(), lineNode.GetName()+" projected")

# Get transforms
planeToWorld = vtk.vtkMatrix4x4()
planeNode.GetObjectToWorldMatrix(planeToWorld)
worldToPlane = vtk.vtkMatrix4x4()
vtk.vtkMatrix4x4.Invert(planeToWorld, worldToPlane)

# Project each point
for pointIndex in range(2):
    point_World =[*lineNode.GetNthControlPointPositionWorld(pointIndex), 1.0]
    point_Plane = worldToPlane.MultiplyPoint(point_World)
    projectedPoint_Plane =[point_Plane[0], point_Plane[1], 0.0, 1.0]
    projectedPoint_World = planeToWorld.MultiplyPoint(projectedPoint_Plane)
    projectedLineNode.AddControlPoint(projectedPoint_World[0:3])


lineNode = getNode('for R mastoid lateral angle')
planeNode = getNode('CorR')

# Create new node for storing the projected line node
projectedLineNode = slicer.mrmlScene.AddNewNodeByClass(lineNode.GetClassName(), lineNode.GetName()+" projected")

# Get transforms
planeToWorld = vtk.vtkMatrix4x4()
planeNode.GetObjectToWorldMatrix(planeToWorld)
worldToPlane = vtk.vtkMatrix4x4()
vtk.vtkMatrix4x4.Invert(planeToWorld, worldToPlane)

# Project each point
for pointIndex in range(2):
    point_World =[*lineNode.GetNthControlPointPositionWorld(pointIndex), 1.0]
    point_Plane = worldToPlane.MultiplyPoint(point_World)
    projectedPoint_Plane =[point_Plane[0], point_Plane[1], 0.0, 1.0]
    projectedPoint_World = planeToWorld.MultiplyPoint(projectedPoint_Plane)
    projectedLineNode.AddControlPoint(projectedPoint_World[0:3])
```

8. Script to pre-populate the angles with 2 landmarks (point defining the initial arm and vertex of angle)

```python
F=getNode('soft tissue')
H=getNode('EA R intersection')
A=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsAngleNode')
firstPoint = F.GetNthControlPointPositionVector(5)
A.AddControlPoint(firstPoint)
secondPoint = H.GetNthControlPointPositionVector(0)
A.AddControlPoint(secondPoint)
A.SetName('ear angle R')

F=getNode('soft tissue')
H=getNode('EA L intersection')
A=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsAngleNode')
firstPoint = F.GetNthControlPointPositionVector(7)
A.AddControlPoint(firstPoint)
secondPoint = H.GetNthControlPointPositionVector(0)
A.AddControlPoint(secondPoint)
A.SetName('ear angle L')

F=getNode('soft tissue')
H=getNode('EIA R intersection')
A=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsAngleNode')
firstPoint = F.GetNthControlPointPositionVector(3)
A.AddControlPoint(firstPoint)
secondPoint = H.GetNthControlPointPositionVector(0)
A.AddControlPoint(secondPoint)
A.SetName('ear insertion angle R')

F=getNode('soft tissue')
H=getNode('EIA L intersection')
A=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsAngleNode')
firstPoint = F.GetNthControlPointPositionVector(1)
A.AddControlPoint(firstPoint)
secondPoint = H.GetNthControlPointPositionVector(0)
A.AddControlPoint(secondPoint)
A.SetName('ear insertion angle L')


F=getNode('soft tissue')
A=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsAngleNode')
firstPoint = F.GetNthControlPointPositionVector(12)
A.AddControlPoint(firstPoint)
secondPoint = F.GetNthControlPointPositionVector(2)
A.AddControlPoint(secondPoint)
A.SetName('ear protrusion R')


F=getNode('soft tissue')
A=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsAngleNode')
firstPoint = F.GetNthControlPointPositionVector(13)
A.AddControlPoint(firstPoint)
secondPoint = F.GetNthControlPointPositionVector(0)
A.AddControlPoint(secondPoint)
A.SetName('ear protrusion L')

G=getNode('hard tissue')
H=getNode('MAA R intersection')
A=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsAngleNode')
firstPoint = G.GetNthControlPointPositionVector(7)
A.AddControlPoint(firstPoint)
secondPoint = H.GetNthControlPointPositionVector(0)
A.AddControlPoint(secondPoint)
A.SetName('mastoid anterior angle R')


G=getNode('hard tissue')
H=getNode('MAA L intersection')
A=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsAngleNode')
firstPoint = G.GetNthControlPointPositionVector(6)
A.AddControlPoint(firstPoint)
secondPoint = H.GetNthControlPointPositionVector(0)
A.AddControlPoint(secondPoint)
A.SetName('mastoid anterior angle L')


G=getNode('hard tissue')
H=getNode('MLA R intersection')
A=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsAngleNode')
firstPoint = G.GetNthControlPointPositionVector(7)
A.AddControlPoint(firstPoint)
secondPoint = H.GetNthControlPointPositionVector(0)
A.AddControlPoint(secondPoint)
A.SetName('mastoid lateral angle R')


G=getNode('hard tissue')
H=getNode('MLA L intersection')
A=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsAngleNode')
firstPoint = G.GetNthControlPointPositionVector(6)
A.AddControlPoint(firstPoint)
secondPoint = H.GetNthControlPointPositionVector(0)
A.AddControlPoint(secondPoint)
A.SetName('mastoid lateral angle L')

G=getNode('hard tissue')
H=getNode('MRA R intersection')
A=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsAngleNode')
firstPoint = G.GetNthControlPointPositionVector(9)
A.AddControlPoint(firstPoint)
secondPoint = H.GetNthControlPointPositionVector(0)
A.AddControlPoint(secondPoint)
A.SetName('mandible ramus angle R')

G=getNode('hard tissue')
H=getNode('MRA L intersection')
A=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsAngleNode')
firstPoint = G.GetNthControlPointPositionVector(10)
A.AddControlPoint(firstPoint)
secondPoint = H.GetNthControlPointPositionVector(0)
A.AddControlPoint(secondPoint)
A.SetName('mandible ramus angle L')

F=getNode('soft tissue')
H=getNode('NRA intersection')
A=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsAngleNode')
firstPoint = F.GetNthControlPointPositionVector(15)
A.AddControlPoint(firstPoint)
secondPoint = H.GetNthControlPointPositionVector(0)
A.AddControlPoint(secondPoint)
A.SetName('nasal root angle')


I=getNode('for hard nose angle')
H=getNode('HNA intersection')
A=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsAngleNode')
firstPoint = I.GetNthControlPointPositionVector(1)
A.AddControlPoint(firstPoint)
secondPoint = H.GetNthControlPointPositionVector(0)
A.AddControlPoint(secondPoint)
A.SetName('hard nose angle')
```

9. Script to copy angle measurements to clipboard via the shortcut Ctrl+T

```python

def copyAngleMeasurementsToClipboard():
  measurements =[]
  # Collect all line measurements from the scene
  angleNodes = getNodesByClass('vtkMRMLMarkupsAngleNode')
  for angleNode in angleNodes:
    if angleNode.GetNumberOfDefinedControlPoints() < 3:
      # incomplete line, skip it
      continue
    # Get node filename that the length was measured on
    try:
      volumeNode = slicer.mrmlScene.GetNodeByID(angleNode.GetNthControlPointAssociatedNodeID(0))
      imagePath = volumeNode.GetStorageNode().GetFileName()
    except:
      imagePath = '(unknown)'
    # Get angle node n
    measurementName = angleNode.GetName()
    # Get angle measurement
    angleNode.GetMeasurement('angle').SetEnabled(True)
    angle = str(angleNode.GetMeasurement('angle').GetValue())
    # Add fields to results
    measurements.append('\t'.join([imagePath, measurementName, angle]))
  # Copy all measurements to clipboard (to be pasted into Excel)
  outputText = "\n".join(measurements) + "\n"
  slicer.app.clipboard().setText(outputText)
  slicer.util.delayDisplay(f"Copied {len(measurements)} angle measurements to the clipboard.")

shortcut2 = qt.QShortcut(slicer.util.mainWindow())
shortcut2.setKey(qt.QKeySequence("Ctrl+t"))
shortcut2.connect( 'activated()', copyAngleMeasurementsToClipboard)

```
 
