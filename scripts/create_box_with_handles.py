from pxr import Usd, UsdGeom, Gf, UsdPhysics
import os

def create_usd():
    usd_path = r"c:\Users\alexa\projects\humanoid_training\artifacts\box_with_handles.usd"
    os.makedirs(os.path.dirname(usd_path), exist_ok=True)
    if os.path.exists(usd_path):
        os.remove(usd_path)
        
    stage = Usd.Stage.CreateNew(usd_path)
    
    # Create root Xform
    root_xform = UsdGeom.Xform.Define(stage, "/box_with_handles")
    stage.SetDefaultPrim(root_xform.GetPrim())
    
    # Apply UsdPhysics.RigidBodyAPI to the root Xform to make it a dynamic rigid body
    UsdPhysics.RigidBodyAPI.Apply(root_xform.GetPrim())
    
    # Create main box
    # Box dimensions: (x=0.22, y=0.32, z=0.18)
    # Cube default size is 2, so scale is half of the target dimension
    box = UsdGeom.Cube.Define(stage, "/box_with_handles/main_box")
    box.GetSizeAttr().Set(2.0)
    box.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    box.AddScaleOp().Set(Gf.Vec3d(0.11, 0.16, 0.09)) # size x=0.22, y=0.32, z=0.18
    
    # Apply UsdPhysics.CollisionAPI to enable collisions
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    
    # Create left handle (ledge)
    # The ledge is centered at y = 0.16 (box side) + 0.03 (ledge half-width) = 0.19
    # Size of ledge: x=0.12, y=0.06, z=0.02
    left_ledge = UsdGeom.Cube.Define(stage, "/box_with_handles/left_ledge")
    left_ledge.GetSizeAttr().Set(2.0)
    left_ledge.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.19, 0.0))
    left_ledge.AddScaleOp().Set(Gf.Vec3d(0.06, 0.03, 0.01))
    
    # Apply UsdPhysics.CollisionAPI
    UsdPhysics.CollisionAPI.Apply(left_ledge.GetPrim())
    
    # Create right handle (ledge)
    # Centered at y = -0.19
    right_ledge = UsdGeom.Cube.Define(stage, "/box_with_handles/right_ledge")
    right_ledge.GetSizeAttr().Set(2.0)
    right_ledge.AddTranslateOp().Set(Gf.Vec3d(0.0, -0.19, 0.0))
    right_ledge.AddScaleOp().Set(Gf.Vec3d(0.06, 0.03, 0.01))
    
    # Apply UsdPhysics.CollisionAPI
    UsdPhysics.CollisionAPI.Apply(right_ledge.GetPrim())
    
    stage.GetRootLayer().Save()
    print(f"USD asset with Physics APIs created successfully at {usd_path}")

if __name__ == "__main__":
    create_usd()
