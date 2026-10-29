# Scene materials

These assets are generated locally by ../scripts/build_scenes.py with a fixed random seed.
No external textures, downloaded models or third-party licensed art are required.

Each material has a 512 x 512 albedo, tangent-space normal and roughness PNG.
Materials: wood, brick, tiles, concrete, asphalt, metal, plaster, fabric and three
painted delivery-marker colors. Gazebo uses primitive collision and visual geometry
with PBR materials; the occupancy maps are generated from the same obstacle layout.

These are procedural training scenes, not photorealistic reconstructions.
SDF files use the model://scene_materials/textures namespace, resolved through GZ_SIM_RESOURCE_PATH.
Run prepare_assets.sh after a fresh checkout to validate all generated assets.
