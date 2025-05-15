import open3d as o3d
import numpy as np
import time
import argparse

def visualize_hawor_npz(npz_filepath):
    """
    Visualizes hand motion from an .npz file saved by the modified demo.py script.

    Args:
        npz_filepath (str): Path to the .npz file.
    """
    try:
        data = np.load(npz_filepath, allow_pickle=True)
        print(f"Successfully loaded data from: {npz_filepath}")
        print("Available keys in the .npz file:", list(data.keys()))
    except FileNotFoundError:
        print(f"Error: File not found at {npz_filepath}")
        return
    except Exception as e:
        print(f"Error loading .npz file: {e}")
        return

    # --- Get data from .npz file ---
    left_verts_seq = data.get('left_hand_vertices')
    left_faces = data.get('left_hand_faces')
    right_verts_seq = data.get('right_hand_vertices')
    right_faces = data.get('right_hand_faces')
    rx_matrix = data.get('world_transformation_Rx') # This is the R_x used in demo.py

    if rx_matrix is None:
        print("Error: 'world_transformation_Rx' not found in .npz file. Cannot reorient vertices.")
        # Default to identity if not found, so no reorientation, might look flipped
        rx_matrix = np.eye(3)


    # --- Prepare Meshes ---
    left_hand_mesh = None
    right_hand_mesh = None
    num_frames = 0

    if left_verts_seq is not None and left_faces is not None:
        num_frames = left_verts_seq.shape[0]
        print(f"Found left hand data with {num_frames} frames.")
        # Reorient vertices by applying Rx_matrix again (v_orig = Rx @ v_saved)
        # Vertices are (T, N, 3). Rx_matrix is (3,3).
        # We need to transform each (N,3) vertex set for each frame.
        # v_frame_reoriented = v_frame_saved @ rx_matrix.T
        reoriented_left_verts_seq = np.array([frame_verts @ rx_matrix.T for frame_verts in left_verts_seq])

        left_hand_mesh = o3d.geometry.TriangleMesh()
        left_hand_mesh.vertices = o3d.utility.Vector3dVector(reoriented_left_verts_seq[0])
        left_hand_mesh.triangles = o3d.utility.Vector3iVector(left_faces)
        left_hand_mesh.paint_uniform_color([0.7, 0.3, 0.3]) # Reddish color for left hand
        left_hand_mesh.compute_vertex_normals()
    else:
        print("Left hand data not found or incomplete in the .npz file.")

    if right_verts_seq is not None and right_faces is not None:
        if num_frames == 0: # If left hand was missing, get num_frames from right hand
            num_frames = right_verts_seq.shape[0]
        print(f"Found right hand data with {num_frames} frames.")
        reoriented_right_verts_seq = np.array([frame_verts @ rx_matrix.T for frame_verts in right_verts_seq])

        right_hand_mesh = o3d.geometry.TriangleMesh()
        right_hand_mesh.vertices = o3d.utility.Vector3dVector(reoriented_right_verts_seq[0])
        right_hand_mesh.triangles = o3d.utility.Vector3iVector(right_faces)
        right_hand_mesh.paint_uniform_color([0.3, 0.7, 0.3]) # Greenish color for right hand
        right_hand_mesh.compute_vertex_normals()
    else:
        print("Right hand data not found or incomplete in the .npz file.")

    if num_frames == 0:
        print("No valid hand data found to visualize.")
        return

    # --- Setup Open3D Visualizer ---
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="HaWoR NPZ Visualization", width=1280, height=720)

    if left_hand_mesh:
        vis.add_geometry(left_hand_mesh)
    if right_hand_mesh:
        vis.add_geometry(right_hand_mesh)

    # Add a coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)

    # Adjust view control
    view_control = vis.get_view_control()
    view_control.set_zoom(0.8)
    view_control.set_lookat(np.array([0.0, 0.0, 0.0])) # Look at origin
    view_control.set_up(np.array([0.0, -1.0, 0.0]))    # Set Y-down as "up" if original MANO was Y-up
                                                     # After reorientation, data should be Y-up, so Open3D's default Y-up should be fine.
                                                     # Let's use Open3D default up: [0,1,0]
    view_control.set_up(np.array([0.0, 1.0, 0.0]))
    view_control.set_front(np.array([0.0, 0.0, -1.0])) # Look along negative Z

    print("\nStarting animation. Press 'ESC' or close the window to exit.")
    print("Interactive controls: Mouse drag to rotate, scroll to zoom.")

    # --- Animation Loop ---
    current_frame = 0
    last_update_time = time.time()
    fps = 30  # Target FPS
    frame_duration = 1.0 / fps

    try:
        while True:
            if current_frame >= num_frames:
                current_frame = 0 # Loop animation

            if left_hand_mesh:
                left_hand_mesh.vertices = o3d.utility.Vector3dVector(reoriented_left_verts_seq[current_frame])
                left_hand_mesh.compute_vertex_normals()
                vis.update_geometry(left_hand_mesh)

            if right_hand_mesh:
                right_hand_mesh.vertices = o3d.utility.Vector3dVector(reoriented_right_verts_seq[current_frame])
                right_hand_mesh.compute_vertex_normals()
                vis.update_geometry(right_hand_mesh)

            if not vis.poll_events(): # poll_events returns True if window is open
                break
            vis.update_renderer()

            current_frame += 1
            
            # Control FPS
            elapsed_time = time.time() - last_update_time
            sleep_time = frame_duration - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_update_time = time.time()

    except KeyboardInterrupt:
        print("Animation stopped by user.")
    finally:
        vis.destroy_window()
        print("Visualization window closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize HaWoR output from an .npz file using Open3D.")
    parser.add_argument("npz_file", type=str, help="Path to the .npz file containing the HaWoR output.")
    args = parser.parse_args()

    visualize_hawor_npz(args.npz_file)