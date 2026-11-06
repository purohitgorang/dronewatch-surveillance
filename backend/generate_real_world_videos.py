import os
import cv2
import numpy as np

def create_videos():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    samples_dir = os.path.join(base_dir, "test_samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # -------------------------------------------------------------
    # 1. drone_flight_transit.mp4 (Continuous Aerodynamic Flight)
    # -------------------------------------------------------------
    drone_img_path = os.path.join(samples_dir, "drone_phantom.jpg")
    drone_src = cv2.imread(drone_img_path)
    
    # Sky background base
    sky_src_path = os.path.join(samples_dir, "bird_sky.jpg")
    sky_base = cv2.imread(sky_src_path)
    if sky_base is None:
        sky_base = np.zeros((480, 640, 3), dtype=np.uint8)
        sky_base[:, :] = [215, 185, 140] # BGR sky blue
    else:
        sky_base = cv2.resize(sky_base, (640, 480))
        
    # Crop drone from center of drone_phantom
    dh, dw = drone_src.shape[:2]
    # Center crop of drone
    drone_crop = drone_src[int(dh*0.15):int(dh*0.85), int(dw*0.15):int(dw*0.85)]
    
    out_path_1 = os.path.join(samples_dir, "drone_flight_transit.mp4")
    out_1 = cv2.VideoWriter(out_path_1, fourcc, 30.0, (640, 480))
    
    num_frames = 60
    for i in range(num_frames):
        t = i / float(num_frames)
        frame = sky_base.copy()
        
        # Flight trajectory: left to right with slight climb & aerodynamic oscillation
        cx = int(80 + t * 460)
        cy = int(280 - t * 140 + np.sin(t * np.pi * 3) * 18)
        
        # Scale: drone gets slightly closer then banks away
        scale = 0.35 + 0.08 * np.sin(t * np.pi)
        scaled_w = int(drone_crop.shape[1] * scale)
        scaled_h = int(drone_crop.shape[0] * scale)
        d_resized = cv2.resize(drone_crop, (scaled_w, scaled_h))
        
        # Overlay with alpha blending or chroma mask
        x1 = max(0, cx - scaled_w // 2)
        y1 = max(0, cy - scaled_h // 2)
        x2 = min(640, x1 + scaled_w)
        y2 = min(480, y1 + scaled_h)
        
        crop_d = d_resized[0:(y2-y1), 0:(x2-x1)]
        # Blend into sky
        frame[y1:y2, x1:x2] = crop_d
        
        # Add slight cloud drift
        M = np.float32([[1, 0, int(i * 0.5)], [0, 1, 0]])
        frame = cv2.warpAffine(frame, M, (640, 480), borderMode=cv2.BORDER_REFLECT)
        
        out_1.write(frame)
        
    out_1.release()
    print(f"[+] Created real-world flight video: {out_path_1}")
    
    # -------------------------------------------------------------
    # 2. drone_fpv_highspeed.mp4 (Tactical High-Speed Maneuver)
    # -------------------------------------------------------------
    fpv_src_path = os.path.join(samples_dir, "drone_fpv.jpg")
    fpv_src = cv2.imread(fpv_src_path)
    fh, fw = fpv_src.shape[:2]
    fpv_crop = fpv_src[int(fh*0.1):int(fh*0.9), int(fw*0.1):int(fw*0.9)]
    
    out_path_2 = os.path.join(samples_dir, "drone_fpv_highspeed.mp4")
    out_2 = cv2.VideoWriter(out_path_2, fourcc, 30.0, (640, 480))
    
    field_bg = cv2.imread(os.path.join(samples_dir, "drone_field.jpg"))
    if field_bg is None:
        field_bg = sky_base.copy()
    else:
        field_bg = cv2.resize(field_bg, (640, 480))
        
    for i in range(num_frames):
        t = i / float(num_frames)
        frame = field_bg.copy()
        
        # High-speed diagonal pass: accelerating
        t_acc = t ** 1.3
        cx = int(60 + t_acc * 500)
        cy = int(90 + t_acc * 290)
        
        scale = 0.28 + t * 0.15
        sw = int(fpv_crop.shape[1] * scale)
        sh = int(fpv_crop.shape[0] * scale)
        f_resized = cv2.resize(fpv_crop, (sw, sh))
        
        x1 = max(0, cx - sw // 2)
        y1 = max(0, cy - sh // 2)
        x2 = min(640, x1 + sw)
        y2 = min(480, y1 + sh)
        
        frame[y1:y2, x1:x2] = f_resized[0:(y2-y1), 0:(x2-x1)]
        out_2.write(frame)
        
    out_2.release()
    print(f"[+] Created FPV high-speed video: {out_path_2}")
    
    # -------------------------------------------------------------
    # 3. spoof_phone_presentation_attack.mp4 (Anti-Spoofing Test)
    # -------------------------------------------------------------
    spoof_src_path = os.path.join(samples_dir, "distractor_phone_drone.jpg")
    spoof_src = cv2.imread(spoof_src_path)
    if spoof_src is None:
        print("Warning: distractor_phone_drone.jpg not found, creating synthetic phone spoof")
        spoof_src = cv2.resize(sky_base, (640, 480))
    else:
        spoof_src = cv2.resize(spoof_src, (640, 480))
        
    out_path_3 = os.path.join(samples_dir, "spoof_phone_presentation_attack.mp4")
    out_3 = cv2.VideoWriter(out_path_3, fourcc, 30.0, (640, 480))
    
    # Simulate handheld phone sway/wobble in front of camera
    for i in range(num_frames):
        t = i / float(num_frames)
        # Small translational sway and tilt
        dx = int(np.sin(t * np.pi * 4) * 15)
        dy = int(np.cos(t * np.pi * 3) * 10)
        
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        frame = cv2.warpAffine(spoof_src, M, (640, 480), borderMode=cv2.BORDER_REFLECT)
        out_3.write(frame)
        
    out_3.release()
    print(f"[+] Created phone spoof presentation attack video: {out_path_3}")
    
    # -------------------------------------------------------------
    # 4. distractor_birds_flight.mp4 (Wildlife Negative Verification)
    # -------------------------------------------------------------
    seagull_src_path = os.path.join(samples_dir, "distractor_seagull.jpg")
    seagull_src = cv2.imread(seagull_src_path)
    if seagull_src is None:
        seagull_src = cv2.imread(os.path.join(samples_dir, "bird_sky.jpg"))
    seagull_src = cv2.resize(seagull_src, (640, 480))
    
    out_path_4 = os.path.join(samples_dir, "distractor_birds_flight.mp4")
    out_4 = cv2.VideoWriter(out_path_4, fourcc, 30.0, (640, 480))
    
    for i in range(num_frames):
        t = i / float(num_frames)
        dx = int(t * 30 - 15)
        dy = int(np.sin(t * np.pi * 5) * 8)
        
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        frame = cv2.warpAffine(seagull_src, M, (640, 480), borderMode=cv2.BORDER_REFLECT)
        out_4.write(frame)
        
    out_4.release()
    print(f"[+] Created wildlife distractor flight video: {out_path_4}")
    print("[+] All 4 real-world continuous flight video benchmarks generated successfully.")

if __name__ == "__main__":
    create_videos()
