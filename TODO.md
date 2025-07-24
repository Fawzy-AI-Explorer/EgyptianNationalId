# 📋 TODO - Egyptian ID Card Data Extraction Project

## Phase 1                                 cropped image  => (OCR) => Text      
 
- [x] Install required libraries:
  - [x] 
  - [x] 
  - [x] 

- [x] Connect to the camera using OpenCV
- [x] Capture video frames in a continuous loop
  - [x] Add exit condition (`q` key)
  - [x] Add capture condition (`spacebar` to save frame)

- [x] Define cropping coordinates (fixed area for the ID card)
- [x] Crop the frame to isolate the card region
- [x] Display:
  - [x] Original camera feed with guide rectangle
  - [x] Cropped region
  - [x] Preprocessed version of cropped region

- [x] Apply preprocessing:
  - [x] Convert to grayscale
  - [x] Apply Gaussian blur
  - [x] Apply adaptive threshold

---

## 🧠 Phase 2: Field Detection & OCR

- [ ] Analyze fixed layout of Egyptian ID card
  - [ ] Measure ROIs for:
    - [ ] Full Name
    - [ ] National ID number
    - [ ] Date of Birth
    - [ ] Address

- [ ] Define Regions of Interest (ROIs) on the cropped image
- [ ] Crop each field individually using ROI coordinates
- [ ] Preprocess each field for better OCR (same as above or optimized per field)

- [ ] Apply OCR on each field using Tesseract
  - [ ] Use `lang='ara'` for Arabic fields
  - [ ] Use `lang='eng'` or `digits` for ID numbers

- [ ] Clean and validate OCR output:
  - [ ] Remove noise
  - [ ] Validate ID number length (14 digits)
  - [ ] Post-process text formatting

---

## 🧪 Phase 3: User Interface & Interaction

- [ ] Add instruction overlay or visual guide for user
- [ ] Create key hints (e.g., Press [Space] to capture, [Q] to quit)
- [ ] Add confirmation or re-capture option before saving
- [ ] Optional: Save OCR results to file (JSON, CSV, etc.)

---

## 🚀 Stretch Goals

- [ ] Add face alignment check (face inside card vs live image)
- [ ] Use YOLO or template matching to detect card automatically (instead of fixed crop)
- [ ] Web interface using Streamlit or Flask
- [ ] Support multi-card batch processing

